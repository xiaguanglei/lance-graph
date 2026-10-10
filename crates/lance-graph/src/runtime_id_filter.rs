// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Opt-in completed JOIN ID membership pushdown into unchanged Lance scanners.
//! All business joins remain in place; this only narrows an indexed node scan.
use arrow_schema::{DataType, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    catalog::Session,
    common::{config::ConfigOptions, Result, ScalarValue, Statistics},
    datasource::TableProvider,
    execution::{SessionState, TaskContext},
    logical_expr::{col, lit, Expr, Operator, TableProviderFilterPushDown, TableType},
    physical_expr::{
        expressions::{BinaryExpr, Column, DynamicFilterPhysicalExpr, InListExpr, Literal},
        PhysicalExpr,
    },
    physical_plan::{
        filter_pushdown::{
            ChildPushdownResult, FilterPushdownPhase, FilterPushdownPropagation, PushedDown,
        },
        stream::RecordBatchStreamAdapter,
        DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
        SendableRecordBatchStream,
    },
};
use futures::{stream, TryStreamExt};
use lance::{datafusion::LanceTableProvider, index::DatasetIndexExt};
use lance_index::{scalar::expression::ScalarIndexExpr, IndexType};
use std::{
    collections::BTreeSet,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

/// Bounded counters for one query or a caller-chosen metrics lifetime.
#[derive(Debug, Default)]
pub struct RuntimeIdFilterMetrics {
    pub index_scans: AtomicU64,
    pub filtered_keys: AtomicU64,
    pub output_rows: AtomicU64,
    pub timeout_fallbacks: AtomicU64,
    pub unsupported_fallbacks: AtomicU64,
    pub planning_fallbacks: AtomicU64,
    pub skipped_no_index: AtomicU64,
    pub skipped_static_id: AtomicU64,
    pub rejected_producer_filters: AtomicU64,
}
#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
    use arrow_schema::{Field, Schema};
    use datafusion::{execution::context::SessionContext, physical_plan::collect};
    use lance::{dataset::WriteParams, Dataset};
    use lance_index::{scalar::ScalarIndexParams, IndexType};

    #[tokio::test]
    async fn provenance_guard_preserves_null_aware_anti_red_then_green() {
        use datafusion::{
            logical_expr::JoinType,
            physical_plan::joins::{HashJoinExecBuilder, PartitionMode},
        };
        let temp = tempfile::tempdir().unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "entity_id",
            DataType::Utf8,
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec![Some("b"), None]))],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema.clone()),
            temp.path().join("thing.lance").to_str().unwrap(),
            None::<WriteParams>,
        )
        .await
        .unwrap();
        dataset
            .create_index_builder(
                &["entity_id"],
                IndexType::BTree,
                &ScalarIndexParams::default(),
            )
            .name("id_index".to_string())
            .await
            .unwrap();
        let ctx = SessionContext::new();
        let build = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec![Some("a")]))],
        )
        .unwrap();
        ctx.register_batch("build", build).unwrap();
        let provider = RuntimeProvider::new(
            Arc::new(LanceTableProvider::new(Arc::new(dataset), false, false)),
            "entity_id",
            Arc::new(RuntimeIdFilterMetrics::default()),
        );
        let make = |unsafe_previous: bool| {
            let ctx = ctx.clone();
            let provider = &provider;
            async move {
                let original = provider.scan(&ctx.state(), None, &[], None).await.unwrap();
                let mut source = original.downcast_ref::<RuntimeExec>().unwrap().clone();
                let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
                    vec![Arc::new(Column::new("entity_id", 0))],
                    Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))),
                ));
                source.runtime_filters = vec![dynamic.clone()];
                source.validated_producer = unsafe_previous;
                let left = ctx
                    .table("build")
                    .await
                    .unwrap()
                    .create_physical_plan()
                    .await
                    .unwrap();
                let on = vec![(
                    Arc::new(Column::new("entity_id", 0)) as Arc<dyn PhysicalExpr>,
                    Arc::new(Column::new("entity_id", 0)) as Arc<dyn PhysicalExpr>,
                )];
                Arc::new(
                    HashJoinExecBuilder::new(left, Arc::new(source), on, JoinType::LeftAnti)
                        .with_partition_mode(PartitionMode::CollectLeft)
                        .with_null_aware(true)
                        .build()
                        .unwrap()
                        .with_dynamic_filter_expr(dynamic)
                        .unwrap(),
                ) as Arc<dyn ExecutionPlan>
            }
        };
        // RED: the former source-only consumer drops probe NULL and emits "a".
        let unsafe_rows = collect(make(true).await, ctx.task_ctx())
            .await
            .unwrap()
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>();
        assert_eq!(
            unsafe_rows, 1,
            "counterexample must exercise the prior unsound consumer"
        );
        // GREEN: actual HashJoin provenance rejects anti, leaving NULL intact.
        let guarded = bind_runtime_id_sources(make(false).await).unwrap();
        let rows = collect(guarded, ctx.task_ctx())
            .await
            .unwrap()
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>();
        assert_eq!(rows, 0);
        assert!(
            provider
                .metrics
                .rejected_producer_filters
                .load(Ordering::Relaxed)
                > 0
        );
    }

    fn in_list(keys: Vec<Option<&str>>) -> Arc<dyn PhysicalExpr> {
        let schema = Schema::new(vec![Field::new("entity_id", DataType::Utf8, true)]);
        Arc::new(
            InListExpr::try_new_from_array(
                Arc::new(Column::new("entity_id", 0)),
                Arc::new(StringArray::from(keys)),
                false,
                &schema,
            )
            .unwrap(),
        )
    }
    #[test]
    fn bounded_exact_membership_alias_null_and_large_fallback() {
        assert_eq!(
            membership(
                &in_list(vec![Some("a"), Some("a"), Some("b")]),
                "entity_id",
                0
            )
            .unwrap()
            .1,
            2
        );
        assert!(membership(&in_list(vec![Some("a"), None]), "entity_id", 0).is_none());
        assert!(membership(&in_list(vec![Some("a")]), "wrong_alias", 0).is_none());
        assert!(membership(&in_list(vec![Some("a")]), "entity_id", 1).is_none());
        for n in [149, 150, 151] {
            let keys = (0..n).map(|i| format!("id{i}")).collect::<Vec<_>>();
            assert_eq!(
                membership(
                    &in_list(keys.iter().map(|s| Some(s.as_str())).collect()),
                    "entity_id",
                    0
                )
                .is_some(),
                n <= 150
            );
        }
        for bytes in [128 * 1024, 128 * 1024 + 1] {
            let long = "记".repeat(bytes / 3) + &"x".repeat(bytes % 3);
            assert_eq!(
                membership(&in_list(vec![Some(&long)]), "entity_id", 0).is_some(),
                bytes <= 128 * 1024
            );
        }
        let and = Arc::new(BinaryExpr::new(
            in_list(vec![Some("a")]),
            Operator::And,
            Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))),
        )) as Arc<dyn PhysicalExpr>;
        assert!(membership(&and, "entity_id", 0).is_some());
        let or = Arc::new(BinaryExpr::new(
            in_list(vec![Some("a")]),
            Operator::Or,
            Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))),
        )) as Arc<dyn PhysicalExpr>;
        assert!(membership(&or, "entity_id", 0).is_none());
    }
    #[tokio::test]
    async fn unresolved_producer_times_out_to_full_scan_and_reset_is_safe() {
        let temp = tempfile::tempdir().unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "entity_id",
            DataType::Utf8,
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec![
                Some("a"),
                Some("b"),
                None,
            ]))],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            temp.path().join("thing.lance").to_str().unwrap(),
            None::<WriteParams>,
        )
        .await
        .unwrap();
        dataset
            .create_index_builder(
                &["entity_id"],
                IndexType::BTree,
                &ScalarIndexParams::default(),
            )
            .name("id_index".to_string())
            .await
            .unwrap();
        let provider = RuntimeProvider::new(
            Arc::new(LanceTableProvider::new(Arc::new(dataset), false, false)),
            "entity_id",
            Arc::new(RuntimeIdFilterMetrics::default()),
        );
        let ctx = SessionContext::new();
        let plan = provider.scan(&ctx.state(), None, &[], None).await.unwrap();
        let mut source = plan.downcast_ref::<RuntimeExec>().unwrap().clone();
        source
            .runtime_filters
            .push(Arc::new(DynamicFilterPhysicalExpr::new(
                vec![Arc::new(Column::new("entity_id", 0))],
                Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))),
            )));
        source.wait = Duration::from_millis(1);
        source.validated_producer = true; // Synthetic completed-producer contract test.
        let before = std::time::Instant::now();
        let repeated: Arc<dyn ExecutionPlan> = Arc::new(source.clone());
        for _ in 0..2 {
            let result = tokio::time::timeout(
                Duration::from_secs(2),
                collect(repeated.clone(), ctx.task_ctx()),
            )
            .await
            .unwrap()
            .unwrap();
            assert_eq!(result.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
        }
        assert!(before.elapsed() < Duration::from_secs(2));
        assert_eq!(source.metrics.timeout_fallbacks.load(Ordering::Relaxed), 2);
        let reset = Arc::new(source.clone()).reset_state().unwrap();
        assert!(reset
            .downcast_ref::<RuntimeExec>()
            .unwrap()
            .runtime_filters
            .is_empty());
        assert_eq!(
            collect(reset, ctx.task_ctx())
                .await
                .unwrap()
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>(),
            3
        );
        // A partial producer is not safe: it must finish before an ID set is
        // used. Completion adds "b", so both non-NULL original rows survive.
        let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::new(Column::new("entity_id", 0))],
            in_list(vec![Some("a")]),
        ));
        source.runtime_filters = vec![dynamic.clone()];
        source.wait = Duration::from_millis(100);
        let producer = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(5)).await;
            dynamic.update(in_list(vec![Some("a"), Some("b")])).unwrap();
            dynamic.mark_complete();
        });
        let completed: Arc<dyn ExecutionPlan> = Arc::new(source.clone());
        for _ in 0..2 {
            assert_eq!(
                collect(completed.clone(), ctx.task_ctx())
                    .await
                    .unwrap()
                    .iter()
                    .map(RecordBatch::num_rows)
                    .sum::<usize>(),
                2
            );
        }
        producer.await.unwrap();
        // Request cancellation while waiting neither leaks a stream nor
        // prevents the same immutable source from subsequently falling back.
        source.runtime_filters = vec![Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::new(Column::new("entity_id", 0))],
            in_list(vec![Some("a")]),
        ))];
        assert!(tokio::time::timeout(
            Duration::from_millis(1),
            collect(Arc::new(source.clone()), ctx.task_ctx())
        )
        .await
        .is_err());
        source.wait = Duration::from_millis(1);
        assert_eq!(
            collect(Arc::new(source), ctx.task_ctx())
                .await
                .unwrap()
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>(),
            3
        );
        let mut multiple = plan.downcast_ref::<RuntimeExec>().unwrap().clone();
        multiple.validated_producer = true;
        multiple.runtime_filters = (0..2)
            .map(|_| {
                Arc::new(DynamicFilterPhysicalExpr::new(
                    vec![Arc::new(Column::new("entity_id", 0))],
                    in_list(vec![Some("a")]),
                ))
            })
            .collect();
        let start = std::time::Instant::now();
        assert_eq!(
            collect(Arc::new(multiple.clone()), ctx.task_ctx())
                .await
                .unwrap()
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>(),
            3
        );
        assert!(
            start.elapsed() < Duration::from_millis(175),
            "two producers share one 100ms deadline"
        );
        let nulls = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::new(Column::new("entity_id", 0))],
            in_list(vec![Some("a"), None]),
        ));
        nulls.mark_complete();
        multiple.runtime_filters = vec![nulls];
        assert_eq!(
            collect(Arc::new(multiple), ctx.task_ctx())
                .await
                .unwrap()
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>(),
            3
        );
        // A static ID predicate or a fetch keeps the original provider plan.
        assert!(provider
            .scan(&ctx.state(), None, &[col("entity_id").eq(lit("a"))], None)
            .await
            .unwrap()
            .downcast_ref::<RuntimeExec>()
            .is_none());
        assert!(provider
            .scan(&ctx.state(), None, &[], Some(1))
            .await
            .unwrap()
            .downcast_ref::<RuntimeExec>()
            .is_none());
    }
}

#[derive(Debug)]
pub(crate) struct RuntimeProvider {
    inner: Arc<LanceTableProvider>,
    id: String,
    metrics: Arc<RuntimeIdFilterMetrics>,
}

impl RuntimeProvider {
    pub fn new(
        inner: Arc<LanceTableProvider>,
        id: &str,
        metrics: Arc<RuntimeIdFilterMetrics>,
    ) -> Self {
        Self {
            inner,
            id: id.into(),
            metrics,
        }
    }
}

#[async_trait]
impl TableProvider for RuntimeProvider {
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }
    fn table_type(&self) -> TableType {
        self.inner.table_type()
    }
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        self.inner.supports_filters_pushdown(filters)
    }
    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let fallback = self.inner.scan(state, projection, filters, limit).await?;
        // Limited scans must remain unchanged: filtering before an existing scan
        // fetch could change the subset the original join receives.
        let Some(session) = state.as_any().downcast_ref::<SessionState>() else {
            return Ok(fallback);
        };
        let static_id = filters.iter().any(|filter| {
            filter
                .column_refs()
                .iter()
                .any(|column| column.name == self.id)
        });
        if static_id {
            self.metrics
                .skipped_static_id
                .fetch_add(1, Ordering::Relaxed);
        }
        if limit.is_some()
            || fallback.output_partitioning().partition_count() != 1
            || fallback.output_ordering().is_some()
            || static_id
        {
            return Ok(fallback);
        }
        let Ok(id_index) = fallback.schema().index_of(&self.id) else {
            return Ok(fallback);
        };
        if fallback.schema().field(id_index).data_type() != &DataType::Utf8 {
            return Ok(fallback);
        }
        // Admission is snapshot-local and metadata-only. Legacy/unknown index
        // metadata or an index-list error conservatively keeps the original
        // scan. Lance remains responsible for coverage, append and deletions.
        let dataset = self.inner.dataset();
        let Some(field) = dataset.schema().field(&self.id) else {
            return Ok(fallback);
        };
        let indexed = dataset.load_indices().await.ok().is_some_and(|indices| {
            indices.iter().any(|index| {
                index.fields.first() == Some(&field.id)
                    && index
                        .index_details
                        .as_ref()
                        .is_some_and(|details| IndexType::BTree.matches_details(details))
            })
        });
        if !indexed {
            self.metrics
                .skipped_no_index
                .fetch_add(1, Ordering::Relaxed);
            return Ok(fallback);
        }
        Ok(Arc::new(RuntimeExec {
            properties: fallback.properties().clone(),
            fallback,
            inner: self.inner.clone(),
            id: self.id.clone(),
            id_index,
            session: session.clone(),
            projection: projection.cloned(),
            filters: filters.to_vec(),
            runtime_filters: Vec::new(),
            validated_producer: false,
            metrics: self.metrics.clone(),
            wait: Duration::from_millis(100),
        }))
    }
}

#[derive(Debug, Clone)]
struct RuntimeExec {
    properties: Arc<PlanProperties>,
    fallback: Arc<dyn ExecutionPlan>,
    inner: Arc<LanceTableProvider>,
    id: String,
    id_index: usize,
    session: SessionState,
    projection: Option<Vec<usize>>,
    filters: Vec<Expr>,
    runtime_filters: Vec<Arc<DynamicFilterPhysicalExpr>>,
    validated_producer: bool,
    metrics: Arc<RuntimeIdFilterMetrics>,
    wait: Duration,
}

// The producer may add conservative min/max bounds using AND. A compatible
// membership conjunct alone is a safe superset: the original join is retained.
// OR / CASE / map membership / casts and NULL literals are never converted.
fn membership(expr: &Arc<dyn PhysicalExpr>, id: &str, index: usize) -> Option<(Expr, usize)> {
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        if binary.op() == &Operator::And {
            return membership(binary.left(), id, index)
                .or_else(|| membership(binary.right(), id, index));
        }
        return None;
    }
    let list = expr.downcast_ref::<InListExpr>()?;
    let column = list.expr().downcast_ref::<Column>()?;
    if list.negated() || column.name() != id || column.index() != index || list.len() > 150 {
        return None;
    }
    let mut bytes = 0usize;
    let values = list
        .list()
        .iter()
        .map(|value| {
            let literal = value.downcast_ref::<Literal>()?;
            let ScalarValue::Utf8(Some(text)) = literal.value() else {
                return None;
            };
            bytes = bytes.checked_add(text.len())?;
            if bytes > 128 * 1024 {
                return None;
            }
            Some(text.clone())
        })
        .collect::<Option<BTreeSet<_>>>()?;
    let count = values.len();
    Some((
        col(id).in_list(values.into_iter().map(lit).collect(), false),
        count,
    ))
}

impl DisplayAs for RuntimeExec {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "RuntimeNodeIdScan: id={}, filters={}, version={}",
            self.id,
            self.runtime_filters.len(),
            self.inner.dataset().version().version
        )
    }
}

impl ExecutionPlan for RuntimeExec {
    fn name(&self) -> &'static str {
        "RuntimeNodeIdScan"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
    // This is a provider source, not a filter over an already read child. The
    // fallback plan stays private so DF reaches the source's physical hook.
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(datafusion::error::DataFusionError::Internal(
                "runtime ID source cannot have children".into(),
            ));
        }
        Ok(self)
    }
    fn reset_state(self: Arc<Self>) -> Result<Arc<dyn ExecutionPlan>> {
        let mut reset = self.as_ref().clone();
        reset.runtime_filters.clear();
        reset.validated_producer = false;
        // Execute always creates a fresh source, including after reset: some
        // Lance read plans contain one-shot streams.
        Ok(Arc::new(reset))
    }
    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
        // Conservative, unchanged statistics retain the baseline join choice.
        let statistics = self.fallback.partition_statistics(partition)?;
        if self.runtime_filters.is_empty() {
            Ok(statistics)
        } else {
            Ok(Arc::new(statistics.as_ref().clone().to_inexact()))
        }
    }
    fn handle_child_pushdown_result(
        &self,
        phase: FilterPushdownPhase,
        result: ChildPushdownResult,
        _: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
        let mut updated = self.clone();
        if phase == FilterPushdownPhase::Post {
            updated.validated_producer = false;
            for parent in &result.parent_filters {
                if let Some(dynamic) = parent.filter.downcast_ref::<DynamicFilterPhysicalExpr>() {
                    // Use the live, remapped child schema, never textual aliases.
                    let children = dynamic.children();
                    if children.len() == 1
                        && children[0]
                            .downcast_ref::<Column>()
                            .is_some_and(|c| c.name() == self.id && c.index() == self.id_index)
                    {
                        if let Ok(dynamic) =
                            Arc::downcast::<DynamicFilterPhysicalExpr>(parent.filter.clone())
                        {
                            if !updated
                                .runtime_filters
                                .iter()
                                .any(|existing| Arc::ptr_eq(existing, &dynamic))
                            {
                                updated.runtime_filters.push(dynamic);
                            }
                        }
                    }
                }
            }
        }
        // Report No: this is an optional join prefilter and may fall back. No
        // business predicate may be removed based on this optimization.
        Ok(FilterPushdownPropagation::with_parent_pushdown_result(vec![
            PushedDown::No;
            result.parent_filters.len()
        ])
        .with_updated_node(Arc::new(updated)))
    }
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(datafusion::error::DataFusionError::Internal(
                "runtime ID source supports only partition 0".into(),
            ));
        }
        let this = self.clone();
        let schema = self.schema();
        let future = async move {
            let mut filters = this.filters.clone();
            let mut keys = 0usize;
            let mut reason = "unsupported_expression";
            let mut applied = false;
            let deadline = tokio::time::Instant::now() + this.wait;
            for dynamic in this
                .runtime_filters
                .iter()
                .filter(|_| this.validated_producer)
            {
                if tokio::time::timeout_at(deadline, dynamic.wait_complete())
                    .await
                    .is_err()
                {
                    reason = "completion_timeout";
                    continue;
                }
                let Ok(current) = dynamic.current() else {
                    reason = "runtime_expression_error";
                    continue;
                };
                if let Some((predicate, count)) = membership(&current, &this.id, this.id_index) {
                    filters.push(predicate);
                    keys += count;
                    applied = true;
                }
            }
            // Never reuse the hidden fallback execution plan: Lance sources
            // may contain one-shot streams. Runtime expression/planning
            // failures are optional optimization failures; index execution
            // errors still propagate, avoiding unsafe replay after output.
            let candidate = this
                .inner
                .scan(&this.session, this.projection.as_ref(), &filters, None)
                .await;
            let valid = |plan: &Arc<dyn ExecutionPlan>| {
                plan.schema() == this.schema()
                    && plan.output_partitioning().partition_count() == 1
                    && plan.output_ordering().is_none()
            };
            let plan = match candidate {
                Ok(plan) if valid(&plan) => plan,
                result if applied => {
                    reason = if result.is_err() {
                        "runtime_planning_error"
                    } else {
                        "runtime_plan_contract_mismatch"
                    };
                    applied = false;
                    keys = 0;
                    // Preserve genuine original-source errors on fallback.
                    this.inner
                        .scan(&this.session, this.projection.as_ref(), &this.filters, None)
                        .await?
                }
                Err(error) => return Err(error),
                Ok(_) => {
                    return Err(datafusion::error::DataFusionError::Internal(
                        "fresh original scan changed its schema/partition/ordering contract".into(),
                    ));
                }
            };
            if !valid(&plan) {
                return Err(datafusion::error::DataFusionError::Internal(
                    "fresh original fallback changed its schema/partition/ordering contract".into(),
                ));
            }
            if applied {
                this.metrics
                    .filtered_keys
                    .fetch_add(keys as u64, Ordering::Relaxed);
                if id_btree_hit(&plan, &this.id) {
                    this.metrics.index_scans.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                let counter = match reason {
                    "completion_timeout" => &this.metrics.timeout_fallbacks,
                    "runtime_planning_error"
                    | "runtime_plan_contract_mismatch"
                    | "runtime_expression_error" => &this.metrics.planning_fallbacks,
                    _ => &this.metrics.unsupported_fallbacks,
                };
                counter.fetch_add(1, Ordering::Relaxed);
            }
            Ok(Box::pin(RecordBatchStreamAdapter::new(
                this.schema(),
                plan.execute(partition, context)?.inspect_ok(move |batch| {
                    if applied {
                        this.metrics
                            .output_rows
                            .fetch_add(batch.num_rows() as u64, Ordering::Relaxed);
                    }
                }),
            )) as SendableRecordBatchStream)
        };
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(future).try_flatten(),
        )))
    }
}

fn id_btree_hit(plan: &Arc<dyn ExecutionPlan>, id: &str) -> bool {
    if let Some(index) = plan.downcast_ref::<lance::io::exec::scalar_index::ScalarIndexExec>() {
        if has_id_btree(index.expr(), id) {
            return true;
        }
    }
    plan.children().iter().any(|child| id_btree_hit(child, id))
}

fn has_id_btree(expr: &ScalarIndexExpr, id: &str) -> bool {
    match expr {
        ScalarIndexExpr::Query(query) => query.column == id && query.index_type == "BTree",
        ScalarIndexExpr::And(left, right) | ScalarIndexExpr::Or(left, right) => {
            has_id_btree(left, id) || has_id_btree(right, id)
        }
        ScalarIndexExpr::Not(inner) => has_id_btree(inner, id),
    }
}

/// Source-only adapter; non-Lance providers and relationship sources are untouched.
pub(crate) struct RuntimeIdSourceCatalog {
    pub inner: Arc<dyn lance_graph_catalog::GraphSourceCatalog>,
    pub config: crate::GraphConfig,
    pub metrics: Arc<RuntimeIdFilterMetrics>,
}

impl lance_graph_catalog::GraphSourceCatalog for RuntimeIdSourceCatalog {
    fn node_source(&self, label: &str) -> Option<Arc<dyn datafusion::logical_expr::TableSource>> {
        let source = self.inner.node_source(label)?;
        let Some(mapping) = self.config.get_node_mapping(label) else {
            return Some(source);
        };
        let Ok(provider) = datafusion::datasource::source_as_provider(&source) else {
            return Some(source);
        };
        if provider.downcast_ref::<LanceTableProvider>().is_none() {
            return Some(source);
        }
        let any: Arc<dyn std::any::Any + Send + Sync> = provider;
        let Ok(inner) = Arc::downcast::<LanceTableProvider>(any) else {
            return Some(source);
        };
        Some(Arc::new(datafusion::datasource::DefaultTableSource::new(
            Arc::new(RuntimeProvider::new(
                inner,
                &mapping.id_field,
                self.metrics.clone(),
            )),
        )))
    }
    fn relationship_source(
        &self,
        rel_type: &str,
    ) -> Option<Arc<dyn datafusion::logical_expr::TableSource>> {
        self.inner.relationship_source(rel_type)
    }
}

/// Provenance shared by optional node-index and relationship-adjacency consumers.
/// Only completed filters from null-rejecting inner hash joins may narrow rows.
pub(crate) fn completed_inner_join_filter_ids(plan: &Arc<dyn ExecutionPlan>) -> BTreeSet<u64> {
    use datafusion::{
        common::NullEquality,
        logical_expr::JoinType,
        physical_plan::joins::{HashJoinExec, PartitionMode},
    };
    fn collect(plan: &Arc<dyn ExecutionPlan>, allowed: &mut BTreeSet<u64>) {
        if let Some(join) = plan.downcast_ref::<HashJoinExec>() {
            if join.join_type() == &JoinType::Inner
                && join.partition_mode() == &PartitionMode::CollectLeft
                && join.null_equality() == NullEquality::NullEqualsNothing
            {
                if let Some(filter) = join.dynamic_filter_expr() {
                    allowed.insert(filter.inner().expression_id);
                }
            }
        }
        for child in plan.children() {
            collect(child, allowed);
        }
    }
    let mut allowed = BTreeSet::new();
    collect(plan, &mut allowed);
    allowed
}

/// Validate producer provenance on the final optimized physical tree, after
/// DataFusion's post-phase has remapped expression children. Unknown/TopK,
/// anti/mark/outer joins and NULL-equal joins are deliberately not supported.
pub(crate) fn bind_runtime_id_sources(
    plan: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    fn bind(
        plan: Arc<dyn ExecutionPlan>,
        allowed: &BTreeSet<u64>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if let Some(source) = plan.downcast_ref::<RuntimeExec>() {
            let mut source = source.clone();
            let count = source.runtime_filters.len();
            source
                .runtime_filters
                .retain(|filter| allowed.contains(&filter.inner().expression_id));
            source.metrics.rejected_producer_filters.fetch_add(
                (count - source.runtime_filters.len()) as u64,
                Ordering::Relaxed,
            );
            source.validated_producer = true;
            return Ok(Arc::new(source));
        }
        let children = plan
            .children()
            .into_iter()
            .map(|child| bind(child.clone(), allowed))
            .collect::<Result<Vec<_>>>()?;
        datafusion::physical_plan::with_new_children_if_necessary(plan, children)
    }
    let allowed = completed_inner_join_filter_ids(&plan);
    bind(plan, &allowed)
}
