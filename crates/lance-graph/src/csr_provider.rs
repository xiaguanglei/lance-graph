// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Optional snapshot-local adjacency access path for ordinary relationship scans.
//! Only positive, bounded endpoint equality/membership predicates are consumed.
//! Original predicates are evaluated exactly inside the provider; JOINs and
//! runtime parent filters remain in place. Volatile predicates are not pushed.
//! No global cache, persistent sidecar, business rules, or BFS/path rewrite.

use crate::datafusion_planner::col;
use crate::{csr_walk::AdjacencyView, GraphConfig};
use arrow_array::{Array, Int64Array, RecordBatch, RecordBatchOptions, StringArray, UInt64Array};
use arrow_schema::{DataType, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    catalog::Session,
    common::{
        config::ConfigOptions,
        tree_node::{TreeNode, TreeNodeRecursion},
        DFSchema, DataFusionError, Result, ScalarValue, Statistics,
    },
    datasource::{source_as_provider, DefaultTableSource, TableProvider},
    execution::{SessionState, TaskContext},
    logical_expr::{lit, Expr, Operator, TableProviderFilterPushDown, TableSource, TableType},
    physical_expr::{
        expressions::{BinaryExpr, Column, DynamicFilterPhysicalExpr, InListExpr, Literal},
        EquivalenceProperties, PhysicalExpr,
    },
    physical_plan::{
        coalesce_partitions::CoalescePartitionsExec,
        execution_plan::{Boundedness, EmissionType},
        filter::FilterExec,
        filter_pushdown::{
            ChildPushdownResult, FilterPushdownPhase, FilterPushdownPropagation, PushedDown,
        },
        projection::ProjectionExec,
        stream::RecordBatchStreamAdapter,
        DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
        PlanProperties, SendableRecordBatchStream,
    },
};
use futures::{stream, TryStreamExt};
use lance::{
    datafusion::LanceTableProvider, dataset::ProjectionRequest, index::DatasetIndexExt, Dataset,
};
use lance_graph_catalog::GraphSourceCatalog;
use lance_index::{scalar::expression::ScalarIndexExpr, IndexType};
use std::{
    collections::BTreeSet,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

pub(crate) use crate::csr_cache::AdjacencySnapshot;
pub use crate::csr_cache::{CsrAdjacencyCache, CsrCacheStats};

/// Resource guards, not result truncation limits. Exceeding any guard keeps the
/// original Lance scan. Bytes conservatively estimate CSR construction scratch;
/// this accounting is not a hard allocator/RSS cap.
#[derive(Debug, Clone)]
pub struct CsrScanOptions {
    /// Largest edge table (live rows) that gets an adjacency snapshot.
    pub max_edges: usize,
    pub max_vertices: usize,
    /// Peak scratch of one cold build (about 64 bytes per edge plus vertices).
    pub max_bytes: usize,
    /// Cached entries (adjacency snapshots and node-ID maps of all versions);
    /// the least recently used idle entry is evicted beyond this.
    pub max_cache_slots: usize,
    /// Retained rows / bytes of all cached entries (about 32 bytes per edge
    /// plus vertices); idle entries are evicted to stay within them.
    pub max_cache_rows: usize,
    pub max_cache_bytes: usize,
    pub max_bound_keys: usize,
    pub max_take_rows: usize,
    /// Bytes of stored property columns cached for native paths, so hot
    /// queries read node/edge properties from memory instead of storage.
    /// 0 (the default) disables it. Columns are cached on first use per
    /// version (shared across versions per unchanged fragment) and evicted
    /// least-recently-used within this budget, separately from adjacency.
    pub max_column_cache_bytes: usize,
    /// Largest single cached column (all its fragments of one version), so
    /// one wide column (long text) cannot take the whole budget and evict
    /// the small hot ones. 0 allows up to `max_column_cache_bytes`.
    pub max_column_bytes_per_column: usize,
    /// Load a missing column in a background task and serve the query that
    /// missed it from storage, so a first query is never slower than without
    /// the cache (default). When false the query waits for the load.
    pub column_cache_background_fill: bool,
}

impl Default for CsrScanOptions {
    fn default() -> Self {
        Self {
            max_edges: 1_000_000,
            max_vertices: 500_000,
            max_bytes: 128 * 1024 * 1024,
            max_cache_slots: 16,
            max_cache_rows: 2_000_000,
            max_cache_bytes: 256 * 1024 * 1024,
            max_bound_keys: 256,
            max_take_rows: 20_000,
            max_column_cache_bytes: 0,
            max_column_bytes_per_column: 0,
            column_cache_background_fill: true,
        }
    }
}

/// Actual execution counters; a plan name alone is not proof of CSR consumption.
#[derive(Debug, Default)]
pub struct CsrScanMetrics {
    pub planned_scans: AtomicU64,
    pub build_count: AtomicU64,
    pub cache_hits: AtomicU64,
    pub csr_scans: AtomicU64,
    /// Rows returned by successful physical `Dataset::take_rows` calls. Empty
    /// results/zero-column projections are synthesized and do not count as I/O.
    pub take_rows: AtomicU64,
    pub fallbacks: AtomicU64,
    /// Static endpoint BTree metadata admissions; not a physical I/O counter.
    pub native_index_plans: AtomicU64,
    /// Executed CSR scans consuming a completed, provenance-approved JOIN set.
    pub runtime_scans: AtomicU64,
    pub runtime_keys: AtomicU64,
    /// Executed native scans narrowed by a completed JOIN set.
    pub runtime_native_scans: AtomicU64,
    /// Such native plans containing a relevant endpoint BTree scalar-index node.
    pub runtime_native_index_plans: AtomicU64,
    pub timeout_fallbacks: AtomicU64,
    pub unsupported_runtime_fallbacks: AtomicU64,
    pub runtime_planning_fallbacks: AtomicU64,
    pub rejected_producer_filters: AtomicU64,
    pub native_multihop_planned: AtomicU64,
    pub native_multihop_executions: AtomicU64,
    pub native_multihop_rows: AtomicU64,
    pub native_multihop_edge_checks: AtomicU64,
    pub native_multihop_fallbacks: AtomicU64,
    /// Projected input consumed while preparing native per-query masks,
    /// including guarded attempts that later fall back. Bytes are Arrow array
    /// memory estimates, not process RSS or storage/network bytes.
    pub native_multihop_input_rows: AtomicU64,
    pub native_multihop_input_bytes: AtomicU64,
    /// Completed node reads whose executed plan contains a BTree lookup on an
    /// actual node-ID field, rather than merely an index on an ACL property.
    pub native_multihop_node_index_reads: AtomicU64,
    /// Completed node reads served by `take_rows` through a cached, pinned
    /// node-ID -> row-ID map instead of a filtered node table scan.
    pub native_multihop_node_take_reads: AtomicU64,
    /// Node-ID maps built (cold or incremental) for this query; hits are not
    /// counted here.
    pub native_multihop_node_key_builds: AtomicU64,
    /// Snapshot/node-map builds (included above) derived incrementally from a
    /// cached version instead of a full table scan.
    pub incremental_builds: AtomicU64,
    /// Idle cache entries evicted to make room for this query's builds.
    pub cache_evictions: AtomicU64,
    /// Wall time this query spent building cache entries (cold or delta).
    pub cache_build_nanos: AtomicU64,
    /// Node/edge property reads served from the column cache (no storage I/O).
    pub column_cache_reads: AtomicU64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Endpoint {
    Text(String),
    Number(u64),
    /// Int64 keys. Every hop requires identical key types, so values never
    /// meet `Number` keys.
    Signed(i64),
}

/// Node and edge key types the CSR paths index.
pub(crate) fn endpoint_type(datatype: &DataType) -> bool {
    matches!(
        datatype,
        DataType::Utf8 | DataType::UInt64 | DataType::Int64
    )
}

pub(crate) fn array_endpoint(array: &dyn Array, row: usize) -> Option<Endpoint> {
    if array.is_null(row) {
        return None;
    }
    let any = array.as_any();
    if let Some(array) = any.downcast_ref::<StringArray>() {
        Some(Endpoint::Text(array.value(row).into()))
    } else if let Some(array) = any.downcast_ref::<UInt64Array>() {
        Some(Endpoint::Number(array.value(row)))
    } else {
        any.downcast_ref::<Int64Array>()
            .map(|array| Endpoint::Signed(array.value(row)))
    }
}

fn scalar_endpoint(value: &ScalarValue, datatype: &DataType) -> Option<Endpoint> {
    match (value, datatype) {
        (ScalarValue::Utf8(Some(value)), DataType::Utf8) => Some(Endpoint::Text(value.clone())),
        (ScalarValue::UInt64(Some(value)), DataType::UInt64) => Some(Endpoint::Number(*value)),
        (ScalarValue::Int64(Some(value)), DataType::UInt64) => {
            u64::try_from(*value).ok().map(Endpoint::Number)
        }
        (ScalarValue::Int64(Some(value)), DataType::Int64) => Some(Endpoint::Signed(*value)),
        (ScalarValue::UInt64(Some(value)), DataType::Int64) => {
            i64::try_from(*value).ok().map(Endpoint::Signed)
        }
        _ => None,
    }
}

// A necessary AND conjunct is a safe candidate superset. An OR is safe only if
// both branches independently constrain this same field: union their necessary
// supersets, retaining every original residual. Never guess a missing branch's
// bounds (cross-field OR, NOT, casts or NULL).
pub(crate) fn bound_keys(
    expr: &Expr,
    field: &str,
    datatype: &DataType,
    max: usize,
) -> Option<BTreeSet<Endpoint>> {
    if let Expr::BinaryExpr(binary) = expr {
        if binary.op == Operator::And {
            return intersect_optional(
                bound_keys(&binary.left, field, datatype, max),
                bound_keys(&binary.right, field, datatype, max),
            );
        }
        if binary.op == Operator::Or {
            let mut keys = bound_keys(&binary.left, field, datatype, max)?;
            keys.extend(bound_keys(&binary.right, field, datatype, max)?);
            return (keys.len() <= max).then_some(keys);
        }
        if binary.op == Operator::Eq {
            for (column, literal) in [(&binary.left, &binary.right), (&binary.right, &binary.left)]
            {
                if let (Expr::Column(column), Expr::Literal(value, _)) =
                    (column.as_ref(), literal.as_ref())
                {
                    if column.name == field && max != 0 {
                        return Some(BTreeSet::from([scalar_endpoint(value, datatype)?]));
                    }
                }
            }
        }
    }
    if let Expr::InList(list) = expr {
        let Expr::Column(column) = list.expr.as_ref() else {
            return None;
        };
        if list.negated || column.name != field || list.list.len() > max {
            return None;
        }
        return list
            .list
            .iter()
            .map(|expr| match expr {
                Expr::Literal(value, _) => scalar_endpoint(value, datatype),
                _ => None,
            })
            .collect();
    }
    None
}

fn intersect_optional<T: Ord>(
    left: Option<BTreeSet<T>>,
    right: Option<BTreeSet<T>>,
) -> Option<BTreeSet<T>> {
    match (left, right) {
        (Some(left), Some(right)) => Some(
            left.into_iter()
                .filter(|value| right.contains(value))
                .collect(),
        ),
        (left, right) => left.or(right),
    }
}

fn candidate_rows(
    snapshot: &AdjacencySnapshot,
    source: Option<&BTreeSet<Endpoint>>,
    target: Option<&BTreeSet<Endpoint>>,
    max: usize,
) -> Option<Vec<u64>> {
    let collect = |keys: &BTreeSet<Endpoint>, outgoing: bool| {
        let index = snapshot.view(outgoing);
        let mut rows = BTreeSet::new();
        for key in keys {
            if let Some(vertex) = snapshot.vertex(key) {
                for (id, _) in index.neighbor_edges(vertex) {
                    rows.insert(id);
                    if rows.len() > max {
                        return None;
                    }
                }
            }
        }
        Some(rows)
    };
    let source = match source {
        Some(keys) => Some(collect(keys, true)?),
        None => None,
    };
    let target = match target {
        Some(keys) => Some(collect(keys, false)?),
        None => None,
    };
    intersect_optional(source, target).map(|rows| rows.into_iter().collect())
}

// Only ordinary scalar columns whose full type/nullability/metadata matches the
// raw snapshot are eligible without asking the native scanner to plan first.
// In particular, exclude system columns and blob/JSON/extension conversions.
fn ordinary_projection(
    provider: &SchemaRef,
    raw: &Schema,
    projection: Option<&Vec<usize>>,
) -> Option<SchemaRef> {
    let projected = match projection {
        Some(indices) => Arc::new(provider.project(indices).ok()?),
        None => provider.clone(),
    };
    if projected.metadata() != raw.metadata() {
        return None;
    }
    let mut names = BTreeSet::new();
    for field in projected.fields() {
        let original = raw.field_with_name(field.name()).ok()?;
        if field.as_ref() != original
            || !names.insert(field.name())
            || matches!(
                field.name().as_str(),
                "_rowid"
                    | "_rowaddr"
                    | "_rowoffset"
                    | "_row_created_at_version"
                    | "_row_last_updated_at_version"
            )
            || field.metadata().contains_key("ARROW:extension:name")
            || !matches!(
                field.data_type(),
                DataType::Boolean
                    | DataType::Int8
                    | DataType::Int16
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::UInt8
                    | DataType::UInt16
                    | DataType::UInt32
                    | DataType::UInt64
                    | DataType::Float16
                    | DataType::Float32
                    | DataType::Float64
                    | DataType::Utf8
                    | DataType::LargeUtf8
                    | DataType::Date32
                    | DataType::Date64
                    | DataType::Time32(_)
                    | DataType::Time64(_)
                    | DataType::Timestamp(_, _)
                    | DataType::Duration(_)
            )
        {
            return None;
        }
    }
    Some(projected)
}

// Exact pushdown requires evaluating the full original predicate, not merely
// its necessary endpoint bound. Append hidden predicate columns after caller
// columns so projection remapping remains explicit and deterministic.
fn exact_filter_projection(
    session: &SessionState,
    provider: &SchemaRef,
    raw: &Schema,
    projection: Option<&Vec<usize>>,
    filters: &[Expr],
) -> Option<(Vec<usize>, SchemaRef, Option<Arc<dyn PhysicalExpr>>)> {
    let mut indices = projection
        .cloned()
        .unwrap_or_else(|| (0..provider.fields().len()).collect());
    let mut referenced = BTreeSet::new();
    for filter in filters {
        let mut supported = true;
        filter
            .apply(|expression| {
                supported &= match expression {
                    Expr::Column(column) => column.relation.is_none(),
                    Expr::Alias(_)
                    | Expr::Literal(_, _)
                    | Expr::BinaryExpr(_)
                    | Expr::Like(_)
                    | Expr::SimilarTo(_)
                    | Expr::Not(_)
                    | Expr::IsNotNull(_)
                    | Expr::IsNull(_)
                    | Expr::IsTrue(_)
                    | Expr::IsFalse(_)
                    | Expr::IsUnknown(_)
                    | Expr::IsNotTrue(_)
                    | Expr::IsNotFalse(_)
                    | Expr::IsNotUnknown(_)
                    | Expr::Negative(_)
                    | Expr::Between(_)
                    | Expr::Case(_)
                    | Expr::Cast(_)
                    | Expr::TryCast(_)
                    | Expr::ScalarFunction(_)
                    | Expr::InList(_) => true,
                    _ => false,
                };
                Ok(if supported {
                    TreeNodeRecursion::Continue
                } else {
                    TreeNodeRecursion::Stop
                })
            })
            .ok()?;
        if !supported || filter.is_volatile() {
            return None;
        }
        for column in filter.column_refs() {
            referenced.insert(provider.index_of(&column.name).ok()?);
        }
    }
    for index in referenced {
        if !indices.contains(&index) {
            indices.push(index);
        }
    }
    let schema = ordinary_projection(provider, raw, Some(&indices))?;
    let predicate = if let Some(predicate) = filters.iter().cloned().reduce(Expr::and) {
        let schema = DFSchema::try_from(schema.as_ref().clone()).ok()?;
        Some(session.create_physical_expr(predicate, &schema).ok()?)
    } else {
        None
    };
    Some((indices, schema, predicate))
}

fn exact_csr_output(
    mut plan: Arc<dyn ExecutionPlan>,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    output: &SchemaRef,
) -> Result<Arc<dyn ExecutionPlan>> {
    if let Some(predicate) = predicate {
        plan = Arc::new(FilterExec::try_new(predicate, plan)?);
    }
    if &plan.schema() != output {
        let fields = output.fields().iter().enumerate().map(|(index, field)| {
            (
                Arc::new(Column::new(field.name(), index)) as Arc<dyn PhysicalExpr>,
                field.name().clone(),
            )
        });
        plan = Arc::new(ProjectionExec::try_new(fields, plan)?);
    }
    if &plan.schema() != output {
        return Err(DataFusionError::Plan(
            "CSR exact projection changed caller schema".into(),
        ));
    }
    Ok(plan)
}

// Metadata admission is exact and pinned; execution coverage, append fragments
// and deletion masks remain the native Lance scanner's responsibility.
async fn endpoint_btrees(
    dataset: &Dataset,
    source: &str,
    target: &str,
) -> Option<BTreeSet<String>> {
    let indices = dataset.load_indices().await.ok()?;
    let mut fields = BTreeSet::new();
    for name in [source, target] {
        let field = dataset.schema().field(name)?;
        if indices.iter().any(|index| {
            index.fields.as_slice() == [field.id]
                && index
                    .index_details
                    .as_ref()
                    .is_some_and(|details| IndexType::BTree.matches_details(details))
        }) {
            fields.insert(name.to_string());
        }
    }
    Some(fields)
}

#[derive(Debug, Clone)]
struct RuntimeEndpointFilter {
    dynamic: Arc<DynamicFilterPhysicalExpr>,
    source: bool,
}

// Consume only an exact positive membership conjunct after the approved JOIN
// producer completes. Min/max, OR, NULL, casts and partial key sets are not CSR
// bounds. The whole original JOIN/filter remains in the physical plan.
fn runtime_membership(
    expr: &Arc<dyn PhysicalExpr>,
    name: &str,
    index: usize,
    datatype: &DataType,
    max: usize,
) -> Option<BTreeSet<Endpoint>> {
    if expr
        .downcast_ref::<Literal>()
        .is_some_and(|literal| literal.value() == &ScalarValue::Boolean(Some(false)))
    {
        // A completed empty build side is a proven empty inner JOIN, not an
        // unsupported or partially populated producer set.
        return Some(BTreeSet::new());
    }
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        if binary.op() == &Operator::And {
            return intersect_optional(
                runtime_membership(binary.left(), name, index, datatype, max),
                runtime_membership(binary.right(), name, index, datatype, max),
            );
        }
        return None;
    }
    let list = expr.downcast_ref::<InListExpr>()?;
    let column = list.expr().downcast_ref::<Column>()?;
    if list.negated() || column.name() != name || column.index() != index || list.len() > max {
        return None;
    }
    let mut bytes = 0usize;
    list.list()
        .iter()
        .map(|value| {
            let literal = value.downcast_ref::<Literal>()?;
            let key = scalar_endpoint(literal.value(), datatype)?;
            bytes = bytes.checked_add(match &key {
                Endpoint::Text(text) => text.len(),
                Endpoint::Number(_) | Endpoint::Signed(_) => 8,
            })?;
            if bytes > 128 * 1024 {
                return None;
            }
            Some(key)
        })
        .collect()
}

fn endpoint_predicate(name: &str, keys: &BTreeSet<Endpoint>) -> Expr {
    col(name).in_list(
        keys.iter()
            .map(|key| match key {
                Endpoint::Text(text) => lit(text.clone()),
                Endpoint::Number(value) => lit(*value),
                Endpoint::Signed(value) => lit(*value),
            })
            .collect(),
        false,
    )
}

pub(crate) fn endpoint_btree_plan(plan: &Arc<dyn ExecutionPlan>, names: &BTreeSet<String>) -> bool {
    fn expression(expr: &ScalarIndexExpr, names: &BTreeSet<String>) -> bool {
        match expr {
            ScalarIndexExpr::Query(query) => {
                names.contains(&query.column) && query.index_type == "BTree"
            }
            ScalarIndexExpr::And(left, right) | ScalarIndexExpr::Or(left, right) => {
                expression(left, names) || expression(right, names)
            }
            ScalarIndexExpr::Not(inner) => expression(inner, names),
        }
    }
    if let Some(index) = plan.downcast_ref::<lance::io::exec::scalar_index::ScalarIndexExec>() {
        if expression(index.expr(), names) {
            return true;
        }
    }
    plan.children()
        .iter()
        .any(|child| endpoint_btree_plan(child, names))
}

fn single_partition(
    plan: Arc<dyn ExecutionPlan>,
    schema: &SchemaRef,
) -> Result<Arc<dyn ExecutionPlan>> {
    if &plan.schema() != schema {
        return Err(DataFusionError::Internal(
            "CSR fallback changed the source schema".into(),
        ));
    }
    Ok(if plan.output_partitioning().partition_count() == 1 {
        plan
    } else {
        Arc::new(CoalescePartitionsExec::new(plan))
    })
}

pub(crate) struct CsrSourceCatalog {
    inner: Arc<dyn GraphSourceCatalog>,
    config: GraphConfig,
    cache: Arc<CsrAdjacencyCache>,
    metrics: Arc<CsrScanMetrics>,
}

impl CsrSourceCatalog {
    pub(crate) fn new(
        inner: Arc<dyn GraphSourceCatalog>,
        config: GraphConfig,
        cache: Arc<CsrAdjacencyCache>,
        metrics: Arc<CsrScanMetrics>,
    ) -> Self {
        Self {
            inner,
            config,
            cache,
            metrics,
        }
    }
}

impl GraphSourceCatalog for CsrSourceCatalog {
    fn node_source(&self, label: &str) -> Option<Arc<dyn TableSource>> {
        self.inner.node_source(label)
    }
    fn relationship_source(&self, rel_type: &str) -> Option<Arc<dyn TableSource>> {
        let source = self.inner.relationship_source(rel_type)?;
        let Some(mapping) = self.config.get_relationship_mapping(rel_type) else {
            return Some(source);
        };
        let Ok(provider) = source_as_provider(&source) else {
            return Some(source);
        };
        if provider.downcast_ref::<LanceTableProvider>().is_none() {
            return Some(source);
        }
        let any: Arc<dyn std::any::Any + Send + Sync> = provider;
        let Ok(inner) = Arc::downcast::<LanceTableProvider>(any) else {
            return Some(source);
        };
        Some(Arc::new(DefaultTableSource::new(Arc::new(
            CsrRelationshipProvider {
                inner,
                source: mapping.source_id_field.clone(),
                target: mapping.target_id_field.clone(),
                cache: self.cache.clone(),
                metrics: self.metrics.clone(),
            },
        ))))
    }
}

#[derive(Debug)]
struct CsrRelationshipProvider {
    inner: Arc<LanceTableProvider>,
    source: String,
    target: String,
    cache: Arc<CsrAdjacencyCache>,
    metrics: Arc<CsrScanMetrics>,
}

pub(crate) fn underlying_lance(
    provider: &Arc<dyn TableProvider>,
) -> Option<Arc<LanceTableProvider>> {
    if let Some(csr) = provider.downcast_ref::<CsrRelationshipProvider>() {
        return Some(csr.inner.clone());
    }
    let any: Arc<dyn std::any::Any + Send + Sync> = provider.clone();
    Arc::downcast::<LanceTableProvider>(any).ok()
}

#[async_trait]
impl TableProvider for CsrRelationshipProvider {
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
        // Keep native Exact semantics and statistics. CSR candidates receive a
        // complete internal FilterExec before restoring caller projection.
        let mut support = self.inner.supports_filters_pushdown(filters)?;
        for (filter, support) in filters.iter().zip(&mut support) {
            if filter.is_volatile() {
                *support = TableProviderFilterPushDown::Unsupported;
            }
        }
        Ok(support)
    }
    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if filters.iter().any(Expr::is_volatile) {
            return Err(DataFusionError::Plan(
                "CSR source does not accept volatile filter pushdown".into(),
            ));
        }
        let Some(session) = state.as_any().downcast_ref::<SessionState>() else {
            return self.inner.scan(state, projection, filters, limit).await;
        };
        let schema = self.inner.schema();
        let dataset_schema = Schema::from(self.inner.dataset().schema());
        let output_schema = ordinary_projection(&schema, &dataset_schema, projection);
        let endpoint_supported = |field: &str| match (
            schema.field_with_name(field),
            dataset_schema.field_with_name(field),
        ) {
            (Ok(actual), Ok(raw)) => {
                actual == raw
                    && endpoint_type(actual.data_type())
                    && !actual.metadata().contains_key("ARROW:extension:name")
            }
            _ => false,
        };
        if limit.is_some()
            || self.source == self.target
            || output_schema.is_none()
            || !endpoint_supported(&self.source)
            || !endpoint_supported(&self.target)
        {
            return self.inner.scan(state, projection, filters, limit).await;
        }
        let bound = |field: &str| -> Option<BTreeSet<Endpoint>> {
            let datatype = schema.field_with_name(field).ok()?.data_type();
            if !endpoint_type(datatype) {
                return None;
            }
            filters.iter().fold(None, |keys, filter| {
                intersect_optional(
                    keys,
                    bound_keys(filter, field, datatype, self.cache.options().max_bound_keys),
                )
            })
        };
        let sources = bound(&self.source);
        let targets = bound(&self.target);
        if sources.is_some() || targets.is_some() {
            let Some(indexed) =
                endpoint_btrees(self.inner.dataset().as_ref(), &self.source, &self.target).await
            else {
                return self.inner.scan(state, projection, filters, limit).await;
            };
            if sources.is_some() && indexed.contains(&self.source)
                || targets.is_some() && indexed.contains(&self.target)
            {
                let plan = self.inner.scan(state, projection, filters, limit).await?;
                self.metrics
                    .native_index_plans
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(plan);
            }
        } else if !output_schema
            .as_ref()
            .unwrap()
            .fields()
            .iter()
            .any(|field| field.name() == &self.source || field.name() == &self.target)
        {
            // No static restriction and no projected endpoint to receive a
            // remapped runtime JOIN filter: retain the ordinary native plan.
            return self.inner.scan(state, projection, filters, limit).await;
        }
        let Some((input_projection, input_schema, predicate)) =
            exact_filter_projection(session, &schema, &dataset_schema, projection, filters)
        else {
            return self.inner.scan(state, projection, filters, limit).await;
        };
        self.metrics.planned_scans.fetch_add(1, Ordering::Relaxed);
        // Do not erase the known manifest cardinality merely because the
        // optional source can later use adjacency. Unknown relationship stats
        // can make DataFusion put a large indexed node source on the build
        // side, where completed JOIN membership cannot narrow its read.
        // This is metadata-only; legacy/partial manifests remain unknown.
        let mut statistics = Statistics::new_unknown(&input_schema);
        let dataset = self.inner.dataset();
        if dataset.manifest().writer_version.is_some() {
            if let Some(rows) = dataset
                .manifest()
                .fragments
                .iter()
                .try_fold(0usize, |rows, fragment| {
                    rows.checked_add(fragment.num_rows()?)
                })
            {
                statistics.num_rows = datafusion::common::stats::Precision::Exact(rows);
            }
        }
        let plan: Arc<dyn ExecutionPlan> = Arc::new(CsrScanExec {
            // Do not plan the native scanner on a successful CSR path. Nor may
            // endpoint candidates advertise filter-derived constants/order to
            // DF before the retained residual filter has been evaluated.
            properties: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(input_schema),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Incremental,
                Boundedness::Bounded,
            )),
            inner: self.inner.clone(),
            source: self.source.clone(),
            target: self.target.clone(),
            cache: self.cache.clone(),
            metrics: self.metrics.clone(),
            session: session.clone(),
            projection: Some(input_projection),
            filters: filters.to_vec(),
            sources,
            targets,
            runtime_filters: Vec::new(),
            validated_runtime: false,
            wait: Duration::from_millis(100),
            statistics: Arc::new(statistics),
        });
        // Compilation/schema errors are handled before reading data. Genuine
        // filtering/take execution errors propagate and are never replayed.
        match exact_csr_output(plan, predicate, &output_schema.unwrap()) {
            Ok(plan) => Ok(plan),
            Err(_) => self.inner.scan(state, projection, filters, limit).await,
        }
    }
}

#[derive(Debug, Clone)]
struct CsrScanExec {
    properties: Arc<PlanProperties>,
    inner: Arc<LanceTableProvider>,
    source: String,
    target: String,
    cache: Arc<CsrAdjacencyCache>,
    metrics: Arc<CsrScanMetrics>,
    session: SessionState,
    projection: Option<Vec<usize>>,
    filters: Vec<Expr>,
    sources: Option<BTreeSet<Endpoint>>,
    targets: Option<BTreeSet<Endpoint>>,
    runtime_filters: Vec<RuntimeEndpointFilter>,
    validated_runtime: bool,
    wait: Duration,
    statistics: Arc<Statistics>,
}

struct RuntimeBounds {
    sources: Option<BTreeSet<Endpoint>>,
    targets: Option<BTreeSet<Endpoint>>,
    keys: usize,
}

impl CsrScanExec {
    async fn completed_runtime_bounds(&self) -> Option<RuntimeBounds> {
        if !self.validated_runtime || self.runtime_filters.is_empty() {
            return None;
        }
        let deadline = tokio::time::Instant::now() + self.wait;
        let mut bounds = RuntimeBounds {
            sources: None,
            targets: None,
            keys: 0,
        };
        for filter in &self.runtime_filters {
            if tokio::time::timeout_at(deadline, filter.dynamic.wait_complete())
                .await
                .is_err()
            {
                self.metrics
                    .timeout_fallbacks
                    .fetch_add(1, Ordering::Relaxed);
                return None;
            }
            let name = if filter.source {
                &self.source
            } else {
                &self.target
            };
            let schema = self.schema();
            let keys = filter.dynamic.current().ok().and_then(|expression| {
                let index = schema.index_of(name).ok()?;
                runtime_membership(
                    &expression,
                    name,
                    index,
                    schema.field(index).data_type(),
                    self.cache.options().max_bound_keys,
                )
            });
            let Some(keys) = keys else {
                self.metrics
                    .unsupported_runtime_fallbacks
                    .fetch_add(1, Ordering::Relaxed);
                // Each approved JOIN contributes a necessary conjunct. Once
                // this producer has completed, an unsupported large/map set
                // need not discard another producer's bounded necessary set.
                // All original JOINs/parent filters remain; never consume an
                // incomplete producer or infer a bound from its min/max range.
                continue;
            };
            bounds.keys = bounds.keys.checked_add(keys.len())?;
            let destination = if filter.source {
                &mut bounds.sources
            } else {
                &mut bounds.targets
            };
            *destination = intersect_optional(destination.take(), Some(keys));
        }
        (bounds.sources.is_some() || bounds.targets.is_some()).then_some(bounds)
    }
}

impl DisplayAs for CsrScanExec {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "CsrAdjacencyScan: source={}, target={}, version={}, source_keys={}, target_keys={}, runtime_filters={}",
            self.source,
            self.target,
            self.inner.dataset().version_id(),
            self.sources.as_ref().map_or(0, BTreeSet::len),
            self.targets.as_ref().map_or(0, BTreeSet::len),
            self.runtime_filters.len()
        )
    }
}

impl ExecutionPlan for CsrScanExec {
    fn name(&self) -> &'static str {
        "CsrAdjacencyScan"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::Internal(
                "CSR source has no children".into(),
            ));
        }
        Ok(self)
    }
    fn reset_state(self: Arc<Self>) -> Result<Arc<dyn ExecutionPlan>> {
        let mut reset = self.as_ref().clone();
        reset.runtime_filters.clear();
        reset.validated_runtime = false;
        Ok(Arc::new(reset))
    }
    fn partition_statistics(&self, _: Option<usize>) -> Result<Arc<Statistics>> {
        if !self.filters.is_empty()
            || self.sources.is_some()
            || self.targets.is_some()
            || !self.runtime_filters.is_empty()
        {
            // A candidate subset need not have the raw snapshot cardinality.
            // Retain its estimate for join costing, never certify an exact
            // count through static/runtime adjacency narrowing.
            Ok(Arc::new(self.statistics.as_ref().clone().to_inexact()))
        } else {
            Ok(self.statistics.clone())
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
            updated.validated_runtime = false;
            for parent in &result.parent_filters {
                let Some(dynamic) = parent.filter.downcast_ref::<DynamicFilterPhysicalExpr>()
                else {
                    continue;
                };
                let children = dynamic.children();
                if children.len() != 1 {
                    continue;
                }
                let Some(column) = children[0].downcast_ref::<Column>() else {
                    continue;
                };
                let schema = self.schema();
                let source = if column.name() == self.source
                    && schema.index_of(&self.source).ok() == Some(column.index())
                {
                    true
                } else if column.name() == self.target
                    && schema.index_of(&self.target).ok() == Some(column.index())
                {
                    false
                } else {
                    continue;
                };
                if let Ok(dynamic) =
                    Arc::downcast::<DynamicFilterPhysicalExpr>(parent.filter.clone())
                {
                    if !updated.runtime_filters.iter().any(|existing| {
                        existing.source == source && Arc::ptr_eq(&existing.dynamic, &dynamic)
                    }) {
                        updated
                            .runtime_filters
                            .push(RuntimeEndpointFilter { dynamic, source });
                    }
                }
            }
        }
        // This optional prefilter may fall back. Never remove a parent filter,
        // an original JOIN, or a business residual based on CSR admission.
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
            return Err(DataFusionError::Internal(
                "CSR source supports partition 0 only".into(),
            ));
        }
        let this = self.clone();
        let schema = self.schema();
        let future = async move {
            let dataset = this.inner.dataset();
            let runtime = this.completed_runtime_bounds().await;
            let mut force_native = false;
            if let Some(bounds) = &runtime {
                let indexed = endpoint_btrees(dataset.as_ref(), &this.source, &this.target).await;
                let names = BTreeSet::from_iter(
                    [
                        bounds.sources.as_ref().map(|_| this.source.clone()),
                        bounds.targets.as_ref().map(|_| this.target.clone()),
                    ]
                    .into_iter()
                    .flatten(),
                );
                // A completed bounded JOIN set can use the native BTree without
                // paying O(E) CSR construction. Unknown metadata is conservative
                // native admission, never evidence of a physical index hit.
                if indexed
                    .as_ref()
                    .is_none_or(|fields| !fields.is_disjoint(&names))
                {
                    let mut filters = this.filters.clone();
                    if let Some(keys) = &bounds.sources {
                        filters.push(endpoint_predicate(&this.source, keys));
                    }
                    if let Some(keys) = &bounds.targets {
                        filters.push(endpoint_predicate(&this.target, keys));
                    }
                    let candidate = this
                        .inner
                        .scan(&this.session, this.projection.as_ref(), &filters, None)
                        .await
                        .and_then(|plan| single_partition(plan, &this.schema()));
                    if let Ok(plan) = candidate {
                        if endpoint_btree_plan(&plan, &names) {
                            this.metrics
                                .runtime_native_index_plans
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        this.metrics
                            .runtime_native_scans
                            .fetch_add(1, Ordering::Relaxed);
                        this.metrics
                            .runtime_keys
                            .fetch_add(bounds.keys as u64, Ordering::Relaxed);
                        return plan.execute(partition, context);
                    }
                    // Optional runtime planning failed before output. Replan
                    // only the original scan; genuine execution errors propagate.
                    this.metrics
                        .runtime_planning_fallbacks
                        .fetch_add(1, Ordering::Relaxed);
                    force_native = true;
                }
            }
            let sources = intersect_optional(
                this.sources.clone(),
                runtime.as_ref().and_then(|bounds| bounds.sources.clone()),
            );
            let targets = intersect_optional(
                this.targets.clone(),
                runtime.as_ref().and_then(|bounds| bounds.targets.clone()),
            );
            let rows = if !force_native && (sources.is_some() || targets.is_some()) {
                if sources.as_ref().is_some_and(BTreeSet::is_empty)
                    || targets.as_ref().is_some_and(BTreeSet::is_empty)
                {
                    Some(Vec::new())
                } else {
                    this.cache
                        .snapshot(dataset.clone(), &this.source, &this.target, &this.metrics)
                        .await
                        .as_ref()
                        .and_then(|snapshot| {
                            candidate_rows(
                                snapshot,
                                sources.as_ref(),
                                targets.as_ref(),
                                this.cache.options().max_take_rows,
                            )
                        })
                }
            } else {
                // No approved completed endpoint set means no wait/topology
                // construction beyond the optional producer check above.
                None
            };
            if let Some(rows) = rows {
                let schema = this.schema();
                let batch = if schema.fields().is_empty() {
                    // Row count is known from the complete pinned adjacency.
                    // Lance cannot read a fragment with an empty data schema.
                    RecordBatch::try_new_with_options(
                        schema.clone(),
                        Vec::new(),
                        &RecordBatchOptions::new().with_row_count(Some(rows.len())),
                    )?
                } else if rows.is_empty() {
                    RecordBatch::new_empty(schema.clone())
                } else {
                    let names: Vec<_> = schema
                        .fields()
                        .iter()
                        .map(|field| field.name().as_str())
                        .collect();
                    let projection = dataset
                        .schema()
                        .project_preserve_system_columns(&names)
                        .map_err(DataFusionError::from)?;
                    // Do not fall back/replay genuine take errors after execution.
                    let batch = dataset
                        .take_rows(&rows, ProjectionRequest::Schema(Arc::new(projection)))
                        .await
                        .map_err(DataFusionError::from)?;
                    this.metrics
                        .take_rows
                        .fetch_add(batch.num_rows() as u64, Ordering::Relaxed);
                    batch
                };
                if batch.schema() != schema {
                    return Err(DataFusionError::Internal(
                        "CSR take changed the source schema".into(),
                    ));
                }
                this.metrics.csr_scans.fetch_add(1, Ordering::Relaxed);
                if let Some(bounds) = &runtime {
                    this.metrics.runtime_scans.fetch_add(1, Ordering::Relaxed);
                    this.metrics
                        .runtime_keys
                        .fetch_add(bounds.keys as u64, Ordering::Relaxed);
                }
                return Ok(Box::pin(RecordBatchStreamAdapter::new(
                    schema,
                    stream::once(async move { Ok(batch) }),
                )) as SendableRecordBatchStream);
            }
            this.metrics.fallbacks.fetch_add(1, Ordering::Relaxed);
            // Replan, never reuse a potentially one-shot hidden native stream.
            let plan = this
                .inner
                .scan(&this.session, this.projection.as_ref(), &this.filters, None)
                .await?;
            let plan = single_partition(plan, &this.schema())?;
            plan.execute(partition, context)
        };
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(future).try_flatten(),
        )))
    }
}

/// Bind only completed-membership-capable producers proven on the final,
/// post-pushdown physical tree. Shared proof rejects TopK, outer/anti/mark joins,
/// non-CollectLeft joins and NULL-equal joins. Never infer provenance by aliases.
pub(crate) fn bind_csr_runtime_sources(
    plan: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    fn bind(
        plan: Arc<dyn ExecutionPlan>,
        allowed: &BTreeSet<u64>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if let Some(source) = plan.downcast_ref::<CsrScanExec>() {
            let mut source = source.clone();
            let before = source.runtime_filters.len();
            source
                .runtime_filters
                .retain(|filter| allowed.contains(&filter.dynamic.inner().expression_id));
            source.metrics.rejected_producer_filters.fetch_add(
                (before - source.runtime_filters.len()) as u64,
                Ordering::Relaxed,
            );
            source.validated_runtime = true;
            return Ok(Arc::new(source));
        }
        let children = plan
            .children()
            .into_iter()
            .map(|child| bind(child.clone(), allowed))
            .collect::<Result<Vec<_>>>()?;
        datafusion::physical_plan::with_new_children_if_necessary(plan, children)
    }
    let allowed = crate::runtime_id_filter::completed_inner_join_filter_ids(&plan);
    bind(plan, &allowed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{RecordBatch, RecordBatchIterator};
    use arrow_schema::{Field, Schema};
    use datafusion::{
        common::NullEquality,
        execution::context::SessionContext,
        logical_expr::{col, lit, JoinType},
        physical_plan::{
            collect,
            joins::{HashJoinExecBuilder, PartitionMode},
        },
        prelude::SessionConfig,
    };
    use lance::dataset::WriteParams;
    use lance_index::scalar::ScalarIndexParams;

    fn has_csr_scan(plan: &Arc<dyn ExecutionPlan>) -> bool {
        plan.downcast_ref::<CsrScanExec>().is_some()
            || plan.children().into_iter().any(has_csr_scan)
    }

    #[test]
    fn positive_membership_only_and_safe_conjuncts() {
        let eq = col("src").eq(lit("a"));
        let other = col("status").eq(lit("visible"));
        assert_eq!(
            bound_keys(&eq.clone().and(other.clone()), "src", &DataType::Utf8, 4),
            Some(BTreeSet::from([Endpoint::Text("a".into())]))
        );
        assert!(bound_keys(&eq.or(other), "src", &DataType::Utf8, 4).is_none());
        assert!(bound_keys(
            &col("src").in_list(vec![lit("a")], true),
            "src",
            &DataType::Utf8,
            4
        )
        .is_none());
        assert!(bound_keys(
            &col("src").eq(lit(ScalarValue::Utf8(None))),
            "src",
            &DataType::Utf8,
            4
        )
        .is_none());
        assert!(bound_keys(&col("src").eq(lit("a")), "src", &DataType::Utf8, 0).is_none());
        assert_eq!(
            bound_keys(&col("src").eq(lit(7u64)), "src", &DataType::UInt64, 4),
            Some(BTreeSet::from([Endpoint::Number(7)]))
        );
    }

    #[test]
    fn ordinary_projection_rejects_system_transformed_and_duplicate_fields() {
        let raw = Schema::new(vec![
            Field::new("src", DataType::Utf8, true),
            Field::new("weight", DataType::UInt64, false),
            Field::new("payload", DataType::Binary, false),
        ]);
        let provider = Arc::new(Schema::new(vec![
            raw.field(0).clone(),
            raw.field(1).clone(),
            raw.field(2).clone(),
            Field::new("_rowid", DataType::UInt64, false),
        ]));
        assert!(ordinary_projection(&provider, &raw, Some(&vec![1, 0])).is_some());
        assert!(ordinary_projection(&provider, &raw, Some(&vec![])).is_some());
        for indices in [vec![2], vec![3], vec![0, 0], vec![99]] {
            assert!(ordinary_projection(&provider, &raw, Some(&indices)).is_none());
        }
        assert!(ordinary_projection(&provider, &raw, None).is_none());
        let changed = Arc::new(Schema::new(vec![raw.field(0).clone().with_metadata(
            std::collections::HashMap::from([("changed".into(), "true".into())]),
        )]));
        assert!(ordinary_projection(&changed, &raw, None).is_none());
        let extension_field = Field::new("src", DataType::Utf8, true).with_metadata(
            std::collections::HashMap::from([("ARROW:extension:name".into(), "custom".into())]),
        );
        let extension = Arc::new(Schema::new(vec![extension_field]));
        assert!(ordinary_projection(&extension, extension.as_ref(), None).is_none());
    }

    #[test]
    fn same_field_or_unions_bounds() {
        let expression = col("src")
            .eq(lit("a"))
            .or(col("src").eq(lit("b")))
            .or(col("src").eq(lit("a")));
        assert_eq!(
            bound_keys(&expression, "src", &DataType::Utf8, 2),
            Some(BTreeSet::from([
                Endpoint::Text("a".into()),
                Endpoint::Text("b".into())
            ]))
        );
    }

    #[test]
    fn cross_field_or_null_and_not_have_no_guessed_branch_bound() {
        let source = col("src").eq(lit("a"));
        for other in [
            col("dst").eq(lit("b")),
            col("src").eq(lit(ScalarValue::Utf8(None))),
            col("src").in_list(vec![lit("b")], true),
        ] {
            assert!(bound_keys(&source.clone().or(other), "src", &DataType::Utf8, 4).is_none());
        }
    }

    #[test]
    fn nested_and_or_uses_both_necessary_same_field_supersets() {
        let expression = col("src")
            .eq(lit("a"))
            .and(col("status").eq(lit("visible")))
            .or(col("src")
                .in_list(vec![lit("b"), lit("c")], false)
                .and(col("dst").eq(lit("target"))))
            .and(col("src").in_list(vec![lit("a"), lit("c")], false));
        assert_eq!(
            bound_keys(&expression, "src", &DataType::Utf8, 4),
            Some(BTreeSet::from([
                Endpoint::Text("a".into()),
                Endpoint::Text("c".into())
            ]))
        );
        assert!(bound_keys(&expression, "dst", &DataType::Utf8, 4).is_none());
    }

    #[test]
    fn same_field_or_union_budget_falls_back() {
        let expression = col("src").eq(lit("a")).or(col("src").eq(lit("b")));
        assert!(bound_keys(&expression, "src", &DataType::Utf8, 1).is_none());
        assert!(bound_keys(&expression, "src", &DataType::Utf8, 0).is_none());
    }

    async fn fixture(
        nullable: bool,
        options: CsrScanOptions,
    ) -> (tempfile::TempDir, CsrRelationshipProvider) {
        let temp = tempfile::tempdir().unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("src", DataType::Utf8, true),
            Field::new("dst", DataType::Utf8, false),
            Field::new("weight", DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![
                    Some("a"),
                    Some("a"),
                    Some("b"),
                    if nullable { None } else { Some("a") },
                ])),
                Arc::new(StringArray::from(vec!["b", "b", "b", "c"])),
                Arc::new(UInt64Array::from(vec![1, 2, 3, 4])),
            ],
        )
        .unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            temp.path().join("edges.lance").to_str().unwrap(),
            None::<WriteParams>,
        )
        .await
        .unwrap();
        (
            temp,
            CsrRelationshipProvider {
                inner: Arc::new(LanceTableProvider::new(Arc::new(dataset), false, false)),
                source: "src".into(),
                target: "dst".into(),
                cache: Arc::new(CsrAdjacencyCache::new(options)),
                metrics: Arc::new(CsrScanMetrics::default()),
            },
        )
    }

    #[tokio::test]
    async fn reverse_identity_matches_dense_mapping_and_fits_the_same_budget() {
        let (_temp, provider) = fixture(false, CsrScanOptions::default()).await;
        let snapshot = provider
            .cache
            .snapshot(provider.inner.dataset(), "src", "dst", &provider.metrics)
            .await
            .unwrap();
        for key in ["a", "b", "c"] {
            let key = Endpoint::Text(key.into());
            let id = snapshot.vertex(&key).unwrap();
            assert_eq!(snapshot.endpoint(id), Some(&key));
        }
        assert!(snapshot.vertex(&Endpoint::Text("missing".into())).is_none());
        // The charged size is what the cache must fit; one byte less is rejected.
        let charged = provider.cache.stats().bytes;
        assert!(charged > 0);
        let cache = CsrAdjacencyCache::new(CsrScanOptions {
            max_cache_bytes: charged - 1,
            ..Default::default()
        });
        assert!(cache
            .snapshot(
                provider.inner.dataset(),
                "src",
                "dst",
                &CsrScanMetrics::default()
            )
            .await
            .is_none());
    }

    #[tokio::test]
    async fn parallel_self_loop_and_two_endpoint_intersection() {
        let (_temp, provider) = fixture(false, CsrScanOptions::default()).await;
        let ctx = SessionContext::new();
        let filters = [col("src").eq(lit("a")), col("dst").eq(lit("b"))];
        for _ in 0..2 {
            let plan = provider
                .scan(&ctx.state(), None, &filters, None)
                .await
                .unwrap();
            assert!(has_csr_scan(&plan));
            let batches = collect(plan, ctx.task_ctx()).await.unwrap();
            assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
        }
        assert_eq!(provider.metrics.build_count.load(Ordering::Relaxed), 1);
        assert_eq!(provider.metrics.cache_hits.load(Ordering::Relaxed), 1);
        assert_eq!(provider.metrics.take_rows.load(Ordering::Relaxed), 4);
        let plan = provider
            .scan(&ctx.state(), None, &[col("dst").eq(lit("b"))], None)
            .await
            .unwrap();
        assert_eq!(
            collect(plan, ctx.task_ctx())
                .await
                .unwrap()
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>(),
            3
        );
    }

    #[tokio::test]
    async fn null_and_budget_fallback_preserve_native_rows() {
        for nullable in [false, true] {
            let options = CsrScanOptions {
                max_edges: if nullable { 10 } else { 0 },
                ..Default::default()
            };
            let (_temp, provider) = fixture(nullable, options).await;
            let ctx = SessionContext::new();
            let plan = provider
                .scan(&ctx.state(), None, &[col("src").eq(lit("a"))], None)
                .await
                .unwrap();
            let batches = collect(plan, ctx.task_ctx()).await.unwrap();
            assert_eq!(
                batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
                if nullable { 2 } else { 3 }
            );
            assert_eq!(provider.metrics.csr_scans.load(Ordering::Relaxed), 0);
            assert_eq!(provider.metrics.fallbacks.load(Ordering::Relaxed), 1);
        }
    }

    #[tokio::test]
    async fn datafusion_retains_property_residual_and_projection() {
        let (_temp, mut provider) = fixture(false, CsrScanOptions::default()).await;
        // Default namespace providers expose system columns. Ordinary projected
        // properties still qualify; selecting system columns must fall back.
        provider.inner = Arc::new(LanceTableProvider::new(
            provider.inner.dataset(),
            true,
            true,
        ));
        assert_eq!(provider.schema().fields().len(), 5);
        let metrics = provider.metrics.clone();
        let ctx = SessionContext::new();
        ctx.register_table("native", provider.inner.clone())
            .unwrap();
        ctx.register_table("csr", Arc::new(provider)).unwrap();
        let query =
            "SELECT weight FROM {} WHERE src = 'a' AND dst = 'b' AND weight >= 2 ORDER BY weight";
        let native = ctx
            .sql(&query.replace("{}", "native"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let csr = ctx
            .sql(&query.replace("{}", "csr"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(native, csr);
        assert_eq!(csr.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
        assert_eq!(metrics.csr_scans.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.take_rows.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn empty_projection_unknown_keys_and_unbounded_fallback() {
        let (_temp, provider) = fixture(false, CsrScanOptions::default()).await;
        let ctx = SessionContext::new();
        let projection = vec![];
        for (key, expected) in [("a", 3), ("unknown", 0)] {
            let plan = provider
                .scan(
                    &ctx.state(),
                    Some(&projection),
                    &[col("src").eq(lit(key))],
                    None,
                )
                .await
                .unwrap();
            let batches = collect(plan, ctx.task_ctx()).await.unwrap();
            assert!(batches.iter().all(|batch| batch.num_columns() == 0));
            assert_eq!(
                batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
                expected
            );
        }
        assert_eq!(provider.metrics.csr_scans.load(Ordering::Relaxed), 2);
        // Exact filtering must read hidden src before emitting zero columns.
        assert_eq!(provider.metrics.take_rows.load(Ordering::Relaxed), 3);
        let unknown = provider
            .scan(&ctx.state(), None, &[col("src").eq(lit("unknown"))], None)
            .await
            .unwrap();
        assert_eq!(
            collect(unknown, ctx.task_ctx())
                .await
                .unwrap()
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>(),
            0
        );
        assert_eq!(provider.metrics.take_rows.load(Ordering::Relaxed), 3);
        let plan = provider.scan(&ctx.state(), None, &[], None).await.unwrap();
        assert_eq!(plan.name(), "CsrAdjacencyScan");
        assert_eq!(
            collect(plan, ctx.task_ctx())
                .await
                .unwrap()
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>(),
            4
        );
        assert_eq!(provider.metrics.build_count.load(Ordering::Relaxed), 1);
        assert_eq!(provider.metrics.csr_scans.load(Ordering::Relaxed), 3);
        assert_eq!(provider.metrics.fallbacks.load(Ordering::Relaxed), 1);
        assert_eq!(
            provider.metrics.timeout_fallbacks.load(Ordering::Relaxed),
            0
        );
    }

    fn physical_keys(name: &str, index: usize, keys: Vec<Option<&str>>) -> Arc<dyn PhysicalExpr> {
        let mut fields = vec![Field::new("unused", DataType::Utf8, true); index + 1];
        fields[index] = Field::new(name, DataType::Utf8, true);
        Arc::new(
            InListExpr::try_new_from_array(
                Arc::new(Column::new(name, index)),
                Arc::new(StringArray::from(keys)),
                false,
                &Schema::new(fields),
            )
            .unwrap(),
        )
    }

    #[test]
    fn completed_membership_is_bounded_typed_and_exactly_remapped() {
        let keys = physical_keys("src", 1, vec![Some("a"), Some("a"), Some("b")]);
        assert_eq!(
            runtime_membership(&keys, "src", 1, &DataType::Utf8, 4),
            Some(BTreeSet::from([
                Endpoint::Text("a".into()),
                Endpoint::Text("b".into())
            ]))
        );
        for (name, index, max) in [("wrong", 1, 4), ("src", 0, 4), ("src", 1, 1)] {
            assert!(runtime_membership(&keys, name, index, &DataType::Utf8, max).is_none());
        }
        assert!(runtime_membership(
            &physical_keys("src", 0, vec![Some("a"), None]),
            "src",
            0,
            &DataType::Utf8,
            4,
        )
        .is_none());
        let long = "x".repeat(128 * 1024 + 1);
        assert!(runtime_membership(
            &physical_keys("src", 0, vec![Some(&long)]),
            "src",
            0,
            &DataType::Utf8,
            4,
        )
        .is_none());
        let numbers: Arc<dyn PhysicalExpr> = Arc::new(
            InListExpr::try_new_from_array(
                Arc::new(Column::new("src", 0)),
                Arc::new(UInt64Array::from(vec![7, 9])),
                false,
                &Schema::new(vec![Field::new("src", DataType::UInt64, false)]),
            )
            .unwrap(),
        );
        assert_eq!(
            runtime_membership(&numbers, "src", 0, &DataType::UInt64, 4),
            Some(BTreeSet::from([Endpoint::Number(7), Endpoint::Number(9)]))
        );
        assert!(scalar_endpoint(&ScalarValue::Int64(Some(-1)), &DataType::UInt64).is_none());
        assert_eq!(
            scalar_endpoint(&ScalarValue::Int64(Some(-1)), &DataType::Int64),
            Some(Endpoint::Signed(-1))
        );
        assert_eq!(
            scalar_endpoint(&ScalarValue::UInt64(Some(7)), &DataType::Int64),
            Some(Endpoint::Signed(7))
        );
        assert!(scalar_endpoint(&ScalarValue::UInt64(Some(u64::MAX)), &DataType::Int64).is_none());
        assert!(scalar_endpoint(&ScalarValue::Int32(Some(7)), &DataType::Int64).is_none());
        let null: Arc<dyn PhysicalExpr> = Arc::new(Literal::new(ScalarValue::Utf8(None)));
        let disjunction: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(keys.clone(), Operator::Or, null.clone()));
        assert!(runtime_membership(&disjunction, "src", 1, &DataType::Utf8, 4).is_none());
        let conjunction: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(keys, Operator::And, null));
        assert_eq!(
            runtime_membership(&conjunction, "src", 1, &DataType::Utf8, 4)
                .unwrap()
                .len(),
            2
        );
        let empty: Arc<dyn PhysicalExpr> =
            Arc::new(Literal::new(ScalarValue::Boolean(Some(false))));
        assert_eq!(
            runtime_membership(&empty, "src", 0, &DataType::Utf8, 0),
            Some(BTreeSet::new())
        );
    }

    async fn index_source(provider: &mut CsrRelationshipProvider) {
        let mut dataset = provider.inner.dataset().as_ref().clone();
        dataset
            .create_index_builder(&["src"], IndexType::BTree, &ScalarIndexParams::default())
            .name("source_index".into())
            .await
            .unwrap();
        provider.inner = Arc::new(LanceTableProvider::new(Arc::new(dataset), false, false));
    }

    #[tokio::test]
    async fn static_endpoint_btree_retains_native_plan_without_topology() {
        let (_temp, mut provider) = fixture(false, CsrScanOptions::default()).await;
        index_source(&mut provider).await;
        let ctx = SessionContext::new();
        let plan = provider
            .scan(&ctx.state(), None, &[col("src").eq(lit("a"))], None)
            .await
            .unwrap();
        assert_ne!(plan.name(), "CsrAdjacencyScan");
        assert!(endpoint_btree_plan(&plan, &BTreeSet::from(["src".into()])));
        let native = provider
            .inner
            .scan(&ctx.state(), None, &[col("src").eq(lit("a"))], None)
            .await
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                datafusion::physical_plan::displayable(plan.as_ref()).indent(false)
            ),
            format!(
                "{}",
                datafusion::physical_plan::displayable(native.as_ref()).indent(false)
            )
        );
        assert_eq!(
            provider
                .supports_filters_pushdown(&[&col("src").eq(lit("a"))])
                .unwrap(),
            vec![TableProviderFilterPushDown::Exact]
        );
        assert_eq!(
            collect(plan, ctx.task_ctx())
                .await
                .unwrap()
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>(),
            3
        );
        assert_eq!(
            provider.metrics.native_index_plans.load(Ordering::Relaxed),
            1
        );
        assert_eq!(provider.metrics.build_count.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn exact_filters_read_hidden_columns_then_restore_projection_and_row_count() {
        let (_temp, provider) = fixture(false, CsrScanOptions::default()).await;
        let ctx = SessionContext::new();
        let filters = [col("src").eq(lit("a")), col("weight").gt(lit(1u64))];
        assert_eq!(
            provider
                .supports_filters_pushdown(&filters.iter().collect::<Vec<_>>())
                .unwrap(),
            vec![TableProviderFilterPushDown::Exact; 2]
        );
        for projection in [vec![1], vec![2, 1], vec![]] {
            let native = provider
                .inner
                .scan(&ctx.state(), Some(&projection), &filters, None)
                .await
                .unwrap();
            let csr = provider
                .scan(&ctx.state(), Some(&projection), &filters, None)
                .await
                .unwrap();
            assert!(has_csr_scan(&csr));
            assert_eq!(csr.schema(), native.schema());
            let native = collect(native, ctx.task_ctx()).await.unwrap();
            let csr = collect(csr, ctx.task_ctx()).await.unwrap();
            assert_eq!(csr, native);
            assert_eq!(csr.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
            assert!(csr
                .iter()
                .all(|batch| batch.num_columns() == projection.len()));
        }
        let no_rows = [col("src").eq(lit("a")), col("src").is_null()];
        let csr = provider
            .scan(&ctx.state(), Some(&vec![]), &no_rows, None)
            .await
            .unwrap();
        assert_eq!(
            collect(csr, ctx.task_ctx())
                .await
                .unwrap()
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>(),
            0
        );
        assert_eq!(provider.metrics.build_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn exact_projection_rejects_nonlocal_or_transformed_filter_columns() {
        let ctx = SessionContext::new();
        let raw = Schema::new(vec![
            Field::new("src", DataType::Utf8, true),
            Field::new("weight", DataType::UInt64, false),
            Field::new("payload", DataType::Binary, true),
        ]);
        let provider = Arc::new(Schema::new(vec![
            raw.field(0).clone(),
            raw.field(1).clone(),
            raw.field(2).clone(),
            Field::new("_rowid", DataType::UInt64, false),
        ]));
        for expression in [
            col("payload").is_not_null(),
            col("_rowid").eq(lit(1u64)),
            col("other.src").eq(lit("a")),
            col("missing").is_not_null(),
        ] {
            assert!(exact_filter_projection(
                &ctx.state(),
                &provider,
                &raw,
                Some(&vec![1]),
                &[expression]
            )
            .is_none());
        }
    }

    #[tokio::test]
    async fn metadata_cardinality_keeps_join_costs_but_not_exact_prefilter_counts() {
        use datafusion::common::stats::Precision;
        let (_temp, provider) = fixture(false, CsrScanOptions::default()).await;
        let ctx = SessionContext::new();
        let plain = provider.scan(&ctx.state(), None, &[], None).await.unwrap();
        let source = plain.downcast_ref::<CsrScanExec>().unwrap();
        assert_eq!(
            source.partition_statistics(None).unwrap().num_rows,
            Precision::Exact(4)
        );
        // Statistics come from the pinned manifest, not a native scan/CSR build.
        assert_eq!(provider.metrics.build_count.load(Ordering::Relaxed), 0);
        assert_eq!(provider.metrics.take_rows.load(Ordering::Relaxed), 0);
        assert_eq!(
            source.partition_statistics(None).unwrap().total_byte_size,
            Precision::Absent
        );
        let filtered = provider
            .scan(&ctx.state(), None, &[col("src").eq(lit("a"))], None)
            .await
            .unwrap();
        let filtered = filtered.downcast_ref::<FilterExec>().unwrap();
        let filtered = filtered.input().downcast_ref::<CsrScanExec>().unwrap();
        assert_eq!(
            filtered.partition_statistics(None).unwrap().num_rows,
            Precision::Inexact(4)
        );
        let property = provider
            .scan(&ctx.state(), None, &[col("weight").gt_eq(lit(2u64))], None)
            .await
            .unwrap();
        let property = property.downcast_ref::<FilterExec>().unwrap();
        let property = property.input().downcast_ref::<CsrScanExec>().unwrap();
        assert!(property.sources.is_none() && property.targets.is_none());
        assert_eq!(
            property.partition_statistics(None).unwrap().num_rows,
            Precision::Inexact(4)
        );
        let mut runtime = source.clone();
        runtime.runtime_filters.push(RuntimeEndpointFilter {
            dynamic: Arc::new(DynamicFilterPhysicalExpr::new(
                vec![Arc::new(Column::new("src", 0))],
                physical_keys("src", 0, vec![Some("a")]),
            )),
            source: true,
        });
        assert_eq!(
            runtime.partition_statistics(None).unwrap().num_rows,
            Precision::Inexact(4)
        );
    }

    #[tokio::test]
    async fn actual_sql_inner_join_consumes_complete_runtime_bounds() {
        for indexed in [false, true] {
            let (_temp, mut provider) = fixture(false, CsrScanOptions::default()).await;
            if indexed {
                index_source(&mut provider).await;
            }
            let metrics = provider.metrics.clone();
            let ctx = SessionContext::new_with_config(
                SessionConfig::new()
                    .with_target_partitions(1)
                    .with_repartition_joins(false),
            );
            let build_schema =
                Arc::new(Schema::new(vec![Field::new("key", DataType::Utf8, false)]));
            ctx.register_batch(
                "build",
                RecordBatch::try_new(
                    build_schema,
                    vec![Arc::new(StringArray::from(vec!["a", "b"]))],
                )
                .unwrap(),
            )
            .unwrap();
            ctx.register_table("native", provider.inner.clone())
                .unwrap();
            ctx.register_table("csr", Arc::new(provider)).unwrap();
            let query = "SELECT e.weight FROM build b JOIN {} e ON b.key = e.src WHERE e.weight >= 2 ORDER BY e.weight";
            let native = ctx
                .sql(&query.replace("{}", "native"))
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
            let plan = ctx
                .sql(&query.replace("{}", "csr"))
                .await
                .unwrap()
                .create_physical_plan()
                .await
                .unwrap();
            let plan = bind_csr_runtime_sources(plan).unwrap();
            let csr = collect(plan, ctx.task_ctx()).await.unwrap();
            assert_eq!(native, csr);
            assert_eq!(csr.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
            assert!(metrics.runtime_keys.load(Ordering::Relaxed) > 0);
            if indexed {
                assert!(metrics.runtime_native_scans.load(Ordering::Relaxed) > 0);
                assert!(metrics.runtime_native_index_plans.load(Ordering::Relaxed) > 0);
                assert_eq!(metrics.build_count.load(Ordering::Relaxed), 0);
            } else {
                assert!(metrics.runtime_scans.load(Ordering::Relaxed) > 0);
                assert_eq!(metrics.build_count.load(Ordering::Relaxed), 1);
            }
        }
    }

    #[tokio::test]
    async fn completed_runtime_waits_for_all_producers_and_reset_clears_bindings() {
        let (_temp, provider) = fixture(false, CsrScanOptions::default()).await;
        let ctx = SessionContext::new();
        let plan = provider
            .scan(&ctx.state(), Some(&vec![1, 0, 2]), &[], None)
            .await
            .unwrap();
        let mut source = plan.downcast_ref::<CsrScanExec>().unwrap().clone();
        source.validated_runtime = true;
        source.wait = Duration::from_millis(1);
        let completed = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::new(Column::new("src", 1))],
            physical_keys("src", 1, vec![Some("a")]),
        ));
        completed.mark_complete();
        let pending = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::new(Column::new("dst", 0))],
            physical_keys("dst", 0, vec![Some("b")]),
        ));
        source.runtime_filters = vec![
            RuntimeEndpointFilter {
                dynamic: completed.clone(),
                source: true,
            },
            RuntimeEndpointFilter {
                dynamic: pending.clone(),
                source: false,
            },
        ];
        assert_eq!(
            collect(Arc::new(source.clone()), ctx.task_ctx())
                .await
                .unwrap()
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>(),
            4
        );
        assert_eq!(
            provider.metrics.timeout_fallbacks.load(Ordering::Relaxed),
            1
        );
        assert_eq!(provider.metrics.build_count.load(Ordering::Relaxed), 0);
        pending.mark_complete();
        assert_eq!(
            collect(Arc::new(source.clone()), ctx.task_ctx())
                .await
                .unwrap()
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>(),
            2
        );
        assert_eq!(provider.metrics.runtime_scans.load(Ordering::Relaxed), 1);
        let reset = Arc::new(source.clone()).reset_state().unwrap();
        let cleared = reset.downcast_ref::<CsrScanExec>().unwrap();
        assert!(!cleared.validated_runtime && cleared.runtime_filters.is_empty());
        assert_eq!(
            collect(reset, ctx.task_ctx())
                .await
                .unwrap()
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>(),
            4
        );
        let nulls = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::new(Column::new("src", 1))],
            physical_keys("src", 1, vec![Some("a"), None]),
        ));
        nulls.mark_complete();
        source.runtime_filters = vec![RuntimeEndpointFilter {
            dynamic: nulls,
            source: true,
        }];
        assert_eq!(
            collect(Arc::new(source.clone()), ctx.task_ctx())
                .await
                .unwrap()
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>(),
            4
        );
        assert_eq!(
            provider
                .metrics
                .unsupported_runtime_fallbacks
                .load(Ordering::Relaxed),
            1
        );
        // Completed unsupported predicates may be ignored only after waiting
        // for every producer. Preserve a finite necessary bound in either order.
        let unsupported = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::new(Column::new("dst", 0))],
            Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))),
        ));
        unsupported.mark_complete();
        let finite = RuntimeEndpointFilter {
            dynamic: completed,
            source: true,
        };
        let unknown = RuntimeEndpointFilter {
            dynamic: unsupported,
            source: false,
        };
        for filters in [
            vec![finite.clone(), unknown.clone()],
            vec![unknown.clone(), finite],
        ] {
            source.runtime_filters = filters;
            let bounds = source.completed_runtime_bounds().await.unwrap();
            assert_eq!(
                bounds.sources.unwrap(),
                BTreeSet::from([Endpoint::Text("a".into())])
            );
            assert!(bounds.targets.is_none());
        }
        source.runtime_filters = vec![unknown.clone()];
        assert!(source.completed_runtime_bounds().await.is_none());
        let empty = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::new(Column::new("src", 1))],
            Arc::new(Literal::new(ScalarValue::Boolean(Some(false)))),
        ));
        empty.mark_complete();
        source.runtime_filters = vec![
            unknown,
            RuntimeEndpointFilter {
                dynamic: empty,
                source: true,
            },
        ];
        assert!(source
            .completed_runtime_bounds()
            .await
            .unwrap()
            .sources
            .unwrap()
            .is_empty());
        let pending = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::new(Column::new("dst", 0))],
            Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))),
        ));
        source.runtime_filters.push(RuntimeEndpointFilter {
            dynamic: pending,
            source: false,
        });
        assert!(source.completed_runtime_bounds().await.is_none());
    }

    #[tokio::test]
    async fn completed_small_join_bound_survives_completed_large_map_join() {
        let (_temp, provider) = fixture(false, CsrScanOptions::default()).await;
        let mut config = SessionConfig::new()
            .with_target_partitions(1)
            .with_repartition_joins(false);
        config
            .options_mut()
            .optimizer
            .hash_join_inlist_pushdown_max_distinct_values = 1;
        let ctx = SessionContext::new_with_config(config);
        let build_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Utf8, false)]));
        ctx.register_batch(
            "small",
            RecordBatch::try_new(
                build_schema.clone(),
                vec![Arc::new(StringArray::from(vec!["a"]))],
            )
            .unwrap(),
        )
        .unwrap();
        ctx.register_batch(
            "large",
            RecordBatch::try_new(
                build_schema,
                vec![Arc::new(StringArray::from(vec!["b", "c", "unused"]))],
            )
            .unwrap(),
        )
        .unwrap();
        let make = |csr: bool| {
            let ctx = ctx.clone();
            let provider = &provider;
            async move {
                let small = ctx
                    .table("small")
                    .await
                    .unwrap()
                    .create_physical_plan()
                    .await
                    .unwrap();
                let large = ctx
                    .table("large")
                    .await
                    .unwrap()
                    .create_physical_plan()
                    .await
                    .unwrap();
                let source_dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
                    vec![Arc::new(Column::new("src", 0))],
                    Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))),
                ));
                // The outer JOIN sees small.key before the edge columns. The
                // consumer remaps dst to its own source schema but shares the
                // producer identity/state, just like DF post pushdown.
                let target_dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
                    vec![Arc::new(Column::new("dst", 2))],
                    Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))),
                ));
                let source: Arc<dyn ExecutionPlan> = if csr {
                    let plan = provider.scan(&ctx.state(), None, &[], None).await.unwrap();
                    let mut leaf = plan.downcast_ref::<CsrScanExec>().unwrap().clone();
                    let remapped = target_dynamic
                        .clone()
                        .with_new_children(vec![Arc::new(Column::new("dst", 1))])
                        .unwrap();
                    leaf.runtime_filters = vec![
                        RuntimeEndpointFilter {
                            dynamic: source_dynamic.clone(),
                            source: true,
                        },
                        RuntimeEndpointFilter {
                            dynamic: Arc::downcast::<DynamicFilterPhysicalExpr>(remapped).unwrap(),
                            source: false,
                        },
                    ];
                    Arc::new(leaf)
                } else {
                    provider
                        .inner
                        .scan(&ctx.state(), None, &[], None)
                        .await
                        .unwrap()
                };
                let first_on = vec![(
                    Arc::new(Column::new("key", 0)) as Arc<dyn PhysicalExpr>,
                    Arc::new(Column::new("src", 0)) as Arc<dyn PhysicalExpr>,
                )];
                let first: Arc<dyn ExecutionPlan> = Arc::new(
                    HashJoinExecBuilder::new(small, source, first_on, JoinType::Inner)
                        .with_partition_mode(PartitionMode::CollectLeft)
                        .build()
                        .unwrap()
                        .with_dynamic_filter_expr(source_dynamic)
                        .unwrap(),
                );
                let second_on = vec![(
                    Arc::new(Column::new("key", 0)) as Arc<dyn PhysicalExpr>,
                    Arc::new(Column::new("dst", 2)) as Arc<dyn PhysicalExpr>,
                )];
                let joined: Arc<dyn ExecutionPlan> = Arc::new(
                    HashJoinExecBuilder::new(large, first, second_on, JoinType::Inner)
                        .with_partition_mode(PartitionMode::CollectLeft)
                        .build()
                        .unwrap()
                        .with_dynamic_filter_expr(target_dynamic)
                        .unwrap(),
                );
                bind_csr_runtime_sources(joined).unwrap()
            }
        };
        let weights = |batches: Vec<RecordBatch>| {
            let mut weights: Vec<_> = batches
                .iter()
                .flat_map(|batch| {
                    batch
                        .column(4)
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .unwrap()
                        .values()
                        .iter()
                        .copied()
                })
                .collect();
            weights.sort_unstable();
            weights
        };
        let native = weights(collect(make(false).await, ctx.task_ctx()).await.unwrap());
        let csr = weights(collect(make(true).await, ctx.task_ctx()).await.unwrap());
        assert_eq!(native, vec![1, 2, 4]);
        assert_eq!(csr, native);
        assert_eq!(provider.metrics.runtime_scans.load(Ordering::Relaxed), 1);
        assert_eq!(provider.metrics.runtime_keys.load(Ordering::Relaxed), 1);
        assert_eq!(
            provider
                .metrics
                .unsupported_runtime_fallbacks
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(provider.metrics.build_count.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn final_join_proof_rejects_anti_outer_partitioned_and_null_equal() {
        let (_temp, provider) = fixture(false, CsrScanOptions::default()).await;
        let ctx = SessionContext::new();
        let build = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("src", DataType::Utf8, true)])),
            vec![Arc::new(StringArray::from(vec![Some("a")]))],
        )
        .unwrap();
        ctx.register_batch("build", build).unwrap();
        for (join_type, mode, nulls, accepted) in [
            (
                JoinType::Inner,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                true,
            ),
            (
                JoinType::LeftAnti,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            ),
            (
                JoinType::Left,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            ),
            (
                JoinType::Inner,
                PartitionMode::Partitioned,
                NullEquality::NullEqualsNothing,
                false,
            ),
            (
                JoinType::Inner,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNull,
                false,
            ),
        ] {
            let plan = provider.scan(&ctx.state(), None, &[], None).await.unwrap();
            let mut source = plan.downcast_ref::<CsrScanExec>().unwrap().clone();
            let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
                vec![Arc::new(Column::new("src", 0))],
                Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))),
            ));
            source.runtime_filters.push(RuntimeEndpointFilter {
                dynamic: dynamic.clone(),
                source: true,
            });
            let left = ctx
                .table("build")
                .await
                .unwrap()
                .create_physical_plan()
                .await
                .unwrap();
            let on = vec![(
                Arc::new(Column::new("src", 0)) as Arc<dyn PhysicalExpr>,
                Arc::new(Column::new("src", 0)) as Arc<dyn PhysicalExpr>,
            )];
            let joined: Arc<dyn ExecutionPlan> = Arc::new(
                HashJoinExecBuilder::new(left, Arc::new(source), on, join_type)
                    .with_partition_mode(mode)
                    .with_null_equality(nulls)
                    .build()
                    .unwrap()
                    .with_dynamic_filter_expr(dynamic)
                    .unwrap(),
            );
            let joined = bind_csr_runtime_sources(joined).unwrap();
            let leaf = joined.children()[1].downcast_ref::<CsrScanExec>().unwrap();
            assert_eq!(!leaf.runtime_filters.is_empty(), accepted);
        }
    }

    #[tokio::test]
    async fn provenance_guard_preserves_null_aware_anti_red_then_green() {
        let (_temp, mut provider) = fixture(true, CsrScanOptions::default()).await;
        index_source(&mut provider).await;
        let ctx = SessionContext::new();
        ctx.register_batch(
            "build",
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("src", DataType::Utf8, true)])),
                vec![Arc::new(StringArray::from(vec![Some("z")]))],
            )
            .unwrap(),
        )
        .unwrap();
        let make = |unsafe_previous: bool| {
            let ctx = ctx.clone();
            let provider = &provider;
            async move {
                let plan = provider.scan(&ctx.state(), None, &[], None).await.unwrap();
                let mut source = plan.downcast_ref::<CsrScanExec>().unwrap().clone();
                let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
                    vec![Arc::new(Column::new("src", 0))],
                    Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))),
                ));
                source.runtime_filters.push(RuntimeEndpointFilter {
                    dynamic: dynamic.clone(),
                    source: true,
                });
                source.validated_runtime = unsafe_previous;
                let left = ctx
                    .table("build")
                    .await
                    .unwrap()
                    .create_physical_plan()
                    .await
                    .unwrap();
                let on = vec![(
                    Arc::new(Column::new("src", 0)) as Arc<dyn PhysicalExpr>,
                    Arc::new(Column::new("src", 0)) as Arc<dyn PhysicalExpr>,
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
        let red = collect(make(true).await, ctx.task_ctx()).await.unwrap();
        assert_eq!(red.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
        let green = collect(
            bind_csr_runtime_sources(make(false).await).unwrap(),
            ctx.task_ctx(),
        )
        .await
        .unwrap();
        assert_eq!(green.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
        assert!(
            provider
                .metrics
                .rejected_producer_filters
                .load(Ordering::Relaxed)
                > 0
        );
        assert_eq!(provider.metrics.build_count.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn system_projection_and_scan_limit_keep_native_provider() {
        let (_temp, mut provider) = fixture(false, CsrScanOptions::default()).await;
        provider.inner = Arc::new(LanceTableProvider::new(
            provider.inner.dataset(),
            true,
            true,
        ));
        let ctx = SessionContext::new();
        let filters = [col("src").eq(lit("a"))];
        for (projection, limit, expected) in
            [(Some(vec![3, 4]), None, 3), (Some(vec![2]), Some(1), 1)]
        {
            let plan = provider
                .scan(&ctx.state(), projection.as_ref(), &filters, limit)
                .await
                .unwrap();
            assert_ne!(plan.name(), "CsrAdjacencyScan");
            assert_eq!(
                collect(plan, ctx.task_ctx())
                    .await
                    .unwrap()
                    .iter()
                    .map(RecordBatch::num_rows)
                    .sum::<usize>(),
                expected
            );
        }
        assert_eq!(provider.metrics.planned_scans.load(Ordering::Relaxed), 0);
        assert_eq!(provider.metrics.build_count.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn volatile_filters_stay_above_provider() {
        let (_temp, provider) = fixture(false, CsrScanOptions::default()).await;
        let bound = col("src").eq(lit("a"));
        let volatile = datafusion::logical_expr::create_udf(
            "volatile_test",
            vec![],
            DataType::Boolean,
            datafusion::logical_expr::Volatility::Volatile,
            Arc::new(|_| {
                Ok(datafusion::logical_expr::ColumnarValue::Scalar(
                    ScalarValue::Boolean(Some(true)),
                ))
            }),
        )
        .call(vec![]);
        assert_eq!(
            provider
                .supports_filters_pushdown(&[&bound, &volatile])
                .unwrap(),
            vec![
                TableProviderFilterPushDown::Exact,
                TableProviderFilterPushDown::Unsupported
            ]
        );
        let ctx = SessionContext::new();
        assert!(provider
            .scan(&ctx.state(), None, &[volatile], None)
            .await
            .is_err());
    }
}
