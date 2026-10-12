// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! One physical CSR operator for bounded fixed paths. Node/edge predicates are
//! prepared once per alias, before traversal. The original complete WHERE stays
//! above this source. Unsupported shapes and resource guards replay the ordinary
//! path plan before emitting any row; genuine storage failures are not replayed.

mod materialize;
#[cfg(test)]
mod tests;

use crate::{
    ast::{
        CypherQuery as QueryAst, GraphPattern, PropertyValue, ReadingClause, RelationshipDirection,
    },
    case_insensitive::qualify_column,
    csr_provider::{bound_keys, underlying_lance, Endpoint},
    datafusion_planner::{
        analysis::PlanningContext, expression, DataFusionPlanner, GraphPhysicalPlanner,
    },
    logical_plan::LogicalOperator,
    CsrAdjacencyCache, CsrScanMetrics,
};
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::{
    catalog::Session,
    common::{
        tree_node::{Transformed, TreeNode},
        DataFusionError, Result, Statistics,
    },
    datasource::{source_as_provider, DefaultTableSource, MemTable, TableProvider},
    execution::{SessionState, TaskContext},
    logical_expr::{col, Expr, LogicalPlan, LogicalPlanBuilder, TableSource, TableType},
    physical_expr::EquivalenceProperties,
    physical_plan::{
        coalesce_partitions::CoalescePartitionsExec,
        execution_plan::{Boundedness, EmissionType},
        stream::RecordBatchStreamAdapter,
        DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
        PlanProperties, SendableRecordBatchStream,
    },
};
use futures::{stream, TryStreamExt};
use lance::Dataset;
use std::{
    collections::{BTreeSet, HashSet},
    sync::{atomic::Ordering, Arc},
};

#[derive(Clone)]
pub(crate) struct NativeMultihopOptions {
    pub cache: Arc<CsrAdjacencyCache>,
    pub metrics: Arc<CsrScanMetrics>,
    pub predicate: Option<Expr>,
}

pub(crate) fn supports_query(ast: &QueryAst) -> bool {
    if ast.with_clause.is_some()
        || !ast.post_with_reading_clauses.is_empty()
        || ast.post_with_where_clause.is_some()
        || ast.reading_clauses.len() != 1
    {
        return false;
    }
    matches!(&ast.reading_clauses[0], ReadingClause::Match(clause)
        if clause.patterns.len() == 1 && matches!(&clause.patterns[0], GraphPattern::Path(path)
            if (2..=3).contains(&path.segments.len()) && path.segments.iter().all(|s| s.relationship.length.is_none())))
}

#[derive(Debug, Clone)]
pub(super) struct Slice {
    alias: String,
    provider: Arc<dyn TableProvider>,
    key: String,
    filters: Vec<Expr>,
    // Mapping/pattern restrictions have no outer WHERE to replay them.
    mandatory_filters: usize,
}

#[derive(Clone)]
pub(super) struct PathSpec {
    legacy: LogicalPlan,
    slices: Vec<Slice>, // node, edge, node, edge, node, ...
    directions: Vec<RelationshipDirection>,
    edge_dataset: Arc<Dataset>,
    source: String,
    target: String,
    source_keys: BTreeSet<Endpoint>,
    cache: Arc<CsrAdjacencyCache>,
    metrics: Arc<CsrScanMetrics>,
    schema: SchemaRef,
}

impl std::fmt::Debug for PathSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PathSpec")
            .field("hops", &self.directions.len())
            .field("version", &self.edge_dataset.version_id())
            .finish()
    }
}

fn qualified(expr: Expr, alias: &str) -> crate::Result<Expr> {
    Ok(expr
        .transform(|expr| {
            Ok(match expr {
                Expr::Column(mut column) => {
                    column.name = qualify_column(alias, &column.name);
                    Transformed::yes(Expr::Column(column))
                }
                value => Transformed::no(value),
            })
        })?
        .data)
}

fn local_filters(expr: &Expr, alias: &str, result: &mut Vec<Expr>) {
    if let Expr::BinaryExpr(binary) = expr {
        if binary.op == datafusion::logical_expr::Operator::And {
            local_filters(&binary.left, alias, result);
            local_filters(&binary.right, alias, result);
            return;
        }
    }
    let prefix = format!("{}__", alias.to_lowercase());
    let columns = expr.column_refs();
    if !columns.is_empty()
        && total_mask(expr)
        && columns
            .iter()
            .all(|column| column.relation.is_none() && column.name.starts_with(&prefix))
    {
        result.push(expr.clone());
    }
}

// Pre-masks also see rows never touched by the original path. Throwing
// arithmetic/casts/UDFs must remain in the unchanged residual WHERE.
fn total_mask(expr: &Expr) -> bool {
    let mut supported = true;
    let walked = expr.apply(|expr| {
        supported &= match expr {
            Expr::Column(_)
            | Expr::Literal(_, _)
            | Expr::Not(_)
            | Expr::IsNull(_)
            | Expr::IsNotNull(_)
            | Expr::InList(_) => true,
            Expr::BinaryExpr(binary) => matches!(
                binary.op,
                datafusion::logical_expr::Operator::And
                    | datafusion::logical_expr::Operator::Or
                    | datafusion::logical_expr::Operator::Eq
                    | datafusion::logical_expr::Operator::NotEq
                    | datafusion::logical_expr::Operator::Lt
                    | datafusion::logical_expr::Operator::LtEq
                    | datafusion::logical_expr::Operator::Gt
                    | datafusion::logical_expr::Operator::GtEq
            ),
            _ => false,
        };
        Ok(if supported {
            datafusion::common::tree_node::TreeNodeRecursion::Continue
        } else {
            datafusion::common::tree_node::TreeNodeRecursion::Stop
        })
    });
    walked.is_ok() && supported && !expr.is_volatile()
}

fn slice(
    alias: &str,
    provider: Arc<dyn TableProvider>,
    key: &str,
    mapping: Option<Expr>,
    properties: &std::collections::HashMap<String, PropertyValue>,
    predicate: Option<&Expr>,
) -> crate::Result<Option<Slice>> {
    // Known replayable sources only. Unknown one-shot or externally mutable
    // providers must not be consumed before falling back.
    if underlying_lance(&provider).is_none() && provider.downcast_ref::<MemTable>().is_none() {
        return Ok(None);
    }
    let mut filters = Vec::new();
    if let Some(expr) = mapping {
        // Mapping predicates see the same extra candidate rows as WHERE masks.
        // Do not execute a throwing CAST/UDF/arithmetic expression early.
        if !total_mask(&expr) {
            return Ok(None);
        }
        filters.push(qualified(expr, alias)?);
    }
    for (name, value) in properties {
        if !matches!(
            value,
            PropertyValue::String(_)
                | PropertyValue::Integer(_)
                | PropertyValue::Float(_)
                | PropertyValue::Boolean(_)
                | PropertyValue::Null
        ) {
            return Ok(None);
        }
        filters.push(
            col(qualify_column(alias, name)).eq(expression::to_df_value_expr(
                &crate::ast::ValueExpression::Literal(value.clone()),
            )),
        );
    }
    let mandatory_filters = filters.len();
    if let Some(predicate) = predicate {
        local_filters(predicate, alias, &mut filters);
    }
    Ok(Some(Slice {
        alias: alias.to_owned(),
        provider,
        key: key.to_owned(),
        filters,
        mandatory_filters,
    }))
}

pub(crate) fn try_source(
    planner: &DataFusionPlanner,
    ctx: &mut PlanningContext<'_>,
    op: &LogicalOperator,
) -> crate::Result<Option<Arc<dyn TableSource>>> {
    let Some(options) = &planner.native_multihop else {
        return Ok(None);
    };
    if options.predicate.as_ref().is_some_and(Expr::is_volatile) {
        return Ok(None);
    }
    let mut chain = Vec::new();
    let mut root = op;
    while let LogicalOperator::Expand { input, .. } = root {
        chain.push(root);
        root = input;
    }
    if !(2..=3).contains(&chain.len()) {
        return Ok(None);
    }
    chain.reverse();
    let LogicalOperator::ScanByLabel {
        variable,
        label,
        properties,
    } = root
    else {
        return Ok(None);
    };
    let Some(catalog) = &planner.catalog else {
        return Ok(None);
    };
    let Some(node_map) = planner.config.get_node_mapping(label) else {
        return Ok(None);
    };
    let Some(node_source) = catalog.node_source(label) else {
        return Ok(None);
    };
    let Ok(provider) = source_as_provider(&node_source) else {
        return Ok(None);
    };
    let Some(first) = slice(
        variable,
        provider,
        &node_map.id_field,
        planner.mapping_filter(node_map.filter_conditions.as_deref())?,
        properties,
        options.predicate.as_ref(),
    )?
    else {
        return Ok(None);
    };
    let Some(datatype) = first
        .provider
        .schema()
        .field_with_name(&first.key)
        .ok()
        .map(|field| field.data_type().clone())
    else {
        return Ok(None);
    };
    let root_predicate = first.filters.iter().cloned().reduce(Expr::and);
    let Some(source_keys) = root_predicate.as_ref().and_then(|predicate| {
        bound_keys(
            predicate,
            &qualify_column(variable, &first.key),
            &datatype,
            options.cache.options().max_bound_keys,
        )
    }) else {
        return Ok(None);
    };
    let mut slices = vec![first];
    let mut directions = Vec::new();
    let mut aliases = HashSet::from([variable.to_lowercase()]);
    if variable.contains("__") {
        return Ok(None);
    }
    let mut edge_dataset: Option<Arc<Dataset>> = None;
    let mut endpoints: Option<(String, String)> = None;
    let mut previous = variable.clone();
    for expand in &chain {
        let LogicalOperator::Expand {
            source_variable,
            target_variable,
            target_label,
            relationship_types,
            direction,
            relationship_variable,
            properties,
            target_properties,
            ..
        } = expand
        else {
            unreachable!()
        };
        let Some(alias) = relationship_variable else {
            return Ok(None);
        };
        if source_variable != &previous
            || relationship_types.len() != 1
            || matches!(direction, RelationshipDirection::Undirected)
            || alias.contains("__")
            || target_variable.contains("__")
            || !aliases.insert(alias.to_lowercase())
            || !aliases.insert(target_variable.to_lowercase())
        {
            return Ok(None);
        }
        let Some(mapping) = planner
            .config
            .get_relationship_mapping(&relationship_types[0])
        else {
            return Ok(None);
        };
        let Some(source) = catalog.relationship_source(&mapping.relationship_type) else {
            return Ok(None);
        };
        let Ok(provider) = source_as_provider(&source) else {
            return Ok(None);
        };
        let Some(raw) = underlying_lance(&provider) else {
            return Ok(None);
        };
        let dataset = raw.dataset();
        if [
            mapping.source_id_field.as_str(),
            mapping.target_id_field.as_str(),
        ]
        .iter()
        .any(|field| {
            raw.schema()
                .field_with_name(field)
                .map_or(true, |value| value.data_type() != &datatype)
        }) {
            return Ok(None);
        }
        if let Some(original) = &edge_dataset {
            if !Arc::ptr_eq(original, &dataset) {
                return Ok(None);
            }
        } else {
            edge_dataset = Some(dataset);
        }
        let fields = (
            mapping.source_id_field.clone(),
            mapping.target_id_field.clone(),
        );
        if endpoints.as_ref().is_some_and(|old| old != &fields) {
            return Ok(None);
        }
        endpoints = Some(fields);
        let Some(edge) = slice(
            alias,
            raw,
            &mapping.source_id_field,
            planner.relationship_mapping_filter(mapping)?,
            properties,
            options.predicate.as_ref(),
        )?
        else {
            return Ok(None);
        };
        let Some(node_map) = planner.config.get_node_mapping(target_label) else {
            return Ok(None);
        };
        let Some(source) = catalog.node_source(target_label) else {
            return Ok(None);
        };
        let Ok(provider) = source_as_provider(&source) else {
            return Ok(None);
        };
        let Some(node) = slice(
            target_variable,
            provider,
            &node_map.id_field,
            planner.mapping_filter(node_map.filter_conditions.as_deref())?,
            target_properties,
            options.predicate.as_ref(),
        )?
        else {
            return Ok(None);
        };
        if node
            .provider
            .schema()
            .field_with_name(&node.key)
            .map_or(true, |field| field.data_type() != &datatype)
        {
            return Ok(None);
        }
        slices.extend([edge, node]);
        directions.push(direction.clone());
        previous = target_variable.clone();
    }
    // Cheap logical fallback construction also supplies the exact old Arrow
    // qualifier/type/nullability contract; normal execution has no hop JOINs.
    let legacy =
        DataFusionPlanner::with_catalog(planner.config.clone(), catalog.clone()).plan(op)?;
    let schema = legacy.schema().inner().clone();
    let (source, target) = endpoints.unwrap();
    for expand in chain {
        if let LogicalOperator::Expand {
            relationship_types, ..
        } = expand
        {
            ctx.next_relationship_instance(&relationship_types[0])?;
        }
    }
    options
        .metrics
        .native_multihop_planned
        .fetch_add(1, Ordering::Relaxed);
    Ok(Some(Arc::new(DefaultTableSource::new(Arc::new(
        NativePathProvider(Arc::new(PathSpec {
            legacy,
            slices,
            directions,
            edge_dataset: edge_dataset.unwrap(),
            source,
            target,
            source_keys,
            cache: options.cache.clone(),
            metrics: options.metrics.clone(),
            schema,
        })),
    )))))
}

#[derive(Debug)]
struct NativePathProvider(Arc<PathSpec>);

#[async_trait]
impl TableProvider for NativePathProvider {
    fn schema(&self) -> SchemaRef {
        self.0.schema.clone()
    }
    fn table_type(&self) -> TableType {
        TableType::Base
    }
    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _: &[Expr],
        _: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let session = state
            .as_any()
            .downcast_ref::<SessionState>()
            .ok_or_else(|| {
                DataFusionError::Plan("Native CSR multi-hop requires SessionState".into())
            })?
            .clone();
        let indices = projection
            .cloned()
            .unwrap_or_else(|| (0..self.0.schema.fields().len()).collect());
        let schema = Arc::new(self.0.schema.project(&indices)?);
        Ok(Arc::new(NativePathExec {
            spec: self.0.clone(),
            session,
            indices,
            properties: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(schema),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Final,
                Boundedness::Bounded,
            )),
        }))
    }
}

#[derive(Debug)]
struct NativePathExec {
    spec: Arc<PathSpec>,
    session: SessionState,
    indices: Vec<usize>,
    properties: Arc<PlanProperties>,
}

impl DisplayAs for NativePathExec {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NativeCsrMultiHop: hops={}, version={}, seeds={}",
            self.spec.directions.len(),
            self.spec.edge_dataset.version_id(),
            self.spec.source_keys.len()
        )
    }
}

impl ExecutionPlan for NativePathExec {
    fn name(&self) -> &'static str {
        "NativeCsrMultiHop"
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
                "Native CSR multi-hop has no children".into(),
            ));
        }
        Ok(self)
    }
    fn partition_statistics(&self, _: Option<usize>) -> Result<Arc<Statistics>> {
        Ok(Arc::new(Statistics::new_unknown(self.schema().as_ref())))
    }
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(
                "Native CSR multi-hop supports partition 0 only".into(),
            ));
        }
        let spec = self.spec.clone();
        let state = self.session.clone();
        let indices = self.indices.clone();
        let schema = self.schema();
        let output_schema = schema.clone();
        let selected = stream::once(async move {
            if let Some(batch) = materialize::execute(&spec, &state, &indices, &schema).await? {
                spec.metrics
                    .native_multihop_executions
                    .fetch_add(1, Ordering::Relaxed);
                spec.metrics
                    .native_multihop_rows
                    .fetch_add(batch.num_rows() as u64, Ordering::Relaxed);
                Ok(Box::pin(RecordBatchStreamAdapter::new(
                    schema,
                    stream::once(async move { Ok(batch) }),
                )) as SendableRecordBatchStream)
            } else {
                spec.metrics
                    .native_multihop_fallbacks
                    .fetch_add(1, Ordering::Relaxed);
                let columns = indices
                    .iter()
                    .map(|index| col(spec.schema.field(*index).name()))
                    .collect::<Vec<_>>();
                let fallback = LogicalPlanBuilder::from(spec.legacy.clone())
                    .project(columns)?
                    .build()?;
                let plan = state.create_physical_plan(&fallback).await?;
                let plan = if plan.output_partitioning().partition_count() == 1 {
                    plan
                } else {
                    Arc::new(CoalescePartitionsExec::new(plan)) as Arc<dyn ExecutionPlan>
                };
                if plan.schema() != schema {
                    return Err(DataFusionError::Internal(
                        "Native CSR fallback schema drift".into(),
                    ));
                }
                plan.execute(0, context)
            }
        })
        .try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            output_schema,
            selected,
        )))
    }
}
