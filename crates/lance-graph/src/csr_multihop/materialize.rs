// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{PathSpec, Slice};
use crate::{
    case_insensitive::qualify_column,
    csr_provider::{
        array_endpoint, endpoint_btree_plan, underlying_lance, AdjacencySnapshot, CsrScanMetrics,
        Endpoint,
    },
    csr_walk::{csr_walk_views_filtered, AdjacencyView, CsrWalkOptions},
};
use arrow_array::{
    Array, ArrayRef, BooleanArray, LargeStringArray, RecordBatch, RecordBatchOptions, StringArray,
    UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Schema, SchemaRef};
use datafusion::{
    common::{
        tree_node::{Transformed, TreeNode},
        DFSchema, Result,
    },
    execution::SessionState,
    logical_expr::{col, lit, Expr},
};
use futures::TryStreamExt;
use lance::dataset::ProjectionRequest;
use std::{
    collections::{BTreeSet, HashMap},
    sync::{atomic::Ordering, Arc},
};

type NodeRows = HashMap<Endpoint, Vec<u32>>;

// Logical comparisons can acquire throwing casts during DataFusion coercion
// (for example UTF8 > numeric). Candidate masks must not evaluate such casts
// on a different row domain than the retained JOIN plan. Validate the typed
// expression before touching topology or candidate data, not just its AST.
pub(super) fn typed_mask_is_total(slice: &Slice, state: &SessionState) -> bool {
    use datafusion::physical_expr::expressions::{CastExpr, Column, Literal};
    let Some(predicate) = slice.filters.iter().cloned().reduce(Expr::and) else {
        return true;
    };
    let schema = Schema::new(
        slice
            .provider
            .schema()
            .fields()
            .iter()
            .map(|field| {
                Arc::new(
                    field
                        .as_ref()
                        .clone()
                        .with_name(qualify_column(&slice.alias, field.name())),
                )
            })
            .collect::<Vec<_>>(),
    );
    let Ok(df_schema) = DFSchema::try_from(schema.clone()) else {
        return false;
    };
    let Ok(predicate) = state.create_physical_expr(predicate, &df_schema) else {
        return false;
    };
    let mut pending = vec![predicate];
    while let Some(expr) = pending.pop() {
        if let Some(cast) = expr.downcast_ref::<CastExpr>() {
            // A coerced literal is safe only if conversion itself succeeds.
            if cast.expr().downcast_ref::<Literal>().is_some() {
                if expr
                    .evaluate(&RecordBatch::new_empty(Arc::new(schema.clone())))
                    .is_err()
                {
                    return false;
                }
            } else {
                let Ok(source) = cast.expr().data_type(&schema) else {
                    return false;
                };
                if !total_widening(&source, cast.cast_type()) {
                    return false;
                }
            }
        }
        // The logical admission gate already rejects functions/arithmetic.
        // Coercion must not introduce any other row-dependent transformation.
        if expr.downcast_ref::<Column>().is_none() {
            pending.extend(expr.children().into_iter().cloned());
        }
    }
    true
}

fn total_widening(source: &DataType, target: &DataType) -> bool {
    use DataType::*;
    source == target
        || matches!(
            (source, target),
            (Int8, Int16 | Int32 | Int64 | Float32 | Float64)
                | (Int16, Int32 | Int64 | Float32 | Float64)
                | (Int32, Int64 | Float64)
                | (Int64, Float64)
                | (
                    UInt8,
                    UInt16 | UInt32 | UInt64 | Int16 | Int32 | Int64 | Float32 | Float64
                )
                | (UInt16, UInt32 | UInt64 | Int32 | Int64 | Float32 | Float64)
                | (UInt32, UInt64 | Int64 | Float64)
                | (UInt64, Float64)
                | (Float32, Float64)
        )
}

fn raw_name<'a>(name: &'a str, alias: &str) -> Option<&'a str> {
    name.strip_prefix(&format!("{}__", alias.to_lowercase()))
}

fn required(spec: &PathSpec, projection: &[usize]) -> Option<Vec<BTreeSet<String>>> {
    let mut names = spec
        .slices
        .iter()
        .map(|slice| BTreeSet::from([slice.key.clone()]))
        .collect::<Vec<_>>();
    for (slot, slice) in spec.slices.iter().enumerate() {
        for filter in &slice.filters {
            for column in filter.column_refs() {
                names[slot].insert(raw_name(&column.name, &slice.alias)?.to_owned());
            }
        }
        for index in projection {
            if let Some(name) = raw_name(spec.schema.field(*index).name(), &slice.alias) {
                names[slot].insert(name.to_owned());
            }
        }
    }
    Some(names)
}

// A physical table can bind several aliases. Push the UNION of each alias's
// necessary local predicates, never their intersection: restricting Entity to
// the seed alias, for example, would remove unconstrained neighbor entities.
// Alias-specific masks and the complete residual WHERE still run afterwards.
fn union_filter<'a>(
    slices: impl Iterator<Item = (&'a Slice, &'a BTreeSet<Endpoint>)>,
) -> Result<Vec<Expr>> {
    let slices = slices.collect::<Vec<_>>();
    let mut predicates = Vec::new();
    for (slice, candidates) in &slices {
        let predicate = slice
            .filters
            .iter()
            .cloned()
            .reduce(Expr::and)
            .unwrap_or_else(|| lit(true));
        let predicate = predicate
            .transform(|expr| {
                Ok(match expr {
                    Expr::Column(mut column) => {
                        column.name = raw_name(&column.name, &slice.alias)
                            .expect("admitted single-alias predicate")
                            .to_owned();
                        Transformed::yes(Expr::Column(column))
                    }
                    value => Transformed::no(value),
                })
            })?
            .data;
        predicates.push(predicate.and(endpoint_list(&slice.key, candidates)));
    }
    let Some(union) = predicates.into_iter().reduce(Expr::or) else {
        return Ok(Vec::new());
    };
    // Several aliases of one table give `(k IN A AND p) OR (k IN B AND q)`,
    // which a scalar index cannot answer. The redundant necessary conjunct
    // `k IN A∪B` lets Lance use a node-ID BTree and refine with the OR.
    let key = &slices[0].0.key;
    if slices.len() > 1 && slices.iter().all(|(slice, _)| &slice.key == key) {
        let all = slices
            .iter()
            .flat_map(|(_, candidates)| candidates.iter().cloned())
            .collect();
        return Ok(vec![endpoint_list(key, &all).and(union)]);
    }
    Ok(vec![union])
}

fn endpoint_list(key: &str, keys: &BTreeSet<Endpoint>) -> Expr {
    col(key).in_list(
        keys.iter()
            .map(|key| match key {
                Endpoint::Text(value) => lit(value.clone()),
                Endpoint::Number(value) => lit(*value),
            })
            .collect(),
        false,
    )
}

// A necessary reachable superset, not the final result. Unique per-level
// frontiers avoid repeated I/O without changing cyclic or parallel path bags.
// Unsupported high-degree work falls back before any output is emitted.
fn candidates(
    spec: &PathSpec,
    snapshot: &AdjacencySnapshot,
) -> Option<(Vec<BTreeSet<Endpoint>>, BTreeSet<u64>)> {
    let limits = spec.cache.options();
    let mut vertices = vec![spec.source_keys.clone()];
    let mut frontier = spec
        .source_keys
        .iter()
        .filter_map(|key| snapshot.vertex(key))
        .collect::<BTreeSet<_>>();
    let mut rows = BTreeSet::new();
    let mut examined = 0usize;
    for direction in &spec.directions {
        let index = match direction {
            crate::ast::RelationshipDirection::Outgoing => snapshot.view(true),
            crate::ast::RelationshipDirection::Incoming => snapshot.view(false),
            crate::ast::RelationshipDirection::Undirected => return None,
        };
        let mut next = BTreeSet::new();
        for source in frontier {
            for (edge, target) in index.neighbor_edges(source) {
                examined = examined.checked_add(1)?;
                rows.insert(edge);
                next.insert(target);
                if examined > limits.max_edges
                    || rows.len() > limits.max_take_rows.saturating_mul(spec.slices.len())
                    || next.len() > limits.max_take_rows
                {
                    return None;
                }
            }
        }
        vertices.push(
            next.iter()
                .map(|id| snapshot.endpoint(*id).cloned())
                .collect::<Option<BTreeSet<_>>>()?,
        );
        frontier = next;
    }
    Some((vertices, rows))
}

async fn edge_rows(
    spec: &PathSpec,
    rows: &BTreeSet<u64>,
    columns: &BTreeSet<String>,
) -> Result<Option<RecordBatch>> {
    let requested = columns
        .iter()
        .filter(|name| name.as_str() != "_rowid")
        .collect::<Vec<_>>();
    let projection = spec
        .edge_dataset
        .schema()
        .project_preserve_system_columns(requested.as_slice())
        .map_err(|error| datafusion::common::DataFusionError::External(Box::new(error)))?;
    let rows = rows.iter().copied().collect::<Vec<_>>();
    let batch = spec
        .edge_dataset
        .take_rows(&rows, ProjectionRequest::Schema(Arc::new(projection)))
        .await
        .map_err(|error| datafusion::common::DataFusionError::External(Box::new(error)))?;
    if batch.num_rows() != rows.len() {
        return Ok(None);
    }
    let mut fields = batch.schema().fields().to_vec();
    fields.push(Arc::new(arrow_schema::Field::new(
        "_rowid",
        DataType::UInt64,
        false,
    )));
    let mut arrays = batch.columns().to_vec();
    arrays.push(Arc::new(UInt64Array::from(rows)));
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
    spec.metrics
        .native_multihop_input_rows
        .fetch_add(batch.num_rows() as u64, Ordering::Relaxed);
    spec.metrics
        .native_multihop_input_bytes
        .fetch_add(batch.get_array_memory_size() as u64, Ordering::Relaxed);
    if batch.get_array_memory_size() > spec.cache.options().max_bytes {
        return Ok(None);
    }
    Ok(Some(batch))
}

async fn read(
    provider: &Arc<dyn datafusion::datasource::TableProvider>,
    state: &SessionState,
    columns: &BTreeSet<String>,
    filters: &[Expr],
    id_fields: &BTreeSet<String>,
    metrics: &CsrScanMetrics,
    max_rows: usize,
    max_bytes: usize,
) -> Result<Option<RecordBatch>> {
    let schema = provider.schema();
    let Some(indices) = columns
        .iter()
        .map(|name| schema.index_of(name).ok())
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    let plan = provider.scan(state, Some(&indices), filters, None).await?;
    let indexed = endpoint_btree_plan(&plan, id_fields);
    let projected = plan.schema();
    let mut batches = Vec::new();
    let mut rows = 0usize;
    let mut bytes = 0usize;
    let mut stream = datafusion::physical_plan::execute_stream(plan, state.task_ctx())?;
    while let Some(batch) = stream.try_next().await? {
        metrics
            .native_multihop_input_rows
            .fetch_add(batch.num_rows() as u64, Ordering::Relaxed);
        metrics
            .native_multihop_input_bytes
            .fetch_add(batch.get_array_memory_size() as u64, Ordering::Relaxed);
        let Some(next_rows) = rows.checked_add(batch.num_rows()) else {
            return Ok(None);
        };
        let Some(next_bytes) = bytes.checked_add(batch.get_array_memory_size()) else {
            return Ok(None);
        };
        if next_rows > max_rows || next_rows > u32::MAX as usize || next_bytes > max_bytes {
            return Ok(None);
        }
        rows = next_rows;
        bytes = next_bytes;
        batches.push(batch);
    }
    let batch = arrow::compute::concat_batches(&projected, &batches)?;
    if indexed {
        metrics
            .native_multihop_node_index_reads
            .fetch_add(1, Ordering::Relaxed);
    }
    Ok(Some(batch))
}

enum NodeRead {
    Rows(RecordBatch),
    /// Not applicable here; the filtered provider scan remains.
    Scan,
    /// A resource guard was hit; replay the legacy plan.
    Reject,
}

// Reachable node IDs are known exactly, so a pinned ID -> row-ID map turns the
// O(table) filtered scan into a take of just those rows. All rows of every
// candidate ID are taken; per-alias masks are evaluated afterwards exactly as
// for scanned rows. The take is preferred even over a node-ID BTree (measured
// ~1.5 ms faster per query); the BTree still serves the scan when the map does
// not fit the cache budget.
async fn take_nodes(
    spec: &PathSpec,
    provider: &Arc<dyn datafusion::datasource::TableProvider>,
    columns: &BTreeSet<String>,
    slots: &[usize],
    candidates: &[BTreeSet<Endpoint>],
    max_rows: usize,
    max_bytes: usize,
) -> Result<NodeRead> {
    let Some(lance) = underlying_lance(provider) else {
        return Ok(NodeRead::Scan);
    };
    let key = &spec.slices[slots[0]].key;
    if slots.iter().any(|slot| &spec.slices[*slot].key != key) {
        return Ok(NodeRead::Scan);
    }
    let dataset = lance.dataset();
    // Only plain stored columns: system or provider-converted fields keep the
    // scan so the Arrow contract stays identical.
    let provider_schema = provider.schema();
    let dataset_schema = Schema::from(dataset.schema());
    if columns.iter().any(|name| {
        name.starts_with('_')
            || provider_schema.field_with_name(name).ok()
                != dataset_schema.field_with_name(name).ok()
    }) {
        return Ok(NodeRead::Scan);
    }
    let Some(keys) = spec
        .cache
        .node_keys(dataset.clone(), key, &spec.metrics)
        .await
    else {
        return Ok(NodeRead::Scan);
    };
    let mut rows = Vec::new();
    for slot in slots {
        for id in &candidates[*slot / 2] {
            rows.extend(keys.rows(id));
            if rows.len() > max_rows.saturating_mul(slots.len()) {
                return Ok(NodeRead::Reject);
            }
        }
    }
    rows.sort_unstable();
    rows.dedup();
    if rows.len() > max_rows || rows.len() > u32::MAX as usize {
        return Ok(NodeRead::Reject);
    }
    let names = columns.iter().map(String::as_str).collect::<Vec<_>>();
    let projection = dataset
        .schema()
        .project(&names)
        .map_err(|error| datafusion::common::DataFusionError::External(Box::new(error)))?;
    let batch = if rows.is_empty() {
        RecordBatch::new_empty(Arc::new(Schema::from(&projection)))
    } else {
        dataset
            .take_rows(&rows, ProjectionRequest::Schema(Arc::new(projection)))
            .await
            .map_err(|error| datafusion::common::DataFusionError::External(Box::new(error)))?
    };
    spec.metrics
        .native_multihop_input_rows
        .fetch_add(batch.num_rows() as u64, Ordering::Relaxed);
    spec.metrics
        .native_multihop_input_bytes
        .fetch_add(batch.get_array_memory_size() as u64, Ordering::Relaxed);
    if batch.num_rows() != rows.len() || batch.get_array_memory_size() > max_bytes {
        return Ok(NodeRead::Reject);
    }
    spec.metrics
        .native_multihop_node_take_reads
        .fetch_add(1, Ordering::Relaxed);
    Ok(NodeRead::Rows(batch))
}

fn mask(slice: &Slice, batch: &RecordBatch, state: &SessionState) -> Result<Vec<bool>> {
    let fields = batch
        .schema()
        .fields()
        .iter()
        .map(|field| {
            Arc::new(
                field
                    .as_ref()
                    .clone()
                    .with_name(qualify_column(&slice.alias, field.name())),
            )
        })
        .collect::<Vec<_>>();
    let qualified = RecordBatch::try_new(Arc::new(Schema::new(fields)), batch.columns().to_vec())?;
    let Some(predicate) = slice.filters.iter().cloned().reduce(Expr::and) else {
        return Ok(vec![true; batch.num_rows()]);
    };
    let schema = DFSchema::try_from(qualified.schema().as_ref().clone())?;
    let predicate = state.create_physical_expr(predicate, &schema)?;
    let result = predicate
        .evaluate(&qualified)?
        .into_array(batch.num_rows())?;
    let values = result
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| {
            datafusion::common::DataFusionError::Execution("Native CSR mask must be boolean".into())
        })?;
    // SQL WHERE admits only TRUE; false and NULL both reject this physical row.
    Ok((0..batch.num_rows())
        .map(|row| !values.is_null(row) && values.value(row))
        .collect())
}

fn node_rows(slice: &Slice, batch: &RecordBatch, mask: &[bool]) -> Option<NodeRows> {
    let ids = batch.column_by_name(&slice.key)?;
    let mut result = NodeRows::new();
    for (row, accepted) in mask.iter().copied().enumerate() {
        if accepted {
            if let Some(id) = array_endpoint(ids.as_ref(), row) {
                result.entry(id).or_default().push(u32::try_from(row).ok()?);
            }
        }
    }
    Some(result)
}

fn max_value_bytes(array: &ArrayRef) -> Option<usize> {
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        Some(
            values
                .iter()
                .flatten()
                .map(str::len)
                .max()
                .unwrap_or(0)
                .checked_add(8)?,
        )
    } else if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        Some(
            values
                .iter()
                .flatten()
                .map(str::len)
                .max()
                .unwrap_or(0)
                .checked_add(16)?,
        )
    } else if matches!(array.data_type(), DataType::Boolean) {
        Some(2)
    } else {
        array.data_type().primitive_width()?.checked_add(1)
    }
}

/// All data is prepared from the bound catalog and one exact edge snapshot.
/// None means an optional guard rejected the candidate before any output; the
/// caller executes the retained ordinary plan. Storage/expression errors stay
/// errors rather than replaying a partly failed read.
pub(super) async fn execute(
    spec: &PathSpec,
    state: &SessionState,
    projection: &[usize],
    output_schema: &SchemaRef,
) -> Result<Option<RecordBatch>> {
    // Coercion may make an otherwise simple WHERE expression throwing. It is
    // only an optional early mask: keep it in the unchanged residual WHERE,
    // not on the reachable superset. Mapping/pattern filters are mandatory;
    // if one is unsafe, replay the legacy plan before reading candidates.
    let mut admitted = spec.clone();
    for slice in &mut admitted.slices {
        let mut filters = Vec::with_capacity(slice.filters.len());
        for (index, predicate) in slice.filters.iter().enumerate() {
            let mask = Slice {
                filters: vec![predicate.clone()],
                ..slice.clone()
            };
            if typed_mask_is_total(&mask, state) {
                filters.push(predicate.clone());
            } else if index < slice.mandatory_filters {
                return Ok(None);
            }
        }
        slice.filters = filters;
    }
    let spec = &admitted;
    let options = spec.cache.options();
    let Some(requirements) = required(spec, projection) else {
        return Ok(None);
    };
    let Some(snapshot) = spec
        .cache
        .snapshot(
            spec.edge_dataset.clone(),
            &spec.source,
            &spec.target,
            &spec.metrics,
        )
        .await
    else {
        return Ok(None);
    };
    let Some((node_candidates, reachable_edges)) = candidates(spec, &snapshot) else {
        return Ok(None);
    };
    let mut columns = BTreeSet::from(["_rowid".to_owned()]);
    for slot in (1..requirements.len()).step_by(2) {
        columns.extend(requirements[slot].iter().cloned());
    }
    // Query-local raw node batches may be reused, but masks and alias bindings
    // are always recomputed. No identity/ACL/parameter mask enters the CSR cache.
    let mut node_groups: HashMap<
        usize,
        (
            Arc<dyn datafusion::datasource::TableProvider>,
            BTreeSet<String>,
            Vec<usize>,
        ),
    > = HashMap::new();
    for slot in (0..spec.slices.len()).step_by(2) {
        let provider = &spec.slices[slot].provider;
        let key = Arc::as_ptr(provider) as *const () as usize;
        let group = node_groups
            .entry(key)
            .or_insert_with(|| (provider.clone(), BTreeSet::new(), Vec::new()));
        group.1.extend(requirements[slot].iter().cloned());
        group.2.push(slot);
    }
    // Candidates come from the snapshot alone, so the edge read and every
    // node read are independent: issue them concurrently. Each read keeps the
    // full byte guard and their sum is checked before any mask or output.
    let node_candidates = &node_candidates;
    let node_reads = futures::future::try_join_all(node_groups.into_iter().map(
        |(key, (provider, columns, slots))| async move {
            let batch = match take_nodes(
                spec,
                &provider,
                &columns,
                &slots,
                node_candidates,
                options.max_vertices,
                options.max_bytes,
            )
            .await?
            {
                NodeRead::Rows(batch) => Some(batch),
                NodeRead::Reject => None,
                NodeRead::Scan => {
                    read(
                        &provider,
                        state,
                        &columns,
                        &union_filter(
                            slots
                                .iter()
                                .map(|slot| (&spec.slices[*slot], &node_candidates[*slot / 2])),
                        )?,
                        &slots
                            .iter()
                            .map(|slot| spec.slices[*slot].key.clone())
                            .collect(),
                        &spec.metrics,
                        options.max_vertices,
                        options.max_bytes,
                    )
                    .await?
                }
            };
            Ok::<_, datafusion::common::DataFusionError>(batch.map(|batch| (key, batch)))
        },
    ));
    let (edge_batch, node_reads) =
        futures::future::try_join(edge_rows(spec, &reachable_edges, &columns), node_reads).await?;
    let Some(edge_batch) = edge_batch else {
        return Ok(None);
    };
    let Some(ids) = edge_batch
        .column_by_name("_rowid")
        .and_then(|array| array.as_any().downcast_ref::<UInt64Array>())
    else {
        return Ok(None);
    };
    let mut edge_positions = HashMap::new();
    for row in 0..edge_batch.num_rows() {
        if ids.is_null(row) || edge_positions.insert(ids.value(row), row as u32).is_some() {
            return Ok(None);
        }
    }
    let mut node_batches = HashMap::new();
    let mut input_bytes = edge_batch.get_array_memory_size();
    for read in node_reads {
        let Some((key, batch)) = read else {
            return Ok(None);
        };
        input_bytes = input_bytes.saturating_add(batch.get_array_memory_size());
        if input_bytes > options.max_bytes {
            return Ok(None);
        }
        node_batches.insert(key, Arc::new(batch));
    }
    let edge_batch = Arc::new(edge_batch);
    let mut batches = Vec::new();
    let mut node_maps = Vec::new();
    let mut edge_masks = Vec::new();
    for (slot, slice) in spec.slices.iter().enumerate() {
        let batch = if slot % 2 == 1 {
            edge_batch.clone()
        } else {
            node_batches[&(Arc::as_ptr(&slice.provider) as *const () as usize)].clone()
        };
        let accepted = mask(slice, &batch, state)?;
        if slot % 2 == 0 {
            let Some(rows) = node_rows(slice, &batch, &accepted) else {
                return Ok(None);
            };
            node_maps.push(rows);
        } else {
            edge_masks.push(accepted);
        }
        batches.push(batch);
    }
    let indices = spec
        .directions
        .iter()
        .map(|direction| match direction {
            crate::ast::RelationshipDirection::Outgoing => snapshot.view(true),
            crate::ast::RelationshipDirection::Incoming => snapshot.view(false),
            crate::ast::RelationshipDirection::Undirected => {
                unreachable!("admission rejects undirected")
            }
        })
        .collect::<Vec<_>>();
    let indices = indices.iter().collect::<Vec<_>>();
    let hops = indices.len() as u32;
    let mut tuples: Vec<Vec<u32>> = Vec::new();
    let mut examined = 0usize;
    for key in &spec.source_keys {
        let Some(root_rows) = node_maps[0].get(key) else {
            continue;
        };
        let Some(start) = snapshot.vertex(key) else {
            continue;
        };
        let remaining = options.max_edges.saturating_sub(examined);
        if remaining == 0 {
            return Ok(None);
        }
        let walks = match csr_walk_views_filtered(
            &indices,
            start,
            CsrWalkOptions {
                min_hops: hops,
                max_hops: hops,
                max_frontier_states: options.max_take_rows,
                max_paths: options.max_take_rows,
                max_examined_edges: remaining,
            },
            |hop, edge, _, target| {
                let slot = hop as usize - 1;
                edge_positions
                    .get(&edge)
                    .is_some_and(|row| edge_masks[slot][*row as usize])
                    && snapshot
                        .endpoint(target)
                        .is_some_and(|id| node_maps[hop as usize].contains_key(id))
            },
        ) {
            Ok(walks) => walks,
            Err(_) => return Ok(None), // guard only; no result or fallback stream emitted yet.
        };
        examined = examined.saturating_add(walks.examined_edges);
        for walk in walks.paths {
            for root_row in root_rows {
                let mut states = vec![vec![*root_row]];
                for hop in 0..indices.len() {
                    let edge_row = edge_positions[&walk.edge_ids[hop]];
                    let Some(vertex) = snapshot.endpoint(walk.vertices[hop + 1]) else {
                        return Ok(None);
                    };
                    let target_rows = &node_maps[hop + 1][vertex];
                    let Some(count) = states.len().checked_mul(target_rows.len()) else {
                        return Ok(None);
                    };
                    if count > options.max_take_rows {
                        return Ok(None);
                    }
                    let mut next = Vec::with_capacity(count);
                    for prefix in states {
                        for target_row in target_rows {
                            let mut row = prefix.clone();
                            row.extend([edge_row, *target_row]);
                            next.push(row);
                        }
                    }
                    states = next;
                }
                if tuples.len().saturating_add(states.len()) > options.max_take_rows {
                    return Ok(None);
                }
                tuples.extend(states);
            }
        }
    }
    // Project in exactly the old plan's Arrow order. Take retains NULLs, physical
    // types and dictionary/timestamp metadata; it never stringifies properties.
    let mut selected = Vec::new();
    let mut row_bytes = 0usize;
    for index in projection {
        let name = spec.schema.field(*index).name();
        let Some((slot, raw)) = spec
            .slices
            .iter()
            .enumerate()
            .find_map(|(slot, slice)| raw_name(name, &slice.alias).map(|raw| (slot, raw)))
        else {
            return Ok(None);
        };
        let Some(array) = batches[slot].column_by_name(raw) else {
            return Ok(None);
        };
        let Some(width) = max_value_bytes(array) else {
            return Ok(None);
        };
        let Some(bytes) = row_bytes.checked_add(width) else {
            return Ok(None);
        };
        row_bytes = bytes;
        selected.push((slot, array));
    }
    if row_bytes
        .checked_mul(tuples.len())
        .is_none_or(|bytes| bytes > options.max_bytes)
    {
        return Ok(None);
    }
    let arrays = selected
        .into_iter()
        .map(|(slot, array)| {
            let rows = UInt32Array::from(tuples.iter().map(|row| row[slot]).collect::<Vec<_>>());
            arrow::compute::take(array.as_ref(), &rows, None)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let batch = RecordBatch::try_new_with_options(
        output_schema.clone(),
        arrays,
        &RecordBatchOptions::new().with_row_count(Some(tuples.len())),
    )?;
    spec.metrics
        .native_multihop_edge_checks
        .fetch_add(examined as u64, Ordering::Relaxed);
    Ok(Some(batch))
}
