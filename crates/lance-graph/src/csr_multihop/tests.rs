// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::GraphConfig;
use arrow_array::{
    Array, BooleanArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use datafusion::execution::context::SessionContext;
use lance::{datafusion::LanceTableProvider, dataset::WriteParams, Dataset};
use lance_graph_catalog::{GraphSourceCatalog, InMemoryCatalog};

struct Fixture {
    _temp: tempfile::TempDir,
    config: GraphConfig,
    catalog: Arc<dyn GraphSourceCatalog>,
    cache: Arc<CsrAdjacencyCache>,
    edges: Arc<Dataset>,
}

#[tokio::test]
async fn implicit_row_casts_are_not_candidate_masks() {
    use datafusion::{
        datasource::MemTable,
        logical_expr::{col, lit},
    };
    let provider = Arc::new(
        MemTable::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Utf8, false),
                Field::new("text", DataType::Utf8, false),
                Field::new("confidence", DataType::Float32, false),
                Field::new("count", DataType::Int32, false),
                Field::new(
                    "at",
                    DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into())),
                    true,
                ),
            ])),
            vec![vec![]],
        )
        .unwrap(),
    );
    let state = SessionContext::new().state();
    let make = |predicate: Expr| Slice {
        alias: "a".into(),
        provider: provider.clone(),
        key: "id".into(),
        filters: vec![predicate],
        mandatory_filters: 0,
    };
    let unsafe_compare = col("a__text").gt(lit(0_i64));
    assert!(
        total_mask(&unsafe_compare),
        "pre-coercion AST does not contain a cast"
    );
    assert!(!materialize::typed_mask_is_total(
        &make(unsafe_compare),
        &state
    ));
    let unsafe_membership = col("a__text").in_list(vec![lit(1_i64)], false);
    assert!(!materialize::typed_mask_is_total(
        &make(unsafe_membership),
        &state
    ));
    assert!(materialize::typed_mask_is_total(
        &make(col("a__text").eq(lit("value"))),
        &state
    ));
    assert!(materialize::typed_mask_is_total(
        &make(col("a__confidence").gt(lit(0.5_f64))),
        &state
    ));
    assert!(materialize::typed_mask_is_total(
        &make(col("a__count").gt(lit(1_i64))),
        &state
    ));
    // DataFusion may coerce Microsecond to Nanosecond. That multiplication can
    // overflow, so it must remain a residual WHERE rather than an early mask.
    assert!(!materialize::typed_mask_is_total(
        &make(col("a__at").lt(lit("2024-01-01T00:00:00Z"))),
        &state,
    ));
}

#[tokio::test]
async fn implicit_throwing_cast_replays_legacy_before_any_candidate_read() {
    let mut fixture = fixture(false, false, 100, false).await;
    fixture
        .config
        .node_mappings
        .get_mut("entity")
        .unwrap()
        .filter_conditions = Some("name > 0".into());
    assert!(query(COOCCURS, &fixture, false)
        .execute_with_context_limited(context(&fixture), 1000)
        .await
        .is_err());
    let native = query(COOCCURS, &fixture, true);
    assert!(native
        .execute_with_context_limited(context(&fixture), 1000)
        .await
        .is_err());
    let metrics = native.csr_adjacency_metrics();
    assert_eq!(
        metrics.native_multihop_executions.load(Ordering::Relaxed),
        0
    );
    assert_eq!(metrics.native_multihop_fallbacks.load(Ordering::Relaxed), 1);
    assert_eq!(
        metrics.native_multihop_input_rows.load(Ordering::Relaxed),
        0
    );
    assert_eq!(metrics.build_count.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn implicit_throwing_where_remains_residual() {
    let fixture = fixture(false, false, 100, false).await;
    let text = COOCCURS.replace(
        "AND m.scope = 'user'",
        "AND m.scope = 'user' AND b.name > 0",
    );
    assert!(query(&text, &fixture, false)
        .execute_with_context_limited(context(&fixture), 1000)
        .await
        .is_err());
    let native = query(&text, &fixture, true);
    assert!(native
        .execute_with_context_limited(context(&fixture), 1000)
        .await
        .is_err());
    let metrics = native.csr_adjacency_metrics();
    assert_eq!(
        metrics.native_multihop_executions.load(Ordering::Relaxed),
        1
    );
    assert_eq!(metrics.native_multihop_fallbacks.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn timestamp_unit_coercion_keeps_native_path_and_residual_filter() {
    use arrow_array::TimestampMicrosecondArray;
    let mut fixture = fixture(false, false, 100, false).await;
    let schema = Arc::new(Schema::new(vec![
        Field::new("memory_id", DataType::Utf8, false),
        Field::new("visible", DataType::Boolean, true),
        Field::new("scope", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new(
            "created_at",
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into())),
            true,
        ),
    ]));
    let memories = write(
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["m0", "mh", "mn"])),
                Arc::new(BooleanArray::from(vec![Some(true), Some(false), None])),
                Arc::new(StringArray::from(vec!["user"; 3])),
                Arc::new(StringArray::from(vec!["evidence"; 3])),
                Arc::new(
                    TimestampMicrosecondArray::from(vec![Some(1_672_531_200_000_000); 3])
                        .with_timezone("UTC"),
                ),
            ],
        )
        .unwrap(),
        &fixture._temp.path().join("timestamp_memories.lance"),
    )
    .await;
    fixture.catalog = Arc::new(
        InMemoryCatalog::new()
            .with_node_source("Entity", fixture.catalog.node_source("Entity").unwrap())
            .with_node_source(
                "Memory",
                Arc::new(DefaultTableSource::new(Arc::new(LanceTableProvider::new(
                    memories, false, false,
                )))),
            )
            .with_relationship_source(
                "MENTIONS",
                fixture.catalog.relationship_source("MENTIONS").unwrap(),
            )
            .with_relationship_source("REL", fixture.catalog.relationship_source("REL").unwrap()),
    );
    let text = COOCCURS.replace(
        "AND m.scope = 'user'",
        "AND m.scope = 'user' AND m.created_at < '2024-01-01T00:00:00Z'",
    );
    assert_eq!(compare(&text, &fixture, true).await.num_rows(), 2);
}

async fn write(batch: RecordBatch, path: &std::path::Path) -> Arc<Dataset> {
    let schema = batch.schema();
    Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            path.to_str().unwrap(),
            None::<WriteParams>,
        )
        .await
        .unwrap(),
    )
}

async fn fixture(
    duplicates: bool,
    null_endpoint: bool,
    max_paths: usize,
    cached_entities: bool,
) -> Fixture {
    let temp = tempfile::tempdir().unwrap();
    let entity_schema = Arc::new(Schema::new(vec![
        Field::new("entity_id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("allowed", DataType::Boolean, true),
    ]));
    let mut ids = vec!["e0", "e1", "e2", "e3"];
    let mut names = vec!["root", "one", "two", "three"];
    let mut allowed = vec![Some(true), Some(true), Some(true), Some(true)];
    if duplicates {
        ids.extend(["e0", "e0", "e1", "e1"]);
        names.extend(["root", "rejected-root", "one", "rejected-one"]);
        allowed.extend([Some(true), Some(false), Some(true), Some(false)]);
    }
    let entities_batch = RecordBatch::try_new(
        entity_schema.clone(),
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(StringArray::from(names)),
            Arc::new(BooleanArray::from(allowed)),
        ],
    )
    .unwrap();
    let entities = write(entities_batch.clone(), &temp.path().join("entities.lance")).await;
    let memory_schema = Arc::new(Schema::new(vec![
        Field::new("memory_id", DataType::Utf8, false),
        Field::new("visible", DataType::Boolean, true),
        Field::new("scope", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
    ]));
    let mut mids = vec!["m0", "mh", "mn"];
    let mut vis = vec![Some(true), Some(false), None];
    let mut scopes = vec!["user", "user", "user"];
    let mut content = vec!["public evidence", "private evidence", "null evidence"];
    if duplicates {
        mids.extend(["m0", "m0"]);
        vis.extend([Some(true), Some(false)]);
        scopes.extend(["user", "user"]);
        content.extend(["second public evidence", "rejected duplicate"]);
    }
    let memories = write(
        RecordBatch::try_new(
            memory_schema,
            vec![
                Arc::new(StringArray::from(mids)),
                Arc::new(BooleanArray::from(vis)),
                Arc::new(StringArray::from(scopes)),
                Arc::new(StringArray::from(content)),
            ],
        )
        .unwrap(),
        &temp.path().join("memories.lance"),
    )
    .await;
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("source_id", DataType::Utf8, true),
        Field::new("target_id", DataType::Utf8, false),
        Field::new("relationship_type", DataType::Utf8, false),
        Field::new("edge_id", DataType::Utf8, false),
        Field::new("confidence", DataType::Float32, true),
    ]));
    let mut sources = vec![
        Some("m0"),
        Some("m0"),
        Some("m0"),
        Some("mh"),
        Some("mh"),
        Some("mn"),
        Some("mn"),
        Some("m0"),
        Some("e0"),
        Some("e1"),
        Some("e2"),
        Some("e1"),
    ];
    let mut targets = vec![
        "e0", "e0", "e1", "e0", "e2", "e0", "e3", "missing", "e1", "e2", "e3", "e0",
    ];
    let mut types = vec!["MENTIONS"; 8];
    types.extend(["REL"; 4]);
    let mut eid = vec![
        "parallel",
        "parallel",
        "m0-e1",
        "private-root",
        "private-next",
        "null-root",
        "null-next",
        "missing-node",
        "r01",
        "r12",
        "r23",
        "cycle",
    ];
    let mut confidence = vec![
        Some(0.8),
        Some(0.7),
        Some(0.9),
        Some(0.8),
        Some(0.8),
        None,
        Some(0.0),
        Some(-0.0),
        Some(0.9),
        Some(0.5),
        Some(0.6),
        Some(0.4),
    ];
    if null_endpoint {
        sources.push(None);
        targets.push("e0");
        types.push("MENTIONS");
        eid.push("null-endpoint");
        confidence.push(Some(0.9));
    }
    let edges = write(
        RecordBatch::try_new(
            edge_schema,
            vec![
                Arc::new(StringArray::from(sources)),
                Arc::new(StringArray::from(targets)),
                Arc::new(StringArray::from(types)),
                Arc::new(StringArray::from(eid)),
                Arc::new(Float32Array::from(confidence)),
            ],
        )
        .unwrap(),
        &temp.path().join("edges.lance"),
    )
    .await;
    let mut config = GraphConfig::builder()
        .with_node_label("Entity", "entity_id")
        .with_node_label("Memory", "memory_id")
        .with_relationship("MENTIONS", "source_id", "target_id")
        .with_relationship("REL", "source_id", "target_id")
        .build()
        .unwrap();
    for mapping in config.relationship_mappings.values_mut() {
        mapping.type_field = Some("relationship_type".into());
    }
    let entity_provider: Arc<dyn TableProvider> = if cached_entities {
        Arc::new(MemTable::try_new(entity_schema, vec![vec![entities_batch]]).unwrap())
    } else {
        Arc::new(LanceTableProvider::new(entities, false, false))
    };
    let edge_source: Arc<dyn TableSource> = Arc::new(DefaultTableSource::new(Arc::new(
        LanceTableProvider::new(edges.clone(), false, false),
    )));
    let catalog = Arc::new(
        InMemoryCatalog::new()
            .with_node_source("Entity", Arc::new(DefaultTableSource::new(entity_provider)))
            .with_node_source(
                "Memory",
                Arc::new(DefaultTableSource::new(Arc::new(LanceTableProvider::new(
                    memories, false, false,
                )))),
            )
            .with_relationship_source("MENTIONS", edge_source.clone())
            .with_relationship_source("REL", edge_source),
    );
    Fixture {
        _temp: temp,
        config,
        catalog,
        edges,
        cache: Arc::new(CsrAdjacencyCache::new(crate::CsrScanOptions {
            max_take_rows: max_paths,
            ..Default::default()
        })),
    }
}

fn query(text: &str, fixture: &Fixture, native: bool) -> crate::CypherQuery {
    crate::CypherQuery::new(text)
        .unwrap()
        .with_config(fixture.config.clone())
        .with_shared_csr_cache(fixture.cache.clone())
        .with_native_multihop(native)
}

fn context(fixture: &Fixture) -> SessionContext {
    let ctx = SessionContext::new();
    for label in fixture.config.node_mappings.keys() {
        let provider = source_as_provider(&fixture.catalog.node_source(label).unwrap()).unwrap();
        ctx.register_table(label.as_str(), provider).unwrap();
    }
    for relation in fixture.config.relationship_mappings.keys() {
        let provider =
            source_as_provider(&fixture.catalog.relationship_source(relation).unwrap()).unwrap();
        ctx.register_table(relation.as_str(), provider).unwrap();
    }
    ctx
}

fn row_keys(batch: &RecordBatch) -> Vec<Vec<Option<String>>> {
    let mut rows = (0..batch.num_rows())
        .map(|row| {
            batch
                .columns()
                .iter()
                .map(|array| {
                    if array.is_null(row) {
                        return None;
                    }
                    if let Some(array) = array.as_any().downcast_ref::<Float32Array>() {
                        return Some(format!("f32:{:08x}", array.value(row).to_bits()));
                    }
                    Some(arrow::util::display::array_value_to_string(array.as_ref(), row).unwrap())
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    rows.sort();
    rows
}

async fn compare(text: &str, fixture: &Fixture, expected_native: bool) -> RecordBatch {
    let baseline = query(text, fixture, false)
        .execute_with_context_limited(context(fixture), 1000)
        .await
        .unwrap();
    let native = query(text, fixture, true);
    let result = native
        .execute_with_context_limited(context(fixture), 1000)
        .await
        .unwrap();
    assert_eq!(
        baseline.schema(),
        result.schema(),
        "complete Arrow contract"
    );
    assert_eq!(
        row_keys(&baseline),
        row_keys(&result),
        "NULL/F32 bits/bag multiplicity"
    );
    assert_eq!(
        native
            .csr_adjacency_metrics()
            .native_multihop_executions
            .load(Ordering::Relaxed)
            > 0,
        expected_native
    );
    result
}

const COOCCURS: &str = "MATCH (a:Entity)<-[r1:MENTIONS]-(m:Memory)-[r2:MENTIONS]->(b:Entity) \
    WHERE a.entity_id IN ['e0'] AND b.entity_id <> a.entity_id AND m.visible = true AND m.scope = 'user' \
    RETURN a.entity_id AS origin, b.entity_id AS neighbor, m.memory_id AS evidence, m.content AS content, \
    r1.edge_id AS edge1, r2.edge_id AS edge2, r1.confidence AS c1, r2.confidence AS c2";

#[tokio::test]
async fn actual_two_hop_cooccurs_with_memory_acl_parallel_edges_and_cached_entity() {
    for cached in [false, true] {
        let fixture = fixture(false, false, 100, cached).await;
        let result = compare(COOCCURS, &fixture, true).await;
        assert_eq!(result.num_rows(), 2);
        assert!(row_keys(&result)
            .iter()
            .all(|row| row[1].as_deref() == Some("e1") && row[2].as_deref() == Some("m0")));
        let plan = query(COOCCURS, &fixture, true)
            .explain_with_context(context(&fixture))
            .await
            .unwrap();
        assert!(plan.contains("NativeCsrMultiHop"));
        assert!(!plan.contains("HashJoinExec"));
    }
}

#[tokio::test]
async fn duplicate_node_rows_keep_exact_filtered_cartesian_multiplicity() {
    let fixture = fixture(true, false, 100, false).await;
    let text = COOCCURS
        .replace("(a:Entity)", "(a:Entity {allowed:true})")
        .replace("(b:Entity)", "(b:Entity {allowed:true})");
    let result = compare(&text, &fixture, true).await;
    assert_eq!(result.num_rows(), 16); // 2 root ×2 evidence ×2 target ×2 physical parallel edges.
}

#[tokio::test]
async fn three_hops_preserve_cycles_evidence_properties_and_ordered_limit() {
    let fixture = fixture(false, false, 100, false).await;
    let text = "MATCH (a:Entity {entity_id:'e0'})-[r1:REL]->(b:Entity)-[r2:REL]->(c:Entity)-[r3:REL]->(d:Entity) \
        WHERE r1.confidence >= 0.5 AND r2.confidence IS NOT NULL AND d.allowed = true \
        RETURN a.entity_id, b.entity_id, c.entity_id, d.entity_id, r1.edge_id, r2.edge_id, r3.edge_id \
        ORDER BY d.entity_id LIMIT 20";
    let result = compare(text, &fixture, true).await;
    assert_eq!(result.num_rows(), 2);
    assert!(row_keys(&result)
        .iter()
        .any(|row| row[2].as_deref() == Some("e0") && row[6].as_deref() == Some("r01")));
}

#[tokio::test]
async fn cross_alias_or_remains_residual_distinct_and_zero_column_count_are_exact() {
    let fixture = fixture(false, false, 100, false).await;
    let text = COOCCURS.replace(
        "AND m.visible = true AND m.scope = 'user'",
        "AND (m.visible = true OR b.entity_id = 'e2')",
    );
    let result = compare(&text, &fixture, true).await;
    assert_eq!(result.num_rows(), 3);
    let prefix = COOCCURS.split(" RETURN ").next().unwrap();
    let distinct = compare(
        &format!("{prefix} RETURN DISTINCT b.entity_id ORDER BY b.entity_id"),
        &fixture,
        true,
    )
    .await;
    assert_eq!(distinct.num_rows(), 1);
    let count = compare(
        &format!("{prefix} RETURN COUNT(*) AS paths"),
        &fixture,
        true,
    )
    .await;
    assert_eq!(count.num_rows(), 1);
    compare(
        &COOCCURS.replace("['e0']", "['not-present']"),
        &fixture,
        true,
    )
    .await;
}

#[tokio::test]
async fn budget_and_nullable_topology_fallback_without_partial_output() {
    for (null_endpoint, budget, admitted) in
        [(true, 100, false), (false, 6, true), (false, 5, false)]
    {
        let fixture = fixture(false, null_endpoint, budget, true).await;
        let result = compare(COOCCURS, &fixture, admitted).await;
        assert_eq!(result.num_rows(), 2);
    }
}

#[tokio::test]
async fn common_snapshot_cache_reused_and_version_change_isolated() {
    let mut fixture = fixture(false, false, 100, false).await;
    let first = query(COOCCURS, &fixture, true);
    first
        .execute_with_catalog_and_context(fixture.catalog.clone(), SessionContext::new())
        .await
        .unwrap();
    assert_eq!(
        first
            .csr_adjacency_metrics()
            .build_count
            .load(Ordering::Relaxed),
        1
    );
    let second = query(COOCCURS, &fixture, true);
    second
        .execute_with_catalog_and_context(fixture.catalog.clone(), SessionContext::new())
        .await
        .unwrap();
    assert_eq!(
        second
            .csr_adjacency_metrics()
            .build_count
            .load(Ordering::Relaxed),
        0
    );
    assert_eq!(
        second
            .csr_adjacency_metrics()
            .cache_hits
            .load(Ordering::Relaxed),
        1
    );
    assert_eq!(fixture.edges.version_id(), 1);
    let new_batch = RecordBatch::try_new(
        Arc::new(Schema::from(fixture.edges.schema())),
        vec![
            Arc::new(StringArray::from(vec![Some("m0")])),
            Arc::new(StringArray::from(vec!["e2"])),
            Arc::new(StringArray::from(vec!["MENTIONS"])),
            Arc::new(StringArray::from(vec!["new-version-edge"])),
            Arc::new(Float32Array::from(vec![Some(0.7)])),
        ],
    )
    .unwrap();
    let updated = Dataset::write(
        RecordBatchIterator::new(vec![Ok(new_batch.clone())], new_batch.schema()),
        fixture.edges.uri(),
        Some(WriteParams {
            mode: lance::dataset::WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(updated.version_id(), 2);
    let edge_source: Arc<dyn TableSource> = Arc::new(DefaultTableSource::new(Arc::new(
        LanceTableProvider::new(Arc::new(updated), false, false),
    )));
    fixture.catalog = Arc::new(
        InMemoryCatalog::new()
            .with_node_source("Entity", fixture.catalog.node_source("Entity").unwrap())
            .with_node_source("Memory", fixture.catalog.node_source("Memory").unwrap())
            .with_relationship_source("MENTIONS", edge_source.clone())
            .with_relationship_source("REL", edge_source),
    );
    let third = query(COOCCURS, &fixture, true);
    let result = third
        .execute_with_context_limited(context(&fixture), 1000)
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 4);
    assert_eq!(
        third
            .csr_adjacency_metrics()
            .build_count
            .load(Ordering::Relaxed),
        1
    );
    assert_eq!(
        third
            .csr_adjacency_metrics()
            .cache_hits
            .load(Ordering::Relaxed),
        0
    );
}

#[test]
fn unsafe_or_throwing_subexpressions_are_not_pre_masks() {
    let a = col("a__id").eq(datafusion::logical_expr::lit("e0"));
    let b = col("b__id").eq(datafusion::logical_expr::lit("e1"));
    let mut selected = Vec::new();
    local_filters(&a.clone().or(b), "a", &mut selected);
    assert!(selected.is_empty());
    local_filters(
        &Expr::Cast(datafusion::logical_expr::expr::Cast::new(
            Box::new(col("a__raw")),
            DataType::Int64,
        ))
        .gt(datafusion::logical_expr::lit(0)),
        "a",
        &mut selected,
    );
    assert!(selected.is_empty());
    local_filters(&a, "a", &mut selected);
    assert_eq!(selected.len(), 1);
}

#[tokio::test]
async fn count_only_uses_zero_output_columns_and_empty_paths_are_exact() {
    let fixture = fixture(false, false, 100, true).await;
    let text = "MATCH (a:Entity {entity_id:'e0'})<-[r1:MENTIONS]-(m:Memory {visible:true})-[r2:MENTIONS]->(b:Entity) RETURN COUNT(*) AS paths";
    let count = compare(text, &fixture, true).await;
    assert_eq!(row_keys(&count), vec![vec![Some("6".to_owned())]]);
    let empty = compare(&text.replace("'e0'", "'not-present'"), &fixture, true).await;
    assert_eq!(row_keys(&empty), vec![vec![Some("0".to_owned())]]);
}

#[tokio::test]
async fn broad_visible_scope_does_not_prefetch_unreachable_large_bodies() {
    let mut fixture = fixture(false, false, 100, true).await;
    let schema = source_as_provider(&fixture.catalog.node_source("Memory").unwrap())
        .unwrap()
        .schema();
    let n = 20_000;
    let large = "x".repeat(8_192);
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(
                std::iter::once("m0".to_owned())
                    .chain((0..n).map(|i| format!("unreachable-{i}")))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(BooleanArray::from(vec![true; n + 1])),
            Arc::new(StringArray::from(vec!["user"; n + 1])),
            Arc::new(StringArray::from(
                std::iter::once("public evidence".to_owned())
                    .chain(std::iter::repeat_n(large, n))
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    let memories = write(batch, &fixture._temp.path().join("broad-visible.lance")).await;
    fixture.catalog = Arc::new(
        InMemoryCatalog::new()
            .with_node_source("Entity", fixture.catalog.node_source("Entity").unwrap())
            .with_node_source(
                "Memory",
                Arc::new(DefaultTableSource::new(Arc::new(LanceTableProvider::new(
                    memories, false, false,
                )))),
            )
            .with_relationship_source(
                "MENTIONS",
                fixture.catalog.relationship_source("MENTIONS").unwrap(),
            )
            .with_relationship_source("REL", fixture.catalog.relationship_source("REL").unwrap()),
    );
    let result = compare(COOCCURS, &fixture, true).await;
    assert_eq!(result.num_rows(), 2);
    let q = query(COOCCURS, &fixture, true);
    q.execute_with_context_limited(context(&fixture), 100)
        .await
        .unwrap();
    assert!(
        q.csr_adjacency_metrics()
            .native_multihop_input_rows
            .load(Ordering::Relaxed)
            < 50
    );
    assert!(
        q.csr_adjacency_metrics()
            .native_multihop_input_bytes
            .load(Ordering::Relaxed)
            < 1_000_000
    );
    assert_eq!(
        q.csr_adjacency_metrics()
            .native_multihop_fallbacks
            .load(Ordering::Relaxed),
        0
    );
}

#[tokio::test]
async fn unsafe_mapping_expression_is_not_admitted() {
    let fixture = fixture(false, false, 100, true).await;
    let provider = source_as_provider(&fixture.catalog.node_source("Memory").unwrap()).unwrap();
    let mapping = col("scope").eq(Expr::Cast(datafusion::logical_expr::expr::Cast::new(
        Box::new(datafusion::logical_expr::lit("user")),
        DataType::Utf8,
    )));
    assert!(slice(
        "m",
        provider,
        "memory_id",
        Some(mapping),
        &std::collections::HashMap::new(),
        None
    )
    .unwrap()
    .is_none());
}

#[test]
fn bag_oracle_does_not_confuse_null_with_literal_null_text() {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, true)])),
        vec![Arc::new(StringArray::from(vec![None, Some("NULL")]))],
    )
    .unwrap();
    assert_eq!(
        row_keys(&batch),
        vec![vec![None], vec![Some("NULL".to_owned())]]
    );
}

#[test]
fn public_shape_hint_is_not_native_execution_evidence() {
    assert!(crate::CypherQuery::new(COOCCURS)
        .unwrap()
        .with_native_multihop(false)
        .native_multihop_candidate());
    assert!(
        crate::CypherQuery::new("MATCH (a:Entity)-[r:REL]->(b:Entity) RETURN b.entity_id")
            .unwrap()
            .native_multihop_candidate()
    );
    for text in [
        "MATCH (a:Entity)-[:REL]->(b)-[:REL]->(c)-[:REL]->(d)-[:REL]->(e) RETURN e.entity_id",
        "MATCH (a:Entity)-[r:REL*1..3]->(b:Entity) RETURN b.entity_id",
        "MATCH (a:Entity)-[r:REL]->(b:Entity)-[s:REL]->(c:Entity) WITH c RETURN c.entity_id",
    ] {
        assert!(!crate::CypherQuery::new(text)
            .unwrap()
            .native_multihop_candidate());
    }
}

#[tokio::test]
async fn reachable_node_rows_are_taken_through_a_reused_pinned_id_map() {
    let fixture = fixture(true, false, 100, false).await;
    let text = "MATCH (a:Entity {entity_id:'e0'})-[r1:REL]->(b:Entity)-[r2:REL]->(c:Entity)-[r3:REL]->(d:Entity) \
        WHERE b.allowed = true RETURN b.name, c.entity_id, d.entity_id, r3.edge_id";
    compare(text, &fixture, true).await;
    let first = query(text, &fixture, true);
    first
        .execute_with_catalog_and_context(fixture.catalog.clone(), SessionContext::new())
        .await
        .unwrap();
    let metrics = first.csr_adjacency_metrics();
    // compare() already built the Entity ID map; this query only reuses it.
    assert_eq!(
        metrics
            .native_multihop_node_key_builds
            .load(Ordering::Relaxed),
        0
    );
    assert_eq!(
        metrics
            .native_multihop_node_take_reads
            .load(Ordering::Relaxed),
        1
    );
    assert_eq!(
        metrics
            .native_multihop_node_index_reads
            .load(Ordering::Relaxed),
        0
    );
    // Only rows of reachable IDs (with their duplicates) are read, not the table.
    let snapshot = fixture
        .cache
        .node_keys(
            underlying_lance(
                &source_as_provider(&fixture.catalog.node_source("Entity").unwrap()).unwrap(),
            )
            .unwrap()
            .dataset(),
            "entity_id",
            &metrics,
        )
        .await
        .unwrap();
    assert_eq!(snapshot.rows(&Endpoint::Text("e0".into())).len(), 3);
    assert_eq!(snapshot.rows(&Endpoint::Text("e3".into())).len(), 1);
    assert!(snapshot.rows(&Endpoint::Text("missing".into())).is_empty());
}

#[tokio::test]
async fn shared_table_aliases_hoist_id_list_for_node_btree() {
    use lance::index::DatasetIndexExt;
    use lance_index::{scalar::ScalarIndexParams, IndexType};
    let mut fixture = fixture(false, false, 100, false).await;
    let entities = source_as_provider(&fixture.catalog.node_source("Entity").unwrap()).unwrap();
    let mut entities = underlying_lance(&entities)
        .unwrap()
        .dataset()
        .as_ref()
        .clone();
    entities
        .create_index_builder(
            &["entity_id"],
            IndexType::BTree,
            &ScalarIndexParams::default(),
        )
        .await
        .unwrap();
    fixture.catalog = Arc::new(
        InMemoryCatalog::new()
            .with_node_source(
                "Entity",
                Arc::new(DefaultTableSource::new(Arc::new(LanceTableProvider::new(
                    Arc::new(entities),
                    false,
                    false,
                )))),
            )
            .with_node_source("Memory", fixture.catalog.node_source("Memory").unwrap())
            .with_relationship_source(
                "MENTIONS",
                fixture.catalog.relationship_source("MENTIONS").unwrap(),
            )
            .with_relationship_source("REL", fixture.catalog.relationship_source("REL").unwrap()),
    );
    // Four Entity aliases with different masks share one pushed node read.
    let text = "MATCH (a:Entity {entity_id:'e0'})-[r1:REL]->(b:Entity)-[r2:REL]->(c:Entity)-[r3:REL]->(d:Entity) \
        WHERE b.allowed = true AND c.allowed = true AND d.allowed = true \
        RETURN b.entity_id, c.entity_id, d.entity_id, r1.edge_id, r2.edge_id, r3.edge_id";
    compare(text, &fixture, true).await;
    let q = query(text, &fixture, true);
    q.execute_with_catalog_and_context(fixture.catalog.clone(), SessionContext::new())
        .await
        .unwrap();
    let metrics = q.csr_adjacency_metrics();
    assert_eq!(
        metrics
            .native_multihop_node_index_reads
            .load(Ordering::Relaxed),
        1
    );
    // An existing ID BTree is preferred over a resident ID map.
    assert_eq!(
        metrics
            .native_multihop_node_take_reads
            .load(Ordering::Relaxed),
        0
    );
}

#[tokio::test]
async fn single_hop_outgoing_incoming_and_edge_properties_are_exact() {
    for cached in [false, true] {
        let fixture = fixture(false, false, 100, cached).await;
        let out = compare(
            "MATCH (a:Entity {entity_id:'e1'})-[r:REL]->(b:Entity) \
             RETURN a.name, b.entity_id, b.name, r.edge_id, r.confidence",
            &fixture,
            true,
        )
        .await;
        assert_eq!(out.num_rows(), 2); // r12 and the e1 -> e0 cycle edge.
                                       // Incoming keeps both physical parallel edges and the edge ACL mask.
        let incoming = compare(
            "MATCH (e:Entity)<-[r:MENTIONS]-(m:Memory) \
             WHERE e.entity_id = 'e0' AND m.visible = true AND r.confidence > 0.5 \
             RETURN m.memory_id, m.content, r.edge_id, r.confidence",
            &fixture,
            true,
        )
        .await;
        assert_eq!(incoming.num_rows(), 2);
        let filtered = compare(
            "MATCH (a:Entity)-[r:REL]->(b:Entity) \
             WHERE a.entity_id IN ['e0', 'e1', 'e2'] AND r.confidence >= 0.5 AND b.name <> 'two' \
             RETURN a.entity_id, b.entity_id, r.edge_id",
            &fixture,
            true,
        )
        .await;
        assert_eq!(filtered.num_rows(), 2); // r01, r23; r12 targets 'two', cycle is 0.4.
    }
}

#[tokio::test]
async fn single_hop_duplicates_null_endpoints_and_budget_fallback() {
    let duplicated = fixture(true, false, 100, false).await;
    let result = compare(
        "MATCH (m:Memory {memory_id:'m0'})-[r:MENTIONS]->(e:Entity {allowed:true}) \
         RETURN m.content, e.name, r.edge_id",
        &duplicated,
        true,
    )
    .await;
    // 3 m0 rows x (2 parallel e0 edges x 2 allowed e0 rows + 1 edge x 2 allowed e1 rows).
    assert_eq!(result.num_rows(), 3 * (2 * 2 + 2));
    for (null_endpoint, budget, admitted) in
        [(true, 100, false), (false, 3, true), (false, 2, false)]
    {
        let fixture = fixture(false, null_endpoint, budget, true).await;
        let result = compare(
            "MATCH (m:Memory {memory_id:'m0'})-[r:MENTIONS]->(e:Entity) RETURN e.entity_id, r.edge_id",
            &fixture,
            admitted,
        )
        .await;
        // The edge to an absent 'missing' node never joins.
        assert_eq!(result.num_rows(), 3);
    }
}

#[tokio::test]
async fn single_hop_aggregates_ordering_parameters_and_empty_seed() {
    let fixture = fixture(false, false, 100, false).await;
    let prefix =
        "MATCH (m:Memory)-[r:MENTIONS]->(e:Entity) WHERE m.memory_id IN ['m0', 'mh', 'mn']";
    let ordered = compare(
        &format!("{prefix} RETURN e.entity_id, r.edge_id ORDER BY r.edge_id DESC LIMIT 3"),
        &fixture,
        true,
    )
    .await;
    assert_eq!(ordered.num_rows(), 3);
    let distinct = compare(
        &format!("{prefix} RETURN DISTINCT e.entity_id ORDER BY e.entity_id"),
        &fixture,
        true,
    )
    .await;
    assert_eq!(distinct.num_rows(), 4);
    let grouped = compare(
        &format!("{prefix} RETURN m.memory_id, COUNT(*) AS edges"),
        &fixture,
        true,
    )
    .await;
    assert_eq!(grouped.num_rows(), 3);
    compare(
        "MATCH (m:Memory)-[r:MENTIONS]->(e:Entity) WHERE m.memory_id = 'absent' RETURN e.entity_id",
        &fixture,
        true,
    )
    .await;
    let text = "MATCH (a:Entity)-[r:REL]->(b:Entity) WHERE a.entity_id = $seed RETURN b.entity_id";
    let baseline = query(text, &fixture, false)
        .with_parameter("seed", "e1")
        .execute_with_context_limited(context(&fixture), 1000)
        .await
        .unwrap();
    let native = query(text, &fixture, true).with_parameter("seed", "e1");
    let result = native
        .execute_with_context_limited(context(&fixture), 1000)
        .await
        .unwrap();
    assert_eq!(row_keys(&baseline), row_keys(&result));
    assert_eq!(result.num_rows(), 2);
    assert_eq!(
        native
            .csr_adjacency_metrics()
            .native_multihop_executions
            .load(Ordering::Relaxed),
        1
    );
}

#[tokio::test]
async fn anonymous_relationships_use_join_plan_qualifiers() {
    let fixture = fixture(false, false, 100, false).await;
    let one = compare(
        "MATCH (a:Entity {entity_id:'e0'})-[:REL]->(b:Entity) RETURN b.entity_id, b.name",
        &fixture,
        true,
    )
    .await;
    assert_eq!(one.num_rows(), 1);
    let mixed = compare(
        "MATCH (a:Entity {entity_id:'e0'})-[:REL]->(b:Entity)-[r:REL]->(c:Entity)-[:REL]->(d:Entity) \
         RETURN b.entity_id, c.entity_id, d.entity_id, r.edge_id",
        &fixture,
        true,
    )
    .await;
    assert_eq!(mixed.num_rows(), 2);
    // Same relationship type twice: rel_1 and rel_2 must not be confused.
    let two = compare(
        "MATCH (a:Entity)<-[:MENTIONS]-(m:Memory)-[:MENTIONS]->(b:Entity) \
         WHERE a.entity_id = 'e0' AND b.entity_id <> 'e0' RETURN m.memory_id, b.entity_id",
        &fixture,
        true,
    )
    .await;
    assert_eq!(two.num_rows(), 4); // m0 reaches e1 through both parallel e0 edges.
                                   // A user variable named like a generated qualifier is not admitted.
    compare(
        "MATCH (rel_1:Entity {entity_id:'e0'})-[:REL]->(b:Entity) RETURN b.entity_id",
        &fixture,
        false,
    )
    .await;
}

#[tokio::test]
async fn unseeded_or_target_bound_single_hop_keeps_join_plan() {
    let fixture = fixture(false, false, 100, false).await;
    for text in [
        "MATCH (a:Entity)-[r:REL]->(b:Entity) RETURN a.entity_id, b.entity_id, r.edge_id",
        "MATCH (a:Entity)-[r:REL]->(b:Entity {entity_id:'e2'}) RETURN a.entity_id, r.edge_id",
        "MATCH (a:Entity)-[r:REL]->(b:Entity) WHERE a.entity_id = 'e0' OR b.entity_id = 'e2' \
         RETURN a.entity_id, b.entity_id",
        "MATCH (a:Entity {entity_id:'e0'})-[r:REL]-(b:Entity) RETURN b.entity_id",
    ] {
        let result = compare(text, &fixture, false).await;
        assert!(result.num_rows() > 0, "{text}");
    }
}
