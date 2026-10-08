// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use datafusion::{
    datasource::{DefaultTableSource, MemTable, TableProvider},
    execution::context::SessionContext,
};
use lance::{
    datafusion::LanceTableProvider, dataset::WriteParams, index::DatasetIndexExt, Dataset,
};
use lance_graph::{
    CypherQuery, DirNamespace, GraphConfig, InMemoryCatalog, RuntimeIdFilterMetrics,
};
use lance_index::{scalar::ScalarIndexParams, IndexType};
use std::{
    path::Path,
    sync::{atomic::Ordering, Arc},
};

async fn write(path: &Path, batch: RecordBatch) -> Dataset {
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    Dataset::write(reader, path.to_str().unwrap(), None::<WriteParams>)
        .await
        .unwrap()
}
fn one_column(field: &str, values: Vec<String>) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(field, DataType::Utf8, false)])),
        vec![Arc::new(StringArray::from(values))],
    )
    .unwrap()
}
async fn fixture(
    n: usize,
    indexed: bool,
) -> (tempfile::TempDir, Arc<Dataset>, RecordBatch, RecordBatch) {
    let temp = tempfile::tempdir().unwrap();
    let hub = n == 512;
    let ids = (0..if hub { 4096 } else { 512 })
        .map(|i| match i {
            0 => "O'Reilly".to_string(),
            1 => "实体/记忆🙂".to_string(),
            _ => format!("id{i}"),
        })
        .collect::<Vec<_>>();
    let mut thing = write(
        &temp.path().join("thing.lance"),
        one_column("key", ids.clone()),
    )
    .await;
    if indexed {
        thing
            .create_index_builder(&["key"], IndexType::BTree, &ScalarIndexParams::default())
            .name("thing_key".to_string())
            .await
            .unwrap();
    }
    let seed = one_column(
        "seed_id",
        if hub {
            vec![ids[0].clone()]
        } else {
            ids[..n].to_vec()
        },
    );
    let edges = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("src", DataType::Utf8, false),
            Field::new("dst", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(StringArray::from(if hub {
                vec![ids[0].clone(); n]
            } else {
                ids[..n].to_vec()
            })),
            Arc::new(StringArray::from(ids[..n].to_vec())),
        ],
    )
    .unwrap();
    write(&temp.path().join("seed.lance"), seed.clone()).await;
    write(&temp.path().join("knows.lance"), edges.clone()).await;
    (temp, Arc::new(thing), seed, edges)
}
fn config() -> GraphConfig {
    GraphConfig::builder()
        .with_node_label("seed", "seed_id")
        .with_node_label("thing", "key")
        .with_relationship("knows", "src", "dst")
        .build()
        .unwrap()
}
fn values(batch: &RecordBatch) -> Vec<String> {
    let strings = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let mut out = strings
        .iter()
        .map(|s| s.unwrap().to_string())
        .collect::<Vec<_>>();
    out.sort();
    out
}
#[tokio::test]
async fn six_public_entrypoints_use_configured_id_btree_with_default_off() {
    let (temp, thing, seed, edges) = fixture(2, true).await;
    let mut expected = None;
    for path in 0..6 {
        // Actual public A/B/A, not a manually registered private provider.
        for enabled in [false, true, false] {
            let ctx = SessionContext::new();
            let metrics = Arc::new(RuntimeIdFilterMetrics::default());
            let query = CypherQuery::new("MATCH (s:seed)-[:knows]->(t:thing) RETURN t.key")
                .unwrap()
                .with_config(config());
            let query = if enabled {
                query.with_runtime_id_filters(metrics.clone())
            } else {
                query
            };
            let providers: [(&str, Arc<dyn TableProvider>); 3] = [
                (
                    "thing",
                    Arc::new(LanceTableProvider::new(thing.clone(), false, false)),
                ),
                (
                    "seed",
                    Arc::new(MemTable::try_new(seed.schema(), vec![vec![seed.clone()]]).unwrap()),
                ),
                (
                    "knows",
                    Arc::new(MemTable::try_new(edges.schema(), vec![vec![edges.clone()]]).unwrap()),
                ),
            ];
            let mut catalog = InMemoryCatalog::new();
            for (label, provider) in providers {
                ctx.register_table(label, provider.clone()).unwrap();
                let source = Arc::new(DefaultTableSource::new(provider));
                catalog = if label == "knows" {
                    catalog.with_relationship_source(label, source)
                } else {
                    catalog.with_node_source(label, source)
                };
            }
            let namespace = DirNamespace::new(temp.path().to_str().unwrap());
            let batch = match path {
                0 => query.execute_with_context(ctx).await,
                1 => query.execute_with_context_limited(ctx, 100).await,
                2 => {
                    query
                        .execute_with_catalog_and_context(Arc::new(catalog), ctx)
                        .await
                }
                3 => {
                    query
                        .execute_with_catalog_and_context_limited(Arc::new(catalog), ctx, 100)
                        .await
                }
                4 => query.execute_with_namespace(namespace, None).await,
                _ => {
                    query
                        .execute_with_namespace_arc(Arc::new(namespace), None)
                        .await
                }
            }
            .unwrap();
            assert_eq!(batch.num_rows(), 2);
            if let Some(expected) = &expected {
                assert_eq!(&values(&batch), expected);
            } else {
                expected = Some(values(&batch));
            }
            assert_eq!(
                metrics.index_scans.load(Ordering::Relaxed) > 0,
                enabled,
                "entrypoint {path}"
            );
            if enabled && path >= 4 {
                assert!(metrics.skipped_no_index.load(Ordering::Relaxed) > 0);
            }
        }
    }
}
#[tokio::test]
async fn topk_is_not_a_join_producer_and_preserves_ordered_limit() {
    let (temp, _, _, _) = fixture(2, true).await;
    let metrics = Arc::new(RuntimeIdFilterMetrics::default());
    let query = CypherQuery::new("MATCH (t:thing) RETURN t.key ORDER BY t.key LIMIT 3")
        .unwrap()
        .with_config(config());
    let a = query
        .execute_with_namespace(DirNamespace::new(temp.path().to_str().unwrap()), None)
        .await
        .unwrap();
    let b = query
        .with_runtime_id_filters(metrics.clone())
        .execute_with_namespace(DirNamespace::new(temp.path().to_str().unwrap()), None)
        .await
        .unwrap();
    assert_eq!(a.column(0), b.column(0));
    assert_eq!(a.num_rows(), 3);
    assert_eq!(metrics.index_scans.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.timeout_fallbacks.load(Ordering::Relaxed), 0);
    assert!(metrics.rejected_producer_filters.load(Ordering::Relaxed) > 0);
}
#[tokio::test]
async fn bounded_producer_never_truncates_high_degree_and_no_index_is_unchanged() {
    for (n, indexed) in [
        (149, true),
        (150, true),
        (151, true),
        (512, true),
        (3, false),
    ] {
        let (temp, _, seed, edges) = fixture(n, indexed).await;
        if n == 512 {
            assert_eq!(seed.num_rows(), 1);
            assert_eq!(edges.num_rows(), 512);
            assert!(edges
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .all(|id| id == Some("O'Reilly")));
        }
        let metrics = Arc::new(RuntimeIdFilterMetrics::default());
        let query = CypherQuery::new("MATCH (s:seed)-[:knows]->(t:thing) RETURN t.key")
            .unwrap()
            .with_config(config());
        let a = query
            .execute_with_namespace(DirNamespace::new(temp.path().to_str().unwrap()), None)
            .await
            .unwrap();
        let b = query
            .with_runtime_id_filters(metrics.clone())
            .execute_with_namespace(DirNamespace::new(temp.path().to_str().unwrap()), None)
            .await
            .unwrap();
        assert_eq!(a.num_rows(), n);
        assert_eq!(values(&a), values(&b));
        assert_eq!(
            metrics.index_scans.load(Ordering::Relaxed) > 0,
            indexed && n <= 150
        );
        if indexed && n > 150 {
            assert!(metrics.unsupported_fallbacks.load(Ordering::Relaxed) > 0);
        }
        if n > 150 {
            assert_eq!(metrics.filtered_keys.load(Ordering::Relaxed), 0);
        }
        if !indexed {
            assert!(metrics.skipped_no_index.load(Ordering::Relaxed) > 0);
        }
    }
}

#[tokio::test]
async fn public_hook_btree_aba_preserves_alias_acl_orphans_and_duplicate_multiplicity() {
    let temp = tempfile::tempdir().unwrap();
    let mut keys = vec![
        Some("seed".to_string()),
        Some("n".to_string()),
        Some("n".to_string()),
        Some("denied".to_string()),
        None,
    ];
    keys.extend((0..256).map(|i| Some(format!("unused{i}"))));
    let mut names = vec![
        "self".to_string(),
        "first".to_string(),
        "second".to_string(),
        "denied".to_string(),
        "null".to_string(),
    ];
    names.extend((0..256).map(|i| format!("unused{i}")));
    let mut orgs = vec!["allowed"; keys.len()];
    orgs[3] = "denied";
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, true),
            Field::new("name", DataType::Utf8, false),
            Field::new("org", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(StringArray::from(keys)),
            Arc::new(StringArray::from(names)),
            Arc::new(StringArray::from(orgs)),
        ],
    )
    .unwrap();
    let mut thing = write(&temp.path().join("thing.lance"), batch).await;
    thing
        .create_index_builder(&["key"], IndexType::BTree, &ScalarIndexParams::default())
        .name("key".to_string())
        .await
        .unwrap();
    let thing = Arc::new(thing);
    let seed = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("scope", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["seed"])),
            Arc::new(StringArray::from(vec!["allowed"])),
        ],
    )
    .unwrap();
    let edge = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("src", DataType::Utf8, false),
            Field::new("dst", DataType::Utf8, true),
            Field::new("audit", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["seed"; 7])),
            Arc::new(StringArray::from(vec![
                Some("n"),
                Some("n"),
                Some("seed"),
                Some("denied"),
                Some("ghost"),
                None,
                Some("n"),
            ])),
            Arc::new(StringArray::from(vec![
                "allowed", "allowed", "allowed", "allowed", "allowed", "allowed", "denied",
            ])),
        ],
    )
    .unwrap();
    let config = GraphConfig::builder()
        .with_node_label("seed", "id")
        .with_node_label("thing", "key")
        .with_relationship("links", "src", "dst")
        .build()
        .unwrap();
    let mut expected: Option<RecordBatch> = None;
    // A NULL producer remains unsupported; removing only that unmatched edge
    // exercises the real BTree path with the same exact ordered result.
    for producer_has_null in [false, true] {
        let edge = if producer_has_null {
            edge.clone()
        } else {
            arrow::compute::filter_record_batch(
                &edge,
                &arrow::compute::is_not_null(edge.column(1)).unwrap(),
            )
            .unwrap()
        };
        for enabled in [false, true, false] {
            let ctx = SessionContext::new();
            ctx.register_table(
                "thing",
                Arc::new(LanceTableProvider::new(thing.clone(), false, false)),
            )
            .unwrap();
            ctx.register_table(
                "seed",
                Arc::new(MemTable::try_new(seed.schema(), vec![vec![seed.clone()]]).unwrap()),
            )
            .unwrap();
            ctx.register_table(
                "links",
                Arc::new(MemTable::try_new(edge.schema(), vec![vec![edge.clone()]]).unwrap()),
            )
            .unwrap();
            let metrics = Arc::new(RuntimeIdFilterMetrics::default());
            let query = CypherQuery::new("MATCH (s:seed)-[r:links]->(t:thing) WHERE s.scope = 'allowed' AND r.audit = 'allowed' AND t.org = 'allowed' RETURN t.key AS key, t.name AS name ORDER BY name LIMIT 20").unwrap().with_config(config.clone());
            let query = if enabled {
                query.with_runtime_id_filters(metrics.clone())
            } else {
                query
            };
            let batch = query.execute_with_context_limited(ctx, 20).await.unwrap();
            assert_eq!(batch.num_rows(), 5);
            if let Some(expected) = &expected {
                assert_eq!(batch.schema(), expected.schema());
                assert_eq!(batch.columns(), expected.columns());
            } else {
                expected = Some(batch);
            }
            assert_eq!(
                metrics.index_scans.load(Ordering::Relaxed) > 0,
                enabled && !producer_has_null,
                "{metrics:?}"
            );
            assert_eq!(metrics.timeout_fallbacks.load(Ordering::Relaxed), 0);
            if enabled && producer_has_null {
                assert!(metrics.unsupported_fallbacks.load(Ordering::Relaxed) > 0);
            }
        }
    }
}
