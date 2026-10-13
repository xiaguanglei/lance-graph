// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Storage-backed, nonempty fixed-path A/B/A, not AMS/LoCoMo or model latency.
//! `cargo run --release -p lance-graph-benches --example native_csr_multihop -- [--id-btree] 2000 20000 200000`
//! Both arms use the same data/version/lock/profile and existing single-hop CSR;
//! only the native-path flag differs. Covers one-, two- and three-hop seeded
//! paths plus shapes that must keep the JOIN plan (unseeded/target-bound,
//! variable length). A query must run natively exactly when the library
//! reports a seeded candidate shape; set BENCH_OBSERVE_ADMISSION=1 to measure
//! older revisions that admit fewer shapes or miss node-ID BTrees, and
//! BENCH_QUERIES=a,b to run a subset. Every timed run still checks the exact
//! schema and result bag. Temporary datasets are scoped by RAII.

use arrow_array::{Array, BooleanArray, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use datafusion::{datasource::DefaultTableSource, execution::context::SessionContext};
use lance::{datafusion::LanceTableProvider, index::DatasetIndexExt, Dataset};
use lance_graph::{CsrAdjacencyCache, CsrScanOptions, CypherQuery, GraphConfig, InMemoryCatalog};
use lance_index::{scalar::ScalarIndexParams, IndexType};
use std::{
    collections::BTreeMap,
    sync::{atomic::Ordering, Arc},
    time::Instant,
};

async fn write(batch: RecordBatch, path: &std::path::Path) -> Arc<Dataset> {
    Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
            path.to_str().unwrap(),
            None,
        )
        .await
        .unwrap(),
    )
}

fn bag(batch: &RecordBatch) -> BTreeMap<Vec<Option<String>>, usize> {
    let mut result = BTreeMap::new();
    for row in 0..batch.num_rows() {
        let key = batch
            .columns()
            .iter()
            .map(|array| {
                if array.is_null(row) {
                    None
                } else {
                    Some(arrow_value(array.as_ref(), row))
                }
            })
            .collect();
        *result.entry(key).or_insert(0) += 1;
    }
    result
}

fn arrow_value(array: &dyn Array, row: usize) -> String {
    array
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("benchmark projects UTF8")
        .value(row)
        .into()
}

#[tokio::main]
async fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let indexed = args.first().is_some_and(|arg| arg == "--id-btree");
    let sizes = args
        .iter()
        .skip(usize::from(indexed))
        .map(|value| value.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    let sizes = if sizes.is_empty() {
        vec![20_000, 200_000]
    } else {
        sizes
    };
    println!(
        "synthetic_local_lance; timed_includes_parse_plan_mask_traverse_collect; profile={}",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    for n in sizes {
        assert!(n >= 10 && n <= 200_000);
        let temp = tempfile::tempdir().unwrap();
        let mut entities = write(
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("entity_id", DataType::Utf8, false),
                    Field::new("allowed", DataType::Boolean, false),
                ])),
                vec![
                    Arc::new(StringArray::from(
                        (0..n).map(|i| format!("e{i}")).collect::<Vec<_>>(),
                    )),
                    Arc::new(BooleanArray::from(
                        (0..n).map(|i| i % 10 != 9).collect::<Vec<_>>(),
                    )),
                ],
            )
            .unwrap(),
            &temp.path().join("entities.lance"),
        )
        .await;
        let mut memories = write(
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("memory_id", DataType::Utf8, false),
                    Field::new("visible", DataType::Boolean, false),
                    Field::new("scope", DataType::Utf8, false),
                    Field::new("content", DataType::Utf8, false),
                ])),
                vec![
                    Arc::new(StringArray::from(
                        (0..n).map(|i| format!("m{i}")).collect::<Vec<_>>(),
                    )),
                    Arc::new(BooleanArray::from(
                        (0..n).map(|i| i % 5 != 4).collect::<Vec<_>>(),
                    )),
                    Arc::new(StringArray::from(
                        (0..n)
                            .map(|i| if i % 2 == 0 { "user" } else { "private" })
                            .collect::<Vec<_>>(),
                    )),
                    Arc::new(StringArray::from(
                        (0..n)
                            .map(|i| format!("evidence_{i}_{}", "x".repeat(64)))
                            .collect::<Vec<_>>(),
                    )),
                ],
            )
            .unwrap(),
            &temp.path().join("memories.lance"),
        )
        .await;
        if indexed {
            let start = Instant::now();
            Arc::get_mut(&mut entities)
                .unwrap()
                .create_index_builder(
                    &["entity_id"],
                    IndexType::BTree,
                    &ScalarIndexParams::default(),
                )
                .name("entity_id_btree".to_owned())
                .await
                .unwrap();
            Arc::get_mut(&mut memories)
                .unwrap()
                .create_index_builder(
                    &["memory_id"],
                    IndexType::BTree,
                    &ScalarIndexParams::default(),
                )
                .name("memory_id_btree".to_owned())
                .await
                .unwrap();
            println!(
                "index_build n={n} ms={:.3} entity={:?} memory={:?}",
                start.elapsed().as_secs_f64() * 1000.,
                entities.load_indices().await.unwrap(),
                memories.load_indices().await.unwrap()
            );
        }
        let mut sources = Vec::with_capacity(n * 4);
        let mut targets = Vec::with_capacity(n * 4);
        let mut types = Vec::with_capacity(n * 4);
        let mut ids = Vec::with_capacity(n * 4);
        for i in 0..n {
            for delta in 0..2 {
                sources.push(format!("m{i}"));
                targets.push(format!("e{}", (i + delta) % n));
                types.push("MENTIONS");
                ids.push(format!("mention_{i}_{delta}"));
            }
            for delta in 1..=2 {
                sources.push(format!("e{i}"));
                targets.push(format!("e{}", (i + delta) % n));
                types.push("REL");
                ids.push(format!("relation_{i}_{delta}"));
            }
        }
        let edges = write(
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("source_id", DataType::Utf8, false),
                    Field::new("target_id", DataType::Utf8, false),
                    Field::new("relationship_type", DataType::Utf8, false),
                    Field::new("edge_id", DataType::Utf8, false),
                ])),
                vec![
                    Arc::new(StringArray::from(sources)),
                    Arc::new(StringArray::from(targets)),
                    Arc::new(StringArray::from(types)),
                    Arc::new(StringArray::from(ids)),
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
        let source = Arc::new(DefaultTableSource::new(Arc::new(LanceTableProvider::new(
            edges.clone(),
            false,
            false,
        ))));
        let catalog = Arc::new(
            InMemoryCatalog::new()
                .with_node_source(
                    "Entity",
                    Arc::new(DefaultTableSource::new(Arc::new(LanceTableProvider::new(
                        entities.clone(),
                        false,
                        false,
                    )))),
                )
                .with_node_source(
                    "Memory",
                    Arc::new(DefaultTableSource::new(Arc::new(LanceTableProvider::new(
                        memories.clone(),
                        false,
                        false,
                    )))),
                )
                .with_relationship_source("MENTIONS", source.clone())
                .with_relationship_source("REL", source),
        );
        let cache_options = CsrScanOptions {
            max_bytes: 512 * 1024 * 1024,
            max_cache_bytes: 512 * 1024 * 1024,
            ..Default::default()
        };
        let cache = Arc::new(CsrAdjacencyCache::new(cache_options.clone()));
        println!(
            "data n={n} entity_rows={n} memory_rows={n} edge_rows={} versions={}/{}/{} node_id_btree={indexed} cache_budget_mib=512",
            n * 4, memories.version_id(), entities.version_id(), edges.version_id()
        );
        let seeds = (0..10)
            .map(|i| format!("'m{i}'"))
            .collect::<Vec<_>>()
            .join(", ");
        let hop1_seeds = format!("MATCH (m:Memory)-[:MENTIONS]->(e:Entity) WHERE m.memory_id IN [{seeds}] AND m.visible = true RETURN m.memory_id, e.entity_id");
        // (name, hops, seeded, text); unseeded shapes must keep the JOIN plan.
        for (name, hops, seeded, text) in [
            ("hop1_out", 1, true, "MATCH (a:Entity {entity_id:'e0'})-[r:REL]->(b:Entity) WHERE b.allowed = true RETURN b.entity_id, r.edge_id"),
            ("hop1_in_acl", 1, true, "MATCH (e:Entity {entity_id:'e1'})<-[r:MENTIONS]-(m:Memory) WHERE m.visible = true AND m.scope = 'user' RETURN m.memory_id, m.content, r.edge_id"),
            ("hop1_10seeds", 1, true, hop1_seeds.as_str()),
            ("cooccurs2", 2, true, "MATCH (a:Entity)<-[r1:MENTIONS]-(m:Memory)-[r2:MENTIONS]->(b:Entity) WHERE a.entity_id IN ['e0'] AND b.entity_id <> a.entity_id AND m.visible = true AND m.scope = 'user' AND b.allowed = true RETURN b.entity_id, m.memory_id, m.content, r1.edge_id, r2.edge_id"),
            ("relation2", 2, true, "MATCH (a:Entity {entity_id:'e0'})-[r1:REL]->(b:Entity)-[r2:REL]->(c:Entity) WHERE c.allowed = true RETURN b.entity_id, c.entity_id, r2.edge_id"),
            ("relation3", 3, true, "MATCH (a:Entity {entity_id:'e0'})-[r1:REL]->(b:Entity)-[r2:REL]->(c:Entity)-[r3:REL]->(d:Entity) WHERE b.allowed = true AND c.allowed = true AND d.allowed = true RETURN b.entity_id, c.entity_id, d.entity_id, r1.edge_id, r2.edge_id, r3.edge_id"),
            ("mention3", 3, true, "MATCH (m:Memory {memory_id:'m0'})-[:MENTIONS]->(a:Entity)-[:REL]->(b:Entity)-[:REL]->(c:Entity) WHERE c.allowed = true RETURN a.entity_id, b.entity_id, c.entity_id"),
            ("hop1_target_bound", 1, false, "MATCH (a:Entity)-[r:REL]->(b:Entity {entity_id:'e5'}) RETURN a.entity_id, r.edge_id"),
            ("varlen1_2", 2, false, "MATCH (a:Entity {entity_id:'e0'})-[:REL*1..2]->(b:Entity) RETURN b.entity_id"),
        ] {
            // Optional comma-separated subset, e.g. BENCH_QUERIES=hop1_out,relation3.
            if std::env::var("BENCH_QUERIES").is_ok_and(|names| !names.split(',').any(|q| q == name)) {
                continue;
            }
            let make = |native| CypherQuery::new(text).unwrap().with_config(config.clone()).with_shared_csr_cache(cache.clone()).with_native_multihop(native);
            let expect_native = seeded && make(true).native_multihop_candidate();
            let baseline = make(false).with_shared_csr_cache(Arc::new(CsrAdjacencyCache::new(cache_options.clone()))).execute_with_catalog_and_context(catalog.clone(), SessionContext::new()).await.unwrap();
            assert!(baseline.num_rows() > 0, "{name}");
            let start = Instant::now(); let cold = make(true);
            let first = cold.execute_with_catalog_and_context(catalog.clone(), SessionContext::new()).await.unwrap();
            assert_eq!(baseline.schema(), first.schema()); assert_eq!(bag(&baseline), bag(&first));
            // Older revisions admit fewer shapes than their hint reports; with
            // BENCH_OBSERVE_ADMISSION=1 record the actual admission instead.
            let actual = cold.csr_adjacency_metrics().native_multihop_executions.load(Ordering::Relaxed) == 1;
            let observe = std::env::var_os("BENCH_OBSERVE_ADMISSION").is_some();
            let expect_native = if observe { expect_native && actual } else { expect_native };
            assert_eq!(actual, expect_native, "native admission/execution of {name}: {:?}", cold.csr_adjacency_metrics());
            println!("cold n={n} query={name} hops={hops} ms={:.3} builds={} rows={}", start.elapsed().as_secs_f64()*1000., cold.csr_adjacency_metrics().build_count.load(Ordering::Relaxed), first.num_rows());
            for (arm, native) in [("old1", false), ("native", true), ("old2", false)] {
                for _ in 0..5 { make(native).execute_with_catalog_and_context(catalog.clone(), SessionContext::new()).await.unwrap(); }
                let mut samples = Vec::new(); let mut inputs = 0; let mut bytes = 0; let mut checks = 0; let mut index_reads = 0; let mut take_reads = 0;
                for _ in 0..20 {
                    let start = Instant::now(); let q = make(native);
                    let result = q.execute_with_catalog_and_context(catalog.clone(), SessionContext::new()).await.unwrap();
                    samples.push(start.elapsed().as_secs_f64()*1000.);
                    assert_eq!(baseline.schema(), result.schema()); assert_eq!(bag(&baseline), bag(&result));
                    let metrics = q.csr_adjacency_metrics();
                    assert_eq!(metrics.native_multihop_executions.load(Ordering::Relaxed), u64::from(native && expect_native));
                    assert_eq!(metrics.native_multihop_fallbacks.load(Ordering::Relaxed), 0);
                    // Nodes come from the resident ID map, or from a node-ID BTree
                    // scan when the map does not fit; never from an unindexed scan.
                    if native && expect_native && !observe { assert!(metrics.native_multihop_node_take_reads.load(Ordering::Relaxed) + metrics.native_multihop_node_index_reads.load(Ordering::Relaxed) >= 1, "{name} node reads (btree={indexed}): {metrics:?}"); }
                    index_reads += metrics.native_multihop_node_index_reads.load(Ordering::Relaxed); take_reads += metrics.native_multihop_node_take_reads.load(Ordering::Relaxed);
                    inputs += metrics.native_multihop_input_rows.load(Ordering::Relaxed); bytes += metrics.native_multihop_input_bytes.load(Ordering::Relaxed); checks += metrics.native_multihop_edge_checks.load(Ordering::Relaxed);
                }
                let raw = samples.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>().join(",");
                samples.sort_by(f64::total_cmp);
                println!("hot n={n} query={name} hops={hops} arm={arm} p50_ms={:.3} p95_ms={:.3} rows={} native={} fallback=0 input_rows_avg={} arrow_bytes_avg={} edge_checks_avg={} node_id_btree_reads_avg={} node_id_take_reads_avg={} raw_ms=[{raw}]", (samples[9]+samples[10])/2., samples[18], baseline.num_rows(), native && expect_native, inputs / 20, bytes / 20, checks / 20, index_reads / 20, take_reads / 20);
            }
        }
    }
}
