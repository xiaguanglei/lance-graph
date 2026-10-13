// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Hot latency of the production read path: `GraphSnapshotProvider` over a
//! directory namespace, `CypherQuery::execute_with_snapshot`, warmed caches.
//! `cargo run --release -p lance-graph-benches --example graph_snapshot_latency -- 2000 20000 200000`
//! Set BENCH_COLUMN_CACHE_MB to enable the property column cache, ITERS for
//! the sample count. Every query is first checked against the JOIN plan
//! (schema and row bag) and must run natively.

use arrow_array::{BooleanArray, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance::{dataset::WriteParams, Dataset};
use lance_graph::{
    CsrAdjacencyCache, CsrScanOptions, CypherQuery, DirNamespace, GraphConfig,
    GraphSnapshotProvider,
};
use std::{
    sync::{atomic::Ordering, Arc},
    time::Instant,
};

const QUERIES: &[(&str, u32, &str)] = &[
    ("hop1_out", 1, "MATCH (a:Entity {entity_id:'e0'})-[r:REL]->(b:Entity) WHERE b.allowed = true RETURN b.entity_id, r.edge_id"),
    ("hop1_in_acl", 1, "MATCH (e:Entity {entity_id:'e1'})<-[r:MENTIONS]-(m:Memory) WHERE m.visible = true AND m.scope = 'user' RETURN m.memory_id, m.content, r.edge_id"),
    ("cooccurs2", 2, "MATCH (a:Entity)<-[r1:MENTIONS]-(m:Memory)-[r2:MENTIONS]->(b:Entity) WHERE a.entity_id IN ['e0'] AND b.entity_id <> a.entity_id AND m.visible = true AND m.scope = 'user' AND b.allowed = true RETURN b.entity_id, m.memory_id, m.content, r1.edge_id, r2.edge_id"),
    ("relation3", 3, "MATCH (a:Entity {entity_id:'e0'})-[r1:REL]->(b:Entity)-[r2:REL]->(c:Entity)-[r3:REL]->(d:Entity) WHERE b.allowed = true AND c.allowed = true AND d.allowed = true RETURN b.entity_id, c.entity_id, d.entity_id, r1.edge_id, r2.edge_id, r3.edge_id"),
    ("mention3", 3, "MATCH (m:Memory {memory_id:'m0'})-[:MENTIONS]->(a:Entity)-[:REL]->(b:Entity)-[:REL]->(c:Entity) WHERE c.allowed = true RETURN a.entity_id, b.entity_id, c.entity_id"),
];

async fn write(batch: RecordBatch, uri: &str) {
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        uri,
        Some(WriteParams::default()),
    )
    .await
    .unwrap();
}

fn strings(values: impl Iterator<Item = String>) -> Arc<StringArray> {
    Arc::new(StringArray::from(values.collect::<Vec<_>>()))
}

async fn generate(root: &str, n: usize) {
    write(
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("entity_id", DataType::Utf8, false),
                Field::new("allowed", DataType::Boolean, false),
            ])),
            vec![
                strings((0..n).map(|i| format!("e{i}"))),
                Arc::new(BooleanArray::from(
                    (0..n).map(|i| i % 10 != 9).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap(),
        &format!("{root}/Entity.lance"),
    )
    .await;
    write(
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("memory_id", DataType::Utf8, false),
                Field::new("visible", DataType::Boolean, false),
                Field::new("scope", DataType::Utf8, false),
                Field::new("content", DataType::Utf8, false),
            ])),
            vec![
                strings((0..n).map(|i| format!("m{i}"))),
                Arc::new(BooleanArray::from(
                    (0..n).map(|i| i % 5 != 4).collect::<Vec<_>>(),
                )),
                strings((0..n).map(|i| if i % 2 == 0 { "user" } else { "private" }.into())),
                strings((0..n).map(|i| format!("evidence_{i}_{}", "x".repeat(64)))),
            ],
        )
        .unwrap(),
        &format!("{root}/Memory.lance"),
    )
    .await;
    let mut edges = (vec![], vec![], vec![], vec![]);
    for i in 0..n {
        for delta in 0..2 {
            edges.0.push(format!("m{i}"));
            edges.1.push(format!("e{}", (i + delta) % n));
            edges.2.push("MENTIONS".to_string());
            edges.3.push(format!("mention_{i}_{delta}"));
        }
        for delta in 1..=2 {
            edges.0.push(format!("e{i}"));
            edges.1.push(format!("e{}", (i + delta) % n));
            edges.2.push("REL".to_string());
            edges.3.push(format!("relation_{i}_{delta}"));
        }
    }
    write(
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("source_id", DataType::Utf8, false),
                Field::new("target_id", DataType::Utf8, false),
                Field::new("relationship_type", DataType::Utf8, false),
                Field::new("edge_id", DataType::Utf8, false),
            ])),
            vec![
                strings(edges.0.into_iter()),
                strings(edges.1.into_iter()),
                strings(edges.2.into_iter()),
                strings(edges.3.into_iter()),
            ],
        )
        .unwrap(),
        &format!("{root}/graph_edges.lance"),
    )
    .await;
}

fn bag(batch: &RecordBatch) -> Vec<Vec<String>> {
    let mut rows = (0..batch.num_rows())
        .map(|row| {
            batch
                .columns()
                .iter()
                .map(|array| {
                    datafusion::arrow::util::display::array_value_to_string(array.as_ref(), row)
                        .unwrap()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    rows.sort();
    rows
}

#[tokio::main]
async fn main() {
    let sizes = std::env::args()
        .skip(1)
        .map(|value| value.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    let sizes = if sizes.is_empty() {
        vec![2_000, 20_000, 200_000]
    } else {
        sizes
    };
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let column_mb: usize = std::env::var("BENCH_COLUMN_CACHE_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    println!(
        "graph_snapshot_latency; execute_with_snapshot; column_cache_mb={column_mb}; profile={}",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    for n in sizes {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().to_str().unwrap().to_owned();
        generate(&root, n).await;
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
        let provider = GraphSnapshotProvider::new(Arc::new(DirNamespace::new(root)), config)
            .with_table("MENTIONS", "graph_edges")
            .with_table("REL", "graph_edges")
            .with_cache(Arc::new(CsrAdjacencyCache::new(CsrScanOptions {
                max_cache_bytes: 1 << 30,
                max_column_cache_bytes: column_mb << 20,
                ..Default::default()
            })))
            .with_max_staleness(std::time::Duration::from_secs(3600));
        let snapshot = provider.latest().await.unwrap();
        let started = Instant::now();
        let report = snapshot.warm().await;
        println!(
            "warm n={n} edges={} ms={:.1} {report:?}",
            n * 4,
            started.elapsed().as_secs_f64() * 1000.
        );
        for (name, hops, text) in QUERIES {
            let joins = CypherQuery::new(text)
                .unwrap()
                .with_native_multihop(false)
                .execute_with_snapshot(&snapshot)
                .await
                .unwrap();
            let first = CypherQuery::new(text).unwrap();
            let started = Instant::now();
            let native = first.execute_with_snapshot(&snapshot).await.unwrap();
            let first_ms = started.elapsed().as_secs_f64() * 1000.;
            assert_eq!(joins.schema(), native.schema(), "{name}");
            assert_eq!(bag(&joins), bag(&native), "{name}");
            for _ in 0..20 {
                CypherQuery::new(text)
                    .unwrap()
                    .execute_with_snapshot(&snapshot)
                    .await
                    .unwrap();
            }
            let mut samples = Vec::with_capacity(iters);
            let (mut native_runs, mut column_reads) = (0, 0);
            for _ in 0..iters {
                let query = CypherQuery::new(text).unwrap();
                let started = Instant::now();
                query.execute_with_snapshot(&snapshot).await.unwrap();
                samples.push(started.elapsed().as_secs_f64() * 1000.);
                let metrics = query.csr_adjacency_metrics();
                native_runs += metrics.native_multihop_executions.load(Ordering::Relaxed);
                column_reads += metrics.column_cache_reads.load(Ordering::Relaxed);
            }
            assert_eq!(native_runs, iters as u64, "{name} must run natively");
            samples.sort_by(f64::total_cmp);
            println!(
                "hot n={n} query={name} hops={hops} rows={} first_ms={first_ms:.3} p50_ms={:.3} p90_ms={:.3} join_ms_once={:.3} column_reads_avg={}",
                native.num_rows(),
                samples[iters / 2],
                samples[iters * 9 / 10],
                {
                    let started = Instant::now();
                    CypherQuery::new(text)
                        .unwrap()
                        .with_native_multihop(false)
                        .execute_with_snapshot(&snapshot)
                        .await
                        .unwrap();
                    started.elapsed().as_secs_f64() * 1000.
                },
                column_reads / iters as u64,
            );
        }
        println!("cache n={n} {:?}", provider.cache().stats());
    }
}
