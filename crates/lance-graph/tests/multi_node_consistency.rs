// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! End-to-end consistency of a multi-node service. Each "node" is an
//! independent `GraphSnapshotProvider` with its own CSR cache over the same
//! storage; writers commit through separate dataset handles.

use arrow_array::{Array, BooleanArray, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance::dataset::{optimize::compact_files, WriteMode, WriteParams};
use lance::Dataset;
use lance_graph::{
    CsrAdjacencyCache, CsrScanOptions, CypherQuery, GraphConfig, GraphSnapshot,
    GraphSnapshotProvider, GraphVersions,
};
use lance_graph_catalog::DirNamespace;
use std::{
    collections::BTreeMap,
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

const EDGES: &str = "graph_edges";

/// Queries covering one, two and three hops, anonymous relationships,
/// aggregation, DISTINCT and ordering.
const QUERIES: &[&str] = &[
    "MATCH (m:Memory {memory_id:'m0'})-[r:MENTIONS]->(e:Entity) RETURN e.entity_id, r.edge_id",
    "MATCH (e:Entity {entity_id:'e1'})<-[r:MENTIONS]-(m:Memory) WHERE m.visible = true RETURN m.memory_id, m.content, r.edge_id",
    "MATCH (a:Entity)<-[:MENTIONS]-(m:Memory)-[:MENTIONS]->(b:Entity) WHERE a.entity_id IN ['e0', 'e2'] AND b.entity_id <> a.entity_id RETURN a.entity_id, m.memory_id, b.entity_id",
    "MATCH (a:Entity {entity_id:'e0'})-[r1:REL]->(b:Entity)-[r2:REL]->(c:Entity)-[r3:REL]->(d:Entity) WHERE d.allowed = true RETURN b.entity_id, c.entity_id, d.entity_id, r3.edge_id",
    "MATCH (m:Memory)-[:MENTIONS]->(e:Entity) WHERE m.memory_id IN ['m0', 'm1', 'm2', 'm3'] RETURN e.entity_id, COUNT(*) AS mentions",
    "MATCH (a:Entity)-[:REL]->(b:Entity) WHERE a.entity_id IN ['e0', 'e1', 'e2'] RETURN DISTINCT b.entity_id",
    "MATCH (a:Entity)-[r:REL]->(b:Entity) RETURN a.entity_id, b.entity_id, r.edge_id",
];

struct Graph {
    _temp: tempfile::TempDir,
    root: String,
    config: GraphConfig,
}

impl Graph {
    fn uri(&self, table: &str) -> String {
        format!("{}/{table}.lance", self.root)
    }

    fn provider(&self, cache: Arc<CsrAdjacencyCache>) -> GraphSnapshotProvider {
        GraphSnapshotProvider::new(
            Arc::new(DirNamespace::new(self.root.clone())),
            self.config.clone(),
        )
        .with_table("MENTIONS", EDGES)
        .with_table("REL", EDGES)
        .with_cache(cache)
    }

    /// Committed versions as a writer would report them after its commits.
    async fn versions(&self) -> GraphVersions {
        let mut versions = GraphVersions::new();
        for table in ["Entity", "Memory", EDGES] {
            let dataset = Dataset::open(&self.uri(table)).await.unwrap();
            versions = versions.with(table, dataset.version_id());
        }
        versions
    }

    async fn append(&self, table: &str, batch: RecordBatch) -> u64 {
        write(&self.uri(table), batch, WriteMode::Append).await
    }

    async fn delete(&self, table: &str, predicate: &str) -> u64 {
        let mut dataset = Dataset::open(&self.uri(table)).await.unwrap();
        dataset.delete(predicate).await.unwrap();
        dataset.version_id()
    }
}

async fn write(uri: &str, batch: RecordBatch, mode: WriteMode) -> u64 {
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        uri,
        Some(WriteParams {
            mode,
            max_rows_per_file: 64,
            ..Default::default()
        }),
    )
    .await
    .unwrap()
    .version_id()
}

fn entities(ids: &[(&str, bool)]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("entity_id", DataType::Utf8, false),
            Field::new("allowed", DataType::Boolean, false),
        ])),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            )),
            Arc::new(BooleanArray::from(
                ids.iter().map(|(_, allowed)| *allowed).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

fn memories(ids: &[(&str, bool)]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("memory_id", DataType::Utf8, false),
            Field::new("visible", DataType::Boolean, false),
            Field::new("content", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            )),
            Arc::new(BooleanArray::from(
                ids.iter().map(|(_, visible)| *visible).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                ids.iter()
                    .map(|(id, _)| format!("content of {id}"))
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

/// (source, target, type, edge id)
fn edges(rows: &[(&str, &str, &str, &str)]) -> RecordBatch {
    let column = |index: usize| {
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| match index {
                    0 => row.0,
                    1 => row.1,
                    2 => row.2,
                    _ => row.3,
                })
                .collect::<Vec<_>>(),
        ))
    };
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("source_id", DataType::Utf8, false),
            Field::new("target_id", DataType::Utf8, false),
            Field::new("relationship_type", DataType::Utf8, false),
            Field::new("edge_id", DataType::Utf8, false),
        ])),
        vec![column(0), column(1), column(2), column(3)],
    )
    .unwrap()
}

async fn graph() -> Graph {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().to_str().unwrap().to_owned();
    let entity_rows = (0..12)
        .map(|i| (format!("e{i}"), i % 4 != 3))
        .collect::<Vec<_>>();
    let memory_rows = (0..10)
        .map(|i| (format!("m{i}"), i % 3 != 2))
        .collect::<Vec<_>>();
    write(
        &format!("{root}/Entity.lance"),
        entities(
            &entity_rows
                .iter()
                .map(|(id, allowed)| (id.as_str(), *allowed))
                .collect::<Vec<_>>(),
        ),
        WriteMode::Create,
    )
    .await;
    write(
        &format!("{root}/Memory.lance"),
        memories(
            &memory_rows
                .iter()
                .map(|(id, visible)| (id.as_str(), *visible))
                .collect::<Vec<_>>(),
        ),
        WriteMode::Create,
    )
    .await;
    let mut rows = Vec::new();
    for i in 0..10 {
        for d in 0..3 {
            rows.push((
                format!("m{i}"),
                format!("e{}", (i + d) % 12),
                "MENTIONS",
                format!("men_{i}_{d}"),
            ));
        }
    }
    // A parallel MENTIONS edge, as produced by repeated extraction.
    rows.push(("m0".into(), "e0".into(), "MENTIONS", "men_0_0_dup".into()));
    for i in 0..12 {
        for d in 1..=2 {
            rows.push((
                format!("e{i}"),
                format!("e{}", (i + d) % 12),
                "REL",
                format!("rel_{i}_{d}"),
            ));
        }
    }
    let rows = rows
        .iter()
        .map(|(s, t, ty, id)| (s.as_str(), t.as_str(), *ty, id.as_str()))
        .collect::<Vec<_>>();
    write(
        &format!("{root}/{EDGES}.lance"),
        edges(&rows),
        WriteMode::Create,
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
    Graph {
        _temp: temp,
        root,
        config,
    }
}

fn rows(batch: &RecordBatch) -> Vec<Vec<Option<String>>> {
    (0..batch.num_rows())
        .map(|row| {
            batch
                .columns()
                .iter()
                .map(|array| {
                    (!array.is_null(row)).then(|| {
                        arrow::util::display::array_value_to_string(array.as_ref(), row).unwrap()
                    })
                })
                .collect()
        })
        .collect()
}

fn bag(batch: &RecordBatch) -> Vec<Vec<Option<String>>> {
    let mut rows = rows(batch);
    rows.sort();
    rows
}

fn query(text: &str, native: bool) -> CypherQuery {
    CypherQuery::new(text).unwrap().with_native_multihop(native)
}

/// Results of every query: the JOIN plan is the oracle for the native path.
async fn results(snapshot: &GraphSnapshot) -> Vec<Vec<Vec<Option<String>>>> {
    let mut all = Vec::new();
    for text in QUERIES {
        let native = query(text, true)
            .execute_with_snapshot(snapshot)
            .await
            .unwrap();
        let joins = query(text, false)
            .execute_with_snapshot(snapshot)
            .await
            .unwrap();
        assert_eq!(native.schema(), joins.schema(), "{text}");
        assert_eq!(
            bag(&native),
            bag(&joins),
            "{text} at {:?}",
            snapshot.versions()
        );
        all.push(bag(&native));
    }
    all
}

fn mentions(memory: &str, entities: &[&str], tag: &str) -> RecordBatch {
    let ids = entities
        .iter()
        .map(|entity| format!("{tag}_{memory}_{entity}"))
        .collect::<Vec<_>>();
    edges(
        &entities
            .iter()
            .zip(&ids)
            .map(|(entity, id)| (memory, *entity, "MENTIONS", id.as_str()))
            .collect::<Vec<_>>(),
    )
}

#[tokio::test]
async fn read_your_writes_and_bounded_staleness_across_nodes() {
    let graph = graph().await;
    let stale = Duration::from_secs(3600);
    let node_a = graph
        .provider(Arc::new(CsrAdjacencyCache::default()))
        .with_max_staleness(stale);
    let node_b = graph
        .provider(Arc::new(CsrAdjacencyCache::default()))
        .with_max_staleness(stale);
    let fresh = graph
        .provider(Arc::new(CsrAdjacencyCache::default()))
        .with_max_staleness(Duration::ZERO);
    let before = node_b.latest().await.unwrap();
    node_a.latest().await.unwrap();
    // Write through node A: the new memory first, then its edges.
    graph.append("Memory", memories(&[("m_new", true)])).await;
    graph
        .append(EDGES, mentions("m_new", &["e1", "e2"], "new"))
        .await;
    let token = GraphVersions::from_token(&graph.versions().await.to_token()).unwrap();
    let text = "MATCH (e:Entity {entity_id:'e1'})<-[r:MENTIONS]-(m:Memory) RETURN m.memory_id";
    let sees_write = |batch: &RecordBatch| {
        rows(batch)
            .iter()
            .any(|row| row[0].as_deref() == Some("m_new"))
    };
    // Within its staleness bound node B still serves the pinned old versions.
    let stale_snapshot = node_b.latest().await.unwrap();
    assert!(Arc::ptr_eq(&before, &stale_snapshot));
    let result = query(text, true)
        .execute_with_snapshot(&stale_snapshot)
        .await
        .unwrap();
    assert!(!sees_write(&result));
    // The client's token forces a refresh on whichever node serves it.
    let snapshot = node_b.at_least(&token).await.unwrap();
    assert!(snapshot.versions().covers(&token));
    let result = query(text, true)
        .execute_with_snapshot(&snapshot)
        .await
        .unwrap();
    assert!(sees_write(&result));
    // Monotonic: later reads on node B do not go back.
    assert!(node_b.latest().await.unwrap().versions().covers(&token));
    // A zero staleness bound observes the write without a token.
    let snapshot = fresh.latest().await.unwrap();
    assert!(sees_write(
        &query(text, true)
            .execute_with_snapshot(&snapshot)
            .await
            .unwrap()
    ));
    // Versions that no node can see yet are an error, not a silent stale read.
    let future = token.clone().with(EDGES, token.get(EDGES).unwrap() + 100);
    assert!(node_b.at_least(&future).await.is_err());
}

#[tokio::test]
async fn pinned_versions_reproduce_results_after_later_writes() {
    let graph = graph().await;
    let node = graph.provider(Arc::new(CsrAdjacencyCache::default()));
    let first = node.refresh().await.unwrap();
    let token = first.versions().to_token();
    let expected = results(&first).await;
    // Appends, deletes, an update (delete + insert) and compaction on every table.
    graph.append("Entity", entities(&[("e12", true)])).await;
    graph.append("Memory", memories(&[("m10", true)])).await;
    graph
        .append(EDGES, mentions("m10", &["e0", "e12"], "late"))
        .await;
    graph
        .delete(EDGES, "edge_id = 'men_1_0' OR edge_id = 'rel_0_1'")
        .await;
    graph.delete("Memory", "memory_id = 'm3'").await;
    graph.append("Memory", memories(&[("m3", false)])).await;
    let mut dataset = Dataset::open(&graph.uri(EDGES)).await.unwrap();
    compact_files(&mut dataset, Default::default(), None)
        .await
        .unwrap();
    let latest = node.refresh().await.unwrap();
    assert_ne!(latest.versions(), first.versions());
    let changed = results(&latest).await;
    assert_ne!(changed, expected);
    // Any node can reproduce the old read exactly from the token.
    let pinned = node
        .at(&GraphVersions::from_token(&token).unwrap())
        .await
        .unwrap();
    assert_eq!(results(&pinned).await, expected);
    let other = graph.provider(Arc::new(CsrAdjacencyCache::default()));
    let pinned = other
        .at(&GraphVersions::from_token(&token).unwrap())
        .await
        .unwrap();
    assert_eq!(results(&pinned).await, expected);
    // Tokens must name every table for an exact read.
    let partial = GraphVersions::new().with(EDGES, 1);
    assert!(node.at(&partial).await.is_err());
}

#[tokio::test]
async fn nodes_agree_on_rows_and_pages_for_the_same_versions() {
    let graph = graph().await;
    graph.append(EDGES, mentions("m4", &["e9"], "extra")).await;
    let token = graph.versions().await;
    // Node 1 runs native paths, node 2 cannot build adjacency (JOIN plans),
    // node 3 starts cold and evicts constantly.
    let nodes = [
        graph.provider(Arc::new(CsrAdjacencyCache::default())),
        graph.provider(Arc::new(CsrAdjacencyCache::new(CsrScanOptions {
            max_edges: 1,
            ..Default::default()
        }))),
        graph.provider(Arc::new(CsrAdjacencyCache::new(CsrScanOptions {
            max_cache_slots: 1,
            ..Default::default()
        }))),
        // Node 4 serves node/edge properties from its column cache.
        graph.provider(Arc::new(CsrAdjacencyCache::new(CsrScanOptions {
            max_column_cache_bytes: 32 << 20,
            ..Default::default()
        }))),
    ];
    let mut snapshots = Vec::new();
    for node in &nodes {
        snapshots.push(node.at_least(&token).await.unwrap());
    }
    let reference = results(&snapshots[0]).await;
    for snapshot in &snapshots[1..] {
        assert_eq!(snapshot.versions(), snapshots[0].versions());
        assert_eq!(results(snapshot).await, reference);
    }
    let native = query(QUERIES[0], true);
    native.execute_with_snapshot(&snapshots[0]).await.unwrap();
    assert_eq!(
        native
            .csr_adjacency_metrics()
            .native_multihop_executions
            .load(Ordering::Relaxed),
        1
    );
    let joins = query(QUERIES[0], true);
    joins.execute_with_snapshot(&snapshots[1]).await.unwrap();
    assert_eq!(
        joins
            .csr_adjacency_metrics()
            .native_multihop_executions
            .load(Ordering::Relaxed),
        0
    );
    // Deterministic order: identical row order on every node and plan, and
    // SKIP/LIMIT pages that tile the full result without overlap.
    for text in [
        QUERIES[4],
        QUERIES[2],
        QUERIES[6],
        "MATCH (a:Entity)-[r:REL]->(b:Entity) RETURN a.entity_id, b.entity_id ORDER BY a.entity_id",
        "MATCH (m:Memory)-[:MENTIONS]->(e:Entity) RETURN e.entity_id, COUNT(*) AS mentions ORDER BY mentions DESC",
    ] {
        let ordered = |native: bool| {
            CypherQuery::new(text)
                .unwrap()
                .with_native_multihop(native)
                .with_deterministic_order(true)
        };
        let full = ordered(true)
            .execute_with_snapshot(&snapshots[0])
            .await
            .unwrap();
        assert!(full.num_rows() > 3, "{text}");
        // The caller's ORDER BY stays primary; added keys only break ties.
        let full_rows = rows(&full);
        if text.ends_with("ORDER BY a.entity_id") {
            assert!(full_rows.windows(2).all(|w| w[0][0] <= w[1][0]), "{text}");
        }
        if text.ends_with("ORDER BY mentions DESC") {
            let count = |row: &Vec<Option<String>>| row[1].as_ref().unwrap().parse::<u64>().unwrap();
            assert!(full_rows.windows(2).all(|w| count(&w[0]) >= count(&w[1])), "{text}");
        }
        for (snapshot, native) in snapshots.iter().zip([true, false, true, true]) {
            let other = ordered(native).execute_with_snapshot(snapshot).await.unwrap();
            assert_eq!(rows(&other), rows(&full), "{text}");
        }
        let mut paged = Vec::new();
        for page in 0..full.num_rows().div_ceil(3) {
            let paged_text = format!("{text} SKIP {} LIMIT 3", page * 3);
            let snapshot = &snapshots[page % snapshots.len()];
            let batch = CypherQuery::new(&paged_text)
                .unwrap()
                .with_deterministic_order(true)
                .execute_with_snapshot(snapshot)
                .await
                .unwrap();
            paged.extend(rows(&batch));
        }
        assert_eq!(paged, rows(&full), "{text}");
    }
}

#[tokio::test]
async fn edges_written_before_their_nodes_stay_hidden_until_the_node_exists() {
    let graph = graph().await;
    let node = graph
        .provider(Arc::new(CsrAdjacencyCache::default()))
        .with_max_staleness(Duration::ZERO);
    // Wrong write order: the edge commits first.
    graph
        .append(EDGES, mentions("m_late", &["e5"], "early"))
        .await;
    let text =
        "MATCH (e:Entity {entity_id:'e5'})<-[r:MENTIONS]-(m:Memory) RETURN m.memory_id, r.edge_id";
    let snapshot = node.latest().await.unwrap();
    let before = query(text, true)
        .execute_with_snapshot(&snapshot)
        .await
        .unwrap();
    assert!(rows(&before)
        .iter()
        .all(|row| row[0].as_deref() != Some("m_late")));
    results(&snapshot).await;
    graph.append("Memory", memories(&[("m_late", true)])).await;
    let snapshot = node.latest().await.unwrap();
    let after = query(text, true)
        .execute_with_snapshot(&snapshot)
        .await
        .unwrap();
    assert!(rows(&after)
        .iter()
        .any(|row| row[0].as_deref() == Some("m_late")));
    results(&snapshot).await;
}

#[tokio::test]
async fn concurrent_writers_on_two_nodes_are_both_visible() {
    let graph = graph().await;
    let node = graph
        .provider(Arc::new(CsrAdjacencyCache::default()))
        .with_max_staleness(Duration::ZERO);
    let first = node.latest().await.unwrap();
    first.warm().await;
    let (a, b) = tokio::join!(
        graph.append(EDGES, mentions("m1", &["e11"], "writer_a")),
        graph.append(EDGES, mentions("m2", &["e11"], "writer_b")),
    );
    assert_ne!(a, b);
    let snapshot = node.latest().await.unwrap();
    let text = "MATCH (e:Entity {entity_id:'e11'})<-[r:MENTIONS]-(m:Memory) RETURN r.edge_id";
    let edges = rows(
        &query(text, true)
            .execute_with_snapshot(&snapshot)
            .await
            .unwrap(),
    )
    .into_iter()
    .map(|row| row[0].clone().unwrap())
    .collect::<Vec<_>>();
    assert!(edges.contains(&"writer_a_m1_e11".to_owned()), "{edges:?}");
    assert!(edges.contains(&"writer_b_m2_e11".to_owned()), "{edges:?}");
    results(&snapshot).await;
    // The new versions were derived from the warmed ones, not rebuilt.
    assert!(node.cache().stats().incremental_builds >= 1);
}

#[tokio::test]
async fn expired_versions_never_return_wrong_rows_and_refresh_recovers() {
    let graph = graph().await;
    let node = graph.provider(Arc::new(CsrAdjacencyCache::default()));
    let old = node.refresh().await.unwrap();
    let old_results = results(&old).await;
    graph.append(EDGES, mentions("m9", &["e3"], "after")).await;
    graph.delete(EDGES, "edge_id = 'men_2_1'").await;
    let mut dataset = Dataset::open(&graph.uri(EDGES)).await.unwrap();
    compact_files(&mut dataset, Default::default(), None)
        .await
        .unwrap();
    // Retention shorter than a reader's handle: old data files are removed.
    let dataset = Dataset::open(&graph.uri(EDGES)).await.unwrap();
    dataset
        .cleanup_old_versions(Default::default(), Some(true), Some(false))
        .await
        .unwrap();
    // A fresh node pins the expired version: it must fail, or (when served
    // entirely from caches) return exactly the old rows, never other rows.
    let reader = graph.provider(Arc::new(CsrAdjacencyCache::default()));
    for text in QUERIES {
        let Ok(snapshot) = reader.at(old.versions()).await else {
            continue;
        };
        if let Ok(batch) = query(text, true).execute_with_snapshot(&snapshot).await {
            let index = QUERIES.iter().position(|q| q == text).unwrap();
            assert_eq!(bag(&batch), old_results[index], "{text}");
        }
    }
    let latest = node.refresh().await.unwrap();
    assert_ne!(latest.versions(), old.versions());
    results(&latest).await;
}

#[tokio::test]
async fn warm_prepares_every_table_before_the_first_query() {
    let graph = graph().await;
    let node = graph.provider(Arc::new(CsrAdjacencyCache::default()));
    let snapshot = node.latest().await.unwrap();
    let report = snapshot.warm().await;
    assert_eq!(report.adjacency, 2); // MENTIONS and REL share one edge table
    assert_eq!(report.node_ids, 2);
    assert_eq!(report.skipped, 0);
    let builds = node.cache().stats().full_builds;
    for text in QUERIES {
        let q = query(text, true);
        q.execute_with_snapshot(&snapshot).await.unwrap();
        let metrics = q.csr_adjacency_metrics();
        assert_eq!(metrics.build_count.load(Ordering::Relaxed), 0, "{text}");
        assert_eq!(
            metrics
                .native_multihop_node_key_builds
                .load(Ordering::Relaxed),
            0,
            "{text}"
        );
    }
    assert_eq!(node.cache().stats().full_builds, builds);
    // After a write, warming the new versions derives them incrementally.
    graph.append(EDGES, mentions("m5", &["e7"], "post")).await;
    let next = node.refresh().await.unwrap();
    let report = next.warm().await;
    assert_eq!(report.skipped, 0);
    assert_eq!(node.cache().stats().full_builds, builds);
    assert!(node.cache().stats().incremental_builds >= 1);
}

#[tokio::test]
async fn restarted_nodes_return_the_same_rows() {
    let graph = graph().await;
    graph.delete("Entity", "entity_id = 'e4'").await;
    let token = graph.versions().await;
    let warm = graph.provider(Arc::new(CsrAdjacencyCache::default()));
    let snapshot = warm.at(&token).await.unwrap();
    snapshot.warm().await;
    let expected = results(&snapshot).await;
    // A replacement node with an empty cache and new handles.
    let restarted = graph.provider(Arc::new(CsrAdjacencyCache::default()));
    let snapshot = restarted.at(&token).await.unwrap();
    assert_eq!(results(&snapshot).await, expected);
    assert_eq!(snapshot.versions(), &token);
}

#[test]
fn version_tokens_round_trip_merge_and_cover() {
    let a = GraphVersions::new().with("Entity", 3).with(EDGES, 7);
    let b = GraphVersions::new().with("Entity", 5).with("Memory", 2);
    let token = a.to_token();
    assert_eq!(GraphVersions::from_token(&token).unwrap(), a);
    assert!(GraphVersions::from_token("not a token").is_err());
    let mut merged = a.clone();
    merged.merge_max(&b);
    assert_eq!(
        merged.iter().collect::<BTreeMap<_, _>>(),
        BTreeMap::from([("Entity", 5), ("Memory", 2), (EDGES, 7)])
    );
    assert!(merged.covers(&a) && merged.covers(&b));
    assert!(!a.covers(&b));
    assert!(a.covers(&GraphVersions::new()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_queries_and_writes_on_one_node_stay_exact() {
    let graph = Arc::new(graph().await);
    // A small cache forces evictions and rebuilds while queries run.
    let node = Arc::new(
        graph
            .provider(Arc::new(CsrAdjacencyCache::new(CsrScanOptions {
                max_cache_slots: 3,
                // Room for about one column: constant column evictions too.
                max_column_cache_bytes: 4 << 10,
                ..Default::default()
            })))
            .with_max_staleness(Duration::ZERO),
    );
    let writer = {
        let graph = graph.clone();
        tokio::spawn(async move {
            for i in 0..6 {
                let memory = format!("m_w{i}");
                graph
                    .append("Memory", memories(&[(&memory, i % 2 == 0)]))
                    .await;
                graph
                    .append(EDGES, mentions(&memory, &["e0", "e1"], &format!("w{i}")))
                    .await;
                graph
                    .delete(EDGES, &format!("edge_id = 'men_{}_1'", i % 10))
                    .await;
            }
        })
    };
    let readers = (0..6)
        .map(|reader| {
            let node = node.clone();
            tokio::spawn(async move {
                for round in 0..6 {
                    let snapshot = node.latest().await.unwrap();
                    let text = QUERIES[(reader + round) % QUERIES.len()];
                    let native = query(text, true).execute_with_snapshot(&snapshot).await.unwrap();
                    let joins = query(text, false).execute_with_snapshot(&snapshot).await.unwrap();
                    assert_eq!(bag(&native), bag(&joins), "{text} at {:?}", snapshot.versions());
                    // Parameters bind the seed the same way on both plans.
                    let text = "MATCH (e:Entity)<-[r:MENTIONS]-(m:Memory) WHERE e.entity_id = $id RETURN m.memory_id, r.edge_id";
                    let run = |native: bool| {
                        query(text, native).with_parameter("id", format!("e{}", reader % 2))
                    };
                    let native = run(true).execute_with_snapshot(&snapshot).await.unwrap();
                    let joins = run(false).execute_with_snapshot(&snapshot).await.unwrap();
                    assert_eq!(bag(&native), bag(&joins), "{:?}", snapshot.versions());
                }
            })
        })
        .collect::<Vec<_>>();
    writer.await.unwrap();
    for reader in readers {
        reader.await.unwrap();
    }
    let latest = node.refresh().await.unwrap();
    results(&latest).await;
    assert!(node.cache().stats().entries <= 3);
}

#[tokio::test]
async fn missing_tables_are_reported_not_silently_empty() {
    let graph = graph().await;
    let node = graph
        .provider(Arc::new(CsrAdjacencyCache::default()))
        .with_table("REL", "no_such_table");
    assert!(node.latest().await.is_err());
    let node = graph.provider(Arc::new(CsrAdjacencyCache::default()));
    let token = graph.versions().await;
    let unknown = token.clone().with("Entity", 999);
    assert!(node.at(&unknown).await.is_err());
    assert!(node.at(&token).await.is_ok());
}

/// Schema changes that keep data files (renames, drop + re-add, a swapped
/// pair of columns that reverses every edge), rewrites (update, overwrite),
/// restore and compaction: long-lived nodes that derive caches incrementally
/// (with and without the column cache, one constantly evicting) return the
/// same rows as a cold node and as the JOIN plan at every version.
#[tokio::test]
async fn schema_changes_and_rewrites_stay_exact_on_every_node() {
    use lance::dataset::{ColumnAlteration, NewColumnTransform, UpdateBuilder};
    let graph = graph().await;
    let nodes = [
        graph.provider(Arc::new(CsrAdjacencyCache::default())),
        graph.provider(Arc::new(CsrAdjacencyCache::new(CsrScanOptions {
            max_column_cache_bytes: 32 << 20,
            ..Default::default()
        }))),
        graph.provider(Arc::new(CsrAdjacencyCache::new(CsrScanOptions {
            max_column_cache_bytes: 2 << 10,
            max_cache_slots: 2,
            ..Default::default()
        }))),
    ];
    let open = |table: &str| {
        let uri = graph.uri(table);
        async move { Dataset::open(&uri).await.unwrap() }
    };
    let rename = |from: &str, to: &str| ColumnAlteration::new(from.into()).rename(to.into());
    let add = |name: &str, expr: &str| {
        NewColumnTransform::SqlExpressions(vec![(name.to_owned(), expr.to_owned())])
    };
    let memory_v1 = open("Memory").await.version_id();
    let mut previous: Option<Vec<Vec<Vec<Option<String>>>>> = None;
    for step in 0..9 {
        match step {
            0 => {}
            // New `content` under the old name; the old data stays as another field.
            1 => {
                let mut memory = open("Memory").await;
                memory
                    .alter_columns(&[rename("content", "content_old")])
                    .await
                    .unwrap();
                memory
                    .add_columns(add("content", "concat('new ', memory_id)"), None, None)
                    .await
                    .unwrap();
            }
            // Swap two returned columns' names: metadata only, same files.
            2 => {
                let mut memory = open("Memory").await;
                for (from, to) in [
                    ("content", "tmp"),
                    ("content_old", "content"),
                    ("tmp", "content_old"),
                ] {
                    memory.alter_columns(&[rename(from, to)]).await.unwrap();
                }
            }
            // Drop and re-add a filtered column with different values.
            3 => {
                let mut memory = open("Memory").await;
                memory.drop_columns(&["visible"]).await.unwrap();
                memory
                    .add_columns(add("visible", "memory_id > 'm4'"), None, None)
                    .await
                    .unwrap();
            }
            // Invert `allowed` without rewriting any existing file.
            4 => {
                let mut entity = open("Entity").await;
                entity
                    .add_columns(add("blocked", "NOT allowed"), None, None)
                    .await
                    .unwrap();
                entity.drop_columns(&["allowed"]).await.unwrap();
                entity
                    .alter_columns(&[rename("blocked", "allowed")])
                    .await
                    .unwrap();
            }
            // Swap endpoint names: every edge reverses, data files unchanged.
            5 => {
                let mut edges = open(EDGES).await;
                for (from, to) in [
                    ("source_id", "tmp"),
                    ("target_id", "source_id"),
                    ("tmp", "target_id"),
                ] {
                    edges.alter_columns(&[rename(from, to)]).await.unwrap();
                }
            }
            // Rewrite REL edge ids, then compact and delete.
            6 => {
                let edges = open(EDGES).await;
                UpdateBuilder::new(Arc::new(edges))
                    .update_where("relationship_type = 'REL'")
                    .unwrap()
                    .set("edge_id", "concat(edge_id, '_u')")
                    .unwrap()
                    .build()
                    .unwrap()
                    .execute()
                    .await
                    .unwrap();
                let mut edges = open(EDGES).await;
                compact_files(&mut edges, Default::default(), None)
                    .await
                    .unwrap();
                edges.delete("edge_id = 'rel_0_1_u'").await.unwrap();
            }
            // Overwrite entities with the same ids and other flags.
            7 => {
                let rows = (0..12)
                    .map(|i| (format!("e{i}"), i % 2 == 0))
                    .collect::<Vec<_>>();
                write(
                    &graph.uri("Entity"),
                    entities(
                        &rows
                            .iter()
                            .map(|(id, allowed)| (id.as_str(), *allowed))
                            .collect::<Vec<_>>(),
                    ),
                    WriteMode::Overwrite,
                )
                .await;
            }
            // Swap the endpoints back (the original field layout, whose
            // entries the long-lived nodes may still hold) and restore the
            // original memory table as a new version.
            8 => {
                let mut edges = open(EDGES).await;
                for (from, to) in [
                    ("source_id", "tmp"),
                    ("target_id", "source_id"),
                    ("tmp", "target_id"),
                ] {
                    edges.alter_columns(&[rename(from, to)]).await.unwrap();
                }
                let mut memory = open("Memory")
                    .await
                    .checkout_version(memory_v1)
                    .await
                    .unwrap();
                memory.restore().await.unwrap();
            }
            _ => unreachable!(),
        }
        let token = graph.versions().await;
        let cold = graph
            .provider(Arc::new(CsrAdjacencyCache::default()))
            .at(&token)
            .await
            .unwrap();
        let expected = results(&cold).await;
        for (index, node) in nodes.iter().enumerate() {
            let snapshot = node.at_least(&token).await.unwrap();
            assert_eq!(snapshot.versions(), &token, "step {step} node {index}");
            assert_eq!(
                results(&snapshot).await,
                expected,
                "step {step} node {index}"
            );
        }
        if step > 0 {
            assert_ne!(
                previous.as_ref(),
                Some(&expected),
                "step {step} must change results"
            );
        }
        previous = Some(expected);
    }
    // The column cache was used, incrementally, across these versions.
    let stats = nodes[1].cache().stats();
    assert!(stats.column_entries > 0, "{stats:?}");
}
