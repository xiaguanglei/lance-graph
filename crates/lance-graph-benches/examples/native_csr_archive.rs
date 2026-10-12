// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Read-only A/B/A on existing AMS-processed archive datasets. This is a graph
//! component benchmark, not complete AMS recall or an independent 20K gold set.
//! Args: cases.json warehouse1 [warehouse2 ...]. No metadata/data is mutated.

use arrow_array::{Array, Float32Array, RecordBatch};
use datafusion::{datasource::DefaultTableSource, execution::context::SessionContext};
use lance::{datafusion::LanceTableProvider, Dataset};
use lance_graph::{CsrAdjacencyCache, CypherQuery, GraphConfig, InMemoryCatalog};
use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    sync::{atomic::Ordering, Arc},
    time::Instant,
};

fn bag(batch: &RecordBatch) -> BTreeMap<Vec<Option<String>>, usize> {
    let mut result = BTreeMap::new();
    for row in 0..batch.num_rows() {
        let key = batch
            .columns()
            .iter()
            .map(|array| {
                if array.is_null(row) {
                    None
                } else if let Some(values) = array.as_any().downcast_ref::<Float32Array>() {
                    Some(format!("f32:{:08x}", values.value(row).to_bits()))
                } else if let Some(values) =
                    array.as_any().downcast_ref::<arrow_array::StringArray>()
                {
                    Some(values.value(row).to_owned())
                } else {
                    panic!("unhandled benchmark output type: {:?}", array.data_type())
                }
            })
            .collect();
        *result.entry(key).or_insert(0) += 1;
    }
    result
}

#[tokio::main]
async fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    assert!(args.len() >= 2, "cases.json warehouse1 [warehouse2 ...]");
    let cases: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&args[0]).unwrap()).unwrap();
    let case = cases["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "derived_cooccurs_acl_nonempty")
        .unwrap();
    let text = case["request"]["query"].as_str().unwrap();
    let params: HashMap<String, serde_json::Value> =
        serde_json::from_value(case["request"]["parameters"].clone()).unwrap();
    println!("read_only_processed_AMS_archive fixed_2hop complete_case_ACL same_lock_profile A/B/A; profile={}", if cfg!(debug_assertions) { "debug" } else { "release" });
    for root in &args[1..] {
        let open = |name| async move {
            Arc::new(
                Dataset::open(
                    Path::new(root)
                        .join(format!("{name}.lance"))
                        .to_str()
                        .unwrap(),
                )
                .await
                .unwrap(),
            )
        };
        let memories = open("memories").await;
        let entities = open("entities").await;
        let edges = open("memory_graph_edges").await;
        let mut config = GraphConfig::builder()
            .with_node_label("Entity", "entity_id")
            .with_node_label("Memory", "memory_id")
            .with_relationship("MENTIONS", "source_id", "target_id")
            .build()
            .unwrap();
        config
            .relationship_mappings
            .get_mut("mentions")
            .unwrap()
            .type_field = Some("relationship_type".into());
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
                .with_relationship_source(
                    "MENTIONS",
                    Arc::new(DefaultTableSource::new(Arc::new(LanceTableProvider::new(
                        edges.clone(),
                        false,
                        false,
                    )))),
                ),
        );
        println!(
            "asset={root} rows={}/{}/{} versions={}/{}/{}",
            memories.count_rows(None).await.unwrap(),
            entities.count_rows(None).await.unwrap(),
            edges.count_rows(None).await.unwrap(),
            memories.version_id(),
            entities.version_id(),
            edges.version_id()
        );
        for (name, ds) in [("M", &memories), ("E", &entities), ("G", &edges)] {
            use lance::index::DatasetIndexExt;
            println!("asset_index {name} {:?}", ds.load_indices().await.unwrap());
        }
        let cache = Arc::new(CsrAdjacencyCache::default());
        let make = |native| {
            CypherQuery::new(text)
                .unwrap()
                .with_config(config.clone())
                .with_parameters(params.clone())
                .with_shared_csr_cache(cache.clone())
                .with_native_multihop(native)
        };
        let baseline = make(false)
            .with_shared_csr_cache(Arc::new(CsrAdjacencyCache::default()))
            .execute_with_catalog_and_context(catalog.clone(), SessionContext::new())
            .await
            .unwrap();
        assert!(baseline.num_rows() > 0, "frozen case must be nonempty");
        let start = Instant::now();
        let cold = make(true);
        let first = cold
            .execute_with_catalog_and_context(catalog.clone(), SessionContext::new())
            .await
            .unwrap();
        assert_eq!(baseline.schema(), first.schema());
        assert_eq!(bag(&baseline), bag(&first));
        assert_eq!(
            cold.csr_adjacency_metrics()
                .native_multihop_executions
                .load(Ordering::Relaxed),
            1,
            "{:?}",
            cold.csr_adjacency_metrics()
        );
        println!(
            "cold ms={:.3} metrics={:?}",
            start.elapsed().as_secs_f64() * 1000.,
            cold.csr_adjacency_metrics()
        );
        for (arm, native) in [("old1", false), ("native", true), ("old2", false)] {
            for _ in 0..10 {
                make(native)
                    .execute_with_catalog_and_context(catalog.clone(), SessionContext::new())
                    .await
                    .unwrap();
            }
            let mut samples = Vec::new();
            let mut rows = 0;
            let mut bytes = 0;
            let mut checks = 0;
            for _ in 0..40 {
                let start = Instant::now();
                let q = make(native);
                let result = q
                    .execute_with_catalog_and_context(catalog.clone(), SessionContext::new())
                    .await
                    .unwrap();
                samples.push(start.elapsed().as_secs_f64() * 1000.);
                assert_eq!(baseline.schema(), result.schema());
                assert_eq!(bag(&baseline), bag(&result));
                let metrics = q.csr_adjacency_metrics();
                assert_eq!(
                    metrics.native_multihop_executions.load(Ordering::Relaxed),
                    u64::from(native)
                );
                assert_eq!(metrics.native_multihop_fallbacks.load(Ordering::Relaxed), 0);
                rows += metrics.native_multihop_input_rows.load(Ordering::Relaxed);
                bytes += metrics.native_multihop_input_bytes.load(Ordering::Relaxed);
                checks += metrics.native_multihop_edge_checks.load(Ordering::Relaxed);
            }
            let raw = samples
                .iter()
                .map(|value| format!("{value:.3}"))
                .collect::<Vec<_>>()
                .join(",");
            samples.sort_by(f64::total_cmp);
            println!("hot arm={arm} p50_ms={:.3} p95_ms={:.3} output_rows={} native={native} fallback=0 input_rows_avg={} arrow_bytes_avg={} edge_checks_avg={} raw_ms=[{raw}]", (samples[19]+samples[20])/2., samples[37], baseline.num_rows(), rows/40, bytes/40, checks/40);
        }
        assert_eq!(
            Dataset::open(memories.uri()).await.unwrap().version_id(),
            memories.version_id()
        );
        assert_eq!(
            Dataset::open(entities.uri()).await.unwrap().version_id(),
            entities.version_id()
        );
        assert_eq!(
            Dataset::open(edges.uri()).await.unwrap().version_id(),
            edges.version_id()
        );
    }
}
