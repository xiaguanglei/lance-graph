# Lance Graph Query Engine

A graph query engine for Lance datasets with Cypher syntax support. This crate enables querying Lance's columnar datasets using familiar graph query patterns, interpreting tabular data as property graphs.

## Features

- Cypher query parsing and AST construction
- Graph configuration for mapping Lance tables to nodes and relationships
- Semantic validation with typed `GraphError` diagnostics
- Pluggable execution strategies (DataFusion planner by default, Lance Native placeholder)
- Async query execution that returns Arrow `RecordBatch` results
- JSON-serializable parameter binding for reusable query templates
- Logical plan debugging via `CypherQuery::explain`

## Quick Start

```rust
use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance_graph::{CypherQuery, ExecutionStrategy, GraphConfig};

let config = GraphConfig::builder()
    .with_node_label("Person", "person_id")
    .with_relationship("KNOWS", "src_person_id", "dst_person_id")
    .build()?;

let schema = Arc::new(Schema::new(vec![
    Field::new("person_id", DataType::Int32, false),
    Field::new("name", DataType::Utf8, false),
    Field::new("age", DataType::Int32, false),
]));
let batch = RecordBatch::try_new(
    schema,
    vec![
        Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
        Arc::new(StringArray::from(vec!["Alice", "Bob"])) as ArrayRef,
        Arc::new(Int32Array::from(vec![29, 35])) as ArrayRef,
    ],
)?;

let mut tables = HashMap::new();
tables.insert("Person".to_string(), batch);

let query = CypherQuery::new("MATCH (p:Person) WHERE p.age > $min RETURN p.name")?
    .with_config(config)
    .with_parameter("min", 30);

let runtime = tokio::runtime::Runtime::new()?;
// Use default DataFusion-based execution
let result = runtime.block_on(query.execute(tables, None))?;
```

The query expects a `HashMap<String, RecordBatch>` keyed by the labels and relationship types referenced in the Cypher text. Each record batch should expose the columns configured through `GraphConfig` (ID fields, property fields, etc.). Relationship mappings also expect a batch keyed by the relationship type (for example `KNOWS`) that contains the configured source/target ID columns and any optional property columns.

## Configuring Graph Mappings

Graph mappings are declared with `GraphConfig::builder()`:

```rust
use lance_graph::{GraphConfig, NodeMapping, RelationshipMapping};

let config = GraphConfig::builder()
    .with_node_label("Person", "person_id")
    .with_relationship("KNOWS", "src_person_id", "dst_person_id")
    .build()?;
```

For finer control, build `NodeMapping` and `RelationshipMapping` instances explicitly:

```rust
let person = NodeMapping::new("Person", "person_id")
    .with_properties(vec!["name".into(), "age".into()])
    .with_filter("kind = 'person'");

let knows = RelationshipMapping::new("KNOWS", "src_person_id", "dst_person_id")
    .with_properties(vec!["since".into()]);

let config = GraphConfig::builder()
    .with_node_mapping(person)
    .with_relationship_mapping(knows)
    .build()?;
```

## Executing Cypher Queries

- `CypherQuery::new` parses Cypher text into the internal AST.
- `with_config` attaches the graph configuration used for validation and execution.
- `with_parameter` / `with_parameters` bind JSON-serializable values that can be referenced as `$param` in the Cypher text.
- `execute` is asynchronous and returns an Arrow `RecordBatch`. Pass `None` to use the default DataFusion planner. `ExecutionStrategy::LanceNative` is reserved for future native execution support and currently errors.
- `explain` is asynchronous and returns a formatted string containing the graph logical plan alongside the DataFusion logical and physical plans.

All queries use the DataFusion planner for optimization and execution.

### Optional runtime node-ID filters

`CypherQuery::with_runtime_id_filters(Arc<RuntimeIdFilterMetrics>)` opts in to
consuming completed inner-JOIN membership filters on indexed Lance node scans.
Node ID fields come from `GraphConfig`; only UTF-8 IDs with a snapshot-local
BTree index participate. The original JOIN and all predicates remain in place.

```rust,ignore
use lance_graph::{CypherQuery, RuntimeIdFilterMetrics};
use std::sync::Arc;

let metrics = Arc::new(RuntimeIdFilterMetrics::default());
let query = CypherQuery::new("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN b.name")?
    .with_config(config)
    .with_runtime_id_filters(metrics.clone());
let result = query.execute_with_context(ctx).await?;
```

The default is off. Static ID predicates retain the existing native index path.
Non-Lance sources, relationship scans, unsupported producers/expressions,
ordered or limited source scans, and membership above 150 IDs or 128 KiB keep
the original scan. An incomplete producer has a shared 100 ms deadline per
source before falling back to a fresh original scan; the wait remains inside
the caller's query future and cancellation budget. Counters retain no IDs.
This does not enable the CSR foundation or the Lance Native executor.

### CSR adjacency cache and native paths

Seeded fixed paths of one to three hops over Lance tables
(`MATCH (a:Entity {id:'x'})-[:REL]->(b) ...`, `WHERE a.id IN [...]`) run as a
single CSR traversal instead of per-hop JOINs when the node and edge key
columns share one type (`Utf8`, `Int64` or `UInt64`); every other shape, and any path
whose topology does not fit the limits, plans the ordinary JOINs. Results are
identical either way. Share one cache per caller security context so snapshots
are reused across queries:

```rust,ignore
use lance_graph::{CsrAdjacencyCache, CsrScanOptions, CypherQuery};
use std::sync::Arc;

let cache = Arc::new(CsrAdjacencyCache::new(CsrScanOptions {
    max_edges: 5_000_000,                    // largest edge table to index
    max_bytes: 512 << 20,                    // one cold build: ~64 B/edge
    max_cache_bytes: 1 << 30,                // all cached versions: ~32 B/edge
    max_cache_rows: 10_000_000,
    ..Default::default()
}));
// After a write, optionally prepare the new version before queries need it.
cache.warm_adjacency(edges_dataset.clone(), "source_id", "target_id").await;
let query = CypherQuery::new(text)?.with_config(config).with_shared_csr_cache(cache.clone());
```

- Entries are keyed by the committed version's identity, so reopening a
  dataset reuses its snapshot. A new version is derived from a cached one by
  rescanning only appended or rewritten fragments and hiding deleted rows; a
  large change (over a quarter of the table), stable row IDs or a schema change
  rebuild instead.
- Least recently used entries that no running query holds are evicted to stay
  within `max_cache_bytes`, `max_cache_rows` and `max_cache_slots`.
- A cold build runs once per version without blocking queries on other
  entries. `cache.stats()` reports usage, full/incremental builds, evictions
  and rejected builds; `query.csr_adjacency_metrics()` reports per-query
  native executions, fallbacks, builds and build time.
- `max_column_cache_bytes` (default 0, off) additionally keeps the property
  columns that native traversals read in memory, per fragment and per version,
  under its own budget; unchanged fragments are shared across versions. It
  roughly halves hot latency because Lance has no data page cache. Prefill
  with `cache.warm_columns(dataset, &["col", ...])`; `stats().column_entries`
  / `column_bytes` and the per-query `column_cache_reads` show its use.
- Multi-node services should read through `GraphSnapshotProvider` (one per
  node and security context): it pins every table to one version per query,
  serves read-your-writes via `GraphVersions` tokens (`at_least`) and exact
  reads (`at`), and owns the node's shared cache. Use
  `CypherQuery::with_deterministic_order(true)` for paginated or limited
  results. See `docs/multi_node_deployment.md` for the full server-side
  protocol (write order, staleness, retention, monitoring).

A builder (`CypherQueryBuilder`) is also available for constructing queries programmatically without parsing text.

## Supported Cypher Surface

- Node patterns `(:Label)` with optional variables.
- Relationship patterns with fixed direction and type, including multi-hop paths.
- Property comparisons against literal values with `AND`/`OR`/`NOT`/`EXISTS`.
- RETURN lists of property accesses, optional `DISTINCT`, `ORDER BY`, `SKIP` (offset), and `LIMIT`.
- Positional and named parameters (e.g. `$min_age`).

Basic aggregations like `COUNT` are supported. Optional matches and subqueries are parsed but not executed yet.

## Crate Layout

- `ast` – Cypher AST definitions.
- `parser` – Nom-based Cypher parser.
- `semantic` – Lightweight semantic checks on the AST.
- `logical_plan` – Builders for graph logical plans.
- `datafusion_planner` – DataFusion-based execution planning.
- `config` – Graph configuration types and builders.
- `query` – High level `CypherQuery` API and runtime.
- `error` – `GraphError` and result helpers.
- `namespace` – Namespace helpers (re-exported from `lance-graph-catalog`).
- `source_catalog` – Catalog helpers for looking up table metadata (re-exported from `lance-graph-catalog`).

`lance-graph` re-exports the catalog and namespace types from the `lance-graph-catalog` crate for
API compatibility. You can depend on `lance-graph-catalog` directly if you only need catalog or
namespace utilities.

## Error Handling

Most APIs return `Result<T, GraphError>`. Errors include parsing failures, missing mappings, and execution issues surfaced from DataFusion.

## Testing

```bash
cargo test -p lance-graph
```

## Benchmarks

See the repository root `README.md` for benchmark setup, run commands, and report locations.

## Python Bindings

See the Python package docs for setup and development:

- Python package README: `python/README.md`
- Runnable examples (from repo root): `examples/README.md`

## License

Apache-2.0. See the top-level LICENSE file for details.
