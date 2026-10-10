# Lance Graph Benchmarks

Performance benchmarks for the `lance-graph` crate.

## Running Benchmarks

From the repository root:

```bash
# Run all benchmarks
cargo bench -p lance-graph-benches

# Run specific benchmark
cargo bench -p lance-graph-benches --bench graph_execution

# In-memory adjacency/filtered reachability only (no Cypher or storage I/O)
cargo bench -p lance-graph-benches --bench csr_adjacency -- --warm-up-time 1
```

## Benchmarks

- **graph_execution**: End-to-end query execution benchmarks
  - Basic node filtering and projection
  - Single-hop relationship expansion
  - Two-hop relationship expansion
  - Tests with datasets of varying sizes (100, 10K, 1M rows)
- **csr_adjacency**: Snapshot-local adjacency and filtered BFS microbenchmarks
  - Dense 20K / 200K vertices, four identity-preserving outgoing edges per vertex
  - Neighbor iteration and edge allow-mask filtering
  - Three-hop vertex reachability with/without edge predicates
  - Construction is outside timing; no claim of Cypher path equivalence or
    end-to-end speedup. The caller owns the snapshot binding and prepared mask.
  - BFS includes its per-call O(vertices) visited allocation/clearing; report
    that cost separately from edge predicate iteration and storage query cost.

## Note

This crate is not published to crates.io and is excluded from releases.
It exists solely for performance testing during development.
