// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! CSR (Compressed Sparse Row) adjacency index for native graph traversal
//!
//! Instead of translating graph traversals into SQL joins, this module provides
//! a CSR-based adjacency index that enables O(1) neighbor lookup. Inspired by
//! [GraphAr](https://graphar.apache.org)'s approach of encoding CSR offset
//! tables alongside columnar edge data.
//!
//! # Layout
//!
//! ```text
//! Offset Array:  [0, 3, 5, 5, 9, ...]   (one entry per vertex + 1 sentinel)
//! Neighbor Array: [2, 5, 7, 1, 4, 0, 3, 6, 8, ...]  (destination vertex IDs)
//! ```
//!
//! For vertex `v`, its neighbors are `neighbors[offsets[v]..offsets[v+1]]`.
//!
//! Edge identities are kept alongside the neighbor array so traversal predicates
//! can resolve edge properties without losing parallel edges. The default identity
//! is the input ordinal, **not** a Lance row ID; callers may supply an opaque
//! `u64` identity instead. Mixing explicit IDs and ordinals can produce equal
//! IDs; callers must provide a consistent identity domain for property lookup.
//! These identities are meaningful only with the edge data / snapshot used to
//! build the index. In particular, a Lance row address must not be reused after
//! compaction or a dataset version change. This module neither owns a dataset
//! snapshot nor persists an index, and is not the default Cypher executor.
//! Vertices use dense `u64` IDs; external UUID/string IDs must first be mapped
//! into that domain. Predicates are synchronous and should use already prepared
//! properties or an allow-mask from the same snapshot, not perform per-edge I/O.
//!
//! # Filtered reachability
//!
//! ```
//! use lance_graph::CsrIndexBuilder;
//!
//! let index = CsrIndexBuilder::new()
//!     .with_num_vertices(3)
//!     .try_add_edge_with_id(10, 0, 1)?
//!     .try_add_edge_with_id(20, 0, 1)? // separate parallel edge
//!     .try_add_edge_with_id(30, 1, 2)?
//!     .try_build()?;
//! // Resolve conditions once from the same snapshot; this is a generic allow-mask.
//! let allowed = |id, _, _| id != 10;
//! assert_eq!(index.bfs_filtered(0, 2, allowed), [vec![0], vec![1], vec![2]]);
//! # Ok::<(), lance_graph::error::GraphError>(())
//! ```

use arrow_array::{Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

use crate::error::{GraphError, Result};

/// In-memory CSR adjacency index for fast neighbor lookup.
///
/// Stores graph topology and edge identity in three aligned arrays:
/// - `offsets[v]` = start position of vertex v's neighbors in the neighbor array
/// - `neighbors[offsets[v]..offsets[v+1]]` = destination vertex IDs
/// - `edge_ids[offsets[v]..offsets[v+1]]` = identities for those edge entries
#[derive(Debug, Clone)]
pub struct CsrIndex {
    offsets: Vec<u64>,
    neighbors: Vec<u64>,
    edge_ids: Vec<u64>,
    num_vertices: u64,
}

impl CsrIndex {
    /// Look up all neighbors of a vertex. Returns an empty slice for vertices
    /// with no outgoing edges or vertex IDs beyond the index range.
    pub fn neighbors(&self, vertex_id: u64) -> &[u64] {
        &self.neighbors[self.edge_range(vertex_id)]
    }

    fn edge_range(&self, vertex_id: u64) -> std::ops::Range<usize> {
        let Ok(v) = usize::try_from(vertex_id) else {
            return 0..0;
        };
        if v >= self.offsets.len() - 1 {
            return 0..0;
        }
        self.offsets[v] as usize..self.offsets[v + 1] as usize
    }

    /// Return `(edge_id, destination)` pairs for outgoing edges, preserving
    /// parallel edges and input order within each source vertex.
    ///
    /// An invalid vertex returns an empty iterator. Identities refer to the
    /// original build snapshot, not to positions in this sorted CSR array.
    pub fn neighbor_edges(&self, vertex_id: u64) -> impl Iterator<Item = (u64, u64)> + '_ {
        let range = self.edge_range(vertex_id);
        self.edge_ids[range.clone()]
            .iter()
            .copied()
            .zip(self.neighbors[range].iter().copied())
    }

    /// Lazily filter outgoing edges using `predicate(edge_id, source, destination)`.
    ///
    /// The predicate may consult properties from the same build snapshot. It is
    /// evaluated separately for every parallel edge; rejected edges are not
    /// silently replaced by another edge to the same destination.
    /// Source/destination are traversal endpoints: in an incoming index they
    /// are reversed relative to the original edge row. Use `edge_id` to resolve
    /// that original row's properties.
    pub fn filtered_neighbors<'a, F>(
        &'a self,
        vertex_id: u64,
        mut predicate: F,
    ) -> impl Iterator<Item = (u64, u64)> + 'a
    where
        F: FnMut(u64, u64, u64) -> bool + 'a,
    {
        self.neighbor_edges(vertex_id)
            .filter(move |&(edge_id, dst)| predicate(edge_id, vertex_id, dst))
    }

    /// Return the out-degree of a vertex (number of outgoing edges).
    pub fn degree(&self, vertex_id: u64) -> u32 {
        self.neighbors(vertex_id).len() as u32
    }

    /// Return the total number of vertices in the index.
    pub fn num_vertices(&self) -> u64 {
        self.num_vertices
    }

    /// Return the total number of edges in the index.
    pub fn num_edges(&self) -> u64 {
        self.neighbors.len() as u64
    }

    /// Export the CSR index as an Arrow RecordBatch (offset table).
    ///
    /// Schema: `vertex_id: u64, offset: u64, degree: u64`
    ///
    /// This can be persisted as a Lance dataset for later loading.
    pub fn to_record_batch(&self) -> Result<RecordBatch> {
        let n = self.num_vertices as usize;
        let vertex_ids: Vec<u64> = (0..n as u64).collect();
        let offsets: Vec<u64> = self.offsets[..n].to_vec();
        let degrees: Vec<u64> = (0..n)
            .map(|i| self.offsets[i + 1] - self.offsets[i])
            .collect();

        let schema = Arc::new(Schema::new(vec![
            Field::new("vertex_id", DataType::UInt64, false),
            Field::new("offset", DataType::UInt64, false),
            Field::new("degree", DataType::UInt64, false),
        ]));

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vertex_ids)),
                Arc::new(UInt64Array::from(offsets)),
                Arc::new(UInt64Array::from(degrees)),
            ],
        )
        .map_err(|e| GraphError::PlanError {
            message: format!("Failed to create CSR offset RecordBatch: {}", e),
            location: snafu::Location::new(file!(), line!(), column!()),
        })
    }

    /// Export the neighbor (destination) array as an Arrow RecordBatch.
    ///
    /// Schema: `dst_id: u64`
    ///
    /// This is the edge list sorted by source vertex, suitable for sequential scans.
    pub fn neighbors_to_record_batch(&self) -> Result<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "dst_id",
            DataType::UInt64,
            false,
        )]));

        RecordBatch::try_new(
            schema,
            vec![Arc::new(UInt64Array::from(self.neighbors.clone()))],
        )
        .map_err(|e| GraphError::PlanError {
            message: format!("Failed to create CSR neighbors RecordBatch: {}", e),
            location: snafu::Location::new(file!(), line!(), column!()),
        })
    }

    /// Export edge identity separately without changing the legacy exports.
    ///
    /// Schema: `src_id: u64, dst_id: u64, edge_id: u64`. Rows are sorted by source,
    /// with parallel edges preserved. This batch alone does not bind identities
    /// to a dataset URI/version; callers must retain that snapshot information.
    pub fn edges_to_record_batch(&self) -> Result<RecordBatch> {
        let mut sources = Vec::with_capacity(self.neighbors.len());
        for src in 0..self.num_vertices {
            sources.extend(std::iter::repeat_n(src, self.neighbors(src).len()));
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("src_id", DataType::UInt64, false),
            Field::new("dst_id", DataType::UInt64, false),
            Field::new("edge_id", DataType::UInt64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(sources)),
                Arc::new(UInt64Array::from(self.neighbors.clone())),
                Arc::new(UInt64Array::from(self.edge_ids.clone())),
            ],
        )
        .map_err(|e| csr_error(format!("Failed to create CSR edges RecordBatch: {e}")))
    }

    /// Perform a k-hop BFS traversal from a starting vertex.
    ///
    /// Returns all vertices reachable within `max_hops` hops, organized by
    /// distance. Each entry in the returned Vec is the set of vertices at
    /// that hop distance (index 0 = starting vertex, index 1 = 1-hop neighbors, etc.).
    pub fn bfs(&self, start: u64, max_hops: u32) -> Vec<Vec<u64>> {
        self.bfs_filtered(start, max_hops, |_, _, _| true)
    }

    /// BFS over edges accepted by `predicate(edge_id, source, destination)`.
    ///
    /// Edge validation happens **before** marking a destination visited. Thus a
    /// rejected parallel edge cannot prevent an accepted edge from reaching it.
    /// Vertices are returned once, at their shortest accepted-edge hop distance.
    /// Invalid starts return no levels; `max_hops == 0` returns only the start.
    /// This is vertex reachability, not Cypher path enumeration or evidence
    /// aggregation: predicates must not depend on a particular path history.
    /// Predicates must be stable throughout this traversal. Source/destination
    /// are the traversal direction, including for an incoming index.
    pub fn bfs_filtered<F>(&self, start: u64, max_hops: u32, mut predicate: F) -> Vec<Vec<u64>>
    where
        F: FnMut(u64, u64, u64) -> bool,
    {
        self.try_bfs_filtered(start, max_hops, |edge_id, src, dst| {
            Ok(predicate(edge_id, src, dst))
        })
        .expect("CSR traversal allocation failed")
    }

    /// Fallible form of [`Self::bfs_filtered`]. A property/predicate error aborts
    /// traversal instead of treating that edge as allowed or returning partial
    /// results. No capacity proportional to `max_hops` is allocated up front.
    pub fn try_bfs_filtered<F>(
        &self,
        start: u64,
        max_hops: u32,
        mut predicate: F,
    ) -> Result<Vec<Vec<u64>>>
    where
        F: FnMut(u64, u64, u64) -> Result<bool>,
    {
        let mut levels = Vec::new();

        if start >= self.num_vertices {
            return Ok(levels);
        }

        let mut visited = Vec::new();
        visited
            .try_reserve_exact(self.num_vertices as usize)
            .map_err(|e| csr_error(format!("Cannot allocate CSR traversal state: {e}")))?;
        visited.resize(self.num_vertices as usize, false);

        visited[start as usize] = true;
        levels.push(vec![start]);

        for _ in 0..max_hops {
            let frontier = levels.last().unwrap();
            let mut next_level = Vec::new();

            for &v in frontier {
                for (edge_id, neighbor) in self.neighbor_edges(v) {
                    if !predicate(edge_id, v, neighbor)? {
                        continue;
                    }
                    let n = neighbor as usize;
                    if n < visited.len() && !visited[n] {
                        visited[n] = true;
                        next_level.push(neighbor);
                    }
                }
            }

            if next_level.is_empty() {
                break;
            }
            levels.push(next_level);
        }

        Ok(levels)
    }

    /// Find the shortest path between two vertices using BFS.
    ///
    /// Returns `None` if no path exists, or `Some(path)` where path is the
    /// sequence of vertex IDs from `start` to `end` (inclusive).
    pub fn shortest_path(&self, start: u64, end: u64) -> Option<Vec<u64>> {
        if start >= self.num_vertices || end >= self.num_vertices {
            return None;
        }
        if start == end {
            return Some(vec![start]);
        }

        let n = self.num_vertices as usize;

        let mut visited = vec![false; n];
        let mut parent: Vec<Option<u64>> = vec![None; n];
        let mut queue = std::collections::VecDeque::new();

        visited[start as usize] = true;
        queue.push_back(start);

        while let Some(current) = queue.pop_front() {
            for &neighbor in self.neighbors(current) {
                let ni = neighbor as usize;
                if ni < n && !visited[ni] {
                    visited[ni] = true;
                    parent[ni] = Some(current);

                    if neighbor == end {
                        let mut path = vec![end];
                        let mut node = end;
                        while let Some(p) = parent[node as usize] {
                            path.push(p);
                            node = p;
                        }
                        path.reverse();
                        return Some(path);
                    }
                    queue.push_back(neighbor);
                }
            }
        }

        None
    }
}

/// Builder for constructing a CSR index from edge data.
///
/// Accepts edges as (source, destination) pairs and builds the compressed
/// sparse row representation.
#[derive(Debug)]
pub struct CsrIndexBuilder {
    edges: Vec<InputEdge>,
    num_vertices: Option<u64>,
}

#[derive(Debug)]
struct InputEdge {
    src: u64,
    dst: u64,
    id: u64,
    ordinal: usize,
}

fn csr_error(message: impl Into<String>) -> GraphError {
    GraphError::PlanError {
        message: message.into(),
        location: snafu::Location::new(file!(), line!(), column!()),
    }
}

fn edge_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt64Array> {
    let array = batch
        .column_by_name(name)
        .ok_or_else(|| csr_error(format!("Edge batch missing '{name}' column")))?
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| csr_error(format!("{name} column must be UInt64")))?;
    if array.null_count() != 0 {
        return Err(csr_error(format!("{name} column must not contain NULL")));
    }
    Ok(array)
}

impl CsrIndexBuilder {
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            num_vertices: None,
        }
    }

    /// Set the total number of vertices explicitly. If not set, it is inferred
    /// from the maximum vertex ID seen in the edges.
    pub fn with_num_vertices(mut self, n: u64) -> Self {
        self.num_vertices = Some(n);
        self
    }

    /// Add a single directed edge from `src` to `dst`, assigning its input
    /// ordinal as the edge identity. Use [`Self::try_add_edge_with_id`] for an
    /// explicit identity and [`Self::try_build`] for checked construction.
    pub fn add_edge(mut self, src: u64, dst: u64) -> Self {
        let ordinal = self.edges.len();
        self.edges.push(InputEdge {
            src,
            dst,
            id: ordinal as u64,
            ordinal,
        });
        self
    }

    /// Add a directed edge with a caller-supplied identity. Endpoints must fit
    /// the configured dense vertex range (if set) and permit `max_id + 1`.
    /// Identities are opaque and need not be consecutive; every input edge is
    /// preserved even if identities/endpoints repeat. Callers resolving row
    /// properties must choose identities unambiguous within their snapshot.
    pub fn try_add_edge_with_id(mut self, edge_id: u64, src: u64, dst: u64) -> Result<Self> {
        self.validate_endpoints(src, dst)?;
        self.edges
            .try_reserve(1)
            .map_err(|e| csr_error(format!("Cannot allocate CSR input edge: {e}")))?;
        let ordinal = self.edges.len();
        self.edges.push(InputEdge {
            src,
            dst,
            id: edge_id,
            ordinal,
        });
        Ok(self)
    }

    fn validate_endpoints(&self, src: u64, dst: u64) -> Result<()> {
        for endpoint in [src, dst] {
            if endpoint.checked_add(1).is_none() {
                return Err(csr_error("CSR vertex ID overflows its dense vertex count"));
            }
            if self.num_vertices.is_some_and(|n| endpoint >= n) {
                return Err(csr_error(format!(
                    "CSR vertex {endpoint} is outside the configured vertex range"
                )));
            }
        }
        Ok(())
    }

    /// Add edges from non-null UInt64 `src_id` and `dst_id` columns. Identities
    /// are input ordinals across all batches added to this builder.
    pub fn add_edges_from_batch(mut self, batch: &RecordBatch) -> Result<Self> {
        let src_array = edge_column(batch, "src_id")?;
        let dst_array = edge_column(batch, "dst_id")?;
        for i in 0..batch.num_rows() {
            let ordinal = self.edges.len() as u64;
            self = self.try_add_edge_with_id(ordinal, src_array.value(i), dst_array.value(i))?;
        }
        Ok(self)
    }

    /// Add edges from non-null UInt64 `src_id`, `dst_id` and `edge_id` columns.
    /// Extra property columns remain in the caller's snapshot and are not copied
    /// into the CSR index. Missing, mistyped or NULL values return an error;
    /// no row is silently discarded.
    pub fn try_add_edges_with_ids_from_batch(mut self, batch: &RecordBatch) -> Result<Self> {
        let src_array = edge_column(batch, "src_id")?;
        let dst_array = edge_column(batch, "dst_id")?;
        let id_array = edge_column(batch, "edge_id")?;
        for i in 0..batch.num_rows() {
            self = self.try_add_edge_with_id(
                id_array.value(i),
                src_array.value(i),
                dst_array.value(i),
            )?;
        }
        Ok(self)
    }

    /// Build the CSR index.
    ///
    /// Sorts edges by source vertex, then builds offset and neighbor arrays.
    pub fn build(self) -> CsrIndex {
        self.try_build().expect("Invalid CSR index input")
    }

    /// Build with checked endpoint/count arithmetic and fallible array allocation.
    /// Use this instead of [`Self::build`] for untrusted or dataset-derived input.
    /// Dense IDs require storage proportional to the vertex count, not merely
    /// the number of edges; callers should remap sparse external IDs first.
    /// Duplicate identities are retained, not deduplicated: callers must ensure
    /// they identify the same property record, especially when mixing explicit
    /// identities with the legacy ordinal-based APIs.
    pub fn try_build(mut self) -> Result<CsrIndex> {
        for edge in &self.edges {
            self.validate_endpoints(edge.src, edge.dst)?;
        }
        let inferred = self
            .edges
            .iter()
            .flat_map(|edge| [edge.src, edge.dst])
            .max()
            .map(|max| {
                max.checked_add(1)
                    .ok_or_else(|| csr_error("CSR vertex count overflow"))
            })
            .transpose()?
            .unwrap_or(0);
        let num_vertices = self.num_vertices.unwrap_or(inferred);
        let vertices = usize::try_from(num_vertices)
            .map_err(|_| csr_error("CSR vertex count cannot fit the platform address space"))?;
        let offsets_len = vertices
            .checked_add(1)
            .ok_or_else(|| csr_error("CSR offset array length overflow"))?;

        let mut offsets = Vec::new();
        offsets
            .try_reserve_exact(offsets_len)
            .map_err(|e| csr_error(format!("Cannot allocate CSR offsets: {e}")))?;
        offsets.resize(offsets_len, 0u64);
        let mut neighbors = Vec::new();
        neighbors
            .try_reserve_exact(self.edges.len())
            .map_err(|e| csr_error(format!("Cannot allocate CSR destinations: {e}")))?;
        let mut edge_ids = Vec::new();
        edge_ids
            .try_reserve_exact(self.edges.len())
            .map_err(|e| csr_error(format!("Cannot allocate CSR edge identities: {e}")))?;

        // Sorting by ordinal within a source retains parallel-edge input order.
        self.edges
            .sort_unstable_by_key(|edge| (edge.src, edge.ordinal));
        for edge in &self.edges {
            let slot = &mut offsets[edge.src as usize + 1];
            *slot = slot
                .checked_add(1)
                .ok_or_else(|| csr_error("CSR degree overflow"))?;
            neighbors.push(edge.dst);
            edge_ids.push(edge.id);
        }
        for v in 1..offsets.len() {
            offsets[v] = offsets[v]
                .checked_add(offsets[v - 1])
                .ok_or_else(|| csr_error("CSR offset overflow"))?;
        }
        Ok(CsrIndex {
            offsets,
            neighbors,
            edge_ids,
            num_vertices,
        })
    }
}

impl Default for CsrIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Build both outgoing (CSR) and incoming (CSC) adjacency indices from edge data.
///
/// Returns `(outgoing_index, incoming_index)`.
pub fn build_bidirectional_index(edges: &[(u64, u64)], num_vertices: u64) -> (CsrIndex, CsrIndex) {
    let mut outgoing_builder = CsrIndexBuilder::new().with_num_vertices(num_vertices);
    let mut incoming_builder = CsrIndexBuilder::new().with_num_vertices(num_vertices);

    for &(src, dst) in edges {
        outgoing_builder = outgoing_builder.add_edge(src, dst);
        incoming_builder = incoming_builder.add_edge(dst, src);
    }

    (outgoing_builder.build(), incoming_builder.build())
}

/// Checked outgoing/incoming construction from `(edge_id, src_id, dst_id)`.
/// The incoming index reverses endpoints, but retains the same edge identity so
/// both directions resolve the same properties in the caller's build snapshot.
/// Parallel edges and self-loops remain distinct adjacency entries.
pub fn try_build_bidirectional_index_with_ids(
    edges: &[(u64, u64, u64)],
    num_vertices: u64,
) -> Result<(CsrIndex, CsrIndex)> {
    let mut outgoing = CsrIndexBuilder::new().with_num_vertices(num_vertices);
    let mut incoming = CsrIndexBuilder::new().with_num_vertices(num_vertices);
    for &(edge_id, src, dst) in edges {
        outgoing = outgoing.try_add_edge_with_id(edge_id, src, dst)?;
        incoming = incoming.try_add_edge_with_id(edge_id, dst, src)?;
    }
    Ok((outgoing.try_build()?, incoming.try_build()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test graph:
    //   0 -> 1, 2, 3
    //   1 -> 2
    //   2 -> 3
    //   3 -> (none)
    fn sample_index() -> CsrIndex {
        CsrIndexBuilder::new()
            .with_num_vertices(4)
            .add_edge(0, 1)
            .add_edge(0, 2)
            .add_edge(0, 3)
            .add_edge(1, 2)
            .add_edge(2, 3)
            .build()
    }

    #[test]
    fn test_basic_neighbor_lookup() {
        let idx = sample_index();

        assert_eq!(idx.neighbors(0), &[1, 2, 3]);
        assert_eq!(idx.neighbors(1), &[2]);
        assert_eq!(idx.neighbors(2), &[3]);
        assert_eq!(idx.neighbors(3), &[] as &[u64]);
    }

    #[test]
    fn test_degree() {
        let idx = sample_index();
        assert_eq!(idx.degree(0), 3);
        assert_eq!(idx.degree(1), 1);
        assert_eq!(idx.degree(2), 1);
        assert_eq!(idx.degree(3), 0);
    }

    #[test]
    fn test_metadata() {
        let idx = sample_index();
        assert_eq!(idx.num_vertices(), 4);
        assert_eq!(idx.num_edges(), 5);
    }

    #[test]
    fn test_out_of_range_vertex() {
        let idx = sample_index();
        assert_eq!(idx.neighbors(99), &[] as &[u64]);
        assert_eq!(idx.degree(99), 0);
    }

    #[test]
    fn test_empty_graph() {
        let idx = CsrIndexBuilder::new().with_num_vertices(0).build();
        assert_eq!(idx.num_vertices(), 0);
        assert_eq!(idx.num_edges(), 0);
        assert_eq!(idx.neighbors(0), &[] as &[u64]);
    }

    #[test]
    fn test_isolated_vertices() {
        let idx = CsrIndexBuilder::new()
            .with_num_vertices(5)
            .add_edge(1, 3)
            .build();

        assert_eq!(idx.neighbors(0), &[] as &[u64]);
        assert_eq!(idx.neighbors(1), &[3]);
        assert_eq!(idx.neighbors(2), &[] as &[u64]);
        assert_eq!(idx.neighbors(3), &[] as &[u64]);
        assert_eq!(idx.neighbors(4), &[] as &[u64]);
    }

    #[test]
    fn test_build_from_record_batch() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("src_id", DataType::UInt64, false),
            Field::new("dst_id", DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![0, 0, 1, 2])),
                Arc::new(UInt64Array::from(vec![1, 2, 2, 0])),
            ],
        )
        .unwrap();

        let idx = CsrIndexBuilder::new()
            .add_edges_from_batch(&batch)
            .unwrap()
            .build();

        assert_eq!(idx.neighbors(0), &[1, 2]);
        assert_eq!(idx.neighbors(1), &[2]);
        assert_eq!(idx.neighbors(2), &[0]);
    }

    #[test]
    fn test_to_record_batch() {
        let idx = sample_index();
        let batch = idx.to_record_batch().unwrap();

        assert_eq!(batch.num_rows(), 4);
        assert_eq!(batch.num_columns(), 3);

        let offsets = batch
            .column_by_name("offset")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(offsets.value(0), 0);
        assert_eq!(offsets.value(1), 3);
        assert_eq!(offsets.value(2), 4);
        assert_eq!(offsets.value(3), 5);

        let degrees = batch
            .column_by_name("degree")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(degrees.value(0), 3);
        assert_eq!(degrees.value(1), 1);
        assert_eq!(degrees.value(2), 1);
        assert_eq!(degrees.value(3), 0);
    }

    #[test]
    fn test_neighbors_to_record_batch() {
        let idx = sample_index();
        let batch = idx.neighbors_to_record_batch().unwrap();

        assert_eq!(batch.num_rows(), 5);
        let dst = batch
            .column_by_name("dst_id")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(dst.value(0), 1);
        assert_eq!(dst.value(1), 2);
        assert_eq!(dst.value(2), 3);
        assert_eq!(dst.value(3), 2);
        assert_eq!(dst.value(4), 3);
    }

    #[test]
    fn test_bidirectional_index() {
        let edges = vec![(0, 1), (0, 2), (1, 2)];
        let (outgoing, incoming) = build_bidirectional_index(&edges, 3);

        // Outgoing
        assert_eq!(outgoing.neighbors(0), &[1, 2]);
        assert_eq!(outgoing.neighbors(1), &[2]);
        assert_eq!(outgoing.neighbors(2), &[] as &[u64]);

        // Incoming (reversed edges)
        assert_eq!(incoming.neighbors(0), &[] as &[u64]);
        assert_eq!(incoming.neighbors(1), &[0]);
        assert_eq!(incoming.neighbors(2), &[0, 1]);
    }

    #[test]
    fn test_bfs_traversal() {
        // Graph: 0->1, 0->2, 1->3, 2->3, 3->4
        let idx = CsrIndexBuilder::new()
            .with_num_vertices(5)
            .add_edge(0, 1)
            .add_edge(0, 2)
            .add_edge(1, 3)
            .add_edge(2, 3)
            .add_edge(3, 4)
            .build();

        let levels = idx.bfs(0, 3);
        assert_eq!(levels.len(), 4);
        assert_eq!(levels[0], vec![0]); // start
        assert_eq!(levels[1], vec![1, 2]); // 1-hop
        assert_eq!(levels[2], vec![3]); // 2-hop (3 reached from both 1 and 2, but visited once)
        assert_eq!(levels[3], vec![4]); // 3-hop
    }

    #[test]
    fn test_bfs_limited_hops() {
        let idx = CsrIndexBuilder::new()
            .with_num_vertices(4)
            .add_edge(0, 1)
            .add_edge(1, 2)
            .add_edge(2, 3)
            .build();

        let levels = idx.bfs(0, 1);
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0], vec![0]);
        assert_eq!(levels[1], vec![1]);
    }

    #[test]
    fn test_bfs_disconnected() {
        let idx = CsrIndexBuilder::new()
            .with_num_vertices(4)
            .add_edge(0, 1)
            // 2, 3 are disconnected
            .build();

        let levels = idx.bfs(0, 10);
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0], vec![0]);
        assert_eq!(levels[1], vec![1]);
    }

    #[test]
    fn test_bfs_invalid_start() {
        let idx = CsrIndexBuilder::new().with_num_vertices(3).build();
        let levels = idx.bfs(99, 5);
        assert!(levels.is_empty());
    }

    #[test]
    fn test_shortest_path_direct() {
        let idx = CsrIndexBuilder::new()
            .with_num_vertices(3)
            .add_edge(0, 1)
            .add_edge(1, 2)
            .add_edge(0, 2)
            .build();

        // Direct edge 0->2 is shorter than 0->1->2
        let path = idx.shortest_path(0, 2).unwrap();
        assert_eq!(path, vec![0, 2]);
    }

    #[test]
    fn test_shortest_path_multi_hop() {
        let idx = CsrIndexBuilder::new()
            .with_num_vertices(4)
            .add_edge(0, 1)
            .add_edge(1, 2)
            .add_edge(2, 3)
            .build();

        let path = idx.shortest_path(0, 3).unwrap();
        assert_eq!(path, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_shortest_path_same_vertex() {
        let idx = sample_index();
        let path = idx.shortest_path(2, 2).unwrap();
        assert_eq!(path, vec![2]);
    }

    #[test]
    fn test_shortest_path_unreachable() {
        let idx = CsrIndexBuilder::new()
            .with_num_vertices(3)
            .add_edge(0, 1)
            // no path from 0 to 2
            .build();

        assert!(idx.shortest_path(0, 2).is_none());
    }

    #[test]
    fn test_shortest_path_invalid_vertices() {
        let idx = sample_index();
        assert!(idx.shortest_path(99, 0).is_none());
        assert!(idx.shortest_path(0, 99).is_none());
        assert!(idx.shortest_path(99, 99).is_none());
        assert!(idx.shortest_path(u64::MAX, u64::MAX).is_none());
    }

    #[test]
    fn test_auto_inferred_num_vertices() {
        let idx = CsrIndexBuilder::new().add_edge(0, 5).add_edge(3, 7).build();

        // Should infer num_vertices = 8 (max ID 7 + 1)
        assert_eq!(idx.num_vertices(), 8);
        assert_eq!(idx.neighbors(0), &[5]);
        assert_eq!(idx.neighbors(3), &[7]);
    }

    #[test]
    fn test_self_loops() {
        let idx = CsrIndexBuilder::new()
            .with_num_vertices(3)
            .add_edge(0, 0)
            .add_edge(1, 1)
            .add_edge(0, 1)
            .build();

        assert_eq!(idx.neighbors(0), &[0, 1]);
        assert_eq!(idx.neighbors(1), &[1]);
    }

    #[test]
    fn test_parallel_edges() {
        let idx = CsrIndexBuilder::new()
            .with_num_vertices(2)
            .add_edge(0, 1)
            .add_edge(0, 1)
            .add_edge(0, 1)
            .build();

        // CSR preserves multi-edges
        assert_eq!(idx.neighbors(0), &[1, 1, 1]);
        assert_eq!(idx.degree(0), 3);
    }

    #[test]
    fn test_input_ordinals_survive_source_sort() {
        let idx = CsrIndexBuilder::new()
            .add_edge(2, 0)
            .add_edge(0, 2)
            .add_edge(0, 1)
            .add_edge(1, 0)
            .try_build()
            .unwrap();
        assert_eq!(idx.neighbor_edges(0).collect::<Vec<_>>(), [(1, 2), (2, 1)]);
        assert_eq!(idx.neighbor_edges(2).collect::<Vec<_>>(), [(0, 0)]);
        assert!(idx.neighbor_edges(u64::MAX).next().is_none());
    }

    #[test]
    fn test_filtered_parallel_edge_validates_before_visited() {
        let idx = CsrIndexBuilder::new()
            .try_add_edge_with_id(80, 0, 1)
            .unwrap()
            .try_add_edge_with_id(20, 0, 1)
            .unwrap()
            .try_add_edge_with_id(u64::MAX, 1, 2)
            .unwrap()
            .try_build()
            .unwrap();
        let mut evaluated = Vec::new();
        assert_eq!(
            idx.bfs_filtered(0, 2, |id, src, dst| {
                evaluated.push((id, src, dst));
                id != 80
            }),
            [vec![0], vec![1], vec![2]]
        );
        assert_eq!(evaluated, [(80, 0, 1), (20, 0, 1), (u64::MAX, 1, 2)]);
        assert_eq!(
            idx.filtered_neighbors(0, |id, _, _| id == 20)
                .collect::<Vec<_>>(),
            [(20, 1)]
        );
        assert_eq!(idx.neighbors(0), [1, 1]);
    }

    #[test]
    fn test_rejected_edge_does_not_block_later_accepted_path() {
        let idx = CsrIndexBuilder::new()
            .add_edge(0, 1) // ordinal 0 rejected
            .add_edge(0, 2)
            .add_edge(2, 1)
            .add_edge(1, 3)
            .add_edge(3, 0) // cycle
            .try_build()
            .unwrap();
        assert_eq!(
            idx.bfs_filtered(0, u32::MAX, |id, _, _| id != 0),
            [vec![0], vec![2], vec![1], vec![3]]
        );
        assert_eq!(idx.bfs_filtered(0, 0, |_, _, _| false), [vec![0]]);
        assert_eq!(idx.bfs_filtered(0, 3, |_, _, _| false), [vec![0]]);
        assert!(idx.bfs_filtered(u64::MAX, 1, |_, _, _| true).is_empty());
    }

    #[test]
    fn test_fallible_predicate_error_is_not_treated_as_allow() {
        let idx = sample_index();
        let result = idx.try_bfs_filtered(0, 2, |id, _, _| {
            if id == 1 {
                Err(csr_error("Property snapshot is unavailable"))
            } else {
                Ok(true)
            }
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_bidirectional_identity_and_parallel_edges() {
        let (outgoing, incoming) =
            try_build_bidirectional_index_with_ids(&[(40, 0, 1), (10, 0, 1), (99, 1, 1)], 2)
                .unwrap();
        assert_eq!(
            outgoing.neighbor_edges(0).collect::<Vec<_>>(),
            [(40, 1), (10, 1)]
        );
        assert_eq!(
            incoming.neighbor_edges(1).collect::<Vec<_>>(),
            [(40, 0), (10, 0), (99, 1)]
        );
        let mut calls = Vec::new();
        assert_eq!(
            incoming.bfs_filtered(1, 1, |id, src, dst| {
                calls.push((id, src, dst));
                id == 10
            }),
            [vec![1], vec![0]]
        );
        assert_eq!(calls, [(40, 1, 0), (10, 1, 0), (99, 1, 1)]);
    }

    #[test]
    fn test_explicit_and_ordinal_identity_collision_preserves_entries() {
        let idx = CsrIndexBuilder::new()
            .try_add_edge_with_id(1, 0, 1)
            .unwrap()
            .add_edge(0, 1) // ordinal 1; intentionally the same opaque identity
            .try_build()
            .unwrap();
        assert_eq!(idx.neighbor_edges(0).collect::<Vec<_>>(), [(1, 1), (1, 1)]);
    }

    fn identified_batch(
        src: Vec<Option<u64>>,
        dst: Vec<Option<u64>>,
        ids: Vec<Option<u64>>,
    ) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("src_id", DataType::UInt64, true),
                Field::new("dst_id", DataType::UInt64, true),
                Field::new("edge_id", DataType::UInt64, true),
            ])),
            vec![
                Arc::new(UInt64Array::from(src)),
                Arc::new(UInt64Array::from(dst)),
                Arc::new(UInt64Array::from(ids)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_identified_arrow_batch_and_export_compatibility() {
        let batch = identified_batch(
            vec![Some(1), Some(0), Some(0)],
            vec![Some(2), Some(2), Some(1)],
            vec![Some(70), Some(90), Some(50)],
        );
        let idx = CsrIndexBuilder::new()
            .try_add_edges_with_ids_from_batch(&batch)
            .unwrap()
            .try_build()
            .unwrap();
        let exported = idx.edges_to_record_batch().unwrap();
        assert_eq!(exported.num_rows(), 3);
        assert_eq!(
            edge_column(&exported, "src_id").unwrap().values().as_ref(),
            &[0, 0, 1]
        );
        assert_eq!(
            edge_column(&exported, "dst_id").unwrap().values().as_ref(),
            &[2, 1, 2]
        );
        assert_eq!(
            edge_column(&exported, "edge_id").unwrap().values().as_ref(),
            &[90, 50, 70]
        );
        let legacy = idx.neighbors_to_record_batch().unwrap();
        assert_eq!(legacy.num_columns(), 1);
        assert_eq!(legacy.schema().field(0).name(), "dst_id");
        let offsets = idx.to_record_batch().unwrap();
        assert_eq!(offsets.num_columns(), 3);
        assert_eq!(offsets.schema().field(0).name(), "vertex_id");
        assert_eq!(offsets.schema().field(1).name(), "offset");
        assert_eq!(offsets.schema().field(2).name(), "degree");
    }

    #[test]
    fn test_ordinals_continue_across_batches() {
        let batch = identified_batch(vec![Some(0)], vec![Some(1)], vec![Some(999)]);
        let idx = CsrIndexBuilder::new()
            .add_edges_from_batch(&batch)
            .unwrap()
            .add_edges_from_batch(&batch)
            .unwrap()
            .try_build()
            .unwrap();
        assert_eq!(idx.neighbor_edges(0).collect::<Vec<_>>(), [(0, 1), (1, 1)]);
    }

    #[test]
    fn test_arrow_nulls_are_rejected_not_read_as_zero() {
        for column in 0..3 {
            let mut values = [vec![Some(0)], vec![Some(1)], vec![Some(5)]];
            values[column][0] = None;
            let [src, dst, ids] = values;
            let batch = identified_batch(src, dst, ids);
            assert!(CsrIndexBuilder::new()
                .try_add_edges_with_ids_from_batch(&batch)
                .is_err());
            if column < 2 {
                assert!(CsrIndexBuilder::new().add_edges_from_batch(&batch).is_err());
            }
        }
    }

    #[test]
    fn test_arrow_missing_and_mistyped_identity_are_rejected() {
        let base = identified_batch(vec![Some(0)], vec![Some(1)], vec![Some(5)]);
        let missing = base.project(&[0, 1]).unwrap();
        assert!(CsrIndexBuilder::new()
            .try_add_edges_with_ids_from_batch(&missing)
            .is_err());
        let wrong = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("src_id", DataType::UInt64, false),
                Field::new("dst_id", DataType::UInt64, false),
                Field::new("edge_id", DataType::Int32, false),
            ])),
            vec![
                Arc::new(UInt64Array::from(vec![0])),
                Arc::new(UInt64Array::from(vec![1])),
                Arc::new(arrow_array::Int32Array::from(vec![5])),
            ],
        )
        .unwrap();
        assert!(CsrIndexBuilder::new()
            .try_add_edges_with_ids_from_batch(&wrong)
            .is_err());
    }

    #[test]
    fn test_checked_endpoint_range_and_overflow() {
        assert!(CsrIndexBuilder::new()
            .with_num_vertices(2)
            .try_add_edge_with_id(1, 0, 2)
            .is_err());
        assert!(CsrIndexBuilder::new()
            .try_add_edge_with_id(1, u64::MAX, 0)
            .is_err());
        assert!(CsrIndexBuilder::new()
            .add_edge(0, 2)
            .with_num_vertices(2)
            .try_build()
            .is_err());
        assert!(CsrIndexBuilder::new()
            .add_edge(u64::MAX, 0)
            .try_build()
            .is_err());
        assert!(CsrIndexBuilder::new()
            .with_num_vertices(u64::MAX)
            .try_build()
            .is_err());
        assert!(try_build_bidirectional_index_with_ids(&[(1, 0, 2)], 2).is_err());
    }

    #[test]
    fn test_filtered_bfs_deterministic_reference_matrix() {
        let graphs = [
            vec![(91, 0, 1), (301, 0, 1), (77, 1, 2), (88, 2, 0), (52, 2, 2)],
            vec![(41, 0, 3), (705, 3, 1), (67, 1, 4), (902, 4, 3), (26, 3, 3)],
            vec![(103, 1, 0), (42, 1, 0), (67, 2, 1), (701, 4, 2), (28, 4, 4)],
        ];
        for edges in graphs {
            let (outgoing, incoming) = try_build_bidirectional_index_with_ids(&edges, 6).unwrap();
            for (reverse, index) in [(false, &outgoing), (true, &incoming)] {
                let reference = |allowed: &[u64], start: u64, hops: u32| {
                    let mut seen = [false; 6];
                    seen[start as usize] = true;
                    let mut levels = vec![vec![start]];
                    for _ in 0..hops {
                        let mut next = Vec::new();
                        for &vertex in levels.last().unwrap() {
                            for &(id, src, dst) in &edges {
                                let (src, dst) = if reverse { (dst, src) } else { (src, dst) };
                                if src == vertex && allowed.contains(&id) && !seen[dst as usize] {
                                    seen[dst as usize] = true;
                                    next.push(dst);
                                }
                            }
                        }
                        if next.is_empty() {
                            break;
                        }
                        levels.push(next);
                    }
                    levels
                };
                for mode in 0..3 {
                    // A prepared stable identity mask, independent of traversal/path history.
                    let allowed: Vec<_> = edges
                        .iter()
                        .filter(|&&(id, _, _)| mode == 1 || (mode == 2 && id % 3 != 1))
                        .map(|&(id, _, _)| id)
                        .collect();
                    for start in 0..6 {
                        for hops in [0, 1, 2, 6] {
                            let actual =
                                index.bfs_filtered(start, hops, |id, _, _| allowed.contains(&id));
                            assert_eq!(actual, reference(&allowed, start, hops));
                            if mode == 1 {
                                assert_eq!(actual, index.bfs(start, hops));
                            }
                            if mode == 0 {
                                assert_eq!(actual, [vec![start]]);
                            }
                        }
                    }
                }
            }
        }
    }
}
