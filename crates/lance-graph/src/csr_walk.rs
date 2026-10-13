// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bounded native multi-hop walks retaining every edge identity.
//!
//! Unlike reachability BFS, this preserves the bag of bounded walks: parallel
//! edges and cycles are not deduplicated. The mask is evaluated before the next
//! state exists, once for each traversed edge and hop. Callers must bind it to
//! the same snapshot as the CSR; it must not perform per-edge I/O. Resource
//! exhaustion returns an error, never a silently truncated successful result.

use crate::{CsrIndex, GraphError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsrWalk {
    pub vertices: Vec<u64>,
    /// Opaque identities in the original CSR build domain, not array positions.
    pub edge_ids: Vec<u64>,
}

#[derive(Debug, Clone, Copy)]
pub struct CsrWalkOptions {
    pub min_hops: u32,
    pub max_hops: u32,
    pub max_frontier_states: usize,
    pub max_paths: usize,
    pub max_examined_edges: usize,
}

impl Default for CsrWalkOptions {
    fn default() -> Self {
        Self {
            min_hops: 1,
            max_hops: 3,
            max_frontier_states: 20_000,
            max_paths: 20_000,
            max_examined_edges: 1_000_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsrWalkResult {
    pub paths: Vec<CsrWalk>,
    pub examined_edges: usize,
}

fn traversal_error(message: &str) -> GraphError {
    GraphError::ExecutionError {
        message: format!("Native CSR multi-hop: {message}"),
        location: snafu::Location::new(file!(), line!(), column!()),
    }
}

/// Execute all hops within one traversal without planning a JOIN per hop.
///
/// `allowed(hop, edge_id, from, to)` is a prepared property/allowed-row mask.
/// Endpoints follow the traversal direction of this CSR. An incoming CSR keeps
/// the same original edge ID, allowing original properties to be retrieved.
/// Node existence/visibility can be included in the mask. The start vertex must
/// already have passed the caller's source-node filter.
pub fn csr_walk_filtered<F>(
    index: &CsrIndex,
    start: u64,
    options: CsrWalkOptions,
    mut allowed: F,
) -> Result<CsrWalkResult>
where
    F: FnMut(u32, u64, u64, u64) -> bool,
{
    walk_indices(&[index], start, options, true, &mut allowed)
}

/// Traverse a fixed sequence of outgoing/incoming CSR views from the same
/// snapshot and dense vertex domain. Each hop can use a different direction
/// and a different prepared edge/node mask. Callers must not combine unrelated
/// dense mappings merely because their vertex counts happen to be equal.
pub fn csr_walk_steps_filtered<F>(
    steps: &[&CsrIndex],
    start: u64,
    options: CsrWalkOptions,
    mut allowed: F,
) -> Result<CsrWalkResult>
where
    F: FnMut(u32, u64, u64, u64) -> bool,
{
    walk_indices(steps, start, options, false, &mut allowed)
}

/// Read-only adjacency a bounded walk can traverse: a plain [`CsrIndex`] or a
/// layered snapshot view (base CSR, hidden fragments and an appended delta).
pub(crate) trait AdjacencyView {
    fn num_vertices(&self) -> u64;
    fn neighbor_edges(&self, vertex: u64) -> impl Iterator<Item = (u64, u64)> + '_;
}

impl AdjacencyView for CsrIndex {
    fn num_vertices(&self) -> u64 {
        CsrIndex::num_vertices(self)
    }
    fn neighbor_edges(&self, vertex: u64) -> impl Iterator<Item = (u64, u64)> + '_ {
        CsrIndex::neighbor_edges(self, vertex)
    }
}

/// [`csr_walk_steps_filtered`] over any adjacency view of one snapshot.
pub(crate) fn csr_walk_views_filtered<V, F>(
    steps: &[&V],
    start: u64,
    options: CsrWalkOptions,
    mut allowed: F,
) -> Result<CsrWalkResult>
where
    V: AdjacencyView,
    F: FnMut(u32, u64, u64, u64) -> bool,
{
    walk_indices(steps, start, options, false, &mut allowed)
}

fn walk_indices<V: AdjacencyView, F>(
    steps: &[&V],
    start: u64,
    options: CsrWalkOptions,
    repeat: bool,
    allowed: &mut F,
) -> Result<CsrWalkResult>
where
    F: FnMut(u32, u64, u64, u64) -> bool,
{
    let Some(first) = steps.first() else {
        return Err(traversal_error("empty step sequence"));
    };
    if options.min_hops > options.max_hops
        || options.max_hops > crate::MAX_VARIABLE_LENGTH_HOPS
        || options.max_frontier_states == 0
        || options.max_paths == 0
        || options.max_examined_edges == 0
        || (!repeat && options.max_hops as usize > steps.len())
        || steps
            .iter()
            .any(|index| index.num_vertices() != first.num_vertices())
    {
        return Err(traversal_error("invalid range or resource budget"));
    }
    let mut result = CsrWalkResult {
        paths: Vec::new(),
        examined_edges: 0,
    };
    if start >= first.num_vertices() {
        return Ok(result);
    }
    let seed = CsrWalk {
        vertices: vec![start],
        edge_ids: Vec::new(),
    };
    if options.min_hops == 0 {
        result.paths.push(seed.clone());
    }
    let mut frontier = vec![seed];
    for hop in 1..=options.max_hops {
        let index = steps[if repeat { 0 } else { hop as usize - 1 }];
        let mut next = Vec::new();
        for state in frontier {
            let source = *state.vertices.last().expect("walk contains its seed");
            for (edge_id, target) in index.neighbor_edges(source) {
                if result.examined_edges == options.max_examined_edges {
                    return Err(traversal_error("examined-edge budget exceeded"));
                }
                result.examined_edges += 1;
                if !allowed(hop, edge_id, source, target) {
                    continue;
                }
                if hop >= options.min_hops && result.paths.len() == options.max_paths {
                    return Err(traversal_error("path-result budget exceeded"));
                }
                if hop < options.max_hops && next.len() == options.max_frontier_states {
                    return Err(traversal_error("frontier-state budget exceeded"));
                }
                let mut path = state.clone();
                path.vertices.push(target);
                path.edge_ids.push(edge_id);
                if hop >= options.min_hops {
                    result.paths.push(path.clone());
                }
                if hop < options.max_hops {
                    next.push(path);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrIndexBuilder;

    fn graph() -> CsrIndex {
        let mut builder = CsrIndexBuilder::new().with_num_vertices(4);
        for (id, source, target) in [
            (100, 0, 1),
            (101, 0, 1),
            (102, 1, 2),
            (103, 2, 0),
            (104, 1, 3),
        ] {
            builder = builder.try_add_edge_with_id(id, source, target).unwrap();
        }
        builder.try_build().unwrap()
    }

    #[test]
    fn two_and_three_hop_walks_keep_parallel_edges_and_cycles() {
        let result = csr_walk_filtered(
            &graph(),
            0,
            CsrWalkOptions {
                min_hops: 2,
                max_hops: 3,
                ..Default::default()
            },
            |_, _, _, _| true,
        )
        .unwrap();
        let ids = result
            .paths
            .iter()
            .map(|path| path.edge_ids.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            vec![
                vec![100, 102],
                vec![100, 104],
                vec![101, 102],
                vec![101, 104],
                vec![100, 102, 103],
                vec![101, 102, 103]
            ]
        );
        assert_eq!(result.paths[4].vertices, [0, 1, 2, 0]);
    }

    #[test]
    fn every_hop_evaluates_the_edge_mask_before_expanding() {
        let mut calls = Vec::new();
        let result = csr_walk_filtered(
            &graph(),
            0,
            CsrWalkOptions::default(),
            |hop, id, source, target| {
                calls.push((hop, id, source, target));
                id != 100 && !(hop == 2 && target == 3)
            },
        )
        .unwrap();
        assert_eq!(
            result
                .paths
                .iter()
                .map(|p| p.edge_ids.clone())
                .collect::<Vec<_>>(),
            [vec![101], vec![101, 102], vec![101, 102, 103]]
        );
        assert_eq!(
            calls,
            [
                (1, 100, 0, 1),
                (1, 101, 0, 1),
                (2, 102, 1, 2),
                (2, 104, 1, 3),
                (3, 103, 2, 0)
            ]
        );
    }

    #[test]
    fn incoming_walk_retains_original_evidence_identity() {
        let mut incoming = CsrIndexBuilder::new().with_num_vertices(3);
        for (id, source, target) in [(41, 0, 1), (42, 1, 2)] {
            incoming = incoming.try_add_edge_with_id(id, target, source).unwrap();
        }
        let result = csr_walk_filtered(
            &incoming.try_build().unwrap(),
            2,
            CsrWalkOptions {
                min_hops: 2,
                max_hops: 2,
                ..Default::default()
            },
            |_, id, _, _| id == 41 || id == 42,
        )
        .unwrap();
        assert_eq!(
            result.paths,
            [CsrWalk {
                vertices: vec![2, 1, 0],
                edge_ids: vec![42, 41]
            }]
        );
    }

    #[test]
    fn mixed_direction_two_hops_keep_visible_evidence_only() {
        // Dense domain: entities 0/1/2, evidence memories 3/4.
        let mut outgoing = CsrIndexBuilder::new().with_num_vertices(5);
        let mut incoming = CsrIndexBuilder::new().with_num_vertices(5);
        for (id, memory, entity) in [(50, 3, 0), (51, 3, 1), (52, 4, 0), (53, 4, 2)] {
            outgoing = outgoing.try_add_edge_with_id(id, memory, entity).unwrap();
            incoming = incoming.try_add_edge_with_id(id, entity, memory).unwrap();
        }
        let outgoing = outgoing.try_build().unwrap();
        let incoming = incoming.try_build().unwrap();
        let result = csr_walk_steps_filtered(
            &[&incoming, &outgoing],
            0,
            CsrWalkOptions {
                min_hops: 2,
                max_hops: 2,
                ..Default::default()
            },
            |hop, _, _, target| (hop != 1 || target == 3) && (hop != 2 || target != 0),
        )
        .unwrap();
        assert_eq!(
            result.paths,
            [CsrWalk {
                vertices: vec![0, 3, 1],
                edge_ids: vec![50, 51]
            }]
        );
    }

    #[test]
    fn budgets_fail_without_returning_partial_success() {
        for options in [
            CsrWalkOptions {
                max_paths: 1,
                ..Default::default()
            },
            CsrWalkOptions {
                min_hops: 3,
                max_frontier_states: 1,
                ..Default::default()
            },
            CsrWalkOptions {
                max_examined_edges: 2,
                ..Default::default()
            },
        ] {
            assert!(csr_walk_filtered(&graph(), 0, options, |_, _, _, _| true).is_err());
        }
    }

    #[test]
    fn invalid_start_and_zero_hops_are_explicit() {
        assert!(
            csr_walk_filtered(&graph(), 99, CsrWalkOptions::default(), |_, _, _, _| true)
                .unwrap()
                .paths
                .is_empty()
        );
        let result = csr_walk_filtered(
            &graph(),
            0,
            CsrWalkOptions {
                min_hops: 0,
                max_hops: 0,
                ..Default::default()
            },
            |_, _, _, _| true,
        )
        .unwrap();
        assert_eq!(
            result.paths,
            [CsrWalk {
                vertices: vec![0],
                edge_ids: vec![]
            }]
        );
    }
}
