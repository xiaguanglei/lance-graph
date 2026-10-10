// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! In-memory CSR microbenchmarks, not Cypher or end-to-end storage benchmarks.
//! Construction and the snapshot-local edge allow-mask are outside the timed
//! traversal. No files, external services or persistent index are involved.
//! BFS includes allocation/clearing of its O(vertices) visited array; differences
//! between sizes must not be reported as storage-query scale improvements.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lance_graph::CsrIndexBuilder;

fn bench_csr_adjacency(c: &mut Criterion) {
    let mut group = c.benchmark_group("csr_adjacency");
    for vertices in [20_000u64, 200_000] {
        let mut builder = CsrIndexBuilder::new().with_num_vertices(vertices);
        let mut allowed = Vec::new();
        for src in 0..vertices {
            for delta in 1..=4 {
                let edge_id = allowed.len() as u64;
                builder = builder
                    .try_add_edge_with_id(edge_id, src, (src + delta) % vertices)
                    .unwrap();
                // This precomputed mask stands for generic edge conditions
                // evaluated against the same snapshot used to build the CSR.
                allowed.push(delta <= 2);
            }
        }
        let index = builder.try_build().unwrap();
        assert_eq!(index.neighbors(0).len(), 4);
        assert_eq!(
            index
                .filtered_neighbors(0, |id, _, _| allowed[id as usize])
                .count(),
            2
        );
        assert_eq!(
            index.bfs_filtered(0, 3, |id, _, _| allowed[id as usize]),
            [vec![0], vec![1, 2], vec![3, 4], vec![5, 6]]
        );

        group.bench_with_input(
            BenchmarkId::new("neighbor_ids", vertices),
            &index,
            |b, index| {
                b.iter(|| {
                    black_box(
                        index
                            .neighbor_edges(black_box(0))
                            .fold(0u64, |sum, (edge_id, dst)| {
                                sum.wrapping_add(black_box(edge_id) ^ black_box(dst))
                            }),
                    )
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("filtered_neighbors", vertices),
            &index,
            |b, index| {
                b.iter(|| {
                    black_box(
                        index
                            .filtered_neighbors(black_box(0), |id, _, _| {
                                black_box(allowed[id as usize])
                            })
                            .count(),
                    )
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("bfs_three_hops", vertices),
            &index,
            |b, index| b.iter(|| black_box(index.bfs(black_box(0), black_box(3)))),
        );
        group.bench_with_input(
            BenchmarkId::new("filtered_bfs_three_hops", vertices),
            &index,
            |b, index| {
                b.iter(|| {
                    black_box(index.bfs_filtered(black_box(0), black_box(3), |id, _, _| {
                        black_box(allowed[id as usize])
                    }))
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_csr_adjacency);
criterion_main!(benches);
