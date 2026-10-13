# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Version-pinned snapshot reads (GraphSnapshotProvider) over Lance tables."""

from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pytest
from lance_graph import (
    CypherQuery,
    DirNamespace,
    GraphConfig,
    GraphSnapshot,
    GraphSnapshotProvider,
    GraphVersions,
)

lance = pytest.importorskip("lance")

N = 50

QUERIES = {
    "hop1": "MATCH (a:Person {person_id: 3})-[k:KNOWS]->(b:Person) "
    "WHERE b.age > 20 RETURN b.name, k.since",
    "hop1_in": "MATCH (a:Person {person_id: 3})<-[:KNOWS]-(b:Person) RETURN b.name",
    "hop2": "MATCH (a:Person {person_id: 3})-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) "
    "WHERE c.age >= 30 RETURN b.name, c.name",
    "hop3": "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)"
    "-[:KNOWS]->(d:Person) WHERE a.person_id IN [1, 2] RETURN a.name, d.name, d.age",
}


def people_table(n=N):
    return pa.table(
        {
            "person_id": list(range(n)),
            "name": [f"p{i}" for i in range(n)],
            "age": [18 + (i * 7) % 40 for i in range(n)],
        }
    )


def edge_table(pairs):
    return pa.table(
        {
            "src": [s for s, _ in pairs],
            "dst": [d for _, d in pairs],
            "since": [2000 + (s + d) % 20 for s, d in pairs],
        }
    )


def initial_edges(n=N):
    return [(i, (i + delta) % n) for i in range(n) for delta in (1, 7)]


def rows(table):
    data = table.to_pydict()
    columns = list(data)
    return sorted(zip(*(data[c] for c in columns))) if columns else []


@pytest.fixture
def graph(tmp_path):
    config = (
        GraphConfig.builder()
        .with_node_label("Person", "person_id")
        .with_relationship("KNOWS", "src", "dst")
        .build()
    )
    people = people_table()
    edges = edge_table(initial_edges())
    lance.write_dataset(people, tmp_path / "Person.lance")
    lance.write_dataset(edges, tmp_path / "knows_edges.lance")
    return {
        "root": tmp_path,
        "config": config,
        "people": people,
        "edges": edges,
    }


def make_provider(graph, **options):
    return GraphSnapshotProvider(
        DirNamespace(str(graph["root"])),
        graph["config"],
        tables={"KNOWS": "knows_edges"},
        **options,
    )


def join_result(graph, text, people=None, edges=None):
    """Independent in-memory execution of the same query (JOIN plan)."""
    return (
        CypherQuery(text)
        .with_config(graph["config"])
        .execute(
            {
                "Person": people if people is not None else graph["people"],
                "KNOWS": edges if edges is not None else graph["edges"],
            }
        )
    )


def edge_version(graph):
    return lance.dataset(str(graph["root"] / "knows_edges.lance")).version


@pytest.mark.parametrize("column_cache", [0, 16 << 20])
@pytest.mark.parametrize("name", sorted(QUERIES))
def test_snapshot_matches_join_plan(graph, name, column_cache):
    provider = make_provider(graph, max_column_cache_bytes=column_cache)
    snapshot = provider.latest()
    assert isinstance(snapshot, GraphSnapshot)
    expected = join_result(graph, QUERIES[name])
    assert len(expected) > 0
    for _ in range(2):  # cold, then cached adjacency / columns
        result = snapshot.execute(QUERIES[name])
        assert result.schema == expected.schema
        assert rows(result) == rows(expected)
    stats = provider.cache_stats()
    assert stats["full_builds"] >= 1
    if column_cache:
        assert stats["column_entries"] >= 1
        assert stats["column_bytes"] > 0
    else:
        assert stats["column_entries"] == 0


def test_warm_and_cache_stats(graph):
    provider = make_provider(graph)
    report = provider.latest().warm()
    assert report == {"adjacency": 1, "node_ids": 1, "skipped": 0}
    stats = provider.cache_stats()
    assert set(stats) == {
        "entries",
        "bytes",
        "rows",
        "full_builds",
        "incremental_builds",
        "evictions",
        "rejected_builds",
        "column_entries",
        "column_bytes",
    }
    assert stats["entries"] == 2
    assert stats["rejected_builds"] == 0
    # Warming again reuses every entry.
    provider.latest().warm()
    assert provider.cache_stats()["full_builds"] == stats["full_builds"]


def test_too_small_budget_falls_back_with_same_results(graph):
    provider = make_provider(graph, max_edges=10)
    snapshot = provider.latest()
    assert snapshot.warm()["skipped"] >= 1
    for text in QUERIES.values():
        assert rows(snapshot.execute(text)) == rows(join_result(graph, text))
    assert provider.cache_stats()["rejected_builds"] >= 1


def test_parameters_and_query_objects(graph):
    snapshot = make_provider(graph).latest()
    text = "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.person_id = $id RETURN b.name"
    expected = join_result(graph, text.replace("$id", "4"))
    by_dict = snapshot.execute(text, parameters={"id": 4})
    assert rows(by_dict) == rows(expected)
    query = CypherQuery(text).with_parameter("id", 4)
    assert rows(snapshot.execute(query)) == rows(expected)
    assert rows(query.execute_with_snapshot(snapshot)) == rows(expected)


def test_deterministic_pages_tile_the_result(graph):
    snapshot = make_provider(graph).latest()
    text = (
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) "
        "WHERE a.person_id IN [1, 2, 3] RETURN a.name, c.name"
    )
    full = snapshot.execute(text, deterministic_order=True).to_pydict()
    total = len(full["a.name"])
    assert total == 12
    ordered = list(zip(full["a.name"], full["c.name"]))
    assert ordered == sorted(ordered)
    pages = []
    for skip in range(0, total, 5):
        page = snapshot.execute(
            f"{text} SKIP {skip} LIMIT 5", deterministic_order=True
        ).to_pydict()
        pages.extend(zip(page["a.name"], page["c.name"]))
        # Same page through a query built with the option.
        again = (
            CypherQuery(f"{text} SKIP {skip} LIMIT 5")
            .with_deterministic_order()
            .execute_with_snapshot(snapshot)
            .to_pydict()
        )
        assert (
            list(zip(again["a.name"], again["c.name"]))
            == pages[-len(again["a.name"]) :]
        )
    assert pages == ordered


def test_read_your_writes_and_exact_versions(graph):
    provider = make_provider(graph, max_staleness_ms=3_600_000)
    before = provider.latest()
    old_versions = before.versions()
    text = QUERIES["hop1"]
    old_rows = rows(before.execute(text))

    # Another node writes: append an edge 3 -> 41 (age 25) and return the new version.
    new_edges = edge_table([(3, 41)])
    lance.write_dataset(new_edges, graph["root"] / "knows_edges.lance", mode="append")
    token = GraphVersions({"knows_edges": edge_version(graph)}).to_token()

    # Within the staleness bound latest() keeps the pinned versions.
    assert provider.latest().versions() == old_versions
    after = provider.at_least(token)
    assert after.versions().covers(GraphVersions.from_token(token))
    assert after.versions().get("knows_edges") == edge_version(graph)
    expected = join_result(
        graph, text, edges=pa.concat_tables([graph["edges"], new_edges])
    )
    assert rows(after.execute(text)) == rows(expected)
    assert rows(after.execute(text)) != old_rows

    # Exact older versions still read the old graph (pagination tokens).
    pinned = provider.at(old_versions.to_token())
    assert pinned.versions() == old_versions
    assert rows(pinned.execute(text)) == old_rows
    assert rows(provider.at(old_versions).execute(text)) == old_rows

    # refresh() moves latest() forward; the new version is derived incrementally.
    assert provider.refresh().versions() == after.versions()
    assert provider.cache_stats()["incremental_builds"] >= 1


def test_versions_token_roundtrip_and_merge():
    a = GraphVersions({"Person": 3, "edges": 5})
    b = GraphVersions({"edges": 7, "other": 1})
    assert GraphVersions.from_token(a.to_token()) == a
    assert a.to_dict() == {"Person": 3, "edges": 5}
    assert a.covers(GraphVersions({"edges": 4}))
    assert not a.covers(b)
    a.merge_max(b)
    assert a.to_dict() == {"Person": 3, "edges": 7, "other": 1}
    assert a.covers(b)
    assert GraphVersions().to_dict() == {}
    assert "GraphVersions(" in repr(a)


def test_invalid_versions_are_errors(graph):
    provider = make_provider(graph)
    with pytest.raises(Exception):
        GraphVersions.from_token("not a token")
    with pytest.raises(TypeError):
        provider.at_least(42)
    with pytest.raises(Exception):
        provider.at(GraphVersions({"knows_edges": 999, "Person": 1}))
    with pytest.raises(Exception):
        provider.at_least(GraphVersions({"knows_edges": 999}))


def test_concurrent_queries_share_one_provider(graph):
    provider = make_provider(graph, max_column_cache_bytes=16 << 20)
    expected = {name: rows(join_result(graph, text)) for name, text in QUERIES.items()}

    def run(i):
        name = sorted(QUERIES)[i % len(QUERIES)]
        return name, rows(provider.latest().execute(QUERIES[name]))

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, range(64)))
    for name, result in results:
        assert result == expected[name]
    # Concurrent first use builds every entry once, like a sequential run.
    sequential = make_provider(graph, max_column_cache_bytes=16 << 20)
    for text in QUERIES.values():
        sequential.latest().execute(text)
    concurrent_stats = provider.cache_stats()
    sequential_stats = sequential.cache_stats()
    for key in ("full_builds", "entries", "column_entries"):
        assert concurrent_stats[key] == sequential_stats[key], key
