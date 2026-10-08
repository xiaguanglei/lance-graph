// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// Compatibility backport of gateway 3b6f3c77 graph mapping contracts.

use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use datafusion::{datasource::MemTable, execution::context::SessionContext};
use lance_graph::{CypherQuery, GraphConfig, NodeMapping, RelationshipMapping};
use std::sync::Arc;

fn context() -> SessionContext {
    let ctx = SessionContext::new();
    let nodes = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Int64, false),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Carol"])),
            Arc::new(Int64Array::from(vec![25, 35, 20])),
        ],
    )
    .unwrap();
    let nodes = Arc::new(MemTable::try_new(nodes.schema(), vec![vec![nodes]]).unwrap());
    ctx.register_table("person", nodes.clone()).unwrap();
    ctx.register_table("friend", nodes).unwrap();
    let edges = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
            Field::new("kind", DataType::Utf8, false),
            Field::new("since", DataType::Int64, true),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![1, 1, 1])),
            Arc::new(Int64Array::from(vec![2, 3, 3])),
            Arc::new(StringArray::from(vec!["KNOWS", "OTHER", "KNOWS"])),
            Arc::new(Int64Array::from(vec![Some(2020), Some(2020), None])),
        ],
    )
    .unwrap();
    ctx.register_table(
        "knows",
        Arc::new(MemTable::try_new(edges.schema(), vec![vec![edges]]).unwrap()),
    )
    .unwrap();
    ctx
}

fn names(batch: &RecordBatch) -> Vec<String> {
    let values = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let mut names = values
        .iter()
        .map(|value| value.unwrap().to_string())
        .collect::<Vec<_>>();
    names.sort();
    names
}

#[tokio::test]
async fn configured_node_and_typed_relationship_filters_match_gateway_contracts() {
    let node_config = GraphConfig::builder()
        .with_node_mapping(NodeMapping::new("person", "id").with_filter("age > 30"))
        .build()
        .unwrap();
    let query = CypherQuery::new("MATCH (p:person) RETURN p.name")
        .unwrap()
        .with_config(node_config);
    assert_eq!(
        names(&query.execute_with_context(context()).await.unwrap()),
        ["Bob"]
    );

    let config = GraphConfig::builder()
        .with_node_label("person", "id")
        .with_relationship_mapping(
            RelationshipMapping::new("KNOWS", "src", "dst")
                .with_type_field("kind")
                .with_filter("since IS NOT NULL"),
        )
        .build()
        .unwrap();
    for text in [
        "MATCH (p:person)-[:KNOWS]->(f:person) RETURN f.name",
        "MATCH (p:person)-[:KNOWS*1..1]->(f:person) RETURN f.name",
        "MATCH (f:person)<-[:KNOWS]-(p:person) RETURN f.name",
    ] {
        let query = CypherQuery::new(text).unwrap().with_config(config.clone());
        assert_eq!(
            names(&query.execute_with_context(context()).await.unwrap()),
            ["Bob"],
            "{text}"
        );
    }
    let invalid = GraphConfig::builder()
        .with_node_mapping(NodeMapping::new("person", "id").with_filter("age > 30 OR age < 20"))
        .build()
        .unwrap();
    let query = CypherQuery::new("MATCH (p:person) RETURN p.name")
        .unwrap()
        .with_config(invalid);
    assert!(query.execute_with_context(context()).await.is_err());
}

#[tokio::test]
async fn target_alias_filter_is_not_lost_when_source_and_target_share_a_table() {
    let config = GraphConfig::builder()
        .with_node_label("person", "id")
        .with_node_mapping(NodeMapping::new("friend", "id").with_filter("age < 30"))
        .with_relationship_mapping(
            RelationshipMapping::new("KNOWS", "src", "dst")
                .with_type_field("kind")
                .with_filter("since IS NOT NULL"),
        )
        .build()
        .unwrap();
    for text in [
        "MATCH (p:person)-[:KNOWS]->(f:friend) RETURN f.name",
        "MATCH (p:person)-[:KNOWS*1..1]->(f:friend) RETURN f.name",
        "MATCH (f:friend)<-[:KNOWS]-(p:person) RETURN f.name",
    ] {
        let result = CypherQuery::new(text)
            .unwrap()
            .with_config(config.clone())
            .execute_with_context(context())
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 0, "{text}");
    }
}
