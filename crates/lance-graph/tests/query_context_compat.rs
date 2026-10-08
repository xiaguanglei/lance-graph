// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use datafusion::{
    datasource::{DefaultTableSource, MemTable, TableProvider},
    execution::context::SessionContext,
};
use lance_graph::{CypherQuery, GraphConfig, InMemoryCatalog};
use std::sync::Arc;

fn query(text: &str) -> CypherQuery {
    CypherQuery::new(text).unwrap().with_config(
        GraphConfig::builder()
            .with_node_label("thing", "key")
            .build()
            .unwrap(),
    )
}

fn sources() -> (SessionContext, Arc<InMemoryCatalog>) {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(0..10))],
    )
    .unwrap();
    let provider: Arc<dyn TableProvider> =
        Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    let ctx = SessionContext::new();
    ctx.register_table("thing", provider.clone()).unwrap();
    let catalog = InMemoryCatalog::new()
        .with_node_source("thing", Arc::new(DefaultTableSource::new(provider)));
    (ctx, Arc::new(catalog))
}

#[tokio::test]
async fn bounded_context_and_catalog_return_one_extra_row_without_changing_unlimited() {
    for explicit in [false, true] {
        let q = query("MATCH (t:thing) RETURN t.key ORDER BY t.key");
        let (ctx, catalog) = sources();
        let full = if explicit {
            q.execute_with_catalog_and_context(catalog, ctx).await
        } else {
            q.execute_with_context(ctx).await
        }
        .unwrap();
        assert_eq!(full.num_rows(), 10);
        for max_rows in [0, 3, 10] {
            let (ctx, catalog) = sources();
            let bounded = if explicit {
                q.execute_with_catalog_and_context_limited(catalog, ctx, max_rows)
                    .await
            } else {
                q.execute_with_context_limited(ctx, max_rows).await
            }
            .unwrap();
            assert_eq!(bounded.num_rows(), 10.min(max_rows + 1));
            assert_eq!(bounded.schema(), full.schema());
            assert_eq!(
                bounded.column(0),
                full.slice(0, bounded.num_rows()).column(0)
            );
        }
    }
}

#[tokio::test]
async fn bounded_collection_preserves_existing_limit_empty_schema_and_saturated_budget() {
    for (text, max_rows, expected) in [
        ("MATCH (t:thing) RETURN t.key LIMIT 2", 3, 2),
        ("MATCH (t:thing) WHERE t.key < 0 RETURN t.key", 3, 0),
        ("MATCH (t:thing) RETURN t.key", usize::MAX, 10),
    ] {
        let q = query(text);
        let (ctx, _) = sources();
        let full = q.execute_with_context(ctx).await.unwrap();
        let (ctx, catalog) = sources();
        let bounded = q
            .execute_with_catalog_and_context_limited(catalog, ctx, max_rows)
            .await
            .unwrap();
        assert_eq!(bounded.num_rows(), expected);
        assert_eq!(bounded.schema(), full.schema());
        assert_eq!(bounded.column(0), full.column(0));
    }
}

#[tokio::test]
async fn context_explain_uses_registered_provider_and_missing_tables_remain_errors() {
    let q = query("MATCH (t:thing) WHERE t.key > 3 RETURN t.key");
    let (ctx, _) = sources();
    let plan = q.explain_with_context(ctx).await.unwrap();
    assert!(plan.contains("physical_plan"), "{plan}");
    assert!(plan.contains("key"), "{plan}");
    let missing = SessionContext::new();
    assert!(q.explain_with_context(missing.clone()).await.is_err());
    assert!(q.execute_with_context_limited(missing, 3).await.is_err());
}
