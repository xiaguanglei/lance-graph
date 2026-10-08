// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Built-in [`TableReader`] implementations for common data formats.
//!
//! - [`ParquetTableReader`] — reads Parquet tables using DataFusion's built-in support.
//! - [`DeltaTableReader`] — reads Delta Lake tables (behind `delta` feature flag).

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::execution::context::SessionContext;

use lance_graph_catalog::catalog_provider::{
    CatalogError, CatalogResult, DataSourceFormat, TableInfo,
};
use lance_graph_catalog::table_reader::TableReader;

/// Reads Parquet tables using DataFusion's built-in `register_parquet()`.
pub struct ParquetTableReader;

#[async_trait]
impl TableReader for ParquetTableReader {
    fn name(&self) -> &str {
        "parquet"
    }

    fn supported_formats(&self) -> &[DataSourceFormat] {
        &[DataSourceFormat::Parquet]
    }

    async fn register_table(
        &self,
        ctx: &SessionContext,
        table_name: &str,
        table_info: &TableInfo,
        _schema: arrow_schema::SchemaRef,
        _storage_options: &HashMap<String, String>,
    ) -> CatalogResult<()> {
        let location = table_info.storage_location.as_deref().ok_or_else(|| {
            CatalogError::Other(format!("Table '{}' has no storage_location", table_name))
        })?;

        ctx.register_parquet(
            table_name,
            location,
            datafusion::datasource::file_format::options::ParquetReadOptions::default(),
        )
        .await
        .map_err(|e| {
            CatalogError::Other(format!(
                "Failed to register Parquet table '{}' at '{}': {}",
                table_name, location, e
            ))
        })
    }
}

/// Reads Delta Lake tables using the `deltalake` crate.
///
/// Opens the Delta table at the storage location and registers it as a
/// DataFusion `TableProvider`, enabling full SQL query support including
/// time travel, schema evolution, and partition pruning.
///
/// Supports cloud storage (S3, Azure, GCS) via `storage_options`.
#[cfg(feature = "delta")]
pub struct DeltaTableReader;

#[cfg(feature = "delta")]
#[async_trait]
impl TableReader for DeltaTableReader {
    fn name(&self) -> &str {
        "delta"
    }

    fn supported_formats(&self) -> &[DataSourceFormat] {
        &[DataSourceFormat::Delta]
    }

    async fn register_table(
        &self,
        ctx: &SessionContext,
        table_name: &str,
        table_info: &TableInfo,
        _schema: arrow_schema::SchemaRef,
        storage_options: &HashMap<String, String>,
    ) -> CatalogResult<()> {
        let location = table_info.storage_location.as_deref().ok_or_else(|| {
            CatalogError::Other(format!("Table '{}' has no storage_location", table_name))
        })?;

        let table_url = url::Url::parse(location).map_err(|e| {
            CatalogError::Other(format!(
                "Invalid storage location URL '{}': {}",
                location, e
            ))
        })?;

        let delta_table = if storage_options.is_empty() {
            deltalake::open_table(table_url).await
        } else {
            deltalake::open_table_with_storage_options(table_url, storage_options.clone()).await
        }
        .map_err(|e| {
            CatalogError::Other(format!(
                "Failed to open Delta table '{}' at '{}': {}",
                table_name, location, e
            ))
        })?;

        let provider = delta_table
            .table_provider()
            .with_session(Arc::new(ctx.state()))
            .await
            .map_err(|e| {
                CatalogError::Other(format!(
                    "Failed to create Delta table provider '{}': {}",
                    table_name, e
                ))
            })?;

        ctx.register_table(table_name, provider).map_err(|e| {
            CatalogError::Other(format!(
                "Failed to register Delta table '{}': {}",
                table_name, e
            ))
        })?;

        Ok(())
    }
}

/// Returns the default set of table readers.
///
/// Includes Parquet support, and Delta Lake support when the `delta` feature is enabled.
pub fn default_table_readers() -> Vec<Arc<dyn TableReader>> {
    let mut readers: Vec<Arc<dyn TableReader>> = vec![Arc::new(ParquetTableReader)];
    #[cfg(feature = "delta")]
    readers.push(Arc::new(DeltaTableReader));
    readers
}

#[cfg(all(test, feature = "delta"))]
mod tests {
    use super::*;
    use arrow_array::{Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use lance_graph_catalog::catalog_provider::TableType;

    fn table_info(location: Option<String>) -> TableInfo {
        TableInfo {
            name: "delta_nodes".into(),
            catalog_name: "test".into(),
            schema_name: "test".into(),
            table_type: TableType::External,
            data_source_format: DataSourceFormat::Delta,
            columns: vec![],
            storage_location: location,
            comment: None,
            properties: HashMap::new(),
            created_at: None,
            updated_at: None,
        }
    }

    #[tokio::test]
    async fn test_delta_reader_registers_local_table_and_filters() {
        let dir = tempfile::tempdir().unwrap();
        let location = url::Url::from_directory_path(dir.path()).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        deltalake::DeltaTable::try_from_url(location.clone())
            .await
            .unwrap()
            .write(vec![batch])
            .await
            .unwrap();

        let ctx = SessionContext::new();
        DeltaTableReader
            .register_table(
                &ctx,
                "delta_nodes",
                &table_info(Some(location.to_string())),
                schema,
                &HashMap::new(),
            )
            .await
            .unwrap();
        let batches = ctx
            .sql("SELECT id FROM delta_nodes WHERE id >= 2 ORDER BY id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let values: Vec<i64> = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect();
        assert_eq!(values, vec![2, 3]);
    }

    #[tokio::test]
    async fn test_delta_reader_rejects_invalid_locations() {
        for (location, expected) in [
            (None, "has no storage_location"),
            (Some("not-a-url".into()), "Invalid storage location URL"),
        ] {
            let err = DeltaTableReader
                .register_table(
                    &SessionContext::new(),
                    "delta_nodes",
                    &table_info(location),
                    Arc::new(Schema::empty()),
                    &HashMap::new(),
                )
                .await
                .unwrap_err();
            assert!(err.to_string().contains(expected), "{err}");
        }
    }
}
