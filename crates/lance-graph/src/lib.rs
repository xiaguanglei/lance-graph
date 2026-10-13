// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Graph Query Engine
//!
//! This crate provides graph query capabilities for Lance datasets using Cypher syntax.
//! It interprets Lance datasets as property graphs and translates Cypher queries into
//! DataFusion SQL queries for execution.
//!
//! # Features
//!
//! - Cypher query parsing and AST representation
//! - Graph pattern matching on columnar data
//! - Property graph interpretation of Lance datasets
//! - Translation to optimized SQL via DataFusion
//! - Support for nodes, relationships, and properties
//!
//! # Example
//!
//! ```no_run
//! use lance_graph::{CypherQuery, GraphConfig, Result};
//!
//! # fn example() -> Result<()> {
//! let config = GraphConfig::builder()
//!     .with_node_label("Person", "person_id")
//!     .with_relationship("KNOWS", "src_person_id", "dst_person_id")
//!     .build()?;
//!
//! let query = CypherQuery::new("MATCH (p:Person) WHERE p.age > 30 RETURN p.name")?
//!     .with_config(config);
//!
//! // Execute against a dataset (would need actual dataset integration)
//! // let results = query.execute(&dataset).await?;
//! # Ok(())
//! # }
//! ```

pub mod ast;
pub mod case_insensitive;
pub mod config;
mod csr_cache;
pub mod csr_index;
mod csr_multihop;
mod csr_provider;
pub mod csr_walk;
pub mod datafusion_planner;
pub mod error;
mod graph_snapshot;
pub mod lance_native_planner;
pub mod lance_vector_search;
pub mod logical_plan;
mod native_session;
pub mod parameter_substitution;
pub mod parser;
pub mod query;
mod runtime_id_filter;
pub mod semantic;
pub mod spark_dialect;
pub mod sql_catalog;
pub mod sql_query;
pub mod table_readers;

/// Maximum allowed hops for variable-length relationship expansion (e.g., *1..N)
pub const MAX_VARIABLE_LENGTH_HOPS: u32 = 20;

pub use config::{GraphConfig, NodeMapping, RelationshipMapping};
pub use csr_index::{
    build_bidirectional_index, try_build_bidirectional_index_with_ids, CsrIndex, CsrIndexBuilder,
};
pub use csr_provider::{CsrAdjacencyCache, CsrCacheStats, CsrScanMetrics, CsrScanOptions};
pub use csr_walk::{
    csr_walk_filtered, csr_walk_steps_filtered, CsrWalk, CsrWalkOptions, CsrWalkResult,
};
pub use error::{GraphError, Result};
pub use graph_snapshot::{GraphSnapshot, GraphSnapshotProvider, GraphVersions, WarmReport};
pub use lance_graph_catalog::{
    DirNamespace, GraphSourceCatalog, InMemoryCatalog, SimpleTableSource,
};
// Catalog provider re-exports
pub use lance_graph_catalog::{
    CatalogError, CatalogInfo, CatalogProvider, CatalogResult, ColumnInfo, Connector,
    DataSourceFormat, SchemaInfo, TableInfo, TableReader, TableType,
};
#[cfg(feature = "unity-catalog")]
pub use lance_graph_catalog::{UnityCatalogConfig, UnityCatalogProvider};
pub use lance_vector_search::VectorSearch;
pub use query::{CypherQuery, ExecutionStrategy, SqlDialect};
pub use runtime_id_filter::RuntimeIdFilterMetrics;
pub use sql_query::SqlQuery;
#[cfg(feature = "delta")]
pub use table_readers::DeltaTableReader;
pub use table_readers::{default_table_readers, ParquetTableReader};
