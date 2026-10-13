// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! High-level Cypher query interface for Lance datasets

use crate::ast::CypherQuery as CypherAST;
use crate::ast::ReadingClause;
use crate::config::GraphConfig;
use crate::error::{GraphError, Result};
use crate::logical_plan::LogicalPlanner;
use crate::parser::parse_cypher_query;
use crate::spark_dialect::build_spark_dialect;
use arrow_array::RecordBatch;
use arrow_schema::{Field, Schema, SchemaRef};
use datafusion_sql::unparser::dialect::{
    CustomDialect, DefaultDialect, MySqlDialect, PostgreSqlDialect, SqliteDialect,
};
use datafusion_sql::unparser::Unparser;
use lance_graph_catalog::DirNamespace;
use lance_namespace::models::DescribeTableRequest;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// SQL dialect to use when generating SQL from Cypher queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SqlDialect {
    /// Generic SQL (DataFusion default dialect)
    #[default]
    Default,
    /// Spark SQL dialect (backtick quoting, STRING type, EXTRACT, etc.)
    Spark,
    /// PostgreSQL dialect
    PostgreSql,
    /// MySQL dialect
    MySql,
    /// SQLite dialect
    Sqlite,
}

/// Wrapper to hold the concrete dialect type and provide an `Unparser` reference.
pub enum DialectUnparser {
    Default(DefaultDialect),
    Spark(Box<CustomDialect>),
    PostgreSql(PostgreSqlDialect),
    MySql(MySqlDialect),
    Sqlite(SqliteDialect),
}

impl DialectUnparser {
    pub fn as_unparser(&self) -> Unparser<'_> {
        match self {
            DialectUnparser::Default(d) => Unparser::new(d),
            DialectUnparser::Spark(d) => Unparser::new(d.as_ref()),
            DialectUnparser::PostgreSql(d) => Unparser::new(d),
            DialectUnparser::MySql(d) => Unparser::new(d),
            DialectUnparser::Sqlite(d) => Unparser::new(d),
        }
    }
}

impl SqlDialect {
    /// Create a `DialectUnparser` configured for this dialect.
    pub fn unparser(&self) -> DialectUnparser {
        match self {
            SqlDialect::Default => DialectUnparser::Default(DefaultDialect {}),
            SqlDialect::Spark => DialectUnparser::Spark(Box::new(build_spark_dialect())),
            SqlDialect::PostgreSql => DialectUnparser::PostgreSql(PostgreSqlDialect {}),
            SqlDialect::MySql => DialectUnparser::MySql(MySqlDialect {}),
            SqlDialect::Sqlite => DialectUnparser::Sqlite(SqliteDialect {}),
        }
    }
}

/// Normalize an Arrow schema to have lowercase field names.
///
/// This ensures that column names in the dataset match the normalized
/// qualified column names used internally (e.g., "fullName" becomes "fullname").
pub(crate) fn normalize_schema(schema: SchemaRef) -> Result<SchemaRef> {
    let fields: Vec<_> = schema
        .fields()
        .iter()
        .map(|f| {
            Arc::new(Field::new(
                f.name().to_lowercase(),
                f.data_type().clone(),
                f.is_nullable(),
            ))
        })
        .collect();
    Ok(Arc::new(Schema::new(fields)))
}

/// Normalize a RecordBatch to have lowercase column names.
///
/// This creates a new RecordBatch with a normalized schema while
/// preserving all the data arrays.
pub(crate) fn normalize_record_batch(batch: &RecordBatch) -> Result<RecordBatch> {
    let normalized_schema = normalize_schema(batch.schema())?;
    RecordBatch::try_new(normalized_schema, batch.columns().to_vec()).map_err(|e| {
        GraphError::PlanError {
            message: format!("Failed to normalize record batch schema: {}", e),
            location: snafu::Location::new(file!(), line!(), column!()),
        }
    })
}

/// Execution strategy for Cypher queries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionStrategy {
    /// Use DataFusion query planner (default, full feature support)
    #[default]
    DataFusion,
    /// Use Lance native executor (not yet implemented)
    LanceNative,
}

/// A Cypher query that can be executed against Lance datasets
#[derive(Debug, Clone)]
pub struct CypherQuery {
    /// The original Cypher query string
    query_text: String,
    /// Parsed AST representation
    ast: CypherAST,
    /// Graph configuration for mapping
    config: Option<GraphConfig>,
    /// Query parameters
    parameters: HashMap<String, serde_json::Value>,
    /// Optional completed JOIN membership consumer for indexed Lance nodes.
    runtime_id_filter_metrics: Option<Arc<crate::RuntimeIdFilterMetrics>>,
    /// Snapshot-local adjacency candidates; original Cypher filters and JOINs remain.
    csr_adjacency_enabled: bool,
    csr_native_multihop_enabled: bool,
    csr_adjacency_cache: Arc<crate::CsrAdjacencyCache>,
    csr_adjacency_metrics: Arc<crate::CsrScanMetrics>,
    /// Order rows by the result itself (see `with_deterministic_order`).
    deterministic_order: bool,
}
impl CypherQuery {
    /// Create a new Cypher query from a query string
    pub fn new(query: &str) -> Result<Self> {
        let ast = parse_cypher_query(query)?;

        Ok(Self {
            query_text: query.to_string(),
            ast,
            config: None,
            parameters: HashMap::new(),
            runtime_id_filter_metrics: None,
            csr_adjacency_enabled: true,
            csr_native_multihop_enabled: true,
            csr_adjacency_cache: Arc::new(crate::CsrAdjacencyCache::default()),
            csr_adjacency_metrics: Arc::new(crate::CsrScanMetrics::default()),
            deterministic_order: false,
        })
    }

    /// Set the graph configuration for this query
    pub fn with_config(mut self, config: GraphConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Enable or disable bounded CSR adjacency scans in the DataFusion executor.
    ///
    /// Enabled by default for Lance relationship endpoints bound statically or
    /// by a completed, proven null-rejecting inner JOIN. Existing endpoint BTree
    /// indexes are preferred over building adjacency. Unsupported predicates,
    /// sources, producers and exhausted budgets keep the original scan.
    /// Original JOINs and residual filters preserve Cypher multiplicity,
    /// relationship properties and caller visibility rules.
    pub fn with_csr_adjacency(mut self, enabled: bool) -> Self {
        self.csr_adjacency_enabled = enabled;
        self
    }

    /// Disable only native fixed-path traversal (one to three hops, seeded by
    /// bound source IDs) when comparing against the unchanged per-hop CSR/JOIN
    /// executor. Other adjacency and version rules remain.
    pub fn with_native_multihop(mut self, enabled: bool) -> Self {
        self.csr_native_multihop_enabled = enabled;
        self
    }

    /// Cheap AST-shape hint for choosing replayable table providers: one fixed
    /// one- to three-hop MATCH without WITH. This does not validate storage,
    /// bindings, predicates, indexes, budgets or actual native execution, and
    /// is intentionally independent of execution flags and parameter values.
    /// Use `csr_adjacency_metrics()` after execution to prove consumption.
    pub fn native_multihop_candidate(&self) -> bool {
        crate::csr_multihop::supports_query(&self.ast)
    }

    /// Make the row order a function of the result rows, identical on every
    /// node and execution path: an existing ORDER BY gets every remaining
    /// column as a tie-breaker, otherwise all orderable output columns are
    /// sorted, below any SKIP/LIMIT so pages never overlap or skip rows.
    /// Without it only the bag of rows is defined (a native CSR path and the
    /// JOIN plan, or two nodes, may return a different order or LIMIT subset).
    /// Nested columns (structs, lists, vectors) are not used as sort keys.
    pub fn with_deterministic_order(mut self, enabled: bool) -> Self {
        self.deterministic_order = enabled;
        self
    }

    /// Execute against a version-pinned [`crate::GraphSnapshot`] (every table
    /// at a fixed version) with the snapshot's shared CSR cache; the query's
    /// own cache setting is replaced, its metrics are kept. The snapshot's
    /// graph config is used unless the query has one.
    pub async fn execute_with_snapshot(
        &self,
        snapshot: &crate::GraphSnapshot,
    ) -> Result<RecordBatch> {
        let mut query = self.clone().with_shared_csr_cache(snapshot.cache().clone());
        if query.config.is_none() {
            query.config = Some(snapshot.config().clone());
        }
        let (ctx, native_ctx) = snapshot.contexts();
        query
            .execute_with_catalog_and_context_inner(
                snapshot.catalog(),
                ctx.clone(),
                Some(native_ctx.clone()),
                None,
            )
            .await
    }

    /// Reuse adjacency across queries in the same caller's storage context.
    ///
    /// Entries are pinned to the committed version's content identity; a new
    /// version never consumes stale adjacency. This is an explicit bounded
    /// cache, not a global cache of query results or ACL decisions. Without this
    /// method the cache lives with this query (and its clones) only.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use lance_graph::{CsrAdjacencyCache, CypherQuery};
    /// # fn example() -> lance_graph::Result<()> {
    /// let cache = Arc::new(CsrAdjacencyCache::default());
    /// let query = CypherQuery::new(
    ///     "MATCH (a:Person)-[r:KNOWS]->(b:Person) WHERE a.person_id = 'p1' RETURN b.person_id"
    /// )?.with_shared_csr_cache(cache.clone());
    /// // Reuse `cache` with queries over the same pinned source catalog.
    /// # let _ = query;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_shared_csr_cache(mut self, cache: Arc<crate::CsrAdjacencyCache>) -> Self {
        self.csr_adjacency_cache = cache;
        self
    }

    /// Execution counters distinguish actual CSR reads from planned/fallback scans.
    pub fn csr_adjacency_metrics(&self) -> Arc<crate::CsrScanMetrics> {
        self.csr_adjacency_metrics.clone()
    }

    /// Opt in to bounded completed JOIN ID filters on UTF-8 Lance node IDs.
    ///
    /// Node ID fields come from GraphConfig. Only snapshot-local BTree indexed
    /// nodes participate; other sources, relationship scans, static ID filters,
    /// ordered/limited scans and unsupported runtime expressions stay unchanged.
    /// The original JOIN is retained. An incomplete producer times out after
    /// 100ms per source and falls back to a fresh original scan. This wait stays
    /// inside the caller's existing query future and cancellation budget.
    ///
    /// The default is disabled. Counters are bounded, with no retained ID data.
    pub fn with_runtime_id_filters(mut self, metrics: Arc<crate::RuntimeIdFilterMetrics>) -> Self {
        self.runtime_id_filter_metrics = Some(metrics);
        self
    }

    /// Add a parameter to the query
    pub fn with_parameter<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.parameters
            .insert(key.into().to_lowercase(), value.into());
        self
    }

    /// Add multiple parameters to the query
    pub fn with_parameters(mut self, params: HashMap<String, serde_json::Value>) -> Self {
        for (k, v) in params {
            self.parameters.insert(k.to_lowercase(), v);
        }
        self
    }

    /// Get the original query text
    pub fn query_text(&self) -> &str {
        &self.query_text
    }

    /// Get the parsed AST
    pub fn ast(&self) -> &CypherAST {
        &self.ast
    }

    /// Get the graph configuration
    pub fn config(&self) -> Option<&GraphConfig> {
        self.config.as_ref()
    }

    /// Get query parameters
    pub fn parameters(&self) -> &HashMap<String, serde_json::Value> {
        &self.parameters
    }

    /// Get the required config, returning an error if not set
    fn require_config(&self) -> Result<&GraphConfig> {
        self.config.as_ref().ok_or_else(|| GraphError::ConfigError {
            message: "Graph configuration is required for query execution".to_string(),
            location: snafu::Location::new(file!(), line!(), column!()),
        })
    }

    /// Execute the query against provided in-memory datasets
    ///
    /// This method uses the DataFusion planner by default for comprehensive query support
    /// including joins, aggregations, and complex patterns. You can optionally specify
    /// a different execution strategy.
    ///
    /// # Arguments
    /// * `datasets` - HashMap of table name to RecordBatch (nodes and relationships)
    /// * `strategy` - Optional execution strategy (defaults to DataFusion)
    ///
    /// # Returns
    /// A single RecordBatch containing the query results
    ///
    /// # Errors
    /// Returns error if query parsing, planning, or execution fails
    ///
    /// # Example
    /// ```ignore
    /// use std::collections::HashMap;
    /// use arrow::record_batch::RecordBatch;
    /// use lance_graph::query::CypherQuery;
    ///
    /// // Create in-memory datasets
    /// let mut datasets = HashMap::new();
    /// datasets.insert("Person".to_string(), person_batch);
    /// datasets.insert("KNOWS".to_string(), knows_batch);
    ///
    /// // Parse and execute query
    /// let query = CypherQuery::parse("MATCH (p:Person)-[:KNOWS]->(f) RETURN p.name, f.name")?
    ///     .with_config(config);
    /// // Use the default DataFusion strategy
    /// let result = query.execute(datasets, None).await?;
    /// ```
    pub async fn execute(
        &self,
        datasets: HashMap<String, arrow::record_batch::RecordBatch>,
        strategy: Option<ExecutionStrategy>,
    ) -> Result<arrow::record_batch::RecordBatch> {
        let strategy = strategy.unwrap_or_default();
        match strategy {
            ExecutionStrategy::DataFusion => self.execute_datafusion(datasets).await,
            ExecutionStrategy::LanceNative => Err(GraphError::UnsupportedFeature {
                feature: "Lance native execution strategy is not yet implemented".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            }),
        }
    }

    /// Execute the query using a namespace-backed table resolver.
    ///
    /// The namespace is provided by value and will be shared internally as needed.
    pub async fn execute_with_namespace(
        &self,
        namespace: DirNamespace,
        strategy: Option<ExecutionStrategy>,
    ) -> Result<arrow::record_batch::RecordBatch> {
        self.execute_with_namespace_arc(std::sync::Arc::new(namespace), strategy)
            .await
    }

    /// Execute the query using a shared namespace instance.
    pub async fn execute_with_namespace_arc(
        &self,
        namespace: std::sync::Arc<DirNamespace>,
        strategy: Option<ExecutionStrategy>,
    ) -> Result<arrow::record_batch::RecordBatch> {
        let namespace_trait: std::sync::Arc<dyn lance_namespace::LanceNamespace + Send + Sync> =
            namespace;
        self.execute_with_namespace_internal(namespace_trait, strategy)
            .await
    }

    async fn execute_with_namespace_internal(
        &self,
        namespace: std::sync::Arc<dyn lance_namespace::LanceNamespace + Send + Sync>,
        strategy: Option<ExecutionStrategy>,
    ) -> Result<arrow::record_batch::RecordBatch> {
        let strategy = strategy.unwrap_or_default();
        match strategy {
            ExecutionStrategy::DataFusion => {
                let (catalog, ctx) = self
                    .build_catalog_and_context_from_namespace(namespace)
                    .await?;
                self.execute_with_catalog_and_context(std::sync::Arc::new(catalog), ctx)
                    .await
            }
            ExecutionStrategy::LanceNative => Err(GraphError::UnsupportedFeature {
                feature: "Lance native execution strategy is not yet implemented".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            }),
        }
    }

    /// Explain the query execution plan using in-memory datasets
    ///
    /// Returns a formatted string showing the query execution plan at different stages:
    /// - Graph Logical Plan (graph-specific operators)
    /// - DataFusion Logical Plan (optimized relational plan)
    /// - DataFusion Physical Plan (execution plan with optimizations)
    ///
    /// This is useful for understanding query performance, debugging, and optimization.
    ///
    /// # Arguments
    /// * `datasets` - HashMap of table name to RecordBatch (nodes and relationships)
    ///
    /// # Returns
    /// A formatted string containing the execution plan at multiple levels
    ///
    /// # Errors
    /// Returns error if planning fails
    ///
    /// # Example
    /// ```ignore
    /// use std::collections::HashMap;
    /// use arrow::record_batch::RecordBatch;
    /// use lance_graph::query::CypherQuery;
    ///
    /// // Create in-memory datasets
    /// let mut datasets = HashMap::new();
    /// datasets.insert("Person".to_string(), person_batch);
    /// datasets.insert("KNOWS".to_string(), knows_batch);
    ///
    /// let query = CypherQuery::parse("MATCH (p:Person) WHERE p.age > 30 RETURN p.name")?
    ///     .with_config(config);
    ///
    /// let plan = query.explain(datasets).await?;
    /// println!("{}", plan);
    /// ```
    pub async fn explain(
        &self,
        datasets: HashMap<String, arrow::record_batch::RecordBatch>,
    ) -> Result<String> {
        use std::sync::Arc;

        // Build catalog and context from datasets
        let (catalog, ctx) = self
            .build_catalog_and_context_from_datasets(datasets)
            .await?;

        // Delegate to the internal explain method
        self.explain_internal(Arc::new(catalog), ctx).await
    }

    /// Explain using the actual table providers used by graph execution.
    /// The batch-based `explain` API only plans over in-memory schema batches.
    pub async fn explain_with_context(
        &self,
        ctx: datafusion::execution::context::SessionContext,
    ) -> Result<String> {
        let catalog = self.catalog_from_context(&ctx).await?;
        self.explain_internal(catalog, ctx).await
    }

    /// Convert the Cypher query to a SQL string in the specified dialect.
    ///
    /// This method generates a SQL string that corresponds to the DataFusion logical plan
    /// derived from the Cypher query, using the specified SQL dialect for unparsing.
    ///
    /// **WARNING**: This method is experimental and the generated SQL dialect may change.
    ///
    /// **Note**: All table names and column names in the generated SQL are lowercased
    /// (e.g., `Person` becomes `person`, `fullName` becomes `fullname`), consistent with
    /// the system's case-insensitive identifier behavior.
    ///
    /// # Arguments
    /// * `datasets` - HashMap of table name to RecordBatch (nodes and relationships)
    /// * `dialect` - The SQL dialect to use for generating the output SQL.
    ///   Defaults to `SqlDialect::Default` (generic DataFusion SQL).
    ///   Use `SqlDialect::Spark` for Spark SQL, `SqlDialect::PostgreSql`, etc.
    ///
    /// # Returns
    /// A SQL string representing the query in the specified dialect
    pub async fn to_sql(
        &self,
        datasets: HashMap<String, arrow::record_batch::RecordBatch>,
        dialect: Option<SqlDialect>,
    ) -> Result<String> {
        use std::sync::Arc;

        let dialect = dialect.unwrap_or_default();
        let _config = self.require_config()?;

        // Build catalog and context from datasets using the helper
        let (catalog, ctx) = self
            .build_catalog_and_context_from_datasets(datasets)
            .await?;

        // Generate Logical Plan
        let (_, df_plan) = self.create_logical_plans(Arc::new(catalog))?;

        // Optimize the plan using DataFusion's default optimizer rules
        // This helps simplify the plan (e.g., merging projections) to produce cleaner SQL
        let optimized_plan = ctx
            .state()
            .optimize(&df_plan)
            .map_err(|e| GraphError::PlanError {
                message: format!("Failed to optimize plan: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        // Unparse to SQL using the specified dialect
        let dialect_unparser = dialect.unparser();
        let unparser = dialect_unparser.as_unparser();
        let sql_ast = unparser
            .plan_to_sql(&optimized_plan)
            .map_err(|e| GraphError::PlanError {
                message: format!("Failed to unparse plan to SQL: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        Ok(sql_ast.to_string())
    }

    /// Execute query with a DataFusion SessionContext, automatically building the catalog
    ///
    /// This is a convenience method that builds the graph catalog by querying the
    /// SessionContext for table schemas. The GraphConfig determines which tables to
    /// look up (node labels and relationship types).
    ///
    /// This method is ideal for integrating with DataFusion's rich data source ecosystem
    /// (CSV, Parquet, Delta Lake, Iceberg, etc.) without manually building a catalog.
    ///
    /// # Arguments
    /// * `ctx` - DataFusion SessionContext with pre-registered tables
    ///
    /// # Returns
    /// Query results as an Arrow RecordBatch
    ///
    /// # Errors
    /// Returns error if:
    /// - GraphConfig is not set (use `.with_config()` first)
    /// - Required tables are not registered in the SessionContext
    /// - Query execution fails
    ///
    /// # Example
    /// ```ignore
    /// use datafusion::execution::context::SessionContext;
    /// use datafusion::prelude::CsvReadOptions;
    /// use lance_graph::{CypherQuery, GraphConfig};
    ///
    /// // Step 1: Create GraphConfig
    /// let config = GraphConfig::builder()
    ///     .with_node_label("Person", "person_id")
    ///     .with_relationship("KNOWS", "src_id", "dst_id")
    ///     .build()?;
    ///
    /// // Step 2: Register data sources in DataFusion
    /// let ctx = SessionContext::new();
    /// ctx.register_csv("Person", "data/persons.csv", CsvReadOptions::default()).await?;
    /// ctx.register_parquet("KNOWS", "s3://bucket/knows.parquet", Default::default()).await?;
    ///
    /// // Step 3: Execute query (catalog is built automatically)
    /// let query = CypherQuery::parse("MATCH (p:Person)-[:KNOWS]->(f) RETURN p.name")?
    ///     .with_config(config);
    /// let result = query.execute_with_context(ctx).await?;
    /// ```
    ///
    /// # Note
    /// The catalog is built by querying the SessionContext for schemas of tables
    /// mentioned in the GraphConfig. Table names must match between GraphConfig
    /// (node labels/relationship types) and SessionContext (registered table names).
    pub async fn execute_with_context(
        &self,
        ctx: datafusion::execution::context::SessionContext,
    ) -> Result<arrow::record_batch::RecordBatch> {
        let catalog = self.catalog_from_context(&ctx).await?;
        self.execute_with_catalog_and_context(catalog, ctx).await
    }

    /// Execute against a context while collecting at most `max_rows + 1` rows.
    ///
    /// The extra row lets an HTTP caller reject an oversized result explicitly
    /// rather than silently truncating it, without first materializing the
    /// entire result set.
    pub async fn execute_with_context_limited(
        &self,
        ctx: datafusion::execution::context::SessionContext,
        max_rows: usize,
    ) -> Result<arrow::record_batch::RecordBatch> {
        let catalog = self.catalog_from_context(&ctx).await?;
        self.execute_with_catalog_and_context_limited(catalog, ctx, max_rows)
            .await
    }

    async fn catalog_from_context(
        &self,
        ctx: &datafusion::execution::context::SessionContext,
    ) -> Result<std::sync::Arc<dyn lance_graph_catalog::GraphSourceCatalog>> {
        use datafusion::datasource::DefaultTableSource;
        use lance_graph_catalog::InMemoryCatalog;
        use std::sync::Arc;

        let config = self.require_config()?;
        let mut catalog = InMemoryCatalog::new();
        for label in config.node_mappings.keys() {
            let table_provider =
                ctx.table_provider(label)
                    .await
                    .map_err(|e| GraphError::ConfigError {
                        message: format!(
                            "Node label '{}' not found in SessionContext: {}",
                            label, e
                        ),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    })?;
            catalog =
                catalog.with_node_source(label, Arc::new(DefaultTableSource::new(table_provider)));
        }
        for rel_type in config.relationship_mappings.keys() {
            let table_provider =
                ctx.table_provider(rel_type)
                    .await
                    .map_err(|e| GraphError::ConfigError {
                        message: format!(
                            "Relationship type '{}' not found in SessionContext: {}",
                            rel_type, e
                        ),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    })?;
            catalog = catalog.with_relationship_source(
                rel_type,
                Arc::new(DefaultTableSource::new(table_provider)),
            );
        }
        Ok(Arc::new(catalog))
    }

    /// Execute query with an explicit catalog and session context
    ///
    /// This is the most flexible API for advanced users who want to provide their own
    /// catalog implementation or have fine-grained control over both the catalog and
    /// session context.
    ///
    /// # Arguments
    /// * `catalog` - Graph catalog containing node and relationship schemas for planning
    /// * `ctx` - DataFusion SessionContext with registered data sources for execution
    ///
    /// # Returns
    /// Query results as an Arrow RecordBatch
    ///
    /// # Errors
    /// Returns error if query parsing, planning, or execution fails
    ///
    /// # Example
    /// ```ignore
    /// use std::sync::Arc;
    /// use datafusion::execution::context::SessionContext;
    /// use lance_graph::InMemoryCatalog;
    /// use lance_graph::query::CypherQuery;
    ///
    /// // Create custom catalog
    /// let catalog = InMemoryCatalog::new()
    ///     .with_node_source("Person", custom_table_source);
    ///
    /// // Create SessionContext
    /// let ctx = SessionContext::new();
    /// ctx.register_table("Person", custom_table).unwrap();
    ///
    /// // Execute with explicit catalog and context
    /// let query = CypherQuery::parse("MATCH (p:Person) RETURN p.name")?
    ///     .with_config(config);
    /// let result = query.execute_with_catalog_and_context(Arc::new(catalog), ctx).await?;
    /// ```
    pub async fn execute_with_catalog_and_context(
        &self,
        catalog: std::sync::Arc<dyn lance_graph_catalog::GraphSourceCatalog>,
        ctx: datafusion::execution::context::SessionContext,
    ) -> Result<arrow::record_batch::RecordBatch> {
        self.execute_with_catalog_and_context_inner(catalog, ctx, None, None)
            .await
    }

    /// The bounded variant of [`Self::execute_with_catalog_and_context`].
    pub async fn execute_with_catalog_and_context_limited(
        &self,
        catalog: std::sync::Arc<dyn lance_graph_catalog::GraphSourceCatalog>,
        ctx: datafusion::execution::context::SessionContext,
        max_rows: usize,
    ) -> Result<arrow::record_batch::RecordBatch> {
        self.execute_with_catalog_and_context_inner(catalog, ctx, None, Some(max_rows))
            .await
    }

    pub(crate) async fn execute_with_catalog_and_context_inner(
        &self,
        catalog: std::sync::Arc<dyn lance_graph_catalog::GraphSourceCatalog>,
        ctx: datafusion::execution::context::SessionContext,
        native_ctx: Option<datafusion::execution::context::SessionContext>,
        max_rows: Option<usize>,
    ) -> Result<arrow::record_batch::RecordBatch> {
        use arrow::compute::concat_batches;

        // Create logical plans (phases 1-3)
        let (_logical_plan, df_logical_plan) = self.create_logical_plans(catalog)?;
        // A plan driven by one native CSR operator produces a small result on
        // one partition: plan and run it without repartitioning.
        let ctx = match native_ctx {
            Some(native_ctx)
                if crate::csr_multihop::is_native_plan(&df_logical_plan)
                    && crate::native_session::lean_plan(&df_logical_plan) =>
            {
                native_ctx
            }
            _ => ctx,
        };

        // Execute the DataFusion plan (phase 4)
        let df = ctx
            .execute_logical_plan(df_logical_plan)
            .await
            .map_err(|e| GraphError::ExecutionError {
                message: format!("Failed to execute DataFusion plan: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        let df = if let Some(max_rows) = max_rows {
            // DataFusion represents FETCH as an i64 literal. A saturated
            // usize budget must not become a negative limit on 64-bit hosts.
            let fetch = max_rows.saturating_add(1).min(i64::MAX as usize);
            df.limit(0, Some(fetch))
                .map_err(|e| GraphError::ExecutionError {
                    message: format!("Failed to apply result limit: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?
        } else {
            df
        };

        // Get schema before collecting (in case result is empty)
        let result_schema = df.schema().inner().clone();

        // Collect results
        let collected = if self.runtime_id_filter_metrics.is_some() || self.csr_adjacency_enabled {
            // Physical producer identity is available only after post-phase
            // pushdown. Bind then collect the same bounded logical DataFrame.
            let task_context = Arc::new(df.task_ctx());
            match df.create_physical_plan().await {
                Ok(plan) => match self.bind_runtime_sources(plan) {
                    Ok(plan) => datafusion::physical_plan::collect(plan, task_context).await,
                    Err(error) => Err(error),
                },
                Err(error) => Err(error),
            }
        } else {
            df.collect().await
        };
        let batches = collected.map_err(|e| GraphError::ExecutionError {
            message: format!("Failed to collect query results: {}", e),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;

        if batches.is_empty() {
            // Return empty batch with the schema from the DataFrame
            // This preserves column structure even when there are no rows
            return Ok(arrow::record_batch::RecordBatch::new_empty(result_schema));
        }

        // Combine all batches
        let schema = batches[0].schema();
        concat_batches(&schema, &batches).map_err(|e| GraphError::ExecutionError {
            message: format!("Failed to concatenate result batches: {}", e),
            location: snafu::Location::new(file!(), line!(), column!()),
        })
    }

    /// Execute using the DataFusion planner with in-memory datasets
    ///
    /// # Overview
    /// This convenience method creates both a catalog and session context from the provided
    /// in-memory RecordBatches. It's ideal for testing and small datasets that fit in memory.
    ///
    /// For production use with external data sources (CSV, Parquet, databases), use
    /// `execute_with_context` instead, which automatically builds the catalog
    /// from the SessionContext.
    ///
    /// # Arguments
    /// * `datasets` - HashMap of table name to RecordBatch (nodes and relationships)
    ///
    /// # Returns
    /// A single RecordBatch containing the query results
    async fn execute_datafusion(
        &self,
        datasets: HashMap<String, arrow::record_batch::RecordBatch>,
    ) -> Result<arrow::record_batch::RecordBatch> {
        use std::sync::Arc;

        // Build catalog and context from datasets
        let (catalog, ctx) = self
            .build_catalog_and_context_from_datasets(datasets)
            .await?;

        // Delegate to common execution logic
        self.execute_with_catalog_and_context(Arc::new(catalog), ctx)
            .await
    }

    /// Helper to build catalog and context from in-memory datasets
    async fn build_catalog_and_context_from_datasets(
        &self,
        datasets: HashMap<String, arrow::record_batch::RecordBatch>,
    ) -> Result<(
        lance_graph_catalog::InMemoryCatalog,
        datafusion::execution::context::SessionContext,
    )> {
        use datafusion::datasource::{DefaultTableSource, MemTable};
        use datafusion::execution::context::SessionContext;
        use lance_graph_catalog::InMemoryCatalog;
        use std::sync::Arc;

        if datasets.is_empty() {
            return Err(GraphError::ConfigError {
                message: "No input datasets provided".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            });
        }

        // Create session context and catalog
        let ctx = SessionContext::new();
        let mut catalog = InMemoryCatalog::new();

        // Register all datasets as tables
        for (name, batch) in &datasets {
            // Normalize the schema to lowercase field names
            let normalized_batch = normalize_record_batch(batch)?;

            let mem_table = Arc::new(
                MemTable::try_new(
                    normalized_batch.schema(),
                    vec![vec![normalized_batch.clone()]],
                )
                .map_err(|e| GraphError::PlanError {
                    message: format!("Failed to create MemTable for {}: {}", name, e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?,
            );

            // Normalize table name to lowercase
            let normalized_name = name.to_lowercase();

            // Register in session context for execution
            ctx.register_table(&normalized_name, mem_table.clone())
                .map_err(|e| GraphError::PlanError {
                    message: format!("Failed to register table {}: {}", name, e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

            let table_source = Arc::new(DefaultTableSource::new(mem_table));

            // Register as both node and relationship source with original name
            // The config lookup is already case-insensitive, so we can use original name
            catalog = catalog
                .with_node_source(name, table_source.clone())
                .with_relationship_source(name, table_source);
        }

        Ok((catalog, ctx))
    }

    /// Helper to build catalog and context using a namespace resolver
    async fn build_catalog_and_context_from_namespace(
        &self,
        namespace: std::sync::Arc<dyn lance_namespace::LanceNamespace + Send + Sync>,
    ) -> Result<(
        lance_graph_catalog::InMemoryCatalog,
        datafusion::execution::context::SessionContext,
    )> {
        use datafusion::datasource::{DefaultTableSource, TableProvider};
        use datafusion::execution::context::SessionContext;
        use lance::datafusion::LanceTableProvider;
        use lance_graph_catalog::InMemoryCatalog;
        use std::sync::Arc;

        let config = self.require_config()?;

        let mut required_tables: HashSet<String> = HashSet::new();
        // Use original label/type names (not lowercase keys) for namespace resolution
        // The namespace needs the original casing to find files on disk
        required_tables.extend(config.node_mappings.values().map(|m| m.label.clone()));
        required_tables.extend(
            config
                .relationship_mappings
                .values()
                .map(|m| m.relationship_type.clone()),
        );

        if required_tables.is_empty() {
            return Err(GraphError::ConfigError {
                message:
                    "Graph configuration does not reference any node labels or relationship types"
                        .to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            });
        }

        let ctx = SessionContext::new();
        let mut catalog = InMemoryCatalog::new();
        let mut providers: HashMap<String, Arc<dyn TableProvider>> = HashMap::new();

        for table_name in required_tables {
            let mut request = DescribeTableRequest::new();
            request.id = Some(vec![table_name.clone()]);

            let response =
                namespace
                    .describe_table(request)
                    .await
                    .map_err(|e| GraphError::ConfigError {
                        message: format!(
                            "Namespace failed to resolve table '{}': {}",
                            table_name, e
                        ),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    })?;

            let location = response.location.ok_or_else(|| GraphError::ConfigError {
                message: format!(
                    "Namespace did not provide a location for table '{}'",
                    table_name
                ),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

            let dataset = lance::dataset::Dataset::open(&location)
                .await
                .map_err(|e| GraphError::ConfigError {
                    message: format!("Failed to open dataset for table '{}': {}", table_name, e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

            let dataset = Arc::new(dataset);
            let provider: Arc<dyn TableProvider> =
                Arc::new(LanceTableProvider::new(dataset.clone(), true, true));

            // Register with lowercase table name for case-insensitive behavior
            let normalized_table_name = table_name.to_lowercase();
            ctx.register_table(&normalized_table_name, provider.clone())
                .map_err(|e| GraphError::PlanError {
                    message: format!(
                        "Failed to register table '{}' in SessionContext: {}",
                        table_name, e
                    ),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

            // Store provider with normalized (lowercase) key for consistent lookup
            providers.insert(normalized_table_name.clone(), provider);
        }

        for label in config.node_mappings.keys() {
            let provider = providers
                .get(label)
                .ok_or_else(|| GraphError::ConfigError {
                    message: format!(
                        "Namespace did not resolve dataset for node label '{}'",
                        label
                    ),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

            let table_source = Arc::new(DefaultTableSource::new(provider.clone()));
            catalog = catalog.with_node_source(label, table_source);
        }

        for rel_type in config.relationship_mappings.keys() {
            let provider = providers
                .get(rel_type)
                .ok_or_else(|| GraphError::ConfigError {
                    message: format!(
                        "Namespace did not resolve dataset for relationship type '{}'",
                        rel_type
                    ),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

            let table_source = Arc::new(DefaultTableSource::new(provider.clone()));
            catalog = catalog.with_relationship_source(rel_type, table_source);
        }

        Ok((catalog, ctx))
    }

    /// Internal helper to explain the query execution plan with explicit catalog and session context
    async fn explain_internal(
        &self,
        catalog: std::sync::Arc<dyn lance_graph_catalog::GraphSourceCatalog>,
        ctx: datafusion::execution::context::SessionContext,
    ) -> Result<String> {
        // Create all plans (phases 1-4)
        let (logical_plan, df_logical_plan, physical_plan) =
            self.create_plans(catalog, &ctx).await?;

        // Format the explain output
        self.format_explain_output(&logical_plan, &df_logical_plan, physical_plan.as_ref())
    }

    /// Helper to create logical plans (graph logical, DataFusion logical)
    ///
    /// This performs phases 1-3 of query execution (semantic analysis, graph logical planning,
    /// DataFusion logical planning) without creating the physical plan.
    fn create_logical_plans(
        &self,
        catalog: std::sync::Arc<dyn lance_graph_catalog::GraphSourceCatalog>,
    ) -> Result<(
        crate::logical_plan::LogicalOperator,
        datafusion::logical_expr::LogicalPlan,
    )> {
        use crate::datafusion_planner::{DataFusionPlanner, GraphPhysicalPlanner};
        use crate::semantic::SemanticAnalyzer;

        let config = self.require_config()?;

        // Phase 1: Semantic Analysis
        let mut analyzer = SemanticAnalyzer::new(config.clone());
        let semantic = analyzer.analyze(&self.ast, &self.parameters)?;
        if !semantic.errors.is_empty() {
            return Err(GraphError::PlanError {
                message: format!("Semantic analysis failed:\n{}", semantic.errors.join("\n")),
                location: snafu::Location::new(file!(), line!(), column!()),
            });
        }

        // Phase 2: Graph Logical Plan
        let mut logical_planner = LogicalPlanner::new(config);
        // Use the transformed AST (with parameters substituted)
        let logical_plan = logical_planner.plan(&semantic.ast)?;

        // Phase 3: DataFusion Logical Plan
        let catalog = if self.csr_adjacency_enabled {
            Arc::new(crate::csr_provider::CsrSourceCatalog::new(
                catalog,
                config.clone(),
                self.csr_adjacency_cache.clone(),
                self.csr_adjacency_metrics.clone(),
            )) as Arc<dyn lance_graph_catalog::GraphSourceCatalog>
        } else {
            catalog
        };
        let catalog = if let Some(metrics) = &self.runtime_id_filter_metrics {
            Arc::new(crate::runtime_id_filter::RuntimeIdSourceCatalog {
                inner: catalog,
                config: config.clone(),
                metrics: metrics.clone(),
            }) as Arc<dyn lance_graph_catalog::GraphSourceCatalog>
        } else {
            catalog
        };
        let mut df_planner = DataFusionPlanner::with_catalog(config.clone(), catalog);
        if self.csr_adjacency_enabled
            && self.csr_native_multihop_enabled
            && crate::csr_multihop::supports_query(&semantic.ast)
        {
            df_planner.native_multihop = Some(crate::csr_multihop::NativeMultihopOptions {
                cache: self.csr_adjacency_cache.clone(),
                metrics: self.csr_adjacency_metrics.clone(),
                predicate: semantic.ast.where_clause.as_ref().map(|clause| {
                    crate::datafusion_planner::expression::to_df_boolean_expr(&clause.expression)
                }),
            });
        }
        let df_logical_plan = df_planner.plan(&logical_plan)?;
        let df_logical_plan = if self.deterministic_order {
            deterministic_order(df_logical_plan).map_err(|e| GraphError::PlanError {
                message: format!("Failed to order results deterministically: {e}"),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?
        } else {
            df_logical_plan
        };

        Ok((logical_plan, df_logical_plan))
    }

    /// Bind optional consumers only after the final optimizer has identified
    /// their actual null-rejecting inner-JOIN producers. A source cannot certify
    /// its own dynamic filter merely because an expression reached its callback.
    fn bind_runtime_sources(
        &self,
        mut plan: Arc<dyn datafusion::physical_plan::ExecutionPlan>,
    ) -> datafusion::common::Result<Arc<dyn datafusion::physical_plan::ExecutionPlan>> {
        if self.runtime_id_filter_metrics.is_some() {
            plan = crate::runtime_id_filter::bind_runtime_id_sources(plan)?;
        }
        if self.csr_adjacency_enabled {
            plan = crate::csr_provider::bind_csr_runtime_sources(plan)?;
        }
        Ok(plan)
    }

    /// Helper to create all plans (graph logical, DataFusion logical, physical)
    async fn create_plans(
        &self,
        catalog: std::sync::Arc<dyn lance_graph_catalog::GraphSourceCatalog>,
        ctx: &datafusion::execution::context::SessionContext,
    ) -> Result<(
        crate::logical_plan::LogicalOperator,
        datafusion::logical_expr::LogicalPlan,
        std::sync::Arc<dyn datafusion::physical_plan::ExecutionPlan>,
    )> {
        // Phases 1-3: Create logical plans
        let (logical_plan, df_logical_plan) = self.create_logical_plans(catalog)?;

        // Phase 4: DataFusion Physical Plan
        let df = ctx
            .execute_logical_plan(df_logical_plan.clone())
            .await
            .map_err(|e| GraphError::ExecutionError {
                message: format!("Failed to execute DataFusion plan: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        let physical_plan =
            df.create_physical_plan()
                .await
                .map_err(|e| GraphError::ExecutionError {
                    message: format!("Failed to create physical plan: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
        let physical_plan =
            self.bind_runtime_sources(physical_plan)
                .map_err(|e| GraphError::ExecutionError {
                    message: format!("Failed to bind runtime source producers: {e}"),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

        Ok((logical_plan, df_logical_plan, physical_plan))
    }

    /// Format explain output as a table
    fn format_explain_output(
        &self,
        logical_plan: &crate::logical_plan::LogicalOperator,
        df_logical_plan: &datafusion::logical_expr::LogicalPlan,
        physical_plan: &dyn datafusion::physical_plan::ExecutionPlan,
    ) -> Result<String> {
        // Format output with query first, then table
        let mut output = String::new();

        // Show Cypher query before the table
        output.push_str("Cypher Query:\n");
        output.push_str(&format!("  {}\n\n", self.query_text));

        // Build table rows (without the query)
        let mut rows = vec![];

        // Row 1: Graph Logical Plan
        let graph_plan_str = format!("{:#?}", logical_plan);
        rows.push(("graph_logical_plan", graph_plan_str));

        // Row 2: DataFusion Logical Plan
        let df_logical_str = format!("{}", df_logical_plan.display_indent());
        rows.push(("logical_plan", df_logical_str));

        // Row 3: DataFusion Physical Plan
        let df_physical_str = format!(
            "{}",
            datafusion::physical_plan::displayable(physical_plan).indent(true)
        );
        rows.push(("physical_plan", df_physical_str));

        // Calculate column widths
        let plan_type_width = rows.iter().map(|(t, _)| t.len()).max().unwrap_or(10);
        let plan_width = rows
            .iter()
            .map(|(_, p)| p.lines().map(|l| l.len()).max().unwrap_or(0))
            .max()
            .unwrap_or(50);

        // Build table
        let separator = format!(
            "+{}+{}+",
            "-".repeat(plan_type_width + 2),
            "-".repeat(plan_width + 2)
        );

        output.push_str(&separator);
        output.push('\n');

        // Header
        output.push_str(&format!(
            "| {:<width$} | {:<plan_width$} |\n",
            "plan_type",
            "plan",
            width = plan_type_width,
            plan_width = plan_width
        ));
        output.push_str(&separator);
        output.push('\n');

        // Data rows
        for (plan_type, plan_content) in rows {
            let lines: Vec<&str> = plan_content.lines().collect();
            if lines.is_empty() {
                output.push_str(&format!(
                    "| {:<width$} | {:<plan_width$} |\n",
                    plan_type,
                    "",
                    width = plan_type_width,
                    plan_width = plan_width
                ));
            } else {
                // First line with plan_type
                output.push_str(&format!(
                    "| {:<width$} | {:<plan_width$} |\n",
                    plan_type,
                    lines[0],
                    width = plan_type_width,
                    plan_width = plan_width
                ));

                // Remaining lines with empty plan_type
                for line in &lines[1..] {
                    output.push_str(&format!(
                        "| {:<width$} | {:<plan_width$} |\n",
                        "",
                        line,
                        width = plan_type_width,
                        plan_width = plan_width
                    ));
                }
            }
        }

        output.push_str(&separator);
        output.push('\n');

        Ok(output)
    }

    /// Get all node labels referenced in this query
    pub fn referenced_node_labels(&self) -> Vec<String> {
        let mut labels = Vec::new();

        for clause in &self.ast.reading_clauses {
            if let ReadingClause::Match(match_clause) = clause {
                for pattern in &match_clause.patterns {
                    self.collect_node_labels_from_pattern(pattern, &mut labels);
                }
            }
        }

        labels.sort();
        labels.dedup();
        labels
    }

    /// Get all relationship types referenced in this query
    pub fn referenced_relationship_types(&self) -> Vec<String> {
        let mut types = Vec::new();

        for clause in &self.ast.reading_clauses {
            if let ReadingClause::Match(match_clause) = clause {
                for pattern in &match_clause.patterns {
                    self.collect_relationship_types_from_pattern(pattern, &mut types);
                }
            }
        }

        types.sort();
        types.dedup();
        types
    }

    /// Get all variables used in this query
    pub fn variables(&self) -> Vec<String> {
        let mut variables = Vec::new();

        for clause in &self.ast.reading_clauses {
            match clause {
                ReadingClause::Match(match_clause) => {
                    for pattern in &match_clause.patterns {
                        self.collect_variables_from_pattern(pattern, &mut variables);
                    }
                }
                ReadingClause::Unwind(unwind_clause) => {
                    variables.push(unwind_clause.alias.clone());
                }
            }
        }

        variables.sort();
        variables.dedup();
        variables
    }

    // Collection helper methods

    fn collect_node_labels_from_pattern(
        &self,
        pattern: &crate::ast::GraphPattern,
        labels: &mut Vec<String>,
    ) {
        match pattern {
            crate::ast::GraphPattern::Node(node) => {
                labels.extend(node.labels.clone());
            }
            crate::ast::GraphPattern::Path(path) => {
                labels.extend(path.start_node.labels.clone());
                for segment in &path.segments {
                    labels.extend(segment.end_node.labels.clone());
                }
            }
        }
    }

    fn collect_relationship_types_from_pattern(
        &self,
        pattern: &crate::ast::GraphPattern,
        types: &mut Vec<String>,
    ) {
        if let crate::ast::GraphPattern::Path(path) = pattern {
            for segment in &path.segments {
                types.extend(segment.relationship.types.clone());
            }
        }
    }

    fn collect_variables_from_pattern(
        &self,
        pattern: &crate::ast::GraphPattern,
        variables: &mut Vec<String>,
    ) {
        match pattern {
            crate::ast::GraphPattern::Node(node) => {
                if let Some(var) = &node.variable {
                    variables.push(var.clone());
                }
            }
            crate::ast::GraphPattern::Path(path) => {
                if let Some(var) = &path.start_node.variable {
                    variables.push(var.clone());
                }
                for segment in &path.segments {
                    if let Some(var) = &segment.relationship.variable {
                        variables.push(var.clone());
                    }
                    if let Some(var) = &segment.end_node.variable {
                        variables.push(var.clone());
                    }
                }
            }
        }
    }
}

impl CypherQuery {
    /// Execute Cypher query, then apply vector search reranking on results
    ///
    /// This is a convenience method for the common GraphRAG pattern:
    /// 1. Run Cypher query to get candidate entities via graph traversal
    /// 2. Rerank candidates by vector similarity
    ///
    /// # Arguments
    /// * `datasets` - HashMap of table name to RecordBatch (nodes and relationships)
    /// * `vector_search` - VectorSearch configuration for reranking
    ///
    /// # Returns
    /// A RecordBatch with the top-k results sorted by vector similarity
    ///
    /// # Example
    /// ```ignore
    /// use lance_graph::{CypherQuery, VectorSearch};
    /// use lance_graph::ast::DistanceMetric;
    ///
    /// let results = query
    ///     .execute_with_vector_rerank(
    ///         datasets,
    ///         VectorSearch::new("embedding")
    ///             .query_vector(query_vec)
    ///             .metric(DistanceMetric::Cosine)
    ///             .top_k(10)
    ///     )
    ///     .await?;
    /// ```
    pub async fn execute_with_vector_rerank(
        &self,
        datasets: HashMap<String, arrow::record_batch::RecordBatch>,
        vector_search: crate::lance_vector_search::VectorSearch,
    ) -> Result<arrow::record_batch::RecordBatch> {
        // Step 1: Execute Cypher query to get candidates
        let candidates = self.execute(datasets, None).await?;

        // Step 2: Apply vector search reranking
        vector_search.search(&candidates).await
    }
}

/// Builder for constructing Cypher queries programmatically
#[derive(Debug, Default)]
pub struct CypherQueryBuilder {
    match_clauses: Vec<crate::ast::MatchClause>,
    where_expression: Option<crate::ast::BooleanExpression>,
    return_items: Vec<crate::ast::ReturnItem>,
    order_by_items: Vec<crate::ast::OrderByItem>,
    limit: Option<u64>,
    distinct: bool,
    skip: Option<u64>,
    config: Option<GraphConfig>,
    parameters: HashMap<String, serde_json::Value>,
}

impl CypherQueryBuilder {
    /// Create a new query builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a MATCH clause for a node pattern
    pub fn match_node(mut self, variable: &str, label: &str) -> Self {
        let node = crate::ast::NodePattern {
            variable: Some(variable.to_string()),
            labels: vec![label.to_string()],
            properties: HashMap::new(),
        };

        let match_clause = crate::ast::MatchClause {
            patterns: vec![crate::ast::GraphPattern::Node(node)],
        };

        self.match_clauses.push(match_clause);
        self
    }

    /// Set the graph configuration
    pub fn with_config(mut self, config: GraphConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Add a RETURN item
    pub fn return_property(mut self, variable: &str, property: &str) -> Self {
        let prop_ref = crate::ast::PropertyRef::new(variable, property);
        let return_item = crate::ast::ReturnItem {
            expression: crate::ast::ValueExpression::Property(prop_ref),
            alias: None,
        };

        self.return_items.push(return_item);
        self
    }

    /// Set DISTINCT flag
    pub fn distinct(mut self, distinct: bool) -> Self {
        self.distinct = distinct;
        self
    }

    /// Add a LIMIT clause
    pub fn limit(mut self, limit: u64) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Add a SKIP clause
    pub fn skip(mut self, skip: u64) -> Self {
        self.skip = Some(skip);
        self
    }

    /// Build the final CypherQuery
    pub fn build(self) -> Result<CypherQuery> {
        if self.match_clauses.is_empty() {
            return Err(GraphError::PlanError {
                message: "Query must have at least one MATCH clause".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            });
        }

        if self.return_items.is_empty() {
            return Err(GraphError::PlanError {
                message: "Query must have at least one RETURN item".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            });
        }

        let ast = crate::ast::CypherQuery {
            reading_clauses: self
                .match_clauses
                .into_iter()
                .map(crate::ast::ReadingClause::Match)
                .collect(),
            where_clause: self
                .where_expression
                .map(|expr| crate::ast::WhereClause { expression: expr }),
            with_clause: None, // WITH not supported via builder yet
            post_with_reading_clauses: vec![],
            post_with_where_clause: None,
            return_clause: crate::ast::ReturnClause {
                distinct: self.distinct,
                items: self.return_items,
            },
            order_by: if self.order_by_items.is_empty() {
                None
            } else {
                Some(crate::ast::OrderByClause {
                    items: self.order_by_items,
                })
            },
            limit: self.limit,
            skip: self.skip,
        };

        // Generate query text from AST (simplified)
        let query_text = "MATCH ... RETURN ...".to_string(); // TODO: Implement AST->text conversion

        let query = CypherQuery {
            query_text,
            ast,
            config: self.config,
            parameters: self.parameters,
            runtime_id_filter_metrics: None,
            csr_adjacency_enabled: true,
            csr_native_multihop_enabled: true,
            csr_adjacency_cache: Arc::new(crate::CsrAdjacencyCache::default()),
            csr_adjacency_metrics: Arc::new(crate::CsrScanMetrics::default()),
            deterministic_order: false,
        };

        Ok(query)
    }
}

/// Sort by the result rows: extend the first ORDER BY below LIMIT/OFFSET and
/// projections with every orderable column of its input as tie-breakers, or
/// add one on the output columns. Equal rows on all keys are interchangeable
/// (projections are functions of their input), so the order is determined.
fn deterministic_order(
    plan: datafusion::logical_expr::LogicalPlan,
) -> datafusion::common::Result<datafusion::logical_expr::LogicalPlan> {
    use datafusion::logical_expr::{Expr, Limit, LogicalPlan, Projection, Sort, SortExpr};
    fn has_sort(plan: &LogicalPlan) -> bool {
        match plan {
            LogicalPlan::Sort(_) => true,
            LogicalPlan::Projection(projection) => has_sort(&projection.input),
            LogicalPlan::SubqueryAlias(alias) => has_sort(&alias.input),
            _ => false,
        }
    }
    fn keys(plan: &LogicalPlan, existing: &[SortExpr]) -> Vec<SortExpr> {
        plan.schema()
            .iter()
            .filter(|(_, field)| {
                let datatype = field.data_type();
                datatype.is_primitive()
                    || matches!(
                        datatype,
                        arrow_schema::DataType::Boolean
                            | arrow_schema::DataType::Utf8
                            | arrow_schema::DataType::LargeUtf8
                            | arrow_schema::DataType::Utf8View
                            | arrow_schema::DataType::Binary
                            | arrow_schema::DataType::LargeBinary
                            | arrow_schema::DataType::BinaryView
                            | arrow_schema::DataType::Dictionary(_, _)
                    )
            })
            .map(|(qualifier, field)| {
                Expr::Column(datafusion::common::Column::new(
                    qualifier.cloned(),
                    field.name(),
                ))
            })
            .filter(|column| !existing.iter().any(|sort| &sort.expr == column))
            .map(|column| column.sort(true, true))
            .collect()
    }
    match plan {
        LogicalPlan::Limit(limit) => Ok(LogicalPlan::Limit(Limit {
            input: Arc::new(deterministic_order(Arc::unwrap_or_clone(limit.input))?),
            ..limit
        })),
        LogicalPlan::Sort(sort) => {
            let mut expr = sort.expr.clone();
            expr.extend(keys(&sort.input, &sort.expr));
            Ok(LogicalPlan::Sort(Sort { expr, ..sort }))
        }
        LogicalPlan::Projection(projection) if has_sort(&projection.input) => {
            let input = deterministic_order(Arc::unwrap_or_clone(projection.input))?;
            Ok(LogicalPlan::Projection(Projection::try_new(
                projection.expr,
                Arc::new(input),
            )?))
        }
        other => {
            let expr = keys(&other, &[]);
            if expr.is_empty() {
                return Ok(other);
            }
            Ok(LogicalPlan::Sort(Sort {
                expr,
                input: Arc::new(other),
                fetch: None,
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GraphConfig;

    #[test]
    fn test_default_executor_adjacency_options() {
        let query = CypherQuery::new("MATCH (n:Person) RETURN n.name").unwrap();
        assert!(query.csr_adjacency_enabled);
        let cloned = query.clone();
        assert!(Arc::ptr_eq(
            &query.csr_adjacency_cache,
            &cloned.csr_adjacency_cache
        ));
        assert!(Arc::ptr_eq(
            &query.csr_adjacency_metrics(),
            &cloned.csr_adjacency_metrics()
        ));
        let cache = Arc::new(crate::CsrAdjacencyCache::default());
        let query = query
            .with_csr_adjacency(false)
            .with_shared_csr_cache(cache.clone());
        assert!(!query.csr_adjacency_enabled);
        assert!(Arc::ptr_eq(&query.csr_adjacency_cache, &cache));
    }

    #[test]
    fn test_parse_simple_cypher_query() {
        let query = CypherQuery::new("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(query.query_text(), "MATCH (n:Person) RETURN n.name");
        assert_eq!(query.referenced_node_labels(), vec!["Person"]);
        assert_eq!(query.variables(), vec!["n"]);
    }

    #[test]
    fn test_query_with_parameters() {
        let mut params = HashMap::new();
        params.insert("minAge".to_string(), serde_json::Value::Number(30.into()));
        params.insert("maxAge".to_string(), serde_json::Value::Number(50.into()));

        let query = CypherQuery::new(
            "MATCH (n:Person) WHERE n.age > $minAge AND n.age < $maxAge RETURN n.name",
        )
        .unwrap()
        .with_parameters(params);

        assert!(query.parameters().contains_key("minage"));
        assert!(query.parameters().contains_key("maxage"));
    }

    #[test]
    fn test_query_builder() {
        let config = GraphConfig::builder()
            .with_node_label("Person", "person_id")
            .build()
            .unwrap();

        let query = CypherQueryBuilder::new()
            .with_config(config)
            .match_node("n", "Person")
            .return_property("n", "name")
            .limit(10)
            .build()
            .unwrap();

        assert_eq!(query.referenced_node_labels(), vec!["Person"]);
        assert_eq!(query.variables(), vec!["n"]);
    }

    #[test]
    fn test_relationship_query_parsing() {
        let query =
            CypherQuery::new("MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a.name, b.name")
                .unwrap();
        assert_eq!(query.referenced_node_labels(), vec!["Person"]);
        assert_eq!(query.referenced_relationship_types(), vec!["KNOWS"]);
        assert_eq!(query.variables(), vec!["a", "b", "r"]);
    }

    #[tokio::test]
    async fn test_execute_basic_projection_and_filter() {
        use arrow_array::{Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        // Build a simple batch: name (Utf8), age (Int64)
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["Alice", "Bob", "Carol", "David"])),
                Arc::new(Int64Array::from(vec![28, 34, 29, 42])),
            ],
        )
        .unwrap();

        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();

        let q = CypherQuery::new("MATCH (p:Person) WHERE p.age > 30 RETURN p.name, p.age")
            .unwrap()
            .with_config(cfg);

        let mut data = HashMap::new();
        data.insert("Person".to_string(), batch);

        let out = q.execute(data, None).await.unwrap();
        assert_eq!(out.num_rows(), 2);
        let names = out
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let ages = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        // Expect Bob (34) and David (42)
        let result: Vec<(String, i64)> = (0..out.num_rows())
            .map(|i| (names.value(i).to_string(), ages.value(i)))
            .collect();
        assert!(result.contains(&("Bob".to_string(), 34)));
        assert!(result.contains(&("David".to_string(), 42)));
    }

    #[tokio::test]
    async fn test_execute_single_hop_path_join_projection() {
        use arrow_array::{Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        // People table: id, name, age
        let person_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int64, true),
        ]));
        let people = RecordBatch::try_new(
            person_schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["Alice", "Bob", "Carol"])),
                Arc::new(Int64Array::from(vec![28, 34, 29])),
            ],
        )
        .unwrap();

        // KNOWS relationship: src_person_id -> dst_person_id
        let rel_schema = Arc::new(Schema::new(vec![
            Field::new("src_person_id", DataType::Int64, false),
            Field::new("dst_person_id", DataType::Int64, false),
        ]));
        let knows = RecordBatch::try_new(
            rel_schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2])), // Alice -> Bob, Bob -> Carol
                Arc::new(Int64Array::from(vec![2, 3])),
            ],
        )
        .unwrap();

        // Config: Person(id) and KNOWS(src_person_id -> dst_person_id)
        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_person_id", "dst_person_id")
            .build()
            .unwrap();

        // Query: MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN b.name
        let q = CypherQuery::new("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN b.name")
            .unwrap()
            .with_config(cfg);

        let mut data = HashMap::new();
        // Register tables using labels / rel types as names
        data.insert("Person".to_string(), people);
        data.insert("KNOWS".to_string(), knows);

        let out = q.execute(data, None).await.unwrap();
        // Expect two rows: Bob, Carol (the targets)
        let names = out
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let got: Vec<String> = (0..out.num_rows())
            .map(|i| names.value(i).to_string())
            .collect();
        assert_eq!(got.len(), 2);
        assert!(got.contains(&"Bob".to_string()));
        assert!(got.contains(&"Carol".to_string()));
    }

    #[tokio::test]
    async fn test_execute_order_by_asc() {
        use arrow_array::{Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        // name, age (int)
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["Bob", "Alice", "David", "Carol"])),
                Arc::new(Int64Array::from(vec![34, 28, 42, 29])),
            ],
        )
        .unwrap();

        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();

        // Order ascending by age
        let q = CypherQuery::new("MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age ASC")
            .unwrap()
            .with_config(cfg);

        let mut data = HashMap::new();
        data.insert("Person".to_string(), batch);

        let out = q.execute(data, None).await.unwrap();
        let ages = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let collected: Vec<i64> = (0..out.num_rows()).map(|i| ages.value(i)).collect();
        assert_eq!(collected, vec![28, 29, 34, 42]);
    }

    #[tokio::test]
    async fn deterministic_order_breaks_ties_and_tiles_pages_in_memory() {
        use arrow_array::{Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("name", DataType::Utf8, true),
                Field::new("age", DataType::Int64, true),
            ])),
            vec![
                Arc::new(StringArray::from(vec![
                    Some("d"),
                    Some("b"),
                    None,
                    Some("a"),
                    Some("c"),
                    Some("b"),
                ])),
                Arc::new(Int64Array::from(vec![30, 20, 30, 30, 20, 30])),
            ],
        )
        .unwrap();
        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();
        let run = |text: &str| {
            let q = CypherQuery::new(text)
                .unwrap()
                .with_config(cfg.clone())
                .with_deterministic_order(true);
            let data = HashMap::from([("Person".to_string(), batch.clone())]);
            async move {
                let out = q.execute(data, None).await.unwrap();
                (0..out.num_rows())
                    .map(|row| {
                        out.columns()
                            .iter()
                            .map(|array| {
                                arrow::util::display::array_value_to_string(array, row).unwrap()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            }
        };
        // Ties on age are broken by name (NULLs first); age stays primary.
        let full = run("MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age DESC").await;
        let expected = [
            ["", "30"],
            ["a", "30"],
            ["b", "30"],
            ["d", "30"],
            ["b", "20"],
            ["c", "20"],
        ];
        assert_eq!(full, expected.map(|row| row.map(String::from).to_vec()));
        // Without ORDER BY every column is a key; pages tile the full result.
        let all = run("MATCH (p:Person) RETURN p.age, p.name").await;
        let mut pages = Vec::new();
        for skip in [0, 2, 4] {
            pages.extend(
                run(&format!(
                    "MATCH (p:Person) RETURN p.age, p.name SKIP {skip} LIMIT 2"
                ))
                .await,
            );
        }
        assert_eq!(pages, all);
        assert_eq!(all[0], vec!["20".to_string(), "b".to_string()]);
    }

    #[tokio::test]
    async fn test_execute_order_by_desc_with_skip_limit() {
        use arrow_array::{Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["Bob", "Alice", "David", "Carol"])),
                Arc::new(Int64Array::from(vec![34, 28, 42, 29])),
            ],
        )
        .unwrap();

        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();

        // Desc by age, skip 1 (drop 42), take 2 -> [34, 29]
        let q =
            CypherQuery::new("MATCH (p:Person) RETURN p.age ORDER BY p.age DESC SKIP 1 LIMIT 2")
                .unwrap()
                .with_config(cfg);

        let mut data = HashMap::new();
        data.insert("Person".to_string(), batch);

        let out = q.execute(data, None).await.unwrap();
        assert_eq!(out.num_rows(), 2);
        let ages = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let collected: Vec<i64> = (0..out.num_rows()).map(|i| ages.value(i)).collect();
        assert_eq!(collected, vec![34, 29]);
    }

    #[tokio::test]
    async fn test_execute_skip_without_limit() {
        use arrow_array::{Int64Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![Field::new("age", DataType::Int64, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![10, 20, 30, 40]))],
        )
        .unwrap();

        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();

        let q = CypherQuery::new("MATCH (p:Person) RETURN p.age ORDER BY p.age ASC SKIP 2")
            .unwrap()
            .with_config(cfg);

        let mut data = HashMap::new();
        data.insert("Person".to_string(), batch);

        let out = q.execute(data, None).await.unwrap();
        assert_eq!(out.num_rows(), 2);
        let ages = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let collected: Vec<i64> = (0..out.num_rows()).map(|i| ages.value(i)).collect();
        assert_eq!(collected, vec![30, 40]);
    }

    #[tokio::test]
    async fn test_execute_datafusion_simple_scan() {
        use arrow_array::{Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        // Create test data
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec!["Alice", "Bob"])),
            ],
        )
        .unwrap();

        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();

        // Test simple scan without filters
        let query = CypherQuery::new("MATCH (p:Person) RETURN p.name")
            .unwrap()
            .with_config(cfg);

        let mut datasets = HashMap::new();
        datasets.insert("Person".to_string(), batch);

        // Execute using DataFusion pipeline
        let result = query.execute_datafusion(datasets).await.unwrap();

        // Should return all rows
        assert_eq!(
            result.num_rows(),
            2,
            "Should return all 2 rows without filtering"
        );
        assert_eq!(result.num_columns(), 1, "Should return 1 column (name)");

        // Verify data
        let names = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let name_set: std::collections::HashSet<String> = (0..result.num_rows())
            .map(|i| names.value(i).to_string())
            .collect();
        let expected: std::collections::HashSet<String> =
            ["Alice", "Bob"].iter().map(|s| s.to_string()).collect();
        assert_eq!(name_set, expected, "Should return Alice and Bob");
    }

    #[tokio::test]
    async fn test_execute_with_context_simple_scan() {
        use arrow_array::{Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use datafusion::datasource::MemTable;
        use datafusion::execution::context::SessionContext;
        use std::sync::Arc;

        // Create test data
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["Alice", "Bob", "Carol"])),
                Arc::new(Int64Array::from(vec![28, 34, 29])),
            ],
        )
        .unwrap();

        // Create SessionContext and register data source
        let mem_table =
            Arc::new(MemTable::try_new(schema.clone(), vec![vec![batch.clone()]]).unwrap());
        let ctx = SessionContext::new();
        ctx.register_table("Person", mem_table).unwrap();

        // Create query
        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();

        let query = CypherQuery::new("MATCH (p:Person) RETURN p.name")
            .unwrap()
            .with_config(cfg);

        // Execute with context (catalog built automatically)
        let result = query.execute_with_context(ctx).await.unwrap();

        // Verify results
        assert_eq!(result.num_rows(), 3);
        assert_eq!(result.num_columns(), 1);

        let names = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(names.value(0), "Alice");
        assert_eq!(names.value(1), "Bob");
        assert_eq!(names.value(2), "Carol");
    }

    #[tokio::test]
    async fn test_execute_with_context_with_filter() {
        use arrow_array::{Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use datafusion::datasource::MemTable;
        use datafusion::execution::context::SessionContext;
        use std::sync::Arc;

        // Create test data
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
                Arc::new(StringArray::from(vec!["Alice", "Bob", "Carol", "David"])),
                Arc::new(Int64Array::from(vec![28, 34, 29, 42])),
            ],
        )
        .unwrap();

        // Create SessionContext
        let mem_table =
            Arc::new(MemTable::try_new(schema.clone(), vec![vec![batch.clone()]]).unwrap());
        let ctx = SessionContext::new();
        ctx.register_table("Person", mem_table).unwrap();

        // Create query with filter
        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();

        let query = CypherQuery::new("MATCH (p:Person) WHERE p.age > 30 RETURN p.name, p.age")
            .unwrap()
            .with_config(cfg);

        // Execute with context
        let result = query.execute_with_context(ctx).await.unwrap();

        // Verify: should return Bob (34) and David (42)
        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.num_columns(), 2);

        let names = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let ages = result
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        let results: Vec<(String, i64)> = (0..result.num_rows())
            .map(|i| (names.value(i).to_string(), ages.value(i)))
            .collect();

        assert!(results.contains(&("Bob".to_string(), 34)));
        assert!(results.contains(&("David".to_string(), 42)));
    }

    #[tokio::test]
    async fn test_execute_with_context_relationship_traversal() {
        use arrow_array::{Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use datafusion::datasource::MemTable;
        use datafusion::execution::context::SessionContext;
        use std::sync::Arc;

        // Create Person nodes
        let person_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let person_batch = RecordBatch::try_new(
            person_schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["Alice", "Bob", "Carol"])),
            ],
        )
        .unwrap();

        // Create KNOWS relationships
        let knows_schema = Arc::new(Schema::new(vec![
            Field::new("src_id", DataType::Int64, false),
            Field::new("dst_id", DataType::Int64, false),
            Field::new("since", DataType::Int64, false),
        ]));
        let knows_batch = RecordBatch::try_new(
            knows_schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2])),
                Arc::new(Int64Array::from(vec![2, 3])),
                Arc::new(Int64Array::from(vec![2020, 2021])),
            ],
        )
        .unwrap();

        // Create SessionContext and register tables
        let person_table = Arc::new(
            MemTable::try_new(person_schema.clone(), vec![vec![person_batch.clone()]]).unwrap(),
        );
        let knows_table = Arc::new(
            MemTable::try_new(knows_schema.clone(), vec![vec![knows_batch.clone()]]).unwrap(),
        );

        let ctx = SessionContext::new();
        ctx.register_table("Person", person_table).unwrap();
        ctx.register_table("KNOWS", knows_table).unwrap();

        // Create query
        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_id", "dst_id")
            .build()
            .unwrap();

        let query = CypherQuery::new("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name")
            .unwrap()
            .with_config(cfg);

        // Execute with context
        let result = query.execute_with_context(ctx).await.unwrap();

        // Verify: should return 2 relationships (Alice->Bob, Bob->Carol)
        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.num_columns(), 2);

        let src_names = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let dst_names = result
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        let relationships: Vec<(String, String)> = (0..result.num_rows())
            .map(|i| {
                (
                    src_names.value(i).to_string(),
                    dst_names.value(i).to_string(),
                )
            })
            .collect();

        assert!(relationships.contains(&("Alice".to_string(), "Bob".to_string())));
        assert!(relationships.contains(&("Bob".to_string(), "Carol".to_string())));
    }

    #[tokio::test]
    async fn test_execute_with_context_order_by_limit() {
        use arrow_array::{Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use datafusion::datasource::MemTable;
        use datafusion::execution::context::SessionContext;
        use std::sync::Arc;

        // Create test data
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("score", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
                Arc::new(StringArray::from(vec!["Alice", "Bob", "Carol", "David"])),
                Arc::new(Int64Array::from(vec![85, 92, 78, 95])),
            ],
        )
        .unwrap();

        // Create SessionContext
        let mem_table =
            Arc::new(MemTable::try_new(schema.clone(), vec![vec![batch.clone()]]).unwrap());
        let ctx = SessionContext::new();
        ctx.register_table("Student", mem_table).unwrap();

        // Create query with ORDER BY and LIMIT
        let cfg = GraphConfig::builder()
            .with_node_label("Student", "id")
            .build()
            .unwrap();

        let query = CypherQuery::new(
            "MATCH (s:Student) RETURN s.name, s.score ORDER BY s.score DESC LIMIT 2",
        )
        .unwrap()
        .with_config(cfg);

        // Execute with context
        let result = query.execute_with_context(ctx).await.unwrap();

        // Verify: should return top 2 scores (David: 95, Bob: 92)
        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.num_columns(), 2);

        let names = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let scores = result
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        // First row should be David (95)
        assert_eq!(names.value(0), "David");
        assert_eq!(scores.value(0), 95);

        // Second row should be Bob (92)
        assert_eq!(names.value(1), "Bob");
        assert_eq!(scores.value(1), 92);
    }

    #[tokio::test]
    async fn test_to_sql() {
        use arrow_array::RecordBatch;
        use arrow_schema::{DataType, Field, Schema};
        use std::collections::HashMap;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::new_empty(schema.clone());

        let mut datasets = HashMap::new();
        datasets.insert("Person".to_string(), batch);

        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();

        let query = CypherQuery::new("MATCH (p:Person) RETURN p.name")
            .unwrap()
            .with_config(cfg);

        let sql = query.to_sql(datasets, None).await.unwrap();
        println!("Generated SQL: {}", sql);

        assert!(sql.contains("SELECT"));
        assert!(sql.to_lowercase().contains("from person"));
        // Note: DataFusion unparser might quote identifiers or use aliases
        // We check for "p.name" which is the expected output alias
        assert!(sql.contains("p.name"));
    }

    async fn write_lance_dataset(path: &std::path::Path, batch: arrow_array::RecordBatch) {
        use arrow_array::{RecordBatch, RecordBatchIterator};
        use lance::dataset::{Dataset, WriteParams};

        let schema = batch.schema();
        let batches: Vec<std::result::Result<RecordBatch, arrow::error::ArrowError>> =
            vec![std::result::Result::Ok(batch)];
        let reader = RecordBatchIterator::new(batches.into_iter(), schema);

        Dataset::write(reader, path.to_str().unwrap(), None::<WriteParams>)
            .await
            .expect("write lance dataset");
    }

    fn build_people_batch() -> arrow_array::RecordBatch {
        use arrow_array::{ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("person_id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Int32, false),
        ]));

        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])) as ArrayRef,
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Carol", "David"])) as ArrayRef,
            Arc::new(Int32Array::from(vec![28, 34, 29, 42])) as ArrayRef,
        ];

        RecordBatch::try_new(schema, columns).expect("valid person batch")
    }

    fn build_friendship_batch() -> arrow_array::RecordBatch {
        use arrow_array::{ArrayRef, Int64Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("person1_id", DataType::Int64, false),
            Field::new("person2_id", DataType::Int64, false),
        ]));

        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(vec![1, 1, 2, 3])) as ArrayRef,
            Arc::new(Int64Array::from(vec![2, 3, 4, 4])) as ArrayRef,
        ];

        RecordBatch::try_new(schema, columns).expect("valid friendship batch")
    }

    #[tokio::test]
    async fn executes_against_directory_namespace() {
        use arrow_array::StringArray;
        use tempfile::tempdir;

        let tmp_dir = tempdir().unwrap();
        write_lance_dataset(&tmp_dir.path().join("Person.lance"), build_people_batch()).await;
        write_lance_dataset(
            &tmp_dir.path().join("FRIEND_OF.lance"),
            build_friendship_batch(),
        )
        .await;

        let config = GraphConfig::builder()
            .with_node_label("Person", "person_id")
            .with_relationship("FRIEND_OF", "person1_id", "person2_id")
            .build()
            .expect("valid graph config");

        let query = CypherQuery::new("MATCH (p:Person) WHERE p.age > 30 RETURN p.name")
            .expect("query parses")
            .with_config(config);

        let namespace = DirNamespace::new(tmp_dir.path().to_string_lossy().into_owned());

        let result = query
            .execute_with_namespace(namespace.clone(), None)
            .await
            .expect("namespace execution succeeds");

        use arrow_array::Array;
        let names = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("string column");

        let mut values: Vec<String> = (0..names.len())
            .map(|i| names.value(i).to_string())
            .collect();
        values.sort();
        assert_eq!(values, vec!["Bob".to_string(), "David".to_string()]);
    }

    #[tokio::test]
    async fn test_execute_fails_on_semantic_error() {
        use arrow_array::RecordBatch;
        use arrow_schema::{DataType, Field, Schema};
        use std::collections::HashMap;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let batch = RecordBatch::new_empty(schema);
        let mut datasets = HashMap::new();
        datasets.insert("Person".to_string(), batch);

        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();

        // Query referencing undefined variable 'x'
        let query = CypherQuery::new("MATCH (n:Person) RETURN x.name")
            .unwrap()
            .with_config(cfg);

        let result = query.execute(datasets, None).await;

        assert!(result.is_err());
        match result {
            Err(GraphError::PlanError { message, .. }) => {
                assert!(message.contains("Semantic analysis failed"));
                assert!(message.contains("Undefined variable: 'x'"));
            }
            _ => panic!(
                "Expected PlanError with semantic failure message, got {:?}",
                result
            ),
        }
    }
}
