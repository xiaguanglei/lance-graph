// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Consistent, version-pinned reads of a multi-table graph.
//!
//! A graph spans several independently committed Lance tables (node labels and
//! relationship types). A service with many nodes needs every query to read
//! one fixed version of each table, and needs to tell clients which versions
//! a write produced so another node can serve read-your-writes:
//!
//! * [`GraphVersions`] is that version vector; it serializes to a token.
//! * [`GraphSnapshot`] pins every table to one version and executes queries
//!   with the node's shared [`CsrAdjacencyCache`].
//! * [`GraphSnapshotProvider`] is the per-node entry point: bounded-staleness
//!   `latest`, read-your-writes `at_least`, exact `at`, and `warm`.
//!
//! Snapshots never change once created; every node that opens the same
//! versions reads identical data. Cross-table atomicity of writes is the
//! writer's protocol (see `docs/multi_node_deployment.md`).

use crate::{CsrAdjacencyCache, GraphConfig, GraphError, Result};
use datafusion::{
    datasource::DefaultTableSource, execution::context::SessionContext, prelude::SessionConfig,
};
use lance::{datafusion::LanceTableProvider, dataset::builder::DatasetBuilder, Dataset};
use lance_graph_catalog::InMemoryCatalog;
use lance_namespace::{models::DescribeTableRequest, LanceNamespace};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

fn error(message: impl Into<String>) -> GraphError {
    GraphError::ConfigError {
        message: message.into(),
        location: snafu::Location::new(file!(), line!(), column!()),
    }
}

/// Committed version of every table a graph reads, keyed by table name.
///
/// Return it to clients after a write and pass it back on later reads (to any
/// node) so they observe at least that write: see
/// [`GraphSnapshotProvider::at_least`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphVersions {
    tables: BTreeMap<String, u64>,
}

impl GraphVersions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or replace one table's version.
    pub fn with(mut self, table: impl Into<String>, version: u64) -> Self {
        self.tables.insert(table.into(), version);
        self
    }

    pub fn get(&self, table: &str) -> Option<u64> {
        self.tables.get(table).copied()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, u64)> {
        self.tables
            .iter()
            .map(|(table, version)| (table.as_str(), *version))
    }

    /// True when every table in `required` is present here at the same or a
    /// newer version.
    pub fn covers(&self, required: &GraphVersions) -> bool {
        required
            .tables
            .iter()
            .all(|(table, version)| self.get(table).is_some_and(|own| own >= *version))
    }

    /// Per-table maximum of both vectors, e.g. to combine several writes.
    pub fn merge_max(&mut self, other: &GraphVersions) {
        for (table, version) in &other.tables {
            let entry = self.tables.entry(table.clone()).or_insert(*version);
            *entry = (*entry).max(*version);
        }
    }

    /// Compact text token (JSON object of table -> version).
    pub fn to_token(&self) -> String {
        serde_json::to_string(&self.tables).expect("versions serialize")
    }

    pub fn from_token(token: &str) -> Result<Self> {
        serde_json::from_str(token)
            .map(|tables| Self { tables })
            .map_err(|e| error(format!("Invalid graph version token: {e}")))
    }
}

/// Every table of a graph pinned to one committed version.
pub struct GraphSnapshot {
    contexts: Arc<Contexts>,
    datasets: BTreeMap<String, Arc<Dataset>>,
    versions: GraphVersions,
    catalog: Arc<InMemoryCatalog>,
    cache: Arc<CsrAdjacencyCache>,
    config: GraphConfig,
    bindings: Bindings,
}

/// Execution sessions reused by every query of a provider: creating one costs
/// about as much as planning a native query.
struct Contexts {
    general: SessionContext,
    native: SessionContext,
}

impl Contexts {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            general: SessionContext::new(),
            native: crate::native_session::native_session(SessionConfig::new()),
        })
    }
}

/// Which table backs each node label and relationship type.
#[derive(Debug, Clone)]
struct Bindings {
    nodes: BTreeMap<String, String>,
    relationships: BTreeMap<String, String>,
}

impl Bindings {
    fn tables(&self) -> BTreeSet<String> {
        self.nodes
            .values()
            .chain(self.relationships.values())
            .cloned()
            .collect()
    }
}

/// Outcome of [`GraphSnapshot::warm`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WarmReport {
    /// Relationship adjacency snapshots now cached.
    pub adjacency: usize,
    /// Node-ID maps now cached.
    pub node_ids: usize,
    /// Entries that are unsupported or do not fit the cache limits; queries
    /// over them plan ordinary JOINs.
    pub skipped: usize,
    /// Node ID columns now in the column cache (when it is enabled).
    pub columns: usize,
}

impl std::fmt::Debug for GraphSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphSnapshot")
            .field("versions", &self.versions)
            .finish()
    }
}

impl GraphSnapshot {
    /// Pin explicitly opened datasets (each already at the wanted version).
    /// `tables` maps table name -> dataset; every label/type of `config` must
    /// be backed by one of them (see [`GraphSnapshotProvider::with_table`] for
    /// the default naming).
    pub fn from_datasets(
        config: GraphConfig,
        tables: BTreeMap<String, Arc<Dataset>>,
        table_names: &HashMap<String, String>,
        cache: Arc<CsrAdjacencyCache>,
    ) -> Result<Self> {
        let bindings = bindings(&config, table_names);
        Self::assemble(config, bindings, tables, cache, Contexts::new())
    }

    fn assemble(
        config: GraphConfig,
        bindings: Bindings,
        datasets: BTreeMap<String, Arc<Dataset>>,
        cache: Arc<CsrAdjacencyCache>,
        contexts: Arc<Contexts>,
    ) -> Result<Self> {
        // One provider per physical table, so labels/types sharing a table also
        // share its dataset identity (required by native multi-hop paths).
        let mut providers = HashMap::new();
        let mut source = |table: &String| -> Result<Arc<DefaultTableSource>> {
            if let Some(source) = providers.get(table) {
                return Ok(Arc::clone(source));
            }
            let dataset = datasets
                .get(table)
                .ok_or_else(|| error(format!("Graph snapshot has no table '{table}'")))?;
            let source = Arc::new(DefaultTableSource::new(Arc::new(LanceTableProvider::new(
                dataset.clone(),
                false,
                false,
            ))));
            providers.insert(table.clone(), Arc::clone(&source));
            Ok(source)
        };
        let mut catalog = InMemoryCatalog::new();
        for (label, table) in &bindings.nodes {
            catalog = catalog.with_node_source(label, source(table)?);
        }
        for (rel_type, table) in &bindings.relationships {
            catalog = catalog.with_relationship_source(rel_type, source(table)?);
        }
        let versions = GraphVersions {
            tables: datasets
                .iter()
                .map(|(table, dataset)| (table.clone(), dataset.version_id()))
                .collect(),
        };
        Ok(Self {
            contexts,
            datasets,
            versions,
            catalog: Arc::new(catalog),
            cache,
            config,
            bindings,
        })
    }

    pub fn versions(&self) -> &GraphVersions {
        &self.versions
    }

    pub fn dataset(&self, table: &str) -> Option<&Arc<Dataset>> {
        self.datasets.get(table)
    }

    pub fn config(&self) -> &GraphConfig {
        &self.config
    }

    pub fn cache(&self) -> &Arc<CsrAdjacencyCache> {
        &self.cache
    }

    pub(crate) fn catalog(&self) -> Arc<InMemoryCatalog> {
        self.catalog.clone()
    }

    /// (general, native-path) execution sessions shared by the provider.
    pub(crate) fn contexts(&self) -> (&SessionContext, &SessionContext) {
        (&self.contexts.general, &self.contexts.native)
    }

    /// Build every relationship adjacency and node-ID map of these versions
    /// (and, with the column cache enabled, load every node ID column), e.g.
    /// before a node reports ready or after a write notification, so the
    /// first query does not pay a cold build. Hot property columns can be
    /// added with [`GraphSnapshot::warm_columns`].
    pub async fn warm(&self) -> WarmReport {
        let mut report = WarmReport::default();
        for (rel_type, table) in &self.bindings.relationships {
            let Some(mapping) = self.config.get_relationship_mapping(rel_type) else {
                continue;
            };
            if self
                .cache
                .warm_adjacency(
                    self.datasets[table].clone(),
                    &mapping.source_id_field,
                    &mapping.target_id_field,
                )
                .await
            {
                report.adjacency += 1;
            } else {
                report.skipped += 1;
            }
        }
        for (label, table) in &self.bindings.nodes {
            let Some(mapping) = self.config.get_node_mapping(label) else {
                continue;
            };
            if self
                .cache
                .warm_node_ids(self.datasets[table].clone(), &mapping.id_field)
                .await
            {
                report.node_ids += 1;
            } else {
                report.skipped += 1;
            }
            if self.cache.options().max_column_cache_bytes > 0 {
                report.columns += self
                    .cache
                    .warm_columns(self.datasets[table].clone(), &[&mapping.id_field])
                    .await;
            }
        }
        report
    }

    /// Load these stored columns of the table behind a node label or
    /// relationship type into the column cache (needs
    /// `max_column_cache_bytes`), e.g. properties most queries return or
    /// filter on. Returns how many are cached; 0 for an unknown label.
    pub async fn warm_columns(&self, label_or_type: &str, columns: &[&str]) -> usize {
        let table = self
            .bindings
            .nodes
            .iter()
            .chain(&self.bindings.relationships)
            .find(|(name, _)| name.eq_ignore_ascii_case(label_or_type))
            .map(|(_, table)| table);
        match table.and_then(|table| self.datasets.get(table)) {
            Some(dataset) => self.cache.warm_columns(dataset.clone(), columns).await,
            None => 0,
        }
    }
}

fn bindings(config: &GraphConfig, table_names: &HashMap<String, String>) -> Bindings {
    let table = |name: &str| {
        table_names
            .get(&name.to_lowercase())
            .cloned()
            .unwrap_or_else(|| name.to_owned())
    };
    Bindings {
        nodes: config
            .node_mappings
            .values()
            .map(|mapping| (mapping.label.clone(), table(&mapping.label)))
            .collect(),
        relationships: config
            .relationship_mappings
            .values()
            .map(|mapping| {
                (
                    mapping.relationship_type.clone(),
                    table(&mapping.relationship_type),
                )
            })
            .collect(),
    }
}

/// Per-node source of graph snapshots resolved through a Lance namespace.
///
/// Create one per service node (and per security context) and share it across
/// requests: it owns the node's [`CsrAdjacencyCache`] and keeps dataset handles
/// so refreshing only reads the newest manifests.
pub struct GraphSnapshotProvider {
    namespace: Arc<dyn LanceNamespace + Send + Sync>,
    config: GraphConfig,
    table_names: HashMap<String, String>,
    cache: Arc<CsrAdjacencyCache>,
    max_staleness: Duration,
    latest: tokio::sync::Mutex<Option<(Instant, Arc<GraphSnapshot>)>>,
    pinned: Mutex<Vec<Arc<GraphSnapshot>>>,
    contexts: Arc<Contexts>,
}

const PINNED_SNAPSHOTS: usize = 8;

impl std::fmt::Debug for GraphSnapshotProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphSnapshotProvider")
            .field("table_names", &self.table_names)
            .field("max_staleness", &self.max_staleness)
            .finish()
    }
}

impl GraphSnapshotProvider {
    /// Tables are named after their label / relationship type unless
    /// overridden with [`Self::with_table`]. The default staleness bound is
    /// one second; use [`Self::at_least`] for read-your-writes.
    pub fn new(namespace: Arc<dyn LanceNamespace + Send + Sync>, config: GraphConfig) -> Self {
        Self {
            namespace,
            config,
            table_names: HashMap::new(),
            cache: Arc::new(CsrAdjacencyCache::default()),
            max_staleness: Duration::from_secs(1),
            latest: tokio::sync::Mutex::new(None),
            pinned: Mutex::new(Vec::new()),
            contexts: Contexts::new(),
        }
    }

    /// Back a label or relationship type by another table, e.g. several
    /// relationship types stored in one edge table.
    pub fn with_table(mut self, label_or_type: &str, table: impl Into<String>) -> Self {
        self.table_names
            .insert(label_or_type.to_lowercase(), table.into());
        self
    }

    /// Use an explicitly sized cache (one per node and security context).
    pub fn with_cache(mut self, cache: Arc<CsrAdjacencyCache>) -> Self {
        self.cache = cache;
        self
    }

    /// How old [`Self::latest`] may be before it rereads the manifests.
    /// `Duration::ZERO` checks for new versions on every call.
    pub fn with_max_staleness(mut self, max_staleness: Duration) -> Self {
        self.max_staleness = max_staleness;
        self
    }

    pub fn cache(&self) -> &Arc<CsrAdjacencyCache> {
        &self.cache
    }

    /// Newest versions seen within the staleness bound.
    pub async fn latest(&self) -> Result<Arc<GraphSnapshot>> {
        let mut latest = self.latest.lock().await;
        if let Some((checked, snapshot)) = latest.as_ref() {
            if checked.elapsed() <= self.max_staleness {
                return Ok(snapshot.clone());
            }
        }
        self.refresh_locked(&mut latest).await
    }

    /// Reread every table's newest manifest now.
    pub async fn refresh(&self) -> Result<Arc<GraphSnapshot>> {
        let mut latest = self.latest.lock().await;
        self.refresh_locked(&mut latest).await
    }

    /// A snapshot that includes at least `required` (read-your-writes and
    /// monotonic reads across nodes). Refreshes when the cached one is older;
    /// errors if storage does not yet show the required versions.
    pub async fn at_least(&self, required: &GraphVersions) -> Result<Arc<GraphSnapshot>> {
        let snapshot = self.latest().await?;
        if snapshot.versions.covers(required) {
            return Ok(snapshot);
        }
        let snapshot = self.refresh().await?;
        if snapshot.versions.covers(required) {
            return Ok(snapshot);
        }
        Err(error(format!(
            "Graph versions {} are not visible yet (latest {})",
            required.to_token(),
            snapshot.versions.to_token()
        )))
    }

    /// Exactly these versions, e.g. a reproducible read or paging through one
    /// result with a token. Recently used pinned snapshots are reused.
    pub async fn at(&self, versions: &GraphVersions) -> Result<Arc<GraphSnapshot>> {
        if let Some(snapshot) = self
            .pinned
            .lock()
            .expect("snapshot lock")
            .iter()
            .find(|snapshot| &snapshot.versions == versions)
        {
            return Ok(snapshot.clone());
        }
        let bindings = bindings(&self.config, &self.table_names);
        let latest = self.latest().await?;
        let mut datasets = BTreeMap::new();
        for table in bindings.tables() {
            let version = versions.get(&table).ok_or_else(|| {
                error(format!("Graph version token has no version for '{table}'"))
            })?;
            let dataset = latest.datasets[&table].as_ref();
            let pinned = if dataset.version_id() == version {
                dataset.clone()
            } else {
                dataset
                    .checkout_version(version)
                    .await
                    .map_err(|e| error(format!("Cannot open '{table}' at v{version}: {e}")))?
            };
            datasets.insert(table, Arc::new(pinned));
        }
        let snapshot = Arc::new(GraphSnapshot::assemble(
            self.config.clone(),
            bindings,
            datasets,
            self.cache.clone(),
            self.contexts.clone(),
        )?);
        let mut pinned = self.pinned.lock().expect("snapshot lock");
        pinned.insert(0, snapshot.clone());
        pinned.truncate(PINNED_SNAPSHOTS);
        Ok(snapshot)
    }

    async fn refresh_locked(
        &self,
        latest: &mut Option<(Instant, Arc<GraphSnapshot>)>,
    ) -> Result<Arc<GraphSnapshot>> {
        let bindings = bindings(&self.config, &self.table_names);
        let previous = latest.as_ref().map(|(_, snapshot)| snapshot.clone());
        let reads = bindings.tables().into_iter().map(|table| {
            let previous = previous
                .as_ref()
                .and_then(|snapshot| snapshot.datasets.get(&table).cloned());
            async move {
                let dataset = match previous {
                    // Keep the handle's session caches; only the newest manifest is read.
                    Some(dataset) => {
                        let mut dataset = dataset.as_ref().clone();
                        dataset
                            .checkout_latest()
                            .await
                            .map_err(|e| error(format!("Cannot refresh '{table}': {e}")))?;
                        dataset
                    }
                    None => self.open(&table).await?,
                };
                Ok::<_, GraphError>((table, Arc::new(dataset)))
            }
        });
        let mut datasets = futures::future::try_join_all(reads)
            .await?
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        // Unchanged versions keep their existing handles and snapshot identity.
        if let Some(previous) = &previous {
            for (table, dataset) in datasets.iter_mut() {
                if previous.versions.get(table) == Some(dataset.version_id()) {
                    *dataset = previous.datasets[table].clone();
                }
            }
            if datasets
                .iter()
                .all(|(table, dataset)| Arc::ptr_eq(dataset, &previous.datasets[table]))
            {
                *latest = Some((Instant::now(), previous.clone()));
                return Ok(previous.clone());
            }
        }
        let snapshot = Arc::new(GraphSnapshot::assemble(
            self.config.clone(),
            bindings,
            datasets,
            self.cache.clone(),
            self.contexts.clone(),
        )?);
        *latest = Some((Instant::now(), snapshot.clone()));
        Ok(snapshot)
    }

    async fn open(&self, table: &str) -> Result<Dataset> {
        let mut request = DescribeTableRequest::new();
        request.id = Some(vec![table.to_owned()]);
        let response = self
            .namespace
            .describe_table(request)
            .await
            .map_err(|e| error(format!("Namespace failed to resolve table '{table}': {e}")))?;
        let location = response
            .location
            .ok_or_else(|| error(format!("Namespace gave no location for '{table}'")))?;
        let mut builder = DatasetBuilder::from_uri(&location);
        if let Some(options) = response.storage_options {
            builder = builder.with_storage_options(options);
        }
        builder
            .load()
            .await
            .map_err(|e| error(format!("Failed to open table '{table}': {e}")))
    }
}
