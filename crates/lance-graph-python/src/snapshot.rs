// Copyright 2024 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Version-pinned graph snapshots for multi-node services.

use std::{collections::HashMap, sync::Arc, time::Duration};

use lance_graph::{
    CsrAdjacencyCache, CsrScanOptions, CypherQuery as RustCypherQuery,
    GraphSnapshot as RustGraphSnapshot, GraphSnapshotProvider as RustGraphSnapshotProvider,
    GraphVersions as RustGraphVersions,
};
use pyo3::{exceptions::PyTypeError, prelude::*, types::PyDict};

use crate::graph::{
    graph_error_to_pyerr, python_to_json, record_batch_to_python_table, CypherQuery, GraphConfig,
};
use crate::namespace::PyDirNamespace;
use crate::RT;

/// Committed version of every table of a graph, keyed by table name.
///
/// Return ``to_token()`` to clients after a write and pass it back on later
/// reads (to any node) to read at least that write.
#[pyclass(name = "GraphVersions", module = "lance.graph", from_py_object)]
#[derive(Clone)]
pub struct GraphVersions {
    pub(crate) inner: RustGraphVersions,
}

#[pymethods]
impl GraphVersions {
    #[new]
    #[pyo3(signature = (versions=None))]
    fn new(versions: Option<HashMap<String, u64>>) -> Self {
        let inner = versions
            .unwrap_or_default()
            .into_iter()
            .fold(RustGraphVersions::new(), |versions, (table, version)| {
                versions.with(table, version)
            });
        Self { inner }
    }

    /// Parse a token produced by ``to_token``.
    #[staticmethod]
    fn from_token(token: &str) -> PyResult<Self> {
        RustGraphVersions::from_token(token)
            .map(|inner| Self { inner })
            .map_err(graph_error_to_pyerr)
    }

    fn to_token(&self) -> String {
        self.inner.to_token()
    }

    fn get(&self, table: &str) -> Option<u64> {
        self.inner.get(table)
    }

    /// Every table of ``other`` is present here at the same or a newer version.
    fn covers(&self, other: &GraphVersions) -> bool {
        self.inner.covers(&other.inner)
    }

    /// Per-table maximum of both, e.g. to combine several writes.
    fn merge_max(&mut self, other: &GraphVersions) {
        self.inner.merge_max(&other.inner);
    }

    fn to_dict(&self) -> HashMap<String, u64> {
        self.inner
            .iter()
            .map(|(table, version)| (table.to_owned(), version))
            .collect()
    }

    fn __eq__(&self, other: &GraphVersions) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!("GraphVersions({})", self.inner.to_token())
    }
}

fn versions_from(value: &Bound<'_, PyAny>) -> PyResult<RustGraphVersions> {
    if let Ok(versions) = value.extract::<GraphVersions>() {
        return Ok(versions.inner);
    }
    if let Ok(token) = value.extract::<String>() {
        return RustGraphVersions::from_token(&token).map_err(graph_error_to_pyerr);
    }
    Err(PyTypeError::new_err(
        "expected GraphVersions or a version token string",
    ))
}

/// Every table of a graph pinned to one committed version.
#[pyclass(name = "GraphSnapshot", module = "lance.graph")]
pub struct GraphSnapshot {
    pub(crate) inner: Arc<RustGraphSnapshot>,
}

#[pymethods]
impl GraphSnapshot {
    fn versions(&self) -> GraphVersions {
        GraphVersions {
            inner: self.inner.versions().clone(),
        }
    }

    /// Build every relationship adjacency and node-ID map of these versions.
    /// Returns ``{"adjacency": n, "node_ids": n, "skipped": n}``.
    fn warm(&self, py: Python) -> PyResult<HashMap<String, usize>> {
        let snapshot = self.inner.clone();
        let report = RT.block_on(Some(py), async move { snapshot.warm().await })?;
        Ok(HashMap::from([
            ("adjacency".to_owned(), report.adjacency),
            ("node_ids".to_owned(), report.node_ids),
            ("skipped".to_owned(), report.skipped),
        ]))
    }

    /// Execute a Cypher query (text or ``CypherQuery``) on these versions.
    ///
    /// ``deterministic_order=True`` makes row order and SKIP/LIMIT pages
    /// identical on every node; use it for paginated or limited results
    /// (a ``CypherQuery`` built ``with_deterministic_order()`` keeps it).
    #[pyo3(signature = (query, parameters=None, deterministic_order=false))]
    fn execute(
        &self,
        py: Python,
        query: &Bound<'_, PyAny>,
        parameters: Option<&Bound<'_, PyDict>>,
        deterministic_order: bool,
    ) -> PyResult<Py<PyAny>> {
        let mut rust = if let Ok(query) = query.extract::<PyRef<'_, CypherQuery>>() {
            query.inner.clone()
        } else {
            RustCypherQuery::new(&query.extract::<String>()?).map_err(graph_error_to_pyerr)?
        };
        if let Some(parameters) = parameters {
            for (key, value) in parameters.iter() {
                rust = rust.with_parameter(&key.extract::<String>()?, python_to_json(&value)?);
            }
        }
        if deterministic_order {
            rust = rust.with_deterministic_order(true);
        }
        let snapshot = self.inner.clone();
        let batch = RT
            .block_on(Some(py), async move {
                rust.execute_with_snapshot(&snapshot).await
            })?
            .map_err(graph_error_to_pyerr)?;
        record_batch_to_python_table(py, &batch)
    }

    fn __repr__(&self) -> String {
        format!("GraphSnapshot({})", self.inner.versions().to_token())
    }
}

/// Per-node source of version-pinned graph snapshots over a namespace.
///
/// Create one per service node (and security context) and share it across
/// requests: it owns the node's CSR cache and dataset handles.
#[pyclass(name = "GraphSnapshotProvider", module = "lance.graph")]
pub struct GraphSnapshotProvider {
    inner: Arc<RustGraphSnapshotProvider>,
}

#[pymethods]
impl GraphSnapshotProvider {
    /// Parameters
    /// ----------
    /// namespace : DirNamespace
    ///     Resolves table names to dataset locations.
    /// config : GraphConfig
    /// tables : dict, optional
    ///     Label / relationship type -> table name, when it differs (e.g.
    ///     several relationship types stored in one edge table).
    /// max_staleness_ms : int
    ///     How old ``latest()`` may be before manifests are reread.
    /// max_edges, max_build_bytes, max_cache_bytes, max_cache_rows, max_cache_slots : int, optional
    ///     CSR cache limits; ``max_build_bytes`` is ``CsrScanOptions::max_bytes``
    ///     (peak of one cold build). Defaults as in the Rust ``CsrScanOptions``.
    /// max_column_cache_bytes : int
    ///     Property column cache budget; 0 disables it.
    #[new]
    #[pyo3(signature = (
        namespace,
        config,
        tables=None,
        max_staleness_ms=1000,
        max_edges=None,
        max_build_bytes=None,
        max_cache_bytes=None,
        max_cache_rows=None,
        max_cache_slots=None,
        max_column_cache_bytes=0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        namespace: &Bound<'_, PyDirNamespace>,
        config: &GraphConfig,
        tables: Option<HashMap<String, String>>,
        max_staleness_ms: u64,
        max_edges: Option<usize>,
        max_build_bytes: Option<usize>,
        max_cache_bytes: Option<usize>,
        max_cache_rows: Option<usize>,
        max_cache_slots: Option<usize>,
        max_column_cache_bytes: usize,
    ) -> Self {
        let defaults = CsrScanOptions::default();
        let options = CsrScanOptions {
            max_edges: max_edges.unwrap_or(defaults.max_edges),
            max_bytes: max_build_bytes.unwrap_or(defaults.max_bytes),
            max_cache_bytes: max_cache_bytes.unwrap_or(defaults.max_cache_bytes),
            max_cache_rows: max_cache_rows.unwrap_or(defaults.max_cache_rows),
            max_cache_slots: max_cache_slots.unwrap_or(defaults.max_cache_slots),
            max_column_cache_bytes,
            ..defaults
        };
        let mut provider =
            RustGraphSnapshotProvider::new(namespace.borrow().inner.clone(), config.inner.clone())
                .with_cache(Arc::new(CsrAdjacencyCache::new(options)))
                .with_max_staleness(Duration::from_millis(max_staleness_ms));
        for (label, table) in tables.unwrap_or_default() {
            provider = provider.with_table(&label, table);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Newest versions within the staleness bound.
    fn latest(&self, py: Python) -> PyResult<GraphSnapshot> {
        let provider = self.inner.clone();
        let snapshot = RT
            .block_on(Some(py), async move { provider.latest().await })?
            .map_err(graph_error_to_pyerr)?;
        Ok(GraphSnapshot { inner: snapshot })
    }

    /// Reread every table's newest manifest now.
    fn refresh(&self, py: Python) -> PyResult<GraphSnapshot> {
        let provider = self.inner.clone();
        let snapshot = RT
            .block_on(Some(py), async move { provider.refresh().await })?
            .map_err(graph_error_to_pyerr)?;
        Ok(GraphSnapshot { inner: snapshot })
    }

    /// A snapshot including at least ``versions`` (GraphVersions or token):
    /// read-your-writes on any node.
    fn at_least(&self, py: Python, versions: &Bound<'_, PyAny>) -> PyResult<GraphSnapshot> {
        let versions = versions_from(versions)?;
        let provider = self.inner.clone();
        let snapshot = RT
            .block_on(Some(py), async move { provider.at_least(&versions).await })?
            .map_err(graph_error_to_pyerr)?;
        Ok(GraphSnapshot { inner: snapshot })
    }

    /// Exactly ``versions`` (every table), e.g. for pagination.
    fn at(&self, py: Python, versions: &Bound<'_, PyAny>) -> PyResult<GraphSnapshot> {
        let versions = versions_from(versions)?;
        let provider = self.inner.clone();
        let snapshot = RT
            .block_on(Some(py), async move { provider.at(&versions).await })?
            .map_err(graph_error_to_pyerr)?;
        Ok(GraphSnapshot { inner: snapshot })
    }

    /// Cache usage and lifetime counters of this node.
    fn cache_stats(&self) -> HashMap<String, u64> {
        let stats = self.inner.cache().stats();
        HashMap::from([
            ("entries".to_owned(), stats.entries as u64),
            ("bytes".to_owned(), stats.bytes as u64),
            ("rows".to_owned(), stats.rows as u64),
            ("full_builds".to_owned(), stats.full_builds),
            ("incremental_builds".to_owned(), stats.incremental_builds),
            ("evictions".to_owned(), stats.evictions),
            ("rejected_builds".to_owned(), stats.rejected_builds),
            ("column_entries".to_owned(), stats.column_entries as u64),
            ("column_bytes".to_owned(), stats.column_bytes as u64),
        ])
    }
}
