// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bounded cache of pinned CSR adjacency snapshots and node-ID -> row-ID maps.
//!
//! * Entries are keyed by the immutable content identity of a committed dataset
//!   version (URI, branch, version, commit timestamp, transaction file), so a
//!   reopened dataset at the same version reuses its snapshot.
//! * A new version is derived incrementally from a cached snapshot of the same
//!   table when only some fragments differ: base fragments whose files,
//!   overlays or deletions changed are hidden by fragment id, and changed or new
//!   fragments are rescanned into a small delta. Anything the delta cannot
//!   express exactly (stable row IDs, schema changes, large deltas) rebuilds.
//! * Memory is charged per retained allocation (a base shared by several
//!   versions counts once). Least recently used entries that no in-flight query
//!   holds are evicted to make room; nothing is evicted from under a reader.
//! * Each key builds once, outside the cache lock, so concurrent readers of
//!   other entries never wait on a cold build and duplicate builds coalesce.
//!
//! Only topology and row identities are cached: properties, ACL masks and
//! parameters are read and evaluated per query. Share one cache only within a
//! caller's security context (see [`CsrAdjacencyCache`]).

use crate::{
    csr_index::CsrIndex,
    csr_provider::{array_endpoint, CsrScanMetrics, CsrScanOptions, Endpoint},
    csr_walk::AdjacencyView,
    CsrIndexBuilder,
};
use arrow_array::{Array, ArrayRef, RecordBatch, RecordBatchOptions, UInt64Array};
use arrow_schema::Schema;
use futures::TryStreamExt;
use lance::{table::format::Fragment, Dataset};
use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};
use tokio::sync::OnceCell;

// Conservative per-item costs for the budget (not an RSS ceiling). Two exactly
// reserved 24-byte builder edges plus the first finished CSR (neighbor + id)
// are live at the peak; a finished snapshot keeps 2 x 16 bytes per edge.
const PEAK_EDGE_BYTES: usize = 64;
const RETAINED_EDGE_BYTES: usize = 32;
// Hash entry (key, id, control, growth slack) + reverse entry + 2 offsets.
const VERTEX_BYTES: usize = 160;
// Owned text is stored twice (map key and reverse vector) with allocator slack.
const TEXT_OVERHEAD: usize = 16;
// Node-ID maps: (dense, row id) scratch and the final row-ID array per row.
const NODE_ROW_BYTES: usize = 24;
const NODE_KEY_BYTES: usize = 96;
// A delta larger than this share of its base (or than the floor) rebuilds.
const DELTA_DIVISOR: usize = 4;
const DELTA_FLOOR_ROWS: usize = 4096;
// Column entries are bounded by their own byte budget; this only caps count.
const MAX_COLUMN_SLOTS: usize = 256;

/// Lifetime counters of one cache, for monitoring.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CsrCacheStats {
    /// Ready adjacency snapshots and node-ID maps currently cached.
    pub entries: usize,
    /// Charged bytes / rows of cached entries; shared bases count once.
    pub bytes: usize,
    pub rows: usize,
    pub full_builds: u64,
    pub incremental_builds: u64,
    pub evictions: u64,
    /// Builds rejected as unsupported or larger than the configured limits.
    pub rejected_builds: u64,
    /// Cached property columns and their bytes (`max_column_cache_bytes`).
    pub column_entries: usize,
    pub column_bytes: usize,
}

/// Share only within one caller's security context. Entries are pinned to the
/// content identity of a committed version (URI, branch, version, commit
/// timestamp and transaction file), which never changes once committed, so a
/// caller that can open that version can read exactly this topology anyway.
/// A new snapshot never consumes stale adjacency. There is deliberately no
/// process-global singleton.
///
/// Memory is bounded by `max_cache_bytes`/`max_cache_rows`/`max_cache_slots`;
/// least recently used entries not held by an in-flight query are evicted. New
/// versions of a cached table are derived incrementally when possible. Native
/// multi-hop also caches pinned node-ID -> row-ID maps (IDs only, never
/// properties or masks) under the same budget; when one does not fit, the
/// filtered node scan is used instead.
#[derive(Debug)]
pub struct CsrAdjacencyCache {
    options: CsrScanOptions,
    state: Mutex<CacheState>,
    full_builds: AtomicU64,
    incremental_builds: AtomicU64,
    evictions: AtomicU64,
    rejected_builds: AtomicU64,
}

#[derive(Debug, Default)]
struct CacheState {
    slots: HashMap<CacheKey, Slot>,
    tick: u64,
    reserved_bytes: usize,
    reserved_rows: usize,
    reserved_column_bytes: usize,
}

#[derive(Debug)]
struct Slot {
    cell: Arc<OnceCell<Outcome>>,
    last_used: u64,
}

#[derive(Debug, Clone)]
enum Outcome {
    Ready(Cached),
    /// Deterministically unsupported or over the configured limits.
    Rejected,
    /// Transiently out of budget or slots; not cached, retried later.
    Busy,
}

#[derive(Debug, Clone)]
enum Cached {
    Adjacency(Arc<AdjacencySnapshot>),
    NodeIds(Arc<NodeKeySnapshot>),
    Column(Arc<ColumnSnapshot>),
}

impl Cached {
    fn in_use(&self) -> bool {
        match self {
            Cached::Adjacency(value) => Arc::strong_count(value) > 1,
            Cached::NodeIds(value) => Arc::strong_count(value) > 1,
            Cached::Column(value) => Arc::strong_count(value) > 1,
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct Lineage {
    uri: String,
    branch: Option<String>,
    kind: Kind,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
enum Kind {
    Adjacency {
        source: String,
        target: String,
    },
    NodeIds {
        key: String,
    },
    /// One stored property column (`max_column_cache_bytes`).
    Column {
        name: String,
    },
}

impl CacheKey {
    fn is_column(&self) -> bool {
        matches!(self.lineage.kind, Kind::Column { .. })
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct CacheKey {
    lineage: Lineage,
    version: u64,
    timestamp_nanos: u128,
    transaction: Option<String>,
}

impl CacheKey {
    fn new(dataset: &Dataset, kind: Kind) -> Self {
        let manifest = dataset.manifest();
        Self {
            lineage: Lineage {
                uri: dataset.uri().to_owned(),
                branch: manifest.branch.clone(),
                kind,
            },
            version: manifest.version,
            timestamp_nanos: manifest.timestamp_nanos,
            transaction: manifest.transaction_file.clone(),
        }
    }
}

/// What makes the rows of one fragment: data files, overlays and deletions.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FragmentSig {
    files: Vec<(String, Option<u32>)>,
    overlays: Vec<(String, Option<u32>, u64)>,
    deletion: Option<(u64, u64, Option<u32>, bool)>,
    physical_rows: Option<usize>,
}

impl FragmentSig {
    fn new(fragment: &Fragment) -> Self {
        Self {
            files: fragment
                .files
                .iter()
                .map(|file| (file.path.clone(), file.base_id))
                .collect(),
            overlays: fragment
                .overlays
                .iter()
                .map(|overlay| {
                    (
                        overlay.data_file.path.clone(),
                        overlay.data_file.base_id,
                        overlay.committed_version,
                    )
                })
                .collect(),
            deletion: fragment.deletion_file.as_ref().map(|file| {
                (
                    file.id,
                    file.read_version,
                    file.base_id,
                    matches!(
                        file.file_type,
                        lance::table::format::DeletionFileType::Bitmap
                    ),
                )
            }),
            physical_rows: fragment.physical_rows,
        }
    }
}

impl FragmentSig {
    /// Same data files, overlays and row count: only deletions may differ.
    fn same_data(&self, other: &Self) -> bool {
        self.files == other.files
            && self.overlays == other.overlays
            && self.physical_rows == other.physical_rows
    }
}

/// Deleted row offsets of one fragment at this version, sorted.
async fn deleted_offsets(dataset: &Dataset, fragment: u64) -> Option<Vec<u32>> {
    let fragment = dataset.get_fragment(usize::try_from(fragment).ok()?)?;
    Some(match fragment.get_deletion_vector().await.ok()? {
        Some(vector) => vector.to_sorted_iter().collect(),
        None => Vec::new(),
    })
}

/// Deleted offsets of every fragment with a deletion file.
async fn deletions(dataset: &Dataset) -> Option<HashMap<u64, Vec<u32>>> {
    let mut result = HashMap::new();
    for fragment in dataset.manifest().fragments.iter() {
        if fragment.deletion_file.is_some() {
            result.insert(fragment.id, deleted_offsets(dataset, fragment.id).await?);
        }
    }
    Some(result)
}

fn deletion_bytes(deletions: &HashMap<u64, Vec<u32>>) -> usize {
    deletions
        .values()
        .map(|offsets| 32 + offsets.len() * 4)
        .sum()
}

fn signatures(dataset: &Dataset) -> HashMap<u64, FragmentSig> {
    dataset
        .manifest()
        .fragments
        .iter()
        .map(|fragment| (fragment.id, FragmentSig::new(fragment)))
        .collect()
}

fn field_ids(dataset: &Dataset, names: &[&str]) -> Option<Vec<i32>> {
    names
        .iter()
        .map(|name| dataset.schema().field(name).map(|field| field.id))
        .collect()
}

fn text_cost(endpoint: &Endpoint) -> usize {
    match endpoint {
        Endpoint::Text(text) => 2 * (text.len() + TEXT_OVERHEAD),
        Endpoint::Number(_) | Endpoint::Signed(_) => 0,
    }
}

/// Live rows from the manifest, None when any fragment lacks row counts.
fn manifest_rows(dataset: &Dataset) -> Option<usize> {
    dataset
        .manifest()
        .fragments
        .iter()
        .try_fold(0usize, |rows, fragment| {
            rows.checked_add(fragment.num_rows()?)
        })
}

/// A base row is live unless its fragment is hidden or it was deleted since.
fn live(masked: &HashSet<u64>, deleted: &HashSet<u64>, row: u64) -> bool {
    (masked.is_empty() || !masked.contains(&fragment_of(row)))
        && (deleted.is_empty() || !deleted.contains(&row))
}

// Row addresses are `fragment_id << 32 | offset`; stable row IDs are not, so
// fragment-level deltas are only possible without them.
fn fragment_of(row_id: u64) -> u64 {
    row_id >> 32
}

/// Vertex identity shared by all versions derived from one base: base ids
/// first, then vertices first seen in a delta.
#[derive(Debug, Clone, Default)]
struct VertexExtension {
    ids: HashMap<Endpoint, u64>,
    endpoints: Vec<Endpoint>,
    text_bytes: usize,
}

/// A fully built, immutable CSR of one version.
#[derive(Debug)]
pub(crate) struct AdjacencyBase {
    outgoing: CsrIndex,
    incoming: CsrIndex,
    vertices: HashMap<Endpoint, u64>,
    endpoints: Vec<Endpoint>,
    fragments: HashMap<u64, FragmentSig>,
    deletions: HashMap<u64, Vec<u32>>,
    fields: Vec<i32>,
    rows: usize,
    bytes: usize,
}

#[derive(Debug)]
struct EdgeFragment {
    sig: FragmentSig,
    // (source, target, row id) in the base + extension dense domain.
    edges: Vec<(u64, u64, u64)>,
}

/// Topology of one pinned version: a shared base, base fragments hidden in
/// this version, and a delta CSR of changed or appended fragments.
#[derive(Debug)]
pub(crate) struct AdjacencySnapshot {
    base: Arc<AdjacencyBase>,
    masked: HashSet<u64>,
    // Base row addresses deleted since the base (deletion-only fragments).
    deleted: HashSet<u64>,
    extension: VertexExtension,
    delta_fragments: HashMap<u64, Arc<EdgeFragment>>,
    delta_outgoing: CsrIndex,
    delta_incoming: CsrIndex,
    num_vertices: u64,
    delta_rows: usize,
    delta_bytes: usize,
    // Uses fragment-encoded row addresses (incremental derivation allowed).
    row_addresses: bool,
}

/// One traversal direction of an [`AdjacencySnapshot`].
pub(crate) struct AdjacencyLayers<'a> {
    base: &'a CsrIndex,
    masked: &'a HashSet<u64>,
    deleted: &'a HashSet<u64>,
    delta: &'a CsrIndex,
    num_vertices: u64,
}

impl AdjacencyView for AdjacencyLayers<'_> {
    fn num_vertices(&self) -> u64 {
        self.num_vertices
    }
    fn neighbor_edges(&self, vertex: u64) -> impl Iterator<Item = (u64, u64)> + '_ {
        let (masked, deleted) = (self.masked, self.deleted);
        self.base
            .neighbor_edges(vertex)
            .filter(move |(id, _)| live(masked, deleted, *id))
            .chain(self.delta.neighbor_edges(vertex))
    }
}

impl AdjacencySnapshot {
    pub(crate) fn vertex(&self, key: &Endpoint) -> Option<u64> {
        self.base
            .vertices
            .get(key)
            .or_else(|| self.extension.ids.get(key))
            .copied()
    }

    pub(crate) fn endpoint(&self, id: u64) -> Option<&Endpoint> {
        let id = usize::try_from(id).ok()?;
        self.base.endpoints.get(id).or_else(|| {
            self.extension
                .endpoints
                .get(id.checked_sub(self.base.endpoints.len())?)
        })
    }

    /// Outgoing (`true`) or incoming adjacency of this version.
    pub(crate) fn view(&self, outgoing: bool) -> AdjacencyLayers<'_> {
        let (base, delta) = if outgoing {
            (&self.base.outgoing, &self.delta_outgoing)
        } else {
            (&self.base.incoming, &self.delta_incoming)
        };
        AdjacencyLayers {
            base,
            masked: &self.masked,
            deleted: &self.deleted,
            delta,
            num_vertices: self.num_vertices,
        }
    }

    #[cfg(test)]
    pub(crate) fn edges(&self) -> Vec<(Endpoint, Endpoint, u64)> {
        let view = self.view(true);
        let mut edges = Vec::new();
        for vertex in 0..self.num_vertices {
            for (id, target) in view.neighbor_edges(vertex) {
                edges.push((
                    self.endpoint(vertex).unwrap().clone(),
                    self.endpoint(target).unwrap().clone(),
                    id,
                ));
            }
        }
        edges.sort();
        edges
    }

    #[cfg(test)]
    pub(crate) fn is_incremental(&self) -> bool {
        !self.masked.is_empty() || !self.deleted.is_empty() || !self.delta_fragments.is_empty()
    }
}

/// Pinned node-ID -> row-ID lookup of one version: a shared base map, hidden
/// base fragments and a delta of changed or appended fragments.
#[derive(Debug)]
pub(crate) struct NodeKeySnapshot {
    base: Arc<NodeIdBase>,
    masked: HashSet<u64>,
    deleted: HashSet<u64>,
    delta: HashMap<Endpoint, Vec<u64>>,
    delta_fragments: HashMap<u64, Arc<NodeFragment>>,
    delta_rows: usize,
    delta_bytes: usize,
    row_addresses: bool,
}

#[derive(Debug)]
pub(crate) struct NodeIdBase {
    keys: HashMap<Endpoint, u32>,
    // CSR-style grouping: rows of key `k` are `row_ids[offsets[k]..offsets[k + 1]]`.
    offsets: Vec<u32>,
    row_ids: Vec<u64>,
    fragments: HashMap<u64, FragmentSig>,
    deletions: HashMap<u64, Vec<u32>>,
    fields: Vec<i32>,
    bytes: usize,
}

#[derive(Debug)]
struct NodeFragment {
    sig: FragmentSig,
    entries: Vec<(Endpoint, u64)>,
}

impl NodeKeySnapshot {
    /// Every live physical row whose ID equals `key`, including duplicates.
    pub(crate) fn rows<'a>(&'a self, key: &Endpoint) -> impl Iterator<Item = u64> + 'a {
        let base: &[u64] = match self.base.keys.get(key) {
            Some(&dense) => {
                let dense = dense as usize;
                &self.base.row_ids
                    [self.base.offsets[dense] as usize..self.base.offsets[dense + 1] as usize]
            }
            None => &[],
        };
        let (masked, deleted) = (&self.masked, &self.deleted);
        base.iter()
            .copied()
            .filter(move |id| live(masked, deleted, *id))
            .chain(self.delta.get(key).into_iter().flatten().copied())
    }
}

/// Incremental derivation plan relative to a cached snapshot's base.
struct DeltaPlan {
    /// Base fragments hidden entirely (removed, rewritten or rows restored).
    masked: HashSet<u64>,
    /// Fragments rescanned into the delta (new or rewritten).
    changed: Vec<Fragment>,
    /// Base row addresses deleted since the base in otherwise unchanged
    /// fragments: hidden without rescanning their data.
    deleted: HashSet<u64>,
}

async fn delta_plan(
    dataset: &Dataset,
    base_fragments: &HashMap<u64, FragmentSig>,
    base_deletions: &HashMap<u64, Vec<u32>>,
    base_rows: usize,
) -> Option<DeltaPlan> {
    let current = signatures(dataset);
    let mut masked = base_fragments
        .keys()
        .filter(|id| !current.contains_key(id))
        .copied()
        .collect::<HashSet<_>>();
    let mut changed = Vec::new();
    let mut deleted = HashSet::new();
    let mut delta_rows = 0usize;
    for fragment in dataset.manifest().fragments.iter() {
        let sig = &current[&fragment.id];
        match base_fragments.get(&fragment.id) {
            Some(base) if base == sig => continue,
            Some(base) if base.same_data(sig) => {
                // Only deletions changed: hide new deletions unless a row came back.
                let before = base_deletions
                    .get(&fragment.id)
                    .map_or(&[][..], Vec::as_slice);
                let after = deleted_offsets(dataset, fragment.id).await?;
                if before
                    .iter()
                    .all(|offset| after.binary_search(offset).is_ok())
                {
                    deleted.extend(
                        after
                            .into_iter()
                            .filter(|offset| before.binary_search(offset).is_err())
                            .map(|offset| fragment.id << 32 | u64::from(offset)),
                    );
                    continue;
                }
                masked.insert(fragment.id);
            }
            Some(_) => {
                masked.insert(fragment.id);
            }
            None => {}
        }
        delta_rows = delta_rows.checked_add(fragment.physical_rows?)?;
        changed.push(fragment.clone());
    }
    let limit = (base_rows / DELTA_DIVISOR).max(DELTA_FLOOR_ROWS);
    (delta_rows <= limit && deleted.len() <= limit).then_some(DeltaPlan {
        masked,
        changed,
        deleted,
    })
}

async fn scan_fragment(
    dataset: &Dataset,
    fragment: Fragment,
    columns: &[&str],
) -> Option<Vec<arrow_array::RecordBatch>> {
    let mut scanner = dataset.scan();
    scanner.with_fragments(vec![fragment]);
    scanner.project(columns).ok()?;
    scanner.with_row_id().batch_size(8192);
    scanner
        .try_into_stream()
        .await
        .ok()?
        .try_collect()
        .await
        .ok()
}

/// One stored property column of a pinned version, kept per fragment so
/// versions share unchanged fragments. Values are the live rows at build time
/// in row-offset order; later deletions never request the removed rows.
#[derive(Debug)]
pub(crate) struct ColumnSnapshot {
    fragments: HashMap<u64, Arc<ColumnFragment>>,
}

#[derive(Debug)]
struct ColumnFragment {
    sig: FragmentSig,
    offsets: Vec<u32>,
    values: ArrayRef,
    bytes: usize,
}

impl ColumnSnapshot {
    /// Source array and position of a row address, if cached.
    fn locate(&self, row: u64) -> Option<(&Arc<ColumnFragment>, usize)> {
        let fragment = self.fragments.get(&fragment_of(row))?;
        let position = fragment.offsets.binary_search(&(row as u32)).ok()?;
        Some((fragment, position))
    }
}

/// Scan `column` of the given fragments (all when None) into per-fragment
/// arrays. None when the column cannot be cached within `budget` bytes.
async fn scan_column(
    dataset: &Dataset,
    column: &str,
    fragments: Option<Vec<Fragment>>,
    budget: usize,
) -> Option<HashMap<u64, Arc<ColumnFragment>>> {
    let signatures = signatures(dataset);
    let mut scanner = dataset.scan();
    if let Some(fragments) = fragments {
        if fragments.is_empty() {
            return Some(HashMap::new());
        }
        scanner.with_fragments(fragments);
    }
    scanner.project(&[column]).ok()?;
    scanner.with_row_address().batch_size(8192);
    let mut batches = scanner.try_into_stream().await.ok()?;
    let mut chunks: HashMap<u64, (Vec<u32>, Vec<ArrayRef>)> = HashMap::new();
    let mut scanned = 0usize;
    while let Some(batch) = batches.try_next().await.ok()? {
        let values = batch.column_by_name(column)?;
        let addresses = batch
            .column_by_name("_rowaddr")?
            .as_any()
            .downcast_ref::<UInt64Array>()?;
        // Scans are fragment-ordered: slice each run of one fragment.
        let mut start = 0;
        while start < batch.num_rows() {
            let fragment = fragment_of(addresses.value(start));
            let mut end = start + 1;
            while end < batch.num_rows() && fragment_of(addresses.value(end)) == fragment {
                end += 1;
            }
            let entry = chunks.entry(fragment).or_default();
            entry
                .0
                .extend((start..end).map(|row| addresses.value(row) as u32));
            entry.1.push(values.slice(start, end - start));
            start = end;
        }
        scanned = scanned.checked_add(batch.get_array_memory_size())?;
        if scanned > budget.saturating_mul(2) {
            return None;
        }
    }
    let mut result = HashMap::new();
    let mut bytes = 0usize;
    for (fragment, (offsets, chunks)) in chunks {
        let refs = chunks
            .iter()
            .map(|chunk| chunk.as_ref())
            .collect::<Vec<_>>();
        let values = arrow::compute::concat(&refs).ok()?;
        let size = values.get_array_memory_size() + offsets.len() * 4;
        bytes = bytes.checked_add(size)?;
        if bytes > budget {
            return None;
        }
        result.insert(
            fragment,
            Arc::new(ColumnFragment {
                sig: signatures.get(&fragment)?.clone(),
                offsets,
                values,
                bytes: size,
            }),
        );
    }
    Some(result)
}

impl Default for CsrAdjacencyCache {
    fn default() -> Self {
        Self::new(CsrScanOptions::default())
    }
}

struct Reservation<'a> {
    cache: &'a CsrAdjacencyCache,
    bytes: usize,
    rows: usize,
    column: bool,
}

impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        let mut state = self.cache.state.lock().expect("CSR cache lock");
        if self.column {
            state.reserved_column_bytes -= self.bytes;
        } else {
            state.reserved_bytes -= self.bytes;
            state.reserved_rows -= self.rows;
        }
    }
}

impl CsrAdjacencyCache {
    pub fn new(options: CsrScanOptions) -> Self {
        Self {
            options,
            state: Mutex::new(CacheState::default()),
            full_builds: AtomicU64::new(0),
            incremental_builds: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            rejected_builds: AtomicU64::new(0),
        }
    }

    pub(crate) fn options(&self) -> &CsrScanOptions {
        &self.options
    }

    /// Current usage and lifetime counters.
    pub fn stats(&self) -> CsrCacheStats {
        let state = self.state.lock().expect("CSR cache lock");
        let used = usage(&state.slots);
        CsrCacheStats {
            entries: state
                .slots
                .iter()
                .filter(|(key, slot)| {
                    !key.is_column() && matches!(slot.cell.get(), Some(Outcome::Ready(_)))
                })
                .count(),
            column_entries: state
                .slots
                .iter()
                .filter(|(key, slot)| {
                    key.is_column() && matches!(slot.cell.get(), Some(Outcome::Ready(_)))
                })
                .count(),
            bytes: used.bytes,
            rows: used.rows,
            column_bytes: used.column_bytes,
            full_builds: self.full_builds.load(Ordering::Relaxed),
            incremental_builds: self.incremental_builds.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            rejected_builds: self.rejected_builds.load(Ordering::Relaxed),
        }
    }

    /// Build (or incrementally derive) the adjacency of `source -> target` for
    /// this dataset version ahead of queries, e.g. right after a write.
    /// Returns whether a snapshot is now cached.
    pub async fn warm_adjacency(&self, dataset: Arc<Dataset>, source: &str, target: &str) -> bool {
        self.snapshot(dataset, source, target, &CsrScanMetrics::default())
            .await
            .is_some()
    }

    /// Build (or incrementally derive) the node-ID map used by native paths.
    pub async fn warm_node_ids(&self, dataset: Arc<Dataset>, id_field: &str) -> bool {
        self.node_keys(dataset, id_field, &CsrScanMetrics::default())
            .await
            .is_some()
    }

    /// Cache these stored columns of this dataset version ahead of queries
    /// (needs `max_column_cache_bytes`). Returns how many are now cached.
    pub async fn warm_columns(&self, dataset: Arc<Dataset>, columns: &[&str]) -> usize {
        let mut cached = 0;
        for column in columns {
            if self
                .column(dataset.clone(), column, &CsrScanMetrics::default())
                .await
                .is_some()
            {
                cached += 1;
            }
        }
        cached
    }

    /// Rows `rows` (row addresses, any order, duplicates allowed) of stored
    /// `columns`, served from cached columns without storage I/O. None when
    /// the column cache is disabled, a column does not fit, or a row is not
    /// cached; callers then read through Lance.
    pub(crate) async fn cached_take(
        &self,
        dataset: &Arc<Dataset>,
        rows: &[u64],
        columns: &[&str],
        metrics: &CsrScanMetrics,
    ) -> Option<RecordBatch> {
        if self.options.max_column_cache_bytes == 0
            || dataset.manifest().uses_stable_row_ids()
            || columns.is_empty()
        {
            return None;
        }
        let schema = Arc::new(Schema::from(&dataset.schema().project(columns).ok()?));
        let mut arrays = Vec::with_capacity(schema.fields().len());
        for field in schema.fields() {
            let column = self.column(dataset.clone(), field.name(), metrics).await?;
            let mut sources: Vec<&dyn Array> = Vec::new();
            let mut source_of: HashMap<usize, usize> = HashMap::new();
            let mut indices = Vec::with_capacity(rows.len());
            for row in rows {
                let (fragment, position) = column.locate(*row)?;
                let key = Arc::as_ptr(fragment) as usize;
                let source = *source_of.entry(key).or_insert_with(|| {
                    sources.push(fragment.values.as_ref());
                    sources.len() - 1
                });
                indices.push((source, position));
            }
            arrays.push(if rows.is_empty() {
                arrow_array::new_empty_array(field.data_type())
            } else {
                arrow::compute::interleave(&sources, &indices).ok()?
            });
        }
        let batch = RecordBatch::try_new_with_options(
            schema,
            arrays,
            &RecordBatchOptions::new().with_row_count(Some(rows.len())),
        )
        .ok()?;
        metrics.column_cache_reads.fetch_add(1, Ordering::Relaxed);
        Some(batch)
    }

    async fn column(
        &self,
        dataset: Arc<Dataset>,
        name: &str,
        metrics: &CsrScanMetrics,
    ) -> Option<Arc<ColumnSnapshot>> {
        let key = CacheKey::new(&dataset, Kind::Column { name: name.into() });
        let built = self
            .lookup(key.clone(), metrics, |candidate| async move {
                let started = Instant::now();
                let previous = match candidate {
                    Some(Cached::Column(snapshot)) => Some(snapshot),
                    _ => None,
                };
                let outcome = self
                    .build_column(&dataset, name, previous, &key, metrics)
                    .await;
                metrics
                    .cache_build_nanos
                    .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
                outcome
            })
            .await;
        match built {
            Some((Cached::Column(snapshot), _)) => Some(snapshot),
            _ => None,
        }
    }

    /// Reuse every fragment whose data is unchanged since `previous` (only
    /// deletions may differ: deleted rows are never requested) and scan the
    /// rest; without a previous version, scan the whole column.
    async fn build_column(
        &self,
        dataset: &Dataset,
        name: &str,
        previous: Option<Arc<ColumnSnapshot>>,
        key: &CacheKey,
        metrics: &CsrScanMetrics,
    ) -> Outcome {
        if dataset.schema().field(name).is_none() {
            return self.rejected();
        }
        let budget = self.options.max_column_cache_bytes;
        let current = signatures(dataset);
        let mut fragments = HashMap::new();
        let mut missing = Vec::new();
        for fragment in dataset.manifest().fragments.iter() {
            let reused = previous.as_ref().and_then(|previous| {
                previous
                    .fragments
                    .get(&fragment.id)
                    .filter(|cached| cached.sig.same_data(&current[&fragment.id]))
            });
            match reused {
                Some(cached) => {
                    fragments.insert(fragment.id, cached.clone());
                }
                None => missing.push(fragment.clone()),
            }
        }
        let incremental = previous.is_some() && missing.len() < current.len();
        let estimate = missing
            .iter()
            .map(|fragment| fragment.physical_rows.unwrap_or(0))
            .sum::<usize>()
            .saturating_mul(16);
        let Some(reservation) = self.reserve(key, estimate.min(budget), 0, metrics) else {
            return Outcome::Busy;
        };
        let Some(scanned) = scan_column(dataset, name, Some(missing), budget).await else {
            return self.rejected();
        };
        drop(reservation);
        let new_bytes = scanned
            .values()
            .map(|fragment| fragment.bytes)
            .sum::<usize>();
        fragments.extend(scanned);
        let total = fragments
            .values()
            .map(|fragment| fragment.bytes)
            .sum::<usize>();
        if total > budget {
            return self.rejected();
        }
        if self.reserve(key, new_bytes, 0, metrics).is_none() {
            return Outcome::Busy;
        }
        if incremental {
            self.incremental_builds.fetch_add(1, Ordering::Relaxed);
            metrics.incremental_builds.fetch_add(1, Ordering::Relaxed);
        } else {
            self.full_builds.fetch_add(1, Ordering::Relaxed);
        }
        Outcome::Ready(Cached::Column(Arc::new(ColumnSnapshot { fragments })))
    }

    /// Cheap plan-time check: false when this version's adjacency is known to
    /// be rejected or its manifest row count cannot fit the configured limits,
    /// so callers can plan the ordinary JOIN instead of a later replay.
    pub(crate) fn admits_adjacency(&self, dataset: &Dataset, source: &str, target: &str) -> bool {
        let key = CacheKey::new(
            dataset,
            Kind::Adjacency {
                source: source.into(),
                target: target.into(),
            },
        );
        if let Ok(state) = self.state.try_lock() {
            if let Some(slot) = state.slots.get(&key) {
                return !matches!(slot.cell.get(), Some(Outcome::Rejected));
            }
        }
        match manifest_rows(dataset) {
            Some(rows) => {
                rows <= self.options.max_edges
                    && rows <= self.options.max_cache_rows
                    && rows.saturating_mul(PEAK_EDGE_BYTES) <= self.options.max_bytes
                    && rows.saturating_mul(RETAINED_EDGE_BYTES) <= self.options.max_cache_bytes
            }
            None => true,
        }
    }

    pub(crate) async fn snapshot(
        &self,
        dataset: Arc<Dataset>,
        source: &str,
        target: &str,
        metrics: &CsrScanMetrics,
    ) -> Option<Arc<AdjacencySnapshot>> {
        let kind = Kind::Adjacency {
            source: source.into(),
            target: target.into(),
        };
        let key = CacheKey::new(&dataset, kind);
        let built = self
            .lookup(key, metrics, |candidate| async move {
                let started = Instant::now();
                let candidate = match candidate {
                    Some(Cached::Adjacency(snapshot)) => Some(snapshot),
                    _ => None,
                };
                let outcome = self
                    .build_adjacency(&dataset, source, target, candidate, metrics)
                    .await;
                metrics
                    .cache_build_nanos
                    .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
                outcome
            })
            .await;
        match built {
            Some((Cached::Adjacency(snapshot), hit)) => {
                if hit {
                    metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
                }
                Some(snapshot)
            }
            _ => None,
        }
    }

    /// The node-ID map of `dataset.key`. None means unsupported or over budget;
    /// callers keep their filtered scan.
    pub(crate) async fn node_keys(
        &self,
        dataset: Arc<Dataset>,
        key: &str,
        metrics: &CsrScanMetrics,
    ) -> Option<Arc<NodeKeySnapshot>> {
        let cache_key = CacheKey::new(&dataset, Kind::NodeIds { key: key.into() });
        let built = self
            .lookup(cache_key, metrics, |candidate| async move {
                let started = Instant::now();
                let candidate = match candidate {
                    Some(Cached::NodeIds(snapshot)) => Some(snapshot),
                    _ => None,
                };
                let outcome = self.build_node_ids(&dataset, key, candidate, metrics).await;
                metrics
                    .cache_build_nanos
                    .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
                outcome
            })
            .await;
        match built {
            Some((Cached::NodeIds(snapshot), _)) => Some(snapshot),
            _ => None,
        }
    }

    /// Shared lookup: hit, join an in-flight build of the same key, or build
    /// once outside the lock. Returns the entry and whether it was a hit.
    async fn lookup<B, F>(
        &self,
        key: CacheKey,
        metrics: &CsrScanMetrics,
        build: B,
    ) -> Option<(Cached, bool)>
    where
        B: FnOnce(Option<Cached>) -> F,
        F: std::future::Future<Output = Outcome>,
    {
        let (cell, candidate) = {
            let mut state = self.state.lock().expect("CSR cache lock");
            state.tick += 1;
            let tick = state.tick;
            let cell = if let Some(slot) = state.slots.get_mut(&key) {
                slot.last_used = tick;
                match slot.cell.get() {
                    Some(Outcome::Ready(cached)) => return Some((cached.clone(), true)),
                    Some(_) => return None,
                    // Another query is building it: wait on the same cell.
                    None => slot.cell.clone(),
                }
            } else {
                let column = key.is_column();
                let (count, limit) = if column {
                    (
                        state.slots.keys().filter(|key| key.is_column()).count(),
                        MAX_COLUMN_SLOTS,
                    )
                } else {
                    (
                        state.slots.keys().filter(|key| !key.is_column()).count(),
                        self.options.max_cache_slots,
                    )
                };
                if count >= limit && !self.evict_one(&mut state, &key, metrics) {
                    return None;
                }
                let cell = Arc::new(OnceCell::new());
                state.slots.insert(
                    key.clone(),
                    Slot {
                        cell: cell.clone(),
                        last_used: tick,
                    },
                );
                cell
            };
            // Most recent ready entry of the same table/fields to derive from
            // (also used if a cancelled in-flight build must be redone here).
            let candidate = state
                .slots
                .iter()
                .filter(|(other, _)| other.lineage == key.lineage)
                .filter_map(|(other, slot)| match slot.cell.get() {
                    Some(Outcome::Ready(cached)) => Some((other.version, cached.clone())),
                    _ => None,
                })
                .max_by_key(|(version, _)| *version)
                .map(|(_, cached)| cached);
            (cell, candidate)
        };
        let outcome = cell.get_or_init(|| build(candidate)).await.clone();
        match outcome {
            Outcome::Ready(cached) => Some((cached, false)),
            Outcome::Rejected => None,
            Outcome::Busy => {
                // Not a property of the version: let a later query retry.
                let mut state = self.state.lock().expect("CSR cache lock");
                if state
                    .slots
                    .get(&key)
                    .is_some_and(|slot| Arc::ptr_eq(&slot.cell, &cell))
                {
                    state.slots.remove(&key);
                }
                None
            }
        }
    }

    /// Evict the least recently used finished entry that no query holds.
    fn evict_one(
        &self,
        state: &mut CacheState,
        protect: &CacheKey,
        metrics: &CsrScanMetrics,
    ) -> bool {
        let victim = state
            .slots
            .iter()
            .filter(|(key, _)| *key != protect && key.is_column() == protect.is_column())
            .filter(|(_, slot)| match slot.cell.get() {
                Some(Outcome::Ready(cached)) => !cached.in_use(),
                Some(_) => true,
                None => false,
            })
            .min_by_key(|(_, slot)| slot.last_used)
            .map(|(key, _)| key.clone());
        let Some(victim) = victim else {
            return false;
        };
        state.slots.remove(&victim);
        self.evictions.fetch_add(1, Ordering::Relaxed);
        metrics.cache_evictions.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Reserve budget for a build, evicting idle entries as needed. None means
    /// the request cannot fit right now (transient: the caller reports Busy).
    fn reserve(
        &self,
        protect: &CacheKey,
        bytes: usize,
        rows: usize,
        metrics: &CsrScanMetrics,
    ) -> Option<Reservation<'_>> {
        let mut state = self.state.lock().expect("CSR cache lock");
        // Property columns have their own budget and only evict each other.
        let column = protect.is_column();
        loop {
            let used = usage(&state.slots);
            let fits = if column {
                used.column_bytes
                    .saturating_add(state.reserved_column_bytes)
                    .saturating_add(bytes)
                    <= self.options.max_column_cache_bytes
            } else {
                used.bytes
                    .saturating_add(state.reserved_bytes)
                    .saturating_add(bytes)
                    <= self.options.max_cache_bytes
                    && used
                        .rows
                        .saturating_add(state.reserved_rows)
                        .saturating_add(rows)
                        <= self.options.max_cache_rows
            };
            if fits {
                if column {
                    state.reserved_column_bytes += bytes;
                } else {
                    state.reserved_bytes += bytes;
                    state.reserved_rows += rows;
                }
                return Some(Reservation {
                    cache: self,
                    bytes,
                    rows,
                    column,
                });
            }
            if !self.evict_one(&mut state, protect, metrics) {
                return None;
            }
        }
    }

    fn rejected(&self) -> Outcome {
        self.rejected_builds.fetch_add(1, Ordering::Relaxed);
        Outcome::Rejected
    }

    async fn build_adjacency(
        &self,
        dataset: &Arc<Dataset>,
        source: &str,
        target: &str,
        candidate: Option<Arc<AdjacencySnapshot>>,
        metrics: &CsrScanMetrics,
    ) -> Outcome {
        let key = CacheKey::new(
            dataset,
            Kind::Adjacency {
                source: source.into(),
                target: target.into(),
            },
        );
        let row_addresses = !dataset.manifest().uses_stable_row_ids();
        let fields = field_ids(dataset, &[source, target]);
        if let (Some(previous), Some(fields)) = (&candidate, &fields) {
            if row_addresses && previous.row_addresses && &previous.base.fields == fields {
                if let Some(plan) = delta_plan(
                    dataset,
                    &previous.base.fragments,
                    &previous.base.deletions,
                    previous.base.rows,
                )
                .await
                {
                    // None: not expressible as a delta, rebuild below.
                    if let Some(outcome) = self
                        .derive_adjacency(dataset, source, target, previous, plan, &key, metrics)
                        .await
                    {
                        return outcome;
                    }
                }
            }
        }
        let Some(fields) = fields else {
            return self.rejected();
        };
        let Some(expected) = dataset.count_rows(None).await.ok() else {
            return self.rejected();
        };
        if expected > self.options.max_edges
            || expected > self.options.max_cache_rows
            || expected.saturating_mul(PEAK_EDGE_BYTES) > self.options.max_bytes
        {
            return self.rejected();
        }
        // Reserve the retained edge arrays now; vertices are charged on commit.
        let Some(reservation) = self.reserve(
            &key,
            expected.saturating_mul(RETAINED_EDGE_BYTES),
            expected,
            metrics,
        ) else {
            return Outcome::Busy;
        };
        let base = match build_base(dataset, source, target, expected, fields, &self.options).await
        {
            Some(base) => base,
            None => return self.rejected(),
        };
        drop(reservation);
        if self.reserve(&key, base.bytes, base.rows, metrics).is_none() {
            return Outcome::Busy;
        }
        let snapshot = AdjacencySnapshot {
            num_vertices: base.endpoints.len() as u64,
            base: Arc::new(base),
            masked: HashSet::new(),
            deleted: HashSet::new(),
            extension: VertexExtension::default(),
            delta_fragments: HashMap::new(),
            delta_outgoing: empty_csr(),
            delta_incoming: empty_csr(),
            delta_rows: 0,
            delta_bytes: 0,
            row_addresses,
        };
        self.full_builds.fetch_add(1, Ordering::Relaxed);
        metrics.build_count.fetch_add(1, Ordering::Relaxed);
        Outcome::Ready(Cached::Adjacency(Arc::new(snapshot)))
    }

    /// Derive a version from `previous` by hiding changed base fragments and
    /// rescanning changed or new ones. None: fall back to a full rebuild.
    #[allow(clippy::too_many_arguments)]
    async fn derive_adjacency(
        &self,
        dataset: &Dataset,
        source: &str,
        target: &str,
        previous: &Arc<AdjacencySnapshot>,
        plan: DeltaPlan,
        key: &CacheKey,
        metrics: &CsrScanMetrics,
    ) -> Option<Outcome> {
        let base = &previous.base;
        let mut extension = previous.extension.clone();
        let mut fragments = HashMap::new();
        for fragment in plan.changed {
            let sig = FragmentSig::new(&fragment);
            if let Some(reused) = previous
                .delta_fragments
                .get(&fragment.id)
                .filter(|reused| reused.sig == sig)
            {
                fragments.insert(fragment.id, reused.clone());
                continue;
            }
            let id = fragment.id;
            let batches = scan_fragment(dataset, fragment, &[source, target]).await?;
            let mut edges = Vec::new();
            for batch in batches {
                let sources = batch.column_by_name(source)?;
                let targets = batch.column_by_name(target)?;
                let ids = batch
                    .column_by_name("_rowid")?
                    .as_any()
                    .downcast_ref::<UInt64Array>()?;
                for row in 0..batch.num_rows() {
                    if ids.is_null(row) || fragment_of(ids.value(row)) != id {
                        return None;
                    }
                    let mut dense = [0u64; 2];
                    for (slot, array) in [sources, targets].into_iter().enumerate() {
                        let value = array_endpoint(array.as_ref(), row)?;
                        dense[slot] = match base.vertices.get(&value) {
                            Some(id) => *id,
                            None => match extension.ids.get(&value) {
                                Some(id) => *id,
                                None => {
                                    let id =
                                        (base.endpoints.len() + extension.endpoints.len()) as u64;
                                    extension.text_bytes += text_cost(&value);
                                    extension.endpoints.push(value.clone());
                                    extension.ids.insert(value, id);
                                    id
                                }
                            },
                        };
                    }
                    edges.push((dense[0], dense[1], ids.value(row)));
                }
            }
            fragments.insert(id, Arc::new(EdgeFragment { sig, edges }));
        }
        let num_vertices = (base.endpoints.len() + extension.endpoints.len()) as u64;
        let delta_rows = fragments.values().map(|f| f.edges.len()).sum::<usize>();
        let hidden = base
            .outgoing
            .edge_ids()
            .iter()
            .filter(|id| !live(&plan.masked, &plan.deleted, **id))
            .count();
        let live = base.rows - hidden + delta_rows;
        if live > self.options.max_edges || num_vertices as usize > self.options.max_vertices {
            return Some(self.rejected());
        }
        let delta_bytes = delta_rows
            .checked_mul(RETAINED_EDGE_BYTES + 24)?
            .checked_add(num_vertices as usize * 16)?
            .checked_add(extension.endpoints.len() * VERTEX_BYTES + extension.text_bytes)?
            .checked_add((plan.masked.len() + plan.deleted.len()) * 16)?;
        let _reservation = self.reserve(key, delta_bytes, delta_rows, metrics)?;
        let mut outgoing = CsrIndexBuilder::new().try_reserve_edges(delta_rows).ok()?;
        let mut incoming = CsrIndexBuilder::new().try_reserve_edges(delta_rows).ok()?;
        for fragment in fragments.values() {
            for (src, dst, id) in &fragment.edges {
                outgoing = outgoing.try_add_edge_with_id(*id, *src, *dst).ok()?;
                incoming = incoming.try_add_edge_with_id(*id, *dst, *src).ok()?;
            }
        }
        let snapshot = AdjacencySnapshot {
            base: base.clone(),
            masked: plan.masked,
            deleted: plan.deleted,
            extension,
            delta_fragments: fragments,
            delta_outgoing: outgoing.with_num_vertices(num_vertices).try_build().ok()?,
            delta_incoming: incoming.with_num_vertices(num_vertices).try_build().ok()?,
            num_vertices,
            delta_rows,
            delta_bytes,
            row_addresses: true,
        };
        self.incremental_builds.fetch_add(1, Ordering::Relaxed);
        metrics.build_count.fetch_add(1, Ordering::Relaxed);
        metrics.incremental_builds.fetch_add(1, Ordering::Relaxed);
        Some(Outcome::Ready(Cached::Adjacency(Arc::new(snapshot))))
    }

    async fn build_node_ids(
        &self,
        dataset: &Arc<Dataset>,
        key_field: &str,
        candidate: Option<Arc<NodeKeySnapshot>>,
        metrics: &CsrScanMetrics,
    ) -> Outcome {
        let key = CacheKey::new(
            dataset,
            Kind::NodeIds {
                key: key_field.into(),
            },
        );
        let Some(field) = dataset.schema().field(key_field) else {
            return self.rejected();
        };
        if !crate::csr_provider::endpoint_type(&field.data_type()) {
            return self.rejected();
        }
        let row_addresses = !dataset.manifest().uses_stable_row_ids();
        let fields = vec![field.id];
        if let Some(previous) = &candidate {
            if row_addresses && previous.row_addresses && previous.base.fields == fields {
                let base_rows = previous.base.row_ids.len();
                if let Some(plan) = delta_plan(
                    dataset,
                    &previous.base.fragments,
                    &previous.base.deletions,
                    base_rows,
                )
                .await
                {
                    if let Some(outcome) = self
                        .derive_node_ids(dataset, key_field, previous, plan, &key, metrics)
                        .await
                    {
                        return outcome;
                    }
                }
            }
        }
        let Some(expected) = dataset.count_rows(None).await.ok() else {
            return self.rejected();
        };
        // Node maps may use at most half of the cache so adjacency keeps room.
        if expected > self.options.max_cache_rows / 2
            || expected > u32::MAX as usize
            || expected.saturating_mul(NODE_ROW_BYTES) > self.options.max_cache_bytes / 2
        {
            return self.rejected();
        }
        let Some(reservation) = self.reserve(
            &key,
            expected.saturating_mul(NODE_ROW_BYTES),
            expected,
            metrics,
        ) else {
            return Outcome::Busy;
        };
        let Some(base) = build_node_base(dataset, key_field, expected, fields, &self.options).await
        else {
            return self.rejected();
        };
        drop(reservation);
        if self
            .reserve(&key, base.bytes, base.row_ids.len(), metrics)
            .is_none()
        {
            return Outcome::Busy;
        }
        self.full_builds.fetch_add(1, Ordering::Relaxed);
        metrics
            .native_multihop_node_key_builds
            .fetch_add(1, Ordering::Relaxed);
        Outcome::Ready(Cached::NodeIds(Arc::new(NodeKeySnapshot {
            base: Arc::new(base),
            masked: HashSet::new(),
            deleted: HashSet::new(),
            delta: HashMap::new(),
            delta_fragments: HashMap::new(),
            delta_rows: 0,
            delta_bytes: 0,
            row_addresses,
        })))
    }

    async fn derive_node_ids(
        &self,
        dataset: &Dataset,
        key_field: &str,
        previous: &Arc<NodeKeySnapshot>,
        plan: DeltaPlan,
        key: &CacheKey,
        metrics: &CsrScanMetrics,
    ) -> Option<Outcome> {
        let mut fragments = HashMap::new();
        for fragment in plan.changed {
            let sig = FragmentSig::new(&fragment);
            if let Some(reused) = previous
                .delta_fragments
                .get(&fragment.id)
                .filter(|reused| reused.sig == sig)
            {
                fragments.insert(fragment.id, reused.clone());
                continue;
            }
            let id = fragment.id;
            let batches = scan_fragment(dataset, fragment, &[key_field]).await?;
            let mut entries = Vec::new();
            for batch in batches {
                let values = batch.column_by_name(key_field)?;
                let ids = batch
                    .column_by_name("_rowid")?
                    .as_any()
                    .downcast_ref::<UInt64Array>()?;
                for row in 0..batch.num_rows() {
                    if ids.is_null(row) || fragment_of(ids.value(row)) != id {
                        return None;
                    }
                    // A NULL ID never equals a bound key; the row cannot be selected.
                    match array_endpoint(values.as_ref(), row) {
                        Some(value) => entries.push((value, ids.value(row))),
                        None if values.is_null(row) => {}
                        None => return None,
                    }
                }
            }
            fragments.insert(id, Arc::new(NodeFragment { sig, entries }));
        }
        let mut delta: HashMap<Endpoint, Vec<u64>> = HashMap::new();
        let mut delta_rows = 0usize;
        let mut delta_bytes = (plan.masked.len() + plan.deleted.len()) * 16;
        for fragment in fragments.values() {
            for (value, row) in &fragment.entries {
                delta_rows += 1;
                delta_bytes += NODE_ROW_BYTES + NODE_KEY_BYTES + text_cost(value);
                delta.entry(value.clone()).or_default().push(*row);
            }
        }
        let _reservation = self.reserve(key, delta_bytes, delta_rows, metrics)?;
        self.incremental_builds.fetch_add(1, Ordering::Relaxed);
        metrics
            .native_multihop_node_key_builds
            .fetch_add(1, Ordering::Relaxed);
        metrics.incremental_builds.fetch_add(1, Ordering::Relaxed);
        Some(Outcome::Ready(Cached::NodeIds(Arc::new(NodeKeySnapshot {
            base: previous.base.clone(),
            masked: plan.masked,
            deleted: plan.deleted,
            delta,
            delta_fragments: fragments,
            delta_rows,
            delta_bytes,
            row_addresses: true,
        }))))
    }
}

#[derive(Debug, Default)]
struct Usage {
    bytes: usize,
    rows: usize,
    column_bytes: usize,
}

/// Charged usage of ready entries; a base or column fragment shared by several
/// versions counts once.
fn usage(slots: &HashMap<CacheKey, Slot>) -> Usage {
    let mut shared = HashSet::new();
    let mut used = Usage::default();
    for slot in slots.values() {
        match slot.cell.get() {
            Some(Outcome::Ready(Cached::Adjacency(snapshot))) => {
                if shared.insert(Arc::as_ptr(&snapshot.base) as usize) {
                    used.bytes += snapshot.base.bytes;
                    used.rows += snapshot.base.rows;
                }
                used.bytes += snapshot.delta_bytes;
                used.rows += snapshot.delta_rows;
            }
            Some(Outcome::Ready(Cached::NodeIds(snapshot))) => {
                if shared.insert(Arc::as_ptr(&snapshot.base) as usize) {
                    used.bytes += snapshot.base.bytes;
                    used.rows += snapshot.base.row_ids.len();
                }
                used.bytes += snapshot.delta_bytes;
                used.rows += snapshot.delta_rows;
            }
            Some(Outcome::Ready(Cached::Column(snapshot))) => {
                for fragment in snapshot.fragments.values() {
                    if shared.insert(Arc::as_ptr(fragment) as usize) {
                        used.column_bytes += fragment.bytes;
                    }
                }
            }
            _ => {}
        }
    }
    used
}

fn empty_csr() -> CsrIndex {
    CsrIndexBuilder::new().with_num_vertices(0).build()
}

async fn build_base(
    dataset: &Dataset,
    source: &str,
    target: &str,
    expected_rows: usize,
    fields: Vec<i32>,
    options: &CsrScanOptions,
) -> Option<AdjacencyBase> {
    let fragments = signatures(dataset);
    let deletions = deletions(dataset).await?;
    let budget = options.max_bytes;
    let edge_bytes = expected_rows.checked_mul(PEAK_EDGE_BYTES)?;
    let mut vertices: HashMap<Endpoint, u64> = HashMap::new();
    let mut endpoints: Vec<Endpoint> = Vec::new();
    let mut text_bytes = 0usize;
    let mut max_batch_bytes = 0usize;
    let estimate = |vertices: usize, text_bytes: usize, max_batch_bytes: usize| {
        edge_bytes
            .checked_add(vertices.checked_mul(VERTEX_BYTES)?)?
            .checked_add(text_bytes)?
            .checked_add(max_batch_bytes)
    };
    let mut outgoing = CsrIndexBuilder::new()
        .try_reserve_edges(expected_rows)
        .ok()?;
    let mut incoming = CsrIndexBuilder::new()
        .try_reserve_edges(expected_rows)
        .ok()?;
    let mut rows = 0usize;
    let mut scanner = dataset.scan();
    scanner.project(&[source, target]).ok()?;
    scanner.with_row_id().batch_size(8192);
    let mut batches = scanner.try_into_stream().await.ok()?;
    while let Some(batch) = batches.try_next().await.ok()? {
        max_batch_bytes = max_batch_bytes.max(batch.get_array_memory_size());
        if estimate(vertices.len(), text_bytes, max_batch_bytes)? > budget {
            return None;
        }
        let sources = batch.column_by_name(source)?;
        let targets = batch.column_by_name(target)?;
        let ids = batch
            .column_by_name("_rowid")?
            .as_any()
            .downcast_ref::<UInt64Array>()?;
        for row in 0..batch.num_rows() {
            if ids.is_null(row) || rows >= expected_rows {
                return None;
            }
            let mut dense = [0u64; 2];
            for (slot, array) in [sources, targets].into_iter().enumerate() {
                let value = array_endpoint(array.as_ref(), row)?;
                dense[slot] = match vertices.get(&value) {
                    Some(id) => *id,
                    None => {
                        if vertices.len() >= options.max_vertices {
                            return None;
                        }
                        text_bytes = text_bytes.checked_add(text_cost(&value))?;
                        if estimate(vertices.len() + 1, text_bytes, max_batch_bytes)? > budget {
                            return None;
                        }
                        let id = vertices.len() as u64;
                        endpoints.push(value.clone());
                        vertices.insert(value, id);
                        id
                    }
                };
            }
            // Stable storage row IDs, never input ordinals or CSR array positions.
            outgoing = outgoing
                .try_add_edge_with_id(ids.value(row), dense[0], dense[1])
                .ok()?;
            incoming = incoming
                .try_add_edge_with_id(ids.value(row), dense[1], dense[0])
                .ok()?;
            rows += 1;
        }
    }
    if rows != expected_rows {
        return None;
    }
    endpoints.shrink_to_fit();
    let count = vertices.len() as u64;
    // Retained: two CSRs plus both vertex directions and their text.
    let bytes = rows
        .checked_mul(RETAINED_EDGE_BYTES)?
        .checked_add(vertices.len().checked_mul(VERTEX_BYTES)?)?
        .checked_add(text_bytes)?
        .checked_add(deletion_bytes(&deletions))?;
    Some(AdjacencyBase {
        outgoing: outgoing.with_num_vertices(count).try_build().ok()?,
        incoming: incoming.with_num_vertices(count).try_build().ok()?,
        vertices,
        endpoints,
        fragments,
        deletions,
        fields,
        rows,
        bytes,
    })
}

async fn build_node_base(
    dataset: &Dataset,
    key: &str,
    expected_rows: usize,
    fields: Vec<i32>,
    options: &CsrScanOptions,
) -> Option<NodeIdBase> {
    let fragments = signatures(dataset);
    let deletions = deletions(dataset).await?;
    let budget = options.max_bytes;
    let row_bytes = expected_rows.checked_mul(NODE_ROW_BYTES)?;
    let estimate = |keys: usize, text: usize, batch: usize| {
        row_bytes
            .checked_add(keys.checked_mul(NODE_KEY_BYTES)?)?
            .checked_add(text)?
            .checked_add(batch)
    };
    if estimate(0, 0, 0)? > budget {
        return None;
    }
    let mut keys: HashMap<Endpoint, u32> = HashMap::new();
    let mut pairs: Vec<(u32, u64)> = Vec::with_capacity(expected_rows);
    let mut text_bytes = 0usize;
    let mut max_batch_bytes = 0usize;
    let mut scanner = dataset.scan();
    scanner.project(&[key]).ok()?;
    scanner.with_row_id().batch_size(8192);
    let mut batches = scanner.try_into_stream().await.ok()?;
    while let Some(batch) = batches.try_next().await.ok()? {
        max_batch_bytes = max_batch_bytes.max(batch.get_array_memory_size());
        let values = batch.column_by_name(key)?;
        let ids = batch
            .column_by_name("_rowid")?
            .as_any()
            .downcast_ref::<UInt64Array>()?;
        for row in 0..batch.num_rows() {
            if ids.is_null(row) || pairs.len() >= expected_rows {
                return None;
            }
            // A NULL ID never equals a bound key; the row cannot be selected.
            let Some(value) = array_endpoint(values.as_ref(), row) else {
                if values.is_null(row) {
                    continue;
                }
                return None;
            };
            let dense = match keys.get(&value) {
                Some(dense) => *dense,
                None => {
                    text_bytes = text_bytes.checked_add(text_cost(&value) / 2)?;
                    if estimate(keys.len() + 1, text_bytes, max_batch_bytes)? > budget {
                        return None;
                    }
                    let dense = u32::try_from(keys.len()).ok()?;
                    keys.insert(value, dense);
                    dense
                }
            };
            pairs.push((dense, ids.value(row)));
        }
    }
    let bytes = estimate(keys.len(), text_bytes, 0)?.checked_add(deletion_bytes(&deletions))?;
    // Counting sort keeps the physical order of duplicate IDs.
    let mut offsets = vec![0u32; keys.len() + 1];
    for (dense, _) in &pairs {
        offsets[*dense as usize + 1] += 1;
    }
    for index in 1..offsets.len() {
        offsets[index] += offsets[index - 1];
    }
    let mut next = offsets.clone();
    let mut row_ids = vec![0u64; pairs.len()];
    for (dense, row_id) in pairs {
        let slot = &mut next[dense as usize];
        row_ids[*slot as usize] = row_id;
        *slot += 1;
    }
    keys.shrink_to_fit();
    Some(NodeIdBase {
        keys,
        offsets,
        row_ids,
        fragments,
        deletions,
        fields,
        bytes,
    })
}

#[cfg(test)]
mod tests_support {
    use super::*;
    use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use lance::dataset::{WriteMode, WriteParams};

    pub(super) fn options() -> CsrScanOptions {
        CsrScanOptions {
            max_cache_slots: 64,
            ..Default::default()
        }
    }

    pub(super) fn edge_batch(rows: &[(Option<String>, String, u64)]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("src", DataType::Utf8, true),
                Field::new("dst", DataType::Utf8, false),
                Field::new("w", DataType::UInt64, false),
            ])),
            vec![
                Arc::new(StringArray::from(
                    rows.iter().map(|row| row.0.clone()).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    rows.iter().map(|row| row.1.clone()).collect::<Vec<_>>(),
                )),
                Arc::new(UInt64Array::from(
                    rows.iter().map(|row| row.2).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap()
    }

    pub(super) struct Rng(pub(super) u64);
    impl Rng {
        pub(super) fn next(&mut self, bound: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) % bound
        }
        pub(super) fn rows(
            &mut self,
            count: usize,
            start: u64,
        ) -> Vec<(Option<String>, String, u64)> {
            (0..count as u64)
                .map(|i| {
                    (
                        Some(format!("v{}", self.next(60))),
                        format!("v{}", self.next(60)),
                        start + i,
                    )
                })
                .collect()
        }
    }

    pub(super) async fn write(
        uri: &str,
        rows: &[(Option<String>, String, u64)],
        mode: WriteMode,
        stable: bool,
    ) -> Dataset {
        write_params(uri, rows, mode, stable, 40).await
    }

    pub(super) async fn write_with(
        uri: &str,
        rows: &[(Option<String>, String, u64)],
        mode: WriteMode,
        max_rows_per_file: usize,
    ) -> Dataset {
        write_params(uri, rows, mode, false, max_rows_per_file).await
    }

    async fn write_params(
        uri: &str,
        rows: &[(Option<String>, String, u64)],
        mode: WriteMode,
        stable: bool,
        max_rows_per_file: usize,
    ) -> Dataset {
        let batch = edge_batch(rows);
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
            uri,
            Some(WriteParams {
                mode,
                max_rows_per_file,
                max_rows_per_group: max_rows_per_file.min(1024),
                enable_stable_row_ids: stable,
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }

    /// Topology and node rows of a cold build: the oracle for derived versions.
    pub(super) async fn fresh(
        dataset: &Arc<Dataset>,
    ) -> (Vec<(Endpoint, Endpoint, u64)>, Vec<(Endpoint, Vec<u64>)>) {
        let cache = CsrAdjacencyCache::new(options());
        let metrics = CsrScanMetrics::default();
        let edges = cache
            .snapshot(dataset.clone(), "src", "dst", &metrics)
            .await
            .unwrap()
            .edges();
        (edges, node_rows(&cache, dataset, &metrics).await)
    }

    pub(super) async fn node_rows(
        cache: &CsrAdjacencyCache,
        dataset: &Arc<Dataset>,
        metrics: &CsrScanMetrics,
    ) -> Vec<(Endpoint, Vec<u64>)> {
        let keys = cache
            .node_keys(dataset.clone(), "dst", metrics)
            .await
            .unwrap();
        (0..60)
            .map(|i| {
                let key = Endpoint::Text(format!("v{i}"));
                let mut rows = keys.rows(&key).collect::<Vec<_>>();
                rows.sort_unstable();
                (key, rows)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::tests_support::*;
    use super::*;
    use lance::dataset::{optimize::compact_files, WriteMode};

    #[tokio::test]
    async fn derived_versions_equal_cold_builds_across_writes_and_time_travel() {
        let temp = tempfile::tempdir().unwrap();
        let uri = temp.path().join("edges.lance");
        let uri = uri.to_str().unwrap();
        let mut rng = Rng(7);
        let mut dataset = write(uri, &rng.rows(400, 0), WriteMode::Create, false).await;
        let cache = CsrAdjacencyCache::new(options());
        let mut next = 1000;
        let mut versions = Vec::new();
        for step in 0..12 {
            match step % 6 {
                0 | 3 => {
                    dataset = write(uri, &rng.rows(30, next), WriteMode::Append, false).await;
                    next += 30;
                }
                1 => {
                    let modulus = 5 + rng.next(5);
                    dataset.delete(&format!("w % {modulus} = 0")).await.unwrap();
                }
                2 => {
                    // Update: delete then append replacement rows.
                    dataset.delete("w % 11 = 1").await.unwrap();
                    dataset = write(uri, &rng.rows(10, next), WriteMode::Append, false).await;
                    next += 10;
                }
                4 => {
                    compact_files(&mut dataset, Default::default(), None)
                        .await
                        .unwrap();
                }
                _ => {
                    // Time travel to an earlier version, then back to the latest.
                    let earlier = versions[rng.next(versions.len() as u64) as usize];
                    let old = Arc::new(dataset.checkout_version(earlier).await.unwrap());
                    let metrics = CsrScanMetrics::default();
                    let derived = cache
                        .snapshot(old.clone(), "src", "dst", &metrics)
                        .await
                        .unwrap();
                    let expected = fresh(&old).await;
                    assert_eq!(derived.edges(), expected.0, "time travel to v{earlier}");
                    assert_eq!(node_rows(&cache, &old, &metrics).await, expected.1);
                }
            }
            versions.push(dataset.version_id());
            let current = Arc::new(dataset.clone());
            let metrics = CsrScanMetrics::default();
            let derived = cache
                .snapshot(current.clone(), "src", "dst", &metrics)
                .await
                .unwrap();
            let expected = fresh(&current).await;
            assert_eq!(
                derived.edges(),
                expected.0,
                "step {step} v{}",
                current.version_id()
            );
            assert_eq!(
                node_rows(&cache, &current, &metrics).await,
                expected.1,
                "step {step}"
            );
            if step > 0 {
                assert!(
                    derived.is_incremental(),
                    "step {step} should reuse a cached base"
                );
                // Already cached versions (after time travel) are plain hits.
                assert_eq!(
                    metrics.incremental_builds.load(Ordering::Relaxed),
                    metrics.build_count.load(Ordering::Relaxed)
                        + metrics
                            .native_multihop_node_key_builds
                            .load(Ordering::Relaxed),
                    "step {step}: every build must be derived"
                );
            }
        }
        let stats = cache.stats();
        assert_eq!(stats.full_builds, 2); // the first adjacency and node map only
        assert!(stats.incremental_builds >= 16, "{stats:?}"); // 9 new versions x (edges + node IDs)
    }

    #[tokio::test]
    async fn stable_row_ids_rebuild_instead_of_deriving() {
        let temp = tempfile::tempdir().unwrap();
        let uri = temp.path().join("stable.lance");
        let uri = uri.to_str().unwrap();
        let mut rng = Rng(11);
        write(uri, &rng.rows(100, 0), WriteMode::Create, true).await;
        let cache = CsrAdjacencyCache::new(options());
        let first = Arc::new(Dataset::open(uri).await.unwrap());
        cache
            .snapshot(first, "src", "dst", &CsrScanMetrics::default())
            .await
            .unwrap();
        let mut dataset = write(uri, &rng.rows(10, 500), WriteMode::Append, true).await;
        dataset.delete("w % 3 = 0").await.unwrap();
        let current = Arc::new(dataset);
        let derived = cache
            .snapshot(current.clone(), "src", "dst", &CsrScanMetrics::default())
            .await
            .unwrap();
        assert!(!derived.is_incremental());
        assert_eq!(derived.edges(), fresh(&current).await.0);
        assert_eq!(cache.stats().incremental_builds, 0);
    }

    #[tokio::test]
    async fn reopened_versions_hit_and_concurrent_requests_build_once() {
        let temp = tempfile::tempdir().unwrap();
        let uri = temp.path().join("edges.lance");
        let uri = uri.to_str().unwrap();
        write(uri, &Rng(3).rows(200, 0), WriteMode::Create, false).await;
        let cache = CsrAdjacencyCache::new(options());
        let (a, b) = (
            Arc::new(Dataset::open(uri).await.unwrap()),
            Arc::new(Dataset::open(uri).await.unwrap()),
        );
        let (m1, m2) = (CsrScanMetrics::default(), CsrScanMetrics::default());
        let (left, right) = tokio::join!(
            cache.snapshot(a, "src", "dst", &m1),
            cache.snapshot(b, "src", "dst", &m2),
        );
        assert!(Arc::ptr_eq(&left.unwrap(), &right.unwrap()));
        let metrics = CsrScanMetrics::default();
        let reopened = Arc::new(Dataset::open(uri).await.unwrap());
        cache
            .snapshot(reopened, "src", "dst", &metrics)
            .await
            .unwrap();
        assert_eq!(metrics.cache_hits.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.build_count.load(Ordering::Relaxed), 0);
        assert_eq!(cache.stats().full_builds, 1);
    }

    #[tokio::test]
    async fn idle_entries_are_evicted_but_never_from_under_a_reader() {
        let temp = tempfile::tempdir().unwrap();
        let mut datasets = Vec::new();
        for (index, seed) in [5u64, 6, 7].into_iter().enumerate() {
            let uri = temp.path().join(format!("edges{index}.lance"));
            datasets.push(Arc::new(
                write(
                    uri.to_str().unwrap(),
                    &Rng(seed).rows(300, 0),
                    WriteMode::Create,
                    false,
                )
                .await,
            ));
        }
        let probe = CsrAdjacencyCache::new(options());
        probe
            .snapshot(
                datasets[0].clone(),
                "src",
                "dst",
                &CsrScanMetrics::default(),
            )
            .await
            .unwrap();
        let one = probe.stats().bytes;
        // Room for one snapshot at a time.
        let cache = CsrAdjacencyCache::new(CsrScanOptions {
            max_cache_bytes: one + one / 2,
            ..options()
        });
        let held = cache
            .snapshot(
                datasets[0].clone(),
                "src",
                "dst",
                &CsrScanMetrics::default(),
            )
            .await
            .unwrap();
        let metrics = CsrScanMetrics::default();
        assert!(cache
            .snapshot(datasets[1].clone(), "src", "dst", &metrics)
            .await
            .is_none());
        assert_eq!(cache.stats().evictions, 0);
        drop(held);
        // Busy outcomes are not cached: the retry now evicts the idle entry.
        for dataset in [&datasets[1], &datasets[2], &datasets[0]] {
            let snapshot = cache
                .snapshot(dataset.clone(), "src", "dst", &metrics)
                .await
                .unwrap();
            assert_eq!(snapshot.edges(), fresh(dataset).await.0);
        }
        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.evictions, 3);
        assert!(stats.bytes <= one + one / 2);
        assert_eq!(metrics.cache_evictions.load(Ordering::Relaxed), 3);
        // Slot limit evicts the least recently used entry as well.
        let slots = CsrAdjacencyCache::new(CsrScanOptions {
            max_cache_slots: 2,
            ..options()
        });
        for dataset in datasets.iter().chain(datasets.iter()) {
            slots
                .snapshot(dataset.clone(), "src", "dst", &CsrScanMetrics::default())
                .await
                .unwrap();
        }
        assert_eq!(slots.stats().entries, 2);
        assert_eq!(slots.stats().evictions, 4);
    }

    #[tokio::test]
    async fn admission_rejects_oversized_and_unsupported_versions_at_plan_time() {
        let temp = tempfile::tempdir().unwrap();
        let uri = temp.path().join("edges.lance");
        let uri = uri.to_str().unwrap();
        let mut rows = Rng(9).rows(100, 0);
        let small = Arc::new(write(uri, &rows, WriteMode::Create, false).await);
        assert!(CsrAdjacencyCache::new(options()).admits_adjacency(&small, "src", "dst"));
        let tight = CsrAdjacencyCache::new(CsrScanOptions {
            max_edges: 99,
            ..options()
        });
        assert!(!tight.admits_adjacency(&small, "src", "dst"));
        rows.push((None, "v1".into(), 999));
        let nullable = Arc::new(write(uri, &rows, WriteMode::Overwrite, false).await);
        let cache = CsrAdjacencyCache::new(options());
        assert!(cache.admits_adjacency(&nullable, "src", "dst"));
        assert!(cache
            .snapshot(nullable.clone(), "src", "dst", &CsrScanMetrics::default())
            .await
            .is_none());
        assert!(!cache.admits_adjacency(&nullable, "src", "dst"));
        assert_eq!(cache.stats().rejected_builds, 1);
    }

    #[tokio::test]
    async fn deletions_in_one_large_fragment_hide_rows_without_rescanning() {
        let temp = tempfile::tempdir().unwrap();
        let uri = temp.path().join("one.lance");
        let uri = uri.to_str().unwrap();
        let mut dataset =
            write_with(uri, &Rng(13).rows(3000, 0), WriteMode::Create, 1_000_000).await;
        assert_eq!(dataset.manifest().fragments.len(), 1);
        let cache = CsrAdjacencyCache::new(options());
        let first = Arc::new(dataset.clone());
        cache
            .snapshot(first.clone(), "src", "dst", &CsrScanMetrics::default())
            .await
            .unwrap();
        for (round, predicate) in ["w = 17", "w % 97 = 3", "w < 5"].into_iter().enumerate() {
            dataset.delete(predicate).await.unwrap();
            let current = Arc::new(dataset.clone());
            let derived = cache
                .snapshot(current.clone(), "src", "dst", &CsrScanMetrics::default())
                .await
                .unwrap();
            assert!(
                derived.delta_fragments.is_empty(),
                "round {round} rescanned"
            );
            assert!(!derived.deleted.is_empty());
            assert_eq!(derived.edges(), fresh(&current).await.0, "round {round}");
            assert_eq!(
                node_rows(&cache, &current, &CsrScanMetrics::default()).await,
                fresh(&current).await.1
            );
        }
        // Back to the first version: deleted rows return, so the fragment is
        // rescanned from the newest cached base instead of hiding rows.
        let back = Arc::new(dataset.checkout_version(first.version_id()).await.unwrap());
        let cold = CsrAdjacencyCache::new(options());
        let latest = Arc::new(dataset.clone());
        cold.snapshot(latest, "src", "dst", &CsrScanMetrics::default())
            .await
            .unwrap();
        let derived = cold
            .snapshot(back.clone(), "src", "dst", &CsrScanMetrics::default())
            .await
            .unwrap();
        assert_eq!(derived.edges(), fresh(&back).await.0);
    }

    async fn live_rows(dataset: &Dataset) -> Vec<u64> {
        let mut scanner = dataset.scan();
        scanner.project::<&str>(&[]).unwrap();
        scanner.with_row_address();
        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        batches
            .iter()
            .flat_map(|batch| {
                let ids = batch
                    .column_by_name("_rowaddr")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .clone();
                (0..ids.len()).map(move |row| ids.value(row))
            })
            .collect()
    }

    #[tokio::test]
    async fn column_cache_reads_equal_storage_reads_across_versions() {
        use lance::dataset::ProjectionRequest;
        let temp = tempfile::tempdir().unwrap();
        let uri = temp.path().join("edges.lance");
        let uri = uri.to_str().unwrap();
        let mut rng = Rng(21);
        let mut dataset = write(uri, &rng.rows(300, 0), WriteMode::Create, false).await;
        let cache = CsrAdjacencyCache::new(CsrScanOptions {
            max_column_cache_bytes: 16 << 20,
            ..options()
        });
        assert!(CsrAdjacencyCache::new(options())
            .cached_take(
                &Arc::new(dataset.clone()),
                &[0],
                &["dst"],
                &CsrScanMetrics::default()
            )
            .await
            .is_none());
        let columns = ["src", "dst", "w"];
        let mut versions = Vec::new();
        for step in 0..10 {
            match step % 5 {
                0 => {
                    dataset = write(
                        uri,
                        &rng.rows(25, 1000 + step * 100),
                        WriteMode::Append,
                        false,
                    )
                    .await
                }
                1 => {
                    dataset
                        .delete(&format!("w % {} = 1", 3 + step))
                        .await
                        .unwrap();
                }
                2 => {
                    compact_files(&mut dataset, Default::default(), None)
                        .await
                        .unwrap();
                }
                3 => {
                    dataset.delete("w % 13 = 2").await.unwrap();
                    dataset = write(
                        uri,
                        &rng.rows(5, 5000 + step * 100),
                        WriteMode::Append,
                        false,
                    )
                    .await;
                }
                _ => {}
            }
            versions.push(dataset.version_id());
            let mut targets = vec![Arc::new(dataset.clone())];
            if step % 5 == 4 {
                let earlier = versions[rng.next(versions.len() as u64) as usize];
                targets.push(Arc::new(dataset.checkout_version(earlier).await.unwrap()));
            }
            for target in targets {
                let live = live_rows(&target).await;
                // Random live rows, unordered, with duplicates.
                let rows = (0..40)
                    .map(|_| live[rng.next(live.len() as u64) as usize])
                    .collect::<Vec<_>>();
                let projection = Arc::new(target.schema().project(&columns).unwrap());
                let expected = target
                    .take_rows(&rows, ProjectionRequest::Schema(projection))
                    .await
                    .unwrap();
                let metrics = CsrScanMetrics::default();
                match cache.cached_take(&target, &rows, &columns, &metrics).await {
                    Some(batch) => {
                        assert_eq!(batch, expected, "step {step} v{}", target.version_id());
                        assert_eq!(metrics.column_cache_reads.load(Ordering::Relaxed), 1);
                    }
                    // Only a time-travel read of rows deleted when a shared
                    // fragment was cached may miss; it then reads storage.
                    None => assert!(target.version_id() < dataset.version_id(), "step {step}"),
                }
            }
        }
        let stats = cache.stats();
        assert!(stats.column_bytes > 0 && stats.column_bytes <= 16 << 20);
        assert!(stats.incremental_builds > 0, "{stats:?}");
        // A budget too small for one column is rejected, not partially cached.
        let tiny = CsrAdjacencyCache::new(CsrScanOptions {
            max_column_cache_bytes: 64,
            ..options()
        });
        let current = Arc::new(dataset.clone());
        assert!(tiny
            .cached_take(
                &current,
                &[live_rows(&current).await[0]],
                &["dst"],
                &CsrScanMetrics::default()
            )
            .await
            .is_none());
        assert_eq!(tiny.stats().column_bytes, 0);
    }
}
