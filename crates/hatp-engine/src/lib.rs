//! HatP Engine — a persistent, embeddable MVCC key-value core.

#![doc(html_root_url = "https://docs.rs/hatp-engine/0.1.0")]

use bytes::Bytes;
use parking_lot::{Mutex, ReentrantMutex, RwLock};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use hatp_types::codec::{array_slot_to_scalar, encode_pk_value};

pub mod bloom;
pub mod clock;
pub mod compaction;
pub mod column_predicate;
pub mod crc32c;
#[macro_use]
pub mod fault_injector;
pub mod file_io;
pub mod key_index;
pub mod manifest;
pub mod memtable;
pub mod metrics;
pub mod mmap_io;
pub mod pressure;
pub mod row_codec;
pub mod sst_format;
pub mod version;
pub mod vortex_sst;
pub(crate) mod vortex_runtime;
pub mod wal;
pub mod watch;

// Re-export version module's public types for ergonomic external access.
// Users can write `hatp_engine::TxnTs` instead of `hatp_engine::version::TxnTs`.
pub use hatp_types::{OPEN_ENDED_TS, TxnTs};
pub use version::{Snapshot, VersionChain, VersionedValue};

pub use clock::{Clock, ManualClock, SystemClock};

pub use watch::{WatchEvent, Watcher};

use manifest::{Manifest, VersionEdit};
use memtable::{MemTable, prefix_upper_exclusive};
use wal::{OpType, Wal, WalRecord};

/// Result type returned by engine operations.
pub type Result<T> = std::result::Result<T, EngineError>;

/// Errors produced by the durable engine.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// A filesystem operation failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Arrow / Vortex runtime error.
    #[error("arrow error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),
    /// A length or counter exceeded the on-disk representation.
    #[error("numeric value is out of range: {0}")]
    OutOfRange(&'static str),
    /// An on-disk record is malformed.
    #[error("corrupt data: {0}")]
    Corrupt(&'static str),
    /// An on-disk record carried a malformed dynamic payload.
    #[error("corrupt data: {0}")]
    CorruptMessage(String),
    /// Vortex runtime error surfaced by the SST layer.
    #[error("vortex error: {0}")]
    Vortex(#[from] vortex_error::VortexError),
    /// A transaction batch contains no mutations.
    #[error("transaction batch is empty")]
    EmptyBatch,
    /// A checksum on a self-encoded record did not match.
    #[error("checksum mismatch")]
    ChecksumMismatch,
    /// The write path was Hard-throttled because the memtable is nearly full
    /// and flush/compaction has not drained it. The caller should retry after
    /// a flush (R-17 / SILK backpressure).
    #[error("write throttled: memtable at capacity, flush required")]
    Throttled,
    /// A row's primary-key column resolved to `NULL`. Rejecting this
    /// at ingest time matches the "no implicit NULL PK" semantics the
    /// REST API advertises; silent defaulting is the bug this variant
    /// exists to prevent.
    #[error("primary key column `{column}` in table `{table}` is NULL")]
    NullPrimaryKey {
        /// Table name supplied to `ingest_plan`.
        table: String,
        /// Column name (one of the requested PK columns).
        column: String,
    },
    /// SSI write-write conflict detected during pre-commit validation.
    /// Returned by an [`EngineHook::on_pre_commit`] implementation when a
    /// concurrent transaction writes to a key in this transaction's write set.
    #[error("SSI write-write conflict in transaction {txn} on key {:02x?}", key)]
    WriteConflict {
        /// Transaction that detected the conflict.
        txn: u64,
        /// Key that caused the conflict.
        key: Vec<u8>,
    },
    /// SSI read-write antidependency detected during pre-commit validation.
    /// Returned by an [`EngineHook::on_pre_commit`] implementation when a
    /// concurrent transaction committed a write to a key in this transaction's
    /// read set after this transaction began. Catches write-skew anomalies
    /// that pure write-write checks miss.
    #[error("SSI read-write conflict in transaction {txn} on key {:02x?}", key)]
    ReadWriteConflict {
        /// Transaction that detected the conflict.
        txn: u64,
        /// Key that was read by this txn and concurrently written by a peer.
        key: Vec<u8>,
    },
}

/// One atomic mutation in a write batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mutation {
    /// Insert or replace a value.
    Put {
        /// Key to update.
        key: Bytes,
        /// New value.
        value: Bytes,
    },
    /// Delete a key by installing a tombstone.
    Delete {
        /// Key to delete.
        key: Bytes,
    },
}

/// WAL durability mode (PR 5.1).
///
/// Controls when `sync_data` (or `FlushFileBuffers` on Windows) is called
/// after a WAL write. In embedded scenarios, trading a small durability
/// window for lower write latency is often acceptable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMode {
    /// Every commit calls `sync_data` — full durability, highest latency.
    /// This is the default and the only mode that guarantees no data loss
    /// on power failure.
    Full,
    /// A background thread calls `sync_data` every `interval` microseconds.
    /// The write path returns immediately after `write_all`. On crash, the
    /// last `interval`-worth of writes may be lost. Recovery replays the
    /// un-synced WAL tail automatically.
    Group { interval_us: u64 },
}

/// Configuration used when opening an [`Engine`].
#[derive(Clone)]
pub struct EngineConfig {
    /// Directory that owns WAL, manifest, and SST files.
    pub path: PathBuf,
    /// Approximate memtable bytes that trigger a flush recommendation.
    pub memtable_flush_bytes: usize,
    /// WAL auto-truncation threshold (bytes). When the WAL size exceeds this value,
    /// a WAL truncation is triggered after a flush. Default 64 MB.
    pub wal_max_size_bytes: u64,
    /// Time source: called when stamping `created_at` wall-clock timestamps on new SSTs.
    /// Default [`SystemClock`]; tests / deterministic simulations can inject
    /// [`ManualClock`] or a custom implementation.
    pub clock: Arc<dyn Clock>,
    /// Interval for the background worker's periodic MVCC GC. GC cleans invisible versions in the memtable;
    /// SST-level GC is handled by compaction. Default 10 s.
    pub gc_interval: Duration,
    /// Compaction strategy picker.
    /// Default [`compaction::MinOverlappingRatioPicker`] (LazyLeveling).
    pub compaction_picker: Arc<dyn compaction::CompactionPicker>,
    /// Write-pressure throttle config: soft/hard throttling when the memtable approaches the flush threshold (SILK backpressure).
    /// Default soft threshold 0.7, hard threshold 0.95.
    pub pressure: pressure::PressureConfig,
    /// MemTable backend selection: `BTreeMap` (default, `RwLock<BTreeMap>`) or
    /// `SkipMap` (`crossbeam_skiplist` lock-free concurrency, PR 2.3).
    pub memtable_impl: memtable::MemTableImpl,
    /// SST format implementation. Default [`sst_format::VortexFormat`].
    pub sst_format: Arc<dyn sst_format::SstFormat>,
    /// WAL durability mode. Default [`SyncMode::Full`] (fsync on every commit).
    /// In embedded scenarios, set to [`SyncMode::Group`] to reduce write latency.
    pub sync_mode: SyncMode,
}

impl std::fmt::Debug for EngineConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EngineConfig")
            .field("path", &self.path)
            .field("memtable_flush_bytes", &self.memtable_flush_bytes)
            .field("wal_max_size_bytes", &self.wal_max_size_bytes)
            .field("clock", &"<dyn Clock>")
            .field("gc_interval", &self.gc_interval)
            .field("compaction_picker", &"<dyn CompactionPicker>")
            .field("pressure", &self.pressure)
            .field("memtable_impl", &self.memtable_impl)
            .field("sst_format", &"<dyn SstFormat>")
            .field("sync_mode", &self.sync_mode)
            .finish()
    }
}

impl EngineConfig {
    /// Creates a configuration rooted at `path`.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            memtable_flush_bytes: pressure::DEFAULT_MEMTABLE_FLUSH_BYTES,
            wal_max_size_bytes: 64 * 1024 * 1024,
            clock: Arc::new(SystemClock),
            gc_interval: Duration::from_secs(10),
            compaction_picker: Arc::new(compaction::MinOverlappingRatioPicker::default()),
            pressure: pressure::PressureConfig::default(),
            memtable_impl: memtable::MemTableImpl::SkipMap,
            sst_format: Arc::new(sst_format::VortexFormat::new()),
            sync_mode: SyncMode::Full,
        }
    }

    /// Chained setter: set the WAL max size.
    #[must_use]
    pub fn with_wal_max_size(mut self, bytes: u64) -> Self {
        self.wal_max_size_bytes = bytes;
        self
    }

    /// Chained setter: inject a custom time source.
    #[must_use]
    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    /// Chained setter: set the background MVCC GC interval.
    #[must_use]
    pub fn with_gc_interval(mut self, interval: Duration) -> Self {
        self.gc_interval = interval;
        self
    }

    /// Chained setter: inject a custom compaction strategy picker.
    #[must_use]
    pub fn with_compaction_picker(mut self, picker: Arc<dyn compaction::CompactionPicker>) -> Self {
        self.compaction_picker = picker;
        self
    }

    /// Chained setter: configure write-pressure throttling. `memtable_flush_bytes` is synced to
    /// `PressureConfig`, ensuring the throttle threshold stays consistent with the flush threshold.
    #[must_use]
    pub fn with_pressure(mut self, mut pressure: pressure::PressureConfig) -> Self {
        pressure.memtable_flush_bytes = self.memtable_flush_bytes;
        self.pressure = pressure;
        self
    }

    /// Chained setter: select the MemTable backend (`BTreeMap` or `SkipMap`).
    #[must_use]
    pub fn with_memtable_impl(mut self, impl_kind: memtable::MemTableImpl) -> Self {
        self.memtable_impl = impl_kind;
        self
    }

    /// Chained setter: inject a custom SST format implementation.
    #[must_use]
    pub fn with_sst_format(mut self, format: Arc<dyn sst_format::SstFormat>) -> Self {
        self.sst_format = format;
        self
    }

    /// Chained setter: set the WAL durability mode.
    #[must_use]
    pub fn with_sync_mode(mut self, mode: SyncMode) -> Self {
        self.sync_mode = mode;
        self
    }
}

/// Per-table column statistics computed by decoding row payloads at flush /
/// compaction time. Populates the [`sst_format::ZoneMap`] column stats that
/// DataFusion's cost-based optimizer consumes via [`Engine::table_zone_maps`].
#[derive(Debug, Default, Clone)]
struct TableColumnStats {
    min: std::collections::HashMap<String, datafusion_common::ScalarValue>,
    max: std::collections::HashMap<String, datafusion_common::ScalarValue>,
    null_count: std::collections::HashMap<String, usize>,
}

/// Durable single-process engine.
///
/// # Group-commit model
///
/// A thread that calls `write_with_tx_sync` pushes a [`PendingWrite`] into
/// `pending_writes` before acquiring `write_guard`.  The first thread to
/// acquire the guard drains the entire queue and commits every pending
/// transaction in a single WAL `fsync` call, then notifies each waiter via
/// its `result_tx`.  A buffered channel (`sync_channel(1)`) avoids the
/// rendezvous deadlock where `send` and `recv` would otherwise block each
/// other inside the same critical section.
struct PendingWrite {
    tx_id: u64,
    mutations: Vec<Mutation>,
    result_tx: std::sync::mpsc::SyncSender<Result<u64>>,
}

pub struct Engine {
    config: EngineConfig,
    wal: Arc<Wal>,
    memtable: Arc<MemTable>,
    manifest: Mutex<Manifest>,
    /// Lock-free handle to the manifest's version set. R4: read paths
    /// (`get` / `scan_*` / zone-map probes) consult this clone of the
    /// inner `Arc<ArcSwap<VersionSet>>` instead of acquiring
    /// `manifest` (`Mutex<Manifest>`) just to call `Manifest::version_set`.
    /// Writers mutate the inner content of this ArcSwap under the manifest
    /// mutex; readers see the swap lock-free.
    version_set_arc: Arc<arc_swap::ArcSwap<manifest::VersionSet>>,
    next_tx_id: AtomicU64,
    /// The next commit sequence number (`commit_ts`) to be assigned. `write` /
    /// `write_with_tx` atomically increments this once after a successful WAL
    /// `fsync`; the returned value is the `commit_ts` for this commit. It
    /// increases strictly in **commit order**, decoupled from `next_tx_id`
    /// (which increases in **reservation order**): a transaction that reserved
    /// early but committed late must have its version's `begin_ts` equal to its
    /// true commit sequence number, otherwise a snapshot that started before it
    /// but committed after it would observe a version that should not be
    /// visible, breaking snapshot isolation (the deep root cause of R-07, and
    /// the motivation for planned PR 3.7 "decouple commit_ts from tx_id").
    ///
    /// `snapshot_ts()` returns `commit_seq - 1` (the most recently committed
    /// sequence number); reserving an id does not consume a commit sequence
    /// number, so it never advances the snapshot read point of subsequent
    /// transactions "ahead" to an uncommitted timestamp.
    commit_seq: Arc<AtomicU64>,
    /// Serialises every durable mutation (commit / flush / compaction / drop /
    /// backup). `ReentrantMutex` so a sub-compaction worker can re-enter the
    /// guard inside an already-guarded `sub_compact` call (PR 2.6).
    write_guard: Arc<ReentrantMutex<()>>,
    /// Serialises SST file operations (flush phase 2, compaction).  Separated
    /// from `write_guard` so a long-running compaction never blocks OLTP writes.
    compaction_guard: Arc<ReentrantMutex<()>>,
    /// Write-pressure throttle (SILK backpressure). Consulted on the write
    /// path before the write guard is taken, so a throttled write never blocks
    /// a healthy one.
    throttle: pressure::PressureThrottle,
    /// Application-layer hook: invoked on every put/delete/commit.
    /// Default [`NoopHook`], performance impact < 1ns (one virtual function call).
    hook: Arc<dyn EngineHook>,
    /// `true` when `hook` is `NoopHook`. Detected at construction time via `TypeId`,
    /// allowing the write path to skip the `catch_unwind` barrier entirely (saving ~50-100ns/call).
    hook_is_noop: bool,
    /// Background compaction worker handle.
    compaction_worker: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// Shutdown signal for compaction worker.
    compaction_shutdown: Arc<std::sync::atomic::AtomicBool>,
    /// Background WAL fsync worker (PR 5.1). Only spawned when
    /// `sync_mode` is [`SyncMode::Group`].
    wal_sync_worker: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// Shutdown signal for WAL fsync worker.
    wal_sync_shutdown: Arc<std::sync::atomic::AtomicBool>,
    /// `true` after [`Engine::shutdown`] has been called or the engine
    /// has been dropped. The background compaction worker checks this so
    /// `Drop` doesn't race with a manual `shutdown()`.
    dropped: Arc<std::sync::atomic::AtomicBool>,
    /// Table prefixes that have been logically deleted.
    ///
    /// Keys in these prefixes are filtered out during scans so that
    /// orphaned SST data from a dropped table does not appear in results.
    /// The prefix is recorded here so new writes to a re-created table
    /// with the same name will not collide with stale orphan data.
    ///
    /// # Durability
    ///
    /// This set is persisted to the manifest as [`VersionEdit::DroppedPrefix`]
    /// / [`VersionEdit::UndroppedPrefix`] edits and replayed on open, so the
    /// drop filter survives a restart (PR 0.4). The in-memory `ArcSwap` is the
    /// hot-path view of that durable state; `drop_table_prefix` writes the
    /// manifest edit before mutating it so a crash cannot diverge the two.
    ///
    /// Lock-free via `ArcSwap` — readers see a consistent snapshot without
    /// acquiring any lock.  Writers (drop / undrop) clone the current set,
    /// mutate, and atomically swap the pointer.
    dropped_tables: Arc<arc_swap::ArcSwap<BTreeSet<Vec<u8>>>>,
    /// Active read-snapshot pins: `snapshot_ts -> refcount`. Compaction and
    /// memtable GC consult [`Engine::watermark`] (the minimum pinned snapshot)
    /// so they never drop an MVCC version that a live transaction can still
    /// observe. The default watermark when nothing is pinned is
    /// [`Engine::snapshot_ts`].
    pinned_snapshots: Mutex<BTreeMap<u64, usize>>,
    /// Cached total row count across all SST files.
    ///
    /// This is maintained incrementally:
    /// - Incremented on `flush` (added rows from memtable)
    /// - Decremented on compaction delete (removed SST files)
    ///
    /// This avoids the O(total_SST_bytes) scan that `scan_stats` previously
    /// performed on every call. The count is approximate — it includes
    /// tombstones and superseded versions, same as before.
    sst_row_count: AtomicUsize,
    /// Per-table physical row count (`table\0` prefix -> rows). Maintained
    /// incrementally alongside `sst_row_count` so DataFusion statistics pushdown
    /// (PR 2.9) can report a per-table `num_rows` instead of the global count
    /// (which mis-reports one table's size as every table's size).
    table_rows: RwLock<BTreeMap<Vec<u8>, usize>>,
    /// Prometheus-compatible metrics.
    metrics: Arc<metrics::EngineMetrics>,
    /// Weak self-reference, populated at construction via [`Arc::new_cyclic`],
    /// so `&self` read methods can upgrade to an [`Arc`] and pin their snapshot
    /// against concurrent compaction.
    self_ref: Weak<Engine>,
    /// Commit-progress watcher (PR 3.1): publishes every durable commit to
    /// CDC subscribers and wakes `wait_for_resolved` waiters.
    watcher: watch::Watcher,
    /// Group-commit pending queue.  When multiple threads call `write` concurrently,
    /// the first to acquire `write_guard` drains the entire queue and commits all
    /// pending transactions in a single WAL `fsync`.  Each entry carries a oneshot
    /// sender so the caller can receive its `commit_ts` (or error).
    pending_writes: Mutex<Vec<PendingWrite>>,
    /// Lazily populated bloom-filter cache (PR 3.10), keyed by SST file id.
    /// `Engine::get` consults it without re-reading the sidecar file on every
    /// point lookup; a flush/compaction inserts the freshly written filter.
    ///
    /// Lock-free via `ArcSwap` — readers see a consistent snapshot without
    /// acquiring any lock.  Writers (flush / compaction) clone the current
    /// map, insert the new filter, and atomically swap the pointer.
    bloom_cache: Arc<arc_swap::ArcSwap<BTreeMap<u64, Arc<bloom::BloomFilter>>>>,
    /// Lazily populated key-index cache (PR 5.0), keyed by SST file id.
    /// `Engine::get` consults it for O(log n) point lookups without opening
    /// the Vortex file. A missing/corrupt sidecar falls back to the Vortex
    /// read path. Same lock-free rcu pattern as bloom_cache.
    key_index_cache: Arc<arc_swap::ArcSwap<BTreeMap<u64, Arc<key_index::KeyIndex>>>>,
    /// Per-table Arrow schemas (`table name -> schema`). Registered on ingest
    /// and persisted to the `hatp.schemas` sidecar so `scan_table` can decode
    /// compact row payloads without the schema being repeated per row. When a
    /// table has no registered schema (e.g. raw `Engine::write` keys), scans
    /// fall back to the caller-supplied `fallback_schema`.
    table_schemas: RwLock<std::collections::HashMap<String, arrow_schema::SchemaRef>>,
    /// Per-file, per-table column statistics (min / max / null_count), computed
    /// at flush / compaction time by decoding each row's compact payload. Keyed
    /// by SST file id then table name. This is an in-memory derived cache: it is
    /// rebuilt on the next flush / compaction after a restart, and `statistics()`
    /// degrades to `Absent` column stats for files that predate this cache.
    column_stats: RwLock<BTreeMap<u64, BTreeMap<String, TableColumnStats>>>,
    /// Hot-key cache: key → SST file_id.  Populated on successful point
    /// lookups; the next `get` for the same key skips the candidate-files
    /// search and goes directly to the key index.  Stale entries (file
    /// compacted away) are harmless — the key-index lookup returns `None`
    /// and the full search path runs.  Capped at [`HOT_KEY_CACHE_MAX`].
    hot_key_cache: RwLock<std::collections::HashMap<Bytes, u64>>,
    /// Cached primary-key column indices per table.
    pk_column_cache: RwLock<std::collections::HashMap<String, Vec<(usize, String)>>>,
    /// Monotonically increasing version counter bumped on every flush and
    /// compaction.  DataFusion's `TableProviderAdapter` uses it as a
    /// cache-invalidation signal.
    stats_version: AtomicU64,
}

impl Drop for Engine {
    /// Stops the background compaction worker when the last
    /// `Arc<Engine>` is dropped. Without this, dropping a `Database`
    /// would orphan the worker thread (it would keep polling every 5s
    /// until the process exits).
    fn drop(&mut self) {
        // Signal shutdown once and join the worker. `shutdown` already
        // implements this; calling it here covers the "dropped without
        // explicit shutdown" path.
        self.dropped
            .store(true, std::sync::atomic::Ordering::SeqCst);
        self.shutdown();
    }
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("config", &self.config)
            .field("crash_test", &"...")
            .field("hook", &"<dyn EngineHook>")
            .finish()
    }
}

/// RAII guard returned by [`Engine::pin_snapshot`]. Pinning a snapshot
/// advances the engine's GC/compaction watermark so MVCC versions visible
/// at that timestamp survive until the guard is dropped.
///
/// The guard holds an [`Arc<Engine>`] (not a bare `&Engine`) so a transaction
/// can outlive the call site that pinned it; dropping the guard releases the
/// pin. Pins are refcounted, so pinning the same timestamp twice is safe.
#[derive(Debug)]
pub struct SnapshotGuard {
    engine: Arc<Engine>,
    ts: u64,
}

impl Drop for SnapshotGuard {
    fn drop(&mut self) {
        self.engine.unpin_snapshot(self.ts);
    }
}

/// Engine callback hook.
///
/// Invoked on the critical path of every write/delete/transaction commit. The default
/// [`NoopHook`] does nothing. In integration scenarios, `hatp-tx` can provide
/// `TxManagerHook` (transaction manager integration point), allowing the post-commit
/// callback to advance the transaction state machine via `TxManager::commit(...)`;
/// `hatp-frontend` can also use it to push mutations to downstream consumers such as
/// CDC outbox.
///
/// # Performance note
/// Implementations must be **non-blocking and zero-allocation**. `hatp-engine` invokes
/// the hook sequentially under the write-path lock, so long-running tasks directly
/// slow commit throughput.
///
/// # Downcast support
///
/// `as_any` allows downcasting `Arc<dyn EngineHook>` to a concrete type. The `hatp`
/// bus layer uses it to detect whether `TxManagerHook` is injected, to decide whether
/// to enable SSI. **Implementors must return `self`**, because the trait default method
/// has no `Self: Sized` bound (dyn-safe trait cannot use `where Self: Sized` default
/// methods). `NoopHook` / `TxManagerHook` both follow this convention.
pub trait EngineHook: Send + Sync + 'static {
    /// Invoked after a put is committed.
    fn on_put(&self, _key: &[u8], _value: &[u8], _tx_id: u64) {}
    /// Invoked after a delete is committed.
    fn on_delete(&self, _key: &[u8], _tx_id: u64) {}
    /// Invoked after a transaction commit completes (`tx_id` has been fsync'd + memtable applied).
    /// `commit_ts` is the commit sequence number assigned to this commit (commit order, strictly
    /// increasing), decoupled from `tx_id` (reservation order); SSI antidependency checks must
    /// use `commit_ts` rather than `tx_id`.
    fn on_tx_commit(&self, _tx_id: u64, _commit_ts: u64) {}
    /// SSI abort callback.
    fn on_ssi_abort(&self, _tx_id: u64) {}
    /// Pre-commit hook: called before WAL append, for SSI conflict detection and other
    /// "validate before persisting" scenarios. Returning `Err` **completely skips**
    /// WAL/memtable/SST writes; the caller receives the error. The default implementation
    /// returns `Ok(())`, backward-compatible.
    fn on_pre_commit(&self, _tx_id: u64, _mutations: &[Mutation]) -> Result<()> {
        Ok(())
    }
    /// Erases `self` to `&dyn Any`, for downcasting `Arc<dyn EngineHook>`.
    /// Implementations must return `self` (see trait-level docs).
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Default no-op implementation.
///
/// All methods are empty; the compiler inlines them at the `Engine` call site, near-zero cost.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopHook;

impl EngineHook for NoopHook {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Detects whether `hook` is `NoopHook` at `Engine` construction time, returning `bool`.
/// Uses `TypeId` comparison, zero virtual-dispatch overhead.
#[inline]
fn detect_noop_hook(hook: &dyn EngineHook) -> bool {
    hook.as_any().type_id() == std::any::TypeId::of::<NoopHook>()
}

// ── Cache capacity limits ───────────────────────────────────────────────────
// Derived caches (bloom, key index, hot key) are rebuildable from SST files.
// When a cache exceeds its limit, the entire cache is cleared — a simple,
// amortised-O(1) eviction that avoids the pointer-chasing of a per-entry LRU.

/// Max hot-key cache entries.  Exceeding triggers a full clear.
const HOT_KEY_CACHE_MAX: usize = 16_384;
/// Max bloom filters cached (each ~1–64 KiB).
const BLOOM_CACHE_MAX: usize = 512;
/// Max key indexes cached.
const KEY_INDEX_CACHE_MAX: usize = 256;
/// Max per-SST column-statistics maps.
const COLUMN_STATS_CACHE_MAX: usize = 1024;

/// Return the on-disk SST file path for a given file id.
///
/// Files use the `.vortex` extension to match the Vortex file format.
fn sst_path(path: &Path, file_id: u64) -> PathBuf {
    path.join(format!("sst-{file_id:020}.vortex"))
}

/// Return the bloom-filter sidecar path for a given SST file id (PR 3.10).
///
/// The bloom filter is a derived cache: its path is deterministically derived
/// from the file id (not stored in the manifest), and a missing/corrupt
/// sidecar simply falls back to the full SST read.
fn bloom_path(path: &Path, file_id: u64) -> PathBuf {
    path.join(format!("bloom-{file_id:020}.bf"))
}

/// Return the key-index sidecar path for a given SST file id (PR 5.0).
///
/// Like the bloom filter, the key index is a derived cache — never stored in
/// the manifest, and a missing/corrupt sidecar falls back to the Vortex read.
fn key_index_path(path: &Path, file_id: u64) -> PathBuf {
    path.join(format!("kidx-{file_id:020}.bin"))
}

/// Return the table-schema registry sidecar path.
fn schemas_path(path: &Path) -> PathBuf {
    path.join("hatp.schemas")
}

/// Loads the persisted table-schema registry from `hatp.schemas`, or an empty
/// map when the sidecar does not exist yet.
fn load_table_schemas(path: &Path) -> std::collections::HashMap<String, arrow_schema::SchemaRef> {
    let file = schemas_path(path);
    let bytes = match std::fs::read(&file) {
        Ok(bytes) => bytes,
        Err(_) => return std::collections::HashMap::new(),
    };
    let raw: std::collections::HashMap<String, arrow_schema::Schema> = match serde_json::from_slice(&bytes) {
        Ok(raw) => raw,
        Err(error) => {
            tracing::warn!(error = %error, "table schema sidecar decode failed; ignoring");
            return std::collections::HashMap::new();
        }
    };
    raw.into_iter()
        .map(|(name, schema)| (name, Arc::new(schema)))
        .collect()
}

/// Persists the table-schema registry to `hatp.schemas` (atomic tmp + rename).
/// Best-effort: a failed write is logged — the registry can be rebuilt by the
/// next ingest of each table.
fn save_table_schemas(path: &Path, schemas: &std::collections::HashMap<String, arrow_schema::SchemaRef>) {
    use std::io::Write;
    let raw: std::collections::HashMap<String, &arrow_schema::Schema> = schemas
        .iter()
        .map(|(name, schema)| (name.clone(), schema.as_ref()))
        .collect();
    let json = match serde_json::to_vec(&raw) {
        Ok(json) => json,
        Err(error) => {
            tracing::error!(error = %error, "table schema serialize failed; skipping persist");
            return;
        }
    };
    let file = schemas_path(path);
    let tmp = file.with_extension("schemas.tmp");
    let result = (|| -> std::io::Result<()> {
        let mut handle = std::fs::File::create(&tmp)?;
        handle.write_all(&json)?;
        handle.sync_all()?;
        std::fs::rename(&tmp, &file)?;
        Ok(())
    })();
    if let Err(error) = result {
        tracing::error!(error = %error, "table schema persist failed");
    }
}

/// Writes the bloom-filter sidecar for a freshly written SST (PR 3.10) and
/// returns the built filter (for the engine's in-memory cache). Best-effort:
/// a failed write logs a warning, returns `None`, and the authoritative read
/// path works without it.
fn write_bloom_for(
    path: &Path,
    file_id: u64,
    rows: &[memtable::VersionedRow],
) -> Option<Arc<bloom::BloomFilter>> {
    let mut filter = bloom::BloomFilter::new(rows.len(), 0.01);
    for row in rows {
        filter.insert(&row.key);
    }
    if let Err(error) = filter.write_to(&bloom_path(path, file_id)) {
        tracing::warn!(file_id, error = %error, "bloom filter write failed; falling back to full read");
        return None;
    }
    Some(Arc::new(filter))
}

impl Engine {
    /// Opens an engine and replays every committed WAL transaction.
    pub fn open(config: EngineConfig) -> Result<Arc<Self>> {
        Self::open_with_hook(config, Arc::new(NoopHook))
    }

    /// Opens an engine with an [`EngineHook`].
    ///
    /// Recovery flow:
    /// 1. WAL replay → memtable visible;
    /// 2. SST list is reconstructed from manifest; `get` / `scan` queries Vortex directly;
    /// 3. `highest_tx` is derived from SST `begin_ts` after flush (WAL is empty).
    pub fn open_with_hook(
        config: EngineConfig,
        hook: Arc<dyn EngineHook>,
    ) -> Result<Arc<Self>> {
        fs::create_dir_all(&config.path)?;
        let wal_path = config.path.join("hatp.wal");
        let manifest_path = config.path.join("MANIFEST");
        let recovered = Wal::recover(&wal_path)?;
        let memtable = Arc::new(MemTable::with_impl(config.memtable_impl));
        let manifest = Manifest::open(&manifest_path)?;
        // R4: take a lock-free-friendly clone of the manifest's `ArcSwap` so
        // the engine's read path (`get` / `scan_*`) doesn't need to acquire
        // the manifest mutex just to load the version set. Writers still go
        // through the mutex (they have to serialize against the file
        // handle); they mutate the inner ArcSwap content that the engine
        // shares via this clone.
        let version_set_arc = manifest.version_set_arc();
        // Metrics are created before the engine so the recovery scan below can
        // record failures (a corrupt SST must not abort the whole open, but it
        // must be observable).
        let metrics = Arc::new(metrics::EngineMetrics::new());
        // ── 1. Reconcile committed progress against SSTs first ──────────────
        // SST rows carry `begin_ts == commit_ts` (the commit-order sequence,
        // see `write_with_tx`). The highest SST `begin_ts` is the durable
        // commit watermark: WAL replay must continue allocating `commit_ts`
        // from `sst_max_commit + 1` so commit order stays monotonic across a
        // flush boundary. Reads return `None` if `snapshot_ts < begin_ts`, so
        // getting this number wrong would mask committed data.
        //
        // read as-is (backward-compatible with plain SSTs).
        //
        // NOTE: This recovery scan also accumulates `sst_row_count` incrementally,
        // so `scan_stats` can return the cached count without re-scanning all SSTs.
        let manifest_snapshot = manifest.version_set();
        let mut initial_sst_rows = 0_usize;
        let mut recovered_table_rows: BTreeMap<Vec<u8>, usize> = BTreeMap::new();
        let mut sst_max_commit = 0_u64;
        for files in manifest_snapshot.levels().values() {
            for file_id in files {
                let path = sst_path(&config.path, *file_id);
                match config.sst_format.read_all(&path) {
                    Ok(rows) => {
                        initial_sst_rows = initial_sst_rows.saturating_add(rows.len());
                        for row in &rows {
                            sst_max_commit = sst_max_commit.max(row.version.begin_ts);
                            // Recover per-table row counts so statistics survive restart without losing the SST portion.
                            let entry = recovered_table_rows
                                .entry(key_table_prefix(&row.key).to_vec())
                                .or_insert(0_usize);
                            *entry = (*entry).saturating_add(1_usize);
                        }
                    }
                    Err(error) => {
                        metrics.record_sst_read_failure();
                        tracing::warn!(file_id, error = %error, "SST read failed during recovery");
                    }
                }
            }
        }
        // ── 2. Replay the WAL in commit order ───────────────────────────────
        // Records stream into `pending` keyed by tx_id; when the matching
        // `Commit` arrives we flush the whole batch through the memtable in a
        // single lock acquisition via `apply_records_batch`. Because
        // `write_guard` serialises commits and each `append_batch_commit_and_sync`
        // writes one transaction's frames contiguously, the physical order of
        // `Commit` frames equals the true commit order — so assigning
        // `commit_ts = next_commit_ts++` here reconstructs the exact
        // commit-order sequence the live engine produced.
        let mut pending = BTreeMap::<u64, Vec<WalRecord>>::new();
        let mut highest_tx = 0_u64;
        let mut next_commit_ts = sst_max_commit.saturating_add(1);
        let mut committed_batch: Vec<WalRecord> = Vec::with_capacity(64);
        for record in recovered {
            highest_tx = highest_tx.max(record.tx_id);
            match record.op {
                OpType::Put | OpType::Delete => {
                    pending.entry(record.tx_id).or_default().push(record);
                }
                OpType::Commit => {
                    if let Some(records) = pending.remove(&record.tx_id) {
                        committed_batch.clear();
                        committed_batch.extend(records);
                        memtable.apply_records_batch(&committed_batch, next_commit_ts)?;
                        next_commit_ts = next_commit_ts.saturating_add(1);
                    }
                }
                OpType::Abort => {
                    pending.remove(&record.tx_id);
                }
            }
        }
        // Recover persisted dropped-table prefixes (PR 0.4): load into
        // `dropped_tables` immediately after replay so orphaned SST data continues to be filtered after restart.
        let recovered_dropped: BTreeSet<Vec<u8>> =
            manifest_snapshot.dropped_prefixes().into_iter().collect();
        // Recover the persisted table schema registry (`hatp.schemas`) for scan_table to decode
        // compact row payloads.
        let recovered_schemas = load_table_schemas(&config.path);
        // `next_tx_id` lives in the *reservation* space (tx_id); the commit
        // sequence is a separate space. Recovery reconciles both: the next
        // tx_id continues after the highest WAL tx_id, while the next commit
        // sequence continues after the highest SST/WAL commit.
        let next_tx_id = highest_tx.saturating_add(1);

        // Create the engine instance first. `new_cyclic` hands the constructor
        // a `Weak` so `&self` read methods can later upgrade to an `Arc` and
        // pin their snapshot (see `self_ref`). `wal` / `commit_seq` /
        // `write_guard` are created outside the closure so the group-commit
        // worker can clone them.
        let pressure_config = config.pressure;
        let wal = Arc::new(Wal::open(&wal_path)?);
        let commit_seq = Arc::new(AtomicU64::new(next_commit_ts));
        // The watcher's resolved watermark starts at the recovered committed
        // watermark: every replayed commit is already durable and applied.
        let watcher = watch::Watcher::new(next_commit_ts.saturating_sub(1));
        let write_guard = Arc::new(ReentrantMutex::new(()));
        let compaction_guard = Arc::new(ReentrantMutex::new(()));
        let engine: Arc<Self> = Arc::new_cyclic(|weak| {
            let hook_is_noop = detect_noop_hook(hook.as_ref());
            Self {
            config,
            wal,
            memtable,
            manifest: Mutex::new(manifest),
            version_set_arc,
            next_tx_id: AtomicU64::new(next_tx_id),
            commit_seq,
            write_guard,
            compaction_guard,
            throttle: pressure::PressureThrottle::new(pressure_config),
            hook,
            hook_is_noop,
            compaction_worker: Mutex::new(None),
            compaction_shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            wal_sync_worker: Mutex::new(None),
            wal_sync_shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            dropped: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            dropped_tables: Arc::new(arc_swap::ArcSwap::from_pointee(recovered_dropped)),
            pinned_snapshots: Mutex::new(BTreeMap::new()),
            sst_row_count: AtomicUsize::new(initial_sst_rows),
            table_rows: RwLock::new(recovered_table_rows),
            metrics,
            self_ref: weak.clone(),
            watcher,
            pending_writes: Mutex::new(Vec::new()),
            bloom_cache: Arc::new(arc_swap::ArcSwap::from_pointee(BTreeMap::new())),
            key_index_cache: Arc::new(arc_swap::ArcSwap::from_pointee(BTreeMap::new())),
            table_schemas: RwLock::new(recovered_schemas),
            column_stats: RwLock::new(BTreeMap::new()),
            hot_key_cache: RwLock::new(std::collections::HashMap::new()),
            pk_column_cache: RwLock::new(std::collections::HashMap::new()),
            stats_version: AtomicU64::new(1),
            }
        });


        // Start background compaction worker. The thread owns its own
        // `Arc` clones of `engine` / `shutdown` so the borrowed `&Engine`
        // / `&AtomicBool` signature of `compaction_worker_loop` stays
        // lifetime-clean.
        let engine_arc = Arc::clone(&engine);
        let shutdown_arc = Arc::clone(&engine.compaction_shutdown);
        let handle = std::thread::spawn(move || {
            compaction_worker_loop(&engine_arc, &shutdown_arc);
        });

        // Store the handle
        *engine.compaction_worker.lock() = Some(handle);

        // PR 5.1: spawn background WAL fsync thread for SyncMode::Group.
        if let SyncMode::Group { interval_us } = engine.config.sync_mode {
            let wal = Arc::clone(&engine.wal);
            let shutdown = Arc::clone(&engine.wal_sync_shutdown);
            let handle = std::thread::spawn(move || {
                wal_sync_worker_loop(&wal, &shutdown, interval_us);
            });
            *engine.wal_sync_worker.lock() = Some(handle);
        }

        Ok(engine)
    }

    /// Stops the background compaction worker.
    ///
    /// Idempotent: a second call is a no-op. The `Engine::Drop` impl calls
    /// this so callers don't need to.
    pub fn shutdown(&self) {
        // Set the shutdown flag; the worker loop checks it every iteration.
        self.compaction_shutdown
            .store(true, std::sync::atomic::Ordering::SeqCst);
        // Take the handle out — a second `shutdown` call would otherwise
        // try to join an already-joined thread and block forever.
        if let Some(handle) = self.compaction_worker.lock().take() {
            let _ = handle.join();
        }
        // PR 5.1: stop the WAL fsync worker too.
        self.wal_sync_shutdown
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(handle) = self.wal_sync_worker.lock().take() {
            let _ = handle.join();
        }
    }

    /// Returns the engine's [`EngineHook`]. Used by [`hatp::Database`]
    /// to downcast a `TxManagerHook` and decide whether SSI is wired
    /// into the engine. The hook is opaque to anything outside the
    /// `hatp` facade — callers should not drive it directly.
    #[must_use]
    pub fn hook(&self) -> &Arc<dyn EngineHook> {
        &self.hook
    }

    /// Returns the engine's [`metrics::EngineMetrics`] for monitoring and observability.
    #[must_use]
    pub fn metrics(&self) -> Arc<metrics::EngineMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Returns the registered Arrow [`arrow_schema::SchemaRef`] for `table`,
    /// or `None` when the engine has no schema registered (raw-KV path).
    /// Frontend DML planning uses this to translate DataFusion `Expr`s into
    /// engine-native column predicates; the engine itself only consults the
    /// schema on the scan path.
    #[must_use]
    pub fn table_schema(&self, table: &str) -> Option<arrow_schema::SchemaRef> {
        self.table_schemas.read().get(table).cloned()
    }

    /// Returns the directory containing this engine's durable state.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.config.path
    }

    /// Returns the live memtable.
    #[must_use]
    pub fn memtable(&self) -> Arc<MemTable> {
        Arc::clone(&self.memtable)
    }

    /// Returns the latest *committed* snapshot timestamp.
    ///
    /// This is `commit_seq - 1` — the most recent `commit_ts` handed out by
    /// [`Engine::write_with_tx`] after a durable `fsync`. It is **not**
    /// `next_tx_id - 1` (which counts ids merely reserved via
    /// [`Engine::reserve_tx_id`]): reserving an id does not consume a commit
    /// sequence number, so a read at this timestamp sees every committed
    /// version and never a reserved-but-uncommitted one.
    #[must_use]
    pub fn snapshot_ts(&self) -> u64 {
        self.commit_seq.load(Ordering::Acquire).saturating_sub(1)
    }

    /// Returns a clone of the engine's commit-progress [`Watcher`] (PR 3.1).
    /// CDC / replication consumers use it to wait for a resolved watermark and
    /// stream commit events without holding the engine's write lock.
    #[must_use]
    pub fn watcher(&self) -> watch::Watcher {
        self.watcher.clone()
    }

    /// Returns the cached bloom filter for an SST file id (PR 3.10), lazily
    /// reading the sidecar into the cache on first use. A missing/corrupt
    /// sidecar yields `None` and the caller falls back to the full read.
    fn bloom_for(&self, file_id: u64) -> Option<Arc<bloom::BloomFilter>> {
        let snapshot = self.bloom_cache.load();
        if let Some(cached) = snapshot.get(&file_id) {
            return Some(Arc::clone(cached));
        }
        let filter = bloom::BloomFilter::read_from(&bloom_path(&self.config.path, file_id))?;
        let filter = Arc::new(filter);
        self.bloom_insert(file_id, Arc::clone(&filter));
        Some(filter)
    }

    /// Inserts a bloom filter into the cache (lock-free rcu).
    fn bloom_insert(&self, file_id: u64, filter: Arc<bloom::BloomFilter>) {
        self.bloom_cache.rcu(|current| {
            let mut map = (**current).clone();
            if map.len() >= BLOOM_CACHE_MAX { map.clear(); }
            map.insert(file_id, Arc::clone(&filter));
            map
        });
    }

    /// Removes a bloom filter from the cache (lock-free rcu).
    fn bloom_remove(&self, file_id: u64) {
        self.bloom_cache.rcu(|current| {
            let mut map = (**current).clone();
            map.remove(&file_id);
            map
        });
    }

    /// Returns the cached key index for an SST file id (PR 5.0), lazily
    /// reading the sidecar into the cache on first use. A missing/corrupt
    /// sidecar yields `None` and the caller falls back to the Vortex read.
    fn key_index_for(&self, file_id: u64) -> Option<Arc<key_index::KeyIndex>> {
        let snapshot = self.key_index_cache.load();
        if let Some(cached) = snapshot.get(&file_id) {
            return Some(Arc::clone(cached));
        }
        let index = key_index::KeyIndex::read_from(&key_index_path(&self.config.path, file_id))?;
        let index = Arc::new(index);
        self.key_index_cache.rcu(|current| {
            let mut map = (**current).clone();
            map.insert(file_id, Arc::clone(&index));
            map
        });
        Some(index)
    }

    /// Inserts a key index into the cache (lock-free rcu).
    fn key_index_insert(&self, file_id: u64, index: Arc<key_index::KeyIndex>) {
        self.key_index_cache.rcu(|current| {
            let mut map = (**current).clone();
            if map.len() >= KEY_INDEX_CACHE_MAX { map.clear(); }
            map.insert(file_id, Arc::clone(&index));
            map
        });
    }

    /// Removes a key index from the cache (lock-free rcu).
    fn key_index_remove(&self, file_id: u64) {
        self.key_index_cache.rcu(|current| {
            let mut map = (**current).clone();
            map.remove(&file_id);
            map
        });
    }

    /// Inserts a key → file_id mapping into the hot-key cache.  When the
    /// cache exceeds [`HOT_KEY_CACHE_MAX`], the entire cache is cleared
    /// (amortised O(1) eviction).  A stale entry is harmless — the key
    /// index returns `None` for a compacted file and the full search runs.
    fn hot_key_cache_insert(&self, key: &[u8], file_id: u64) {
        let mut cache = self.hot_key_cache.write();
        if cache.len() >= HOT_KEY_CACHE_MAX {
            cache.clear();
        }
        cache.insert(Bytes::copy_from_slice(key), file_id);
    }

    /// Returns the engine stats version.  Bumped on every flush and compaction.
    /// DataFusion uses this as a cache-invalidation signal.
    #[must_use]
    pub fn stats_version(&self) -> u64 {
        self.stats_version.load(Ordering::Acquire)
    }

    /// Pins `snapshot_ts` so GC and compaction never drop an MVCC version
    /// visible at that timestamp. Returns a [`SnapshotGuard`] that releases
    /// the pin on drop.
    ///
    /// Pins are refcounted: pinning the same timestamp N times requires N
    /// drops before the watermark may advance past it.
    ///
    /// Callers that already know their snapshot (e.g. a retry that reuses the
    /// original snapshot) should pin it here. New transactions must use
    /// [`Engine::begin_read`] instead, which reads the snapshot and pins it
    /// atomically so a concurrent compaction cannot advance the watermark past
    /// the snapshot before the pin lands.
    #[must_use]
    pub fn pin_snapshot(self: &Arc<Self>, snapshot_ts: u64) -> SnapshotGuard {
        *self.pinned_snapshots.lock().entry(snapshot_ts).or_insert(0) += 1;
        SnapshotGuard {
            engine: Arc::clone(self),
            ts: snapshot_ts,
        }
    }

    /// Atomically reads the current committed snapshot and pins it, returning
    /// `(snapshot_ts, guard)`.
    ///
    /// The read and the pin happen under the same lock that
    /// [`Engine::watermark`] uses to read the minimum pinned snapshot, which
    /// closes the race where a transaction reads its snapshot, a compaction
    /// computes a watermark from a *later* committed timestamp, and only then
    /// the transaction pins — the watermark could otherwise exceed the
    /// transaction's snapshot and drop versions it still needs.
    #[must_use]
    pub fn begin_read(self: &Arc<Self>) -> (u64, SnapshotGuard) {
        let mut pins = self.pinned_snapshots.lock();
        let snapshot_ts = self.commit_seq.load(Ordering::Acquire).saturating_sub(1);
        *pins.entry(snapshot_ts).or_insert(0) += 1;
        drop(pins);
        (
            snapshot_ts,
            SnapshotGuard {
                engine: Arc::clone(self),
                ts: snapshot_ts,
            },
        )
    }

    /// Releases one pin for `snapshot_ts`. Called by [`SnapshotGuard::drop`].
    fn unpin_snapshot(&self, snapshot_ts: u64) {
        let mut pins = self.pinned_snapshots.lock();
        if let Some(count) = pins.get_mut(&snapshot_ts) {
            *count -= 1;
            if *count == 0 {
                pins.remove(&snapshot_ts);
            }
        }
    }

    /// Returns the oldest snapshot still observable by a live reader.
    ///
    /// This is `min(minimum pinned snapshot, snapshot_ts())`: when no
    /// transaction has pinned a snapshot, it equals [`Engine::snapshot_ts`]
    /// (any future reader starts at or after it); when a transaction holds an
    /// older snapshot, compaction and GC stop at that older timestamp so the
    /// transaction's read set stays intact.
    ///
    /// Both the pinned minimum and the committed timestamp are read under the
    /// pin lock, mirroring [`Engine::begin_read`], so the watermark is always
    /// `<=` any transaction's snapshot.
    #[must_use]
    pub fn watermark(&self) -> u64 {
        let pins = self.pinned_snapshots.lock();
        let min_pinned = pins.keys().next().copied().unwrap_or(u64::MAX);
        let committed = self.commit_seq.load(Ordering::Acquire).saturating_sub(1);
        min_pinned.min(committed)
    }

    /// Pins the current committed snapshot for the duration of a `&self` read
    /// (upgrading the weak self-reference to an [`Arc`] first). Returns `None`
    /// only while the engine is mid-destruction, in which case no compaction
    /// worker can run either, so the read is still consistent.
    ///
    /// The returned guard must be held until the read finishes; dropping it
    /// releases the pin so compaction may reclaim superseded versions again.
    fn pin_read_snapshot(&self) -> Option<(u64, SnapshotGuard)> {
        let arc = self.self_ref.upgrade()?;
        let (ts, guard) = arc.begin_read();
        Some((ts, guard))
    }

    /// Returns the list of SST file IDs currently managed by the manifest.
    #[must_use]
    pub fn sst_file_ids(&self) -> Vec<u64> {
        self.version_set_arc
            .load_full()
            .levels()
            .values()
            .flat_map(|files| files.iter().copied())
            .collect()
    }

    /// Reserves a monotonically increasing transaction id without writing
    /// any data. Callers (SSI transactions) hold it between `begin` and
    /// `commit`, then pass it to [`Engine::write_with_tx`].
    ///
    /// Reserving advances `next_tx_id` so concurrent SSI transactions each
    /// get a unique, ordered id, but it does **not** advance
    /// [`Engine::snapshot_ts`] (which tracks committed progress only). An
    /// id that is reserved but never committed leaves a hole in the id
    /// space — harmless, since MVCC only requires monotonic ids, not dense
    /// ones.
    #[must_use]
    pub fn reserve_tx_id(&self) -> u64 {
        self.alloc_tx_id()
    }

    /// Allocates a non-zero tx_id. `0` is a reserved sentinel: the compare-exchange forward loop
    /// ensures that even a theoretical wrap to 0 jumps to 1 instead of returning an id that
    /// [`Engine::write_with_tx`] would reject (R-21 thorough, PR 0.10).
    fn alloc_tx_id(&self) -> u64 {
        let mut current = self.next_tx_id.load(Ordering::Acquire);
        loop {
            // 0 is a reserved sentinel; if it wraps to 0, restart from 1 (u64::MAX allocations is an
            // astronomical number; this branch exists only for theoretical completeness).
            let start = if current == 0 { 1 } else { current };
            let next = start.wrapping_add(1);
            match self.next_tx_id.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return start,
                Err(observed) => current = observed,
            }
        }
    }

    /// Drops a table and removes all its data from the engine.
    ///
    /// This method:
    /// 1. Removes all matching keys from the memtable immediately via
    ///    [`MemTable::erase_prefix`]
    /// 2. Records the table prefix so future scans filter out orphaned
    ///    SST data (lazy cleanup)
    ///
    /// # SST Cleanup
    ///
    /// SST data is not removed immediately — it is filtered lazily by
    /// [`scan_table`](Self::scan_table) and [`get`](Self::get) until the
    /// next compaction pass.  This avoids a full SST rewrite on every
    /// `DROP TABLE`.
    ///
    /// # Returns
    ///
    /// Returns the number of keys removed from the memtable.
    ///
    /// # Re-creating a dropped table
    ///
    /// Recording the prefix here makes `get` / `scan_table` filter **all**
    /// keys under `table\0`, including writes made to a table re-created
    /// with the same name after this call. Callers that re-create the table
    /// must clear the mark first via [`Engine::undrop_table_prefix`];
    /// otherwise the re-created table's data is invisible until the engine
    /// restarts (the `dropped_tables` set is in-memory only).
    pub fn drop_table_prefix(&self, table: &str) -> Result<usize> {
        let prefix = table_prefix(table);
        {
            let _guard = self.compaction_guard.lock();
            self.manifest.lock().append_batch(vec![VersionEdit::DroppedPrefix(prefix.clone())])?;
            self.dropped_tables.rcu(|current| {
                let mut set = (**current).clone();
                set.insert(prefix.clone());
                set
            });
        }
        let removed_count = {
            let _guard = self.write_guard.lock();
            self.memtable.erase_prefix(&prefix)
        };

        tracing::info!(
            table = %table,
            memtable_keys_removed = removed_count,
            "table dropped; SST data filtered lazily until next compaction",
        );

        Ok(removed_count)
    }

    /// Clears the dropped-table mark for `table`, so a table re-created with
    /// the same name is visible again.
    ///
    /// This is the symmetric counterpart to [`Engine::drop_table_prefix`].
    /// The clear is persisted to the manifest, so a restart does not resurrect
    /// the drop filter for a re-created table. Orphaned SST data from the
    /// previous incarnation is still filtered by compaction later, but new
    /// writes become visible immediately.
    ///
    /// # Errors
    ///
    /// Returns an [`EngineError`] if the manifest append fails.
    pub fn undrop_table_prefix(&self, table: &str) -> Result<()> {
        let _guard = self.compaction_guard.lock();
        let prefix = table_prefix(table);
        self.manifest
            .lock()
            .append_batch(vec![VersionEdit::UndroppedPrefix(prefix.clone())])?;
        self.dropped_tables.rcu(|current| {
            let mut set = (**current).clone();
            set.remove(&prefix);
            set
        });
        Ok(())
    }

    /// Returns `true` if `key` belongs to a dropped table.
    ///
    /// Keys are `table\0 + pk` and [`dropped_tables`](Self::dropped_tables)
    /// stores each dropped table's `table\0` prefix, so we extract the table
    /// prefix from `key` and do an O(log n) set lookup instead of a linear
    /// scan over every dropped prefix.
    fn is_dropped_prefix(&self, key: &[u8]) -> bool {
        let prefix = key_table_prefix(key);
        self.dropped_tables.load_full().contains(prefix)
    }

    /// Atomically commits a non-empty batch and returns its commit sequence
    /// number (`commit_ts`). The commit sequence is strictly increasing in
    /// **commit order**.
    ///
    /// After successful commit the engine's [`EngineHook`] is invoked.
    pub fn write(&self, mutations: &[Mutation]) -> Result<u64> {
        if mutations.is_empty() {
            return Err(EngineError::EmptyBatch);
        }
        let tx_id = self.alloc_tx_id();
        self.write_with_tx(tx_id, mutations)
    }

    /// Commits `mutations` under an explicit `tx_id`.
    ///
    /// Callers that pre-reserved a transaction id (e.g. via
    /// `hatp_tx::TxManager::begin_ssi`) must use this entry point instead of
    /// [`Engine::write`] so the reserved id and the engine-allocated id stay
    /// in sync. `tx_id == 0` is rejected as a sentinel.
    ///
    /// The pre-commit hook ([`EngineHook::on_pre_commit`]) runs **before** any
    /// WAL/memtable/SST side-effect. A failing pre-commit aborts the batch
    /// without touching durable storage.
    ///
    /// Returns the commit sequence number (`commit_ts`) on success — NOT the
    /// `tx_id`. `commit_ts` is the strictly increasing commit-order sequence;
    /// callers (notably `hatp::Transaction::commit`) can use it as a snapshot
    /// read point even though they reserved the `tx_id` externally.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::EmptyBatch`] if `mutations` is empty,
    /// [`EngineError::Corrupt`] if `tx_id == 0`, the pre-commit error
    /// (e.g. [`EngineError::WriteConflict`]), or any underlying I/O error.
    pub fn write_with_tx(&self, tx_id: u64, mutations: &[Mutation]) -> Result<u64> {
        if mutations.is_empty() {
            return Err(EngineError::EmptyBatch);
        }
        if tx_id == 0 {
            return Err(EngineError::Corrupt("tx_id 0 is reserved as sentinel"));
        }
        // Reserve the externally-supplied `tx_id` so `snapshot_ts()` covers
        // it. We compare-and-swap the counter up to `tx_id + 1` but only
        // when strictly greater than the current value — using a raw
        // `fetch_max` here would stick the counter at the max of *every*
        // reserved id ever seen, leaving holes smaller than the current
        // counter unreachable (and making `Engine::write` skip ids when
        // its `fetch_add(1)` lands on a hole). The compare-exchange loop
        // is monotonic, lock-free, and safe under `write_guard`.
        let mut current = self.next_tx_id.load(Ordering::Acquire);
        let target = tx_id.saturating_add(1);
        while target > current {
            match self.next_tx_id.compare_exchange_weak(
                current,
                target,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
        // SILK backpressure (R-17): consult the throttle before the write guard
        // is taken, so a throttled write never blocks a healthy one. Soft
        // throttles sleep briefly to give flush time to drain; Hard rejects
        // the write outright so the caller can flush and retry.
        match self.throttle.should_throttle(&self.metrics.snapshot()) {
            pressure::ThrottleLevel::None => {}
            pressure::ThrottleLevel::Soft => std::thread::sleep(self.throttle.soft_sleep()),
            pressure::ThrottleLevel::Hard => return Err(EngineError::Throttled),
        }
        self.write_with_tx_sync(tx_id, mutations)
    }

    /// The synchronous commit path: group-commit aware.
    ///
    /// 1. Push this request into `pending_writes`.
    /// 2. Acquire `write_guard`.
    /// 3. If another thread already handled our request (detected by our
    ///    `result_tx` being consumed), return immediately.
    /// 4. Otherwise, drain every pending request, pre-commit each, write
    ///    one WAL batch + one `fsync` for all of them, apply to memtable, and
    ///    notify every waiter via `result_tx`.
    fn write_with_tx_sync(&self, tx_id: u64, mutations: &[Mutation]) -> Result<u64> {
        // Fast path: when the pending queue is empty, bypass the channel
        // allocation and group-commit machinery entirely.  Single-writer
        // workloads (the common case in embedded OLTP) save ~100ns per tx.
        {
            let pending = self.pending_writes.lock();
            if pending.is_empty() {
                drop(pending);
                let _guard = self.write_guard.lock();
                let pending = self.pending_writes.lock();
                if pending.is_empty() {
                    drop(pending);
                    return self.commit_single(tx_id, mutations);
                }
                // Another writer pushed while we waited — fall through.
            }
        }

        let (result_tx, result_rx) = std::sync::mpsc::sync_channel(1);
        self.pending_writes.lock().push(PendingWrite {
            tx_id,
            mutations: mutations.to_vec(),
            result_tx,
        });

        let _guard = self.write_guard.lock();

        // Another thread may have already drained our request while we were
        // waiting for the guard.  If so, the result is already in the channel.
        if let Ok(result) = result_rx.try_recv() {
            return result;
        }

        // Drain the pending queue.
        let batch: Vec<PendingWrite> = {
            let mut pending = self.pending_writes.lock();
            std::mem::take(&mut *pending)
        };
        if batch.is_empty() {
            // Should not happen: we just pushed our own request.  If it does
            // (another thread drained between our push and the guard), fall
            // through to result_rx.
            return result_rx
                .recv()
                .unwrap_or(Err(EngineError::Corrupt("group commit channel closed")));
        }

        if let Err(e) = self.flush_pending_batch(batch) {
            // WAL failure: every transaction that passed pre-commit has already
            // been notified via result_tx.  Propagate the error to our caller.
            let _ = result_rx.recv();
            return Err(e);
        }
        // Always read our result from the channel — flush_pending_batch
        // sends every outcome (success or pre-commit failure) there.
        result_rx
            .recv()
            .unwrap_or(Err(EngineError::Corrupt("group commit channel closed")))
    }

    /// Fast-path commit for a single transaction (no channel, no group-commit).
    /// Caller must hold `write_guard`.  Skips `catch_unwind` for NoopHook (R1).
    fn commit_single(&self, tx_id: u64, mutations: &[Mutation]) -> Result<u64> {
        if self.hook_is_noop {
            let _ = self.hook.on_pre_commit(tx_id, mutations);
        } else {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.hook.on_pre_commit(tx_id, mutations)
            })) {
                Ok(Ok(())) => {}
                Ok(Err(err)) => {
                    self.metrics.record_abort();
                    if matches!(err, EngineError::WriteConflict { .. } | EngineError::ReadWriteConflict { .. }) {
                        self.metrics.record_ssi_conflict();
                    }
                    return Err(err);
                }
                Err(_) => return Err(EngineError::CorruptMessage("on_pre_commit panicked".to_string())),
            }
        }
        let wal_bytes = match self.config.sync_mode {
            SyncMode::Full => self.wal.append_mutations_commit_and_sync(tx_id, mutations),
            SyncMode::Group { .. } => {
                let records = build_wal_records(tx_id, mutations);
                Ok(self.wal.append_multi_batch(&[(tx_id, records.as_slice())])?)
            }
        };
        let wal_bytes = wal_bytes?;
        self.metrics.record_wal_write(wal_bytes);
        let commit_ts = self.commit_seq.fetch_add(1, Ordering::AcqRel);
        self.memtable.apply_mutations_batch(tx_id, mutations, commit_ts)?;
        if self.hook_is_noop {
            for m in mutations {
                match m {
                    Mutation::Put { key, value } => self.hook.on_put(key, value, tx_id),
                    Mutation::Delete { key } => self.hook.on_delete(key, tx_id),
                }
            }
            self.hook.on_tx_commit(tx_id, commit_ts);
        } else {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                for m in mutations {
                    match m {
                        Mutation::Put { key, value } => self.hook.on_put(key, value, tx_id),
                        Mutation::Delete { key } => self.hook.on_delete(key, tx_id),
                    }
                }
                self.hook.on_tx_commit(tx_id, commit_ts);
            }));
        }
        self.metrics.record_commit();
        if self.watcher.has_subscribers() {
            self.watcher.publish(tx_id, commit_ts, Arc::from(mutations.to_vec()));
        } else {
            self.watcher.publish_empty(tx_id, commit_ts);
        }
        self.metrics.set_memtable_bytes(self.memtable.approximate_bytes() as u64);
        Ok(commit_ts)
    }

    /// Commits a batch of pending writes under `write_guard` (caller must hold
    /// the lock).  One WAL `fsync` for the entire batch.
    fn flush_pending_batch(&self, mut batch: Vec<PendingWrite>) -> Result<()> {
        // ── 1. Pre-commit hook for each transaction ────────────────────────
        // Transactions whose pre-commit hook fails are pushed to the back
        // (after the successful ones) so the caller can retry them later.
        // We only keep the ones that passed.
        let mut passed = 0_usize;
        for i in 0..batch.len() {
            // SAFETY: `i` is bounded by `batch.len()`.
            #[allow(clippy::indexing_slicing)]
            {
            let req = &batch[i];
            let pre_commit_result = if self.hook_is_noop {
                self.hook.on_pre_commit(req.tx_id, &req.mutations)
            } else {
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.hook.on_pre_commit(req.tx_id, &req.mutations)
                })) {
                    Ok(r) => r,
                    Err(_) => Err(EngineError::CorruptMessage(
                        "on_pre_commit panicked".to_string(),
                    )),
                }
            };
            match pre_commit_result {
                Ok(()) => {
                    batch.swap(passed, i);
                    passed += 1;
                }
                Err(err) => {
                    self.metrics.record_abort();
                    if matches!(
                        err,
                        EngineError::WriteConflict { .. } | EngineError::ReadWriteConflict { .. }
                    ) {
                        self.metrics.record_ssi_conflict();
                    }
                    let _ = batch[i].result_tx.send(Err(err));
                }
            }
            }
        }
        batch.truncate(passed);
        if batch.is_empty() {
            return Ok(());
        }

        // ── 2. Build WAL refs from mutations directly (no WalRecord alloc) ──
        let wal_mutation_refs: Vec<(u64, &[Mutation])> = batch
            .iter()
            .map(|req| (req.tx_id, req.mutations.as_slice()))
            .collect();

        // ── 3. Single WAL append (fsync depends on sync_mode) ──────────
        let wal_bytes = match self.config.sync_mode {
            SyncMode::Full => self.wal.append_multi_mutations_commit_and_sync(&wal_mutation_refs),
            SyncMode::Group { .. } => self.wal.append_multi_mutations(&wal_mutation_refs),
        };
        let wal_bytes = match wal_bytes {
            Ok(bytes) => bytes,
            Err(e) => {
                let msg = format!("WAL append failed: {e}");
                for req in &batch {
                    let _ = req
                        .result_tx
                        .send(Err(EngineError::CorruptMessage(msg.clone())));
                }
                return Err(e);
            }
        };
        self.metrics.record_wal_write(wal_bytes);

        // ── 4. Allocate commit_ts, apply memtable, notify waiters ─────────
        let mut batch_iter = batch.into_iter();
        for req in batch_iter.by_ref() {
            let tx_id = req.tx_id;
            let commit_ts = self.commit_seq.fetch_add(1, Ordering::AcqRel);
            if let Err(e) = self.memtable.apply_mutations_batch(tx_id, &req.mutations, commit_ts) {
                let _ = req.result_tx.send(Err(EngineError::CorruptMessage(
                    format!("memtable apply failed: {e}"),
                )));
                for remaining_req in batch_iter {
                    let _ = remaining_req.result_tx.send(Err(
                        EngineError::CorruptMessage(format!(
                            "memtable apply failed for preceding transaction {tx_id}: {e}"
                        )),
                    ));
                }
                return Err(e);
            }
            if self.hook_is_noop {
                for mutation in &req.mutations {
                    match mutation {
                        Mutation::Put { key, value } => self.hook.on_put(key, value, tx_id),
                        Mutation::Delete { key } => self.hook.on_delete(key, tx_id),
                    }
                }
                self.hook.on_tx_commit(tx_id, commit_ts);
            } else {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    for mutation in &req.mutations {
                        match mutation {
                            Mutation::Put { key, value } => self.hook.on_put(key, value, tx_id),
                            Mutation::Delete { key } => self.hook.on_delete(key, tx_id),
                        }
                    }
                    self.hook.on_tx_commit(tx_id, commit_ts);
                }));
            }
            self.metrics.record_commit();
            let _ = req.result_tx.send(Ok(commit_ts));
            if self.watcher.has_subscribers() {
                self.watcher
                    .publish(tx_id, commit_ts, Arc::from(req.mutations.clone()));
            } else {
                self.watcher.publish_empty(tx_id, commit_ts);
            }
        }
        self.metrics
            .set_memtable_bytes(self.memtable.approximate_bytes() as u64);
        Ok(())
    }

    /// Reads the newest visible, non-deleted value at `snapshot_ts`.
    ///
    /// MVCC safety: the read consults the memtable first, then any
    /// SST files registered in the manifest. Tombstones filter the
    /// row before it is returned.
    ///
    /// A key under a dropped table prefix returns `None`:
    /// [`Engine::drop_table_prefix`] erases the memtable immediately but
    /// leaves SST data for lazy compaction, so the point read must apply
    /// the same prefix filter as [`Engine::scan_table`].
    pub fn get(&self, key: &[u8], snapshot_ts: u64) -> Result<Option<Bytes>> {
        if self.is_dropped_prefix(key) {
            return Ok(None);
        }
        let snapshot = Snapshot::new(snapshot_ts);
        if let Some(version) = self.memtable.get(key, snapshot) {
            return Ok(version.value);
        }
        // Hot-key cache: O(1) route to the SST file that last contained this
        // key.  A stale entry (file compacted away) is harmless — the key
        // index returns `None` and we fall through to the full search.
        if let Some(&cached_file_id) = self.hot_key_cache.read().get(key) {
            if let Some(idx) = self.key_index_for(cached_file_id) {
                if let Some(version) = idx.get(key) {
                    if snapshot.sees(&version) {
                        return Ok(version.value);
                    }
                }
            }
        }
        // R4: lock-free version-set read. Writers go through the manifest
        // mutex and atomically swap the inner `Arc<VersionSet>`; we observe
        // the swap via `load_full` without any mutex acquisition.
        let version_set = self.version_set_arc.load_full();
        // PR 5.2: O(log n) candidate file selection via binary search over
        // the sorted key-range index, then iterate only the candidate files.
        let candidates = version_set.candidate_files(key);
        // PR 5.2: file_id → level lookup via O(log n) binary search over the
        // pre-computed `file_level` vector — no per-call HashMap allocation.
        let mut newest: Option<version::VersionedValue> = None;
        let mut found_in_leveled = false;
        for file_id in candidates {
            let level = version_set.file_level_for(file_id).unwrap_or(0);
            // L1+ early exit: if we already found the key in a leveled file,
            // no other L1+ file can contain a different version.
            if found_in_leveled && level > 0 {
                continue;
            }
            // Key-range pruning: skip SSTs whose [min_key, max_key] cannot
            // contain `key`. `min_key` / `max_key` are recorded by flush and
            // compaction from the sorted row order; an empty bound means the
            // metadata is absent (legacy manifest) and the file must be read.
            if let Some((min_key, max_key, _, _)) = version_set.file_metadata(file_id) {
                if !min_key.is_empty()
                    && !max_key.is_empty()
                    && (min_key.as_ref() > key || max_key.as_ref() < key)
                {
                    continue;
                }
            }
            // Bloom pre-filter (PR 3.10): a negative result proves the key is
            // absent from this SST, so the point lookup is skipped entirely.
            if let Some(filter) = self.bloom_for(file_id) {
                if !filter.contains(key) {
                    continue;
                }
            }
            // PR 5.0: key-index fast path — O(log n) binary search, no
            // Vortex file open. A missing/corrupt key index falls through
            // to the authoritative Vortex read below.
            if let Some(idx) = self.key_index_for(file_id) {
                if let Some(version) = idx.get(key) {
                    if snapshot.sees(&version)
                        && newest
                            .as_ref()
                            .map(|current| version.begin_ts > current.begin_ts)
                            .unwrap_or(true)
                    {
                        newest = Some(version);
                        // Populate hot-key cache for next lookup.
                        self.hot_key_cache_insert(key, file_id);
                    }
                    // Key found in index (definitive).  Mark leveled exit.
                    if level > 0 {
                        found_in_leveled = true;
                    }
                    continue;
                }
                // Key not in index (definitive negative) — skip Vortex.
                continue;
            }
            // Fallback: authoritative Vortex read (legacy SST without key index).
            let path = sst_path(&self.config.path, file_id);
            if let Some(version) =
                self.config
                    .sst_format
                    .get(&path, key, snapshot)?
            {
                if newest
                    .as_ref()
                    .map(|current| version.begin_ts > current.begin_ts)
                    .unwrap_or(true)
                {
                    newest = Some(version);
                }
                // Mark leveled exit so we skip remaining L1+ files.
                if level > 0 {
                    found_in_leveled = true;
                }
            }
        }
        Ok(newest.and_then(|version| version.value))
    }

    /// Returns whether the configured memtable flush threshold is reached.
    #[must_use]
    pub fn should_flush(&self) -> bool {
        self.memtable.approximate_bytes() >= self.config.memtable_flush_bytes
    }

    /// Compacts all files from `source_level` together with the current target
    /// level into one immutable file. Manifest edits make the output visible
    /// before retiring inputs, so crashes cannot lose committed versions.
    pub fn compact(
        &self,
        source_level: u32,
        target_level: u32,
    ) -> Result<Option<sst_format::SstHandle>> {
        self.compact_with_watermark(source_level, target_level, self.watermark())
    }

    /// Compaction core with an explicit GC watermark.
    ///
    /// The watermark is the oldest snapshot any live reader can still observe
    /// ([`Engine::watermark`]). Only superseded versions whose visibility
    /// interval is entirely below the watermark are dropped, so a pinned
    /// transaction snapshot keeps its historical versions intact.
    fn compact_with_watermark(
        &self,
        source_level: u32,
        target_level: u32,
        watermark: u64,
    ) -> Result<Option<sst_format::SstHandle>> {
        let _guard = self.compaction_guard.lock();
        let inputs = self.compaction_inputs(source_level, target_level);
        if inputs.len() < 2 {
            return Ok(None);
        }
        let file_id = self.manifest.lock().version_set().next_file_id();
        let (handle, deleted_rows, prefix_delta) =
            self.compact_chunk(&inputs, watermark, file_id)?;
        self.commit_compaction(
            &inputs,
            source_level,
            target_level,
            std::slice::from_ref(&handle),
            deleted_rows,
            prefix_delta,
        )?;
        Ok(Some(handle))
    }

    /// Parallel sub-compaction (PR 2.6): splits `inputs` by size and merges each
    /// chunk on a rayon thread, then commits one atomic manifest batch. The
    /// `write_guard` is reentrant, so `compact_chunk`'s pure file operations run
    /// in parallel while the manifest stays serialised behind one commit.
    pub fn sub_compact(
        &self,
        source_level: u32,
        target_level: u32,
        watermark: u64,
    ) -> Result<Option<Vec<sst_format::SstHandle>>> {
        let _guard = self.compaction_guard.lock();
        let inputs = self.compaction_inputs(source_level, target_level);
        if inputs.len() < 2 {
            return Ok(None);
        }
        let (chunks, base_file_id) = {
            let version_set = self.manifest.lock().version_set();
            let chunks = split_by_size(&inputs, &version_set, 64 * 1024 * 1024);
            (chunks, version_set.next_file_id())
        };

        // Data-parallel merge of each chunk (pure file I/O, no shared locks).
        use rayon::prelude::*;
        let results: Vec<Result<ChunkOutput>> = chunks
            .par_iter()
            .enumerate()
            .map(|(index, chunk)| {
                let file_id = base_file_id.saturating_add(index as u64);
                self.compact_chunk(chunk, watermark, file_id)
            })
            .collect();

        let mut handles = Vec::with_capacity(results.len());
        let mut deleted_rows = 0_usize;
        let mut prefix_delta: BTreeMap<Vec<u8>, isize> = BTreeMap::new();
        for result in results {
            let (handle, deleted, delta) = result?;
            deleted_rows = deleted_rows.saturating_add(deleted);
            for (prefix, change) in delta {
                let entry = prefix_delta.entry(prefix).or_insert(0);
                *entry = (*entry).saturating_add(change);
            }
            handles.push(handle);
        }
        self.commit_compaction(
            &inputs,
            source_level,
            target_level,
            &handles,
            deleted_rows,
            prefix_delta,
        )?;
        Ok(Some(handles))
    }

    /// Collects the file ids from `source_level` + `target_level`, sorted and deduped.
    fn compaction_inputs(&self, source_level: u32, target_level: u32) -> Vec<u64> {
        let version_set = self.manifest.lock().version_set();
        let mut inputs = version_set.files(source_level);
        inputs.extend(version_set.files(target_level));
        inputs.sort_unstable();
        inputs.dedup();
        inputs
    }

    /// Reads `chunk` inputs, drops superseded versions, and writes one output
    /// SST. Returns the handle and the number of input rows consumed (for the
    /// row-count delta).
    fn compact_chunk(&self, chunk: &[u64], watermark: u64, file_id: u64) -> Result<ChunkOutput> {
        let mut rows: Vec<memtable::VersionedRow> = Vec::new();
        let mut deleted_rows = 0_usize;
        for input in chunk {
            let path = sst_path(&self.config.path, *input);
            fault_point!("compact::read_input");
            let file_rows = self.config
                    .sst_format
                    .read_all(&path)?;
            deleted_rows = deleted_rows.saturating_add(file_rows.len());
            rows.extend(file_rows);
        }
        // Per-prefix input counts (before merge), used to update per-table row counts.
        let input_counts = count_rows_by_prefix(&rows);
        let rows = dedup_superseded(rows, watermark);
        let output_counts = count_rows_by_prefix(&rows);
        let prefix_delta = prefix_count_delta(&input_counts, &output_counts);
        let path = sst_path(&self.config.path, file_id);
        fault_point!("compact::write_output_before");
        let handle = self.config.sst_format.write(
                    &path,
                    &rows,
                    file_id,
                    // PR 5.3: pass table schema for columnar SST.
                    rows.first().and_then(|row| {
                        let prefix = key_table_prefix(&row.key);
                        let table_name = String::from_utf8_lossy(
                            prefix.strip_suffix(&[0]).unwrap_or(prefix),
                        ).into_owned();
                        self.table_schemas.read().get(&table_name).cloned()
                    }).as_deref(),
                )?;
        fault_point!("compact::write_output_after");
        if let Some(filter) = write_bloom_for(&self.config.path, file_id, &rows) {
            self.bloom_insert(file_id, filter);
        }
        // PR 5.0: write key-index sidecar for compacted SST.
        let kidx = key_index::KeyIndex::from_rows(&rows);
        if let Err(error) = kidx.write_to(&key_index_path(&self.config.path, file_id)) {
            tracing::warn!(file_id, error = %error, "key index write failed during compaction");
        } else {
            self.key_index_insert(file_id, Arc::new(kidx));
        }
        self.compute_column_stats(file_id, &rows);
        Ok((handle, deleted_rows, prefix_delta))
    }

    /// Computes per-table column statistics for a freshly written SST by
    /// decoding each row's compact payload against its table's Arrow schema.
    /// Called under the write guard (flush / compaction); the result is cached
    /// in `self.column_stats` and read by [`Engine::table_zone_maps`] to feed
    /// DataFusion's cost-based optimizer.
    fn compute_column_stats(&self, file_id: u64, rows: &[memtable::VersionedRow]) {
        let mut by_table: BTreeMap<String, Vec<&memtable::VersionedRow>> = BTreeMap::new();
        for row in rows {
            let prefix = key_table_prefix(&row.key);
            // `table_prefix` (constructed by `assemble_table_key`) always ends with `\0`;
            // stripping the trailing `\0` yields the table name; bare keys without a trailing `\0` are excluded from column stats.
            let Some(name) = prefix.strip_suffix(&[0]) else {
                continue;
            };
            by_table
                .entry(String::from_utf8_lossy(name).into_owned())
                .or_default()
                .push(row);
        }

        let mut file_stats: BTreeMap<String, TableColumnStats> = BTreeMap::new();
        for (table, table_rows) in by_table {
            let Some(schema) = self.table_schemas.read().get(&table).cloned() else {
                continue;
            };
            let mut stats = TableColumnStats::default();
            for row in table_rows {
                // Tombstones have no column values; skip.
                let Some(payload) = row.version.value.as_ref() else {
                    continue;
                };
                let Ok(values) =
                    crate::row_codec::decode_row_values(payload, schema.as_ref())
                else {
                    continue;
                };
                for (field, scalar) in schema.fields().iter().zip(values.iter()) {
                    let name = field.name().clone();
                    if scalar.is_null() {
                        *stats.null_count.entry(name.clone()).or_insert(0) += 1;
                        continue;
                    }
                    match stats.min.get(&name) {
                        None => {
                            stats.min.insert(name.clone(), scalar.clone());
                        }
                        Some(prev) if scalar < prev => {
                            stats.min.insert(name.clone(), scalar.clone());
                        }
                        Some(_) => {}
                    }
                    match stats.max.get(&name) {
                        None => {
                            stats.max.insert(name.clone(), scalar.clone());
                        }
                        Some(prev) if scalar > prev => {
                            stats.max.insert(name.clone(), scalar.clone());
                        }
                        Some(_) => {}
                    }
                }
            }
            file_stats.insert(table, stats);
        }

        if !file_stats.is_empty() {
            let mut s = self.column_stats.write();
            if s.len() >= COLUMN_STATS_CACHE_MAX { s.clear(); }
            s.insert(file_id, file_stats);
        }
    }

    /// Commits the compaction result to the manifest, retires the inputs, and
    /// updates the cached row count. One atomic manifest batch covers the whole
    /// compaction, so a crash never exposes a half-applied result.
    fn commit_compaction(
        &self,
        inputs: &[u64],
        source_level: u32,
        target_level: u32,
        handles: &[sst_format::SstHandle],
        deleted_rows: usize,
        prefix_delta: BTreeMap<Vec<u8>, isize>,
    ) -> Result<()> {
        let mut manifest = self.manifest.lock();
        let snapshot = manifest.version_set();
        let new_rows: usize = handles.iter().map(|handle| handle.rows).sum();
        let mut edits: Vec<VersionEdit> =
            Vec::with_capacity(handles.len().saturating_add(inputs.len()).saturating_add(1));
        for handle in handles {
            edits.push(VersionEdit::AddFile {
                file_id: handle.file_id,
                level: target_level,
                min_key: handle.min_key.clone(),
                max_key: handle.max_key.clone(),
                bytes: handle.bytes,
                created_at: self.config.clock.now_secs(),
            });
        }
        let max_file_id = handles
            .iter()
            .map(|handle| handle.file_id)
            .max()
            .unwrap_or(0);
        edits.push(VersionEdit::NextFileId(max_file_id.saturating_add(1)));
        for input in inputs {
            let level = if snapshot.files(source_level).contains(input) {
                source_level
            } else {
                target_level
            };
            edits.push(VersionEdit::DeleteFile {
                file_id: *input,
                level,
            });
        }
        fault_point!("compact::manifest_commit_before");
        manifest.append_batch(edits)?;
        fault_point!("compact::manifest_commit_after");
        drop(manifest);
        // Update SST row count incrementally: subtract deleted rows, add new rows.
        self.sst_row_count
            .fetch_sub(deleted_rows, Ordering::Relaxed);
        self.sst_row_count.fetch_add(new_rows, Ordering::Relaxed);
        // Per-table row count delta（PR 2.9）。
        if !prefix_delta.is_empty() {
            let mut table_rows = self.table_rows.write();
            for (prefix, change) in prefix_delta {
                if change > 0 {
                    let entry = table_rows.entry(prefix).or_insert(0);
                    *entry = (*entry).saturating_add(change as usize);
                } else if let Some(entry) = table_rows.get_mut(&prefix) {
                    *entry = entry.saturating_sub(change.unsigned_abs());
                    if *entry == 0 {
                        table_rows.remove(&prefix);
                    }
                }
            }
        }
        // Inputs are unlinked only after the manifest durably commits the
        // new level. A torn OS link just wastes one old file.
        for input in inputs {
            let path = sst_path(&self.config.path, *input);
            match fs::remove_file(path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
            // Retire the input's bloom sidecar alongside its SST (PR 3.10).
            // Missing sidecars are tolerated (NotFound) — the filter is derived.
            let bloom_file = bloom_path(&self.config.path, *input);
            if let Err(error) = fs::remove_file(bloom_file) {
                if error.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!(file_id = input, error = %error, "failed to remove stale bloom sidecar");
                }
            }
            // Retire the input's key-index sidecar alongside its SST (PR 5.0).
            let kidx_file = key_index_path(&self.config.path, *input);
            if let Err(error) = fs::remove_file(kidx_file) {
                if error.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!(file_id = input, error = %error, "failed to remove stale key index sidecar");
                }
            }
            // Drop the retired SST's cached filter so a later lookup does not
            // serve a stale bloom for a file id that has been reallocated.
            self.bloom_remove(*input);
            // Drop the retired SST's cached key index alongside the bloom.
            self.key_index_remove(*input);
            // Retire the input's cached column stats alongside the SST.
            self.column_stats.write().remove(input);
        }
        self.metrics.record_compaction();
        self.stats_version.fetch_add(1, Ordering::Release);
        Ok(())
    }

    /// Writes a consistent memtable snapshot into an immutable SST and records
    /// it in the manifest.  Split into two phases so SST writes never block
    /// incoming OLTP commits.
    pub fn flush(&self) -> Result<Option<sst_format::SstHandle>> {
        let rows = {
            let _guard = self.write_guard.lock();
            let mut rows = Vec::new();
            self.memtable.drain_into(&mut rows);
            if rows.is_empty() { return Ok(None); }
            self.memtable.clear();
            rows
        };
        let _guard = self.compaction_guard.lock();
        self.flush_sst_locked(&rows)
    }

    /// Core of [`Self::flush`]; caller must hold [`Self::compaction_guard`].
    fn flush_sst_locked(&self, rows: &[memtable::VersionedRow]) -> Result<Option<sst_format::SstHandle>> {
        if rows.is_empty() {
            return Ok(None);
        }
        let mut manifest = self.manifest.lock();
        let file_id = manifest.version_set().next_file_id();
        let path = sst_path(&self.config.path, file_id);
        fault_point!("flush::sst_write_before");
        let handle = self.config.sst_format.write(
            &path,
            &rows,
            file_id,
            // PR 5.3: pass the table schema so Vortex can write business columns.
            // Look up the schema from the first row's key prefix.
            rows.first().and_then(|row| {
                let prefix = key_table_prefix(&row.key);
                let table_name = String::from_utf8_lossy(
                    prefix.strip_suffix(&[0]).unwrap_or(prefix),
                ).into_owned();
                self.table_schemas.read().get(&table_name).cloned()
            }).as_deref(),
        )?;
        fault_point!("flush::sst_write_after");
        if let Some(filter) = write_bloom_for(&self.config.path, file_id, &rows) {
            self.bloom_insert(file_id, filter);
        }
        // PR 5.0: write key-index sidecar for O(log n) point lookups.
        // Best-effort — a failed write is logged and the authoritative Vortex
        // read path still works.
        let key_index = key_index::KeyIndex::from_rows(&rows);
        if let Err(error) = key_index.write_to(&key_index_path(&self.config.path, file_id)) {
            tracing::warn!(file_id, error = %error, "key index write failed; falling back to Vortex reads");
        } else {
            self.key_index_insert(file_id, Arc::new(key_index));
        }
        self.compute_column_stats(file_id, &rows);
        fault_point!("flush::manifest_append_before");
        let created_at = self.config.clock.now_secs();
        manifest.append_batch(vec![
            VersionEdit::AddFile {
                file_id,
                level: 0,
                min_key: handle.min_key.clone(),
                max_key: handle.max_key.clone(),
                bytes: handle.bytes,
                created_at,
            },
            VersionEdit::NextFileId(file_id.saturating_add(1)),
        ])?;
        fault_point!("flush::manifest_append_after");
        // Increment SST row count incrementally to avoid re-scanning all SSTs on each scan_stats call.
        self.sst_row_count.fetch_add(rows.len(), Ordering::Relaxed);
        // Per-table row count (PR 2.9 statistics pushdown): group the flushed
        // rows by `table\0` prefix so `table_row_count` can report a per-table
        // number instead of the global one.
        {
            let mut table_rows = self.table_rows.write();
            for row in rows {
                let entry = table_rows
                    .entry(key_table_prefix(&row.key).to_vec())
                    .or_insert(0);
                *entry = (*entry).saturating_add(1);
            }
        }
        self.memtable.clear();

        // WAL auto-truncate: if WAL has grown beyond the threshold, reset it.
        let wal_size = self.wal.current_size();
        if wal_size > self.config.wal_max_size_bytes {
            tracing::debug!(
                wal_size_bytes = wal_size,
                threshold_bytes = self.config.wal_max_size_bytes,
                "WAL exceeds size threshold, truncating"
            );
            self.wal.reset()?;
        }

        // Record flush metrics.
        self.metrics.record_flush();
        self.metrics.record_sst_write(handle.bytes);
        self.stats_version.fetch_add(1, Ordering::Release);

        Ok(Some(handle))
    }

    ///
    /// The memtable count covers every version currently in memory;
    /// the SST count is the sum of rows in every level. Tombstones and
    /// non-visible versions are included — this is a physical count,
    /// not a logical (MVCC-filtered) one.
    ///
    /// # Performance Note
    ///
    /// The SST row count is cached incrementally via `sst_row_count`, updated
    /// on every `flush` (increment) and `compact` (decrement deleted, increment new).
    /// This makes `scan_stats` O(1) instead of O(total_SST_bytes) for each call.
    /// The memtable count still requires a full scan — callers should cache
    /// the result if called frequently.
    #[must_use]
    pub fn scan_stats(&self) -> Option<ScanStats> {
        // Memtable scan: still O(n) on memtable versions.
        // This could also be cached if needed.
        let memtable_rows = self.memtable.all_versions().len();
        // SST row count: O(1) from cached value.
        let sst_rows = self.sst_row_count.load(Ordering::Relaxed);
        Some(ScanStats {
            memtable_rows,
            sst_rows,
        })
    }

    /// Returns the total on-disk bytes across all SST files, aggregated from
    /// the manifest's per-file metadata (no file reads). Used by DataFusion
    /// statistics pushdown for memory/IO cost estimation (PR 2.9).
    #[must_use]
    pub fn total_sst_bytes(&self) -> u64 {
        let version_set = self.version_set_arc.load_full();
        version_set
            .levels()
            .values()
            .flatten()
            .filter_map(|file_id| version_set.file_metadata(*file_id))
            .map(|(_, _, bytes, _)| *bytes)
            .sum()
    }

    /// Returns the physical row count for one table (`table\0` prefix) **in SSTs
    /// only**. Used by DataFusion statistics pushdown (PR 2.9).
    ///
    /// Why not add the memtable: after a flush the WAL is not immediately
    /// truncated (it is only reset above `wal_max_size_bytes`), so on restart
    /// the WAL replay re-materialises already-flushed rows into the memtable,
    /// making memtable and SST overlap. Counting SST rows alone is stable across
    /// restart, overlap-free, and slightly under-estimates a table with
    /// un-flushed writes — a conservative estimate is safer than double-counting.
    #[must_use]
    pub fn table_row_count(&self, table: &str) -> usize {
        let prefix = table_prefix(table);
        self.table_rows.read().get(&prefix).copied().unwrap_or(0)
    }

    /// Returns all SST ZoneMaps for `table`. Each entry is `(file_id, ZoneMap)`.
    ///
    /// The caller can use the per-column min/max/null_count to drive DataFusion's
    /// cost-based optimizer, and the per-file row counts for partition pruning.
    #[must_use]
    pub fn table_zone_maps(&self, table: &str, row_schema: Option<&Schema>) -> Vec<(u64, crate::sst_format::ZoneMap)> {
        let prefix = table_prefix(table);
        let mut out = Vec::new();
        let file_ids: Vec<u64> = self
            .version_set_arc
            .load_full()
            .levels()
            .values()
            .flat_map(|files| files.iter().copied())
            .collect();
        for &file_id in &file_ids {
            let path = sst_path(&self.config.path, file_id);
            if let Ok(mut zone) = self.config.sst_format.zone_map(&path, row_schema) {
                // Only include files whose key range overlaps with this table.
                // Key is prefixed with table name (`table\0`).
                if !zone.min_key.is_empty() {
                    // Skip files that end before this table's prefix.
                    if zone.max_key.as_ref() < &prefix[..] {
                        continue;
                    }
                    // Skip files whose min_key doesn't start with the table prefix.
                    if !zone.min_key.starts_with(&prefix) {
                        continue;
                    }
                }
                // Fill per-column statistics from the flush/compaction-time
                // cache. Files written before this cache existed (e.g. reopened
                // from an older run) fall back to empty stats, which DataFusion
                // reports as `Absent`.
                if let Some(table_stats) =
                    self.column_stats.read().get(&file_id).and_then(|m| m.get(table))
                {
                    zone.column_min = table_stats.min.clone();
                    zone.column_max = table_stats.max.clone();
                    zone.column_null_count = table_stats.null_count.clone();
                }
                out.push((file_id, zone));
            }
        }
        out
    }

    /// Removes versions that are no longer visible to any active snapshot.
    ///
    /// This method is the entry point for MVCC garbage collection. It delegates
    /// to [`MemTable::garbage_collect`] to prune superseded versions from the
    /// in-memory version chains.
    ///
    /// # When to Call
    ///
    /// GC should be triggered periodically or when the memtable accumulates many
    /// superseded versions. A simple heuristic is to call this when the memtable
    /// exceeds a size threshold or after a configurable number of transactions.
    ///
    /// # Returns
    ///
    /// The number of versions removed from the memtable.
    ///
    /// # Note on SST GC
    ///
    /// SST-level garbage collection (removing files with only obsolete versions)
    /// is handled by compaction. This method only GC's the memtable.
    pub fn garbage_collect(&self) -> usize {
        let oldest = self.watermark();
        self.memtable.garbage_collect(oldest)
    }

    /// Scans the visible key/value pairs in `[lower, upper)` (memtable + every
    /// SST, MVCC-filtered to `snapshot_ts`), returning the newest visible
    /// `(key, value)` per key. Tombstones are omitted.
    ///
    /// This is the storage-level primitive under SSI range-read tracking
    /// (PR 3.8) and the CDC / materialized-view replication consumers
    /// (PR 3.14/3.15): callers pass their transaction snapshot and record the
    /// `[lower, upper)` range as a read-conflict range.
    pub fn scan_range(
        &self,
        lower: &[u8],
        upper: &[u8],
        snapshot_ts: u64,
    ) -> Result<Vec<(Bytes, Bytes)>> {
        // Pin the caller's snapshot so a concurrent compaction cannot reclaim
        // a version this scan still needs (same guarantee as `scan_table`).
        let _guard = self
            .self_ref
            .upgrade()
            .map(|arc| arc.pin_snapshot(snapshot_ts));
        let snapshot = Snapshot::new(snapshot_ts);
        let mut visible: BTreeMap<Bytes, VersionedValue> = BTreeMap::new();
        for (key, value) in self.memtable.scan_range_versions(lower, upper, snapshot) {
            insert_newest(&mut visible, key, value);
        }
        let file_ids: Vec<u64> = self
            .version_set_arc
            .load_full()
            .levels()
            .values()
            .flat_map(|files| files.iter().copied())
            .collect();
        for file_id in file_ids {
            // Key-range pruning: skip SSTs whose [min_key, max_key] cannot
            // overlap [lower, upper).  `min_key` / `max_key` are recorded by
            // flush and compaction from the sorted row order.
            if let Some((min_key, max_key, _, _)) =
                self.version_set_arc.load_full().file_metadata(file_id)
            {
                if !min_key.is_empty()
                    && !max_key.is_empty()
                    && (max_key.as_ref() < lower || min_key.as_ref() >= upper)
                {
                    continue;
                }
            }
            let path = sst_path(&self.config.path, file_id);
            let rows = self.config.sst_format.scan_with_filter(
                &path,
                lower,
                upper,
                snapshot,
                None,
                None,
                None,
            )?;
            for row in rows {
                if self.is_dropped_prefix(&row.key) {
                    continue;
                }
                insert_newest(&mut visible, row.key, row.version);
            }
        }
        Ok(visible
            .into_iter()
            .filter_map(|(key, version)| version.value.map(|value| (key, value)))
            .collect())
    }

    /// Scans **every** visible key/value pair (memtable + every SST,
    /// MVCC-filtered to `snapshot_ts`), returning the newest visible version per
    /// key as `(key, Option<value>)` — `None` is a delete tombstone, so a
    /// replication consumer can propagate deletes instead of resurrecting
    /// deleted keys. Unlike [`Engine::scan_range`], there is no exclusive upper
    /// bound — a finite upper bound would silently exclude keys at or beyond it
    /// (security-review MEDIUM). Used by replication bootstrap / snapshot sync.
    pub fn scan_all(&self, snapshot_ts: u64) -> Result<Vec<(Bytes, Option<Bytes>)>> {
        let _guard = self
            .self_ref
            .upgrade()
            .map(|arc| arc.pin_snapshot(snapshot_ts));
        let snapshot = Snapshot::new(snapshot_ts);
        let mut visible: BTreeMap<Bytes, VersionedValue> = BTreeMap::new();
        for row in self.memtable.all_versions() {
            if snapshot.sees(&row.version) {
                insert_newest(&mut visible, row.key, row.version);
            }
        }
        let file_ids: Vec<u64> = self
            .version_set_arc
            .load_full()
            .levels()
            .values()
            .flat_map(|files| files.iter().copied())
            .collect();
        for file_id in file_ids {
            let path = sst_path(&self.config.path, file_id);
            let rows = self
                .config
                .sst_format
                .read_all(&path)?;
            for row in rows {
                if !snapshot.sees(&row.version) || self.is_dropped_prefix(&row.key) {
                    continue;
                }
                insert_newest(&mut visible, row.key, row.version);
            }
        }
        Ok(visible
            .into_iter()
            .map(|(key, version)| (key, version.value))
            .collect())
    }

    /// Scan every visible row in `table` and return the result as
    /// DataFusion [`RecordBatch`]es.
    ///
    /// `table` is encoded into the keys that [`Engine::ingest_plan`]
    /// writes — the scan walks the memtable and every SST, filters by
    /// the table prefix, decodes the stored Arrow IPC payload, and
    /// applies `projection`. Empty tables return a single empty batch
    /// whose schema is `fallback_schema` (when the table has no rows,
    /// the engine has no Arrow metadata to recover the schema from,
    /// so the caller must supply it). `projection` is then applied to
    /// `fallback_schema` to keep the column count consistent.
    pub fn scan_table(
        &self,
        table: &str,
        projection: Option<&[usize]>,
        fallback_schema: Option<&arrow_schema::SchemaRef>,
    ) -> Result<Vec<RecordBatch>> {
        let partitions = self.scan_table_arrow(table, projection, fallback_schema, usize::MAX, None)?;
        Ok(partitions.into_iter().flatten().collect())
    }

    /// Vortex→Arrow direct scan for DataFusion integration.  Reads business
    /// columns from L1+ SST files via `vortex-arrow` (parallelised with rayon),
    /// bypassing the opaque value blob entirely.  Memtable + L0 SSTs are merged
    /// via the row-codec path.  Each L1+ SST file is one DataFusion partition.
    #[allow(clippy::too_many_arguments)]
    pub fn scan_table_arrow(
        &self,
        table: &str,
        projection: Option<&[usize]>,
        fallback_schema: Option<&arrow_schema::SchemaRef>,
        batch_size: usize,
        predicates: Option<&[column_predicate::ColumnPredicate]>,
    ) -> Result<Vec<Vec<RecordBatch>>> {
        let read_pin = self.pin_read_snapshot();
        let snapshot_ts = match &read_pin {
            Some((ts, _)) => *ts,
            None => self.snapshot_ts(),
        };
        let snapshot = Snapshot::new(snapshot_ts);
        let prefix = table_prefix(table);
        let upper = prefix_upper_exclusive(&prefix);
        let table_dropped = self.is_dropped_prefix(&prefix);

        let row_schema: arrow_schema::SchemaRef = self
            .table_schemas.read().get(table).cloned()
            .or_else(|| fallback_schema.cloned())
            .ok_or_else(|| EngineError::CorruptMessage(format!(
                "scan_table_arrow: no schema for `{table}` and no fallback"
            )))?;

        let mut partitions: Vec<Vec<RecordBatch>> = Vec::new();

        // ── Partition 0: memtable + L0 SSTs (merged — they may overlap) ─
        let mut visible: std::collections::BTreeMap<Bytes, VersionedValue> =
            std::collections::BTreeMap::new();
        for (key, value) in self.memtable.scan_prefix_versions(&prefix, snapshot) {
            insert_newest(&mut visible, key, value);
        }
        let version_set = self.version_set_arc.load_full();
        if !table_dropped {
            let l0_files = version_set.files(0);
            let schema_ref = self.table_schemas.read().get(table).cloned();
            let named_preds: Option<Vec<(String, column_predicate::ColumnPredicate)>> =
                match (&schema_ref, predicates) {
                    (Some(s), Some(p)) => Some(column_predicate::predicates_with_fields(p, s)),
                    _ => None,
                };
            for &file_id in &l0_files {
                if let Some(ref preds) = named_preds {
                    if self.skip_file_by_column_stats(file_id, table, preds) { continue; }
                }
                let path = sst_path(&self.config.path, file_id);
                let rows = self.config.sst_format.scan_with_filter(
                    &path, &prefix, &upper, snapshot,
                    projection, named_preds.as_deref(), schema_ref.as_ref(),
                )?;
                for row in rows {
                    if self.is_dropped_prefix(&row.key) { continue; }
                    insert_newest(&mut visible, row.key, row.version);
                }
            }
        }
        if !visible.is_empty() {
            let batches = self.rows_to_batches(table, visible, projection, Some(&row_schema), batch_size)?;
            if !batches.is_empty() { partitions.push(batches); }
        }

        // ── Partitions 1+: each L1+ SST via Vortex→Arrow (rayon parallel) ─
        if !table_dropped {
            let preds: Vec<column_predicate::ColumnPredicate> =
                predicates.map(|p| p.to_vec()).unwrap_or_default();
            let mut file_ids: Vec<u64> = Vec::new();
            for level in 1.. {
                let files = version_set.files(level);
                if files.is_empty() { break; }
                for file_id in files {
                    if !preds.is_empty()
                        && self.skip_file_by_column_stats(file_id, table,
                            &column_predicate::predicates_with_fields(&preds, &row_schema))
                    { continue; }
                    file_ids.push(file_id);
                }
            }
            if !file_ids.is_empty() {
                use rayon::prelude::*;
                let results: Vec<Result<Vec<RecordBatch>>> = file_ids
                    .par_iter()
                    .map(|&file_id| {
                        let path = sst_path(&self.config.path, file_id);
                        crate::vortex_sst::scan_to_arrow_batches(
                            &path, &prefix, &upper, snapshot, &preds,
                            projection, row_schema.as_ref(), batch_size,
                            &crate::vortex_sst::new_session(),
                        )
                    })
                    .collect();
                for result in results {
                    let batches = result?;
                    if !batches.is_empty() && !batches.iter().all(|b| b.num_rows() == 0) {
                        partitions.push(batches);
                    }
                }
            }
        }

        if partitions.is_empty() {
            let projected_schema = match projection {
                Some(proj) => { validate_projection(row_schema.as_ref(), proj)?; Arc::new(row_schema.project(proj)?) }
                None => Arc::clone(&row_schema),
            };
            partitions.push(vec![RecordBatch::new_empty(projected_schema)]);
        }
        Ok(partitions)
    }

    fn skip_file_by_column_stats(
        &self, file_id: u64, table: &str,
        named_preds: &[(String, column_predicate::ColumnPredicate)],
    ) -> bool {
        let stats_guard = self.column_stats.read();
        let Some(table_stats) = stats_guard.get(&file_id).and_then(|m| m.get(table)) else { return false; };
        for (field, pred) in named_preds {
            if let column_predicate::ColumnPredicate::Eq { value, .. } = pred {
                if table_stats.min.get(field).map(|cm| cm > value).unwrap_or(false)
                    || table_stats.max.get(field).map(|cx| cx < value).unwrap_or(false)
                { return true; }
            }
        }
        false
    }

    /// P3: variant of [`Self::scan_table_batched`] that accepts a list of
    /// pushdown-eligible column predicates. The predicates are forwarded
    /// to the SST format's `scan_with_filter` so they short-circuit rows
    /// before the row-blob decode step.
    /// Builds the column accumulator for the memtable+L0 merge path in

    /// Builds the column accumulator we use for both scan paths. Pulled into
    /// a free function so `scan_table_batched` and
    /// `scan_table_with_filters` stay aligned without duplicating the
    /// per-batch slicing logic.
    fn rows_to_batches(
        &self,
        table: &str,
        visible: std::collections::BTreeMap<Bytes, crate::version::VersionedValue>,
        projection: Option<&[usize]>,
        fallback_schema: Option<&arrow_schema::SchemaRef>,
        batch_size: usize,
    ) -> Result<Vec<RecordBatch>> {
        let row_schema: arrow_schema::SchemaRef = self
            .table_schemas
            .read()
            .get(table)
            .cloned()
            .or_else(|| fallback_schema.cloned())
            .ok_or_else(|| {
                EngineError::CorruptMessage(format!(
                    "scan_table: no schema registered for table `{table}` and no \
                     fallback_schema supplied"
                ))
            })?;
        let mut columns: Vec<Vec<datafusion_common::ScalarValue>> =
            (0..row_schema.fields().len())
                .map(|_| Vec::new())
                .collect();
        // PR 5.4: when a projection is specified, only decode the projected
        // columns from the opaque row blob. This avoids decoding unused columns
        // (e.g. `SELECT id FROM t` with 20 columns only decodes `id`).
        // Validate projection first so out-of-range indices are caught early.
        if let Some(proj) = projection {
            validate_projection(row_schema.as_ref(), proj)?;
        }
        let cols_to_decode: Vec<usize> = match projection {
            Some(proj) => proj.to_vec(),
            None => (0..row_schema.fields().len()).collect(),
        };
        // Reusable buffer to avoid per-row Vec<ScalarValue> allocation
        let mut row_buf: Vec<datafusion_common::ScalarValue> = Vec::new();
        for (_, value) in visible {
            if let Some(payload) = value.value {
                if cols_to_decode.len() == row_schema.fields().len() {
                    // Full projection: zero-alloc fast path
                    crate::row_codec::decode_row_into(
                        &payload,
                        row_schema.as_ref(),
                        &mut row_buf,
                    )?;
                    for (col_idx, scalar) in (0..).zip(row_buf.drain(..)) {
                        if let Some(col) = columns.get_mut(col_idx) {
                            col.push(scalar);
                        }
                    }
                } else {
                    let row = crate::row_codec::decode_row_narrow(
                        &payload,
                        row_schema.as_ref(),
                        cols_to_decode.iter().copied(),
                    )?;
                    for (col_idx, scalar) in cols_to_decode.iter().zip(row) {
                        if let Some(col) = columns.get_mut(*col_idx) {
                            col.push(scalar);
                        }
                    }
                }
            }
        }
        if cols_to_decode.iter().all(|&idx| columns.get(idx).is_some_and(|c| c.is_empty())) {
            // Apply the projection to the empty schema so the
            // returned batches have the right shape — this matches
            // the old `scan_table_batched` behaviour for empty
            // tables (see tests::scan_table_valid_projection_on_empty_table_returns_projected_schema).
            let projected_schema = match projection {
                Some(proj) => {
                    validate_projection(row_schema.as_ref(), proj)?;
                    Arc::new(row_schema.project(proj)?)
                }
                None => Arc::clone(&row_schema),
            };
            return Ok(vec![RecordBatch::new_empty(projected_schema)]);
        }
        // Slice the per-column scalar vectors into batches of `batch_size`.
        // `columns` is non-empty here because the previous short-circuit
        // already returned when every column was empty.
        // PR 5.4: compute total_rows from the first *projected* column,
        // since non-projected columns stay empty.
        let total_rows = cols_to_decode
            .first()
            .and_then(|&idx| columns.get(idx))
            .map(Vec::len)
            .unwrap_or(0);
        let mut batches: Vec<RecordBatch> = Vec::new();
        let mut start = 0;
        // PR 5.4: only build arrays for projected columns, then project
        // the schema at the end. This avoids allocating and converting
        // unprojected columns.
        let build_cols: Vec<usize> = match projection {
            Some(proj) => proj.to_vec(),
            None => (0..row_schema.fields().len()).collect(),
        };
        while start < total_rows {
            let end = (start + batch_size).min(total_rows);
            let mut arrays = Vec::with_capacity(build_cols.len());
            for &col_idx in &build_cols {
                let col = columns.get(col_idx).ok_or_else(|| {
                    EngineError::Corrupt("scan batch column index out of range")
                })?;
                let window = col.get(start..end).ok_or_else(|| {
                    EngineError::Corrupt("scan batch window out of range")
                })?;
                let array = datafusion_common::ScalarValue::iter_to_array(window.iter().cloned())
                    .map_err(|err| {
                        EngineError::CorruptMessage(format!("build scan batch: {err}"))
                    })?;
                arrays.push(array);
            }
            let projected_schema = if build_cols.len() == row_schema.fields().len() {
                Arc::clone(&row_schema)
            } else {
                Arc::new(row_schema.project(&build_cols)?)
            };
            let batch = RecordBatch::try_new(projected_schema, arrays)
                .map_err(|err| {
                    EngineError::CorruptMessage(format!("build scan batch: {err}"))
                })?;
            batches.push(batch);
            start = end;
        }
        // Projection is already applied above — skip the post-hoc projection.
        Ok(batches)
    }


    /// Build a flat list of [`Mutation::Put`] entries from an iterable of
    /// [`RecordBatch`]es, encoding each row and computing its storage key.
    ///
    /// This is a convenience wrapper around [`row_codec::encode_row`] that
    /// accepts batches directly without requiring a DataFusion
    /// [`ExecutionPlan`]. Use this when importing data from external sources
    /// (CSV, Parquet, etc.).
    ///
    /// # Row-at-a-time encoding (intentional)
    ///
    /// Each row is encoded independently because the primary key is a
    /// per-value order-preserving encoding (`encode_pk_value`) whose byte
    /// layout must match `delete_plan`/`build_pk_key` exactly. Vectorizing the
    /// value encoding would save per-column downcasts but complicate the PK
    /// key assembly for negligible gain on the ingest path (dominated by WAL
    /// append + memtable apply, not scalar conversion). Do not "optimize" this
    /// into a columnar encoder without re-proving the PK byte invariance.
    pub fn ingest_batches(
        &self,
        batches: &[RecordBatch],
        table: &str,
        primary_key: &[String],
    ) -> Result<Vec<Mutation>> {
        if primary_key.is_empty() {
            return Err(EngineError::Corrupt(
                "ingest_batches requires a table with a primary key; the legacy \
                 row-index fallback encoding was removed",
            ));
        }
        if let Some(schema) = batches.first().map(RecordBatch::schema) {
            let mut schemas = self.table_schemas.write();
            if schemas.get(table) != Some(&schema) {
                schemas.insert(table.to_string(), schema);
                save_table_schemas(&self.config.path, &schemas);
            }
        }
        let mut merged: BTreeMap<Bytes, Bytes> = BTreeMap::new();
        let pk_columns = self.resolve_pk_cached(table, primary_key, batches.first())?;
        for batch in batches {
            for row in 0..batch.num_rows() {
                let key = build_pk_key(table, batch, row, &pk_columns)?;
                // Each row is encoded as a compact row payload (schema lives in
                // the per-table registry, not in the value).
                let payload = Bytes::from(row_codec::encode_row(batch, row)?);
                merged.insert(key, payload);
            }
        }
        Ok(merged
            .into_iter()
            .map(|(key, value)| Mutation::Put { key, value })
            .collect())
    }

    /// Delete rows in `table` that match any of `predicates` (AND semantics),
    /// using only the engine-native [`ColumnPredicate`] type.
    ///
    /// This is the **neutral** counterpart of [`Engine::delete_plan`]. The
    /// frontend's `predicate_translator` produces the predicate list from
    /// DataFusion `Expr`s; the engine itself stays free of DataFusion type
    /// bindings on its public DML surface (the planner-side glue lives in
    /// `hatp_frontend::dml_planner`).
    ///
    /// Fast path: when the predicate set contains exactly one equality per
    /// PK column, we use O(log n) point lookups (memtable + SST get).
    /// Otherwise we scan the table and apply the predicates as a residual
    /// filter using [`crate::column_predicate::apply_row_predicates`].
    ///
    /// **Capability gap vs [`Engine::delete_plan`]:** the neutral path only
    /// handles AND-conjoined single-column comparisons (Eq / Lt / LtEq /
    /// Gt / GtEq / Range). OR, NOT, LIKE, BETWEEN, multi-column expressions
    /// are NOT supported here — callers needing those shapes must use
    /// `delete_plan` with a DataFusion `Expr` slice instead. The frontend's
    /// `dml_planner` uses `delete_plan` by default to preserve full SQL
    /// semantics; `delete_with_column_predicates` is exposed for callers
    /// (tests, embedded APIs) that have already pre-translated their
    fn resolve_pk_cached(
        &self, table: &str, primary_key: &[String], first_batch: Option<&RecordBatch>,
    ) -> Result<Vec<(usize, String)>> {
        if let Some(c) = self.pk_column_cache.read().get(table) { return Ok(c.clone()); }
        let resolved = match first_batch {
            Some(b) => resolve_pk_columns(b, primary_key)?,
            None => return Err(EngineError::Corrupt("ingest_batches: no batches")),
        };
        self.pk_column_cache.write().insert(table.to_string(), resolved.clone());
        Ok(resolved)
    }

    /// predicates and want the neutral surface.
    pub fn delete_with_column_predicates(
        &self,
        table: &str,
        primary_key: &[String],
        predicates: &[crate::column_predicate::ColumnPredicate],
    ) -> Result<Vec<Mutation>> {
        if primary_key.is_empty() {
            return Err(EngineError::Corrupt(
                "delete_with_column_predicates requires a table with a primary key; \
                 the legacy row-index fallback encoding was removed",
            ));
        }
        if predicates.is_empty() {
            return Err(EngineError::Corrupt(
                "delete_with_column_predicates: no predicates; full-table DELETE is \
                 rejected; use `DELETE WHERE true` explicitly",
            ));
        }

        // The schema is required for both the fast path (PK column name → index
        // mapping) and the generic path (row_codec decoding).  Look it up once.
        let schema = self
            .table_schemas
            .read()
            .get(table)
            .cloned()
            .ok_or_else(|| {
                EngineError::CorruptMessage(format!(
                    "delete_with_column_predicates: no schema registered for table \
                     `{table}`"
                ))
            })?;

        // Fast path: when WHERE is exactly one equality predicate per PK
        // column (AND-conjoined), use O(log n) point lookups (memtable + SST
        // get). The extraction is exact: any OR / NOT / comparison / LIKE /
        // non-PK equality / duplicate equality falls through to the generic
        // scan+filter path below — predicates are never silently dropped.
        if let Some(encoded_parts) =
            try_extract_pk_eq_from_predicates(predicates, primary_key, &schema)
        {
            let key: Bytes = assemble_table_key(table, &encoded_parts);

            // ── Point lookup: memtable first (O(log n)) ──
            // Pin the snapshot for the lookup duration so concurrent compaction
            // cannot drop the version this delete is about to target.
            let read_pin = self.pin_read_snapshot();
            let snapshot_ts = match &read_pin {
                Some((ts, _)) => *ts,
                None => self.snapshot_ts(),
            };
            let snapshot = Snapshot::new(snapshot_ts);

            // `found` records whether the NEWEST visible version of `key` is live
            // (value == Some). A visible tombstone means the key is already deleted
            // and DELETE must report `EmptyBatch` — it must NOT fall through to an
            // older live version in a deeper level (which would resurrect the row).
            let found = if let Some(version) = self.memtable.get(&key, snapshot) {
                // The memtable is newer than any SST, so its visible version (live
                // or tombstone) is authoritative.
                version.value.is_some()
            } else {
                let mut found_in_sst = false;
                let file_ids = self.manifest.lock().file_ids();
                for file_id in file_ids {
                    let path = sst_path(&self.config.path, file_id);
                    match self
                        .config
                        .sst_format
                        .get(&path, &key, snapshot)
                    {
                        Ok(Some(version)) => {
                            found_in_sst = version.value.is_some();
                            break;
                        }
                        Ok(None) => {}
                        Err(err) => {
                            return Err(err);
                        }
                    }
                }
                found_in_sst
            };

            if !found {
                return Err(EngineError::EmptyBatch);
            }
            return Ok(vec![Mutation::Delete { key }]);
        }

        // Generic path: full scan + residual ColumnPredicate evaluation.
        let pk_columns = resolve_pk_columns_via_schema(&schema, primary_key)?;
        let batches = self.scan_table(table, None, None)?;
        let mut mutations: Vec<Mutation> = Vec::new();
        for batch in &batches {
            if batch.num_rows() == 0 {
                continue;
            }
            // Materialise each row's column scalars once; both the predicate
            // check and the PK key derivation read them.
            for row_idx in 0..batch.num_rows() {
                let row_scalars: Vec<datafusion_common::ScalarValue> = (0..batch.num_columns())
                    .map(|col| array_slot_to_scalar(batch.column(col).as_ref(), row_idx))
                    .collect();
                if !row_matches_predicates(&row_scalars, predicates) {
                    continue;
                }
                let key = build_pk_key_from_scalars(table, &pk_columns, &row_scalars)?;
                mutations.push(Mutation::Delete { key });
            }
        }
        if mutations.is_empty() {
            return Err(EngineError::EmptyBatch);
        }
        Ok(mutations)
    }

    // ============================================================================
    // Neutral helpers (no DataFusion Expr dependency — kept in engine)
    // ============================================================================
}

// KEEP: neutral helpers (no DataFusion Expr dependency)
fn resolve_pk_columns_via_schema(
    schema: &arrow_schema::SchemaRef,
    primary_key: &[String],
) -> Result<Vec<(usize, String)>> {
    let mut out = Vec::with_capacity(primary_key.len());
    for column in primary_key {
        let (idx, _) = schema.column_with_name(column).ok_or_else(|| {
            EngineError::Corrupt(
                "resolve_pk_columns_via_schema: primary key column not present in schema",
            )
        })?;
        out.push((idx, column.clone()));
    }
    Ok(out)
}

fn row_matches_predicates(
    scalars: &[datafusion_common::ScalarValue],
    predicates: &[crate::column_predicate::ColumnPredicate],
) -> bool {
    for pred in predicates {
        let col = pred.column();
        match scalars.get(col) {
            None => return false,
            Some(scalar) => {
                if !pred.evaluate(scalar) {
                    return false;
                }
            }
        }
    }
    true
}

fn build_pk_key_from_scalars(
    table: &str,
    pk_columns: &[(usize, String)],
    row_scalars: &[datafusion_common::ScalarValue],
) -> Result<Bytes> {
    let mut encoded_parts = Vec::with_capacity(pk_columns.len());
    for (idx, column) in pk_columns {
        let scalar = row_scalars.get(*idx).ok_or_else(|| {
            EngineError::Corrupt(
                "build_pk_key_from_scalars: primary-key column index out of range",
            )
        })?;
        let bytes = encode_pk_value(scalar).ok_or_else(|| EngineError::NullPrimaryKey {
            table: table.to_string(),
            column: column.clone(),
        })?;
        encoded_parts.push(bytes);
    }
    Ok(assemble_table_key(table, &encoded_parts))
}

/// Attempts to extract a single-row PK point-lookup from `predicates`.
///
/// Returns `Some(encoded_parts)` in PK-declaration order when **every** PK
/// column is covered by exactly one [`ColumnPredicate::Eq`] and there are no
/// extra predicates, ORs, or non-Eq comparisons.  Each extracted value is
/// encoded via [`encode_pk_value`] so the caller can assemble the storage key
/// with [`assemble_table_key`].
///
/// Returns `None` when the predicate set is not an exact PK match — the
/// caller falls through to the generic scan+filter path.
///
/// This is the engine-native counterpart of the frontend's
/// `dml_planner::try_translate_pk_predicates`, which operates on DataFusion
/// `Expr`s.  Both must produce identical key encodings for the same row.
fn try_extract_pk_eq_from_predicates(
    predicates: &[crate::column_predicate::ColumnPredicate],
    primary_key: &[String],
    schema: &arrow_schema::Schema,
) -> Option<Vec<Vec<u8>>> {
    // Must be exactly one predicate per PK column: any extra predicate
    // (e.g. `WHERE pk = 1 AND other_col = 2`) is not a pure PK lookup.
    if predicates.len() != primary_key.len() {
        return None;
    }
    let mut parts = Vec::with_capacity(primary_key.len());
    for pk_col_name in primary_key {
        let (pk_idx, _) = schema.column_with_name(pk_col_name)?;
        // Find the single Eq predicate targeting this PK column.  Using
        // `find` instead of a set ensures we detect duplicates (two Eq
        // predicates on the same column would both match `find`, but the
        // length check above already rejects extra predicates).
        let eq_pred = predicates.iter().find(|p| {
            p.column() == pk_idx && matches!(p, crate::column_predicate::ColumnPredicate::Eq { .. })
        })?;
        if let crate::column_predicate::ColumnPredicate::Eq { value, .. } = eq_pred {
            let encoded = encode_pk_value(value)?;
            parts.push(encoded);
        } else {
            // Unreachable: guarded by `matches!` above.
            return None;
        }
    }
    Some(parts)
}


/// Aggregate stats for [`Engine::scan_table`] cost estimation.
///
/// # Accuracy Note
///
/// The `sst_rows` count is cached incrementally (updated on `flush` and `compact`).
/// The `memtable_rows` count is computed on each call by scanning the memtable.
///
/// Both counts are **physical** (include tombstones and superseded MVCC versions),
/// not logical (MVCC-filtered) counts. Use these for cost estimation, not for
/// returning exact row counts to users.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanStats {
    /// Number of versions in the memtable (approximate, computed on call).
    pub memtable_rows: usize,
    /// Number of rows across all SST files (cached incrementally).
    pub sst_rows: usize,
}

/// Insert `value` into `map` keyed by `key`, keeping the entry with the
/// greatest `begin_ts` (the newest visible version). Used by `scan_table` to
/// collapse multiple SST / memtable copies of one logical key into a single
/// row.
fn insert_newest(map: &mut BTreeMap<Bytes, VersionedValue>, key: Bytes, value: VersionedValue) {
    if let Some(current) = map.get_mut(&key) {
        if value.begin_ts > current.begin_ts {
            *current = value;
        }
    } else {
        map.insert(key, value);
    }
}

/// Compaction merge: collapse a flat list of rows into one row per visible
/// MVCC version, dropping superseded versions that no active snapshot can
/// observe.
///
/// # Why `end_ts` is recomputed here
///
/// An SST freezes each version's `end_ts` at flush time. If a key is written
/// once per flush (write → flush → write → flush), the earlier SST keeps the
/// older version with `end_ts == OPEN_ENDED_TS`, because the superseding
/// version did not exist yet. Recomputing `end_ts` from the sorted `begin_ts`
/// sequence recovers the true visibility intervals before deciding what to
/// drop — otherwise every historical version looks still-open and nothing is
/// ever GC'd (the original "compaction never GCs superseded versions" defect).
///
/// Semantics:
/// 1. Sort by key ascending, then `begin_ts` descending (newest first).
/// 2. Collapse exact `(key, begin_ts)` duplicates (defensive — a version
///    normally lives in one SST, but merged levels can carry repeats).
/// 3. Recompute `end_ts`: newest → `OPEN_ENDED_TS`, each older version → the
///    next-newer version's `begin_ts`.
/// 4. Keep the newest version always, and an older version only while its
///    recomputed interval still overlaps the watermark (`end_ts > watermark`).
fn dedup_superseded(
    mut rows: Vec<memtable::VersionedRow>,
    watermark: u64,
) -> Vec<memtable::VersionedRow> {
    rows.sort_by(|left, right| {
        left.key
            .cmp(&right.key)
            .then_with(|| right.version.begin_ts.cmp(&left.version.begin_ts))
    });
    rows.dedup_by(|left, right| {
        left.key == right.key && left.version.begin_ts == right.version.begin_ts
    });
    let mut out: Vec<memtable::VersionedRow> = Vec::with_capacity(rows.len());
    let mut iter = rows.into_iter().peekable();
    while let Some(head) = iter.next() {
        let key = head.key.clone();
        // `head` is the newest version (begin_ts descending). It is always
        // kept and its recomputed interval is open-ended.
        let head_begin = head.version.begin_ts;
        let mut newest = head.version;
        newest.end_ts = OPEN_ENDED_TS;
        out.push(memtable::VersionedRow {
            key: key.clone(),
            version: newest,
        });
        // Older versions of the same key: recompute `end_ts` from the previous
        // (newer) version's `begin_ts`, and keep only those an active snapshot
        // can still observe.
        let mut prev_begin = head_begin;
        while let Some(next) = iter.next_if(|row| row.key == key) {
            let mut version = next.version;
            version.end_ts = prev_begin;
            let begin = version.begin_ts;
            if version.end_ts > watermark {
                out.push(memtable::VersionedRow {
                    key: key.clone(),
                    version,
                });
            }
            prev_begin = begin;
        }
    }
    out
}

/// Splits compaction inputs into chunks whose combined bytes stay under
/// `max_bytes` (PR 2.6). Each chunk is merged independently by
/// [`Engine::sub_compact`] on a separate rayon thread.
fn split_by_size(
    inputs: &[u64],
    version_set: &manifest::VersionSet,
    max_bytes: u64,
) -> Vec<Vec<u64>> {
    let mut chunks: Vec<Vec<u64>> = Vec::new();
    let mut current: Vec<u64> = Vec::new();
    let mut current_bytes = 0_u64;
    for input in inputs {
        let bytes = version_set
            .file_metadata(*input)
            .map(|(_, _, bytes, _)| *bytes)
            .unwrap_or(0);
        if current_bytes.saturating_add(bytes) > max_bytes && !current.is_empty() {
            chunks.push(std::mem::take(&mut current));
            current_bytes = 0;
        }
        current.push(*input);
        current_bytes = current_bytes.saturating_add(bytes);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Build the per-table prefix used to multiplex logical tables in a single
/// logical key value store.
fn table_prefix(table: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(table.len().saturating_add(1));
    out.extend_from_slice(table.as_bytes());
    out.push(0);
    out
}

/// Extracts the table prefix from a storage key (up to and including the first `\0`).
///
/// The storage key format is `table\0 + pk` (see `assemble_table_key`), so the first `\0`
/// delimits the `table\0` prefix. Falls back to the entire key when no `\0` is present
/// (defensive; should not happen in normal operation).
fn key_table_prefix(key: &[u8]) -> &[u8] {
    let prefix_end = key
        .iter()
        .position(|&byte| byte == 0)
        .map_or(key.len(), |index| index + 1);
    key.get(..prefix_end).unwrap_or(key)
}

/// Output of `compact_chunk`: the output SST handle, global deleted row count, and per-table net change.
type ChunkOutput = (sst_format::SstHandle, usize, BTreeMap<Vec<u8>, isize>);

/// Counts physical rows (including tombstones / superseded) grouped by `table\0` prefix.
fn count_rows_by_prefix(rows: &[memtable::VersionedRow]) -> BTreeMap<Vec<u8>, usize> {
    let mut counts = BTreeMap::new();
    for row in rows {
        let entry = counts
            .entry(key_table_prefix(&row.key).to_vec())
            .or_insert(0_usize);
        *entry = (*entry).saturating_add(1_usize);
    }
    counts
}

/// Computes per-prefix row count net change (`output - input`, negative = net deletion).
fn prefix_count_delta(
    input: &BTreeMap<Vec<u8>, usize>,
    output: &BTreeMap<Vec<u8>, usize>,
) -> BTreeMap<Vec<u8>, isize> {
    let keys: BTreeSet<&Vec<u8>> = input.keys().chain(output.keys()).collect();
    let mut delta = BTreeMap::new();
    for key in keys {
        let change = output.get(key).copied().unwrap_or(0) as isize
            - input.get(key).copied().unwrap_or(0) as isize;
        if change != 0 {
            delta.insert(key.clone(), change);
        }
    }
    delta
}

/// Validates that every projection index is in-bounds for `schema`, returning
/// a descriptive [`EngineError::CorruptMessage`] instead of letting Arrow's
/// `project` surface a bare out-of-bounds error (R-25). A projection index
/// that equals or exceeds the column count means the caller passed a stale
/// projection (e.g. against a schema that has since been altered).
fn validate_projection(schema: &arrow_schema::Schema, projection: &[usize]) -> Result<()> {
    if let Some(max) = projection.iter().max() {
        if *max >= schema.fields().len() {
            return Err(EngineError::CorruptMessage(format!(
                "projection index {max} out of range (schema has {} columns)",
                schema.fields().len()
            )));
        }
    }
    Ok(())
}

/// Resolve the column indices of every requested primary-key column in
/// `batch`. Returns `Err(NullPrimaryKey)` if any of the requested PK
/// columns is absent from the batch's schema (the caller passed a
/// stale PK list).
fn resolve_pk_columns(
    batch: &arrow_array::RecordBatch,
    primary_key: &[String],
) -> Result<Vec<(usize, String)>> {
    let mut out = Vec::with_capacity(primary_key.len());
    for column in primary_key {
        let (idx, _) = batch.schema().column_with_name(column).ok_or_else(|| {
            EngineError::Corrupt(
                "resolve_pk_columns: primary key column not present in input batch",
            )
        })?;
        out.push((idx, column.clone()));
    }
    Ok(out)
}

/// Assembles the storage key for a table row from encoded PK parts.
///
/// Format: `table\0` then, per PK column in order, `\0 + encoded_part`.
/// Shared by [`build_pk_key`] and [`Engine::delete_plan`] so the two key
/// layouts cannot drift — a mismatch here would make DELETE miss rows that
/// INSERT wrote (this bit us once; see the note in `delete_plan`).
pub fn assemble_table_key(table: &str, encoded_parts: &[Vec<u8>]) -> Bytes {
    let mut out = Vec::with_capacity(
        table.len()
            + 1
            + encoded_parts
                .iter()
                .map(|part| part.len() + 1)
                .sum::<usize>(),
    );
    out.extend_from_slice(table.as_bytes());
    out.push(0);
    for part in encoded_parts {
        out.push(0);
        out.extend_from_slice(part);
    }
    Bytes::from(out)
}

/// Build the primary-key bytes for a single row in `batch`. Concatenates
/// the encoded PK parts (one per requested column) in order via
/// [`assemble_table_key`]. The `\0`-delimited layout is unambiguous because
/// [`encode_pk_value`] never emits a bare `0x00` inside a variable-length
/// (string) part — see the invariants documented there.
///
/// There is no empty-PK fallback: callers must reject an empty `primary_key`
/// before reaching this function (a table MUST declare a primary key).
fn build_pk_key(
    table: &str,
    batch: &arrow_array::RecordBatch,
    row: usize,
    pk_columns: &[(usize, String)],
) -> Result<Bytes> {
    let mut encoded_parts = Vec::with_capacity(pk_columns.len());
    for (idx, column) in pk_columns {
        let array = batch.column(*idx);
        let scalar = array_slot_to_scalar(array.as_ref(), row);
        let bytes = encode_pk_value(&scalar).ok_or_else(|| EngineError::NullPrimaryKey {
            table: table.to_string(),
            column: column.clone(),
        })?;
        encoded_parts.push(bytes);
    }
    Ok(assemble_table_key(table, &encoded_parts))
}

/// Background compaction worker loop.
/// Periodically runs MVCC GC and compaction.
fn compaction_worker_loop(engine: &Engine, shutdown: &std::sync::atomic::AtomicBool) {
    const COMPACTION_INTERVAL_MS: u64 = 5000; // 5 seconds between compaction attempts
    /// Panic threshold after which the worker quits instead of spinning forever
    /// (R-17). Below it, the worker backs off exponentially.
    const MAX_PANICS_BEFORE_QUIT: u64 = 100;

    // Track the last GC so the memtable is reclaimed on `gc_interval` even
    // when there is no compaction work (R-09 close-the-loop).
    let mut last_gc_at = std::time::Instant::now();
    // Consecutive panic count; reset on a clean iteration.
    let mut panic_count: u64 = 0;

    loop {
        // Check for shutdown signal
        if shutdown.load(std::sync::atomic::Ordering::Relaxed) {
            tracing::debug!("compaction worker shutting down");
            break;
        }

        // ── Periodic MVCC GC (PR 1.6 / R-09) ───────────────────────────────
        // Prune memtable versions invisible to every active snapshot. The
        // watermark comes from the engine (min pinned snapshot / snapshot_ts),
        // so a live transaction's versions are never dropped.
        if last_gc_at.elapsed() >= engine.config.gc_interval {
            let watermark = engine.watermark();
            let dropped = engine.memtable().garbage_collect(watermark);
            if dropped > 0 {
                engine.metrics().record_gc_versions_dropped(dropped as u64);
                tracing::debug!(
                    dropped,
                    watermark,
                    "background GC dropped memtable versions"
                );
            }
            last_gc_at = std::time::Instant::now();
        }

        // ── Adaptive compaction picker (PR 2.5) ─────────────────────────────
        // Feed a fresh metrics snapshot so a Dostoevsky/SILK picker can switch
        // layouts under workload shifts. The default picker ignores this.
        engine
            .config
            .compaction_picker
            .update_from_workload(&engine.metrics().snapshot());

        // Run a single iteration under `catch_unwind` so a panic in the
        // compaction path (or in Vortex / DataFusion underneath) cannot
        // kill the background worker thread. The release profile keeps
        // `panic = "unwind"`, so an uncaught panic here would unwind out
        // of the worker thread and stop compaction permanently; catching
        // it per-iteration keeps the loop alive.
        //
        // `AssertUnwindSafe` is justified because the iteration only reads
        // engine state (manifest, SST paths) and re-locks the parking_lot
        // mutexes, which are already poisoning-tolerant.
        let iteration = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_compaction_iteration(engine);
        }));
        match iteration {
            Ok(()) => {
                // A clean iteration resets the panic backoff.
                panic_count = 0;
            }
            Err(payload) => {
                // R-17: count panics and back off exponentially. A persistent
                // panic (e.g. a corrupt SST that crashes the picker) must not
                // turn into a hot spin loop; after the threshold the worker
                // quits so the failure is loud rather than silently retried
                // forever.
                panic_count = panic_count.saturating_add(1);
                engine.metrics().record_compaction_panic();
                tracing::error!(
                    panic = %format_payload(&payload),
                    panic_count,
                    "compaction worker iteration panicked"
                );
                if panic_count > MAX_PANICS_BEFORE_QUIT {
                    tracing::error!(
                        panic_count,
                        "compaction worker quitting after too many consecutive panics"
                    );
                    break;
                }
                // Exponential backoff: 100ms, 200ms, … capped at 6.4s.
                let backoff_ms = 100_u64.saturating_mul(1_u64 << panic_count.min(6));
                std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
            }
        }

        // Sleep before next check. The sleep is *outside* the catch so a
        // panic during sleep (impossible in practice) is treated like any
        // other panic, but it does not need to be caught.
        std::thread::sleep(std::time::Duration::from_millis(COMPACTION_INTERVAL_MS));
    }
}

/// One pass of the compaction loop: gather file metadata, let the picker
/// choose a job, and run it. Extracted into its own function so the loop
/// body stays small and can be wrapped in `catch_unwind`.
///
/// # Performance note
///
/// File metadata (min_key, max_key, bytes) is read from the in-memory
/// [`VersionSet`] which is populated from the manifest on startup and
/// updated on every flush/compaction. This makes the compaction picker
/// O(number of files) instead of O(total SST bytes).
fn run_compaction_iteration(engine: &Engine) {
    let job =
        {
            let manifest = engine.manifest.lock();
            // Clone the version set once; the loop below reads levels + per-file
            // metadata off the owned snapshot instead of re-cloning per file.
            let version_set = manifest.version_set();

            // Collect file metadata from the in-memory VersionSet.
            // The manifest now stores min_key, max_key, and bytes for each SST file
            // at flush/compaction time, so we don't need to re-read every SST file.
            let mut files =
                Vec::with_capacity(version_set.levels().values().map(|s| s.len()).sum());
            for (level, file_ids) in version_set.levels() {
                for file_id in file_ids {
                    // Try to get metadata from the manifest first (fast path).
                    // Fall back to reading the SST file only if metadata is missing
                    // (backward compat with old manifest entries).
                    let (min_key, max_key, bytes, created_at) =
                        if let Some((min_k, max_k, sz, created)) =
                            version_set.file_metadata(*file_id)
                        {
                            (min_k.clone(), max_k.clone(), *sz, *created)
                        } else {
                            // Fallback: read SST file for metadata (old manifest format).
                            // This path should only trigger on existing databases with old manifests.
                            let path = sst_path(&engine.config.path, *file_id);
                            let file_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                            let (min_k, max_k) = match engine
                                .config
                                .sst_format
                                .read_all(&path)
                            {
                                Ok(rows) if !rows.is_empty() => (
                                    rows.first().map(|r| r.key.clone()).unwrap_or_default(),
                                    rows.last().map(|r| r.key.clone()).unwrap_or_default(),
                                ),
                                _ => (Bytes::new(), Bytes::new()),
                            };
                            (min_k, max_k, file_bytes, 0)
                        };
                    files.push(compaction::FileMeta {
                        file_id: *file_id,
                        level: *level,
                        bytes,
                        min_key,
                        max_key,
                        created_at,
                    });
                }
            }

            // Use the engine's configured picker (shared instance, not rebuilt per
            // iteration) to select a job.
            let picker = engine.config.compaction_picker.clone();
            picker.pick(&files)
        };

    // Execute compaction job if selected
    if let Some(job) = job {
        tracing::info!(?job, "executing compaction job");
        // Large inputs use parallel sub-compaction (PR 2.6); small inputs use single-threaded compact.
        // The threshold comes from `sub_compaction_max_bytes`, consistent with `sub_compact`'s sharding threshold.
        // When a profiler is configured (PR 3.3), wrap the iteration for
        // sampled latency capture.
        let run_job = || {
            if job.estimated_bytes > 64 * 1024 * 1024 {
                engine
                    .sub_compact(job.source_level, job.target_level, engine.watermark())
                    .map(|_| ())
            } else {
                engine
                    .compact(job.source_level, job.target_level)
                    .map(|_| ())
            }
        };
        let result = run_job();
        if let Err(e) = result {
            tracing::error!(error = %e, "compaction job failed");
        }
    }
}



/// Encodes `mutations` into WAL records under `tx_id` (no `Commit` marker —
/// the caller appends that via `append_batch_commit_and_sync` /
/// `append_multi_batch_commit_and_sync`).
/// Format a catch_unwind payload for error logging.
fn format_payload(payload: &Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "<unknown panic>".to_string())
}

fn build_wal_records(tx_id: u64, mutations: &[Mutation]) -> Vec<WalRecord> {
    let mut records = Vec::with_capacity(mutations.len());
    for mutation in mutations {
        records.push(match mutation {
            Mutation::Put { key, value } => WalRecord::put(tx_id, key.clone(), value.clone()),
            Mutation::Delete { key } => WalRecord::delete(tx_id, key.clone()),
        });
    }
    records
}

/// Background WAL fsync worker (PR 5.1). Periodically calls `sync_data` on
/// the WAL file so the write path can return immediately after `write_all`.
/// Only spawned when `sync_mode` is [`SyncMode::Group`].
fn wal_sync_worker_loop(
    wal: &Arc<Wal>,
    shutdown: &std::sync::atomic::AtomicBool,
    interval_us: u64,
) {
    let interval = std::time::Duration::from_micros(interval_us);
    while !shutdown.load(std::sync::atomic::Ordering::Relaxed) {
        std::thread::sleep(interval);
        if shutdown.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        if let Err(error) = wal.sync_data() {
            tracing::error!(error = %error, "WAL background fsync failed");
        }
    }
    // Final fsync on shutdown so no committed data is lost.
    if let Err(error) = wal.sync_data() {
        tracing::error!(error = %error, "WAL final fsync on shutdown failed");
    }
}

