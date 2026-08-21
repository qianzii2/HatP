//! Synchronous embedded database API on top of the durable engine.
//!
//! Core API:
//! - `Database::open` / `put` / `get` / `delete` / `flush` (OLTP)
//! - `Database::execute_sql` (DataFusion SQL)
//! - `Transaction::begin` / `put` / `delete` / `get` / `commit` / `rollback`

use bytes::Bytes;
use hatp_engine::{Engine, EngineConfig, EngineError, Mutation, SnapshotGuard};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use hatp_frontend::catalog::{Catalog, CatalogSnapshot};
use hatp_frontend::execution::FrontendSession;
use hatp_tx::{TxManager, TxManagerHook, TxnId};

/// Number of DDL statements buffered before the catalog is persisted eagerly.
/// A crash between the last persist and now must not lose more than this many
/// DDL statements (R-16).
const CATALOG_SAVE_THRESHOLD: u64 = 16;

/// Downcast an engine's [`hatp_engine::EngineHook`] to a [`TxManagerHook`]
/// and clone out the underlying [`TxManager`].
///
/// The downcast uses `EngineHook::as_any` (defined on `hatp_engine`'s
/// trait). Returns `None` when the hook is not a `TxManagerHook`, in
/// which case `Database::begin_ssi` degrades to plain snapshot
/// isolation — see the docs on that method.
fn detect_tx_manager(engine: &Arc<Engine>) -> Option<Arc<TxManager>> {
    let hook = engine.hook();
    let any = hook.as_any();
    // We can't downcast to `TxManagerHook` directly here because we
    // don't have its concrete type available without a use statement
    // that could form a cycle. Instead, walk the engine's hook as
    // `&dyn std::any::Any` and try downcasting to the public re-export
    // of TxManagerHook.
    any.downcast_ref::<TxManagerHook>().map(|h| h.manager())
}

/// Result type returned by the public embedded API.
pub type Result<T> = std::result::Result<T, DatabaseError>;

/// Errors surfaced by the public database facade.
#[derive(Debug, thiserror::Error)]
pub enum DatabaseError {
    /// The storage engine rejected or failed an operation.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// A transaction with no writes was committed.
    #[error("cannot commit a read-only transaction through the write path")]
    ReadOnlyCommit,
    /// DataFusion returned an error.
    #[error("datafusion error: {0}")]
    DataFusion(#[from] datafusion_common::DataFusionError),
    /// Arrow returned an error.
    #[error("arrow error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),
    /// Frontend session reported a failure.
    #[error("frontend error: {0}")]
    Frontend(String),
    /// SSI pre-commit validation detected a write-write conflict. The
    /// `txn` and `key` identify the conflicting transaction and key.
    #[error("SSI write-write conflict in transaction {txn} on key {:02x?}", key)]
    SsiConflict {
        /// Conflicting transaction id.
        txn: u64,
        /// Key that triggered the conflict.
        key: Vec<u8>,
    },
    /// SSI pre-commit validation detected a read-write antidependency. A
    /// concurrent transaction committed a write to a key this transaction
    /// previously read, after this transaction began. This catches write-skew
    /// anomalies that the write-write check misses.
    #[error("SSI read-write conflict in transaction {txn} on key {:02x?}", key)]
    SsiReadWriteConflict {
        /// Conflicting transaction id.
        txn: u64,
        /// Key that was read by this txn and concurrently written by a peer.
        key: Vec<u8>,
    },
    /// The caller requested a configuration that contradicts the engine's
    /// actual state (e.g. an `open_with_engine` path mismatch).
    #[error("invalid configuration: {0}")]
    Config(String),
}

/// User-facing handle to an open HatP database.
///
/// Clones share the same inner state: dropping one clone neither shuts the
/// engine down nor persists the catalog — only the final clone does (see the
/// [`Drop`] impl).
#[derive(Clone)]
pub struct Database {
    inner: Arc<DatabaseInner>,
}

struct DatabaseInner {
    engine: Arc<Engine>,
    /// Frontend session bundles DataFusion. The catalog lives
    /// behind an `Arc` so this struct can be `Clone` cheaply.
    frontend: Arc<FrontendSession>,
    /// Shared handle to the catalog that backs the [`FrontendSession`].
    catalog: Arc<Catalog>,
    /// Optional transaction manager. Present when the database was opened
    /// via [`Database::open_with_engine`] with a [`TxManagerHook`]-backed
    /// engine; otherwise `None` and [`Database::begin`] returns plain SI
    /// transactions.
    tx_manager: Option<Arc<TxManager>>,
    /// DDL statements since the last catalog persist (R-16). When this reaches
    /// [`CATALOG_SAVE_THRESHOLD`] the catalog is persisted eagerly so a crash
    /// cannot lose an unbounded amount of DDL.
    ddl_count_since_save: AtomicU64,
}

impl std::fmt::Debug for Database {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Database")
            .field("engine", &self.inner.engine)
            .field("frontend", &"<FrontendSession>")
            .finish()
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        // Persist the catalog only when this is the last handle. Clones share
        // `inner`; `Arc::strong_count == 1` here means no other clone exists,
        // so writing the catalog now is safe (no concurrent writer) and
        // reflects the final DDL state. A failed write is best-effort: the
        // engine data is already durable, and the catalog can be re-created.
        if Arc::strong_count(&self.inner) == 1 {
            let _ = save_catalog(self.inner.engine.path(), &self.inner.catalog);
        }
    }
}

impl Database {
    /// Opens or creates a database at the given filesystem path and performs
    /// WAL recovery before returning.
    ///
    /// The returned [`Database`] owns an empty [`FrontendSession`] backed by a
    /// fresh `SessionContext`. Tables must be registered explicitly via
    /// [`Database::register_table`] before SQL can scan them.
    ///
    /// This constructor does **not** install a [`TxManagerHook`]; transactions
    /// created via [`Database::begin`] are snapshot-isolated but never
    /// participate in SSI conflict detection. Use
    /// [`Database::open_with_tx_manager`] to get SSI support.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_engine(
            path.as_ref(),
            Engine::open(EngineConfig::new(path.as_ref()))?,
        )
    }

    /// Open a [`Database`] around an existing engine. Lets callers inject a
    /// pre-configured [`Engine`] (e.g. with a [`hatp_engine::EngineHook`]
    /// or fail-point harness).
    ///
    /// The `path` argument is NOT used to configure storage — the engine
    /// already owns its durable directory via [`EngineConfig`].  It is
    /// retained purely for diagnostic correlation: a `debug!` log compares
    /// `path` against `engine.path()` so mismatches can be spotted in
    /// tracing output.  If you are constructing a `Database` from a
    /// pre-built engine, pass the same path that was given to
    /// [`EngineConfig::new`].
    ///
    /// If `engine`'s hook is a [`hatp_tx::TxManagerHook`], the resulting
    /// [`Database`] shares the same [`TxManager`] so [`Database::begin_ssi`]
    /// can return transactions wired through the engine's pre-commit
    /// hook. Detection goes through `EngineHook::as_any` (see
    /// `hatp-engine::EngineHook`).
    pub fn open_with_engine(path: impl AsRef<Path>, engine: Arc<Engine>) -> Result<Self> {
        // The caller's `path` must match the engine's data directory. A
        // mismatch means the caller's catalog / index assumptions point at a
        // different directory than the engine actually owns; this used to be a
        // `debug!` log and would silently load the catalog from the *engine's*
        // path, masking the operator's error. Compare both literally and
        // canonicalised (a caller passing "./db" vs "db" must still match).
        let requested = path.as_ref();
        let engine_path = engine.path();
        let paths_match = requested == engine_path
            || match (requested.canonicalize(), engine_path.canonicalize()) {
                (Ok(requested_canon), Ok(engine_canon)) => requested_canon == engine_canon,
                _ => false,
            };
        if !paths_match {
            return Err(DatabaseError::Config(format!(
                "engine path mismatch: opened '{}' but requested '{}'",
                engine_path.display(),
                requested.display()
            )));
        }
        let catalog = Arc::new(load_catalog(engine.path())?);
        let frontend = Arc::new(FrontendSession::with_catalog(Arc::clone(&catalog)));
        hatp_frontend::execution::register_catalog_with(
            frontend.session(),
            Arc::clone(&catalog),
            Arc::clone(&engine),
        )
        .map_err(|err| DatabaseError::Frontend(err.to_string()))?;
        // Detect a TxManagerHook via `EngineHook::as_any` downcast.
        // When found, the same manager is exposed to `begin_ssi` so the
        // transaction can join the SSI conflict-detection flow the
        // engine pre-commit hook is already running. When the hook is
        // anything else (NoopHook, user-supplied, etc.) we leave
        // `tx_manager` as `None` and `begin_ssi` degrades to plain SI
        // (visible via `Transaction::is_ssi`).
        let tx_manager = detect_tx_manager(&engine);
        Ok(Self {
            inner: Arc::new(DatabaseInner {
                engine,
                frontend,
                catalog,
                tx_manager,
                ddl_count_since_save: AtomicU64::new(0),
            }),
        })
    }

    /// Opens a database with a [`TxManager`] wired into the engine's
    /// [`hatp_engine::EngineHook`] so SSI transactions can be started.
    ///
    /// The returned [`Database::begin_ssi`] returns transactions whose
    /// `commit` validates against concurrent SSI writes; conflicting writes
    /// fail with [`DatabaseError::SsiConflict`].
    pub fn open_with_tx_manager(path: impl AsRef<Path>, manager: Arc<TxManager>) -> Result<Self> {
        let hook: Arc<dyn hatp_engine::EngineHook> =
            Arc::new(TxManagerHook::new(Arc::clone(&manager)));
        let engine = Engine::open_with_hook(
            EngineConfig::new(path.as_ref()),
            hook,
        )?;
        let catalog = Arc::new(load_catalog(engine.path())?);
        let frontend = Arc::new(FrontendSession::with_catalog(Arc::clone(&catalog)));
        hatp_frontend::execution::register_catalog_with(
            frontend.session(),
            Arc::clone(&catalog),
            Arc::clone(&engine),
        )
        .map_err(|err| DatabaseError::Frontend(err.to_string()))?;
        Ok(Self {
            inner: Arc::new(DatabaseInner {
                engine,
                frontend,
                catalog,
                tx_manager: Some(manager),
                ddl_count_since_save: AtomicU64::new(0),
            }),
        })
    }

    /// Begins a snapshot transaction with an empty write buffer.
    #[must_use]
    pub fn begin(&self) -> Transaction {
        let (snapshot_ts, snapshot_guard) = self.inner.engine.begin_read();
        Transaction {
            database: self.clone(),
            snapshot_ts,
            mutations: Vec::new(),
            reserved_tx_id: None,
            is_ssi: false,
            tx_manager: self.inner.tx_manager.clone(),
            _snapshot_guard: snapshot_guard,
        }
    }

    /// Begins an SSI (Serializable Snapshot Isolation) transaction.
    ///
    /// The returned [`Transaction`] records every read/write key into the
    /// shared [`TxManager`]; `commit` runs SSI validation via the engine's
    /// pre-commit hook and returns [`DatabaseError::SsiConflict`] (on a
    /// concurrent write to any key in this transaction's write set) or
    /// [`DatabaseError::SsiReadWriteConflict`] (on a concurrent committed
    /// write to any key in this transaction's read set).
    ///
    /// Requires a [`TxManager`] wired into the engine (use
    /// [`Database::open_with_tx_manager`]). Returns a snapshot transaction
    /// if no manager is installed — the caller can detect this by checking
    /// [`Transaction::is_ssi`] (will be `false`).
    #[must_use]
    pub fn begin_ssi(&self) -> Transaction {
        let Some(manager) = self.inner.tx_manager.clone() else {
            return self.begin();
        };
        // Reserve the id from the engine (the same monotonically increasing
        // sequence autocommit `put`/`delete` use), and take the snapshot read
        // point BEFORE reserving. This keeps the SSI snapshot aligned with
        // engine-committed timestamps so it sees every prior autocommit write,
        // and guarantees the reserved id never collides with a later write.
        let (start_ts, snapshot_guard) = self.inner.engine.begin_read();
        let reserved = self.inner.engine.reserve_tx_id();
        let ssi = manager.begin_ssi_with_id(TxnId(reserved), start_ts);
        Transaction {
            database: self.clone(),
            snapshot_ts: start_ts,
            mutations: Vec::new(),
            reserved_tx_id: Some(ssi.handle.id),
            is_ssi: true,
            tx_manager: Some(manager),
            _snapshot_guard: snapshot_guard,
        }
    }

    /// Atomically inserts or replaces one key/value pair.
    pub fn put(&self, key: impl Into<Bytes>, value: impl Into<Bytes>) -> Result<u64> {
        Ok(self.inner.engine.write(&[Mutation::Put {
            key: key.into(),
            value: value.into(),
        }])?)
    }

    /// Atomically deletes one key.
    pub fn delete(&self, key: impl Into<Bytes>) -> Result<u64> {
        Ok(self
            .inner
            .engine
            .write(&[Mutation::Delete { key: key.into() }])?)
    }

    /// Reads the newest committed value.
    pub fn get(&self, key: &[u8]) -> Result<Option<Bytes>> {
        // Atomically read + pin the snapshot so the point read cannot race a
        // background compaction that reclaims the version it needs.
        let (snapshot_ts, _guard) = self.inner.engine.begin_read();
        Ok(self.inner.engine.get(key, snapshot_ts)?)
    }

    /// Reads a value at an explicit snapshot timestamp.
    pub fn get_at(&self, key: &[u8], snapshot_ts: u64) -> Result<Option<Bytes>> {
        // Pin the caller-supplied snapshot for the read duration.
        let _guard = self.inner.engine.pin_snapshot(snapshot_ts);
        Ok(self.inner.engine.get(key, snapshot_ts)?)
    }

    /// Flushes a consistent memtable snapshot into an immutable SST file.
    /// Returns the number of rows flushed.
    pub fn flush(&self) -> Result<usize> {
        let rows_before = self.inner.engine.memtable().len();
        let _ = self.inner.engine.flush()?;
        Ok(rows_before)
    }

    /// Returns the latest committed timestamp visible to a new transaction.
    #[must_use]
    pub fn snapshot_ts(&self) -> u64 {
        self.inner.engine.snapshot_ts()
    }

    /// Returns whether the memtable has reached its configured flush threshold.
    #[must_use]
    pub fn should_flush(&self) -> bool {
        self.inner.engine.should_flush()
    }

    /// Borrow the underlying engine (for advanced integrations).
    #[must_use]
    pub fn engine(&self) -> &Arc<Engine> {
        &self.inner.engine
    }

    /// Borrow the underlying frontend session.
    #[must_use]
    pub fn frontend(&self) -> &Arc<FrontendSession> {
        &self.inner.frontend
    }

    /// Borrow the catalog handle that backs the [`FrontendSession`].
    #[must_use]
    pub fn catalog(&self) -> &Arc<Catalog> {
        &self.inner.catalog
    }

    /// Register a logical table with the DataFusion catalog. The table is
    /// immediately available as a [`hatp_frontend::execution::TableProviderAdapter`].
    ///
    /// Registering a table counts as a DDL statement: after
    /// [`CATALOG_SAVE_THRESHOLD`] DDL statements the catalog is persisted
    /// eagerly, so a crash does not lose an unbounded amount of DDL (R-16).
    pub fn register_table(
        &self,
        schema: hatp_frontend::schema::TableSchema,
    ) -> Result<Arc<hatp_frontend::execution::TableProviderAdapter>> {
        // The Catalog already contains the table definition via FrontendSession.
        // We just need to register the table with the engine for DataFusion.
        let adapter = self
            .inner
            .frontend
            .register_table(schema, Arc::clone(&self.inner.engine))
            .map_err(|err| DatabaseError::Frontend(err.to_string()))?;
        self.schedule_catalog_save();
        Ok(adapter)
    }

    /// Counts one DDL statement and persists the catalog once the buffered
    /// count reaches the threshold. Best-effort: a failed persist is logged by
    /// `save_catalog` and the counter is reset anyway (a retry would re-attempt
    /// on the next threshold crossing).
    fn schedule_catalog_save(&self) {
        let count = self
            .inner
            .ddl_count_since_save
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        if count >= CATALOG_SAVE_THRESHOLD {
            self.inner.ddl_count_since_save.store(0, Ordering::Relaxed);
            save_catalog(self.inner.engine.path(), &self.inner.catalog);
        }
    }

    /// Persists the catalog immediately (the public entry point for the eager
    /// DDL-save threshold and for callers that want a deterministic save
    /// before a checkpoint).
    pub fn save_catalog(&self) {
        let _ = save_catalog(self.inner.engine.path(), &self.inner.catalog);
    }

    /// Execute a SQL query against the embedded DataFusion context. The
    /// underlying `TableProviderAdapter`s serve scans from the engine.
    pub async fn execute_sql(
        &self,
        query: &str,
    ) -> Result<hatp_frontend::execution::SessionExecutionOutcome> {
        self.inner
            .frontend
            .execute(query)
            .await
            .map_err(|err| DatabaseError::Frontend(err.to_string()))
    }
}

/// Buffered snapshot transaction.
#[derive(Debug)]
pub struct Transaction {
    database: Database,
    snapshot_ts: u64,
    mutations: Vec<Mutation>,
    /// Reserved transaction id for SSI transactions; `None` for plain SI.
    reserved_tx_id: Option<TxnId>,
    /// `true` if this is an SSI transaction (writes are validated against
    /// concurrent SSI contexts via the engine's pre-commit hook).
    is_ssi: bool,
    /// Shared transaction manager reference for SSI bookkeeping.
    tx_manager: Option<Arc<TxManager>>,
    /// Pins this transaction's snapshot against GC/compaction so historical
    /// MVCC versions stay observable for the transaction's lifetime. The field
    /// is never read directly — it exists purely for its `Drop` side effect
    /// (releasing the pin).
    _snapshot_guard: SnapshotGuard,
}

impl Transaction {
    /// Returns this transaction's stable snapshot timestamp.
    #[must_use]
    pub fn snapshot_ts(&self) -> u64 {
        self.snapshot_ts
    }

    /// Returns the reserved transaction id for SSI transactions, or
    /// `None` for plain SI. Exposed so callers (tests, debug tools) can
    /// correlate a `Transaction` with its `TxManager` state.
    #[must_use]
    pub fn reserved_tx_id(&self) -> Option<TxnId> {
        self.reserved_tx_id
    }

    /// Returns `true` if this transaction runs under SSI conflict detection.
    #[must_use]
    pub fn is_ssi(&self) -> bool {
        self.is_ssi
    }

    /// SSI write-set bookkeeping shared by `put` and `delete`.
    fn record_ssi_write(&self, key: &Bytes) {
        if !self.is_ssi {
            return;
        }
        if let (Some(manager), Some(id)) = (&self.tx_manager, self.reserved_tx_id) {
            // Mutate under the SSI context lock (`with_ssi_txn`), NOT via the
            // get→clone→mutate→update pattern — the latter is a TOCTOU race
            // (R-04) where two threads each clone, mutate, and the last write
            // silently drops the other's write.
            let _ = manager.with_ssi_txn(id, |ssi| ssi.record_write(key.clone()));
        }
    }

    /// Buffers a put and makes it immediately visible to this transaction.
    pub fn put(&mut self, key: impl Into<Bytes>, value: impl Into<Bytes>) {
        let key = key.into();
        self.record_ssi_write(&key);
        self.mutations.push(Mutation::Put {
            key,
            value: value.into(),
        });
    }

    /// Buffers a delete and makes it immediately visible to this transaction.
    pub fn delete(&mut self, key: impl Into<Bytes>) {
        let key = key.into();
        self.record_ssi_write(&key);
        self.mutations.push(Mutation::Delete { key });
    }

    /// Reads buffered writes first, then this transaction's stable snapshot.
    ///
    /// The buffered-mutation scan is a linear reverse walk — O(n) per
    /// `get` call.  OLTP transactions typically buffer O(1)–O(10) writes,
    /// and for sets this small a cache-friendly linear scan is faster than
    /// hashing each key (same reasoning as `hatp_tx::SsiTxn::read_set`).
    /// If long-running transactions with hundreds of buffered writes ever
    /// become common, switch the mutation buffer to `HashMap<Bytes, Mutation>`.
    pub fn get(&self, key: &[u8]) -> Result<Option<Bytes>> {
        for mutation in self.mutations.iter().rev() {
            match mutation {
                Mutation::Put {
                    key: candidate,
                    value,
                } if candidate.as_ref() == key => {
                    return Ok(Some(value.clone()));
                }
                Mutation::Delete { key: candidate } if candidate.as_ref() == key => {
                    return Ok(None);
                }
                Mutation::Put { .. } | Mutation::Delete { .. } => {}
            }
        }
        // SSI read tracking: record the read key in the SSI context so the
        // pre-commit hook can detect rw-antidependency violations. Plain SI
        // transactions skip this bookkeeping. Mutate under the SSI context
        // lock to avoid the R-04 TOCTOU race.
        if self.is_ssi {
            if let (Some(manager), Some(id)) = (&self.tx_manager, self.reserved_tx_id) {
                let _ =
                    manager.with_ssi_txn(id, |ssi| ssi.record_read(Bytes::copy_from_slice(key)));
            }
        }
        self.database.get_at(key, self.snapshot_ts)
    }

    /// Scans the visible key/value pairs in `[lower, upper)` at this
    /// transaction's snapshot, recording the range as an SSI read-conflict
    /// range (PR 3.8). A concurrent transaction committing a write inside
    /// this range after this transaction began is then detected at commit
    /// time as a read-write antidependency (phantom).
    ///
    /// Buffered writes are folded into the result (read-your-writes, matching
    /// [`Transaction::get`]): a buffered `Put` overwrites the scanned value for
    /// its key and a buffered `Delete` removes it.
    pub fn scan_range(&self, lower: &[u8], upper: &[u8]) -> Result<Vec<(Bytes, Bytes)>> {
        // SSI range-read tracking: record the read range in the SSI context
        // so the pre-commit hook can detect phantom antidependencies.
        if self.is_ssi {
            if let (Some(manager), Some(id)) = (&self.tx_manager, self.reserved_tx_id) {
                let _ = manager.with_ssi_txn(id, |ssi| {
                    ssi.add_read_conflict_range(
                        Bytes::copy_from_slice(lower),
                        Bytes::copy_from_slice(upper),
                    );
                });
            }
        }
        let mut kvs = self
            .database
            .inner
            .engine
            .scan_range(lower, upper, self.snapshot_ts)
            .map_err(DatabaseError::Engine)?;
        // Fold buffered writes (read-your-writes), same as `get`. Iterated in
        // order so a later mutation of the same key wins.
        for mutation in &self.mutations {
            match mutation {
                Mutation::Put { key, value } if key.as_ref() >= lower && key.as_ref() < upper => {
                    if let Some(existing) = kvs.iter_mut().find(|(k, _)| k.as_ref() == key.as_ref())
                    {
                        existing.1 = value.clone();
                    } else {
                        kvs.push((key.clone(), value.clone()));
                    }
                }
                Mutation::Delete { key } if key.as_ref() >= lower && key.as_ref() < upper => {
                    kvs.retain(|(k, _)| k.as_ref() != key.as_ref());
                }
                Mutation::Put { .. } | Mutation::Delete { .. } => {}
            }
        }
        // Folded writes may push keys out of the engine's sorted order; restore
        // key order so the result matches a plain scan's ordering contract.
        kvs.sort_by(|(left, _), (right, _)| left.cmp(right));
        Ok(kvs)
    }

    /// Returns the number of buffered mutations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.mutations.len()
    }

    /// Returns whether this transaction contains no writes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mutations.is_empty()
    }

    /// Atomically commits all buffered mutations.
    ///
    /// For SSI transactions, returns [`DatabaseError::SsiConflict`] on
    /// pre-commit conflict detection; `self` is consumed in this case so
    /// the caller cannot retry. Callers needing retry semantics should use
    /// [`Transaction::try_commit`] instead.
    ///
    /// # Errors
    ///
    /// Returns [`DatabaseError::ReadOnlyCommit`] if there are no buffered
    /// writes, [`DatabaseError::SsiConflict`] on SSI conflict, or any
    /// underlying [`DatabaseError::Engine`].
    pub fn commit(self) -> Result<u64> {
        self.try_commit().map_err(|(_, err)| err)
    }

    /// Atomically commits all buffered mutations, returning the commit
    /// sequence number (`commit_ts`) on success. On failure, returns
    /// `(self, DatabaseError)` so SSI callers can inspect or retry without
    /// losing state.
    ///
    /// # Errors
    ///
    /// Returns `(Transaction, DatabaseError)` carrying:
    /// - [`DatabaseError::ReadOnlyCommit`] if there are no buffered writes;
    /// - [`DatabaseError::SsiConflict`] if the pre-commit hook rejected the
    ///   batch on write-write conflict;
    /// - [`DatabaseError::Engine`] for any other engine failure.
    ///
    /// The `Err` variant is intentionally unboxed: it returns the original
    /// transaction so SSI callers can retry without re-staging reads/writes.
    /// The size cost (~176 bytes) is paid only on the error path, which is
    /// rare; boxing it would add an allocation and an extra indirection on
    /// every retry for no measurable win. This is why the
    /// `clippy::result_large_err` lint is allowed here.
    #[allow(clippy::result_large_err)]
    pub fn try_commit(&self) -> std::result::Result<u64, (Transaction, DatabaseError)> {
        if self.mutations.is_empty() {
            return Err((self.clone_for_retry(), DatabaseError::ReadOnlyCommit));
        }
        let commit_result = match self.reserved_tx_id {
            Some(id) => self
                .database
                .inner
                .engine
                .write_with_tx(id.0, &self.mutations),
            None => self.database.inner.engine.write(&self.mutations),
        };
        match commit_result {
            Ok(commit_ts) => Ok(commit_ts),
            Err(EngineError::WriteConflict { txn, key }) => {
                Err((self.clone_for_retry(), DatabaseError::SsiConflict { txn, key }))
            }
            Err(EngineError::ReadWriteConflict { txn, key }) => Err((
                self.clone_for_retry(),
                DatabaseError::SsiReadWriteConflict { txn, key },
            )),
            Err(other) => Err((self.clone_for_retry(), DatabaseError::Engine(other))),
        }
    }

    /// Internal helper: rebuild a Transaction that owns identical state.
    /// Used by `try_commit` so SSI callers can recover the original
    /// transaction on error.
    fn clone_for_retry(&self) -> Transaction {
        Transaction {
            database: self.database.clone(),
            snapshot_ts: self.snapshot_ts,
            mutations: self.mutations.clone(),
            reserved_tx_id: self.reserved_tx_id,
            is_ssi: self.is_ssi,
            tx_manager: self.tx_manager.clone(),
            _snapshot_guard: self.database.inner.engine.pin_snapshot(self.snapshot_ts),
        }
    }

    /// Explicitly abandons all buffered writes.
    ///
    /// Because the WAL is append-only, discarding the in-memory buffer is the
    /// only correct action — there is nothing to "undo" on disk.  SSI context
    /// cleanup happens automatically in [`Transaction::drop`], so the caller
    /// does not need to call `rollback` before dropping the transaction.
    pub fn rollback(self) {}
}

impl Drop for Transaction {
    /// Releases any SSI context this transaction holds in the shared
    /// `TxManager`.
    ///
    /// Without this, a Transaction that was neither committed nor
    /// aborted leaked its `tx_manager.ssi_contexts` entry. The leaked
    /// `write_set` then caused **ghost conflicts**: a future SSI
    /// transaction whose `validate_ssi` saw the dead entry would abort
    /// even though the original transaction was gone. The integration
    /// test `transaction_ssi_abort_release` previously had to call
    /// `manager.abort(t1_id)` by hand to avoid the leak.
    ///
    /// Safe to run on committed/aborted transactions: `TxManager::abort`
    /// returns `UnknownTxn` for ids that have already been removed, and
    /// we discard the result.
    fn drop(&mut self) {
        if let (Some(manager), Some(id)) = (&self.tx_manager, self.reserved_tx_id) {
            let _ = manager.abort(id);
        }
    }
}

/// Loads the persisted catalog from `path/catalog.json`, or returns an empty
/// catalog when no snapshot exists yet (first open).
///
/// The engine's data directory is the single source of truth: the catalog
/// snapshot lives next to `hatp.wal` / `MANIFEST` / SST files, so a reopen
/// of the same path restores the DDL (tables, primary keys, index metadata)
/// that was defined in a previous process.
fn load_catalog(path: &Path) -> Result<Catalog> {
    let file = path.join("catalog.json");
    let bytes = match std::fs::read(&file) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Catalog::new()),
        Err(err) => return Err(DatabaseError::Engine(err.into())),
    };
    let snapshot: CatalogSnapshot = serde_json::from_slice(&bytes)
        .map_err(|err| DatabaseError::Frontend(format!("catalog decode failed: {err}")))?;
    Catalog::from_snapshot(snapshot)
        .map_err(|err| DatabaseError::Frontend(format!("catalog apply failed: {err}")))
}

/// Best-effort persistence of the catalog to `path/catalog.json` using an
/// atomic tmp+rename with fsync. Returns `true` when the catalog was durably
/// written, `false` on failure (logged and swallowed: the engine data is
/// already durable, and the catalog can be reconstructed by replaying DDL).
fn save_catalog(path: &Path, catalog: &Catalog) -> bool {
    use std::io::Write;

    let snapshot = catalog.to_snapshot();
    let json = match serde_json::to_vec(&snapshot) {
        Ok(json) => json,
        Err(err) => {
            tracing::error!(error = %err, "catalog serialization failed; skipping persist");
            return false;
        }
    };
    let file = path.join("catalog.json");
    let tmp = path.join("catalog.json.tmp");
    let result = (|| -> std::io::Result<()> {
        let mut handle = std::fs::File::create(&tmp)?;
        handle.write_all(&json)?;
        // fsync before rename so a power loss cannot publish an empty file.
        handle.sync_all()?;
        std::fs::rename(&tmp, &file)?;
        Ok(())
    })();
    match result {
        Ok(()) => true,
        Err(err) => {
            tracing::error!(error = %err, "catalog persist failed");
            false
        }
    }
}
