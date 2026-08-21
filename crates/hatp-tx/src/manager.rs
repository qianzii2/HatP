//! Transaction manager — global sequence allocator and lifecycle state machine.
//!
//! This module provides the transaction state machine for HatP. It uses
//! Serializable Snapshot Isolation (SSI) semantics: reads see all commits up
//! to the transaction's start timestamp, and `validate_ssi` rejects
//! transactions whose read- or write-set conflicts with a peer that committed
//! after this transaction began.
//!
//! # SSI implementation status
//!
//! SSI is **fully wired** through [`TxManagerHook`] →
//! [`hatp_engine::EngineHook::on_pre_commit`]:
//!
//! - [`hatp_engine::Engine::write_with_tx`] calls `on_pre_commit` before
//!   WAL append; `TxManagerHook` delegates to [`TxManager::validate_ssi`].
//! - A [`TxError::WriteConflict`] or [`TxError::ReadWriteConflict`] from
//!   `validate_ssi` is converted to `hatp_engine::EngineError::WriteConflict`
//!   or `EngineError::ReadWriteConflict` and surfaced to the caller
//!   (`hatp::Transaction::try_commit`).
//! - [`TxManager::abort`] is called from [`hatp::Transaction::drop`].
//!
//! Conflict detection:
//! - **Write-write**: the peer's write-set overlaps this txn's write-set.
//! - **Read-write (antidependency)**: a peer committed a write to a key this
//!   txn read, *after* this txn began. This catches write-skew anomalies
//!   that pure write-write checks miss.

use crate::error::{Result, TxError};
use bytes::Bytes;
use parking_lot::Mutex;
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::Arc;

/// Stable transaction identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TxnId(pub u64);

/// Isolation level requested when a transaction begins.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum IsolationHint {
    /// All reads use the transaction's begin snapshot. Snapshot Isolation
    /// semantics — writes never conflict with concurrent reads.
    #[default]
    Snapshot,
}

/// Transaction lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnState {
    /// The transaction may read and buffer writes.
    Active,
    /// The commit record is durable.
    Committed,
    /// The transaction has been rolled back.
    Aborted,
}

/// Immutable transaction metadata returned to callers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TxnHandle {
    /// Allocated transaction identifier.
    pub id: TxnId,
    /// Requested isolation mode.
    pub isolation: IsolationHint,
    /// Current lifecycle state.
    pub state: TxnState,
}

/// SSI transaction context with read/write sets for conflict detection.
///
/// # Design note: `HashSet` for read/write sets
///
/// Conflict detection is O(membership-test) against the commit history, and a
/// long-running analytical transaction can touch tens of thousands of keys.
/// `Vec::contains` is O(n) per key — the "performance cliff" tracked as R-22.
/// `HashSet<Bytes>` makes `record_read` / `record_write` / `validate_ssi`
/// membership tests O(1) for all transaction sizes, at a small constant-factor
/// cost for the O(1)–O(10) OLTP case.
///
/// # Design note: Read-Set Tracking (RW Antidependency)
///
/// The `read_set` is populated by `record_read()` and **is checked** by
/// `validate_ssi()` for read-write antidependency violations: a peer
/// committed a write to a key in this transaction's read set, after this
/// transaction began. This catches the classic write-skew anomaly under SSI.
#[derive(Debug, Clone)]
pub struct SsiTxn {
    /// Transaction handle.
    pub handle: TxnHandle,
    /// Start timestamp (snapshot point, in the engine's commit-order sequence).
    pub start_ts: TxnTs,
    /// Keys read by this transaction. See struct-level note about RW-antidependency.
    pub read_set: HashSet<Bytes>,
    /// Keys written by this transaction.
    pub write_set: HashSet<Bytes>,
    /// Key ranges (`[lower, upper)`) read by this transaction via a range scan
    /// (PR 3.8, FDB conflict range). A peer committing a write to any key
    /// inside one of these ranges *after* this transaction began is a
    /// read-write antidependency (phantom), mirroring the point-key
    /// `read_set` check.
    pub read_conflict_ranges: Vec<(Bytes, Bytes)>,
}

impl SsiTxn {
    /// Creates a new SSI transaction.
    pub fn new(handle: TxnHandle, start_ts: TxnTs) -> Self {
        Self {
            handle,
            start_ts,
            read_set: HashSet::new(),
            write_set: HashSet::new(),
            read_conflict_ranges: Vec::new(),
        }
    }

    /// Records a key read by this transaction.
    pub fn record_read(&mut self, key: Bytes) {
        self.read_set.insert(key);
    }

    /// Records a key written by this transaction.
    pub fn record_write(&mut self, key: Bytes) {
        self.write_set.insert(key);
    }

    /// Records a key range (`[lower, upper)`) read by this transaction (PR 3.8).
    /// A later peer commit writing inside this range is a read-write
    /// antidependency.
    pub fn add_read_conflict_range(&mut self, lower: Bytes, upper: Bytes) {
        self.read_conflict_ranges.push((lower, upper));
    }

    /// Checks if this transaction has a write-write conflict with another.
    /// Returns true if there is a conflict (this txn should abort).
    ///
    /// # Usage Context
    ///
    /// This method is provided for **explicit user-level conflict checking** and
    /// testing purposes. It compares two in-memory `SsiTxn` contexts directly.
    ///
    /// The **production SSI validation path** uses `TxManager::validate_ssi()`,
    /// which checks against `commit_history` (committed transactions), NOT active
    /// peers. This distinction matters:
    ///
    /// - `has_write_conflict(txn1, txn2)` — compares two active transactions
    /// - `validate_ssi(txn)` — checks txn against committed write-sets
    ///
    /// Both mechanisms are correct; they serve different isolation levels.
    /// The commit-history approach implements **first-committer-wins** for
    /// concurrent SSI transactions. Direct peer comparison would implement
    /// **first-writer-wins**, which can cause mutual aborts.
    pub fn has_write_conflict(&self, other: &SsiTxn) -> bool {
        // Write-write conflict: both txn wrote the same key. O(1) membership
        // per key over a `HashSet`.
        self.write_set
            .iter()
            .any(|my_key| other.write_set.contains(my_key))
    }
}

/// Timestamp type — the single source of truth is `hatp_types::TxnTs`.
///
/// Re-exported here (rather than `pub type TxnTs = hatp_engine::TxnTs`) so the
/// transaction layer depends directly on the shared crate instead of going
/// through the engine's own re-export; `hatp_engine` re-exports the same type,
/// so all three spellings are one `u64` alias.
pub use hatp_types::TxnTs;

/// Owned entry stored in the commit history. Carries the committed
/// transaction's write-set and the engine-allocated `commit_ts` used by
/// `validate_ssi` to decide whether a peer committed *after* the validating
/// transaction began.
#[derive(Debug, Clone)]
struct CommitEntry {
    /// Keys written by the committing transaction.
    write_set: HashSet<Bytes>,
    /// Engine-allocated commit sequence number at the moment of commit
    /// (commit order, strictly increasing). Used to decide whether a peer
    /// committed *after* the validating transaction began.
    commit_ts: u64,
}

/// Bounded commit history for SSI conflict detection.
///
/// Entries are appended in commit order and garbage-collected against the
/// active-transaction watermark: an entry whose `commit_ts <= watermark` can
/// never again conflict (the WW/RW checks both require `commit_ts >
/// txn.start_ts`, and `watermark == min(active start_ts)`), so it is dropped.
/// This replaces the old fixed-cap `BTreeMap` whose `COMMIT_HISTORY_CAP == 1024`
/// eviction could drop a still-relevant entry and let a long transaction's
/// conflict go undetected (R-03).
#[derive(Debug, Default)]
struct CommitHistory {
    ring: Mutex<VecDeque<CommitEntry>>,
}

impl CommitHistory {
    /// Appends a committed write-set in commit order.
    fn push(&self, entry: CommitEntry) {
        self.ring.lock().push_back(entry);
    }

    /// Drops every entry whose `commit_ts <= watermark`. Returns the number
    /// dropped. Called after each push so the history stays window-bounded.
    fn garbage_collect(&self, watermark: u64) -> usize {
        let mut ring = self.ring.lock();
        let before = ring.len();
        // Entries are ordered by commit_ts (commit order == insertion order),
        // so all removable entries sit at the front.
        while ring.front().is_some_and(|e| e.commit_ts <= watermark) {
            ring.pop_front();
        }
        before.saturating_sub(ring.len())
    }
}

/// Owns the transaction state table. The manager is **not** an id allocator:
/// transaction ids are allocated by the engine (`Engine::reserve_tx_id`) and
/// passed in explicitly, so the tx_id space has exactly one source of truth.
#[derive(Debug)]
pub struct TxManager {
    /// Live transactions keyed by id.
    transactions: Mutex<BTreeMap<TxnId, TxnHandle>>,
    /// SSI transaction contexts (only for Active transactions).
    ssi_contexts: Mutex<BTreeMap<TxnId, SsiTxn>>,
    /// Write-sets of committed SSI transactions, used by `validate_ssi` for
    /// both first-committer-wins (WW) and read-write antidependency (RW)
    /// conflict detection.
    ///
    /// `validate_ssi` consults ONLY this table. Two SSI peers that are merely
    /// *staged* do NOT conflict with each other — the `Engine::write_guard`
    /// already serialises pre-commit, so the survivor is determined by commit
    /// order. The first to commit promotes its write-set here; the second's
    /// `validate_ssi` then sees the conflict and aborts.
    ///
    /// Entries are garbage-collected by watermark (see [`CommitHistory`]);
    /// memory is bounded by the active-transaction window, not a fixed cap.
    commit_history: CommitHistory,
    /// Group-commit provisional staging: write-sets of transactions that passed
    /// SSI validation but whose WAL batch has not yet been flushed.
    /// `validate_ssi` also checks these so two conflicting transactions
    /// coalesced into one batch still resolve first-committer-wins. On WAL
    /// success `commit_at` promotes them durably; on WAL failure
    /// [`TxManager::discard_staged`] drops them and the transactions stay
    /// Active so the caller can retry.
    provisional: Mutex<Vec<CommitEntry>>,
}

impl Default for TxManager {
    fn default() -> Self {
        Self {
            transactions: Mutex::new(BTreeMap::new()),
            ssi_contexts: Mutex::new(BTreeMap::new()),
            commit_history: CommitHistory::default(),
            provisional: Mutex::new(Vec::new()),
        }
    }
}

impl TxManager {
    /// Creates a fresh manager.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Begins a transaction with an engine-allocated `id` and an explicit
    /// isolation mode. The manager does **not** allocate ids — the engine's
    /// [`hatp_engine::Engine::reserve_tx_id`] is the single source of truth
    /// (R-07). Callers that need to allocate an id must do so upstream.
    ///
    /// Returns the transaction handle.
    #[must_use]
    pub fn begin_with_id(&self, id: TxnId, isolation: IsolationHint) -> TxnHandle {
        let handle = TxnHandle {
            id,
            isolation,
            state: TxnState::Active,
        };
        self.transactions.lock().insert(id, handle);
        handle
    }

    /// Begins an SSI transaction with an engine-allocated `id` and an
    /// explicit `start_ts` (snapshot read point).
    ///
    /// The production path (`hatp::Database::begin_ssi`) uses this so the
    /// SSI context key matches the id the engine will use for WAL `tx_id`,
    /// and `start_ts` is the engine's commit-order snapshot — **not** the
    /// id. The manager does not allocate ids or timestamps; both are
    /// supplied by the engine, keeping the two sequences (reservation order
    /// vs commit order) strictly separated.
    pub fn begin_ssi_with_id(&self, id: TxnId, start_ts: TxnTs) -> SsiTxn {
        let handle = self.begin_with_id(id, IsolationHint::Snapshot);
        let txn = SsiTxn::new(handle, start_ts);
        self.ssi_contexts.lock().insert(id, txn.clone());
        txn
    }

    /// Gets the SSI context for a transaction.
    pub fn get_ssi_txn(&self, id: TxnId) -> Option<SsiTxn> {
        self.ssi_contexts.lock().get(&id).cloned()
    }

    /// Updates the SSI context for a transaction.
    pub fn update_ssi_txn(&self, txn: &SsiTxn) {
        let mut contexts = self.ssi_contexts.lock();
        contexts.insert(txn.handle.id, txn.clone());
    }

    /// Runs `f` with exclusive access to the SSI context for `id`, if it
    /// exists. This replaces the get→clone→mutate→update pattern, which had a
    /// TOCTOU race (R-04): two threads doing `get_ssi_txn` + `update_ssi_txn`
    /// each clone the same context, mutate their private copy, and the last
    /// `update_ssi_txn` silently drops the other's writes.
    ///
    /// Returns [`TxError::UnknownTxn`] when no active SSI context exists for
    /// `id` (e.g. the transaction already committed or aborted).
    pub fn with_ssi_txn<R>(&self, id: TxnId, f: impl FnOnce(&mut SsiTxn) -> R) -> Result<R> {
        let mut contexts = self.ssi_contexts.lock();
        let ssi = contexts
            .get_mut(&id)
            .ok_or(TxError::UnknownTxn { txn: id })?;
        Ok(f(ssi))
    }

    /// Validates a SSI transaction. The validator consults ONLY the
    /// committed-history table — peers that haven't yet committed are
    /// invisible to it. This is exactly what first-committer-wins
    /// requires:
    ///
    /// 1. Two SSI transactions are both staged with overlapping
    ///    write-sets. Neither one has committed yet, so neither sees
    ///    the other in `commit_history`. Both reach pre-commit; the
    ///    first wins because the `Engine::write_guard` serialises
    ///    pre-commit and the loser simply runs `validate_ssi` *after*
    ///    the winner already promoted its write-set to
    ///    `commit_history`. The loser's `validate_ssi` then sees the
    ///    conflict and aborts.
    ///
    /// 2. A long-running active peer is irrelevant — its effect on a
    ///    later committer is captured the moment it commits and shows
    ///    up in `commit_history`.
    ///
    /// Two conflict classes are checked against the committed history:
    ///
    /// - **Write-write** ([`TxError::WriteConflict`]): the peer's write-set
    ///   overlaps this txn's write-set.
    /// - **Read-write antidependency** ([`TxError::ReadWriteConflict`]):
    ///   a peer committed a write to a key this txn previously read, AND
    ///   that peer committed *after* this txn began (peer.commit_ts >
    ///   this_txn.start_ts). This catches write-skew anomalies that pure
    ///   write-write checks miss.
    ///
    /// Active-vs-active *write-write* detection (the pre-fix behaviour) was
    /// wrong: it produced mutual aborts even though `write_guard` makes one
    /// of them the legitimate winner. PR 3.9 re-introduces active-peer checks
    /// **only** for the Cahill dangerous structure (an rw-cycle), which
    /// `write_guard` cannot resolve — see [`TxManager::validate_cahill`].
    pub fn validate_ssi(&self, txn: &SsiTxn) -> Result<()> {
        let contexts = self.ssi_contexts.lock();
        self.validate_ssi_with_contexts(txn, &contexts)
    }

    /// Validates `txn` against the durable history, provisional staging, and
    /// the given active-context snapshot (PR 3.9). `contexts` is supplied by
    /// the caller so `on_pre_commit` can hold the `ssi_contexts` lock across
    /// `record_write` + validate without re-locking (a plain `Mutex` would
    /// self-deadlock).
    fn validate_ssi_with_contexts(
        &self,
        txn: &SsiTxn,
        contexts: &BTreeMap<TxnId, SsiTxn>,
    ) -> Result<()> {
        // 1. Check the durable commit history.
        {
            let history = self.commit_history.ring.lock();
            for entry in history.iter() {
                Self::check_conflict(txn, entry.commit_ts, &entry.write_set)?;
            }
        }
        // 2. Check group-commit provisional staging: transactions that passed
        //    validation in the same WAL batch but are not yet durable. Their
        //    write-sets are visible to later transactions in the batch so
        //    first-committer-wins holds even when the fsync is coalesced.
        {
            let provisional = self.provisional.lock();
            for entry in provisional.iter() {
                Self::check_conflict(txn, entry.commit_ts, &entry.write_set)?;
            }
        }
        // 3. Cahill dangerous-structure detection (PR 3.9): abort when `txn`
        //    forms an rw-cycle with a still-active peer (see below).
        Self::validate_cahill(txn, contexts)?;
        Ok(())
    }

    /// Pre-commit entry point for the engine hook (PR 3.9): records the
    /// mutation keys into the SSI write-set and validates, all under one
    /// `ssi_contexts` lock acquisition. This closes the R-04 TOCTOU (write-set
    /// must be final before validation) **and** avoids re-locking `ssi_contexts`
    /// inside `validate_cahill` (which would self-deadlock on a plain `Mutex`).
    pub fn record_write_and_validate(&self, id: TxnId, keys: &[bytes::Bytes]) -> Result<()> {
        let mut contexts = self.ssi_contexts.lock();
        {
            let Some(txn) = contexts.get_mut(&id) else {
                // Context vanished. Distinguish "concurrently aborted" (which
                // must be rejected, not silently downgraded to a non-SSI commit)
                // from "never was an SSI transaction" (UnknownTxn → the hook
                // lets the durable autocommit path proceed).
                drop(contexts);
                if let Ok(handle) = self.get(id) {
                    if handle.state == TxnState::Aborted {
                        return Err(TxError::InvalidState {
                            txn: id,
                            expected: TxnState::Active,
                            actual: TxnState::Aborted,
                        });
                    }
                }
                return Err(TxError::UnknownTxn { txn: id });
            };
            for key in keys {
                txn.record_write(key.clone());
            }
        }
        let txn = contexts
            .get(&id)
            .ok_or(TxError::UnknownTxn { txn: id })?;
        let result = self.validate_ssi_with_contexts(txn, &contexts);
        // A rejected commit is terminal for SSI: remove the context and mark
        // the transaction Aborted so a symmetric rw-cycle (two committers each
        // seeing the other still-active) cannot livelock by mutually rejecting
        // forever. The caller's `try_commit` returns the transaction for
        // retry, but the retry is rejected via `on_pre_commit`'s Aborted check
        // (see `hatp-tx::hook`), never silently downgraded to a non-SSI commit.
        if result.is_err() {
            // Terminal abort, in the SAME lock order as `TxManager::abort`
            // (transactions first, then ssi_contexts): mark Aborted under the
            // transactions lock BEFORE removing the SSI context. Reversing the
            // order opens a window where a concurrent retry sees no context and
            // is silently downgraded to a non-SSI commit (security-review HIGH).
            drop(contexts);
            let mut transactions = self.transactions.lock();
            if let Some(handle) = transactions.get_mut(&id) {
                handle.state = TxnState::Aborted;
            }
            drop(transactions);
            self.ssi_contexts.lock().remove(&id);
        }
        result
    }

    /// Cahill dangerous-structure detection (PR 3.9).
    ///
    /// Two concurrent transactions `T1` (active) and `T2` (committing) form a
    /// dangerous structure when both rw-edges exist:
    ///
    /// - `T1 → T2`: `T2` wrote a key `T1` read (point or range), so `T2` must
    ///   follow `T1` in any serial order.
    /// - `T2 → T1`: `T1` wrote a key `T2` read (point or range), so `T1` must
    ///   follow `T2`.
    ///
    /// A cycle (which can involve 2+ active peers, not only 2) has no serial
    /// order, so the committer is aborted. This is the rw-cycle that plain
    /// first-committer rw-antidependency checks miss when no transaction has
    /// committed yet — the exact gap `write_guard` (which only serialises
    /// write-write) cannot close.
    ///
    /// Implementation: BFS over the rw-antidependency graph restricted to
    /// active peers in `contexts`. We start from `txn`, follow edges `A → B`
    /// (meaning "A wrote a key B reads"), and abort if we can reach `txn`
    /// again — that path plus `txn → … → txn` is a cycle. A 2-cycle shows up
    /// as `txn → peer → txn`; longer cycles (e.g. T1 → T2 → T3 → T1, which
    /// pairwise checks miss) show up as a longer path.
    fn validate_cahill(txn: &SsiTxn, contexts: &BTreeMap<TxnId, SsiTxn>) -> Result<()> {
        // `ssi_contexts` only holds Active transactions (commit/abort remove
        // the entry), so no per-peer state guard is needed here.
        // Edge `a → b`: a's write_set overlaps b's read set (point or range).
        let outgoing = |from: &SsiTxn| -> Vec<TxnId> {
            contexts
                .iter()
                .filter_map(|(peer_id, peer)| {
                    if *peer_id == from.handle.id {
                        return None;
                    }
                    if Self::write_set_hits_reads(
                        &from.write_set,
                        &peer.read_set,
                        &peer.read_conflict_ranges,
                    ) {
                        Some(*peer_id)
                    } else {
                        None
                    }
                })
                .collect()
        };

        let mut visited: HashSet<TxnId> = HashSet::new();
        visited.insert(txn.handle.id);
        let mut frontier: Vec<TxnId> = outgoing(txn);
        while let Some(next) = frontier.pop() {
            if next == txn.handle.id {
                // Cycle reaches the committer → dangerous structure.
                // Pick a representative key from `txn`'s write_set that hit
                // some peer's read set (or range) so the error is actionable.
                let key = txn
                    .write_set
                    .iter()
                    .find(|k| {
                        contexts.values().any(|peer| {
                            peer.read_set.contains(*k)
                                || peer
                                    .read_conflict_ranges
                                    .iter()
                                    .any(|(lo, hi)| {
                                        k.as_ref() >= lo.as_ref() && k.as_ref() < hi.as_ref()
                                    })
                        })
                    })
                    .cloned()
                    .unwrap_or_default();
                return Err(TxError::ReadWriteConflict {
                    txn: txn.handle.id,
                    key: key.to_vec(),
                });
            }
            if visited.insert(next) {
                if let Some(peer) = contexts.get(&next) {
                    frontier.extend(outgoing(peer));
                }
            }
        }
        Ok(())
    }

    /// Returns `true` when any key in `write_set` is read by the given point
    /// `read_set` or falls inside any recorded `read_ranges` (`[lower, upper)`).
    fn write_set_hits_reads(
        write_set: &HashSet<Bytes>,
        read_set: &HashSet<Bytes>,
        read_ranges: &[(Bytes, Bytes)],
    ) -> bool {
        write_set.iter().any(|key| {
            read_set.contains(key)
                || read_ranges
                    .iter()
                    .any(|(lo, hi)| key.as_ref() >= lo.as_ref() && key.as_ref() < hi.as_ref())
        })
    }

    /// Checks one peer entry (durable or provisional) against `txn`'s write/read
    /// sets. Only peers that committed AFTER `txn` began (`commit_ts >
    /// txn.start_ts`) can conflict: a peer committed at `commit_ts <= start_ts` is
    /// part of `txn`'s snapshot and is a legitimate base for its writes, not a
    /// concurrent conflict. This applies to BOTH write-write and read-write.
    fn check_conflict(txn: &SsiTxn, commit_ts: u64, write_set: &HashSet<Bytes>) -> Result<()> {
        if commit_ts <= txn.start_ts {
            return Ok(());
        }
        // Write-write: peer's write-set overlaps this txn's write-set.
        for my_key in &txn.write_set {
            if write_set.contains(my_key) {
                return Err(TxError::WriteConflict {
                    txn: txn.handle.id,
                    key: my_key.to_vec(),
                });
            }
        }
        // Read-write antidependency: a peer committed a write to a key this txn
        // read, after this txn began. Catches write-skew.
        for my_key in &txn.read_set {
            if write_set.contains(my_key) {
                return Err(TxError::ReadWriteConflict {
                    txn: txn.handle.id,
                    key: my_key.to_vec(),
                });
            }
        }
        // Read-write antidependency over range reads (PR 3.8, FDB conflict
        // range): a peer committed a write to a key inside a range this txn
        // scanned, after this txn began. Catches phantoms. If the peer's write
        // also lands in `read_set` or `write_set`, the earlier checks already
        // returned; this branch only needs to test the recorded ranges.
        for (lower, upper) in &txn.read_conflict_ranges {
            for peer_key in write_set {
                if peer_key.as_ref() >= lower.as_ref() && peer_key.as_ref() < upper.as_ref() {
                    return Err(TxError::ReadWriteConflict {
                        txn: txn.handle.id,
                        key: peer_key.to_vec(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Returns the current metadata for a known transaction.
    pub fn get(&self, id: TxnId) -> Result<TxnHandle> {
        self.transactions
            .lock()
            .get(&id)
            .copied()
            .ok_or(TxError::UnknownTxn { txn: id })
    }

    /// Group commit: after validation passes but before the WAL batch is
    /// flushed, records the transaction's write-set in provisional staging so
    /// later transactions in the same batch can see it via
    /// [`TxManager::validate_ssi`] (first-committer-wins). `commit_ts` is the
    /// provisional commit-order sequence. Returns [`TxError::UnknownTxn`] when
    /// no active SSI context exists for `id`.
    pub fn stage_commit(&self, id: TxnId, commit_ts: u64) -> Result<()> {
        let ssi = self
            .ssi_contexts
            .lock()
            .get(&id)
            .cloned()
            .ok_or(TxError::UnknownTxn { txn: id })?;
        self.provisional.lock().push(CommitEntry {
            write_set: ssi.write_set,
            commit_ts,
        });
        Ok(())
    }

    /// Group commit: the WAL batch failed, so drop every provisional entry.
    /// The staged transactions stay Active and can be retried (their write-sets
    /// were never promoted to `commit_history`).
    pub fn discard_staged(&self) {
        self.provisional.lock().clear();
    }

    /// Commits `id` and records `commit_ts` as the engine-allocated commit
    /// sequence number. The engine hook (`TxManagerHook::on_tx_commit`) uses
    /// this entry point so the SSI history captures the true commit order —
    /// never the reservation order `tx_id`.
    pub fn commit_at(&self, id: TxnId, commit_ts: u64) -> Result<TxnHandle> {
        let mut transactions = self.transactions.lock();
        let handle = transactions
            .get_mut(&id)
            .ok_or(TxError::UnknownTxn { txn: id })?;

        if handle.state != TxnState::Active {
            return Err(TxError::InvalidState {
                txn: id,
                expected: TxnState::Active,
                actual: handle.state,
            });
        }

        handle.state = TxnState::Committed;
        let result = *handle;

        // Promote the SSI write-set to `commit_history` (so a later SSI
        // transaction's `validate_ssi` can detect first-committer-wins and
        // read-write antidependencies) and drop the active SSI context. The
        // history is watermark-bounded, not fixed-cap.
        drop(transactions);
        {
            let mut contexts = self.ssi_contexts.lock();
            if let Some(ssi) = contexts.remove(&id) {
                self.commit_history.push(CommitEntry {
                    write_set: ssi.write_set,
                    commit_ts,
                });
                // Reclaim entries no transaction can conflict with anymore.
                // watermark = min(remaining active start_ts); u64::MAX when none.
                let watermark = contexts
                    .values()
                    .map(|t| t.start_ts)
                    .min()
                    .unwrap_or(u64::MAX);
                self.commit_history.garbage_collect(watermark);
            }
        }

        Ok(result)
    }

    /// Aborts a transaction that has not reached a terminal state.
    pub fn abort(&self, id: TxnId) -> Result<TxnHandle> {
        let mut transactions = self.transactions.lock();
        let handle = transactions
            .get_mut(&id)
            .ok_or(TxError::UnknownTxn { txn: id })?;

        match handle.state {
            TxnState::Active => {
                handle.state = TxnState::Aborted;
                let result = *handle;

                // Clean up SSI context
                drop(transactions);
                self.ssi_contexts.lock().remove(&id);

                Ok(result)
            }
            actual => Err(TxError::InvalidState {
                txn: id,
                expected: TxnState::Active,
                actual,
            }),
        }
    }

    /// Returns all non-terminal (Active) transactions in identifier order.
    #[must_use]
    pub fn inflight(&self) -> Vec<TxnHandle> {
        self.transactions
            .lock()
            .values()
            .filter(|handle| handle.state == TxnState::Active)
            .copied()
            .collect()
    }

    /// Returns all active SSI transactions for conflict detection.
    #[must_use]
    pub fn active_ssi_transactions(&self) -> Vec<SsiTxn> {
        self.ssi_contexts
            .lock()
            .values()
            .filter(|txn| txn.handle.state == TxnState::Active)
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;

    #[test]
    fn lifecycle_reaches_commit() {
        let manager = TxManager::new();
        let transaction = manager.begin_with_id(TxnId(1), IsolationHint::Snapshot);
        assert_eq!(transaction.state, TxnState::Active);
        assert_eq!(
            manager.commit_at(transaction.id, 1).map(|tx| tx.state),
            Ok(TxnState::Committed)
        );
        assert!(manager.inflight().is_empty());
    }

    /// Table-driven: terminal states cannot transition further (Active→Committed→X, Active→Aborted→X)
    #[test]
    fn terminal_state_rejects_further_transitions() {
        let cases: Vec<(&str, TxnState, bool)> = vec![
            ("Committed→commit", TxnState::Committed, false),
            ("Committed→abort", TxnState::Committed, false),
            ("Aborted→commit", TxnState::Aborted, false),
            ("Aborted→abort", TxnState::Aborted, false),
        ];
        for (name, terminal, _expect_ok) in cases {
            let mgr = TxManager::new();
            let id = mgr.begin_with_id(TxnId(1), IsolationHint::Snapshot).id;
            // First transition to terminal state
            if terminal == TxnState::Committed {
                mgr.commit_at(id, 1).expect("first commit");
            } else {
                mgr.abort(id).expect("first abort");
            }
            // Second transition must fail
            let result = if terminal == TxnState::Committed {
                mgr.abort(id).map(|_| ())
            } else {
                mgr.commit_at(id, 1).map(|_| ())
            };
            assert!(result.is_err(), "{name}: expected Err, got Ok");
        }
    }

    #[test]
    fn ssi_txn_records_reads_and_writes() {
        let mgr = TxManager::new();
        let mut txn = mgr.begin_ssi_with_id(TxnId(1), 0);

        txn.record_read(Bytes::from_static(b"key1"));
        txn.record_read(Bytes::from_static(b"key2"));
        txn.record_write(Bytes::from_static(b"key2"));
        txn.record_write(Bytes::from_static(b"key3"));

        assert_eq!(txn.read_set.len(), 2);
        assert_eq!(txn.write_set.len(), 2);
        assert!(txn.read_set.contains(&Bytes::from_static(b"key1")));
        assert!(txn.write_set.contains(&Bytes::from_static(b"key3")));
    }

    #[test]
    fn ssi_write_conflict_detection() {
        let mgr = TxManager::new();

        let mut txn1 = mgr.begin_ssi_with_id(TxnId(1), 0);
        let mut txn2 = mgr.begin_ssi_with_id(TxnId(2), 0);

        txn1.record_write(Bytes::from_static(b"conflict_key"));
        txn2.record_write(Bytes::from_static(b"conflict_key"));

        assert!(txn1.has_write_conflict(&txn2));
        assert!(txn2.has_write_conflict(&txn1));
    }

    #[test]
    fn ssi_no_conflict_different_keys() {
        let mgr = TxManager::new();

        let mut txn1 = mgr.begin_ssi_with_id(TxnId(1), 0);
        let mut txn2 = mgr.begin_ssi_with_id(TxnId(2), 0);

        txn1.record_write(Bytes::from_static(b"key1"));
        txn2.record_write(Bytes::from_static(b"key2"));

        assert!(!txn1.has_write_conflict(&txn2));
        assert!(!txn2.has_write_conflict(&txn1));
    }

    #[test]
    fn ssi_validate_passes_with_no_conflicts() {
        let mgr = TxManager::new();
        let mut txn = mgr.begin_ssi_with_id(TxnId(1), 0);
        txn.record_write(Bytes::from_static(b"unique_key"));
        mgr.update_ssi_txn(&txn);

        let result = mgr.validate_ssi(&txn);
        assert!(result.is_ok());
    }

    #[test]
    fn ssi_validate_first_committer_wins() {
        // Two SSI transactions both stage the same key. Neither has committed
        // yet, so `validate_ssi` returns Ok for both — `Engine::write_guard`
        // serialises pre-commit and the survivor is determined by commit order.
        let mgr = TxManager::new();
        let mut a = mgr.begin_ssi_with_id(TxnId(1), 0);
        a.record_write(Bytes::from_static(b"col"));
        mgr.update_ssi_txn(&a);
        let mut b = mgr.begin_ssi_with_id(TxnId(2), 0);
        b.record_write(Bytes::from_static(b"col"));
        mgr.update_ssi_txn(&b);
        assert!(
            mgr.validate_ssi(&a).is_ok(),
            "staged peer must not conflict pre-commit"
        );
        assert!(
            mgr.validate_ssi(&b).is_ok(),
            "staged peer must not conflict pre-commit"
        );

        // After `a` commits at commit_ts=1 (which is > b's start_ts of 0), its
        // write-set is in `commit_history` and `b`'s subsequent `validate_ssi`
        // reports the write-write conflict.
        mgr.commit_at(a.handle.id, 1).expect("commit a");
        let err = mgr
            .validate_ssi(&b)
            .expect_err("b must now conflict with committed a");
        assert!(matches!(err, TxError::WriteConflict { .. }));
    }

    #[test]
    fn ssi_no_write_conflict_when_peer_committed_before_start() {
        // Regression guard for the WW-condition fix: a peer that committed
        // BEFORE this txn began (commit_ts <= start_ts) is part of this txn's
        // snapshot, so writing the same key is a legitimate overwrite, not a
        // concurrent write-write conflict.
        let mgr = TxManager::new();
        let mut a = mgr.begin_ssi_with_id(TxnId(1), 0);
        a.record_write(Bytes::from_static(b"col"));
        mgr.update_ssi_txn(&a);
        mgr.commit_at(a.handle.id, 1).expect("commit a");

        let mut b = mgr.begin_ssi_with_id(TxnId(2), 5);
        b.record_write(Bytes::from_static(b"col"));
        mgr.update_ssi_txn(&b);
        assert!(
            mgr.validate_ssi(&b).is_ok(),
            "peer committed before start_ts must not produce a write-write conflict"
        );
    }

    #[test]
    fn commit_history_gc_keeps_entries_visible_to_long_txn() {
        // A long-running transaction's read set must keep the committed
        // write-set alive past the point where the old fixed-cap history would
        // have evicted it (R-03). The watermark-bounded history drops only
        // entries with `commit_ts <= min(active start_ts)`.
        let mgr = TxManager::new();
        let mut long = mgr.begin_ssi_with_id(TxnId(10), 0);
        long.record_read(Bytes::from_static(b"col"));
        mgr.update_ssi_txn(&long);

        let mut a = mgr.begin_ssi_with_id(TxnId(1), 1);
        a.record_write(Bytes::from_static(b"col"));
        mgr.update_ssi_txn(&a);
        mgr.commit_at(a.handle.id, 1).expect("commit a");

        // `long` began at 0, so the entry committed at 1 is still visible and
        // must surface as a read-write conflict.
        let err = mgr
            .validate_ssi(&long)
            .expect_err("long txn must still see the committed write");
        assert!(matches!(err, TxError::ReadWriteConflict { .. }));
    }

    #[test]
    fn with_ssi_txn_mutates_under_lock() {
        let mgr = TxManager::new();
        let txn = mgr.begin_ssi_with_id(TxnId(1), 0);
        assert!(txn.write_set.is_empty());
        mgr.with_ssi_txn(TxnId(1), |t| t.record_write(Bytes::from_static(b"k")))
            .expect("context present");
        let updated = mgr.get_ssi_txn(TxnId(1)).expect("context still present");
        assert!(updated.write_set.contains(&Bytes::from_static(b"k")));
    }

    #[test]
    fn ssi_validate_sees_provisional_staging() {
        // Regression guard for the group-commit first-committer-wins bug: two
        // conflicting transactions coalesced into one WAL batch must NOT both
        // pass validation. `stage_commit` records the earlier transaction's
        // write-set in provisional staging, which `validate_ssi` consults, so
        // the later transaction sees the conflict. `discard_staged` (WAL
        // failure) must then release it.
        let mgr = TxManager::new();
        let mut a = mgr.begin_ssi_with_id(TxnId(1), 0);
        a.record_write(Bytes::from_static(b"col"));
        mgr.update_ssi_txn(&a);
        let mut b = mgr.begin_ssi_with_id(TxnId(2), 0);
        b.record_write(Bytes::from_static(b"col"));
        mgr.update_ssi_txn(&b);

        // Before staging, neither sees the other (both still Active).
        assert!(mgr.validate_ssi(&a).is_ok());
        assert!(mgr.validate_ssi(&b).is_ok());

        // Stage `a` at provisional commit_ts = 1: `b` must now see the WW
        // conflict against `a`'s provisional write-set.
        mgr.stage_commit(TxnId(1), 1).expect("stage a");
        let err = mgr
            .validate_ssi(&b)
            .expect_err("b must see a's provisional write-set");
        assert!(matches!(err, TxError::WriteConflict { .. }));

        // Discard (WAL failure): provisional is cleared, `b` validates clean.
        mgr.discard_staged();
        assert!(mgr.validate_ssi(&b).is_ok());
    }

    #[test]
    fn ssi_cahill_dangerous_structure_aborts_committer() {
        // Cahill rw-cycle (PR 3.9): T1 reads X and writes Y; T2 reads Y and
        // writes X. Neither has committed, so `validate_ssi` must detect the
        // T1 → T2 and T2 → T1 edges and abort the committer — `write_guard`
        // (which only serialises write-write) cannot resolve this.
        let mgr = TxManager::new();
        let mut t1 = mgr.begin_ssi_with_id(TxnId(1), 0);
        t1.record_read(Bytes::from_static(b"X"));
        t1.record_write(Bytes::from_static(b"Y"));
        mgr.update_ssi_txn(&t1);
        let mut t2 = mgr.begin_ssi_with_id(TxnId(2), 0);
        t2.record_read(Bytes::from_static(b"Y"));
        t2.record_write(Bytes::from_static(b"X"));
        mgr.update_ssi_txn(&t2);

        // T1 committing must abort: it forms a dangerous structure with active T2.
        let err = mgr
            .validate_ssi(&t1)
            .expect_err("T1 must be aborted on the Cahill rw-cycle");
        assert!(matches!(err, TxError::ReadWriteConflict { .. }));
    }

    #[test]
    fn ssi_cahill_range_cycle_uses_recorded_ranges() {
        // The rw-cycle can be formed via a range read (PR 3.8 + 3.9): T1 scans
        // a range containing X and writes Y; T2 reads Y and writes X.
        let mgr = TxManager::new();
        let mut t1 = mgr.begin_ssi_with_id(TxnId(1), 0);
        t1.add_read_conflict_range(Bytes::from_static(b"X"), Bytes::from_static(b"X\0"));
        t1.record_write(Bytes::from_static(b"Y"));
        mgr.update_ssi_txn(&t1);
        let mut t2 = mgr.begin_ssi_with_id(TxnId(2), 0);
        t2.record_read(Bytes::from_static(b"Y"));
        t2.record_write(Bytes::from_static(b"X"));
        mgr.update_ssi_txn(&t2);

        // T2 writes X, which falls inside T1's recorded range, and T1 writes Y
        // which T2 read → rw-cycle → T1 aborts.
        let err = mgr
            .validate_ssi(&t1)
            .expect_err("range read must form a Cahill rw-cycle");
        assert!(matches!(err, TxError::ReadWriteConflict { .. }));
    }
}
