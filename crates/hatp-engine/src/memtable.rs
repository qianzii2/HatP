//! Concurrent in-memory MVCC write buffer.
//!
//! Two interchangeable backends sit behind a single [`MemTable`] handle:
//! - [`BTreeMapTable`] — `RwLock<BTreeMap>` (default): simple, cache-friendly.
//! - [`SkipMapTable`] — `crossbeam_skiplist::SkipMap`: lock-free concurrent
//!   reads/writes (PR 2.3, R-22 "thorough").
//!
//! [`Engine`](crate::Engine) picks one via `EngineConfig::memtable_impl` and
//! keeps a single [`MemTable`] that forwards to the chosen [`MemTableBackend`].

use crate::version::{Snapshot, TxnTs, VersionChain, VersionedValue};
use crate::wal::{OpType, WalRecord};
use crate::{EngineError, Result};
use bytes::Bytes;
use crossbeam_skiplist::SkipMap;
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Thread-safe approximate byte counter.
#[derive(Debug, Default)]
pub struct MemTableSize(AtomicUsize);

impl MemTableSize {
    /// Adds bytes and returns the new total, saturating at `usize::MAX`.
    pub fn add(&self, bytes: usize) -> usize {
        let mut current = self.0.load(Ordering::Acquire);
        loop {
            let next = current.saturating_add(bytes);
            match self
                .0
                .compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return next,
                Err(observed) => current = observed,
            }
        }
    }

    /// Returns the current total.
    #[must_use]
    pub fn get(&self) -> usize {
        self.0.load(Ordering::Acquire)
    }

    /// Resets the total to zero.
    pub fn reset(&self) {
        self.0.store(0, Ordering::Release);
    }
}

/// A key and committed version used by SST writers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionedRow {
    /// Logical key.
    pub key: Bytes,
    /// Committed MVCC version.
    pub version: VersionedValue,
}

/// Which concrete backend a [`MemTable`] should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemTableImpl {
    /// `RwLock<BTreeMap>` — deterministic ordering, low memory.
    BTreeMap,
    /// `crossbeam_skiplist::SkipMap` — lock-free concurrent access (default).
    #[default]
    SkipMap,
}

/// The contract every memtable backend implements. [`MemTable`] forwards to a
/// `Arc<dyn MemTableBackend>`, so swapping backends never changes the engine's
/// call sites.
pub trait MemTableBackend: Send + Sync + 'static + std::fmt::Debug {
    /// Applies a batch of WAL records atomically at `commit_ts`.
    ///
    /// # `ts == commit_ts` (intentional)
    ///
    /// Every version in one committed transaction shares the single
    /// `commit_ts` (the engine-allocated commit sequence number), distinct from
    /// `tx_id` (reservation order). See [`crate::Engine::write_with_tx`].
    fn apply_records_batch(&self, records: &[WalRecord], commit_ts: u64) -> Result<()>;

    /// Applies a batch of [`Mutation`]s directly, bypassing the intermediate
    /// [`WalRecord`] allocation.  Default implementation delegates to
    /// [`apply_records_batch`]; backends can override for zero-alloc paths.
    fn apply_mutations_batch(
        &self,
        tx_id: u64,
        mutations: &[crate::Mutation],
        commit_ts: u64,
    ) -> Result<()> {
        let records: Vec<WalRecord> = mutations
            .iter()
            .map(|m| match m {
                crate::Mutation::Put { key, value } => {
                    WalRecord::put(tx_id, key.clone(), value.clone())
                }
                crate::Mutation::Delete { key } => WalRecord::delete(tx_id, key.clone()),
            })
            .collect();
        self.apply_records_batch(&records, commit_ts)
    }

    /// Returns the version visible at `snapshot`, including tombstones.
    fn get(&self, key: &[u8], snapshot: Snapshot) -> Option<VersionedValue>;

    /// Returns visible versions (with full MVCC metadata) whose key starts with
    /// `prefix`, in key order.
    fn scan_prefix_versions(
        &self,
        prefix: &[u8],
        snapshot: Snapshot,
    ) -> Vec<(Bytes, VersionedValue)>;

    /// Returns visible versions (with full MVCC metadata) whose key falls in
    /// `[lower, upper)`, in key order. Superset of
    /// [`MemTableBackend::scan_prefix_versions`] (a prefix scan is the special
    /// case `upper = prefix_upper_exclusive(prefix)`).
    fn scan_range_versions(
        &self,
        lower: &[u8],
        upper: &[u8],
        snapshot: Snapshot,
    ) -> Vec<(Bytes, VersionedValue)>;

    /// Returns a stable copy of every committed version in key order.
    fn all_versions(&self) -> Vec<VersionedRow>;

    /// Pushes every committed version into `out` in key order.  The default
    /// implementation delegates to [`all_versions`]; backends that can iterate
    /// without cloning (e.g. [`SkipMapTable`] under the write guard) override
    /// this to avoid the intermediate allocation.
    fn drain_into(&self, out: &mut Vec<VersionedRow>) {
        out.extend(self.all_versions());
    }

    /// Garbage-collects versions invisible to every active snapshot.
    fn garbage_collect(&self, oldest_snapshot: TxnTs) -> usize;

    /// Removes every row after a durable flush.
    fn clear(&self);

    /// Atomically removes all entries whose key starts with `prefix`, returning
    /// the number of keys removed.
    fn erase_prefix(&self, prefix: &[u8]) -> usize;

    /// Returns the approximate memory footprint in bytes.
    fn approximate_bytes(&self) -> usize;

    /// Returns the number of logical keys.
    fn len(&self) -> usize;

    /// Returns whether no logical keys are present.
    fn is_empty(&self) -> bool;

    /// Returns visible (key, value) pairs whose key starts with `prefix`,
    /// derived from [`MemTableBackend::scan_prefix_versions`].
    fn scan_prefix(&self, prefix: &[u8], snapshot: Snapshot) -> Vec<(Bytes, Bytes)> {
        self.scan_prefix_versions(prefix, snapshot)
            .into_iter()
            .filter_map(|(key, version)| version.value.map(|value| (key, value)))
            .collect()
    }
}

/// `RwLock<BTreeMap<Bytes, VersionChain>>` backend (the pre-PR-2.3 default).
#[derive(Debug, Default)]
pub struct BTreeMapTable {
    entries: RwLock<BTreeMap<Bytes, VersionChain>>,
    bytes: MemTableSize,
}

impl MemTableBackend for BTreeMapTable {
    fn apply_records_batch(&self, records: &[WalRecord], commit_ts: u64) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        // One write-lock acquisition for the whole batch (hot-path optimisation).
        let mut entries = self.entries.write();
        for record in records {
            match record.op {
                OpType::Put => {
                    let value = record
                        .value
                        .as_ref()
                        .ok_or(EngineError::Corrupt("put WAL record has no value"))?;
                    let key = &record.key;
                    let chain = entries.entry(key.clone()).or_default();
                    chain.insert(VersionedValue::live(value.clone(), commit_ts, record.tx_id));
                    self.bytes
                        .add(key.len().saturating_add(value.len()).saturating_add(24));
                }
                OpType::Delete => {
                    let key = &record.key;
                    let chain = entries.entry(key.clone()).or_default();
                    chain.insert(VersionedValue::tombstone(commit_ts, record.tx_id));
                    self.bytes.add(key.len().saturating_add(24));
                }
                OpType::Commit | OpType::Abort => {}
            }
        }
        Ok(())
    }

    fn apply_mutations_batch(
        &self,
        tx_id: u64,
        mutations: &[crate::Mutation],
        commit_ts: u64,
    ) -> Result<()> {
        if mutations.is_empty() {
            return Ok(());
        }
        let mut entries = self.entries.write();
        for mutation in mutations {
            match mutation {
                crate::Mutation::Put { key, value } => {
                    let chain = entries.entry(key.clone()).or_default();
                    chain.insert(VersionedValue::live(value.clone(), commit_ts, tx_id));
                    self.bytes
                        .add(key.len().saturating_add(value.len()).saturating_add(24));
                }
                crate::Mutation::Delete { key } => {
                    let chain = entries.entry(key.clone()).or_default();
                    chain.insert(VersionedValue::tombstone(commit_ts, tx_id));
                    self.bytes.add(key.len().saturating_add(24));
                }
            }
        }
        Ok(())
    }

    fn get(&self, key: &[u8], snapshot: Snapshot) -> Option<VersionedValue> {
        self.entries
            .read()
            .get(key)
            .and_then(|chain| chain.get(snapshot))
    }

    fn scan_prefix_versions(
        &self,
        prefix: &[u8],
        snapshot: Snapshot,
    ) -> Vec<(Bytes, VersionedValue)> {
        use std::ops::Bound;
        let lower = Bytes::copy_from_slice(prefix);
        let upper = Bytes::from(prefix_upper_exclusive(prefix));
        let read_guard = self.entries.read();
        let mut out = Vec::new();
        for (key, chain) in read_guard.range((Bound::Included(lower), Bound::Excluded(upper))) {
            if !key.starts_with(prefix) {
                continue;
            }
            if let Some(version) = chain.get(snapshot) {
                out.push((key.clone(), version));
            }
        }
        out
    }

    fn scan_range_versions(
        &self,
        lower: &[u8],
        upper: &[u8],
        snapshot: Snapshot,
    ) -> Vec<(Bytes, VersionedValue)> {
        use std::ops::Bound;
        let read_guard = self.entries.read();
        let mut out = Vec::new();
        let lower_bound = Bound::Included(Bytes::copy_from_slice(lower));
        let upper_bound = Bound::Excluded(Bytes::copy_from_slice(upper));
        for (key, chain) in read_guard.range((lower_bound, upper_bound)) {
            if let Some(version) = chain.get(snapshot) {
                out.push((key.clone(), version));
            }
        }
        out
    }

    fn all_versions(&self) -> Vec<VersionedRow> {
        self.entries
            .read()
            .iter()
            .flat_map(|(key, chain)| {
                chain
                    .versions()
                    .iter()
                    .cloned()
                    .map(|version| VersionedRow {
                        key: key.clone(),
                        version,
                    })
            })
            .collect()
    }

    fn garbage_collect(&self, oldest_snapshot: TxnTs) -> usize {
        self.entries
            .write()
            .values_mut()
            .map(|chain| chain.garbage_collect(oldest_snapshot))
            .sum()
    }

    fn clear(&self) {
        self.entries.write().clear();
        self.bytes.reset();
    }

    fn erase_prefix(&self, prefix: &[u8]) -> usize {
        use std::ops::Bound;
        let lower = Bytes::copy_from_slice(prefix);
        let upper = Bytes::from(prefix_upper_exclusive(prefix));

        let mut entries = self.entries.write();
        let keys: Vec<Bytes> = entries
            .range((Bound::Included(lower), Bound::Excluded(upper)))
            .map(|(k, _)| k.clone())
            .collect();

        let count = keys.len();
        for key in keys {
            entries.remove(&key);
        }
        count
    }

    fn approximate_bytes(&self) -> usize {
        self.bytes.get()
    }

    fn len(&self) -> usize {
        self.entries.read().len()
    }

    fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }
}

/// `Arc<ArcSwap<VersionChain>>` — lock-free reads with atomic-swap writes.
/// The inner `ArcSwap` lets readers observe the chain without any lock;
/// writers clone the chain, mutate the clone, and atomically swap the pointer.
/// This is the default backend (PR 2.3, R-22), now fully lock-free on reads.
#[derive(Debug, Default)]
pub struct SkipMapTable {
    entries: SkipMap<Bytes, Arc<arc_swap::ArcSwap<VersionChain>>>,
    bytes: MemTableSize,
}

impl MemTableBackend for SkipMapTable {
    fn apply_records_batch(&self, records: &[WalRecord], commit_ts: u64) -> Result<()> {
        for record in records {
            match record.op {
                OpType::Put => {
                    let value = record
                        .value
                        .as_ref()
                        .ok_or(EngineError::Corrupt("put WAL record has no value"))?;
                    let key = &record.key;
                    let swap = self.entries.get_or_insert(
                        key.clone(),
                        Arc::new(arc_swap::ArcSwap::from_pointee(VersionChain::new())),
                    );
                    swap.value().rcu(|current| {
                        let mut new = (**current).clone();
                        new.insert(VersionedValue::live(
                            value.clone(),
                            commit_ts,
                            record.tx_id,
                        ));
                        Arc::new(new)
                    });
                    self.bytes
                        .add(key.len().saturating_add(value.len()).saturating_add(24));
                }
                OpType::Delete => {
                    let key = &record.key;
                    let swap = self.entries.get_or_insert(
                        key.clone(),
                        Arc::new(arc_swap::ArcSwap::from_pointee(VersionChain::new())),
                    );
                    swap.value().rcu(|current| {
                        let mut new = (**current).clone();
                        new.insert(VersionedValue::tombstone(commit_ts, record.tx_id));
                        Arc::new(new)
                    });
                    self.bytes.add(key.len().saturating_add(24));
                }
                OpType::Commit | OpType::Abort => {}
            }
        }
        Ok(())
    }

    fn apply_mutations_batch(
        &self,
        tx_id: u64,
        mutations: &[crate::Mutation],
        commit_ts: u64,
    ) -> Result<()> {
        for mutation in mutations {
            match mutation {
                crate::Mutation::Put { key, value } => {
                    let swap = self.entries.get_or_insert(
                        key.clone(),
                        Arc::new(arc_swap::ArcSwap::from_pointee(VersionChain::new())),
                    );
                    swap.value().rcu(|current| {
                        let mut new = (**current).clone();
                        new.insert(VersionedValue::live(value.clone(), commit_ts, tx_id));
                        Arc::new(new)
                    });
                    self.bytes
                        .add(key.len().saturating_add(value.len()).saturating_add(24));
                }
                crate::Mutation::Delete { key } => {
                    let swap = self.entries.get_or_insert(
                        key.clone(),
                        Arc::new(arc_swap::ArcSwap::from_pointee(VersionChain::new())),
                    );
                    swap.value().rcu(|current| {
                        let mut new = (**current).clone();
                        new.insert(VersionedValue::tombstone(commit_ts, tx_id));
                        Arc::new(new)
                    });
                    self.bytes.add(key.len().saturating_add(24));
                }
            }
        }
        Ok(())
    }

    fn get(&self, key: &[u8], snapshot: Snapshot) -> Option<VersionedValue> {
        self.entries
            .get(key)
            .and_then(|entry| entry.value().load_full().get(snapshot))
    }

    fn scan_prefix_versions(
        &self,
        prefix: &[u8],
        snapshot: Snapshot,
    ) -> Vec<(Bytes, VersionedValue)> {
        use std::ops::Bound;
        let lower = Bytes::copy_from_slice(prefix);
        let upper = Bytes::from(prefix_upper_exclusive(prefix));
        let mut out = Vec::new();
        for entry in self
            .entries
            .range((Bound::Included(lower), Bound::Excluded(upper)))
        {
            let key = entry.key();
            if !key.starts_with(prefix) {
                continue;
            }
            if let Some(version) = entry.value().load_full().get(snapshot) {
                out.push((key.clone(), version));
            }
        }
        out
    }

    fn scan_range_versions(
        &self,
        lower: &[u8],
        upper: &[u8],
        snapshot: Snapshot,
    ) -> Vec<(Bytes, VersionedValue)> {
        use std::ops::Bound;
        let lower_bound = Bound::Included(Bytes::copy_from_slice(lower));
        let upper_bound = Bound::Excluded(Bytes::copy_from_slice(upper));
        let mut out = Vec::new();
        for entry in self.entries.range((lower_bound, upper_bound)) {
            let key = entry.key();
            if let Some(version) = entry.value().load_full().get(snapshot) {
                out.push((key.clone(), version));
            }
        }
        out
    }

    fn all_versions(&self) -> Vec<VersionedRow> {
        let mut out = Vec::new();
        for entry in self.entries.iter() {
            let key = entry.key().clone();
            let versions = entry.value().load_full();
            for version in versions.versions().iter().cloned() {
                out.push(VersionedRow {
                    key: key.clone(),
                    version,
                });
            }
        }
        out
    }

    fn drain_into(&self, out: &mut Vec<VersionedRow>) {
        // Push directly — no intermediate Vec.  The caller (flush) holds
        // `write_guard`, so the SkipMap is quiescent and the `load_full()`
        // snapshot is stable for the duration of the drain.
        for entry in self.entries.iter() {
            let key = entry.key().clone();
            let versions = entry.value().load_full();
            // 99% of keys have 1 version — avoid the inner clone in the
            // common case by peeking at len.
            let vs = versions.versions();
            if vs.len() == 1 {
                out.push(VersionedRow {
                    key,
                    version: vs[0].clone(),
                });
            } else {
                for version in vs.iter().cloned() {
                    out.push(VersionedRow {
                        key: key.clone(),
                        version,
                    });
                }
            }
        }
    }

    fn garbage_collect(&self, oldest_snapshot: TxnTs) -> usize {
        let mut dropped = 0_usize;
        for entry in self.entries.iter() {
            let swap = entry.value();
            let current = swap.load_full();
            let mut new = (*current).clone();
            let before = new.versions().len();
            new.garbage_collect(oldest_snapshot);
            let after = new.versions().len();
            dropped = dropped.saturating_add(before.saturating_sub(after));
            if before != after {
                swap.store(Arc::new(new));
            }
        }
        dropped
    }

    fn clear(&self) {
        self.entries.clear();
        self.bytes.reset();
    }

    fn erase_prefix(&self, prefix: &[u8]) -> usize {
        use std::ops::Bound;
        let lower = Bytes::copy_from_slice(prefix);
        let upper = Bytes::from(prefix_upper_exclusive(prefix));
        let keys: Vec<Bytes> = self
            .entries
            .range((Bound::Included(lower), Bound::Excluded(upper)))
            .map(|entry| entry.key().clone())
            .collect();
        let count = keys.len();
        for key in keys {
            self.entries.remove(&key);
        }
        count
    }

    fn approximate_bytes(&self) -> usize {
        self.bytes.get()
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Ordered memtable handle. Forwards to a [`MemTableBackend`]; the engine holds
/// an `Arc<MemTable>` and never knows which backend is active.
#[derive(Debug)]
pub struct MemTable {
    backend: Arc<dyn MemTableBackend>,
}

impl Default for MemTable {
    fn default() -> Self {
        Self::new()
    }
}

impl MemTable {
    /// Creates an empty memtable with the default backend ([`BTreeMapTable`]).
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: Arc::new(BTreeMapTable::default()),
        }
    }

    /// Creates a memtable using the requested backend implementation.
    #[must_use]
    pub fn with_impl(impl_kind: MemTableImpl) -> Self {
        match impl_kind {
            MemTableImpl::BTreeMap => Self {
                backend: Arc::new(BTreeMapTable::default()),
            },
            MemTableImpl::SkipMap => Self {
                backend: Arc::new(SkipMapTable::default()),
            },
        }
    }

    /// Creates a memtable over an arbitrary backend.
    #[must_use]
    pub fn with_backend(backend: Arc<dyn MemTableBackend>) -> Self {
        Self { backend }
    }

    /// Applies a batch of WAL records in a single backend call.
    pub fn apply_records_batch(&self, records: &[WalRecord], commit_ts: u64) -> Result<()> {
        self.backend.apply_records_batch(records, commit_ts)
    }

    /// Applies a batch of [`Mutation`]s directly, bypassing the intermediate
    /// [`WalRecord`] allocation.  The `tx_id` is the transaction identifier.
    pub fn apply_mutations_batch(
        &self,
        tx_id: u64,
        mutations: &[crate::Mutation],
        commit_ts: u64,
    ) -> Result<()> {
        self.backend.apply_mutations_batch(tx_id, mutations, commit_ts)
    }

    /// Returns the version visible at `snapshot`, including tombstones.
    #[must_use]
    pub fn get(&self, key: &[u8], snapshot: Snapshot) -> Option<VersionedValue> {
        self.backend.get(key, snapshot)
    }

    /// Returns visible (key, value) pairs whose key starts with `prefix`.
    #[must_use]
    pub fn scan_prefix(&self, prefix: &[u8], snapshot: Snapshot) -> Vec<(Bytes, Bytes)> {
        self.backend.scan_prefix(prefix, snapshot)
    }

    /// Returns visible versions (with full MVCC metadata) whose key starts with
    /// `prefix`, in key order.
    #[must_use]
    pub fn scan_prefix_versions(
        &self,
        prefix: &[u8],
        snapshot: Snapshot,
    ) -> Vec<(Bytes, VersionedValue)> {
        self.backend.scan_prefix_versions(prefix, snapshot)
    }

    /// Returns visible versions (with full MVCC metadata) whose key falls in
    /// `[lower, upper)`, in key order.
    #[must_use]
    pub fn scan_range_versions(
        &self,
        lower: &[u8],
        upper: &[u8],
        snapshot: Snapshot,
    ) -> Vec<(Bytes, VersionedValue)> {
        self.backend.scan_range_versions(lower, upper, snapshot)
    }

    /// Returns a stable copy of every committed version in key order.
    #[must_use]
    pub fn all_versions(&self) -> Vec<VersionedRow> {
        self.backend.all_versions()
    }

    /// Pushes every committed version into `out` without an intermediate
    /// allocation.  Prefer this over [`all_versions`] when the caller already
    /// owns a reusable buffer.
    pub fn drain_into(&self, out: &mut Vec<VersionedRow>) {
        self.backend.drain_into(out);
    }

    /// Garbage-collects versions invisible to every active snapshot.
    pub fn garbage_collect(&self, oldest_snapshot: TxnTs) -> usize {
        self.backend.garbage_collect(oldest_snapshot)
    }

    /// Removes every row after a durable flush.
    pub fn clear(&self) {
        self.backend.clear();
    }

    /// Atomically removes all entries whose key starts with `prefix`,
    /// returning the number of keys removed.
    pub fn erase_prefix(&self, prefix: &[u8]) -> usize {
        self.backend.erase_prefix(prefix)
    }

    /// Returns the approximate memory footprint in bytes.
    #[must_use]
    pub fn approximate_bytes(&self) -> usize {
        self.backend.approximate_bytes()
    }

    /// Returns the number of logical keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.backend.len()
    }

    /// Returns whether no logical keys are present.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.backend.is_empty()
    }
}

/// Returns the smallest dictionary-order upper bound (exclusive) such that
/// `prefix <= key < upper` holds **iff** `key.starts_with(prefix)`.
///
/// This is the single source of truth for prefix-range scans (shared by both
/// memtable backends and by the engine's SST table scan). It supersedes the old
/// `prefix + 0xFF` inclusive bound, which missed keys whenever `prefix` ended in
/// `0xFF`.
///
/// Table prefixes built by `crate::table_prefix` always end in `\0`, so a
/// finite bound always exists for real callers. The only prefix without a
/// finite upper bound is one of all-`0xFF` bytes, which `table_prefix` cannot
/// produce; for that impossible input this returns `prefix` unchanged, yielding
/// an empty `[prefix, prefix)` range instead of panicking.
#[must_use]
pub(crate) fn prefix_upper_exclusive(prefix: &[u8]) -> Vec<u8> {
    let mut upper = prefix.to_vec();
    while let Some(last) = upper.last_mut() {
        if *last == u8::MAX {
            upper.pop();
        } else {
            *last += 1;
            return upper;
        }
    }
    prefix.to_vec()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn put_record(tx_id: u64, key: &[u8], value: &[u8]) -> WalRecord {
        WalRecord::put(
            tx_id,
            Bytes::copy_from_slice(key),
            Bytes::copy_from_slice(value),
        )
    }

    /// Same set of mutations produce identical scan output on both backends (PR 2.3 equivalence test).
    fn assert_backends_equivalent(records: &[WalRecord], commit_ts: u64) {
        let btree = MemTable::with_impl(MemTableImpl::BTreeMap);
        let skip = MemTable::with_impl(MemTableImpl::SkipMap);
        btree.apply_records_batch(records, commit_ts).unwrap();
        skip.apply_records_batch(records, commit_ts).unwrap();

        let snap = Snapshot::new(commit_ts);
        let b_scan = btree.scan_prefix(b"", snap);
        let s_scan = skip.scan_prefix(b"", snap);
        assert_eq!(b_scan, s_scan, "skipmap must match btreemap outputs");

        let b_all = btree.all_versions();
        let s_all = skip.all_versions();
        assert_eq!(b_all.len(), s_all.len());
    }

    #[test]
    fn skipmap_matches_btreemap_outputs() {
        let records = vec![
            put_record(1, b"a", b"1"),
            put_record(1, b"b", b"2"),
            put_record(2, b"a", b"3"), // overwrite a
        ];
        assert_backends_equivalent(&records, 2);
    }

    #[test]
    fn skipmap_get_and_prefix_scan_round_trip() {
        let table = MemTable::with_impl(MemTableImpl::SkipMap);
        let records = vec![
            put_record(1, b"t\x00k1", b"v1"),
            put_record(1, b"t\x00k2", b"v2"),
        ];
        table.apply_records_batch(&records, 1).unwrap();
        let snap = Snapshot::new(1);

        assert_eq!(
            table.get(b"t\x00k1", snap).unwrap().value.unwrap().as_ref(),
            b"v1"
        );
        let scanned = table.scan_prefix(b"t\x00", snap);
        assert_eq!(scanned.len(), 2);
    }
}
