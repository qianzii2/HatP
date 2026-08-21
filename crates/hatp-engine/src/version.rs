//! MVCC version chains, snapshots, and garbage collection.

use bytes::Bytes;
use smallvec::SmallVec;

// Re-export from hatp_types so internal and sibling modules can access
// via `use crate::version::{TxnTs, OPEN_ENDED_TS}`.
pub use hatp_types::{OPEN_ENDED_TS, TxnTs};

/// One version of a row value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionedValue {
    /// Encoded row payload (`None` is a delete tombstone).
    pub value: Option<Bytes>,
    /// Timestamp at which this version became visible.
    pub begin_ts: TxnTs,
    /// Timestamp at which this version was superseded.
    pub end_ts: TxnTs,
    /// Originating transaction identifier.
    pub tx_id: u64,
}

impl VersionedValue {
    /// Creates a live value whose visibility has no upper bound yet.
    #[must_use]
    pub fn live(value: Bytes, ts: TxnTs, tx_id: u64) -> Self {
        Self {
            value: Some(value),
            begin_ts: ts,
            end_ts: OPEN_ENDED_TS,
            tx_id,
        }
    }

    /// Creates a delete tombstone.
    #[must_use]
    pub fn tombstone(ts: TxnTs, tx_id: u64) -> Self {
        Self {
            value: None,
            begin_ts: ts,
            end_ts: OPEN_ENDED_TS,
            tx_id,
        }
    }

    /// Returns whether this committed version is visible to `snapshot_ts`.
    #[must_use]
    pub fn visible_at(&self, snapshot_ts: TxnTs) -> bool {
        self.begin_ts <= snapshot_ts && snapshot_ts < self.end_ts
    }
}

/// Consistent read point used by a transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Snapshot {
    /// Highest committed timestamp visible to the reader.
    pub ts: TxnTs,
}

impl Snapshot {
    /// Creates a snapshot at `ts`.
    #[must_use]
    pub const fn new(ts: TxnTs) -> Self {
        Self { ts }
    }

    /// Tests visibility without exposing interval details to callers.
    #[must_use]
    pub fn sees(self, version: &VersionedValue) -> bool {
        version.visible_at(self.ts)
    }
}

/// Ordered versions for one logical key, newest first.
///
/// Uses `SmallVec<[VersionedValue; 4]>` — in steady-state OLTP, 99% of keys
/// have 1–2 versions (one live + possibly one tombstone).  Heap allocation
/// only kicks in past 4 versions (a long contention history), which is rare.
#[derive(Debug, Clone, Default)]
pub struct VersionChain {
    versions: SmallVec<[VersionedValue; 4]>,
}

impl VersionChain {
    /// Creates an empty version chain.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Installs a committed version while preserving visibility intervals.
    ///
    /// The chain is stored newest-first (highest `begin_ts` at index 0).
    /// Logically we want `versions[i].begin_ts > versions[i+1].begin_ts`
    /// for every `i`. The two-step algorithm — find the splice index
    /// with `partition_point`, then `Vec::insert` — replaces the
    /// previous `push` + `Vec::sort` pattern. Measured in
    /// `crates/hatp-bench/benches/version_chain.rs`:
    ///
    /// ```text
    /// chain length 16, naive_push_sort: 8.97 µs
    /// chain length 16, sorted_upper_bound: 5.63 µs     (1.6x faster)
    /// chain length 64, naive_push_sort: 146.7 µs
    /// chain length 64, sorted_upper_bound: 46.4 µs     (3.2x faster)
    /// ```
    ///
    /// `self.versions[idx]` is in-bounds: `idx` comes from `partition_point`
    /// and the guarded `if idx < self.versions.len()` check precedes it.
    #[allow(clippy::indexing_slicing)]
    pub fn insert(&mut self, mut version: VersionedValue) {
        // Fast path: chain is empty (every new write starts here).
        if self.versions.is_empty() {
            self.versions.push(version);
            return;
        }
        // Fast path: insert at head (newest) — the production case for
        // monotonically increasing commit_ts.  Skip the partition_point
        // binary search entirely.
        if self.versions[0].begin_ts < version.begin_ts {
            // Adjust the old head's end_ts so visibility intervals stay
            // contiguous.
            self.versions[0].end_ts = version.begin_ts;
            self.versions.insert(0, version);
            return;
        }
        // Find the first index whose `begin_ts <= version.begin_ts`.
        let idx = self
            .versions
            .partition_point(|candidate| candidate.begin_ts > version.begin_ts);
        // Fast path: collide with an existing entry (same `begin_ts`).
        // The newest-first sorted invariant guarantees the equal-`begin_ts`
        // entry, if any, sits exactly at `idx` — so this is an O(1) probe
        // instead of the previous O(n) `iter_mut().find` linear scan.
        if idx < self.versions.len() && self.versions[idx].begin_ts == version.begin_ts {
            let existing = &mut self.versions[idx];
            version.end_ts = existing.end_ts;
            *existing = version;
            return;
        }
        // If there is a newer neighbour (idx > 0), this version's visibility
        // interval must end at that neighbour's begin_ts. Without this, a
        // mid-chain insert keeps its constructor-assigned OPEN_ENDED_TS and
        // stays visible forever, overlapping the newer version. The monotonic
        // tx_id path always inserts at idx == 0 (the head), so this branch is
        // only reachable when callers supply out-of-order timestamps — it is
        // still required for MVCC correctness under `write_with_tx`.
        if idx > 0 {
            version.end_ts = self.versions[idx - 1].begin_ts;
        }
        // Range-adjust the older neighbour's `end_ts` so the visibility
        // intervals stay contiguous. `idx` may be `versions.len()` (oldest).
        if idx < self.versions.len() {
            let older = &mut self.versions[idx];
            if older.end_ts > version.begin_ts {
                older.end_ts = version.begin_ts;
            }
        }
        self.versions.insert(idx, version);
    }

    /// Returns the visible version at a snapshot.
    #[must_use]
    pub fn get(&self, snapshot: Snapshot) -> Option<VersionedValue> {
        // Fast path: the chain is small (1-2 entries) in the steady state
        // (most keys are written once and never overwritten). Check head
        // directly — no iterator, no closure allocation.
        if let Some(head) = self.versions.first() {
            if snapshot.sees(head) {
                return Some(head.clone());
            }
        } else {
            return None;
        }
        // Slow path: manual loop avoids the iterator + closure allocation of
        // `.iter().find(|v| snapshot.sees(v))`.  The chain is still ≤4 entries
        // 99% of the time, so a linear scan is optimal.
        for version in &self.versions {
            if snapshot.sees(version) {
                return Some(version.clone());
            }
        }
        None
    }

    /// Removes superseded versions that no active snapshot can observe.
    /// The newest version is always retained even when it is a tombstone.
    ///
    /// `self.versions[0]` is in-bounds: the empty-chain fast path returns
    /// before this point.
    #[allow(clippy::indexing_slicing)]
    pub fn garbage_collect(&mut self, oldest_snapshot: TxnTs) -> usize {
        let before = self.versions.len();
        // Fast path: empty chain.
        if self.versions.is_empty() {
            return 0;
        }
        // The newest version (`versions[0]`) is always retained even when
        // superseded; that's the GC invariant. We only retain a tail
        // version if its `end_ts` is still observable.
        let newest_ts = self.versions[0].begin_ts;
        self.versions
            .retain(|version| version.begin_ts == newest_ts || version.end_ts > oldest_snapshot);
        before.saturating_sub(self.versions.len())
    }

    /// Returns all versions in newest-first order.
    #[must_use]
    pub fn versions(&self) -> &[VersionedValue] {
        &self.versions
    }

    /// Returns whether the chain is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.versions.is_empty()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn insert_out_of_order_sets_end_ts() {
        let mut chain = VersionChain::new();
        chain.insert(VersionedValue::live(Bytes::from_static(b"new"), 100, 1));
        chain.insert(VersionedValue::live(Bytes::from_static(b"old"), 50, 2));

        let versions = chain.versions();
        assert_eq!(versions.len(), 2);
        // Newest first.
        assert_eq!(versions[0].begin_ts, 100);
        assert_eq!(versions[0].end_ts, OPEN_ENDED_TS);
        assert_eq!(versions[1].begin_ts, 50);
        assert_eq!(
            versions[1].end_ts, 100,
            "mid-chain insert must be capped by the newer neighbour's begin_ts"
        );

        // A snapshot between the two timestamps sees the older value.
        let older = chain.get(Snapshot::new(75)).expect("visible at 75");
        assert_eq!(older.value.as_ref().expect("live").as_ref(), b"old");
        // A snapshot past the newer timestamp sees the newer value.
        let newer = chain.get(Snapshot::new(150)).expect("visible at 150");
        assert_eq!(newer.value.as_ref().expect("live").as_ref(), b"new");
    }

    #[test]
    fn insert_newer_keeps_open_ended() {
        let mut chain = VersionChain::new();
        chain.insert(VersionedValue::live(Bytes::from_static(b"old"), 50, 1));
        chain.insert(VersionedValue::live(Bytes::from_static(b"new"), 100, 2));
        let versions = chain.versions();
        assert_eq!(versions.len(), 2);
        assert_eq!(versions[0].begin_ts, 100);
        assert_eq!(versions[0].end_ts, OPEN_ENDED_TS);
        assert_eq!(versions[1].end_ts, 100);
    }
}
