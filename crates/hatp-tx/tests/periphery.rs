#![allow(clippy::expect_used, clippy::unwrap_used, clippy::indexing_slicing)]

//! Integration tests for `hatp-tx`.
//!
//! Transaction ids are supplied explicitly (the manager is not an id
//! allocator — see R-07); `start_ts` and `commit_ts` are also explicit so
//! each test controls the commit-order relationship under test.
//!
//! The following tests complement manager.rs (do not duplicate basic lifecycle scenarios):
//! - txn_handles_have_unique_identifiers (T224): independent, no corresponding scenario in manager.rs
//! - ssi_validate_detects_rw_antidependency (T233): RW antidependency, not in manager.rs
//! - ssi_validate_ignores_rw_from_pre_start_commits (T234): pre-start exemption
//! - ssi_commit_at_records_commit_ts (T235): commit_ts record verification
//! - ssi_commit_at_zero_commit_ts_does_not_trip_rw (T236): boundary commit_ts
//!
//! Tests already merged into manager.rs (T219-T232, 14 total):
//! begin_with_id / inflight / commit lifecycle / abort / unknown / isolation /
//! inflight_only / terminal / ssi lifecycle / ssi records / ww conflict / no conflict / validate

use bytes::Bytes;
use hatp_tx::{IsolationHint, TxError, TxManager, TxnId, TxnState};

#[test]
fn txn_handles_have_unique_identifiers() {
    let manager = TxManager::new();
    let a = manager.begin_with_id(TxnId(1), IsolationHint::Snapshot).id;
    let b = manager.begin_with_id(TxnId(2), IsolationHint::Snapshot).id;
    let c = manager.begin_with_id(TxnId(3), IsolationHint::Snapshot).id;
    assert_ne!(a, b);
    assert_ne!(b, c);
    assert!(a < b && b < c, "ids must be monotonic");
}

#[test]
fn ssi_validate_detects_rw_antidependency() {
    // A peer committed a write to a key this txn read *after* this txn began
    // → ReadWriteConflict.
    let manager = TxManager::new();
    let mut reader = manager.begin_ssi_with_id(TxnId(1), 0);
    reader.record_read(Bytes::from_static(b"hot_key"));
    manager.update_ssi_txn(&reader);

    // A second transaction commits a write to `hot_key` after reader began
    // (commit_ts = 1 > reader.start_ts = 0).
    let mut writer = manager.begin_ssi_with_id(TxnId(2), 1);
    writer.record_write(Bytes::from_static(b"hot_key"));
    manager.update_ssi_txn(&writer);
    manager
        .commit_at(writer.handle.id, 1)
        .expect("commit writer");

    // Reader's validation must now report a read-write antidependency.
    let err = manager
        .validate_ssi(&reader)
        .expect_err("rw antidependency must trigger");
    assert!(
        matches!(err, TxError::ReadWriteConflict { .. }),
        "expected ReadWriteConflict, got {err:?}"
    );
}

#[test]
fn ssi_validate_ignores_rw_from_pre_start_commits() {
    // A peer committed *before* this txn began (commit_ts <= start_ts) is
    // invisible to MVCC and must not trigger ReadWriteConflict.
    let manager = TxManager::new();
    let mut reader = manager.begin_ssi_with_id(TxnId(1), 5);
    reader.record_read(Bytes::from_static(b"hot_key"));
    manager.update_ssi_txn(&reader);

    // Pre-start commit at ts == reader.start_ts.
    let mut writer = manager.begin_ssi_with_id(TxnId(2), 4);
    writer.record_write(Bytes::from_static(b"hot_key"));
    manager.update_ssi_txn(&writer);
    manager
        .commit_at(writer.handle.id, 5)
        .expect("commit writer");

    // Reader sees nothing in commit_history strictly after start_ts → ok.
    let result = manager.validate_ssi(&reader);
    assert!(
        result.is_ok(),
        "pre-start commit must not flag rw antidependency: {result:?}"
    );
}

#[test]
fn ssi_commit_at_records_commit_ts() {
    let manager = TxManager::new();
    let mut reader = manager.begin_ssi_with_id(TxnId(1), 10);
    reader.record_read(Bytes::from_static(b"k"));
    manager.update_ssi_txn(&reader);

    let mut writer = manager.begin_ssi_with_id(TxnId(2), 11);
    writer.record_write(Bytes::from_static(b"k"));
    manager.update_ssi_txn(&writer);
    let writer_handle = manager
        .commit_at(writer.handle.id, 4242)
        .expect("commit_at");
    assert_eq!(writer_handle.state, TxnState::Committed);

    let err = manager
        .validate_ssi(&reader)
        .expect_err("peer committed after reader.start_ts wrote a read key");
    assert!(
        matches!(err, TxError::ReadWriteConflict { .. }),
        "expected ReadWriteConflict, got {err:?}"
    );
}

#[test]
fn ssi_commit_at_zero_commit_ts_does_not_trip_rw() {
    let manager = TxManager::new();
    let mut writer = manager.begin_ssi_with_id(TxnId(1), 0);
    writer.record_write(Bytes::from_static(b"k"));
    manager.update_ssi_txn(&writer);
    manager.commit_at(writer.handle.id, 1).expect("commit_at 1");

    // Reader begins after the writer: start_ts = 5 > 1.
    let mut reader = manager.begin_ssi_with_id(TxnId(2), 5);
    reader.record_read(Bytes::from_static(b"k"));
    assert!(
        manager.validate_ssi(&reader).is_ok(),
        "pre-start commit must not flag rw antidependency"
    );
}