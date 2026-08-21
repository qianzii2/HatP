//! SSI Transaction Comprehensive Integration Tests (referencing FoundationDB ConflictRange + Cahill algorithm)
//!
//! # Business Scenarios
//!
//! Serializable Snapshot Isolation (SSI) is HatP's transaction isolation layer.
//! Scenario: multiple concurrent transactions operate on shared data in OLTP scenarios, the engine must detect and prevent the following anomalies:
//! - Write-write conflicts (two transactions write the same key)
//! - Read-write antidependencies (one transaction writes a key that another transaction has read)
//! - Phantom reads (after a range scan, another transaction inserts a new key)
//! - Cahill dangerous structures (rw-cycles)
//!
//! # Test Strategy
//!
//! Reference FoundationDB ConflictRange.cpp: create two transactions, one commits changes,
//! the other detects conflicts.
//! Reference Cahill paper: construct rw-cycles, verify the committer is aborted.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    // The two `assert!(false, ...)` arm guards in the Cahill test are
    // unreachable fallbacks — they exist so that adding a new `TxError`
    // variant does not silently turn the test into a no-op.
    clippy::assertions_on_constants
)]

use bytes::Bytes;
use hatp_tx::{IsolationHint, TxManager, TxnId};

// ── Helpers ─────────────────────────────────────────────────────────────────

fn new_manager() -> std::sync::Arc<TxManager> {
    TxManager::new()
}

// ── Scenario 1: Conflict-free SSI transaction commits successfully ──────────

/// Business scenario: two transactions write different keys, both should commit successfully.
/// This is the SSI happy path — verify normal commits do not produce false positives.
#[test]
fn ssi_no_conflict_different_keys() {
    let mgr = new_manager();
    let mut t1 = mgr.begin_ssi_with_id(TxnId(1), 0);
    let mut t2 = mgr.begin_ssi_with_id(TxnId(2), 0);

    t1.record_write(Bytes::from_static(b"key_a"));
    t2.record_write(Bytes::from_static(b"key_b"));
    mgr.update_ssi_txn(&t1);
    mgr.update_ssi_txn(&t2);

    assert!(mgr.validate_ssi(&t1).is_ok(), "t1 must pass validation");
    assert!(mgr.validate_ssi(&t2).is_ok(), "t2 must pass validation");

    mgr.commit_at(t1.handle.id, 1).unwrap();
    mgr.commit_at(t2.handle.id, 2).unwrap();
    assert_eq!(mgr.inflight().len(), 0, "all transactions must be terminal");
}

// ── Scenario 2: Write-write conflict — first committer wins ─────────────────

/// Business scenario: two transactions simultaneously write the same key. The first committer succeeds, the second is rejected.
/// Reference: FoundationDB's first-committer-wins semantics.
/// Reference: RocksDB's `WriteConflictTest`.
#[test]
fn ssi_write_write_first_committer_wins() {
    let mgr = new_manager();
    let mut a = mgr.begin_ssi_with_id(TxnId(1), 0);
    let mut b = mgr.begin_ssi_with_id(TxnId(2), 0);

    a.record_write(Bytes::from_static(b"shared_key"));
    b.record_write(Bytes::from_static(b"shared_key"));
    mgr.update_ssi_txn(&a);
    mgr.update_ssi_txn(&b);

    // Both are staged, neither should conflict (engine's write_guard serializes pre-commit).
    assert!(mgr.validate_ssi(&a).is_ok());
    assert!(mgr.validate_ssi(&b).is_ok());

    // a commits.
    mgr.commit_at(a.handle.id, 1).unwrap();

    // b must now see the conflict (a's write_set is already in commit_history).
    let err = mgr.validate_ssi(&b).unwrap_err();
    assert!(
        matches!(err, hatp_tx::TxError::WriteConflict { .. }),
        "b must detect write-write conflict with committed a, got {err:?}"
    );
}

// ── Scenario 3: Peer commit before snapshot start does not conflict ─────────

/// Business scenario: a peer transaction committed **before the current transaction started**.
/// Because the current transaction's snapshot already includes the peer's write,
/// overwriting the same key is legitimate, not a conflict.
/// Reference: RocksDB's `ssi_no_write_conflict_when_peer_committed_before_start`.
#[test]
fn ssi_no_conflict_when_peer_committed_before_start_ts() {
    let mgr = new_manager();
    let mut a = mgr.begin_ssi_with_id(TxnId(1), 0);
    a.record_write(Bytes::from_static(b"col"));
    mgr.update_ssi_txn(&a);
    mgr.commit_at(a.handle.id, 1).unwrap();

    // b starts after a committed (start_ts = 5 > 1).
    let mut b = mgr.begin_ssi_with_id(TxnId(2), 5);
    b.record_write(Bytes::from_static(b"col"));
    mgr.update_ssi_txn(&b);

    assert!(
        mgr.validate_ssi(&b).is_ok(),
        "peer committed before current start_ts must not conflict"
    );
}

// ── Scenario 4: Read-write antidependency (write-skew) ──────────────────────

/// Business scenario: the classic doctors-on-call write-skew.
/// Doc1 reads doc2_oncall → writes doc1_oncall; Doc2 reads doc1_oncall → writes doc2_oncall.
/// Both think they can go off-call, but at least one should be aborted.
/// Reference: FoundationDB's Cahill paper example.
#[test]
fn ssi_read_write_antidependency_doctors_write_skew() {
    let mgr = new_manager();

    let mut doc1 = mgr.begin_ssi_with_id(TxnId(1), 0);
    doc1.record_read(Bytes::from_static(b"doc2_oncall"));
    doc1.record_write(Bytes::from_static(b"doc1_oncall"));
    mgr.update_ssi_txn(&doc1);

    let mut doc2 = mgr.begin_ssi_with_id(TxnId(2), 0);
    doc2.record_read(Bytes::from_static(b"doc1_oncall"));
    doc2.record_write(Bytes::from_static(b"doc2_oncall"));
    mgr.update_ssi_txn(&doc2);

    // Cahill rw-cycle: doc1 reads doc2_oncall writes doc1_oncall; doc2 reads doc1_oncall writes doc2_oncall.
    // Verify doc1 is aborted (it is the committer).
    let err = mgr.validate_ssi(&doc1).unwrap_err();
    assert!(
        matches!(err, hatp_tx::TxError::ReadWriteConflict { .. }),
        "Cahill rw-cycle must abort the committer, got {err:?}"
    );
}

// ── Scenario 5: Already-aborted txn cannot be committed ─────────────────────

/// Business scenario: after a transaction has been aborted, any retry commit must be rejected.
/// Reference: FoundationDB's "aborted transaction retry bypasses conflict detection" security fix.
#[test]
fn aborted_txn_cannot_be_committed() {
    let mgr = new_manager();
    let id = mgr.begin_with_id(TxnId(1), IsolationHint::Snapshot).id;
    mgr.abort(id).unwrap();

    let err = mgr.commit_at(id, 1).unwrap_err();
    assert!(
        matches!(err, hatp_tx::TxError::InvalidState { .. }),
        "aborted txn must not be committable"
    );
}

// ── Scenario 6: Already-committed txn cannot be re-committed ────────────────

/// Business scenario: idempotency guarantee — re-committing an already committed txn is rejected.
#[test]
fn committed_txn_cannot_be_recommitted() {
    let mgr = new_manager();
    let id = mgr.begin_with_id(TxnId(1), IsolationHint::Snapshot).id;
    mgr.commit_at(id, 1).unwrap();

    let err = mgr.commit_at(id, 2).unwrap_err();
    assert!(
        matches!(err, hatp_tx::TxError::InvalidState { .. }),
        "committed txn must not be re-committable"
    );
}

// ── Scenario 7: Unknown txn ID query returns error ──────────────────────────

/// Business scenario: caller passes a nonexistent transaction id. The manager must return UnknownTxn error,
/// no panic, no incorrect data.
#[test]
fn unknown_txn_id_returns_expected_error() {
    let mgr = new_manager();
    let err = mgr.get(TxnId(999)).unwrap_err();
    assert!(matches!(err, hatp_tx::TxError::UnknownTxn { .. }));
}

// ── Scenario 8: Group-commit staging conflict detection ─────────────────────

/// Business scenario: two conflicting transactions are coalesced into the same WAL batch. The first txn's
/// write-set is provisionally staged, and the second txn's validate_ssi must see it.
/// Reference: the transaction layer's group-commit first-committer-wins fix.
#[test]
fn ssi_group_commit_provisional_staging_detects_conflict() {
    let mgr = new_manager();
    let mut a = mgr.begin_ssi_with_id(TxnId(1), 0);
    a.record_write(Bytes::from_static(b"col"));
    mgr.update_ssi_txn(&a);
    let mut b = mgr.begin_ssi_with_id(TxnId(2), 0);
    b.record_write(Bytes::from_static(b"col"));
    mgr.update_ssi_txn(&b);

    // Before staging, neither conflicts.
    assert!(mgr.validate_ssi(&a).is_ok());
    assert!(mgr.validate_ssi(&b).is_ok());

    // Stage a at commit_ts=1.
    mgr.stage_commit(TxnId(1), 1).unwrap();

    // b must now see a's provisional write-set.
    let err = mgr.validate_ssi(&b).unwrap_err();
    assert!(
        matches!(err, hatp_tx::TxError::WriteConflict { .. }),
        "b must see a's provisional write-set"
    );

    // After discarding the staging, b is clean again.
    mgr.discard_staged();
    assert!(mgr.validate_ssi(&b).is_ok());
}

// ── Scenario 9: Range read conflict (phantom protection) ────────────────────

/// Business scenario: transaction A scanned the `[t\0, t\0\xff)` range, transaction B inserted
/// a new key within the range. Transaction A detects the range conflict on commit.
/// Reference: FoundationDB's ConflictRange.cpp.
#[test]
fn ssi_range_read_conflict_detects_phantom() {
    let mgr = new_manager();
    let mut reader = mgr.begin_ssi_with_id(TxnId(1), 0);
    reader.add_read_conflict_range(
        Bytes::from_static(b"t\0"),
        Bytes::from_static(b"t\0\xff"),
    );
    reader.record_write(Bytes::from_static(b"other_key"));
    mgr.update_ssi_txn(&reader);

    let mut writer = mgr.begin_ssi_with_id(TxnId(2), 0);
    writer.record_write(Bytes::from_static(b"t\0k1")); // within reader's range
    mgr.update_ssi_txn(&writer);
    mgr.commit_at(writer.handle.id, 1).unwrap();

    let err = mgr.validate_ssi(&reader).unwrap_err();
    assert!(
        matches!(err, hatp_tx::TxError::ReadWriteConflict { .. }),
        "range read must detect phantom insert, got {err:?}"
    );
}

// ── Scenario 10: Manager lifecycle — inflight only contains active transactions ─

/// Business scenario: query the active transaction list. Committed and aborted transactions must not appear in inflight.
/// Tests that inflight only contains active transactions (complements manager.rs lifecycle_reaches_commit)
#[test]
fn inflight_only_contains_active_transactions() {
    let mgr = new_manager();
    let a = mgr.begin_with_id(TxnId(1), IsolationHint::Snapshot).id;
    let b = mgr.begin_with_id(TxnId(2), IsolationHint::Snapshot).id;
    mgr.commit_at(b, 1).unwrap();

    let inflight = mgr.inflight();
    let ids: Vec<_> = inflight.iter().map(|h| h.id).collect();
    assert!(ids.contains(&a), "active txn must be in inflight");
    assert!(!ids.contains(&b), "committed txn must NOT be in inflight");
    assert_eq!(ids.len(), 1, "exactly one active txn expected");
}

// ── Scenario 11: SSI context cleared on txn abort ───────────────────────────

/// Business scenario: after transaction abort, its SSI context is cleared, no ghost conflict leak.
/// Reference: the integration test `transaction_ssi_drop_releases_context`.
#[test]
fn ssi_context_cleared_on_abort() {
    let mgr = new_manager();
    let mut txn = mgr.begin_ssi_with_id(TxnId(1), 0);
    txn.record_write(Bytes::from_static(b"k"));
    mgr.update_ssi_txn(&txn);
    mgr.abort(txn.handle.id).unwrap();

    assert!(
        mgr.get_ssi_txn(TxnId(1)).is_none(),
        "SSI context must be cleared on abort"
    );
}

// ── Scenario 12: Commit history GC does not discard entries visible to long txn ─

/// Business scenario: after a long transaction begins, short transactions commit many writes. The long txn's read set
/// must keep the short transactions' writes in commit history, not be discarded by GC.
/// Reference: R-03 fix: fixed-capacity history would discard entries needed by long transactions.
#[test]
fn commit_history_gc_preserves_entries_for_long_transaction() {
    let mgr = new_manager();
    let mut long = mgr.begin_ssi_with_id(TxnId(10), 0);
    long.record_read(Bytes::from_static(b"col"));
    mgr.update_ssi_txn(&long);

    // Short transaction writes and commits.
    let mut short = mgr.begin_ssi_with_id(TxnId(1), 1);
    short.record_write(Bytes::from_static(b"col"));
    mgr.update_ssi_txn(&short);
    mgr.commit_at(short.handle.id, 1).unwrap();

    // The long transaction must see the read-write conflict.
    let err = mgr.validate_ssi(&long).unwrap_err();
    assert!(
        matches!(err, hatp_tx::TxError::ReadWriteConflict { .. }),
        "long txn must still see the committed write, got {err:?}"
    );
}

// ── SC-06: Commit History GC stress ─────────────────────────────────────────
/// After 1000 short transaction commits, the long transaction still detects the conflict.
/// Verifies that the watermark-bounded commit_history GC does not discard entries needed by the long txn.
#[test]
fn ssi_commit_history_gc_stress_preserves_conflict_for_long_txn() {
    let mgr = new_manager();

    // Long transaction: start_ts=0, reads shared_key
    let mut long = mgr.begin_ssi_with_id(hatp_tx::TxnId(9999), 0);
    long.record_read(Bytes::from_static(b"shared_key"));
    mgr.update_ssi_txn(&long);

    // Commit 1000 short transactions, the 500th writes shared_key
    for i in 1..=1000_u64 {
        let mut short = mgr.begin_ssi_with_id(hatp_tx::TxnId(i), i);
        if i == 500 {
            short.record_write(Bytes::from_static(b"shared_key"));
        } else {
            short.record_write(Bytes::copy_from_slice(format!("key_{i}").as_bytes()));
        }
        mgr.update_ssi_txn(&short);
        mgr.commit_at(short.handle.id, i).unwrap();
    }

    // Positive assertion: long transaction must detect the 500th short txn's conflict
    let err = mgr.validate_ssi(&long).unwrap_err();
    assert!(
        matches!(err, hatp_tx::TxError::ReadWriteConflict { .. }),
        "after 1000 commits, long txn must still detect conflict on shared_key"
    );

    // Negative assertion: after long txn ends, commit_history can be safely GC'd
    mgr.abort(long.handle.id).unwrap();
    // Subsequent transactions have no conflict
    let mut t = mgr.begin_ssi_with_id(hatp_tx::TxnId(10001), 1001);
    t.record_write(Bytes::from_static(b"shared_key"));
    mgr.update_ssi_txn(&t);
    assert!(
        mgr.validate_ssi(&t).is_ok(),
        "fresh txn after long txn abort must not see stale conflicts"
    );
}

// ── SC-07 (G17): Concurrent transaction lifecycle ───────────────────────────
/// 10 transactions mixed commit/abort/active, verify inflight and SSI context correctness
#[test]
fn ssi_concurrent_lifecycle_mixed_states() {
    let mgr = new_manager();

    // Create 10 transactions: 5 commit, 3 abort, 2 active
    let mut committed = Vec::new();
    let mut aborted = Vec::new();
    let mut active = Vec::new();

    for i in 0..5_u64 {
        let mut t = mgr.begin_ssi_with_id(hatp_tx::TxnId(i + 1), i);
        t.record_write(Bytes::copy_from_slice(format!("committed_{i}").as_bytes()));
        mgr.update_ssi_txn(&t);
        mgr.commit_at(t.handle.id, i + 1).expect("commit");
        committed.push(t.handle.id);
    }
    for i in 0..3_u64 {
        let mut t = mgr.begin_ssi_with_id(hatp_tx::TxnId(i + 6), 5);
        t.record_write(Bytes::copy_from_slice(format!("aborted_{i}").as_bytes()));
        mgr.update_ssi_txn(&t);
        mgr.abort(t.handle.id).expect("abort");
        aborted.push(t.handle.id);
    }
    for i in 0..2_u64 {
        let t = mgr.begin_ssi_with_id(hatp_tx::TxnId(i + 9), 5);
        active.push(t.handle.id);
    }

    // Positive assertion: inflight contains only 2 Active transactions
    let inflight = mgr.inflight();
    let inflight_ids: Vec<_> = inflight.iter().map(|h| h.id).collect();
    assert_eq!(inflight.len(), 2, "only 2 active txns expected");
    for id in &active {
        assert!(inflight_ids.contains(id), "active txn must be in inflight");
    }
    for id in &committed {
        assert!(!inflight_ids.contains(id), "committed txn not in inflight");
    }
    for id in &aborted {
        assert!(!inflight_ids.contains(id), "aborted txn not in inflight");
    }

    // Positive assertion: aborted transactions' SSI contexts are cleared
    for id in &aborted {
        assert!(mgr.get_ssi_txn(*id).is_none(), "aborted SSI context cleared");
    }

    // Negative assertion: subsequent transactions are unaffected by ghost conflicts
    let mut t = mgr.begin_ssi_with_id(hatp_tx::TxnId(100), 100);
    t.record_write(Bytes::from_static(b"fresh_key"));
    mgr.update_ssi_txn(&t);
    assert!(
        mgr.validate_ssi(&t).is_ok(),
        "fresh txn must not see ghost conflicts"
    );
}

// ── SC-08 (G18): Cahill 3-party rw-cycle ────────────────────────────────────
/// Three transactions form a cycle: T1 reads A writes B, T2 reads B writes C,
/// T3 reads C writes A. validate_cahill must detect the dangerous structure
/// and abort at least one transaction.
#[test]
fn ssi_cahill_three_party_cycle_detected() {
    let mgr = new_manager();

    // Seed all three keys
    let mut t1 = mgr.begin_ssi_with_id(hatp_tx::TxnId(1), 0);
    let mut t2 = mgr.begin_ssi_with_id(hatp_tx::TxnId(2), 0);
    let mut t3 = mgr.begin_ssi_with_id(hatp_tx::TxnId(3), 0);

    // T1: reads A, writes B
    t1.record_read(Bytes::from_static(b"A"));
    t1.record_write(Bytes::from_static(b"B"));
    mgr.update_ssi_txn(&t1);

    // T2: reads B, writes C
    t2.record_read(Bytes::from_static(b"B"));
    t2.record_write(Bytes::from_static(b"C"));
    mgr.update_ssi_txn(&t2);

    // T3: reads C, writes A
    t3.record_read(Bytes::from_static(b"C"));
    t3.record_write(Bytes::from_static(b"A"));
    mgr.update_ssi_txn(&t3);

    // T1 commits — must detect the rw-cycle with T2 and T3
    let result = mgr.record_write_and_validate(
        hatp_tx::TxnId(1),
        &[Bytes::from_static(b"B")],
    );
    match result {
        Err(hatp_tx::TxError::ReadWriteConflict { .. }) => {
            // Expected: Cahill cycle detected for T1
        }
        Err(hatp_tx::TxError::WriteConflict { .. }) => {
            // Also acceptable: WW conflict may be detected first
        }
        Err(hatp_tx::TxError::InvalidState { .. })
        | Err(hatp_tx::TxError::UnknownTxn { .. }) => {
            // `record_write_and_validate` only returns conflict errors during
            // normal validation. Reaching a terminal-state error here means the
            // transaction state machine is broken, not a legitimate test path.
            assert!(
                false,
                "unexpected terminal-state error from record_write_and_validate"
            );
        }
        Ok(()) => {
            // T1 passed — T2 must fail
            let r2 = mgr.record_write_and_validate(
                hatp_tx::TxnId(2),
                &[Bytes::from_static(b"C")],
            );
            match r2 {
                Err(hatp_tx::TxError::ReadWriteConflict { .. })
                | Err(hatp_tx::TxError::WriteConflict { .. }) => {}
                Err(hatp_tx::TxError::InvalidState { .. })
                | Err(hatp_tx::TxError::UnknownTxn { .. }) => {
                    // `record_write_and_validate` only returns conflict errors
                    // during normal validation. Reaching a terminal-state
                    // error here means the transaction state machine is
                    // broken, not a legitimate test path.
                    assert!(
                        false,
                        "unexpected terminal-state error from record_write_and_validate"
                    );
                }
                Ok(()) => {
                    // Both T1 and T2 passed — T3 must fail
                    let r3 = mgr.record_write_and_validate(
                        hatp_tx::TxnId(3),
                        &[Bytes::from_static(b"A")],
                    );
                    assert!(
                        r3.is_err(),
                        "3-party Cahill cycle: at least one transaction must abort"
                    );
                }
            }
        }
    }
}