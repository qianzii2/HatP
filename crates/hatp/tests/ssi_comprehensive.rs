//! SSI Transaction Comprehensive Tests — referencing FoundationDB ConflictRange + Cahill algorithm
//!
//! Test scenarios (ordered by SSI paper priority):
//! 1. First-committer-wins (WW) — two transactions write the same key
//! 2. Read-write antidependency (RW) — read then concurrently written
//! 3. Cahill dangerous structure — T1 reads X writes Y, T2 reads Y writes X
//! 4. Range read phantom — range scan followed by concurrent insert
//! 5. No conflict on different keys — different keys do not conflict
//! 6. Pre-start commit no conflict — commits before the current transaction starts do not conflict
//! 7. Group-commit staging — conflict detection within the same batch
//! 8. Abort releases context — cleanup SSI context after abort
//! 9. Read-only transaction commit — read-only transaction commit
//! 10. Drop releases context — auto-cleanup when transaction is dropped
//!
//! References:
//! - FoundationDB `ConflictRange.cpp`: dual-transaction conflict detection
//! - PostgreSQL SSI paper "Serializable Snapshot Isolation in PostgreSQL"
//! - Cahill et al. "Serializable isolation for snapshot databases"

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::indexing_slicing)]

use hatp::Database;
use hatp::DatabaseError;
use hatp_tx::TxManager;
use std::sync::Arc;
use tempfile::Builder;

fn unique_dir(label: &str) -> tempfile::TempDir {
    Builder::new()
        .prefix(&format!("hatp-ssi-{label}-"))
        .tempdir()
        .expect("tempdir")
}

fn open_ssi_db(dir: &tempfile::TempDir) -> (Database, Arc<TxManager>) {
    let manager = TxManager::new();
    let db = Database::open_with_tx_manager(dir.path(), manager.clone()).expect("open");
    (db, manager)
}

// =============================================================================
// Scenario 1: First-committer-wins — two transactions write the same key
// Business scenario: two concurrent transactions modify the same row, the first commits successfully, the second detects conflict
// Expected: the second transaction's commit returns SsiConflict
// Reference: FoundationDB ConflictRange dual-transaction conflict detection
// =============================================================================
#[test]
fn ssi_first_committer_wins_same_key() {
    let dir = unique_dir("fcw");
    let (db, _manager) = open_ssi_db(&dir);

    let mut t1 = db.begin_ssi();
    t1.put("shared_key", "from_t1");

    let mut t2 = db.begin_ssi();
    t2.put("shared_key", "from_t2");

    // t1 commits first: must succeed
    let t1_commit = t1.commit().expect("t1 must commit successfully");
    assert!(t1_commit > 0, "commit_ts must be positive");

    // Positive assertion: t2 must detect write-write conflict on commit
    // t1 has committed, its write_set is in commit_history, t2's validate_ssi must see it
    let (_, err) = t2
        .try_commit()
        .expect_err("t2 must detect write-write conflict with committed t1");
    match err {
        DatabaseError::SsiConflict { txn, key } => {
            assert!(txn > 0, "conflict must identify a valid transaction");
            assert_eq!(key, b"shared_key", "conflict key must be the contested key");
        }
        other => panic!("expected SsiConflict, got {other:?}"),
    }

    // Negative assertion: final value must be t1's value (t2 was rejected)
    let final_value = db.get(b"shared_key").expect("get").map(|v| v.to_vec());
    assert_eq!(
        final_value,
        Some(b"from_t1".to_vec()),
        "final value must be t1's value since t2 was rejected"
    );
    // Negative assertion: must not be t2's value
    assert_ne!(final_value, Some(b"from_t2".to_vec()), "t2's value must not persist");
}

// =============================================================================
// Scenario 2: Read-write antidependency
// Business scenario: T1 reads a key, T2 (SSI transaction) commits a write to that key after T1
// Expected: T1 detects ReadWriteConflict on commit
// Reference: FoundationDB SSI antidependency detection
// =============================================================================
#[test]
fn ssi_read_write_antidependency() {
    let dir = unique_dir("rw-ad");
    let (db, _manager) = open_ssi_db(&dir);

    // Seed data
    db.put("hot_key", "seed_value").expect("seed");

    let mut reader = db.begin_ssi();
    // reader reads hot_key (recorded in read_set)
    let seen = reader.get(b"hot_key").expect("get").expect("present");
    assert_eq!(seen.to_vec(), b"seed_value".to_vec(), "reader must see seeded value");

    // Another SSI transaction writes hot_key and commits
    // Note: must use SSI transaction (not autocommit), otherwise write_set does not enter commit_history
    let mut writer = db.begin_ssi();
    writer.put("hot_key", "concurrent_write");
    writer.commit().expect("writer must commit");

    // reader now writes its own key (triggers commit) — must detect conflict
    reader.put("other_key", "reader_value");
    let (_, err) = reader
        .try_commit()
        .expect_err("reader must detect read-write antidependency");
    match err {
        DatabaseError::SsiReadWriteConflict { key, .. } => {
            assert_eq!(key, b"hot_key", "conflict key must be the read key");
        }
        other => panic!("expected SsiReadWriteConflict, got {other:?}"),
    }

    // Negative assertion: final value must be writer's value
    assert_eq!(
        db.get(b"hot_key").expect("get").map(|v| v.to_vec()),
        Some(b"concurrent_write".to_vec()),
        "writer's value must persist"
    );
}

// =============================================================================
// Scenario 3: Cahill dangerous structure
// Business scenario: T1 reads X writes Y, T2 reads Y writes X — forms rw-cycle
// Expected: at least one transaction is aborted
// Reference: Cahill et al. "Serializable isolation for snapshot databases"
// =============================================================================
#[test]
fn ssi_cahill_dangerous_structure_detected() {
    let dir = unique_dir("cahill");
    let (db, _manager) = open_ssi_db(&dir);

    // Seed: X=1, Y=1
    db.put("X", "1").expect("seed X");
    db.put("Y", "1").expect("seed Y");

    let mut t1 = db.begin_ssi();
    t1.get(b"X").expect("t1 read X").expect("present");
    t1.put("Y", "t1_wrote_Y");

    let mut t2 = db.begin_ssi();
    t2.get(b"Y").expect("t2 read Y").expect("present");
    t2.put("X", "t2_wrote_X");

    // Cahill rw-cycle: t1 reads X writes Y, t2 reads Y writes X
    // Both transactions cannot commit simultaneously
    let r1 = t1.try_commit();
    let r2 = t2.try_commit();
    let committed = u8::from(r1.is_ok()) + u8::from(r2.is_ok());

    assert!(
        committed <= 1,
        "Cahill rw-cycle: at most one transaction may commit, got {committed}"
    );

    // Positive assertion: final state is consistent
    let x = db.get(b"X").expect("get X").map(|v| v.to_vec());
    let y = db.get(b"Y").expect("get Y").map(|v| v.to_vec());
    assert!(
        (x == Some(b"1".to_vec()) && y == Some(b"t1_wrote_Y".to_vec()))
            || (x == Some(b"t2_wrote_X".to_vec()) && y == Some(b"1".to_vec())),
        "final state must be consistent: X={x:?}, Y={y:?}"
    );
}

// =============================================================================
// Scenario 4: Range read phantom
// Business scenario: T1 scans range [t\0, t\0\xff), T2 inserts a new key within the range
// Expected: T1 detects ReadWriteConflict (phantom) on commit
// Reference: FoundationDB ConflictRange range conflict detection
// =============================================================================
#[test]
fn ssi_range_read_detects_phantom() {
    let dir = unique_dir("phantom");
    let (db, _manager) = open_ssi_db(&dir);

    db.put("t\0k1", "v1").expect("seed k1");

    let mut reader = db.begin_ssi();
    let seen = reader.scan_range(b"t\0", b"t\0\xff").expect("range scan");
    assert_eq!(seen.len(), 1, "reader must see the seeded key");
    assert_eq!(seen[0].0.as_ref(), b"t\0k1", "reader must see exact key");

    // Concurrent write of new key within the scan range
    let mut writer = db.begin_ssi();
    writer.put("t\0k2", "v2");
    writer.commit().expect("writer commits");

    // reader writes its own key to trigger commit
    reader.put("t\0k3", "v3");
    let (_, err) = reader
        .try_commit()
        .expect_err("reader must detect phantom insert via range read conflict");
    match err {
        DatabaseError::SsiReadWriteConflict { key, .. } => {
            assert_eq!(key, b"t\0k2", "conflict key must be the phantom insert");
        }
        other => panic!("expected SsiReadWriteConflict, got {other:?}"),
    }

    // Negative assertion: phantom insert value must persist
    assert_eq!(
        db.get(b"t\0k2").expect("get").map(|v| v.to_vec()),
        Some(b"v2".to_vec()),
        "phantom insert must persist"
    );
}

// =============================================================================
// Scenario 5: Different keys do not conflict
// Business scenario: two concurrent transactions modify different keys
// Expected: both transactions commit successfully
// Reference: FoundationDB "no conflict different keys"
// =============================================================================
#[test]
fn ssi_no_conflict_on_different_keys() {
    let dir = unique_dir("no-conflict");
    let (db, _manager) = open_ssi_db(&dir);

    let mut t1 = db.begin_ssi();
    t1.put("a", "1");

    let mut t2 = db.begin_ssi();
    t2.put("b", "2");

    let id1 = t1.commit().expect("t1 must commit");
    let id2 = t2.commit().expect("t2 must commit");

    assert_ne!(id1, id2, "distinct transaction ids");

    // Positive assertion: both keys have correct values
    assert_eq!(db.get(b"a").expect("a").map(|v| v.to_vec()), Some(b"1".to_vec()));
    assert_eq!(db.get(b"b").expect("b").map(|v| v.to_vec()), Some(b"2".to_vec()));

    // Negative assertion: must not have wrong values
    assert_ne!(db.get(b"a").expect("a").map(|v| v.to_vec()), Some(b"2".to_vec()));
    assert_ne!(db.get(b"b").expect("b").map(|v| v.to_vec()), Some(b"1".to_vec()));
}

// =============================================================================
// Scenario 6: Pre-start commit does not conflict
// Business scenario: peer transactions committed before the current transaction starts do not conflict
// Expected: the current transaction commits successfully
// =============================================================================
#[test]
fn ssi_pre_start_commit_does_not_conflict() {
    let dir = unique_dir("pre-start");
    let (db, _manager) = open_ssi_db(&dir);

    // Commit a transaction first
    db.put("key", "v1").expect("pre-start put");

    // A transaction started later writes the same key
    let mut t = db.begin_ssi();
    t.put("key", "v2");

    // Should commit successfully (pre-start commit does not conflict)
    t.commit().expect("pre-start commit must not conflict");

    assert_eq!(db.get(b"key").expect("get").map(|v| v.to_vec()), Some(b"v2".to_vec()));
}

// =============================================================================
// Scenario 7: Abort releases SSI context
// Business scenario: after transaction abort, SSI context is cleaned up
// Expected: subsequent transactions are unaffected
// =============================================================================
#[test]
fn ssi_abort_releases_context() {
    let dir = unique_dir("abort-release");
    let (db, manager) = open_ssi_db(&dir);

    let mut t1 = db.begin_ssi();
    t1.put("k", "v1");
    let t1_id = t1.reserved_tx_id().expect("reserved id");
    drop(t1); // Abort — drop triggers abort

    // Verify context is cleaned up
    assert!(manager.get_ssi_txn(t1_id).is_none(), "SSI context must be released on abort");
    assert!(manager.inflight().is_empty(), "no active transactions after abort");

    // Subsequent transactions are unaffected
    let mut t2 = db.begin_ssi();
    t2.put("k", "v2");
    t2.commit().expect("fresh SSI commit after abort");

    assert_eq!(db.get(b"k").expect("get").map(|v| v.to_vec()), Some(b"v2".to_vec()));
}

// =============================================================================
// Scenario 8: Read-only transaction commit
// Business scenario: read-only transaction attempts to commit
// Expected: returns ReadOnlyCommit error
// =============================================================================
#[test]
fn ssi_read_only_transaction_rejected_on_commit() {
    let dir = unique_dir("readonly");
    let (db, _manager) = open_ssi_db(&dir);

    db.put("k", "v").expect("seed");

    let t = db.begin_ssi();
    let v = t.get(b"k").expect("get").expect("present");
    assert_eq!(v.to_vec(), b"v".to_vec(), "must read seeded value");

    // Read-only transaction commit
    let result = t.try_commit();
    assert!(result.is_err(), "read-only transaction must be rejected");
    if let Err((_, DatabaseError::ReadOnlyCommit)) = result {
        // Expected
    } else if let Err((_, err)) = result {
        panic!("expected ReadOnlyCommit, got {err:?}");
    }
}

// =============================================================================
// Scenario 9: Drop auto-releases SSI context
// Business scenario: SSI context is auto-cleaned when transaction is dropped
// Expected: no ghost conflict
// =============================================================================
#[test]
fn ssi_drop_releases_context_no_ghost_conflict() {
    let dir = unique_dir("ghost");
    let (db, manager) = open_ssi_db(&dir);

    let mut t1 = db.begin_ssi();
    t1.put("k", "v1");
    let t1_id = t1.reserved_tx_id().expect("reserved id");
    drop(t1); // No commit, no explicit abort

    // Verify context is released
    assert!(manager.get_ssi_txn(t1_id).is_none(), "SSI context must be released on drop");
    assert!(manager.inflight().is_empty(), "no active transactions after drop");

    // Subsequent transactions are unaffected by ghost conflict
    let mut t2 = db.begin_ssi();
    t2.put("k", "v2");
    t2.commit().expect("fresh SSI commit after drop");

    assert_eq!(db.get(b"k").expect("get").map(|v| v.to_vec()), Some(b"v2".to_vec()));
}

// =============================================================================
// Scenario 10: Write skew detection
// Business scenario: two doctors each read the other's value then toggle their own
// Expected: at least one is aborted
// Reference: PostgreSQL SSI paper write-skew example
// =============================================================================
#[test]
fn ssi_write_skew_doctors_example() {
    let dir = unique_dir("write-skew");
    let (db, _manager) = open_ssi_db(&dir);

    db.put("doc1_oncall", "1").expect("seed doc1");
    db.put("doc2_oncall", "1").expect("seed doc2");

    let mut doc1 = db.begin_ssi();
    doc1.get(b"doc2_oncall").expect("d1 sees doc2").expect("present");

    let mut doc2 = db.begin_ssi();
    doc2.get(b"doc1_oncall").expect("d2 sees doc1").expect("present");

    doc1.put("doc1_oncall", "0");
    doc2.put("doc2_oncall", "0");

    let r1 = doc1.try_commit();
    let r2 = doc2.try_commit();
    let committed = u8::from(r1.is_ok()) + u8::from(r2.is_ok());

    assert!(
        committed <= 1,
        "write-skew: at most one doctor may go off-call, got {committed}"
    );

    // Positive assertion: at least one doctor still on-call
    let d1 = db.get(b"doc1_oncall").expect("d1").map(|v| v.to_vec());
    let d2 = db.get(b"doc2_oncall").expect("d2").map(|v| v.to_vec());
    assert!(
        d1 == Some(b"1".to_vec()) || d2 == Some(b"1".to_vec()),
        "at least one doctor must stay on-call: d1={d1:?}, d2={d2:?}"
    );

    // Negative assertion: both doctors cannot be off-call
    assert!(
        !(d1 == Some(b"0".to_vec()) && d2 == Some(b"0".to_vec())),
        "both doctors off-call is an inconsistent state"
    );
}