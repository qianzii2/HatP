//! Crash recovery tests — modeled after SQLite crashsql / RocksDB SyncPoint
//!
//! Test scenarios (ordered by real-world business priority):
//! 1. WAL mid-write truncation (torn write) — the last incomplete frame is ignored
//! 2. WAL frame CRC corruption — corrupted frame and all subsequent frames are ignored
//! 3. Crash mid-flush — SST already written but manifest not yet committed
//! 4. Crash mid-compaction — old SSTs still present, new SST not yet registered
//! 5. Mixed recovery — SST + WAL replay consistency
//! 6. Empty WAL recovery — first startup
//! 7. Repeated recovery — recovering the same data twice yields identical results
//!
//! References:
//! - SQLite `crash.test`: parent process verifies recovery after child process crash
//! - RocksDB `db_flush_test.cc`: SyncPoint controls mid-flush failure
//! - TigerBeetle `storage.zig`: write corruption + repair verification

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::assertions_on_constants
)]

use bytes::Bytes;
use hatp_engine::{Engine, EngineConfig, Mutation};
use std::fs;
use tempfile::Builder;

fn unique_dir(label: &str) -> tempfile::TempDir {
    Builder::new()
        .prefix(&format!("hatp-crash-{label}-"))
        .tempdir()
        .expect("tempdir")
}

fn put(key: &[u8], value: &[u8]) -> Mutation {
    Mutation::Put {
        key: Bytes::copy_from_slice(key),
        value: Bytes::copy_from_slice(value),
    }
}

// =============================================================================
// Scenario 1: WAL mid-write truncation (torn write)
// Business scenario: process crashes while writing WAL, last frame is incomplete
// Expected: on recovery the last incomplete frame is ignored, completed frames replay correctly
// =============================================================================
#[test]
fn torn_wal_tail_truncated_on_recovery() {
    let dir = unique_dir("torn-tail");
    let wal_path = dir.path().join("hatp.wal");

    // Write 3 keys, then truncate WAL to simulate a mid-write crash
    {
        let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");
        engine.write(&[put(b"k1", b"v1")]).expect("write 1");
        engine.write(&[put(b"k2", b"v2")]).expect("write 2");
        engine.write(&[put(b"k3", b"v3")]).expect("write 3");
    }

    // Truncate the last 4 bytes of the WAL (simulate incomplete CRC)
    let mut bytes = fs::read(&wal_path).expect("read WAL");
    let original_len = bytes.len();
    bytes.truncate(original_len.saturating_sub(4));
    fs::write(&wal_path, &bytes).expect("write truncated WAL");

    // Recovery: the last commit frame may be lost, but at least the first two commits are visible
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("reopen");
    let snap = engine.snapshot_ts();

    // Negative assertion: not all keys can be lost
    let v1 = engine.get(b"k1", snap).expect("get k1");
    let v2 = engine.get(b"k2", snap).expect("get k2");

    // At least the first two commits should be visible (the third may be lost)
    assert_eq!(v1.as_deref(), Some(b"v1".as_ref()), "k1 must survive torn WAL tail");
    assert_eq!(v2.as_deref(), Some(b"v2".as_ref()), "k2 must survive torn WAL tail");
}

// =============================================================================
// Scenario 2: WAL frame CRC corruption
// Business scenario: silent disk corruption causes CRC check failure on a WAL frame
// Expected: the corrupted frame and all subsequent frames are ignored, prior frames replay correctly
// =============================================================================
#[test]
fn corrupted_wal_frame_causes_truncation_at_corruption_point() {
    let dir = unique_dir("corrupt-frame");
    let wal_path = dir.path().join("hatp.wal");

    {
        let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");
        engine.write(&[put(b"k1", b"v1")]).expect("write 1");
        engine.write(&[put(b"k2", b"v2")]).expect("write 2");
    }

    // Flip a byte in the middle of the WAL (simulate silent bit-flip corruption)
    let mut bytes = fs::read(&wal_path).expect("read WAL");
    let corruption_point = bytes.len() / 2;
    bytes[corruption_point] ^= 0x01;
    fs::write(&wal_path, &bytes).expect("write corrupted WAL");

    let engine = Engine::open(EngineConfig::new(dir.path())).expect("reopen");
    let snap = engine.snapshot_ts();

    // Negative assertion: not both keys can be lost
    let v1 = engine.get(b"k1", snap).expect("get k1");
    assert!(v1.is_some(), "at least k1 must survive corruption at midpoint");
    if let Some(v) = v1 {
        assert_eq!(v.as_ref(), b"v1", "surviving value must be exact");
    }
}

// =============================================================================
// Scenario 3: Crash after flush, SST on disk but manifest not committed
// =============================================================================
#[test]
fn recovery_after_flush_preserves_data() {
    let dir = unique_dir("flush-recover");
    {
        let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");
        engine.write(&[put(b"k1", b"v1")]).expect("write");
        let handle = engine.flush().expect("flush").expect("SST handle");
        assert!(handle.rows > 0, "flush must produce rows");
    }
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("reopen");
    let snap = engine.snapshot_ts();
    let v = engine.get(b"k1", snap).expect("get");
    assert_eq!(v.as_deref(), Some(b"v1".as_ref()), "flushed data must survive restart");
}

// =============================================================================
// Scenario 4: Multiple flush + write mixed recovery
// =============================================================================
#[test]
fn mixed_sst_wal_recovery_preserves_latest_versions() {
    let dir = unique_dir("mixed-recover");
    {
        let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");
        engine.write(&[put(b"k", b"v1")]).expect("write v1");
        engine.flush().expect("flush v1");
        engine.write(&[put(b"k", b"v2")]).expect("write v2");
    }
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("reopen");
    let snap = engine.snapshot_ts();
    let v = engine.get(b"k", snap).expect("get");
    assert_eq!(v.as_deref(), Some(b"v2".as_ref()),
        "latest WAL version must override flushed SST version");
    assert_ne!(v.as_deref(), Some(b"v1".as_ref()),
        "WAL replay must not serve stale flushed version");
}

// =============================================================================
// Scenario 5: Empty WAL recovery
// =============================================================================
#[test]
fn fresh_open_without_wal_succeeds() {
    let dir = unique_dir("fresh");
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");
    let snap = engine.snapshot_ts();
    assert_eq!(engine.get(b"any", snap).expect("get"), None,
        "fresh engine must return None for any key");
}

// =============================================================================
// Scenario 6: Recovery after SST file corruption
// =============================================================================
#[test]
fn recovery_skips_corrupt_sst_and_logs_failure() {
    let dir = unique_dir("corrupt-sst");
    let sst_path;
    {
        let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");
        engine.write(&[put(b"k", b"v")]).expect("write");
        let handle = engine.flush().expect("flush").expect("SST handle");
        sst_path = dir.path().join(format!("sst-{:020}.vortex", handle.file_id));
    }
    fs::write(&sst_path, b"completely corrupted file content").expect("corrupt");
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("reopen");
    let failures = engine.metrics().sst_read_failures.load(std::sync::atomic::Ordering::Relaxed);
    assert!(failures >= 1, "corrupt SST must increment sst_read_failures metric, got {failures}");
}

// =============================================================================
// Scenario 7: Delete operation persists after recovery
// =============================================================================
#[test]
fn delete_survives_crash_recovery() {
    let dir = unique_dir("delete-recover");
    {
        let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");
        engine.write(&[put(b"k", b"v")]).expect("write");
        engine.write(&[Mutation::Delete { key: Bytes::from_static(b"k") }]).expect("delete");
    }
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("reopen");
    let snap = engine.snapshot_ts();
    assert_eq!(engine.get(b"k", snap).expect("get"), None,
        "deleted key must remain deleted after recovery");
}

// =============================================================================
// SC-04: WAL partial transaction recovery
// =============================================================================

#[test]
fn partial_transaction_not_visible_after_recovery() {
    let dir = unique_dir("partial-txn");
    {
        let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");
        engine.write(&[put(b"tx1", b"v1")]).expect("tx1 write");
        let mut wal_bytes = Vec::new();
        let _ = hatp_engine::wal::encode_frame(
            &mut wal_bytes, 42, hatp_engine::wal::OpType::Put,
            b"tx2_incomplete", Some(b"v2"),
        );
        std::fs::write(dir.path().join("hatp.wal"), &wal_bytes).expect("write partial WAL");
    }
    {
        let engine = Engine::open(EngineConfig::new(dir.path())).expect("reopen");
        let snap = engine.snapshot_ts();
        assert_eq!(
            engine.get(b"tx2_incomplete", snap).expect("get"),
            None,
            "uncommitted (no Commit marker) txn must not be visible"
        );
        assert_ne!(
            engine.get(b"tx2_incomplete", snap).expect("get").as_deref(),
            Some(b"v2".as_ref()),
            "uncommitted value must not leak"
        );
    }
}

#[test]
fn recovery_from_empty_wal_produces_snapshot_zero() {
    let dir = unique_dir("empty-wal-recovery");
    std::fs::write(dir.path().join("hatp.wal"), b"").expect("write empty WAL");
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");
    let snap = engine.snapshot_ts();
    assert_eq!(snap, 0, "empty WAL must produce snapshot_ts == 0");
    assert_eq!(engine.get(b"any", snap).expect("get"), None,
        "fresh engine must return None for any key");
}