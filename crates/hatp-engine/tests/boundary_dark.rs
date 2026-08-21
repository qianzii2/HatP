//! Boundary value & dark scenario tests — referencing SQLite boundary1.test + RocksDB error paths
//!
//! Test scenarios:
//! 1. Empty key / empty value write and read
//! 2. \0 delimiter boundary key
//! 3. Large key / large value
//! 4. Duplicate key overwrite
//! 5. Concurrent writes to the same key
//! 6. Empty memtable flush
//! 7. Duplicate flush
//! 8. Empty table scan
//! 9. Range scan boundary
//! 10. Error propagation: empty batch, empty mutations
//!
//! References:
//! - SQLite `boundary1.test`: 64 rowid boundary values × 5 operators
//! - RocksDB empty value/large value/boundary write_buffer_size tests

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::assertions_on_constants
)]

mod common;

use bytes::Bytes;
use common::boundaries;
use hatp_engine::{Engine, EngineConfig, EngineError, Mutation};
use tempfile::Builder;

fn unique_dir(label: &str) -> tempfile::TempDir {
    Builder::new()
        .prefix(&format!("hatp-boundary-{label}-"))
        .tempdir()
        .expect("tempdir")
}

fn put(key: &[u8], value: &[u8]) -> Mutation {
    Mutation::Put {
        key: Bytes::copy_from_slice(key),
        value: Bytes::copy_from_slice(value),
    }
}

fn delete(key: &[u8]) -> Mutation {
    Mutation::Delete {
        key: Bytes::copy_from_slice(key),
    }
}

// =============================================================================
// Scenario 1: Empty key and empty value write
// Scenario: User writes empty string as key or value
// Expected: Engine handles it normally, no crash
// =============================================================================
#[test]
fn empty_key_and_value_round_trip() {
    let dir = unique_dir("empty-kv");
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");

    // Empty key + empty value
    let tx = engine.write(&[put(b"", b"")]).expect("write empty");
    assert_eq!(
        engine.get(b"", tx).expect("get").as_deref(),
        Some(b"".as_ref()),
        "empty key with empty value"
    );

    // Empty key + non-empty value
    let tx = engine.write(&[put(b"", b"hello")]).expect("write empty key");
    assert_eq!(
        engine.get(b"", tx).expect("get").as_deref(),
        Some(b"hello".as_ref()),
        "empty key with non-empty value"
    );

    // Non-empty key + empty value
    let tx = engine.write(&[put(b"k", b"")]).expect("write empty value");
    assert_eq!(
        engine.get(b"k", tx).expect("get").as_deref(),
        Some(b"".as_ref()),
        "non-empty key with empty value"
    );
}

// =============================================================================
// Scenario 2: \0 delimiter boundary key
// Scenario: key contains \0 (used internally by engine for table name prefix delimiter)
// Expected: Engine handles correctly, no key boundary confusion
// =============================================================================
#[test]
fn nul_byte_boundary_keys_are_distinct() {
    let dir = unique_dir("nul-boundary");
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");

    let keys = vec![
        (b"a\0b".as_ref(), b"v1".as_ref()),
        (b"a\0c".as_ref(), b"v2".as_ref()),
        (b"a\0\0b".as_ref(), b"v3".as_ref()),
        (b"a".as_ref(), b"v4".as_ref()),
    ];

    for &(key, value) in &keys {
        engine.write(&[put(key, value)]).expect("write");
    }

    let snap = engine.snapshot_ts();

    for &(key, expected) in &keys {
        let got = engine.get(key, snap).expect("get");
        assert_eq!(
            got.as_deref(),
            Some(expected),
            "key {:?} must have exact value {:?}",
            key, expected
        );
    }

    // Negative assertion: different keys must not have the same value
    let v1 = engine.get(b"a\0b", snap).expect("get").as_deref().map(|v| v.to_vec());
    let v2 = engine.get(b"a\0c", snap).expect("get").as_deref().map(|v| v.to_vec());
    assert_ne!(v1, v2, "different NUL-delimited keys must be distinct");
}

// =============================================================================
// Scenario 3: Large value write
// Scenario: User writes a 64KB large value
// Expected: Engine handles normally, round-trip is correct
// =============================================================================
#[test]
fn large_value_round_trip() {
    let dir = unique_dir("large-val");
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");

    let large_value = vec![0x42_u8; 65536]; // 64KB
    let tx = engine.write(&[put(b"big", &large_value)]).expect("write large");
    let got = engine.get(b"big", tx).expect("get");
    assert_eq!(got.as_deref(), Some(&large_value[..]), "64KB value round-trip");

    // Negative assertion: must not be the wrong value
    let wrong = vec![0x43_u8; 65536];
    assert_ne!(got.as_deref(), Some(&wrong[..]), "must not return wrong value");
}

// =============================================================================
// Scenario 4: Duplicate key overwrite
// Scenario: Same key written multiple times in the same transaction
// Expected: Last write wins
// =============================================================================
#[test]
fn same_transaction_last_write_wins() {
    let dir = unique_dir("last-wins");
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");

    let tx = engine.write(&[
        put(b"key", b"first"),
        put(b"key", b"second"),
        put(b"key", b"third"),
    ]).expect("write");

    let got = engine.get(b"key", tx).expect("get");
    assert_eq!(
        got.as_deref(),
        Some(b"third".as_ref()),
        "last write must win"
    );

    // Negative assertion: must not be an intermediate value
    assert_ne!(got.as_deref(), Some(b"first".as_ref()));
    assert_ne!(got.as_deref(), Some(b"second".as_ref()));
}

// =============================================================================
// Scenario 5: Empty memtable flush
// Scenario: flush empty memtable
// Expected: Returns None (no SST output)
// =============================================================================
#[test]
fn flush_empty_memtable_returns_none() {
    let dir = unique_dir("empty-flush");
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");

    let result = engine.flush().expect("flush");
    assert!(result.is_none(), "flush of empty memtable must return None");

    // Negative assertion: no SST files must be produced
    let sst_files = engine.sst_file_ids();
    assert!(sst_files.is_empty(), "no SST files must be created");
}

// =============================================================================
// Scenario 6: Duplicate flush
// Scenario: flush again after flush
// Expected: Second flush returns None
// =============================================================================
#[test]
fn double_flush_returns_none_on_second() {
    let dir = unique_dir("double-flush");
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");

    engine.write(&[put(b"k", b"v")]).expect("write");
    let first = engine.flush().expect("first flush");
    assert!(first.is_some(), "first flush must produce SST");

    let second = engine.flush().expect("second flush");
    assert!(second.is_none(), "second flush on empty memtable must return None");
}

// =============================================================================
// Scenario 7: Empty table scan
// Scenario: Scan a key prefix that doesn't exist in memtable
// Expected: Returns empty result, no crash
// =============================================================================
#[test]
fn scan_empty_table_returns_empty() {
    let dir = unique_dir("empty-scan");
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");

    // Scan a key prefix with no data
    let snap = engine.snapshot_ts();
    let result = engine.scan_range(b"t\0", b"t\0\xff", snap).expect("scan");
    assert!(result.is_empty(), "scan of empty prefix must return zero results");
}

// =============================================================================
// Scenario 8: Range scan boundary
// Scenario: Different range scan boundaries
// Expected: lower > upper returns empty, lower == upper returns empty
// =============================================================================
#[test]
fn range_scan_boundary_behaviors() {
    let dir = unique_dir("scan-boundary");
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");

    engine.write(&[put(b"a", b"1"), put(b"b", b"2"), put(b"c", b"3")]).expect("write");
    let snap = engine.snapshot_ts();

    // lower == upper: returns empty
    let empty = engine.scan_range(b"b", b"b", snap).expect("scan");
    assert!(empty.is_empty(), "lower == upper must return empty");

    // lower > upper: returns empty
    let empty2 = engine.scan_range(b"c", b"a", snap).expect("scan");
    assert!(empty2.is_empty(), "lower > upper must return empty");

    // Normal range
    let mid = engine.scan_range(b"a", b"c", snap).expect("scan");
    assert_eq!(mid.len(), 2, "[a, c) must contain 2 keys: a and b");
    assert_eq!(mid[0].0.as_ref(), b"a", "first key must be a");
    assert_eq!(mid[1].0.as_ref(), b"b", "second key must be b");

    // Negative assertion: must not contain c
    let has_c = mid.iter().any(|(k, _)| k.as_ref() == b"c");
    assert!(!has_c, "c must not be in [a, c) range");
}

// =============================================================================
// Scenario 9: Error path — empty batch
// Scenario: Submit empty mutations
// Expected: Returns EmptyBatch error
// =============================================================================
#[test]
fn empty_batch_rejected() {
    let dir = unique_dir("empty-batch");
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");

    let result = engine.write(&[]);
    assert!(result.is_err(), "empty batch must be rejected");
    match result {
        Err(EngineError::EmptyBatch) => { /* expected */ }
        Err(e) => panic!("expected EmptyBatch, got {e:?}"),
        Ok(_) => panic!("must not succeed"),
    }
}

// =============================================================================
// Scenario 10: Error path — batch containing only delete
// Scenario: Delete a nonexistent key
// Expected: Commit succeeds normally, get returns None
// =============================================================================
#[test]
fn delete_nonexistent_key_succeeds() {
    let dir = unique_dir("delete-nonexistent");
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");

    let tx = engine.write(&[delete(b"never_existed")]).expect("delete");
    let got = engine.get(b"never_existed", tx).expect("get");
    assert_eq!(got, None, "deleted nonexistent key must return None");
}

// =============================================================================
// Scenario 11: put/get consistency under concurrent write pressure
// Scenario: Rapid consecutive writes to the same key
// Expected: Each get returns the latest value
// =============================================================================
#[test]
fn rapid_writes_to_same_key_are_visible() {
    let dir = unique_dir("rapid-writes");
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");

    let mut prev_ts = 0u64;
    for i in 0..100u64 {
        let value = format!("v{i}");
        let tx = engine.write(&[put(b"key", value.as_bytes())]).expect("write");
        // Positive assertion: current commit_ts sees the latest value
        let got = engine.get(b"key", tx).expect("get");
        assert_eq!(got.as_deref(), Some(value.as_bytes()), "iteration {i}: current value");
        // Negative assertion: old snapshot must not see new value
        if prev_ts > 0 {
            let old = engine.get(b"key", prev_ts).expect("get");
            let expected_old = format!("v{}", i - 1);
            assert_eq!(old.as_deref(), Some(expected_old.as_bytes()),
                "snapshot {prev_ts} must see old value, not v{i}");
        }
        prev_ts = tx;
    }
    // Negative assertion: final snapshot must not see intermediate values
    let final_val = engine.get(b"key", engine.snapshot_ts()).expect("get");
    assert_eq!(final_val.as_deref(), Some(b"v99".as_ref()), "final must be v99");
    assert_ne!(final_val.as_deref(), Some(b"v0".as_ref()), "final must not be v0");
    assert_ne!(final_val.as_deref(), Some(b"v50".as_ref()), "final must not be v50");
}

// =============================================================================
// Scenario 12: get correctness after flush
// Scenario: get immediately after flush
// Expected: Value is returned correctly
// =============================================================================
#[test]
fn get_after_flush_returns_correct_value() {
    let dir = unique_dir("get-after-flush");
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");

    engine.write(&[put(b"k", b"v")]).expect("write");
    let handle = engine.flush().expect("flush").expect("SST");
    assert!(handle.rows > 0, "flush must produce rows");

    let snap = engine.snapshot_ts();
    let got = engine.get(b"k", snap).expect("get");
    assert_eq!(
        got.as_deref(),
        Some(b"v".as_ref()),
        "value must be correct after flush"
    );

    // Negative assertion: SST file must exist
    let sst_path = dir.path().join(format!("sst-{:020}.vortex", handle.file_id));
    assert!(sst_path.exists(), "SST file must exist on disk");
}

// =============================================================================
// Scenario 13: Comprehensive boundary key test
// Scenario: All boundary keys can be correctly written and read
// Expected: Each boundary key has correct round-trip
// =============================================================================
#[test]
fn boundary_keys_round_trip() {
    let dir = unique_dir("all-boundaries");
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");

    let keys = boundaries::boundary_keys();
    let names = boundaries::boundary_key_names();

    for (i, key) in keys.iter().enumerate() {
        let value = format!("boundary_val_{i}");
        let tx = engine
            .write(&[put(key, value.as_bytes())])
            .unwrap_or_else(|_| panic!("write failed for boundary key: {}", names[i]));

        let got = engine
            .get(key, tx)
            .unwrap_or_else(|_| panic!("get failed for boundary key: {}", names[i]));

        assert_eq!(
            got.as_deref(),
            Some(value.as_bytes()),
            "boundary key '{}' round-trip failed", names[i]
        );
    }
}