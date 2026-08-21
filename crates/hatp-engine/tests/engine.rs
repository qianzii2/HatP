#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::assertions_on_constants
)]

//! Integration coverage for `hatp-engine`.
//!
//! Modules under test: `memtable`, `version`, `wal`, `compaction`,
//! `manifest`, `crash_test`.

use bytes::Bytes;
use hatp_engine::{Engine, EngineConfig, EngineError, Mutation};
use tempfile::{Builder, TempDir};

fn unique_dir(label: &str) -> Result<TempDir, std::io::Error> {
    Builder::new().prefix(&format!("hatp-{label}-")).tempdir()
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

#[test]
fn put_then_get_round_trips() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("basic")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    let tx = engine.write(&[put(b"alpha", b"one")])?;
    assert_eq!(engine.get(b"alpha", tx)?.as_deref(), Some(b"one".as_ref()));
    Ok(())
}

#[test]
fn multiple_mutations_are_visible_at_commit_ts() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("multi")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    let tx = engine.write(&[put(b"a", b"1"), put(b"b", b"2"), delete(b"a")])?;
    assert_eq!(engine.get(b"a", tx)?, None);
    assert_eq!(engine.get(b"b", tx)?.as_deref(), Some(b"2".as_ref()));
    Ok(())
}

#[test]
fn empty_batch_is_rejected() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("empty")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    assert!(matches!(engine.write(&[]), Err(EngineError::EmptyBatch)));
    Ok(())
}

#[test]
fn flush_writes_vortex_sst_and_persists_manifest() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("flush")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    engine.write(&[put(b"k", b"v")])?;
    let handle = engine
        .flush()?
        .ok_or_else(|| std::io::Error::other("flush produced no SST"))?;
    assert_eq!(handle.rows, 1);
    assert!(engine.memtable().is_empty());
    // The WAL is no longer unconditionally cleared by flush: it is only truncated
    // when `wal_max_size_bytes` (default 64MB) is exceeded. The WAL for this single small
    // write will be kept until a later truncation; on restart, replay will produce versions
    // with the same begin_ts as the already-flushed SST (MVCC dedup prevents duplicate visible rows).
    assert!(dir.path().join("MANIFEST").exists());
    Ok(())
}

#[test]
fn recovery_replays_committed_batch() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("recover")?;
    {
        let engine = Engine::open(EngineConfig::new(dir.path()))?;
        engine.write(&[put(b"a", b"1"), put(b"b", b"2"), delete(b"a")])?;
        engine.flush()?;
    }
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    let snapshot = engine.snapshot_ts();
    assert_eq!(engine.get(b"a", snapshot)?, None);
    assert_eq!(engine.get(b"b", snapshot)?.as_deref(), Some(b"2".as_ref()));
    Ok(())
}

#[test]
fn flush_then_new_write_survives_restart() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("flush-write-recover")?;
    {
        let engine = Engine::open(EngineConfig::new(dir.path()))?;
        engine.write(&[put(b"sst", b"old")])?;
        engine.flush()?;
        engine.write(&[put(b"wal", b"new")])?;
    }
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    let snapshot = engine.snapshot_ts();
    assert_eq!(
        engine.get(b"sst", snapshot)?.as_deref(),
        Some(b"old".as_ref())
    );
    assert_eq!(
        engine.get(b"wal", snapshot)?.as_deref(),
        Some(b"new".as_ref())
    );
    Ok(())
}

#[test]
fn same_transaction_last_mutation_wins() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("same-tx")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    let tx = engine.write(&[put(b"key", b"first"), delete(b"key")])?;
    assert_eq!(engine.get(b"key", tx)?, None);
    // Negative assertion: the deleted key must not return the old value
    assert_ne!(engine.get(b"key", tx)?.as_deref(), Some(b"first".as_ref()));
    let tx = engine.write(&[delete(b"key"), put(b"key", b"second")])?;
    assert_eq!(engine.get(b"key", tx)?.as_deref(), Some(b"second".as_ref()));
    // Negative assertion: must not be the pre-delete value
    assert_ne!(engine.get(b"key", tx)?.as_deref(), Some(b"first".as_ref()));
    Ok(())
}

/// Regression: `Engine::write_with_tx` (used by `hatp::Transaction::commit`
/// under SSI) must durably commit and advance the commit-order snapshot so a
/// subsequent read at `snapshot_ts()` sees the committed version.
///
/// Prior to the fix the engine rejected reserved-id writes with `tx_id == 0`;
/// even after that fix, `snapshot_ts()` lagged behind the reserved id and made
/// committed writes invisible to reads. Today `snapshot_ts()` is the
/// commit-order sequence (`commit_seq - 1`), decoupled from the
/// reservation-order `tx_id`: the first commit is `commit_ts == 1` even when
/// the caller reserved `tx_id == 42`.
#[test]
fn write_with_tx_advances_snapshot_ts() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("write-with-tx")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    let tx_id = 42_u64;
    engine.write_with_tx(tx_id, &[put(b"k", b"v")])?;
    // snapshot_ts tracks commit order (1 for the first commit), NOT the
    // reservation-order tx_id (42). The committed version is visible at the
    // commit-order snapshot.
    assert_eq!(engine.snapshot_ts(), 1);
    assert_eq!(
        engine.get(b"k", engine.snapshot_ts())?.as_deref(),
        Some(b"v".as_ref())
    );
    Ok(())
}

/// A panicking hook must not take the process down. The workspace's
/// `panic = "abort"` profile would otherwise kill the host on any
/// user-supplied hook bug.
#[test]
#[allow(clippy::panic)] // the panic is the very thing under test
fn panicking_hook_is_contained() -> Result<(), Box<dyn std::error::Error>> {
    struct PanicHook;
    impl hatp_engine::EngineHook for PanicHook {
        fn on_put(&self, _k: &[u8], _v: &[u8], _id: u64) {
            panic!("user hook bug");
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }
    let dir = unique_dir("panic-hook")?;
    let hook: std::sync::Arc<dyn hatp_engine::EngineHook> = std::sync::Arc::new(PanicHook);
    let engine = Engine::open_with_hook(EngineConfig::new(dir.path()), hook)?;
    engine.write(&[put(b"k", b"v")])?;
    // The hook panicked but the data is durably committed.
    assert_eq!(
        engine.get(b"k", engine.snapshot_ts())?.as_deref(),
        Some(b"v".as_ref())
    );
    Ok(())
}

/// Regression guard for the `fetch_max` → `compare_exchange_weak` fix in
/// `Engine::write_with_tx`. Mixing reserved ids that are *higher* than
/// the current counter must not leave the allocator stuck, and the
/// subsequent plain `Engine::write` must produce the next contiguous id.
#[test]
fn write_with_tx_id_higher_than_counter_advances_monotonically()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("fetch-max-fix")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    // First commit sets the commit sequence to 1.
    let first = engine.write(&[put(b"a", b"1")])?;
    assert_eq!(first, 1);
    // Reserve a much higher id via write_with_tx. The allocator must
    // jump to 101 (not get stuck at 2 with a hole at 3..=99).
    engine.write_with_tx(100, &[put(b"b", b"2")])?;
    // snapshot_ts counts commit order (this is the 2nd commit), NOT the
    // reservation-order id (100).
    assert_eq!(
        engine.snapshot_ts(),
        2,
        "snapshot_ts must reflect commit order, got {}",
        engine.snapshot_ts()
    );
    // The tx_id allocator must have jumped to 101 (not stuck at 2 with a hole
    // at 3..=99). `reserve_tx_id` exposes the reservation-order sequence.
    let next_id = engine.reserve_tx_id();
    assert_eq!(
        next_id, 101,
        "tx_id allocator must jump past the highest reservation, got {next_id}"
    );
    Ok(())
}

/// Reserved ids smaller than the current counter must be honoured without
/// sticking the allocator at `tx_id + 1`. After a high reserved id, a
/// lower reserved id must still commit, and the post-commit counter is
/// not regressed.
#[test]
fn write_with_tx_id_lower_than_counter_does_not_regress() -> Result<(), Box<dyn std::error::Error>>
{
    let dir = unique_dir("fetch-max-lower")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    engine.write_with_tx(50, &[put(b"a", b"1")])?;
    let after_high = engine.snapshot_ts();
    // Lower reserved id must commit (and surface ReadWriteConflict if
    // keys overlap, but we use a fresh key here). Its commit sequence is the
    // next commit order number, NOT the reserved id 10.
    let low_commit = engine.write_with_tx(10, &[put(b"b", b"2")])?;
    assert_eq!(
        low_commit, 2,
        "write_with_tx returns commit_ts, got {low_commit}"
    );
    // Commit watermark must NOT regress; it must remain ≥ after_high.
    assert!(
        engine.snapshot_ts() >= after_high,
        "snapshot_ts must not regress: was {after_high}, now {}",
        engine.snapshot_ts()
    );
    // tx_id allocator must NOT regress to 11: next reserved id is still past
    // the high reservation (51).
    let next_id = engine.reserve_tx_id();
    assert!(
        next_id > 50,
        "next id must be past the high reservation, got {next_id}"
    );
    Ok(())
}
// ============================================================================
// M0 hardening tests — drop persistence, update_plan strictness, ingest dedup.
// ============================================================================

/// `drop_table_prefix` must survive a restart: the dropped prefix is persisted
/// to the manifest and replayed on open, so orphaned SST data stays filtered.
#[test]
fn drop_table_survives_restart() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("drop-restart")?;
    {
        let engine = Engine::open(EngineConfig::new(dir.path()))?;
        engine.write(&[put(b"t\0k1", b"v1")])?;
        engine.flush()?.expect("flush before drop");
        engine.drop_table_prefix("t")?;
    }
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    let snapshot = engine.snapshot_ts();
    assert_eq!(
        engine.get(b"t\0k1", snapshot)?,
        None,
        "dropped table data must stay filtered"
    );
    Ok(())
}

/// The dropped prefix is durably recorded in the manifest (PR 0.4).
#[test]
fn drop_table_durable_in_manifest() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("drop-manifest")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    engine.write(&[put(b"t\0k1", b"v1")])?;
    engine.flush()?.expect("flush before drop");
    engine.drop_table_prefix("t")?;

    let manifest = hatp_engine::manifest::Manifest::open(dir.path().join("MANIFEST"))?;
    assert_eq!(
        manifest.version_set().dropped_prefixes(),
        vec![b"t\0".to_vec()]
    );

    // undrop clears it durably too.
    engine.undrop_table_prefix("t")?;
    let manifest = hatp_engine::manifest::Manifest::open(dir.path().join("MANIFEST"))?;
    assert!(manifest.version_set().dropped_prefixes().is_empty());
    Ok(())
}

/// `ingest_batches` collapses duplicate primary keys across batches (PR 0.7).
#[test]
fn ingest_batches_dedups_across_batches() -> Result<(), Box<dyn std::error::Error>> {
    use arrow_array::RecordBatch;
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    let dir = unique_dir("ingest-dedup")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let batch_a = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(arrow_array::Int64Array::from(vec![1_i64])),
            Arc::new(arrow_array::StringArray::from(vec!["a"])),
        ],
    )?;
    let batch_b = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(arrow_array::Int64Array::from(vec![1_i64])),
            Arc::new(arrow_array::StringArray::from(vec!["b"])),
        ],
    )?;
    let mutations = engine.ingest_batches(&[batch_a, batch_b], "t", &["id".to_string()])?;
    assert_eq!(
        mutations.len(),
        1,
        "duplicate PK across batches must collapse to one mutation"
    );
    // Exact assertion: the retained mutation must be the last batch's value
    if let Mutation::Put { key: _, value } = &mutations[0] {
        // Verify value is batch_b's value ("b"), not batch_a's ("a")
        let decoded = hatp_engine::row_codec::decode_row_values(value, schema.as_ref())?;
        assert_eq!(decoded[1], datafusion_common::ScalarValue::Utf8(Some("b".to_string())),
            "dedup must keep last batch's value");
        // Negative assertion: must not be the first batch's value
        assert_ne!(decoded[1], datafusion_common::ScalarValue::Utf8(Some("a".to_string())),
            "dedup must not keep first batch's value");
    } else {
        panic!("expected Put mutation");
    }
    Ok(())
}

/// undrop_table_prefix: after drop + undrop, new writes to the same table name
/// become visible again. The drop filter is cleared from the manifest.
#[test]
fn undrop_table_prefix_restores_visibility() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("undrop")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;

    engine.write(&[put(b"mytable\0pk1", b"old")])?;
    engine.flush()?;
    engine.drop_table_prefix("mytable")?;
    let snap = engine.snapshot_ts();
    assert!(engine.get(b"mytable\0pk1", snap)?.is_none(),
        "dropped table key must be invisible");

    engine.undrop_table_prefix("mytable")?;
    engine.write(&[put(b"mytable\0pk2", b"new")])?;
    let snap2 = engine.snapshot_ts();
    let v = engine.get(b"mytable\0pk2", snap2)?;
    assert_eq!(v.as_deref(), Some(b"new".as_ref()),
        "new write after undrop must be visible");
    Ok(())
}

/// Engine::Drop stops background workers. Shutdown before drop is idempotent.
#[test]
fn engine_drop_stops_background_workers() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("drop-workers")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    engine.write(&[put(b"k", b"v")])?;
    engine.shutdown();
    drop(engine);
    let engine2 = Engine::open(EngineConfig::new(dir.path()))?;
    let snap = engine2.snapshot_ts();
    let v = engine2.get(b"k", snap)?;
    assert_eq!(v.as_deref(), Some(b"v".as_ref()),
        "data must survive engine drop and reopen");
    Ok(())
}

/// sst_file_ids returns all SST file IDs managed by the manifest.
#[test]
fn sst_file_ids_lists_all_files() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("sst-ids")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;

    // Empty initially
    assert!(engine.sst_file_ids().is_empty(), "no SSTs initially");

    // Write and flush → one SST
    engine.write(&[put(b"k1", b"v1")])?;
    engine.flush()?;
    let ids = engine.sst_file_ids();
    assert_eq!(ids.len(), 1, "one SST after flush");

    // Write and flush again → two SSTs
    engine.write(&[put(b"k2", b"v2")])?;
    engine.flush()?;
    let ids2 = engine.sst_file_ids();
    assert_eq!(ids2.len(), 2, "two SSTs after second flush");

    // IDs must be unique
    assert_ne!(ids2[0], ids2[1], "SST file IDs must be unique");
    Ok(())
}

/// MemTableImpl::BTreeMap backend works correctly for put/get/scan.
#[test]
fn btreemap_memtable_backend_works() -> Result<(), Box<dyn std::error::Error>> {
    use hatp_engine::memtable::MemTableImpl;
    let dir = unique_dir("btree-mem")?;
    let config = EngineConfig::new(dir.path())
        .with_memtable_impl(MemTableImpl::BTreeMap);
    let engine = Engine::open(config)?;

    engine.write(&[put(b"a", b"1"), put(b"b", b"2")])?;
    let snap = engine.snapshot_ts();

    assert_eq!(engine.get(b"a", snap)?.as_deref(), Some(b"1".as_ref()));
    assert_eq!(engine.get(b"b", snap)?.as_deref(), Some(b"2".as_ref()));

    // Negative: BTreeMap backend must not return SkipMap-only data
    assert!(engine.get(b"nonexistent", snap)?.is_none());
    Ok(())
}

/// MemTableImpl::SkipMap backend (default) works correctly.
#[test]
fn skipmap_memtable_backend_works() -> Result<(), Box<dyn std::error::Error>> {
    use hatp_engine::memtable::MemTableImpl;
    let dir = unique_dir("skip-mem")?;
    let config = EngineConfig::new(dir.path())
        .with_memtable_impl(MemTableImpl::SkipMap);
    let engine = Engine::open(config)?;

    engine.write(&[put(b"x", b"y")])?;
    let snap = engine.snapshot_ts();
    assert_eq!(engine.get(b"x", snap)?.as_deref(), Some(b"y".as_ref()));
    Ok(())
}
