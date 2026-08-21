#![allow(clippy::expect_used, clippy::unwrap_used, clippy::indexing_slicing)]

//! scan_table / update_plan (PR 2.13 / PR 2.10) integration tests.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};
use hatp_engine::{Engine, EngineConfig, EngineError};

#[test]
fn scan_table_projection_out_of_range_errors() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(EngineConfig::new(dir.path())).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Int64, false),
    ]));

    // Empty table + out-of-bounds projection index (2 >= 2 columns) → explicit
    // CorruptMessage, not a raw Arrow error.
    let err = engine
        .scan_table("t", Some(&[2]), Some(&schema))
        .expect_err("projection index out of range must error");
    assert!(matches!(err, EngineError::CorruptMessage(_)));
}

#[test]
fn scan_table_valid_projection_on_empty_table_returns_projected_schema() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(EngineConfig::new(dir.path())).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Int64, false),
    ]));

    let batches = engine.scan_table("t", Some(&[0]), Some(&schema)).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_columns(), 1);
}

#[test]
fn scan_table_batched_chunks_into_batch_size() {
    use arrow_array::RecordBatch;

    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(EngineConfig::new(dir.path())).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(arrow_array::Int64Array::from(vec![1_i64, 2, 3, 4])),
            Arc::new(arrow_array::StringArray::from(vec!["a", "b", "c", "d"])),
        ],
    )
    .unwrap();
    let mutations = engine
        .ingest_batches(&[batch], "t", &["id".to_string()])
        .unwrap();
    engine.write(&mutations).unwrap();

    // batch_size = 2 → 2 batches, each ≤ 2 rows, 4 rows total (PR 2.8).
    let partitions = engine.scan_table_arrow("t", None, None, 2, None).unwrap();
    let batches: Vec<RecordBatch> = partitions.into_iter().flatten().collect();
    assert_eq!(batches.len(), 2);
    for b in &batches {
        assert!(b.num_rows() <= 2);
    }
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 4);

    // Default scan_table → 1 batch of 4 rows (no chunking).
    let all = engine.scan_table("t", None, None).unwrap();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].num_rows(), 4);
}

#[test]
fn table_row_count_is_per_table_and_survives_restart() {
    use arrow_array::RecordBatch;

    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(EngineConfig::new(dir.path())).unwrap();

    // Table a: 3 rows, table b: 2 rows.
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let batch_a = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(arrow_array::Int64Array::from(vec![1_i64, 2, 3]))],
    )
    .unwrap();
    let batch_b = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(arrow_array::Int64Array::from(vec![1_i64, 2]))],
    )
    .unwrap();
    let mutations_a = engine
        .ingest_batches(&[batch_a], "a", &["id".to_string()])
        .unwrap();
    engine.write(&mutations_a).unwrap();
    let mutations_b = engine
        .ingest_batches(&[batch_b], "b", &["id".to_string()])
        .unwrap();
    engine.write(&mutations_b).unwrap();
    engine.flush().unwrap();

    // After flush: a=3, b=2 (per-table, not global 5).
    assert_eq!(engine.table_row_count("a"), 3);
    assert_eq!(engine.table_row_count("b"), 2);

    // Survives restart (open recovers and rebuilds per-table counts).
    drop(engine);
    let engine2 = Engine::open(EngineConfig::new(dir.path())).unwrap();
    assert_eq!(engine2.table_row_count("a"), 3);
    assert_eq!(engine2.table_row_count("b"), 2);
}

// =============================================================================
// SC-05: Multi-version range scan — same key, multiple versions, visibility at different snapshots
// =============================================================================

/// Same key with multiple versions; different snapshots see different versions via scan_range
#[test]
fn scan_range_sees_correct_version_at_snapshot() {
    use bytes::Bytes;
    use hatp_engine::{Engine, EngineConfig, Mutation};

    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(EngineConfig::new(dir.path())).unwrap();

    // Write v1@ts=1, v2@ts=2, v3@ts=3 (same key overwritten 3 times)
    engine.write(&[Mutation::Put {
        key: Bytes::from_static(b"multi"),
        value: Bytes::from_static(b"v1"),
    }]).unwrap(); // ts=1
    engine.write(&[Mutation::Put {
        key: Bytes::from_static(b"multi"),
        value: Bytes::from_static(b"v2"),
    }]).unwrap(); // ts=2
    engine.write(&[Mutation::Put {
        key: Bytes::from_static(b"multi"),
        value: Bytes::from_static(b"v3"),
    }]).unwrap(); // ts=3
    engine.flush().unwrap();

    // snapshot=1: only sees v1
    let r1 = engine.scan_range(b"m", b"n", 1).unwrap();
    assert_eq!(r1.len(), 1, "snapshot=1 must see exactly 1 key");
    assert_eq!(r1[0].1.as_ref(), b"v1", "snapshot=1 must see v1");
    assert_ne!(r1[0].1.as_ref(), b"v2", "snapshot=1 must NOT see v2");

    // snapshot=2: sees v2 (v3 not visible)
    let r2 = engine.scan_range(b"m", b"n", 2).unwrap();
    assert_eq!(r2[0].1.as_ref(), b"v2", "snapshot=2 must see v2");

    // snapshot=3: sees v3 (latest)
    let r3 = engine.scan_range(b"m", b"n", 3).unwrap();
    assert_eq!(r3[0].1.as_ref(), b"v3", "snapshot=3 must see v3");
}

/// Delete operations (tombstones) are correctly filtered in scan_range
#[test]
fn scan_range_filters_tombstones() {
    use bytes::Bytes;
    use hatp_engine::{Engine, EngineConfig, Mutation};

    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(EngineConfig::new(dir.path())).unwrap();

    engine.write(&[Mutation::Put {
        key: Bytes::from_static(b"alive"),
        value: Bytes::from_static(b"keep"),
    }]).unwrap(); // ts=1
    engine.write(&[Mutation::Put {
        key: Bytes::from_static(b"dead"),
        value: Bytes::from_static(b"gone"),
    }]).unwrap(); // ts=2
    engine.write(&[Mutation::Delete {
        key: Bytes::from_static(b"dead"),
    }]).unwrap(); // ts=3: tombstone

    let r = engine.scan_range(b"a", b"e", 3).unwrap();

    // Positive assertion: only alive appears in results
    assert_eq!(r.len(), 1, "tombstone key must be filtered out");
    assert_eq!(r[0].0.as_ref(), b"alive", "only alive key must remain");
    assert_eq!(r[0].1.as_ref(), b"keep", "alive value must be correct");

    // Negative assertion: dead must not appear in results
    let has_dead = r.iter().any(|(k, _)| k.as_ref() == b"dead");
    assert!(!has_dead, "tombstone key must not appear in scan results");
}

/// scan_all returns every visible version including tombstones (None value),
/// used by replication bootstrap / snapshot sync. A finite upper bound would
/// silently exclude keys — scan_all has no upper bound.
#[test]
fn scan_all_returns_tombstones_and_live_values() {
    use bytes::Bytes;
    use hatp_engine::{Engine, EngineConfig, Mutation};

    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(EngineConfig::new(dir.path())).unwrap();

    // Write 3 keys, flush, then delete one
    engine.write(&[
        Mutation::Put { key: Bytes::from_static(b"k1"), value: Bytes::from_static(b"v1") },
        Mutation::Put { key: Bytes::from_static(b"k2"), value: Bytes::from_static(b"v2") },
        Mutation::Put { key: Bytes::from_static(b"k3"), value: Bytes::from_static(b"v3") },
    ]).unwrap(); // ts=1
    engine.flush().unwrap();
    engine.write(&[Mutation::Delete { key: Bytes::from_static(b"k2") }]).unwrap(); // ts=2: tombstone

    let snap = engine.snapshot_ts();
    let all = engine.scan_all(snap).unwrap();

    // Positive: returns all 3 keys including tombstone
    assert_eq!(all.len(), 3, "scan_all must return all keys including tombstone");
    let k1 = all.iter().find(|(k, _)| k.as_ref() == b"k1").unwrap();
    let k2 = all.iter().find(|(k, _)| k.as_ref() == b"k2").unwrap();
    let k3 = all.iter().find(|(k, _)| k.as_ref() == b"k3").unwrap();
    assert_eq!(k1.1.as_deref(), Some(b"v1".as_ref()), "live value must be present");
    assert_eq!(k2.1, None, "tombstone must be None");
    assert_eq!(k3.1.as_deref(), Some(b"v3".as_ref()), "live value must be present");

    // Negative: must NOT silently drop the tombstone
    assert!(all.iter().any(|(k, v)| k.as_ref() == b"k2" && v.is_none()),
        "tombstone key must appear with None value");
}

/// scan_all on empty database returns empty Vec, not an error.
#[test]
fn scan_all_empty_database_returns_empty() {
    use hatp_engine::{Engine, EngineConfig};

    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(EngineConfig::new(dir.path())).unwrap();
    let snap = engine.snapshot_ts();
    let all = engine.scan_all(snap).unwrap();
    assert!(all.is_empty(), "empty database must return empty Vec");
}

/// scan_all filters keys belonging to dropped tables.
#[test]
fn scan_all_filters_dropped_table_keys() {
    use bytes::Bytes;
    use hatp_engine::{Engine, EngineConfig, Mutation};

    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(EngineConfig::new(dir.path())).unwrap();

    // Write a key with a table-like prefix
    engine.write(&[Mutation::Put {
        key: Bytes::from_static(b"mytable\0pk1"),
        value: Bytes::from_static(b"v1"),
    }]).unwrap();
    engine.flush().unwrap();

    // Drop the table prefix
    engine.drop_table_prefix("mytable").unwrap();

    let snap = engine.snapshot_ts();
    let all = engine.scan_all(snap).unwrap();

    // Negative: dropped table key must not appear
    let has_dropped = all.iter().any(|(k, _)| k.as_ref().starts_with(b"mytable\0"));
    assert!(!has_dropped, "dropped table keys must be filtered from scan_all");
}