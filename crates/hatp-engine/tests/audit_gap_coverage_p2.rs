#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::assertions_on_constants
)]

//! Audit gap coverage P2: SST sidecar missing/corrupt recovery,
//! rayon parallel scan correctness.

use bytes::Bytes;
use hatp_engine::{Engine, EngineConfig, Mutation};
use std::sync::Arc;

fn put(key: &[u8], value: &[u8]) -> Mutation {
    Mutation::Put {
        key: Bytes::copy_from_slice(key),
        value: Bytes::copy_from_slice(value),
    }
}

/// After deleting the bloom sidecar file, Engine::get still returns the
/// correct value by falling through to the full SST read path.
#[test]
fn get_succeeds_after_bloom_sidecar_deleted() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(EngineConfig::new(dir.path())).unwrap();

    engine.write(&[put(b"bf_key", b"bf_value")]).unwrap();
    engine.flush().unwrap();

    // Delete the bloom sidecar file to force the fallback path
    let mut deleted = false;
    for entry in std::fs::read_dir(dir.path()).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("bloom-") && name.ends_with(".bf") {
            std::fs::remove_file(entry.path()).unwrap();
            deleted = true;
            break;
        }
    }
    assert!(deleted, "must have deleted a bloom sidecar file");

    // Positive: get still returns correct data after bloom deleted
    let snap = engine.snapshot_ts();
    let v = engine.get(b"bf_key", snap).unwrap();
    assert_eq!(
        v.as_deref(),
        Some(b"bf_value".as_ref()),
        "get must return correct value after bloom sidecar deleted"
    );

    // Negative: must not return wrong data
    let missing = engine.get(b"nonexistent", snap).unwrap();
    assert!(missing.is_none(), "missing key must return None");
}

/// After deleting the key-index sidecar file, Engine::get still returns
/// the correct value by falling through to the Vortex read path.
#[test]
fn get_succeeds_after_key_index_sidecar_deleted() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(EngineConfig::new(dir.path())).unwrap();

    engine.write(&[put(b"kidx_key", b"kidx_value")]).unwrap();
    engine.flush().unwrap();

    // Delete the key-index sidecar file
    let mut deleted = false;
    for entry in std::fs::read_dir(dir.path()).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("kidx-") && name.ends_with(".bin") {
            std::fs::remove_file(entry.path()).unwrap();
            deleted = true;
            break;
        }
    }
    assert!(deleted, "must have deleted a key-index sidecar file");

    // Positive: get still returns correct data after key-index deleted
    let snap = engine.snapshot_ts();
    let v = engine.get(b"kidx_key", snap).unwrap();
    assert_eq!(
        v.as_deref(),
        Some(b"kidx_value".as_ref()),
        "get must return correct value after key-index sidecar deleted"
    );

    // Negative: must not return wrong data
    let missing = engine.get(b"nonexistent", snap).unwrap();
    assert!(missing.is_none(), "missing key must return None");
}

/// scan_table_arrow returns correct row counts from multiple SST files.
/// Each SST is one DataFusion partition; rayon parallelizes across them.
#[test]
fn scan_table_arrow_returns_all_rows_from_multiple_ssts() {
    use arrow_array::{Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    let dir = tempfile::tempdir().unwrap();
    let engine = Arc::new(Engine::open(EngineConfig::new(dir.path())).unwrap());

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    // Create 3 separate SSTs by writing and flushing in batches
    for batch in 0..3_u8 {
        let start = batch as i64 * 100;
        let ids: Vec<i64> = (start..start + 100).collect();
        let names: Vec<String> = ids.iter().map(|i| format!("row_{i}")).collect();
        let batch_data = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .unwrap();
        let mutations = engine
            .ingest_batches(&[batch_data], "t", &["id".to_string()])
            .unwrap();
        engine.write(&mutations).unwrap();
        engine.flush().unwrap();
    }

    // Positive: scan returns all 300 rows from 3 SSTs
    let result = engine.scan_table_arrow("t", None, Some(&schema), usize::MAX, None);
    assert!(result.is_ok(), "scan_table_arrow must succeed");
    let partitions = result.unwrap();
    let total_rows: usize = partitions
        .iter()
        .flatten()
        .map(|b| b.num_rows())
        .sum();
    assert_eq!(total_rows, 300, "must return all 300 rows from 3 SSTs");

    // Negative: nonexistent table returns 0 rows, not an error
    let empty = engine.scan_table_arrow("nonexistent", None, Some(&schema), usize::MAX, None);
    assert!(empty.is_ok(), "scan of nonexistent table must not error");
    let empty_rows: usize = empty
        .unwrap()
        .iter()
        .flatten()
        .map(|b| b.num_rows())
        .sum();
    assert_eq!(empty_rows, 0, "nonexistent table must return 0 rows");
}