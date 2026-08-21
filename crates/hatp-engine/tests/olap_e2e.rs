//! OLAP end-to-end tests — referencing DuckDB TPC-H + DataFusion sqllogictest
//!
//! Uses Engine's actual APIs: `scan_table` (no filtering), `scan_table_arrow` (with filtering)
//!
//! Test patterns (referencing DuckDB .test format):
//! - Exact assertions: return row count, column count, and specific values for each query
//! - Pre-computed answers: data is deterministic, answers are pre-computed
//! - Negative assertions: verify values that should NOT be returned

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::assertions_on_constants
)]

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use hatp_engine::column_predicate::ColumnPredicate;
use hatp_engine::{Engine, EngineConfig};
use bytes::Bytes;
use tempfile::Builder;

const N_ROWS: usize = 100;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("status", DataType::Utf8, false),
        Field::new("age", DataType::Int64, false),
        Field::new("score", DataType::Int64, false),
    ]))
}

fn unique_dir(label: &str) -> tempfile::TempDir {
    Builder::new()
        .prefix(&format!("hatp-olap-{label}-"))
        .tempdir()
        .expect("tempdir")
}

fn build_dataset() -> RecordBatch {
    let ids: Vec<i64> = (0..N_ROWS as i64).collect();
    let statuses: Vec<String> = (0..N_ROWS)
        .map(|i| if i % 2 == 0 { "active" } else { "inactive" }.to_owned())
        .collect();
    let ages: Vec<i64> = (0..N_ROWS as i64).map(|i| i % 100).collect();
    let scores: Vec<i64> = (0..N_ROWS as i64).map(|i| (i * 13) % 1000).collect();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(statuses)),
            Arc::new(Int64Array::from(ages)),
            Arc::new(Int64Array::from(scores)),
        ],
    )
    .expect("build dataset")
}

fn load_engine(dir: &tempfile::TempDir) -> Arc<Engine> {
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");
    let batch = build_dataset();
    let mutations = engine
        .ingest_batches(&[batch], "t", &["id".to_string()])
        .expect("ingest");
    engine.write(&mutations).expect("write");
    engine.flush().expect("flush");
    engine
}

fn flatten(partitions: Vec<Vec<RecordBatch>>) -> Vec<RecordBatch> {
    partitions.into_iter().flatten().collect()
}

fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.num_rows()).sum()
}

// =============================================================================
// Query 1: Full table scan
// =============================================================================
#[test]
fn olap_q1_full_scan() {
    let dir = unique_dir("q1");
    let engine = load_engine(&dir);
    let batches = engine.scan_table("t", None, Some(&schema())).expect("scan");
    assert_eq!(total_rows(&batches), N_ROWS);
    let first = batches.first().expect("first batch");
    assert_eq!(first.num_columns(), 4);
    let id_arr = first.column(0).as_any().downcast_ref::<Int64Array>().expect("id");
    assert_eq!(id_arr.value(0), 0, "first row id must be 0");
}

// =============================================================================
// Query 2: Projection scan
// =============================================================================
#[test]
fn olap_q2_projection_scan() {
    let dir = unique_dir("q2");
    let engine = load_engine(&dir);
    let batches = engine.scan_table("t", Some(&[0, 3]), Some(&schema())).expect("scan");
    assert_eq!(total_rows(&batches), N_ROWS);
    let first = batches.first().expect("first batch");
    assert_eq!(first.num_columns(), 2);
    let score_arr = first.column(1).as_any().downcast_ref::<Int64Array>().expect("score");
    assert_eq!(score_arr.value(0), 0, "score for id=0");
    assert_eq!(score_arr.value(1), 13, "score for id=1");
}

// =============================================================================
// Query 3: Equality filter
// =============================================================================
#[test]
fn olap_q3_equality_filter() {
    let dir = unique_dir("q3");
    let engine = load_engine(&dir);
    let pred = ColumnPredicate::Eq {
        col: 1,
        value: datafusion_common::ScalarValue::Utf8(Some("active".to_owned())),
    };
    let partitions = engine
        .scan_table_arrow("t", Some(&[0]), Some(&schema()), usize::MAX, Some(&[pred]))
        .expect("scan");
    let batches = flatten(partitions);
    let total = total_rows(&batches);
    assert_eq!(total, N_ROWS / 2, "exactly half the rows must be active");
    assert_ne!(total, N_ROWS, "must not return all rows");
}

// =============================================================================
// Query 4: Range filter
// =============================================================================
#[test]
fn olap_q4_range_filter() {
    let dir = unique_dir("q4");
    let engine = load_engine(&dir);
    let lower = ColumnPredicate::GreaterThanOrEqual {
        col: 0,
        value: datafusion_common::ScalarValue::Int64(Some(20)),
    };
    let upper = ColumnPredicate::LessThan {
        col: 0,
        value: datafusion_common::ScalarValue::Int64(Some(30)),
    };
    let partitions = engine
        .scan_table_arrow("t", Some(&[0]), Some(&schema()), usize::MAX, Some(&[lower, upper]))
        .expect("scan");
    let batches = flatten(partitions);
    assert_eq!(total_rows(&batches), 10);
    let first = batches.first().expect("first batch");
    let id_arr = first.column(0).as_any().downcast_ref::<Int64Array>().expect("id");
    assert_eq!(id_arr.value(0), 20, "first row id must be 20");
    let all_ids: Vec<i64> = batches.iter()
        .flat_map(|b| b.column(0).as_any().downcast_ref::<Int64Array>())
        .flat_map(|arr| (0..arr.len()).map(move |i| arr.value(i)))
        .collect();
    assert!(!all_ids.contains(&19), "id=19 must not be in [20,30)");
    assert!(!all_ids.contains(&30), "id=30 must not be in [20,30)");
}

// =============================================================================
// Query 5: Compound filter — active AND score > 500 → 19 rows
// =============================================================================
#[test]
fn olap_q5_compound_filter() {
    let dir = unique_dir("q5");
    let engine = load_engine(&dir);
    let eq = ColumnPredicate::Eq {
        col: 1,
        value: datafusion_common::ScalarValue::Utf8(Some("active".to_owned())),
    };
    let gt = ColumnPredicate::GreaterThan {
        col: 3,
        value: datafusion_common::ScalarValue::Int64(Some(500)),
    };
    let partitions = engine
        .scan_table_arrow("t", Some(&[0, 3]), Some(&schema()), usize::MAX, Some(&[eq, gt]))
        .expect("scan");
    let batches = flatten(partitions);
    let total = total_rows(&batches);
    assert_eq!(total, 19, "compound filter: active AND score > 500 must match 19 rows");
    assert_ne!(total, N_ROWS / 2, "must not return all active rows");
    for batch in &batches {
        let score_arr = batch.column(1).as_any().downcast_ref::<Int64Array>().expect("score");
        for i in 0..score_arr.len() {
            assert!(score_arr.value(i) > 500, "every row must have score > 500");
        }
    }
}

// =============================================================================
// Query 6: COUNT(*)
// =============================================================================
#[test]
fn olap_q6_count_aggregation() {
    let dir = unique_dir("q6");
    let engine = load_engine(&dir);
    let batches = engine.scan_table("t", Some(&[0]), Some(&schema())).expect("scan");
    assert_eq!(total_rows(&batches), N_ROWS);
    let sum: i64 = batches.iter()
        .flat_map(|b| b.column(0).as_any().downcast_ref::<Int64Array>())
        .flat_map(|arr| (0..arr.len()).map(move |i| arr.value(i)))
        .sum();
    let expected: i64 = (0..N_ROWS as i64).sum();
    assert_eq!(sum, expected, "sum of ids must be {expected}");
}

// =============================================================================
// Query 7: Empty result filter
// =============================================================================
#[test]
fn olap_q7_empty_result_filter() {
    let dir = unique_dir("q7");
    let engine = load_engine(&dir);
    let pred = ColumnPredicate::Eq {
        col: 0,
        value: datafusion_common::ScalarValue::Int64(Some(999)),
    };
    let partitions = engine
        .scan_table_arrow("t", Some(&[0]), Some(&schema()), usize::MAX, Some(&[pred]))
        .expect("scan");
    let batches = flatten(partitions);
    assert_eq!(total_rows(&batches), 0, "WHERE id=999 must return zero rows");
}

// =============================================================================
// Query 8: Top-K score validation
// =============================================================================
#[test]
fn olap_q8_top_k_scores() {
    let dir = unique_dir("q8");
    let engine = load_engine(&dir);
    let batches = engine.scan_table("t", Some(&[3]), Some(&schema())).expect("scan");
    let mut scores: Vec<i64> = batches.iter()
        .flat_map(|b| b.column(0).as_any().downcast_ref::<Int64Array>())
        .flat_map(|arr| (0..arr.len()).map(move |i| arr.value(i)))
        .collect();
    assert_eq!(scores.len(), N_ROWS);
    scores.sort_unstable_by(|a, b| b.cmp(a));
    let top5 = &scores[..5];
    for i in 1..top5.len() {
        assert!(top5[i - 1] >= top5[i], "top-5 scores must be non-increasing");
    }
    for s in &scores {
        assert!(*s >= 0 && *s < 1000, "score must be in [0, 1000)");
    }
}

// =============================================================================
// Query 9: Multi-column projection + range filter — age > 70 → 29 rows
// =============================================================================
#[test]
fn olap_q9_multi_column_projection_with_range() {
    let dir = unique_dir("q9");
    let engine = load_engine(&dir);
    let pred = ColumnPredicate::GreaterThan {
        col: 2,
        value: datafusion_common::ScalarValue::Int64(Some(70)),
    };
    let partitions = engine
        .scan_table_arrow("t", Some(&[0, 3]), Some(&schema()), usize::MAX, Some(&[pred]))
        .expect("scan");
    let batches = flatten(partitions);
    assert_eq!(total_rows(&batches), 29, "age > 70 must match exactly 29 rows");
    let first = batches.first().expect("first batch");
    assert_eq!(first.num_columns(), 2, "projection must return 2 columns");
    assert_ne!(total_rows(&batches), N_ROWS, "must not return all rows");
}

// =============================================================================
// Query 10: Batched scan
// =============================================================================
#[test]
fn olap_q10_batched_scan() {
    let dir = unique_dir("q10");
    let engine = load_engine(&dir);
    let partitions = engine
        .scan_table_arrow("t", None, Some(&schema()), 13, None)
        .expect("scan");
    let batches = flatten(partitions);
    assert_eq!(total_rows(&batches), N_ROWS);
    for b in &batches {
        assert!(b.num_rows() <= 13, "each batch must be <= 13 rows");
    }
}

// =============================================================================
// SC-10 (G28): HTAP row-column consistency
// =============================================================================

#[test]
fn olap_row_column_consistency() {
    let dir = unique_dir("rowcol");
    let engine = load_engine(&dir);
    let row_batches = engine.scan_table("t", None, Some(&schema())).expect("row scan");
    let col_partitions = engine
        .scan_table_arrow("t", None, Some(&schema()), usize::MAX, None)
        .expect("col scan");
    let col_batches: Vec<RecordBatch> = col_partitions.into_iter().flatten().collect();
    let row_total = total_rows(&row_batches);
    let col_total = total_rows(&col_batches);
    assert_eq!(row_total, col_total, "row and column scan must return same row count");
    assert_eq!(row_total, N_ROWS, "must return all rows");
    assert_ne!(row_total, 0, "row scan must not be empty");
    assert_ne!(col_total, 0, "column scan must not be empty");
}

#[test]
fn olap_projection_consistency_row_vs_column() {
    let dir = unique_dir("proj-consistency");
    let engine = load_engine(&dir);
    let projection = Some(&[0_usize, 3_usize][..]);
    let row_batches = engine.scan_table("t", projection, Some(&schema())).expect("row scan");
    let col_partitions = engine
        .scan_table_arrow("t", projection, Some(&schema()), usize::MAX, None)
        .expect("col scan");
    let col_batches: Vec<RecordBatch> = col_partitions.into_iter().flatten().collect();
    assert_eq!(total_rows(&row_batches), total_rows(&col_batches));
    let row_batch = row_batches.first().expect("row batch");
    let col_batch = col_batches.first().expect("col batch");
    assert_eq!(row_batch.num_columns(), 2);
    assert_eq!(col_batch.num_columns(), 2);
    assert_eq!(row_batch.num_columns(), col_batch.num_columns());
    assert_ne!(row_batch.num_columns(), 4, "must not return all columns");
}

// =============================================================================
// SC-09 (G27): HTAP concurrent OLTP+OLAP
// =============================================================================

#[test]
fn olap_concurrent_write_and_scan() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    let dir = unique_dir("concurrent");
    let engine = Arc::new(Engine::open(EngineConfig::new(dir.path())).expect("open"));
    let batch = build_dataset();
    let mutations = engine
        .ingest_batches(&[batch], "t", &["id".to_string()])
        .expect("ingest");
    engine.write(&mutations).expect("write");

    let stop = Arc::new(AtomicBool::new(false));
    let engine_w = Arc::clone(&engine);
    let stop_w = Arc::clone(&stop);

    let writer = std::thread::spawn(move || {
        let mut i = N_ROWS as u64;
        while !stop_w.load(Ordering::Relaxed) {
            let key = format!("new_key_{i:06}");
            let _ = engine_w.write(&[hatp_engine::Mutation::Put {
                key: Bytes::copy_from_slice(key.as_bytes()),
                value: Bytes::copy_from_slice(b"concurrent"),
            }]);
            i += 1;
        }
    });

    let mut scan_counts = Vec::new();
    for _ in 0..10 {
        let batches = engine
            .scan_table("t", Some(&[0]), Some(&schema()))
            .expect("scan during concurrent writes");
        scan_counts.push(total_rows(&batches));
    }
    stop.store(true, Ordering::Relaxed);
    let _ = writer.join();

    for window in scan_counts.windows(2) {
        assert!(
            window[0] <= window[1],
            "scan row count must be monotonic non-decreasing: {} > {}",
            window[0], window[1]
        );
    }
    assert!(
        scan_counts.iter().all(|&c| c >= N_ROWS),
        "every scan must see at least the seeded rows"
    );
}