//! ColumnPredicate integration tests — referencing RocksDB point query + range scan tests
//!
//! Verify ColumnPredicate Eq / Range / bound behavior matches engine scan paths.
//!
//! Note: `scan_table_with_filters` does not exist on Engine's public API.
//! Use `scan_table_arrow` with the predicates parameter.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::assertions_on_constants
)]

use std::sync::Arc;
use tempfile::Builder;

use arrow_array::{Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use hatp_engine::column_predicate::{Bound, ColumnPredicate};
use hatp_engine::{Engine, EngineConfig};

fn unique_dir(label: &str) -> tempfile::TempDir {
    Builder::new()
        .prefix(&format!("hatp-pred-{label}-"))
        .tempdir()
        .expect("tempdir")
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("age", DataType::Int32, false),
    ]))
}

fn load_rows(engine: &Engine, rows: Vec<(i64, i32)>) {
    let batch = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(rows.iter().map(|r| r.0).collect::<Vec<_>>())),
            Arc::new(Int32Array::from(rows.iter().map(|r| r.1).collect::<Vec<_>>())),
        ],
    )
    .expect("batch build");
    let mutations = engine
        .ingest_batches(&[batch], "t", &["id".to_string()])
        .expect("ingest");
    engine.write(&mutations).expect("write");
    engine.flush().expect("flush");
}

fn flatten(partitions: Vec<Vec<RecordBatch>>) -> Vec<RecordBatch> {
    partitions.into_iter().flatten().collect()
}

fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.num_rows()).sum()
}

#[test]
fn engine_scan_with_eq_predicate_filters_rows() {
    let dir = unique_dir("eq");
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");
    load_rows(
        &engine,
        vec![(1, 25), (2, 30), (3, 35), (4, 40), (5, 45)],
    );

    // Unfiltered baseline
    let all = engine.scan_table("t", None, Some(&schema())).expect("scan");
    assert_eq!(total_rows(&all), 5);

    // Filtered: only rows where `id == 3`
    let pred = ColumnPredicate::Eq {
        col: 0,
        value: datafusion_common::ScalarValue::Int64(Some(3)),
    };
    let hits = flatten(
        engine
            .scan_table_arrow("t", Some(&[0]), Some(&schema()), usize::MAX, Some(&[pred]))
            .expect("scan with filters"),
    );
    let matched = total_rows(&hits);
    assert_eq!(matched, 1, "only the id=3 row should remain");

    // Negative assertion: must not return more than 1 row
    assert_ne!(matched, 5, "must not return all rows");
}

#[test]
fn engine_scan_with_range_predicates_filters_rows() {
    let dir = unique_dir("range");
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");
    load_rows(
        &engine,
        vec![(1, 20), (2, 30), (3, 40), (4, 50), (5, 60)],
    );

    let lower = ColumnPredicate::GreaterThanOrEqual {
        col: 1,
        value: datafusion_common::ScalarValue::Int32(Some(30)),
    };
    let upper = ColumnPredicate::LessThanOrEqual {
        col: 1,
        value: datafusion_common::ScalarValue::Int32(Some(50)),
    };
    let hits = flatten(
        engine
            .scan_table_arrow("t", Some(&[1]), Some(&schema()), usize::MAX, Some(&[lower, upper]))
            .expect("scan with range filters"),
    );
    let matched = total_rows(&hits);
    // age in {30, 40, 50}
    assert_eq!(matched, 3, "age in [30, 50] must match 3 rows");

    // Negative assertion
    assert_ne!(matched, 5, "must not return all rows");
}

#[test]
fn bound_range_predicate_matches_inclusive_and_exclusive() {
    let pred = ColumnPredicate::Range {
        col: 0,
        lower: Some(Bound::inclusive(datafusion_common::ScalarValue::Int64(Some(10)))),
        upper: Some(Bound::exclusive(datafusion_common::ScalarValue::Int64(Some(20)))),
    };
    assert!(pred.evaluate(&datafusion_common::ScalarValue::Int64(Some(10))));
    assert!(pred.evaluate(&datafusion_common::ScalarValue::Int64(Some(15))));
    assert!(!pred.evaluate(&datafusion_common::ScalarValue::Int64(Some(20))));
    assert!(!pred.evaluate(&datafusion_common::ScalarValue::Int64(Some(9))));
}

#[test]
fn end_to_end_datafusion_where_filters_correct_number_of_rows() {
    let dir = unique_dir("e2e");
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");
    load_rows(
        &engine,
        (1..=100)
            .map(|i| (i, ((i * 7) % 60) as i32))
            .collect(),
    );

    let pred = ColumnPredicate::GreaterThan {
        col: 0,
        value: datafusion_common::ScalarValue::Int64(Some(50)),
    };
    let hits = flatten(
        engine
            .scan_table_arrow("t", Some(&[0]), Some(&schema()), usize::MAX, Some(&[pred]))
            .expect("scan with filters"),
    );
    let matched: i64 = hits
        .iter()
        .flat_map(|b| b.column(0).as_any().downcast_ref::<Int64Array>())
        .flat_map(|arr| (0..arr.len()).map(move |i| arr.value(i)))
        .sum();
    // ids in (51..=100) sum to 51+52+...+100 = (51+100)*50/2 = 3775
    assert_eq!(matched, 3775, "sum of ids > 50 must be 3775");

    // Negative assertion: must not be the sum of all ids
    let all_sum: i64 = (1..=100).sum();
    assert_ne!(matched, all_sum, "must not return all rows");
}