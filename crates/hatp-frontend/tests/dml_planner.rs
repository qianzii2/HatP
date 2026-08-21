//! Integration tests for `hatp_frontend::dml_planner`.
//!
//! These tests verify that the frontend's DML planning layer correctly
//! translates DataFusion `Expr`s into engine `Mutation`s. The engine-only
//! OLTP tests (get / write / scan_range) remain in `hatp_engine::tests`.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::indexing_slicing)]

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use datafusion_expr::{col, lit};
use hatp_engine::{Engine, EngineConfig};
use tempfile::Builder;

fn unique_dir(label: &str) -> Result<tempfile::TempDir, std::io::Error> {
    Builder::new()
        .prefix(&format!("hatp-dml-{label}-"))
        .tempdir()
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("a", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]))
}

/// Seed `(id, a, name)` rows and return the engine + primary key.
fn seed_rows(
    dir: &tempfile::TempDir,
    rows: &[(i64, i64, &str)],
) -> Result<Arc<Engine>, Box<dyn std::error::Error>> {
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    let ids: Vec<i64> = rows.iter().map(|r| r.0).collect();
    let avals: Vec<i64> = rows.iter().map(|r| r.1).collect();
    let names: Vec<&str> = rows.iter().map(|r| r.2).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema()),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Int64Array::from(avals)),
            Arc::new(arrow_array::StringArray::from(names)),
        ],
    )?;
    let mutations = engine.ingest_batches(&[batch], "t", &["id".to_string()])?;
    engine.write(&mutations)?;
    Ok(engine)
}

/// Return the sorted `id` values currently visible in the table.
fn visible_ids(engine: &Engine) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    let batches = engine.scan_table("t", None, None)?;
    let mut out = Vec::new();
    for batch in &batches {
        let (idx, _) = batch.schema().column_with_name("id").expect("id column present");
        let arr = batch.column(idx);
        for row in 0..batch.num_rows() {
            let scalar = hatp_types::codec::array_slot_to_scalar(arr.as_ref(), row);
            if let datafusion_common::ScalarValue::Int64(Some(v)) = scalar {
                out.push(v);
            }
        }
    }
    out.sort_unstable();
    Ok(out)
}

/// Return the `a` value for `id`, or `None` if the id is gone.
fn a_for(engine: &Engine, id: i64) -> Result<Option<i64>, Box<dyn std::error::Error>> {
    let batches = engine.scan_table("t", None, None)?;
    for batch in &batches {
        let (id_idx, _) = batch.schema().column_with_name("id").expect("id col");
        let (a_idx, _) = batch.schema().column_with_name("a").expect("a col");
        for row in 0..batch.num_rows() {
            let id_scalar =
                hatp_types::codec::array_slot_to_scalar(batch.column(id_idx).as_ref(), row);
            if let datafusion_common::ScalarValue::Int64(Some(v)) = id_scalar {
                if v == id {
                    let a =
                        hatp_types::codec::array_slot_to_scalar(batch.column(a_idx).as_ref(), row);
                    if let datafusion_common::ScalarValue::Int64(Some(a)) = a {
                        return Ok(Some(a));
                    }
                }
            }
        }
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// UPDATE tests
// ---------------------------------------------------------------------------

/// `UPDATE ... WHERE id = 1 OR id = 3` must touch only rows 1 and 3 (the old
/// code dropped the OR arm and updated every row).
#[test]
fn update_or_filter_only_touches_matching_rows() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("update-or")?;
    let engine = seed_rows(&dir, &[(1, 10, "alice"), (2, 20, "bob"), (3, 30, "carol")])?;

    let or_filter = col("id").eq(lit(1_i64)).or(col("id").eq(lit(3_i64)));
    let mutations = hatp_frontend::dml_planner::plan_update(
        &engine,
        "t",
        &[("a".to_string(), lit(99_i64))],
        &[or_filter],
        &["id".to_string()],
    )?;
    engine.write(&mutations)?;

    assert_eq!(a_for(&engine, 1)?, Some(99), "id=1 must be updated");
    assert_eq!(a_for(&engine, 2)?, Some(20), "id=2 must NOT be updated");
    assert_eq!(a_for(&engine, 3)?, Some(99), "id=3 must be updated");
    Ok(())
}

/// `UPDATE ... WHERE a > 15` must evaluate the comparison predicate (the old
/// code skipped comparisons entirely, updating every row).
#[test]
fn update_comparison_filter() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("update-cmp")?;
    let engine = seed_rows(&dir, &[(1, 10, "alice"), (2, 20, "bob"), (3, 30, "carol")])?;

    let mutations = hatp_frontend::dml_planner::plan_update(
        &engine,
        "t",
        &[("name".to_string(), lit("hot"))],
        &[col("a").gt(lit(15_i64))],
        &["id".to_string()],
    )?;
    engine.write(&mutations)?;

    assert_eq!(a_for(&engine, 1)?, Some(10));
    assert_eq!(a_for(&engine, 2)?, Some(20));
    assert_eq!(a_for(&engine, 3)?, Some(30));
    let batches = engine.scan_table("t", None, None)?;
    let mut names: Vec<(i64, String)> = Vec::new();
    for batch in &batches {
        let (id_idx, _) = batch.schema().column_with_name("id").expect("id");
        let (n_idx, _) = batch.schema().column_with_name("name").expect("name");
        for row in 0..batch.num_rows() {
            let id =
                hatp_types::codec::array_slot_to_scalar(batch.column(id_idx).as_ref(), row);
            let name =
                hatp_types::codec::array_slot_to_scalar(batch.column(n_idx).as_ref(), row);
            if let (datafusion_common::ScalarValue::Int64(Some(id)),
                    datafusion_common::ScalarValue::Utf8(Some(n))) = (id, name)
            {
                names.push((id, n));
            }
        }
    }
    names.sort_by_key(|(id, _)| *id);
    assert_eq!(
        names,
        vec![
            (1, "alice".to_string()),
            (2, "hot".to_string()),
            (3, "hot".to_string())
        ]
    );
    Ok(())
}

/// `SET col = CAST(col AS type)` must evaluate the CAST via DataFusion's
/// expression evaluator.
#[test]
fn update_cast_expr_evaluates() -> Result<(), Box<dyn std::error::Error>> {
    use arrow_schema::DataType;

    let dir = unique_dir("update-cast")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1_i64])),
            Arc::new(arrow_array::StringArray::from(vec!["alice"])),
        ],
    )?;
    let mutations = engine.ingest_batches(&[batch], "t", &["id".to_string()])?;
    engine.write(&mutations)?;

    let cast = datafusion_expr::Expr::Cast(datafusion_expr::Cast::new(
        Box::new(col("id")),
        DataType::Utf8,
    ));
    let mutations = hatp_frontend::dml_planner::plan_update(
        &engine,
        "t",
        &[("name".to_string(), cast)],
        &[col("id").eq(lit(1_i64))],
        &["id".to_string()],
    )?;
    engine.write(&mutations)?;

    let batches = engine.scan_table("t", None, None)?;
    let batch = batches.first().expect("scan batch");
    let (idx, _) = batch.schema().column_with_name("name").expect("name col");
    let name = hatp_types::codec::array_slot_to_scalar(batch.column(idx).as_ref(), 0);
    assert_eq!(
        name,
        datafusion_common::ScalarValue::Utf8(Some("1".to_string()))
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// DELETE tests
// ---------------------------------------------------------------------------

/// `DELETE WHERE id = 1 OR id = 2` must delete both rows (the old code kept
/// only `id = 1` in the equality map and deleted just that row).
#[test]
fn delete_or_filter_deletes_both_rows() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("delete-or")?;
    let engine = seed_rows(&dir, &[(1, 10, "alice"), (2, 20, "bob"), (3, 30, "carol")])?;

    let or_filter = col("id").eq(lit(1_i64)).or(col("id").eq(lit(2_i64)));
    let mutations = hatp_frontend::dml_planner::plan_delete(
        &engine,
        "t",
        &["id".to_string()],
        &[or_filter],
    )?;
    engine.write(&mutations)?;

    assert_eq!(visible_ids(&engine)?, vec![3]);
    Ok(())
}

/// A non-PK equality filter (`WHERE name = 'bob'`) must fall through to the
/// generic scan+filter path and delete exactly the matching row.
#[test]
fn delete_non_pk_equality_filter() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("delete-nonpk")?;
    let engine = seed_rows(&dir, &[(1, 10, "alice"), (2, 20, "bob"), (3, 30, "carol")])?;

    let mutations = hatp_frontend::dml_planner::plan_delete(
        &engine,
        "t",
        &["id".to_string()],
        &[col("name").eq(lit("bob"))],
    )?;
    engine.write(&mutations)?;

    assert_eq!(visible_ids(&engine)?, vec![1, 3]);
    Ok(())
}

/// `DELETE WHERE name LIKE 'a%'` must delete matching rows via the generic
/// path (the old code silently ignored LIKE and deleted nothing / errored).
#[test]
fn delete_like_filter() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("delete-like")?;
    let engine = seed_rows(&dir, &[(1, 10, "alice"), (2, 20, "bob"), (3, 30, "amy")])?;

    let like = datafusion_expr::Expr::Like(datafusion_expr::Like::new(
        false,
        Box::new(col("name")),
        Box::new(lit("a%")),
        None,
        false,
    ));
    let mutations = hatp_frontend::dml_planner::plan_delete(
        &engine,
        "t",
        &["id".to_string()],
        &[like],
    )?;
    engine.write(&mutations)?;

    assert_eq!(visible_ids(&engine)?, vec![2]);
    Ok(())
}
