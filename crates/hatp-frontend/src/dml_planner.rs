//! DML planning for the OLAP → OLTP boundary.
//!
//! All DataFusion-aware DML work lives here. The engine exposes only
//! **neutral** APIs that take already-translated values (column indexes, raw
//! scalars, `RecordBatch` slices) — see
//! [`hatp_engine::Engine::ingest_batches`] and
//! [`hatp_engine::Engine::delete_with_column_predicates`]. This module is the
//! single place that bridges DataFusion's [`Expr`] AST into those neutral types.
//!
//! ## Why the engine no longer accepts `Expr`
//!
//! `hatp_engine` is the OLTP core. Keeping DataFusion out of its dependency
//! graph lets the engine stay free of the planner, optimizer, and physical
//! expression crates that have nothing to do with point queries or
//! group-commit. The cost is that any DML feature must be re-implemented
//! here in terms of the engine's neutral types; the benefit is a clean
//! layering where the engine can be reused without a DataFusion runtime.

use std::any::Any;
use std::sync::Arc;

use arrow_array::{RecordBatch, BooleanArray};
use arrow_schema::{Schema as ArrowSchema, SchemaRef as ArrowSchemaRef, DataType};
use datafusion_common::{Result as DfResult, ScalarValue, Column};
use datafusion_datasource::memory::MemorySourceConfig;
use datafusion_expr::{Expr, Operator};
use datafusion_physical_expr::create_physical_expr;
use datafusion_physical_plan::ExecutionPlan;
use hatp_engine::column_predicate::ColumnPredicate;
use hatp_engine::{Engine, Mutation};

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Translate an `INSERT INTO ... SELECT ...` plan into a flat list of
/// [`Mutation::Put`] entries.
///
/// `plan` must be a [`MemorySourceConfig`] (the shape `TableProviderAdapter`
/// builds when a `SELECT` is the input of an `INSERT`). The plan's
/// partitions are concatenated into a single batch list, then handed to the
/// engine's `ingest_batches` path so cross-partition duplicate primary keys
/// are collapsed exactly once.
pub fn plan_insert(
    engine: &Engine,
    plan: &dyn ExecutionPlan,
    table: &str,
    primary_key: &[String],
) -> DfResult<Vec<Mutation>> {
    let any = plan as &dyn Any;
    let Some(source) = any.downcast_ref::<MemorySourceConfig>() else {
        return Err(datafusion_common::DataFusionError::Plan(
            "plan_insert currently only supports MemorySourceConfig (i.e. direct \
             table scans via TableProviderAdapter). Joins / aggregates must use \
             INSERT ... SELECT via execute_sql + the TableProviderAdapter::insert_into \
             path."
                .to_owned(),
        ));
    };
    let mut batches: Vec<RecordBatch> = Vec::new();
    for partition in source.partitions() {
        batches.extend_from_slice(partition);
    }
    engine
        .ingest_batches(&batches, table, primary_key)
        .map_err(|err| datafusion_common::DataFusionError::External(Box::new(err)))
}

/// Translate a `DELETE FROM t WHERE filters` into a flat list of
/// [`Mutation::Delete`] entries.
///
/// Internally:
/// 1. Fast path: if `filters` reduces to exactly one equality per PK column,
///    use the engine's `delete_with_column_predicates` point-lookup path.
/// 2. Generic path: full table scan + vectorized filter evaluation via
///    DataFusion physical expressions. This covers OR / LIKE / comparison /
///    non-PK equality — the full SQL filter semantics.
pub fn plan_delete(
    engine: &Engine,
    table: &str,
    primary_key: &[String],
    filters: &[Expr],
) -> DfResult<Vec<Mutation>> {
    if filters.is_empty() {
        return Err(datafusion_common::DataFusionError::Plan(
            "DELETE requires a WHERE clause; full-table DELETE is rejected".to_owned(),
        ));
    }
    if primary_key.is_empty() {
        return Err(datafusion_common::DataFusionError::Plan(
            "DELETE requires a table with a primary key".to_owned(),
        ));
    }

    // Try to route through the point-lookup fast path: one `col = lit` per PK column.
    // We need column indexes, so we must have the schema.
    if let Some(schema) = engine.table_schema(table) {
        if let Some(predicates) = try_translate_pk_predicates(filters, &schema) {
            // Engine's neutral DELETE path handles the point lookup internally.
            return match engine.delete_with_column_predicates(table, primary_key, &predicates) {
                Ok(muts) => Ok(muts),
                Err(hatp_engine::EngineError::EmptyBatch) => Ok(Vec::new()),
                Err(err) => Err(datafusion_common::DataFusionError::External(Box::new(err))),
            };
        }
    }

    // Generic scan+filter path: full SQL filter semantics.
    let batches = engine
        .scan_table(table, None, None)
        .map_err(|err| datafusion_common::DataFusionError::External(Box::new(err)))?;
    let mut mutations: Vec<Mutation> = Vec::new();
    for batch in &batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let mask = eval_filters_mask(filters, batch.schema().as_ref(), batch)?;
        let matched = arrow_select::filter::filter_record_batch(batch, &mask)
            .map_err(|err| datafusion_common::DataFusionError::External(Box::new(err)))?;
        let pk_columns = resolve_pk_columns(&matched, primary_key)?;
        for row in 0..matched.num_rows() {
            let key = build_pk_key(table, &matched, row, &pk_columns)?;
            mutations.push(Mutation::Delete { key });
        }
    }
    Ok(mutations)
}

/// Translate an `UPDATE t SET col = expr WHERE filters` into a flat list of
/// [`Mutation::Put`] entries.
///
/// The full scan + vectorized filter + assignment-rewrite path. Assignments and
/// filters are evaluated with DataFusion's physical expression engine so that
/// SQL semantics (CAST, LIKE, NULL handling, arithmetic) are preserved.
pub fn plan_update(
    engine: &Engine,
    table: &str,
    assignments: &[(String, Expr)],
    filters: &[Expr],
    primary_key: &[String],
) -> DfResult<Vec<Mutation>> {
    if primary_key.is_empty() {
        return Err(datafusion_common::DataFusionError::Plan(
            "UPDATE requires a table with a primary key".to_owned(),
        ));
    }

    let batches = engine
        .scan_table(table, None, None)
        .map_err(|err| datafusion_common::DataFusionError::External(Box::new(err)))?;
    if batches.is_empty() {
        return Ok(Vec::new());
    }

    let mut mutations: Vec<Mutation> = Vec::new();
    for batch in &batches {
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            continue;
        }
        let schema = batch.schema();

        // WHERE mask (empty filters = match every row).
        let mask = eval_filters_mask(filters, schema.as_ref(), batch)?;

        // Build the updated batch: assigned columns are re-evaluated
        // vectorized, unassigned columns are kept as-is.
        let mut updated_columns: Vec<Arc<dyn arrow_array::Array>> =
            Vec::with_capacity(schema.fields().len());
        for (col_idx, field) in schema.fields().iter().enumerate() {
            let col_name = field.name();
            if let Some((_, expr)) = assignments.iter().find(|(n, _)| n == col_name) {
                let array =
                    eval_expr_to_array(expr, schema.as_ref(), batch, field.data_type())?;
                updated_columns.push(array);
            } else {
                updated_columns.push(Arc::clone(batch.column(col_idx)));
            }
        }
        let updated_batch = RecordBatch::try_new(Arc::clone(&schema), updated_columns)
            .map_err(|err| datafusion_common::DataFusionError::External(Box::new(err)))?;

        // Keep only rows matching the WHERE mask.
        let matched = arrow_select::filter::filter_record_batch(&updated_batch, &mask)
            .map_err(|err| datafusion_common::DataFusionError::External(Box::new(err)))?;

        // Encode rows and reconstruct PK keys.
        let pk_columns = resolve_pk_columns(&matched, primary_key)?;
        for row in 0..matched.num_rows() {
            let payload = ::bytes::Bytes::from(
                hatp_engine::row_codec::encode_row(&matched, row)
                    .map_err(|e| datafusion_common::DataFusionError::External(Box::new(e)))?,
            );
            let key = build_pk_key(table, &matched, row, &pk_columns)?;
            mutations.push(Mutation::Put { key, value: payload });
        }
    }
    Ok(mutations)
}

// ---------------------------------------------------------------------------
// DataFusion physical expression helpers (migrated from hatp-engine)
// ---------------------------------------------------------------------------

/// Strip the table qualifier from every `Expr::Column` in `expr`.
///
/// `scan_table` produces batches whose Arrow schema has unqualified column
/// names, while DataFusion's DML path may hand columns qualified with the table
/// name (`t.id`). `create_physical_expr` matches columns by qualifier + name,
/// so a qualified column would fail to resolve against an unqualified
/// `DFSchema`. Stripping the qualifier makes both forms resolve.
fn unqualify_columns(expr: &Expr) -> DfResult<Expr> {
    use datafusion_common::tree_node::{Transformed, TreeNode};
    let transformed = expr
        .clone()
        .transform(|node| {
            if let Expr::Column(col) = &node {
                Ok(Transformed::yes(Expr::Column(Column::from_name(
                    col.name.clone(),
                ))))
            } else {
                Ok(Transformed::no(node))
            }
        })
        .map_err(|err| datafusion_common::DataFusionError::Plan(format!("unqualify columns: {err}")))?;
    Ok(transformed.data)
}

/// Evaluate a logical [`Expr`] against one `batch` and return a single column
/// array cast to `target_type`.
fn eval_expr_to_array(
    expr: &Expr,
    schema: &ArrowSchema,
    batch: &RecordBatch,
    target_type: &DataType,
) -> DfResult<Arc<dyn arrow_array::Array>> {
    use datafusion_expr::execution_props::ExecutionProps;
    let expr = unqualify_columns(expr)?;
    let df_schema =
        datafusion_common::DFSchema::try_from(Arc::new(schema.clone()))
            .map_err(|err| {
                datafusion_common::DataFusionError::Plan(format!("build DFSchema: {err}"))
            })?;
    let physical =
        create_physical_expr(&expr, &df_schema, &ExecutionProps::new())
            .map_err(|err| {
                datafusion_common::DataFusionError::Plan(format!("physicalize expression: {err}"))
            })?;
    let value = physical.evaluate(batch).map_err(|err| {
        datafusion_common::DataFusionError::Plan(format!("evaluate expression: {err}"))
    })?;
    let casted = value.cast_to(target_type, None).map_err(|err| {
        datafusion_common::DataFusionError::Plan(format!("cast expression: {err}"))
    })?;
    casted.into_array(batch.num_rows()).map_err(|err| {
        datafusion_common::DataFusionError::Plan(format!("expression to array: {err}"))
    })
}

/// Fold `filters` into a single AND-conjoined expression, or `None` when the
/// slice is empty (meaning "no predicate — match every row").
fn conjunction_expr(filters: &[Expr]) -> Option<Expr> {
    filters.iter().cloned().reduce(|left, right| Expr::BinaryExpr(
        datafusion_expr::BinaryExpr {
            left: Box::new(left),
            op: Operator::And,
            right: Box::new(right),
        },
    ))
}

/// Evaluate `filters` (AND semantics) against `batch` into a boolean mask.
fn eval_filters_mask(
    filters: &[Expr],
    schema: &ArrowSchema,
    batch: &RecordBatch,
) -> DfResult<BooleanArray> {
    let Some(expr) = conjunction_expr(filters) else {
        return Ok(BooleanArray::from(vec![true; batch.num_rows()]));
    };
    let array = eval_expr_to_array(&expr, schema, batch, &DataType::Boolean)?;
    let mask = array
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| {
            datafusion_common::DataFusionError::Plan(
                "WHERE expression must evaluate to boolean".to_owned(),
            )
        })?;
    Ok(mask.clone())
}

/// Try to translate filters to `ColumnPredicate`s that can be handled by
/// the engine's `delete_with_column_predicates` fast path (point lookup).
/// Returns `None` if the filters contain anything that can't be expressed
/// as a list of AND-conjoined single-column comparisons.
fn try_translate_pk_predicates(
    filters: &[Expr],
    schema: &ArrowSchemaRef,
) -> Option<Vec<ColumnPredicate>> {
    let mut out = Vec::new();
    for filter in filters {
        if !collect_predicates_for_pk(filter, schema, &mut out) {
            return None;
        }
    }
    if out.is_empty() {
        return None;
    }
    Some(out)
}

/// Recursively collect `ColumnPredicate`s from an AND tree.
fn collect_predicates_for_pk(
    expr: &Expr,
    schema: &ArrowSchemaRef,
    out: &mut Vec<ColumnPredicate>,
) -> bool {
    match expr {
        Expr::BinaryExpr(b) if b.op == Operator::And => {
            collect_predicates_for_pk(&b.left, schema, out)
                && collect_predicates_for_pk(&b.right, schema, out)
        }
        Expr::BinaryExpr(b) if b.op == Operator::Eq => {
            let (col_name, lit) = match (&*b.left, &*b.right) {
                (Expr::Column(c), Expr::Literal(v, _)) => (c.name.as_str(), v),
                (Expr::Literal(v, _), Expr::Column(c)) => (c.name.as_str(), v),
                _ => return false,
            };
            let Ok(col_idx) = schema.index_of(col_name) else { return false; };
            out.push(ColumnPredicate::Eq {
                col: col_idx,
                value: lit.clone(),
            });
            true
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Key-building helpers (mirrored from hatp-engine)
// ---------------------------------------------------------------------------

fn encode_pk_value(scalar: &ScalarValue) -> Option<Vec<u8>> {
    hatp_types::codec::encode_pk_value(scalar)
}

fn assemble_table_key(table: &str, encoded_parts: &[Vec<u8>]) -> bytes::Bytes {
    hatp_engine::assemble_table_key(table, encoded_parts)
}

fn resolve_pk_columns(
    batch: &RecordBatch,
    primary_key: &[String],
) -> DfResult<Vec<(usize, String)>> {
    let mut out = Vec::with_capacity(primary_key.len());
    for col_name in primary_key {
        let (idx, _) = batch.schema().column_with_name(col_name).ok_or_else(|| {
            datafusion_common::DataFusionError::Plan(format!(
                "primary key column `{col_name}` not found in batch schema"
            ))
        })?;
        out.push((idx, col_name.clone()));
    }
    Ok(out)
}

fn build_pk_key(
    table: &str,
    batch: &RecordBatch,
    row: usize,
    pk_columns: &[(usize, String)],
) -> DfResult<bytes::Bytes> {
    use hatp_types::codec::array_slot_to_scalar;
    let mut encoded_parts = Vec::with_capacity(pk_columns.len());
    for &(idx, ref col_name) in pk_columns {
        let scalar = array_slot_to_scalar(batch.column(idx).as_ref(), row);
        let bytes = encode_pk_value(&scalar).ok_or_else(|| {
            datafusion_common::DataFusionError::Plan(format!(
                "null primary key column `{col_name}`"
            ))
        })?;
        encoded_parts.push(bytes);
    }
    Ok(assemble_table_key(table, &encoded_parts))
}
