//! Translates DataFusion `Expr` AST into the engine's structural
//! [`ColumnPredicate`] representation.
//!
//! This module lives in the frontend because DataFusion's `Expr` type is the
//! one canonical input form on the OLAP side, and the storage layer
//! ([`hatp_engine`]) deliberately knows nothing about DataFusion. The
//! translation rules mirror `column_predicate::ColumnPredicate::evaluate` so
//! every predicate that comes out of `try_pushdown` is mechanically
//! equivalent to one DataFusion would have evaluated in memory.
//!
//! ## Supported shapes (only)
//!
//! | Form                              | Encoding                                 |
//! |-----------------------------------|------------------------------------------|
//! | `col = literal`                   | [`ColumnPredicate::Eq`]                  |
//! | `col < literal`                   | [`ColumnPredicate::LessThan`]            |
//! | `col <= literal`                  | [`ColumnPredicate::LessThanOrEqual`]     |
//! | `col > literal`                   | [`ColumnPredicate::GreaterThan`]         |
//! | `col >= literal`                  | [`ColumnPredicate::GreaterThanOrEqual`]  |
//!
//! Disjunctions, multi-column AND, `IN`, `LIKE`, `BETWEEN` are not supported
// here — the caller treats them as residual filters that DataFusion evaluates
//! in memory.

use arrow_schema::{Schema, SchemaRef};
use datafusion_common::ScalarValue;
use datafusion_expr::{BinaryExpr, Expr, Operator};
use hatp_engine::column_predicate::ColumnPredicate;

/// Decomposes a DataFusion `Expr` into a list of [`ColumnPredicate`]s. A
/// predicate that is not a single-column comparison with a literal returns
/// nothing (it is a residual that the caller leaves for in-memory filtering).
///
/// The returned `Vec<ColumnPredicate>` is in source order; the caller is free
/// to sort or merge them. (See [`ColumnPredicate::Range`] for the merge
/// pattern.)
pub fn expr_to_column_predicates(expr: &Expr, schema: &Schema) -> Vec<ColumnPredicate> {
    let mut out = Vec::new();
    collect(expr, schema, &mut out);
    out
}

/// Same as [`expr_to_column_predicates`] but takes a [`SchemaRef`] (matches the
/// `scan_with_filter` signature on the engine side).
#[inline]
pub fn expr_to_column_predicates_with_ref(
    expr: &Expr,
    schema: &SchemaRef,
) -> Vec<ColumnPredicate> {
    expr_to_column_predicates(expr, schema.as_ref())
}

fn collect(expr: &Expr, schema: &Schema, out: &mut Vec<ColumnPredicate>) {
    match expr {
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            // Only `AND` chains decompose; `OR` is a residual.
            if *op == Operator::And {
                collect(left, schema, out);
                collect(right, schema, out);
                return;
            }
            if let Some(pred) = try_single_column_compare(left, op, right, schema) {
                out.push(pred);
            }
        }
        Expr::Not(inner) => {
            // Negations are residuals for now.  (Expr::Negative is arithmetic
            // negation `-col` and never appears as a top-level WHERE filter;
            // it is handled by the catch-all `_` arm below.)
            let _ = inner;
        }
        _ => {
            // Anything that is not a single-column comparison is a residual.
        }
    }
}

fn try_single_column_compare(
    left: &Expr,
    op: &Operator,
    right: &Expr,
    schema: &Schema,
) -> Option<ColumnPredicate> {
    let (col_idx, value) = match (left, right) {
        (Expr::Column(c), Expr::Literal(value, _)) => {
            (schema.index_of(&c.name).ok()?, value.clone())
        }
        (Expr::Literal(value, _), Expr::Column(c)) => {
            (schema.index_of(&c.name).ok()?, value.clone())
        }
        _ => return None,
    };
    match op {
        Operator::Eq => Some(ColumnPredicate::Eq { col: col_idx, value }),
        Operator::Lt => Some(ColumnPredicate::LessThan { col: col_idx, value }),
        Operator::LtEq => Some(ColumnPredicate::LessThanOrEqual { col: col_idx, value }),
        Operator::Gt => Some(ColumnPredicate::GreaterThan { col: col_idx, value }),
        Operator::GtEq => Some(ColumnPredicate::GreaterThanOrEqual { col: col_idx, value }),
        _ => None,
    }
}

/// Resolve the schema field name for a [`ColumnPredicate`] by index. Returns
/// `None` when the predicate references an unknown column (defensive: should
/// not happen for predicates produced by [`expr_to_column_predicates`]).
#[inline]
pub fn column_predicate_field<'a>(
    pred: &ColumnPredicate,
    schema: &'a Schema,
) -> Option<&'a str> {
    schema.fields().get(pred.column()).map(|f| f.name().as_str())
}

/// Extract the literal value from a single-column comparison predicate.
/// Returns `None` for `Range` variants (which carry two bounds, not a single
/// scalar).
pub fn column_predicate_scalar(pred: &ColumnPredicate) -> Option<&ScalarValue> {
    match pred {
        ColumnPredicate::Eq { value, .. }
        | ColumnPredicate::LessThan { value, .. }
        | ColumnPredicate::LessThanOrEqual { value, .. }
        | ColumnPredicate::GreaterThan { value, .. }
        | ColumnPredicate::GreaterThanOrEqual { value, .. } => Some(value),
        ColumnPredicate::Range { .. } => None,
    }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field};
    use datafusion_expr::{col, lit};

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("age", DataType::Int32, true),
        ])
    }

    #[test]
    fn eq_predicate_translates() {
        let s = schema();
        let expr = col("age").eq(lit(30i32));
        let preds = expr_to_column_predicates(&expr, &s);
        assert_eq!(preds.len(), 1);
        assert!(matches!(preds[0], ColumnPredicate::Eq { .. }));
    }

    #[test]
    fn range_predicate_translates() {
        let s = schema();
        let lower = col("age").gt(lit(20i32));
        let upper = col("age").lt_eq(lit(40i32));
        let mut preds = expr_to_column_predicates(&lower, &s);
        preds.extend(expr_to_column_predicates(&upper, &s));
        // Two independent predicates on the same column — the caller merges.
        assert_eq!(preds.len(), 2);
    }

    #[test]
    fn and_chain_decomposes() {
        let s = schema();
        let expr = col("id").eq(lit(1i64)).and(col("age").gt(lit(20i32)));
        let preds = expr_to_column_predicates(&expr, &s);
        assert_eq!(preds.len(), 2);
    }

    #[test]
    fn unsupported_is_residual() {
        let s = schema();
        // `id + 1 > 0` cannot be pushed down to a single-column scan.
        let expr = (col("id") + lit(1i64)).gt(lit(0i64));
        let preds = expr_to_column_predicates(&expr, &s);
        assert_eq!(preds.len(), 0);
    }

    #[test]
    fn or_chain_is_residual() {
        let s = schema();
        // Disjunctions (`a OR b`) cannot decompose into independent
        // single-column predicates — keep them as residual filters.
        let expr = col("id").eq(lit(1i64)).or(col("id").eq(lit(2i64)));
        let preds = expr_to_column_predicates(&expr, &s);
        assert_eq!(preds.len(), 0, "OR must not decompose");
    }

    #[test]
    fn negation_is_residual() {
        use std::ops::Not;
        let s = schema();
        let expr = col("id").eq(lit(1i64)).not();
        let preds = expr_to_column_predicates(&expr, &s);
        assert_eq!(preds.len(), 0, "NOT must not decompose");
    }

    #[test]
    fn literal_only_expr_is_residual() {
        let s = schema();
        let expr = lit(1_i64).eq(lit(1_i64));
        let preds = expr_to_column_predicates(&expr, &s);
        assert_eq!(preds.len(), 0);
    }

    #[test]
    fn column_with_unknown_name_is_residual() {
        let s = schema();
        let expr = col("missing").eq(lit(1_i64));
        let preds = expr_to_column_predicates(&expr, &s);
        assert_eq!(preds.len(), 0);
    }

    #[test]
    fn field_lookup_resolves_index() {
        let s = schema();
        let pred = ColumnPredicate::Eq {
            col: 1,
            value: ScalarValue::Int32(Some(7)),
        };
        assert_eq!(column_predicate_field(&pred, &s), Some("age"));
        let bad = ColumnPredicate::Eq {
            col: 9,
            value: ScalarValue::Int32(Some(7)),
        };
        assert_eq!(column_predicate_field(&bad, &s), None);
    }

    #[test]
    fn column_predicate_scalar_value_extracts() {
        let s = schema();
        let expr = col("age").eq(lit(7i32));
        let preds = expr_to_column_predicates(&expr, &s);
        assert_eq!(
            column_predicate_scalar(&preds[0]),
            Some(&ScalarValue::Int32(Some(7)))
        );
        // Range is built by the caller, not the translator — out of scope here.
        let range = ColumnPredicate::Range {
            col: 1,
            lower: None,
            upper: None,
        };
        assert!(column_predicate_scalar(&range).is_none());
    }
}