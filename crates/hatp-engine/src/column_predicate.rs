//! P3.2 — single-column predicate pushdown (Eq / Range).
//!
//! The storage layer receives pre-translated [`ColumnPredicate`] enums from the
//! frontend's [`crate::predicate_translator`] module. The SST layer evaluates them
//! against decoded row values (legacy `value` blob) or passes them to Vortex as
//! [`vortex_array::expr::Expression`]s (columnar format).
//!
//! ## Supported shapes (only)
//!
//! | Form                              | Encoding                              |
//! |-----------------------------------|---------------------------------------|
//! | `col = literal`                   | [`ColumnPredicate::Eq`]               |
//! | `col < literal`                   | [`ColumnPredicate::LessThan`]         |
//! | `col <= literal`                  | [`ColumnPredicate::LessThanOrEqual`]   |
//! | `col > literal`                   | [`ColumnPredicate::GreaterThan`]      |
//! | `col >= literal`                  | [`ColumnPredicate::GreaterThanOrEqual`]|
//!
//! Conjunctions on the same column can be merged into a single
//! `(lower, upper)` interval via [`ColumnPredicate::Range`]. Disjunctions,
//! multi-column AND, `IN`, `LIKE`, and `BETWEEN` are not supported and
//! remain as residual filters that run over the already-narrowed row set.
//!
//! ## Why this module exists
//!
//! `hatp_engine` deliberately has no DataFusion dependency. The frontend's planner
//! translates DataFusion `Expr` ASTs into `ColumnPredicate` enums, which carry
//! column indexes and `ScalarValue` literals — no planner concepts leak into the
//! storage layer. The SST layer ([`crate::vortex_sst_v2`]) then interprets
//! `ColumnPredicate` using the per-table Arrow schema to produce Vortex expressions
//! for hot-column pushdown.
//!
//! ## Vortex expression pushdown
//!
//! The SST layer translates `&[ColumnPredicate]` into `&[(String, ColumnPredicate)]`
//! tuples using the per-table Arrow schema via [`predicates_with_fields`]. Those
//! tuples are then handed to `SstFormat::scan_with_filter`, which evaluates the
//! predicates against decoded row values.

use arrow_schema::SchemaRef;
use datafusion_common::ScalarValue;

use crate::memtable::VersionedRow;
use crate::Result;

/// A single-column predicate that the SST layer can evaluate against the
/// per-row `value` blob. Every variant carries an explicit column index
/// into the per-table schema to avoid name look-ups during the scan.
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnPredicate {
    /// `col <op> literal` on a single column. The encoding mirrors what
    /// `RowCodec` produces for the matching column slot.
    Eq {
        col: usize,
        value: ScalarValue,
    },
    LessThan {
        col: usize,
        value: ScalarValue,
    },
    LessThanOrEqual {
        col: usize,
        value: ScalarValue,
    },
    GreaterThan {
        col: usize,
        value: ScalarValue,
    },
    GreaterThanOrEqual {
        col: usize,
        value: ScalarValue,
    },
    /// Closed- or open-bound range. Built by merging compatible
    /// `GreaterThan*` + `LessThan*` predicates on the same column.
    Range {
        col: usize,
        /// Inclusive lower bound; `None` means "no lower bound".
        lower: Option<Bound>,
        /// Exclusive upper bound; `None` means "no upper bound".
        upper: Option<Bound>,
    },
}

/// Bound type for [`ColumnPredicate::Range`].
#[derive(Debug, Clone, PartialEq)]
pub struct Bound {
    pub value: ScalarValue,
    pub inclusive: bool,
}

impl Bound {
    pub fn inclusive(value: ScalarValue) -> Self {
        Self { value, inclusive: true }
    }
    pub fn exclusive(value: ScalarValue) -> Self {
        Self { value, inclusive: false }
    }
}

impl ColumnPredicate {
    /// Returns the column index this predicate addresses.
    pub fn column(&self) -> usize {
        match self {
            Self::Eq { col, .. }
            | Self::LessThan { col, .. }
            | Self::LessThanOrEqual { col, .. }
            | Self::GreaterThan { col, .. }
            | Self::GreaterThanOrEqual { col, .. }
            | Self::Range { col, .. } => *col,
        }
    }

    /// Evaluates the predicate against the scalar `value`. Returns `true`
    /// when the value satisfies the predicate, `false` otherwise.
    /// `null` values never satisfy any predicate (matches SQL semantics).
    pub fn evaluate(&self, value: &ScalarValue) -> bool {
        if value.is_null() {
            return false;
        }
        match self {
            Self::Eq { value: rhs, .. } => value == rhs,
            Self::LessThan { value: rhs, .. } => value < rhs,
            Self::LessThanOrEqual { value: rhs, .. } => value <= rhs,
            Self::GreaterThan { value: rhs, .. } => value > rhs,
            Self::GreaterThanOrEqual { value: rhs, .. } => value >= rhs,
            Self::Range {
                lower, upper, ..
            } => {
                if let Some(low) = lower {
                    let ok = if low.inclusive {
                        value >= &low.value
                    } else {
                        value > &low.value
                    };
                    if !ok {
                        return false;
                    }
                }
                if let Some(high) = upper {
                    let ok = if high.inclusive {
                        value <= &high.value
                    } else {
                        value < &high.value
                    };
                    if !ok {
                        return false;
                    }
                }
                true
            }
        }
    }
}

/// Filters `rows` by evaluating `predicates` against the decoded value
/// blob of each row. Returns the rows that satisfy every predicate (AND
/// semantics).
///
/// **Performance.** The legacy Vortex SST stores each row's columns
/// inside a single opaque `value` blob, so every kept row pays one
/// `row_codec::decode_row_values` call. The decode runs even when a row
/// would have been dropped by a prior predicate, so the predicate order
/// matters: sort `predicates` by selectivity (most-restrictive first) to
/// minimise the rows that reach the per-column evaluation. The column
/// index lookup (`predicates[i].column()`) is a single integer
/// comparison, so the overhead per row is dominated by the decode, not
/// the predicate loop.
pub fn apply_row_predicates(
    rows: &[VersionedRow],
    row_schema: &SchemaRef,
    predicates: &[ColumnPredicate],
) -> Result<Vec<VersionedRow>> {
    if predicates.is_empty() {
        return Ok(rows.to_vec());
    }
    if row_schema.fields().is_empty() {
        // Empty schema means the row was never registered with an
        // Arrow schema. Predicates have no columns to address, so
        // return everything (the engine's caller will re-filter in
        // memory).
        return Ok(rows.to_vec());
    }
    let schema = row_schema.as_ref();
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let Some(payload) = row.version.value.as_ref() else {
            // Tombstones have no payload; predicates never match a
            // tombstone, so skip them outright.
            continue;
        };
        let decoded = crate::row_codec::decode_row_values(payload, schema)?;
        if row_matches(&decoded, predicates) {
            out.push(VersionedRow {
                key: row.key.clone(),
                version: row.version.clone(),
            });
        }
    }
    Ok(out)
}

/// Returns true when `decoded` satisfies every predicate. Caller is
/// responsible for ensuring `decoded.len() >= max(pred.column())`; we
/// defensively treat a missing column as "does not match" rather than
/// panicking, because a corrupt row blob would otherwise crash the scan.
fn row_matches(decoded: &[ScalarValue], predicates: &[ColumnPredicate]) -> bool {
    for pred in predicates {
        let col = pred.column();
        match decoded.get(col) {
            None => return false,
            Some(scalar) => {
                if !pred.evaluate(scalar) {
                    return false;
                }
            }
        }
    }
    true
}

/// Translate `&[ColumnPredicate]` (column-index based) to `Vec<(String, ColumnPredicate)>`
/// (field-name based) using `schema` to look up column names.
///
/// This lives here — rather than in [`crate::vortex_sst_v2`] — because it is a
/// pure schema translation between two engine-internal types, not a Vortex-specific
/// operation. Callers at the engine level (e.g. [`crate::Engine::scan_visible_rows`])
/// use this to produce the `(field_name, predicate)` tuples required by
/// [`crate::SstFormat::scan_with_filter`].
pub(crate) fn predicates_with_fields(
    predicates: &[ColumnPredicate],
    schema: &SchemaRef,
) -> Vec<(String, ColumnPredicate)> {
    predicates
        .iter()
        .filter_map(|p| {
            schema
                .fields()
                .get(p.column())
                .map(|f| (f.name().to_string(), p.clone()))
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn eq_predicate_evaluates() {
        let pred = ColumnPredicate::Eq {
            col: 1,
            value: ScalarValue::Int32(Some(30)),
        };
        assert!(pred.evaluate(&ScalarValue::Int32(Some(30))));
        assert!(!pred.evaluate(&ScalarValue::Int32(Some(31))));
    }

    #[test]
    fn range_predicate_evaluates() {
        let pred = ColumnPredicate::Range {
            col: 1,
            lower: Some(Bound::exclusive(ScalarValue::Int32(Some(20)))),
            upper: Some(Bound::inclusive(ScalarValue::Int32(Some(40)))),
        };
        assert!(pred.evaluate(&ScalarValue::Int32(Some(25))));
        assert!(!pred.evaluate(&ScalarValue::Int32(Some(20))));
        assert!(!pred.evaluate(&ScalarValue::Int32(Some(41))));
        assert!(pred.evaluate(&ScalarValue::Int32(Some(40))));
    }

    #[test]
    fn range_predicate_handles_missing_bound() {
        // `id > 10` (no upper) becomes `Range { lower: Some(...), upper: None }`.
        let pred = ColumnPredicate::Range {
            col: 0,
            lower: Some(Bound::exclusive(ScalarValue::Int64(Some(10)))),
            upper: None,
        };
        assert!(pred.evaluate(&ScalarValue::Int64(Some(11))));
        assert!(!pred.evaluate(&ScalarValue::Int64(Some(10))));
    }

    #[test]
    fn null_never_satisfies_predicate() {
        let pred = ColumnPredicate::Eq {
            col: 0,
            value: ScalarValue::Int32(Some(1)),
        };
        assert!(!pred.evaluate(&ScalarValue::Int32(None)));
    }
}