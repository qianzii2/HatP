//! SST format abstraction.
//!
//! Decouples the engine's flush / compaction / read paths from the concrete
//! Vortex layout so alternative formats (WiscKey vLog, PebblesDB FLSM, custom
//! SST) can be swapped in behind one trait. [`VortexFormat`] is the default and
//! delegates to the existing [`crate::vortex_sst`] free functions.

use crate::column_predicate::ColumnPredicate;
use crate::memtable::VersionedRow;
use crate::version::{Snapshot, VersionedValue};
use crate::vortex_sst;
use crate::{EngineError, Result};
use arrow_schema::{Schema, SchemaRef};
use bytes::Bytes;
use datafusion_common::ScalarValue;
use std::collections::HashMap;
use std::path::Path;

// Re-export the SST handle so callers depend on the abstraction, not the
// concrete Vortex module.
pub use crate::vortex_sst::SstHandle;

/// Per-file key-range and size summary, exposed for DataFusion statistics
/// pushdown. Both key-level and (where supported) column-level statistics
/// are populated by [`SstFormat::zone_map`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZoneMap {
    /// Inclusive smallest key.
    pub min_key: Bytes,
    /// Inclusive largest key.
    pub max_key: Bytes,
    /// Number of rows in the file.
    pub total_rows: usize,
    /// Bytes on disk.
    pub total_bytes: u64,
    /// Per-column minimum value across all rows in this SST.
    /// Computed by decoding `hot`/`cold` narrow blobs in [`ColumnarVortexFormat::zone_map`];
    /// always empty for [`VortexFormat`] (opaque blobs, no per-column stats).
    pub column_min: HashMap<String, ScalarValue>,
    /// Per-column maximum value across all rows in this SST.
    pub column_max: HashMap<String, ScalarValue>,
    /// Per-column null count. Sum across columns may not equal
    /// `total_rows` because a row can have NULLs in multiple columns.
    pub column_null_count: HashMap<String, usize>,
}

/// The SST read/write contract.
///
/// `scan_with_filter` is the load-bearing entry point: the unfiltered
/// `scan` method forwards to it with `predicates = None`, and the engine's
/// filter pushdown path (`scan_table_with_filters`) calls it with
/// `predicates = Some(...)`. Implementations that cannot push predicates
/// into Vortex evaluate them after the row-decode step (still cheaper than
/// letting the engine scan every row, because the Vortex key-range +
/// visibility filter already culled most candidates upstream).
#[allow(clippy::too_many_arguments)]
pub trait SstFormat: Send + Sync + 'static {
    /// Writes `rows` to a new SST at `path` and returns a durable handle.
    fn write(
        &self,
        path: &Path,
        rows: &[VersionedRow],
        file_id: u64,
        row_schema: Option<&Schema>,
    ) -> Result<SstHandle>;

    /// Reads every row from `path` in original order.
    fn read_all(&self, path: &Path) -> Result<Vec<VersionedRow>>;

    /// Point lookup for `key` at `snapshot`.
    fn get(
        &self,
        path: &Path,
        key: &[u8],
        snapshot: Snapshot,
    ) -> Result<Option<VersionedValue>>;

    /// Range scan over `[lower, upper)` at `snapshot`, with optional pushdown.
    fn scan_with_filter(
        &self,
        path: &Path,
        lower: &[u8],
        upper: &[u8],
        snapshot: Snapshot,
        projection: Option<&[usize]>,
        predicates: Option<&[(String, ColumnPredicate)]>,
        row_schema: Option<&SchemaRef>,
    ) -> Result<Vec<VersionedRow>>;

    /// Returns the file's key-range / size summary.
    fn zone_map(
        &self,
        path: &Path,
        row_schema: Option<&Schema>,
    ) -> Result<ZoneMap>;
}

/// Returns `true` when a [`ColumnPredicate`] can be pushed into the Vortex
/// layer (via `col_N` business columns).  Predicates that Vortex cannot
/// express natively (`<=`, `Range`, non-primitive types) return `false` and
/// are applied as a post-filter by `apply_row_predicates`.
fn can_vortex_pushdown(pred: &ColumnPredicate) -> bool {
    match pred {
        ColumnPredicate::Eq { value, .. }
        | ColumnPredicate::LessThan { value, .. }
        | ColumnPredicate::GreaterThan { value, .. }
        | ColumnPredicate::GreaterThanOrEqual { value, .. } => {
            use datafusion_common::ScalarValue;
            matches!(
                value,
                ScalarValue::Int8(Some(_))
                    | ScalarValue::Int16(Some(_))
                    | ScalarValue::Int32(Some(_))
                    | ScalarValue::Int64(Some(_))
                    | ScalarValue::UInt8(Some(_))
                    | ScalarValue::UInt16(Some(_))
                    | ScalarValue::UInt32(Some(_))
                    | ScalarValue::UInt64(Some(_))
                    | ScalarValue::Float32(Some(_))
                    | ScalarValue::Float64(Some(_))
                    | ScalarValue::Utf8(Some(_))
                    | ScalarValue::LargeUtf8(Some(_))
                    | ScalarValue::Boolean(Some(_))
            )
        }
        ColumnPredicate::LessThanOrEqual { .. } | ColumnPredicate::Range { .. } => false,
    }
}

/// The default Vortex-backed format.
#[derive(Debug, Clone, Copy, Default)]
pub struct VortexFormat;

impl VortexFormat {
    /// Creates a new Vortex format handle.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl SstFormat for VortexFormat {
    fn write(
        &self,
        path: &Path,
        rows: &[VersionedRow],
        file_id: u64,
        _row_schema: Option<&Schema>,
    ) -> Result<SstHandle> {
        vortex_sst::write(path, rows, file_id, _row_schema)
    }

    fn read_all(&self, path: &Path) -> Result<Vec<VersionedRow>> {
        vortex_sst::read_all(path)
    }

    fn get(
        &self,
        path: &Path,
        key: &[u8],
        snapshot: Snapshot,
    ) -> Result<Option<VersionedValue>> {
        vortex_sst::get(path, key, snapshot)
    }

    fn scan_with_filter(
        &self,
        path: &Path,
        lower: &[u8],
        upper: &[u8],
        snapshot: Snapshot,
        _projection: Option<&[usize]>,
        predicates: Option<&[(String, ColumnPredicate)]>,
        row_schema: Option<&SchemaRef>,
    ) -> Result<Vec<VersionedRow>> {
        // Single pass: split predicates into pushdownable and residual.
        let mut pred_slice: Vec<ColumnPredicate> = Vec::new();
        let mut residual: Vec<ColumnPredicate> = Vec::new();
        if let Some(preds) = predicates {
            for (_, pred) in preds {
                if can_vortex_pushdown(pred) {
                    pred_slice.push(pred.clone());
                } else {
                    residual.push(pred.clone());
                }
            }
        }
        let rows = vortex_sst::scan(path, lower, upper, snapshot, &pred_slice)?;
        if residual.is_empty() {
            Ok(rows)
        } else {
            let schema = row_schema.ok_or_else(|| {
                EngineError::CorruptMessage(
                    "scan_with_filter requires row_schema when predicates are \
                     supplied: the legacy Vortex layout decodes the `value` blob \
                     row by row and needs the per-table Arrow schema"
                        .to_owned(),
                )
            })?;
            crate::column_predicate::apply_row_predicates(&rows, schema, &residual)
        }
    }

    fn zone_map(&self, path: &Path, _row_schema: Option<&Schema>) -> Result<ZoneMap> {
        let rows = vortex_sst::read_all(path)?;
        let min_key = rows.first().map(|row| row.key.clone()).unwrap_or_default();
        let max_key = rows.last().map(|row| row.key.clone()).unwrap_or_default();
        let total_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        Ok(ZoneMap {
            min_key,
            max_key,
            total_rows: rows.len(),
            total_bytes,
            column_min: HashMap::new(),
            column_max: HashMap::new(),
            column_null_count: HashMap::new(),
        })
    }
}