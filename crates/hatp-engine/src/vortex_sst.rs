//! Vortex-backed SST files.
//!
//! Each SST is a Vortex file with the following schema:
//!
//! **Base columns** (always present):
//!
//! | Column     | Type      | Notes                                  |
//! |------------|-----------|----------------------------------------|
//! | `key`      | Binary    | non-null                               |
//! | `value`    | Binary    | nullable; null = tombstone             |
//! | `begin_ts` | UInt64    | non-null, transaction commit timestamp |
//! | `end_ts`   | UInt64    | non-null; [`OPEN_ENDED_TS`] = live    |
//! | `tombstone`| Boolean   | non-null; mirrors value-is-null       |
//!
//! **Business columns** (present when `row_schema` is supplied at write time):
//!
//! Named `col_0`, `col_1`, ... matching the schema field order. These are
//! Vortex-native columns — DataFusion can prune and push down predicates
//! directly without decoding the opaque `value` blob.
//!
//! Vortex's columnar encoding provides zone maps and a sparse index per
//! column, so a `get` / `scan` query only touches the relevant chunks.

use crate::memtable::VersionedRow;
use crate::version::{OPEN_ENDED_TS, Snapshot, VersionedValue};
use crate::{EngineError, Result};
use bytes::Bytes;
use std::path::Path;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::arrays::{BoolArray, PrimitiveArray, StructArray, VarBinViewArray};
use vortex_array::expr::{Expression, and, col, eq, gt, gt_eq, lit, lt};
use vortex_array::stream::ArrayStream;
use vortex_array::stream::ArrayStreamExt;
use vortex_arrow::ArrowSessionExt;
use vortex_array::validity::Validity;
use vortex_buffer::{BitBuffer, Buffer};
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::current::CurrentThreadRuntime;
use vortex_session::VortexSession;

/// Handle returned by `write` once an SST is durably persisted.
/// See `write_async` for the async variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SstHandle {
    /// Logical file id assigned by the manifest.
    pub file_id: u64,
    /// Smallest key in the SST.
    pub min_key: Bytes,
    /// Largest key in the SST.
    pub max_key: Bytes,
    /// Number of rows written.
    pub rows: usize,
    /// Bytes on disk after encoding.
    pub bytes: u64,
}

/// Lazily initializes a single-threaded [`CurrentThreadRuntime`] (smol-based)
/// used to drive Vortex's blocking API.  Delegate to the shared
/// [`crate::vortex_runtime`] so all SST readers share one static runtime.
fn runtime() -> &'static CurrentThreadRuntime {
    crate::vortex_runtime::runtime()
}

/// Build a [`VortexSession`] pre-configured with the shared runtime handle.
pub fn new_session() -> VortexSession {
    crate::vortex_runtime::new_session()
}

/// Writes the rows in `rows` to a new Vortex SST at `path` and returns a
/// durable handle to it.
///
/// When `row_schema` is provided, each row's opaque value blob is decoded
/// and individual business columns are written as separate Vortex columns
/// named `col_0`, `col_1`, ... (matching the schema field order). This
/// enables DataFusion column pruning and predicate pushdown at the Vortex
/// layer. SSTs written without a schema keep the old opaque-blob-only layout.
pub async fn write_async(
    path: &Path,
    rows: &[VersionedRow],
    file_id: u64,
    row_schema: Option<&arrow_schema::Schema>,
) -> Result<SstHandle> {
    let session = new_session();
    let min_key = rows.first().map(|row| row.key.clone()).unwrap_or_default();
    let max_key = rows.last().map(|row| row.key.clone()).unwrap_or_default();
    let row_count = rows.len();

    let mut values: Vec<Option<&[u8]>> = Vec::with_capacity(rows.len());
    let mut begin_ts: Vec<u64> = Vec::with_capacity(rows.len());
    let mut end_ts: Vec<u64> = Vec::with_capacity(rows.len());
    let mut tombstone: Vec<bool> = Vec::with_capacity(rows.len());
    for row in rows {
        values.push(row.version.value.as_ref().map(|v| v.as_ref()));
        begin_ts.push(row.version.begin_ts);
        end_ts.push(row.version.end_ts);
        tombstone.push(row.version.value.is_none());
    }

    let key_array =
        VarBinViewArray::from_iter_bin(rows.iter().map(|r| r.key.as_ref())).into_array();
    let value_array =
        VarBinViewArray::from_iter_nullable_bin(values.iter().copied()).into_array();
    let begin_array =
        PrimitiveArray::new(Buffer::from(begin_ts), Validity::NonNullable).into_array();
    let end_array = PrimitiveArray::new(Buffer::from(end_ts), Validity::NonNullable).into_array();
    let tomb_array = BoolArray::from(BitBuffer::from(tombstone)).into_array();

    // Build the list of (name, array) pairs.  We own the column-name strings
    // so the `&str` references are valid for the scope of `from_fields`.
    let mut field_names: Vec<String> = Vec::new();
    let mut field_arrays: Vec<vortex_array::ArrayRef> = vec![
        key_array,
        value_array,
        begin_array,
        end_array,
        tomb_array,
    ];

    // Append business columns when the table schema is known.
    if let Some(schema) = row_schema {
        if !schema.fields().is_empty() {
            let ncols = schema.fields().len();
            // Pre-decode all rows into per-column scalar vectors.
            let mut col_scalars: Vec<Vec<datafusion_common::ScalarValue>> =
                (0..ncols).map(|_| Vec::with_capacity(row_count)).collect();
            for row in rows {
                match &row.version.value {
                    Some(payload) => {
                        match crate::row_codec::decode_row_values(payload, schema) {
                            Ok(decoded) => {
                                for (ci, s) in decoded.into_iter().enumerate() {
                                    if ci < ncols {
                                        col_scalars[ci].push(s);
                                    }
                                }
                            }
                            Err(_) => {
                                for (ci, field) in schema.fields().iter().enumerate() {
                                    col_scalars[ci].push(typed_null(field.data_type()));
                                }
                            }
                        }
                    }
                    None => {
                        for (ci, field) in schema.fields().iter().enumerate() {
                            col_scalars[ci].push(typed_null(field.data_type()));
                        }
                    }
                }
            }
            // Build one Vortex array per business column.
            for (col_idx, field) in schema.fields().iter().enumerate() {
                let name = format!("col_{col_idx}");
                let array = build_column_array(&col_scalars[col_idx], field.data_type())?;
                field_names.push(name);
                field_arrays.push(array);
            }
        }
    }

    // Build the `(&str, ArrayRef)` slice for `from_fields`.
    let static_names: [&str; 5] = ["key", "value", "begin_ts", "end_ts", "tombstone"];
    let mut field_refs: Vec<(&str, &vortex_array::ArrayRef)> = static_names
        .iter()
        .zip(field_arrays.iter().take(5))
        .map(|(n, a)| (*n, a))
        .collect();
    for (name, array) in field_names.iter().zip(field_arrays.iter().skip(5)) {
        field_refs.push((name.as_str(), array));
    }

    let struct_array = StructArray::from_fields(
        &field_refs
            .iter()
            .map(|(n, a)| (*n, (*a).clone()))
            .collect::<Vec<_>>(),
    )
    .map_err(|err| EngineError::CorruptMessage(err.to_string()))?;

    // Write Vortex data into an in-memory buffer first so we can seal it
    // before any bytes hit disk.
    let mut buffer = Vec::new();
    let _summary = session
        .write_options()
        .blocking(runtime())
        .write(&mut buffer, struct_array.into_array().to_array_iterator())
        .map_err(|err| EngineError::CorruptMessage(err.to_string()))?;

    let disk_bytes = buffer;
    let bytes_on_disk = disk_bytes.len() as u64;
    std::fs::write(path, &disk_bytes)?;

    Ok(SstHandle {
        file_id,
        min_key,
        max_key,
        rows: row_count,
        bytes: bytes_on_disk,
    })
}

/// Synchronous wrapper around [`write_async`].
pub fn write(
    path: &Path,
    rows: &[VersionedRow],
    file_id: u64,
    row_schema: Option<&arrow_schema::Schema>,
) -> Result<SstHandle> {
    runtime().block_on(write_async(path, rows, file_id, row_schema))
}

// ── Business-column helpers ─────────────────────────────────────────────────

/// Returns a typed `None` [`datafusion_common::ScalarValue`] for `data_type`.
fn typed_null(data_type: &arrow_schema::DataType) -> datafusion_common::ScalarValue {
    use arrow_schema::DataType;
    match data_type {
        DataType::Int8 => datafusion_common::ScalarValue::Int8(None),
        DataType::Int16 => datafusion_common::ScalarValue::Int16(None),
        DataType::Int32 => datafusion_common::ScalarValue::Int32(None),
        DataType::Int64 => datafusion_common::ScalarValue::Int64(None),
        DataType::UInt8 => datafusion_common::ScalarValue::UInt8(None),
        DataType::UInt16 => datafusion_common::ScalarValue::UInt16(None),
        DataType::UInt32 => datafusion_common::ScalarValue::UInt32(None),
        DataType::UInt64 => datafusion_common::ScalarValue::UInt64(None),
        DataType::Float32 => datafusion_common::ScalarValue::Float32(None),
        DataType::Float64 => datafusion_common::ScalarValue::Float64(None),
        DataType::Boolean => datafusion_common::ScalarValue::Boolean(None),
        DataType::Utf8 => datafusion_common::ScalarValue::Utf8(None),
        DataType::LargeUtf8 => datafusion_common::ScalarValue::LargeUtf8(None),
        DataType::Binary => datafusion_common::ScalarValue::Binary(None),
        DataType::LargeBinary => datafusion_common::ScalarValue::LargeBinary(None),
        DataType::Date32 => datafusion_common::ScalarValue::Date32(None),
        DataType::Date64 => datafusion_common::ScalarValue::Date64(None),
        _ => datafusion_common::ScalarValue::Null,
    }
}

/// Builds a single Vortex column array from a column of [`ScalarValue`]s.
/// Uses type-specific fast paths for common types; falls back to the
/// `ScalarValue::iter_to_array` + `vortex_arrow` conversion for others.
fn build_column_array(
    scalars: &[datafusion_common::ScalarValue],
    data_type: &arrow_schema::DataType,
) -> Result<vortex_array::ArrayRef> {
    use arrow_schema::DataType;
    use datafusion_common::ScalarValue;

    /// Helper: build a primitive column from a typed value extractor.
    macro_rules! prim_col {
        ($ty:ty, $variant:ident) => {{
            let mut values: Vec<$ty> = Vec::with_capacity(scalars.len());
            let mut bits = Vec::with_capacity(scalars.len());
            for s in scalars {
                match s {
                    ScalarValue::$variant(Some(v)) => {
                        values.push(*v as $ty);
                        bits.push(true);
                    }
                    _ => {
                        values.push(0 as $ty);
                        bits.push(false);
                    }
                }
            }
            PrimitiveArray::new(
                Buffer::from(values),
                Validity::from(BitBuffer::from(bits)),
            )
            .into_array()
        }};
    }

    match data_type {
        DataType::Int64 => Ok(prim_col!(i64, Int64)),
        DataType::Int32 => Ok(prim_col!(i32, Int32)),
        DataType::Int16 => Ok(prim_col!(i16, Int16)),
        DataType::Int8 => Ok(prim_col!(i8, Int8)),
        DataType::UInt64 => Ok(prim_col!(u64, UInt64)),
        DataType::UInt32 => Ok(prim_col!(u32, UInt32)),
        DataType::UInt16 => Ok(prim_col!(u16, UInt16)),
        DataType::UInt8 => Ok(prim_col!(u8, UInt8)),
        DataType::Float64 => {
            let mut values: Vec<f64> = Vec::with_capacity(scalars.len());
            let mut bits = Vec::with_capacity(scalars.len());
            for s in scalars {
                match s {
                    ScalarValue::Float64(Some(v)) => {
                        values.push(*v);
                        bits.push(true);
                    }
                    _ => {
                        values.push(0.0_f64);
                        bits.push(false);
                    }
                }
            }
            Ok(PrimitiveArray::new(Buffer::from(values), Validity::from(BitBuffer::from(bits))).into_array())
        }
        DataType::Float32 => {
            let mut values: Vec<f32> = Vec::with_capacity(scalars.len());
            let mut bits = Vec::with_capacity(scalars.len());
            for s in scalars {
                match s {
                    ScalarValue::Float32(Some(v)) => {
                        values.push(*v);
                        bits.push(true);
                    }
                    _ => {
                        values.push(0.0_f32);
                        bits.push(false);
                    }
                }
            }
            Ok(PrimitiveArray::new(Buffer::from(values), Validity::from(BitBuffer::from(bits))).into_array())
        }
        DataType::Boolean => {
            let mut bits = Vec::with_capacity(scalars.len());
            for s in scalars {
                bits.push(matches!(s, ScalarValue::Boolean(Some(true))));
            }
            Ok(BoolArray::from(BitBuffer::from(bits)).into_array())
        }
        DataType::Utf8 | DataType::LargeUtf8 => {
            let mut strings: Vec<Option<Vec<u8>>> = Vec::with_capacity(scalars.len());
            for s in scalars {
                match s {
                    ScalarValue::Utf8(Some(v)) | ScalarValue::LargeUtf8(Some(v)) => {
                        strings.push(Some(v.as_bytes().to_vec()));
                    }
                    _ => strings.push(None),
                }
            }
            Ok(VarBinViewArray::from_iter_nullable_bin(strings.iter().map(|v| v.as_deref()))
                .into_array())
        }
        DataType::Binary | DataType::LargeBinary => {
            let mut bins: Vec<Option<Vec<u8>>> = Vec::with_capacity(scalars.len());
            for s in scalars {
                match s {
                    ScalarValue::Binary(Some(v)) | ScalarValue::LargeBinary(Some(v)) => {
                        bins.push(Some(v.clone()));
                    }
                    _ => bins.push(None),
                }
            }
            Ok(VarBinViewArray::from_iter_nullable_bin(bins.iter().map(|v| v.as_deref()))
                .into_array())
        }
        DataType::Date32 => Ok(prim_col!(i32, Date32)),
        DataType::Date64 => Ok(prim_col!(i64, Date64)),
        // Timestamps: all stored as i64.
        DataType::Timestamp(_, _) => {
            let mut values: Vec<i64> = Vec::with_capacity(scalars.len());
            let mut bits = Vec::with_capacity(scalars.len());
            for s in scalars {
                match s {
                    ScalarValue::TimestampSecond(Some(v), _)
                    | ScalarValue::TimestampMillisecond(Some(v), _)
                    | ScalarValue::TimestampMicrosecond(Some(v), _)
                    | ScalarValue::TimestampNanosecond(Some(v), _) => {
                        values.push(*v);
                        bits.push(true);
                    }
                    _ => {
                        values.push(0_i64);
                        bits.push(false);
                    }
                }
            }
            Ok(PrimitiveArray::new(Buffer::from(values), Validity::from(BitBuffer::from(bits))).into_array())
        }
        // For types without a dedicated fast path, store as binary debug strings.
        // The `value` column still carries the authoritative opaque blob; business
        // columns are a best-effort acceleration structure for common types.
        _ => {
            let mut bins: Vec<Option<Vec<u8>>> = Vec::with_capacity(scalars.len());
            for s in scalars {
                if s.is_null() {
                    bins.push(None);
                } else {
                    bins.push(Some(format!("{s:?}").into_bytes()));
                }
            }
            Ok(VarBinViewArray::from_iter_nullable_bin(bins.iter().map(|v| v.as_deref()))
                .into_array())
        }
    }
}

// ── Predicate-to-Vortex translation ─────────────────────────────────────────

/// Translates a [`ColumnPredicate`] into a Vortex [`Expression`] targeting
/// the `col_N` business column.  Returns `None` for predicates that cannot
/// be expressed as a simple single-column comparison.
///
/// Column names are pre-computed as `&'static str` (up to 256 columns) to
/// avoid the `Box::leak` memory leak that the previous implementation had.
fn column_pred_to_vortex(
    pred: &crate::column_predicate::ColumnPredicate,
) -> Option<Expression> {
    use crate::column_predicate::ColumnPredicate;
    use datafusion_common::ScalarValue;
    use std::sync::OnceLock;

    /// Pre-computed `col_0` .. `col_255` — leaked once, not per-query.
    static COL_NAMES: OnceLock<&'static [&'static str]> = OnceLock::new();
    let names: &[&str] = COL_NAMES.get_or_init(|| {
        let v: Vec<String> = (0..256).map(|i| format!("col_{i}")).collect();
        // Leak the Vec and its Strings once — ~2 KB total, never freed.
        // This is acceptable for an embedded engine that runs for the
        // lifetime of the process.
        let leaked: &'static [String] = Vec::leak(v);
        // Convert &[String] to &[&str] — we need to leak the &str refs too.
        let strs: Vec<&str> = leaked.iter().map(|s| s.as_str()).collect();
        Vec::leak(strs)
    });
    let col_name: &'static str = names.get(pred.column()).copied().unwrap_or("col_0");
    let col_expr = col(col_name);
    match pred {
        ColumnPredicate::Eq { value, .. } => match value {
            ScalarValue::Int8(Some(v)) => Some(eq(col_expr, lit(*v))),
            ScalarValue::Int16(Some(v)) => Some(eq(col_expr, lit(*v))),
            ScalarValue::Int32(Some(v)) => Some(eq(col_expr, lit(*v))),
            ScalarValue::Int64(Some(v)) => Some(eq(col_expr, lit(*v))),
            ScalarValue::UInt8(Some(v)) => Some(eq(col_expr, lit(*v))),
            ScalarValue::UInt16(Some(v)) => Some(eq(col_expr, lit(*v))),
            ScalarValue::UInt32(Some(v)) => Some(eq(col_expr, lit(*v))),
            ScalarValue::UInt64(Some(v)) => Some(eq(col_expr, lit(*v))),
            ScalarValue::Float32(Some(v)) => Some(eq(col_expr, lit(*v))),
            ScalarValue::Float64(Some(v)) => Some(eq(col_expr, lit(*v))),
            ScalarValue::Utf8(Some(v)) => Some(eq(col_expr, lit(v.as_bytes()))),
            ScalarValue::LargeUtf8(Some(v)) => Some(eq(col_expr, lit(v.as_bytes()))),
            ScalarValue::Boolean(Some(v)) => Some(eq(col_expr, lit(*v))),
            _ => None,
        },
        ColumnPredicate::LessThan { value, .. } => match value {
            ScalarValue::Int8(Some(v)) => Some(lt(col_expr, lit(*v))),
            ScalarValue::Int16(Some(v)) => Some(lt(col_expr, lit(*v))),
            ScalarValue::Int32(Some(v)) => Some(lt(col_expr, lit(*v))),
            ScalarValue::Int64(Some(v)) => Some(lt(col_expr, lit(*v))),
            ScalarValue::Float32(Some(v)) => Some(lt(col_expr, lit(*v))),
            ScalarValue::Float64(Some(v)) => Some(lt(col_expr, lit(*v))),
            _ => None,
        },
        ColumnPredicate::LessThanOrEqual { .. } => None, // Vortex has no <=
        ColumnPredicate::GreaterThan { value, .. } => match value {
            ScalarValue::Int8(Some(v)) => Some(gt(col_expr, lit(*v))),
            ScalarValue::Int16(Some(v)) => Some(gt(col_expr, lit(*v))),
            ScalarValue::Int32(Some(v)) => Some(gt(col_expr, lit(*v))),
            ScalarValue::Int64(Some(v)) => Some(gt(col_expr, lit(*v))),
            ScalarValue::Float32(Some(v)) => Some(gt(col_expr, lit(*v))),
            ScalarValue::Float64(Some(v)) => Some(gt(col_expr, lit(*v))),
            _ => None,
        },
        ColumnPredicate::GreaterThanOrEqual { value, .. } => match value {
            ScalarValue::Int8(Some(v)) => Some(gt_eq(col_expr, lit(*v))),
            ScalarValue::Int16(Some(v)) => Some(gt_eq(col_expr, lit(*v))),
            ScalarValue::Int32(Some(v)) => Some(gt_eq(col_expr, lit(*v))),
            ScalarValue::Int64(Some(v)) => Some(gt_eq(col_expr, lit(*v))),
            ScalarValue::Float32(Some(v)) => Some(gt_eq(col_expr, lit(*v))),
            ScalarValue::Float64(Some(v)) => Some(gt_eq(col_expr, lit(*v))),
            _ => None,
        },
        ColumnPredicate::Range { .. } => None, // caller decomposes
    }
}

/// Reads every row from `path` in original order.
///
/// # Errors
///
/// Returns [`EngineError::CorruptMessage`] if the file is unreadable
/// or does not conform to the 5-column SST schema.
pub fn read_all(path: &Path) -> Result<Vec<VersionedRow>> {
    runtime().block_on(async {
        let session = new_session();
        let mut ctx = ExecutionCtx::new(session);
        let source: std::sync::Arc<dyn vortex_io::VortexReadAt> = {
            let file = vortex_io::std_file::FileReadAt::open(path, runtime().handle())?;
            std::sync::Arc::new(file)
        };
        let file = ctx
            .session()
            .open_options()
            .open(source)
            .await
            .map_err(|err| EngineError::CorruptMessage(err.to_string()))?;
        let stream = file
            .scan()
            .map_err(|err| EngineError::CorruptMessage(err.to_string()))?
            .into_array_stream()
            .map_err(|err| EngineError::CorruptMessage(err.to_string()))?;
        collect_rows(stream, &mut ctx).await
    })
}

/// Returns the visible version for `key` at `snapshot`, or `None` if
/// the key has no visible version in this SST.
///
/// A visible **tombstone** is returned with `value == None` (not filtered
/// out), so the caller can distinguish "key deleted at this snapshot" from
/// "key absent from this file".
pub fn get(
    path: &Path,
    key: &[u8],
    snapshot: Snapshot,
) -> Result<Option<VersionedValue>> {
    runtime().block_on(async {
        let session = new_session();
        let mut ctx = ExecutionCtx::new(session);
        let source: std::sync::Arc<dyn vortex_io::VortexReadAt> = {
            let file = vortex_io::std_file::FileReadAt::open(path, runtime().handle())?;
            std::sync::Arc::new(file)
        };
        let file = ctx
            .session()
            .open_options()
            .open(source)
            .await
            .map_err(|err| EngineError::CorruptMessage(err.to_string()))?;
        // Visibility predicate only — do NOT add `tombstone == false` here:
        // MVCC visibility intervals are non-overlapping, so at most one row
        // per key matches, and returning the tombstone is what lets `get`
        // hide an older live version behind a newer delete.
        let filter: Expression = and(
            eq(col("key"), lit(key)),
            and(
                gt_eq(lit(snapshot.ts), col("begin_ts")),
                gt(col("end_ts"), lit(snapshot.ts)),
            ),
        );
        let stream = file
            .scan()
            .map_err(|err| EngineError::CorruptMessage(err.to_string()))?
            .with_filter(filter)
            .into_array_stream()
            .map_err(|err| EngineError::CorruptMessage(err.to_string()))?;
        let rows = collect_rows(stream, &mut ctx).await?;
        let Some(row) = rows.into_iter().next() else {
            return Ok(None);
        };
        Ok(Some(row.version))
    })
}

/// Returns all rows whose key falls inside `[lower, upper)` and whose
/// visibility interval contains `snapshot`.
///
/// When `predicates` is non-empty, each predicate is translated to a Vortex
pub fn scan(
    path: &Path,
    lower: &[u8],
    upper: &[u8],
    snapshot: Snapshot,
    predicates: &[crate::column_predicate::ColumnPredicate],
) -> Result<Vec<VersionedRow>> {
    runtime().block_on(async {
        let session = new_session();
        let mut ctx = ExecutionCtx::new(session);
        let source: std::sync::Arc<dyn vortex_io::VortexReadAt> = {
            let file = vortex_io::std_file::FileReadAt::open(path, runtime().handle())?;
            std::sync::Arc::new(file)
        };
        let file = ctx
            .session()
            .open_options()
            .open(source)
            .await
            .map_err(|err| EngineError::CorruptMessage(err.to_string()))?;
        let mut filter: Expression = and(
            and(gt_eq(col("key"), lit(lower)), lt(col("key"), lit(upper))),
            and(
                gt_eq(lit(snapshot.ts), col("begin_ts")),
                gt(col("end_ts"), lit(snapshot.ts)),
            ),
        );
        // Append business-column predicates so Vortex can short-circuit rows.
        for pred in predicates {
            if let Some(expr) = column_pred_to_vortex(pred) {
                filter = and(filter, expr);
            }
        }
        let stream = file
            .scan()
            .map_err(|err| EngineError::CorruptMessage(err.to_string()))?
            .with_filter(filter)
            .into_array_stream()
            .map_err(|err| EngineError::CorruptMessage(err.to_string()))?;
        let rows = collect_rows(stream, &mut ctx).await?;
        Ok(rows)
    })
}

pub(crate) async fn collect_rows<S>(stream: S, ctx: &mut ExecutionCtx) -> Result<Vec<VersionedRow>>
where
    S: ArrayStream,
{
    let read = stream
        .read_all()
        .await
        .map_err(|err| EngineError::CorruptMessage(err.to_string()))?;
    let row_count = read.len();
    let struct_array: StructArray = read
        .execute(ctx)
        .map_err(|err| EngineError::CorruptMessage(err.to_string()))?;
    let key_arr: vortex_array::ArrayRef = struct_array
        .unmasked_field_by_name("key")
        .map_err(|_| EngineError::Corrupt("missing key column"))?
        .clone();
    let begin_arr: PrimitiveArray = struct_array
        .unmasked_field_by_name("begin_ts")
        .map_err(|_| EngineError::Corrupt("missing begin_ts column"))?
        .clone()
        .execute(ctx)
        .map_err(|err| EngineError::CorruptMessage(err.to_string()))?;
    let end_arr: PrimitiveArray = struct_array
        .unmasked_field_by_name("end_ts")
        .map_err(|_| EngineError::Corrupt("missing end_ts column"))?
        .clone()
        .execute(ctx)
        .map_err(|err| EngineError::CorruptMessage(err.to_string()))?;
    let _ = struct_array
        .unmasked_field_by_name("tombstone")
        .map_err(|_| EngineError::Corrupt("missing tombstone column"))?;
    let key_arr: VarBinViewArray = key_arr
        .execute(ctx)
        .map_err(|err| EngineError::CorruptMessage(err.to_string()))?;
    let value_arr: vortex_array::ArrayRef = struct_array
        .unmasked_field_by_name("value")
        .map_err(|_| EngineError::Corrupt("missing value column"))?
        .clone();
    let value_arr: VarBinViewArray = value_arr
        .execute(ctx)
        .map_err(|err| EngineError::CorruptMessage(err.to_string()))?;
    let begin_slice = begin_arr.as_slice::<u64>();
    let end_slice = end_arr.as_slice::<u64>();
    let value_validity = value_arr.validity()?;
    let mut out = Vec::with_capacity(row_count);
    for row in 0..row_count {
        let key_buffer = key_arr.bytes_at(row);
        let key = Bytes::copy_from_slice(&key_buffer);
        let is_value_null = !value_validity
            .execute_is_valid(row, ctx)
            .map_err(|err| EngineError::CorruptMessage(err.to_string()))?;
        let value = if is_value_null {
            None
        } else {
            let value_buffer = value_arr.bytes_at(row);
            Some(Bytes::copy_from_slice(&value_buffer))
        };
        let begin_ts = begin_slice
            .get(row)
            .copied()
            .ok_or(EngineError::Corrupt("missing begin_ts row"))?;
        let end_ts_raw = end_slice
            .get(row)
            .copied()
            .ok_or(EngineError::Corrupt("missing end_ts row"))?;
        let version = VersionedValue {
            value,
            begin_ts,
            end_ts: if end_ts_raw == OPEN_ENDED_TS {
                OPEN_ENDED_TS
            } else {
                end_ts_raw
            },
            tx_id: 0,
        };
        out.push(VersionedRow { key, version });
    }
    Ok(out)
}

// ── Vortex → Arrow direct path (bypasses the opaque value blob) ────────────

/// Reads an SST file and converts its business columns directly to Arrow
/// [`RecordBatch`]es via `vortex-arrow`.  This skips the `value` blob decode
/// entirely — Vortex's columnar layout is preserved all the way to DataFusion.
///
/// When `projection` is provided, only the projected columns are read from
/// Vortex (column pruning at the file level).  MVCC visibility and business
/// column predicates are pushed into the Vortex filter expression.
pub fn scan_to_arrow_batches(
    path: &Path,
    lower: &[u8],
    upper: &[u8],
    snapshot: Snapshot,
    predicates: &[crate::column_predicate::ColumnPredicate],
    projection: Option<&[usize]>,
    row_schema: &arrow_schema::Schema,
    batch_size: usize,
    session: &VortexSession,
) -> Result<Vec<arrow_array::RecordBatch>> {
    use arrow_array::RecordBatch;
    use arrow_schema::SchemaRef;
    use std::sync::Arc;

    runtime().block_on(async {
        let mut ctx = ExecutionCtx::new(session.clone());
        let source: std::sync::Arc<dyn vortex_io::VortexReadAt> = {
            let file = vortex_io::std_file::FileReadAt::open(path, runtime().handle())?;
            std::sync::Arc::new(file)
        };
        let file = ctx
            .session()
            .open_options()
            .open(source)
            .await
            .map_err(|err| EngineError::CorruptMessage(err.to_string()))?;

        // ── Build the filter expression ────────────────────────────────
        let mut filter: Expression = and(
            and(gt_eq(col("key"), lit(lower)), lt(col("key"), lit(upper))),
            and(
                gt_eq(lit(snapshot.ts), col("begin_ts")),
                gt(col("end_ts"), lit(snapshot.ts)),
            ),
        );
        for pred in predicates {
            if let Some(expr) = column_pred_to_vortex(pred) {
                filter = and(filter, expr);
            }
        }

        let stream = file
            .scan()
            .map_err(|err| EngineError::CorruptMessage(err.to_string()))?
            .with_filter(filter)
            .into_array_stream()
            .map_err(|err| EngineError::CorruptMessage(err.to_string()))?;

        // TODO(Vortex): `read_all()` reads all columns (key, value, begin_ts,
        // end_ts, tombstone, col_0..col_N) even when only a subset is projected.
        // Vortex 0.83's `scan()` API does not support column-level projection.
        // Once it does, replace with a projected scan for the needed columns.
        let read = stream
            .read_all()
            .await
            .map_err(|err| EngineError::CorruptMessage(err.to_string()))?;
        let row_count = read.len();
        if row_count == 0 {
            let projected_schema: SchemaRef = match projection {
                Some(proj) => Arc::new(row_schema.project(proj)?),
                None => Arc::new(row_schema.clone()),
            };
            return Ok(vec![RecordBatch::new_empty(projected_schema)]);
        }

        let struct_array: StructArray = read
            .execute(&mut ctx)
            .map_err(|err| EngineError::CorruptMessage(err.to_string()))?;

        let ncols = row_schema.fields().len();
        let cols_to_read: Vec<usize> = match projection {
            Some(proj) => proj.to_vec(),
            None => (0..ncols).collect(),
        };

        // ── Read projected business columns from Vortex ─────────────────
        let mut arrow_columns: Vec<Arc<dyn arrow_array::Array>> =
            Vec::with_capacity(cols_to_read.len());
        for &col_idx in &cols_to_read {
            let col_name = format!("col_{col_idx}");
            let field = row_schema.field(col_idx);
            let vortex_arr: vortex_array::ArrayRef = struct_array
                .unmasked_field_by_name(&col_name)
                .map_err(|_| {
                    EngineError::CorruptMessage(format!(
                        "business column `{col_name}` missing from SST `{}`",
                        path.display()
                    ))
                })?
                .clone();
            let arrow_arr = session
                .arrow()
                .execute_arrow(vortex_arr, Some(field), &mut ctx)
                .map_err(|err| EngineError::CorruptMessage(err.to_string()))?;
            arrow_columns.push(arrow_arr);
        }

        let projected_schema: SchemaRef = if cols_to_read.len() == ncols {
            Arc::new(row_schema.clone())
        } else {
            Arc::new(row_schema.project(&cols_to_read)?)
        };

        // ── Slice into batches ──────────────────────────────────────────
        let effective_batch_size = if batch_size == usize::MAX {
            row_count
        } else {
            batch_size.max(1)
        };
        let mut batches = Vec::new();
        let mut start = 0;
        while start < row_count {
            let end = (start + effective_batch_size).min(row_count);
            let sliced: Vec<Arc<dyn arrow_array::Array>> = arrow_columns
                .iter()
                .map(|col| col.slice(start, end - start))
                .collect();
            let batch = RecordBatch::try_new(Arc::clone(&projected_schema), sliced)
                .map_err(|err| EngineError::CorruptMessage(err.to_string()))?;
            batches.push(batch);
            start = end;
        }
        Ok(batches)
    })
}
