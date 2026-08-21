//! Compact single-row codec — encodes one row's values with no Arrow IPC
//! metadata.
//!
//! # Why not Arrow IPC per row
//!
//! The previous `ingest_batches` encoded every row as its own Arrow IPC
//! stream, so each stored value carried a full schema flatbuffer plus batch
//! metadata (~200+ bytes of overhead) for a single row. The schema is now
//! shared per table (stored once in the engine's schema registry), and a value
//! is just the row's column values in a fixed, self-delimiting layout:
//!
//! ```text
//!   per column:
//!     [1 byte marker: 0 = NULL, 1 = present]
//!     [if present: value bytes (little-endian, or u32-length-prefixed)]
//! ```
//!
//! Decoding is driven by the shared [`arrow_schema::Schema`], so the type of
//! each column is known without repeating it per row.

use crate::{EngineError, Result};
use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, Schema, TimeUnit};
use datafusion_common::ScalarValue;
use hatp_types::codec::array_slot_to_scalar;
use std::sync::Arc;

/// Encodes row `row` of `batch` into compact bytes (one marker byte per
/// column, followed by the non-null values).
pub fn encode_row(batch: &RecordBatch, row: usize) -> Result<Vec<u8>> {
    encode_row_narrow(batch, row, 0..batch.num_columns())
}

/// P2.2: encodes row `row` of `batch` but only the columns listed in `cols`,
/// yielding a narrower blob. The marker layout is identical to [`encode_row`]
/// (1 byte null marker + value bytes per column), so a `decode_row_narrow`
/// call with the same `cols` slice round-trips the row. Used by column-group
/// writers that split a row into per-group blobs so that a hot column can be
/// read without paying for cold-column bytes.
pub fn encode_row_narrow(
    batch: &RecordBatch,
    row: usize,
    cols: impl IntoIterator<Item = usize>,
) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(8);
    for col in cols {
        let array = batch.column(col);
        let scalar = array_slot_to_scalar(array.as_ref(), row);
        if scalar.is_null() {
            out.push(0);
        } else {
            out.push(1);
            scalar_to_value_bytes(&mut out, &scalar)?;
        }
    }
    Ok(out)
}

/// Decodes compact bytes produced by [`encode_row`] back into a single-row
/// [`RecordBatch`] whose columns match `schema`.
///
/// This is a convenience wrapper over [`decode_row_values`]. The production
/// table scan uses [`decode_row_values`] directly to accumulate columns and
/// materialise arrays in one shot (avoiding one `RecordBatch` per key); this
/// function exists for single-row round-trips (tests, debugging).
pub fn decode_row(bytes: &[u8], schema: &Schema) -> Result<RecordBatch> {
    let values = decode_row_values(bytes, schema)?;
    let mut columns = Vec::with_capacity(values.len());
    for (scalar, field) in values.iter().zip(schema.fields()) {
        columns.push(scalar_to_array(scalar, 1, field.data_type())?);
    }
    RecordBatch::try_new(Arc::new(schema.clone()), columns)
        .map_err(|err| EngineError::CorruptMessage(format!("row codec: build batch: {err}")))
}

/// Decodes compact bytes into one [`ScalarValue`] per column (null columns
/// become a *typed* null so callers can build a column without a schema
/// lookup). This is the streaming-friendly entry point: a table scan
/// accumulates these per column and materialises arrays in one shot instead of
/// building a single-row [`RecordBatch`] per key.
pub fn decode_row_values(bytes: &[u8], schema: &Schema) -> Result<Vec<ScalarValue>> {
    decode_row_narrow(bytes, schema, 0..schema.fields().len())
}

/// Reuses `out` to decode one row into [`ScalarValue`]s, avoiding the
/// per-row heap allocation of [`decode_row_values`].  The caller must
/// `out.clear()` before each call; the vector is resized to match the
/// schema on the first call and reused thereafter.
pub fn decode_row_into(
    bytes: &[u8],
    schema: &Schema,
    out: &mut Vec<ScalarValue>,
) -> Result<()> {
    let ncols = schema.fields().len();
    out.clear();
    out.reserve(ncols);
    let mut pos = 0usize;
    #[allow(clippy::indexing_slicing)]
    for col in 0..ncols {
        let data_type = schema
            .fields()
            .get(col)
            .ok_or_else(|| {
                EngineError::CorruptMessage(format!(
                    "row codec: column index {col} out of range (schema has {} columns)",
                    schema.fields().len()
                ))
            })?
            .data_type();
        let marker = *bytes
            .get(pos)
            .ok_or_else(|| EngineError::Corrupt("row codec: truncated null marker"))?;
        pos += 1;
        let scalar = if marker == 0 {
            typed_null(data_type)
        } else {
            let (scalar, next) = value_bytes_to_scalar(bytes, pos, data_type)?;
            pos = next;
            scalar
        };
        out.push(scalar);
    }
    Ok(())
}

/// Full-projection fast path: all columns, no gaps.  Avoids the
/// `wanted`/`col_order` heap allocations that `decode_row_narrow` uses
/// for partial projections.  Called once per row during a full table scan.
fn decode_row_full(bytes: &[u8], schema: &Schema, ncols: usize) -> Result<Vec<ScalarValue>> {
    let mut values = Vec::with_capacity(ncols);
    let mut pos = 0usize;
    #[allow(clippy::indexing_slicing)]
    for col in 0..ncols {
        let data_type = schema
            .fields()
            .get(col)
            .ok_or_else(|| {
                EngineError::CorruptMessage(format!(
                    "row codec: column index {col} out of range (schema has {} columns)",
                    schema.fields().len()
                ))
            })?
            .data_type();
        let marker = *bytes
            .get(pos)
            .ok_or_else(|| EngineError::Corrupt("row codec: truncated null marker"))?;
        pos += 1;
        let scalar = if marker == 0 {
            typed_null(data_type)
        } else {
            let (scalar, next) = value_bytes_to_scalar(bytes, pos, data_type)?;
            pos = next;
            scalar
        };
        values.push(scalar);
    }
    Ok(values)
}

/// P2.2: decodes a narrow blob produced by [`encode_row_narrow`]. The
/// `cols` slice must mirror the one used during encoding — the marker/value
/// stream is index-aligned with `cols`, not the full schema. The output
/// vector holds one `ScalarValue` per entry in `cols`, in the same order.
///
/// PR 5.4: when `cols` is a subset of the full schema, unrequested columns
/// are skipped (their marker and value bytes are consumed but not decoded).
/// This allows the scan path to only materialize projected columns.
pub fn decode_row_narrow(
    bytes: &[u8],
    schema: &Schema,
    cols: impl IntoIterator<Item = usize>,
) -> Result<Vec<ScalarValue>> {
    let cols: Vec<usize> = cols.into_iter().collect();
    let ncols = schema.fields().len();
    // Fast path: full projection (all columns, no gaps).  Skip the
    // wanted/col_order lookup tables and decode directly into the
    // output vector — saves two heap allocations per row.
    if cols.len() == ncols && cols.iter().enumerate().all(|(i, &c)| i == c) {
        return decode_row_full(bytes, schema, ncols);
    }
    // The indexing below is bounds-checked: `wanted` and `col_order` are
    // sized to `ncols`, and `col < ncols` is verified before each access.
    // `values` is sized to `cols.len()` and `col_order[col]` maps to it.
    #[allow(clippy::indexing_slicing)]
    {
    // Build a quick lookup: which columns are requested?
    let mut wanted = vec![false; ncols];
    let mut col_order = vec![usize::MAX; ncols]; // maps schema col → output position
    for (out_idx, &col) in cols.iter().enumerate() {
        if col < ncols {
            wanted[col] = true;
            col_order[col] = out_idx;
        }
    }
    let mut values: Vec<Option<ScalarValue>> = vec![None; cols.len()];
    let mut pos = 0usize;
    // Walk ALL columns in schema order, decode only the requested ones.
    for col in 0..ncols {
        let data_type = schema
            .fields()
            .get(col)
            .ok_or_else(|| {
                EngineError::CorruptMessage(format!(
                    "row codec: column index {col} out of range (schema has {} columns)",
                    schema.fields().len()
                ))
            })?
            .data_type();
        let marker = *bytes
            .get(pos)
            .ok_or_else(|| EngineError::Corrupt("row codec: truncated null marker"))?;
        pos += 1;
        if wanted[col] {
            let scalar = if marker == 0 {
                typed_null(data_type)
            } else {
                let (scalar, next) = value_bytes_to_scalar(bytes, pos, data_type)?;
                pos = next;
                scalar
            };
            values[col_order[col]] = Some(scalar);
        } else if marker != 0 {
            // Skip past the value bytes of an unrequested column.
            let (_, next) = value_bytes_to_scalar(bytes, pos, data_type)?;
            pos = next;
        }
    }
    // Unwrap the Option values (all must be Some by construction).
    let mut out = Vec::with_capacity(values.len());
    for v in values {
        out.push(v.unwrap_or(ScalarValue::Null));
    }
    Ok(out)
    } // end allow(clippy::indexing_slicing)
}

/// Returns a typed `None` [`ScalarValue`] for `data_type` (a bare
/// [`ScalarValue::Null`] would lose the column type when building an array).
fn typed_null(data_type: &DataType) -> ScalarValue {
    match data_type {
        DataType::Int8 => ScalarValue::Int8(None),
        DataType::Int16 => ScalarValue::Int16(None),
        DataType::Int32 => ScalarValue::Int32(None),
        DataType::Int64 => ScalarValue::Int64(None),
        DataType::UInt8 => ScalarValue::UInt8(None),
        DataType::UInt16 => ScalarValue::UInt16(None),
        DataType::UInt32 => ScalarValue::UInt32(None),
        DataType::UInt64 => ScalarValue::UInt64(None),
        DataType::Float32 => ScalarValue::Float32(None),
        DataType::Float64 => ScalarValue::Float64(None),
        DataType::Boolean => ScalarValue::Boolean(None),
        DataType::Utf8 => ScalarValue::Utf8(None),
        DataType::LargeUtf8 => ScalarValue::LargeUtf8(None),
        DataType::Binary => ScalarValue::Binary(None),
        DataType::LargeBinary => ScalarValue::LargeBinary(None),
        DataType::Date32 => ScalarValue::Date32(None),
        DataType::Date64 => ScalarValue::Date64(None),
        DataType::Timestamp(TimeUnit::Second, tz) => {
            ScalarValue::TimestampSecond(None, tz.clone())
        }
        DataType::Timestamp(TimeUnit::Millisecond, tz) => {
            ScalarValue::TimestampMillisecond(None, tz.clone())
        }
        DataType::Timestamp(TimeUnit::Microsecond, tz) => {
            ScalarValue::TimestampMicrosecond(None, tz.clone())
        }
        DataType::Timestamp(TimeUnit::Nanosecond, tz) => {
            ScalarValue::TimestampNanosecond(None, tz.clone())
        }
        DataType::Time32(TimeUnit::Second) => ScalarValue::Time32Second(None),
        DataType::Time32(TimeUnit::Millisecond) => ScalarValue::Time32Millisecond(None),
        DataType::Time64(TimeUnit::Microsecond) => ScalarValue::Time64Microsecond(None),
        DataType::Time64(TimeUnit::Nanosecond) => ScalarValue::Time64Nanosecond(None),
        DataType::Interval(arrow_schema::IntervalUnit::YearMonth) => {
            ScalarValue::IntervalYearMonth(None)
        }
        DataType::Interval(arrow_schema::IntervalUnit::DayTime) => ScalarValue::IntervalDayTime(None),
        DataType::Interval(arrow_schema::IntervalUnit::MonthDayNano) => {
            ScalarValue::IntervalMonthDayNano(None)
        }
        DataType::Duration(TimeUnit::Second) => ScalarValue::DurationSecond(None),
        DataType::Duration(TimeUnit::Millisecond) => ScalarValue::DurationMillisecond(None),
        DataType::Duration(TimeUnit::Microsecond) => ScalarValue::DurationMicrosecond(None),
        DataType::Duration(TimeUnit::Nanosecond) => ScalarValue::DurationNanosecond(None),
        // Unsupported types fall back to a bare NULL; such columns are already
        // rejected at encode time, so this is unreachable in practice.
        _ => ScalarValue::Null,
    }
}

/// Appends a non-null [`ScalarValue`] to `out` in the compact layout.
fn scalar_to_value_bytes(out: &mut Vec<u8>, scalar: &ScalarValue) -> Result<()> {
    match scalar {
        ScalarValue::Int8(Some(v)) => out.push(*v as u8),
        ScalarValue::Int16(Some(v)) => out.extend_from_slice(&v.to_le_bytes()),
        ScalarValue::Int32(Some(v)) => out.extend_from_slice(&v.to_le_bytes()),
        ScalarValue::Int64(Some(v)) => out.extend_from_slice(&v.to_le_bytes()),
        ScalarValue::UInt8(Some(v)) => out.push(*v),
        ScalarValue::UInt16(Some(v)) => out.extend_from_slice(&v.to_le_bytes()),
        ScalarValue::UInt32(Some(v)) => out.extend_from_slice(&v.to_le_bytes()),
        ScalarValue::UInt64(Some(v)) => out.extend_from_slice(&v.to_le_bytes()),
        ScalarValue::Float32(Some(v)) => out.extend_from_slice(&v.to_bits().to_le_bytes()),
        ScalarValue::Float64(Some(v)) => out.extend_from_slice(&v.to_bits().to_le_bytes()),
        ScalarValue::Boolean(Some(v)) => out.push(u8::from(*v)),
        ScalarValue::Utf8(Some(s)) | ScalarValue::LargeUtf8(Some(s)) => {
            put_len_prefixed(out, s.as_bytes())?;
        }
        ScalarValue::Binary(Some(b)) | ScalarValue::LargeBinary(Some(b)) => {
            put_len_prefixed(out, b)?;
        }
        ScalarValue::Date32(Some(v)) => out.extend_from_slice(&v.to_le_bytes()),
        ScalarValue::Date64(Some(v)) => out.extend_from_slice(&v.to_le_bytes()),
        ScalarValue::TimestampSecond(Some(v), _)
        | ScalarValue::TimestampMillisecond(Some(v), _)
        | ScalarValue::TimestampMicrosecond(Some(v), _)
        | ScalarValue::TimestampNanosecond(Some(v), _) => out.extend_from_slice(&v.to_le_bytes()),
        ScalarValue::Time32Second(Some(v)) | ScalarValue::Time32Millisecond(Some(v)) => {
            out.extend_from_slice(&v.to_le_bytes());
        }
        ScalarValue::Time64Microsecond(Some(v)) | ScalarValue::Time64Nanosecond(Some(v)) => {
            out.extend_from_slice(&v.to_le_bytes());
        }
        ScalarValue::IntervalYearMonth(Some(v)) => out.extend_from_slice(&v.to_le_bytes()),
        ScalarValue::IntervalDayTime(Some(v)) => {
            out.extend_from_slice(&v.days.to_le_bytes());
            out.extend_from_slice(&v.milliseconds.to_le_bytes());
        }
        ScalarValue::IntervalMonthDayNano(Some(v)) => {
            out.extend_from_slice(&v.months.to_le_bytes());
            out.extend_from_slice(&v.days.to_le_bytes());
            out.extend_from_slice(&v.nanoseconds.to_le_bytes());
        }
        ScalarValue::DurationSecond(Some(v))
        | ScalarValue::DurationMillisecond(Some(v))
        | ScalarValue::DurationMicrosecond(Some(v))
        | ScalarValue::DurationNanosecond(Some(v)) => out.extend_from_slice(&v.to_le_bytes()),
        _ => {
            return Err(EngineError::CorruptMessage(format!(
                "row codec: unsupported value type {}",
                scalar.data_type()
            )));
        }
    }
    Ok(())
}

/// Reads a value of type `data_type` at `pos`; returns `(scalar, next_pos)`.
fn value_bytes_to_scalar(
    bytes: &[u8],
    pos: usize,
    data_type: &DataType,
) -> Result<(ScalarValue, usize)> {
    /// Reads `N` bytes at `pos` into an array. Caller guarantees `pos + N <= bytes.len()`.
    /// Uses unsafe pointer copy to eliminate bounds checks on the hot decode path.
    #[inline(always)]
    unsafe fn read_fixed<const N: usize>(bytes: &[u8], pos: usize) -> [u8; N] {
        debug_assert!(pos + N <= bytes.len());
        let mut arr = [0u8; N];
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr().add(pos), arr.as_mut_ptr(), N); }
        arr
    }

    // SAFETY: all arms below are guarded by the decode_row_narrow loop which
    // checks each marker byte and advances pos accordingly, guaranteeing
    // pos + N <= bytes.len() for every fixed-width read.
    unsafe {
    match data_type {
        DataType::Int8 => {
            let a = read_fixed::<1>(bytes, pos);
            Ok((ScalarValue::Int8(Some(a[0] as i8)), pos + 1))
        }
        DataType::Int16 => {
            let a = read_fixed::<2>(bytes, pos);
            Ok((ScalarValue::Int16(Some(i16::from_le_bytes(a))), pos + 2))
        }
        DataType::Int32 => {
            let a = read_fixed::<4>(bytes, pos);
            Ok((ScalarValue::Int32(Some(i32::from_le_bytes(a))), pos + 4))
        }
        DataType::Int64 => {
            let a = read_fixed::<8>(bytes, pos);
            Ok((ScalarValue::Int64(Some(i64::from_le_bytes(a))), pos + 8))
        }
        DataType::UInt8 => {
            let a = read_fixed::<1>(bytes, pos);
            Ok((ScalarValue::UInt8(Some(a[0])), pos + 1))
        }
        DataType::UInt16 => {
            let a = read_fixed::<2>(bytes, pos);
            Ok((ScalarValue::UInt16(Some(u16::from_le_bytes(a))), pos + 2))
        }
        DataType::UInt32 => {
            let a = read_fixed::<4>(bytes, pos);
            Ok((ScalarValue::UInt32(Some(u32::from_le_bytes(a))), pos + 4))
        }
        DataType::UInt64 => {
            let a = read_fixed::<8>(bytes, pos);
            Ok((ScalarValue::UInt64(Some(u64::from_le_bytes(a))), pos + 8))
        }
        DataType::Float32 => {
            let a = read_fixed::<4>(bytes, pos);
            Ok((
                ScalarValue::Float32(Some(f32::from_bits(u32::from_le_bytes(a)))),
                pos + 4,
            ))
        }
        DataType::Float64 => {
            let a = read_fixed::<8>(bytes, pos);
            Ok((
                ScalarValue::Float64(Some(f64::from_bits(u64::from_le_bytes(a)))),
                pos + 8,
            ))
        }
        DataType::Boolean => {
            let a = read_fixed::<1>(bytes, pos);
            Ok((ScalarValue::Boolean(Some(a[0] != 0)), pos + 1))
        }
        DataType::Utf8 => {
            let (s, next) = take_len_prefixed(bytes, pos)?;
            let s = String::from_utf8(s)
                .map_err(|_| EngineError::Corrupt("row codec: non-UTF8 string"))?;
            Ok((ScalarValue::Utf8(Some(s)), next))
        }
        DataType::LargeUtf8 => {
            let (s, next) = take_len_prefixed(bytes, pos)?;
            let s = String::from_utf8(s)
                .map_err(|_| EngineError::Corrupt("row codec: non-UTF8 string"))?;
            Ok((ScalarValue::LargeUtf8(Some(s)), next))
        }
        DataType::Binary => {
            let (b, next) = take_len_prefixed(bytes, pos)?;
            Ok((ScalarValue::Binary(Some(b)), next))
        }
        DataType::LargeBinary => {
            let (b, next) = take_len_prefixed(bytes, pos)?;
            Ok((ScalarValue::LargeBinary(Some(b)), next))
        }
        DataType::Date32 => {
            let a = read_fixed::<4>(bytes, pos);
            Ok((ScalarValue::Date32(Some(i32::from_le_bytes(a))), pos + 4))
        }
        DataType::Date64 => {
            let a = read_fixed::<8>(bytes, pos);
            Ok((ScalarValue::Date64(Some(i64::from_le_bytes(a))), pos + 8))
        }
        DataType::Timestamp(TimeUnit::Second, tz) => {
            let a = read_fixed::<8>(bytes, pos);
            Ok((
                ScalarValue::TimestampSecond(Some(i64::from_le_bytes(a)), tz.clone()),
                pos + 8,
            ))
        }
        DataType::Timestamp(TimeUnit::Millisecond, tz) => {
            let a = read_fixed::<8>(bytes, pos);
            Ok((
                ScalarValue::TimestampMillisecond(Some(i64::from_le_bytes(a)), tz.clone()),
                pos + 8,
            ))
        }
        DataType::Timestamp(TimeUnit::Microsecond, tz) => {
            let a = read_fixed::<8>(bytes, pos);
            Ok((
                ScalarValue::TimestampMicrosecond(Some(i64::from_le_bytes(a)), tz.clone()),
                pos + 8,
            ))
        }
        DataType::Timestamp(TimeUnit::Nanosecond, tz) => {
            let a = read_fixed::<8>(bytes, pos);
            Ok((
                ScalarValue::TimestampNanosecond(Some(i64::from_le_bytes(a)), tz.clone()),
                pos + 8,
            ))
        }
        DataType::Time32(TimeUnit::Second) => {
            let a = read_fixed::<4>(bytes, pos);
            Ok((ScalarValue::Time32Second(Some(i32::from_le_bytes(a))), pos + 4))
        }
        DataType::Time32(TimeUnit::Millisecond) => {
            let a = read_fixed::<4>(bytes, pos);
            Ok((
                ScalarValue::Time32Millisecond(Some(i32::from_le_bytes(a))),
                pos + 4,
            ))
        }
        DataType::Time64(TimeUnit::Microsecond) => {
            let a = read_fixed::<8>(bytes, pos);
            Ok((
                ScalarValue::Time64Microsecond(Some(i64::from_le_bytes(a))),
                pos + 8,
            ))
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            let a = read_fixed::<8>(bytes, pos);
            Ok((
                ScalarValue::Time64Nanosecond(Some(i64::from_le_bytes(a))),
                pos + 8,
            ))
        }
        DataType::Interval(arrow_schema::IntervalUnit::YearMonth) => {
            let months = i32::from_le_bytes(read_fixed::<4>(bytes, pos));
            Ok((
                ScalarValue::IntervalYearMonth(Some(months)),
                pos + 4,
            ))
        }
        DataType::Interval(arrow_schema::IntervalUnit::DayTime) => {
            let days = i32::from_le_bytes(read_fixed::<4>(bytes, pos));
            let milliseconds = i32::from_le_bytes(read_fixed::<4>(bytes, pos + 4));
            Ok((
                ScalarValue::IntervalDayTime(Some(arrow_array::types::IntervalDayTime {
                    days,
                    milliseconds,
                })),
                pos + 8,
            ))
        }
        DataType::Interval(arrow_schema::IntervalUnit::MonthDayNano) => {
            let months = i32::from_le_bytes(read_fixed::<4>(bytes, pos));
            let days = i32::from_le_bytes(read_fixed::<4>(bytes, pos + 4));
            let nanoseconds = i64::from_le_bytes(read_fixed::<8>(bytes, pos + 8));
            Ok((
                ScalarValue::IntervalMonthDayNano(Some(
                    arrow_array::types::IntervalMonthDayNano {
                        months,
                        days,
                        nanoseconds,
                    },
                )),
                pos + 16,
            ))
        }
        DataType::Duration(TimeUnit::Second) => {
            let a = read_fixed::<8>(bytes, pos);
            Ok((ScalarValue::DurationSecond(Some(i64::from_le_bytes(a))), pos + 8))
        }
        DataType::Duration(TimeUnit::Millisecond) => {
            let a = read_fixed::<8>(bytes, pos);
            Ok((
                ScalarValue::DurationMillisecond(Some(i64::from_le_bytes(a))),
                pos + 8,
            ))
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            let a = read_fixed::<8>(bytes, pos);
            Ok((
                ScalarValue::DurationMicrosecond(Some(i64::from_le_bytes(a))),
                pos + 8,
            ))
        }
        DataType::Duration(TimeUnit::Nanosecond) => {
            let a = read_fixed::<8>(bytes, pos);
            Ok((
                ScalarValue::DurationNanosecond(Some(i64::from_le_bytes(a))),
                pos + 8,
            ))
        }
        other => Err(EngineError::CorruptMessage(format!(
            "row codec: unsupported column type {other}"
        ))),
    }
    } // unsafe block
}

/// Appends `bytes` as a `u32` length prefix + raw bytes.
fn put_len_prefixed(out: &mut Vec<u8>, bytes: &[u8]) -> Result<()> {
    let len =
        u32::try_from(bytes.len()).map_err(|_| EngineError::OutOfRange("row value > u32::MAX"))?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

/// Reads a `u32` length + payload at `pos`; returns `(payload, next_pos)`.
fn take_len_prefixed(bytes: &[u8], pos: usize) -> Result<(Vec<u8>, usize)> {
    let len_bytes = bytes
        .get(pos..pos + 4)
        .ok_or_else(|| EngineError::Corrupt("row codec: truncated length"))?;
    let len = u32::from_le_bytes(
        len_bytes
            .try_into()
            .map_err(|_| EngineError::Corrupt("row codec: truncated length"))?,
    ) as usize;
    let payload = bytes
        .get(pos + 4..pos + 4 + len)
        .ok_or_else(|| EngineError::Corrupt("row codec: truncated payload"))?;
    Ok((payload.to_vec(), pos + 4 + len))
}

/// Convert a scalar value to an array of `len` elements.
///
/// `target_type` is the declared type of the column being assigned: a
/// [`ScalarValue::Null`] broadcasts to a correctly-typed null array instead of
/// an untyped [`arrow_array::NullArray`], so `SET col = NULL` (or an expression
/// that evaluates to NULL) lands in the column without a schema mismatch.
pub(crate) fn scalar_to_array(
    sv: &ScalarValue,
    len: usize,
    target_type: &DataType,
) -> Result<Arc<dyn Array>> {
    use arrow_array::*;

    // ── Helper: broadcast a primitive scalar into a Vec ─────────────────
    macro_rules! prim {
        ($arr_ty:ident, $v:expr) => {{ Arc::new(<$arr_ty>::from(vec![$v; len])) as Arc<dyn Array> }};
    }

    let arr: Arc<dyn Array> = match sv {
        ScalarValue::Null => arrow_array::new_null_array(target_type, len),
        ScalarValue::Int8(v) => prim!(Int8Array, *v),
        ScalarValue::Int16(v) => prim!(Int16Array, *v),
        ScalarValue::Int32(v) => prim!(Int32Array, *v),
        ScalarValue::Int64(v) => prim!(Int64Array, *v),
        ScalarValue::UInt8(v) => prim!(UInt8Array, *v),
        ScalarValue::UInt16(v) => prim!(UInt16Array, *v),
        ScalarValue::UInt32(v) => prim!(UInt32Array, *v),
        ScalarValue::UInt64(v) => prim!(UInt64Array, *v),
        ScalarValue::Float32(v) => prim!(Float32Array, *v),
        ScalarValue::Float64(v) => prim!(Float64Array, *v),
        ScalarValue::Utf8(v) => Arc::new(StringArray::from(vec![v.as_deref(); len])),
        ScalarValue::LargeUtf8(v) => Arc::new(LargeStringArray::from(vec![v.as_deref(); len])),
        ScalarValue::Binary(v) => Arc::new(BinaryArray::from(vec![v.as_deref(); len])),
        ScalarValue::LargeBinary(v) => Arc::new(LargeBinaryArray::from(vec![v.as_deref(); len])),
        ScalarValue::Boolean(v) => prim!(BooleanArray, *v),
        ScalarValue::Date32(v) => prim!(Date32Array, *v),
        ScalarValue::Date64(v) => prim!(Date64Array, *v),
        ScalarValue::TimestampSecond(v, _) => prim!(TimestampSecondArray, *v),
        ScalarValue::TimestampMillisecond(v, _) => prim!(TimestampMillisecondArray, *v),
        ScalarValue::TimestampMicrosecond(v, _) => prim!(TimestampMicrosecondArray, *v),
        ScalarValue::TimestampNanosecond(v, _) => prim!(TimestampNanosecondArray, *v),
        ScalarValue::Time32Second(v) => prim!(Time32SecondArray, *v),
        ScalarValue::Time32Millisecond(v) => prim!(Time32MillisecondArray, *v),
        ScalarValue::Time64Microsecond(v) => prim!(Time64MicrosecondArray, *v),
        ScalarValue::Time64Nanosecond(v) => prim!(Time64NanosecondArray, *v),
        ScalarValue::IntervalYearMonth(v) => {
            Arc::new(IntervalYearMonthArray::from(vec![*v; len]))
        }
        ScalarValue::IntervalDayTime(v) => Arc::new(IntervalDayTimeArray::from(vec![*v; len])),
        ScalarValue::IntervalMonthDayNano(v) => {
            Arc::new(IntervalMonthDayNanoArray::from(vec![*v; len]))
        }
        ScalarValue::DurationSecond(v) => prim!(DurationSecondArray, *v),
        ScalarValue::DurationMillisecond(v) => prim!(DurationMillisecondArray, *v),
        ScalarValue::DurationMicrosecond(v) => prim!(DurationMicrosecondArray, *v),
        ScalarValue::DurationNanosecond(v) => prim!(DurationNanosecondArray, *v),
        // Every ScalarValue variant must appear exactly once. The catch-all
        // `_` arm handles any future variants we haven't enumerated.
        _ => {
            return Err(EngineError::CorruptMessage(format!(
                "scalar_to_array: unsupported ScalarValue variant {:?}",
                sv.data_type()
            )));
        }
    };
    Ok(arr)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use arrow_array::{Int64Array, StringArray};
    use arrow_schema::{Field, Schema};

    #[test]
    fn round_trips_ints_strings_and_nulls() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![42_i64])),
                Arc::new(StringArray::from(vec![Some("alice")])),
            ],
        )
        .expect("batch");
        let encoded = encode_row(&batch, 0).expect("encode");
        let decoded = decode_row(&encoded, schema.as_ref()).expect("decode");
        assert_eq!(decoded.num_rows(), 1);
        assert_eq!(decoded.num_columns(), 2);
        assert_eq!(
            decoded
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            42
        );
        assert_eq!(
            decoded
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(0),
            "alice"
        );
    }

    #[test]
    fn round_trips_null_value() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "name",
            DataType::Utf8,
            true,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(StringArray::from(vec![None::<&str>]))],
        )
        .expect("batch");
        let encoded = encode_row(&batch, 0).expect("encode");
        let decoded = decode_row(&encoded, schema.as_ref()).expect("decode");
        assert!(decoded.column(0).is_null(0));
    }
}
