//! Cross-crate shared Arrow ↔ [`ScalarValue`] ↔ bytes encoding utilities.
//!
//! # Why centralized here
//!
//! Previously `hatp-engine` and `hatp-frontend` each maintained near-identical
//! downcast encoding logic (`scalar_from_array` / `encode_pk` / `array_value_to_bytes` /
//! `scalar_to_index_bytes`). This module converges them into three orthogonal entry points,
//! eliminating duplication:
//!
//! - [`array_slot_to_scalar`]: Arrow array slot → [`ScalarValue`] (single implementation).
//! - [`encode_pk_value`]: `ScalarValue` → primary key order-preserving byte fragment.
//! - [`scalar_to_index_be`]: `ScalarValue` → secondary index bytes.
//!
//! # Two byte encodings are NOT interchangeable (important — historically mistaken as "duplicate" and nearly merged)
//!
//! Primary key encoding and secondary index encoding are **two semantically distinct encodings**.
//! Do not merge them again:
//!
//! 1. **Narrow encoding (primary key)**: integers/floats use their **original byte width**
//!    (`Int32` = 4 bytes, `Float32` = 4 bytes). [`encode_pk_value`] is only used during
//!    primary key construction; `delete_plan` depends on its exact byte format, and any
//!    change must remain bit-identical with `build_pk_key`.
//!
//! 2. **Index encoding (unified 8-byte width)**: small integers/floats are promoted to
//!    `i64` / `u64` / `f64` (8 bytes). This allows different declared-width numerics to
//!    compare and match within the same BTree index — e.g. an `Int32` column can be matched
//!    by an SQL `Int64` literal (`WHERE col = 5`), because `Int32(5) as i64 == Int64(5)`.
//!
//! Both guarantee **byte-order == logical-order** (order-preserving): signed integers use
//! sign-bit flipping, floats use IEEE-754 total-order transform, strings use byte order.
//! This ensures the memtable's `BTreeMap`, SST range scans, and secondary index range
//! queries all produce correct ordering.
//! (Historically, raw BE caused `-1.0 > 0.0` and negatives sorted after positives — a
//! sorting bug.)

use arrow_array::{
    BinaryArray, BooleanArray, Date32Array, Date64Array, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Int64Array, LargeBinaryArray, LargeStringArray, StringArray,
    TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, TimeUnit};
use bytes::Bytes;
use datafusion_common::ScalarValue;

/// Converts Arrow array slot at `row` into a [`ScalarValue`].
///
/// NULL slots and unrecognized types both return [`ScalarValue::Null`] (the caller
/// rejects or falls back based on this). This is the single array→scalar conversion
/// implementation shared by `hatp-engine` primary key encoding and `hatp-frontend`
/// index encoding.
#[must_use]
pub fn array_slot_to_scalar(array: &dyn arrow_array::Array, row: usize) -> ScalarValue {
    if array.is_null(row) {
        return ScalarValue::Null;
    }
    let any = array.as_any();
    // Dispatch once on the dense `DataType` tag (lowered to a jump table)
    // instead of probing ~22 `downcast_ref` type-ids in sequence. The
    // per-arm downcast still yields `Null` as a defensive fallback if the
    // physical array ever disagrees with its declared `DataType`.
    macro_rules! primitive {
        ($ty:ty, $variant:ident) => {
            match any.downcast_ref::<$ty>() {
                Some(arr) => ScalarValue::$variant(Some(arr.value(row))),
                None => ScalarValue::Null,
            }
        };
    }
    match array.data_type() {
        DataType::Int64 => primitive!(Int64Array, Int64),
        DataType::Int32 => primitive!(Int32Array, Int32),
        DataType::Int16 => primitive!(Int16Array, Int16),
        DataType::Int8 => primitive!(Int8Array, Int8),
        DataType::UInt64 => primitive!(UInt64Array, UInt64),
        DataType::UInt32 => primitive!(UInt32Array, UInt32),
        DataType::UInt16 => primitive!(UInt16Array, UInt16),
        DataType::UInt8 => primitive!(UInt8Array, UInt8),
        DataType::Float64 => primitive!(Float64Array, Float64),
        DataType::Float32 => primitive!(Float32Array, Float32),
        DataType::Utf8 => match any.downcast_ref::<StringArray>() {
            Some(arr) => ScalarValue::Utf8(Some(arr.value(row).to_string())),
            None => ScalarValue::Null,
        },
        DataType::LargeUtf8 => match any.downcast_ref::<LargeStringArray>() {
            Some(arr) => ScalarValue::LargeUtf8(Some(arr.value(row).to_string())),
            None => ScalarValue::Null,
        },
        DataType::Binary => match any.downcast_ref::<BinaryArray>() {
            Some(arr) => ScalarValue::Binary(Some(arr.value(row).to_vec())),
            None => ScalarValue::Null,
        },
        DataType::LargeBinary => match any.downcast_ref::<LargeBinaryArray>() {
            Some(arr) => ScalarValue::LargeBinary(Some(arr.value(row).to_vec())),
            None => ScalarValue::Null,
        },
        DataType::Boolean => primitive!(BooleanArray, Boolean),
        DataType::Date32 => primitive!(Date32Array, Date32),
        DataType::Date64 => primitive!(Date64Array, Date64),
        DataType::Timestamp(TimeUnit::Second, _) => {
            match any.downcast_ref::<TimestampSecondArray>() {
                Some(arr) => ScalarValue::TimestampSecond(Some(arr.value(row)), None),
                None => ScalarValue::Null,
            }
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            match any.downcast_ref::<TimestampMillisecondArray>() {
                Some(arr) => ScalarValue::TimestampMillisecond(Some(arr.value(row)), None),
                None => ScalarValue::Null,
            }
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            match any.downcast_ref::<TimestampMicrosecondArray>() {
                Some(arr) => ScalarValue::TimestampMicrosecond(Some(arr.value(row)), None),
                None => ScalarValue::Null,
            }
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            match any.downcast_ref::<TimestampNanosecondArray>() {
                Some(arr) => ScalarValue::TimestampNanosecond(Some(arr.value(row)), None),
                None => ScalarValue::Null,
            }
        }
        _ => ScalarValue::Null,
    }
}

/// Encodes a primary key scalar into an order-preserving, reversible byte fragment.
///
/// This fragment is assembled by `hatp-engine`'s `assemble_table_key` into a `\0`-delimited
/// key. To guarantee key uniqueness (two different PK sequences must never produce identical
/// bytes, which would silently overwrite data) and **byte-order == logical-order** (the
/// memtable `BTreeMap`, SST range scans all depend on this), the encoding follows these
/// invariants:
///
/// - **Fixed-width types** (integer/float/bool/date/timestamp): encoded as fixed-length
///   bytes. May contain `0x00` internally (harmless — width is fixed by column type, not
///   delimited by `0x00`).
/// - **Signed integers**: BE + sign-bit flip (`v ^ MIN`), so negative < 0 < positive.
/// - **Unsigned integers**: BE (naturally order-preserving).
/// - **Floats**: IEEE-754 total-order (negatives bitwise-inverted, positives sign-bit set),
///   fixing the "raw BE causes `-1.0 > 0.0`" sorting bug.
/// - **Strings**: FoundationDB-style escaping — each `0x00` is written as `0x00 0xFF`,
///   with a trailing `0x00` terminator. The escaped fragment contains no bare `0x00`
///   (the terminator is the only one), so the `\0` delimiter cannot be confused with
///   string content.
///
/// Historical context: the old implementation used `s.as_bytes()` directly and added a
/// `column=` prefix. String PK values containing `\0` would collide with `assemble_table_key`'s
/// `\0` delimiter, causing two different primary keys to encode to the same key and silently
/// overwrite data. This implementation fixes that root cause and no longer writes column
/// names into keys.
///
/// This is a separate encoding from [`scalar_to_index_be`]; do not mix them
/// (index encoding unifies to 8 bytes).
///
/// NULL or unsupported types (binary, decimal, list…) return `None`; callers reject the
/// write and report `NullPrimaryKey`.
#[must_use]
pub fn encode_pk_value(scalar: &ScalarValue) -> Option<Vec<u8>> {
    let bytes = match scalar {
        // ── Signed integers: sign-bit flip + BE (order-preserving) ──
        ScalarValue::Int8(Some(v)) => sign_flip_be(i64::from(*v), 1),
        ScalarValue::Int16(Some(v)) => sign_flip_be(i64::from(*v), 2),
        ScalarValue::Int32(Some(v)) => sign_flip_be(i64::from(*v), 4),
        ScalarValue::Int64(Some(v)) => sign_flip_be(*v, 8),
        // ── Unsigned integers: BE (naturally order-preserving) ──
        ScalarValue::UInt8(Some(v)) => Vec::from(&v.to_be_bytes()),
        ScalarValue::UInt16(Some(v)) => Vec::from(&v.to_be_bytes()),
        ScalarValue::UInt32(Some(v)) => Vec::from(&v.to_be_bytes()),
        ScalarValue::UInt64(Some(v)) => Vec::from(&v.to_be_bytes()),
        // ── Floats: total-order ──
        ScalarValue::Float32(Some(v)) => float_total_order_f32(*v),
        ScalarValue::Float64(Some(v)) => float_total_order_f64(*v),
        // ── Boolean: single byte (false before true) ──
        ScalarValue::Boolean(Some(v)) => vec![u8::from(*v)],
        // ── Strings: FoundationDB-style escaping ──
        ScalarValue::Utf8(Some(s)) | ScalarValue::LargeUtf8(Some(s)) => escape_bytes(s.as_bytes()),
        // ── Date / Timestamp: underlying signed integer ──
        ScalarValue::Date32(Some(v)) => sign_flip_be(i64::from(*v), 4),
        ScalarValue::Date64(Some(v)) => sign_flip_be(*v, 8),
        ScalarValue::TimestampSecond(Some(v), _) => sign_flip_be(*v, 8),
        ScalarValue::TimestampMillisecond(Some(v), _) => sign_flip_be(*v, 8),
        ScalarValue::TimestampMicrosecond(Some(v), _) => sign_flip_be(*v, 8),
        ScalarValue::TimestampNanosecond(Some(v), _) => sign_flip_be(*v, 8),
        // NULL or unsupported types (binary, decimal, list…): PK rejects these.
        _ => return None,
    };
    Some(bytes)
}

/// Signed integer → order-preserving BE: takes the low `width` bytes of `v.to_be_bytes()`
/// and flips the sign bit, so negative < 0 < positive. E.g. `Int32`: `-2^31 → 0x00000000`,
/// `-1 → 0x7FFFFFFF`, `0 → 0x80000000`, `2^31-1 → 0xFFFFFFFF`.
///
/// `width` must satisfy `1 <= width <= 8` (callers always pass 1/2/4/8).
fn sign_flip_be(v: i64, width: usize) -> Vec<u8> {
    debug_assert!(
        (1..=8).contains(&width),
        "sign_flip_be width out of range: {width}"
    );
    let flipped = (v as u64) ^ (1_u64 << (width * 8 - 1));
    let be_bytes = flipped.to_be_bytes();
    let start = be_bytes.len().saturating_sub(width);
    // SAFETY: `width` is 1..=8 (debug_assert above), `be_bytes` is 8 bytes,
    // so `start..` is always in bounds.
    #[allow(clippy::indexing_slicing)]
    {
        Vec::from(&be_bytes[start..])
    }
}

/// `f64` → IEEE-754 total-order BE: negatives bitwise-inverted, positives sign-bit set.
/// This makes `-Inf < negative < -0.0 <= +0.0 < positive < +Inf < NaN` consistent with byte order.
fn float_total_order_f64(v: f64) -> Vec<u8> {
    let bits = v.to_bits();
    let ordered = if bits & (1_u64 << 63) != 0 {
        !bits
    } else {
        bits ^ (1_u64 << 63)
    };
    Vec::from(&ordered.to_be_bytes())
}

/// `f32` → IEEE-754 total-order BE (32-bit variant of [`float_total_order_f64`]).
fn float_total_order_f32(v: f32) -> Vec<u8> {
    let bits = v.to_bits();
    let ordered = if bits & (1_u32 << 31) != 0 {
        !bits
    } else {
        bits ^ (1_u32 << 31)
    };
    Vec::from(&ordered.to_be_bytes())
}

/// String → FoundationDB-style escaped bytes: `0x00` → `0x00 0xFF`, trailing `0x00`
/// terminator. The escaped output contains no bare `0x00` (the terminator is the only one),
/// so it can be concatenated with `\0` delimiters without ambiguity.
fn escape_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len().saturating_add(1));
    for &byte in bytes {
        if byte == 0x00 {
            out.push(0x00);
            out.push(0xFF);
        } else {
            out.push(byte);
        }
    }
    out.push(0x00);
    out
}

/// Encodes a [`ScalarValue`] as secondary index bytes (unified 8-byte width, order-preserving).
///
/// This is the **second** encoding, separate from [`encode_pk_value`]: small integers/floats
/// are promoted to 8 bytes so different declared-width numerics are comparable and matchable
/// within the same index. See the module-level comment.
///
/// Like primary key encoding, this also guarantees **byte-order == logical-order** (signed
/// integer sign-bit flip, float total-order), so BTree index range queries produce correct
/// ordering.
///
/// Unsupported types fall back to their `Debug` string (preserving old behavior: index columns
/// should be rejected at DDL time; this is just a "no panic" fallback).
#[must_use]
pub fn scalar_to_index_be(scalar: &ScalarValue) -> Bytes {
    match scalar {
        // Signed integers: sign-bit flip + 8-byte BE (order-preserving)
        ScalarValue::Int64(Some(v)) => Bytes::from(sign_flip_be(*v, 8)),
        ScalarValue::Int32(Some(v)) => Bytes::from(sign_flip_be(i64::from(*v), 8)),
        ScalarValue::Int16(Some(v)) => Bytes::from(sign_flip_be(i64::from(*v), 8)),
        ScalarValue::Int8(Some(v)) => Bytes::from(sign_flip_be(i64::from(*v), 8)),
        // Unsigned integers: 8-byte BE (naturally order-preserving)
        ScalarValue::UInt64(Some(v)) => Bytes::copy_from_slice(&v.to_be_bytes()),
        ScalarValue::UInt32(Some(v)) => Bytes::copy_from_slice(&u64::from(*v).to_be_bytes()),
        ScalarValue::UInt16(Some(v)) => Bytes::copy_from_slice(&u64::from(*v).to_be_bytes()),
        ScalarValue::UInt8(Some(v)) => Bytes::copy_from_slice(&u64::from(*v).to_be_bytes()),
        // Floats: total-order 8 bytes (Float32 promoted to f64, so it matches Float64 literals)
        ScalarValue::Float64(Some(v)) => Bytes::from(float_total_order_f64(*v)),
        ScalarValue::Float32(Some(v)) => Bytes::from(float_total_order_f64(f64::from(*v))),
        ScalarValue::Utf8(Some(v)) => Bytes::copy_from_slice(v.as_bytes()),
        ScalarValue::LargeUtf8(Some(v)) => Bytes::copy_from_slice(v.as_bytes()),
        ScalarValue::Binary(Some(v)) | ScalarValue::LargeBinary(Some(v)) => {
            Bytes::copy_from_slice(v)
        }
        ScalarValue::Boolean(Some(true)) => Bytes::from_static(b"\x01"),
        ScalarValue::Boolean(Some(false)) => Bytes::from_static(b"\x00"),
        // Date/Timestamp: underlying signed integer
        ScalarValue::Date32(Some(v)) => Bytes::from(sign_flip_be(i64::from(*v), 8)),
        ScalarValue::Date64(Some(v)) => Bytes::from(sign_flip_be(*v, 8)),
        ScalarValue::TimestampSecond(Some(v), _) => Bytes::from(sign_flip_be(*v, 8)),
        ScalarValue::TimestampMillisecond(Some(v), _) => Bytes::from(sign_flip_be(*v, 8)),
        ScalarValue::TimestampMicrosecond(Some(v), _) => Bytes::from(sign_flip_be(*v, 8)),
        ScalarValue::TimestampNanosecond(Some(v), _) => Bytes::from(sign_flip_be(*v, 8)),
        _ => Bytes::from(format!("{scalar:?}").into_bytes()),
    }
}

/// FNV-1a 64-bit — deterministic, byte-order-sensitive hash shared across
/// crates (bloom filter + simulation digest).  4-way unrolled loop with
/// unsafe pointer reads to avoid bounds checks on the hot path.
#[must_use]
pub fn fnv1a(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let len = data.len();
    let ptr = data.as_ptr();
    let mut i = 0;
    // 4-way unrolled loop — process 4 bytes per iteration.
    let chunks = len / 4;
    for _ in 0..chunks {
        // SAFETY: i..i+4 is in-bounds (chunks * 4 <= len).
        unsafe {
            hash ^= u64::from(*ptr.add(i));
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            hash ^= u64::from(*ptr.add(i + 1));
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            hash ^= u64::from(*ptr.add(i + 2));
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            hash ^= u64::from(*ptr.add(i + 3));
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        i += 4;
    }
    // Tail: 0-3 remaining bytes.
    for j in i..len {
        // SAFETY: j < len.
        let byte = unsafe { *ptr.add(j) };
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// DJB2 — deterministic second independent hash (bloom double-hashing).
/// 4-way unrolled loop with unsafe pointer reads.
#[must_use]
pub fn djb2(data: &[u8]) -> u64 {
    let mut hash: u64 = 5381;
    let len = data.len();
    let ptr = data.as_ptr();
    let mut i = 0;
    let chunks = len / 4;
    for _ in 0..chunks {
        unsafe {
            hash = hash.wrapping_mul(33).wrapping_add(u64::from(*ptr.add(i)));
            hash = hash.wrapping_mul(33).wrapping_add(u64::from(*ptr.add(i + 1)));
            hash = hash.wrapping_mul(33).wrapping_add(u64::from(*ptr.add(i + 2)));
            hash = hash.wrapping_mul(33).wrapping_add(u64::from(*ptr.add(i + 3)));
        }
        i += 4;
    }
    for j in i..len {
        let byte = unsafe { *ptr.add(j) };
        hash = hash.wrapping_mul(33).wrapping_add(u64::from(byte));
    }
    hash
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use arrow_array::{BooleanArray, Date32Array, Float64Array, Int32Array, Int64Array, StringArray};

    #[test]
    fn int32_index_encoding_matches_int64_literal() {
        // Int32 column and Int64 literal (SQL default integer literal) must encode to the same
        // bytes — this is the reason the index encoding "unified 8-byte width" exists.
        let col = scalar_to_index_be(&ScalarValue::Int32(Some(5)));
        let lit = scalar_to_index_be(&ScalarValue::Int64(Some(5)));
        assert_eq!(col, lit);
        assert_eq!(col.len(), 8);
    }

    #[test]
    fn pk_encoding_preserves_original_width() {
        // PK narrow encoding preserves original width: Int32 = 4 bytes; after sign-bit flip 5 = 0x80000005.
        let part = encode_pk_value(&ScalarValue::Int32(Some(5))).expect("encodable");
        assert_eq!(part.len(), 4);
        assert_eq!(part, vec![0x80, 0x00, 0x00, 0x05]);
    }

    #[test]
    fn pk_encoding_rejects_null() {
        assert!(encode_pk_value(&ScalarValue::Null).is_none());
        assert!(encode_pk_value(&ScalarValue::Int32(None)).is_none());
    }

    #[test]
    fn pk_encoding_signed_ints_are_order_preserving() {
        // Negative < 0 < positive, byte-order must be consistent (historically raw BE placed negatives after positives).
        let neg = encode_pk_value(&ScalarValue::Int32(Some(-1))).expect("encodable");
        let zero = encode_pk_value(&ScalarValue::Int32(Some(0))).expect("encodable");
        let pos = encode_pk_value(&ScalarValue::Int32(Some(1))).expect("encodable");
        assert!(neg < zero, "negative integer byte-order error");
        assert!(zero < pos, "positive integer byte-order error");
    }

    #[test]
    fn pk_encoding_floats_are_order_preserving() {
        let neg = encode_pk_value(&ScalarValue::Float64(Some(-1.0))).expect("encodable");
        let zero = encode_pk_value(&ScalarValue::Float64(Some(0.0))).expect("encodable");
        let pos = encode_pk_value(&ScalarValue::Float64(Some(1.0))).expect("encodable");
        assert!(neg < zero, "negative float byte-order error (-1.0 should sort before 0.0)");
        assert!(zero < pos, "positive float byte-order error");
    }

    #[test]
    fn pk_encoding_strings_escape_nul_and_are_reversible() {
        // When a string PK contains NUL, escaping guarantees it differs from other strings.
        // Historically, raw `as_bytes()` would let `a\0b` collide with other NUL-containing
        // strings and the `\0` delimiter, causing two PKs to encode to the same key and silently
        // overwrite data.
        let with_nul =
            encode_pk_value(&ScalarValue::Utf8(Some("a\0b".to_string()))).expect("encodable");
        let plain = encode_pk_value(&ScalarValue::Utf8(Some("ab".to_string()))).expect("encodable");
        let empty = encode_pk_value(&ScalarValue::Utf8(Some(String::new()))).expect("encodable");
        assert_ne!(with_nul, plain, "NUL-containing string and plain string encoding conflict");
        assert_ne!(plain, empty, "non-empty and empty string encoding conflict");
        // The escaped fragment must not contain bare `0x00 0x00` (bare `0x00` only appears
        // as `0x00 0xFF` escapes or as the trailing terminator).
        for window in with_nul.windows(2) {
            assert_ne!(window, [0x00, 0x00], "string encoding contains bare NUL, breaking delimiter safety");
        }
        assert_eq!(with_nul.last(), Some(&0x00), "string encoding must end with a NUL terminator");
    }

    #[test]
    fn pk_encoding_key_uniqueness_for_multi_part() {
        // Reproduce the historical key-collision counterexample: (part1, part2) must encode
        // to different bytes than other combinations. Here we only verify distinguishability
        // of two single-part encodings; assembly is handled by engine's assemble_table_key
        // (its tests are in hatp-engine).
        let a = encode_pk_value(&ScalarValue::Utf8(Some("a\0b".to_string()))).unwrap();
        let b = encode_pk_value(&ScalarValue::Utf8(Some("a".to_string()))).unwrap();
        let c = encode_pk_value(&ScalarValue::Utf8(Some("b".to_string()))).unwrap();
        assert_ne!(a, b);
        assert_ne!(b, c);
    }

    #[test]
    fn array_slot_to_scalar_roundtrip_string() {
        let arr = StringArray::from(vec!["alice", "bob"]);
        assert_eq!(
            array_slot_to_scalar(&arr, 1),
            ScalarValue::Utf8(Some("bob".to_string()))
        );
    }

    #[test]
    fn array_slot_to_scalar_roundtrip_int64() {
        let arr = Int64Array::from(vec![42_i64, -7]);
        assert_eq!(array_slot_to_scalar(&arr, 0), ScalarValue::Int64(Some(42)));
        assert_eq!(array_slot_to_scalar(&arr, 1), ScalarValue::Int64(Some(-7)));
    }

    #[test]
    fn array_slot_to_scalar_roundtrip_float64() {
        let arr = Float64Array::from(vec![1.5_f64, 0.0]);
        assert_eq!(array_slot_to_scalar(&arr, 0), ScalarValue::Float64(Some(1.5)));
        assert_eq!(array_slot_to_scalar(&arr, 1), ScalarValue::Float64(Some(0.0)));
    }

    #[test]
    fn array_slot_to_scalar_roundtrip_boolean() {
        let arr = BooleanArray::from(vec![true, false]);
        assert_eq!(array_slot_to_scalar(&arr, 0), ScalarValue::Boolean(Some(true)));
        assert_eq!(array_slot_to_scalar(&arr, 1), ScalarValue::Boolean(Some(false)));
    }

    #[test]
    fn array_slot_to_scalar_roundtrip_date32() {
        let arr = Date32Array::from(vec![0_i32, 19000]);
        assert_eq!(array_slot_to_scalar(&arr, 0), ScalarValue::Date32(Some(0)));
        assert_eq!(array_slot_to_scalar(&arr, 1), ScalarValue::Date32(Some(19000)));
    }

    #[test]
    fn array_slot_to_scalar_null_slot_is_null() {
        let arr = Int64Array::from(vec![Some(1), None]);
        assert_eq!(array_slot_to_scalar(&arr, 1), ScalarValue::Null);
    }

    #[test]
    fn array_slot_to_scalar_reports_null_for_nullable_int() {
        let arr = Int32Array::from(vec![Some(7), None]);
        assert_eq!(array_slot_to_scalar(&arr, 0), ScalarValue::Int32(Some(7)));
        assert_eq!(array_slot_to_scalar(&arr, 1), ScalarValue::Null);
    }

    #[test]
    fn fnv1a_and_djb2_are_deterministic_and_order_sensitive() {
        // Shared hashes must be deterministic across processes (bloom + sim digest both depend on this).
        assert_eq!(fnv1a(b"abc"), fnv1a(b"abc"));
        assert_eq!(djb2(b"abc"), djb2(b"abc"));
        assert_ne!(fnv1a(b"abc"), fnv1a(b"abd"), "single-byte difference must change hash");
        assert_ne!(fnv1a(b"abc"), djb2(b"abc"), "two hash functions must be independent");
        assert_ne!(fnv1a(b""), fnv1a(b"a"), "empty input must differ from non-empty");
    }

    #[test]
    fn fnv1a_empty_input() {
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
    }

    #[test]
    fn fnv1a_single_byte() {
        let h = fnv1a(b"a");
        assert_ne!(h, 0, "single byte must not hash to zero");
        assert_ne!(h, fnv1a(b""), "must differ from empty input");
    }

    #[test]
    fn fnv1a_long_input() {
        let data = vec![0x41_u8; 65536];
        let h1 = fnv1a(&data);
        let h2 = fnv1a(&data);
        assert_eq!(h1, h2, "long input must be deterministic");
    }

    #[test]
    fn djb2_empty_input() {
        assert_eq!(djb2(b""), 5381);
    }

    #[test]
    fn djb2_single_byte() {
        let h = djb2(b"a");
        assert_ne!(h, 0, "single byte must not hash to zero");
        assert_ne!(h, djb2(b""), "must differ from empty input");
    }
}
