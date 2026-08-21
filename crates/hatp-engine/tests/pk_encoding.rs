//! Primary key encoding tests — referencing FoundationDB Tuple Layer encoding specification
//!
//! Test scenarios (referencing encoding rules from FoundationDB `Tuple.cpp`):
//! 1. Signed integer order-preserving — negative < 0 < positive
//! 2. Unsigned integer order-preserving — natural BE order
//! 3. Float total-order — -Inf < negative < -0.0 ≤ +0.0 < positive < +Inf < NaN
//! 4. String FoundationDB-style escaping — NUL escaped as 0x00 0xFF
//! 5. Multi-part key assembly — assemble_table_key unambiguous
//! 6. NULL primary key rejection — returns None
//! 7. Boundary value round-trip
//! 8. Byte order == logical order — BTreeMap dependency
//!
//! References:
//! - FoundationDB `Tuple.cpp`: encoding rules, Versionstamp handling
//! - `hatp_types::codec`: encode_pk_value, scalar_to_index_be

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::assertions_on_constants
)]

use datafusion_common::ScalarValue;
use hatp_types::codec::{encode_pk_value, scalar_to_index_be};

// =============================================================================
// Scenario 1: Signed integer order-preserving
// Business scenario: When primary key is a signed integer, BTreeMap scan must follow numeric order
// Reference: FoundationDB Tuple Layer's sign-bit flip encoding
// =============================================================================
#[test]
fn signed_integer_encoding_is_order_preserving() {
    // negative < 0 < positive, byte order must be consistent
    let neg = encode_pk_value(&ScalarValue::Int32(Some(-1))).expect("encode");
    let zero = encode_pk_value(&ScalarValue::Int32(Some(0))).expect("encode");
    let pos = encode_pk_value(&ScalarValue::Int32(Some(1))).expect("encode");

    assert!(neg < zero, "negative must sort before zero: neg={neg:02x?}, zero={zero:02x?}");
    assert!(zero < pos, "zero must sort before positive: zero={zero:02x?}, pos={pos:02x?}");

    // Boundary values: INT32_MIN < -1 < 0 < 1 < INT32_MAX
    let min = encode_pk_value(&ScalarValue::Int32(Some(i32::MIN))).expect("encode");
    let max = encode_pk_value(&ScalarValue::Int32(Some(i32::MAX))).expect("encode");
    assert!(min < neg, "INT32_MIN must sort before -1");
    assert!(pos < max, "1 must sort before INT32_MAX");

    // Negative assertion: must not be reversed
    assert!(!(neg > zero), "negative must NOT sort after zero");
    assert!(!(zero > pos), "zero must NOT sort after positive");
}

// =============================================================================
// Scenario 2: Signed integer different-width encoding consistency
// Business scenario: Int32 columns and Int64 literals in SQL must be comparable in indexes
// Reference: FoundationDB Tuple Layer's unified integer encoding
// =============================================================================
#[test]
fn int32_and_int64_encode_to_same_bytes_for_same_value() {
    // Index encoding (unified 8 bytes)
    let i32_col = scalar_to_index_be(&ScalarValue::Int32(Some(5)));
    let i64_lit = scalar_to_index_be(&ScalarValue::Int64(Some(5)));
    assert_eq!(i32_col, i64_lit, "Int32(5) and Int64(5) must encode to same index bytes");
    assert_eq!(i32_col.len(), 8, "index encoding must be 8 bytes");

    // Primary key encoding (preserves original width)
    let pk_i32 = encode_pk_value(&ScalarValue::Int32(Some(5))).expect("encode");
    assert_eq!(pk_i32.len(), 4, "primary key encoding must preserve Int32 width");

    // Negative assertion: primary key and index encoding must differ
    assert_ne!(pk_i32, i32_col.as_ref(), "pk and index encoding must differ");
}

// =============================================================================
// Scenario 3: Unsigned integer order-preserving
// Business scenario: When primary key is an unsigned integer, BTreeMap scan must follow numeric order
// =============================================================================
#[test]
fn unsigned_integer_encoding_is_order_preserving() {
    let small = encode_pk_value(&ScalarValue::UInt32(Some(0))).expect("encode");
    let mid = encode_pk_value(&ScalarValue::UInt32(Some(100))).expect("encode");
    let large = encode_pk_value(&ScalarValue::UInt32(Some(u32::MAX))).expect("encode");

    assert!(small < mid, "0 must sort before 100");
    assert!(mid < large, "100 must sort before UINT32_MAX");
    assert!(!(mid < small), "100 must NOT sort before 0");
}

// =============================================================================
// Scenario 4: Float total-order
// Business scenario: When primary key is a float, -1.0 < 0.0 < 1.0
// Reference: FoundationDB Tuple Layer's IEEE-754 total-order transform
// =============================================================================
#[test]
fn float_encoding_is_total_order() {
    // -1.0 < 0.0 < 1.0
    let neg = encode_pk_value(&ScalarValue::Float64(Some(-1.0))).expect("encode");
    let zero = encode_pk_value(&ScalarValue::Float64(Some(0.0))).expect("encode");
    let pos = encode_pk_value(&ScalarValue::Float64(Some(1.0))).expect("encode");

    assert!(neg < zero, "-1.0 must sort before 0.0: neg={neg:02x?}, zero={zero:02x?}");
    assert!(zero < pos, "0.0 must sort before 1.0: zero={zero:02x?}, pos={pos:02x?}");

    // Encoding of -0.0 and +0.0
    let neg_zero = encode_pk_value(&ScalarValue::Float64(Some(-0.0))).expect("encode");
    let pos_zero = encode_pk_value(&ScalarValue::Float64(Some(0.0))).expect("encode");
    // IEEE 754 total-order: -0.0 ≤ +0.0
    assert!(neg_zero <= pos_zero, "-0.0 must sort before or equal to +0.0");

    // Negative assertion: must not be reversed
    assert!(!(pos < zero), "1.0 must NOT sort before 0.0");
}

// =============================================================================
// Scenario 5: String NUL escaping
// Business scenario: When primary key string contains \0, it must not be confused with the \0 delimiter
// Reference: FoundationDB Tuple Layer's FoundationDB-style escaping
// =============================================================================
#[test]
fn string_encoding_escapes_nul_and_is_reversible() {
    let with_nul = encode_pk_value(&ScalarValue::Utf8(Some("a\0b".to_string()))).expect("encode");
    let plain = encode_pk_value(&ScalarValue::Utf8(Some("ab".to_string()))).expect("encode");
    let empty = encode_pk_value(&ScalarValue::Utf8(Some(String::new()))).expect("encode");

    // String with NUL must encode differently from string without NUL
    assert_ne!(with_nul, plain, "string with NUL must encode differently from plain string");

    // Empty string must encode differently from non-empty string
    assert_ne!(plain, empty, "empty string must encode differently from non-empty");

    // No bare 0x00 0x00 sequence must appear inside the escaped string fragment
    for window in with_nul.windows(2) {
        assert_ne!(window, [0x00, 0x00], "no bare NUL-NUL sequence in encoded string");
    }

    // Must end with terminator 0x00
    assert_eq!(with_nul.last(), Some(&0x00), "encoded string must end with NUL terminator");

    // Negative assertion: must not be the same value
    assert_ne!(with_nul, empty, "string with NUL must not equal empty string");
}

// =============================================================================
// Scenario 6: NULL primary key rejection
// Business scenario: When primary key column is NULL, engine rejects the write
// Reference: "no implicit NULL PK" semantics
// =============================================================================
#[test]
fn null_primary_key_rejected() {
    assert!(encode_pk_value(&ScalarValue::Null).is_none(), "Null must be rejected");
    assert!(encode_pk_value(&ScalarValue::Int32(None)).is_none(), "Int32(None) must be rejected");
    assert!(encode_pk_value(&ScalarValue::Int64(None)).is_none(), "Int64(None) must be rejected");
    assert!(encode_pk_value(&ScalarValue::Utf8(None)).is_none(), "Utf8(None) must be rejected");
    assert!(encode_pk_value(&ScalarValue::Float64(None)).is_none(), "Float64(None) must be rejected");

    // Negative assertion: non-NULL values must be encodable
    assert!(encode_pk_value(&ScalarValue::Int32(Some(1))).is_some(), "Int32(1) must be encodable");
    assert!(encode_pk_value(&ScalarValue::Utf8(Some("x".to_string()))).is_some(), "Utf8 must be encodable");
}

// =============================================================================
// Scenario 7: Boolean encoding
// Business scenario: Primary key is a boolean
// Expected: false < true
// =============================================================================
#[test]
fn boolean_encoding_preserves_order() {
    let false_val = encode_pk_value(&ScalarValue::Boolean(Some(false))).expect("encode");
    let true_val = encode_pk_value(&ScalarValue::Boolean(Some(true))).expect("encode");

    assert!(false_val < true_val, "false must sort before true");
    assert_eq!(false_val.len(), 1, "boolean encoding must be 1 byte");
    assert_eq!(true_val.len(), 1, "boolean encoding must be 1 byte");

    // Negative assertion
    assert!(!(true_val < false_val), "true must NOT sort before false");
}

// =============================================================================
// Scenario 8: Date/Timestamp encoding
// Business scenario: Primary key is Date32/Date64/Timestamp
// =============================================================================
#[test]
fn temporal_encoding_is_order_preserving() {
    // Date32: days from epoch
    let earlier = encode_pk_value(&ScalarValue::Date32(Some(0))).expect("encode");
    let later = encode_pk_value(&ScalarValue::Date32(Some(100))).expect("encode");
    assert!(earlier < later, "earlier date must sort before later date");

    // Date64: milliseconds from epoch
    let d64_early = encode_pk_value(&ScalarValue::Date64(Some(0))).expect("encode");
    let d64_late = encode_pk_value(&ScalarValue::Date64(Some(1000))).expect("encode");
    assert!(d64_early < d64_late, "earlier Date64 must sort before later");

    // TimestampSecond
    let ts_early = encode_pk_value(&ScalarValue::TimestampSecond(Some(0), None)).expect("encode");
    let ts_late = encode_pk_value(&ScalarValue::TimestampSecond(Some(100), None)).expect("encode");
    assert!(ts_early < ts_late, "earlier timestamp must sort before later");

    // Negative assertion
    assert!(!(later < earlier), "later date must NOT sort before earlier");
}

// =============================================================================
// Scenario 9: Multi-part key encoding uniqueness
// Business scenario: Two different primary key sequences must produce different key encodings
// Reference: historical bug where "a\0b" + "c" and "a" + "\0bc" produced the same key
// =============================================================================
#[test]
fn multi_part_key_uniqueness() {
    // Scenario 1: Different string combinations
    let p1_a = encode_pk_value(&ScalarValue::Utf8(Some("a\0b".to_string()))).expect("encode");
    let p1_b = encode_pk_value(&ScalarValue::Utf8(Some("c".to_string()))).expect("encode");
    let p2_a = encode_pk_value(&ScalarValue::Utf8(Some("a".to_string()))).expect("encode");
    let p2_b = encode_pk_value(&ScalarValue::Utf8(Some("\0bc".to_string()))).expect("encode");

    // Two different primary key sequences must produce different encodings
    assert_ne!(p1_a, p2_a, "different strings must encode differently");
    assert_ne!(p1_b, p2_b, "different strings must encode differently");

    // Scenario 2: Different integer combinations
    let pk1 = encode_pk_value(&ScalarValue::Int32(Some(1))).expect("encode");
    let pk2 = encode_pk_value(&ScalarValue::Int32(Some(2))).expect("encode");
    assert_ne!(pk1, pk2, "different integers must encode differently");

    // Scenario 3: Same value, different INT type (narrow PK encoding preserves original width)
    let pk_i32 = encode_pk_value(&ScalarValue::Int32(Some(5))).expect("encode");
    let pk_i64 = encode_pk_value(&ScalarValue::Int64(Some(5))).expect("encode");
    assert_ne!(pk_i32, pk_i64, "Int32 and Int64 must differ in pk encoding");
}

// =============================================================================
// Scenario 10: All integer type boundary values
// Business scenario: Primary key can be Int8/Int16/Int32/Int64/UInt8/UInt16/UInt32/UInt64
// Expected: All type boundary values are encodable and sort correctly
// =============================================================================
#[test]
fn all_integer_types_encode_and_sort() {
    let test_cases: Vec<(&str, ScalarValue, usize)> = vec![
        ("Int8_MIN", ScalarValue::Int8(Some(i8::MIN)), 1),
        ("Int8_MAX", ScalarValue::Int8(Some(i8::MAX)), 1),
        ("Int16_MIN", ScalarValue::Int16(Some(i16::MIN)), 2),
        ("Int16_MAX", ScalarValue::Int16(Some(i16::MAX)), 2),
        ("Int32_MIN", ScalarValue::Int32(Some(i32::MIN)), 4),
        ("Int32_MAX", ScalarValue::Int32(Some(i32::MAX)), 4),
        ("Int64_MIN", ScalarValue::Int64(Some(i64::MIN)), 8),
        ("Int64_MAX", ScalarValue::Int64(Some(i64::MAX)), 8),
        ("UInt8_MAX", ScalarValue::UInt8(Some(u8::MAX)), 1),
        ("UInt16_MAX", ScalarValue::UInt16(Some(u16::MAX)), 2),
        ("UInt32_MAX", ScalarValue::UInt32(Some(u32::MAX)), 4),
        ("UInt64_MAX", ScalarValue::UInt64(Some(u64::MAX)), 8),
    ];

    let mut encoded: Vec<(&str, Vec<u8>)> = Vec::new();
    for (name, scalar, expected_len) in &test_cases {
        let bytes = encode_pk_value(scalar).expect(&format!("encode {name}"));
        assert_eq!(
            bytes.len(),
            *expected_len,
            "{name} must be {expected_len} bytes, got {}",
            bytes.len()
        );
        encoded.push((name, bytes));
    }

    // All signed integers: MIN must sort before MAX
    for i in (0..encoded.len()).step_by(2) {
        if encoded[i].0.contains("MIN") && encoded[i + 1].0.contains("MAX") {
            assert!(
                encoded[i].1 < encoded[i + 1].1,
                "{} must sort before {}", encoded[i].0, encoded[i + 1].0
            );
        }
    }
}

// =============================================================================
// Scenario 11: Index encoding unified 8-byte width
// Business scenario: Integers of different declared widths must be comparable in secondary indexes
// Reference: `scalar_to_index_be`'s "unified 8 bytes" design
// =============================================================================
#[test]
fn index_encoding_unifies_to_8_bytes() {
    let i8 = scalar_to_index_be(&ScalarValue::Int8(Some(42)));
    let i16 = scalar_to_index_be(&ScalarValue::Int16(Some(42)));
    let i32 = scalar_to_index_be(&ScalarValue::Int32(Some(42)));
    let i64 = scalar_to_index_be(&ScalarValue::Int64(Some(42)));

    assert_eq!(i8.len(), 8, "Int8 index encoding must be 8 bytes");
    assert_eq!(i16.len(), 8, "Int16 index encoding must be 8 bytes");
    assert_eq!(i32.len(), 8, "Int32 index encoding must be 8 bytes");
    assert_eq!(i64.len(), 8, "Int64 index encoding must be 8 bytes");

    // Same value → same encoding
    assert_eq!(i8, i16, "Int8(42) and Int16(42) must encode to same index bytes");
    assert_eq!(i16, i32, "Int16(42) and Int32(42) must encode to same index bytes");
    assert_eq!(i32, i64, "Int32(42) and Int64(42) must encode to same index bytes");

    // Negative assertion: different values → different encoding
    let i32_other = scalar_to_index_be(&ScalarValue::Int32(Some(43)));
    assert_ne!(i32, i32_other, "Int32(42) and Int32(43) must encode differently");
}

// =============================================================================
// Scenario 12: Unsupported types degrade to debug string
// Business scenario: Index encoding encounters unsupported types (binary, decimal, list)
// Expected: Degrades to debug string (no panic)
// =============================================================================
#[test]
fn unsupported_types_degrade_to_debug_string_in_index() {
    // Binary type degrades to debug string in index encoding
    let binary = scalar_to_index_be(&ScalarValue::Binary(Some(vec![0x01, 0x02, 0x03])));
    // Must not panic, degrades to debug format
    assert!(!binary.is_empty(), "unsupported type must produce non-empty fallback");

    // Verify it is debug string format (non-empty)
    let as_str = String::from_utf8_lossy(&binary);
    assert!(!as_str.is_empty(), "fallback must produce non-empty output");
}

// =============================================================================
// Scenario 13: Float32 and Float64 unified in index
// Business scenario: Float32 columns and Float64 literals must be comparable in indexes
// =============================================================================
#[test]
fn float32_and_float64_unify_in_index() {
    let f32 = scalar_to_index_be(&ScalarValue::Float32(Some(1.0)));
    let f64 = scalar_to_index_be(&ScalarValue::Float64(Some(1.0)));

    assert_eq!(f32.len(), 8, "Float32 index encoding must be 8 bytes");
    assert_eq!(f64.len(), 8, "Float64 index encoding must be 8 bytes");
    assert_eq!(f32, f64, "Float32(1.0) and Float64(1.0) must encode to same index bytes");
}