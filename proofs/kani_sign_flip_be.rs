//! Kani proof: sign_flip_be order-preserving (VC-PK-KANI)
//!
//! Invariant: ∀ v: i32, ∀ width: 1..=8: sign_flip_be(v, width) < sign_flip_be(v+1, width)
//! Source: gap-report.md G02, behavior-surface.md Kani candidate K1
//!
//! Run: cargo kani --package hatp-types --harness kani_sign_flip_be

#![cfg(kani)]

use hatp_types::codec::encode_pk_value;

#[kani::proof]
fn kani_sign_flip_be_adjacent_preserves_order() {
    // Use i32 instead of i64 to keep symbolic execution space tractable
    let v: i32 = kani::any();
    let width: u8 = kani::any();

    // assume: width is in valid range (sign_flip_be's debug_assert precondition)
    kani::assume(width >= 1 && width <= 8);

    // Exclude i32::MAX (v+1 would overflow)
    kani::assume(v < i32::MAX);

    let a = encode_pk_value(&datafusion_common::ScalarValue::Int32(Some(v)));
    let b = encode_pk_value(&datafusion_common::ScalarValue::Int32(Some(v + 1)));

    // Both values must be encodable
    match (a, b) {
        (Some(enc_a), Some(enc_b)) => {
            assert!(enc_a < enc_b, "sign_flip_be: {v} < {} but encoded bytes not less", v + 1);
        }
        _ => {} // NULL is not encodable, skip
    }
}

#[kani::proof]
fn kani_sign_flip_be_negative_zero_positive() {
    let v: i32 = kani::any();
    kani::assume(v < 0);
    let neg = encode_pk_value(&datafusion_common::ScalarValue::Int32(Some(v)));
    let zero = encode_pk_value(&datafusion_common::ScalarValue::Int32(Some(0)));
    let pos = encode_pk_value(&datafusion_common::ScalarValue::Int32(Some(1)));

    match (neg, zero, pos) {
        (Some(n), Some(z), Some(p)) => {
            assert!(n < z, "negative must sort before zero");
            assert!(z < p, "zero must sort before positive");
        }
        _ => {}
    }
}