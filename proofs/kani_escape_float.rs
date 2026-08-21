//! Kani proof: escape_bytes produces no bare NUL (VC-PK-KANI supplement)
//! Invariant: ∀ s: escape_bytes(s) contains no bare 0x00 0x00 sequence internally
#![cfg(kani)]
use hatp_types::codec::encode_pk_value;
use datafusion_common::ScalarValue;

#[kani::proof]
fn kani_escape_bytes_no_bare_nul() {
    let len: u8 = kani::any();
    kani::assume(len <= 32);
    let s: String = (0..len).map(|_| if kani::any::<bool>() { 'a' } else { '\0' }).collect();

    let encoded = encode_pk_value(&ScalarValue::Utf8(Some(s)));
    if let Some(bytes) = encoded {
        for w in bytes.windows(2) {
            assert!(w != [0x00, 0x00], "escape_bytes: bare NUL-NUL sequence found");
        }
        assert_eq!(bytes.last(), Some(&0x00), "escape_bytes: must end with NUL terminator");
    }
}

#[kani::proof]
fn kani_float_total_order_preserves() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    // Exclude NaN (ScalarValue PartialOrd makes no guarantees for NaN)
    kani::assume(!a.is_nan() && !b.is_nan() && a < b);

    let ea = encode_pk_value(&ScalarValue::Float64(Some(a)));
    let eb = encode_pk_value(&ScalarValue::Float64(Some(b)));
    match (ea, eb) {
        (Some(enc_a), Some(enc_b)) => {
            assert!(enc_a < enc_b, "float_total_order: {a} < {b} but encoded not less");
        }
        _ => {}
    }
}