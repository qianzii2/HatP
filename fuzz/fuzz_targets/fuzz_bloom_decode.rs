//! Bloom decode fuzz target (VC-001)
//! Oracle layers: 1) no panic 2) assert_invariants 3) contains no panic
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Layer 1: reject oversized inputs to prevent OOM
    if data.len() > 10_000_000 {
        return;
    }
    // Layer 2: decode must not panic
    if let Some(filter) = hatp_engine::bloom::BloomFilter::from_bytes(data) {
        // Layer 3: invariants must hold
        assert!(filter.assert_invariants());
        // Layer 4: contains on test key must not panic
        let _ = filter.contains(b"test_key_12345");
        // Layer 5: roundtrip (to_bytes -> from_bytes) must be equivalent
        let bytes = filter.to_bytes();
        if let Some(decoded) = hatp_engine::bloom::BloomFilter::from_bytes(&bytes) {
            assert!(decoded.assert_invariants());
            assert_eq!(filter, decoded, "roundtrip mismatch");
        }
    }
});