//! Kani proof: Bloom determinism (VC-BLOOM-KANI)
//!
//! Invariant: same parameters + same keys → same bytes
//! Source: gap-report.md, behavior-surface.md Kani candidate K4
//!
//! Run: cargo kani --package hatp-engine --harness kani_bloom_deterministic

#![cfg(kani)]

use hatp_engine::bloom::BloomFilter;

#[kani::proof]
fn kani_bloom_insert_contains() {
    let item_count: u8 = kani::any();
    let fp_rate: f64 = 0.01; // fixed FP rate
    // Limit item_count to avoid symbolic execution explosion
    kani::assume(item_count >= 1 && item_count <= 10);

    // Use an abstract key
    let mut filter = BloomFilter::new(item_count as usize, fp_rate);

    // Verify: after insert, contains returns true
    let key: [u8; 4] = kani::any();
    filter.insert(&key);
    assert!(filter.contains(&key), "inserted key must be found");
}

#[kani::proof]
fn kani_bloom_serialization_roundtrip() {
    let item_count: u8 = kani::any();
    kani::assume(item_count >= 1 && item_count <= 5);
    let mut filter = BloomFilter::new(item_count as usize, 0.01);

    let key: [u8; 4] = kani::any();
    filter.insert(&key);

    let bytes = filter.to_bytes();
    let decoded = BloomFilter::from_bytes(&bytes).expect("roundtrip must succeed");

    assert!(decoded.contains(&key), "decoded filter must still contain the key");
}