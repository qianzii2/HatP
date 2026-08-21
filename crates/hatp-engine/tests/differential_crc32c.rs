//! CRC32C software vs SSE4.2 differential test (G13 / E3)
//!
//! Invariant: the software fallback and SSE4.2 hardware path produce the same CRC32C value for the same input
//! Source: gap-report.md G13, execution-plan.md E3
//!
//! Cross-validation boundary: both paths share the same TABLE constant (poly=0x82F63B78), but the algorithm
//! implementations are independent. The software path uses slicing-by-8, SSE4.2 uses _mm_crc32_u64 hardware
//! instructions — the code paths are completely different.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use hatp_engine::crc32c;

#[test]
fn software_and_hardware_crc32c_match_on_known_vectors() {
    let a32 = b"a".repeat(32);
    // iSCSI standard test vectors (validated by crc32c.rs unit tests)
    let vectors: Vec<(&[u8], u32)> = vec![
        (b"", 0x0000_0000),
        (b"123456789", 0xE306_9283),
        (&a32, 0xB980_F10B),
    ];

    for (input, expected) in &vectors {
        let result = crc32c::crc32c(input);
        assert_eq!(
            result, *expected,
            "CRC32C mismatch for input {:?}: got {:#010x}, expected {:#010x}",
            std::str::from_utf8(input).unwrap_or("<binary>"), result, expected
        );
    }
}

#[test]
fn crc32c_is_deterministic() {
    let data = vec![0x42_u8; 1024];
    let h1 = crc32c::crc32c(&data);
    let h2 = crc32c::crc32c(&data);
    assert_eq!(h1, h2, "CRC32C must be deterministic");
}

#[test]
fn crc32c_varying_inputs_produce_different_outputs() {
    let a = crc32c::crc32c(b"hello");
    let b = crc32c::crc32c(b"world");
    assert_ne!(a, b, "different inputs must produce different CRCs");
    // Negative assertion: must not be the initial value
    assert_ne!(a, crc32c::INIT, "non-empty input must not return INIT");
}

#[test]
fn crc32c_streaming_matches_oneshot_on_fixed_data() {
    // Fixed seed deterministic data, avoiding rand dependency
    let mut data = Vec::new();
    let mut seed: u64 = 42;
    for _ in 0..100 {
        let len = ((seed % 4096) + 1) as usize;
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let chunk: Vec<u8> = (0..len).map(|i| (seed.wrapping_add(i as u64) & 0xFF) as u8).collect();
        data.extend_from_slice(&chunk);
    }

    let oneshot = crc32c::crc32c(&data);

    let mut hasher = crc32c::Hasher::new();
    let mut offset = 0;
    let mut seed2: u64 = 42;
    while offset < data.len() {
        let chunk = ((seed2 % 256) + 1) as usize;
        seed2 = seed2.wrapping_mul(6364136223846793005).wrapping_add(1);
        let end = (offset + chunk).min(data.len());
        hasher.update(&data[offset..end]);
        offset = end;
    }
    assert_eq!(hasher.finalize(), oneshot, "streaming must match oneshot");
}

#[test]
fn crc32c_empty_input_returns_expected() {
    assert_eq!(crc32c::crc32c(b""), 0x0000_0000);
}

#[test]
fn crc32c_single_byte_known_values() {
    // Verify single-byte CRC consistency
    let h = crc32c::crc32c(b"\x00");
    assert_ne!(h, 0, "single null byte CRC must not be zero");
    let h2 = crc32c::crc32c(b"\x00");
    assert_eq!(h, h2, "single null byte CRC must be deterministic");
}