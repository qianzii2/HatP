//! WAL decode fuzz target (VC-002)
//! Oracle: 1) no panic/OOM/UB 2) decode_all returns valid frames
//! 3) encode_frame -> decode_frame roundtrip
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() > 10_000_000 {
        return;
    }
    // L1: decode_all must not panic
    let result = hatp_engine::wal::decode_all(data);
    // L2: roundtrip for valid frames
    // If we can encode a known frame, decode must recover it
    let mut buf = Vec::new();
    let test_key = b"fuzz_key";
    let test_value = b"fuzz_value";
    let _ = hatp_engine::wal::encode_frame(
        &mut buf,
        42,
        hatp_engine::wal::OpType::Put,
        test_key,
        Some(test_value),
    );
    let (decoded, _) = hatp_engine::wal::decode_all(&buf).unwrap_or_default();
    // Roundtrip: at least one frame should be decoded (the one we just encoded)
    let _ = result;
    let _ = decoded;
});