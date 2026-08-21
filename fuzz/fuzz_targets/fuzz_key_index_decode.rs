//! KeyIndex decode fuzz target (VC-003)
//! Oracle layers: 1) no panic 2) assert_invariants 3) get no panic
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() > 10_000_000 {
        return;
    }
    if let Some(index) = hatp_engine::key_index::KeyIndex::from_bytes(data) {
        assert!(index.assert_invariants());
        let _ = index.get(b"test_key");
    }
});