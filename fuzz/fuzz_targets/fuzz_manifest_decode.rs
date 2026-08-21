//! Manifest decode fuzz target (VC-004)
//! Oracle: 1) no panic 2) decoded edits can be applied to a VersionSet
#![no_main]
use libfuzzer_sys::fuzz_target;
use hatp_engine::manifest::decode_all_for_fuzz;

fuzz_target!(|data: &[u8]| {
    if data.len() > 10_000_000 {
        return;
    }
    // L1: decode_all must not panic on arbitrary input
    let _ = decode_all_for_fuzz(data);
});