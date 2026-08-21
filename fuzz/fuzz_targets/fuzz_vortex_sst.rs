//! Vortex SST read_all fuzz target (VC-006)
//! Oracle: 1) no panic 2) OOM guard 3) if rows returned, keys are sorted
#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::Write;
use hatp_engine::vortex_sst;

fuzz_target!(|data: &[u8]| {
    if data.len() > 10_000_000 {
        return;
    }
    // Write fuzz data to a temp file, then try to read it as Vortex SST
    let dir = match tempfile::tempdir() {
        Ok(d) => d,
        Err(_) => return,
    };
    let path = dir.path().join("sst.vortex");
    if std::fs::write(&path, data).is_err() {
        return;
    }
    // L1: read_all must not panic
    if let Ok(rows) = vortex_sst::read_all(&path) {
        // L2: if rows returned, keys must be in ascending order
        for w in rows.windows(2) {
            assert!(w[0].key <= w[1].key, "Vortex SST rows not sorted");
        }
    }
});