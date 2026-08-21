#![allow(clippy::expect_used, clippy::unwrap_used, clippy::indexing_slicing)]

//! Metrics integration coverage — PR 0.8: a corrupt SST read during recovery
//! must bump `sst_read_failures` instead of being silently swallowed.

use bytes::Bytes;
use hatp_engine::{Engine, EngineConfig, Mutation};
use std::sync::atomic::Ordering;
use tempfile::{Builder, TempDir};

fn unique_dir(label: &str) -> Result<TempDir, std::io::Error> {
    Builder::new()
        .prefix(&format!("hatp-metrics-{label}-"))
        .tempdir()
}

#[test]
fn sst_read_failures_incremented_on_error() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("sst-fail")?;
    let sst_path;
    {
        let engine = Engine::open(EngineConfig::new(dir.path()))?;
        engine.write(&[Mutation::Put {
            key: Bytes::from_static(b"k"),
            value: Bytes::from_static(b"v"),
        }])?;
        let handle = engine.flush()?.expect("flush must produce an SST");
        sst_path = dir
            .path()
            .join(format!("sst-{:020}.vortex", handle.file_id));
    }

    // Corrupt the SST: recovery must skip it, log, and record the failure.
    std::fs::write(&sst_path, b"this is not a vortex file")?;

    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    let failures = engine.metrics().sst_read_failures.load(Ordering::Relaxed);
    // Exact assertion: exactly one SST read failure (only one SST file is corrupt)
    assert_eq!(
        failures, 1,
        "corrupt SST must increment sst_read_failures to exactly 1, got {failures}"
    );
    // Negative assertion: must not be 0 (must not silently swallow the error)
    assert_ne!(failures, 0, "sst_read_failures must not be zero for corrupt SST");
    Ok(())
}
