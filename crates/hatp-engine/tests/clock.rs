#![allow(clippy::expect_used, clippy::unwrap_used, clippy::indexing_slicing)]

//! `Clock` trait injection integration coverage.
//!
//! Verifies that the engine's write path (flush / compact) actually calls the injected clock
//! when stamping SSTs with `created_at` wall-clock timestamps, rather than silently calling
//! the real system time — this is the prerequisite for deterministic simulation (planned PR 0.1).

use bytes::Bytes;
use hatp_engine::manifest::Manifest;
use hatp_engine::{Clock, Engine, EngineConfig, ManualClock, Mutation};
use std::sync::Arc;
use tempfile::{Builder, TempDir};

fn unique_dir(label: &str) -> Result<TempDir, std::io::Error> {
    Builder::new()
        .prefix(&format!("hatp-clock-{label}-"))
        .tempdir()
}

fn put(key: &[u8], value: &[u8]) -> Mutation {
    Mutation::Put {
        key: Bytes::copy_from_slice(key),
        value: Bytes::copy_from_slice(value),
    }
}

/// Reads back each file's `created_at` from the on-disk MANIFEST, sorted by `file_id`.
fn manifest_created_ats(engine: &Engine) -> Vec<(u64, u64)> {
    let manifest = Manifest::open(engine.path().join("MANIFEST"))
        .expect("open manifest for created_at inspection");
    let version_set = manifest.version_set();
    let mut pairs: Vec<(u64, u64)> = engine
        .sst_file_ids()
        .iter()
        .map(|file_id| {
            let (_, _, _, created_at) = version_set
                .file_metadata(*file_id)
                .expect("flush/compact must persist file metadata");
            (*file_id, *created_at)
        })
        .collect();
    pairs.sort_unstable();
    pairs
}

#[test]
fn flush_uses_injected_clock_for_created_at() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("flush")?;
    let clock = Arc::new(ManualClock::new(1_000_000));
    let config = EngineConfig::new(dir.path()).with_clock(Arc::clone(&clock) as Arc<dyn Clock>);
    let engine = Engine::open(config)?;

    engine.write(&[put(b"k", b"v")])?;
    let handle = engine
        .flush()?
        .ok_or_else(|| std::io::Error::other("flush produced no SST"))?;

    let created_ats = manifest_created_ats(&engine);
    assert_eq!(created_ats, vec![(handle.file_id, 1_000_000)]);
    Ok(())
}

#[test]
fn compact_uses_injected_clock_for_created_at() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("compact")?;
    let clock = Arc::new(ManualClock::new(100));
    let config = EngineConfig::new(dir.path()).with_clock(Arc::clone(&clock) as Arc<dyn Clock>);
    let engine = Engine::open(config)?;

    // Two flushes produce two L0 SSTs, both with created_at = 100.
    engine.write(&[put(b"a", b"1")])?;
    engine.flush()?.expect("first flush");
    engine.write(&[put(b"b", b"2")])?;
    engine.flush()?.expect("second flush");

    // Advance clock, trigger compaction — the new SST's created_at must be 200.
    clock.set(200);
    let compacted = engine
        .compact(0, 1)?
        .expect("two L0 files must compact into one L1 file");

    let created_ats = manifest_created_ats(&engine);
    // The old L0 files have been deleted; only the compaction output L1 file remains.
    assert_eq!(created_ats, vec![(compacted.file_id, 200)]);
    Ok(())
}

#[test]
fn default_config_uses_system_clock() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("default")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    engine.write(&[put(b"k", b"v")])?;
    engine.flush()?.expect("flush");

    for (_, created_at) in manifest_created_ats(&engine) {
        // The real system clock must give a time after 2020-01-01 (1577836800).
        assert!(
            created_at > 1_577_836_800,
            "unexpected created_at {created_at}"
        );
        // Negative assertion: must not be 0 (ManualClock default), must be real time
        assert_ne!(created_at, 0, "SystemClock must not return 0");
        // Must not be a ridiculous future time (e.g. year 2100 = 4102444800)
        assert!(created_at < 4_102_444_800, "created_at too far in the future: {created_at}");
    }
    Ok(())
}
