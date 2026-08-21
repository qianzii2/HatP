#![allow(clippy::expect_used, clippy::unwrap_used, clippy::indexing_slicing)]

//! Compaction integration coverage — the PR 0.3 "compaction GCs superseded
//! versions" fix and its watermark boundary behaviour.

use bytes::Bytes;
use hatp_engine::{Engine, EngineConfig, Mutation};
use tempfile::{Builder, TempDir};

fn unique_dir(label: &str) -> Result<TempDir, std::io::Error> {
    Builder::new()
        .prefix(&format!("hatp-compact-{label}-"))
        .tempdir()
}

fn put(key: &[u8], value: &[u8]) -> Mutation {
    Mutation::Put {
        key: Bytes::copy_from_slice(key),
        value: Bytes::copy_from_slice(value),
    }
}

/// Write v1, v2, v3 for the same key, each in its own flush (the write →
/// flush pattern that freezes `end_ts = OPEN_ENDED_TS` in each SST).
fn seed_three_versions(engine: &Engine) -> Result<(), Box<dyn std::error::Error>> {
    engine.write(&[put(b"k", b"v1")])?;
    engine
        .flush()?
        .ok_or_else(|| std::io::Error::other("flush v1 produced no SST"))?;
    engine.write(&[put(b"k", b"v2")])?;
    engine
        .flush()?
        .ok_or_else(|| std::io::Error::other("flush v2 produced no SST"))?;
    engine.write(&[put(b"k", b"v3")])?;
    engine
        .flush()?
        .ok_or_else(|| std::io::Error::other("flush v3 produced no SST"))?;
    Ok(())
}

#[test]
fn compact_dedups_superseded_versions() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("dedup")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    seed_three_versions(&engine)?;

    // No pinned snapshot: watermark == snapshot_ts() (== 3, v3's begin_ts).
    // v2's recomputed end_ts is 3 and v1's is 2, both <= watermark, so only v3
    // survives.
    let handle = engine
        .compact(0, 1)?
        .ok_or_else(|| std::io::Error::other("three L0 files must compact"))?;
    assert_eq!(handle.rows, 1, "superseded versions must be dropped");
    assert_eq!(
        engine.get(b"k", engine.snapshot_ts())?.as_deref(),
        Some(b"v3".as_ref())
    );
    Ok(())
}

#[test]
fn compact_preserves_versions_above_watermark() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("watermark")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    seed_three_versions(&engine)?;

    // A live reader pins snapshot = 2 (v2's begin_ts). v2's recomputed end_ts
    // (3) is still > watermark (2), so v2 must survive; v1 (end_ts 2) is dropped.
    let guard = engine.pin_snapshot(2);
    let handle = engine
        .compact(0, 1)?
        .ok_or_else(|| std::io::Error::other("three L0 files must compact"))?;
    assert_eq!(handle.rows, 2, "v2 must survive for the pinned snapshot");
    // The pinned snapshot still sees v2.
    assert_eq!(engine.get(b"k", 2)?.as_deref(), Some(b"v2".as_ref()));
    // Releasing the pin lets a later compaction reclaim v2.
    drop(guard);
    Ok(())
}

#[test]
fn compact_preserves_latest_at_watermark() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("latest-at-watermark")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    seed_three_versions(&engine)?;

    // Boundary: watermark equals the newest version's begin_ts (3). The newest
    // version must still be kept (it is the current value), even though an
    // older version with `end_ts <= watermark` would be dropped.
    let guard = engine.pin_snapshot(3);
    let handle = engine
        .compact(0, 1)?
        .ok_or_else(|| std::io::Error::other("three L0 files must compact"))?;
    assert_eq!(
        handle.rows, 1,
        "only the newest version survives at the boundary"
    );
    assert_eq!(engine.get(b"k", 3)?.as_deref(), Some(b"v3".as_ref()));
    drop(guard);
    Ok(())
}

/// SC-01: compaction boundary — single file input (fewer than 2 does not trigger merge)
#[test]
fn compact_single_file_returns_none() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("single-file")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    engine.write(&[put(b"k", b"v")])?;
    engine.flush()?.expect("flush");

    // Positive assertion: single file input does not trigger compaction
    let result = engine.compact(0, 1)?;
    assert!(result.is_none(), "single file must not trigger compaction");

    // Negative assertion: data is still readable
    assert_eq!(
        engine.get(b"k", engine.snapshot_ts())?.as_deref(),
        Some(b"v".as_ref())
    );
    Ok(())
}

/// SC-01: compaction boundary — large value data integrity after merge
#[test]
fn compact_large_value_preserves_data() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("large-val")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    let large = vec![0x42_u8; 65536]; // 64KB
    engine.write(&[put(b"k1", &large)])?;
    engine.flush()?.expect("flush 1");
    engine.write(&[put(b"k2", b"small")])?;
    engine.flush()?.expect("flush 2");

    let handle = engine
        .compact(0, 1)?
        .expect("two files must compact");

    // Positive assertion: data intact after merge
    assert!(handle.rows >= 2, "compacted SST must contain both keys");
    assert_eq!(
        engine.get(b"k1", engine.snapshot_ts())?.as_deref(),
        Some(&large[..]),
        "large value must survive compaction"
    );
    assert_eq!(
        engine.get(b"k2", engine.snapshot_ts())?.as_deref(),
        Some(b"small".as_ref()),
        "small value must survive compaction"
    );
    // Negative assertion: must not return wrong data
    assert_ne!(
        engine.get(b"k1", engine.snapshot_ts())?.as_deref(),
        Some(b"small".as_ref()),
        "large value must not be corrupted"
    );
    Ok(())
}
