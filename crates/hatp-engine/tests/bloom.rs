#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::assertions_on_constants
)]

//! PR 3.10 integration tests: per-SST bloom filter sidecar generation, point-lookup pre-filtering, and
//! missing fallback.

use bytes::Bytes;
use hatp_engine::bloom::BloomFilter;
use hatp_engine::{Engine, EngineConfig, Mutation};
use tempfile::{Builder, TempDir};

fn unique_dir(label: &str) -> Result<TempDir, std::io::Error> {
    Builder::new().prefix(&format!("hatp-{label}-")).tempdir()
}

fn put(key: &[u8], value: &[u8]) -> Mutation {
    Mutation::Put {
        key: Bytes::copy_from_slice(key),
        value: Bytes::copy_from_slice(value),
    }
}

/// `flush` must write a `bloom-{file_id:020}.bf` sidecar next to the SST.
#[test]
fn flush_writes_bloom_sidecar() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("bloom-flush")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    engine.write(&[put(b"t\0k1", b"v1"), put(b"t\0k2", b"v2")])?;
    let handle = engine.flush()?.expect("flush produces an SST");
    let bloom_file = dir.path().join(format!("bloom-{:020}.bf", handle.file_id));
    assert!(bloom_file.exists(), "bloom sidecar must be written on flush");

    let filter = BloomFilter::read_from(&bloom_file).expect("bloom sidecar decodes");
    assert!(filter.contains(b"t\0k1"));
    assert!(filter.contains(b"t\0k2"));
    Ok(())
}

/// A point read must stay correct with a bloom sidecar present: hits return
/// the value, misses return `None`.
#[test]
fn get_correctness_with_bloom_sidecar() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("bloom-get")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    for i in 0..64 {
        engine.write(&[put(
            format!("t\0key-{i}").as_bytes(),
            format!("value-{i}").as_bytes(),
        )])?;
    }
    engine.flush()?.expect("flush");
    let snapshot = engine.snapshot_ts();

    // Exact assertion: two boundary keys have correct values
    assert_eq!(
        engine.get(b"t\0key-3", snapshot)?,
        Some(Bytes::from_static(b"value-3"))
    );
    assert_eq!(
        engine.get(b"t\0key-63", snapshot)?,
        Some(Bytes::from_static(b"value-63"))
    );
    // Exact assertion: all 64 keys must be readable
    for i in 0..64 {
        let key = format!("t\0key-{i}");
        let expected = format!("value-{i}");
        assert_eq!(
            engine.get(key.as_bytes(), snapshot)?.as_deref(),
            Some(expected.as_bytes()),
            "key {i} must be readable through bloom"
        );
    }
    // Negative assertion: non-existent key returns None
    assert_eq!(
        engine.get(b"t\0key-1000", snapshot)?,
        None,
        "absent key must return None through the bloom pre-filter"
    );
    // Negative assertion: non-existent key must not return any value
    assert_ne!(
        engine.get(b"t\0key-1000", snapshot)?,
        Some(Bytes::from_static(b"value-0")),
        "absent key must not return a value"
    );
    Ok(())
}

/// Deleting the bloom sidecar must not change read results: the authoritative
/// full read is the fallback, so correctness never depends on the cache.
#[test]
fn get_correctness_without_bloom_sidecar() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("bloom-fallback")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    engine.write(&[put(b"t\0k1", b"v1")])?;
    let handle = engine.flush()?.expect("flush");
    let bloom_file = dir.path().join(format!("bloom-{:020}.bf", handle.file_id));
    std::fs::remove_file(&bloom_file)?;

    let snapshot = engine.snapshot_ts();
    assert_eq!(
        engine.get(b"t\0k1", snapshot)?,
        Some(Bytes::from_static(b"v1")),
        "read must fall back to the full SST when the bloom sidecar is missing"
    );
    Ok(())
}

/// Compaction must retire the input's bloom sidecar alongside the SST.
#[test]
fn compaction_retires_bloom_sidecar() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_dir("bloom-compact")?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;
    engine.write(&[put(b"t\0k1", b"v1")])?;
    let first = engine.flush()?.expect("first flush").file_id;
    engine.write(&[put(b"t\0k1", b"v2")])?;
    engine.flush()?.expect("second flush");
    engine.compact(0, 1)?;

    let first_bloom = dir.path().join(format!("bloom-{first:020}.bf"));
    assert!(
        !first_bloom.exists(),
        "compaction must remove the retired SST's bloom sidecar"
    );
    // The compacted output still resolves the newest value.
    let snapshot = engine.snapshot_ts();
    assert_eq!(
        engine.get(b"t\0k1", snapshot)?,
        Some(Bytes::from_static(b"v2")),
        "compacted output must still serve the newest version"
    );
    Ok(())
}
