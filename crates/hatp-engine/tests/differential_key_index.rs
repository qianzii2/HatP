//! KeyIndex vs Vortex differential test (VC-019)
//!
//! Invariant: for the same key point lookup, KeyIndex and Vortex return consistent VersionedValue
//! Cross-validation boundary: KeyIndex is hand-written binary search, Vortex is a columnar engine — zero shared code
//! Independence: KeyIndex uses raw pointer + read_unaligned, Vortex uses columnar decode
//! Positive evidence: key_index.rs:169-200 (KeyIndex::get), vortex_sst.rs:599-677 (collect_rows)

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use bytes::Bytes;
use hatp_engine::{Engine, EngineConfig, Mutation};
use hatp_engine::key_index::KeyIndex;
use hatp_engine::version::VersionedValue;
use tempfile::Builder;

fn put(key: &[u8], value: &[u8]) -> Mutation {
    Mutation::Put {
        key: Bytes::copy_from_slice(key),
        value: Bytes::copy_from_slice(value),
    }
}

#[test]
fn key_index_and_vortex_point_lookup_agree() {
    let dir = Builder::new()
        .prefix("hatp-diff-kidx-vortex-")
        .tempdir()
        .expect("tempdir");
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");

    // Write 100 keys, flush to SST
    for i in 0..100u64 {
        let key = format!("key_{i:05}");
        let value = format!("value_{i:05}");
        engine.write(&[put(key.as_bytes(), value.as_bytes())]).expect("write");
    }
    let handle = engine.flush().expect("flush").expect("SST handle");

    let snap = engine.snapshot_ts();

    // Read the key-index sidecar
    let kidx_path = dir.path().join(format!("kidx-{:020}.bin", handle.file_id));
    let kidx = KeyIndex::read_from(&kidx_path).expect("key index exists");

    // For every key, compare KeyIndex vs Engine::get (which uses Vortex fallback)
    for i in 0..100u64 {
        let key = format!("key_{i:05}");
        let expected = format!("value_{i:05}");

        // KeyIndex lookup
        let kidx_val: Option<VersionedValue> = kidx.get(key.as_bytes());
        assert!(kidx_val.is_some(), "KeyIndex must find key_{i:05}");
        let kidx_val = kidx_val.unwrap();
        assert_eq!(
            kidx_val.value.as_deref(),
            Some(expected.as_bytes()),
            "KeyIndex value mismatch for key_{i:05}"
        );

        // Engine::get (Vortex path)
        let engine_val = engine.get(key.as_bytes(), snap).expect("engine get");
        assert_eq!(
            engine_val.as_deref(),
            Some(expected.as_bytes()),
            "Engine::get value mismatch for key_{i:05}"
        );
    }

    // Negative: non-existent key must be absent in both
    assert!(kidx.get(b"key_nonexistent").is_none(), "KeyIndex must miss absent key");
    assert_eq!(engine.get(b"key_nonexistent", snap).expect("get"), None,
        "Engine::get must miss absent key");
}

#[test]
fn key_index_and_vortex_agree_on_tombstone() {
    let dir = Builder::new()
        .prefix("hatp-diff-kidx-tomb-")
        .tempdir()
        .expect("tempdir");
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");

    // Write then delete key
    engine.write(&[put(b"alive", b"keep"), put(b"dead", b"gone")]).expect("write");
    engine.write(&[Mutation::Delete {
        key: Bytes::from_static(b"dead"),
    }]).expect("delete");
    let handle = engine.flush().expect("flush").expect("SST");

    let snap = engine.snapshot_ts();
    let kidx_path = dir.path().join(format!("kidx-{:020}.bin", handle.file_id));
    let kidx = KeyIndex::read_from(&kidx_path).expect("key index");

    // Tombstone: KeyIndex returns value=None
    let tomb = kidx.get(b"dead").expect("tombstone present in key index");
    assert!(tomb.value.is_none(), "KeyIndex tombstone must have value=None");

    // Engine::get must also return None
    assert_eq!(engine.get(b"dead", snap).expect("get"), None,
        "Engine::get must return None for tombstone");

    // Live key must be present in both
    let live = kidx.get(b"alive").expect("live key present");
    assert_eq!(live.value.as_deref(), Some(b"keep".as_ref()));
    assert_eq!(engine.get(b"alive", snap).expect("get").as_deref(),
        Some(b"keep".as_ref()));
}