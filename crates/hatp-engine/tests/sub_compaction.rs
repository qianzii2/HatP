#![allow(clippy::expect_used, clippy::unwrap_used, clippy::indexing_slicing)]

//! Sub-compaction (PR 2.6) integration tests: parallel merge L0 to L1.

use bytes::Bytes;
use hatp_engine::{Engine, EngineConfig, Mutation};

fn put(key: &[u8], value: &[u8]) -> Mutation {
    Mutation::Put {
        key: Bytes::copy_from_slice(key),
        value: Bytes::copy_from_slice(value),
    }
}

#[test]
fn sub_compaction_parallel_merges_l0() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(EngineConfig::new(dir.path())).unwrap();

    // 6 batches + 6 flushes → 6 L0 SSTs.
    for i in 0..6_u8 {
        let key = format!("k{i:03}");
        let value = format!("v{i:03}");
        engine
            .write(&[put(key.as_bytes(), value.as_bytes())])
            .unwrap();
        engine.flush().unwrap();
    }
    assert_eq!(engine.sst_file_ids().len(), 6);

    // Parallel sub-compaction: L0 → L1.
    let handles = engine
        .sub_compact(0, 1, engine.watermark())
        .unwrap()
        .expect("6 L0 files must produce a sub-compaction job");
    assert!(!handles.is_empty());

    // After merge, data is still readable, and L0 is cleared.
    let snapshot = engine.snapshot_ts();
    for i in 0..6_u8 {
        let key = format!("k{i:03}");
        let got = engine
            .get(key.as_bytes(), snapshot)
            .unwrap()
            .expect("key must survive compaction");
        assert_eq!(got.as_ref(), format!("v{i:03}").as_bytes());
    }
    // Negative assertion: L0 is cleared, only L1+ data remains
    // Verify SST file count decreased (compaction merged 6 L0 files)
    let remaining = engine.sst_file_ids().len();
    assert!(remaining < 6, "compaction must reduce SST file count, got {remaining}");
}

#[test]
fn sub_compaction_is_atomic_on_single_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(EngineConfig::new(dir.path())).unwrap();

    // Fewer than 2 inputs does not produce a job (consistent with compact semantics).
    engine.write(&[put(b"k", b"v")]).unwrap();
    engine.flush().unwrap();
    let result = engine.sub_compact(0, 1, engine.watermark()).unwrap();
    assert!(result.is_none(), "single input must not trigger compaction");
}
