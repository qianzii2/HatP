#![allow(clippy::expect_used, clippy::unwrap_used, clippy::indexing_slicing)]

//! `Engine::get` level-order early-exit regression test.
//!
//! `get` returns after hitting a **visible** version in L0, skipping L1+ (deeper levels
//! have older `begin_ts`). But if the L0 version is too new (`begin_ts > snapshot`)
//! and thus not visible, it must fall back to L1's older version — it must not be
//! misjudged as "not present".

use bytes::Bytes;
use hatp_engine::{Engine, EngineConfig, Mutation};

fn put(key: &[u8], value: &[u8]) -> Mutation {
    Mutation::Put {
        key: Bytes::copy_from_slice(key),
        value: Bytes::copy_from_slice(value),
    }
}

#[test]
fn get_early_exit_returns_l0_newest_and_falls_back_to_l1() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(EngineConfig::new(dir.path())).unwrap();

    // commit 1: k=v1 → flush (L0 file 1)
    engine.write(&[put(b"k", b"v1")]).unwrap();
    engine.flush().unwrap();
    // commit 2: j=v2 → flush (L0 file 2)
    engine.write(&[put(b"j", b"v2")]).unwrap();
    engine.flush().unwrap();
    // L0 → L1 compaction: k=v1, j=v2 merged into L1, L0 cleared.
    engine.compact(0, 1).unwrap().expect("two L0 files must compact");
    // commit 3: k=v3 → flush (new L0 file, overwriting L1's k=v1)
    engine.write(&[put(b"k", b"v3")]).unwrap();
    engine.flush().unwrap();

    // snapshot=3 (latest): L0's v3 is visible → early exit, returns v3 (does not read L1's v1).
    assert_eq!(
        engine.get(b"k", 3).unwrap().as_deref(),
        Some(b"v3".as_ref())
    );
    // snapshot=2: L0's v3 (begin_ts=3) is not visible → cannot early exit, falls back to L1's v1.
    assert_eq!(
        engine.get(b"k", 2).unwrap().as_deref(),
        Some(b"v1".as_ref())
    );
    // Negative assertion: snapshot=2 must not see v3
    assert_ne!(
        engine.get(b"k", 2).unwrap().as_deref(),
        Some(b"v3".as_ref()),
        "snapshot=2 must NOT see v3 (begin_ts=3 > snapshot)"
    );
}

#[test]
fn get_early_exit_visible_l0_tombstone_hides_l1() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(EngineConfig::new(dir.path())).unwrap();

    // k=v1 → flush → compact into L1.
    engine.write(&[put(b"k", b"v1")]).unwrap();
    engine.flush().unwrap();
    engine.write(&[put(b"j", b"v2")]).unwrap();
    engine.flush().unwrap();
    engine.compact(0, 1).unwrap().expect("compact");
    // commit 3: delete k → flush (tombstone in L0).
    engine.write(&[Mutation::Delete {
        key: Bytes::from_static(b"k"),
    }]).unwrap();
    engine.flush().unwrap();

    // snapshot=3: L0's tombstone is visible → early exit, get returns None (deleted).
    assert_eq!(engine.get(b"k", 3).unwrap(), None);
    // snapshot=2 (before delete): L0's tombstone (begin_ts=3) is not visible → fall back to L1's v1.
    assert_eq!(
        engine.get(b"k", 2).unwrap().as_deref(),
        Some(b"v1".as_ref())
    );
}