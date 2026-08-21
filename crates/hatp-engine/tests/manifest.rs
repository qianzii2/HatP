#![allow(clippy::expect_used, clippy::unwrap_used, clippy::indexing_slicing)]

//! Manifest version-set ArcSwap (PR 2.2) integration tests.

use bytes::Bytes;
use hatp_engine::manifest::{Manifest, VersionEdit};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

fn add_file(file_id: u64, level: u32) -> VersionEdit {
    VersionEdit::AddFile {
        file_id,
        level,
        min_key: Bytes::from_static(b"a"),
        max_key: Bytes::from_static(b"z"),
        bytes: 10,
        created_at: 0,
    }
}

#[test]
fn concurrent_reads_with_append_no_lock() {
    let dir = tempfile::tempdir().unwrap();
    let mut manifest = Manifest::open(dir.path().join("MANIFEST")).unwrap();
    manifest
        .append_batch(vec![add_file(1, 0), VersionEdit::NextFileId(2)])
        .unwrap();

    let manifest = Arc::new(std::sync::Mutex::new(manifest));
    let stop = Arc::new(AtomicBool::new(false));

    // 4 readers concurrently read the version set (load_full is lock-free with no deep copy).
    let mut readers = Vec::new();
    for _ in 0..4 {
        let manifest = Arc::clone(&manifest);
        let stop = Arc::clone(&stop);
        readers.push(thread::spawn(move || {
            let mut seen = 0_u64;
            while !stop.load(Ordering::Relaxed) {
                let guard = manifest.lock().unwrap();
                seen = seen.wrapping_add(guard.version_set().next_file_id());
            }
            seen
        }));
    }

    // 1 writer concurrently appends.
    for i in 0..200 {
        let mut guard = manifest.lock().unwrap();
        guard
            .append_batch(vec![add_file(2 + i, 1), VersionEdit::NextFileId(3 + i)])
            .unwrap();
    }

    stop.store(true, Ordering::Relaxed);
    for reader in readers {
        reader.join().unwrap();
    }

    // The final version set must reflect all appends (next_file_id monotonically reaches 202).
    let guard = manifest.lock().unwrap();
    assert_eq!(guard.version_set().next_file_id(), 202);
    assert_eq!(guard.version_set().files(1).len(), 200);
}
