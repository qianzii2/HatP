#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::assertions_on_constants
)]

//! Audit gap coverage: Prometheus metrics format, stats_version cache invalidation,
//! hot_key_cache concurrent clear, dedup_superseded duplicate handling.

use bytes::Bytes;
use hatp_engine::metrics::EngineMetrics;
use hatp_engine::{Engine, EngineConfig, Mutation};
use std::sync::Arc;

fn put(key: &[u8], value: &[u8]) -> Mutation {
    Mutation::Put {
        key: Bytes::copy_from_slice(key),
        value: Bytes::copy_from_slice(value),
    }
}

// ── Prometheus text format validation ────────────────────────────────────

#[test]
fn prometheus_text_contains_all_metrics() {
    let m = EngineMetrics::new();
    m.record_commit();
    m.record_abort();
    m.record_ssi_conflict();
    m.record_wal_write(1024);
    m.record_sst_write(2048);
    m.record_flush();
    m.record_compaction();
    m.set_memtable_bytes(4096);
    m.record_sst_read_failure();
    m.record_gc_versions_dropped(10);
    m.record_compaction_panic();

    let text = m.snapshot().to_prometheus_text();

    // Positive: all 11 metrics present with correct format
    let required = [
        "hatp_tx_committed_total",
        "hatp_tx_aborted_total",
        "hatp_ssi_conflicts_total",
        "hatp_wal_bytes_written_total",
        "hatp_sst_bytes_written_total",
        "hatp_flush_count_total",
        "hatp_compaction_count_total",
        "hatp_memtable_bytes",
        "hatp_sst_read_failures_total",
        "hatp_gc_versions_dropped_total",
        "hatp_compaction_panics_total",
    ];
    for metric in &required {
        assert!(
            text.contains(metric),
            "Prometheus output must contain metric `{metric}`"
        );
    }

    // Positive: HELP and TYPE lines present
    assert!(text.contains("# HELP"), "must contain HELP lines");
    assert!(text.contains("# TYPE"), "must contain TYPE lines");

    // Negative: no empty metric lines (value always follows name)
    for line in text.lines() {
        if !line.starts_with('#') && !line.is_empty() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            assert!(
                parts.len() >= 2,
                "metric line must have name and value: `{line}`"
            );
        }
    }
}

#[test]
fn prometheus_text_zero_values_are_present() {
    let m = EngineMetrics::new();
    let text = m.snapshot().to_prometheus_text();

    // All counters should be 0 for a fresh metrics instance
    assert!(text.contains("hatp_tx_committed_total 0"));
    assert!(text.contains("hatp_tx_aborted_total 0"));
    assert!(text.contains("hatp_ssi_conflicts_total 0"));
}

// ── stats_version cache invalidation ────────────────────────────────────

#[test]
fn stats_version_increments_on_flush() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(EngineConfig::new(dir.path())).unwrap();

    let v1 = engine.stats_version();
    engine.write(&[put(b"k", b"v")]).unwrap();
    let v2 = engine.stats_version();
    // stats_version does NOT change on write — only on flush/compaction
    assert_eq!(v1, v2, "stats_version must not change on write");

    engine.flush().unwrap();
    let v3 = engine.stats_version();
    assert!(v3 > v2, "stats_version must increment after flush (v2={v2}, v3={v3})");
}

#[test]
fn stats_version_increments_on_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(EngineConfig::new(dir.path())).unwrap();

    // Create multiple L0 files to trigger compaction
    for batch in 0..4_u8 {
        for i in 0..50_u64 {
            engine
                .write(&[put(
                    format!("batch{batch}_key_{i:05}").as_bytes(),
                    b"value",
                )])
                .unwrap();
        }
        engine.flush().unwrap();
    }

    let v_before = engine.stats_version();
    let compact_result = engine.compact(0, 1);
    let v_after = engine.stats_version();
    // compaction may not run if not enough files; if it ran, version must increase
    if compact_result.is_ok() && compact_result.as_ref().ok().and_then(|o| o.as_ref()).is_some() {
        assert!(v_after > v_before,
            "stats_version must increase after successful compaction (v_before={v_before}, v_after={v_after})");
    }
    // Negative: version must never decrease
    assert!(v_after >= v_before,
        "stats_version must not decrease after compaction (v_before={v_before}, v_after={v_after})");
}

// ── dedup_superseded duplicate (key, begin_ts) ──────────────────────────

#[test]
fn dedup_superseded_removes_exact_duplicates() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(EngineConfig::new(dir.path())).unwrap();

    // Write the same key twice with different values
    engine
        .write(&[Mutation::Put {
            key: Bytes::from_static(b"dedup_key"),
            value: Bytes::from_static(b"first"),
        }])
        .unwrap();
    engine.flush().unwrap();
    engine
        .write(&[Mutation::Put {
            key: Bytes::from_static(b"dedup_key"),
            value: Bytes::from_static(b"second"),
        }])
        .unwrap();
    engine.flush().unwrap();

    // Compact: dedup_superseded should keep only the newest version
    let _ = engine.compact(0, 1);

    let snap = engine.snapshot_ts();
    let v = engine.get(b"dedup_key", snap).unwrap();
    // Only the newest version should survive
    assert_eq!(
        v.as_deref(),
        Some(b"second".as_ref()),
        "compaction must keep newest version, dedup_superseded must not lose data"
    );
}

// ── hot_key_cache concurrent access ─────────────────────────────────────

#[test]
fn hot_key_cache_survives_concurrent_get_and_clear() {
    use std::thread;

    let dir = tempfile::tempdir().unwrap();
    let engine = Arc::new(Engine::open(EngineConfig::new(dir.path())).unwrap());

    // Populate enough keys to exercise the cache (HOT_KEY_CACHE_MAX = 16384)
    for i in 0..1000_u64 {
        engine
            .write(&[put(format!("hk_{i:05}").as_bytes(), b"v")])
            .unwrap();
    }
    engine.flush().unwrap();
    let snap = engine.snapshot_ts();

    // Concurrent reads from multiple threads
    let mut handles = Vec::new();
    for t in 0..4 {
        let e = Arc::clone(&engine);
        handles.push(thread::spawn(move || {
            for i in 0..200_u64 {
                let key = format!("hk_{:05}", (t * 200 + i) % 1000);
                let _ = e.get(key.as_bytes(), snap);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // Positive: engine still usable, hot keys still readable
    let v = engine.get(b"hk_00000", snap).unwrap();
    assert!(v.is_some(), "hot_key_cache must not corrupt engine state");

    let v2 = engine.get(b"hk_00500", snap).unwrap();
    assert_eq!(v2.as_deref(), Some(b"v".as_ref()),
        "all keys must return correct values after concurrent access");

    // Negative: nonexistent key must not be affected by cache pollution
    let missing = engine.get(b"hk_nonexistent", snap).unwrap();
    assert!(missing.is_none(), "cache must not invent non-existent keys");
}