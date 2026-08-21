//! HatP LSM benchmarks (VC-020)
//!
//! Scenarios: fillseq, fillrandom, readrandom, readwhilewriting, compact
//! Mirrors RocksDB db_bench + sled criterion patterns
//! Run: cargo bench -p hatp-engine

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use bytes::Bytes;
use hatp_engine::{Engine, EngineConfig, Mutation};
use std::sync::Arc;
use std::time::Duration;
use tempfile::Builder;

fn temp_engine() -> (Arc<Engine>, tempfile::TempDir) {
    let dir = Builder::new().prefix("hatp-bench-").tempdir().expect("tempdir");
    let engine = Engine::open(EngineConfig::new(dir.path())).expect("open");
    (engine, dir)
}

fn put(key: &[u8], value: &[u8]) -> Mutation {
    Mutation::Put { key: Bytes::copy_from_slice(key), value: Bytes::copy_from_slice(value) }
}

const N_KEYS: usize = 10_000;

// ── fillseq: sequential write ──────────────────────────────────────────────

fn bench_fillseq(c: &mut Criterion) {
    c.bench_function("fillseq", |b| {
        b.iter(|| {
            let (engine, _dir) = temp_engine();
            for i in 0..N_KEYS {
                let key = format!("key_{i:010}");
                let val = format!("value_{i:010}");
                engine.write(&[put(key.as_bytes(), val.as_bytes())]).expect("write");
            }
            black_box(&engine);
        });
    });
}

// ── fillrandom: random write ────────────────────────────────────────────────

fn bench_fillrandom(c: &mut Criterion) {
    c.bench_function("fillrandom", |b| {
        b.iter(|| {
            let (engine, _dir) = temp_engine();
            // Deterministic "random" using simple LCG
            let mut rng: u64 = 42;
            for _ in 0..N_KEYS {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let key = format!("key_{rng:020}");
                let val = "value";
                engine.write(&[put(key.as_bytes(), val.as_bytes())]).expect("write");
            }
            black_box(&engine);
        });
    });
}

// ── readrandom: random point lookup ─────────────────────────────────────────

fn bench_readrandom(c: &mut Criterion) {
    // Pre-populate
    let (engine, _dir) = temp_engine();
    let mut keys: Vec<Vec<u8>> = Vec::with_capacity(N_KEYS);
    for i in 0..N_KEYS {
        let key = format!("key_{i:010}");
        engine.write(&[put(key.as_bytes(), b"value")]).expect("write");
        keys.push(key.into_bytes());
    }
    engine.flush().expect("flush");

    let snap = engine.snapshot_ts();
    c.bench_function("readrandom", |b| {
        b.iter(|| {
            let mut rng: u64 = 42;
            for _ in 0..1000 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let idx = (rng as usize) % keys.len();
                let val = engine.get(&keys[idx], snap).expect("get");
                black_box(val);
            }
        });
    });
}

// ── point_lookup: single key lookup (sled pattern) ─────────────────────────

fn bench_point_lookup(c: &mut Criterion) {
    let (engine, _dir) = temp_engine();
    let mut keys: Vec<Vec<u8>> = Vec::with_capacity(N_KEYS);
    for i in 0..N_KEYS {
        let key = format!("key_{i:010}");
        engine.write(&[put(key.as_bytes(), b"value")]).expect("write");
        keys.push(key.into_bytes());
    }
    engine.flush().expect("flush");
    let snap = engine.snapshot_ts();

    let mut group = c.benchmark_group("point_lookup");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(5));

    group.bench_function("hit", |b| {
        b.iter(|| {
            let val = engine.get(b"key_0000005000", snap).expect("get");
            black_box(val);
        });
    });

    group.bench_function("miss", |b| {
        b.iter(|| {
            let val = engine.get(b"key_nonexistent", snap).expect("get");
            black_box(val);
        });
    });
    group.finish();
}

// ── range_scan: sequential scan (sled pattern) ─────────────────────────────

fn bench_range_scan(c: &mut Criterion) {
    let (engine, _dir) = temp_engine();
    for i in 0..N_KEYS {
        let key = format!("key_{i:010}");
        engine.write(&[put(key.as_bytes(), b"value")]).expect("write");
    }
    engine.flush().expect("flush");
    let snap = engine.snapshot_ts();

    let mut group = c.benchmark_group("range_scan");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("100_keys", |b| {
        b.iter(|| {
            let results = engine.scan_range(b"key_0000000000", b"key_0000000100", snap)
                .expect("scan");
            black_box(results);
        });
    });

    group.bench_function("1000_keys", |b| {
        b.iter(|| {
            let results = engine.scan_range(b"key_0000000000", b"key_0000001000", snap)
                .expect("scan");
            black_box(results);
        });
    });
    group.finish();
}

// ── compact: compaction throughput ──────────────────────────────────────────

fn bench_compact(c: &mut Criterion) {
    c.bench_function("compact", |b| {
        b.iter(|| {
            let (engine, _dir) = temp_engine();
            // Write 3 batches, flush each to create L0 SSTs
            for batch in 0..3u8 {
                for i in 0..100u64 {
                    let key = format!("batch{batch}_key_{i:05}");
                    engine.write(&[put(key.as_bytes(), b"value")]).expect("write");
                }
                engine.flush().expect("flush");
            }
            engine.compact(0, 1).expect("compact");
            black_box(&engine);
        });
    });
}

criterion_group!(
    benches,
    bench_fillseq,
    bench_fillrandom,
    bench_readrandom,
    bench_point_lookup,
    bench_range_scan,
    bench_compact,
);
criterion_main!(benches);