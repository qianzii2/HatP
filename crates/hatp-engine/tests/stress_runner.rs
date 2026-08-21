//! db_stress-style deterministic stress runner (VC-008)
//!
//! Mirrors RocksDB db_stress: seed → DeterministicRandom → pick operation →
//! execute on Engine + MirrorStore → compare → optionally crash/restart →
//! full VerifyDB at end.
//!
//! Run: cargo test --test stress_runner -- --seed 42 --ops 1000

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

mod common;

use bytes::Bytes;
use common::deterministic::DeterministicRandom;
use common::mirror::MirrorStore;
use hatp_engine::{Engine, EngineConfig, Mutation};
use std::env;
use std::sync::Arc;
use tempfile::Builder;

// ── Operation space ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum StressOp {
    Put { key: Bytes, value: Bytes },
    Delete { key: Bytes },
    Get { key: Bytes },
    ScanRange { lower: Bytes, upper: Bytes },
    Flush,
    Compact,
    Restart,
}

struct StressRunner {
    rng: DeterministicRandom,
    mirror: MirrorStore,
    engine: Arc<Engine>,
    dir: tempfile::TempDir,
    seed: u64,
    op_count: usize,
}

impl StressRunner {
    fn new(seed: u64) -> Self {
        let rng = DeterministicRandom::new(seed);
        let mirror = MirrorStore::new();
        let dir = Builder::new()
            .prefix(&format!("hatp-stress-{seed}-"))
            .tempdir()
            .expect("tempdir");
        let engine = Engine::open(EngineConfig::new(dir.path())).expect("open engine");
        Self {
            rng,
            mirror,
            engine,
            dir,
            seed,
            op_count: 0,
        }
    }

    fn random_key(&mut self) -> Bytes {
        let k = self.rng.next_u64_bounded(10_000);
        Bytes::from(format!("key_{k:05}").into_bytes())
    }

    fn random_value(&mut self) -> Bytes {
        let len = self.rng.next_u64_bounded(256) as usize + 1;
        let mut v = Vec::with_capacity(len);
        for _ in 0..len {
            v.push(self.rng.next_u64_bounded(256) as u8);
        }
        Bytes::from(v)
    }

    fn pick_op(&mut self) -> StressOp {
        let roll = self.rng.next_u64_bounded(100);
        match roll {
            0..=24 => StressOp::Put {
                key: self.random_key(),
                value: self.random_value(),
            },
            25..=34 => StressOp::Delete {
                key: self.random_key(),
            },
            35..=64 => StressOp::Get {
                key: self.random_key(),
            },
            65..=74 => {
                let lower = self.random_key();
                let upper = Bytes::from(format!(
                    "{}",
                    String::from_utf8_lossy(&lower)
                ));
                StressOp::ScanRange {
                    lower,
                    upper: Bytes::from_static(b"key_\xff"),
                }
            }
            75..=84 => StressOp::Flush,
            85..=94 => StressOp::Compact,
            95..=99 => StressOp::Restart,
            _ => StressOp::Get {
                key: self.random_key(),
            },
        }
    }

    fn apply(&mut self, op: &StressOp) -> Result<(), String> {
        self.op_count += 1;
        match op {
            StressOp::Put { key, value } => {
                self.mirror.put(key.clone(), value.clone());
                self.engine
                    .write(&[Mutation::Put {
                        key: key.clone(),
                        value: value.clone(),
                    }])
                    .map_err(|e| format!("put failed: {e}"))?;
            }
            StressOp::Delete { key } => {
                self.mirror.delete(key.clone());
                self.engine
                    .write(&[Mutation::Delete {
                        key: key.clone(),
                    }])
                    .map_err(|e| format!("delete failed: {e}"))?;
            }
            StressOp::Get { key } => {
                let engine_val = self
                    .engine
                    .get(key, self.engine.snapshot_ts())
                    .map_err(|e| format!("get failed: {e}"))?;
                let mirror_val = self.mirror.get(key);
                let expected = mirror_val.and_then(|v| v);
                if engine_val != expected {
                    return Err(format!(
                        "seed={} op={} key={key:?}: engine={engine_val:?} mirror={expected:?}",
                        self.seed, self.op_count
                    ));
                }
            }
            StressOp::ScanRange { lower, upper } => {
                let engine_scan = self
                    .engine
                    .scan_range(lower, upper, self.engine.snapshot_ts())
                    .map_err(|e| format!("scan failed: {e}"))?;
                let mirror_scan = self.mirror.scan_range(lower, upper);
                if engine_scan != mirror_scan {
                    return Err(format!(
                        "seed={} op={} scan [{lower:?}, {upper:?}): \
                         engine={engine_scan:?} mirror={mirror_scan:?}",
                        self.seed, self.op_count
                    ));
                }
            }
            StressOp::Flush => {
                self.engine
                    .flush()
                    .map_err(|e| format!("flush failed: {e}"))?;
            }
            StressOp::Compact => {
                self.engine
                    .compact(0, 1)
                    .map_err(|e| format!("compact failed: {e}"))?;
            }
            StressOp::Restart => {
                let path = self.engine.path().to_path_buf();
                drop(std::mem::replace(
                    &mut self.engine,
                    Engine::open(EngineConfig::new(&path))
                        .map_err(|e| format!("restart failed: {e}"))?,
                ));
            }
        }
        Ok(())
    }

    fn verify_all(&self) -> Result<(), String> {
        let snap = self.engine.snapshot_ts();
        let all_keys = self.mirror.scan_range(b"", b"\xff");
        for (key, expected) in all_keys {
            let actual = self
                .engine
                .get(&key, snap)
                .map_err(|e| format!("verify get failed: {e}"))?;
            if actual.as_deref() != Some(expected.as_ref()) {
                return Err(format!(
                    "seed={} VERIFY key={key:?}: engine={actual:?} mirror={expected:?}",
                    self.seed
                ));
            }
        }
        Ok(())
    }
}

// ── Test entry point ────────────────────────────────────────────────────────

#[test]
fn stress_runner_deterministic() {
    let seed: u64 = env::var("STRESS_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let ops: usize = env::var("STRESS_OPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let mut runner = StressRunner::new(seed);
    for i in 0..ops {
        let op = runner.pick_op();
        if let Err(e) = runner.apply(&op) {
            panic!("stress failure at seed={seed} op={i}: {e}");
        }
    }
    if let Err(e) = runner.verify_all() {
        panic!("verify failure: {e}");
    }
    eprintln!("OK seed={seed} ops={ops}");
}