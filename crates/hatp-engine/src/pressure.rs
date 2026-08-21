//! Write-pressure throttling — prevent an unbounded memtable when flush /
//! compaction cannot keep up (SILK-style backpressure, R-17 companion).
//!
//! The throttle is a read-only check on the memtable gauge: when the memtable
//! approaches its flush threshold, the engine slows (Soft) or rejects (Hard)
//! incoming writes so the durable path has time to drain. It deliberately does
//! NOT enter the group-commit worker or the write lock, so a coalesced fsync
//! window is never split by a throttle sleep (PR 2.7's
//! "group_commit_survives_throttle" invariant).

use crate::metrics::MetricsSnapshot;

/// How aggressively the engine should throttle a write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThrottleLevel {
    /// No backpressure: proceed immediately.
    None,
    /// Soft backpressure: the caller should briefly delay before committing.
    Soft,
    /// Hard backpressure: reject the write so the caller can retry later.
    Hard,
}

/// Default memtable flush threshold (bytes).
///
/// This is the single source of truth shared by `EngineConfig::memtable_flush_bytes` and
/// `PressureConfig::memtable_flush_bytes`. Both locations previously hard-coded `64 * 1024 * 1024`
/// independently; if one was missed during a change, the throttle threshold would drift from the
/// flush threshold. Change it here only; do not scatter hard-coded values again.
pub const DEFAULT_MEMTABLE_FLUSH_BYTES: usize = 64 * 1024 * 1024;

/// Tuning for [`PressureThrottle`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PressureConfig {
    /// Memtable-occupancy ratio (against `memtable_flush_bytes`) at which Soft
    /// throttling begins.
    pub soft_memtable_ratio: f64,
    /// Memtable-occupancy ratio at which writes are Hard-rejected.
    pub hard_memtable_ratio: f64,
    /// Flush threshold in bytes. Mirrors `EngineConfig::memtable_flush_bytes`.
    pub memtable_flush_bytes: usize,
    /// Soft-throttle delay applied before each throttled write.
    pub soft_sleep: std::time::Duration,
}

impl Default for PressureConfig {
    fn default() -> Self {
        Self {
            soft_memtable_ratio: 0.7,
            hard_memtable_ratio: 0.95,
            memtable_flush_bytes: DEFAULT_MEMTABLE_FLUSH_BYTES,
            soft_sleep: std::time::Duration::from_millis(1),
        }
    }
}

/// Stateless throttle: every decision is derived from the [`MetricsSnapshot`]
/// passed in, so the check is a couple of atomic loads and no lock is taken.
#[derive(Debug, Clone, Copy, Default)]
pub struct PressureThrottle {
    config: PressureConfig,
}

impl PressureThrottle {
    /// Creates a throttle with `config`.
    #[must_use]
    pub fn new(config: PressureConfig) -> Self {
        Self { config }
    }

    /// Classifies the current write pressure from `metrics`.
    ///
    /// The token-bucket analogy from the plan is realised as a ratio check on
    /// `memtable_bytes` (the "tokens consumed") against `memtable_flush_bytes`
    /// (the "bucket capacity"): the fuller the bucket, the harder we throttle.
    #[must_use]
    pub fn should_throttle(&self, metrics: &MetricsSnapshot) -> ThrottleLevel {
        let capacity = self.config.memtable_flush_bytes.max(1) as f64;
        let ratio = metrics.memtable_bytes as f64 / capacity;
        if ratio >= self.config.hard_memtable_ratio {
            ThrottleLevel::Hard
        } else if ratio >= self.config.soft_memtable_ratio {
            ThrottleLevel::Soft
        } else {
            ThrottleLevel::None
        }
    }

    /// The delay a Soft-throttled caller should sleep before committing.
    #[must_use]
    pub fn soft_sleep(&self) -> std::time::Duration {
        self.config.soft_sleep
    }
}
