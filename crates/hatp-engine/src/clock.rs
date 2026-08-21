//! Injectable time source (`Clock`), allowing the engine's wall-clock timestamps to be replaced.
//!
//! The engine depends on real wall-clock in only one place: stamping `created_at` timestamps
//! on newly written SSTs, for age-based compaction priority tie-breaking. Abstracting this
//! into the [`Clock`] trait:
//!
//! - Tests can use [`ManualClock`] to freeze time and assert that SST `created_at`
//!   exactly equals the injected value;
//! - Deterministic simulation (FoundationDB-style; see project plan M3) can
//!   manually advance time to reproduce compaction behavior across time windows.
//!
//! Using wall-clock instead of monotonic clock for `created_at` is intentional: it must
//! remain comparable after process restart. NTP adjustments may briefly disrupt age-based
//! tie-breaking, but that only affects compaction scheduling order and never touches
//! MVCC correctness.

use std::sync::atomic::{AtomicU64, Ordering};

/// Time source abstraction: returns Unix epoch seconds.
///
/// Implementations must be [`Send`] + [`Sync`], because the engine puts it in an
/// [`Arc`](std::sync::Arc) and shares it across multiple threads (foreground write path
/// + background compaction worker).
pub trait Clock: Send + Sync {
    /// Current Unix epoch seconds.
    ///
    /// Second-level granularity is intentional: `created_at` is only used for compaction's
    /// age-based tie-breaking (heuristic scheduling), not for correctness. Multiple SSTs
    /// written within the same second will share the same `created_at`, and tie-breaking
    /// degrades to a stable sort — this does not affect MVCC correctness, so there is no
    /// need (nor should) to change this trait for finer-grained time sources.
    fn now_secs(&self) -> u64;
}

/// Real wall-clock implementation, using [`std::time::SystemTime`] directly.
///
/// If the system time is before the Unix epoch (rare), it falls back to `0`, matching
/// the old inline `now_secs()` behavior exactly.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_secs(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0)
    }
}

/// Clock for testing / simulation: readable and writable, initialized to `0`.
///
/// Unlike [`SystemClock`], time does not advance automatically; callers
/// explicitly advance via [`set`](ManualClock::set) or [`advance`](ManualClock::advance).
/// Internally uses [`AtomicU64`], safe for cross-thread sharing and reading.
#[derive(Debug, Default)]
pub struct ManualClock {
    now_secs: AtomicU64,
}

impl ManualClock {
    /// Constructs a manual clock with the given Unix epoch seconds.
    #[must_use]
    pub fn new(initial_secs: u64) -> Self {
        Self {
            now_secs: AtomicU64::new(initial_secs),
        }
    }

    /// Sets the current time to `secs`.
    pub fn set(&self, secs: u64) {
        self.now_secs.store(secs, Ordering::Release);
    }

    /// Advances by `delta` seconds and returns the time after this call.
    ///
    /// Concurrency semantics: `fetch_add` is atomic, so even if multiple threads call
    /// `advance` concurrently, the return value is the time after this call
    /// (= value before call + `delta`), but it does not guarantee equality with the
    /// global final time (other threads may advance further afterward).
    /// Single-threaded advancement in tests/simulation is unaffected.
    pub fn advance(&self, delta: u64) -> u64 {
        self.now_secs.fetch_add(delta, Ordering::AcqRel) + delta
    }
}

impl Clock for ManualClock {
    fn now_secs(&self) -> u64 {
        self.now_secs.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_clock_set_and_advance() {
        let clock = ManualClock::new(100);
        assert_eq!(clock.now_secs(), 100);
        clock.set(500);
        assert_eq!(clock.now_secs(), 500);
        assert_eq!(clock.advance(60), 560);
        assert_eq!(clock.now_secs(), 560);
    }

    #[test]
    fn system_clock_returns_epoch_secs() {
        let clock = SystemClock;
        let now = clock.now_secs();
        // Real-world time must be after 2020-01-01 (1577836800).
        assert!(now > 1_577_836_800, "unexpected epoch time {now}");
    }
}