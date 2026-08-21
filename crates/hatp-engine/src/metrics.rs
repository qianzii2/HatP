//! Prometheus-compatible metrics for the HatP engine.
//!
//! Metrics are recorded as lock-free `AtomicU64` counters/gauges and exposed
//! as Prometheus text format via [`MetricsSnapshot::to_prometheus_text`] (a
//! hand-rolled exporter — no `metrics`-crate dependency, keeping the core
//! library lightweight). An external scraper can poll that text output
//! directly; the engine itself has no HTTP endpoint.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// ── Metric definitions ─────────────────────────────────────────────────────────────

/// Global metrics for the engine.
///
/// All fields are lock-free atomics — recording is fast and never blocks.
#[derive(Debug, Default)]
pub struct EngineMetrics {
    /// Number of transactions committed.
    pub tx_committed: AtomicU64,
    /// Number of transactions aborted (including SSI conflicts).
    pub tx_aborted: AtomicU64,
    /// Number of SSI conflicts detected.
    pub ssi_conflicts: AtomicU64,
    /// Total bytes written to WAL.
    pub wal_bytes_written: AtomicU64,
    /// Total bytes flushed to SST.
    pub sst_bytes_written: AtomicU64,
    /// Number of flushes performed.
    pub flush_count: AtomicU64,
    /// Number of compactions performed.
    pub compaction_count: AtomicU64,
    /// Approximate bytes in memtable.
    pub memtable_bytes: AtomicU64,
    /// Number of SST read failures (e.g. during recovery or point lookups).
    /// Bumped whenever an SST cannot be decoded, so a corrupt file that is
    /// silently skipped is still observable in monitoring.
    pub sst_read_failures: AtomicU64,
    /// Number of MVCC versions dropped by the periodic GC worker.
    pub gc_versions_dropped: AtomicU64,
    /// Number of times the background compaction worker caught a panic.
    pub compaction_panics: AtomicU64,
}

impl EngineMetrics {
    /// Creates a fresh metrics instance.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl EngineMetrics {
    /// Records a committed transaction.
    pub fn record_commit(&self) {
        self.tx_committed.fetch_add(1, Ordering::Relaxed);
    }

    /// Records an aborted transaction.
    pub fn record_abort(&self) {
        self.tx_aborted.fetch_add(1, Ordering::Relaxed);
    }

    /// Records an SSI conflict.
    pub fn record_ssi_conflict(&self) {
        self.ssi_conflicts.fetch_add(1, Ordering::Relaxed);
    }

    /// Records WAL bytes written.
    pub fn record_wal_write(&self, bytes: u64) {
        self.wal_bytes_written.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Records SST bytes written.
    pub fn record_sst_write(&self, bytes: u64) {
        self.sst_bytes_written.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Records a flush operation.
    pub fn record_flush(&self) {
        self.flush_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a compaction operation.
    pub fn record_compaction(&self) {
        self.compaction_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Sets the memtable size estimate.
    pub fn set_memtable_bytes(&self, bytes: u64) {
        self.memtable_bytes.store(bytes, Ordering::Relaxed);
    }

    /// Records an SST read failure.
    pub fn record_sst_read_failure(&self) {
        self.sst_read_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Records MVCC versions dropped by the GC worker.
    pub fn record_gc_versions_dropped(&self, count: u64) {
        self.gc_versions_dropped.fetch_add(count, Ordering::Relaxed);
    }

    /// Records a compaction worker panic.
    pub fn record_compaction_panic(&self) {
        self.compaction_panics.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns a snapshot of all metrics at this instant.
    #[must_use]
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            tx_committed: self.tx_committed.load(Ordering::Relaxed),
            tx_aborted: self.tx_aborted.load(Ordering::Relaxed),
            ssi_conflicts: self.ssi_conflicts.load(Ordering::Relaxed),
            wal_bytes_written: self.wal_bytes_written.load(Ordering::Relaxed),
            sst_bytes_written: self.sst_bytes_written.load(Ordering::Relaxed),
            flush_count: self.flush_count.load(Ordering::Relaxed),
            compaction_count: self.compaction_count.load(Ordering::Relaxed),
            memtable_bytes: self.memtable_bytes.load(Ordering::Relaxed),
            sst_read_failures: self.sst_read_failures.load(Ordering::Relaxed),
            gc_versions_dropped: self.gc_versions_dropped.load(Ordering::Relaxed),
            compaction_panics: self.compaction_panics.load(Ordering::Relaxed),
        }
    }
}

/// Point-in-time snapshot of all engine metrics.
#[derive(Debug, Clone, Default)]
pub struct MetricsSnapshot {
    pub tx_committed: u64,
    pub tx_aborted: u64,
    pub ssi_conflicts: u64,
    pub wal_bytes_written: u64,
    pub sst_bytes_written: u64,
    pub flush_count: u64,
    pub compaction_count: u64,
    pub memtable_bytes: u64,
    pub sst_read_failures: u64,
    pub gc_versions_dropped: u64,
    pub compaction_panics: u64,
}

impl MetricsSnapshot {
    /// Formats this snapshot as Prometheus text format.
    #[must_use]
    pub fn to_prometheus_text(&self) -> String {
        let mut out = String::new();

        out.push_str("# HELP hatp_tx_committed_total Total committed transactions\n");
        out.push_str("# TYPE hatp_tx_committed_total counter\n");
        out.push_str(&format!("hatp_tx_committed_total {}\n", self.tx_committed));

        out.push_str("# HELP hatp_tx_aborted_total Total aborted transactions\n");
        out.push_str("# TYPE hatp_tx_aborted_total counter\n");
        out.push_str(&format!("hatp_tx_aborted_total {}\n", self.tx_aborted));

        out.push_str("# HELP hatp_ssi_conflicts_total Total SSI conflicts\n");
        out.push_str("# TYPE hatp_ssi_conflicts_total counter\n");
        out.push_str(&format!(
            "hatp_ssi_conflicts_total {}\n",
            self.ssi_conflicts
        ));

        out.push_str("# HELP hatp_wal_bytes_written_total Total bytes written to WAL\n");
        out.push_str("# TYPE hatp_wal_bytes_written_total counter\n");
        out.push_str(&format!(
            "hatp_wal_bytes_written_total {}\n",
            self.wal_bytes_written
        ));

        out.push_str("# HELP hatp_sst_bytes_written_total Total bytes written to SST\n");
        out.push_str("# TYPE hatp_sst_bytes_written_total counter\n");
        out.push_str(&format!(
            "hatp_sst_bytes_written_total {}\n",
            self.sst_bytes_written
        ));

        out.push_str("# HELP hatp_flush_count_total Total flushes\n");
        out.push_str("# TYPE hatp_flush_count_total counter\n");
        out.push_str(&format!("hatp_flush_count_total {}\n", self.flush_count));

        out.push_str("# HELP hatp_compaction_count_total Total compactions\n");
        out.push_str("# TYPE hatp_compaction_count_total counter\n");
        out.push_str(&format!(
            "hatp_compaction_count_total {}\n",
            self.compaction_count
        ));

        out.push_str("# HELP hatp_memtable_bytes Approximate memtable bytes\n");
        out.push_str("# TYPE hatp_memtable_bytes gauge\n");
        out.push_str(&format!("hatp_memtable_bytes {}\n", self.memtable_bytes));

        out.push_str("# HELP hatp_sst_read_failures_total Total SST read failures\n");
        out.push_str("# TYPE hatp_sst_read_failures_total counter\n");
        out.push_str(&format!(
            "hatp_sst_read_failures_total {}\n",
            self.sst_read_failures
        ));

        out.push_str("# HELP hatp_gc_versions_dropped_total MVCC versions dropped by GC\n");
        out.push_str("# TYPE hatp_gc_versions_dropped_total counter\n");
        out.push_str(&format!(
            "hatp_gc_versions_dropped_total {}\n",
            self.gc_versions_dropped
        ));

        out.push_str("# HELP hatp_compaction_panics_total Compaction worker panics\n");
        out.push_str("# TYPE hatp_compaction_panics_total counter\n");
        out.push_str(&format!(
            "hatp_compaction_panics_total {}\n",
            self.compaction_panics
        ));

        out
    }
}

/// Shared metrics instance for the engine.
pub type SharedMetrics = Arc<EngineMetrics>;

/// Creates a new shared metrics instance.
#[must_use]
pub fn create_metrics() -> SharedMetrics {
    Arc::new(EngineMetrics::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot() {
        let m = EngineMetrics::new();
        m.record_commit();
        m.record_commit();
        m.record_abort();
        m.record_wal_write(1024);

        let s = m.snapshot();
        assert_eq!(s.tx_committed, 2);
        assert_eq!(s.tx_aborted, 1);
        assert_eq!(s.wal_bytes_written, 1024);
    }

    #[test]
    fn test_prometheus_text() {
        let m = EngineMetrics::new();
        m.record_commit();
        let text = m.snapshot().to_prometheus_text();
        assert!(text.contains("hatp_tx_committed_total 1"));
    }
}
