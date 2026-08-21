//! Change-data-capture watch API (PR 3.1, FDB Watch).
//!
//! A [`Watcher`] observes durable commit progress without holding the engine's
//! write lock. Two primitives compose the CDC contract:
//!
//! - [`Watcher::wait_for_resolved`] — waits until the resolved watermark has
//!   reached `ts` and returns the current `commit_ts`. This is the
//!   **consistency** primitive: a replication/CDC consumer waits on it before
//!   reading a snapshot, so it never observes a half-committed prefix.
//! - [`Watcher::subscribe`] — a lossy, ordered stream of [`WatchEvent`]s. It is
//!   **not** a durability guarantee (a slow consumer can lag and drop events);
//!   the correctness anchor is `wait_for_resolved`, and a lagging consumer
//!   re-synchronises by re-reading from the resolved watermark.
//!
//! # Why a dedicated `resolved` watermark (not `Engine::commit_seq`)
//!
//! `commit_seq` is advanced **provisionally**: the group-commit worker bumps it
//! in pre-commit order *before* the WAL `fsync` (a later WAL failure leaves a
//! harmless hole), and the sync path bumps it after the `fsync` but *before*
//! the memtable apply. Reading `commit_seq` directly would therefore let a
//! waiter observe a commit that is neither durable nor applied. The watcher
//! instead tracks its own `resolved` watermark that [`Watcher::publish`]
//! advances only after a commit is fully durable **and** applied — the exact
//! point at which a CDC consumer may safely read it.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{Notify, broadcast};

use futures::StreamExt;

use crate::Mutation;

/// One durable commit, published after the WAL `fsync` and memtable apply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchEvent {
    /// Transaction that committed.
    pub tx_id: u64,
    /// Commit-order sequence number (the MVCC `begin_ts` of its versions).
    pub commit_ts: u64,
    /// The mutations carried by this commit (shared, no per-subscriber copy).
    /// `Mutation::Delete` carries `value = None` (a tombstone); replication
    /// consumers translate it to their own delete representation.
    pub mutations: Arc<[Mutation]>,
}

/// A cheap-to-clone handle that observes commit progress.
#[derive(Debug, Clone)]
pub struct Watcher {
    /// Wakes every `wait_for_resolved` waiter on a new commit.
    notify: Arc<Notify>,
    /// Highest fully durable **and** applied `commit_ts` (see module note).
    resolved: Arc<AtomicU64>,
    /// Bounded, ordered event feed for [`Watcher::subscribe`].
    events: broadcast::Sender<WatchEvent>,
}

impl Watcher {
    /// Creates a watcher whose resolved watermark starts at `initial_resolved`.
    /// `pub(crate)`: only the engine constructs one; callers obtain a clone via
    /// [`crate::Engine::watcher`].
    pub(crate) fn new(initial_resolved: u64) -> Self {
        let notify = Arc::new(Notify::new());
        // Bounded to keep memory in check under a slow subscriber. Dropped
        // events are detected by `subscribe`'s `RecvError::Lagged` and the
        // consumer re-synchronises from the resolved watermark.
        let (events, _) = broadcast::channel(4096);
        Self {
            notify,
            resolved: Arc::new(AtomicU64::new(initial_resolved)),
            events,
        }
    }

    /// Returns the most recent durably committed **and** applied `commit_ts`.
    #[must_use]
    pub fn resolved_ts(&self) -> u64 {
        self.resolved.load(Ordering::Acquire)
    }

    /// Returns `true` if any [`Watcher::subscribe`] consumer is currently
    /// attached. The engine uses this to skip the per-commit
    /// `Arc<[Mutation]>` allocation in [`Watcher::publish`], which is the
    /// common case in benchmarks / OLTP workloads that never open a CDC
    /// subscription.
    #[must_use]
    pub fn has_subscribers(&self) -> bool {
        self.events.receiver_count() > 0
    }

    /// Waits until the resolved watermark is `>= ts`, then returns it. Passing
    /// a `ts` in the future blocks until that commit is durable and applied;
    /// passing a `ts` that has already resolved returns immediately.
    pub async fn wait_for_resolved(&self, ts: u64) -> u64 {
        loop {
            let current = self.resolved_ts();
            if current >= ts {
                return current;
            }
            // `notify_waiters` (not `notify_one`) wakes every waiter, because
            // each waiter re-checks the shared watermark independently.
            self.notify.notified().await;
        }
    }

    /// Subscribes to the ordered commit event stream, starting at the next
    /// commit after subscription. Events may be dropped if the consumer lags
    /// past the bounded buffer (see the module-level note) — use
    /// [`Watcher::wait_for_resolved`] for the consistency guarantee.
    ///
    /// The returned stream owns its receiver (it does not borrow `self`), so it
    /// is `'static` and may outlive this watcher.
    pub fn subscribe(&self) -> impl futures::Stream<Item = WatchEvent> + use<> {
        // Reuses subscribe_with_errors, only filtering out Lagged errors (convenience path).
        self.subscribe_with_errors()
            .filter_map(|result| std::future::ready(result.ok()))
    }

    /// Like [`Watcher::subscribe`], but does **not** swallow `Lagged`: the
    /// stream yields `Err(RecvError::Lagged)` when the bounded buffer overflowed
    /// and events were dropped. Replication consumers must treat a `Lagged` as
    /// a gap and re-synchronise via a full snapshot (see
    /// `hatp_frontend::replica`) — silently ignoring it lets a slow replica
    /// diverge from the source with no self-healing (security-review MEDIUM).
    pub fn subscribe_with_errors(
        &self,
    ) -> impl futures::Stream<
        Item = Result<WatchEvent, tokio_stream::wrappers::errors::BroadcastStreamRecvError>,
    > + use<> {
        tokio_stream::wrappers::BroadcastStream::new(self.events.subscribe())
    }

    /// Publishes a commit to every waiter and subscriber. Called by the engine
    /// (sync path and group-commit worker) after a commit is durable **and**
    /// applied. The resolved watermark is monotonic, so a group-commit batch
    /// publishing out of `commit_seq` order is safe (`fetch_max`).
    pub(crate) fn publish(&self, tx_id: u64, commit_ts: u64, mutations: Arc<[Mutation]>) {
        // The fast path is the common one: a benchmark / OLTP workload has
        // no `Watcher::subscribe` consumers, so the broadcast send would
        // just return `Err(SendError)` and the `Arc<[Mutation]>` allocation
        // would be a pure waste. We still need to advance the resolved
        // watermark (cheap atomic) and notify any `wait_for_resolved` waiters
        // (one `notify_waiters` wake-up is unconditional), but the per-event
        // `Arc<[Mutation]>` allocation is only paid when somebody will read
        // it.
        if self.events.receiver_count() > 0 {
            self.resolved.fetch_max(commit_ts, Ordering::Release);
            let _ = self.events.send(WatchEvent {
                tx_id,
                commit_ts,
                mutations,
            });
        } else {
            // No broadcast subscribers. The resolved watermark has to advance
            // even when no events are emitted, so `wait_for_resolved` wakes
            // up correctly; the `Arc<[Mutation]>` is dropped here so the
            // engine didn't pay for an allocation no one will read.
            self.resolved.fetch_max(commit_ts, Ordering::Release);
        }
        self.notify.notify_waiters();
    }

    /// Publishes a commit without an attached mutation payload. Used by the
    /// engine fast path when no [`Watcher::subscribe`] consumer is attached
    /// (the common case in benchmarks / OLTP workloads). The resolved
    /// watermark still advances and `wait_for_resolved` waiters still wake,
    /// but the per-event `Arc<[Mutation]>` allocation is skipped.
    pub(crate) fn publish_empty(&self, tx_id: u64, commit_ts: u64) {
        self.resolved.fetch_max(commit_ts, Ordering::Release);
        let _ = tx_id;
        self.notify.notify_waiters();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::StreamExt;

    /// Publishes an empty-mutation commit (test helper).
    fn publish(watcher: &Watcher, tx_id: u64, commit_ts: u64) {
        watcher.publish(tx_id, commit_ts, Arc::new([]));
    }

    #[test]
    fn wait_for_resolved_returns_immediately_when_already_resolved() {
        let watcher = Watcher::new(4);
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let resolved = rt.block_on(watcher.wait_for_resolved(4));
        assert_eq!(resolved, 4);
    }

    #[test]
    fn wait_for_resolved_wakes_on_new_commit() {
        let watcher = Watcher::new(0);
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let waiter = watcher.clone();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let b = barrier.clone();
        let handle = std::thread::spawn(move || {
            // Notify the main thread that the waiter is ready
            b.wait();
            rt.block_on(waiter.wait_for_resolved(3))
        });
        // Wait for the waiter thread to be ready (Barrier sync, no sleep)
        barrier.wait();
        publish(&watcher, 10, 3);
        let resolved = handle.join().expect("waiter joins");
        assert!(
            resolved >= 3,
            "waiter must observe a watermark >= 3, got {resolved}"
        );
    }

    #[test]
    fn resolved_is_monotonic_under_out_of_order_publish() {
        let watcher = Watcher::new(0);
        publish(&watcher, 1, 5);
        publish(&watcher, 2, 3); // out of order: must not regress the watermark
        assert_eq!(watcher.resolved_ts(), 5);
    }

    #[test]
    fn subscribe_streams_events_in_order() {
        let watcher = Watcher::new(0);
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let mut stream = watcher.subscribe();
        publish(&watcher, 1, 1);
        publish(&watcher, 2, 2);
        let first = rt.block_on(stream.next()).expect("first event");
        let second = rt.block_on(stream.next()).expect("second event");
        assert_eq!(first.commit_ts, 1);
        assert_eq!(second.commit_ts, 2);
    }
}
