//! `hatp-engine` ↔ `hatp-tx` integration hook.
//!
//! This is the standard [`EngineHook`](hatp_engine::EngineHook) implementation that
//! [`TxManager`](crate::manager::TxManager) injects into [`hatp_engine::Engine`]:
//! after each `Engine::write` commit, it calls back
//! `TxManager::commit_at(TxnId(tx_id), tx_id)`.
//!
//! # Usage
//!
//! ```no_run
//! use std::sync::Arc;
//! use hatp_engine::{Engine, EngineConfig, EngineHook};
//! use hatp_tx::TxManagerHook;
//!
//! let manager = hatp_tx::TxManager::new();
//! let hook: Arc<dyn EngineHook> = Arc::new(TxManagerHook::new(manager.clone()));
//! let engine = Engine::open_with_hook(EngineConfig::new("/tmp/hatp"), hook)?;
//! # Ok::<(), hatp_engine::EngineError>(())
//! ```

use crate::manager::{TxManager, TxnId, TxnState};
use hatp_engine::{EngineError, EngineHook, Mutation};
use std::sync::Arc;

/// Transaction manager implementation of `EngineHook`.
///
/// After each engine commit, calls [`TxManager::commit_at`], passing the engine-assigned
/// `tx_id` as `commit_ts`, aligning the SSI history timestamps with the engine's monotonic
/// counter. Unknown ids return `UnknownTxn` but are not treated as errors — this is
/// idempotent.
///
/// Pre-commit hook ([`EngineHook::on_pre_commit`]) performs SSI conflict detection:
/// - Retrieves the active SSI context for `tx_id`;
/// - Writes all mutation keys into that context's `write_set`;
/// - Calls [`TxManager::validate_ssi`]; on failure, returns
///   `hatp_engine::EngineError::WriteConflict` or
///   `EngineError::ReadWriteConflict` to prevent the write from being persisted.
#[derive(Debug)]
pub struct TxManagerHook {
    manager: Arc<TxManager>,
}

impl TxManagerHook {
    /// Constructs a hook, incrementing the internal `Arc<TxManager>` reference count by 1.
    #[must_use]
    pub fn new(manager: Arc<TxManager>) -> Self {
        Self { manager }
    }

    /// Returns the underlying manager (for upper layers to call `begin` / `commit` directly).
    #[must_use]
    pub fn manager(&self) -> Arc<TxManager> {
        Arc::clone(&self.manager)
    }
}

impl EngineHook for TxManagerHook {
    fn on_pre_commit(&self, tx_id: u64, mutations: &[Mutation]) -> hatp_engine::Result<()> {
        // 1. No SSI context → non-SSI transaction, let the write proceed.
        let id = TxnId(tx_id);
        // 1b. Reject retries of transactions already aborted by SSI conflict: otherwise,
        //     after the context is removed, `get_ssi_txn` returns `None`, misclassifying
        //     this retry as a "non-SSI commit" and bypassing conflict detection
        //     (security-review HIGH: the other half of the symmetric rw-ring livelock).
        //     An id in the terminal Aborted state must never be committed. Use `WriteConflict`
        //     (not Corrupt) to report it, so `try_commit` maps it to a clean
        //     `SsiConflict`, consistent with the "conflict means abort" semantics.
        if let Ok(handle) = self.manager.get(id) {
            if handle.state == TxnState::Aborted {
                return Err(hatp_engine::EngineError::WriteConflict {
                    txn: tx_id,
                    key: Vec::new(),
                });
            }
        }
        if self.manager.get_ssi_txn(id).is_none() {
            return Ok(());
        }
        // 2. Collect mutation keys and complete record_write + validate atomically
        //    within `record_write_and_validate` (same `ssi_contexts` lock, eliminating
        //    R-04 TOCTOU, and validate_cahill no longer nests locks — PR 3.9).
        let keys: Vec<bytes::Bytes> = mutations
            .iter()
            .map(|mutation| match mutation {
                Mutation::Put { key, .. } => key.clone(),
                Mutation::Delete { key } => key.clone(),
            })
            .collect();
        match self.manager.record_write_and_validate(id, &keys) {
            Ok(()) => Ok(()),
            // The context may have vanished between the `is_none` probe and the
            // lock (a concurrent abort): the commit is no longer an SSI commit.
            Err(crate::error::TxError::UnknownTxn { .. }) => Ok(()),
            Err(other) => Err(map_tx_error(other)),
        }
    }

    fn on_tx_commit(&self, tx_id: u64, commit_ts: u64) {
        // The `commit_ts` from the engine is a commit sequence number (strictly
        // increasing), decoupled from the id assigned by `TxManager` (reservation
        // order). `commit_at` uses this `commit_ts` to record SSI history, so that
        // antidependency detection (`entry.commit_ts > txn.start_ts`) reflects the
        // real commit order — using `tx_id` would miss read-write conflicts under
        // out-of-order commits. Unknown / committed ids go through `commit_at`'s
        // idempotent path.
        let _ = self.manager.commit_at(TxnId(tx_id), commit_ts);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Maps a [`crate::error::TxError`] into the engine's [`EngineError`] surface.
fn map_tx_error(err: crate::error::TxError) -> EngineError {
    match err {
        crate::error::TxError::WriteConflict { txn, key } => {
            EngineError::WriteConflict { txn: txn.0, key }
        }
        crate::error::TxError::ReadWriteConflict { txn, key } => {
            EngineError::ReadWriteConflict { txn: txn.0, key }
        }
        // Other errors (InvalidState / UnknownTxn) are treated as corrupt — they should
        // not appear on the active SSI path; falling through to corrupt prompts upper
        // layers to investigate.
        crate::error::TxError::InvalidState { .. } | crate::error::TxError::UnknownTxn { .. } => {
            EngineError::Corrupt("ssi pre-commit validation failed")
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;
    use crate::manager::IsolationHint;
    use hatp_engine::{Engine, EngineConfig};

    #[test]
    fn hook_does_not_panic_on_unknown_id() {
        let tmp = std::env::temp_dir().join(format!(
            "hatp-tx-hook-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&tmp).unwrap();

        let manager = TxManager::new();
        let hook: Arc<dyn EngineHook> = Arc::new(TxManagerHook::new(manager.clone()));
        let engine = Engine::open_with_hook(EngineConfig::new(&tmp), hook).expect("open engine");

        let _tx = engine
            .write(&[hatp_engine::Mutation::Put {
                key: bytes::Bytes::from_static(b"k"),
                value: bytes::Bytes::from_static(b"v"),
            }])
            .expect("write");

        // Explicit begin/commit still works independently (id provided by caller,
        // commit_ts passed explicitly):
        let handle = manager.begin_with_id(TxnId(999), IsolationHint::Snapshot);
        let committed = manager.commit_at(handle.id, 1).expect("commit");
        assert!(matches!(
            committed.state,
            crate::manager::TxnState::Committed
        ));

        std::fs::remove_dir_all(&tmp).ok();
    }
}