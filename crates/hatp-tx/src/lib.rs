//! HatP transaction layer — single-writer state machine + `EngineHook`.

#![doc(html_root_url = "https://docs.rs/hatp-tx/0.1.0")]

/// Common error type and result alias for the transaction layer.
pub mod error;

/// Transaction manager — owns the global sequence and the inflight set.
pub mod manager;

/// `hatp-engine` ↔ `hatp-tx` integration hook.
pub mod hook;

pub use error::{Result, TxError};
pub use hook::TxManagerHook;
pub use manager::{IsolationHint, TxManager, TxnHandle, TxnId, TxnState};
