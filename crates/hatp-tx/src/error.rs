//! Error types for the transaction layer.

use std::fmt;

/// Errors produced by the transaction layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TxError {
    /// Operation attempted on a transaction that is no longer in the
    /// expected state.
    InvalidState {
        /// Identifier of the transaction.
        txn: crate::manager::TxnId,
        /// Expected state (caller-facing, never localized).
        expected: crate::manager::TxnState,
        /// Observed state.
        actual: crate::manager::TxnState,
    },
    /// Unknown transaction identifier.
    UnknownTxn {
        /// Identifier that did not match any live transaction.
        txn: crate::manager::TxnId,
    },
    /// Write-write conflict detected during SSI validation.
    WriteConflict {
        /// Transaction that detected the conflict.
        txn: crate::manager::TxnId,
        /// Key that caused the conflict.
        key: Vec<u8>,
    },
    /// Read-write antidependency detected during SSI validation: a peer
    /// committed a write to a key this transaction previously read, after
    /// this transaction began. Without this check, write-skew anomalies
    /// pass validation silently under SI semantics.
    ReadWriteConflict {
        /// Transaction that detected the conflict.
        txn: crate::manager::TxnId,
        /// Key that was read by this txn and concurrently written by a peer.
        key: Vec<u8>,
    },
}

impl fmt::Display for TxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TxError::InvalidState {
                txn,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "transaction {txn:?} is in {actual:?}, expected {expected:?}"
                )
            }
            TxError::UnknownTxn { txn } => {
                write!(f, "unknown transaction {txn:?}")
            }
            TxError::WriteConflict { txn, key } => {
                write!(
                    f,
                    "SSI write-write conflict in transaction {txn:?} on key {:x?}",
                    key
                )
            }
            TxError::ReadWriteConflict { txn, key } => {
                write!(
                    f,
                    "SSI read-write conflict in transaction {txn:?} on key {:x?}",
                    key
                )
            }
        }
    }
}

impl std::error::Error for TxError {}

/// Convenience alias for fallible transaction operations.
pub type Result<T> = std::result::Result<T, TxError>;
