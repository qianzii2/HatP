//! Shared types, constants, and codecs for HatP crates.
//!
//! This crate contains fundamental type aliases and constants used
//! across multiple HatP crates to avoid duplication and circular
//! dependencies, plus the [`codec`] module that centralizes the
//! Arrow ↔ `ScalarValue` ↔ bytes encoding shared by `hatp-engine`
//! and `hatp-frontend`.

pub mod codec;

/// Monotonically increasing transaction timestamp.
///
/// Both `hatp-engine` and `hatp-tx` use this alias for `u64` to
/// document the semantic meaning of timestamp fields in their MVCC
/// and transaction-management implementations.
pub type TxnTs = u64;

/// Sentinel used as the open end of the newest version's visibility range.
///
/// A `VersionChain` entry with `end_ts = OPEN_ENDED_TS` is visible to
/// all transactions whose `snapshot_ts >= begin_ts`.
pub const OPEN_ENDED_TS: TxnTs = u64::MAX;
