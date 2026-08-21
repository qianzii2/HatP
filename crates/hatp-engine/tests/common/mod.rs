//! Deterministic testing infrastructure — HatP test framework
//!
//! Design principles (from reference projects):
//! - Deterministic PRNG (FoundationDB deterministicRandom / TigerBeetle stdx.PRNG)
//! - In-memory mirror verification (FoundationDB ApiCorrectness MemoryKeyValueStore)
//! - Fault injection (RocksDB SyncPoint / SQLite do_ioerr_test)
//! - Boundary value generation (SQLite boundary1.test)
//! - Precise assertion + negative assertion (all reference projects)
//!
//! These modules are test helper utilities; no dependency on third-party test frameworks.

pub mod boundaries;
pub mod deterministic;
pub mod mirror;