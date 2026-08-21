//! HatP — embedded HTAP engine facade.
//!
//! Top-level glue crate that re-exports the local sub-crates and exposes
//! the [`embedded::Database`] facade. The OLTP write path lives in
//! [`hatp_engine`]; the OLAP read path lives in [`hatp_frontend`]
//! (DataFusion + Vortex + Arrow). This crate wires them together and
//! provides synchronous `put` / `get` / `delete` / `scan` / `execute_sql`
//! entry points.

#![doc(html_root_url = "https://docs.rs/hatp/0.1.0")]

/// Re-export of the frontend types so downstream consumers can use
/// `hatp::frontend::*` and friends.
pub use hatp_frontend as frontend;

/// Re-export of the OLTP engine types.
pub use hatp_engine as engine;

/// Re-export of the transaction layer.
pub use hatp_tx as tx;

/// Durable database facade and buffered transaction API.
pub use embedded::{Database, DatabaseError, Transaction};

/// Library entry point that boots an embedded database.
pub mod embedded;


