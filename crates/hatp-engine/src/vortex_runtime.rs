//! Shared Vortex runtime helpers used by the SST reader.
//!
//! Provides a single-threaded smol-based [`CurrentThreadRuntime`] to
//! drive Vortex's blocking API without conflicting with a Tokio runtime that
//! DataFusion may have spun up.

use std::sync::OnceLock;
use vortex_io::runtime::current::CurrentThreadRuntime;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;
use vortex::VortexSessionDefault;

/// Returns a shared single-threaded smol runtime.  Initialised once per process.
pub(crate) fn runtime() -> &'static CurrentThreadRuntime {
    static CELL: OnceLock<CurrentThreadRuntime> = OnceLock::new();
    CELL.get_or_init(CurrentThreadRuntime::new)
}

/// Builds a [`VortexSession`] pre-configured with the shared runtime handle.
pub(crate) fn new_session() -> VortexSession {
    VortexSession::default().with_handle(runtime().handle())
}