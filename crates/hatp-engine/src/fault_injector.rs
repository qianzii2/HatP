//! Fault injection framework — usable in production code, zero overhead in non-test builds.
//!
//! # Usage
//!
//! 1. Insert `fault_point!("name")` macros in production code
//! 2. Register callbacks through `FaultInjector` (in tests/common/) during tests
//! 3. In non-test builds, `fault_point!` expands to a no-op
//!
//! # References
//! - RocksDB `TEST_SYNC_POINT` macro
//! - SQLite `sqlite3_io_error_hit` global variable
//! - TigerBeetle `FaultAtlas` declarative fault injection

/// Fault injection action types.
#[derive(Debug, Clone)]
pub enum FaultAction {
    /// Do nothing, continue normally.
    None,
    /// Return an I/O error.
    IoError(std::io::ErrorKind),
    /// Return an error message.
    Error(String),
    /// Pause (for synchronizing concurrent tests).
    Pause,
    /// Trigger the fault on the Nth invocation, then recover.
    FailOnNth(u64, Box<FaultAction>),
}

/// Insert a fault point in a hot path.
///
/// When fault injection is not enabled (non-test builds), this is a zero-overhead no-op.
/// In test builds, it calls the `trigger` function to query the global registry.
#[macro_export]
macro_rules! fault_point {
    ($name:expr) => {
        #[cfg(test)]
        {
            let _ = $crate::fault_injector::trigger($name);
        }
    };
}

/// Triggers the fault point with the specified name.
///
/// Only has effect in test builds. Returns `None` to indicate normal pass-through;
/// returns `Some(action)` indicating the injected action.
#[must_use]
pub fn trigger(name: &str) -> Option<FaultAction> {
    crate::fault_injector::registry::with_registry(|registry| {
        if !registry.enabled {
            return None;
        }
        let count = registry.hit_counts.entry(name.to_string()).or_insert(0);
        *count += 1;
        if let Some(callback) = registry.callbacks.get_mut(name) {
            Some(callback())
        } else {
            None
        }
    })
}

/// Gets the number of times the specified fault point has been triggered.
#[must_use]
pub fn hit_count(name: &str) -> u64 {
    crate::fault_injector::registry::with_registry(|registry| {
        registry.hit_counts.get(name).copied().unwrap_or(0)
    })
}

// ── Internal registry (test-only) ──────────────────────────────────────────────

pub(crate) mod registry {
    use super::FaultAction;
    use std::collections::HashMap;
    use std::sync::Mutex;

    pub struct FaultRegistry {
        pub callbacks: HashMap<String, Box<dyn FnMut() -> FaultAction + Send>>,
        pub hit_counts: HashMap<String, u64>,
        pub enabled: bool,
    }

    impl FaultRegistry {
        pub fn new() -> Self {
            Self {
                callbacks: HashMap::new(),
                hit_counts: HashMap::new(),
                enabled: false,
            }
        }
    }

    static FAULT_REGISTRY: std::sync::LazyLock<Mutex<FaultRegistry>> =
        std::sync::LazyLock::new(|| Mutex::new(FaultRegistry::new()));

    pub fn with_registry<R>(f: impl FnOnce(&mut FaultRegistry) -> R) -> R {
        let mut registry = FAULT_REGISTRY.lock().expect("fault registry lock");
        f(&mut registry)
    }
}

// ── Test helper: FaultInjector guard ───────────────────────────────────────────

/// Fault injection guard — enables fault injection within its scope and auto-restores
/// on drop.
#[must_use = "FaultInjector guard auto-restores on drop; bind to a variable to keep it active"]
#[derive(Debug)]
pub struct FaultInjector {
    _private: (),
}

impl FaultInjector {
    /// Enables fault injection. Returns a guard that auto-disables on drop.
    pub fn enable() -> Self {
        registry::with_registry(|r| {
            r.enabled = true;
            r.callbacks.clear();
            r.hit_counts.clear();
        });
        Self { _private: () }
    }

    /// Sets a fault point callback.
    pub fn set_callback<F>(name: &str, callback: F)
    where
        F: FnMut() -> FaultAction + Send + 'static,
    {
        registry::with_registry(|r| {
            r.callbacks.insert(name.to_string(), Box::new(callback));
        });
    }

    /// Sets the "fail on Nth invocation" pattern.
    ///
    /// References SQLite `do_ioerr_test`: retry in a loop, making the Nth call fail
    /// each time.
    pub fn set_fail_on_nth(name: &str, n: u64, action: FaultAction) {
        let mut count = 0_u64;
        Self::set_callback(name, move || {
            count += 1;
            if count == n {
                action.clone()
            } else {
                FaultAction::None
            }
        });
    }
}

#[cfg(test)]
impl Drop for FaultInjector {
    fn drop(&mut self) {
        registry::with_registry(|r| {
            r.enabled = false;
            r.callbacks.clear();
            r.hit_counts.clear();
        });
    }
}