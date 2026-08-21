//! In-memory mirror verification — similar to FoundationDB ApiCorrectness's MemoryKeyValueStore
//!
//! Maintains a simple in-memory KV store, executing the same mutations in parallel
//! with engine operations, then compares item by item to verify engine state
//! matches the in-memory mirror.
//!
//! Reference: FoundationDB `ApiCorrectness::MemoryKeyValueStore`
//!            RocksDB `ASSERT_EQ(Get("key"), "expected")`

use bytes::Bytes;
use std::collections::BTreeMap;

/// In-memory mirror KV store — used to verify engine state
///
/// Every engine operation is synchronously applied to this mirror, then the
/// two results are compared. This is the core mechanism for "verifying state"
/// (as opposed to "verifying interaction").
#[derive(Debug, Clone, Default)]
pub struct MirrorStore {
    /// Latest version of values, sorted by key
    entries: BTreeMap<Bytes, Option<Bytes>>,
}

impl MirrorStore {
    /// Create an empty mirror
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or update a key
    pub fn put(&mut self, key: Bytes, value: Bytes) {
        self.entries.insert(key, Some(value));
    }

    /// Delete a key (write a tombstone)
    pub fn delete(&mut self, key: Bytes) {
        self.entries.insert(key, None);
    }

    /// Read the value of a key
    #[must_use]
    pub fn get(&self, key: &[u8]) -> Option<Option<Bytes>> {
        self.entries.get(key).cloned()
    }

    /// Range scan `[lower, upper)`, returning `(key, value)` pairs
    #[must_use]
    pub fn scan_range(&self, lower: &[u8], upper: &[u8]) -> Vec<(Bytes, Bytes)> {
        self.entries
            .range(Bytes::copy_from_slice(lower)..Bytes::copy_from_slice(upper))
            .filter_map(|(k, v)| v.as_ref().map(|val| (k.clone(), val.clone())))
            .collect()
    }

    /// Return the number of keys
    #[must_use]
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the mirror is empty
    #[must_use]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Compare engine state against mirror state, returning a list of differences
///
/// This is the foundation for precise assertions: each difference is enumerated,
/// not a single big snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub struct StateDiff {
    /// Keys present in the mirror but missing from the engine
    pub missing_in_engine: Vec<Bytes>,
    /// Keys returned by the engine but absent from the mirror
    pub extra_in_engine: Vec<Bytes>,
    /// Keys with mismatched values
    pub value_mismatches: Vec<(Bytes, Option<Bytes>, Option<Bytes>)>,
}

impl StateDiff {
    /// Whether the states are consistent
    #[must_use]
    #[allow(dead_code)]
    pub fn is_consistent(&self) -> bool {
        self.missing_in_engine.is_empty()
            && self.extra_in_engine.is_empty()
            && self.value_mismatches.is_empty()
    }

    /// Format a human-readable difference report
    #[must_use]
    #[allow(dead_code)]
    pub fn report(&self) -> String {
        let mut lines = Vec::new();
        if !self.missing_in_engine.is_empty() {
            lines.push(format!(
                "{} keys missing in engine: {:?}",
                self.missing_in_engine.len(),
                &self.missing_in_engine[..self.missing_in_engine.len().min(10)]
            ));
        }
        if !self.extra_in_engine.is_empty() {
            lines.push(format!(
                "{} extra keys in engine: {:?}",
                self.extra_in_engine.len(),
                &self.extra_in_engine[..self.extra_in_engine.len().min(10)]
            ));
        }
        if !self.value_mismatches.is_empty() {
            lines.push(format!(
                "{} value mismatches: {:?}",
                self.value_mismatches.len(),
                &self.value_mismatches[..self.value_mismatches.len().min(10)]
            ));
        }
        if lines.is_empty() {
            lines.push("State is consistent".to_string());
        }
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirror_put_get_round_trip() {
        let mut mirror = MirrorStore::new();
        mirror.put(Bytes::from_static(b"k"), Bytes::from_static(b"v"));
        assert_eq!(mirror.get(b"k"), Some(Some(Bytes::from_static(b"v"))));
        assert_eq!(mirror.get(b"nonexistent"), None);
    }

    #[test]
    fn mirror_delete_then_get_is_none() {
        let mut mirror = MirrorStore::new();
        mirror.put(Bytes::from_static(b"k"), Bytes::from_static(b"v"));
        mirror.delete(Bytes::from_static(b"k"));
        assert_eq!(mirror.get(b"k"), Some(None));
    }

    #[test]
    fn mirror_scan_range_respects_bounds() {
        let mut mirror = MirrorStore::new();
        mirror.put(Bytes::from_static(b"a"), Bytes::from_static(b"1"));
        mirror.put(Bytes::from_static(b"b"), Bytes::from_static(b"2"));
        mirror.put(Bytes::from_static(b"c"), Bytes::from_static(b"3"));
        let scan = mirror.scan_range(b"a", b"c");
        assert_eq!(scan.len(), 2);
        assert_eq!(scan[0], (Bytes::from_static(b"a"), Bytes::from_static(b"1")));
        assert_eq!(scan[1], (Bytes::from_static(b"b"), Bytes::from_static(b"2")));
    }
}