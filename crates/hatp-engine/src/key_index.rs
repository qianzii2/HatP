//! Per-SST key index sidecar — O(log n) point-lookup without Vortex (PR 5.0).
//!
//! # Motivation
//!
//! `Engine::get` currently opens the Vortex file, builds a filter expression,
//! and runs `scan().with_filter(filter)` — a full scan path for a single-key
//! lookup. Vortex is columnar, so even a "filtered scan" touches metadata for
//! every column chunk. For an OLTP point query this is heavyweight.
//!
//! The key index is a sorted, self-describing binary sidecar that stores every
//! key plus its MVCC version metadata. A point lookup is a single binary search
//! over the key array — no file open, no Vortex session, no columnar decode.
//!
//! # Format
//!
//! ```text
//!   magic         [u8; 4]   "KIDX"
//!   key_count     u32  LE
//!   for each key:
//!     key_len     u32  LE
//!     key_bytes   [u8; key_len]
//!     value_len   u32  LE   (0xFFFF_FFFF = tombstone)
//!     value_bytes [u8; value_len]  (omitted when tombstone)
//!     begin_ts    u64  LE
//!     end_ts      u64  LE
//! ```
//!
//! Keys are in ascending dictionary order (the flush/compaction input is
//! already sorted by key). The index is a derived cache — missing/corrupt
//! sidecar falls back to the authoritative Vortex read, exactly like the
//! bloom filter.
//!
//! # Why not a B+Tree / hash index
//!
//! A binary search over a flat sorted array is cache-friendly, zero-overhead,
//! and the entire index is typically a single `mmap` or `read` away. For
//! SST files with 10k–100k keys (typical OLTP flush size), `log2(100k) ≈ 17`
//! string comparisons is < 1 µs. A B+Tree with internal nodes adds pointer
//! chasing; a hash index loses range-scan support (which this module
//! intentionally does not provide — range scans still go through Vortex).
//!
//! The byte-level slicing is bounds-checked by explicit `data.len()` guards
//! before every read or by the precomputed `offsets` array; `.get(..)` would
//! only re-introduce the same checks. Same pattern as `wal.rs` and `crc32c.rs`.
#![allow(clippy::indexing_slicing)]

use crate::version::VersionedValue;
use crate::Result;
use bytes::Bytes;
use std::path::Path;

/// File-format magic.
const MAGIC: [u8; 4] = *b"KIDX";

/// Sentinel `value_len` that means "tombstone" (mirrors WAL's `VALUE_NONE`).
const VALUE_NONE: u32 = 0xFFFF_FFFF;

/// Fixed per-entry overhead: key_len(4) + value_len(4) + begin_ts(8) + end_ts(8).
/// In-memory representation of a key index, suitable for binary search.
#[derive(Debug, Clone)]
pub struct KeyIndex {
    /// Raw bytes of the entire index file (mmap-friendly, zero-copy reads).
    data: Bytes,
    /// Number of entries.
    count: u32,
    /// Byte offsets of each entry within `data` (precomputed at load time
    /// so binary search is O(log n) rather than O(n log n)).
    offsets: Vec<u32>,
}

impl KeyIndex {
    // ── construction ────────────────────────────────────────────────────

    /// Builds a key index from sorted rows. The caller must ensure `rows` is
    /// sorted by key ascending (flush/compaction already guarantees this).
    #[must_use]
    pub fn from_rows(rows: &[crate::memtable::VersionedRow]) -> Self {
        let count = rows.len() as u32;
        let mut offsets = Vec::with_capacity(rows.len());
        // Estimate: each key averages ~32 bytes, values ~64 bytes, overhead 24.
        let estimated = rows.len() * 120 + 8;
        let mut data = Vec::with_capacity(estimated);
        data.extend_from_slice(&MAGIC);
        data.extend_from_slice(&count.to_le_bytes());
        for row in rows {
            let key = &row.key;
            let key_len = u32::try_from(key.len()).unwrap_or(u32::MAX);
            offsets.push(data.len() as u32);
            data.extend_from_slice(&key_len.to_le_bytes());
            data.extend_from_slice(key);
            let (value_len, value_bytes) = match &row.version.value {
                Some(v) => {
                    let len = u32::try_from(v.len()).unwrap_or(u32::MAX);
                    (len, v.as_ref())
                }
                None => (VALUE_NONE, &[][..]),
            };
            data.extend_from_slice(&value_len.to_le_bytes());
            if value_len != VALUE_NONE {
                data.extend_from_slice(value_bytes);
            }
            data.extend_from_slice(&row.version.begin_ts.to_le_bytes());
            data.extend_from_slice(&row.version.end_ts.to_le_bytes());
        }
        Self {
            data: Bytes::from(data),
            count,
            offsets,
        }
    }

    // ── I/O ─────────────────────────────────────────────────────────────

    /// Writes the index to `path` (atomic enough for a derived cache).
    pub fn write_to(&self, path: &Path) -> Result<()> {
        std::fs::write(path, self.data.as_ref())?;
        Ok(())
    }

    /// Reads and decodes an index from `path`, returning `None` when the file
    /// is missing or malformed.
    #[must_use]
    pub fn read_from(path: &Path) -> Option<Self> {
        let data = Bytes::from(crate::mmap_io::mmap_read_to_vec(path).ok()?);
        Self::from_bytes(&data)
    }

    /// Validates internal invariants (SQLite pattern: fuzz decode → assert_invariants).
    /// Returns `true` if offsets, count, and key ordering are consistent.
    #[must_use]
    pub fn assert_invariants(&self) -> bool {
        if self.count == 0 {
            return self.offsets.is_empty();
        }
        if self.offsets.len() != self.count as usize {
            return false;
        }
        // Verify keys are in ascending order.
        let mut last_key: Option<&[u8]> = None;
        for &offset in &self.offsets {
            let entry_start = offset as usize;
            if entry_start + 4 > self.data.len() {
                return false;
            }
            let key_len = unsafe {
                let p = self.data.as_ptr().add(entry_start) as *const u32;
                u32::from_le(p.read_unaligned()) as usize
            };
            if entry_start + 4 + key_len > self.data.len() {
                return false;
            }
            let key = unsafe { self.data.get_unchecked(entry_start + 4..entry_start + 4 + key_len) };
            if let Some(last) = last_key {
                if key < last {
                    return false;
                }
            }
            last_key = Some(key);
        }
        true
    }

    /// Parses raw bytes into a KeyIndex. Public for fuzz testing.
    #[must_use]
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 8 {
            return None;
        }
        if data[..4] != MAGIC {
            return None;
        }
        let count = u32::from_le_bytes(data[4..8].try_into().ok()?);
        let mut offsets = Vec::with_capacity(count as usize);
        let mut cursor = 8usize;
        let raw = data;
        let raw_len = raw.len();
        for _ in 0..count {
            if cursor + 4 > raw_len { return None; }
            offsets.push(cursor as u32);
            let key_len = unsafe {
                let p = raw.as_ptr().add(cursor) as *const u32;
                u32::from_le(p.read_unaligned()) as usize
            };
            cursor += 4 + key_len;
            if cursor + 4 > raw_len { return None; }
            let value_len = unsafe {
                let p = raw.as_ptr().add(cursor) as *const u32;
                u32::from_le(p.read_unaligned()) as usize
            };
            cursor += 4;
            if value_len != VALUE_NONE as usize { cursor += value_len; }
            cursor += 16;
            if cursor > raw_len { return None; }
        }
        Some(Self { data: Bytes::copy_from_slice(data), count, offsets })
    }

    // ── lookup ──────────────────────────────────────────────────────────

    /// Binary search for `key`. Returns the MVCC version if found, `None`
    /// otherwise. The caller is responsible for visibility filtering
    /// (comparing `begin_ts` / `end_ts` against its snapshot).
    #[must_use]
    pub fn get(&self, key: &[u8]) -> Option<VersionedValue> {
        if self.count == 0 {
            return None;
        }
        // Binary search over the precomputed offset array. O(log n).
        // Inline the key comparison to avoid `Bytes::copy_from_slice`
        // allocation on every probe — compare raw slices directly.
        let mut lo = 0u32;
        let mut hi = self.count;
        let data = self.data.as_ref();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let entry_start = self.offsets[mid as usize] as usize;
            // SAFETY: offsets are validated at load time; entry_start + 4
            // is within data bounds.
            let key_len = unsafe {
                let p = data.as_ptr().add(entry_start) as *const u32;
                u32::from_le(p.read_unaligned()) as usize
            };
            let entry_key = unsafe {
                data.get_unchecked(entry_start + 4..entry_start + 4 + key_len)
            };
            match entry_key.cmp(key) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => {
                    return Some(self.version_at(entry_start));
                }
            }
        }
        None
    }

    /// Reads the full version at `entry_start`.
    fn version_at(&self, entry_start: usize) -> VersionedValue {
        let data = self.data.as_ref();
        // SAFETY: offsets are validated at load time; all reads are in-bounds.
        let key_len = unsafe {
            let p = data.as_ptr().add(entry_start) as *const u32;
            u32::from_le(p.read_unaligned()) as usize
        };
        let value_cursor = entry_start + 4 + key_len;
        let value_len = unsafe {
            let p = data.as_ptr().add(value_cursor) as *const u32;
            u32::from_le(p.read_unaligned())
        };
        let value = if value_len == VALUE_NONE {
            None
        } else {
            let v_start = value_cursor + 4;
            Some(Bytes::copy_from_slice(unsafe {
                data.get_unchecked(v_start..v_start + value_len as usize)
            }))
        };
        let ts_offset = value_cursor
            + 4
            + if value_len == VALUE_NONE {
                0
            } else {
                value_len as usize
            };
        let begin_ts = unsafe {
            let p = data.as_ptr().add(ts_offset) as *const u64;
            u64::from_le(p.read_unaligned())
        };
        let end_ts = unsafe {
            let p = data.as_ptr().add(ts_offset + 8) as *const u64;
            u64::from_le(p.read_unaligned())
        };
        VersionedValue {
            value,
            begin_ts,
            end_ts,
            tx_id: 0, // not stored in SST
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::memtable::VersionedRow;
    use crate::version::OPEN_ENDED_TS;

    fn row(key: &[u8], value: &[u8], begin_ts: u64, end_ts: u64) -> VersionedRow {
        VersionedRow {
            key: Bytes::copy_from_slice(key),
            version: VersionedValue {
                value: Some(Bytes::copy_from_slice(value)),
                begin_ts,
                end_ts,
                tx_id: 0,
            },
        }
    }

    fn tombstone_row(key: &[u8], begin_ts: u64) -> VersionedRow {
        VersionedRow {
            key: Bytes::copy_from_slice(key),
            version: VersionedValue {
                value: None,
                begin_ts,
                end_ts: OPEN_ENDED_TS,
                tx_id: 0,
            },
        }
    }

    #[test]
    fn round_trip_single_key() {
        let rows = vec![row(b"hello", b"world", 1, OPEN_ENDED_TS)];
        let index = KeyIndex::from_rows(&rows);
        let found = index.get(b"hello").expect("found");
        assert_eq!(found.value.as_deref(), Some(b"world".as_ref()));
        assert_eq!(found.begin_ts, 1);
    }

    #[test]
    fn round_trip_tombstone() {
        let rows = vec![tombstone_row(b"deleted", 5)];
        let index = KeyIndex::from_rows(&rows);
        let found = index.get(b"deleted").expect("found tombstone");
        assert!(found.value.is_none());
        assert_eq!(found.begin_ts, 5);
    }

    #[test]
    fn binary_search_multiple_keys() {
        let mut rows: Vec<_> = (0..1000u64)
            .map(|i| {
                let key = format!("key-{i:05}");
                let value = format!("value-{i:05}");
                row(key.as_bytes(), value.as_bytes(), i, OPEN_ENDED_TS)
            })
            .collect();
        rows.sort_by(|a, b| a.key.cmp(&b.key));
        let index = KeyIndex::from_rows(&rows);

        for i in 0..1000u64 {
            let key = format!("key-{i:05}");
            let found = index.get(key.as_bytes()).expect("found");
            let expected = format!("value-{i:05}");
            assert_eq!(found.value.as_deref(), Some(expected.as_bytes()));
        }
        assert!(index.get(b"key-99999").is_none());
        assert!(index.get(b"nonexistent").is_none());
    }

    #[test]
    fn file_round_trip() {
        use tempfile::Builder;
        let dir = Builder::new()
            .prefix("hatp-kidx-")
            .tempdir()
            .expect("tempdir");
        let path = dir.path().join("test.kidx");

        let rows = vec![
            row(b"alpha", b"1", 1, OPEN_ENDED_TS),
            row(b"beta", b"2", 2, OPEN_ENDED_TS),
            row(b"gamma", b"3", 3, OPEN_ENDED_TS),
        ];
        let index = KeyIndex::from_rows(&rows);
        index.write_to(&path).expect("write");

        let loaded = KeyIndex::read_from(&path).expect("read");
        assert_eq!(loaded.count, 3);
        assert_eq!(
            loaded.get(b"beta").and_then(|v| v.value).as_deref(),
            Some(b"2".as_ref())
        );
    }

    #[test]
    fn rejects_corrupt_input() {
        assert!(KeyIndex::read_from(std::path::Path::new("/nonexistent")).is_none());
    }

    #[test]
    fn invariants_hold_for_valid_index() {
        let rows = vec![row(b"alpha", b"1", 1, OPEN_ENDED_TS), row(b"beta", b"2", 2, OPEN_ENDED_TS)];
        let index = KeyIndex::from_rows(&rows);
        assert!(index.assert_invariants());
    }

    #[test]
    fn invariants_reject_corrupt_count() {
        let rows = vec![row(b"k", b"v", 1, OPEN_ENDED_TS)];
        let mut index = KeyIndex::from_rows(&rows);
        index.count = 0;
        assert!(!index.assert_invariants());
    }
}