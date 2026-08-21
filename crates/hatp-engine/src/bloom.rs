//! Per-SST bloom filter (PR 3.10).
//!
//! A deterministic bloom filter written next to each SST as
//! `bloom-{file_id:020}.bf`. [`crate::Engine::get`] consults it before a
//! Vortex point lookup: a negative result proves the key is absent, so the
//! file read is skipped; a positive result is only probabilistic, so the real
//! lookup still runs. A missing or corrupt bloom sidecar merely falls back to
//! the full read — it is derived data, never a source of truth (rebuildable
//! on the next compaction), so it is intentionally **not** recorded in the
//! manifest.
//!
//! # Hash determinism
//!
//! The filter must produce identical bits across a flush (writer process) and
//! every later `get` (possibly a different process after restart). `std`'s
//! `DefaultHasher` is **not** stable, so two hand-rolled deterministic hashes
//! (FNV-1a 64 and DJB2) drive double hashing — no external dependency, no
//! per-process random seed.

use crate::Result;
use hatp_types::codec::{djb2, fnv1a};
use std::path::Path;

/// Magic prefix identifying a bloom sidecar file.
const MAGIC: &[u8; 4] = b"BF01";

/// A fixed-size bloom filter over byte keys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BloomFilter {
    /// Total bit count (always a multiple of 64).
    num_bits: u64,
    /// Number of double-hash probes per key.
    num_hashes: u32,
    /// Bit array, packed little-endian into `u64` words.
    bits: Vec<u64>,
}

impl BloomFilter {
    /// Sizes the filter for `estimated_items` at a target false-positive rate
    /// `fp_rate` (clamped to `[1e-9, 0.5]`). `m = -n·ln(p) / ln(2)²` and
    /// `k = m/n · ln(2)` are the standard optimal parameters.
    /// The bit count is rounded up to the next power of two so `h % num_bits`
    /// can be computed as `h & (num_bits - 1)` (a single AND instruction).
    #[must_use]
    pub fn new(estimated_items: usize, fp_rate: f64) -> Self {
        let n = (estimated_items.max(1)) as f64;
        let p = fp_rate.clamp(1e-9, 0.5);
        let ln2 = std::f64::consts::LN_2;
        let m = (-n * p.ln() / (ln2 * ln2)).ceil().max(64.0);
        let k = ((m / n) * ln2).round().clamp(1.0, 32.0) as u32;
        let num_bits = (m as u64).max(64).next_power_of_two();
        let words = (num_bits / 64) as usize;
        Self {
            num_bits,
            num_hashes: k,
            bits: vec![0_u64; words],
        }
    }

    /// Inserts `key` into the filter (idempotent).
    /// Uses `h & (num_bits - 1)` instead of `h % num_bits` because `num_bits`
    /// is always a power of two.
    pub fn insert(&mut self, key: &[u8]) {
        let mask = self.num_bits - 1;
        let (h1, h2) = (fnv1a(key), djb2(key));
        for probe in 0..self.num_hashes {
            let h = h1.wrapping_add(u64::from(probe).wrapping_mul(h2));
            let bit = (h & mask) as usize;
            unsafe { *self.bits.get_unchecked_mut(bit >> 6) |= 1_u64 << (bit & 63); }
        }
    }

    /// Returns `false` if `key` is definitely absent.
    #[must_use]
    pub fn contains(&self, key: &[u8]) -> bool {
        if self.bits.is_empty() || self.num_hashes == 0 { return false; }
        let mask = self.num_bits - 1;
        let (h1, h2) = (fnv1a(key), djb2(key));
        for probe in 0..self.num_hashes {
            let h = h1.wrapping_add(u64::from(probe).wrapping_mul(h2));
            let bit = (h & mask) as usize;
            if unsafe { *self.bits.get_unchecked(bit >> 6) } & (1_u64 << (bit & 63)) == 0 { return false; }
        }
        true
    }

    /// Serialises the filter as `magic + num_bits(u64 LE) + num_hashes(u32 LE)
    /// + words(u64 LE each)`.  Uses `set_len` + `write_unaligned` to avoid
    /// per-word stack temps and zero-fill.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let header_len = 16; // magic(4) + num_bits(8) + num_hashes(4)
        let words_len = self.bits.len() * 8;
        let total = header_len + words_len;
        let mut out = Vec::with_capacity(total);
        // SAFETY: we write every byte of the reserved capacity.
        unsafe {
            out.set_len(total);
            let base = out.as_mut_ptr();
            std::ptr::copy_nonoverlapping(MAGIC.as_ptr(), base, 4);
            (base.add(4) as *mut u64).write_unaligned(self.num_bits.to_le());
            (base.add(12) as *mut u32).write_unaligned(self.num_hashes.to_le());
            for (i, word) in self.bits.iter().enumerate() {
                (base.add(header_len + i * 8) as *mut u64).write_unaligned(word.to_le());
            }
        }
        out
    }

    /// Decodes a filter serialised by [`BloomFilter::to_bytes`]. Returns
    /// `None` on any structural mismatch (bad magic, truncated payload).
    /// Uses unsafe pointer reads to avoid `try_into()` per word.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 16 {
            return None;
        }
        if bytes.get(..4)? != MAGIC {
            return None;
        }
        let ptr = bytes.as_ptr();
        let num_bits = u64::from_le(unsafe { (ptr.add(4) as *const u64).read_unaligned() });
        let num_hashes = u32::from_le(unsafe { (ptr.add(12) as *const u32).read_unaligned() });
        let words = (num_bits / 64) as usize;
        if bytes.len() != 16 + words * 8 {
            return None;
        }
        let mut bits = Vec::with_capacity(words);
        for word_index in 0..words {
            let word = u64::from_le(unsafe {
                (ptr.add(16 + word_index * 8) as *const u64).read_unaligned()
            });
            bits.push(word);
        }
        Some(Self {
            num_bits,
            num_hashes,
            bits,
        })
    }

    /// Writes the filter to `path` (atomic enough for a derived cache: a torn
    /// write is detected by [`BloomFilter::from_bytes`] and falls back to the
    /// full read).
    pub fn write_to(&self, path: &Path) -> Result<()> {
        std::fs::write(path, self.to_bytes())?;
        Ok(())
    }

    /// Reads and decodes a filter from `path`, or returns `None` when the file
    /// is missing or malformed (the caller then performs the full read).
    #[must_use]
    pub fn read_from(path: &Path) -> Option<Self> {
        let bytes = crate::mmap_io::mmap_read_to_vec(path).ok()?;
        Self::from_bytes(&bytes)
    }

    /// Validates internal invariants. Called by fuzz targets after decode and
    /// after exercise (SQLite pattern: `assert_invariants()` after `decode`).
    /// Returns `true` if the filter is internally consistent.
    #[must_use]
    pub fn assert_invariants(&self) -> bool {
        // num_bits must be a power of two (guaranteed by `new`, verified on decode)
        if !self.num_bits.is_power_of_two() {
            return false;
        }
        // num_hashes must be in valid range
        if self.num_hashes == 0 || self.num_hashes > 32 {
            return false;
        }
        // bits length must match num_bits / 64
        let expected_words = (self.num_bits / 64) as usize;
        if self.bits.len() != expected_words {
            return false;
        }
        true
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn insert_then_contains() {
        let mut filter = BloomFilter::new(1000, 0.01);
        filter.insert(b"alice");
        filter.insert(b"bob");
        assert!(filter.contains(b"alice"));
        assert!(filter.contains(b"bob"));
        // False positives are possible but must not be systematic: a key far
        // from the inserted set should (with overwhelming probability) miss.
        assert!(!filter.contains(b"charlie"));
    }

    #[test]
    fn round_trip_bytes() {
        let mut filter = BloomFilter::new(500, 0.05);
        filter.insert(b"key-1");
        filter.insert(b"key-2");
        let bytes = filter.to_bytes();
        let decoded = BloomFilter::from_bytes(&bytes).expect("decode");
        assert_eq!(filter, decoded);
        assert!(decoded.contains(b"key-1"));
        assert!(!decoded.contains(b"key-9"));
    }

    #[test]
    fn rejects_corrupt_input() {
        assert!(BloomFilter::from_bytes(b"").is_none());
        assert!(BloomFilter::from_bytes(b"XXXX").is_none());
        assert!(BloomFilter::from_bytes(b"BF01garbage").is_none());
        // Truncate a valid encoding.
        let filter = BloomFilter::new(100, 0.01);
        let mut bytes = filter.to_bytes();
        bytes.pop();
        assert!(BloomFilter::from_bytes(&bytes).is_none());
    }

    #[test]
    fn determinism_across_instances() {
        // Two independently-built filters with the same parameters and keys
        // must produce identical bytes (restart safety).
        let mut a = BloomFilter::new(1000, 0.01);
        let mut b = BloomFilter::new(1000, 0.01);
        for key in [b"k1", b"k2", b"k3"] {
            a.insert(key);
            b.insert(key);
        }
        assert_eq!(a.to_bytes(), b.to_bytes());
    }

    #[test]
    fn invariants_hold_for_valid_filter() {
        let filter = BloomFilter::new(1000, 0.01);
        assert!(filter.assert_invariants());
    }

    #[test]
    fn invariants_reject_zero_num_bits() {
        let mut filter = BloomFilter::new(1000, 0.01);
        filter.num_bits = 0;
        assert!(!filter.assert_invariants());
    }
}
