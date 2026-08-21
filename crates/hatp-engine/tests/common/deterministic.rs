//! Deterministic PRNG — all randomness is seed-controlled and fully reproducible
//!
//! Reference: FoundationDB `DeterministicRandom` (Xoroshiro256**)
//!            TigerBeetle `stdx.PRNG`
//!
//! Key invariant: same seed → same sequence. Any test that depends on randomness
//! must use this PRNG; direct use of `rand::thread_rng()` is forbidden.

use std::num::Wrapping;

/// Xoroshiro256** — fast, high-quality, deterministic PRNG
///
/// This is the same algorithm family used by FoundationDB. Period 2^256 - 1.
#[derive(Debug, Clone)]
pub struct DeterministicRandom {
    s: [Wrapping<u64>; 4],
}

impl DeterministicRandom {
    /// Construct a PRNG from a 64-bit seed.
    ///
    /// Uses SplitMix64 to initialize the internal state from the seed, ensuring
    /// uniform mapping of the seed space.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut sm = SplitMix64 { state: seed };
        let s = [Wrapping(sm.next()), Wrapping(sm.next()), Wrapping(sm.next()), Wrapping(sm.next())];
        Self { s }
    }

    /// Return the next `u64`.
    pub fn next_u64(&mut self) -> u64 {
        let result = rotl(self.s[1] * Wrapping(5), 7) * Wrapping(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = rotl(self.s[3], 45);
        result.0
    }

    /// Return a `u64` in the range `[0, max)`.
    pub fn next_u64_bounded(&mut self, max: u64) -> u64 {
        if max <= 1 {
            return 0;
        }
        let limit = u64::MAX - u64::MAX % max;
        loop {
            let v = self.next_u64();
            if v < limit {
                return v % max;
            }
        }
    }

    /// Return an `f64` in the range `[0.0, 1.0)`.
    #[allow(dead_code)]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * 1.1102230246251565e-16
    }

    /// Return `true` with probability `probability`.
    #[allow(dead_code)]
    pub fn coin_flip(&mut self, probability: f64) -> bool {
        self.next_f64() < probability
    }

    /// Return a `usize` in the range `[min, max]`.
    #[allow(dead_code)]
    pub fn next_usize_in_range(&mut self, min: usize, max: usize) -> usize {
        assert!(min <= max, "min {min} must be <= max {max}");
        min + self.next_u64_bounded((max - min + 1) as u64) as usize
    }

    /// Generate `len` random bytes as `Vec<u8>`.
    #[allow(dead_code)]
    pub fn next_bytes(&mut self, len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len);
        let mut remaining = len;
        while remaining >= 8 {
            let v = self.next_u64().to_le_bytes();
            out.extend_from_slice(&v);
            remaining -= 8;
        }
        if remaining > 0 {
            let v = self.next_u64().to_le_bytes();
            out.extend_from_slice(&v[..remaining]);
        }
        out
    }

    /// Generate a random ASCII string of `len` bytes.
    #[allow(dead_code)]
    pub fn next_ascii_string(&mut self, len: usize) -> String {
        let bytes: Vec<u8> = (0..len)
            .map(|_| 0x61 + self.next_u64_bounded(26) as u8)
            .collect();
        String::from_utf8(bytes).expect("ASCII")
    }
}

/// SplitMix64 — used to initialize the Xoroshiro state from a seed
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

#[inline]
fn rotl(x: Wrapping<u64>, k: u32) -> Wrapping<u64> {
    (x << k as usize) | (x >> (64 - k) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_produces_same_sequence() {
        let mut a = DeterministicRandom::new(42);
        let mut b = DeterministicRandom::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64(), "determinism broken");
        }
    }

    #[test]
    fn different_seeds_produce_different_sequences() {
        let mut a = DeterministicRandom::new(1);
        let mut b = DeterministicRandom::new(2);
        let mut same = 0;
        for _ in 0..100 {
            if a.next_u64() == b.next_u64() {
                same += 1;
            }
        }
        assert!(same < 10, "different seeds should rarely collide, got {same}/100");
    }

    #[test]
    fn bounded_range_is_uniform() {
        let mut rng = DeterministicRandom::new(99);
        let mut counts = [0u64; 10];
        for _ in 0..10_000 {
            let v = rng.next_u64_bounded(10);
            counts[v as usize] += 1;
        }
        for c in &counts {
            assert!(*c > 700, "uniformity check: all buckets must be >700, got {c}");
        }
    }
}