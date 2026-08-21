# proofs/ — Formal Verification (Kani)

## Overview

9 `kani::proof` covering key invariants in the engine. Each proof verifies that code satisfies
an invariant for **all possible inputs**, not just sampled values. Kani uses model checking
(rather than SMT solving) and is purpose-built for Rust verification.

## Proof Index

| Proof | File | Invariant |
|-------|------|-----------|
| `kani_version_chain_newest_first` | `kani_engine_invariants.rs` | After inserting any 3 timestamps, `versions[0].begin_ts` is the maximum |
| `kani_version_chain_single_insert_is_newest` | `kani_engine_invariants.rs` | Single insert → chain head equals the inserted value |
| `kani_commit_seq_monotonic` | `kani_engine_invariants.rs` | After `fetch_add`, the new value is strictly greater than the old |
| `kani_token_bucket_monotonic` | `kani_engine_invariants.rs` | Throttle level never relaxes as memtable grows |
| `kani_key_index_from_bytes_no_panic` | `kani_engine_invariants.rs` | Any 32-byte input never panics |
| `kani_wal_encode_decode_roundtrip` | `kani_engine_invariants.rs` | `encode_frame` → `decode_frame` equivalence |
| `kani_crc32c_detects_single_bit_flip` | `kani_engine_invariants.rs` | For any 1–8 bytes, flipping any bit changes the CRC |
| `kani_bloom_deterministic` | `kani_bloom_deterministic.rs` | Insert → contains returns true; serialization roundtrip |
| `kani_escape_bytes_no_bare_nul` | `kani_escape_float.rs` | Any string encoding contains no bare NUL-NUL sequence |
| `kani_float_total_order_preserves` | `kani_escape_float.rs` | `a < b` → `encoded(a) < encoded(b)` |
| `kani_sign_flip_be_adjacent_preserves_order` | `kani_sign_flip_be.rs` | Adjacent integers preserve encoding order |
| `kani_sign_flip_be_negative_zero_positive` | `kani_sign_flip_be.rs` | `negative < zero < positive` |

## Running

```bash
cargo kani --package hatp-engine --harness kani_crc32c_detects_single_bit_flip
cargo kani --package hatp-types --harness kani_sign_flip_be_adjacent_preserves_order
```

## Design Decisions

### Why Kani instead of Creusot or Verus?

Kani is the most mature formal verification tool for Rust (maintained by AWS). It model-checks
MIR directly without requiring specification annotations in the source. Creusot/Verus require
`#[requires]` / `#[ensures]` contracts, which are incompatible with HatP's existing code style.

### Why does `kani::assume` restrict input ranges?

Kani's symbolic execution of full-range `u64` would cause state explosion. `kani::assume(ts < u64::MAX/2)`
bounds the search space while still covering all realistic inputs (`commit_ts` starts at 1 and
increments monotonically, never reaching `u64::MAX`).

### Why are proofs outside the workspace?

Kani requires `#![cfg(kani)]` attributes, which are invisible under `cargo test`. Keeping proofs
outside the workspace prevents `cargo test` from failing on unresolved `kani::proof` attributes.