# fuzz/ — Fuzz Testing

## Overview

6 `libfuzzer`-based fuzz targets covering every binary parse entry point in the engine.
Each target follows the **SQLite pattern**: `decode → assert_invariants → exercise → roundtrip`.

## Fuzz Target Index

| Target | Module | Oracle Layers |
|--------|--------|---------------|
| `fuzz_bloom_decode` | `bloom::BloomFilter::from_bytes` | 1) no panic 2) invariants 3) contains no panic 4) roundtrip |
| `fuzz_wal_decode` | `wal::decode_all` | 1) no panic 2) roundtrip encode→decode |
| `fuzz_key_index_decode` | `key_index::KeyIndex::from_bytes` | 1) no panic 2) invariants 3) get no panic |
| `fuzz_manifest_decode` | `manifest::decode_all_for_fuzz` | 1) no panic |
| `fuzz_row_codec` | `row_codec::decode_row_values` | 1) no panic 2) roundtrip encode→decode |
| `fuzz_vortex_sst` | `vortex_sst::read_all` | 1) no panic 2) returned rows key-sorted |

## Running

```bash
cargo +nightly fuzz run fuzz_wal_decode
cargo +nightly fuzz run fuzz_bloom_decode
# ... each target
```

## Design Decisions

### Why a 10MB input cap per target?

Prevents OOM from the fuzzer generating excessively large inputs. 10MB is far beyond any
legitimate input (WAL frames < 1KB, SSTs < 10MB, Bloom < 1MB) but still covers edge cases.

### Why no `[workspace]` member in `Cargo.toml`?

Fuzz targets require the `nightly` toolchain and `libfuzzer-sys`. Keeping them outside the
workspace avoids polluting `cargo test` stable builds.