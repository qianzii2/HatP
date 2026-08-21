# hatp-types — Shared Types, Constants & Codecs

**Role**: The bottom-most shared crate in HatP. All other sub-crates depend on it for unified type aliases
and encoding functions, avoiding duplication and circular dependencies.

**Boundary**: Depends on no other HatP sub-crate. Only `bytes`, `arrow`, `datafusion-common`.

## Invariants

| # | Invariant | Location |
|---|-----------|----------|
| T1 | `encode_pk_value(a) < encode_pk_value(b)` iff `a < b` (same type, order-preserving) | `codec.rs:149-179` |
| T2 | `scalar_to_index_be` unifies to 8 bytes so Int32 columns match Int64 literals | `codec.rs:253-284` |
| T3 | Encoded strings never contain bare `0x00 0x00` sequences (FoundationDB-style NUL escaping) | `codec.rs:228-240` |
| T4 | `fnv1a` and `djb2` are deterministic, byte-order-sensitive, independent hash functions | `codec.rs:290-344` |
| T5 | `array_slot_to_scalar` is the single Arrow→ScalarValue conversion entry point | `codec.rs:48-121` |

## Two Encodings, Not Interchangeable

**Primary key encoding (`encode_pk_value`)**: Preserves original byte width (Int32 = 4 bytes, Int64 = 8 bytes).
Used by `assemble_table_key` to construct primary key keys.

**Index encoding (`scalar_to_index_be`)**: Unifies to 8 bytes (small integers/floats promoted to i64/u64/f64).
Used by secondary indexes so different declared-width numerics are comparable and matchable within the same index.

Both guarantee **byte-order == logical-order**:
- Signed integers: big-endian + sign-bit flip (`v ^ MIN`), so `negative < 0 < positive` matches byte order
- Floats: IEEE-754 total-order transform (negatives bitwise-inverted, positives sign-bit set)
- Strings: FoundationDB-style escaping (`0x00` → `0x00 0xFF`, trailing `0x00` terminator)

## Design Decisions

### Why FNV-1a and DJB2 instead of `std::hash::DefaultHasher`?

`DefaultHasher` is not stable across processes or Rust versions. Bloom filters and simulation digests
must produce identical hashes in the flush process (writer) and every subsequent read process.
FNV-1a and DJB2 are pure mathematical formulas with no per-process random seed — naturally cross-process stable.

### Why does `sign_flip_be` use unsafe pointer slicing?

`sign_flip_be` takes the low `width` bytes of `i64.to_be_bytes()`. The `start..` slice is guarded by
`debug_assert!(1..=8)`. The `Vec::from(&be_bytes[start..])` path uses `#[allow(clippy::indexing_slicing)]`
in release builds to avoid unnecessary bounds checks on the hot path.

## Verification

| Method | Coverage |
|--------|----------|
| Unit tests | 14 tests covering roundtrip, order-preservation, NUL escaping, hash determinism |
| Proptest (2,000 cases) | PK encoding order-preserving across 13 types × random values |
| Kani proofs | `sign_flip_be` order (adjacent, negative-zero-positive), `escape_bytes` no bare NUL, float total-order |