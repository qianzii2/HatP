//! Boundary value generator — similar to SQLite boundary1.test
//!
//! Generates a collection of key boundary values for testing engine correctness
//! under extreme conditions.
//!
//! Reference: SQLite `boundary1.test` (64 rowid boundary values × 5 operators × 5 ORDER BY)
//!            RocksDB parameterized tests (empty values, large values, boundary write_buffer_size)

use bytes::Bytes;

/// Boundary key set — taken from critical LSM engine boundaries
///
/// These keys cover the engine's internal encoding paths that use `\0` as a separator:
/// - Empty key
/// - Key containing only `\0`
/// - Key with `\0` prefix
/// - Key with `\0` suffix
/// - Long key
/// - Maximum single-byte key (ending with 0xFF)
#[must_use]
pub fn boundary_keys() -> Vec<Bytes> {
    vec![
        Bytes::from_static(b""),
        Bytes::from_static(b"\0"),
        Bytes::from_static(b"\0\0"),
        Bytes::from_static(b"\0a"),
        Bytes::from_static(b"a\0"),
        Bytes::from_static(b"a\0b"),
        Bytes::from_static(b"a\0\0b"),
        Bytes::from_static(b"\xff"),
        Bytes::from_static(b"\xff\xff"),
        Bytes::from_static(b"\xff\xff\xff\xff"),
        Bytes::copy_from_slice(&[0xFF_u8; 64]),
        Bytes::copy_from_slice(&[0x00_u8; 64]),
        Bytes::copy_from_slice(&[0x41_u8; 1024]), // 1KB 'A'
        Bytes::copy_from_slice(&[0x41_u8; 65536]), // 64KB 'A'
    ]
}

/// Boundary value set — covering various value sizes
#[must_use]
pub fn boundary_values() -> Vec<Bytes> {
    vec![
        Bytes::from_static(b""),
        Bytes::from_static(b"a"),
        Bytes::copy_from_slice(&[0x42_u8; 1024]),
        Bytes::copy_from_slice(&[0x42_u8; 65536]),
        Bytes::copy_from_slice(&[0x42_u8; 262144]), // 256KB
    ]
}

/// Boundary key names — human-readable labels
#[must_use]
pub fn boundary_key_names() -> Vec<&'static str> {
    vec![
        "empty",
        "single_nul",
        "double_nul",
        "nul_a",
        "a_nul",
        "a_nul_b",
        "a_nul_nul_b",
        "ff",
        "ff_ff",
        "ff_x4",
        "ff_x64",
        "nul_x64",
        "1KB_A",
        "64KB_A",
    ]
}

/// Concurrency level boundaries — for testing concurrent writes
#[must_use]
#[allow(dead_code)]
pub fn concurrency_levels() -> Vec<usize> {
    vec![1, 2, 4, 8, 16]
}

/// Batch size boundaries — for testing flush/compaction triggering
#[must_use]
#[allow(dead_code)]
pub fn batch_sizes() -> Vec<usize> {
    vec![0, 1, 2, 10, 100, 1000]
}

/// Generate the Cartesian product of all boundary keys × values
///
/// Returns `(name, key, value)` triples
#[must_use]
pub fn boundary_key_value_pairs() -> Vec<(&'static str, Bytes, Bytes)> {
    let keys = boundary_keys();
    let names = boundary_key_names();
    let values = boundary_values();
    let mut out = Vec::new();
    for (i, key) in keys.into_iter().enumerate() {
        for value in &values {
            out.push((names[i], key.clone(), value.clone()));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_keys_are_non_empty_list() {
        assert!(!boundary_keys().is_empty());
    }

    #[test]
    fn boundary_key_names_match_keys() {
        assert_eq!(boundary_keys().len(), boundary_key_names().len());
    }

    #[test]
    fn boundary_key_value_pairs_are_non_empty() {
        let pairs = boundary_key_value_pairs();
        assert!(!pairs.is_empty());
        // Each key is paired with at least one value
        assert!(pairs.len() >= boundary_keys().len());
    }
}