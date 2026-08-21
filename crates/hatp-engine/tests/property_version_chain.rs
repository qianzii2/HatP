//! VersionChain newest-first property test (G04 / C10)
//! Invariant: after insert, versions[0].begin_ts is the maximum
//! Source: gap-report.md G04, execution-plan.md C10

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use bytes::Bytes;
use hatp_engine::version::{VersionChain, VersionedValue, OPEN_ENDED_TS};
use proptest::prelude::*;

fn random_ts_sequence() -> impl Strategy<Value = Vec<u64>> {
    proptest::collection::vec(1u64..u64::MAX, 1..=100)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5000))]

    #[test]
    fn version_chain_newest_first_after_inserts(timestamps in random_ts_sequence()) {
        let mut chain = VersionChain::new();
        for (&ts, i) in timestamps.iter().zip(0u64..) {
            chain.insert(VersionedValue::live(
                Bytes::copy_from_slice(&ts.to_le_bytes()), ts, i,
            ));
        }
        if chain.is_empty() { return Ok(()); }
        let versions = chain.versions();
        // Invariant: newest-first (highest begin_ts at index 0)
        for w in versions.windows(2) {
            assert!(w[0].begin_ts >= w[1].begin_ts,
                "newest-first violated: {} >= {}",
                w[0].begin_ts, w[1].begin_ts);
        }
    }

    #[test]
    fn version_chain_visibility_intervals_contiguous(timestamps in random_ts_sequence()) {
        let mut chain = VersionChain::new();
        for (&ts, i) in timestamps.iter().zip(0u64..) {
            chain.insert(VersionedValue::live(
                Bytes::copy_from_slice(&ts.to_le_bytes()), ts, i,
            ));
        }
        if chain.is_empty() { return Ok(()); }
        let versions = chain.versions();
        // Invariant: adjacent version visibility intervals are contiguous
        for w in versions.windows(2) {
            assert_eq!(w[1].end_ts, w[0].begin_ts,
                "visibility gap: newer={} older_end={}",
                w[0].begin_ts, w[1].end_ts);
        }
        // Newest version end_ts = OPEN_ENDED_TS
        assert_eq!(versions[0].end_ts, OPEN_ENDED_TS);
    }
}

#[test]
fn version_chain_single_insert_is_newest() {
    let mut chain = VersionChain::new();
    chain.insert(VersionedValue::live(Bytes::from_static(b"v"), 42, 1));
    assert_eq!(chain.versions()[0].begin_ts, 42);
    assert_eq!(chain.versions()[0].end_ts, OPEN_ENDED_TS);
}

#[test]
fn version_chain_monotonic_timestamps_stay_newest_first() {
    let mut chain = VersionChain::new();
    // Monotonically increasing ts insert (production path)
    for ts in 1..=50u64 {
        chain.insert(VersionedValue::live(
            Bytes::copy_from_slice(&ts.to_le_bytes()), ts, ts,
        ));
    }
    let versions = chain.versions();
    assert_eq!(versions[0].begin_ts, 50);
    assert_eq!(versions[49].begin_ts, 1);
    for w in versions.windows(2) {
        assert!(w[0].begin_ts > w[1].begin_ts);
    }
}

#[test]
fn version_chain_out_of_order_inserts_still_newest_first() {
    let mut chain = VersionChain::new();
    // Out-of-order insert
    let tss: [u64; 5] = [100, 50, 200, 150, 75];
    for (&ts, i) in tss.iter().zip(0u64..) {
        chain.insert(VersionedValue::live(
            Bytes::copy_from_slice(&ts.to_le_bytes()), ts, i,
        ));
    }
    let versions = chain.versions();
    assert_eq!(versions[0].begin_ts, 200);
    assert_eq!(versions[4].begin_ts, 50);
    for w in versions.windows(2) {
        assert!(w[0].begin_ts >= w[1].begin_ts);
    }
}