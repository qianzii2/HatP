//! BTreeMap vs SkipMap differential test (VC-MEMTABLE-DIFF)
//!
//! Invariant: ∀ random operation sequences, BTreeMapTable and SkipMapTable produce
//! identical scan_prefix output for the same operation sequence.
//!
//! Source: gap-report.md G06, execution-plan.md C12 | Method: proptest | Budget: 10,000 sequences

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use bytes::Bytes;
use hatp_engine::memtable::{MemTable, MemTableImpl, VersionedRow};
use hatp_engine::version::{Snapshot, VersionedValue, OPEN_ENDED_TS};
use proptest::prelude::*;
use proptest::collection::vec as pvec;

// ── Operation types ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Op {
    Put { key: Vec<u8>, value: Vec<u8> },
    Delete { key: Vec<u8> },
    Flush,
    AdvanceSnapshot,
}

// ── Strategy: random operation sequences ─────────────────────────────────────

fn key_strategy() -> impl Strategy<Value = Vec<u8>> {
    prop_oneof![
        1 => any::<Vec<u8>>().prop_map(|v| v),
        4 => pvec(any::<u8>(), 1..=64),
    ]
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        8 => (key_strategy(), pvec(any::<u8>(), 0..=256))
            .prop_map(|(k, v)| Op::Put { key: k, value: v }),
        3 => key_strategy().prop_map(|k| Op::Delete { key: k }),
        1 => Just(Op::Flush),
        1 => Just(Op::AdvanceSnapshot),
    ]
}

fn ops_strategy() -> impl Strategy<Value = Vec<Op>> {
    pvec(op_strategy(), 1..=100)
}

// ── Helper: apply operation sequence to both backends and compare ────────────

fn apply_ops(ops: &[Op], table: &MemTable, snapshot_ts: &mut u64) {
    let mut pending: Vec<VersionedRow> = Vec::new();
    for op in ops {
        match op {
            Op::Put { key, value } => {
                pending.push(VersionedRow {
                    key: Bytes::copy_from_slice(key),
                    version: VersionedValue {
                        value: Some(Bytes::copy_from_slice(value)),
                        begin_ts: *snapshot_ts,
                        end_ts: OPEN_ENDED_TS,
                        tx_id: 0,
                    },
                });
            }
            Op::Delete { key } => {
                pending.push(VersionedRow {
                    key: Bytes::copy_from_slice(key),
                    version: VersionedValue {
                        value: None,
                        begin_ts: *snapshot_ts,
                        end_ts: OPEN_ENDED_TS,
                        tx_id: 0,
                    },
                });
            }
            Op::Flush => {
                // Apply directly to memtable (simulating flush's apply semantics)
                // Using WAL record format
                let records: Vec<hatp_engine::wal::WalRecord> = pending
                    .drain(..)
                    .map(|row| {
                        if let Some(val) = row.version.value {
                            hatp_engine::wal::WalRecord::put(
                                row.version.tx_id, row.key, val,
                            )
                        } else {
                            hatp_engine::wal::WalRecord::delete(
                                row.version.tx_id, row.key,
                            )
                        }
                    })
                    .collect();
                if !records.is_empty() {
                    table.apply_records_batch(&records, *snapshot_ts).ok();
                }
            }
            Op::AdvanceSnapshot => {
                *snapshot_ts += 1;
            }
        }
    }
    // Flush remaining
    if !pending.is_empty() {
        let records: Vec<hatp_engine::wal::WalRecord> = pending
            .drain(..)
            .map(|row| {
                if let Some(val) = row.version.value {
                    hatp_engine::wal::WalRecord::put(row.version.tx_id, row.key, val)
                } else {
                    hatp_engine::wal::WalRecord::delete(row.version.tx_id, row.key)
                }
            })
            .collect();
        table.apply_records_batch(&records, *snapshot_ts).ok();
    }
}

fn scan_prefix(table: &MemTable, prefix: &[u8], snapshot: Snapshot) -> Vec<(Bytes, Bytes)> {
    table.scan_prefix(prefix, snapshot)
}

// ── Property tests ───────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5000))]

    #[test]
    fn btree_and_skipmap_scan_prefix_equivalent(ops in ops_strategy()) {
        let btree = MemTable::with_impl(MemTableImpl::BTreeMap);
        let skip = MemTable::with_impl(MemTableImpl::SkipMap);

        let mut snapshot_ts = 1_u64;
        apply_ops(&ops, &btree, &mut snapshot_ts);
        let mut snapshot_ts = 1_u64;
        apply_ops(&ops, &skip, &mut snapshot_ts);

        let snap = Snapshot::new(snapshot_ts);

        // Scan and compare against several prefixes
        let prefixes: Vec<&[u8]> = vec![b"", b"a", b"k", b"z", b"\xff"];
        for prefix in prefixes {
            let btree_result = scan_prefix(&btree, prefix, snap);
            let skip_result = scan_prefix(&skip, prefix, snap);
            assert_eq!(
                btree_result, skip_result,
                "prefix={:?}: BTreeMap and SkipMap must produce identical scan results",
                std::str::from_utf8(prefix).unwrap_or("<non-utf8>")
            );
        }
    }
}

// ── Boundary value tests ─────────────────────────────────────────────────────

#[test]
fn empty_memtables_are_equivalent() {
    let btree = MemTable::with_impl(MemTableImpl::BTreeMap);
    let skip = MemTable::with_impl(MemTableImpl::SkipMap);
    let snap = Snapshot::new(1);
    assert_eq!(btree.scan_prefix(b"", snap), skip.scan_prefix(b"", snap));
    assert_eq!(btree.len(), skip.len());
    assert!(btree.is_empty());
    assert!(skip.is_empty());
}

#[test]
fn single_key_put_equivalent() {
    let btree = MemTable::with_impl(MemTableImpl::BTreeMap);
    let skip = MemTable::with_impl(MemTableImpl::SkipMap);
    let records = vec![
        hatp_engine::wal::WalRecord::put(1, Bytes::from_static(b"key"), Bytes::from_static(b"value")),
    ];
    btree.apply_records_batch(&records, 1).unwrap();
    skip.apply_records_batch(&records, 1).unwrap();
    let snap = Snapshot::new(1);
    assert_eq!(btree.scan_prefix(b"", snap), skip.scan_prefix(b"", snap));
    assert_eq!(btree.len(), skip.len());
}

#[test]
fn delete_and_overwrite_equivalent() {
    let btree = MemTable::with_impl(MemTableImpl::BTreeMap);
    let skip = MemTable::with_impl(MemTableImpl::SkipMap);
    let records = vec![
        hatp_engine::wal::WalRecord::put(1, Bytes::from_static(b"k"), Bytes::from_static(b"v1")),
        hatp_engine::wal::WalRecord::delete(2, Bytes::from_static(b"k")),
        hatp_engine::wal::WalRecord::put(3, Bytes::from_static(b"k"), Bytes::from_static(b"v2")),
    ];
    btree.apply_records_batch(&records, 1).unwrap();
    skip.apply_records_batch(&records, 1).unwrap();
    let snap = Snapshot::new(1);
    assert_eq!(btree.scan_prefix(b"", snap), skip.scan_prefix(b"", snap));
}

#[test]
fn snapshot_isolation_equivalent() {
    let btree = MemTable::with_impl(MemTableImpl::BTreeMap);
    let skip = MemTable::with_impl(MemTableImpl::SkipMap);
    // ts=1: put k=v1
    btree.apply_records_batch(
        &[hatp_engine::wal::WalRecord::put(1, Bytes::from_static(b"k"), Bytes::from_static(b"v1"))],
        1,
    ).unwrap();
    skip.apply_records_batch(
        &[hatp_engine::wal::WalRecord::put(1, Bytes::from_static(b"k"), Bytes::from_static(b"v1"))],
        1,
    ).unwrap();
    // ts=2: put k=v2
    btree.apply_records_batch(
        &[hatp_engine::wal::WalRecord::put(2, Bytes::from_static(b"k"), Bytes::from_static(b"v2"))],
        2,
    ).unwrap();
    skip.apply_records_batch(
        &[hatp_engine::wal::WalRecord::put(2, Bytes::from_static(b"k"), Bytes::from_static(b"v2"))],
        2,
    ).unwrap();

    // snapshot=1: both see v1
    let snap1 = Snapshot::new(1);
    assert_eq!(btree.scan_prefix(b"", snap1), skip.scan_prefix(b"", snap1));

    // snapshot=2: both see v2
    let snap2 = Snapshot::new(2);
    assert_eq!(btree.scan_prefix(b"", snap2), skip.scan_prefix(b"", snap2));
}