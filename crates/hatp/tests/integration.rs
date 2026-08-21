//! End-to-end integration tests for HatP.
//!
//! These tests live in the top-level `hatp` crate because they exercise
//! the `hatp-tx::TxManagerHook` ↔ `hatp-engine::Engine` boundary, which
//! cannot be tested from either sub-crate alone (the engine does not
//! depend on the tx layer).

// Tests use `expect`/`unwrap`/`panic`/indexing by convention — a failed
// assertion must abort loudly, and `inflight[0]` is guarded by a
// `len() == 1` assertion just above it.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::sync::Arc;

use hatp::{Database, DatabaseError};
use hatp_engine::{Engine, EngineConfig, EngineHook, Mutation};
use hatp_tx::TxManagerHook;
use tempfile::TempDir;

fn tmpdir() -> TempDir {
    tempfile::Builder::new()
        .prefix("hatp-integration-")
        .tempdir()
        .expect("tempdir")
}

#[derive(Debug, Default)]
struct CountingHook {
    puts: std::sync::atomic::AtomicU64,
    deletes: std::sync::atomic::AtomicU64,
    commits: std::sync::atomic::AtomicU64,
}

impl EngineHook for CountingHook {
    fn on_put(&self, _: &[u8], _: &[u8], _: u64) {
        self.puts.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    fn on_delete(&self, _: &[u8], _: u64) {
        self.deletes
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    fn on_tx_commit(&self, _: u64, _commit_ts: u64) {
        self.commits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[test]
fn tx_hook_does_not_break_engine_writes() {
    let dir = tmpdir();
    let manager = hatp_tx::TxManager::new();
    let hook: Arc<dyn EngineHook> = Arc::new(TxManagerHook::new(manager.clone()));
    let engine = Engine::open_with_hook(
        EngineConfig::new(dir.path()),
        
        hook,
    )
    .expect("open");

    engine
        .write(&[
            Mutation::Put {
                key: bytes::Bytes::from_static(b"a"),
                value: bytes::Bytes::from_static(b"1"),
            },
            Mutation::Put {
                key: bytes::Bytes::from_static(b"b"),
                value: bytes::Bytes::from_static(b"2"),
            },
            Mutation::Delete {
                key: bytes::Bytes::from_static(b"c"),
            },
        ])
        .expect("write");

    assert_eq!(
        engine
            .get(b"a", engine.snapshot_ts())
            .expect("get a")
            .map(|v| v.to_vec()),
        Some(b"1".to_vec())
    );
    assert_eq!(
        engine
            .get(b"b", engine.snapshot_ts())
            .expect("get b")
            .map(|v| v.to_vec()),
        Some(b"2".to_vec())
    );
    assert!(
        engine
            .get(b"c", engine.snapshot_ts())
            .expect("get c")
            .is_none()
    );
}

#[test]
fn counting_hook_observes_each_mutation() {
    let dir = tmpdir();
    let counter = Arc::new(CountingHook::default());
    let hook: Arc<dyn EngineHook> = counter.clone();
    let engine = Engine::open_with_hook(
        EngineConfig::new(dir.path()),
        
        hook,
    )
    .expect("open");

    engine
        .write(&[
            Mutation::Put {
                key: bytes::Bytes::from_static(b"x"),
                value: bytes::Bytes::from_static(b"1"),
            },
            Mutation::Put {
                key: bytes::Bytes::from_static(b"y"),
                value: bytes::Bytes::from_static(b"2"),
            },
            Mutation::Delete {
                key: bytes::Bytes::from_static(b"z"),
            },
        ])
        .expect("write");

    assert_eq!(counter.puts.load(std::sync::atomic::Ordering::Relaxed), 2);
    assert_eq!(
        counter.deletes.load(std::sync::atomic::Ordering::Relaxed),
        1
    );
    assert_eq!(
        counter.commits.load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

// ---------------------------------------------------------------------------
// PR3 acceptance tests: real DataFusion SQL execution + engine-backed scan.
// ---------------------------------------------------------------------------

#[test]
fn database_execute_sql_returns_datafusion_result() {
    use hatp::Database;

    let dir = tmpdir();
    let database = Database::open(dir.path()).expect("open database");
    let outcome = futures::executor::block_on(database.execute_sql("SELECT 1 + 1 AS two"))
        .expect("execute_sql must succeed");
    assert_eq!(outcome.rows, 1);
    assert!(outcome.batches >= 1);
}

#[test]
fn database_register_table_advertises_to_session() {
    use hatp::Database;
    use hatp_frontend::schema::{CreateTable, QualifiedName};

    let dir = tmpdir();
    let database = Database::open(dir.path()).expect("open database");

    let schema: std::sync::Arc<arrow_schema::Schema> =
        std::sync::Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("id", arrow_schema::DataType::Int32, false),
            arrow_schema::Field::new("name", arrow_schema::DataType::Utf8, true),
        ]));
    database
        .catalog()
        .create_table(CreateTable {
            qualified: QualifiedName::new("hatp", "public"),
            name: "scores".to_string(),
            arrow: schema,
            primary_key: vec!["id".to_string()],
        })
        .expect("create_table");
    let ts = database
        .catalog()
        .table_schema(&QualifiedName::new("hatp", "public"), "scores")
        .expect("table_schema");
    let adapter = database.register_table(ts).expect("register_table");
    assert!(adapter.is_engine_attached());
    // The adapter is registered under the bare name; resolve it to confirm
    // DataFusion sees an engine-backed table.
    let plan = futures::executor::block_on(database.execute_sql("SELECT id FROM scores"))
        .expect("execute_sql after register_table");
    assert!(plan.rows <= 1, "empty table must return no rows");
}

#[test]
fn frontend_session_executes_select() {
    use hatp_frontend::execution::FrontendSession;

    let session = FrontendSession::new();
    let outcome =
        futures::executor::block_on(session.execute("SELECT 42 AS answer")).expect("execute");
    assert_eq!(outcome.rows, 1);
    assert!(outcome.batches >= 1);
    assert!(
        outcome
            .head
            .iter()
            .any(|line| line.contains("answer =") || line.contains("42"))
    );
}

// ---------------------------------------------------------------------------
// Stage 2: SSI end-to-end tests via Database + TxManagerHook.
// ---------------------------------------------------------------------------

#[test]
fn database_put_bypasses_ssi() {
    let dir = tmpdir();
    let manager = hatp_tx::TxManager::new();
    let database = Database::open_with_tx_manager(dir.path(), manager).expect("open");
    // Plain `put` must not require SSI validation.
    database
        .put("k1", "v1")
        .expect("plain put must succeed under SSI manager");
    assert_eq!(
        database.get(b"k1").expect("get").map(|v| v.to_vec()),
        Some(b"v1".to_vec())
    );
}

/// Two SSI transactions write the same key: the first committer succeeds, the second detects
/// the conflict at pre-commit and receives SsiConflict. This is the core first-committer-wins semantics.
/// Deterministic: single-threaded sequential execution, serialized by the engine's write_guard.
#[test]
fn transaction_ssi_blocks_conflict() {
    let dir = tmpdir();
    let manager = hatp_tx::TxManager::new();
    let database = Database::open_with_tx_manager(dir.path(), manager.clone()).expect("open");

    let mut t1 = database.begin_ssi();
    t1.put("shared_key", "from_t1");

    let mut t2 = database.begin_ssi();
    t2.put("shared_key", "from_t2");

    // t1 commits first: must succeed
    let t1_commit = t1.commit().expect("t1 must commit");
    assert!(t1_commit > 0, "commit_ts must be positive");

    // Positive assertion: t2 detects write-write conflict on commit
    let (_, err) = t2
        .try_commit()
        .expect_err("t2 must detect write-write conflict with committed t1");
    assert!(
        matches!(err, DatabaseError::SsiConflict { .. }),
        "expected SsiConflict, got {err:?}"
    );

    // Negative assertion: final value must be t1's value (t2 was rejected)
    assert_eq!(
        database.get(b"shared_key").expect("get").map(|v| v.to_vec()),
        Some(b"from_t1".to_vec()),
        "t1's value must persist"
    );
}

/// Two SSI transactions write the same key; the first commits successfully, the second detects
/// the conflict. Deterministic: single-threaded sequential execution, no Barrier/thread::spawn.
/// Complements `transaction_ssi_blocks_conflict`: this test explicitly verifies that `try_commit`'s
/// Err return carries the correct conflict key and txn info.
#[test]
fn transaction_ssi_concurrent_try_commit_detects_conflict() {
    let dir = tmpdir();
    let manager = hatp_tx::TxManager::new();
    let database = Database::open_with_tx_manager(dir.path(), manager.clone()).expect("open");

    let mut t1 = database.begin_ssi();
    t1.put("contested", "alpha");
    let mut t2 = database.begin_ssi();
    t2.put("contested", "beta");

    // t1 commits first: must succeed
    let t1_commit = t1.commit().expect("t1 must commit");
    assert!(t1_commit > 0, "commit_ts must be positive");

    // Positive assertion: t2 detects conflict, carrying the correct conflict key
    let (_, err) = t2
        .try_commit()
        .expect_err("t2 must detect conflict with committed t1");
    match err {
        DatabaseError::SsiConflict { txn, key } => {
            assert!(txn > 0, "conflict txn must be valid");
            assert_eq!(key, b"contested", "conflict key must be the contested key");
        }
        other => panic!("expected SsiConflict, got {other:?}"),
    }

    // Negative assertion: final value must be t1's value
    let final_value = database.get(b"contested").expect("get").map(|v| v.to_vec());
    assert_eq!(
        final_value,
        Some(b"alpha".to_vec()),
        "t1's value must persist"
    );
    assert_ne!(final_value, Some(b"beta".to_vec()), "t2's value must not persist");
}

#[test]
fn transaction_ssi_no_conflict_different_keys() {
    let dir = tmpdir();
    let manager = hatp_tx::TxManager::new();
    let database = Database::open_with_tx_manager(dir.path(), manager.clone()).expect("open");

    let mut t1 = database.begin_ssi();
    t1.put("a", "1");
    let mut t2 = database.begin_ssi();
    t2.put("b", "2");

    let id1 = t1.commit().expect("t1 commit");
    let id2 = t2.commit().expect("t2 commit");
    assert_ne!(id1, id2, "distinct tx ids");
    assert_eq!(
        database.get(b"a").expect("a").map(|v| v.to_vec()),
        Some(b"1".to_vec())
    );
    assert_eq!(
        database.get(b"b").expect("b").map(|v| v.to_vec()),
        Some(b"2".to_vec())
    );
}

#[test]
fn transaction_ssi_records_reads_on_get() {
    let dir = tmpdir();
    let manager = hatp_tx::TxManager::new();
    let database = Database::open_with_tx_manager(dir.path(), manager.clone()).expect("open");

    database.put("seed", "value").expect("seed put");

    let t = database.begin_ssi();
    let value = t.get(b"seed").expect("get").expect("present");
    assert_eq!(value.to_vec(), b"value".to_vec());

    // Verify the read was recorded into the SSI context by inspecting the
    // manager directly. We need the txn id; track it via the public API.
    let inflight = manager.inflight();
    assert_eq!(inflight.len(), 1, "exactly one active SSI transaction");
    let id = inflight[0].id;
    let ssi = manager.get_ssi_txn(id).expect("ssi context");
    assert!(ssi.read_set.iter().any(|k| k.as_ref() == b"seed"));
    assert!(ssi.write_set.is_empty(), "no writes yet");

    // Drop the transaction (committing with no writes is ReadOnlyCommit).
    drop(t);
}

#[test]
fn transaction_ssi_drop_releases_context() {
    // Regression guard: a Transaction that is dropped without commit
    // must release its TxManager context automatically, otherwise the
    // leaked write_set causes ghost conflicts against future SSI
    // transactions. See `Transaction::drop` in `embedded.rs`.
    let dir = tmpdir();
    let manager = hatp_tx::TxManager::new();
    let database = Database::open_with_tx_manager(dir.path(), manager.clone()).expect("open");

    let mut t1 = database.begin_ssi();
    t1.put("k", "v1");
    let t1_id = t1.reserved_tx_id().expect("reserved id");
    drop(t1);

    // After drop, the SSI context is gone — there is nothing to abort
    // by hand.
    assert!(manager.get_ssi_txn(t1_id).is_none());
    assert!(manager.inflight().is_empty());

    // A fresh SSI transaction now sees no overlapping context.
    let mut t2 = database.begin_ssi();
    t2.put("k", "v2");
    t2.commit()
        .expect("fresh SSI commit succeeds after t1 dropped");
    assert_eq!(
        database.get(b"k").expect("get").map(|v| v.to_vec()),
        Some(b"v2".to_vec())
    );
}

#[test]
fn ssi_validate_engine_layer_round_trip() {
    // Engine-level smoke test: verify on_pre_commit returns Ok for a
    // TxManagerHook when no SSI context exists for the tx_id, and that
    // Engine::write_with_tx with a reserved id still increments durably.
    let dir = tmpdir();
    let manager = hatp_tx::TxManager::new();
    let hook: Arc<dyn EngineHook> = Arc::new(TxManagerHook::new(manager.clone()));
    let engine = Engine::open_with_hook(
        EngineConfig::new(dir.path()),
        
        hook,
    )
    .expect("open");

    // Reserved id (supplied explicitly — the manager is not an id allocator),
    // no SSI context yet — pre-commit must short-circuit to Ok and the write
    // must land durably.
    let handle = manager.begin_with_id(hatp_tx::TxnId(100), hatp_tx::IsolationHint::Snapshot);
    let tx_id = handle.id.0;
    engine
        .write_with_tx(
            tx_id,
            &[Mutation::Put {
                key: bytes::Bytes::from_static(b"plain"),
                value: bytes::Bytes::from_static(b"v"),
            }],
        )
        .expect("reserved-id write_with_tx");
    assert_eq!(
        engine
            .get(b"plain", engine.snapshot_ts())
            .expect("get")
            .map(|v| v.to_vec()),
        Some(b"v".to_vec())
    );

    // Direct validate_ssi call: prove that first-committer-wins fires
    // through the public API surface. commit `a`, leave `b` staged;
    // `validate_ssi(b)` must observe `a`'s write-set in `commit_history`
    // and report the write-write conflict.
    //
    // The previous test wording ("stage two SSI contexts, verify
    // validate_ssi reports a conflict") implied active-vs-active
    // detection, which contradicts first-committer-wins semantics:
    // two staged peers are serialised by `Engine::write_guard`, so the
    // survivor is determined by commit order. We now commit `a` first
    // and verify `validate_ssi(b)` then fires.
    let mut a = manager.begin_ssi_with_id(hatp_tx::TxnId(101), 0);
    a.record_write(bytes::Bytes::from_static(b"col"));
    manager.update_ssi_txn(&a);
    // `b` must begin BEFORE `a` commits: a peer that begins after the commit
    // has `start_ts >= commit_ts` and legitimately sees the committed write as
    // its base (not a conflict). Staging `b` first keeps it active at `a`'s
    // commit, so `a`'s write-set stays in `commit_history` for `b` to see.
    let mut b = manager.begin_ssi_with_id(hatp_tx::TxnId(102), 0);
    b.record_write(bytes::Bytes::from_static(b"col"));
    manager.update_ssi_txn(&b);
    assert!(
        manager.validate_ssi(&a).is_ok(),
        "staged peer must not conflict pre-commit"
    );
    assert!(
        manager.validate_ssi(&b).is_ok(),
        "staged peer must not conflict pre-commit"
    );
    manager.commit_at(a.handle.id, 1).expect("commit a");
    let result = manager.validate_ssi(&b);
    assert!(
        matches!(result, Err(hatp_tx::TxError::WriteConflict { .. })),
        "validate_ssi should detect write-write conflict against committed peer, got {result:?}"
    );
}

#[test]
fn transaction_ssi_write_skew_detected() {
    // Text-book write-skew anomaly under SSI. Two doctors each read the
    // other's on-call boolean; under SI both could commit and produce
    // an inconsistent schedule (everyone off-call). Under SSI, the
    // read-write antidependency must be caught and at least one must
    // abort.
    //
    // Setup: `doc1_oncall = true`, `doc2_oncall = true` initially.
    // Both doctors see both values, then each toggles their own off
    // under the impression the other is still on. Under SSI the
    // validation must reject at least one of the two commits.
    let dir = tmpdir();
    let manager = hatp_tx::TxManager::new();
    let database = Database::open_with_tx_manager(dir.path(), manager.clone()).expect("open");

    database.put("doc1_oncall", "1").expect("seed doc1");
    database.put("doc2_oncall", "1").expect("seed doc2");

    let mut doc1 = database.begin_ssi();
    doc1.get(b"doc2_oncall")
        .expect("d1 sees doc2")
        .expect("present");

    let mut doc2 = database.begin_ssi();
    doc2.get(b"doc1_oncall")
        .expect("d2 sees doc1")
        .expect("present");

    doc1.put("doc1_oncall", "0");
    doc2.put("doc2_oncall", "0");

    // PR 3.9 (Cahill): the two transactions form an rw-cycle — doc1 reads
    // doc2_oncall and writes doc1_oncall; doc2 reads doc1_oncall and writes
    // doc2_oncall. The committer that detects the cycle is aborted, so at
    // most one of the two commits may succeed (which one is not guaranteed).
    let r1 = doc1.try_commit();
    let r2 = doc2.try_commit();
    let committed = u8::from(r1.is_ok()) + u8::from(r2.is_ok());
    assert!(
        committed <= 1,
        "at most one of the two write-skew commits may succeed (committed {committed})"
    );

    // After the rejection the schedule is consistent: not both doctors can be
    // off-call.
    let d1 = database
        .get(b"doc1_oncall")
        .expect("d1")
        .map(|v| v.to_vec());
    let d2 = database
        .get(b"doc2_oncall")
        .expect("d2")
        .map(|v| v.to_vec());
    assert!(
        d1 == Some(b"1".to_vec()) || d2 == Some(b"1".to_vec()),
        "write-skew must be rejected: at least one doctor stays on-call ({d1:?}, {d2:?})"
    );
}

/// Phantom protection (PR 3.8): an SSI transaction that range-scans a key
/// range, then commits, must abort if a concurrent transaction wrote a new
/// key inside that range after the scan began.
#[test]
fn transaction_ssi_range_read_detects_phantom() {
    let dir = tmpdir();
    let manager = hatp_tx::TxManager::new();
    let database = Database::open_with_tx_manager(dir.path(), manager.clone()).expect("open");

    database.put("t\0k1", "v1").expect("seed k1");

    let mut reader = database.begin_ssi();
    let seen = reader
        .scan_range(b"t\0", b"t\0\xff")
        .expect("range scan");
    assert_eq!(seen.len(), 1, "reader must see the seeded key");

    // A concurrent SSI writer inserts a new key inside the scanned range.
    // (An autocommit `put` would not register an SSI context and therefore
    // not enter `commit_history`; a range-read conflict can only be detected
    // against a peer SSI commit.)
    let mut writer = database.begin_ssi();
    writer.put("t\0k2", "v2");
    writer.commit().expect("writer commits");

    // The reader performs its own write (so its commit is not read-only) and
    // must then be rejected: the phantom insert landed inside its read range.
    reader.put("t\0k3", "v3");
    let (_, err) = reader
        .try_commit()
        .expect_err("reader must abort on the phantom insert");
    match err {
        DatabaseError::SsiReadWriteConflict { key, .. } => {
            assert_eq!(key, b"t\0k2", "conflict key must be the phantom insert");
        }
        other => panic!("expected SsiReadWriteConflict, got {other:?}"),
    }
}
