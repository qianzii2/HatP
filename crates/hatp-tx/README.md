# hatp-tx — Transaction Layer (SSI State Machine + EngineHook)

**Role**: HatP's transaction management layer. Provides Serializable Snapshot Isolation (SSI) semantics,
injecting conflict detection into the engine's pre-commit hook via `TxManagerHook`.

**Boundary**: Only depends on `hatp-engine` and `hatp-types`. No awareness of DataFusion, Catalog, or SQL.

## Invariants

| # | Invariant | Location |
|---|-----------|----------|
| TX1 | First-committer-wins: two concurrent txns writing the same key — first commits, second detects WriteConflict at `validate_ssi` | `manager.rs:validate_ssi` |
| TX2 | Read-write antidependency: a peer committed a write to a key this txn read *after* this txn began → ReadWriteConflict | `manager.rs:validate_ssi` |
| TX3 | Cahill rw-cycle: T1 reads X writes Y, T2 reads Y writes X → at least one aborted | `manager.rs:validate_cahill` |
| TX4 | An already-aborted txn cannot be re-committed (prevents retry-bypassing conflict detection) | `hook.rs:on_pre_commit` |
| TX5 | commit_history is GC'd by watermark, never evicting entries still visible to a long-running txn | `manager.rs:CommitHistory` |
| TX6 | Group-commit staging conflict: two conflicting txns in one WAL batch — the second detects the first's provisional write-set | `manager.rs:stage_commit` |

## Module Map

| Module | Content | Key Types |
|--------|---------|-----------|
| `manager.rs` | SSI state machine, conflict detection, commit_history, group commit | `TxManager`, `SsiTxn`, `CommitHistory` |
| `hook.rs` | EngineHook impl, injected into the engine's pre-commit path | `TxManagerHook` |
| `error.rs` | Transaction-layer error types | `TxError` |

## SSI Conflict Detection Flow

```
Transaction::try_commit()
  │
  ▼
Engine::write_with_tx(tx_id, mutations)
  │
  ▼
EngineHook::on_pre_commit(tx_id, mutations)  ← TxManagerHook
  │
  ├─ 1. Check if tx_id already Aborted (reject retry)
  ├─ 2. Collect mutation keys → record_write(tx_id, keys)
  ├─ 3. validate_ssi(tx_id):
  │     ├─ Check commit_history for WriteConflict
  │     ├─ Check commit_history for ReadWriteConflict
  │     ├─ Check Cahill rw-cycle (validate_cahill)
  │     └─ Check provisional staging (group-commit)
  └─ 4. Return Ok(()) or EngineError::WriteConflict / ReadWriteConflict
  │
  ▼
WAL append + fsync
  │
  ▼
EngineHook::on_tx_commit(tx_id, commit_ts)
  │
  └─ TxManager::commit_at(tx_id, commit_ts) → write to commit_history
```

## Design Decisions

### Why first-committer-wins instead of first-writer-wins?

`Engine::write_guard` already serializes pre-commit. Two staged SSI peers do not conflict with each other —
they are resolved by commit order. First-writer-wins would cause mutual aborts (both detect the other wrote
their key); first-committer-wins aborts only one.

### Why watermark-bounded instead of fixed-capacity commit_history?

The old implementation used a `BTreeMap` capped at `COMMIT_HISTORY_CAP = 1024`. A long-running transaction
that started before 1,024 short transactions committed would lose the entries it needed to detect conflicts
(R-03). Watermark-bounded history only drops entries with `commit_ts <= min(active start_ts)`, guaranteeing
long transactions see everything they need.

### Why `HashSet<Bytes>` instead of `Vec<Bytes>` for read_set/write_set?

Long transactions can touch tens of thousands of keys. `Vec::contains` is O(n) per key (the R-22 performance
cliff). `HashSet` makes `validate_ssi` membership tests O(1) for all transaction sizes, at a small constant-factor
cost for the O(1–10) key OLTP case.

## Verification

| Method | Coverage |
|--------|----------|
| Unit tests | 25+ scenarios (lifecycle, conflict detection, Cahill cycle, group commit, GC stress) |
| Integration tests (hatp) | Full SSI pipeline: conflict/no-conflict/read-set recording/context release/write-skew/phantom |