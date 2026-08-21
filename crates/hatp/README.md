# hatp — Embedded HTAP Database Facade

**Role**: The top-level glue crate of HatP. Fuses `hatp-engine` (OLTP storage), `hatp-tx` (SSI transactions),
and `hatp-frontend` (DataFusion SQL) into a single `Database` handle, providing synchronous
`put` / `get` / `delete` / `scan` / `execute_sql` entry points.

**Boundary**: Pure glue layer. Implements no storage, transaction, or SQL logic — all work is delegated
to sub-crates.

## Public API

```rust
// Open a database
let db = Database::open("/path/to/data")?;

// Auto-commit
db.put(b"key", b"value")?;
let v = db.get(b"key")?;
db.delete(b"key")?;

// Transactions (Snapshot Isolation)
let mut tx = db.begin();
tx.put(b"a", b"1");
let v = tx.get(b"a")?;  // read-your-writes
tx.commit()?;

// SSI Transactions (Serializable Snapshot Isolation)
let db = Database::open_with_tx_manager(path, manager)?;
let mut tx = db.begin_ssi();
tx.put(b"shared", b"value");
match tx.try_commit() {
    Ok(commit_ts) => /* success */,
    Err((tx, DatabaseError::SsiConflict{..})) => /* retry */,
}

// SQL queries
let outcome = db.execute_sql("SELECT id, name FROM users WHERE age > 30").await?;

// Register a table
db.register_table(table_schema)?;
```

## Catalog Persistence

`Database` persists the catalog to `catalog.json` at:
- Every 16 DDL operations (`CATALOG_SAVE_THRESHOLD`)
- `Database::drop` (last handle)

Persistence uses atomic `tmp + rename + fsync`, co-located with the engine's WAL/SST files.

## Design Decisions

### Why is `Database` a pure facade with no logic in this crate?

The 5 sub-crates have a strict one-way dependency direction: `types → engine → tx → frontend → hatp`.
`hatp` only does `Arc` wrapping and delegation. This lets each sub-crate be tested and understood
independently, and `hatp-engine` can be used directly in DataFusion-free scenarios.

### Why does `Transaction::try_commit` return `Result<u64, (Transaction, DatabaseError)>`?

SSI conflicts are recoverable — the caller may want to retry. `try_commit` returns the original
`Transaction` (with all buffered reads/writes) on failure, so the caller can inspect the conflict key
and decide whether to retry. The `Result` is intentionally unboxed (`clippy::result_large_err` is
allowed) because the error path is rare and boxing would add an allocation and indirection on every retry.

### Why does `Transaction::drop` auto-release SSI context?

The old implementation required callers to manually call `manager.abort(id)`, otherwise the leaked
`write_set` caused ghost conflicts — future transactions saw the dead transaction's write-set and
aborted spuriously. `Drop` automatically calls `manager.abort()`, which is idempotent for already
committed/aborted transactions.

## Verification

| Method | Coverage |
|--------|----------|
| Integration tests | Auto-commit roundtrip, transaction read-your-writes, snapshot isolation, restart recovery, catalog recovery |
| SSI end-to-end | 10 scenarios: first-committer-wins, RW antidependency, Cahill cycle, range phantom, context release, write-skew doctors example |