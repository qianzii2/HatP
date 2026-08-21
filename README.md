# HatP — An HTAP Engine in ~10,000 Lines of Code

HatP is an embedded HTAP (Hybrid Transactional/Analytical Processing) database engine. It delivers both OLTP (MVCC + WAL + LSM-Tree) and OLAP (SQL + columnar scans) in a single process, with performance to match. ~10,000 lines of production code. Add comments, tests, fuzz targets, formal verification, and the rest of the infrastructure, and the whole project comes to about 23,000 lines.

```
                  ┌──────────────────────────────────────────┐
                  │                  hatp                    │
                  │     Database / Transaction Facade        │
                  │  put / get / delete / scan / execute_sql │
                  └────────┬──────────────────┬──────────────┘
                           │                  │
              ┌────────────▼──────┐  ┌────────▼──────────────┐
              │  hatp-frontend    │  │     hatp-tx            │
              │  DataFusion       │  │  SSI Transaction Mgr   │
              │  Catalog / DDL    │  │  Cahill Cycle Detection│
              │  TableProvider    │  │  First-committer-wins  │
              └────────┬──────────┘  └────────┬──────────────┘
                       │                      │
                       └──────────┬───────────┘
                                  │
                       ┌──────────▼──────────┐
                       │    hatp-engine       │
                       │  MVCC KV + WAL       │
                       │  LSM-Tree + Vortex   │
                       │  Compaction / Bloom  │
                       └──────────┬──────────┘
                                  │
                       ┌──────────▼──────────┐
                       │    hatp-types        │
                       │  TxnTs / Codec / Hash│
                       └─────────────────────┘
```

## Why so few lines of code?

### Because I leaned on mature ecosystems

DataFusion (SQL parsing, optimizer, execution engine)

Vortex (columnar encoding, predicate pushdown, zone maps)

Arrow (in-memory columnar format, IPC)

crossbeam-skiplist (concurrent SkipMap)

rayon (data-parallel compaction)

tokio (async runtime)

memmap2 (zero-copy I/O)

### And I went all-in on the parts I own

MVCC version chains + SSI conflict detection

WAL / MANIFEST custom binary formats

LSM-Tree (flush / compaction)

MemTable (SkipMap, lock-free)

Bloom + KeyIndex sidecar indexes

CRC32C (SSE4.2 hardware-accelerated)

SILK backpressure + fault injection

### Why not reuse an existing OLTP engine too?

I tried. [Didn't end well 😂 — after a bunch of hacks I gave up and decided to build something more engineered.](https://github.com/qianzii2/rockduck) So the OLTP side is hand-rolled. One, DataFusion and Vortex are on a strong trajectory, and owning the OLTP layer lets me integrate with them tightly instead of piling up adapter cruft. Two, no SQL to write — a pure KV layer is a manageable amount of work.

## How I pulled it off

Same data, two workloads: high-frequency point lookups on one side, full-table analytical scans on the other.

### Write path (OLTP)

```
INSERT/UPDATE/DELETE
  → WAL append + fsync (group commit for batched durability)
  → MemTable write (SkipMap, lock-free reads)
  → threshold reached → flush to Vortex columnar SST
  → background LSM-Tree compaction merges SSTs
```

### Read path — two lanes

Point lookup (OLTP):

```
get(key) → MemTable lookup → hit? return
         → miss → Bloom filter (skip SSTs that don't contain this key)
                → KeyIndex binary search (O(log n), no Vortex file open)
                → Vortex columnar read (last resort)
```

Analytical query (OLAP):

```
SELECT ... WHERE ...
  → DataFusion parses SQL + optimizes
  → TableProviderAdapter pushes the query down to the engine
  → Vortex → Arrow direct path: business columns read zero-copy from Vortex into Arrow
  → DataFusion runs vectorized filter, aggregation, JOIN
```

### How OLTP and OLAP stay out of each other's way

- The write path holds `write_guard` (ReentrantMutex), serializing all writes
- The read path takes no locks at all: MemTable reads go through `ArcSwap` for a consistent snapshot; SST reads use mmap, zero-copy
- Compaction has its own lock (`compaction_guard`), never blocking writes
- SILK backpressure: when the MemTable nears capacity, writes automatically slow down, buying time for flush

## The OLTP side — design details

### MVCC version chains

- `SmallVec<[VersionedValue; 4]>`: 99% of keys have 1–2 versions, no heap allocation under 4
- `partition_point` binary insertion: O(log n) instead of O(n) sort
- `commit_ts` decoupled from `tx_id`: commit order ≠ reservation order, so a snapshot never sees uncommitted data

### SSI conflict detection

- First-committer-wins: two concurrent transactions writing the same key are serialized by `write_guard` at pre-commit. The first to commit enters `commit_history`; the second's `validate_ssi` catches the conflict.
- Read-write antidependency: catches write-skew anomalies
- Cahill cycle detection: T1 reads X writes Y, T2 reads Y writes X → at least one aborts
- Range-read conflicts: a concurrent insert inside a previously scanned range is detected as a phantom
- Group-commit provisional staging: conflicting transactions in the same WAL batch are also detected

### WAL (custom binary format)

- WAL1 format: 28 bytes of fixed overhead per frame (magic 4 + tx_id 8 + op 1 + key_len 4 + value_len 4 + CRC32C 4)
- Versus Arrow IPC's 200+ bytes per frame: for a single-row write (~80 bytes of payload), the payload-to-overhead ratio goes from ~25% to ~74%
- Torn tail frames are auto-truncated on recovery

### LSM-Tree Compaction

- SILK-style priority scheduling (MinOverlappingRatio)
- Age-based tie-breaking (`created_at` timestamps)
- Parallel sub-compaction (rayon data-parallel)
- Watermark protection: versions referenced by active snapshots are never reclaimed

### MemTable

- Default: `crossbeam_skiplist::SkipMap` + `ArcSwap` — writers clone the chain and atomically swap the pointer; readers take no locks
- Backward-compatible `RwLock<BTreeMap>` backend (proptest verifies both backends are equivalent)

## The engineering side

### Correctness verification

| Tier | Tool | Coverage |
|------|------|----------|
| Property testing | proptest (5,000 cases) | VersionChain newest-first, BTreeMap ≡ SkipMap equivalence |
| Fuzz testing | 6 libfuzzer targets | Every binary parse entry point: WAL, Bloom, KeyIndex, Manifest, RowCodec, Vortex SST |
| Formal verification | 9 Kani proofs | CRC32C single-bit-flip detection, WAL encode-decode equivalence, sign_flip_be order preservation, float total order |
| Concurrency testing | 6 loom tests | SkipMap RCU no lost update, group commit, Watcher monotonicity |
| Deterministic stress | stress_runner (seed-controlled) | Engine vs MirrorStore state consistency |
| Runtime detection | Miri / TSan / ASan / cargo-careful | CI-automated UB / data race / memory error detection |

### Crash recovery

Torn WAL tail, WAL frame CRC corruption, crash mid-flush, crash mid-compaction, mixed WAL+SST recovery, empty WAL first startup, corrupt SST recovery — every scenario has a matching integration test.

### Fault injection

SST file externally deleted, SST content corrupted, WAL file deleted, WAL truncated to zero bytes, WAL overwritten with random garbage, MANIFEST file deleted, MANIFEST corrupted, disk full during flush, corrupt input SST during compaction.

### Discipline

- Workspace-level `unsafe_code = "forbid"` (engine allows it where needed, with `// SAFETY:` comments)
- `unwrap_used = "deny"`, `expect_used = "deny"`, `panic = "deny"`
- `todo = "deny"`, `unimplemented = "deny"`
- Every `unsafe` block carries a `// SAFETY:` comment explaining the preconditions
- `cargo deny check` audits the supply chain for vulnerabilities and license compliance

## Quick start

```rust
use hatp::Database;
use bytes::Bytes;

// Open or create a database
let db = Database::open("/tmp/mydb")?;

// OLTP: auto-commit writes
db.put(Bytes::from_static(b"key"), Bytes::from_static(b"value"))?;
let value = db.get(b"key")?;

// OLTP: transactions
let mut tx = db.begin();
tx.put(Bytes::from_static(b"a"), Bytes::from_static(b"1"));
tx.put(Bytes::from_static(b"b"), Bytes::from_static(b"2"));
tx.commit()?;

// SSI serializable transactions
let db = Database::open_with_tx_manager(path, manager)?;
let mut tx = db.begin_ssi();
tx.put(b"shared", b"value");
match tx.try_commit() {
    Ok(commit_ts) => { /* success */ }
    Err((tx, DatabaseError::SsiConflict{..})) => { /* retry */ }
}

// OLAP: SQL queries
let outcome = db.execute_sql("SELECT * FROM my_table WHERE age > 30").await?;
println!("{} rows returned", outcome.rows);
```

## Build & test

```bash
cargo build --release
cargo test --profile test
cargo test -p hatp-engine                     # engine tests
cargo +nightly fuzz run fuzz_wal_decode       # fuzz testing
cargo kani --package hatp-engine --harness kani_crc32c_detects_single_bit_flip  # formal verification
cargo bench -p hatp-engine                    # benchmarks
```

## Crate organization

| Crate | Role | Dependencies |
|-------|------|-------------|
| hatp-types | Shared types, codecs, hashing | Only bytes + arrow + DataFusion ScalarValue |
| hatp-engine | Storage engine: MVCC + WAL + LSM + Vortex SST | No DataFusion dependency |
| hatp-tx | Transaction layer: SSI state machine | Only engine + types |
| hatp-frontend | SQL frontend: DataFusion + Catalog + DDL/DML | Never touches WAL/SST directly |
| hatp | Top-level facade: Database / Transaction | Pure glue layer |

## Current limitations

- Single-machine embedded only (no network layer)
- Column pruning in Vortex 0.83's `scan()` path is not yet fully implemented (waiting on upstream)

## License

Apache-2.0