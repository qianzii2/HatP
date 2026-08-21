# hatp-engine — Durable MVCC Storage Engine

**Role**: The OLTP core of HatP. Provides MVCC key-value storage, LSM-Tree, WAL durability,
Vortex columnar SSTs, and compaction.

**Boundary**: No DataFusion dependency. No awareness of SQL, Catalog, or transaction managers.
Exposes only byte-level KV operations and RecordBatch ingestion interfaces.

## Data Flow

```
  Write Path                    Read Path
  ──────────                    ─────────
  Mutation                      get(key, snapshot)
    │                             │
    ▼                             ▼
  WAL (append+fsync)            MemTable (SkipMap/BTreeMap)
    │                             │ miss
    ▼                             ▼
  MemTable.apply()              Bloom → KeyIndex → Vortex SST
    │                             │
    ▼                             ▼
  flush → Vortex SST            Return VersionedValue
    │
    ▼
  Compaction (L0→L1+)
```

## Module Map

| Module | Content | Key Types |
|--------|---------|-----------|
| `lib.rs` | Engine struct, EngineConfig, EngineError, Mutation, EngineHook | `Engine`, `EngineConfig` |
| `version.rs` | MVCC version chains, snapshots, GC | `VersionChain`, `VersionedValue`, `Snapshot` |
| `memtable.rs` | Dual-backend memtable (BTreeMap / SkipMap) | `MemTable`, `MemTableBackend` |
| `wal.rs` | Custom binary WAL (WAL1 format) | `Wal`, `WalRecord`, `OpType` |
| `manifest.rs` | MANIFEST persistence (MAN1 format), VersionSet | `Manifest`, `VersionSet`, `VersionEdit` |
| `vortex_sst.rs` | Vortex columnar SST read/write | `SstHandle`, `write_async`, `scan_to_arrow_batches` |
| `compaction.rs` | Compaction strategy pickers | `CompactionPicker`, `CompactionJob`, `FileMeta` |
| `bloom.rs` | Per-SST Bloom filter (FNV-1a + DJB2) | `BloomFilter` |
| `key_index.rs` | Per-SST binary-search Key index (KIDX format) | `KeyIndex` |
| `row_codec.rs` | Compact row codec | `encode_row`, `decode_row_values` |
| `crc32c.rs` | CRC32C (software slicing-by-8 + SSE4.2 hardware) | `crc32c`, `Hasher` |
| `column_predicate.rs` | Single-column predicates (Eq/Range) | `ColumnPredicate`, `Bound` |
| `sst_format.rs` | SST format abstraction | `SstFormat`, `VortexFormat`, `ZoneMap` |
| `metrics.rs` | Prometheus-compatible metrics | `EngineMetrics`, `MetricsSnapshot` |
| `pressure.rs` | SILK backpressure throttling | `PressureThrottle`, `ThrottleLevel` |
| `clock.rs` | Injectable time source | `Clock`, `SystemClock`, `ManualClock` |
| `watch.rs` | CDC change-data-capture | `Watcher`, `WatchEvent` |
| `fault_injector.rs` | Fault injection framework | `fault_point!`, `FaultInjector` |

## Design Decisions

### Why Vortex for SSTs?

Vortex generates zone maps and sparse indexes per column inside the file, enabling DataFusion to prune
columns and push down predicates without decoding the opaque value blob. Compared to the old Arrow IPC
SST layout, OLAP scans only touch the relevant column chunks.

### Why KeyIndex as a separate sidecar instead of a Vortex built-in index?

Vortex's columnar scan path is heavyweight for OLTP point queries — even a "filtered scan" touches
metadata for every column chunk. The KeyIndex is a sorted binary sidecar; a point lookup is a single
binary search (`log2(100k) ≈ 17` string comparisons) with no Vortex file open.

### Why `SmallVec<[VersionedValue; 4]>` for VersionChain?

In steady-state OLTP, 99% of keys have 1–2 versions. Heap allocation only kicks in past 4 versions
(a long contention history, rare). The `newest-first` ordering lets `get(snapshot)` return on the
first check of `versions[0]` in the overwhelming majority of cases.

### Why custom binary formats for WAL and MANIFEST?

See the root README's "Design Decision #2". Per-frame overhead: WAL = 28 bytes, MANIFEST = 9 bytes,
vs. Arrow IPC's 200+ bytes.

## Correctness Guarantees

| Tier | Tool | Scope |
|------|------|-------|
| Unit tests | cargo test | 50+ |
| Integration tests | `tests/` directory | 28 test files |
| Crash recovery | `tests/crash_recovery.rs` | 7 scenarios |
| Fault injection | `tests/fault_injection_integration.rs` | 9 scenarios |
| Boundary values | `tests/boundary_dark.rs` | 14 boundary keys × 5 boundary values |
| Property testing | proptest | 5,000 sequences (VersionChain, memtable differential) |
| Stress testing | `tests/stress_runner.rs` | Seed-controlled op sequences + MirrorStore verification |
| Fuzz testing | 6 fuzz targets | No panic on arbitrary bytes |
| Formal verification | Kani | 7 engine-level proofs |
| Concurrency testing | Loom | 6 concurrency tests |
| Runtime detection | Miri/TSan/ASan | CI automated |