# hatp-frontend — DataFusion Integration & SQL Frontend

**Role**: HatP's OLAP side. Bridges DataFusion's SQL engine with `hatp-engine`'s storage layer,
providing Catalog management, DDL/DML planning, predicate translation, and a `TableProvider` adapter.

**Boundary**: Owns the DataFusion dependency. Never touches WAL or SST files directly — all storage
access goes through `hatp_engine::Engine`'s public API.

## Module Map

| Module | Content | Key Types |
|--------|---------|-----------|
| `catalog.rs` | In-memory catalog (3-tier: catalog→schema→table), JSON snapshot persistence | `Catalog`, `CatalogSnapshot` |
| `schema.rs` | DDL types: `CatalogId`, `SchemaId`, `QualifiedName`, `TableSchema`, `CreateTable`, `DropObject` | `TableSchema`, `CreateTable` |
| `execution.rs` | DataFusion `SessionContext` integration, `TableProviderAdapter` | `FrontendSession`, `TableProviderAdapter` |
| `dml_planner.rs` | INSERT/UPDATE/DELETE plan translation (DataFusion Expr → Mutation) | `plan_insert`, `plan_delete`, `plan_update` |
| `predicate_translator.rs` | DataFusion Expr → `ColumnPredicate` translation | `expr_to_column_predicates` |

## DML Data Flow

```
SQL: INSERT INTO t SELECT ...
SQL: DELETE FROM t WHERE ...
SQL: UPDATE t SET col = expr WHERE ...
  │
  ▼
DataFusion LogicalPlan / Expr  ← user-supplied SQL
  │
  ▼
predicate_translator::expr_to_column_predicates()
  │
  ├─ col = literal  → ColumnPredicate::Eq
  ├─ col < literal  → ColumnPredicate::LessThan
  ├─ col > literal  → ColumnPredicate::GreaterThan
  └─ AND chains recursively decomposed
  │
  ▼
dml_planner::plan_insert / plan_delete / plan_update
  │
  ├─ Fast path: PK predicates → engine.delete_with_column_predicates()
  └─ Generic path: full table scan + DataFusion vectorized filter
  │
  ▼
Vec<Mutation>  → Engine::write()
```

## Design Decisions

### Why doesn't the engine accept DataFusion Expr?

`hatp-engine` is the OLTP core. Keeping DataFusion entirely out of its dependency graph lets the engine
be reused without a DataFusion runtime. The cost is that DML features must be re-implemented in
`hatp-frontend` — this is intentional layering separation.

### Why does TableProviderAdapter::scan use the Vortex→Arrow direct path?

The old approach decoded each SST's opaque value blob row-by-row into `RecordBatch`es. The new approach
(`scan_to_arrow_batches`) reads business columns (`col_0`, `col_1`, ...) directly from Vortex and converts
them to Arrow arrays via `vortex-arrow` with zero-copy. DataFusion can prune columns at the Vortex layer —
only the requested columns are read.

### Why does `statistics()` use a cache with version invalidation?

DataFusion calls `statistics()` on every query planning. Without caching, every call re-aggregates all
SST zone maps. The cache version comes from `engine.stats_version()`, which increments after each
flush/compaction, invalidating the cache.

## Verification

| Method | Coverage |
|--------|----------|
| Unit tests | Catalog snapshot roundtrip, TableSchema validation, DataFusion integration |
| DML integration tests | UPDATE OR/LIKE/CAST, DELETE OR/LIKE/non-PK filters |