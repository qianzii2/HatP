//! DataFusion integration surface.
//!
//! The frontend owns:
//! - [`FrontendSession`] — bundles a [`SessionContext`] with the catalog.
//!   SQL goes through [`FrontendSession::execute`], which
//!   truly runs against DataFusion (no plan-only stub).
//! - [`TableProviderAdapter`] — bridges [`TableSchema`] into DataFusion's
//!   [`TableProvider`]. With an attached engine, `scan` produces a real
//!   `MemoryExec` over rows read from `hatp-engine`; DML routes through
//!   `engine.write(&mutations)`.
//! - Secondary-index-matched filters are pushed down; other predicates are
//!   filtered by DataFusion over the materialized batches (see the structural
//!   note in this module on why business-column push-down is not possible).

use std::fmt;
use std::sync::Arc;

use crate::catalog::Catalog;
use crate::dml_planner;
use crate::predicate_translator::expr_to_column_predicates;
use crate::schema::TableSchema;
use arrow_array::{Array, RecordBatch};
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;
use datafusion_catalog::MemoryCatalogProvider;
use datafusion_catalog::MemorySchemaProvider;
use datafusion_catalog::TableProvider;
use datafusion_catalog::{CatalogProvider, SchemaProvider};
use datafusion_common::Constraints;
use datafusion_common::Result as DfResult;
use datafusion_datasource::memory::MemorySourceConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_expr::Expr;
use datafusion_expr::TableProviderFilterPushDown;
use datafusion_expr::TableType;
use datafusion_expr::dml::InsertOp;
use datafusion_physical_plan::empty::EmptyExec;
use datafusion_session::Session;
use hatp_engine::column_predicate::ColumnPredicate;





/// Result of executing a single SQL statement.
#[derive(Debug, Clone)]
pub struct SessionExecutionOutcome {
    /// Number of rows produced by the query.
    pub rows: usize,
    /// Number of [`RecordBatch`]es returned by `collect()`.
    pub batches: usize,
    /// First few rows of the first batch (capped at 5) for CLI display.
    pub head: Vec<String>,
}

/// One-stop helper that bundles everything the frontend exposes.
pub struct FrontendSession {
    /// Catalog (logical metadata). Shared with the `Database` facade
    /// (top-level `hatp` crate) so both sides see the same definitions.
    pub catalog: Arc<Catalog>,
    /// Underlying DataFusion session.
    session: SessionContext,
}

impl fmt::Debug for FrontendSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FrontendSession")
            .field("catalog", &self.catalog)
            .field("session", &"<SessionContext>")
            .finish()
    }
}

impl FrontendSession {
    /// Builds a fresh frontend session.
    #[must_use]
    pub fn new() -> Self {
        Self {
            catalog: Arc::new(Catalog::new()),
            session: SessionContext::new(),
        }
    }

    /// Build a [`FrontendSession`] that shares `catalog` with the caller.
    #[must_use]
    pub fn with_catalog(catalog: Arc<Catalog>) -> Self {
        Self {
            catalog,
            session: SessionContext::new(),
        }
    }

    /// Returns the underlying DataFusion [`SessionContext`].
    #[must_use]
    pub fn session(&self) -> &SessionContext {
        &self.session
    }

    /// Attach a pre-built [`SessionContext`].
    #[must_use]
    pub fn with_session(mut self, ctx: SessionContext) -> Self {
        self.session = ctx;
        self
    }

    /// Reconstruct from existing collaborators.
    #[must_use]
    pub fn from_parts(catalog: Arc<Catalog>, ctx: SessionContext) -> Self {
        Self {
            catalog,
            session: ctx,
        }
    }

    /// Register `schema` as a DataFusion table backed by `engine`.
    pub fn register_table(
        &self,
        schema: crate::schema::TableSchema,
        engine: Arc<hatp_engine::Engine>,
    ) -> DfResult<Arc<TableProviderAdapter>> {
        let adapter = TableProviderAdapter::with_engine(
            Arc::clone(&self.catalog),
            schema.clone(),
            engine,
        );
        self.session
            .register_table(schema.name.clone(), adapter.clone())?;
        Ok(adapter)
    }

    /// Run `query` against this session and return the produced rows.
    pub async fn execute(&self, query: &str) -> DfResult<SessionExecutionOutcome> {
        let df = self.session.sql(query).await?;
        let batches: Vec<RecordBatch> = df.collect().await?;
        let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        let head = head_preview(&batches, 5);
        Ok(SessionExecutionOutcome {
            rows,
            batches: batches.len(),
            head,
        })
    }
}

impl Default for FrontendSession {
    fn default() -> Self {
        Self::new()
    }
}

/// First `cap` rows of the first batch as `column = value` strings.
fn head_preview(batches: &[RecordBatch], cap: usize) -> Vec<String> {
    let Some(batch) = batches.first() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let schema = batch.schema();
    for row in 0..batch.num_rows().min(cap) {
        let mut parts = Vec::with_capacity(batch.num_columns());
        for col in 0..batch.num_columns() {
            let array = batch.column(col);
            let field = schema.field(col);
            let value = if array.is_null(row) {
                "NULL".to_owned()
            } else {
                format!("{:?}", array.slice(row, 1))
            };
            parts.push(format!("{} = {}", field.name(), value));
        }
        out.push(format!("row[{row}]: {}", parts.join(", ")));
    }
    out
}

/// Adapter that turns a [`TableSchema`] into a DataFusion [`TableProvider`].
pub struct TableProviderAdapter {
    /// Owning catalog handle.
    catalog: Arc<Catalog>,
    /// Pre-resolved primary-key columns for the underlying table.
    primary_key: Vec<String>,
    /// Arrow schema exposed to DataFusion.
    schema: SchemaRef,
    /// Logical table name (relative to the schema).
    name: String,
    /// Engine handle. When `Some`, `scan` / DML route through the engine.
    engine: Option<Arc<hatp_engine::Engine>>,
    /// Cached aggregated statistics, guarded by the engine's `stats_version`.
    /// DataFusion calls `statistics()` on every query planning; without this
    /// cache the aggregation re-scans all SST zone-maps every time.
    statistics_cache: parking_lot::RwLock<Option<(u64, datafusion_common::Statistics)>>,
}

impl fmt::Debug for TableProviderAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TableProviderAdapter")
            .field("name", &self.name)
            .field("fields", &self.schema.fields().len())
            .field("engine_attached", &self.engine.is_some())
            .finish()
    }
}

impl TableProviderAdapter {
    /// Wraps `schema` and registers the adapter's parent table with the supplied catalog.
    #[must_use]
    pub fn new(catalog: Arc<Catalog>, schema: TableSchema) -> Arc<Self> {
        Arc::new(Self {
            catalog,
            primary_key: schema.primary_key.clone(),
            schema: schema.arrow.clone(),
            name: schema.name,
            engine: None,
            statistics_cache: parking_lot::RwLock::new(None),
        })
    }

    /// Like [`TableProviderAdapter::new`] but attaches an [`hatp_engine::Engine`].
    #[must_use]
    pub fn with_engine(
        catalog: Arc<Catalog>,
        schema: TableSchema,
        engine: Arc<hatp_engine::Engine>,
    ) -> Arc<Self> {
        Arc::new(Self {
            catalog,
            primary_key: schema.primary_key.clone(),
            schema: schema.arrow.clone(),
            name: schema.name,
            engine: Some(engine),
            statistics_cache: parking_lot::RwLock::new(None),
        })
    }

    /// Borrow the owning catalog.
    #[must_use]
    pub fn catalog(&self) -> &Arc<Catalog> {
        &self.catalog
    }

    /// Borrow the logical table name.
    #[must_use]
    pub fn table_name(&self) -> &str {
        &self.name
    }

    /// Whether this adapter is bound to an engine.
    #[must_use]
    pub fn is_engine_attached(&self) -> bool {
        self.engine.is_some()
    }

    /// Borrow the attached engine, if any.
    #[must_use]
    pub fn engine(&self) -> Option<&Arc<hatp_engine::Engine>> {
        self.engine.as_ref()
    }
}

// NOTE on filter push-down (a structural limit, not an omission):
// Only secondary-index-matched filters are pushed down. A *column-value*
// predicate (e.g. `WHERE age > 30`) cannot be pushed into the storage layer
// because each SST stores a whole row as an opaque value blob (the compact
// row codec keeps the schema in a per-table registry, not per row), so Vortex
// sees `key` / `begin_ts` / `end_ts` / `tombstone` columns but none of the
// business columns. Pushing a business-column predicate down would require a
// columnar (or column-group) value layout — a storage-format change that is
// out of scope here. Until then, `scan` reads the table and DataFusion filters
// the materialized batches; correctness does not depend on push-down.

#[async_trait]
impl TableProvider for TableProviderAdapter {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn constraints(&self) -> Option<&Constraints> {
        None
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn statistics(&self) -> Option<datafusion_common::Statistics> {
        // B.6: Aggregate ZoneMap-derived statistics for DataFusion CBO.
        // The result is cached and invalidated by the engine's `stats_version`.
        use datafusion_common::stats::Precision;
        use datafusion_common::{ColumnStatistics, Statistics};
        let engine = self.engine.as_ref()?;
        let version = engine.stats_version();
        // Check the cache: if the version matches, return the cached result.
        {
            let cache = self.statistics_cache.read();
            if let Some((cached_version, cached_stats)) = cache.as_ref() {
                if *cached_version == version {
                    return Some(cached_stats.clone());
                }
            }
        }
        // Cache miss — recompute and store.
        let rows = engine.table_row_count(&self.name);
        let bytes = engine.total_sst_bytes() as usize;
        let ncols = self.schema.fields().len().max(1);

        let zone_maps = engine.table_zone_maps(&self.name, Some(self.schema.as_ref()));
        let mut column_stats: Vec<ColumnStatistics> = Vec::new();
        for field in self.schema.fields().iter() {
            let col_name = field.name();
            let mut agg_nulls: Option<usize> = Some(0);
            for (_file_id, zm) in &zone_maps {
                if let Some(&n) = zm.column_null_count.get(col_name) {
                    if let Some(ref mut acc) = agg_nulls {
                        *acc += n;
                    }
                } else {
                    agg_nulls = None;
                }
            }
            let null_count = agg_nulls.unwrap_or(rows);

            let mut agg_min: Option<datafusion_common::ScalarValue> = None;
            let mut agg_max: Option<datafusion_common::ScalarValue> = None;
            for (_file_id, zm) in &zone_maps {
                if let Some(m) = zm.column_min.get(col_name) {
                    if matches!(m, datafusion_common::ScalarValue::Null) {
                        continue;
                    }
                    match &agg_min {
                        None => agg_min = Some(m.clone()),
                        Some(prev) => {
                            if m < prev {
                                agg_min = Some(m.clone());
                            }
                        }
                    }
                }
                if let Some(m) = zm.column_max.get(col_name) {
                    if matches!(m, datafusion_common::ScalarValue::Null) {
                        continue;
                    }
                    match &agg_max {
                        None => agg_max = Some(m.clone()),
                        Some(prev) => {
                            if m > prev {
                                agg_max = Some(m.clone());
                            }
                        }
                    }
                }
            }

            let byte_size = bytes.saturating_div(ncols);
            column_stats.push(ColumnStatistics {
                min_value: match agg_min {
                    Some(v) => Precision::Inexact(v),
                    None => Precision::Absent,
                },
                max_value: match agg_max {
                    Some(v) => Precision::Inexact(v),
                    None => Precision::Absent,
                },
                null_count: Precision::Inexact(null_count),
                distinct_count: Precision::Absent,
                sum_value: Precision::Absent,
                byte_size: Precision::Inexact(byte_size),
            });
        }

        let stats = Statistics {
            num_rows: Precision::Inexact(rows),
            total_byte_size: Precision::Inexact(bytes),
            column_statistics: column_stats,
        };
        *self.statistics_cache.write() = Some((version, stats.clone()));
        Some(stats)
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DfResult<Vec<TableProviderFilterPushDown>> {
        // Filters that can be translated to single-column comparison predicates are pushed down to the engine
        // layer for filtering at decode time (Inexact: the engine's ScalarValue comparison does not perform
        // DataFusion's implicit type conversions; DataFusion will re-validate on the result to ensure exact semantics).
        Ok(filters
            .iter()
            .map(|f| {
                if expr_to_column_predicates(f, &self.schema).is_empty() {
                    TableProviderFilterPushDown::Unsupported
                } else {
                    TableProviderFilterPushDown::Inexact
                }
            })
            .collect())
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        _limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let Some(engine) = &self.engine else {
            return datafusion_common::not_impl_err!(
                "TableProviderAdapter::scan requires `with_engine`"
            );
        };
        let projection_slice: Option<&[usize]> = projection.map(|v| v.as_slice());
        let schema = match projection {
            Some(projection) => Arc::new(self.schema.project(projection)?),
            None => self.schema.clone(),
        };

        // Extract pushdown-eligible single-column predicates.
        let mut predicates: Vec<ColumnPredicate> = Vec::new();
        for filter in filters {
            predicates.extend(expr_to_column_predicates(filter, &self.schema));
        }

        let pred_slice: Option<&[ColumnPredicate]> = if predicates.is_empty() {
            None
        } else {
            Some(&predicates)
        };

        // Vortex→Arrow direct scan: business columns read from Vortex files
        // via vortex-arrow, bypassing the opaque value blob entirely.  Memtable
        // data is handled via the row-codec path.  Each SST file is one
        // DataFusion partition.
        let partitions = engine
            .scan_table_arrow(
                &self.name,
                projection_slice,
                Some(&self.schema),
                usize::MAX,
                pred_slice,
            )
            .map_err(|err| datafusion_common::DataFusionError::External(Box::new(err)))?;

        let mut out = partitions;
        if out.is_empty() {
            out.push(vec![RecordBatch::new_empty(schema.clone())]);
        }

        let cfg = MemorySourceConfig::try_new(&out, schema, None)?;
        Ok(DataSourceExec::from_data_source(cfg))
    }

    async fn insert_into(
        &self,
        _state: &dyn Session,
        input: Arc<dyn ExecutionPlan>,
        _insert_op: InsertOp,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let Some(engine) = &self.engine else {
            return datafusion_common::not_impl_err!(
                "TableProviderAdapter::insert_into requires `with_engine`"
            );
        };
        let mutations = dml_planner::plan_insert(
            engine,
            input.as_ref(),
            &self.name,
            &self.primary_key,
        )?;
        if mutations.is_empty() {
            return Ok(Arc::new(EmptyExec::new(self.schema.clone())));
        }
        engine
            .write(&mutations)
            .map_err(|err| datafusion_common::DataFusionError::External(Box::new(err)))?;

        Ok(Arc::new(EmptyExec::new(self.schema.clone())))
    }

    async fn delete_from(
        &self,
        _state: &dyn Session,
        filters: Vec<Expr>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let Some(engine) = &self.engine else {
            return datafusion_common::not_impl_err!(
                "TableProviderAdapter::delete_from requires `with_engine`"
            );
        };
        let mutations = dml_planner::plan_delete(
            engine,
            &self.name,
            &self.primary_key,
            &filters,
        )?;
        if mutations.is_empty() {
            return Ok(Arc::new(EmptyExec::new(self.schema.clone())));
        }
        engine
            .write(&mutations)
            .map_err(|err| datafusion_common::DataFusionError::External(Box::new(err)))?;

        Ok(Arc::new(EmptyExec::new(self.schema.clone())))
    }

    async fn update(
        &self,
        _state: &dyn Session,
        assignments: Vec<(String, Expr)>,
        filters: Vec<Expr>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let Some(engine) = &self.engine else {
            return datafusion_common::not_impl_err!(
                "TableProviderAdapter::update requires `with_engine`"
            );
        };

        // Generate mutations via the frontend's update planner, which
        // delegates to the engine's UPDATE path (still DataFusion-aware
        // because the UPDATE rewrite uses `create_physical_expr` for
        // vectorized arithmetic / LIKE / NULL semantics).
        let mutations = dml_planner::plan_update(
            engine,
            &self.name,
            &assignments,
            &filters,
            &self.primary_key,
        )?;

        if mutations.is_empty() {
            return Ok(Arc::new(EmptyExec::new(self.schema.clone())));
        }

        // Commit the update.
        engine
            .write(&mutations)
            .map_err(|err| datafusion_common::DataFusionError::External(Box::new(err)))?;



        Ok(Arc::new(EmptyExec::new(self.schema.clone())))
    }

    async fn truncate(&self, _state: &dyn Session) -> DfResult<Arc<dyn ExecutionPlan>> {
        datafusion_common::not_impl_err!("TRUNCATE is not supported; use DELETE WHERE true")
    }
}

/// One DataFusion `MemorySchemaProvider` populated from the catalog.
fn build_schema_provider(
    catalog: Arc<Catalog>,
    engine: Arc<hatp_engine::Engine>,
    name: &str,
) -> DfResult<Arc<MemorySchemaProvider>> {
    let sp = MemorySchemaProvider::new();
    for table_name in catalog.table_names("hatp", name) {
        if let Some(schema) = catalog.table_schema(
            &crate::schema::QualifiedName::new("hatp", name),
            &table_name,
        ) {
            sp.register_table(
                table_name,
                TableProviderAdapter::with_engine(catalog.clone(), schema, Arc::clone(&engine)),
            )
            .map_err(|e| datafusion_common::DataFusionError::External(Box::new(e)))?;
        }
    }
    Ok(Arc::new(sp))
}

/// Register every catalog entry as a `MemoryCatalogProvider` on the
/// DataFusion session context. The catalog is exposed under the synthetic
/// catalog name `hatp` so it does not clash with the default `datafusion`
/// catalog. Every table is attached to the engine, so `scan` / DML route
/// through the storage layer immediately after open (no separate
/// `register_table` call required).
pub fn register_catalog_with(
    ctx: &SessionContext,
    catalog: Arc<Catalog>,
    engine: Arc<hatp_engine::Engine>,
) -> DfResult<()> {
    let cat_provider = MemoryCatalogProvider::new();
    for schema_name in catalog.schema_names("hatp") {
        let sp = build_schema_provider(catalog.clone(), Arc::clone(&engine), schema_name.as_str())?;
        cat_provider.register_schema(schema_name.as_str(), sp)?;
    }
    ctx.register_catalog("hatp", Arc::new(cat_provider));
    Ok(())
}

// Test utilities use `expect` / `unwrap` / `expect_err` liberally to keep
// assertion code readable. These are allow-listed here so the workspace's
// `unwrap_used = "deny"` lint does not break the test suite.
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;

    use crate::schema::CreateTable;
    use crate::schema::QualifiedName;

    fn arrow_two() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn populated_catalog() -> Arc<Catalog> {
        let cat = Catalog::new();
        cat.create_table(CreateTable {
            qualified: QualifiedName::new("hatp", "public"),
            name: "users".to_string(),
            arrow: arrow_two(),
            primary_key: vec!["id".to_string()],
        })
        .expect("create table");
        Arc::new(cat)
    }

    #[test]
    fn frontend_session_default_smoke() {
        let session = FrontendSession::new();
        assert!(session.catalog.catalog_names().is_empty());
    }

    #[test]
    fn table_provider_adapter_exposes_schema() {
        let cat = populated_catalog();
        let ts = cat
            .table_schema(&QualifiedName::new("hatp", "public"), "users")
            .expect("table");
        let adapter = TableProviderAdapter::new(cat, ts);
        assert_eq!(adapter.schema().fields().len(), 2);
        assert_eq!(adapter.table_type(), TableType::Base);
        assert_eq!(adapter.table_name(), "users");
    }

    #[test]
    fn table_provider_adapter_scan_returns_not_impl_without_engine() {
        let cat = populated_catalog();
        let ts = cat
            .table_schema(&QualifiedName::new("hatp", "public"), "users")
            .expect("table");
        let adapter = TableProviderAdapter::new(cat, ts);
        let ctx = SessionContext::new();
        let result = futures::executor::block_on(adapter.scan(&ctx.state(), None, &[], None));
        assert!(result.is_err(), "scan must surface a not-impl error");
    }

    #[test]
    fn register_catalog_with_attaches_schema_provider() {
        let cat = populated_catalog();
        let ctx = SessionContext::new();
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = hatp_engine::Engine::open(hatp_engine::EngineConfig::new(dir.path()))
            .expect("open engine");
        register_catalog_with(&ctx, cat.clone(), engine).expect("register");
        let found = ctx.catalog_names().into_iter().any(|n| n == "hatp");
        assert!(found);
    }
}
