//! Catalog persistence.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use parking_lot::RwLock;
use serde::Deserialize;
use serde::Serialize;
use thiserror::Error;

use crate::schema::CatalogId;
use crate::schema::CreateTable;
use crate::schema::DropObject;
use crate::schema::QualifiedName;
use crate::schema::SchemaError;
use crate::schema::SchemaId;
use crate::schema::TableSchema;

/// Errors produced by the catalog layer.
#[derive(Debug, Error)]
pub enum CatalogError {
    /// A table of the same name already exists in the schema.
    #[error("table `{0}` already exists in `{1}`")]
    TableExists(String, SchemaId),
    /// The named table is not present in the schema.
    #[error("table `{0}` is not present in `{1}`")]
    UnknownTable(String, SchemaId),
    /// A catalog (database) of the same name is already registered.
    #[error("catalog `{0}` already exists")]
    CatalogAlreadyExists(CatalogId),
    /// A catalog cannot be dropped because it still contains schemas.
    #[error("catalog `{0}` is not empty — still contains schemas")]
    CatalogNotEmpty(CatalogId),
    /// A schema of the same name already exists in the catalog.
    #[error("schema `{0}` already exists in catalog `{1}`")]
    SchemaExists(SchemaId, CatalogId),
    /// The named schema is not present in the catalog.
    #[error("schema `{0}` is not present in catalog `{1}`")]
    UnknownSchema(SchemaId, CatalogId),
    /// The named catalog is not registered.
    #[error("catalog `{0}` is not registered")]
    UnknownCatalog(CatalogId),
    /// A schema-level DDL was rejected by the schema validator.
    #[error("schema validation failed: {0}")]
    Schema(#[from] SchemaError),
    /// A schema cannot be dropped because it still contains tables.
    #[error("schema `{0}` is not empty — still contains tables in catalog `{1}`")]
    SchemaNotEmpty(SchemaId, CatalogId),
    /// Serialization / deserialization failure.
    #[error("catalog serialization failed: {0}")]
    Serde(#[from] serde_json::Error),
}

/// Convenience alias used by callers.
pub type Result<T> = std::result::Result<T, CatalogError>;

/// Storage entry for a single table within a [`Catalog`].
#[derive(Debug, Clone)]
pub struct TableEntry {
    /// Logical schema.
    pub schema: TableSchema,
}

impl TableEntry {
    /// Create a new entry from a [`TableSchema`].
    #[must_use]
    pub fn new(schema: TableSchema) -> Self {
        Self { schema }
    }
}

/// Storage entry for a single schema (namespace) inside a catalog.
#[derive(Debug, Default, Clone)]
pub struct SchemaEntry {
    /// Tables within this schema, keyed by table name.
    pub tables: BTreeMap<String, TableEntry>,
}

impl SchemaEntry {
    /// Create an empty schema entry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// One catalog (database) inside the top-level [`Catalog`].
#[derive(Debug, Default, Clone)]
pub struct CatalogEntry {
    /// Schemas (namespaces) within this catalog, keyed by schema name.
    pub schemas: BTreeMap<SchemaId, SchemaEntry>,
}

impl CatalogEntry {
    /// Create an empty catalog entry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            schemas: BTreeMap::new(),
        }
    }
}

/// In-memory catalog of all logical metadata.
///
/// Cloning is cheap — every clone shares the same `Arc<RwLock<…>>` core.
/// Reads are lock-free under high concurrency because [`parking_lot::RwLock`]
/// uses an unfair reader-writer strategy that prefers readers when there
/// is no writer waiting.
#[derive(Debug, Clone, Default)]
pub struct Catalog {
    /// All catalog state lives behind a single RW lock for simplicity.
    inner: Arc<RwLock<CatalogState>>,
}

/// Owned (non-`Arc`) catalog state.
#[derive(Debug, Default)]
struct CatalogState {
    /// Top-level catalog (database) namespaces.
    catalogs: BTreeMap<CatalogId, CatalogEntry>,
}

impl Catalog {
    /// Creates an empty in-memory catalog.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reconstruct a catalog from a [`CatalogSnapshot`] previously produced
    /// by [`Catalog::to_snapshot`].
    pub fn from_snapshot(snapshot: CatalogSnapshot) -> Result<Self> {
        let cat = Self::new();
        cat.apply_snapshot(snapshot)?;
        Ok(cat)
    }

    /// Materialise the catalog state as a serializable snapshot.
    #[must_use]
    pub fn to_snapshot(&self) -> CatalogSnapshot {
        let state = self.inner.read();
        let catalogs = state
            .catalogs
            .iter()
            .map(|(cat_id, entry)| CatalogRecord {
                name: cat_id.clone(),
                schemas: entry
                    .schemas
                    .iter()
                    .map(|(schema_id, schema_entry)| SchemaRecord {
                        name: schema_id.clone(),
                        tables: schema_entry
                            .tables
                            .iter()
                            .map(|(table_name, table_entry)| TableRecord {
                                name: table_name.clone(),
                                arrow_schema: (*table_entry.schema.arrow).clone(),
                                primary_key: table_entry.schema.primary_key.clone(),
                            })
                            .collect(),
                    })
                    .collect(),
            })
            .collect();
        CatalogSnapshot { catalogs }
    }

    /// Apply a snapshot in-place, replacing the existing state.
    ///
    /// Used by both [`Self::from_snapshot`] and the boot sequence when an
    /// already-open catalog decides to refresh its view from disk.
    pub fn apply_snapshot(&self, snapshot: CatalogSnapshot) -> Result<()> {
        let mut state = CatalogState::default();
        for cat_record in snapshot.catalogs {
            let mut cat_entry = CatalogEntry::new();
            for schema_record in cat_record.schemas {
                let mut schema_entry = SchemaEntry::new();
                for table_record in schema_record.tables {
                    let qn =
                        QualifiedName::new(cat_record.name.as_str(), schema_record.name.as_str());
                    let mut ts = TableSchema::new(
                        qn,
                        table_record.name.clone(),
                        Arc::new(table_record.arrow_schema),
                    )?;
                    if !table_record.primary_key.is_empty() {
                        ts = ts.with_primary_key(table_record.primary_key.clone())?;
                    }
                    let table_entry = TableEntry::new(ts);
                    schema_entry.tables.insert(table_record.name, table_entry);
                }
                cat_entry.schemas.insert(schema_record.name, schema_entry);
            }
            state.catalogs.insert(cat_record.name, cat_entry);
        }
        *self.inner.write() = state;
        Ok(())
    }

    /// Create a new top-level catalog (database).
    pub fn create_catalog(&self, name: impl Into<String>) -> Result<()> {
        let id = CatalogId(name.into());
        let mut state = self.inner.write();
        if state.catalogs.contains_key(&id) {
            return Err(CatalogError::CatalogAlreadyExists(id));
        }
        state.catalogs.insert(id, CatalogEntry::new());
        Ok(())
    }

    /// Drop a top-level catalog. The catalog must contain no schemas.
    pub fn drop_catalog(&self, name: &str) -> Result<()> {
        let id = CatalogId(name.to_string());
        let mut state = self.inner.write();
        let entry = state
            .catalogs
            .remove(&id)
            .ok_or_else(|| CatalogError::UnknownCatalog(id.clone()))?;
        if entry.schemas.is_empty() {
            Ok(())
        } else {
            state.catalogs.insert(id.clone(), entry);
            Err(CatalogError::CatalogNotEmpty(id))
        }
    }

    /// Create a new schema (namespace) inside a catalog.
    pub fn create_schema(&self, catalog: &str, schema: &str) -> Result<()> {
        let cat_id = CatalogId(catalog.to_string());
        let sch_id = SchemaId(schema.to_string());
        let mut state = self.inner.write();
        let entry = state.catalogs.entry(cat_id.clone()).or_default();
        if entry.schemas.contains_key(&sch_id) {
            return Err(CatalogError::SchemaExists(sch_id, cat_id));
        }
        entry.schemas.insert(sch_id, SchemaEntry::new());
        Ok(())
    }

    /// Drop an empty schema. Returns an error if the schema still owns tables.
    pub fn drop_schema(&self, catalog: &str, schema: &str) -> Result<()> {
        let cat_id = CatalogId(catalog.to_string());
        let sch_id = SchemaId(schema.to_string());
        let mut state = self.inner.write();
        let entry = state
            .catalogs
            .get_mut(&cat_id)
            .ok_or_else(|| CatalogError::UnknownCatalog(cat_id.clone()))?;
        let removed = entry
            .schemas
            .remove(&sch_id)
            .ok_or_else(|| CatalogError::UnknownSchema(sch_id.clone(), cat_id.clone()))?;
        if removed.tables.is_empty() {
            Ok(())
        } else {
            entry.schemas.insert(sch_id.clone(), removed);
            Err(CatalogError::SchemaNotEmpty(sch_id.clone(), cat_id))
        }
    }

    /// Apply a [`CreateTable`] DDL statement. Validates the schema and creates
    /// the table entry.
    pub fn create_table(&self, ddl: CreateTable) -> Result<()> {
        let schema = ddl.into_schema()?;
        let table_name = schema.name.clone();
        let qualified = schema.qualified.clone();
        let mut state = self.inner.write();
        let cat_entry = state.catalogs.entry(qualified.catalog.clone()).or_default();
        let schema_entry = cat_entry
            .schemas
            .entry(qualified.schema.clone())
            .or_default();
        if schema_entry.tables.contains_key(&table_name) {
            return Err(CatalogError::TableExists(
                table_name,
                qualified.schema.clone(),
            ));
        }
        let table_entry = TableEntry::new(schema);
        schema_entry.tables.insert(table_name, table_entry);
        Ok(())
    }

    /// Drop a logical table.
    pub fn drop(&self, what: DropObject) -> Result<()> {
        match what {
            DropObject::Table { qualified, name } => self.drop_table(&qualified, &name),
        }
    }

    fn drop_table(&self, qualified: &QualifiedName, name: &str) -> Result<()> {
        // NOTE: Layer Separation — Catalog vs Engine
        //
        // This method removes ONLY the catalog metadata (logical table definition).
        // It does NOT remove engine data (memtable + SST files). Rationale:
        //
        // 1. The catalog is a logical layer; the engine is a physical layer.
        //    Separating them allows the catalog to be swapped without touching engine data.
        // 2. Engine data cleanup requires coordination with the engine's MVCC system:
        //    - Must write tombstones for all keys (not just delete them).
        //    - Must wait for in-flight reads to complete.
        //    - Must update dropped_tables to filter orphaned SST data on scans.
        //
        // Callers that need full table removal should:
        // 1. Call `Catalog::drop_table()` to remove catalog metadata.
        // 2. Call `Engine::drop_table_prefix()` to filter engine data.
        // 3. Wait for compaction to garbage-collect orphaned SST data.
        //
        // This separation is intentional — full cleanup is the caller's responsibility.
        let mut state = self.inner.write();
        let cat_entry = state
            .catalogs
            .get_mut(&qualified.catalog)
            .ok_or_else(|| CatalogError::UnknownCatalog(qualified.catalog.clone()))?;
        let schema_entry = cat_entry
            .schemas
            .get_mut(&qualified.schema)
            .ok_or_else(|| {
                CatalogError::UnknownSchema(qualified.schema.clone(), qualified.catalog.clone())
            })?;
        let _ = schema_entry.tables.remove(name).ok_or_else(|| {
            CatalogError::UnknownTable(name.to_string(), qualified.schema.clone())
        })?;
        Ok(())
    }

    /// Look up a [`TableSchema`] by fully-qualified name.
    #[must_use]
    pub fn table_schema(&self, qualified: &QualifiedName, name: &str) -> Option<TableSchema> {
        let state = self.inner.read();
        state
            .catalogs
            .get(&qualified.catalog)?
            .schemas
            .get(&qualified.schema)?
            .tables
            .get(name)
            .map(|e| e.schema.clone())
    }

    /// Look up a table entry by fully-qualified name.
    #[must_use]
    pub fn table_entry(&self, qualified: &QualifiedName, name: &str) -> Option<TableEntry> {
        let state = self.inner.read();
        state
            .catalogs
            .get(&qualified.catalog)?
            .schemas
            .get(&qualified.schema)?
            .tables
            .get(name)
            .cloned()
    }

    /// Returns an ordered list of all catalogs known to this instance.
    #[must_use]
    pub fn catalog_names(&self) -> Vec<CatalogId> {
        self.inner.read().catalogs.keys().cloned().collect()
    }

    /// Returns an ordered list of all schemas in `catalog`.
    #[must_use]
    pub fn schema_names(&self, catalog: &str) -> Vec<SchemaId> {
        let state = self.inner.read();
        state
            .catalogs
            .get(&CatalogId(catalog.to_string()))
            .map(|c| c.schemas.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Returns an ordered list of all tables in `(catalog, schema)`.
    #[must_use]
    pub fn table_names(&self, catalog: &str, schema: &str) -> Vec<String> {
        let state = self.inner.read();
        state
            .catalogs
            .get(&CatalogId(catalog.to_string()))
            .and_then(|c| c.schemas.get(&SchemaId(schema.to_string())))
            .map(|s| s.tables.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Returns an ordered list of every table known to this catalog.
    #[must_use]
    pub fn all_tables(&self) -> Vec<(QualifiedName, String, TableSchema)> {
        let state = self.inner.read();
        let mut out = Vec::new();
        for (cat, entry) in &state.catalogs {
            for (schema, schema_entry) in &entry.schemas {
                for (table, table_entry) in &schema_entry.tables {
                    out.push((
                        QualifiedName::new(cat.as_str(), schema.as_str()),
                        table.clone(),
                        table_entry.schema.clone(),
                    ));
                }
            }
        }
        out
    }
}

impl fmt::Display for Catalog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.inner.read();
        write!(f, "Catalog(")?;
        let cats: Vec<String> = state
            .catalogs
            .iter()
            .map(|(c, e)| {
                let n_schemas = e.schemas.len();
                format!("{c}: {n_schemas} schemas")
            })
            .collect();
        write!(f, "{} catalogs: [{}]", cats.len(), cats.join(", "))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Snapshot / serialization types
// ─────────────────────────────────────────────────────────────────────────────

/// One catalog record inside a [`CatalogSnapshot`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogRecord {
    /// Catalog (database) name.
    pub name: CatalogId,
    /// Schemas inside the catalog, in declaration order.
    pub schemas: Vec<SchemaRecord>,
}

/// One schema record inside a [`CatalogRecord`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchemaRecord {
    /// Schema name.
    pub name: SchemaId,
    /// Tables inside the schema, in declaration order.
    pub tables: Vec<TableRecord>,
}

/// One table record inside a [`SchemaRecord`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableRecord {
    /// Table name.
    pub name: String,
    /// Arrow schema (serialized via `arrow_schema`'s built-in `Serialize`).
    pub arrow_schema: arrow_schema::Schema,
    /// Primary-key columns.
    pub primary_key: Vec<String>,
}

/// On-disk shape of a serialized catalog.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogSnapshot {
    /// All catalogs, in declaration order.
    pub catalogs: Vec<CatalogRecord>,
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

    fn arrow() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn ddl(catalog: &str, schema: &str, table: &str) -> CreateTable {
        CreateTable {
            qualified: QualifiedName::new(catalog, schema),
            name: table.to_string(),
            arrow: arrow(),
            primary_key: vec!["id".to_string()],
        }
    }

    #[test]
    fn catalog_create_and_lookup() {
        let cat = Catalog::new();
        cat.create_catalog("c").expect("create catalog");
        cat.create_schema("c", "public").expect("create schema");
        cat.create_table(ddl("c", "public", "users"))
            .expect("create table");
        let ts = cat
            .table_schema(&QualifiedName::new("c", "public"), "users")
            .expect("table present");
        assert_eq!(ts.name, "users");
    }

    #[test]
    fn duplicate_table_is_rejected() {
        let cat = Catalog::new();
        cat.create_table(ddl("c", "s", "t")).expect("first");
        let err = cat.create_table(ddl("c", "s", "t")).expect_err("dup");
        assert!(matches!(err, CatalogError::TableExists(..)));
    }

    #[test]
    fn snapshot_round_trip_preserves_state() {
        let cat = Catalog::new();
        cat.create_table(ddl("cat", "public", "users"))
            .expect("create");
        let snapshot = cat.to_snapshot();
        let cat2 = Catalog::from_snapshot(snapshot).expect("from snapshot");
        let snap1 = cat.to_snapshot();
        let snap2 = cat2.to_snapshot();
        assert_eq!(snap1, snap2);
        assert!(
            cat2.table_entry(&QualifiedName::new("cat", "public"), "users")
                .is_some()
        );
    }

    #[test]
    fn drop_catalog_with_schemas_fails() {
        let cat = Catalog::new();
        cat.create_catalog("c").expect("create");
        cat.create_schema("c", "public").expect("create schema");
        let err = cat.drop_catalog("c").expect_err("must not drop");
        assert!(matches!(err, CatalogError::CatalogNotEmpty(..)));
    }

    #[test]
    fn drop_schema_with_tables_fails() {
        let cat = Catalog::new();
        cat.create_catalog("c").expect("create");
        cat.create_schema("c", "public").expect("create schema");
        cat.create_table(ddl("c", "public", "users"))
            .expect("create");
        let err = cat.drop_schema("c", "public").expect_err("must not drop");
        assert!(matches!(err, CatalogError::SchemaNotEmpty(..)));
    }

    #[test]
    fn all_tables_returns_full_inventory() {
        let cat = Catalog::new();
        cat.create_table(ddl("c", "s", "t1")).expect("t1");
        cat.create_table(ddl("c", "s", "t2")).expect("t2");
        let all = cat.all_tables();
        assert_eq!(all.len(), 2);
        let names: Vec<String> = all.iter().map(|(_, n, _)| n.clone()).collect();
        assert!(names.contains(&"t1".to_string()));
        assert!(names.contains(&"t2".to_string()));
    }
}
