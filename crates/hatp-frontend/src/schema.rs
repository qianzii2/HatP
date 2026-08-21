//! Schema and DDL surface.
//!
//! `TableSchema` is the canonical *logical* description of a table: an Arrow
//! [`SchemaRef`] carrying column names, types and nullability, plus a stable
//! catalog identifier and a notion of primary key columns. The frontend's
//! [`Catalog`](crate::catalog::Catalog) stores one `TableSchema` per logical
//! table.
//!

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use arrow_schema::FieldRef;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use serde::Deserialize;
use serde::Serialize;
use thiserror::Error;

/// Errors produced by the schema / DDL layer.
#[derive(Debug, Error)]
pub enum SchemaError {
    /// The schema is empty (zero fields). HatP requires at least one column.
    #[error("schema must contain at least one field")]
    EmptySchema,
    /// A column referenced in the DDL (e.g. primary key) does not exist in
    /// the underlying Arrow schema.
    #[error("column `{0}` is not present in the schema")]
    UnknownColumn(String),
    /// The schema contains duplicate column names — illegal in Arrow and in
    /// the HatP catalog.
    #[error("duplicate column name `{0}`")]
    DuplicateColumn(String),
    /// The supplied Arrow schema failed a structural validation.
    #[error("invalid schema: {0}")]
    Invalid(String),
}

/// Convenience alias used across the frontend.
pub type Result<T> = std::result::Result<T, SchemaError>;

/// Logical database / namespace identifier.
///
/// Mirrors the first tier of the DataFusion `CatalogProvider` tree.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CatalogId(pub String);

impl CatalogId {
    /// Borrow the underlying string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CatalogId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Logical schema (a.k.a. *namespace*) identifier.
///
/// Mirrors the second tier of the DataFusion `SchemaProvider` tree.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SchemaId(pub String);

impl SchemaId {
    /// Borrow the underlying string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SchemaId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Fully-qualified `(catalog, schema)` pair used to scope a table or index.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct QualifiedName {
    /// Catalog (database) namespace.
    pub catalog: CatalogId,
    /// Schema (namespace) within the catalog.
    pub schema: SchemaId,
}

impl QualifiedName {
    /// Build a `QualifiedName` from raw strings.
    #[must_use]
    pub fn new(catalog: &str, schema: &str) -> Self {
        Self {
            catalog: CatalogId(catalog.to_string()),
            schema: SchemaId(schema.to_string()),
        }
    }
}

impl fmt::Display for QualifiedName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.catalog, self.schema)
    }
}

/// Logical description of a table: the Arrow schema plus catalog-level
/// metadata (primary key, identifiers for serialization).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableSchema {
    /// Fully-qualified parent.
    pub qualified: QualifiedName,
    /// Table name (unique within the schema).
    pub name: String,
    /// Underlying Arrow schema.
    pub arrow: SchemaRef,
    /// Names of the columns that make up the primary key, in order. Empty
    /// when the table has no PK.
    pub primary_key: Vec<String>,
}

impl TableSchema {
    /// Build a new `TableSchema` from a logical identity and an Arrow schema.
    ///
    /// Validates that:
    /// - the schema has at least one field;
    /// - no two fields share the same name.
    pub fn new(
        qualified: QualifiedName,
        name: impl Into<String>,
        arrow: SchemaRef,
    ) -> Result<Self> {
        Self::validate_unique_columns(&arrow)?;
        if arrow.fields().is_empty() {
            return Err(SchemaError::EmptySchema);
        }
        Ok(Self {
            qualified,
            name: name.into(),
            arrow,
            primary_key: Vec::new(),
        })
    }

    /// Set the primary key columns. Returns an error if any column is not
    /// present in the underlying Arrow schema.
    pub fn with_primary_key<I, S>(mut self, columns: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let names: Vec<String> = columns
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();
        for name in &names {
            if self.arrow.column_with_name(name).is_none() {
                return Err(SchemaError::UnknownColumn(name.clone()));
            }
        }
        self.primary_key = names;
        Ok(self)
    }

    /// Look up the [`FieldRef`] for a column by name.
    #[must_use]
    pub fn field(&self, name: &str) -> Option<FieldRef> {
        self.arrow
            .column_with_name(name)
            .map(|(_, f)| Arc::new(f.clone()))
    }

    /// Returns the primary-key field references in declaration order, or an
    /// empty slice if no PK is defined.
    #[must_use]
    pub fn pk_fields(&self) -> Vec<FieldRef> {
        self.primary_key
            .iter()
            .filter_map(|n| self.field(n))
            .collect()
    }

    /// Borrow the underlying Arrow schema.
    #[must_use]
    pub fn arrow(&self) -> &Schema {
        self.arrow.as_ref()
    }

    fn validate_unique_columns(schema: &SchemaRef) -> Result<()> {
        let mut seen: BTreeMap<&str, ()> = BTreeMap::new();
        for field in schema.fields() {
            if seen.insert(field.name().as_str(), ()).is_some() {
                return Err(SchemaError::DuplicateColumn(field.name().clone()));
            }
        }
        Ok(())
    }
}

/// Describes a *create* DDL statement handed to the catalog.
///
/// Used by [`Catalog::create_table`](crate::catalog::Catalog::create_table)
/// to register a new logical table. The statement is purely declarative;
/// the catalog performs the actual creation and persists the resulting
/// [`TableSchema`].
#[derive(Debug, Clone, PartialEq)]
pub struct CreateTable {
    /// Schema qualified name.
    pub qualified: QualifiedName,
    /// Logical table name.
    pub name: String,
    /// Arrow schema describing the table.
    pub arrow: SchemaRef,
    /// Primary-key columns (empty = no PK).
    pub primary_key: Vec<String>,
}

impl CreateTable {
    /// Convert this DDL statement into a [`TableSchema`] suitable for
    /// registration. Performs structural validation but does *not* touch
    /// the catalog state.
    pub fn into_schema(self) -> Result<TableSchema> {
        let primary_key = self.primary_key;
        TableSchema::new(self.qualified, self.name, self.arrow)
            .and_then(|table| table.with_primary_key(primary_key))
    }
}

/// Description of a *drop* DDL statement.
#[derive(Debug, Clone, PartialEq)]
pub enum DropObject {
    /// Drop a logical table.
    Table {
        /// Qualified parent.
        qualified: QualifiedName,
        /// Name of the table to drop.
        name: String,
    },
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

    fn arrow_two_columns() -> SchemaRef {
        let fields: Vec<FieldRef> = vec![
            Arc::new(Field::new("id", DataType::Int64, false)),
            Arc::new(Field::new("name", DataType::Utf8, true)),
        ];
        let schema = Schema::new(fields);
        Arc::new(schema)
    }

    #[test]
    fn new_rejects_empty_schema() {
        let arrow = Arc::new(Schema::empty());
        let err = TableSchema::new(QualifiedName::new("cat", "schema"), "t", arrow)
            .expect_err("empty schema must error");
        assert!(matches!(err, SchemaError::EmptySchema));
    }

    #[test]
    fn new_rejects_duplicate_columns() {
        let arrow = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("id", DataType::Int64, false),
        ]));
        let err = TableSchema::new(QualifiedName::new("cat", "s"), "t", arrow)
            .expect_err("duplicate names must error");
        assert!(matches!(err, SchemaError::DuplicateColumn(ref n) if n == "id"));
    }

    #[test]
    fn with_primary_key_accepts_known_columns() {
        let ts = TableSchema::new(QualifiedName::new("c", "s"), "t", arrow_two_columns())
            .expect("valid")
            .with_primary_key(["id"])
            .expect("valid pk");
        assert_eq!(ts.primary_key, vec!["id".to_string()]);
        assert_eq!(ts.pk_fields().len(), 1);
    }

    #[test]
    fn with_primary_key_rejects_unknown_column() {
        let err = TableSchema::new(QualifiedName::new("c", "s"), "t", arrow_two_columns())
            .expect("valid")
            .with_primary_key(["nonexistent"])
            .expect_err("unknown col");
        assert!(matches!(err, SchemaError::UnknownColumn(ref n) if n == "nonexistent"));
    }

    #[test]
    fn create_table_into_schema_round_trip() {
        let ddl = CreateTable {
            qualified: QualifiedName::new("cat", "public"),
            name: "users".to_string(),
            arrow: arrow_two_columns(),
            primary_key: vec!["id".to_string()],
        };
        let ts = ddl.into_schema().expect("valid DDL");
        assert_eq!(ts.primary_key, vec!["id".to_string()]);
    }
}
