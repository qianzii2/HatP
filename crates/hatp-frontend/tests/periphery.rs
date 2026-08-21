//! Integration tests for `hatp-frontend`.
//!
//! These tests exercise the catalog API as a downstream user would. The
//! `hatp_frontend::dml` module was deleted because it
//! was an empty wrapper around `hatp_engine::Mutation` with no
//! production callers.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use hatp_frontend::catalog::Catalog;
use hatp_frontend::schema::CreateTable;
use hatp_frontend::schema::QualifiedName;

fn two_column_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]))
}

#[test]
fn catalog_round_trips_table_definitions() {
    let catalog = Catalog::new();
    catalog
        .create_table(CreateTable {
            qualified: QualifiedName::new("hatp", "public"),
            name: "users".to_string(),
            arrow: two_column_schema(),
            primary_key: vec!["id".to_string()],
        })
        .expect("create_table");
    let snap1 = catalog.to_snapshot();
    let restored = Catalog::from_snapshot(snap1.clone()).expect("from_snapshot");
    assert_eq!(restored.to_snapshot(), snap1);
    let tables = catalog.table_names("hatp", "public");
    assert_eq!(tables, vec!["users".to_string()]);
}

#[test]
fn catalog_rejects_duplicate_table() {
    let catalog = Catalog::new();
    catalog
        .create_table(CreateTable {
            qualified: QualifiedName::new("hatp", "public"),
            name: "users".to_string(),
            arrow: two_column_schema(),
            primary_key: vec!["id".to_string()],
        })
        .expect("first");
    let err = catalog
        .create_table(CreateTable {
            qualified: QualifiedName::new("hatp", "public"),
            name: "users".to_string(),
            arrow: two_column_schema(),
            primary_key: vec!["id".to_string()],
        })
        .expect_err("duplicate must fail");
    assert!(format!("{err}").contains("users"));
}

#[test]
fn frontend_session_can_be_built_with_shared_catalog() {
    let catalog = Arc::new(Catalog::new());
    let session = hatp_frontend::execution::FrontendSession::with_catalog(catalog.clone());
    assert!(session.catalog.table_names("hatp", "public").is_empty());
}
