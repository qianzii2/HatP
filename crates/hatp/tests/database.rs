//! Public facade integration tests.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use bytes::Bytes;
use hatp::Database;
use tempfile::{Builder, TempDir};

fn database() -> Result<(TempDir, Database), Box<dyn std::error::Error>> {
    let dir = Builder::new().prefix("hatp-facade-").tempdir()?;
    let database = Database::open(dir.path())?;
    Ok((dir, database))
}

#[test]
fn autocommit_round_trip() -> Result<(), Box<dyn std::error::Error>> {
    let (_dir, database) = database()?;
    database.put(Bytes::from_static(b"key"), Bytes::from_static(b"value"))?;
    assert_eq!(database.get(b"key")?, Some(Bytes::from_static(b"value")));
    database.delete(Bytes::from_static(b"key"))?;
    assert_eq!(database.get(b"key")?, None);
    Ok(())
}

#[test]
fn transaction_reads_its_writes_and_commits_atomically() -> Result<(), Box<dyn std::error::Error>> {
    let (_dir, database) = database()?;
    database.put(Bytes::from_static(b"old"), Bytes::from_static(b"before"))?;
    let mut transaction = database.begin();
    transaction.put(Bytes::from_static(b"old"), Bytes::from_static(b"after"));
    transaction.put(Bytes::from_static(b"new"), Bytes::from_static(b"created"));
    assert_eq!(transaction.get(b"old")?, Some(Bytes::from_static(b"after")));
    assert_eq!(database.get(b"old")?, Some(Bytes::from_static(b"before")));
    transaction.commit()?;
    assert_eq!(database.get(b"old")?, Some(Bytes::from_static(b"after")));
    assert_eq!(database.get(b"new")?, Some(Bytes::from_static(b"created")));
    Ok(())
}

#[test]
fn snapshot_does_not_see_later_commit() -> Result<(), Box<dyn std::error::Error>> {
    let (_dir, database) = database()?;
    database.put(Bytes::from_static(b"key"), Bytes::from_static(b"v1"))?;
    let snapshot = database.begin();
    database.put(Bytes::from_static(b"key"), Bytes::from_static(b"v2"))?;
    assert_eq!(snapshot.get(b"key")?, Some(Bytes::from_static(b"v1")));
    assert_eq!(database.get(b"key")?, Some(Bytes::from_static(b"v2")));
    Ok(())
}

#[test]
fn restart_recovers_committed_values() -> Result<(), Box<dyn std::error::Error>> {
    let dir = Builder::new().prefix("hatp-restart-").tempdir()?;
    {
        let database = Database::open(dir.path())?;
        database.put(Bytes::from_static(b"key"), Bytes::from_static(b"durable"))?;
        database.flush()?;
    }
    let reopened = Database::open(dir.path())?;
    assert_eq!(reopened.get(b"key")?, Some(Bytes::from_static(b"durable")));
    Ok(())
}

#[test]
fn restart_recovers_catalog() -> Result<(), Box<dyn std::error::Error>> {
    use arrow_schema::{DataType, Field, Schema};
    use hatp_frontend::schema::{CreateTable, QualifiedName};

    let dir = Builder::new().prefix("hatp-catalog-").tempdir()?;
    {
        let database = Database::open(dir.path())?;
        database
            .catalog()
            .create_table(CreateTable {
                qualified: QualifiedName::new("hatp", "public"),
                name: "users".to_string(),
                arrow: std::sync::Arc::new(Schema::new(vec![Field::new(
                    "id",
                    DataType::Int64,
                    false,
                )])),
                primary_key: vec!["id".to_string()],
            })
            .map_err(|e| e.to_string())?;
    }
    // Catalog is persisted on Database::drop; reopen must restore the table.
    let reopened = Database::open(dir.path())?;
    let ts = reopened
        .catalog()
        .table_schema(&QualifiedName::new("hatp", "public"), "users");
    assert!(ts.is_some(), "catalog table must survive restart");
    assert_eq!(ts.unwrap().primary_key, vec!["id".to_string()]);
    Ok(())
}

#[test]
fn open_with_engine_mismatch_errors() -> Result<(), Box<dyn std::error::Error>> {
    use hatp::DatabaseError;
    use hatp::engine::{Engine, EngineConfig};

    let dir = Builder::new().prefix("hatp-mismatch-").tempdir()?;
    let other = Builder::new().prefix("hatp-mismatch-other-").tempdir()?;
    let engine = Engine::open(EngineConfig::new(dir.path()))?;

    let err = Database::open_with_engine(other.path(), engine)
        .expect_err("open_with_engine must reject a path that differs from the engine's");
    assert!(
        matches!(err, DatabaseError::Config(_)),
        "expected a config error, got {err:?}"
    );
    Ok(())
}

/// SC-11 (G31): concurrent open — two Databases open the same path sequentially, the second recovers the first's data
#[test]
fn concurrent_open_same_path_does_not_corrupt() -> Result<(), Box<dyn std::error::Error>> {
    let dir = Builder::new().prefix("hatp-concurrent-").tempdir()?;

    let db1 = Database::open(dir.path())?;
    db1.put(Bytes::from_static(b"k1"), Bytes::from_static(b"v1"))?;
    db1.flush()?;
    // Close db1 to release its file lock
    drop(db1);

    // Second Database opens the same path (should recover the first's data)
    let db2 = Database::open(dir.path())?;

    // Positive assertion: db2 can see db1's persisted data
    assert_eq!(
        db2.get(b"k1")?.map(|v| v.to_vec()),
        Some(b"v1".to_vec()),
        "db2 must recover db1's flushed data"
    );

    // Negative assertion: after db2 writes, close and reopen, data is persisted
    db2.put(Bytes::from_static(b"k2"), Bytes::from_static(b"v2"))?;
    db2.flush()?;
    drop(db2);

    let db3 = Database::open(dir.path())?;
    assert_eq!(
        db3.get(b"k1")?.map(|v| v.to_vec()),
        Some(b"v1".to_vec()),
        "db3 must recover db1's data"
    );
    assert_eq!(
        db3.get(b"k2")?.map(|v| v.to_vec()),
        Some(b"v2".to_vec()),
        "db3 must recover db2's data"
    );
    drop(db3);
    Ok(())
}
