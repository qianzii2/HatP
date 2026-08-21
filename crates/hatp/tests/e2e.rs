#![allow(clippy::expect_used, clippy::unwrap_used)]

//! End-to-end integration tests for the HatP facade.

use bytes::Bytes;
use hatp::Database;
use hatp::DatabaseError;
use tempfile::{Builder, TempDir};

fn tempdir(prefix: &str) -> TempDir {
    Builder::new().prefix(prefix).tempdir().expect("tempdir")
}

#[test]
fn open_write_flush_read_round_trip() -> Result<(), DatabaseError> {
    let dir = tempdir("hatp-e2e-roundtrip-");
    let database = Database::open(dir.path())?;
    for i in 0..100_u64 {
        let key = Bytes::from(format!("k{i}"));
        let value = Bytes::from(format!("v{i}"));
        database.put(key, value)?;
    }
    let _flushed_rows = database.flush()?;
    for i in 0..100_u64 {
        let key = format!("k{i}");
        let value = format!("v{i}");
        assert_eq!(
            database.get(key.as_bytes())?,
            Some(Bytes::from(value)),
            "key={key}"
        );
    }
    Ok(())
}

#[test]
fn restart_recovers_after_flush() -> Result<(), DatabaseError> {
    let dir = tempdir("hatp-e2e-restart-");
    {
        let database = Database::open(dir.path())?;
        for i in 0..50_u64 {
            database.put(Bytes::from(format!("k{i}")), Bytes::from(format!("v{i}")))?;
        }
        let _ = database.flush()?;
    }
    let reopened = Database::open(dir.path())?;
    for i in 0..50_u64 {
        let key = format!("k{i}");
        let value = format!("v{i}");
        assert_eq!(
            reopened.get(key.as_bytes())?,
            Some(Bytes::from(value)),
            "key={key}"
        );
    }
    Ok(())
}

#[test]
fn transaction_commit_round_trip() -> Result<(), DatabaseError> {
    let dir = tempdir("hatp-e2e-tx-");
    let database = Database::open(dir.path())?;
    let mut tx = database.begin();
    tx.put(Bytes::from_static(b"a"), Bytes::from_static(b"1"));
    tx.put(Bytes::from_static(b"b"), Bytes::from_static(b"2"));
    tx.commit()?;
    assert_eq!(database.get(b"a")?, Some(Bytes::from_static(b"1")));
    assert_eq!(database.get(b"b")?, Some(Bytes::from_static(b"2")));
    Ok(())
}


