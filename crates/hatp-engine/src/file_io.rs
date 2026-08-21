//! I/O abstraction layer — modelled after the TigerBeetle Storage trait.
//!
//! Design principles:
//! - Medium granularity: does not abstract filesystem management (create/delete/rename),
//!   only abstracts data read/write.
//! - WAL append and sync_data remain independent (SyncMode::Group calls from different
//!   threads).
//! - Vortex async ReadAt is not included in this abstraction (it uses a separate fuzz
//!   target).
//! - All methods use &self (WAL's Mutex<File> already provides interior mutability).

use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Engine data I/O abstraction. Does not abstract filesystem management, only
/// abstracts read/write/sync operations on an already-open file.
pub trait FileIO: Send + Sync {
    /// Append write (used by WAL)
    fn write_all(&self, buf: &[u8]) -> io::Result<()>;
    /// fsync data (used by WAL + Manifest)
    fn sync_data(&self) -> io::Result<()>;
    /// Truncate file (WAL recovery + Manifest snapshot)
    fn set_len(&self, len: u64) -> io::Result<()>;
    /// Read entire file (WAL recovery + Manifest recovery + sidecar loading)
    fn read_all(&self) -> io::Result<Vec<u8>>;
    /// Atomic write (tmp+rename+fsync) — used by schema/catalog/SST/Bloom/KeyIndex
    fn write_atomic(&self, path: &Path, data: &[u8]) -> io::Result<()>;
}

// ── Real filesystem implementation ────────────────────────────────────────────

use std::fs::{self, File, OpenOptions};

/// I/O implementation holding an open file handle. Used for WAL and Manifest
/// append-write scenarios.
#[derive(Debug)]
pub struct RealFileIO {
    file: parking_lot::Mutex<File>,
}

impl RealFileIO {
    pub fn open_append(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).read(true).append(true).open(path)?;
        Ok(Self { file: parking_lot::Mutex::new(file) })
    }
}

impl FileIO for RealFileIO {
    fn write_all(&self, buf: &[u8]) -> io::Result<()> {
        self.file.lock().write_all(buf)
    }
    fn sync_data(&self) -> io::Result<()> {
        self.file.lock().sync_data()
    }
    fn set_len(&self, len: u64) -> io::Result<()> {
        let mut file = self.file.lock();
        file.set_len(len)?;
        file.seek(SeekFrom::Start(0))?;
        Ok(())
    }
    fn read_all(&self) -> io::Result<Vec<u8>> {
        let mut file = self.file.lock();
        file.seek(SeekFrom::Start(0))?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        Ok(buf)
    }
    fn write_atomic(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, data)?;
        fs::rename(&tmp, path)?;
        // fsync parent dir
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                File::open(parent)?.sync_all()?;
            }
        }
        Ok(())
    }
}

// ── In-memory filesystem (for testing + Miri + simulator) ─────────────────────

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// In-memory filesystem. All files are stored in a HashMap; no real disk access.
#[allow(missing_debug_implementations)]
pub struct FakeFileIO {
    files: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    current: Mutex<String>,
    cursor: Mutex<usize>,
}

impl FakeFileIO {
    pub fn new(name: &str) -> Self {
        Self {
            files: Arc::new(Mutex::new(HashMap::new())),
            current: Mutex::new(name.to_string()),
            cursor: Mutex::new(0),
        }
    }
    /// Get the current file content (for test assertions)
    pub fn get(&self, name: &str) -> Option<Vec<u8>> {
        self.files.lock().unwrap().get(name).cloned()
    }
    /// Write corrupted data (for fault injection)
    pub fn corrupt(&self, name: &str, data: Vec<u8>) {
        self.files.lock().unwrap().insert(name.to_string(), data);
    }
}

impl FileIO for FakeFileIO {
    fn write_all(&self, buf: &[u8]) -> io::Result<()> {
        let name = self.current.lock().unwrap().clone();
        let mut files = self.files.lock().unwrap();
        let entry = files.entry(name).or_default();
        entry.extend_from_slice(buf);
        Ok(())
    }
    fn sync_data(&self) -> io::Result<()> {
        Ok(()) // In-memory filesystem: no fsync needed
    }
    fn set_len(&self, len: u64) -> io::Result<()> {
        let name = self.current.lock().unwrap().clone();
        let mut files = self.files.lock().unwrap();
        if let Some(entry) = files.get_mut(&name) {
            entry.truncate(len as usize);
        }
        *self.cursor.lock().unwrap() = 0;
        Ok(())
    }
    fn read_all(&self) -> io::Result<Vec<u8>> {
        let name = self.current.lock().unwrap().clone();
        let files = self.files.lock().unwrap();
        Ok(files.get(&name).cloned().unwrap_or_default())
    }
    fn write_atomic(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        let name = path.to_string_lossy().to_string();
        self.files.lock().unwrap().insert(name, data.to_vec());
        Ok(())
    }
}