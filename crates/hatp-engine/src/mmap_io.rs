//! Memory-mapped I/O helpers — zero-copy file reads via mmap (PR 6.0).
//!
//! Used by bloom filter and key-index sidecar loading to avoid the `read()`
//! system call. The kernel's page cache is shared across processes, and on
//! subsequent accesses the data may already be resident.
//!
//! Vortex SST files are NOT mmap'd here — Vortex 0.83's `VortexReadAt` is an
//! async trait (`fn read_at(…) -> Pin<Box<dyn Future<Output = Result<BufferHandle>>>>`)
//! that requires `BufferHandle` returns, incompatible with a simple sync mmap
//! slice.  Once Vortex exposes a sync or completion-based read API, SST files
//! should be the first target for mmap — they are the largest I/O source in
//! OLAP scans and would benefit most from zero-copy page-cache access.

use memmap2::Mmap;
use std::fs::File;
use std::io;
use std::path::Path;

/// Opens a file read-only and memory-maps it. Returns the mapped region
/// or falls back to [`std::fs::read`] on failure (e.g. empty file).
pub(crate) fn mmap_read(path: &Path) -> io::Result<Mmap> {
    let file = File::open(path)?;
    let len = file.metadata()?.len();
    if len == 0 {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "empty file"));
    }
    // SAFETY: the file is opened read-only. The backing file is managed by
    // the engine's compaction guard — sidecar files are never truncated or
    // overwritten while a reader holds a mapping.
    unsafe { Mmap::map(&file) }
}

/// Reads the entire contents of `path` into a `Vec<u8>` via mmap, falling
/// back to `std::fs::read` if mmap fails (e.g. empty file, permission error).
pub(crate) fn mmap_read_to_vec(path: &Path) -> io::Result<Vec<u8>> {
    match mmap_read(path) {
        Ok(mmap) => Ok(mmap.to_vec()),
        Err(_) => std::fs::read(path),
    }
}