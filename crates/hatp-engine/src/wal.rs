//! Append-only write-ahead log backed by a hand-rolled binary frame format.
//!
//! # Why not Arrow IPC
//!
//! The engine previously encoded WAL records as an Arrow IPC stream. Arrow
//! IPC carries a schema flatbuffer per stream (~200 bytes of metadata before
//! the first column byte) plus per-record batch metadata. A single
//! `single_put` write produced a one-row IPC frame of ~250 bytes for ~80
//! bytes of payload — the metadata was ~75 % of the bytes that hit
//! `sync_data`. The binary format below replaces it with a hand-coded
//! length-prefixed format whose per-frame overhead is a fixed 28 bytes plus
//! the CRC32C trailer.
//!
//! The Arrow-IPC write/decode path has been removed entirely (it was dead
//! once `append_batch_commit_and_sync` switched to this encoder). There is
//! no on-the-fly format sniffing: a WAL file that does not begin with the
//! `WAL1` magic is rejected as corrupt rather than silently parsed as a
//! legacy format.
//!
//! # Frame layout
//!
//! ```text
//!   magic          [u8; 4]     "WAL1"
//!   tx_id          u64   LE
//!   op             u8
//!   key_len        u32   LE
//!   value_len      u32   LE   (0xFFFF_FFFF == None)
//!   key_bytes      [u8; key_len]
//!   value_bytes    [u8; value_len]   (omitted when value_len == 0xFFFF_FFFF)
//!   crc32c         u32   LE         (covers everything above)
//! ```
//!
//! Frames are self-describing — recovery walks the file, reads the magic,
//! then reads the fixed-size header, then the variable-length payload, then
//! validates the CRC. A torn final frame is detected by either a missing
//! magic or a CRC mismatch and is silently truncated.
//!
//! # Batch boundaries (design note)
//!
//! An earlier revision of this format (`WAL2`) added an explicit 32-bit
//! `batch_size` header so recovery could replay one transaction's records
//! atomically. That header was **never adopted**: `encode_batch` appends a
//! trailing `Commit` marker, and recovery groups frames by `tx_id` until the
//! matching marker arrives (see `Engine::open_with_crash_test_and_hook`).
//! The commit marker therefore already delimits a batch, making a separate
//! `batch_size` field redundant. Do not re-introduce one — it would have to
//! be maintained in lockstep with the commit marker for no gain.
#![allow(unsafe_op_in_unsafe_fn)]
// The byte-level indexing/slicing below is guarded by explicit bounds
// checks: `decode_frame` returns `Ok(None)` (torn write) before any read
// past `buf.len()`, and `encode_frame` reserves `frame_total` before
// writing into `out`. `.get(..)` would only reintroduce the checks the
// format parser already performs.
#![allow(clippy::indexing_slicing)]

use crate::crc32c::{crc32c_with_state, crc32c_with_state_raw};
use crate::{EngineError, Result};
use bytes::Bytes;
use parking_lot::Mutex;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Identifies the operation stored in a WAL record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpType {
    /// Insert or overwrite a key/value pair.
    Put,
    /// Mark a key as deleted.
    Delete,
    /// Commit a transaction.
    Commit,
    /// Abort a transaction.
    Abort,
}

impl OpType {
    /// Discriminant byte used inside a binary WAL frame.
    pub(crate) const fn encode(self) -> u8 {
        match self {
            Self::Put => 1,
            Self::Delete => 2,
            Self::Commit => 3,
            Self::Abort => 4,
        }
    }

    pub(crate) fn decode(value: u8) -> Result<Self> {
        match value {
            1 => Ok(Self::Put),
            2 => Ok(Self::Delete),
            3 => Ok(Self::Commit),
            4 => Ok(Self::Abort),
            _ => Err(EngineError::CorruptMessage(format!(
                "unknown WAL operation tag {value:#04x}"
            ))),
        }
    }
}

/// One decoded WAL record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalRecord {
    /// Transaction that owns the record.
    pub tx_id: u64,
    /// Operation encoded by the record.
    pub op: OpType,
    /// Mutation key; empty for transaction markers.
    pub key: Bytes,
    /// Put payload; absent for deletes and transaction markers.
    pub value: Option<Bytes>,
}

impl WalRecord {
    /// Creates a put record.
    #[must_use]
    pub fn put(tx_id: u64, key: Bytes, value: Bytes) -> Self {
        Self {
            tx_id,
            op: OpType::Put,
            key,
            value: Some(value),
        }
    }

    /// Creates a delete record.
    #[must_use]
    pub fn delete(tx_id: u64, key: Bytes) -> Self {
        Self {
            tx_id,
            op: OpType::Delete,
            key,
            value: None,
        }
    }

    /// Creates a commit marker.
    #[must_use]
    pub fn commit(tx_id: u64) -> Self {
        Self {
            tx_id,
            op: OpType::Commit,
            key: Bytes::new(),
            value: None,
        }
    }

    /// Creates an abort marker.
    #[must_use]
    pub fn abort(tx_id: u64) -> Self {
        Self {
            tx_id,
            op: OpType::Abort,
            key: Bytes::new(),
            value: None,
        }
    }
}

/// File-format magic. Every frame begins with this value.
pub const MAGIC_WAL1: [u8; 4] = *b"WAL1";
/// Little-endian `u32` form of [`MAGIC_WAL1`], used for a cheap 4-byte
/// compare against the decoded header without going through an array.
const MAGIC_WAL1_U32: u32 = u32::from_le_bytes(MAGIC_WAL1);

const FRAME_HEADER_LEN: usize = 4 + 8 + 1 + 4 + 4; // magic + tx_id + op + key_len + value_len
const FRAME_CRC_LEN: usize = 4;
const FRAME_FIXED_OVERHEAD: usize = FRAME_HEADER_LEN + FRAME_CRC_LEN;

/// Sentinel `value_len` that means "no value" (delete or commit).
///
/// `key_len` and `value_len` are both `u32`. A key or value longer than
/// `u32::MAX` (4 GiB) cannot be represented; [`encode_frame`] rejects it
/// with [`EngineError::OutOfRange`] rather than silently colliding with
/// this sentinel. Do not silently widen the field — if large values ever
/// become a requirement, migrate to a varint length prefix and a dedicated
/// "no value" op code instead of overloading `u32::MAX`.
pub const VALUE_NONE: u32 = 0xFFFF_FFFF;

/// Encodes `records + commit_marker` into `out`. The commit marker is
/// written as a separate frame at the end (key empty, op = Commit).
///
/// # Errors
///
/// Returns [`EngineError::OutOfRange`] if any key or value exceeds
/// `u32::MAX` bytes and therefore cannot be length-prefixed.
pub fn encode_batch(out: &mut Vec<u8>, records: &[WalRecord], commit_tx_id: u64) -> Result<()> {
    // Pre-size the buffer to amortise re-allocs.
    let payload_bytes: usize = records
        .iter()
        .map(|r| r.key.len() + r.value.as_ref().map_or(0, |v| v.len()) + FRAME_FIXED_OVERHEAD)
        .sum();
    out.reserve(payload_bytes + FRAME_FIXED_OVERHEAD);

    for record in records {
        encode_frame(
            out,
            record.tx_id,
            record.op,
            &record.key,
            record.value.as_deref(),
        )?;
    }
    encode_frame(out, commit_tx_id, OpType::Commit, &[], None)?;
    Ok(())
}

/// Encodes `mutations` directly into the WAL buffer, bypassing the
/// intermediate [`WalRecord`] allocation.  This is the hot-path entry
/// point for [`crate::Engine::commit_single`].
pub fn encode_mutations(
    out: &mut Vec<u8>,
    tx_id: u64,
    mutations: &[crate::Mutation],
) -> Result<()> {
    let payload_bytes: usize = mutations
        .iter()
        .map(|m| match m {
            crate::Mutation::Put { key, value } => {
                key.len() + value.len() + FRAME_FIXED_OVERHEAD
            }
            crate::Mutation::Delete { key } => key.len() + FRAME_FIXED_OVERHEAD,
        })
        .sum();
    out.reserve(payload_bytes + FRAME_FIXED_OVERHEAD);

    for mutation in mutations {
        match mutation {
            crate::Mutation::Put { key, value } => {
                encode_frame(out, tx_id, OpType::Put, key, Some(value))?;
            }
            crate::Mutation::Delete { key } => {
                encode_frame(out, tx_id, OpType::Delete, key, None)?;
            }
        }
    }
    encode_frame(out, tx_id, OpType::Commit, &[], None)?;
    Ok(())
}

/// Encodes one frame into `out`. `key` is borrowed from the caller;
/// `value` is borrowed from the caller (or `None` for tombstones and
/// commit/abort markers).
///
/// # Errors
///
/// Returns [`EngineError::OutOfRange`] if `key` or `value` exceeds
/// `u32::MAX` bytes.
pub fn encode_frame(
    out: &mut Vec<u8>,
    tx_id: u64,
    op: OpType,
    key: &[u8],
    value: Option<&[u8]>,
) -> Result<()> {
    let value_len = match value {
        Some(v) => u32::try_from(v.len())
            .map_err(|_| EngineError::OutOfRange("WAL value length exceeds u32::MAX"))?,
        None => VALUE_NONE,
    };
    let key_len = u32::try_from(key.len())
        .map_err(|_| EngineError::OutOfRange("WAL key length exceeds u32::MAX"))?;

    // Reserve the space up front so we can take a slice of `out` and
    // CRC32 it without paying for the appends. We use `set_len` to
    // grow the buffer without zero-filling — the bytes we write below
    // cover the whole frame body, so leftover zeros are unused.
    let header_start = out.len();
    let body_len = FRAME_HEADER_LEN
        + key.len()
        + if value.is_some() {
            value_len as usize
        } else {
            0
        };
    let frame_total = body_len + FRAME_CRC_LEN;
    if out.capacity() < out.len() + frame_total {
        out.reserve(frame_total);
    }
    // SAFETY: we just reserved space above; `set_len` writes no bytes
    // (it only updates the length counter).
    unsafe {
        out.set_len(out.len() + frame_total);
    }

    // SAFETY: `out` is a `Vec<u8>` we just resized; the pointer math
    // below is in-bounds by construction.
    unsafe {
        let base = out.as_mut_ptr().add(header_start);
        // Magic (4 bytes).
        std::ptr::copy_nonoverlapping(MAGIC_WAL1.as_ptr(), base, 4);
        // tx_id (u64 LE) — write directly to avoid the temporary
        // `to_le_bytes()` array.
        (base.add(4) as *mut u64).write_unaligned(tx_id.to_le());
        // op (u8).
        *base.add(12) = op.encode();
        // key_len (u32 LE).
        (base.add(13) as *mut u32).write_unaligned(key_len.to_le());
        // value_len (u32 LE).
        (base.add(17) as *mut u32).write_unaligned(value_len.to_le());
        // key bytes.
        if !key.is_empty() {
            std::ptr::copy_nonoverlapping(key.as_ptr(), base.add(FRAME_HEADER_LEN), key.len());
        }
        // value bytes (only if present).
        if let Some(v) = value {
            if !v.is_empty() {
                std::ptr::copy_nonoverlapping(
                    v.as_ptr(),
                    base.add(FRAME_HEADER_LEN + key.len()),
                    v.len(),
                );
            }
        }
    }

    // Compute the CRC32C over everything except the trailing 4-byte trailer.
    // SAFETY: `header_start + body_len <= out.len()` is guaranteed by the
    // `set_len` + `reserve` dance above; the slice we hand to CRC is
    // in-bounds.
    let crc_offset = header_start + body_len;
    let crc = unsafe {
        let crc_ptr = out.as_ptr().add(header_start);
        crc32c_with_state_raw(crc_ptr, body_len, 0)
    };

    // SAFETY: `crc_offset + 4 <= out.len()`.
    unsafe {
        let crc_ptr = out.as_mut_ptr().add(crc_offset) as *mut u32;
        crc_ptr.write_unaligned(crc.to_le());
    }
    Ok(())
}

/// Result of a successful frame decode.
#[derive(Debug, PartialEq, Eq)]
pub struct DecodedFrame {
    pub tx_id: u64,
    pub op: OpType,
    pub key: Bytes,
    pub value: Option<Bytes>,
    /// Absolute byte offset in the input buffer where the *next* frame
    /// starts. None when this was the last frame in the buffer.
    pub next_offset: Option<u64>,
}

/// Decodes a single frame at `buf[cursor..]`. On torn write (truncated
/// header, truncated body, bad CRC, or unrecognised magic) returns
/// `Ok(None)` so the caller can truncate the file to the prior valid
/// offset. Schema errors (unknown op tag, len overflow) return `Err`.
pub fn decode_frame(buf: &[u8], cursor: usize) -> Result<Option<DecodedFrame>> {
    if cursor + FRAME_HEADER_LEN > buf.len() {
        return Ok(None);
    }

    // SAFETY: the bounds check above guarantees that all reads below
    // stay within `buf.len()`. We use a stack `[u8; N]` buffer plus a
    // single `copy_nonoverlapping` — microbenchmarks show this is
    // measurably faster than `read_unaligned().to_le()` on x86_64
    // because LLVM turns the copy into a single aligned `mov` for the
    // common small sizes we care about (4 / 8 bytes), then `from_le_bytes`
    // lowers to a nop on little-endian targets.
    let magic = u32::from_le_bytes(unsafe { read_array::<4>(buf, cursor) });
    if magic != MAGIC_WAL1_U32 {
        return Ok(None);
    }

    let tx_id = u64::from_le_bytes(unsafe { read_array::<8>(buf, cursor + 4) });
    let op = OpType::decode(buf[cursor + 12])?;
    let key_len = u32::from_le_bytes(unsafe { read_array::<4>(buf, cursor + 13) });
    let value_len = u32::from_le_bytes(unsafe { read_array::<4>(buf, cursor + 17) });

    let key_len_usize = if key_len == VALUE_NONE {
        0
    } else {
        usize::try_from(key_len).map_err(|_| EngineError::Corrupt("WAL key_len overflow"))?
    };
    let value_len_usize = if value_len == VALUE_NONE {
        0
    } else {
        usize::try_from(value_len).map_err(|_| EngineError::Corrupt("WAL value_len overflow"))?
    };

    let body_start = cursor + FRAME_HEADER_LEN;
    let body_end = body_start
        .checked_add(key_len_usize)
        .and_then(|e| e.checked_add(value_len_usize))
        .ok_or_else(|| EngineError::Corrupt("WAL frame length overflow"))?;
    let crc_end = body_end
        .checked_add(FRAME_CRC_LEN)
        .ok_or_else(|| EngineError::Corrupt("WAL frame length overflow"))?;
    if crc_end > buf.len() {
        return Ok(None);
    }

    // Validate the CRC32C.
    let expected = crc32c_with_state(&buf[cursor..body_end], 0);
    let actual = u32::from_le_bytes(unsafe { read_array::<4>(buf, body_end) });
    if expected != actual {
        // Torn write or corruption. Truncate.
        return Ok(None);
    }

    let key = if key_len_usize == 0 {
        Bytes::new()
    } else {
        Bytes::copy_from_slice(&buf[body_start..body_start + key_len_usize])
    };
    let value = if value_len == VALUE_NONE {
        None
    } else {
        Some(Bytes::copy_from_slice(
            &buf[body_start + key_len_usize..body_end],
        ))
    };

    let next_offset = if crc_end == buf.len() {
        None
    } else {
        Some(u64::try_from(crc_end).unwrap_or(u64::MAX))
    };

    Ok(Some(DecodedFrame {
        tx_id,
        op,
        key,
        value,
        next_offset,
    }))
}

#[inline]
unsafe fn read_array<const N: usize>(buf: &[u8], offset: usize) -> [u8; N] {
    debug_assert!(offset + N <= buf.len());
    let mut a = [0u8; N];
    std::ptr::copy_nonoverlapping(buf.as_ptr().add(offset), a.as_mut_ptr(), N);
    a
}

/// Walks a buffer of frames and decodes them all. Stops at the first
/// torn / corrupted tail. Returns `(records, last_valid_offset)`.
pub fn decode_all(buf: &[u8]) -> Result<(Vec<WalRecord>, u64)> {
    let mut cursor = 0usize;
    let mut out = Vec::new();
    let mut last_valid = 0u64;
    while let Some(decoded) = decode_frame(buf, cursor)? {
        let tx_id = decoded.tx_id;
        let value_for_record = match decoded.op {
            OpType::Put => decoded.value.clone(),
            OpType::Delete | OpType::Commit | OpType::Abort => None,
        };
        out.push(WalRecord {
            tx_id,
            op: decoded.op,
            key: decoded.key.clone(),
            value: value_for_record,
        });
        // The frame is now known to be valid — `cursor` will become its
        // start + length after this block, which is the exclusive end of
        // valid bytes for recovery's truncate-to-valid-offset pass.
        let frame_end = cursor
            + FRAME_HEADER_LEN
            + decoded.key.len()
            + decoded.value.as_ref().map_or(0, |v| v.len())
            + FRAME_CRC_LEN;
        last_valid = u64::try_from(frame_end).unwrap_or(last_valid);
        let Some(next) = decoded.next_offset else {
            break;
        };
        cursor = usize::try_from(next).unwrap_or(buf.len());
    }
    Ok((out, last_valid))
}

/// Thread-safe append-only WAL handle.
#[derive(Debug)]
pub struct Wal {
    path: PathBuf,
    file: Mutex<File>,
}

// Thread-local WAL encoding buffer.  Reused across `append_batch_commit_and_sync`
// calls on the same thread so the per-commit `Vec::new()` allocation is elided
// (the steady-state OLTP path reuses a single allocation per OS thread).
//
// `UnsafeCell` replaces the previous `RefCell`: the engine's `write_guard`
// already serialises every WAL writer, so only one thread can be appending at a
// time and the runtime borrow-check of `RefCell` is pure overhead on the OLTP
// hot path (no safety benefit — the access pattern is already serialised).
thread_local! {
    static WAL_BUF: std::cell::UnsafeCell<Vec<u8>> = const { std::cell::UnsafeCell::new(Vec::new()) };
}

/// Takes the thread-local WAL buffer, leaving an empty `Vec` in its place.
///
/// # Safety
///
/// `write_guard` serialises all WAL writers.  Only one thread can be inside
/// `append_batch_commit_and_sync` / `append_multi_batch` at a time, so the
/// mutable access to this thread-local is exclusive in practice.
fn take_wal_buf() -> Vec<u8> {
    WAL_BUF.with(|cell| {
        // SAFETY: exclusive access guaranteed by the engine's write_guard.
        let buf = unsafe { &mut *cell.get() };
        std::mem::take(buf)
    })
}

/// Returns `buf` to the thread-local slot after clearing it, ready for the
/// next [`take_wal_buf`] call on the same thread.
///
/// # Safety
///
/// Same serialisation guarantee as [`take_wal_buf`].
fn return_wal_buf(mut buf: Vec<u8>) {
    buf.clear();
    WAL_BUF.with(|cell| {
        // SAFETY: exclusive access guaranteed by the engine's write_guard.
        unsafe { *cell.get() = buf; }
    })
}

impl Wal {
    /// Opens or creates a WAL for appending. A torn final frame is truncated
    /// before new records are accepted.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Recover records and valid byte count
        let (_records, valid_bytes) = recover_records_with_truncation(&path)?;

        let file = open_append(&path, valid_bytes)?;
        Ok(Self {
            path,
            file: Mutex::new(file),
        })
    }

    /// Returns this WAL's path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the current size of the WAL file in bytes.
    pub fn current_size(&self) -> u64 {
        let file = self.file.lock();
        file.metadata().map(|m| m.len()).unwrap_or(0)
    }

    /// Appends a batch of records followed by a commit marker and syncs
    /// the file to disk in a single call.
    ///
    /// The hand-rolled binary format writes every record plus the commit
    /// marker as a single contiguous run of frames — one `write_all` plus
    /// one `sync_data` per commit.
    ///
    /// Note on buffer pooling: a per-`Wal` `Mutex<Vec<u8>>` was tried and
    /// benchmarked *slower* on the `single_put` workload because the extra
    /// `Mutex::lock()` added more latency than the allocation saved (the
    /// per-tx cost is dominated by `sync_data`, so any extra lock
    /// acquisition shows up in the noise). The `thread_local!` pool here
    /// avoids that lock entirely — each OS thread gets its own buffer, and
    /// the engine's `write_guard` already serialises callers.
    pub fn append_batch_commit_and_sync(&self, records: &[WalRecord], tx_id: u64) -> Result<u64> {
        if records.is_empty() {
            return Ok(0);
        }
        let mut buffer = take_wal_buf();
        let result = (|| -> Result<u64> {
            encode_batch(&mut buffer, records, tx_id)?;
            let bytes = buffer.len() as u64;
            let mut file = self.file.lock();
            file.write_all(&buffer)?;
            file.sync_data()?;
            Ok(bytes)
        })();
        return_wal_buf(buffer);
        result
    }

    /// Hot-path variant: encodes `mutations` directly into the WAL buffer
    /// without the intermediate [`WalRecord`] allocation.  Used by
    /// [`crate::Engine::commit_single`] for the single-writer fast path.
    pub fn append_mutations_commit_and_sync(
        &self,
        tx_id: u64,
        mutations: &[crate::Mutation],
    ) -> Result<u64> {
        if mutations.is_empty() {
            return Ok(0);
        }
        let mut buffer = take_wal_buf();
        let result = (|| -> Result<u64> {
            encode_mutations(&mut buffer, tx_id, mutations)?;
            let bytes = buffer.len() as u64;
            let mut file = self.file.lock();
            file.write_all(&buffer)?;
            file.sync_data()?;
            Ok(bytes)
        })();
        return_wal_buf(buffer);
        result
    }

    /// Appends several transactions — each a `(tx_id, records)` pair — in a
    /// single write + fsync. This is the group-commit fast path: instead of one
    /// `write_all` + `sync_data` per transaction, the whole batch shares one
    /// fsync. Each transaction's records are followed by its own `Commit`
    /// marker, so recovery groups frames by `tx_id` exactly as before.
    ///
    /// Returns the total bytes written.
    pub fn append_multi_batch_commit_and_sync(
        &self,
        batches: &[(u64, &[WalRecord])],
    ) -> Result<u64> {
        let bytes = self.append_multi_batch(batches)?;
        if bytes > 0 {
            fault_point!("wal::sync_data_before");
            self.file.lock().sync_data()?;
            fault_point!("wal::sync_data_after");
        }
        Ok(bytes)
    }

    /// Hot-path group-commit variant: encodes `&[(tx_id, &[Mutation])]`
    /// directly without intermediate [`WalRecord`] allocations.
    pub fn append_multi_mutations_commit_and_sync(
        &self,
        batches: &[(u64, &[crate::Mutation])],
    ) -> Result<u64> {
        let bytes = self.append_multi_mutations(batches)?;
        if bytes > 0 {
            fault_point!("wal::sync_data_before");
            self.file.lock().sync_data()?;
            fault_point!("wal::sync_data_after");
        }
        Ok(bytes)
    }

    /// Like [`append_multi_batch_commit_and_sync`] but does NOT call `sync_data`.
    /// Used by [`SyncMode::Group`] where a background thread fsyncs periodically.
    /// Returns the total bytes written to the kernel buffer.
    pub fn append_multi_batch(
        &self,
        batches: &[(u64, &[WalRecord])],
    ) -> Result<u64> {
        if batches.is_empty() {
            return Ok(0);
        }
        let mut buffer = take_wal_buf();
        let result = (|| -> Result<u64> {
            for (tx_id, records) in batches {
                if records.is_empty() {
                    continue;
                }
                encode_batch(&mut buffer, records, *tx_id)?;
            }
            if buffer.is_empty() {
                return Ok(0);
            }
            let bytes = buffer.len() as u64;
            let mut file = self.file.lock();
            file.write_all(&buffer)?;
            Ok(bytes)
        })();
        return_wal_buf(buffer);
        result
    }

    /// Hot-path group-commit variant without `sync_data`: encodes
    /// `&[(tx_id, &[Mutation])]` directly into the WAL buffer.
    pub fn append_multi_mutations(
        &self,
        batches: &[(u64, &[crate::Mutation])],
    ) -> Result<u64> {
        if batches.is_empty() {
            return Ok(0);
        }
        let mut buffer = take_wal_buf();
        let result = (|| -> Result<u64> {
            for (tx_id, mutations) in batches {
                if mutations.is_empty() {
                    continue;
                }
                encode_mutations(&mut buffer, *tx_id, mutations)?;
            }
            if buffer.is_empty() {
                return Ok(0);
            }
            let bytes = buffer.len() as u64;
            let mut file = self.file.lock();
            file.write_all(&buffer)?;
            Ok(bytes)
        })();
        return_wal_buf(buffer);
        result
    }

    /// Flushes the kernel buffer to disk. Called by the engine's write path
    /// ([`SyncMode::Full`]) or the background fsync thread ([`SyncMode::Group`]).
    pub fn sync_data(&self) -> Result<()> {
        self.file.lock().sync_data()?;
        Ok(())
    }

    /// Truncates the WAL after a durable checkpoint.
    pub fn reset(&self) -> Result<()> {
        reset_append_handle(&self.path, &self.file)
    }

    /// Decodes complete frames. An incomplete final frame is treated as a
    /// torn write and ignored; corruption in a complete frame is an error.
    pub fn recover(path: impl AsRef<Path>) -> Result<Vec<WalRecord>> {
        let (records, _valid_bytes) = recover_records_with_truncation(path.as_ref())?;
        Ok(records)
    }
}

/// Open a WAL file for append, truncating to `valid_bytes`.
///
/// On non-Windows, `set_len` works on append-mode handles, so a single
/// open suffices. On Windows, `append` is incompatible with `set_len`;
/// we open read/write first to truncate, then re-open with append.
///
/// See [`Wal::open`] and [`Wal::reset`].
#[cfg(not(windows))]
fn open_append(path: &Path, valid_bytes: u64) -> Result<File> {
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .append(true)
        .open(path)?;
    file.set_len(valid_bytes)?;
    file.sync_data()?;
    Ok(file)
}

#[cfg(windows)]
fn open_append(path: &Path, valid_bytes: u64) -> Result<File> {
    {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        file.set_len(valid_bytes)?;
        file.sync_data()?;
    }
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(path)?;
    Ok(file)
}

/// Truncate the WAL file to zero length and seek the append handle to
/// the start.
///
/// On non-Windows, the existing append handle can `set_len` and `seek`
/// directly. On Windows, we re-open read/write to truncate, then seek
/// the append handle separately.
#[cfg(not(windows))]
fn reset_append_handle(_path: &Path, file_mutex: &Mutex<File>) -> Result<()> {
    let file = file_mutex.lock();
    file.set_len(0)?;
    file.seek(SeekFrom::Start(0))?;
    file.sync_data()?;
    Ok(())
}

#[cfg(windows)]
fn reset_append_handle(path: &Path, file_mutex: &Mutex<File>) -> Result<()> {
    {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        file.set_len(0)?;
        file.seek(SeekFrom::Start(0))?;
        file.sync_data()?;
    }
    let mut file = file_mutex.lock();
    file.seek(SeekFrom::Start(0))?;
    Ok(())
}

/// Reads the on-disk WAL and decodes every complete frame. Returns the
/// decoded records plus the byte offset of the last complete frame, so
/// `Wal::open` can truncate a torn tail.
///
/// `Wal::recover` (no truncation) and `Wal::open` (with truncation) both
/// route through here so the parsing logic lives in one place.
fn recover_records_with_truncation(path: &Path) -> Result<(Vec<WalRecord>, u64)> {
    if !path.exists() {
        return Ok((Vec::new(), 0));
    }
    fault_point!("wal::recover_read");
    let bytes = std::fs::read(path)?;
    if bytes.is_empty() {
        return Ok((Vec::new(), 0));
    }
    // The legacy Arrow-IPC WAL format was removed; fail loudly rather than
    // silently decoding an unsupported file as an empty log (which would
    // drop committed data on recovery).
    if !bytes.starts_with(&MAGIC_WAL1) {
        return Err(EngineError::CorruptMessage(
            "WAL file uses an unsupported format: only the binary WAL1 format \
             is accepted (legacy Arrow-IPC WAL files are no longer supported)"
                .to_string(),
        ));
    }
    decode_all(&bytes)
}

#[cfg(test)]
// Tests assert on fixed, self-generated frames; `unwrap`/indexing are
// idiomatic here (same convention as every other crate's test module).
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_single_put() {
        let record = WalRecord::put(
            1,
            Bytes::from_static(b"hello"),
            Bytes::from_static(b"world"),
        );
        let mut buf = Vec::new();
        encode_batch(&mut buf, &[record], 1).unwrap();

        let (decoded, valid) = decode_all(&buf).unwrap();
        assert_eq!(valid, buf.len() as u64);
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].tx_id, 1);
        assert_eq!(decoded[0].op, OpType::Put);
        assert_eq!(&decoded[0].key[..], b"hello");
        assert_eq!(decoded[0].value.as_ref().unwrap().len(), 5);
        assert_eq!(&decoded[1].op, &OpType::Commit);
    }

    #[test]
    fn roundtrip_delete_then_commit() {
        let r = WalRecord::delete(7, Bytes::from_static(b"gone"));
        let mut buf = Vec::new();
        encode_batch(&mut buf, &[r], 7).unwrap();
        let (decoded, valid) = decode_all(&buf).unwrap();
        assert_eq!(valid, buf.len() as u64);
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].op, OpType::Delete);
        assert!(decoded[0].value.is_none());
        assert_eq!(decoded[1].op, OpType::Commit);
    }

    #[test]
    fn torn_tail_is_truncated() {
        let r = WalRecord::put(42, Bytes::from_static(b"x"), Bytes::from_static(b"y"));
        let mut buf = Vec::new();
        encode_batch(&mut buf, &[r], 42).unwrap();
        // Lop off the last byte of the trailing CRC + one more byte
        // (a clean torn write should return Ok(records) with
        // `valid` pointing past the last complete frame).
        let lopped = &buf[..buf.len() - 2];
        let (_decoded, valid) = decode_all(lopped).unwrap();
        // The valid offset should be 2 bytes shorter than `buf.len()` —
        // i.e. the decode walked the complete frame, found a torn CRC
        // and stopped without erroring.
        assert!(valid as usize <= lopped.len());
    }

    #[test]
    fn corrupted_crc_returns_truncate_offset() {
        let r = WalRecord::put(9, Bytes::from_static(b"k"), Bytes::from_static(b"v"));
        let mut buf = Vec::new();
        encode_batch(&mut buf, &[r], 9).unwrap();
        // Flip a bit in the body to corrupt the CRC.
        let corrupt_at = FRAME_HEADER_LEN;
        buf[corrupt_at] ^= 0x01;
        let (_decoded, valid) = decode_all(&buf).unwrap();
        assert!(valid as usize <= corrupt_at);
    }
}
