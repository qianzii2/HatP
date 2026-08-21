//! Durable MANIFEST — append-only binary edit log + periodic snapshot rewrite.
//!
//! # Format
//!
//! The MANIFEST is a sequence of self-describing binary frames (no Arrow IPC,
//! which could not be appended across schema messages and forced a full rewrite
//! on every edit). Each frame is:
//!
//! ```text
//!   magic    [u8; 4]  "MAN1"
//!   op       u8       1=AddFile 2=DeleteFile 3=NextFileId
//!                     4=DroppedPrefix 5=UndroppedPrefix
//!   payload  (op-dependent, all integers little-endian)
//!   crc32c   u32 LE   (covers magic .. end of payload)
//! ```
//!
//! # Append + snapshot
//!
//! [`Manifest::append_batch`] appends the new frames in one `write_all` +
//! `sync_data` (no full rewrite). Once the edit count reaches
//! [`SNAPSHOT_THRESHOLD`], the current [`VersionSet`] is serialized as a fresh
//! "current state" batch and atomically replaces the file (tmp + rename), so
//! the log never grows without bound and recovery stays O(live files).
//!
//! A torn final frame (crash mid-append) is detected by a bad magic / CRC and
//! silently truncated on open, exactly like the WAL.

use crate::crc32c::{crc32c_with_state, crc32c_with_state_raw};
use crate::{EngineError, Result};
use arc_swap::ArcSwap;
use bytes::Bytes;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// File-format magic. Every frame begins with this value.
const MAGIC: [u8; 4] = *b"MAN1";
/// Frame fixed overhead: magic (4) + op (1) + crc (4).
const FRAME_OVERHEAD: usize = 9;

const OP_ADD_FILE: u8 = 1;
const OP_DELETE_FILE: u8 = 2;
const OP_NEXT_FILE_ID: u8 = 3;
const OP_DROPPED_PREFIX: u8 = 4;
const OP_UNDROPPED_PREFIX: u8 = 5;

/// Number of accumulated edits before the log is rewritten as a snapshot of
/// the current [`VersionSet`] (bounds the file size and recovery cost).
const SNAPSHOT_THRESHOLD: usize = 1024;

/// Atomic edit applied to the manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VersionEdit {
    /// Add a file to a level.
    AddFile {
        /// File id.
        file_id: u64,
        /// Level the file lives at.
        level: u32,
        /// Inclusive smallest key in the file.
        min_key: Bytes,
        /// Inclusive largest key in the file.
        max_key: Bytes,
        /// Approximate bytes on disk.
        bytes: u64,
        /// Wall-clock seconds at which this file was written, used by the
        /// compaction picker for age-based prioritization.
        created_at: u64,
    },
    /// Remove a file from a level.
    DeleteFile {
        /// File id.
        file_id: u64,
        /// Level the file lived at.
        level: u32,
    },
    /// Bump the next globally unique file id.
    NextFileId(u64),
    /// Record a dropped-table prefix so orphaned SST data is filtered after
    /// a restart. `DROP TABLE` persists this edit; the prefix is the raw
    /// `table\0` key prefix.
    DroppedPrefix(Vec<u8>),
    /// Clear a previously recorded dropped-table prefix (a table re-created
    /// with the same name after `undrop_table_prefix`).
    UndroppedPrefix(Vec<u8>),
}

/// Immutable-file membership reconstructed from manifest edits.
///
/// Stores both file membership (levels -> file_ids) and per-file metadata
/// (min_key, max_key, bytes) for use by the compaction picker.
#[derive(Debug, Clone, Default)]
pub struct VersionSet {
    levels: BTreeMap<u32, BTreeSet<u64>>,
    /// Per-file metadata: file_id -> (min_key, max_key, bytes, created_at).
    file_meta: BTreeMap<u64, (Bytes, Bytes, u64, u64)>,
    next_file_id: u64,
    /// Dropped-table prefixes (`table\0`) accumulated from `DroppedPrefix`
    /// / `UndroppedPrefix` edits. Replayed into the engine on open so the
    /// drop filter survives a restart.
    dropped: BTreeSet<Vec<u8>>,
    /// Files sorted by min_key for O(log n) point-query file selection (PR 5.2).
    /// Each entry is `(min_key, file_id)`. Rebuilt on every apply().
    files_by_key: Vec<(Bytes, u64)>,
    /// Pre-built file_id → level lookup, used by `Engine::get` for L1+ early-exit.
    /// Rebuilt on every apply() alongside [`files_by_key`] so the hot read path
    /// never allocates a HashMap per call.
    file_level: Vec<(u64, u32)>,
}

impl VersionSet {
    /// Applies one edit to the in-memory state.  Does NOT rebuild the
    /// derived indexes — call [`rebuild_indexes`] after a batch of edits.
    pub fn apply(&mut self, edit: VersionEdit) {
        match edit {
            VersionEdit::AddFile {
                file_id,
                level,
                min_key,
                max_key,
                bytes,
                created_at,
            } => {
                self.levels.entry(level).or_default().insert(file_id);
                self.file_meta
                    .insert(file_id, (min_key, max_key, bytes, created_at));
                self.next_file_id = self.next_file_id.max(file_id.saturating_add(1));
            }
            VersionEdit::DeleteFile { file_id, level } => {
                if let Some(files) = self.levels.get_mut(&level) {
                    files.remove(&file_id);
                    if files.is_empty() {
                        self.levels.remove(&level);
                    }
                }
                self.file_meta.remove(&file_id);
            }
            VersionEdit::NextFileId(file_id) => {
                self.next_file_id = self.next_file_id.max(file_id);
            }
            VersionEdit::DroppedPrefix(prefix) => {
                self.dropped.insert(prefix);
            }
            VersionEdit::UndroppedPrefix(prefix) => {
                self.dropped.remove(&prefix);
            }
        }
    }

    /// Rebuilds the derived indexes after a batch of edits.  Call this once
    /// after applying all edits, not after each individual edit.
    pub fn rebuild_indexes(&mut self) {
        self.rebuild_files_by_key();
        self.rebuild_file_level();
    }

    /// Returns candidate file_ids whose [min_key, max_key] range may contain
    /// `search_key`. Uses binary search over the sorted files_by_key for
    /// O(log n) lookup instead of iterating all files.
    #[must_use]
    pub fn candidate_files(&self, search_key: &[u8]) -> Vec<u64> {
        if self.files_by_key.is_empty() {
            return Vec::new();
        }
        // Binary search for the last file whose min_key <= search_key.
        let idx = self
            .files_by_key
            .partition_point(|(min_key, _)| min_key.as_ref() <= search_key);
        // All files up to `idx` have min_key <= search_key. Collect those
        // whose max_key >= search_key.
        let mut out = Vec::new();
        #[allow(clippy::indexing_slicing)]
        {
        for (min_key, file_id) in self.files_by_key[..idx].iter().rev() {
            if let Some((_, max_key, _, _)) = self.file_meta.get(file_id) {
                if max_key.is_empty() || max_key.as_ref() >= search_key {
                    out.push(*file_id);
                }
            }
            // Stop early if this file's max_key is too far below search_key
            // (files are sorted by min_key, but max_key is not monotonic).
            // We can't stop early in general because L0 files overlap.
            // For L1+ (leveled), max_key is also monotonic, but we don't
            // track per-file level here. The cost is O(candidate_files) which
            // is typically 1-3 for leveled compaction.
            let _ = min_key;
        }
        } // end allow(clippy::indexing_slicing)
        out
    }

    /// Rebuilds `files_by_key` from `file_meta`, sorted by min_key ascending.
    fn rebuild_files_by_key(&mut self) {
        self.files_by_key.clear();
        self.files_by_key.extend(
            self.file_meta
                .iter()
                .map(|(id, (min_key, _, _, _))| (min_key.clone(), *id)),
        );
        self.files_by_key.sort_by(|a, b| a.0.cmp(&b.0));
    }

    /// Rebuilds `file_level` from `levels`, sorted by file_id for binary search.
    /// This is the pre-computed replacement for the HashMap that `Engine::get`
    /// used to rebuild on every call.
    fn rebuild_file_level(&mut self) {
        self.file_level.clear();
        self.file_level.extend(
            self.levels
                .iter()
                .flat_map(|(level, files)| files.iter().map(move |f| (*f, *level))),
        );
        self.file_level.sort_unstable_by_key(|(id, _)| *id);
    }

    /// Returns the level of `file_id`, or `None` when unknown. Binary search
    /// over the pre-sorted `file_level` vector — O(log n) and allocation-free.
    #[must_use]
    pub fn file_level_for(&self, file_id: u64) -> Option<u32> {
        let idx = self
            .file_level
            .binary_search_by_key(&file_id, |(id, _)| *id)
            .ok()?;
        // SAFETY: `idx` comes from a successful binary_search_by_key.
        #[allow(clippy::indexing_slicing)]
        Some(self.file_level[idx].1)
    }

    /// Returns files at `level` in ascending id order.
    #[must_use]
    pub fn files(&self, level: u32) -> Vec<u64> {
        self.levels
            .get(&level)
            .map(|files| files.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Returns a copy of this version set with every `edits` applied.
    ///
    /// This is the immutable-append helper used by [`Manifest::append_batch`]:
    /// the current version set is shared (`Arc<ArcSwap>`), so mutating it in
    /// place would be visible to readers mid-append. Cloning into a fresh
    /// `VersionSet`, applying the edits, and then atomically swapping the
    /// pointer keeps readers on a consistent snapshot.
    #[must_use]
    pub fn applied(&self, edits: &[VersionEdit]) -> VersionSet {
        let mut next = self.clone();
        for edit in edits {
            next.apply(edit.clone());
        }
        next.rebuild_indexes();
        next
    }

    /// Returns all populated levels and their files.
    #[must_use]
    pub fn levels(&self) -> &BTreeMap<u32, BTreeSet<u64>> {
        &self.levels
    }

    /// Returns the id to allocate next.
    #[must_use]
    pub fn next_file_id(&self) -> u64 {
        self.next_file_id
    }

    /// Returns the metadata (min_key, max_key, bytes, created_at) for a file,
    /// if known.
    #[must_use]
    pub fn file_metadata(&self, file_id: u64) -> Option<&(Bytes, Bytes, u64, u64)> {
        self.file_meta.get(&file_id)
    }

    /// Returns every currently-dropped table prefix (`table\0`).
    #[must_use]
    pub fn dropped_prefixes(&self) -> Vec<Vec<u8>> {
        self.dropped.iter().cloned().collect()
    }

    /// Serializes this version set into a "current state" edit batch — the
    /// snapshot written on a log rewrite.
    fn to_edits(&self) -> Vec<VersionEdit> {
        let mut edits = Vec::new();
        for (level, files) in &self.levels {
            for file_id in files {
                let (min_key, max_key, bytes, created_at) = self
                    .file_meta
                    .get(file_id)
                    .cloned()
                    .unwrap_or_else(|| (Bytes::new(), Bytes::new(), 0, 0));
                edits.push(VersionEdit::AddFile {
                    file_id: *file_id,
                    level: *level,
                    min_key,
                    max_key,
                    bytes,
                    created_at,
                });
            }
        }
        edits.push(VersionEdit::NextFileId(self.next_file_id));
        for prefix in &self.dropped {
            edits.push(VersionEdit::DroppedPrefix(prefix.clone()));
        }
        edits
    }
}

/// Append-only manifest backed by a binary edit log.
///
/// Writers append frames and advance the [`VersionSet`]; readers obtain a
/// consistent shared snapshot via [`Manifest::version_set`] (an
/// `ArcSwap::load_full`, lock-free and no deep copy). A rewrite (snapshot)
/// happens automatically past [`SNAPSHOT_THRESHOLD`] edits.
#[derive(Debug)]
pub struct Manifest {
    path: PathBuf,
    /// Append-mode handle. Owned exclusively (all mutation is behind the
    /// engine's `Mutex<Manifest>`), so no inner lock is needed.
    file: File,
    /// Shared, lock-free-readable version set.
    version_set: Arc<ArcSwap<VersionSet>>,
    /// Edits appended since the last snapshot rewrite.
    edit_count: usize,
}

impl Manifest {
    /// Opens or creates a manifest and replays every complete frame.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let edits = read_manifest_file(&path)?;
        let mut version_set = VersionSet::default();
        for edit in &edits {
            version_set.apply(edit.clone());
        }
        version_set.rebuild_indexes();
        let edit_count = edits.len();
        let file = open_append(&path)?;
        Ok(Self {
            path,
            file,
            version_set: Arc::new(ArcSwap::from_pointee(version_set)),
            edit_count,
        })
    }

    /// Appends a batch of edits as a single durable manifest commit.
    ///
    /// Crash semantics: the frames are written in one `write_all` + `sync_data`;
    /// a torn final frame is truncated on the next open. The in-memory
    /// [`VersionSet`] is only swapped in after the file is durable.
    pub fn append_batch(&mut self, edits: Vec<VersionEdit>) -> Result<()> {
        if edits.is_empty() {
            return Ok(());
        }
        let mut buf = Vec::with_capacity(edits.len() * 40);
        for edit in &edits {
            encode_frame(&mut buf, edit)?;
        }
        self.file.write_all(&buf)?;
        self.file.sync_data()?;
        // Durable now: publish the new version set atomically.
        let new_version_set = self.version_set.load().applied(&edits);
        self.version_set.store(Arc::new(new_version_set));
        self.edit_count = self.edit_count.saturating_add(edits.len());
        // Bound the log: rewrite as a snapshot once it grows past the
        // threshold. Do this after publishing so readers already see the edit.
        if self.edit_count >= SNAPSHOT_THRESHOLD {
            self.rewrite()?;
        }
        Ok(())
    }

    /// Rewrites the manifest as a single "current state" snapshot: serialize
    /// the [`VersionSet`] to a temp file, fsync, atomically rename over the
    /// manifest, then reopen the append handle.
    fn rewrite(&mut self) -> Result<()> {
        let edits = self.version_set.load().to_edits();
        let mut buf = Vec::with_capacity(edits.len() * 40);
        for edit in &edits {
            encode_frame(&mut buf, edit)?;
        }
        // `MANIFEST` has no extension, so `set_extension("tmp")` yields the
        // sibling `MANIFEST.tmp` (the old `set_extension("manifest.tmp")`
        // produced the redundant `MANIFEST.manifest.tmp`).
        let mut tmp = self.path.clone();
        tmp.set_extension("tmp");
        {
            let mut file = File::create(&tmp)?;
            file.write_all(&buf)?;
            // fsync the temp file BEFORE the rename publishes it.
            file.sync_all()?;
        }
        std::fs::rename(&tmp, &self.path)?;
        // fsync the parent directory so the rename itself is durable.
        sync_parent_dir(&self.path)?;
        // Reopen the append handle on the newly replaced file.
        self.file = open_append(&self.path)?;
        self.edit_count = 0;
        Ok(())
    }

    /// Returns the reconstructed version set as a shared snapshot.
    ///
    /// `load_full` bumps the inner `Arc`'s refcount and returns without locking
    /// or deep-copying the `BTreeMap`s.
    #[must_use]
    pub fn version_set(&self) -> Arc<VersionSet> {
        self.version_set.load_full()
    }

    /// Returns a clone of the inner `Arc<ArcSwap<VersionSet>>`. The caller can
    /// then `load_full()` it without acquiring the outer `Mutex<Manifest>`
    /// (which only protects the file handle, not the in-memory version set).
    /// Read-mostly call sites that only need the version set should prefer
    /// this method over going through the outer mutex.
    #[must_use]
    pub fn version_set_arc(&self) -> Arc<ArcSwap<VersionSet>> {
        Arc::clone(&self.version_set)
    }

    /// Returns every SST file id across all levels, in `(level, file_id)`
    /// order.
    ///
    /// The order comes from `BTreeMap<level, BTreeSet<file_id>>`, which is
    /// ascending by level then by file id. `delete_plan` relies on this order:
    /// L0 (the newest flush) is visited first, so the first positive hit is
    /// the newest visible version.
    #[must_use]
    pub fn file_ids(&self) -> Vec<u64> {
        self.version_set
            .load()
            .levels()
            .values()
            .flat_map(|files| files.iter().copied())
            .collect()
    }

    /// Returns the manifest path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

// ── Binary encoding / decoding ────────────────────────────────────────────────

/// Appends one frame for `edit` to `out`.  Uses `set_len` + `write_unaligned`
/// (same pattern as WAL encoder) to avoid zero-fill and per-field memcpy calls.
fn encode_frame(out: &mut Vec<u8>, edit: &VersionEdit) -> Result<()> {
    let header_start = out.len();
    // Compute total frame size: magic(4) + op(1) + payload + crc(4).
    let body_len = 5 + edit_payload_len(edit); // magic + op + payload, no CRC
    let frame_total = body_len + 4; // + CRC32C
    if out.capacity() < out.len() + frame_total {
        out.reserve(frame_total);
    }
    // SAFETY: we just reserved; set_len skips zero-fill.
    unsafe { out.set_len(out.len() + frame_total); }
    let base = unsafe { out.as_mut_ptr().add(header_start) };
    // SAFETY: base..base+frame_total is in-bounds.
    unsafe {
        std::ptr::copy_nonoverlapping(MAGIC.as_ptr(), base, 4);
        let mut pos = base.add(5); // after magic(4) + op(1)
        match edit {
            VersionEdit::AddFile {
                file_id,
                level,
                min_key,
                max_key,
                bytes,
                created_at,
            } => {
                *base.add(4) = OP_ADD_FILE;
                (pos as *mut u64).write_unaligned(file_id.to_le());
                pos = pos.add(8);
                (pos as *mut u32).write_unaligned(level.to_le());
                pos = pos.add(4);
                pos = write_len_prefixed_raw(pos, min_key)?;
                pos = write_len_prefixed_raw(pos, max_key)?;
                (pos as *mut u64).write_unaligned(bytes.to_le());
                pos = pos.add(8);
                (pos as *mut u64).write_unaligned(created_at.to_le());
            }
            VersionEdit::DeleteFile { file_id, level } => {
                *base.add(4) = OP_DELETE_FILE;
                (pos as *mut u64).write_unaligned(file_id.to_le());
                pos = pos.add(8);
                (pos as *mut u32).write_unaligned(level.to_le());
            }
            VersionEdit::NextFileId(file_id) => {
                *base.add(4) = OP_NEXT_FILE_ID;
                (pos as *mut u64).write_unaligned(file_id.to_le());
            }
            VersionEdit::DroppedPrefix(prefix) => {
                *base.add(4) = OP_DROPPED_PREFIX;
                write_len_prefixed_raw(pos, prefix)?;
            }
            VersionEdit::UndroppedPrefix(prefix) => {
                *base.add(4) = OP_UNDROPPED_PREFIX;
                write_len_prefixed_raw(pos, prefix)?;
            }
        }
    }
    // CRC32C over everything except the trailing 4-byte trailer.
    let crc = unsafe {
        crc32c_with_state_raw(base, body_len, 0)
    };
    unsafe {
        let crc_ptr = base.add(body_len) as *mut u32;
        crc_ptr.write_unaligned(crc.to_le());
    }
    Ok(())
}

/// Returns the payload byte count for `edit` (excludes magic + op + crc).
fn edit_payload_len(edit: &VersionEdit) -> usize {
    match edit {
        VersionEdit::AddFile { min_key, max_key, .. } => {
            8 + 4 + 4 + min_key.len() + 4 + max_key.len() + 8 + 8
        }
        VersionEdit::DeleteFile { .. } => 8 + 4,
        VersionEdit::NextFileId(_) => 8,
        VersionEdit::DroppedPrefix(prefix) | VersionEdit::UndroppedPrefix(prefix) => {
            4 + prefix.len()
        }
    }
}

/// Writes `bytes` as u32 LE length prefix + raw bytes at `pos`.
/// Returns the advanced pointer.
unsafe fn write_len_prefixed_raw(pos: *mut u8, bytes: &[u8]) -> Result<*mut u8> {
    let len = u32::try_from(bytes.len())
        .map_err(|_| EngineError::OutOfRange("manifest field > u32::MAX"))?;
    // SAFETY: pos is in-bounds (part of the reserved frame buffer).
    unsafe { (pos as *mut u32).write_unaligned(len.to_le()); }
    let data_pos = unsafe { pos.add(4) };
    if !bytes.is_empty() {
        // SAFETY: data_pos..data_pos+bytes.len() is within the reserved frame.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), data_pos, bytes.len()); }
    }
    Ok(unsafe { data_pos.add(bytes.len()) })
}

/// Decodes one frame at `buf[cursor..]`. Returns `Ok(None)` on a torn tail
/// (truncated header/body or bad magic/CRC) so the caller truncates there.
fn decode_frame(buf: &[u8], cursor: usize) -> Result<Option<(VersionEdit, usize)>> {
    if cursor + FRAME_OVERHEAD > buf.len() {
        return Ok(None);
    }
    if buf.get(cursor..cursor + 4) != Some(&MAGIC[..]) {
        return Ok(None);
    }
    // `cursor + FRAME_OVERHEAD <= buf.len()` was checked above, so `cursor + 4`
    // is in-bounds.
    let Some(&op) = buf.get(cursor + 4) else {
        return Ok(None);
    };
    // `body_start` is where the op-dependent payload begins.
    let mut pos = cursor + 5;
    let edit = match op {
        OP_ADD_FILE => {
            let (file_id, pos1) = take_u64(buf, pos)?;
            let (level, pos2) = take_u32(buf, pos1)?;
            let (min_key, pos3) = take_len_prefixed(buf, pos2)?;
            let (max_key, pos4) = take_len_prefixed(buf, pos3)?;
            let (bytes, pos5) = take_u64(buf, pos4)?;
            let (created_at, pos6) = take_u64(buf, pos5)?;
            pos = pos6;
            VersionEdit::AddFile {
                file_id,
                level,
                min_key: Bytes::from(min_key),
                max_key: Bytes::from(max_key),
                bytes,
                created_at,
            }
        }
        OP_DELETE_FILE => {
            let (file_id, pos1) = take_u64(buf, pos)?;
            let (level, pos2) = take_u32(buf, pos1)?;
            pos = pos2;
            VersionEdit::DeleteFile { file_id, level }
        }
        OP_NEXT_FILE_ID => {
            let (file_id, pos1) = take_u64(buf, pos)?;
            pos = pos1;
            VersionEdit::NextFileId(file_id)
        }
        OP_DROPPED_PREFIX | OP_UNDROPPED_PREFIX => {
            let (prefix, pos1) = take_len_prefixed(buf, pos)?;
            pos = pos1;
            if prefix.is_empty() {
                // A malformed empty prefix would poison the drop filter by
                // matching every key.
                return Err(EngineError::Corrupt(
                    "manifest drop-prefix frame has an empty prefix",
                ));
            }
            if op == OP_DROPPED_PREFIX {
                VersionEdit::DroppedPrefix(prefix)
            } else {
                VersionEdit::UndroppedPrefix(prefix)
            }
        }
        _ => {
            return Err(EngineError::CorruptMessage(format!(
                "unknown manifest operation tag {op:#04x}"
            )));
        }
    };
    // CRC covers cursor .. pos (payload end).
    let crc_end = pos;
    let Some(crc_bytes) = buf.get(crc_end..crc_end + 4) else {
        return Ok(None); // torn CRC
    };
    let expected = u32::from_le_bytes(match crc_bytes.try_into() {
        Ok(arr) => arr,
        Err(_) => return Ok(None),
    });
    let actual = crc32c_with_state(
        buf.get(cursor..crc_end)
            .ok_or_else(|| EngineError::Corrupt("manifest: crc range out of bounds"))?,
        0,
    );
    if expected != actual {
        return Ok(None); // torn write or corruption → truncate
    }
    Ok(Some((edit, crc_end + 4)))
}

/// Reads a `u64` at `pos`; returns `(value, next_pos)`.
fn take_u64(buf: &[u8], pos: usize) -> Result<(u64, usize)> {
    let bytes = buf
        .get(pos..pos + 8)
        .ok_or_else(|| EngineError::Corrupt("manifest: truncated u64"))?;
    let arr: [u8; 8] = bytes
        .try_into()
        .map_err(|_| EngineError::Corrupt("manifest: truncated u64"))?;
    Ok((u64::from_le_bytes(arr), pos + 8))
}

/// Reads a `u32` at `pos`; returns `(value, next_pos)`.
fn take_u32(buf: &[u8], pos: usize) -> Result<(u32, usize)> {
    let bytes = buf
        .get(pos..pos + 4)
        .ok_or_else(|| EngineError::Corrupt("manifest: truncated u32"))?;
    let arr: [u8; 4] = bytes
        .try_into()
        .map_err(|_| EngineError::Corrupt("manifest: truncated u32"))?;
    Ok((u32::from_le_bytes(arr), pos + 4))
}

/// Reads a `u32` length + payload at `pos`; returns `(payload, next_pos)`.
fn take_len_prefixed(buf: &[u8], pos: usize) -> Result<(Vec<u8>, usize)> {
    let (len, pos1) = take_u32(buf, pos)?;
    let len = len as usize;
    let bytes = buf
        .get(pos1..pos1 + len)
        .ok_or_else(|| EngineError::Corrupt("manifest: truncated payload"))?;
    Ok((bytes.to_vec(), pos1 + len))
}

/// Decodes every complete frame. A torn final frame is ignored; corruption in
/// a complete frame is an error.
fn decode_all(buf: &[u8]) -> Result<Vec<VersionEdit>> {
    let mut cursor = 0usize;
    let mut edits = Vec::new();
    while cursor < buf.len() {
        match decode_frame(buf, cursor)? {
            Some((edit, next)) => {
                edits.push(edit);
                cursor = next;
            }
            None => break,
        }
    }
    Ok(edits)
}

/// Public wrapper for fuzz testing: arbitrary bytes → decode_all, never panics.
#[doc(hidden)]
pub fn decode_all_for_fuzz(buf: &[u8]) -> crate::Result<Vec<VersionEdit>> {
    decode_all(buf)
}

/// Read every complete frame from a manifest file. A non-existent or empty
/// manifest yields an empty edit log.
fn read_manifest_file(path: &Path) -> Result<Vec<VersionEdit>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    fault_point!("manifest::recover_read");
    let bytes = std::fs::read(path)?;
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    // Fail loudly on an unsupported legacy file rather than silently treating
    // it as empty (which would drop every committed SST reference).
    if !bytes.starts_with(&MAGIC) {
        return Err(EngineError::CorruptMessage(
            "MANIFEST uses an unsupported format: only the binary MAN1 format \
             is accepted (legacy Arrow-IPC manifests are no longer supported)"
                .to_string(),
        ));
    }
    decode_all(&bytes)
}

/// Opens the manifest file for append (creating it if absent).
fn open_append(path: &Path) -> Result<File> {
    OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(path)
        .map_err(EngineError::from)
}

/// fsync the parent directory so a rename (or create) of `path` is durable.
///
/// On Unix a directory can be opened and `sync_all`-ed; on Windows opening a
/// directory handle fails, and the OS journals directory metadata itself, so
/// this is a no-op there.
#[cfg(not(windows))]
fn sync_parent_dir(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    File::open(parent)?.sync_all()?;
    Ok(())
}

#[cfg(windows)]
fn sync_parent_dir(_path: &Path) -> Result<()> {
    Ok(())
}

// The byte-level decoding above is bounds-checked via `.get(..)` and explicit
// `Ok(None)` on truncation; keep it that way (unlike the WAL's hot path, the
// manifest is not performance-critical, so safety-by-construction wins over
// raw pointer indexing).

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn add_file(file_id: u64, level: u32) -> VersionEdit {
        VersionEdit::AddFile {
            file_id,
            level,
            min_key: Bytes::from_static(b"a"),
            max_key: Bytes::from_static(b"z"),
            bytes: 10,
            created_at: 100,
        }
    }

    #[test]
    fn round_trip_add_and_next_file_id() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("MANIFEST");
        {
            let mut manifest = Manifest::open(&path).unwrap();
            manifest
                .append_batch(vec![add_file(1, 0), VersionEdit::NextFileId(2)])
                .unwrap();
        }
        let manifest = Manifest::open(&path).unwrap();
        let vs = manifest.version_set();
        assert_eq!(vs.files(0), vec![1]);
        assert_eq!(vs.next_file_id(), 2);
    }

    #[test]
    fn dropped_prefix_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("MANIFEST");
        {
            let mut manifest = Manifest::open(&path).unwrap();
            manifest
                .append_batch(vec![
                    VersionEdit::DroppedPrefix(b"users\0".to_vec()),
                    VersionEdit::UndroppedPrefix(b"users\0".to_vec()),
                ])
                .unwrap();
        }
        let manifest = Manifest::open(&path).unwrap();
        assert!(manifest.version_set().dropped_prefixes().is_empty());
    }

    #[test]
    fn torn_tail_is_truncated() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("MANIFEST");
        {
            let mut manifest = Manifest::open(&path).unwrap();
            manifest.append_batch(vec![add_file(1, 0)]).unwrap();
        }
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.pop(); // lop one CRC byte → torn frame
        std::fs::write(&path, &bytes).unwrap();
        // Reopening must not error and must yield the still-complete prefix.
        let manifest = Manifest::open(&path).unwrap();
        assert!(manifest.version_set().files(0).is_empty());
    }

    #[test]
    fn snapshot_rewrite_bounds_the_log() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("MANIFEST");
        {
            let mut manifest = Manifest::open(&path).unwrap();
            // Append past the threshold: the log must be rewritten to a
            // compact "current state" snapshot (much shorter than the append
            // history).
            for i in 0..(SNAPSHOT_THRESHOLD + 10) {
                manifest
                    .append_batch(vec![VersionEdit::NextFileId(i as u64 + 1)])
                    .unwrap();
            }
        }
        let manifest = Manifest::open(&path).unwrap();
        assert_eq!(
            manifest.version_set().next_file_id(),
            (SNAPSHOT_THRESHOLD + 10) as u64
        );
        // The rewritten file is a snapshot of live state, not the full history.
        let size = std::fs::metadata(&path).unwrap().len();
        assert!(size < 8 * 1024, "snapshot must be compact, got {size} bytes");
    }

    #[test]
    fn rejects_unsupported_format() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("MANIFEST");
        std::fs::write(&path, b"ARROW_IPC_LEGACY").unwrap();
        let err = Manifest::open(&path).unwrap_err();
        assert!(matches!(err, EngineError::CorruptMessage(_)));
    }
}
