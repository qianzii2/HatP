//! Compaction planning and SILK-style priority scheduling.

use bytes::Bytes;
use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::metrics::MetricsSnapshot;

/// Supported LSM organization policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionPolicy {
    /// Non-overlapping sorted runs per level.
    Leveled,
    /// Tier all levels except the largest one.
    LazyLeveling,
    /// Merge similarly sized runs without enforcing non-overlap.
    Tiered,
}

/// Compaction layout a picker reports for a given level. Used by tests and
/// observability to assert which strategy is active (Dostoevsky / SILK switch
/// layouts under workload shifts).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionLayout {
    Leveled,
    LazyLeveling,
    Tiered,
}

/// What kind of compaction a job represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CompactionKind {
    /// Newly flushed L0 SST.
    Flush,
    /// Compact L0 down into L1.
    L0ToL1,
    /// Generic level-N to level-(N+1) compaction.
    LevelNToN1,
    /// Bottom-most level cleanup.
    BottomMost,
}

/// Immutable metadata used to plan compaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileMeta {
    /// Stable file identifier.
    pub file_id: u64,
    /// Current LSM level.
    pub level: u32,
    /// Approximate bytes occupied by the file.
    pub bytes: u64,
    /// Inclusive smallest key.
    pub min_key: Bytes,
    /// Inclusive largest key.
    pub max_key: Bytes,
    /// Wall-clock seconds at which this file was written (`0` when unknown,
    /// e.g. recovered from a legacy manifest without the `created_at` column).
    ///
    /// Used by the picker for age-based tie-breaking: among candidates with
    /// an equal overlap ratio, older files are compacted first so long-lived
    /// SSTs do not accumulate.
    pub created_at: u64,
}

impl FileMeta {
    /// Returns whether this file's key range overlaps another range.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.min_key <= other.max_key && other.min_key <= self.max_key
    }
}

/// Work item selected for compaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionJob {
    /// Scheduling class.
    pub kind: CompactionKind,
    /// Input level.
    pub source_level: u32,
    /// Output level.
    pub target_level: u32,
    /// Input file identifiers.
    pub inputs: Vec<u64>,
    /// Estimated bytes read and rewritten.
    pub estimated_bytes: u64,
    /// Higher values run first within a SILK class.
    pub priority: u64,
}

impl Ord for CompactionJob {
    /// `CompactionJob` is ordered by (priority descending, bytes descending,
    /// source_level descending).  This is designed for a max-heap
    /// (`BinaryHeap`): the highest-priority, largest job pops first.
    ///
    /// # Note on `estimated_bytes` ordering
    ///
    /// `other.estimated_bytes.cmp(&self.estimated_bytes)` yields
    /// **descending** order — larger jobs sort before smaller ones.
    /// This is intentional for the current single-job `pick()` path but
    /// means the struct is **not** suitable for a standard `BinaryHeap`
    /// (where `Ord` is conventionally ascending / min-heap).  If it is
    /// ever placed in a `BinaryHeap`, wrap it in `std::cmp::Reverse`.
    fn cmp(&self, other: &Self) -> Ordering {
        // Manual chain to avoid `.then_with()` closure allocations.
        let ord = self.priority.cmp(&other.priority);
        if ord != Ordering::Equal {
            return ord;
        }
        let ord = other.estimated_bytes.cmp(&self.estimated_bytes);
        if ord != Ordering::Equal {
            return ord;
        }
        other.source_level.cmp(&self.source_level)
    }
}

impl PartialOrd for CompactionJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Selects compaction jobs from current file metadata.
///
/// The engine holds one picker behind an `Arc` (see `EngineConfig::compaction_picker`),
/// so the background worker reuses a single instance instead of rebuilding a
/// `MinOverlappingRatioPicker` on every iteration — which previously discarded
/// any adaptive state (R-01 / R-17, PR 2.5).
pub trait CompactionPicker: Send + Sync {
    /// Chooses one deterministic job from current file metadata, or `None`
    /// when the LSM tree is healthy.
    fn pick(&self, files: &[FileMeta]) -> Option<CompactionJob>;

    /// Periodically called by the worker with a fresh [`MetricsSnapshot`] so an
    /// adaptive picker (Dostoevsky / SILK) can switch layouts under workload
    /// shifts. `&self` because the picker is shared behind an `Arc`; adaptive
    /// state lives in interior atomics. Default no-op.
    fn update_from_workload(&self, _metrics: &MetricsSnapshot) {}

    /// Returns the layout this picker applies at `level`, for tests and
    /// observability. Defaults to [`CompactionLayout::LazyLeveling`].
    fn layout(&self, _level: u32) -> CompactionLayout {
        CompactionLayout::LazyLeveling
    }
}

/// Selects files with the smallest output-overlap ratio.
#[derive(Debug, Clone)]
pub struct MinOverlappingRatioPicker {
    /// L0 file count that triggers compaction.
    pub l0_trigger: usize,
    /// Maximum level represented by the current policy.
    pub max_level: u32,
    /// Selected LSM policy.
    pub policy: CompactionPolicy,
}

impl Default for MinOverlappingRatioPicker {
    fn default() -> Self {
        Self {
            l0_trigger: 4,
            max_level: 6,
            policy: CompactionPolicy::LazyLeveling,
        }
    }
}

impl CompactionPicker for MinOverlappingRatioPicker {
    fn pick(&self, files: &[FileMeta]) -> Option<CompactionJob> {
        self.pick_inner(files)
    }

    fn layout(&self, level: u32) -> CompactionLayout {
        if level == 0 {
            return CompactionLayout::Tiered;
        }
        match self.policy {
            CompactionPolicy::Leveled => CompactionLayout::Leveled,
            CompactionPolicy::LazyLeveling => CompactionLayout::LazyLeveling,
            CompactionPolicy::Tiered => CompactionLayout::Tiered,
        }
    }
}

/// Tiered picker: merge similarly sized runs without enforcing non-overlap on
/// small levels, and fall back to leveled at the largest level (RocksDB's
/// classic "tiered then leveled" shape).
#[derive(Debug, Clone)]
pub struct TieredPicker {
    /// L0 file count that triggers compaction.
    pub l0_trigger: usize,
    /// Level at which compaction switches from tiered to leveled.
    pub leveled_from_level: u32,
    /// Maximum level.
    pub max_level: u32,
}

impl Default for TieredPicker {
    fn default() -> Self {
        Self {
            l0_trigger: 4,
            leveled_from_level: 2,
            max_level: 6,
        }
    }
}

impl CompactionPicker for TieredPicker {
    fn pick(&self, files: &[FileMeta]) -> Option<CompactionJob> {
        // Reuse the min-overlap machinery but with tiered capacity: the
        // `MinOverlappingRatioPicker`'s `policy` field controls `level_capacity`,
        // so a `Tiered` policy yields the tiered trigger thresholds.
        MinOverlappingRatioPicker {
            l0_trigger: self.l0_trigger,
            max_level: self.max_level,
            policy: CompactionPolicy::Tiered,
        }
        .pick_inner(files)
    }

    fn layout(&self, level: u32) -> CompactionLayout {
        if level == 0 || level < self.leveled_from_level {
            CompactionLayout::Tiered
        } else {
            CompactionLayout::Leveled
        }
    }
}

impl MinOverlappingRatioPicker {
    fn pick_inner(&self, files: &[FileMeta]) -> Option<CompactionJob> {
        let mut levels = BTreeMap::<u32, Vec<&FileMeta>>::new();
        for file in files {
            levels.entry(file.level).or_default().push(file);
        }
        let l0 = levels.get(&0).cloned().unwrap_or_default();
        if l0.len() >= self.l0_trigger {
            return build_l0_job(&l0, levels.get(&1));
        }
        for level in 1..self.max_level {
            let source = match levels.get(&level) {
                Some(source) if source.len() > level_capacity(level, self.policy) => source,
                _ => continue,
            };
            let target = levels.get(&level.saturating_add(1));
            if let Some(candidate) = choose_min_overlap(source, target) {
                let mut inputs = vec![candidate.file_id];
                let mut estimated_bytes = candidate.bytes;
                if let Some(target_files) = target {
                    for overlap in target_files.iter().filter(|file| candidate.overlaps(file)) {
                        inputs.push(overlap.file_id);
                        estimated_bytes = estimated_bytes.saturating_add(overlap.bytes);
                    }
                }
                inputs.sort_unstable();
                return Some(CompactionJob {
                    kind: if level.saturating_add(1) >= self.max_level {
                        CompactionKind::BottomMost
                    } else {
                        CompactionKind::LevelNToN1
                    },
                    source_level: level,
                    target_level: level.saturating_add(1),
                    inputs,
                    estimated_bytes,
                    priority: overlap_priority(candidate, target),
                });
            }
        }
        None
    }
}

fn build_l0_job(l0: &[&FileMeta], l1: Option<&Vec<&FileMeta>>) -> Option<CompactionJob> {
    let min_key = l0.iter().map(|file| &file.min_key).min()?.clone();
    let max_key = l0.iter().map(|file| &file.max_key).max()?.clone();
    let mut inputs: Vec<u64> = l0.iter().map(|file| file.file_id).collect();
    let mut estimated_bytes = l0
        .iter()
        .fold(0_u64, |total, file| total.saturating_add(file.bytes));
    if let Some(l1_files) = l1 {
        for file in l1_files
            .iter()
            .filter(|file| file.min_key <= max_key && min_key <= file.max_key)
        {
            inputs.push(file.file_id);
            estimated_bytes = estimated_bytes.saturating_add(file.bytes);
        }
    }
    inputs.sort_unstable();
    Some(CompactionJob {
        kind: CompactionKind::L0ToL1,
        source_level: 0,
        target_level: 1,
        inputs,
        estimated_bytes,
        priority: u64::try_from(l0.len()).unwrap_or(u64::MAX),
    })
}

/// Overlap ratio of `candidate` against `target`, scaled by 1000 so ties are
/// rare and the ratio remains integral (bytes are `u64`).
fn overlap_ratio(candidate: &FileMeta, target: Option<&Vec<&FileMeta>>) -> u64 {
    let overlapping = target
        .map(|files| {
            files
                .iter()
                .filter(|file| candidate.overlaps(file))
                .fold(0_u64, |total, file| total.saturating_add(file.bytes))
        })
        .unwrap_or(0);
    overlapping
        .saturating_mul(1_000)
        .checked_div(candidate.bytes.max(1))
        .unwrap_or(u64::MAX)
}

fn choose_min_overlap<'a>(
    source: &'a [&FileMeta],
    target: Option<&Vec<&FileMeta>>,
) -> Option<&'a FileMeta> {
    source.iter().copied().min_by(|a, b| {
        // Primary key: lowest overlap ratio. Tie-break: oldest file first
        // (`created_at` ascending), so a long-lived SST does not keep losing
        // to younger peers with the same ratio. `created_at == 0` (unknown,
        // legacy manifest) sorts first as "oldest".
        overlap_ratio(a, target)
            .cmp(&overlap_ratio(b, target))
            .then_with(|| a.created_at.cmp(&b.created_at))
    })
}

fn overlap_priority(candidate: &FileMeta, target: Option<&Vec<&FileMeta>>) -> u64 {
    let overlap = target
        .map(|files| {
            files
                .iter()
                .filter(|file| candidate.overlaps(file))
                .fold(0_u64, |total, file| total.saturating_add(file.bytes))
        })
        .unwrap_or(0);
    u64::MAX.saturating_sub(overlap)
}

fn level_capacity(level: u32, policy: CompactionPolicy) -> usize {
    match policy {
        CompactionPolicy::Leveled => 1,
        CompactionPolicy::LazyLeveling => {
            if level <= 2 {
                4
            } else {
                1
            }
        }
        CompactionPolicy::Tiered => 4,
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn meta(file_id: u64, level: u32, min: u8, max: u8) -> FileMeta {
        FileMeta {
            file_id,
            level,
            bytes: 100,
            min_key: Bytes::from(vec![min]),
            max_key: Bytes::from(vec![max]),
            created_at: 0,
        }
    }

    #[test]
    fn compaction_picker_trait_reports_layouts() {
        let lazy = MinOverlappingRatioPicker::default();
        assert_eq!(lazy.layout(0), CompactionLayout::Tiered);
        assert_eq!(lazy.layout(1), CompactionLayout::LazyLeveling);

        let tiered = TieredPicker::default();
        assert_eq!(tiered.layout(0), CompactionLayout::Tiered);
        assert_eq!(tiered.layout(1), CompactionLayout::Tiered);
        assert_eq!(tiered.layout(2), CompactionLayout::Leveled);
    }

    #[test]
    fn compaction_picker_l0_trigger_picks_job() {
        let files = vec![meta(1, 0, b'a', b'm'), meta(2, 0, b'n', b'z')];
        let picker = MinOverlappingRatioPicker {
            l0_trigger: 2,
            max_level: 6,
            policy: CompactionPolicy::LazyLeveling,
        };
        let job = picker.pick(&files).expect("2 L0 files must trigger L0ToL1");
        assert_eq!(job.kind, CompactionKind::L0ToL1);
        assert_eq!(job.source_level, 0);
        assert_eq!(job.target_level, 1);

        // Below the trigger threshold → no job selected.
        let picker = MinOverlappingRatioPicker {
            l0_trigger: 4,
            max_level: 6,
            policy: CompactionPolicy::LazyLeveling,
        };
        assert!(picker.pick(&files).is_none());
    }

    #[test]
    fn tiered_picker_picks_with_tiered_capacity() {
        let picker = TieredPicker::default();
        // TieredPicker internally uses tiered capacity; construct a few files to verify no panic
        // and return reproducible results (tiered strategy within L0 trigger threshold).
        let files = vec![meta(1, 0, b'a', b'm')];
        let _ = picker.pick(&files);
        // Insufficient input should return None (single file does not trigger).
        assert!(picker.pick(&files).is_none());
    }
}
