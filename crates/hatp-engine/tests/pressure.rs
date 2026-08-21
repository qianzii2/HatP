//! Write-pressure throttling (PR 2.7) integration tests.

use hatp_engine::metrics::EngineMetrics;
use hatp_engine::pressure::{PressureConfig, PressureThrottle, ThrottleLevel};

#[test]
fn pressure_throttles_when_memtable_grows() {
    let config = PressureConfig {
        soft_memtable_ratio: 0.7,
        hard_memtable_ratio: 0.95,
        memtable_flush_bytes: 1_000,
        soft_sleep: std::time::Duration::from_millis(1),
    };
    let throttle = PressureThrottle::new(config);
    let metrics = EngineMetrics::new();

    // Empty memtable → no throttling.
    assert_eq!(
        throttle.should_throttle(&metrics.snapshot()),
        ThrottleLevel::None
    );

    // 800 / 1000 = 0.8 → soft throttling.
    metrics.set_memtable_bytes(800);
    assert_eq!(
        throttle.should_throttle(&metrics.snapshot()),
        ThrottleLevel::Soft
    );

    // 980 / 1000 = 0.98 → hard throttling (reject).
    metrics.set_memtable_bytes(980);
    assert_eq!(
        throttle.should_throttle(&metrics.snapshot()),
        ThrottleLevel::Hard
    );

    // Drop back → no throttling.
    metrics.set_memtable_bytes(100);
    assert_eq!(
        throttle.should_throttle(&metrics.snapshot()),
        ThrottleLevel::None
    );
}

#[test]
fn hard_throttle_rejects_write_without_touching_storage() {
    use bytes::Bytes;
    use hatp_engine::{Engine, EngineConfig, Mutation};

    let dir = tempfile::tempdir().unwrap();
    let mut config = EngineConfig::new(dir.path());
    config.memtable_flush_bytes = 1_024;
    config.pressure = PressureConfig {
        soft_memtable_ratio: 0.7,
        hard_memtable_ratio: 0.0, // any non-empty memtable is Hard
        memtable_flush_bytes: 1_024,
        soft_sleep: std::time::Duration::from_millis(1),
    };
    let engine = Engine::open(config).unwrap();

    let err = engine
        .write(&[Mutation::Put {
            key: Bytes::from_static(b"k"),
            value: Bytes::from_static(b"v"),
        }])
        .unwrap_err();
    assert!(matches!(err, hatp_engine::EngineError::Throttled));
    // Rejected writes must not produce any SST/WAL data.
    assert!(engine.sst_file_ids().is_empty());
    // Negative assertion: memtable must be empty (write was rejected, no data residue)
    assert!(engine.memtable().is_empty(), "rejected write must not leave data in memtable");
    // Negative assertion: snapshot_ts must not advance (write was rejected, must not consume commit seq)
    assert_eq!(engine.snapshot_ts(), 0, "rejected write must not advance snapshot_ts");
}