# HatP — 用1 万行代码实现 HTAP

HatP 是一个嵌入式 HTAP（混合事务/分析处理）数据库引擎。它可以在一个进程中同时提供 OLTP（MVCC + WAL + LSM-Tree）和 OLAP（SQL + 列式扫描），并且保证性能。只用了10000行代码！加上注释、测试、fuzz、形式化验证等基础设施，全项目约 23,000 行。

```
                  ┌──────────────────────────────────────────┐
                  │                  hatp                    │
                  │     Database / Transaction Facade        │
                  │  put / get / delete / scan / execute_sql │
                  └────────┬──────────────────┬──────────────┘
                           │                  │
              ┌────────────▼──────┐  ┌────────▼──────────────┐
              │  hatp-frontend    │  │     hatp-tx            │
              │  DataFusion       │  │  SSI Transaction Mgr   │
              │  Catalog / DDL    │  │  Cahill Cycle Detection│
              │  TableProvider    │  │  First-committer-wins  │
              └────────┬──────────┘  └────────┬──────────────┘
                       │                      │
                       └──────────┬───────────┘
                                  │
                       ┌──────────▼──────────┐
                       │    hatp-engine       │
                       │  MVCC KV + WAL       │
                       │  LSM-Tree + Vortex   │
                       │  Compaction / Bloom  │
                       └──────────┬──────────┘
                                  │
                       ┌──────────▼──────────┐
                       │    hatp-types        │
                       │  TxnTs / Codec / Hash│
                       └─────────────────────┘
```

## 我的代码行数为啥这么少

### 复用了成熟生态

DataFusion（SQL 解析、优化器、执行引擎）

Vortex（列式编码、谓词下推、zone map）

Arrow（内存列式格式、IPC）

crossbeam-skiplist（并发 SkipMap）

rayon（数据并行 compaction）

tokio（异步运行时）

memmap2（零拷贝 I/O）

### 其余则是我需要做到极致的地方

MVCC 版本链 + SSI 冲突检测

WAL / MANIFEST 自定义二进制格式

LSM-Tree（flush / compaction）

MemTable（SkipMap lock-free）

Bloom + KeyIndex 侧边索引

CRC32C（SSE4.2 硬件加速）

SILK 反压 + 故障注入

### 为什么oltp侧不也同样复用成熟生态呢

曾尝试过，[很难对齐😂，在做了各种花活之后放弃，决定做一个更工程化的](https://github.com/qianzii2/rockduck)。所以 OLTP 侧选择了自己实现。一是 DataFusion 和 Vortex 的发展势头很好，OLTP 自己写能更紧密地对接它们，避免中间层越做越厚。二是不用写 SQL，纯 KV 层的工程量可控。

## 我是这样做到的

对于同一份数据，既要支持高频点查（OLTP），又要支持全表扫描分析（OLAP）。

### 写路径（OLTP）

```
INSERT/UPDATE/DELETE
  → WAL 追加 + fsync（group commit 批量持久化）
  → MemTable 写入（SkipMap，lock-free 读）
  → 达到阈值 → flush 为 Vortex 列式 SST
  → 后台 LSM-Tree compaction 合并 SST
```

### 读路径分两条

点查（OLTP）：

```
get(key) → MemTable 查 → 命中？返回
         → 未命中 → Bloom 过滤器（跳过不包含该 key 的 SST）
                  → KeyIndex 二分查找（O(log n)，不打开 Vortex 文件）
                  → Vortex 列式读取（最后手段）
```

分析查询（OLAP）：

```
SELECT ... WHERE ...
  → DataFusion 解析 SQL + 优化
  → TableProviderAdapter 将查询下推到引擎
  → Vortex→Arrow 直接路径：业务列从 Vortex 文件零拷贝读入 Arrow
  → DataFusion 执行向量化过滤、聚合、JOIN
```

### 那如何做到 OLTP 和 OLAP 不打架

- 写路径持有 `write_guard`（ReentrantMutex），保证写入串行化
- 读路径完全不持锁：读 MemTable 通过 `ArcSwap` 拿到一致快照，读 SST 通过 mmap 零拷贝
- Compaction 有独立锁（`compaction_guard`），不阻塞写入
- SILK 反压：当 MemTable 接近写满，写入自动降速，给 flush 争取时间

## 我在OLTP 侧的设计

### MVCC 版本链

- `SmallVec<[VersionedValue; 4]>`：99% 的 key 只有 1-2 个版本，4 个以内不堆分配
- `partition_point` 二分插入：O(log n) 替代 O(n) 排序
- `commit_ts` 与 `tx_id` 解耦：提交顺序 ≠ 预约顺序，避免 snapshot 看到未提交数据

### SSI 冲突检测

- 先提交者胜（first-committer-wins）：两个并发事务写同一 key，`write_guard` 串行化预提交，先提交的进 `commit_history`，后提交的 `validate_ssi` 检测到冲突
- 读写反依赖（read-write antidependency）：检测写偏斜（write-skew）异常
- Cahill 环检测：T1 读 X 写 Y，T2 读 Y 写 X → 至少一个被中止
- 范围读冲突：range scan 后，并发插入的新 key 被检测为 phantom
- Group commit 暂存冲突：同一 WAL 批次内的冲突事务也能检测

### WAL（自定义二进制格式）

- WAL1 格式：每帧 28 字节固定开销（magic 4 + tx_id 8 + op 1 + key_len 4 + value_len 4 + CRC32C 4）
- 对比 Arrow IPC 的 200+ 字节/帧，单行写入（~80 字节 payload）有效载荷占比从 ~25% 提升到 ~74%
- 恢复时自动截断撕裂的尾部帧

### LSM-Tree Compaction

- SILK 风格优先级调度（MinOverlappingRatio）
- age-based tie-breaking（`created_at` 时间戳）
- 并行 sub-compaction（rayon 数据并行）
- watermark 保护：被活跃快照引用的版本不会被回收

### MemTable

- 默认 `crossbeam_skiplist::SkipMap` + `ArcSwap`：写入 clone 链后原子交换指针，读完全不持锁
- 向后兼容 `RwLock<BTreeMap>` 后端（proptest 验证两后端等价）

## 我的工程化工作

### 正确性验证

| 层级 | 工具 | 覆盖范围 |
|------|------|---------|
| 属性测试 | proptest（5,000 用例） | VersionChain 最新优先、BTreeMap≡SkipMap 等价 |
| Fuzz 测试 | 6 个 libfuzzer 目标 | 每个二进制解析入口：WAL、Bloom、KeyIndex、Manifest、RowCodec、Vortex SST |
| 形式化验证 | 9 个 Kani proof | CRC32C 单比特翻转检测、WAL 编解码等价、sign_flip_be 保序、float 全序 |
| 并发测试 | 6 个 loom 测试 | SkipMap RCU 无丢失更新、group commit、Watcher 单调性 |
| 确定性压力 | stress_runner（种子可复现） | Engine vs MirrorStore 状态一致性 |
| 运行时检测 | Miri / TSan / ASan / cargo-careful | CI 自动化 UB / 数据竞争 / 内存错误 |

### 崩溃恢复

撕裂 WAL 尾部、WAL 帧 CRC 损坏、崩溃在 flush 中间、崩溃在 compaction 中间、WAL+SST 混合恢复、空 WAL 首次启动、SST 文件损坏后恢复——每个场景都有对应的集成测试。

### 故障注入

SST 文件被外部删除、SST 内容被损坏、WAL 文件被删除、WAL 被截断为 0 字节、WAL 被覆盖为随机垃圾、MANIFEST 文件被删除、MANIFEST 被损坏、flush 磁盘满、compaction 输入 SST 损坏。

### 纪律

- workspace 级 `unsafe_code = "forbid"`（engine 按需 allow 并注释 SAFETY 前提）
- `unwrap_used = "deny"`、`expect_used = "deny"`、`panic = "deny"`
- `todo = "deny"`、`unimplemented = "deny"`
- 每个 `unsafe` 块都有 `// SAFETY:` 注释说明边界条件
- `cargo deny check` 审计供应链漏洞和许可证

## 快速开始

```rust
use hatp::Database;
use bytes::Bytes;

// 打开或创建数据库
let db = Database::open("/tmp/mydb")?;

// OLTP: 自动提交写入
db.put(Bytes::from_static(b"key"), Bytes::from_static(b"value"))?;
let value = db.get(b"key")?;

// OLTP: 事务
let mut tx = db.begin();
tx.put(Bytes::from_static(b"a"), Bytes::from_static(b"1"));
tx.put(Bytes::from_static(b"b"), Bytes::from_static(b"2"));
tx.commit()?;

// SSI 可串行化事务
let db = Database::open_with_tx_manager(path, manager)?;
let mut tx = db.begin_ssi();
tx.put(b"shared", b"value");
match tx.try_commit() {
    Ok(commit_ts) => { /* 成功 */ }
    Err((tx, DatabaseError::SsiConflict{..})) => { /* 重试 */ }
}

// OLAP: SQL 查询
let outcome = db.execute_sql("SELECT * FROM my_table WHERE age > 30").await?;
println!("{} rows returned", outcome.rows);
```

## 构建与测试

```bash
cargo build --release
cargo test --profile test
cargo test -p hatp-engine                     # 引擎测试
cargo +nightly fuzz run fuzz_wal_decode       # Fuzz 测试
cargo kani --package hatp-engine --harness kani_crc32c_detects_single_bit_flip  # 形式化验证
cargo bench -p hatp-engine                    # 基准测试
```

## Crate 组织

| Crate | 角色 | 依赖 |
|-------|------|------|
| hatp-types | 共享类型、编解码、哈希 | 仅 bytes + arrow + DataFusion ScalarValue |
| hatp-engine | 存储引擎：MVCC + WAL + LSM + Vortex SST | 无 DataFusion 依赖 |
| hatp-tx | 事务层：SSI 状态机 | 仅 engine + types |
| hatp-frontend | SQL 前端：DataFusion + Catalog + DDL/DML | 不直接接触 WAL/SST |
| hatp | 顶层门面：Database / Transaction | 纯胶水层 |

## 目前

- 仅支持单机嵌入运行（无网络层）
- Vortex 0.83 的列裁剪在 `scan()` 路径上尚未完全实现（依赖上游升级）

## License

Apache-2.0