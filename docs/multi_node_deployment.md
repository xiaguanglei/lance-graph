# lance-graph 多节点后端部署指南

本文面向把 lance-graph 嵌入后端服务、并部署为多个节点的团队，说明哪些事情引擎已经保证、
哪些必须由服务端完成，以及推荐的读写协议。文中的接口均为 `lance_graph` crate 的公开 API。

## 1. 一致性模型：引擎保证什么

| 层面 | 由谁负责 | 说明 |
| --- | --- | --- |
| 单个已提交版本的数据 | Lance | 版本一旦提交不可变；同一版本在任何节点读到的数据完全相同。 |
| CSR 邻接/节点 ID 缓存 | 引擎 | 只缓存“某个已提交版本”的拓扑与行号，键是版本的内容标识（URI、分支、版本号、提交时间戳、事务文件）。缓存是版本的纯函数，**节点之间不需要同步或失效缓存**，也不会读到过期拓扑。新版本从已缓存版本增量推导，结果与全量重建逐行一致（有随机化测试覆盖）。 |
| 原生路径与 JOIN 计划等价 | 引擎 | 两条路径返回相同的结果集合（含重复次数）。放不下、超预算或不支持的形状自动走 JOIN，结果不变。 |
| 一次查询内读同一组版本 | 引擎（`GraphSnapshot`） | 通过 `GraphSnapshotProvider` 获取的快照把所有表固定在一组版本上。 |
| 行顺序确定 | 引擎（需开启） | `with_deterministic_order(true)` 让行顺序只由结果本身决定，各节点、各执行路径一致，分页不重不漏。 |
| 各节点读到哪组版本、多久刷新 | **服务端** | 见第 3、4 节。 |
| 跨表写入的原子性与顺序 | **服务端** | Lance 没有跨表事务，见第 3 节。 |
| 存储提交原子性、并发写冲突重试 | **服务端 + 存储** | 见第 5 节。 |
| 旧版本清理、compaction 调度 | **服务端** | 见第 6 节。 |

## 2. 每个节点的初始化

每个服务节点（以及每个安全上下文，例如每个租户）创建**一个**长期存在的
`GraphSnapshotProvider`，所有请求共用。不要每个请求新建：`CypherQuery` 默认的缓存只在单次查询内有效，
每次都会冷构建。

```rust
use lance_graph::{CsrAdjacencyCache, CsrScanOptions, GraphSnapshotProvider};
use std::{sync::Arc, time::Duration};

let cache = Arc::new(CsrAdjacencyCache::new(CsrScanOptions {
    max_edges: 5_000_000,        // 单张边表的最大行数
    max_bytes: 512 << 20,        // 单次冷构建峰值，约 64 B/边 + 顶点
    max_cache_bytes: 2 << 30,    // 本节点所有缓存版本，约 32 B/边 + 顶点
    max_cache_rows: 20_000_000,
    max_cache_slots: 32,
    max_column_cache_bytes: 512 << 20, // 属性列缓存（可选，默认 0 关闭），热查询延迟约减半
    ..Default::default()
}));
let provider = GraphSnapshotProvider::new(namespace, graph_config)
    .with_table("MENTIONS", "graph_edges")   // 多个关系类型共用一张边表时声明
    .with_table("REL", "graph_edges")
    .with_cache(cache)
    .with_max_staleness(Duration::from_millis(500));

// 就绪探针通过前预热，避免第一个用户请求承担冷构建（80 万边约 600 ms）。
let snapshot = provider.latest().await?;
let report = snapshot.warm().await;   // report.skipped > 0 表示有表放不下，会走 JOIN
```

内存估算（每个节点各自一份）：边约 32 B/条常驻，顶点约 160 B/个加上 ID 字符串长度的两倍，
节点 ID 映射约 24 B/行 + 96 B/个不同 ID。多个版本共享同一个底座时只计一次，增量部分很小。
例：80 万边 + 两张 20 万行节点表约 170 MB。

属性列缓存（`max_column_cache_bytes`）：Lance 本身没有数据页缓存，原生遍历每次都要从存储读取
返回列和过滤列。开启后这些列按 fragment、按版本缓存在内存中，未变化的 fragment 在新旧版本间共享，
预算与拓扑缓存独立，放不下时回退为直接读存储（结果不变）。大小约等于被查询列的 Arrow 内存大小
（字符串列 ≈ 总字节数 + 4 B/行）。本地 20 万节点基准中，热查询 p50 从约 2.3–2.8 ms 降到约 1.0–1.6 ms。
可在预热时调用 `cache.warm_columns(dataset, &["memory_id", "content", ...])` 预先加载，
否则在第一次查询时按需加载。

本地延迟基准（生产读路径：provider + `execute_with_snapshot`，并与 JOIN 结果逐行校验）：
`cargo run --release -p lance-graph-benches --example graph_snapshot_latency -- 2000 20000 200000`，
设置 `BENCH_COLUMN_CACHE_MB=512` 开启列缓存。

多租户：一个缓存只能在同一安全上下文内共享。不同租户的数据集不同或权限不同时，按租户分别创建
provider 和 cache。

## 3. 写入协议

1. **先写节点表，再写边表。** 读的一方在两次提交之间可能看到“有节点没边”，这是正常的；
   反过来先写边会出现“边指向不存在的节点”，查询会暂时看不到这条边（不会报错，也不会产生错误数据），
   直到节点写入。
2. **提交后收集版本向量并返回给客户端。** 每次提交得到的 `version_id()` 组成 `GraphVersions`：

   ```rust
   let token = GraphVersions::new()
       .with("Memory", memory_version)
       .with("graph_edges", edge_version)
       .to_token();          // 放进写接口的响应里
   ```

   一次业务操作涉及多次提交时用 `merge_max` 合并。
3. **（可选）整体发布图版本。** 如果需要“读的一方永远只看到完整的业务写入”，在第 1、2 步之后把完整的
   版本向量原子地写入一个“图版本”记录（数据库一行、命名空间元数据或一张小表），读的一方只读这条记录
   指向的版本（`provider.at(&versions)`），而不是各表的最新版本。
4. **写后通知（可选）。** 通过消息总线通知各节点；收到后执行 `provider.refresh()` 和 `snapshot.warm()`，
   新版本在各节点提前增量推导好（约 15–25 ms），用户请求不再承担。

## 4. 读取协议

| 场景 | 调用 | 说明 |
| --- | --- | --- |
| 普通读 | `provider.latest()` | 最多滞后 `max_staleness`；`Duration::ZERO` 表示每次检查新版本（每张表多读一次 manifest）。 |
| 读自己的写（客户端带 token） | `provider.at_least(&GraphVersions::from_token(t)?)` | 任何节点都能看到至少该版本；若存储暂不可见返回错误，服务端应短暂退避重试。之后同一节点不会退回旧版本。 |
| 可复现读 / 分页 / 导出 | `provider.at(&versions)` | 精确读取一组版本。第一页返回 `snapshot.versions().to_token()`，后续页带着它请求。token 必须包含所有表。 |
| 发布了图版本记录 | `provider.at(&published)` | 见第 3 节第 3 步。 |

执行查询：

```rust
let snapshot = provider.at_least(&token).await?;
let result = CypherQuery::new(text)?
    .with_parameters(params)
    .with_deterministic_order(true)       // 对外分页/截断的接口必须开启
    .execute_with_snapshot(&snapshot)
    .await?;
// 把 snapshot.versions().to_token() 一并返回给客户端
```

**分页与 LIMIT：** 没有 `ORDER BY` 或排序键有重复时，不同节点、不同执行路径（原生 CSR / JOIN）返回的
行顺序和 LIMIT 子集可能不同，翻页会重复或漏行。对外接口凡是 `SKIP/LIMIT`，必须开启
`with_deterministic_order(true)`，并且所有页使用同一个版本 token（`provider.at`）。该选项保留用户的
`ORDER BY` 作为主排序，只追加其余列作为平局键；嵌套类型列（结构体、列表、向量）不作为排序键。

## 5. 并发写入与存储

- 多个节点可以同时写同一张表。Lance 采用乐观并发：追加与追加会自动合并（有测试覆盖）；删除、更新、
  compaction 与其他写入可能冲突，需要按 Lance 返回的冲突错误重试。
- **前提是存储层的提交是原子的**（同一版本号只能有一个提交者成功）。HDFS 依赖原子 rename；对象存储需要
  条件写入或外部提交存储。请确认你们内部 Lance 在 HDFS / QBFS / 对象存储上使用的 commit handler
  满足这一点，否则并发写可能互相覆盖。这一点引擎无法替代。
- 同一业务对象的写入建议由同一个写入方串行化（例如按 memory_id 分区），减少冲突重试。

## 6. 版本清理与 compaction

- `cleanup_old_versions(older_than, ..)` 会删除旧版本的数据文件。仍在读旧版本的节点（`latest` 未过期的
  旧快照、正在使用的分页 token）会报错（只会报错，不会返回错误数据，有测试覆盖）。
  **保留时间必须大于：`max_staleness` + 最长查询时间 + 分页 token 的有效期。**
- token 指向的版本已被清理时，`provider.at` 或查询返回错误：服务端应返回“结果已过期，请重新查询”，
  不要静默改用新版本，否则分页会错乱。
- compaction 会重写 fragment，下一个版本的缓存会按需重新构建（超过四分之一变化即全量重建）。
  安排在低峰期，并在完成后对各节点发送写后通知以预热。
- 启用稳定行号（stable row IDs）的数据集每个新版本都会全量重建 CSR，写入频繁时不建议启用。

## 7. 发布、扩缩容与故障

- 新节点（扩容、滚动发布、重启）缓存为空：在就绪探针通过前执行 `latest()` + `warm()`。
- `at_least` 报“版本尚不可见”：多见于存储最终一致或写入刚提交；退避 50–200 ms 重试 2–3 次后返回错误。
- 缓存放不下时查询自动走 JOIN（结果不变，只是变慢），通过监控发现并调大预算。

## 8. 监控

- 节点级：`cache.stats()` —— `entries`、`bytes`、`rows`、`full_builds`、`incremental_builds`、
  `evictions`、`rejected_builds`；开启列缓存时还有 `column_entries`、`column_bytes`。
  建议告警：`rejected_builds` 增长（图超出 `max_edges`/预算，原生路径失效）；`evictions` 持续增长
  （预算不足、同一批数据反复重建）；`full_builds` 在无 compaction 时增长（增量推导没有生效）。
- 查询级：`query.csr_adjacency_metrics()` —— `native_multihop_executions`、`native_multihop_fallbacks`、
  `build_count`、`incremental_builds`、`cache_build_nanos`、`cache_evictions`、`column_cache_reads`。按查询打点，统计原生路径
  命中率和构建耗时分布。
- 记录每个请求使用的 `snapshot.versions()`，便于排查“不同节点结果不同”（版本不同还是逻辑问题）。

## 9. 绑定与接入方式

- Rust 服务：直接使用本文的 API。
- Python 服务：当前 Python 绑定的 `CypherEngine` 会在构造时把数据集整体读入内存表，**不使用 CSR、
  不感知版本、也不会刷新**；`CypherQuery.execute_with_namespace` 每次读各表最新版本且每次新建缓存。
  多节点部署前需要先为 `GraphSnapshotProvider` / `GraphSnapshot` / `GraphVersions` /
  `with_deterministic_order` 增加 Python 绑定（引擎侧工作，尚未完成）。

## 10. 上线检查清单

- [ ] 每个节点、每个安全上下文一个 `GraphSnapshotProvider`，共享缓存，预算按第 2 节估算。
- [ ] 就绪探针前执行 `warm()`，`report.skipped == 0`。
- [ ] 写入顺序：先节点表、后边表；写接口返回版本 token。
- [ ] 读接口支持 token：带 token 用 `at_least`，分页用 `at` + `with_deterministic_order(true)`。
- [ ] 确认存储提交原子性；并发写冲突有重试。
- [ ] 旧版本保留时间 > `max_staleness` + 最长查询 + 分页 token 有效期；token 过期返回明确错误。
- [ ] compaction / 清理在低峰期执行，完成后通知各节点预热。
- [ ] 监控 `cache.stats()` 与查询级指标，配置第 8 节的告警。
