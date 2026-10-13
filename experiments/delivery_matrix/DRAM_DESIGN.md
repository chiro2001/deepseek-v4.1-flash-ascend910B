# CED 单侧 DRAM 与物理共享：实现前验收边界

这是第二阶段设计，**不是实现/容量达标收据**。先完成三形态基线与 README，
再在新候选中实现；冻结 best、DCP 与 CED 基线文件不原地覆盖。

## 必须交付的三个独立条件

1. CED 允许 `off|prefill|decode` 三档，P 与 D 不能同时创建 DRAM KV 池。
   关闭侧保留正常 Mooncake P/D 传输，不能靠删除整个 connector 实现单侧选项。
2. `DEVICE_INDEX=0` 时 P/D Engram host 查表也必须映射相同底层文件/inode；
   不是各进程先 mmap 再 `copy_` 到私有 shard。weight 和 FP32 scale 均共享，
   只允许有界查询 staging，不能出现完整表的私有副本。
3. KV 池内相同内容只能有一份物理存储，别名/引用与数据所有权分离。
   只少建一个 P/D 池并不证明 TP8 去重；相同逻辑页也不自动证明实际字节相同。

## 已确认的来源与待验证风险

- actual best `engram_hbm.py` SHA 为
  `b422e9899204b7a55af05832073bdea41e24cdc14900ea739c1dcd82822c77ae`，
  与当前 tracked 载荷相同。不能用 DCP-derived 文件倒灌基线。
- Engram 的 `NodeShardedEngram` 当前构造私有 CPU weight/scale，
  `load_checkpoint` 再逐片 `copy_`；需要在 opt-in mmap 分支省掉整个私有分配。
  INT8/F32 checkpoint 的形状、dtype、字节范围、读权限须在绑定前验证；
  BF16 动态量化不能未经验证伪装成直接 INT8 mmap。
- V4.1 canonical long-KV 的 cache spec 使用完整 `num_kv_heads=1`，
  runner 不按 TP 切该平面。这支持 replicated 的源码语义，但不是 TP8
  实际逐字节一致证据；draft/state/packed alias 各自仍需验证。
- `ced-dram-offload` @8125757 的 P-only 插件只是参考：其真实 P+mock D
  续算/尾块精度没有通过，不能冒充本阶段验收，也不能把它的旧 `p2_pool`
  覆盖当前较新的配额、池满和过淘汰修复。
- MultiConnector 按顺序选择第一个 positive external hit；D 使用 DRAM
  命中时不能仍发 Mooncake pull 或泄漏 P delayed-free 块。P 命中时需要恢复
  Engram 先前 SWA token mirror 与必要的 compressor/recurrent 状态，不能
  只复制 long-KV 后宣称续算完整。

## 实机验收矩阵

| 条件 | 必须留证 | 不能替代它的证据 |
|---|---|---|
| Engram 相同物理页 | P/D TP8 的 inode/offset、实际查表字节/输出一致、smaps PSS/共享页 | 同文件名、VIRT/RSS 相加或 device 进程占用 |
| 单侧 DRAM | 另一侧零池分配；选中侧真实 D2H 与 H2D、命中后续算正确 | 只打印配置或 CPU/mock consumer 测试 |
| KV 唯一存储 | TP8 内容一致门或内容寻址别名、单 writer 竞态/eviction/refcount、实际分配字节 | shape 相同、按 world_size 除账或 TP1 mmap 测试 |
| 容量收益 | 相同 DRAM 预算、可服务/保留 tokens 与 PSS，预算计入 staging/元数据 | 稀疏文件大小、仅理论 8×或 Engram 节省量冒充 KV 去重 |
| 精度与稳定性 | P/D 两档各自 GSM/Vision/编程/长针、负控、池满/取消/重用 | 基线质量 receipt 继承或单条 clean 文本 |

所有变更均 opt-in，新候选 fingerprint/源码 manifest 单独绑定；失败门保留
完整原件。没有完成上述实机检查就不在 README 标“共享/卸载达标”。
