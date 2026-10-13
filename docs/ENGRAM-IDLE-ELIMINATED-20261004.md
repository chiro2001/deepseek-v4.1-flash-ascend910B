# ★ 设备索引把 engram 空闲洞打掉了 19×：AI core 空闲 10.6% → 5.9%（2026-10-04）

> 承接 `IDLE-GRANULARITY-20261004.md`：我此前用空闲洞的**前驱算子**聚合，
> 发现前三名（占空闲 63%）是 `GatherElementsV2` / `hcom_broadcast_` / `hcom_alltoallv_`
> → 疑似 **Engram host lookup 路径**的签名。
> 本文用**同工具、同脚法**对比"设备索引关"与"设备索引开"两份 profile，给出判决。

## 1. 两份 profile（同实例、同配置，只有 `ENGRAM_DEVICE_INDEX` 不同）

| | A：`=0`（host lookup） | B：`=1`（设备索引） |
|---|---|---|
| run | `k6full_1004_100156` | `final_1004_1815` |
| 稳态窗口 | 12,380.6 ms | 3,845.1 ms |
| **AI core 占空比** | **85.1%** | **91.6%** |
| AICPU 占空比 | 4.6% | 4.4% |
| **AI core 纯空闲** | **10.6%** | **5.9%** |

⇒ 设备索引让 **AI core 占空比 +6.5 个百分点**，纯空闲 **−4.7 个百分点**。
（步时口径：A 的 27.42 ms → B 的 25.26–25.37 ms，与 `[bneck]` 的 −0.8~−1.4 ms/步 自洽。）

## 2. 空闲洞的前驱算子：engram 链路**整体消失**（`scripts/gap_context.py`）

| 前驱算子 | A（`=0`）加权空闲 | B（`=1`）加权空闲 | 变化 |
|---|---:|---:|---|
| **`GatherElementsV2`** | **361.1 ms** | **5.7 ms** | **−63×** |
| **`hcom_broadcast_`** | **287.9 ms** | 跌出 top-10 | — |
| **`hcom_alltoallv_`** | **220.7 ms** | 跌出 top-10 | — |
| `IndexPutV2` | 78.5 ms | 跌出 top-10 | — |
| `ZerosLike` | 75.3 ms | 8.4 ms | −9× |
| `RepeatInterleave` | 85.1 ms | 24.4 ms | −3.5× |

**这是判决性证据**：前驱侧 top-3（`GatherElementsV2` / `hcom_broadcast_` / `hcom_alltoallv_`）
正是 `route-probe` 的 a2a / bcast / scatter 相位 —— **Engram host lookup 路径**。
设备索引把它们整条替换掉后，这些洞**基本归零**。

⇒ **推论**：我在 `IDLE-GRANULARITY` 里说的"减少 kernel 数是唯一方向"要修正为：
**那 63% 的空闲已经被设备索引吃掉了**（这也是它 −3% 步时的机制来源：不是"少算"，
而是"AI core 不再等 host 路径"）。

## 3. 剩余空闲换了主角：稀疏索引小算子链

B 的剩余空闲（5.9%）里，前驱 top：

| 前驱 → 后继 | 加权空闲 | 洞数 | 均值/洞 |
|---|---:|---:|---:|
| `RepeatInterleave → IndexCheck` | 24.4 ms | 149 | 163.7 µs |
| `Add → Range` | 20.2 ms | 148 | 136.5 µs |
| `LogicalOr → Less` | 13.3 ms | 148 | 90.1 µs |
| `RmsNorm → IndexCheck` | 11.7 ms | 149 | 78.5 µs |
| `Cast → Fill` | 10.3 ms | 148 | 69.9 µs |
| `LogicalNot → FloorDiv` | 8.8 ms | 148 | 59.3 µs |
| `ZerosLike → Cast` | 8.4 ms | 148 | 56.9 µs |

这些算子的实测特征（`IndexCheck` 4033 次、Block Num=6、3.18 µs；`RepeatInterleave` 390 次、1 block、8.28 µs）：
**都是 3–8 µs 的极小 AI_VECTOR_CORE 算子**，但每个后面跟着 **60–260 µs 的空洞**。
`IndexCheck` 的 Op Name 是 `aclnnIndex_IndexCheck_IndexCheck` ⇒ 它是 `torch` **高级索引的边界检查**，
属于 V4.1 的**稀疏索引选择 / 候选过滤**链路。

⇒ 它们的洞**不是自身耗时**（只有几 µs），而是**在等别的东西**（另一条 stream 的产出，
很可能是 AICPU metadata 或 gate 路径）。**这才是下一步该修的地方**，且与
metadata 线（子代理已证明 metadata 走**独立 eager stream**）**是同一个问题**。

## 4. 对四维度的净意义

| 项 | 状态 |
|---|---|
| engram host 空闲（A 的 10.6% 里的大部分） | ✅ **已由 `ENGRAM_DEVICE_INDEX=1` 消除**（已落地、已验收） |
| 剩余纯空闲 5.9%（≈1.5 ms/步） | ⬜ **待攻**：稀疏索引小算子链在等跨 stream 产出 |
| AICPU metadata 4.4% | ⬜ 待攻（子代理在查 host 是否 overlap、以及长上下文 4× 成本） |
| 两者可能有共同根因 | **跨 stream 同步**——都表现为"AI core 算完一小段后干等" |

## 5. 方法学补充

`scripts/gap_context.py`（按前驱算子聚合空闲）是目前**定位"谁在让 AI core 干等"最有效的工具**，
它把"空闲总量"变成"可归因的依赖边"。配合 `duty_fix.py`（占空比分解）与 `idle_gaps.py`（粒度），
三者合起来可以回答：**有多少空闲 → 长什么样 → 谁造成的**。
