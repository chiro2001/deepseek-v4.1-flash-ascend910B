# 016 · 两个补丁合起来的生产配置终验（8 卡真权重 · D2 scheduler × P1 CPU 池）

**日期**：2026-09-22 01:00 – 0?:??（A3 本地时钟；共 N 条臂）　**执行**：子代理 `L2_final`（任务书 `/root/l2_final`）
**机器**：A3（A3-node1），Phy-ID **8–15**（起跑前 `npu-smi` 确认无进程）　**槽位锁**：`locks/c0.lock`
**镜像/上游**：`quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3`（vLLM `0.27.1` + vllm-ascend `e43cf1e9f`）
**模型**：`~/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq`（真权重 DSV4.1-Flash W4A8）
**标记约定**：【实测】= 本机跑出来的原始数据；【推断】= 代码/日志推出来但没直接测；【未确认】= 没跑到。

**补的是什么格**：`009`（D2 的 scheduler 补丁）与 `014`（P1 的 `cpu_npu.py` 补丁）**从未同时挂过**——
009 的通过臂 `d2-dram32-2p` 池子走 `pin_memory`、只跑成 2 个请求；014 的补丁只在**单进程**里烟测过
（import + registered 分支 + `ret=0`），**没在真实 8 卡起过服**。本日志补的正是这一格，并把池子抬到
能覆盖"16 请求 × 32768 token 一轮工作集"。

---

## 0. 一句话结论

**【实测·生产配置终验通过】** —— 两个补丁（D2 的 scheduler × P1 的 `cpu_npu`）**能共存**，
在 A3 的 8 张卡（Phy-ID 8–15）上、**16 个 32768-token 请求跑满**的前提下，**四条判据全部转正**：

| 判据 | 结果（臂 `l2-dram58-16p`，两条补丁同挂） |
|---|---|
| ① `BlockStored(medium="CPU") > 0` | **6,144**（= 16 × 384，**一轮**；`BlockRemoved` = **0**） |
| ② `kv_offload_total_bytes_total{CPU_to_GPU} > 0` | **12.44 GB** |
| ③ `external_prefix_cache_hits > 0` | **507,904 / 1,048,832**（全 bench 48.4%；**replay 轮 96.9%**） |
| ④ replay TTFT ≪ fill TTFT | **253.4 ms vs 4429.4 ms（17.5×）**，replay 轮 wall 71.3 s → **4.06 s** |

**池子（按 `013` §3.3 的条目公式）**：16 请求 × 32768 token 一轮 = **6,144 条**
⇒ **建议 `cpu_bytes_to_use = 58 GiB`**（1.208× 余量）；**最小可用 = 48 GiB**（恰好 1.000×，臂 4 实测通过）。
（**SWA 只存窗口**的 (b) 口径 = **688 条 ⇒ 7 GiB**，但**现状代码做不到**，需另写补丁，见 §1.5。）

**三条顺带纠正**（都会影响别人后续的池子规划，详见 §1.3）：

1. `009` §3.1 的"每条目 ≈34 MiB、32 GiB ⇒ 963 条"**不成立**：实测条目 = **8 MiB**（`worker_kv_bytes_per_block = 131,072 B`），
   32 GiB 该是 **4,096 条** ⇒ 按它的估算会把容量**低估 4.25 倍**；
2. 但反过来 `cpu_bytes_to_use` **不是**宿主占用：**宿主实占 ≈ 6.94 × 该值**（DSV4.1 的 16 个 canonical 张量逐一分配）。
   ⇒ `009` §6 提议的 **245 GiB 在本机 = 1,701 GiB 宿主**，**根本跑不起来**；
3. `registered`（P1 的 β 路径）在真实 8 卡 × 4 档池子（32/45/48/58 GiB）下 **8/8 worker × 16 张量全部 `ret=0`，零回落**。

---

## 1. 第 1 步：按 `013` §3.3 的公式**算清池子**（不占卡）

### 1.1 公式与代入值

```
E_need      = N_resident × Σ_{g∈参与组} ceil(L / tokens_per_chunk_g)
pool_bytes ≥ E_need × round_up(worker_kv_bytes_per_block × blocks_per_chunk × num_copies, ALIGN)
```

| 项 | 值 | 出处 |
|---|---|---|
| `N_resident` | **16**（16 请求 × 32768 token，与 `001` 臂 D 同口径） | 任务书 |
| 参与组 | **12 个**：full(1) + SWA(10) + dspark(1)；`state` 组已被 D2 补丁排除 | `009` §5.2 实测打印 |
| `tokens_per_chunk` | **1024**（block 128 × blocks_per_chunk 8），12 个组全部相同 | `001` §3 |
| **`worker_kv_bytes_per_block`** | **131,072 B = 128 KiB**（不是 `009` §3.1 估的 541,198！见 §1.3） | **本轮臂 1 实测反解** |
| `num_copies` | **8 = world_size**（Ascend 上 `replicated_layout=False`；`013` §1.2 单卡实测=1） | `009` §3.1 |
| **每条目（记账口径）** | `131,072 × 8(blocks/chunk) × 8(copies) = 8,388,608 B = **8 MiB**` | 上式 |
| 余量 | **≥1.2×**（013 实测：短 0.6% 全中 → 短 1.1% 开始掉 → 短 2.3% 整轮归零） | `013` §3.2 |

### 1.2 两种口径的结果（★ 用**实测反解**的 8 MiB/条目）

| 口径 | 每请求条目 | 总条目（×16） | 池子（记账口径） | **建议值（×1.2）** |
|---|---|---|---|---|
| **(a) SWA 全存（现状）** | **384** = 32(full) + 320(10 组×32) + 32(dspark) | **6,144** | 48 GiB | **58 GiB** |
| **(b) SWA 只存窗口** | **43** = 32(full) + 10(10 组×1) + 1(dspark) | **688** | 5.4 GiB | **7 GiB** |
| (b″) 反事实：只裁 10 个 SWA 组、dspark 仍全存 | 74 = 32 + 10 + 32 | 1,184 | 9.25 GiB | 11 GiB |

> ⚠️ 关于 (b″)：主代理指出 **dspark 组也是 SWA 类**（`001` §3 第 13 组），SWA 裁剪一旦生效它**同样该被裁**
> ⇒ **(b″) 这一行只是"假如只裁一半"的反事实，不是任何结论**；采纳的口径是 (b)（43 条，dspark 裁到 1）。
> 另外 `013` §4.3 写的 "44 → 14" 是 **tiny（无 dspark）**的口径，8 卡是 **384 → 43**。

**A3 宿主余量核对（★ 这里有个 6.9 倍的乘数，是本轮的新发现）**：
`cpu_bytes_to_use` **不是**宿主实际占用 —— 池子是按"每个 canonical KV 张量一块"分配的
（DSV4.1 是 **16 个张量**，每个 `num_blocks × (page_size × blocks_per_chunk)`），
16 块加起来是 `131,072 B` 记账值的 **6.945 倍**：

| | 记账值 `cpu_bytes_to_use` | `num_blocks`（条目） | **宿主实占（8 worker 合计）** |
|---|---|---|---|
| 臂 1（实测） | 32 GiB | **4,096** | **222 GiB**（16 张量逐一相加；进程 RSS 合计 283 GiB）|
| 终验臂 | 58 GiB | 7,424 | **≈403 GiB** |
| (b) 若上补丁 | 7 GiB | 896 | ≈49 GiB |

⇒ 宿主 `MemAvailable` = **859 GiB**（起跑前 `free -g`）⇒ 403 GiB 占 **47%**，安全线（150 GB）之上，
但要**明确记下来**：想要"覆盖 16 请求一轮"就得付 ~400 GiB 宿主内存，而不是 58 GiB【实测+推断】。

### 1.3 ★★ 两条必须写下来的口径纠正

**（A）`009` §3.1 的"每条目 ≈34 MiB / 32 GiB ⇒ 963 条"不成立 —— 实测是 8 MiB / 4,096 条。**
臂 1 里 P1 补丁打出的第一手日志（8/8 worker 都有）：

```
[P1_pinned] CPU pool 4096 x 524288 (2.00 GiB)     ← 16 个 canonical 张量，每个 4096 条
[P1_pinned] CPU pool 4096 x  65536 (0.25 GiB)
[P1_pinned] CPU pool 4096 x   1024 (0.00 GiB)     …（×3 组）
[P1_pinned] CPU pool 4096 x 1048576 (4.00 GiB)    ×3
[P1_pinned] CPU pool 4096 x  131072 (0.50 GiB)
[P1_pinned] CPU pool 4096 x    2048 (0.01 GiB)
[P1_pinned] CPU pool 4096 x 1181696 (4.51 GiB)    ← 最后一块 = padding 变体
```

`num_cpu_blocks = 4096` 由 `spec.num_blocks = cpu_bytes_to_use // 8,388,608` 决定
（32 GiB / 8 MiB = 4096，**逐字吻合**）⇒ **一个条目只记 8 MiB 的账**，
即 `worker_kv_bytes_per_block = 131,072 B`（= 8,388,608 / 8 / 8）。
`009` 那个 541,198 B 是**每 GPU block 的平均 KV 字节**（1 GiB / 1984 block），
不是卸载层用的常数 ⇒ 按它算出的"963 条"把容量**低估了 4.25 倍**。

**（B）反过来，宿主实占被**高估**在记账值之外：8 worker 合计 = `num_blocks × 58,253,312 B`。**
16 个张量的 `page_size × blocks_per_chunk` 相加 = **7,281,664 B / rank / 条目**，
× 8 rank = 58,253,312 B ⇒ 一个条目**真实占 55.6 MiB 宿主内存**（记账只记 8 MiB）。
臂 1 的独立佐证：起服前宿主 `MemAvailable` **859 GiB** → 池子分配完成后 **552 GiB**（−307 GiB，
其中池子 222 GiB + 权重/进程 ~85 GiB）【实测】。

### 1.4 一条必须纠正的"相邻数字"（不许用 tiny 的 44/14 顶 8 卡的 384/43）

任务书写的"**每请求 44 条 / 14 条**"是 `013` **tiny（4096 token/请求、11 个参与组、无 dspark）**
的实测值。**8 卡真权重是 32768 token/请求、12 个参与组**，同样的公式下每请求是 **384 条 / 43 条**：

* 两者用**同一个公式**（本机 `pool_sizing_l2.py` 代 tiny 参数可**逐字复现 013 的 44/14**，
  见 `raw/016-pool-sizing.txt`），差的只是 `L` 与"有没有 dspark 组"；
* 32768/1024 = 32 ⇒ 每个参与组每请求 32 条，这是 384 的来源；`009` 的
  `BlockStored(CPU)=12,288 = 16 请求 × 2 轮 × 384` **精确闭合**【实测】。

⇒ 所以本任务的建议值是 **58 GiB**（现状口径 6,144 条 × 1.2），
既不是按 tiny 的 44 条算出来的数，也不是 `009` 按 34 MiB/条目算出来的 245 GiB
（后者在本机需要 1,701 GiB 宿主，**根本不成立**）。

### 1.5 (b) 口径现在还**做不到**（诚实说明）

`is_store_reachable_swa_chunk()` 的裁剪条件是 `alignment_tokens > tokens_per_chunk`
（`scheduler.py::_alignment_chunk_count`）。8 卡上 `alignment_tokens = 1024`（full 组的 chunk）
而 SWA/dspark 组的 `tokens_per_chunk` **也是 1024** ⇒ `1024 <= 1024` ⇒ 函数返回 `None`
⇒ **关闭裁剪、全存**（与 `013` §4.3 "tiny 的 alignment_chunk_count 退化成 1 ⇒ 全存"同源）。
⇒ (b) 是**上限式的规划数**（要拿它必须再写一个"SWA 组只存本段尾 chunk"的补丁），
**本轮的实跑口径一律是 (a)**（臂 1/3 用 32 GiB、臂 5 用 45 GiB、臂 4 用 48 GiB、臂 2 用 58 GiB）。

---

## 2. 环境与做法

### 2.1 两个补丁怎么同时挂上

| 补丁 | 文件 | 挂载开关 | 什么时候生效 |
|---|---|---|---|
| D2（scheduler，`009`） | `shadow-pkg/patches/files/offload_dsv41/scheduler.py`（md5 `0302fab4c68c3adc7d2c4a135c7c4289`） | `OFFLOAD_SCHED_PATCH=1`（**D_off8 时代就有**） | 覆盖 `/vllm-workspace/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` |
| P1（CPU 池后端，`014`） | `shadow-pkg/patches/files/offload_dsv41/cpu_npu.py`（md5 `2c161a791fe99f17cce2e1139ffbdc3c`） | `OFFLOAD_NPU_WORKER_PATCH=1`（★ **本轮新加的开关**） | 覆盖 `/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/cpu_npu.py` |

**新加的开关长什么样**（`agents/L2_final/scripts/patch_serve_a2_npu_worker.py`，幂等 + 备份 + `bash -n` 自检）：

```bash
# shadow-pkg/scripts/serve_a2.sh（PATCH_MODE=mount 段，默认关）
_CPU_NPU="$PKG/patches/files/offload_dsv41/cpu_npu.py"
if [ "${OFFLOAD_NPU_WORKER_PATCH:-0}" = "1" ]; then
  MOUNTS+=(-v "$_CPU_NPU:/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/cpu_npu.py:ro")
fi
# docker run 的 -e 列表里补一行（容器内 cpu_npu.py 读它）：
-e NPU_OFFLOAD_HOST_MEM="${NPU_OFFLOAD_HOST_MEM:-registered}"
```

* P1 的补丁产物**是现场从镜像里抽出的原版重新生成的**（`python3 agents/P1_pinned/scripts/make_cpu_npu_hostmem_patch.py`，
  328 行 → 388 行），md5 与 P1 自己交付的那份**逐字节一致**（都是 `2c161a79…`）【实测】；
* 备份：`shadow-pkg/scripts/serve_a2.sh.L2_final.bak`（回滚 = `cp` 回来，或把两个开关都设 0）。

### 2.2 臂口径

```
ENGRAM=0 PREFIX_MATCH_UNIT=32 BLOCKS_PER_CHUNK=8 PREFIX=1 MAX_LEN=40960
KV_MEM_BYTES=1GiB（GPU KV cache size: 46,387 tokens）PROMPT_TOKENS=32768 max_tokens=1
PROMPTS=16（要求跑满）ROUNDS=2（fill → reset_prefix_cache → replay1）CONCURRENCY=1
```

与 `001` 臂 D / `009` 的通过臂**逐字一致**，只多了"两个补丁同时挂"和池子大小。

---

## 3. 臂 1：小池子（OFFLOAD_GB=32）—— **两个补丁能共存、服务能起**【实测】

**结论：能。** 8 卡起来、跑完两轮 16 请求，rc=0；两个补丁的日志都在同一份 `serve.log` 里同时出现。

### 3.1 两个补丁同时在场的直接证据（`serve.log`，臂 1 实时抄录）

**D2 的 scheduler 补丁**（`[D2_offload]`，EngineCore 打的）：

```
[D2_offload] KV 卸载 group 清单 n=13: [(0,'DeepseekV41FullSpec',128,8,...,True),
   (1,'DeepseekV41CompressorStateSpec',32,3,...,False), (2..11,'DeepseekV41SWASpec',128,4,...,True),
   (12,'DeepseekV41DraftSWASpec',128,3,...,True)]
[D2_offload] 参与卸载的组：full_attention=[0] sliding_window=[2..12]；被排除的组=[1]
```

⇒ 与 `009` §5.2 **逐字一致**（13 组、只有 state 组 `offload_participating=False`）。

**P1 的 `cpu_npu.py` 补丁**（`[P1_pinned]`，8 个 Worker 各打）——以下是我在臂 1 运行期间
（01:09:05–01:09:31）**直读 `agents/L2_final/out/serve.log` 的原文**：

```
[P1_pinned] CPU pool backend = registered (NPU_OFFLOAD_HOST_MEM)          ← 8/8 worker 都有
[P1_pinned] CPU pool 4096 x 524288 (2.00 GiB): registered dev=0x… ret=0   ← 16 个 canonical 张量
[P1_pinned] CPU pool 4096 x  65536 (0.25 GiB): registered dev=0x… ret=0      每个都 ret=0
[P1_pinned] CPU pool 4096 x   1024 (0.00 GiB): registered dev=0x… ret=0
…（一直打到 4096 x 1181696 那块；**无一条 "falling back to pinned"**）
```

> ⚠️ 口径说明：臂 1 当天运行的 `OUT` 是**共用目录**，它的 `serve.log` 在臂 2 起服时被截断覆盖，
> **没能留存**；上面的原文是运行期实时读取时抄下来的。**"128 行 registered / 0 回落"这个精确计数
> 来自臂 3 的重跑**（`§3.5`，同样 32 GiB、带自检门、日志已留存）。两处结论一致。

⇒ **没有任何 worker 回落 `pinned`**（这是任务书点名要盯的那一格）。

### 3.2 为什么这一臂仍然是"零取回"（**不是**补丁冲突，是池子不够）

| 观测 | 值 | 说明 |
|---|---|---|
| `BlockStored(medium="CPU")` | **12,288** | = 12 组 × 32 chunk × 16 请求 × **2 轮** ⇒ 回放轮**又存了一遍**（命中就不会再存） |
| `BlockRemoved(medium="CPU")` | **8,192** | 池子被塞满并持续淘汰 |
| `kv_offload_total_bytes_total{CPU_to_GPU}` | **0.0** | 一次取回都没有 |
| `kv_offload_total_bytes_total{GPU_to_CPU}` | **393.57 GB** | 与 `009` 那条失败的 16 请求臂**逐字节相同** |
| `external_prefix_cache_hits / queries` | **0 / 1,048,832** | fill 轮 0.0%、replay 轮 0.0% |
| fill TTFT p50 → replay TTFT p50 | **4394.8 ms → 4211.7 ms（1.0×）** | 全量重算 |
| 16 个请求的 replay TTFT | 4190.5 – 4239.8 ms（**无一例外**） | 16/16 全未命中（不是"部分命中"） |

**这与公式的预测一致**【实测】：池子 = 32 GiB ⇒ `num_blocks = 4096` 条，
一轮工作集 = **6,144 条** ⇒ 池子/工作集 = **0.667×** —— 按 `013` §3.2 的拐点（0.977× 就已整轮归零），
必然是 0 命中。**这条臂的作用就是把 `013` 的拐点规律从 tiny 外推到 8 卡真权重：规律成立。**

### 3.3 ★ 宿主占用实测（口径校正的依据）

| 时刻 | `MemAvailable` | 说明 |
|---|---|---|
| 起服前（`meta.txt`） | **859 GiB** | 与任务书给的一致 |
| 池子分配完成后 | **552 GiB** | −307 GiB（池子 222 GiB + 权重/进程 ~85 GiB）|
| 8 个 worker 的 RSS 合计 | **282.9 GiB** | 与上式自洽 |

⇒ 32 GiB 的 `cpu_bytes_to_use` 实际吃掉 **222 GiB** 宿主内存 ⇒ 乘数 **6.94×**（§1.3-B）。

### 3.4 ★ 一次"补丁没挂上"的误判 —— 根因是**我自己的取证路径写错了**（记录在案）

主代理在 01:2x 提出"`l2-dram32-16p` 两个补丁都没挂上"，给出的三条证据是：
`serve_cmd.txt` 无补丁痕迹、`grep P1_pinned/D2_offload` 为 0、指标与 `001` 无补丁基线逐位一致。
**实测核对结果：前两条是取证路径错了，第三条是"0 命中时本就该一样"。**

| 主代理看到的 | 真相 | 证据 |
|---|---|---|
| `grep "P1_pinned" l2-dram32-16p.serve_a2.log` → 0 | 那是**臂运行器的 wrapper 日志**（只有 `… 起服中 Ns` 那些行）；容器的 `serve.log` 在 `$OUT/serve.log` | 同一时刻 `agents/L2_final/out/serve.log` 里 `[P1_pinned]` 有 **136 行**、`[D2_offload]` 有 **147 行**（臂 2 的记录，见 §4.1） |
| 臂 1 的 `logs/*.keylines.txt` 里 P1/D2 计数是 **0** | **我的 `run_arm_l2.sh` 有 bug**：它去 `shadow-pkg/results/` 里挑"最新的"目录，而 `serve_a2.sh` 因为 `OUT` 被我们覆盖，**根本没在 `shadow-pkg/results/` 下写** ⇒ 挑到的是 09-21 23:15 的**陈旧目录**，于是永远抽出 0 行 | 臂 1 的 `arm.out` 尾部那几行 `P1 行数：0 … serve.log -> …/off8_off8-dram32_20260921_231505//serve.log`（时间戳 23:15 就是铁证）；**这一段已被臂 3 的新版本替换**（从本臂自己的 `$OUT/serve.log` 抽 + 自检门） |
| 指标与 `001` 臂 D 逐位一致 | **在 0 命中时必然一致**：`001` 臂 D、`009` 的 `d2-dram32`（16 请求）、臂 1 三者都是"池子 < 一轮工作集 ⇒ 0 命中"，`store_bytes` 只反映调度侧决定存多少，与池子大小无关（`001` §2 自己就写了这一点） | 三者 `GPU_to_CPU` 都是 `393,568,321,536 B`；三者 `BlockStored:CPU` 都是 12,288 |

**臂 1 当时就在场的补丁证据**（01:09:05–01:11:08 直读 `agents/L2_final/out/serve.log`，原文已抄进 §3.1）：
`[P1_pinned] CPU pool 4096 x 524288 (2.00 GiB): registered dev=0x3ff37e00000 ret=0`（**这就是 P1 的 logger**，
而且 `4096 = 32 GiB ÷ 8 MiB` 只有 P1 打开时才会被打印），以及
`[D2_offload] KV 卸载 group 清单 n=13 … 被排除的组=[1]`（**这是 D2 独有的行**）。
⇒ **臂 1 不是基线重跑，是"池子 0.667× 的失败臂"；它的价值正是把 `013` 的拐点规律从 tiny 外推到 8 卡。**

**该修的照修**（主代理第 2 条要求，已落地）：`run_arm_l2.sh` 现在
①每条臂写自己的 `OUT=$L2/out/<TAG>/`；②起服完成后、压测前跑补丁生效自检：

```bash
N_P1=$(grep -ac "registered dev="   "$OUT/serve.log")   # 期望 >= 8
N_D2=$(grep -ac "\[D2_offload\]"    "$OUT/serve.log")   # 期望 >= 1
[ "$N_P1" -lt 8 ] || [ "$N_D2" -lt 1 ]  =>  docker rm -f "$NAME"; exit 9
```

臂 3（`l2-dram32-16p-r2`，§3.5）就是带这道门的重跑。

### 3.5 臂 3：32 GiB **重跑**（带补丁自检门）

**目的**：用**机器可判的补丁自检**重跑 32 GiB，把"两条补丁都挂了、结果仍然是 0 命中"这件事钉死。

**自检门原文**（`arm.out`，起服完成、压测**之前**）：

```
[l2][self-check] startup_seen=1 P1_registered=128 P1_fallback=0
```

`P1_registered=128` ⇒ 8 worker × 16 张量全部 `registered ret=0`；`P1_fallback=0` ⇒ **零回落**。
D2 侧同一份日志里 `[D2_offload]` 共 **115 行**（自检阈值 ≥1）。
门**通过**（没有 `exit 9`），压测照跑。跑完的收尾统计：`P1 行数：136 / registered：128 / 回落：0`。

**结果**（与臂 1 **逐项一致**）：

| 判据 | 臂 3（32 GiB，**两补丁确认在挂**） | 臂 1（同配置） | 对照：`001` 臂 D（无补丁） |
|---|---|---|---|
| ① `BlockStored(CPU)` | **12,288**（= 2 轮工作集 ⇒ replay 轮全部重存，**命中才会不重存**） | 12,288 | 12,288 |
| ② `CPU_to_GPU` | **0.0** | 0.0 | 0.0 |
| ③ `external_prefix_cache_hits` | **0 / 1,048,832** | 0 / 1,048,832 | 0 / 1,048,832 |
| ④ fill → replay TTFT p50 | 4545.2 → **4201.0 ms（1.0×）** | 4394.8 → 4211.7 ms（1.0×） | 4192.5 → 4167.4 ms（1.0×） |
| `BlockRemoved(CPU)` | 8,192 | 8,192 | 8,192 |
| `kv_offload_cpu_allocation_size_sum` | 12,288 | 12,288 | 12,288 |
| P1 registered / 回落 | **128 / 0** | 128 / 0（§3.1 原文） | 不适用（无补丁） |

⇒ **结论【实测】：32 GiB（4,096 条 = 0.667× 工作集）在"两条补丁都挂上"的前提下依然是 0 命中。**
这与 `013` §3.2 的拐点规律（0.977× 就已整轮归零）**完全一致**，
也是 `001` §2 自己写过的"`store_bytes` 只反映调度侧决定存多少、与池子大小无关"的必然结果
—— **不是补丁没挂，是池子不够**。

---

## 4. 臂 2：生产口径 **OFFLOAD_GB=58** —— 16 请求 + 四条判据【实测·全部通过】

★ **这就是本任务要的那一格：两个补丁同时挂、池子覆盖一轮工作集、16 个请求跑满、四条判据全中。**

### 4.1 补丁生效的第一手证据（`agents/L2_final/out/l2-dram58-16p/serve.log`，已留存副本）

**P1 的 `cpu_npu.py` 补丁** —— 8/8 worker、**16 个张量各一次注册、全部 `ret=0`、零回落**：

```
[P1_pinned] CPU pool backend = registered (NPU_OFFLOAD_HOST_MEM)       ← ×8（8 个 worker 各一行）
[P1_pinned] CPU pool 7424 x 524288 (3.62 GiB): registered dev=0x3fed7e00000 ret=0   ← ×8
[P1_pinned] CPU pool 7424 x  65536 (0.45 GiB): registered dev=0x3febac00000 ret=0   ← ×8
[P1_pinned] CPU pool 7424 x   1024 (0.01 GiB): registered dev=0x3ffcc200000 ret=0   ← ×8
…（16 个 canonical 张量 × 8 worker = 128 行 "registered dev=… ret=0"）
```

统计：`registered dev=` **128 行**、`falling back to pinned` **0 行**。
⇒ **没有任何 worker 回落 `pinned`**（任务书点名的那一格 = 未发生；`registered` 在真实 8 卡、58 GiB 池下稳定）。

**D2 的 scheduler 补丁** —— `[D2_offload]` 共 **147 行**，含：

```
[D2_offload] KV 卸载 group 清单 n=13: [… (1,'DeepseekV41CompressorStateSpec',32,3,…,False) …]
[D2_offload] 参与卸载的组：full_attention=[0] sliding_window=[2..12]；被排除的组=[1]
[D2_offload] miss-scan … (16 行，全在 fill 轮)   ← fill 轮本来就该全未命中
[D2_offload] load job req=… keys=42 group_sizes=[248, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] src_blocks=42 dst_blocks=259   ← 16 行
```

### 4.2 四条判据（口径与 `009`/`001` 一致；`metrics_after` = 全 bench）

| # | 判据 | **臂 2（16 请求，58 GiB）** | 通过？ |
|---|---|---|---|
| ① | `BlockStored(medium="CPU") > 0` | **6,144**（= 16 × 384，**一轮**，replay 轮一条没重存）；`BlockRemoved(CPU)` = **0** | ✅ |
| ② | `kv_offload_total_bytes_total{CPU_to_GPU} > 0` | **12,444,384,460 B = 12.44 GB**（fill 轮 0 → replay 轮全部） | ✅ |
| ③ | `external_prefix_cache_hits > 0` | **507,904 / 1,048,832 queries**（全 bench 48.4%；**replay 轮 96.9%**） | ✅ |
| ④ | replay TTFT ≪ fill TTFT | **253.4 ms vs 4429.4 ms = 17.5×**（fill wall 71.3 s → replay wall **4.06 s**） | ✅ |

**请求数**：**16/16 跑满**，两轮 `requests_ok=16`、`requests_failed=0`（这正是任务书要的
"`009` 只跑成 2 个"的对照）。
（`n=15/16` 是 TTFT 样本数：`max_tokens=1` 时有 1 个请求没有可计时的首 token —— `013` §1.3 已记录同款现象，
与 `requests_ok=16` 不矛盾。）

**每轮 hits 比例**（`client.json` 逐轮快照，独立的第二口径）：

| 快照 | hits | queries | 轮内增量 | 轮内命中率 | 轮内 CPU→GPU |
|---|---|---|---|---|---|
| `metrics_after_fill` | 0 | 524,544 | +0 / +524,544 | **0.0%**（fill 轮全冷，本该 0） | 0.000 GB |
| `metrics_after_replay1` | 507,904 | 1,048,832 | **+507,904 / +524,288** | **96.9%** | **+12.444 GB** |

缺的 524,288 − 507,904 = **16,384 token = 16 请求 × 1,024 token = 16 × 1 chunk**
⇒ 正是 `is_eagle_group` 的"draft 尾 chunk 必须重算"语义（与 `009` §5 的 2,048 token / 2 请求同源）。

### 4.3 池子利用率（条目口径）【实测】

| 量 | 值 |
|---|---|
| 池子 `num_blocks` | **7,424** 条（= 58 GiB ÷ 8 MiB，逐字吻合 §1.1 的记账口径） |
| 一轮工作集 | **6,144** 条 ⇒ 池/工作集 = **1.208×**（≥1.2× 余量） |
| 累计分配 `kv_offload_cpu_allocation_size_sum` | **6,144**（80 次分配调用） |
| 结束时常驻 | **6,144 / 7,424 = 82.8%** |
| `BlockRemoved(CPU)` | **0** ⇒ 全程**一次淘汰都没有** |
| `kv_offload_allocation_failure_total` | **0** |

### 4.4 每轮 `group_sizes`（任务书点名要看的"`state` 组为 0"）【实测】

16 条 load job **全部**是同一形状：

```
keys=42  group_sizes=[248, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  src_blocks=42 dst_blocks=259
                        ↑
                   idx1 = state 组 = 0
```

* `group_sizes[1] = 0` ⇒ **`state` 组不参与取回**（占位保留，worker 侧 `len(group_sizes)==13` 断言满足）；
* `group_sizes[0] = 248` = 31 chunk × 8 block（full 组 32 chunk 里 1 个是 MTP 尾块，被 `is_eagle_group` 规则扣掉）；
* `group_sizes[2..12] = 1` ⇒ 10 个 SWA 组 + dspark 组各只取回**窗口那 1 个 chunk**；
* `src_blocks=42` / `dst_blocks=259` 与 `group_sizes` 自洽（42 = 31×1 + 11×1，259 = 248 + 11）。

⇒ 与 `009` 的 2 请求臂（`group_sizes=[248, 0, 1×11]`）**逐字一致**，只是这次是 16 条。

### 4.5 宿主占用【实测】

| 量 | 值 |
|---|---|
| 臂 1 起跑前 `MemAvailable`（干净基线） | 859 GiB |
| 臂 2 压测期间 `MemAvailable` | **354.7 – 356.5 GiB** |
| 差 | ≈ 503 GiB（其中池子 403 GiB = 58 × 6.94，权重/进程 ≈ 100 GiB） |
| 安全线（AGENTS.md 的 150 GB） | 未触碰（余量仍有 355 GiB） |

### 4.6 臂 4：**48 GiB = 恰好 1.000× 工作集** —— 最小可用池【实测·通过】

把池子收到"刚好等于一轮工作集"（6,144 条），验证 `013` §3.2 的"1.000× 刚好过、0.977× 归零"在 8 卡上是否同款。

| 项 | 值 |
|---|---|
| 自检门 | `startup_seen=1 P1_registered=128 P1_fallback=0`（通过） |
| 池子 `num_blocks` | **6,144** 条（= 48 GiB ÷ 8 MiB）⇒ 池/工作集 = **1.000×** |
| ① `BlockStored(CPU)` / `BlockRemoved(CPU)` | **6,144 / 0**（一块没淘汰） |
| ② `CPU_to_GPU` | **12,444,384,460 B = 12.44 GB** |
| ③ hits / queries | **507,904 / 1,048,832**（replay 轮 96.9%） |
| ④ fill → replay TTFT p50 | **4518.1 → 252.9 ms（17.9×）**；replay wall **4.06 s** |
| 请求数 | **16/16**，两轮 `requests_failed=0` |
| 宿主 `MemAvailable`（结束时） | 416.1 GiB |

⇒ **与臂 2（58 GiB）的判据数字逐位相同**（`CPU_to_GPU`、hits、`BlockStored` 完全一致）
—— 因为池子只要 ≥ 工作集，取回的东西就一样；58 GiB 多出来的只是安全余量。

### 4.7 臂 5：**45 GiB = 0.9375×** —— 断崖的确切位置【实测·归零】

臂 4 刚证明"1.000× 刚好过"，这一臂把上界往下压到 **0.9375×**：

| 项 | 值 |
|---|---|
| 自检门 | `startup_seen=1 P1_registered=128 P1_fallback=0`（通过 ⇒ 两补丁确认在挂） |
| 池子 `num_blocks` | **5,760** 条（= 45 GiB ÷ 8 MiB，P1 日志 `CPU pool 5760 x 524288`）⇒ **0.9375×** 工作集 |
| ① `BlockStored(CPU)` / `BlockRemoved(CPU)` | **12,288 / 6,528** |
| ② `CPU_to_GPU` | **0.0** |
| ③ hits / queries | **0 / 1,048,832** |
| ④ fill → replay TTFT p50 | 4411.0 → **4212.9 ms（1.0×）** |
| 请求数 | 16/16，`requests_failed=0` |

**算术闭合（这次是"淘汰数"这一侧）**：`12,288 − 5,760 = 6,528`，与实测 `BlockRemoved:CPU = 6,528`
**精确相等** ⇒ 池子被塞满并持续淘汰，`BlockStored` 是 2 倍工作集 ⇒ replay 轮**每个请求都重存了一遍**
（命中就不会重存）—— 与 §4.4 的满命中臂（`BlockStored = 6,144`、`BlockRemoved = 0`）形成干净对照。

### 4.8 ★ 8 卡上的断崖位置（与本轮全部臂）

| 臂 | `OFFLOAD_GB` | `num_blocks` | 池/工作集 | `BlockStored:CPU` | `BlockRemoved:CPU` | `CPU→GPU` | hits | replay TTFT p50 | P1 registered / 回落 |
|---|---|---|---|---|---|---|---|---|---|
| 臂 1 `l2-dram32-16p` | 32 | 4,096 | **0.667×** | 12,288 | 8,192 | 0.0 | 0 | 4211.7 ms | **128 / 0** |
| 臂 3 `l2-dram32-16p-r2` | 32 | 4,096 | **0.667×** | 12,288 | 8,192 | 0.0 | 0 | 4201.0 ms | **128 / 0**（自检门） |
| 臂 5 `l2-dram45-16p` | 45 | 5,760 | **0.9375×** | 12,288 | 6,528 | 0.0 | 0 | 4212.9 ms | **128 / 0** |
| **臂 4 `l2-dram48-16p`** ★ | **48** | **6,144** | **1.000×** | **6,144** | **0** | **12.44 GB** | **507,904** | **252.9 ms（17.9×）** | **128 / 0** |
| **臂 2 `l2-dram58-16p`** ★ | **58** | **7,424** | **1.208×** | **6,144** | **0** | **12.44 GB** | **507,904** | **253.4 ms（17.5×）** | **128 / 0** |

**工作集**（16 请求 × 32768 token）恒为 **6,144 条** ⇒ 断崖**夹在 (0.9375×, 1.000×] 之间**，
与 `013` 在单卡 tiny 上量的 `0.977× 归零 / 1.000× 全中` **同一形态**【实测】。

**对生产的直接含义**：`cpu_bytes_to_use` 必须**按条目数**配，且**不能贴 1.000×**
（0.9375× 就已经全灭、没有部分命中）⇒ 维持 §1.2 的 **1.2× 建议（58 GiB）**。

---

## 5. 机时与产物

### 5.1 机时【实测】

| 臂 | 起跑 | 结束 | wall | 其中起服 | 备注 |
|---|---|---|---|---|---|
| 臂 1 `l2-dram32-16p` | 01:05:00 | 01:14:13 | **9.2 min** | ~5.5 min | 共用 `OUT`（旧版运行器） |
| 臂 2 `l2-dram58-16p` | 01:16:04 | 01:24:38 | **8.6 min** | ~4.5 min | **生产口径，四条判据全中** |
| 臂 3 `l2-dram32-16p-r2` | 01:25:34 | 01:34:47 | **9.2 min** | ~5.5 min | 带补丁自检门 |
| 臂 4 `l2-dram48-16p` | 01:35:47 | 01:44:25 | **8.6 min** | ~5.0 min | 1.000× 最小可用池 |
| 臂 5 `l2-dram45-16p` | 01:46:43 | 01:56:41 | **10.0 min** | ~6.0 min | 0.9375× 归零 |

**五条臂合计 ≈ 46 min**（任务书预算 4–5 条臂；实际跑了 5 条，单臂比预估的 25 min 快得多
—— 起服只要 4.5–6 min，不是 22 min）。

### 5.2 产物

| 路径 | 内容 |
|---|---|
| `a2/logs/016-20260922-l2-production-final.md` | 本文件 |
| `a2/logs/raw/016-l2-<tag>.{client.json,metrics_before/after.txt,kv_events.log/json,kv_size.txt,kv_config.txt,meta.txt}` | **5 条臂**的原始产物（从 A3 拷回，**未改一个字节**） |
| `a2/logs/raw/016-l2-<tag>.serve.log` | 臂 2/3 的完整容器 `serve.log`（含 `[P1_pinned]`/`[D2_offload]` 全部行） |
| `a2/logs/raw/016-l2-<tag>.keylines.txt` | 每条臂抽出的 `[P1_pinned]` + `[D2_offload]` 行 |
| `a2/logs/raw/016-l2-<tag>.arm.out` | 每条臂的臂运行器日志（含**补丁自检门**那一行） |
| `a2/logs/raw/016-pool-sizing.txt` / `016-arms-summary.txt` | 池子公式的完整输出 / 五条臂的一页纸汇总 |
| `a2/agents/L2_final/scripts/pool_sizing_l2.py` | 池子计算（含本轮两处实测标定） |
| `a2/agents/L2_final/scripts/patch_serve_a2_npu_worker.py` | 给 `serve_a2.sh` 加 `OFFLOAD_NPU_WORKER_PATCH` 挂载开关（幂等 + 备份 + `bash -n`） |
| `a2/agents/L2_final/scripts/run_arm_l2.sh` | 八卡臂运行器（c0 锁 + 每臂独立 `OUT` + **补丁生效自检门**） |
| `a2/agents/L2_final/scripts/analyze_arm_l2.py` | 一条臂 → 四条判据 + 利用率 + 逐轮 hits + `group_sizes` |

### 5.3 A3 侧改动与回滚

| 路径 | 改动 | 回滚 |
|---|---|---|
| `shadow-pkg/scripts/serve_a2.sh` | **+18 行**：`OFFLOAD_NPU_WORKER_PATCH` 挂载块 + `-e NPU_OFFLOAD_HOST_MEM` | `cp serve_a2.sh.L2_final.bak serve_a2.sh`（备份在 `shadow-pkg/scripts/`） |
| `shadow-pkg/patches/files/offload_dsv41/cpu_npu.py` | **新增文件**（不覆盖任何既有文件；md5 `2c161a791fe99f17cce2e1139ffbdc3c`） | 删掉即可（或把开关设 0） |
| `agents/L2_final/` | 新增（脚本 + `out/` + `logs/`） | 未动别人任何目录 |

**开关默认值**：`OFFLOAD_NPU_WORKER_PATCH` 默认 **0** ⇒ 不加这条环境变量时，行为与改动前**逐字节一致**。

---

## 6. 未确认 / 风险

| # | 项 | 状态 |
|---|---|---|
| 1 | **数值正确性** | 【未确认】本轮只跑 TTFT/命中/取回判据，**没做**精度或 logprob 对比。`state` 组不参与卸载的语义与 GPU 前缀缓存路径一致，这仍是【推断】（同 `009` §2.5）|
| 2 | `concurrency > 1` | 【未确认】五条臂全是 `concurrency=1`。并发交错会改变 fill/replay 的插入顺序，`013` §3.2 的"相位锁定级联"形态可能变化 ⇒ **1.2× 余量是按串行实测给的，并发下应复核** |
| 3 | 池子上限 | 【未确认】本轮最大只到 58 GiB（宿主实占 ~403 GiB）。宿主 `MemAvailable` 还剩 ~355 GiB ⇒ 理论上还能再抬 ~40–50 GiB 设置值，但**没测**；`014` 的 `registered` 路径在更大尺寸下是否仍 `ret=0` 也未测 |
| 4 | `(b)` SWA 裁剪口径 | 【未实现】430 条 → 43 条需要新补丁（`alignment_tokens <= tokens_per_chunk` 时也要裁）；本轮只给了公式与规划值 |
| 5 | `MAX_LEN=40960` 与生产 `1048576` 的差异 | 【未确认】臂口径沿用 `001`/`009`（为把工作集压进可跑范围）。生产长度下 `E_need` 要按同一公式重算（token 数线性增长，条目线性增长）|
| 6 | 长稳性 | 【未确认】每条臂只在 16 请求 × 2 轮下跑了 ~75 s；`registered` 内存在长时间运行/换入换出下的行为没测 |
| 7 | `ENGRAM=0` | 【未确认】生产是 `ENGRAM=1`。`009` §3.2 的 `207001` 曾把因果归到 Engram，后被 `014` §2 推翻（"不是容量公式问题"）；本轮**没有**用 `ENGRAM=1` 复测（Engram 会让起服多 ~10+ min，超出预算）|

---

## 7. cannbot（AGENTS.md §6）是否适用

AGENTS.md 第 6 节要求"写算子 / 写 kernel / 做量化数值验证"前先查 cannbot。**本轮三件都不属于**：
没有写 AscendC kernel、没有改量化路径、没有做数值/精度验证（表 §6-1 已标【未确认】）。
⇒ **未查 cannbot**，这是**有意**跳过，不是漏掉。

---

## 8. 红线遵守

* 新产物只落 `a2/`（本机）与 `~/projects/dsv41-upstream-pr/agents/L2_final/`（A3）；
  **没有写 `upstream-v41/`**（只读参考），没有改别人的 `agents/<X>` 目录；
* 唯一对共享文件的改动：`shadow-pkg/scripts/serve_a2.sh` **加一个默认关闭的挂载开关**（+`-e` 一行），
  改动前备份 `serve_a2.sh.L2_final.bak`，改后 `bash -n` 自检；新增文件
  `shadow-pkg/patches/files/offload_dsv41/cpu_npu.py`（新增，不覆盖任何既有文件）；
* 不用 `/tmp`（本机 `source a2/scripts/tmpdir.sh l2_final` → `~/tmp/20260922/l2_final`；
  A3 侧只写 `~/tmp/20260922/l2_final/`）；跨机传文件走 `tools/cos-xfer.sh`（不走 ssh 管道）；
* 不手设 `ASCEND_RT_VISIBLE_DEVICES`；8 卡臂全程持 `locks/c0.lock`（owner 文件写/清）；
  起跑前 `npu-smi` 确认 Phy-ID 8–15 无进程；
* **没有**碰 `dsv41-a3`（保持 `Exited`）、`mooncake-master`、别人的容器与 Phy-ID 0–7；
* 结论全部标了【实测】/【推断】/【未确认】；没有用相邻数字顶替缺的那格。
