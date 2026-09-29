# V4.1 DCP8 上下文交接（2026-09-29 19:xx）

**任务**：8 chip 模拟 A2 上实现 `TP8 + --decode-context-parallel-size 8`，
① KV 容量接近 8×　② 正确性达标　③ decode 性能接近 DCP1。

**本文替代之前的进度报告**，是接手者需要知道的全部。历史细节见
`docs/V41-DCP-PROGRESS-20260929.md`（35 KB，含每一轮的门/证据/自我纠错）。

**约定**：【实测】= 真机跑出来的；【推断】= 由代码/公式推出；【未确认】= 没验。

---

## 0. 一句话状态（★ 2026-09-29 深夜更新，性能优化已完成）

| 目标 | 状态 |
|---|---|
| ① 容量 | ✅ **达成**：8,634,871 token。相对**同配置 DCP1 对照**（1,242,687）⇒ **6.95×**；相对任务书给的基线（1,096,072）⇒ **7.88×** |
| ② 正确性 | ✅ **长上下文多选针 2K/8K/16K × 2 种 = 6/6 全过**；短问答与各版本**逐项一致** |
| ③ 性能 | ✅ **达成**：DCP1 30.83 → DCP8 **35.00 / 35.18 / 35.35 ms/step（中位 35.18，1.141×）**，tok/s 28.57 / 28.43 / 28.29，A=1.0；**无数量级退化** |

**性能拆解与优化已做完**，过程与结论见
**`docs/V41-DCP-PERF-20260929-EVENING.md`**（本文 §5 的消融表**已作废**，原因见该文 §1）。

**交付物**：仓内打包器 `experimental/v41-dcp/package_release.sh`（可复现构建 + 正/负控自检）
与包内 `RELEASE.md`；⚠️ **对外发布已撤回**（见下）。

> ⛔ **2026-09-30 撤回**：本页提到的 GitHub Release **已删除**（tag 也已清理，URL 404）。
> 原因：DCP8 尚未稳定 —— 512 B 对齐修复消掉了"logits 均匀分布"的硬故障，
> 但**服务端 `prompt_tokens=16` 仍会给出置信但错误的答案**（`8+7` → `numbersaplenty.c`），
> 而 T=17/18 正常 ⇒ 长度相关问题仍在。
> **恢复发布条件**：短问答与 DCP1 对齐。详见 **`docs/V41-DCP-RCA-20260930.md`**。

---

## 1. 代码与环境（接手第一件事）

### 1.1 仓库与分支

| 项 | 值 |
|---|---|
| 主仓（干活目录） | **`/home/chiro/projects/dsv41/main-merge`** |
| 分支 | **`feat/v41-dcp8`** —— **已 push 到 origin**（`6dcfc5d` 之后的所有提交都已推） |
| 工作树 | **干净** |
| 已提交的关键 commit | `3278910` 去掉 LSE 的 all_gather（零通信参考点）· `43ce760` ori 项本地折叠 · `6dcfc5d` q gather 提前 + 后处理提前切片 · `e5fa11e` 发布包 · `ce15277` **路线 B 打通长上下文** · `656106f` 打包 all_reduce |

### 1.2 overlay 机制（★ 最重要的工作方式）

所有 DCP 改动都在 **`experimental/v41-dcp/overlay/vllm_ascend/`（11 个 .py）**，
通过 `scripts/serve_a2.sh` 的 **`V41_DCP_MOUNT=<dir>`** 整树挂载进容器，
**免重打镜像**。同步脚本：`/home/chiro/projects/dsv41/a2sim-ref/dcp_sync.sh`
（镜像式同步，会删除远端多余文件）。

**三重保险**（都在 `serve_a2.sh` 里，别动）：
1. 起服前打印挂载清单；
2. 起服后**逐文件 md5 比对**容器内 vs 宿主，不一致直接 die；
3. 与 `patches/files/*` 的重复目的地自动去重（overlay 优先，且**最后追加**）。

### 1.3 机器与资源

| 项 | 值 |
|---|---|
| 机器 | **a3-21**（账号 `l00886679`）—— **共用机，绝不抢别人的卡** |
| 设备号换算 | npu-smi 进程表前两列是 `(NpuID, ChipID)`，**设备号 = NpuID×2 + ChipID** |
| 选卡器 | `a3-21:~/dcp_stage_capacity.sh` 的 `free_enough_devices()` / `usable_devices()` / `pick_block()`；`DRY_SELECT=1 bash ~/dcp_stage_capacity.sh` 只看结果不起服 |
| 选卡判据 | **进程表为空 + 空闲 HBM ≥ `MIN_FREE_GIB`（默认 53）** —— 只看进程不够，实测高 8 张进程空但 HBM 只剩 50 GiB，起服必报 `Free memory on device ... less than desired` |
| 优先级 | `CHIP_PREF_ORDER` 默认 `8..15 0..7`（低 8 张被别人的 `acl_bw` 反复抢） |
| 设备 2 | 有 **6.85 GB 驱动级残留**（npu-smi 报 53.1 GiB），非我们的进程，容器清了也在 |

### 1.4 起服（一条命令）

```bash
# a3-21 上，从 ~/cedpd-repo 起（PKG 默认就是它）
nohup setsid env PREFIX=0 BAT_TOKENS=2048 \
  EXTRA_KV_ARGS="--no-async-scheduling" \
  DCP_EXTRA_ENV="V41_DCP_ALLOW_CAPACITY_PROBE=1" \
  bash ~/dcp_stage_capacity.sh > ~/dcp_capN.nohup.log 2>&1 < /dev/null & disown
```

* **冷启动 12–20 分钟**（8 rank 加载 490 GB + 编译 150~176 个 static kernel）；
  `STATIC_KERNEL=0` 可省掉编译但会慢 ~1 ms/step。
* 健康检查：`curl -s --noproxy '*' http://127.0.0.1:19210/health`。
* Serve 参数口径：`TP=8 DP=1 SPEC=0 MAX_SEQS=16 MAX_LEN=1048576 5 GiB 池`。
* ★ **`DCP_EXTRA_ENV` 必须用 `${VAR:-default}`**（已修）：写死会把调用方传的
  额外开关**静默吞掉**，曾白等一轮起服。

---

## 2. 设计（最终版，别改回旧版）

### 2.1 四个 cache 平面的 DCP 语义

| 平面 | DCP 语义 | 谁决定 |
|---|---|---|
| `long_kv`（4 层 full MLA，ratio 1/2） | **分片** 1/dcp | `AscendMLAAttentionSpec.max_memory_usage_bytes` |
| `indexer.k_cache`（4 层） | **分片**（配合 remap） | 同上 |
| `swa`（40 层折 10 组，window 128） | **复制**（每 rank 全量） | 见 §2.3 |
| `compressor.state_cache`（3 层 FP32 32 行环） | 复制 | `AscendCircularBufferSpec` |

### 2.2 ★★★ 核心：跨 rank 的 LSE 合并（`_v41_dcp_merge_attention`）

**必须沿 head 维 all-gather q**：TP 切的是 **head**（rank r 持 `[8r, 8r+8)`），
而 `all_reduce` 是**逐元素**求和 —— 不 gather 就等于把**不同 head** 加起来。
（这是我早期判断"不需要 gather q"的错误；也是短上下文从"全乱"变"全对"的分水岭。）
算力中性：每 rank 工作量 `H_total×L/dcp` = `H_local×L`。

**合并式**（路线 B，已真机验证）：

```
每个 rank 都带真实 ori（内核不允许摘掉，见 §3）
第一次调用：ori ⊕ 自己的 cmp 分片  → (O_r, L_r)，只有 rank 0 带真 sink
第二次调用：cmp_sparse_indices 全 -1 → 纯 ori 的 (L_ori, O_ori)，所有 rank sink = -1e30

Σ_r e^{L_r}·O_r − (Σ_r e^{L_ori}·O_ori − (Σ_r e^{L_ori}·O_ori)/dcp)
────────────────────────────────────────────────────────────────────────
       Σ_r e^{L_r} − (Σ_r e^{L_ori} − (Σ_r e^{L_ori})/dcp)
```

推导：`e^{L_r}·O_r = A_r·O_ori + Z_r·O_cmp + S_r·O_sink`（**原始加权和**，可精确分解）
⇒ 减掉 `dcp` 份 ori、补 1 份 ⇒ 恰好 `S + A + ΣZ`，与全局集合一致。

**两个必须遵守的实现细节（都踩过）**：

1. **归一化必须统一**：`weights = exp(L_r − lse_max)` 是**相对**量，
   而第二次给的是**绝对** LSE。必须用**同一个 `lse_max`** 归一
   （`exp(ori_lse − lse_max)`）。混用 ⇒ 差 `exp(−lse_max)` 因子 ⇒ 输出成复读机
   （离线复核：混用 rel=2.11，统一后 rel=2.1e-3 = bf16 地板）。
2. **修正量必须 rank 无关**：早期版用 `all_reduce(x) − x` 扣除，各 rank 拿到
   **不同分母**却共享同一分子 ⇒ 连 L=59（修复前完全正确）都坏。
   改用两个 `all_reduce` 结果构造**同一个**数值 `X_all − X_all/dcp`。
3. `token_mask` 在路线 B 生效时**必须屏蔽**（否则多扣 `A`）。

### 2.3 为什么滑窗必须复制（A3 硬约束，四条路全堵）

| # | 路 | 被堵原因（实测/源证） |
|---|---|---|
| 1 | `ori_kv=None` | op 层**硬必填**：`tensor of oriKv is nullptr`（`sparse_flash_mla_tiling.cpp:270`） |
| 2 | 收窄窗口（`ori_win_left`） | **硬绑 127**、`ori_mask_mode` 必须 4（metadata EZ0024/EZ0027） |
| 3 | `ori_sparse_indices` 给显式键集 | **A5-only**（`tiling.cpp:1246`） |
| 4 | `seqused_ori_kv = 0` | **会连 cmp 一起清零**（见 §3 根因） |

⇒ 滑窗**必须每 rank 全量复制**。而且这**不贵**：滑窗是「最近 128 + 在飞 token」
的**滚动窗口**，成本 `cdiv(127+in_flight, 128)+1` 与序列长度**无关**
（1M 下 18 块/组）；若按序列分片反而是 1024~2048 块，**贵 8~16 倍**。

### 2.4 容量杠杆（目标①怎么来的）

`request_blocks = full_blocks(dcp) + 1 + 10 × (cdiv(127 + in_flight, 128) + 1)`，
`in_flight = max_concurrent_batches × BAT`，`pp=1` 时 `async_scheduling` 让
`max_concurrent_batches = 2`。

⇒ **关 async + BAT=2048** 把 `request_blocks` 从 2325 压到 1205 ⇒ 7.88×
（实测 **6.95×**，因为 DCP1 基线也受益于同一调参：1,242,687）。
解析模型：**`a2sim-ref/v41_capacity_sweep.py`**，已被真机**三次逐位验证**
（1,096,072 / 4,475,277 / 1,242,687）。

---

## 3. ★ 根因链（长上下文为什么曾经全错）

**内核源证** `sparse_flash_mla_csa_kernel.h:414-421`：

```cpp
// 行无效通过ori部分判断, ori部分如果有行无效那么ori和cmp都有
if (tempLoopInfo.s1EndIdx < -(tempLoopInfo.actOriS2Size - tempLoopInfo.actS1Size)) {
    tempLoopInfo.actOriS2Size = 0;
    tempLoopInfo.actCmpS2Size = 0;      // ★ cmp 被一起清零
    return;
}
```

`actOriS2Size = seqused_ori_kv`。给 rank>0 传 0 ⇒ 右式 = `actS1Size(=L)` ⇒
几乎每个查询块都满足 ⇒ **ori 与 cmp 被一起清零**
（真机实测 rank 1-7 的 LSE 与输出为**精确 0.0**，run `dcpcap_0929_172847`）。

**这就是"路线 A 幻影 ori"失败的同一堵墙**（还有一个原因：null block 实际**不是**
全零 —— 实测 `lse_min=-0.509 < 0`，全零键不可能为负 ⇒ 被 profiling/warmup 写过）。

⇒ 最终走**路线 B**（§2.2 的第二次纯 ori 调用）。

---

## 4. 已排除的嫌疑（14 项，全部有实测依据，**别重查**）

LSE 形状/布局（`(N2,T1,G)` 确为 `(token, head)`）· sink 重复计 ·
head 维未 gather · 空 rank 的有限 LSE（**kernel 写死 0.0，`isfinite` 兜不住**）·
可见性坐标系错配（`prepare_indexer_indices` 用全局界过滤局部索引，1803→0 次错判）·
压缩平面可见长度用全局值（`local_compressed_len` 已修，600+ 长度 × 8 rank 零 mismatch）·
全局 top-k 不足（**定量证否**：`L/dcp/ratio < 512` 时每 rank 选中**全部**本地键）·
`cmp_sparse_indices` 解释空间（kernel 源证 = 压缩序列位置，`sparseBlockSize=1`）·
`cmp_residual_kv` 语义（CSA 里被约掉，传 0 或全局等价）· 物理展开 ·
**跨 rank 合并**（`V41_DCP_MERGE_RANK0_ONLY=1` 判别实验：完整合并与仅 rank0 **都错**）·
tail 位置判据（**已作废**，L=59 纯滑窗就失败 ⇒ 是我的提示词构造问题）·
算子参数与 tiling（生产命中 **static kernel 预编译 bin**，绕过了 `cmpRatio should be 4` 门控）·
`seqused_ori_kv=0` 抑制 ori。

**★ 方法论教训（给接手者）**：
* **针式自由格式在长长度上是无效判据** —— 模型会进复读循环
  （`PLUM-BLOSSOM-BLOSSOM-BLOSSOM…`），被误判成"检索失败"。
  **改用多选格式**（"是 A7 还是 B9，只回答一个"）后全部通过。
* **独立复现两次的异常不要被"构造不同"解释掉**（子代理两次测到 cmp 零贡献，
  我驳回了两次，最后证明他们是对的）。
* **离线对拍必须跑真实代码路径**，只验公式会漏掉量纲/rank 相关性这类一处之差。
* **诊断代码自己不能用宽 `except` 掩盖失败**（我因此白等两轮起服）。

---

## 5. ★ 性能拆解（本轮重点，用户明确要求）

### 5.1 现状数字【实测】

| | DCP1 | DCP8 | 比值 |
|---|---|---|---|
| ms/step（单流 128 token） | **30.83** | **44.22** | 1.43× |
| tok/s（单流，API 口径） | 31.1–31.4 | 21.8–22.1 | 0.70× |
| A（平均接受长度） | 1.0（`SPEC=0`） | 1.0 | — |

差 **+13.39 ms/step**，40 层 ⇒ **+335 µs/层**。
口径：`BAT_TOKENS=2048`、`--no-async-scheduling`、`PREFIX=0`、`MAX_LEN=1M`、
5 GiB 池；流式取相邻 token 间隔**中位数**，单流 `A=1` ⇒ `tok/s = 1000/ms/step`
（与 `usage.completion_tokens/墙钟` 交叉验证一致）。

### 5.2 ★ 已完成的消融（真机实测，**结论反直觉**）

> ⛔ **本节结论已作废（2026-09-29 22:00）**：下表是用"改文件即刻生效"的
> `/tmp/v41_perf_flags` 测的，但 **decode 走 `FULL_DECODE_ONLY` 整图捕获，
> Python 分支只在捕获那一刻求值** ⇒ replay 时开关根本不执行，三臂读数全是噪声
> （我复测：base 42.97 / nopack 43.04 / skip2nd 42.94，差 ≤0.07）。
> 正确的拆解方法（2-chip 图级累积式夹具）与最终归因见
> **`docs/V41-DCP-PERF-20260929-EVENING.md`** 与
> `a2sim-ref/dcp2perf/REPORT_AB3.md`。以下原文保留仅供追溯。

| 臂 | ms/step | 相对基线 |
|---|---|---|
| BASELINE（打包 all_reduce + 第二次调用） | **43.19** | — |
| `no_pack=1`（回到 4 次独立 all_reduce） | **42.09** | **反而快 1.1 ms** |
| `skip_2nd=1`（跳过第二次纯 ori 调用） | **42.49** | 只省 **0.7 ms** |
| `skip_gather=1`（跳过 q all_gather） | **崩溃**（shape mismatch，预期：不 gather 时 head 数不匹配） | — |

**⇒ 集合通信 + 第二次调用合计只解释 ≤2 ms / 13.4 ms（≈15%）。**
⇒ **打包 all_reduce 没有收益**（1 次 vs 4 次在该实现下等价，说明该 collective
不是延迟瓶颈，或被测噪声淹没）；**第二次调用只占 0.7 ms**。
⇒ **剩下 ~11.4 ms/step（≈285 µs/层）来源未知**，这是当前要查的核心。

### 5.3 未完成的拆解工作（接手第一件事）

**已就位的工具**（未提交，在 `dsa_v41.py` 里）：

* **`PERF_FLAG_PATH = "/tmp/v41_perf_flags"`** —— 文件驱动的消融开关，
  **改文件即刻生效，不需要重启**（对比 env 要 12 分钟）。每步读一次
  （Python 层，不碰 device stream ⇒ 与图捕获兼容，前提是测量期间开关不变）。
  支持：`skip_2nd=1` / `skip_merge=1` / `no_pack=1` / `skip_gather=1` / `timing=1`。
  用法：`docker exec dsv41-dcpcap bash -lc 'printf %s "skip_2nd=1" > /tmp/v41_perf_flags'`
* **计时探针 `_time_mark` / `_time_dump`** —— 打开 `timing=1` 后逐层打印
  `[V41-TIME] layer-N total=Xms gather_q=… smla_1st=… build_mask=… smla_2nd=… merge_comm=…`
  占比。**注意它本身会拖慢**（每阶段一次 `torch.npu.synchronize()`），
  给的是**相对占比**而非稳态绝对性能。

**建议的下一步**（按信息量排序）：
1. **先跑 `timing=1`**：把 `[V41-TIME]` 逐层占比拿全，看 ~285 µs/层落在哪个阶段。
   若 `smla_1st` 占比大 ⇒ 是**算力/访存**（不是通信）⇒ 与 DCP 无关的固有开销，
   应该对比 DCP1 同层的同一指标。**这是最可能的答案**：DCP8 下 head 从 8 变 64，
   `npu_sparse_flash_mla` 的 **512 键 × 64 head** 计算量是 DCP1 的 8 倍/rank？
   —— 不，算力应中性（§2.2），但**访存**可能不是：q 变 8 倍、LSE 变 8 倍。
2. **对比 DCP1 的逐层计时**（需要一次 DCP1 重启，~12 min）。
3. 若确认是 attention 算力/访存 ⇒ 优化方向是**降 q 的冗余**（例如只在
   decode 的少数层做 head gather）或换 SFA 的 **all-to-all** 方案
   （每个 rank 只算自己 head 切片、用 a2a 分发，避免 8× q 冗余）。

### 5.4 已验证不可行的优化（别重试）

* **避免 head 维 all_gather**：已论证**不可行**。DCP 下 head 切片 `s` 的正确
  结果需要**所有 8 个 KV 分片**对切片 `s` 的部分结果；只有「每 rank 都算出全部
  8 个切片」才能凑齐 ⇒ q 必须 gather 到全 head（与 SFA 同构）。
* **`skip_gather=1`**：直接 shape mismatch 崩溃。

---

## 6. 验证工具（都在 `/home/chiro/projects/dsv41/a2sim-ref/`）

| 脚本 | 用途 |
|---|---|
| `dcp_realtext.py` | **真实文本长上下文针**（红楼梦语料）。支持 `--position head/tail/middle`。⚠️ **只有 head 有效**（tail 的提示词构造让模型续写原文） |
| `mc.py`（在 `/home/chiro/tmp/`） | **多选格式**长针 —— **这是有效的长上下文判据**（自由格式在长长度无效，见 §4） |
| `dcp_perf.py` | decode 三元组（ms/step 中位数、tok/s、A） |
| `dcp_capacity_sweep.py` | 纯解析容量模型（秒级，三次逐位验证） |
| `dcp_stage_capacity.sh` | 一键起服（含选卡、重试、md5 守门） |
| `dcp_sync.sh` | 本地 overlay → `a3-21:~/dcpw` 镜像式同步 |
| `dcp_equiv/` | LSE 合并的离线数值夹具（真算子，非 HTTP） |
| `dcp2tiny/` | 2-chip tiny 线（**注意：tiny 的 token 级判据是死的**，见其 REPORT_2） |

---

## 7. 交付物清单

| 类别 | 位置 | 状态 |
|---|---|---|
| vllm_ascend DCP 实现 | `experimental/v41-dcp/overlay/`（11 个 .py） | ✅ |
| 可运行启动命令/脚本 | `a2sim-ref/dcp_stage_capacity.sh` + §1.4 | ✅ |
| 正确性实测数据 | §0 表格 + `docs/V41-DCP-PROGRESS-20260929.md` §7 | ✅ |
| 性能实测数据 | §5.1 | ✅（基线） |
| 容量实测数据 | §0 + 解析模型三次验证 | ✅ |
| 仓内文档 | `V41-DCP-PROGRESS-20260929.md`（过程）· `DCP-MERGE-TOPOLOGY-20260929.md`（合并语义）· `DCP-OPERATOR-INVENTORY-20260929.md`（算子清单）· `SFA-DCP-PORTING-MANUAL-20260929.md`（照抄手册）· `V41-DCP-TWO-TRACK-PLAN-20260929.md`（双轨计划） | ✅ |
| 可复现 overlay/patch 打包 | `V41_DCP_MOUNT` + 三重 md5 守门 | ⚠️ **未做成 release 包** |

**未完成**：③ 性能优化（§5.3）；overlay 的 release 打包；未 push。

---

## 8. 环境坑（都被踩过，接手者省时间）

1. **a3-21 是共用机**：别人的 `acl_bw` 会在你选好卡后 20 秒内出现，
   fail-closed 的 `serve_a3.sh` 直接退出 ⇒ 已加重试（自动重选 8 张，最多 12 次）。
2. **选卡必须"进程空 + HBM 够"双条件**（§1.3）。
3. **设备号必须是递增连续块**：给 `8 9 10 11 14 15 0 1` 会 `aclInit 107001 / Invalid device ID`。
4. **`DCP_EXTRA_ENV` 用 `${VAR:-default}`**，写死会静默吞掉调用方的开关。
5. **capture 区内绝不能做 host 同步**：`int(device_tensor)` / `.item()` /
   布尔索引（→ `aclnnNonzeroV2`）会让 **8 个 worker 全部挂掉**。
6. **容器内没有 `py-spy`**；`npu-smi` 进程表两列是 `(NpuID, ChipID)`。
7. **起服日志会被 `: > "$LOG"` 截断** —— 宿主侧诊断不要写进 `serve.log`。
8. **禁止用 `rm -f`**（工具安全策略会拦），用 `mv` 到暂存目录。
9. **不要用 `/tmp` 存产物**（用 `~/tmp/` 或 `results/`）。
10. **a3-22 不可用**（0-15 全被别人的进程占）。

---

## 9. 用户偏好（必须遵守）

* **全程简体中文**。
* 报性能必须给 **`(ms/step, A, tok/s)` 三元组并标口径**。
* 结论必须标 **【实测】/【推断】/【未确认】**，判据要绑"实际生效后的可观测痕迹"。
* **优先级：正确性 → 推理速度 → 容量**（用户明确说过"容量不对不是硬门槛"）。
* 出报告要 scp 到 `192.168.101.5` 的 `D:\Downloads`。
* 大文件走 COS。
* **不要擅自重启用户的生产服务**。

---

## 10. 接手后建议的动作顺序

1. `cd /home/chiro/projects/dsv41/main-merge && git status` —— 确认那 1 处未提交改动
   （性能消融开关），**建议先 commit**（功能已验证可起服）。
2. 检查 a3-21 上的服务：`docker ps | grep dcpcap` + `curl 19210/health`。
   （交接时正在起 run `dcpcap_0929_190852`，用 8-15 卡。）
3. **跑 `timing=1` 拿逐层占比**（§5.3 第 1 步）—— 这是当前最高信息量的一步，
   不需要重启（改文件即可）。
4. 按占比决定优化方向；若确认是 attention 算力/访存，考虑
   「只在 decode 的少数层做 head gather」或 SFA 的 all-to-all 方案。
5. 目标 ①② 已达成，交付前建议补：overlay 的 release 打包 + push。
