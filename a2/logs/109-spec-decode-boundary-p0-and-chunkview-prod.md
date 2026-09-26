# 109 — **P0：spec-decode 在上下文边界越界，引擎 8 rank 全死**（上游 vLLM bug，已算术闭合）；+ CHUNKVIEW 生产格实测

> 2026-09-23 02:2x–02:3x CST。执行：子代理 **`ARM_1M`**（8 卡实测，发现 P0）+ **`PROF_int8`**（归因 + 单卡实测）
> + **主代理**（复核原始日志、算术核验、裁决）。标记：**【实测】/【推断】/【未确认】/【实测·代码】/【实测·算术闭合】**。

---

## 0. 一句话

两条**独立于性能**、但决定 A2 能否上线的结论：

① ★★★ **P0（正确性/可用性）**：容器内 `vllm/v1/core/sched/scheduler.py` 在上下文边界**用 `num_sampled_tokens_per_step`(恒=1)
   而不是 `num_lookahead_tokens`(=5，投机) 裁剪** ⇒ 请求走到 `max_model_len − 6` 以内就会让**整个引擎（8 rank）崩掉**。
   **与我们所有补丁无关**（已用挂载点 + 代码两路排除）⇒ **A2 不打补丁也会崩**。
   ★ **口径更正（主代理 02:3x 复核）**：我初稿写"上游 vLLM 的 bug"**过强**。实测：**镜像内 vLLM 是 `0.27.1` 且带厂商内部代号注释**
   （`scheduler.py:531`/`:913` 的 `Marconi shared-prefix junction`）⇒ **是厂商定制版，不是原版上游**。
   准确表述：**缺陷在"镜像里的这份 vLLM"（3196 行、md5 `b959163e`）**；我们仓内另有一份参考快照
   （`graph_prep/ref/vllm/v1/core/sched/scheduler.py`，2915 行、md5 `db867e96`）**含同一模式**（`:121` 定义 / `:531` 裁剪）。
   ⇒ **"上游原版是否同源" = 【未确认】**（无干净上游可比对）。
   ★ 对 A2 的处置**不受影响**（要部署的就是这个镜像）。
② ★★ **CHUNKVIEW 的生产格实测到手**：7938 页下修复 **+33.4 ms/step（残差的 67%）**，且该修复对池大小**是平的**（±3%）。

---

## 1. ★★★ P0：spec-decode 边界越界

### 1.1 【实测】崩溃现场（`r8-1m-bcd3`，1M 单发）
```
num_computed_tokens=[1048573] + num_scheduled_tokens={...: 2} = 1,048,575
num_spec_tokens_to_schedule=5
Index 1048576 out of range[0 1048576)!      x8 ranks
[ERROR] ERR02005 DIST internal error        x16
既有四码 EE1016/507057/EH0012/207001 全 0    ← 这是一个**新签名**
```
★ 主代理逐字复核了这些字段（`grep` 原始 serve.log），**完全一致**。
★ 后果：8 rank 全死、`finish_reasons=[]`（空）、`usage=None`、`/health=000`、引擎整段退出（`MemAvailable` 718 → 1461）。

### 1.2 ★★★ 【实测·算术闭合】四个数字 + 一个索引，逐步对上
| 步 | 式 | 值 | 现场 |
|---|---|---:|---|
| `num_computed_tokens` | — | 1,048,573 | ✅ 一致 |
| 裁剪上限 | `M − nc − num_sampled_tokens_per_step = 1048576 − 1048573 − 1` | **2** | ✅ **正是 `num_scheduled_tokens={…: 2}`** |
| 裁剪后绝对位置 | `1048573 + 2` | **1,048,575 = M−1** | ✅ 最后一个合法位置 |
| 再加 draft 槽 | `num_spec_tokens_to_schedule` | **5** | ✅ 同名 |
| 越界位置 | `1048575 + 1` | **1,048,576 = M** | ✅ **正是 `Index 1048576 out of range[0 1048576)`** |

### 1.3 【实测·代码】缺裁剪的那一行 + 两个界用了两个常数
```python
# vllm/v1/core/sched/scheduler.py:524-532
num_new_tokens = min(num_new_tokens, token_budget)
# Make sure the input position does not exceed the max model len.
num_new_tokens = min(
    num_new_tokens,
    self.max_model_len - request.num_computed_tokens
    - self.num_sampled_tokens_per_step,      # ★ 恒等于 1
)
```
- 同文件 `:119-122`：`self.num_sampled_tokens_per_step = 1 if not is_diffusion else 0` —— **恒等于 1，与 `num_spec_tokens` 无关**。
- 同文件 `:267-270`：`if speculative_config.use_dspark(): self.num_lookahead_tokens = self.num_spec_tokens`（**= 5**）。
- 同文件 `:575-579`：`allocate_slots(..., num_lookahead_tokens=self.num_lookahead_tokens)` ⇒ **KV 内存按 5 预留 ✓**
⇒ ★★ **同一个文件里"KV 界留了 5、位置界只留了 1"** —— **差的那 4 个正好就是越界量**。

### 1.4 【实测·代码】排除我们自己（三条硬证据）
| 事实 | 证据 |
|---|---|
| 我们 patch 的是 **offloading 连接器**的 scheduler | `patched/scheduler.py:10-30` 的 import 全是 `vllm.distributed.kv_transfer.kv_connector.v1.offloading.*` |
| **引擎侧调度器从未被挂载** | `docker inspect` 的**全部挂载点里 `grep -i sched` 无任何输出** ⇒ 容器内 `vllm/v1/core/sched/scheduler.py` 是**镜像原版** |
| 我们**一处都没碰** spec-decode | `grep -c`：`num_spec_tokens_to_schedule`=**0**、`scheduled_spec_decode_tokens`=**0**、`max_model_len`=**0**、`spec_token_ids`=**0**、`pad_spec_decode`=**0** |
⇒ **不用回退任何东西**。**"A2 不打补丁也会崩" = 【推断·强】**（同镜像文件 + 我们补丁不碰它；**未在 A2 现场复现**）。

### 1.5 对 A2 的处置（**必须传达到位**）
- **不是"只有 1M 才炸"**：触发窗口 = `num_computed_tokens ≥ max_model_len − 1 − draft数`。
  A2 生产 `sp_tokens=5` ⇒ **任何请求走到 `max_model_len − 6` 以内都有风险**。
- ★ **上游未修前，A2 的 `max_model_len` 是"不能碰的边界"**：
  业务侧必须留余量（`prompt + max_tokens ≤ max_model_len − 6 − 安全余量`），或接近边界时关投机。
- 建议的裁剪式（`PROF_int8` 给，含**两个必须一起处理的坑**）：
```python
num_new_tokens = min(
    num_new_tokens,
    self.max_model_len - request.num_computed_tokens
    - max(self.num_lookahead_tokens, self.num_sampled_tokens_per_step),
)
```
  ★ 坑①：边界处该值会变**负**（现场 `1048576−1048573−5 = −2`），而下游有 `assert num_new_tokens > 0`（`:903` 附近）
  ⇒ **必须同时 `max(0, …)`**，让请求走已有的 `if num_new_tokens == 0: continue`（`:558`），再由 `check_stop(request, max_model_len)`（`:2107`）收尾。
  ★ 坑②：取 `max(...)` 才能同时覆盖"无投机只需 1 个采样位"与"有投机要 5–6 个 lookahead"。
- 【看不清】：抛 `Index … out of range` 的**具体算子**（设备侧断言）—— "为什么会有 ≥M 的位置"由算术唯一确定，
  但**哪个算子在哪个位置**日志看不出。

### 1.6 ★ 顺带钉死"1M 单发"的**口径**（三个探针）
| `prompt_tokens` | `max_tokens` | 结果 |
|---:|---:|---|
| 1,048,576 | 128 | **400**（total 1,048,704） |
| 1,048,576 | 1 | **400**（total ≥1,048,577，连 1 个输出都放不下） |
| **1,048,448** | 128 | 通过验证 → **prefill 完整跑完（411 s）** → **边界崩** |
⇒ 可用上限 = `max_model_len − max_tokens`，且**还要再留 ~5–16 token 的 spec 余量**才能真正收尾。
⇒ ★ 教训：任何"打满 `max_model_len`"的测试若同时要输出 token，**字面值必然被 400 拒**。

---

## 2. ★★ CHUNKVIEW：生产格实测（`PROF_int8` 在 c1，图内 slope 口径）

| 池页数 | 页视图 µs/层 | ×40 ms/step | **chunk 视图** µs/层 | ×40 ms/step | **收益** |
|---:|---:|---:|---:|---:|---:|
| 320 | 122.7 | 4.91 | 153.2 | 6.13 | −0.2 |
| 3,341 | 490.5 | 19.62 | 155.9 | 6.24 | +13.4 |
| **7,938（★ 8 卡生产真值）** | **989.1** | **39.56** | **153.3** | **6.13** | **★ +33.4 ms/step** |
| 13,366 | 1,570.7 | 62.83 | 159.0 | 6.36 | +56.5 |

⇒ ★★ **chunk 视图在 320 → 13,366 页之间只动 ±3%（153–159 µs/层）** —— 与池大小**解耦**；
页视图则近线性（斜率 **0.1110 µs/页**、截距 87.5 µs、**R²>0.999**）。
⇒ **7938 页 = 生产：修复值 33.4 ms/step = 残差的 67%**。
⇒ 外推 A2（28,577 页）：页视图 ≈**131 ms/step**、chunk ≈**6.4 ms/step** ⇒ **修复值 ≈125 ms/step**。
★ 装置自证：两份 overlay 唯一差别是 `dsa_v41.py`（S `94aeebb7` vs CHUNKVIEW `d84f087c`），
`chunk-view` 标记命中 **0 vs 5**。

---

## 3. 【实测】int64 溢出：**不成立**（三测全 PASS，含 cannbot 的精确位置）

| 测试 | 内容 | 结果 |
|---|---|---|
| T1 | 裸张量两端（首 7 / 末 200 / `2³¹` 处 111） | **PASS** |
| T2 | ★ 用 **AST 从 `dsa_v41.py` 抠出的真实读路径**（`kv8_page_view`+`kv8_gather_pages`）取**末页** | 末页首字节 **99**（期望 99）| **PASS** |
| T3 | ★★ 分配 **`2³² + 4 KiB`** 平面，在 `2³²−64 / 2³² / 2³²+64` / **`4,294,969,344`（= cannbot 报告的失败位置）** 写读 | **全部读回 250** | **PASS** |

⇒ 收敛结论：**调用侧已是 int64/uint64**（`dsa_v41.py:373` 的 `phys.to(torch.int64) * per_page`、`gpu_worker.py:107` 的
`block_ids.astype(np.uint64) * row_stride` —— **恰好是 cannbot 处方要求的写法**），**且实测 >2³² 正常**
⇒ §12 的正确性风险**实质解除**（闭源算子内部仍标【未确认】，但已有正面读数）。
★ 反向强判据（保留）：**A3 自己的池总量 = `1.999 × 2³¹` = `0.9997 × 2³²`**，若走 signed int32 **A3 现在就已经坏**，
而 A3 的 int8 路径已过逐位等价与精度门 ⇒ 总量必为 64 位。

---

## 4. 【实测】池账：**`cpu_bytes_to_use` 是"8 份副本的总量"**（85 GiB 是总量，非 per-rank）
```
aligned_kv_bytes_per_chunk = 131,072 × 8 × 1 = 1,048,576      ← 已含 world_size
num_chunks = 23,068,672,000 // 1,048,576 = 22,000             ★ 日志 num_units=22,000 ✓
旧 = 22,000 × 832,128 = 18,306,816,000                        ★ 日志 旧=18,306,816,000 ✓
91,268,055,040 // 1,048,576 = 87,040                          ★ = A2 的 OFFLOAD_GB=85 ✓
```
⇒ `replicated_layout=False` 的语义是 **"同一个池里存 8 份副本"**，**不是"每 rank 各占一份独立池"** ⇒ "680 GiB"不成立。
★ **一处口径并存（不矛盾）**：`docs/4AXIS-SUMMARY.md §1.4` 的 **296.7 GiB** 是**按"3 个 1M 会话 × 24,064 units"
反推**的（回答"池能装多少上下文"），而本节 85 GiB 是**池的字节预算**（回答"池有多大"）—— **两个问题、两个口径**。
