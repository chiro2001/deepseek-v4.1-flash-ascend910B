# 056 — ②a（draft 的 KV 也走 INT8）单 die 判定：**容量 ✅ / 图捕获 ✅ / 但引擎在第一个 spec-decode step 就死 ⇒ Q3 无读数，不能替代 ②c**

> 2026-09-22 13:0x–13:5x CST。执行：子代理 **D_draftINT8**。
> 机器：**A3（A3-node1）的 c1**（`tools/a3_chip.sh c1`；**全程只用 c1**，未碰 c0/c2、未碰 `dsv41-a3` / `mooncake-master` / 别人的容器，
> 未手设 `ASCEND_RT_VISIBLE_DEVICES`；每臂起服前查 `/dev/shm`，`Used 0`；未用 `/tmp`；未写 `upstream-v41/`）。
> 模型：`agents/T_draftceiling/models/model-tiny-draft`（tiny + 只还原 draft 两个键 ⇒ 有 g12 draft 组）。
> 影子包：`agents/D_draftINT8/pkg-ddi` = `C1_graph1die/pkg-gs`（= X_integrate/pkg-ring + **发布件** `attention/dsa_v41.py` `94aeebb7…` + R 的 draft-aware 槽位补丁）+ ②a 两处改动。
> 产物：本日志 + `logs/raw/056-d-draftint8/` + `agents/D_draftINT8/`。
> 标记：**【实测】**= 有判别力的判据跑出来的数；**【算术】**= 由源码几何算出（脚本可复跑）；**【推断】**；**【未确认】**。

---

## 0. 一句话结论

> ★★ **读这张表请先看这一行**：**Q1 与 Q2 都是「过」的，只有 Q3 是「死」的。**
> ⇒ 不要把它读成"②a 这个方向整个没价值"。**Q1 的通过本身是一个独立有价值的结论**：
> 它实测证明 **`049` 的 `GRAPH_SAFE` 修复确实覆盖了 draft 面**（`050` 写下那条"必炸"时 `049` 还没出生）。
> ★ 但要分清**为什么死的**：②a 死在**运行期**（draft 的 KV 被声明成两平面、却没有一层按两平面去存/取），
> **不是**死在 spec / 页几何 / 图捕获 —— 那三件事本轮都**逐字对上了**。

| # | 问题 | 答案 | 判据 |
|---|---|---|---|
| **Q1** | ②a 能不能图捕获？ | **能**（**已不再是阻塞** —— `049` 的 `GRAPH_SAFE` 把 draft 面那条路也覆盖了）**【实测】** | `capture_finished=1`、`EE1016=0`、`Not_Supported=0`、`capture failed=0` |
| **Q2** | 容量对不对？ | **逐字命中零参数模型**：tiny 档 D **23,651 → 39,846（×1.6847）**，`R8-SLOTS` 的 `draft=66,560 / capacity=66,560` 与算术**逐字相同**【实测】 | §3 |
| **Q3** | 投机解码退化了没有？ | ★★ **测不出来 —— 引擎在第一个 spec-decode step 就死**（16/16 请求失败，0 条 `SpecDecoding` 读数）。**【实测：失败】 / 【未确认：指标】** | §4 |
| **能不能替代 ②c？** | ★★ **不能。** ②a 在**真实请求**下不可用（引擎崩），而 ②c 的 Q3 是**实测过**的（提案序列 4367/4367 逐条相同）。⇒ **交付推荐仍是 ②c**，②a 退回"研究项"。 | §5 |

**★ 本轮最值钱的一条（对后续所有路线都成立）**：**②a 的 spec 侧改动是有效的**（页 66,560、容量 ×1.6847、图能捕获）——
它死在**运行期**：draft 层的 KV 存取走的是 `dsa_v1.py::AscendDSAImpl`（不是 `dsa_v41.py` 的 target 路径），
而 **KV8 的量化存取机制（`kv8_swa_store` / `kv8_ori_plane`）只实现在 `dsa_v41.py` 一侧**。
★ **主代理独立复核过这条**【实测·源码】：`dsa_v1.py` 里 `kv8_swa_store = 0 次 / kv8_ori_plane = 0 次`，
而 `dsa_v41.py` 里是 `3 次 / 7 次` ⇒ **"要先把 KV8 的量化存取移植进 `dsa_v1.py`"** 这条结论**有源码级支撑**（仍标【推断】的是"它是否就是本轮第一个异常"）。

---

## 1. 实验设计（三条臂**只有一个变量**）

| 臂 | `VLLM_V41_DRAFT_INT8` | 图 | 结果 |
|---|---:|---|---|
| `ddi-d-i8-eager` | **1** | eager | 起服 ✅（55 s）→ **首个请求即崩**（§4） |
| `ddi-d-i8-graph` | **1** | `FULL_DECODE_ONLY` | 起服 ✅、**图捕获 9/9** → **首个请求即崩**（同 §4） |
| `ddi-d-bf16-e` | **0**（= 逐字旧行为） | eager | ★ **全绿**：`fill` ok=16/16、`ttft.p50=738.1 ms`、`sha=0ebccb55b30c…` |

* 三条臂**同一个包**、同一批服务参数、同 `prompt_salt=20260922`、同 `max_tokens=64`（★ `mt=1` 的 SpecDecoding 没有统计功效，`S_graphfix` 已踩过）；
  唯一差别是 `VLLM_V41_DRAFT_INT8`。臂内 md5 见 §2.3。
* ★ 对照臂的输出 sha `0ebccb55b30ce1a5d17822e4e625984cd106f79615beef957a9c5754cf1b3a13` 与 `054`（②c）**四臂逐字相同**
  ⇒ 本轮的 harness / 权重 / 采样口径与 ②c 那轮**同源**，可交叉引用。

---

## 2. 改动面（②a = **2 个文件 / 2 处**）+ 构建门

### 2.1 两处改动（生成器 `agents/D_draftINT8/scripts/patch_draft_int8.py`，env 门控，默认关）

| # | 文件 | 改什么 | 为什么必须 |
|---|---|---|---|
| ① | `core/deepseek_v41.py::DeepseekV41DraftSWASpec.__post_init__` | 放行 `int8 + scale_dim`（校验模板**逐条照抄**紧邻的 `DeepseekV41SWASpec.__post_init__`）；`num_kv_heads==1` / `compress_ratio==1` 两条**不放松** | 上游硬卡：`if self.dtype != torch.bfloat16 … raise ValueError("Aurora DSpark requires one uncompressed BF16 KV plane")` |
| ② | `models/deepseek_v41/dspark.py::DeepseekV41DSparkSWACache.get_kv_cache_spec` | 传 `dtype=int8, scale_dim=4, scale_dtype=fp16` | 直接决定 draft 组的页大小 |

* ★ **`core/kv_cache_interface.py` 不用改**：`AscendSlidingWindowMLASpec` 在 KV8 shadow 里**已经有** `scale_dim/scale_dtype`，
  且 `real_page_size_bytes = storage_block_size × num_kv_heads × (head_size×itemsize + scale_dim×itemsize(scale_dtype))`
  ⇒ 代进 draft：`128 × 1 × (512×1 + 4×2) = 66,560 B`。**页大小公式是通用化的，这是"存储侧复用 scale_dim 机制"的原话。**
* ★ **`plan_cache_slots` 的 draft 检查也不用改**：`block_size 128==128`、`head_size 512==512`、`sliding_window 128==128`、
  `Σdraft = 66,560 ≤ capacity = 66,560`（**余量恰好 0**）⇒ 四条全过。**真机实测确认没有 raise**（否则起服就死，见 §3）。

### 2.2 ★ 回答"改 draft 会不会顺带改 target 的 40 个 SWA 层"（**源码级**）

**不会。** 父类 `AscendDeepseekV4SWACache.get_kv_cache_spec` 的 dtype 来自
`attention/dsa_attn_kv_plan.py::get_dsv4_attn_kv_dtype(vllm_config)`：
```python
return (torch.bfloat16
        if not _supports_dsv4_compressed_cache() or is_a5_bf16_kv_enabled(vllm_config)
        else torch.float8_e4m3fn)
```
—— 它是**硬件档案判据**（`DSV4_COMPRESSED_CACHE` 只挂在 A5），**不是** `KV_DTYPE` / `cache_dtype` 那种全局开关。
而 target 的 40 个 SWA 层走**另一个类** `AscendDeepseekV41SWACache`（用 `swa_plane_kwargs()` 读 `VLLM_V41_KV8_SWA`）。
⇒ ②a 用**独立 env**（`VLLM_V41_DRAFT_INT8`）+ **只覆写 draft 类**产出的 spec，**对 target 零影响**。
（本轮 `R8-SLOTS` 的 target 侧读数 `slot3` 在 ②a 与基线之间逐字相同，是这条的**运行期旁证**。）

### 2.3 构建门（"只差这两个文件"）

`scripts/build_pkg.py` 对 base 做 `diff -rq`，**差异文件必须恰好 2 个**，否则 exit：
```
[ddi-build] base dsa_v41.py md5=94aeebb757d6d5708268754481a05e0a  ✅ 与发布件一致
[ddi-build] 差异数=2（期望 2）✅
   a166c948ef101f13744112842d7a228b  shadow/vllm_ascend/models/deepseek_v41/dspark.py
   29033d6eb5a1da81c88b3b0ef00adaf1  shadow/vllm_ascend/core/deepseek_v41.py
   94aeebb757d6d5708268754481a05e0a  shadow/vllm_ascend/attention/dsa_v41.py   ← 发布件，未动
   7e17f7cae054f0b2339b41b7c3642f0e  shadow/vllm_ascend/core/kv_cache_interface.py
```
★ **md5 现算**（`md5sum`，不是抄的）：`dspark.py = a166c948ef101f13744112842d7a228b`、`core/deepseek_v41.py = 29033d6eb5a1da81c88b3b0ef00adaf1`。

### 2.4 过程事故两条（**都是我自己的构造缺陷，不是 ②a 的机制问题**，如实记录）

1. **`NameError: name 'os' is not defined`（`dspark.py:65`）** —— 我的探针用了 `os` 但 `dspark.py` 头部**没有** `import os`。
   ⇒ 两条臂第一次起服全部死在这里（`EngineCore failed to start`）。修法：探针块自带 `import os`（已加注释说明这是"探针引入新失败"的实例）。
2. **`ModuleNotFoundError: No module named 'vllm_ascend.attention.kv8_prefill_triton'`** —— base 来自 `pkg-ring`，**没有**这个文件；
   我默认开了 `VLLM_V41_KV8_PREFILL=1`（那是 `pkg-kv8pf` 才有的件）。
   ⇒ 默认改 **0**（prefill 融合只影响 prefill 期 kernel 选择，与 spec/页/容量/投机无关；`054` 的档 D 臂也是同口径）。

---

## 3. Q1 / Q2 的实测（**两问都过**）

### 3.1 Q1 —— 图捕获 ✅（**②a 原来的"必炸"阻塞确实已被 `049` 解除**）

| 臂 | `Graph capturing finished` | `EE1016` | `Not_Supported` | `capture failed` |
|---|---:|---:|---:|---:|
| `ddi-d-i8-graph`（`DRAFT_INT8=1` + `FULL_DECODE_ONLY`） | **1** | **0** | **0** | **0** |
| `ddi-d-i8-eager`（`GRAPH=0`） | 0（eager 本就不捕获） | 0 | 0 | 0 |

★ 与 `053` 的单 die 结论一致（那里测的是档 C：`capturing=True rows_bound=6 … EE1016=0`）。
★ **本轮的"反例"是天然存在的**：`ddi-d-bf16-e` 与 `ddi-d-i8-graph` 用**同一个包**，只差一个 env ⇒
若 Q1 的判据没有判别力，两边应当给出同样的数；实际两边**都**能起服，而 **②a 在请求期死、基线不死的对照是 §4**（那才是本轮真正的判别量）。

### 3.2 Q2 —— 容量 ✅（零参数模型**逐字命中**，含 ②a 这一格）

**探针（`[DDI-2a-PROBE v1]`，热路径 trace，12 行 = 3 draft 层 × 4 次调用，用来证伪"补丁没生效"）**：
```
draft-spec pid=P call#1..12 env='1' -> block=128 storage_block=128 dtype=torch.int8 scale_dim=4
                                        page_bytes=66560 head=512 window=128 cls=deepseek_v4
```
★ 打的是**实际生效后**的值（`env='1'` 来自运行期 `os.environ`，不是"我打算用的值"）。

**逐槽读数（`[R8-SLOTS]`）—— ②a vs 基线，同一几何**：
```
②a   : slot=0/1/2  kv=33280 index=8320 aliases_max=66560 draft=66560 capacity=66560 legacy_capacity=66560
基线 : slot=0/1/2  kv=33280 index=8320 aliases_max=66560 draft=131072 capacity=131072 legacy_capacity=66560 [draft-aware]
两者 : slot=3      kv=66560 index=16640 aliases_max=66560 draft=0      capacity=83200 legacy_capacity=83200   ← 逐字相同
```
⇒ ★ **draft 从 131,072 掉到 66,560，与 target SWA 的 `aliases_max=66,560` 顶平（不顶穿）** ——
这正是"②a 的 Σ 与 ②c 完全相同、只赢在 BPR"那句话的 slot 层证据。

**容量（引擎自证，`GPU KV cache size`）**：
| 格 | Σslot_pages | BPR | 预测 | **实测** | 判定 |
|---|---:|---:|---:|---:|---|
| tiny 档 D + draft BF16（基线） | 476,416 | 780 | 23,651 | **23,651** | ✅逐字 |
| ★ **tiny 档 D + draft INT8（②a）** | **282,880** | **780** | **39,846** | **39,846** | ✅逐字（**×1.6847**） |
| tiny 档 D + draft block 64（②c，`054`） | 282,880 | 844 | 36,825 | 36,825 | ✅逐字（×1.5570） |
★ **②a 与 ②c 的 Σ 相同（282,880），差别只在 BPR（780 vs 844）⇒ ②a 严格更优（+8.2%）** —— 这条**算术已实测坐实**。
★ 8 卡口径（**【外推】**，本机未测）：②a 档 D **817,898（×1.6843）**；档 C **626,488（×1.4650）**。

**模型对账**（`scripts/d_model.py` 可复跑）：**11 个已知实测点全部逐字命中**（tiny 无 draft 3 点 + tiny draft128 3 点 + tiny draft64 2 点 + 8 卡 3 点），
本轮**再添 2 点**（②a tiny 档 D 的预测/实测、②a 的基线）⇒ **零参数模型现在 13/13**。
```
[对账] 已知点不符的个数 = 0
```

---

## 4. ★★ Q3：**测不出来 —— ②a 在第一个 spec-decode step 把引擎打死**

### 4.1 现象（三条臂的对称读数）

| 臂 | 起服 | `fill` | `replay` | 输出 sha | `SpecDecoding` |
|---|---|---|---|---|---|
| `ddi-d-i8-eager` | ✅ 55 s | **ok=0 / failed=16** | ok=0 | 无 | **0 条** |
| `ddi-d-i8-graph` | ✅（图捕获 9/9） | **ok=0 / failed=16** | ok=0 | 无 | **0 条** |
| **`ddi-d-bf16-e`** | ✅ | **ok=16 / failed=0**，`ttft.p50=738.1 ms` | 正常 | `0ebccb55b30c…` | ★ 有 |

**★ 对照臂（`ddi-d-bf16-e`）的完整读数【实测】** —— 它是本轮**唯一**能跑完 Q3 口径的臂：
```
pass1 fill    : ok=16 fail=0  ttft.p50=738.1 ms  sha=0ebccb55b30ce1a5d17822e4e625984cd106f79615beef957a9c5754cf1b3a13
pass1 replay1 : ok=16 fail=0  ttft.p50=731.2 ms  sha=0ebccb55b30ce1a5d17822e4e625984cd106f79615beef957a9c5754cf1b3a13
pass1 replay2 : ok=16 fail=0  ttft.p50=727.2 ms  sha=0ebccb55b30ce1a5d17822e4e625984cd106f79615beef957a9c5754cf1b3a13
[sha256] fill == replay1 == replay2，common=16，mismatched=[]，match=True
```
★ **三个 sha 与 `054`（②c 那轮）四臂的输出逐字节相同** ⇒ **本轮的包 / 权重 / harness / 采样口径与 ②c 那轮同源**，
而 `DRAFT_INT8=1` 两条臂在同一口径下 **ok=0** ⇒ 这是**同包同参数、只换一个变量**的对照，判据有判别力。

### 4.2 确切的失败点（**文件:行号 + 错误串**）

```
(EngineCore) ERROR [core.py:1351] RuntimeError: The previous device metadata submission has not been released
  File ".../pkg-ddi/shadow/vllm_ascend/worker/worker.py",            line 683,  in execute_model
  File ".../pkg-ddi/shadow/vllm_ascend/worker/model_runner_v1.py",   line 2335, in execute_model
  File ".../pkg-ddi/shadow/vllm_ascend/worker/model_runner_v1.py",   line 3568, in _build_attention_metadata
  File ".../pkg-ddi/shadow/vllm_ascend/worker/device_metadata.py",   line 74,   in submit
      raise RuntimeError("The previous device metadata submission has not been released")
```
* 触发时机：**第一个 decode step**（`step_counter=0`、`num_scheduled_tokens=6`、
  `scheduled_spec_decode_tokens={req: [-1,-1,-1,-1,-1]}` ⇒ 正是 **spec-decode 的 6 行 query 形状**，与 `048`/`053` 的击穿形状同一个）。
* 后果：`EngineDeadError` → 16 个在途请求全部失败 → 引擎进程退出。**图臂与 eager 臂都是同一条**。

### 4.3 为什么会这样（**推理链，逐条标注强度**）

1. **`submit` 的这条错误是"症状"不是"病"**：`DeviceMetadataExecutor.submit` 只在 `self._submission_in_flight == True` 时抛它；
   而 `_submission_in_flight` 只由 `release()` 清（`worker/device_metadata.py:127`；调用点在 `model_runner_v1.py:2465-2466` 与 `:3917-3918`）。
   ⇒ 上一轮的提交**没被 release**。**【实测·源码】**
2. **两个 release 都不在 `finally` 里**：它们紧跟在 `self._model_forward(...)` 之后（`:2465` 与 `:3917`），
   ⇒ **只要 forward 抛过一次异常，release 就会被跳过**，下一次 `submit` 报这条错。**【实测·源码】**
3. ★ **病灶在最可能的那一处（【推断】，尚未被本轮探针证实）**：draft 层的 attention 实现是
   `models/deepseek_v41/dspark.py` → `DeepseekV41DSparkAttention` →（V4）`AscendDeepseekSparseAttention` →
   **`attention/dsa_v1.py::AscendDSAImpl`** —— **不是** `dsa_v41.py::DeepseekV41EagerAttentionImpl`。
   而 KV8 的两条量化存取路径（`kv8_swa_store` / `kv8_ori_plane`）**只写在 `dsa_v41.py` 里**；
   `dsa_v1.py` 侧对 `swa_kv_cache` 是**裸张量**语义：
   * 写：`get_dsa_attn_kv_plan(...).dsa_kv_compress_scatter(swa_kv_cache, kv, slot_mapping)`
     → `updates = x.reshape((slot_mapping.shape[0],) + tuple(cache.shape[2:]))` + `torch_npu.npu_scatter_nd_update_`（无 int8/scale 分支）
   * 读：`attn_kwargs.update(ori_kv=swa_kv_cache, …)` 直接喂 `npu_sparse_flash_mla`（无 tuple/反量化分支）
   ⇒ draft 的 spec 说"我是 `(payload, scale)` 两平面"，但**没有任何一层按两平面去存/取它**。**【推断，待 §4.4 证实】**
4. 本轮**没有**在日志里看到第一条异常（`tuple` / `scale_dim` / `shape` / `AttributeError` 命中均为 **0**）——
   ⇒ 它被吞在 engine 的 dump 之前（`dump_input.py` 只在引擎级 fatal 时落盘）。**【实测：没抓到】**

### 4.4 ★ 诊断探针（设计的初衷；**结果见 §4.4b —— 已跑完**）

`agents/D_draftINT8/probe/usercustomize.py`（`PROBE=1` 时经 `PYTHONPATH` 生效）：
* 包 `DeviceMetadataExecutor.submit/release` ⇒ 打**配对 trace**（第 N 次 submit ↔ 第 N 次 release）；
* 包 `dsa_v1.py::AscendDSAImpl._forward_attention` ⇒ 打进/出 + **首个异常 + 完整 traceback**（打印后原样 re-raise）；
* 遵守 `AGENTS §5b`：**一个 target 一个 finder 实例**、每个 target 打**带替换后函数地址**的已生效横幅、用 `importlib.util.find_spec` 并 `try/finally` 插回。

驱动脚本 `scripts/d_diag.sh`（短几何 `1 × 256 token, max_tokens=4`，几十秒就能走到崩溃点）。
**状态：已在 `c1` 空出后跑完（`ddi-diag`，13:36）⇒ 结果见 §4.4b。**

### 4.4b ★★★ 诊断臂已跑（`ddi-diag`，c1，13:36）—— **泄漏点被精确定位到"某一次提交"**

★ 先说结论（全部**实测**）：**探针生效**，且它把"症状"定位到了**具体哪一次提交**：

```
[DDI-DBG v1] ARMED DeviceMetadataExecutor.submit -> ...submit @0xfffefd1fd800; release -> ...release @0xfffefd0647c0
[DDI-DBG v1] ARMED AscendDSAImpl._forward_attention -> ..._forward_attention @0xfffefd...      ← ★ 这条**从未被调用**
[DDI-DBG v1] submit#1  in_flight=False tasks=7  frontiers=[(0,·),(1,·),(1,·),(2,·),(2,·),(2,·),(2,·)] bd=None
[DDI-DBG v1] release#1 (submit#1)                                                              ← ✅ 配上了
[DDI-DBG v1] submit#2  in_flight=False tasks=1  frontiers=[(2, 281470296526752)] bd=None        ← ★★ 只 1 个任务
                                                                                                  ★★ 此后**没有 release#2**
[DDI-DBG v1] submit#3  in_flight=True  tasks=7  frontiers=[…与 submit#1 逐字相同的 7 个…]       ← ⛔ 在这里抛 RuntimeError
```

**三条可读的推论**（强度逐条标注）：
1. **泄漏的是 `submit#2`** —— 它**没有配对的 release**，所以 `submit#3` 一进来就撞 `in_flight=True`。**【实测】**
2. **`submit#2` 的形状与其余每一次都不同**：**只 1 个任务**、且它的 `group_id`（`281470296526752`）
   在 `submit#1`/`submit#3` 的 7 个里**一次都没出现过** ⇒ 它来自**另一个 metadata builder**（= **draft 侧**那一个，`dsa_v1.py::AscendDSAMetadataBuilder`；target 侧的一次提交恒定是 7 个任务：1 COMPRESSOR + 2 INDEXER + 4 ATTENTION）。**【实测 + 推断】**
3. ★★ **`release` 的缺口在 draft 侧的 execute 路径上**：`_prepare_device_metadata_for_forward` 只在
   `submission_in_flight` 为真时返回 executor（`model_runner_v1.py:3002-3010`），而 `release()` 的调用点只有两处
   （`:2465-2466` 与 `:3917-3918`）—— **都在 target 的路径上**；draft 的 forward 走完之后**没有第三个 release 点**。**【实测·源码 + 推断】**

**★ 一条负面但重要的事实（防止误判）**：**首个异常在日志里彻底看不见** ——
`has no attribute` / `tuple` / `AttributeError` / `npu_scatter` / `ori_kv` 在本臂日志里命中**全是 0**，
崩溃前也只有 `serving.py` 那两条（即 `submit#3` 之后的结果）。
⇒ **不能把 §4.3 那条"`dsa_v1` 缺量化存取"当作"已证实的首发异常"** —— 它仍是**【推断】**，
只是**方向被这条 trace 增强了**（泄漏确实发生在 draft 侧那一次提交上）。

**★ 探针本身的一条修正（也是我这轮的第 3 个构造缺陷，已记）**：我把 `_forward_attention` 钩在
`dsa_v1.py::AscendDSAImpl` 上，但它**一次都没被调用** ⇒ **draft 的 attention 不走这个类**：
`ops/dsa.py:35` 显示它用的是 `models/layer/attention/layer.py::DSAAttention`（`AscendDeepseekSparseAttention.dsa_attn = DSAAttention(...)`）。
⇒ **要拿"病行"，下一个探针 target 应该是 `DSAAttention`（`models/layer/attention/layer.py`），不是 `dsa_v1.py`。**
（★ 这条本身也是一条教训：**探针"从未被调用"绝不能当作"那条路没问题"** —— `044` 的 `ring_calls=0` 就是这个坑。）

★ **另一个实现选择（记下来供复用）**：本轮我**放弃了 `sys.meta_path` finder**，改成**后台线程轮询 `sys.modules`**
（`usercustomize` 里 60 行），因为 `044` 记过 finder 的两种坑（多 target 共用被永久摘除、`PathFinder.find_spec` 绕过别人的重定向）。
轮询版的代价只是"补丁在模块 import 之后几十毫秒生效"（首个请求在起服后 ~12 s 才发生）——
**在大模型起服这种"模块先 import、请求后到来"的场景里，这个 trade 是划算的**。**【实测：两条横幅都打出来了，targets 都 patched】**

★ **诊断臂的定位（写清以免被误读）**：它**不改变 §5 的选型判决**。
它的价值是**把"症状行"（`device_metadata.py:74`）推进到"泄漏的提交"（`submit#2`，draft 侧）**，
并为将来"把 KV8 的量化存取移植到 draft 面"那条路**留一张地图**。

---

## 5. ★★ "这条能不能替代 ②c"——**明确判断**

| 判据 | ②c（draft block 128→64，保 BF16） | ★ ②a（draft INT8） |
|---|---|---|
| Q1 图捕获 | ✅ 实测（`053`/`054`） | ✅ **本轮实测** |
| Q2 容量（tiny 档 D） | ✅ 实测 36,825（×1.5570） | ✅ **本轮实测 39,846（×1.6847）** |
| **Q3 投机解码（用户硬约束）** | ✅ **提案序列 4367/4367 逐条相同**（`054`） | ❌ **本轮无读数（引擎崩）** |
| 8 卡端到端 | ⏳ 排队 | ❌ **无从谈起**（单 die 都不通） |
| 精度风险 | **零**（dtype 不变） | draft 的 KV 被量化 ⇒ **接受率是唯一的实质风险**，且**现在无法证伪** |
| 交付可用性 | ✅ 现成 | ❌ **真实请求即崩** |

⇒ ★★ **结论：②a 不能替代 ②c。交付路线不变（②c 为序 1）。**
★ 而且这一条**不是"保守"**：②a 的 **×1.6847 vs ②c 的 ×1.5570**（tiny 口径；8 卡外推 ×1.6843 vs ×1.6009）在纸面上确实更优，
但它**在第一个 spec-decode step 就不可用** —— 这不是"收益打个折"，是"路线当前不成立"。

### 5.1 顺带回答任务书里的三条可判伪预测
1. "②a 的 `MeanAccLen/AvgDraftAcc/per-pos` 应与 BF16 基线在噪声内相同" —— **没测到**（§4）。
2. "若显著降接受率 ⇒ 价值要扣掉吞吐损失" —— **不适用**（没有读数）。
3. "输出 sha 应当不变" —— ★ **反而有一条正面旁证**：**基线臂**（同包、同 harness、同 salt）的输出
   `0ebccb55…` 与 `054` 那轮**四臂逐字相同** ⇒ harness 没漂。

★★ **一条对"将来怎么测 ②a 的 Q3"很有用的实测观察**：**基线臂自己的 `SpecDecoding` 就在大幅摆动** ——
同一次运行、相邻两个 10 s 窗口：
```
窗口 A: MeanAccLen 1.21  Accepted 10  Drafted 240  per-pos 0.208,0,0,0,0  AvgDraftAcc  4.2%
窗口 B: MeanAccLen 2.00  Accepted 48  Drafted 240  per-pos 1.000,0,0,0,0  AvgDraftAcc 20.0%
```
⇒ ★ **这些"四项指标"在 tiny 几何上是滑动窗口聚合量、绝对读数极不稳定**（`per-pos[0]` 从 0.208 蹦到 1.000）。
⇒ **要判 ②a 的 Q3，不能拿"某个窗口的四项数值相近"当证据**；`054` 用的**提案 token 序列逐条比对**（4367/4367）才是这条几何上有判别力的工具。
（这同时也解释了任务书里"统计功效"那条提醒为什么必须执行 —— 分母 `Drafted` 只有 240，而检验量本身在窗口间摆动 5×。）

### 5.2 对后续的**可执行建议**（如果还要救 ②a）

②a 与 ②c 的**收益差是 +8.2%**（tiny 口径）——值不值一条新实现，得看谁来做：
* **最小改动尝试（风险中）**：把 `dsa_v1.py` 的 draft 面接上同样的两平面机制
  （存：`kv8_store_rows` 的量化版 scatter；读：`kv8_ori_plane` 式的整页 gather+dequant）。
  但注意 `dsa_v1.py` 的 draft 路径走的是 **`.item()` 宿主同步**（`has_prefill` 判定），
  要图稳定必须**再走一遍 `049` 那套 `rows_bound` 改造** ⇒ **不是"接两个函数"那么小**。
* **更省的那条已在纸面上**：②a **与 ②c 是正交的**（一个改 dtype、一个改 block）。
  若 ②c 上线后还想再挤容量，**先测 ②c+②a 组合的页**（`64 × (512 + 4×2) = 33,280`，比 ②c 的 65,536 再减半），
  但**同一个运行期缺口会照样挡住它** —— 所以**先修 `dsa_v1` 面，再谈组合**。

---

## 6. 交付物与复跑

| 文件 | 作用 |
|---|---|
| `agents/D_draftINT8/scripts/patch_draft_int8.py` | ②a 的两处改动生成器（锚点计数 + AST + `py_compile`，**默认关**） |
| `agents/D_draftINT8/scripts/build_pkg.py` | 造影子包 + **"差异恰好 2 个文件"** 的机械门 + 发布件 md5 断言 |
| `agents/D_draftINT8/scripts/d_arm.sh` | 单臂（TIER=B/C/D × GRAPH × DRAFT_INT8，`PROBE=1` 开诊断探针） |
| `agents/D_draftINT8/scripts/d_chain.sh` | 串行链（`ddi-build → i8-eager → i8-graph → bf16-e`），rc=75 = 锁被占，退避不抢 |
| `agents/D_draftINT8/scripts/d_diag.sh` | 诊断臂（短几何 + 探针） |
| `agents/D_draftINT8/scripts/d_model.py` | 零参数容量模型（13 点对账，**可复跑**） |
| `agents/D_draftINT8/probe/usercustomize.py` | 诊断探针（submit/release 配对 + 首个异常） |

复跑（A3 上）：
```bash
bash ~/projects/dsv41-upstream-pr/agents/D_draftINT8/scripts/d_chain.sh
bash ~/projects/dsv41-upstream-pr/agents/D_draftINT8/scripts/collect_raw.sh   # → COS ddi/raw056.tgz
```

---

## 7. 状态（收尾）

1. ✅ **诊断臂已跑**（§4.4b）：泄漏点从"症状行"精确到 **`submit#2`（draft 侧的 1 任务提交，无配对 release）**；
2. ⏳ **"病行"仍缺**：下一个探针 target 应为 `models/layer/attention/layer.py::DSAAttention`
   （本轮钩错了类，见 §4.4b 末）；本轮**到此收尾，不再往 ②a 上加实验**；
3. ✅ `logs/README.md` 已登记本文件（编号 056，插在 `054` 之后，脚本 `scripts/register_log.py` 幂等可复跑）。
★ **诊断臂的定位（写清以免被误读）**：它**不改变交付结论**（②a 已经出局，交付推荐仍是 ②c）。
它的价值是 **把"症状行"换成"病行"**，并为将来"把量化存取移植进 `dsa_v1.py`"那条路**留一张地图**。
⇒ **跑完即收尾，不再往 ②a 上加实验。**

---

## 8. ★ 诚实边界（**这条结论的适用范围**）

* 本条的**全部实测都在单 die（A3 c1）的 tiny-draft 几何上**：`--load-format dummy`、TP=1、`max_model_len=8192`。
  **8 卡真权重 + 完整 131,072-token workload 一律未跑** ⇒ **标【未确认】**。
  ⇒ ★ **不要把"引擎死"这个结论过度外推到 8 卡**：它**可能**在 8 卡上以**不同的形态**出现
  （例如被别的错误先挡住、或者因为 batch 形状不同而根本走不到那条路）。
  单 die 能证明的是：**"②a 在至少一种生产形状下不可用"** —— 这已足以**否决它作为交付选项**，
  但**不足以断言**它在任何拓扑下都不可用。
* **②a 的 `SpecDecoding` 四项（`MeanAccLen` / `AvgDraftAcc` / `per-pos` / `Accepted|Drafted`）全部【未确认】** ——
  没有读数，**不是"没有退化"**。任务书里的三条可判伪预测（§5.1）同样**未被检验**。
* ②a **与 ②c 的组合**（draft 页 `64 × (512 + 4×2) = 33,280`）**未测**；且**同一个运行期缺口会照样挡住它**。
* 8 卡容量（档 D ×1.6843 / 档 C ×1.4650）是**模型【外推】**，不是实测。

---

## 9. ★★ 一条可复用的方法论（本轮**真正的关键动作**）

本轮我**连续踩了两个"构造缺陷"**，两次都**没有**把它报成"机制失败"：

| # | 构造缺陷 | 症状 | 我是怎么把它和"机制失败"分开的 |
|---|---|---|---|
| 1 | 探针用了 `os`，但 `dspark.py` 头部**没有** `import os` | 起服即死：`NameError: name 'os' is not defined`（`dspark.py:65`） | 栈帧落在**我自己的探针行**上；修完 `import os` 后同一包**起服成功** ⇒ 是包的问题 |
| 2 | 默认开了 `VLLM_V41_KV8_PREFILL=1`，而 base（`pkg-ring`）里**没有** `kv8_prefill_triton.py` | 起服即死：`ModuleNotFoundError` | 缺的是**别人的文件**、且与本实验的变量（draft 页几何）**无关** ⇒ 关掉后起服成功 |

★ **可复用的判据（建议进 `AGENTS.md §5b`）**：

> **新臂第一次失败时，先问"这是机制失败，还是我的包/探针/开关有问题？"**
> 分开的办法不是读日志猜，而是**跑一条"同包同参数、只换一个变量"的对照臂**。
> * 若**对照臂也失败** ⇒ 是 harness / 包 / 环境；
> * 若**对照臂全绿而实验臂失败** ⇒ 才是**该变量特有的**，可以下结论。
>
> 本轮正是靠 `DRAFT_INT8=0` 的对照臂（跑在同一包里、同一批服务参数上）
> 把"harness 问题"与"②a 特有"**分开**的 —— **这是本轮能给出判决的关键动作**，
> 而且它顺带给了"harness 没漂"的正面证据（基线输出 `0ebccb55…` 与 `054` 四臂**逐字相同**）。
