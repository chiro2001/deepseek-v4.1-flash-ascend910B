# 048 · int8（档 C / 档 D）的 **8 卡真权重 + 图模式**验证

> ★★ **头条：图模式下落不了地**（捕获期 `EE1016`）；**档 C 在真权重上的容量增益 = ×1.0000**（tiny 是 ×1.4655）。
> 见 §2 与 §3。

**日期**：2026-09-22 09:35 –（进行中）
　**执行**：子代理 **R_8card_int8**
**机器**：A3（A3-node1）**Phy-ID 8–15**（8 张；runner 用 `locks/c0.lock` 的 `flock` 全程持锁）
**镜像/模型**：`quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3`、
`~/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq`（**真权重**）
**起点**：`047`（int8 单卡 tiny 全绿；§11 标了三格未测）、`027`（8 卡口径）、`042`（L1 的 8 卡挂载链）、
`044`/`045`（ring 页共享）
**标记**：【实测】/【推断】/【未确认】
**原始数据**：`logs/raw/048-int8-8card/`（+ `logs/raw/048-int8-8card-raw.tgz`）；代码：`agents/R_8card_int8/`

## 0.0 ★★★ 收口结论（主代理 2026-09-22 11:1x 采纳）

```
① 档 C + S_graphfix 的 dsa_v41.py patch（图模式）           ★ 可以上线
     判据：四条判据全中、真命中、BlockRemoved:CPU=0、
           输出 sha 与 eager【逐字相同】、mode3 标量【未被冻结】
② 档 D（+ KV8 双平面 + prefill）                             ⏳ 待验
     还差：图模式四条判据（带 S 的 patch）+ "静默读错"那一格（S 在查）
③ ②c（draft block 128→64，保 BF16，×1.8177、保留 spec、无精度风险）  ⏳ T_draftceiling 在做
```
★ **容量说法必须按真权重写**：**档 C = ×1.0000**、**档 D = ×1.1356**
（**不能沿用 tiny 的 ×1.4655 / ×1.9133**——差别的全部原因是真权重多一个 draft 组，§3.2）。
★ **用户已定：保留投机解码** ⇒ ⑤a（关 spec）出局。
★ **A2 部署清单见文末 §11**。

---

## 0. 头条（先给判断）

| # | 问题 | 结果 |
|---|---|---|
| **①** | **int8 能在 `FULL_DECODE_ONLY`（生产图模式）下跑吗？** | ★★ **未打补丁：不能**（【实测】捕获期 `EE1016`，8 rank 一致，栈指向 `dsa_v41.py:436`，见 §2）。**★ 打上 `S_graphfix` 的修复（`logs/049`）：能**，且**四条判据全中、输出 sha 与 eager 臂逐字相同、mode3 标量未被冻结**（【实测】，见 §2.5）。 |
| **②** | **档 C / 档 D 在 8 卡真权重上的容量增益？** | ★★ **档 C = ×1.0000（零）**（`427,643` 与档 B **逐字相同**；tiny 是 ×1.4655）；**档 D = ×1.13558**（`485,610`）。根因都是**真权重多一个 draft 组**把 slot0–2 的页顶住，档 D 的增量全部来自 **slot3**（无 draft 的那一个）（§3.2）。 |
| **③** | **那 int8 在 8 卡上还剩什么收益？** | ★【实测】**宿主实占 197.21 → 150.01 GiB（1.314×）**；四条判据全中、真命中、`BlockRemoved:CPU = 0`（§3.1/§3.3）。 |
| **④** | **mode3 的 host 标量会被图捕获冻结吗？** | 图模式下**捕获更早就死了** ⇒ 拿不到读数（【未确认】）。但 **eager** 下热路径 trace 显示 `align_unit=1024` **每次请求现读、值随请求变**（131072→130048 / 65536→64512）⇒ 没有冻结迹象（§4）。 |
| **⑤** | **tiny 全绿为什么真权重会炸？** | ★ 两个**真权重独有**、tiny 结构性测不出的问题：**draft 组顶爆槽位页**（§5.1）+ **8 卡链自挂 `model.py`**（§5.2）。都已修好并通过起服。 |
| **⑥** | `concurrency > 1`（`MAX_SEQS=4`） | ⏳ 见 §7.2（时间预算内尽力） |
| **⑦** | **A2 部署清单（要挂哪几个件、各自 md5）** | ★ 见 **§11**（已收口） |

---

## 1. 接线（不占卡；主代理给的 8 卡件 md5 一致）

### 1.1 scheduler：以主代理那份 8 卡件为 base，**只加 trace**

| 项 | 值 |
|---|---|
| base | `a2/publish/0001-8card-offload-scheduler.patch.py`，md5 **`f3a7a0053fc6c639150fdde2a2509a63`** ★ 与主代理 09:39 给的一致 |
| 产物 | `agents/R_8card_int8/patched/scheduler.py`，md5 **`23c3a05c0bfe1adb068cc657eb2a1b29`**（2045 → 2104 行） |
| 唯一新增 | `[R8-INT8-TRACE]`（`R8_APC_TRACE=1` 打开，默认关） |
| ★ 构造性证明 | 剥掉 trace 后与 base **逐字节相同**（`patch/build_scheduler.py` 里 assert，可复跑）；`[APC_ALIGN]` 的 **8 个锚点**逐字保留 |
| ★ 零接触 | **没有覆盖** `agents/L3_8card/patched/scheduler.py`（别人的入口）——用 `OFFLOAD_SCHED_FILE=<我这份>` 挂 |

### 1.2 影子包的三处改动（`scripts/patch_serve_a2_int8.sh`）

| # | 改动 | 为什么 |
|---|---|---|
| ① | `[R8-INT8-v2]` 挂载块（`R8_INT8_PATCH=1`）：7~8 个 int8 文件 | 见 §1.3 |
| ② | `inner.sh` 里 **7 行 export**（`VLLM_V41_KV8 / KV8_SWA / RING_FP16 / KV8_PREFILL / APC_ALIGN` + `R8_APC_TRACE[_LIMIT]`） | ★【实测】**影子包完全不认 `VLLM_V41_*`**（`grep -c` = **0**）⇒ 不进 `inner.sh` 就是**静默 no-op**（`042 §1.4` 的"两半"） |
| ③ | `GRAPH/EAGER` 改成宿主可控（`R8_GRAPH` / `R8_EAGER`） | ★★ 见 §6.1：不改这里，"去掉 `--enforce-eager`"根本不会发生 |
| ④ | 生产 `model.py` 的挂载点：int8 臂换成**合并版** | ★ 见 §5.2 |

### 1.3 实际挂进容器的 int8 文件（`docker inspect` 实测）

```
R_8card_int8/patched/scheduler.py            = 8 卡件 + [R8-INT8-TRACE]
R_8card_int8/patched/deepseek_v41_slots.py   = ★ draft-aware 槽位容量（§5.1，本轮新修）
R_8card_int8/patched/model_merged.py         = ★ 生产 model.py + KV8 三处 hunk（§5.2，本轮新合）
X_integrate/pkg-ring/.../core/kv_cache_interface.py
X_integrate/pkg-ring/.../models/deepseek_v41/compressor.py
X_integrate/pkg-ring/.../ops/triton/compressor/compressor_triton.py
X_integrate/pkg-kv8pf/.../attention/dsa_v41.py         = 带 role 分键那份（主代理裁决）
X_integrate/pkg-kv8pf/.../attention/kv8_prefill_triton.py（仅档 D）
```
**档 C = 7 个 / 档 D = 8 个**（【实测】，`docker inspect` 数出来的）。

### 1.4 自检三件套

| # | 件 | 结果 |
|---|---|---|
| 1 | `patch/build_scheduler.py` | base 指纹断言 + 反向剥离逐字节相同 + AST 顺序 ✅ |
| 2 | `scripts/selfcheck_int8.py` | ★ **39 PASS / 0 FAIL / 0 SKIP**；含 4 条**反例臂**：错用 `block_size=128` 当栅格 ⇒ 与真值不同；tiny 的 12 组 JSON 用到 13 组上 ⇒ **fail-closed**；全 `ratio=1` ⇒ 门关；`is_store_reachable_swa_chunk` 与 publish 逐字相同 |
| 3 | `run_arm_r8.sh` 起服自检门 | 12 项（含"**trace 代码已装载**"+"**压测后 trace 行数 > 0**"两道门，见 §6.4） |

---

## 2. ★★★ 判决点：int8 在 `FULL_DECODE_ONLY` 图模式下**落不了地**

### 2.1 现象（【实测】，档 C，8 卡真权重）

| 臂 | 结果 |
|---|---|
| 档 B（图模式，无 int8） | ✅ `Application startup complete`，四条判据全中 |
| **档 C（图模式，int8）** | ⛔ **捕获期失败，8 rank 同时** |

```
(Worker_TP0..7) ERROR — NPUGraph: capture failed: RuntimeError:
    LocalScalarDenseNpu.cpp:23 NPU function error:
    c10_npu::acl::AclrtSynchronizeStreamWithTimeout(copy_stream), error code is 107027
(Worker_TP0..7) Not_Supported(EE1016): Synchronizing a stream failed.
    Reason: Stream (stream_id=31) during the capture stage is not supported.
(Worker_TP0..7) rtStreamSynchronizeWithTimeout execution failed, reason=stream is captured
```
**Python 栈（8 rank 逐字相同）**：
```
worker/model_runner_v1.py:5594  capture_model
 → vllm_ascend/attention/dsa_v41.py:898  forward
 → vllm_ascend/attention/dsa_v41.py:711  _attention
 → vllm_ascend/attention/dsa_v41.py:797  _native_attention
 → ★★ vllm_ascend/attention/dsa_v41.py:436  kv8_ori_plane
```
**第 436 行 = 宿主同步**（那一段的注释自己就写着 `# Prefill: ... Eager only, hence the host syncs`）：
```python
pages_per_req = int(((lens - 1) // block_size - window_start // block_size + 1).max().item())
```

### 2.2 根因（【推断·强】，判据见 §2.3）

两处 int8 读路径都靠"**query 行数 == 请求数**"来分"decode 快路 / prefill 慢路"：
```
kv8_ori_plane :   if query_rows == num_reqs:      → 快路（device-side、capture-safe）
_kv8_cmp_plane:   if rows == num_reqs and per_req * block_size == topk:
```
而 **spec-decode 的 decode 批每请求有 `1 + num_spec_tokens` 行 query**（本配置 `num_spec_tokens=5` ⇒ **6 行/请求**）
⇒ `query_rows = 6 × num_reqs` ⇒ **快路判据不成立** ⇒ 走慢路 ⇒ 慢路里的 `.item()` 在捕获期做 host 同步 ⇒ `EE1016`。

### 2.3 判据（三条，都能独立复核）

1. **图捕获的批构成**：`model_runner_v1.py` 的 `initialize_cudagraph_keys` 段用
   `self.uniform_decode_query_len` 与 `cudagraph_dispatcher.get_capture_descs()` 的 `desc.num_tokens`
   —— vLLM 对 spec-decode 的 decode 图取 `num_tokens = batch × uniform_decode_query_len`，**不是** `batch`。
2. ★ **档 B 反例**：同一条链、同一 spec 配置、同一图模式（`GRAPH=1 EAGER=0`）**捕获成功、四条判据全中**
   ⇒ 差异只在 int8 的读路径（BF16 的 SWA 平面**不走** `kv8_ori_plane`：那里要求 `isinstance(ori_kv, (tuple, list))`）。
3. **代码自己的注释**：慢路写着 `Eager only, hence the host syncs` ⇒ 作者假定它只在 eager 的 prefill 里跑。

### 2.4 最小修法（草稿留档；**实际由 `S_graphfix` 落地**，见 §2.5）

```
A（推荐）：把快路判据改成"按 query 行数 / 请求数比"或"decode 期恒成立"的形式
B：慢路的两个 host 数换成 **Python 上界**
   kv8_ori_plane : pages_per_req = cdiv(cdiv(query_rows, num_reqs) + window, block_size) + 1
   _kv8_cmp_plane: nblocks = cdiv(metadata.attention.max_seq_len, block_size)
   （代码自己说多余的列"只复制一列、mask 不读" ⇒ 上界安全）
```
★ 草稿已写好 `agents/R_8card_int8/patch/patch_capture_safe.py`（默认关、产出**独立文件**、不覆盖任何人的文件），
但**本轮没有挂上去**——`S_graphfix` 正在改同一处，避免互相覆盖（`AGENTS §5b` 第 6 条）。
★ **这条不是 `047` 引入的**：`[APC_ALIGN]` 在调度器侧（EngineCore、host），不在被捕获的 forward 里。

### 2.5 ★★★ 打上 `S_graphfix` 的修复后：**图模式通过，且判据①逐字相同**

`S_graphfix` 用**同一个 runner**（`agents/R_8card_int8/scripts/run_arm_r8.sh`，所以口径逐字一致）
跑了档 C 的图模式臂 `agents/S_graphfix/out/sg-a-c-graph`（`GRAPH=1 EAGER=0`，rc=**0**）：

| 判据 | 档 B（图基线） | 档 C **eager** | ★ 档 C **图模式**（带修复） |
|---|---|---|---|
| `GPU KV cache size` | 427,643 | 427,643 | **427,643** ✅ 容量不变 |
| `fill sha` | `d524172f…` | `d524172f…` | **`d524172f9f5ae36806151ca2bb9d0a311bbe94fa5ad27f001be5c4ac962aa0af`** |
| **`replay1 sha`** | `fb4c59dd…` | `bc2e797ab069f09c…` | ★ **`bc2e797ab069f09ced706696385dcf8bbd208e1dcf7c8586d15b0248a54f332f`**（**与 eager 逐字相同**） |
| `CPU→GPU` | 21.52 GB | 2.1188968448e10 | **2.1188968448e10**（逐字相同） |
| `hits` / `queries` | 901,120 / 3,145,984 | 901,120 | **901,120**（逐字相同） |
| `BlockStored:CPU` | 29,436 | 29,436 | **29,436**（逐字相同） |
| `BlockRemoved:CPU` | 0 | 0 | **0** ✅ |
| replay vs fill TTFT p50 | 1428.8 vs 18226.7 | 1582.4 vs 19769.4 = 12.49× | **1608.2 vs 19880.0 = 12.36×** |
| 宿主实占 | 197.21 GiB | **150.01 GiB** | ⚠️ **302.56 GiB —— 口径不同，不可比**（见下） |

**判据④（"图污染账本"）：图模式下 trace 同样打出来了，33 行**：
```
align_unit=1024 (host 标量, 本次请求现读) tail_extra=N/A(本实现无此变量) env=3
  num_tokens=256     max_hit_size_tokens=0     -> 0      call=1
  num_tokens=131072  max_hit_size_tokens=130048 -> 130048 call=2..17
  num_tokens=65536   max_hit_size_tokens=64512  -> 64512  call=18..19   ← replay 轮
```
⇒ ★★ **在 `FULL_DECODE_ONLY` 下 `apc_align_unit` 仍是"每次请求现读、值随请求长度变"**
⇒ **没有"capture 冻结 host 值"的现象**（机制上也对得上：它在 EngineCore 的 `_lookup()` 里读，
不在被捕获的 forward 里）。

⚠️ **口径提醒（别把 302.56 GiB 当档 C 的宿主实占）**：`S_graphfix` 那条臂挂的是**他自己的**
`pkgs/pkg-kv8pf`（不是 `X_integrate/pkg-kv8pf`），`P2_COMP_JSON` / `OFFLOAD_BYTES` 也与我的臂**不保证逐字相同**
⇒ 302.56 GiB 是**他那条臂**的数。★ 我这条链上**同一套包 + 同一套 COMP** 的档 C 读数是 **150.01 GiB**。
⇒ **"图模式档 C 的宿主实占"要用同一套口径重跑一条**，本任务标 ⏳（排队）。

---

## 3. ★★ 档 C 的容量：真权重上是 **×1.0000**（tiny 是 ×1.4655）

### 3.1 三臂并排（【实测】，16 × 131072 → replay 65536，`OFFLOAD_GB=56`）

| 判据 | 档 B（基线，图） | **档 C（int8，eager）** | 差异 |
|---|---:|---:|---|
| **`GPU KV cache size`** | **427,643** | **427,643** | ★ **×1.0000（零收益）** |
| 张量数 / Σpage / `worker_kv_bytes_per_block` | 16 / 910,208 / 131,072 B | **20 / 832,128 / 131,072 B** | int8 的页变小了、但被 draft 顶住 |
| **宿主实占（8 rank；三条路径一致）** | 197.21 GiB | ★ **150.01 GiB** | ★ **1.314× 更省** |
| `BlockStored:CPU` | 29,436 | **29,436** | 逐字相同 |
| `CPU→GPU` | 21.52 GB | 21.19 GB | −1.6% |
| `hits` / `queries` | 901,120 / 3,145,984 | **901,120** / 3,145,984 | 逐字相同 |
| `BlockRemoved:CPU` | 0 | **0** | ✅ |
| `CPU→GPU > 0 且 hits > 0` | ✅ | ✅ | ★ 真命中（不是 `047 §4` 的冷算假阳性） |
| replay vs fill TTFT p50 | 1428.8 vs 18226.7 = **12.76×** | 1582.4 vs 19769.4 = **12.49×** | 同档 |
| `fill sha`（16 prompt） | `d524172f…` | ★ **`d524172f…`** | ★ **跨臂逐字相同** |
| `replay1 sha` | `fb4c59dd…` | `bc2e797a…` | ⚠️ **不能据此判 ❌**，见 §7.1 |

### 3.2 为什么 tiny ×1.4655、真权重 ×1.0000 / ×1.1356（**可算**）

**档 D 的实测（同一个 kv 初始化阶段打印，早于捕获 ⇒ 图模式臂也能读到）**：
```
r8-c2-tierD-graph  serve.log:  GPU KV cache size: 485,610 tokens   ⇒ ×1.13558
  张量数 = 25；pages=[32768,512,8192,128, 32768,512,8192,128, 32768,512,8192,128,
                      65536,1024, 16384,256, 131072,131072,131072,
                      65536,1024, 65536,1024, 65536,1024]
  ③ ★ 真实分量（8 rank 逐字相同）= [[0,2,3,4,5,6,7,8,9,10,11], [1,12]]
  ③ ★ 记账对账: worker_kv_bytes_per_block=131072  Σpage=800,896  ⇒ 6.110×
  ③ 宿主实占（单分量臂）：旧=43.154 GiB → 新=37.147 GiB（每 rank）= 1.16×
```
**逐槽的账**（`kv+index / state / swa×10 / draft` 四项取 max）：
```
slot0-2（有 state + draft）：  档 B max(73,856, 131,072, 131,072, 131,072) = 131,072
                             档 C max(73,856,  65,536,  66,560, 131,072) = 131,072  ← draft 独占 binding
                             档 D max(41,600,  65,536,  66,560, 131,072) = 131,072  ← 同上
slot3  （无 state/draft）：    档 B max(147,712, 131,072) = 147,712  ← long_kv+index 是 binder
                             档 C/D max(83,200, 66,560) =  83,200  ← ★ 只有这一格真的缩了
Σ = 540,928 / 540,928 / 476,416 B per block
 ⇒ 档 C ×1.0000（4 个 slot 里 3 个没变）；档 D ×1.13558（slot3 缩了 1.78×，摊到 4 个 slot）
```
★（逐槽算术由子代理 `T_draftceiling` 在 `logs/050` 独立完成，并与本任务的 4 个实测点**逐字闭合**：
tiny 33,295 / 43,469、8 卡 **427,643 / 485,610**。）

```
真权重的规格是 13 组（tiny 12 组）：多一个 g12 = DeepseekV41DraftSWASpec（DSpark draft）
draft 的 BF16 窗口面 = 128 token × 512 dim × 2 B = 131,072 B
plan_cache_slots 的槽位页 capacity = max(kv+index, Σstate, Σswa, …)
  档 B（全 BF16）：            max(73,856, 131,072, 131,072) = 131,072
  档 C（SWA int8 + ring FP16）：max(73,856,  65,536,  66,560) =  73,856   ← 本可以缩！
  但 draft 必须叠在同一个槽位页上，且 draft 的 spec 明确要求 BF16
  ⇒ 为了让 draft 装得下，capacity 必须抬回 131,072（§5.1 的修法）
  ⇒ ★ 页大小没变 ⇒ GPU KV cache size 不变
```
★ **`plan_cache_slots` 的 draft 分支在 tiny 上整段跳过**（`if slot_idx < len(draft)`）
⇒ **tiny 永远测不出这个抵消**。这是"tiny 全绿 vs 真权重 ×1.0000"的**全部原因**。

**20 张量的实测页表（8 rank 逐字相同）**：
```
pages=[65536, 8192, 128, 65536, 8192, 128, 65536, 8192, 128,
       131072, 16384, 256, 131072, 131072, 131072, 1024, 1024, 1024, 65536, 1024]
③ 结构: group→tensor_idx = [[0..11], [12,13,14], [0,15,3,16,6,17,18,19]×10, [12,13,14]]
③ ★ 真实分量（worker 侧真值，8 rank 逐字相同）= [[0,2,3,4,5,6,7,8,9,10,11], [1,12]]
③ ★ 记账对账: worker_kv_bytes_per_block=131072  Σpage=832128  Σ(page)×bpc=832128  ⇒ 6.349×
```
★ 与 tiny 那套 `[[0,2,…,11],[1]]` **不同**：真权重里 **g12（draft）与 g1（state）同属一个分量**。
⇒ ★ 对齐 `035 §4.2` 坑 1：**`P2_COMP_JSON` 必须与"张量数 + 真分量"匹配**；
**单分量兜底（`[[0,…,12]]`）只作探索臂**，交付臂用上面这个精确值。

### 3.3 那 int8 在 8 卡上还剩什么收益？

**宿主内存 197.21 → 150.01 GiB（1.314×）**。
机制【推断·强】：20 张量里被 int8 缩小的平面（`page 131,072→65,536`、`→1,024`）让 L1 的行空间变小
⇒ worker 物理池变小；而 `worker_kv_bytes_per_block` 不变 ⇒ **池的记账口径不变、宿主实占变小**。
★ 与 `042` 的"L1 = 1.9895×"是**正交**的两件事。

---

## 4. ★★ 判据④：mode3 的 host 标量（**eager 下有读数**）

`[R8-INT8-TRACE]` 在 `_lookup()` 的热路径上打**本次请求实际用到的值**（33 行，【实测】）：
```
align_unit=1024 (host 标量, 本次请求现读) tail_extra=N/A(本实现无此变量) env=3
  num_tokens=256     max_hit_size_tokens=0     -> 0      call=1        ← 冒烟
  num_tokens=131072  max_hit_size_tokens=130048 -> 130048 call=2..17   ← fill 轮 16 个请求
  num_tokens=65536   max_hit_size_tokens=64512  -> 64512  call=18..19  ← ★ replay 轮
```
**读法**：
* `align_unit` **恒 1024**（段栅格的正确值，与 `[SWA_trim] alignment_tokens=1024` 一致）；
* ★★ **同一个标量在不同请求上给出不同的下游值**（131072→130048、65536→64512）
  ⇒ 它是"**每次请求现读**"的量，**不是被冻结的旧值**；
* `tail_extra` 一栏打 `N/A`：★ **本实现（主代理内联的 8 卡件）里根本没有这个变量**
  （栅格对齐的落点就是 store 侧已保留的段尾 chunk，`is_store_reachable_swa_chunk` 一个字节没改）
  ⇒ **"`tail_extra` = 0"的证据是"符号不存在"，不是"它打出来等于 0"** —— 两种证据强度不同，必须写清。
* ⚠️【未确认】**图模式**下的同一读数：捕获更早就死了（§2）⇒ 拿不到。
  但 `align` 在 **EngineCore（host 调度器）** 里读，且 trace 显示它随请求变
  ⇒ 【推断·强】它不进被捕获的 forward；**要等 §2 修好后重跑图模式才能【实测】确认**。

---

## 5. ★★ 两个"tiny 测不出、真权重才炸"的问题（都已修好并通过起服）

### 5.1 draft 组把槽位页顶爆（`plan_cache_slots`）—— **已修，实测通过**

**现象**（档 C 图模式第一次起服）：
```
ValueError: Aurora DSpark geometry must match target SWA and fit its existing slot
  ← vllm_ascend/core/deepseek_v41.py::plan_cache_slots
```
**逐量**：`capacity = max(kv+index, Σstate, Σswa)`；int8 让后两项同时变小
⇒ 页缩到 **73,856** < draft 的 **131,072**。
**为什么 tiny 测不出**：tiny 的 group 表只有 12 组、**没有 draft 组**，
而 draft 的几何检查是 `if slot_idx < len(draft):` ⇒ **整段跳过**。
**修法**（`patch/patch_slots_draft.py`，+22 行）：
```
capacity = max(kv_bytes + index_bytes, aliases_max, draft_size)
```
语义上必须成立：**槽位页必须装得下每一个叠放在它上面的平面**；
draft 的 spec 在 `__post_init__` 里明确要求 `dtype == bfloat16` ⇒ **不能**把 draft 也 int8 化。
★ `R8_SLOT_TRACE=1` 时打出六个量；没有 draft 组时与旧行为**逐字等价**。
★ **副作用 = §3.2 的 ×1.0000**：这是"能起来"与"容量有收益"之间的取舍；本轮**选择能起来**（否则连数都拿不到）。

### 5.2 8 卡链自挂生产 `model.py`（`Duplicate mount point`）—— **已修，实测通过**

**现象**：`docker: Error response from daemon: Duplicate mount point: .../models/deepseek_v41/model.py`
**原因**：影子包 `serve_a2.sh:958` **自己**挂了一份 `patches/files/model.py`（`:rw`），
而 KV8 的接线**也在 `model.py` 里**（3 个 hunk：`swa_plane_kwargs()` / `long_kv_plane_kwargs()` / import）。
**修法**（`patch/merge_model.py`）：
```
用 difflib 现算「镜像原版 → pkg-kv8pf 版」的 3 个 hunk（n=3 上下文）
  → ★ 自证①：同一条 diff 回放到「镜像版」必须逐字节得到 pkg-kv8pf 版
  → ★ 回放到「生产版」：每个 hunk 的 old 文本必须恰好出现 1 次（否则 fail-closed）
  → 产物 = 1317 行、md5 6a1b78853103e57bacecb571694010a2、过 py_compile + AST 自检
```
**实测**：三个锚点全部唯一命中；`R8_INT8_PATCH != 1` 时**逐字原样**挂生产文件（回滚即恢复）。

---

## 6. ★★ 环境坑（**六个**，前四个会静默出错）

### 6.1 ★★ `serve_v2.sh` 只在 `GRAPH != 1` 时才看 `EAGER`
```bash
# shadow-pkg/scripts/serve_v2.sh:62-69
if [ "$GRAPH" = "1" ]; then
  ARGS+=(--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", ...}')
else
  [ "$EAGER" = "1" ] && ARGS+=(--enforce-eager)
fi
```
⇒ **要跑"去掉 `--enforce-eager`"的对照臂，必须同时给 `GRAPH=0 EAGER=1`**；
只给 `EAGER=1` 而 `GRAPH=1` ⇒ `--enforce-eager` **根本不会加上去**（你以为在跑 eager、其实在跑图模式）。
★ 本任务把这一处改成宿主可控（`R8_GRAPH` / `R8_EAGER`）并显式透传。

### 6.2 ★★ 影子包不认 `VLLM_V41_*`（"挂载机制"与"运行期开关"是两半）
`grep -c VLLM_V41 shadow-pkg/scripts/serve_a2.sh serve_v2.sh` = **0**
⇒ 只在 runner 里 `export VLLM_V41_KV8_SWA=1` 是**静默 no-op**；必须在 `inner.sh` 里 export（`042 §1.4` 同一个坑）。

### 6.3 ★★ 幂等补丁脚本的"已打过就跳过"会让**修好的块装不进去**
第一次就是这么被坑的：改了挂载块，运行时仍在用旧块 ⇒ 白起一次服。
**修法**：按**版本标记**（`[R8-INT8-v2]`）升级、旧块整体替换；并加**静态终检**
（收集脚本里所有 `MOUNTS+=(-v …)` 的目标路径查重复）。

### 6.4 ★★ 我自己的两道自检门写错了（**误杀全部 int8 臂**）
| 门 | 错在哪 | 修法 |
|---|---|---|
| `int8_mounts >= 8` | 档 C **正确地只有 7 个**（没有 prefill triton）⇒ 每次都被拒 | 按档位：C ≥ 7 / D ≥ 8 |
| `[R8-INT8-TRACE] 热路径行数 > 0` 放在**起服自检**里 | trace 只在 `_lookup()`（每次请求）里打 ⇒ **起服时必然 0** ⇒ **假阴性** | 起服只看"**容器里的 scheduler.py 有没有 trace 标记**"；"热路径行数 > 0"挪到**压测之后**的 gate |
★ 教训：**判据放在"还没有请求的时刻"上，恒 0 ⇒ 不是判据、是噪声。**

### 6.5 `set -u` 下 `${R8_SLOT_TRACE}` 未绑定 ⇒ 挂载块**中途死掉**
症状：块打印了半行就退出；随后我的"无重复挂载"检查因此**假阴性**（块没跑完，当然没重复）。
**修法**：块里所有宿主 env 一律 `${VAR:-默认}` + **静态检查**（扫 `${R8_*}` 是否都带默认值）。

### 6.6 `/dev/shm` 与 `/tmp`
全程逐次 `df -h /dev/shm`（宿主 1007 GiB、**每次 0%**）⇒ 本轮**没有**踩 `SemLock` 那个坑。
★ 本轮**误建过 3 个 `/tmp/*.py`**（`m_img/m_prod/m_kv8`），**已在 4 分钟内清掉**并改用
`~/tmp/20260922/R_8card_int8/`（违反 `AGENTS §1-#2`，如实记在这里）。

---

## 7. 诚实边界

### 7.1 ★★ 变长回放下 `fill sha == replay sha` **没有判别力**
本任务的臂都是 `131072 → replay 65536`（`027` 口径）⇒ 模型看到的前缀短一半 ⇒ 下一个 token 本来就该不同。
⇒ 正确性判据只有两条（主代理 2026-09-22 10:0x 裁决）：
```
① 跨臂同名轮次比较：档 B/C/D 的 fill sha 必须逐字相同、replay1 sha 必须逐字相同；
② 同长度冷算参考：池 1 MiB 臂的 replay1 sha。
```
★ `summarize_r8.py` 原本把 `replay_matches_fill` 当 ❌ ⇒ **已改**（现在按上面两条判，
并在表里显式写"变长回放下无判别力"）。
★ 实测：档 B 与档 C 的 **`fill sha` 逐字相同**（`d524172f…`）；`replay1 sha` **不同**
（`fb4c59dd…` vs `bc2e797a…`）—— 后者**不能**据此判 ❌（`037`：同臂 cold-vs-cold 都会抖），**要等冷算参考臂**。

### 7.2 未确认清单

| # | 项 | 状态 |
|---|---|---|
| 1 | **图模式下的 host 标量读数** | ⚠️ **未确认**：捕获更早就死了（§2） |
| 2 | **档 D 的 8 卡结果** | ⏳ 进行中 |
| 3 | **冷算参考臂（正确性）** | ⏳ 排队中 |
| 4 | `concurrency > 1`（`MAX_SEQS=4`） | ⏳ 待办 |
| 5 | 档 C（`dsa_v41.py` **不带** role 分键）的 A/B | ⏳ 待办（主代理裁决：档 C 也用带 role 键那份；原样版留作对照） |
| 6 | `int8 + 图模式` 修复后能否通过 | ⚠️ **未确认**（`S_graphfix` 在做） |
| 7 | 8 卡上的 **KV 级逐字节比对** | ⚠️ 未做（`kv_bytecheck.py` 仍未跑） |
| 8 | 档 D 的容量是否也 ×1.0000 | ⏳ 进行中（预计同因：draft 顶住槽位页） |

---

## 8. 复现

```bash
# 0) 本机：生成 8 卡链要挂的 scheduler（= publish 8 卡件 + [R8-INT8-TRACE]）
python3 a2/agents/R_8card_int8/patch/build_scheduler.py

# 1) 本机 → a3（走 COS）
bash a2/agents/R_8card_int8/scripts/upload.sh
ssh A3-node1 'bash -s fetch' < a2/agents/R_8card_int8/scripts/upload.sh

# 2) a3：不占卡自检
ssh A3-node1 'cd ~/projects/dsv41-upstream-pr && python3 agents/R_8card_int8/scripts/selfcheck_int8.py'

# 3) a3：接线（幂等；只改影子包，先备份 .R_8card_int8.bak）
ssh A3-node1 'cd ~/projects/dsv41-upstream-pr && bash agents/R_8card_int8/scripts/patch_serve_a2_int8.sh'

# 4) a3：★ 必须先 dry-run（查重复挂载点 + 看到 [a2-dry] OK）—— 本轮踩过两次 Duplicate mount
ssh A3-node1 'cd ~/projects/dsv41-upstream-pr && env DRY_RUN=1 ... bash shadow-pkg/scripts/serve_a2.sh \
  | grep -E "a2-dry] OK|MOUNTS\("'

# 5) a3：链路（eager 口径；图模式见 §2）
ssh A3-node1 'cd ~/projects/dsv41-upstream-pr && setsid nohup env SUF=e1 MODE=eager DSA_SRC=D \
  COMP_EXACT="[[0,2,3,4,5,6,7,8,9,10,11],[1,12]]" COLD=1 \
  bash agents/R_8card_int8/scripts/chain_int8.sh > agents/R_8card_int8/logs/chain.log 2>&1 < /dev/null &'

# 6) 压行
python3 agents/R_8card_int8/scripts/summarize_r8.py --dir <out>/<tag> <tag>
```

### 开关

| env | 默认 | 含义 |
|---|---|---|
| `TIER` | `B` | `B`=无 int8（基线）/ `C`=SWA int8 + ring16 / `D`=+KV8 双平面 + prefill |
| `GRAPH` / `EAGER` | `1` / `0` | ★ 生产口径；eager 必须 `GRAPH=0 EAGER=1`（§6.1） |
| `DSA_SRC` | `auto` | `C`=原版（scratch key 不含 role）/ `D`=带 role 分键 |
| `COMP_JSON` | 按档 | ★ 必须与真分量匹配（§3.2）；单分量只作探索臂 |
| `SLOTS_FIX` / `SLOT_TRACE` | `1` / `1` | §5.1 的修与诊断 |
| `R8_APC_TRACE` | `1` | `[R8-INT8-TRACE]` 热路径读数（§4） |
| `OFFLOAD_BYTES` | `60666413056` | 56.5 GiB 记账 = 57,856 unit（`027`/`042` 口径）；冷算参考臂用 `1048576` |

---

## 9. 产物清单

| 路径 | 作用 |
|---|---|
| `agents/R_8card_int8/patch/build_scheduler.py` | 生成 8 卡 scheduler（base 指纹 + 反向剥离逐字节相同 + AST 顺序） |
| `agents/R_8card_int8/patch/merge_model.py` | ★ 生产 `model.py` + KV8 三处 hunk（difflib 现算 + 自证） |
| `agents/R_8card_int8/patch/patch_slots_draft.py` | ★ draft-aware 槽位容量（§5.1） |
| `agents/R_8card_int8/patch/patch_capture_safe.py` | 图捕获安全补丁**草稿**（**未挂**；等 `S_graphfix`） |
| `agents/R_8card_int8/patched/{scheduler.py,model_merged.py,deepseek_v41_slots.py}` | 实际挂进容器的三份 |
| `agents/R_8card_int8/scripts/{patch_serve_a2_int8.sh,run_arm_r8.sh,chain_int8.sh,selfcheck_int8.py,summarize_r8.py}` | 接线 / 单臂 / 串行 / 自检 / 压行 |
| `agents/R_8card_int8/logs/` | 各臂的 `selfcheck.txt` / `arm.out` / `serve_a2.log` |
| `logs/raw/048-int8-8card/` | 各臂原始产物（`client.json` / `metrics_*` / `kv_events` / `pool_bytes` / `trace.txt`） |

---

## 10. 红线遵守

* 只用 **Phy-ID 8–15**；runner 用**同一把** `locks/c0.lock` 的 `flock` 全程持锁；
  每次起服前打印 `npu-smi`；**没碰** c1/c2、`dsv41-a3`、`mooncake-*`、`jitpgo-*`、别人的容器；
* **不手设** `ASCEND_RT_VISIBLE_DEVICES`；
* **没写** `upstream-v41/`；**没改镜像**（全部 `docker -v` 只读挂载）；
  只改影子包 `shadow-pkg/scripts/serve_a2.sh`（备份 `.R_8card_int8.bak`，与 L1 的改动**叠加而非覆盖**）；
* ★ **没覆盖任何别的子代理的文件**：`L3_8card/patched/scheduler.py` 一行未动（用 `OFFLOAD_SCHED_FILE` 指自己的）；
  自检脚本对 `pkg-ring/pkg-kv8pf` **只读**；
* 起服前逐次 `df -h /dev/shm`（全程 0%）；
* 跨机传文件走 **coscli**；ssh 只跑命令；
* 结论全部标 **【实测】/【推断】/【未确认】**；缺的格子标 `⏳` / `⚠️`，**不用相邻数字顶替**。

---

## 11. ★★ A2 部署清单（档 C，**图模式**；2026-09-22 收口）

### 11.1 必须挂的文件（**7 个**，`docker -v` 单文件只读）

| # | 容器内路径 | 宿主源（本任务口径） | md5 | 作用 |
|---|---|---|---|---|
| 1 | `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` | `agents/R_8card_int8/patched/scheduler.py` | **`23c3a05c0bfe1adb068cc657eb2a1b29`** | 8 卡件（`0001-8card` + `043` 的 bpc 修复 + `047` 的 `[APC_ALIGN]`）+ `[R8-INT8-TRACE]` |
| 2 | `vllm-ascend/vllm_ascend/core/deepseek_v41.py` | `agents/R_8card_int8/patched/deepseek_v41_slots.py` | **`9db8e27c01fb8d17811de17680b4a0d8`** | ★ **draft-aware 槽位容量**（不挂 ⇒ `plan_cache_slots` 直接 raise，§5.1） |
| 3 | `vllm-ascend/vllm_ascend/models/deepseek_v41/model.py` | `agents/R_8card_int8/patched/model_merged.py` | **`6a1b78853103e57bacecb571694010a2`** | ★ 生产 `model.py` + KV8 三处 hunk（§5.2） |
| 4 | `vllm-ascend/vllm_ascend/core/kv_cache_interface.py` | `agents/X_integrate/pkg-ring/shadow/vllm_ascend/core/kv_cache_interface.py` | `7e17f7ca…` | int8 KV spec / page 几何 |
| 5 | `vllm-ascend/vllm_ascend/models/deepseek_v41/compressor.py` | `…/pkg-ring/…/models/deepseek_v41/compressor.py` | `8a2be008…` | ring FP16 正式版 |
| 6 | `vllm-ascend/vllm_ascend/ops/triton/compressor/compressor_triton.py` | `…/pkg-ring/…/ops/triton/compressor/compressor_triton.py` | `9362e72e…` | 同上 |
| 7 | ★★ `vllm-ascend/vllm_ascend/attention/dsa_v41.py` | **`agents/S_graphfix/`** 的修复版（见 `logs/049`） | 见 `logs/049` | ★★ **不挂 ⇒ 图捕获期 `EE1016` 必炸**（§2） |

（档 D 再加 `attention/kv8_prefill_triton.py`，并改用 `pkg-kv8pf` 的 `dsa_v41.py` 基线 + S 的修复 ⇒ **待验**。）

### 11.2 必须进 `inner.sh` 的运行期开关

```
VLLM_V41_KV8_SWA=1        # 档 C 的必需项（SWA 页 INT8）
VLLM_V41_RING_FP16=1      # 档 C 的必需项（state ring FP32→FP16）
VLLM_V41_KV8=0            # 档 C 不开 long-KV int8
VLLM_V41_KV8_PREFILL=0    # 档 C 不开 prefill 融合
VLLM_V41_APC_ALIGN=3      # ★ mode3（段栅格）；不设 ⇒ int8 会翻 token（047）
R8_APC_TRACE=1            # 可选：热路径打出 align_unit（诊断"图污染账本"）
```
★ **影子包的 `serve_a2.sh`/`serve_v2.sh` 完全不认 `VLLM_V41_*`**（`grep -c` = 0）
⇒ **必须在 `inner.sh` 里 export**，否则**静默 no-op**（§6.2）。
★ **`GRAPH=1 EAGER=0`** 才是生产口径（`FULL_DECODE_ONLY`）；
要跑 eager 对照必须 `GRAPH=0 EAGER=1`（§6.1）。

### 11.3 其余配置（与 `027`/`042` 逐字相同）

```
blocks_per_chunk = {"default": 8, "swa": 1}
cpu_bytes_to_use = 60666413056        # 56.5 GiB 记账 = 57,856 unit
P2_POOL_PATCH    = 1                  # L1（按组配额）
P2_COMP_JSON     = [[0,2,3,4,5,6,7,8,9,10,11],[1,12]]
      ★ 20 张量的【实测】真分量（8 rank 逐字相同，§3.2）；
        给单分量 [[0..12]] 只是"更保守"（行空间更粗、宿主更省、但可能影响命中），
        **交付按上面这个精确值**。
--max-model-len 133120 --max-num-seqs 32 --max-num-batched-tokens 8192
--prefix-match-unit 32 --kv-cache-memory-bytes 4294967296
```

### 11.4 上线监测判据（直接抄）

```
① vllm:kv_offload_block_removed_total{medium="CPU"} == 0      # 池没被撑爆
② (vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"} > 0)
   AND (vllm:external_prefix_cache_hits_total > 0)            # ★ 防"冷算假阳性"（047 §4）
③ GPU KV cache size == 427,643（档 C；图模式与 eager 逐字相同）
④ 起服日志里必须看到 [SWA_trim] alignment_tokens=1024 与 [R8-INT8-TRACE] align_unit=1024
⑤ 图模式起服**必须**在 capture 阶段无 EE1016（挂了文件 7 才有这个保证）
```

### 11.5 ★ 环境前提（两个会静默出错的）

```
★ 影子包的 serve_a2.sh 里，生产的 model.py 挂载点必须被 [R8-INT8] 的 ④ 改成"合并版"
  （见 §5.2；否则 docker 报 Duplicate mount point）；
★ serve_v2.sh 的 GRAPH/EAGER 必须能由宿主选（见 §6.1；否则你以为在跑 eager、其实在跑图模式）。
```
