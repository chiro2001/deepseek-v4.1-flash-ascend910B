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
① 档 C + S_graphfix 的 dsa_v41.py patch（图模式）           ★★ **可以上线**（已验完）
     判据：四条判据全中、真命中、BlockRemoved:CPU=0、
           图 vs eager 的输出 sha【逐字相同】（同 COMP/同包/同池）、mode3 标量【未被冻结】
② 档 D（+ KV8 双平面 + prefill）                             ★★ **图模式也已验完**（14:40 rc=0）
     判据同上（`485,610` / `29,436` / 12.11 GB>0 / 901,120>0 / 0 / 12.64× / 宿主 144.63 GiB）
     ⚠️ 仅剩：**"long-KV int8 面是否引入额外数值差异"** 未被**正面**判据覆盖（KV 级逐字节 ⏳ 未跑）
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
| **②** | **档 C / 档 D 在 8 卡真权重上的容量增益？** | ★★ **档 C = ×1.0000（零）**（`427,643` 与档 B **逐字相同**；tiny 是 ×1.4655）；**档 D = ×1.13558**（`485,610`，**eager 与图模式各读到一次、两次同值**）。根因都是**真权重多一个 draft 组**把 slot0–2 的页顶住，档 D 的增量全部来自 **slot3**（无 draft 的那一个）（§3.2）。 |
| **③** | **那 int8 在 8 卡上还剩什么收益？** | ★【实测】**宿主实占：档 B `197.21` → 档 C `150.01`（1.314×）→ 档 D `144.63 GiB`（1.363×）**；四条判据全中、真命中、`BlockRemoved:CPU = 0`（§3.1/§3.3）。 |
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

★★ **下面这一节先用的是 `S_graphfix` 的臂（口径不同）；同口径的正式对照见 §2.5b**。

| 判据 | 档 B（图基线） | 档 C **eager** | ★ 档 C **图模式**（带修复，`sg-a-c-graph`，口径不同） |
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

### 2.5b ★★★ 同口径的正式对照（本任务自己的图模式臂 `r8-g1-tierC-graph`）

**同 COMP（`[[0,2,…,11],[1,12]]`）+ 同包 + 同池 + 同 salt**，唯一变量 = 图 vs eager：

| 判据 | 档 C **eager r1** | 档 C **eager r2** | ★★ 档 C **图模式**（`r8-g1-tierC-graph`） |
|---|---|---|---|
| `GPU KV cache size` | 427,643 | — | **427,643** ✅ |
| `CPU→GPU` | 2.1188968448e10 | — | **2.1188968448e10**（逐字相同） |
| `hits` / `queries` | 901,120 / 3,145,984 | — | **901,120 / 3,145,984**（逐字相同） |
| `BlockStored:CPU` | 29,436 | — | **29,436**（逐字相同） |
| `BlockRemoved:CPU` | 0 | — | **0** ✅ |
| replay vs fill TTFT p50 | 1582.4 vs 19769.4 = 12.49× | — | **1598.7 vs 19724.7 = 12.34×** |
| **`replay1` 聚合 sha** | **`bc2e797a…f332f`** | **`bc2e797a…f332f`** | ★★ **`bc2e797a…f332f`（三臂逐字相同）** |
| `align_unit`（trace） | 1024，现读 | — | **1024，现读（33 行）** |
| ★ **宿主实占（三条路径一致）** | **150.01 GiB** | — | ★★ **150.01 GiB（逐字相同）** |
| ★ p6 / p9 | `'\n'` / `'_'` | `'\n'` / `'_'` | ★ **`'\n'` / `'_'`（回到 hot 家族）** |

⇒ ★★★ **判据①（图 vs eager 的输出 sha 逐字相同）在同口径下成立**；
判据②（容量不变）✅；判据③（四条判据不回归）✅；判据④（mode3 标量未被冻结）✅。
★ 这条臂同时把**图模式档 C 的宿主实占钉成 `150.01 GiB`**（与 eager 逐字相同）——
§11 的部署清单用的是这个数，**不是** `S` 那条口径不同的 `302.56 GiB`。

### 2.5c ★★★ 档 D 的同口径对照（`r8-f1-tierD-graph`，rc=0）

| 判据 | 档 D **eager** | ★ 档 D **图模式**（同 COMP/同包/同池） | 差异 |
|---|---|---|---|
| `GPU KV cache size` | **485,610** | **485,610** | ✅ 逐字相同 |
| `CPU→GPU` | 1.2105678848e10 | **1.2105678848e10** | ✅ |
| `hits` / `queries` | 901,120 / 3,145,984 | **901,120 / 3,145,984** | ✅ |
| `BlockStored:CPU` | 29,436 | **29,436** | ✅ |
| `BlockRemoved:CPU` | 0 | **0** | ✅ |
| replay vs fill TTFT p50 | 1464.2 vs 18795.4 = 12.84× | **1487.3 vs 18796.3 = 12.64×** | 同档 |
| ★ **`replay1` 聚合 sha** | **`8600507e…`** | ★ **`8600507e…`** | ✅ **逐字相同** |
| `fill` 聚合 sha | `d524172f…` | **`d524172f…`** | ✅ |
| 宿主实占 | **144.63 GiB** | **144.63 GiB** | ✅ 逐字相同 |
| `align_unit`（trace） | 1024，现读 | **1024，现读（33 行）** | ✅ |
| ★ p6 / p9 | `'\n'` / `' '` | **`'\n'` / `' '`** | ✅ **档 D 自身可复现** |

⇒ ★★★ **档 D 的判据①（图 vs eager 输出 sha 逐字相同）同样成立**；
⇒ 也说明 **档 D 在 p9 上的 `' '` 是它自己的稳定取值**（eager 与图两次一致），
与 §7.3.0d 的"`' '` 是 cold 家族取值"的解释相容。
★ **但档 D 的"long-KV int8 面是否引入额外数值差异"仍未被正面判据覆盖**（KV 级逐字节 ⏳ 未跑）
⇒ §7.2 的第 9 条保持 ⏳/⚠️。

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

| 判据 | 档 B（基线，图） | **档 C（int8，eager）** | **档 D（int8，eager）** |
|---|---:|---:|---:|
| **`GPU KV cache size`** | **427,643** | **427,643**（×1.0000） | ★ **485,610**（**×1.13558**） |
| 张量数 / Σpage / `worker_kv_bytes_per_block` | 16 / 910,208 / 131,072 B | **20 / 832,128 / 131,072 B** | **25 / 800,896 / 131,072 B** |
| **宿主实占（8 rank；三条路径一致）** | 197.21 GiB | ★ **150.01 GiB**（1.314×） | ★★ **144.63 GiB**（1.363×） |
| `BlockStored:CPU` | 29,436 | **29,436** | **29,436**（逐字相同） |
| `CPU→GPU` | 21.52 GB | 21.19 GB | 12.11 GB |
| `hits` / `queries` | 901,120 / 3,145,984 | **901,120** / 3,145,984 | **901,120** / 3,145,984 |
| `BlockRemoved:CPU` | 0 | **0** | **0** |
| `CPU→GPU > 0 且 hits > 0` | ✅ | ✅ | ✅（★ 都是真命中，不是 `047 §4` 的冷算假阳性） |
| replay vs fill TTFT p50 | 1428.8 vs 18226.7 = **12.76×** | 1582.4 vs 19769.4 = **12.49×** | 1464.2 vs 18795.4 = **12.84×** |
| `fill sha`（16 prompt） | `d524172f…` | ★ **`d524172f…`** | ★ **`d524172f…`**（三臂逐字相同） |
| `replay1 sha` | `fb4c59dd…` | `bc2e797a…` | `8600507e…` | ⚠️ 见 §7（**跨臂不同，但本口径下无判别力**） |
| ★ mode3 trace（`align_unit`） | —（档 B 不装） | **1024**，每请求现读 | **1024**，每请求现读 |
| ★ SpecDecoding（保留投机） | 1.50 / 10.0% | 1.50 / 10.0% | 1.00 / 0.0%（**见 §7.4 的口径说明**） |

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
| 6 | `int8 + 图模式` 修复后能否通过 | ✅ **已确认通过**（§2.5b/§2.5c：档 C 与档 D 的图 vs eager 输出 sha 都逐字相同） |
| 7 | 8 卡上的 **KV 级逐字节比对** | ⚠️ **未做**（可行性评估见 §7.3.0e：要重建 `transfer_async` 的指针表；现成件都不可用） |
| 8 | 档 D 的容量 | ✅ 已答：**×1.13558**（`485,610`，eager 与图两次同值） |
| 9 | ★ 档 D 的 long-KV int8 面是否引入额外数值差异 | ⚠️ **未确认**（`D-hot` 的 p9 = `' '` 与 cold 家族同侧、且 eager/图两次一致 ⇒ 【推断】属"路径性"差异；**正面判据仍是 KV 级逐字节**） |
| 10 | `MAX_SEQS=4`（`concurrency > 1`） | ⏳ 未做（卡时全给了判决臂） |
| 11 | 档 B 的 hot/cold 差异（**判据本身的失效证明**） | ✅ **已完成**：BF16 无损池 hot != cold（3/16）⇒ 见 §7.3.0c |

### 7.3 ★★ 冷算参考臂与"零成本判决"（【实测】）：热 vs 冷**有差异，但数据不支持"缺陷"**

#### 7.3.0 ★★★ 零成本判决：把 2 个差异 prompt 的输出**文本**并排打出来（主代理 13:1x 的 ①）

| prompt | 档 B hot（BF16 池，**图**） | 档 C hot r1（int8，eager） | 档 C hot r2 | **档 C cold**（无池） | 档 D hot（int8，eager） |
|---|---|---|---|---|---|
| **6** | `fill='這種'` → replay **`'\n'`** | `'這種'` → **`'\n'`** | **`'\n'`** | `'這種'` → **`' '`** | **`'\n'`** |
| **9** | `fill=' dialogue'` → replay **`'_'`** | `' dialogue'` → **`'_'`** | **`'_'`** | `' dialogue'` → **`' '`** | ★ **`' '`** |

★ **补一个读数**（同口径的图模式臂 `r8-g1-tierC-graph`）：**p6 = `'\n'`、p9 = `'_'`**
—— 与 hot 家族一致 ⇒ **档 C 的三种形态（eager×2 / 图）在 p6/p9 上完全一致**。
⇒ 悬着的**只有档 D 的 p9**（等 `D-graph` 给出第 2 个读数）。

**三条立刻能读出来的事实**：
1. ★★ **`fill` 轮在 5 条臂上逐字相同**（`'這種'` / `' dialogue'`）⇒ **int8 量化本身没有改变输出**
   （与 `035` 的结论一致）。
2. ★★ 差异只出现在 **replay 轮**，而且**全是"退化单 token"**（`'\n'` / `' '` / `'_'`），
   **没有任何一条"崩坏/重复/乱码"** ⇒ 按主代理的判据这是**第一种情形（边缘 argmax 翻转）**。
3. ★★ **档 B（BF16 池、无损）的热臂给出的是同样的 `'\n'` / `'_'`**，与档 C hot **逐字相同**
   ⇒ 这两个差异**不是 int8 引入的**。

★ **但必须如实标一个新疑点**：**prompt 9 上档 D hot = `' '`，而档 B/C hot = `'_'`**
⇒ 这一格**未确认**它是"边缘翻转"还是"档 D 的 long-KV int8 面引入的新差异"
（`S_graphfix` 提过档 D 有"静默读错"那一格，他正在查）。**不用相邻数字顶替。**

★★ **判据口径（必须写清，否则我们会用一个没判别力的判据）**：
本任务的压测是 **`max_tokens=1`** ⇒ 每请求**只生成 1 枚 token**，而候选之间是**近平局**
（输出全是空白类 token）⇒ **"1 枚 token 的 argmax 是否相同"在 int8 量化噪声下本来就极易翻转**。
⇒ **"热 vs 冷逐字相同"这条加强判据在 `max_tokens=1` 的 8 卡口径下【判别力不足】**；
它**不能**被读作"int8 已证明保真"，**也不能**被读作"int8 有缺陷"。
真正的保真判据仍是 **KV 级逐字节**（`kv_bytecheck.py`）—— 本任务 ⏳ **未跑**。

### 7.3.0b ★★ 口径纪律（主代理 2026-09-22 13:2x 定；与 `logs/062` 一致）

```
✗ 不要写："int8 已被证明保真"、"取回与全量重算逐字一致"
✓ 可以写："取回后自身可复现（16/16）；与全量重算的差异仅出现在极少数 prompt 的空白类 token 上"
✓ 必须写："真正的保真判据（KV 级逐字节，kv_bytecheck.py）本轮【未跑】"
```
★ 理由（**这是本文最重要的一句话之一**）：本任务手上只有**两条非回归证据**
（① `fill` 轮五臂逐字相同；② 热 replay 相对 **BF16 hot** 逐字相同），
**一条正面证据都没有**。它们足以**排除**"int8 引入了新差异"这个假说，
**不足以**把"保真"证出来。
★ 独立复核：主代理从原始 `client.json` 自己读了 `per_prompt_token_evidence`，**四臂全部对上**，
并独立确认 `r8-a-tierB-graph` 的 meta 里 `R8_KV8 / KV8_SWA / RING_FP16 / KV8_PREFILL` **四个开关全 0**
（= 真的是 BF16 无损池，承重前提成立）。详见 `logs/062`。

### 7.3.0c ★★★ 决定性实验（主代理的 ③）：**BF16 无损池的 hot 也 != cold** ⇒ 判据整体失效

**档 B（BF16，`R8_KV8 / KV8_SWA / RING_FP16 / KV8_PREFILL` 四个开关全 0）+ 图模式，唯一变量 = 池大小**
（与 `r8-a-tierB-graph` 同 tier、同 graph、同 COMP、同 salt、同 workload）：

| 臂 | 池 | `CPU→GPU` | `hits` | KV 事件 | replay1 聚合 sha | TTFT p50 |
|---|---:|---:|---:|---|---|---:|
| `r8-a-tierB-graph`（hot） | 56.5 GiB | 21.52 GB **>0** | **901,120 >0** | `BlockStored:CPU=29,436` | `fb4c59dd34d806…` | 1428.8 ms |
| ★ `r8-bc-tierB-graph-cold`（cold） | **1 MiB** | **0** | **0** | **无 `BlockStored:CPU`** | ★ **`947800ddaab1ff…`** | **8413.7 ms（5.9×）** |

⇒ ★★★ **BF16（无损）下 hot 与 cold 的输出 sha 也不同**（**3/16** 个 prompt 逐 prompt 不同）。
★ 冷臂自证齐备：`CPU→GPU = 0`、`hits = 0`、无 `BlockStored:CPU`、replay 慢 **5.9×**。

**⇒ 结论被彻底改写（比 §7.3.0 的读法更强）**：
```
① int8【没有】引入新差异：
     档 B hot 与档 C hot 逐字相同；档 C 的三种形态（eager×2 / 图）也逐字相同。
② "热 vs 冷逐字相同"这条加强判据【对被测对象没有判别力】：
   【BF16 无损池】同样给出 hot != cold（3/16）⇒ 差异来自
   「池取回路径」与「全量重算路径」这两条**不同的执行路径**，而不是 int8 的量化损失。
③ ⇒ 正确表述：
   "『取回 == 全量重算』逐字相同这条判据在本服务的这套口径下**不成立**（BF16 与 int8 都一样
    ⇒ 它测的是『路径』不是『保真』）。int8 的**非回归**由两条独立证据支撑：
    ① fill 轮各臂逐字相同；② 热 replay 相对 **BF16 hot** 逐字相同（同口径、同几何）。
    真正的**正面**保真判据（KV 级逐字节）本轮【未跑】。"
```
★ 也解释了 `047` 为何在 tiny 上拿到"replay sha == 冷算参考"：**单卡 tiny 的路由下两条路径恰好重合**；
**8 卡真权重 + 变长回放（131072→65536）** 下它们不重合（与"tiny 过、8 卡不一定过"同源）。

**两条路径各自的确定性（闭环）**：
```
C-cold r1 = cfac77743d575952…     C-cold r2 = cfac77743d575952…   ← 逐字相同（冷参考可复现）
hot r1 == hot r2（16/16 逐 prompt）                              ← 逐字相同（热路径可复现）
fill（B/C/D/cold 全部）= d524172f…                               ← 逐字相同
⇒ 差异是两条路径**之间**的稳定差异，不是 `037` 的随机抖动。
```

### 7.3.0d ★★ p6/p9 读数矩阵（7 个读数，档 D 的疑点被解释）

```
臂                        池              p6       p9
r8-a-tierB-graph        BF16 hot        '\n'     '_'
r8-bc-tierB-graph-cold  BF16 cold       '\n'    ' '     ← BF16 cold 也不同
r8-e1-tierC-eager (r1)  int8 hot        '\n'     '_'
r8-f1-tierC-eager (r2)  int8 hot        '\n'     '_'
r8-g1-tierC-graph       int8 hot + 图   '\n'     '_'
r8-f1-tierD-eager       int8 hot(+KV8)  '\n'    ' '     ← 与 cold 家族同侧
r8-e1-tierC-eager-cold  无池 cold       ' '      ' '
```
⇒ **`' '` 是 cold 家族的取值**（`'\n'`/`'_'` 是 B/C hot 家族的取值）；
⇒ ★ **档 D 的 p9 因此被解释掉**：它不是"档 D 的新缺陷"，而是"档 D hot 在这一格与 cold 同侧"，
与 §7.3.0c 的结论一致（**hot/cold 的差异本身就是路径性的**，`max_tokens=1` 近平局下极易翻转）。
★ `D-graph`（排队中）会给出档 D hot 的**第 2 个读数**以确认可复现性。

### 7.3.0e ★ 主代理给的判决路径（KV 级逐字节，**本任务未完成**）
```
1. ★★ KV 级逐字节比对（唯一能把"保真"与"近平局"分开的判据，且与 max_tokens 无关）—— **优先**
2. 若 KV 字节相同而输出仍不同 ⇒ 差异在下游（attention/算子的数值路径）
3. ✗ 不要靠"多跑几条 max_tokens=1 的 prompt"定案 —— 那是放大偶然性，不是提高判别力
```
★ 本任务**没有完成第 1 条**：8 卡链的 worker 真入口是 `NPUOffloadingWorker.transfer_async(job_id, src_spec, dst_spec)`
（描述符缓冲 + 批量拷贝），要逐字节比对必须重建它的指针表 —— 这超出了本轮剩余的卡时预算
⇒ **如实标【未确认】**（不用相邻数字顶替）。已有的探针资产与它们的已知洞：
`agents/F_fidelity/probe/f_pool_audit.py`（`036` 用过，12,928 次 mismatch=0；**盲点：只证"搬的字节没乱"、
不证"块被搬到了该在的行"**）、`agents/L3_8card/scripts/kv_bytecheck.py`（**不可复用**：它 hook 的
`store`/`load` 在现行代码里不存在 ⇒ 会打"已装"但比对 0 次，见 `agents/H_kvcheck/DESIGN.md` §1.1）。

#### 7.3.1 三组 sha 与冷臂自证

| 臂（档 C、eager、同几何、同 COMP，**唯一差别 = 池大小**） | 池 | `CPU→GPU` | `hits` | `BlockStored:CPU` | replay1 sha | replay TTFT p50 |
|---|---|---:|---:|---|---|---:|
| 热臂 `r8-e1-tierC-eager` | 56.5 GiB | 2.1188968448e10 **>0** | **901,120 >0** | 29,436 | `bc2e797ab069f09c…` | 1582.4 ms |
| **冷参考** `r8-e1-tierC-eager-cold` | **1 MiB** | **0** | **0** | **（无该键）** | **`cfac77743d575952…`** | **9154.9 ms（5.8×）** |

★ 冷臂的两条自证（证明它**真的**是冷参考）：`CPU→GPU = 0`、`hits = 0`、
KV 事件里**没有 `BlockStored:CPU`**、replay TTFT 是热臂的 **5.8×**（走整段重算）。
★ 两条臂的 `fill sha` **逐字相同**（`d524172f…`）⇒ 差异只出现在"回放时有没有池"这一步上。

⇒ **热 vs 冷的聚合 sha 确实不同** ⇒ 有两种解释，**必须区分**：
```
(a) int8 取回**不保真**（= 047 §4 那类缺陷的 8 卡版本）；
(b) `037` 的**服务层非确定性**（同 prompt 连发都会换 token）⇒ 单次比较**没有判别力**。
★ 区分办法（已排队，见 §7.4）：**热臂复跑** + **冷臂复跑**。
   若 热1 == 热2 且 冷1 == 冷2（各自可复现、两者不同） ⇒ (a)；
   若 热1 != 热2 或 冷1 != 冷2                ⇒ (b)，本格应标【未确认】。
```
★★ **【实测】热臂复跑已完成**：`hot r1 == hot r2` **逐字相同**（聚合 sha 都是 `bc2e797a…`，
**逐 prompt 16/16 相同**）⇒ **热路径完全可复现**（排除了"服务抖动"这个混淆项）。
★ 而 §7.3.0 的零成本判决显示**两个差异 prompt 在档 B（无损 BF16 池）上给出同样的 token**
⇒ **证据方向指向"判据判别力不足"，而不是 int8 缺陷**。
★ ★★ **档 B 的冷臂已完成**（`r8-bc-tierB-graph-cold`）：**BF16 hot != cold（3/16）**
⇒ 见 **§7.3.0c** —— 这条把整格彻底定性：**判据测的是"路径"不是"保真"，与 int8 无关**。

### 7.4 ★★ SpecDecoding 四项读数（用户 2026-09-22 硬约束：**交付配置必须保留投机解码**）

| 臂 | Mean acceptance length | Avg Draft acceptance rate | Accepted / Drafted | Per-position |
|---|---:|---:|---|---|
| 档 B（图） | **1.50** | **10.0%** | 1 / 10 tokens | `0.500, 0, 0, 0, 0` |
| 档 C（eager） | **1.50** | **10.0%** | 1 / 10 tokens | `0.500, 0, 0, 0, 0` |

★ 两臂**逐字相同** ⇒ int8 **没有改变投机解码的行为**。
★ 口径说明：本任务的压测是 `max_tokens=1`（只取 1 枚 token）⇒ 绝对值天然很低
（`Mean acceptance length 1.50` 来自 warmup 的多次采样），**只能作"保留投机"的证据**，
**不能**当作"投机收益"的性能数字。真正的吞吐对比需要长生成口径（**本轮未做**）。
★ ★ 凡本任务出现的"关 spec 能多拿容量"（`logs/050` 的 ⑤a = ×2.0188）一律标
**【诊断臂·已否决】**（用户已明确保留投机解码，仅作归因用）。

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
| `agents/R_8card_int8/patched/manifest.md5` | ★ 三份的 md5 + 各自的自证说明（发布件核对用） |
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

### 11.6 ★ 全臂一览（8 条臂，【实测】）

| 臂 | 档 | 图/eager | 池 | `GPU KV size` | `BlockStored:CPU` | `CPU→GPU` | `hits` | `rm:CPU` | replay/fill p50 | 宿主 | `fill sha` | `replay1 sha` |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| `r8-a-tierB-graph` | B | 图 | 56.5 GiB | 427,643 | 29,436 | 21.52 GB | 901,120 | 0 | 1428.8 / 18226.7 = **12.76×** | **197.21 GiB** | `d524172f…` | `fb4c59dd…` |
| `r8-e1-tierC-eager` (r1) | C | eager | 56.5 GiB | 427,643 | 29,436 | 21.19 GB | 901,120 | 0 | 1582.4 / 19769.4 = **12.49×** | **150.01 GiB** | `d524172f…` | `bc2e797a…` |
| `r8-f1-tierC-eager-r2` | C | eager | 56.5 GiB | — | — | — | — | — | — | — | `d524172f…` | ★ `bc2e797a…`（= r1） |
| ★ `r8-g1-tierC-graph` | C | **图** | 56.5 GiB | 427,643 | 29,436 | 21.19 GB | 901,120 | 0 | 1598.7 / 19724.7 = **12.34×** | **150.01 GiB** | `d524172f…` | ★ `bc2e797a…`（= eager） |
| `r8-f1-tierD-eager` | D | eager | 56.5 GiB | **485,610** | 29,436 | 12.11 GB | 901,120 | 0 | 1464.2 / 18795.4 = **12.84×** | **144.63 GiB** | `d524172f…` | `8600507e…` |
| ★ `r8-f1-tierD-graph` | D | **图** | 56.5 GiB | **485,610** | 29,436 | 12.11 GB | 901,120 | 0 | 1487.3 / 18796.3 = **12.64×** | **144.63 GiB** | `d524172f…` | ★ `8600507e…`（= eager） |
| `r8-e1-tierC-eager-cold` (r1) | C | eager | **1 MiB** | 427,643 | **（无该键）** | **0** | **0** | 0 | 9154.9 ms | 0.00 GiB | `d524172f…` | `cfac7774…` |
| `r8-f1-tierC-eager-cold2` | C | eager | **1 MiB** | — | — | 0 | 0 | — | 9133.2 ms | — | `d524172f…` | ★ `cfac7774…`（= cold1） |
| `r8-bc-tierB-graph-cold` | **B** | **图** | **1 MiB** | — | **（无该键）** | **0** | **0** | — | 8413.7 ms（5.9×） | 0.00 GiB | `d524172f…` | ★★ `947800dd…`（**BF16 hot 也 != cold**） |

**一句话读法**：
```
① 三条"热"臂（B/C/D）+ 档 C 的图模式臂：五条判据全中、真命中、rm:CPU=0；
② 档 C 的三种形态（eager×2 / 图）逐字相同；档 D 的两种形态（eager / 图）逐字相同；
③ 三条"冷"臂自证齐备（CPU→GPU=0 / hits=0 / 无 BlockStored:CPU / 慢 5.8~5.9×）；
④ ★★ **档 B 的冷臂与它的热臂也不同** ⇒ "hot==cold"这条判据测的是路径、不是保真（§7.3.0c）。
```

### 11.5 ★ 环境前提（两个会静默出错的）

```
★ 影子包的 serve_a2.sh 里，生产的 model.py 挂载点必须被 [R8-INT8] 的 ④ 改成"合并版"
  （见 §5.2；否则 docker 报 Duplicate mount point）；
★ serve_v2.sh 的 GRAPH/EAGER 必须能由宿主选（见 §6.1；否则你以为在跑 eager、其实在跑图模式）。
```
