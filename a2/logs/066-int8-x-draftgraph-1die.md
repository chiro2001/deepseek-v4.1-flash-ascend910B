# 066 — 【实测】int8（档 C/D）× `DRAFT_GRAPH=1` 单卡判决：**两档都跑通**，并抓到「档 D 在有前缀命中时 replay 翻 token」

> 2026-09-22 17:1x–17:3x CST（**远端 A3-node1 时间**；本机比远端快约 7 min）。执行：子代理 **`C2_int8_draftgraph`**。
> 机器：**只有 c2（die 7）**，全程 `tools/a3_chip.sh c2` 锁（**未出现退出码 75**；未碰 c0/c1、
> 未碰 Phy-ID 0–7 与 8–15、未碰 `dsv41-a3` / `mooncake-master` / 别人容器）；**未手设 `ASCEND_RT_VISIBLE_DEVICES`**；
> 每臂起服前查 `/dev/shm`；**未写 `upstream-v41/`**；**未发 PR / issue / 评论**。
> 模型：`agents/T_draftceiling/models/model-tiny-draft`（= tiny + 还原 draft 两键 ⇒ 有 g12 draft 组）+ `--load-format dummy`。
> 11 条臂：`out/`（7）+ `out2/`（2）+ `out3/`（2）。产物：本日志 + `logs/raw/066-c2-int8-draftgraph/` + `agents/C2_int8_draftgraph/`。
> 标记：**【实测】**= 有判别力的判据跑出来的数；**【推断】**；**【未确认】**。

---

## 0. 一句话结论

> ★★★ **档 C × `DRAFT_GRAPH=1` 与 档 D × `DRAFT_GRAPH=1` 在单卡上都【实测】跑通** ——
> 捕获期无 `EE1016`、无 `507057`、无 `invalid GM address`，`GRAPH_SAFE=0` 的反例臂**响亮地炸**（证明判据有判别力）。
> ★★ 而且这是**本仓第一次**在 A3 上真的把 draft 放进图里（此前所有标着 "graph" 的臂都是"主模型入图 + draft eager"）。
>
> ⛔ **但同时抓到一个必须带上的限定**：**档 D 在"有前缀命中"时 replay 翻 token**
> （`sha256_match=False | mismatched=[4,5,7]`，3 条臂 100% 复现，`PREFIX_MATCH_UNIT=128` 无效），
> 而 **档 B / 档 C 全部 `True`**。这与我的 overlay **缺 `APC_ALIGN` 实现**高度吻合
> （`047`/`048 §11.2` 明示「不设 `VLLM_V41_APC_ALIGN=3` ⇒ int8 会翻 token」）——
> ⇒ **这不是档 D 的算法缺陷，是缺必需件；但本轮没能闭环验证**（见 §6 边界第 3 条）。

---

## 1. 为什么这件事值得单独立一轮

```
A2 生产：DRAFT_GRAPH=1（实测 +62% 单流：54.7 → 88.7 tok/s）+ 要加 int8（档 C/D）
A3 全部 18 条 8 卡 int8 臂（027/042/048）：★ DRAFT_GRAPH 全是 0
⇒ 「int8 × draft 入图」这个组合**从未开过**
```

`logs/063` 已规范化：单卡**能**替代「软件路径 / 页几何 / 寻址 / 图兼容 / 输出不变」这一层，
而本任务要测的**全在这一层**（`053` 实测：tiny 单卡与 8 卡的捕获期错误签名**逐项逐字一致**）。

---

## 2. ★★ 本仓第一次真的把 draft 放进图里（判据链）

### 2.1 为什么以前没有过

```
stock dspark_proposer.py:75        self.use_cuda_graph = False          ← 硬禁
C2_draft64/pkg/dfix2c              dspark_proposer.py = dac256ad（stock）
DS_draft_graph_int8/pkg-dsi                            = dac256ad
X_integrate/pkg                                        = dac256ad
T_draftceiling/pkg/B                                   = dac256ad
T 的 run_2c_arm.sh:117              直接硬编码 DRAFT_GRAPH=0
D 的 d_arm.sh:61                    写死 SPEC enforce_eager:true
★ 实测取证：grep -rl DSPARK_CAPTURE_VALUE_FIX ~/…/agents/  =  **空**
  （阳性对照：grep -rl DSPARK_GRAPH_CAPTURE_METADATA 能命中 L2_final/T_draftceiling 的 inner.sh
   ⇒ 说明 grep 本身有效，是那三件 draft 版**源码**从未上过 A3）
```

### 2.2 本轮用的 overlay（`scripts/mkoverlay.sh`）

base = `C1_graph1die/pkg-gs`（= `X_integrate/pkg-ring` + `S_graphfix` 的 `dsa_v41.py` `94aeebb7…`
 + `R_8card_int8` 的 draft-aware 槽位补丁），再叠 **4 件整文件**：

| # | 文件 | md5 | 来源 |
|---|---|---|---|
| 1 | `attention/dsa_v41.py` | `94aeebb757d6d5708268754481a05e0a` | 图安全版（档 C/D 必需）|
| 2 | `attention/dsa_v1.py` | `371bb023e2ecb97c01a484fb9b8cef05` | ★ draft 版（2420 行）|
| 3 | `spec_decode/dspark_proposer.py` | `5565afed64b7fe282fa9d622af7cf206` | ★ draft 版（`CAPMETA=2 / VALUE_FIX=3`）|
| 4 | `spec_decode/llm_base_proposer.py` | `a24076eb387e41eae1c97d724f853823` | ★ draft 版（`DISPATCH_FIX=2`）|
| 5 | `attention/kv8_prefill_triton.py` | `796d0ff6eda03716f31c9994b8d8b221` | 档 D 第 6 件（pkg-gs 原本缺）|

★ `mkoverlay.sh` 有**两层**自检：4 件 md5 + **3 项内容级断言**
（`VALUE_FIX≥1` / `DISPATCH_FIX≥1` / `CAPMETA≥1` / `dsa_v1 行数>2200`）⇒ 防"文件名对、内容 stock"（`AGENTS §5b #9`）。

### 2.3 判别力三角（★ 主代理 17:2x 要求的那条）

| 臂 | dspark 三件 | `DRAFT_GRAPH` | `Wrapping draft model` | `[dspark-graph-probe]` | `use_cuda_graph=True` | `runtime_mode=FULL` |
|---|---|---:|---:|---:|---:|---:|
| **`c2-tc-dg1`** | draft 版 | 1 | **1** | 19 | **19** | **9** |
| `c2-tc-dg0` | draft 版 | 0 | 0 | 19 | 0 | 0 |
| ★ **`c2-tc-dg1-stock`** | **stock** | 1 | **0** | **0** | **0** | **0** |

★ **三条判读**：
1. **`Wrapping draft model` 是一条「跨版本可比」的独立判据** —— 它是 **vLLM 上游**打的日志，
   不是我们 patch 里的探针 ⇒ **stock 臂也能读**。只有「draft 版 + 图」才 = 1
   （与 `reports/two-corrections-draftgraph-gatehoist.md` 那条"全日志 0 次 ⇒ draft 从来没进过图"**用的是同一个判据**）。
2. **stock 臂的 `[dspark-graph-probe]` = 0** —— 那个 probe 是 draft 版文件加的，
   stock 里**根本没有这段代码** ⇒ 它无法给出 `use_cuda_graph=False:19`。
   ⇒ **主代理原本想要的"从 True:19 变 False:19"用 stock 拿不到**；上面那条 `Wrapping` 判据才是正解。
3. **`use_cuda_graph=True:19` 的可信度**同时由 `/proc/<pid>/environ` 回读
   （`DSPARK_GRAPH_CAPTURE_METADATA=1 DSPARK_CAPTURE_VALUE_FIX=1`）**和容量指纹**背书 ⇒ 不是"我打算跑什么"。

### 2.4 ★ 反面：`DSPARK_CAPTURE_VALUE_FIX=0` 这条对照在本几何下**没有判别力**

| 臂 | `VALUE_FIX` | A | Per-position | 结论 |
|---|---:|---:|---|---|
| `c2-tc-dg1` | 1 | 1.36 | `0.360,0,0,0,0` | — |
| `c2-tc-dg1-novf` | **0** | **1.42** | `0.421,0,0,0,0` | ⛔ **与上面同形** |
| `c2-tc-dg0`（eager） | 0 | 1.46 | `0.459,0,0,0,0` | — |

⇒ `A2_PACKAGE_SPEC` 里记的坏态是 **A≈1.07**，而这里三条臂的 A 全在 1.36–1.46、
**per-position 都是"只有 pos1 非零"** —— 这正是 `054` 记录的**单卡 tiny 的天然特性**
（`pos1–4` 接受率天然为 0）。⇒ **tiny 上测不出 `VALUE_FIX=0` 的坏态，这条对照作废**（如实标注，不当结论用）。

---

## 3. 11 条臂总表（全部【实测】）

| # | 臂 | 档 | `DRAFT_GRAPH` | `GRAPH_SAFE` | 其他 | ready | died | 捕获 | `EE1016` | `507057` | `kv_size` | `Wrapping` | sha |
|---|---|---|---|---:|---|---:|---:|---|---:|---:|---|---:|---|
| 1 | `c2-tc-dg1` | **C** | 1 | 1 | | 95 s | 0 | ✅ 23 s | **0** | 0 | 20,826 | 1 | `e27369ec…` |
| 2 | `c2-tc-dg0` | C | 0 | 1 | | 50 s | 0 | ✅ 5 s | 0 | 0 | 20,826 | 0 | **`e27369ec…`（与 1 逐字同）** |
| 3 | `c2-tc-dg1-novf` | C | 1 | 1 | `VALUE_FIX=0` | 50 s | 0 | ✅ 5 s | 0 | 0 | 20,826 | 1 | `e27369ec…` |
| 4 | **`c2-td-dg1`** | **D** | 1 | 1 | | 65 s | 0 | ✅ 18 s | **0** | **0** | **23,651** | 1 | ⛔ `81629185…` |
| 5 | `c2-td-dg0` | D | 0 | 1 | | 50 s | 0 | ✅ 6 s | 0 | 0 | 23,651 | 0 | ⛔ **`81629185…`（与 4 逐字同）** |
| 6 | `c2-tc-dg1-1m` | C | 1 | 1 | `MAX_LEN=1M KV=4G` | 70 s | **1** | ❌ | 0 | 0 | — | — | — |
| 7 | ★ **`c2-tc-dg1-legacy`** | C | 1 | **0** | | 50 s | **1** | ❌ | **7** | 0 | 20,826 | **1** | — |
| 8 | `c2-tc-dg1-1m2` | C | 1 | 1 | `MAX_LEN=1M KV=6G` | 60 s | 0 | ✅ 14 s | 0 | 0 | ★ **1,297,562** | 1 | `e27369ec…` |
| 9 | `c2-tc-dg1-stock` | C | 1 | 1 | **stock 三件** | 65 s | 0 | ✅ 5 s | 0 | 0 | 20,826 | **0** | `e27369ec…` |
| 10 | `c2-tb-dg1` | **B** | 1 | 1 | | 51 s | 0 | ✅ | 0 | 0 | 20,826 | 1 | ✅ `e27369ec…` |
| 11 | `c2-td-dg1-pmu128` | D | 1 | 1 | `PMU=128` | 51 s | 0 | ✅ | 0 | 0 | 23,651 | 1 | ⛔ `81629185…` |

★ 容量指纹（`AGENTS §5b #9`）：**B = C = 20,826** ⇒ **容量区分不了 B 与 C**，只能抓"该是 D 却拿到非 D"；
**D = 23,651** 与 `054` 的档 D 读数**逐字相同** ⇒ 档位自报可信。

---

## 4. 三个问题的答案

### 4.1 E1 — 档 C × `DRAFT_GRAPH=1` —— **PASS**【实测】

```
c2-tc-dg1 : capture_finished=1（23 s / 1.37 GiB）  EE1016=0  Not_Supported=0  capture failed=0
            sg_ppr_lines=24（capturing=True:12）  rows_bound=6 命中 24/24  ⇒ int8 的 graph_safe 分支在跑
            use_cuda_graph=True:19 / runtime_mode=FULL:9                ⇒ draft 真的在图里
            kv_size=20,826                                              ⇒ 档 C 指纹
            [SG-PPR] num_reqs=32 query_rows=192 max_query_len=6         ⇒ ★ 436 的击穿形状（与 053/048 逐字同）
```

### 4.2 E2 — 档 D × `DRAFT_GRAPH=1` —— **PASS**，且是**全新结论**【实测】

```
c2-td-dg1 : capture_finished=1（18 s / 2.10 GiB）  EE1016=0  Not_Supported=0  capture failed=0
            ★ err_507057_lines = 0        ← 049 §5.5.3 那个"第一个真实请求才崩"的坑**没有出现**
            ★ invalid GM address = 0      ← 三条计数器全 0（主代理 17:2x 明确要求的那三个）
            kv_size=23,651                ← ★ 档 D 指纹（与 054 逐字同）⇒ 长 KV 面 int8 真的生效
            use_cuda_graph=True:19
```
★ **档 D 的 `_kv8_cmp_plane` 是另一条路径**（窗口面修了它未必修）⇒ 这条是新的一格。

### 4.3 E3 — 1M 几何 —— **1M 装得下，且拿到每 token 的账**【实测】

```
第 1 条（KV=4 GiB）：died=1，工具自己给出门槛
  ValueError: To serve at least one request with the model's max seq len (1048576),
    (4.85 GiB KV cache is needed, which is larger than the available KV cache memory (4.0 GiB).
    Based on the available memory, the estimated maximum model length is 832896.
第 2 条（KV=6 GiB）：died=0，kv_size = 1,297,562 ≥ 1,048,576  ✅
```

★★ **两个独立读数交叉验证每 token 的 KV 账**：
```
4.85 GiB / 1,048,576 = 4,965.9 B/token
6.00 GiB / 1,297,562 = 4,965.9 B/token     ← ★ 逐字吻合（不是外推，是两条读数自洽）
```
⇒ **这是 A2 上线的一个独立门槛**：`--kv-cache-memory-bytes ≥ max_model_len × kv_per_token`
（单卡 tiny 上是 4,966 B/token；**A2 的绝对值必须用 A2 自己的读数**，倍率才可传递）。

### 4.4 反例臂 —— 判据有判别力（`AGENTS §5b` 第 3 条）【实测】

```
c2-tc-dg1-legacy（GRAPH_SAFE=0，其余全同）：
  died=1（50 s 死）  EE1016=7  Not_Supported=7  capture failed=1  capture_finished=0
  RuntimeError: … AclrtSynchronizeStreamWithTimeout(copy_stream), error code is 107027
  Not_Supported(EE1016): Synchronizing a stream failed.
    Reason: Stream (stream_id=34) during the capture stage is not supported.
```
★★ **与 `053` 的 `c1-tc-legacy` 逐字相同**（`EE1016=7` / `capture failed=1` / **`stream_id=34`**）——
那是 **DRAFT_GRAPH=0** 的臂；本轮是 **DRAFT_GRAPH=1**。⇒ **同一个 int8 图兼容缺陷，与 draft 入不入图无关**，
且 `GRAPH_SAFE=1` 的 5 条臂全部 `EE1016=0` ⇒ **对称跑过，不是假阳性**。

★ **另一条新细节**：反例臂的 `Wrapping=1 / probe=2 / rmFULL=0` ⇒ **draft 的图包装已完成，死的是 target 的图捕获**
（`runtime_mode=FULL` 一次都没到）。⇒ 精确定位了死亡位置。

---

## 5. ★★★ 新发现：**档 D 在"有前缀命中"时 replay 翻 token**

### 5.1 现象（3 条臂 100% 复现）

| 臂 | 档 | fill sha | replay sha | match | mismatched |
|---|---|---|---|---|---|
| `c2-tc-dg1` | C | `e27369ec…` | `e27369ec…` | ✅ True | — |
| `c2-tc-dg0` | C | `e27369ec…` | `e27369ec…` | ✅ True | — |
| **`c2-td-dg1`** | **D** | `e27369ec…` | **`81629185…`** | ⛔ **False** | **[4, 5, 7]** |
| **`c2-td-dg0`** | **D** | `e27369ec…` | **`81629185…`** | ⛔ False | [4, 5, 7] |
| `c2-tb-dg1` | **B** | `e27369ec…` | `e27369ec…` | ✅ **True** | — |
| `c2-td-dg1-pmu128` | D | `e27369ec…` | **`81629185…`** | ⛔ False | [4, 5, 7] |

### 5.2 三条判别性推理

1. ★ **档 B（无 int8）match=True** ⇒ mismatch **是 int8 特有**，不是我的负载/harness 问题（这条是决定性对照）；
2. ★ **档 D 两臂 + `PMU=128` 三臂给出完全相同的 replay sha** ⇒ **与 `DRAFT_GRAPH` 无关**、
   且**高度可复现**（不是数值抖动）；
3. **`PREFIX_MATCH_UNIT=128` 无效** ⇒ 不是命中粒度问题。

### 5.3 根因【推断】：缺 `VLLM_V41_APC_ALIGN`（我的 overlay 连实现都没有）

```
048 §11.2「必须进 inner.sh 的运行期开关」：
  VLLM_V41_APC_ALIGN=3      # ★ mode3（段栅格）；不设 ⇒ int8 会翻 token（047）
实测：
  ls Q_apcrecord/pkg/shadow/apc_record_fix.py          ← 实现存在（649 行）
  grep -rln VLLM_V41_APC_ALIGN agents/C2_int8_draftgraph/pkg/   = **空**
  grep -c   VLLM_V41_APC_ALIGN pkg/patch_pgp/pgp_scheduler.py   = **0**
  c2-tc-dg1 / c2-td-dg1 的 server.log 里 `APC_ALIGN` 横幅 = **0 行**
```
`047` 的症状描述与观察**逐条吻合**："命中长度 cap 跨在 ratio=2 的压缩组边界上 ⇒ replay 必须回读 state ring 的残余行……
⇒ int8 几何把它解读成 NaN ⇒ 翻 token"；且它能解释**为什么档 C 不翻**（档 C 不改 `compress_ratio` 的分组）。
⛔ **但这是【推断】，不是【实测】** —— 本轮**没有做**"叠上 `apc_record_fix` 后 mismatch 消失"的闭环实验（见 §6）。

### 5.4 ★★ 对主线的影响（要转告主代理）

> `A3-VALIDATION-ROADMAP.md` Phase 3 的判据表写：**"★ 输出不翻 token | 与同时段的 `KV8_*=0` 臂比…
> （`047` 的 mode3 保证）"**
> ⇒ ★ **实测校正**：**档 C 在这个几何下"侥幸"通过**（`PREFIX_MATCH_UNIT=32` + 512-token prompt 恰好落在栅格上），
> **档 D 必翻**。⇒ **Phase 3 的档 D 判据必须显式带上 `APC_ALIGN=3`，否则这条判据会给出假阳性（档 C）/ 假阴性（档 D）。**

---

## 6. 诚实边界（【未确认】的格子，不许用相邻数字顶替）

1. **档 D 的 mismatch 只定位到"缺 APC_ALIGN"这一级【推断】**：没有做"叠 `Q_apcrecord` 的 `apc_record_fix.py`
   + `VLLM_V41_APC_ALIGN=3` 后 mismatch 是否消失"的实验 ⇒ **闭环未完成**。
   ★ 但有一条**支持性证据**：`046`/`047` 的臂都在 **8 卡 + 卸载**上，而本条是 **单卡 + 无卸载** ⇒
   `apc_record_fix` 的 hook 目标（`kv_cache_manager.py:259` 等）是否在这条路径上被调用，**本轮未验证**。
2. **`c2-tb-dg1` 无法用容量自证是档 B**（B/C 同为 20,826）—— 只能靠 `/proc/<pid>/environ` 回读
   （`VLLM_V41_KV8_SWA=0 VLLM_V41_RING_FP16=0`，见 `out3/c2-tb-dg1.actual_env.txt`）。
3. **tiny + dummy 权重的绝对 sha / TTFT 不可外推**到 A2/A3 真权重（`063 §3.2`）。
4. **单卡的 `kv_size` 绝对值 ≠ A2**（倍率可传递，绝对值不可）。
5. **`DSPARK_CAPTURE_VALUE_FIX=0` 的坏态（A≈1.07）在本几何下测不出来**（§2.4）⇒ 那条对照**作废**。
6. **档 B/C/D 的图捕获都在 tiny 上过的**；真权重 8 卡仍需一轮（`A3-VALIDATION-ROADMAP` Phase 3）。

---

## 7. ★ 本轮我自己踩的三个坑（全部已自检固化）

### 7.1 ★★ 门槛用了绝对值 ⇒ **门恒假 + 链却报成功**

```bash
# 错：/dev/shm 的绝对可用字节 < 10 GiB ⇒ 拒绝起服
SHM_FREE=$(df -P /dev/shm | awk 'NR==2 {print $4}')
if [ "${SHM_FREE:-0}" -lt 10485760 ]; then ... exit 64; fi
```
但 `prbench-*` 容器的 `/dev/shm` **本身就是 64 MiB 的硬上限**（`shm 64M 24K 64M 1%`）
⇒ `avail` 恒 ≤ 65536 KiB ⇒ **门恒假** ⇒ **7 条臂全部被拒（0 条真跑），而 `chain.sh` 报 rc=0**。
**修法**：判据改成**占自身容量的比例**（`used/size ≥ 50%` ⇒ 拒绝），并把它打印进 verdict。
⇒ **教训（建议进 `a2/AGENTS.md §5`）：门槛要用「占自身容量的比例」，不要用绝对值。**

### 7.2 ★★ 链的"成功"必须由**臂账本**定义

7.1 之所以能当场抓住，是因为 `chain.sh` 加了账本：
```bash
if [ -s "$OUT/$tag.verdict.txt" ]; then ARMS_OK=…; else ARMS_NOVD=…; echo "$tag ⛔NO_VERDICT" >>ledger; fi
…
[ "$ARMS_NOVD" = "0" ] && [ "$ARMS_OK" != "0" ] || exit 4      # ★ 没产出 verdict ⇒ 链不是成功
```
⇒ 第二轮链**正确地报了 `rc=4`**（尽管它只是因为下一个坑而误报，见 7.3）。

### 7.3 `OUT` 未 `export` ⇒ verdict 落到**别的目录**

`chain2.sh` 里 `OUT=${OUT:-$G/out2}` 是 **shell 变量**，而 `run_arm.sh` 的 `OUT=${OUT:-…}` 只读**环境变量**
⇒ 回落默认 `.../out` ⇒ 账本报 `NO_VERDICT`，**而两条臂其实都真跑完了**。
★ 值得记的是：**账本（7.2）救了我一次** —— 它把"两臂没跑"和"两臂跑了但数据在别处"都标成需要人看的异常，
而不是静默通过。修法：`export OUT`。

### 7.4 `/tmp` 在容器里不可见（红线不许用，这里还额外证明了它**不管用**）

我一度把 helper 脚本写到宿主 `/tmp/fix_out2.sh` 再 `docker exec` 调用 ⇒ `rc=127 No such file or directory`。
⇒ **容器不共享宿主的 `/tmp`**（各自独立）。正确做法：写到**挂载进容器的路径**
（`~/projects/dsv41-upstream-pr/agents/<自己>/…` → `/work/agents/<自己>/…`）。

---

## 8. 复现

```bash
# 0) 本机 → A3（走 COS）
bash a2/agents/C2_int8_draftgraph/scripts/upload.sh
ssh A3-node1 'cd ~/tmp/20260922/c2_int8_draftgraph && \
  coscli cp cos://uploads-new/share/xfer/c2_int8_draftgraph/c2_int8_draftgraph.tgz ./c2.tgz && \
  tar xzf c2.tgz -C ~/projects/dsv41-upstream-pr/agents/C2_int8_draftgraph --strip-components=1'

# 1) 第一轮 7 臂（覆盖 E1/E2/E3 与两条反例），~9 min
ssh A3-node1 'cd ~/projects/dsv41-upstream-pr && bash tools/a3_chip.sh c2 --timeout 120 --name c2-int8-dg -- \
  bash /work/agents/C2_int8_draftgraph/scripts/chain.sh'

# 2) 第二轮 2 臂（1M 重跑 + stock 对照）
ssh A3-node1 'cd ~/projects/dsv41-upstream-pr && bash tools/a3_chip.sh c2 --timeout 120 --name c2-int8-dg2 -- \
  bash /work/agents/C2_int8_draftgraph/scripts/chain2.sh'

# 3) 第三轮 2 臂（档 B 判别 + PMU=128）
ssh A3-node1 'cd ~/projects/dsv41-upstream-pr && bash tools/a3_chip.sh c2 --timeout 120 --name c2-int8-dg3 -- \
  bash /work/agents/C2_int8_draftgraph/scripts/chain3.sh'

# 4) 原始数据
ls a2/logs/raw/066-c2-int8-draftgraph/{out,out2,out3}/
cat a2/logs/raw/066-c2-int8-draftgraph/out2/01-wrapping-ledger.txt
```

---

## 9. 卡与环境纪律

| 项 | 状态 |
|---|---|
| 用卡 | **只有 c2（die 7）**，全程 `tools/a3_chip.sh c2`；**退出码 75 一次都没出现** |
| c0 / c1 | **未碰**（c0 被 8 卡臂持有）|
| Phy-ID 0–7 / 8–15、`dsv41-a3`、`mooncake-master` | **未碰** |
| `ASCEND_RT_VISIBLE_DEVICES` | **未手设** |
| `/dev/shm` | 每臂起服前查（见 §7.1）；全部 `used=24K / 64M` |
| `/tmp` | **未用**（本机 `~/tmp/20260922/c2_int8_draftgraph/`；容器内可用路径见 §7.4）|
| `upstream-v41/` | **未写** |
| 容器 | 只 `docker exec`（经 `a3_chip.sh` 加锁）；**未改镜像里任何源码**（overlay 只在自己的 `/work/agents/C2_int8_draftgraph/`）|
| 臂收尾 | `run_arm.sh` 的 trap 杀自己起的 `vllm serve` 进程组；另加 `C2_ARM_MARKER` 精确清扫上一臂残留 |
