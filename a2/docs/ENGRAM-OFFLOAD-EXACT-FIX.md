# ENGRAM × 卸载：**零精度损失**的精确修复（设计 + 行号级证据 + 局部补丁草案）

> 2026-09-22 21:1x。作者：子代理 `engram_exact_fix_design`（**不占卡、不起容器**：
> 只读代码 + 写方案 + 写本地可跑的语义测试）。
> 依据：`a2/logs/072`、`a2/logs/073`、`a2/logs/071 §A1`，以及 `21e7d99`（降级修复）之后的代码现状。
> 标记：**【实测】/【推断】/【未确认】**。★ 全文不改任何生产文件 —— 补丁一律以
> `a2/agents/Engram_exactfix/patches/` 下的**独立文件**给出。

---

## 0. 判决（一句话）

**能做，而且是真正的零精度损失**：那条边界上缺的 1–3 个 token 的**真实 id 就在
runner 的 `input_batch.token_ids_cpu` 里**（进程内、CPU、零拷贝、已经是 batch `input_ids`
的同一份来源），把它发布给 model 侧、按**读者坐标**回填镜像即可 ——
**不需要改 numba kernel，不需要改卸载层，不需要 D2H，不增加池子内存**。

代价上界【实测·单测】：**每个请求每步 ≤ (lookback−1)·lookback/2 = 6 个槽位**
（与 batch 大小、上下文长度**无关**）。

---

## 1. 问题与机制（行号级）

### 1.1 镜像是什么：`(page, slot)` 就是**读者的坐标**

`PagedNgramHistory`（`dsv41-release/patches/files/engram_hash.py`）维护 `pages/present`
两级结构（JIT 路径是稠密 numpy：`_jit_pages` / `_jit_page_present`）。**读**与**写**用的是同一个式子：

| 位置 | 代码 | 说明 |
|---|---|---|
| 写（本步批量的 token） | `engram_hash.py:325` `page_indices = block_table[request_ids, positions // block_size]` | 只在**本步 batch 的 token** 上执行 |
| 写（落进镜像） | `engram_jit_kernel.py:102-114`（首写复位在 `:108-111`） | 首次写到的页**整行复位**为 −1 再写；`page<0/≥cap` ⇒ 早退报 `oob` |
| 读（历史窗口） | `engram_jit_kernel.py:135` `flat[i] = block_table[request_ids[row], blk]`（`blk=(pos−sh)//block_size`） | 小批分支 |
| 读（prefill 分支） | `engram_jit_kernel.py:199-206`（缺页 ⇒ `prs[j]=0` 在 `:207-213`） | 同一坐标：`page = block_table[request_ids[r], p // block_size]`、`offb = p % block_size` |

★ 关键推论【推断·强】：**只要"写的地方 == 读的地方"，修补就与页号语义无关** ——
无论 `page` 是 SWA 组的物理块号、还是 SWA-trim 之后的"段尾块号"，都不影响正确性。

### 1.2 为什么必然缺页：写者只覆盖"本步 batch 的 token"

`model.py:1103-1104`：进入 `update()` 的只有本步 batch 的 token
（`ids_host = input_ids[:n]`、`pos_host = positions[:n]`）；
`model.py:1108-1118` 是 `engram_history.update(...)` 的**唯一调用点**。

而被 DRAM 卸载池**取回**的前缀块，其内容是从 CPU 池 DMA 进 GPU 的 ——
**从未流经这个进程的 `update()`** ⇒ 镜像里那几页要么不存在、要么还是上一任占用者的内容。

于是窗口 `p−1, p−2, p−3`（`lookback=4`【实测·`engram_ref/official/config.json:46`
与 `engram_ref/stage2b-engram-draft-alignment.md:49`】）在两种形态下都会错：

* **缺页**（`present[page]==0`）：`21e7d99` 之前 ⇒ `raise KeyError(2486)`（`logs/073`）；
  之后 ⇒ pad（barrier）+ `pageless_history_rows` 计数。
* **陈旧页**（`present[page]==1` 但内容是别人的 token）：**今天仍然静默算错**，
  且 **`miss_rows=0` / `err=-1`，没有任何计数器会响**。← 见 §5 的本地单测场景 3。

### 1.3 现状（`21e7d99` 之后）在真机上的两个读数

| 场景 | 今天的历史 | 今天的计数器 |
|---|---|---|
| 边界正好落在**页首**（`logs/073` 的形态） | `[cur, pad, pad, pad]` | `miss_rows=1`（**可观测**） |
| 边界落在**页中**（同一页里跨过本步写入范围） | `[cur, pad, pad, pad]` | **`miss_rows=0`、`err=-1`** ⇒ 完全静默 |

⇒ 所以"pad 降级"不但有精度损失，**它的观测面还漏了一格**（本地单测 `场景 2` 复现）。

---

## 2. 关键问题 1：真 token id 到底拿不拿得到？

### 2.1 模型侧（`model.py`）能拿到什么

| 变量 | 位置 | 内容 |
|---|---|---|
| `meta.query_start_loc_cpu` | `model.py:1058-1061` | 本步各请求的 token 边界（**只有本步**） |
| `n = int(boundaries[-1])` | `:1062` | 本步 token 总数 |
| `requests`（行→请求） | `:1063` `repeat_interleave(arange(len(boundaries)-1), boundaries.diff())` | 每行的请求序号 = **runner 的 batch 行号** |
| `ids_host / pos_host` | `:1103-1104` | 本步 batch 的 token / 位置（`positions` 还要走一次 D2H） |
| `meta.block_table_cpu`、`meta.storage_block_size` | `:1112-1117` | 页表与页宽（`dsa_v1.py:541` `self.storage_block_size = kv_cache_spec.storage_block_size`） |

⇒ **模型侧只有本步的 token，没有前缀**。前缀的 token 它拿不到 —— 除非从 runner 拿。

### 2.2 runner 侧：完整 token 历史**就在手边**（这正是答案）

运行期文件已核对【实测】：A3 容器里
`/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py`
的 md5 = `9d84d9b073aeece3fd3bbc143ba20567`，与本仓
`op_line/draftsrc/model_runner_v1.py` **逐字节相同**（因此下面引用的行号可复核）。

| 事实 | 位置 |
|---|---|
| batch 的 `input_ids` **就是**从 `input_batch.token_ids_cpu` 里 gather 出来的 | `model_runner_v1.py:1248` `token_indices = positions_np + req_indices * token_ids_cpu.shape[1]`；`:1252-1259` `torch.index_select(...token_ids_cpu_tensor.flatten()...)` |
| 每行的**完整 token 历史** | `npu_input_batch.py:93-99` `token_ids_cpu_tensor = torch.zeros((max_num_reqs, max_model_len), int32, cpu)`、`token_ids_cpu = ....numpy()` |
| 每行**已提交长度**（不含投机 token） | `npu_input_batch.py:109-115` `num_tokens_no_spec`（= `request.num_tokens`，`gpu_input_batch.py:387`）；另有 `:108` `num_tokens`、`:123-129` `num_computed_tokens_cpu` |
| 采样后原地追加 | `model_runner_v1.py:2914-2921` `token_ids_cpu[req_idx, start:end] = sampled_ids` … `num_tokens_no_spec[req_idx] = end_idx`；`req_state.output_token_ids.extend(sampled_ids)` |
| 请求态（可选替代源） | `gpu_input_batch.py:36-58` `CachedRequestState.prompt_token_ids / output_token_ids / get_token_id(idx)` |
| `prepare_engram_inputs` 的调用点 | `model_runner_v1.py:2987-2989`（`_model_forward()` 内、capture/replay **之前**） |

★★ 三条推论：

1. **不需要 D2H**：`token_ids_cpu` 本来就是 CPU 上的表；
2. **不需要新数据结构**：`token_ids_cpu` 的**同一行序**就是模型侧的 batch 行序
   （两侧都由 `num_scheduled_tokens` 的顺序推出：runner `:1248` 的 `req_indices`，
   model `:1063` 的 `repeat_interleave`）；
3. **`q < num_tokens_no_spec` 就是"这个位置真的写过"**：`token_ids_cpu` 与
   `num_tokens_no_spec` 在 `add_request`（`gpu_input_batch.py:369-387`）与采样回写
   （`model_runner_v1.py:2914-2921`）里**成对更新** ⇒ 用这个界过滤掉未写位置，
   不会读到"上一任请求"的残值。

### 2.3 runner 文件能不能被影子包覆盖？

**能，而且是现成机制**（不是新发明）：

| 证据 | 位置 |
|---|---|
| 影子包生成器的挂载块（env 门控、**缺文件直接 die**） | `a2/scripts/make_shadow_pkg.sh:129-189`（`if [ "${OFFLOAD_*_PATCH:-0}" = "1" ] … [ -f ] \|\| die … MOUNTS+=(-v …)`） |
| 生产脚本里**曾经**就挂过 `model_runner_v1.py`（同一容器路径） | `dsv41-release/scripts/serve_a2.sh:821-825` 的 `[DMQ-GUARD-REMOVED]` 注释 —— 撤销的原因是**那个护栏本身会误触发**，不是"不能挂 runner" |
| runner→model 的交接点先例 | `model.py:288-302` `set_engram_host_inputs()`（由 `stage_perf/serve_bneck.sh:153-161` 挂 `probe_mirror/model_runner_v1.py` 调进来） |
| 起服前的 fail-closed 断言先例 | `a2/scripts/serve_a2_offload.sh:336-344`（两个开关必须是 1）、`:409-418`（DRY 时必须看到 4 个卸载补丁） |

⇒ 本方案把 runner 覆盖做成**幂等 + 可校验**：md5 钉死基线、门控默认关、
DRY/RUN 双门断言补丁命中次数，见 §6。

---

## 3. 精确修复设计（方案 A）

### 3.1 三个模式（env `VLLM_V41_ENGRAM_TRUE_TOKENS`）

| 值 | 行为 | 用途 |
|---:|---|---|
| **0**（默认） | 完全不修 ⇒ **与今天逐字相同** | 生产保守基线 / 回滚 |
| **1** | 缺页 ⇒ 用真 token 回填；**已有值只比对不覆盖**，把 `absent/mismatch` 计出来 | ★ **A3 先跑这个**：行为只在"今天会降级"的地方变化，同时把"陈旧页到底有没有"变成一个**数字** |
| **2** | 陈旧槽位也用真 token 覆写 | 若 mode=1 测出 `mismatch>0`（说明静默错误真实存在）⇒ 升到 2 |

### 3.2 修补点 = **读者坐标**（与页号语义无关）

```
q    = positions[row] - shift                      # shift = 1..lookback-1
page = block_table[request_ids[row], q // block_size]   # 与 kernel 读路径同式
slot = q % block_size
present[page]==0 ? 整行复位 + 写 : (比对；mode≥2 才覆写)
```

* **必须放 kernel 之前**：kernel 的 step-1 对"本步首次写到的页"会整行复位
  （`engram_jit_kernel.py:108-111`）—— 放在后面会被擦掉；先回填并把 `present=1`
  置上，kernel 就不会再复位它（同 `:108` 的分支）。
* **同一页里多个槽位只复位一次**（单测抓过一次回归，见 `engram_repair.py` 里的注释）。
* **`page<0` 一律不动**（读者对负页号会走 numpy 负索引/静默 pad，我们绝不写 `pages[-1]`）；
  **`page≥cap` 报 `oob`**，由调用方扩容后重跑（与 kernel 的 oob 路径同款）。

### 3.3 修补集合 = `q < 本请求本步的首位置`（代价上界可证明）

本步 kernel 的 step-1 会写下 batch 里**每一个**位置；对请求内第 i 行、`shift ≤ i` 的槽位
有 `q = p0+i−shift ≥ p0` ⇒ 本步已写、无需修补。只有 `q < p0`（跨到上一步/上一块）才可能
既不在本步写入集合里、又被读到。⇒ **每请求 ≤ 6 槽位**，与 batch/上下文无关。
（`tests/test_repair_plan.py`：4000 组随机布局 + 边界，与暴力枚举逐元素对照，
并检查"行不连续 ⇒ 退化为全扫"的兜底路径。）

### 3.4 与"降级修复"的关系：**互补，不冲突**

`21e7d99` 的 pageless（缺页 ⇒ barrier）**保留**：它是**最后一道兜底**
（修补不可得时：runner 未发布、超出 `num_tokens_no_spec`、行序对不齐…）。
本修复在它**之前**运行，把能拿到的真值填好 ⇒ 现实中 `pageless_history_rows` 应当降到 0。

---

## 4. 方案对比

| | **A. 真 token 修补**（推荐） | **B. 镜像随块随池携带** | **C. pad 降级**（`21e7d99`，已合入） | **D. 不修** |
|---|---|---|---|---|
| 精确性 | ★★★ 真 token ⇒ 与冷启动（fill）**逐值相同** | ★★★ 逐值相同（按归纳：取回时把被逐出页的行搬回来） | ★ 有精度损失（窗口被截断） | ⛔ 崩溃（旧）/ 静默错（陈旧页） |
| 覆盖"缺页" | ✅ | ✅（需在 store 时快照 source 页） | ✅（但损失精度） | ⛔ |
| 覆盖"**陈旧页**" | ✅（mode 2） | ✅（load 时覆盖目标页） | ⛔ **静默错照旧** | ⛔ |
| 改动面 | `engram_hash.py`(+38) / 新模块 `engram_repair.py`(229) / `model.py`(+60) / **runner 1 处发布(+30)** | 卸载 worker 的 store/load 路径（`native/cpu_npu.py:100-190 transfer_async`）+ worker→model 桥 + 每 CPU 页一份镜像行 | kernel + hash（已做完） | — |
| 额外内存 | **0** | ≈ 池子的 2.2%（每 CPU 页 8 KiB / 页宽 1024 token） | 0 | 0 |
| 每步开销 | ≤6 槽位/请求（Python/numpy，µs 级） | 每 store/load 一次行拷贝 | 0 | 0 |
| 风险面 | runner 整文件覆盖（有限、可 md5 钉死）；行序假设（有 fail-closed 校验）；陈旧页覆写（mode 2 才开） | worker 与 model 的**进程内桥**；快照缺失时无法修（会退化成 C）；**错误映射 ⇒ 静默错** | 精度损失 + 观测漏一格 | 不可上线 |
| 需要的实测 | 单卡 1 臂（mode=1）+ 8 卡 1 臂 | 单卡 1 臂 + 8 卡 1 臂（更难，改的是热路径） | 已做完 | — |
| **结论** | ★ **推荐** | 备选（A 的 runner 覆盖被否决时再走） | 保留为兜底 | 否 |

**推荐理由**：A 的"值"来自**权威源**（runner 的 host token 表，就是 `input_ids` 的来源），
不需要搬任何额外状态；而且它天然带一个**自校验**通道（拿真值与镜像比对 ⇒ `mismatch` 计数器），
这正好回答 `logs/071 §A1`（"取回路径从未被逐字节验证"）。

---

## 5. 本地语义测试（已跑通，不需要 NPU/numba）

```
bash a2/agents/Engram_exactfix/tests/run_all.sh
```

跑的是**生产 kernel 的原文件**（`dsv41-release/patches/files/engram_jit_kernel.py`，
只读 + `NUMBA_DISABLE_JIT=1`）加**出货件本身**（`patches/engram_repair.py`）。
四个场景的读数【实测·本机】：

| 场景 | 今天 | 精确修复（mode 1） |
|---|---|---|
| 1. p=页首（= `logs/073` 形态） | `[cur,pad,pad,pad]`，`err=2`、`miss_rows=1` | **`[cur, t₋₁, t₋₂, t₋₃]`**，`err=−1`、`miss_rows=0`、`filled=3` |
| 2. p=页中（同一页跨界） | `[cur,pad,pad,pad]`，**`err=−1`、`miss_rows=0`（完全静默）** | **`[cur, t₋₁, t₋₂, t₋₃]`**，`filled=3` |
| 3. 陈旧页（内容是别人的 token） | `[cur, 22, 21, 20]`（**静默算错**） | mode 1：`mismatch=3`（只测不改）；**mode 2：`[cur, t₋₁, t₋₂, t₋₃]`** |
| 4. 负页号 / 越界页号 | — | `oob` 上报 / 不写 `pages[-1]`，守门全部通过 |

---

## 6. 要改的文件 / 挂载与门控（草案见 `patches/README.md`）

| # | 文件 | 改动 | 门控 |
|---|---|---|---|
| 1 | **新模块** `models/deepseek_v41/engram_repair.py` | `plan_repair_slots` / `apply_repairs` / `build_prev_tok`（**出货件**，单测直接 import 它） | 影子包：`VLLM_V41_ENGRAM_TRUE_TOKENS!=0` 时挂载，**缺文件 die** |
| 2 | `models/deepseek_v41/engram_hash.py` | +38 行：门控 import、`update(..., prev_tok=None)`、kernel 前调用 `apply_repairs` + 计数 | `VLLM_V41_ENGRAM_TRUE_TOKENS` |
| 3 | `models/deepseek_v41/model.py` | +60 行：`set_engram_row_tokens()` 交接点 + `_engram_build_prev_tok()`（含**行序对齐校验**，不一致就返回 None）+ 传参 | 同上 |
| 4 | `worker/model_runner_v1.py` | +30 行：`_model_forward()` 里发布 `token_ids_cpu / num_tokens_no_spec / num_computed_tokens_cpu`（**零拷贝视图**） | `V41_ENGRAM_ROW_IDS=1` |
| 5 | `a2/scripts/serve_a2_offload.sh` + 影子包生成器 | 起服前断言：门控打开 ⇒ ①补丁文件存在 ②DRY 的真实 MOUNTS 里有 `engram_repair.py` ③打印三条 env 的生效值 | fail-closed |

**md5 钉死**：runner 基线必须 = `9d84d9b073aeece3fd3bbc143ba20567`
（= A3 容器里的实际文件，【实测】）；`engram_hash.py` 基线 = `240c5a0444a1434a33d80341ea50e1d3`。
生成 shadow 时校验，不匹配就 die（避免"挂在别的版本上"这种静默漂移）。

---

## 7. 验证路线（成本都在分钟级）

| 步骤 | 内容 | 成本 |
|---|---|---|
| S0 | 本地语义测试 `tests/run_all.sh` | **<1 min**（已跑通） |
| S1 | 静态：`diff` 应用、md5 断言、DRY 起服（`DRY=1`）看真实 MOUNTS | ~5 min |
| S2 | ★ **A3 单卡**（c0，Phy-ID 8–15，真权重）：`ENGRAM=1 + DEVICE_INDEX=0 + 卸载 + DRAFT_GRAPH=1`，`TRUE_TOKENS=1`、`ROW_IDS=1`，**只跑"同 prompt 连发两次"**（cold→hit），不跑 fill | ~25 min（起服 ~12 min + 2 次 replay + 读数） |
| S3 | 读数判据：`absent>0`（证明命中缺页这条病）、`mismatch=?`（**决议 mode 1 还是 2**）、`pageless_history_rows`（应≈0）、输出与 cold 轮逐字一致 | 含在 S2 |
| S4 | A3 8 卡（`R_8card_int8` 的 p2e 形态）一条臂：`fill → replay`，比 `logs/069` 的判据 | ~30 min |
| S5 | A2 窗口：干跑 + 起服 + 三道自检 + 两条计数器读数 | ~30 min |

★ S2 是**信息量最大的一步**：它用最小代价同时回答"缺页真发生了吗""陈旧页真发生了吗"
（后者是**今天完全静默**的那一格）。

---

## 8. 【未确认】清单（共 **8** 项）

1. **缺页的成因**：是"这些页从未被写过"（分配器给到的页号在**已写区间之外**）还是
   "镜像被重编号/重置"。A3 上的**判别测量**：把 `present.sum()`、`present.nonzero().max()`
   与 `absent` 页号一起打出来（一行日志）——若 `absent` 页号 > 已写下界，则为前者。
2. **陈旧页在真机上是否真的发生**（今天完全静默）：只能靠 mode 1 的 `mismatch` 计数器回答。
3. **行序假设**（模型侧 `requests` == runner `input_batch` 行号）：已加对齐校验
   （`num_computed` vs 模型自算首位置），但**未在真机验证**过。
4. **`lookback` 在 A2 生产模型上的取值**：本仓 `config.json` 是 4，
   **未**直接读 A2 模型目录（若为别的值，公式与代价上限随 lookback 变）。
5. **`num_tokens_no_spec` 与 `token_ids_cpu` 的成对性**在 PP>1 的
   placeholder 路径（`model_runner_v1.py:820-844`）上不成立 ⇒ 本方案默认假设 **PP=1**。
6. **draft（dspark）前向**是否会用自己的 batch 再发布一次（同一份 `input_batch`，
   理论上无害），**未在真机确认**。
7. **端到端精确性**：目前只有 kernel 级语义证据；"修补后哈希 / 输出与 cold 轮逐字相同"
   **未在任何加速器上验证**。
8. **A2 实际生效的 `V41_ENGRAM_JIT`**：脚本默认 1（`serve_a2.sh:139`），
   但**未**从 A2 的起服日志回读确认。

---

## 9. 交付物索引

| 件 | 路径 |
|---|---|
| 本设计 | `a2/docs/ENGRAM-OFFLOAD-EXACT-FIX.md` |
| 出货模块（会被挂进容器） | `a2/agents/Engram_exactfix/patches/engram_repair.py` |
| 三个补丁 diff + 完整补丁后文件 | `a2/agents/Engram_exactfix/patches/*.diff` / `*.patched.py` |
| 挂载/门控/回滚说明 | `a2/agents/Engram_exactfix/patches/README.md` |
| 本地语义测试 + 覆盖性测试 | `a2/agents/Engram_exactfix/tests/{test_true_tokens_repair,test_repair_plan}.py`、`run_all.sh` |

---

## 10. 与 `A2-ENGRAM-PATHS.md` 的接口（用户硬约束 `ENGRAM=1` 下的三条路）

那份文档（路 1/2/3 全开 Engram）把本问题放在两个地方，本方案**正好把它们接上**：

| 它的位置 | 原文要点 | 本方案的作用 |
|---|---|---|
| §0 路 2 前置条件 | "A3 的 `ENGRAM-PAGELESS` 修复臂必须 `replay failed=0`" | 这里给出**两条**可选臂：<br>① `TRUE_TOKENS=0`（= 纯 pageless，**有 pad 精度损失**）<br>② ★ `TRUE_TOKENS=1/2` + `ROW_IDS=1`（**零损失**，且多两个判据 `absent>0` / `mismatch`）|
| §7 未确认 #4 | "**pad 历史对输出质量的影响幅度**：有界 ≠ 无损" | ★ 这正是本方案消掉的那一项：mode≥1 下不再有 pad 损失；<br>剩下的"有界"只用于**兜底路径**（`unavailable>0` 时）。⇒ 建议把该条改成"**mode≥1 下应为 0**；若 `unavailable>0` 再评估" |
| §1 内存账 | 池宿主 296.7 GiB、余量 142 GiB | 本方案**不占额外宿主内存**（镜像行就在模型进程里；runner 表是既有数组的零拷贝视图）⇒ 与内存账**零冲突** |
| §6 窗口自检 | `DRY=1` 看真实 MOUNTS | 追加两条：①`engram_repair.py` 必须在真实 MOUNTS 里；②容器内 `md5sum` 必须 == 钉死的 patched 文件 md5（`patches/README.md` §3 的 G1/G2/G3） |
