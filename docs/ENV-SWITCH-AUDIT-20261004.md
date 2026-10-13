# 全量环境开关审计（`V41_*` / `DSPARK_*` / 交付启动器）

**日期**：2026-10-04　**执行**：`/root/ced_die_budget` 子代理（**只读审计**：未启停服务、未改任何文件、未 `exec` 进 `dsv41-tp8k5`）
**审计对象**：`dsv41-tp8k5`（a3-21 dies 8–15，TP8 交付实例，端口 19210，run `fix_1004_1645`）
**开关来源**：`~/cedpd-repo/patches/files/*.py`、`patches/files/draft/*.py`、`patches/*.patch`、`scripts/serve_a2.sh`
**交付值来源**：`sudo docker inspect dsv41-tp8k5`（只读）+ 挂载源文件（`.Mounts` 显示交付实例挂的正是 `~/cedpd-repo/patches/files/...`）
**标注**：【实测】= 有日志/容器事实；【实测·代码】= 有明确代码路径；【推断】= 由实测推导；【未确认】= 无证据。

---

## 0. 摘要

### 0.1 三条要办的事（按影响排序）

| # | 发现 | 类别 | 影响量级 | 依据 |
|---|---|---|---|---|
| **★1** | **交付实例 `V41_ENGRAM_DEVICE_INDEX=0`，把 Engram 算子入图整条关掉**（`serve_a2.sh` 默认是 `auto`，而 A3 上 `auto` 会**开启**它）。日志里 `DEVICE-INDEX` 出现 **0 次** ⇒ 全程走 host 路径 | B | 【实测】每 decode 步 `d2h` **p50 = 10.10 ms**（n=3136）；同窗口 `hp`（= 整步）p50 = **26.245 ms** ⇒ 占 **≈38%** | §2-B1 |
| **★2** | **`bool(os.environ.get(...))` 家族的潜伏 bug**：除已知的 `ROUTE_PROBE` 外还有 **2 处**（`REUSE_EP_GROUP`、`HOST_RESIDENT`）。交付值恰好是 `"1"` 所以**现在无害**，但**显式设 `0` 关不掉**（`bool("0") == True`） | A | 【实测】Python 语义验证；当前 0 µs（值恰好对），未来"想关"时静默失效 | §2-A1 |
| **★3** | **route-probe 的修复尚未对当前进程生效**：修复落盘 `17:00:33`，而容器 `StartedAt = 16:43:51`；`[route-probe]` 位于**第 6904 行 / 共 6912 行**、计数 **2760**、日志 mtime = 当前时刻 ⇒ **仍在实时输出** | A（已修·待生效） | 计时 µs 级 + 日志噪声；**重启即消** | §2-A2 |

### 0.2 一条技术纠正（避免团队去修不存在的问题）

任务书假设的「`int(x or 20)` 把 `"0"` 变成 20」**不成立**：

```
bool(os.environ.get("T",""))          # T="0" → True    ← 真 bug
int(os.environ.get("T","20") or 20)   # T="0" → 0        ← 不是 bug
int(os.environ.get("T","20") or 20)   # T=""  → 20       ← 只有显式空串才回落
```

原因：`os.environ.get()` 返回**字符串**，`"0"` 是**非空字符串**（truthy）⇒ `or` 不触发。
⇒ 全部 `int(... or D)` / `_str or "default"` 写法对 `"0"` **都是正确的**；真正要修的是 `bool(...)` 家族（§2-A1）。
**唯一的反例**是 `engram_hbm.py:64` 的 `V41_ENGRAM_ROUTE_PROBE_EVERY`：`every=0` 会让 `self.n % self.every` **抛 `ZeroDivisionError`**，而 `or 20` 拦不住字符串 `"0"` ⇒ 设 `0` 是**崩溃**而非静默（低危，不建议设 0）。

### 0.3 审计边界（诚实声明）

| 范围 | 状态 |
|---|---|
| `patches/files/*.py` + `patches/files/draft/*.py` | ✅ 全量扫描（95 变量 / 111 读取点） |
| `patches/*.patch`（含 `admission_gate.patch`） | ✅ 扫描（补到 `V41_GATE_MAX_PREFILL` 等 7 个） |
| `scripts/serve_a2.sh` | ✅ 扫描（含 `-e` 传递链） |
| `experimental/ced/*`（CED 专用） | ⚠️ **未展开**：当前实例 `V41_CED_ROLE=""`，该路径不加载 ⇒ 对本次交付无影响 |

---

## 1. 全量清单

### 1.1 统计

| 项 | 数 |
|---|---:|
| 唯一环境变量名（patches + draft） | **95** |
| 读取点（文件:行） | **111** |
| 交付容器 env 里的 `V41_*`/`DSPARK_*` | **86** |
| 补充来源（core patch）新增 | +7（`V41_GATE_MAX_PREFILL`、`V41_CED_*`、`V41_DYNSPEC_SKIP_K0_DRAFT_COPY`） |

### 1.2 全量对照表

「代码默认」列是 `os.environ.get(NAME, DEFAULT)` 里的字面量；**无默认**＝调用时没给第二参数。**交付值**取自 `docker inspect`。

### 统计: 读取点 111 个, 唯一变量名 95 个

| 变量 | 读取处(文件:行) | 代码默认 | 交付值 |
|---|---|---|---|
| `DSPARK_CAPTURE_DISPATCH` | patches/files/draft/dspark_proposer.py:74 | `"0"` | `0` |
| `DSPARK_CAPTURE_MAXSEQLEN` | patches/files/draft/dsa_v1.py:178 | `"0"` | `0` |
| `DSPARK_CAPTURE_NCTX_FIX` | patches/files/draft/dspark_proposer.py:88 | `"1"` | `1` |
| `DSPARK_CAPTURE_NCTX_LOG` | patches/files/draft/dspark_proposer.py:967 | `"0"` | `**未设**` |
| `DSPARK_CAPTURE_PAD_SLOTS` | patches/files/draft/dspark_proposer.py:47 | `"0"` | `0` |
| `DSPARK_CAPTURE_SEQ_LEN` | patches/files/draft/dspark_proposer.py:867 | `"0"` | `0` |
| `DSPARK_CAPTURE_VALUE_FIX` | patches/files/draft/dspark_proposer.py:72 | `"0"` | `1` |
| `DSPARK_DISPATCH_DIAG_STEPS` | patches/files/draft/llm_base_proposer.py:179 | `"0"` | `0` |
| `DSPARK_DISPATCH_QUERY_LEN_FIX` | patches/files/draft/llm_base_proposer.py:199 | `"1"` | `**未设**` |
| `DSPARK_DISPATCH_UNIQUE` | patches/files/draft/llm_base_proposer.py:204 | `"0"` | `**未设**` |
| `DSPARK_DRAFT_METADATA_MODE` | patches/files/draft/dspark_proposer.py:117<br>patches/files/draft/llm_base_proposer.py:373 | `"sync"` | `sync` |
| `DSPARK_DRAFT_NO_ATTN` | patches/files/draft/llm_base_proposer.py:79 | `"0"` | `0` |
| `DSPARK_DRAFT_SERIAL` | patches/files/draft/llm_base_proposer.py:173 | `"0"` | `0` |
| `DSPARK_DRAFT_SYNC_AFTER` | patches/files/draft/llm_base_proposer.py:84 | `"0"` | `0` |
| `DSPARK_DRAFT_SYNC_BEFORE` | patches/files/draft/llm_base_proposer.py:133 | `"0"` | `0` |
| `DSPARK_DRAFT_USE_CUDAGRAPH` | patches/files/draft/dspark_proposer.py:194 | `"1"` | `1` |
| `DSPARK_DSA_PROBE` | patches/files/draft/dsa_v1.py:154 | `"0"` | `0` |
| `DSPARK_DSA_PROBE_CAPTURE` | patches/files/draft/dsa_v1.py:158 | `"0"` | `0` |
| `DSPARK_DSA_PROBE_STEPS` | patches/files/draft/dsa_v1.py:155 | `"400"` | `60` |
| `DSPARK_DSA_WRITE_PROBE` | patches/files/draft/dsa_v1.py:160 | `"0"` | `0` |
| `DSPARK_DSA_WRITE_PROBE_STEPS` | patches/files/draft/dsa_v1.py:161 | `"12"` | `12` |
| `DSPARK_FIA_PAD_REQS_FIX` | patches/files/draft/llm_base_proposer.py:214 | `"0"` | `**未设**` |
| `DSPARK_GRAPH_AB_LEGACY_CAPTURES` | patches/files/draft/dspark_proposer.py:41 | `"0"` | `**未设**` |
| `DSPARK_GRAPH_CAPTURE_METADATA` | patches/files/draft/dspark_proposer.py:40 | `"0"` | `1` |
| `DSPARK_GRAPH_DEBUG` | patches/files/draft/dspark_proposer.py:43 | `"0"` | `0` |
| `DSPARK_GRAPH_DEVICE_METADATA` | patches/files/draft/dspark_proposer.py:104<br>patches/files/draft/llm_base_proposer.py:361 | `"0"` | `0` |
| `DSPARK_GRAPH_DEVICE_METADATA_FROM` | patches/files/draft/dspark_proposer.py:105 | `"0"` | `**未设**` |
| `DSPARK_GRAPH_PTR_PROBE` | patches/files/draft/llm_base_proposer.py:226 | `"0"` | `0` |
| `DSPARK_GRAPH_PTR_PROBE_STEPS` | patches/files/draft/llm_base_proposer.py:227 | `"5"` | `5` |
| `DSPARK_GRAPH_SHADOW_EAGER` | patches/files/draft/llm_base_proposer.py:66 | `"0"` | `0` |
| `DSPARK_GRAPH_SHADOW_STEPS` | patches/files/draft/llm_base_proposer.py:67 | `"2"` | `2` |
| `DSPARK_HOIST_CONTEXT_KV` | patches/files/draft/llm_base_proposer.py:111 | `"0"` | `0` |
| `DSPARK_NO_TOPK_SHARE` | patches/files/draft/llm_base_proposer.py:74 | `"0"` | `0` |
| `DSPARK_ROW_DUMP` | patches/files/draft/dspark_proposer.py:44<br>patches/files/draft/llm_base_proposer.py:70 | `"0"` | `0` |
| `DSPARK_ROW_DUMP_STEPS` | patches/files/draft/llm_base_proposer.py:71 | `"8"` | `**未设**` |
| `DSPARK_RT_FLAGS` | patches/files/draft/llm_base_proposer.py:140 | `"0"` | `0` |
| `DSPARK_STEP_PROBE` | patches/files/draft/llm_base_proposer.py:175 | `"0"` | `0` |
| `DSPARK_STEP_PROBE_STEPS` | patches/files/draft/llm_base_proposer.py:176 | `"40"` | `40` |
| `DSPARK_SWA_INDICES_RESIDENT` | patches/files/draft/dsa_v1.py:218 | `"1"` | `1` |
| `DSPARK_TOKEN_DUMP` | patches/files/draft/llm_base_proposer.py:72 | `"0"` | `0` |
| `DSPARK_TOKEN_DUMP_STEPS` | patches/files/draft/llm_base_proposer.py:170 | `"12"` | `12` |
| `RANK` | patches/files/engram_hash.py:162<br>patches/files/engram_hash.py:189 | `'?'` | `**未设**` |
| `V41_BNECK_DECODE_TOKENS` | patches/files/model.py:395 | `"64"` | `**未设**` |
| `V41_BNECK_MODE_FILE` | patches/files/model.py:281 | `"/tmp/v41_bneck_mode"` | `**未设**` |
| `V41_BNECK_PRINT_EVERY` | patches/files/model.py:298 | `"20"` | `**未设**` |
| `V41_CED_ALLOW_DSPARK` | patches/files/model.py:927 | `"1"` | `0` |
| `V41_CED_CAPTURE_DECODE` | patches/files/model.py:68 | `"0"` | `0` |
| `V41_CED_H20_SNAPSHOT_DIR` | patches/files/model.py:60 | `""` | `` |
| `V41_CED_H20_SNAPSHOT_POS` | patches/files/model.py:59 | `""` | `` |
| `V41_CED_LAYER_SNAPSHOT_DIR` | patches/files/model.py:62 | `""` | `` |
| `V41_CED_LAYER_SNAPSHOT_LAYERS` | patches/files/model.py:65 | `"0,1,2,13,14,15,19,20"` | `0,1,2,13,14,15,19,20` |
| `V41_CED_LAYER_SNAPSHOT_POS` | patches/files/model.py:61 | `""` | `` |
| `V41_CED_ROLE` | patches/files/model.py:58<br>patches/files/model.py:136<br>patches/files/model.py:182<br>patches/files/model.py:902<br>… | `"" / "baseline"` | `` |
| `V41_CED_SOURCE_COMPARE` | patches/files/model.py:943 | `"0"` | `0` |
| `V41_CED_SOURCE_COMPARE_CHUNKS` | patches/files/model.py:942 | `"1"` | `1` |
| `V41_DECODE_API_GUARD` | patches/files/v41_decode_guard.py:75 | `"1"` | `1` |
| `V41_DSPARK_SHAPE_PROBE` | patches/files/draft/llm_base_proposer.py:476 | `"0"` | `**未设**` |
| `V41_DUMMY_WO_A_FIX` | patches/files/dsa_v1.py:147<br>patches/files/draft/dsa_v1.py:361 | `"0"` | `**未设**` |
| `V41_DYNSPEC_BT_PERSIST` | patches/files/draft/dspark_proposer.py:421 | `"0"` | `**未设**` |
| `V41_ENGRAM_DEVICE_FALLBACK` | patches/files/engram_hbm.py:283<br>patches/files/model.py:225 | `"0"` | `0` |
| `V41_ENGRAM_DEVICE_GRAPH` | patches/files/model.py:254 | `"1"` | `**未设**` |
| `V41_ENGRAM_DEVICE_GRAPH_MAX` | patches/files/model.py:259 | `"16"` | `**未设**` |
| `V41_ENGRAM_DEVICE_INDEX` | patches/files/engram_hbm.py:276<br>patches/files/model.py:216 | `"0" / "auto"` | `0` |
| `V41_ENGRAM_DEVICE_PAGES` | patches/files/model.py:231 | `"8192"` | `**未设**` |
| `V41_ENGRAM_GATE_CHUNK` | patches/files/engram_gate.py:56 | `""` | `0` |
| `V41_ENGRAM_GATE_HOIST` | patches/files/engram_gate.py:47<br>patches/files/model.py:17 | `"0"` | `0` |
| `V41_ENGRAM_HIST_TRACE_POS` | patches/files/engram_hash.py:115 | `""` | `` |
| `V41_ENGRAM_HOST_RESIDENT` | patches/files/engram_hbm.py:474 | `""` | `1` |
| `V41_ENGRAM_JIT` | patches/files/engram_jit_kernel.py:52<br>patches/files/engram_plan_kernel.py:45 | `"0"` | `1` |
| `V41_ENGRAM_JIT_PAGES` | patches/files/engram_jit_kernel.py:54 | `"4096"` | `**未设**` |
| `V41_ENGRAM_LOCAL_METADATA` | patches/files/engram_hbm.py:143 | `"off"` | `**未设**` |
| `V41_ENGRAM_LOCAL_METADATA_FILE` | patches/files/engram_hbm.py:140 | `"/tmp/v41_engram_localmeta"` | `**未设**` |
| `V41_ENGRAM_LOCAL_OWNER` | patches/files/engram_hbm.py:186 | `"off"` | `**未设**` |
| `V41_ENGRAM_LOCAL_OWNER_FILE` | patches/files/engram_hbm.py:183 | `"/tmp/v41_engram_localowner"` | `**未设**` |
| `V41_ENGRAM_PAD_SKIP` | patches/files/model.py:55 | `"0"` | `**未设**` |
| `V41_ENGRAM_PAGELESS_STRICT` | patches/files/engram_hash.py:30 | `"0"` | `**未设**` |
| `V41_ENGRAM_PG_BUFFER_MB` | patches/files/engram_hbm.py:377 | `""` | `**未设**` |
| `V41_ENGRAM_REUSE_EP_GROUP` | patches/files/engram_hbm.py:376 | `""` | `1` |
| `V41_ENGRAM_ROUTE_PROBE` | patches/files/engram_hbm.py:67 | `""` | `0` |
| `V41_ENGRAM_ROUTE_PROBE_EVERY` | patches/files/engram_hbm.py:68 | `"20"` | `**未设**` |
| `V41_ENGRAM_WITH_DUMMY` | patches/files/model.py:54 | `"0"` | `**未设**` |
| `V41_FORCE_CAND_MODE` | patches/files/indexer.py:251 | `"0"` | `0` |
| `V41_IDS64_HOIST` | patches/files/model.py:57 | `"0"` | `**未设**` |
| `V41_MOE_COMM_ALLGATHER` | patches/files/ascend_forward_context.py:313 | `"0"` | `1` |
| `V41_MOE_INVALID_PROBE` | patches/files/token_dispatcher_moezero.py:65 | `"0"` | `**未设**` |
| `V41_MOE_MASK_RANGE` | patches/files/token_dispatcher_moemask.py:62<br>patches/files/token_dispatcher_moennf.py:75<br>patches/files/token_dispatcher_moezero.py:62 | `"0"` | `1` |
| `V41_MOE_ZERO_INVALID` | patches/files/token_dispatcher_moezero.py:64 | `"0"` | `0` |
| `V41_MOE_ZERO_INVALID_FILE` | patches/files/token_dispatcher_moezero.py:77 | `""` | `/tmp/v41_moe_zero_file` |
| `V41_MOE_ZERO_NONFINITE` | patches/files/token_dispatcher_moennf.py:79 | `"0"` | `0` |
| `V41_MOE_ZERO_NONFINITE_FILE` | patches/files/token_dispatcher_moennf.py:80 | `""` | `/tmp/v41_moe_nf` |
| `V41_O_PROJ_2D` | patches/files/dsa_v1.py:146<br>patches/files/draft/dsa_v1.py:146 | `"0"` | `1` |
| `V41_QLI_NO_CANDIDATE` | patches/files/indexer.py:11 | `"0"` | `1` |
| `V41_ROPE_IDXSEL` | patches/files/rope_dsv4.py:29 | `"0"` | `1` |
| `V41_ROUTE_PIPE` | patches/files/engram_hbm.py:235 | `"off"` | `**未设**` |
| `V41_ROUTE_PIPE_FILE` | patches/files/engram_hbm.py:232 | `"/tmp/v41_route_pipe"` | `**未设**` |

### 容器 env 里存在但 patches 未读取的 V41_*/DSPARK_* 变量

- `V41_CED_BLOCK_DUMP_DIR` = ``
- `V41_CED_BLOCK_TRACE` = `0`
- `V41_CED_DYNAMIC_SPEC_FULL_GRAPHS` = `0`
- `V41_CED_GRAPH_PROMPT_TAIL_EAGER` = `0`
- `V41_CED_KVGEOM` = `0`
- `V41_CED_SNAPSHOT_DIR` = ``
- `V41_CED_SNAPSHOT_POS` = ``
- `V41_CED_SWA_CLIP` = `1`
- `V41_CED_SWA_TRACE` = `0`
- `V41_ENGRAM_GATE_MAX_TOKENS` = `8192`
- `V41_GATE_MAX_PREFILL` = `8`
- `V41_SLOT_MAP_FUSED` = `0`

### 1.3 「容器设了、但上表脚本没抓到」的 12 项 —— 逐项定性

上表的正则只匹配 `os.environ.get("字面量")`。以下 12 项是**间接引用或 core patch 读取**，不是死开关：

| 变量 | 实际读取处 | 定性 |
|---|---|---|
| `V41_ENGRAM_GATE_MAX_TOKENS` | `patches/files/engram_gate.py:92`（常量）+ `:104` | ✅ **有代码读**（常量间接引用） |
| `V41_SLOT_MAP_FUSED` | `patches/files/block_table.py:46` + `:50` | ✅ **有代码读**（常量间接引用） |
| `V41_GATE_MAX_PREFILL` | `patches/admission_gate.patch` | ✅ **有代码读**（vLLM core 补丁） |
| `V41_CED_SWA_CLIP` / `V41_CED_SWA_TRACE` / `V41_CED_BLOCK_TRACE` / `V41_CED_BLOCK_DUMP_DIR` / `V41_CED_KVGEOM` / `V41_CED_SNAPSHOT_*` / `V41_CED_GRAPH_PROMPT_TAIL_EAGER` / `V41_CED_DYNAMIC_SPEC_FULL_GRAPHS` | `experimental/ced/*`（本次未展开） | ⚠️ **当前路径死开关**：实例 `V41_CED_ROLE=""` ⇒ CED 代码不加载 ⇒ 这些值**完全不生效** |

> **结论**：交付容器里**没有**「设了却没人读」的**真实**配置项（真死的只有 CED 系列，而它们本就不是给 TP8 单实例用的）。

---

## 2. 风险清单（按影响排序）

### B1 · 交付把 Engram 算子入图整条关掉了 —— **本轮最大发现**

| 项 | 内容 |
|---|---|
| 开关 | `V41_ENGRAM_DEVICE_INDEX` |
| 读取处 | `patches/files/model.py:216-218`（宽松解析）；`patches/files/engram_hbm.py:276`（严格 `== "1"`） |
| 代码默认 | `model.py` = `"auto"`；`engram_hbm.py` = `"0"` |
| **交付值** | **`0`** |
| 语义 | `auto` = 用驱动侧 `host_mem_pool` 特性探测（A3 通过 ⇒ 开；A2 不通过 ⇒ 回退 host）；`1` = 强制开；`0` = **强制关** |

**证据链**

1. 【实测】`docker inspect` 交付值 `V41_ENGRAM_DEVICE_INDEX=0`；`serve_a2.sh:215` 的默认是 `auto`，`:219-226` 的 `_engram_need_rw()` 还会为它准备 `:rw` 挂载 ⇒ **默认意图是"可能开启"**。
2. 【实测】`V41_ENGRAM_DEVICE_INDEX=0` ⇒ `_ENGRAM_DEVICE_INDEX_MODE = "0"` ⇒ `model.py:1046` 的 `elif ... not in ("1","true","on","yes"): return` **直接返回**，连 `auto` 的能力探测日志都不会打 ⇒ 与「日志里 `DEVICE-INDEX` 出现 **0 次**」一致。
3. 【实测】同窗口 `[bneck]` 读数：`d2h` **p50 = 10.102 ms**（n=3136，min 0.126 / p90 23.122 / max 163.818）；`hp` **p50 = 26.245 ms**（≈ N=1 的 26.2 ms/step）。
4. 【实测·代码】device-index 的设计目的正是**替换掉整条 host 路径**（`engram_hbm.py:266-271` 注释：`V41_ENGRAM_DEVICE_INDEX=1 replaces the whole host lookup path`；`model.py:199` 同义）。

**影响估计**

* 【实测】host 路径的 `d2h` 占每 decode 步 **≈38%**（p50 口径）。
* 【推断】开启后 `d2h` 这 10.1 ms 不会**全部**变成净收益（部分可与其它流重叠），但这是**已知五条路径（d2h 同步 / 分片 / all_gather / all_to_all / broadcast / h2d）里最贵的一条**。E2E 净收益必须 A/B 测，**不要直接按 38% 记账**。
* 【未确认】当初为何设 `0`（可能是为排查某问题临时关闭后未恢复；A2 目标机上这是**正确**取值，但本实例在 A3）。

**一句话修法**：把交付命令里的 `ENGRAM_DEVICE_INDEX=0` 去掉（走 `auto`）或显式设 `1`，重启后看 `[DEVICE-INDEX] 能力探测通过` 与 `[bneck] d2h`。

---

### A1 · `bool(os.environ.get(...))` 家族：共 3 处，2 处仍潜伏

| # | 文件:行 | 变量 | 交付值 | 现状 |
|---|---|---|---|---|
| 1 | `engram_hbm.py:67` | `V41_ENGRAM_ROUTE_PROBE` | `0` | ✅ **代码已修**（改为 `in ("1","true","yes","on")`），但**进程未重载** ⇒ 见 A2 |
| 2 | `engram_hbm.py:376` | `V41_ENGRAM_REUSE_EP_GROUP` | `1` | ⚠️ **未修**。当前值 `"1"` ⇒ `True` 与意图一致（无害），但**设 `0` 关不掉** |
| 3 | `engram_hbm.py:474` | `V41_ENGRAM_HOST_RESIDENT` | `1` | ⚠️ **未修**。同上，**设 `0` 关不掉** |

**为什么是 bug**：【实测】`bool("0") is True` ⇒ `bool(os.environ.get(NAME, ""))` 对 `"0"` / `"false"` / `"off"` **一律返回 True**。
**后果**：这两个开关**只能开、不能关**。默认值是 `""`（=关），所以任何想显式关闭的人（A/B 对照、回退）都会**静默地仍然开着**，实验结论会被污染。
**当前影响**：0（交付值恰好是 `"1"`，与意图一致）—— 所以这是**潜伏风险**，不是现行性能损失。
**修法**：与 #1 同一模式 —— `.strip().lower() in ("1","true","yes","on")`。

---

### A2 · 已修但**未生效**：route-probe 仍在当前进程里输出

| 项 | 内容 |
|---|---|
| 证据 | 容器 `StartedAt` = **2026-10-04T08:43:51Z**（= 北京 16:43:51）；修复落盘 mtime = **17:00:33** ⇒ 修复**晚于**进程加载 |
| 【实测】 | `[route-probe]` **2760 次**；最后一次在**第 6904 行 / 总 6912 行**；`serve.log` mtime = **当前时刻** |
| 语义 | Python 在 import 期读一次 env，之后改文件不影响已加载的模块 ⇒ **必须重启才生效** |
| 影响 | 【实测·代码】`_RP.start()/mark()` 内部有 `if not self.on: return` 守卫 ⇒ 净开销只有 `perf_counter()`（每次 mark 一次，µs 级）+ 每 20 步 × 8 rank 一次 `print` ⇒ **不是 ms 级**。真实危害是**日志噪声**（2760 行 ≈ 全日志 40%）掩盖真实信号 |
| 修法 | 重启（**不要**为此单独重启；与 B1 的 device-index 决策一起做） |

> ⚠️ 注意区分：日志里 `a2a=0.19 / lookup=0.235 / plan=0.07 ...` 这些**数字**是 route 路径的**真实耗时**（探针开不开都要做），**不是探针引入的开销**。别把 0.78 ms 记到探针账上。

---

### B2 · bneck 探针**默认开着**（当前 2760 行）

| 项 | 内容 |
|---|---|
| 读取处 | `patches/files/model.py:281`（路径）、`:298`（`V41_BNECK_PRINT_EVERY`，默认 `"20"`）、`:395`（`V41_BNECK_DECODE_TOKENS`，默认 `"64"`） |
| 交付值 | **均未设** ⇒ 全部走默认 |
| 机制 | `_BP_STATE = _BneckState()` 是**模块级无条件实例化**（`model.py:397`）；`tick()` 里 `if self.print_every <= 0 ...: return` ⇒ 默认 20 ⇒ **每 20 步按 rank 打印一行** |
| 【实测】 | `[bneck]` **2760 行**；`hp` p50 = 26.245 ms、`total` p50 = 11.892 ms |
| 影响 | 每步 1 次 `perf_counter()` + 每 0.25 s 一次 `os.stat`（`refresh()` 节流）+ 每 20 步 8 行 print ⇒ **µs 级，主要是日志噪声** |
| 可关？ | ✅ **可以**：`V41_BNECK_PRINT_EVERY=0` **确实有效**（`"0"` 是 truthy ⇒ `int("0")=0` ⇒ `print_every<=0` 短路）。⚠️ 但**不要**用 `V41_ENGRAM_ROUTE_PROBE_EVERY=0`（那会 `ZeroDivisionError`，见 §0.2） |
| 修法 | 交付加 `-e V41_BNECK_PRINT_EVERY=0`（或把默认从 `"20"` 改成 `"0"`，让它变成"显式开启才打印"） |

---

### B3 · 同一开关在四个脚本里默认值不一致（容易"我以为是 A，跑的是 B"）

| 开关 | `serve_a2.sh` | `serve_v2.sh` | CED `_common.sh` | **交付实例实际** |
|---|---|---|---|---|
| `MULTISTREAM` | `1`（`:343`） | `0`（`:46`） | `0` | **1**（`additional-config` 里 `multistream_overlap_shared_expert: true`，【实测】） |
| `DSA_OVERLAP` | `1`（`:350`） | `1` | `0` | **1**（`multistream_dsv4_dsa_overlap: true`） |
| `CPU_BIND` | `1`（`:335`） | `1` | `0` | **1**（`enable_cpu_binding: true`） |
| `DROPCACHE` | `1`（`:230`） | — | `0` | **1**（未传容器，只是起服前的 shell 行为） |

**为什么重要**
* 历史结论（CED-PD 线）：「**关闭多流是解决乱码问题的最主要开关**」。本实例是 **TP8 单实例**（非 CED），`MULTISTREAM=1` / `DSA_OVERLAP=1` **可能是有意的**（A2 生产就把它们开着）——但**必须显式确认**，不能靠默认值。
* `DROPCACHE=1` 在共用机上会**清整机 page cache**；CED 交付已改成 0，TP8 这条线还是 1。
* 【实测】`serve_a2.sh` 的注释自己写明这是"历史性能口径，与 v3 逐字节一致"，说明是**故意**保留的旧默认 —— 风险在于**同一台机器上不同部署形态拿到不同默认**。

**修法**：把四个开关的默认值收敛到**一处**（例如 `scripts/_defaults.sh`），并在起服日志里打印"本实例实际生效值"（`serve_v2.sh:195` 已经打了一行，可复用）。

---

### C1 · `/tmp` 文件驱动的"隐藏开关"（4 个）—— 能静默改变数值

| 变量 | 默认路径 | 危险取值 |
|---|---|---|
| `V41_BNECK_MODE_FILE` | `/tmp/v41_bneck_mode` | **`nohost`** ⇒ `model.py:1281-1285` 让 Engram host 路径**全短路**，`_bp_zero_lookups()` 返回**全零** lookups ⇒ **静默改变数值**（不报错） |
| `V41_ROUTE_PIPE_FILE` | `/tmp/v41_route_pipe` | `on` ⇒ 切换 route 实现（`engram_hbm.py:235`） |
| `V41_ENGRAM_LOCAL_OWNER_FILE` | `/tmp/v41_engram_localowner` | `fast` ⇒ 切换 owner 分片策略（`:186`） |
| `V41_ENGRAM_LOCAL_METADATA_FILE` | `/tmp/v41_engram_localmeta` | `on`/`validate` ⇒ 切换 metadata 路径（`:143`） |

**现状**：【实测】四个 env 交付都**未设**、路径文件由 `serve_a2.sh` 只写 `v41_engram_localowner`(=`fast`) 与 `v41_hash_mode` ⇒ 当前行为正确。
**风险**：这四类开关**只要容器内有人写一个文件就能改行为**，且**不留起服日志**。`nohost` 尤其危险（数值静默变零）。
**修法**：① 交付显式传 `V41_BNECK_MODE_FILE=` 指向一个**永不创建**的路径；或 ② 启动时若探测到 **非 `stock`** 模式就打印一行 WARNING。

---

### C2 · `V41_ENGRAM_DEVICE_INDEX` 被**两处用不同规则解析**（`auto` 下会分叉）

| 位置 | 解析 |
|---|---|
| `model.py:216-218` | `mode not in ("0","false","off","no","")` ⇒ `auto`/`true`/`on`/`yes` **都算开** |
| `engram_hbm.py:276` | `== "1"` ⇒ **只有 `"1"` 算开** |

**分叉后果**（取值 = `auto`，即 `serve_a2.sh` 的默认）：

* `model.py` ⇒ 走 device 路径（A3 探测通过）；
* `engram_hbm.py` ⇒ `_ENGRAM_DEVICE_INDEX=False` ⇒ `_ENGRAM_DEVICE_TRIM_SHARD=False` ⇒ **保留完整 host 分片**（注释写明这是 **25.75 GB/rank/层** 的 DRAM，`engram_hbm.py:266-275`）。
* ⇒ `auto` 模式 = **设备路径 + 完整 host 分片都占**（DRAM 峰值最高）；`=1` 模式 = **裁掉分片**（省 25.75 GB/rank/层）。

**交付现状**：`V41_ENGRAM_DEVICE_INDEX=0` ⇒ 两处一致（都关）⇒ **本条当前不发作**。但**只要按 B1 改成 `auto` 就会踩到**；改成 `"1"` 则不踩。
**修法**：两处共用同一个解析函数（或至少把 `engram_hbm.py:276` 放宽到与 `model.py` 同口径），并明确 `auto` 是否该裁分片。

---

### C3 · CED 系列在非 CED 实例上是**死开关**（无害但误导）

【实测】交付 env 有 9 个 `V41_CED_*` + `V41_CED_ALLOW_DSPARK=0`，而 `V41_CED_ROLE=""`：

| 变量 | 交付值 | 实际效果 |
|---|---|---|
| `V41_CED_ALLOW_DSPARK` | `0` | **无**（`model.py:927` 只在 `ced_role=="decode"` 时才检查；这里 role 为空） |
| `V41_CED_SWA_CLIP` | `1` | 无（CED attention 未加载） |
| `V41_CED_GRAPH_PROMPT_TAIL_EAGER` | `0` | 无 |
| `V41_CED_DYNAMIC_SPEC_FULL_GRAPHS` | `0` | 无 |
| `V41_CED_BLOCK_TRACE` / `_KVGEOM` / `_CAPTURE_DECODE` / `_SOURCE_COMPARE` / `_*_SNAPSHOT_*` | `0` / 空 | 无 |

**风险**：读日志的人会以为"CED 的护栏都关了"，进而对**真的 CED 实例**误判。**修法**：非 CED 形态不必传这些（或在 `docker inspect` 输出里加注释）。

---

### C4 · 其余需要注意但**当前合理**的项

| 变量 | 交付值 | 判断 |
|---|---|---|
| `V41_SLOT_MAP_FUSED` | `0` | 【实测·代码】`serve_a2.sh:1111` 只在带 `--decode-context-parallel-size` 时把默认改 `on`；本实例无 DCP ⇒ `0` 合理 |
| `V41_MOE_ZERO_INVALID` / `_NONFINITE` | `0` / `0` | 功能关（文件路径已设但不读）⇒ 无影响 |
| `V41_GATE_MAX_PREFILL` | `8` | 与 `serve_a2.sh:342` 默认一致 |
| `V41_ENGRAM_GATE_MAX_TOKENS` | `8192` | 【实测】= `BAT_TOKENS` = `MAX_TOKENS` ⇒ 三处一致 ✅ |
| `V41_ENGRAM_GATE_HOIST` | `0` | `serve_a2.sh` 硬传 0 ⇒ 一致 |
| `V41_CED_SOURCE_COMPARE_CHUNKS` | `1` | 但 `SOURCE_COMPARE=0` ⇒ 不生效 |

---

## 3. 建议动作（每条：改哪里 / 怎么改 / 如何验证）

| 优先级 | 动作 | 改哪里 | 怎么改 | 验证 |
|---|---|---|---|---|
| **P0** | **决定并落地 `ENGRAM_DEVICE_INDEX`** | 交付起服命令 / `serve_a2.sh` 调用方 | 去掉 `ENGRAM_DEVICE_INDEX=0`（走 `auto`）**或**显式 `=1` | 日志出现 `[DEVICE-INDEX] 能力探测通过`（auto）或 `[DEVICE-INDEX] ... 已启用`；`[bneck] d2h` 显著下降 |
| **P0** | **A/B 量化 device-index 收益**（这是唯一能回答"值不值"的路） | 评测脚本 | 同一批 prompt、同 rep 数，两臂：`ENGRAM_DEVICE_INDEX=0` vs `=1`；报 `(ms/step, A, tok/s)` 三元组 | 见 `docs/PERF-METRIC-REFRAME-20261004.md` 的口径 |
| **P1** | **修 2 处 `bool(...)`** | `engram_hbm.py:376`、`:474` | 改成 `.strip().lower() in ("1","true","yes","on")` | 设 `V41_ENGRAM_HOST_RESIDENT=0` 后日志/行为确实变为 False（用负控验） |
| **P1** | **重启使 route-probe 修复生效** | 运维（**与 P0 同一次**） | 不要再单独重启 | `grep -c route-probe serve.log` ≈ 0 |
| **P2** | **关掉 bneck 打印** | 交付起服命令 | 加 `-e V41_BNECK_PRINT_EVERY=0`（**不要**用 `ROUTE_PROBE_EVERY=0`） | `grep -c '\[bneck\]'` ≈ 0 |
| **P2** | **统一四开关默认** | `scripts/serve_a2.sh` / `serve_v2.sh` / `deploy/a3-ced-pd/launch/_common.sh` | 收敛到单一来源；在起服日志打印实际生效值 | `tools/` 加一条 selftest（已有 `selftest_ced_defaults.sh` 可仿） |
| **P3** | **统一 `DEVICE_INDEX` 双解析** | `engram_hbm.py:276` ↔ `model.py:216-218` | 共用解析函数；明确 `auto` 是否裁分片 | 设 `auto` 时两处日志对同一结论 |
| **P3** | **`/tmp` 文件开关加护栏** | 四个 `*_FILE` 默认值 | 交付指向永不存在的路径，或探测到非 `stock` 就 WARNING | 手写 `nohost` 后日志有 WARNING |

---

## 4. 「已查、确认无害」清单（避免后人重复查）

| 组 | 变量 | 为什么无害 |
|---|---|---|
| **严格 `== "1"` 的优化开关，交付值均为 `"1"`** | `V41_MOE_COMM_ALLGATHER`、`V41_O_PROJ_2D`、`V41_QLI_NO_CANDIDATE`、`V41_ROPE_IDXSEL`、`V41_MOE_MASK_RANGE`、`V41_ENGRAM_JIT` | 【实测】与 `serve_a2.sh:200-212`「已验证开关（默认全开）」逐项一致 ✅ |
| **`int(... or D)` 家族**（除 `ROUTE_PROBE_EVERY`） | `V41_ENGRAM_DEVICE_PAGES`、`V41_ENGRAM_DEVICE_GRAPH_MAX`、`V41_BNECK_PRINT_EVERY`、`V41_BNECK_DECODE_TOKENS`、`DSPARK_CAPTURE_MAXSEQLEN`、`DSPARK_CAPTURE_SEQ_LEN` | 【实测】`"0"` 是 truthy ⇒ 正常解析为 0，**不是**被吞成默认值（§0.2） |
| **`... or "off"` 字符串族** | `V41_ENGRAM_LOCAL_METADATA`、`V41_ENGRAM_LOCAL_OWNER`、`V41_ROUTE_PIPE` | 同上；且当前均未设 ⇒ 默认 `off` |
| **全部探针 = 0（除 route-probe/bneck）** | `DSPARK_DSA_PROBE`、`DSPARK_DSA_WRITE_PROBE`、`DSPARK_GRAPH_PTR_PROBE`、`DSPARK_STEP_PROBE`、`DSPARK_ROW_DUMP`、`DSPARK_TOKEN_DUMP`、`DSPARK_DSA_PROBE_CAPTURE`、`DSPARK_GRAPH_DEBUG`、`DSPARK_GRAPH_SHADOW_EAGER`、`V41_MOE_INVALID_PROBE`、`DSPARK_DISPATCH_DIAG_STEPS=0` | 【实测】交付值均为 0/未设 ⇒ 关 |
| **`DSPARK_CAPTURE_*` 与 `DRAFT_GRAPH=1` 配套** | `DSPARK_CAPTURE_VALUE_FIX=1`、`DSPARK_CAPTURE_NCTX_FIX=1`、`DSPARK_GRAPH_CAPTURE_METADATA=1`、`DSPARK_SWA_INDICES_RESIDENT=1`、`DSPARK_DRAFT_USE_CUDAGRAPH=1` | 【实测】正是 `serve_a2.sh:1400-1416` 注释要求的「四件套最小集合」，缺一件会让 `A` 从 ~2.6 掉到 1.07 ✅ |
| **显式关的 CED 开关** | `V41_CED_SNAPSHOT_POS`、`V41_CED_H20_SNAPSHOT_POS`、`V41_CED_LAYER_SNAPSHOT_POS`、`V41_CED_SOURCE_COMPARE`、`V41_ENGRAM_HIST_TRACE_POS`、`V41_CED_BLOCK_DUMP_DIR` | 【实测】均为空/0 ⇒ 关 |
| **MoE 数值护栏** | `V41_MOE_ZERO_INVALID=0`、`V41_MOE_ZERO_NONFINITE=0` | 关；文件路径虽设但不读 |
| **`V41_DECODE_API_GUARD=1`** | — | 【实测】`serve.log` 里 `decode_guard=off`：`v41_decode_guard.py:68` 要求 `V41_CED_ROLE=="decode"`，本实例 role 为空 ⇒ 不注册 middleware（**非 PD 场景不需要**，无害） |
| **`V41_BNECK_MODE_FILE` 等四个 `/tmp` 路径** | — | 【实测】交付未设、文件不存在 ⇒ `refresh()` 的 `os.stat` 抛 `OSError` ⇒ `return self`（保持 `stock`）⇒ 当前零影响（风险见 C1） |
| **`V41_IDS64_HOIST`、`V41_ENGRAM_WITH_DUMMY`、`V41_ENGRAM_PAD_SKIP`、`V41_DUMMY_WO_A_FIX`、`V41_ENGRAM_PAGELESS_STRICT`、`V41_DYNSPEC_BT_PERSIST`、`V41_DSPARK_SHAPE_PROBE`、`DSPARK_FIA_PAD_REQS_FIX`、`DSPARK_DISPATCH_UNIQUE`、`DSPARK_GRAPH_AB_LEGACY_CAPTURES`、`DSPARK_GRAPH_DEVICE_METADATA_FROM`、`DSPARK_ROW_DUMP_STEPS`** | — | 交付均未设；默认 `0`；其中多数在 `== "1"` 门下 ⇒ 关 |

---

## 5. 复现本审计的命令（全部只读）

```bash
# 0) 交付实际 env（只读 inspect，不要 exec）
sudo docker inspect dsv41-tp8k5 --format '{{range .Config.Env}}{{println .}}{{end}}' | sort

# 1) 枚举所有读取点（含默认值）
cd ~/cedpd-repo && grep -rn 'os\.environ\.get' patches/files/*.py patches/files/draft/*.py | grep -v '\.bak'

# 2) 只挑 bool(...) 家族（真 A 类）
grep -rn 'bool(os\.environ\|bool(_os' patches/files/*.py patches/files/draft/*.py | grep -v '\.bak'

# 3) 交付实例运行日志（宿主 bind mount 源，只读）
D=~/cedpd-repo/results/fix_1004_1645
grep -ac '\[route-probe\]' $D/serve.log
grep -ac '\[bneck\]'       $D/serve.log
grep -ac 'DEVICE-INDEX'    $D/serve.log     # 期望 >0，实测 0
grep -ao 'd2h=[0-9.]*' $D/serve.log | sed 's/d2h=//' | sort -g | awk '{a[NR]=$1} END{print "p50="a[int(NR/2)], "p90="a[int(NR*0.9)], "n="NR}'
```

**本次审计未修改任何文件**（报告本身除外），未启停任何服务，未 `exec` 进 `dsv41-tp8k5`。
