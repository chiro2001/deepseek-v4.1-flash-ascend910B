# 上下文交接：DSpark × DCP 打通 + 8 卡绝对性能（2026-10-01 10:40）

## 0. 本线程的目标（用户最新口径）

1. **DSpark 必须开** —— 为了**整体 token 输出速度**（`ms/token` / 吞吐），
   不是 `ms/step`。历史同条件 A/B：`SPEC=0` 24.35 ms/step (A=1, 41 tok/s)
   → `SPEC=1` 32.68 ms/step (A=3.10, **94.79 tok/s**)，**ms/token 10.54 vs 24.35**。
2. **把 DSpark 带来的开销逐个拆解**：为什么增加、能否用**单算子优化**去降。
3. 用户已纠正过我一次：**"开推测解码后步骤时延会增加很多，不是免费的"** —— 我此前误用了
   "A2 基线 38.5 ms"当 SPEC=0（其实那个基线本身就开着 DSpark）。**不要再犯**。

---

## 1. ★★★ 当前最大的 blocker（必须先解决才能谈拆解）

### DSpark × DCP>1 是**上游明确不支持**的组合，第一个请求必崩

| 组合 | 结果 |
|---|---|
| `SPEC=0` + DCP8 | ✅ 32.58 ms/step（本线程实测） |
| `SPEC=1` + DCP1 | ✅ 历史 CED-PD |
| **`SPEC=1` + DCP>1** | ❌ **第一个请求 500，8 worker 全挂** |

**精确根因（两个独立 bug，都要修）**：

**Bug #1 —— 分支遗漏**（`vllm_ascend/spec_decode/llm_base_proposer.py`）
- `speculative.py:984`：`method in ("dflash","dspark") ⇒ parallel_drafting=True`
- ⇒ `extra_slots_per_request = 7`；`pass_hidden_states_to_model=True and method!="dflash"`
  ⇒ `net_num_new_slots_per_request = 7-1 = 6 > 0` ⇒ **`needs_extra_input_slots = True`**
- `set_inputs_first_pass`：
  - `if not needs_extra_input_slots`（**默认 EAGLE 通路**）→ ✅ 有 DCP 处理
    （调 `dcp_manager.prepare_spec_decode_first_pass_inputs`），返回真 `long_seq_args`
  - `else`（**我们走这条**）→ ❌ **完全没有 DCP 处理**，第 2497 行左右
    `return total_num_output_tokens, token_indices_to_sample, new_cad, None` ← **硬编码 None**
- `_propose:1367`：`if dcp_manager is not None: assert long_seq_args is not None` ⇒ **崩**
- `long_seq_args` 下游用途（`llm_base_proposer.py:1654`）：
  `dcp_manager.prepare_spec_decode_mtp_drafting_inputs(..., ori_token_indices_to_sample=...)`
  ⇒ 算 DCP 的 MTP 草稿槽位。**`_propose` 只用了元组第 2 项**（`_` 丢掉 `decode_query_lens`）。
- `dcp_utils.py:167 prepare_spec_decode_first_pass_inputs` 返回
  `long_seq_args=(decode_query_lens, original_sample_indices)`，其中
  `original_sample_indices = token_indices_to_sample.clone()`、`decode_query_lens = self.query_lens_full.cpu[:num_decode_reqs]`。

**★ 上游保护写错了架构名**（`speculative.py:988-993`）：
```python
if (self.method == "dspark"
    and "K3DSparkModel" in self.draft_model_config.architectures   # ← 只挡这个名字
    and decode_context_parallel_size > 1):
    raise ValueError("MLA DSpark does not currently support decode context parallelism; ...")
```
我们的 draft 架构是 **`DSparkDeepseekV41ForCausalLM`** ⇒ 不匹配 ⇒ **绕过保护** ⇒ 撞上 #2。

**Bug #2 —— 守卫过窄**（`vllm_ascend/worker/block_table.py:419`）
- 报错原文：`ValueError: Device tensor inputs are only supported for CP draft slot mapping.`
- 完整调用链（实测 traceback）：
```
model_runner_v1.py:1415  _prepare_input
dcp_utils.py:323         rebuild_async_spec_decode_*  ← 主动传 device 张量
block_table.py:865       MultiGroupBlockTable.compute_slot_mapping_draft
block_table.py:419       raise ValueError(...)
```
- `block_table.py::BlockTable.compute_slot_mapping_draft`（第 186 行，**另一个同名方法**）：
```python
if self.dcp_world_size > 1:                     # ← 注意用的是 dcp_world_size
    self._compute_dcp_slot_mapping(req_indices, positions)   # device 张量 OK
else:
    if isinstance(req_indices, torch.Tensor) and req_indices.device.type != "cpu":
        raise ValueError("Device tensor inputs are only supported for CP draft slot mapping.")
    ...（numpy 路径）
```
- **★ 这证明 DSpark × DCP 是部分实现过的**：`dcp_utils.py:323` 那段是**专门为
  「DCP + 推测解码」写的 device 侧重建路径**。
- 我们掉进 else 的原因：**我的 overlay 把 draft 的 SWA 组归一成复制态**
  （`effective_dcp_world_size=1`，见 `patch_v41_dcp.py::resolve_group_dcp`），
  于是 `dcp_world_size > 1` 仍成立？—— **注意两个名字**：
  `self.dcp_world_size`（真 DCP 度，未归一）与 `self.effective_dcp_world_size`（归一后）。
  第 201 行用的是 **`dcp_world_size`**，所以它其实**会**走 `_compute_dcp_slot_mapping`。
  **⇒ 必须实测确认崩的是哪一个分支**（`block_table.py:419` 属于 `BlockTable.compute_slot_mapping`
  还是 `compute_slot_mapping_draft` —— 两者行号相近，第 419 行在 `compute_slot_mapping` 里）。
  **这是接手第一件要核实的事。**

---

## 2. ★ 已经建好的 tiny-dspark 夹具（关键资产，迭代 2–3 分钟/轮）

### 2.1 模型：`~/models/out/v41-tiny-dspark`（**已建好，验证过能起服**）

**不需要新权重** —— tiny 是 `LOAD_FORMAT=dummy`，目录里没有任何权重文件。
只是 config 变体（从 `~/models/out/v41-tiny` 复制而来，**原目录未动**）：

| `text_config` 字段 | v41-tiny | v41-tiny-dspark |
|---|---|---|
| `num_nextn_predict_layers` | 0 | **3** |
| `dspark_target_layer_ids` | [] | **[37,38,39]** |
| `dspark_n_routed_experts` | 128 | **8** |
| `dspark_num_experts_per_tok` | 3 | **2** |

派生记录：`v41-tiny-dspark/dspark_derivation.json`；
依据：`v41-tiny/l1_dummy_provenance.json` 显示 tiny 转换时**只清掉了 dspark 的这两个开关**，
其余结构参数（`block_size=5`/`markov_rank=256`/`noise_token_id`）原本就在。
`dspark_n_routed_experts` 被 `patch_speculative_config.py:76` 用来覆盖 draft 的 `n_routed_experts`。

### 2.2 起服（**已修好硬编码 SPEC=0 的坑**）

```bash
ssh a3-21 'cd ~/tmp/dcp2tiny && setsid nohup env \
  MODEL=$HOME/models/out/v41-tiny-dspark \
  SPEC=1 SP_TOKENS=7 DRAFT_GRAPH=1 DCP=2 DEVS="2 3" \
  OUTROOT=$HOME/tmp/tinyspark NAME=dsv41-tinyspark PORT=19310 \
  RUN_ID=tinyspark_dcp2_$(date +%m%d_%H%M%S) \
  nohup bash launch_dcp2_tiny.sh > ~/tmp/tinyspark.nohup.log 2>&1 < /dev/null & disown'
```
- `launch_dcp2_tiny.sh` 第 139 行原来硬编码 `SPEC=0`（**会静默吞掉 DSpark**），
  已改为 `${SPEC:-0}` + 透传 `SP_TOKENS`/`DRAFT_GRAPH`（备份 `.bak_spec`）。
- **实测**：起服成功，`speculative_config=SpeculativeConfig(method='dspark', num_spec_tokens=7)`，
  `load_format=dummy`，health=200，0 error，**~2 分钟**。
- **第一个请求 = 500，复现同一个 blocker**（TP1 是 Bug#1、TP0 是 Bug#2）。

---

## 3. 另一条线：8 卡绝对性能（子代理 `line_a_absperf`）

### 3.1 它的产出（回合已结束，**未完成集成**）
- **8 卡基线 32.58 ms/step**（3 轮 32.86/32.49/32.58，SPEC=0）
- profiler 已抓取解析；工具在 `~/projects/dsv41/lineA/tools/{opfwd,commstats,hcom_dedup}.py`
- 通信分组：**503 = MoE 共享专家（82 次）、097 = DCP merge（76 次）、374 = q gather（38 次）**
- 单卡 chip4 wo_a：vendor **14.38 µs** → M16 **7.32** → Triton **5.69**
- ★ **关键纠正：QBMV3 三合一真实上限只有 0.4–0.6 ms/step，不是邻居文档的 2–3 ms**
  （生产 decode 是 **M=1**，5 个形状合计 1.99 ms/step）
- ⚠️ **未做**：QBMV3 / wo_a / M16 的模型集成与端到端 A/B

### 3.2 它算出的疑点（值得继续）
- 设备总时长 **25.35 ms/step**，步跨 **40.47 ms** ⇒ **空闲 37%**
- 若那 15 ms 是 host/同步开销，**减少算子个数**（不只减算子时间）才值钱 ——
  与本线程"每图节点 ~3 µs"的实测同理

---

## 4. 我这条线已完成的：merge 融合算子（AscendC，**单卡验证通过，未上 8 卡**）

| 文件 | 内容 |
|---|---|
| `experimental/v41-dcp/ascendc/merge/op_kernel/v41_merge.asc` | 两个 kernel：`pre`（打包 pack）/`post`（合并出输出） |
| `.../op_host/v41_merge_host.cpp` | ctypes shim（走 torch_npu 当前 stream ⇒ 可被 ACL graph 捕获） |
| `.../bench_merge.py` | 单卡数值验证 + 计时 |
| `.../check_bufs.sh` | **静态自检**：每个 TBuf/TQue 必须有且只有一条 InitBuffer |
| `overlay/vllm_ascend/attention/v41_merge_kernel.py` | 运行时 wrapper（ctypes + tiling/pack/out 常驻缓存 + `available()` 安全门） |
| `overlay/.../dsa_v41.py` | 集成：`_v41_merge_kernel_on()`（env `V41_DCP_MERGE_KERNEL=1`）+ `_merge_kd` 早退分支 |

**实测（chip6，`bench_merge.py`）**：

| T | pre | post | 合计/层 | ×38 层 | 数值 |
|---:|---:|---:|---:|---:|---|
| **1（生产 decode）** | 7.08 µs | 6.61 µs | **13.68 µs** | **0.520 ms/step** | **逐位一致** |
| 16 | 19.99 | 6.39 | 26.38 | 1.002 | 逐位一致 |
| 96 | 79.02 | 8.97 | 88.00 | 3.344 | 1 ULP |

- `.so` 已放在 `~/dcpw/vllm_ascend/attention/v41_merge_kernel.so`（629808 B）
- `serve_a2.sh` 原来**只挂 `*.py`**（`.so` 进不了容器 ⇒ 融合算子静默退回 Python 路径），
  已改为同时挂 `.so`（备份 `serve_a2.sh.bak_dcpson`）

**★ 四个 AscendC 硬坑（都踩过）**：
1. kernel 入口必须 `extern "C" __global__ __vector__`（否则向量指令被静默编译掉）
2. fp32→bf16 必须 `RoundMode::CAST_RINT`（`CAST_NONE` 静默不写）
3. **任何 TBuf/TQue 用前必须 InitBuffer 且恰好一条** —— 踩了两次，
   第二次是脚本按行号改代码时误删 `pre` 的 `qDen_` InitBuffer ⇒ vector core exception，
   而且**换 chip 后仍复现**（一度误判设备坏了）。已加 `check_bufs.sh`
4. **不手写 SetFlag/WaitFlag** —— 第一版导致"进程内首次 launch 完全不产出、第 2 次起才对"；
   改用标准 `TQue` 由框架管同步。另 `aicore` 不支持 double。

---

## 5. 当前环境状态（2026-10-01 10:40）

| 容器 | chip | 端口 | 状态 |
|---|---|---|---|
| `dsv41-abs` | 8–15 | 19210 | **正在起**（10:28 提交，8–10 分钟） |
| `dsv41-tinyspark` | 2–3 | 19310 | 运行中但**第一个请求即崩**（DSpark blocker） |
| `dsv41-merge` | 6 | — | 空转（我做 AscendC 验证用） |
| `dsv41-lineA` | 4 | — | **空转**（子代理回合已结束） |
| `dsv41-op-peak` | 7 | — | 邻居的，**chip7 的 AIV 被我打坏过**（fp32 `Div`） |

**chip 占用**：`0 1 8 9 10 11 12 13 14 15`（0/1 是别人的）

⚠️ **`block_table.py` 的 overlay 与容器 md5 不一致**：
- overlay（`~/dcpw/.../block_table.py`）= `cde938dc09fd08e62bb11fa13e4ee0ab`
- `dsv41-merge` 容器内 = `2ab8541624bd89e781e8438daef20f44`（容器起在 1 小时前，挂的是旧版）
⇒ **任何要改 `block_table.py` 的实验都必须新起容器**（或确认挂载的是当前 overlay）。

**已改的脚本（都有 `.bak` 备份）**：
- `~/dcp_stage_capacity.sh` —— `SPEC=0` 硬编码 → `${SPEC:-0}` + 透传 SP_TOKENS/DRAFT_GRAPH（`.bak_spec`）
- `~/tmp/dcp2tiny/launch_dcp2_tiny.sh` —— 同上（`.bak_spec`）
- `~/cedpd-repo/scripts/serve_a2.sh` —— DCP 挂载放行 `.so`（`.bak_dcpson`）

---

## 6. 接手后第一件要做的事（按优先级）

1. **核实 Bug#2 的精确定位**：读 `block_table.py` 第 419 行所属方法，
   并确认 `compute_slot_mapping_draft`（第 186 行）用的是 `dcp_world_size`（未归一）
   还是 `effective_dcp_world_size`。这决定修法是"补 device 侧实现"还是"改判定条件"。
2. **在 overlay 里加 `vllm_ascend/worker/dcp_utils.py`**（目前 overlay **没有**这个文件，
   而 Bug#1 的修法大概率要动它或其调用方 `llm_base_proposer.py`；
   `llm_base_proposer.py` 也**不在 overlay 里**，需要新增挂载）。
3. **修 Bug#1**：给 `else` 分支补 DCP 准备，最小改法：
   `long_seq_args = (dcp_manager.query_lens_full.cpu[:num_decode_reqs], token_indices_to_sample.clone())`
   —— 但**必须验证 parallel-drafting 的 token 布局与 `if` 分支可比**
   （它走 `CopyAndExpandEagleInputs`，每请求多 N 个槽位）。
4. **每轮用 tiny 验证**（2–3 分钟）：`ask2.py` 发请求 → 看是否还崩 → 看 A 值。
   判据：`A > 1.0` 才算草稿生效；`A ≈ 1.0` = 草稿算错但被 verify 拒掉（输出仍正确，不静默算错）。
5. 两条都通后再回 8 卡做 DSpark 开销拆解（三个桶：draft 前向 / target M=1→8 / 验证+head）。

---

## 7. 关键文档索引

| 文档 | 内容 |
|---|---|
| `docs/V41-DSPARK-X-DCP-BLOCKER-20261001.md`（246 行） | ★ 本文的详细版：两个 bug 的完整代码证据、tiny-dspark 夹具、三条出路 |
| `docs/V41-DSPARK-COST-BREAKDOWN-PLAN-20261001.md`（177 行） | 拆解方案：三个桶 + 必答项（DCP 摊薄）+ 单算子候选清单 |
| `docs/V41-DSPARK-X-DCP-JOINT-OPT-20261001.md`（218 行） | 初版分析（**§0 有我对"免费"那句话的更正**） |
| `docs/V41-DCP8-CORRECTNESS-FIX-AND-PERF-PLAN-20261001.md`（588 行） | 6 轮全部实测：正确性修复、13 项尝试台账、节点成本模型、AscendC §2.13 |
| `~/projects/dsv41/lineA/tools/` | 子代理的 profiler 解析工具 |

## 8. 用户偏好与纪律

- **始终用简体中文回复**
- 结论必须标 **【实测】/【推断】/【未确认】**
- **重启 8 卡服务前必须先问子代理**（它可能在用）
- 每次改动后必须跑回归：短问答 `17×23` → `391`、T=904 → `Q7`、长针 2000/8000/16000
- 不用 `/tmp` 存产物（用 `~/tmp/`）；禁用 `rm -f`（用 `mv` 到 `.bak`）
- 用户会质疑结论，**允许并鼓励反驳**，但必须有实测证据
