# 064 — ②a（draft 的 KV 走 INT8 + 入图）**单卡跑通**：病灶钉到 `dsa_attn_kv_plan.py:125`，修法是「两平面 + 索引重映射」

> 2026-09-22 15:0x–15:3x CST（**远端 A3-node1 时间**；本机比远端快约 7 min）。执行：子代理 **DS_draft_graph_int8**。
> 机器：**A3（A3-node1）的 c1 / c2**（`tools/a3_chip.sh`）。**全程未碰 c0**（`T_draftceiling` 的 8 卡臂 + Phy-ID 8–15 在跑）、
> 未碰 `dsv41-a3`（保持 Exited）/ `mooncake-master` / 别人的容器 / Phy-ID 0–7；未手设 `ASCEND_RT_VISIBLE_DEVICES`；
> 每臂起服前查 `/dev/shm`（`Used 0`）；未用 `/tmp`；未写 `upstream-v41/`；**未发 PR / issue / 评论**。
> 模型：`agents/T_draftceiling/models/model-tiny-draft`（tiny + 只还原 draft 两个键 ⇒ 有 g12 draft 组）= **与 056 同一几何**。
> 产物：本日志 + `logs/raw/064-draft-graph-int8/` + `agents/DS_draft_graph_int8/`。
> 标记：**【实测】**= 有判别力的判据跑出来的数；**【推断】**；**【未确认】**。

---

## 0. 一句话结论

> ★★★ **②a 在单卡上「跑通」了，而且是入图跑通的**：`capture_finished=1` / `EE1016=0` / `Not_Supported=0`，
> 端到端两轮 `ok=1`，**输出 sha 与 BF16 基线逐字相同**（`b8329c05…`），**零 MTE、零异常**。
> 并且**第一次**让 ②a 产出了 spec-decode 读数（此前 056 是「无读数」）。

| 问题 | 056（修复前） | **本轮（修复后）** |
|---|---|---|
| **Q1 图捕获** | ✅ | ✅ **仍然过**（`capture_finished=1`，本轮的判别性更强：**同臂内补丁确实生效**，见 §5） |
| **Q2 容量** | ✅ 39,846 | ✅ 39,846（逐字复现） |
| **Q3 投机解码** | ⛔ **无读数**（引擎死在第一个 spec step） | ★★ **有读数 + 端到端跑通**：`ok=1`；spec 计数器 `draft_tokens=40 / accepted=20`（per-pos `8,3,3,3,3`） |
| **病灶** | 【推断】"`dsa_v1.py` 缺量化存取" | ★★★ **【实测】`attention/dsa_attn_kv_plan.py:125`**（`npu_scatter_nd_update_sk(cache=tuple)`）+ 读侧 `ori_sparse_indices` 的**越界寻址** |

**★ 但有一条必须一起看**：**eager（非入图）+ INT8** 那一格，输出与 BF16 不同（`a6a760e9…` vs `b8329c05…`）；
**入图 + INT8 那一格与 BF16 逐字相同**。⇒ 交付形态（入图）**没有**这个现象，但 **eager 格的差异原因仍【未确认】**（§6 第 1 条）。

---

## 1. ★★★ 病灶（056 拿不到的那条「首个异常」）

056 §4.3 把病灶**降级成【推断】**（因为探针钩错了类、首个异常在日志里完全看不见）。
本轮把探针钩到真身并**拿到了决定性 traceback**【实测】：

```
[DS-DBG v3] [17.577s] ★★ proposer._run_merged_draft RAISED RuntimeError:
    _C_ascend::npu_scatter_nd_update_sk() Expected a value of type 'Tensor' for argument 'var'
    but instead found type 'tuple'.
  File "spec_decode/llm_base_proposer.py", line 1262, in _run_merged_draft
    self.build_model_inputs_first_pass(...)
  File "spec_decode/dflash_proposer.py", line 304, in build_model_inputs_first_pass
    self.model.precompute_and_store_context_kv(...)
  File "models/deepseek_v4/dspark.py", line 412, in precompute_and_store_context_kv
  File "models/deepseek_v4/dspark.py", line 259, in precompute_and_store_context_kv
  File "models/deepseek_v4/dspark.py", line 243, in _store_standard_swa_kv
    get_dsa_attn_kv_plan(...).dsa_kv_compress_scatter(swa_kv_cache, shared_kv, slot_mapping)
★ File "attention/dsa_attn_kv_plan.py", line 125, in dsa_kv_compress_scatter
    torch.ops._C_ascend.npu_scatter_nd_update_sk(cache, slot_mapping, x)
RuntimeError: ... found type 'tuple'
```

### 1.1 ★ 病灶的**正确表述**（替换 056 的那条【推断】）

**不是** "`dsa_v1.py` 缺量化存取"，而是：

> **`DsaAttnKvPlan.dsa_kv_compress_scatter` 只会写单平面**，而 ②a 让 draft 的 SWA 平面变成了
> INT8 `(payload, scale)` 两平面（页 66,560 B）⇒ 调用方把**整个 tuple** 交进来时直接炸。

★ 一个 helper 被**三处共用**（这就是"最小面"的来源）：

| 位置 | 谁调 | 本轮是否修 |
|---|---|---|
| **写①** `models/deepseek_v4/dspark.py:243::_store_standard_swa_kv` | context 预存（第一批 draft 的窗口行） | ✅ **同一个 helper，一处覆盖** |
| **写②** `attention/dsa_v1.py::_mla_prolog_single_stream` | forward 内的窗口写 | ✅ 同一 helper |
| **读** `attention/dsa_v1.py::_forward_attention` | `attn_op(..., ori_kv=swa_kv_cache)` | ✅ 单独一处（§2） |

### 1.2 ★★ 读侧的第二个病灶：`ori_sparse_indices` 是 **flat slot id**（越界读）

把写侧修好后，失败**前移**到读侧：`npu_sparse_attn_sharedkv() ... argument 'ori_kv' ... found type 'tuple'`。
照 056 的路线（"复用 `kv8_ori_plane`：只反量化窗口行 + 重指向 block table"）修完后，拿到**设备侧 AI Core 错误**：

```
The error from device(chipId:3, dieId:1) ... errorStr: MTE accesses an invalid GM address ...
RuntimeError: ... current working operator name is aclnnNeg
```

**根因【实测·源码】**：draft 走的是 `compress_ratio<=1` 分支，它会设
`attn_kwargs["ori_sparse_indices"] = dspark_swa_indices`（`dsa_v1.py:2089-2090`），而
`build_dspark_swa_indices` 造的是 **flat slot id**：

```python
slot_ids = (block_ids * block_size + block_offsets).to(torch.int32)   # dsa_v1.py:392
```

—— 它**绕开 `ori_block_table`、直接按 flat 槽位索引 `ori_kv`**，值域覆盖**整块物理 cache**
（本几何 `3795×128 = 485,760` 行）。而 `kv8_ori_plane` 给出的 scratch 只有
`num_reqs × pages_per_req = 1×3 = 384` 行 ⇒ **必然越界**。

**★ 判别性实验（两臂，只有一处变量）**：

| 臂 | 读侧 `ori_kv` | 结果 |
|---|---|---|
| `ds-remap2`（当时是"只重指向 block table"的版本） | 3 页 scratch（384 行） | ⛔ MTE / aicore exception |
| **`ds-full2`（诊断：全尺寸 485,760 行 BF16 scratch）** | 全物理 cache | ★ **零 MTE、`ok=1`、两轮 sha 逐字相同** |

⇒ **假设被证实**：越界读来自 flat slot 寻址。`full` 只是判别臂，**不是交付形态**（临时 ~0.5 GB/layer）。

---

## 2. 修法（两处，都在 `agents/DS_draft_graph_int8/scripts/mk_pkg_dsi.py` 里锚点化生成）

### 2.1 写侧（`--store`，默认开）

在 `dsa_kv_compress_scatter` 里加两平面分支：`cache` 是 tuple ⇒ 按页内行做 per-group 动态量化
（口径与目标侧 `kv8_quantize_latent` **逐字一致**：dim 512 / 4 组 / fp16 scale），再分别散射。
**形状规则照抄目标侧**：值的尾部形状必须等于平面的尾部形状 ⇒ `payload=(T,1,512) int8` / `scale=(T,1,4) fp16`。

### 2.2 读侧（`--read`，默认开；`DS_2PLANE_READ=remap`）

两步，**都必须是设备侧、无 host 同步**（否则图捕获会 `EE1016`）：

1. 复用**已发布**的 `kv8_ori_plane`：只把本步真正读到的窗口行反量化到 BF16 scratch，并把 **block_table 重指向 scratch**；
2. ★ **把 flat slot id 重映射到 scratch 坐标** —— 先用**原始** block table 建「物理页 → scratch 页」的映射
   （`phys = gather(orig_table, first_block + delta)`，`delta < blocks_per_req`），再按
   `blk = slot // bs` 在 ≤3 个候选里匹配；匹配不到或原值 `-1` ⇒ 输出 `-1`（保持掩码语义）。

★ **第二次踩坑（已记）**：我第一版拿 **物理页号** 直接和 **逻辑块号**（`first_block`）比较 ⇒ `kept=0`
（所有可见槽位被误判越界、全置 -1）⇒ 算子报 `aclnnSparseAttnSharedkv failed`。改成"物理→scratch"映射后
`kept=665/665`。**这是"看似跑通实则语义错"的典型，靠 `kept` 这个自报判据当场抓住。**

### 2.3 出包机械门（`mk_pkg_dsi.py`）

锚点各命中**恰好 1 次**、`ast.parse` + `py_compile`（**编译到临时文件，不往包里写 `.pyc`，并删陈旧字节码**）、
与 base 的**源码差异恰好 = 1（只 store）/ 2（store+read）**、`dsa_v41.py` md5 必须 = 发布件 `94aeebb7…`。

---

## 3. ★★ 对称臂总账（同一份探针 / 同一个包 / 同一几何）

| 臂 | 包 | graph | draft KV | 容量 | `capture_finished` | ok | chunks | **输出 sha** | MTE | 异常 |
|---|---|---:|---|---:|---:|---:|---:|---|---:|---|
| `ds-ctl` | 旧 `pkg-ddi`（无补丁） | 0 | BF16 | 23,651 | n/a | 1 | 3 | `b8329c05…` | 0 | 无 |
| `ds-ctl2` | **`pkg-dsi`** | 0 | BF16 | — | n/a | 1 | 3 | `b8329c05…` | 0 | 无 |
| `ds-remap2` | `pkg-dsi` | 0 | **INT8** | — | n/a | 1 | **2** | `a6a760e9…` | 0 | 无 |
| `ds-full2`（诊断） | `pkg-dsi` | 0 | **INT8** | — | n/a | 1 | **2** | `a6a760e9…` | 0 | 无 |
| ★ **`ds-graph2`** | `pkg-dsi` | **1** | **INT8** | — | **1** | 1 | 3 | ★ **`b8329c05…`** | 0 | 无 |

**四条读数**：

1. ★ **`ds-ctl` ≡ `ds-ctl2`（逐字）** ⇒ **补丁对 BF16 是 no-op**（反向门）。这条同时证明"改动面没有外溢"。
2. ★ **`ds-full2` ≡ `ds-remap2`（逐字）** ⇒ **便宜的 remap 与"全尺寸安全版"语义等价**——不只是"不崩"。
3. ★★ **`ds-graph2`（入图 + INT8）≡ `ds-ctl2`（eager + BF16）** ⇒ **入图形态下 ②a 的输出与基线逐字相同**。
4. ⚠️ **`ds-remap2`（eager + INT8）≠ 其它三格** ⇒ 见 §6 第 1 条（**【未确认】**）。

### 3.1 ★ 证伪"补丁没生效"（AGENTS §5b 第 2 条）

入图臂里打的是**同一份 v3 探针关掉后**的服务，所以必须用**补丁自己的 trace** 证伪：

```
[DS-2plane] store rows=256 ... payload_plane=(3795,128,1,512)/int8 scale_plane=(3795,128,1,4)/fp16
[DS-2plane] remap-idx layer=mtp.0/1/2 rows=(5,1,256) kept=665/665 phys=[[36,37,0]] base=[0]
```
* 入图臂里 **`DS-2plane` 命中 144 行**、**`remap-idx` 命中 36 行** ⇒ 补丁**确实跑到了**（不是"没生效所以看起来对"）。
* `kept=665/665` ⇒ 可见槽位**全部**成功重映射（没有"全被置 -1 也能跑"的假象）。
* `phys=[[36,37,0]]` ⇒ 物理页 36/37，**第三列是 kv8_ori_plane 的 clamp 值 0**，且被 `delta < blocks_per_req`
  掩掉 —— 与设计一致。

---

## 4. ★★ 057 那个"submit/release 对称表"（补上 056 缺的那一列）

056 §4.4b 只打了 ❌ 臂。本轮**同一份探针、同一行格式**在两臂都跑：

| 格 | `DRAFT_INT8=1`（❌） | `DRAFT_INT8=0`（✅） |
|---|---|---|
| `submit#1` task 数 | 7（target，`_build_attention_metadata`） | **7（逐字同构）** |
| `submit#2` task 数 | 1（draft，`_propose`→`build_draft_attn_metadata`） | **1（逐字同构）** |
| `submit#2` 有 release 吗 | ⛔ **无** | ✅ **有（release#2）** |
| `submit#2` 的 group_id 在 target 里出现过吗 | 否 | **否（也一样）** |
| 后续形态 | `submit#3 in_flight=True` ⇒ RuntimeError | **稳定交替 7↔1，每步都配对释放**（打到 #10） |

⇒ **不是**"②a 多了一次提交"，也**不是**"builder 少交了任务"：两臂**逐字同构**。
差别只有一处：❌ 臂的 `_propose` **抛异常提前退出** ⇒ `llm_base_proposer.py:1177-1178` 的 `release()` 被跳过。
★ 056 §4.3 第 2 条那句【推断】("两个 release 都不在 finally 里 ⇒ forward 抛一次异常就跳过 release") **升为【实测】**。
★ 你提的那个 48-bit 数字（`281470296526752`）：**两臂都出现同形态的数字且互不相同** ⇒ 按 §5b 第 3 条，
**它是噪声不是信号**，这条轴到此定死。

---

## 5. 三个"构造缺陷"（全是包/探针/脚本的问题，**没有被报成机制失败**）

| # | 缺陷 | 症状 | 怎么和"机制失败"分开的 |
|---|---|---|---|
| 1 | `py_compile` 往包里写 `.pyc` | 出包门报"差异 3 ≠ 期望 1" | 差异清单里两条是 `__pycache__/*.pyc` ⇒ 编到临时文件 + 删陈旧字节码 |
| 2 | 读侧补丁用整行锚点改 `dsa_v1.py` 头部 import | 门报"锚点命中 0 次" | `dsa_v1.py` 头部是 `import torch / torch.distributed / torch_npu`，与 `dsa_attn_kv_plan.py` 不同 ⇒ 改成**就地取 env** |
| 3 | ★ **探针自己把图捕获打死了** | 入图臂起服死：`torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped (module: time)`，栈里有 `probe/usercustomize.py:42 print(time.time())` | 栈帧落在**我自己的探针行**上；且 v3 探针的 `dspark.model.forward` 钩子正好在 `_dummy_run` 被 AOT 全图编译的路径上 ⇒ 加 `DS_PROBE=0` **再跑同包同参数**即全绿 |

★ 第 3 条与 056 的 `NameError: os` 同源（**探针引入新失败**）。**规则**：入图/捕获类实验一律默认**关探针**跑一次
基线与实验臂；要用探针时只钩**请求路径**、不在被 `torch.compile`/AOT 覆盖的前向里 `print`/`time`。

---

## 6. 诚实边界与下一步

1. ⚠️ **【未确认】eager + INT8 的输出 sha 与其余三格不同**（`a6a760e9…`，chunks 2 vs 3）。
   * 交付形态是**入图**，而**入图 + INT8 与 BF16 逐字相同** ⇒ **不阻塞交付**；
   * 但"eager 格为什么不同"没查：可能是 eager 特有路径（写② `_mla_prolog_single_stream` 与 eager 的
     `_kv8_graph_rows_bound(False, None)` 组合），也可能是"块数不同导致 step 边界不同"的假象。
   * **最小证伪实验**：同包同几何跑 `GRAPH=0 DRAFT_INT8=0`（已有：`b8329c05`）vs `GRAPH=0 DRAFT_INT8=1`（已有：`a6a760e9`）
     ⇒ **两臂同 graph 模式、只差 INT8** ⇒ 这一格**已经是干净对照**。要再往下就要**逐步 token dump**（本轮未做）。
2. ⏳ **8 卡真权重未跑**：本轮全部在单 die、tiny-draft、`--load-format dummy`、TP=1、`max_model_len=8192` 上。
   ⇒ **容量/性能/接受率一律不可外推到 8 卡**（056 §8 的同一条边界依然成立）。
3. ⏳ 本轮**没有**测：②a 的**吞吐/接受率退化**（spec 读数只有 2 请求的极小样本：`draft=40 / accepted=20`）；
   `DS_2PLANE_READ=remap` 的**性能代价**（每次 decode 多一次窗口反量化 + 索引重映射）。
4. ⏳ 写侧 `npu_dynamic_quant` 的**数值口径**没有与目标侧做逐元素对账（只对账了**形状与分组**）。

### 6.1 交付物

| 文件 | 作用 |
|---|---|
| `agents/DS_draft_graph_int8/scripts/mk_pkg_dsi.py` | ★ 两处补丁的生成器（锚点 + AST + py_compile + 差异门 + md5 门），`--store/--read` 可单独开关 |
| `agents/DS_draft_graph_int8/probe/usercustomize.py` | ★ 探针 v3（submit/release 配对 + 调用栈 + 首个异常 + `kv_plan.scatter` 的**两平面判据** + `remap-idx` 自报）；`kept=` 是"语义是否真对"的自报量 |
| `agents/DS_draft_graph_int8/scripts/ds_arm_fix.sh` / `ds_fix.sh` | 修复臂（复用 `D_draftINT8/d_arm.sh`，只换包/探针/输出目录）；`DS_PROBE` / `DS_2PLANE_READ` 都**在日志里自报** |
| `agents/DS_draft_graph_int8/scripts/ds_diag.sh` | 诊断臂（复现 Q3） |
| `agents/D_draftINT8/pkg-ddi` → `agents/DS_draft_graph_int8/pkg-dsi` | 影子包（base = ②a 两处改动；`pkg-dsi` = 再加"两平面存储 + 索引重映射"） |

复跑（A3 上，c1 或 c2）：
```bash
bash ~/projects/dsv41-upstream-pr/tools/a3_chip.sh c2 --timeout 1800 --name ds-graph -- \
  env TAG=ds-graph TIER=D GRAPH=1 DRAFT_INT8=1 DS_BUILD_PKG=1 DS_FIX_STORE=1 DS_FIX_READ=1 \
      DS_2PLANE_READ=remap DS_PROBE=0 PROMPTS=1 PROMPT_TOKENS=256 MAX_TOKENS=4 \
      PASS1_ROUNDS=1 PASS2_ROUNDS=1 \
  bash /work/agents/DS_draft_graph_int8/scripts/ds_arm_fix.sh
```

---

## 7. 这条结论对 ②c（draft block 128→64）意味着什么

* ②a 与 ②c **正交**（一个改 dtype、一个改 block）。本轮**没有**测组合（`64×(512+4×2)=33,280` 页）。
* ★ 但有一件事变了：056 说"②a 在真实请求下不可用" ⇒ **这条已经不成立**。
  ⇒ 若 ②c 上线后还要再挤容量，**②c+②a 现在是一个真实候选**（同一条运行期缺口已被本轮的
  "两平面存储 + 索引重映射"补上），代价是**多一套索引重映射的代码面**。
* 选型建议不变：**先 ②c**（dtype 不动 ⇒ 零精度风险），把本轮的 ②a 当"容量再挤一档"的储备。
