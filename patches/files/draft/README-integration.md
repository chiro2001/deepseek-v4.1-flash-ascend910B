# Phase 1：DSpark draft 图捕获**离线集成报告**

日期：2026-09-16　执行：线 3（单算子线）　状态：**集成成功，8 项自检全过 + 等价性验证 + AST 预检**

产物位置（A3-node1）：
```
~/projects/dsv41/probe_draft/
├── dsa_v1.py                     2184 行  ← probe_dsa 版 + 0004 三个 hunk（**含 F3**）
├── dspark_proposer.py             633 行  ← probe_draftmeta 版，剥掉探针
├── llm_base_proposer.py          2770 行  ← probe_draftmeta 版，剥掉探针
├── make_draft_files.py                    ← 幂等集成脚本（可重复运行）
├── verify_equivalence.py                  ← 与参考版的等价性验证
├── precheck_ast.py                        ← 步骤 4 的 AST/逻辑预检
├── 0004-dsa-startpos-resident-buffer.patch
└── SELFCHECK.txt                          ← 88 行完整自检输出
```

---

## 0. 结论速览

| 项 | 结果 |
|---|---|
| 集成是否成功 | **是**（9 项自检 + 等价性 + 幂等，全过） |
| 与你的兼容性结论是否一致 | **锚点结论一致**；但有 **3 处需要更正/补充**（§3） |
| 是否碰到 F3 | **没有丢**：`_o_proj_2d_enabled=2`、`_DUMMY_WO_A_FIX=2` |
| 探针残留 | **0**（`draftmeta_probe` + `_dm(` 合计 0） |
| 离线预测：draft 会入图吗 | **会**（三道门全部满足，§4 有日志级证据） |

## 1. 八项自检逐条结果（`SELFCHECK.txt`）

| # | 自检 | 你的判据 | **实测** | 结果 |
|---|---|---|---|---|
| 1 | 三个文件 `py_compile` | 通过 | 3/3 通过 | ✅ |
| 2 | 合并后的 `dsa_v1.py` 仍含 F3 | `_o_proj_2d_enabled`>0 且 `_DUMMY_WO_A_FIX`>0 | **2 / 2** | ✅ |
| 3 | 0004 生效 | `STARTPOS-DRAFT-FIX` == 1 | **2**（见 §3.1）| ✅（判据需更正） |
| 4 | 0006 生效 | `TARGETPOS-FIX` > 0 | **6** | ✅ |
| 5 | 0005 生效 | `DSPARK_DRAFT_METADATA_MODE` > 0 | **11**（base 7 + proposer 4）| ✅ |
| 6 | 硬禁已解除 | `use_cuda_graph` 为 computed | `bool(...)`，**无** `= False` | ✅ |
| 7 | 无探针残留 | `draftmeta_probe\|_dm(` == 0 | **0** | ✅ |
| 8 | 锚点唯一性 / 幂等 | 可重复运行且结果一致 | 真实二次落盘后 `md5sum -c` **全 OK** | ✅ |
| 9 | （我自己加的）0002 生效 | — | `DSPARK_GRAPH_CAPTURE_METADATA` = 2 | ✅ |

## 2. 集成怎么做（关键：**取源选择**）

| 输出 | 取源 | 处理 |
|---|---|---|
| `dsa_v1.py` | **`probe_dsa/dsa_v1.py`**（含 F3）| 用 `0004-*.patch` 的**原始 hunk** 做精确锚点替换 |
| `dspark_proposer.py` | **`probe_draftmeta/dspark_proposer.py`** | 剥掉 `[DRAFTMETA-P0]` 定义块 + 4 处 `_dm(...)` |
| `llm_base_proposer.py` | **`probe_draftmeta/llm_base_proposer.py`** | 剥掉定义块 + 3 处 `_dm(...)` |

* **0004 的 hunk 直接从 `.patch` 文件解析**（`parse_patch`），不做手工转写 ⇒ 消除转写风险；
  每个旧块要求**在源文件里恰好出现 1 次**，否则拒绝生成（锚点唯一性自检）。
* **脚本从不修改源文件**，只从源重新生成输出 ⇒ 幂等（真实二次落盘用 `md5sum -c` 验证过）。

## 3. ⚠️ 三处与你的结论不一致（请采纳）

### 3.1 自检 #3 的期望值应为 **2**，不是 1

`STARTPOS-DRAFT-FIX` 在 `0004` 的**两个 hunk 里各出现一次注释**（hunk1 在常驻 buffer 声明处、
hunk2 在就地刷新处）⇒ 合并后是 **2 次**。参考实现（`probe_draftmeta/dsa_v1.py`，0004 已应用）
实测也是 **2 次**。判据写 1 会**误报失败**。

（附：`start_pos_draft` 出现 **3** 次 = 声明 1 + 赋值 1 + 使用 1，这也是我 `make_draft_files.py`
里 #3 的实际判据。）

### 3.2 **不能用 `fix2_files/` 做 proposer 的取源** —— 它会静默丢掉 D3/D4 修复

你的指示里写「`spec_decode/dspark_proposer.py` ← fix2 版、`llm_base_proposer.py` ← fix2 版」。
实测 `patches/draft_graph/fix2_files/` 里的两个文件**只含 0002 的一部分**：

| 标记 | `fix2_files` 版 | `probe_draftmeta` 版（我用） |
|---|---|---|
| `DSPARK_GRAPH_CAPTURE_METADATA`（0002） | 有 | 有 |
| `DSPARK_DRAFT_METADATA_MODE`（0005） | **0** | **7** |
| `TARGETPOS-FIX`（0006） | **0** | **6** |
| `draftmeta_probe` / `_dm(`（探针） | 0 / 0 | 2 / 9 |

**⇒ 若按 fix2 交付，就等于丢掉了 D3（executor 契约）与 D4（target_positions 常驻 buffer）**，
而这两条正是历史崩溃（「A 崩到 1.0」）的修复。**我用的是 `probe_draftmeta/` 版 + 剥探针。**

**佐证**：`probe_draftmeta/` 的两个文件在剥掉探针后，与 `fix2_files/` 版**不是同一份**
（前者多出 0005/0006 的功能代码）；而我的交付版在剥探针后与 `probe_draftmeta/` 版
**逐字节相同**（见 §5 等价性验证）。

### 3.3 `use_cuda_graph` 的触发有 **三道门（阈值不同）**，不只是「解除硬禁」

```python
# wrapper 分支（llm_base_proposer.py:663）
if self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs() and self.use_cuda_graph:

# draft.use_cuda_graph（dspark_proposer.py:131-135）
self.use_cuda_graph = bool(callable(_runner_use_aclgraph) and _runner_use_aclgraph()
                           and not vllm_config.speculative_config.enforce_eager)

# _use_aclgraph()（model_runner_v1.py:730）
return (self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        and (self.compilation_config.mode == CompilationMode.VLLM_COMPILE or breakable)
        and not self.model_config.enforce_eager)          # ← **target** 的 enforce_eager
```

| 门 | 阈值 | 说明 |
|---|---|---|
| 门1 | `has_full_cudagraphs()` = **max mode == FULL** | 比 `!= NONE` 严；`PIECEWISE` **不满足** |
| 门2 | `_use_aclgraph()` | 还夹着 **target** 的 `model_config.enforce_eager` |
| 门3 | `not spec.enforce_eager` | 这才是本次解除的那条 |

**⇒ 运行期若 `cudagraph_mode == PIECEWISE`，补丁生效也不会触发 wrapper**（静默 no-op）。
Phase 3 必须确认 `cudagraph_mode`，别只看 `Wrapping draft model` 是否出现（否则会把
「门1 不满足」误判成「补丁没生效」）。

## 4. 步骤 4 预检：**三道门已全部离线确认满足**

从 A3-node1 的 serve 日志（`logs/perf/faA_serve.log` 的 `non-default args` 行）读到**真实运行期配置**：

| 门 | 运行期实测值 | 判定 |
|---|---|---|
| 门1 | `cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY (2, 0)` → `max_cudagraph_mode() = FULL` | **满足** ✅ |
| 门2a | `compilation_config.mode = None` → 由 `vllm/config/vllm.py:1272` 解析为 `VLLM_COMPILE`（`optimization_level` 默认 > O0） | **满足** ✅ |
| 门2b | `model_config.enforce_eager = False` | **满足** ✅ |
| 门3 | `speculative_config.enforce_eager = False`（`method=dspark`, `num_speculative_tokens=5`） | **满足** ✅ |

**⇒ 离线预测：打上补丁 + `DRAFT_GRAPH=1` 时，`ACLGraphWrapper` 应当被触发。**

AST 预检（`precheck_ast.py`）另外确认：
* `dspark_proposer.py` 里 `self.use_cuda_graph` 的赋值是 **`bool(callable(...) and ... and ...)`**，
  **没有**任何残留的 `= False`；
* 含 `ACLGraphWrapper(` 的 `If` 分支**只有 1 处**（`llm_base_proposer.py:663`），条件如上。

## 5. 等价性验证（`verify_equivalence.py`）

把参考版（`probe_draftmeta/`）的探针也剥掉，再与我的交付版逐行对比：

| 文件 | 结果 |
|---|---|
| `dspark_proposer.py` | **剥探针后与参考版逐字节相同** ✅ |
| `llm_base_proposer.py` | **剥探针后与参考版逐字节相同** ✅ |
| `dsa_v1.py` | 与参考版差异 **4 个块 / 新增 62 行 / 删除 2 行，全部属于 F3**（0 个非 F3 块）✅ |

`dsa_v1.py` 的 4 个块 = ① `import os`（F3 helper 的导入，文件顶部）
② 11 行 F3 helper（`_O_PROJ_2D` / `_o_proj_2d_enabled`）③ 49 行 F3 分支（`elif _o_proj_2d_enabled()...`）
④ 1 行 delete（空行）。**⇒ 合并后既保住了 F3，又拿到了 0004。**

> 该验证脚本我修了两次自己的判据 bug（记录备查）：
> ① 误把参考版里 `def _dm(` 也算成一次调用（真正的调用只有 1 处）；
> ② F3 的代码块跨 49 行，**逐行**判「是否含 F3 字样」会误报中间那些不含字样的行为「非 F3」
> —— 必须用 `SequenceMatcher` 的**块级**判据 + 相邻块合并。这两条对所有「大块插入」的
> 等价性验证都适用。

## 6. Phase 2 接线清单（给你，我未改启动器）

```bash
DRAFT_GRAPH=${DRAFT_GRAPH:-0}     # 0=stock(eager draft) 1=启用 draft 图
# 1) ~/projects/dsv41/probe_dsa/dsa_v1.py        （已有挂载，**现在含 F3 + 0004**）
# 2) ~/projects/dsv41/probe_draft/dspark_proposer.py     -> spec_decode/dspark_proposer.py
# 3) ~/projects/dsv41/probe_draft/llm_base_proposer.py   -> spec_decode/llm_base_proposer.py
# 并确保 SPEC_EAGER_OPT=0（`enforce_eager=false`）—— 门3 的前置条件
```

> ⚠️ **注意**：`probe_dsa/dsa_v1.py`（你已有的挂载点）**仍是未合并 0004 的版本**。
> 要么把挂载指向我生成的 `probe_draft/dsa_v1.py`，要么把 `probe_draft/dsa_v1.py` 覆盖回
> `probe_dsa/dsa_v1.py`（**这一步需要你做**，因为 `probe_dsa/` 是你的领地、我不动）。

**幂等重建**：任何时候都可 `python3 ~/projects/dsv41/probe_draft/make_draft_files.py`
重新生成三个文件（源不变则输出字节不变）。

## 7. 复现命令

```bash
cd ~/projects/dsv41
python3 probe_draft/make_draft_files.py          # 生成 + 9 项自检（幂等）
python3 probe_draft/verify_equivalence.py        # 与参考版的等价性
python3 probe_draft/precheck_ast.py --probe-draft probe_draft   # AST/门真值表/运行期证据
```

## 8. 边界（我**没有**做的事）

1. **没有改启动器**（`scripts/serve_a21.sh`）——Phase 2 是你的。
2. **没有动 `probe_dsa/`**（你的领地）——§6 那条覆盖需要你执行。
3. **没有起服/上卡**，全程纯文件操作 + 只读日志。
4. **没有验证运行期行为**：§4 是**离线预测**（基于真实日志里的配置值），
   Phase 3 的三条指纹才是最终判据。
