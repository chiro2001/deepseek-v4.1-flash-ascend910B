# A2 主线交接（★ 2026-09-22 15:0x 全面重写）

> **接手先读这四份**：本文件 → [`README.md`](README.md) → [`AGENTS.md`](AGENTS.md) → **[`docs/A2-DEPLOY-NOW.md`](docs/A2-DEPLOY-NOW.md)**（上线清单，含全部参数与判据）。
> 日志索引：**[`logs/README.md`](logs/README.md)**（001–063）。旧工作区 `../upstream-v41/` **已冻结、只读**。
> ★★ **本机时钟比 A3 快约 7 分钟**（两边都是 CST +0800，是漂移）—— 判断远端进度**只能**读远端 `date` + 日志字节增长，别用本机时间做减法。
> 标记约定：**【实测】** = 有原始数据；**【推断】** = 代码/算式推出但没直接测；**【未确认】** = 没跑到。**不许用相邻数字顶替缺的那格。**

---

## 0. 一句话现状

> ★★★ **2026-09-22 16:00 最新（本轮最大的一条）**：**档 D + ②c 读到 `777,318` tokens**
> —— 与 `050`/`051` 的预测**逐字命中**，= **×1.8177 vs 档 B（427,643）**。
> `R8_SLOT_TRACE` 同时给出机制实测：`slot0–2 capacity = max(33280+8320, 66560, 65536) = 66560`（= `aliases_max` binding）、
> `slot3 = max(66560+16640, 66560, 0) = 83200` —— **四个数字全部自洽**。
> ⏳ **压测仍在跑**（`r8-t-dc2-d-D2` 正在编译 static kernel）⇒ 四条功能判据 + 投机四数**待回填**。
> ⇒ ★ **至此"能否碰 ×1.84"有了答案：能，而且超了（1.8177×）。**

> ## ★★★★ 2026-09-22 16:1x **两件大事**
>
> **① ★★★★★ A2 的阻塞彻底解除**（`logs/065`，**用户在 A2 实机跑的，全档通过**）：
> ```
> 注册（匿名/普通内存）：1=True 8=True 32=True 64=True     ← ★ 四档全过，每档都做了真实 H2D→D2H 逐字节对账
> host_mem_pool = 0（实测确认 ⇒ 必须走 registered，不能靠 aclrtMallocHost 撑池）
> H2D：pageable 5.3 → 注册后 16–21 GB/s（3–4×）
> 单次 pinned 32 GiB ✓（logs/012 那条"(4,8] GiB 上限"作废）
> 64 GiB 注册耗时 22.5 s ⇒【推断】392 GiB 池首注册 ≈137 s 一次性成本
> ```
> ⇒ **池需 ~49 GiB/worker 而 64 GiB 单块可注册 ⇒ 池可整体注册，不必分片**。
> ⇒ ★ **下一步就是 `docs/A2-DEPLOY-NOW.md` 的「第二步：造 shadow-pkg」+ 起服，探测这条路已经走完**。
> ⚠️ 起服时盯一条：**8/8 worker 的 `[P1_pinned] ... ret=0`**（探针是单进程单块，生产是 8 worker 各 ~49 GiB）。
>
> **② ★★ 候选路线翻转：②a 取代 ②c 成为首选**（`T` 的判决 + **主代理独立验算**，**六格全部落定**）：
> ```
> 档   draft 形态          Σslot_pages  p_draft  BPR    容量            vs 该档现状
> C    现状 BF16 b128         540,928    130   2,471    427,643【实测】   ×1.0000
> C    ②c  BF16 b64           369,280    259   2,600   ★595,404【实测】   ×1.3923
> C    ★ ②a INT8 b128         369,280    130   2,471   ★626,488【推断】   ×1.4650
> D    现状 BF16 b128         476,416    130   2,471    485,610【实测】   ×1.0000
> D    ②c  BF16 b64           282,880    259   2,600   ★777,318【实测】   ×1.6007
> D    ★ ②a INT8 b128         282,880    130   2,471   ★817,898【推断】   ×1.6843
> ```
> ★ **②a 在 C/D 两档都比 ②c 高 5.2%**，因为 **Σ 相同 ≠ 容量相同** —— BPR 里还有 `p_draft` 那一项
> （②a 的 draft 块仍是 128 ⇒ `p_draft=130`；②c 是 64 ⇒ `259`）。
> ★ **②c 两个档都有页缝隙**（C：73,856 vs 65,536 = 8,320 B；D：66,560 vs 65,536 = 1,024 B）
> ⇒ 投机解码会塌（D 已实测 **42.4% → 0.1%**）；**②a 页缝隙为 0** ⇒ 这是它取代 ②c 的真正理由。
> ⚠️ **`c-C3` 那条「档 C + ②c = −5%」已作废**：根因是**第三次"静默降级"** ——
> `swa_plane_kwargs()` 的唯一调用点在 `models/deepseek_v41/model.py:483`，而当时 runner **只在档 D 才挂它**
> ⇒ 档 C 臂的 SWA 保持 BF16 ⇒ 页几何与档 B 完全相同。
> 补挂后 `t-dc2-c-C5` 读到 `aliases_max=66,560 → capacity=73,856 → Σ=369,280 → 容量 595,404`，
> 与算术预测**逐字相同**（主代理已独立核实这四条 trace + 那个唯一调用点）。
> * ②c 被实测否决：**接受率 42.4% → 0.1%**（`draft 页 65,536 < slot 页步长 66,560` ⇒ 视图非连续，
>   与 `034` 的 state ring 同款坑，修点 = draft 读路径的 `CACHE_PAGE=stride(0)`）；
> * ★ **主代理纠正了 `T` 的一处算术**：`T` 说"②a 与 ②c 同值 777,318"——**只看 Σ 是不够的**，
>   BPR 里还有 `p_draft` 项：②a 的 draft 块仍是 128 ⇒ `p_draft=130` ⇒ **BPR 保持 2471**；
>   ②c 是 64 ⇒ `p_draft=259` ⇒ BPR 涨到 2600。⇒ **②a 是 817,898（比 ②c 多 5.2%）**，且**无页缝隙**。
> * ★ 口径：817,898 是【推断】（零参数模型；该模型在 8 卡上已被 **4 个实测点**支持：B/C/D 的 draft128 三点 + ②c 的 777,318）。

| 线 | 状态 |
|---|---|
| **线 1 · DRAM KV 卸载** | ✅ **可上线** —— 8 卡真权重，档 B/C/D **三判据全绿** |
| **线 2 · KV8 / int8** | ⚠️ **能力已通、容量目标部分达成**：档 C **×1.0000**、档 D **×1.1356**（目标 **×1.84**） |
| **唯一阻塞上线** | ⛔ **A2 本机的池后端探测** —— 只有用户能在 A2 上跑（§6 第 1 条） |

**线 2 的关键背景**（一句话）：容量上不去**不是 int8 的错**，是 **draft 组（BF16 / block=128）顶死 slots 0–2**；
**唯一还活着的解法是 ②c（draft block 128→64，保 BF16）**，预测把 HBM 推到 **×1.8177**，其 8 卡端到端**正在 c0 跑**（§6 第 2 条）。

---

## 1. 现场状态（远端时间 2026-09-22 15:1x）

| 槽位 | 现在跑什么 | 备注 |
|---|---|---|
| **c0** | ★ `t-dc2-c-C3`（**真·②c**，`--variant b64` 无条件写死，15:28 起） | **8 卡臂走 c0 锁 + Phy-ID 8–15** |
| **c1** | `DS_draft_graph_int8`（②a 病行探针） | |
| **c2** | 空闲（die 7） | 主代理已归还（曾用于 `ds-remap2`） |

### ★★★ 本轮最大的一条已结案：**"档 D 降低接受率"被实测否掉**（`t-dc2-b-D2`，`GRAPH_SAFE=1`）
```
档 D ：稳态 interval MeanAccLen 2.69 / AvgDraftAcc 33.8% / Per-pos .591 .355 .290 .237 .215
档 C基线：                         2.46 /              29.2% /          .509 .311 .264 .208 .170
⇒ ★ 档 D 每位都更高（容量 485,610 第二次独立复现、fill 8/8、sha fill==replay）
```
★ 此前 `max_tokens=1` 下的 `1.00/0%`（D）与 `1.50/10%`（C）是**口径假象 + 样本量差异**（`Drafted` 10 vs 15），**不是缺陷**。
★ **口径纪律（新）**：两臂 cumulative 分母不同（`num_drafts` 166 vs 214）⇒ **只能比 interval 行**，且**必须剔除 interval #1**（只含 warmup 的 3 个 draft 步，会给出假的 `1.00/0%`）。

### ★★★ ②c 的实测结论（**2026-09-22 15:3x**，`t-dc2-c-C3`）

### ⛔⛔ **2026-09-22 17:1x 重大更正：过去所有标着 "graph" 的臂，draft 都没真的进图**

> **起因**：`c1_offload_x_draftgraph_1die` 在单卡上造了含 **draft 版三文件**的包，
> 结果打出 `Wrapping draft model with ACLGraphWrapper: runtime_mode=FULL` ——
> 而这条 INFO 在 `reports/two-corrections-draftgraph-gatehoist.md` 里记的是「全日志 **0 次**」。
> **主代理独立复核，全部为真：**
>
> ```
> ① 所有 tiny 影子包里的 dspark_proposer.py md5 全是 dac256ad… = stock：
>      C2_draft64/pkg/dfix2c / DS_draft_graph_int8/pkg-dsi / X_integrate/pkg / T_draftceiling/pkg/B
>    stock 第 75 行 =  `self.use_cuda_graph = False`（硬禁，无条件覆盖基类）
> ② T_draftceiling/scripts/run_2c_arm.sh:117  **硬编码 DRAFT_GRAPH=0**
> ③ D_draftINT8/scripts/d_arm.sh:61           写死 SPEC enforce_eager:true
>      ⇒ 它的 GRAPH=1 只是【主模型入图 + draft 永远 eager】
> ```
> ⇒ ★★ **过去所有标着 "graph" 的臂 —— `DS/ds-graph2`、`C2/c2c-*-graph`、`T` 的全部臂 ——
> 都是"主模型入图 + draft eager"，没有一条真的把 draft 放进图里。**
>
> **连带更正三条口径**：
> 1. `logs/064`（②a 单卡跑通）的 `ds-graph2` —— **它的 "入图" 指主模型入图，不是 draft 入图**；
> 2. 本文件前面写的"8 卡全部 18 条臂 `DRAFT_GRAPH=0`"**低估了问题**：
>    真正的情况是**连 `DRAFT_GRAPH=1` 的臂也没让 draft 进图**；
> 3. `063` 那句"单卡能替代图模式兼容"—— 对**主模型**图成立，对 **draft** 图**此前从未被测过**。
>
> ★ **要真让 draft 入图，必须同时满足三件**（缺一即静默退回 stock）：
> ```
> a) 装 draft 版三文件  spec_decode/dspark_proposer.py（解除 line 75 硬禁）
>                       spec_decode/llm_base_proposer.py
>                       attention/dsa_v1.py
> b) DSPARK_GRAPH_CAPTURE_METADATA=1
> c) DSPARK_CAPTURE_VALUE_FIX=1
> ```
> ★ **起服后必须证伪"draft 没进图"**（三条判据）：
> ```
> grep -c "Wrapping draft model with ACLGraphWrapper" <serve.log>   # 期望 >= 8（8 rank）
> grep -oE "runtime_mode=(FULL|NONE)" <serve.log> | sort | uniq -c  # dg1 应见 FULL
> grep -oE "DSPARK_GRAPH_CAPTURE_METADATA=[01]|DSPARK_CAPTURE_VALUE_FIX=[01]" <serve.log> | sort -u
> ```
> **第 1 条为 0 ⇒ 这一格【未验证】，不是"通过"。**
>
> ★ **唯一在 A3 上真跑过 draft 图的地方**：`agents/C1_offload_draftgraph/`（本日 17:1x 建的包，
> 含 draft 三文件 + 内容级断言）。`C2_int8_draftgraph` 也在同日用它跑通了档 C × draft 图。

#### ⚠️⚠️ **同一条更正的回撤（17:2x）—— 我把它推得太宽了，P1 纠正得对**

> 我上一条把"所有 graph 臂都是假 draft 入图"推到了**所有包**。`a3_p1_offload_draftgraph` 实测反驳：
> **不同的包，draft 文件集不同**，必须分开看。
>
> | 包 | `dspark_proposer.py` md5 | 判据行 | draft 能进图吗 |
> |---|---|---|---|
> | **`shadow-pkg`**（8 卡链 / A2 生产链用的那个） | **`5565afed…`** | 第 188 行 = **`self.use_cuda_graph = bool(...)`（计算式）** | ★★ **能** |
> | `X_integrate/pkg` | `dac256ad…` = stock | 第 75 行 = `= False`（硬禁） | ⛔ 不能 |
> | `T_draftceiling/pkg/B`、`C2/pkg/dfix2c`、`DS/pkg-dsi` | `dac256ad…` = stock（或路径不存在） | 同上 | ⛔ 不能 |
>
> **而且 8 卡链有硬门，不会静默通过**（`serve_a2.sh:1346-1357`，主代理已逐行核实）：
> ```
> DRAFT-GUARD：DRAFT_GRAPH=1 时
>   grep -c "DSPARK_GRAPH_CAPTURE_METADATA" dspark_proposer.py  < 1  => die
>   容器内 DSPARK_GRAPH_CAPTURE_METADATA != 1                     => die
> ```
> ⇒ **"stock dspark + `DRAFT_GRAPH=1`" 在 8 卡链上根本起不了服** —— 与 tiny 包（无此门）不同。
> 所以 **8 卡上的 `DRAFT_GRAPH=1` 是真的 draft 入图**。
>
> ★ **还核实了一条对 A2 上线的影响**：`make_shadow_pkg.sh` 从 `$PKG/patches/files/*` 建**软链**，
> 而发布仓的 `patches/files/draft/{dsa_v1,dspark_proposer,llm_base_proposer}.py`
> **在 git 里且已 tracked**（md5 `371bb023…` / `5565afed…` / `a24076eb…`，与 A3 shadow-pkg **逐字相同**）。
> ⇒ **用户从 GitHub clone 造的 shadow 是完整的，能装 draft 三文件。**（我一度以为这里有缺口，实测后撤回。）
>
> ⇒ **正确的教训**：**"draft 入图"的判据必须以那三条为准**（`Wrapping` 行数 / `runtime_mode` / `DRAFT-GUARD` 两行），
> **且要分清是哪个包** —— 而不是笼统地说"过去的臂都假"。
>
> | 场景 | 结论 |
> |---|---|
> | tiny 包（`X` / `C2` / `DS` / `T/pkg`）标 "graph" | ⛔ **主模型入图 + draft eager**（P1 的原始发现，成立） |
> | **8 卡链 `shadow-pkg`** | ✅ **draft 真入图**（有 DRAFT-GUARD 硬门） |
```
②c 真的生效了（早停闸命中）：group 清单 (12, 'DeepseekV41DraftSWASpec', 64, ...) ✅
★ 但容量反而降：427,643 → 406,425（−5%），而 050/051 的预测是 595,404（+39%）⇒ ★ 该预测作废
★ 与 054 的 tiny 实测方向一致：tiny 档 B  20,826 → 19,247（−7.6%）【实测，不是预测】
```
⇒ ★★ **②c 的收益与档位强相关；至少档 B / 档 C 上是负收益**。`050`/`051` 那条"档 C 595,404"**作废**。
⇒ ★ **档 D + ②c 是唯一还没试过的组合**（预测 777,318 = ×1.8177 vs 档 B），它才是"能否碰 ×1.84"的判据臂。
⚠️ **机制仍未定论**（我推过一版"档 C 的 binding 是 `kv+index`"，但它与 tiny 档 C ×1.4655 的实测**矛盾**，已撤回）：
两个候选 —— (a) binding 是某个 ②c 不缩小的平面 / (b) BPR 涨幅 > Σ 降幅。
★ **判据**：`R8_SLOT_TRACE=1`（core:261 自带，三个臂都没设 ⇒ 这就是 `grep R8-SLOTS` 0 命中的原因）
每槽打 `slot / kv / index / aliases_max / draft / capacity / [draft-aware]` ⇒ **一次就能定**。

| 子代理 | 状态 |
|---|---|
| **`T_draftceiling`** | ⏳ **running**（③c2 重跑中） |
| **`DS_draft_graph_int8`** | ⏳ **running**（②a，c1）—— 已复现 Q3 失败（`failed=1`、容量 39,846），探针已装 |
| `R_8card_int8` | ✅ **completed 收官**（048，8 条臂全绿） |
| `S_graphfix` / `C1_graph1die` / `C2_draft64` / `D_draftINT8` / `G_kv8fix` / `H_kvcheck` / `J_mgrhardening` / `KV8_p0` / `P2_poolsizing` | ✅ completed |

**`T_draftceiling` 的三臂（`chain_2c_v5`）当前的账**：

```
t-dc2-a-C        档 C 基线                    ✅ 已过（★ 它的 runner 少挂 model.py ⇒ 实际就是档 C）
t-dc2-c-C        档 C + ②c(draft64)           ⛔ rc=9 起服失败 —— ★ 已定性，见下
t-dc2-b-D-legacy 档 D（GRAPH_SAFE=0）         ⛔ 起服成功但首个请求崩 507057 —— ★ 已定性，见下（已按建议改名存档）
t-dc2-c-C2       档 C + ②c（GRAPH_SAFE=1）    ⏳ **重跑中**（15:12 起）—— ★ 这条才回答"②c 能否碰到 ×1.84"
```

### ★★★ 两条已定性的失败（**主代理独立核实，2026-09-22 15:0x–15:1x**）

**(1) `t-dc2-b-D`（档 D / `GRAPH_SAFE=0`）—— 不是新缺陷，就是 `049` 判死的那一格**
```
起服 ✅ / 容量 485,610 ✅ / warmup ✅ / fill 轮 ⛔ 8/8 全失败（rc=0 是假象）
栈：llm_base_proposer.py:1043 _propose → SUSPECT REMOTE ERROR → 507057 → EngineDeadError
```
⇒ `049` §5.5.3 的反例臂 `sg-c-d-cmplegacy`（**预期失败**，用来证明补丁必要）**逐字同款**；
⇒ ★ 同时段 `R` 的 `r8-f1-tierD-graph`（**`GRAPH_SAFE=1`**）**rc=0 全绿** ⇒ **唯一变量就是那个开关**（第三次独立复现）。
⇒ 根因是 T 的 runner **从不设 `GRAPH_SAFE`**（`grep` 零命中），用的是从 `R_8card_int8` 抄的模板默认值 **0**。
⚠️ **别搞混**：`logs/t-dc2-b-D.client.log`（13:43）是更早一臂的残留，不是 15:01 那轮的读数。

**(2) ★ ②c 从来没生效过（2026-09-22 15:2x 主代理独立查出，T 已确认并修好）**
```
t-dc2-c-C2 的卸载层 group 清单：(12, 'DeepseekV41DraftSWASpec', 128, 3, ...)   ← ★ 还是 128
容量 427,643 == 基线 t-dc2-a-C；CPU→GPU / GPU_to_CPU 两臂逐字相同（2.348023808e+09）
```
根因是**两条门控都空**：① `VLLM_V41_DRAFT_BLOCK` 没进 `inner.sh`（白名单不转发）；
② ★ **flag 文件路径假设错了** —— `run_2c_arm.sh` 把 flag 建在**宿主** `$T/`，而 core 读的是**容器内**
`/work/agents/T_draftceiling/draft_block_64.flag`，而 `r8-*` 容器（由 `serve_a2.sh` 起）**没有 `/work` 挂载** ⇒ 静默回落 128。
⇒ **T 的修法（绕开所有挂载/env 假设）**：新增 `--variant b64`，把 `block_size=64` **无条件写进被挂载的 core 文件本身**
（`e2e-b64/deepseek_v41_core.py`，md5 `2b1dc8b4…`）⇒ "这一臂是不是 ②c"与**文件**绑定，不可能被挂载假设吃掉；
并按建议加了**早停闸**（起服后解析 group 清单，draft 块不是 64 就立刻 `exit 9`，不浪费 25 min）。
★ **重跑臂 `t-dc2-c-C3` 正在 c0**（15:28 起）—— **这条才真正回答"②c 能否碰到 ×1.84"**。

**(2b) `t-dc2-c-C` 的 rc=9 —— 死在 `rejection_sampler_triton_warmup` 内部（★ 主代理读源码补的一格）**
```
Worker-6 died unexpectedly (exit code: None)   ← 信号带走，无 Python traceback；其他 7 个是被 EngineCore 连坐
崩点 = kernel_warmup.py:44 "Starting Triton kernel warmup." 之后、第一个 "complete" 之前
```
生产 `kernel_warmup()` 的顺序是 `rejection_sampler → penalties → rms → deepseek_v41_indexer`，
而 **`_run_warmup` 的日志是跑完之后才打的** ⇒ **死在四种 warmup 里的第一个**；
同臂的 `/dev/shm`（1007G 可用）与宿主内存（1605 GiB available）**都已排除**。
★ **待分开的两种解释**：(a) draft block=64 这个**新形状**触发 / (b) **②c 补丁本身**触发
⇒ 判据臂 = **"补丁在、形状不变（block 仍 128）"**（已建议给 `T`）。
★ **2026-09-22 15:2x 更新**：这一格的优先级**降到 ②c 之后** —— 若 `c-C3`（同样是 draft=64）能跑通，
则"draft=64 会崩 warmup"这条**直接被推翻**。

---

## 2. 硬性红线（**违反即回滚**）

| # | 规则 |
|---|---|
| **1** | ★★ **绝不发 PR / issue / 评论**（用户未授权）；只写草稿与**自己的** repo |
| **2** | **绝不 push 到 `vllm-project/vllm-ascend`**；只能推 `chiro2001/*` |
| **3** | **绝不写 `upstream-v41/`**（已冻结只读）；新产物一律落 **`a2/`** |
| **4** | **绝不碰** `dsv41-a3`（保持 `Exited`）/ `mooncake-master` / 别人的容器 / **Phy-ID 0–7** |
| **5** | 占卡走锁：`bash ~/projects/dsv41-upstream-pr/tools/a3_chip.sh <c0\|c1\|c2> --name X -- <cmd>`；**退出码 75 = 没抢到锁，不是失败** |
| **6** | **绝不手设 `ASCEND_RT_VISIBLE_DEVICES`**（由槽位脚本注入） |
| **7** | **绝不用 `/tmp`**：`source ~/projects/dsv41/a2/scripts/tmpdir.sh <任务名>` |
| **8** | ★★ **起服前先 `df -h /dev/shm`**（满 ⇒ `OSError: [Errno 28]` 在 `SemLock`，**极易误判成"补丁坏了"**） |
| **9** | 写算子 / kernel 前**必须先查 cannbot**（`AGENTS.md` §6） |
| **10** | 结论必须标 **【实测】/【推断】/【未确认】**；**不许用相邻数字顶替缺的那格** |
| **11** | 传文件走 `tools/cos-xfer.sh`（**不要 scp**） |
| **12** | ★★ **交付配置必须保留投机解码**；**⑤a（关投机换容量）已由用户否决** |
| **13** | ★★ **看进度要用远端 `date` + 日志字节增长**；本机时钟快 ~7 min |
| **14** | ★★ **引用一个东西之前先 `ls` / `md5sum`**（本轮踩过两次：引用不存在的工具、发错版本的 `model.py`） |
| **15** | 始终用**简体中文**回复 |

---

## 3. 两条主线的完成度

### 3.1 线 1 · DRAM KV 卸载 ✅ 可上线（8 卡真权重）

| 判据 | 档 B | 档 C | 档 D |
|---|---|---|---|
| `BlockStored:CPU` | 29,436 | 29,436 | 29,436 |
| `CPU→GPU` | 21.52 GB | 21.19 GB | 12.11 GB |
| `hits` | 901,120 | 901,120 | 901,120 |
| **replay÷fill** | 12.87× | 12.50× | 12.87× |
| `BlockRemoved:CPU` | 0 | 0 | 0 |
| 图模式 | ✅ | ✅ | ✅ |

★ **int8 的真实收益是宿主内存，不是 HBM 容量**：**197.21 → 150.01 GiB（×1.3146）**。

### 3.2 线 2 · KV8 / int8 —— 能力通、目标部分达成

| | 结果 |
|---|---|
| "读侧不反量化喂原算子" | ✅ PA_BBND scratch + identity block table（**TND 路线实测不存在**） |
| 图兼容（Phase 1.2/1.3 卡点） | ✅ 已解（`049` 的 `GRAPH_SAFE`，档 C/D 都过捕获） |
| ★ **×1.84 目标** | ⛔ **未达**：A2 真权重 **档 C ×1.0000**、**档 D ×1.1356** |
| 根因 | **draft 组（BF16/block=128）顶死 slots 0–2**（`050` 零参数模型 **13/13 逐字命中**） |

**三条"解开 draft 天花板"的路**：

* ★ **②c（draft block 128→64，保 BF16）预测 ×1.8177** —— **唯一活路**，8 卡端到端在跑；
* ②a（draft 也 INT8，预测 ×1.9126）—— **已实测否决**（见 §4.1）；
* ③c（per-request scratch）—— 未做。

---

## 4. 本轮最重要的三条因果链

### 4.1 ②a（draft 也 INT8）—— **Q1/Q2 过、Q3 死**（`056`）

```
Q1 图捕获 ✅ capture_finished=1 / EE1016=0   ← 顺带推翻 050 的"必炸"判断（049 已覆盖 draft 面）
Q2 容量 ✅ page_bytes=66560；tiny 档 D 23,651→39,846 逐字命中模型（13/13）
Q3 ⛔ RuntimeError: The previous device metadata submission has not been released
       @ worker/device_metadata.py:74（触发：num_scheduled_tokens=6 + 5 spec token）
       ①eager 臂也挂同一条 ⇒ ★ 与"入图"无关，是【请求路径】的问题
       ②对照臂（唯一变量 DRAFT_INT8=0）全绿 ⇒ ②a 特有
```
**诊断臂**把泄漏点定位到 `submit#2`（只 1 个任务、`group_id` 在 target 的 7 任务里从未出现 ⇒ 来自 **draft 侧 builder**）。
⚠️ **"病灶 = `dsa_v1.py` 缺量化存取"已降级为【推断】**（探针钩错类：真身是 `DSAAttention`；首个异常在日志里看不见）。

### 4.2 "热 ≠ 冷" —— **一条判据整体作废**（`062`）

> **BF16 无损池的 hot 也 != cold**（3/16）⇒ "热 == 冷逐字相同"测的是**路径**不是**保真**，**与 int8 无关**。

⇒ int8 的非回归由两条独立证据支撑（`fill` 轮五臂逐字相同、热 replay 相对 **BF16 hot** 逐字相同）；
**正面保真判据（KV 级逐字节）仍未跑** —— 那是**新探针工程**（要重建 `transfer_async` 的指针表），需独占 c0 一轮。

### 4.3 档 C 基线的投机读数 —— **`max_tokens=1` 是无效口径**（`T`）

```
档 C 基线：MeanAccLen 2.46 / AvgDraftAcc 29.2% / Drafted 530
max_tokens=1：MeanAccLen 1.50 / AvgDraftAcc 10%
⇒ ★ 用 max_tokens=1 判断接受率无效（候选近平局、样本量不足）
```

---

## 5. 交付物状态（**已推送 GitHub**）

**`chiro2001/deepseek-v4.1-flash-ascend910B` → 分支 `feat/kv8-dram-offload-pending`**（最新 `2eeee4b`，工作区干净）。

关键件：

```
a2/docs/A2-DEPLOY-NOW.md   ★★ 上线清单（三条命令 + 判据账 + 档位门 + 回滚表 + 跨芯片外推 + A2 模型差异）
a2/DELIVERY.md             ★ 交付单一入口
a2/publish/kv8-graphsafe/dsa_v41.py    ★ 档 C/D 必需件 md5 94aeebb7…（已实测）
a2/publish/kv8-int8-pkg/   ★ 档 C/D 的另外 6 个挂载件（7 件齐全，含合并版 model.py c4b70d00…）
a2/publish/0004-draft-block64.patch.py ★ ②c 的补丁
a2/scripts/a2_one_shot_probe.sh  ★ A2 上第一条命令（探测池后端）
a2/scripts/make_shadow_pkg.sh    ★ 从发布包自己造 shadow（含 7 件挂载 + 防重复挂载门）
a2/scripts/check_artifact_identity.sh ★ 交付件身份台账的机械门
a2/logs/README.md          ★ 日志索引（001–063，0 死链）
```

### ★ 本轮修掉的七个"静默缺口"（都不会在任何测试里报错，只让上线的人第一步卡住）

| # | 缺口 |
|---|---|
| ① | shadow-pkg 只存在于开发机 |
| ② | 档 C/D 需 7 个挂载件而包里只有 1 个 |
| ③ | ②c 补丁不在包里 |
| ④ | dry-run 从来没验到挂载块 |
| ⑤ | ★ `model.py` 会打掉 A2 的 5 处生产补丁（是我自己引入的） |
| ⑥ | 档位静默降档 |
| ⑦ | **A2 的模型与 A3 实测的不是同一个**（`v41-w4a8-flat` vs `v41-w4a8-engram-dr-vision-qrot-mtpq`；共同点：都有 Engram 2 层 + mtpq 4 分片） |

---

## 6. 下一步（按优先级）

| # | 事项 | 谁 | 说明 |
|---|---|---|---|
| **1** | ★★ **A2 本机的池后端探测** | **用户** | `cd <dsv41-release>/a2/scripts` 后跑 `A2_CONTAINER=dsv41-a2 A2PROBE_FLOOR_GIB=300 LIGHT=1 bash a2_one_shot_probe.sh` → 看 **`★ 注册内存的设备往返判据 = True/False`**（**H2H 通过不算数**） |
| **2** | ★★ **等 `t-dc2-b-D`（档 D）出数** | `T_draftceiling` | 同时定性 ②c 起服失败 + 给档 D 的投机四数 |
| **3** | ★★ **②c 的 8 卡端到端**（`t-dc2-c-C` 重跑或定性） | `T_draftceiling` | **决定线 2 能否碰到 ×1.84** |
| **4** | ⏳ **②a 的"病行"探针**（钩到 `DSAAttention` 真身，而非上一轮钩错的类） | ★ **已派出**：`DS_draft_graph_int8`（**c1**，~85–141 s/臂） | 抓 `submit#2` 那条 1-task 提交的首个异常；**判据必须在 `DRAFT_INT8=1`/`=0` 对称臂上都跑** |
| **5** | ⏳ **KV 级逐字节保真**（`048`/`062` 都标【未跑】） | 待派 | 新探针工程，需独占 c0 一轮 |

---

## 7. 关键路径

```
a2/
├── HANDOVER.md            ← 本文件（每次大进展后重写）
├── AGENTS.md              ← 红线 / 环境 / §5b 探针纪律（9 条）/ cannbot 索引
├── DELIVERY.md            ← 交付单一入口
├── docs/A2-DEPLOY-NOW.md  ← ★★ 上线清单
├── docs/A2-GO-LIVE.md     ← 详细参数与判据
├── logs/README.md         ← ★★ 日志索引（001–063，每份一行摘要）
├── logs/NNN-*.md          ← 实验日志（【实测】/【推断】/【未确认】）
├── logs/raw/NNN-*/        ← 原始数据
├── publish/               ← ★ 可交付补丁集 + 参数定值 + 就绪度
├── scripts/
│   ├── a2_one_shot_probe.sh   ← ★ A2 上跑这一条（探测 + DECISION）
│   ├── make_shadow_pkg.sh     ← 造 shadow（7 件挂载 + 防重复挂载门）
│   └── tmpdir.sh              ← ★ 临时空间协议
└── agents/<代号>/         ← 每个子代理的工作区
```

**对外**：`chiro2001/deepseek-v4.1-flash-ascend910B` → 分支 **`feat/kv8-dram-offload-pending`**
（⚠️ **未经用户允许，不发 PR / 不发 issue / 不评论**）

---

## 8. 必读的五份日志（按重要度）

| 顺序 | 文件 | 作用 |
|---|---|---|
| 1 | `logs/050-20260922-draft-ceiling.md` | ★★★ draft 天花板 + 零参数容量模型（13/13 命中） |
| 2 | `logs/056-20260922-draft-int8-1die.md` | ★★ ②a 的完整判决（Q1/Q2 过、Q3 死） |
| 3 | `logs/062-20260922-hot-cold-verdict.md` | ★★★ "热≠冷"的定性（判据判别力不足，非 int8 缺陷） |
| 4 | `logs/055-20260922-a2-launch-path.md` | ★★ 上线路径打通的三个缺口 |
| 5 | `logs/063-20260922-single-die-substitution.md` | ★ "单卡能否替代 8 卡"的规范化答案 |
