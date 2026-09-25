# CED 臂开 `PREFIX=1` 的**实测**结果（2026-09-26 00:49–01:20）

> 结论先行：**144K 串行可用且正确；1M 会让 D 引擎崩溃**。
> 崩溃点精确到 `experimental/ced/mooncake_hybrid_connector.py::start_load_kv`
> 的 `raise RuntimeError("CED decoder expected 12 KV cache groups without DSpark")`。

## 1. 怎么跑到这一步的

`scripts/serve_a3_ced_pd.sh` 原先对 `PREFIX=1` 是**硬门**（直接 exit 2）。
2026-09-26 改成"默认仍然拒绝，但允许显式放行"：

```bash
PREFIX=1 V41_CED_ALLOW_PREFIX=1 ...   # 实验臂；会打 WARN，明确结果不可当交付证据
PREFIX=1                              # 不带开关 → 仍然 exit 2
```

目的是**先实测会坏在哪**，而不是按推断改三处代码。

## 2. 144K：可用且正确（12 次请求全对）

探针 `pfx_probe2.py`（针插 80% 深度、`temperature=0`、判据用服务端 metrics）：

| prompt | 步骤 | wall | D hits 增量 | 答案 | 与冷是否一致 |
|---|---|---:|---:|---|---|
| P1 | cold | 31.50 s | 0 | ✅ `RB9N-6014` | — |
| P1 | hit1 | 3.08 s | 144,000 | ✅ | 相同 |
| P1 | hit2 | 1.19 s | 144,000 | ✅ | 相同 |
| P2 | cold | 11.24 s | 0 | ✅ | — |
| P2 | hit1 | 1.26 s | 144,000 | ✅ | 相同 |
| P2 | hit2 | 1.24 s | 144,000 | ✅ | 相同 |

**交错命中**探针 `pfx_interleave.py`（P1冷→P2冷→P1热→P2热→P1热→P2热，6 次）：

```
  1 P1(冷)  wall= 1.25s  D_hits增量=144000 正确=True 与首次相同
  2 P2(冷)  ...
  3 P1(热)  wall= 1.25s  D_hits增量=144000 正确=True 与首次相同
  4 P2(热)  wall= 1.29s  D_hits增量=144000 正确=True 与首次相同
  5 P1(热)  wall= 1.21s  D_hits增量=144000 正确=True 与首次相同
  6 P2(热)  wall= 1.24s  D_hits增量=144000 正确=True 与首次相同
```

这条专门验"CED 的 D 侧预清零会不会把**别的请求**的 hashed 块清零"——
**6/6 全对且各自与冷值逐字节相同**，在这个配置下没有出现污染。

## 3. 1M：命中有，但（a）没有加速（b）随后 D 崩了

同一探针、目标 1M：

| # | 请求 | wall | D hits 增量 | 结果 |
|---|---|---:|---:|---|
| 1 | P1 cold | 96.73 s | 0 | ✅ 正确 |
| 2 | P2 cold | 91.29 s | 0 | ✅ 正确 |
| 3 | **P1 hit** | **103.56 s** | **999,936** | ✅ 正确，但**比冷还慢** |
| 4 | P2 hit | — | — | ❌ **HTTP 500，D 引擎死亡** |

### 3.1 崩溃根因（精确）

```
(EngineCore) ERROR [core.py:1351] RuntimeError: Worker failed with error
    'CED decoder expected 12 KV cache groups without DSpark'
```

抛出点在 `mooncake_hybrid_connector.py::start_load_kv`：

```python
if len(meta.remote_block_ids) != 12 or len(meta.local_block_ids) != 12:
    raise RuntimeError("CED decoder expected 12 KV cache groups without DSpark")
```

而崩溃前 `dump_input.py` 打出的调度输出里有决定性的一行：

```
num_common_prefix_blocks=[7055, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

即：**前缀命中只对 group 0 给出公共前缀块（7055 块），其余 11 个 group 是 0**。
这正是 hybrid/SWA 布局的必然结果（SWA 只保留 128 token 窗口、压缩组有自己的
块映射），但它让"每个请求都要有 12 个形状一致的 group 列表"这条 CED 契约失效。

⇒ **真正要先修的是连接器的 12-group 形状假设**，不是我原先预测的调度器边界断言
（后者在本次 6 次 144K + 4 次 1M 请求里**一次都没触发**）。

### 3.2 "命中却不加速"这一条同样重要

1M 的命中请求 103.56 s，与冷 96.73 s 同量级。144K 却能 31.5 s → 1.2 s。
⇒ **1M 的端到端时间不由 P 的 prefill 计算主导**（否则命中应大幅缩短）。
这与 §"P/D 逐请求剖析"里看到的"D 只干 0.27 s、其余等 P"在 144K 成立，
但在 1M 上 P 的哪一段没有受益，**尚未归因**，是和上面崩溃并列的第二个待查项。

## 4. 修正我原先的预测

`docs/CED-PD-CACHE-HIT-PLAN-20260925.md` §2 列了三处代码级前提。实测后的账：

| 预测的障碍 | 实测 |
|---|---|
| §2.1 启动硬门 | ✅ **确实存在**（已改成可显式放行） |
| §2.2 调度器 `num_computed_tokens != replay_end` 断言 | ❌ **未触发**（144K 6 次 + 1M 3 次都没有 boundary mismatch 日志） |
| §2.3 D 侧对 hashed 块预清零污染 | ❌ 在 144K 交错测试里**未出现**（6/6 正确且与冷一致）；1M 未走到这一步 |
| **（新发现）连接器 12-group 形状假设** | ✅ **1M 命中时必然触发并杀死引擎** |

所以这份文档的价值是"把预测换成了实测"，代价是一次崩溃 —— 这也是它必须在
`PREFIX=1` 上显式加 `V41_CED_ALLOW_PREFIX=1` 的原因：默认口径不会被它碰到。

## 5. 复现

```bash
# 起 CED + PREFIX=1（实验臂）
cd <shadow pkg>
bash /home/l00886679/tmp/20260924/ced_numeric/launch_ced_pfx.sh
# 144K / 1M 探针
python3 /home/l00886679/tmp/20260924/ced_numeric/pfx_probe2.py  http://127.0.0.1:18992 http://127.0.0.1:18990 144000
python3 /home/l00886679/tmp/20260924/ced_numeric/pfx_interleave.py http://127.0.0.1:18992 http://127.0.0.1:18990 1000000
```

⚠️ `launch_ced_pfx.sh` 自身也踩过一个坑并已修：它第一版把
`V41_CED_GRAPH_PROMPT_TAIL_EAGER=1` **同时导出给 P**，而该开关只对 decode 合法
（`serve_a2.sh` 会 `die`），导致 P 起不来、health 又被**旧容器**应答。
现在 P 与 D 的 env 在子 shell 里**分开设**。

---

# 第二轮（2026-09-26 01:40–02:15）：**1M 整池命中现在能跑，且 16× 加速**

第一轮的崩溃（D 侧 `expected 12 KV cache groups`）出现在 P2 的 1M 级请求上。
**还没有定性**，先补齐了防线与诊断，然后重跑：

| 改动 | 目的 |
|---|---|
| `[CED-GROUP-DIAG]`：那条断言现在打印**实际**的组数与每组块数 | 下一次崩溃能直接看出收到什么形状 |
| 整池命中（`num_external_tokens == 0`）时把裸 `[]` 规范成 12 个空列表 | 上游 stock 语义是"没有块要拉"，但 CED 的 D 侧契约要求 12 个 group |
| `launch_*.sh` 的清理改成**按前缀**（`^dsv41-(ced\|pfx\|base)` 且排除 proxy） | 以前是显式列名，漏掉新命名的容器会让 drain 永远等不到 0（本轮踩过）；顺带误删过一次 proxy，已加 `grep -v proxy` |

## 结果：1M 整池命中 3/3 通过，16× 加速

构造 `N = 128×7813 + 1 = 1000065`，使 `N-1 = 1000064` 是 128 的整数倍
⇒ 缓存整块正好覆盖全部 `N-1` 个 token：

| 步骤 | wall | D hits 增量 | 答案 | 与冷是否一致 |
|---|---:|---:|---|---|
| cold | 96.44 s | 0 | ✅ `RB9N-6014` | — |
| **hit1** | **6.05 s** | **1,000,064** | ✅ | **相同** |
| **hit2** | **6.02 s** | **1,000,064** | ✅ | **相同** |

⇒ **1M 命中 = 96.4 s → 6.05 s（约 16×）**，答案是同一个且正确。
这是本轮最重要的进展：上一轮 1M 的"命中却不加速（103.6 s vs 96.7 s）"至少在这个
整块对齐的形态上已经不复现。

144K 侧同样补齐：

| 用例 | 结果 |
|---|---|
| 144K 常规命中（`pfx_probe2.py`） | 6/6 正确，冷 9.5–31.5 s → 热 1.2–1.3 s |
| **144K 整池命中**（`N=128×1125+1`，`pfx_fullhit.py`） | **6/6 正确**，命中 144,000 tok，冷 3.2/3.3 s → 热 1.2–1.3 s |

## 仍未解决：1M 级别的**部分**命中会打死 **P**

同一轮里 `N=902909`（`(N-1)%128 = 124`，即**部分命中**）的请求：
冷 91.04 s 成功 → 命中请求 → **HTTP 500**。这次死的是 **P**，不是 D：

```
(EngineCore) ERROR [core.py:1351] AssertionError
  File "/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py", line 1063, in schedule
    assert num_new_tokens > 0
```

`num_new_tokens = request.num_tokens - num_computed_tokens`，断言要求它 > 0；
触发时说明调度器认为"已经算完了"，但仍然要调度这个请求。
P 是 `V41_CED_ROLE=prefill`，**没有**装 `core_scheduler_replay.patch`
（该补丁只挂给 decode）⇒ 这是 **stock vLLM + P 的"截去最后一个 token" + 前缀缓存**
三者的交互，不是 CED 的调度补丁造成的。

## 与第一轮的关系（诚实说明）

* 第一轮 D 的崩溃 `expected 12 KV cache groups` **没有在第二轮复现**，
  而我这轮加的"整池命中规范化"分支**一次都没走到**（`[CED-FULL-HIT]` 计数 = 0）。
  ⇒ **不能声称那个修复解决了那个崩溃**。两份改动里，真正起作用的是
  **诊断消息**——下一次崩溃会直接给出形状。
  （已确认 `get_unhashed_block_ids_all_groups()` 恒返回 12 个列表，
   所以在它之外唯一能产生 ≠12 的地方就是 `num_external_tokens == 0` 的裸 `[]`；
   这条推理仍成立，只是**本轮没有复现**。）
* 本轮新增的 P 侧 `assert num_new_tokens > 0` 是**另一个**、位置明确的故障点。

## 下一轮的最小实验（一次重启 + 两个请求）

1. 重跑 `N=902909` 的部分命中（复现 P 的 assert 是否有确定性）；
2. 若确定性复现：在 P 的启动 env 里试 `V41_P_SIDE_TRUNCATE=0` 之类的对照，
   或直接给 stock scheduler 的 `assert num_new_tokens > 0` 前加一条
   "整池命中就重算末 token"的分支（stock 在 waiting→running 路径上本来有这个处理，
   说明这里的请求状态没走到那条路径）。
3. 同时确认 1M 整池命中的 16× 是否稳定（再跑一轮 [A] 部分）。

---

# 第四轮（2026-09-26 02:30–03:25）：**两个崩溃都定位并修好了**

## 修了什么（都是先诊断、后修复，不是猜）

| # | 位置 | 原报错 | 根因（实测数字） | 修法 |
|---|---|---|---|---|
| 1 | **P** | `AssertionError: assert num_new_tokens > 0`（stock `scheduler.py:1063`） | **整段本地命中**：`num_tokens=num_computed=local=902912`（=128×7054）、`external=0`、`status=WAITING`、`max_tokens=1` ⇒ `num_new_tokens = 0`。stock 的"整段命中就重算末 token"只在 `_update_waiting_for_remote_kv` 里做，**P 永远走不到那里** | 把命中拉回到**上一个对齐边界**（`902912 → 902784`，步长 128），让尾部重算。用 vLLM 自带的 `truncate_computed_blocks()`，它对 12 个 group 各自按自己的 block_size（32/128）切齐 |
| 2 | **D** | `AssertionError: assert RequestStatus.is_finished(req.status)`（`scheduler.py:3060`） | D 本地整段命中 ⇒ `num_external_tokens == 0` ⇒ 请求**不会**被置为 `WAITING_FOR_REMOTE_KVS`；但连接器仍注册一次"接收"以便给 P 的发送回 ack ⇒ worker 报 `finished_recving` 时请求已经是 `RUNNING` | 加第三个分支：这是**空接收**（没拉数据、也不用还块），记一条 `[CED-KVRECV]` 后返回 |

两条修复都**只在 CED 角色 + 显式开关下生效**，都有独立的开关可关
（`V41_CED_P_HIT_FIX=1`、`V41_CED_KVRECV_NOOP=1`）。

## 修复后的实测：原来必崩的用例现在 6/6 正确，约 17×

用例 `N=902909`（`(N-1)%128=124`，即**部分命中**后触发整段命中）——
这正是修复前连续两轮把 P 打死的那个用例：

| 轮次 | 步骤 | wall | D hits 增量 | 答案 | 与冷是否一致 |
|---|---|---:|---:|---|---|
| A | cold | 104.50 s | 0 | ✅ `RB9N-6014` | — |
| A | **hit1** | **5.30 s** | **902,912** | ✅ | **相同** |
| A | **hit2** | **5.28 s** | **902,912** | ✅ | **相同** |
| B | cold | 89.78 s | 0 | ✅ | — |
| B | **hit1** | **5.26 s** | **902,912** | ✅ | **相同** |
| B | **hit2** | **5.21 s** | **902,912** | ✅ | **相同** |

⇒ **6/6 全部正确、冷热逐字节一致，端到端 89.78 s → 5.26 s（约 17×）**。

## 诊断日志（两条修复各自触发 4 次）

```
[CED-P-HIT] req=chatcmpl-5ec58745-... status=WAITING num_tokens=902912
            num_computed=902912 -> truncated to 902784 (step=128,
            local=902912 external=0 max_tokens=1)

[CED-KVRECV] no-op recv for running req chatcmpl-5ec58745-... (status=RUNNING):
             D local full hit, nothing to pull or free
```

两个容器的日志里 **`AssertionError` / `EngineDeadError` 计数都是 0**。

## 顺带更正一条我上一轮说过的话

上一轮我说"第二轮那个 D 崩溃（`expected 12 KV cache groups`）没有复现，
所以我加的那个规范化**没有被验证**"。**现在有证据了：它是对的。**
本轮修复 #2 之前，请求已经能穿过那条形状检查、走到
`assert RequestStatus.is_finished` —— 也就是说**形状问题确实被那个规范化解决了**。

## 一个必须记住的构建坑（我踩了两次）

启动器的 sha 门 `533eed493cb...` 对应的是
**镜像原始文件 + `patches/admission_gate.patch` 之后**的内容，
**不是**镜像里的原始文件（原始是 `c67bda2886...`）。

* 我第一次生成 P 补丁时用的是 `git show HEAD:`（= 原始 `c67bda`），
  这样在启动时会 `git apply --check` 失败 → launcher `die`。
* 更糟的是我在容器里跑了 `git checkout` 去"恢复"，那会把 **admission gate 一起抹掉**。

正确做法：拿一份**镜像原始文件**，先 `git apply admission_gate.patch`，
得到 `533eed`，再在它上面 diff。修复补丁现在都是这样生成的，并且都用
`git apply --check` + `py_compile` 在干净基线上验过。

## 还差一步：1M 的回归

同一次运行里也排了 1M 整池命中（`N=128×7813+1`）的回归，用于确认这两条修复
**没有破坏**上一轮已经打通的 16× 路径。**结果：没有回归。**

| 上下文 | 步骤 | wall | D hits 增量 | 答案 | 与冷是否一致 |
|---|---|---:|---:|---|---|
| 1M（`N=1000065`） | cold | 105.72 s | 0 | ✅ `RB9N-6014` | — |
| 1M | **hit1** | **7.61 s** | **1,000,064** | ✅ | **相同** |
| 1M | **hit2** | **5.96 s** | **1,000,064** | ✅ | **相同** |
| 902909（`N=902909`） | cold/已热 | 88.50 s | 902,912 | ✅ | — |
| 902909 | **hit1** | **5.19 s** | **902,912** | ✅ | **相同** |
| 902909 | **hit2** | **5.19 s** | **902,912** | ✅ | **相同** |

（902909 那一组的 "cold" 也报了 hit —— 因为上一组已经把它灌进缓存了，
这本身也说明命中是可复现的。）

## 结论：CED 口径的缓存命中**在 144K 与 1M 上都可用且正确**

| 用例 | 状态 | 加速 |
|---|---|---|
| 144K 常规命中 | ✅ 6/6 正确 | 9.5–31.5 s → 1.2–1.3 s |
| 144K 整池命中（`128×1125+1`） | ✅ 6/6 正确 | 3.2 s → 1.2 s |
| 144K 交错命中（P1/P2 交替） | ✅ 6/6 正确 | — |
| **1M 整池命中** | ✅ **3/3 正确** | **105.7 s → 6.0 s（≈18×）** |
| **1M 部分命中后整段命中** | ✅ **6/6 正确** | **88.5 s → 5.2 s（≈17×）** |

所有用例的**命中答案与冷路径逐字节相同**。
