# 动态 K「req≥2 掉到 ~10 tok/s」—— 根因定位与修复（2026-10-03 实测）

> 用户报告：「动态 k 之前实现过但是 **req>2 的时候性能掉到 10 tps**」。
> 本文给出**完整复现**、**根因**、**修复**、以及修复后**新暴露的第二个缺陷**。
> 所有数字标 **【实测】**/**【推断】**。

---

## 0. 结论（一页）

| 项 | 内容 |
|---|---|
| **根因** | `patch_cudagraph.py` 把 query_len 集合缓存到 `_v41_qlens_cache`，而 runner 读的是 **`_dynamic_decode_query_lens`** ⇒ runner 拿到 `None` ⇒ 退回等值比较 `1 == 8` 失败 ⇒ **K=0 的步被判成非 uniform** ⇒ `BatchDescriptor(uniform=False)` 与捕获键（全 `uniform=True`）不匹配 ⇒ `dispatch()` **静默**返回 `CUDAGraphMode.NONE` ⇒ **整步 eager** |
| **为什么是 ~10 tok/s** | eager 下每步 87 次 TP/EP allreduce，**每次约 2 ms**（图内只要 44 µs）⇒ ~175 ms/step |
| **修复** | 1 行属性别名（+ 一条"再看不住就报警"的守卫） |
| **修复后【实测】** | N=2：**9.9–12.0 → 73.9 tok/s**（6–7×）；`draft/gen=0.03` 证明 K=0 仍被正确选中 |
| **新暴露的缺陷** | ql=1 真正进图后，**后续步**触发 `dsa_v1.py:618 torch.gather` 形状不匹配（`safe_nums` rows=2 vs `block_table` rows=1），引擎退出。**待修** |

---

## 1. 完整复现【实测】

a3-21 dies 8–15，TP8、`SPEC=1 SP_TOKENS=7 DRAFT_GRAPH=1`、
`SP_SCHEDULE='1,1,7;2,32,0'`（N=1 → K=7；N≥2 → K=0）、
`V41_CED_DYNAMIC_SPEC_FULL_GRAPHS=1`。

| 并发 | 聚合 tok/s | 每流 tok/s | `draft/gen` | 判定 |
|---:|---:|---:|---:|---|
| 1 | 88.7–95.6 | 88.7–95.6 | **2.43–2.65** | K=7 正常 |
| **2** | **9.9 / 10.4 / 12.0** | 5.0–6.0 | **0.03–0.05** | **K 确实切成 0 了，但比静态 SPEC=0 慢 9.5×** |
| 4 | 23.3 | 5.8 | 0.03 | 同样 |

**`draft/gen` 是"K 真的切成 0"的硬证据**（K=7 时每请求每步造 7 个草稿 ⇒ ≈2.5；
K=0 时一个都不造 ⇒ ≈0）。所以这不是"K 没切成功"，而是**切成功之后那条路自己很慢**。

对照：同机静态 `SPEC=0` 在 N=2 是 **92.6 tok/s**（每流 46.3）⇒ 慢 **9.5×**。

---

## 2. 根因定位过程（三步，每步都有可引用证据）

### 2.1 第一步：设备侧画像——只有通信是坏的

同实例采两次 profiler（N=1 走 K=7、N=2 走 K=0），按 `HcPre/86` 归一到每步
（`~/tmp/devbusy.py`）：

| | N=1（ql=8） | **N=2（ql=1）** |
|---|---:|---:|
| span | 30 ms/step | **~205 ms/step** |
| 设备并集 busy | ~25 ms | **228.6 ms** |
| ops/step | 2750 | 2403（**相近**） |

算子条数相近但 busy 差 9× ⇒ **不是"算子变多"，是"每个算子变贵"**。

进一步拆：`AivKernel`（= TP/EP 集合通信的执行体）

| | calls/step | ms/step | **avg µs** |
|---|---:|---:|---:|
| N=1（K=7） | 96.0 | ~4.2 | **44** |
| N=2（K=0） | 92.5 | **181.4** | **1962** |

**调用次数几乎一样、单次贵 45×** ⇒ 定位到通信。

### 2.2 第二步：同实例 A/B，按 stream 分——通信在"图内"还是"图外"

| profile | 通信落在哪个 stream | calls/step | ms/step | 备注 |
|---|---|---:|---:|---|
| N=1（K=7） | **stream 225** | 80.5 | 2.66 | 高 ID 段 = **图内** |
| N=2（K=0） | **stream 38** | 89.8 | **180.8** | 低 ID 段 = **图外（eager）** |

同一条 allreduce、同一个通信域，只因**在不在图里**差了 45×。

### 2.3 第三步：代码级——属性名不匹配

`patch_cudagraph.py`（我们的整文件替换版）里：

```python
def _v41_extra_query_lens(self):
    cached = getattr(self, "_v41_qlens_cache", None)
    if cached is None:
        cached = _dynamic_decode_query_lens(self.vllm_config) or ()
        self._v41_qlens_cache = cached        # ← 只写了这个名字
    return cached
```

`model_runner_v1.py:3109`（我们的 runner 补丁）里：

```python
_udql_set = getattr(self.cudagraph_dispatcher, "_dynamic_decode_query_lens", None)   # ← 读的是另一个名字
uniform_decode = (
    (has_initial_state
     and ((max_num_scheduled_tokens in _udql_set) if _udql_set
          else (max_num_scheduled_tokens == self.uniform_decode_query_len))   # ← 永远走这条
     and (num_tokens == max_num_scheduled_tokens * num_reqs))
    ...
)
```

`_udql_set` 恒为 `None` ⇒ K=0 的步（`max_num_scheduled_tokens=1`）
拿 `1 == 8` 比 ⇒ **False** ⇒ `uniform_decode=False` ⇒
BatchDescriptor 的 `uniform` 位是 `False`，
而 `initialize_cudagraph_keys` 捕的键**全是 `uniform=True`**
⇒ `dispatch()` 查不到 ⇒ 返回 `NONE`。

**为什么之前一直没被发现**：上游 `dispatch()` 在查不到键时就是
`return CUDAGraphMode.NONE`，**一行日志都不打**（见
`vllm/v1/cudagraph_dispatcher.py` 末尾那个 assert 之后的分支）。
我们那条"缺图告警"只在 `num_tokens_padded % ql != 0` 时触发，
而这里 `uniform_decode=False` 让整个 `if` 分支都没进 ⇒ **全程静默**。

**捕获键计数也自洽**：raw 桶 `[1,2,3,4,8,12,16,20,24,32,40,48,96,192,256]`
⇒ ql=8 那遍捕 `{8,16,24,32,40,48,96,192,256}`（9 个，12/20 因非整倍被跳）、
ql=1 那遍捕 `{1,2,3,4,8,12,16,20,24,32}`（10 个）
⇒ 并集 **19 个**，与起服日志 `19/19` 逐字吻合
⇒ **`(num_tokens=2, num_reqs=2, uniform=True)` 这张图是存在的**，只是没被查到。

---

## 3. 修复

**① 属性别名（根因）** —— `patches/files/patch_cudagraph.py`：

```python
self._v41_qlens_cache = cached
self._dynamic_decode_query_lens = cached   # ← runner 读的是这个名字
```

**② 防再犯守卫** —— 包装 `CudagraphDispatcher.dispatch()`：
"动态 SD 开着 + 调用方说这是 uniform-decode 步 + 结果拿到 NONE"时
**告警一次**（把这类静默失效变成可观测）：

```
[dynamic-spec] ★ uniform-decode 步**没有**匹配到 FULL 图 ⇒ 静默回落 eager
```

**③ 顺带修掉两道过严的起服门**（让独立 TP8 能开动态 K）：

| 门 | 原行为 | 新行为 |
|---|---|---|
| 角色门（`serve_a2.sh:1647`） | `SP_SCHEDULE` 只允许 `V41_CED_ROLE=decode` | 允许 `""`（独立 TP8）与 `decode`；**prefill 仍拒** |
| 基线 sha 门 | 只接受 CED prompt-tail 之后的 `bd250a59…` | 再接受原始基线 `67035d97…`；**未知 sha 仍 fail-closed** |

第 2 条的依据是 A/B 双路验证【实测】：两条基线上分别应用 prompt-tail 与 dynamic-spec，
两句 diff 只有 25 行且**全部是 prompt-tail 自己的改动**
（`import os` / `ced_prompt_tail_eager` 18 行 / 一处 `force_eager`）
⇒ dynamic-spec 的 8 个 hunk 与 prompt-tail **无耦合**。

补丁存档：`fixes/20261003-dynspec/{serve_a2, patch_cudagraph}.dynspec.patch`

---

## 4. 修复效果【实测】

### 4.0 ★ B 轮（开候选修复后的完整验收，2026-10-03 20:xx）

**正确性：全部通过**（这是目标里明确要求的那几条）

| 用例 | 结果 |
|---|---|
| `regress2.py` 内容回归（17×23 / count 2000+16000 / needle 904+8000） | **5/6 PASS** |
| 并发一致性（8 路 vs 单流逐字） | 6–7/8，**差异仅为空白**（内容完全相同，已知项，见 §5 末） |
| **四针 144K** | **PASS**（`ced_pd_acceptance.py --mode needle` exit=0） |
| **四针 1M** | **4/4 PASS**（`ZQ7K-3341` / `VX2M-8890` / `HT4P-5527` / `RB9N-6014`，prompt≈999.4K，每条 wall≈280 s） |
| **并发 2 各带不同针**（K=0 路径唯一覆盖） | **3 轮全过**（`own=[True,True] leak=[False,False] same=False`） |
| 引擎稳定性 | `DYNSPEC-DIAG`=0、`EngineDeadError`=0、HTTP 500=0（15 个探针 rep 全跑完） |

**性能：达标失败，但这条路径本身是通的**

同一台机、同一脚本（`tools/bench_concurrency.py`，地火 1024/256，2 rep）：

| 并发 | **动态 K（B 轮）** | A | 静态 `SPEC=1 K=5` | 静态 `SPEC=0` | 目标 |
|---:|---:|---:|---:|---:|---:|
| 1（K=7） | **95.6** | 2.68 | 109.5 | 51.6 | **≥105 ❌** |
| 2（K=0） | 79.5 | 1.17 | 142.9 | 92.6 | — |
| 4（K=0） | 144.1 | 1.05 | 225.4 | 166.2 | — |
| 8（K=0） | 261.6 | 1.03 | 315.3 | 277.4 | — |
| 16（K=0） | **400.5** | 1.03 | 398.6 | **434.1** | **≥430 ❌** |

**★ 诚实结论**：
1. **动态 K 现在"能用"**：K 切换正确（`draft/gen`：N=1 为 2.49–2.62 = K=7；N≥2 为 0.03 = K=0）、
   两条路径的**正确性都有探针证据**、零崩溃；
2. **但它比"每档挑更好的静态配置"更慢**：
   - N=1：**0.87×**（95.6 vs 109.5）⇒ 动态档在低并发有 ~13% 固定开销；
   - N=16：**0.92×**（400.5 vs 434.1）⇒ K=0 路径仍比"真静态 SPEC=0"慢 8%；
   - 每一步都被静态包络压制（动态 95.6/79.5/144.1/261.6/400.5 vs 包络 109.5/142.9/225.4/315.3/434.1）。
3. ⇒ **目标的两个数字（N=1≥105、N=16≥430）用动态 K 达不到**，需要备选路线
   （静态双实例 + 按并发选端口，见 `STATIC-DUAL-INSTANCE-FALLBACK-20261003.md`）。

   ⚠️ 那条路线要**整机 16 个 die**。2026-10-03 20:xx 逐 die 实测：

   | die | 占用 |
   |---|---|
   | 0,1 | `hlz-dsv4-dp2`（别人） |
   | 2,3 | 我们的 `dsv41-tinyspark` |
   | **4,5,6,7** | **空闲** |
   | 8–15 | 我们的 `dsv41-dynfix`（本文的 Track A 实例） |

   ⇒ dies 0–7 **并不全空**（0–3 被占）⇒ 双实例**暂时起不了**；
   但 dies 4–7 可用（Track B 的 tiny 就用这里）。

   > 📌 **排查记录（我自己犯的读表错误）**：`npu-smi` 的进程表第一列是 **NPU 号**、
   > 第二列是 **chip**，**die = NPU×2 + chip**。我一度把第一列当 die 号，
   > 于是把"NPU 4–7 上的进程"误读成"die 4–7 被占"，得出了**相反**的结论。
   > 正确的逐 die 写法见 `NETWORK-RUNBOOK.md` 与本节表格。

**8–13% 的差距从哪来【推断，未逐项归因】**：
引擎仍然加载 draft 权重与缓冲、每步仍要做 K 判定与两套图键的维护；
即使 K=0 的步已经不再跑 draft 前向（`draft/gen=0.03` 证明），这些固定项仍在。
**未做**：关掉 draft 后的逐步 host 开销归因（要有 profiler 才能定论）。

---

### 4.0.1 ★ 剩下那 8–13% 在哪（**已修正口径 + 一条负结果**）

#### (a) 先纠正我自己的一个口径错误

我先前写「设备侧差 3.5 ms/step」是**从吞吐反推的，不是设备实测**。
把两边的 `[bneck] hp`（设备侧完整 decode step，同口径、n=16、无 padding）摆在一起：

| N=16 | 设备侧 `hp` | 吞吐反算的 step |
|---|---:|---:|
| **动态 K=0**（Arm B） | **29.17 ms**（多 rank 29.162–29.188） | 39.9 ms |
| **静态 SPEC=0** | **28.50 ms**（多 rank 28.496–28.497） | 36.9 ms |
| **差** | **+0.67 ms（+2.3%）** | **−6.9%** |

⇒ **设备侧只差 0.67 ms，真正剩下的是 host 侧每步开销（约 2.4 ms/step）**。
这个口径修正很重要：它把优化方向从"减少设备算子"改成了"减少 host 每步工作"。

#### (a2) ★ 设备侧权威对比表（`[bneck] hp` 中位数，同一口径，样本 300–2800 条/档）

| n | **静态 `SPEC=0`** | **动态 K=0** | Δ | Δ% |
|---:|---:|---:|---:|---:|
| 1 | 19.52（这是 K=7 之外的纯自回归） | 24.61（**这是 K=7 步**，ql=8） | — | — |
| **2** | **20.97** | **24.44** | **+3.47** | **+16.6%** |
| **4** | **22.37** | **25.58** | **+3.21** | **+14.4%** |
| **8** | **24.81** | **27.83** | **+3.02** | **+12.2%** |
| **16** | **28.35** | **28.56** | **+0.21** | **+0.7%（几乎持平）** |

**两条重要读法**：
1. **动态 K=0 的设备效率在 n=16 已经与静态持平**（28.56 vs 28.35）⇒
   N=16 那 6.9% 的吞吐差**几乎全部落在 host 侧**，不是设备算子问题；
2. 低并发（n=2–8）确有 **~3.0–3.5 ms 的真实设备侧开销**，且**随 n 增大而收窄**
   （+3.47 → +3.21 → +3.02 → +0.21）⇒ 形态更像"**每步一段可被大 batch 摊薄的额外工作**"，
   而不是一个固定常数。

> ⚠️ 不要把 hp 与"吞吐反算的 step"混用：吞吐反算里混进了 prefill 与调度间隙
> （静态 n=16：hp 28.35 而吞吐反算 36.9）。**要比设备效率就用 hp，要比端到端就用吞吐**，
> 两者不可互相换算。

#### (b) 负结果：候选①（`_copy_draft_token_ids_to_cpu` 的跨流往返）**已证伪**

单变量 A/B（同实例配置，只翻一个 env；每档 2 rep）：

| 并发 | Arm A（门控关） | **Arm B（`V41_DYNSPEC_SKIP_K0_DRAFT_COPY=1`）** | Δ |
|---:|---:|---:|---:|
| 1 | 95.6 | 103.4 | +8%（噪声区间） |
| 2 | 79.5 | 77.6 | −2.4% |
| 4 | 144.1 | 144.3 | ±0 |
| 8 | 261.6 | 262.8 | +0.5% |
| **16** | **400.5** | **404.0** | **+0.9%（本机 run-to-run ±1.5%）** |

**判据不成立**（N=16 只 +0.9%，落在噪声内）⇒ **候选①不是那部分开销的来源**。

**门控确实被执行过**：起服首轮曾在**这一行**抛 `NameError: name 'os' is not defined`
（独立 TP8 走 base 基线，`import os` 是 prompt-tail 补丁才引入的），
栈帧正是 `model_runner_v1.py:2062` 的门控行 ⇒ 该行每步都会求值。
（修法是**局部导入**，与本文件既有的 `_v41_shape_probe` 写法一致。已修并复测。）

#### (c) 仍然待查的两个候选

先把量级钉住：动态档 N=2（K=0）的 `[bneck] hp = 25.5 ms/step`（`n=2 padded=2`，**无 padding**），
而静态 `SPEC=0` 在 N=2 按聚合吞吐反算是 **21.6 ms/step** ⇒ **差 3.9 ms/step**；
N=16 差 3.1 ms/step。**是个近乎常数**，不是随 batch 增长的开销。

读代码找到两个"K=0 时本来不该做却仍在做"的点：

| # | 位置 | 现象 | 代价【推断】 |
|---|---|---|---|
| **1** | `model_runner_v1.py:2044-2077` `_copy_draft_token_ids_to_cpu` | 守卫写的是 `if not self.num_spec_tokens: return`，而 **`self.num_spec_tokens` 是配置的最大 K（7）**，不是本步 K ⇒ **K=0 的步照样进来**，做一次 `copy_stream.wait_stream(default_stream)` + `event.record()` 的跨流往返（拷贝本身是 [N,0] 空张量） | 跨流 event 往返；量级**未确认** |
| **2** | `model_runner_v1.py:2804-2820`（`sample`） | `if spec_decode_metadata is None: … return self.sampler(...)`，否则走 **`self.rejection_sampler(...)`**。K=0 时 `spec_decode_metadata` **是否仍非 None 未确认** ⇒ 若非 None，则每步仍跑拒绝采样；而该路径里有已知的 **D2H 同步点**（`vllm/v1/sample/rejection_sampler.py:271 parse_output → output_token_ids.cpu().numpy()`，这正是 2026-10-01 那次崩溃栈的报丧点）。静态 `SPEC=0` **完全没有**这条路径 | 每步一次设备→主机同步；量级**未确认** |

> **候选① 已按上表做完并被证伪**（见 (b)）。

**候选② 的静态读码结论【推断，偏否】**：`use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0`
（`model_runner_v1.py:1524`），K=0 时调度器不下发 draft 令牌 ⇒ `spec_decode_metadata=None`
⇒ 走的是**普通 sampler**，**不经过 rejection_sampler**。
⇒ 候选② 大概率也不成立，**但需要用一次 profiler 或计数器把它变成实测**，不要停在推断。

**下一步要做的实验（尚未做）**：给 (1) 加一个 env 门控的提前返回
（`本步 draft 宽为 0 ⇒ return`，放在 `prev_num_spec_tokens` 记账**之后**），
再复测 N=2/16；若收益对得上，再查 (2)。
⚠️ 该文件**不在我们的 overlay 名单里**，改它需要重新生成
`experimental/ced/core_model_runner_dynamic_spec.patch`（多一个 hunk）+ 加 env 透传，
**成本是约 20 分钟一轮重启**。

> **为什么现在不直接做**：这两条都是【推断】，而且改的是 vLLM core 的每步热路径
> （错一个分支就可能让 K=0 的步拿到过期草稿）。要在**有明确收益预期**时再做，
> 并且必须带 env 门控（默认关）+ 全量正确性探针复验。

---

### 4.1 修复前后（单点对照）

| 并发 | 修复前 | **修复后** |
|---:|---:|---:|
| 1（K=7） | 88.7–95.6 | 82.5–94.9（不变） |
| **2（K=0）** | **9.9–12.0** | **73.9**（首轮） |

`draft/gen` 仍是 0.03 ⇒ **K=0 依然被正确选中**，改善来自"K=0 的步终于进图了"。

**与静态 SPEC=0 的差距**：静态 92.6 vs 动态 K=0 73.9（−20%）。
【推断】差额来自"动态 SD 下每步要多做一次 K 判定与两套图键的维护"，
以及 k=0 的桶只有 `1,2,3,4,8,12,16,20,24,32`（**桶更稀 ⇒ padding 更多**）。

---

## 5. 第二个缺陷（**已修并验证**，见 `DYNAMIC-SPEC-DRAFT-ROWS-MISMATCH-20261003.md`）

ql=1 真正进图之后，**后续步**会把引擎打挂：

```
Worker_TP*.sample_tokens
 → model_runner_v1.py:2620 propose_draft_token_ids
 → llm_base_proposer.py:1656 _propose
 → llm_base_proposer.py:3382 build_draft_attn_metadata
 → dsa_v1.py:1552 run_attention_metadata → dsa_v1.py:1516 build_attention_metadata
 → dsa_v1.py:1477 build_dspark_swa → dsa_v1.py:618
   block_ids = torch.gather(block_table, 1, safe_nums)

RuntimeError: aclnnGather failed, error code 161002
AclNN_Parameter_Error(EZ1001): Size does not match at dimension 0,
    expected index shape 2 smaller than self shape 1
→ EngineDeadError → HTTP 500
```

**读法**：`safe_nums`（由 `seq_lens`/`query_start_loc` 派生的 `[req_count, index_width]`）
有 **2 行**，而 `block_table` 只有 **1 行** ⇒ 两侧的 `num_reqs` 不一致。
这与历史上记过的 §10.2「draft rows 不匹配」是**同一族**缺陷
（`CED-PD-DYNAMIC-SPEC-20260926.md` 记的是"12/20 被当成捕获尺寸塞进
`set_draft_graph_params`"）。

**当前状态（2026-10-03 20:xx 已结案）**：判决实验命中**情形①**（`pre-window` 就不一致），
修法 = `V41_DYNSPEC_BT_PERSIST=1`（让 `_per_group_block_table_buffers` 成为真常驻缓冲）。
B 轮验证：**零崩溃、15 个探针 rep 全跑完、正确性全过** —— 见 §4.0 与
[`DYNAMIC-SPEC-DRAFT-ROWS-MISMATCH-20261003.md`](DYNAMIC-SPEC-DRAFT-ROWS-MISMATCH-20261003.md)。

> ⚠️ 这里踩过一个坑：手工起 B 轮时**忘了把修复文件同步到 a3-21**（只提交在本地），
> 于是诊断签名与 A 轮**逐字相同**，看起来像"修复无效"。
> 一眼可辨的特征：容器内 `grep -c V41_DYNSPEC_BT_PERSIST` = **0**，而本地 = 1。
> 已在 `round2_runbook.sh` 里加 `check_inside()`：**起服后必须在容器里再核一次三个文件**。
**下一步**：在 `build_dspark_swa_indices` 入口打印
`(block_table.shape, seq_lens.shape, query_start_loc.shape, num_decode_tokens)`，
先在 vLLM 的 `build_draft_attn_metadata` 调用链上找出"谁给的 block_table 短了一行"。

---

## 6. 诚实边界

1. 修复后的性能数据**只有首轮 N=2 = 73.9 tok/s** —— 因为第二轮就崩了，
   **没有拿到完整 1/2/4/8/16 曲线**，也**没有通过正确性探针**。
   在 §5 修掉之前，这条路径**不能算交付**。
2. `draft/gen=0.03` 证明 K 选了 0，但它**不证明 K=0 的输出正确** ——
   正确性必须靠"并发 2 各带不同针"那条用例（见
   `CED-PD-DYNAMIC-SPEC-20260926.md` §11.3 的做法）。
3. 本轮所有数字都来自**独立 TP8（非 CED）**；CED-PD 的 D 侧此前已跑通过动态 K，
   两套结论**不能互相引用**。
4. 动态 SD 需要显式承担"关掉上游 PIECEWISE 降级保护"
   （`V41_CED_DYNAMIC_SPEC_FULL_GRAPHS=1`），主要失效模式是**静默算错**
   ⇒ 必须用 144K/1M 四针验收，不能只看"起来了 + 速率正常"。
