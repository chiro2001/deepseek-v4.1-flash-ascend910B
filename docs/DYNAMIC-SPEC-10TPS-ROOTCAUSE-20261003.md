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

| 并发 | 修复前 | **修复后** |
|---:|---:|---:|
| 1（K=7） | 88.7–95.6 | 82.5–94.9（不变） |
| **2（K=0）** | **9.9–12.0** | **73.9**（首轮） |

`draft/gen` 仍是 0.03 ⇒ **K=0 依然被正确选中**，改善来自"K=0 的步终于进图了"。

**与静态 SPEC=0 的差距**：静态 92.6 vs 动态 K=0 73.9（−20%）。
【推断】差额来自"动态 SD 下每步要多做一次 K 判定与两套图键的维护"，
以及 k=0 的桶只有 `1,2,3,4,8,12,16,20,24,32`（**桶更稀 ⇒ padding 更多**）。

---

## 5. ★ 修复后新暴露的第二个缺陷（**待修，已定位到函数**）

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

**当前状态**：首轮 N=2 能跑完（73.9 tok/s），第二轮崩 ⇒ 不是每步必崩，
更像"某个形状/某个步（batch 变小、或某请求结束时）才触发"。
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
