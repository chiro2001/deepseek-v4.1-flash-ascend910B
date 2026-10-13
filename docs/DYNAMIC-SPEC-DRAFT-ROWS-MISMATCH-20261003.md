# 动态 K 第二个缺陷：草稿元数据行数不匹配（2026-10-03，**进行中**）

> 承接 [`DYNAMIC-SPEC-10TPS-ROOTCAUSE-20261003.md`](DYNAMIC-SPEC-10TPS-ROOTCAUSE-20261003.md)。
> 属性别名修好之后（N=2 从 ~10 → **73.9 tok/s**），ql=1 的步**真的进了图**，
> 于是把下一个缺陷暴露出来：**首轮跑完、第二轮把引擎打挂**。
> 本文记录**已确认的事实**、**已排除的可能**、以及**待运行的判决实验**。
> 结论标 **【实测】/【推断】/【未确认】**。

---

## 1. 现象与栈【实测】

```
Worker_TP*.sample_tokens
 → model_runner_v1.py:2620 propose_draft_token_ids
 → llm_base_proposer.py:1971 propose_draft_token_ids
 → llm_base_proposer.py:1656 _propose
 → llm_base_proposer.py:3382 build_draft_attn_metadata
 → dsa_v1.py:1552 run_attention_metadata → :1516 build_attention_metadata
 → dsa_v1.py:1477 build_dspark_swa → :618
     block_ids = torch.gather(block_table, 1, safe_nums)

RuntimeError: aclnnGather failed, error code 161002
AclNN_Parameter_Error(EZ1001): Size does not match at dimension 0,
    expected index shape 2 smaller than self shape 1
→ EngineDeadError → HTTP 500
```

**读法**：`safe_nums`（由 `seq_lens`/`query_start_loc` 派生的 `[req_count, index_width]`）
有 **2 行**，而 `block_table` 只有 **1 行**。`torch.gather(dim=1)` 要求除 `dim` 外**所有维度相等**
⇒ 两侧的 `num_reqs` 视图不一致。

**触发条件**【实测】：不是每步必崩 —— N=2 的 **r0 跑完**（73.9 tok/s 那一轮），
**r1 崩**。同一实例此前单独跑 N=2 连续 90 秒（旧代码、全程 eager）**不崩**。
⇒ **是"ql=1 真进图"与"某个特定的 batch 形状变化"共同触发的**。

---

## 2. 已排除的可能【实测/推断】

| 假设 | 判据 | 结论 |
|---|---|---|
| 属性别名没生效 | `draft/gen=0.03`（K 确实选了 0）+ `patch_cudagraph.py` 命中数 2→7 | **已生效** |
| 捕获键缺失 | 起服日志 `19/19`，与"ql=8 捕 9 个 + ql=1 捕 10 个"的推算**逐字吻合** | **键都在** |
| `(2,2,uniform=True)` 这张图不存在 | 同上（2 ∈ ql=1 的桶集合，且 `num_reqs = 2//1 = 2` 合法） | **存在** |
| 缺图告警被日志级别吞掉 | 已加 `dispatch()` 守卫，本次运行**没有**打印 `★` 告警 | **这次命中了图** |

### 2.1 ★ 静态推导出的根因（比探针更早收敛，探针用于**确认**）

关键在 `dsa_v1.py` 这两行（build 期，同一个 `num_reqs`）：

```python
1407:  self.seq_lens   = common_attn_metadata.seq_lens[:num_reqs]
1408:  self.block_table = common_attn_metadata.block_table_tensor[:num_reqs]
```

**两行的 `[:num_reqs]` 截断效果并不相同**：

| 张量 | 自身行数 | 截断后行数 |
|---|---:|---:|
| `seq_lens` | ≥ num_reqs（常驻缓冲） | **num_reqs** |
| `block_table_tensor` | 若**只有 1 行** | **1**（截断取不到更多） |

⇒ 两者行数不一致，run 期 `build_dspark_swa_indices` 拿 2 行的 `safe_nums`
（由 `seq_lens` 派生）去 gather 1 行的 `block_table` ⇒ 正是那句报错。

**为什么"广播补齐"没有兜住它**【推断，与代码一致】：
唯一会把行数补齐的是 `SlidingWindowAdapter.apply()`（`out[:num_reqs].copy_(gathered)`，
见 §3）。而它只在 `self.draft_window_size is not None` 时才会被构造
（`llm_base_proposer.py:788`：`if self.draft_window_size is not None and self.method != "mtp"`），
**我们的动态 K 启动命令没有设 `draft_window_size`** ⇒ `self.sliding_window is None`
⇒ `build_draft_attn_metadata` 里 `if self.sliding_window is not None:` 整段被跳过
⇒ 短表**原封不动**进了 `dsa_v1`。

**短表的来源**（`block_table_tensor` 为什么只有 1 行）：
`_per_group_block_table_buffers[gid]` 在 `dspark_proposer.set_per_group_attn_metadata`
里被赋成 runner 传来的 `blk_table.get_device_tensor()[:num_reqs_padded]`
—— 一个按**目标侧** padded 行数切出来的视图；而草稿侧在 `_propose` 里会走自己的
`_pad_query_start_loc_for_fia`，其 mixed-batch 分支**补一个 dummy 请求并 +1**
（见 `llm_base_proposer` 顶部注释）⇒ 草稿侧要的行数比目标侧多 1。

**⇒ 修法**：把 `_per_group_block_table_buffers` 做成**真常驻缓冲**
（`max_num_reqs + 2` 行），每次原地拷入目标侧的行、多余行清零。
已实现并**默认关闭**（env `V41_DYNSPEC_BT_PERSIST=1`）：
`patches/files/draft/dspark_proposer.py` 的 `set_per_group_attn_metadata`。
多余行对应草稿侧补出来的 dummy 请求，`seq_lens=0` ⇒ 窗口掩码把它们的 slot 全置 -1，
读哪一块都不影响结果；且传给 Triton kernel 的 `block_table_stride` 仍是列数，语义不变。

---

## 3. ★ 一个已确认的**静默正确性隐患**（独立于上面那个崩溃）

`vllm_ascend/spec_decode/utils.py::SlidingWindowAdapter`：

```python
def compute_sliding_window_block_table(self, common_attn_metadata, out) -> None:
    num_reqs = common_attn_metadata.seq_lens.shape[0]      # ← 用 seq_lens 的行数
    ...
    gathered = torch.gather(self.full_block_table, 1, src_cols_clamped)   # 行数 = block_table 的行数
    ...
    out[:num_reqs].copy_(gathered * valid_mask.to(gathered.dtype))        # ← 行数不一致时**广播**

def apply(self, common_attn_metadata) -> None:
    self.full_block_table = common_attn_metadata.block_table_tensor
    num_reqs = common_attn_metadata.seq_lens.shape[0]
    ...
    common_attn_metadata.block_table_tensor = self._block_table_clone[:num_reqs]
```

**`copy_` 会广播**：当 `gathered` 只有 1 行、`out[:num_reqs]` 有 2 行时，
**不报错**，而是把第 0 行复制到所有行。

**广播语义已实测**（本机 numpy，语义与 `torch.copy_` 相同）：

```
out = zeros((4,3)); out[:2] = [[7,8,9]]
=> out[:2] = [[7,8,9],[7,8,9]]      # 1 行被静默广播成 2 行
```

**危害**：若 `gathered` 行数少于 `seq_lens` 行数，草稿的 block table 会**静默变成同一份**
（所有请求都去读第 0 个请求的 block），**属于"静默算错"而不是崩溃**。
这与本仓 `AGENTS.md` 反复强调的那类失效同族。

**处置（候选补丁，待运行时证据确认后再上）**：在 `compute_sliding_window_block_table`
与 `apply` 里把"行数必须相等"写成**显式断言**（fail loudly），不要再依赖广播。
注意：`spec_decode/utils.py` **不在我们现有的 overlay 挂载清单里**，
要打它需要先在 `serve_a2.sh` 的挂载列表里加一项（改动很小）。

---

## 4. 判决实验：**已运行，结论为情形 ①**【实测 2026-10-03 19:1x】

服务 `dsv41-dynfix`（`SP_SCHEDULE='1,1,7;2,32,0'`，候选修复**关**）在 N=1→2 探针下复现，
`[DYNSPEC-DIAG]` 打出决定性的一行：

```
[DYNSPEC-DIAG] pre-window gid=12 行数不一致:
  num_reqs=2
  block_table=(1, 8192)(rows=1)        ← 只有 1 行
  seq_lens=(2,)(rows=2)                ← 2 行
  qsl=(3,)
  per_group_buffer=(1, 8192)(rows=1)   ← per-group 缓冲本身就只有 1 行
```

随后 `dsa_v1` 的断言接住（把原本含糊的 aclnn dim-0 报错换成带全部形状的可读错误）：

```
RuntimeError: [DYNSPEC-DIAG] build_dspark_swa_indices 行数不一致:
  block_table=(1, 8192) safe_nums=(2, 256) seq_lens=(2,)
  query_start_loc=(3,) num_decode_tokens=7 index_width=256
```

### 4.1 判读（三种情形互斥，命中的是 ①）

| 情形 | 判据 | 本轮结果 |
|---|---|---|
| **① `pre-window` 就不一致** ⇒ per-group block table 切片太短 | `pre-window` 行警告 | **★ 命中** |
| ② 只有 `post-window` 不一致 ⇒ 问题在 `SlidingWindowAdapter` | 只在 `post-window` 出现 | 未出现（本轮连 `sliding_window` 都是 None） |
| ③ 两者都一致、只有 `dsa_v1` 报错 ⇒ build/run 期 `num_reqs` 偏移 | `pre`/`post` 都不警告 | 未出现 |

⇒ **候选修复（`V41_DYNSPEC_BT_PERSIST=1`：让 `_per_group_block_table_buffers` 成为真常驻缓冲）
正是对症的那一个**，不需要再去改 `spec_decode/utils.py`。

### 4.2 复现用的探针

```bash
# 追加一段 N=1 与 N=2 的 decode（N=1 走 K=7、N≥2 走 K=0）
python3 ~/tmp/dynprobe.py 19210 1,2 7 3 256
# 崩溃/降级证据
grep -a "DYNSPEC-DIAG" $(ls -t ~/cedpd-repo/results/dynfix2_*/serve.log | head -1)
```

> **B 轮（开 `V41_DYNSPEC_BT_PERSIST=1`）随后执行**，判据：**无** `DYNSPEC-DIAG`、
> **无** `EngineDeadError`，且 `dynprobe` 的 1/2/4/8/16 全部跑完。

---

## 4.3 原始（A 轮之前的）待运行说明，保留备查

已在两个**已挂载**的文件里插好探针，**正常路径零输出**（只在行数不一致时打印）：

| 文件 | 位置 | 打印内容 |
|---|---|---|
| `patches/files/draft/llm_base_proposer.py` | `_v41_diag_bt()`，在 dspark 分支切完 block_table 后（`pre-window`）与 `sliding_window.apply` 之后（`post-window`）各调一次 | `num_reqs` / `block_table` 形状 / `seq_lens` 形状 / `query_start_loc` 形状 / **per-group 缓冲区的总行数** |
| `patches/files/draft/dsa_v1.py` | `build_dspark_swa_indices` 里 `gather` **之前** | 行数不一致时抛出带全部形状的 `RuntimeError`（替代 aclnn 那句含糊的 dim-0 报错） |

**预期能一次分清三种情形**：

1. `pre-window` 就不一致（`block_table` 行数 < `num_reqs`）⇒ **per-group 缓冲区的切片短了**，
   要查 `_per_group_block_table_buffers` 的分配与 `num_reqs`/`num_reqs_padded` 的关系；
2. 只有 `post-window` 不一致 ⇒ 问题在 `SlidingWindowAdapter`（§3 那条）；
3. 两者都一致、崩在 dsa_v1 ⇒ 是 `dsa_v1` 自己的 `self.block_table` 与 `common_attn_metadata`
   **不是同一个对象**（例如它保留了上一次的引用）。

复现命令（VPN 恢复后跑）：

```bash
# 服务已按下面的配置起好（run dynfix2），直接打探针
python3 ~/tmp/dynprobe.py 19210 1,2,4,8,16 7 3 256
# 然后
grep -a "DYNSPEC-DIAG" $(ls -t ~/cedpd-repo/results/dynfix2_*/serve.log | head -1)
```

---

## 5. 当前**未完成**的验收项（诚实清单）

| 目标里的要求 | 状态 |
|---|---|
| N=1 走 K=7、≥105 tok/s | ✅ 实测 82.5–94.9（**未达 105**，见下） |
| N≥12 关 spec、N=16 ≥430 tok/s | ❌ **未测**（服务在第二轮就崩了） |
| K=7 与 K=0 两条路径的正确性探针（144K/1M 四针 + 并发 2 各带不同针 + `regress2.py`） | ❌ **未跑** |
| 完整 1/2/4/8/16 曲线 | ❌ **未拿到**（只有一个 N=2 数据点） |

> ⚠️ N=1 的 82.5–94.9 也低于目标 105：注意这**不是回归** ——
> 目标值 105 是按"静态 K=5 交付档"写的（实测 109.0–109.5），
> 而本轮为了对齐历史表用的是 **K=7**（静态 K=7 的 N=1 是 109.8）。
> K=7 动态档的 N=1 只有 ~90 ⇒ 要么改表用 `1,1,5;2,32,0`（K=5），
> 要么接受 N=1 的收益只在"降 K 之前"体现。**待定，先记录。**

---

## 6. 诚实边界

1. §3 的广播隐患是**代码级确认**，但"它在本次崩溃里是否真的被触发"**尚未确认**
   —— 需要 §4 的探针输出。
2. 动态 SD 需要显式承担"关掉上游 PIECEWISE 降级保护"
   （`V41_CED_DYNAMIC_SPEC_FULL_GRAPHS=1`），主要失效模式是**静默算错**
   ⇒ 必须用 144K/1M 四针验收，不能只看"起来了 + 速率正常"。
3. 本轮全部数字来自**独立 TP8（非 CED）**，与 CED-PD 的 D 侧结论**不能互相引用**。
