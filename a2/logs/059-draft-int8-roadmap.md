# 059 — ★★ 路线级发现：**②a（draft INT8）在两个档上都优于 ②c，且档 D 达 ×1.9126 ≥ 原始的 ×1.84 目标**

> 2026-09-22 13:4x CST。执行：**主代理**（纯算术 + 读源码，**不占卡**）。
> 起因：横向对账 `050`/`051`/`054` 的容量数字时，顺手把"②a vs ②c"放进同一张模型里算了一遍。
> 标记：【实测】/【算术】/【推断】/【未确认】。

---

## 0. 一句话

**`050` 把 ②c（draft BF16 + block 64）列为序 1、把 ②a（draft INT8）列为序 2，理由是 ②a 被
"图捕获期 `.item()`"挡住 —— 而那个阻塞正是 `049` 修掉的那一个。**
用已验证的零参数模型算：**②a 与 ②c 的 Σ（页容量）完全相同，但 ②a 保持 block=128 ⇒ BPR 更小**
⇒ **②a 严格优于 ②c（+5.2% token）**，且 **档 D = `817,898` = ×1.9126 ≥ 原始目标 ×1.84**。
⇒ ★ **②a 应当升为序 1**（`D_draftINT8` 已在 c1 上开始验）。

---

## 1. 算术（【算术】，全部由 `050 §1.6` 的零参数模型给出）

模型（`050`，9 个实测点逐字命中；`C2_draft64` 的 `c2_model.py` 是它的可复跑实现）：
```
Σslot_pages = 3 × max(kv+index(ratio2), state, swa, draft) + max(kv+index(ratio1), swa)
BPR         = cdiv(max_len, 128) + 1 + 10 × P(128) + P(draft_block)      ← draft 关时末项 = 0
P(b)        = cdiv(min(window−1 + max_in_flight, max_len), b) + 1
num_blocks  = avail_bytes // Σslot_pages − 1
tokens      = int(num_blocks / BPR × max_len)
```
8 卡口径（`avail = 4 GiB`，`max_len = 133120`）：

| 路线 | draft 页 | block | **Σslot_pages** | **BPR** | **tokens** | vs 档 B |
|---|---:|---:|---:|---:|---:|---:|
| 档 B（基线） | — | — | 540,928 | 2471 | 427,643 | ×1.0000 |
| 档 D（基线） | 131,072 | 128 | 476,416 | 2471 | 485,610 | ×1.1356 |
| **②c** draft BF16 | 65,536 | **64** | **282,880** | **2600** | 777,318 | ×1.8177 |
| ★ **②a** draft INT8 | **66,560** | **128** | **282,880** | **2471** | ★ **817,898** | ★ **×1.9126** |

★ **关键一行**：两者的 `Σ` **完全相同（282,880）** —— 因为档 D 的 slots 0–2 由
`max(kv2 41,600, state 65,536, swa 66,560, draft)` 决定，②a 的 draft 页 **66,560** 与 ②c 的 **65,536**
**都顶不过/顶平 `swa=66,560`** ⇒ 页容量落点一样。
**但 BPR 不同**：②a **保持 block=128** ⇒ `P_draft = 130`；②c 的 block=64 ⇒ `P_draft = 259`
（窗口 128 跨 2 块）⇒ BPR 2471 vs 2600。
⇒ ★★ **②a 用"更小的 BPR"赢，与页容量无关** —— 这正是 `051` 自己标出来的那个副作用，②a 天然没有。

档 C 同理（Σ 由 `max(kv2 73,856, …)` 决定，两者都 = 369,280）：
**②a = 626,488（×1.4650）vs ②c = 595,404（×1.3923）** ⇒ ②a 同样占优。

> ★ 与 `050` 的原始数字对账：②a 档 C `626,419` / 档 D `817,746` —— 与我这里差 **0.01%**（`int()` 取整口径），
> ⇒ **两份独立实现的算术一致**。

---

## 2. ★★ ②a 的"阻塞"已经被 `049` 修掉了（【推断·强】，正在实测）

`050` 给 ②a 的判词是"**不能单独上**"：
> draft 是 **non-causal multi-token decode**，q_len=6 ⇒ 走 `kv8_ori_plane` 的
> **prefill 分支（带 `.item()` 宿主同步）** ⇒ **图捕获期必炸** = `048` 的 `S_graphfix` 阻塞。

而 `049` 改的**就是这个判据**。`a2/publish/kv8-graphsafe/dsa_v41.py:548-553` 的注释逐字写着：
```
[S_graphfix] Capture-safe rebuild for decode-shaped batches, **including speculative
decoding**, where every request carries ``1 + num_spec_tokens`` query rows instead of one.
The old predicate (``query_rows == num_reqs``) only recognised the pure-decode shape, so a
spec-decode batch fell into the eager prefill branch below, whose ``.max().item()`` syncs
the stream and aborts graph capture (Not_Supported(EE1016), logs/048).
```
判据本身（`:500-505`）：
```python
bound = min(max(max_query_len, 0) or query_rows, query_rows); bound = max(1, bound)
if num_prefills > 0 and not _sg_is_capturing():
    return False, None          # 真·eager prefill 才走 legacy 分支
return True, bound              # ★ 其余（含 spec-decode 的 q_len=6）走 capture-safe 上界分支
```
⇒ **②a 面对的那个条件（q_len=6 的 spec-decode 批）现在落进的是上界分支，不是 `.item()` 分支。**

★ 旁证（【实测】，来自 `053` 的单 die 判定）：
`[SG-PPR] native_attention capturing=True num_reqs=32 query_rows=192 num_prefills=0 max_query_len=6 ... rows_bound=6`
—— **q_len=6 的捕获期批确实走上了新分支**。

### 2.1 ★★★ 主代理把**分支路由**读完了：**②a 走的路已经被彻底旁路掉**（源码级）

`a2/publish/kv8-graphsafe/dsa_v41.py:1660-1710`（**发布的 `94aeebb7` 那份**）：
```python
def kv8_ori_plane(..., rows_bound=None):
    if rows_bound is not None:
        # [S_graphfix] Decode-shaped batch (incl. spec decode): hand the host
        # bound to the plain-torch rebuild.
        return _kv8_ori_plane_decode(..., rows_bound=rows_bound)   # ★ 纯 torch，无 D2H
    if query_rows == num_reqs:
        return _kv8_ori_plane_decode(...)                          # 旧 decode 快路
    return _kv8_pf_ori(..., max_q_len=query_rows)                  # ★ 唯一会走 Triton 的分支
```
`rows_bound` 的来源 `_kv8_graph_rows_bound()`（`:500-505`）：
```python
if not _kv8_graph_safe_enabled(): return False, None      # 开关关 ⇒ legacy
bound = min(max_query_len or query_rows, query_rows); bound = max(1, bound)
if num_prefills > 0 and not _sg_is_capturing():
    return False, None                                    # ★ 只有真·eager prefill 才回 legacy
return True, bound                                        # ★ 其余（含 spec-decode q_len=6）走 decode 路
```
⇒ **三条推理**：
1. **主路径**：draft 的 `num_prefills` 在 decode 步必然是 **0** ⇒ 落 `return True, bound`
   ⇒ `rows_bound is not None` ⇒ **纯 torch 的 decode 重建**，`_kv8_pf_ori`（Triton prefill kernel）**不会被调用**；
2. ★ **即使 `num_prefills` 恰好 > 0**（如捕获期 dummy 批），**`_sg_is_capturing()` 为 True**
   ⇒ 条件 `num_prefills > 0 and not capturing` **仍为 False** ⇒ **照样走 decode 路**；
3. **捕获期实测**（`053`，单 die）：`capturing=True ... num_prefills=0 max_query_len=6 rows_bound=6`
   —— **q_len=6 的批确实走上了新分支**。
⇒ ★★ **`050` 那句「图捕获期必炸」是 `048` 时代（`GRAPH_SAFE` 还不存在）的结论，现已不适用。**
⇒ **②a 的风险等级从「可能被挡」下调为「大概率直接能跑」**（【推断·强】）。

⚠️ **仍必须真机验的两条**（推理替代不了实测 —— 本项目已栽过 `038`/`043`/`046` 几次）：
- ★ draft 面的 `metadata.swa` **有没有 `max_query_len`**（没有 ⇒ `bound` 退成 `query_rows=192`，
  仍走新路但**上界偏大**：多占 scratch，**不影响正确性**）；
- ★ **反例臂**：`GRAPH_SAFE=0` 时**必须响亮地炸**（否则说明这条路没被走到，**判据无判别力**）。
⇒ 已派 `D_draftINT8`（c1）做真机臂。

---

## 3. 改动面 —— ★★ 主代理逐行核了源码：**大概率是 2 处，不是 3 处**

（下面是 2026-09-22 13:5x 主代理对 `a2/agents/KV8_swa/shadow/vllm_ascend/core/` 的逐行核对，
该 shadow 就是**档 C 已经在用的那一份**。）

### 3.1 必须改的两处
| # | 位置 | 动作 | 依据 |
|---|---|---|---|
| 1 | `deepseek_v41.py::DeepseekV41DraftSWASpec.__post_init__`（`:106-112`） | **放行 int8 + scale** | 现在硬卡 `dtype != bfloat16 ⇒ raise "Aurora DSpark requires one uncompressed BF16 KV plane"`。★ 紧邻的 `DeepseekV41SWASpec.__post_init__`（`:92-97`）**已给出 int8 的校验模板**，照写即可 |
| 2 | `dspark.py::DeepseekV41DSparkSWACache.get_kv_cache_spec` | 传 `dtype=int8, scale_dim=4, scale_dtype=fp16` | 见 3.2 —— 页大小公式**已经通用化** |

### 3.2 ★★ `kv_cache_interface.py` **不用改**（`050` 归的第 3 处其实早已就位）
`KV8_swa/shadow/.../kv_cache_interface.py:178-196`：
```python
class AscendSlidingWindowMLASpec(SlidingWindowMLASpec):
    # KV8: an INT8 SWA plane carries its per-group dequant scales inside the
    # same page, exactly like ``AscendMLAAttentionSpec`` does for the shared
    # long-KV plane and like the indexer already does for its key cache.
    scale_dim: int = 0
    scale_dtype: torch.dtype = torch.int8

    @property
    def real_page_size_bytes(self) -> int:
        return self.storage_block_size * self.num_kv_heads * (
            self.head_size * get_dtype_size(self.dtype)
            + self.scale_dim * get_dtype_size(self.scale_dtype))
```
⇒ 代进 draft：`128 × 1 × (512×1 + 4×2) = 66,560 B` —— **与 `050` 的算术逐字吻合**
⇒ **这正是任务书里"存储侧复用 `AscendMLAAttentionSpec` 的 `scale_dim` 机制"那句话的落点**。

### 3.3 ★★ `plan_cache_slots` 的 draft 检查 —— 源码级确认**会通过**（余量恰好为 0）
`KV8_swa/shadow/.../deepseek_v41.py:240-248`：
```python
if (draft_spec.block_size != swa_spec.block_size          # 128 == 128 ✅
    or draft_spec.head_size != swa_spec.head_size          # 512 == 512 ✅
    or draft_spec.sliding_window != swa_spec.sliding_window # 128 == 128 ✅
    or sum(_cache_plane_sizes(draft_spec)) > capacity):     # 66,560 <= 66,560 ✅（**恰好相等**）
    raise ValueError("Aurora DSpark geometry must match target SWA and fit its existing slot")
```
⇒ 四个条件全过，**第 4 条的余量恰好为 0** —— 这是 `050 §1.2` 那句
"②a 的 draft 页 66,560 与 ②c 的 65,536 **都顶不过/顶平 `swa=66,560`**"的**源码级确认**。

⚠️ **仍是【推断】**：以上是读源码得出的；**必须真机确认它真的不 raise**（若 raise，把报错原文贴回来）。
⇒ 仍要按 `051 §3` 做**三臂对称自检**（`upstream` / `draftaware` / `patched`）——
只有那样才能证明"没改别的地方也刚好能跑"，而不是"我改对了"。

### 3.4 两档都要试
`KV8_swa` 那份是**档 C 的 shadow**；tier D 要用 `pkg-kv8pf`（带 long-KV int8）。
★ **②a 在档 C 上也有收益**（626,488 = **×1.4650**）⇒ 别只做 D。

---

## 3bis （原 `050` 给的 3 处，保留以便对照）

| # | 位置 | 动作 |
|---|---|---|
| 1 | `DeepseekV41DraftSWASpec.__post_init__` | **放行 int8 + scale**（现在硬要求 BF16 ⇒ raise） |
| 2 | `dspark.py::DeepseekV41DSparkSWACache.get_kv_cache_spec` | 给 `dtype=int8, scale_dim=4, scale_dtype=float16` |
| 3 | `plan_cache_slots` 的 draft 检查 | `050` 说**已满足**（draft 66,560 ≤ slot 66,560）—— **仍需实测确认** |

★ 读路径**理论上自动复用** KV8 SWA（`reshape_cache` 用 `getattr(spec,'scale_dim',0)` 决定是否返回 tuple）
—— **要用探针证实，不能只读代码**。

---

## 4. ★★ ②a 的真正风险：**它动的是"投机解码"本身**（用户的硬约束）

> ★ **2026-09-22 13:5x 补充（从 `T_draftceiling` 的实测推出来的一个尖锐推论）**：
> `054` 实测 ②c（draft 仍 BF16，只改 block）的两臂 **draft 提案序列 4367/4367 逐条相同** ——
> 那说明 ②c **没有动投机路径**。而 ②a **恰恰就是动投机路径**（换 draft 的 KV dtype）。
> ⇒ ★★ **不要把"②c 保投机"的经验外推到 ②a** —— 两者的风险性质不同，**必须各自实测**。
> ⇒ 这也是为什么 §5 的路线顺序里，②a 的**前置**必须包含 Q3（接受率 A/B），而不只是 Q1/Q2。

用户 2026-09-22 明确：**必须保留投机解码**。②a 与 ②c 在这一点上**风险等级不同**：

| | ②a（draft INT8） | ②c（draft BF16 + block 64） |
|---|---|---|
| draft 的 KV dtype | **改了**（BF16 → INT8+scale） | **没改**（仍 BF16） |
| 对输出分布 | 理论上**不改**（只影响 draft 的**提案**，target 仍用 BF16 全精度验证） | 不改 |
| **对接受率** | ⚠️ **可能有影响** ⇒ **必须做接受率 A/B** | 理论上无影响；`054` 已实测"提案序列逐条相同" |
| 副作用 | 无（block 不变） | ★ DRAM 池 **+5.0%**（`sw_chunks` 1→2）；且 BPR 变大 |

⇒ ★ **判据**：`SpecDecoding` 四项（`MeanAccLen` / `AvgDraftAcc` / `per-pos` / `Accepted|Drafted`）**相对同几何 BF16 基线不许回退**，
且 **`max_tokens ≥ 64`**（`max_tokens=1` 只有 ~15 个 drafted token，**无统计功效**；`S_graphfix` 已在 `049` 里踩过并写死"不用绝对值"）。

---


### 4.1 ★★★ `T_draftceiling` 抓到一个**比「样本小」更致命的混淆**（2026-09-22 14:1x）

> **`Drafted` 的分母不同**（档 C 的 `max_tokens=1` 臂是 **10 token**，档 D 是 **15 token**）
> ⇒ 那不是「同一批请求换个配置」，**两次测量的请求数/步数本来就不一样**
> ⇒ 用 `Accepted/Drafted` 直接比 C 与 D 是**跨口径比较** —— 比「样本小」更糟。

它给的**二项 CI 估算**（【推断】）：`0/15` 的 95% CI 上界 ≈ **22%**，`1/10` ≈ **26%** ⇒ **大幅重叠**
⇒ **这两条读数不能支持「档 D 降低接受率」**。

★ **正确做法（它已排进链里当第 3 条臂）**：同几何、同 md5、同 workload（`8×4096 → max_tokens 64`）、
**只有被考察的那个变量不同**：
```
① t-dc2-b-D  draft128 / long-KV int8（= 档 D 现状）    ← 回答「档 D 的接受率」
② t-dc2-c-D  draft64  / long-KV int8（= 档 D + ②c）   ← ②c 的交付判据
③ t-dc2-a-C  draft128 / long-KV **保持 BF16**（= 档 C）← ★ 回答「C vs D 的差是真的还是 max_tokens=1 的假象」
```
★★ **为什么 ③ 值这 25 分钟**：C 与 D 的**唯一差别就是 long-KV 平面**。
若三条在同几何下**逐字相同** ⇒ 「档 D 掉接受率」**彻底出局**；
若**真的不同** ⇒ 我们就找到了一个**必须报给用户的新副作用**（int8 long-KV 影响投机）。**两种结果都有价值。**

★ 另外一条源码级校准（`S_graphfix` 核的）：`MeanAccLen = 1 + accepted / num_drafts`
⇒ **`MeanAccLen` 的分母是「draft 步数」而不是 token 数**。
⇒ 同几何下三条臂的 `num_drafts` 应当可比；**若某条臂的 `num_drafts` 明显不同，那本身就先报警**（workload 没跑成一样的）。

## 4bis. ★ 2026-09-22 14:3x：②a 的**第一次真机尝试**（`D_draftINT8`，c1）—— 失败在一行 `import`

**结果**：`ddi-d-i8-eager` 与 `ddi-d-i8-graph` 两条臂**都在 61 s 内 rc=11**（服务没起来）。
**确切失败点**（主代理从 `out/ddi-d-i8-*.crash.txt` 读的原始栈）：
```
NameError: name 'os' is not defined. Did you forget to import 'os'
  File ".../pkg-ddi/shadow/vllm_ascend/models/deepseek_v41/dspark.py", line 65, in get_kv_cache_spec
```
**根因**：那份 `dspark.py` 的**文件头没有 `import os`**，而补丁在 `get_kv_cache_spec` 内部用了
`os.getpid()` / `os.environ.get("VLLM_V41_DRAFT_INT8", ...)`（探针代码）。**一行修复。**

★★ **定性：这不是「②a 被机制挡住」，而是「补丁的构造缺陷」** ——
探针代码引入了新依赖却没加 import。⇒ **不计入 ②a 的机制判决**，修完重跑。

★ **同时拿到一条源码级的正面结论**（`D_draftINT8` 自己查的，写在它的注释里）：
> 父类的 dtype 来自 **`get_dsv4_attn_kv_dtype()`**（**A2/A3 恒为 BF16，非全局 `KV_DTYPE` 开关**）
> ⇒ 用独立 env 门控，**不会**顺带改动 target 的 40 个 SWA 层。
⇒ ★ **这比主代理上一轮的推断更安全**：不存在「改 draft 会顺带改 target」的风险
⇒ **②a 的改动面确实是「只动 draft 一处」**（与 §3 的逐行核对一致）。

⚠️ **一条流程提醒**：修 import 后**必须重建包并确认新包 md5 变了** ——
`048` 记过一个坑：**幂等补丁「已打过就跳过」会让修好的块装不进去**。

## 4ter. ★★★ 2026-09-22 14:5x：【实测】②a **图捕获成功**、**draft 页 66,560**、**容量逐字命中模型**

`D_draftINT8`（c1）修掉那行 `import os` 后重跑，**第一批数就出来了**（tier D，tiny 几何）：

```
### arm=ddi-d-i8-graph tier=D graph=1 draft_int8=1 graph_safe=1 ready_s=85 died=0
capture_finished=1  ee1016=0  not_supported=0  capture_failed=0        ← ★★★ 图捕获成功
[DDI-2a-PROBE v1] draft-spec ... env='1' -> block=128 storage_block=128
                  dtype=torch.int8  scale_dim=4  page_bytes=66560   ← ★ 页大小 = 66,560（int8+scale）
GPU KV cache size: 39,846 tokens                                     ← ★ 容量
[R8-SLOTS] slot=0 kv=33280 index=8320 aliases_max=66560 draft=66560 capacity=66560
[R8-SLOTS] slot=3 kv=66560 index=16640 aliases_max=66560 draft=0   capacity=83200
```

### ① ★★★ Q1（图捕获）**通过** —— 主代理上一轮的源码级推断被真机证实
`049` 那条「只改判据、让 spec-decode 批走 capture-safe 上界分支」的修法，
**确实覆盖了 draft 自己的那次 `kv8_ori_plane` 调用** ⇒ `050` 列的「②a 图捕获期必炸」**已不成立**。

### ② ★★★ Q2（容量）**逐字命中** —— 零参数模型现在 **12/12**

| 臂（tiny 几何，max_len=8192 / avail=1GiB） | Σslot_pages | BPR | 模型 | 实测 |
|---|---:|---:|---:|---:|
| 档 D 基线（draft BF16/128） | 476,416 | 780 | 23,651 | **23,651** ✅ |
| ②c（draft BF16/**64**） | 282,880 | 844 | 36,825 | **36,825** ✅ |
| ★ **②a（draft INT8/128）** | **282,880** | **780** | **39,846** | ★ **39,846** ✅ |

★ **②a 的 Σ 与 ②c 完全相同（282,880）、但 BPR 更小（780 < 844）** ——
这正是 §1 那句「**②a 用更小的 BPR 赢、与页容量无关**」在**第三种几何**上的独立证实。
⇒ ★★ **②a 在 tiny 上（39,846）也严格优于 ②c（36,825）**。

### ③ 8 卡预测（按已验证的模型外推）
```
档 B 基线       Σ=540,928 BPR=2471 ⇒ 427,643   ×1.0000
档 C 基线       Σ=540,928 BPR=2471 ⇒ 427,643   ×1.0000
档 D 基线       Σ=476,416 BPR=2471 ⇒ 485,610   ×1.1355
②c (BF16/64)  Σ=282,880 BPR=2600 ⇒ 777,318   ×1.8177
★②a (int8/128) Σ=282,880 BPR=2471 ⇒ 817,898   ★ ×1.9126   ← ★ 超过原始 ×1.84 目标
```

### ④ ⏳ 还差什么：**Q3（接受率）** —— 这才是 ②a 真正的风险面
`054` 已证 ②c 的 draft 提案序列 **4367/4367 逐条相同**（它没动投机）；
而 **②a 动的就是投机**（换 draft 的 KV dtype）⇒ **必须做 `SpecDecoding` 四项的 A/B**，
且**必须 `max_tokens ≥ 64`**（`max_tokens=1` 只有 ~15 个 drafted token，无统计功效）。

## 5. 建议的路线顺序（**取代 `050`/`051` 的"②c 序 1"**）

```
序 1  ★ ②a（draft INT8，block 128）—— 期望 ×1.9126（C）/ ×1.9126（D），**超过原始 ×1.84 目标**
       前置：单 die 验 Q1（图捕获）/ Q2（容量）/ Q3（接受率）
序 2     ②c（draft BF16，block 64）—— ×1.3923（C）/ ×1.8177（D）；**②a 失败时的回落**
       现状：单 die 三问已过（054），8 卡端到端在 c0 排队
```
★ **②c 的 8 卡臂不必取消** —— 它是 ②a 失败时的保险，且两者**共用同一套判据与 harness**（可互相校准）。
但**交付选型要等 ②a 的 Q1/Q3 出来再定**。

---

## 5bis. ★★★ 判决（2026-09-22 15:1x 更新）：**②a 已实测失败 ⇒ 交付推荐仍是 ②c**

`D_draftINT8` 的 Q1/Q2 已过（§4ter），**但 ②a 现在仍不能进交付**，原因只有一条：

| 判据 | ②c | ★ ②a |
|---|---|---|
| Q1 图捕获 | ✅【实测】`054`（b64/b128 捕获 23 s / 9 s，`EE1016=0`） | ✅【实测】`§4ter`（`capture_finished=1`，`EE1016=0`） |
| Q2 容量 | ✅【实测】逐字命中模型 | ✅【实测】逐字命中模型（**12/12**） |
| **Q3 接受率（用户硬约束）** | ✅【实测】**提案序列 4367/4367 逐条相同**（它没动投机） | ⏳ **未测** —— ②a **改了 draft 的 KV dtype**，这是唯一可能影响接受率的地方 |
| 8 卡端到端 | ⏳ c0 排队中 | ⏳ 未排 |

> ★★★ **判决（`056`，2026-09-22 15:1x）：②a 的 Q3 实测失败 —— 它不可用。**
> ```
> RuntimeError: The previous device metadata submission has not been released
>   @ worker/device_metadata.py:74（触发形状 num_scheduled_tokens=6 + 5 个 spec token）
> ②a 臂：16/16 请求失败、0 条 SpecDecoding 读数
> 对照臂（同包同参数，唯一变量 DRAFT_INT8=0）：ok=16/16 / ttft.p50=738.1 ms / sha 0ebccb55b30c…（与 054 四臂逐字相同）
> ```
> ⇒ ★★ **②a 特有，不是 harness**。
> ★ **【推断】病灶**：draft 面的 attention 实现在 `dsa_v1.py::AscendDSAImpl`，
> 而 KV8 的量化存取（`kv8_swa_store` / `kv8_ori_plane`）**只写在 `dsa_v41.py`**
> （主代理独立核实：`dsa_v1.py` 命中 **0**、`dsa_v41.py` 命中 **3 + 7**）。
> ⇒ ★ **②a 不是「物理不可行」**，而是**要先把 KV8 的量化存取移植进 `dsa_v1.py`** ——
> 一处**新的、更大的改动**，**不在本轮范围** ⇒ **②a 退回研究项**。

⇒ ★★ **规则（判决后）**：**交付推荐 = ②c**（序 1）；**②a 需先完成量化存取移植**才值得再评估。
★ 理由：②c 的 Q3 是**实测过**的（而且是最强形态 —— 提案序列逐条相同），
②a 的 Q3 是**空白**，而「draft KV 换 dtype 会不会影响接受率」正是本路线**唯一的实质风险**。

★ **可判伪的预测**（若 Q3 出来与它不符，说明我上面的推理链有错）：
1. ②a 的 `MeanAccLen` / `AvgDraftAcc` / `per-pos` 应当与同几何 BF16 基线**在噪声内相同**
   （量化只影响 draft 的**提案质量**，而 draft 是被 target 全精度验证的 ⇒ 理论上只该**轻微**降接受率、不该改输出分布）；
2. 若 ②a **显著降接受率** ⇒ 它的价值要从「×1.9126」里**扣掉那部分吞吐损失**，甚至可能不如 ②c（后者零代价）；
3. ★ 输出 sha 应当**不变**（量化 draft 不应改变 target 的验证结果）。

## 6. 诚实边界
## 6. 诚实边界

1. ★ **本日志全部是【算术】+【推断】**，**没有任何新的实测**；
2. ②a 的**容量预测（817,898 / 626,488）尚未在真机上验过**；
3. ②a 的**图捕获是否真的解除阻塞【未确认】**（`D_draftINT8` 在验；若失败则 ②a 仍被挡、②c 仍是唯一路线）；
4. ②a 的**接受率影响【未确认】** —— 这是它相对 ②c 唯一的实质风险；
5. `050` 说"`plan_cache_slots` 的 draft 检查已满足"**未经我复核**。

## 7. 交付

| 件 | 位置 |
|---|---|
| 本日志（路线级发现 + 算术） | `a2/logs/059-20260922-draft-int8-roadmap.md` |
| 零参数模型的可复跑实现（9 点逐字命中） | `a2/agents/C2_draft64/scripts/c2_model.py`（`C2_draft64` 交付） |
| 正在实测 ②a 的子代理 | `D_draftINT8`（c1）⇒ 将产出 `logs/056` |
