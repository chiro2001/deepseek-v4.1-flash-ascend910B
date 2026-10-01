# ★★★ DSpark × DCP 并发崩溃根因锁定：第二次纯 ori 的 SMLA 调用（2026-10-01）

> 用户要求：修好推测解码，让它能与 DCP8 共存。
> 本文记录从"以为是上游 DSpark bug"到"锁定在我们 DCP overlay 的一行"的完整过程。

---

## 0. 先纠正一个把我带偏两小时的错误结论

我此前写过"DCP=1 也崩 ⇒ 与 DCP 无关"。**那个对照不纯。**

```python
# vllm_ascend/patch/platform/patch_v41_dcp.py:63
def v41_dcp_active() -> bool:
    return os.environ.get("V41_DCP") == "1" or os.environ.get("V41_DCP_ALLOW_CAPACITY_PROBE") == "1"
```

tiny 夹具**默认就传 `V41_DCP_ALLOW_CAPACITY_PROBE=1`** ⇒ 即使 `DCP=1`，
我们的 DCP 代码路径也**全程激活**。之所以一直没发现，是因为我核对时只看了
`DCP=1` 这个参数、没看这个 env。

重做干净的对照（同夹具、同 model、只改这个 env）：

| 配置 | 并发 1 | 并发 2 | 并发 4 | 并发 8 |
|---|---|---|---|---|
| `dcp_active=False` | ✅ 67.4 | ✅ 30.8 | ✅ 158.3 | ✅ **202.5** |
| `dcp_active=True` | ✅ | ❌ 崩 | — | — |

**⇒ 根因 100% 在我们的 DCP overlay 里。** 这也与用户提供的事实一致：
A2 生产（不挂 overlay，用镜像 baked 文件）的 DSpark 入图 + 并发**完全正常**。

---

## 1. 排除矩阵（全部在 tiny 夹具实测，K=7 DRAFT_GRAPH=1）

| 假设 | 开关 | 并发 2 结果 |
|---|---|---|
| draft 入图 | `DRAFT_GRAPH=0` | ❌ 仍崩 |
| draft 的 attention | `DSPARK_DRAFT_NO_ATTN=1` | ❌ 仍崩 |
| topk buffer 共享 | `DSPARK_NO_TOPK_SHARE=1` | ❌ 仍崩 |
| async scheduling | `--no-async-scheduling` | ❌ 仍崩 |
| attn 持久缓存 | `V41_DCP_NO_ATTN_CACHE=1` | ❌ 仍崩 |
| metadata head 数错配 | +`world_size>1` 条件 | ❌ 仍崩（但**该修法本身是对的**，见 §4） |

---

## 2. ★ 根因：第二次纯 ori 调用的 `seqused_cmp_kv` 非零

`dsa_v41.py` 的 `if dcp_active:` 段（~3544-3737）会做**第二次 SMLA 调用**，
只为拿 `(A, A·O_ori)`（`A = e^{L_ori}`），供跨 rank 合并时减掉多算的 `(dcp−1)` 份 ori：

```python
_neg = torch.full((_rows_n, 1, _topk_n), -1, ...)      # cmp_sparse_indices 全 -1
_ori_out, _ori_lse = torch.ops._C_ascend.npu_sparse_flash_mla(
    q,
    ori_kv=..., cmp_kv=source_cache,
    cmp_sparse_indices=_neg,          # ← 全 -1
    seqused_cmp_kv=_ori_cmp_lens,     # ← **= cmp_seq_lens，非零！**
    ...
    return_softmax_lse=True,
)
```

代码注释声称 `actCmpS2Size = min(bound, CountValid(-1)) = 0` ⇒ cmp 不会被读。
**但实测 batch≥2 时该前提不成立** —— device 侧
`SparseFlashMla_..._1090_mix_aic` 报
`The scalar instruction accesses an invalid GM address`。

代码里本来就有正确语义的开关 `ori_zero_cmp`：

```python
if _perf_flags().get("ori_zero_cmp") == "1" and cmp_seq_lens is not None:
    _ori_cmp_lens = torch.zeros_like(cmp_seq_lens)
```

但它是**文件驱动**的，而 decode 走整图捕获 ⇒ 生产路径上来不及生效
（代码注释 `:652-653` 自己承认过这一点）。

### 修法：把"确定性置零"变成默认

```python
_ori_cmp_lens = cmp_seq_lens
_raw_cmp = os.environ.get("V41_DCP_ORI_RAW_CMP", "0") == "1"   # 退回旧行为用
if (not _raw_cmp and cmp_seq_lens is not None) or \
   (_perf_flags().get("ori_zero_cmp") == "1" and cmp_seq_lens is not None):
    _ori_cmp_lens = torch.zeros_like(cmp_seq_lens)
```

语义上这**更严格**：第二次调用本来只要纯 ori，置零后算子没有任何机会去读 cmp 键。

---

## 3. 实测证据（每一行都是独立起服的一轮）

### 3.1 DCP=1（`dcp_active=True`）

| 轮次 | 改动 | 并发 1 | 并发 2 | 并发 4 | 并发 8 |
|---|---|---|---|---|---|
| 基线 | — | ✅ | ❌ | — | — |
| +head 数修正 | `world_size>1` | ✅ | ❌ | — | — |
| **跳过第二次调用** | 实验性 | ✅ 64.8 | ✅ 30.4 | ✅ 157.4 | ✅ **202.8** |

（"跳过"那轮与 `dcp_active=False` 基线的 67.4/30.8/158.3/202.5 逐项吻合）

### 3.2 DCP=2（真分片，第二次调用语义必需、不能跳）

| 轮次 | 改动 | 并发 1 | 并发 2 | 并发 4 |
|---|---|---|---|---|
| 基线 | — | ✅ | ❌ | — |
| 跳过第二次 | 实验性 | ✅ | ❌ | — |
| **+`ori_zero_cmp` 默认开** | 本文修法 | ✅ 41.7 | ✅ 26.3 | ✅ 104.8 |

### 3.3 正确性（tiny，batch≥2 是重点）

判据：同一 prompt、`temperature=0`，**并发输出必须与单流参考逐字一致**。

```
=== 参考：单流连打 3 次 ===
  #0 ' admit witnesses admit witnesses ...'
  单流自一致: YES

=== 并发 4（同一 prompt 同时发 4 个）===
  与单流参考一致: 4/4
VERDICT: PASS
```

（tiny 是 dummy 权重，输出本身是固定串；**关键判据是"并发与单流一致"**，
它能抓住"batch>1 路径算错"这一类问题。）

---

## 4. 顺带修掉的一个真实缺陷

`dsa_v41.py:4240`：

```python
if cache_kind == "long_kv" and _v41_dcp_on() and has_compressed:
    n_local_heads = int(...num_attention_heads)     # 放大到全量 head
```

注释说明的意图是"走跨 rank 合并的层，q 会被 all-gather 成全部 head"。
但 `_v41_dcp_gather_heads` 的实际行为是 `if dcp_size <= 1: return q`（no-op）
⇒ `world_size == 1` 时 q 仍是 TP 分片（32 head），metadata 却按 64 head 建
⇒ 2× 错配。**修法：条件收紧为 `and _v41_dcp_group().world_size > 1`。**

这条单独**不修复崩溃**（实测仍崩），但它是真缺陷，且对 DCP8 无影响。

---

## 5. 为什么"跳过第二次调用"在 DCP>1 上是错的

跳过会落到"路线 A"，而路线 A 需要 `token_mask` 兜住空 rank 的 `LSE=0.0`
（kernel 写死的有限值，不是 -inf）。而 mask 只在
`_needs_mask = _pf.get('skip_2nd') == '1'` 时才构造 —— 只改跳过条件会让 mask 缺失。
DCP=1 能过是因为 merge 在 `dcp_size <= 1` 第 8 行就 early-return，mask 根本没被用。

⇒ **正确修法是让第二次调用本身合法（置零 cmp 长度），不是跳过它。**

---

## 6. 待办

- [ ] 8 卡 DCP=8 验证（并发 1/2/4/8 + 精度回归）
- [ ] 单流性能不倒退（基线 ms/step 40.44、A=3.29）
- [ ] 把补丁从 overlay 落到 `feat/v41-dcp8` 并推送
