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

## 6. ★ 最终验收（8 卡真权重，DCP=8 + SPEC=1）

修复后的 overlay（`dsa_v41.py` md5 `9c431296…`）+ 命令行**不带** `--no-async-scheduling`
（= 修复前会崩的那个配置）。

### 6.1 并发矩阵（`ignore_eos=True`，差减法隔离 prefill）

| 并发 | ms/step | A | 聚合 tok/s | 每流 tok/s |
|---:|---:|---:|---:|---:|
| **1** | 37.56 | 2.94 | **67.3** | 67.3 |
| **2** | 47.71 | 2.51 | **89.7** | 44.8 |
| **4** | 55.30 | 2.60 | **161.5** | 40.4 |
| **8** | 88.24 | 2.41 | **187.0** | 23.4 |

* **1/2/4/8 全部不崩**（修复前并发 2 即 HTTP 500、8 worker 全挂）
* 聚合吞吐**单调上升**：67.3 → 89.7 → 161.5 → 187.0（正扩展性）
* 并发压测后复核：`17×23→391`、`T=2000→Q7`、`T=16000→Q7` 仍全 PASS

### 6.1b ⚠️ 并发 16 仍崩 —— 但**签名不同**，属另一个问题（未修）

同一次验收里把档位加到 16（`MAX_SEQS=16` 的上限）：

| 并发 | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 结果 | ✅ | ✅ | ✅ | ✅ | ❌ **崩** |

**签名与本文根因完全不同**：

| | 本文根因（并发 2/4） | 并发 16 |
|---|---|---|
| 错误码 | `507011` (AI Core Error) | **`507035`** |
| fault kernel | `SparseFlashMla_..._mix_aic` | **无 SMLA**；`aivec error` / vector core exception |
| 失败算子 | SparseFlashMla | **`copy_between_host_and_device_opapi`**（H2D 拷贝） |
| 崩点 | attention | `model_runner_v1.py:1462` `_prepare_inputs` |
| 具体错误 | `scalar instruction accesses an invalid GM address` | `The address for the MTE instruction to read on-chip buffer is out of bounds`（UB 越界，不是 GM） |

**触发路径**与首次崩溃相同（`admission_gate` 连做 16 步 prefill-only、
攒下 120 个 deferred decode、然后一次性释放），但那是**两条不同 bug 的共同前置条件**。

**⇒ 边界**：本次修复覆盖 **并发 1/2/4/8**（= 目标口径）。
并发 16 是独立问题，建议单独开线，不要在本文的修复上叠加猜测。

### 6.1c 最终验收轮（`dcpcap_1001_1700_accept`，服务至今存活）

| 并发 | ms/step | A | 聚合 tok/s | 每流 tok/s |
|---:|---:|---:|---:|---:|
| 1 | 37.81 | 2.50 | 56.7 | 56.7 |
| 2 | 24.03 | 2.80 | **200.4** | 100.2 |
| 4 | 41.95 | 2.52 | **206.0** | 51.5 |
| 8 | 84.93 | 2.37 | **191.0** | 23.9 |

* 并发压测**全程 0 ERROR**、`health=200`
* 压测后复核：`17×23→391`、`T=2000→Q7` 仍 PASS
* 多流相对单流的聚合收益 **≈3.4–3.6×**

**⚠️ 口径注意（两次验收轮的数字差异）**：并发 1 的聚合吞吐在 56.7–67.3 tok/s 之间浮动，
根因是 **A 随生成内容而变**（2.40–2.94）—— 不同档位的 prompt 取自语料不同位置，
温度=0 但内容不同 ⇒ 接受长度不同。**跨轮比较请用 ms/token，不要用聚合吞吐单点值。**

### 6.2 精度回归

| 用例 | 结果 |
|---|---|
| `17×23 → 391` | ✅ PASS |
| 长针 T=904 → `Q7` | ✅ PASS |
| 长针 T=2000 → `Q7` | ✅ PASS |
| 长针 T=8000 → `Q7` | ✅ PASS |
| 长针 T=16000 → `Q7` | ✅ PASS |

### 6.3 单流 2.30× 收益：**2.32×，不倒退**

同一 prompt、同一脚本口径的 A/B（`ignore_eos=True`，并发 1）：

| | ms/token | 聚合 tok/s |
|---|---:|---:|
| `SPEC=0 × DCP8` | 29.61 | 33.8 |
| `SPEC=1 × DCP8`（修复后） | 37.56 / 2.94 = **12.78** | 67.3 |
| **比值** | **2.32×** | 1.99× |

### 6.4 ★ 顺带发现：`--no-async-scheduling` 会**破坏精度**（新问题，已定位未修）

修复过程中发现：把 `--no-async-scheduling` 加回去（其余完全不变），长上下文精度**立即崩**：

| 配置（均含本次补丁） | 17×23 | T=904 | T=2000 | T=8000 | T=16000 |
|---|---|---|---|---|---|
| + `--no-async-scheduling` | PASS | PASS | ❌ FAIL | ❌ FAIL | ❌ FAIL |
| **不带该 flag** | PASS | PASS | ✅ PASS | ✅ PASS | ✅ PASS |

失败形态是输出乱码（`#【东22 西17 南? 北?】##` 之类），不是拒绝服务。
机制【推断】：该 flag 关掉 async spec decode ⇒ `rebuild_async_spec_decode_inputs`
在 `should_rebuild` 处早退 ⇒ DCP 的乐观 seq-len 修正链不执行 ⇒ 草稿槽位错位。

**⇒ 交付配置必须不带 `--no-async-scheduling`。** 这是之前几轮反复踩的那个 flag：
它既不是崩溃的解药（关掉照样崩在并发 2），又会引入精度回归。

---

## 7. 交付物

| 文件 | 说明 |
|---|---|
| `experimental/v41-dcp/tools/apply_dspark_batch_fix.py` | 幂等补丁脚本（两处，改 `dsa_v41.py`） |
| `experimental/v41-dcp/overlay/vllm_ascend/attention/dsa_v41.py` | 修复后的 overlay（md5 `9c431296…`） |
| 本文件 | 根因、排除矩阵、验收数据 |

**回退开关**：`V41_DCP_ORI_RAW_CMP=1` 恢复"第二次调用用原始 cmp 长度"的旧行为（会崩）。
