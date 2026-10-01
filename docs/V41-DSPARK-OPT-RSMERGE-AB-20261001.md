# V41_DCP_RS_MERGE 的 A/B 实测（2026-10-01）

## 0. 背景：为什么想到 reduce_scatter

拆解发现 DSpark 让 merge 的 `all_reduce` 从"延迟 bound"变成"带宽 bound"：

`dsa_v41.py:805-817` 的注释自己写着：

> 「瓶颈是**集合通信的次数（延迟）**，不是带宽（**T=1 时单次才 ~128 KB**）」

而 DSpark 把 target 的 token 行数（T）从 **1** 提到 **8**：

| | T | 打包张量 `[T,H,2D+2]` fp32 | 字节 |
|---|---:|---|---:|
| SPEC=0 | 1 | `1×64×1026` | **256.5 KB** |
| SPEC=1 | 8 | `8×64×1026` | **2052.0 KB** |

⇒ **数据量 ×8**，进入带宽域。实测单次耗时 **13.0 → 21.7 µs（+67%）**。

而 `_v41_dcp_merge_attention` 归约后**每个 rank 只需要自己那 8 个 head**
（`o_proj` 是 TP 切分的），`all_reduce` 却让每个 rank 都收到全部 64 个 head 的和
⇒ **88.9% 的接收量是白拿的**。

代码里已有 `V41_DCP_RS_MERGE=1` 开关实现 `reduce_scatter_tensor` 替代
（数学恒等：逐元素求和后切片 == 只收自己那段的求和）。

## 1. A/B 实测（tiny，TP2+DCP2）

| 臂 | ms/step（5 轮中位） | 离散 |
|---|---:|---|
| `V41_DCP_RS_MERGE=0`（基线） | **36.24** | 36.08–36.50（±0.6%） |
| `V41_DCP_RS_MERGE=1` | **36.49** | 36.48–36.70（±0.3%） |
| | | **+0.7%（略慢）** |

**⇒ 在 DCP=2 上没有收益。**

## 2. 为什么 DCP=2 看不出收益

`reduce_scatter` 的收益随 **DCP 度**增长：

| DCP | all_reduce 接收量/rank | reduce_scatter 接收量/rank | 省 |
|---:|---|---:|---:|
| 2 | 全量 | 1/2 | 50% |
| **8** | 全量 | **1/8** | **88.9%** |

且 DCP=2 时 ring all_reduce 只需 1 跳，通信本身就很便宜 —— 收益被固定延迟淹没。

**⇒ 结论：RS_MERGE 的价值必须在 DCP=8 上验证，tiny（DCP=2）测不出来。**

## 3. 下一步

1. 在 **TP8 + DCP=8** 上做同样 A/B（起服约 10 分钟/臂）。
2. 若 DCP=8 有收益 ⇒ 考虑把默认值转正（需完整精度回归）。
3. 若仍无收益 ⇒ 说明瓶颈不在接收量而在别处（如 `permute+contiguous` 的额外拷贝），
   转去优化 `_pack` 的构造（`_hm.permute(1,0,2).reshape()` 若触发 contiguous 会多一次全量拷贝）。

## 4. 复现

```bash
# tiny（DCP=2）：约 4 分钟/臂
bash ~/tmp/rs_ab.sh 0 base
bash ~/tmp/rs_ab.sh 1 rsm
```

---

## 5. ★★★ DCP=8 实测：RS_MERGE=1 **破坏正确性**（A 掉到 1.00）

run `dcpcap_1001_2200_rsm8`（TP8 + DCP8 + SPEC=1 + `V41_DCP_RS_MERGE=1`，其余与交付配置逐项相同）。

### 5.1 判据：A（平均接受长度）

| run | RS_MERGE | `SpecDecoding metrics` 的 A |
|---|---:|---|
| `dcpcap_1001_1830_deliver` | **0**（默认） | **2.47 / 2.42 / 2.60** ✅ |
| `dcpcap_1001_1940_restore` | **0** | **3.00 / 2.28 / 2.41** ✅ |
| **`dcpcap_1001_2200_rsm8`** | **1** | **1.00 / 1.00 / 1.00** ❌ |

**`A ≡ 1.00` 的含义**：所有草稿 token 都被 verify 拒绝，只有 target 自己采的那个被接受。
这正是"**merge 结果算错 ⇒ 草稿与 target 对不上**"的确定性签名。

`draft_tokens_total = 28777`（draft 在正常产出），但 `accepted` 恒为 0。

### 5.2 性能：看起来"更快"，但那是假象

| 臂 | ms/step（8 轮中位） | A | 每 token |
|---|---:|---:|---:|
| RS_MERGE=0 | ~43.2 | 2.50–2.80 | **~15.6 ms** |
| RS_MERGE=1 | **36.06** | **1.00** | 36.06 ms |

`ms/step` 看起来更小了（43.2 → 36.1），但**因为草稿全被拒绝，每步只产出 1 个 token**
⇒ 每 token 从 15.6 ms **劣化到 36.06 ms（2.3×）**。

**⇒ 这不是优化，是正确性回归。RS_MERGE=1 不能启用。**

### 5.3 为什么 DCP=2 上"正常"、DCP=8 上崩

tiny（DCP=2）的 A 也是 2.00（dummy 权重下的固定值），当时**没有察觉异常**；
而 DCP=8 上 A 从 2.5 掉到 1.0 —— 差异可能在：

- DCP=2 时 `reduce_scatter` 切 2 段，`_rows = H//dcp = 32`，与 `head_slice` 恰好对齐；
- DCP=8 时 `_rows = 64//8 = 8`，**若 `head_slice` 的范围与 reduce_scatter 的切法不一致，取到的就不是本 rank 的 head**。

【推断】`_hm = _pack.permute(1,0,2).reshape(_H, -1)` 之后 `reduce_scatter_tensor` 按
**第 0 维（H 维）** 切成 `dcp` 段，第 r 段 = head `[8r, 8r+8)`。
代码注释声称这与 `head_slice` 对齐，但**在 DCP=8 + DSpark（T=8）下实测不成立**。

## 6. 结论

| 项 | 结果 |
|---|---|
| RS_MERGE 在 DCP=2 的性能 | +0.7%（无收益） |
| RS_MERGE 在 DCP=8 的正确性 | ❌ **A 掉到 1.00（草稿全被拒）** |
| 能否启用 | **不能** |

**⇒ 保持 `V41_DCP_RS_MERGE=0`（默认）。** 若要启用，必须先修 §5.3 的
head 对齐问题，并重新做完整精度回归。
