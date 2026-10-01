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
