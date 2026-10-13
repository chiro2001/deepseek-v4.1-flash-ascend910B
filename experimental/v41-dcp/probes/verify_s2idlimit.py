"""离线核算 CSA 内核的因果界 `cmpS2IdLimit` 与我们传入的索引是否同坐标系。

内核源码（`sparse_flash_mla_csa_kernel.h`）：
    cmpMaskS2Size = actualCmpS2Size * cmpRatio + residual
    cmpMaskRight  = cmpMaskS2Size - actS1Size
    cmpS2IdLimit  = (cmpMaskRight + s1EndIdx + 1) / cmpRatio        # ← 用 **query 的全局位置**
    thresHold     = 同上
    actCmpS2Size  = min(actCmpS2Size, min(512, max(thresHold,0)))
    CountValidCmpSparseLen(bound)   # 二分找第一个 -1（假设有效项是连续前缀）

`GetKeyGmOffset(realS2Idx, ..., s2IdLimit)`：`realS2Idx >= s2IdLimit` ⇒ 该键被**静默丢弃**
（`CopyInSingleKv` 直接 return，且**不**增加 `mte2Size`）。

**关键**：`cmpS2IdLimit` 是从 **query 的全局位置** `s1EndIdx` 推出的"压缩域因果界"，
隐含"压缩 token g ↔ 未压缩 g*ratio"的**全局线性**映射。
而 DCP8 下我们传的是**本 rank 的局部行号**（交错分片，每 rank 只有 1/8 的行）。
本脚本用真实 dump 核算两者的差。
"""
import sys

import torch

path = sys.argv[1]
d = torch.load(path, map_location="cpu", weights_only=False)
s = d["scalars"]
T = int(s["max_seqlen_q"])
ratio = int(s["cmp_ratio"])
cseq = int(d["seqused_cmp_kv"].max())          # 本 rank 的压缩行数
actS1 = T
cmpMaskS2Size = cseq * ratio
cmpMaskRight = cmpMaskS2Size - actS1
flat = d["cmp_indices"].to(torch.int64).squeeze(1)

print("dump=%s" % path.split("/")[-1])
print("  T=%d ratio=%d cseq(本rank行数)=%d actS1=%d" % (T, ratio, cseq, actS1))
print("  cmpMaskS2Size = %d*%d = %d" % (cseq, ratio, cmpMaskS2Size))
print("  cmpMaskRight  = %d - %d = %d" % (cmpMaskS2Size, actS1, cmpMaskRight))
print()
print("  按内核公式，逐 query 行的 s2IdLimit 与该行可见键数对比：")
print("  %-6s %-12s %-12s %-12s %-14s %s" % ("t", "s2IdLimit", "n_valid", "内核会用", "索引>=界的个数", "后果"))

rows_bad = 0
tot_dropped = 0
for t in range(T):
    thres = (cmpMaskRight + t + 1) // ratio
    s2id = thres
    n_valid = int((flat[t] >= 0).sum())
    bound = min(cseq, min(512, max(thres, 0)))
    a = flat[t]
    # CountValidCmpSparseLen(bound)：假设有效项是连续前缀
    cnt = 0
    for i in range(bound):
        if i < a.numel() and a[i] >= 0:
            cnt += 1
        else:
            break
    use = min(bound, cnt)
    ge = int((a[:use][a[:use] >= 0] >= s2id).sum()) if use else 0
    if ge:
        rows_bad += 1
        tot_dropped += ge
    if t % 120 == 0 or (t >= T - 3):
        print("  %-6d %-12d %-12d %-12d %-14d %s"
              % (t, s2id, n_valid, use, ge, "★ 有键被丢弃" if ge else "ok"))
print()
print("  汇总：有键被静默丢弃的行数 = %d/%d（%.1f%%），累计丢弃键数 = %d"
      % (rows_bad, T, 100.0 * rows_bad / T, tot_dropped))
