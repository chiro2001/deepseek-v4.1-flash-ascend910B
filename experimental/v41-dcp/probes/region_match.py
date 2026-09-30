"""比对两个区域：
  A) 内核公式预测会**静默丢弃键**的 query 行区间
  B) 实测出现 NaN 的 query 行区间
若二者高度重叠 ⇒ 丢弃键导致 kvMergeGm_ 出现"未写洞"，下游按期望长度读 ⇒ NaN。
"""
import sys

import torch

path = sys.argv[1]
d = torch.load(path, map_location="cpu", weights_only=False)
s = d["scalars"]
T = int(s["max_seqlen_q"]); ratio = int(s["cmp_ratio"])
cseq = int(d["seqused_cmp_kv"].max())
cmpMaskRight = cseq * ratio - T
flat = d["cmp_indices"].to(torch.int64).squeeze(1)

drop_rows = []
for t in range(T):
    thres = (cmpMaskRight + t + 1) // ratio
    use = min(min(cseq, min(512, max(thres, 0))), flat.shape[1])
    a = flat[t][:use]
    ge = int((a[a >= 0] >= thres).sum()) if use else 0
    if ge:
        drop_rows.append(t)

print("dump=%s  T=%d cseq=%d ratio=%d cmpMaskRight=%d" % (path.split("/")[-1], T, cseq, ratio, cmpMaskRight))
print("  A) 预测丢弃键的行: %d 个, 区间 [%s, %s]"
      % (len(drop_rows), drop_rows[0] if drop_rows else "-", drop_rows[-1] if drop_rows else "-"))
print("     连续段: %s" % ("连续" if drop_rows and drop_rows[-1] - drop_rows[0] + 1 == len(drop_rows) else "非连续"))
# 顺便：预测"actCmpS2Size=0（完全不用 cmp）"的行
zero_rows = [t for t in range(T) if (cmpMaskRight + t + 1) // ratio <= 0]
print("  A2) 预测 actCmpS2Size=0（完全跳过 cmp）的行: %d 个, 区间 [%s, %s]"
      % (len(zero_rows), zero_rows[0] if zero_rows else "-", zero_rows[-1] if zero_rows else "-"))
