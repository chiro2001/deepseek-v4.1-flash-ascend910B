"""直接看 dump 里的块表前 12 列 + 统计真正需要的页数。"""
import sys

import torch

for path in sys.argv[1:]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    s = d["scalars"]
    T = d["q"].shape[0]
    obt = d["ori_block_table"][0]
    cbt = d["cmp_block_table"][0]
    ori_rows = d["ori_pages"].shape[0]
    cmp_rows = d["cmp_pages"].shape[0]
    print("=== %s T=%d window=%d" % (path.split("/")[-1], T, s["ori_win_left"] + 1))
    print("  ori_bt[0:12] = %s" % obt[:12].tolist())
    print("  cmp_bt[0:12] = %s" % cbt[:12].tolist())
    # 本请求需要几个 ori 页（块大小 = ori_pages 的页行数）
    bs = d["ori_page_rows"]
    need = (T + bs - 1) // bs
    used = [int(obt[c]) for c in range(min(need, obt.numel()))]
    print("  ori 页行数=%d ⇒ 需要 %d 列；这 %d 列的值=%s；唯一页=%s"
          % (bs, need, need, used, sorted(set(used))))
    print("  实际落盘 ori 页数=%d  cmp 页数=%d" % (ori_rows, cmp_rows))
    cs = int(d["seqused_cmp_kv"].max())
    needc = (cs + d["cmp_page_rows"] - 1) // d["cmp_page_rows"]
    usedc = [int(cbt[c]) for c in range(min(max(1, needc), cbt.numel()))]
    print("  cmp 需 %d 列 值=%s" % (needc, usedc))
