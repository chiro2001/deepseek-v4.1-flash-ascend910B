#!/usr/bin/env python3
"""仿真 `scripts/serve_a2.sh` 的 CAPTURE_SIZES 推导，用于**回归核对**。

背景（`docs/MAXSEQS-128-MEASURED-20261007.md`）：桶表按 `T = N × (1+SP_TOKENS)` 对齐。
旧逻辑只撒 N∈{1..8,10,12,16} 的档，再直接跳到最大桶，于是：

  · MAX_SEQS=64 ：表是 `…,96,384` ⇒ conc=17 的 102 token 被 padding 到 384（**+276%**）；
  · MAX_SEQS=128：表是 `…,96,768` ⇒ 最坏 **+653%**。

修复（2026-10-07）：仅当 **MAX_SEQS > 32** 时把 `_n_list` 扩展到
`{20,24,32,40,48,56,64,72,80,96,112,128}`，补齐中间档；
**MAX_SEQS ≤ 32（含交付基线 32）的桶表逐字节不变**。

本脚本同时实现旧/新逻辑并对比，任何一侧改动都会立刻在输出里暴露。
用法: python3 capture_sizes_sim.py [sp_tokens]
"""
import sys

SP = int(sys.argv[1]) if len(sys.argv) > 1 else 5
STEP = SP + 1

BASE_N = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16]
EXT_N = [20, 24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128, 144, 160, 192, 224, 256]


def derive(max_seqs, extended):
    """复刻 serve_a2.sh 的推导顺序。"""
    caps = [1, 2, 3, 4]
    cap_max = max(max_seqs * STEP, 32)
    n_list = BASE_N + (EXT_N if (extended and max_seqs > 32) else [])
    for n in n_list:
        c = n * STEP
        if STEP <= c <= cap_max:
            caps.append(c)
    if STEP not in caps:                      # 保证 1+SP 档存在
        caps.append(STEP)
    if cap_max not in caps:                   # 保证最大桶 >= cap_max
        caps.append(cap_max)
    return caps


def worst_pad(max_seqs, caps):
    worst = (None, 0, 0)
    n_pad = 0
    for conc in range(1, max_seqs + 1):
        t = conc * STEP
        nxt = min([c for c in caps if c >= t], default=None)
        if nxt is None:
            continue
        if nxt > t:
            n_pad += 1
            if nxt / t > worst[1] / (worst[2] or 1):
                worst = (conc, nxt, t)
    return n_pad, worst


print("SP_TOKENS=%d ⇒ 每请求 %d token/步；桶表按 N×%d 对齐" % (SP, STEP, STEP))
print()
ok = True
for ms in (1, 2, 8, 16, 32, 48, 64, 128, 256):
    old = derive(ms, extended=False)
    new = derive(ms, extended=True)
    same = "逐字节相同 ✅" if old == new else "已扩展"
    print("MAX_SEQS=%-4d 桶数 %2d→%-2d  %s" % (ms, len(old), len(new), same))
    print("    cap_max=%d（N=%d × %d）" % (ms * STEP, ms, STEP))
    if new != old:
        added = [x for x in new if x not in old]
        lost = [x for x in old if x not in new]
        print("    新增桶: %s" % added)
        if lost:
            print("    ⚠️ 丢失桶: %s" % lost)
            ok = False
    if ms <= 32 and old != new:
        print("    ⚠️ MAX_SEQS<=32 必须逐字节不变！")
        ok = False
    if new[-1] < ms * STEP:
        print("    ⚠️ 最大桶 %d < cap_max %d ⇒ 大 batch 会被判为不可图" % (new[-1], ms * STEP))
        ok = False
    print("    最坏 padding：旧 +%.0f%% → 新 +%.0f%%"
          % (100 * (worst_pad(ms, old)[1][1] / (worst_pad(ms, old)[1][2] or 1) - 1),
             100 * (worst_pad(ms, new)[1][1] / (worst_pad(ms, new)[1][2] or 1) - 1)))
    print()

print("总体：%s" % ("✅ 全部核对通过（≤32 不变、>32 补齐、最大桶齐备）" if ok else "❌ 有核对项失败"))
sys.exit(0 if ok else 1)
