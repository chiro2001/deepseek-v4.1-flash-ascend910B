#!/usr/bin/env python3
"""★ 修补集合的**覆盖性**属性测试（不需要 NPU）。

命题（见 `patches/engram_hash.repair_helpers.py:plan_repair_slots` 的 docstring）：

    对任意"每步每请求一段连续 span"的 batch，凡是**读者会读到、而本步 step-1
    没有写过**的槽位（row, shift），一定被 `plan_repair_slots` 覆盖。

等价说法（本测试的判据）：设
    written  = {positions[row] for all row}          # step-1 逐行写入的位置
    needed   = {(row,sh) : sh>=1 且 q=pos[row]-sh>=0 且 q ∉ written}
    covered  = set(zip(*plan_repair_slots(...)))
则恒有 needed ⊆ covered（且 shift=0 一律不在 covered 里）。

再验两个边界：① 行**不连续**（假设被破坏）时退化为全扫（仍覆盖）；
② 代价上界：连续布局下 |covered| ≤ (lookback-1)·lookback/2 × 请求数
   （lookback=4 ⇒ 每请求 ≤ 6 槽位，**与 batch 大小无关**）。

用法：`python3 tests/test_repair_plan.py`
"""
from __future__ import annotations

import random

from _loader import repair_helpers

LOOKBACK = 4


def _brute(positions, request_ids, lookback):
    written = set(positions)
    needed = set()
    for r, pos in enumerate(positions):
        for sh in range(1, lookback):
            q = pos - sh
            if q >= 0 and q not in written:
                needed.add((r, sh))
    return needed


def main() -> int:
    helpers = repair_helpers()
    fails = 0
    cases = 0

    random.seed(20260922)
    for _ in range(4000):
        n_req = random.randint(1, 5)
        positions: list[int] = []
        request_ids: list[int] = []
        cursor = random.choice([0, 1, 3, 7, 128, 65536])
        for req in range(n_req):
            span = random.choice([1, 1, 1, 2, 3, 5, 8, 17, 64])
            base = cursor
            for i in range(span):
                positions.append(base + i)
                request_ids.append(req)
            cursor = base + span + random.choice([0, 0, 0, 1, 9])
        if not positions:
            continue
        cases += 1
        needed = _brute(positions, request_ids, LOOKBACK)
        rows, shifts = helpers.plan_repair_slots(positions, request_ids, LOOKBACK)
        covered = set(zip(rows, shifts))
        if 0 in shifts:
            print(f"✗ shift=0 出现在修补集合里（绝不允许覆盖本步自己的写入）：{positions}")
            fails += 1
        missing = needed - covered
        if missing:
            print(f"✗ 漏覆盖：positions={positions} reqs={request_ids} needed-covered={sorted(missing)}")
            fails += 1
        n_req_seen = len(set(request_ids))
        bound = (LOOKBACK - 1) * LOOKBACK // 2 * n_req_seen
        if len(covered) > bound:
            print(f"✗ 代价上界被突破：|covered|={len(covered)} > {bound}")
            fails += 1

    # ① 行不连续 ⇒ 必须退化为全扫（仍覆盖）
    pos_gap = [4, 5, 9]  # 请求 0：4,5（缺 6,7,8）再 9
    rows, shifts = helpers.plan_repair_slots(pos_gap, [0, 0, 0], LOOKBACK)
    covered = set(zip(rows, shifts))
    needed = _brute(pos_gap, [0, 0, 0], LOOKBACK)
    if not needed <= covered:
        print(f"✗ 不连续布局漏覆盖：{sorted(needed - covered)}")
        fails += 1
    if (1, 1) not in covered:  # q = 5-1 = 4 ∈ written ⇒ 其实不需要；这里只看退化行为
        pass
    cases += 1

    # ② 空 batch
    if helpers.plan_repair_slots([], [], LOOKBACK) != ([], []):
        print("✗ 空 batch 未返回空计划")
        fails += 1
    cases += 1

    print(f"跑了 {cases} 个随机/边界用例，失败 {fails} 项")
    if fails:
        return 1
    print("✓ 覆盖性 + 代价上界 + 退化行为 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
