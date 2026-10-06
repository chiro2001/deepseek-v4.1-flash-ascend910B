#!/usr/bin/env python3
"""决策相关测量：同一问题多轮，**答案**是否稳定/正确？

背景：已证明 tp8k5 在 decode 存在微小抖动，且可见性取决于分布饱和程度（→ 见
docs/NONDET-VISIBILITY-CONTENT-DEPENDENT-20261007.md）。
但"逐位不同"不等于"答案不同"。本工具问：**实践上答案会不会变/变错？**

对每类任务重复 N 轮、temperature=0，给出：
  · 逐字一致率（最严格）
  · 归一化一致率（去空白/标点）
  · 首 N 字符一致率（弱判据，对应"分叉位置分布"）

用法: answer_stability.py <base> [rounds] [max_tokens]
"""
import json
import re
import sys
import time
import urllib.request
from collections import Counter

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:19210"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
MT = int(sys.argv[3]) if len(sys.argv) > 3 else 64

TASKS = [
    # (名称, 任务类型, prompt, 正确答案关键词)
    ("抽取-针", "抽取", "请从下面这段里找出机房门禁密码并只回答密码本身：\n"
                    "值班记录：今天巡检正常。机房门禁密码是 ZQ7K-3341，请勿外传。备份口令未变。", "ZQ7K-3341"),
    ("事实-首都", "事实", "中国的首都是哪座城市？只回答城市名。", "北京"),
    ("算术", "算术", "17 乘 23 等于多少？只回答数字。", "391"),
    ("抽取-列表", "抽取", "只回答三个名字，用逗号分隔：这段文字提到了苹果、香蕉和橘子。", None),
    ("开放-介绍", "开放", "请用一句话介绍北京。", None),
    ("开放-续写", "续写", "下面是一段小说开头，请接着写一句：\n"
                       "那天夜里，风从山口灌下来，吹得灯笼在檐下乱晃。", None),
]


def norm(s):
    return re.sub(r"[\s，。、！？：；,.!?:;\"'（）()【】\[\]-]+", "", s or "")


def ask(prompt, mt):
    payload = {"model": "deepseek-v41", "prompt": prompt, "max_tokens": mt,
               "temperature": 0.0}
    req = urllib.request.Request(BASE + "/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return d["choices"][0]["text"], time.perf_counter() - t0


print("tp8k5 答案稳定性：每任务 %d 轮，max_tokens=%d，temperature=0" % (ROUNDS, MT))
print()
print("%-12s %-6s %8s %8s %8s %8s  %s" % ("任务", "类型", "逐字一致", "归一一致", "前8字一致", "正确率", "样例"))
for name, kind, prompt, key in TASKS:
    outs, hit, dts = [], 0, []
    for _ in range(ROUNDS):
        try:
            t, dt = ask(prompt, MT)
        except Exception as exc:  # noqa: BLE001
            t, dt = "ERR %r" % (exc,), 0.0
        outs.append(t)
        dts.append(dt)
        if key and key in t:
            hit += 1
    raw = len(set(outs)) == 1
    nrm = len({norm(o) for o in outs}) == 1
    p8 = len({(norm(o) or "")[:8] for o in outs}) == 1
    acc = ("%d/%d" % (hit, ROUNDS)) if key else "—"
    sample = (Counter(outs).most_common(1)[0][0] or "")[:34].replace("\n", "\\n")
    print("%-12s %-6s %8s %8s %8s %8s  %s"
          % (name, kind, "是" if raw else "否", "是" if nrm else "否",
             "是" if p8 else "否", acc, repr(sample)))
