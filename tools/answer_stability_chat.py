#!/usr/bin/env python3
"""答案稳定性（**用正确的接口**）：/v1/chat/completions + 服务默认 chat template。

上一版用 /v1/completions 裸 prompt，未套 deepseek_v41 的 chat template ⇒ 分布外
⇒ 抽取/事实类返回**空串**（10/10 逐字一致但正确率 0/10）——那是接口误用，不是模型能力。
本版改用服务实际对外接口（serve_a2.sh 已配 --tokenizer-mode=deepseek_v41
--default-chat-template-kwargs={"enable_thinking":false}）。

用法: answer_stability_chat.py <base> [rounds] [max_tokens]
"""
import json
import re
import sys
import time
import urllib.request
from collections import Counter

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:19210"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
MT = int(sys.argv[3]) if len(sys.argv) > 3 else 128

TASKS = [
    ("抽取-针", "抽取", "从这段里找出机房门禁密码并只回答密码本身：\n"
                       "值班记录：今天巡检正常。机房门禁密码是 ZQ7K-3341，请勿外传。", "ZQ7K-3341"),
    ("事实-首都", "事实", "中国的首都是哪座城市？只回答城市名。", "北京"),
    ("算术", "算术", "17 乘 23 等于多少？只回答数字。", "391"),
    ("抽取-列表", "抽取", "只回答三个名字，用逗号分隔：这段文字提到了苹果、香蕉和橘子。", None),
    ("开放-介绍", "开放", "请用一句话介绍北京。", None),
    ("长文-概括", "长文", None, None),          # 动态填
]

CORPUS = "/home/l00886679/cedpd-repo/data/hongloumeng.txt"
body = open(CORPUS, encoding="utf-8").read()
i = body.find("正文")
TASKS[-1] = ("长文-概括", "长文", (body[i + 2:i + 2 + 8000] if i >= 0 else body[:8000])
            + "\n\n请用一句话概括这段文字。", None)


def norm(s):
    return re.sub(r"[\s，。、！？：；,.!?:;\"'（）()【】\[\]-]+", "", s or "")


def ask(prompt, mt):
    payload = {"model": "deepseek-v41",
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": mt, "temperature": 0.0}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    ch = d["choices"][0]
    msg = ch.get("message") or {}
    txt = msg.get("content") or msg.get("reasoning_content") or ""
    return txt, time.perf_counter(), d.get("usage", {})


print("tp8k5 答案稳定性（/v1/chat/completions）：每任务 %d 轮，max_tokens=%d，temperature=0"
      % (ROUNDS, MT))
print()
print("%-12s %-6s %8s %8s %8s %8s  %s" % ("任务", "类型", "逐字一致", "归一一致", "前8字一致", "正确率", "样例"))
for name, kind, prompt, key in TASKS:
    outs, hit = [], 0
    for _ in range(ROUNDS):
        try:
            t, _, _ = ask(prompt, MT)
        except Exception as exc:  # noqa: BLE001
            t = "ERR %r" % (exc,)
        outs.append(t)
        if key and key in t:
            hit += 1
    raw = len(set(outs)) == 1
    nrm = len({norm(o) for o in outs}) == 1
    p8 = len({(norm(o) or "")[:8] for o in outs}) == 1
    acc = ("%d/%d" % (hit, ROUNDS)) if key else "—"
    sample = (Counter(outs).most_common(1)[0][0] or "")[:36].replace("\n", "\\n")
    print("%-12s %-6s %8s %8s %8s %8s  %s"
          % (name, kind, "是" if raw else "否", "是" if nrm else "否",
             "是" if p8 else "否", acc, repr(sample)))
