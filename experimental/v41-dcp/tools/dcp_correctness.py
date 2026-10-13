#!/usr/bin/env python3
"""DCP 正确性回归（短问答 + 长上下文多选针）。用法见 --help。

为什么用**多选格式**做长针：交接文档 §4 的方法论教训 —— 自由格式长针在长长度上
会进复读循环（`PLUM-BLOSSOM-BLOSSOM…`），把"检索失败"与"模型复读"混在一起。
多选（"是 A7 还是 B9，只回答一个"）有唯一、可判定的期望输出。
"""
import argparse, json, random, sys, urllib.request

SHORT = [
    ("1+1 等于几？只回答数字。", "2"),
    ("17 × 23 等于多少？只回答数字。", "391"),
    ("背出《七律·到韶山》的前一句下半句：为有牺牲多壮志，", "敢教日月换新天"),
    ("《红楼梦》的作者是谁？只回答人名。", "曹雪芹"),
    ("中华人民共和国的首都是哪座城市？只回答城市名。", "北京"),
    ("用一句不超过二十字的话说明什么是 KV cache。", None),
]


def chat(url, model, prompt, max_tokens=64, timeout=1800):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.0}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return (d["choices"][0]["message"]["content"] or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:19210")
    ap.add_argument("--model", default="deepseek-v41")
    ap.add_argument("--tok-dir", default="/home/l00886679/models/out/v41-flat-verify3")
    ap.add_argument("--corpus", default="/tmp/hongloumeng.txt")
    ap.add_argument("--lengths", default="2000,8000,16000")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    rows = []
    ok = 0
    print("=== 短问答 ===", flush=True)
    for q, exp in SHORT:
        try:
            out = chat(a.base_url, a.model, q, 64)
        except Exception as e:  # noqa: BLE001
            out = "<ERR %s>" % e
        good = None if exp is None else (exp.replace(" ", "") in out.replace(" ", ""))
        ok += int(bool(good))
        rows.append({"kind": "short", "q": q, "out": out, "exp": exp, "ok": good})
        print("  q=%-28s out=%-24r ok=%s" % (q[:28], out[:24], good), flush=True)
    print("  短问答 %d/%d" % (ok, len(SHORT)), flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok_dir, trust_remote_code=True)
    corpus = open(a.corpus, encoding="utf-8", errors="ignore").read()
    rng = random.Random(20260929)
    print("=== 长上下文多选针（针放**开头**）===", flush=True)
    for L in [int(x) for x in a.lengths.split(",")]:
        for needle, gold, decoy in (("【档案编号：A7】", "A7", "B9"),
                                    ("【档案编号：K3】", "K3", "M8")):
            q = "\n\n上文提到的档案编号是 %s 还是 %s？只回答一个编号，不要其他内容。" % (gold, decoy)
            nid = tok.encode(needle); qid = tok.encode(q)
            body = tok.decode(tok.encode(corpus[rng.randrange(0, 200000):])[: max(32, L - len(nid) - len(qid))])
            prompt = needle + "\n\n" + body + q
            try:
                out = chat(a.base_url, a.model, prompt, 8)
            except Exception as e:  # noqa: BLE001
                out = "<ERR %s>" % e
            good = gold in out and decoy not in out
            rows.append({"kind": "mc", "L": L, "needle": needle, "out": out, "ok": good})
            print("  L=%5d needle=%s out=%-14r ok=%s" % (L, gold, out[:14], good), flush=True)

    mc = [r for r in rows if r["kind"] == "mc"]
    print("\n汇总：短问答 %d/%d，长针 %d/%d" % (ok, len(SHORT), sum(bool(r["ok"]) for r in mc), len(mc)), flush=True)
    if a.json_out:
        json.dump(rows, open(a.json_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return 0 if (ok == len(SHORT) and all(r["ok"] for r in mc)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
