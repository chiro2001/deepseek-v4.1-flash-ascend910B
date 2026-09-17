#!/usr/bin/env python3
"""精度评测（官方 encoder 路径）：GSM8K / C-Eval，经 /v1/completions，带断点续跑。
用法: acc_eval.py --task gsm8k --limit 200 --mode chat --out x.json --tag w4a8 --conc 4
"""
import argparse, json, os, re, sys, threading, time, urllib.request

# P4: P2 host-tier build crashes (ERR00100/AI-core fault) when multiple long
# prefills are scheduled in the same batch.  Serialize request submission until
# the first streamed content token (=> prefill finished), while decodes of other
# workers continue concurrently.
PREFILL_LOCK = threading.Lock()
SERIALIZE_PREFILL = True

# [v4 参数化] 原始脚本硬编码 `/home/user/models/DeepSeek-V4.1-Flash/encoding`。
# 这里改为：① 环境变量 `ENC_DIR`；② 命令行 `--enc-dir`（两者都优先于默认值）；
# ③ 再退回到包内自带副本 `tests/encoding/`（如果 A2 侧没有官方目录）。
ENC_DIR = os.environ.get(
    "ENC_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "encoding"),
)
# [v4] 允许命令行 `--enc-dir` 覆盖（**必须在 import 之前解析**，否则 encoding 已被加载）。
if "--enc-dir" in sys.argv:
    try:
        ENC_DIR = sys.argv[sys.argv.index("--enc-dir") + 1]
    except IndexError:
        pass
for _k in ("--enc-dir=",):
    for _a in sys.argv:
        if _a.startswith(_k):
            ENC_DIR = _a[len(_k):]
if not os.path.isdir(ENC_DIR):
    _guess = os.path.expanduser("~/models/DeepSeek-V4.1-Flash/encoding")
    if os.path.isdir(_guess):
        ENC_DIR = _guess
if not os.path.isdir(ENC_DIR):
    sys.stderr.write(
        f"[acc_eval][FAIL] 找不到官方 encoding 目录：{ENC_DIR}\n"
        "  请用 `--enc-dir /path/to/DeepSeek-V4.1-Flash/encoding` 或 `export ENC_DIR=...` 指定。\n"
        "  （这个目录来自官方 checkpoint，包内不含；GSM8K/C-Eval 必须要有它。）\n"
    )
    raise SystemExit(2)
sys.path.insert(0, ENC_DIR)
from encoding import encode_messages  # noqa: E402

def build_prompt(question, mode, effort=75):
    msgs = [{"role": "user", "content": question}]
    if mode == "thinking":
        return encode_messages(msgs, thinking_mode="thinking", reasoning_effort=effort)
    return encode_messages(msgs, thinking_mode="chat")

def complete(base, prompt, max_tokens=512, timeout=600, temperature=0.0):
    if not SERIALIZE_PREFILL:
        body = {"model": "deepseek-v41", "prompt": prompt, "max_tokens": max_tokens,
                "temperature": temperature}
        req = urllib.request.Request(base.rstrip("/") + "/v1/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["choices"][0]["text"]
    # streaming + global prefill gate: hold the lock only through first content token
    body = {"model": "deepseek-v41", "prompt": prompt, "max_tokens": max_tokens,
            "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base.rstrip("/") + "/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    parts = []
    got_first = False
    r = None
    with PREFILL_LOCK:
        r = urllib.request.urlopen(req, timeout=timeout)
        try:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return "".join(parts)
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                text = ((obj.get("choices") or [{}])[0].get("text") or "")
                if text:
                    parts.append(text)
                    got_first = True
                    break
        finally:
            if r is not None and not got_first:
                r.close()
    if not got_first or r is None:
        return "".join(parts)
    try:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            text = ((obj.get("choices") or [{}])[0].get("text") or "")
            if text:
                parts.append(text)
    finally:
        r.close()
    return "".join(parts)

def strip_think(t):
    if t is None:
        return ""
    if "</think>" in t:
        return t.rsplit("</think>", 1)[-1].strip()
    if "<think>" in t:
        return ""
    return t.strip()

NUM = re.compile(r"-?[\d,]*\.?\d+")

def extract_gsm8k(text):
    if not text:
        return ""
    m = re.findall(r"####\s*([^\n]+)", text)
    if m:
        nums = NUM.findall(m[-1].replace(",", "").replace("$", ""))
        if nums:
            return nums[-1].rstrip(".")
    nums = NUM.findall(text.replace(",", "").replace("$", ""))
    return nums[-1].rstrip(".") if nums else ""

def norm_num(s):
    try:
        return str(int(round(float(s))))
    except Exception:
        return str(s).strip()

def extract_choice(text):
    if not text:
        return ""
    t = text.strip()
    m = re.search(r"答案\s*[:：是为]?\s*([ABCD])\b", t)
    if m:
        return m.group(1)
    lines = [x.strip().strip("：:。.、* ") for x in t.splitlines() if x.strip()]
    solo = [x for x in lines if x in ("A", "B", "C", "D")]
    if solo:
        return solo[-1]
    m = re.findall(r"\b([ABCD])\b", t)
    return m[-1] if m else ""

def load_gsm8k(limit):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    tr = load_dataset("openai/gsm8k", "main", split="train")
    shots = [{"q": tr[i]["question"], "a": tr[i]["answer"]} for i in range(8)]
    items = []
    for i in range(min(limit, len(ds))):
        ex = ds[i]
        p = "".join(f"Question: {s['q']}\nAnswer: {s['a']}\n\n" for s in shots)
        p += f"Question: {ex['question']}\nAnswer:"
        items.append({"idx": f"gsm8k-{i}", "prompt": p, "gold": extract_gsm8k(ex["answer"])})
    return items

CEVAL_SUBJECTS = ["modern_chinese_history", "law", "college_medicine", "computer_network"]

def load_ceval(limit, shots=0):
    from datasets import load_dataset
    per = max(1, limit // len(CEVAL_SUBJECTS))
    items = []
    for subj in CEVAL_SUBJECTS:
        try:
            dev = load_dataset("ceval/ceval-exam", subj, split="dev")
            val = load_dataset("ceval/ceval-exam", subj, split="val")
        except Exception as e:
            print(f"[warn] {subj}: {type(e).__name__}", flush=True)
            continue
        def fmt(ex, with_ans):
            s = ex["question"] + "\nA. " + ex["A"] + "\nB. " + ex["B"] + "\nC. " + ex["C"] + "\nD. " + ex["D"] + "\n"
            return s + ("答案：" + ex["answer"] + "\n\n" if with_ans else "请逐步思考，并在最后一行只输出选项字母（A/B/C/D）。")
        head = "".join(fmt(dev[i], True) for i in range(min(shots, len(dev))))
        for j in range(min(per, len(val))):
            items.append({"idx": f"{subj}-{j}", "prompt": head + fmt(val[j], False),
                          "gold": str(val[j]["answer"]).strip()})
    return items

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="gsm8k")
ap.add_argument("--limit", type=int, default=200)
ap.add_argument("--base-url", default="http://127.0.0.1:8001")
# [v4] `--base` 是 `--base-url` 的别名（本包的 tests/t_gsm8k.py 用 `--base`）。
ap.add_argument("--base", dest="base_url", help="--base-url 的别名")
ap.add_argument("--enc-dir", help="官方 encoding 目录（覆盖环境变量 ENC_DIR 与默认猜测）")
ap.add_argument("--conc", type=int, default=4)
ap.add_argument("--mode", default="chat", choices=["chat", "thinking"])
ap.add_argument("--effort", type=int, default=75)
ap.add_argument("--max-tokens", type=int, default=512)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default="w4a8")
ap.add_argument("--serialize-prefill", type=int, default=1,
                help="1=hold a global lock until first token (avoid concurrent prefills)")
a = ap.parse_args()
SERIALIZE_PREFILL = bool(a.serialize_prefill)
if a.enc_dir:
    sys.path.insert(0, a.enc_dir)

items = load_gsm8k(a.limit) if a.task == "gsm8k" else load_ceval(a.limit)
extractor = extract_gsm8k if a.task == "gsm8k" else extract_choice
for it in items:
    it["full_prompt"] = build_prompt(it["prompt"], a.mode, a.effort)

ckpt = a.out + ".jsonl"
done_map = {}
if os.path.exists(ckpt):
    for line in open(ckpt, errors="ignore"):
        try:
            r = json.loads(line)
            # P4: do not trust failed/empty rows across server crashes -> retry them
            if r.get("err") or not (r.get("raw") or "").strip():
                continue
            done_map[str(r["idx"])] = r
        except Exception:
            pass
print(f"[{a.tag}] task={a.task} mode={a.mode} n={len(items)} conc={a.conc} 缓存={len(done_map)}", flush=True)

results = [None] * len(items)
lock = threading.Lock()
done = [0]
t0 = time.time()

def worker(sl):
    for i in sl:
        it = items[i]
        if str(it["idx"]) in done_map:
            results[i] = done_map[str(it["idx"])]
            with lock:
                done[0] += 1
            continue
        rec = {"idx": it["idx"], "gold": it["gold"], "pred": "", "raw": "", "ok": False, "err": None}
        for attempt in range(4):
            try:
                txt = strip_think(complete(a.base_url, it["full_prompt"], a.max_tokens))
                rec["raw"] = txt
                rec["pred"] = extractor(txt)
                rec["ok"] = (norm_num(rec["pred"]) == norm_num(it["gold"])) if a.task == "gsm8k" \
                    else (rec["pred"].upper() == it["gold"].upper())
                rec["err"] = None
                break
            except Exception as e:
                rec["err"] = f"{type(e).__name__}: {str(e)[:120]}"
                time.sleep(5 * (attempt + 1))
        results[i] = rec
        with lock:
            with open(ckpt, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[0] += 1
            if done[0] % 20 == 0:
                ok = sum(1 for r in results if r and r["ok"])
                print(f"  {done[0]}/{len(items)} acc={ok}/{done[0]} ({ok/done[0]*100:.1f}%) {time.time()-t0:.0f}s", flush=True)

threads = [threading.Thread(target=worker, args=(list(range(j, len(items), a.conc)),)) for j in range(a.conc)]
for t in threads: t.start()
for t in threads: t.join()

results = [r for r in results if r]
ok = sum(1 for r in results if r["ok"])
empty = sum(1 for r in results if not (r["raw"] or "").strip())
summary = {"tag": a.tag, "task": a.task, "mode": a.mode, "n": len(results), "correct": ok,
           "acc": round(ok / len(results) * 100, 2) if results else 0,
           "empty": empty, "errors": sum(1 for r in results if r["err"]),
           "seconds": round(time.time() - t0, 1)}
json.dump({"summary": summary, "results": results}, open(a.out, "w"), ensure_ascii=False, indent=1)
print(f"== {a.tag} {a.task}/{a.mode}: {ok}/{len(results)} = {summary['acc']}% (空={empty}, {summary['seconds']}s) ==")
