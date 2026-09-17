#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vision_accuracy_check.py — end-to-end vision acceptance probe for a served
DeepSeek-V4.1-Flash W4A8 (DSpark) checkpoint.

WHAT IT DOES
------------
Sends >=20 short image-QA cases to an OpenAI-compatible vLLM endpoint and scores
them programmatically (short-answer contains/exact match).  Built-in cases use
the two official images shipped with the model:

    <official>/inference/examples/images/carrots.jpeg
    <official>/inference/examples/images/corn.jpeg

Their identity ("carrots"/"corn") comes from the official interleaved example
(`inference/examples/example_harmony.json` / `example.txt`); the remaining
attributes (color, background, growth location, edible part, shape, surface,
central position) were derived from the official images with a simple
color/geometry analysis.  You can replace the suite with your own JSON via
`--cases` (same schema: id/image/question/accept/forbid).

PASS CRITERIA (defaults)
------------------------
  * short-answer hit rate >= 80% over all cases;
  * 0 empty/garbled answers and 0 request errors;
  * the two mandatory negative checks pass:
      neg-text-01  pure text request does not regress
      neg-swap-01  asking "is this a carrot?" on the corn image answers "no"
  * neg-blank-01 (a generated blank image must not be identified as carrot/corn)
    is reported as an extra hallucination check (counted in the hit rate).

RECORDED PER CASE / SUMMARY
---------------------------
  * TTFT (streaming first delta), latency, prompt/completion tokens;
  * per-image token cost: prompt_tokens(image) - prompt_tokens(text-only probe);
  * HBM / KV metrics scraped from /metrics before and after each case
    (gauge deltas) plus the static BF16 vision-weight bytes;
  * DSpark speculation counters (`vllm:spec_decode_*`) and acceptance rate;
  * server model id, image paths, vision weight bytes.

USAGE
-----
    # text-only smoke is not enough: start the server with vision enabled, e.g.
    #   VISION=1 MAX_LEN=8192 SPEC=1 bash scripts/launch_serve_half.sh
    python scripts/vision_accuracy_check.py \
        --server http://127.0.0.1:8000 \
        --images-dir /path/to/DeepSeek-V4.1-Flash/inference/examples/images \
        --official-dir /path/to/DeepSeek-V4.1-Flash \
        --out logs/perf/vision_accuracy.json --expect-dspark

    # offline validation of the scoring logic (no server):
    python scripts/vision_accuracy_check.py --self-test
    python scripts/vision_accuracy_check.py --dry-run --images-dir <images>

STATUS
------
The harness itself is offline self-tested (`--self-test`).  It has NOT been run
end-to-end on an NPU server in this delivery (docs/ACCURACY.md §5 records
"Vision 端到端 = 未跑"), so treat the first run on real hardware as the
acceptance run itself.
"""
import argparse
import base64
import json
import os
import re
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib

VISION_TOTAL_BYTES_DEFAULT = 970536960  # 266 BF16 tensors = 0.9705 GB (measured)
SPEC_METRIC_RE = re.compile(r"spec_decode")
MEM_METRIC_RE = re.compile(r"memory|hbm|kv_cache|kv_usage|npu")
METRIC_LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9.eE+-]+)\s*$")

# ---------------------------------------------------------------------------
# built-in suite: 10 questions x 2 official images + 3 negative checks
# ---------------------------------------------------------------------------
CASES = [
    # ---- carrots.jpeg ----
    dict(id="carrots-01", image="carrots.jpeg",
         question="图中主要是什么食材？只回答食材名称。",
         accept=["胡萝卜", "萝卜", "carrot"], forbid=["玉米", "corn"]),
    dict(id="carrots-02", image="carrots.jpeg",
         question="用英文回答：图中主要是什么食材？只回答一个英文单词。",
         accept=["carrot"], forbid=["corn"]),
    dict(id="carrots-03", image="carrots.jpeg",
         question="图中食材的主要颜色是什么？只回答颜色。",
         accept=["橙", "橘", "orange"], forbid=["绿", "green"]),
    dict(id="carrots-04", image="carrots.jpeg",
         question="图片的背景大致是什么颜色？只回答颜色。",
         accept=["白", "white"], forbid=[]),
    dict(id="carrots-05", image="carrots.jpeg",
         question="图中的食材通常生长在土壤的下面还是上面？只回答“下面”或“上面”。",
         accept=["下面", "地下", "土壤下", "below", "underground"],
         forbid=["上面", "地上", "above"]),
    dict(id="carrots-06", image="carrots.jpeg",
         question="这种食材通常食用的是植物的哪个部位？只回答一个词。",
         accept=["根", "root"], forbid=["种子", "果实", "叶"]),
    dict(id="carrots-07", image="carrots.jpeg",
         question="图中食材的形状更接近细长条还是圆球？只回答“细长条”或“圆球”。",
         accept=["细长", "长条", "长", "elongated"], forbid=["圆球", "球状", "round", "sphere"]),
    dict(id="carrots-08", image="carrots.jpeg",
         question="这张照片的类型是食物特写、自然风景还是人物照？只回答类型。",
         accept=["食物", "食品", "特写", "food"], forbid=["风景", "人物", "landscape", "portrait"]),
    dict(id="carrots-09", image="carrots.jpeg",
         question="图中食材表面是光滑的还是有明显颗粒感？只回答“光滑”或“颗粒”。",
         accept=["光滑", "smooth"], forbid=["颗粒", "凹凸", "bumpy"]),
    dict(id="carrots-10", image="carrots.jpeg",
         question="画面中央是否存在食物？只回答“是”或“否”。",
         accept=["是", "yes", "有"], forbid=["否", "没有", "no"]),
    # ---- corn.jpeg ----
    dict(id="corn-01", image="corn.jpeg",
         question="图中主要是什么食材？只回答食材名称。",
         accept=["玉米", "苞米", "粟米", "corn", "maize"], forbid=["胡萝卜", "carrot"]),
    dict(id="corn-02", image="corn.jpeg",
         question="用英文回答：图中主要是什么食材？只回答一个英文单词。",
         accept=["corn", "maize"], forbid=["carrot"]),
    dict(id="corn-03", image="corn.jpeg",
         question="图中食材的主要颜色是什么？只回答颜色。",
         accept=["黄", "yellow"], forbid=["橙", "orange", "紫", "purple"]),
    dict(id="corn-04", image="corn.jpeg",
         question="图片的背景大致是什么颜色？只回答颜色。",
         accept=["白", "white"], forbid=[]),
    dict(id="corn-05", image="corn.jpeg",
         question="图中的食材通常生长在土壤的下面还是上面？只回答“下面”或“上面”。",
         accept=["上面", "地上", "地面", "above", "ground"],
         forbid=["地下", "土壤下", "underground", "below"]),
    dict(id="corn-06", image="corn.jpeg",
         question="这种食材通常食用的是植物的哪个部位？只回答一个词。",
         accept=["种子", "籽", "籽粒", "果实", "穗", "kernel", "seed", "grain"],
         forbid=["根", "root", "叶"]),
    dict(id="corn-07", image="corn.jpeg",
         question="图中食材的形状更接近细长棒状还是圆球？只回答“细长棒状”或“圆球”。",
         accept=["长", "棒", "椭", "圆柱", "穗", "ear", "long", "ellip"],
         forbid=["圆球", "球状", "round", "sphere"]),
    dict(id="corn-08", image="corn.jpeg",
         question="这张照片的类型是食物特写、自然风景还是人物照？只回答类型。",
         accept=["食物", "食品", "特写", "food"], forbid=["风景", "人物", "landscape", "portrait"]),
    dict(id="corn-09", image="corn.jpeg",
         question="图中食材表面是光滑的还是有明显颗粒感？只回答“光滑”或“颗粒”。",
         accept=["颗粒", "粒", "凹凸", "bumpy", "kernel"], forbid=["光滑", "smooth"]),
    dict(id="corn-10", image="corn.jpeg",
         question="画面中央是否存在食物？只回答“是”或“否”。",
         accept=["是", "yes", "有"], forbid=["否", "没有", "no"]),
    # ---- negative checks ----
    dict(id="neg-text-01", image=None, mandatory=True,
         question="中国的首都是哪里？只回答城市名。",
         accept=["北京"], forbid=["上海", "广州", "深圳"]),
    dict(id="neg-swap-01", image="corn.jpeg", mandatory=True,
         question="图中的食材是胡萝卜吗？只回答“是”或“否”。",
         accept=["否", "不是", "不", "no"], forbid=["是的", "对"]),
    dict(id="neg-blank-01", image="__blank__",
         question="图中主要是什么食材？如果图中没有清晰可辨的食材，请只回答“无法判断”。",
         accept=["无法判断", "无法", "不能", "不知道", "没有", "看不清", "unclear"],
         forbid=["胡萝卜", "玉米", "carrot", "corn"]),
]


def make_blank_png(path, size=96, rgb=(160, 160, 160)):
    """Write a solid-colour PNG using only the stdlib (negative-case image)."""
    w = h = size
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(png)
    return path


def norm(text):
    return re.sub(r"[\s\u3000]+", "", (text or "").lower())


def is_garbled(text):
    t = (text or "").strip()
    if not t:
        return True
    if "\ufffd" in t:
        return True
    if len(t) >= 8:
        # a single character (or a tiny pair) repeated over and over
        for n in (1, 2):
            rep = re.match(r"^(.)\1*$", t) if n == 1 else None
            if rep:
                return True
        if len(set(t)) <= 2 and len(t) >= 16:
            return True
        # a short token repeated many times ("哈哈哈..." / "the the the ...")
        head = t[:12]
        if len(t) >= 24 and t.count(head) >= 2 and len(set(t)) <= 4:
            return True
    printable = sum(1 for ch in t if ch.isprintable())
    return printable / max(len(t), 1) < 0.9


def score_case(case, answer):
    """Return (passed, reason)."""
    if answer is None:
        return False, "request_error"
    a = norm(answer)
    if is_garbled(answer):
        return False, "empty_or_garbled"
    for tok in case.get("forbid", []):
        if norm(tok) and norm(tok) in a:
            return False, "forbidden_token:%s" % tok
    for tok in case.get("accept", []):
        if norm(tok) and norm(tok) in a:
            return True, "matched:%s" % tok
    return False, "no_expected_token"


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------
def _request(url, payload=None, timeout=300, extra_headers=None):
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


def http_get_json(url, timeout=30):
    with _request(url, None, timeout=timeout) as r:
        return json.loads(r.read())


def http_get_text(url, timeout=30):
    with _request(url, None, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def chat(base, model, case, image_uri, max_tokens, temperature, stream, timeout):
    content = []
    if image_uri:
        content.append({"type": "image_url", "image_url": {"url": image_uri}})
    content.append({"type": "text", "text": case["question"]})
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content if image_uri else case["question"]}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if not stream:
        obj = None
        with _request(base.rstrip("/") + "/v1/chat/completions", body, timeout=timeout) as r:
            obj = json.loads(r.read())
        choice = obj["choices"][0]
        text = (choice.get("message") or {}).get("content") or choice.get("text") or ""
        return text, None, obj.get("usage") or {}, obj
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    usage = {}
    with _request(base.rstrip("/") + "/v1/chat/completions", body, timeout=timeout,
                  extra_headers={"Accept": "text/event-stream"}) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                delta = (choice.get("delta") or {})
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks.append(piece)
    return "".join(chunks), ttft, usage, None


def scrape_metrics(url, timeout=10):
    text = http_get_text(url, timeout=timeout)
    gauges, counters = {}, {}
    for line in text.splitlines():
        m = METRIC_LINE_RE.match(line.strip())
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", m.group(3)
        if "spec_decode" in name:
            key = name
            counters[key] = counters.get(key, 0.0) + float(val)
        elif MEM_METRIC_RE.search(name):
            gauges[name + labels] = float(val)
    return {"spec_decode": counters, "memory": gauges}


def vision_weight_bytes(official_dir):
    """Sum the official vision/aligner/image_* tensor bytes (stdlib safetensors header parse)."""
    if not official_dir:
        return VISION_TOTAL_BYTES_DEFAULT, None
    try:
        idx = json.load(open(os.path.join(official_dir, "model.safetensors.index.json")))["weight_map"]
    except Exception:
        return VISION_TOTAL_BYTES_DEFAULT, None
    sizes = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1,
             "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1}
    total = 0
    cache = {}
    for key, shard in idx.items():
        if not key.startswith(("vision.", "aligner.", "image_")):
            continue
        if shard not in cache:
            with open(os.path.join(official_dir, shard), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                cache[shard] = json.loads(fh.read(n).decode("utf-8"))
        meta = cache[shard][key]
        numel = 1
        for d in meta["shape"]:
            numel *= int(d)
        total += numel * sizes[meta["dtype"]]
    return total, len([k for k in idx if k.startswith(("vision.", "aligner.", "image_"))])


def resolve_image(images_dir, name, blank_dir):
    if name == "__blank__":
        return make_blank_png(os.path.join(blank_dir, "blank-gray.png"))
    if os.path.isabs(name):
        return name
    return os.path.join(images_dir, name)


def data_uri(path):
    ext = os.path.splitext(path)[1].lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp"}.get(ext, "application/octet-stream")
    with open(path, "rb") as fh:
        return "data:%s;base64,%s" % (mime, base64.b64encode(fh.read()).decode())


# ---------------------------------------------------------------------------
# self-test / dry-run
# ---------------------------------------------------------------------------
def self_test():
    for case in CASES:
        good = case["accept"][0]
        ok, why = score_case(case, good)
        assert ok, (case["id"], good, why)
        bad = case["forbid"][0] if case.get("forbid") else "完全无关的回答"
        ok2, why2 = score_case(case, bad)
        assert not ok2, (case["id"], bad, why2)
        assert not score_case(case, "")[0], (case["id"], "empty")
        assert not score_case(case, "\ufffd\ufffd\ufffd")[0], (case["id"], "garbled")
        assert not score_case(case, None)[0], (case["id"], "error")
    real_img_cases = [c for c in CASES if c["image"] not in (None, "__blank__")]
    assert len(CASES) >= 20 and len(real_img_cases) >= 20, (len(CASES), len(real_img_cases))
    mandatory = [c for c in CASES if c.get("mandatory")]
    assert len(mandatory) >= 2, len(mandatory)
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))
    with tempfile.TemporaryDirectory() as td:
        png = make_blank_png(os.path.join(td, "blank.png"))
        assert os.path.getsize(png) > 0
        assert data_uri(png).startswith("data:image/png;base64,")
    print("SELF-TEST PASS: %d cases (%d image QA + %d negative incl. blank-image probe); "
          "scoring/empty/garbled/error/blank-PNG paths OK"
          % (len(CASES), len(real_img_cases), len(CASES) - len(real_img_cases)))
    return 0


def dry_run(args):
    images = []
    for c in CASES:
        if c["image"] not in (None, "__blank__") and c["image"] not in images:
            images.append(c["image"])
    print("suite       : %d cases -> %s" % (len(CASES), ", ".join(c["id"] for c in CASES)))
    print("images-dir  : %s" % args.images_dir)
    print("images      : %s" % ", ".join(images))
    print("server      : %s" % args.server)
    print("criteria    : hit_rate >= %.2f, empty/garbled == 0, errors == 0, mandatory negatives pass"
          % args.min_hit_rate)
    print("dry-run: no request sent")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--official-dir", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--cases", default=None, help="optional JSON list overriding the built-in suite")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--min-hit-rate", type=float, default=0.80)
    ap.add_argument("--no-stream", action="store_true", help="disable SSE (TTFT unavailable)")
    ap.add_argument("--no-measure-image-tokens", action="store_true")
    ap.add_argument("--expect-dspark", action="store_true",
                    help="record/warn about vllm:spec_decode_* metrics")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.images_dir is None and not args.cases:
        ap.error("--images-dir is required for the built-in suite")
    if args.dry_run:
        return dry_run(args)

    cases = CASES
    if args.cases:
        cases = json.load(open(args.cases))
        for c in cases:
            c.setdefault("accept", [])
            c.setdefault("forbid", [])
    images_dir = args.images_dir or ""

    blank_dir = tempfile.mkdtemp(prefix="vision_blank_")
    resolved = {}
    for c in cases:
        if c.get("image") is None:
            resolved[c["id"]] = None
            continue
        p = resolve_image(images_dir, c["image"], blank_dir)
        if not os.path.isfile(p):
            print("[error] image not found: %s" % p, file=sys.stderr)
            return 2
        resolved[c["id"]] = p
    uris = {cid: (data_uri(p) if p else None) for cid, p in resolved.items()}

    model = args.model
    try:
        models = http_get_json(args.server.rstrip("/") + "/v1/models", timeout=30)
        ids = [m.get("id") for m in models.get("data", []) if m.get("id")]
        if not model:
            model = ids[0] if ids else None
        print("[server] %s models=%s using=%s" % (args.server, ids, model))
    except Exception as exc:
        print("[error] cannot reach %s/v1/models: %s" % (args.server, exc), file=sys.stderr)
        return 2
    if not model:
        print("[error] no model id; pass --model", file=sys.stderr)
        return 2

    metrics_url = args.server.rstrip("/") + "/metrics"
    try:
        base_metrics = scrape_metrics(metrics_url)
    except Exception as exc:
        base_metrics = None
        print("[warn] /metrics unavailable: %s" % exc)

    results = []
    n_pass = n_empty = n_err = 0
    for c in cases:
        cid = c["id"]
        rec = {"id": cid, "image": c.get("image"), "question": c["question"],
               "accept": c.get("accept"), "forbid": c.get("forbid"),
               "mandatory": bool(c.get("mandatory"))}
        pre = post = None
        try:
            pre = scrape_metrics(metrics_url)
        except Exception:
            pass
        t0 = time.perf_counter()
        try:
            try:
                text, ttft, usage, raw = chat(args.server, model, c, uris[cid],
                                              args.max_tokens, args.temperature,
                                              not args.no_stream, args.timeout)
            except Exception as stream_exc:
                if args.no_stream:
                    raise
                text, ttft, usage, raw = chat(args.server, model, c, uris[cid],
                                              args.max_tokens, args.temperature,
                                              False, args.timeout)
                rec["stream_fallback"] = "%s: %s" % (type(stream_exc).__name__, str(stream_exc)[:120])
            rec["answer"] = text
            rec["ttft_s"] = None if ttft is None else round(ttft, 4)
            rec["latency_s"] = round(time.perf_counter() - t0, 4)
            rec["usage"] = usage
            passed, why = score_case(c, text)
            rec["passed"], rec["reason"] = passed, why
            if is_garbled(text):
                n_empty += 1
            if passed:
                n_pass += 1
        except Exception as exc:
            rec["answer"] = None
            rec["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
            rec["passed"], rec["reason"] = False, "request_error"
            n_err += 1
            rec["latency_s"] = round(time.perf_counter() - t0, 4)
        try:
            post = scrape_metrics(metrics_url)
        except Exception:
            pass
        if pre and post:
            deltas = {}
            for k, v in post["memory"].items():
                if k in pre["memory"]:
                    deltas[k] = v - pre["memory"][k]
            rec["metrics_memory_delta"] = dict(sorted(deltas.items(), key=lambda kv: -abs(kv[1]))[:12])
            rec["spec_decode"] = post["spec_decode"]
        results.append(rec)
        print("  [%s] %-14s %s" % ("PASS" if rec["passed"] else "FAIL", cid,
                                   (rec.get("answer") or rec.get("error") or "")[:60].replace("\n", " ")))

    # ---- image token cost probes -------------------------------------------------
    image_tokens = {}
    if not args.no_measure_image_tokens:
        probe = dict(id="probe", question="只回答一个字：好")
        seen = sorted({c["image"] for c in cases if c["image"] not in (None, "__blank__")})
        for name in seen:
            p = os.path.join(images_dir, name)
            try:
                _, _, u_img, _ = chat(args.server, model, probe, data_uri(p), 4, 0.0, False, args.timeout)
                _, _, u_txt, _ = chat(args.server, model, probe, None, 4, 0.0, False, args.timeout)
                image_tokens[name] = {
                    "prompt_tokens_with_image": u_img.get("prompt_tokens"),
                    "prompt_tokens_text_only": u_txt.get("prompt_tokens"),
                    "image_tokens": (u_img.get("prompt_tokens") or 0) - (u_txt.get("prompt_tokens") or 0),
                    "max_image_tokens_config": 1024,
                }
            except Exception as exc:
                image_tokens[name] = {"error": str(exc)[:200]}
        print("[image-tokens] %s" % json.dumps(image_tokens, ensure_ascii=False))

    # ---- DSpark / metrics --------------------------------------------------------
    dspark = {"expected": bool(args.expect_dspark)}
    try:
        final = scrape_metrics(metrics_url)
        spec = final.get("spec_decode", {})
        dspark["spec_decode_metrics"] = spec
        accepted = spec.get("vllm:spec_decode_num_accepted_tokens_total")
        draft = spec.get("vllm:spec_decode_num_draft_tokens_total")
        if draft:
            dspark["acceptance_rate"] = round((accepted or 0.0) / draft, 4)
            dspark["mean_acceptance_length"] = round(1.0 + (accepted or 0.0) / draft, 4)
        if args.expect_dspark and not spec:
            dspark["warning"] = ("no vllm:spec_decode_* metrics found; server may not be running "
                                 "with --speculative-config method=dspark")
    except Exception as exc:
        dspark["error"] = str(exc)[:200]

    vbytes, vcount = vision_weight_bytes(args.official_dir)
    mandatory = [r for r in results if r["mandatory"]]
    mandatory_ok = all(r["passed"] for r in mandatory)
    total = len(results)
    hit_rate = n_pass / total if total else 0.0
    criteria = {
        "hit_rate_min": args.min_hit_rate,
        "hit_rate": round(hit_rate, 4),
        "empty_or_garbled_max": 0,
        "empty_or_garbled": n_empty,
        "errors_max": 0,
        "errors": n_err,
        "mandatory_negative_checks": [r["id"] for r in mandatory],
        "mandatory_negative_ok": mandatory_ok,
    }
    verdict = "PASS" if (hit_rate >= args.min_hit_rate and n_empty == 0 and n_err == 0 and mandatory_ok) else "FAIL"
    summary = {
        "script": "scripts/vision_accuracy_check.py",
        "server": args.server,
        "model": model,
        "images_dir": images_dir,
        "official_dir": args.official_dir,
        "cases": total,
        "passed": n_pass,
        "failed": total - n_pass,
        "empty_or_garbled": n_empty,
        "errors": n_err,
        "criteria": criteria,
        "verdict": verdict,
        "vision_weight_bytes": vbytes,
        "vision_weight_count": vcount,
        "image_tokens": image_tokens,
        "dspark": dspark,
        "latency": {
            "ttft_s": [round(r["ttft_s"], 4) for r in results if r.get("ttft_s") is not None],
            "latency_s": [r.get("latency_s") for r in results if r.get("latency_s")],
        },
    }
    out = {"summary": summary, "results": results,
           "baseline_metrics_before": base_metrics}
    print("== vision_accuracy_check ==")
    print("cases: %d  pass: %d (%.1f%%)  empty/garbled: %d  errors: %d"
          % (total, n_pass, hit_rate * 100, n_empty, n_err))
    print("mandatory negatives: %s" % ("OK" if mandatory_ok else "FAILED"))
    print("vision weights: %d tensors / %.4f GB (BF16, FLOAT, not quantized)"
          % (vcount or 0, vbytes / 1e9))
    print("dspark: %s" % json.dumps({k: v for k, v in dspark.items() if k != "spec_decode_metrics"},
                                    ensure_ascii=False))
    print("verdict: %s" % verdict)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
            fh.write("\n")
        print("json: %s" % os.path.abspath(args.out))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
