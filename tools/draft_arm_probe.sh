#!/usr/bin/env bash
# =============================================================================
# draft_arm_probe.sh —— 一次跑完"一个 draft 臂"的全部判据
#
# 为什么要单独一个脚本：`draft_graph_guard.sh` 只看 A 与 tok/s，
# 而定位 draft 问题时**逐位置接受率**（pos0..pos4）比 A 更早暴露问题：
#   * 健康（stock eager draft）：pos0 ≈ 0.77、A ≈ 2.8
#   * 入图静默失效          ：pos0 ≈ 0.22、A ≈ 1.05
# 另外把设备侧自测步钟 `[bneck] hp=` 一并取出来（**不依赖客户端**）。
#
# 用法：
#   bash tools/draft_arm_probe.sh <base_url> <run_dir> [标签]
# 产出：
#   <run_dir>/arm_<标签>/{guard.json,guard.log,specdec.log,bneck.log,summary.txt}
# =============================================================================
set -uo pipefail

BASE=${1:-http://127.0.0.1:8020}
RUN=${2:?需要 run 目录（results/<run_id>）}
LABEL=${3:-arm}
PKG=${PKG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT="$RUN/arm_$LABEL"
mkdir -p "$OUT"
LOG="$RUN/serve.log"

say() { printf '\n=== %s ===\n' "$*"; }

say "0. health"
code=$(curl -s -o /dev/null -w '%{http_code}' -m 10 "$BASE/health" || echo 000)
[ "$code" = "200" ] || { echo "[probe] FAIL: health=$code"; exit 2; }
echo "health=$code"

# 只统计本次测量窗口内的行
MARK=$(wc -l < "$LOG")

say "1. 基准（1024 prompt / 256 output；conc=1 档跑 **8 条**请求取中位）"
# ⚠️ 口径陷阱：`bench_concurrency` 的 prompt 条数 = `--concurrency` 列表里的**最大值**。
#    传 `--concurrency 1` 只发 **1 条**请求 —— 拿它可以和"64 条中位数"比就会得出
#    完全错误的结论（A 的发放级方差很大）。这里固定 `1,2,4,8` ⇒ 8 条 prompt，
#    取 conc=1 那一行（8 条串行）作为该臂的 A/tok-s。
cd "$PKG"
timeout 900 python3 tools/bench_concurrency.py \
  --base-url "$BASE" --model deepseek-v41 \
  --concurrency 1,2,4,8 --prompt-tokens 1024 --output-tokens 256 --repeats 1 \
  --corpus-file data/dihuo.txt --suffix-dir data/dihuo_local \
  --json-out "$OUT/guard.json" > "$OUT/guard.log" 2>&1
tail -n +$MARK "$LOG" > "$OUT/window.log"
grep -oE "SpecDecoding metrics:.*" "$OUT/window.log" | tail -3 > "$OUT/specdec.log"
grep -oE "\[bneck\] mode=[a-z]+ steps=[0-9]+ .*hp=[0-9.]+" "$OUT/window.log" | tail -8 > "$OUT/bneck.log"

python3 - "$OUT" <<'PY'
import json, re, sys, statistics as st
out = sys.argv[1]
rows = json.load(open(f"{out}/guard.json"))["rows"]
r = next(x for x in rows if x["conc"] == 1.0)      # conc=1 档（8 条串行）
A = r["accept_len"]; tok = r["per_stream_med"]; ttft = r["ttft_med"]
n_ok = r["ok"]
hps = [float(m.group(1)) for m in re.finditer(r"hp=([0-9.]+)", open(f"{out}/bneck.log").read())]
pos = None
for line in open(f"{out}/specdec.log"):
    m = re.search(r"Per-position acceptance rate: ([0-9., ]+?),", line)
    if m:
        pos = [float(x) for x in m.group(1).split(",") if x.strip()]
    m2 = re.search(r"Mean acceptance length: ([0-9.]+)", line)
    m3 = re.search(r"Avg Draft acceptance rate: ([0-9.]+)%", line)
    if m2: A_srv = float(m2.group(1))
    if m3: dr = float(m3.group(1))
lines = [
    f"label={out.split('/')[-1]}",
    f"conc=1 请求数={int(n_ok)}",
    f"A(client)={A:.3f}  A(server,last-window)={locals().get('A_srv', float('nan')):.3f}  draft_acc={locals().get('dr', float('nan')):.1f}%",
    f"tok/s={tok:.1f}  ttft={ttft:.2f}s",
    f"hp(ms/step) median={st.median(hps):.2f} n={len(hps)}" if hps else "hp: 无数据（该窗口没打 [bneck]）",
    f"pos0..posN={pos}",
    f"pos0 {'OK(>=0.6)' if (pos and pos[0] >= 0.6) else 'BAD(<0.6) —— 第一个 draft token 就错了' if pos else 'n/a'}",
]
txt = "\n".join(lines) + "\n"
open(f"{out}/summary.txt", "w").write(txt)
print(txt)
PY

say "2. 判据"
A=$(python3 -c "import json;print(json.load(open('$OUT/guard.json'))['rows'][0]['accept_len'])")
TOK=$(python3 -c "import json;print(json.load(open('$OUT/guard.json'))['rows'][0]['per_stream_med'])")
python3 - "$A" "$TOK" <<'PY'
import sys
A = float(sys.argv[1]); tok = float(sys.argv[2])
ok = A >= 1.3 and tok >= 80
print(f"A={A:.3f} tok/s={tok:.1f}  =>  {'PASS（draft 真的在出 token）' if ok else 'FAIL（疑似静默失效或退化）'}")
print("参考：stock eager draft  A≈2.7–3.0 / 90–111 tok/s")
print("      draft 入图(坏)     A≈1.05–1.84 / 42–58 tok/s")
sys.exit(0 if ok else 1)
PY
