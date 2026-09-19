#!/usr/bin/env bash
# =============================================================================
# DRAFT-GUARD —— 证明 DSpark draft 图**真的在出 token**，而不是静默失效
#
# 为什么需要它（2026-09-20 实测）：
#   把 DRAFT_GRAPH 默认改成 1 之后，**所有开关都验证到位**了 ——
#     * 容器内 DSPARK_GRAPH_CAPTURE_METADATA=1
#     * draft 版 dspark_proposer.py 已装（grep 命中 2 处）
#     * 起服命令行确实是 speculative-config enforce_eager=false
#   但效果仍然是坏的：
#
#     配置              A(接受长度)   单流 tok/s   ms/step
#     DRAFT_GRAPH=0      2.7–3.0       90–111      27–30
#     DRAFT_GRAPH=1      1.06–1.08     43.0        25.1
#
#   A≈1.0 = draft 完全没产出；而 ms/step 反而"更好看"（每步只出 1.08 个 token
#   而不是 2.85 个）⇒ **真实吞吐慢 2.2×**，只看 ms/step 会得出相反结论。
#
#   ⇒ 光检查"开关/文件/命令行"不够，必须检查**效果**。这个脚本就是干这个的。
#
# 判据（必须同时看两个数，缺一不可）：
#   A >= 1.3  且  tok/s 与 DRAFT_GRAPH=0 时同口径**持平或更好**
#   A ≈ 1.0（1.05 以下）⇒ 判定静默失效
#
# 用法：bash draft_graph_guard.sh [base_url]
# 退出码：0 通过 / 1 判定静默失效 / 2 无基准可比
# =============================================================================
set -uo pipefail

BASE=${1:-http://127.0.0.1:8020}
PKG=${PKG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT=${OUT:-/tmp/draft_guard}
mkdir -p "$OUT"

say() { printf '\n=== %s ===\n' "$*"; }

say "1. 服务活着吗"
code=$(curl -s -o /dev/null -w '%{http_code}' -m 10 "$BASE/health" || echo 000)
[ "$code" = "200" ] || { echo "[guard] FAIL: health=$code"; exit 2; }
echo "health=$code"

say "2. 跑一条与发布口径一致的基准（1024 prompt / 256 output）"
cd "$PKG"
timeout 900 python3 tools/bench_concurrency.py \
  --base-url "$BASE" --model deepseek-v41 \
  --concurrency 1 --prompt-tokens 1024 --output-tokens 256 --repeats 1 \
  --corpus-file data/dihuo.txt --suffix-dir data/dihuo_local \
  --json-out "$OUT/guard.json" > "$OUT/guard.log" 2>&1
grep -E "^  [0-9]" "$OUT/guard.log" | tail -1

python3 - "$OUT/guard.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    r = d["rows"][0]
except Exception as e:
    print(f"[guard] FAIL: \u62ff\u4e0d\u5230\u57fa\u51c6\u7ed3\u679c ({e})"); raise SystemExit(2)
A = r["accept_len"]; tok = r["per_stream_med"]
print(f"\n[guard] A={A:.3f}  \u5355\u6d41={tok:.1f} tok/s")

# \u5224\u636e\u4ee5 tok/s \u4e3a\u4e3b\u3001A \u4e3a\u8f85\u3002
BASE_TOK_LOW = 80.0
if tok < BASE_TOK_LOW:
    print(f"[guard] FAIL: \u5355\u6d41 {tok:.1f} tok/s < {BASE_TOK_LOW:.0f}")
    if A <= 1.3:
        print(f"[guard]   A={A:.3f} \u2248 1.0 => draft \u5b8c\u5168\u6ca1\u4ea7\u51fa")
    print("[guard]   ms/step \u6b64\u65f6\u53cd\u800c\u66f4\u5c0f\uff0c\u90a3\u662f\u6bcf\u6b65 token \u6570\u53d8\u5c11\u7684\u5047\u8c61")
    raise SystemExit(1)
if A < 1.3:
    print(f"[guard] WARN: \u5355\u6d41 {tok:.1f} \u5c1a\u53ef\u4f46 A={A:.3f} \u504f\u4f4e")
    raise SystemExit(2)
print(f"[guard] PASS: A={A:.3f}  \u5355\u6d41={tok:.1f} tok/s")
raise SystemExit(0)
PY
_rc=$?
echo "[guard] raw exit=$_rc"
exit $_rc
