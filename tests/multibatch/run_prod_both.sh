#!/usr/bin/env bash
# =============================================================================
# run_prod_both.sh -- 生产口径的**两臂**对照：PREFIX=1（生产对齐）与 PREFIX=0（压力臂）。
#
#   为什么要两臂：`PREFIX=1` 才有"A2 生产形态"（前缀命中 ⇒ decode 队列里随时插入新请求）；
#   `PREFIX=0` 是压力臂（每次真 prefill ⇒ 128K 的 chunked prefill 与短请求 decode 更容易
#   混进同一个 batch step）。两臂都跑才能把"前缀缓存"与"混合 batch"两个变量分开看。
#   两臂各占**一次起服**（PREFIX 不能在运行时切换），所以是串行两个会话，预计 2×(起服+3块)。
#
# 用法（宿主机，包根目录）：
#   MODEL=/path/to/... bash tests/multibatch/run_prod_both.sh
#   MODEL=... ROUNDS=8 CONC=8 LONG_CTX=131072 SKIP=C bash tests/multibatch/run_prod_both.sh
# ⚠️ 会停掉并重建名为 $NAME 的容器（默认 dsv41-a2），且**会占满 8 张卡** ——
#    跑之前先确认目标机没有别人在用（run_test.sh 的 [0] 预检会检查 HBM）。
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/../.." && pwd)"

MODEL=${MODEL:-}
NAME=${NAME:-dsv41-a2}
PORT=${PORT:-8100}
MAX_SEQS=${MAX_SEQS:-32}
ROUNDS=${ROUNDS:-8}
CONC=${CONC:-8}
LONG_CTX=${LONG_CTX:-131072}
SKIP=${SKIP:-}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
rc_all=0

[ -n "$MODEL" ] || { echo "必须设置 MODEL=<模型目录>"; exit 2; }

for p in 1 0; do
  TAG="prod_p${p}_$STAMP"
  echo "===================== 臂 PREFIX=$p （TAG=$TAG）====================="
  MODEL="$MODEL" NAME="$NAME" PORT="$PORT" TAG="$TAG" MAX_SEQS="$MAX_SEQS" PREFIX="$p" \
    CONC="$CONC" ROUNDS="$ROUNDS" LONG_CTX="$LONG_CTX" SKIP="$SKIP" STOP_FIRST=1 \
    bash "$PKG/tests/multibatch/multibatch_session.sh" || rc_all=1
done

echo
echo "===================== 两臂对照 ====================="
python3 - "$PKG" "$STAMP" <<'PY'
import json, os, sys
pkg, stamp = sys.argv[1], sys.argv[2]
rows = []
for p in (1, 0):
    f = os.path.join(pkg, "results", f"prod_p{p}_{stamp}", "summary.json")
    if not os.path.exists(f):
        rows.append((p, None)); continue
    rows.append((p, json.load(open(f, encoding="utf-8"))))
print("| 臂 | A 轮内召回 | A 全长召回 | B 逐项不一致 | C 逐项不一致 | verdict |")
print("|---|---|---|---|---|---|")
for p, d in rows:
    if d is None:
        print(f"| PREFIX={p} | — | — | — | — | ❌ 未产出 summary.json |"); continue
    mt, cc, mx = d.get("multiturn"), d.get("concurrency"), d.get("mixed")
    a1 = f"{mt['needle_hits']}/{mt['needle_total']}" if mt else "—"
    a2 = f"{sum(mt['recall'].values())}/3" if mt else "—"
    b = len(cc["mismatch"]) if cc else "—"
    c = len(mx["mismatch"]) if mx else "—"
    print(f"| PREFIX={p} | {a1} | {a2} | {b} | {c} | {d.get('verdict')} |")
PY
exit "$rc_all"
