#!/usr/bin/env bash
# 组装 DSpark 版 checkpoint（量化主干 + MTP 反量化 BF16）。任一步失败即退出。
set -euo pipefail
H=/home/user; P=$H/projects/dsv41
QF=$H/models/out/v41-w4a8-stage1
OF=$H/models/DeepSeek-V4.1-Flash
OUT=$H/models/out/v41-w4a8-dspark
PY=$P/env/mslim-venv/bin/python
echo "[dspark] dry-run"
$PY "$P/scripts/make_dspark_ckpt.py" --quant-dir "$QF" --official-dir "$OF" --out-dir "$OUT" --dry-run
echo "[dspark] real run"
time $PY "$P/scripts/make_dspark_ckpt.py" --quant-dir "$QF" --official-dir "$OF" --out-dir "$OUT"
echo "[dspark] verify"
$PY "$P/scripts/make_dspark_ckpt.py" --quant-dir "$QF" --official-dir "$OF" --out-dir "$OUT" --verify-only
echo "[DSPARK-CKPT-OK] $OUT"
