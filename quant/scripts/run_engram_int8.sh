#!/usr/bin/env bash
# Engram FP8 -> INT8(group32, 二次幂 fp32 scale) 转换（容器内执行）
set -uo pipefail
H=/home/user; P=$H/projects/dsv41
SRC=${SRC:-$H/models/DeepSeek-V4.1-Flash}
DST=${DST:-$H/models/out/engram-int8}
CHUNK=${CHUNK:-2097152}
PY=$P/env/mslim-venv/bin/python
mkdir -p "$DST"
echo "[engram] src=$SRC dst=$DST chunk=$CHUNK"
nproc
time "$PY" "$P/scripts/engram_convert_int8.py" --src "$SRC" --dst "$DST" --chunk "$CHUNK" \
      --keys layers.1.engram.embed layers.14.engram.embed
echo "[engram] ls:"; ls -la "$DST"
echo "[ENGRAM-OK] $DST"
