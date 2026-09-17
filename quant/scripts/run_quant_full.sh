#!/usr/bin/env bash
# 全量（40 层）W4A8 量化。前置：make_full_model.py 生成可量化目录；权重下载完成。
set -euo pipefail
H=/home/user; V=$H/projects/dsv41/env/mslim-venv; R=$H/projects/dsv41/src/msmodelslim
SRC=${SRC:-$H/models/DeepSeek-V4.1-Flash}
MODEL=${MODEL:-$H/models/DeepSeek-V4.1-Flash-quant}
SAVE=${SAVE:-$H/models/out/v41-w4a8}
MTP=${MTP:-0}          # 阶段一：0（关 MTP）；阶段三再开 3
ENGRAM=${ENGRAM:-off}  # 阶段一：off
# 1) 装配目录
if [ ! -f "$MODEL/model.safetensors.index.json" ]; then
  $V/bin/python $H/projects/dsv41/scripts/make_full_model.py --src "$SRC" --dst "$MODEL" \
    --layers 40 --mtp "$MTP" --engram "$ENGRAM" --calib-max-seq-len 8192
fi
# 2) 量化（先用 12-15 里的 1 张卡：device_id 0）
mkdir -p "$SAVE"
$V/bin/msmodelslim quant \
  --model_path "$MODEL" --save_path "$SAVE" \
  --config $R/lab_practice/deepseek_v41/deepseek_v4_1_flash_w4a8.yaml \
  --device npu --device_id ${DEVICE_ID:-0} \
  --model_type deepseek_v41 --trust_remote_code true --log_level info
echo "[ok] $SAVE"
