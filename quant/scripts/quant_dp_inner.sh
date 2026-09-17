#!/usr/bin/env bash
set -euo pipefail
H=/home/user; P=$H/projects/dsv41; V=$P/env/mslim-venv; R=$P/src/msmodelslim
MODEL=${MODEL:-$H/models/DeepSeek-V4.1-Flash-partial7}
SAVE=${SAVE:-$H/models/out/partial7-w4a8-dp16}
DEVICE_IDS=${DEVICE_IDS:-"0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"}
CFG=${CFG:-$R/lab_practice/deepseek_v41/deepseek_v4_1_flash_w4a8.yaml}
rm -rf "$SAVE"; mkdir -p "$SAVE"
$V/bin/msmodelslim quant \
  --model_path "$MODEL" --save_path "$SAVE" --config "$CFG" \
  --device npu --device_id $DEVICE_IDS \
  --model_type deepseek_v41 --trust_remote_code true --log_level info
echo "[QUANT-OK] $SAVE"
