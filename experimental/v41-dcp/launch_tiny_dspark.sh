#!/usr/bin/env bash
# =============================================================================
# tiny-dspark 夹具：TP2 + DCP2 + DSpark，**2–3 分钟起服**（vs 8 卡 8–10 分钟）
#
# 用途：迭代 DSpark × DCP 的集成 bug（见
#   docs/HANDOVER-20261001-DSPARK-DCP.md 与 docs/V41-DSPARK-X-DCP-BLOCKER-20261001.md）。
#
# 为什么要它：8 卡实例每轮要 8–10 分钟，而且开了 DSpark 后第一个请求就崩；
# tiny 只需要一个 **config 变体**（tiny 是 LOAD_FORMAT=dummy，没有任何权重文件）
# ⇒ "DSpark 版 tiny" = 改 4 个 config 字段，代价≈0。
#
# 前置（已建好，勿重复）：
#   ~/models/out/v41-tiny-dspark   —— 由 v41-tiny 派生，只改了：
#       text_config.num_nextn_predict_layers    0    -> 3
#       text_config.dspark_target_layer_ids     []   -> [37,38,39]
#       text_config.dspark_n_routed_experts     128  -> 8
#       text_config.dspark_num_experts_per_tok  3    -> 2
#     派生记录见该目录的 dspark_derivation.json。
#
# 用法（a3-21）：
#   bash launch_tiny_dspark.sh              # 默认 chip 2/3，端口 19310
#   SPEC=0 bash launch_tiny_dspark.sh       # 对照组（不开 DSpark）
set -uo pipefail

DCP=${DCP:-2}
DEVS=${DEVS:-"2 3"}
PORT=${PORT:-19310}
NAME=${NAME:-dsv41-tinyspark}
SPEC=${SPEC:-1}
SP_TOKENS=${SP_TOKENS:-7}
DRAFT_GRAPH=${DRAFT_GRAPH:-1}
OUTROOT=${OUTROOT:-$HOME/tmp/tinyspark}
MODEL=${MODEL:-$HOME/models/out/v41-tiny-dspark}

[ -f "$MODEL/config.json" ] || { echo "[FAIL] 缺 $MODEL/config.json（用 mk_tiny_dspark.py 生成）" >&2; exit 2; }
[ -f "$HOME/tmp/dcp2tiny/launch_dcp2_tiny.sh" ] || { echo "[FAIL] 缺 launch_dcp2_tiny.sh" >&2; exit 2; }

echo "[tiny-dspark] MODEL=$MODEL SPEC=$SPEC SP_TOKENS=$SP_TOKENS DRAFT_GRAPH=$DRAFT_GRAPH"
echo "[tiny-dspark] DCP=$DCP DEVS='$DEVS' PORT=$PORT NAME=$NAME OUTROOT=$OUTROOT"

# ★ 顶层 launcher 必须把 SPEC/SP_TOKENS/DRAFT_GRAPH 透传进去 —— 我们已修掉
#   launch_dcp2_tiny.sh 里 `export SPEC=0` 的硬编码（它会静默吞掉 DSpark）。
cd "$HOME/tmp/dcp2tiny" || exit 1
setsid nohup env \
  MODEL="$MODEL" DCP="$DCP" DEVS="$DEVS" PORT="$PORT" NAME="$NAME" \
  OUTROOT="$OUTROOT" SPEC="$SPEC" SP_TOKENS="$SP_TOKENS" DRAFT_GRAPH="$DRAFT_GRAPH" \
  RUN_ID="tinyspark_dcp${DCP}_$(date +%m%d_%H%M%S)" \
  nohup bash launch_dcp2_tiny.sh > "$OUTROOT.launch.log" 2>&1 < /dev/null &

echo "[tiny-dspark] 已提交；日志 $OUTROOT.launch.log"
echo "[tiny-dspark] 轮询： for i in \$(seq 1 20); do curl -s --noproxy '*' -m 4 -o /dev/null -w '%{http_code}\\n' http://127.0.0.1:$PORT/health; sleep 15; done"
