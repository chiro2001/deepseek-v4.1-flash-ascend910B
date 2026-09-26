#!/usr/bin/env bash
# [CED-DSPARK] 2026-09-26：只重启 CED 的 D 侧，把 SPEC 从 0 改成 1（DSpark 开启），
# 其余每一个参数都与**当前交付口径的 D** 逐项相同（从 /proc/<vllm pid>/environ 抄的），
# 因此任何行为差异都能归因到 SPEC/DSpark 这一个变量。
#
#   ARM=eager_draft   → DRAFT_GRAPH=0（草稿 eager，先验正确性）
#   ARM=draft_graph   → DRAFT_GRAPH=1（草稿入图，验性能）
#   ARM=delivery      → SPEC=0，回到验收口径（对照组）
#
# P 与 proxy 不动。用法：
#   ARM=eager_draft bash ~/tmp/20260926/dspark/experiments/dspark/launch_d_dspark.sh
set -uo pipefail
PKG=/home/l00886679/tmp/20260926/dspark
ARM=${ARM:-eager_draft}
case "$ARM" in
  eager_draft) SPEC_ARM=1; DRAFT_GRAPH_ARM=0 ;;
  draft_graph) SPEC_ARM=1; DRAFT_GRAPH_ARM=1 ;;
  delivery)    SPEC_ARM=0; DRAFT_GRAPH_ARM=0 ;;
  *) echo "未知 ARM=$ARM"; exit 2 ;;
esac
cd "$PKG" || exit 1
STAMP=$(date +%m%d_%H%M%S)
RUN_ID=ced_d4b_${ARM}_$STAMP
LOG=$PKG/results/d_${ARM}_$STAMP.log
echo "[dspark-d] $(date -Is) ARM=$ARM STAMP=$STAMP RUN_ID=$RUN_ID"

# 只清 D：P 与 proxy 必须活着（proxy 名字也以 dsv41-ced 开头，务必别误伤）。
docker ps -a --format '{{.Names}}' | grep -E '^dsv41-ced-d' | xargs -r docker rm -f >/dev/null 2>&1 || true
for i in $(seq 1 60); do
  n=$(npu-smi info 2>/dev/null | grep -c VLLMWorker)
  echo "   drain $i: VLLMWorker=$n（期待 8：只剩 P 的 8 个）"
  [ "$n" -le 8 ] && break
  sleep 10
done

( export MODEL=/home/l00886679/models/out/v41-flat-verify3
  export SERVED_NAME=deepseek-v41-ced-pd PATCH_MODE=mount
  export MAX_LEN=1048576 MAX_SEQS=4 BAT_TOKENS=8192 GPU_UTIL=0.92 BLOCK=128
  export KV_DTYPE=bfloat16 ENGRAM=1 ENGRAM_DEVICE_INDEX=0 CPU_BIND=0 VISION=1
  # ↓↓↓ 与交付口径 D 逐项一致，只有 SPEC/DRAFT_GRAPH 是变量
  export SPEC=$SPEC_ARM SP_TOKENS=${SP_TOKENS_ARM:-7} DRAFT_GRAPH=$DRAFT_GRAPH_ARM
  # [STATIC_KERNEL] 交付/历史口径都曾是 0（那是排障降级值，不是性能设定）。
  # 历史 23.9ms 那一发用的是 1；cannbot verdict 实测 static kernel 128K −2.87ms。
  # 用 STATIC_KERNEL_ARM 覆盖即可做单变量实验；起服后**必须核对真的编译**
  # （compile start 计数 > 0、无相关 warning），不能只看 env 传进去了。
  export STATIC_KERNEL=${STATIC_KERNEL_ARM:-0} NPUGRAPH_EX=1 PROFILE=1
  export MULTISTREAM=0 DSA_OVERLAP=0 PREFIX=0
  export KV_CACHE_MEMORY_BYTES=15728022528
  export CED_EXPERIMENTAL_GRAPH=1 V41_CED_GRAPH_PROMPT_TAIL_EAGER=1 V41_CED_SWA_CLIP=1
  export V41_CED_ALLOW_DSPARK=1
  # [DRAFT-FOUR-PIECE] 草稿入图的四件套。A2 生产口径是
  # `DSPARK_CAPTURE_VALUE_FIX=1` + 三个代码默认 1；这里显式写出来，
  # 免得将来"静默退化"时又要靠 A 值反推。
  # ⚠️ 注意 `DSPARK_HOIST_CONTEXT_KV` 保持 A2 口径 0：它是同一根因的**另一条**
  # 修法（图外写 context KV）。若这一臂 A≈1.0，下一轮单变量就是把它改成 1。
  if [ "$DRAFT_GRAPH_ARM" = 1 ]; then
    export DSPARK_CAPTURE_VALUE_FIX=${DSPARK_CAPTURE_VALUE_FIX:-1}
    export DSPARK_CAPTURE_NCTX_FIX=${DSPARK_CAPTURE_NCTX_FIX:-1}
    export DSPARK_SWA_INDICES_RESIDENT=${DSPARK_SWA_INDICES_RESIDENT:-1}
    export DSPARK_DISPATCH_QUERY_LEN_FIX=${DSPARK_DISPATCH_QUERY_LEN_FIX:-1}
    export DSPARK_DRAFT_METADATA_MODE=${DSPARK_DRAFT_METADATA_MODE:-sync}
    export DSPARK_GRAPH_CAPTURE_METADATA=1
  fi
  # [SLOT-MAP-FUSED] decode 稳态每步 `_compute_slot_mapping_kernel` 启动
  # KV 组数次（D 侧 13 组）。单次 device 只有 2.5–3.2 µs，但每次要付 ~65–70 µs
  # 的 host/排队代价 ⇒ 每步约 0.8 ms 的 host 串行。融合成一次二维 grid 启动。
  #  0/off = stock 逐组；verify = 两条路径都跑并逐元素比对；1/on = 融合
  export V41_SLOT_MAP_FUSED=${V41_SLOT_MAP_FUSED:-0}
  export NAME=dsv41-ced-d4b RUN_ID=$RUN_ID
  export PORT=18991 KV_PORT=19091 DEVS="8 9 10 11 12 13 14 15"
  mkdir -p "$PKG/results/$RUN_ID"
  exec bash scripts/serve_a3_ced_pd.sh decode
) > "$LOG" 2>&1 &

for i in $(seq 1 180); do
  # 不写 `|| echo 000`（会拼成 000000）；见 AGENTS.md §3.2
  h=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:18991/health 2>/dev/null)
  h=${h:-000}
  [ "$h" = "200" ] && { echo "[dspark-d] D health=200 after ${i}0s"; break; }
  [ $((i % 6)) -eq 0 ] && echo "   ... D=${i}0s h=$h"
  sleep 10
done

echo "[dspark-d] 判据:"
printf "   SPEC/SP_TOKENS   : %s\n" "$(grep -ao 'num_speculative_tokens[^,}]*' "$PKG/results/$RUN_ID/serve.log" 2>/dev/null | head -1)"
printf "   draft graph      : %s\n" "$(grep -ac 'DSPARK_GRAPH_CAPTURE_METADATA\|dspark-graph-capture' "$PKG/results/$RUN_ID/serve.log" 2>/dev/null)"
printf "   CED upper SWA组  : %s\n" "$(grep -a 'upper SWA groups' "$PKG/results/$RUN_ID/serve.log" 2>/dev/null | tail -1)"
printf "   num_blocks       : %s\n" "$(grep -ao 'num_blocks: [0-9]*' "$PKG/results/$RUN_ID/serve.log" 2>/dev/null | head -1)"
printf "   KV cache groups  : %s\n" "$(grep -ao 'kv_cache_groups[^,]*' "$PKG/results/$RUN_ID/serve.log" 2>/dev/null | head -1)"
echo "[dspark-d] READY $(date -Is) LOG=$LOG"
