#!/usr/bin/env bash
set -uo pipefail
H="$HOME/tmp/ab_mkc/run_ab_mkc_1001_215615"
CT=dsv41-tinyspark
CTD=/opt/dsv41/results/ab_mkc_1001_215615
# DCP=1 基线（无 DBO）：与 DBO 臂保持同一并行度
sed -e 's|export KV_ARGS_EXTRA="${KV_ARGS_EXTRA:-}"|export KV_ARGS_EXTRA="--decode-context-parallel-size 1"|' \
    -e 's|^export PROFILE=0|export PROFILE=1|' \
    -e "s|^export PROFILE_DIR=.*|export PROFILE_DIR=$CTD/prof_dcp1|" \
    "$H/inner.sh" > "$H/inner_dcp1.sh"
chmod +x "$H/inner_dcp1.sh"
grep -nE 'KV_ARGS_EXTRA|PROFILE=' "$H/inner_dcp1.sh" | head -4
for pid in $(docker exec "$CT" bash -lc "pgrep -f 'vllm serve|VLLM::EngineCore|VLLM::Worker|resource_tracker'" 2>/dev/null); do
  docker exec "$CT" bash -lc "kill -9 $pid" 2>/dev/null || true
done
sleep 8
docker exec -d "$CT" bash -lc "cd /workspace && bash $CTD/inner_dcp1.sh > $CTD/serve_dcp1.log 2>&1"
echo "[Mcp1] submitted $(date '+%H:%M:%S')"
