#!/usr/bin/env bash
set -uo pipefail
H="$HOME/tmp/ab_mkc/run_ab_mkc_1001_215615"
CT=dsv41-tinyspark
CTD=/opt/dsv41/results/ab_mkc_1001_215615
sed -e 's/DSA_OVERLAP=1/DSA_OVERLAP=0/' \
    -e 's|^export PROFILE=0|export PROFILE=1|' \
    -e "s|^export PROFILE_DIR=.*|export PROFILE_DIR=$CTD/prof_nodsa|" \
    "$H/inner.sh" > "$H/inner_nodsa.sh"
chmod +x "$H/inner_nodsa.sh"
echo "--- 改动确认 ---"
grep -nE 'DSA_OVERLAP|PROFILE' "$H/inner_nodsa.sh" | head -5
for pid in $(docker exec "$CT" bash -lc "pgrep -f 'vllm serve|VLLM::EngineCore|VLLM::Worker|resource_tracker'" 2>/dev/null); do
  docker exec "$CT" bash -lc "kill -9 $pid" 2>/dev/null || true
done
sleep 8
echo "[B] old killed, launching..."
docker exec -d "$CT" bash -lc "cd /workspace && bash $CTD/inner_nodsa.sh > $CTD/serve_nodsa.log 2>&1"
echo "[B] submitted $(date '+%H:%M:%S')"
