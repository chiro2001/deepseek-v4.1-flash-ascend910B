#!/usr/bin/env bash
# 拓扑验证 v4：用 HCCL_NPU_SOCKET_PORT_RANGE 区分两个实例（CANN 的"单卡多进程"方案）
set -uo pipefail
H="$HOME/tmp/ab_mkc/run_ab_mkc_1001_215615"
CT=dsv41-tinyspark
CTD=/opt/dsv41/results/ab_mkc_1001_215615
docker exec $CT bash -lc "pkill -9 -f 'vllm serve'; pkill -9 -f EngineCore; pkill -9 -f 'VLLM::'" 2>/dev/null || true
sleep 12
i=0
for P in 19400 19401; do
  LO=$((60000 + i*1000)); HI=$((LO+800))
  sed -e "s|PORT=19310|PORT=$P|" \
      -e 's|GPU_UTIL=0.85|GPU_UTIL=0.30|' \
      -e 's|KV_CACHE_MEMORY_BYTES=17179869184|KV_CACHE_MEMORY_BYTES=4294967296|' \
      -e 's|export KV_ARGS_EXTRA="${KV_ARGS_EXTRA:-}"|export KV_ARGS_EXTRA="--decode-context-parallel-size 1"|' \
      -e "s|export V41_KV_TIER=off|export V41_KV_TIER=off\nexport HCCL_NPU_SOCKET_PORT_RANGE=$LO-$HI\nexport MASTER_PORT=$((LO+900))|" \
      "$H/inner.sh" > "$H/inner_dual_$P.sh"
  chmod +x "$H/inner_dual_$P.sh"
  grep -nE 'HCCL_NPU_SOCKET_PORT_RANGE|MASTER_PORT' "$H/inner_dual_$P.sh" | head -2
  docker exec -d $CT bash -lc "cd /workspace && bash $CTD/inner_dual_$P.sh > $CTD/serve_dual_$P.log 2>&1"
  echo "  launched $P (range $LO-$HI)"
  i=$((i+1)); sleep 25
done
echo "[dual4] submitted $(date '+%H:%M:%S')"
