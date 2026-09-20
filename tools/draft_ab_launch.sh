#!/usr/bin/env bash
# 起一个**单 chip**容器跑 DSpark draft 前向的 eager-vs-graph 数值对照。
# 在 A3-node1 上执行。只用自己名字前缀的容器：dsg-draft-ab。
#   bash draft_ab_launch.sh 3
set -euo pipefail

CHIP=${CHIP:-3}
NAME=${NAME:-dsg-draft-ab}
IMAGE=${IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3}
PKG=${PKG:-/home/user/projects/dsv41-release}
WORK=${WORK:-$PKG/lite-runs/draft-ab}
MODEL=${MODEL:-/home/user/models/out/v41-w4a8-dspark}

mkdir -p "$WORK"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "[launch] $NAME 已存在，直接 start"
  docker start "$NAME" >/dev/null
else
  docker run -d --name "$NAME" \
    --privileged --network host --ipc host --shm-size 64g \
    --device "/dev/davinci${CHIP}" \
    --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v "$WORK:/work" \
    -v "$MODEL:$MODEL:ro" \
    -v "$(dirname "$(dirname "$MODEL")"):$(dirname "$(dirname "$MODEL")"):ro" \
    -v "$PKG/patches/files/draft:/draft-patch:ro" \
    -v "$PKG/patches/files/draft/dspark_proposer.py:/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py:ro" \
    -v "$PKG/patches/files/draft/llm_base_proposer.py:/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py:ro" \
    -v "$PKG/patches/files/draft/dsa_v1.py:/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v1.py:ro" \
    -e "ASCEND_RT_VISIBLE_DEVICES=${CHIP}" \
    -e VLLM_ADMISSION_GATE=1 \
    -w /work \
    "$IMAGE" bash -lc "sleep infinity"
fi

echo "[launch] $NAME 状态："
docker ps --filter "name=$NAME" --format '  {{.ID}} {{.Status}} {{.Image}}'
echo "[launch] 自检（应当只看到 1 张卡）："
docker exec "$NAME" bash -lc 'python3 -c "import torch,torch_npu;print(\"device_count\",torch.npu.device_count())" 2>&1 | tail -1'
