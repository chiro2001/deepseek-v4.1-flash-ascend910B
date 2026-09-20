#!/usr/bin/env bash
# =============================================================================
# run_probe_engram_hostreg.sh —— 在 **A2 宿主机**上一条命令跑完 host-register 定界。
#
# 为什么要有它：`probe_engram_hostreg.py` 需要 CANN 的 acl python 绑定，
# 而宿主通常没有；起服的容器又会在失败后被 `serve_a2.sh` 清掉。所以这里
# **临时起一个探针容器**（用你已 build 的镜像），跑完自动删。
#
# 用法：
#   MODEL=/home/user/models/out/v41-w4a8-flat \
#   IMAGE=dsv41-a2:v8 \
#   DEVS="0 1 2 3 4 5 6 7" \
#   bash run_probe_engram_hostreg.sh
#
#   # 加并发测试（检验"8 rank 各注册完整 206 GiB 是否装得下"）：
#   FANOUT=8 bash run_probe_engram_hostreg.sh
#
#   # ★ 分级并发（推荐：先用小规模快速定位阈值，再往上加）
#   #   每个子进程只注册前 K 个文件（按文件名排序）
#   FANOUT=8 PROBE_CHILD_FILES=1 bash run_probe_engram_hostreg.sh   # 8×11.4 GiB
#   FANOUT=8 PROBE_CHILD_FILES=2 bash run_probe_engram_hostreg.sh   # 8×103  GiB
#   FANOUT=8 bash run_probe_engram_hostreg.sh                       # 8×206  GiB（最慢）
#
# 前提：手机器上已有镜像（`docker images | grep dsv41-a2`）。
# 不占 NPU 算力：脚本不加载模型，只调 aclrtHostRegister。
# =============================================================================
set -uo pipefail

MODEL=${MODEL:?请给 MODEL=<模型目录>（其下有 engram_int8/）}
IMAGE=${IMAGE:-dsv41-a2:v8}
DEVS=${DEVS:-0 1 2 3 4 5 6 7}
FANOUT=${FANOUT:-0}
QUICK=${QUICK:-}
NAME=${NAME:-dsv41-hostreg-probe}
PROBE=${PROBE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/probe_engram_hostreg.py}

die() { printf '\n[probe][FAIL] %s\n' "$*" >&2; exit 2; }
say() { printf '\n[probe-shell] %s\n' "$*"; }

[ -f "$PROBE" ] || die "找不到探针脚本：$PROBE"
[ -d "$MODEL" ] || die "MODEL 不是目录：$MODEL"
command -v docker >/dev/null 2>&1 || die "docker 不可用"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "镜像不存在：$IMAGE（先 docker images 看一下）"

# 组装 device 参数（与 serve_a2.sh 同形）
DEV_ARGS=()
for d in $DEVS; do DEV_ARGS+=(--device "/dev/davinci$d"); done
ARGS=()
[ "$FANOUT" != "0" ] && ARGS+=(--fanout "$FANOUT")
[ "$QUICK" = "1" ] && ARGS+=(--quick)

say "镜像=$IMAGE  模型=$MODEL  卡=$DEVS  并发=$FANOUT"
say "临时容器名=$NAME（跑完自动删除；如已存在会先清理同名容器）"

docker rm -f "$NAME" >/dev/null 2>&1 || true

# ⚠️ 【关键】engram 目录必须挂 **:rw**：`HostMappedSafetensors` 用
#    `os.open(O_RDWR)` + `MAP_SHARED`，`:ro` 挂载会先报 EROFS(Errno 30)，
#    那样根本走不到 host_register，测出来的是**假结果**（本脚本 v1 就栽在这里）。
docker run --rm --name "$NAME" --privileged --network host --shm-size 16g \
  --ulimit memlock=-1 \
  "${DEV_ARGS[@]}" \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v "$MODEL":"$MODEL":ro \
  -v "$MODEL/engram_int8":"$MODEL/engram_int8":rw \
  -v "$PROBE":/tmp/probe_engram_hostreg.py:ro \
  -e ASCEND_RT_VISIBLE_DEVICES="$(echo $DEVS | tr ' ' ',')" \
  -e PROBE_HOLD_S="${PROBE_HOLD_S:-3}" \
  -e PROBE_CHILD_FILES="${PROBE_CHILD_FILES:-0}" \
  -w /workspace \
  "$IMAGE" \
  bash -lc "python3 /tmp/probe_engram_hostreg.py --model-dir '$MODEL' ${ARGS[*]:-}"
rc=$?

say "退出码=$rc（0=全过 / 2=找到失败点 / 1=探针自身出错）"
exit $rc
