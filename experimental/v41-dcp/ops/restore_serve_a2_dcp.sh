#!/usr/bin/env bash
# =============================================================================
# 一键恢复 serve_a2.sh 的 DCP 挂载改动（幂等）
#
# ★ 为什么需要：DCP 挂载机制（V41_DCP_MOUNT 整文件覆盖 + DCP_EXTRA_ENV 透传 +
#   DCP-MOUNT-GUARD 逐个 md5 校验 + `*.so` 白名单）**从未提交进 git**，
#   只存在于 a3-21 的工作区。任何 `git reset --hard` / `git checkout` /
#   `git stash` 都会把它静默抹掉。
#
#   2026-10-01 11:17:39 我执行 `git reset --hard origin/main` 时就是这么丢的，
#   代价是一轮 10 分钟起服 + 撞 `NotImplementedError: V4.1 initial runtime
#   requires PP=DCP=PCP=1`（看起来像 overlay 没生效，实际是挂载块没了）。
#
# 内容来源（按优先级）：
#   1) ~/dcp_durable/serve_a2.sh.dcp-mount-20261001   ← 权威存档（含 KV32 守卫）
#   2) ~/cedpd-repo/scripts/serve_a2.sh.bak_dcpson    ← 旧备份（无 KV32 守卫，兜底）
#
# 用法：bash ~/restore_serve_a2_dcp.sh
# 判据（起服后查，**不要**看"起服成功"）：
#   docker inspect <name> --format {{range .Mounts}}{{.Destination}}{{\n}}{{end}} | grep -c vllm_ascend   # >= 15
#   docker exec <name> md5sum .../spec_decode/llm_base_proposer.py                                          # = a12ededf...
#   grep -a "PP/DCP/PCP guard bypassed" <run>/serve.log                                                      # overlay 生效铁证
# =============================================================================
set -uo pipefail
cd "$HOME/cedpd-repo" || exit 1

DST=scripts/serve_a2.sh
DURABLE=$HOME/dcp_durable/serve_a2.sh.dcp-mount-20261001
DURABLE_SHA=$HOME/dcp_durable/serve_a2.sh.dcp-mount-20261001.sha256

if [ -f "$DST" ] && grep -q "V41_DCP_MOUNT" "$DST" && grep -q "V41-DCP-SO-MOUNT" "$DST"; then
  echo "[ok] $DST 已含完整 DCP 挂载（含 *.so 白名单），无需恢复"
  exit 0
fi

[ -f "$DURABLE" ] || { echo "[FAIL] 缺权威存档 $DURABLE" >&2; exit 2; }
if [ -f "$DURABLE_SHA" ]; then
  (cd "$(dirname "$DURABLE")" && sha256sum -c "$(basename "$DURABLE_SHA")") || {
    echo "[FAIL] 存档 sha256 校验不过，拒绝覆盖" >&2; exit 3; }
fi

cp -p "$DST" "$DST.bak_before_restore.$(date +%H%M%S)" 2>/dev/null || true
cp -p "$DURABLE" "$DST"
bash -n "$DST" || { echo "[FAIL] 恢复后语法检查不过" >&2; exit 4; }
echo "[ok] 已从权威存档恢复 $DST"
echo "     V41_DCP_MOUNT x$(grep -c V41_DCP_MOUNT "$DST")  V41-DCP-SO-MOUNT x$(grep -c V41-DCP-SO-MOUNT "$DST")  KV32-POOL-GUARD x$(grep -c KV32-POOL-GUARD "$DST")"
