#!/usr/bin/env bash
# 静态自检：每个 TQue/TBuf 成员都必须**恰好**有一条 InitBuffer。
#
# 为什么需要它（2026-10-01 实测两次踩坑）：
#   · 漏 `qOriF_` 的 InitBuffer => `Get<float>()` 基址是垃圾 =>
#     `The GM address accessed by scalar exceeds 48 bits`（vector core exception）；
#   · 用脚本按行号改代码时误删了 `pre` 的 `qDen_` InitBuffer => 同样是
#     vector core exception，而且**换 chip 后仍复现**，一度误判为"设备坏了"。
set -uo pipefail
F=${1:-op_kernel/v41_merge.asc}
fail=0
for cls in MergePre MergePost; do
  body=$(awk "/^class $cls /,/^};/" "$F")
  members=$(echo "$body" | grep -oE "TQue<[^>]*>[^;]*;|TBuf<[^>]*>[^;]*;" | sed -E "s/.*>[[:space:]]+//; s/;//; s/,/ /g" | tr -s " " "\n" | grep -v "^$")
  inits=$(echo "$body" | grep -oE "InitBuffer\([A-Za-z_0-9]+" | sed "s/InitBuffer(//" | sort -u)
  for m in $members; do
    n=$(echo "$body" | grep -cE "InitBuffer\($m,")
    if [ "$n" -ne 1 ]; then
      echo "[FAIL] $cls: member '$m' has $n InitBuffer (must be exactly 1)"; fail=1
    fi
  done
  for i in $inits; do
    echo "$members" | grep -qx "$i" || { echo "[FAIL] $cls: InitBuffer($i) but $i is not a member"; fail=1; }
  done
done
[ $fail -eq 0 ] && echo "[OK] $F buffer-init self-check passed"
exit $fail
