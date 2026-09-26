#!/usr/bin/env bash
# 静态核（static kernel）**真的生效**了吗？
#
# 背景：`STATIC_KERNEL=0` 曾是"健康检查三连"的降级值；而 `=1` 本身也可能
# **静默失效** —— torch_npu 的 `static_kernel.py:647` 在
# "多卡但没有 LOCAL_WORLD_SIZE" 时会打一条 warning 并**把功能关掉**
# （见 reports/static-kernel-silent-disable-fix.md）。
# 只看 env 传进去是不够的。
#
# 判据（★ 用"生效"证据，不要用"编译次数"）：
#   ① `static_kernel.py:650` 的 warning 命中数 == 0
#   ② `static shape kernel will be used` 出现次数 > 0   ← **真正的生效证据**
#
# ⚠️ 不要拿 `static kernel compile start` 当判据：它只表示"是否发生了**新**
# 编译"。第二次起服会命中 `cache/skcache/compile_outputs/static_kernel_cache`
# ⇒ 编译次数为 0，但静态核**照样在跑**（已踩，差点误判成静默失效）。
# 两个真实样本：
#   SK=1 首次：used=1 compile=4 torch.compile 2.80s
#   SK=1 再次：used=1 compile=0 torch.compile 1.52s   ← 缓存命中
#   SK=0 两次：used=0 compile=0 torch.compile 0.20s
#
# 用法：bash check_static_kernel.sh <run_dir>
set -uo pipefail
R=${1:?用法: check_static_kernel.sh <run_dir>}
L=$R/serve.log
[ -f "$L" ] || { echo "[sk] 找不到 $L"; exit 2; }
# ⚠️ 不能写 `grep -c ... || echo 0`：grep 无匹配时会**先打印 0 再退出 1**，
# 于是 `|| echo 0` 又追加一个 0，变量变成 "0\n0"（已踩）。
# 用 `grep -c ... || true` 或直接靠 grep 的 stdout。
warn=$(grep -ac "static_kernel.py:650" "$L" 2>/dev/null || true)
start=$(grep -ac "static kernel compile start" "$L" 2>/dev/null || true)
warn=${warn:-0}; start=${start:-0}
used=$(grep -ac "static shape kernel will be used" "$L" 2>/dev/null || true)
used=${used:-0}
took=$(grep -ao "torch.compile took [0-9.]*" "$L" 2>/dev/null | tail -1)
lws=$(grep -ao "LOCAL_WORLD_SIZE=[0-9]*" "$R/inner.sh" 2>/dev/null | head -1)
echo "[sk] run=$(basename "$R")"
echo "     ① static_kernel.py:650 warning = $warn                 （必须 0）"
echo "     ② 'static shape kernel will be used' = $used          （必须 > 0 ← 生效证据）"
echo "     ③ 'static kernel compile start' = $start              （0 = 命中编译缓存，正常）"
echo "     ④ $took"
echo "     ⑤ inner.sh $lws"
if [ "$warn" != "0" ]; then
  echo "[sk] ✗ 静态核被**静默禁用**（命中 $warn 次 warning）"
  grep -a "static_kernel.py:650" "$L" | head -2 | sed 's/^/       /'
  exit 1
fi
if [ "$used" = "0" ]; then
  echo "[sk] ✗ 没有 'static shape kernel will be used' ⇒ 静态核没生效"
  [ "$start" != "0" ] && echo "       (但 compile start=$start > 0，矛盾，请人工核对)"
  exit 1
fi
echo "[sk] ✓ 静态核已生效"
