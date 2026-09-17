#!/usr/bin/env bash
# =============================================================================
# 【实验开关】MOE_ZERO=1 —— 把 init_routing 未写入的无效行清零
#
#   ❌ 未在本机完成端到端验证（打包时同会话 A/B 仍在跑）。**默认关闭。**
#
# 用法（容器内执行）：
#   bash /opt/dsv41/tools/enable_moe_zero.sh on       # 切到 moezero 版 token_dispatcher
#   bash /opt/dsv41/tools/enable_moe_zero.sh off      # 切回已验证版
#   bash /opt/dsv41/tools/enable_moe_zero.sh status
#
# 注意：moezero 版是 moemask 版的超集（也含 mask-range），切换是**整文件替换**，
#       不要同时挂两个（docker 会报 Duplicate mount point）。
#       切完必须重启服务才生效。
# =============================================================================
set -uo pipefail
A=${A:-/vllm-workspace/vllm-ascend/vllm_ascend}
TGT=$A/ops/fused_moe/token_dispatcher.py
ZERO=/opt/dsv41/patches/files/token_dispatcher_moezero.py
VER=/opt/dsv41/patches/files/token_dispatcher_moemask.py
case "${1:-status}" in
  on)
    [ -f "$ZERO" ] || { echo "缺 $ZERO"; exit 1; }
    [ -f "$TGT.a2orig" ] || cp -f "$TGT" "$TGT.a2orig"
    cp -f "$ZERO" "$TGT"
    python3 -m py_compile "$TGT" || exit 1
    echo "已切到 MOE_ZERO 版；请重启服务，并在运行时把开关写进 /tmp/v41_moe_zero_file（1=开 0=关）"
    echo "  例：docker exec <容器> bash -lc \"printf 1 > /tmp/v41_moe_zero_file\""
    ;;
  off)
    [ -f "$VER" ] || { echo "缺 $VER"; exit 1; }
    cp -f "$VER" "$TGT"; python3 -m py_compile "$TGT" || exit 1
    echo "已切回已验证版（moe-mask-range）"
    ;;
  status)
    md5sum "$TGT" "$ZERO" "$VER" 2>/dev/null
    ;;
  *) echo "用法: $0 on|off|status"; exit 2 ;;
esac
