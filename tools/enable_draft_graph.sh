#!/usr/bin/env bash
# =============================================================================
# 【实验开关】DRAFT_GRAPH=1 —— DSpark draft 入 ACLGraph
#
#   ❌ **上卡验证未做**（目标机预计省 draft 派发的 ~24 ms/轮，但正确性未验证）。
#   **默认关闭。** 打开前请先确认已读懂 reports/draft-graph-numinput-fix.md。
#
# 用法：
#   bash /opt/dsv41/tools/enable_draft_graph.sh on      # 安装 draft 版三个整文件
#   bash /opt/dsv41/tools/enable_draft_graph.sh off     # 还原
#   bash /opt/dsv41/tools/enable_draft_graph.sh status
#
# 前置：
#   * cudagraph_mode 必须是 FULL_DECODE_ONLY（本包默认就是）
#   * 起服时需传 DSPARK_DRAFT_METADATA_MODE=sync（serve_a2.sh DRAFT_GRAPH=1 已代劳）
#   * draft 版 dsa_v1.py 是「stock+0004+F3」合并版，会同时覆盖 F3 的 dsa_v1.py
# =============================================================================
set -uo pipefail
A=${A:-/vllm-workspace/vllm-ascend/vllm_ascend}
D=/opt/dsv41/patches/draft
pairs=(
  "$D/dsa_v1.py:$A/attention/dsa_v1.py"
  "$D/dspark_proposer.py:$A/spec_decode/dspark_proposer.py"
  "$D/llm_base_proposer.py:$A/spec_decode/llm_base_proposer.py"
)
case "${1:-status}" in
  on)
    for p in "${pairs[@]}"; do
      src=${p%%:*}; tgt=${p##*:}
      [ -f "$src" ] || { echo "缺 $src"; exit 1; }
      [ -f "$tgt" ] || { echo "缺目标 $tgt"; exit 1; }
      [ -f "$tgt.a2orig" ] || cp -f "$tgt" "$tgt.a2orig"
      cp -f "$src" "$tgt"; python3 -m py_compile "$tgt" || exit 1
      echo "installed $(basename $tgt)"
    done
    echo "已安装 draft 图补丁；重启服务时带上 DRAFT_GRAPH=1（serve_a2.sh 会设 ensure_eager=false）"
    ;;
  off)
    for p in "${pairs[@]}"; do
      tgt=${p##*:}
      [ -f "$tgt.a2orig" ] && { cp -f "$tgt.a2orig" "$tgt"; python3 -m py_compile "$tgt"; echo "reverted $(basename $tgt)"; } \
        || echo "无备份，跳过 $(basename $tgt)"
    done
    ;;
  status)
    for p in "${pairs[@]}"; do
      tgt=${p##*:}
      if [ -f "$tgt.a2orig" ]; then echo "PATCHED   $tgt"; else echo "stock     $tgt"; fi
    done
    ;;
  *) echo "用法: $0 on|off|status"; exit 2 ;;
esac
