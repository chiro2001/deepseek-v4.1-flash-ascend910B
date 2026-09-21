#!/usr/bin/env bash
# =============================================================================
# 【一键使能】让 codex（OpenAI Responses API 客户端）能直接连本包起的 vLLM
#
#   背景：本包的 `tokenizer_mode=deepseek_v41` 用的是 vllm-ascend 的
#   `patch_deepseek_v41_frontend/encoding.py`，它只认 chat-completions 的块词汇表
#   （`text` / `tool_result` / `image_url`），而 codex 发的是 Responses 的
#   （`input_text` / `output_text` / `input_image`）。由此产生三个真实缺陷：
#
#     ① `input_text` 块被渲染成**字面量** `[Unsupported input_text]`
#        —— HTTP 200，但**用户的话根本没进模型**（静默损坏）；
#     ② codex 把系统指令放在 `developer` 角色里，而该角色要求 content 非空
#        —— 直接 `AssertionError` ⇒ **HTTP 500**；
#     ③ `<｜User｜>` / `<｜Assistant｜>` 等控制 token 能从正文注入
#        —— 实测能**伪造轮次边界**（prompt injection 通道）。
#
#   本脚本安装的版本修掉这三条（+105/−5 行，见 patches/files/.../encoding.py.diff），
#   并且**不影响 chat-completions 的老行为**（已跑通 53 + 51 项单测）。
#
# 用法（容器内执行；served 脚本默认起名叫 dsv41-a2 / dsv41-a3）：
#   bash /opt/dsv41/tools/enable_codex_responses.sh on       # 安装（自动备份原文件）
#   bash /opt/dsv41/tools/enable_codex_responses.sh off      # 还原
#   bash /opt/dsv41/tools/enable_codex_responses.sh status   # 看当前状态
#
# 从宿主一键装进正在跑的容器：
#   docker exec dsv41-a2 bash /opt/dsv41/tools/enable_codex_responses.sh on
#
# ⚠️ 改的是**容器可写层里的文件** ⇒ 生效需要**重启服务**（约 15 分钟）；
#    容器重建/换镜像后会丢失，届时重跑本脚本即可。
# =============================================================================
set -uo pipefail

A=${A:-/vllm-workspace/vllm-ascend/vllm_ascend}
TGT="$A/patch/platform/patch_deepseek_v41_frontend/encoding.py"
SRC=${SRC:-/opt/dsv41/patches/patch_deepseek_v41_frontend/encoding.py}
BAK="$TGT.a2orig"

# 改后版本的指纹（用于 status 与幂等判断）
WANT_MD5=${WANT_MD5:-c20ee3b61bc02a6f4d6f9b9853be5ce4}

md5of() { md5sum "$1" 2>/dev/null | cut -d' ' -f1; }

case "${1:-status}" in
  on)
    [ -f "$SRC" ] || { echo "❌ 缺补丁源文件 $SRC"; echo "   （应随发布包一起分发：patches/files/patch_deepseek_v41_frontend/encoding.py）"; exit 1; }
    [ -f "$TGT" ] || { echo "❌ 目标不存在 $TGT —— 该镜像不是 deepseek_v41 前端？"; exit 1; }

    cur=$(md5of "$TGT")
    if [ "$cur" = "$WANT_MD5" ]; then
      echo "✅ 已经是修补版（md5=$cur），无需重复安装"
      exit 0
    fi

    # 只在"当前是原版"时备份，避免用已打补丁的文件覆盖备份
    if [ ! -f "$BAK" ]; then
      cp -f "$TGT" "$BAK" && echo "已备份原文件 → $BAK (md5=$(md5of "$BAK"))"
    else
      echo "备份已存在，保留：$BAK (md5=$(md5of "$BAK"))"
    fi

    cp -f "$SRC" "$TGT" || { echo "❌ 复制失败"; exit 1; }
    python3 -m py_compile "$TGT" || { echo "❌ 语法检查失败，回滚"; cp -f "$BAK" "$TGT"; exit 1; }

    now=$(md5of "$TGT")
    echo "installed $(basename "$TGT")  md5=$now"
    if [ "$now" != "$WANT_MD5" ]; then
      echo "⚠️  md5 与预期不符（期望 $WANT_MD5）——补丁源文件可能被改过，请核对"
    fi

    # 顺带装一份单测（若目录存在），便于就地回归
    TD="$A/../../tests/ut/patch/platform/deepseek_v41"
    TS=${TS:-/opt/dsv41/patches/patch_deepseek_v41_frontend/test_responses_compat.py}
    if [ -d "$TD" ] && [ -f "$TS" ]; then
      cp -f "$TS" "$TD/" && echo "installed test_responses_compat.py → $TD"
    fi

    cat <<'EOF'

✅ 已安装。**需要重启服务才生效**（模块在起服时被导入）。

重启前请先清理干净（否则会卡在 rtsMallocHost 起不来）：
  docker exec <容器> bash -c 'ps -eo pid,args | grep -E "[V]LLM::|[v]llm serve"'   # 应输出空
  docker stop -t 10 <容器> && docker start <容器>

起服后自检（三条都该 200）：
  curl -s localhost:<port>/v1/responses -H 'Content-Type: application/json' \
    -d '{"model":"<served_name>","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Reply with exactly one word: PONG"}]}],"max_output_tokens":16}'
  # → 修复前会返回 "[Unsupported input_text]"（HTTP 200 但内容是错的）；修复后应返回 PONG
EOF
    ;;

  off)
    if [ -f "$BAK" ]; then
      cp -f "$BAK" "$TGT" && python3 -m py_compile "$TGT" && echo "reverted $(basename "$TGT") (md5=$(md5of "$TGT"))"
      echo "⚠️ 同样需要重启服务才生效"
    else
      echo "无备份 $BAK，跳过（可能从未安装过）"
    fi
    ;;

  status)
    if [ ! -f "$TGT" ]; then
      echo "❌ 目标不存在：$TGT"; exit 1
    fi
    cur=$(md5of "$TGT")
    echo "target : $TGT"
    echo "md5    : $cur"
    if [ "$cur" = "$WANT_MD5" ]; then
      echo "state  : PATCHED (codex/Responses 可用)"
    elif [ -f "$BAK" ] && [ "$cur" = "$(md5of "$BAK")" ]; then
      echo "state  : STOCK (未打补丁；codex 直连会 500 / 静默丢内容)"
    else
      echo "state  : UNKNOWN (既非补丁版也非已知备份版)"
    fi
    [ -f "$BAK" ] && echo "backup : $BAK (md5=$(md5of "$BAK"))"
    ;;

  *) echo "用法: $0 on|off|status"; exit 2 ;;
esac
