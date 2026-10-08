#!/usr/bin/env bash
# =============================================================================
# install.sh —— 把本包装进一个**已存在的 vllm-ascend 容器**
#
# 两种交付面（**逐字节等价**，见 README §3）：
#   ① baked ：把 payload 拷进容器真实路径（生产 / 跨机复现，不需要仓库）
#   ② mount ：起服时 -v 挂仓库文件（开发 / 改代码 / 做单变量）
#   本脚本实现 ①。② 见 launch/with-fusion.env。
#
# 用法（在目标机上）：
#   bash install.sh <container>            # 安装（幂等；重复跑会覆盖为包内版本）
#   bash install.sh <container> --dry-run  # 只打印将要做什么
#   bash install.sh <container> --rollback # 换回安装前的备份
#
# 安装动作（全部可回滚）：
#   A1  /vllm-workspace/3out_opp/                       ← payload/opp/    （纯新增）
#   A2  .../vllm_ascend/vllm_ascend_C...so              ← payload/so/     （先备份 .orig-*）
#   B1  .../models/deepseek_v4/model.py                 ← payload/py/     （先备份 .bak-fusion-*）
#   B2  .../attention/dsa_v41.py                        ← payload/py/     （先备份 .bak-fusion-*）
#   C   清 __pycache__（否则 Python 仍加载旧 .pyc）
#
# ★ 本脚本**不重启服务**、**不改启动脚本**。装完需要人工重启并带上环境变量
#   （脚本末尾会打印确切的两行 export）。
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="$HERE"
# 兼容两种摆法：在 deploy 目录里跑（payload/ 子目录）或已被拷进 payload/ 里跑
if [ ! -d "$PAYLOAD/opp" ] && [ -d "$HERE/payload/opp" ]; then PAYLOAD="$HERE/payload"; fi

CT="${1:-}"
DRY=0
ROLLBACK=0
shift || true
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)  DRY=1; shift ;;
        --rollback) ROLLBACK=1; shift ;;
        *) echo "未知参数 $1" >&2; exit 2 ;;
    esac
done
[ -n "$CT" ] || { echo "用法: bash install.sh <container> [--dry-run|--rollback]" >&2; exit 2; }

SO_NAME=vllm_ascend_C.cpython-312-aarch64-linux-gnu.so
ASC=/vllm-workspace/vllm-ascend/vllm_ascend
OPP=/vllm-workspace/3out_opp

say() { printf '\033[1m[install]\033[0m %s\n' "$*"; }
die() { printf '\033[1m[install][FAIL]\033[0m %s\n' "$*" >&2; exit 1; }
run() { if [ "$DRY" = 1 ]; then echo "    (dry) $*"; else docker exec "$CT" bash -lc "$*"; fi; }

docker inspect "$CT" >/dev/null 2>&1 || die "找不到容器 $CT"
[ "$DRY" = 1 ] || [ -d "$PAYLOAD/opp" ] || die "payload 不完整（缺 opp/）。先跑 build_payload.sh"

# ---------------------------------------------------------------- 前置检查
# 基础镜像的 opp/vendors 必须是空的：本包的 ASCEND_CUSTOM_OPP_PATH 会**指向我们的
# stage**，若基础镜像自带 vendor 算子且该变量是"替换式"语义，就会把它们屏蔽掉。
if [ "$DRY" = 1 ]; then
    say "前置检查：（dry）跳过「官方 vendors 是否为空」的探测"
else
    say "前置检查：基础镜像是否自带 OPP vendor 算子…"
    n_vendor="$(docker exec "$CT" bash -lc 'ls -d /usr/local/Ascend/cann-9.1.0/opp/vendors/*/ 2>/dev/null | wc -l' | tr -d '[:space:]')"
    n_vendor="${n_vendor:-0}"
    say "  官方 vendors 目录数 = $n_vendor（期望 0）"
    if [ "$n_vendor" != "0" ]; then
        say "  ⚠ 基础镜像自带 vendor 算子。请先确认 ASCEND_CUSTOM_OPP_PATH 是追加式语义，"
        say "    否则本包会屏蔽官方 vendor。确认后可加 SKIP_VENDOR_CHECK=1 跳过。"
        [ "${SKIP_VENDOR_CHECK:-0}" = "1" ] || die "拒绝在未确认的情况下安装（SKIP_VENDOR_CHECK=1 强制继续）"
    fi
fi

# ---------------------------------------------------------------- 回滚
if [ "$ROLLBACK" = 1 ]; then
    say "回滚…"
    run "set -e
      F=$ASC/$SO_NAME
      B=\$(ls -1t \$F.orig-* 2>/dev/null | head -1)
      if [ -n \"\$B\" ]; then cp -f \"\$B\" \"\$F\"; echo \"  .so  <- \$B\"; else echo '  WARN 没找到 .so 备份'; fi
      for f in $ASC/models/deepseek_v4/model.py $ASC/attention/dsa_v41.py; do
        b=\$(ls -1t \$f.bak-fusion-* 2>/dev/null | head -1)
        if [ -n \"\$b\" ]; then cp -f \"\$b\" \"\$f\"; echo \"  \$f <- \$b\"; else echo \"  WARN 没找到 \$f 的备份\"; fi
      done
      find $ASC -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
      echo '  已回滚。仍需重启服务并去掉 ASCEND_CUSTOM_OPP_PATH / V41_LNORM_FUSE'"
    exit 0
fi

# ---------------------------------------------------------------- 备份
TS="$(date +%H%M%S)"
say "备份现有文件（时间戳 $TS）…"
run "set -e
  F=$ASC/$SO_NAME
  [ -f \$F.orig-$TS ] || cp -a \$F \$F.orig-$TS
  for f in $ASC/models/deepseek_v4/model.py $ASC/attention/dsa_v41.py; do
    [ -f \$f.bak-fusion-$TS ] || cp -a \$f \$f.bak-fusion-$TS
  done
  echo '  备份完成'"

# ---------------------------------------------------------------- 拷贝
if [ "$DRY" = 0 ]; then
    say "拷入 payload…"
    docker exec "$CT" bash -lc "rm -rf $OPP; mkdir -p $OPP"
    # ★ 用 tar 管道而不是 `docker cp <src>/. <ct>:<dir>/`：
    #   docker cp 往新建目录 / 带符号链接的目录拷时会报
    #   `evalSymlinksInScope: ... is not in ...`（实测两次），tar 没这个问题。
    tar -cf - -C "$PAYLOAD/opp" . | docker exec -i "$CT" tar -C "$OPP" -xf - \
        || die "拷入 OPP 失败"
    docker cp "$PAYLOAD/so/$SO_NAME" "$CT:$ASC/$SO_NAME" || die "拷入 .so 失败"
    docker cp "$PAYLOAD/py/model.py" "$CT:$ASC/models/deepseek_v4/model.py" || die "拷入 model.py 失败"
    docker cp "$PAYLOAD/py/dsa_v41.py" "$CT:$ASC/attention/dsa_v41.py" || die "拷入 dsa_v41.py 失败"
    docker exec "$CT" bash -lc "
      chmod -R u+rwX,go+rX $OPP 2>/dev/null || true
      chmod 755 $ASC/$SO_NAME
      find $ASC -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true"
else
    say "(dry) 将拷贝 opp/ → $OPP ，so/ 与 py/ 各就位，并清 __pycache__"
fi

# ---------------------------------------------------------------- 后置校验
say "后置校验（逐文件 sha256 比对：包内 vs 容器）…"
if [ "$DRY" = 1 ]; then
    say "  (dry) 跳过"
else
    fail=0
    while read -r want rel; do
        [ -n "$rel" ] || continue
        case "$rel" in
            opp/*)  tgt="$OPP/${rel#opp/}" ;;
            so/*)   tgt="$ASC/$SO_NAME" ;;
            py/model.py)    tgt="$ASC/models/deepseek_v4/model.py" ;;
            py/dsa_v41.py)  tgt="$ASC/attention/dsa_v41.py" ;;
            *) continue ;;
        esac
        got="$(docker exec "$CT" sha256sum "$tgt" 2>/dev/null | awk '{print $1}')"
        if [ "$got" != "$want" ]; then
            echo "  ✗ $rel  (期望 ${want:0:12}… 实得 ${got:0:12}…)"
            fail=$((fail+1))
        fi
    done < "$PAYLOAD/PAYLOAD.sha256"
    [ "$fail" = 0 ] || die "有 $fail 个文件与包内不一致"
    say "  ✓ 全部一致"
fi

cat <<EOF

$(say "安装完成。还需要**人工**做一件事：带下面两个变量重启服务。")

  export ASCEND_CUSTOM_OPP_PATH=$OPP/vendors/custom_transformer
  export V41_LNORM_FUSE=1

  （可选，配套的融合点 A，用官方算子、不依赖本包：export V41_QNORM_FUSE=1）

  重启后自检：
    curl -s http://127.0.0.1:<port>/health          # 期望 200
    tr '\\0' '\\n' < /proc/\$(pgrep -f 'vllm serve' | head -1)/environ | grep -E 'V41_LNORM_FUSE|ASCEND_CUSTOM_OPP_PATH'

  回滚：bash install.sh $CT --rollback
EOF
