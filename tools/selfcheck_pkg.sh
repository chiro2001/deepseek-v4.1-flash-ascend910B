#!/usr/bin/env bash
# =============================================================================
# selfcheck_pkg.sh —— 包内一致性自检（**起服前先跑，10 秒**）
#
#   bash tools/selfcheck_pkg.sh
#
# 为什么需要：打包/改版时最容易出的一类错是**版本标记不一致** ——
# 例如 build_image.sh 产出 `dsv41-a2:v4`，而 serve_a2.sh 默认去找
# `dsv41-a2:v5` ⇒ 起服直接 "镜像不存在"，白等一场。本脚本把这类问题
# 在 10 秒内抓出来。
#
# 退出码：0 = 通过；1 = 有不一致（必须修）
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
cd "$PKG"

fail=0
ok()   { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; fail=$((fail+1)); }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$*"; }

echo "[selfcheck] 包根: $PKG"

# ---------------------------------------------------------------- 1) 镜像 tag
# 从四个地方抽取 tag，必须完全一致
tag_of() {  # $1=file  $2=regex
  [ -f "$1" ] || { echo ""; return; }
  grep -oE "$2" "$1" 2>/dev/null | head -1
}
TAG_BUILD=$(tag_of scripts/build_image.sh 'IMAGE_TAG:-dsv41-a2:v[0-9]+' | sed 's/.*://')
TAG_SERVE=$(tag_of scripts/serve_a2.sh    'IMAGE:-dsv41-a2:v[0-9]+'    | sed 's/.*://')
TAG_TEST=$(tag_of  scripts/run_test.sh    'IMAGE:-dsv41-a2:v[0-9]+'    | sed 's/.*://')

echo "  镜像 tag: build=$TAG_BUILD serve=$TAG_SERVE run_test=$TAG_TEST"
if [ -z "$TAG_BUILD" ] || [ -z "$TAG_SERVE" ] || [ -z "$TAG_TEST" ]; then
  bad "有脚本里找不到镜像 tag（正则失配？）"
elif [ "$TAG_BUILD" = "$TAG_SERVE" ] && [ "$TAG_SERVE" = "$TAG_TEST" ]; then
  ok "镜像 tag 三处一致：$TAG_BUILD"
else
  bad "镜像 tag 不一致！build_image 产出 '$TAG_BUILD'，但 serve_a2/run_test 找 '$TAG_SERVE'/'$TAG_TEST'"
  echo "        修法：sed -i 's/dsv41-a2:vX/dsv41-a2:vY/g' scripts/*.sh Dockerfile"
fi

# ------------------------------------------------- 2) tag 与包目录名是否对齐
PKGNAME=$(basename "$PKG")
if printf '%s' "$PKGNAME" | grep -qE 'a2_pkg_v([0-9]+)'; then
  WANT="v$(printf '%s' "$PKGNAME" | sed -n 's/.*a2_pkg_v\([0-9]\+\)/\1/p')"
  if [ "$TAG_SERVE" = "$WANT" ]; then
    ok "包目录名与镜像 tag 对齐：$PKGNAME / $TAG_SERVE"
  else
    warn "包目录名 $PKGNAME 暗示 tag $WANT，但脚本用 $TAG_SERVE（不致命，但容易混淆）"
  fi
fi

# ------------------------------------------------- 3) 关键脚本存在 + 语法
for f in scripts/build_image.sh scripts/serve_a2.sh scripts/run_test.sh \
         tools/model_mount_args.sh tools/check_model_dir.sh \
         tools/preflight_a2.sh tools/negative_control.sh \
         build_scripts/00_ensure_pgo.sh \
         tests/t_quote.sh tests/multibatch/multibatch_session.sh; do
  if [ ! -f "$f" ]; then
    bad "缺文件：$f"
  elif bash -n "$f" 2>/dev/null; then
    ok "$f（存在 + 语法通过）"
  else
    bad "$f 语法错误"
  fi
done

# Python 文件要用 py_compile（别拿 bash -n 检查 .py —— v6 第一版就犯过）
for f in tests/t_vision.py tests/t_gsm8k.py tests/acc_eval.py \
         tests/vision_accuracy_check.py tests/p15_stream_curve_filefiller.py \
         tests/multibatch/multibatch_gate.py \
         tools/check_dockerfile.py tools/fisher_recheck.py tools/steep_summary.py \
         tools/model_mount_args.sh; do
  [ -f "$f" ] || { bad "缺文件：$f"; continue; }
  case "$f" in *.py)
    if python3 -c "import ast,sys; ast.parse(open(sys.argv[1],encoding='utf-8').read())" "$f" 2>/dev/null; then
      ok "$f（Python 语法通过）"
    else
      bad "$f Python 语法错误"
    fi ;;
  esac
done

# 必需的数据文件（v5 曾漏打包 p15 → 性能测试全废）
for f in data/hongloumeng.txt data/suffix_quote.txt; do
  [ -f "$f" ] && ok "$f" || bad "缺数据文件：$f"
done

# ------------------------------------------- 4) 软链挂载逻辑确实接在 serve 里
if grep -q 'MODEL_MOUNTS' scripts/serve_a2.sh && \
   ! grep -qE '^\s*-v "\$MODEL:\$MODEL' scripts/serve_a2.sh; then
  ok "serve_a2.sh 用的是 MODEL_MOUNTS（不再单层挂载）"
else
  bad "serve_a2.sh 里仍是单层 -v \$MODEL:\$MODEL（软链会悬空）"
fi

# ------------------------------------------- 5) Dockerfile 续行链（A2 实测踩过的坑）
# 漏一个 `\` 或行内写 `#` 都会让 RUN 提前结束 / 后续命令被注释掉，
# 表现为 "unknown instruction: local" 之类难以定位的报错。
if [ -f Dockerfile ]; then
  if python3 tools/check_dockerfile.py Dockerfile >/tmp/dsck.$$ 2>&1; then
    ok "Dockerfile 续行链合法（$(grep -c '^\s*RUN' Dockerfile) 条 RUN）"
  else
    bad "Dockerfile 续行链有问题："
    sed 's/^/        /' /tmp/dsck.$$ | grep -E "ERROR|问题" | head -6
  fi
  rm -f /tmp/dsck.$$
fi

# ------------------------------------------------- 6) 执行位（缺了也能跑，但要提醒）
_noexec=0
for f in $(find . -name "*.sh" -not -path "./results/*" 2>/dev/null); do
  [ -x "$f" ] || _noexec=$((_noexec+1))
done
if [ "$_noexec" = "0" ]; then ok "所有 .sh 都有执行位"
else warn "$_noexec 个 .sh 缺执行位（不影响——脚本都用 bash 调用）"; fi

# ------------------------------------------------- 7) MANIFEST 自校验（若存在）
if [ -f MANIFEST.sha256 ]; then
  n_total=$(grep -c . MANIFEST.sha256)
  n_bad=$(sha256sum -c MANIFEST.sha256 2>/dev/null | grep -c -v ': OK' || true)
  if [ "$n_bad" = "0" ]; then ok "MANIFEST.sha256 自校验通过（$n_total 项）"
  else bad "MANIFEST.sha256 有 $n_bad/$n_total 项不匹配（文件被改过？重跑 sha256sum）"; fi
else
  warn "没有 MANIFEST.sha256"
fi

echo
if [ "$fail" = "0" ]; then
  echo "[selfcheck] 全部通过 ✅  可以开始：bash scripts/build_image.sh"
else
  echo "[selfcheck] 有 $fail 项失败 ❌  先修再跑" >&2
fi
exit "$fail"
