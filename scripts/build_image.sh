#!/usr/bin/env bash
# =============================================================================
# 【命令 ①】一次执行生成「修改后的容器镜像」
#
#   cd a2_pkg_v5
#   bash scripts/build_image.sh
#
# 它会：
#   1. 从基础镜像里**探测** vllm_ascend / vLLM / python 的安装路径（不同镜像布局不同，不写死）
#   2. 把 patches/files/ 下全部已验证补丁烘焙进镜像（原文件留 `.a2orig` 备份）
#   3. 把 PGO 版 python 产物放进 /opt/dsv41/pgo/（起服时**挂载**覆盖 libpython，可回滚）
#   4. 在新镜像里逐文件校验 md5 + py_compile，并打印下一条命令
#
# 可选环境变量：
#   BASE_IMAGE  基础镜像（默认 quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-openeuler）
#   IMAGE_TAG   产出镜像名（默认 dsv41-a2:v8）
#   NO_CACHE    1 = 不使用构建缓存
#   SKIP_PGO    1 = 不打包 PGO 产物（镜像会小 30 MB）
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
cd "$PKG"

BASE_IMAGE=${BASE_IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-openeuler}
IMAGE_TAG=${IMAGE_TAG:-dsv41-a2:v8}
CACHE_ARGS=()
[ "${NO_CACHE:-0}" = "1" ] && CACHE_ARGS+=(--no-cache)

say() { printf '\n\033[1m[build]\033[0m %s\n' "$*"; }
die() { printf '\n\033[31m[build][FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------- 0) docker 权限 ----------
DOCKER="docker"
if ! $DOCKER info >/dev/null 2>&1; then
  if sudo -n docker info >/dev/null 2>&1; then DOCKER="sudo -n docker";
  else die "无法访问 docker（试过 docker 与 sudo -n docker）"; fi
fi

# ---------- 0.5) 本脚本自身依赖（第 4 步要用它们推导/核对校验和）----------
command -v python3 >/dev/null 2>&1 || die "需要宿主机 python3（用于由 patches/files 推导期望 md5）"
for f in tools/check_checksums.py tools/verify_baked_tree.sh; do
  [ -f "$PKG/$f" ] || die "缺 $f —— 它是第 4 步自校验的依据（v8 起不再手写 md5 表）"
done

# ---------- 1) 基础镜像 + 路径探测 ----------
say "基础镜像: $BASE_IMAGE"
$DOCKER image inspect "$BASE_IMAGE" >/dev/null 2>&1 \
  || die "基础镜像不存在。请先 docker pull $BASE_IMAGE，或用 BASE_IMAGE=<你的镜像> bash scripts/build_image.sh"

probe() { $DOCKER run --rm "$BASE_IMAGE" python3 -c "$1" 2>/dev/null | grep -oE 'PKG=[^ ]+' | head -1 | cut -d= -f2; }
say "探测安装路径…"
ASCEND_PKG=$(probe "import vllm_ascend, os; print('PKG=' + os.path.dirname(vllm_ascend.__file__))")
[ -n "${ASCEND_PKG:-}" ] || ASCEND_PKG=/vllm-workspace/vllm-ascend/vllm_ascend
VLLM_ROOT=$(probe "import vllm, os; print('PKG=' + os.path.dirname(os.path.dirname(vllm.__file__)))")
[ -n "${VLLM_ROOT:-}" ] || VLLM_ROOT=/vllm-workspace/vllm
PY_VER=$($DOCKER run --rm "$BASE_IMAGE" python3 -c "import sys;print('%d.%d.%d'%sys.version_info[:3])" 2>/dev/null | tail -1)
say "  ASCEND_PKG = $ASCEND_PKG"
say "  VLLM_ROOT  = $VLLM_ROOT"
say "  python     = ${PY_VER:-unknown}"

# ---------- 2) PGO 产物：路径 + 版本校验，写 TARGET_PATH.txt ----------
mkdir -p optim/pgo
if [ "${SKIP_PGO:-0}" != "1" ] && [ -f optim/pgo/libpython3.12.so.1.0 ]; then
  say "PGO：探测镜像内 libpython 路径…"
  LIBPATH=$($DOCKER run --rm -i "$BASE_IMAGE" python3 - <<'PY' 2>/dev/null | tail -1
import sysconfig, os
name = sysconfig.get_config_var('INSTSONAME') or 'libpython%s.so.1.0' % sysconfig.get_config_var('VERSION')
for d in (sysconfig.get_config_var('LIBDIR'), '/usr/local/python3.12.13/lib', '/usr/lib', '/usr/local/lib'):
    if d and os.path.exists(os.path.join(d, name)):
        print(os.path.join(d, name)); break
PY
)
  if [ -n "${LIBPATH:-}" ]; then
    echo "$LIBPATH" > optim/pgo/TARGET_PATH.txt
    say "  libpython 目标 = $LIBPATH"
    # 版本一致性：PGO 产物是 3.12.13 编译的；镜像 python 版本必须一致
    if [ "$PY_VER" != "3.12.13" ]; then
      say "  ⚠️ 镜像 python 是 $PY_VER，PGO 产物是 3.12.13 编译的 → **起服时请用 PYTHON_PGO=0**（serve_a2.sh 会给出警告）"
      echo "MISMATCH python=$PY_VER expected=3.12.13" > optim/pgo/VERSION_MISMATCH.txt
    else
      rm -f optim/pgo/VERSION_MISMATCH.txt 2>/dev/null || true
      say "  ✓ 版本一致（3.12.13）"
    fi
  else
    say "  ⚠️ 探测不到 libpython 落点 → PGO 默认不可用（起服时 PYTHON_PGO=0）"
    echo "" > optim/pgo/TARGET_PATH.txt
  fi
else
  if [ "${SKIP_PGO:-0}" = "1" ]; then
    say "PGO：按 SKIP_PGO=1 跳过（镜像不打包 PGO 产物）"
  else
    say "PGO：无产物 ⇒ 本次不打包 PGO"
    say "      要启用，先在**本机**编译一次（产物与 CPU/gcc/glibc 绑定，不随包分发）："
    say "        bash build_scripts/00_ensure_pgo.sh     # 约 30–40 min，之后增量/秒退"
    say "      不想用也行：PGO 的预期收益仅 0~3%，且需自行实测。详见 optim/pgo/README.md"
  fi
fi

# ---------- 3) 构建 ----------
say "构建镜像 $IMAGE_TAG …（首次会拉基础镜像层，可能较久）"
$DOCKER build "${CACHE_ARGS[@]}" \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "ASCEND_PKG=$ASCEND_PKG" \
  --build-arg "VLLM_ROOT=$VLLM_ROOT" \
  --build-arg "SKIP_PGO=${SKIP_PGO:-0}" \
  --build-arg "ALLOW_ASCEND_VERSION_MISMATCH=${ALLOW_ASCEND_VERSION_MISMATCH:-0}" \
  -t "$IMAGE_TAG" \
  -f "$PKG/Dockerfile" "$PKG" 2>&1 | tail -30
[ "${PIPESTATUS[0]}" = "0" ] || die "docker build 失败"

# ---------- 4) 自校验（镜像内逐文件 md5 + py_compile + 备份存在性）----------
#
# ⚠️ 这里**不再有手写 md5 表**。v7→v8 就是因为这张表没跟上而让用户白等 10–20 分钟：
#    改了 patches/files/model.py 却忘了改 chk 里的期望值 -> "FAIL md5 .../model.py"。
# 现在期望值在**构建时**由 `tools/check_checksums.py` 从包内载荷字节现算，
# 落位表（哪个文件装到哪个路径、inst 还是 newf）由 Dockerfile 的 inst/newf 推导 ⇒
# "改了文件忘了同步校验和"在结构上不可能再发生，也不会再漏掉某个文件。
say "生成校验清单（由 patches/files/ 的字节 + Dockerfile 落位表推导）…"
CHK_TSV=$(mktemp -t dsv41chk.XXXXXX)
trap 'rm -f "$CHK_TSV"' EXIT
python3 "$PKG/tools/check_checksums.py" \
    --dockerfile "$PKG/Dockerfile" \
    --payload    "$PKG/patches/files" \
    --sums       "$PKG/patches/MD5SUMS" \
    --manifest   "$CHK_TSV" --quiet-ok \
  || die "包内一致性检查失败（见上）—— 先修好再 build，别浪费 10–20 分钟"
say "  清单 $(grep -c . "$CHK_TSV") 项：$(cut -f1 "$CHK_TSV" | sort | uniq -c | tr '\n' ' ')"

say "校验烘焙结果…"
$DOCKER run --rm \
  -v "$CHK_TSV:/tmp/dsv41-chk.tsv:ro" \
  -v "$PKG/tools/verify_baked_tree.sh:/tmp/dsv41-verify.sh:ro" \
  "$IMAGE_TAG" bash -lc '
  set -uo pipefail
  A="${ASCEND_PKG:-/vllm-workspace/vllm-ascend/vllm_ascend}"
  bash /tmp/dsv41-verify.sh --root "$A" --manifest /tmp/dsv41-chk.tsv --image-extras
' || die "自校验失败（见上）"

say "完成：镜像 $IMAGE_TAG"
cat <<EOF

  【命令 ②】起服务并跑验收测试：

    MODEL=/path/to/your/v41-w4a8-engram-dr-vision-qrot-mtpq \\
      IMAGE=$IMAGE_TAG bash scripts/run_test.sh

  先读一页说明：  less README.md
  现场排错：      less REPRO.md  §7（失败排查表）
EOF
