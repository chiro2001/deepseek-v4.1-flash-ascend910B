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
#   IMAGE_TAG   产出镜像名（默认 dsv41-a2:v6）
#   NO_CACHE    1 = 不使用构建缓存
#   SKIP_PGO    1 = 不打包 PGO 产物（镜像会小 30 MB）
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
cd "$PKG"

BASE_IMAGE=${BASE_IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-openeuler}
IMAGE_TAG=${IMAGE_TAG:-dsv41-a2:v6}
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
  -t "$IMAGE_TAG" \
  -f "$PKG/Dockerfile" "$PKG" 2>&1 | tail -30
[ "${PIPESTATUS[0]}" = "0" ] || die "docker build 失败"

# ---------- 4) 自校验（镜像内逐文件 md5 + py_compile + 备份存在性）----------
say "校验烘焙结果…"
$DOCKER run "$IMAGE_TAG" bash -lc '
  set -uo pipefail
  A="${ASCEND_PKG:-/vllm-workspace/vllm-ascend/vllm_ascend}"
  rc=0
  chk() { # relpath expected_md5
    local t="$A/$1"
    if [ ! -f "$t" ]; then echo "  FAIL 缺文件 $1"; rc=1; return; fi
    local got; got=$(md5sum "$t" | cut -d" " -f1)
    if [ "$got" != "$2" ]; then echo "  FAIL md5 $1: got=$got want=$2"; rc=1; return; fi
    python3 -m py_compile "$t" 2>/dev/null || { echo "  FAIL py_compile $1"; rc=1; return; }
    echo "  OK  $got  $1"
  }
  chk models/deepseek_v41/engram_hbm.py            6f227a749aa6ba6f1290446611202028
  chk models/deepseek_v41/engram_hash.py           3a842bbb6d0dd783c65087ccef347370
  chk models/deepseek_v41/engram_jit_kernel.py     1add256a203d7f6dfd98874c575ce24a
  chk models/deepseek_v41/engram_plan_kernel.py    0be62d7775374b0167a54f5b393a65ac
  chk models/deepseek_v41/engram_gate.py           146010cac42261e9dc4380699e156252
  chk models/deepseek_v41/model.py                 5b7c45261e2d63838b9e4a25f87ad350
  chk ascend_forward_context.py                    6cccd4259bd65c907ef9d9dd42a83dca
  chk attention/dsa_v1.py                          9a36e709b0937589eab05c5316a62591
  chk models/deepseek_v41/indexer.py               f61f242df4f060106ce1bf4500ff5844
  chk ops/fused_moe/token_dispatcher.py            a695735ae3e03096a432468eb9ad6b83
  chk ops/rope_dsv4.py                             6a19890850ac7cb41c535b070c2dfbf6
  # 备份必须存在（回滚用）
  for f in models/deepseek_v41/engram_hbm.py models/deepseek_v41/engram_gate.py \
           ops/fused_moe/token_dispatcher.py ops/rope_dsv4.py attention/dsa_v1.py ; do
    [ -f "$A/$f.a2orig" ] || { echo "  FAIL 缺备份 $f.a2orig"; rc=1; }
  done
  # 未验证项**不应**被装进运行时（只在 /opt/dsv41/patches 里待命）
  [ -f /opt/dsv41/patches/draft/dspark_proposer.py ] || { echo "  FAIL 缺 draft 补丁"; rc=1; }
  [ -f /opt/dsv41/patches/draft/llm_base_proposer.py ] || { echo "  FAIL 缺 draft 补丁"; rc=1; }
  [ -f /opt/dsv41/patches/files/token_dispatcher_moezero.py ] || { echo "  FAIL 缺 MOE_ZERO 补丁"; rc=1; }
  [ -f /opt/dsv41/scripts/serve_v2.sh ] || { echo "  FAIL 缺 serve_v2.sh"; rc=1; }
  [ -f /opt/dsv41/scripts/serve_a2.sh ] || { echo "  FAIL 缺 serve_a2.sh"; rc=1; }
  [ -f /opt/dsv41/BUILD_INFO.txt ]      || { echo "  FAIL 缺 BUILD_INFO.txt"; rc=1; }
  grep -q libjemalloc /opt/dsv41/scripts/serve_v2.sh || { echo "  FAIL serve_v2 未启用 jemalloc"; rc=1; }
  exit $rc
' || die "自校验失败（见上）"

say "完成：镜像 $IMAGE_TAG"
cat <<EOF

  【命令 ②】起服务并跑验收测试：

    MODEL=/path/to/your/v41-w4a8-engram-dr-vision-qrot-mtpq \\
      IMAGE=$IMAGE_TAG bash scripts/run_test.sh

  先读一页说明：  less README.md
  现场排错：      less REPRO.md  §7（失败排查表）
EOF
