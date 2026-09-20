#!/usr/bin/env bash
# =============================================================================
# 00_ensure_pgo.sh —— PGO 产物"确保存在"入口（幂等；只编译一次）
#
#   bash build_scripts/00_ensure_pgo.sh
#
# ## 设计
#
#   1. 算**构建指纹**：cpu_part + gcc 版本 + glibc 版本 + 源码 sha256 + configure 参数
#   2. 与 optim/pgo/.build_marker 比对：
#        一致且产物 md5 正确  -> **秒退**（"只编译一次"）
#        不一致 / 无 marker    -> 起一个 **一次性容器** 编译（不用服务镜像！）
#   3. 编译工作区落在 **容器外** `optim/pgo/build/`（保留 .o，可增量）
#   4. 产物落 `optim/pgo/{python3,libpython3.12.so.1.0}`，写 marker
#
# ## 为什么必须在 A2 上重编（以及不承诺什么）
#
#   * 现在包里的那份 .so 是在 **Ubuntu jammy** 里编的、跑在 **openEuler** 上。
#     重编能保证 glibc/发行版精确匹配（更稳），但：
#   * **构建参数里没有 `-march`/`-mtune`** ⇒ 代码生成是通用 aarch64，
#     **重编本身不会带来明显性能收益**（预期 0~3%，且必须实测）。
#   * 想针对本机核优化，唯一手段是给 configure 加 `-mtune=native`（安全，不用新指令）。
#     那属于**可选实验**，默认不开：`PGO_MTUNE=1 bash build_scripts/00_ensure_pgo.sh`
#
# 环境变量：
#   PGO_FORCE=1         忽略指纹，强制重编
#   PGO_MTUNE=1         给 CFLAGS 加 -mtune=native（默认 0 = 通用）
#   PGO_BUILD_IMAGE     编译容器镜像（默认自动探测：优先用本包的 $IMAGE 同源镜像）
#   PGO_JOBS            并行度（默认 min(48, nproc)）
#   PGO_SKIP_INSTALL=1  只编译不做自检
#
# 退出码：0 = 产物可用；非 0 = 失败（详情见 optim/pgo/build/logs/）
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
cd "$PKG"

PGO=optim/pgo
BUILD="$PGO/build"
MARK="$PGO/.build_marker"

say()  { printf '\033[1m[ensure-pgo]\033[0m %s\n' "$*"; }
die()  { printf '\033[31m[ensure-pgo][FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

DOCKER=${DOCKER:-}
[ -n "$DOCKER" ] || { docker info >/dev/null 2>&1 && DOCKER=docker || DOCKER="sudo -n docker"; }
$DOCKER info >/dev/null 2>&1 || die "无法访问 docker"

mkdir -p "$PGO" "$BUILD" "$BUILD/logs"

# ---------- 1) 产物体检（缺一个就当没编过） ----------
ART_OK=0
if [ -s "$PGO/libpython3.12.so.1.0" ] && [ -s "$PGO/python3" ]; then ART_OK=1; fi

# ---------- 2) 算指纹 ----------
_cpu_part=$(grep -m1 'CPU part' /proc/cpuinfo 2>/dev/null | awk '{print $NF}')
[ -n "${_cpu_part:-}" ] || _cpu_part=$(uname -m)
_gcc=$(gcc -dumpversion 2>/dev/null || echo none)
_glibc=$(ldd --version 2>/dev/null | head -1 | awk '{print $NF}')
_src_sha=$(sha256sum "$BUILD/Python-3.12.13.tgz" 2>/dev/null | cut -d' ' -f1)
[ -n "${_src_sha:-}" ] || _src_sha=nosrc
_mtune=${PGO_MTUNE:-0}
FP="cpu=${_cpu_part} gcc=${_gcc} glibc=${_glibc} src=${_src_sha} mtune=${_mtune}"
FP_H=$(printf '%s' "$FP" | sha256sum | cut -d' ' -f1)

say "构建指纹：$FP"
say "           sha256=$FP_H"

_need=0
_why=""
if [ "${PGO_FORCE:-0}" = "1" ]; then
  _need=1; _why="PGO_FORCE=1（强制重编）"
elif [ "$ART_OK" != "1" ]; then
  _need=1; _why="缺产物（$PGO/ 下没有 python3 / libpython）"
elif [ ! -f "$MARK" ]; then
  _need=1; _why="无 .build_marker（首次编译）"
else
  OLD=$(grep -m1 '^fingerprint=' "$MARK" 2>/dev/null | cut -d= -f2-)
  if [ "${OLD:-}" = "$FP_H" ]; then
    say "✅ 指纹一致且产物在位 —— 跳过编译（这就是「只编译一次」）"
    say "   libpython md5 = $(md5sum "$PGO/libpython3.12.so.1.0" | cut -d' ' -f1)"
    say "   想强制重编：PGO_FORCE=1 bash build_scripts/00_ensure_pgo.sh"
    exit 0
  fi
  _need=1; _why="指纹变化（旧 ${OLD:0:12}… → 新 ${FP_H:0:12}…）"
fi
[ "$_need" = "1" ] && say "→ 需要编译：$_why"

# ---------- 3) 选编译容器镜像 ----------
# 优先与 build_image.sh 用的基础镜像同源（glibc 才会精确匹配目标机）
BASE=${BASE_IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-openeuler}
IMG=${PGO_BUILD_IMAGE:-}
if [ -z "$IMG" ]; then
  # 顺序 = 优先级：本包烘焙出的 v8 在前，老机器上可能只留下了 v5/v6，保留作 fallback。
  for c in "$BASE" "dsv41-a2:v8" "dsv41-a2:v6" "dsv41-a2:v5"; do
    if $DOCKER image inspect "$c" >/dev/null 2>&1; then IMG="$c"; break; fi
  done
fi
[ -n "$IMG" ] || die "找不到可用的编译镜像（试过 $BASE / dsv41-a2:v8 / dsv41-a2:v6 / dsv41-a2:v5）"
say "编译容器镜像：$IMG"

JOBS=${PGO_JOBS:-}
if [ -z "$JOBS" ]; then
  n=$(nproc 2>/dev/null || echo 8); [ "$n" -gt 48 ] && n=48; JOBS=$n
fi
say "并行度：$JOBS"
[ "${PGO_MTUNE:-0}" = "1" ] && say "额外 CFLAGS：-mtune=native（安全，不用新指令）"

# ---------- 4) 编译（一次性容器；工作区在容器外） ----------
say "起一次性编译容器（不挂 /dev/davinci*，不碰服务容器）…"
_env=(-e "PGO_JOBS=$JOBS" -e "PGO_MTUNE=${PGO_MTUNE:-0}" -e "PGO_SKIP_INSTALL=${PGO_SKIP_INSTALL:-0}")
_t0=$(date +%s)
$DOCKER run --rm \
  --name pgo-build-a2 \
  --cpuset-cpus "${PGO_CPUSET:-$(cat /sys/devices/system/node/node0/cpulist 2>/dev/null || echo 0-3)}" \
  -v "$PKG/$PGO:/work" \
  -v "$HERE:/work/scripts:ro" \
  "${_env[@]}" \
  -w /work \
  "$IMG" \
  bash -lc 'set -uo pipefail
    cd /work
    echo "[build] distro: $(. /etc/os-release 2>/dev/null; echo ${PRETTY_NAME:-unknown})"
    bash /work/scripts/01_setup_build_env.sh || exit 11
    bash /work/scripts/02_fetch_source.sh    || exit 12
    bash /work/scripts/03_configure.sh       || exit 13
    bash /work/scripts/04_make.sh            || exit 14
    bash /work/scripts/05_install.sh         || exit 15
    bash /work/scripts/06_package.sh         || exit 16
    echo "[build] DONE"
  ' 2>&1 | tail -40
_rc=${PIPESTATUS[0]}
_t1=$(date +%s)
if [ "$_rc" != "0" ]; then
  die "编译失败（rc=$_rc，用时 $((_t1-_t0))s）。日志见 $PGO/build/logs/"
fi
say "编译完成，用时 $((_t1-_t0))s"

# ---------- 5) 落位 + 写 marker ----------
# 06_package.sh 把产物放到 /work/out/；这里搬到 $PGO/ 顶层（挂载点用）
for f in libpython3.12.so.1.0 python3; do
  if [ -s "$BUILD/out/$f" ]; then
    cp -f "$BUILD/out/$f" "$PGO/$f"
  fi
done
if [ ! -s "$PGO/libpython3.12.so.1.0" ] || [ ! -s "$PGO/python3" ]; then
  die "编译结束但 $PGO/ 下缺产物（检查 $BUILD/out/）"
fi
chmod +x "$PGO/python3" 2>/dev/null || true

{
  echo "fingerprint=$FP_H"
  echo "$FP"
  echo "built_at=$(date -Is)"
  echo "build_image=$IMG"
  echo "jobs=$JOBS"
  echo "libpython_md5=$(md5sum "$PGO/libpython3.12.so.1.0" | cut -d' ' -f1)"
  echo "python3_md5=$(md5sum "$PGO/python3" | cut -d' ' -f1)"
} > "$MARK"

# TARGET_PATH.txt：容器内 libpython 落点（build_image.sh / serve_a2.sh 也会自己探测）
if [ ! -s "$PGO/TARGET_PATH.txt" ]; then
  _tp=$($DOCKER run --rm --entrypoint python3 "$IMG" - <<'PYX' 2>/dev/null | tail -1
import os, sysconfig
name = sysconfig.get_config_var('INSTSONAME') or 'libpython%s.so.1.0' % sysconfig.get_config_var('VERSION')
for d in (sysconfig.get_config_var('LIBDIR'), '/usr/local/python3.12.13/lib', '/usr/lib', '/usr/local/lib'):
    if d and os.path.exists(os.path.join(d, name)):
        print(os.path.join(d, name)); break
PYX
)
  case "${_tp:-}" in /*.so*) printf '%s' "$_tp" > "$PGO/TARGET_PATH.txt"; say "TARGET_PATH.txt = $_tp" ;; esac
fi

say "✅ 产物就绪"
say "   libpython md5 = $(md5sum "$PGO/libpython3.12.so.1.0" | cut -d' ' -f1)"
say "   marker        = $MARK"
say "   工作区保留在  = $BUILD（下次增量；换 CPU/gcc/源码会自动重编）"
