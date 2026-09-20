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
#   PGO_CPUSET          编译容器可用的 CPU 列表（默认 node0 的 cpulist）
#   PGO_RM_CONTAINER=1  编译结束后删掉编译容器（成功/失败都删；默认保留，便于续编/排错）
#   PGO_SKIP_INSTALL=1  ⚠️ **预留未接线**：会被透传进容器，但六个步骤都不读它（等价于无效）
#   INSECURE_TLS=1      跳过 TLS 证书校验（**仅内网自签证书/中间盒**；默认关，见下）
#   PGO_CONTAINER_NAME  编译容器名（默认 pgo-build-a2）
#
# ## 编译容器：复用 + 续编（A2 真机教训，2026-09-20）
#
# 原来用 `docker run --rm`：编译 30–40 分钟，一旦中途失败（网络抖动 / 依赖源 / 上一次
# 的残留文件），`--rm` 把「已装好的构建依赖 + 已完成的编译进度」一起丢掉，只能从头再来。
# 现在：固定容器名 + 失败保留 + 参数一致时自动复用：
#   * 参数（PGO_JOBS/PGO_MTUNE/PGO_SKIP_INSTALL/PGO_FORCE）一致 → `docker start -ai` 续编
#   * 参数变化 / PGO_FORCE=1 → 删旧容器、重建（全新编译）
#   * 失败时容器**保留**，重跑本脚本即自动续编；手动进去看现场：`docker start -ai`
#   * 编译**成功**后默认也保留容器；要清理加 `PGO_RM_CONTAINER=1`
#
# ⚠️ 代理（http_proxy/https_proxy/no_proxy，大小写都认）与 `scripts/openEuler.repo`
#    **只在容器创建时生效**；改了它们要 `PGO_FORCE=1` 重建容器才会应用。
#
# ## INSECURE_TLS=1 是什么、为什么默认关
#
# 内网环境常见自签证书 / TLS 中间盒：`curl` 会因为证书链校验失败而拒绝下载（CPython
# 源码、测试语料），yum/dnf 也会卡在同样的校验上。`INSECURE_TLS=1` 会：
#   * 给 `02_fetch_source.sh` 的 curl 加 `-k`；
#   * 往容器内 /etc/yum.conf 与 /etc/dnf/dnf.conf 写 `sslverify=False`（幂等，重复跑不叠加）。
# **这是安全降级**（传输层不再验身份）⇒ 默认**关**，public 网络**不要**打开。
# 完整性仍由 `02_fetch_source.sh` 的双源 sha256 交叉校验兜底（华为云 vs 阿里云逐字节比对），
# 以及 `tools/fetch_corpus.sh` 的 sha256 校验 —— `-k` 只影响传输层。
# 本包**不含**任何内网源地址/凭据；内网用户请自己把 repo 文件放成 `scripts/openEuler.repo`
# （存在才会挂载，见下面第 4 节的说明）。
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
# warn 走 stderr：供「在 $(...) 里被调用」的函数使用 —— 否则提示文字会被命令替换抓走，
# 混进返回值里（实测踩过：_pick_artifact 的提示被当成构件路径）。
warn() { printf '\033[33m[ensure-pgo]\033[0m %s\n' "$*" >&2; }
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
    # 上一轮若是「编译链非零退出、但产物新鲜被采纳」写下的 marker，这里要提醒，
    # 别让那次失败被永久忘掉。
    _brc=$(grep -m1 '^build_rc=' "$MARK" 2>/dev/null | cut -d= -f2-)
    if [ -n "${_brc:-}" ] && [ "$_brc" != "0" ]; then
      warn "注意：这份产物的 marker 里记着 build_rc=$_brc（当时编译链非零退出，但产物是本次新生成的 ⇒ 采纳了）。"
      warn "      要彻底干净地重编：PGO_FORCE=1 bash build_scripts/00_ensure_pgo.sh"
    fi
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

# ---------- 4) 编译容器（可复用 + 续编；工作区在容器外） ----------
say "编译容器（不挂 /dev/davinci*，不碰服务容器）…"
CONT_NAME=${PGO_CONTAINER_NAME:-pgo-build-a2}
# 编译容器的去留：默认**保留**（续编/排错都要用它）；PGO_RM_CONTAINER=1 才删。
# 返回 0 = 已删；1 = 保留了（调用方负责提示）。
_reap_container() {
  [ "${PGO_RM_CONTAINER:-0}" = "1" ] || return 1
  $DOCKER rm -f "$CONT_NAME" >/dev/null 2>&1 || true
  say "已按 PGO_RM_CONTAINER=1 删除编译容器 $CONT_NAME（下次全新编译，本次进度不再保留）"
  return 0
}
_env=(-e "PGO_JOBS=$JOBS" -e "PGO_MTUNE=${PGO_MTUNE:-0}" -e "PGO_SKIP_INSTALL=${PGO_SKIP_INSTALL:-0}"
      -e "PGO_FORCE=${PGO_FORCE:-0}" -e "INSECURE_TLS=${INSECURE_TLS:-0}")

# [PROXY] 只透传**真的设了**的代理变量：
#   * 本脚本是 set -u，写成 `-e http_proxy=$http_proxy` 在变量未定义时直接 unbound 报错；
#   * 传空串在部分工具里语义含糊（"不使用代理"还是"使用空代理"）。
# 只打印变量**名**（代理 URL 里常带凭据，不该进终端/日志）。
_proxy_env=()
for _v in http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY; do
  _pv=${!_v:-}
  [ -n "$_pv" ] && _proxy_env+=(-e "$_v=$_pv")
done
if [ "${#_proxy_env[@]}" -gt 0 ]; then
  # 注意：数组里是 `-e` 与 `NAME=value` **两个独立元素**，所以只匹配 NAME=… 那种元素
  # （匹配 `^-e ` 会一个都匹配不到，打印出空列表 —— 实测踩过）。
  say "  透传代理变量：$(printf '%s\n' "${_proxy_env[@]}" | sed -n 's/^\([A-Za-z_][A-Za-z_0-9]*\)=.*/\1/p' | tr '\n' ' ')"
fi

# [REPO] 内网自建源（可选）：**只有文件真存在才挂**。
# 否则 docker 会在宿主上创建一个同名**目录**再挂进去，容器里 /etc/yum.repos.d/openEuler.repo
# 变成一个目录 ⇒ yum/dnf 直接报错。本包不含该文件（内网用户自行放置）。
_repo_args=()
if [ -f "$PKG/scripts/openEuler.repo" ]; then
  _repo_args=(-v "$PKG/scripts/openEuler.repo:/etc/yum.repos.d/openEuler.repo:ro")
  say "  挂载内网源：scripts/openEuler.repo"
else
  say "  无 scripts/openEuler.repo ⇒ 用容器自带 repo（要换源就把它放进 scripts/）"
fi

# 复用 or 重建：按「编译参数是否一致」决定
REUSE=0
_cnt=$($DOCKER inspect -f '{{.Id}}' "$CONT_NAME" 2>/dev/null || true)
if [ -n "$_cnt" ]; then
  _running=$($DOCKER inspect -f '{{.State.Running}}' "$_cnt" 2>/dev/null || true)
  if [ "$_running" = "true" ]; then
    die "编译容器 $CONT_NAME（$_cnt）正在运行 —— 等它结束，或 $DOCKER stop $CONT_NAME"
  fi
  _old_env=$($DOCKER inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$_cnt" 2>/dev/null || true)
  _old_of() { printf '%s\n' "$_old_env" | grep -m1 "^$1=" | cut -d= -f2-; }
  _old_jobs=$(_old_of PGO_JOBS          || true)
  _old_mtune=$(_old_of PGO_MTUNE        || true)
  _old_skip=$(_old_of PGO_SKIP_INSTALL  || true)
  _old_force=$(_old_of PGO_FORCE        || true)
  if [ "${PGO_FORCE:-0}" = "1" ]; then
    say "PGO_FORCE=1 ⇒ 删除旧容器、全新编译"
    $DOCKER rm -f "$_cnt" >/dev/null 2>&1 || true
  elif [ "$_old_jobs" = "$JOBS" ] \
    && [ "$_old_mtune" = "${PGO_MTUNE:-0}" ] \
    && [ "$_old_skip" = "${PGO_SKIP_INSTALL:-0}" ] \
    && [ "$_old_force" = "${PGO_FORCE:-0}" ]; then
    REUSE=1
    say "复用既有编译容器 $CONT_NAME（保留已装依赖与上次进度；工作区在宿主 $PKG/$PGO）"
    say "  注意：代理 / scripts/openEuler.repo 的改动只在容器创建时生效，要应用请 PGO_FORCE=1"
  else
    say "编译参数变化（旧 jobs=$_old_jobs mtune=$_old_mtune skip=$_old_skip force=$_old_force"
    say "            → 新 $JOBS/${PGO_MTUNE:-0}/${PGO_SKIP_INSTALL:-0}/${PGO_FORCE:-0}），重建容器"
    $DOCKER rm -f "$_cnt" >/dev/null 2>&1 || true
  fi
fi

_build_cmd='set -uo pipefail
    cd /work
    if [ "${INSECURE_TLS:-0}" = "1" ]; then
      echo "[build] INSECURE_TLS=1 ⇒ 关闭包管理器 TLS 校验（仅限自签证书内网；公网别开）"
      for f in /etc/yum.conf /etc/dnf/dnf.conf; do
        [ -f "$f" ] || continue
        if grep -q "^sslverify=" "$f" 2>/dev/null; then
          sed -i "s/^sslverify=.*/sslverify=False/" "$f"
        else
          printf "sslverify=False\n" >> "$f"
        fi
      done
      echo "[build] sslverify=False 已写入 yum/dnf 配置（幂等：重复运行不会叠加）"
    fi
    echo "[build] distro: $(. /etc/os-release 2>/dev/null; echo ${PRETTY_NAME:-unknown})"
    bash /work/scripts/01_setup_build_env.sh || exit 11
    bash /work/scripts/02_fetch_source.sh    || exit 12
    bash /work/scripts/03_configure.sh       || exit 13
    bash /work/scripts/04_make.sh            || exit 14
    bash /work/scripts/05_install.sh         || exit 15
    bash /work/scripts/06_package.sh         || exit 16
    echo "[build] DONE"
  '

mkdir -p "$BUILD/logs"
STAMP="$BUILD/logs/.build_start_stamp"
: > "$STAMP"
_t0=$(date +%s)
if [ "$REUSE" = "1" ]; then
  $DOCKER start -ai "$CONT_NAME" 2>&1 | tee "$BUILD/logs/build.log"
else
  $DOCKER run --name "$CONT_NAME" \
    --cpuset-cpus "${PGO_CPUSET:-$(cat /sys/devices/system/node/node0/cpulist 2>/dev/null || echo 0-3)}" \
    "${_repo_args[@]+"${_repo_args[@]}"}" \
    "${_proxy_env[@]+"${_proxy_env[@]}"}" \
    -v "$PKG/$PGO:/work" \
    -v "$HERE:/work/scripts:ro" \
    "${_env[@]}" \
    -w /work \
    "$IMG" \
    bash -lc "$_build_cmd" 2>&1 | tee "$BUILD/logs/build.log"
fi
# 实测（bash 5.3）：if/else 里最后执行的那条流水线仍会写 PIPESTATUS ⇒ 这里拿到的是
# docker 的退出码（不是 tee 的）。见 tools/negative_control.sh 同类写法的注释。
_rc=${PIPESTATUS[0]}
_t1=$(date +%s)
if [ "$_rc" != "0" ]; then
  say "⚠️  编译链非零退出（rc=$_rc，用时 $((_t1-_t0))s）"
  if [ "${PGO_RM_CONTAINER:-0}" = "1" ]; then
    say "   PGO_RM_CONTAINER=1 ⇒ 容器将在本次结束时删除（没有进度可续编）"
  else
    say "   容器**已保留**为 $CONT_NAME；续编（推荐）：重跑 bash build_scripts/00_ensure_pgo.sh"
    say "   进现场：$DOCKER start -ai $CONT_NAME        日志：$BUILD/logs/build.log"
  fi
  say "   ⇒ 第 5 步只采纳**本次新生成**的产物；上次残留的产物会被显式忽略（防假成功）"
else
  say "编译完成，用时 $((_t1-_t0))s"
fi

# ---------- 5) 落位 + 写 marker ----------
#
# 产物在哪：06_package.sh 写的是**容器内** /work/out，而 /work 就是宿主 $PGO 的挂载点
# ⇒ 真实路径是 **$PGO/out**（容器内落位表见 build_scripts/06_package.sh 的 OUT=/work/out）。
# $BUILD/out 是更早的历史路径（.gitignore 里还留着 optim/pgo/build/out/），保留作 fallback。
#
# ⚠️ 防「假成功」：第 4 步非零退出时，绝不能把上次残留的产物当成本次成果 —— 那会写出一份
# **指纹正确但产物陈旧**的 marker，之后每次都被"秒钟退出"骗过去。所以 rc≠0 时只采纳
# mtime 晚于本次开工时间（$STAMP）的产物；两个产物都拿不到就 die（容器仍保留，可续编）。
_pick_artifact() {   # $1=文件名；找到并打印可用源路径，返回 0；否则返回 1
  local f=$1 o
  for o in "$PGO/out" "$BUILD/out"; do
    [ -s "$o/$f" ] || continue
    if [ "$_rc" != "0" ] && [ ! "$o/$f" -nt "$STAMP" ]; then
      # 必须走 stderr：本函数在 $(...) 里执行，stdout 是**返回值**
      warn "  忽略陈旧产物：$o/$f（mtime 早于本次开工 ⇒ 不是这次编出来的）"
      continue
    fi
    printf '%s' "$o/$f"
    return 0
  done
  return 1
}
_picked=0
for f in libpython3.12.so.1.0 python3; do
  if _src=$(_pick_artifact "$f"); then
    cp -f "$_src" "$PGO/$f"
    say "已复制产物 $f ← $_src"
    _picked=$((_picked+1))
  fi
done
# ⚠️ 判据必须是「**本次真的拷到了两件产物**」，不能只看 $PGO/ 下有没有文件 ——
# 上一轮成功留下的顶层产物会让这一轮的失败看起来成功（离线下实测复现过）。
if [ "$_picked" != "2" ]; then
  if [ "$_rc" != "0" ]; then
    _reap_container || say "   容器已保留为 $CONT_NAME：续编请重跑 bash build_scripts/00_ensure_pgo.sh"
    die "编译链失败（rc=$_rc）且没有本次新生成的齐全产物（只拿到 $_picked/2 件）⇒ 不写 marker、不报成功（日志 $BUILD/logs/build.log）"
  fi
  die "编译结束（rc=0）但只拿到 $_picked/2 件产物（检查 $PGO/out/ 与 $BUILD/out/，以及 06_package.sh 的日志）"
fi
chmod +x "$PGO/python3" 2>/dev/null || true

if [ "$_rc" != "0" ]; then
  say "⚠️  编译链 rc=$_rc，但两个产物都是**本次新生成**的且齐全 ⇒ 采纳。"
  say "   典型场景：最后一步 06_package.sh 在打完包之后才报错（产物已落盘）。"
  say "   若不确定，请看 $BUILD/logs/build.log 并重跑一次（参数未变会复用容器续编）。"
fi

{
  echo "fingerprint=$FP_H"
  echo "$FP"
  echo "built_at=$(date -Is)"
  echo "build_image=$IMG"
  echo "jobs=$JOBS"
  echo "build_rc=$_rc"
  echo "build_container=$CONT_NAME"
  echo "libpython_md5=$(md5sum "$PGO/libpython3.12.so.1.0" | cut -d' ' -f1)"
  echo "python3_md5=$(md5sum "$PGO/python3" | cut -d' ' -f1)"
} > "$MARK"

# 编译容器：默认保留（续编/排错都要用）；PGO_RM_CONTAINER=1 才清掉
_reap_container || say "编译容器保留为 $CONT_NAME（不需要时：$DOCKER rm -f $CONT_NAME）"

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
