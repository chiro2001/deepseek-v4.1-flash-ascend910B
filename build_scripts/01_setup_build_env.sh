#!/bin/bash
# =============================================================================
# 01_setup_build_env.sh —— 在**一次性构建容器**内准备 CPython 编译环境
#
# ## 目标
# 精确复现原镜像 CPython 3.12.13 的模块集合：
#   yes    : _ctypes _curses _curses_panel _decimal _gdbm readline _sqlite3
#            zlib _bz2 _lzma _ssl _hashlib pyexpat _crypt
#   missing: _dbm nis _tkinter _uuid        （明确不装对应的 -dev 包）
# ⇒ 需要：zlib / bz2 / lzma / ssl / ffi / ncurses / readline / sqlite3 /
#          gdbm / expat / crypt 的开发包
#
# ## ★ distro-aware（v6 新增，v5 的脚本只支持 Ubuntu）
# v5 的这份脚本硬编码 `apt-get` + ubuntu-ports 源。而 **A2 的镜像是
# openEuler 24.03**（没有 apt）⇒ 原样搬到 A2 会直接失败。
# 这里按 `/etc/os-release` 分流：Debian 系走 apt，RPM 系走 dnf/yum。
# **注意**：编译容器不一定是宿主机的发行版 —— 这里探测的是**容器内**的。
#
# ## 关于 `--no-upgrade`（v5 留下的重要教训，务必保留等价语义）
#   「避免把镜像里已有的运行时库（libssl3/libffi8/...）升级掉，否则新编译的
#     扩展模块可能依赖服务容器里不存在的新符号」
#   apt  → `--no-upgrade`
#   dnf  → 不加 `upgrade`，并且**不要**用 `--allowerasing`；
#          另加 `--setopt=install_weak_deps=False` 避免拉入无关依赖
# =============================================================================
set -euo pipefail

mkdir -p /work/logs /work/src /work/out

# ---------- 发行版探测 ----------
DISTRO_ID=""; DISTRO_LIKE=""; DISTRO_NAME="unknown"
if [ -r /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  DISTRO_ID="${ID:-}"; DISTRO_LIKE="${ID_LIKE:-}"; DISTRO_NAME="${PRETTY_NAME:-unknown}"
fi
echo "=== 构建容器发行版：$DISTRO_NAME (ID=$DISTRO_ID LIKE=$DISTRO_LIKE) ==="
echo "=== uname: $(uname -a) ==="

is_deb=0; is_rpm=0
case " $DISTRO_ID $DISTRO_LIKE " in
  *" debian "*|*" ubuntu "*) is_deb=1 ;;
esac
case " $DISTRO_ID $DISTRO_LIKE " in
  *" rhel "*|*" fedora "*|*" centos "*|*" openeuler "*) is_rpm=1 ;;
esac
[ "$DISTRO_ID" = "ubuntu" ] || [ "$DISTRO_ID" = "debian" ] && is_deb=1
[ "$DISTRO_ID" = "openEuler" ] || [ "$DISTRO_ID" = "openeuler" ] && is_rpm=1
if [ "$is_deb" = 0 ] && [ "$is_rpm" = 0 ]; then
  command -v apt-get >/dev/null 2>&1 && is_deb=1
  command -v dnf     >/dev/null 2>&1 && is_rpm=1
fi
[ "$is_deb" = 1 ] || [ "$is_rpm" = 1 ] || { echo "无法判定包管理器（既非 deb 也非 rpm）"; exit 1; }
echo "=== 包管理器：$( [ "$is_deb" = 1 ] && echo 'apt (deb)' || echo 'dnf/yum (rpm)' ) ==="

# ---------- 记录安装前的包状态 ----------
if [ "$is_deb" = 1 ]; then
  dpkg -l | awk '/^ii/{print $2, $3}' | sort > /work/logs/pkg_versions_before.txt
  # 容器内 DNS 可能解析不到 k8s 内网 apt 源 ⇒ 改用公网镜像（仅影响本一次性容器）
  rm -f /etc/apt/sources.list.d/* 2>/dev/null || true
  if grep -qi ubuntu /etc/os-release 2>/dev/null; then
    cat > /etc/apt/sources.list <<'EOF'
deb http://mirrors.aliyun.com/ubuntu-ports jammy main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu-ports jammy-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu-ports jammy-security main restricted universe multiverse
EOF
  else
    cat > /etc/apt/sources.list <<'EOF'
deb http://mirrors.aliyun.com/debian bookworm main contrib non-free
deb http://mirrors.aliyun.com/debian-security bookworm-security main contrib non-free
EOF
  fi
  apt-get update -y 2>&1 | tail -5
  PKGS="zlib1g-dev libbz2-dev liblzma-dev libssl-dev libffi-dev
        libncurses-dev libreadline-dev libsqlite3-dev libgdbm-dev
        libexpat1-dev libcrypt-dev ca-certificates curl xz-utils"
  # --no-upgrade：见文件头说明（防止把镜像里的运行时库升级掉）
  apt-get install -y --no-install-recommends --no-upgrade $PKGS 2>&1 | tail -15
  dpkg -l | awk '/^ii/{print $2, $3}' | sort > /work/logs/pkg_versions_after.txt
  diff /work/logs/pkg_versions_before.txt /work/logs/pkg_versions_after.txt \
    > /work/logs/pkg_versions_diff.txt || true
  echo "=== 已安装包变化（前 60 行）==="; head -60 /work/logs/pkg_versions_diff.txt || true
else
  rpm -qa --qf '%{NAME} %{VERSION}-%{RELEASE}\n' 2>/dev/null | sort > /work/logs/pkg_versions_before.txt
  PM=$(command -v dnf || command -v yum)
  [ -n "$PM" ] || { echo "找不到 dnf/yum"; exit 1; }
  # 先换源（openEuler 默认源在国内多半可用；若不可用再退回此处换镜像）
  if [ -d /etc/yum.repos.d ] && [ -z "${PGO_KEEP_REPOS:-}" ]; then
    for r in /etc/yum.repos.d/*.repo; do
      [ -f "$r" ] || continue
      sed -i 's|^\(mirrorlist=\)|#\1|' "$r" 2>/dev/null || true
    done
  fi
  $PM makecache -y 2>&1 | tail -5 || true
  PKGS="zlib-devel bzip2-devel xz-devel openssl-devel libffi-devel
        ncurses-devel readline-devel sqlite-devel gdbm-devel
        expat-devel libxcrypt-devel ca-certificates curl xz"
  # 语义等价于 apt 的 --no-upgrade：只 install，不 upgrade，且不因此替换既有包
  $PM install -y --setopt=install_weak_deps=False --best $PKGS 2>&1 | tail -20
  rpm -qa --qf '%{NAME} %{VERSION}-%{RELEASE}\n' 2>/dev/null | sort > /work/logs/pkg_versions_after.txt
  diff /work/logs/pkg_versions_before.txt /work/logs/pkg_versions_after.txt \
    > /work/logs/pkg_versions_diff.txt || true
  echo "=== 已安装包变化（前 60 行）==="; head -60 /work/logs/pkg_versions_diff.txt || true
fi

# ---------- 头文件核对（两个发行版路径不同，全部候选都试） ----------
echo "=== 关键头文件检查 ==="
_find_hdr() {  # $1 = 多个候选路径（空格分隔）
  for p in $1; do [ -f "$p" ] && { echo "$p"; return 0; }; done
  return 1
}
check_hdr() { # $1=名字 $2=候选路径...
  local name=$1; shift
  if p=$(_find_hdr "$*"); then echo "OK   $name  ($p)"; else echo "MISS $name  (试过: $*)"; fi
}
check_hdr openssl  /usr/include/openssl/ssl.h
check_hdr zlib     /usr/include/zlib.h
check_hdr lzma     /usr/include/lzma.h
check_hdr bzlib    /usr/include/bzlib.h /usr/include/bzlib.h
check_hdr ffi      /usr/include/ffi.h /usr/include/*/ffi.h /usr/lib64/libffi/include/ffi.h
check_hdr sqlite3  /usr/include/sqlite3.h
check_hdr readline /usr/include/readline/readline.h
check_hdr ncurses  /usr/include/ncurses.h /usr/include/ncurses/ncurses.h
check_hdr panel    /usr/include/panel.h /usr/include/ncurses/panel.h
check_hdr gdbm     /usr/include/gdbm.h
check_hdr expat    /usr/include/expat.h
check_hdr crypt    /usr/include/crypt.h /usr/include/libxcrypt/crypt.h

echo "=== 确认缺失头文件（应与原构建一致为缺失）==="
for h in /usr/include/uuid/uuid.h /usr/include/db.h /usr/include/tk.h /usr/include/rpcsvc/yp_prot.h; do
  if [ -f "$h" ]; then echo "PRESENT(unexpected) $h"; else echo "absent(ok) $h"; fi
done

echo "=== 工具链 ==="
gcc --version | head -1
make --version | head -1
ld --version | head -1
echo "affinity cpus: $(python3 -c 'import os;print(len(os.sched_getaffinity(0)))' 2>/dev/null || echo '?')"
echo "=== 01_setup_build_env 完成 ==="
