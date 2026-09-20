#!/bin/bash
# CPython 3.12.13 PGO+LTO 完整构建（后台运行，日志落 /work/logs/make.log）
# --enable-optimizations 使默认 all 规则 = profile-opt：
#   1) build_all_generate_profile  (-fprofile-generate)
#   2) 运行 PROFILE_TASK 训练   (默认: -m test --pgo --timeout=1200)
#   3) build_all_use_profile      (-fprofile-use -fprofile-correction)
#
# 环境变量：
#   PGO_JOBS            并行度（默认 48）
#   PGO_IMAGE_LIBDIR    镜像内自带的 libpython 目录（默认 /usr/local/python3.12.13/lib）
set -uo pipefail
cd /work/src/cpython-3.12.13

# ---------------------------------------------------------------------------
# 隔离镜像内自带的同 prefix libpython（A2 真机实测踩到）
# ---------------------------------------------------------------------------
# 本镜像已装好一份**非 PGO** 的 CPython（prefix=/usr/local/python3.12.13）。
# 新编出的 ./python 带 DT_RPATH=/usr/local/python3.12.13/lib，而 ld.so 的搜索顺序是
#   DT_RPATH **先于** LD_LIBRARY_PATH
# 于是它抓到镜像里那份旧 libpython —— 那份没有 gcov 运行库符号，报：
#   ./python: undefined symbol: __gcov_indirect_call
# 这里只在**构建容器内**把旧库移到隔离目录（构建容器本来也不用它，move 后 ld.so 找不到
# RPATH 目标，就会退到 Makefile 设的 LD_LIBRARY_PATH=. 去用**新编**的 libpython）。
# 产物部署时带的是自己新编的那份，不受影响。
#
# 幂等性：第二次运行时 $OLD_LIBDIR 下已经没有 libpython ⇒ 整块直接跳过（不会报错）。
# 隔离目录在宿主可见（/work = $PKG/optim/pgo），且落在 .gitignore 覆盖的 build/logs 下。
OLD_LIBDIR=${PGO_IMAGE_LIBDIR:-/usr/local/python3.12.13/lib}
QDIR=/work/logs/image-libpython-quarantine
if [ -e "$OLD_LIBDIR/libpython3.12.so.1.0" ]; then
  mkdir -p "$QDIR"
  if mv -f "$OLD_LIBDIR"/libpython3.12.so* "$QDIR"/ 2>/dev/null; then
    echo "=== 已隔离镜像自带 libpython（$OLD_LIBDIR → $QDIR）==="
    echo "    原因：DT_RPATH 优先于 LD_LIBRARY_PATH，旧库会让 ./python 报 __gcov_indirect_call"
  else
    echo "!!! 隔离失败：$OLD_LIBDIR/libpython3.12.so* 没能移到 $QDIR（权限？）"
    echo "    make 可能仍会抓到旧库并报 undefined symbol: __gcov_indirect_call"
  fi
else
  echo "=== 镜像内 $OLD_LIBDIR 没有 libpython ⇒ 无需隔离（重复运行时走这里）==="
fi

# 上次失败可能留下空/半成品的 pybuilddir.txt、platform，
# 不删的话 make 会认为它们"已是最新"而直接跳过重生成。
rm -f pybuilddir.txt platform

date -Is > /work/logs/make_start_iso.txt
START=$(date +%s)
echo "$START" > /work/logs/make_start_epoch.txt
echo "START=$START ($(date -Is))" > /work/logs/make.log

JOBS=${PGO_JOBS:-48}
echo "JOBS=$JOBS  PROFILE_TASK=${PROFILE_TASK:-<CPython 默认 -m test --pgo>}" >> /work/logs/make.log
# PROFILE_TASK 保持 CPython 默认（-m test --pgo）。它的收益已被实测证明可跨负载转移：
# 训练用测试套件，而在 tiny_call / dict_loop / list_append 等**完全不同**的模式上
# 仍拿到 −16~23%（reports/cpython-pgo-verified.md）。
nice -n 10 make -j"$JOBS" >> /work/logs/make.log 2>&1
RC=$?
echo "$RC" > /work/logs/make_exit.txt

END=$(date +%s)
echo "$END" > /work/logs/make_end_epoch.txt
echo "$((END-START))" > /work/logs/make_seconds.txt
date -Is > /work/logs/make_end_iso.txt
echo "DONE rc=$RC elapsed=$((END-START))s at $(date -Is)" >> /work/logs/make.log
