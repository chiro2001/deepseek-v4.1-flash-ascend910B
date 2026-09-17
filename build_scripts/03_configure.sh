#!/bin/bash
# CPython 3.12.13 PGO+LTO configure
#   目标：精确复现原镜像构建（--enable-shared, prefix=/usr/local/python3.12.13,
#         LDFLAGS=-Wl,-rpath,/usr/local/python3.12.13/lib），额外加 PGO 与 LTO
set -euo pipefail
cd /work/src/cpython-3.12.13

export TZ=Asia/Shanghai
START=$(date +%s)

echo "=== 目标镜像原始 CONFIG_ARGS（对照基线）==="
echo "'--enable-shared' 'LDFLAGS=-Wl,-rpath /usr/local/python3.12.13/lib' '--prefix=/usr/local/python3.12.13'"
echo
echo "=== 本次 configure 开始 $(date -Is) ==="

# [MTUNE] 可选实验：`PGO_MTUNE=1` 时加 `-mtune=native`。
#
# 为什么默认关：
#   * 现有构建参数里**没有** -march/-mtune ⇒ 代码生成是通用 aarch64；
#     实证：构建日志里 `grep -oE '\-m(arch|tune|cpu)=' make.log` 为空。
#   * `-mtune` 只改调度/展开启发式，**不生成新指令** ⇒ 安全（不会 SIGILL），
#     但收益预期只有 0~3%，且**必须实测**（A/B 同会话对比 ms/step）。
#   * `-march=native` 会生成新指令，一旦搬到别的 CPU 就 SIGILL —— **不提供**。
EXTRA_CFLAGS=""
if [ "${PGO_MTUNE:-0}" = "1" ]; then
  EXTRA_CFLAGS="-mtune=native"
  echo "  ★ PGO_MTUNE=1 → 追加 CFLAGS=$EXTRA_CFLAGS（安全，不生成新指令）"
  echo "    注意：跨 CPU 搬运此产物会失去该项收益（但不影响正确性）"
fi

# shellcheck disable=SC2086
nice -n 10 ./configure \
    --prefix=/usr/local/python3.12.13 \
    --enable-shared \
    --enable-optimizations \
    --with-lto \
    ${EXTRA_CFLAGS:+CFLAGS="$EXTRA_CFLAGS"} \
    LDFLAGS="-Wl,-rpath,/usr/local/python3.12.13/lib" \
    > /work/logs/configure.log 2>&1

END=$(date +%s)
echo "=== configure 完成 $(date -Is) 用时 $((END-START))s ==="

echo
echo "=== 生成的 CONFIG_ARGS ==="
./python -c "import sysconfig; print(sysconfig.get_config_var('CONFIG_ARGS'))" 2>/dev/null || \
  grep -m1 '^CONFIG_ARGS=' Makefile

echo
echo "=== 关键构建变量 ==="
grep -E '^(CFLAGS|CFLAGS_NODIST|LDFLAGS|LDFLAGS_NODIST|OPT|CONFIGURE_CFLAGS|CONFIGURE_LDFLAGS|PY_CFLAGS|PROFILE_TASK|DEF_MAKE_ALL_RULE|DEF_MAKE_RULE|LTOFLAGS|AR|CC)=' Makefile | head -30

echo
echo "=== LTO / PGO 相关 configure 结论 ==="
grep -iE "checking for --with-lto|checking for --enable-optimizations|Link-Time-Optimization|checking whether to enable optimizations|checking for gcc-ar|checking for --with-computed-gotos" /work/logs/configure.log | head -20

echo
echo "=== 模块检测结果（对照原镜像）==="
grep -E '^MODULE_(FFI|ZLIB|BZIP2|LZMA|SSL|HASHLIB|SQLITE3|READLINE|CURSES|GDBM|DBM|UUID|TKINTER|NIS|CRYPT|DECIMAL|EXPAT)_STATE' Makefile | sort

echo
echo "=== configure 阶段耗时 $((END-START))s（明细见 /work/logs/configure.log）==="
echo "$((END-START))" > /work/logs/configure_seconds.txt
