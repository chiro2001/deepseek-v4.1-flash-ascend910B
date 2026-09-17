#!/bin/bash
# CPython 3.12.13 安装到 DESTDIR staging
#   - 保持内建 prefix = /usr/local/python3.12.13（不动 configure 的 prefix）
#   - 产物落 /staging/usr/local/python3.12.13，可直接覆盖进服务容器
set -euo pipefail
cd /work/src/cpython-3.12.13

rm -rf /staging
mkdir -p /staging

START=$(date +%s)
echo "=== make install DESTDIR=/staging 开始 $(date -Is) ==="
nice -n 10 make -j"${PGO_JOBS:-48}" install DESTDIR=/staging > /work/logs/install.log 2>&1
RC=$?
END=$(date +%s)
echo "=== 结束 rc=$RC 用时 $((END-START))s ==="
echo "$((END-START))" > /work/logs/install_seconds.txt
echo "$RC" > /work/logs/install_exit.txt

echo
echo "=== staging 树 ==="
ls -l /staging/usr/local/python3.12.13/bin/ | head -20
echo "..."
ls -l /staging/usr/local/python3.12.13/lib/libpython3.12.so*
echo
echo "=== staging 内解释器自检（DESTDIR 场景用 LD_LIBRARY_PATH 找库）==="
LD_LIBRARY_PATH=/staging/usr/local/python3.12.13/lib \
  /staging/usr/local/python3.12.13/bin/python3.12 -VV
echo
echo "=== 内建 prefix 确认（必须仍是 /usr/local/python3.12.13）==="
LD_LIBRARY_PATH=/staging/usr/local/python3.12.13/lib \
  /staging/usr/local/python3.12.13/bin/python3.12 -c \
  "import sys,sysconfig; print('prefix=',sys.prefix); print('CONFIG_ARGS=',sysconfig.get_config_var('CONFIG_ARGS')); print('Py_ENABLE_SHARED=',sysconfig.get_config_var('Py_ENABLE_SHARED'))"

echo
echo "=== du -sh staging ==="
du -sh /staging/usr/local/python3.12.13
