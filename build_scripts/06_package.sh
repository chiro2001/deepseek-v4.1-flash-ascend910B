#!/bin/bash
# 产出交付物到 /work/out/
#   python3                  真实解释器二进制（来自 staging bin/python3.12）
#   libpython3.12.so.1.0     共享库
#   install_staging.tar.gz   完整 staging 树（tar 根为 /，解包到 / 即还原）
#   sourcetree_manifest.txt  原始 staging 结构清单（备查）
set -euo pipefail

STAGE=/staging/usr/local/python3.12.13
OUT=/work/out
mkdir -p "$OUT"

echo "=== [1] 导出两个关键文件 ==="
# 注意：镜像内 bin/python3 是指向 bin/python3.12 的符号链接，
# 真实二进制文件是 python3.12。这里把它导出为 python3（任务要求的命名），
# apply.sh 会自动识别目标的符号链接结构并落到正确位置。
cp -f "$STAGE/bin/python3.12" "$OUT/python3"
cp -f "$STAGE/lib/libpython3.12.so.1.0" "$OUT/libpython3.12.so.1.0"
chmod 755 "$OUT/python3" "$OUT/libpython3.12.so.1.0"

echo
echo "=== [2] 记录原始权限/md5/大小 ==="
{
  echo "# 原始 staging 文件属性（用于部署时对照）"
  echo "## bin/python3.12"
  ls -l "$STAGE/bin/python3.12"
  stat -c 'mode=%a size=%s mtime=%y' "$STAGE/bin/python3.12"
  md5sum "$STAGE/bin/python3.12"
  echo "## bin/python3 -> $(readlink "$STAGE/bin/python3")"
  echo "## lib/libpython3.12.so.1.0"
  ls -l "$STAGE/lib/libpython3.12.so.1.0"
  stat -c 'mode=%a size=%s mtime=%y' "$STAGE/lib/libpython3.12.so.1.0"
  md5sum "$STAGE/lib/libpython3.12.so.1.0"
  echo "## 符号链接"
  ls -l "$STAGE/lib/" | grep -E 'libpython3.12' || true
} | tee "$OUT/artifact_stat.txt"

echo
echo "=== [3] 打包完整 staging 树（tar 根 = /）==="
tar czf "$OUT/install_staging.tar.gz" -C /staging usr/local/python3.12.13
ls -l "$OUT/install_staging.tar.gz"

echo
echo "=== [4] staging 结构清单（前 40 行）==="
tar tzf "$OUT/install_staging.tar.gz" | head -40 | tee "$OUT/sourcetree_manifest_head.txt"
echo "总条目数: $(tar tzf "$OUT/install_staging.tar.gz" | wc -l)"

echo
echo "=== [5] stlib/site-packages 检查（ensurepip 是否装好 pip）==="
ls "$STAGE/lib/python3.12/site-packages/" 2>/dev/null | head -10
echo
echo "=== [6] out 目录清单 ==="
ls -l "$OUT"
