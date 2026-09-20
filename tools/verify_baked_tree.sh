#!/usr/bin/env bash
# =============================================================================
# verify_baked_tree.sh —— 校验「烘焙进镜像的补丁文件」是否与发布包逐字节一致
#
# 用法（镜像内，由 scripts/build_image.sh 调用）：
#     bash /tmp/dsv41-verify.sh --root "$ASCEND_PKG" --manifest /tmp/dsv41-chk.tsv --image-extras
# 本地/CI 也能跑（**不需要 docker**），用于给这套检查本身做正/负控：
#     bash tools/verify_baked_tree.sh --root /tmp/fake-tree --manifest /tmp/m.tsv
#
# 清单由 `tools/check_checksums.py --manifest` 生成，格式：kind<TAB>目标路径<TAB>md5
#   inst = 覆盖已有文件（**必须先有 .a2orig 备份**，回滚要用）
#   newf = 新增文件
#
# 为什么清单要「生成」而不是手写在脚本里：见 tools/check_checksums.py 头部 ——
# v7→v8 就是 build_image.sh 里那份手写 md5 表没跟上，用户跑到最后一步才报 checksum 失败。
# =============================================================================
set -uo pipefail

ROOT=""
MANIFEST=""
IMAGE_EXTRAS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --root)         ROOT=${2:-}; shift 2 ;;
    --manifest)     MANIFEST=${2:-}; shift 2 ;;
    --image-extras) IMAGE_EXTRAS=1; shift ;;
    -h|--help)      sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "[verify][FAIL] 未知参数：$1" >&2; exit 2 ;;
  esac
done

ROOT=${ROOT:-${ASCEND_PKG:-/vllm-workspace/vllm-ascend/vllm_ascend}}
[ -d "$ROOT" ] || { echo "[verify][FAIL] 目录不存在：$ROOT" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "[verify][FAIL] 清单不存在：$MANIFEST" >&2; exit 2; }

rc=0
n_ok=0
# py_compile 只用来判"语法可加载"，落点放到临时目录 —— 这样即使树是**只读挂载**
# （本地干跑、或以后把运行时目录挂成 ro）也不会因为写不了 __pycache__ 误报 FAIL。
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/dsv41-pycache"
while IFS=$'\t' read -r kind dst want; do
  [ -n "${kind:-}" ] || continue
  [ -n "${dst:-}" ]  || continue
  t="$ROOT/$dst"
  if [ ! -f "$t" ]; then echo "  FAIL 缺文件 $dst"; rc=1; continue; fi
  got=$(md5sum "$t" | cut -d' ' -f1)
  if [ "$got" != "$want" ]; then
    echo "  FAIL md5 $dst: got=$got want=$want"; rc=1; continue
  fi
  if ! python3 -m py_compile "$t" 2>/dev/null; then
    echo "  FAIL py_compile $dst"; rc=1; continue
  fi
  if [ "$kind" = "inst" ]; then
    [ -f "$t.a2orig" ] || { echo "  FAIL 缺备份 $dst.a2orig（回滚要用）"; rc=1; }
  fi
  n_ok=$((n_ok+1))
  echo "  OK  $got  $dst"
done < "$MANIFEST"

if [ "$IMAGE_EXTRAS" = "1" ]; then
  # 未验证项**不应**被装进运行时，只在 /opt/dsv41/ 里待命
  for f in patches/draft/dspark_proposer.py patches/draft/llm_base_proposer.py \
           patches/files/token_dispatcher_moezero.py scripts/serve_v2.sh \
           scripts/serve_a2.sh BUILD_INFO.txt; do
    [ -f "/opt/dsv41/$f" ] || { echo "  FAIL 缺 /opt/dsv41/$f"; rc=1; }
  done
  grep -q libjemalloc /opt/dsv41/scripts/serve_v2.sh \
    || { echo "  FAIL serve_v2 未启用 jemalloc"; rc=1; }
fi

if [ "$rc" = "0" ]; then
  echo "  [verify] 共 $n_ok 项，全部逐字节一致 ✅"
else
  echo "  [verify] 有文件不一致 ❌" >&2
fi
exit "$rc"
