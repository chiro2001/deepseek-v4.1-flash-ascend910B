#!/usr/bin/env bash
# =============================================================================
# build_payload.sh —— 从「发布产物」组装可安装的 payload 树
#
# 本包是**二进制面**（自研算子的 OPP 树 + 重编的 torch 扩展），不是从源码现编。
# 二进制由 experimental/fusion-3out/ 的源码构建得到，构建步骤见 REPRODUCE.md；
# 本脚本只管「把已构建好的产物摆成可安装的形状」。
#
# 用法：
#   bash build_payload.sh                          # 用包内 artifacts/*.tgz（默认）
#   bash build_payload.sh --artifacts <tar.tgz>    # 用指定的产物包
#   bash build_payload.sh --from-container <name>   # 直接从运行中的容器取（开发用）
#
# 产出：deploy/a3-tp8-fusion/payload/
#   payload/{opp,so,py}/       见 PAYLOAD.md
#   payload/install.sh         目标机上一条命令安装
#   payload/PAYLOAD.sha256     逐文件 sha256，供跨机核对
#
# 注意：payload/ 是生成物，不要手改；改内容请改源码后重新构建（见 REPRODUCE.md）。
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/payload"
ARTIFACTS=""
FROM_CONTAINER=""

say() { printf '\033[1m[payload]\033[0m %s\n' "$*"; }
die() { printf '\033[1m[payload][FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --artifacts)      ARTIFACTS="$2"; shift 2 ;;
        --from-container) FROM_CONTAINER="$2"; shift 2 ;;
        --out)            OUT="$2"; shift 2 ;;
        -h|--help)        sed -n '2,20p' "$0"; exit 0 ;;
        *) die "未知参数 $1" ;;
    esac
done

SO_NAME=vllm_ascend_C.cpython-312-aarch64-linux-gnu.so
SO_DST=/vllm-workspace/vllm-ascend/vllm_ascend
VLLM_ASC=/vllm-workspace/vllm-ascend/vllm_ascend

# ---------------------------------------------------------------- 取得产物
_tmp=""
cleanup() { [ -n "$_tmp" ] && rm -rf "$_tmp"; return 0; }
trap cleanup EXIT
_tmp="$(mktemp -d)"
SRC="$_tmp"

if [ -n "$FROM_CONTAINER" ]; then
    command -v docker >/dev/null 2>&1 || die "要用 --from-container 需要 docker"
    docker inspect "$FROM_CONTAINER" >/dev/null 2>&1 || die "找不到容器 $FROM_CONTAINER"
    say "从容器 $FROM_CONTAINER 取产物…"
    rm -rf "$SRC"; mkdir -p "$SRC/opp" "$SRC/so" "$SRC/py"
    # ★ 用容器内 tar 管道而不是 `docker cp <ct>:/dir/.`：
    #   docker cp 在这类目录上会因符号链接报
    #   `evalSymlinksInScope: ... is not in ...`（实测），tar 没有这个问题，
    #   而且能一并保留权限位（OPP 里有可执行脚本）。
    docker exec "$FROM_CONTAINER" tar -C /vllm-workspace/3out_opp -cf - . \
        | tar -C "$SRC/opp" -xf - || die "取 opp 失败"
    docker cp "$FROM_CONTAINER:$SO_DST/$SO_NAME" "$SRC/so/" || die "取 .so 失败"
    docker cp "$FROM_CONTAINER:$VLLM_ASC/models/deepseek_v4/model.py" "$SRC/py/model.py" || die "取 model.py 失败"
    docker cp "$FROM_CONTAINER:$VLLM_ASC/attention/dsa_v41.py" "$SRC/py/dsa_v41.py" || die "取 dsa_v41.py 失败"
    # 容器内的权限位可能是 700，统一放宽到可读（发布包要能被普通用户解开）
    chmod -R u+rwX,go+rX "$SRC" 2>/dev/null || true
else
    [ -n "$ARTIFACTS" ] || ARTIFACTS="$HERE/artifacts/fusion-artifacts.tgz"
    [ -f "$ARTIFACTS" ] || die "找不到产物包 $ARTIFACTS（先用 --from-container，或指定 --artifacts）"
    say "解包产物 $ARTIFACTS …"
    rm -rf "$SRC"; mkdir -p "$SRC"
    tar -xzf "$ARTIFACTS" -C "$SRC" || die "解包失败"
    # 产物包是在构建机上打的，权限位可能很紧，统一放宽
    chmod -R u+rwX,go+rX "$SRC" 2>/dev/null || true
fi

for d in opp so py; do
    [ -d "$SRC/$d" ] || die "产物缺少 $d/ 目录"
done
[ -s "$SRC/so/$SO_NAME" ] || die "缺少或空的 torch 扩展 $SO_NAME"
[ -s "$SRC/py/model.py" ] || die "缺少 model.py"
[ -s "$SRC/py/dsa_v41.py" ] || die "缺少 dsa_v41.py"

# ★ 完整性校验：逐个算子查「kernel 二进制 + aclnn 头 + tiling」三件套都在。
#   只查目录存在是不够的 —— `docker cp` 在带符号链接/权限的目录上会**静默漏文件**
#   （实测：一个 84 文件的 OPP 树被取成 55 个，而顶层目录一个不少）。
VENDOR="$SRC/opp/vendors/custom_transformer"
KERNEL_DIR="$VENDOR/op_impl/ai_core/tbe/kernel/ascend910_93"
for op in rms_norm_dynamic_quant rms_norm_dynamic_quant_bf16; do
    [ -d "$KERNEL_DIR/$op" ] || die "OPP 树里缺少算子目录 $op"
    n_o=$(find "$KERNEL_DIR/$op" -name '*.o' | wc -l)
    [ "$n_o" -ge 1 ] || die "算子 $op 的 kernel 目录里没有 .o（树不完整）"
    [ -f "$VENDOR/op_api/include/aclnnop/aclnn_$op.h" ] || die "缺少 aclnn 头 aclnn_$op.h"
done
[ -f "$VENDOR/op_api/lib/libcust_opapi.so" ] || die "缺少 libcust_opapi.so"
[ -f "$VENDOR/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so" ] \
    || die "缺少 libcust_opmaster_rt2.0.so（tiling）"
n_opp=$(find "$SRC/opp" -type f | wc -l)
[ "$n_opp" -ge 80 ] || die "OPP 树只有 $n_opp 个文件，明显不完整（期望 ≥80）"

# Python 件必须是已打补丁的版本，否则装上去等于没融合（且不会报错）
grep -q "LNORM-FUSE" "$SRC/py/model.py" || die "model.py 没有 LNORM-FUSE 标记（拿到的不是已打补丁版本）"
grep -q "LNORM-FUSE" "$SRC/py/dsa_v41.py" || die "dsa_v41.py 没有 LNORM-FUSE 标记"

# ---------------------------------------------------------------- 摆成 payload
say "组装 payload → $OUT"
rm -rf "$OUT"; mkdir -p "$OUT"
cp -a "$SRC/opp" "$OUT/opp"
cp -a "$SRC/so"  "$OUT/so"
cp -a "$SRC/py"  "$OUT/py"
cp -p "$HERE/install.sh"  "$OUT/install.sh"
cp -p "$HERE/PAYLOAD.md"  "$OUT/PAYLOAD.md"
cp -p "$HERE/SWITCHES.md" "$OUT/SWITCHES.md"

# ---------------------------------------------------------------- 逐文件 sha256
# 清单覆盖 payload 里全部文件（含文档）：文档是否被改也是包身份的一部分。
# 只列 opp/so/py 会漏掉 install.sh，而安装脚本被篡改正是最该抓的。
# ★ 必须排除 PAYLOAD.sha256 自己：重定向会**先创建空文件**，
#   若把它收进清单，存的是"空文件的哈希"，之后永远校验失败（实测踩到）。
(
    cd "$OUT"
    find . -type f ! -name 'PAYLOAD.sha256' -print0 | LC_ALL=C sort -z | sed -z 's|^\./||' | xargs -0 sha256sum
) > "$OUT/PAYLOAD.sha256"

say "清单自检（sha256sum -c）…"
( cd "$OUT" && sha256sum -c --quiet PAYLOAD.sha256 ) || die "刚生成的清单自校验失败"

say "文件数：$(find "$OUT" -type f | wc -l)"
say "体积  ：$(du -sh "$OUT" | cut -f1)"
say "OPP 算子：$(ls "$OUT/opp/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910_93/" | tr '\n' ' ')"
say "OK payload 就绪：$OUT"
