#!/usr/bin/env bash
# =============================================================================
# build_a3_tp8_image.sh —— 把"当前能在 A3 上跑通的 TP8 形态"固化成**派生镜像**
#
# 为什么需要"固化"这一步：
#   现在 A3 跑得通的形态 = 官方镜像 + **15 个 `-v` 挂载**（`PATCH_MODE=mount`）：
#     · `scripts/`                   → `/opt/dsv41/scripts`（serve_v2/serve_a3/run_test …）
#     · `patches/files/*.py`（13 个）→ `/vllm-workspace/vllm-ascend/vllm_ascend/...`
#     · `patches/admission_gate.patch` → `/opt/dsv41/admission_gate.patch`
#   内网 A3 上没有这些文件 ⇒ 只用官方镜像起不来。把挂载件**烘焙成一层**，
#   内网就只需要"官方基础镜像 + 这一层"。
#
# 它做什么：
#   ① 用 A3 入口的 **DRY_RUN** 取**真实挂载清单**（不手抄 —— "我以为挂了这些"是本仓
#      反复栽过的那类错）；
#   ② 按清单把宿主文件复制进 build context（**保持容器内目标路径**）；
#   ③ `docker build` ⇒ 派生镜像（只多 1 层，几百 KB 量级）；
#   ④ 自检：镜像内**每个文件 md5 == 宿主源文件**（判据绑内容，不绑路径）；
#   ⑤ 反例对照：确认基础镜像里那份**本来不同**（否则这层可能什么都没改）。
#
# 用法（a3-21 上，需要 docker 权限）：
#   bash tools/build_a3_tp8_image.sh
#   TAG=local/dsv41-a3-tp8:v1 bash tools/build_a3_tp8_image.sh
#
# 退出码：0 = 镜像已产出且自检通过；2 = 任一步失败（fail-closed）
# =============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PKG=${PKG:-$(cd "$HERE/.." && pwd)}
PAIRS_PY=$HERE/mount_list_pairs.py
BASE=${BASE:-quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3}
TAG=${TAG:-local/dsv41-a3-tp8:$(date +%Y%m%d-%H%M)}
# 模型只用来让 serve_a3.sh 的前置检查通过；本脚本**不读任何权重**
PROBE_MODEL=${PROBE_MODEL:-/tmp/_probe_model_a3tp8}
WORK=${WORK:-$HOME/dsv41-image-build}
CTX=$WORK/ctx-a3-tp8

say() { printf '\n\033[1m======== %s ========\033[0m\n' "$*"; }
ok()  { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
bad() { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; }

say "① 前置"
for c in docker python3; do
    command -v "$c" >/dev/null 2>&1 || { bad "没有 $c"; exit 2; }
done
[ -f "$PAIRS_PY" ] || { bad "缺 $PAIRS_PY"; exit 2; }
[ -f "$PKG/scripts/serve_a3.sh" ] || { bad "缺 $PKG/scripts/serve_a3.sh"; exit 2; }
docker image inspect "$BASE" >/dev/null 2>&1 || { bad "本地没有基础镜像 $BASE"; exit 2; }
ok "基础镜像：$BASE（$(docker inspect -f '{{len .RootFS.Layers}}' "$BASE") 层）"
ok "A3 入口：$PKG/scripts/serve_a3.sh"

mkdir -p "$PROBE_MODEL"
[ -f "$PROBE_MODEL/config.json" ] || printf '{"model_type":"deepseek_v41"}\n' > "$PROBE_MODEL/config.json"

say "② 取**真实挂载清单**（A3 入口 DRY_RUN）"
RAW=$WORK/mounts_raw.txt
mkdir -p "$WORK"
( cd "$PKG" && DEVS="0 1 2 3 4 5 6 7" DRY_RUN=1 MODEL="$PROBE_MODEL" \
    MAX_LEN=133120 MAX_SEQS=32 bash scripts/serve_a3.sh ) > "$RAW" 2>&1 || true
if ! python3 "$PAIRS_PY" --check "$RAW" > "$WORK/pairs_check.txt" 2>&1; then
    bad "挂载清单解析/校验失败："; cat "$WORK/pairs_check.txt" | sed 's/^/        /'
    tail -12 "$RAW" | sed 's/^/        /'; exit 2
fi
ok "$(cat "$WORK/pairs_check.txt")"
PAIRS=$WORK/mounts_pairs.tsv
python3 "$PAIRS_PY" "$RAW" > "$PAIRS" || { bad "导出 TSV 失败"; exit 2; }

say "③ 造 build context（保持容器内目标路径）"
[ -d "$CTX" ] && rm -rf "$CTX"
mkdir -p "$CTX"
_copied=0
while IFS=$'\t' read -r src dst mode; do
    [ -n "${src:-}" ] || continue
    [ -e "$src" ] || { bad "挂载源不存在：$src"; exit 2; }
    if [ -d "$src" ]; then
        mkdir -p "$CTX$dst"; cp -a "$src/." "$CTX$dst/"; ok "目录 $src → $dst"
    else
        mkdir -p "$CTX$(dirname "$dst")"; cp -a "$src" "$CTX$dst"
    fi
    _copied=$((_copied + 1))
done < "$PAIRS"
ok "已复制 $_copied 个挂载源（$(du -sh "$CTX" | cut -f1)）"

{
    echo "# 由 tools/build_a3_tp8_image.sh 生成：把 A3 的 TP8 工作形态固化成一层。"
    echo "# 基础镜像 = $BASE；本层只含 挂载件（补丁 + 起服脚本）。"
    echo "FROM $BASE"
    echo ""
    # ★ build context 的**目录布局已经镜像了容器内目标路径**
    #   （比如 `$CTX/opt/dsv41/scripts/serve_v2.sh`、`$CTX/vllm-workspace/vllm-ascend/...`）
    #   ⇒ 一条 `COPY . /` 就能全部落到正确位置，且**只产生 1 层**。
    #   （第一版按挂载逐条 COPY ⇒ 15 层；层数多会让"层补丁"包里的 tar 从 1 个变 15 个。）
    echo "# 目录布局已镜像容器内目标路径 ⇒ 一条 COPY 落地，只产生 1 层"
    echo "COPY . /"
    echo ""
    echo "# 挂载时是 ro/rw；烘焙后统一可写（容器内本来就是 root）"
    echo "LABEL dsv41.a3.tp8=1 dsv41.base=$BASE"
} > "$CTX/Dockerfile"
# 把 Dockerfile 自己排除掉，别让它被 COPY 进镜像
printf 'Dockerfile\n.dockerignore\n' > "$CTX/.dockerignore"
ok "Dockerfile 生成（$(grep -c '^COPY' "$CTX/Dockerfile") 条 COPY；.dockerignore 已排除 Dockerfile 自身）"

say "④ docker build"
if docker build -t "$TAG" "$CTX" > "$WORK/build.log" 2>&1; then
    ok "构建成功：$TAG（$(docker inspect -f '{{len .RootFS.Layers}}' "$TAG") 层）"
else
    bad "构建失败（$WORK/build.log）"; tail -20 "$WORK/build.log" | sed 's/^/        /'; exit 2
fi

say "⑤ 自检：镜像内每个文件 md5 == 宿主源文件（判据绑内容）"
_v=0; _n=0
while IFS=$'\t' read -r src dst mode; do
    [ -n "${src:-}" ] || continue
    if [ -d "$src" ]; then
        while IFS= read -r f; do
            rel=${f#"$src"/}
            h1=$(md5sum "$f" | cut -d' ' -f1)
            h2=$(docker run --rm --entrypoint md5sum "$TAG" "$dst/$rel" 2>/dev/null | cut -d' ' -f1)
            _n=$((_n+1))
            [ "$h1" = "$h2" ] || { bad "不一致：$dst/$rel（宿主 ${h1:0:12} / 镜像 ${h2:0:12}）"; _v=1; }
        done < <(find "$src" -type f)
    else
        h1=$(md5sum "$src" | cut -d' ' -f1)
        h2=$(docker run --rm --entrypoint md5sum "$TAG" "$dst" 2>/dev/null | cut -d' ' -f1)
        _n=$((_n+1))
        [ "$h1" = "$h2" ] || { bad "不一致：$dst（宿主 ${h1:0:12} / 镜像 ${h2:0:12}）"; _v=1; }
    fi
done < "$PAIRS"
if [ "$_v" = "0" ]; then ok "全部 $_n 个文件逐字节一致"
else bad "有文件不一致 ⇒ 产物不可用"; exit 2; fi

_bh=$(docker run --rm --entrypoint md5sum "$BASE" \
    /vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_hash.py 2>/dev/null | cut -d' ' -f1)
_nh=$(docker run --rm --entrypoint md5sum "$TAG" \
    /vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_hash.py 2>/dev/null | cut -d' ' -f1)
if [ -n "$_bh" ] && [ "$_bh" != "$_nh" ]; then
    ok "反例对照：engram_hash.py 基础=${_bh:0:12} vs 新=${_nh:0:12}（这层确实改了东西）"
else
    printf '  \033[33mWARN\033[0m  反例对照：engram_hash.py 与基础镜像相同或读不到（base=%s new=%s）\n' "${_bh:0:12}" "${_nh:0:12}"
fi

{
    echo "TAG=$TAG"
    echo "BASE=$BASE"
    echo "LAYERS=$(docker inspect -f '{{len .RootFS.Layers}}' "$TAG")"
    echo "SIZE=$(docker image inspect -f '{{.Size}}' "$TAG")"
    echo "BUILT_AT=$(date -Is)"
    echo "HOST=$(hostname)"
} | tee "$WORK/image-info.txt"

cat <<EOF

✓ 派生镜像就绪：$TAG

下一步：用它造"层补丁"包（只带基础镜像之上的层）
  sudo python3 $HERE/make_image_patch_kit.py --out <kit 目录> --base $BASE \\
       --job 'dsv41-a3-tp8|$TAG|both'
  （把该工具的 KIND 换成我们自己的 v41 一档；见 tools/make_image_patch_kit.py 的说明）
EOF
