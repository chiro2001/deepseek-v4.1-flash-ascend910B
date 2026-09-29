#!/usr/bin/env bash
# =============================================================================
# package_release.sh —— 把 V4.1 DCP8 的 overlay 打成**可复现、可校验**的发布包
#
# 为什么需要它：这套 DCP 实现不是改镜像，而是通过 `V41_DCP_MOUNT=<dir>`
#   把 12 个 `.py` 整树挂进容器（`scripts/serve_a2.sh` 里有 md5 守门）。
#   挂载方案好用在"改一行就能试"，坏处在"发出去之后无法证明对方跑的是哪一份"——
#   而本仓已经在"我以为挂了新版、实际是旧版"上踩过多次。
#   ⇒ 发布包必须自带：逐文件 sha256、构建时的 git 身份、以及**能自证**的校验器。
#
# 用法：
#   bash package_release.sh build  [outdir]      # 产出 tar.zst + .sha256
#   bash package_release.sh verify <tarball>     # 解包并逐文件校验（含负控）
#   bash package_release.sh selftest             # 造一个被篡改的包，证明 verify 抓得住
#
# 产物命名：v41-dcp-overlay-<commit12>-<md5(overlay)>.tar.zst
#   `--mtime` 固定为提交时间、owner/group 归零、`--sort=name`
#   ⇒ **同一份输入必然得到同一个 sha256**（可用它当"这就是我跑的那版"的身份）。
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$HERE"                      # experimental/v41-dcp
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

die() { echo "[pkg][FAIL] $*" >&2; exit 1; }
say() { echo "[pkg] $*"; }

# ★ 不要用 `trap ... RETURN`：它是**进程级**的，嵌套函数返回时会二次触发，
#   而此时被引用的 `local` 已经出作用域 ⇒ `set -u` 直接报 unbound variable
#   （2026-09-29 踩过）。统一登记到数组，只在 EXIT 清一次。
_TMPDIRS=()
new_tmpdir() {
    local d; d="$(mktemp -d)"
    _TMPDIRS+=("$d")
    printf '%s' "$d"
}
_cleanup() {
    # ★ 末尾显式 `return 0`：EXIT trap 的返回值会**泄漏成脚本退出码**
    #   （空数组在这里展开成一个空串 ⇒ `[ -n "" ]` 返回 1 ⇒ 整个脚本退出码变 1）。
    local d
    for d in ${_TMPDIRS[@]+"${_TMPDIRS[@]}"}; do
        [ -n "$d" ] && rm -rf "$d"
    done
    return 0
}
trap _cleanup EXIT

# ---------- 收集要打包的文件（排除 __pycache__ / .pyc / dist）----------
list_members() {
    ( cd "$PKG_ROOT" && find overlay launch tools \
        -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print | LC_ALL=C sort )
}

overlay_fingerprint() {
    # overlay 的聚合指纹：逐文件 sha256 再 sha256（与 BUILD_INFO 里那张表同源）
    ( cd "$PKG_ROOT" && find overlay -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 \
        | LC_ALL=C sort -z | xargs -0 sha256sum ) | sha256sum | cut -c1-12
}

git_info() {
    git -C "$REPO_ROOT" rev-parse --short=12 HEAD 2>/dev/null || echo "nogit"
}
git_branch() {
    git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "nogit"
}
git_mtime() {
    git -C "$REPO_ROOT" log -1 --format=%ct 2>/dev/null || echo 0
}

# ---------- 生成 BUILD_INFO.txt 与 MANIFEST.sha256 ----------
write_metadata() {
    local stage="$1"
    {
        echo "# V4.1 DCP8 overlay 发布包 —— 构建身份"
        # ★ 不写"当前时间"：那会让同一份输入两次构建得到不同 sha256。
        #   用**提交时间** ⇒ 包内容与 sha256 都是确定的（可复现构建）。
        echo "# 构建基准时间: $(date -d "@$(git_mtime)" -Is 2>/dev/null || echo 0)"
        echo "# 构建主机   : $(hostname)"
        echo "# 仓库       : $REPO_ROOT"
        echo "# 分支       : $(git_branch)"
        echo "# 提交       : $(git_info)"
        echo "# 提交时间   : $(date -d "@$(git_mtime)" -Is 2>/dev/null || echo unknown)"
        echo "# overlay 指纹: $(overlay_fingerprint)"
        echo "#"
        echo "# 用法（在目标机上）："
        echo "#   1) 解包到任意目录 D"
        echo "#   2) 起服时传 V41_DCP_MOUNT=D/overlay"
        echo "#      （serve_a2.sh 会在起服后逐文件 md5 比对，不一致直接 die）"
        echo "#   3) 或先跑： bash tools/dcp_sync.sh   # 同步到 \$HOST:\$DST 并核对 md5"
        echo "#"
        echo "# 本包的 sha256 由同目录的 .sha256 文件给出；内容身份见下（逐文件 sha256）。"
    } > "$stage/BUILD_INFO.txt"

    {
        echo "# MANIFEST.sha256 —— 逐文件 sha256（相对包根）"
        echo "# 校验：cd <解包目录> && sha256sum -c MANIFEST.sha256"
        # ★ 覆盖包内**全部**文件（含 README.md）——只列 overlay/launch/tools 会漏掉
        #   README，而"文档没被篡改"同样属于包身份的一部分。
        #   两个元数据文件自身不能进清单（会递归）。
        ( cd "$stage" && find . -type f ! -path '*/__pycache__/*' ! -name '*.pyc' \
            ! -name 'MANIFEST.sha256' ! -name 'BUILD_INFO.txt' -print0 \
            | LC_ALL=C sort -z | sed -z 's|^\./||' | xargs -0 sha256sum )
    } > "$stage/MANIFEST.sha256"
}

# ---------- build ----------
do_build() {
    local outdir="${1:-$PKG_ROOT/dist}"
    mkdir -p "$outdir"
    local stage; stage="$(new_tmpdir)"

    say "收集文件…"
    while IFS= read -r rel; do
        [ -n "$rel" ] || continue
        mkdir -p "$stage/$(dirname "$rel")"
        cp -p "$PKG_ROOT/$rel" "$stage/$rel"
    done < <(list_members)

    cp -p "$PKG_ROOT/RELEASE.md" "$stage/README.md" 2>/dev/null || true

    write_metadata "$stage"

    local name="v41-dcp-overlay-$(git_info)-$(overlay_fingerprint)"
    local tarball="$outdir/$name.tar.zst"
    local mtime; mtime="$(git_mtime)"

    say "打包（可复现参数：owner/group=0, mtime=@$mtime, sort=name）…"
    if command -v zstd >/dev/null 2>&1; then
        # ★ `-f`：同名产物（同 commit + 同 overlay 指纹 ⇒ 同一个名字）要能覆盖重建，
        #   否则第二次构建会静默保留旧包 —— 而"重建"正是可复现性验证的动作。
        tar --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$mtime" \
            -C "$stage" -cf - . | zstd -q -f -19 -o "$tarball"
    else
        tarball="$outdir/$name.tar.gz"
        tar --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$mtime" \
            -C "$stage" -cf - . | gzip -9n > "$tarball"
    fi
    # ★ 写**完整行**（`<hash>  <basename>`）而不是只写哈希：
    #   只有哈希的那种文件 `sha256sum -c` 直接用不了
    #   （`no properly formatted checksum lines found`，2026-09-29 实测被绊）。
    #   标准格式才能让使用者一条命令自证，而不必手抄哈希比对。
    ( cd "$(dirname "$tarball")" && sha256sum "$(basename "$tarball")" ) > "$tarball.sha256"

    say "产物：$tarball  ($(du -h "$tarball" | cut -f1))"
    say "sha256：$(cat "$tarball.sha256")"
    echo
    say "自检（verify 同一条路径，确保发出去的包自己能验过）："
    do_verify "$tarball"
}

# ---------- verify ----------
extract_tarball() {   # <tarball> <destdir>
    local tb="$1" dest="$2"
    mkdir -p "$dest"
    case "$tb" in
        *.tar.zst) command -v zstd >/dev/null 2>&1 || { echo "[pkg][FAIL] 缺 zstd，无法解 .tar.zst" >&2; return 1; }
                   zstd -q -dc "$tb" | tar -C "$dest" -xf - ;;
        *.tar.gz)  tar -C "$dest" -xzf "$tb" ;;
        *.tar)     tar -C "$dest" -xf "$tb" ;;
        *) echo "[pkg][FAIL] 未知归档格式：$tb" >&2; return 1 ;;
    esac
}

# ★ 本函数**只 return，不 exit** —— 因为它同时被 selftest 的负控调用；
#   若在里面 `exit`，被重定向的负控会连"检出"都打印不出来（2026-09-29 踩过）。
verify_impl() {   # <tarball>；0=通过，1=失败（已打印原因）
    local tb="${1:?usage: verify <tarball>}"
    [ -f "$tb" ] || { echo "[pkg][FAIL] 找不到 $tb" >&2; return 1; }
    local dest; dest="$(new_tmpdir)"
    extract_tarball "$tb" "$dest" || return 1

    [ -f "$dest/MANIFEST.sha256" ] || { echo "[pkg][FAIL] 包内没有 MANIFEST.sha256（不是本脚本产出的包？）" >&2; return 1; }
    [ -f "$dest/BUILD_INFO.txt" ]  || { echo "[pkg][FAIL] 包内没有 BUILD_INFO.txt" >&2; return 1; }

    if [ -f "$tb.sha256" ]; then
        local expect got
        # 兼容两种写法：完整行（`<hash>  <name>`）或只有哈希
        expect="$(awk 'NR==1{print $1}' "$tb.sha256")"
        got="$(sha256sum "$tb" | awk '{print $1}')"
        if [ "$got" != "$expect" ]; then
            echo "[pkg][FAIL] 归档自身 sha256 不符：期望 $expect 实得 $got" >&2
            return 1
        fi
        say "归档 sha256 相符：$(echo "$expect" | cut -c1-16)…"
    fi

    say "逐文件校验…"
    if ! ( cd "$dest" && sha256sum -c --quiet MANIFEST.sha256 ); then
        echo "[pkg][FAIL] MANIFEST 校验失败（文件被篡改或缺失）" >&2
        return 1
    fi

    # 顺带证明"使用者拿到包后能一条命令自证归档完整性"（.sha256 是标准格式）
    if [ -f "$tb.sha256" ] && command -v sha256sum >/dev/null 2>&1; then
        ( cd "$(dirname "$tb")" && sha256sum -c --quiet "$(basename "$tb").sha256" ) \
            || { echo "[pkg][FAIL] \`sha256sum -c\` 自证失败（.sha256 格式或内容不对）" >&2; return 1; }
        say "使用者侧自证通过：sha256sum -c <包>.sha256"
    fi

    local n; n="$(grep -c . "$dest/MANIFEST.sha256" 2>/dev/null || true)"
    say "文件数：$(( n > 2 ? n - 2 : n ))"
    say "构建身份：$(grep -E '^# (提交|overlay 指纹)' "$dest/BUILD_INFO.txt" | tr '\n' ' ')"
    say "✅ 校验通过：$tb"
    return 0
}

do_verify() { verify_impl "$@"; }

# ---------- selftest（负控）----------
do_selftest() {
    local out; out="$(new_tmpdir)"
    say "① 造一个正常包…"
    do_build "$out" >/dev/null
    local tb; tb="$(ls "$out"/*.tar.* | head -1)"
    say "② 正常包应**通过**校验"
    do_verify "$tb" >/dev/null && say "   ✓ PASS"

    say "③ 篡改包内一个文件，再打成同名 tar…"
    local stage; stage="$(new_tmpdir)"
    extract_tarball "$tb" "$stage"
    local victim="$stage/overlay/vllm_ascend/attention/dsa_v41.py"
    [ -f "$victim" ] || die "找不到用于篡改的文件"
    printf '\n# tampered by selftest\n' >> "$victim"
    local bad="$out/tampered.tar"
    tar --sort=name --owner=0 --group=0 --numeric-owner -C "$stage" -cf "$bad" .

    say "④ 篡改包必须**被抓住**"
    if do_verify "$bad" >/dev/null 2>&1; then
        echo "[pkg][FAIL] 篡改包竟然通过了校验 —— 校验器无效" >&2
        exit 1
    fi
    say "   ✓ PASS（篡改被检出）"
    echo
    say "selftest 通过：正常包可验、篡改包被抓"
}

case "${1:-build}" in
    build)    shift || true; do_build "${1:-$PKG_ROOT/dist}" ;;
    verify)   shift || true; do_verify "${1:?usage: verify <tarball>}" || exit 1 ;;
    selftest) do_selftest ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) die "未知子命令：$1（build|verify|selftest）" ;;
esac
