#!/usr/bin/env bash
# =============================================================================
# package_release.sh —— 把本发布形态打成**可复现、可自证**的 tar.zst
#
# 为什么需要：这套东西是**二进制面**（OPP 树 + .so）。二进制一旦发出去，
# "对方跑的是哪一份"只能靠哈希自证，不能靠"我记得拷的是那个"。
# 本仓在"我以为装了新版、其实还是旧版"上踩过。
#
# 用法：
#   bash package_release.sh build  [outdir]     # 产出 tar.zst + .sha256
#   bash package_release.sh verify <tarball>    # 解包并逐文件校验（含负控）
#   bash package_release.sh selftest            # 篡改一个文件，证明 verify 抓得住
#
# 产物命名：dsv41-fusion-<commit12>-<payload_fp12>.tar.zst
#   --mtime 固定为提交时间、owner/group 归零、--sort=name
#   ⇒ **同一份输入必然得到同一个 sha256**
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
PAYLOAD="$HERE/payload"

die() { echo "[pkg][FAIL] $*" >&2; exit 1; }
say() { echo "[pkg] $*"; }

_TMPDIRS=()
new_tmpdir() { local d; d="$(mktemp -d)"; _TMPDIRS+=("$d"); printf '%s' "$d"; }
_cleanup() { local d; for d in ${_TMPDIRS[@]+"${_TMPDIRS[@]}"}; do [ -n "$d" ] && rm -rf "$d"; done; return 0; }
trap _cleanup EXIT

payload_fp() {
    # payload 的聚合指纹：逐文件 sha256 再 sha256（与 PAYLOAD.sha256 同源）
    ( cd "$PAYLOAD" && find . -type f ! -name 'PAYLOAD.sha256' -print0 \
        | LC_ALL=C sort -z | sed -z 's|^\./||' | xargs -0 sha256sum ) | sha256sum | cut -c1-12
}
git_info()   { git -C "$REPO_ROOT" rev-parse --short=12 HEAD 2>/dev/null || echo nogit; }
git_branch() { git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo nogit; }
git_mtime()  { git -C "$REPO_ROOT" log -1 --format=%ct 2>/dev/null || echo 0; }

write_metadata() {
    local stage="$1"
    {
        echo "# DeepSeek-V4.1-Flash / A3 TP8 —— 解码融合优化包（构建身份）"
        # 不写"当前时间"：那会让同一份输入两次构建得到不同 sha256。
        echo "# 构建基准时间: $(date -d "@$(git_mtime)" -Is 2>/dev/null || echo 0)"
        echo "# 仓库       : $REPO_ROOT"
        echo "# 分支       : $(git_branch)"
        echo "# 提交       : $(git_info)"
        echo "# payload 指纹: $(payload_fp)"
        echo "#"
        echo "# 用法（目标机）："
        echo "#   1) tar -I zstd -xf <本包> -C <dir>"
        echo "#   2) bash <dir>/install.sh <container>"
        echo "#   3) 带 SWITCHES.md 里的两个变量重启服务"
        echo "#"
        echo "# 内容身份：见同目录 PAYLOAD.sha256（逐文件 sha256）"
    } > "$stage/BUILD_INFO.txt"
    {
        echo "# 校验：cd <解包目录> && sha256sum -c PAYLOAD.sha256"
        ( cd "$stage" && find . -type f ! -name 'PAYLOAD.sha256' ! -name 'BUILD_INFO.txt' -print0 \
            | LC_ALL=C sort -z | sed -z 's|^\./||' | xargs -0 sha256sum )
    } > "$stage/PAYLOAD.sha256"
}

do_build() {
    local outdir="${1:-$HERE/dist}"
    [ -d "$PAYLOAD/opp" ] || die "payload 不完整（缺 opp/）。先跑 build_payload.sh"
    [ -s "$PAYLOAD/PAYLOAD.sha256" ] || die "payload 缺 PAYLOAD.sha256。先跑 build_payload.sh"
    mkdir -p "$outdir"

    # ★ 先验 payload 自身没被改过：否则会把"被篡改的 payload"打成正式包
    ( cd "$PAYLOAD" && sha256sum -c --quiet PAYLOAD.sha256 ) || die "payload 自身校验失败（文件被改过？重新跑 build_payload.sh）"

    local stage; stage="$(new_tmpdir)"
    # 用 tar 搬运以保留目录结构；-p 保权限位（OPP 里有可执行脚本）
    ( cd "$PAYLOAD" && tar -cf - . ) | ( cd "$stage" && tar -xpf - )
    # 文档与脚本也进包（消费者不解仓库也能看懂）
    for f in README.md SWITCHES.md PAYLOAD.md REPRODUCE.md; do
        [ -f "$HERE/$f" ] && cp -p "$HERE/$f" "$stage/$f"
    done
    [ -d "$HERE/launch" ] && cp -a "$HERE/launch" "$stage/launch"
    write_metadata "$stage"

    local name="dsv41-fusion-$(git_info)-$(payload_fp)"
    local tarball="$outdir/$name.tar.zst"
    local mtime; mtime="$(git_mtime)"

    say "打包（可复现参数：owner/group=0, mtime=@$mtime, sort=name）…"
    if command -v zstd >/dev/null 2>&1; then
        tar --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$mtime" \
            -C "$stage" -cf - . | zstd -q -f -19 -o "$tarball"
    else
        tarball="$outdir/$name.tar.gz"
        say "（没装 zstd，退回 gzip）"
        tar --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$mtime" \
            -C "$stage" -cf - . | gzip -9n > "$tarball"
    fi
    # 标准格式（<hash>  <name>），使用者才能一条 sha256sum -c 自证
    ( cd "$(dirname "$tarball")" && sha256sum "$(basename "$tarball")" ) > "$tarball.sha256"

    say "产物：$tarball  ($(du -h "$tarball" | cut -f1))"
    say "sha256：$(cat "$tarball.sha256")"
    echo
    say "自检（确保发出去的包自己能验过）："
    do_verify "$tarball"
}

extract_tarball() {
    local tb="$1" dest="$2"; mkdir -p "$dest"
    case "$tb" in
        *.tar.zst) command -v zstd >/dev/null 2>&1 || { echo "[pkg][FAIL] 缺 zstd" >&2; return 1; }
                   zstd -q -dc "$tb" | tar -C "$dest" -xf - ;;
        *.tar.gz)  tar -C "$dest" -xzf "$tb" ;;
        *.tar)     tar -C "$dest" -xf "$tb" ;;
        *) echo "[pkg][FAIL] 未知格式：$tb" >&2; return 1 ;;
    esac
}

# 只 return 不 exit —— 它同时被负控调用
verify_impl() {
    local tb="${1:?usage: verify <tarball>}"
    [ -f "$tb" ] || { echo "[pkg][FAIL] 找不到 $tb" >&2; return 1; }
    local dest; dest="$(new_tmpdir)"
    extract_tarball "$tb" "$dest" || return 1

    [ -f "$dest/PAYLOAD.sha256" ] || { echo "[pkg][FAIL] 包内没有 PAYLOAD.sha256" >&2; return 1; }
    [ -f "$dest/BUILD_INFO.txt" ] || { echo "[pkg][FAIL] 包内没有 BUILD_INFO.txt" >&2; return 1; }

    if [ -f "$tb.sha256" ]; then
        local expect got
        expect="$(awk 'NR==1{print $1}' "$tb.sha256")"
        got="$(sha256sum "$tb" | awk '{print $1}')"
        [ "$got" = "$expect" ] || { echo "[pkg][FAIL] 归档 sha256 不符：期望 $expect 实得 $got" >&2; return 1; }
        say "归档 sha256 相符：$(echo "$expect" | cut -c1-16)…"
    fi

    say "逐文件校验…"
    ( cd "$dest" && sha256sum -c --quiet PAYLOAD.sha256 ) \
        || { echo "[pkg][FAIL] PAYLOAD.sha256 校验失败（被篡改或缺失）" >&2; return 1; }

    if [ -f "$tb.sha256" ]; then
        ( cd "$(dirname "$tb")" && sha256sum -c --quiet "$(basename "$tb").sha256" ) \
            || { echo "[pkg][FAIL] sha256sum -c 自证失败" >&2; return 1; }
        say "使用者侧自证通过：sha256sum -c <包>.sha256"
    fi

    # 关键内容抽查：包"能验过"不等于"装得对"
    [ -d "$dest/opp/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910_93/rms_norm_dynamic_quant_bf16" ] \
        || { echo "[pkg][FAIL] 缺少自研算子 kernel 目录" >&2; return 1; }
    grep -q "LNORM-FUSE" "$dest/py/model.py" || { echo "[pkg][FAIL] py/model.py 不含 LNORM-FUSE 标记" >&2; return 1; }

    say "文件数：$(find "$dest" -type f | wc -l)"
    say "构建身份：$(grep -E '^# (提交|payload 指纹)' "$dest/BUILD_INFO.txt" | tr '\n' ' ')"
    say "✅ 校验通过：$tb"
    return 0
}
do_verify() { verify_impl "$@"; }

do_selftest() {
    local out; out="$(new_tmpdir)"
    say "① 造一个正常包…"
    do_build "$out" >/dev/null
    local tb; tb="$(ls "$out"/*.tar.* | head -1)"
    say "② 正常包应通过校验"
    do_verify "$tb" >/dev/null && say "   ✓ PASS"

    say "③ 篡改包内一个二进制文件，再打回同名 tar…"
    local stage; stage="$(new_tmpdir)"
    extract_tarball "$tb" "$stage"
    local victim
    victim="$(find "$stage/py" -name '*.py' | head -1)"
    [ -n "$victim" ] || die "找不到用于篡改的文件"
    printf '\n# tampered by selftest\n' >> "$victim"
    local bad="$out/tampered.tar"
    tar --sort=name --owner=0 --group=0 --numeric-owner -C "$stage" -cf "$bad" .

    say "④ 篡改包必须被抓住"
    if do_verify "$bad" >/dev/null 2>&1; then
        echo "[pkg][FAIL] 篡改包竟然通过校验 —— 校验器无效" >&2; exit 1
    fi
    say "   ✓ PASS（篡改被检出）"
    echo
    say "selftest 通过：正常包可验、篡改包被抓"
}

case "${1:-build}" in
    build)    shift || true; do_build "${1:-$HERE/dist}" ;;
    verify)   shift || true; do_verify "${1:?usage: verify <tarball>}" || exit 1 ;;
    selftest) do_selftest ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) die "未知子命令：$1（build|verify|selftest）" ;;
esac
