#!/usr/bin/env bash
# =============================================================================
# fetch_corpus.sh —— 下载测试语料（**本仓库不附带受版权保护的原文**）
#
#   bash tools/fetch_corpus.sh            # 取默认语料（《地火》）
#   bash tools/fetch_corpus.sh --check    # 只校验本地文件是否已是正确版本
#
# 为什么不在仓库里放原文
# ---------------------------------------------------------------------------
# 《地火》是刘慈欣的文学作品，**著作权不属于本项目的 Apache-2.0 许可范围**。
# 因此仓库只提供来源链接 + 校验和，原文由使用者自行下载。
# 上游仓库 <https://github.com/VeejaLiu/ScienceFictionCollection> 的 MIT 许可
# 只覆盖其整理工作，不覆盖作品本身；请自行判断你所在地区的合规使用方式。
#
# 脚本做三件事
# ---------------------------------------------------------------------------
#   1. 下载原始 txt
#   2. 校验 sha256（防止上游改动导致数字不可复现）
#   3. 做与基准一致的清洗（去掉末尾的 "(完)" 标记），写入 data/dihuo.txt
#      并再校验一次结果哈希
# 只有两步哈希都对，才算成功 —— 这样 bench 数字可以逐字节复现。
#
# 环境变量
# ---------------------------------------------------------------------------
#   INSECURE_TLS=1   给 curl 加 -k（跳过 TLS 证书校验）。**仅限内网自签证书 / TLS 中间盒**，
#                    默认关；public 网络**不要**打开（关掉校验等于放弃传输层身份验证）。
#                    完整性不受影响：下面仍然逐字节比对 sha256，-k 只影响传输层。
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
cd "$PKG"

URL='https://raw.githubusercontent.com/VeejaLiu/ScienceFictionCollection/master/001%20-%20%E5%88%98%E6%85%88%E6%AC%A3(Cixin%20Liu)/%5B2000-1%5D%E3%80%8A%E5%9C%B0%E7%81%AB%E3%80%8B.txt'
SRC_SHA256='9e810bc1686ce17162340791d8d89932a2f0ab1e8a654ef3c2f3ad646fca5ef5'
OUT_SHA256='00aefd13430d4be45424db3c3f832b6c012d9c60c199518f45feed5d0a57e52b'
OUT='data/dihuo.txt'

say() { printf '\033[1m[corpus]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[corpus][FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

_check() {
  [ -f "$OUT" ] || return 1
  [ "$(sha256sum "$OUT" | cut -d' ' -f1)" = "$OUT_SHA256" ]
}

if [ "${1:-}" = "--check" ]; then
  if _check; then
    say "OK  $OUT 已是正确版本（sha256 ${OUT_SHA256:0:16}…）"
    exit 0
  fi
  say "MISSING/MISMATCH  $OUT"
  say "  请执行：bash tools/fetch_corpus.sh"
  exit 1
fi

if _check; then
  say "已存在且校验通过，跳过下载：$OUT"
  exit 0
fi

command -v curl >/dev/null || die "需要 curl"
TMP=$(mktemp -t dihuo.XXXXXX) || die "mktemp 失败"
trap 'rm -f "$TMP"' EXIT

say "下载原文 …"
say "  来源（自备链接）：见下方 URL"
# [TLS] 默认空 = 正常校验证书；只有显式 INSECURE_TLS=1 才降级（内网自签证书场景）
CURL_TLS=""
[ "${INSECURE_TLS:-0}" = "1" ] && CURL_TLS="-k"
[ -n "$CURL_TLS" ] && say "  ⚠️ INSECURE_TLS=1：已关闭 TLS 证书校验（仅建议内网使用）"
# shellcheck disable=SC2086  # $CURL_TLS 只在 INSECURE_TLS=1 时展开为 -k
curl $CURL_TLS -fsSL --retry 3 -m 120 -o "$TMP" "$URL" || die "下载失败（网络？）"

_got=$(sha256sum "$TMP" | cut -d' ' -f1)
if [ "$_got" != "$SRC_SHA256" ]; then
  {
    echo "原文 sha256 不匹配 —— 上游文件可能已改动，或下载被拦截。"
    echo "  期望 $SRC_SHA256"
    echo "  实际 $_got"
    echo "  文件 $(stat -c %s "$TMP") bytes"
    echo "为避免产生不可复现的性能数字，这里**故意不继续**。"
    echo "你可以：① 改用自己确认过的文本后再手动放好；② 调整脚本里的哈希。"
  } >&2
  exit 2
fi
say "原文校验通过"

python3 - "$TMP" "$OUT" <<'PYEOF' || die "清洗失败"
import sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8", errors="replace").read()
# 去掉末尾的完稿标记（与基准口径一致）
for tail in ("(完)", "（完）", "(全文完)", "（全文完）"):
    text = text.rstrip()
    if text.endswith(tail):
        text = text[: -len(tail)]
text = text.strip() + "\n"
open(dst, "w", encoding="utf-8").write(text)
print(f"  清洗后 {len(text)} 字符 -> {dst}")
PYEOF

_out=$(sha256sum "$OUT" | cut -d' ' -f1)
[ "$_out" = "$OUT_SHA256" ] || die "清洗结果哈希不匹配（期望 $OUT_SHA256，实际 $_out）"

say "✅ 就绪：$OUT（sha256 ${OUT_SHA256:0:16}…）"
say "   现在可以跑：python3 tools/bench_concurrency.py --corpus-file data/dihuo.txt --suffix-dir data/dihuo_local …"
