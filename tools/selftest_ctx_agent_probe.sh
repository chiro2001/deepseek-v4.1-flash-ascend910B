#!/usr/bin/env bash
# =============================================================================
# selftest_ctx_agent_probe.sh —— ctx_agent_probe.py 的沙箱自测（零真机、零 NPU）
#
# 为什么需要：这个探针**本身就是判据**。判据错了比"没有判据"更危险 ——
#   它会把好模型判成坏的（误杀），或把坏模型判成好的（漏判）。
#   所以先用**假服务**（tools/_fake_vllm.py）把两条路都走一遍：
#     ① 干净回答 ⇒ 四模式全 PASS、rc=0、JSON 落盘、复用逐字相同
#     ② 带 U+FFFD/NUL 且复读的回答 ⇒ 必须 FAIL、rc=1、指纹计数 >0
#     ③ 工具参数少一字符 + 带乱码 ⇒ toolargs 必须 FAIL（逐字判，不许"包含就算过"）
#     ④ 服务没有 /tokenize ⇒ 回退按字符近似，仍要跑完
#     ⑤ 服务不可达 ⇒ rc=2（不许静默"全过"）
#
# 用法： bash tools/selftest_ctx_agent_probe.sh
# 退出码：0 = 全过；9 = 有失败
# =============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PKG=$(cd "$HERE/.." && pwd)
PROBE=${PROBE_SRC:-$PKG/tools/ctx_agent_probe.py}
FAKE=$PKG/tools/_fake_vllm.py
CHK=$PKG/tools/_check_probe_json.py
for f in "$PROBE" "$FAKE" "$CHK"; do
    [ -f "$f" ] || { echo "⛔ 缺 $f" >&2; exit 9; }
done

V=0
F=0
ok()  { printf '  \033[32mPASS\033[0m  %s\n' "$*"; V=$((V+1)); }
bad() { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; F=$((F+1)); }
say() { printf '\n==== %s ====\n' "$*"; }

T=$(mktemp -d)
SRV=""
cleanup() {
    if [ -n "$SRV" ]; then kill "$SRV" 2>/dev/null; fi
    rm -rf "$T"
}
trap cleanup EXIT

start_srv() {
    if [ -n "$SRV" ]; then kill "$SRV" 2>/dev/null; sleep 0.3; fi
    rm -f "$T/port.txt"
    MODE="$1" NOTOKENIZE="${2:-0}" python3 "$FAKE" "$T/port.txt" >"$T/srv.log" 2>&1 &
    SRV=$!
    for _ in $(seq 1 60); do
        [ -s "$T/port.txt" ] && break
        sleep 0.1
    done
    PORT=$(cat "$T/port.txt" 2>/dev/null || echo "")
    if [ -z "$PORT" ]; then
        echo "⛔ 假服务没起来" >&2
        cat "$T/srv.log" >&2
        exit 9
    fi
}

run_probe() {
    local outp="$1"
    shift
    ( cd "$PKG" && timeout 300 python3 "$PROBE" --base-url "http://127.0.0.1:$PORT" \
        --model stub --out "$outp" --context-tokens 2048 --turns 2 --repeats 2 "$@" ) \
        >"$T/probe.log" 2>&1
}

jsonchk() {
    local name="$1" jf="$2" kind="$3"
    if python3 "$CHK" "$jf" "$kind" >/dev/null 2>&1; then ok "$name"
    else bad "$name"; fi
}

say "① 干净回答 ⇒ 四模式全 PASS、rc=0、JSON 落盘"
start_srv clean 0
run_probe "$T/clean.json" --mode all
rc=$?
if [ "$rc" = "0" ]; then ok "①a 退出码 0"
else bad "①a 退出码 $rc（期望 0）"; tail -15 "$T/probe.log" | sed 's/^/        /'; fi
[ -f "$T/clean.json" ] && ok "①b JSON 证据落盘" || bad "①b 没落盘"
jsonchk "①c needle 逐字命中" "$T/clean.json" needle_exact
jsonchk "①d 无乱码指纹"       "$T/clean.json" no_garbling
jsonchk "①e 四模式齐 + 复用逐字相同 + total_fails=0" "$T/clean.json" all_modes_clean
jsonchk "①f toolargs 逐字保真" "$T/clean.json" toolargs_ok

say "② 带乱码 + 复读的回答 ⇒ 必须 FAIL，且指纹计数 >0"
start_srv garbled 0
run_probe "$T/bad.json" --mode needle --repeats 1
rc=$?
[ "$rc" = "1" ] && ok "②a 退出码 1（识别为失败）" || bad "②a 退出码 $rc（期望 1）"
jsonchk "②b U+FFFD 与 NUL 都被数出来" "$T/bad.json" fd_detected
jsonchk "②c 复读被检出"               "$T/bad.json" repeat_detected
jsonchk "②d 逐字判为 False"            "$T/bad.json" not_exact

say "③ 工具参数少一字符 + 乱码 ⇒ toolargs 必须 FAIL"
run_probe "$T/bad2.json" --mode toolargs --repeats 1
rc=$?
[ "$rc" = "1" ] && ok "③a toolargs 判 FAIL" || bad "③a 退出码 $rc（期望 1）"
jsonchk "③b content_exact=False 被记录" "$T/bad2.json" toolargs_fail

say "④ 服务没有 /tokenize ⇒ 回退按字符近似，仍要跑完"
start_srv clean 1
run_probe "$T/notok.json" --mode needle --repeats 1
rc=$?
[ "$rc" = "0" ] && ok "④a 无 /tokenize 也能跑完并 PASS" || bad "④a 退出码 $rc（期望 0）"
grep -q "实际上下文" "$T/probe.log" && ok "④b 打印了实际上下文长度（可复算）" || bad "④b 没打印实际长度"

say "⑤ 服务不可达 ⇒ rc=2"
_p=$PORT
PORT=1
run_probe "$T/dead.json" --mode needle
rc=$?
PORT=$_p
[ "$rc" = "2" ] && ok "⑤ 连不上 ⇒ rc=2" || bad "⑤ 退出码 $rc（期望 2）"

echo
echo "=============== 通过 $V 条 / 失败 $F 条 ==============="
if [ "$F" = "0" ] && [ "$V" -ge 11 ]; then
    echo "✅ 自测全过"
    exit 0
fi
echo "❌ 不合格"
exit 9
