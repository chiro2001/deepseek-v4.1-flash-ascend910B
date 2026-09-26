#!/usr/bin/env bash
# selftest_collect_evidence.sh —— collect_evidence.sh 的**沙箱自测**（零容器、零 NPU、零网络）
#
# 为什么需要它：收集器本身是"出问题时唯一能救命的工具"，而它**没有自测**。
#   用户点名要"能指定 log 位置"，而"位置"有三个入口（SERVE_LOG / RUN_DIR / RUN_ID+自动取最新），
#   入口逻辑一旦写错，表现是**静默抽错文件**（或抽到别人的臂）—— 这类错在真出问题时最贵。
#
# 用法： bash a2/scripts/selftest_collect_evidence.sh
# 退出码：0 = 全过；9 = 有失败
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC_REPO=$(cd "$HERE/../.." && pwd)
SCRIPT=${SCRIPT_SRC:-$SRC_REPO/a2/scripts/collect_evidence.sh}
[ -f "$SCRIPT" ] || { echo "⛔ 找不到待测脚本：$SCRIPT" >&2; exit 9; }
# The selftest supplies its own location in every arm.  A caller's run
# selection must never redirect a fixture to a real service log.
unset SERVE_LOG RUN_DIR RUN_ID

V=0
ok()  { printf '  \033[32mPASS\033[0m  %s\n' "$*"; V=$((V+1)); }
bad() { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; }
say() { printf '\n==== %s ====\n' "$*"; }

T=$(mktemp -d); trap 'rm -rf "$T"' EXIT

# ---------------------------------------------------------------- 夹具
# 造一个"像真的" run 目录：serve_cmd.txt + inner.sh + serve.log（含被抽的关键串）
mk_run() {   # $1 = 目录名（相对 $T/results）
    local d="$T/results/$1"
    mkdir -p "$d"
    cat > "$d/serve_cmd.txt" <<EOF
[serve_a2] run_id=$1 image=dsv41-a2:v9 model=/m/x port=8077 devs='0 1 2 3 4 5 6 7' util=0.90 max_len=1048576
EOF
    cat > "$d/inner.sh" <<'EOF'
export VLLM_V41_KV8_SWA=1
export MAX_LEN=1048576
EOF
    cat > "$d/serve.log" <<'EOF'
INFO P1_pinned rank=0 ret=0
INFO P1_pinned rank=7 ret=0
INFO D2_offload 被排除的组=[1]
INFO GPU KV cache size: 427,643 tokens
INFO kv_offload_total_bytes_total{transfer_type="GPU_to_CPU"} 1.2e+11
INFO kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"} 3.4e+10
INFO SpecDecoding metrics: Mean acceptance length: 2.90
EOF
    printf '%s\n' "$d"
}

run_collect() {  # $* = 传给收集器的参数（env 由调用方给）
    env PORT=59999 URL=http://127.0.0.1:59999 NO_PROBE=1 \
        CTR="__selftest_no_such_ctr__" \
        OUTDIR="$T/out" \
        bash "$SCRIPT" "$@"
}

# ================================================================ ① --run-dir
say "① --run-dir <目录> ⇒ 抽那一个 run（不是最新的那个）"
D1=$(mk_run a2_20260101_000001)
D2=$(mk_run a2_20260102_000002)
rm -f "$T/out/EVIDENCE.txt"
run_collect --run-dir "$D1" >/dev/null 2>&1
rc=$?
if [ $rc -eq 0 ] && [ -f "$T/out/EVIDENCE.txt" ]; then
    ok "rc=0 且生成 EVIDENCE.txt"
else
    bad "rc=$rc，EVIDENCE.txt 缺失"
fi
grep -q "run 目录：$D1" "$T/out/EVIDENCE.txt" 2>/dev/null \
    && ok "run 目录 = 指定的那一个（不是 mtime 最新的 $D2）" \
    || bad "run 目录没落在 --run-dir 上"
grep -q "a2_20260101_000001" "$T/out/EVIDENCE.txt"       && ok "抽到了该 run 的 serve_cmd.txt" || bad "serve_cmd.txt 没抽到"
grep -q "VLLM_V41_KV8_SWA=1" "$T/out/EVIDENCE.txt"       && ok "抽到了该 run 的 inner.sh env"  || bad "inner.sh 没抽到"
grep -q "CPU_to_GPU.*3.4e+10" "$T/out/EVIDENCE.txt"      && ok "抽到了卸载计数（★ 判断存/取的关键数）" || bad "卸载计数没抽到"

# ================================================================ ② --serve-log（★ 用户点名）
say "② --serve-log <文件> ⇒ 只认这个文件（它不在 results 树里也能用）"
LOGDIR="$T/别人的目录/logs"
mkdir -p "$LOGDIR"
cp "$D2/serve.log" "$LOGDIR/random_name.log"
echo 'INFO D2_offload 被排除的组=[1]' >> "$LOGDIR/random_name.log"
rm -f "$T/out/EVIDENCE.txt"
run_collect --serve-log "$LOGDIR/random_name.log" >/dev/null 2>&1
rc=$?
[ $rc -eq 0 ] && [ -f "$T/out/EVIDENCE.txt" ] && ok "rc=0 且生成 EVIDENCE.txt" || bad "rc=$rc（给了存在的日志却失败）"
grep -q "serve.log：$LOGDIR/random_name.log" "$T/out/EVIDENCE.txt" 2>/dev/null \
    && ok "EVIDENCE 里钉住了日志的**绝对路径**（判据绑唯一对象）" || bad "没钉住日志路径"
grep -q "run 目录：$LOGDIR" "$T/out/EVIDENCE.txt" 2>/dev/null \
    && ok "run 目录自动取日志的父目录" || bad "run 目录没取父目录"
[ "$(grep -c 'P1_pinned ret=0' "$T/out/EVIDENCE.txt" 2>/dev/null)" = "1" ] \
    && ok "起服门计数从**指定日志**里数出来（P1_pinned ret=0 = 1）" || bad "起服门计数不对"
echo "（同目录缺 serve_cmd.txt / inner.sh ⇒ 只跳过那两段，不报错）"

# ================================================================ ③ env 形式
say "③ SERVE_LOG=<文件> 环境变量形式（与 --serve-log 等价）"
rm -f "$T/out/EVIDENCE.txt"
SERVE_LOG="$LOGDIR/random_name.log" run_collect >/dev/null 2>&1
rc=$?
[ $rc -eq 0 ] && grep -q "serve.log：$LOGDIR/random_name.log" "$T/out/EVIDENCE.txt" \
    && ok "SERVE_LOG= 也生效" || bad "rc=$rc，SERVE_LOG= 形式没生效"

# ================================================================ ④ 自动取最新（回归）
say "④ 不给任何位置 ⇒ 取 RESULTS 根里**最新**的 run（老行为不许被破坏）"
rm -rf "$T/results"; D5=$(mk_run a2_20260105_000005); D9=$(mk_run a2_20260109_000009)
touch -d '2026-01-05 00:00:05' "$D5/serve.log" 2>/dev/null || true
touch -d '2026-01-09 00:00:09' "$D9/serve.log" 2>/dev/null || true
# The collector orders run directories, not their serve.log files.  Set both
# directory mtimes explicitly: filesystems with coarse timestamp resolution
# can otherwise tie two directories created in the same second.
touch -d '2026-01-05 00:00:05' "$D5" 2>/dev/null || true
touch -d '2026-01-09 00:00:09' "$D9" 2>/dev/null || true
rm -f "$T/out/EVIDENCE.txt"
env SERVE_LOG= RUN_DIR= RUN_ID= RESULTS="$T/results" PORT=59999 URL=http://127.0.0.1:59999 NO_PROBE=1 \
    CTR="__selftest_no_such_ctr__" OUTDIR="$T/out" bash "$SCRIPT" >/dev/null 2>&1
rc=$?
{ [ $rc -eq 0 ] && grep -q "run 目录：$D9" "$T/out/EVIDENCE.txt"; } \
    && ok "自动取到了 mtime 最新的 $D9（并且 RESULTS= 可换根）" || bad "rc=$rc：自动取最新坏了"

# ================================================================ ⑤ 负例：位置不存在
say "⑤ 位置不存在 ⇒ rc=64，且**不许**生成半份 EVIDENCE"
rm -f "$T/out/EVIDENCE.txt"
run_collect --serve-log "$T/根本没有这个.log" >/dev/null 2>&1
rc1=$?
run_collect --run-dir  "$T/根本没有这个目录" >/dev/null 2>&1
rc2=$?
env RESULTS="$T/空results" PORT=59999 NO_PROBE=1 CTR=x OUTDIR="$T/out" bash "$SCRIPT" >/dev/null 2>&1
rc3=$?
[ "$rc1" = "64" ] && ok "SERVE_LOG 不存在 ⇒ rc=64"      || bad "rc1=$rc1（期望 64）"
[ "$rc2" = "64" ] && ok "RUN_DIR  不存在 ⇒ rc=64"      || bad "rc2=$rc2（期望 64）"
[ "$rc3" = "64" ] && ok "results 里没有 run ⇒ rc=64"   || bad "rc3=$rc3（期望 64）"
[ -f "$T/out/EVIDENCE.txt" ] && bad "失败时仍生成了 EVIDENCE.txt（半份证据会误导）" || ok "失败时没有半份产物"

# ================================================================ ⑥ 计数助手不许打两个 0
say "⑥ 「无匹配」的计数必须只打一个 0（历史踩过：\$(grep -c ... || echo 0) 输出两行）"
rm -f "$T/out/EVIDENCE.txt"
run_collect --serve-log "$LOGDIR/random_name.log" >/dev/null 2>&1
n=$(grep -c -E '^  KeyError  +=' "$T/out/EVIDENCE.txt" 2>/dev/null)
blank=$(sed -n '/KeyError/,+1p' "$T/out/EVIDENCE.txt" | grep -c '^  0$')
[ "$n" = "1" ] && [ "$blank" = "0" ] \
    && ok "KeyError 计数行恰好 1 行、没有多余的 '0' 行" || bad "计数行 n=$n blank=$blank"

# ================================================================ ⑦ 参数校验
say "⑦ --run-id / --outdir / --help 可用"
rm -f "$T/out2/EVIDENCE.txt"
env RESULTS="$T/results" PORT=59999 NO_PROBE=1 CTR=x OUTDIR="$T/out2" \
    bash "$SCRIPT" --run-id a2_20260105_000005 >/dev/null 2>&1
[ -f "$T/out2/EVIDENCE.txt" ] && grep -q "a2_20260105_000005" "$T/out2/EVIDENCE.txt" \
    && ok "--run-id + OUTDIR 可用" || bad "--run-id/OUTDIR 组合失败"
bash "$SCRIPT" --help >/dev/null 2>&1 && ok "--help rc=0" || bad "--help 非 0"

# ================================================================ ⑧ 乱码证据（★ 本轮真正的靶子）
# 用一个**假的 OpenAI 兼容服务**回答"带乱码指纹的原文"，验证：
#   ① 探针的 --out JSON 真的落到了 OUTDIR；
#   ② EVIDENCE 里出现"模型答案原文"（乱码要看原文，不是看通过/失败）；
#   ③ "乱码指纹"能数出 U+FFFD 等替换/控制字符。
say "⑧ 服务活着时：模型答案原文 + 乱码指纹进 EVIDENCE（假服务，零真机）"
cat > "$T/fake_srv.py" <<'PY'
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ANSWER = "\u597d\u7684\uff0c\u7b54\u6848\u662f 391\u3002\ufffd \u5c3e\u5df4"

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  pass
    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self._json({"status": "ok"})
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); self.rfile.read(n)
        self._json({"choices": [{"message": {"role": "assistant", "content": ANSWER},
                                 "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 8}})

srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
open(sys.argv[1], "w").write(str(srv.server_address[1]))
srv.serve_forever()
PY
python3 "$T/fake_srv.py" "$T/port.txt" > "$T/srv.log" 2>&1 &
SRVPID=$!
for _ in $(seq 1 60); do [ -s "$T/port.txt" ] && break; sleep 0.1; done
FAKEPORT=$(cat "$T/port.txt" 2>/dev/null || echo "")
if [ -z "$FAKEPORT" ]; then
    bad "假服务没起来（跳过本条，其余不受影响）"
else
    rm -f "$T/out/EVIDENCE.txt" "$T/out/textprobe.json"
    env PORT="$FAKEPORT" URL="http://127.0.0.1:$FAKEPORT" NO_PROBE=0 PROBES=1 \
        MODEL_NAME=stub-model CTR="__selftest_no_such_ctr__" OUTDIR="$T/out" \
        timeout 300 bash "$SCRIPT" --serve-log "$LOGDIR/random_name.log" >/dev/null 2>&1
    [ -f "$T/out/textprobe.json" ] && ok "探针证据落到 OUTDIR/textprobe.json" \
        || bad "textprobe.json 没落盘"
    grep -q "模型答案原文" "$T/out/EVIDENCE.txt" 2>/dev/null \
        && ok "EVIDENCE 里有「模型答案原文」段" || bad "缺「模型答案原文」段"
    grep -q "乱码指纹" "$T/out/EVIDENCE.txt" 2>/dev/null \
        && ok "EVIDENCE 里有「乱码指纹」段" || bad "缺「乱码指纹」段"
    # U+FFFD 那一格的计数必须 >=1（假答案里放了一个替换字符）
    fffd=$(sed -n '/乱码指纹/,/同上全文/p' "$T/out/EVIDENCE.txt" \
           | grep -a 'U+FFFD(原始字节)' | grep -oE '= [0-9]+' | tr -dc '0-9')
    [ -n "$fffd" ] && [ "$fffd" -ge 1 ] 2>/dev/null \
        && ok "U+FFFD 指纹计数 = $fffd（≥1）" || bad "U+FFFD 指纹计数不对（'$fffd'）"
    grep -aq "答案是 391" "$T/out/textprobe.json" 2>/dev/null \
        && ok "JSON 里保留了答案原文（可离线复算）" || bad "JSON 里没有答案原文"
fi
kill "$SRVPID" 2>/dev/null || true

echo
# ================================================================ ⑨ 平台默认值（A3）
# 用户要求"A3 也要能一起拉起来" ⇒ 收集器必须认 A3 的**容器名/端口/模型名**三个平台差异。
# 判据要绑**内容**：容器名错 ⇒ §② 容器内指纹会**整段静默跳过**（老版本就是只打一句"容器不在"）；
#   模型名错 ⇒ 探针被服务端 400 ⇒ 表现是"探针全失败"，很容易被误读成"模型坏了"。
say "⑨ PLAT=a3 ⇒ 容器名/端口/模型名整组切换；且平台名非法必须拒绝"
rm -f "$T/out/EVIDENCE.txt"
env RESULTS="$T/results" PLAT=a3 NO_PROBE=1 CTR="" PORT="" MODEL_NAME="" OUTDIR="$T/out" \
    bash "$SCRIPT" --run-id a2_20260105_000005 >/dev/null 2>&1
grep -q "PLAT=a3  容器=dsv41-a3  端口=8020  模型名=deepseek-v41" "$T/out/EVIDENCE.txt" 2>/dev/null \
    && ok "A3 三个平台默认值整组切换（容器/端口/模型名）" || bad "A3 平台默认值没生效"
# ★ 显式覆盖必须赢（A3 上有人用别的端口/容器名时）
rm -f "$T/out/EVIDENCE.txt"
env RESULTS="$T/results" PLAT=a3 PORT=8500 CTR=my-ctr MODEL_NAME=my-model NO_PROBE=1 OUTDIR="$T/out" \
    bash "$SCRIPT" --run-id a2_20260105_000005 >/dev/null 2>&1
grep -q "PLAT=a3  容器=my-ctr  端口=8500  模型名=my-model" "$T/out/EVIDENCE.txt" 2>/dev/null \
    && ok "显式 PORT/CTR/MODEL_NAME 覆盖压过平台默认" || bad "显式覆盖被平台默认吃掉了"
env RESULTS="$T/results" PLAT=bogus NO_PROBE=1 OUTDIR="$T/out" \
    bash "$SCRIPT" --run-id a2_20260105_000005 >/dev/null 2>&1
[ $? = 64 ] && ok "PLAT 非法 ⇒ rc=64（不猜平台）" || bad "PLAT 非法没有被拒绝"
# A2 默认不许被这次改动带偏（回归锚）
rm -f "$T/out/EVIDENCE.txt"
env RESULTS="$T/results" NO_PROBE=1 CTR="" PORT="" MODEL_NAME="" OUTDIR="$T/out" \
    bash "$SCRIPT" --run-id a2_20260105_000005 >/dev/null 2>&1
grep -q "PLAT=a2  容器=dsv41-a2  端口=8077  模型名=deepseek-v4-flash" "$T/out/EVIDENCE.txt" 2>/dev/null \
    && ok "A2 默认（不传 PLAT）仍是 dsv41-a2 / 8077 / deepseek-v4-flash" || bad "A2 默认被带偏了"

echo
echo "=============== 通过 $V 条 ==============="
[ "$V" -ge 27 ] && { echo "✅ 自测全过"; exit 0; } || { echo "❌ 不合格（通过数 $V < 27）"; exit 9; }
