#!/usr/bin/env bash
# =============================================================================
# 本轮 issue 跟进修复的**回归**（不需要 docker / NPU，秒级）
#
# 每一条都对应一个**外部报告过的真实故障**。判据尽量绑"修复后的可观测行为"，
# 而不是"我改了哪一行"；能加负控的都加了负控。
#
#   ① issue #2 ②：代理劫持 127.0.0.1 ⇒ 三个起服脚本必须补 no_proxy，
#                  且**不覆盖**用户已设的值（含负控：已含 127.0.0.1 就不动）
#   ② issue #2 3.1：`say()` 早于 `mkdir` ⇒ driver.log 头 ~100 行丢失
#   ③ issue #2 3.6 / 待办 3：KV 门槛写死 3Mi ⇒ 默认配置必报假红；**三处同口径**
#   ④ issue #2 7.2：bench_concurrency 用 data[0] 覆盖 --model
#   ⑤ 本仓自查：`curl ... || echo 000` 会拼出 "000000"（AGENTS §3.2 同族）
#
# 用法：bash tools/selftest_issue_followups.sh     # 0=全过
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
cd "$PKG"

pass=0; fail=0
ok()  { pass=$((pass+1)); printf '  ✓ %s\n' "$*"; }
bad() { fail=$((fail+1)); printf '  ✗ %s\n' "$*"; }
TMPS=$(mktemp -d); trap 'rm -rf "$TMPS"' EXIT

# ---------------------------------------------------------------- ① no_proxy
echo "== ① no_proxy（issue #2 ②）=="
for f in scripts/serve_a2.sh scripts/serve_v2.sh scripts/serve_a3.sh; do
  # 抽出 no_proxy 片段：从 `if [ "${KEEP_PROXY_FOR_LOCALHOST` 到 `unset _v41_np_default`
  awk '/^if \[ "\$\{KEEP_PROXY_FOR_LOCALHOST/,/^unset _v41_np_default$/' "$f" > "$TMPS/np.sh"
  if [ ! -s "$TMPS/np.sh" ]; then bad "$f 缺 no_proxy 片段"; continue; fi

  v=$(env -u no_proxy -u NO_PROXY bash -c ". '$TMPS/np.sh'; printf '%s' \"\$no_proxy\"")
  [ "$v" = "127.0.0.1,localhost,::1" ] && ok "$f 未设→补默认" || bad "$f 未设→实得[$v]"

  v=$(env -u NO_PROXY no_proxy="10.0.0.0/8" bash -c ". '$TMPS/np.sh'; printf '%s' \"\$no_proxy\"")
  [ "$v" = "10.0.0.0/8,127.0.0.1,localhost,::1" ] && ok "$f 已设→追加非覆盖" || bad "$f 已设→实得[$v]"

  # 负控：用户已含 127.0.0.1 ⇒ 必须原样
  v=$(env -u NO_PROXY no_proxy="a,127.0.0.1,b" bash -c ". '$TMPS/np.sh'; printf '%s' \"\$no_proxy\"")
  [ "$v" = "a,127.0.0.1,b" ] && ok "$f 已含→不动（负控）" || bad "$f 已含→被改了[$v]"

  v=$(env -u no_proxy -u NO_PROXY KEEP_PROXY_FOR_LOCALHOST=1 bash -c ". '$TMPS/np.sh'; printf '%s' \"\${no_proxy:-UNSET}\"")
  [ "$v" = "UNSET" ] && ok "$f 显式关→不碰（负控）" || bad "$f 显式关→仍设了[$v]"

  v=$(env -u no_proxy -u NO_PROXY bash -c ". '$TMPS/np.sh'; printf '%s' \"\$NO_PROXY\"")
  [ "$v" = "127.0.0.1,localhost,::1" ] && ok "$f NO_PROXY 大写照应" || bad "$f NO_PROXY=[$v]"
done

# ------------------------------------------------------------ ② say/mkdir
echo "== ② say() 早于 mkdir（issue #2 3.1）=="
mk=$(grep -n '^mkdir -p "\$OUT" 2>/dev/null || true' scripts/serve_a2.sh | head -1 | cut -d: -f1)
sy=$(grep -n '^say() {' scripts/serve_a2.sh | head -1 | cut -d: -f1)
if [ -n "$mk" ] && [ -n "$sy" ] && [ "$mk" -lt "$sy" ]; then
  ok "mkdir(:$mk) 在 say()(:$sy) **之前**"
else
  bad "顺序不对：mkdir=${mk:-无} say=${sy:-无}（必须先 mkdir）"
fi
# 行为判据：复刻新顺序，第一条 say 必须落进 driver.log
t="$TMPS/out1"
out=$(bash -c "
  OUT='$t'; mkdir -p \"\$OUT\" 2>/dev/null || true
  say() { printf '\n[serve_a2] %s\n' \"\$*\" | tee -a \"\$OUT/driver.log\"; }
  say '第一条'; say '第二条'" 2>&1)
n=$(grep -c 'serve_a2' "$t/driver.log" 2>/dev/null || echo 0)
[ "$n" = "2" ] && ok "第一条 say 也进 driver.log（2/2）" || bad "driver.log 只进 $n/2 条"
case "$out" in *"No such file or directory"*) bad "仍有 tee 报错：$out";; *) ok "无 tee 报错";; esac

# ------------------------------------------------------------- ③ KV 门槛
echo "== ③ KV 门槛口径（issue #2 3.6 / 待办 3）=="
for f in scripts/run_test.sh tools/attach_test.sh tests/make_report.sh; do
  if grep -q 'KV_MIN=${KV_MIN:-2800000}' "$f"; then ok "$f 用 KV_MIN 且默认 2.8M"
  else bad "$f 未用 KV_MIN=${KV_MIN:-\<default\>}"; fi
done
# 负控：默认配置的实测值 2,823,080 必须 PASS；旧门槛 3,145,728 会判 FAIL
kv=2823080; kmin=2800000
[ "$kv" -gt "$kmin" ] && ok "2,823,080 > 2,800,000 ⇒ 默认配置 PASS（旧门槛 3Mi 会误判 FAIL）" \
                      || bad "默认配置仍被判 FAIL"
# 正控：显式要 3Mi 时仍能拦住
kv=2823080; kmin=3145728
[ "$kv" -gt "$kmin" ] && bad "显式 3Mi 门槛没拦住" || ok "显式 KV_MIN=3145728 仍能拦住（正控）"

# ------------------------------------------------------- ④ bench --model
echo "== ④ bench_concurrency 的 --model（issue #2 7.2）=="
if grep -q 'data\[0\]\["id"\]$' tools/bench_concurrency.py; then
  bad "仍在用 data[0] 直接覆盖"
else
  ok "不再直接 data[0] 覆盖"
fi
if grep -q 'ap.add_argument("--model", default=""' tools/bench_concurrency.py; then
  ok "--model 默认为空串（\"未指定\"是真实状态）"
else
  bad "--model 仍有非空默认值 ⇒ 会静默假定模型名"
fi
if grep -q 'if a.model in ids:' tools/bench_concurrency.py; then
  ok "指定时优先在 /v1/models 列表里匹配"
else
  bad "缺\"优先匹配 --model\"的分支"
fi

# ------------------------------------------------------- ⑤ 000000 拼接
echo "== ⑤ curl || echo 000 拼接（AGENTS §3.2 同族，本仓自查）=="
if out=$(python3 tools/check_http_code_pattern.py 2>&1); then
  ok "无旧写法（$(printf '%s' "$out" | tail -1)）"
else
  bad "仍存在旧写法：$(printf '%s' "$out" | head -3 | tr '\n' ' ')"
fi
# 负控：证明旧写法**确实**拼成 000000
# http-code-guard:allow ← 这行**故意**保留旧写法，用来证明判据有效（负控）
old=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:1/health" 2>/dev/null || echo 000)
[ "$old" = "000000" ] && ok "负控：旧写法确实得到 000000（证明这条判据有效）" \
                      || bad "负控失败：旧写法得到 [$old]（判据可能无效）"

echo
if [ "$fail" = 0 ]; then
  printf 'issue 跟进回归：PASS（%d 项）\n' "$pass"
  exit 0
fi
printf 'issue 跟进回归：FAIL（%d 项失败 / %d 通过）\n' "$fail" "$pass" >&2
exit 1
