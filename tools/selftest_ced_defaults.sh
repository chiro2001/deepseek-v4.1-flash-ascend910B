#!/usr/bin/env bash
# [CED-DEFAULTS] A3 PD 分离（CED）交付默认值的**离线**自测。
#
# 为什么要有它：2026-09-27 把 DSpark 与前缀缓存转成默认开。这类"改默认值"的
# 改法最容易出的两个错都不是语法错，bash -n 查不出来：
#   ① 默认值改了，但某个角色/某条门把它又覆盖回去（例如 decode 分支里仍写着
#      `SPEC=${SPEC:-0}`）⇒ 用户拿到的是旧口径；
#   ② 门被顺手拆掉 ⇒ 本该拒绝的非法组合（P 带 SPEC、SPEC=2…）静默放行。
# 所以这里**既查默认解析、也查门仍然咬人**（正例 + 负控）。
#
# 做法：把 `serve_a3_ced_pd.sh` 拷到临时目录，把最后那行 `exec … serve_a3_pd.sh`
# 换成打印解析结果 —— 不碰 docker、不碰 NPU、秒级。
set -uo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$PKG/scripts/serve_a3_ced_pd.sh"
TMPD=$(mktemp -d "${TMPDIR:-/tmp}/ceddefaults.XXXXXX")
trap 'rm -rf "$TMPD"' EXIT
pass=0; fail=0
ok()  { printf '  \033[32mPASS\033[0m %s\n' "$*"; pass=$((pass+1)); }
bad() { printf '  \033[31mFAIL\033[0m %s\n' "$*"; fail=$((fail+1)); }

# 造一个"只解析不 exec"的副本：最后一行是 exec，替换成打印。
sed 's|^exec bash "$HERE/serve_a3_pd.sh" "$role"$|echo "RESOLVED spec=${SPEC:-<unset>} draft=${DRAFT_GRAPH:-<unset>} prefix=${PREFIX:-<unset>} static=${STATIC_KERNEL:-<unset>} allow_dspark=${V41_CED_ALLOW_DSPARK:-<unset>} allow_prefix=${V41_CED_ALLOW_PREFIX:-<unset>}"|' \
  "$SRC" > "$TMPD/ced.sh"
if grep -q "^exec bash" "$TMPD/ced.sh"; then
  bad "替换 exec 失败（脚本尾行变了？本自测需要同步更新）"; exit 1
fi

# run <期望字串> <说明> <role> [env...]
run() {
  local expect="$1" label="$2" role="$3"; shift 3
  local out rc
  out=$(env -u SPEC -u DRAFT_GRAPH -u PREFIX -u STATIC_KERNEL \
        -u V41_CED_ALLOW_DSPARK -u V41_CED_ALLOW_PREFIX -u V41_CED_ROLE \
        "$@" MODEL=/nonexistent bash "$TMPD/ced.sh" "$role" 2>&1)
  rc=$?
  if [ -n "$expect" ] && [ "$rc" = "0" ]; then
    # 逐个 key=value 都要出现（不能前缀匹配：期望里可能只挑几个字段）
    line=$(printf '%s' "$out" | grep "^RESOLVED " | head -1)
    miss=""
    for tok in $expect; do
      case " $line " in *" $tok "*) ;; *) miss="$miss $tok" ;; esac
    done
    if [ -z "$miss" ]; then ok "$label → $(printf '%s' "$expect" | tr '\n' ' ')"
    else bad "$label：缺少$miss，实际：$line"; fi
  elif [ -z "$expect" ] && [ "$rc" != "0" ]; then
    ok "$label → 被拒绝（rc=$rc，负控成立）"
  elif [ -n "$expect" ]; then
    bad "$label：期望成功解析，实际 rc=$rc：$(printf '%s' "$out" | tail -2 | tr '\n' ' ')"
  else
    bad "$label：期望被拒绝，实际 rc=0（**门失效**）：$(printf '%s' "$out" | grep RESOLVED | head -1)"
  fi
}

echo "[ced-defaults] 交付默认值（2026-09-27：DSpark 开 + 前缀缓存开）"
run "" "负控: prefill + SPEC=1 必须拒绝"        prefill SPEC=1
run "" "负控: decode + SPEC=2 必须拒绝"         decode  SPEC=2
run "" "负控: decode + DRAFT_GRAPH=2 必须拒绝"  decode  DRAFT_GRAPH=2

echo "[ced-defaults] 正控：默认解析"
run "spec=1 draft=1 prefix=1 static=1 allow_dspark=1" "decode 默认（= 交付口径）" decode
run "spec=0 draft=0 prefix=1 static=0" "prefill 默认（P 永不开 DSpark）" prefill

echo "[ced-defaults] 显式关闭仍有效（旧写法兼容）"
run "spec=0 draft=0" "V41_CED_ALLOW_DSPARK=0 ⇒ 退回 SPEC=0" decode V41_CED_ALLOW_DSPARK=0
run "prefix=0"       "PREFIX=0 ⇒ 关缓存"                    decode PREFIX=0
run "prefix=0"       "V41_CED_ALLOW_PREFIX=0 ⇒ 关缓存"      decode V41_CED_ALLOW_PREFIX=0
run "spec=0 draft=0" "decode 显式 SPEC=0 DRAFT_GRAPH=0"     decode SPEC=0 DRAFT_GRAPH=0

echo "[ced-defaults] 引擎侧同门（model.py 的默认必须是 1，否则默认启动会在构造期崩）"
if grep -q '_os_ids.environ.get("V41_CED_ALLOW_DSPARK", "1")' "$PKG/patches/files/model.py"; then
  ok "patches/files/model.py 的 V41_CED_ALLOW_DSPARK 默认 = 1"
else
  bad "patches/files/model.py 的 V41_CED_ALLOW_DSPARK 默认不是 1 ⇒ 脚本默认与引擎门**不一致**"
fi

# deploy 形态（launch/_common.sh）的 PREFIX 默认也要跟上，否则两种交付面口径不同
if grep -qE '^export PREFIX=\$\{PREFIX:-1\}' "$PKG/deploy/a3-ced-pd/launch/_common.sh"; then
  ok "deploy/a3-ced-pd/launch/_common.sh 的 PREFIX 默认 = 1"
else
  bad "deploy/a3-ced-pd/launch/_common.sh 的 PREFIX 默认不是 1 ⇒ 两种交付面口径不一致"
fi

echo
if [ "$fail" = "0" ]; then
  echo "[ced-defaults] 全部通过 ✅（$pass 项）"
  exit 0
fi
echo "[ced-defaults] 有 $fail 项失败 ❌（通过 $pass 项）" >&2
exit 1
