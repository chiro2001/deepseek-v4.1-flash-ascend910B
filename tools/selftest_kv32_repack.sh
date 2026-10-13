#!/usr/bin/env bash
# KV32 slot 重排补丁的离线自检：1 正控 + 3 负控，全部**不占卡**、秒级。
#
# 为什么需要它：这个补丁只改 placement 的 (offset, size)，一旦写错不会起服报错 ——
# 而是在长上下文时表现为"静默读错数据"。三种真实失效模式各配一个负控：
#   ① 没有挪动（改了等于没改）   → 槽长仍 [131072,131072,131072,147712]
#   ② 挪动偏移侵入同源平面       → slot0 里同组真实范围相交
#   ③ 挪动大小写错（声明页漂移） → page_size_padded 变 ⇒ npr/容量口径变
#
# 判据：正控 exit=0 且打 "PASS"；三个负控都 exit!=0 且各自命中对应的那条检查。
#
# 跑法：bash tools/selftest_kv32_repack.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
PATCH=${PATCH:-patches/files/deepseek_v41.repack.py}
SIM=tools/kv32_repack_sim.py
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
n_ok=0; n_bad=0

ok()   { echo "  ✓ $1"; n_ok=$((n_ok + 1)); }
bad()  { echo "  ✗ $1"; n_bad=$((n_bad + 1)); }

echo "== 正控：真补丁必须 PASS（被检 $PATCH）=="
out=$(python3 "$SIM" "$PATCH" 2>&1); rc=$?
if [ "$rc" = "0" ] && printf '%s' "$out" | grep -q "SIM: PASS"; then
  ok "真补丁 PASS：$(printf '%s' "$out" | grep -m1 'SIM: PASS')"
  printf '%s\n' "$out" | grep -E "声明页|同组真实范围|slot[0-3]（页" | sed 's/^/      /'
else
  bad "真补丁没有 PASS（rc=$rc）"; printf '%s\n' "$out" | tail -3 | sed 's/^/      /'
fi

# 用 python 从真补丁派生三个变异体（保证变异点存在，否则自检自身失效）
python3 - "$PATCH" "$TMP" <<'PY'
import pathlib, sys
src = pathlib.Path(sys.argv[1]).read_text(); tmp = pathlib.Path(sys.argv[2])
MOVE = '_extra[j] = [(i, _used[j], p["index_bytes"])]'
assert MOVE in src, "变异锚点 MOVE 不存在（补丁被改写？）"
(tmp / "m_noop.py").write_text(src.replace(MOVE, '_extra[j] = []  # noop'))
(tmp / "m_overlap.py").write_text(src.replace(MOVE, '_extra[j] = [(i, _used[j] - 4000, p["index_bytes"])]'))
(tmp / "m_size.py").write_text(src.replace(MOVE, '_extra[j] = [(i, _used[j], 100)]'))
print("mutants ok")
PY
[ $? -eq 0 ] || { bad "变异体生成失败"; echo "结果：$n_ok 通过，$((n_bad+1)) 失败"; exit 1; }

check_neg() {   # <文件> <期望命中的关键词> <标签>
  local f=$1 want=$2 label=$3
  local out rc
  out=$(python3 "$SIM" "$f" 2>&1); rc=$?
  if [ "$rc" = "0" ]; then
    bad "[$label] 应判失败却 PASS ⇒ 检查没有鉴别力"
  elif printf '%s' "$out" | grep -q "$want"; then
    ok "[$label] 按预期被拦（命中「$want」）"
  else
    bad "[$label] 判失败但没命中「$want」"; printf '%s\n' "$out" | tail -2 | sed 's/^/      /'
  fi
}

echo "== 负控 1：没有挪动 ⇒ 被补丁自带的覆盖检查拦住（比本检查更早）=="
# noop 变体把挪动去掉、但 index 的 placement 也被去掉 ⇒ 补丁自己的
# "V4.1 slot placement must cover each resource exactly once" 先报错。
# 两种报错都算通过（关键是**必须判失败**）。
_NEG1=$(python3 tools/kv32_repack_sim.py "$TMP/m_noop.py" 2>&1); _RC1=$?
if [ "$_RC1" = "0" ]; then
  bad "[noop] 应判失败却 PASS ⇒ 检查没有鉴别力"
elif printf '%s' "$_NEG1" | grep -qE "槽长不齐|must cover each resource exactly once"; then
  ok "[noop] 按预期被拦（补丁自带覆盖检查）"
else
  bad "[noop] 判失败但没命中预期关键词"; printf '%s\n' "$_NEG1" | tail -2 | sed 's/^/      /'
fi
echo "== 负控 2：偏移侵入同源平面 ⇒ 同组真实范围相交 =="
check_neg "$TMP/m_overlap.py" "同组真实范围相交" "overlap"
echo "== 负控 3：挪动大小写错 ⇒ 声明页漂移 =="
check_neg "$TMP/m_size.py" "声明页漂移" "size"

echo
echo "结果：$n_ok 通过，$n_bad 失败"
[ "$n_bad" = "0" ] || exit 1
