#!/usr/bin/env bash
# =============================================================================
# selftest_make_shadow_pkg.sh —— `make_shadow_pkg.sh` 生成物的**回归自测**（零 docker、零 NPU）
#
# 为什么需要（2026-09-23 真机两次踩到，都是"生成物"层面的错，`bash -n` 查不出来）：
#   ① int8 块与生产块都挂 `models/deepseek_v41/model.py` ⇒ docker `Duplicate mount point`；
#   ② 去重时只删了"路径"那一半、留下孤立的 `-v` ⇒ docker `invalid reference format.`
#   ⇒ 判据必须落在**生成物**（真实 MOUNTS 清单）上：成对完整 + 目标唯一。
#
# 用法： bash a2/scripts/selftest_make_shadow_pkg.sh
# 退出码：0 = 全过；9 = 有失败
# =============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
CHK=$HERE/check_mount_list.py

V=0; F=0
ok()  { printf '  \033[32mPASS\033[0m  %s\n' "$*"; V=$((V+1)); }
bad() { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; F=$((F+1)); }
say() { printf '\n==== %s ====\n' "$*"; }

T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
SH=$T/shadow

say "① 造 shadow（PKG=$REPO）"
if PKG="$REPO" DST="$SH" bash "$HERE/make_shadow_pkg.sh" >"$T/gen.log" 2>&1; then
    ok "生成 rc=0"
else
    bad "生成失败"; tail -8 "$T/gen.log" | sed 's/^/        /'
fi
[ -f "$SH/scripts/serve_a2.sh" ] && ok "生成物在位：scripts/serve_a2.sh" || bad "生成物缺失"

mkdir -p "$T/fakemodel"; echo '{"model_type":"deepseek_v41"}' > "$T/fakemodel/config.json"

dry() {   # $1 = 输出日志；其余 = 额外 env
    local log="$1"; shift
    ( cd "$SH" && env SHADOW_PKG="$SH" PATCH_MODE=mount DRY_RUN=1 ENGRAM=0 \
        MODEL="$T/fakemodel" "$@" bash scripts/serve_a2.sh ) >"$log" 2>&1
    return $?
}

say "② int8 开（PATCH_MODE=mount）⇒ 必须**摘掉生产块那条**，且清单成对完整、目标唯一"
dry "$T/int8.log" A2_KV8_SWA=1 A2_RING_FP16=1 A2_GRAPH_SAFE=1
rc=$?
[ "$rc" = "0" ] && ok "②a 干跑 rc=0" || { bad "②a rc=$rc"; tail -10 "$T/int8.log" | sed 's/^/        /'; }
grep -q "摘掉生产块那条挂载" "$T/int8.log" && ok "②b 打印了「摘掉生产块」（说明去重逻辑真的跑了）" \
    || bad "②b 没有「摘掉生产块」——去重没生效"
if python3 "$CHK" --from-log "$T/int8.log" >"$T/chk1.log" 2>&1; then
    ok "②c 挂载清单成对完整 + 目标唯一（$(cat "$T/chk1.log")）"
else
    bad "②c 挂载清单有问题"; cat "$T/chk1.log" | sed 's/^/        /'
fi
# 目标恰好一次
_cnt=$(grep -ao "models/deepseek_v41/model.py:ro\|models/deepseek_v41/model.py:rw" "$T/int8.log" | wc -l)
[ "$_cnt" = "1" ] && ok "②d model.py 目标恰好出现 1 次" || bad "②d model.py 目标出现 $_cnt 次（应为 1）"

say "③ 反例对照：int8 关 ⇒ **不许**摘（生产块那条必须在，且同样恰好 1 次）"
dry "$T/noi8.log"
rc=$?
[ "$rc" = "0" ] && ok "③a 干跑 rc=0" || bad "③a rc=$rc"
if grep -q "摘掉生产块那条挂载" "$T/noi8.log"; then
    bad "③b int8 关的时候也去摘了（不该）"
else
    ok "③b int8 关 ⇒ 没摘任何东西"
fi
_cnt2=$(grep -ao "models/deepseek_v41/model.py:ro\|models/deepseek_v41/model.py:rw" "$T/noi8.log" | wc -l)
[ "$_cnt2" = "1" ] && ok "③c int8 关时 model.py 目标出现 1 次（生产块仍在）" \
    || bad "③c int8 关时 model.py 出现 $_cnt2 次（应 1；0 说明挂载被误删）"
python3 "$CHK" --from-log "$T/noi8.log" >/dev/null 2>&1 && ok "③d 清单成对完整" || bad "③d 清单有问题"

say "④ 校验器自身的负控（判据必须能报错，否则等于没判）"
# ★ 用 --from-log（喂**构造的 MOUNTS 行**）：`--tokens -v …` 会被 argparse 当成选项，不可靠。
printf '[a2-dry] MOUNTS(3): -v A:/t:ro -v\n'        > "$T/t_orphan.log"
printf '[a2-dry] MOUNTS(4): -v A:/t:ro -v B:/t:ro\n' > "$T/t_dup.log"
printf '[a2-dry] MOUNTS(4): -v A:/t1:ro -v B:/t2:ro\n' > "$T/t_ok.log"
if python3 "$CHK" --from-log "$T/t_orphan.log" >/dev/null 2>&1; then
    bad "④a 孤立 -v 竟然判成 OK"
else
    ok "④a 孤立 -v 被判 FAIL"
fi
if python3 "$CHK" --from-log "$T/t_dup.log" >/dev/null 2>&1; then
    bad "④b 重复目标竟然判成 OK"
else
    ok "④b 重复目标被判 FAIL"
fi
python3 "$CHK" --from-log "$T/t_ok.log" >/dev/null 2>&1 \
    && ok "④c 正常清单判 OK" || bad "④c 正常清单被误判"

echo
echo "=============== 通过 $V 条 / 失败 $F 条 ==============="
if [ "$F" = "0" ] && [ "$V" -ge 10 ]; then echo "✅ 自测全过"; exit 0; fi
echo "❌ 不合格"; exit 9
