#!/usr/bin/env bash
# =============================================================================
# 动态 K「草稿元数据行数不匹配」第二轮取证 —— 一条命令跑完（2026-10-03）
#
# 背景与判据见 ../docs/DYNAMIC-SPEC-DRAFT-ROWS-MISMATCH-20261003.md
#
# 这一轮要回答的**唯一问题**：
#   dsa_v1.build_dspark_swa_indices 的 `expected index shape 2 smaller than
#   self shape 1` 到底出在哪一段？三种互斥情形：
#     ① `pre-window` 就不一致  ⇒ per-group block table 切片太短
#        ⇒ 开 V41_DYNSPEC_BT_PERSIST=1（候选修复）复测
#     ② 只有 `post-window` 不一致 ⇒ 问题在 SlidingWindowAdapter
#        ⇒ 下一步去修 spec_decode/utils.py（该文件目前**不在**我们的挂载清单里）
#     ③ 两者都一致、只有 dsa_v1 报错 ⇒ build 期与 run 期的 num_reqs 偏移
#        ⇒ 候选修复无效，要去查 builder 存切片那两处
#
# 用法（本机 = server-mini）：
#     bash fixes/20261003-dynspec/round2_runbook.sh
#
# 全部只读 pin 在 a3-21 的 ~/cedpd-repo + ~/tmp/a3perf；不动交付档以外的任何东西。
# =============================================================================
set -uo pipefail

HOST=${HOST:-a3-21}
REPO_REMOTE=${REPO_REMOTE:-/home/l00886679/cedpd-repo}
LOCAL_REPO=${LOCAL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PORT=${PORT:-19210}
# R2_ROUND=A（默认）= 基线 + 探针；B = 开候选修复复测
R2_ROUND=${R2_ROUND:-A}
case "$R2_ROUND" in A) PERSIST=0 ;; B) PERSIST=1 ;; *) echo "R2_ROUND 只能是 A 或 B"; exit 2 ;; esac
NAME=${NAME:-dsv41-dynfix}
STAMP=$(date +%m%d_%H%M%S)
OUT_REMOTE="$HOME/tmp/a3perf/dynr2_$STAMP"

say() { printf '\033[1m[r2]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[r2][FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------- 0) 连通性 ----------
say "探测 $HOST …"
timeout 25 ssh -o ConnectTimeout=12 "$HOST" 'echo OK' >/dev/null 2>&1 \
  || die "连不上 $HOST（若是 VPN 断开，先等恢复再跑本脚本）"

# ---------- 1) 同步三个带诊断的文件 ----------
# 远端此前有过 ad-hoc 的手工插入，这里用仓库版本**覆盖**，保证两边逐字节一致。
say "同步诊断文件（仓库版本 → 远端 overlay）"
for f in dsa_v1.py llm_base_proposer.py dspark_proposer.py; do
  scp -q "$LOCAL_REPO/patches/files/draft/$f" "$HOST:$REPO_REMOTE/patches/files/draft/$f" \
    || die "scp $f 失败"
done
say "远端语法自检"
timeout 60 ssh "$HOST" "cd $REPO_REMOTE && for f in dsa_v1 llm_base_proposer dspark_proposer; do python3 -c \"import ast,sys; ast.parse(open('patches/files/draft/\$f.py',encoding='utf-8').read())\" || exit 1; done; echo SYNTAX_OK" \
  | tail -1 | grep -q SYNTAX_OK || die "远端语法自检失败"
say "  语法 OK"

# ---------- 2) 重启动态 K 实例（默认档：候选修复**关**） ----------
say "重启 $NAME（SP_SCHEDULE='1,1,7;2,32,0'，ROUND=$R2_ROUND，V41_DYNSPEC_BT_PERSIST=$PERSIST）"
timeout 120 ssh "$HOST" "docker rm -f $NAME >/dev/null 2>&1; mkdir -p $OUT_REMOTE; \
cd $REPO_REMOTE && nohup env \
  MODEL=/home/l00886679/models/out/v41-flat-verify3 \
  DEVS='8 9 10 11 12 13 14 15' TP=8 DP=1 NAME=$NAME PORT=$PORT \
  SPEC=1 SP_TOKENS=7 DRAFT_GRAPH=1 SP_SCHEDULE='1,1,7;2,32,0' \
  V41_CED_DYNAMIC_SPEC_FULL_GRAPHS=1 V41_DYNSPEC_BT_PERSIST=$PERSIST \
  MAX_SEQS=32 BAT_TOKENS=8192 PREFIX=1 ENGRAM=1 ENGRAM_DEVICE_INDEX=0 \
  GPU_UTIL=0.92 PATCH_MODE=mount PROFILE=1 VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  RUN_ID=dynr2${R2_ROUND}_$STAMP \
  bash scripts/serve_a3.sh > $OUT_REMOTE/serve.log 2>&1 & echo STARTED" \
  | tail -1 | grep -q STARTED || die "起服命令提交失败"

# ---------- 3) 等就绪（动态 K 要为 ql=1/8 各建一组图，约 25 分钟） ----------
say "等就绪（最多 40 分钟；每 60s 报一次）"
ok=0
for i in $(seq 1 40); do
  sleep 60
  code=$(timeout 20 ssh "$HOST" "curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health" 2>/dev/null | tail -1)
  say "  ${i}min health=$code"
  if [ "$code" = "200" ]; then ok=1; break; fi
  # 起服已经 die 的话不必等满
  if timeout 20 ssh "$HOST" "grep -aq '\[serve_a2\]\[FAIL\]' $OUT_REMOTE/serve.log 2>/dev/null && echo FAILED" 2>/dev/null | grep -q FAILED; then
    die "起服失败，见 $OUT_REMOTE/serve.log"
  fi
done
[ "$ok" = "1" ] || die "40 分钟仍未就绪"
say "就绪 ✓"

# ---------- 4) 打探针 ----------
say "跑 dynprobe（1/2/4/8/16 × 3 rep）"
timeout 1800 ssh "$HOST" "cd ~ && python3 ~/tmp/dynprobe.py $PORT 1,2,4,8,16 7 3 256 2>&1 | tail -30" || true

# ---------- 5) 收诊断 ----------
say "收集 [DYNSPEC-DIAG] / dynamic-spec 告警"
timeout 60 ssh "$HOST" "L=$OUT_REMOTE/serve.log; \
  echo '=== DYNSPEC-DIAG ==='; grep -a 'DYNSPEC-DIAG' \$L | head -20; \
  echo '=== dynamic-spec ★ ==='; grep -a 'dynamic-spec] ★' \$L | head -5; \
  echo '=== no uniform decode graph ==='; grep -ac 'no uniform decode graph' \$L; \
  echo '=== 500/EngineDead ==='; grep -ac 'EngineDeadError' \$L" || true

cat <<EOF

[r2] 完成。远端产物：$OUT_REMOTE

判读（照抄 docs/DYNAMIC-SPEC-DRAFT-ROWS-MISMATCH-20261003.md §4）：
  ① 有 pre-window 不一致  → 候选修复对症：加 V41_DYNSPEC_BT_PERSIST=1 重跑本脚本
  ② 只有 post-window 不一致 → 去修 spec_decode/utils.py（需先加挂载）
  ③ 两者都一致、dsa_v1 才报错 → build/run 偏移，候选修复无效
若本轮**没有** DYNSPEC-DIAG 且引擎存活 ⇒ 说明只是偶发；把 dynprobe 的
1/2/4/8/16 曲线当交付数据，并补跑正确性探针（144K/1M 四针 + 并发 2 各带不同针）。
第二轮（候选修复）直接用：  R2_ROUND=B bash fixes/20261003-dynspec/round2_runbook.sh
（判据：B 轮应**无** DYNSPEC-DIAG、无 EngineDead，且 dynprobe 1/2/4/8/16 全部跑完）
EOF
