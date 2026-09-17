#!/bin/bash
# CPython 3.12.13 PGO+LTO 完整构建（后台运行，日志落 /work/logs/make.log）
# --enable-optimizations 使默认 all 规则 = profile-opt：
#   1) build_all_generate_profile  (-fprofile-generate)
#   2) 运行 PROFILE_TASK 训练   (默认: -m test --pgo --timeout=1200)
#   3) build_all_use_profile      (-fprofile-use -fprofile-correction)
set -uo pipefail
cd /work/src/cpython-3.12.13

date -Is > /work/logs/make_start_iso.txt
START=$(date +%s)
echo "$START" > /work/logs/make_start_epoch.txt
echo "START=$START ($(date -Is))" > /work/logs/make.log

JOBS=${PGO_JOBS:-48}
echo "JOBS=$JOBS  PROFILE_TASK=${PROFILE_TASK:-<CPython 默认 -m test --pgo>}" >> /work/logs/make.log
# PROFILE_TASK 保持 CPython 默认（-m test --pgo）。它的收益已被实测证明可跨负载转移：
# 训练用测试套件，而在 tiny_call / dict_loop / list_append 等**完全不同**的模式上
# 仍拿到 −16~23%（reports/cpython-pgo-verified.md）。
nice -n 10 make -j"$JOBS" >> /work/logs/make.log 2>&1
RC=$?
echo "$RC" > /work/logs/make_exit.txt

END=$(date +%s)
echo "$END" > /work/logs/make_end_epoch.txt
echo "$((END-START))" > /work/logs/make_seconds.txt
date -Is > /work/logs/make_end_iso.txt
echo "DONE rc=$RC elapsed=$((END-START))s at $(date -Is)" >> /work/logs/make.log
