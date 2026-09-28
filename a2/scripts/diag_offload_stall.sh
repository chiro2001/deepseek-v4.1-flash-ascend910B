#!/usr/bin/env bash
# =============================================================================
# diag_offload_stall.sh —— A2「DRAM 池满之后卡住 / 慢」的**只读取证**（不重启、不发模型请求）
#
# 背景：2026-09-28 在 A3（chip 8–15，同镜像同配置）复现 A2 形态时，
#   抓到的第一手证据是 **HBM 分配失败后卡在分配器重试**（不是"慢"，是"卡"）：
#       [ERROR] RUNTIME: MallocPhysical: halMemCreate failed ... ErrCode=207001
#               desc=[driver error:out of memory]
#       [ERROR] APP: NPUCachingAllocator.cpp:malloc:1265: "Get a block from the
#               existing pool failed. Try to free cached blocks and reallocate."
#   形态：engine 停在某一步不返回 / NPU 的 AICore **0%** / 8 个 worker 各烧 ~25% CPU。
#   ⇒ 本脚本把这条证据与另外三条（gate 串行、取回字节数、逐组淘汰）一起收齐。
#
# 用法（在 A2 宿主上，容器名默认 dsv41-a2）：
#     bash a2/scripts/diag_offload_stall.sh
#     NAME=dsv41-a2 PORT=8077 bash a2/scripts/diag_offload_stall.sh
#
# 判据口径（每条都写清"看什么值 = 什么结论"）：
#   ① plog 里的 207001 / out of memory  ⇒ ★ HBM 真的分配失败过（配合 ② 一起看）
#   ② 两次 /metrics 的 step 计数差值     ⇒ 引擎有没有在推进（0 = 卡住）
#   ③ [admission_gate] prefill-only 行   ⇒ decode 被串行挡了多少（假设 a）
#   ④ kv_offload_total_bytes CPU_to_GPU  ⇒ 取回**真的搬了字节**吗（假设 b）
#   ⑤ [D2_offload] miss-scan             ⇒ key 在池里但前缀链头被挤掉（= 取回救不了）
# =============================================================================
set -uo pipefail

NAME=${NAME:-dsv41-a2}
PORT=${PORT:-8077}
RUN_DIR=${RUN_DIR:-$(ls -dt "$HOME"/projects/dsv41-release/results/a2_* 2>/dev/null | head -1)}
LOG=${LOG:-$RUN_DIR/serve.log}

hr() { printf '%s\n' "------------------------------------------------------------------------"; }
sec() { printf '\n=== %s ===\n' "$1"; }

echo "A2 卸载卡顿取证   $(date '+%F %T %Z')   host=$(hostname)"
echo "容器=$NAME 端口=$PORT 日志=$LOG"

sec "① HBM 分配失败（★ 最关键的一条）"
if docker exec "$NAME" bash -lc 'ls /root/ascend/log/debug/plog/ >/dev/null 2>&1'; then
    n_oom=$(docker exec "$NAME" bash -lc \
        'grep -l "207001\|out of memory" /root/ascend/log/debug/plog/*.log 2>/dev/null | wc -l')
    echo "  含 207001 / out of memory 的 plog 文件数：$n_oom"
    if [ "${n_oom:-0}" != "0" ]; then
        docker exec "$NAME" bash -lc \
            'grep -h "207001\|out of memory\|Try to free cached blocks" /root/ascend/log/debug/plog/*.log 2>/dev/null | tail -12' \
            | cut -c1-200 | sed 's/^/    /'
        echo "  ⇒ 结论：发生过 HBM 分配失败（NPUCachingAllocator 在重试）。"
        echo "     处置：给激活留更多 HBM（GPU_UTIL 降 0.02–0.04，或 KV_CACHE_MEMORY_BYTES 降 1–2 GiB），"
        echo "           或让 engram gate 分块以压低峰值激活（GATE_CHUNK=512 且 GATE_MAX_TOKENS=8192）。"
    else
        echo "  （没找到 207001 —— 那【卡住】更可能是 gate 串行 / 取回路径，见 ③④⑤）"
    fi
else
    echo "  （容器内没有 /root/ascend/log/debug/plog，跳过；若容器名不对会看到这行）"
fi

sec "② 引擎还在推进吗（两次采样）"
m() { curl -sS --noproxy '*' -m 10 "http://127.0.0.1:$PORT/metrics"; }
S1=$(m); sleep 10; S2=$(m)
for k in iteration_tokens_total_count num_requests_running num_requests_waiting; do
    a=$(printf '%s' "$S1" | grep -E "^vllm:$k\{" | head -1 | awk '{print $2}')
    b=$(printf '%s' "$S2" | grep -E "^vllm:$k\{" | head -1 | awk '{print $2}')
    printf '  %-28s %s -> %s\n' "$k" "${a:-NA}" "${b:-NA}"
done
echo "  （iteration_tokens_total_count 10 s 内不涨、而 requests_running>0 ⇒ 引擎卡在某一步）"

sec "③ [admission_gate]：prefill-only 串行（假设 a）"
if [ -f "$LOG" ]; then
    echo "  行数：$(grep -ac 'admission_gate' "$LOG")"
    grep -a 'admission_gate' "$LOG" | tail -12 | cut -c1-200 | sed 's/^/    /'
    echo "  （连续多条 prefill-only 且间隔的 step 数 == 条数 ⇒ 这些 step 全是 prefill，decode 被推迟）"
    echo "  episode 结束行里的 deferred_decode_reqs = 被推迟的 decode 请求数"
    printf '  ★「prefill 选不中、被迫插 decode」的次数 = %s\n' \
        "$(grep -ac 'prefill-only step could not be scheduled' "$LOG")"
    grep -a 'could not be scheduled' "$LOG" | tail -6 | cut -c1-200 | sed 's/^/    /'
    echo "  （这条连成串 ⇒ ★ 正在等 KV 取回的请求把每个 step 都"占"成 prefill 步，"
    echo "    而它自己又不能算 ⇒ 只能靠 2/4/8 步的强制 decode 兜底 = 取回期间 decode 被限流）"
else
    echo "  找不到日志 $LOG（用 LOG=<serve.log 绝对路径> 指定）"
fi

sec "④ 取回字节数：CPU_to_GPU 有没有真的涨（假设 b）"
printf '%s' "$S2" | grep -E '^vllm:kv_offload_total_bytes_total\{|^vllm:kv_offload_size_sum\{' | sed 's/^/  /'
printf '  external_prefix_cache_hits_total = %s\n' "$(printf '%s' "$S2" | grep -E '^vllm:external_prefix_cache_hits_total\{' | awk '{print $2}')"
echo "  （CPU_to_GPU 的 bytes 不涨但 external hits 在涨 ⇒ 命中被判出来了但**字节没搬**）"

sec "⑤ 池满时的逐组淘汰 / 前缀链头缺失（假设 b 的另一面）"
if [ -f "$LOG" ]; then
    printf '  配额不足行数=%s，其中 abandoned=True=%s\n' \
        "$(grep -ac '配额不足' "$LOG")" "$(grep -ac 'abandoned=True' "$LOG")"
    echo "  [D2_offload] miss-scan（present>0 但扫描 MISS ⇒ 池里有块、前缀链头被 LRU 挤掉）："
    grep -a 'miss-scan' "$LOG" | tail -8 | cut -c1-200 | sed 's/^/    /'
    echo "  [D2_offload] lookup-summary："
    grep -a 'lookup-summary' "$LOG" | tail -5 | cut -c1-200 | sed 's/^/    /'
fi

sec "⑥ NPU 侧瞬时状态（AICore / HBM）"
npu-smi info 2>/dev/null | grep -E '^\| [0-9]+ +[0-9]+ ' | sed 's/^/  /' | head -12
echo "  （AICore 反复 0/100 或长期 0 + 上层 worker 各烧 ~25% CPU ⇒ 与本次在 A3 抓到的形态同族）"

hr
echo "把上面 ①②③④⑤ 的输出回传即可定位是【HBM 分配重试】【gate 串行】还是【取回没生效】。"
