#!/usr/bin/env bash
# collect_evidence.sh —— **一条命令**把"能让我（或任何人）离线复算"的证据收集成一个文件
#
# 为什么需要它：一次出问题的臂会产生 5–10 个文件、`serve.log` 动辄几万行，
#   直接 `cat` 回来**没法看**；而只挑几行又会丢掉上下文（本仓反复栽在"证据不全 ⇒ 结论想当然"）。
#   本脚本按**固定清单**抽取"唯一能定位问题的那几十行"，并**保留原始行**（可复算）。
#
# 它做的事（**全部只读**；不改配置、不重启、不碰 NPU）：
#   ① 找到最近一次 run 目录（或 `RUN_ID=` 指定）
#   ② 抽 run 元信息 / 容器内指纹 / inner.sh 的真实 env（= 实际生效的配置）
#   ③ 抽 serve.log 里的：起服门、KV 容量、SpecDecoding(A)、错误与警告、卸载计数
#   ④ 抽 metrics / kv_events（有就抽）
#   ⑤ ★ 若服务还活着：跑一次**文本探针**并把它一起收进来（乱码问题最关键的一段原文）
#   ⑥ 全部汇成一个 `EVIDENCE.txt`，并打印路径
#
# 用法：
#   bash a2/scripts/collect_evidence.sh                    # 最近一次 run
#   RUN_ID=a2_20260923_094624 bash a2/scripts/collect_evidence.sh
#   RUN_DIR=/任意/路径/a2_20260923_094624 bash a2/scripts/collect_evidence.sh
#   SERVE_LOG=/任意/路径/serve.log bash a2/scripts/collect_evidence.sh      # ★ 只给日志文件
#   bash a2/scripts/collect_evidence.sh --serve-log /任意/路径/serve.log    # ★ 同上（flag 形式）
#   bash a2/scripts/collect_evidence.sh --run-dir  /任意/路径/run           # ★ 同上（flag 形式）
#   RESULTS=/另一个/results bash a2/scripts/collect_evidence.sh            # 换 results 根
#   PORT=8077 bash a2/scripts/collect_evidence.sh
#   NO_PROBE=1 bash a2/scripts/collect_evidence.sh         # 跳过文本探针
#
# ★ "位置"优先级（高 -> 低）：--serve-log / SERVE_LOG  >  --run-dir / RUN_DIR  >  RUN_ID  >  results 里最新的一个
#   —— 给了 SERVE_LOG 就**只需要那个文件存在**，同目录下的 serve_cmd.txt / inner.sh 有就抽、没有就跳过；
#      此时 RUN_DIR 默认取它的**父目录**（所以产物仍能跟着日志走）。
#
# ★★ 平台：`PLAT=a2|a3`（默认 a2）。它只改三个**平台不同**的默认值，显式给的一律优先：
#     PLAT  | 容器名     | 端口 | 模型名（API body 的 "model" 字段）
#     a2    | dsv41-a2   | 8077 | deepseek-v4-flash
#     a3    | dsv41-a3   | 8020 | deepseek-v41        ← `scripts/serve_a3.sh` 的 SERVED_NAME 默认
#   ★ 为什么必须区分模型名：探针要拿它填 `/v1/chat/completions` 的 `"model"` 字段，
#     填错会被服务端 400 ⇒ 表现是"探针全失败"，很容易被误读成"模型坏了"。
#   ★ 为什么必须区分容器名：容器名对不上 ⇒ **§② 容器内指纹整段静默跳过**（老版本就是这个形态），
#     而指纹恰恰是"挂的到底是哪一份"的唯一判据。
#
# 退出码：0 = 已生成；64 = 找不到 run 目录 / 日志文件
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PKG=$(cd "$HERE/../.." && pwd)

# ------------------------------------------------- 命令行 flag（★ 与 env 等价，flag 赢）
while [ $# -gt 0 ]; do
    case "$1" in
        --serve-log|-l) SERVE_LOG=${2:-}; shift 2 ;;
        --serve-log=*)  SERVE_LOG=${1#*=}; shift ;;
        --run-dir|-d)   RUN_DIR=${2:-};   shift 2 ;;
        --run-dir=*)    RUN_DIR=${1#*=};   shift ;;
        --run-id)       RUN_ID=${2:-};     shift 2 ;;
        --run-id=*)     RUN_ID=${1#*=};    shift ;;
        --outdir|-o)    OUTDIR=${2:-};     shift 2 ;;
        --outdir=*)     OUTDIR=${1#*=};    shift ;;
        --plat)         PLAT=${2:-};       shift 2 ;;
        --plat=*)       PLAT=${1#*=};      shift ;;
        --port)         PORT=${2:-};       shift 2 ;;
        --port=*)       PORT=${1#*=};      shift ;;
        -h|--help)      sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "⚠ 未知参数：$1（用 --help 看用法）" >&2; shift ;;
    esac
done

# ★★ 平台差异收敛成一张小表（别的都不分平台）—— 见文件头"平台"那一段。
PLAT=${PLAT:-a2}
case "$PLAT" in
  a2) _def_port=8077; _def_ctr=dsv41-a2; _def_model=deepseek-v4-flash ;;
  a3) _def_port=8020; _def_ctr=dsv41-a3; _def_model=deepseek-v41 ;;
  *)  echo "⛔ PLAT 只能是 a2|a3，得到 '$PLAT'" >&2; exit 64 ;;
esac
PORT=${PORT:-$_def_port}
URL=${URL:-http://127.0.0.1:$PORT}
CTR=${CTR:-$_def_ctr}
MODEL_NAME=${MODEL_NAME:-$_def_model}
SHADOW=${SHADOW_PKG:-$HOME/projects/dsv41-upstream-pr/shadow-pkg}
RESULTS=${RESULTS:-$SHADOW/results}
STAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR=${OUTDIR:-$HOME/evidence_$STAMP}
OUT=$OUTDIR/EVIDENCE.txt
PROBES=${PROBES:-3}                      # 文本探针重发次数（乱码要看"稳不稳"）
NO_PROBE=${NO_PROBE:-0}

# ---------------------------------------------------------------- 找 run 目录 / 日志
# ★ 本段三种入口都要能用（用户点名："collect 脚本要能指定 log 位置"）：
#   ① SERVE_LOG=<文件>：只认这个文件；RD = 它的父目录（用于抽 serve_cmd.txt / inner.sh）
#   ② RUN_DIR=<目录> ：RD = 它；SL = 它下面的 serve.log
#   ③ RUN_ID / 自动取最新：RD = $RESULTS/<id> 或 $RESULTS 里最新的目录
#   退出码仍只在"三种都失败"时才 64。
if [ -n "${SERVE_LOG:-}" ]; then
    [ -f "$SERVE_LOG" ] || { echo "⛔ --serve-log / SERVE_LOG 指向的文件不存在或不是文件：$SERVE_LOG" >&2; exit 64; }
    SL="$SERVE_LOG"
    RD=${RUN_DIR:-$(cd "$(dirname "$SL")" && pwd)}
elif [ -n "${RUN_DIR:-}" ]; then
    [ -d "$RUN_DIR" ] || { echo "⛔ --run-dir / RUN_DIR 指向的目录不存在：$RUN_DIR" >&2; exit 64; }
    RD="$RUN_DIR"
    SL="$RD/serve.log"
    [ -f "$SL" ] || echo "  ⚠ $RD 下没有 serve.log —— 别的段照样抽，serve.log 相关的段会写 (无)" >&2
else
    if [ -n "${RUN_ID:-}" ]; then
        RD="$RESULTS/$RUN_ID"
    else
        RD=$(ls -td "$RESULTS"/*/ 2>/dev/null | head -1)
        RD=${RD%/}
    fi
    SL="$RD/serve.log"
fi
[ -n "${RD:-}" ] || {
    echo "⛔ 找不到 run 目录（找过 $RESULTS）。用 --run-dir <目录> / --serve-log <日志> / RUN_ID=<名字> 指定。" >&2
    exit 64; }

mkdir -p "$OUTDIR" || exit 64

SH="$(dirname "$HERE")"                   # a2/

# ★ 计数助手：`grep -c` 在"无匹配"时**仍会打印 0 但退出码是 1**，
#   若写成 `$(grep -c ... || echo 0)` 就会输出两行 0（实测踩到）。
#   ⇒ 捕获到变量里再 `${n:-0}`（文件不存在时 grep 不打计数，正好落到默认值）。
_cnt() { local n; n=$(grep -c -a -- "$1" "$2" 2>/dev/null); echo "${n:-0}"; }

_hdr() { printf '\n%s\n%s\n%s\n' "================================================================================" "$*" "================================================================================"; }
_sec() { _hdr "§ $*"; }

{
_hdr "EVIDENCE —— 一次出问题的臂的全部可复算证据" \
     "  生成：$(date '+%F %T')   宿主：$(hostname)   收集器：collect_evidence.sh" \
     "  PLAT=$PLAT  容器=$CTR  端口=$PORT  模型名=$MODEL_NAME（★ 平台默认；显式给的一律优先）" \
     "  run 目录：$RD" \
     "  serve.log：$SL$([ -f "$SL" ] || printf '  ← ⚠ 这个文件不存在')"

# ---------------------------------------------------------------- ① 元信息
_sec "① run 元信息（**实际起服用的**配置；不是「我以为传了什么」）"
echo "--- serve_cmd.txt ---"
sed -n '1,40p' "$RD/serve_cmd.txt" 2>/dev/null || echo "(无 serve_cmd.txt)"
echo
echo "--- 宿主环境里与本次相关的变量（收集时快照）---"
for v in PLAT OFFLOAD OFFLOAD_GB KV8_SWA KV8_RING_FP16 KV8_FULL ENGRAM ENGRAM_DEVICE_INDEX \
         DRAFT_GRAPH MAX_LEN MAX_SEQS BAT_TOKENS PATCH_MODE P2_POOL_PATCH P2_COMP_JSON \
         L1_POOL_PATCH PROFILE V41_PROFILE DROPCACHE IMAGE NAME DEVS; do
    printf '  %-22s = %s\n' "$v" "$(printenv "$v" 2>/dev/null || echo '<未设>')"
done

# ---------------------------------------------------------------- ② 容器内指纹
_sec "② 容器内指纹（**判据绑内容**：证明「挂的到底是哪一份」）"
_dev=$(grep -m1 -oE "devs='[^']*'" "$RD/serve_cmd.txt" 2>/dev/null | head -1)
_name=$(grep -m1 -oE "run_id=[^ ]*" "$RD/serve_cmd.txt" 2>/dev/null | head -1)
# （CTR 已在参数区按 PLAT 给默认值 —— 这里**不再兜底**，否则会把 a3 的默认值又拉回 a2）
if command -v docker >/dev/null 2>&1 && docker inspect "$CTR" >/dev/null 2>&1; then
    echo "  （容器 $CTR 还在 ⇒ 直接反查）"
    docker exec "$CTR" bash -lc '
      A=/vllm-workspace/vllm-ascend/vllm_ascend
      for f in attention/dsa_v41.py attention/kv8_fuse_triton.py \
               ops/fused_moe/token_dispatcher.py \
               models/deepseek_v41/engram_hash.py models/deepseek_v41/engram_jit_kernel.py \
               models/deepseek_v41/model.py; do
        [ -f "$A/$f" ] && md5sum "$A/$f" || echo "MISSING $A/$f"
      done' 2>/dev/null | sed 's/^/    /'
    echo "  --- 容器内 vllm 侧（offload 相关）---"
    docker exec "$CTR" bash -lc '
      V=/vllm-workspace/vllm/vllm
      for f in distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py \
               distributed/kv_transfer/kv_connector/v1/offloading/config.py \
               v1/kv_offload/cpu/spec.py v1/kv_offload/cpu/pgp_manager.py \
               v1/kv_offload/cpu/p2_pool.py; do
        [ -f "$V/$f" ] && md5sum "$V/$f" || echo "MISSING $V/$f"
      done' 2>/dev/null | sed 's/^/    /'
else
    echo "  （容器 $CTR 不在 ⇒ 跳过；下面 §③ 的 inner.sh 是替代证据）"
    echo "     ★ 若容器其实在跑，核对两件事：① PLAT=$PLAT 对不对（a2=dsv41-a2:8077 / a3=dsv41-a3:8020）"
    echo "                                  ② 或显式覆盖：CTR=<真实容器名> PORT=<真实端口>"
fi

# ---------------------------------------------------------------- ③ inner.sh 真实 env
_sec "③ inner.sh 的关键 env（**容器里真正生效的配置** —— 防「env 没进去」这类静默）"
IN="$RD/inner.sh"
if [ -f "$IN" ]; then
    grep -nE '^\s*export (VLLM_V41|MAX_LEN|MAX_SEQS|BAT_TOKENS|ENGRAM|GRAPH|STATIC_KERNEL|NPUGRAPH_EX|DSPARK|SPEC|MTP|APC|KV8|RING|TP|VLLM_ADMISSION|VLLM_SERVER)' "$IN" \
      | head -60 | sed 's/^/  /'
else
    echo "  (无 inner.sh)"
fi

# ---------------------------------------------------------------- ④ serve.log
_sec "④ serve.log —— 起服门（★ 任一为 0 就该停）   [$SL]"
if [ -f "$SL" ]; then
    echo "  行数：$(wc -l < "$SL")"
    for pat in EH0012 'hdc disconnect' 'DEVICE-INDEX' 'aclrtHostRegister failed' \
               'capture failed' 'EE1016' 'KeyError' 'PAGELESS' 'ENGRAM-PREVTOK-ERR'; do
        printf '  %-28s = %s\n' "$pat" "$(_cnt "$pat" "$SL")"
    done
    printf '  %-28s = %s\n' "P1_pinned ret=0" "$(grep -a 'P1_pinned' "$SL" 2>/dev/null | grep -c 'ret=0' || true)"
    printf '  %-28s = %s\n' "D2_offload 行" "$(_cnt 'D2_offload' "$SL")"

    _sec "④b serve.log —— KV 容量 + SpecDecoding（★ 乱码时 A 值往往异常）"
    grep -a -m3 'GPU KV cache size' "$SL" | sed 's/^/  /'
    grep -a 'SpecDecoding metrics' "$SL" | tail -5 | sed 's/^/  /'

    _sec "④c serve.log —— 卸载计数器（★ CPU_to_GPU 增量 = 真的从 DRAM 取回了）"
    grep -a -oE 'kv_offload_total_bytes_total[^ ]* [0-9.eE+-]+' "$SL" | tail -6 | sed 's/^/  /'
    grep -a -oE 'kv_offload_cpu_cache_usage_perc[^ ]* [0-9.eE+-]+' "$SL" | tail -3 | sed 's/^/  /'
    printf '  %-34s = %s\n' "external_prefix_cache_hits_total 行数" "$(_cnt 'external_prefix_cache_hits_total' "$SL")"
    # ★ 一眼判读：只存不取 vs 双向都动
    _g2c=$(grep -a -oE 'transfer_type="GPU_to_CPU"[^ ]* [0-9.eE+-]+' "$SL" | tail -1 | grep -oE '[0-9.eE+-]+$')
    _c2g=$(grep -a -oE 'transfer_type="CPU_to_GPU"[^ ]* [0-9.eE+-]+' "$SL" | tail -1 | grep -oE '[0-9.eE+-]+$')
    printf '  %-34s = %s\n' "最后一次 GPU_to_CPU（存进 DRAM）" "${_g2c:-<日志里没有>}"
    printf '  %-34s = %s\n' "最后一次 CPU_to_GPU（★ 从 DRAM 取回）" "${_c2g:-<日志里没有>}"
    if [ -n "${_c2g:-}" ] && [ "${_c2g%%.*}" != "0" ]; then
        echo "  ⇒ ★ 有取回发生（CPU_to_GPU 非 0）"
    elif [ -n "${_g2c:-}" ]; then
        echo "  ⇒ ⚠ 只有存、**没有取回**（CPU_to_GPU 缺失或为 0）—— 若你正怀疑卸载，先看这条"
    fi

    _sec "④d serve.log —— 错误与警告（去掉噪声；这是最可能有线索的一段）"
    grep -a -nE 'ERROR|Traceback|RuntimeError|AssertionError|FAIL|WARNING' "$SL" 2>/dev/null \
      | grep -avE 'experimental and subject to change|Unknown vLLM environment variable|Speculative quantization|GPU-specific parameter|NPU Triton causal_conv1d' \
      | tail -40 | sed 's/^/  /'

    _sec "④e serve.log —— 尾部 30 行（原始上下文）"
    tail -30 "$SL" | sed 's/^/  /'

    _sec "④f 你看到的「乱码」在日志里长什么样（找 U+FFFD 替换字符 / 大量 \x）"
    echo "  （只在 server 端打印过输出时才有内容；下面抽含 U+FFFD 或 大量 \\x 的行）"
    LC_ALL=C grep -a -c $'\xef\xbf\xbd' "$SL" 2>/dev/null | sed 's/^/  U+FFFD 行数：/'
    LC_ALL=C grep -a -n $'\xef\xbf\xbd' "$SL" 2>/dev/null | head -10 | sed 's/^/  /'
else
    echo "  (无 serve.log：$SL)"
fi

# ---------------------------------------------------------------- ⑤ metrics / kv_events
_sec "⑤ 引擎 metrics（**现在这一刻**的快照；若容器还在会另存一份原始文件）"
if curl -sf -m 10 "$URL/health" >/dev/null 2>&1; then
    curl -s -m 15 "$URL/metrics" > "$OUTDIR/metrics_now.txt" 2>/dev/null
    echo "  已存：$OUTDIR/metrics_now.txt（$(wc -l < "$OUTDIR/metrics_now.txt" 2>/dev/null || echo 0) 行）"
    grep -E '^vllm:(kv_offload|external_prefix_cache|num_requests)' "$OUTDIR/metrics_now.txt" \
      | head -20 | sed 's/^/  /'
else
    echo "  （$URL/health 不通 ⇒ 服务已停；跳过）"
fi
for f in kv_events.json metrics_after.txt; do
    if [ -f "$RD/$f" ]; then
        cp -f "$RD/$f" "$OUTDIR/" 2>/dev/null && echo "  已复制：$f"
    fi
done

# ---------------------------------------------------------------- ⑥ 文本探针
_sec "⑥ ★ 文本正确性探针（**乱码问题最关键的一段原文**）"
if [ "$NO_PROBE" = "1" ]; then
    echo "  （NO_PROBE=1 ⇒ 跳过）"
elif curl -sf -m 10 "$URL/health" >/dev/null 2>&1; then
    P=$SH/scripts/text_correctness_probe.py
    if [ -f "$P" ]; then
        echo "  跑 $PROBES 发（乱码要看「稳不稳」）……"
        TP=$OUTDIR/textprobe.json
        timeout 900 python3 "$P" --base-url "$URL" --model "${MODEL_NAME:-deepseek-v4-flash}" --mode all \
          --out "$TP" 2>&1 | tail -80 | sed 's/^/  /'
        # ★ 乱码的主判据 = **模型答案的逐字原文**（不是"通过/失败"这种二值）。
        #   probe 的 stdout 可能被 tail 截断，所以再从 JSON 里把答案原文抽出来单独列一遍。
        if [ -f "$TP" ]; then
            echo
            echo "  --- 模型答案原文（从 $TP 抽；★ 乱码就看这一段的 repr）---"
            grep -a -oE '"(answer|text)": .*' "$TP" | head -20 | cut -c1-400 | sed 's/^/  /'
            echo "  --- 乱码指纹（U+FFFD 替换字符 / 控制字符 / 孤立代理对）---"
            printf '      %-24s = %s\n' "U+FFFD(原始字节)" \
                "$(LC_ALL=C grep -a -c $'\xef\xbf\xbd' "$TP" 2>/dev/null || true)"
            printf '      %-24s = %s\n' "U+FFFD(\\ufffd 转义)" \
                "$(LC_ALL=C grep -a -c -- '\\ufffd' "$TP" 2>/dev/null || true)"
            printf '      %-24s = %s\n' "U+0000(\\u0000 转义)" \
                "$(LC_ALL=C grep -a -c -- '\\u0000' "$TP" 2>/dev/null || true)"
            printf '      %-24s = %s\n' "NUL(\\x00 原始字节)" \
                "$(LC_ALL=C grep -a -c $'\x00' "$TP" 2>/dev/null || true)"
            printf '      %-24s = %s\n' "孤立代理对(\\ud8)" \
                "$(LC_ALL=C grep -a -c -- '\\ud8' "$TP" 2>/dev/null || true)"
            echo "  （同上全文已存：$TP —— 贴 EVIDENCE 或直接 scp 这一份都行）"
        fi
    else
        echo "  （缺 $P）"
    fi
else
    echo "  （服务已停 ⇒ 跳过；★ 但要复现乱码**必须在服务活着时**跑这一项）"
fi

_hdr "END —— 把 $OUT 整个贴回来（或 scp/上传）就够了"
} > "$OUT" 2>&1

echo "✅ 已生成：$OUT   （$(wc -l < "$OUT") 行，$(du -h "$OUT" | cut -f1)）"
echo
echo "★ 直接贴这一份就够。它已经包含：run 配置 / 容器内指纹 / inner.sh 真实 env /"
echo "  serve.log 的起服门·A 值·卸载计数·错误·尾部 / metrics 快照 / 文本探针原文。"
echo "★ 若还想要原始文件：  tar -czf $OUTDIR.tar.gz -C $(dirname "$OUTDIR") $(basename "$OUTDIR")"
exit 0
