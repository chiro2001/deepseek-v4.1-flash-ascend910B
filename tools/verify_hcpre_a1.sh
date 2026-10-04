#!/usr/bin/env bash
# 校验 HcPre A1 内核**确实被执行**（不只是文件被换了）。
#
# 为什么要这道关：`V41_HC_OPP_PKG` 只把 vendor 文件拷进容器；static kernel 缓存
# （cache/skcache/compile_outputs）的 key **不含 .o 内容**，不清缓存就会静默复用旧内核。
# R5 正是因此把 A1 误判为"无效"。判据分两层：
#   L1（文件）容器内 HcPre 的 .o md5 == 期望值
#   L2（执行）采一份小 profile，看 HcPre 的 `aic_mac_time` 是否落在 A1 的预期档
#      （A1：M≤64 时 kL0Size 32→128，L0 轮数 32→8 ⇒ 该列应明显下移）
#
# 用法: verify_hcpre_a1.sh <port> <run_dir> <期望 .o md5>
set -uo pipefail

PORT=${1:-19210}
RUN=${2:?run 目录（含 prof）}
WANT_MD5=${3:-de790a5080da18a3a9302fa3a348a091}
IMG=${IMG:-quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3}
NAME=${NAME:-dsv41-tp8k5}
HOST=http://127.0.0.1:$PORT
REPO=$HOME/cedpd-repo

say() { echo "[verify-A1] $*"; }

say "L1 文件校验（容器内 HcPre .o md5）"
got=$(docker exec "$NAME" bash -lc "md5sum /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910_93/hc_pre/HcPre_6c9f0fa710253eff38b1c0e6be6a67be.o" 2>/dev/null | awk '{print $1}')
say "  容器=$got  期望=$WANT_MD5"
if [ "$got" != "$WANT_MD5" ]; then
  say "  ⛔ L1 不通过：内核文件不是 A1（可能是缓存未清/挂载失败）"
  exit 1
fi
say "  ✅ L1 通过"

say "L2 执行校验：采一份小 profile…"
curl -s -o /dev/null -m 10 -XPOST "$HOST/start_profile" || { say "start_profile 失败"; exit 1; }
python3 "$REPO/tools/bench_concurrency.py" --base-url "$HOST" \
  --concurrency 1 --prompt-tokens 1024 --output-tokens 64 --repeats 1 --spec-tokens 5 \
  > "$HOME/tmp/a1_verify_load.log" 2>&1 || true
curl -s -o /dev/null -m 600 -XPOST "$HOST/stop_profile" || true
say "  载荷完成，等导出（约 1–2 min）"

# 导出 rank0 最新捕获
D=$REPO/results/$RUN/prof
target=$(ls -dt "$D"/dp0_pp0_tp0_* 2>/dev/null | head -1)
[ -n "${target:-}" ] || { say "找不到 prof 子目录"; exit 1; }
cat > /tmp/vA1_$$.py <<'PY'
import sys
import torch_npu  # noqa
from torch_npu.profiler.profiler import analyse
analyse(sys.argv[1], max_process_number=16)
print("ANALYSE_DONE")
PY
docker run --rm -v "$D":/pf -v /tmp/vA1_$$.py:/a.py:ro "$IMG" \
  bash -lc "python3 /a.py /pf/$(basename "$target")" > "$HOME/tmp/a1_verify_export.log" 2>&1 || true
sudo -n chmod -R a+rX "$D" 2>/dev/null || true

csv=$target/ASCEND_PROFILER_OUTPUT/kernel_details.csv
if [ ! -f "$csv" ]; then
  say "  ⛔ 导出未生成 CSV，见 $HOME/tmp/a1_verify_export.log"
  exit 1
fi
say "L2 HcPre 的资源指纹："
python3 - "$csv" <<'PY'
import sys
import pandas as pd
df = pd.read_csv(sys.argv[1], low_memory=False)
ren = {"Name": "Op Name", "Type": "OP Type", "Duration(us)": "Task Duration(us)"}
df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
h = df[df["OP Type"] == "HcPre"]
if h.empty:
    print("  没抓到 HcPre")
else:
    for c in ("Task Duration(us)", "aic_mac_time(us)", "aic_scalar_time(us)", "aiv_time(us)"):
        if c in h.columns:
            v = pd.to_numeric(h[c], errors="coerce").dropna()
            if len(v):
                print(f"  {c:22s} p50={v.median():9.3f}  n={len(v)}")
    print(f"  样本 kL0 期望：A1 档位下 aic_mac_time 应低于 stock（stock M≤64 时 L0 轮数=32）")
PY
say "完成（把上面的值与 stock 基线比：stock 的 HcPre aic_mac_time 约 1.28 µs）"
exit 0
