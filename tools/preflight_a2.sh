#!/usr/bin/env bash
# =============================================================================
# preflight_a2.sh —— 起服前自检（**30 秒，不起服务容器**）
#
#   MODEL=/path/to/model bash tools/preflight_a2.sh
#
# ## 为什么需要它
#
# v5 的教训：**10 个 bug 里有 9 个是"跑到最后一步才暴露"**，而起服要 5–25 分钟。
# 每次白等一轮，代价被放大十几倍。这个脚本把已知的 11 类 A2 环境差异 + 包内
# 文件完整性，压缩到 30 秒内一次性回答。
#
# 分三档：
#   FATAL  起服必然失败（或缺文件导致测试必然失败）—— 必须修
#   WARN   能跑但结果会失真/降级（如 PGO 静默降级、绑核过窄）
#   INFO   只记录，便于和 A3 对照
#
# 退出码：0 = 无 FATAL；1 = 有 FATAL
# SKIP_DOCKER=1 可跳过需要 docker 的检查（默认在有镜像时做）
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
cd "$PKG"

MODEL=${MODEL:-}
IMAGE=${IMAGE:-dsv41-a2:v8}

FATAL=0
WARN=0

hr()   { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mOK\033[0m     %s\n' "$*"; }
warn() { WARN=$((WARN+1)); printf '  \033[33mWARN\033[0m   %s\n' "$*"; }
bad()  { FATAL=$((FATAL+1)); printf '  \033[31mFATAL\033[0m  %s\n' "$*"; }
info() { printf '  info   %s\n' "$*"; }
fix()  { printf '         -> %s\n' "$*"; }

printf '\033[1m[preflight] %s  (pkg=%s)\033[0m\n' "$(date '+%F %T')" "$PKG"

# =============================================================================
hr "1/9 包内文件完整性"
# =============================================================================
# v5 的两个致命缺文件。这里**显式断言**，因为它们的失败发生在"跑到那一步"时。
for f in \
  "tests/t_quote.sh" \
  "tests/t_vision.py" \
  "tests/t_gsm8k.py" \
  "tests/acc_eval.py" \
  "tests/make_report.sh" \
  "tests/vision_accuracy_check.py" \
  "tests/p15_stream_curve_filefiller.py" \
  "tests/multibatch/multibatch_gate.py" \
  "tools/check_model_dir.sh" \
  "tools/model_mount_args.sh" \
  "scripts/build_image.sh" \
  "scripts/serve_a2.sh" \
  "scripts/run_test.sh" \
  "data/hongloumeng.txt" \
  "data/suffix_quote.txt" ; do
  if [ -f "$f" ]; then ok "$f"; else
    bad "缺文件 $f"
    case "$f" in
      tests/vision_accuracy_check.py) fix "cp tools/vision_accuracy_check.py tests/" ;;
      tests/p15_stream_curve_filefiller.py) fix "从 COS 下载后放 tests/（v5 曾漏打包）" ;;
    esac
  fi
done

# 上面两个文件即使"存在但不能 import"也等于缺
for f in tests/vision_accuracy_check.py tests/p15_stream_curve_filefiller.py \
         tests/acc_eval.py tests/multibatch/multibatch_gate.py; do
  [ -f "$f" ] || continue
  python3 -c "import ast,sys; ast.parse(open(sys.argv[1],encoding='utf-8').read())" "$f" 2>/dev/null \
    || bad "$f 语法错误（存在但不可用）"
done

# =============================================================================
hr "2/9 Dockerfile 续行链"
# =============================================================================
if [ -f Dockerfile ] && [ -f tools/check_dockerfile.py ]; then
  if python3 tools/check_dockerfile.py Dockerfile >/dev/null 2>&1; then
    ok "Dockerfile 续行链合法（v3/v4 曾因行内 # + 漏 \\ 而构建不出来）"
  else
    bad "Dockerfile 续行链有问题（会导致 unknown instruction）"
    python3 tools/check_dockerfile.py Dockerfile 2>&1 | grep -E "ERROR" | head -5 | sed 's/^/         /'
  fi
else
  warn "缺 Dockerfile 或 tools/check_dockerfile.py，跳过"
fi

# 镜像 tag 三处一致（v5 曾 build 产 v4 / serve 找 v5）
_tb=$(grep -oE 'IMAGE_TAG:-dsv41-a2:v[0-9]+' scripts/build_image.sh 2>/dev/null | head -1 | sed 's/.*://')
_ts=$(grep -oE 'IMAGE:-dsv41-a2:v[0-9]+'    scripts/serve_a2.sh  2>/dev/null | head -1 | sed 's/.*://')
_tr=$(grep -oE 'IMAGE:-dsv41-a2:v[0-9]+'    scripts/run_test.sh  2>/dev/null | head -1 | sed 's/.*://')
if [ -n "$_tb" ] && [ "$_tb" = "$_ts" ] && [ "$_ts" = "$_tr" ]; then
  ok "镜像 tag 三处一致：$_tb"
else
  bad "镜像 tag 不一致：build=$_tb serve=$_ts run_test=$_tr"
  fix "sed -i 's/dsv41-a2:vX/dsv41-a2:vY/g' scripts/*.sh Dockerfile"
fi

# =============================================================================
hr "3/9 模型目录（软链链 / 必需文件 / 分片）"
# =============================================================================
if [ -z "$MODEL" ]; then
  warn "未设置 MODEL，跳过模型检查"
elif [ ! -d "$MODEL" ]; then
  bad "MODEL 不是目录：$MODEL"
else
  if [ -f tools/model_mount_args.sh ]; then
    _out=$(mktemp); _err=$(mktemp)
    if bash tools/model_mount_args.sh "$MODEL" >"$_out" 2>"$_err"; then
      _n=$(grep -c . "$_out" || echo 0)
      ok "软链链健康：需要挂 $_n 个目录（v5 已修"只挂一层会悬空"）"
      _hops=$(grep -oE '软链跳数分布: .*' "$_err" | head -1 || true)
      [ -n "$_hops" ] && info "$_hops"
      info "挂载清单（前 6 个）："
      head -6 "$_out" | sed 's/^/           /'
      [ "$_n" -gt 6 ] && info "  … 共 $_n 个"
    else
      bad "模型目录存在悬空软链 ⇒ 起服必然失败"
      grep -E 'FATAL|->' "$_err" | head -6 | sed 's/^/         /'
      fix "装配脚本必须写绝对路径软链，且源目录（L1）不能移动/删除"
    fi
    rm -f "$_out" "$_err"
  fi
  # 必需文件（走软链读，能读到才算数）
  for f in config.json quant_model_weights.safetensors.index.json; do
    [ -e "$MODEL/$f" ] && ok "$f" || bad "缺 $f"
  done
  # 分片 / 可选件
  _v=$(ls "$MODEL"/vision-*.safetensors 2>/dev/null | wc -l)
  _m=$(ls "$MODEL"/mtpq-*.safetensors 2>/dev/null | wc -l)
  _e=$([ -e "$MODEL/engram_int8" ] && echo 1 || echo 0)
  _q=$([ -e "$MODEL/optional/quarot.safetensors" ] && echo 1 || echo 0)
  [ "$_v" -ge 1 ] && ok "vision 分片 $_v 个（缺则视觉测试跳过）" || warn "无 vision 分片"
  [ "$_m" -ge 1 ] && ok "mtpq 分片 $_m 个（省 2.42 GB/rank，KV 靠它过 3Mi）" \
                  || warn "无 mtpq 分片 ⇒ 会用 BF16 draft，KV 可能 < 3Mi"
  [ "$_e" = 1 ] && ok "engram_int8 存在（206 GB，常驻 DRAM）" || warn "无 engram_int8"
  [ "$_q" = 1 ] && ok "optional/quarot.safetensors 存在" \
               || warn "无 quarot ⇒ 视觉可能 10/23 而非 23/23"
fi

# =============================================================================
hr "4/9 官方目录（视觉图片 + GSM8K chat 模板）"
# =============================================================================
OD=${OFFICIAL_DIR:-}
if [ -z "$OD" ]; then
  for c in "$HOME/models/DeepSeek-V4.1-Flash" /home/*/models/DeepSeek-V4.1-Flash; do
    [ -d "$c" ] && { OD="$c"; break; }
  done
fi
if [ -n "$OD" ] && [ -d "$OD" ]; then
  ok "官方目录：$OD"
  if [ -d "$OD/inference/examples/images" ]; then
    _i=$(ls "$OD"/inference/examples/images/*.jpe?g 2>/dev/null | wc -l)
    ok "视觉图片目录：$_i 张（至少 2 张才能跑 23 例）"
  else
    bad "缺 $OD/inference/examples/images ⇒ 视觉测试会跳过"
  fi
  if [ -d "$OD/encoding" ]; then ok "$OD/encoding（GSM8K chat 模板）"
  else bad "缺 $OD/encoding ⇒ GSM8K 会跳过"; fi
else
  bad "找不到官方 checkpoint 目录（OFFICIAL_DIR）"
  fix "OFFICIAL_DIR=/path/to/DeepSeek-V4.1-Flash，或放到 ~/models/DeepSeek-V4.1-Flash"
fi

# =============================================================================
hr "5/9 宿主测试依赖（datasets / PYHOST）"
# =============================================================================
PYHOST_=${PYHOST:-}
if [ -z "$PYHOST_" ]; then
  for c in "$HOME/venvs/lmeval311/bin/python" "$HOME/venvs/lmeval/bin/python" \
           /home/*/venvs/lmeval311/bin/python "$(command -v python3)"; do
    [ -x "$c" ] && { PYHOST_="$c"; break; }
  done
fi
ok "PYHOST 会选：$PYHOST_"
if "$PYHOST_" -c "import datasets, sys; print(datasets.__version__)" >/tmp/_ds.$$ 2>&1; then
  _dv=$(cat /tmp/_ds.$$)
  if [ "$_dv" = "5.0.1" ]; then
    ok "datasets $_dv（与预置 GSM8K 缓存兼容）"
  else
    warn "datasets $_dv ≠ 5.0.1：预置缓存可能不兼容 ⇒ GSM8K 会失败"
    fix "pip install 'datasets==5.0.1'（用 $PYHOST_）"
  fi
else
  bad "PYHOST 的解释器没有 datasets ⇒ GSM8K 必然失败"
  fix "$PYHOST_ -m pip install 'datasets==5.0.1'"
  fix "或 export PYHOST=<带 datasets 的解释器>（本包支持用 PYHOST 覆盖）"
fi
rm -f /tmp/_ds.$$
# HF 缓存
if [ -d "$HOME/.cache/huggingface/datasets/openai___gsm8k" ]; then
  _sh=$(find "$HOME/.cache/huggingface/datasets/openai___gsm8k" -name '*.arrow' 2>/dev/null | wc -l)
  ok "GSM8K HF 缓存在位（$_sh 个 arrow）"
else
  warn "无 ~/.cache/huggingface/datasets/openai___gsm8k ⇒ 离线 GSM8K 会失败"
  fix "解包 a2_testdata.tar.gz 后 cp -a hf_datasets/openai___gsm8k ~/.cache/huggingface/datasets/"
fi

# =============================================================================
hr "6/9 CPU / 线程 / 绑核（A2 与 A3 的核心差异）"
# =============================================================================
_nproc=$(nproc)
_phys=$(lscpu -p=CPU,CORE,SOCKET 2>/dev/null | grep -v '^#' | awk -F, '{print $2}' | sort -u | wc -l)
info "逻辑 CPU：$_nproc   物理核：${_phys:-未知}"
info "OMP_NUM_THREADS=${OMP_NUM_THREADS:-<未设>}（A3 镜像自带 1；未设时 torch 会用可见核数）"
if [ -n "${OMP_NUM_THREADS:-}" ]; then
  ok "OMP_NUM_THREADS 已设（与 A3 一致）"
else
  warn "OMP_NUM_THREADS 未设 ⇒ torch 可能按可见核数起线程 × 8 个 worker = 超订"
  fix "export OMP_NUM_THREADS=1（A3 实测配置）"
fi
_ml=$(ulimit -l 2>/dev/null || echo '?')
info "memlock (ulimit -l) = $_ml   （A2 需要大值；本包已默认 --ulimit memlock=-1）"

# NUMA / NPU 拓扑：决定 detect_numa 会绑到哪些核
if command -v npu-smi >/dev/null 2>&1; then
  echo "  info   NPU 在各 NUMA 节点的分布："
  for d in 0 1 2 3 4 5 6 7; do
    b=$(npu-smi info 2>/dev/null | awk -v c="$d" '$2==c' \
        | grep -oE '[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9]' | head -1)
    if [ -n "$b" ]; then
      n=$(cat "/sys/bus/pci/devices/${b,,}/numa_node" 2>/dev/null || echo '?')
      cp=$(cat "/sys/devices/system/node/node${n}/cpulist" 2>/dev/null || echo '?')
      printf '           chip%-2s numa=%-3s cpus=%s\n' "$d" "$n" "$cp"
    fi
  done
  _nodes=$(for d in 0 1 2 3 4 5 6 7; do
    b=$(npu-smi info 2>/dev/null | awk -v c="$d" '$2==c' \
        | grep -oE '[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9]' | head -1)
    [ -n "$b" ] && cat "/sys/bus/pci/devices/${b,,}/numa_node" 2>/dev/null
  done | grep -E '^[0-9]+$' | sort -un | tr '\n' ' ')
  info "detect_numa 会选出节点：${_nodes:-<无>}"
  _cpus=""
  for n in $_nodes; do _c="$(cat /sys/devices/system/node/node$n/cpulist 2>/dev/null)"; _cpus="${_cpus:+$_cpus,}$_c"; done
  if [ -n "$_cpus" ]; then
    _cnt=$(echo "$_cpus" | tr ',' '\n' | awk -F- '{s+=(NF>1?$2-$1+1:1)} END{print s}')
    info "默认 CPUSET=auto 会绑到：$_cpus  （共 ${_cnt} 个逻辑 CPU）"
    if [ "${_cnt:-0}" -lt 32 ]; then
      warn "绑核仅 ${_cnt} 个逻辑 CPU（A3 是 320）⇒ host 侧可能成为瓶颈"
      fix "想让 vLLM 自己按 NUMA 精细绑（含 NPU 中断）：CPUSET=-1 MEMS=-1 CPU_BIND=1"
    fi
  fi
else
  warn "找不到 npu-smi，跳过 NPU/NUMA 拓扑"
fi

# =============================================================================
hr "7/9 编译缓存（skcache / PGO）"
# =============================================================================
_skc="$PKG/cache/skcache/compile_outputs"
if [ -d "$_skc/static_kernel_cache" ]; then
  _cf=$(ls "$_skc"/static_kernel_cache/*.json 2>/dev/null | wc -l)
  ok "static_kernel_cache/ 命中（$_cf 个缓存文件）"
  ls "$_skc"/static_kernel_cache/ 2>/dev/null | head -3 | sed 's/^/           /'
  info "缓存键 = CANN-<版本>_<SoC>；换 SoC/CANN 会自动重编，换 driver 不会"
else
  info "无 static_kernel_cache/ ⇒ 本次要冷编译（A2 首次 15–20 min，一次性）"
fi
_ts=$(find "$_skc" -maxdepth 1 -type d -name 'ts*_outputs' 2>/dev/null | wc -l)
if [ "${_ts:-0}" -gt 0 ]; then
  warn "有 $_ts 个 ts*_outputs 临时目录（旧版从不清理，A3 上曾攒到 1491 个/847MB）"
  fix "本包起服时会自动 GC；也可手动：find $_skc -maxdepth 1 -type d -name 'ts*_outputs' -delete"
fi

if [ -f optim/pgo/libpython3.12.so.1.0 ]; then
  _md5=$(md5sum optim/pgo/libpython3.12.so.1.0 | cut -d' ' -f1)
  info "PGO libpython md5 = $_md5（参考 f1ebbee1405d0e31136aa4480b57b3dc）"
  if [ -s optim/pgo/TARGET_PATH.txt ]; then
    ok "TARGET_PATH.txt = $(cat optim/pgo/TARGET_PATH.txt)"
  else
    warn "TARGET_PATH.txt 缺失/为空（旧版会因此**静默降级** PYTHON_PGO=0）"
    fix "本包起服时会自动探测并落盘；也可先跑一次 scripts/build_image.sh"
  fi
else
  warn "无 PGO 产物 ⇒ PYTHON_PGO 会降级为 0（失去 ~4.4% 服务侧收益）"
fi

# =============================================================================
hr "8/9 主机资源（DRAM / 磁盘 / 残留容器）"
# =============================================================================
if [ -r /proc/meminfo ]; then
  _av=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
  info "MemAvailable ≈ ${_av} GiB"
  # Engram int8 常驻 DRAM 206 GB + 权重 page cache；A2 是 754 GiB 总量
  if [ "${_av:-0}" -lt 250 ]; then
    warn "可用内存 ${_av} GiB < 250 ⇒ Engram（206 GB 常驻）可能吃紧"
    fix "sync && echo 3 | sudo tee /proc/sys/vm/drop_caches 清 page cache 后再起服"
  else
    ok "可用内存充足（Engram 需 ~206 GiB 常驻）"
  fi
fi
_du=$(df -BG --output=avail "$PKG" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$_du" ]; then
  if [ "$_du" -lt 20 ]; then warn "包所在盘可用 ${_du} GB < 20（缓存要 ~1.7 GB，首次编译更多）"
  else ok "磁盘可用 ${_du} GB"; fi
fi
if command -v docker >/dev/null 2>&1 || sudo -n docker info >/dev/null 2>&1; then
  DOCKER="docker"; docker info >/dev/null 2>&1 || DOCKER="sudo -n docker"
  _left=$($DOCKER ps -a --format '{{.Names}}' 2>/dev/null | grep -x 'dsv41-a2' || true)
  if [ -n "$_left" ]; then
    bad "残留容器 dsv41-a2 存在（旧版失败不清理 ⇒ 悬挂 ~313 GB，导致后续起服叠加失败）"
    fix "$DOCKER rm -f dsv41-a2"
  else
    ok "无残留 dsv41-a2 容器"
  fi
  if $DOCKER image inspect "$IMAGE" >/dev/null 2>&1; then
    ok "镜像 $IMAGE 存在"
  else
    # [RETAG-HINT 已撤销] 这里曾写死："v6 的 patches/ + Dockerfile + optim/ 与 v5
    # 逐字节相同 ⇒ 可直接 `docker tag dsv41-a2:v5 dsv41-a2:v6`，省 10–20 min"。
    # 那个结论**只对 v6 成立**：v8 的镜像内容与 v5/v6 不同（新增 engram_device_index.py /
    # engram_graph.py，改 model.py / engram_hbm.py），照提示 retag 会得到一个
    # 「名字叫 v8、内容却是 v6」的镜像 —— 属发布事故。而且 tag 名本身无法证明内容等价，
    # 唯一可靠的判据是逐文件 md5（build_image.sh 第 4 步会做）。
    # ⇒ 现在只提示"重建"，不再提供任何 retag 捷径。
    warn "镜像 $IMAGE 不存在 ⇒ 先跑 bash scripts/build_image.sh 烘焙（约 10–20 min）"
    fix "若确实想沿用旧镜像，请**显式指定并自行确认内容**：IMAGE=<已有 tag> bash scripts/run_test.sh"
  fi
else
  warn "docker 不可用，跳过容器检查"
fi

# =============================================================================
hr "9/9 结论"
# =============================================================================
printf '  FATAL=%d  WARN=%d\n' "$FATAL" "$WARN"
if [ "$FATAL" -gt 0 ]; then
  printf '\n\033[31m[preflight] 有 %d 个 FATAL —— 先修再起服（否则必失败或测试必失败）\033[0m\n' "$FATAL"
  exit 1
fi
if [ "$WARN" -gt 0 ]; then
  printf '\n\033[33m[preflight] 可以起服，但有 %d 个 WARN（见上，可能让结果失真）\033[0m\n' "$WARN"
else
  printf '\n\033[32m[preflight] 全部通过 ✅\033[0m\n'
fi
printf '  下一步：MODEL=%s MODE=full bash scripts/run_test.sh\n\n' "${MODEL:-<模型目录>}"
exit 0
