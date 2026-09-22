#!/usr/bin/env bash
# =============================================================================
# make_shadow_pkg.sh —— 在**本机**从 dsv41-release 造一个 shadow-pkg
#
# 为什么需要它：`a2/scripts/serve_a2_offload.sh` 依赖 shadow-pkg，
#   而 A2 上**根本没有** shadow-pkg（它只存在于 A3 的开发机上）。
#   ⇒ A2 上线时第一条命令就会卡在 `⚠ 找不到 shadow-pkg`。
#   本脚本把"造 shadow"这一步**可复现化**：不依赖 A3，也不需要手改任何生产脚本。
#
# 它做什么（**只改副本，绝不动 dsv41-release**）：
#   1. 在 $DST 建一个目录树，**除 `scripts/` 外全部软链**到 $PKG（省空间、防漂移）；
#   2. 把 $PKG/scripts/ 拷成真目录；
#   3. 对副本做 **5 处精确锚点插入**（锚点必须**恰好命中一次**，否则 fail-closed 不写）：
#        ① serve_a2.sh：在 `-e LOAD_FORMAT=...` 后加 `-e NPU_OFFLOAD_HOST_MEM`
#        ② serve_a2.sh：在 `[ -n "$PGO_LIB" ] && MOUNTS+=...` 后加**卸载/int8 挂载块**
#        ③ serve_a2.sh：在 `echo "[serve_a2] PATCH_MODE=..."` 后加 `KV_ARGS_EXTRA` 校验
#        ④ serve_a2.sh：在 inner.sh 的 `export V41_KV_TIER=off` 后加 env 透传
#        ⑤ serve_v2.sh：在 `ARGS+=($EXTRA)` 后加 `ARGS+=($KV_ARGS_EXTRA)`
#   4. 自检：把 5 处插进去的行数报出来；任何一处没命中 ⇒ exit 2 且**不落盘**。
#
# 用法（在一台有 dsv41-release 的机器上，含 A2）：
#   PKG=<dsv41-release 路径> DST=<放 shadow 的路径> bash a2/scripts/make_shadow_pkg.sh
#   # 默认 PKG=$(pwd 的 dsv41-release)、DST=$HOME/shadow-pkg
#
# 造完之后：`SHADOW_PKG=$DST bash a2/scripts/serve_a2_offload.sh`
# =============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
A2=$(cd "$HERE/.." && pwd)

PKG=${PKG:-$(cd "$A2/.." 2>/dev/null && pwd)/dsv41-release}
DST=${DST:-$HOME/shadow-pkg}

die() { echo "[make_shadow] ✗ $*" >&2; exit 2; }
say() { echo "[make_shadow] $*"; }

[ -d "$PKG" ] || die "PKG 不存在：$PKG（用 PKG=<dsv41-release 路径> 指定）"
[ -f "$PKG/scripts/serve_a2.sh" ] || die "$PKG/scripts/serve_a2.sh 不存在"
[ -f "$PKG/scripts/serve_v2.sh" ] || die "$PKG/scripts/serve_v2.sh 不存在"

echo "=============================================================="
echo "造 shadow-pkg"
echo "  PKG（只读源）: $PKG"
echo "  DST（产物）  : $DST"
echo "=============================================================="

# ---------------------------------------------------------------- 1. 目录树
if [ -e "$DST" ]; then
    say "DST 已存在 ⇒ 只覆盖 scripts/ 下的两份文件（保留其它内容）"
    mkdir -p "$DST/scripts"
else
    mkdir -p "$DST"
    _n=0
    for e in "$PKG"/* "$PKG"/.[!.]*; do
        [ -e "$e" ] || continue
        b=$(basename "$e")
        # ★ `patches` 必须是**真目录**：serve_a2_offload.sh 会把补丁 cp 进
        #   `$SHADOW/patches/files/offload_dsv41/`，若这里是软链就会**污染 dsv41-release**。
        case "$b" in scripts|patches) continue ;; esac
        ln -s "$e" "$DST/$b" 2>/dev/null && _n=$((_n + 1))
    done
    say "软链了 $_n 项（scripts/ 除外）"
    mkdir -p "$DST/scripts"
    for e in "$PKG"/scripts/*; do
        [ -e "$e" ] || continue
        b=$(basename "$e")
        case "$b" in serve_a2.sh|serve_v2.sh) continue ;; esac
        ln -s "$e" "$DST/scripts/$b" 2>/dev/null
    done
    say "scripts/ 其余项也软链"
    # patches/：真目录 + 逐项软链（`files/` 例外，留成真目录给 serve_a2_offload.sh 写）
    mkdir -p "$DST/patches/files"
    for e in "$PKG"/patches/*; do
        [ -e "$e" ] || continue
        b=$(basename "$e")
        case "$b" in files) continue ;; esac
        ln -s "$e" "$DST/patches/$b" 2>/dev/null
    done
    for e in "$PKG"/patches/files/*; do
        [ -e "$e" ] || continue
        b=$(basename "$e")
        ln -s "$e" "$DST/patches/files/$b" 2>/dev/null
    done
    say "patches/ 建为真目录（files/ 可写；其余逐项软链）"
fi

# ---------------------------------------------------------------- 2. 两处副本
cp "$PKG/scripts/serve_a2.sh" "$DST/scripts/serve_a2.sh"
cp "$PKG/scripts/serve_v2.sh" "$DST/scripts/serve_v2.sh"

# ------------------------------------------------- 3. 生成"要插进去"的 fragment
# ★ 用 python 做精确插入（锚点计数必须 == 1，否则 fail-closed）
python3 - "$DST" <<'PYEOF'
import pathlib, re, sys

dst = pathlib.Path(sys.argv[1])
a2 = dst / "scripts" / "serve_a2.sh"
v2 = dst / "scripts" / "serve_v2.sh"


class Anchor:
    """一次精确插入：锚点整行必须**恰好出现一次**，且（可选）校验行内内容。"""

    def __init__(self, path, anchor, payload, label, after=True, require=None):
        self.path, self.anchor, self.payload = path, anchor, payload
        self.label, self.after, self.require = label, after, require

    def apply(self, text):
        lines = text.splitlines(keepends=True)
        hits = [i for i, ln in enumerate(lines) if ln.rstrip("\n") == self.anchor]
        if len(hits) != 1:
            raise SystemExit(f"✗ [{self.label}] 锚点命中 {len(hits)} 次（必须恰好 1 次）：{self.anchor!r}")
        if self.require and self.require not in lines[hits[0]]:
            raise SystemExit(f"✗ [{self.label}] 锚点行内容不符：{lines[hits[0]]!r}")
        i = hits[0] + (1 if self.after else 0)
        lines[i:i] = [p + "\n" for p in self.payload]
        return "".join(lines)


# ---- ①：容器 run 里补一个 -e（池后端要在**容器**里可见）
MOUNT_ENV = Anchor(
    a2,
    '  -e LOAD_FORMAT="$LOAD_FORMAT" \\',
    ['  -e NPU_OFFLOAD_HOST_MEM="${NPU_OFFLOAD_HOST_MEM:-registered}" \\'],
    "① NPU_OFFLOAD_HOST_MEM",
)

# ---- ②：MOUNTS 末尾追加卸载/int8 挂载块（全部 env 门控、缺文件即 die）
MOUNT_BLOCK = Anchor(
    a2,
    '[ -n "$PGO_LIB" ] && MOUNTS+=(-v "$PKG/optim/pgo/libpython3.12.so.1.0:$PGO_LIB:ro")',
    [
        '',
        '# ---------- [A2-OFFLOAD] 由 a2/scripts/make_shadow_pkg.sh 注入（见 a2/logs/052/054） ----------',
        '# 全部 env 门控；缺文件时**直接 die**（响亮失败，绝不静默降级）。',
        '_A2F=${A2_OFFLOAD_FILES:-$PKG/a2/patches}',
        'if [ "${OFFLOAD_SCHED_PATCH:-0}" = "1" ]; then',
        '  _SCHED="${OFFLOAD_SCHED_FILE:-$_A2F/0001-offload-scheduler.patch.py}"',
        '  [ -f "$_SCHED" ] || die "OFFLOAD_SCHED_PATCH=1 但缺 $_SCHED"',
        '  MOUNTS+=(-v "$_SCHED:/vllm-workspace/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:ro")',
        '  echo "[serve_a2] [A2-OFFLOAD] scheduler.py <- $_SCHED"',
        '  _PGPM="${OFFLOAD_PGP_MANAGER:-$_A2F/0001b-offload-per-group-bpc-manager.patch.py}"',
        '  if [ -f "$_PGPM" ]; then',
        '    MOUNTS+=(-v "$_PGPM:/vllm-workspace/vllm/vllm/v1/kv_offload/cpu/pgp_manager.py:ro")',
        '    echo "[serve_a2] [A2-OFFLOAD] pgp_manager.py <- $_PGPM"',
        '  fi',
        '  _PGPH="${OFFLOAD_PGP_HOOKS:-$_A2F/0001c-offload-per-group-bpc-hooks.patch.py}"',
        '  if [ -f "$_PGPH" ]; then',
        '    MOUNTS+=(-v "$_PGPH:/vllm-workspace/vllm/vllm/v1/kv_offload/cpu/pgp_hooks.py:ro")',
        '    echo "[serve_a2] [A2-OFFLOAD] pgp_hooks.py <- $_PGPH"',
        '  fi',
        'fi',
        'if [ "${OFFLOAD_NPU_WORKER_PATCH:-0}" = "1" ]; then',
        '  _CPU_NPU="${OFFLOAD_CPU_NPU_FILE:-$_A2F/0002-offload-cpu-pool-host-registered.patch.py}"',
        '  [ -f "$_CPU_NPU" ] || die "OFFLOAD_NPU_WORKER_PATCH=1 但缺 $_CPU_NPU"',
        '  MOUNTS+=(-v "$_CPU_NPU:/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/cpu_npu.py:ro")',
        '  echo "[serve_a2] [A2-OFFLOAD] cpu_npu.py <- $_CPU_NPU（NPU_OFFLOAD_HOST_MEM=${NPU_OFFLOAD_HOST_MEM:-registered}）"',
        'fi',
        '# ---------- ★★ 档 C / 档 D（int8 KV）：**7 个整文件挂载件**（见 a2/publish/kv8-int8-pkg/README.md） ----------',
        '# 2026-09-22 15:2x 补：此前发布包里**只有 dsa_v41.py 一个**，A2 根本起不了档 C/D。',
        '# 触发条件 = 开了任一个 int8 开关；缺文件直接 die（不静默降级成档 B）。',
        'if [ "${A2_KV8:-0}" = "1" ] || [ "${A2_KV8_SWA:-0}" = "1" ] || [ "${A2_RING_FP16:-0}" = "1" ]; then',
        '  _I8="${A2_KV8_INT8_PKG:-$PKG/a2/patches/kv8-int8-pkg/vllm_ascend}"',
        '  _DSA="${A2_KV8_DSA:-$PKG/a2/patches/kv8-graphsafe/dsa_v41.py}"',
        '  _V=/vllm-workspace/vllm-ascend/vllm_ascend',
        '  [ -f "$_DSA" ] || die "开了 int8 但缺 $_DSA"',
        '  for _f in core/deepseek_v41.py core/kv_cache_interface.py models/deepseek_v41/model.py \\',
        '            models/deepseek_v41/compressor.py ops/triton/compressor/compressor_triton.py \\',
        '            attention/kv8_prefill_triton.py; do',
        '    [ -f "$_I8/$_f" ] || die "开了 int8 但缺 $_I8/$_f（见 a2/publish/kv8-int8-pkg/README.md）"',
        '    MOUNTS+=(-v "$_I8/$_f:$_V/$_f:ro")',
        '  done',
        '  MOUNTS+=(-v "$_DSA:$_V/attention/dsa_v41.py:ro")',
        '  echo "[serve_a2] [A2-INT8] 已挂 7 个整文件件（6 个来自 $_I8 + dsa_v41.py）"',
        'fi',
    ],
    "② MOUNTS 卸载/int8 块",
)

# ---- ③：KV_ARGS_EXTRA 合法性校验（单引号会破坏 inner.sh 的字面量）
KVARGS_CHECK = Anchor(
    a2,
    '} | tee "$OUT/serve_cmd.txt"',
    [
        '',
        '# [A2-OFFLOAD] KV_ARGS_EXTRA 以单引号字面量嵌进 inner.sh ⇒ 值里不能有单引号（空格必须保留）。',
        'if [ -n "${KV_ARGS_EXTRA:-}" ]; then',
        '  case "$KV_ARGS_EXTRA" in',
        '    *"\'"*) die "[A2-OFFLOAD] KV_ARGS_EXTRA 含单引号，拒绝注入";;',
        '  esac',
        '  say "[A2-OFFLOAD] KV_ARGS_EXTRA=$KV_ARGS_EXTRA"',
        'fi',
    ],
    "③ KV_ARGS_EXTRA 校验",
)

# ---- ④：inner.sh 里透传 env（★ 这一段在**宿主**上展开成字面量，容器里是普通 export）
INNER_ENV = Anchor(
    a2,
    'export V41_KV_TIER=off',
    [
        '# [A2-OFFLOAD] 以下在**宿主**上展开成字面量，容器里只是普通 export。',
        "export KV_ARGS_EXTRA='${KV_ARGS_EXTRA:-}'",
        "export VLLM_V41_KV8='${A2_KV8:-0}'",
        "export VLLM_V41_KV8_SWA='${A2_KV8_SWA:-0}'",
        "export VLLM_V41_RING_FP16='${A2_RING_FP16:-0}'",
        "export VLLM_V41_KV8_PREFILL='${A2_KV8_PREFILL:-0}'",
        "export VLLM_V41_APC_ALIGN='${A2_APC_ALIGN:-0}'",
        "export VLLM_V41_KV8_GRAPH_SAFE='${A2_GRAPH_SAFE:-0}'",
        "export P2_POOL_PATCH='${P2_POOL_PATCH:-0}'",
        "export P2_WORKER_ROWS='${P2_WORKER_ROWS:-0}'",
        "export P2_COMP_JSON='${P2_COMP_JSON:-}'",
        "export P2_STRUCT_LOG='${P2_STRUCT_LOG:-1}'",
        "export P2_POOL_LOG='${P2_POOL_LOG:-1}'",
        "export PGP_MGR_HARDEN='${PGP_MGR_HARDEN:-0}'",
        "export PGP_MGR_STATS='${PGP_MGR_STATS:-0}'",
    ],
    "④ inner.sh env 透传",
)

# ---- ⑤：serve_v2.sh 把 KV_ARGS_EXTRA 变成 vllm 的 CLI 参数
V2_ARGS = Anchor(
    v2,
    '[ -n "$EXTRA" ] && ARGS+=($EXTRA)',
    [
        '',
        '# [A2-OFFLOAD] KV 卸载（--kv-transfer-config）/ 池大小 / KV 事件。',
        '# 与上一行同款展开（因此**值里不能有单引号**，见 serve_a2.sh 的校验）。',
        'if [ -n "${KV_ARGS_EXTRA:-}" ]; then',
        '  echo "[serve-v2] [A2-OFFLOAD] KV_ARGS_EXTRA=$KV_ARGS_EXTRA"',
        '  ARGS+=($KV_ARGS_EXTRA)',
        'fi',
    ],
    "⑤ serve_v2 ARGS",
)

plan = [
    ("scripts/serve_a2.sh", [MOUNT_ENV, MOUNT_BLOCK, KVARGS_CHECK, INNER_ENV]),
    ("scripts/serve_v2.sh", [V2_ARGS]),
]

for rel, anchors in plan:
    p = dst / rel
    text = p.read_text()
    before = len(text.splitlines())
    for a in anchors:
        text = a.apply(text)
    p.write_text(text)
    after = len(text.splitlines())
    print(f"[make_shadow] ✓ {rel}: {before} → {after} 行（+{after - before}），插入 {len(anchors)} 处")
    for a in anchors:
        print(f"                 · {a.label}")
PYEOF
rc=$?
if [ "$rc" != 0 ]; then
    die "插入失败（**没有**留下半成品：请删掉 $DST/scripts 下的两个 .sh 后重试）"
fi

chmod +x "$DST/scripts/serve_a2.sh" "$DST/scripts/serve_v2.sh"

# ---------------------------------------------------------------- 4. 自检
echo "-------------------------------------------------------------"
_ok=1
check() {  # check <文件> <必须出现的字面量> <说明>
    if grep -qF "$2" "$1"; then
        printf '✓  %s\n' "$3"
    else
        printf '✗  %s（缺：%s）\n' "$3" "$2"
        _ok=0
    fi
}
check "$DST/scripts/serve_a2.sh" 'A2-OFFLOAD'                       "①/②/③/④ 块已插入"
check "$DST/scripts/serve_a2.sh" 'NPU_OFFLOAD_HOST_MEM="${NPU_OFFLOAD_HOST_MEM' "-e NPU_OFFLOAD_HOST_MEM 已插入"
check "$DST/scripts/serve_a2.sh" "export VLLM_V41_KV8_GRAPH_SAFE='\${A2_GRAPH_SAFE:-0}'" "int8 env 透传已插入"
check "$DST/scripts/serve_v2.sh" 'ARGS+=($KV_ARGS_EXTRA)'           "⑤ serve_v2 已认 KV_ARGS_EXTRA"

if [ "$_ok" != 1 ]; then
    die "自检未通过 ⇒ 产物不可用"
fi

echo "-------------------------------------------------------------"
cat <<EOF
✓ shadow-pkg 造好了：$DST

下一步（在 A2 上）：
  export SHADOW_PKG=$DST
  MODEL=<模型目录> OFFLOAD_GB=56 MAX_LEN=131072 MAX_SEQS=16 \\
  NPU_OFFLOAD_HOST_MEM=registered \\
    bash $A2/scripts/serve_a2_offload.sh

★ 干跑检查（不起服务）：
  DRY=1 SHADOW_PKG=$DST MODEL=<模型目录> bash $A2/scripts/serve_a2_offload.sh
EOF
