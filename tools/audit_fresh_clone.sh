#!/usr/bin/env bash
# ★ 交付一致性审计：**从全新 clone 出发**，核对交付路径引用的每个文件是否存在。
#
# 为什么需要：本仓已两次出现"工作区改了、git 里没有"的缺口
#   * gmm1 armF 内核（只在 ~/tmp 的临时目录里）
#   * `ENGRAM_WKV_TP` 的实现（`patches/files/model.py` 只有启动器默认、没有代码）
# 两者都**不报错**，只是静默少掉性能/功能。本脚本把"仓库 → 可部署"这条链做一次静态自检。
#
# 用法（在仓库根目录）: bash tools/audit_fresh_clone.sh [ref]
#   默认 ref = HEAD
# 退出码：0 = 全部存在；非 0 = 有缺失项（逐条打印）
set -uo pipefail

REF=${1:-HEAD}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TMP=$(mktemp -d /tmp/audit-clone-XXXXXX)
say() { printf '%s\n' "$*"; }
fail=0

say "== ① 全新 clone（ref=$REF）到 $TMP"
git -C "$REPO" archive "$REF" | tar -x -C "$TMP" || { say "⛔ archive 失败"; exit 2; }
say "   $(find "$TMP" -type f | wc -l) 个文件"

say
say "== ② **实际挂载/执行**的路径是否都在 clone 里"
# 只认两种"真的会被用到"的形式，避免把注释里的路径当依赖（首版就因此报了 20+ 假阳性）：
#   (a) 挂载：      -v "$PREFIX/xxx:..."   （MOUNTS+=(-v "$F/xxx:...")）
#   (b) 执行/判存： bash "$PREFIX/xxx"、[ -f "$PREFIX/xxx" ]
# 前缀按脚本里的赋值解析（$PKG=仓库根、$F=patches/files、$HERE=工具目录…），
# 并对**文档化可选**的项做白名单（PGO 二进制、PROBE 派生件、运行期目录）。
python3 - "$TMP" <<'PY'
import os, re, sys
root = sys.argv[1]
srcs = []
for d in ("scripts", "tools", "deploy"):
    for dp, _, fns in os.walk(os.path.join(root, d)):
        srcs += [os.path.join(dp, fn) for fn in fns if fn.endswith(".sh")]
text = "\n".join(open(f, encoding="utf-8", errors="ignore").read() for f in srcs)

# 前缀 -> 仓库内相对目录
PREFIX = {
    "$PKG": "", "$REPO": "", "$HERE": "tools", "$SELF": "scripts",
    "$F": "patches/files", "$PB": "reports/probe", "$DCPMOUNT": "../dcpw_min",
    "$TRACK": "", "$B": "",
}
# 文档化"不在仓库里"的（不算缺陷）
ALLOW = (
    "optim/pgo/",            # 二进制按 README 明说不入库
    "reports/probe/",        # PROBE 派生件，README 明说不随包
    "cache/", "results/", "logs/", "probe_capture/", "payload/",
)

refs = set()
for m in re.finditer(r'-v\s+"?(\$(?:PKG|REPO|HERE|SELF|F|PB|DCPMOUNT)/[^":]+)', text):
    refs.add(m.group(1))
for m in re.finditer(r'(?:bash|exec\s+bash)\s+"?(\$(?:PKG|REPO|HERE|SELF|F)/[^"\s]+)', text):
    refs.add(m.group(1))
for m in re.finditer(r'\[\s+-[fx]\s+"?(\$(?:PKG|REPO|HERE|SELF|F)/[^"\s]+)', text):
    refs.add(m.group(1))

missing, ok, skipped = [], 0, 0
for r in sorted(refs):
    if "$" in r.replace("$PKG", "").replace("$REPO", "").replace("$HERE", "").replace("$SELF", "").replace("$F", "").replace("$PB", "").replace("$DCPMOUNT", ""):
        skipped += 1
        continue
    rel = r
    for k, v in sorted(PREFIX.items(), key=lambda kv: -len(kv[0])):
        if rel.startswith(k):
            rel = os.path.join(v, rel[len(k):]) if v else rel[len(k):].lstrip("/")
            break
    rel = rel.lstrip("./")
    if rel.startswith(ALLOW) or any(rel.startswith(a) for a in ALLOW):
        skipped += 1
        continue
    if os.path.exists(os.path.join(root, rel)):
        ok += 1
    else:
        missing.append(rel)
print("   解析出 %d 个挂载/执行目标：存在 %d、缺失 %d、跳过（文档化可选）%d"
      % (ok + len(missing), ok, len(missing), skipped))
for m in missing:
    print("     ⛔ 缺:", m)
sys.exit(1 if missing else 0)
PY
[ $? -ne 0 ] && fail=1

say
say "== ③ 交付默认的**代码级**自检（每项必须在 clone 里出现）"
check() { # <文件> <模式> <说明>
  if [ ! -f "$TMP/$1" ]; then say "   ⛔ $1 不存在（$3）"; fail=1; return; fi
  if grep -q -e "$2" "$TMP/$1"; then say "   ✅ $3"; else say "   ⛔ $1 里找不到 [$2]（$3）"; fail=1; fi
}
check patches/files/model.py "_wkv_tp" "engram wkv 分片实现"
check patches/files/model.py "V41_ENGRAM_PAD_SKIP" "engram padded 只清要读的行"
check scripts/serve_a2.sh "ENGRAM_WKV_TP=\${ENGRAM_WKV_TP:-1}" "wkv 分片默认开"
check scripts/serve_a2.sh "PAD_SKIP=\${PAD_SKIP:-1}" "pad-skip 默认开"
check scripts/serve_a2.sh "ENGRAM_DEVICE_INDEX=1" "设备索引默认开"
check scripts/serve_a2.sh "V41_GATE_MAX_PREFILL" "admission gate"
check scripts/serve_a2.sh "ENGRAM_WKV_TP" "哑开关透传（wkv）"
check scripts/serve_a2.sh "FORCE_EPLB" "哑开关透传（force_eplb）"
check scripts/serve_v2.sh "eplb_config" "EPLB 接线"
check tools/build_a3_tp8_image.sh "OPP_SRC" "镜像烘焙自定义内核"
check tools/ab_gate.py "STABLE-BUCKET" "稳定桶判据提醒"
for k in GroupedMatmulSwigluQuantV2_fa3d6d3de6e1f32e170e39d2ddd3a20e.o; do
  if find "$TMP/kernels" -name "$k" | grep -q .; then say "   ✅ gmm1 armF 内核 $k"; else say "   ⛔ 缺内核 $k"; fail=1; fi
done

say
if [ "$fail" = "0" ]; then say "✅ 审计通过：全新 clone 具备交付默认的全部组成"; else say "⛔ 审计失败：见上面 ⛔ 行"; fi
say "   （审计目录保留在 $TMP，便于人工复核）"
exit "$fail"
