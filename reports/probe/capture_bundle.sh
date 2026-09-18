#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# 出错时一键取证：把"端到端 I/O + 模型内部状态 + 服务日志尾部"打包到一个目录。
#
# 用法：
#   bash capture_bundle.sh                      # 自动命名
#   bash capture_bundle.sh "长上下文退化-1"      # 带标签
#
# 产出：~/projects/dsv41-release/bundles/<时间>-<标签>/
#   ├── SUMMARY.md          本次取证的概要（人读）
#   ├── sparse-meta/        L1 模型内部状态（JSONL，每 step 每层一行）
#   ├── tensors/            L2 张量快照（若当时开着）
#   ├── serve-tail.log      服务日志尾部（请求级 I/O 在这里）
#   ├── metrics.txt         当前运行指标
#   └── dev-endpoints.txt   dev mode 端点清单
#
# 设计：只读拷贝，不打断服务；对缺失的项宽容（不因一项失败而整体失败）。

set -uo pipefail

PKG=${PKG:-$HOME/projects/dsv41-release}
PORT=${PORT:-8020}
PROBE_DIR=$PKG/probe_capture
LABEL=${1:-capture}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=$PKG/bundles/${STAMP}-${LABEL}
TAIL_LINES=${TAIL_LINES:-4000}

mkdir -p "$OUT"
ok() { printf '  \033[32m✓\033[0m %s\n' "$*"; }
no() { printf '  \033[33m—\033[0m %s\n' "$*"; }

echo "取证目录：$OUT"

# ---------- 1) L1 模型内部状态 ----------
if [ -d "$PROBE_DIR/meta" ] && compgen -G "$PROBE_DIR/meta/*.jsonl" >/dev/null; then
  mkdir -p "$OUT/sparse-meta"
  cp "$PROBE_DIR"/meta/*.jsonl "$OUT/sparse-meta/" 2>/dev/null
  n=$(wc -l "$OUT"/sparse-meta/*.jsonl 2>/dev/null | tail -1 | awk '{print $1}')
  sz=$(du -sh "$OUT/sparse-meta" | cut -f1)
  ok "L1 元数据：$n 行，$sz"
else
  no "L1 元数据：无（插针未触发或未挂载）"
fi

# ---------- 2) L2 张量快照 ----------
if [ -d "$PROBE_DIR/tensors" ] && compgen -G "$PROBE_DIR/tensors/*.pt" >/dev/null; then
  cp -r "$PROBE_DIR/tensors" "$OUT/tensors"
  sz=$(du -sh "$OUT/tensors" | cut -f1)
  n=$(ls "$OUT/tensors"/*.pt 2>/dev/null | wc -l)
  ok "L2 张量：$n 个文件，$sz"
else
  no "L2 张量：无（未开启；见 ENABLE 开关）"
fi

# ---------- 3) 服务日志尾部（端到端 I/O 在这里） ----------
LOG=$(ls -t "$PKG"/results/a2_*/serve.log 2>/dev/null | head -1)
if [ -n "$LOG" ] && [ -f "$LOG" ]; then
  tail -n "$TAIL_LINES" "$LOG" > "$OUT/serve-tail.log"
  ok "服务日志尾部：$TAIL_LINES 行（来自 $LOG）"

  # 顺带抓出请求级统计，方便一眼看有没有退化
  grep "SpecDecoding" "$OUT/serve-tail.log" | tail -20 > "$OUT/spec-decoding-tail.txt" 2>/dev/null
  if [ -s "$OUT/spec-decoding-tail.txt" ]; then
    ok "起草接受率尾迹：spec-decoding-tail.txt"
  fi
else
  no "服务日志：找不到"
fi

# ---------- 4) 运行指标 ----------
if curl -s -m 10 "http://127.0.0.1:$PORT/metrics" -o "$OUT/metrics.txt" 2>/dev/null; then
  ok "运行指标：$(wc -l < "$OUT/metrics.txt") 行"
else
  no "运行指标：取不到（服务没起？）"
fi

# ---------- 5) dev mode 端点清单 ----------
if curl -s -m 10 "http://127.0.0.1:$PORT/openapi.json" 2>/dev/null \
     | python3 -c "import json,sys;print('\n'.join(sorted(json.load(sys.stdin).get('paths',{}))))" \
     > "$OUT/dev-endpoints.txt" 2>/dev/null; then
  ok "端点清单：$(wc -l < "$OUT/dev-endpoints.txt") 个"
else
  no "端点清单：取不到"
fi

# ---------- 6) 概要 ----------
cat > "$OUT/SUMMARY.md" <<EOF
# 取证包 $STAMP — $LABEL

生成时间：$(date '+%Y-%m-%d %H:%M:%S %Z')
服务端口：$PORT

## 包含

| 文件/目录 | 内容 |
|---|---|
| \`sparse-meta/\` | **模型内部状态**（L1）：每个 index-source 层、每个 step 的
  选择结果指纹（\`sel_hash\`/\`cand_hash\`）、越界检查（\`dirty\`） |
| \`tensors/\` | **张量快照**（L2）：selected / candidates / qr 的完整内容（若开启） |
| \`serve-tail.log\` | 服务日志尾部，**含请求级 I/O**（\`--enable-log-requests\`） |
| \`spec-decoding-tail.txt\` | 最近的起草接受率（退化的快速判据） |
| \`metrics.txt\` | vLLM 运行指标 |
| \`dev-endpoints.txt\` | 可用运维端点（dev mode） |

## 立刻能回答的问题

1. 退化发生时，**哪一层**的 \`sel_hash\` 第一次与相邻 step 不一致？
2. \`dirty=true\` 出现过吗？（candidates 越界 = kernel 缺陷的直接证据）
3. 同一 prompt 成功/失败两次的 \`sel_hash\` 差异从哪一层开始？
4. 起草接受率在哪个时间点掉下去？

## 常用检索命令

\`\`\`bash
# 有没有越界（最高优先级信号）
grep -h '"dirty": *true' $OUT/sparse-meta/*.jsonl | head

# 各层选择结果指纹（看是否某层开始抖动）
python3 - <<'PY'
import json, glob, collections
by = collections.defaultdict(list)
for f in glob.glob("$OUT/sparse-meta/*.jsonl"):
    for line in open(f):
        r = json.loads(line)
        by[r["layer"]].append((r["step"], r.get("sel_hash"), r.get("cand_hash")))
for layer in sorted(by):
    hs = [h for _, h, _ in by[layer] if h]
    print(f"L{layer}: {len(hs)} 条, 不同指纹 {len(set(hs))} 个")
PY
\`\`\`
EOF
ok "SUMMARY.md"

echo
echo "完成。总体积：$(du -sh "$OUT" | cut -f1)"
echo "查看：  less $OUT/SUMMARY.md"
echo
echo "提示：L2 张量快照默认关闭。需要时执行："
echo "      echo 'ring:4' > $PROBE_DIR/ENABLE     # 开启（保留最近 4 个 step）"
echo "      bash $0                                 # 复现问题"
echo "      rm $PROBE_DIR/ENABLE                    # 关闭"
