#!/usr/bin/env bash
# 从仓库组装 docker build 用的 payload 树。
#
# 这是 **payload 清单的唯一实现**：镜像层 patch 包里装的东西，
# 就是把这里产出的树 `COPY` 进镜像。清单的说明见同目录 PAYLOAD.md。
#
# 用法：
#   bash deploy/a3-ced-pd/build_payload.sh [输出目录]
#   默认输出到 deploy/a3-ced-pd/payload/
#
# 产出：
#   payload/ascend/      → /tmp/bake/ascend/     （vllm-ascend 补丁件，13 个）
#   payload/ced/         → /tmp/bake/ced/        （CED 件，2 个）
#   payload/draft/       → /tmp/bake/draft/      （draft 版 3 个，baked+DRAFT_GRAPH=1 用）
#   payload/patches/     → /tmp/bake/patches/    （4 个 .patch）
#   payload/scripts/     → /tmp/bake/scripts/    （起服脚本）
#   payload/PAYLOAD.sha256                        （每个文件的 sha256，供跨机核对）
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/../.." && pwd)"
OUT="${1:-$HERE/payload}"

say() { printf '\033[1m[payload]\033[0m %s\n' "$*"; }

# 缺任何一个都必须**响亮失败**，不能静默少装一个文件（少装=起服后才发现）。
need() { [ -f "$PKG/$1" ] || { echo "[payload] FATAL: 缺 $PKG/$1" >&2; exit 21; }; }

say "仓库根 = $PKG"
say "输出   = $OUT"

rm -rf "$OUT"
mkdir -p "$OUT"/{ascend,ced,draft,patches,scripts}

# ---- A. vllm-ascend 补丁件（13）----
A=(
  engram_hbm.py engram_hash.py engram_gate.py engram_jit_kernel.py
  engram_plan_kernel.py engram_device_index.py engram_graph.py
  model.py indexer.py ascend_forward_context.py rope_dsv4.py block_table.py
)
for f in "${A[@]}"; do
  need "patches/files/$f"
  cp -f "$PKG/patches/files/$f" "$OUT/ascend/$f"
done
# token_dispatcher 要改名（仓库里带 moemask 后缀，容器内叫 token_dispatcher.py）
need "patches/files/token_dispatcher_moemask.py"
cp -f "$PKG/patches/files/token_dispatcher_moemask.py" "$OUT/ascend/token_dispatcher.py"

# ---- B. CED 件（2）----
need "experimental/ced/mooncake_hybrid_connector.py"
need "experimental/ced/dsa_v41.py"
cp -f "$PKG/experimental/ced/mooncake_hybrid_connector.py" "$OUT/ced/"
cp -f "$PKG/experimental/ced/dsa_v41.py"                  "$OUT/ced/"

# ---- C. draft 版（3）----
for f in dsa_v1.py dspark_proposer.py llm_base_proposer.py; do
  need "patches/files/draft/$f"
  cp -f "$PKG/patches/files/draft/$f" "$OUT/draft/$f"
done

# ---- D. vLLM core 补丁（4）----
for p in admission_gate.patch; do
  need "patches/$p"; cp -f "$PKG/patches/$p" "$OUT/patches/$p"
done
for p in core_scheduler_replay.patch core_scheduler_prefill_hit.patch core_model_runner_prompt_tail.patch; do
  need "experimental/ced/$p"; cp -f "$PKG/experimental/ced/$p" "$OUT/patches/$p"
done

# ---- E. 起服脚本（8）----
for s in serve_a2.sh serve_v2.sh serve_a3.sh serve_a3_pd.sh \
         serve_a3_pd_proxy.sh serve_a3_ced_pd.sh serve_a3_ced_single.sh run_test.sh; do
  need "scripts/$s"; cp -f "$PKG/scripts/$s" "$OUT/scripts/$s"
done
chmod +x "$OUT"/scripts/*.sh

# ---- F. 清单（跨机核对用）----
# ⚠️ 必须**先删掉可能的旧清单**再生成：清单自己要放在 payload/ 里，
# 如果生成时它已存在就会被算进去（第一版把空文件的 sha256 e3b0c442… 也算了一条）。
rm -f "$OUT/PAYLOAD.sha256"
( cd "$OUT" && find . -type f ! -name PAYLOAD.sha256 -printf '%P\n' | sort \
    | xargs sha256sum ) > "$OUT/PAYLOAD.sha256"

n=$(find "$OUT" -type f ! -name PAYLOAD.sha256 | wc -l)
sz=$(du -sh "$OUT" | cut -f1)
say "完成：$n 个文件，$sz"
say "清单：$OUT/PAYLOAD.sha256"
sed 's/^/    /' "$OUT/PAYLOAD.sha256" | head -6
echo "    ..."
