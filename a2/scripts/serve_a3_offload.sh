#!/usr/bin/env bash
# =============================================================================
# serve_a3_offload.sh —— A3（8×910C）DRAM KV 卸载一键起服
#
# ★ 它与 `serve_a2_offload.sh` **共用同一份脚本体**（本文件只有 20 行）：
#   把 `PLAT=a3` 传下去，其余全部由 `serve_a2_offload.sh` 的平台块给默认值。
#
#   为什么不复制一份脚本：
#     A2/A3 的**引擎、优化开关、挂载件、判据完全同源**（`scripts/serve_a3.sh` 自己就是
#     `exec bash serve_a2.sh`）⇒ 复制一份 = 以后必然两处分叉。本仓已经栽过好几次
#     "同一个东西两份副本、改了一份"（见 a2/AGENTS.md 的探针纪律）。
#
#   PLAT=a3 会把这些默认值换掉（**照 A2 抄在 A3 上是错的，其中 DEVS 是危险的**）：
#     DEVS        : 0 1 2 3 4 5 6 7  →  **8 9 10 11 12 13 14 15**（★ 0–7 不是我们的，别抢）
#     IMAGE       : dsv41-a2:v9       →  官方 quay…:deepseek-v4.1-flash-a3（A3 上没有 v9）
#     PATCH_MODE  : baked             →  **mount**（官方镜像里没有我们的补丁 ⇒ baked 会**静默零补丁**）
#     起服入口     : serve_a2.sh       →  **serve_a3.sh**（它多一道**选卡/占用校验**）
#     NAME/PORT   : dsv41-a2 / 8077   →  dsv41-a3 / 8020
#     PYTHON_PGO  : 1                 →  0（A2 的 PGO 产物与 A3 镜像的 libpython md5 不同）
#     DROPCACHE   : 1                 →  **0**（A3 是共用机，清整机 page cache 会打到别人）
#
# 用法（在 A3 上；**先看哪些卡空着**）：
#   bash tools/list_chips.sh                       # 只读，打印每张卡的占用与属主
#   MODEL=/path/to/model bash a2/scripts/serve_a3_offload.sh                  # 真起服
#   MODEL=/path/to/model DRY=1 bash a2/scripts/serve_a3_offload.sh            # 干跑（不改任何东西）
#   DEVS="8 9 10 11 12 13 14 15" MODEL=… bash a2/scripts/serve_a3_offload.sh  # 显式指定卡
#   MODEL=… PROFILE=1 bash a2/scripts/serve_a3_offload.sh                     # 开 profiler
#
# ⚠️ 前置（否则会在起服前就被拦，**这是设计**）：
#   ① shadow-pkg 要用**当前版**生成器重造（它得认 `A2_*` 与 `L1_POOL_PATCH`）：
#        PKG=$(pwd) DST=$HOME/projects/dsv41-upstream-pr/shadow-pkg bash a2/scripts/make_shadow_pkg.sh
#   ② `DEVS` 里的卡必须空闲（`serve_a3.sh` 会查；确有自己残留进程时 `ALLOW_BUSY=1`）
# =============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 唯一的"平台差异"就在这一行 —— 其余全部复用。
export PLAT=a3
exec bash "$HERE/serve_a2_offload.sh" "$@"
