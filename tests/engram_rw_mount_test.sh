#!/usr/bin/env bash
# =============================================================================
# engram_rw_mount_test.sh —— engram 表目录挂载权限的**离线**回归测试
#
#   bash tests/engram_rw_mount_test.sh
#
# 为什么要有它：v8 起 serve 脚本要求 engram 表目录**可写**挂载
# （aclrtHostRegister 只接受可写映射，只读 VMA → ret=507899），而 A2 真机上
# 报错发生在**容器内、起服中途**，没有 A2 就没法验证。这里用**假的软链模型树**
# + `DRY_RUN=1`（只解析开关、只打印挂载清单，不碰 docker / 不占卡 / 不需要 NPU）
# 把三条挂载路径全覆盖：
#   * auto（默认，逐目录）
#   * ancestor（只挂公共祖先；祖先必须保持 :ro，engram 单独叠加 :rw）
#   * 单层 fallback（MODEL_MOUNT_MODE=none / 找不到 tools/model_mount_args.sh）
# 外加：真实目录形态（engram_int8 是实体目录，旧代码这里**会漏**）、
#       物理落盘目录在**另一棵目录树**（A3 真机就是这样）、
#       ENGRAM_DEVICE_INDEX=0（显式关闭 ⇒ 不该有任何 :rw）、
#       宿主不可写 / 模型缺 engram 目录（必须**响亮失败**并给出修法）。
#
# 判据：最后一行 `[engram-rw] pass=N fail=0`；退出码非 0 = 有断言失败。
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
SERVE="$PKG/scripts/serve_a2.sh"

pass=0; fail=0
ok()   { pass=$((pass+1)); printf '  \033[32mPASS\033[0m  %s\n' "$*"; }
bad()  { fail=$((fail+1)); printf '  \033[31mFAIL\033[0m  %s\n' "$*"; }
hr()   { printf '\n\033[1m--- %s\033[0m\n' "$*"; }
note() { printf '        %s\n' "$*"; }

WORK=$(mktemp -d)
trap 'chmod -R u+w "$WORK" 2>/dev/null; rm -rf "$WORK"' EXIT

# 语法先过一遍（这一类改动最容易在 heredoc/引号上翻车）
bash -n "$SERVE" || { echo "[engram-rw] FATAL: $SERVE 语法不过"; exit 2; }

# ---------------------------------------------------------------------------
# 假模型树：完整复刻 A3 真机的**软链链条**（L5→L4→L3→物理目录）
#
#   $R/raw/engram-int8/                         ← 真正落盘的文件（A3 上在
#   $R/models/out/v41-…-engram-dr-vision/        另一棵目录树里）
#        config.json（实体） + engram_int8/（实体目录，里面是 4 个软链）
#   $R/models/out/v41-…-engram-dr-vision-mtpq/    ← L4：engram_int8 -> L3 的
#   $R/models/out/v41-…-qrot-mtpq/               ← L5 = MODEL：engram_int8 -> L4 的
#
# $2 = wide ：物理目录在**另一棵树**（触发 ancestor 覆盖检查 → 退回 auto）
# $2 = narrow：所有目录是同一个父目录下的兄弟（ancestor 模式可用）
# ---------------------------------------------------------------------------
make_tree() {
  local R="$1" kind="${2:-wide}" OUT RAW
  if [ "$kind" = "narrow" ]; then
    OUT="$R/out"; RAW="$R/out/engram-int8"          # 与三层模型目录**同级** ⇒ ancestor 可用
  else
    OUT="$R/models/out"; RAW="$R/raw/engram-int8"   # 另一棵树 ⇒ ancestor 覆盖不到（A3 真机）
  fi
  mkdir -p "$RAW" "$OUT"
  : > "$RAW/layers_1_engram_embed.weight.safetensors"
  : > "$RAW/layers_1_engram_embed.scale.safetensors"
  : > "$RAW/layers_14_engram_embed.weight.safetensors"
  : > "$RAW/layers_14_engram_embed.scale.safetensors"

  local L3="$OUT/v41-w4a8-engram-dr-vision"
  local L4="$OUT/v41-w4a8-engram-dr-vision-mtpq"
  local L5="$OUT/v41-w4a8-engram-dr-vision-qrot-mtpq"
  mkdir -p "$L3/engram_int8" "$L4" "$L5"
  printf '{"text_config":{"engram_layer_ids":[1,14]}}\n' > "$L3/config.json"
  printf '{}\n' > "$L3/quant_model_weights.safetensors.index.json"
  local f
  for f in layers_1_engram_embed.weight layers_1_engram_embed.scale \
           layers_14_engram_embed.weight layers_14_engram_embed.scale; do
    ln -sfn "$RAW/$f.safetensors" "$L3/engram_int8/$f.safetensors"
  done
  ln -sfn "$L3/config.json" "$L4/config.json"
  ln -sfn "$L3/quant_model_weights.safetensors.index.json" "$L4/quant_model_weights.safetensors.index.json"
  ln -sfn "$L3/engram_int8" "$L4/engram_int8"
  ln -sfn "$L4/config.json" "$L5/config.json"
  ln -sfn "$L4/quant_model_weights.safetensors.index.json" "$L5/quant_model_weights.safetensors.index.json"
  ln -sfn "$L4/engram_int8" "$L5/engram_int8"
  printf '%s\n' "$L5"
}

# 假模型树：engram_int8 是**实体目录**（旧代码的 `case ${_d##*/}` 在这里漏掉，
# 因为 model_mount_args.sh 只列软链目标，不列 MODEL 下的实体子目录）
make_tree_realdir() {
  local R="$1" M
  M="$R/out/model-dir"
  mkdir -p "$M/engram_int8"
  printf '{"text_config":{"engram_layer_ids":[1,14]}}\n' > "$M/config.json"
  : > "$M/quant_model_weights.safetensors.index.json"
  : > "$M/engram_int8/layers_1_engram_embed.weight.safetensors"
  : > "$M/engram_int8/layers_1_engram_embed.scale.safetensors"
  printf '%s\n' "$M"
}

# dry-run 一次：$1=MODEL；其余参数是环境赋值
dry() {
  local _m="$1"; shift
  env DRY_RUN=1 MODEL="$_m" OUT_DRYRUN_DIR="$WORK/dryout" "$@" bash "$SERVE" 2>&1
}

has()  { printf '%s' "$1" | grep -qF -- "$2"; }
hasnt(){ ! printf '%s' "$1" | grep -qF -- "$2"; }

# ===========================================================================
hr "1) auto（默认）：软链链条 + 物理目录在另一棵树 —— 全部 engram 目录必须 :rw"
# ===========================================================================
M1=$(make_tree "$WORK/t1" wide)
OUT1=$(dry "$M1" MODEL_MOUNT_MODE=auto); rc1=$?
if [ "$rc1" -eq 0 ] && has "$OUT1" "[a2-dry] OK"; then ok "auto 模式 dry-run 成功（rc=0）"; else
  bad "auto 模式 dry-run 失败（rc=$rc1）"; printf '%s\n' "$OUT1" | tail -5 | sed 's/^/        /'; fi
has "$OUT1" "-v $M1:$M1:ro" \
  && ok "模型根目录仍是 :ro（只放开 engram 那几层）" \
  || bad "模型根目录不是 :ro（改宽了？）"
has "$OUT1" "-v $M1/engram_int8:$M1/engram_int8:rw" \
  && ok "MODEL/engram_int8（软链，容器里被 open 的路径）= :rw" \
  || bad "MODEL/engram_int8 不是 :rw"
has "$OUT1" "-v $WORK/t1/raw/engram-int8:$WORK/t1/raw/engram-int8:rw" \
  && ok "物理落盘目录（另一棵树）= :rw" \
  || bad "物理落盘目录不是 :rw ⇒ O_RDWR 会 EROFS"
hasnt "$OUT1" "-v $M1:$M1:rw" \
  && ok "整棵模型目录没有被开成 :rw（没有照抄用户的 workaround）" \
  || bad "整棵模型目录被开成 :rw（过于宽泛）"
has "$OUT1" "ver=" && ok "dry-run 打印脚本版本/指纹（用户报障时用得上）" || bad "dry-run 没打印脚本版本"

# ===========================================================================
hr "2) ancestor：祖先必须 :ro，engram 目录**叠加** :rw（覆盖检查通过的分支）"
# ===========================================================================
M2=$(make_tree "$WORK/t2" narrow)
OUT2=$(dry "$M2" MODEL_MOUNT_MODE=ancestor); rc2=$?
ANC2="$WORK/t2/out"
[ "$rc2" -eq 0 ] && ok "ancestor 模式 dry-run 成功（rc=0）" || bad "ancestor 模式 dry-run 失败（rc=$rc2）"
has "$OUT2" "ancestor 模式" && ok "确实走了 ancestor 分支（祖先 $ANC2 覆盖全部目录）" \
  || bad "没有走 ancestor 分支"
has "$OUT2" "-v $ANC2:$ANC2:ro" && ok "祖先本身是 :ro" || bad "祖先不是 :ro"
has "$OUT2" "-v $M2/engram_int8:$M2/engram_int8:rw" \
  && ok "MODEL/engram_int8 = :rw（嵌套叠加，只放开这一层）" \
  || bad "MODEL/engram_int8 不是 :rw"
has "$OUT2" "-v $WORK/t2/out/engram-int8:$WORK/t2/out/engram-int8:rw" \
  && ok "物理落盘目录 = :rw" || bad "物理落盘目录不是 :rw"
hasnt "$OUT2" "-v $ANC2:$ANC2:rw" && ok "祖先没有被改成 :rw" || bad "祖先被改成 :rw（过于宽泛）"
hasnt "$OUT2" "-v $M2:$M2:ro" && ok "ancestor 模式没有退化成逐目录挂载" || bad "ancestor 模式退化了"

# ===========================================================================
hr "3) ancestor + 模型树横跨两棵目录树（A3 真机布局）⇒ 必须退回 auto 而不是静默半坏"
# ===========================================================================
OUT3=$(dry "$M1" MODEL_MOUNT_MODE=ancestor); rc3=$?
[ "$rc3" -eq 0 ] && ok "dry-run 成功（rc=0）" || bad "dry-run 失败（rc=$rc3）"
has "$OUT3" "覆盖不到" && ok "覆盖检查生效：打印'覆盖不到 … 退回 auto'" \
  || bad "没有触发覆盖检查（老代码会挂一个覆盖不到的祖先，容器里软链全悬空）"
has "$OUT3" "-v $WORK/t1/raw/engram-int8:$WORK/t1/raw/engram-int8:rw" \
  && ok "退回 auto 后物理落盘目录仍是 :rw" || bad "退回 auto 后 engram 挂载不对"

# ===========================================================================
hr "4) 单层 fallback（MODEL_MOUNT_MODE=none）：MODEL :ro + engram 叠加 :rw"
# ===========================================================================
OUT4=$(dry "$M1" MODEL_MOUNT_MODE=none); rc4=$?
[ "$rc4" -eq 0 ] && ok "none 模式 dry-run 成功（rc=0）" || bad "none 模式 dry-run 失败（rc=$rc4）"
has "$OUT4" "-v $M1:$M1:ro" && ok "只挂 MODEL 一层且为 :ro（旧行为，仅用于复现）" || bad "MODEL 挂载不对"
has "$OUT4" "-v $M1/engram_int8:$M1/engram_int8:rw" && ok "engram 表目录叠加 :rw" || bad "engram 没有叠加 :rw"
has "$OUT4" "-v $WORK/t1/raw/engram-int8:$WORK/t1/raw/engram-int8:rw" \
  && ok "物理落盘目录也叠加 :rw" || bad "物理落盘目录没叠加 :rw"

# ===========================================================================
hr "5) 单层 fallback 的另一半：tools/model_mount_args.sh 缺失"
# ===========================================================================
FAKE="$WORK/fakepkg"
mkdir -p "$FAKE/scripts" "$FAKE/tools"
cp "$SERVE" "$FAKE/scripts/serve_a2.sh"
OUT5=$(env DRY_RUN=1 MODEL="$M1" OUT_DRYRUN_DIR="$WORK/dryout" bash "$FAKE/scripts/serve_a2.sh" 2>&1); rc5=$?
[ "$rc5" -eq 0 ] && ok "缺工具时 dry-run 成功（rc=0）" || bad "缺工具时 dry-run 失败（rc=$rc5）"
has "$OUT5" "找不到 tools/model_mount_args.sh" && ok "打印了'找不到工具'告警" || bad "没有告警"
has "$OUT5" "-v $M1/engram_int8:$M1/engram_int8:rw" \
  && ok "缺工具路径也把 engram 目录叠加成 :rw（用户报障的第 3 条路径）" \
  || bad "缺工具路径没有叠加 :rw"

# ===========================================================================
# ⚠️ 标题里不要用反引号：双引号内的反引号会被 shell 做命令替换（本文件第一版就
#    在这里报 "unexpected end of file from `case`"）。
hr '6) engram_int8 是**实体目录**（旧代码按 basename 匹配时漏掉的形态）'
# ===========================================================================
M6=$(make_tree_realdir "$WORK/t6")
OUT6=$(dry "$M6" MODEL_MOUNT_MODE=auto); rc6=$?
[ "$rc6" -eq 0 ] && ok "dry-run 成功（rc=0）" || bad "dry-run 失败（rc=$rc6）"
has "$OUT6" "-v $M6:$M6:ro" && ok "MODEL = :ro" || bad "MODEL 不是 :ro"
has "$OUT6" "-v $M6/engram_int8:$M6/engram_int8:rw" \
  && ok "实体 engram_int8 被单独挂成 :rw（旧代码这里会漏 ⇒ ret=507899）" \
  || bad "实体 engram_int8 没有被挂成 :rw"

# ===========================================================================
hr "7) ENGRAM_DEVICE_INDEX=0（显式关闭）：不该出现任何 engram :rw"
# ===========================================================================
OUT7=$(dry "$M1" ENGRAM_DEVICE_INDEX=0); rc7=$?
[ "$rc7" -eq 0 ] && ok "dry-run 成功（rc=0）" || bad "dry-run 失败（rc=$rc7）"
hasnt "$OUT7" "engram_int8:rw" && hasnt "$OUT7" "engram-int8:rw" \
  && ok "显式 0 ⇒ engram 目录保持 :ro（不无谓放宽权限）" \
  || bad "显式 0 时仍出现了 engram :rw"

# ===========================================================================
hr "8) 宿主上不可写 ⇒ **只告警不拦**（容器以 root 跑，A3 真机就是 root:root 0600）"
# ===========================================================================
if [ "$(id -u)" = "0" ]; then
  note "以 root 运行：access(W_OK) 对 root 恒真，跳过这一条（在 A2 上用普通用户跑就有意义）"
else
  M8=$(make_tree_realdir "$WORK/t8")
  chmod 555 "$M8/engram_int8"
  OUT8=$(dry "$M8" MODEL_MOUNT_MODE=auto); rc8=$?
  chmod 755 "$M8/engram_int8" 2>/dev/null || true
  # 为什么**不能**在这里 die：A3 真机上 engram 表是 root:root 0600，跑脚本的普通用户
  # [ -w ] 为假 —— 但容器是以 root 起的，rw 挂载下 O_RDWR 完全正常。拿它当硬判据是
  # 假阳性，会把一个能跑的部署拦死（本修复的第一版就在 A3 上被自己拦住了）。
  [ "$rc8" -eq 0 ] && ok "宿主不可写**不**拦起服（rc=0，与 A3 真机形态一致）" \
    || bad "宿主不可写被误判为致命（rc=$rc8）—— A3 正常部署会被拦死"
  has "$OUT8" "WARNING: 宿主上" && ok "但仍然响亮的告警（点名路径）" || bad "没有告警，用户会失去线索"
  has "$OUT8" "ENGRAM_DEVICE_INDEX=0" && ok "告警里给出非 root 容器的修法" || bad "告警里没给修法"
  has "$OUT8" "-v $M8/engram_int8:$M8/engram_int8:rw" \
    && ok "挂载仍是 :rw（真正决定 O_RDWR 成败的是挂载模式）" || bad "挂载不是 :rw"
fi

# ===========================================================================
hr "9) config 声明了 engram 但模型目录里没有 engram_int8/ ⇒ 必须 die"
# ===========================================================================
M9="$WORK/t9/model-noengram"; mkdir -p "$M9"
printf '{"text_config":{"engram_layer_ids":[1,14]}}\n' > "$M9/config.json"
OUT9=$(dry "$M9" MODEL_MOUNT_MODE=auto); rc9=$?
[ "$rc9" -ne 0 ] && ok "缺 engram 表 ⇒ 退出码非 0（rc=$rc9）" || bad "缺表却仍然通过（自检失效！）"
has "$OUT9" "声明了 engram_layer_ids" && ok "报错点名 config 声明与目录缺失" || bad "报错没点明原因"

printf '\n[engram-rw] pass=%d fail=%d\n' "$pass" "$fail"
[ "$fail" -eq 0 ] || exit 1
exit 0
