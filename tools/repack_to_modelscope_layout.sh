#!/usr/bin/env bash
# =============================================================================
# repack_to_modelscope_layout.sh —— 把"软链农场"形态的量化产物整理成
# **扁平、自包含、与 ModelScope 发布形态一致**的单目录。
#
# 为什么需要它
# ------------
# DP8 量化流水线产出的最终目录（L5 叶子，如
# `v41-w4a8-engram-dr-vision-qrot-mtpq/`）是一个**软链农场**：
# ~94 个软链 + 1 个实体文件，跨 5 层、跨多棵目录树
# （`v41-w4a8-stage1/` 主干、`v41-w4a8-dspark/` + `-mtpq/` MTP、
#  `v41-w4a8-engram-int8/` Engram 表、`v41-w4a8-*-vision/` 视觉 …）。
#
# 三个真实代价（都是踩过的）：
#   1. 起服必须把**每一层**都挂进容器，只挂 L5 会全部悬空（No such file）；
#   2. 任何一层被移动/改名，L5 立刻半坏，且**不报错**（软链悬空是静默的）；
#   3. 与 ModelScope 上发布的形态**不一致** —— 别人下载后跑
#      `engram_int8/reassemble_engram_weights.sh`，得到的是一个**扁平单目录**；
#      而本地是软链农场，两边路径/权限/实验现象无法对齐。
#
# 本脚本把 L5 整理成一个**自包含**目录：每个条目都是**实体文件**
# （默认**硬链接**，同一文件系统下 0 额外空间、瞬时完成）。
#
# 关于 engram 分片：ModelScope 把两个 ~98 GB 的 weight 切成 6×16 GiB
# （`part-XX-of-06`），那是**单文件 50 GB 上传上限**的妥协，**不是加载器要求**
# —— 加载器要的是拼回来的单文件。所以本脚本默认产出**单文件**
# （= 别人 reassemble 之后的样子，也是能直接起服的样子）；
# 要往 ModelScope 传时再加 `--emit-parts` 生成分片。
#
# 用法
# ----
#   SRC=/home/user/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq
#   DST=/home/user/models/out/v41-w4a8-flat
#
#   bash repack_to_modelscope_layout.sh --src "$SRC" --dst "$DST"                # 计划
#   bash repack_to_modelscope_layout.sh --src "$SRC" --dst "$DST" --apply        # 执行
#   bash repack_to_modelscope_layout.sh --src "$SRC" --dst "$DST" --apply --emit-parts
#
# 选项
#   --src DIR      源目录（L5 叶子）必填
#   --dst DIR      目标目录（扁平自包含）必填，必须为空或不存在
#   --apply        真正执行（不写 = 只打印计划）
#   --copy         用复制而不是硬链接（跨文件系统时默认报错，退出码 3）
#   --emit-parts   额外把 engram 两个大 weight 切成 16 GiB 片并写 PARTS.sha256
#   --part-size N  分片大小（默认 16G，与已发布形态一致）
#   --readme FILE  额外放入 README.md（模型卡）
#   --no-verify    跳过最后的一致性核对（不建议）
#
# 退出码：0 成功；2 参数/预检失败；3 硬链接跨文件系统且未指定 --copy
# =============================================================================
set -uo pipefail

SRC=""; DST=""; APPLY=0; USE_COPY=0; EMIT_PARTS=0
PART_SIZE="16G"; README_SRC=""; DO_VERIFY=1

die() { printf '\n[repack][FAIL] %s\n' "$*" >&2; exit 2; }
say() { printf '\n[repack] %s\n' "$*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --src)        SRC=${2:-}; shift 2 ;;
    --dst)        DST=${2:-}; shift 2 ;;
    --apply)      APPLY=1; shift ;;
    --copy)       USE_COPY=1; shift ;;
    --emit-parts) EMIT_PARTS=1; shift ;;
    --part-size)  PART_SIZE=${2:-}; shift 2 ;;
    --readme)     README_SRC=${2:-}; shift 2 ;;
    --no-verify)  DO_VERIFY=0; shift ;;
    -h|--help)    sed -n '2,52p' "$0"; exit 0 ;;
    *)            die "未知参数：$1（用 --help 看用法）" ;;
  esac
done

[ -n "$SRC" ] || die "缺少 --src（L5 叶子目录）"
[ -n "$DST" ] || die "缺少 --dst（目标扁平目录）"
[ -L "$SRC" ] && die "--src 本身是软链，请给出实体目录：$SRC"
[ -d "$SRC" ] || die "--src 不是目录：$SRC"
SRC=$(cd "$SRC" && pwd -P) || die "无法解析 --src"

case "$DST" in
  "$SRC"|"$SRC"/*) die "--dst 不能在 --src 里面（会自包含递归）" ;;
esac
if [ -e "$DST" ]; then
  [ -d "$DST" ] || die "--dst 已存在且不是目录：$DST"
  [ -z "$(ls -A "$DST" 2>/dev/null)" ] || die "--dst 已存在且非空：$DST
       本脚本只往空目录/新目录里写，避免与已有内容混淆。"
fi

say "源目录：$SRC"
say "目标目录：$DST"
if [ "$APPLY" = "1" ]; then
  if [ "$USE_COPY" = "1" ]; then say "模式：APPLY（复制）"; else say "模式：APPLY（硬链接）"; fi
else
  say "模式：dry-run（只打印计划；加 --apply 才真正执行）"
fi

# ---------------------------------------------------------------- 扫描
declare -a NAMES=() REALPATHS=() KINDS=()
dangling=0

scan_one() {
  local entry="$1" rel rp kind
  rel=${entry#"$SRC"/}
  # 一律解析到实体路径；硬链接只能直接建在实体上，用软链路径建会误判
  rp=$(readlink -f "$entry" 2>/dev/null)
  if [ -z "$rp" ] || [ ! -f "$rp" ]; then
    printf '  [悬空/非文件] %s\n' "$rel" >&2
    dangling=$((dangling+1)); return 0
  fi
  if [ -L "$entry" ]; then kind=link; else kind=file; fi
  NAMES+=("$rel"); REALPATHS+=("$rp"); KINDS+=("$kind")
}

# 用 `find -L` 跟随软链：这样"指向目录的软链"（engram_int8 / optional）
# 会被展开成目录、其内容以 `<dir>/<file>` 的**逻辑路径**列出，
# 而 `readlink -f` 再给实体路径 —— 两者都拿到，才能既保持 L5 的命名、
# 又建出真硬链接。（早期版本只看 -f，把这类目录软链误判成悬空。）
while IFS= read -r -d '' e; do
  [ -d "$e" ] && continue          # 目录（含软链指向的目录）跳过，内容会被逐文件列出
  scan_one "$e"
done < <(find -L "$SRC" -mindepth 1 -print0 2>/dev/null | sort -z)

n=${#NAMES[@]}
[ "$n" -gt 0 ] || die "源目录里没扫到任何文件：$SRC"
[ "$dangling" -eq 0 ] || die "有 $dangling 个悬空软链 —— 先修软链再整理（否则产物是坏的）"

uniq_n=$(printf '%s\n' "${REALPATHS[@]}" | sort -u | wc -l)
# ⚠️ 口径提醒：`du` 会把**重复 inode 各算一次**（同一个实体被多个软链指向时，
# 或者硬链接被 du 遍历到时），所以下面这个数是"**按逻辑条目累计的字节数**"，
# 不是"独占磁盘占用"。真正决定空间的是：硬链接 ⇒ 追加 0；--copy ⇒ 追加这么多。
total_bytes=$(printf '%s\n' "${REALPATHS[@]}" | sort -u | xargs -r -d '\n' stat -c '%s' 2>/dev/null | awk '{s+=$1} END{print s+0}')
human() { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1 B"; }
total_h=$(human "$total_bytes")
[ -n "$total_h" ] || total_h="?"

say "扫描结果：条目 $n 个（去重后实体 $uniq_n 个，按条目累计约 ${total_h}）"
printf '  %-56s %-6s %s\n' "条目（相对路径）" "类型" "实体来源"
shown=0
for ((i=0;i<n;i++)); do
  printf '  %-56s %-6s %s\n' "${NAMES[$i]}" "${KINDS[$i]}" "${REALPATHS[$i]}"
  shown=$((shown+1))
  [ "$shown" -ge 40 ] && break
done
[ "$n" -gt 40 ] && printf '  …（共 %d 条，上面只列前 40 条）\n' "$n"

# ---------------------------------------------------------------- 关键文件核对
say "关键文件核对"
missing=0
has_name() { local k="$1" x; for x in "${NAMES[@]}"; do [ "$x" = "$k" ] && return 0; done; return 1; }
for k in config.json quant_model_weights.safetensors.index.json engram_extra.safetensors; do
  if has_name "$k"; then printf '  OK      %s\n' "$k"
  else printf '  缺失    %s\n' "$k" >&2; missing=$((missing+1)); fi
done
if grep -q "engram_layer_ids" "$SRC/config.json" 2>/dev/null; then
  hit=0
  for x in "${NAMES[@]}"; do case "$x" in engram_int8/*) hit=1; break ;; esac; done
  if [ "$hit" = "1" ]; then printf '  OK      engram_int8/*（config 声明了 engram_layer_ids）\n'
  else printf '  缺失    engram_int8/*（config 声明了 engram_layer_ids！）\n' >&2; missing=$((missing+1)); fi
else
  printf '  info    config.json 未声明 engram_layer_ids ⇒ 不要求 engram_int8/\n'
fi
[ "$missing" -eq 0 ] || die "关键文件缺失 $missing 项 —— 整理出来也起不了服"

# ---------------------------------------------------------------- 文件系统
DF_PARENT="$DST"
while [ ! -d "$DF_PARENT" ] && [ "$DF_PARENT" != "/" ]; do DF_PARENT=$(dirname "$DF_PARENT"); done
DEST_FS=$(df -P "$DF_PARENT" 2>/dev/null | awk 'NR==2{print $1}')
SRC_FS=$(df -P "$SRC" 2>/dev/null | awk 'NR==2{print $1}')
say "文件系统：源=$SRC_FS 目标=$DEST_FS"
CROSS=0; [ "$SRC_FS" = "$DEST_FS" ] || CROSS=1
if [ "$CROSS" = "1" ] && [ "$USE_COPY" = "0" ]; then
  say "⚠️  源与目标不在同一文件系统 ⇒ 硬链接不可用。"
  say "    推荐把 --dst 放到 $SRC_FS 上（0 额外空间）；或显式加 --copy（需约 ${total_h}）。"
  [ "$APPLY" = "1" ] && exit 3
fi
[ "$CROSS" = "1" ] && say "⚠️  将使用复制（需约 ${total_h} 可用空间）"

# ---------------------------------------------------------------- 硬链接权限预检
# Linux 默认 fs.protected_hardlinks=1：不许给"不属于自己"的文件建硬链接。
# 量化产物大多是 root:root，于是 ln 会失败 —— 早期版本在这里一次性报 94 条
# FAIL，很难定位。这里提前判掉，并给出可直接照做的两条路。
if [ "$CROSS" = "0" ] && [ "$USE_COPY" = "0" ]; then
  myuid=$(id -u)
  prot=$(cat /proc/sys/fs/protected_hardlinks 2>/dev/null || echo 0)
  notmine=0
  for ((i=0;i<n;i++)); do
    ou=$(stat -c '%u' "${REALPATHS[$i]}" 2>/dev/null)
    [ "$ou" = "$myuid" ] || notmine=$((notmine+1))
  done
  if [ "$myuid" != "0" ] && [ "$prot" = "1" ] && [ "$notmine" -gt 0 ]; then
    say "硬链接权限预检不通过："
    say "  当前 uid=$myuid，但有 $notmine/$n 个源文件不属于你，且 fs.protected_hardlinks=1"
    say "  ⇒ 对这些文件建硬链接会被内核拒绝（即使目标目录你可写）。"
    say ""
    say "  两条可行路径（二选一）："
    say "    A) 用 sudo 跑（推荐：0 额外空间、瞬时）"
    say "       sudo bash $0 --src $SRC --dst $DST --apply"
    say "    B) 改为复制（需约 ${total_h} 可用空间）"
    say "       bash $0 --src $SRC --dst $DST --apply --copy"
    say ""
    say "  注：也可先 sudo chown -R <你的uid> <源目录>，但那会改原目录归属，本脚本不替你做决定。"
    if [ "$APPLY" = "1" ]; then exit 4; fi
  fi
fi

if [ "$APPLY" != "1" ]; then
  say "dry-run 结束：以上为计划。确认后加 --apply 执行。"
  exit 0
fi

# ---------------------------------------------------------------- 执行
mkdir -p "$DST" || die "无法创建 $DST"
fail=0
for ((i=0;i<n;i++)); do
  rel=${NAMES[$i]}; rp=${REALPATHS[$i]}; out="$DST/$rel"
  mkdir -p "$(dirname "$out")" || { printf '  [FAIL] mkdir %s\n' "$(dirname "$out")" >&2; fail=$((fail+1)); continue; }
  if [ "$CROSS" = "1" ] || [ "$USE_COPY" = "1" ]; then
    cp -f "$rp" "$out" || { printf '  [FAIL] cp %s\n' "$rel" >&2; fail=$((fail+1)); }
  else
    ln "$rp" "$out" 2>/dev/null || { printf '  [FAIL] ln %s\n' "$rel" >&2; fail=$((fail+1)); }
  fi
done
say "写入完成：$n 条，失败 $fail 条"
[ "$fail" -eq 0 ] || die "有 $fail 条写入失败"

# ---------------------------------------------------------------- 核对
if [ "$DO_VERIFY" = "1" ]; then
  bad=0
  if [ "$CROSS" = "0" ] && [ "$USE_COPY" = "0" ]; then
    say "一致性核对：逐条比对 inode（硬链接应完全相同）"
    for ((i=0;i<n;i++)); do
      a=$(stat -c '%i' "${REALPATHS[$i]}" 2>/dev/null)
      b=$(stat -c '%i' "$DST/${NAMES[$i]}" 2>/dev/null)
      if [ -z "$a" ] || [ "$a" != "$b" ]; then
        printf '  [MISMATCH] %s（inode %s vs %s）\n' "${NAMES[$i]}" "$a" "$b" >&2; bad=$((bad+1))
      fi
    done
  else
    say "一致性核对：逐条比对大小（复制模式）"
    for ((i=0;i<n;i++)); do
      a=$(stat -c '%s' "${REALPATHS[$i]}" 2>/dev/null)
      b=$(stat -c '%s' "$DST/${NAMES[$i]}" 2>/dev/null)
      [ "$a" = "$b" ] || { printf '  [SIZE-MISMATCH] %s\n' "${NAMES[$i]}" >&2; bad=$((bad+1)); }
    done
  fi
  # 索引引用的分片必须真实存在（起服最容易踩的一处）
  IDX="$DST/quant_model_weights.safetensors.index.json"
  if [ -f "$IDX" ]; then
    REFS=$(mktemp)
    python3 -c "import json,sys;wm=json.load(open(sys.argv[1])).get('weight_map',{});print(chr(10).join(sorted(set(wm.values()))))" "$IDX" > "$REFS" 2>/dev/null
    if [ -s "$REFS" ]; then
      miss=0
      while IFS= read -r f; do
        [ -n "$f" ] || continue
        [ -f "$DST/$f" ] || { printf '  [缺分片] %s\n' "$f" >&2; miss=$((miss+1)); }
      done < "$REFS"
      if [ "$miss" = "0" ]; then printf '  OK      index.json 引用的 %s 个分片都在目标目录里\n' "$(wc -l < "$REFS")"
      else bad=$((bad+miss)); fi
    else
      printf '  [WARN] 无法解析 index.json（跳过该检查）\n' >&2
    fi
    rm -f "$REFS"
  fi
  if [ "$bad" -eq 0 ]; then say "核对通过 ✅（$n 条全部一致）"
  else die "核对发现 $bad 条不一致"; fi
fi

# ---------------------------------------------------------------- 可选：分片
if [ "$EMIT_PARTS" = "1" ]; then
  say "生成 engram 分片（--emit-parts）"
  E="$DST/engram_int8"
  [ -d "$E" ] || die "--emit-parts 需要 $E 存在"
  have_sha=1; command -v sha256sum >/dev/null 2>&1 || have_sha=0
  : > "$E/PARTS.sha256"
  # 把 --part-size（如 16G/4K）换算成字节，用于**按实际片数命名**
  # （不能把 "of-06" 写死：片数取决于文件大小 ÷ 分片大小）
  case "$PART_SIZE" in
    *[gG]) PS_BYTES=$(( ${PART_SIZE%[gG]} * 1024 * 1024 * 1024 )) ;;
    *[mM]) PS_BYTES=$(( ${PART_SIZE%[mM]} * 1024 * 1024 )) ;;
    *[kK]) PS_BYTES=$(( ${PART_SIZE%[kK]} * 1024 )) ;;
    *)     PS_BYTES=$(( PART_SIZE )) ;;
  esac
  [ "$PS_BYTES" -gt 0 ] 2>/dev/null || die "无法解析 --part-size=$PART_SIZE"
  for base in layers_1_engram_embed.weight layers_14_engram_embed.weight; do
    f="$E/$base.safetensors"
    if [ ! -f "$f" ]; then say "  跳过（不存在）：$base"; continue; fi
    fsize=$(stat -c '%s' "$f")
    nparts=$(( (fsize + PS_BYTES - 1) / PS_BYTES ))
    if [ "$have_sha" = "1" ]; then
      printf '\n# %s 重组后 sha256\n%s  %s.safetensors\n' \
        "$base" "$(sha256sum "$f" | cut -d' ' -f1)" "$base" >> "$E/PARTS.sha256"
    fi
    split -b "$PART_SIZE" -d -a 2 "$f" "$f.part-"
    i=0
    for p in "$f".part-*; do
      [ -e "$p" ] || continue
      nw=$(printf '%s.safetensors.part-%02d-of-%02d' "$base" "$i" "$nparts")
      mv -f "$p" "$E/$nw"
      [ "$have_sha" = "1" ] && sha256sum "$E/$nw" >> "$E/PARTS.sha256"
      i=$((i+1))
    done
    if [ "$i" -ne "$nparts" ]; then
      printf '  [WARN] %s 实际切出 %d 片，预期 %d 片（命名按预期值）\n' "$base" "$i" "$nparts" >&2
    fi
    say "  $base → $i 片（每片 $PART_SIZE，命名 part-NN-of-$(printf '%02d' "$nparts")）"
  done
  say "  PARTS.sha256 已写入（含分片 sha256 与重组 sha256）"
  # 生成配套的 reassemble 脚本：已发布那份内嵌的是**它自己那批权重**的
  # sha256，套到本机权重上会校验失败 —— 必须按本机实际值现生成一份。
  if [ "$have_sha" = "1" ]; then
    RS="$E/reassemble_engram_weights.sh"
    {
      printf '%s\n' '#!/usr/bin/env bash'
      printf '%s\n' '# 由 repack_to_modelscope_layout.sh --emit-parts 生成：把 engram 分片按序拼回单文件。'
      printf '%s\n' '# 用法：cd engram_int8 && bash reassemble_engram_weights.sh [--delete-parts|--check-only]'
      printf '%s\n' 'set -uo pipefail'
      printf '%s\n' 'cd "$(dirname "$0")" || exit 2'
      printf '%s\n' 'DEL=0; CHK=0'
      printf '%s\n' 'for a in "$@"; do case "$a" in --delete-parts) DEL=1 ;; --check-only) CHK=1 ;; *) echo "未知参数: $a" >&2; exit 2 ;; esac; done'
      printf '%s\n' 'declare -A WANT=('
      for base in layers_1_engram_embed.weight layers_14_engram_embed.weight; do
        f="$E/$base.safetensors"
        [ -f "$f" ] || continue
        printf '  [%s.safetensors]="%s"\n' "$base" "$(sha256sum "$f" | cut -d" " -f1)"
      done
      printf '%s\n' ')'
      printf '%s\n' 'fail=0'
      printf '%s\n' 'for t in "${!WANT[@]}"; do'
      printf '%s\n' '  parts=(); while IFS= read -r p; do [ -n "$p" ] && parts+=("$p"); done < <(ls -1 "$t".part-*-of-* 2>/dev/null | sort)'
      printf '%s\n' '  if [ "${#parts[@]}" -eq 0 ]; then echo "[FAIL] $t 一个分片都没找到" >&2; fail=$((fail+1)); continue; fi'
      printf '%s\n' '  # 分片名 must 自洽：of-NN 应等于实际片数，且编号连续'
      printf '%s\n' '  want_n=${parts[0]##*-of-}; want_n=$((10#$want_n))'
      printf '%s\n' '  if [ "$want_n" -ne "${#parts[@]}" ]; then echo "[FAIL] $t 名为 of-$want_n，实际 ${#parts[@]} 片" >&2; fail=$((fail+1)); continue; fi'
      printf '%s\n' '  if [ -f "$t" ] && [ "$(sha256sum "$t" | cut -d" " -f1)" = "${WANT[$t]}" ]; then echo "[OK]   $t 已存在且校验通过"; continue; fi'
      printf '%s\n' '  [ "$CHK" = "1" ] && { echo "[CHK]  $t 需要重组（--check-only 不写文件）"; continue; }'
      printf '%s\n' '  tmp="$t.reassembling"; rm -f "$tmp"'
      printf '%s\n' '  for p in "${parts[@]}"; do cat "$p" >> "$tmp" || { echo "[FAIL] 拼接 $p 失败" >&2; fail=$((fail+1)); rm -f "$tmp"; continue 2; }; done'
      printf '%s\n' '  got=$(sha256sum "$tmp" | cut -d" " -f1)'
      printf '%s\n' '  if [ "$got" != "${WANT[$t]}" ]; then echo "[FAIL] $t 校验不通过：got $got" >&2; rm -f "$tmp"; fail=$((fail+1)); continue; fi'
      printf '%s\n' '  mv -f "$tmp" "$t"; echo "[OK]   $t 已还原并通过校验"'
      printf '%s\n' '  [ "$DEL" = "1" ] && { rm -f "${parts[@]}"; echo "       已删除分片"; }'
      printf '%s\n' 'done'
      printf '%s\n' '[ "$fail" -eq 0 ] && { echo "全部通过 ✅"; exit 0; } || { echo "有 $fail 项失败 ❌" >&2; exit 1; }'
    } > "$RS"
    chmod +x "$RS"
    say "  已生成 $RS（内嵌本机权重的正确 sha256）"
  else
    say "  ⚠️  无 sha256sum，未生成 reassemble 脚本"
  fi
fi

# ---------------------------------------------------------------- README
if [ -n "$README_SRC" ]; then
  [ -f "$README_SRC" ] || die "--readme 文件不存在：$README_SRC"
  cp -f "$README_SRC" "$DST/README.md" && say "已放入 README.md"
fi

# ---------------------------------------------------------------- 收尾
say "完成 ✅  扁平自包含目录：$DST"
printf '  条目 %d 个 / 去重实体 %d 个 / 按条目累计约 %s\n' "$n" "$uniq_n" "$total_h"
if [ "$CROSS" = "0" ] && [ "$USE_COPY" = "0" ]; then
  printf '  本次新增磁盘占用：**0**（全部是硬链接；共享 inode，删任一边前请先看 link count）\n'
else
  printf '  本次新增磁盘占用：约 %s（复制模式）\n' "$total_h"
fi
if command -v df >/dev/null 2>&1; then
  printf '  现在**不要**用 `du -sh %s` 判断占用 —— du 会把共享 inode 逐份累加，\n' "$DST"
  printf '  那会报出接近 %s 的假数字。要看真实增量请比较 `df` 前后差值，\n' "$total_h"
  printf '  或数唯一 inode：find %s -type f -printf "%%i\\n" | sort -u | wc -l\n' "$DST"
fi
say "下一步："
cat <<'EOS'
  1) 用新目录起服（不再需要逐层挂载）：
       MODEL=<新目录> DRY_RUN=1 bash scripts/serve_a2.sh | grep -E 'script=|模型挂载'
     期望：挂载层数从 15 层降到 1~2 层（MODEL + engram_int8）。

  2) 本脚本**不删原目录**。硬链接模式下新旧共享 inode、不额外占空间。
     确认新目录能起服后，再自行决定是否清理旧目录；
     ⚠️ 清理前先确认新目录里的文件 link count ≥ 2：
        stat -c '%h %n' <新目录>/config.json      # 期望 >= 2

  3) 要往 ModelScope 传：加 --emit-parts 生成分片，并按已发布的
     engram_int8/reassemble_engram_weights.sh + PARTS.sha256 约定上传。
EOS
