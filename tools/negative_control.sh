#!/usr/bin/env bash
# =============================================================================
# negative_control.sh —— **证明每个自检真的能抓到对应的 bug**
#
#   bash tools/negative_control.sh
#
# ## 为什么需要它（v5 最大的方法论缺口）
#
# v5 有 selfcheck_pkg.sh，但它第一次跑就"全过"——而包其实是坏的（缺两个文件）。
# 原因是：**自检只证明了"我检查的东西是好的"，没证明"检查本身有效"**。
#
# 本脚本对每个已知 bug **故意造一个坏输入**，然后断言对应的检查必须报错：
#   检查通过坏输入 = 检查失效（假阴性）= FAIL
#   检查拒绝坏输入 = 检查有效          = PASS
#
# 造坏输入全部在 `mktemp -d` 的临时副本里做，**不动真实包**。
#
# 覆盖的 bug（对应 v5 的教训清单）：
#   NC1  缺 tests/p15_stream_curve_filefiller.py     （性能测试全废）
#   NC2  缺 tests/vision_accuracy_check.py           （视觉必 FAIL）
#   NC3  Dockerfile 行内注释 + 漏续行符              （构建不出来）
#   NC4  镜像 tag 三处不一致                         （起服即失败）
#   NC5  软链链条有断链                              （挂载悬空）
#   NC6  `-v` 参数被拆成单个元素（mapfile 的坑）      （docker 报 invalid characters）
#   NC7  serve_a2.sh 退回单层挂载                    （软链悬空，回到 v4 状态）
#   NC8  skcache 临时目录堆积                        （磁盘无限增长）
#   NC9  补丁载荷被改动但 md5 清单/落位表没跟上       （v7→v8 的 build_image checksum 失败）
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"

PASS=0
FAIL=0

hr()   { printf '\n\033[1m=== %s\033[0m\n' "$*"; }
good() { PASS=$((PASS+1)); printf '  \033[32mPASS\033[0m  %s\n' "$*"; }
bad()  { FAIL=$((FAIL+1)); printf '  \033[31mFAIL\033[0m  %s\n' "$*"; }
note() { printf '        %s\n' "$*"; }

printf '\033[1m[negctl] %s  pkg=%s\033[0m\n' "$(date '+%F %T')" "$PKG"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
note "临时工作区：$WORK（真实包不会被修改）"

# ---------------------------------------------------------------------------
hr "NC1/NC2  缺文件必须被 preflight 的『包内文件完整性』抓到"
# ---------------------------------------------------------------------------
for miss in "tests/p15_stream_curve_filefiller.py:NC1" \
            "tests/vision_accuracy_check.py:NC2"; do
  _f="${miss%%:*}"; _id="${miss##*:}"
  _d="$WORK/$_id"; mkdir -p "$_d/tools" "$_d/tests"
  cp "$PKG/tools/preflight_a2.sh" "$_d/tools/"
  cp "$PKG/Dockerfile" "$_d/" 2>/dev/null || true
  cp "$PKG/tools/check_dockerfile.py" "$_d/tools/" 2>/dev/null || true
  # 只放需要的文件，**故意不放** _f
  for keep in tests/t_quote.sh tests/t_vision.py tests/t_gsm8k.py tests/acc_eval.py \
              tests/make_report.sh tests/multibatch/multibatch_gate.py \
              tools/check_model_dir.sh tools/model_mount_args.sh \
              scripts/build_image.sh scripts/serve_a2.sh scripts/run_test.sh \
              data/hongloumeng.txt data/suffix_quote.txt; do
    mkdir -p "$_d/$(dirname "$keep")"; cp "$PKG/$keep" "$_d/$keep" 2>/dev/null || true
  done
  [ "$_id" = "NC2" ] && cp "$PKG/tests/p15_stream_curve_filefiller.py" "$_d/tests/" 2>/dev/null || true
  [ "$_id" = "NC1" ] && cp "$PKG/tests/vision_accuracy_check.py" "$_d/tests/" 2>/dev/null || true
  _out=$(cd "$_d" && SKIP_DOCKER=1 bash tools/preflight_a2.sh 2>&1)
  if printf '%s' "$_out" | grep -q "FATAL.*缺文件.*$(basename "$_f")"; then
    good "$_id preflight 正确报 FATAL：缺 $(basename "$_f")"
  else
    bad "$_id preflight **没抓到**缺 $_f（检查失效！）"
    printf '%s\n' "$_out" | grep -E "FATAL|1/9" | head -3 | sed 's/^/          /'
  fi
done

# 正控：文件齐全时不应报这个 FATAL
_d="$WORK/NC_pos"; mkdir -p "$_d"
cp -a "$PKG/." "$_d/" 2>/dev/null || true
_out=$(cd "$_d" && SKIP_DOCKER=1 MODEL=/nonexistent bash tools/preflight_a2.sh 2>&1)
if printf '%s' "$_out" | grep -qE "FATAL.*缺文件"; then
  bad "正控：文件齐全时仍报『缺文件』（检查过严）"
else
  good "正控：文件齐全时不误报缺文件"
fi

# ---------------------------------------------------------------------------
hr "NC3  Dockerfile 行内注释 + 漏续行符必须被 check_dockerfile.py 抓到"
# ---------------------------------------------------------------------------
_d="$WORK/NC3"; mkdir -p "$_d/tools"
cp "$PKG/tools/check_dockerfile.py" "$_d/tools/"
cat > "$_d/Dockerfile" <<'EOF'
FROM alpine
RUN set -euo pipefail; \
    inst() { # src_in_tmp  target_rel
      local tgt="/tmp/$2"; \
      echo "installed $2"; \
    }; \
    inst a b
EOF
if python3 "$_d/tools/check_dockerfile.py" "$_d/Dockerfile" >/dev/null 2>&1; then
  bad "NC3 check_dockerfile 对**坏的** Dockerfile 判为通过（检查失效！）"
else
  good "NC3 check_dockerfile 正确拒绝坏 Dockerfile（unknown instruction 的前身）"
fi
# 正控
if python3 "$PKG/tools/check_dockerfile.py" "$PKG/Dockerfile" >/dev/null 2>&1; then
  good "NC3 正控：真实 Dockerfile 判为通过"
else
  bad "NC3 正控：真实 Dockerfile 被判为坏（误报）"
fi

# 真机验证：坏的 Dockerfile 必须让 docker build 失败
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  if docker build -f "$_d/Dockerfile" -t negctl:broken "$_d" >/dev/null 2>&1; then
    bad "NC3 docker 竟然接受了坏 Dockerfile（与预期不符）"
  else
    good "NC3 端到端：docker build 确实拒绝坏 Dockerfile"
  fi
else
  note "NC3 端到端跳过（docker 不可用）"
fi

# ---------------------------------------------------------------------------
hr "NC4  镜像 tag 三处不一致必须被 preflight 抓到"
# ---------------------------------------------------------------------------
_d="$WORK/NC4"; cp -a "$PKG/." "$_d/" 2>/dev/null || true
# ⚠️ 这里的源串必须与 scripts/build_image.sh 的**当前**默认 tag 一致，
#    否则 sed 静默不匹配 ⇒ NC4 变成假通过（最坏的一类 bug）。
sed -i 's/dsv41-a2:v8/dsv41-a2:v9/' "$_d/scripts/build_image.sh"
_out=$(cd "$_d" && SKIP_DOCKER=1 bash tools/preflight_a2.sh 2>&1)
if printf '%s' "$_out" | grep -qE "FATAL.*tag 不一致"; then
  good "NC4 preflight 正确报 FATAL：镜像 tag 不一致"
else
  bad "NC4 preflight **没抓到** tag 不一致（检查失效！）"
fi

# ---------------------------------------------------------------------------
hr "NC5  软链断链必须被 model_mount_args.sh 抓到"
# ---------------------------------------------------------------------------
_d="$WORK/NC5"; mkdir -p "$_d/tools" "$_d/L1" "$_d/L2"
cp "$PKG/tools/model_mount_args.sh" "$_d/tools/"
echo '{"a":1}' > "$_d/L1/config.json"
ln -sfn "$_d/L1/config.json" "$_d/L2/config.json"          # 好链
ln -sfn "/nonexistent/gone.json" "$_d/L2/engram_extra.safetensors"   # 坏链
if bash "$_d/tools/model_mount_args.sh" "$_d/L2" >/dev/null 2>&1; then
  bad "NC5 model_mount_args 对含断链的目录判为通过（检查失效！）"
else
  good "NC5 model_mount_args 正确拒绝含断链的目录（rc≠0）"
fi
# 正控：健康的 5 层链条
_d5="$WORK/NC5pos"; mkdir -p "$_d5/tools"
cp "$PKG/tools/model_mount_args.sh" "$_d5/tools/"
mkdir -p "$_d5/L1" "$_d5/L2" "$_d5/L3" "$_d5/L4" "$_d5/L5"
echo '{}' > "$_d5/L1/config.json"
for a in 2 3 4 5; do ln -sfn "$_d5/L$((a-1))/config.json" "$_d5/L$a/config.json"; done
_n=$(bash "$_d5/tools/model_mount_args.sh" "$_d5/L5" 2>/dev/null | grep -c .)
if [ "${_n:-0}" -eq 5 ]; then
  good "NC5 正控：5 层链条解析出 5 个目录"
else
  bad "NC5 正控：5 层链条只解析出 $_n 个（应 5）—— realpath 折叠的老 bug 回归了？"
fi

# ---------------------------------------------------------------------------
# 注意：标题里**不要用反引号** —— 双引号内的反引号会被 shell 做命令替换
# （本脚本第一版就在这里报了 `-v: command not found`）。要用就转义。
hr 'NC6  "-v" 参数必须是『两个独立数组元素』（v5 的 mapfile 坑）'
# ---------------------------------------------------------------------------
# 复刻 serve_a2.sh 的组装逻辑：上游只吐裸路径，这里拼 -v + 路径
_d="$WORK/NC6"; mkdir -p "$_d/tools"
cp "$PKG/tools/model_mount_args.sh" "$_d/tools/"
mkdir -p "$_d/L1" "$_d/L2"; echo '{}' > "$_d/L1/config.json"
ln -sfn "$_d/L1/config.json" "$_d/L2/config.json"
mapfile -t _mdirs < <(bash "$_d/tools/model_mount_args.sh" "$_d/L2" 2>/dev/null | sed '/^[[:space:]]*$/d')
ARGS=(); for _x in "${_mdirs[@]}"; do ARGS+=(-v "$_x:$_x:ro"); done
# 断言：偶数个元素，且奇数下标都以 "-v" 开头
_ok=1
[ $(( ${#ARGS[@]} % 2 )) -eq 0 ] || _ok=0
for ((i=0;i<${#ARGS[@]};i+=2)); do [ "${ARGS[$i]}" = "-v" ] || _ok=0; done
if [ "$_ok" = "1" ]; then
  good "NC6 参数组装正确：${#ARGS[@]} 个元素（$((${#ARGS[@]}/2)) 个挂载），-v 与路径分离"
else
  bad "NC6 参数组装错误：-v 与路径被粘在一起（docker 会报 invalid characters）"
fi
# 负控：若上游吐 "-v /p:/p:ro"（v5 的写法），单元素会带前导空格
_legacy=$(printf '%s\n' "-v $WORK/NC6/L2:$WORK/NC6/L2:ro")
if printf '%s' "$_legacy" | grep -q '^-v .*:.*:ro$'; then
  note "NC6 负控形态已复现：'$_legacy' 作为**单个**参数时，pflag 会把 '-v ' 后的空格算进值"
fi
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  if docker run --rm "-v $WORK/NC6/L2:$WORK/NC6/L2:ro" alpine true >/dev/null 2>&1; then
    note "NC6 docker 端到端跳过（本机 docker 未按预期拒绝）"
  else
    good "NC6 端到端：docker 确实拒绝粘在一起的 '-v path' 单参数"
  fi
fi

# ---------------------------------------------------------------------------
hr "NC7  serve_a2.sh 必须用 MODEL_MOUNTS（不能用单层 -v \$MODEL）"
# ---------------------------------------------------------------------------
if grep -q 'MODEL_MOUNTS' "$PKG/scripts/serve_a2.sh" \
   && ! grep -qE '^\s*-v "\$MODEL:\$MODEL' "$PKG/scripts/serve_a2.sh"; then
  good "NC7 serve_a2.sh 用 MODEL_MOUNTS（软链链会逐层挂载）"
else
  bad "NC7 serve_a2.sh 仍是单层 -v \$MODEL:\$MODEL ⇒ 软链会悬空"
fi

# ---------------------------------------------------------------------------
hr "NC8  skcache 临时目录 GC 必须存在且只删 ts*"
# ---------------------------------------------------------------------------
if grep -q 'ts\*_outputs' "$PKG/scripts/serve_a2.sh" \
   && grep -q 'static_kernel_cache' "$PKG/scripts/serve_a2.sh"; then
  good "NC8 serve_a2.sh 有 ts*_outputs GC 且显式保留 static_kernel_cache/"
else
  bad "NC8 缺 skcache GC（A3 上曾攒到 1491 个目录 / 847 MB）"
fi
# 端到端：造一个假 skcache，跑 GC，断言只删 ts*、保留 cache
_d="$WORK/NC8"; mkdir -p "$_d/cache/skcache/compile_outputs/static_kernel_cache" \
                          "$_d/cache/skcache/compile_outputs/ts20260101_pid1_outputs"
echo '{}' > "$_d/cache/skcache/compile_outputs/static_kernel_cache/CANN-9.1.0_X.json"
echo 'x'  > "$_d/cache/skcache/compile_outputs/ts20260101_pid1_outputs/f"
find "$_d/cache/skcache/compile_outputs" -maxdepth 1 -type d -name 'ts*_outputs' -exec rm -rf {} + 2>/dev/null
if [ -f "$_d/cache/skcache/compile_outputs/static_kernel_cache/CANN-9.1.0_X.json" ] \
   && [ ! -d "$_d/cache/skcache/compile_outputs/ts20260101_pid1_outputs" ]; then
  good "NC8 端到端：GC 删掉 ts* 且**保住了** static_kernel_cache/"
else
  bad "NC8 端到端：GC 行为错误（误删缓存 或 没删临时目录）"
fi

# ---------------------------------------------------------------------------
hr "NC9  补丁载荷改动而 md5 清单没跟上，必须被 check_checksums 抓到"
# ---------------------------------------------------------------------------
# 这正是用户报的那个 bug：v8 改了 patches/files/{model,engram_hbm}.py，而 build_image.sh
# 里那张手写 md5 表没跟上 ⇒ build 跑到最后一步才 FAIL md5。
# 现在的判据：期望 md5 **由载荷字节现算**，任何"载荷 vs 清单"不一致都必须报出来。
if bash "$PKG/tools/check_checksums.sh" >/dev/null 2>&1; then
  good "NC9 正控：真实包三方一致（Dockerfile 落位表 / patches/files / MD5SUMS）"
else
  bad "NC9 正控：真实包三方**不一致**（先修真实包，否则 build_image 必然失败）"
fi

_d="$WORK/NC9"; mkdir -p "$_d/tools" "$_d/patches/files" "$_d/patches/vllm-ascend"
cp "$PKG/Dockerfile" "$_d/"
cp "$PKG/tools/check_checksums.py" "$PKG/tools/check_checksums.sh" "$_d/tools/"
cp -a "$PKG/patches/files/." "$_d/patches/files/"
cp "$PKG/patches/MD5SUMS" "$_d/patches/MD5SUMS"
cp "$PKG/patches/vllm-ascend/MD5SUMS" "$_d/patches/vllm-ascend/MD5SUMS"
# 造坏输入：往载荷尾部加一行注释（内容变了，两份 md5 清单都还是旧值）
printf '\n# negctl NC9\n' >> "$_d/patches/files/model.py"
_out=$(bash "$_d/tools/check_checksums.sh" 2>&1)
if printf '%s' "$_out" | grep -q "FAIL" && printf '%s' "$_out" | grep -q "model.py"; then
  good "NC9 负控：载荷被改动后 check_checksums 报 FAIL 且点名 model.py"
else
  bad "NC9 负控：载荷被改动却判为通过（检查失效！）"
fi

# ---------------------------------------------------------------------------
hr "NC10  镜像内校验脚本必须真的能分辨『逐字节一致』与『被改过一个字节』"
# ---------------------------------------------------------------------------
# verify_baked_tree.sh 是 build_image.sh 最后一步用的脚本；这里在**假树**上做正/负控，
# 不需要 docker，也不需要 10–20 分钟的 build。
_d="$WORK/NC10"; mkdir -p "$_d/tools"
cp "$PKG/tools/verify_baked_tree.sh" "$_d/tools/"
# 用**落位表**把载荷铺成一棵模拟镜像树（等价于"理想 build 的结果"，不需要 docker）
python3 "$PKG/tools/check_checksums.py" --manifest "$_d/chk.tsv" \
        --materialize "$_d/tree" --quiet-ok >/dev/null 2>&1
_n=$(grep -c . "$_d/chk.tsv")
if bash "$_d/tools/verify_baked_tree.sh" --root "$_d/tree" --manifest "$_d/chk.tsv" >/dev/null 2>&1; then
  good "NC10 正控：假树（$_n 项 + 备份）判为逐字节一致"
else
  bad "NC10 正控：一致的假树被判为不一致（误报）"
fi
printf '\n# negctl NC10\n' >> "$_d/tree/models/deepseek_v41/model.py"
if bash "$_d/tools/verify_baked_tree.sh" --root "$_d/tree" --manifest "$_d/chk.tsv" >/dev/null 2>&1; then
  bad "NC10 负控：被改过一个字节的 model.py 仍判为一致（检查失效！）"
else
  good "NC10 负控：改一个字节即 FAIL（就是用户当年看到的那条）"
fi
cp "$PKG/patches/files/model.py" "$_d/tree/models/deepseek_v41/model.py"
rm -f "$_d/tree/models/deepseek_v41/model.py.a2orig"
if bash "$_d/tools/verify_baked_tree.sh" --root "$_d/tree" --manifest "$_d/chk.tsv" >/dev/null 2>&1; then
  bad "NC10 备份负控：缺 .a2orig 回滚备份却判为通过（检查失效！）"
else
  good "NC10 备份负控：缺 .a2orig 回滚备份即 FAIL"
fi

# ---------------------------------------------------------------------------
hr "汇总"
# ---------------------------------------------------------------------------
