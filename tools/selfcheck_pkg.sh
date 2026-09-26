#!/usr/bin/env bash
# =============================================================================
# selfcheck_pkg.sh —— 包内一致性自检（**起服前先跑，10 秒**）
#
#   bash tools/selfcheck_pkg.sh
#
# 为什么需要：打包/改版时最容易出的一类错是**版本标记不一致** ——
# 例如 build_image.sh 产出 `dsv41-a2:v4`，而 serve_a2.sh 默认去找
# `dsv41-a2:v5` ⇒ 起服直接 "镜像不存在"，白等一场。本脚本把这类问题
# 在 10 秒内抓出来。
#
# 退出码：0 = 通过；1 = 有不一致（必须修）
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
cd "$PKG"

fail=0
ok()   { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; fail=$((fail+1)); }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$*"; }

echo "[selfcheck] 包根: $PKG"

# ---------------------------------------------------------------- 1) 镜像 tag
# 从四个地方抽取 tag，必须完全一致
tag_of() {  # $1=file  $2=regex
  [ -f "$1" ] || { echo ""; return; }
  grep -oE "$2" "$1" 2>/dev/null | head -1
}
TAG_BUILD=$(tag_of scripts/build_image.sh 'IMAGE_TAG:-dsv41-a2:v[0-9]+' | sed 's/.*://')
TAG_SERVE=$(tag_of scripts/serve_a2.sh    'IMAGE:-dsv41-a2:v[0-9]+'    | sed 's/.*://')
TAG_TEST=$(tag_of  scripts/run_test.sh    'IMAGE:-dsv41-a2:v[0-9]+'    | sed 's/.*://')

echo "  镜像 tag: build=$TAG_BUILD serve=$TAG_SERVE run_test=$TAG_TEST"
if [ -z "$TAG_BUILD" ] || [ -z "$TAG_SERVE" ] || [ -z "$TAG_TEST" ]; then
  bad "有脚本里找不到镜像 tag（正则失配？）"
elif [ "$TAG_BUILD" = "$TAG_SERVE" ] && [ "$TAG_SERVE" = "$TAG_TEST" ]; then
  ok "镜像 tag 三处一致：$TAG_BUILD"
else
  bad "镜像 tag 不一致！build_image 产出 '$TAG_BUILD'，但 serve_a2/run_test 找 '$TAG_SERVE'/'$TAG_TEST'"
  echo "        修法：sed -i 's/dsv41-a2:vX/dsv41-a2:vY/g' scripts/*.sh Dockerfile"
fi

# ------------------------------------------------- 2) tag 与包目录名是否对齐
PKGNAME=$(basename "$PKG")
if printf '%s' "$PKGNAME" | grep -qE 'a2_pkg_v([0-9]+)'; then
  WANT="v$(printf '%s' "$PKGNAME" | sed -n 's/.*a2_pkg_v\([0-9]\+\)/\1/p')"
  if [ "$TAG_SERVE" = "$WANT" ]; then
    ok "包目录名与镜像 tag 对齐：$PKGNAME / $TAG_SERVE"
  else
    warn "包目录名 $PKGNAME 暗示 tag $WANT，但脚本用 $TAG_SERVE（不致命，但容易混淆）"
  fi
fi

# ------------------------------------------------- 3) 关键脚本存在 + 语法
for f in scripts/build_image.sh scripts/serve_a2.sh scripts/run_test.sh \
         tools/model_mount_args.sh tools/check_model_dir.sh \
         tools/preflight_a2.sh tools/negative_control.sh \
         tools/check_checksums.sh tools/verify_baked_tree.sh \
         build_scripts/00_ensure_pgo.sh \
         tests/t_quote.sh tests/multibatch/multibatch_session.sh; do
  if [ ! -f "$f" ]; then
    bad "缺文件：$f"
  elif bash -n "$f" 2>/dev/null; then
    ok "$f（存在 + 语法通过）"
  else
    bad "$f 语法错误"
  fi
done

# Python 文件要用 py_compile（别拿 bash -n 检查 .py —— v6 第一版就犯过）
for f in tests/t_vision.py tests/t_gsm8k.py tests/acc_eval.py \
         tests/vision_accuracy_check.py tests/p15_stream_curve_filefiller.py \
         tests/multibatch/multibatch_gate.py \
         tools/check_dockerfile.py tools/fisher_recheck.py tools/steep_summary.py \
         tools/check_checksums.py \
         tools/model_mount_args.sh; do
  [ -f "$f" ] || { bad "缺文件：$f"; continue; }
  case "$f" in *.py)
    if python3 -c "import ast,sys; ast.parse(open(sys.argv[1],encoding='utf-8').read())" "$f" 2>/dev/null; then
      ok "$f（Python 语法通过）"
    else
      bad "$f Python 语法错误"
    fi ;;
  esac
done

# 必需的数据文件（v5 曾漏打包 p15 → 性能测试全废）
for f in data/hongloumeng.txt data/suffix_quote.txt; do
  [ -f "$f" ] && ok "$f" || bad "缺数据文件：$f"
done

# ------------------------------------------- 4) 软链挂载逻辑确实接在 serve 里
if grep -q 'MODEL_MOUNTS' scripts/serve_a2.sh && \
   ! grep -qE '^\s*-v "\$MODEL:\$MODEL' scripts/serve_a2.sh; then
  ok "serve_a2.sh 用的是 MODEL_MOUNTS（不再单层挂载）"
else
  bad "serve_a2.sh 里仍是单层 -v \$MODEL:\$MODEL（软链会悬空）"
fi

# ------------------------------------------- 5) Dockerfile 续行链（A2 实测踩过的坑）
# 漏一个 `\` 或行内写 `#` 都会让 RUN 提前结束 / 后续命令被注释掉，
# 表现为 "unknown instruction: local" 之类难以定位的报错。
#
# ★ 2026-09-25 实测：下面几处把检查器输出写到**硬编码的 `/tmp`**。当 `/tmp`
#   被别的进程占满时，写失败 ⇒ 检查器输出读不到 ⇒ 自检报 FAIL，**看起来像代码坏了**
#   其实是环境问题（本机当时 /tmp 8G 用满）。改用 `${TMPDIR:-/tmp}` 后，
#   只要 `TMPDIR=<有空间的分区>` 就能正常自检。
_selfcheck_tmp=${TMPDIR:-/tmp}
[ -d "$_selfcheck_tmp" ] || _selfcheck_tmp=/tmp
if [ -f Dockerfile ]; then
  if python3 tools/check_dockerfile.py Dockerfile >"$_selfcheck_tmp/dsck.$$" 2>&1; then
    ok "Dockerfile 续行链合法（$(grep -c '^\s*RUN' Dockerfile) 条 RUN）"
  else
    bad "Dockerfile 续行链有问题："
    sed 's/^/        /' "$_selfcheck_tmp/dsck.$$" | grep -E "ERROR|问题" | head -6
  fi
  rm -f "$_selfcheck_tmp/dsck.$$"
fi

# ------------------------------------------- 5b) 起服脚本的 `docker run` 续行链
#
# ⚠️ 2026-09-20 A2 真机炸过一次：一段**注释块被插进 `docker run` 的续行链中间**，
#    续行在那里终止 ⇒ 后半段 `-e ...` 变成独立命令、`docker run` 丢掉 IMAGE 参数。
#    报错是 `"docker run" requires at least 1 argument` + `-e: command not found`，
#    **不指向注释**，排查成本很高。
#    而 `bash -n` **抓不到**（拼接后语法合法）—— 与上面 Dockerfile 的坑是同一类，
#    所以这里用同样的思路加一道静态检查。
if [ -f tools/check_serve_run_chain.py ]; then
  if python3 tools/check_serve_run_chain.py scripts/serve_a2.sh >"$_selfcheck_tmp/rcck.$$" 2>&1; then
    ok "起服脚本 docker run 续行链合法（$(grep -c '^\s*\$DOCKER run' scripts/serve_a2.sh 2>/dev/null) 处）"
  else
    bad "起服脚本的 docker run 续行链有问题（注释/空行插在续行链里？）："
    sed 's/^/        /' "$_selfcheck_tmp/rcck.$$" | grep -E "FAIL|第 .* 行|修法" | head -6
  fi
  rm -f "$_selfcheck_tmp/rcck.$$"
fi

# ------------------------------------------------- 6) 执行位（缺了也能跑，但要提醒）
_noexec=0
for f in $(find . -name "*.sh" -not -path "./results/*" 2>/dev/null); do
  [ -x "$f" ] || _noexec=$((_noexec+1))
done
if [ "$_noexec" = "0" ]; then ok "所有 .sh 都有执行位"
else warn "$_noexec 个 .sh 缺执行位（不影响——脚本都用 bash 调用）"; fi

# ------------------------------------------------- 7) MANIFEST 自校验（若存在）
if [ -f MANIFEST.sha256 ]; then
  n_total=$(grep -c . MANIFEST.sha256)
  n_bad=$(sha256sum -c MANIFEST.sha256 2>/dev/null | grep -c -v ': OK' || true)
  if [ "$n_bad" = "0" ]; then ok "MANIFEST.sha256 自校验通过（$n_total 项）"
  else bad "MANIFEST.sha256 有 $n_bad/$n_total 项不匹配（文件被改过？重跑 sha256sum）"; fi
else
  warn "没有 MANIFEST.sha256"
fi

# ------------------------------------------------- 8) 校验和一致性（v8 补的坑）
# build_image.sh 早期把补丁 md5 **硬编码**在 chk() 里，v7→v8 忘了同步 ⇒ 用户 build 到
# 最后一步才报 "FAIL md5 .../model.py"。现在期望 md5 由载荷字节现算，本项检查则保证
# 「Dockerfile 落位表 / patches/files / MD5SUMS」三方一致、且没有"装了却没被校验"的文件。
if [ -x tools/check_checksums.sh ] || [ -f tools/check_checksums.sh ]; then
  if out=$(bash tools/check_checksums.sh 2>&1); then
    n=$(printf '%s' "$out" | sed -n 's/^\[chk\] 落位表 \([0-9]*\) 项.*/\1/p' | head -1)
    warn_cnt=$(printf '%s' "$out" | grep -c '^\[chk\]\[WARN\]' || true)
    if [ "${warn_cnt:-0}" = "0" ]; then
      ok "校验和一致性：Dockerfile 落位表（${n:-?} 项）/ patches/files / MD5SUMS 三方一致"
    else
      warn "校验和一致性通过，但有 $warn_cnt 条 WARN（非运行时载荷的清单过期，见下）"
      printf '%s\n' "$out" | grep '^\[chk\]\[WARN\]' | sed 's/^/        /'
    fi
  else
    bad "校验和一致性失败（build_image.sh 会因此拒绝构建）："
    printf '%s\n' "$out" | grep -E '^\s+-|^\[chk\]\[FAIL\]' | sed 's/^/        /' | head -12
  fi
else
  bad "缺 tools/check_checksums.sh（v8 起 build_image.sh 依赖它推导期望 md5）"
fi

# ------------------------------------------------- 9) A3 部署器 + 其沙箱自测
# 为什么单列：`tools/deploy_a3.sh` 是**新机器上跑的第一条命令**，它的每条门都必须
# 「该拦的拦住、该过的过」；而它的门里有一堆外部依赖（docker/npu-smi/选卡/模型自检）
# ⇒ 用**桩**做沙箱自测（零真机）。判据尽量绑"下游真收到的值"，不是"deploy 的自述"。
for _f in tools/deploy_a3.sh docs/A3-DEPLOY.md; do
  if [ -f "$_f" ]; then ok "A3 新机部署件在位：$_f"
  else warn "缺 $_f（新机器上少一条从零到干跑的路径）"; fi
done
if [ -f tools/diag_a3_hccl.sh ]; then
  if bash -n tools/diag_a3_hccl.sh 2>/dev/null; then ok "HCCL 诊断助手在位且语法通过：tools/diag_a3_hccl.sh"
  else bad "tools/diag_a3_hccl.sh 语法不通过"; fi
else
  warn "缺 tools/diag_a3_hccl.sh（新机 HCCL 建链失败时少一条一次收齐证据的路径）"
fi
if [ -f tools/stop_a3_safe.sh ]; then
  if bash -n tools/stop_a3_safe.sh 2>/dev/null; then ok "安全停机脚本在位且语法通过：tools/stop_a3_safe.sh"
  else bad "tools/stop_a3_safe.sh 语法不通过"; fi
else
  warn '缺 tools/stop_a3_safe.sh（vLLM 卡死时少一条"容器停不下来"的分步解法）'
fi
if [ -f tools/selftest_stop_a3_safe.sh ]; then
  if out=$(bash tools/selftest_stop_a3_safe.sh 2>&1); then
    n=$(printf '%s' "$out" | grep -c 'PASS' || true)
    ok "安全停机脚本沙箱自测：${n:-?} 条全过（正常停 / 卡住现场 / 只诊断不许动手 / sudo 不可用 / D 状态）"
  else
    bad "安全停机脚本沙箱自测失败："
    printf '%s' "$out" | grep -E 'FAIL' | sed 's/^/        /' | head -8
  fi
else
  warn "缺 tools/selftest_stop_a3_safe.sh"
fi

# ------------------------------------------------- 9a-2) ★ run_test.sh 的 prefill batch 默认
# 乱码根因修复(`8eb2613`)改的是模板；而 run_test.sh **显式**把 BAT_TOKENS 传下去
#   ⇒ 它的默认值会覆盖模板。长期是 2048 ⇒ "用标准验证入口验证长上下文精度"
#   等于在一个已知会退化的配置上验证。这里把这条件链钉死。
if [ -f tools/selftest_run_test_bat.sh ]; then
  if out=$(bash tools/selftest_run_test_bat.sh 2>&1); then
    n=$(printf '%s' "$out" | grep -c 'PASS' || true)
    ok "run_test prefill batch 守护：${n:-?} 条全过（默认 8192 / 真透传 / 与模板一致 / KV 门槛自洽 / 负控）"
  else
    bad "run_test prefill batch 守护失败："
    printf '%s' "$out" | grep -E 'FAIL' | sed 's/^/        /' | head -8
  fi
else
  bad "缺 tools/selftest_run_test_bat.sh（无法自动抓"验证入口把 8192 覆盖回 2048"）"
fi

# ------------------------------------------------- 9b) A3 镜像「层补丁」工具链
# 2026-09-23：把 A3 的 TP8 工作形态固化成"官方基础镜像 + 1 个工作层"（0.2 MiB），
#   已发布 dsv41-a3-tp8-imagekit-v1。这里查工具链在位 + 语法 + 挂载解析器的正负控。
for _f in tools/build_a3_tp8_image.sh tools/package_image_kits.sh docs/A3-IMAGE-KIT.md; do
  [ -s "$_f" ] && ok "镜像层工具链在位：$_f" || bad "缺 $_f"
done
bash -n tools/build_a3_tp8_image.sh 2>/dev/null && ok "build_a3_tp8_image.sh 语法通过" \
  || bad "build_a3_tp8_image.sh 语法不通过"
for _f in tools/make_image_patch_kit.py tools/mount_list_pairs.py tools/kit_tools/fs_manifest.py; do
  if python3 -m py_compile "$_f" 2>/dev/null; then ok "可编译：$_f"; else bad "$_f 编译不通过"; fi
done
# ★ 挂载解析器的**负控**：孤立 -v / 重复目标必须被判 FAIL（否则判据等于没有）
_p=$(mktemp -d)
printf '[a2-dry] MOUNTS(3): -v A:/t:ro -v\n' > "$_p/orphan.txt"
printf '[a2-dry] MOUNTS(4): -v A:/t:ro -v B:/t:ro\n' > "$_p/dup.txt"
printf '[a2-dry] MOUNTS(4): -v A:/t1:ro -v B:/t2:ro\n' > "$_p/ok.txt"
python3 tools/mount_list_pairs.py --check "$_p/orphan.txt" >/dev/null 2>&1 \
  && bad "孤立 -v 竟判 OK" || ok "负控：孤立 -v 被判 FAIL"
python3 tools/mount_list_pairs.py --check "$_p/dup.txt" >/dev/null 2>&1 \
  && bad "重复目标竟判 OK" || ok "负控：重复目标被判 FAIL"
python3 tools/mount_list_pairs.py --check "$_p/ok.txt" >/dev/null 2>&1 \
  && ok "正控：正常清单判 OK" || bad "正常清单被误判"
rm -rf "$_p"

# ------------------------------------------------- 9c) 长上下文 × Agent 精度探针 + 其沙箱自测
# 现场反馈：问题主要出现在**长上下文 + Agent（工具调用/多轮）**场景，且**不开 DRAM 卸载也有**
#   ⇒ 判据必须补上这一格（此前的题库只有几十 token、sha 判据对长上下文语义无判别力）。
# ★ 判据按**文件类型**给：`.py` 用 py_compile；`.md` 只查存在与非空
#   （第一版对 .md 也调 py_compile ⇒ 误报"编译不通过"；同族：判据要绑对它检查的对象。）
for _f in tools/ctx_agent_probe.py tools/_fake_vllm.py tools/_check_probe_json.py; do
  if [ -f "$_f" ]; then
    if python3 -m py_compile "$_f" 2>/dev/null; then ok "长上下文档位在位且可编译：$_f"
    else bad "$_f 编译不通过"; fi
  else
    bad "缺 $_f"
  fi
done
if [ -s docs/CTX-AGENT-REPRO.md ]; then ok "A2 复现手册在位且非空：docs/CTX-AGENT-REPRO.md"
else bad "缺 docs/CTX-AGENT-REPRO.md（A2 上少一条粘贴即用的复现路径）"; fi
if [ -f tools/selftest_ctx_agent_probe.sh ]; then
  if out=$(bash tools/selftest_ctx_agent_probe.sh 2>&1); then
    n=$(printf '%s' "$out" | grep -c 'PASS' || true)
    ok "长上下文探针沙箱自测：${n:-?} 条全过（干净必过 / 乱码必抓 / 工具参数逐字 / 无 tokenize 回退 / 连不上 rc=2）"
  else
    bad "长上下文探针沙箱自测失败："
    printf '%s' "$out" | grep -E 'FAIL' | sed 's/^/        /' | head -8
  fi
else
  warn "缺 tools/selftest_ctx_agent_probe.sh"
fi
if [ -f tools/deploy_a3.sh ] && ! bash -n tools/deploy_a3.sh 2>/dev/null; then
  bad "tools/deploy_a3.sh 语法不通过"
fi
if [ -f tools/selftest_deploy_a3.sh ]; then
  if out=$(bash tools/selftest_deploy_a3.sh 2>&1); then
    n=$(printf '%s' "$out" | grep -c 'PASS' || true)
    ok "A3 部署器沙箱自测：${n:-?} 条全过（镜像缺失 / 模型断链 / 选卡不足 / 干跑判据 / 起服不吞 rc）"
  else
    bad "A3 部署器沙箱自测失败："
    printf '%s' "$out" | grep -E 'FAIL' | sed 's/^/        /' | head -8
  fi
else
  warn "缺 tools/selftest_deploy_a3.sh（无法自动抓"部署器门失效"）"
fi
# ------------------------------------------------- 9d) 起服脚本的**沙箱端到端自测**（v9 补的坑）
# 2026-09-23 同日两次同类事故：`serve_a2_offload.sh` 里**变量在定义之前被使用**
#   ① 头部引用了 `$OUT`（那只在 shadow 的 serve_a2.sh 里定义）⇒ set -u 崩溃；
#   ② 头部引用了 `$LAUNCH_DIR`，而它当时定义在文件后半 ⇒ `line 247: LAUNCH_DIR: unbound variable`。
# `bash -n` **查不出来**（只查语法）。本项把脚本放进沙箱真跑（DRY=1）⇒ 任何未定义变量都会暴露。
if [ -f a2/scripts/selftest_serve_a2_offload.sh ]; then
  if out=$(bash a2/scripts/selftest_serve_a2_offload.sh 2>&1); then
    n=$(printf '%s' "$out" | grep -c '^✓ \[' || true)
    ok "起服脚本沙箱自测：${n:-?} 个用例全过（未定义变量 / PROFILE 透传 / 门拦截）"
  else
    bad "起服脚本沙箱自测失败（含未定义变量、门失效等）："
    printf '%s' "$out" | grep -E '^⛔' | sed 's/^/        /' | head -8
  fi
else
  warn "缺 a2/scripts/selftest_serve_a2_offload.sh（无法自动抓"变量未定义"这类崩溃）"
fi

# ------------------------------------------------- 9e) ★ 两处 BAT_TOKENS 默认必须一致
# 乱码根因修复(`8eb2613`)改的是**模板** scripts/serve_a2.sh；而 a2/scripts/serve_a2_offload.sh
#   **显式**把 BAT_TOKENS 传给模板 ⇒ 包装脚本的默认值会覆盖模板。两者不一致时，
#   用包装脚本起服就会静默退回修复前（1M 下 2048 ⇒ 256 刀 ⇒ 通过率≈0）。
# ★ 取值不能用 `grep -oE '[0-9]+$'` —— 值后面紧跟 `}`，`$` 锚不匹配（实测踩到，导致本项一直
#   "读不到"却仍打印 OK 的假象）。改用 sed 在完整串上取 `:-` 与 `}` 之间的数字。
_bat_of() { grep -oE 'BAT_TOKENS=\$\{BAT_TOKENS:-[0-9]+\}' "$1" 2>/dev/null | head -1 \
              | sed -n 's/.*:-\{0,1\}\([0-9]\{1,\}\)}.*/\1/p'; }
_tmpl=$(_bat_of scripts/serve_a2.sh)
_wrap=$(_bat_of a2/scripts/serve_a2_offload.sh)
if [ -n "$_tmpl" ] && [ -n "$_wrap" ]; then
  if [ "$_tmpl" = "$_wrap" ]; then ok "BAT_TOKENS 默认一致：模板=$_tmpl 包装=$_wrap"
  else bad "BAT_TOKENS 默认**不一致**：模板=$_tmpl 包装=$_wrap ⇒ 用包装脚本起服会退回 $_wrap（长上下文退化）"; fi
else
  warn "读不到 BAT_TOKENS 默认（模板='$_tmpl' 包装='$_wrap'）"
fi

# ------------------------------------------------- 9f) shadow 生成物回归（重复挂载/孤立 -v）
# 2026-09-23 真机两次踩到、且 `bash -n` 查不出来：① int8 块与生产块挂同一目标 ⇒ docker
#   `Duplicate mount point`；② 去重只删一半 ⇒ 留孤立 `-v` ⇒ `invalid reference format.`
if [ -f a2/scripts/selftest_make_shadow_pkg.sh ]; then
  if out=$(bash a2/scripts/selftest_make_shadow_pkg.sh 2>&1); then
    n=$(printf '%s' "$out" | grep -c 'PASS' || true)
    ok "shadow 生成物回归：${n:-?} 条全过（int8 去重 / 成对完整 / 目标唯一 / 校验器负控）"
  else
    bad "shadow 生成物回归失败："
    printf '%s' "$out" | grep -E 'FAIL' | sed 's/^/        /' | head -8
  fi
else
  warn "缺 a2/scripts/selftest_make_shadow_pkg.sh"
fi

# ------------------------------------------------- 9h) ★ 本轮的 issue 跟进修复
# 每条都对应一个**外部报告过的真实故障**，不是内部重构。判据：正控过 + 负控能抓。
#   issue #2 ② 代理劫持 127.0.0.1 ⇒ 三个起服脚本必须补 no_proxy（且**不覆盖**用户已设的）
#   issue #2 3.1 `say()` 早于 `mkdir` ⇒ driver.log 头 ~100 行丢失
#   issue #2 3.6 / 待办 3 KV 门槛写死 3Mi ⇒ 默认配置必报假红；三处必须同口径
#   issue #2 7.2 bench_concurrency 用 data[0] 覆盖 --model ⇒ 压测目标被悄悄改掉
#   本仓自查  `curl ... \|\| echo 000` 会拼出 "000000"（AGENTS §3.2 同族），5 处
if [ -f tools/selftest_issue_followups.sh ]; then
  if out=$(bash tools/selftest_issue_followups.sh 2>&1); then
    n=$(printf '%s' "$out" | grep -c '^  ✓' || true)
    ok "issue 跟进修复回归：${n:-?} 条全过（no_proxy / say-mkdir / KV 门槛口径 / bench --model / http_code 拼接）"
  else
    bad "issue 跟进修复回归失败："
    printf '%s' "$out" | grep -E '✗|FAIL' | sed 's/^/        /' | head -10
  fi
else
  bad "缺 tools/selftest_issue_followups.sh（无法自动抓本轮这几处回归）"
fi

# ------------------------------------------------- 9g) 证据收集器的**沙箱自测**
# 收集器是"出问题时唯一救命的工具"，而它的入口有三条（SERVE_LOG / RUN_DIR / RUN_ID+自动取最新）。
# 入口写错的表现是**静默抽错文件**（抽到别人的臂），在真出问题时最贵。
if [ -f a2/scripts/selftest_collect_evidence.sh ]; then
  if out=$(bash a2/scripts/selftest_collect_evidence.sh 2>&1); then
    n=$(printf '%s' "$out" | grep -c 'PASS' || true)
    ok "证据收集器沙箱自测：${n:-?} 条全过（指定 log 位置 / 自动取最新 / 负例 rc=64）"
  else
    bad "证据收集器沙箱自测失败（指定 log 位置 / 计数助手 / 负例）："
    printf '%s' "$out" | grep -E 'FAIL' | sed 's/^/        /' | head -8
  fi
else
  warn "缺 a2/scripts/selftest_collect_evidence.sh（无法自动抓"抽错日志"这类静默错误）"
fi

echo
if [ "$fail" = "0" ]; then
  echo "[selfcheck] 全部通过 ✅  可以开始：bash scripts/build_image.sh"
else
  echo "[selfcheck] 有 $fail 项失败 ❌  先修再跑" >&2
fi
exit "$fail"
