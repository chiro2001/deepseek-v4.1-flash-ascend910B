#!/usr/bin/env bash
# =============================================================================
# model_mount_args.sh -- 为"软链构造的模型目录"生成完整的 docker -v 参数。
#
#   bash tools/model_mount_args.sh /path/to/final-model-dir
#   -> 一行一个 "-v <dir>:<dir>:ro"（可直接喂给 docker run）
#
# ## 为什么需要它（真实故障）
#
# 量化流水线最终产出的目录是**软链构造**的（零拷贝），而且软链是**绝对路径**：
#
#   v41-...-qrot-mtpq/                      <- L5 最终目录（94 个软链 + 1 个实体）
#     config.json -> /abs/.../v41-...-mtpq/config.json
#                        v41-...-mtpq/      <- L4（93 软链 + 2 实体）
#                          config.json -> /abs/.../v41-...-vision/config.json
#                            v41-...-vision/     <- L3
#                              ...
#                                v41-w4a8-stage1/   <- L1，真正的 87 个实体分片
#
# 而 `docker run -v "$MODEL:$MODEL:ro"` **只挂了 L5**。容器里那些绝对路径软链
# 全部悬空 -> 宿主机上一切正常，进容器立刻 `No such file or directory`
# （第一个报错通常是 config.json / tokenizer 加载失败）。
#
# 实测复现：
#   docker run --rm -v <leaf>:/m:ro alpine cat /m/config.json
#   -> cat: can't open '/m/config.json': No such file or directory
#
# 本脚本把**软链指向的每一层目录**都解析出来，逐层挂载，使容器内路径与宿主一致。
# 链条深度不限（实测 5 层），并去重。
#
# 退出码：0 正常；1 存在悬空软链（起服必然失败，应立刻修）
# =============================================================================
set -uo pipefail

MODEL=${1:-}
[ -n "$MODEL" ] || { echo "用法: bash model_mount_args.sh <模型目录>" >&2; exit 2; }
MODEL=$(readlink -f "$MODEL" 2>/dev/null || echo "$MODEL")
[ -d "$MODEL" ] || { echo "不是目录: $MODEL" >&2; exit 2; }

MODEL="$MODEL" python3 - <<'PYEOF'
import os
import sys

model = os.environ["MODEL"]
MAX_DIRS = 64          # 安全上限
MAX_HOPS = 16          # 单条软链最多追几跳

# ---------------------------------------------------------------------------
# 关键：必须按**字面软链**（readlink）逐跳解析，不能用 os.path.realpath()。
# realpath 会把 L5 -> L4 -> L3 -> L2 -> L1 一次折叠成 L1，于是 L2/L3/L4 不会被
# 挂载；而**容器是逐跳解析的**：打开 /abs/L5/config.json 读出 "/abs/L4/..."，
# 再去开 /abs/L4/config.json —— L4 不在容器里就悬空。
# 所以每一跳的目标目录都必须挂。
# ---------------------------------------------------------------------------
dirs = {model}
broken = []
seen = set()


def literal_target(link):
    """软链的字面目标；相对软链按 '软链所在目录' 解析（与容器行为一致）。"""
    raw = os.readlink(link)
    if os.path.isabs(raw):
        return os.path.normpath(raw)
    return os.path.normpath(os.path.join(os.path.dirname(link), raw))


def scan(d, hops=0):
    """扫描目录 d 的软链，逐跳登记每个目标所在目录，并递归进新目录。"""
    if hops > MAX_HOPS or len(dirs) > MAX_DIRS:
        return
    try:
        entries = sorted(os.listdir(d))
    except OSError:
        return
    for name in entries:
        p = os.path.join(d, name)
        if not os.path.islink(p):
            # 真目录也进去看一层（应对 engram_int8/ 这类子目录软链）
            if os.path.isdir(p) and p not in seen:
                seen.add(p)
                scan(p, hops + 1)
            continue
        if p in seen:
            continue
        seen.add(p)

        # 逐跳走，把每一跳的目标目录都加进来
        cur = p
        for _ in range(MAX_HOPS):
            nxt = literal_target(cur)
            need = nxt if os.path.isdir(nxt) else os.path.dirname(nxt)
            if need and need not in dirs:
                dirs.add(need)
                scan(need, hops + 1)      # 新目录里可能还有软链
            if not os.path.exists(nxt):
                broken.append((cur, os.readlink(cur)))
                break
            if not os.path.islink(nxt):
                break                     # 追到实体了
            cur = nxt                     # 还是个软链 -> 继续追下一跳


scan(model)

# ---------------------------------------------------------------------------
# 输出格式：**每行一个裸路径**（不要带 "-v " 前缀）。
#
# 为什么：调用方用 `mapfile` 读这个输出，**每行只会成为数组的一个元素**。
# 如果这里输出 "-v /path:/path:ro"，那么一个元素就是整串
#     "-v /path:/path:ro"
# 展开给 docker 时它是**单个参数**，Go 的 pflag 会把 `-v` 后面的空格也算作值
# ⇒ docker 报错：
#     create  /path: " /path" includes invalid characters for a local volume name
# （注意错误信息里路径前面那个**空格**，就是这个 bug 的指纹）
# 实测于 A2：起服在 docker run 处直接失败。
#
# 所以这里只吐路径，由调用方组装成 `-v` 和 `路径:路径:ro` **两个**数组元素。
# ---------------------------------------------------------------------------
for d in sorted(dirs, key=lambda x: (x.count("/"), x)):
    print(d)

# 顺带给出"公共祖先"这条更省事的备选（挂一个目录覆盖全部）
if len(dirs) > 1:
    common = os.path.commonpath(sorted(dirs))
    print("", file=sys.stderr)
    print("[model_mount_args] 共 %d 个目录需要挂载（软链链条）" % len(dirs), file=sys.stderr)
    print("[model_mount_args] 备选（更省事但更宽）：%s" % common, file=sys.stderr)

if broken:
    print("", file=sys.stderr)
    print("[model_mount_args] FATAL: 发现 %d 个悬空软链（起服必然失败）:" % len(broken),
          file=sys.stderr)
    for b, tgt in broken[:10]:
        print("   %s -> %s" % (b, tgt), file=sys.stderr)
    sys.exit(1)
sys.exit(0)
PYEOF
rc=$?

if [ $rc -ne 0 ]; then
  echo "" >&2
  echo "[model_mount_args] 提示：悬空软链通常是量化装配时源目录被移动/删除，" >&2
  echo "                   或者软链被写成了相对路径（必须绝对路径）。" >&2
  exit 1
fi
