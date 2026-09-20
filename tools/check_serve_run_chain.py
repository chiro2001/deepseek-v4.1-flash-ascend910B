#!/usr/bin/env python3
"""检查 `serve_a2.sh` 里 `docker run` 的**续行参数链有没有被注释/空行打断**。

为什么需要这个检查
------------------
2026-09-20 在 A2 真机上炸过一次真实故障：

    "docker run" requires at least 1 argument.
    See 'docker run --help'.
    .../serve_a2.sh: line 1066: -e: command not found
    [serve_a2][FAIL] docker run 失败

根因：一段**注释块被插进了 docker run 的续行链中间** ——

    ...                                         <- 续行（行尾反斜杠）
      -e DSPARK_HOIST_CONTEXT_KV="...:-0}"      <- 这一行也以反斜杠结尾
      # [DRAFT-FOUR-PIECE] 说明……               <- ★ 注释：续行在这里**终止**
      # ...（后面整段注释都被当成新命令）
      -e DSPARK_CAPTURE_VALUE_FIX="...:-1}"     <- 于是这行成了**独立命令**，-e 找不到
      ...
      IMAGE bash -lc "..."

由于 shell 在**执行期**才把续行拼起来，而拼接结果**语法完全合法**（`docker run ... -e X`
缺少 IMAGE 参数不是语法错误），所以：

  * `bash -n` ✅ 通过（抓不到）
  * `tools/check_dockerfile.py` ✅ 通过（它只查 Dockerfile）
  * `tools/selfcheck_pkg.sh` ✅ 通过（它只跑 `bash -n`）
  * **起服时才炸**，而且报错信息（`requires at least 1 argument` / `-e: command not found`）
    指向的是 docker 和 `-e`，**不指向注释** —— 排查成本很高。

本脚本把这类错变成**秒级、离线的静态检查**。

检查规则
--------
1. 找到 `$DOCKER run`（或 `docker run`）所在行，向后扫描它的**续行链**；
2. 链内每一行都必须是「以反斜杠结尾的续行」或「链的最后一行（命令真正的结尾）」；
3. 链内**不允许**出现空行或以 `#` 开头的行（在**去掉前导空白后**判断 —— `  # ...`
   在续行链里同样会中断它）。
4. 链的结尾那一行必须能被解析出 **IMAGE 参数**（作为一个独立 token 出现，
   且在该行的 `-e/-v/--xxx` 之外）—— 这是本次故障的直接判据。

用法：`python3 tools/check_serve_run_chain.py [scripts/serve_a2.sh ...]`
退出码：0 通过；1 发现问题；2 脚本本身出错。
"""

import re
import sys
from pathlib import Path

DEFAULT_TARGETS = ["scripts/serve_a2.sh", "scripts/serve_a3.sh", "scripts/serve_v2.sh"]


def find_docker_run(lines):
    """返回 [(行号(0基), 命令行文本)]，匹配 `$DOCKER run` / `${DOCKER} run` / `docker run`。"""
    out = []
    pat = re.compile(r"^\s*(?:\$\{?DOCKER\}?|\"\{\}DOCKER\"|docker)\s+run\b")
    for i, l in enumerate(lines):
        if pat.match(l):
            out.append((i, l))
    return out


def check_chain(path, lines, start):
    """检查从 start 开始的续行链。返回 (结束行号, [错误信息])。"""
    errs = []
    i = start
    chain = []
    while True:
        if i >= len(lines):
            errs.append(f"第 {start + 1} 行起的续行链未闭合（文件结束）")
            return i, errs
        raw = lines[i]
        stripped = raw.strip()
        # ★ 在续行链里出现的注释 / 空行 = 故障点
        if i > start:
            if stripped == "":
                errs.append(
                    f"第 {i + 1} 行：续行链里的**空行** —— 它会中断链（上一行以 `\\` 结尾）"
                )
                return i, errs
            if stripped.startswith("#"):
                errs.append(
                    f"第 {i + 1} 行：续行链里的**注释** —— 它会中断链、"
                    f"把后续 `-e/-v` 变成独立命令。原文：{stripped[:70]}"
                )
                return i, errs
        chain.append(raw)
        if not raw.rstrip().endswith("\\"):
            break
        i += 1
    return i, errs


def extract_image(chain):
    """从续行链里粗提 IMAGE：最后一个不含 `-`/`=` 的裸 token（启发式，够用）。"""
    text = " ".join(l.rstrip("\\").strip() for l in chain)
    toks = text.split()
    # 去掉 shell 变量引用与带 = 的 kv
    cands = [t for t in toks if not t.startswith("-") and "=" not in t
             and not t.startswith("$") and not t.startswith('"$')]
    return cands[-1] if cands else None


def check_file(path):
    p = Path(path)
    if not p.is_file():
        return None  # 未提供的脚本不算错（serve_a3.sh 可能只是 wrapper）
    lines = p.read_text(encoding="utf-8").splitlines()
    runs = find_docker_run(lines)
    if not runs:
        return f"{path}: 未找到 `docker run`（若该脚本本就不起容器则忽略）"
    errs = []
    for start, _ in runs:
        end, e = check_chain(path, lines, start)
        if e:
            errs += [f"{path}:{m}" for m in e]
            continue
        img = extract_image(lines[start:end + 1])
        if img is None:
            errs.append(
                f"{path}:{start + 1}: 续行链（{start + 1}–{end + 1} 行）里**找不到 IMAGE 参数** "
                f"⇒ `docker run` 会报 'requires at least 1 argument'"
            )
    return "\n".join(errs) if errs else None


def main(argv):
    targets = argv[1:] or DEFAULT_TARGETS
    try:
        problems = []
        checked = 0
        for t in targets:
            r = check_file(t)
            if r is None:
                continue
            checked += 1
            if r.startswith(("scripts/",)) and "未找到" in r:
                print(f"[run-chain] WARN {r}")
                continue
            problems.append(r)
        if problems:
            print("[run-chain] FAIL ❌", file=sys.stderr)
            for m in problems:
                print(f"  {m}", file=sys.stderr)
            print(
                "\n  修法：把注释/空行移到 `$DOCKER run` **之前**。"
                "\n  （`bash -n` 抓不到这类错；这就是本检查存在的理由。）",
                file=sys.stderr,
            )
            return 1
        print(f"[run-chain] 通过 ✅（检查了 {checked} 个脚本的 `docker run` 续行链）")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[run-chain] 检查器自身出错：{exc!r}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
