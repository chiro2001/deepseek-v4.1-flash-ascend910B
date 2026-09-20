#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_checksums.py —— 发布包「补丁载荷 / 落位表 / md5 清单」三方一致性检查（10 秒，不需要 docker）。

    python3 tools/check_checksums.py                    # 检查，0=一致 1=不一致 2=用法错误
    python3 tools/check_checksums.py --emit-chk         # 打印每个镜像内路径的期望 md5（人工审计用）
    python3 tools/check_checksums.py --manifest F.tsv   # 写出 inst/newf 落位清单（镜像内校验用）
    python3 tools/check_checksums.py --strict-others    # 连"非运行时载荷"的清单也要求一致

## 为什么需要（真实故障，已发生两次）

`scripts/build_image.sh` 的镜像内自检 `chk()` 曾**硬编码**每份补丁的 md5。
v7 → v8 改了 `engram_hbm.py` / `model.py`、新增 `engram_device_index.py` /
`engram_graph.py`，但没人更新那张表 ⇒ 用户 `bash scripts/build_image.sh`
跑完 10–20 分钟，在**最后一步**报：

    FAIL md5 models/deepseek_v41/model.py: got=d22eec4c… want=5b7c4526…

即「烘焙进镜像的字节是对的，校验和清单是陈旧的」。同一张表还漏了 2 个文件
（v8 新增的两个）——所以即使把 md5 改对，仍会有文件"装了但没人校验"。

## 设计：三个唯一真相，其余全部推导

  | 事实 | 唯一权威来源 |
  |---|---|
  | 装到镜像的**哪些位置** | `Dockerfile` 的 `inst`/`newf` 落位表 |
  | 载荷**内容**是什么 | `patches/files/**` 的实际字节 |
  | 期望 md5 是**多少** | `patches/MD5SUMS`（对 `patches/files/**` 的公开快照） |

  其余一切都是推导出来的，不存在需要手工同步的第二份清单：

  * `tools/check_checksums.sh` 用本脚本在三者之间做交叉校验（已进 `tools/selfcheck_pkg.sh`）；
  * `scripts/build_image.sh` 用 `--manifest` 拿到「镜像内路径 + 期望 md5」，
    交给 `tools/verify_baked_tree.sh` 在镜像里逐条核对 —— 期望值由**载荷字节直接算出**，
    因此"改了文件忘了改校验和"这一类错误在结构上不可能再发生。

## 退出码
  0 = 三方一致；1 = 有不一致 / 缺项 / 漏项；2 = 用法或环境错误
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

# Dockerfile 续行链里的落位指令，形如：
#     inst engram_hbm.py                models/deepseek_v41/engram_hbm.py; \
#     newf engram_device_index.py       models/deepseek_v41/engram_device_index.py; \
_INST_RE = re.compile(r"^\s*(inst|newf)\s+(\S+)\s+(\S+)\s*$")
_COPY_RE = re.compile(r"^\s*COPY\s+patches/files/(\S*)\s+(\S+)\s*$")

# 实验/编辑残留 —— **不是发布载荷**，因此不要求出现在 MD5SUMS 里（只报 NOTE）。
# 判据：Python 不会 import 这些名字（后缀不是 .py），它们也不该被 COPY/inst 引用。
# 注意：不要往这里加 `*.py`，否则会把真正的载荷漏掉。
_ARTIFACT_RE = re.compile(r"(\.bak|\.bak-.*|\.orig|\.rej|~|\.swp|\.swo|\.DS_Store|#.*#)$")


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _strip_continuations(line: str) -> str:
    """`inst a b; \\` → `inst a b;`（去掉续行符与行尾空白）"""
    line = line.rstrip()
    while line.endswith("\\"):
        line = line[:-1].rstrip()
    return line


def _strip_terminator(token: str) -> str:
    """`models/x/model.py;` → `models/x/model.py`（去掉 shell 语句终结符）"""
    return token.rstrip(";\\").strip()


def parse_md5sums(path: str) -> tuple[dict[str, str], list[str]]:
    """返回 (相对路径 -> md5, 错误列表)；兼容 `md5sum` 的 `<md5>  <path>` 与 `<md5> *<path>`。"""
    sums: dict[str, str] = {}
    errors: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = re.match(r"^([0-9a-fA-F]{32})[ \t]+\*?(.+?)\s*$", line)
            if not m:
                errors.append(f"{path}:{lineno} 无法解析：{line!r}")
                continue
            digest, rel = m.group(1).lower(), m.group(2).strip()
            if rel.startswith("./"):
                rel = rel[2:]
            if rel in sums and sums[rel] != digest:
                errors.append(f"{path}:{lineno} 同一路径出现两个不同 md5：{rel}")
            sums[rel] = digest
    return sums, errors


def parse_dockerfile(path: str) -> tuple[list[tuple[str, str, str]], set[str], dict[str, str], list[str]]:
    """返回 ([(kind, 中间名, dst), ...], 仅随包发布的载荷（文件或目录）, /tmp/bake 改名映射, 错误列表)。

    `inst` = 覆盖已有文件（先备份 `.a2orig`）；`newf` = 新增文件（无备份）。
    「仅随包发布」= `COPY patches/files/<sub>/ /opt/dsv41/patches/...`：镜像里待命、
    **不进运行时**的实验项 —— 不必出现在 inst/newf 里，但必须真被 COPY。

    `inst`/`newf` 的第一个参数是 `/tmp/bake/` 下的**中间文件名**，可能与载荷名不同
    （现有例子：`COPY patches/files/token_dispatcher_moemask.py /tmp/bake/token_dispatcher.py`
    再 `inst token_dispatcher.py ...`）。改名映射从 COPY 指令推导，避免手工维护。
    """
    entries: list[tuple[str, str, str]] = []
    staged: set[str] = set()
    bake_map: dict[str, str] = {}
    errors: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            if raw.lstrip().startswith("#"):
                continue
            line = _strip_continuations(raw.rstrip("\n"))
            c = _COPY_RE.match(line)
            if c:
                sub, dest = c.group(1).rstrip("/"), c.group(2).strip()
                if sub and "/opt/dsv41/patches" in dest:
                    staged.add(sub)
                elif sub and "/tmp/bake" in dest:
                    bake_map[os.path.basename(dest)] = sub
                continue
            m = _INST_RE.match(line)
            if m:
                kind, src, dst = m.group(1), m.group(2), _strip_terminator(m.group(3))
                if not src.endswith(".py"):
                    errors.append(f"{path}:{lineno} 落位项载荷不是已知类型：{src}")
                entries.append((kind, src, dst))
    return entries, staged, bake_map, errors


def main(argv: list[str] | None = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="Dockerfile 落位表 / patches/files / MD5SUMS 三方一致性检查")
    ap.add_argument("--dockerfile", default=os.path.join(pkg, "Dockerfile"))
    ap.add_argument("--payload", default=os.path.join(pkg, "patches", "files"),
                    help="补丁载荷目录（权威字节）")
    ap.add_argument("--sums", default=os.path.join(pkg, "patches", "MD5SUMS"),
                    help="公开 md5 清单（对载荷的冻结快照）")
    ap.add_argument("--series-sums", default=os.path.join(pkg, "patches", "vllm-ascend", "MD5SUMS"),
                    help="patch 系列的 md5 清单（按镜像内目标路径命名，用于交叉核对）")
    ap.add_argument("--manifest", help="写出镜像内校验清单 TSV：kind<TAB>目标路径<TAB>md5")
    ap.add_argument("--materialize", metavar="DIR",
                    help="按落位表把载荷铺成『模拟镜像树』（供 tools/verify_baked_tree.sh 干跑，"
                         "不需要 docker；inst 项会生成 .a2orig 占位文件）")
    ap.add_argument("--emit-chk", action="store_true", help="打印每个镜像内路径的期望 md5（审计用）")
    ap.add_argument("--strict-others", action="store_true",
                    help="非运行时载荷（draft/、PGO 产物等）的 md5 不一致也判 FAIL")
    ap.add_argument("--strict-artifacts", action="store_true",
                    help="连实验残留（*.bak-probe 等）也要求登记进 MD5SUMS")
    ap.add_argument("--quiet-ok", action="store_true", help="全部通过时不逐条打印 OK")
    args = ap.parse_args(argv)

    problems: list[str] = []
    notes: list[str] = []
    warns: list[str] = []

    for path, what in ((args.dockerfile, "Dockerfile"), (args.sums, "MD5SUMS")):
        if not os.path.isfile(path):
            print(f"[chk][FAIL] 找不到 {what}：{path}", file=sys.stderr)
            return 2
    if not os.path.isdir(args.payload):
        print(f"[chk][FAIL] 找不到载荷目录：{args.payload}", file=sys.stderr)
        return 2

    entries, staged, bake_map, errs = parse_dockerfile(args.dockerfile)
    problems += errs
    sums, errs = parse_md5sums(args.sums)
    problems += errs
    series = {}
    if os.path.isfile(args.series_sums):
        series, errs = parse_md5sums(args.series_sums)
        problems += errs
    if not entries:
        problems.append(f"{args.dockerfile} 里没有解析到任何 inst/newf 落位项（正则失配？）")
    if not sums:
        problems.append(f"{args.sums} 里没有解析到任何 md5（格式变了？）")

    # ---------- 1) 落位表 → 载荷实际字节 → 两条 md5 清单 ----------
    rows: list[tuple[str, str, str, str]] = []   # kind, src, dst, md5(载荷实际字节)
    dst_to_src: dict[str, str] = {}
    # 载荷名 != 中间名时（Dockerfile 里 COPY 到 /tmp/bake/ 时改了名），按改名映射还原
    RUNTIME_MOUNTED = {
        # 由 serve_a2.sh 的门控在**运行时以 host 挂载**方式提供的实验载荷（不进镜像）
        "token_dispatcher_moennf.py": "serve_a2.sh 的 MOE_NF=1 分支按需挂载（host 侧 patches/files/）",
    }
    for kind, mid, dst in entries:
        src = bake_map.get(mid, mid)
        if dst in dst_to_src:
            notes.append(f"落位表里 {dst} 出现两次（src={dst_to_src[dst]} 与 {src}）")
        dst_to_src[dst] = src
        src_path = os.path.join(args.payload, src)
        if not os.path.isfile(src_path):
            problems.append(f"MISSING 载荷：{os.path.relpath(src_path, pkg)}"
                            f"（Dockerfile 中间名 {mid} 要装成 {dst}）")
            continue
        digest = md5_file(src_path)
        rows.append((kind, src, dst, digest))
        want = sums.get(src, "")
        if not want:
            problems.append(f"{os.path.relpath(args.sums, pkg)} 缺项：{src}"
                            f"（镜像内 {dst}，载荷 md5={digest}）")
        elif want != digest:
            problems.append(f"STALE {os.path.relpath(args.sums, pkg)}：{src} "
                            f"清单={want} 载荷={digest}")
        # 系列清单按镜像内目标路径命名（vllm_ascend/...）——用落位表反查
        srow = series.get("vllm_ascend/" + dst, "")
        if not series:
            pass
        elif not srow:
            warns.append(f"{os.path.relpath(args.series_sums, pkg)} 未收录 vllm_ascend/{dst}")
        elif srow != digest:
            problems.append(f"STALE {os.path.relpath(args.series_sums, pkg)}：vllm_ascend/{dst} "
                            f"清单={srow} 载荷={digest}")

    # ---------- 2) 载荷里每个文件都应"有归属 + 有校验和" ----------
    in_docker = {bake_map.get(mid, mid) for _k, mid, _d in entries}
    stage_docs: list[str] = []
    artifacts: list[str] = []
    payload_files: list[str] = []
    for root, dirs, files in os.walk(args.payload):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in files:
            if name.endswith((".pyc", ".pyo")):
                continue
            rel = os.path.relpath(os.path.join(root, name), args.payload).replace(os.sep, "/")
            payload_files.append(rel)
    for rel in sorted(payload_files):
        if _ARTIFACT_RE.search(rel) and not args.strict_artifacts:
            # 部署副本上常见（并行调试留下的 *.bak-probe）；它不是发布载荷，不参与校验
            artifacts.append(rel)
            continue
        if rel not in in_docker:
            parent = rel.split("/", 1)[0] if "/" in rel else ""
            if rel in staged or (parent and parent in staged) or not rel.endswith(".py"):
                stage_docs.append(rel)
            elif rel in RUNTIME_MOUNTED:
                notes.append(f"运行时按需挂载（{RUNTIME_MOUNTED[rel]}）：{rel}")
            else:
                problems.append(f"孤儿载荷：{rel} 既没被 inst/newf 烘焙，也没被 COPY 随包发布")
        if rel not in sums:
            problems.append(f"{os.path.relpath(args.sums, pkg)} 缺项（新增/改动的载荷必须同步）：{rel}")

    # ---------- 3) 非运行时行（draft/、PGO 产物…）：默认 WARN，--strict-others 则 FAIL ----------
    anchors = [args.payload, os.path.dirname(os.path.abspath(args.sums)),
               os.path.join(pkg, "scripts"), os.path.join(pkg, "optim", "pgo")]
    for rel, want in sorted(sums.items()):
        if rel in in_docker:
            continue
        hit = next((os.path.join(a, rel) for a in anchors if os.path.isfile(os.path.join(a, rel))), "")
        if not hit:
            if rel in {"python3", "libpython3.12.so.1.0"}:
                notes.append(f"按需生成（本包不含，属正常）：{rel}")
            else:
                notes.append(f"清单里有、本包内找不到（历史/可选产物）：{rel}")
            continue
        got = md5_file(hit)
        if got != want:
            msg = f"STALE {os.path.relpath(args.sums, pkg)}：{rel} 清单={want} 实际={got}"
            (problems if args.strict_others else warns).append(msg)

    # ---------- 4) 产出 ----------
    if not args.quiet_ok:
        for kind, src, dst, digest in rows:
            print(f"  {kind:4s} {dst:52s} {digest}  <- {src}")
    if args.emit_chk:
        print()
        for kind, _src, dst, digest in rows:
            print(f"  {kind:4s} {dst:52s} {digest}")
    if args.manifest:
        with open(args.manifest, "w", encoding="utf-8") as fh:
            for kind, _src, dst, digest in rows:
                fh.write(f"{kind}\t{dst}\t{digest}\n")
        notes.append(f"已写出镜像内校验清单 {args.manifest}（{len(rows)} 项）")
    if args.materialize:
        import shutil
        for kind, src, dst, _digest in rows:
            tgt = os.path.join(args.materialize, dst)
            os.makedirs(os.path.dirname(tgt), exist_ok=True)
            shutil.copyfile(os.path.join(args.payload, src), tgt)
            if kind == "inst":
                # 真镜像里这里是**基础镜像的原文件**；干跑树只要求"存在"，
                # 所以放一份同内容占位（verify_baked_tree 只断言 .a2orig 存在）。
                shutil.copyfile(tgt, tgt + ".a2orig")
        notes.append(f"已按落位表铺出模拟镜像树 {args.materialize}（{len(rows)} 项，"
                     "仅用于不需要 docker 的干跑验证）")

    print()
    print(f"[chk] 落位表 {len(entries)} 项（{sum(1 for k, *_ in entries if k == 'inst')} inst / "
          f"{sum(1 for k, *_ in entries if k == 'newf')} newf）"
          f" / 载荷 {len(payload_files)} 个文件 / MD5SUMS {len(sums)} 条")
    if stage_docs:
        print(f"[chk][NOTE] 仅随包发布、不烘焙进运行时（{len(stage_docs)}）："
              + "、".join(stage_docs) )
    if artifacts:
        print(f"[chk][NOTE] 实验/编辑残留（既非载荷也不进镜像，已忽略；--strict-artifacts 可强制检查）："
              + "、".join(artifacts))
    for n in notes:
        print(f"[chk][NOTE] {n}")
    for w in warns:
        print(f"[chk][WARN] {w}")
    if problems:
        print(f"[chk][FAIL] {len(problems)} 个问题 ❌")
        for p in problems:
            print(f"    - {p}")
        print("  ⇒ 修法：改了 `patches/files/**` 之后，把上面对应的 md5 行同步到清单里")
        return 1
    print("[chk] 三方一致 ✅（期望 md5 全部由载荷字节推导，无手工清单）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
