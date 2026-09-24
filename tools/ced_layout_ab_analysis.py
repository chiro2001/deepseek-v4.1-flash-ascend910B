#!/usr/bin/env python3
"""把「D 侧物理块形态」与「请求通过/失败」关联起来的判定工具。

用途：CED 的 1M 偶发乱码根因是 D 的 replay 读到自己未持有的 SWA 页，触发条件是
物理块分配碎片化（见 docs/CED-D-1M-LAYOUT-BUG-20260924.md）。做 `V41_CED_SWA_CLIP`
单变量 A/B 时，结论不能只看"通过率"，必须同时给出**块形态**，否则无法区分
「修复生效」与「这一轮恰好没碎片」。

输入（都来自既有工具，不需要新探针）：
  --probe-dir   `ced_seq_probe.py` / `ced_layer_trace_sequence.sh` 的落盘目录
                （读 `<tag>_NN.summary.json` 的 content / token_ids / usage）
  --serve-log   D 的 serve.log（`V41_CED_BLOCK_TRACE=1` 时含 `[CED-BLOCKS]` 行）
  --rank        取哪个 tp rank 的块形态（默认 0；各 rank 应当一致）

输出：逐请求表（结果 / g0 descents / 需要的逻辑页是否越界）+ 分组通过率 +
判定行。退出码 0 = 判定成立。

判定规则：
  * `descents == 0`（连续分配）那一组必须**全过**；
  * `descents > 0`（碎片）那一组的通过率必须与连续组**无显著差异**，才算修复生效；
  * 若碎片组仍有失败，输出这些失败请求的形态，便于继续定位。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

EXPECTED = "RB9N-6014"
# 前缀 `[CED-BLOCKS]` 允许缺失：既有的证据里有用 `sed` 去掉前缀后单独保存的块形态行。
BLOCK_RE = re.compile(
    r"(?:\[CED-BLOCKS\]\s+)?role=(?P<role>\S+)\s+tp=(?P<tp>\S+)\s+req=(?P<req>\S+)\s+"
    r"tail_page=(?P<tail>\d+)\s+(?P<extra>[^|]*)\|\s*(?P<groups>.*)$"
)
GROUP_RE = re.compile(r"g(?P<gid>\d+):\((?P<stats>[^)]*)\)")


def parse_summary(path: str) -> dict:
    data = json.load(open(path, encoding="utf-8"))
    usage = data.get("usage") or {}
    tokens = data.get("token_ids")
    content = data.get("content")
    return {
        "file": os.path.basename(path),
        "tag": os.path.basename(path).split(".")[0],
        "http_status": data.get("http_status"),
        "content": content,
        "content_repr": repr(content)[:48],
        "first_token_id": (tokens or [None])[0] if isinstance(tokens, list) else None,
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "passed": content is not None and EXPECTED in content,
    }


def parse_blocks(log_path: str, rank: str):
    """返回 [(req 短 id, descents_of_g0, 需要的逻辑页, 持有页), ...]，按出现顺序。"""
    rows = []
    if not os.path.isfile(log_path):
        return rows
    for line in open(log_path, encoding="utf-8", errors="replace"):
        if "role=" not in line or "tail_page=" not in line:
            continue
        match = BLOCK_RE.search(line)
        if not match or match.group("role") != "decode" or match.group("tp") != rank:
            continue
        groups = {int(g.group("gid")): g.group("stats") for g in GROUP_RE.finditer(match.group("groups"))}
        g0 = groups.get(0, "")
        descents = None
        match_d = re.search(r"descents=(\d+)", g0)
        if match_d:
            descents = int(match_d.group(1))
        at_page = re.search(r"at_page(\d+)=(\d+)", g0)
        pages = re.search(r"pages?=(\d+)", match.group("extra") or "")
        rows.append(
            {
                "req": match.group("req")[len("chatcmpl-"):][:8],
                "tail_page": int(match.group("tail")),
                "descents_g0": descents,
                "g0": g0,
                "groups": groups,
                "extra": (match.group("extra") or "").strip(),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--serve-log", required=True)
    ap.add_argument("--rank", default="0")
    ap.add_argument("--expected", default=EXPECTED)
    ap.add_argument("--json", default="")
    ap.add_argument("--allow-tail-align", action="store_true",
                    help="探针数与块形态行数不一致时，仍按尾部对齐（默认拒绝，见下）")
    args = ap.parse_args()

    summaries = sorted(glob.glob(os.path.join(args.probe_dir, "*.summary.json")))
    if not summaries:
        print(f"[layout-ab] 在 {args.probe_dir} 找不到 *.summary.json", file=sys.stderr)
        return 2
    probes = [parse_summary(p) for p in summaries]
    blocks = parse_blocks(args.serve_log, args.rank)

    # ★ 块形态行与探针必须**逐一对应**。数量不等说明有一侧被截断/被 reset/拷了一半，
    #   此时任何"尾部对齐"都会静默错配，可能给出"修复生效"的假结论
    #   （我第一版就是这样把 lt1 配到 lt3 的块形态上）。所以默认拒绝，必须显式放行。
    if len(blocks) != len(probes):
        message = (
            f"块形态行 {len(blocks)} 条 != 探针 {len(probes)} 条。两者必须一一对应，"
            f"否则无法把「块形态」与「结果」配对。请确认 serve.log 与该批探针来自同一次"
            f"D 实例、且都没有被截断；确实要按尾部对齐请显式加 --allow-tail-align。"
        )
        if not args.allow_tail_align:
            print(f"[layout-ab] 拒绝执行：{message}", file=sys.stderr)
            return 2
        print(f"[layout-ab] 警告（--allow-tail-align）：{message}", file=sys.stderr)
    offset = max(0, len(blocks) - len(probes)) if args.allow_tail_align else 0

    print(f"{'#':>3} {'tag':>14} {'result':>7} {'first_tok':>10} {'comp':>5} "
          f"{'descents':>9} {'g0':>28}")
    grouped = {True: [], False: []}
    for index, probe in enumerate(probes):
        block = blocks[offset + index] if offset + index < len(blocks) else None
        descents = block["descents_g0"] if block else None
        g0 = (block["g0"][:26] + "..") if block and len(block["g0"]) > 28 else (block["g0"] if block else "")
        verdict = "PASS" if probe["passed"] else "FAIL"
        first = probe["first_token_id"]
        if descents is not None:
            grouped[descents > 0].append(probe["passed"])
        print(f"{index + 1:>3} {probe['tag']:>14} {verdict:>7} {str(first):>10} "
              f"{str(probe['completion_tokens']):>5} {str(descents):>9} {g0:>28}")

    print()
    cont_pass, cont_tot = sum(grouped[False]), len(grouped[False])
    frag_pass, frag_tot = sum(grouped[True]), len(grouped[True])
    print(f"连续分配（descents==0）：{cont_pass}/{cont_tot} 通过")
    print(f"碎片分配（descents >0）：{frag_pass}/{frag_tot} 通过")

    failures = [p for p in probes if not p["passed"]]
    if failures:
        print("\n失败请求：")
        for probe in failures:
            idx = probes.index(probe)
            block = blocks[offset + idx] if offset + idx < len(blocks) else None
            print(f"  {probe['tag']}: content={probe['content_repr']} "
                  f"first_token={probe['first_token_id']} "
                  f"descents={block['descents_g0'] if block else '?'}")

    verdict_ok = cont_tot > 0 and cont_pass == cont_tot and frag_tot > 0 and frag_pass == frag_tot
    print()
    if frag_tot == 0:
        print("判定：本批**没有**碎片形态请求 ⇒ 对修复无判别力，需要重跑到出现 descents>0。")
    elif verdict_ok:
        print("判定：连续与碎片两组都全过 ✅ —— 修复在本批生效。")
    else:
        print("判定：碎片组仍有失败 ❌ —— 修复未完全生效，见上面失败清单。")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"probes": probes, "blocks": blocks,
                       "continuous": {"passed": cont_pass, "total": cont_tot},
                       "fragmented": {"passed": frag_pass, "total": frag_tot},
                       "verdict_ok": verdict_ok}, handle, ensure_ascii=False, indent=2)
        print(f"已写 {args.json}")
    return 0 if (verdict_ok or frag_tot == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
