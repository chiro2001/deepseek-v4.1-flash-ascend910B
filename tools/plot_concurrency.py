#!/usr/bin/env python3
"""把 bench_concurrency.py 产出的 JSON 画成并发吞吐曲线图。

用法::

    python3 tools/plot_concurrency.py results/bench/conc_dihuo.json -o docs/img
    # 同时给多份结果叠加对比：
    python3 tools/plot_concurrency.py a.json b.json --labels "A3" "A2" -o docs/img

只依赖 matplotlib。画两联图：
  左：单流吞吐 + 总吞吐 vs 并发（双 y 轴）
  右：接受长度 与 单流效率 vs 并发（判断"是不是在真干活"）
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description="并发吞吐曲线")
    ap.add_argument("json", nargs="+", help="bench_concurrency.py 的输出 JSON（可多个）")
    ap.add_argument("--labels", nargs="*", default=None, help="每条曲线的名字")
    ap.add_argument("-o", "--outdir", default="docs/img", help="输出目录")
    ap.add_argument("--prefix", default="concurrency", help="输出文件名前缀")
    ap.add_argument("--title", default="", help="图标题")
    ap.add_argument("--lang", choices=["auto", "zh", "en"], default="auto",
                    help="标签语言（auto=有中文字体就用中文）")
    a = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError:
        print("需要 matplotlib：pip install matplotlib", file=sys.stderr)
        return 3

    # 中文字体自动检测（找不到就退回英文标签，避免出豆腐块）
    cjk = None
    for fam in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans SC",
                "WenQuanYi Zen Hei", "Source Han Sans SC", "SimHei"):
        try:
            if font_manager.findfont(fam, fallback_to_default=False):
                cjk = fam
                break
        except Exception:  # noqa: BLE001
            continue
    if a.lang == "en":
        cjk = None
    elif a.lang == "zh" and not cjk:
        print("[plot] 警告：系统无中文字体，仍用英文标签", file=sys.stderr)
    if cjk:
        matplotlib.rcParams["font.family"] = cjk
        matplotlib.rcParams["axes.unicode_minus"] = False
    L = ({"per": "单流", "agg": "总吞吐", "acc": "接受长度", "eff": "单流效率",
          "x": "并发请求数", "y1": "吞吐 (tokens/s)", "t1": "吞吐曲线：单流 vs 总吞吐",
          "y2l": "接受长度 (tok/step)", "y2r": "单流效率 (%)",
          "t2": "自检：投机接受长度（左轴）与单流效率（右轴）",
          "warn": "> 3.5 需警惕复读退化"} if cjk else
         {"per": "per-stream", "agg": "aggregate", "acc": "accept length",
          "eff": "per-stream eff. %", "x": "Concurrent requests",
          "y1": "Throughput (tokens/s)", "t1": "Throughput: per-stream vs aggregate",
          "y2l": "Accept length (tok/step)", "y2r": "Per-stream efficiency (%)",
          "t2": "Sanity: speculative accept length & per-stream efficiency",
          "warn": "> 3.5 : suspect degenerate repetition"})

    runs = [load(p) for p in a.json]
    labels = a.labels or [os.path.basename(p).rsplit(".", 1)[0] for p in a.json]
    if len(labels) < len(runs):
        labels += [f"run{i}" for i in range(len(labels), len(runs))]

    os.makedirs(a.outdir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    # 右图用**双 y 轴**：接受长度(左轴, ~2-5) 与 单流效率(右轴, 0-100%) 量纲差 20 倍，
    # 共用一个轴会把效率曲线压成贴边直线，看不出趋势。
    ax2r = ax2.twinx()
    h_acc, h_eff = [], []

    for i, (d, lab) in enumerate(zip(runs, labels)):
        rows = d["rows"]
        c = colors[i % len(colors)]
        x = [r["conc"] for r in rows]
        per = [r["per_stream_med"] for r in rows]
        tot = [r["total_decode"] for r in rows]

        ax1.plot(x, per, "o-", color=c, lw=2, label=f"{lab} · {L['per']}")
        ax1.plot(x, tot, "s--", color=c, lw=2, alpha=0.65, label=f"{lab} · {L['agg']}")

        acc = [r.get("accept_len", 0) for r in rows]
        eff = [r["per_stream_med"] / rows[0]["per_stream_med"] * 100 for r in rows]
        h1, = ax2.plot(x, acc, "o-", color=c, lw=2, label=f"{lab} · {L['acc']}")
        h2, = ax2r.plot(x, eff, "s--", color=c, lw=2, alpha=0.55,
                        label=f"{lab} · {L['eff']}")
        h_acc.append(h1)
        h_eff.append(h2)

        for xi, yi in zip(x, per):
            ax1.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points",
                         xytext=(0, 7), ha="center", fontsize=8, color=c)
        for xi, yi in zip(x, tot):
            ax1.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points",
                         xytext=(0, -13), ha="center", fontsize=8, color=c, alpha=0.8)
        for xi, yi in zip(x, acc):
            ax2.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                         xytext=(0, 7), ha="center", fontsize=7.5, color=c)
        for xi, yi in zip(x, eff):
            ax2r.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points",
                          xytext=(0, -13), ha="center", fontsize=7.5, color=c, alpha=0.75)

    ax1.set_xscale("log", base=2)
    ax1.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax1.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax1.set_xlabel(L["x"])
    ax1.set_ylabel(L["y1"])
    ax1.set_title(L["t1"])
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=8, ncol=1)

    ax2.set_xscale("log", base=2)
    ax2.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax2.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax2.set_xlabel(L["x"])
    ax2.set_ylabel(L["y2l"])
    ax2r.set_ylabel(L["y2r"])
    ax2.set_title(L["t2"])
    ax2.grid(alpha=0.3, which="both")
    # 接受长度的合理区间参考带：> 3.5 通常意味着复读退化
    ax2.axhline(3.5, color="gray", ls=":", lw=1)
    ax2.annotate(L["warn"], (1.02, 3.58), fontsize=7, color="gray")
    ax2.set_ylim(bottom=0)
    ax2r.set_ylim(0, 105)
    # 左右两轴各自对齐到不同刻度，明确"折线属于哪个轴"
    ax2.tick_params(axis="y", colors="#444444")
    ax2r.tick_params(axis="y", colors="#444444")
    ax2.legend(handles=h_acc + h_eff, fontsize=8, ncol=1, loc="center left")

    if a.title:
        fig.suptitle(a.title, fontsize=13)
    fig.tight_layout()
    png = os.path.join(a.outdir, f"{a.prefix}.png")
    fig.savefig(png, dpi=150)
    print(f"[plot] 已写 {png}")

    # 另外存一份 csv，方便贴到文档/表格
    csv = os.path.join(a.outdir, f"{a.prefix}.csv")
    with open(csv, "w", encoding="utf-8") as fh:
        fh.write("concurrency,per_stream_tok_s,total_tok_s,speedup,"
                 "per_stream_efficiency_pct,accept_len,accept_rate_pct,ttft_s\n")
        for d in runs:
            rows = d["rows"]
            base_tot = rows[0]["total_decode"]
            base_per = rows[0]["per_stream_med"]
            for r in rows:
                fh.write(f"{r['conc']:g},{r['per_stream_med']:.1f},{r['total_decode']:.1f},"
                         f"{r['total_decode']/base_tot:.2f},"
                         f"{r['per_stream_med']/base_per*100:.1f},"
                         f"{r.get('accept_len',0):.2f},{r.get('accept_rate',0):.1f},"
                         f"{r['ttft_med']:.2f}\n")
    print(f"[plot] 已写 {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
