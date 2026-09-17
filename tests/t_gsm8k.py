#!/usr/bin/env python3
"""GSM8K 薄封装：调用 acc_eval.py，归一出 {acc} 供报告使用。需要本地已有数据集缓存。

⚠️ **口径警告（v4 新增）**：本脚本用 `--conc 4`，而 a3 上跑历史 GSM8K 时还额外带了
`--serialize-prefill 1` —— 该开关的语义（见 `acc_eval.py` 的 help 原文）是
**"hold a global lock until first token (avoid concurrent prefills)"**，
即**只有 decode 并发，prefill 被串行化**。
⇒ 本脚本的通过**不能**证明"并发 batch"或"prefill+decode 混合"正确。
要覆盖那些形态，必须额外跑 `tests/multibatch/`（见 `EXPECTED_PERF.md` §7）。
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_enc_dir(explicit=None):
    """找官方 `encoding` 目录（GSM8K/C-Eval 必须用它做 chat 模板）。"""
    cands = [explicit, os.environ.get("ENC_DIR"), str(HERE / "encoding"),
             os.path.expanduser("~/models/DeepSeek-V4.1-Flash/encoding")]
    for c in cands:
        if c and Path(c).is_dir():
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--enc-dir", default=None, help="官方 encoding 目录（缺则用 ENC_DIR / ~/models/...）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    enc = find_enc_dir(a.enc_dir)
    if not enc:
        print("[t_gsm8k][SKIP] 找不到官方 encoding 目录（GSM8K 必须要它）。")
        print("           用 `--enc-dir /path/to/DeepSeek-V4.1-Flash/encoding` 或 export ENC_DIR=... 指定。")
        Path(a.out).write_text(json.dumps(
            {"correct": None, "total": None, "acc": None, "skipped": "no encoding dir"},
            ensure_ascii=False, indent=2))
        return 0

    cmd = [sys.executable, str(HERE / "acc_eval.py"),
           "--task", "gsm8k", "--limit", str(a.limit), "--mode", "chat",
           "--conc", str(a.conc), "--base", a.base, "--enc-dir", enc,
           "--out", a.out + ".raw.json"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    txt = (p.stdout or "") + "\n" + (p.stderr or "")
    print(txt[-3000:])
    m = re.search(r"(\d+)\s*/\s*(\d+)\s*=\s*([0-9.]+)%", txt)
    out = {
        "correct": int(m.group(1)) if m else None,
        "total": int(m.group(2)) if m else None,
        "acc": (float(m.group(3)) / 100.0) if m else None,
        "returncode": p.returncode,
    }
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[t_gsm8k] {out}")
    return 0 if out["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
