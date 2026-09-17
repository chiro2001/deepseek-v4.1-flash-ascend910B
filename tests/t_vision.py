#!/usr/bin/env python3
"""视觉验收薄封装：调用 vision_accuracy_check.py，归一出 {cases, pass_n, verdict} 供报告使用。"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", default="deepseek-v41")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-hit-rate", default="0.80")
    a = ap.parse_args()

    cmd = [
        sys.executable, str(HERE / "vision_accuracy_check.py"),
        "--server", a.server, "--model", a.model,
        "--images-dir", a.images_dir,
        "--out", a.out + ".raw.json",
        "--min-hit-rate", a.min_hit_rate,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    txt = (p.stdout or "") + "\n" + (p.stderr or "")
    print(txt[-4000:])

    m = re.search(r"cases:\s*(\d+)\s+pass:\s*(\d+)", txt)
    verdict = "PASS" if re.search(r"verdict:\s*PASS", txt) else "FAIL"
    out = {
        "cases": int(m.group(1)) if m else None,
        "pass_n": int(m.group(2)) if m else None,
        "verdict": verdict,
        "returncode": p.returncode,
        "threshold_19_of_23": (int(m.group(2)) >= 19) if m else False,
    }
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[t_vision] {out}")
    return 0 if out["threshold_19_of_23"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
