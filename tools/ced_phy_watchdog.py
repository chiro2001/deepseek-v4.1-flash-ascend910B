#!/usr/bin/env python3
"""a3-22 单卡实验的只读看门狗：只负责在越界时停“我们自己的”容器。

安全门（主 Agent 定，2026-09-24 修正版）：
  * 目标卡「非本项目进程 HBM 合计」> foreign_cap_mb ⇒ 停我们的容器
  * 目标卡整卡 HBM > chip_cap_mb（主 Agent 给的 52 GB 硬线）⇒ 停我们的容器
  * 外部进程 PID 变化只记录、不触发停机（yxt 的任务本身在自然起落）

只调用 `docker stop <我们的容器>`；不 kill 任何外部进程，也不读别人内存内容。
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pwd
import re
import subprocess
import time
from pathlib import Path

OUR_USER = "l00886679"


def sample(phy: int) -> tuple[int, int, dict[str, tuple[str, int]]]:
    out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True).stdout
    aic = hbm = -1
    procs: dict[str, tuple[str, int]] = {}
    inproc = False
    for ln in out.splitlines():
        if "Process id" in ln:
            inproc = True
            continue
        if not ln.startswith("|"):
            continue
        fields = [x.strip() for x in ln.split("|")]
        if not inproc:
            if len(fields) >= 4 and fields[2].startswith("0000:"):
                left = fields[1].split()
                if len(left) == 2 and left[1].isdigit() and int(left[1]) == phy:
                    m = re.match(r"(\d+)\s+.*?(\d+)\s*/\s*65536", fields[3])
                    if m:
                        aic, hbm = int(m.group(1)), int(m.group(2))
        else:
            if len(fields) >= 6 and fields[2].isdigit():
                left = fields[1].split()
                if len(left) == 2 and int(left[0]) * 2 + int(left[1]) == phy:
                    try:
                        user = pwd.getpwuid(os.stat(f"/proc/{fields[2]}").st_uid).pw_name
                    except Exception:
                        user = "?"
                    procs[fields[2]] = (user, int(fields[4]) if fields[4].isdigit() else -1)
    return aic, hbm, procs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phy", type=int, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--interval", type=float, default=120.0)
    parser.add_argument("--foreign-cap-mb", type=int, default=36864)
    parser.add_argument("--chip-cap-mb", type=int, default=52000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    aic, hbm, procs = sample(args.phy)
    foreign = {pid: v for pid, v in procs.items() if v[0] != OUR_USER}
    baseline = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phy": args.phy,
        "container": args.container,
        "aic": aic,
        "hbm_mb": hbm,
        "foreign": {pid: {"user": v[0], "hbm_mb": v[1]} for pid, v in foreign.items()},
    }
    events: list[dict] = []
    stop_reason = None
    with args.out.open("a") as fh:
        fh.write(json.dumps({"baseline": baseline}, ensure_ascii=False) + "\n")

    while True:
        time.sleep(args.interval)
        aic, hbm, procs = sample(args.phy)
        gone = sorted(set(foreign) - set(procs))
        foreign_now = {pid: v for pid, v in procs.items() if v[0] != OUR_USER}
        ev = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "aic": aic,
            "hbm_mb": hbm,
            "foreign_now": {pid: v[1] for pid, v in foreign_now.items()},
            "gone": gone,
        }
        foreign_mb = sum(v[1] for v in foreign_now.values() if v[1] > 0)
        ev["foreign_mb"] = foreign_mb
        if foreign_mb > args.foreign_cap_mb:
            ev["stop_reason"] = f"foreign_hbm {foreign_mb} > cap {args.foreign_cap_mb}"
        elif hbm > args.chip_cap_mb:
            ev["stop_reason"] = f"chip_hbm {hbm} > cap {args.chip_cap_mb}"
        events.append(ev)
        with args.out.open("a") as fh:
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
        if "stop_reason" in ev:
            stop_reason = ev["stop_reason"]
            break

    subprocess.run(["docker", "stop", args.container], capture_output=True, text=True)
    with args.out.open("a") as fh:
        fh.write(json.dumps({"stopped": args.container, "reason": stop_reason}, ensure_ascii=False) + "\n")
    print(f"[watchdog] stopped {args.container}: {stop_reason}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
