#!/usr/bin/env python3
"""Hold per-Phy-ID locks for one foreground CED experiment command."""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
from contextlib import ExitStack
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--devs", nargs="+", type=int, required=True)
    parser.add_argument("--lock-dir", type=Path, default=Path("/run/lock"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    devs = sorted(args.devs)
    if not devs or len(set(devs)) != len(devs) or any(d < 0 for d in devs):
        parser.error("--devs must contain distinct non-negative Phy-IDs")
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("A foreground command is required after --")
    with ExitStack() as stack:
        for dev in devs:
            path = args.lock_dir / f"expert-prefetch-npu-{dev}.lock"
            try:
                handle = stack.enter_context(path.open("a+"))
            except PermissionError:
                # An existing root-owned 0644 lock can be exclusively flocked
                # using a read-only FD; never replace its inode or contents.
                try:
                    handle = stack.enter_context(path.open("r"))
                except FileNotFoundError:
                    print(f"[CED-LOCK] cannot create {path}; run the lock supervisor with sudo", flush=True)
                    return 77
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"[CED-LOCK] busy: {path}", flush=True)
                return 75
        env = os.environ.copy()
        # Set only after all locks are held; child processes inherit the
        # visibility mapping, while descriptors remain in this supervisor.
        env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, args.devs))
        print(f"[CED-LOCK] held Phy-ID={args.devs}", flush=True)
        child = subprocess.Popen(command, env=env, start_new_session=True)

        def forward(signum, _frame):
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(signum, forward)
        result = child.wait()
        print(f"[CED-LOCK] command exited: {result}", flush=True)
        return result if result >= 0 else 128 - result


if __name__ == "__main__":
    raise SystemExit(main())
