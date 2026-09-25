#!/usr/bin/env python3
"""CED 1+1 刀锋游标行走：把 D 侧 g0 的**最大物理块号**精确推到目标值。

背景：假说「请求一旦分到 id >= B 的块就静默失败」。要验证/否证它，需要在同一
套 1+1（P=chip0 / D=chip1，tiny dummy 权重）上做**单变量对照**：

  * 臂 B：D 池 C=29129（可用最大块号 = C-1 = 29128）→ 精确落在 29128
  * 臂 A：D 池 C=29128（可用最大块号 = 29127）→ 同一序列只能到 29127

机制：D 侧块分配是「自由链表的顺序游标」——串行请求会把游标单调往前推
R(n) 个块（R = 本请求吃掉的总块数），推过池尾才回绕。所以只要知道
`(上一次 max, 本请求 n)` → 下一次 max，就能用**一次定长请求**把某一条请求的
max 精确钉在目标上（±1 块）。本脚本：

  1) 记下每个请求的响应证据（首 token id / logprob / top-5 / usage / 墙钟）；
  2) 等 `[CED-BLOCK-DUMP]` 的 `decode_<rid>_g0.txt` 落盘，读出完整块列表；
  3) 输出 (max_id, n_blocks, first_token, logprob, verdict) 一行一条 JSONL。

用法：
  # 低块号基线：同布局重复 K 次
  python3 tools/ced_knifeedge_walk.py fixed --n 65536 --count 3 ...
  # 走到目标
  python3 tools/ced_knifeedge_walk.py walk --target-max 29128 ...

只发 HTTP 请求 + 读 dump 目录，不写服务端状态。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from itertools import cycle, islice
from pathlib import Path

TOKEN_PATTERN = (28669, 6441, 58603, 693, 85450, 84483, 22089, 320)


def r_of_n(n: int, overhead: int = 21) -> int:
    """本请求从 D 池里吃掉的块数（= 本请求自己的 g0 条数 + 其余组的固定 21 块）。"""
    return c_of_n(n) + overhead


def n_of_r(r: int, overhead: int = 21) -> int:
    """r_of_n 的逆。"""
    return n_of_c(r - overhead)


def c_of_n(n: int) -> int:
    """本请求自己的 g0 块数（= dump 条数）。

    实测 18/18 点吻合（arm B 4 点 + arm A1 14 点，含 128/129/256/257/385/1025/
    1153/1280/20864/26496/26497/32001/45442/45568/45569/61313/65535）：
        c(n) = ceil((n - 1) / 128)
    与主 Agent 给的 R = ceil((N-1)/128) + 21 完全一致。
    """
    return -(-(max(1, n) - 1) // 128)


def n_of_c(c: int) -> int:
    """c_of_n 的最小逆：prompt 长度 n 使 c(n) == c。"""
    if c < 1:
        raise ValueError(f"c={c} 非法（最小 1）")
    return (c - 1) * 128 + 2


def build_body(model: str, n: int, top_logprobs: int, max_tokens: int) -> bytes:
    ids = list(islice(cycle(TOKEN_PATTERN), n))
    return json.dumps(
        {
            "model": model,
            "prompt": ids,
            "temperature": 0,
            "max_tokens": max_tokens,
            "logprobs": top_logprobs,
        }
    ).encode()


def post(url: str, body: bytes, timeout: float):
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(), time.time() - started
    except urllib.error.HTTPError as error:
        return error.code, error.read(), time.time() - started
    except Exception as error:  # noqa: BLE001 - 传输层错误也要留证据
        return -1, json.dumps({"transport_error": repr(error)}).encode(), time.time() - started


def summarize(raw: bytes) -> dict:
    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as error:
        return {"parse_error": str(error), "raw_head": raw[:400].decode("utf-8", "replace")}
    choices = payload.get("choices") or []
    summary: dict = {"id": payload.get("id"), "usage": payload.get("usage")}
    if not choices:
        summary["choices"] = []
        return summary
    choice = choices[0]
    logprobs = (choice.get("logprobs") or {}).get("token_logprobs") or []
    tokens = (choice.get("logprobs") or {}).get("tokens") or []
    tops = (choice.get("logprobs") or {}).get("top_logprobs") or []
    first = None
    if logprobs:
        first = {
            "token": tokens[0] if tokens else None,
            "logprob": round(logprobs[0], 6),
            "top5": [
                {"token": key, "logprob": round(value, 6)}
                for key, value in list((tops[0] if tops else {}).items())[:5]
            ],
        }
    summary.update(
        {
            "text": choice.get("text"),
            "finish_reason": choice.get("finish_reason"),
            "first_token": first,
        }
    )
    return summary


def rid_of(response_id) -> str | None:
    if not isinstance(response_id, str):
        return None
    if response_id.startswith("chatcmpl-"):
        parts = response_id.split("-")
        if len(parts) > 1:
            return parts[1]
    return response_id


class DumpReader:
    """读 dump 目录。默认走宿主路径；`container` 非空时改走 `docker exec`。

    为什么要后者：a3-22 的 /home（ext4）在高负载下会让宿主看不到容器刚写入的
    文件（容器内 `ls` 正常），此时宿主 glob 永远也找不到 dump。走容器视图可绕开。
    """

    def __init__(self, dump_dir: str, container: str = "", container_prefix: str = "") -> None:
        self.dump_dir = dump_dir
        self.container = container
        if not self.container and container_prefix:
            self.container = self._resolve(container_prefix)

    @staticmethod
    def _resolve(prefix: str) -> str:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=60,
        ).stdout.split()
        matches = [name for name in out if name.startswith(prefix)]
        if len(matches) != 1:
            raise SystemExit(f"[dump] 前缀 {prefix!r} 匹配到 {len(matches)} 个容器：{matches}")
        return matches[0]

    def _exec(self, script: str) -> str:
        result = subprocess.run(
            ["docker", "exec", self.container, "bash", "-lc", script],
            capture_output=True, text=True, timeout=120,
        )
        return result.stdout

    def list_names(self) -> list[str]:
        if self.container:
            return [n for n in self._exec(f"ls -1 {self.dump_dir}").split() if n.endswith(".txt")]
        try:
            return [p.name for p in Path(self.dump_dir).iterdir()]
        except OSError:
            return []

    def read(self, name: str) -> list[int]:
        if self.container:
            text = self._exec(f"cat {self.dump_dir}/{name}")
        else:
            text = (Path(self.dump_dir) / name).read_text(encoding="utf-8", errors="replace")
        return [int(line) for line in text.split() if line.strip()]


def read_dump(
    reader: DumpReader, rid: str, timeout: float
) -> tuple[list[int], str] | None:
    """按 `decode_{rid}*_g0.txt` 找 dump。

    注意 /v1/completions 的响应 id 是 `cmpl-<uuid>`，而 dump 文件名用的是**完整**
    request id（形如 `cmpl-<uuid>-0-<hash>`），所以必须前缀匹配而不是精确匹配。
    """
    deadline = time.time() + timeout
    while True:
        names = [
            name
            for name in reader.list_names()
            if name.startswith(f"decode_{rid}") and name.endswith("_g0.txt")
        ]
        for name in sorted(names, reverse=True):
            try:
                ids = reader.read(name)
            except (OSError, subprocess.SubprocessError, ValueError):
                continue
            if ids:
                return ids, name
        if time.time() >= deadline:
            return None
        time.sleep(1.0)


class Runner:
    def __init__(self, args) -> None:
        self.args = args
        self.out = Path(args.out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.dump_dir = Path(args.dump_dir)
        self.reader = DumpReader(
            args.dump_dir,
            container=getattr(args, "dump_container", "") or "",
            container_prefix=getattr(args, "dump_container_prefix", "") or "",
        )
        self.index = 0
        self.last_step_max: int | None = None

    def send(self, n: int, label: str) -> dict:
        self.index += 1
        body = build_body(self.args.model, n, self.args.top_logprobs, self.args.max_tokens)
        started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        status, raw, wall = post(self.args.url, body, self.args.timeout)
        summary = summarize(raw)
        rid = rid_of(summary.get("id"))
        record = {
            "index": self.index,
            "label": label,
            "n_prompt": n,
            "http_status": status,
            "wall_s": round(wall, 3),
            "start": started,
            "rid": rid,
            "response": summary,
        }
        ids = None
        dump_name = None
        if rid:
            found = read_dump(self.reader, rid, self.args.dump_timeout)
            if found:
                ids, dump_name = found
        if ids:
            record["dump"] = {
                "file": dump_name,
                "count": len(ids),
                "min": min(ids),
                "max": max(ids),
                "first": ids[0],
                "last": ids[-1],
                "monotone": all(b > a for a, b in zip(ids, ids[1:])),
                "span": max(ids) - min(ids) + 1,
            }
        else:
            record["dump"] = None
        with self.out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        dump = record["dump"] or {}
        first = summary.get("first_token") or {}
        print(
            f"[walk] #{self.index:03d} {label} n={n} http={status} wall={wall:.1f}s "
            f"dump_n={dump.get('count')} min={dump.get('min')} max={dump.get('max')} "
            f"mono={dump.get('monotone')} "
            f"tok={first.get('token')!r} lp={first.get('logprob')} "
            f"usage={((summary.get('usage') or {}).get('prompt_tokens'))}",
            flush=True,
        )
        return record

    def last_max(self) -> int | None:
        rows = [
            json.loads(line)
            for line in self.out.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in reversed(rows):
            if row.get("dump"):
                return int(row["dump"]["max"])
        return None

    def fixed(self) -> int:
        for _ in range(self.args.count):
            self.send(self.args.n, "fixed")
        return 0

    def send_one(self) -> int:
        self.send(self.args.n, self.args.label)
        return 0

    def last_record(self) -> dict | None:
        if not self.out.exists():
            return None
        rows = [
            json.loads(line)
            for line in self.out.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return rows[-1] if rows else None

    def plan(self) -> int:
        """把游标精确停在 target_max 上。

        实测模型（arm B 三点验证，误差 0）：
            本请求 max = 上一次 max + ceil((n-1)/128) + overhead
        所以 Δ(n) 可以取 [22, 534] 内任意整数（MAX_LEN=65536 ⇒ n ≤ 65535）。

        方案：先用 coarse（Δ=534）逼近；当剩余 R 落进 (1156, 1690] 时，用一个
        filler 吃掉 m = R - j·Δ_small（j = floor((R-22)/Δ_small) ≥ 3），随后连发
        j 个**完全相同**的 small 请求 —— 倒数第三个起分别是 T-2Δ、T-Δ、T，最后
        一个恰好落在池顶。这样"块号越界"那一条请求的前后对照 prompt 完全一致。
        """
        target = self.args.target_max
        over = self.args.overhead
        small_n = self.args.small_n
        coarse_n = self.args.approach_n
        delta_small = r_of_n(small_n, over)
        delta_coarse = r_of_n(coarse_n, over)
        for value, name in ((delta_small, "--small-n"), (delta_coarse, "--approach-n")):
            if not 22 <= value <= 534:
                print(f"[plan][FAIL] {name} 的 Δ={value} 不在 [22,534]", file=sys.stderr)
                return 2
        record = self.last_record()
        if not record or not record.get("dump"):
            print("[plan][FAIL] 先跑 baseline（fixed 模式）建立游标起点", file=sys.stderr)
            return 2
        m_cur = int(record["dump"]["max"])
        print(
            f"[plan] target={target} 当前 max={m_cur} 剩余={target - m_cur} "
            f"Δ_small({small_n})={delta_small} Δ_coarse({coarse_n})={delta_coarse}",
            flush=True,
        )
        for step in range(self.args.max_requests):
            remaining = target - m_cur
            if remaining <= 0:
                print(f"[plan] 已到/越过目标 max={m_cur} target={target}", flush=True)
                break
            if remaining > delta_coarse + over + 3 * delta_small:
                record = self.send(coarse_n, f"coarse[{step}]")
            else:
                j = max(3, (remaining - over) // delta_small)
                m = remaining - j * delta_small
                if m > 0:
                    filler_n = n_of_r(m, over)
                    print(f"[plan] filler Δ={m} n={filler_n}；随后 {j} 个 Δ={delta_small}", flush=True)
                    record = self.send(filler_n, f"filler(d={m})")
                    m_cur = int((record.get("dump") or {}).get("max", -1))
                for k in range(j):
                    record = self.send(small_n, f"trio[{k + 1}/{j}]")
                    m_cur = int((record.get("dump") or {}).get("max", -1))
                break
            new_max = (record.get("dump") or {}).get("max")
            if new_max is None:
                print("[plan][FAIL] 本请求没有 dump，停止", file=sys.stderr)
                return 4
            print(
                f"[plan] 步 {step}: max {m_cur} -> {new_max} "
                f"(实测 Δ={new_max - m_cur}，预测 {delta_coarse})",
                flush=True,
            )
            m_cur = int(new_max)
        final = self.last_record()
        final_max = int(((final or {}).get("dump") or {}).get("max", -1))
        print(
            f"[plan] 结果：最终 max={final_max} target={target} "
            f"{'落点精确 ✓' if final_max == target else '落点不精确 ✗'}",
            flush=True,
        )
        for k in range(self.args.controls):
            self.send(small_n, f"postwrap[{k + 1}]")
        return 0 if final_max == target else 5

    def land(self) -> int:
        """闭环落点：反复"测量 max → 算差额 D → 发一个正好吃掉 D 的请求"。

        实测关系（A1 臂内 60+ 次连续请求逐点核对）：
            max(next) = max(prev) + 21 + c(next)
        其中 c(n) = ceil((n-1)/128) 是本请求的 g0 块数（18/18 吻合）。
        所以"要落在池顶 top"就给下一个请求取 c = top - max(prev) - 21。

        每步都读 dump 实测 max，因此对 ±1 的建模误差自愈；只有当一次推进把
        max 留在 (top-22, top) 之间时才会卡住（下一次最少推进 22 块）。
        """
        top = self.args.target_max
        overhead = self.args.overhead
        guard = 0
        while True:
            guard += 1
            if guard > self.args.max_requests:
                print(f"[land][FAIL] 超过最大请求数，停在 max={self.current_max()}", file=sys.stderr)
                return 3
            record = self.last_record()
            if not record or not record.get("dump"):
                print("[land][FAIL] 没有可用 dump（先跑一次 fixed/coarse）  " , file=sys.stderr)
                return 2
            m = int(record["dump"]["max"])
            if m == top:
                print(f"[land] ✓ 已精确落在 target={top}", flush=True)
                break
            if m > top:
                print(f"[land] ✗ 已越过 target={top}（当前 {m}）", flush=True)
                break
            need = top - m - overhead
            if need < 1:
                print(
                    f"[land][FAIL] 卡住：max={m}，距池顶 {top - m} 块，但最小推进 {overhead + 1} 块"
                    f" ⇒ 本臂无法落点，需要换一个步长可控的起手位置",
                    file=sys.stderr,
                )
                return 6
            if need > 512:
                n = self.args.approach_n
                label = f"coarse(need={need})"
            else:
                n = n_of_c(need)
                label = f"land(c={need})"
            self.send(n, label)
        for k in range(self.args.controls):
            self.send(self.args.controls_n, f"postwrap[{k + 1}]")
        return 0

    def current_max(self) -> int | None:
        record = self.last_record()
        return int(record["dump"]["max"]) if record and record.get("dump") else None

    def sweep(self) -> int:
        """用**固定的 final_n** 把最后一条请求精确钉在池顶 top。

        目的：让"触及边界的请求"与"低块号对照请求"用**同一个 prompt 长度**，
        这样两组的输出可以直接逐字段比。

        推导（max(next) = max(prev) + 21 + c(next)）：
            记 need = top - max - 21（下一个请求想落在池顶所需的 c），
               c_f = c(final_n)，s_f = 21 + c_f。
            若先发一个 c_b 的"桥"请求，再连发 (j+1) 个 final_n，则
              need - 21 - c_b - j*s_f = c_f
            ⇒ c_b = need - 21 - c_f - j*s_f。
            只要 c_b ∈ [1, 512] 就能用"桥 + j 个 final_n + 最后 1 个 final_n"
            精确落点，且落点请求与前面 j 个请求 prompt 完全相同。
        每一步都重新测量 need 再重算，所以对单步 ±1 误差自愈。
        """
        top = self.args.target_max
        final_n = self.args.final_n
        c_f = c_of_n(final_n)
        s_f = self.args.overhead + c_f
        limit = c_of_n(self.args.max_prompt)
        for step in range(self.args.max_requests):
            m = self.current_max()
            if m is None:
                print("[sweep][FAIL] 没有可用 dump", file=sys.stderr)
                return 2
            need = top - m - self.args.overhead
            if need == c_f:
                self.send(final_n, f"LAND(n={final_n},c={c_f})")
                break
            if need < 1:
                print(f"[sweep][FAIL] max={m} 距池顶 {top - m} 块，无法再落点", file=sys.stderr)
                return 6
            if need <= limit:
                # 一步就能落点，但用的是"非 final_n"的长度：仍可判据（与自身跨臂比），
                # 只是失去与低块号同 prompt 的直接对照。
                self.send(n_of_c(need), f"LAND-OTHER(c={need})")
                break
            # 找 j 使 c_b 合法
            c_b = None
            for j in range(max(1, (need - 21 - c_f) // s_f), 0, -1):
                cand = need - 21 - c_f - j * s_f
                if 1 <= cand <= limit:
                    c_b = cand
                    break
                if cand > limit:
                    continue
            if c_b is not None:
                self.send(n_of_c(c_b), f"bridge(c={c_b})")
                continue
            self.send(final_n, f"coarse[{step}]")
        else:
            print("[sweep][FAIL] 超过最大请求数", file=sys.stderr)
            return 3
        for k in range(self.args.controls):
            self.send(self.args.controls_n, f"postwrap[{k + 1}]")
        m = self.current_max()
        print(f"[sweep] 结束：max={m} target={top}", flush=True)
        return 0

    def series(self) -> int:
        """从**新鲜池**起，连发同一长度的请求，直到某条的 max 达到 stop_max。

        新鲜池 + 同长度大请求的实测性质（本臂 50/50 点验证）：
            min(k) = 1 + k*(c+21)，max(k) = c + k*(c+21)
        其中 c = c(n) = ceil((n-1)/128)。因此"某条请求恰好停在池顶 top"
        等价于同余式 c + k*(c+21) = top 有整数解：
          * top=29076（C=29077）：c=456, n=58242，第 61 条
          * top=29077（C=29078）：c 无纯同长度解 ⇒ 先用一个 c=457 的"桥"请求，
            再连发 c=456 的请求，第 60 条停在 29077
        """
        stop_max = self.args.stop_max
        stride = self.args.overhead + c_of_n(self.args.n)
        for index in range(1, self.args.count + 1):
            record = self.send(self.args.n, f"series[{index}]")
            dump = record.get("dump")
            if not dump:
                print("[series][FAIL] 本请求没有 dump", file=sys.stderr)
                return 4
            m = int(dump["max"])
            if index > 1 and m - self.last_step_max != stride:
                print(
                    f"[series][WARN] 第 {index} 条步长 {m - self.last_step_max} ≠ 预期 {stride}"
                    f"（free-list 介入？）",
                    flush=True,
                )
            self.last_step_max = m
            if m == stop_max:
                print(f"[series] ✓ 第 {index} 条精确停在 stop_max={stop_max}", flush=True)
                break
            if m > stop_max:
                print(f"[series] ✗ 第 {index} 条越过 stop_max={stop_max}（max={m}）", flush=True)
                break
        for k in range(self.args.controls):
            self.send(self.args.controls_n, f"post[{k + 1}]")
        return 0

    def walk(self) -> int:
        target = self.args.target_max
        overhead = self.args.overhead
        step = self.args.approach_n
        last = self.last_max()
        if last is None:
            print("[walk][FAIL] 没有基线 dump，先跑 fixed 建立游标起点", file=sys.stderr)
            return 2
        print(f"[walk] 起点 max={last} 目标={target} overhead={overhead}", flush=True)
        guard = 0
        while True:
            guard += 1
            if guard > self.args.max_requests:
                print("[walk][FAIL] 超过最大请求数，停在 max=%d" % last, file=sys.stderr)
                return 3
            remaining = target - last
            if remaining <= 0:
                print(f"[walk] 已到/超过目标（last={last}）", flush=True)
                break
            if remaining > r_of_n(step, overhead):
                record = self.send(step, "approach")
            else:
                # 精确落点：让本请求刚好吃掉 remaining 个块
                n = n_of_r(remaining, overhead)
                if n > step:
                    n = step
                record = self.send(n, f"land(target={target})")
            new_max = (record.get("dump") or {}).get("max")
            if new_max is None:
                print("[walk][FAIL] 本请求没拿到 dump，停止", file=sys.stderr)
                return 4
            if new_max > target:
                print(
                    f"[walk] 越过目标：last={new_max} target={target}"
                    f"（回绕后 max 会掉回低位）",
                    flush=True,
                )
                break
            last = int(new_max)
        for _ in range(self.args.controls):
            self.send(self.args.controls_n, "control")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["fixed", "walk", "send", "plan", "land", "sweep", "series"]
    )
    parser.add_argument("--label", default="send")
    parser.add_argument("--url", default="http://127.0.0.1:18962/v1/completions")
    parser.add_argument("--model", default="deepseek-v41-ced-tiny")
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=65536)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--target-max", type=int, default=29128)
    parser.add_argument("--overhead", type=int, default=21)
    parser.add_argument("--approach-n", type=int, default=65536)
    parser.add_argument("--controls", type=int, default=3)
    parser.add_argument("--controls-n", type=int, default=65536)
    parser.add_argument("--max-requests", type=int, default=200)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--dump-timeout", type=float, default=60.0)
    parser.add_argument(
        "--dump-container", default="", help="容器名：经由 docker exec 读 dump（绕开宿主 FS）"
    )
    parser.add_argument(
        "--dump-container-prefix", default="", help="容器名前缀，自动解析为唯一匹配"
    )
    parser.add_argument(
        "--small-n", type=int, default=45442, help="对照三连用的 prompt 长度（Δ=378）"
    )
    parser.add_argument("--final-n", type=int, default=65535, help="落点请求的 prompt 长度")
    parser.add_argument("--max-prompt", type=int, default=65535, help="允许的最大 prompt 长度")
    parser.add_argument("--stop-max", type=int, default=0, help="series 模式的停止块号")
    args = parser.parse_args()

    runner = Runner(args)
    if args.mode == "fixed":
        return runner.fixed()
    if args.mode == "send":
        return runner.send_one()
    if args.mode == "plan":
        return runner.plan()
    if args.mode == "land":
        return runner.land()
    if args.mode == "sweep":
        return runner.sweep()
    if args.mode == "series":
        return runner.series()
    return runner.walk()


if __name__ == "__main__":
    raise SystemExit(main())
