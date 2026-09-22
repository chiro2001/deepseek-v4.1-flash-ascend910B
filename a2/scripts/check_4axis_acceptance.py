#!/usr/bin/env python3
"""四轴同开的**验收判决器** —— 逐条判"目标"里写的那 7 条，并打印原始读数。

用法：
    python3 check_4axis_acceptance.py --log <serve.log> --client <arm>.client.json \
        [--metrics <metrics_after.txt>] [--container <ctr>] [--text-probe-json <t.json>] \
        [--kv-events <kv_events.json>]

设计原则：**每条都打印它的原始读数**（grep 计数 / 计数器值），这样"判过"可以被第三方复算；
凡**没给证据**的项标成"未验"，**不**计入通过（避免"看不见所以全绿"）。

退出码：0 = 已给证据的项全过；1 = 有未过项；2 = 用法错误
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys


class Report:
    def __init__(self) -> None:
        self.p = 0
        self.f = 0
        self.u = 0

    def ok(self, name, detail):
        print(f"  \033[32m✓\033[0m {name:<50} {detail}")
        self.p += 1

    def bad(self, name, detail):
        print(f"  \033[31m✗\033[0m {name:<50} {detail}")
        self.f += 1

    def skip(self, name, detail):
        print(f"  \033[33m?\033[0m {name:<50} {detail}（未验，不算通过）")
        self.u += 1

    @staticmethod
    def info(name, detail):
        print(f"    · {name:<50} {detail}")


def count(text: str, needle: str) -> int:
    """按行计数（与 `grep -c` 同口径：一行算一次）。"""
    return sum(1 for ln in text.splitlines() if needle in ln)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--client", default="")
    ap.add_argument("--metrics", default="")
    ap.add_argument("--container", default="")
    ap.add_argument("--text-probe-json", default="")
    ap.add_argument("--kv-events", default="")
    a = ap.parse_args()

    try:
        log = open(a.log, errors="replace").read()
    except Exception as e:  # noqa: BLE001
        print(f"读不了 --log：{e!r}", file=sys.stderr)
        return 2

    R = Report()
    print("=" * 78)
    print("四轴同开 · 验收判决（ENGRAM=1 × 卸载 × int8档C × draft入图）")
    print(f"  log     = {a.log}")
    print(f"  client  = {a.client or '<未给>'}")
    print(f"  metrics = {a.metrics or '<未给>'}")
    print(f"  ctr     = {a.container or '<未给>'}")
    print(f"  text    = {a.text_probe_json or '<未给>'}")
    print("=" * 78)

    # ------------------------------------------------------------ ① 起服期
    print("① 起服期错误码（全 0 才算过）")
    for name, pat in [("图捕获 EE1016", "EE1016"), ("远程读 507057", "507057"),
                      ("分配器 EH0012", "EH0012"), ("注册资源 207001", "207001"),
                      ("注册回落 aclrtHostRegister failed", "aclrtHostRegister failed")]:
        n = count(log, pat)
        (R.ok if n == 0 else R.bad)(name, f"{pat} = {n}")
    print("   ★ 异常类（用**带冒号**的精确模式，见 logs/079 §3）")
    for name, pat in [("KeyError:", "KeyError:"), ("Traceback", "Traceback"),
                      ("EngineDeadError", "EngineDeadError"),
                      ("device_metadata 泄漏", "has not been released"),
                      ("TypeError（我们的代码）", "TypeError")]:
        n = count(log, pat)
        (R.ok if n == 0 else R.bad)(name, f"{pat} = {n}")

    # ------------------------------------------------------------ ★ host 路径证据
    print("   ★ 走的是哪条路（logs/085：device-index 一开会绕过整条 host 路径）")
    n = count(log, "DEVICE-INDEX")
    (R.ok if n == 0 else R.bad)("未走 device 路径（必须 0）", f"DEVICE-INDEX = {n}")
    n = count(log, "ENGRAM-TRUE-TOKENS")
    (R.ok if n > 0 else R.bad)("修补代码真的在跑（应 >0）", f"ENGRAM-TRUE-TOKENS = {n}")
    n = count(log, "ENGRAM-PAGELESS")
    Report.info("降级提示（0 = pad 兜底没被用到）", f"ENGRAM-PAGELESS = {n}")

    # ------------------------------------------------------------ ⑤ int8
    print("⑤ int8 档位（自报 C + env 真进容器）")
    m = re.search(r"档位\s*[:：]\s*([A-Za-z]+)", log)
    if m:
        (R.ok if m.group(1).upper() == "C" else R.bad)("档位自报 = C", f"读到 {m.group(1)}")
    elif "R8_KV8_SWA=1" in log or "KV8_GRAPH_SAFE=1" in log:
        R.ok("档位证据（无自报行，用 env 证据兜）", "命中 R8_KV8_SWA=1 / KV8_GRAPH_SAFE=1")
    else:
        R.bad("档位自报 = C", "既无自报行也无 env 证据")
    if a.container:
        try:
            env = subprocess.run(
                ["docker", "exec", a.container, "sh", "-c", "env | grep -E 'KV8_SWA|DEVICE_INDEX'"],
                capture_output=True, text=True, timeout=30).stdout.strip().replace("\n", " ")
            (R.ok if re.search(r"VLLM_V41_KV8_SWA=(1|true)", env) else R.bad)(
                "容器内 VLLM_V41_KV8_SWA 非 0", env or "<读不到>")
            Report.info("容器内 device-index", env)
        except Exception as e:  # noqa: BLE001
            R.skip("容器内 env", f"读失败 {e!r}")
    else:
        R.skip("容器内 VLLM_V41_KV8_SWA", "未给 --container")

    # ------------------------------------------------------------ ④ draft 入图
    print("④ draft 真入图（Wrapping = 8 且 A 不恒 1.0）")
    n = count(log, "Wrapping draft model with ACLGraphWrapper")
    (R.ok if n >= 8 else R.bad)("Wrapping draft model（>=8）", f"= {n}")
    accs = re.findall(r"Mean acceptance length:\s*([0-9.]+)", log)
    if accs:
        tail = accs[-3:]
        unhealthy = [v for v in tail if float(v) < 1.05]
        msg = f"最近 3 次 = {tail}"
        if unhealthy and len(unhealthy) == len(tail):
            R.bad("稳态接受长度 A 不恒 1.0", msg + " ⇒ 全部 ~1.0（图可能没生效）")
        else:
            R.ok("稳态接受长度 A 不恒 1.0", msg)
    else:
        R.skip("稳态接受长度 A", "日志里没有 SpecDecoding 行（mt=1 时会这样）")

    # ------------------------------------------------------------ ② + ③ 卸载
    print("② 两轮 requests_failed = 0（客户端 json）")
    if a.client:
        try:
            d = json.load(open(a.client))
            rs = d.get("rounds") or []
            if not rs:
                R.bad("轮次", "client.json 里没有 rounds")
            else:
                badr = 0
                for i, x in enumerate(rs):
                    f = x.get("requests_failed")
                    Report.info(f"round{i+1}",
                                f"ok={x.get('requests_ok')} failed={f} wall={x.get('wall_s')}")
                    if f != 0:
                        badr += 1
                (R.ok if badr == 0 else R.bad)(
                    f"{len(rs)} 轮全部 failed=0", f"失败轮数 = {badr}")
        except Exception as e:  # noqa: BLE001
            R.bad("client.json 可读", repr(e))
    else:
        R.skip("两轮 requests_failed", "未给 --client")

    print("③ 卸载三判据（引擎侧；8 卡臂的计数器在 metrics 文件里）")
    blob = log + ("\n" + open(a.metrics, errors="replace").read() if a.metrics else "")
    m = re.search(r"CPU_to_GPU[\"=: ]+([0-9.eE+]+)", blob)
    if m:
        v = float(m.group(1))
        (R.ok if v > 0 else R.bad)("CPU_to_GPU > 0", f"= {m.group(1)}")
    else:
        R.skip("CPU_to_GPU", "没找到（给 --metrics 试试）")
    m = re.search(r"hits[\"=: ]+([0-9]+)", blob)
    if m:
        v = int(m.group(1))
        (R.ok if v > 0 else R.bad)("hits > 0", f"= {v}")
    else:
        R.skip("hits", "没找到（给 --metrics 试试）")
    m = re.search(r"block_removed_total\{[^}]*CPU[^}]*\}\s*([0-9.eE+]+)", blob)
    if m:
        v = float(m.group(1))
        (R.ok if v == 0 else R.bad)("BlockRemoved:CPU == 0", f"= {m.group(1)}")
    else:
        R.skip("BlockRemoved:CPU", "没找到（给 --metrics / --kv-events 试试）")

    # ------------------------------------------------------------ ⑥ 文本
    print("⑥ 返回文本正确（自然语言判据）")
    if a.text_probe_json:
        try:
            t = json.load(open(a.text_probe_json))
            q = t.get("questions") or {}
            pp = t.get("prefix_pair") or {}
            okq = bool(q.get("n")) and q.get("n_pass") == q.get("n")
            Report.info("题库", f"{q.get('n_pass')}/{q.get('n')}")
            if pp:
                Report.info("prefix-pair",
                            f"n_distinct={pp.get('n_distinct')} tail_same={pp.get('tail_same')}")
                okp = pp.get("tail_same")
                if okp is None:
                    okp = pp.get("same_all")
            else:
                okp = None
                Report.info("prefix-pair", "缺失（建议 --mode all）")
            if okq:
                R.ok("题库全对", f"{q.get('n_pass')}/{q.get('n')}")
            else:
                R.bad("题库全对", f"{q.get('n_pass')}/{q.get('n')}")
            if okp is None:
                R.skip("prefix-pair（同前缀两发一致）", "缺数据")
            elif okp:
                R.ok("prefix-pair（第 2 发起逐字相同）", "tail_same=True")
            else:
                R.bad("prefix-pair（第 2 发起逐字相同）", f"n_distinct={pp.get('n_distinct')}")
        except Exception as e:  # noqa: BLE001
            R.bad("文本证据可读", repr(e))
    else:
        R.skip("返回文本正确", "未给 --text-probe-json")

    print("=" * 78)
    print(f"  通过 {R.p} 项 / 失败 {R.f} 项 / 未验 {R.u} 项")
    if R.f == 0:
        print("  ✓ 已给证据的项全过" + ("" if R.u == 0 else f"；★ 但还有 {R.u} 项**未验**（那不是通过）"))
        return 0
    print("  ✗ 有未过项，见上面逐条读数")
    return 1


if __name__ == "__main__":
    sys.exit(main())
