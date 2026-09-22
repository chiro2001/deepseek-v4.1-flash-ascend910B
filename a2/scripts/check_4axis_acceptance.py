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
    ap.add_argument("--arm-launch-log", default="",
                    help="runner 的启动日志（含 VLLM_V41_ENGRAM_TRUE_TOKENS 的原值）"
                         "，例如 agents/R_8card_int8/logs/<tag>.serve_a2.log")
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
    # ★ 口径修正（2026-09-22 23:5x）：这一条**只在开了精确修复时**才有意义。
    #   若 `VLLM_V41_ENGRAM_TRUE_TOKENS=0`，这段代码按设计**就是不跑的**
    #   （走 pad 兜底）⇒ 报"未验"而不是"失败"，否则会把一条合法配置误判成缺陷。
    # ★ 再修一次口径（2026-09-22 23:5x）：`VLLM_V41_ENGRAM_TRUE_TOKENS=0` 这个字面量
    #   **不一定在 serve.log 里**（runner 把它写在 inner.sh / 容器的 env 里）。
    #   ⇒ 判据顺序：① 日志里有该字样 → 按配置判；② 否则**去容器 env 里读**；
    #      ③ 都读不到才按"应 >0"判（保守）。
    _tt_off = ("VLLM_V41_ENGRAM_TRUE_TOKENS=0" in log) or ("VLLM_V41_ENGRAM_TRUE_TOKENS='0'" in log)
    _tt_known = False
    if _tt_off:
        _tt_known = True
    # ★★★ 2026-09-23 00:5x **判据修正**：原来"判不了"就直接 `bad` ——
    #   而本脚本自己的设计原则是「没证据 ⇒ 标**未验**，不算通过」。实测事故：
    #   `r8-4axis-fit` 臂的容器已被 runner 清理（`KEEP=0`），我误传了别的容器名
    #   ⇒ 查不到它的 `VLLM_V41_ENGRAM_TRUE_TOKENS=0` ⇒ 被**误判成失败**（其实那臂是对的）。
    #   ⇒ 现在：① 先从容器的 **两个** 内核接口读；② 再读 `--arm-launch-log`（runner 的启动日志，
    #      它一定含 `VLLM_V41_ENGRAM_TRUE_TOKENS=`）；③ **都读不到 ⇒ skip（未验），不判失败**。
    if a.arm_launch_log:
        try:
            _ll = open(a.arm_launch_log, errors="replace").read()
            _m = re.search(r"VLLM_V41_ENGRAM_TRUE_TOKENS=(\d+)", _ll)
            if _m:
                _tt_known = True
                if _m.group(1) == "0":
                    _tt_off = True
        except Exception:  # noqa: BLE001
            pass
    if not _tt_off and a.container:
        try:
            _e = subprocess.run(
                ["docker", "exec", a.container, "sh", "-c", "env | grep VLLM_V41_ENGRAM_TRUE_TOKENS"],
                capture_output=True, text=True, timeout=30).stdout
            if "VLLM_V41_ENGRAM_TRUE_TOKENS=0" in _e:
                _tt_off = True
                _tt_known = True
            elif "VLLM_V41_ENGRAM_TRUE_TOKENS=" in _e:
                _tt_known = True
        except Exception:  # noqa: BLE001
            pass
    if _tt_off:
        R.skip("精确修复未开（pad 兜底）", f"ENGRAM-TRUE-TOKENS = {n}；按设计不跑")
    elif not _tt_known:
        R.skip("修补代码是否在跑（判不了）",
               f"ENGRAM-TRUE-TOKENS = {n}；★ 没给 --container/--arm-launch-log ⇒ **未验**（不判失败）")
    else:
        (R.ok if n > 0 else R.bad)("修补代码真的在跑（应 >0）", f"ENGRAM-TRUE-TOKENS = {n}")
    n = count(log, "ENGRAM-PAGELESS")
    Report.info("降级提示", f"ENGRAM-PAGELESS = {n}（>0 = pad 兜底被用过；=0 = 一次没用）")

    # ------------------------------------------------------------ ⑤ int8
    print("⑤ int8 档位（自报 C + env 真进容器）")
    # ★ 口径修正（2026-09-22 23:5x）：8 卡 runner **不打印**"档位"行（那是 A2 的
    #   `serve_a2_offload.sh` 的格式）。它的可核证据是：① 环境变量被 vLLM 报为
    #   "Unknown vLLM environment variable"（= 真的进了容器）；② inner.sh 里的 export；
    #   ③ meta.txt 里的 `R8_KV8_SWA=1 / R8_RING_FP16=1 / R8_APC_ALIGN=3`。
    #   ⇒ 三条任一命中即算有证据；三条都无才判失败。
    m = re.search(r"档位\s*[:：]\s*([A-Za-z]+)", log)
    _kv8_unknown = "Unknown vLLM environment variable detected: VLLM_V41_KV8_GRAPH_SAFE" in log
    if m:
        (R.ok if m.group(1).upper() == "C" else R.bad)("档位自报 = C", f"读到 {m.group(1)}")
    elif _kv8_unknown or "R8_KV8_SWA=1" in log or "KV8_GRAPH_SAFE=1" in log:
        R.ok("档位证据（8 卡 runner 无自报行，用 env 痕迹兜）",
             "命中 KV8_GRAPH_SAFE env 痕迹" if _kv8_unknown else "命中 R8_KV8_SWA=1/KV8_GRAPH_SAFE=1")
    else:
        R.bad("档位自报 = C", "既无自报行也无 env 痕迹")
    if a.container:
        # ★★ 口径修正（2026-09-22 23:5x）：tier C 的 int8 开关**不在**容器的 `docker exec env` 里 ——
        #   它们在 **`inner.sh`**（由 shadow 的 heredoc 生成）里 export，只对被启动的那个
        #   python 进程可见。子代理实测：`docker exec <ctr> env | grep KV8` **看不到**，
        #   而 `inner.sh` 里有。⇒ 先查 inner.sh，再退回 docker exec env。
        #   （这与今天反复强调的"看代码痕迹、不看 env 名字"是同一件事。）
        inner = ""
        try:
            inner = subprocess.run(
                ["docker", "exec", a.container, "sh", "-c",
                 "grep -E 'VLLM_V41_KV8_SWA=|VLLM_V41_RING_FP16=|VLLM_V41_APC_ALIGN=|VLLM_V41_KV8_GRAPH_SAFE=' "
                 "/opt/dsv41/scripts/../../opt/dsv41 2>/dev/null; "
                 "linux=$(ls -d /opt/dsv41/results/*/inner.sh 2>/dev/null | head -1); "
                 "[ -n \"$linux\" ] && grep -E 'VLLM_V41_KV8|APC_ALIGN|GRAPH_SAFE' \"$linux\""],
                capture_output=True, text=True, timeout=40).stdout.strip().replace("\n", " ")
        except Exception:  # noqa: BLE001
            inner = ""
        # ★ 注意 inner.sh 里的形式是**带引号**的：`export VLLM_V41_KV8_SWA='1'`
        if re.search(r"VLLM_V41_KV8_SWA='?(1|true)'?", inner):
            R.ok("tier C 的 int8 开关（从 inner.sh 读到）", inner[:160])
        else:
            # 退回：日志里的 env 证据（runner 会打印 KV8_GRAPH_SAFE / R8_KV8_SWA）
            if "KV8_GRAPH_SAFE=1" in log or "R8_KV8_SWA=1" in log:
                R.ok("tier C 的 int8 开关（日志证据兜）", "命中 KV8_GRAPH_SAFE=1 / R8_KV8_SWA=1")
            else:
                R.bad("tier C 的 int8 开关", f"inner.sh 与日志都没读到（inner={inner[:80]!r}）")
        try:
            env = subprocess.run(
                ["docker", "exec", a.container, "sh", "-c", "env | grep -E 'KV8_SWA|DEVICE_INDEX|TRUE_TOKENS'"],
                capture_output=True, text=True, timeout=30).stdout.strip().replace("\n", " ")
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
    # ★ 口径修正（2026-09-22 23:5x）：prometheus 那行长这样：
    #   vllm:kv_offload_total_bytes_total{...,transfer_type="CPU_to_GPU"} 2.1399530496e+10
    #   ⇒ `CPU_to_GPU` 与数字之间是 `"} `（**含 `}`**），旧正则的字符类没有 `}` ⇒ 抓不到。
    m = re.search(r'transfer_type="CPU_to_GPU"\}\s*([0-9.eE+]+)', blob)
    if not m:
        m = re.search(r"CPU_to_GPU[\"=: \}\s]*([0-9.eE+]+)", blob)
    if m:
        v = float(m.group(1))
        (R.ok if v > 0 else R.bad)("CPU_to_GPU > 0", f"= {m.group(1)}")
    else:
        R.skip("CPU_to_GPU", "没找到（给 --metrics 试试）")
    m = re.search(r'external_prefix_cache_hits_total\{[^}]*\}\s*([0-9.eE+]+)', blob)
    if not m:
        m = re.search(r"hits[\"=: \}\s]*([0-9]+)", blob)
    if m:
        v = float(m.group(1))
        (R.ok if v > 0 else R.bad)("hits > 0", f"= {v}")
    else:
        R.skip("hits", "没找到（给 --metrics 试试）")
    # ★ 口径修正（2026-09-22 23:5x）：`BlockRemoved:CPU` **不在 prometheus metrics 里**
    #   （本 build 没有 `kv_offload_block_removed_total`）—— 它只出现在 **`kv_events.json`
    #   的 `counts`** 字典里（格式：`"BlockRemoved:CPU": 29469`）。
    _br = re.search(r'block_removed_total\{[^}]*CPU[^}]*\}\s*([0-9.eE+]+)', blob)
    _brv = None
    if _br:
        _brv = float(_br.group(1))
        _src = "metrics"
    elif a.kv_events:
        try:
            _kv = json.load(open(a.kv_events))
            _c = _kv.get("counts") or {}
            if "BlockRemoved:CPU" in _c:
                _brv = float(_c["BlockRemoved:CPU"])
                _src = "kv_events.json"
            elif _c:
                # ★★★ 2026-09-23 00:5x **口径修正（这一条差点把"最强形式的达成"判成"未验"）**：
                #   `kv_events.json` 的 `counts` 只收录**出现过的事件类型** ⇒
                #   **键不存在 = 该事件一次都没发生 = 0**（`logs/096` 的 `r8-4axis-fit` 实测：
                #   池降到 `cpu_cache_usage=0.7465` 后 `BlockRemoved:CPU` 的键**直接消失**，
                #   而 `PROMPTS=8` 那条是 `500`）。
                #   ⇒ 这不是"数据缺失"，这是"事件数为 0"的**最强形式**。
                #   ★ 但要防止"拿了一个空/错的文件"也被当成 0 ⇒ 只在 `counts` **非空**
                #     且**至少见过一个 Block 事件**时才这样判。
                if any(k.startswith("Block") for k in _c):
                    _brv = 0.0
                    _src = "kv_events.json（★ 键不存在 ⇒ 该事件 0 次）"
        except Exception:  # noqa: BLE001
            pass
    if _brv is not None:
        (R.ok if _brv == 0 else R.bad)("BlockRemoved:CPU == 0", f"= {_brv:g}（来自 {_src}）")
        if _brv != 0:
            Report.info("说明",
                        "池被撑爆 ⇒ 正常淘汰。要满足本条需放大池（A2 生产 85 GiB / 或减小工作集）")
    else:
        R.skip("BlockRemoved:CPU", "没找到（给 --kv-events <kv_events.json>）")
    m = re.search(r"kv_offload_cpu_cache_usage_perc\{[^}]*\}\s*([0-9.eE+]+)", blob)
    if m:
        Report.info("cpu_cache_usage（淘汰通常发生在接近 1.0 时）", f"= {m.group(1)}")

    # ------------------------------------------------------------ ⑥ 文本
    print("⑥ 返回文本正确（自然语言判据）")
    if a.text_probe_json:
        try:
            t = json.load(open(a.text_probe_json))
            q = t.get("questions") or {}
            pp = t.get("prefix_pair") or {}
            # ★ 口径修正（2026-09-22 23:5x）：兼容**两种**证据 schema ——
            #   本脚本自己的（`questions` 是 list、`n_pass`/`n` 在顶层）
            #   与子代理那份（`questions` 是 list、`n_pass`/`n` 也在顶层）。
            #   关键差别在 prefix_pair 的字段名：本脚本是 `tail_same`/`same_all`，
            #   子代理那份是 `from_2nd_same`/`all_same` ⇒ 两种都认。
            nq = q.get("n") or (len(q) if isinstance(q, list) else None) or t.get("n")
            npass = t.get("n_pass")
            if npass is None and isinstance(q, dict):
                npass = q.get("n_pass")      # ★ 本脚本自己的 schema：questions 是 dict
            if npass is None and isinstance(q, list):
                npass = sum(1 for x in q if isinstance(x, dict) and x.get("pass"))
            Report.info("题库", f"{npass}/{nq}")
            okq = bool(nq) and npass == nq
            if pp:
                tail = pp.get("tail_same")
                if tail is None:
                    tail = pp.get("from_2nd_same")       # 子代理那份的名字
                alls = pp.get("same_all")
                if alls is None:
                    alls = pp.get("all_same")
                Report.info("prefix-pair",
                            f"n_distinct={pp.get('n_distinct')} "
                            f"tail_same={tail} all_same={alls}")
                okp = tail if tail is not None else alls
            else:
                okp = None
                Report.info("prefix-pair", "缺失（建议 --mode all）")
            if okq:
                R.ok("题库全对", f"{npass}/{nq}")
            else:
                R.bad("题库全对", f"{npass}/{nq}")
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
