#!/usr/bin/env python3
"""dynamic speculative decoding（按并发切 K）的**离线**自测。

为什么要有它：这条路要动"图捕获与 dispatch 的键"，而它失效的方式几乎都是
**静默**的 —— K=0 的步错配到 K=7 的图上不会报错，只会悄悄算错；反过来，
建图期把 ql=1 的键捕成 ql=8 的形状也不会报错。`bash -n` 与 `py_compile`
都查不出这两类。所以这里分成两块：

  ① `patches/files/patch_cudagraph.py` 的分支逻辑（用 stub 注入，不需要
     vllm / NPU / 显卡）：query_len 集合推导、按本步 ql 反推 num_reqs、
     缺图时的降级、多 ql 建图与状态恢复、跳过桶取整；
  ② `scripts/serve_v2.sh` 的 `SP_SCHEDULE` → `--speculative-config` 转换
     （含"格式错就退回固定 K"的负控），以及**不设该 env 时 JSON 与历史逐字节
     相同**。

跑法：`python3 tools/selftest_dynamic_spec.py`
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
PASS = FAIL = 0


def ok(msg: str) -> None:
    global PASS
    PASS += 1
    print(f"  \033[32mPASS\033[0m {msg}")


def bad(msg: str) -> None:
    global FAIL
    FAIL += 1
    print(f"  \033[31mFAIL\033[0m {msg}")


# ---------------------------------------------------------------------------
# ① dispatcher 分支逻辑（stub 注入）
# ---------------------------------------------------------------------------


def _install_stubs() -> None:
    """把 vllm 的几个模块换成最小 stub，好让 patch_cudagraph.py 能被 import。

    只 stub 到"模块级代码能跑、被测函数能调到"的程度；被 stub 掉的东西
    （BatchDescriptor / logger / 两个类）都不是被测对象。
    """

    class _Mode:
        """假 CUDAGraphMode：只区分"是不是 FULL"与"等不等于 FULL 本身"。"""

        def __init__(self, name: str, has_full: bool):
            self.name = name
            self._has_full = has_full

        def has_mode(self, other) -> bool:
            return self._has_full and other is CUDAGraphMode.FULL

        def __repr__(self) -> str:  # pragma: no cover - 调试用
            return f"<Mode {self.name}>"

    class CUDAGraphMode:  # noqa: N801 - 模拟上游类名
        FULL = _Mode("FULL", True)
        NONE = _Mode("NONE", False)
        FULL_DECODE_ONLY = _Mode("FULL_DECODE_ONLY", True)

    class BatchDescriptor:
        def __init__(self, num_tokens, num_reqs=None, uniform=False, has_lora=False, num_active_loras=0):
            self.num_tokens = num_tokens
            self.num_reqs = num_reqs
            self.uniform = uniform
            self.has_lora = has_lora
            self.num_active_loras = num_active_loras

        def __eq__(self, other):  # 键去重要靠它
            return isinstance(other, BatchDescriptor) and vars(self) == vars(other)

        def __hash__(self):
            return hash((self.num_tokens, self.num_reqs, self.uniform, self.has_lora))

        def __repr__(self):  # pragma: no cover
            return f"BD(t={self.num_tokens},r={self.num_reqs},u={self.uniform})"

    class CompilationConfig:
        def __init__(self):
            self.cudagraph_capture_sizes = [1, 2, 3, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64]
            self.max_cudagraph_capture_size = 64

        def adjust_cudagraph_sizes_for_spec_decode(self, ql, tp):
            # 真实实现会把桶上取整到 ql 的倍数；这里记录被调用过即可。
            self._adjusted_with = (ql, tp)
            self.cudagraph_capture_sizes = [8, 16, 24, 32]

    class CudagraphDispatcher:
        """假 dispatcher：保留 base 版 `initialize_cudagraph_keys` 的可观察行为。"""

        def __init__(self, vllm_config):
            self.vllm_config = vllm_config
            self.compilation_config = vllm_config.compilation_config
            self.uniform_decode_query_len = 1 + (vllm_config.num_speculative_tokens or 0)
            self.cudagraph_mode = CUDAGraphMode.NONE
            self.keys = set()
            self.init_calls = []          # (ql, sizes_snapshot)
            self.padding_rebuilds = 0
            self._bs_to_padded_graph_size = [0] * 65

        def initialize_cudagraph_keys(self, cudagraph_mode, uniform_decode_query_len=1):
            self.cudagraph_mode = cudagraph_mode
            # 忠实复刻上游 vllm/v1/cudagraph_dispatcher.py 的过滤：
            #   x <= ql * max_num_seqs and x >= ql
            # （上游**不**检查整除 —— 整除由 adjust_cudagraph_sizes_for_spec_decode
            #  保证，而 dynamic SD 下我们故意跳过了那一步，改由本补丁筛。）
            max_tokens = uniform_decode_query_len * self.vllm_config.scheduler_config.max_num_seqs
            effective = [
                s
                for s in self.compilation_config.cudagraph_capture_sizes
                if s <= max_tokens and s >= uniform_decode_query_len
            ]
            self.init_calls.append((uniform_decode_query_len, effective))
            for s in effective:
                if s % uniform_decode_query_len == 0:
                    self._create_padded_batch_descriptor(s, True, False)

        def _compute_bs_to_padded_graph_size(self):
            self.padding_rebuilds += 1
            self._bs_to_padded_graph_size = [0] * (self.compilation_config.max_cudagraph_capture_size + 1)

        def _create_padded_batch_descriptor(self, *a, **kw):  # 由被测模块覆盖
            raise AssertionError("should be replaced by the patched version")

        def add_cudagraph_key(self, mode, desc):
            self.keys.add((id(mode), desc))

    class _Logger:
        def __init__(self):
            self.messages = []

        def _rec(self, level, msg, *a):
            self.messages.append((level, msg % a if a else msg))

        def info(self, msg, *a):
            self._rec("info", msg, *a)

        def warning(self, msg, *a):
            self._rec("warning", msg, *a)

    logger = _Logger()

    mods = {
        "vllm": types.ModuleType("vllm"),
        "vllm.config": types.ModuleType("vllm.config"),
        "vllm.config.compilation": types.ModuleType("vllm.config.compilation"),
        "vllm.forward_context": types.ModuleType("vllm.forward_context"),
        "vllm.logger": types.ModuleType("vllm.logger"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.cudagraph_dispatcher": types.ModuleType("vllm.v1.cudagraph_dispatcher"),
    }
    mods["vllm.config"].CUDAGraphMode = CUDAGraphMode
    mods["vllm.config.compilation"].CompilationConfig = CompilationConfig
    mods["vllm.forward_context"].BatchDescriptor = BatchDescriptor
    mods["vllm.logger"].init_logger = lambda *a, **k: logger
    mods["vllm.v1.cudagraph_dispatcher"].CudagraphDispatcher = CudagraphDispatcher
    for name, mod in mods.items():
        sys.modules[name] = mod
    _install_stubs.logger = logger


def _load_patch_module():
    path = ROOT / "patches" / "files" / "patch_cudagraph.py"
    spec = importlib.util.spec_from_file_location("v41_patch_cudagraph", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Sched:
    def __init__(self, schedule, k):
        self.num_speculative_tokens_per_batch_size = schedule
        self.num_speculative_tokens = k


def _cfg(schedule=None, k=7, num_spec=None):
    cc = sys.modules["vllm.config.compilation"].CompilationConfig()

    class _SchedCfg:
        max_num_seqs = 8

    class _VllmCfg:
        pass

    v = _VllmCfg()
    v.num_speculative_tokens = k
    v.speculative_config = _Sched(schedule, k) if schedule is not None else None
    v.compilation_config = cc
    v.scheduler_config = _SchedCfg()
    return v, cc


def test_query_len_derivation(mod) -> None:
    print("[dynamic-spec] query_len 集合推导")
    v, _ = _cfg(schedule=None)
    if mod._dynamic_decode_query_lens(v) is None:
        ok("无 schedule ⇒ None（调用方全部走原路径）")
    else:
        bad("无 schedule 时未返回 None")

    v, _ = _cfg(schedule=[[1, 1, 7], [2, 8, 0]])
    got = mod._dynamic_decode_query_lens(v)
    if got == (1, 8):
        ok("[[1,1,7],[2,8,0]] ⇒ (1, 8)")
    else:
        bad(f"[[1,1,7],[2,8,0]] 期望 (1,8)，实际 {got}")

    # K 被 min(max_k, …) 钳制：表里写 9 但 num_spec=7 ⇒ 实际形状按 7 算
    v, _ = _cfg(schedule=[[1, 1, 9], [2, 8, 0]], k=7)
    got = mod._dynamic_decode_query_lens(v)
    if got == (1, 8):
        ok("表里 K=9 被钳到 7 ⇒ 仍是 (1, 8)（与调度器 min() 口径一致）")
    else:
        bad(f"钳制场景期望 (1,8)，实际 {got}")


def test_descriptor(mod) -> None:
    print("[dynamic-spec] 按本步 query_len 反推 num_reqs")
    CUDAGraphMode = sys.modules["vllm.config"].CUDAGraphMode
    v, _ = _cfg(schedule=[[1, 1, 7], [2, 8, 0]])
    d = sys.modules["vllm.v1.cudagraph_dispatcher"].CudagraphDispatcher(v)
    d.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    d._bs_to_padded_graph_size = list(range(65))  # 恒等映射，便于直接看 num_tokens

    # K=0 的步：num_tokens = batch = 8 ⇒ 8 个请求
    d._step_uniform_query_len = 1
    bd = d._create_padded_batch_descriptor(8, True, False)
    if bd.num_reqs == 8 and bd.uniform:
        ok("K=0：num_tokens=8 + query_len=1 ⇒ num_reqs=8（uniform）")
    else:
        bad(f"K=0 期望 num_reqs=8/uniform，实际 {bd}")

    # K=7 的步：num_tokens = 8 ⇒ 1 个请求
    d._step_uniform_query_len = 8
    bd = d._create_padded_batch_descriptor(8, True, False)
    if bd.num_reqs == 1 and bd.uniform:
        ok("K=7：num_tokens=8 + query_len=8 ⇒ num_reqs=1（uniform）")
    else:
        bad(f"K=7 期望 num_reqs=1/uniform，实际 {bd}")

    # 关键：同一个 num_tokens 在两种 ql 下必须给出**不同**的键，否则会串图
    d._step_uniform_query_len = 1
    a = d._create_padded_batch_descriptor(8, True, False)
    d._step_uniform_query_len = 8
    b = d._create_padded_batch_descriptor(8, True, False)
    if a != b:
        ok("num_tokens=8 在 ql=1 与 ql=8 下是**不同**的键（不会串图）")
    else:
        bad("两个 query_len 产生了相同键 ⇒ 会静默串图")

    # 缺图：20 不是 8 的倍数 ⇒ 降级为非 uniform（不 raise）
    d._step_uniform_query_len = 8
    try:
        bd = d._create_padded_batch_descriptor(20, True, False)
        if not bd.uniform and bd.num_reqs == 8:
            ok("查不到图时降级为非 uniform（不 raise、不打死引擎）")
        else:
            bad(f"缺图场景期望 uniform=False/num_reqs=8，实际 {bd}")
    except Exception as exc:  # noqa: BLE001
        bad(f"缺图场景抛异常（会把引擎带走）：{exc!r}")

    # 未设 step ql（建图期/静态路径）⇒ 用静态值，等价 base 版
    d._step_uniform_query_len = None
    d.uniform_decode_query_len = 8
    bd = d._create_padded_batch_descriptor(16, True, False)
    if bd.num_reqs == 2 and bd.uniform:
        ok("未设 step ql ⇒ 用静态 query_len（与 base 版等价）")
    else:
        bad(f"静态路径期望 num_reqs=2，实际 {bd}")


def test_initialize_and_adjust(mod) -> None:
    print("[dynamic-spec] 多 query_len 建图 + 跳过桶取整")
    v, cc = _cfg(schedule=[[1, 1, 7], [2, 8, 0]])
    CUDAGraphMode = sys.modules["vllm.config"].CUDAGraphMode
    d = sys.modules["vllm.v1.cudagraph_dispatcher"].CudagraphDispatcher(v)
    d.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    # 模拟"dispatcher 构造时"装上 dynamic 状态；`_v41_dynamic_sd_enabled`
    # 由 runner 侧补丁在调 initialize 前显式写入（opt-in）。
    d._dynamic_decode_query_lens = mod._dynamic_decode_query_lens(v)
    d._v41_dynamic_sd_enabled = True
    cc._v41_dynamic_sd = True

    before = list(cc.cudagraph_capture_sizes)
    d.initialize_cudagraph_keys(CUDAGraphMode.FULL_DECODE_ONLY, 8)
    qls = [c[0] for c in d.init_calls]
    if qls == [8, 1]:
        ok(f"按 (8, 1) 各建一遍 keys（实际调用序列 {qls}）")
    else:
        bad(f"建图调用序列期望 [8, 1]，实际 {qls}")

    # 额外那一遍必须用"筛过的"桶（只留 ql 的整数倍），否则建图期就撞整除断言
    sizes_for_1 = d.init_calls[1][1]
    if sizes_for_1 and all(s % 1 == 0 and s <= 8 for s in sizes_for_1):
        ok(f"ql=1 那一遍只用 ≤8 的桶：{sizes_for_1}")
    else:
        bad(f"ql=1 那一遍的桶不合理：{sizes_for_1}")

    if cc.cudagraph_capture_sizes == before:
        ok("建图后桶列表已恢复（未把子集状态留给运行时）")
    else:
        bad(f"桶列表未恢复：{cc.cudagraph_capture_sizes} != {before}")

    if d.padding_rebuilds >= 1:
        ok(f"恢复后重算了 padding 表（{d.padding_rebuilds} 次）")
    else:
        bad("恢复后没有重算 padding 表 ⇒ padding 会停留在子集状态")

    # ★ 回归：建图期**不得**把非整倍桶变成伪键（真机踩到过）
    v3, _ = _cfg(schedule=[[1, 1, 7], [2, 8, 0]])
    d3 = sys.modules["vllm.v1.cudagraph_dispatcher"].CudagraphDispatcher(v3)
    d3.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    d3._v41_dynamic_sd_enabled = True
    d3._v41_qlens_cache = mod._dynamic_decode_query_lens(v3)   # 懒算缓存的等价写法
    d3.initialize_cudagraph_keys(CUDAGraphMode.FULL_DECODE_ONLY, 8)
    bad_keys = [bd for _, bd in d3.keys if getattr(bd, "uniform", False) is False
                and bd.num_tokens in (12, 20)]
    if not bad_keys:
        ok("建图期非整倍桶（12/20）被跳过，未产生 non-uniform 伪键")
    else:
        bad(f"建图期产生了伪键：{bad_keys}（会把 set_draft_graph_params 的尺寸算歪）")

    # ★ 回归：不依赖 CudagraphDispatcher.__init__ 被 patch（真机踩到的静默失效）
    src = (ROOT / "patches" / "files" / "patch_cudagraph.py").read_text()
    if "CudagraphDispatcher.__init__ = " not in src and "_dispatcher_init" not in src:
        ok("不再 patch CudagraphDispatcher.__init__（改为懒算，免疫 import 顺序）")
    else:
        bad("仍在 patch CudagraphDispatcher.__init__ ⇒ 会因 import 晚于构造而静默失效")
    if "_v41_extra_query_lens" in src and "self.vllm_config" in src:
        ok("query_lens 走懒算（用 self.vllm_config）")
    else:
        bad("query_lens 不是懒算 ⇒ 依赖实例属性，会静默失效")
    # 判据必须始终可见（INFO 被日志级别过滤过，导致无法从日志证明路径走了）
    if 'logger.warning(\n        "[dynamic-spec] building decode graphs' in src:
        ok("建图判据是 warning 级别（始终可见，可作唯一判据）")
    else:
        bad("建图判据不是 warning ⇒ 会被日志级别过滤，无法证明路径真的走了")

    cc2 = sys.modules["vllm.config.compilation"].CompilationConfig()
    cc2._v41_dynamic_sd = True
    cc2.adjust_cudagraph_sizes_for_spec_decode(8, 8)
    if not hasattr(cc2, "_adjusted_with"):
        ok("dynamic SD ⇒ 跳过 adjust（小桶 1/2/3/4 得以保留）")
    else:
        bad("dynamic SD 下仍然做了桶取整 ⇒ K=0 没有图可用")

    cc3 = sys.modules["vllm.config.compilation"].CompilationConfig()
    cc3.adjust_cudagraph_sizes_for_spec_decode(8, 8)
    if getattr(cc3, "_adjusted_with", None) == (8, 8):
        ok("静态路径 ⇒ 正常委派给原实现（行为不变）")
    else:
        bad("静态路径没有委派 ⇒ 改了既有行为")

    # 负控：没给 opt-in 的 dispatcher（真实场景 = draft proposer 自己那个，
    # 它用同一个 vllm_config）**不能**多建图，否则会白捕几轮 ql=1 的 draft 图。
    v2, _ = _cfg(schedule=[[1, 1, 7], [2, 8, 0]])
    d2 = sys.modules["vllm.v1.cudagraph_dispatcher"].CudagraphDispatcher(v2)
    d2.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    d2._dynamic_decode_query_lens = mod._dynamic_decode_query_lens(v2)
    d2.initialize_cudagraph_keys(CUDAGraphMode.FULL_DECODE_ONLY, 8)
    if [c[0] for c in d2.init_calls] == [8]:
        ok("无 opt-in（draft proposer 的 dispatcher）⇒ 只建一遍，不被污染")
    else:
        bad(f"无 opt-in 时仍多建了图：{[c[0] for c in d2.init_calls]}（draft 会被白捕一轮）")


def test_patch_contains_reassert() -> None:
    """运行期补丁必须带"抵消上游 PIECEWISE 降级"的分支（默认关、env 开）。

    这条是**纯文本检查**：我们没法在离线环境里跑真 vLLM 的 config 初始化，
    但可以钉住"补丁里确实有这个分支、且默认值是 0"，防止有人在重构时把它删掉
    —— 删掉的表现是"开了 V41_CED_DYNAMIC_SPEC_REASSERT_MODE=1 也没用"，
    然后又在模型构造期炸，属于浪费一整轮重启的坑。
    """
    print("[dynamic-spec] 上游降级门补丁：存在性 + 默认关")
    gate = ROOT / "experimental" / "ced" / "core_config_dynamic_sd_gate.patch"
    if not gate.is_file():
        bad("缺 core_config_dynamic_sd_gate.patch ⇒ 上游降级无法关闭，dynamic K 起不来")
        return
    gtxt = gate.read_text()
    if "V41_CED_DYNAMIC_SPEC_FULL_GRAPHS" in gtxt:
        ok("gate 补丁含 FULL_GRAPHS 开关")
    else:
        bad("gate 补丁里没有开关 ⇒ 关了没效果")
    if '"V41_CED_DYNAMIC_SPEC_FULL_GRAPHS", "0"' in gtxt:
        ok("gate 补丁默认值是 0（不显式开就保持上游行为）")
    else:
        bad("gate 补丁默认值不是 0 ⇒ 会隐式绕过上游的可靠性保护")
    # 这道门在 VllmConfig.__post_init__ 里，必须**早于** runner 的任何代码；
    # 若有人把它改到 runner 里「事后改回来」，模型构造期就已经失败了。
    if "_maybe_override_dynamic_sd_cudagraph_mode" in gtxt:
        ok("gate 补丁确实打在 _maybe_override_dynamic_sd_cudagraph_mode（__post_init__ 路径）")
    else:
        bad("gate 补丁没有指向那道降级函数 ⇒ 打错位置了")


# ---------------------------------------------------------------------------
# ② serve_v2.sh 的 SP_SCHEDULE → --speculative-config
# ---------------------------------------------------------------------------


def _run_serve_v2(env_extra: dict) -> str:
    """用假 vllm 干跑 serve_v2.sh，拿回它实际拼出的 --speculative-config。"""
    with tempfile.TemporaryDirectory() as td:
        binp = pathlib.Path(td) / "bin"
        binp.mkdir()
        fake = binp / "vllm"
        fake.write_text('#!/usr/bin/env bash\nprintf "ARG|%s\\n" "$@"\n')
        fake.chmod(0o755)
        env = dict(os.environ)
        env["PATH"] = f"{binp}:{env['PATH']}"
        env.update(env_extra)
        for k in ("SP_SCHEDULE", "SPEC", "SP_TOKENS"):
            if k not in env_extra:
                env.pop(k, None)
        env["SPEC"] = env_extra.get("SPEC", "1")
        out = subprocess.run(
            ["bash", str(ROOT / "scripts" / "serve_v2.sh")],
            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120,
        )
        return out.stdout + out.stderr


def _extract_spec_config(text: str) -> str | None:
    """从假 vllm 的 `printf 'ARG|%s\n' "$@"` 输出里取出 --speculative-config 的值。

    注意参数是**成对**打印的：`ARG|--speculative-config` 单独一行，值在下一行。
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "ARG|--speculative-config":
            if i + 1 < len(lines):
                return lines[i + 1].split("ARG|", 1)[1].strip() or None
            return None
        if line.startswith("ARG|--speculative-config="):
            return line.split("=", 1)[1].strip() or None
    return None


def test_schedule_to_json() -> None:
    print("[dynamic-spec] SP_SCHEDULE → --speculative-config")
    # enforce_eager 由 SPEC_EAGER 决定；而 SPEC_EAGER 是 inner.sh 从 DRAFT_GRAPH
    # 推出来的（DRAFT_GRAPH=1 ⇒ SPEC_EAGER=0）。serve_v2.sh 只读 SPEC_EAGER，
    # 所以这里按 inner.sh 的推导结果喂进去。两种组合都钉死，
    # 避免"顺手改了静态路径"这种回归。
    base_dg = _extract_spec_config(
        _run_serve_v2({"SPEC": "1", "SP_TOKENS": "7", "SPEC_EAGER": "0"})
    )
    expect_dg = '{"method":"dspark","num_speculative_tokens":7,"enforce_eager":false}'
    if base_dg == expect_dg:
        ok("不设 SP_SCHEDULE（SPEC_EAGER=0）⇒ JSON 与历史**逐字节相同**")
    else:
        bad(f"SPEC_EAGER=0 时 JSON 变了：{base_dg!r} != {expect_dg!r}")

    base_nd = _extract_spec_config(_run_serve_v2({"SPEC": "1", "SP_TOKENS": "7"}))
    expect_nd = '{"method":"dspark","num_speculative_tokens":7,"enforce_eager":true}'
    if base_nd == expect_nd:
        ok("不设 SP_SCHEDULE（SPEC_EAGER 未设=默认 1）⇒ enforce_eager=true（同历史）")
    else:
        bad(f"SPEC_EAGER 未设时 JSON 变了：{base_nd!r} != {expect_nd!r}")

    withsched = _extract_spec_config(
        _run_serve_v2({"SPEC": "1", "SP_TOKENS": "7", "SP_SCHEDULE": "1,1,7;2,8,0"})
    )
    try:
        parsed = json.loads(withsched or "{}")
    except Exception as exc:  # noqa: BLE001
        parsed = {}
        bad(f"带 schedule 的 JSON 不可解析：{exc!r} / {withsched!r}")
    if parsed.get("num_speculative_tokens_per_batch_size") == [[1, 1, 7], [2, 8, 0]]:
        ok("SP_SCHEDULE=1,1,7;2,8,0 ⇒ [[1,1,7],[2,8,0]]（vLLM 原生格式）")
    else:
        bad(f"schedule 未正确展开：{parsed.get('num_speculative_tokens_per_batch_size')!r}")
    if parsed.get("num_speculative_tokens") == 7 and parsed.get("method") == "dspark":
        ok("schedule 不影响 method / 最大 K（K=7 仍写在 num_speculative_tokens）")
    else:
        bad(f"method 或最大 K 被改坏：{parsed!r}")

    # 负控：格式错必须**退回固定 K** 而不是拼出坏 JSON
    text = _run_serve_v2({"SPEC": "1", "SP_TOKENS": "7", "SP_SCHEDULE": "1,2"})
    badcfg = _extract_spec_config(text)
    if "num_speculative_tokens_per_batch_size" not in (badcfg or ""):
        ok("SP_SCHEDULE 格式错 ⇒ 忽略并退回固定 K（不生成坏 JSON）")
    else:
        bad(f"格式错时仍拼进了 schedule：{badcfg!r}")
    if "WARNING" in text and "SP_SCHEDULE" in text:
        ok("格式错时**响亮告警**（不是静默忽略）")
    else:
        bad("格式错时没有告警 ⇒ 用户会以为动态 K 生效了")


def main() -> int:
    print("=" * 72)
    _install_stubs()
    mod = _load_patch_module()
    test_query_len_derivation(mod)
    test_descriptor(mod)
    test_initialize_and_adjust(mod)
    test_patch_contains_reassert()
    test_schedule_to_json()
    print("=" * 72)
    if FAIL == 0:
        print(f"\033[32m全部通过 ✅\033[0m（{PASS} 项）")
        return 0
    print(f"\033[31m有 {FAIL} 项失败 ❌\033[0m（通过 {PASS} 项）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
