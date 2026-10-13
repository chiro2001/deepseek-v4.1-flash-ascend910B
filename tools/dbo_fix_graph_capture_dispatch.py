#!/usr/bin/env python3
"""[DBO-GRAPH] 让 ubatch 双流分支在**捕获期内**走 stream 版实现（而不是 threading 版）。

背景（本轮实测 + 上一轮微基准）：
  · `_run_ubatches()`（threading + 每线程 set_stream）在 ACL graph 捕获期内提交的算子
    落在**非捕获流**上 ⇒ 图里只留下不完整的双流结构，replay 时挂死
    （实测：conc≥4 首个 2-ubatch decode step 卡住，EngineCore 报 shm 广播超时）；
  · `tools/tiny_graph_ms.py` 已验证**一张图能捕获双流分支**，关键规则是
    **fork event 必须 record 在捕获根流上**（不是默认流）；
  · 本文件的 `_run_ubatches_graph()` 已按该规则写好，但**从未被调用**。

改法：`__call__` 里检测 `torch.npu.is_current_stream_capturing()`，捕获期走
stream 版；eager 期仍走 threading 版（已跑通）。开关 `V41_DBO_GRAPH`（默认 1）。
"""
from pathlib import Path
import shutil
import time
import ast

P = Path("/vllm-workspace/vllm-ascend/vllm_ascend/worker/npu_ubatch_wrapper.py")
s = P.read_text()
if "DBO-GRAPH" in s:
    print("已打过 DBO-GRAPH，跳过")
    raise SystemExit(0)

old_dispatch = "        return self._run_ubatches(ubatch_metadata, self.runnable)\n"
new_dispatch = (
    "        # [DBO-GRAPH 2026-10-06] 捕获期必须用 stream 版（fork event 记在捕获根流上）。\n"
    "        import os as _dbo_os\n"
    "        if _dbo_os.environ.get(\"V41_DBO_GRAPH\", \"1\") == \"1\":\n"
    "            try:\n"
    "                _capturing = bool(torch.npu.is_current_stream_capturing())\n"
    "            except Exception:\n"
    "                _capturing = False\n"
    "            if _capturing:\n"
    "                return self._run_ubatches_graph(ubatch_metadata, self.runnable, compute_stream)\n"
    "        return self._run_ubatches(ubatch_metadata, self.runnable)\n"
)
assert old_dispatch in s, "dispatch 锚点未找到"
s = s.replace(old_dispatch, new_dispatch, 1)

start = s.index("    def _run_ubatches_graph(self, ubatch_metadata, model, root_stream):")
end = s.index("    def supports_graph(self) -> bool:", start)
body = [
    "    def _run_ubatches_graph(self, ubatch_metadata, model, root_stream):",
    '        """捕获期版本：fork 在根流、两分支各走一条流、join 回根流。',
    "",
    "        与 threading 版语义等价（两个 ubatch 的模型输出按顺序拼接），",
    "        但所有 event 都在捕获区内 record/wait，可被一张 ACL graph 完整捕获。",
    '        """',
    "",
    "        @torch.inference_mode()",
    "        def _submit(metadata, model, stream):",
    "            ctx = metadata.context",
    "            from vllm.forward_context import override_forward_context",
    "",
    "            cm = override_forward_context(ctx.forward_context)",
    "            cm.__enter__()",
    "            try:",
    "                return model(",
    "                    input_ids=metadata.input_ids,",
    "                    positions=metadata.positions,",
    "                    intermediate_tensors=metadata.intermediate_tensors,",
    "                    inputs_embeds=metadata.inputs_embeds,",
    "                )",
    "            finally:",
    "                cm.__exit__(None, None, None)",
    "",
    "        e_fork = torch.npu.Event()",
    "        e_join = torch.npu.Event()",
    "        e_fork.record(root_stream)   # ★ 必须 record 在捕获根流上",
    "        outputs = [None] * len(ubatch_metadata)",
    "        side = self._dbo_side_stream",
    "        if side is None:",
    "            side = self._dbo_side_stream = torch.npu.Stream()",
    "        with torch.npu.stream(side):",
    "            side.wait_event(e_fork)",
    "            outputs[1] = _submit(ubatch_metadata[1], model, side)",
    "            e_join.record(side)",
    "        outputs[0] = _submit(ubatch_metadata[0], model, root_stream)",
    "        root_stream.wait_event(e_join)",
    "        return _cat_ubatch_outputs(outputs)",
    "",
]
s = s[:start] + "\n".join(body) + "\n" + s[end:]

old_init = "        self.ready_barrier = threading.Barrier(self.vllm_config.parallel_config.num_ubatches + 1)\n"
new_init = old_init + "        self._dbo_side_stream = None   # [DBO-GRAPH] 捕获期侧流（懒创建）\n"
assert old_init in s, "init 锚点未找到"
s = s.replace(old_init, new_init, 1)

bak = P.with_suffix(".py.bak_dbograph_%s" % time.strftime("%m%d_%H%M%S"))
shutil.copy2(P, bak)
ast.parse(s)
P.write_text(s)
print("patched:", P, "backup:", bak)
