#!/usr/bin/env python3
"""[DBO-CAPS 2026-10-06] 修 DBO 运行时 engine 报 `Supported tasks: []` 的根因。

实测链条：
  · model_runner_v1.py:4220 用 NPUUBatchWrapper 包住 self.model；
  · Python 3.12 的 runtime_checkable Protocol `isinstance` 走 `getattr_static`，
    **不会**调用 wrapper 的 `__getattr__` ⇒ `is_text_generation_model(wrapper)=False`
    （已用最小复现验证：plain=True / wrapped=False，proto attrs =
     [compute_logits,embed_input_ids,forward]）；
  · GPUModelRunner.get_supported_tasks → get_supported_generation_tasks →
    is_text_generation_model(self.get_model()) ⇒ 返回空 tuple；
  · API server 拿到空 supported_tasks ⇒ 不注册 /v1/completions、/v1/chat/completions、
    /v1/responses 等 generate 路由（openapi 只剩 9 条），表现为“服务起来了但全 404”。

修法：给 NPUModelRunner 加一个 `get_supported_tasks`，探测期间临时把 self.model
解包到最内层真实模型（NPUUBatchWrapper/BreakableACLGraphWrapper 都有 unwrap()）。
只影响这一次 RPC，不改运行期模型对象。
"""
from pathlib import Path
import shutil, time, ast

P = Path("/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py")
s = P.read_text()
if "DBO-CAPS" in s:
    print("已打过 DBO-CAPS，跳过")
    raise SystemExit(0)

anchor = "class NPUModelRunner(GPUModelRunner):\n"
assert anchor in s, "NPUModelRunner 锚点未找到"

method = """class NPUModelRunner(GPUModelRunner):
    # [DBO-CAPS 2026-10-06] DBO 的 NPUUBatchWrapper 会遮住模型的 capability：
    # Python 3.12 runtime Protocol 用 getattr_static 做 isinstance，绕过 __getattr__，
    # 于是 is_text_generation_model(wrapper)=False ⇒ engine "Supported tasks: []"
    # ⇒ API server 不注册 /v1/completions 等 generate 路由。
    # 这里在 capability 探测期间临时解包到最内层模型（只影响本次 RPC）。
    def get_supported_tasks(self):
        model = self.model
        inner = model
        for _ in range(4):
            unwrap = getattr(inner, "unwrap", None)
            if not callable(unwrap):
                break
            inner = unwrap()
        if inner is model:
            return super().get_supported_tasks()
        try:
            self.model = inner
            return super().get_supported_tasks()
        finally:
            self.model = model

"""
bak = P.with_suffix(".py.bak_dbocaps_%s" % time.strftime("%m%d_%H%M%S"))
shutil.copy2(P, bak)
s = s.replace(anchor, method, 1)
ast.parse(s)
P.write_text(s)
print("patched:", P, "backup:", bak)
