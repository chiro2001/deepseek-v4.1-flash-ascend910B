#!/usr/bin/env python3
"""模型输出可能是 tuple（V4.1 + spec decode 的 aux hidden states）⇒ 用上游 _cat_ubatch_outputs 的逻辑。"""
import ast, sys
from pathlib import Path
p = Path("/home/l00886679/dcpw/vllm_ascend/worker/npu_ubatch_wrapper.py")
s = p.read_text()
if "_cat_ubatch_outputs" in s:
    print("已打过"); sys.exit(0)
old = """        sorted_results = [value for position, value in sorted(results)]
        result = torch.cat(sorted_results, dim=0)
        return result"""
new = """        sorted_results = [value for position, value in sorted(results)]
        return _cat_ubatch_outputs(sorted_results)"""
if old not in s:
    print("锚点未命中"); sys.exit(2)
s = s.replace(old, new, 1)
# 插入 helper（放在 UbatchMetadata 定义之前）
helper = '''def _cat_ubatch_outputs(sorted_results: list):
    """按 batch 维拼接各 ubatch 的输出。

    上游 GPU 版同款：多数模型返回单个 hidden-states 张量；
    带辅助输出的目标模型（如 spec decode 收集 aux hidden states）返回 tuple
    ⇒ 要对 tuple 的每个分量分别 cat，否则 torch.cat 会把 tuple 当成元素。
    """
    if sorted_results and isinstance(sorted_results[0], tuple):
        return tuple(torch.cat(parts, dim=0) for parts in zip(*sorted_results))
    return torch.cat(sorted_results, dim=0)


'''
anchor = "@dataclass\nclass UbatchMetadata:"
if anchor in s:
    s = s.replace(anchor, helper + anchor, 1)
else:
    print("helper 插入锚点未命中"); sys.exit(3)
ast.parse(s); p.write_text(s)
print("已加 _cat_ubatch_outputs")
