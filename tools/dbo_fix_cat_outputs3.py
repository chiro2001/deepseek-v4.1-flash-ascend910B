#!/usr/bin/env python3
"""_cat_ubatch_outputs：带深度/长度的调试 + 更稳健的空值处理。"""
import ast, sys
from pathlib import Path
p = Path("/home/l00886679/dcpw/vllm_ascend/worker/npu_ubatch_wrapper.py")
s = p.read_text()
start = s.find("def _cat_ubatch_outputs(")
if start < 0:
    print("未找到函数", file=sys.stderr); sys.exit(2)
end = s.find("\n\n\n", start)
if end < 0:
    end = s.find("\n\n@dataclass", start)

new = '''_CAT_DEPTH = [0]


def _describe(x):
    if isinstance(x, torch.Tensor):
        return "Tensor%s" % (tuple(x.shape),)
    if isinstance(x, (tuple, list)):
        return "%s(len=%d)[%s]" % (type(x).__name__, len(x),
                                   (", ".join(_describe(i) for i in x[:3]) if x else ""))
    if isinstance(x, dict):
        return "dict(keys=%s)" % (list(x.keys())[:3],)
    return type(x).__name__


def _cat_ubatch_outputs(sorted_results):
    """按 batch 维拼接各 ubatch 的输出（递归；容忍空序列 / None / 嵌套 / dict）。

    最后一层兜底：**任何情况下都不返回"比输入更少元素"的容器**，
    否则调用方 `hidden_states, _ = outputs` 会 unpack 失败
    （实测报过 expected 2 got 0）。
    """
    import os as _os
    _dbg = _os.environ.get("V41_DBO_DEBUG") == "1"
    d = _CAT_DEPTH[0]
    _CAT_DEPTH[0] = d + 1
    try:
        if _dbg:
            print("[DBO-CAT%d] in: %s" % (d, _describe(sorted_results)), flush=True)
        out = _cat_impl(sorted_results, d, _dbg)
        if _dbg:
            print("[DBO-CAT%d] out: %s" % (d, _describe(out)), flush=True)
        return out
    finally:
        _CAT_DEPTH[0] = d


def _cat_impl(sorted_results, d, _dbg):
    # 非容器：原样返回
    if isinstance(sorted_results, torch.Tensor) or sorted_results is None:
        return sorted_results
    if isinstance(sorted_results, dict):
        if not sorted_results:
            return sorted_results
        v0 = next(iter(sorted_results.values()))
        if isinstance(v0, (tuple, list, dict)) or v0 is None:
            return {k: _cat_ubatch_outputs([r[k] for r in sorted_results]) for k in sorted_results}
        return torch.cat(list(sorted_results.values()), dim=0)
    if not isinstance(sorted_results, (tuple, list)):
        return sorted_results
    if len(sorted_results) == 0:
        return sorted_results
    if all(x is None for x in sorted_results):
        return None

    first = sorted_results[0]
    if isinstance(first, (tuple, list, dict)):
        if isinstance(first, dict):
            return {k: _cat_ubatch_outputs([r[k] for r in sorted_results]) for k in first}
        cls = type(first)
        return cls(_cat_ubatch_outputs([r[i] for r in sorted_results]) for i in range(len(first)))
    if first is None:
        # 首元素 None 但并非全 None ⇒ 逐元素处理，None 位置保持 None
        n = max((len(r) for r in sorted_results if isinstance(r, (tuple, list))), default=0)
        if n == 0:
            return sorted_results
        return [_cat_ubatch_outputs([(r[i] if isinstance(r, (tuple, list)) else None)
                                     for r in sorted_results]) for i in range(n)]
    if isinstance(first, torch.Tensor):
        if any(not isinstance(x, torch.Tensor) for x in sorted_results):
            return sorted_results          # 混合类型：保守原样返回
        return torch.cat(sorted_results, dim=0)
    return sorted_results


'''
s = s[:start] + new + s[end:]
ast.parse(s)
p.write_text(s)
print("_cat_ubatch_outputs 已升级（带深度调试 + None 处理 + 不缩容兜底）")
