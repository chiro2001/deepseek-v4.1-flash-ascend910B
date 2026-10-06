#!/usr/bin/env python3
"""修 RoPE 缓存与 ubatch 冲突（与 _publish_task 同类：共享缓存必须按 ubatch 分键）。"""
import ast, sys
from pathlib import Path

p = Path("/home/l00886679/dcpw/vllm_ascend/attention/dsa_v41.py")
s = p.read_text()
if "_rope_key" in s:
    print("已打过"); sys.exit(0)
old = '''            rope = batch_shared.get("rope")
            if rope is None:
                rope = get_cos_and_sin_dsa(positions, use_cache=coordinates["num_prefills"] == 0)
                batch_shared["rope"] = rope
            cos, sin = rope'''
if old not in s:
    print("锚点未命中"); sys.exit(2)
new = '''            # [DBO] rope 缓存必须按 ubatch 分键：不同 ubatch 的 positions 子序列不同，
            # 共用 key 会让第二个 ubatch 拿到**全量** cos/sin ⇒
            # inplace_partial_rotary_mul 报 "dim0 must be equal"。
            _rope_key = "rope" if getattr(self, "_ubid", None) is None else f"rope:ub{self._ubid}"
            rope = batch_shared.get(_rope_key)
            if rope is None:
                rope = get_cos_and_sin_dsa(positions, use_cache=coordinates["num_prefills"] == 0)
                batch_shared[_rope_key] = rope
            cos, sin = rope'''
s = s.replace(old, new, 1)
ast.parse(s)
p.write_text(s)
print("rope 缓存已按 ubatch 分键")
