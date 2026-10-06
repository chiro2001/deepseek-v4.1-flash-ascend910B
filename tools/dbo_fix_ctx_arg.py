#!/usr/bin/env python3
import ast, sys
from pathlib import Path
p = Path("/home/l00886679/dcpw/vllm_ascend/worker/model_runner_v1.py")
s = p.read_text()

a1 = """                has_sinks=self._has_sinks,
                eplb_heat_collection_status=self.eplb_heat_collection_status if self.dynamic_eplb else False,
            ),"""
b1 = """                has_sinks=self._has_sinks,
                eplb_heat_collection_status=self.eplb_heat_collection_status if self.dynamic_eplb else False,
                ubatch_slices=ubatch_slices_padded if 'ubatch_slices_padded' in locals() else None,
            ),"""
a2 = """                has_sinks = self._has_sinks,
                eplb_heat_collection_status=self.eplb_heat_collection_status if self.dynamic_eplb else False,
            ):"""
b2 = """                has_sinks = self._has_sinks,
                eplb_heat_collection_status=self.eplb_heat_collection_status if self.dynamic_eplb else False,
                ubatch_slices=ubatch_slices_padded if pad_attn else ubatch_slices,
            ):"""
n = 0
if a1 in s and "ubatch_slices_padded if 'ubatch_slices_padded' in locals()" not in s:
    s = s.replace(a1, b1, 1); n += 1
if a2 in s and "ubatch_slices=ubatch_slices_padded if pad_attn" not in s.split(a2)[0][-400:]:
    s = s.replace(a2, b2, 1); n += 1
if n == 0:
    print("锚点未命中或已打", file=sys.stderr); sys.exit(2)
ast.parse(s); p.write_text(s)
print("已给 %d 个调用点补 ubatch_slices" % n)
