import ast, re, sys
from pathlib import Path
p = Path("/home/l00886679/cedpd-repo/patches/files/ascend_forward_context.py")
s = p.read_text()
# 先清掉可能的坏插入
s = s.replace("    eplb_heat_collection_status: bool = False,\n,\n    ubatch_slices=None,\n):",
              "    eplb_heat_collection_status: bool = False,\n    ubatch_slices=None,\n):")
if "ubatch_slices=None," not in s:
    m = re.search(r"def set_ascend_forward_context\((.*?)\n\):", s, re.S)
    if not m:
        print("签名锚点未命中"); sys.exit(2)
    s = s.replace(m.group(0), m.group(0)[:-3] + "\n    ubatch_slices=None,\n):", 1)
if "forward_context.ubatch_slices = ubatch_slices" not in s:
    a = "        forward_context.capturing = False"
    s = s.replace(a, a + "\n        forward_context.ubatch_slices = ubatch_slices", 1)
ast.parse(s)
p.write_text(s)
print("ascend_forward_context.py 已修（含 ubatch_slices 形参 + 属性赋值）")
