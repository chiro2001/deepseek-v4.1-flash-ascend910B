import ast, sys
from pathlib import Path
p = Path("/home/l00886679/dcpw/vllm_ascend/attention/dsa_v41.py")
s = p.read_text()
if "V41_DBO_DEBUG" in s:
    print("已有调试"); sys.exit(0)
# 在 rotary 调用前打印
old = """            kv = attn.kv_norm(kv).view(-1, 1, attn.head_dim)
            torch.ops._C_ascend.inplace_partial_rotary_mul("""
new = """            kv = attn.kv_norm(kv).view(-1, 1, attn.head_dim)
            import os as _os
            if _os.environ.get("V41_DBO_DEBUG") == "1":
                print("[DBO-DBG] rotary kv=%s cos=%s sin=%s ubid=%s pos_len=%s"
                      % (tuple(kv.shape), tuple(cos.shape), tuple(sin.shape),
                         getattr(self, "_ubid", "?"),
                         (len(metadata.positions) if metadata.positions is not None else None)),
                      flush=True)
            torch.ops._C_ascend.inplace_partial_rotary_mul("""
assert old in s
s = s.replace(old, new, 1)
ast.parse(s); p.write_text(s)
print("已加调试打印")
