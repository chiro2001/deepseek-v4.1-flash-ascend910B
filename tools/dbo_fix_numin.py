from pathlib import Path
import ast
p = Path("/home/l00886679/dcpw/vllm_ascend/worker/model_runner_v1.py")
s = p.read_text()
old = '''        for nm in ("decode_token_per_req", "graph_pad_size", "num_input_tokens", "attn_state"):
            extra[nm] = getattr(common_attn_metadata, nm, None)'''
new = '''        for nm in ("decode_token_per_req", "graph_pad_size", "attn_state"):
            extra[nm] = getattr(common_attn_metadata, nm, None)
        # num_input_tokens 必须等于本 ubatch 的（含 padding 的）token 数，
        # 否则 builder 里 `slot_mapping[:num_input_tokens]` 会与切片后的长度不符。
        try:
            extra["num_input_tokens"] = len(base.slot_mapping)
        except Exception:
            extra["num_input_tokens"] = getattr(common_attn_metadata, "num_input_tokens", None)'''
if old not in s:
    print("锚点未命中"); raise SystemExit(2)
s = s.replace(old, new, 1)
ast.parse(s); p.write_text(s)
print("num_input_tokens 已改为按 ubatch 切片")
