from pathlib import Path
import ast
p = Path("/home/l00886679/dcpw/vllm_ascend/worker/npu_ubatch_wrapper.py")
s = p.read_text()
if "V41_DBO_DBG2" in s:
    print("已有"); raise SystemExit(0)
old = "        return self._run_ubatches(ubatch_metadata, self.runnable)"
new = '''        import os as _os
        if _os.environ.get("V41_DBO_DEBUG") == "1":
            print("[DBO-DBG2] args=%d kwargs=%s input_ids=%s positions=%s embeds=%s ubatches=%s ntoks=%s"
                  % (len(args), sorted(kwargs.keys()),
                     (tuple(input_ids.shape) if hasattr(input_ids, "shape") else input_ids),
                     (tuple(positions.shape) if hasattr(positions, "shape") else positions),
                     (tuple(inputs_embeds.shape) if hasattr(inputs_embeds, "shape") else inputs_embeds),
                     len(ubatch_slices), [m.num_tokens for m in ubatch_metadata]), flush=True)
        return self._run_ubatches(ubatch_metadata, self.runnable)'''
if old not in s:
    print("锚点未命中"); raise SystemExit(2)
s = s.replace(old, new, 1)
ast.parse(s); p.write_text(s)
print("wrapper 已加调试")
