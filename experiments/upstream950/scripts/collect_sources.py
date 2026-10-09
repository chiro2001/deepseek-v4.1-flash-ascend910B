"""Read-only snapshot of relevant installed sources and their exact hashes."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

base = Path("/vllm-workspace/vllm-ascend")
root = Path("/work/results/source_snapshot")
if root.exists():
    shutil.rmtree(root)
root.mkdir(parents=True, exist_ok=True)
paths = [
    base / "vllm_ascend/attention/dsa_v41.py",
    base / "vllm_ascend/attention/dsa_v1.py",
    base / "vllm_ascend/models/deepseek_v41/indexer.py",
    base / "vllm_ascend/ops/fused_moe/token_dispatcher.py",
    base / "vllm_ascend/ops/fused_moe/moe_mlp.py",
    base / "vllm_ascend/ops/fused_moe/routed_experts.py",
    base / "vllm_ascend/quantization/methods/w4a8/w4a8.py",
]
for path in (base / "csrc").rglob("*"):
    if path.is_file() and path.suffix in [".h", ".cpp"]:
        if "build" in path.relative_to(base).parts:
            continue
        if any(key in str(path).lower() for key in
               ["sparse_flash_mla", "lightning_indexer", "mla_prolog"]):
            paths.append(path)
manifest = {}
for path in sorted(set(paths)):
    if not path.exists():
        continue
    rel = str(path.relative_to(base))
    dest = root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)
    dest.chmod(0o644)
    manifest[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
heads = {name: subprocess.check_output(["git", "-C", "/vllm-workspace/" + name,
                                       "rev-parse", "HEAD"], text=True).strip()
         for name in ["vllm", "vllm-ascend"]}
result = {"heads": heads, "files": manifest}
(root.parent / "source_manifest.json").write_text(json.dumps(result, indent=2))
print(json.dumps({"heads": heads, "source_files": len(manifest)}))
