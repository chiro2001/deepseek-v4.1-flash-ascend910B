"""Record the actual runtime and local API documentation before candidate work."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import torch
import torch_npu
import vllm
import vllm_ascend

APIS = [
    "npu_mla_prolog", "npu_mla_prolog_v2", "npu_mla_prolog_v3",
    "npu_kv_rmsnorm_rope_cache", "npu_grouped_matmul",
    "npu_sparse_flash_attention", "npu_quant_lightning_indexer",
    "npu_moe_init_routing_v3", "npu_moe_gating_top_k",
]

torch.npu.set_device(0)
x = torch.arange(16, device="npu", dtype=torch.float32)
assert x.sum().item() == 120
torch.npu.synchronize()
root = Path("/work/results")
docs = {}
for name in APIS:
    fn = getattr(torch_npu, name, None)
    docs[name] = {"present": fn is not None, "doc": getattr(fn, "__doc__", None)}
(root / "torch_npu_api_docs.json").write_text(json.dumps(docs, ensure_ascii=False, indent=2))
config = Path("/model/config.json")
result = {
    "physical_chip": int(os.environ["ASCEND_RT_VISIBLE_DEVICES"]),
    "logical_device": 0,
    "device_count": torch.npu.device_count(),
    "device_name": torch.npu.get_device_name(0),
    "torch": torch.__version__, "torch_npu": torch_npu.__version__,
    "vllm": vllm.__version__, "vllm_ascend": vllm_ascend.__file__,
    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "packages": {name: importlib.util.find_spec(name) is not None
                 for name in ["triton", "tilelang", "pypto", "torchair", "cann_ops_transformer"]},
    "cache": {name: os.getenv(name) for name in
              ["VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "ASCEND_CACHE_PATH", "XDG_CACHE_HOME"]},
    "api_presence": {name: docs[name]["present"] for name in APIS},
    "device_smoke": "pass",
}
(root / "environment.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False))
