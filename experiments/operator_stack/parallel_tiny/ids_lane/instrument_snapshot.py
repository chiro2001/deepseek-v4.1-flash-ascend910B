"""Add capture-only reference hooks to a private copy, keeping arithmetic intact."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    assert "ids_lane_private" in str(root) or "tiny-ids-chip4" in str(root)
    files = {"model": root / "source/vllm_ascend/models/deepseek_v41/model.py",
             "router": root / "source/vllm_ascend/ops/fused_moe/router/fused_topk_router.py"}
    receipt = {}
    for role, path in files.items():
        original = path.read_text()
        assert "ids_patches" not in original, "Snapshot already instrumented"
        content = original
        if role == "model":
            needle = "            moe_input_ids = moe_input_ids.to(torch.int64)\n        for layer in self.layers:"
            assert content.count(needle) == 1
            content = content.replace(needle, "            moe_input_ids = moe_input_ids.to(torch.int64)\n        import ids_patches\n        ids_patches.record_model_ids(input_ids, moe_input_ids)\n        for layer in self.layers:")
        else:
            needle = "                input_ids = input_ids.to(torch.int64)\n"
            assert content.count(needle) == 1
            content = content.replace(needle, "                import ids_patches\n                ids_patches.record_router_before(self, input_ids)\n" + needle + "                ids_patches.record_router_after_cast(input_ids)\n")
            needle = "            bias_vl = self.bias_vl\n"
            assert content.count(needle) == 1
            content = content.replace(needle, "            if input_ids is not None:\n                ids_patches.record_router_consumer(input_ids)\n" + needle)
            needle = "            return topk_weights.to(torch.float32), topk_ids.to(torch.int32 if indices_type is None else indices_type)\n"
            assert content.count(needle) == 1
            content = content.replace(needle, "            if input_ids is not None:\n                ids_patches.record_router_result(topk_weights, topk_ids)\n" + needle)
        compile(content, str(path), "exec")
        path.write_text(content)
        receipt[role] = {"path": str(path), "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
                         "instrumented_sha256": hashlib.sha256(content.encode()).hexdigest(),
                         "change": "capture-only tensor-reference hooks; no arithmetic or routing changes"}
    (root / "instrumentation_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
