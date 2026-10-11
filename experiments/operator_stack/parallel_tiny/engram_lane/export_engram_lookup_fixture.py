"""Remote-local CPU-only export of exact original hashes and queried table rows."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import torch
import torch_npu
from safetensors import safe_open
from transformers import AutoTokenizer


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def header(path, key):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        value = json.loads(stream.read(size))[key]
    return {"path": str(path), "resolved_path": str(path.resolve()), **value}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(1)
    assert not torch.npu.is_initialized(), "CPU fixture export must not initialize an NPU"
    config = json.loads((args.model / "config.json").read_text())
    text = dict(config["text_config"])
    text["engram_pad_id"] = text.get("engram_pad_id", text["engram_pad_token_id"])
    text["image_token_id"] = config["image_token_id"]
    index = json.loads((args.model / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    hash_module = importlib.import_module("vllm_ascend.models.deepseek_v41.engram_hash")
    hbm_module = importlib.import_module("vllm_ascend.models.deepseek_v41.engram_hbm")
    history = hash_module.PagedNgramHistory(SimpleNamespace(**text), AutoTokenizer.from_pretrained(args.tokenizer))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tables = {}
    table_headers = {}
    for layer in (1, 14):
        prefix = f"layers.{layer}.engram.embed."
        tensors = []
        for part in ("weight", "scale"):
            key = prefix + part
            with safe_open(str(args.model / index[key]), framework="pt", device="cpu") as file:
                tensors.append(file.get_tensor(key))
        tables[layer] = tuple(tensors)
        table_headers[layer] = [header(args.model / index[prefix + part], prefix + part) for part in ("weight", "scale")]
    prompts = ["请计算 17 加 25 等于多少，并说明过程。", "The quick brown fox jumps over the lazy dog.",
               "讨论内存、检索与模型推理之间的关系。", "Here is an image and a short description."]
    continuation = tokenizer.encode(" The result is forty two. 结果为四十二。", add_special_tokens=False)
    block_table = torch.arange(1, 33, dtype=torch.int32).reshape(4, 8)
    phases = []
    row_records = []
    for request, prompt in enumerate(prompts):
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if request == 3:
            tokens.insert(len(tokens) // 2, history.image_token_id)
            tokens.insert(len(tokens) // 2 + 1, history.image_pad_token_id)
        ids = torch.tensor(tokens, dtype=torch.int64)
        positions = torch.arange(len(tokens), dtype=torch.int64)
        requests = torch.full_like(ids, request)
        hashes, mask = history.update(ids, positions, requests, block_table, 128)
        phases.append({"request": request, "phase": "prefill", "ids": ids,
                       "positions": positions, "requests": requests, "hashes": hashes.clone(), "mask": mask.clone()})
        decode_tokens = continuation[:6]
        if request == 3:
            decode_tokens = [history.image_token_id, history.image_pad_token_id] + continuation[2:6]
        for step, token in enumerate(decode_tokens):
            ids = torch.tensor([token], dtype=torch.int64)
            positions = torch.tensor([len(tokens) + step], dtype=torch.int64)
            requests = torch.tensor([request], dtype=torch.int64)
            hashes, mask = history.update(ids, positions, requests, block_table, 128)
            rows = {}
            for slot, layer in enumerate((1, 14)):
                row_ids = hashes[:, slot].reshape(-1)
                codes, scales = tables[layer]
                selected_codes = torch.index_select(codes, 0, row_ids)
                selected_scales = torch.index_select(scales, 0, row_ids)
                lookup = hbm_module.dequantize_engram_rows(selected_codes, selected_scales).reshape(1, -1)
                rows[layer] = {"rows": row_ids.clone(), "codes": selected_codes, "scales": selected_scales,
                               "lookup": lookup, "lookup_sha256": digest(lookup)}
                row_records.append({"request": request, "step": step, "layer": layer,
                    "rows": row_ids.tolist(), "codes_sha256": digest(selected_codes),
                    "scales_sha256": digest(selected_scales), "lookup_sha256": digest(lookup)})
            phases.append({"request": request, "step": step, "phase": "decode", "ids": ids,
                           "positions": positions, "requests": requests, "hashes": hashes.clone(),
                           "mask": mask.clone(), "rows": rows})
    source_snapshots = args.output / "native_sources"
    source_snapshots.mkdir()
    source_receipts = []
    for module_name in ("engram_jit_kernel", "engram_plan_kernel", "engram_hash", "engram_hbm", "engram_device_index", "engram_gate"):
        module = importlib.import_module("vllm_ascend.models.deepseek_v41." + module_name)
        path = Path(module.__file__)
        data = path.read_bytes()
        (source_snapshots / path.name).write_bytes(data)
        source_receipts.append({"module": module.__name__, "path": str(path), "sha256": hashlib.sha256(data).hexdigest()})
    rotation_path = args.model / "optional/quarot.safetensors"
    with safe_open(str(rotation_path), framework="pt", device="cpu") as file:
        rotation = file.get_tensor("global_rotation")
    rotation_block = rotation[:32, :32].contiguous()
    assert torch.equal(rotation, torch.block_diag(*[rotation_block] * (5120 // 32)))
    fixture = {"source_model": str(args.model), "config": config,
               "block_table": block_table, "token_map": history.token_map,
               "primes": history.primes, "offsets": history.offsets, "multipliers": history.multipliers,
               "pad_id": history.pad_id, "image_token_id": history.image_token_id,
               "image_pad_token_id": history.image_pad_token_id, "phases": phases,
               "table_headers": table_headers, "native_sources": source_receipts,
               "rotation_block": rotation_block, "rotation_source": str(rotation_path.resolve())}
    destination = args.output / "lookup_fixture.pt"
    torch.save(fixture, destination)
    receipt = {"cpu_only": True, "npu_initialized": torch.npu.is_initialized(),
               "source_model": str(args.model), "fixture_bytes": destination.stat().st_size,
               "fixture_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
               "original_hash_source": str(hash_module.__file__), "original_dequant_source": str(hbm_module.__file__),
               "phases": len(phases), "decode_cases": 24, "engram_layers": [1, 14],
               "table_headers": table_headers, "native_sources": source_receipts, "rows": row_records}
    assert not receipt["npu_initialized"]
    (args.output / "fixture_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (args.output / "config.json").write_bytes((args.model / "config.json").read_bytes())
    (args.output / "quant_model_weights.safetensors.index.json").write_bytes((args.model / "quant_model_weights.safetensors.index.json").read_bytes())
    print("ENGRAM_CPU_FIXTURE_COMPLETE", json.dumps({k: receipt[k] for k in ("cpu_only", "npu_initialized", "fixture_bytes", "fixture_sha256", "decode_cases")}), flush=True)


if __name__ == "__main__":
    main()
