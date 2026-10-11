"""Replay real Engram lookup, BF16 projection and the native gate on one chip.

This is a resident Engram consumer replay, not a tiny language model. Hashes,
table rows, dequantization and projection/gate weights are real. The residual
stream is synthetic until graph-external formal activations are supplied.
Eight output shards are concatenated locally; no TP8 communication is timed.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import struct
import sys
import time

import torch
import torch_npu
from safetensors import safe_open


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def compare(actual, expected):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    record = {"shape": list(actual.shape), "dtype": str(actual.dtype),
              "bitwise_equal": bool(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))}
    if actual.dtype == torch.bfloat16:
        def ordered(tensor):
            bits = tensor.view(torch.int16).to(torch.int32) & 65535
            return torch.where((bits & 32768) != 0, 32768 - (bits & 32767), 32768 + bits)
        ulp = (ordered(actual) - ordered(expected)).abs()
        record.update(finite=bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()),
                      max_bf16_ulp=int(ulp.max()), elements_above_one_ulp=int((ulp > 1).sum()),
                      unequal_elements=int((actual.view(torch.int16) != expected.view(torch.int16)).sum()))
        record["passed"] = record["finite"] and record["elements_above_one_ulp"] == 0
    else:
        record["passed"] = record["bitwise_equal"]
    return record


def tensor_header(path, key):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        value = json.loads(stream.read(size))[key]
    return {"path": str(path), "resolved_path": str(path.resolve()), **value}


def load_native_snapshots(directory, fixture):
    importlib.import_module("vllm_ascend.models.deepseek_v41")
    modules, receipts = {}, []
    for basename in ("engram_jit_kernel", "engram_plan_kernel", "engram_hash", "engram_hbm", "engram_device_index", "engram_gate"):
        fullname = "vllm_ascend.models.deepseek_v41." + basename
        path = directory / (basename + ".py")
        data = path.read_bytes()
        reference = next(v for v in fixture["native_sources"] if v["module"] == fullname)
        assert hashlib.sha256(data).hexdigest() == reference["sha256"], fullname
        spec = importlib.util.spec_from_file_location(fullname, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        spec.loader.exec_module(module)
        modules[basename] = module
        receipts.append({"module": fullname, "path": str(path), "sha256": reference["sha256"]})
    return modules, receipts


def validate_cpu_fixture(args):
    assert not torch.npu.is_initialized()
    fixture = torch.load(args.lookup_fixture, map_location="cpu", weights_only=False)
    modules, sources = load_native_snapshots(args.native_sources, fixture)
    rows = []
    for phase in fixture["phases"]:
        if phase["phase"] != "decode":
            continue
        for slot, layer in enumerate((1, 14)):
            selected = phase["rows"][layer]
            assert torch.equal(phase["hashes"][:, slot].reshape(-1), selected["rows"])
            native = modules["engram_hbm"].dequantize_engram_rows(selected["codes"], selected["scales"]).reshape(1, -1)
            repeated = modules["engram_hbm"].dequantize_engram_rows(selected["codes"], selected["scales"]).reshape(1, -1)
            assert torch.equal(native.view(torch.int16), selected["lookup"].view(torch.int16))
            assert torch.equal(native.view(torch.int16), repeated.view(torch.int16))
            assert digest(native) == selected["lookup_sha256"]
            rows.append({"request": phase["request"], "step": phase["step"], "layer": layer,
                         "lookup_sha256": digest(native), "original_cpu_lookup_bitwise_equal": True,
                         "cpu_dequant_aa_bitwise_equal": True})
    assert not torch.npu.is_initialized()
    receipt = {"scope": "CPU-only compatibility and original dequant lookup validation; no NPU arithmetic or performance claim",
               "cpu_only": True, "npu_initialized": False, "rows_verified": len(rows), "native_sources": sources,
               "fixture_sha256": hashlib.sha256(args.lookup_fixture.read_bytes()).hexdigest(), "rows": rows}
    args.output.mkdir(exist_ok=False, parents=True)
    (args.output / "cpu_validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("ENGRAM_CPU_VALIDATION_COMPLETE", json.dumps({k: receipt[k] for k in ("cpu_only", "npu_initialized", "rows_verified")}), flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lookup-fixture", type=Path, required=True)
    parser.add_argument("--native-sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-chip", type=int, required=True)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=128)
    parser.add_argument("--validate-cpu-fixture", action="store_true")
    args = parser.parse_args()
    assert args.physical_chip in (4, 7)
    assert args.physical_chip == int(os.environ["STACK_PHYSICAL_CHIPS"])
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == str(args.physical_chip)
    assert os.environ["HCCL_DETERMINISTIC"] == "strict"
    assert os.environ["V41_ENGRAM_GATE_CHUNK"] == "0"
    assert os.environ["V41_ENGRAM_GATE_HOIST"] == "0"
    if args.validate_cpu_fixture:
        validate_cpu_fixture(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    torch.npu.set_op_timeout_ms(60000)
    torch_npu.npu.config.allow_internal_format = True
    torch.manual_seed(20261011)
    fixture = torch.load(args.lookup_fixture, map_location="cpu", weights_only=False)
    # Load production snapshots only in this process, never edit a shared
    # installed package or another agent's source directory.
    load_native_snapshots(args.native_sources, fixture)
    hash_module = importlib.import_module("vllm_ascend.models.deepseek_v41.engram_hash")
    hbm_module = importlib.import_module("vllm_ascend.models.deepseek_v41.engram_hbm")
    device_module = importlib.import_module("vllm_ascend.models.deepseek_v41.engram_device_index")
    gate_module = importlib.import_module("vllm_ascend.models.deepseek_v41.engram_gate")
    source_modules = [hash_module, hbm_module, device_module, gate_module]
    source_snapshots = args.output / "source_snapshots"
    source_snapshots.mkdir()
    source_receipts = []
    for module in source_modules:
        path = Path(module.__file__)
        data = path.read_bytes()
        reference = next(v for v in fixture["native_sources"] if v["module"] == module.__name__)
        assert hashlib.sha256(data).hexdigest() == reference["sha256"], module.__name__
        (source_snapshots / path.name).write_bytes(data)
        source_receipts.append({"module": module.__name__, "path": str(path),
                                "sha256": hashlib.sha256(data).hexdigest()})
    config_data = (args.model / "config.json").read_bytes()
    config = json.loads(config_data)
    text = dict(config["text_config"])
    assert (text["num_hidden_layers"], text["hidden_size"], text["n_routed_experts"],
            text["num_experts_per_tok"], text["hc_mult"]) == (40, 5120, 384, 6, 4)
    assert text["engram_layer_ids"] == [1, 14]
    text["engram_pad_id"] = text.get("engram_pad_id", text["engram_pad_token_id"])
    text["image_token_id"] = config["image_token_id"]
    assert config == fixture["config"], "Copied checkpoint config must match the CPU fixture source"
    index = json.loads((args.model / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    device_hash = device_module.DeviceNgramHash(
        token_map=fixture["token_map"], primes=fixture["primes"], offsets=fixture["offsets"],
        multipliers=fixture["multipliers"], pad_id=fixture["pad_id"],
        image_token_id=fixture["image_token_id"], image_pad_token_id=fixture["image_pad_token_id"],
        device="npu:0", max_pages=64, block_size=128)
    result = {"completed": False, "precision_passed": False, "native_aa_passed": False,
              "scope": "Resident real Engram hash/table/dequant/BF16 wkv/split/native gate; "
                       "synthetic residual; not a tiny model or a full transformer layer; "
                       "local output concatenation excludes TP8 communication and E2E",
              "physical_chip": args.physical_chip, "logical_chip": 0,
              "visible_filter": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
              "hardware": torch.npu.get_device_name(0), "model": str(args.model),
              "source_model": fixture["source_model"], "lookup_fixture": str(args.lookup_fixture),
              "lookup_fixture_sha256": hashlib.sha256(args.lookup_fixture.read_bytes()).hexdigest(),
              "model_layers": 40, "hidden_size": 5120, "global_experts": 384,
              "model_topk": 6, "model_tp": 8, "replay_tp": 1,
              "replay_engram_layers": [1, 14], "checkpoint_quantization": config.get("quantization_config"),
              "engram_projection_dtype": "BF16", "engram_table_storage": "INT8 group32, FP32 scale",
              "input_shape": [1, 6144], "weight_shape": [25600, 6144],
              "output_shards": 8, "shard_shape": [3200, 6144], "reduction_K_unchanged": True,
              "hc_mult": 4, "consumer_residual": "Seeded synthetic BF16 [1,4,5120], scales .001/1/100",
              "precision_gate": "Native A/A bitwise; candidate projection/key/value/gate original <=1 BF16 ULP",
              "hccl_deterministic": os.environ["HCCL_DETERMINISTIC"], "profiler_during_timing": "OFF",
              "lookup_timing": "Original CPU hash and lazy-mapped real table row lookup were CPU-only pre-read into fixture; excluded from timing",
              "checkpoint_config_sha256": hashlib.sha256(config_data).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "native_sources": source_receipts, "weights": [], "hash_cases": [],
              "lookup_cases": [], "cases": [], "pairs": [], "performance": {}}

    def save():
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    def load(key):
        with safe_open(str(args.model / index[key]), framework="pt", device="cpu") as file:
            return file.get_tensor(key)

    save()
    # The CPU exporter ran the original PagedNgramHistory.update, selected
    # exact rows from full lazy-mapped real tables and called native dequant.
    # Here the original device hash/dequant must reproduce that fixture.
    device_block_table = fixture["block_table"].to("npu")
    inputs = []
    for phase in fixture["phases"]:
        ids, positions, requests = phase["ids"], phase["positions"], phase["requests"]
        native_hashes, native_mask = phase["hashes"], phase["mask"]
        hashes, mask = device_hash(ids.to("npu"), positions.to("npu"), requests.to("npu"), device_block_table)
        record = {"request": phase["request"], "phase": phase["phase"], "tokens": ids.numel(),
                  "hashes": compare(hashes, native_hashes), "mask": compare(mask, native_mask),
                  "hash_sha256": digest(native_hashes)}
        assert record["hashes"]["passed"] and record["mask"]["passed"], record
        result["hash_cases"].append(record)
        if phase["phase"] == "prefill":
            save()
            continue
        lookups = {}
        for slot, layer in enumerate((1, 14)):
            selected = phase["rows"][layer]
            rows = native_hashes[:, slot].reshape(-1)
            assert torch.equal(rows, selected["rows"])
            selected_codes, selected_scales = selected["codes"], selected["scales"]
            lookup = hbm_module.dequantize_engram_rows(selected_codes, selected_scales).reshape(1, -1)
            assert digest(lookup) == selected["lookup_sha256"]
            assert torch.equal(lookup.view(torch.int16), selected["lookup"].view(torch.int16))
            lookup_npu = hbm_module.dequantize_engram_rows(selected_codes.to("npu"), selected_scales.to("npu")).reshape(1, -1)
            checked = compare(lookup_npu, lookup)
            assert checked["bitwise_equal"], checked
            lookups[layer] = lookup_npu
            result["lookup_cases"].append({"request": phase["request"], "step": phase["step"], "layer": layer,
                "rows": rows.tolist(), "lookup_sha256": digest(lookup), "cpu_npu_dequant": checked,
                "codes_sha256": digest(selected_codes), "scales_sha256": digest(selected_scales)})
        inputs.append({"request": phase["request"], "step": phase["step"], "lookups": lookups, "mask": mask})
        save()
    block = fixture["rotation_block"]
    rotation_npu = block.to(dtype=torch.float32, device="npu")
    result["rotation"] = {"path": fixture["rotation_source"], "block_sha256": digest(block)}
    all_aa, all_candidate = True, True
    retained = {}
    for layer in (1, 14):
        prefix = f"layers.{layer}.engram."
        cpu_weight = load(prefix + "wkv.weight").contiguous()
        assert cpu_weight.dtype == torch.bfloat16 and cpu_weight.shape == (25600, 6144)
        weight = cpu_weight.to("npu")
        shards = tuple(cpu_weight.chunk(8, dim=0)[i].contiguous().to("npu") for i in range(8))
        q_weight, k_weight = load(prefix + "q_weight"), load(prefix + "k_weight")
        channel = q_weight.to("npu").float() * k_weight.to("npu").float()
        assert channel.shape == (4, 5120)
        result["weights"].append({"layer": layer, "key": prefix + "wkv.weight",
            "header": tensor_header(args.model / index[prefix + "wkv.weight"], prefix + "wkv.weight"),
            "bytes": cpu_weight.numel() * cpu_weight.element_size(), "sha256": digest(cpu_weight),
            "shard_sha256": [digest(s) for s in cpu_weight.chunk(8, dim=0)],
            "native_format": int(torch_npu.get_npu_format(weight)), "native_stride": list(weight.stride()),
            "shard_formats": [int(torch_npu.get_npu_format(s)) for s in shards],
            "q_weight_sha256": digest(q_weight), "k_weight_sha256": digest(k_weight),
            "table_headers": fixture["table_headers"][layer]})

        def projection(lookup, split=False):
            if split:
                return torch.cat([torch.nn.functional.linear(lookup, shard) for shard in shards], dim=-1)
            return torch.nn.functional.linear(lookup, weight)

        def consume(lookup, residual, mask, split=False):
            kv = projection(lookup, split)
            key, value = kv.split([20480, 5120], dim=-1)
            out = gate_module.engram_gate(residual, key.view(1, 4, 5120), value,
                                          channel, rotation_npu, mask, text["rms_norm_eps"])
            return kv, key, value, out

        for number, case in enumerate(inputs):
            torch.manual_seed(20261011 + layer * 100 + number)
            magnitude = (.001, 1.0, 100.0)[number % 3]
            residual = (torch.randn((1, 4, 5120), device="npu") * magnitude).to(torch.bfloat16)
            lookup, mask = case["lookups"][layer], case["mask"]
            native = consume(lookup, residual, mask)
            repeated = consume(lookup, residual, mask)
            candidate = consume(lookup, residual, mask, True)
            names = ("projection", "key", "value", "gate_output")
            aa = {name: compare(a, b) for name, a, b in zip(names, repeated, native)}
            ab = {name: compare(a, b) for name, a, b in zip(names, candidate, native)}
            record = {"layer": layer, "request": case["request"], "step": case["step"],
                      "residual_magnitude": magnitude, "native_aa": aa, "candidate": ab}
            record["native_aa_passed"] = all(v["bitwise_equal"] for v in aa.values())
            record["candidate_passed"] = all(v["passed"] for v in ab.values())
            all_aa &= record["native_aa_passed"]
            all_candidate &= record["candidate_passed"]
            result["cases"].append(record)
            save()
            print("ENGRAM_REPLAY_PRECISION", json.dumps({k: v for k, v in record.items() if k not in ("native_aa", "candidate")}), flush=True)
        # Bind defaults so graphs for both layers retain their own real weights.
        retained[layer] = (weight, shards, channel)
    result["native_aa_passed"] = all_aa
    result["precision_passed"] = all_aa and all_candidate
    save()
    if result["precision_passed"]:
        for layer in (1, 14):
            weight, shards, channel = retained[layer]
            lookup, mask = inputs[0]["lookups"][layer], inputs[0]["mask"]
            residual = torch.randn((1, 4, 5120), device="npu", dtype=torch.bfloat16)

            def projection_full():
                return torch.nn.functional.linear(lookup, weight)

            def projection_one():
                return torch.nn.functional.linear(lookup, shards[0])

            def projection_eight():
                return torch.cat([torch.nn.functional.linear(lookup, shard) for shard in shards], dim=-1)

            def consumer_full():
                kv = projection_full()
                key, value = kv.split([20480, 5120], dim=-1)
                return gate_module.engram_gate(residual, key.view(1, 4, 5120), value,
                                              channel, rotation_npu, mask, text["rms_norm_eps"])

            def consumer_eight():
                kv = projection_eight()
                key, value = kv.split([20480, 5120], dim=-1)
                return gate_module.engram_gate(residual, key.view(1, 4, 5120), value,
                                              channel, rotation_npu, mask, text["rms_norm_eps"])

            functions = {"full_projection": projection_full, "one_shard_projection": projection_one,
                         "eight_shard_concat": projection_eight, "full_consumer": consumer_full,
                         "eight_shard_consumer": consumer_eight}
            graphs = {}
            outputs = {}
            for name, function in functions.items():
                for _ in range(4):
                    function()
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    outputs[name] = function()
                for _ in range(16):
                    graph.replay()
                torch.npu.synchronize()
                graphs[name] = graph

            def timed(name):
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                begin.record()
                for _ in range(args.repeats):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                return begin.elapsed_time(end) * 1000 / args.repeats

            for pair in range(args.pairs):
                names = list(functions)
                if pair % 2:
                    names.reverse()
                measurements = {name: timed(name) for name in names}
                result["pairs"].append({"layer": layer, "pair": pair, "order": names,
                                         "microseconds_per_replay": measurements})
                save()
            rows = [v["microseconds_per_replay"] for v in result["pairs"] if v["layer"] == layer]
            result["performance"][str(layer)] = {name: {"median_us": statistics.median(v[name] for v in rows),
                "min_us": min(v[name] for v in rows), "max_us": max(v[name] for v in rows)} for name in functions}
            result["performance"][str(layer)]["one_shard_compute_only_savings_us"] = statistics.median(
                v["full_projection"] - v["one_shard_projection"] for v in rows)
            result["performance"][str(layer)]["single_chip_eight_shard_concat_savings_us"] = statistics.median(
                v["full_projection"] - v["eight_shard_concat"] for v in rows)
            save()
            print("ENGRAM_REPLAY_PERF", json.dumps({"layer": layer, "performance": result["performance"][str(layer)]}), flush=True)
    result["completed"] = True
    result["seconds"] = time.monotonic() - started
    result["formal_tp8_eligible"] = False
    result["formal_tp8_requirement"] = "8-rank identical lookup proof, native AllGather reconstruction, original precision gate and E2E audit required"
    save()
    print("ENGRAM_REPLAY_COMPLETE", json.dumps({k: result[k] for k in ("completed", "precision_passed", "native_aa_passed", "seconds", "performance")}), flush=True)
    if not result["precision_passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
