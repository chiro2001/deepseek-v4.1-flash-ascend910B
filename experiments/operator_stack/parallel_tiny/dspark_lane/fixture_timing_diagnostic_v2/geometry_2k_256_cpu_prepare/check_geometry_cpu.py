"""CPU negative controls for expanded rounds, paged SWA and capture-only references."""
import ast
import copy
import hashlib
import json
import os
import re
from pathlib import Path
import sys
from types import SimpleNamespace

from geometry_contract import validate_configuration, validate_query_positions, validate_reader_indices, validate_rounds

BASE = Path(__file__).resolve().parent


def reject(function):
    try:
        function()
    except AssertionError:
        return
    raise AssertionError("Negative control was not rejected")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    controls = []
    validate_configuration(4096, 2048, 3072, 128, 5)
    launcher_tree = ast.parse((BASE / "launch_geometry_gate.py").read_text())
    budget_call = next(node for node in ast.walk(launcher_tree) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute) and node.func.attr == "replace"
                       and len(node.args) == 2 and "original_fresh_gate=fresh_gate" in ast.unparse(node.args[1]))
    budget_code = eval(compile(ast.Expression(budget_call.args[1]), "rendered_budget", "eval"),
                       {"needle": "initial_gate=fresh_gate()"})
    for used in (3289, 7465):
        namespace = {"re": re, "fresh_gate": lambda used=used: {"physical4": {"line": f"0 / 0 {used} / 65536"}}}
        exec(budget_code, namespace)
        assert namespace["initial_gate"]["normal_memory_budget"]["gpu_memory_utilization"] == .70
        controls.append("current_HBM_" + str(used) + "_keeps_normal_budget")
    for line in ("0 / 0 24000 / 65536", "0 / 0"):
        reject(lambda line=line: exec(budget_code, {"re": re, "fresh_gate": lambda: {"physical4": {"line": line}}}))
        controls.append("unavailable_or_missing_current_normal_memory_budget_rejected")
    for name, args in (("old_model_capacity", (2048, 2048, 3072, 128, 5)),
                       ("old_prefill_capacity", (4096, 256, 3072, 128, 5)),
                       ("capture_plus_K_overflow", (4096, 2048, 4096, 128, 5)),
                       ("stale_small_capture_representative", (4096, 2048, 256, 128, 5))):
        reject(lambda args=args: validate_configuration(*args))
        controls.append(name + "_rejected")
    tokens = list(range(256))
    for count in (43, 128, 255):
        rows = [{"req_id": "r", "is_decode": False, "computed_before": 0, "prompt_tokens": 2048,
                 "scheduled_target_tokens": 2048, "verified_draft_tokens": 0, "draft_calls": 1,
                 "sampled_token_ids": [tokens[0]]}]
        remaining, offset = 255, 1
        for index in range(count):
            take = min(6, remaining - (count - index - 1))
            rows.append({"req_id": "r", "is_decode": True, "computed_before": 2048 + offset - 1,
                         "prompt_tokens": 2048, "scheduled_target_tokens": 6, "verified_draft_tokens": 5,
                         "draft_calls": 1, "sampled_token_ids": tokens[offset:offset + take]})
            offset += take
            remaining -= take
        result = validate_rounds(rows, tokens, "r")
        assert result["target_decode_rounds"] == count and result["actual_draft_calls"] == count + 1
        controls.append("fresh_round_count_" + str(count) + "_derived_without_16_step_assumption")
        for phase in (0, -1):
            wrong = copy.deepcopy(rows)
            wrong[phase]["draft_calls"] = 0
            reject(lambda wrong=wrong: validate_rounds(wrong, tokens, "r"))
        controls.append("initial_terminal_missing_for_" + str(count) + "_rejected")
    table = [0] * 32
    table[15], table[16], table[17], table[18] = 11, 3, 7, 14
    import torch
    source_override = os.environ.get("GEOMETRY_CPU_NATIVE_BUILDER")
    source = (Path(source_override) if source_override else BASE.parents[3] / "runtime/evidence/dspark_lane/graph_candidate_source/vllm_ascend/attention/dsa_v1.py")
    assert sha(source) == "24cb4e1ad3b2fcc9f0524153f22c0c6818aba396d290c4b93cdf4f12ced217d5"
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("_aligned_dspark_index_width", "build_dspark_swa_indices")]
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions], type_ignores=[])), str(source), "exec"), namespace)
    native_builder = namespace["build_dspark_swa_indices"]
    for seq_len, start, computed in ((2053, 2048, 2048), (2309, 2304, 2302)):
        indices, lens = native_builder(torch.tensor([table], dtype=torch.int32), 5, 128, 128,
                                       torch.tensor([0, 5], dtype=torch.int32),
                                       torch.tensor([seq_len], dtype=torch.int32), num_decode_tokens=5)
        assert indices.device.type == "cpu" and lens.tolist() == [133] * 5
        actual = indices[:, 0].tolist()
        receipt = validate_reader_indices(actual, table, seq_len, 128, 20, null_block_id=0)
        validate_query_positions(list(range(start, start + 5)), {"is_decode": True, "computed_before": computed,
                                                                 "scheduled_target_tokens": 6}, seq_len)
        assert receipt["selected_slots_count_with_duplicates"] == 665
        assert receipt["native_padding_entries"] == 615 and not receipt["unretained_prefix_scanned"]
        controls.append("native_CPU_builder_matches_visible_cache_at_" + str(seq_len))
        reject(lambda: validate_reader_indices(actual, table[:2], seq_len, 128, 20, null_block_id=0))
        controls.append("old_two_page_table_rejected_at_" + str(seq_len))
        wrong = copy.deepcopy(actual)
        wrong[0][0] = -1
        reject(lambda: validate_reader_indices(wrong, table, seq_len, 128, 20, null_block_id=0))
        controls.append("required_native_slot_cannot_be_filtered_at_" + str(seq_len))
    bad = table.copy()
    bad[15] = 0
    reject(lambda: validate_reader_indices(actual, bad, 2053, 128, 20, null_block_id=0))
    controls.append("native_null_page_in_active_window_rejected")
    reject(lambda: validate_query_positions([3072, 3073, 3074, 3075, 3076],
                                             {"is_decode": True, "computed_before": 2302, "scheduled_target_tokens": 6}, 3077))
    controls.append("stale_capture_positions_rejected")
    reject(lambda: validate_query_positions([2304, 2305, 2306, 2307, 2308],
                                             {"is_decode": True, "computed_before": 2302, "scheduled_target_tokens": 6}, 3077))
    controls.append("stale_native_seq_len_rejected")
    probe_tree = ast.parse((BASE / "geometry_cache_probe.py").read_text())
    sparse = [node for node in ast.walk(probe_tree) if isinstance(node, ast.FunctionDef) and node.name == "sparse"]
    assert len(sparse) == 1
    opaque, answer, forwarded = object(), object(), []
    state = {"active_role": "draft.0", "capture": {}, "eager": {}}
    kwargs = {"layout_q": "TND", "layout_kv": "PA_ND", "ori_mask_mode": 4,
              **{key: object() for key in ("ori_kv", "ori_sparse_indices", "ori_block_table", "seqused_kv", "cu_seqlens_q")}}
    namespace = {"STATE": state, "audit": SimpleNamespace(_capturing=lambda: True),
                 "native": lambda q, **kw: forwarded.append((q, kw)) or answer}
    exec(compile(ast.fix_missing_locations(ast.Module(body=sparse, type_ignores=[])), "capture_reference_control", "exec"), namespace)
    assert namespace["sparse"](opaque, **kwargs) is answer
    assert forwarded == [(opaque, kwargs)] and state["capture"]["draft.0"]["indices"] is kwargs["ori_sparse_indices"]
    controls.append("capture_reference_preserves_original_native_arguments_return_and_opaque_tensors")
    assert "torch_npu" not in sys.modules and "acl" not in sys.modules
    files = [path for path in BASE.iterdir() if path.suffix == ".py"]
    for path in files:
        ast.parse(path.read_text(), filename=str(path))
    output = {"passed": True, "controls_passed": len(controls), "controls": controls,
              "torch_CPU_against_frozen_native_builder": True, "cpu_only_no_npu_sdk_import": True,
              "npu_proof": False, "new_npu_window_authorized": False,
              "source_sha256": {path.name: sha(path) for path in files},
              "frozen_native_builder_sha256": sha(source)}
    (BASE / "cpu_gate.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key not in ("controls", "source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
