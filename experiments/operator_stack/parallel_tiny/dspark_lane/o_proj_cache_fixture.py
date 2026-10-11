"""Pre-compile native dummy O-cache initialization with explicit module ownership."""
import hashlib
import json
import os
from pathlib import Path

import graph_precision_audit as audit

BUFFER_NAME = "_tiny_fixed_native_wo_a_3d"
STATE = {"initialized": False, "entries": {}, "consumer_counts": {}, "source_sha256": None}


def initialize_before_compile(worker):
    import torch
    from vllm_ascend.attention import dsa_v1
    assert os.environ.get("DSPARK_O_PROJ_CACHE_FIXTURE") == "1"
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == os.environ["DSPARK_O_PROJ_CACHE_FIXTURE_SHA256"]
    assert not STATE["initialized"] and not audit._capturing()
    assert hashlib.sha256(Path(dsa_v1.__file__).read_bytes()).hexdigest() == "24cb4e1ad3b2fcc9f0524153f22c0c6818aba396d290c4b93cdf4f12ced217d5"
    runner = worker.model_runner
    before = audit._parameter_metadata(worker)
    groups = [(f"target.{index}", layer) for index, layer in enumerate(runner._tiny_dspark_native_backbone.layers)]
    groups += [(f"draft.{index}", layer) for index, layer in enumerate(runner._tiny_dspark_native_draft.layers.values())]
    assert len(groups) == 43
    entries = {}
    with torch.no_grad():
        for role, layer in groups:
            owner = layer.self_attn
            impl = owner.dsa_attn.dsa_attn.impl
            weight = impl.wo_a.weight
            assert weight.ndim == 2 and weight.dtype == torch.bfloat16 and weight.device.type == "npu"
            assert impl.n_local_groups == 8 and impl.o_lora_rank == 512
            assert getattr(impl, "_dummy_wo_a_3d", None) is None, "Never replace an existing cache after forward/capture"
            assert not hasattr(owner, BUFFER_NAME)
            cached = weight.view(impl.n_local_groups, impl.o_lora_rank, -1).transpose(2, 1).contiguous()
            owner.register_buffer(BUFFER_NAME, cached, persistent=False)
            impl._dummy_wo_a_3d = getattr(owner, BUFFER_NAME)
            torch.npu.synchronize()
            source_cpu, source_info = audit._diagnostic_inspect(weight)
            cache_cpu, cache_info = audit._diagnostic_inspect(cached)
            audit._diagnostic_reject({"source": source_cpu, "cache": cache_cpu},
                                    {"source": source_info, "cache": cache_info}, "fixture." + role,
                                    {"point": "after load before any compile/profile/capture"})
            expected = source_cpu.view(8, 512, -1).transpose(2, 1).contiguous()
            assert audit._bitwise(cache_cpu, expected), "Fixture must preserve the exact native loader permutation"
            assert cached.data_ptr() != weight.data_ptr()
            entries[role] = {"owner": owner, "impl": impl, "owner_id": id(owner), "cache": cache_info,
                             "source": source_info, "actual_consumer_cache_ptr": cached.data_ptr()}
            STATE["consumer_counts"][role] = 0
    assert audit._parameter_metadata(worker) == before, "Original parameters changed during fixture initialization"
    STATE["entries"] = entries
    STATE["initialized"] = True
    STATE["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {role: {name: value for name, value in entry.items() if name not in ("owner", "impl")}
                for role, entry in entries.items()}
    audit._diagnostic_write("fixed_o_proj_cache_fixture", {"fixture_source_sha256": STATE["source_sha256"],
        "entries": manifest, "initialized_before_compile_profile_capture": True,
        "native_loader_permutation_bitwise_equal": True, "original_parameters_unchanged": True,
        "explicit_owner_buffer": BUFFER_NAME, "owner_buffer_persistent": False,
        "capture_or_warmup_weight_mutation": False, "formal_precision_or_performance_claim": False})
    original = dsa_v1.AscendDSAImpl._forward_o_proj
    roles_by_impl = {id(entry["impl"]): role for role, entry in entries.items()}

    def consumer(this, *args, **kwargs):
        role = roles_by_impl.get(id(this))
        if (role is not None and runner._tiny_dspark_recording and not audit._capturing()
                and os.environ.get("DSPARK_FIXTURE_VALIDATE_CONSUMERS") == "1"):
            check_entry(role, include_source=False, point="actual native O-projection consumer")
            STATE["consumer_counts"][role] += 1
        return original(this, *args, **kwargs)

    dsa_v1.AscendDSAImpl._forward_o_proj = consumer
    return {"owner_buffers": len(entries), "fixture_source_sha256": STATE["source_sha256"],
            "before_compile_profile_capture": True, "native_formula_unchanged": True,
            "original_parameter_views_and_values_unchanged": True, "actual_consumer_guard": True}


def check_entry(role, include_source, point):
    entry = STATE["entries"][role]
    owner, impl = entry["owner"], entry["impl"]
    cached = getattr(owner, BUFFER_NAME)
    assert id(owner) == entry["owner_id"] and impl._dummy_wo_a_3d is cached
    assert owner._buffers[BUFFER_NAME] is cached
    value, info = audit._diagnostic_inspect(cached)
    audit._diagnostic_reject({"cache": value}, {"cache": info}, "fixture." + role,
                            {"point": point, "registered_owner_buffer": BUFFER_NAME})
    assert info["view"] == entry["cache"]["view"], (role, "Fixed cache view changed")
    assert info["value_sha256"] == entry["cache"]["value_sha256"], (role, "Fixed cache value changed")
    if include_source:
        source, source_info = audit._diagnostic_inspect(impl.wo_a.weight)
        audit._diagnostic_reject({"source": source}, {"source": source_info}, "fixture." + role,
                                {"point": point, "original_parameter_unchanged": True})
        assert source_info["view"] == entry["source"]["view"]
        assert source_info["value_sha256"] == entry["source"]["value_sha256"]
    return {"role": role, "finite": True, "owner_and_consumer_binding_equal": True,
            "cache_view_and_value_seal_equal": True, "source_checked": include_source}


def validate_all(worker, point):
    assert STATE["initialized"] and not audit._capturing()
    checks = [check_entry(role, include_source=True, point=point) for role in STATE["entries"]]
    result = {"point": point, "checks": checks, "owner_buffers": len(checks),
              "fixture_source_sha256": STATE["source_sha256"], "passed": True,
              "actual_native_consumer_counts": dict(STATE["consumer_counts"]),
              "source_or_cache_values_mutated": False, "full_consumer_precision_proof": False}
    audit._diagnostic_write("fixture_validation." + point, result)
    return result
