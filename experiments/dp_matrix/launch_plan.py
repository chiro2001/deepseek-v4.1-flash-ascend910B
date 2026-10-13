"""New-candidate native DP plans, preserving the immutable best TP8 source.

Stdlib-only preparation: constructing a plan neither loads weights nor admits
the new TP widths, DP communication or DCP implementation to quality gates.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import re

if __package__.startswith("experiments."):
    from experiments.delivery_matrix.service import fingerprint, make_plan as tp8_plan
else:
    from delivery_matrix.service import fingerprint, make_plan as tp8_plan
from .topology import Topology, integer
from .host_memory import ENV as NUMA_ENV, node_list


WORKER = "dp_matrix.worker.DPWorker"
DCP2_DRAFT_MANIFEST_SHA256 = "abb1a60cd24f8040a9f87a3a69471a0e38e2a166aed6d073f43c6229316d5074"
DCP2_CONTRACT_SHA256 = hashlib.sha256(Path(__file__).with_name("dcp2_contract.py").read_bytes()).hexdigest()


def make_plan(candidate, *, layout, source, output, served_model,
              max_model_len=8192, batched_tokens=8192, gpu_memory_utilization=0.90,
              loading_threads=None, host_numa_nodes=None,
              kv_cache_memory_bytes=None, dcp_overlay=None, dcp_draft=None,
              dcp2_overlay_manifest_sha256=None, dcp_draft_manifest_sha256=None):
    topology = Topology.named(layout)
    integer(max_model_len, "max_model_len", minimum=512)
    integer(batched_tokens, "batched_tokens", minimum=128)
    if (type(gpu_memory_utilization) not in (int, float)
            or not 0.5 <= gpu_memory_utilization <= 0.92):
        raise ValueError("An explicit safe memory-utilization target in [0.5,0.92] is required")
    if kv_cache_memory_bytes is not None:
        integer(kv_cache_memory_bytes, "kv_cache_memory_bytes", minimum=1)
    if loading_threads is not None:
        integer(loading_threads, "loading_threads", minimum=1)
        if loading_threads > 128:
            raise ValueError("Loading threads must not exceed the frozen 128-thread budget")
    if host_numa_nodes is not None:
        node_list(host_numa_nodes)
    if not isinstance(served_model, str) or not served_model.startswith("dsv41-dp-"):
        raise ValueError("A distinct served-model identity for the new DP candidate is required")
    if topology.dcp == 2:
        if (layout != "dp4-tp2-dcp2" or not isinstance(dcp_overlay, str)
                or not Path(dcp_overlay).is_absolute() or not isinstance(dcp_draft, str)
                or not Path(dcp_draft).is_absolute()
                or not isinstance(dcp2_overlay_manifest_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", dcp2_overlay_manifest_sha256) is None
                or dcp_draft_manifest_sha256 != DCP2_DRAFT_MANIFEST_SHA256):
            raise ValueError("DP4/TP2/DCP2 requires the exact pinned Engram and best draft overlays")
    elif any(value is not None for value in
             (dcp_overlay, dcp_draft, dcp2_overlay_manifest_sha256, dcp_draft_manifest_sha256)):
        raise ValueError("DCP overlays are admitted only for DP4/TP2/DCP2")
    args = argparse.Namespace(
        source=Path(source), output=Path(output), form="a2-tp8-a3sim",
        served_model=served_model, max_model_len=max_model_len,
        batched_tokens=batched_tokens, device_index=0,
        physical_chips="8,9,10,11,12,13,14,15", physical_worker=None,
        dcp_det_reduce=None,
    )
    base = tp8_plan(candidate, args)
    kw, env = copy.deepcopy(base["model_kwargs"]), copy.deepcopy(base["environment"])
    env.pop(NUMA_ENV, None)
    if host_numa_nodes is not None:
        env[NUMA_ENV] = host_numa_nodes
    if kw.get("enable_expert_parallel") is not True:
        raise ValueError("The immutable source must already enable expert parallelism")
    if kw.get("async_scheduling") is not True or env.get("DSPARK_ASYNC_COUNTER") != "1":
        raise ValueError("The current best async candidate and its real counters are required")
    if loading_threads is not None:
        loader = kw.get("model_loader_extra_config")
        if (not isinstance(loader, dict) or loader.get("enable_multithread_load") is not True
                or type(loader.get("num_threads")) is not int
                or loading_threads > loader["num_threads"]
                or kw.get("safetensors_load_strategy") not in (None, "lazy")):
            raise ValueError("A loading-thread override may only reduce the frozen lazy multithread loader budget")
        loader["num_threads"] = loading_threads
    kw.update(worker_cls=WORKER, tensor_parallel_size=topology.tp,
              data_parallel_size=topology.dp, data_parallel_size_local=topology.dp,
              decode_context_parallel_size=topology.dcp,
              gpu_memory_utilization=float(gpu_memory_utilization),
              kv_cache_memory_bytes=kv_cache_memory_bytes)
    env.update(DELIVERY_FORM=layout, DP_MATRIX_LAYOUT=layout,
               DP_MATRIX_NATIVE_ROOT=str(Path(output) / "native"),
               DELIVERY_BASE_WORKER="parallel_tiny.dspark_formal_prep.dp_matrix_derived_worker.FormalDSparkWorker")
    if topology.dcp == 2:
        env.update(V41_DCP="1", V41_DCP_DET_REDUCE="1", V41_DCP_REPLICATE_INDEXER="1",
                   V41_ENGRAM_REUSE_EP_GROUP="1", V41_ENGRAM_DEVICE_INDEX="0",
                   DELIVERY_DCP_OVERLAY=dcp_overlay, DELIVERY_DCP_DRAFT=dcp_draft,
                   DP_MATRIX_DCP2_OVERLAY_MANIFEST_SHA256=dcp2_overlay_manifest_sha256,
                   DP_MATRIX_DCP2_DRAFT_MANIFEST_SHA256=dcp_draft_manifest_sha256,
                   DP_MATRIX_DCP2_CONTRACT_SHA256=DCP2_CONTRACT_SHA256)
        env.pop("V41_DCP_ALLOW_CAPACITY_PROBE", None)
    env.pop("STACK_TP8_ARM", None)
    env.pop("V41_DCP_ALLOW_CAPACITY_PROBE", None)
    # Match the pre-import shell cleanup: a stale env pin must not reappear
    # when service.serve applies the fingerprinted plan to os.environ.
    env.pop("KV_CACHE_MEMORY_BYTES", None)
    changes = {key: {"base": base["model_kwargs"].get(key), "new": value}
               for key, value in kw.items() if base["model_kwargs"].get(key) != value}
    return {
        "schema": "native_dp_service_plan_v1", "layout": layout,
        "served_model": served_model, "source_root": str(source),
        "source_candidate_fingerprint": fingerprint(candidate),
        "base_tp8_plan_fingerprint": base["runtime_fingerprint"],
        "topology": topology.plan(), "model_kwargs": kw, "environment": env,
        "model_kwargs_derivation": changes,
        "dcp2_overlay_manifest_sha256": dcp2_overlay_manifest_sha256,
        "dcp_draft_manifest_sha256": dcp_draft_manifest_sha256,
        "dcp2_contract_sha256": DCP2_CONTRACT_SHA256 if topology.dcp == 2 else None,
        "runtime_fingerprint": fingerprint({"model_kwargs": kw, "environment": env}),
        "kv_budget_mode": "automatic_profiling" if kv_cache_memory_bytes is None else "fixed_pool",
        "native_data_parallel_required": True, "independent_tp_replicas": False,
        "service_started": False, "actual_quality_passed": False,
        "actual_performance_observed": False, "actual_kv_capacity_observed": False,
    }
