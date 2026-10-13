"""Deferred native worker for a new, provenance-pinned DP candidate."""
import hashlib
from collections.abc import Mapping
import importlib.util
import inspect
import math
import os
from pathlib import Path
import re
import sys

from .build_runtime import verify
from .capacity_metadata import backing_storages
from .runtime_contract import validate_worker_config
from .topology import Topology


def _load(name, path):
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    existing = sys.modules.get(name)
    if existing is not None:
        if (Path(existing.__file__).resolve() != path.resolve()
                or getattr(existing, "__dp_matrix_source_sha256__", None) != source_sha):
            raise ValueError("Runtime module already bound to another source: " + name)
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        module.__dp_matrix_source_sha256__ = source_sha
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_runtime = Path(os.environ["DP_MATRIX_RUNTIME"]).resolve(strict=True)
_manifest = verify(_runtime)
_load("formal_dspark_counters", _runtime / "formal_dspark_counters.py")
_base_name = "parallel_tiny.dspark_formal_prep.dp_matrix_derived_worker"
_load(_base_name, _runtime / "worker.py")
os.environ["DELIVERY_BASE_WORKER"] = _base_name + ".FormalDSparkWorker"
# Imported only after the derived counter and worker are explicitly installed.
from delivery_matrix.worker import DeliveryWorker as Base


def source_file(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class DPWorker(Base):
    def load_model(self, *args, **kwargs):
        from .host_memory import verify_interleave
        self._dp_host_memory_policy_before_load = verify_interleave()
        topology = validate_worker_config(self.vllm_config, os.environ)
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        root = Path(os.environ["DP_MATRIX_NATIVE_ROOT"]) / f"dp{dp_rank}"
        root.mkdir(parents=True, exist_ok=True)
        os.environ["STACK_DSPARK_FORMAL_OUTPUT"] = str(root)
        self._dp_topology = topology
        return super().load_model(*args, **kwargs)

    def initialize_from_config(self, kv_cache_config):
        result = super().initialize_from_config(kv_cache_config)
        self._dp_kv_cache_config = kv_cache_config
        return result

    def _dp_fields(self):
        from vllm.distributed import get_tp_group, get_dcp_group, get_ep_group
        import torch
        topology = Topology.named(os.environ["DP_MATRIX_LAYOUT"],
                                  chips=os.environ["STACK_PHYSICAL_CHIPS"])
        tp, dcp, ep = get_tp_group(), get_dcp_group(), get_ep_group()
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        local_index = int(torch.npu.current_device())
        if (tp.world_size != topology.tp or dcp.world_size != topology.dcp
                or ep.world_size != 8 or not 0 <= local_index < 8):
            raise ValueError("Actual native communication groups/device differ from the new DP plan")
        return {
            "rank": tp.rank_in_group, "tp_rank": tp.rank_in_group,
            "dp_rank": dp_rank, "ep_rank": ep.rank_in_group,
            "dcp_rank": dcp.rank_in_group, "dp": topology.dp,
            "tp": tp.world_size, "dcp": dcp.world_size, "ep_world_size": ep.world_size,
            "enable_expert_parallel": self.vllm_config.parallel_config.enable_expert_parallel,
            "worker_pid": os.getpid(), "local_npu_index": local_index,
            "physical_chip": topology.chips[local_index],
            "physical_chip_scope": "visible-device mapping; independently verify host PID/NPU/cgroup",
            "tp_group_ranks": list(tp.ranks), "dcp_group_ranks": list(dcp.ranks),
            "ep_group_ranks": list(ep.ranks),
        }

    def dp_describe(self):
        from .host_memory import verify_interleave
        result = super().delivery_describe()
        result.update(self._dp_fields())
        counter = self.model_runner._formal_dspark_counter
        result.update(
            dp_worker_source=source_file(__file__),
            derived_counter_source=source_file(sys.modules[counter.__class__.__module__].__file__),
            derived_worker_source=source_file(_runtime / "worker.py"),
            derived_runtime_manifest=source_file(_runtime / "manifest.json"),
            draft_observer_sources=counter.actual_draft_graph_observer.sources,
            draft_propose_source=source_file(inspect.getfile(inspect.unwrap(self.model_runner.drafter._propose))),
            terminal_join_scope="active DP replica's TP group; not a global EP barrier",
            engram_runtime=self._engram_runtime_description(),
            host_memory_policy_before_load=self._dp_host_memory_policy_before_load,
            host_memory_policy_current=verify_interleave(),
        )
        return result

    def _engram_runtime_description(self):
        """Inspect actual loaded Engram objects and group/device metadata only."""
        import torch.distributed as dist
        from vllm_ascend.ascend_config import get_ascend_config

        model = self.model_runner.model
        actual = []
        for name, module in model.named_modules():
            if (hasattr(module, "query_group") and hasattr(module, "weight")
                    and hasattr(module, "storage_format") and name.endswith(".engram.embed")):
                query, weight = module.query_group, module.weight
                actual.append({
                    "module_name": name,
                    "storage_format": module.storage_format,
                    "weight_shape": list(weight.shape),
                    "weight_dtype": str(weight.dtype),
                    "weight_device": str(weight.device),
                    "weight_device_index": weight.device.index,
                    "engram_rank": int(query.rank), "engram_world_size": int(query.size),
                    "engram_group_global_ranks": list(dist.get_process_group_ranks(query.group)),
                    "tp_group_global_ranks": list(dist.get_process_group_ranks(query.tp_group)),
                    "is_tp_source": bool(query.is_source),
                })
        config = getattr(model, "config", None)
        if config is None:
            for child in model.modules():
                candidate = getattr(child, "config", None)
                if candidate is not None and hasattr(candidate, "engram_layer_ids"):
                    config = candidate
                    break
        layers = list(getattr(config, "engram_layer_ids", ()))
        ascend = get_ascend_config()
        if ascend.enable_engram and (not layers or len(actual) != len(layers)):
            raise ValueError("Engram is enabled but actual loaded table objects/layers are incomplete")
        if not ascend.enable_engram and actual:
            raise ValueError("Engram table objects are present while the Ascend config disables Engram")
        module = sys.modules.get("vllm_ascend.models.deepseek_v41.engram_hbm")
        module_path = None if module is None else Path(module.__file__).resolve()
        return {
            "enabled_in_actual_ascend_config": bool(ascend.enable_engram),
            "storage_format_in_actual_ascend_config": getattr(ascend, "engram_storage", None),
            "configured_layer_ids": layers,
            "loaded_layer_ids": sorted({
                int(match.group(1)) for row in actual
                if (match := re.search(r"(?:^|\.)layers\.(\d+)\.engram\.embed$", row["module_name"]))
            }),
            "module_source": None if module_path is None else source_file(module_path),
            "device_index_fastpath_enabled": os.environ.get("V41_ENGRAM_DEVICE_INDEX", "")
                not in ("0", "false", "off", "no", ""),
            "shards": actual,
            "tensor_data_read": False,
        }

    def delivery_begin(self):
        return {**super().delivery_begin(), **self._dp_fields()}

    def delivery_arm(self, tag, max_tokens):
        reply = super().delivery_arm(tag, max_tokens)
        return {**reply, **self._dp_fields(),
                "cumulative_before": dict(self._delivery_totals)}

    def delivery_metrics(self):
        from .draft_proof import from_counter_request
        result = super().delivery_metrics()
        tag = (result.get("last_window") or {}).get("tag")
        counter = self.model_runner._formal_dspark_counter
        for detail in result.get("requests", []):
            if detail.get("tag") == tag:
                request = counter.requests.get(detail["request_id"])
                if request is None:
                    raise ValueError("Actual counter request is missing for draft phase proof")
                detail["draft_phase_counts"] = from_counter_request(request)
        return {**result, **self._dp_fields()}

    def dp_status(self):
        counter = self.model_runner._formal_dspark_counter
        return {**self._dp_fields(), "counter_tag": counter.tag,
                "counter_pending": counter.pending is not None,
                "counter_request_ids": sorted(counter.requests),
                "sample_pending": self.model_runner.execute_model_state is not None}

    def dp_capacity(self):
        # Actual allocator and backing-storage metadata; this is not a successful long-input test.
        config = self._dp_kv_cache_config
        tensors = [{"size_bytes": int(tensor.size), "shared_by": list(tensor.shared_by),
                    "offset": int(tensor.offset), "block_stride": int(tensor.block_stride)}
                   for tensor in config.kv_cache_tensors]
        # V4.1 indexer entries are (keys, scales) tuples; aliases count once.
        # No tensor values, D2H, item(), numpy() or tolist() are inspected.
        storages = backing_storages(self.model_runner.kv_caches)
        groups = []
        for group in config.kv_cache_groups:
            spec = group.kv_cache_spec
            leaves = getattr(spec, "kv_cache_specs", None)
            members = leaves.items() if isinstance(leaves, Mapping) else [("group", spec)]
            member_rows = []
            for name, leaf in members:
                row = {"name": name,
                       "spec_class": type(leaf).__module__ + "." + type(leaf).__name__}
                row.update({key: getattr(leaf, key) for key in
                            ("block_size", "storage_block_size", "page_size_bytes",
                             "compress_ratio", "dcp_world_size", "sliding_window",
                             "extra_retained_tokens")
                            if isinstance(getattr(leaf, key, None), int)})
                row["max_memory_usage_bytes"] = int(leaf.max_memory_usage_bytes(self.vllm_config))
                row["store_on_host"] = bool(getattr(leaf, "store_on_host", False))
                if row["store_on_host"]:
                    raise ValueError("HBM capacity cannot use the V4.1 device formula for a host KV group")
                row["required_blocks_for_max_model_len"] = math.ceil(
                    row["max_memory_usage_bytes"] / row["page_size_bytes"])
                member_rows.append(row)
            groups.append({
                "layer_names": list(group.layer_names),
                "spec_class": type(spec).__module__ + "." + type(spec).__name__,
                "required_blocks_for_max_model_len": max(
                    row["required_blocks_for_max_model_len"] for row in member_rows),
                "members": member_rows,
            })
        import vllm.v1.core.kv_cache_utils as kv_utils
        from vllm_ascend.core.deepseek_v41 import pool_bytes_per_block, request_blocks

        request_block_demand = int(request_blocks(self.vllm_config, config.kv_cache_groups))
        group_block_demand = sum(group["required_blocks_for_max_model_len"] for group in groups)
        if request_block_demand != group_block_demand or request_block_demand <= 0:
            raise ValueError("V4.1 group block accounting differs from the loaded allocator helper")
        get_concurrency = kv_utils.get_max_concurrency_for_kv_cache_config
        helper_path = Path(inspect.getsourcefile(get_concurrency)).resolve()
        if "patch_kv_cache_utils.py" not in helper_path.name:
            raise ValueError("The loaded Ascend V4.1 KV capacity override is not active")
        max_concurrency = float(get_concurrency(self.vllm_config, config))
        expected_concurrency = max(0, int(config.num_blocks) - 1) / request_block_demand
        if not math.isclose(max_concurrency, expected_concurrency, rel_tol=0, abs_tol=1e-12):
            raise ValueError("V4.1 max-concurrency helper differs from its block accounting")
        pool_bytes = int(pool_bytes_per_block(config.kv_cache_groups))
        if sum(row["size_bytes"] for row in tensors) != int(config.num_blocks) * pool_bytes:
            raise ValueError("Actual KV descriptors differ from the V4.1 slot layout")
        max_model_len = int(self.vllm_config.model_config.max_model_len)
        return {**self._dp_fields(), "num_blocks": int(config.num_blocks),
                "kv_cache_tensors": tensors, "kv_cache_groups": groups,
                "descriptor_pool_bytes": sum(row["size_bytes"] for row in tensors),
                "backing_storage_metadata": storages,
                "backing_storage_bytes": sum(row["size_bytes"] for row in storages),
                "physical_memory_scope": "unique backing-storage metadata, cross-check device/allocator/NPU receipts",
                "pool_bytes_per_global_block": pool_bytes,
                "null_block_reserved": True,
                "request_blocks_at_max_model_len": request_block_demand,
                "request_blocks_by_group": [group["required_blocks_for_max_model_len"] for group in groups],
                "max_concurrency_for_max_model_len": max_concurrency,
                "logical_token_capacity_at_max_model_len": int(max_concurrency * max_model_len),
                "capacity_helper_source": source_file(helper_path),
                "request_block_helper_source": source_file(inspect.getsourcefile(request_blocks)),
                "pool_layout_helper_source": source_file(inspect.getsourcefile(pool_bytes_per_block)),
                "capacity_scope": "actual patched Ascend V4.1 helper; per TP/DCP rank; DP capacity uses the bottleneck rank",
                "available_kv_cache_memory_bytes": getattr(self, "available_kv_cache_memory_bytes", None),
                "requested_memory_bytes": getattr(self, "requested_memory", None),
                "peak_activation_memory_bytes": getattr(self, "peak_activation_memory", None),
                "non_torch_memory_bytes": getattr(self, "non_torch_memory", None),
                "model_memory_usage_bytes": getattr(self.model_runner, "model_memory_usage", None),
                "configured_max_model_len": self.vllm_config.model_config.max_model_len,
                "configured_max_num_seqs": self.vllm_config.scheduler_config.max_num_seqs,
                "max_in_flight_tokens": getattr(self.vllm_config, "max_in_flight_tokens", None),
                "kv_cache_memory_bytes": self.vllm_config.cache_config.kv_cache_memory_bytes,
                "long_request_validated": False,
                "logical_token_capacity": int(max_concurrency * max_model_len),
                "logical_capacity_scope": "configured-max-length token equivalent, actual Ascend group allocator; not a long-request PASS"}
