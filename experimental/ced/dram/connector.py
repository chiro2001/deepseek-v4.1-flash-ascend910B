"""CED P-only adapter around Ascend's native CPU KV offloading connector."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.v1.kv_offload.base import CanonicalKVCaches
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.npu import (
    NPUOffloadingSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.offloading_connector import (
    AscendOffloadingConnector,
)

from .contract import alignment_unit, prefill_groups


@dataclass
class CEDOffloadingMetadata(OffloadingConnectorMetadata):
    finished_store_requests: set[str] = field(default_factory=set)


class CEDNPUOffloadingSpec(NPUOffloadingSpec):
    def create_worker(self, kv_caches: CanonicalKVCaches):
        excluded = set(self.extra_config["ced_excluded_groups"])
        refs = [
            [] if idx in excluded else list(group)
            for idx, group in enumerate(kv_caches.group_data_refs)
        ]
        # Native workers allocate one host tensor per canonical tensor. Remove
        # tensors referenced only by uncomputed groups/state, while preserving
        # the group vector and remapping only canonical tensor indices.
        used = sorted({ref.tensor_idx for group in refs for ref in group})
        remap = {old: new for new, old in enumerate(used)}
        filtered = CanonicalKVCaches(
            tensors=[kv_caches.tensors[idx] for idx in used],
            group_data_refs=[
                [replace(ref, tensor_idx=remap[ref.tensor_idx]) for ref in group]
                for group in refs
            ],
        )
        print(
            "[CED-DRAM] canonical tensors %d -> %d excluded=%s"
            % (len(kv_caches.tensors), len(filtered.tensors), sorted(excluded)),
            flush=True,
        )
        return super().create_worker(filtered)


class CEDOffloadingConnector(AscendOffloadingConnector):
    """Save and load encoder KV, reporting one async save barrier per request.

    Native offloading fences page reuse with jobs_to_flush but never reports
    finished_sending. CED adds request completion so MultiConnector can wait
    for *both* PD consumption and the final DRAM writes before freeing pages.
    Native SWA/preemption flushes remain necessary while requests are running.
    """

    def __init__(self, vllm_config, role, kv_cache_config):
        if os.environ.get("V41_CED_ROLE") != "prefill":
            raise ValueError("CEDOffloadingConnector requires V41_CED_ROLE=prefill")
        if not vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("CED DRAM requires prefix caching")
        group_contract = prefill_groups(kv_cache_config.kv_cache_groups)
        if group_contract.missing_swa != (7, 8, 9, 10, 11):
            raise ValueError("CED DRAM received an incompatible P KV group layout")
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        extra["ced_excluded_groups"] = sorted(
            (*group_contract.missing_swa, *group_contract.recurrent)
        )
        super().__init__(vllm_config, role, kv_cache_config)
        self._ced_finish_barriers: set[str] = set()
        self._ced_worker_barriers: set[str] = set()
        self._ced_job_requests: dict[int, str] = {}
        self._ced_request_jobs: dict[str, set[int]] = {}
        self._ced_seen_metadata = None
        scheduler = self.connector_scheduler
        if scheduler is not None:
            sizes = [
                c.tokens_per_chunk
                for c in scheduler.config.kv_group_configs
                if c.offload_participating and c.sliding_window_size_in_chunks is None
            ]
            sizes += [
                c.tokens_per_block
                for c in scheduler.config.kv_group_configs
                if c.offload_participating
            ]
            unit = alignment_unit(sizes)
            scheduler.config = scheduler.config._replace(apc_align_unit=unit)
            self._ced_alignment = unit
            print(
                "[CED-DRAM] participating=%s excluded=%s alignment=%d"
                % (group_contract.participating, extra["ced_excluded_groups"], unit),
                flush=True,
            )

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        tokens, asynchronous = super().get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        if tokens:
            boundary = num_computed_tokens + tokens
            if boundary >= request.num_tokens or boundary % self._ced_alignment:
                raise RuntimeError("CED DRAM lookup must leave an aligned nonzero tail")
            if request.kv_transfer_params is None:
                request.kv_transfer_params = {}
            request.kv_transfer_params["_ced_dram_load_boundary"] = boundary
            request.kv_transfer_params["_ced_dram_alignment"] = self._ced_alignment
            print(
                "[CED-DRAM] lookup req=%s local=%d external=%d boundary=%d tail=%d"
                % (request.request_id, num_computed_tokens, tokens, boundary,
                   request.num_tokens - boundary),
                flush=True,
            )
        return tokens, asynchronous

    def request_finished_all_groups(self, request, block_ids):
        tracked = request.request_id in self.connector_scheduler._req_status
        super().request_finished_all_groups(request, block_ids)
        if tracked:
            self._ced_finish_barriers.add(request.request_id)
        return tracked, None

    def build_connector_meta(self, scheduler_output):
        metadata = super().build_connector_meta(scheduler_output)
        barriers = self._ced_finish_barriers
        self._ced_finish_barriers = set()
        return CEDOffloadingMetadata(
            load_jobs=metadata.load_jobs,
            store_jobs=metadata.store_jobs,
            jobs_to_flush=metadata.jobs_to_flush,
            finished_store_requests=barriers,
        )

    def has_pending_push_work(self):
        return bool(self._ced_finish_barriers) or super().has_pending_push_work()

    def _observe_worker_metadata(self, metadata):
        if not isinstance(metadata, CEDOffloadingMetadata):
            raise TypeError("CED DRAM expected save-barrier metadata")
        if metadata is self._ced_seen_metadata:
            return
        self._ced_seen_metadata = metadata
        # An aborted request can still have an async DRAM load in flight.
        # Its completion barrier must protect destination pages as well as
        # store source pages until the native worker reports completion.
        for job_id, job in {**metadata.load_jobs, **metadata.store_jobs}.items():
            if job_id not in self._ced_job_requests:
                self._ced_job_requests[job_id] = job.req_id
                self._ced_request_jobs.setdefault(job.req_id, set()).add(job_id)
        self._ced_worker_barriers.update(metadata.finished_store_requests)

    def handle_preemptions(self, kv_connector_metadata):
        self._observe_worker_metadata(kv_connector_metadata)
        super().handle_preemptions(kv_connector_metadata)

    def get_finished(self, finished_req_ids):
        self._observe_worker_metadata(self._connector_metadata)
        sending, receiving = super().get_finished(finished_req_ids)
        completed = self.connector_worker._connector_worker_meta.completed_jobs
        for job_id in completed:
            req_id = self._ced_job_requests.pop(job_id, None)
            if req_id is not None:
                self._ced_request_jobs[req_id].discard(job_id)
        sending = set(sending or ())
        for req_id in tuple(self._ced_worker_barriers):
            if not self._ced_request_jobs.get(req_id):
                self._ced_worker_barriers.remove(req_id)
                self._ced_request_jobs.pop(req_id, None)
                sending.add(req_id)
                print("[CED-DRAM] save barrier complete req=%s" % req_id, flush=True)
        return sending, receiving

    def reset_cache(self):
        if self._ced_finish_barriers:
            raise RuntimeError("Cannot reset CED DRAM while save barriers are pending")
        return super().reset_cache()


# MultiConnector's stats deserializer resolves child class names through the
# registry even when construction used kv_connector_module_path. Register the
# external class so API-server metrics use the same class as scheduler/workers.
if "CEDOffloadingConnector" not in KVConnectorFactory._registry:
    KVConnectorFactory.register_connector(
        "CEDOffloadingConnector", __name__, "CEDOffloadingConnector"
    )
