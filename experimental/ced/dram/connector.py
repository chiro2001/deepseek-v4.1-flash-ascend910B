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
    AscendOffloadingConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native import (
    shared_region_single_copy as _sc,
)

from .contract import alignment_unit, prefill_groups


@dataclass
class CEDOffloadingMetadata(OffloadingConnectorMetadata):
    finished_store_requests: set[str] = field(default_factory=set)
    engram_seed_pages: dict[int, list[int]] = field(default_factory=dict)
    engram_block_size: int = 0


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
        # [CED-SINGLE-COPY] 把 worker 换成带"跨 rank 逐字节校验"的版本。
        worker = self.connector_worker
        if worker is not None and _sc.enabled():
            self.connector_worker = CEDOffloadingConnectorWorker(
                worker.spec, vllm_config, kv_cache_config
            )
            print(
                "[CED-SINGLE-COPY] worker verification armed "
                "(verify=%s jobs=%s all_rows=%s)"
                % (
                    _sc.verify_enabled(),
                    os.environ.get("CED_DRAM_SINGLE_COPY_VERIFY_JOBS", "2"),
                    os.environ.get("CED_DRAM_SINGLE_COPY_VERIFY_ALL", "0"),
                ),
                flush=True,
            )
        self._ced_finish_barriers: set[str] = set()
        self._ced_worker_barriers: set[str] = set()
        self._ced_job_requests: dict[int, str] = {}
        self._ced_request_jobs: dict[str, set[int]] = {}
        self._ced_seen_metadata = None
        self._ced_engram_seed_pages: dict[int, list[int]] = {}
        self._ced_engram_block_size = 0
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

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        if not num_external_tokens:
            return
        boundary = request.kv_transfer_params["_ced_dram_load_boundary"]
        # Engram uses layer 0's SWA page table (G2), rather than G0's
        # compressed global table. The aligned load restores the previous
        # complete SWA page; hydrate its token mirror from the request.
        size = self.connector_scheduler.config.kv_group_configs[2].tokens_per_block
        if boundary <= 0 or boundary % size:
            raise RuntimeError("CED Engram restoration needs a complete prior SWA page")
        position = boundary // size - 1
        page = blocks.get_block_ids()[2][position]
        tokens = list(request.prompt_token_ids[position * size:boundary])
        if page <= 0 or len(tokens) != size:
            raise RuntimeError("CED DRAM load lacks the actual prior Engram token page")
        if self._ced_engram_block_size not in (0, size):
            raise RuntimeError("CED Engram block geometry changed within one worker")
        self._ced_engram_block_size = size
        self._ced_engram_seed_pages[page] = tokens

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
        seeds, self._ced_engram_seed_pages = self._ced_engram_seed_pages, {}
        return CEDOffloadingMetadata(
            load_jobs=metadata.load_jobs,
            store_jobs=metadata.store_jobs,
            jobs_to_flush=metadata.jobs_to_flush,
            finished_store_requests=barriers,
            engram_seed_pages=seeds,
            engram_block_size=self._ced_engram_block_size,
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
        if kv_connector_metadata.engram_seed_pages:
            from .engram_history import restore_history_pages
            restore_history_pages(kv_connector_metadata.engram_seed_pages,
                                  kv_connector_metadata.engram_block_size)
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


class CEDOffloadingConnectorWorker(AscendOffloadingConnectorWorker):
    """[CED-SINGLE-COPY] store 提交前做**跨 rank 逐字节**校验（fail-closed）。

    单副本布局把"rank0 写、其余 rank 只 ack"当成既定事实；一旦某个 rank 的
    store 源 KV 与 rank0 不同，该 rank 之后就会从共享区读到别人的 KV ⇒ 静默算错。
    所以在**前 N 个 store job**（默认 2）上，每个 rank 对自己的 GPU 源块做
    sha256 并会合比对，不一致**直接抛错**（引擎停在那一步，而不是悄悄跑坏）。

    为什么是"前 N 个"而不是全部：校验每个 job 都要 D2H + sha256，压满池时
    store 每秒几十 GB，全量校验会显著拖慢压力测试。前 N 个 job 足以覆盖
    "同一几何下 KV 是否逐字节复制"这个**结构性**命题；`VERIFY_ALL=1`
    或 `VERIFY_JOBS=-1` 可以改成全量（取证用）。
    """

    def __init__(self, spec, vllm_config, kv_cache_config):
        super().__init__(spec, vllm_config, kv_cache_config)
        self._sc_enabled = _sc.enabled()
        self._sc_verify = _sc.verify_enabled()
        try:
            self._sc_jobs_left = int(
                os.environ.get("CED_DRAM_SINGLE_COPY_VERIFY_JOBS", "2")
            )
        except ValueError:
            self._sc_jobs_left = 2
        self._sc_all_rows = os.environ.get("CED_DRAM_SINGLE_COPY_VERIFY_ALL", "0") == "1"
        self._sc_jobs_seen = 0
        parallel = getattr(spec.config, "parallel", None)
        rank = int(getattr(parallel, "rank", 0) or 0)
        world = int(getattr(parallel, "world_size", 1) or 1)
        engine_id = getattr(spec.config, "engine_id", "unknown")
        self._sc_verifier = (
            _sc.SourceVerifier(engine_id, rank, world)
            if (self._sc_enabled and self._sc_verify)
            else None
        )
        self._sc_reports: list[dict] = []
        # Keep a handle on the worker's NPU source tensors for hashing.
        self._sc_npu_tensors: list = []

    def register_kv_caches(self, kv_caches):
        super().register_kv_caches(kv_caches)
        worker = self.worker
        npu_tensors = []
        store_handler = getattr(worker, "_store_handler", None)
        if store_handler is not None:
            npu_tensors = list(getattr(store_handler, "src_tensors", ()) or ())
        self._sc_npu_tensors = npu_tensors
        if self._sc_enabled:
            _sc.log(
                "worker registered: npu_source_tensors=%d store_writer=%s"
                % (len(npu_tensors), self._is_store_writer)
            )

    def _sc_verify_store(self, job_id: int, src_spec) -> None:
        if self._sc_verifier is None or self._sc_jobs_left == 0:
            return
        if self._sc_jobs_left > 0:
            self._sc_jobs_left -= 1
        self._sc_jobs_seen += 1
        refs = getattr(self.worker, "kv_cache_groups_data_refs", None)
        if refs is None:
            refs = getattr(getattr(self.worker, "_store_handler", None),
                           "kv_cache_groups_data_refs", None)
        if not refs or not self._sc_npu_tensors:
            raise RuntimeError(
                "[CED-SINGLE-COPY] cannot verify store sources: worker tensors/refs "
                "not registered"
            )
        digests = _sc.digest_store_sources(
            self._sc_npu_tensors, refs, src_spec, self._sc_all_rows
        )
        if not digests:
            raise RuntimeError(
                f"[CED-SINGLE-COPY] store job {job_id} produced no verifiable rows"
            )
        self._sc_verifier.record(job_id, digests)
        # Non-blocking: `record()` spawns the rank0 watcher which either logs OK
        # or (on a genuine mismatch) kills the engine fail-closed. Never wait in
        # the step loop -- doing so deadlocked all eight ranks on 2026-10-10.

    def prepare_store_kv(self, metadata):
        # Verify on **every** rank (writer and non-writer alike): the whole point
        # is comparing rank0's bytes with the others'. Then hand off to the base
        # implementation, which is where the writer gate lives.
        if self._sc_verifier is not None:
            for job_id, entry in list(metadata.store_jobs.items()):
                self._sc_verify_store(job_id, entry.src_spec)
        super().prepare_store_kv(metadata)
