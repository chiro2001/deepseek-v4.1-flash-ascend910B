# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
from typing_extensions import override

from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion


class CPUOffloadingSpec(OffloadingSpec):
    BLOCK_SIZE_ALIGNMENT = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        definitions: dict[str, OffloadingMetricMetadata] = {
            CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by active "
                    "transfers (0.0 = idle, 1.0 = saturated). Sustained high "
                    "values indicate transfers (stores or promotions) may be "
                    "dropped due to insufficient capacity."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "in-flight stores that have not yet "
                    "completed (0.0 = idle, 1.0 = saturated)."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "in-flight loads that have not yet "
                    "completed (0.0 = idle, 1.0 = saturated)."
                ),
            ),
            CPUOffloadingMetrics.CPU_ALLOCATION_SIZE: OffloadingHistogramMetadata(
                documentation=(
                    "Histogram of the number of CPU blocks requested by each "
                    "KV offload prepare_store call."
                ),
                buckets=(1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144),
            ),
        }
        store_threshold = int(extra_config.get("store_threshold", 0))
        if store_threshold >= 2:
            definitions[CPUOffloadingMetrics.STORES_SKIPPED] = (
                OffloadingCounterMetadata(
                    documentation=(
                        "Number of KV offload stores skipped because the reuse "
                        "threshold was not reached."
                    ),
                )
            )
        return definitions

    def __init__(self, config: OffloadingConfig):
        super().__init__(config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        world_size = config.parallel.world_size
        self.num_blocks = 0
        self.kv_bytes_per_chunk = 0
        self.cpu_page_size_per_worker = 0
        self.replicated_layout = config.replicated_layout and self._uses_shared_region()
        if config.worker_kv_bytes_per_block > 0 and world_size > 0:
            num_copies = 1 if self.replicated_layout else world_size
            kv_bytes_per_block = config.worker_kv_bytes_per_block * num_copies
            kv_bytes_per_chunk = kv_bytes_per_block * self.blocks_per_chunk

            # calculate cpu_page_size_per_worker
            self.cpu_page_size_per_worker = kv_bytes_per_chunk // num_copies

            # calculate num_blocks
            aligned_kv_bytes_per_chunk = round_up(
                kv_bytes_per_chunk, self.BLOCK_SIZE_ALIGNMENT
            )
            self.num_blocks = int(cpu_bytes_to_use) // aligned_kv_bytes_per_chunk

            # Expose aligned_kv_bytes_per_chunk as
            # kv_bytes_per_chunk. Note that this might contain
            # some padding. i.e. each offloaded block is of the form,
            # |--- W0-B0---|---- W1-B0---| ... |---- Wn-B0---| *** maybe-pad *** |
            # or |--- B0 (single copy) ---| *** maybe-pad *** |
            self.kv_bytes_per_chunk = aligned_kv_bytes_per_chunk

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._worker: CPUOffloadingWorker | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")
        self.cache_policy_module_path: str | None = self.extra_config.get(
            "cache_policy_module_path"
        )

        # [L3_8card] 池子的第一手数字（记账 vs 物理），进 serve.log 供自检/账目用。
        try:
            from vllm.v1.kv_offload.cpu.pgp_manager import bpc_map_from_extra

            _bpc_map = bpc_map_from_extra(self.extra_config)
            _parallel = getattr(config, "parallel", None)
            print(
                "[L3_8card] CPU 卸载池: num_units=%s kv_bytes_per_unit=%s "
                "cpu_page_size_per_worker=%s replicated_layout=%s "
                "num_copies=%s blocks_per_chunk=%s per_group=%s "
                "cpu_bytes_to_use=%s worker_kv_bytes_per_block=%s world_size=%s"
                % (
                    self.num_blocks,
                    self.kv_bytes_per_chunk,
                    self.cpu_page_size_per_worker,
                    self.replicated_layout,
                    1
                    if self.replicated_layout
                    else getattr(_parallel, "world_size", None),
                    self.blocks_per_chunk,
                    None
                    if _bpc_map is None
                    else {k: v for k, v in sorted(_bpc_map.items())},
                    self.extra_config.get("cpu_bytes_to_use"),
                    getattr(config, "worker_kv_bytes_per_block", None),
                    getattr(_parallel, "world_size", None),
                ),
                flush=True,
            )
        except Exception as _exc:  # noqa: BLE001 - 日志失败不能挡住起服
            print(f"[L3_8card] 池日志失败: {_exc!r}", flush=True)

    @override
    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            # store_threshold: how many times a block must appear in lookup()
            # before it is eligible for CPU offloading.  Values < 2 disable
            # filtering (a threshold of 1 equals no filter; 0 is the default).
            store_threshold = int(self.extra_config.get("store_threshold", 0))

            # Maximum entries in the internal tracker's LRU table.
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))

            # [L3_8card][per-group bpc] unit 池模式 => PerGroupBPCManager；否则原样。
            from vllm.v1.kv_offload.cpu.pgp_manager import (
                PerGroupBPCManager,
                bpc_map_from_extra,
            )

            _bpc_map = bpc_map_from_extra(self.extra_config)
            # [K_l1_8card][L1] 按组配额 manager（P2_POOL_PATCH=1 且权重已注入时；见 a2/logs/042）。
            #   基类仍是 L5 的 PerGroupBPCManager ⇒ "一格 = 1 个 GPU block、一个 key 占 bpc_g 格"
            #   的语义不变，只是 unit 号 = (group << 20) | 组内行号、每组的行数按权重配额切。
            from vllm.v1.kv_offload.cpu import p2_pool as _p2

            _l1_weights = _p2.weights_from_extra(self.extra_config)
            _l1_comp = _p2.comp_from_extra(self.extra_config)
            if _bpc_map and _l1_weights and _p2.patch_enabled():
                _l1_quota_cls = _p2.make_quota_manager(PerGroupBPCManager)
                self._manager = _l1_quota_cls(
                    num_blocks=self.num_blocks,
                    bpc_by_group=_bpc_map,
                    weights_by_group=_l1_weights,
                    base_by_group=_p2.component_bases(
                        _p2.quota_from_config(self.num_blocks, _l1_weights), _l1_comp
                    ),
                    cache_policy=self.eviction_policy,
                    cache_policy_module_path=self.cache_policy_module_path,
                    enable_events=self.kv_events_config.enable_kv_cache_events,
                    store_threshold=store_threshold,
                    max_tracker_size=max_tracker_size,
                )
                print(
                    "[P2_poolsizing] [K_l1_8card] 按组配额 manager 生效：num_units=%d "
                    "基类=PerGroupBPCManager bpc=%s weights=%s 分量=%s"
                    % (
                        self.num_blocks,
                        dict(sorted(_bpc_map.items())),
                        dict(sorted(_l1_weights.items())),
                        None if _l1_comp is None else dict(sorted(_l1_comp.items())),
                    ),
                    flush=True,
                )
            elif _bpc_map:
                self._manager = PerGroupBPCManager(
                    num_blocks=self.num_blocks,
                    bpc_by_group=_bpc_map,
                    cache_policy=self.eviction_policy,
                    cache_policy_module_path=self.cache_policy_module_path,
                    enable_events=self.kv_events_config.enable_kv_cache_events,
                    store_threshold=store_threshold,
                    max_tracker_size=max_tracker_size,
                )
                print(
                    "[L3_8card] PerGroupBPCManager 生效：num_units=%d bpc=%s"
                    % (self.num_blocks, dict(sorted(_bpc_map.items()))),
                    flush=True,
                )
            else:
                self._manager = CPUOffloadingManager(
                    num_blocks=self.num_blocks,
                    cache_policy=self.eviction_policy,
                    cache_policy_module_path=self.cache_policy_module_path,
                    enable_events=self.kv_events_config.enable_kv_cache_events,
                    store_threshold=store_threshold,
                    max_tracker_size=max_tracker_size,
                )
        return self._manager

    def _uses_shared_region(self) -> bool:
        """Whether the worker CPU buffer is the shared mmap region (vs a private
        per-rank tensor); replicated-layout dedup is gated on this being True."""
        return current_platform.is_cuda_alike()

    def create_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        mmap_region: SharedOffloadRegion | None = None
        # num_blocks == 0 would size the region to zero bytes, which cannot be
        # mmap'd; fall back to the tensor path (empty tensors) as before.
        if self._uses_shared_region() and self.num_blocks > 0:
            # Replicated layout puts all ranks on slot 0 (single MLA copy);
            # otherwise each rank takes its own slot by physical device index.
            if self.replicated_layout:
                rank = 0
            else:
                world_size = self.config.parallel.world_size
                rank = torch.accelerator.current_device_index() % world_size
            mmap_region = SharedOffloadRegion(
                engine_id=self.config.engine_id,
                num_blocks=self.num_blocks,
                rank=rank,
                kv_bytes_per_block=self.kv_bytes_per_chunk,
                cpu_page_size=self.cpu_page_size_per_worker,
            )
        return CPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=self.blocks_per_chunk,
            num_cpu_blocks=self.num_blocks,
            mmap_region=mmap_region,
        )

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if not self._worker:
            if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
                raise Exception(
                    "CPU Offloading is currently only supported on CUDA-alike "
                    "and XPU GPUs"
                )
            self._worker = self.create_worker(kv_caches)

        assert self._worker is not None
        return self._worker
