# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Ascend adaptation of vLLM's native CPU offloading spec."""

from __future__ import annotations

from dataclasses import replace

from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec as _CPUOffloadingSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.cpu_npu import (
    NPUOffloadingWorker,
)
# [K_l1_8card][L1] 每张 canonical 张量按**自己的行数**分配宿主池的 worker（见 a2/logs/042）。
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.p2_worker import (
    P2OffloadingWorker,
)


def _normalize_legacy_num_blocks(
    config: OffloadingConfig,
    alignment: int,
) -> tuple[OffloadingConfig, int | None]:
    """Translate legacy block capacity without mutating vLLM's config."""
    extra_config = config.extra_config
    if extra_config.get("cpu_bytes_to_use") is not None:
        return config, None

    num_cpu_blocks = extra_config.get("num_cpu_blocks")
    if num_cpu_blocks is None:
        return config, None
    num_cpu_blocks = int(num_cpu_blocks)
    if num_cpu_blocks <= 0:
        raise ValueError("num_cpu_blocks must be greater than 0")

    world_size = config.parallel.world_size
    worker_kv_bytes_per_block = config.worker_kv_bytes_per_block
    if worker_kv_bytes_per_block <= 0 or world_size <= 0:
        # The scheduler can construct the spec before worker cache sizing is
        # available. Use a non-zero placeholder to pass upstream validation;
        # NPUOffloadingSpec restores the legacy block capacity after init.
        cpu_bytes_to_use = 1
    else:
        kv_bytes_per_chunk = worker_kv_bytes_per_block * world_size * config.cache.blocks_per_chunk
        aligned_kv_bytes_per_chunk = round_up(
            kv_bytes_per_chunk,
            alignment,
        )
        cpu_bytes_to_use = num_cpu_blocks * aligned_kv_bytes_per_chunk

    normalized_extra_config = dict(extra_config)
    normalized_extra_config["cpu_bytes_to_use"] = cpu_bytes_to_use
    return replace(config, extra_config=normalized_extra_config), num_cpu_blocks


class NPUOffloadingSpec(_CPUOffloadingSpec):
    """Use vLLM's CPU manager with an Ascend-specific transfer worker."""

    # vLLM is skipped by the Ascend mypy invocation, so redeclare the
    # inherited worker cache without taking over its runtime initialization.
    _worker: NPUOffloadingWorker | None

    def __init__(self, config: OffloadingConfig):
        normalized_config, legacy_num_blocks = _normalize_legacy_num_blocks(
            config,
            self.BLOCK_SIZE_ALIGNMENT,
        )
        super().__init__(normalized_config)
        if legacy_num_blocks is not None and config.worker_kv_bytes_per_block <= 0:
            self.num_blocks = legacy_num_blocks

    def create_worker(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> NPUOffloadingWorker:
        # [K_l1_8card][L1] 行数 = 引用该张量的组里最大行区间上界（P2_POOL_PATCH=0 ⇒
        # 逐字回退 upstream 的"每张都 num_blocks 行"）。日志/校验在 p2_pool.worker_rows 里。
        from vllm.v1.kv_offload.cpu import p2_pool

        row_counts = p2_pool.worker_rows(self, kv_caches)
        return P2OffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=self.blocks_per_chunk,
            num_cpu_blocks=self.num_blocks,
            row_counts=row_counts,
        )

    def get_worker(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> OffloadingWorker:
        # CPUOffloadingSpec rejects non-CUDA/XPU platforms in get_worker().
        # Keep its worker cache and lifecycle, replacing only that platform
        # gate with the NPU-specific worker construction.
        if self._worker is None:
            self._worker = self.create_worker(kv_caches)
        return self._worker


# Compatibility alias for configurations that load this module directly.
CPUOffloadingSpec = NPUOffloadingSpec
