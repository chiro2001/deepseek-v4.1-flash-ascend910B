# SPDX-License-Identifier: Apache-2.0
# [K_l1_8card] worker 侧：每张 canonical 张量按**自己的行数**分配宿主池。
#
# 由挂载版 `native/npu.py::NPUOffloadingSpec.create_worker()` 实例化（传 row_counts）。
# `__init__` 逐字来自 `agents/P2_poolsizing/patch/p2_hooks.py::make_worker_class`
# 的 `P2OffloadingWorker.__init__`，唯一的改动是 `rows = row_counts[idx]`
# （030 的那份是 `num_cpu_blocks`，因为它的行数是外面算好传进来的 —— 逻辑相同）。
#
# ★ 与 P1（`shadow-pkg/patches/files/offload_dsv41/cpu_npu.py`）的关系：
#   基类与分配函数**都取自 P1 的挂载版**（`_allocate_npu_offload_cpu_tensor`
#   = `aclrtHostRegister(MAPPED)` + `ret=0` 日志）⇒ P1 的行为一行没改，
#   只是"每个张量要几行"这一个参数由 L1 给出。

from __future__ import annotations

import os
import time

import torch
from vllm.logger import logger
from vllm.utils.platform_utils import is_pin_memory_available

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.cpu_npu import (
    NPUOffloadingWorker,
    SingleDirectionNPUOffloadingHandler,
    _allocate_npu_offload_cpu_tensor,
)

# P1 的后端选择变量（pageable / pinned / registered）；只为把它的那行日志复现出来。
_HOST_MEM_MODE = os.environ.get("NPU_OFFLOAD_HOST_MEM", "registered").strip().lower()


def _log(msg: str) -> None:
    if os.environ.get("P2_POOL_LOG", "1") == "1":
        print("[P2_poolsizing] [K_l1_8card] " + msg, flush=True)


class P2OffloadingWorker(NPUOffloadingWorker):
    """`NPUOffloadingWorker` 的"每张张量按行数分配"版本。"""

    def __init__(
        self,
        kv_caches,
        blocks_per_chunk: int,
        num_cpu_blocks: int,
        row_counts=None,
    ) -> None:
        pin_memory = is_pin_memory_available()
        n_tensors = len(kv_caches.tensors)
        if row_counts is None:
            row_counts = [int(num_cpu_blocks)] * n_tensors
        row_counts = [int(r) for r in row_counts]
        assert len(row_counts) == n_tensors, (
            f"[K_l1_8card] row_counts 长度 {len(row_counts)} != 张量数 {n_tensors}"
        )
        logger.info("Allocating %d CPU tensors...", n_tensors)
        logger.info(
            "[P1_pinned] CPU pool backend = %s (NPU_OFFLOAD_HOST_MEM)", _HOST_MEM_MODE
        )
        _log(
            "③ worker 按行分配：num_cpu_blocks=%d rows=%s"
            % (int(num_cpu_blocks), row_counts)
        )

        npu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        total = 0
        for idx, kv_cache_tensor in enumerate(kv_caches.tensors):
            npu_page_size_bytes = kv_cache_tensor.page_size_bytes
            npu_tensor = kv_cache_tensor.tensor
            if npu_tensor.dtype != torch.int8 or npu_tensor.ndim != 2:
                raise ValueError(
                    "Canonical NPU KV cache tensors must be two-dimensional "
                    f"int8 views, got shape={tuple(npu_tensor.shape)}, "
                    f"dtype={npu_tensor.dtype}"
                )
            if npu_tensor.shape[1] != npu_page_size_bytes:
                raise ValueError(
                    "Canonical NPU KV cache page size mismatch: "
                    f"shape[1]={npu_tensor.shape[1]}, "
                    f"page_size_bytes={npu_page_size_bytes}"
                )
            cpu_page_size_bytes = npu_page_size_bytes * blocks_per_chunk
            rows = row_counts[idx]

            start_time = time.monotonic()
            cpu_tensor = _allocate_npu_offload_cpu_tensor(rows, cpu_page_size_bytes)
            total += rows * cpu_page_size_bytes
            _log(
                "③   tensor[%d] rows=%d x page=%d (物理 %.3f GiB, %.3f s)"
                % (
                    idx,
                    rows,
                    cpu_page_size_bytes,
                    rows * cpu_page_size_bytes / (1 << 30),
                    time.monotonic() - start_time,
                )
            )
            npu_tensors.append(npu_tensor)
            cpu_tensors.append(cpu_tensor)

        _log("③ worker 物理池合计 = %d B (%.3f GiB)" % (total, total / (1 << 30)))
        print("[P2_poolsizing] P2_WORKER_HOST_BYTES=%d" % total, flush=True)

        self._store_handler = SingleDirectionNPUOffloadingHandler(
            npu_tensors=npu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            npu_to_cpu=True,
        )
        self._load_handler = SingleDirectionNPUOffloadingHandler(
            npu_tensors=npu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            npu_to_cpu=False,
        )
