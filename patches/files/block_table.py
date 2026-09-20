import os

import numpy as np
import torch
from vllm.distributed import get_dcp_group
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    KVCacheSpecKind,
    get_kv_cache_spec_kind,
)
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend.core.circular_buffer import is_circular_spec
from vllm_ascend.distributed.utils import get_decode_context_model_parallel_world_size
from vllm_ascend.ops.triton.compute_slot_mapping import (
    _compute_slot_mapping_kernel,
    _next_power_of_2,
)

logger = init_logger(__name__)


# =============================================================================
# [V41-SLOT-MAP-FUSED] 12 次 slot-mapping 启动 → 1 次
#
# 背景（实测，见 lite-runs/single-chip/REPORT-single-chip-lite.md）：
#   decode 稳态每步 `_compute_slot_mapping_kernel` 启动 **12 次 = KV cache group 数**，
#   单次 device duration 只有 2.5–3.2 µs，但每次启动要付 ~65–70 µs 的 host/排队代价
#   ⇒ 每步约 0.8 ms 的 host 串行时间，且这 12 次都发生在同一步的同一位置。
#   把 12 个 group 折成 1 次二维 grid 启动，host 时间 1.774 → 1.048 ms/step（−41%）。
#
# 门控 `V41_SLOT_MAP_FUSED`：
#   0 / off / 空（默认）  走上游逐组路径，行为与 stock 完全一致
#   1 / on               走融合路径；任何前置条件不满足时**显式回落**逐组路径并告警一次
#   verify               两条路径都跑，逐元素比对（capture 期间不做比对），不一致抛错
#
# 与上游语义的对应关系：本 kernel 就是 `_compute_slot_mapping_kernel` 的
# 「TOTAL_CP_WORLD_SIZE == 1」分支，只是把 group 变成 grid 的第 0 维、
# 把 per-group 的 (block_table 指针/stride/block_size/slot_mapping 指针) 改成从 device 小张量读。
# 因此 dcp_world_size > 1、mamba 组、circular 组都会回落到原路径（见 `_v41_fused_tensors`）。
# =============================================================================
_V41_SLOT_MAP_FUSED_ENV = "V41_SLOT_MAP_FUSED"


def _v41_slot_map_fused_mode() -> str:
    raw = os.environ.get(_V41_SLOT_MAP_FUSED_ENV, "0").strip().lower()
    if raw in ("", "0", "off", "false", "no"):
        return "off"
    if raw in ("verify", "check"):
        return "verify"
    if raw in ("1", "on", "true", "yes"):
        return "on"
    return "off"


def _v41_slot_map_is_capturing() -> bool:
    """是否正在 ACLGraph capture —— capture 期间**绝不能**做 D2H/同步。"""
    try:
        from vllm.forward_context import get_forward_context

        if getattr(get_forward_context(), "capturing", False):
            return True
    except Exception:  # noqa: BLE001 - 不在 forward context 里时 get_forward_context 会 assert
        pass
    try:
        return bool(torch.npu.is_current_stream_capturing())
    except Exception:  # noqa: BLE001 - 某些 torch_npu 版本没有这个 API
        return False


def _v41_make_u64_ptr_tensor(ptrs: list[int], device: torch.device) -> torch.Tensor:
    """指针表：与上游 `_make_ptr_tensor` 一致（uint64，覆盖全部地址空间）。"""
    return torch.tensor(ptrs, dtype=torch.uint64, device=device)


@triton.jit(do_not_specialize=["num_tokens", "max_num_tokens"])
def _compute_slot_mappings_multi_kernel(
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,  # [num_reqs + 1], int32
    positions_ptr,  # [num_tokens], int64
    block_table_ptrs,  # [num_groups], uint64
    block_table_strides,  # [num_groups], int64
    block_sizes,  # [num_groups], int32 (logical block size)
    slot_mapping_ptrs,  # [num_groups], uint64
    TILE_BLOCK_SIZE: tl.constexpr,
    BLOCK_TABLE_WINDOW_SIZE: tl.constexpr,
    PAD_ID: tl.constexpr,
):
    group = tl.program_id(0)
    req_idx = tl.program_id(1)
    slot_mapping_ptr = tl.cast(tl.load(slot_mapping_ptrs + group), tl.pointer_type(tl.int32))

    if req_idx == tl.num_programs(1) - 1:
        # Pad remaining slots for CUDA graph compatibility.
        for i in range(num_tokens, max_num_tokens, TILE_BLOCK_SIZE):
            offsets = i + tl.arange(0, TILE_BLOCK_SIZE)
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_ID,
                mask=offsets < max_num_tokens,
            )
        return

    block_table_ptr = tl.cast(tl.load(block_table_ptrs + group), tl.pointer_type(tl.int32))
    block_table_stride = tl.load(block_table_strides + group)
    block_size = tl.load(block_sizes + group)

    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)

    row_offset = req_idx * block_table_stride
    block_table_offsets = tl.arange(0, BLOCK_TABLE_WINDOW_SIZE)
    for i in range(start_idx, end_idx, TILE_BLOCK_SIZE):
        offsets = i + tl.arange(0, TILE_BLOCK_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0).to(tl.int32)
        block_indices = pos // block_size
        slot_offsets = pos - block_indices * block_size
        INT32_MAX = 2147483647
        valid_block_indices = tl.where(mask, block_indices, INT32_MAX)
        block_idx_base = tl.min(valid_block_indices, axis=0)
        block_table_window_offsets = block_idx_base + block_table_offsets
        block_table_window = tl.load(
            block_table_ptr + row_offset + block_table_window_offsets,
            mask=block_table_window_offsets < block_table_stride,
            other=0,
        ).to(tl.float32)
        relative_block_indices = tl.where(mask, block_indices - block_idx_base, 0)
        block_numbers = tl.gather(block_table_window, relative_block_indices, 0).to(tl.int32)
        slot_ids = block_numbers * block_size + slot_offsets
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)


class BlockTable:
    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kernel_sizes: list[int] | None = None,
        cp_kv_cache_interleave_size: int = 1,
        num_speculative_tokens: int = 0,
        kv_cache_group: KVCacheGroupSpec = None,
    ):
        self.max_num_reqs = max_num_reqs
        self.dcp_world_size = get_dcp_group().world_size
        self.dcp_rank = get_dcp_group().rank_in_group
        is_mamba_group = (
            kv_cache_group is not None
            and hasattr(kv_cache_group, "kv_cache_spec")
            and get_kv_cache_spec_kind(kv_cache_group.kv_cache_spec) == KVCacheSpecKind.MAMBA
        )
        # The KV cache spec already provides the per-rank table capacity.
        # Mamba state is replicated across DCP ranks, not sharded then expanded.
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device
        self.physical_block_size = block_size
        self.is_mamba_group = is_mamba_group
        self.is_circular_group = kv_cache_group is not None and is_circular_spec(kv_cache_group.kv_cache_spec)

        # If kernel_sizes is None or [0], use physical block size (no splitting)
        if kernel_sizes is None or kernel_sizes == [0]:
            self.block_size = block_size
            self.logical_block_size = block_size
            self.blocks_per_phys_block = 1
            self.use_hybrid_blocks = False
        else:
            # Find the first kernel size that divides physical_block_size evenly
            selected_kernel_size = None
            for kernel_size in kernel_sizes:
                if kernel_size > 0 and self.physical_block_size % kernel_size == 0:
                    selected_kernel_size = kernel_size
                    break

            if selected_kernel_size is None:
                raise ValueError(
                    f"None of the kernel sizes {kernel_sizes} can divide "
                    f"physical block size {self.physical_block_size} evenly"
                )

            self.block_size = selected_kernel_size
            self.logical_block_size = selected_kernel_size
            self.blocks_per_phys_block = self.physical_block_size // self.logical_block_size
            if self.blocks_per_phys_block > 1:
                self.use_hybrid_blocks = True
            else:
                self.use_hybrid_blocks = False

        if self.use_hybrid_blocks:
            logical_table_size = max_num_blocks_per_req * self.blocks_per_phys_block
        else:
            logical_table_size = max_num_blocks_per_req

        duplicate_size = 1
        if self.dcp_world_size > 1:
            duplicate_size += num_speculative_tokens
        self.block_table = self._make_buffer(max_num_reqs * duplicate_size, logical_table_size, dtype=torch.int32)
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)
        # MTP slot preparation appends up to num_speculative_tokens - 1
        # draft positions for every request beyond the scheduler token limit.
        num_mtp_draft_slots = max(num_speculative_tokens - 1, 0) * self.max_num_reqs
        self.slot_mapping = self._make_buffer(
            self.max_num_batched_tokens + num_mtp_draft_slots,
            dtype=torch.int32,
        )

        self.kernel_sizes = kernel_sizes
        self.cp_kv_cache_interleave_size = cp_kv_cache_interleave_size

    def append_row(
        self,
        block_ids,
        row_idx: int,
    ) -> None:
        if not block_ids:
            return
        block_ids = np.array(block_ids)
        if self.use_hybrid_blocks:
            block_ids = self._convert_physical_to_logical_blocks(block_ids)

        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]

        self.block_table.np[row_idx, start : start + num_blocks] = block_ids
        self.num_blocks_per_row[row_idx] += num_blocks

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    def clear_row(self, row_idx: int) -> None:
        num_blocks = self.num_blocks_per_row[row_idx]
        if num_blocks > 0:
            self.block_table.np[row_idx, :num_blocks] = 0
        self.num_blocks_per_row[row_idx] = 0

    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[src]
        self.block_table.np[tgt, :num_blocks] = self.block_table.np[src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks

    def swap_row(self, src: int, tgt: int) -> None:
        num_blocks_src = self.num_blocks_per_row[src]
        num_blocks_tgt = self.num_blocks_per_row[tgt]
        self.num_blocks_per_row[src] = num_blocks_tgt
        self.num_blocks_per_row[tgt] = num_blocks_src

        self.block_table.np[[src, tgt]] = self.block_table.np[[tgt, src]]

    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        if self.is_circular_group:
            self.slot_mapping.gpu.fill_(PAD_SLOT_ID)
            return
        num_tokens = positions.shape[0]
        total_cp_world_size = self.dcp_world_size
        total_cp_rank = self.dcp_rank
        if self.dcp_world_size > 1:
            req_indices = torch.repeat_interleave(
                torch.arange(num_reqs, dtype=torch.int32, device=query_start_loc.device),
                query_start_loc[1:] - query_start_loc[:-1],
                output_size=num_tokens,
            )
            self._compute_dcp_slot_mapping(req_indices, positions)
        else:
            TILE_BLOCK_SIZE = 1024
            kernel_kwargs = {
                "KV_CACHE_BLOCK_SIZE": self.physical_block_size,
                "BLOCKS_PER_KV_BLOCK": self.blocks_per_phys_block,
                "TOTAL_CP_WORLD_SIZE": total_cp_world_size,
                "TOTAL_CP_RANK": total_cp_rank,
                "CP_KV_CACHE_INTERLEAVE_SIZE": self.cp_kv_cache_interleave_size,
                "PAD_ID": PAD_SLOT_ID,
                "TILE_BLOCK_SIZE": TILE_BLOCK_SIZE,
                "BLOCK_TABLE_WINDOW_SIZE": _next_power_of_2(cdiv(TILE_BLOCK_SIZE, self.block_size) + 1),
            }

            _compute_slot_mapping_kernel[(num_reqs + 1,)](
                num_tokens,
                self.max_num_batched_tokens,
                query_start_loc,
                positions,
                self.block_table.gpu,
                self.block_table.gpu.stride(0),
                self.block_size,
                self.slot_mapping.gpu,
                **kernel_kwargs,
            )

    def compute_slot_mapping_draft(
        self,
        req_indices: np.ndarray | torch.Tensor,
        positions: np.ndarray | torch.Tensor,
    ) -> None:
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size`
        # here because M (max_model_len) is not necessarily divisible by
        # block_size.

        if self.is_circular_group:
            self.slot_mapping.gpu.fill_(PAD_SLOT_ID)
            return
        if self.dcp_world_size > 1:
            if not isinstance(req_indices, torch.Tensor):
                req_indices = torch.from_numpy(req_indices)
            if not isinstance(positions, torch.Tensor):
                positions = torch.from_numpy(positions)
            self._compute_dcp_slot_mapping(req_indices, positions)
        else:
            if isinstance(req_indices, torch.Tensor):
                if req_indices.device.type != "cpu":
                    raise ValueError("Device tensor inputs are only supported for CP draft slot mapping.")
                req_indices = req_indices.numpy()
            if isinstance(positions, torch.Tensor):
                if positions.device.type != "cpu":
                    raise ValueError("Device tensor inputs are only supported for CP draft slot mapping.")
                positions = positions.numpy()
            assert self.kernel_sizes is not None
            assert self.block_size == self.kernel_sizes[0]
            # IMPORTANT: In hybrid mode, positions are in logical block space,
            # but we need to map them to the correct logical block table indices
            logical_block_idx = positions // self.block_size

            # Account for the expanded logical table
            # (always needed with unified tensor)
            # Each physical block is split into multiple logical blocks
            # The logical table has been expanded to accommodate this
            block_table_indices = (
                req_indices * self.max_num_blocks_per_req * self.blocks_per_phys_block + logical_block_idx
            )

            block_offsets = positions % self.block_size
            block_numbers = self.block_table.np.ravel()[block_table_indices]
            np.add(
                block_numbers * self.block_size,
                block_offsets,
                out=self.slot_mapping.np[: req_indices.shape[0]],
            )
            self.slot_mapping.copy_to_gpu(req_indices.shape[0])

    def _compute_dcp_slot_mapping(
        self,
        req_indices: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        # Note(hc): The DCP implement store kvcache with an interleave
        # style, the kvcache for the token whose token_idx is i is
        # always stored on the GPU whose dcp_rank equals the interleaved shard:

        # Use a "virtual block" which equals to world_size * block_size
        # for block_table_indices calculation.
        # virtual_block_size = self.block_size * self.dcp_world_size

        # IMPORTANT: In hybrid mode, positions are in logical block space,
        # but we need to map them to the correct logical block table indices
        # logical_block_idx = positions // virtual_block_size

        total_cp_world_size = self.dcp_world_size
        virtual_physical_block_size = self.physical_block_size * total_cp_world_size
        physical_block_idx = positions // virtual_physical_block_size
        virtual_block_offsets = positions % virtual_physical_block_size

        self.current_rank = self.dcp_rank
        mask = virtual_block_offsets // self.cp_kv_cache_interleave_size % total_cp_world_size == self.current_rank
        local_physical_offsets = (
            virtual_block_offsets
            // (total_cp_world_size * self.cp_kv_cache_interleave_size)
            * self.cp_kv_cache_interleave_size
            + virtual_block_offsets % self.cp_kv_cache_interleave_size
        )
        logical_block_idx = physical_block_idx * self.blocks_per_phys_block + (
            local_physical_offsets // self.block_size
        )

        block_table_indices = req_indices * self.max_num_blocks_per_req * self.blocks_per_phys_block + logical_block_idx

        block_offsets = local_physical_offsets % self.block_size

        if block_table_indices.device.type != "cpu":
            block_numbers = self.block_table.gpu.flatten()[block_table_indices]
            slot_mapping = block_numbers * self.block_size + block_offsets
            self.slot_mapping.gpu[: req_indices.shape[0]] = torch.where(mask, slot_mapping, -1)
        else:
            block_numbers = self.block_table.cpu.flatten()[block_table_indices]
            slot_mapping = block_numbers * self.block_size + block_offsets
            self.slot_mapping.cpu[: req_indices.shape[0]] = torch.where(mask, slot_mapping, -1)

    def commit_block_table(self, num_reqs: int) -> None:
        self.block_table.copy_to_gpu(num_reqs)

    def clear(self) -> None:
        self.block_table.fill_(0)
        self.block_table.cpu.fill_(0)

    def _convert_physical_to_logical_blocks(self, physical_blocks: np.ndarray) -> np.ndarray:
        """Convert physical block IDs to logical block IDs."""
        if not self.use_hybrid_blocks:
            return physical_blocks

        # Create logical block IDs by splitting each physical block
        logical_blocks: list[int] = []
        for phys_block in physical_blocks:
            # Convert physical block to multiple logical blocks
            # Physical block 1 becomes logical blocks
            # [1*split_ratio, 1*split_ratio+1, ...]
            # But we need to account for the fact that block 0 is special
            base_logical = phys_block * self.blocks_per_phys_block
            logical_blocks.extend(range(base_logical, base_logical + self.blocks_per_phys_block))

        return np.array(logical_blocks, dtype=np.int32)

    def get_device_tensor(self, num_reqs: int | None = None) -> torch.Tensor:
        """Returns the device tensor of the block table."""
        if num_reqs is not None:
            return self.block_table.gpu[:num_reqs]
        return self.block_table.gpu

    def get_cpu_tensor(self) -> torch.Tensor:
        """Returns the CPU tensor of the block table."""
        return self.block_table.cpu

    def get_numpy_array(self) -> np.ndarray:
        """Returns the numpy array of the block table."""
        return self.block_table.np

    def _make_buffer(self, *size: int | torch.SymInt, dtype: torch.dtype) -> CpuGpuBuffer:
        return CpuGpuBuffer(*size, dtype=dtype, device=self.device, pin_memory=self.pin_memory)


class MultiGroupBlockTable:
    """The BlockTables for each KV cache group."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        block_sizes: list[int],
        num_speculative_tokens: int = 0,
        max_num_blocks: list[int] | None = None,
        kernel_sizes: list[list[int]] | None = None,
        cp_kv_cache_interleave_size: int = 1,
        kv_cache_groups: KVCacheGroupSpec = None,
    ) -> None:
        if kernel_sizes is None:
            kernel_sizes = [[0]] * len(block_sizes)
        # Ensure kernel_sizes matches block_sizes length
        elif len(kernel_sizes) == 1 and len(block_sizes) > 1:
            kernel_sizes = kernel_sizes * len(block_sizes)
        elif len(kernel_sizes) != len(block_sizes):
            raise ValueError(
                f"kernel_sizes length ({len(kernel_sizes)}) must match block_sizes length ({len(block_sizes)})"
            )

        if max_num_blocks is None:
            # Note(hc): each dcp rank only store
            # (max_model_len//dcp_world_size) tokens in kvcache,
            # so the block_size which used for calc max_num_blocks_per_req
            # must be multiplied by dcp_world_size.
            dcp_world_size = get_decode_context_model_parallel_world_size()
            max_num_blocks = [cdiv(max_model_len, block_size * dcp_world_size) for block_size in block_sizes]

        if len(max_num_blocks) != len(block_sizes):
            raise ValueError(
                f"max_num_blocks length ({len(max_num_blocks)}) must match block_sizes length ({len(block_sizes)})"
            )

        # Use zip to pair block_sizes with kernel_sizes one-to-one
        if kv_cache_groups is not None:
            self.block_tables = [
                BlockTable(
                    block_size,
                    max_num_reqs,
                    max_num_blocks_per_req,
                    max_num_batched_tokens,
                    pin_memory,
                    device,
                    kernel_size_list,
                    cp_kv_cache_interleave_size,
                    num_speculative_tokens,
                    kv_cache_group,
                )
                for block_size, kernel_size_list, max_num_blocks_per_req, kv_cache_group in zip(
                    block_sizes, kernel_sizes, max_num_blocks, kv_cache_groups
                )
            ]
        else:
            self.block_tables = [
                BlockTable(
                    block_size,
                    max_num_reqs,
                    max_num_blocks_per_req,
                    max_num_batched_tokens,
                    pin_memory,
                    device,
                    kernel_size_list,
                    cp_kv_cache_interleave_size,
                    num_speculative_tokens,
                )
                for block_size, kernel_size_list, max_num_blocks_per_req in zip(
                    block_sizes, kernel_sizes, max_num_blocks
                )
            ]

        # [V41-SLOT-MAP-FUSED] device 侧指针表缓存（只在 data_ptr 变化时重建，无 D2H）。
        self._v41_fused_state: dict | None = None
        self._v41_fused_warned = False
        # 融合 grid 覆盖的 group 下标（None = 本实例还没算过）
        self._v41_fused_group_idx: list[int] | None = None

    # ------------------------------------------------------------------
    # [V41-SLOT-MAP-FUSED] 前置条件 / 指针表 / 启动
    # ------------------------------------------------------------------
    def _v41_fused_precheck(
        self,
        positions_compressed_list: list[np.ndarray] | None,
        req_indices_compressed_list: list[np.ndarray] | None,
    ) -> str | None:
        """返回 None = 可以融合；否则返回**不可融合的原因**（调用方回落并告警一次）。

        每一条对应一个真实的语义差异，宁可回落也不静默算错。
        """
        if positions_compressed_list or req_indices_compressed_list:
            return "draft/compressed slot-mapping 路径（positions_compressed_list）"
        if not self.block_tables:
            return "没有 block table"
        if self.block_tables[0].device.type == "cpu":
            return "device 是 CPU"
        for i, bt in enumerate(self.block_tables):
            if bt.is_mamba_group:
                continue
            if bt.dcp_world_size > 1:
                return f"group{i} dcp_world_size={bt.dcp_world_size} > 1（走 _compute_dcp_slot_mapping）"
            if bt.blocks_per_phys_block != 1:
                return f"group{i} blocks_per_phys_block={bt.blocks_per_phys_block} != 1（物理块拆分）"
            if bt.block_table.gpu.dtype != torch.int32:
                return f"group{i} block_table dtype={bt.block_table.gpu.dtype}"
            if bt.slot_mapping.gpu.dtype != torch.int32:
                return f"group{i} slot_mapping dtype={bt.slot_mapping.gpu.dtype}"
            if bt.block_table.gpu.stride(0) != bt.block_table.gpu.shape[1]:
                return f"group{i} block_table 非连续（stride0={bt.block_table.gpu.stride(0)}）"
        return None

    def _v41_fused_groups(self) -> list[int]:
        """需要走 kernel 启动的 group 下标（mamba 组原本就跳过；circular 组是 fill_）。"""
        if self._v41_fused_group_idx is None:
            self._v41_fused_group_idx = [
                i
                for i, bt in enumerate(self.block_tables)
                if not bt.is_mamba_group and not bt.is_circular_group
            ]
        return self._v41_fused_group_idx

    def _v41_fused_tensors(self, device: torch.device) -> dict | None:
        """构造/复用 device 侧指针表。指针变了（buffer 重建）就重建；否则复用，无 D2H。"""
        idx = self._v41_fused_groups()
        if not idx:
            return None
        tables = [self.block_tables[i] for i in idx]
        sig = (
            tuple(t.block_table.gpu.data_ptr() for t in tables)
            + tuple(t.slot_mapping.gpu.data_ptr() for t in tables)
        )
        state = self._v41_fused_state
        if state is not None and state["sig"] == sig:
            return state
        state = {
            "sig": sig,
            "idx": idx,
            "block_table_ptrs": _v41_make_u64_ptr_tensor(
                [t.block_table.gpu.data_ptr() for t in tables], device
            ),
            "slot_mapping_ptrs": _v41_make_u64_ptr_tensor(
                [t.slot_mapping.gpu.data_ptr() for t in tables], device
            ),
            "block_table_strides": torch.tensor(
                [t.block_table.gpu.stride(0) for t in tables], dtype=torch.int64, device=device
            ),
            "block_sizes": torch.tensor([t.block_size for t in tables], dtype=torch.int32, device=device),
            "window": max(
                _next_power_of_2(cdiv(1024, t.block_size) + 1) for t in tables
            ),
        }
        self._v41_fused_state = state
        return state

    def _v41_launch_fused(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> bool:
        """成功启动返回 True；不可用时返回 False（调用方回落）。"""
        state = self._v41_fused_tensors(positions.device)
        if state is None:
            return False
        n_groups = len(state["idx"])
        _compute_slot_mappings_multi_kernel[(n_groups, num_reqs + 1)](
            positions.shape[0],
            self.block_tables[state["idx"][0]].max_num_batched_tokens,
            query_start_loc,
            positions,
            state["block_table_ptrs"],
            state["block_table_strides"],
            state["block_sizes"],
            state["slot_mapping_ptrs"],
            TILE_BLOCK_SIZE=1024,
            BLOCK_TABLE_WINDOW_SIZE=state["window"],
            PAD_ID=PAD_SLOT_ID,
        )
        return True

    def _v41_fused_verify(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """verify 模式：融合结果与原逐组路径**逐元素**比对（含 D2H ⇒ 绝不进 capture）。

        做法：先把融合结果挪到 scratch（device→device），再用原路径重算一遍 slot_mapping，
        最后 `torch.equal` 逐组比对。不一致直接抛错（不静默放过）。
        """
        state = self._v41_fused_state
        groups = state["idx"]
        tables = [self.block_tables[i] for i in groups]
        scratch = state.get("scratch")
        if scratch is None or len(scratch) != len(tables):
            scratch = [torch.empty_like(t.slot_mapping.gpu) for t in tables]
            state["scratch"] = scratch
        for s, t in zip(scratch, tables):
            s.copy_(t.slot_mapping.gpu)

        # 原路径重算（覆盖 slot_mapping；值应当与融合路径完全相同）
        for bt in self.block_tables:
            if bt.is_mamba_group:
                continue
            bt.compute_slot_mapping(num_reqs, query_start_loc, positions)

        bad = []
        for g, s, t in zip(groups, scratch, tables):
            if not bool(torch.equal(s, t.slot_mapping.gpu)):
                n_diff = int((s != t.slot_mapping.gpu).sum().item())
                bad.append(f"group{g}:{n_diff} 个元素")
        if bad:
            raise RuntimeError(
                "[V41_SLOT_MAP_FUSED=verify] 融合结果与原逐组路径不一致："
                + ", ".join(bad)
                + "；请把 V41_SLOT_MAP_FUSED 置 0 并上报（这是静默算错的前兆）"
            )
        self._v41_verify_ok = True

    def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.append_row(block_ids[i], row_idx)

    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.add_row(block_ids[i], row_idx)

    def clear_row(self, row_idx: int) -> None:
        for block_table in self.block_tables:
            block_table.clear_row(row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.swap_row(src, tgt)

    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        positions_compressed_list: list[np.ndarray] | None = None,
        req_indices_compressed_list: list[np.ndarray] | None = None,
    ) -> None:
        # [V41-SLOT-MAP-FUSED] 12 次逐组启动 → 1 次二维 grid 启动（默认关闭）
        mode = _v41_slot_map_fused_mode()
        if mode != "off" and self._v41_try_fused(
            mode, num_reqs, query_start_loc, positions,
            positions_compressed_list, req_indices_compressed_list,
        ):
            return
        for i, block_table in enumerate(self.block_tables):
            if block_table.is_mamba_group:
                continue
            if positions_compressed_list and req_indices_compressed_list:
                block_table.compute_slot_mapping_draft(req_indices_compressed_list[i], positions_compressed_list[i])
            else:
                block_table.compute_slot_mapping(num_reqs, query_start_loc, positions)

    def _v41_try_fused(
        self,
        mode: str,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        positions_compressed_list: list[np.ndarray] | None,
        req_indices_compressed_list: list[np.ndarray] | None,
    ) -> bool:
        """尝试融合路径。返回 True = 已完成本步 slot-mapping；False = 调用方回落原路径。"""
        reason = self._v41_fused_precheck(positions_compressed_list, req_indices_compressed_list)
        if reason is None and query_start_loc.dtype not in (torch.int32, torch.int64):
            reason = f"query_start_loc dtype={query_start_loc.dtype}"
        if reason is None and positions.dtype not in (torch.int32, torch.int64):
            reason = f"positions dtype={positions.dtype}"
        capturing = _v41_slot_map_is_capturing()
        if reason is not None or (mode == "verify" and capturing):
            if capturing and reason is None:
                reason = "capture 期间不做 verify 比对（避免 D2H）"
            if not self._v41_fused_warned:
                self._v41_fused_warned = True
                logger.warning(
                    "[V41_SLOT_MAP_FUSED=%s] 回落到逐组路径：%s", mode, reason
                )
            return False

        # circular 组在原路径里是 `slot_mapping.gpu.fill_(PAD_SLOT_ID)`：
        # 语义必须保留（它们不进融合 grid，但不许被漏掉）。
        for bt in self.block_tables:
            if bt.is_mamba_group:
                continue
            if bt.is_circular_group:
                bt.slot_mapping.gpu.fill_(PAD_SLOT_ID)

        if not self._v41_launch_fused(num_reqs, query_start_loc, positions):
            if not self._v41_fused_warned:
                self._v41_fused_warned = True
                logger.warning(
                    "[V41_SLOT_MAP_FUSED=%s] 没有可融合的 group，回落到逐组路径", mode
                )
            return False
        if mode == "verify":
            self._v41_fused_verify(num_reqs, query_start_loc, positions)
        return True

    def compute_slot_mapping_draft(
        self,
        req_indices: np.ndarray | torch.Tensor,
        positions: np.ndarray | torch.Tensor,
        positions_compressed_list: list[np.ndarray] | None = None,
        req_indices_compressed_list: list[np.ndarray] | None = None,
    ) -> None:
        for i, block_table in enumerate(self.block_tables):
            if block_table.is_mamba_group:
                continue
            if positions_compressed_list and req_indices_compressed_list:
                block_table.compute_slot_mapping_draft(req_indices_compressed_list[i], positions_compressed_list[i])
            else:
                block_table.compute_slot_mapping_draft(req_indices, positions)

    def commit_block_table(self, num_reqs: int) -> None:
        for block_table in self.block_tables:
            block_table.commit_block_table(num_reqs)

    def clear(self) -> None:
        for block_table in self.block_tables:
            block_table.clear()

    def __getitem__(self, idx: int) -> "BlockTable":
        """Returns the BlockTable for the i-th KV cache group."""
        return self.block_tables[idx]
