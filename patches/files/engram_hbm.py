# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Node-local BF16/INT8 Engram storage, independent of model TP.

[dsv41-ws-opt] Drop-in superset of ``patches/engram_hbm_int8_host.py`` (P18)
with two default-off device-memory knobs:

* ``V41_ENGRAM_REUSE_EP_GROUP=1``: single-node only.  The Engram node-local
  group is the EP group; reuse its HCCL communicator instead of creating a
  second one (one new HCCL comm allocates non-torch device memory outside the
  torch caching allocator).
* ``V41_ENGRAM_PG_BUFFER_MB=200``: multi-node fallback.  Create the Engram
  node-local HCCL group with an explicit ``hccl_buffer_size`` instead of
  inheriting the global ``HCCL_BUFFSIZE`` env (serve_v2.sh sets 1024 MB for
  the MoE path, which is far larger than the Engram all-to-all messages).

Unset both variables => behaviour is byte-identical to the P18 host patch.
"""

import json
import logging
import os
import socket
from collections import OrderedDict
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn

from vllm_ascend.ops.triton.engram_int8 import gather_dequantize_engram_int8

try:  # [ENGRAM-JIT-PLAN]
    from .engram_plan_kernel import (  # type: ignore
        PLAN_JIT as _ENGRAM_PLAN_JIT,
        engram_plan_kernel as _engram_plan_kernel,
        flatten_ids as _engram_plan_flatten,
        selftest as _engram_plan_selftest,
    )
except Exception:  # 无包上下文 / sidecar 缺失时静默关闭
    try:
        from engram_plan_kernel import (  # type: ignore
            PLAN_JIT as _ENGRAM_PLAN_JIT,
            engram_plan_kernel as _engram_plan_kernel,
            flatten_ids as _engram_plan_flatten,
            selftest as _engram_plan_selftest,
        )
    except Exception:
        _ENGRAM_PLAN_JIT = False
        _engram_plan_kernel = None
        _engram_plan_flatten = None
        _engram_plan_selftest = None

logger = logging.getLogger(__name__)

# ==== [route-probe] route 内部相位计时（默认关，V41_ENGRAM_ROUTE_PROBE=1 打开）====
import time as _rp_time


class _RouteProbe:
    def __init__(self):
        self.on = bool(os.environ.get("V41_ENGRAM_ROUTE_PROBE", ""))
        self.every = int(os.environ.get("V41_ENGRAM_ROUTE_PROBE_EVERY", "20") or 20)
        self.rank0 = True
        try:
            self.rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
        except Exception:
            pass
        self.n = 0
        self.acc = {}
        self.t0 = None

    def start(self):
        if not self.on:
            return
        self.t0 = _rp_time.perf_counter()

    def mark(self, key):
        if not self.on or self.t0 is None:
            return
        now = _rp_time.perf_counter()
        self.acc[key] = self.acc.get(key, 0.0) + (now - self.t0) * 1000.0
        self.t0 = now

    def tick(self):
        if not self.on:
            return
        self.n += 1
        self.t0 = None
        if self.n % self.every:
            return
        if self.rank0:
            body = " ".join(f"{k}={v / self.every:.3f}" for k, v in sorted(self.acc.items()))
            print(f"[route-probe] n={self.n} {body}", flush=True)
        self.acc = {}


_RP = _RouteProbe()


def _fill_dp_equivalence(q, parallel, single_node):
    """[local-metadata] 判断"组内每个 rank 的 metadata 是否必然相同"。

    DP=1 且单节点、PP=PCP=DCP=1（from_vllm 已断言）时，组内所有 rank 处理
    完全相同的 token 序列，`PagedNgramHistory` 的页面镜像与 `_metadata()`
    的 owner 分组都是同一份 ⇒ metadata 逐个 rank 相同，`all_gather` 只是把
    它复制 q.size 遍。

    DP>1（或多节点）时不成立：不同 DP rank 拿到的是不同请求，必须真通信。
    """
    dp = int(getattr(parallel, "data_parallel_size", 1) or 1)
    q.dp_size = dp
    q.equiv_ranks = bool(dp == 1 and single_node)


def _fill_source_rank(q):
    """[local-owner] 一次性地把"谁是这个组里的 requester"广播给所有 rank。

    现有实现的 requester 判定是 `q.is_source`（在 `_metadata` 里决定谁取真实 ids），
    本地推导优化需要每个 rank 都知道 requester 的**组内 rank**，才能在 all_to_all
    里摆正收发长度。这里只在初始化时做一次 CPU all_gather_object。
    """
    try:
        flags = [None] * q.size
        dist.all_gather_object(flags, int(bool(q.is_source)), group=q.cpu_group)
        ranks = tuple(i for i, f in enumerate(flags) if f)
        q.source_group_ranks = ranks if len(ranks) == 1 else None
    except Exception:
        q.source_group_ranks = None


class _LocalMetaMode:
    """文件驱动的 local-metadata 模式（off/validate/on），支持同起服内 A/B。"""

    PATH = os.environ.get("V41_ENGRAM_LOCAL_METADATA_FILE", "/tmp/v41_engram_localmeta")

    def __init__(self):
        self.mode = os.environ.get("V41_ENGRAM_LOCAL_METADATA", "off") or "off"
        self.mtime = -1.0
        self.last = 0.0

    def get(self):
        now = _rp_time.monotonic()
        if now - self.last < 0.25:
            return self.mode
        self.last = now
        try:
            st = os.stat(self.PATH)
        except OSError:
            return self.mode
        if st.st_mtime == self.mtime:
            return self.mode
        self.mtime = st.st_mtime
        try:
            with open(self.PATH) as fh:
                raw = fh.read().strip()
        except OSError:
            return self.mode
        if raw in ("off", "validate", "on", "gather", "fast", "b2g"):
            if raw != self.mode and _RP.rank0:
                print(f"[local-metadata] mode-change -> {raw}", flush=True)
            self.mode = raw
        return self.mode


_LM = _LocalMetaMode()


class _LocalOwnerMode:
    """文件驱动的 local-owner 模式（off/validate/on）。

    DP=1 时每个 rank 都持有**完整的 hashes**（`prepare_engram` 在每层都算了），
    只是现有实现把非 source rank 的 ids 丢掉（`_metadata` 里的 `q.is_source`）。
    于是 owner 其实可以**本地推导**"别人会发给我的那批 id"，从而跳过
    metadata all_gather 与 ids 的 all_to_all —— 只保留值回传的 all_to_all。
    """

    PATH = os.environ.get("V41_ENGRAM_LOCAL_OWNER_FILE", "/tmp/v41_engram_localowner")

    def __init__(self):
        self.mode = os.environ.get("V41_ENGRAM_LOCAL_OWNER", "off") or "off"
        self.mtime = -1.0
        self.last = 0.0
        self.ok = 0
        self.bad = 0

    def get(self):
        now = _rp_time.monotonic()
        if now - self.last < 0.25:
            return self.mode
        self.last = now
        try:
            st = os.stat(self.PATH)
        except OSError:
            return self.mode
        if st.st_mtime == self.mtime:
            return self.mode
        self.mtime = st.st_mtime
        try:
            with open(self.PATH) as fh:
                raw = fh.read().strip()
        except OSError:
            return self.mode
        if raw in ("off", "validate", "on", "gather", "fast", "b2g"):
            if raw != self.mode and _RP.rank0:
                print(f"[local-owner] mode-change -> {raw}", flush=True)
            self.mode = raw
        return self.mode


_LO = _LocalOwnerMode()


class _RoutePipeMode:
    """[route-pipe] 文件驱动的"两张表流水化"开关（off/on）。

    `_route_local_owner` 原本对两张表依次做 plan→lookup→h2d→a2a→bcast，
    其中 a2a / bcast 都是**阻塞**调用，于是第二张表的 a2a 必须等第一张表的
    a2a+bcast 全部返回才下发。设备在那段时间没有活可干（实测每步 1.4~1.6 ms
    的空档就落在 `Fill/Cast` 与 `hcom_alltoallv_` 之间）。

    打开本开关后改成：先把所有表的 CPU 工作在 host 上做完，再依次以
    `async_op=True` 下发各表的 all_to_all，最后统一 wait + scatter + bcast。
    数值语义与阻塞版逐位一致，只是把集合通信的等待挪到最后。
    """

    PATH = os.environ.get("V41_ROUTE_PIPE_FILE", "/tmp/v41_route_pipe")

    def __init__(self):
        self.mode = os.environ.get("V41_ROUTE_PIPE", "off") or "off"
        self.mtime = -1.0
        self.last = 0.0

    def get(self):
        now = _rp_time.monotonic()
        if now - self.last < 0.25:
            return self.mode
        self.last = now
        try:
            st = os.stat(self.PATH)
        except OSError:
            return self.mode
        if st.st_mtime == self.mtime:
            return self.mode
        self.mtime = st.st_mtime
        try:
            with open(self.PATH) as fh:
                raw = fh.read().strip()
        except OSError:
            return self.mode
        if raw in ("off", "on"):
            if raw != self.mode and _RP.rank0:
                print(f"[route-pipe] mode-change -> {raw}", flush=True)
            self.mode = raw
        return self.mode


_PIPE = _RoutePipeMode()


_OFFLOAD_BUFFER_CACHE_SIZE = 8
_OFFLOAD_BUFFER_BYTES_LIMIT = 512 * 1024 * 1024
_BF16_BYTES = 2


def quantize_engram_rows(rows):
    """Group32 symmetric INT8 with FP32 power-of-two scales and ties-to-even."""
    grouped = rows.float().unflatten(-1, (-1, 32))
    maximum = grouped.abs().amax(-1, keepdim=True)
    scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127)
    # NPU exp2 can return one ULP below an exact power of two, changing
    # ties-to-even codes. ldexp constructs the binary scale exactly.
    exponent = torch.ceil(torch.log2(scale))
    scale = torch.where(torch.isfinite(exponent), torch.ldexp(torch.ones_like(scale), exponent.int()), scale)
    codes = torch.round(grouped / scale).clamp(-127, 127).to(torch.int8).flatten(-2)
    return codes, scale.squeeze(-1)


def dequantize_engram_rows(codes, scale):
    # Keep one FP32 work buffer: in-place scaling avoids the extra FP32 result
    # allocation created by the broadcast multiply expression.
    decoded = codes.float().unflatten(-1, (-1, 32))
    decoded.mul_(scale.unsqueeze(-1))
    return decoded.flatten(-2).bfloat16()


def pack_engram_int8_rows(codes, scale):
    """Pack INT8 codes and FP32 group scales as one row-oriented wire buffer."""
    if codes.dtype != torch.int8 or scale.dtype != torch.float32:
        raise TypeError("Engram INT8 wire packing expects int8 codes and FP32 scales")
    return torch.cat((codes.view(torch.uint8), scale.view(torch.uint8)), dim=-1).contiguous()


def unpack_engram_int8_rows(payload, width):
    """Decode the packed INT8 wire buffer without changing BF16 lookup semantics."""
    groups = width // 32
    expected = width + groups * 4
    if payload.dtype != torch.uint8 or payload.shape[-1] != expected:
        raise ValueError(f"Invalid Engram INT8 wire payload: {tuple(payload.shape)}")
    codes = payload[..., :width].contiguous().view(torch.int8)
    scale = payload[..., width:].contiguous().view(torch.float32).reshape(*payload.shape[:-1], groups)
    return dequantize_engram_rows(codes, scale)


class EngramQueryGroup:
    """One node group shared by all Engram layers; TP leaders submit queries.

    All ranks (including idle DP replicas) must call lookup in the same order.
    Counts and all-to-all split sizes are eager metadata, not graph inputs.
    """

    def __init__(self, group, cpu_group, tp_group, tp_source):
        self.group = group
        self.cpu_group = cpu_group
        self.tp_group = tp_group
        self.tp_source = tp_source
        # HCCL metadata avoids the CPU/Gloo rendezvous; standalone Gloo/MPI
        # probes keep CPU metadata. Callers may override this for rollback.
        backend = str(dist.get_backend(group)).lower()
        self.metadata_on_device = backend not in ("gloo", "mpi")
        self.rank = dist.get_rank(group)
        self.size = dist.get_world_size(group)
        self.is_source = dist.get_rank() == tp_source
        # [local-metadata] DP==1 时组内所有 rank 处理同一批 token ⇒ hashes 与
        # metadata 逐个 rank 相同，all_gather 只是把这个常量复制 8 遍。
        # 由 from_vllm 填；缺省 0 表示未知，走原有 collective。
        self.dp_size = 0
        self.equiv_ranks = False
        self.source_group_ranks = None

    @classmethod
    def from_vllm(cls, parallel):
        # Lazy imports keep the transport usable in standalone distributed probes.
        from vllm.distributed import get_ep_group, get_tp_group

        if (
            parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or not parallel.enable_expert_parallel
        ):
            raise ValueError("Engram HBM sharing requires EP and PP=PCP=DCP=1")
        ep, tp = get_ep_group(), get_tp_group()
        hosts = [None] * ep.world_size
        dist.all_gather_object(hosts, socket.gethostname(), group=ep.cpu_group)
        node_groups = [[ep.ranks[i] for i, host in enumerate(hosts) if host == name] for name in dict.fromkeys(hosts)]
        node_sizes = {len(ranks) for ranks in node_groups}
        if len(node_sizes) != 1:
            raise ValueError(
                "Engram requires equal rank counts on every node: "
                f"{node_groups}"
            )
        # [dsv41-ws-opt] Optional device-memory reduction for the Engram group.
        # Unset both env vars -> identical behaviour to the P18 host patch.
        reuse_ep_group = bool(os.environ.get("V41_ENGRAM_REUSE_EP_GROUP", ""))
        raw_pg_buffer_mb = os.environ.get("V41_ENGRAM_PG_BUFFER_MB", "")
        pg_buffer_mb = 0
        if raw_pg_buffer_mb:
            try:
                pg_buffer_mb = int(raw_pg_buffer_mb)
            except ValueError:
                logger.warning("Ignoring non-integer V41_ENGRAM_PG_BUFFER_MB=%r", raw_pg_buffer_mb)
                pg_buffer_mb = 0
        if reuse_ep_group and len(node_groups) == 1 and list(node_groups[0]) == list(ep.ranks):
            # Single-node fast path: the node-local Engram group is exactly the
            # EP group, so its HCCL communicator can be reused.  This avoids a
            # second HCCL communication domain (observed non-torch increase
            # ~2 GiB at HCCL_BUFFSIZE=1024).
            if not set(tp.ranks).issubset(node_groups[0]):
                raise ValueError("Engram requires each TP group to stay within one node")
            logger.info(
                "Engram ws-opt: reusing EP device group for the node-local Engram group "
                "(ranks=%s); no extra HCCL communicator is created",
                node_groups[0],
            )
            q = cls(ep.device_group, ep.cpu_group, tp.device_group, tp.ranks[0])
            _fill_dp_equivalence(q, parallel, len(node_groups) == 1)
            _fill_source_rank(q)
            return q
        pg_options = None
        if pg_buffer_mb > 0:
            try:
                # Prefer constructing Options directly: create_hccl_pg_options()
                # builds a dict whose "dp" entry eagerly calls
                # calculate_dp_buffer_size() and needs the current vLLM config.
                import torch_npu

                pg_options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
                pg_options.hccl_config = {
                    "group_name": "engram_ws_opt",
                    "hccl_buffer_size": pg_buffer_mb,
                }
            except Exception:
                try:
                    from vllm_ascend.utils import create_hccl_pg_options

                    pg_options = create_hccl_pg_options("engram_ws_opt")
                    hccl_config = dict(getattr(pg_options, "hccl_config", {}) or {})
                    hccl_config["group_name"] = "engram_ws_opt"
                    hccl_config["hccl_buffer_size"] = pg_buffer_mb
                    pg_options.hccl_config = hccl_config
                except Exception:
                    # Fail open to the stock new_group() call (same as P18); the
                    # P15 probe reads worker.py's "Actual usage: ... non-torch".
                    logger.warning(
                        "Engram ws-opt: V41_ENGRAM_PG_BUFFER_MB=%d could not be applied; "
                        "falling back to the stock new_group() call",
                        pg_buffer_mb,
                        exc_info=True,
                    )
                    pg_options = None
            if pg_options is not None:
                logger.info(
                    "Engram ws-opt: creating node-local HCCL group with hccl_buffer_size=%d MB",
                    pg_buffer_mb,
                )
        selected = None
        for ranks in node_groups:
            # Every world rank creates groups in the same order.
            cpu = dist.new_group(ranks, backend="gloo")
            if pg_options is None:
                device = dist.new_group(ranks, backend=dist.get_backend(ep.device_group))
            else:
                device = dist.new_group(ranks, backend=dist.get_backend(ep.device_group), pg_options=pg_options)
            if dist.get_rank() in ranks:
                if not set(tp.ranks).issubset(ranks):
                    raise ValueError("Engram requires each TP group to stay within one node")
                selected = cls(device, cpu, tp.device_group, tp.ranks[0])
        _fill_dp_equivalence(selected, parallel, len(node_groups) == 1)
        _fill_source_rank(selected)
        return selected


class NodeShardedEngram(nn.Module):
    """Contiguous row shards; only BF16 rows cross the node-local fabric."""

    def __init__(self, rows, width, query_group, device=None, storage_format="bf16"):
        super().__init__()
        if storage_format not in ("bf16", "int8", "fp8", "mxfp8"):
            raise ValueError("Engram storage_format must be bf16, int8, fp8, or mxfp8")
        if storage_format in ("int8", "fp8", "mxfp8") and width % 32:
            raise ValueError("INT8 Engram requires a width divisible by 32")
        self.storage_format = storage_format
        self.rows, self.width = rows, width
        self.query_group = query_group
        # Kept opt-in while the reduced-payload protocol is benchmarked.
        self.compressed_int8_wire = False
        # The fused gather/dequant kernel wins even for one row on A3; retain
        # the threshold as a local rollback knob for future kernel changes.
        self.use_triton_int8 = True
        self.triton_int8_min_rows = 1
        # [dsv41-patch] int8 也支持 host 驻留（env V41_ENGRAM_HOST_RESIDENT=1）
        self.host_resident = bool(os.environ.get("V41_ENGRAM_HOST_RESIDENT", "")) and storage_format == "int8"
        self.offload_pinned = storage_format in ("fp8", "mxfp8") or self.host_resident
        self._offload_buffers = OrderedDict()
        self._offload_buffer_bytes = 0
        self._offload_buffer_bytes_limit = _OFFLOAD_BUFFER_BYTES_LIMIT
        self._offload_buffer_index = {}
        self._offload_events = {}
        # Ceil partition leaves at most size-1 unused rows, never a replica.
        self.shard_rows = (rows + query_group.size - 1) // query_group.size
        # [ENGRAM-JIT-PLAN] 规划缓冲（仅 PLAN_JIT 时使用）
        self._plan_cap = 0
        self._plan_size = -1
        self._plan_order = None
        self._plan_counts = None
        self._plan_starts = None
        self._plan_cursor = None
        self._plan_myids = None
        self._plan_views = {}
        self.start = query_group.rank * self.shard_rows
        self.end = min(self.start + self.shard_rows, rows)
        if self.start >= rows:
            raise ValueError("Engram table must have at least one row per rank")
        self._empty_flat = torch.empty(0, dtype=torch.int64, device="cpu")
        # Reuse fixed-size HCCL metadata buffers across requests.
        self._metadata_device_buffers = {}
        self._empty_metadata = torch.zeros(query_group.size + 1, dtype=torch.int64, device="cpu")
        self.weight = nn.Parameter(
            torch.empty(
                self.end - self.start,
                width,
                dtype=(
                    torch.int8
                    if storage_format == "int8"
                    else (
                        torch.float8_e4m3fn
                        if storage_format in ("fp8", "mxfp8")
                        else torch.bfloat16
                    )
                ),
                device=(
                    torch.device("cpu")
                    if (storage_format in ("fp8", "mxfp8") or getattr(self, "host_resident", False))
                    else device
                ),
                pin_memory=False,
            ),
            requires_grad=False,
        )
        if storage_format == "int8":
            self.register_buffer(
                "weight_scale",
                torch.empty(
                    self.end - self.start,
                    width // 32,
                    dtype=torch.float32,
                    device=(torch.device("cpu") if getattr(self, "host_resident", False) else device),
                ),
            )
        elif storage_format in ("fp8", "mxfp8"):
            self.register_buffer(
                "weight_scale",
                torch.empty(
                    self.end - self.start,
                    width // 32,
                    dtype=torch.float8_e8m0fnu,
                    device="cpu",
                    pin_memory=False,
                ),
            )

    def set_rows(self, start, rows):
        """Load BF16 rows into local storage without allocating a BF16 table copy."""
        end = start + rows.shape[0]
        if self.storage_format == "int8":
            codes, scales = quantize_engram_rows(rows.to(self.weight.device))
            if not bool((torch.isfinite(scales) & (scales > 0)).all()):
                raise ValueError("INT8 Engram requires finite positive group scales")
            self.weight.data[start:end].copy_(codes)
            self.weight_scale[start:end].copy_(scales)
        else:
            self.weight.data[start:end].copy_(rows)

    def lookup_local(self, ids):
        # Idle DP replicas still enter routing collectives, but must not launch
        # gather/dequant kernels for an empty owner request.
        if ids.numel() == 0:
            return torch.empty((*ids.shape, self.width), dtype=torch.bfloat16, device=self.weight.device)
        original_shape = ids.shape
        flat_ids = ids.reshape(-1)
        if self.storage_format == "int8":
            if (
                self.use_triton_int8
                and self.weight.device.type == "npu"
                and self.width == 256
                and flat_ids.device.type == "npu"
                and flat_ids.shape[0] >= self.triton_int8_min_rows
            ):
                rows = gather_dequantize_engram_int8(self.weight, self.weight_scale, flat_ids)
            else:
                codes = torch.index_select(self.weight, 0, flat_ids)
                scales = torch.index_select(self.weight_scale, 0, flat_ids)
                rows = dequantize_engram_rows(codes, scales)
                if self.offload_pinned:
                    rows = self._stage_pinned_rows(rows)
        elif self.storage_format in ("fp8", "mxfp8"):
            # index_select avoids the extra advanced-indexing wrapper on the
            # CPU-resident PLE table and keeps row selection explicit.
            rows = torch.index_select(self.weight, 0, flat_ids)
            decoded = rows.float().reshape(-1, self.width // 32, 32)
            scales = torch.index_select(self.weight_scale, 0, flat_ids)
            decoded.mul_(scales.float().unsqueeze(-1))
            decoded = decoded.reshape(-1, self.width)
            rows = self._stage_pinned_rows(decoded) if self.offload_pinned else decoded.bfloat16()
        else:
            rows = torch.index_select(self.weight, 0, flat_ids)
        return rows.view(*original_shape, self.width)

    def _stage_pinned_rows(self, decoded: torch.Tensor) -> torch.Tensor:
        """[dsv41-patch] 把解码后的 BF16 行放入 pinned buffer（fp8/mxfp8 与 int8-host 共用）。"""
        key = decoded.shape[0]
        slots = self._offload_buffers.get(key)
        if slots is None:
            slot_bytes = 2 * key * self.width * _BF16_BYTES
            while self._offload_buffers and (
                len(self._offload_buffers) >= _OFFLOAD_BUFFER_CACHE_SIZE
                or self._offload_buffer_bytes + slot_bytes > self._offload_buffer_bytes_limit
            ):
                evicted, evicted_slots = self._offload_buffers.popitem(last=False)
                self._offload_buffer_bytes -= 2 * evicted * self.width * _BF16_BYTES
                self._offload_buffer_index.pop(evicted, None)
                for slot in evicted_slots:
                    event = self._offload_events.pop(slot.data_ptr(), None)
                    if event is not None:
                        event.synchronize()
            slots = [torch.empty((key, self.width), dtype=torch.bfloat16, pin_memory=True) for _ in range(2)]
            self._offload_buffers[key] = slots
            self._offload_buffer_bytes += slot_bytes
            self._offload_buffer_index[key] = 0
        else:
            self._offload_buffers.move_to_end(key)
        index = self._offload_buffer_index[key]
        decoded_slot = slots[index]
        self._offload_buffer_index[key] = 1 - index
        event = self._offload_events.pop(decoded_slot.data_ptr(), None)
        if event is not None:
            event.synchronize()
        decoded_slot.copy_(decoded)
        return decoded_slot

    def _record_offload_use(self, source_ptr, device, reuse_event=False):
        """Keep a pinned staging slot alive until the submitted device work ends.

        reuse_event=True 时复用同一 ptr 上已有的 Event 对象（重新 record 即可，
        语义等价），省掉每步每表一次 `torch.npu.Event()` 构造 —— 相位实测该项
        0.112 ms/步（两表合计）。
        """
        if not self.offload_pinned or device.type != "npu":
            return
        if reuse_event:
            event = self._offload_events.get(source_ptr)
            if event is None:
                event = torch.npu.Event()
                self._offload_events[source_ptr] = event
        else:
            event = torch.npu.Event()
            self._offload_events[source_ptr] = event
        event.record(torch.npu.current_stream(device))

    def load_checkpoint(self, model_path, key, chunk_rows=65536):
        """Load BF16, INT8, FP8, or MXFP8 Engram tensors with bounded IO.

        FP8/MXFP8 remain CPU resident (PLE_OFFLOAD); only decoded BF16 rows
        enter the node-local all-to-all response buffer.
        """
        root = Path(model_path)
        index = json.loads((root / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
        scale_key = key.removesuffix(".weight") + ".scale"
        with safe_open(root / index[key], framework="pt", device="cpu") as file:
            tensor = file.get_slice(key)
            if tensor.get_shape() != [self.rows, self.width]:
                raise ValueError(f"{key}: expected BF16/FP8 [{self.rows}, {self.width}]")
            source_dtype = tensor.get_dtype()
            if self.storage_format == "int8" and source_dtype in ("I8", "INT8"):
                if scale_key not in index:
                    raise ValueError(f"{key}: INT8 source requires .scale")
                with safe_open(root / index[scale_key], framework="pt", device="cpu") as sf:
                    scale = sf.get_slice(scale_key)
                    if scale.get_shape() != [self.rows, self.width // 32] or scale.get_dtype() != "F32":
                        raise ValueError(f"{scale_key}: expected FP32 [{self.rows}, {self.width // 32}]")
                    for start in range(self.start, self.end, chunk_rows):
                        stop = min(start + chunk_rows, self.end)
                        self.weight.data[start-self.start:stop-self.start].copy_(tensor[start:stop])
                        self.weight_scale[start-self.start:stop-self.start].copy_(scale[start:stop])
                return
            if self.storage_format in ("fp8", "mxfp8"):
                if source_dtype not in ("F8_E4M3", "F8_E4M3FN") or scale_key not in index:
                    raise ValueError(f"{key}: {self.storage_format} requires FP8 weight and .scale")
                with safe_open(root / index[scale_key], framework="pt", device="cpu") as sf:
                    scale = sf.get_slice(scale_key)
                    if scale.get_shape() != [self.rows, self.width // 32]:
                        raise ValueError(f"{scale_key}: expected [{self.rows}, {self.width // 32}]")
                    for start in range(self.start, self.end, chunk_rows):
                        stop = min(start + chunk_rows, self.end)
                        self.weight.data[start-self.start:stop-self.start].copy_(tensor[start:stop])
                        self.weight_scale[start-self.start:stop-self.start].copy_(scale[start:stop])
                return
            if source_dtype != "BF16":
                raise ValueError(f"{key}: expected BF16 source for {self.storage_format}")
            for start in range(self.start, self.end, chunk_rows):
                stop = min(start + chunk_rows, self.end)
                self.set_rows(start - self.start, tensor[start:stop])

    def _metadata(self, ids):
        q = self.query_group
        flat = ids.reshape(-1) if q.is_source else ids.new_empty(0)
        if flat.numel() == 0:
            return self._empty_flat, self._empty_flat, self._empty_metadata
        invalid = bool(flat.min() < 0 or flat.max() >= self.rows)
        owners = flat.clamp(0, self.rows - 1) // self.shard_rows if invalid else flat // self.shard_rows
        # Only owner grouping is required; preserving equal-owner order adds
        # avoidable CPU sort work because the same permutation restores rows.
        order = owners.argsort(stable=False)
        counts = torch.bincount(owners, minlength=q.size)
        metadata = torch.empty(q.size + 1, dtype=torch.int64, device="cpu")
        metadata[:-1].copy_(counts)
        metadata[-1] = int(invalid)
        return flat, order, metadata

    @torch.inference_mode()
    def _forward_with_gathered(self, ids, gathered, routing=None, broadcast=True, output=None):
        """Return ids.shape + [width], bit-preserving, even when a DP is idle.

        IDs reside on CPU; hashing/history already runs at the eager boundary.
        Exactly one TP rank submits the DP's queries. All owners serve requests;
        reverse all-to-all restores requester order before the TP broadcast.
        """
        q = self.query_group
        if ids.device.type != "cpu" or ids.dtype != torch.int64:
            raise ValueError("Engram routing expects CPU int64 IDs")
        if routing is None:
            flat, order, metadata = self._metadata(ids)
        else:
            flat, order, metadata = routing
        counts = [row.tolist() for row in gathered]
        if any(row[-1] for row in counts):
            raise IndexError("Engram hash ID outside table")
        send = metadata[:-1].tolist()
        recv = [row[q.rank] for row in counts]
        total_recv = sum(recv)
        total_requests = sum(send)
        total_global = sum(sum(row[:-1]) for row in counts)
        backend = str(dist.get_backend(q.group)).lower()
        # HCCL collectives require NPU tensors even when the compressed table is CPU resident.
        device = self.weight.device
        if device.type == "cpu" and backend not in ("gloo", "mpi"):
            device = torch.device("npu")
        if device.type == "npu" and device.index is None:
            device = torch.device("npu", torch.npu.current_device())
        # All-zero rounds are skipped identically on every rank (HCCL portability).
        if _RP.on and _RP.t0 is None:
            _RP.start()
        if total_global:
            incoming = torch.empty(total_recv, dtype=torch.int64, device=device)
            ordered_ids = torch.index_select(flat, 0, order).to(device)
            dist.all_to_all_single(incoming, ordered_ids, recv, send, group=q.group)
            local_ids = incoming - self.start
            _RP.mark("a2a1")
            if self.storage_format == "int8" and self.compressed_int8_wire:
                lookup_ids = local_ids.cpu() if self.weight.device.type == "cpu" else local_ids
                _RP.mark("ids_d2h")
                codes = torch.index_select(self.weight, 0, lookup_ids)
                scales = torch.index_select(self.weight_scale, 0, lookup_ids)
                _RP.mark("cpu_gather")
                packed = pack_engram_int8_rows(codes.to(device), scales.to(device))
                _RP.mark("pack_h2d")
                wire_width = self.width + (self.width // 32) * 4
                returned = torch.empty(total_requests * wire_width, dtype=torch.uint8, device=device)
                wire_send = [count * wire_width for count in send]
                wire_recv = [count * wire_width for count in recv]
                dist.all_to_all_single(returned, packed.flatten(), wire_send, wire_recv, group=q.group)
                _RP.mark("a2a2")
                returned = unpack_engram_int8_rows(returned.reshape(total_requests, wire_width), self.width)
                _RP.mark("unpack")
            else:
                local_values = self.lookup_local(local_ids.cpu() if self.weight.device.type == "cpu" else local_ids)
                _RP.mark("ids_d2h_lookup")
                source_ptr = local_values.data_ptr()
                values = local_values.to(
                    device=device, dtype=torch.bfloat16, non_blocking=self.offload_pinned
                ).contiguous()
                returned = torch.empty((total_requests, self.width), dtype=torch.bfloat16, device=device)
                dist.all_to_all_single(returned, values, send, recv, group=q.group)
                _RP.mark("a2a2")
                self._record_offload_use(source_ptr, device)
        else:
            returned = torch.empty((0, self.width), dtype=torch.bfloat16, device=device)
            _RP.mark("skip")
        if output is None:
            result = torch.empty((ids.numel(), self.width), dtype=torch.bfloat16, device=device)
        else:
            result = output
            if result.shape != (ids.numel(), self.width) or result.device != device:
                raise ValueError("Engram output buffer has an incompatible shape or device")
        if q.is_source:
            result[order.to(device)] = returned
        _RP.mark("scatter")
        if broadcast and result.numel():
            dist.broadcast(result, src=q.tp_source, group=q.tp_group)
        _RP.mark("bcast")
        _RP.tick()
        return result.view(*ids.shape, self.width)

    def _local_owner_plan_jit(self, table, ids):
        # [ENGRAM-JIT-PLAN] numba 版：稳定计数排序 + 本 rank 切片。
        # 与 _local_owner_plan_numpy 逐位等价（四个返回值全对账），
        # 数值语义见 sidecar engram_plan_kernel.py 头部注释。
        import numpy as np          # 目标文件里 numpy 只在函数内导入，沿用同一风格
        if ids.numel() == 0:
            return None
        arr = _engram_plan_flatten(ids)
        n = arr.shape[0]
        size = self.query_group.size
        if self._plan_cap < n or self._plan_size != size:
            self._plan_cap = n
            self._plan_size = size
            self._plan_order = np.empty(n, dtype=np.int64)
            self._plan_counts = np.empty(size, dtype=np.int64)
            self._plan_starts = np.empty(size, dtype=np.int64)
            self._plan_cursor = np.empty(size, dtype=np.int64)
            self._plan_myids = np.empty(n, dtype=np.int64)
            self._plan_views = {}
        m = _engram_plan_kernel(
            arr, int(table.rows), int(table.shard_rows), size,
            int(self.query_group.rank),
            self._plan_order, self._plan_counts, self._plan_starts,
            self._plan_cursor, self._plan_myids,
        )
        if m < 0:
            # 与 stock 同型同文本（stock: raise IndexError("Engram hash ID outside table")）
            raise IndexError("Engram hash ID outside table")
        key = (n, m, size)
        views = self._plan_views.get(key)
        if views is None:
            views = (
                torch.from_numpy(self._plan_order[:n]),
                torch.from_numpy(self._plan_counts[:size]),
                torch.from_numpy(self._plan_myids[:m]),
            )
            self._plan_views[key] = views
        # 第一项 `flat`：stock 返回 ids.reshape(-1) 的 torch 张量；所有调用方都丢弃它，
        # 这里返回同一语义的廉价视图。
        return (torch.from_numpy(arr),) + views

    def _local_owner_plan_numpy(self, table, ids):
        if _ENGRAM_PLAN_JIT:
            return self._local_owner_plan_jit(table, ids)
        """[fast] `_local_owner_plan` 的 numpy 版：把 8 个 torch CPU 小算子
        （每个 ~10-30 µs 的 dispatch 开销）换成 ~6 个 numpy 算子（每个 ~1-2 µs）。

        相位实测：plan 相 0.255 ms/步（两表），主要就是这个 dispatch 开销。
        数值语义与 torch 版**逐位相同**（同样的整除/排序/切片），并且所有 rank
        用同一份代码 ⇒ `order` 切片与各 rank 的 `idx` 仍然一致。
        """
        import numpy as np

        flat = ids.reshape(-1)
        if flat.numel() == 0:
            return None
        arr = flat.numpy()
        if arr.min() < 0 or arr.max() >= table.rows:
            raise IndexError("Engram hash ID outside table")
        owners = arr // table.shard_rows
        order = np.argsort(owners, kind="stable")
        size = self.query_group.size
        counts = np.bincount(owners, minlength=size).astype(np.int64)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
        r = self.query_group.rank
        idx = order[starts[r] : starts[r] + counts[r]]
        return (
            flat,
            torch.from_numpy(order),
            torch.from_numpy(counts),
            torch.from_numpy(arr[idx]),
        )

    def _local_owner_plan(self, table, ids):
        """[local-owner] 本地推导本 rank 拥有的那批 id（与 source 的发送顺序逐位一致）。

        返回 (flat, order, counts, my_ids_global) 或 None（前提不满足）。
        """
        flat = ids.reshape(-1)
        if flat.numel() == 0:
            return None
        invalid = bool(flat.min() < 0 or flat.max() >= table.rows)
        owners = (
            flat.clamp(0, table.rows - 1) // table.shard_rows if invalid else flat // table.shard_rows
        )
        if invalid:
            # 与既有实现一致：交给调用方按"ID 越界"报错。
            raise IndexError("Engram hash ID outside table")
        order = owners.argsort(stable=False)
        counts = torch.bincount(owners, minlength=self.query_group.size)
        starts = counts.cumsum(0) - counts
        r = self.query_group.rank
        idx = order[starts[r] : starts[r] + int(counts[r])]
        return flat, order, counts, torch.index_select(flat, 0, idx)

    def _validate_identical(self, table, flat, counts):
        """validate 模式：把本 rank 的 flat/counts 与其它 rank 对账。

        所有 rank 都必须调用（否则集合通信会挂），返回值在全局一致。
        """
        q = self.query_group
        mine = torch.cat([flat.detach().cpu().reshape(-1), counts.detach().cpu().reshape(-1)]).to(torch.int64)
        rows = [torch.empty_like(mine) for _ in range(q.size)]
        dist.all_gather(rows, mine, group=q.cpu_group)
        same = all(torch.equal(row, mine) for row in rows)
        return same

    def _route_local_owner(self, tables, ids_list, broadcast=True, fast=False):
        """[local-owner] 跳过 metadata all_gather 与 ids all_to_all，只保留值回传。"""
        q = self.query_group
        src = q.source_group_ranks[0]
        r = q.rank
        device = self.weight.device
        if device.type == "cpu":
            device = torch.device("npu", torch.npu.current_device())
        if _PIPE.get() == "on" and len(tables) > 1:
            return self._route_local_owner_pipelined(tables, ids_list, broadcast=broadcast, fast=fast)
        outs = []
        _RP.start()
        for table, ids in zip(tables, ids_list):
            planned = (
                self._local_owner_plan_numpy(table, ids) if fast else self._local_owner_plan(table, ids)
            )
            _RP.mark("plan")
            if planned is None:
                outs.append(
                    torch.empty((*ids.shape, table.width), dtype=torch.bfloat16, device=device)
                )
                continue
            _, order, counts, global_ids = planned
            local_ids = global_ids - table.start
            values = table.lookup_local(local_ids)
            _RP.mark("lookup")
            source_ptr = values.data_ptr()
            inp = values.to(device=device, dtype=torch.bfloat16, non_blocking=table.offload_pinned).contiguous()
            _RP.mark("h2d")
            # 与既有路径一致：把 pinned staging slot 钉到设备工作结束之后，
            # 否则下一次 lookup 可能提前复用这块内存（竞态）。
            table._record_offload_use(source_ptr, device, reuse_event=fast)
            _RP.mark("evt")
            input_split = [0] * q.size
            input_split[src] = int(counts[r])
            output_split = counts.tolist() if r == src else [0] * q.size
            returned = torch.empty(
                (int(sum(output_split)), table.width), dtype=torch.bfloat16, device=device
            )
            dist.all_to_all_single(returned, inp, output_split, input_split, group=q.group)
            _RP.mark("a2a")
            result = torch.empty((ids.numel(), table.width), dtype=torch.bfloat16, device=device)
            if r == src:
                result[order.to(device)] = returned
            _RP.mark("scatter")
            if broadcast and result.numel():
                dist.broadcast(result, src=q.tp_source, group=q.tp_group)
            _RP.mark("bcast")
            outs.append(result.view(*ids.shape, table.width))
        _RP.tick()
        return outs

    def _route_local_owner_pipelined(self, tables, ids_list, broadcast=True, fast=False):
        """[route-pipe] 与 `_route_local_owner` 数值等价，只重排 host/通信顺序。

        phase 1：所有表的 plan / lookup / h2d / evt（纯 CPU，不碰集合通信）
        phase 2：依次下发各表的 all_to_all_single(async_op=True)
        phase 3：统一 wait，然后 scatter + broadcast
        """
        q = self.query_group
        src = q.source_group_ranks[0]
        r = q.rank
        device = self.weight.device
        if device.type == "cpu":
            device = torch.device("npu", torch.npu.current_device())
        stage = []
        _RP.start()
        for table, ids in zip(tables, ids_list):
            planned = (
                self._local_owner_plan_numpy(table, ids) if fast else self._local_owner_plan(table, ids)
            )
            _RP.mark("plan")
            if planned is None:
                stage.append((table, ids, None, None, None, None, None))
                continue
            _, order, counts, global_ids = planned
            local_ids = global_ids - table.start
            values = table.lookup_local(local_ids)
            _RP.mark("lookup")
            source_ptr = values.data_ptr()
            inp = values.to(device=device, dtype=torch.bfloat16, non_blocking=table.offload_pinned).contiguous()
            _RP.mark("h2d")
            table._record_offload_use(source_ptr, device, reuse_event=fast)
            _RP.mark("evt")
            input_split = [0] * q.size
            input_split[src] = int(counts[r])
            output_split = counts.tolist() if r == src else [0] * q.size
            returned = torch.empty(
                (int(sum(output_split)), table.width), dtype=torch.bfloat16, device=device
            )
            stage.append((table, ids, order, returned, inp, output_split, input_split))

        handles = []
        for table, ids, order, returned, inp, output_split, input_split in stage:
            if order is None:
                continue
            handles.append(
                dist.all_to_all_single(
                    returned, inp, output_split, input_split, group=q.group, async_op=True
                )
            )
        _RP.mark("a2a")
        for h in handles:
            if h is not None:
                h.wait()

        outs = []
        for table, ids, order, returned, inp, output_split, input_split in stage:
            if order is None:
                outs.append(
                    torch.empty((*ids.shape, table.width), dtype=torch.bfloat16, device=device)
                )
                continue
            result = torch.empty((ids.numel(), table.width), dtype=torch.bfloat16, device=device)
            if r == src:
                result[order.to(device)] = returned
            _RP.mark("scatter")
            if broadcast and result.numel():
                dist.broadcast(result, src=q.tp_source, group=q.tp_group)
            _RP.mark("bcast")
            outs.append(result.view(*ids.shape, table.width))
        _RP.tick()
        return outs

    def _route_local_owner_gather(self, tables, ids_list, broadcast=True):
        """[local-owner+gather] 用**一次 all_gather** 取代 "all_to_all + broadcast"。

        思路：只做 DP=1 时所有 rank 的 ids 相同 ⇒ 每个 rank 都能独立算出"每个 token
        归哪个 rank 所有"以及"自己该贡献哪些行"。于是：
          * 每个 rank 只查自己 shard 的行（无需中转）
          * 一次 all_gather 把各 rank 的行拼在一起（按 owner 分块、块内保持全局顺序）
          * 每个 rank 用自己算出的 owner/序位索引，直接组装出完整结果
        相比原路径省掉：一次 all_to_all + 一次 broadcast（HCCL 小包固定开销为主）。
        """
        q = self.query_group
        r = q.rank
        device = self.weight.device
        if device.type == "cpu":
            device = torch.device("npu", torch.npu.current_device())
        outs = []
        _RP.start()
        for table, ids in zip(tables, ids_list):
            flat = ids.reshape(-1).to(torch.int64)
            if flat.numel() == 0:
                outs.append(torch.empty((*ids.shape, table.width), dtype=torch.bfloat16, device=device))
                continue
            if bool(flat.min() < 0 or flat.max() >= table.rows):
                raise IndexError("Engram hash ID outside table")
            owners = flat // table.shard_rows
            counts = torch.bincount(owners, minlength=q.size)
            _RP.mark("plan")
            mine = torch.nonzero(owners == r, as_tuple=False).flatten()
            local_ids = flat.index_select(0, mine) - table.start
            values = table.lookup_local(local_ids)
            _RP.mark("lookup")
            source_ptr = values.data_ptr()
            vals_dev = values.to(device=device, dtype=torch.bfloat16,
                                 non_blocking=table.offload_pinned).contiguous()
            _RP.mark("h2d")
            table._record_offload_use(source_ptr, device)
            _RP.mark("evt")
            maxc = int(counts.max())
            # 本 rank 只贡献"自己 shard 的那几行"，pad 到 maxc 行（各 rank 形状必须一致）
            contrib = torch.zeros((maxc, table.width), dtype=torch.bfloat16, device=device)
            if maxc:
                contrib[: int(counts[r])] = vals_dev[: int(counts[r])]
            gathered = torch.empty((q.size * maxc, table.width), dtype=torch.bfloat16, device=device)
            dist.all_gather_into_tensor(gathered, contrib, group=q.group)
            _RP.mark("agather")
            # 组装：token i 的行 = gathered[owner_i, 该 owner 块内第 k 个]
            order = torch.argsort(owners, stable=True)              # [n] 按 owner 排序的 token 序号
            owners_sorted = owners.index_select(0, order)
            starts = counts.cumsum(0) - counts                     # [size]
            pos_in_block = torch.arange(order.numel()) - starts.index_select(0, owners_sorted)
            src_idx = (owners_sorted * maxc + pos_in_block).to(device)
            rows_sorted = gathered.index_select(0, src_idx)          # [n, width]（按 order 排列）
            result = torch.empty((flat.numel(), table.width), dtype=torch.bfloat16, device=device)
            result[order.to(device)] = rows_sorted                   # 还原原始 token 顺序
            _RP.mark("assemble")
            outs.append(result.view(*ids.shape, table.width))
        _RP.tick()
        return outs


    def _local_owner_b2g(self, tables, ids_list, broadcast=True):
        """[b2g] 两张表**合并成一次 all_gather**（4 次集合通信 -> 1 次）。

        结构：两张表各自按 owner 分块，把自己 shard 的行按全局序位拼在一行里
        （左半 = 表0，右半 = 表1），一次 all_gather 交换；随后各表用自己的
        `order` 与 `counts` 做一次掩码压缩（去掉 padding）+ scatter 还原。

        与已有路径的数值语义完全一致（lookup 结果 -> 按 order 写回），
        只是把 2×all_to_all + 2×broadcast 合成 1×all_gather。
        """
        q = self.query_group
        r = q.rank
        device = self.weight.device
        if device.type == "cpu":
            device = torch.device("npu", torch.npu.current_device())
        if len(tables) != 2:
            return self._route_local_owner(tables, ids_list, fast=True)

        plans = [t._local_owner_plan_numpy(t, ids) for t, ids in zip(tables, ids_list)]
        if any(p is None for p in plans):
            return [torch.empty((*ids.shape, t.width), dtype=torch.bfloat16, device=device)
                    for t, ids in zip(tables, ids_list)]
        counts = [p[2] for p in plans]
        w0, w1 = tables[0].width, tables[1].width
        maxc = max(int(counts[0].max()), int(counts[1].max()))
        if maxc == 0:
            return [torch.empty((*ids.shape, t.width), dtype=torch.bfloat16, device=device)
                    for t, ids in zip(tables, ids_list)]

        _RP.start()
        contrib = torch.zeros((maxc, w0 + w1), dtype=torch.bfloat16, device=device)
        for ti, (t, p) in enumerate(zip(tables, plans)):
            n_r = int(counts[ti][r])
            _RP.mark("plan%d" % ti)
            if n_r == 0:
                continue
            vals = t.lookup_local(p[3] - t.start)
            _RP.mark("lookup%d" % ti)
            dev = vals.to(device=device, dtype=torch.bfloat16,
                          non_blocking=t.offload_pinned).contiguous()
            t._record_offload_use(vals.data_ptr(), device, reuse_event=True)
            _RP.mark("h2d%d" % ti)
            if ti == 0:
                contrib[:n_r, :w0] = dev
            else:
                contrib[:n_r, w0:] = dev
            _RP.mark("stage%d" % ti)

        gathered = torch.empty((q.size * maxc, w0 + w1), dtype=torch.bfloat16, device=device)
        dist.all_gather_into_tensor(gathered, contrib, group=q.group)
        _RP.mark("agather")
        row_ids = torch.arange(q.size * maxc, device=device)
        outs = []
        for ti, (t, p) in enumerate(zip(tables, plans)):
            flat, order, cnt, gids = p
            keep = (row_ids % maxc) < cnt.to(device).repeat_interleave(maxc)
            cols = slice(0, w0) if ti == 0 else slice(w0, w0 + w1)
            compacted = gathered[keep][:, cols]          # [n, width]，按 order 排列
            result = torch.empty((flat.numel(), t.width), dtype=torch.bfloat16, device=device)
            result[order.to(device)] = compacted         # 还原原始 token 顺序
            outs.append(result.view(*ids_list[ti].shape, t.width))
        _RP.mark("assemble")
        _RP.tick()
        return outs

    def _gather_metadata_device(self, metadata, device, group):
        """All-gather metadata through reusable device buffers."""
        key = (str(device), metadata.numel(), metadata.dtype)
        buffers = self._metadata_device_buffers.get(key)
        if buffers is None:
            buffers = (
                torch.empty(metadata.numel(), dtype=metadata.dtype, device=device),
                torch.empty(self.query_group.size * metadata.numel(),
                            dtype=metadata.dtype, device=device),
            )
            self._metadata_device_buffers[key] = buffers
        metadata_device, gathered_device = buffers
        metadata_device.copy_(metadata, non_blocking=False)
        dist.all_gather_into_tensor(gathered_device, metadata_device, group=group)
        # The reshape/unbind views are copied to CPU before returning.  Keep
        # consumption local to this route so the reusable device buffer remains
        # safe for the next collective.
        return list(gathered_device.reshape(self.query_group.size, *metadata.shape).cpu().unbind(0))

    @torch.inference_mode()
    def forward(self, ids):
        q = self.query_group
        if ids.device.type != "cpu" or ids.dtype != torch.int64:
            raise ValueError("Engram routing expects CPU int64 IDs")
        _, _, metadata = self._metadata(ids)
        if q.metadata_on_device:
            device = self.weight.device if self.weight.device.type == "npu" else torch.device("npu")
            gathered = self._gather_metadata_device(metadata, device, q.group)
        else:
            gathered = [torch.empty_like(metadata) for _ in range(q.size)]
            dist.all_gather(gathered, metadata, group=q.cpu_group)
        return self._forward_with_gathered(ids, gathered)

    @torch.inference_mode()
    def forward_many(self, ids_list):
        """Route several Engram tables with one CPU metadata collective."""
        return self.route_many([self] * len(ids_list), ids_list)

    @torch.inference_mode()
    def route_many(self, tables, ids_list):
        """Route distinct tables while sharing their CPU metadata collective."""
        if not ids_list:
            return []
        q = self.query_group
        if len(tables) != len(ids_list):
            raise ValueError("tables and ids_list must have the same length")
        # [local-owner] DP=1 + 单一 requester 时，owner 可本地推导自己拥有的 id。
        _lo = _LO.get()
        if (
            _lo in ("on", "validate", "gather", "fast", "b2g")
            and len(tables) > 1
            and q.equiv_ranks
            and q.source_group_ranks
        ):
            if _lo == "validate":
                _ok = True
                for _table, _ids in zip(tables, ids_list):
                    try:
                        _plan = _table._local_owner_plan(_table, _ids)
                    except IndexError:
                        _plan = None
                    if _plan is None:
                        continue
                    _ok = _ok and _table._validate_identical(_table, _plan[0], _plan[2])
                if _ok:
                    _LO.ok += 1
                    if _LO.ok in (1, 100, 1000) and _RP.rank0:
                        print(
                            f"[local-owner] VALIDATE OK x{_LO.ok}（{q.size} 个 rank 的 ids/counts 逐位相同）",
                            flush=True,
                        )
                else:
                    _LO.bad += 1
                    if _RP.rank0:
                        print(
                            f"[local-owner] VALIDATE FAILED x{_LO.bad}：各 rank 的 ids/counts 不同，回退旧路径",
                            flush=True,
                        )
                    _lo = "off"  # 回退：继续走下面的旧路径
            if _lo == "on":
                return tables[0]._route_local_owner(tables, ids_list)
            if _lo == "gather":
                return tables[0]._route_local_owner_gather(tables, ids_list)
            if _lo == "fast":
                return tables[0]._route_local_owner(tables, ids_list, fast=True)
            if _lo == "b2g":
                return tables[0]._local_owner_b2g(tables, ids_list)
        _RP.start()
        routing = [table._metadata(ids) for table, ids in zip(tables, ids_list)]
        metadata = [item[2] for item in routing]
        packed = torch.cat(metadata)
        _RP.mark("meta_prep")
        # [local-metadata] DP=1 时 metadata 组内相同，可本地复制代替 all_gather。
        _lm = _LM.get()
        _use_local = _lm == "on" and q.equiv_ranks
        _validate = (_lm == "validate" and q.equiv_ranks)
        gathered_real = None
        if _use_local:
            gathered_packed = [packed] * q.size
        elif q.metadata_on_device:
            device = tables[0].weight.device if tables[0].weight.device.type == "npu" else torch.device("npu")
            _RP.start()
            gathered_real = self._gather_metadata_device(packed, device, q.group)
            _RP.mark("meta_gather")
            gathered_packed = gathered_real
        else:
            gathered_packed = [torch.empty_like(packed) for _ in range(q.size)]
            dist.all_gather(gathered_packed, packed, group=q.cpu_group)
            gathered_real = gathered_packed
        if _validate and gathered_real is not None:
            # 对账：本地副本 vs 真实 gathered（不一致就打印并退回真值）
            _bad = 0
            for _row in gathered_real:
                if not torch.equal(_row, packed):
                    _bad += 1
            if _bad:
                if _RP.rank0:
                    print(
                        f"[local-metadata] VALIDATE FAILED: {_bad}/{q.size} 行与本地副本不同；退回 gathered",
                        flush=True,
                    )
            else:
                _LM.ok = getattr(_LM, "ok", 0) + 1
                if _LM.ok in (1, 100, 1000) and _RP.rank0:
                    print(f"[local-metadata] VALIDATE OK x{_LM.ok}（{q.size} 行全部逐位相同）", flush=True)
        width = q.size + 1
        gathered = [
            [row[offset : offset + width] for row in gathered_packed]
            for offset in range(0, len(ids_list) * width, width)
        ]
        if len(tables) == 1:
            result = tables[0]._forward_with_gathered(ids_list[0], gathered[0], routing[0])
            return [result]
        total = sum(ids.numel() * table.width for table, ids in zip(tables, ids_list))
        device = tables[0].weight.device
        if device.type == "cpu" and str(dist.get_backend(q.group)).lower() not in ("gloo", "mpi"):
            device = torch.device("npu")
        if device.type == "npu" and device.index is None:
            device = torch.device("npu", torch.npu.current_device())
        combined = torch.empty(total, dtype=torch.bfloat16, device=device)
        results = []
        offset = 0
        for table, ids, group, item in zip(tables, ids_list, gathered, routing):
            size = ids.numel() * table.width
            result = table._forward_with_gathered(
                ids,
                group,
                item,
                broadcast=False,
                output=combined[offset : offset + size].view(ids.numel(), table.width),
            )
            results.append(result)
            offset += size
        if total:
            dist.broadcast(combined, src=q.tp_source, group=q.tp_group)
        outputs = []
        offset = 0
        for result in results:
            size = result.numel()
            outputs.append(combined[offset : offset + size].view_as(result))
            offset += size
        return outputs
