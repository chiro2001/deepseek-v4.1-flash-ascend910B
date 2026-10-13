# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# NPU adaptation of gpu_ubatch_wrapper.py for Ascend NPU DBO support.
# Replaces torch.cuda.Stream/Event with torch.npu equivalents.

import threading
from collections.abc import Callable
from dataclasses import dataclass

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import (
    create_forward_context,
    get_forward_context,
    override_forward_context,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.ubatching import UBatchContext


_CAT_DEPTH = [0]


def _describe(x):
    if isinstance(x, torch.Tensor):
        return "Tensor%s" % (tuple(x.shape),)
    if isinstance(x, (tuple, list)):
        return "%s(len=%d)[%s]" % (type(x).__name__, len(x),
                                   (", ".join(_describe(i) for i in x[:3]) if x else ""))
    if isinstance(x, dict):
        return "dict(keys=%s)" % (list(x.keys())[:3],)
    return type(x).__name__


def _cat_ubatch_outputs(sorted_results):
    """按 batch 维拼接各 ubatch 的输出（递归；容忍空序列 / None / 嵌套 / dict）。

    最后一层兜底：**任何情况下都不返回"比输入更少元素"的容器**，
    否则调用方 `hidden_states, _ = outputs` 会 unpack 失败
    （实测报过 expected 2 got 0）。
    """
    import os as _os
    _dbg = _os.environ.get("V41_DBO_DEBUG") == "1"
    d = _CAT_DEPTH[0]
    _CAT_DEPTH[0] = d + 1
    try:
        if _dbg:
            print("[DBO-CAT%d] in: %s" % (d, _describe(sorted_results)), flush=True)
        out = _cat_impl(sorted_results, d, _dbg)
        if _dbg:
            print("[DBO-CAT%d] out: %s" % (d, _describe(out)), flush=True)
        return out
    finally:
        _CAT_DEPTH[0] = d


def _cat_impl(sorted_results, d, _dbg):
    # 非容器：原样返回
    if isinstance(sorted_results, torch.Tensor) or sorted_results is None:
        return sorted_results
    if isinstance(sorted_results, dict):
        if not sorted_results:
            return sorted_results
        v0 = next(iter(sorted_results.values()))
        if isinstance(v0, (tuple, list, dict)) or v0 is None:
            return {k: _cat_ubatch_outputs([r[k] for r in sorted_results]) for k in sorted_results}
        return torch.cat(list(sorted_results.values()), dim=0)
    if not isinstance(sorted_results, (tuple, list)):
        return sorted_results
    if len(sorted_results) == 0:
        return sorted_results
    if all(x is None for x in sorted_results):
        return None

    first = sorted_results[0]
    if isinstance(first, (tuple, list, dict)):
        if isinstance(first, dict):
            return {k: _cat_ubatch_outputs([r[k] for r in sorted_results]) for k in first}
        cls = type(first)
        return cls(_cat_ubatch_outputs([r[i] for r in sorted_results]) for i in range(len(first)))
    if first is None:
        # 首元素 None 但并非全 None ⇒ 逐元素处理，None 位置保持 None
        n = max((len(r) for r in sorted_results if isinstance(r, (tuple, list))), default=0)
        if n == 0:
            return sorted_results
        return [_cat_ubatch_outputs([(r[i] if isinstance(r, (tuple, list)) else None)
                                     for r in sorted_results]) for i in range(n)]
    if isinstance(first, torch.Tensor):
        if any(not isinstance(x, torch.Tensor) for x in sorted_results):
            return sorted_results          # 混合类型：保守原样返回
        return torch.cat(sorted_results, dim=0)
    return sorted_results





@dataclass
class UbatchMetadata:
    context: UBatchContext
    input_ids: torch.Tensor
    positions: torch.Tensor
    inputs_embeds: torch.Tensor | None
    intermediate_tensors: IntermediateTensors | None
    num_tokens: int


class NPUUBatchContext(UBatchContext):
    """NPU version of UBatchContext that uses torch.npu Stream/Event APIs."""

    def update_stream(self, stream):
        self.current_stream = stream
        torch.npu.set_stream(self.current_stream)

    def _signal_comm_done(self):
        self.gpu_comm_done_event.record(self.comm_stream)

    def _signal_compute_done(self):
        self.gpu_compute_done_event.record(self.compute_stream)

    def _wait_compute_done(self):
        self.comm_stream.wait_event(self.gpu_compute_done_event)

    def _wait_comm_done(self):
        self.compute_stream.wait_event(self.gpu_comm_done_event)


def _make_npu_ubatch_contexts(
    num_micro_batches: int,
    compute_stream: torch.npu.Stream,
    comm_stream: torch.npu.Stream,
    forward_contexts: list,
    ready_barrier: threading.Barrier,
    schedule: str = "default",
) -> list[NPUUBatchContext]:
    """Create NPUUBatchContext instances mirroring make_ubatch_contexts."""
    import vllm.v1.worker.ubatching as _ubatching_mod

    assert num_micro_batches > 1, "num_micro_batches must be greater than 1"

    _ubatching_mod._NUM_UBATCHES = num_micro_batches
    if len(_ubatching_mod._CURRENT_CONTEXTS) < num_micro_batches:
        _ubatching_mod._CURRENT_CONTEXTS.extend([None] * (num_micro_batches - len(_ubatching_mod._CURRENT_CONTEXTS)))

    cpu_events = [threading.Event() for _ in range(num_micro_batches)]
    gpu_comm_done_events = [torch.npu.Event() for _ in range(num_micro_batches)]
    gpu_compute_done_events = [torch.npu.Event() for _ in range(num_micro_batches)]

    ctxs = []
    for i in range(num_micro_batches):
        ctx = NPUUBatchContext(
            id=i,
            compute_stream=compute_stream,
            comm_stream=comm_stream,
            forward_context=forward_contexts[i],
            ready_barrier=ready_barrier,
            cpu_wait_event=cpu_events[i],
            cpu_signal_event=cpu_events[(i + 1) % num_micro_batches],
            gpu_comm_done_event=gpu_comm_done_events[i],
            gpu_compute_done_event=gpu_compute_done_events[i],
            schedule=schedule,
        )
        ctxs.append(ctx)

    return ctxs


class NPUUBatchWrapper:
    """NPU version of UBatchWrapper for Ascend DBO support.

    Supports eager mode only (no ACL Graph capture in phase 1).
    """

    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.comm_stream = torch.npu.Stream(device=device)
        self.ready_barrier = threading.Barrier(self.vllm_config.parallel_config.num_ubatches + 1)
        self.device = device

    def __getattr__(self, key: str):
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(f"Attribute {key} not found in NPUUBatchWrapper or its runnable")

    def unwrap(self) -> Callable:
        return self.runnable

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        ubatch_slices = getattr(forward_context, "ubatch_slices", None)

        # If there's no ubatching, just run the runnable directly
        if ubatch_slices is None:
            return self.runnable(*args, **kwargs)

        input_ids = kwargs.get("input_ids") if "input_ids" in kwargs else (args[0] if len(args) > 0 else None)
        positions = kwargs.get("positions") if "positions" in kwargs else (args[1] if len(args) > 1 else None)
        intermediate_tensors = (
            kwargs.get("intermediate_tensors")
            if "intermediate_tensors" in kwargs
            else (args[2] if len(args) > 2 else None)
        )
        inputs_embeds = (
            kwargs.get("inputs_embeds") if "inputs_embeds" in kwargs else (args[3] if len(args) > 3 else None)
        )
        compute_stream = torch.npu.current_stream()

        attn_metadata = forward_context.attn_metadata
        slot_mapping = getattr(forward_context, "slot_mapping", None)
        batch_descriptor = forward_context.batch_descriptor

        # Build per-ubatch forward contexts
        has_slot_mapping = slot_mapping and isinstance(slot_mapping, list)
        forward_contexts = []
        for i, _ubatch_slice in enumerate(ubatch_slices):
            fc = create_forward_context(
                attn_metadata[i] if attn_metadata is not None else None,
                self.vllm_config,
                batch_descriptor=batch_descriptor,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=(slot_mapping[i] if has_slot_mapping and slot_mapping is not None else None),
            )
            # Copy Ascend-specific attributes from the parent forward context
            # so _EXTRA_CTX proxy works correctly inside ubatch threads.
            # Under V2 model runner these live in `additional_kwargs` rather
            # than directly on the forward_context object.
            from vllm_ascend.ascend_forward_context import _ExtraForwardContextProxy

            additional_kwargs = getattr(forward_context, "additional_kwargs", None)
            for attr in _ExtraForwardContextProxy.extra_attrs:
                if hasattr(forward_context, attr):
                    setattr(fc, attr, getattr(forward_context, attr))
                elif additional_kwargs is not None and attr in additional_kwargs:
                    setattr(fc, attr, additional_kwargs[attr])
            setattr(fc, "ubatch_id", i)   # [DBO] 无线程实现：显式标注本 ubatch
            forward_contexts.append(fc)

        ubatch_ctxs = _make_npu_ubatch_contexts(
            num_micro_batches=len(ubatch_slices),
            comm_stream=self.comm_stream,
            compute_stream=compute_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier,
        )

        ubatch_metadata: list[UbatchMetadata] = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            (
                sliced_input_ids,
                sliced_positions,
                sliced_inputs_embeds,
                sliced_intermediate_tensors,
            ) = self._slice_model_inputs(
                ubatch_slice.token_slice,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors,
            )
            ubatch_metadata.append(
                UbatchMetadata(
                    context=ubatch_ctxs[i],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=ubatch_slice.token_slice.stop - ubatch_slice.token_slice.start,
                )
            )

        return self._run_ubatches(ubatch_metadata, self.runnable)

    def _run_ubatches(self, ubatch_metadata: list[UbatchMetadata], model: Callable) -> torch.Tensor:
        @torch.inference_mode()
        def _ubatch_thread(results, model, ubatch_metadata):
            with ubatch_metadata.context:
                model_output = model(
                    input_ids=ubatch_metadata.input_ids,
                    positions=ubatch_metadata.positions,
                    intermediate_tensors=ubatch_metadata.intermediate_tensors,
                    inputs_embeds=ubatch_metadata.inputs_embeds,
                )
            results.append((ubatch_metadata.context.id, model_output))

        results: list[tuple[int, torch.Tensor]] = []

        with override_forward_context(None):
            ubatch_threads = []
            for metadata in ubatch_metadata:
                thread = threading.Thread(
                    target=_ubatch_thread,
                    args=(results, model, metadata),
                )
                ubatch_threads.append(thread)
                thread.start()
            self.ready_barrier.wait()
            ubatch_metadata[0].context.cpu_wait_event.set()
            for thread in ubatch_threads:
                thread.join()

        sorted_results = [value for position, value in sorted(results)]
        return _cat_ubatch_outputs(sorted_results)

    # ------------------------------------------------------------------
    # 图模式（本仓补充；上游 GPU 版把线程 join 包进 torch.cuda.graph）
    #
    # 关键规则（tiny 实测）：捕获区内 `stream.wait_event(e)` 要求 `e.record(...)`
    # **也在捕获区内**。所以 fork event 必须 record 在**捕获根流**上。
    #   root.record(fork) → side.wait_event(fork) → … → side.record(join) → root.wait_event(join)
    # ------------------------------------------------------------------
    def _run_ubatches_graph(self, ubatch_metadata, model, root_stream):
        @torch.inference_mode()
        def _submit(metadata, model, stream):
            ctx = metadata.context
            ctx.forward_context  # noqa: B018  (保持引用)
            fc_prev = None
            try:
                from vllm.forward_context import override_forward_context
                cm = override_forward_context(ctx.forward_context)
                cm.__enter__()
                fc_prev = cm
                model(
                    input_ids=metadata.input_ids,
                    positions=metadata.positions,
                    intermediate_tensors=metadata.intermediate_tensors,
                    inputs_embeds=metadata.inputs_embeds,
                )
            finally:
                if fc_prev is not None:
                    fc_prev.__exit__(None, None, None)

        side = torch.npu.Stream()
        e_fork = torch.npu.Event()
        e_join = torch.npu.Event()
        e_fork.record(root_stream)                      # ★ 必须在捕获根流上
        outputs = [None] * len(ubatch_metadata)
        with torch.npu.stream(side):
            side.wait_event(e_fork)
            _submit(ubatch_metadata[1], model, side)
            e_join.record(side)
        _submit(ubatch_metadata[0], model, root_stream)
        root_stream.wait_event(e_join)
        return None

    def supports_graph(self) -> bool:
        return True

    def _slice_model_inputs(
        self,
        tokens_slice: slice,
        input_ids,
        positions,
        inputs_embeds,
        intermediate_tensors,
    ):
        sliced_input_ids = input_ids[tokens_slice] if input_ids is not None else None
        if positions is not None:
            if positions.ndim == 2:
                sliced_positions = positions[:, tokens_slice]
            else:
                sliced_positions = positions[tokens_slice]
        else:
            sliced_positions = None
        sliced_inputs_embeds = inputs_embeds[tokens_slice] if inputs_embeds is not None else None
        sliced_intermediate_tensors = intermediate_tensors[tokens_slice] if intermediate_tensors is not None else None

        return (
            sliced_input_ids,
            sliced_positions,
            sliced_inputs_embeds,
            sliced_intermediate_tensors,
        )
