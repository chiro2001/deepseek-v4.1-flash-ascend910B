# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import ABC, abstractmethod
from typing import Generic

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed.parallel_state import get_ep_group

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import get_mc2_tokens_capacity
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.lora.fused_moe import (
    all2all_lora_indices,
    has_lora,
    postprocess_lora_indices,
    preprocess_lora_indices,
)
from vllm_ascend.lora.quant_moe import validate_quant_moe_lora_activation_input
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import (
    MoEAllGatherCombineMetadata,
    MoEAllToAllCombineMetadata,
    MoEMC2CombineMetadata,
    MoETokenDispatchInput,
    MoETokenDispatchOutput,
    TMoECombineMetadata,
)
from vllm_ascend.ops.fused_moe.moe_utils import async_all_to_all, gather_from_sequence_parallel_region
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import should_skip_allreduce_across_dp_group

# [MOE-MASK-RANGE] begin ------------------------------------------------------
import os as _os_mask_range

# V41_MOE_MASK_RANGE=1 且 expert_map 是「本地专家连成一段」的标准 EP 映射时，
# expert_map[topk_ids] != -1 可换成 (topk_ids >= first) & (topk_ids < last)，
# 省掉 aclnnIndex 的 Index + IndexCheck 两个大 kernel。
# 注意：掩码本身必须保留——expanded_row_idx 里的 -1 会让 unpermute 读到未写入的行。
_MOE_MASK_RANGE_FAST = _os_mask_range.environ.get("V41_MOE_MASK_RANGE", "0") == "1"
# [CORR-MOE-INVALID] 诊断开关（默认全关 = 与 stock 行为逐位一致）
_MOE_ZERO_INVALID = _os_mask_range.environ.get("V41_MOE_ZERO_INVALID", "0") == "1"
_MOE_INVALID_PROBE = _os_mask_range.environ.get("V41_MOE_INVALID_PROBE", "0") == "1"


class _MoeInvFd:
    fd = None


_MOE_INV_FD = _MoeInvFd()


# 运行时开关（文件轮询，0.25 s）：同会话 A/B/A 才能把"零化无效行"的影响
# 与"起服间差异"分开。文件内容 "1" 打开、"0" 关闭。
_MOE_ZERO_FILE = _os_mask_range.environ.get("V41_MOE_ZERO_INVALID_FILE", "")
_MZ = {"v": _MOE_ZERO_INVALID, "last": 0.0, "mtime": -1.0}


def _moe_zero_invalid() -> bool:
    if not _MOE_ZERO_FILE:
        return _MOE_ZERO_INVALID
    import time as _t
    now = _t.monotonic()
    if now - _MZ["last"] < 0.25:
        return _MZ["v"]
    _MZ["last"] = now
    try:
        st = _os_mask_range.stat(_MOE_ZERO_FILE)
    except OSError:
        return _MZ["v"]
    if st.st_mtime == _MZ["mtime"]:
        return _MZ["v"]
    _MZ["mtime"] = st.st_mtime
    try:
        with open(_MOE_ZERO_FILE) as _f:
            _MZ["v"] = _f.read().strip() == "1"
    except OSError:
        pass
    return _MZ["v"]
_MOE_MASK_RANGE_CACHE: dict = {}


def _eplb_is_off() -> bool:
    """EPLB（含动态 EPLB）会重排 expert_map，此时不能改用范围比较。"""
    try:
        eplb = getattr(get_ascend_config(), "eplb_config", None)
        if eplb is None:
            return True
        for _name in ("dynamic_eplb", "enable_eplb"):
            if bool(getattr(eplb, _name, False)):
                return False
        return True
    except Exception:
        return False


def _expert_map_is_contiguous_range(expert_map, first_expert_idx: int, last_expert_idx: int) -> bool:
    """纯判定（一次 host 同步）：expert_map 是否为「本地专家 = [first, last)」的连续区间映射。"""
    try:
        _m = expert_map.detach().to("cpu", torch.int32).reshape(-1)
        if _m.numel() < int(last_expert_idx):
            return False
        _expect = torch.full_like(_m, -1)
        _expect[first_expert_idx:last_expert_idx] = torch.arange(
            last_expert_idx - first_expert_idx, dtype=torch.int32)
        return bool(torch.equal(_m, _expect))
    except Exception:
        return False


def _is_contiguous_local_range(expert_map, first_expert_idx: int, last_expert_idx: int) -> bool:
    """热路径入口：EPLB 关闭 + (first, last, E) 级别的一次性内容校验（结果缓存）。

    expert_map 的布局是 (rank, EP/EPLB 配置) 级别的全局属性，同一配置下所有层的
    (first, last, E) 相同、布局也相同，所以只校验一次，不进每步热路径。
    """
    if not _eplb_is_off():
        return False
    _key = (int(first_expert_idx), int(last_expert_idx), int(expert_map.numel()))
    _cached = _MOE_MASK_RANGE_CACHE.get(_key)
    if _cached is not None:
        return _cached
    _ok = _expert_map_is_contiguous_range(expert_map, first_expert_idx, last_expert_idx)
    _MOE_MASK_RANGE_CACHE[_key] = _ok
    return _ok
# [MOE-MASK-RANGE] end --------------------------------------------------------

EXPERT_TOKEN_NUMS_TYPE_CUMSUM = 0
EXPERT_TOKEN_NUMS_TYPE_COUNT = 1


def _get_expert_token_nums_type(token_dispatch_input: MoETokenDispatchInput) -> int:
    # grouped_matmul_swiglu_quant_v2 consumes per-expert counts; existing
    # MC2 grouped-matmul paths consume prefix sums.
    if token_dispatch_input.quant.use_w4a8_per_channel_gmm_swiglu:
        return EXPERT_TOKEN_NUMS_TYPE_COUNT
    return EXPERT_TOKEN_NUMS_TYPE_CUMSUM


class MoETokenDispatcher(ABC, Generic[TMoECombineMetadata]):
    def __init__(self, **kwargs) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        self.top_k = kwargs.get("top_k", 0)
        self.num_experts = kwargs.get("num_experts", 0)
        self.lora_context = None

    def set_lora_context(self, lora_context) -> None:
        self.lora_context = lora_context

    @property
    def ep_group(self):
        """Get expert model parallel group."""
        return get_ep_group().device_group

    @property
    def ep_rank(self):
        return get_ep_group().rank_in_group

    @property
    def ep_size(self):
        return get_ep_group().world_size

    @abstractmethod
    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ) -> MoETokenDispatchOutput[TMoECombineMetadata]:
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_combine(
        self,
        hidden_states: torch.Tensor,
        combine_metadata: TMoECombineMetadata,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Combine function not implemented.")


class TokenDispatcherWithMC2(MoETokenDispatcher[MoEMC2CombineMetadata]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        device_group = get_mc2_group().device_group
        # TODO: Try local_rank = ep_group.rank_in_group
        local_rank = torch.distributed.get_rank(group=device_group)
        backend = device_group._get_backend(torch.device("npu"))
        self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
        self.ep_rank_id = get_mc2_group().rank_in_group
        self.ep_world_size = get_mc2_group().world_size
        profile = get_current_hardware_profile()
        self.need_extra_args = profile.supports(HardwareCapability.MOE_DISPATCH_EXTRA_ARGS)
        self.need_shared_expert_args = profile.supports(HardwareCapability.MOE_DISPATCH_SHARED_EXPERT_ARGS)
        self.mc2_comm_alg = get_ascend_config().get_mc2_comm_alg()

        # When enable hierarchical communication or A5 case, param `expert_scales` need to be passed in.
        self.need_expert_scale = self.need_shared_expert_args or self.mc2_comm_alg == "hierarchy"

        # Here we need to calculate the global_bs = max_bs_per_rank * ep_world_size to execute
        # dispatch & combine operators with different input num_tokens per rank.
        vllm_config = get_current_vllm_config()
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        mc2_tokens_capacity = get_mc2_tokens_capacity()
        num_tokens_per_tp_rank = mc2_tokens_capacity // tp_size
        # Surface the per-rank capacity for CANN MegaMoe's get_symm_buffer
        # sizing (used by FusedMC2CommImpl._get_cann_symm_buffer). Without
        # this, MegaMoe falls back to hidden_states.shape[0] which jitters
        # under eager mode and forces sym-buffer rebuilds every step.
        self.max_num_tokens_per_rank = num_tokens_per_tp_rank
        _max_global_bs = num_tokens_per_tp_rank * self.ep_world_size

        # When allreduce across DP is not skipped, tokens are uniform across ranks:
        # use global_bs=0 (uniform mode) and pass mc2_mask.
        # When allreduce is skipped, tokens may differ per rank:
        # use the real global_bs and do NOT pass mc2_mask.
        self.global_bs = _max_global_bs if should_skip_allreduce_across_dp_group(vllm_config) else 0

    def refresh_hccl_group(self) -> None:
        """Refresh MC2 communicator metadata after HCCL groups are recreated."""
        device_group = get_mc2_group().device_group
        local_rank = torch.distributed.get_rank(group=device_group)
        backend = device_group._get_backend(torch.device("npu"))
        self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)

    def get_dispatch_mc2_kwargs(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        hidden_states = token_dispatch_input.hidden_states
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids
        expert_map = token_dispatch_input.routing.expert_map
        global_redundant_expert_num = token_dispatch_input.routing.global_redundant_expert_num
        comm_quant_mode = token_dispatch_input.quant.comm_quant_mode

        assert expert_map is not None, "expert_map is required for MC2 token dispatch."
        # NOTE: quant_mode differs by quant feature:
        # - Legacy int communication quantization uses quant_mode=2.
        # - A5 MXFP communication uses quant_mode=4.
        if comm_quant_mode is not None:
            quant_mode = comm_quant_mode
        elif token_dispatch_input.quant.dispatch_with_quant:
            quant_mode = 4 if self.need_shared_expert_args and token_dispatch_input.quant.is_mxfp else 2
        else:
            quant_mode = 0
        self.moe_expert_num = len(expert_map) + global_redundant_expert_num
        expert_token_nums_type = _get_expert_token_nums_type(token_dispatch_input)
        kwargs_mc2 = {
            "x": hidden_states,
            "expert_ids": topk_ids,
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.moe_expert_num,
            "global_bs": self.global_bs,
            "expert_token_nums_type": expert_token_nums_type,
        }
        if self.global_bs == 0:
            kwargs_mc2["x_active_mask"] = token_dispatch_input.routing.mc2_mask

        stage1_kwargs = {
            "scales": None,
            "quant_mode": quant_mode,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": self.ep_world_size,
            "ep_rank_id": self.ep_rank_id,
            "comm_alg": self.mc2_comm_alg,
        }
        if self.need_extra_args:
            stage1_kwargs.update(
                {
                    "group_tp": self.moe_all_to_all_group_name,
                    "tp_world_size": 1,
                    "tp_rank_id": 0,
                }
            )
        # Only dispatch-enabled MXFP paths pass y_dtype through MC2.
        if (
            self.need_shared_expert_args
            and (token_dispatch_input.quant.is_mxfp or token_dispatch_input.quant.is_fp8)
            and token_dispatch_input.quant.dispatch_with_quant
        ):
            y_dtype = torch.float8_e4m3fn
            if (
                token_dispatch_input.quant.mxfp is not None
                and token_dispatch_input.quant.mxfp.act_quant_type is not None
            ):
                y_dtype = token_dispatch_input.quant.mxfp.act_quant_type
            stage1_kwargs.update({"tp_world_size": 1, "tp_rank_id": 0, "y_dtype": y_dtype})
        if self.need_expert_scale:
            stage1_kwargs.update(
                {
                    "expert_scales": topk_weights.to(torch.float32),
                }
            )

        kwargs_mc2.update(stage1_kwargs)
        return kwargs_mc2

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        kwargs_mc2 = self.get_dispatch_mc2_kwargs(token_dispatch_input)
        output = torch_npu.npu_moe_distribute_dispatch_v2(**kwargs_mc2)
        # comm_stream.wait_stream(torch.npu.current_stream())
        (
            expand_x,
            dynamic_scale,
            assist_info_for_combine,
            expert_token_nums,
            ep_recv_counts,
            tp_recv_counts,
            expand_scales,
        ) = output[0:7]

        group_list_type = kwargs_mc2["expert_token_nums_type"]
        return MoETokenDispatchOutput(
            hidden_states=expand_x,
            dynamic_scale=dynamic_scale,
            group_list=expert_token_nums,
            group_list_type=group_list_type,
            combine_metadata=MoEMC2CombineMetadata(
                topk_ids=token_dispatch_input.topk_ids,
                topk_weights=token_dispatch_input.topk_weights,
                expert_map=token_dispatch_input.routing.expert_map,
                ep_recv_counts=ep_recv_counts,
                tp_recv_counts=tp_recv_counts,
                assist_info_for_combine=assist_info_for_combine,
                expand_scales=expand_scales,
                quant=token_dispatch_input.quant,
                mc2_mask=token_dispatch_input.routing.mc2_mask if self.global_bs == 0 else None,
            ),
        )

    def get_combine_mc_kwargs(self, hidden_states: torch.Tensor, combine_metadata: MoEMC2CombineMetadata):
        expert_map = combine_metadata.expert_map
        topk_ids = combine_metadata.topk_ids
        topk_weights = combine_metadata.topk_weights
        ep_recv_counts = combine_metadata.ep_recv_counts
        tp_recv_counts = combine_metadata.tp_recv_counts
        assist_info_for_combine = combine_metadata.assist_info_for_combine
        expand_scales = combine_metadata.expand_scales
        quant_type = combine_metadata.quant.quant_type
        comm_quant_mode = combine_metadata.quant.comm_quant_mode

        assert expert_map is not None
        # NOTE: quant_mode differs by quant features:
        # - A5 MXFP communication uses quant_mode=4 only for W8A8MXFP currently.
        if comm_quant_mode is not None:
            quant_mode = comm_quant_mode
        elif quant_type == QuantType.W8A8MXFP:
            quant_mode = 4
        else:
            quant_mode = 0
        kwargs_mc2 = {
            "expand_x": hidden_states,
            "expert_ids": topk_ids,
            "expert_scales": topk_weights.to(torch.float32),
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.moe_expert_num,
            "global_bs": self.global_bs,
        }
        if self.global_bs == 0:
            kwargs_mc2["x_active_mask"] = combine_metadata.mc2_mask

        if combine_metadata.quant.dispatch_with_quant:
            tp_recv_counts = torch.empty(1, dtype=torch.int32, device=hidden_states.device)

        stage3_kwargs = {
            "ep_send_counts": ep_recv_counts,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": self.ep_world_size,
            "ep_rank_id": self.ep_rank_id,
            "expand_scales": expand_scales,
            "comm_quant_mode": quant_mode,
            "comm_alg": self.mc2_comm_alg,
            "assist_info_for_combine": assist_info_for_combine,
        }

        if self.need_extra_args:
            stage3_kwargs.update(
                {
                    "tp_send_counts": tp_recv_counts,
                    "group_tp": self.moe_all_to_all_group_name,
                    "tp_world_size": 1,
                    "tp_rank_id": 0,
                }
            )

        kwargs_mc2.update(stage3_kwargs)
        return kwargs_mc2

    def token_combine(self, hidden_states, combine_metadata, bias=None):
        assert bias is None, "Bias is not supported in MoEAlltoAllvTokenDispatcher."

        kwargs_mc2 = self.get_combine_mc_kwargs(hidden_states, combine_metadata)
        combined_output = torch_npu.npu_moe_distribute_combine_v2(**kwargs_mc2)

        return combined_output


class TokenDispatcherWithAllGather(MoETokenDispatcher[MoEAllGatherCombineMetadata]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_num_tokens = kwargs.get("max_num_tokens")
        num_experts_local = kwargs.get("num_local_experts", 0)
        self.num_experts_local = (
            num_experts_local.item() if torch.is_tensor(num_experts_local) else int(num_experts_local)
        )

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        quant_type = token_dispatch_input.quant.quant_type
        dynamic_scale = token_dispatch_input.routing.pertoken_scale
        unquantized_mxfp4_dispatch = quant_type == QuantType.W4A4MXFP and dynamic_scale is None
        # Without prepare-stage scales, MXFP4 stays unquantized in dispatch and
        # is quantized again inside the MLP path.
        with_quant = token_dispatch_input.quant.dispatch_with_quant and quant_type != QuantType.W8A8FP
        with_quant = with_quant and not unquantized_mxfp4_dispatch
        if has_lora(self.lora_context) and token_dispatch_input.quant.is_quant:
            validate_quant_moe_lora_activation_input(
                quant_type=quant_type,
                hidden_states=token_dispatch_input.hidden_states,
                dynamic_scale=dynamic_scale,
            )
            with_quant = False
        is_mxfp = token_dispatch_input.quant.is_mxfp
        hidden_states = token_dispatch_input.hidden_states
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids
        expert_map = token_dispatch_input.routing.expert_map
        act_quant_type = (
            token_dispatch_input.quant.mxfp.act_quant_type
            if token_dispatch_input.quant.mxfp is not None and not unquantized_mxfp4_dispatch
            else None
        )
        global_redundant_expert_num = token_dispatch_input.routing.global_redundant_expert_num
        restore_shape = hidden_states.shape
        # Fuse the first dynamic quant of moe_mlp into initrouting when
        # dispatch_with_quant is on but got a None dynamic_scale.
        if with_quant and dynamic_scale is None:
            if quant_type == QuantType.W4A4MXFP:
                quant_mode = 9
            else:
                quant_mode = 3 if is_mxfp else 1
        else:
            quant_mode = -1

        num_tokens = hidden_states.shape[:-1].numel()
        apply_router_weight_on_input = token_dispatch_input.routing.apply_router_weight_on_input
        if apply_router_weight_on_input:
            assert topk_weights.dim() == 2, "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert topk == 1, "Only support topk=1 when `apply_router_weight_on_input` is True"
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
        if expert_map is not None:
            global_num_experts = len(expert_map) + global_redundant_expert_num
            first_expert_idx = get_ep_group().rank_in_group * self.num_experts_local
            last_expert_idx = first_expert_idx + self.num_experts_local
            # [MOE-MASK-RANGE] 连续区间映射下 expert_map[topk_ids] != -1 等价于
            # 范围比较；掩码本身必须保留（expanded_row_idx 的 -1 会让 unpermute
            # 读到未写入的 permuted_tokens 行，靠这里的 0 权重压掉）。
            if _MOE_MASK_RANGE_FAST and _is_contiguous_local_range(
                    expert_map, first_expert_idx, last_expert_idx):
                topk_weights = topk_weights.masked_fill(
                    (topk_ids < first_expert_idx) | (topk_ids >= last_expert_idx), 0.0)
            else:
                mask = expert_map[topk_ids] != -1
                topk_weights = topk_weights * mask
        else:
            first_expert_idx = 0
            last_expert_idx = self.num_experts_local
            global_num_experts = self.num_experts_local
        sorted_hidden_states, expanded_row_idx, expert_tokens, dynamic_scale = DeviceOperator.npu_moe_init_routing(
            hidden_states,
            topk_ids,
            scale=dynamic_scale,
            active_num=num_tokens * self.top_k,
            expert_num=global_num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[first_expert_idx, last_expert_idx],
            quant_mode=quant_mode,
            act_quant_type=act_quant_type,
        )
        expert_tokens = expert_tokens.to(torch.int64)
        group_list_type = 1  # `count` mode

        # ==== [CORR-MOE-INVALID] 第三源验证：init_routing 的未写入区间（P0-1）====
        #   线3 证据：`expanded_row_idx == -1` 的行是**内存池残留**（非本调用写入），
        #   随分配历史变化。若 with_quant=False，apply_gmm1 的 `_quant_hidden_states`
        #   会对**整个** sorted_x 调 npu_dynamic_quant ⇒ 残留含 Inf/NaN 时 scale 被污染。
        #   V41_MOE_ZERO_INVALID=1 把这些行显式清零（诊断用，不进交付）。
        _zero_now = _moe_zero_invalid()
        if _zero_now or _MOE_INVALID_PROBE:
            try:
                _inv = expanded_row_idx < 0
                if _MOE_INVALID_PROBE:
                    _n_inv = int(_inv.sum().item())
                else:
                    # [PERF] zero-only path: skip the device sync. Writing an all-False
                    # mask is a no-op, so the unconditional write below is safe.
                    _n_inv = 1
                if _MOE_INVALID_PROBE:
                    _x = sorted_hidden_states
                    _bad = ~torch.isfinite(_x.float())
                    _nb = int(_bad.any(dim=-1).sum().item())
                    _nbad_inv = int((_bad.any(dim=-1) & _inv).sum().item())
                    # print 会被 vLLM 日志吞掉 ⇒ 写文件（只追加，不删除）
                    _fd = getattr(_MOE_INV_FD, "fd", None)
                    if _fd is None:
                        import os as _oz
                        _fd = _oz.open(f"/tmp/v41_moe_inv.{_oz.getpid()}.log",
                                       _oz.O_WRONLY | _oz.O_CREAT | _oz.O_APPEND, 0o644)
                        _MOE_INV_FD.fd = _fd
                    import os as _oz2
                    # 残留幅度：invalid 行的绝对值最大/均值（判断是否是"垃圾"）
                    try:
                        _xi = _x[_inv].float().abs()
                        _imax = float(_xi.max().item()) if _xi.numel() else 0.0
                        _imean = float(_xi.mean().item()) if _xi.numel() else 0.0
                    except Exception:
                        _imax = _imean = -1.0
                    _oz2.write(_fd, (
                        f"[moe-inv] rows={_x.shape[0]} invalid={_n_inv} "
                        f"nonfinite_rows={_nb} nonfinite_in_invalid={_nbad_inv} "
                        f"invalid_absmax={_imax:.6g} invalid_absmean={_imean:.6g} "
                        f"with_quant={with_quant} scale_none={dynamic_scale is None} "
                        f"quant_mode={quant_mode}\n"
                    ).encode())
                if _zero_now and _n_inv:
                    # masked_fill_ is a single full-tensor write; it avoids the
                    # nonzero + index_put double pass of boolean indexing
                    # (prefill has M=12288, H=5120).
                    sorted_hidden_states.masked_fill_(_inv.unsqueeze(-1), 0)
                    if dynamic_scale is not None:
                        try:
                            dynamic_scale.masked_fill_(_inv, 0)
                        except Exception:
                            pass
            except Exception:
                pass

        return MoETokenDispatchOutput(
            hidden_states=sorted_hidden_states,
            dynamic_scale=dynamic_scale if with_quant else None,
            group_list=expert_tokens,
            group_list_type=group_list_type,
            combine_metadata=MoEAllGatherCombineMetadata(
                topk_weights=topk_weights,
                expanded_row_idx=expanded_row_idx,
                restore_shape=restore_shape,
            ),
        )

    def token_combine(self, hidden_states, combine_metadata, bias=None):
        final_hidden_states = DeviceOperator.npu_moe_token_unpermute(
            permuted_tokens=hidden_states,
            sorted_indices=combine_metadata.expanded_row_idx,
            probs=combine_metadata.topk_weights.to(hidden_states.dtype),
        )
        if len(combine_metadata.restore_shape) == 3:
            final_hidden_states = final_hidden_states.view(combine_metadata.restore_shape)

        # these values are no longer used, so they need to be set to None for memory release.
        return final_hidden_states


class TokenDispatcherWithAll2AllV(MoETokenDispatcher[MoEAllToAllCombineMetadata]):
    """
    The implementation of the AlltoAll-based token dispatcher, which handles token
    dispatching on the sequence level instead of token level. The core of this implementation
    lies in each device dispatching on the entire sequence, with the hidden state being partitioned.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_local_experts = kwargs.get("num_local_experts", 0)

        assert self.num_local_experts > 0, "Expected at least one expert"
        if self.num_local_experts > 1:
            self.expert_ids_per_ep_rank = torch.tensor(
                [i % self.num_local_experts for i in range(self.num_experts)],
                dtype=torch.int32,
                device=torch.npu.current_device(),
            )

        local_expert_indices_offset = self.ep_rank * self.num_local_experts

        self.local_expert_indices = [local_expert_indices_offset + i for i in range(self.num_local_experts)]
        assert len(self.local_expert_indices) == self.num_local_experts, "Invalid local expert indices"
        for i in range(len(self.local_expert_indices) - 1):
            assert self.local_expert_indices[i] == self.local_expert_indices[i + 1] - 1, (
                "local_expert_indices must be continuous"
            )

        # TODO: Try local_rank = ep_group.rank_in_group
        local_rank = torch.distributed.get_rank(group=self.ep_group)
        backend = self.ep_group._get_backend(torch.device("npu"))
        self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        use_mxfp_quant = token_dispatch_input.quant.is_mxfp
        with_quant = token_dispatch_input.quant.dispatch_with_quant
        if has_lora(self.lora_context) and token_dispatch_input.quant.is_quant:
            validate_quant_moe_lora_activation_input(
                quant_type=token_dispatch_input.quant.quant_type,
                hidden_states=token_dispatch_input.hidden_states,
                dynamic_scale=token_dispatch_input.routing.pertoken_scale,
            )
            # LoRA A requires the original BF16/FP16 activations. The W8A8
            # LoRA backend performs dynamic quantization immediately before
            # each local expert GMM, after the AlltoAll exchange.
            with_quant = False
        dst_type = token_dispatch_input.quant.get_dst_type
        hidden_states = token_dispatch_input.hidden_states
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids

        (
            permutated_local_input_tokens,
            reversed_local_input_permutation_mapping,
            tokens_per_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            hidden_shape,
            hidden_shape_before_permute,
        ) = self._dispatch_preprocess(hidden_states, topk_ids)

        dynamic_scale_after_all2all = None
        if with_quant:
            permutated_local_input_tokens, dynamic_scale = DeviceOperator.npu_dynamic_quant(
                permutated_local_input_tokens, act_quant_type=dst_type, use_mxfp_quant=use_mxfp_quant
            )
            _, dynamic_scale_after_all2all, permute2_ep_all_to_all_handle = async_all_to_all(
                dynamic_scale, output_splits, input_splits, self.ep_group
            )
            permute2_ep_all_to_all_handle.wait()
            dynamic_scale.untyped_storage().resize_(0)

        _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            permutated_local_input_tokens, output_splits, input_splits, self.ep_group
        )
        permute1_ep_all_to_all_handle.wait()
        permutated_local_input_tokens.untyped_storage().resize_(0)

        if self.lora_context is not None:
            all2all_lora_indices(
                self.lora_context,
                output_splits=output_splits,
                input_splits=input_splits,
                ep_group=self.ep_group,
            )

        # Postprocess
        global_input_tokens, dynamic_scale_final, reversed_global_input_permutation_mapping = (
            self._dispatch_postprocess(
                global_input_tokens,
                dynamic_scale_after_all2all,
                global_input_tokens_local_experts_indices,
                with_quant,
                dst_type,
            )
        )

        return MoETokenDispatchOutput(
            hidden_states=global_input_tokens,
            dynamic_scale=dynamic_scale_final,
            group_list=tokens_per_expert,
            group_list_type=1,
            combine_metadata=MoEAllToAllCombineMetadata(
                input_splits=input_splits,
                output_splits=output_splits,
                topk_weights=topk_weights,
                reversed_local_input_permutation_mapping=reversed_local_input_permutation_mapping,
                reversed_global_input_permutation_mapping=reversed_global_input_permutation_mapping,
                hidden_shape=hidden_shape,
                hidden_shape_before_permute=hidden_shape_before_permute,
            ),
        )

    def token_combine(self, hidden_states, combine_metadata, bias=None):
        assert bias is None, "Bias is not supported in MoEAlltoAllvTokenDispatcher."

        # 1. Preprocess using metadata
        hidden_states = self._combine_preprocess(hidden_states, combine_metadata)

        # 2. AllToAll
        _, permutated_local_input_tokens, handle = async_all_to_all(
            hidden_states,
            combine_metadata.input_splits,
            combine_metadata.output_splits,
            self.ep_group,
        )
        handle.wait()
        hidden_states.untyped_storage().resize_(0)

        # 3. Postprocess using metadata
        output = self._combine_postprocess(permutated_local_input_tokens, combine_metadata)

        return output

    def _dispatch_preprocess(self, hidden_states, topk_ids):
        hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        (
            tokens_per_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            num_out_tokens,
        ) = self._preprocess(topk_ids)
        hidden_shape_before_permute = hidden_states.shape

        permutated_local_input_tokens, reversed_local_input_permutation_mapping = torch_npu.npu_moe_token_permute(
            tokens=hidden_states,
            indices=topk_ids,
            num_out_tokens=num_out_tokens,
        )

        if self.lora_context is not None:
            preprocess_lora_indices(
                self.lora_context,
                topk_ids=topk_ids,
                reversed_permutation_mapping=reversed_local_input_permutation_mapping,
            )

        return (
            permutated_local_input_tokens,
            reversed_local_input_permutation_mapping,
            tokens_per_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            hidden_shape,
            hidden_shape_before_permute,
        )

    def _preprocess(self, topk_ids: torch.Tensor):
        num_local_tokens_per_expert = torch.histc(topk_ids, bins=self.num_experts, min=0, max=self.num_experts)

        ep_size = self.ep_size
        num_out_tokens = topk_ids.numel()

        input_splits = (
            num_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
            .sum(axis=1)
            .to(torch.device("cpu"), non_blocking=True)
            .numpy()
        )

        num_global_tokens_per_expert = gather_from_sequence_parallel_region(
            num_local_tokens_per_expert, group=self.ep_group
        ).reshape(ep_size, self.num_experts)
        num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ]
        if num_global_tokens_per_local_expert is None:
            raise ValueError("num_global_tokens_per_local_expert must be set before sum.")

        output_splits = (
            num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu"), non_blocking=True).numpy()
        )
        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0)

        global_input_tokens_local_experts_indices = None
        if self.num_local_experts > 1:
            if num_global_tokens_per_local_expert is None:
                raise ValueError("num_global_tokens_per_local_expert must be set before operations.")
            global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel()
            )
        else:
            torch.npu.synchronize()

        return (
            num_tokens_per_local_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            num_out_tokens,
        )

    def _dispatch_postprocess(
        self,
        global_input_tokens,
        dynamic_scale_after_all2all,
        global_input_tokens_local_experts_indices,
        with_quant,
        dst_type,
    ):
        # Early return if no local experts or no tokens
        if self.num_local_experts <= 1:
            return global_input_tokens, dynamic_scale_after_all2all, None

        assert global_input_tokens_local_experts_indices is not None, (
            "global_input_tokens_local_experts_indices must be provided"
        )

        if with_quant:
            global_input_tokens, reversed_global_input_permutation_mapping, _, dynamic_scale_after_all2all = (
                torch_npu.npu_moe_init_routing_v2(
                    global_input_tokens,
                    global_input_tokens_local_experts_indices.unsqueeze(-1),
                    scale=dynamic_scale_after_all2all,
                    expert_num=self.num_experts,
                    expert_tokens_num_flag=True,
                    active_expert_range=[0, self.num_local_experts],
                    x_dtype=dst_type,
                )
            )
        else:
            global_input_tokens, reversed_global_input_permutation_mapping = torch_npu.npu_moe_token_permute(
                global_input_tokens, global_input_tokens_local_experts_indices
            )
        if self.lora_context is not None:
            postprocess_lora_indices(
                self.lora_context,
                reversed_permutation_mapping=reversed_global_input_permutation_mapping,
            )
        return global_input_tokens, dynamic_scale_after_all2all, reversed_global_input_permutation_mapping

    def _combine_preprocess(
        self, hidden_states: torch.Tensor, combine_metadata: MoEAllToAllCombineMetadata
    ) -> torch.Tensor:
        # Unpermutation 2: expert output to AlltoAll input
        rev_global = combine_metadata.reversed_global_input_permutation_mapping
        if hidden_states.shape[0] > 0 and self.num_local_experts > 1 and rev_global is not None:
            hidden_states = torch_npu.npu_moe_token_unpermute(hidden_states, rev_global)
        return hidden_states

    def _combine_postprocess(
        self,
        permutated_local_input_tokens: torch.Tensor,
        combine_metadata: MoEAllToAllCombineMetadata,
    ) -> torch.Tensor:
        # Unpermutation 1: AlltoAll output to output
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=permutated_local_input_tokens,
            sorted_indices=combine_metadata.reversed_local_input_permutation_mapping.to(torch.int32),
            probs=combine_metadata.topk_weights.to(permutated_local_input_tokens.dtype),
            restore_shape=combine_metadata.hidden_shape_before_permute,
        )
        output = output.view(combine_metadata.hidden_shape)
        return output
