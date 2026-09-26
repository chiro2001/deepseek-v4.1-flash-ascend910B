# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 text model and source-shared hybrid-cache graph."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import os as _os_ids

import os as _os_egd

# [ENGRAM-GATE-HOIST] ----------------------------------------------------
import os as _os_engram_hoist

_ENGRAM_GATE_HOIST = _os_engram_hoist.environ.get("V41_ENGRAM_GATE_HOIST", "0") == "1"
_ENGRAM_GATE_CACHE: dict = {}


def _engram_gate_version_key(tensor):
    # id() 抓「换了张量」；_version 抓「就地改过」（copy_ 会 bump）。
    return (id(tensor), int(getattr(tensor, "_version", -1)), tuple(tensor.shape), tensor.dtype)


def _engram_gate_channel_weight(q_weight, k_weight):
    """q_weight.float() * k_weight.float()：加载后不变的静态权重，按版本缓存。"""
    if not _ENGRAM_GATE_HOIST:
        return q_weight.float() * k_weight.float()          # stock
    key = ("cw", _engram_gate_version_key(q_weight), _engram_gate_version_key(k_weight))
    cached = _ENGRAM_GATE_CACHE.get(key)
    if cached is None:
        cached = q_weight.float() * k_weight.float()
        _ENGRAM_GATE_CACHE[key] = cached
    return cached


def _engram_gate_rotation_f32(rotation):
    """engram_rotation 的 fp32 版本（常量块对角）；engram_gate 内部还会 .float()，
    对 fp32 输入那是 no-op，因此数值与 stock 相同、少一次 cast。"""
    if not _ENGRAM_GATE_HOIST:
        return rotation                                     # stock
    key = ("rot", _engram_gate_version_key(rotation))
    cached = _ENGRAM_GATE_CACHE.get(key)
    if cached is None:
        cached = rotation.float()
        _ENGRAM_GATE_CACHE[key] = cached
    return cached
# [ENGRAM-GATE-HOIST] ----------------------------------------------------

import torch
import numpy as _np

_ENGRAM_WITH_DUMMY = _os_egd.environ.get("V41_ENGRAM_WITH_DUMMY", "0") == "1"
_ENGRAM_PAD_SKIP = _os_egd.environ.get("V41_ENGRAM_PAD_SKIP", "0") == "1"

_IDS64_HOIST = _os_ids.environ.get("V41_IDS64_HOIST", "0") == "1"
_CED_PREFILL_ROLE = _os_ids.environ.get("V41_CED_ROLE", "") == "prefill"
_CED_H20_SNAPSHOT_POS = _os_ids.environ.get("V41_CED_H20_SNAPSHOT_POS", "")
_CED_H20_SNAPSHOT_DIR = _os_ids.environ.get("V41_CED_H20_SNAPSHOT_DIR", "")
_CED_LAYER_SNAPSHOT_POS = _os_ids.environ.get("V41_CED_LAYER_SNAPSHOT_POS", "")
_CED_LAYER_SNAPSHOT_DIR = _os_ids.environ.get("V41_CED_LAYER_SNAPSHOT_DIR", "")
_CED_LAYER_SNAPSHOT_LAYERS = {
    int(value)
    for value in _os_ids.environ.get("V41_CED_LAYER_SNAPSHOT_LAYERS", "0,1,2,13,14,15,19,20").split(",")
    if value.strip()
}
_CED_CAPTURE_DECODE = _os_ids.environ.get("V41_CED_CAPTURE_DECODE", "0") == "1"
from safetensors import safe_open
from transformers import AutoTokenizer
from vllm.distributed import get_pp_group, get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.dsa_v41 import (
    DeepseekV41CacheBackend,
    DeepseekV41CacheLayer,
    DeepseekV41EagerAttentionImpl,
)
from vllm_ascend.core.deepseek_v41 import (
    DeepseekV41FullSpec,
    DeepseekV41SWASpec,
    validate_cache_runtime,
)
from vllm_ascend.models.deepseek_v4.model import (
    AscendDeepseekV4ForCausalLM,
    AscendDeepseekV4SWACache,
    DeepseekV2DecoderLayer,
    DeepseekV4Attention,
    DeepseekV4Model,
)

from .compressor import DeepseekV41Compressor, _read, text_config_of
from .engram_gate import engram_gate
from .engram_hash import PagedNgramHistory
from .engram_hbm import EngramQueryGroup, NodeShardedEngram


def _maybe_snapshot_ced_h20(layer, positions, hidden_states, pre_mix):
    """Export the exact layer-20 input row for an isolated numeric comparison.

    H20 includes the mHC streams and their FP32 mixing coefficients.  The
    diagnostic is opt-in and reads one active prefill row outside graph capture.
    It does not change either tensor or the model's forward result.
    """
    if not _CED_H20_SNAPSHOT_POS or not _CED_H20_SNAPSHOT_DIR:
        return
    context = get_forward_context()
    if (
        context.attn_metadata is None
        or getattr(context, "capturing", False)
        or getattr(context, "in_profile_run", False)
    ):
        return
    metadata = layer.self_attn.v41_impl._get_layer_metadata(context.attn_metadata)
    if metadata.swa.num_prefills == 0 and not _CED_CAPTURE_DECODE:
        return
    target = int(_CED_H20_SNAPSHOT_POS)
    active = metadata.swa.num_actual_tokens
    found = (positions[:active] == target).nonzero(as_tuple=False).flatten().cpu().tolist()
    if not found:
        return
    if len(found) != 1:
        raise RuntimeError(f"CED H20 snapshot position {target} appears {len(found)} times")
    rank = get_tensor_model_parallel_rank()
    output_dir = Path(_CED_H20_SNAPSHOT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"rank{rank}_pos{target}.npz"
    if output.exists():
        return
    row = found[0]
    _np.savez_compressed(
        output,
        position=target,
        rank=rank,
        role=_os_ids.environ.get("V41_CED_ROLE", "baseline"),
        hidden_states=hidden_states[row].detach().float().cpu().numpy().copy(),
        pre_mix=pre_mix[row].detach().float().cpu().numpy().copy(),
    )
    print(f"[CED-H20] snapshot rank={rank} position={target} path={output}", flush=True)


def _maybe_snapshot_ced_layer(
    layer, positions, input_ids, hidden_states, pre_mix, stage, lookup=None, token_mask=None
):
    """Capture one active token around selected encoder and Engram layers."""
    if (
        not _CED_LAYER_SNAPSHOT_POS
        or not _CED_LAYER_SNAPSHOT_DIR
        or layer.layer_idx not in _CED_LAYER_SNAPSHOT_LAYERS
    ):
        return
    context = get_forward_context()
    if (
        context.attn_metadata is None
        or getattr(context, "capturing", False)
        or getattr(context, "in_profile_run", False)
    ):
        return
    metadata = layer.self_attn.v41_impl._get_layer_metadata(context.attn_metadata)
    if metadata.swa.num_prefills == 0 and not _CED_CAPTURE_DECODE:
        return
    target = int(_CED_LAYER_SNAPSHOT_POS)
    active = metadata.swa.num_actual_tokens
    found = (positions[:active] == target).nonzero(as_tuple=False).flatten().cpu().tolist()
    if not found:
        return
    if len(found) != 1:
        raise RuntimeError(f"CED layer snapshot position {target} appears {len(found)} times")
    rank = get_tensor_model_parallel_rank()
    output_dir = Path(_CED_LAYER_SNAPSHOT_DIR) / f"rank{rank}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"layer{layer.layer_idx:02d}_{stage}_pos{target}.npz"
    if output.exists():
        return
    row = found[0]
    snapshot = dict(
        position=target,
        layer=layer.layer_idx,
        stage=stage,
        tp_rank=rank,
        role=_os_ids.environ.get("V41_CED_ROLE", ""),
        input_id=int(input_ids[row].detach().cpu().item()),
        hidden_states=hidden_states[row].detach().float().cpu().numpy().copy(),
        pre_mix=pre_mix[row].detach().float().cpu().numpy().copy(),
    )
    if lookup is not None:
        snapshot["engram_lookup"] = lookup[row].detach().float().cpu().numpy().copy()
    if token_mask is not None:
        snapshot["engram_token_mask"] = bool(token_mask[row].detach().cpu().item())
    _np.savez_compressed(output, **snapshot)
    print(
        f"[CED-LAYER] snapshot rank={rank} layer={layer.layer_idx} "
        f"stage={stage} position={target} path={output}",
        flush=True,
    )

# ==== [DEVICE-INDEX] Engram 端到端设备化 =====================================
# V41_ENGRAM_DEVICE_INDEX=1 时，查表不再经过 host：
#   * 表仍常驻 host DRAM（206 GB 放不进 HBM），但经 aclrtHostRegister 映射成
#     设备可寻址，由 device 算子直接索引，省掉 d2h 同步 / 分片 / all_gather /
#     all_to_all / broadcast / h2d 五条路径；
#   * 4-gram 页镜像与哈希改用 dense device 张量（DeviceNgramHash），
#     语义与 PagedNgramHistory.update 逐位一致（12 个场景 x 2 种 host 实现已对账）；
#   * 因为每个 rank 都能读整张表，不再需要 local-owner 分片。
# 实测依据见 lite-port/engram-probe/COMPARE-device-vs-host.md。
# 三态，而不是布尔：
#   auto（默认）—— 起服时探测 aclrtHostRegister 能力，支持才启用，否则回退 host 路径
#   1          —— 强制启用；探测失败即抛错（用于 A3 验收，避免"以为开了其实没开"）
#   0          —— 强制关闭
# 之所以默认 auto：device-index 的硬件前提只在 A3 实测过，A2（910B3）从未验证，
# 而项目自己的文档记着 A2 上"host 地址可被 device kernel 直接读"**不成立的可能**
# （docs/A2_VS_A3_DIFF.md §5）。默认 1 会让 A2 起不来或静默出错，默认 0 又等于
# A3 白拿不到收益 —— auto 同时满足两者。
_ENGRAM_DEVICE_INDEX_MODE = (
    _os_engram_hoist.environ.get("V41_ENGRAM_DEVICE_INDEX", "auto") or "auto"
).strip().lower()
_ENGRAM_DEVICE_INDEX = _ENGRAM_DEVICE_INDEX_MODE not in ("0", "false", "off", "no", "")
# 每步失败时是否回退到 host 路径。**默认不回退**：回退要求 host 分片仍然有效
# （见 engram_hbm.py 的 _ENGRAM_DEVICE_TRIM_SHARD），而默认构建为了省掉
# 25.75 GB/rank/层 的 DRAM 把分片裁掉了 —— 此时回退会读到没装数据的表并**静默出错**。
# 所以走"全有或全无"：失败就抛，消息里带原因。确实需要回退时两个都设：
#   V41_ENGRAM_DEVICE_FALLBACK=1   （保留完整分片）
#   V41_ENGRAM_DEVICE_INDEX=1
_ENGRAM_DEVICE_FALLBACK = _os_engram_hoist.environ.get("V41_ENGRAM_DEVICE_FALLBACK", "0") == "1"
# 页表容量的起始上界（页数）。真实值来自 cache_config.num_gpu_blocks，但那只在
# profile **之后**才确定，而 profile_run 用合成 block_table 就会走到这里，所以
# 必须先给一个够合成批次的容量（64 请求 x 8192 token / 128 = 4096 页，取 2 倍余量）。
# 8192 页 x 128 x 8 B = 8 MB；拿到 num_gpu_blocks 后再按需增长到 blocks+8。
_ENGRAM_DEVICE_PAGES_BOOT = int(
    _os_engram_hoist.environ.get("V41_ENGRAM_DEVICE_PAGES", "8192") or 8192
)
if _ENGRAM_DEVICE_INDEX:
    # 只在开关打开时导入：模块本身只依赖 torch/ctypes，但保持"关掉即零影响"。
    from .engram_device_index import device_engram_lookup  # noqa: F401
    from .engram_graph import EngramGraphCache  # noqa: F401
# [DEVICE-INDEX-GRAPH] 把整条设备路径放进"每个 batch size 一张图"的缓存里重放。
# 需要它是因为 prepare_engram_inputs() 在 run_model() **之前**调用，也就是在
# 模型自己的 ACLGraph 捕获之外 —— 设备路径于是以 eager 下发，光 hash 就要
# 1.5 ms（且与 batch size 无关，是 ~50 个小算子的纯下发开销）。
# 默认开。曾一度默认关：真机 TP8 起服后第一个请求报
#   RuntimeError: The previous device metadata submission has not been released
# 而误判为"图缓存与引擎的 device_metadata 冲突"。追到根因后是**我们自己的**
# 一个设备不匹配 bug（`build_request_ids` 用 CPU arange 配 device 的 repeats），
# 异常从 forward 里逃出，正好落在引擎 submit/release 之间，把那个标志永久置位；
# 后续请求全部死在 submit 上，与图缓存无关。修法只有一条，且是治本的那条：
#   * `build_request_ids` 两端同设备
# 曾额外加过 `model_runner_v1.py` 的 DMQ 自愈护栏，**已撤销**：那个判据
# （`submission_in_flight == True` 就 release）在**正常运行**中也会短时为真，
# 于是提前释放 device metadata，64 并发实测把服务打挂（57/64 成功，
# `ScatterElements 0x91` → ERR00100 → HCCL watchdog）。详见 CHANGELOG §4.3。
# 单卡实测：n=12 时 2.024 ms → 0.695 ms，捕获重放与 eager 逐位一致。
_ENGRAM_DEVICE_GRAPH = _ENGRAM_DEVICE_INDEX and (
    _os_engram_hoist.environ.get("V41_ENGRAM_DEVICE_GRAPH", "1") == "1"
)
# 图的数量上限：超过就退回 eager。decode 的 n 落在捕获桶里，prefill 的 n 很大
# 且每个请求都不同，所以上限既保护显存也保护起服时间。
_ENGRAM_DEVICE_GRAPH_MAX = int(
    _os_engram_hoist.environ.get("V41_ENGRAM_DEVICE_GRAPH_MAX", "16") or 16
)
# ==== /[DEVICE-INDEX] =======================================================

# ==== [bneck-probe] 运行时瓶颈开关（generated by make_bneck_probe.py）====
import os as _bp_os
import time as _bp_time


def _bp_rank_zero():
    try:
        import torch.distributed as _bp_dist

        return (not _bp_dist.is_initialized()) or _bp_dist.get_rank() == 0
    except Exception:
        return True


class _BneckState:
    """0.25 s 粒度轮询一个文本文件，决定本次 forward 的 Engram host 行为。"""

    def __init__(self):
        self.path = _bp_os.environ.get("V41_BNECK_MODE_FILE", "/tmp/v41_bneck_mode")
        self.mode = "stock"
        self.delay_ms = 0.0
        self.where = "post"
        self.mtime = -1.0
        self.last_check = 0.0
        self.cache = {}
        self.fake_hits = 0
        self.mirror_ok = 0
        self.mirror_bad = 0
        self.decode_gen = 0
        self.last_mode_seen = None
        self.steps = 0
        self.acc = {}
        self.dec_calls = 0
        self.cur_is_decode = False
        self.last_prep = None
        self.print_every = int(_bp_os.environ.get("V41_BNECK_PRINT_EVERY", "20") or 20)
        self.rank0 = _bp_rank_zero()
        self.refresh(force=True)

    def refresh(self, force=False):
        now = _bp_time.monotonic()
        if not force and now - self.last_check < 0.25:
            return self
        self.last_check = now
        try:
            st = _bp_os.stat(self.path)
        except OSError:
            return self
        if st.st_mtime == self.mtime:
            return self
        self.mtime = st.st_mtime
        try:
            with open(self.path) as fh:
                raw = fh.read().strip()
        except OSError:
            return self
        self.mode, self.where, self.delay_ms = self._parse(raw)
        if self.mode != self.last_mode_seen:
            # 模式切换：清窗口，避免跨模式统计混在一起（review M8）
            self.last_mode_seen = self.mode
            self.acc = {}
            self.steps = 0
            self.dec_calls = 0
            self.cache = {}
            if self.rank0 and self.print_every > 0:
                print(f"[bneck] mode-change -> {self.mode} delay={self.delay_ms} where={self.where}", flush=True)
        return self

    @staticmethod
    def _parse(raw):
        for tag in ("pre", "post", "route"):
            head = "delay" + tag
            if raw.startswith(head):
                tail = raw[len(head):]
                try:
                    return "stock", tag, float(tail)
                except ValueError:
                    return "stock", tag, 0.0
        if raw.startswith("delay"):
            tail = raw[len("delay"):]
            try:
                return "stock", "post", float(tail)
            except ValueError:
                return "stock", "post", 0.0
        return raw, None, 0.0

    def maybe_delay(self, where):
        if self.delay_ms > 0 and self.where == where:
            end = _bp_time.perf_counter() + self.delay_ms / 1000.0
            while _bp_time.perf_counter() < end:
                pass

    def stat(self, key, t0):
        dt = (_bp_time.perf_counter() - t0) * 1000.0
        if self.cur_is_decode:
            self.acc[key] = self.acc.get(key, 0.0) + dt
        return dt

    def mark_step(self):
        now = _bp_time.perf_counter()
        if self.last_prep is not None and self.cur_is_decode:
            self.acc["hp"] = self.acc.get("hp", 0.0) + (now - self.last_prep) * 1000.0
        self.last_prep = now

    def tick(self, info=""):
        self.steps += 1
        if self.cur_is_decode:
            self.dec_calls += 1
        if self.print_every <= 0 or self.steps % self.print_every:
            return
        n = float(self.dec_calls)
        if n > 0:
            body = " ".join(f"{k}={v / n:.3f}" for k, v in sorted(self.acc.items()))
            if self.rank0:
                print(f"[bneck] mode={self.mode} steps={self.steps} dec={self.dec_calls} {body} {info}", flush=True)
        self.acc = {}
        self.dec_calls = 0

    def fake_host_ids(self, input_ids, positions, n):
        """返回上一步的 host ids/positions（decode 专用；prefill 由调用方回退）。"""
        cached = self.cache.get("ids")
        # review M7：缓存必须绑定同一次 decode 序列（gen），否则跨请求同形状会串味
        if cached is None or cached[0].shape[0] != n or cached[2] != self.decode_gen:
            ids = input_ids[:n].cpu().long()
            pos = positions[:n].cpu().long()
            self.cache["ids"] = (ids, pos, self.decode_gen)
            return ids, pos
        self.fake_hits += 1
        return cached[0], cached[1]


# 判定"这一步是 decode 还是 prefill"的 token 上限：4 reqs × (S+1=8) = 32，留 2× 余量。
_BP_DECODE_TOKENS = int(_bp_os.environ.get("V41_BNECK_DECODE_TOKENS", "64") or 64)

_BP_STATE = _BneckState()


# ==== [engram-host-mirror] runner 交过来的 host input_ids 镜像 ====
# 见 probe_mirror/engram_host_mirror.py 与 model_runner_v1.py 的
# `_engram_publish_host_inputs`；未挂载 runner 侧时这里恒为 None，行为与 stock 相同。
_ENGRAM_HOST_INPUTS = {"ids": None, "pos": None, "reason": "unset"}


def set_engram_host_inputs(ids_cpu=None, pos_cpu=None, reason=None):
    """runner -> model 的交接点：``ids_cpu`` 与设备 input_ids 逐位相同或为 None。"""
    _ENGRAM_HOST_INPUTS["ids"] = ids_cpu
    _ENGRAM_HOST_INPUTS["pos"] = pos_cpu
    _ENGRAM_HOST_INPUTS["reason"] = reason


def get_engram_host_inputs():
    return _ENGRAM_HOST_INPUTS["ids"], _ENGRAM_HOST_INPUTS["pos"]


def _bp_zero_lookups(model, positions, n):
    """形状必须与 stock 的 `values.flatten(1)` 完全一致：[n, (lookup-1)*heads*width]。"""
    cfg = model.config
    columns = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads
    out = {}
    for layer_id in cfg.engram_layer_ids:
        width = model.layers[layer_id].engram.embed.width
        out[layer_id] = torch.zeros((n, columns * width), dtype=torch.bfloat16, device=positions.device)
    return out


# ==== [/bneck-probe] ====
from .indexer import DeepseekV41Indexer


@dataclass(frozen=True)
class DeepseekV41LayerRole:
    """The attention and future Engram responsibilities of one backbone layer."""

    layer_idx: int
    compress_ratio: int
    kv_source_layer: int | None
    index_source_layer: int | None
    is_kv_source: bool
    is_index_source: bool
    is_candidate_source: bool
    uses_candidate_filter: bool
    engram_slot: int | None

    @property
    def has_long_context(self) -> bool:
        return self.compress_ratio > 0


@dataclass(frozen=True)
class DeepseekV41Topology:
    """Validated, immutable model-wide source/consumer topology."""

    layers: tuple[DeepseekV41LayerRole, ...]
    kv_source_layers: tuple[int, ...]
    index_source_layers: tuple[int, ...]
    candidate_source_layer: int
    candidate_topk_blocks: int
    candidate_block_size: int
    index_topk: int

    def layer(self, layer_idx: int) -> DeepseekV41LayerRole:
        return self.layers[layer_idx]

    def kv_consumers(self, source_layer: int) -> tuple[int, ...]:
        return tuple(role.layer_idx for role in self.layers if role.kv_source_layer == source_layer)

    def index_consumers(self, source_layer: int) -> tuple[int, ...]:
        return tuple(role.layer_idx for role in self.layers if role.index_source_layer == source_layer)


class DeepseekV41SharedAttentionState:
    """Per-forward handoff between index sources and their consumer layers."""

    def __init__(self, topk_indices, candidates):
        self.topk_indices = topk_indices
        self.candidates = candidates

    def reset(self):
        # Source layers overwrite the active rows before any consumer reads
        # them. Keeping the storage intact avoids replay depending on Python
        # state mutation and preserves a fixed address for ACL Graph.
        return None


def _as_int_tuple(config: Any, name: str) -> tuple[int, ...]:
    value = _read(config, name)
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, int) for item in value):
        raise ValueError(f"DeepSeek V4.1 {name} must be a list of integers")
    return tuple(value)


def _latest_source(layer_idx: int, sources: tuple[int, ...]) -> int | None:
    return next((source for source in reversed(sources) if source <= layer_idx), None)


def build_layer_plan(config: Any) -> DeepseekV41Topology:
    """Build and validate the V4.1 layer-sharing graph from a text config.

    ``config`` may be a Transformers config object or the raw ``text_config``
    dictionary.  Extra compression ratios for speculative layers are allowed,
    but only the first ``num_hidden_layers`` entries describe the backbone.
    """

    config = text_config_of(config)
    num_layers = int(_read(config, "num_hidden_layers"))
    ratios = _as_int_tuple(config, "compress_ratios")
    kv_sources = _as_int_tuple(config, "kv_source_layers")
    index_sources = _as_int_tuple(config, "index_source_layers")
    engram_layers = _as_int_tuple(config, "engram_layer_ids")
    candidate_source = int(_read(config, "candidate_source_layer"))
    candidate_topk_blocks = int(_read(config, "candidate_topk_blocks"))
    candidate_block_size = int(_read(config, "candidate_block_size"))
    index_topk = int(_read(config, "index_topk"))

    if num_layers <= 0:
        raise ValueError("DeepSeek V4.1 num_hidden_layers must be positive")
    if len(ratios) < num_layers:
        raise ValueError(
            "DeepSeek V4.1 compress_ratios must cover every backbone layer: "
            f"got {len(ratios)} ratios for {num_layers} layers"
        )
    ratios = ratios[:num_layers]
    if any(ratio not in (0, 1, 2) for ratio in ratios):
        raise ValueError(f"DeepSeek V4.1 backbone only supports compression ratios 0, 1 and 2; got {ratios}")

    for name, sources in (("kv_source_layers", kv_sources), ("index_source_layers", index_sources)):
        if tuple(sorted(set(sources))) != sources:
            raise ValueError(f"DeepSeek V4.1 {name} must be sorted and unique")
        if any(source < 0 or source >= num_layers for source in sources):
            raise ValueError(f"DeepSeek V4.1 {name} contains a layer outside the backbone")
        if any(ratios[source] == 0 for source in sources):
            raise ValueError(f"DeepSeek V4.1 {name} cannot point to a local-only layer")

    if not set(kv_sources).issubset(index_sources):
        raise ValueError("Every DeepSeek V4.1 KV source must also be an index source")
    if candidate_source not in kv_sources:
        raise ValueError("DeepSeek V4.1 candidate_source_layer must be a KV source")
    if candidate_topk_blocks <= 0 or candidate_block_size <= 0 or index_topk <= 0:
        raise ValueError("DeepSeek V4.1 candidate and index TopK values must be positive")
    if len(set(engram_layers)) != len(engram_layers):
        raise ValueError("DeepSeek V4.1 engram_layer_ids must be unique")
    if any(layer < 0 or layer >= num_layers for layer in engram_layers):
        raise ValueError("DeepSeek V4.1 engram_layer_ids contains a layer outside the backbone")

    engram_slots = {layer_idx: slot for slot, layer_idx in enumerate(engram_layers)}
    roles: list[DeepseekV41LayerRole] = []
    for layer_idx, ratio in enumerate(ratios):
        kv_source = _latest_source(layer_idx, kv_sources) if ratio else None
        index_source = _latest_source(layer_idx, index_sources) if ratio else None
        if ratio and (kv_source is None or index_source is None):
            raise ValueError(f"DeepSeek V4.1 layer {layer_idx} has long-context attention but no source layer")
        if kv_source is not None and ratios[kv_source] != ratio:
            raise ValueError(
                f"DeepSeek V4.1 layer {layer_idx} has ratio {ratio}, but its KV source "
                f"layer {kv_source} has ratio {ratios[kv_source]}"
            )

        roles.append(
            DeepseekV41LayerRole(
                layer_idx=layer_idx,
                compress_ratio=ratio,
                kv_source_layer=kv_source,
                index_source_layer=index_source,
                is_kv_source=layer_idx in kv_sources,
                is_index_source=layer_idx in index_sources,
                is_candidate_source=layer_idx == candidate_source,
                # Consumer layers inherit the selection policy of their index
                # source.  For example, layer 26 reuses layer 24 TopK, and that
                # TopK was computed inside layer 20's candidate blocks.
                uses_candidate_filter=index_source is not None and index_source > candidate_source,
                engram_slot=engram_slots.get(layer_idx),
            )
        )

    return DeepseekV41Topology(
        layers=tuple(roles),
        kv_source_layers=kv_sources,
        index_source_layers=index_sources,
        candidate_source_layer=candidate_source,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        index_topk=index_topk,
    )


class AscendDeepseekV41SWACache(AscendDeepseekV4SWACache):
    """V4 execution-compatible SWA plane participating in V4.1 grouping."""

    def get_kv_cache_spec(self, vllm_config):
        spec = super().get_kv_cache_spec(vllm_config)
        return DeepseekV41SWASpec(
            block_size=spec.block_size,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            dtype=spec.dtype,
            sliding_window=spec.sliding_window,
            cache_dtype_str=spec.cache_dtype_str,
            model_version="deepseek_v4",
            alignment=spec.alignment,
        )

    def get_attn_backend(self):
        return DeepseekV41CacheBackend


class DeepseekV41Attention(DeepseekV4Attention):
    """V4 projections plus V4.1 source-owned cache and fused DSA execution."""

    swa_cache_cls = AscendDeepseekV41SWACache

    def __init__(
        self,
        vllm_config,
        config,
        max_position_embeddings=0,
        cache_config=None,
        quant_config=None,
        prefix="",
        topk_indices_buffer=None,
    ):
        config = text_config_of(config)
        validate_cache_runtime(vllm_config)
        layer_idx = int(prefix.split(".")[-2])
        topology = build_layer_plan(config)
        role = topology.layer(layer_idx)
        # Reuse V4's quant-aware projections and stable SWA eager backend.  A
        # zero ratio prevents V4 from creating its incompatible c4/c128 planes.
        original_ratios = config.compress_ratios
        config.compress_ratios = tuple(0 for _ in original_ratios)
        try:
            super().__init__(
                vllm_config=vllm_config,
                config=config,
                max_position_embeddings=max_position_embeddings,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            )
        finally:
            config.compress_ratios = original_ratios
        from vllm_ascend.ops.rope_dsv4 import ComplexExpRotaryEmbedding

        # V4.1 applies YaRN only to layers carrying long-context compressed KV.
        # Pure SWA layers use the unscaled base RoPE even though the allocated
        # lookup table still spans the configured maximum context length.
        self.rotary_emb = ComplexExpRotaryEmbedding(
            vllm_config=vllm_config,
            layername=f"{prefix}.attn",
            head_size=self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            is_neox_style=False,
            scaling_factor=config.rope_parameters["factor"],
            base=(config.compress_rope_theta if role.has_long_context else config.rope_theta),
            beta_fast=config.rope_parameters["beta_fast"],
            beta_slow=config.rope_parameters["beta_slow"],
            original_seq_len=(max_position_embeddings if role.has_long_context else 0),
            rope_groups=["default"],
        )
        block_size = vllm_config.cache_config.block_size
        if block_size <= 0 or block_size % 2:
            raise ValueError("V4.1 logical block_size must be a positive multiple of two")
        owned = []
        if role.is_kv_source:
            owned.extend((f"{prefix}.long_kv_cache", f"{prefix}.indexer.k_cache"))
            if role.compress_ratio == 2:
                owned.append(f"{prefix}.compressor.state_cache")
        duplicates = set(owned) & vllm_config.compilation_config.static_forward_context.keys()
        if duplicates:
            raise ValueError(f"Duplicate V4.1 cache prefixes: {sorted(duplicates)}")
        self.role = role
        self.topology = topology
        self.shared_state = None
        self.prefix = prefix
        width = _read(config, "head_dim")
        self.softmax_scale = width**-0.5
        if role.is_kv_source:
            self.long_kv_cache = DeepseekV41CacheLayer(
                vllm_config,
                f"{prefix}.long_kv_cache",
                DeepseekV41FullSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=width,
                    dtype=torch.bfloat16,
                    compress_ratio=role.compress_ratio,
                ),
            )
        self.compressor = (
            DeepseekV41Compressor(config, role.compress_ratio, vllm_config, f"{prefix}.compressor")
            if role.is_kv_source
            else None
        )
        self.indexer = (
            DeepseekV41Indexer(
                config,
                role.is_kv_source,
                vllm_config,
                f"{prefix}.indexer",
                role.compress_ratio,
                quant_config=quant_config,
            )
            if role.is_index_source
            else None
        )
        root = prefix.rsplit(".layers.", 1)[0]
        source = f"{root}.layers.{role.kv_source_layer}.self_attn"
        self.long_kv_source_prefix = f"{source}.long_kv_cache" if role.has_long_context else None
        self.index_k_source_prefix = f"{source}.indexer.k_cache" if role.has_long_context else None
        self.index_source_layer = role.index_source_layer
        self.v41_impl = DeepseekV41EagerAttentionImpl(
            prefix=prefix,
            role=role,
            topology=topology,
            long_kv_source_prefix=self.long_kv_source_prefix,
            index_k_source_prefix=self.index_k_source_prefix,
        )
        self.v41_layer_name = f"{prefix}.v41_attn"
        context = vllm_config.compilation_config.static_forward_context
        if self.v41_layer_name in context:
            raise ValueError(f"Duplicate V4.1 attention layer: {self.v41_layer_name}")
        context[self.v41_layer_name] = self

    def forward(self, positions, hidden_states, llama_4_scaling=None):
        output = torch.empty_like(hidden_states)
        torch.ops.vllm.dsa_v41_forward(hidden_states, output, self.v41_layer_name)
        return output

    def write_global_source_only(self, normalized_hidden_states):
        """Write layer 20's CSA2 source without running its query or attention.

        CED prefill obtains this layer's main KV and Indexer K from the final
        encoder state.  Reuse the regular attention implementation's writer so
        the cache layout, RoPE and slot mapping stay identical.  This method
        deliberately produces no SWA KV, attention output or logits; callers
        must use it only inside a dedicated producer phase.
        """
        if self.role.layer_idx != 20 or not self.role.is_kv_source or self.role.compress_ratio != 1:
            raise ValueError("CED source-only write requires the ratio-1 Full layer at index 20")
        forward_context = get_forward_context()
        if forward_context.attn_metadata is None:
            return 0
        metadata = self.v41_impl._get_layer_metadata(forward_context.attn_metadata)
        num_tokens = metadata.swa.num_actual_tokens
        if not num_tokens:
            return 0
        positions = metadata.positions[:num_tokens]
        cos, sin = metadata.rope(self.rotary_emb.layername, num_tokens)
        self.v41_impl._write_compressed_source(
            self,
            normalized_hidden_states[:num_tokens],
            positions,
            cos,
            sin,
            metadata,
        )
        return num_tokens


class DeepseekV41DecoderLayer(DeepseekV2DecoderLayer):
    """V4.1 block with the checkpoint's delayed mHC coefficient handoff."""

    attention_cls = DeepseekV41Attention

    def __init__(self, vllm_config, prefix, **kwargs):
        super().__init__(vllm_config, prefix, **kwargs)
        config = vllm_config.model_config.hf_config
        engram_enabled = get_ascend_config().enable_engram
        if engram_enabled and self.layer_idx in config.engram_layer_ids:
            self.engram = torch.nn.Module()
            self.engram.wkv = torch.nn.Linear(
                (config.engram_max_ngram_size - 1) * config.engram_n_heads * config.engram_head_dim,
                (config.hc_mult + 1) * config.hidden_size,
                bias=False,
                dtype=torch.bfloat16,
            )
            self.engram.q_weight = torch.nn.Parameter(
                torch.empty(config.hc_mult, config.hidden_size, dtype=torch.bfloat16)
            )
            self.engram.k_weight = torch.nn.Parameter(
                torch.empty(config.hc_mult, config.hidden_size, dtype=torch.bfloat16)
            )
        else:
            self.engram = None

    @staticmethod
    def hc_collapse(x, pre_mix):
        return (pre_mix.unsqueeze(-1) * x.float()).sum(-2).to(x.dtype)

    def hc_pre(self, x, hc_fn, hc_scale, hc_base, pre_mix=None):
        return torch.ops._C_ascend.npu_hc_pre_v2(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            pre_mix,
            hc_mult=self.hc_mult,
            hc_sinkhorn_iters=self.hc_sinkhorn_iters,
            norm_eps=self.norm_eps,
            hc_eps=self.hc_eps,
        )

    def hc_post(self, x, residual, post, comb):
        return torch.ops._C_ascend.npu_hc_post(
            x.unsqueeze(0),
            residual.unsqueeze(0),
            post.unsqueeze(0),
            comb.unsqueeze(0),
        ).squeeze(0)

    def write_global_source_from_encoder(self, hidden_states, pre_mix):
        """Project the CED decoder's global source from encoder output H20.

        The normal layer-20 forward feeds ``hc_pre -> input_layernorm`` into
        attention before writing its source.  Preserve that exact input path;
        the surrounding layer's SWA, MoE and mHC post are not run here.
        """
        if self.layer_idx != 20:
            raise ValueError("CED decoder source projection belongs to layer 20")
        x, _, _, _ = self.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            pre_mix,
        )
        x = self.input_layernorm(x)
        return self.self_attn.write_global_source_only(x)

    def forward(
        self,
        positions,
        hidden_states,
        pre_mix,
        llama_4_scaling=None,
        input_ids=None,
    ):
        if self.layer_idx >= 20 and _CED_PREFILL_ROLE:
            raise RuntimeError("CED producer executed a decoder layer instead of source-only projection")
        residual = hidden_states
        x, attn_post, attn_comb, attn_pre = self.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            pre_mix,
        )
        x = self.input_layernorm(x)
        x = self.self_attn(positions, x, llama_4_scaling)
        hidden_states = self.hc_post(x, residual, attn_post, attn_comb)

        residual = hidden_states
        x, ffn_post, ffn_comb, ffn_pre = self.hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            attn_pre,
        )
        x, x_fp32 = self.rms_norm_cast(x)
        x = self.mlp(x, input_ids=input_ids, hidden_states_fp32=x_fp32)
        hidden_states = self.hc_post(x, residual, ffn_post, ffn_comb)
        return hidden_states, ffn_pre


class DeepseekV41Model(DeepseekV4Model):
    """Single V4.1 backbone entry, matching ``deepseek_v4/model.py``."""

    decoder_layer_cls = DeepseekV41DecoderLayer

    def __init__(self, *, vllm_config, prefix=""):
        if (
            get_ascend_config().enable_engram
            and vllm_config.load_config.load_format != "dummy"
            and vllm_config.load_config.safetensors_load_strategy != "lazy"
        ):
            raise ValueError("Engram HBM shards require --safetensors-load-strategy lazy")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # V4.1 collapses with the last block's ffn_pre; it has no hc_head
        # projection in the checkpoint.
        del self.hc_head_fn, self.hc_head_base, self.hc_head_scale, self.hc_norm
        topology = build_layer_plan(self.config)
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        candidate_buffer = torch.full(
            (max_tokens, 1, topology.candidate_topk_blocks),
            -1,
            dtype=torch.int32,
            device=self.topk_indices_buffer.device,
        )
        self.candidate_indices_buffer = candidate_buffer
        self.shared_attention_state = DeepseekV41SharedAttentionState(
            self.topk_indices_buffer,
            candidate_buffer,
        )
        # Experimental producer role. The public P/D proxy discards the P
        # response; this role must only serve that internal transfer request.
        # DSpark is disabled until its draft cache has a dedicated D-side
        # initialization protocol.
        ced_role = _os_ids.environ.get("V41_CED_ROLE", "")
        if ced_role not in ("", "prefill", "decode"):
            raise ValueError(f"Unsupported V41_CED_ROLE={ced_role!r}")
        self._ced_prefill_only = ced_role == "prefill"
        # [CED-DSPARK] 2026-09-26：把一刀切禁令按角色拆开。
        #
        #   * prefill：**架构性不可行**，永远拒绝。DSpark 的 aux hidden state 取自
        #     目标层 37/38/39（config.json 的 dspark_target_layer_ids=[37,38,39]
        #     → eagle3_utils 转成 1-based [38,39,40] → 命中 layer_idx 37/38/39），
        #     而 CED 的 P 在第 20 层就 break：这三层的残差在 P 上物理不存在。
        #     runner 又因为 dspark 强制 use_aux_hidden_state_outputs=True 而无条件
        #     解包两个返回值 ⇒ aux=[] ⇒ 启动即崩。
        #   * decode：**默认放行**（2026-09-27 起 DSpark 是交付口径）。要退回
        #     旧行为就把 SPEC 设成 0；显式 `V41_CED_ALLOW_DSPARK=0` 表示
        #     "我明确不要 DSpark"，此时仍然拒绝 —— 这是为了抓"env 与 SPEC
        #     互相矛盾"的配置（报在这里比让它跑起来更容易定位）。
        if ced_role == "prefill" and vllm_config.speculative_config is not None:
            raise ValueError(
                "V41_CED_ROLE=prefill requires SPEC=0: DSpark consumes the residual "
                "streams entering target layers 37/38/39, which the layers 0..19 "
                "producer never executes"
            )
        if (
            ced_role == "decode"
            and vllm_config.speculative_config is not None
            and _os_ids.environ.get("V41_CED_ALLOW_DSPARK", "1") != "1"
        ):
            raise ValueError(
                "V41_CED_ROLE=decode got SPEC!=0 (DSpark enabled) together with "
                "V41_CED_ALLOW_DSPARK=0. These contradict each other: DSpark is the "
                "delivered default on decode, so drop the switch, or set SPEC=0 to "
                "run without it."
            )
        if self._ced_prefill_only:
            print("[CED-P] internal producer: layers 0..19 plus layer-20 global source; response is a transfer marker", flush=True)
        # Development gate: compare the isolated CED layer-20 source write with
        # ordinary layer-20 forwards. A bounded chunk count lets a real long
        # prefill exercise the 8192-token chunk boundaries without logging
        # every subsequent request in a long-running service.
        self._ced_source_compare_remaining = (
            int(_os_ids.environ.get("V41_CED_SOURCE_COMPARE_CHUNKS", "1"))
            if _os_ids.environ.get("V41_CED_SOURCE_COMPARE", "0") == "1"
            else 0
        )
        if self._ced_source_compare_remaining < 0:
            raise ValueError("V41_CED_SOURCE_COMPARE_CHUNKS must be nonnegative")
        if self._ced_prefill_only and self._ced_source_compare_remaining:
            raise ValueError("CED source comparison needs the normal layer-20 forward")
        self._ced_source_compare_count = 0
        for layer in self.layers:
            if isinstance(layer, DeepseekV41DecoderLayer):
                layer.self_attn.shared_state = self.shared_attention_state
        self.engram_root = vllm_config.model_config.model
        config = self.config
        # Target storage is a loader/runtime choice.  Checkpoint metadata is
        # used only by load_checkpoint to validate the source representation.
        # Read the storage choice after AscendConfig validation.
        ascend_config = get_ascend_config()
        storage_format = ascend_config.engram_storage
        if ascend_config.enable_engram:
            query_group = EngramQueryGroup.from_vllm(vllm_config.parallel_config)
            for layer_id, rows in zip(config.engram_layer_ids, config.engram_num_embeddings):
                self.layers[layer_id].engram.embed = NodeShardedEngram(
                    rows,
                    config.engram_head_dim,
                    query_group,
                    storage_format=storage_format,
                )
        self.engram_history = None
        self._engram_input_buffers = None
        self._vllm_config = vllm_config
        self._engram_max_tokens = max(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        )
        self.register_buffer("engram_rotation", torch.eye(32), persistent=False)
        # [ENGRAM-DUMMY] V41_ENGRAM_WITH_DUMMY=1 lets the Engram host path come up
        # under --load-format dummy so its read latency can be measured without
        # paying the ~3-minute real-weight load.  Only the tokenizer and a small
        # quarot.safetensors are read here; the 206 GB table itself is allocated
        # by torch.empty() inside NodeShardedEngram regardless of load format.
        if ascend_config.enable_engram and (
            vllm_config.load_config.load_format != "dummy" or _ENGRAM_WITH_DUMMY
        ):
            with torch.device("cpu"):
                tokenizer = AutoTokenizer.from_pretrained(self.engram_root)
                self.engram_history = PagedNgramHistory(config, tokenizer)
                with safe_open(Path(self.engram_root) / "optional/quarot.safetensors", framework="pt") as file:
                    rotation = file.get_tensor("global_rotation")
                block = rotation[:32, :32].contiguous()
                if not torch.equal(rotation, torch.block_diag(*[block] * (config.hidden_size // 32))):
                    raise ValueError("Engram gate requires repeated block32 global rotation")
            self.engram_rotation.copy_(block)
        # [DEVICE-INDEX] 表已就绪后再建 host-mapped 设备视图（复用 layout 张量）。
        self._engram_dev_hash = None
        self._engram_dev_tables = {}
        self._engram_dev_ready = False
        self._engram_dev_failures = 0
        self._engram_dev_pages = 0
        self._engram_dev_checked = 0
        self._engram_dev_mode_noted = False
        self._engram_dev_graph = None
        if _ENGRAM_DEVICE_INDEX and ascend_config.enable_engram and self.engram_history is not None:
            self._engram_device_setup()

    def _engram_device_setup(self):
        """[DEVICE-INDEX] 建 dense 页镜像 + 把每层 int8 表映射成设备张量。"""
        from .engram_device_index import (
            DeviceNgramHash,
            HostMappedEngramTable,
            probe_host_mapping_capability,
        )

        # [DEVICE-INDEX-DEFAULT] 发布口径：**A3 默认开启 Engram 算子入图，A2 默认关闭**。
        #
        # 这里不按机型名硬编码，而是用**驱动侧的 host_mem_pool 特性**做判据
        # （`probe_host_mapping_capability()` 的第 0 步）：
        #   * A3(910C, PCI 19e5:d803, CPU↔NPU 走 HCCS) ⇒ host_mem_pool=1 ⇒ 开启；
        #   * A2(910B3, PCI 19e5:d802, CPU↔NPU 走 PCIe) ⇒ host_mem_pool=0 ⇒ 关闭。
        # 为什么 A2 必须关：host_mem_pool=0 时 host_register 逐页建元数据
        #（每 4 KiB 页 64 B），整表 206 GiB×2 层 ×8 rank 会让单次 vmalloc 申请
        # ~2.06 GiB 连续内核内存 —— A2 实测 17 分钟后 ret=207001（OOM 语义），
        # 而同时刻宿主机 MemAvailable 仍有 703 GiB。详见 engram_device_index.py
        # 的 host_mem_pool_supported()。
        #
        # auto 下探测不过就回退 host 路径（**功能与精度完全不变**，只是没有该项加速）。
        # 强制模式（=1）则探测失败即抛 —— 保留给 A3 验收与后续机型实验。
        if _ENGRAM_DEVICE_INDEX_MODE == "auto":
            ok, detail = probe_host_mapping_capability()
            if not ok:
                if _bp_rank_zero():
                    print(
                        "[DEVICE-INDEX] 本机不满足 Engram 算子入图的条件，"
                        "自动回退到 host 路径（**功能与精度不变**，只是没有该项加速）。\n"
                        f"  探测结果：{detail}\n"
                        "  发布口径：A3(910C) 默认开启、A2(910B3) 默认关闭 —— 判据是"
                        "驱动侧的 host_mem_pool 特性。\n"
                        "  如需强制启用（A3 验收/机型实验）设 V41_ENGRAM_DEVICE_INDEX=1",
                        flush=True,
                    )
                return
            if _bp_rank_zero():
                print(f"[DEVICE-INDEX] 能力探测通过，Engram 算子入图已启用：{detail}",
                      flush=True)
        elif _ENGRAM_DEVICE_INDEX_MODE not in ("1", "true", "on", "yes"):
            return

        history = self.engram_history
        config = self.config
        root = str(Path(self.engram_root) / "engram_int8")
        self._engram_dev_hash = DeviceNgramHash(
            token_map=history.token_map,
            primes=history.primes,
            offsets=history.offsets,
            multipliers=history.multipliers,
            pad_id=history.pad_id,
            image_token_id=history.image_token_id,
            image_pad_token_id=history.image_pad_token_id,
            # 必须在这里就给一个够用的容量：profile_run 阶段 num_gpu_blocks 还
            # 不知道，而合成 block_table 里的页号可能已经不是 0 —— 容量为 0 时
            # `flat.scatter_` 会越界写。构造时就按上界建表，代价 67 MB HBM。
            max_pages=_ENGRAM_DEVICE_PAGES_BOOT,
            device=torch.device("npu", torch.npu.current_device()),
        )
        for layer_id in config.engram_layer_ids:
            self._engram_dev_tables[layer_id] = HostMappedEngramTable(root, layer_id)
        if _ENGRAM_DEVICE_GRAPH:
            self._engram_dev_graph = EngramGraphCache(
                self._engram_dev_hash,
                self._engram_dev_tables,
                config.engram_layer_ids,
                device=torch.device("npu", torch.npu.current_device()),
                max_graphs=_ENGRAM_DEVICE_GRAPH_MAX,
            )
        self._engram_dev_ready = True
        if _bp_rank_zero():
            print(
                "[DEVICE-INDEX] Engram 表已映射为设备可寻址："
                + ", ".join(
                    f"L{lid}={self._engram_dev_tables[lid].rows}行"
                    for lid in config.engram_layer_ids
                )
                + f"（每张 {self._engram_dev_tables[config.engram_layer_ids[0]].width}B/行，"
                "HBM 占用 0）",
                flush=True,
            )

    @torch.no_grad()
    def _engram_dev_empty(self, dev, n):
        """每一层的空 lookup（形状 [n, n_hash_cols*width]）+ 空 mask。

        形状必须和真实路径一致，因为 ``prepare_engram_inputs`` 用它来建持久
        缓冲区（``values.shape[1]``）。空字典会让缓冲区退化成空字典。
        """
        config = self.config
        width = self._engram_dev_tables[config.engram_layer_ids[0]].width
        cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        return (
            {
                layer_id: torch.empty((n, cols * width), dtype=torch.bfloat16, device=dev)
                for layer_id in config.engram_layer_ids
            },
            torch.empty(n, dtype=torch.bool, device=dev),
        )

    @torch.no_grad()
    def _prepare_engram_device(self, input_ids, positions):
        """[DEVICE-INDEX] 全程设备的 Engram：dense 页镜像 + 哈希 + 整表直索。

        与 host 路径的接口完全一致（返回 ``{layer_id: [n, cols*width]}`` 与 bool
        mask），但没有任何 host 往返：没有 ``.cpu()``、没有分片、没有集合通信。

        **契约**：即使这一步没有任何 token，也必须返回**每一层的空张量**，而不是
        空字典。host 路径就是这样做的（它总是经过 route_many 再填 lookups），而
        ``prepare_engram_inputs`` 会把 mask pad 到静态容量，于是
        ``model.forward`` 里的 ``token_mask.numel()`` 非 0、必然去取
        ``lookups[layer.layer_idx]`` —— 返回 ``{}`` 会直接 KeyError（实测：
        profile_run 阶段 dynamo 报 `Dict key lookup failed for int: 1`）。
        """
        config = self.config
        dev = positions.device
        _bp = _BP_STATE.refresh()
        _bp.cur_is_decode = False
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return self._engram_dev_empty(dev, 0)
        first = self.layers[0].self_attn.dsa_attn.swa_cache_layer
        meta = metadata[first.prefix]
        boundaries = (
            meta.query_start_loc_cpu
            if getattr(meta, "query_start_loc_cpu", None) is not None
            else meta.query_start_loc.detach().cpu()
        ).long()
        n = int(boundaries[-1])
        _bp.cur_is_decode = n <= _BP_DECODE_TOKENS
        if not _bp.cur_is_decode:
            _bp.decode_gen += 1
        if n <= 0:
            return self._engram_dev_empty(dev, 0)
        # 页表容量 = 调度器的 block 池。num_gpu_blocks 只有在显存 profile **之后**
        # 才确定（profile_run 自己就会走到这里），所以先按一个够用的界建表，之后
        # 只在真实池子更大时才长 —— 只增不减，避免无谓的重分配（重分配会清空页镜像）。
        block_size = int(meta.storage_block_size)
        if self._engram_dev_hash.block_size != block_size:
            # 换 block_size 必须重建页表（槽位数变了）。保留已有容量，别缩小。
            keep = max(_ENGRAM_DEVICE_PAGES_BOOT, self._engram_dev_hash.capacity)
            self._engram_dev_hash.configure(block_size, keep)
            self._engram_dev_checked = 0
        blocks = getattr(self._vllm_config.cache_config, "num_gpu_blocks", None)
        if blocks is not None and int(blocks) + 8 > self._engram_dev_hash.capacity:
            self._engram_dev_hash.configure(block_size, int(blocks) + 8)
            self._engram_dev_checked = 0
            if _bp_rank_zero():
                print(
                    f"[DEVICE-INDEX] 页表扩容到 {self._engram_dev_hash.capacity} 页"
                    f"（num_gpu_blocks={blocks}）；之前的页镜像已清空。",
                    flush=True,
                )
        block_table = meta.block_table
        if block_table.device.type != dev.type:
            block_table = block_table.to(dev)
        # 页号越界必须立刻报错：dense 页表把"索引超界"从 host 版的 KeyError
        # 变成了越界读写，一旦静默就会污染别的请求的 n-gram。只在前几次调用里
        # 校验（每次一次同步），之后就信任 num_gpu_blocks。
        # 页号自检只在真实池子已知之后做：profile_run 用的是合成 block_table，
        # 那里的页号不代表调度器，误报会把起服卡死。
        if blocks is not None and self._engram_dev_checked < 3:
            self._engram_dev_checked += 1
            worst = self._engram_dev_hash.check_pages(block_table)
            if _bp_rank_zero():
                print(
                    f"[DEVICE-INDEX] 页表自检 #{self._engram_dev_checked}: "
                    f"容量={self._engram_dev_hash.capacity} 观测最大页号={worst} "
                    f"block_size={block_size} num_gpu_blocks={blocks}",
                    flush=True,
                )
        _t0 = _bp_time.perf_counter()
        ids_n, pos_n = input_ids[:n], positions[:n]
        if self._engram_dev_graph is not None:
            # 零拷贝捕获要求四个输入都是**设备上的常驻 buffer**：
            #   input_ids / positions 是模型自己的持久缓冲切片；
            #   query_start_loc.gpu 也是持久 buffer，且 req 的 searchsorted 在图内做。
            # 之前用 CPU 的 boundaries 会让每步多一次 H2D，而 H2D 会阻塞等待设备
            # 队列排空 —— 实测 route 因此从 0.35 ms 涨到 2.0(n=6)/5.4(n=24) ms。
            start_loc_dev = meta.query_start_loc
            if start_loc_dev is None or start_loc_dev.device.type != "npu":
                res = None
            else:
                res = self._engram_dev_graph.run(
                    ids_n, pos_n, start_loc_dev, block_table
                )
            if res is not None:
                lookups, mask = res
            else:
                lookups, mask = device_engram_lookup(
                    self._engram_dev_hash, self._engram_dev_tables,
                    config.engram_layer_ids, ids_n, pos_n, boundaries, block_table,
                )
        else:
            lookups, mask = device_engram_lookup(
                self._engram_dev_hash,
                self._engram_dev_tables,
                config.engram_layer_ids,
                ids_n,
                pos_n,
                boundaries,
                block_table,
            )
        # hash 与 route 的边界（宿主实现里是两段）在设备路径上不再可分，
        # 统一记在 route，另用 hash 记到第一次表访问为止。
        _bp.stat("route", _t0)
        return lookups, mask

    def prepare_engram(self, input_ids, positions):
        """Eager boundary: every DP participates, including metadata-free dummies."""
        # `nohost` 是 host 路径的消融臂（返回全零 lookup），与 device-index 正交，
        # 仍然交给 host 实现处理。
        if (
            _ENGRAM_DEVICE_INDEX
            and self._engram_dev_ready
            and _BP_STATE.refresh().mode != "nohost"
        ):
            # `/tmp/v41_bneck_mode` 的其它取值（stock/nocomm/mirror/faked2h）都是
            # **host 路径**的消融臂，在设备路径上没有对应物。若被选到，这里会
            # 按设备路径正常执行 —— 也就是说那条臂不再"关掉某一段"。说一次，
            # 免得有人以为自己在做对照实验。
            _m = _BP_STATE.refresh().mode
            if _m not in ("stock", "nohost") and not self._engram_dev_mode_noted:
                self._engram_dev_mode_noted = True
                if _bp_rank_zero():
                    print(
                        f"[DEVICE-INDEX] 注意：bneck mode={_m} 是 host 路径的消融臂，"
                        "在 ENGRAM_DEVICE_INDEX=1 下按正常设备路径执行（该臂失效）；"
                        "设备路径的开关只有 nohost 与重启级的 ENGRAM_DEVICE_INDEX。",
                        flush=True,
                    )
            if not _ENGRAM_DEVICE_FALLBACK:
                return self._prepare_engram_device(input_ids, positions)
            try:
                return self._prepare_engram_device(input_ids, positions)
            except Exception:
                self._engram_dev_failures += 1
                if _bp_rank_zero() and self._engram_dev_failures <= 3:
                    print(
                        "[DEVICE-INDEX] 第 %d 次失败，回退到 host 路径："
                        % self._engram_dev_failures,
                        flush=True,
                    )
                    import traceback

                    traceback.print_exc()
                if self._engram_dev_failures > 3:
                    self._engram_dev_ready = False
        return self._prepare_engram_host(input_ids, positions)

    def _prepare_engram_host(self, input_ids, positions):
        """Host 路径（原 prepare_engram 本体，保留作为对照与回退）。"""
        config = self.config
        if not get_ascend_config().enable_engram:
            return {}, torch.empty(0, dtype=torch.bool, device=positions.device)
        _bp = _BP_STATE.refresh()
        # 默认按"非 decode"记账；只有下面识别出 decode 的那一步才置 True，
        # 并一直保持到 prepare_engram_inputs 的 tick（review M8 修正）。
        _bp.cur_is_decode = False
        columns = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        hashes = torch.empty((0, len(config.engram_layer_ids), columns), dtype=torch.int64, device="cpu")
        mask = torch.empty(0, dtype=torch.bool, device="cpu")
        metadata = get_forward_context().attn_metadata
        if metadata is not None and self.engram_history is not None:
            first = self.layers[0].self_attn.dsa_attn.swa_cache_layer
            meta = metadata[first.prefix]
            boundaries = (
                meta.query_start_loc_cpu
                if getattr(meta, "query_start_loc_cpu", None) is not None
                else meta.query_start_loc.detach().cpu()
            ).long()
            n = int(boundaries[-1])
            requests = torch.repeat_interleave(torch.arange(len(boundaries) - 1, device="cpu"), boundaries.diff())
            _t_meta = _bp_time.perf_counter()
            _bp.cur_is_decode = n <= _BP_DECODE_TOKENS
            if not _bp.cur_is_decode:
                _bp.decode_gen += 1  # prefill 打断 decode 序列（review M7）
            if _bp.mode == "nohost":
                # [bneck] Engram host 全路径短路（D2H/hash/route/all_to_all 都不做）。
                return _bp_zero_lookups(self, positions, n), torch.zeros(
                    n, dtype=torch.bool, device=positions.device
                )
            _bp.maybe_delay("pre")
            if _bp.cur_is_decode:
                _bp.acc["meta"] = _bp.acc.get("meta", 0.0) + (
                    _bp_time.perf_counter() - _t_meta
                ) * 1000.0
            _t0 = _bp_time.perf_counter()
            _mirror = _ENGRAM_HOST_INPUTS["ids"]
            if (
                _bp.mode in ("mirror", "mirror_validate")
                and _mirror is not None
                and _mirror.numel() >= n
            ):
                # [mirror] ids 来自 runner 的 host 镜像 ⇒ 不做 ids 的 D2H。
                ids_host = _mirror[:n].long()
                _dev = input_ids[:n].cpu().long()
                if _bp.mode == "mirror_validate" and not torch.equal(ids_host, _dev):
                    _bad = int((ids_host != _dev).sum())
                    if _bp.rank0 and _bp.mirror_bad < 3:
                        print(
                            f"[mirror] VALIDATE FAILED bad={_bad}/{n} reason={_ENGRAM_HOST_INPUTS['reason']}",
                            flush=True,
                        )
                    _bp.mirror_bad += 1
                    ids_host = _dev
                else:
                    _bp.mirror_ok += 1
                pos_host = positions[:n].cpu().long()
            elif _bp.mode == "faked2h" and n <= 64:
                ids_host, pos_host = _bp.fake_host_ids(input_ids, positions, n)
            else:
                ids_host = input_ids[:n].cpu().long()
                pos_host = positions[:n].cpu().long()
            _bp.stat("d2h", _t0)
            _bp.maybe_delay("post")
            _t1 = _bp_time.perf_counter()
            hashes, mask = self.engram_history.update(
                ids_host,
                pos_host,
                requests,
                (
                    meta.block_table_cpu
                    if getattr(meta, "block_table_cpu", None) is not None
                    else meta.block_table.detach().cpu()
                ),
                meta.storage_block_size,
            )
            _bp.stat("hash", _t1)
            _bp.maybe_delay("route")
        lookups = {}
        tables = [self.layers[layer_id].engram.embed for layer_id in config.engram_layer_ids]
        ids_list = [hashes[:, slot] for slot in range(len(tables))]
        _t2 = _bp_time.perf_counter()
        if _bp.mode == "nocomm":
            # [bneck] 只跳过 route（collective + CPU 查表），保留 D2H/hash。
            routed = [
                torch.zeros((*ids.shape, table.width), dtype=torch.bfloat16, device=positions.device)
                for table, ids in zip(tables, ids_list)
            ]
        elif hasattr(tables[0], "route_many"):
            routed = tables[0].route_many(tables, ids_list)
        else:
            routed = [table(ids) for table, ids in zip(tables, ids_list)]
        _bp.stat("route", _t2)
        for layer_id, values in zip(config.engram_layer_ids, routed):
            lookups[layer_id] = values.flatten(1)
        return lookups, mask.to(positions.device)

    def prepare_engram_inputs(self, input_ids, positions, padded_tokens=None):
        """Refresh persistent inputs before main-model capture or replay."""
        _bp = _BP_STATE.refresh()
        _bp.mark_step()
        _t0 = _bp_time.perf_counter()
        lookups, mask = self.prepare_engram(input_ids, positions)
        _t1 = _bp_time.perf_counter()
        num_tokens = positions.shape[0]
        # The compiled V4.1 backbone uses the scheduler's static token
        # capacity for decode graphs (typically max_num_batched_tokens), even
        # when the current request has one token.  Keep lookup tensors at that
        # capacity so every captured graph sees the same Engram shape.
        output_tokens = max(self._engram_max_tokens, padded_tokens or 0)
        if output_tokens < num_tokens:
            raise ValueError("Engram padded token count is smaller than the input")
        if self._engram_input_buffers is None:
            capacity = self._engram_max_tokens
            self._engram_input_buffers = (
                {layer: values.new_zeros((capacity, values.shape[1])) for layer, values in lookups.items()},
                mask.new_zeros(capacity),
            )
        buffers, mask_buffer = self._engram_input_buffers
        padded_mask = mask_buffer[:output_tokens]
        padded_mask.zero_()
        padded_mask[: mask.numel()].copy_(mask)
        padded_lookups = {}
        for layer, values in lookups.items():
            padded = buffers[layer][:output_tokens]
            if _ENGRAM_PAD_SKIP:
                # [PAD-SKIP] Only the rows the model can actually read need
                # zeroing: `model.forward` does `lookups[layer][:n]` with
                # n = the padded batch size, and `copy_` below writes the valid
                # head.  Zeroing all 2048 rows cost 25.2 MB x 2 layers = 50.3 MB
                # of memset per step (measured pad=0.13 ms, ~387 GB/s).
                _n_read = max(values.shape[0], int(padded_tokens or 0))
                if _n_read > padded.shape[0]:
                    _n_read = padded.shape[0]
                padded[:_n_read].zero_()
            else:
                padded.zero_()
            padded[: values.shape[0]].copy_(values)
            padded_lookups[layer] = padded
        _bp.stat("pad", _t1)
        _bp.stat("total", _t0)
        if padded_lookups:
            _first = next(iter(padded_lookups.values()))
            _shape = tuple(_first.shape)
        else:
            _shape = ()
        _bp.tick(
            f"n={num_tokens} padded={int(padded_tokens or 0)} max={self._engram_max_tokens} "
            f"lookup={_shape} fake={_bp.fake_hits} mirror={_bp.mirror_ok}/{_bp.mirror_bad}"
            + (
                f" | {self._engram_dev_graph.stats()}"
                if getattr(self, "_engram_dev_graph", None) is not None
                else ""
            )
        )
        return {"engram_lookups": padded_lookups, "engram_mask": padded_mask}

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds=None,
        engram_lookups=None,
        engram_mask=None,
    ):
        if not get_pp_group().is_first_rank or not get_pp_group().is_last_rank:
            raise NotImplementedError("V4.1 eager milestone currently requires PP=1")
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        if engram_lookups is None:
            lookups, token_mask = self.prepare_engram(input_ids, positions)
        else:
            lookups, token_mask = engram_lookups, engram_mask
        self.shared_attention_state.reset()
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        pre_mix = hidden_states.new_zeros(hidden_states.shape[0], self.hc_mult, dtype=torch.float32)
        pre_mix[:, 0] = 1.0
        last_layer = None
        aux_hidden_states = []
        moe_input_ids = input_ids
        if self.needs_moe_input_ids:
            moe_input_ids = torch.where(input_ids == -1, 0, input_ids)
        if _IDS64_HOIST and moe_input_ids.dtype != torch.int64:
            # [IDS64-HOIST] fused_topk_router.py:164 casts input_ids to int64 on
            # EVERY layer (40x/step, ~0.20 ms).  The cast is idempotent and
            # moe_input_ids is not mutated inside the loop, so doing it once here
            # makes the in-loop .to(torch.int64) a no-op (same dtype -> returns
            # self, no kernel launched).  The value the router's all_gather sees
            # is unchanged, so communication shape/dtype is identical.
            moe_input_ids = moe_input_ids.to(torch.int64)
        for layer in self.layers:
            if layer.layer_idx == 20:
                _maybe_snapshot_ced_h20(layer, positions, hidden_states, pre_mix)
            _maybe_snapshot_ced_layer(
                layer, positions, input_ids, hidden_states, pre_mix, "pre"
            )
            if self._ced_prefill_only and layer.layer_idx == 20:
                layer.write_global_source_from_encoder(hidden_states, pre_mix)
                break
            last_layer = layer
            # DSpark consumes the residual stream entering its configured
            # target layers. The runner expresses checkpoint IDs as one-based.
            if layer.layer_idx + 1 in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states.mean(dim=1))
            if layer.engram is not None and token_mask.numel():
                n = hidden_states.shape[0]
                # Graph captures keep lookup buffers at static capacity; the
                # model's actual token dimension remains scheduler-dynamic.
                lookup = lookups[layer.layer_idx][:n]
                active_mask = token_mask[:n]
                kv = layer.engram.wkv(lookup)
                key, value = kv.split([self.hc_mult * self.config.hidden_size, self.config.hidden_size], -1)
                hidden_states[:n] = engram_gate(
                    hidden_states[:n],
                    key.view(n, self.hc_mult, self.config.hidden_size),
                    value,
                    _engram_gate_channel_weight(layer.engram.q_weight, layer.engram.k_weight),
                    _engram_gate_rotation_f32(self.engram_rotation),
                    active_mask,
                    self.config.rms_norm_eps,
                )
                _maybe_snapshot_ced_layer(
                    layer, positions, input_ids, hidden_states, pre_mix,
                    "after_engram", lookup=lookup, token_mask=active_mask,
                )
            ced_source_snapshot = None
            if (
                self._ced_source_compare_remaining > 0
                and layer.layer_idx == 20
                and not getattr(get_forward_context(), "capturing", False)
                and get_forward_context().attn_metadata is not None
            ):
                written = layer.write_global_source_from_encoder(hidden_states, pre_mix)
                if written:
                    metadata = layer.self_attn.v41_impl._get_layer_metadata(get_forward_context().attn_metadata)
                    slots = metadata.compressor.cache.slot_mapping[:written]
                    # Sample both ends of every chunk: the last rows are the
                    # ones most likely to cross a block or chunk boundary.
                    sampled = torch.cat((slots[:8], slots[-8:]), dim=0).cpu().tolist()
                    rows = list(dict.fromkeys(
                        (int(block), int(offset))
                        for block, offset in sampled
                        if block >= 0 and offset >= 0
                    ))
                    if rows:
                        index_k, index_scale = layer.self_attn.indexer.k_cache.kv_cache[0]
                        planes = (layer.self_attn.long_kv_cache.kv_cache[0], index_k, index_scale)
                        before = tuple(tuple(plane[block, offset].detach().clone() for block, offset in rows) for plane in planes)
                        positions = metadata.positions[:written]
                        pos_edges = (int(positions[0].item()), int(positions[-1].item()))
                        ced_source_snapshot = (rows, planes, before, pos_edges)
                        self._ced_source_compare_remaining -= 1
                        self._ced_source_compare_count += 1
            hidden_states, pre_mix = layer(positions, hidden_states, pre_mix, None, input_ids=moe_input_ids)
            _maybe_snapshot_ced_layer(
                layer, positions, input_ids, hidden_states, pre_mix, "post"
            )
            if ced_source_snapshot is not None:
                rows, planes, before, pos_edges = ced_source_snapshot
                for plane_idx, (plane, saved) in enumerate(zip(planes, before)):
                    for row_idx, ((block, offset), expected) in enumerate(zip(rows, saved)):
                        if not torch.equal(expected, plane[block, offset]):
                            raise RuntimeError(
                                "CED layer-20 source differs from normal forward: "
                                f"plane={plane_idx} row={row_idx} slot=({block},{offset})"
                            )
                print(
                    "[CED-SOURCE] layer20 source-only cache rows match normal "
                    f"forward: chunk={self._ced_source_compare_count} "
                    f"tokens={written} positions={pos_edges[0]}..{pos_edges[1]} "
                    f"rows={len(rows)}",
                    flush=True,
                )
        assert last_layer is not None
        hidden_states = last_layer.hc_collapse(hidden_states, pre_mix)
        hidden_states = self.norm(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class AscendDeepseekV41ForCausalLM(AscendDeepseekV4ForCausalLM):
    model_cls = DeepseekV41Model
    requires_raw_input_tokens = True
    _DEFERRED_WEIGHT_MARKERS = ()
    _DEFERRED_WEIGHT_PREFIXES = ("aligner.", "vision.", "image_", "mtp.")

    def compute_logits(self, hidden_states):
        logits = super().compute_logits(hidden_states)
        if logits is not None and self.model._ced_prefill_only:
            # The P-side Mooncake request must finish with LENGTH_CAPPED so
            # request_finished_all_groups can publish its cache blocks. The
            # marker is a valid non-EOS token, explicitly *not* a model answer.
            # Never expose this dedicated P endpoint as a chat service.
            marker = 42
            eos = self.config.eos_token_id
            if marker >= logits.shape[-1] or marker == eos or (isinstance(eos, (tuple, list)) and marker in eos):
                raise RuntimeError("CED internal transfer marker conflicts with tokenizer")
            logits.fill_(-10000.0)
            logits[..., marker] = 0.0
        return logits

    def prepare_engram_inputs(self, input_ids, positions, padded_tokens=None):
        return self.model.prepare_engram_inputs(input_ids, positions, padded_tokens)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        engram_lookups=None,
        engram_mask=None,
    ):
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            engram_lookups=engram_lookups,
            engram_mask=engram_mask,
        )

    @classmethod
    def _is_milestone_weight(cls, name):
        return not name.startswith(cls._DEFERRED_WEIGHT_PREFIXES) and not any(
            marker in name for marker in cls._DEFERRED_WEIGHT_MARKERS
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if not get_ascend_config().enable_engram:
            return super().load_weights((name, tensor) for name, tensor in weights if ".engram." not in name)
        engram_loaded = set()

        def milestone_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                if ".engram." in name:
                    # Bypass V4's generic embed -> embed_tokens remapping and TP loader.
                    local_name = name.removeprefix("model.")
                    # FP8/MXFP8 Engram scales are consumed by the CPU loader.
                    if local_name.endswith(".engram.embed.scale"):
                        continue
                    parameter_name = "model." + local_name
                    if local_name.endswith(".engram.embed.weight"):
                        layer_id = int(local_name.split(".")[1])
                        self.model.layers[layer_id].engram.embed.load_checkpoint(self.model.engram_root, local_name)
                    else:
                        param = self.get_parameter(parameter_name)
                        if tensor.dtype != torch.bfloat16 or tensor.shape != param.shape:
                            raise ValueError(f"Unexpected BF16 Engram parameter: {name}")
                        param.data.copy_(tensor)
                    engram_loaded.add(parameter_name)
                elif self._is_milestone_weight(name):
                    yield name, tensor

        loaded = super().load_weights(milestone_weights())
        expected = {name for name, _ in self.named_parameters() if ".engram." in name}
        if engram_loaded != expected:
            raise ValueError(f"Missing Engram weights: {expected - engram_loaded}")
        return loaded | engram_loaded
