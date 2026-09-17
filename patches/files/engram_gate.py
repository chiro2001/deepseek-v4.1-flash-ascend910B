# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[dsv41-ws-opt] Chunked Engram gate, default off.

The stock eager gate materialises four FP32 ``[tokens, hc_mult, hidden]``
tensors at once (``original``, ``key.float()`` and the two temporaries of
``original * channel_weight * key``).  At ``BAT=4096``, ``hc_mult=4`` and
``hidden=5120`` that is ~1.25 GiB of peak device activation even for a
one-token decode, because vLLM profiles with ``max_num_batched_tokens``.

``V41_ENGRAM_GATE_CHUNK=512`` walks the token dimension in chunks and keeps the
same per-token operator order, so the arithmetic is a token-wise independent
slice of the stock implementation.  Unset / 0 => stock path unchanged.

P38 prefill fix (dynamo-safe chunking).  The chunked path used to dispatch on
the *symbolic* token count (``hidden.shape[0] <= chunk``) and iterate
``range(0, n, chunk)`` inside the compiled backbone.  Two independent failures
followed:

1. ``range(0, n, chunk)`` forces Dynamo to specialize the scheduler-dynamic
   token dimension ``n`` to the profile_run constant and contradict vLLM's
   ``mark_dynamic(input_ids, 0)``, aborting profile_run with
   ``ConstraintViolationError: ... specialized it to be a constant (120)``.
2. vLLM compiles with ``fullgraph=True`` and drops shape guards
   (``evaluate_guards=False``), so token-count-dependent Python control flow can
   bake the dummy ``n=120`` decision into the graph and silently bypass R2 for
   every real 4096-token prefill chunk.

The fix removes every token-count-dependent Python control flow from the
compiled graph.  The token dimension is padded to the static
``V41_ENGRAM_GATE_MAX_TOKENS`` (default 4096, set it to
``--max-num-batched-tokens``) with ``F.pad`` -- no dynamic slice/copy -- and the
loop unrolls over the constant ``range(0, MAX, chunk)``.  Padded rows are zero
and their gate output is discarded; the saved token mask is re-applied once at
the end, which is bit-identical to the stock ``masked_fill`` for real rows.
``chunk <= 0`` (default) is the compiled reference path, byte-for-byte
unchanged.  Offline evidence (CPU torch 2.10, ``torch.compile(dynamic=True)``
and ``torch._dynamo.optimize(dynamic=False, guard_filter_fn=drop_all)``):
``logs/perf/p38_r2_engram_test.log``.
"""

# [ENGRAM-GATE-HOIST] ----------------------------------------------------
import os as _os_engram_hoist

# V41_ENGRAM_GATE_HOIST=1：把同一函数里重复的 .float() 合并成一次。
# 默认 0 = stock（两次独立 cast，逐算子不变）。
_ENGRAM_GATE_HOIST = _os_engram_hoist.environ.get("V41_ENGRAM_GATE_HOIST", "0") == "1"
# [ENGRAM-GATE-HOIST] ----------------------------------------------------

import os

import torch


def _gate_chunk_tokens() -> int:
    raw = os.environ.get("V41_ENGRAM_GATE_CHUNK", "")
    if not raw:
        return 0
    try:
        chunk = int(raw)
    except ValueError:
        return 0
    return chunk if chunk > 0 else 0


def _engram_gate_reference(
    hidden: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    channel_weight: torch.Tensor,
    rotation_block: torch.Tensor,
    token_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Stock upstream implementation.  Do not change: default-path oracle."""
    dim = hidden.shape[-1]
    # [ENGRAM-GATE-HOIST] A: hidden 在同一函数里 .float() 两次；关掉门控时
    # hidden_out 仍是一次独立的 .float()，逐算子与 stock 完全一致。
    hidden_restore = hidden.float()
    hidden_out = hidden_restore if _ENGRAM_GATE_HOIST else hidden.float()
    original = (hidden_restore.unflatten(-1, (-1, rotation_block.shape[0])) @ rotation_block.float().T).flatten(-2)
    key = key.float()
    rstd = torch.rsqrt(original.square().mean(-1) + eps)
    rstd *= torch.rsqrt(key.square().mean(-1) + eps)
    dot = (original * channel_weight.float() * key).sum(-1) * rstd * dim**-0.5
    magnitude = dot.abs().clamp_min(1e-6).sqrt()
    gate = torch.sigmoid(torch.where(torch.signbit(dot), -magnitude, magnitude))
    gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return (hidden_out + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden.dtype)


_ENGRAM_GATE_MAX_TOKENS_ENV = "V41_ENGRAM_GATE_MAX_TOKENS"
_ENGRAM_GATE_DEFAULT_MAX_TOKENS = 4096


def _gate_max_tokens(chunk: int) -> int:
    """Static token ceiling used to pad the chunked gate.

    Must be >= the largest token dimension of any forward (prefill chunk or
    spec decode batch); the launcher should set it to
    ``--max-num-batched-tokens``.  ``torch._check`` enforces it at runtime
    without specializing the symbolic token dimension.
    """
    raw = os.environ.get(_ENGRAM_GATE_MAX_TOKENS_ENV, "")
    try:
        value = int(raw) if raw else _ENGRAM_GATE_DEFAULT_MAX_TOKENS
    except ValueError:
        value = _ENGRAM_GATE_DEFAULT_MAX_TOKENS
    if value < chunk:
        value = max(chunk, _ENGRAM_GATE_DEFAULT_MAX_TOKENS)
    return value


def _static_pad_first_dim(tensor: torch.Tensor, target: int) -> torch.Tensor:
    """Pad dim 0 to ``target`` rows with zeros (rank-generic, no dynamic slice)."""
    pad = (0, 0) * (tensor.dim() - 1) + (0, target - tensor.shape[0])
    return torch.nn.functional.pad(tensor, pad)


def _engram_gate_chunked(
    hidden: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    channel_weight: torch.Tensor,
    rotation_block: torch.Tensor,
    token_mask: torch.Tensor,
    eps: float,
    chunk: int,
) -> torch.Tensor:
    """Static-shape chunked gate: pad to MAX, unroll a constant chunk loop.

    No Python control flow depends on ``hidden.shape[0]``, so one compiled
    graph is valid for every token count from 1 to ``MAX`` -- including on
    builds that drop shape guards -- while the per-chunk FP32 workspace stays
    bounded.
    """
    dim = hidden.shape[-1]
    n = hidden.shape[0]
    max_tokens = _gate_max_tokens(chunk)
    # P42 fix (2026-09-13 17:0x, second attempt): the contract guard was written as
    # ``torch._check(n <= max_tokens, <msg>)``.  On this torch build with
    # fullgraph/AOT neither message form works:
    #   * a *lambda* message is captured into the FX graph and torch.fx
    #     split_module cannot extract sympy free symbols from a function object
    #     ("cannot extract sympy expressions from
    #      <function _engram_gate_chunked.<locals>.<lambda>>", split_module.py:239)
    #     -> torch.compile AOT fails at model load;
    #   * a plain string message is rejected by Dynamo itself
    #     ("Can't extract message from torch._check()", GB0288).
    # The guard is therefore dropped.  Callers pass ``n <= V41_ENGRAM_GATE_MAX_TOKENS``
    # by construction (launcher sets MAX = --max-num-batched-tokens); if a caller ever
    # exceeds it, the final ``torch.where`` below broadcasts [max,...] against
    # [n,...] and raises instead of silently truncating.
    channel = channel_weight.float()
    rotation = rotation_block.float()
    padded = _static_pad_first_dim(hidden, max_tokens)
    padded_key = _static_pad_first_dim(key, max_tokens)
    padded_value = _static_pad_first_dim(value, max_tokens)
    for start in range(0, max_tokens, chunk):
        stop = min(start + chunk, max_tokens)
        h = padded[start:stop]
        k = padded_key[start:stop]
        v = padded_value[start:stop]
        # [ENGRAM-GATE-HOIST] B: 同 A，chunked 路径里 h 也被 .float() 两次。
        h_restore = h.float()
        h_out = h_restore if _ENGRAM_GATE_HOIST else h.float()
        original = (h_restore.unflatten(-1, (-1, rotation.shape[0])) @ rotation.T).flatten(-2)
        key_f32 = k.float()
        rstd = torch.rsqrt(original.square().mean(-1) + eps)
        rstd *= torch.rsqrt(key_f32.square().mean(-1) + eps)
        # Keep the same left-to-right order as the reference expression.
        dot = (original * channel * key_f32).sum(-1) * rstd * dim**-0.5
        # ``original`` / ``key_f32`` are not needed by the output expression.
        del original, key_f32
        magnitude = dot.abs().clamp_min(1e-6).sqrt()
        gate = torch.sigmoid(torch.where(torch.signbit(dot), -magnitude, magnitude))
        padded[start:stop] = (h_out + gate.unsqueeze(-1) * v.float().unsqueeze(-2)).to(hidden.dtype)
    # Re-apply the saved-token mask exactly like the stock ``masked_fill``:
    # masked rows get gate=0, i.e. the bf16 residual itself.
    return torch.where(token_mask.unsqueeze(-1).unsqueeze(-1), padded[:n], hidden)


def engram_gate(
    hidden: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    channel_weight: torch.Tensor,
    rotation_block: torch.Tensor,
    token_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply original-basis gating to a rotated residual and rotated value.

    ``hidden`` and ``key`` have shape [tokens, hc_mult, hidden_size].
    The saved rotation consists of identical diagonal blocks. Restore hidden
    in FP32; the value projection already includes the forward rotation.

    Default path is the stock implementation; ``V41_ENGRAM_GATE_CHUNK`` > 0
    switches to the static-shape padded chunked path for every token count.
    """
    chunk = _gate_chunk_tokens()
    if chunk <= 0:
        return _engram_gate_reference(
            hidden,
            key,
            value,
            channel_weight,
            rotation_block,
            token_mask,
            eps,
        )
    # NOTE: deliberately no ``hidden.shape[0] <= chunk`` short-circuit here.
    # Any token-count-dependent branch is frozen at the first traversed shape
    # by the guard-dropping compile wrapper; the padded chunked path handles
    # every size (including 1 token) with static chunk shapes instead.
    return _engram_gate_chunked(
        hidden,
        key,
        value,
        channel_weight,
        rotation_block,
        token_mask,
        eps,
        chunk,
    )
