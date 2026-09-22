#!/usr/bin/env python3
"""[S_graphfix] int8 KV8 读侧的**图兼容性**补丁（env 门控，默认关）。

缺口（`logs/048` 的 8 卡真权重实测）：`kv8_ori_plane` 用
``if query_rows == num_reqs`` 判 "decode vs prefill"；spec-decode 下 decode 批是
``query_rows = num_reqs * (1 + num_spec_tokens)`` ⇒ 误走 **prefill 分支**，而那条
分支的 ``.max().item()`` 在 ACLGraph 捕获期直接
``Not_Supported(EE1016): Synchronizing a stream failed``。

两条改动（同一份 `attention/dsa_v41.py`，全部走
``VLLM_V41_KV8_GRAPH_SAFE=1``，默认 0 = 逐字旧行为）：

  ① `kv8_ori_plane(..., rows_bound=)`：新增**上界分支**。窗口带的起点仍在 device 上
     按真实 ``q_len`` 算（``query_start_loc`` 差分），只有 "每请求最多几页" 这个 host
     标量改成上界公式 ``(rows_bound + window - 1) // block_size + 2``，其中
     ``rows_bound`` 来自 `_native_attention` 读到的 ``metadata.swa.max_query_len``
     （引擎在 CPU 张量上算好的 Python int，不产生 D2H）。``.item()`` 那条
     **prefill 分支原样保留**（eager only，是本任务的 "不许动" 项）。
  ② `_kv8_cmp_plane(..., graph_safe=)`：档 D 的 long-KV INT8 面在 spec-decode 下
     ``rows != num_reqs`` ⇒ 走 "整段压缩前缀重建" 分支，页数来自
     ``cache_seq_lens.max().item()``（D2H；若改成捕获期冻结的页数，则 replay 静默错）。
     新增**按 query 行私有一段**的 selection-based 分支（单行快路径的直系推广）：
     第 ``i`` 行的第 ``t`` 个选择落到合成下标 ``i * topk + t`` ⇒ scratch 页
     ``i * per_req + t // block_size``、页内偏移 ``t % block_size``，scratch 表取
     ``rows * per_req`` 页的恒等表。全部标量来自 shape/config。

★ 与档 D 的 ``VLLM_V41_KV8_PREFILL=1`` 尾部包装共存：
   * `kv8_ori_plane` 包装：``rows_bound`` 非 None 时转交原函数（不进 prefill 融合）；
   * `_kv8_cmp_plane_prefill` 包装：``graph_safe`` 且行数整除时转交类方法
     （不再用捕获期冻结的 ``max_cache_seq_len`` 定页数）。
   真 prefill（batch 里有 prefill 请求）两条路径都不动。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import py_compile
import sys
import tempfile
from pathlib import Path

A_SIG = """def kv8_ori_plane(
    kv_i8: torch.Tensor,
    kv_scale: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    num_reqs: int,
    query_rows: int,
    window: int,
):
"""

N_SIG = '''def _kv8_graph_safe_enabled() -> bool:
    """[S_graphfix] env gate: default off so the legacy paths run unchanged."""
    import os as _os

    return _os.environ.get("VLLM_V41_KV8_GRAPH_SAFE", "0").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )


def _kv8_cmp_legacy() -> bool:
    """[S_graphfix] diagnostic arm: keep the ``cmp`` face on the legacy path.

    ``SG_CMP_LEGACY=1`` leaves the window (SWA) fix in place but sends the
    compressed long-KV face back through the capture-era path.  That isolates
    "what does tier D actually do when only the window face is fixed" - i.e.
    whether a page count frozen at capture time fails loudly or silently.
    Default 0 => no effect.
    """
    import os as _os

    return _os.environ.get("SG_CMP_LEGACY", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


_SG_PPR_SEEN: dict = {}


def _sg_ppr_trace(tag: str, **fields) -> None:
    """[S_graphfix] Hot-path probe for the capture-vs-replay page counts.

    ``SG_TRACE_PPR=1`` prints the *host* scalars that decide the rebuild's scratch
    size, labelled with ``capturing=<bool>`` so a single log shows both sides:

      * capture time: ``mcs=6`` (the dummy batch's seq_len) ⇒ legacy ``ppr=1``;
      * replay time : the real compressed prefix ⇒ hundreds of pages.

    It never reads a device value (that is the defect being measured), and it is
    rate limited so a 40-layer graph cannot flood the log.
    """
    import os as _os

    if _os.environ.get("SG_TRACE_PPR", "0").strip().lower() not in ("1", "true", "yes", "on"):
        return
    try:
        import torch as _t

        capturing = bool(_t.npu.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        capturing = None
    key = (tag, capturing, tuple(sorted((k, str(v)) for k, v in fields.items())))
    if key in _SG_PPR_SEEN or len(_SG_PPR_SEEN) >= 24:
        return
    _SG_PPR_SEEN[key] = 1
    payload = " ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[SG-PPR] {tag} capturing={capturing} {payload}", flush=True)


_SG_CAPTURING_OK = [None]  # None = 未探测；False = 该 API 在本机不可用


def _sg_is_capturing() -> bool:
    """[S_graphfix] ``torch.npu.is_current_stream_capturing()`` - host query, no sync.

    ★ 这个信号**只影响捕获期的 dummy 批**（它的 ``is_prefilling`` 不可信）。若该 API
    在本机不可用，我们**退回 legacy**（= 旧行为、图捕获期会响亮地 EE1016），
    绝不"猜"，以免把真 prefill 误路由到新分支。
    """
    if _SG_CAPTURING_OK[0] is False:
        return False
    try:
        import torch as _t

        ok = bool(_t.npu.is_current_stream_capturing())
        _SG_CAPTURING_OK[0] = True
        return ok
    except Exception:  # noqa: BLE001
        _SG_CAPTURING_OK[0] = False
        return False


def _kv8_graph_rows_bound(swa, query_rows: int, num_reqs: int):
    """[S_graphfix] Host-only per-request query-row bound; ``(False, None)`` = legacy.

    ``graph_safe`` says "this batch's rows are decode rows, so the rebuild must not
    sync the stream"; ``rows_bound`` upper-bounds *every* request's query length and
    is what sizes the scratch.  Both scalars come from CPU-side bookkeeping the
    engine already computed: ``num_prefills`` / ``max_query_len`` are plain ints on
    the layer metadata (the builder copies ``max_query_len`` out of the common
    metadata, which vLLM derives from ``query_start_loc_cpu``), and the fallback is
    ``query_rows`` (the batch's padded row count, a *shape*), which upper-bounds any
    single request's query length.  Prefill batches return the legacy path on
    purpose: prefill is eager, its ``.item()`` was measured cheap (logs/033), and
    this task forbids changing its geometry.

    ★ Capture must take the bound branch too.  During ``_dummy_run`` the batch is a
    synthetic decode batch whose ``is_prefilling`` flags are derived from request
    bookkeeping the dummy run does not own, so ``num_prefills`` there is *not* a
    trustworthy classifier - and capture is exactly what the defect broke.  The
    recorded constants stay valid on replay because ``rows_bound`` is then the
    graph's own uniform query length (``uniform_decode_query_len`` = 1 + spec
    tokens), which is what every replayed step of that graph uses.
    """
    if not _kv8_graph_safe_enabled():
        return False, None
    bound = int(getattr(swa, "max_query_len", 0) or 0)
    if bound <= 0 or bound > int(query_rows):
        bound = int(query_rows)
    bound = max(1, bound)
    if int(getattr(swa, "num_prefills", 0) or 0) > 0 and not _sg_is_capturing():
        # Real eager prefill batch: keep the legacy geometry and its fused kernel.
        return False, None
    return True, bound


def kv8_ori_plane(
    kv_i8: torch.Tensor,
    kv_scale: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    num_reqs: int,
    query_rows: int,
    window: int,
    rows_bound: int | None = None,
):
'''

A_BRANCH = """    if query_rows == num_reqs:
        # Decode: one query row per request, so the span is exactly one window
        # and covers at most two pages.  Fully device-side (capture safe).
        pages_per_req = 2
        window_start = (lens - window).clamp_min(0)
    else:
"""

N_BRANCH = """    if rows_bound is not None:
        # [S_graphfix] Capture-safe rebuild for decode-shaped batches, including
        # speculative decoding, where every request carries ``1 + num_spec_tokens``
        # query rows instead of one.  The old predicate (``query_rows == num_reqs``)
        # only recognised the pure-decode shape, so a spec-decode batch fell into
        # the eager prefill branch below, whose ``.max().item()`` syncs the stream
        # and aborts graph capture (Not_Supported(EE1016), logs/048).
        #
        # Nothing here may read a device value: the band start uses the *real*
        # per-request query length (a device-side diff of ``query_start_loc``, no
        # host round trip) and the page count uses the host-side bound
        # ``rows_bound >= max(q_len)``.  The widest band any request can need is
        # ``rows_bound + window`` rows, and a span of ``s`` rows spans at most
        # ``((s - 2) // block_size) + 2 <= (s - 1) // block_size + 2`` pages, so the
        # bound holds for every in-page alignment.  Extra pages are inert: the block
        # table only points at blocks the mask reads, and the operator ignores rows
        # outside the band.
        q_len = (query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]).to(torch.int64)
        span = torch.minimum(lens, q_len + window)
        window_start = lens - span
        pages_per_req = min(width, max(1, (int(rows_bound) + window - 1) // block_size + 2))
    elif query_rows == num_reqs:
        # Decode: one query row per request, so the span is exactly one window
        # and covers at most two pages.  Fully device-side (capture safe).
        pages_per_req = 2
        window_start = (lens - window).clamp_min(0)
    else:
"""

A_CALL = """        ori_kv = attn.dsa_attn.swa_cache_layer.kv_cache[0]
        if isinstance(ori_kv, (tuple, list)):
"""

N_CALL = """        ori_kv = attn.dsa_attn.swa_cache_layer.kv_cache[0]
        # [S_graphfix] Host-only gate + per-request row bound (no D2H, no
        # capture-time *value*): decode-shaped batches take the bound branch
        # above, real prefill batches keep the legacy eager branch.
        graph_safe, rows_bound = _kv8_graph_rows_bound(metadata.swa, q.shape[0], num_reqs)
        _sg_ppr_trace(
            "native_attention",
            num_reqs=num_reqs,
            query_rows=int(q.shape[0]),
            num_prefills=int(getattr(metadata.swa, "num_prefills", 0) or 0),
            max_query_len=int(getattr(metadata.swa, "max_query_len", 0) or 0),
            swa_mcs=int(getattr(metadata.swa, "max_cache_seq_len", 0) or 0),
            cmp_mcs=int(
                0 if getattr(metadata, "attention", None) is None
                else (getattr(metadata.attention, "max_cache_seq_len", 0) or 0)
            ),
            graph_safe=graph_safe,
            rows_bound=rows_bound,
        )
        if isinstance(ori_kv, (tuple, list)):
"""

A_CALL_ARGS = """                num_reqs,
                q.shape[0],
                attn.window_size,
            )
"""

N_CALL_ARGS = """                num_reqs,
                q.shape[0],
                attn.window_size,
                rows_bound=rows_bound,
            )
"""

A_CMP_CALL = """                cmp_kv, cmp_block_table, cmp_indices = self._kv8_cmp_plane(
                    source_cache,
                    source_scale,
                    cmp_indices,
                    cmp_block_table,
                    num_reqs,
                    cmp_seq_lens,
                )
"""

N_CMP_CALL = """                cmp_kv, cmp_block_table, cmp_indices = self._kv8_cmp_plane(
                    source_cache,
                    source_scale,
                    cmp_indices,
                    cmp_block_table,
                    num_reqs,
                    cmp_seq_lens,
                    graph_safe=graph_safe and not _kv8_cmp_legacy(),
                )
"""

A_CMP_SIG = """    def _kv8_cmp_plane(self, kv_i8, kv_scale, indices, block_table, num_reqs, cache_seq_lens):
"""

N_CMP_SIG = """    def _kv8_cmp_plane(
        self,
        kv_i8,
        kv_scale,
        indices,
        block_table,
        num_reqs,
        cache_seq_lens,
        graph_safe: bool = False,
    ):
"""

A_CMP_FALLBACK = """        # Prefill / multi query-row batch: the topk sets differ per query row, so
        # the whole compressed prefix of every request is rebuilt instead and the
        # operator keeps the original logical indices.
        used = int(cache_seq_lens[:num_reqs].max().item())
"""

N_CMP_FALLBACK = '''        if graph_safe and per_req * block_size == topk and rows % num_reqs == 0:
            # [S_graphfix] Speculative decode: ``rows = num_reqs * (1 + num_spec)``
            # query rows, and the topk sets differ per row.  The fallback below
            # rebuilds each request's whole compressed prefix, which needs
            # ``cache_seq_lens.max()`` -- a D2H sync (capture-fatal) *and*, if the
            # page count were frozen at capture time instead, a silent replay bug:
            # the capture dummy batch reports seq_len == max_query_len (6 here), so a
            # capture-time page count would cover 1 block while replay addresses
            # hundreds.
            #
            # Instead give every query row a private scratch segment: row ``i``'s
            # ``t``-th selection lands at synthetic index ``i * topk + t``, i.e.
            # scratch page ``i * per_req + t // block_size`` at in-page offset
            # ``t % block_size`` -- exact because ``per_req * block_size == topk``
            # (the same identity the single-row fast path above relies on).  The
            # scratch table is then the identity over ``rows * per_req`` pages, and
            # every scalar involved is a shape or a config constant, so the branch is
            # capture-safe and replay-safe.
            reps = rows // num_reqs
            idx = indices[:, 0].to(torch.int64)
            valid = idx >= 0
            safe = torch.where(valid, idx, torch.zeros_like(idx))
            block = safe // block_size
            offset = safe % block_size
            # Row ``i`` belongs to request ``i // reps`` in the padded batch (uniform
            # spec-decode query length).  ``repeat_interleave`` is a device op with a
            # static output shape: no host round trip.
            table_rows = block_table[:num_reqs].to(torch.int64).repeat_interleave(reps, dim=0)
            phys = torch.gather(table_rows, 1, block)
            keys = kv8_gather_rows(kv_i8, phys, offset)
            scales = kv8_gather_rows(kv_scale, phys, offset)
            deq = kv8_dequant_rows(
                keys.reshape(rows * topk, dim),
                scales.reshape(rows * topk, groups),
            )
            segments = rows * per_req
            _sg_ppr_trace(
                "cmp_graph_safe",
                num_reqs=num_reqs,
                rows=rows,
                per_req=per_req,
                segments=segments,
                topk=topk,
            )
            scratch = kv8_scratch_plane(segments, block_size, dim, deq.dtype, deq.device)
            scratch.view(segments * block_size, 1, dim).copy_(deq.view(rows * topk, 1, dim))
            row_base = (
                torch.arange(rows, dtype=torch.int32, device=indices.device) * topk
            ).view(rows, 1)
            columns = torch.arange(topk, dtype=torch.int32, device=indices.device).view(1, topk)
            renumbered = torch.where(valid, row_base + columns, torch.full_like(columns, -1))
            table = (
                torch.arange(segments, dtype=torch.int32, device=indices.device)
                .view(1, segments)
                .repeat(num_reqs, 1)
            )
            return scratch, table, renumbered.unsqueeze(1)
        # Prefill / multi query-row batch: the topk sets differ per query row, so
        # the whole compressed prefix of every request is rebuilt instead and the
        # operator keeps the original logical indices.
        used = int(cache_seq_lens[:num_reqs].max().item())
'''

A_WRAP_ORI = """    def kv8_ori_plane(
        kv_i8,
        kv_scale,
        query_start_loc,
        seq_lens,
        block_table,
        num_reqs,
        query_rows,
        window,
    ):
        \"\"\"Chunked prefill: one Triton launch rebuilds the whole window union.\"\"\"
        if query_rows == num_reqs:
            return _kv8_ori_plane_decode(
                kv_i8,
                kv_scale,
                query_start_loc,
                seq_lens,
                block_table,
                num_reqs,
                query_rows,
                window,
            )
"""

N_WRAP_ORI = """    def kv8_ori_plane(
        kv_i8,
        kv_scale,
        query_start_loc,
        seq_lens,
        block_table,
        num_reqs,
        query_rows,
        window,
        rows_bound=None,
    ):
        \"\"\"Chunked prefill: one Triton launch rebuilds the whole window union.\"\"\"
        if rows_bound is not None:
            # [S_graphfix] Decode-shaped batch (incl. spec decode): hand the host
            # bound to the plain-torch rebuild.  The Triton prefill kernel sizes its
            # scratch from ``max_q_len=query_rows`` and is only needed for eager
            # prefill, where a chunk carries far more than one row per request.
            return _kv8_ori_plane_decode(
                kv_i8,
                kv_scale,
                query_start_loc,
                seq_lens,
                block_table,
                num_reqs,
                query_rows,
                window,
                rows_bound=rows_bound,
            )
        if query_rows == num_reqs:
            return _kv8_ori_plane_decode(
                kv_i8,
                kv_scale,
                query_start_loc,
                seq_lens,
                block_table,
                num_reqs,
                query_rows,
                window,
            )
"""

A_WRAP_CMP = """    def _kv8_cmp_plane_prefill(self, kv_i8, kv_scale, indices, block_table, num_reqs, cache_seq_lens):
        \"\"\"Chunked prefill: page-granular rebuild, one Triton launch.\"\"\"
        rows, _, topk = indices.shape
        block_size = kv_i8.shape[1]
        per_req = (topk + block_size - 1) // block_size
        if rows == num_reqs and per_req * block_size == topk:
            return _kv8_cmp_plane_decode(
                self,
                kv_i8,
                kv_scale,
                indices,
                block_table,
                num_reqs,
                cache_seq_lens,
            )
"""

N_WRAP_CMP = """    def _kv8_cmp_plane_prefill(
        self,
        kv_i8,
        kv_scale,
        indices,
        block_table,
        num_reqs,
        cache_seq_lens,
        graph_safe=False,
    ):
        \"\"\"Chunked prefill: page-granular rebuild, one Triton launch.\"\"\"
        rows, _, topk = indices.shape
        block_size = kv_i8.shape[1]
        per_req = (topk + block_size - 1) // block_size
        if graph_safe and per_req * block_size == topk and rows % num_reqs == 0:
            # [S_graphfix] Uniform multi-row batch (spec decode): the class method's
            # selection-based branch is both capture- and replay-safe.  The Triton
            # prefill kernel below sizes its scratch from ``max_cache_seq_len``, a
            # *value* the capture dummy batch reports as 6 ⇒ never use it on a
            # graph-replayed step.
            return _kv8_cmp_plane_decode(
                self,
                kv_i8,
                kv_scale,
                indices,
                block_table,
                num_reqs,
                cache_seq_lens,
                graph_safe=True,
            )
        if rows == num_reqs and per_req * block_size == topk:
            return _kv8_cmp_plane_decode(
                self,
                kv_i8,
                kv_scale,
                indices,
                block_table,
                num_reqs,
                cache_seq_lens,
                graph_safe=graph_safe,
            )
"""

REPLACEMENTS = [
    (
        "10 档 D 旧 Triton 路的 ppr 探针",
        '        mcs = _kv8_model_meta["max_cache_seq_len"]\n'
        "        ppr = max(1, -(-mcs // block_size)) if mcs else None\n",
        '        mcs = _kv8_model_meta["max_cache_seq_len"]\n'
        "        ppr = max(1, -(-mcs // block_size)) if mcs else None\n"
        "        _sg_ppr_trace(\n"
        '            "cmp_legacy_triton",\n'
        "            num_reqs=num_reqs,\n"
        "            rows=rows,\n"
        "            topk=topk,\n"
        "            per_req=per_req,\n"
        "            max_cache_seq_len=mcs,\n"
        "            legacy_ppr=(ppr if ppr is not None else -1),\n"
        "            block_size=block_size,\n"
        "        )\n",
    ),
    ("1 kv8_ori_plane 签名 + 图安全辅助函数", A_SIG, N_SIG),
    ("2 kv8_ori_plane 分支判据（上界分支）", A_BRANCH, N_BRANCH),
    ("3 _native_attention 的 bound 计算", A_CALL, N_CALL),
    ("4 kv8_ori_plane 调用点", A_CALL_ARGS, N_CALL_ARGS),
    ("5 _kv8_cmp_plane 调用点", A_CMP_CALL, N_CMP_CALL),
    ("6 _kv8_cmp_plane 签名", A_CMP_SIG, N_CMP_SIG),
    ("7 _kv8_cmp_plane 图安全分支", A_CMP_FALLBACK, N_CMP_FALLBACK),
    ("8 档 D kv8_ori_plane 包装路由", A_WRAP_ORI, N_WRAP_ORI),
    ("9 档 D _kv8_cmp_plane 包装路由", A_WRAP_CMP, N_WRAP_CMP),
]


def patch(text: str) -> str:
    for name, old, new in REPLACEMENTS:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"[apply_graphsafe] 锚点 {name} 出现 {n} 次（期望 1）")
        text = text.replace(old, new, 1)
    return text


def audit(text: str) -> list[str]:
    """Structural self-checks; returns the list of failures."""
    bad: list[str] = []

    def need(cond: bool, what: str) -> None:
        if not cond:
            bad.append(what)

    need("<<<<<<<" not in text, "有冲突标记")
    need(text.count("VLLM_V41_KV8_GRAPH_SAFE") == 1, "env 名不唯一")
    need("rows_bound: int | None = None," in text, "1 rows_bound 形参没加上")
    need("if rows_bound is not None:" in text, "2 上界分支没加上")
    need("rows_bound=rows_bound," in text, "4 调用点没传 rows_bound")
    need("graph_safe=graph_safe and not _kv8_cmp_legacy()," in text, "5 cmp 调用点没传 graph_safe")
    need(text.count("def _kv8_cmp_legacy() -> bool:") == 1, "5b 诊断开关 SG_CMP_LEGACY 缺失")
    need(text.count("_sg_ppr_trace(") == 4, "探针 _sg_ppr_trace 调用点应为 4 处")
    need(text.count('_os.environ.get("SG_TRACE_PPR"') == 1, "探针 env 名 SG_TRACE_PPR 应只读取 1 次")
    need("graph_safe: bool = False," in text, "6 cmp 形参没加上")
    need(
        "if graph_safe and per_req * block_size == topk and rows % num_reqs == 0:" in text,
        "7 cmp 图安全分支没加上",
    )
    need("repeat_interleave(reps, dim=0)" in text, "7 行→请求映射缺失")
    need(text.count("rows_bound=None,") == 1, "8 包装形参没加上")
    need(text.count("graph_safe=False,") == 1, "9 包装形参没加上")

    # 默认关 = 逐字旧行为：三条旧路径必须一字不动地还在。
    need(
        "        # Decode: one query row per request, so the span is exactly one window\n"
        "        # and covers at most two pages.  Fully device-side (capture safe).\n"
        "        pages_per_req = 2\n"
        "        window_start = (lens - window).clamp_min(0)" in text,
        "旧 decode 分支被动过",
    )
    need(
        "        pages_per_req = int(\n"
        "            ((lens - 1) // block_size - window_start // block_size + 1).max().item()\n"
        "        )" in text,
        "旧 prefill 分支（.item()）被动过 ⇒ prefill 行为不再保真",
    )
    need(
        "        used = int(cache_seq_lens[:num_reqs].max().item())" in text,
        "旧 cmp prefill 分支被动过",
    )

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        bad.append(f"语法错误：{exc}")
        return bad
    def _takes(node: ast.AST, name: str) -> bool:
        args = getattr(node, "args", None)
        if args is None:
            return False
        return any(a.arg == name for a in args.args + args.kwonlyargs)

    ori_fns = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "kv8_ori_plane" and _takes(n, "rows_bound")
    ]
    cmp_fns = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and _takes(n, "graph_safe") and "cmp_plane" in n.name
    ]
    need(len(ori_fns) == 2, f"带 rows_bound 的 kv8_ori_plane 应 2 个（本体+档 D 包装），实得 {len(ori_fns)}")
    need(len(cmp_fns) == 2, f"带 graph_safe 的 cmp 函数应 2 个（本体+档 D 包装），实得 {len(cmp_fns)}")
    need(
        any(
            isinstance(stmt, ast.If) and "rows_bound is not None" in ast.unparse(stmt.test)
            for fn in ori_fns
            for stmt in fn.body
        ),
        "AST 里没有 `if rows_bound is not None` 分支",
    )
    need(
        any(
            isinstance(stmt, ast.If) and "graph_safe and" in ast.unparse(stmt.test)
            for fn in cmp_fns
            for stmt in fn.body
        ),
        "AST 里没有 `if graph_safe and ...` 分支",
    )
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="X_integrate/pkg-kv8pf/.../attention/dsa_v41.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = Path(args.src)
    raw = src.read_text()
    out = patch(raw)
    bad = audit(out)
    for name, _, _ in REPLACEMENTS:
        print(f"  OK   锚点 {name}")
    if bad:
        for b in bad:
            print(f"  FAIL 自检：{b}")
        return 2
    Path(args.out).write_text(out)
    with tempfile.TemporaryDirectory() as td:
        py_compile.compile(args.out, cfile=str(Path(td) / "x.pyc"), doraise=True)
    print(f"[apply_graphsafe] src md5 = {hashlib.md5(raw.encode()).hexdigest()}  ({src})")
    print(f"[apply_graphsafe] out md5 = {hashlib.md5(out.encode()).hexdigest()}  ({args.out})")
    print(
        f"[apply_graphsafe] +{len(out.splitlines()) - len(raw.splitlines())} 行 / "
        f"锚点 {len(REPLACEMENTS)}/9 / 自检 {'PASS' if not bad else 'FAIL'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
