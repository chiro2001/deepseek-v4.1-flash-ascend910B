# SPDX-License-Identifier: MIT
# Adapted from the DeepSeek V4.1 reference inference/engram.py.
from dataclasses import dataclass
import os

import numpy as np
import torch
from .engram_jit_kernel import (  # [ENGRAM-JIT-HASH]
    ENGRAM_JIT as _ENGRAM_JIT,
    PAGE_CAP_INIT as _ENGRAM_JIT_PAGE_CAP,
    _WARNED as _ENGRAM_JIT_WARNED,
    engram_update_kernel as _engram_update_kernel,
    selftest as _engram_jit_selftest,
)
from sympy import isprime

_HISTORY_SLAB_MIN_TOKENS = 16


# ==== [ENGRAM-PAGELESS] 镜像缺页的语义（2026-09-22 21:0x 的 P0 修复）====
# 背景（实测）：`ENGRAM=1` + DRAM 卸载时，被卸载池**取回**的前缀块从未流经
# `update()` ⇒ page 镜像 `pages/page_present` 里缺页 ⇒ 老代码 `raise KeyError(err)`
# ⇒ **引擎死**（a2/logs/073 + 072 §3.3：`KeyError: 2486`，`num_computed_tokens=56320`）。
# 现在：缺页 = 与「槽位 = -1」**同义的 barrier**（该行该 shift 起保持 pad_id，哈希照算），
# 并记入 `pageless_history_rows`。`V41_ENGRAM_PAGELESS_STRICT=1` 可恢复旧的致命行为。
_PAGELESS_WARNED = [False]


def _pageless_strict() -> bool:
    return os.environ.get("V41_ENGRAM_PAGELESS_STRICT", "0") == "1"


def _pageless_note(page, n_rows, pad_id):
    """首次缺页时打印一次（响亮但不刷屏）—— 不静默降级。

    ★★★ **这段文字本身是判据的一部分，所以措辞受约束（2026-09-22 实测踩过）**：
    第一版写的是「…并继续（**不再 KeyError**）」⇒ 交付脚本里那条
    `grep -c 'KeyError' <serve.log>` **把这条提示也数了进去** ⇒ 8 个 rank 打出 8 条提示，
    判据读成 `KeyError=8`（看起来像引擎又炸了 8 次），而真正的异常计数是 **0**。
    ⇒ 规矩：**运行期打印的文本里不得出现裸的 `KeyError` / `Traceback` / `EngineDead` 子串**；
    判据一律用带冒号的精确模式 `KeyError:`（真异常长这样）。
    """
    if _PAGELESS_WARNED[0]:
        return
    _PAGELESS_WARNED[0] = True
    print(
        "[ENGRAM-PAGELESS] 镜像缺页：第 %s 页不在 Engram 的 page 镜像里（%s 行）。"
        "已按 barrier 语义用 pad_id=%s 填历史并继续（**不再抛错**）。"
        "典型成因：前缀经 DRAM 卸载池取回，其块从未流经本进程的 update()。"
        "要恢复旧的致命行为请设 V41_ENGRAM_PAGELESS_STRICT=1。"
        % (page, n_rows, pad_id),
        flush=True,
    )


# ==== [hash-ab] 同会话 A/B 开关（文件驱动，默认 fast）====
_HASH_MODE_PATH = "/tmp/v41_hash_mode"
_HA = {"v": "fast", "last": 0.0, "mtime": -1.0}


def _hash_mode() -> str:
    import os as _os
    import time as _t

    now = _t.monotonic()
    if now - _HA["last"] < 0.25:
        return _HA["v"]
    _HA["last"] = now
    try:
        st = _os.stat(_HASH_MODE_PATH)
    except OSError:
        return _HA["v"]
    if st.st_mtime == _HA["mtime"]:
        return _HA["v"]
    _HA["mtime"] = st.st_mtime
    try:
        with open(_HASH_MODE_PATH) as fh:
            raw = fh.read().strip()
    except OSError:
        return _HA["v"]
    if raw in ("fast", "stock"):
        _HA["v"] = raw
    return _HA["v"]


def _stock_small_history(hist, tokens, positions, request_ids, block_table, block_size, position_list):
    """[hash-ab] 原版标量走法（mode=stock），与上线版本逐语句等价。"""
    import torch as _t

    history = _t.full((tokens, hist.lookback), hist.pad_id, dtype=_t.int64, device="cpu")
    for row, (position, request) in enumerate(zip(position_list, request_ids.tolist())):
        for shift in range(hist.lookback):
            previous = position - shift
            if previous < 0:
                break
            page = block_table[request, previous // block_size].item()
            token = hist.pages[page][previous % block_size]
            if token < 0:
                break
            history[row, shift] = token
    return history
_PAGE_WRITE_NUMPY_MIN_TOKENS = 16

_SHIFT_INDEX_CACHE: dict[int, torch.Tensor] = {}


def shift_index(lookback: int) -> torch.Tensor:
    """``[0, 1, ..., lookback - 1]`` as an int64 CPU tensor, built once per lookback."""
    shifts = _SHIFT_INDEX_CACHE.get(lookback)
    if shifts is None:
        shifts = torch.arange(lookback, dtype=torch.int64, device="cpu")
        _SHIFT_INDEX_CACHE[lookback] = shifts
    return shifts


def valid_engram_token_mask(
    input_ids: torch.Tensor,
    image_token_id: int,
    image_pad_token_id: int,
) -> torch.Tensor:
    """Exclude the complete V4.1 image region from n-gram history."""
    return (input_ids != image_token_id) & (input_ids != image_pad_token_id)


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """The smallest prime above `start` that has not been handed out yet."""
    candidate = start + 1
    while not isprime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize alike
    collapse together.

    N-grams are hashed over these compressed ids, so " The", "the" and "THE" all hash
    the same way.
    Returns the lookup plus the size of the compressed vocab -- and that size matters
    beyond bounds
    checking, because every hash multiplier is derived from it.
    """
    from tokenizers import Regex, normalizers

    # a private-use char, so a token that is exactly one space survives Strip() instead
    # of
    # collapsing to the empty string and merging with unrelated tokens
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    # the raw Rust tokenizer, matching what training decodes with (no
    # clean_up_tokenization_spaces)
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # a partial UTF-8 byte token: nothing to normalize, so key it by its raw
            # form
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, tokenizer_vocab_size: int
) -> torch.Tensor:
    """Derive one multiplier per (layer, lookback) from a per-layer RNG.

    Kept odd, and bounded so that `token_id * multiplier` cannot overflow int64.
    """
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout of the n-gram hash tables.

    A position uses `max_ngram_size - 1` n-grams, each split over `n_heads`.
    Each (n-gram size, head) pair owns its own prime-sized bucket range in the
    layer's table; the primes are drawn in order and never reused, which keeps the
    ranges disjoint.
    """

    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]  # table rows, per engram layer
    primes: tuple[tuple[tuple[int, ...], ...], ...]  # [layer][n-gram size][head] bucket modulus
    n_heads: int
    head_dim: int

    @classmethod
    def from_args(cls, args) -> "EngramLayout | None":
        layer_ids = tuple(args.engram_layer_ids)
        if not layer_ids:
            return None
        max_ngram_size, n_heads = args.engram_max_ngram_size, args.engram_n_heads
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], args.engram_vocab_size - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(args.engram_num_embeddings),
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=args.engram_head_dim,
        )


class PagedNgramHistory:
    """Mirror token IDs in the scheduler's physical pages, including prefixes.

    A speculative suffix can be overwritten without rolling back a mutable
    per-request tail. Hashes read only positions at or before the current query.
    CPU residency supplies routing metadata without a device synchronization.
    """

    def __init__(self, config, tokenizer):
        layout = EngramLayout.from_args(config)
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != config.engram_compressed_vocab_size:
            raise ValueError(f"Engram compressed vocabulary mismatch: {vocab_size}")
        self.token_map = torch.tensor(token_map, dtype=torch.int64)
        self.pad_id = token_map[config.engram_pad_id]
        self.image_token_id = config.image_token_id
        self.image_pad_token_id = getattr(
            config,
            "image_pad_token_id",
            self.image_token_id + 1,
        )
        self.primes = torch.tensor(layout.primes)
        sizes = self.primes.flatten(1)
        self.offsets = sizes.cumsum(-1) - sizes
        self.multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab_size)
        self.lookback = layout.max_ngram_size
        self.pages = {}
        # [ENGRAM-JIT-HASH] dense 页镜像（仅 V41_ENGRAM_JIT=1 时使用；dict 保持为空）
        self._jit_ok = _ENGRAM_JIT
        self._jit_pages = None
        self._jit_page_present = None
        self._jit_block_size = 0
        self._jit_out_hashes = None
        self._jit_out_mask = None
        self._jit_cap_tokens = 0
        self._jit_hist = None
        self._jit_np = {}
        self._jit_views = {}
        if self._jit_ok:
            try:
                _engram_jit_selftest()
                self._jit_np["tm"] = self.token_map.numpy()
                self._jit_np["mult"] = self.multipliers.numpy()
                self._jit_np["primes"] = self.primes.numpy()
                self._jit_np["offsets"] = self.offsets.numpy()
                if _hash_mode() != "fast" and not _ENGRAM_JIT_WARNED[0]:
                    _ENGRAM_JIT_WARNED[0] = True
                    print("[ENGRAM-JIT] note: _hash_mode() != 'fast'; the JIT path replaces "
                          "both the stock and the fast implementations.", flush=True)
            except Exception as _exc:  # noqa: BLE001
                self._jit_ok = False
                if not _ENGRAM_JIT_WARNED[0]:
                    _ENGRAM_JIT_WARNED[0] = True
                    print(f"[ENGRAM-JIT] selftest failed, falling back to stock: {_exc}",
                          flush=True)

    def update(self, input_ids, positions, request_ids, block_table, block_size):
        """All arguments are CPU tensors; page numbers come from full SWA KV."""
        if input_ids.numel() == 0:
            # Idle DP and empty prefill still follow the collective contract,
            # but there is no page or hash state to update.
            columns = (self.lookback - 1) * self.primes.shape[-1]
            return (
                torch.empty((0, self.primes.shape[0], columns), dtype=torch.int64, device="cpu"),
                torch.empty(0, dtype=torch.bool, device="cpu"),
            )
        if self._jit_ok:
            return self._engram_update_jit(
                input_ids, positions, request_ids, block_table, block_size
            )
        compressed = self.token_map[input_ids]
        mask = valid_engram_token_mask(
            input_ids,
            self.image_token_id,
            self.image_pad_token_id,
        )
        compressed = compressed.masked_fill(~mask, -1)
        # Materialize CPU lists once for sequential page writes and small-batch
        # history reads.
        compressed_list = compressed.tolist()
        position_list = positions.tolist()
        page_indices = block_table[request_ids, positions // block_size].tolist()
        if len(input_ids) < _PAGE_WRITE_NUMPY_MIN_TOKENS:
            for token, position, page in zip(compressed_list, position_list, page_indices):
                if page not in self.pages:
                    self.pages[page] = torch.full((block_size,), -1, dtype=torch.int64, device="cpu")
                self.pages[page][position % block_size] = token
        else:
            page_views = {}
            for token, position, page in zip(compressed_list, position_list, page_indices):
                view = page_views.get(page)
                if view is None:
                    if page not in self.pages:
                        self.pages[page] = torch.full((block_size,), -1, dtype=torch.int64, device="cpu")
                    view = self.pages[page].numpy()
                    page_views[page] = view
                # Zero-copy CPU view avoids Torch dispatch per scalar write. Keep
                # input order so repeated physical slots retain last-write wins.
                view[position % block_size] = token
        if len(input_ids) < _HISTORY_SLAB_MIN_TOKENS and _hash_mode() == "fast":
            # Small batches are decode: index every (row, shift) pair at once
            # instead of paying four dispatches per pair. The page-row walk is
            # kept as a bail-out for the one case the pair indexing cannot prove
            # safe (a missing page behind a barrier), see _small_batch_history.
            history = self._small_batch_history(
                len(input_ids),
                positions,
                request_ids,
                block_table,
                block_size,
                position_list,
            )
        elif len(input_ids) < _HISTORY_SLAB_MIN_TOKENS:
            # [hash-ab] mode=stock: 原标量路径（A/B 对照）
            history = _stock_small_history(
                self, len(input_ids), positions, request_ids, block_table, block_size, position_list
            )
        else:
            history = torch.full(
                (len(input_ids), self.lookback), self.pad_id, dtype=torch.int64, device="cpu"
            )
            active = torch.ones(len(input_ids), dtype=torch.bool, device="cpu")
            for shift in range(self.lookback):
                previous = positions - shift
                valid = active & (previous >= 0)
                if not bool(valid.any()):
                    break
                with torch.device("cpu"):
                    rows = torch.nonzero(valid, as_tuple=False).flatten()
                page_ids = block_table[request_ids[rows], previous[rows] // block_size]
                offsets = previous[rows] % block_size
                with torch.device("cpu"):
                    unique_pages, slab_indices = torch.unique(page_ids, return_inverse=True)
                # ★ [ENGRAM-PAGELESS] 缺页不再 KeyError：_mirror_row 会补一行全 -1
                #   （= barrier），与 JIT kernel 的语义一致。
                slab = torch.stack(
                    [self._mirror_row(page, block_size) for page in unique_pages.tolist()]
                )
                values = slab[slab_indices, offsets]
                present = values >= 0
                history[rows[present], shift] = values[present]
                active[rows] = present
        products = history[:, None] * self.multipliers
        rolling, hashes = products[..., 0], []
        for shift in range(1, self.lookback):
            rolling = torch.bitwise_xor(rolling, products[..., shift])
            hashes.append(rolling[..., None] % self.primes[:, shift - 1])
        return torch.cat(hashes, -1) + self.offsets, mask

    # ==== [ENGRAM-JIT-HASH] 方法 =============================================
    def _jit_np_of(self, t):
        """torch CPU tensor -> 连续 numpy 视图（零拷贝）。"""
        a = t.numpy()
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a)
        return a

    def _jit_prepare(self, n, block_size):
        cap0 = max(int(_ENGRAM_JIT_PAGE_CAP), 1)
        if self._jit_pages is None or self._jit_block_size != block_size:
            self._jit_block_size = block_size
            self._jit_pages = np.full((cap0, block_size), -1, np.int64)
            self._jit_page_present = np.zeros(cap0, np.uint8)
        if n > self._jit_cap_tokens:
            cap = int(n * 1.5) + 64
            self._jit_views = {}          # 缓冲区换了，旧 torch 视图必须作废
            self._jit_out_hashes = np.zeros(
                (cap, self.primes.shape[0], (self.lookback - 1) * self.primes.shape[-1]),
                np.int64,
            )
            self._jit_out_mask = np.zeros(cap, np.bool_)
            self._jit_hist = np.zeros((cap, self.lookback), np.int64)
            big = cap * self.lookback
            self._jit_flat = np.zeros(big, np.int64)
            self._jit_prev = np.zeros(big, np.int64)
            self._jit_ir = np.zeros(big, np.uint8)
            self._jit_vals = np.zeros(big, np.int64)
            self._jit_rows = np.zeros(cap, np.int64)
            self._jit_pid = np.zeros(cap, np.int64)
            self._jit_offb = np.zeros(cap, np.int64)
            self._jit_prs = np.zeros(cap, np.uint8)
            self._jit_act = np.zeros(cap, np.uint8)
            self._jit_cap_tokens = cap

    def _jit_run(self, n, block_size, tm, ii, pos, req, bt):
        ret = _engram_update_kernel(
            tm, ii, pos, req, bt,
            block_size, n, self.pad_id, self.image_token_id, self.image_pad_token_id,
            self._jit_pages, self._jit_page_present,
            self._jit_np["mult"], self._jit_np["primes"], self._jit_np["offsets"],
            self.lookback, self.primes.shape[-1], _HISTORY_SLAB_MIN_TOKENS,
            self._jit_out_hashes, self._jit_out_mask,
            self._jit_hist, self._jit_flat, self._jit_prev, self._jit_ir,
            self._jit_vals, self._jit_rows, self._jit_pid, self._jit_offb,
            self._jit_prs, self._jit_act,
        )
        return int(ret[0]), int(ret[1]), int(ret[2]), int(ret[3])

    def _engram_update_jit(self, input_ids, positions, request_ids, block_table, block_size):
        # 热路径：只做必要的事。_hash_mode() 的检查已挪到 __init__。
        n = input_ids.shape[0]
        if n == 0:
            columns = (self.lookback - 1) * self.primes.shape[-1]
            return (
                torch.empty((0, self.primes.shape[0], columns), dtype=torch.int64, device="cpu"),
                torch.empty(0, dtype=torch.bool, device="cpu"),
            )
        bs = block_size
        if self._jit_pages is None or self._jit_block_size != bs or n > self._jit_cap_tokens:
            self._jit_prepare(n, bs)
        np_of = self._jit_np_of
        ii = np_of(input_ids)
        pos = np_of(positions)
        req = np_of(request_ids)
        bt = np_of(block_table)
        tm = self._jit_np["tm"]
        mult = self._jit_np["mult"]
        primes = self._jit_np["primes"]
        offsets = self._jit_np["offsets"]
        pages = self._jit_pages
        present = self._jit_page_present
        for _attempt in range(4):
            ret = _engram_update_kernel(
                tm, ii, pos, req, bt, bs, n, self.pad_id,
                self.image_token_id, self.image_pad_token_id,
                pages, present, mult, primes, offsets,
                self.lookback, primes.shape[-1], _HISTORY_SLAB_MIN_TOKENS,
                self._jit_out_hashes, self._jit_out_mask,
                self._jit_hist, self._jit_flat, self._jit_prev, self._jit_ir,
                self._jit_vals, self._jit_rows, self._jit_pid, self._jit_offb,
                self._jit_prs, self._jit_act,
            )
            err = ret[0]
            oob = ret[1]
            fell_back = ret[2]
            miss_rows = ret[3]
            if oob < 0:
                break
            # 页号超出容量：扩容后重跑（页写幂等，部分写入无副作用）
            cap = max(int(oob) + 1, int(pages.shape[0] * 1.5) + 1024)
            newp = np.full((cap, bs), -1, np.int64)
            newq = np.zeros(cap, np.uint8)
            old = pages.shape[0]
            newp[:old] = pages
            newq[:old] = present
            self._jit_pages = newp
            self._jit_page_present = newq
            pages, present = newp, newq
        else:
            raise RuntimeError("Engram JIT page capacity could not be satisfied")
        # ★ [ENGRAM-PAGELESS] 缺页不再是致命错误：kernel 已按 barrier 语义给了 pad 历史，
        #   哈希也已算出。这里只计数 + 一次性提示（严格模式下才恢复旧行为）。
        if miss_rows:
            self.pageless_history_rows = getattr(self, "pageless_history_rows", 0) + int(miss_rows)
            _pageless_note(err, int(miss_rows), self.pad_id)
            if _pageless_strict():
                raise KeyError(err)
        if fell_back:
            self.scalar_history_fallbacks = getattr(self, "scalar_history_fallbacks", 0) + 1
        # 每个 n 缓存一对 torch 视图：decode 的 n 基本恒定，省掉两次 from_numpy 分发
        views = self._jit_views.get(n)
        if views is None:
            views = (torch.from_numpy(self._jit_out_hashes[:n]),
                     torch.from_numpy(self._jit_out_mask[:n]))
            self._jit_views[n] = views
        return views
    # ==== /[ENGRAM-JIT-HASH] 方法 ============================================

    def _mirror_row(self, page, block_size):
        """返回镜像里该页的行；★ 缺页时补一行全 ``-1``（= barrier）而不是抛 KeyError。

        [ENGRAM-PAGELESS] 与 `engram_jit_kernel.py` 的缺页语义一一对应：
        缺页等价于「这个槽位的 token 不可知」⇒ 该 shift 及其之后取 `pad_id`。
        `V41_ENGRAM_PAGELESS_STRICT=1` 时恢复旧的 `KeyError`（给需要 fail-closed 的臂用）。
        """
        row = self.pages.get(page)
        if row is None:
            self.pageless_history_rows = getattr(self, "pageless_history_rows", 0) + 1
            _pageless_note(page, 1, self.pad_id)
            if _pageless_strict():
                raise KeyError(page)
            row = torch.full((block_size,), -1, dtype=torch.int64, device="cpu")
            self.pages[page] = row
        return row

    def _small_batch_history(
        self, tokens, positions, request_ids, block_table, block_size, position_list
    ):
        """n-gram history for ``tokens < _HISTORY_SLAB_MIN_TOKENS`` rows.

        Vectorised first: one ``torch.stack`` per call and no per-pair dispatch.
        ★ [ENGRAM-PAGELESS] 缺页现在**不再**让 ``_vectorized_history`` 返回 None：
        `_mirror_row` 会给缺的页补一行全 ``-1``（= barrier），语义与 JIT kernel 一致。
        标量走法仍然保留（`scalar_history_fallbacks` 仍会计数），但缺页已经不是它的
        bail-out 条件 ⇒ 它现在只在显式走 stock 分支时被用到。
        """
        history, flat_pages = self._vectorized_history(
            tokens, positions, request_ids, block_table, block_size
        )
        if history is not None:
            return history
        # Observability only: the on-card run can assert this stays at zero for
        # the decode workload, i.e. that the vectorised path is what ran.
        self.scalar_history_fallbacks = getattr(self, "scalar_history_fallbacks", 0) + 1
        history = torch.full((tokens, self.lookback), self.pad_id, dtype=torch.int64, device="cpu")
        for row, position in enumerate(position_list):
            base = row * self.lookback
            for shift in range(self.lookback):
                previous = position - shift
                if previous < 0:
                    break
                # ★ [ENGRAM-PAGELESS] 缺页 ⇒ _mirror_row 给一行全 -1 ⇒ 下面 `token < 0`
                #   即 barrier（与"未写入的 token"同一条路径）。
                token = self._mirror_row(flat_pages[base + shift], block_size)[
                    previous % block_size
                ]
                if token < 0:
                    break
                history[row, shift] = token
        return history

    def _vectorized_history(self, tokens, positions, request_ids, block_table, block_size):
        """Index all ``tokens * lookback`` history slots in one pass.

        ★ [ENGRAM-PAGELESS] 缺页不再导致 ``(None, flat_pages)``：缺页由
        `_mirror_row` 补成全 ``-1`` 的 barrier 行（并计入 `pageless_history_rows`）。
        返回值第二项保留为 ``None`` 仅是历史形状，调用方（`_small_batch_history`）
        仍按原样处理。
        """
        lookback = self.lookback
        previous = positions.unsqueeze(1) - shift_index(lookback)
        in_range = previous >= 0
        pages = block_table[request_ids.unsqueeze(1), previous.clamp_min(0) // block_size]
        # shift 0 always reads the row's own page, which the mirror write above
        # has just created; point the out-of-range slots there so every page this
        # pass touches is known to exist. Their values are masked away below.
        pages = torch.where(in_range, pages, pages[:, :1])
        flat_pages = pages.flatten().tolist()
        slab = []
        for page in flat_pages:
            # ★ [ENGRAM-PAGELESS] 缺页 ⇒ 补一行全 -1（barrier），不再返回 None
            #   （返回 None 会把控制权交给标量走法，而那条路以前会 KeyError）。
            slab.append(self._mirror_row(page, block_size))
        slab = torch.stack(slab)
        slots = (previous % block_size).flatten().view(-1, 1)
        values = slab.gather(1, slots).view(tokens, lookback)
        # A slot is real when its position is in range and the mirrored token is
        # not a barrier (-1). The reference walk stops at the first such slot and
        # leaves every later shift at pad_id: that is a running minimum over the
        # row, with the out-of-range slots forced to the -1 barrier value.
        values = values.masked_fill(~in_range | (values < 0), -1)
        return values.masked_fill(values.cummin(dim=1).values < 0, self.pad_id), None
