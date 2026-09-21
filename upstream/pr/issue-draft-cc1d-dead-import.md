# Issue 草稿（**未提交**）—— `causal_conv1d_update_npu` 已删除，但绑定仍在尝试

- **目标仓**：`vllm-project/vllm-ascend`
- **形态**：issue（BugFix / 代码卫生 + 一个需要 maintainer 回答的问题）
- **建议标题**：`[Bug][Ops] patch_triton.py still imports causal_conv1d_update_npu, removed by #14620`
- **建议标签**：`bug`
- **基线**：`main` = `c173a64a44dec4ba97aaba6277b1dfc1562eda19`
- **状态**：草稿。**未提交**（用户要求：未经允许不发起 issue/PR）

---

## 正文（英文，直接可贴）

### What happened

`vllm_ascend/patch/worker/patch_triton.py` contains a `try` block that binds an NPU
Triton kernel for the KDA layers:

```python
# npu_kda_causal_conv1d_triton: the PyTorch fallback calls .item() per
# request, which aborts ACL graph capture. The upstream NPU Triton kernel has
# no host sync and accepts the spec-decode kwargs the fallback had to drop.
try:
    from vllm_ascend.ops.triton.mamba.causal_conv1d import (  # type: ignore[attr-defined]
        causal_conv1d_update_npu as _cc1d_update_npu,
    )

    _cc1d.causal_conv1d_update = _cc1d_update_npu
    logger.debug("Bound the NPU Triton causal_conv1d_update for the KDA layers.")
except Exception as _cc1d_err:
    logger.warning(
        "NPU Triton causal_conv1d_update is unavailable (%s); falling back to the"
        " PyTorch implementation, which syncs per request and therefore stalls ACL"
        " graph capture at decode-FULL.",
        _cc1d_err,
    )
```

That symbol no longer exists: **#14620** (`f0a93895`, *"[Refactor][Ops] Remove
causal_conv1d_update_npu"*, 2026-08-25) deleted the kernel and the wrapper, and its
description states it *"Verified there are no remaining references to
causal_conv1d_update_npu"*. The module now exports only `PAD_SLOT_ID` and
`extract_last_width`:

```console
$ grep -rn "causal_conv1d_update_npu" --include="*.py" vllm_ascend/
vllm_ascend/patch/worker/patch_triton.py:326:        causal_conv1d_update_npu as _cc1d_update_npu,

$ cat vllm_ascend/ops/triton/mamba/causal_conv1d.py
# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID  # type: ignore

__all__ = ["PAD_SLOT_ID", "extract_last_width"]


def extract_last_width(x, start_loc, width):
    ...
```

So the `try` cannot succeed. The block was re-added nine days after the removal, by
**#15127** (*"[Feature][Model] Support GLM-5.3-Flash on Ascend 950"*, `78f3a90f2`),
which in the same commit also binds the PyTorch implementation at line 57
(`_cc1d.causal_conv1d_update = _npu_causal_conv1d_update`).

### Evidence that this fires in production

The `except` branch runs on every worker at startup. From an 8-card serve log (TP8,
`patch_triton.py:332`, image-baked `/vllm-workspace/vllm-ascend`):

```
WARNING 09-15 13:48:21 [patch_triton.py:332] NPU Triton causal_conv1d_update is unavailable
  (cannot import name 'causal_conv1d_update_npu' from
   'vllm_ascend.ops.triton.mamba.causal_conv1d'
   (/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/mamba/causal_conv1d.py));
  falling back to the PyTorch implementation, which syncs per request and therefore
  stalls ACL graph capture at decode-FULL.
```

**8 occurrences, one per worker** (13:48:21 → 13:50:19) — and the block is reached for
*every* model, because `patch/worker/__init__.py` imports `patch_triton` unconditionally
under `HAS_TRITON`.

### Minimal reproduction (no NPU required to reason about it)

```bash
python repro_cc1d_dead_import.py --repo /path/to/vllm-ascend          # loads the real file
# or, on a machine without vllm installed:
python repro_cc1d_dead_import.py --repo /path/to/vllm-ascend --stub-vllm
```

```console
  module file : vllm_ascend/ops/triton/mamba/causal_conv1d.py
  file size   : 430 bytes, 15 lines
  declares `causal_conv1d_update_npu`? NO
  __all__     : = ["PAD_SLOT_ID", "extract_last_width"]
  public names: ['PAD_SLOT_ID', 'extract_last_width', 'torch']

  running the statement patch_triton.py actually runs:
      from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_update_npu as _cc1d_update_npu

  -> ImportError-equivalent: AttributeError("module '...causal_conv1d' has no attribute 'causal_conv1d_update_npu'")

  VERDICT: the try block CANNOT succeed on this checkout.
```

### Why it matters beyond "dead code"

The warning itself asserts a performance consequence — *"syncs per request and therefore
stalls ACL graph capture at decode-FULL"* — and the fallback it names really is the one
in use. `vllm_ascend/ops/causal_conv1d.py` selects the state slot with a host sync:

```python
    def _select_state(i: int) -> torch.Tensor | None:
        if conv_state_indices is not None:
            idx = int(conv_state_indices[i].item())   # <- per-request host sync
            if idx == pad_slot_id:
                return None
```

and `conv_state_indices` is always supplied on the decode path (e.g.
`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`,
`conv_state_indices=spec_state_indices_tensor[...]` / `non_spec_state_indices_tensor[...]`).

So for any model with GDN/KDA layers the sync is on the hot path, and the "triton kernel
with no host sync" the comment relies on is not there.

### The question we would like answered

We are not proposing to restore the operator — #14620 removed it deliberately, and its
own description notes the removed tests "documented an overflow case and a probabilistic
failure". What we cannot tell from the code is which of these is intended:

1. **the block is stale** — the intent was "use the Triton kernel if present, fall back
   otherwise", and since it can no longer be present the block (and the warning) should be
   deleted; or
2. **the Triton fast path is still wanted** for the KDA layers, in which case the binding
   needs to be restored (or replaced by an AscendC / sync-free equivalent), and the current
   situation is a silent regression against #15127's intent.

If (2), it would also be worth stating whether the `.item()` above actually breaks graph
capture for GLM-5.3 on Ascend 950 today — that is the thing the warning claims and that we
have no hardware to check.

### Environment

* `vllm-ascend` main `c173a64a44dec4ba97aaba6277b1dfc1562eda19`
* observed on 8 × Ascend 910B3 (A2) and 8 × Ascend 910C (A3/910C class), CANN 9.1.0
* the reproduction above also runs on a single Ascend 910 (910C class, `19e5:d803`)

---

## 提交前核对清单

| 项 | 状态 |
|---|---|
| 基线 SHA 写对 | ✅ `c173a64a` |
| `#14620` / `#15127` 引用与逐字引文 | ✅ 已从本地 checkout 核对 |
| 复现命令可独立跑 | ✅ 脚本已在单卡机实测（`logs/raw/22-cc1d-repro/`） |
| 不指控、只问 | ✅ 用"which of these is intended" |
| 不声称我们更强 / 不提 W4A8 | ✅ |
| **用户授权** | ❌ **尚未获得 → 不得提交** |
