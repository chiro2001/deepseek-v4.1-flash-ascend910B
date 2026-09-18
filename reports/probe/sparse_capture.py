# SPDX-License-Identifier: Apache-2.0
"""稀疏选择插针 —— 事后取证用。

设计文档：../sparse_state_capture_design.md

用法（在 dsa_v41.py::_select_sparse_indices 里插一行）：

    from vllm_ascend.attention.sparse_capture import capture_selection
    capture_selection(self.role.layer_idx, selected, candidates, qr, positions,
                      is_candidate_source=self.role.is_candidate_source)

行为：
  * L1：每步每层一行 JSONL 元数据 + 廉价指纹，约 200 B/行
  * L2：张量快照，由 <root>/ENABLE 开关文件控制，环形缓冲

★ 安全设计（踩过的坑，务必保留）：
  1. **绝不在图捕获期间执行**。解码走 ACL graph 重放，捕获时任何 D2H 同步
     （`.item()` / `.cpu()`）都会把捕获挂死 → 整个服务起不来。
     守卫：`torch.npu.is_current_stream_capturing()`。
  2. **每次指纹只做一次 D2H 同步**：先在设备侧把统计量打包成一个张量，
     最后 `.tolist()` 一次取回。早期版本每个统计量都 `.item()`，
     8 个 rank × 8 层 × 每层 4 次同步 ⇒ 打断流水线 + HCCL 停顿。
  3. **默认关闭**，由开关文件在运行时打开；推理路径上零开销当且仅当关闭。
  4. **不落 /tmp**（tmpfs 吃内存，历史事故见 HANDOVER）。

  任何异常都吞掉（插针不能影响推理）。
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("V41_PROBE_DIR", os.path.expanduser("~/probe_capture")))
META_DIR = ROOT / "meta"          # L1 JSONL
TENSOR_DIR = ROOT / "tensors"     # L2 环形缓冲
ENABLE_FILE = ROOT / "ENABLE"     # L1+L2 开关（存在即开；内容可写 ring:N / l2）
L2_FILE = ROOT / "ENABLE_L2"      # 只开 L2（L1 常开时用）
STATS_FILE = ROOT / "STATS.json"  # 用量自检

# 只有 index-source 层会走到插针；非 prefill（解码）由图重放执行，Python 不会进来
PREFILL_MIN_TOKENS = int(os.environ.get("V41_PROBE_MIN_TOKENS", "0"))


def _rank_tag() -> str:
    """8 个 worker 共享一个挂载目录 ⇒ 文件名必须带 rank，否则互相覆盖。

    优先用进程组里的 rank；取不到就退回 pid（至少不冲突）。
    """
    for k in ("RANK", "VLLM_RANK", "LOCAL_RANK", "GROUP_RANK"):
        v = os.environ.get(k)
        if v:
            return f"r{v}"
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return f"r{dist.get_rank()}"
    except Exception:
        pass
    return f"p{os.getpid()}"


RANK = _rank_tag()

# ---- 预算红线（见设计文档 §3）----
META_MAX_BYTES = 5 * 1024**3      # L1 超 5 GB 轮转
TENSOR_MAX_BYTES = 1024**3        # L2 单次快照上限 1 GB
RING_SIZE_DEFAULT = 4             # 环形缓冲保留最近 N 个 step

_lock = threading.Lock()
_state: dict[str, Any] = {
    "step_counter": 0,
    "ring_idx": 0,
    "bytes_meta": 0,
    "bytes_tensor": 0,
    "starts": 0,
    "last_err": None,
    "l2_tier_downgrades": 0,
    "skipped_capturing": 0,
    "skipped_disabled": 0,
}


def _in_capture() -> bool:
    """图捕获期间必须完全跳过（否则 D2H 同步会挂死捕获）。

    这是本项目踩过的最贵的坑：一次没加守卫的捕获把服务卡死在启动阶段。
    """
    try:
        import torch
        if hasattr(torch, "npu") and hasattr(torch.npu, "is_current_stream_capturing"):
            return bool(torch.npu.is_current_stream_capturing())
    except Exception:
        pass
    return False


def _l1_enabled() -> bool:
    """L1 开关。ENABLE 或 ENABLE_L2 任一存在即开（便于只开 L2 时也保留元数据）。"""
    try:
        return ENABLE_FILE.exists() or L2_FILE.exists()
    except Exception:
        return False


# --------------------------------------------------------------------------
# L1 元数据
# --------------------------------------------------------------------------

def _pack_device_stats(t):
    """在**设备侧**把统计量算好并打包成一个 int64 张量，返回 (packed, n)。

    只做设备算子，不产生 D2H 同步；调用方最后一次性 `.tolist()` 取回。

    打包内容：[min, max, 采样 unique, 顺序敏感校验和, 元素数]
    """
    import torch

    if t is None:
        return None, 0
    flat = t.detach().reshape(-1)
    n = int(flat.numel())
    if n == 0:
        return None, 0
    f = flat.to(torch.int64)
    mn = f.min()
    mx = f.max()
    # 采样 unique（设备侧）
    step = max(1, n // 65536)
    try:
        uniq = torch.tensor(f[::step].unique().numel(), device=f.device, dtype=torch.int64)
    except Exception:
        uniq = torch.tensor(-1, device=f.device, dtype=torch.int64)
    # 顺序敏感校验和：位置加权求和（mod 一个素数避免溢出）
    idx = torch.arange(n, device=f.device, dtype=torch.int64)
    cs = ((f + 2) * ((idx % 1048573) + 1)).sum() % 9223372036854775783
    packed = torch.stack([mn, mx, uniq, cs, torch.tensor(n, device=f.device, dtype=torch.int64)])
    return packed, n


def _to_record(name: str, packed, n: int) -> dict:
    if packed is None:
        return {}
    try:
        mn, mx, uniq, cs, cnt = packed.tolist()
    except Exception:
        return {}
    return {
        f"{name}_min": int(mn),
        f"{name}_max": int(mx),
        f"{name}_uniq": int(uniq),
        f"{name}_n": int(cnt),
        f"{name}_hash": f"{int(cs) & 0xFFFFFFFFFFFF:012x}",
    }


def _check_dirty(selected, candidates) -> bool:
    """candidates 里是否出现非法值（H2 的直接检验）。

    A3 内核用 -1 表示"空槽"，其余应为 [0, num_blocks) 的块号。
    这里只做"有没有明显越界/异常"的粗检，不假设上界 —— 复用已经打包好的 min。
    """
    return False  # 真正的判定在 capture_selection 里用打包结果做，避免额外同步


def _rotate_if_needed(path: Path) -> None:
    try:
        if path.exists() and path.stat().st_size > META_MAX_BYTES:
            path.rename(path.with_suffix(".jsonl.old"))
    except Exception:
        pass


def capture_selection(
    layer_idx: int,
    selected,
    candidates,
    qr=None,
    positions=None,
    *,
    is_candidate_source: bool = False,
    step: int | None = None,
    extra: dict | None = None,
) -> None:
    """在 `attn.indexer.select()` 返回后调用。任何异常都被吞掉。"""

    # ---- 守卫 1：图捕获期间绝对不碰（D2H 会把捕获挂死）----
    if _in_capture():
        try:
            with _lock:
                _state["skipped_capturing"] += 1
        except Exception:
            pass
        return

    # ---- 守卫 2：默认关闭，零开销 ----
    if not _l1_enabled():
        try:
            with _lock:
                _state["skipped_disabled"] += 1
        except Exception:
            pass
        return

    try:
        # step 语义：一次 forward 会按层顺序命中 8 次 ⇒ 用"层回绕"计数 forward
        with _lock:
            if step is None:
                last = _state.get("_last_layer")
                if last is not None and layer_idx <= last:
                    _state["step_counter"] += 1
                elif last is None:
                    _state["step_counter"] = 1
                _state["_last_layer"] = layer_idx
                step = _state["step_counter"]
            rec: dict[str, Any] = {
                "ts": round(time.time(), 3),
                "step": step,
                "layer": layer_idx,
                "candidate_source": bool(is_candidate_source),
            }

        # 设备侧打包（不产生同步），最后一次取回
        sel_p, _ = _pack_device_stats(selected)
        cand_p, _ = _pack_device_stats(candidates)
        rec.update(_to_record("sel", sel_p, 0))
        rec.update(_to_record("cand", cand_p, 0))
        # dirty 判定用已取回的最小值，不再额外同步
        rec["dirty"] = bool(rec.get("sel_min", 0) < -1 or rec.get("cand_min", 0) < -1)
        if extra:
            rec.update(extra)

        with _lock:
            META_DIR.mkdir(parents=True, exist_ok=True)
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            day = time.strftime("%Y%m%d")
            path = META_DIR / f"sparse-{day}-{RANK}.jsonl"
            _rotate_if_needed(path)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
            _state["bytes_meta"] += len(line.encode())
    except Exception as e:  # 插针不能影响推理
        try:
            with _lock:
                _state["last_err"] = f"{type(e).__name__}: {e}"
        except Exception:
            pass
        return

    # L2 在锁外做（落盘慢，别阻塞其它 rank）
    try:
        _maybe_snapshot(layer_idx, selected, candidates, qr, positions, step)
    except Exception as e:
        try:
            with _lock:
                _state["last_err"] = f"L2 {type(e).__name__}: {e}"
        except Exception:
            pass


# --------------------------------------------------------------------------
# L2 张量快照（按需）
# --------------------------------------------------------------------------

def _l2_enabled() -> bool:
    return ENABLE_FILE.exists()


def _ring_size() -> int:
    try:
        txt = ENABLE_FILE.read_text(encoding="utf-8").strip()
        for tok in txt.replace(",", " ").split():
            if tok.startswith("ring:"):
                return max(1, min(64, int(tok.split(":", 1)[1])))
    except Exception:
        pass
    return RING_SIZE_DEFAULT


def _maybe_snapshot(layer_idx, selected, candidates, qr, positions, step) -> None:
    if not _l2_enabled():
        return
    # 硬闸门（防呆）：连文件数上限都不满足就直接放弃，不去算 size_hint。
    # 历史上这里出过 339 GB 的事故 —— 见下方 ring 命名的注释。
    if _tensor_dir_files() > _ring_size() * 8 * 16:
        with _lock:
            _state["l2_tier_downgrades"] += 1
        return
    size_hint = 0
    for t in (selected, candidates, qr):
        if t is not None:
            try:
                size_hint += t.numel() * t.element_size()
            except Exception:
                pass
    if size_hint > TENSOR_MAX_BYTES:
        with _lock:
            _state["l2_tier_downgrades"] += 1
        candidates = None  # 降级：只留 selected
        size_hint = 0
        for t in (selected, qr):
            if t is not None:
                try:
                    size_hint += t.numel() * t.element_size()
                except Exception:
                    pass
        if size_hint > TENSOR_MAX_BYTES:
            return

    import torch

    payload = {"layer": layer_idx, "step": step}
    for name, t in (("selected", selected), ("candidates", candidates)):
        if t is not None:
            payload[name] = t.detach().to("cpu")
    if qr is not None:
        payload["qr"] = qr.detach().to("cpu")
    if positions is not None:
        payload["positions"] = positions.detach().to("cpu")

    TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        idx = _state["ring_idx"] % _ring_size()
        _state["ring_idx"] += 1
    # ★ 文件名必须**只由 (ring_idx, layer, rank) 决定**，不能带 step。
    #   带 step 会让每次写入都是新文件 ⇒ 环形缓冲退化成无限增长。
    #   事故记录：2026-09-18 03:0x，写满 339 GB / 35712 个文件（已清理）。
    path = TENSOR_DIR / f"ring{idx:03d}-L{layer_idx:02d}-{RANK}.pt"
    torch.save(payload, path)
    try:
        with _lock:
            files = list(TENSOR_DIR.glob("ring*.pt"))
            _state["bytes_tensor"] = sum(p.stat().st_size for p in files)
            _state["tensor_files"] = len(files)
            n = _state["tensor_files"]
        # 每 64 次写入做一次自愈裁剪（防历史遗留 + 防命名意外）
        if n % 64 == 0:
            prune_tensors()
    except Exception:
        pass


def _tensor_dir_files() -> int:
    try:
        return sum(1 for _ in TENSOR_DIR.glob("ring*.pt"))
    except Exception:
        return 0


def prune_tensors(keep_ring: int | None = None) -> dict:
    """把张量目录裁到"环大小 × 层数 × rank 数"上界内。

    任何超出上界的文件（含历史遗留的 step-命名文件）一律删除。
    可在外部定期调用，也可在 dump_stats() 里触发。
    """
    ring = keep_ring or _ring_size()
    limit_files = ring * 8 * 16          # 8 个 index-source 层 × 最多 16 个 rank 余量
    try:
        files = sorted(TENSOR_DIR.glob("ring*.pt"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        return {"pruned": 0}
    pruned = 0
    for p in files[limit_files:]:
        try:
            p.unlink()
            pruned += 1
        except Exception:
            pass
    return {"pruned": pruned, "kept": len(files) - pruned, "limit_files": limit_files}


def dump_stats() -> dict:
    """给外部脚本/HTTP 端点看用量。"""
    with _lock:
        out = dict(_state)
    out["l2_enabled"] = _l2_enabled()
    out["ring_size"] = _ring_size()
    out["root"] = str(ROOT)
    out["rank"] = RANK
    out["prune"] = prune_tensors()          # 顺手自愈：超界文件删掉
    try:
        out["meta_files"] = sorted(p.name for p in META_DIR.glob(f"sparse-*-{RANK}.jsonl"))
        out["tensor_files_rank"] = len(list(TENSOR_DIR.glob(f"ring*-{RANK}.pt")))
    except Exception:
        pass
    try:
        STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATS_FILE.with_name(f"STATS-{RANK}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass
    return out


def reset_step_counter() -> None:
    """每个新 request 开始时调用（若接得上）。"""
    with _lock:
        _state["step_counter"] = 0
