# SPDX-License-Identifier: Apache-2.0
# [CED-SINGLE-COPY] 2026-10-10：TP8 下 DRAM 池只存**一份** KV。
#
# 背景（见 docs/CED-P-DRAM-OFFLOAD-PLAN.md 与 a2/logs/031、032）：
#   * 单 tier 的 NPU worker 给**每张 canonical 张量、每个 rank**各分配一块私有宿主内存
#     ⇒ 池里同一份 KV 存 8 份（8 个 rank 的 KV 逐字节相同）；
#   * 上游 vLLM **已经写好**单副本机制（`SharedOffloadRegion` + `replicated_layout`
#     + `OffloadingConnectorWorker._is_store_writer = not replicated or rank==0`
#     + non-writer 直接 ack、load 侧按 `HIT_PENDING` 等齐 world_size 个 ack），
#     但两道门在 NPU 上是关着的：
#       ① `offloading/config.py` 的 `replicated_layout` 只认证"单组 + 裸 MLAAttentionSpec"；
#       ② `CPUOffloadingSpec._uses_shared_region()` 落到 `current_platform.is_cuda_alike()`，
#          NPU 上是 False。
#   本模块给出 Ascend 版：①在 `kv8-offload-pool/offloading_config.py` 放宽（带运行期校验）；
#   ②在这里覆写 `_uses_shared_region()` 并**自建共享 region**。
#
# 物理前提**已实测**（A3 chip0-7，`results/ced_dram_single_copy_20261010/`）：
#   8 个独立进程对同一 `/dev/shm` 文件 `MAP_SHARED`；各自对本段 VMA
#   `aclrtHostRegister(ptr,size,0)` 全部 `ret=0`；rank0 经 NPU DMA 写入的 256 MiB
#   被其余 7 个 rank 用 NPU DMA 读回**逐字节相同**（n_mismatch=0/rank）。

from __future__ import annotations

import hashlib
import json
import mmap
import os
import threading
import time
from pathlib import Path

import torch

_HOST_MEM_MODE = os.environ.get("NPU_OFFLOAD_HOST_MEM", "registered").strip().lower()
_SC_DIR = os.environ.get("CED_DRAM_SINGLE_COPY_DIR", "/dev/shm").rstrip("/")
_SC_UNLINK = os.environ.get("CED_DRAM_SINGLE_COPY_UNLINK", "0") == "1"


def enabled() -> bool:
    return os.environ.get("CED_DRAM_SINGLE_COPY", "0") == "1"


def verify_enabled() -> bool:
    return os.environ.get("CED_DRAM_SINGLE_COPY_VERIFY", "1") == "1"


def log(msg: str) -> None:
    print("[CED-SINGLE-COPY] " + msg, flush=True)


def _wait_for_file(path: str, expected: int, timeout: float = 600.0) -> None:
    """Wait for the creator's file to exist and be truncated to `expected`.

    The 8 TP workers construct their pools concurrently and rank0 creates the
    file only after its own NPU initialisation (tens of seconds), so a peer can
    easily arrive first — the standalone probe hit exactly this race.
    """
    deadline = time.monotonic() + timeout
    while not os.path.exists(path):
        if time.monotonic() > deadline:
            raise TimeoutError(f"[CED-SINGLE-COPY] {path} never appeared")
        time.sleep(0.05)
    while os.stat(path).st_size < expected:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"[CED-SINGLE-COPY] {path} never reached {expected} bytes"
            )
        time.sleep(0.05)


def _populate(mmap_obj: mmap.mmap, size: int) -> str:
    """Pre-fault the mapping; an unpopulated VMA cannot be host-registered.

    `MADV_POPULATE_WRITE` is accepted here but does **not** reliably fault the
    tmpfs pages on this driver/CANN combo: a region populated that way came
    back as `ret=507899` ("read-only VMA") while the same region populated by
    touching one byte per page registered at `ret=0`. Always touch.
    """
    view = memoryview(mmap_obj).cast("B")
    for off in range(0, size, mmap.PAGESIZE):
        view[off] = 0
    return "touch"


class SingleCopyRegion:
    """TP8 单副本池的一格：**一个 canonical 张量一块共享 mmap**。

    与上游 `SharedOffloadRegion` 的区别（以及为什么必须自己写）：
      * 上游的行宽 = `worker_kv_bytes_per_block`（记账口径 131,072 B），行内要装下
        同一 worker 的**全部** canonical 张量 —— 而真实几何 Σpage = 910,208 B，
        它**装不下**（`a2/logs/032` §3.1 实测撞过 `assert new_offset <= ...`）；
      * 本类**每张张量一块 region**：行宽 = `page_bytes × blocks_per_chunk`（真实几何），
        行数 = 该张量自己的上界 `rows`（= P2/L1 的行配额），物理 = Σrows×page，
        与"一份副本"的账一一对应。

    所有 worker 用 `rank=0` 语义 ⇒ **8 个进程映射同一个 inode**（同一物理页），
    各自 `aclrtHostRegister(MAPPED)` 后用自己的 NPU 直接 DMA。实测见
    `results/ced_dram_single_copy_20261010/shared_region_probe_8rank.json`。
    """

    def __init__(
        self,
        engine_id: str,
        tensor_idx: int,
        rows: int,
        page_bytes: int,
        register: bool = True,
    ) -> None:
        if rows < 0 or page_bytes <= 0:
            raise ValueError(
                f"SingleCopyRegion needs rows>=0 and page>0: {rows} x {page_bytes}"
            )
        if rows == 0:
            # A tensor no participating group references: keep the region object
            # shape-compatible without creating an unmappable zero-length file.
            self.engine_id = engine_id
            self.tensor_idx = tensor_idx
            self.rows = 0
            self.page_bytes = int(page_bytes)
            self.total_size_bytes = 0
            self.mmap_path = ""
            self.fd = None
            self.mmap_obj = None
            self._creator = False
            self.populate_mode = "none"
            self._base = torch.empty(0, dtype=torch.int8)
            self.tensor = self._base.view(0, self.page_bytes)
            self.registered = False
            self.register_ret = None
            return
        self.engine_id = engine_id
        self.tensor_idx = tensor_idx
        self.rows = int(rows)
        self.page_bytes = int(page_bytes)
        self.total_size_bytes = self.rows * self.page_bytes
        self.mmap_path = f"{_SC_DIR}/ced-sc-{engine_id}-t{tensor_idx}.mmap"
        self._creator = False
        self.registered = False
        self.register_ret: int | None = None

        try:
            self.fd: int | None = os.open(
                self.mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
            )
            os.ftruncate(self.fd, self.total_size_bytes)
            self._creator = True
        except FileExistsError:
            _wait_for_file(self.mmap_path, self.total_size_bytes)
            self.fd = os.open(self.mmap_path, os.O_RDWR)

        self.mmap_obj: mmap.mmap | None = mmap.mmap(
            self.fd,
            self.total_size_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        self.populate_mode = _populate(self.mmap_obj, self.total_size_bytes)
        self._base = torch.frombuffer(memoryview(self.mmap_obj), dtype=torch.int8)
        # One tensor per region ⇒ plain contiguous (rows, page_bytes) view.
        self.tensor = self._base.view(self.rows, self.page_bytes)

        if register:
            self.register_ret = register_host_region(self, tensor_idx=self.tensor_idx,
                                                    rank=-1)
            self.registered = self.register_ret == 0
            if not self.registered:
                if os.environ.get("CED_DRAM_STRICT_REGISTER", "0") == "1":
                    raise RuntimeError(
                        "[CED-SINGLE-COPY] aclrtHostRegister failed for tensor "
                        f"{tensor_idx} (ret={self.register_ret}); refusing to run "
                        "the single-copy pool on an unregistered mapping."
                    )
                log(
                    f"tensor{tensor_idx}: host_register ret={self.register_ret} "
                    "-> transfers still work but may be slower"
                )

    def cleanup(self) -> None:
        if self.registered:
            try:
                import acl

                ret = acl.rt.host_unregister(self._base.data_ptr())
                log(
                    f"tensor{self.tensor_idx}: host_unregister ret={ret}"
                )
            except Exception as exc:  # noqa: BLE001
                log(f"tensor{self.tensor_idx}: host_unregister failed: {exc!r}")
            self.registered = False
        self._base = None
        self.tensor = None
        if self.mmap_obj is not None:
            try:
                self.mmap_obj.close()
            except Exception:  # noqa: BLE001
                log(f"tensor{self.tensor_idx}: failed to close mmap")
            self.mmap_obj = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:  # noqa: BLE001
                pass
            self.fd = None
        if self._creator and _SC_UNLINK:
            try:
                os.unlink(self.mmap_path)
            except OSError:
                pass


def register_host_region(region, rank: int, tensor_idx: int) -> int:
    """把共享 mmap 注册给 NPU（`aclrtHostRegister(MAPPED)`），返回 ret。"""
    try:
        import acl

        try:
            acl.init()
        except Exception:  # noqa: BLE001 - 已初始化
            pass
        try:
            acl.rt.set_device(int(torch.npu.current_device()))
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        log(f"rank{rank} tensor{tensor_idx}: import acl failed: {exc!r}")
        return -1

    ptr = region._base.data_ptr()
    size = region.total_size_bytes
    try:
        devptr, ret = acl.rt.host_register(ptr, size, 0)
    except Exception as exc:  # noqa: BLE001
        log(f"rank{rank} tensor{tensor_idx}: host_register raised: {exc!r}")
        return -1
    if rank < 0:
        # rank unknown at construction time; report the physical device instead.
        try:
            rank = int(torch.npu.current_device())
        except Exception:  # noqa: BLE001
            rank = -1
    log(
        "rank%d tensor%d region=%s bytes=%d ret=%s devptr=%#x"
        % (rank, tensor_idx, Path(region.mmap_path).name, size, ret, devptr)
    )
    return int(ret)


# --------------------------------------------------------------------------
# [CED-SINGLE-COPY-VERIFY] store 源 KV 的跨 rank 逐字节校验（fail-closed）
# --------------------------------------------------------------------------
class SourceVerifier:
    """校验"8 个 rank 的 store 源 KV 逐字节相同"这一单副本语义前提。

    单副本的正确性**完全依赖**这条前提：rank0 写进共享区、其余 rank 只 ack；
    后续 load 时每个 rank 从共享区 H2D。若某个 rank 的行内容与 rank0 不同，
    该 rank 就会读到**别人的** KV ⇒ 静默算错。所以在**第一次 store**时，
    每个 rank 对自己的 GPU 源行算 sha256（覆盖每个参与组的首块、中块、末块），
    经 `/dev/shm` 会合后比对；任何不一致**直接抛错**。

    范围（写清楚，不做过度声明）：默认只校验**第一个 store job** 的三点采样；
    `CED_DRAM_SINGLE_COPY_VERIFY_ALL=1` 时校验该 job 的**全部**块。
    """

    def __init__(self, engine_id: str, rank: int, world_size: int):
        self.engine_id = engine_id
        self.rank = rank
        self.world_size = world_size
        self.dir = Path("/dev/shm") / f"ced-sc-verify-{engine_id}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.done = False
        self.payload: dict | None = None
        self._watcher: threading.Thread | None = None
        self._lock = threading.Lock()
        self._checked_jobs: set[int] = set()

    def record(self, job_id: int, digests: dict[str, str]) -> None:
        entry = {
            "rank": self.rank,
            "job_id": job_id,
            "pid": os.getpid(),
            "digests": digests,
            "ts": time.time(),
        }
        path = self.dir / f"job{job_id}-rank{self.rank:02d}.json"
        path.write_text(json.dumps(entry, sort_keys=True))
        log(
            "rank%d job%d: recorded %d digests -> %s"
            % (self.rank, job_id, len(digests), path.name)
        )
        self._ensure_watcher(job_id)

    # ------------------------------------------------------------------
    # ★ 2026-10-10：**非阻塞**校验。
    #
    # 第一版在 store 提交路径里做"8 rank 会合 + 阻塞等待"，结果是**把引擎卡死**：
    # 各 rank 在同一个 step 里互相等（`prepare_store_kv` 处在 TP 的
    # `execute_model → get_finished` 之内），10 分钟后批量抛
    # `verification rendezvous timed out: job=0 got 0/8`。
    # 校验本身是"结构性一次性"的，**不该**挡住推理：改成
    #   ① 每个 rank 把摘要**只写一次**文件（毫秒级，不等待）；
    #   ② rank0 起一个**守护线程**轮询目录，凑齐 8 份后比对；
    #   ③ 不一致 ⇒ 打印细节并 `os._exit(3)`（fail-closed：宁可让服务死，
    #      也不让它在"某个 rank 读别人的 KV"的状态下继续跑）。
    # 超时/缺文件**不**杀进程（只响亮告警），避免把"校验器自己的时序问题"
    # 伪装成"KV 不一致"。
    def _ensure_watcher(self, job_id: int) -> None:
        if self.rank != 0:
            return
        with self._lock:
            if job_id in self._checked_jobs:
                return
            self._checked_jobs.add(job_id)
        thread = threading.Thread(
            target=self._watch_job,
            args=(job_id,),
            name=f"ced-sc-verify-{job_id}",
            daemon=True,
        )
        thread.start()
        self._watcher = thread

    def _watch_job(self, job_id: int) -> None:
        timeout = float(os.environ.get("CED_DRAM_SINGLE_COPY_VERIFY_TIMEOUT", "180"))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            files = sorted(self.dir.glob(f"job{job_id}-rank*.json"))
            if len(files) >= self.world_size:
                break
            time.sleep(0.2)
        else:
            files = sorted(self.dir.glob(f"job{job_id}-rank*.json"))
            log(
                "job%d: watcher timed out with %d/%d digests in %s "
                "(files=%s) -- NOT treated as a mismatch"
                % (job_id, len(files), self.world_size, self.dir,
                   [p.name for p in files])
            )
            return
        try:
            entries = [json.loads(p.read_text()) for p in files]
        except Exception as exc:  # noqa: BLE001
            log(f"job{job_id}: cannot read digests: {exc!r}")
            return
        by_rank = {e["rank"]: e for e in entries}
        if len(by_rank) != self.world_size:
            log(f"job{job_id}: missing ranks {sorted(set(range(self.world_size)) - set(by_rank))}")
            return
        reference = by_rank[0]["digests"]
        mismatched = []
        for r, e in sorted(by_rank.items()):
            for key, value in reference.items():
                got = e["digests"].get(key)
                if got != value:
                    mismatched.append(f"rank{r}:{key} {got} != rank0 {value}")
        report = {
            "job_id": job_id,
            "keys": sorted(reference),
            "ranks": sorted(by_rank),
            "mismatched": mismatched,
            "ok": not mismatched,
        }
        self.payload = report
        if mismatched:
            log(
                "job%d: KV is NOT byte-replicated across ranks -- single-copy "
                "layout would return wrong KV. FAIL-CLOSED exit.\n  %s"
                % (job_id, "\n  ".join(mismatched[:20]))
            )
            os._exit(3)
        log(
            "job%d: OK -- %d digests identical across %d ranks (sample=%s)"
            % (job_id, len(reference), self.world_size, sorted(reference)[:4])
        )

    def wait_and_compare(self, job_id: int, timeout: float) -> dict:
        """Deprecated blocking path -- kept only for offline tests.

        Runtime uses `record()` + the rank0 daemon watcher. Blocking here in the
        step loop deadlocked all eight ranks (2026-10-10), so production code
        must not call this.
        """
        raise RuntimeError(
            "[CED-SINGLE-COPY] wait_and_compare is not allowed on the runtime "
            "path (it deadlocks the TP step loop); use record() instead"
        )


def digest_tensor_rows(tensor: torch.Tensor, rows: list[int]) -> str:
    """对 canonical GPU 张量的指定行做 sha256（D2H 到 CPU 后哈希）。"""
    hasher = hashlib.sha256()
    with torch.no_grad():
        for row in rows:
            chunk = tensor[int(row)].detach().to("cpu").contiguous()
            hasher.update(chunk.numpy().tobytes())
    return hasher.hexdigest()


def digest_store_sources(
    npu_tensors: list[torch.Tensor],
    group_data_refs,
    src_spec,
    all_rows: bool,
    max_pairs: int = 96,
) -> dict[str, str]:
    """给一个 store job 的 **GPU 源 KV** 采点并算 sha256。

    store 方向的 GPU 侧 `blocks_per_chunk` 恒为 1（见 `NPUOffloadingWorker`），
    所以 `src_spec.block_ids` 就是每个组按顺序排列的 GPU 物理块号。
    采样：默认每个 job 取首/中/末三个 (张量, 块) 组合；`all_rows=1` 时取全部。
    `key = "t<tensor_idx>:b<block_id>"` 在 8 个 rank 上**逐字相同**（块号是全局的）。
    """
    block_ids = list(int(b) for b in src_spec.block_ids)
    group_sizes = list(int(s) for s in src_spec.group_sizes)

    # Per-group sampling: pick the first/middle/last block **of the chunk list**
    # and hash **every** canonical tensor that the group references at that
    # block. This covers all 16 tensors instead of three arbitrary ones.
    picked: list[tuple[int, int]] = []
    offset = 0
    for gsize, refs in zip(group_sizes, group_data_refs):
        if gsize <= 0:
            continue
        group_blocks = block_ids[offset:offset + gsize]
        offset += gsize
        if all_rows or len(group_blocks) <= 3:
            chosen_blocks = group_blocks
        else:
            chosen_blocks = [
                group_blocks[0],
                group_blocks[len(group_blocks) // 2],
                group_blocks[-1],
            ]
        for block in chosen_blocks:
            for ref in refs:
                picked.append((int(ref.tensor_idx), int(block)))
    if not picked:
        return {}
    if len(picked) > max_pairs:
        step = max(1, len(picked) // max_pairs)
        picked = picked[::step][:max_pairs]

    digests: dict[str, str] = {}
    with torch.no_grad():
        for tensor_idx, block in picked:
            tensor = npu_tensors[tensor_idx]
            if block < 0 or block >= tensor.shape[0]:
                raise RuntimeError(
                    f"[CED-SINGLE-COPY] store source block {block} out of range for "
                    f"tensor{tensor_idx} (rows={tensor.shape[0]})"
                )
            chunk = tensor[block].detach().to("cpu").contiguous()
            digests[f"t{tensor_idx}:b{block}"] = hashlib.sha256(
                chunk.numpy().tobytes()
            ).hexdigest()
    return digests


def sample_rows(rows: list[int], all_rows: bool) -> list[int]:
    if all_rows or len(rows) <= 3:
        return list(rows)
    return [rows[0], rows[len(rows) // 2], rows[-1]]
