# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram lookup that runs entirely on the device.

The V4.1 Engram tables (2 layers x ~3.84e8 rows, INT8 codes + FP32 group
scales, 221 GB) cannot live in HBM, so they stay in host DRAM.  The legacy
path then performs the whole lookup on the host CPU:

* it needs ``input_ids`` on the CPU, i.e. a blocking ``.cpu()`` sync;
* it needs the table split into 1/8 row shards, because a rank can only read
  its own process memory -- hence a metadata ``all_gather``, an ``all_to_all``
  and a ``broadcast``;
* being out of graph, it forces an extra host round trip per step.

A3's compute cores can address host-mapped memory directly
(``aclrtHostMemMapCapabilities`` reports AIC/AIV supported), so the table can
be mapped into the device address space and indexed by ordinary device
operators.  Every rank then sees the *whole* table (``MAP_SHARED`` keeps the
physical pages single-copy, so the node still stores 221 GB once), which
removes the sharding, the owner routing and the collectives.  The lookup
becomes a plain device node that ACLGraph can capture.

``HostMappedSafetensors`` maps one ``.safetensors`` payload out of host DRAM
and registers it so device operators can index it.  ``DeviceNgramHash`` is the
device port of ``PagedNgramHistory.update`` (page table write, 4-gram history
read, polynomial hash); ``engram_device_test.py`` sweeps the corner cases.
"""

from __future__ import annotations

import ctypes
import json
import os
import struct

import torch

__all__ = [
    "HostMappedSafetensors",
    "wrap_device_ptr",
    "npu_dlpack_device_type",
    "EngramDeviceTables",
    "HostMappedEngramTable",
    "DeviceNgramHash",
    "gather_dequantize_device",
    "real_prime_layout",
    "device_engram_lookup",
    "build_request_ids",
]


def real_prime_layout(layer_ids, lookback: int, n_heads: int, vocab: int):
    """The production bucket layout's primes and offsets.

    ``vocab`` is the **raw** ``engram_vocab_size`` (16,000,000 in V4.1-Flash),
    *not* the compressed vocabulary (99,092).  Primes are drawn as consecutive
    primes above ``vocab - 1`` and never reused, so the bucket ranges are
    disjoint; ``offsets`` is their running sum.  Both matter:

    * the raw vocab is what makes the 24 primes ~16M each, so the running sum
      tiles the table (~384M rows).  Using the compressed vocab instead gives
      primes just above 99k and a layout that only fills 0.6% of the table --
      still self-consistent, but not the shape anything real has to hold.
    * the sum must not exceed the table's row count.  Random primes from a wide
      window overflow it, and the CANN gather then aborts with
      ``Index ... out of range`` (measured: an index of 394233561 against a
      384006168-row table crashed the AIV with a vector-core exception).  A
      fixture that ignores this does not exercise the implementation, it just
      takes the device down.
    """
    import sympy

    primes, seen = [], set()
    for _ in layer_ids:
        per_ngram = []
        for _ in range(lookback - 1):
            sizes, current = [], vocab - 1
            for _ in range(n_heads):
                candidate = current + 1
                while not sympy.isprime(candidate) or candidate in seen:
                    candidate += 1
                seen.add(candidate)
                current = candidate
                sizes.append(candidate)
            per_ngram.append(tuple(sizes))
        primes.append(tuple(per_ngram))
    primes_t = torch.tensor(primes, dtype=torch.int64)
    flat = primes_t.flatten(1)
    return primes_t, flat.cumsum(-1) - flat

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
ACL_HOST_REGISTER_MAPPED = 0
PAGE = 4096

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
_libc.mincore.restype = ctypes.c_int

PROT_READ, PROT_WRITE = 1, 2
MAP_SHARED = 1
MAP_PRIVATE = 0x02
MAP_ANONYMOUS = 0x20

_DTYPE_MAP = {
    "I8": torch.int8,
    "U8": torch.uint8,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
}
_NUMPY_DTYPE = {
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}
# DLDataType code: 0 = signed int, 1 = unsigned int, 2 = float
_DL_CODE = {
    torch.int8: 0,
    torch.uint8: 0,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 2,
}


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    # `dl_tensor` must come first: the capsule handed to `from_dlpack` holds a
    # `DLManagedTensor*`, so the device type can be read without torch's own
    # `Tensor.__dlpack_device__()`, which raises "Unknown device type npu".
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.restype = ctypes.c_void_p
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

_KEEPALIVE: list = []
_ACL_READY = False


def npu_dlpack_device_type() -> int:
    """DLDeviceType for NPU, probed at runtime rather than hard-coded."""
    t = torch.zeros(2, device="npu")
    cap = torch.utils.dlpack.to_dlpack(t)
    ptr = _PyCapsule_GetPointer(cap, b"dltensor")
    mt = ctypes.cast(ptr, ctypes.POINTER(_DLManagedTensor)).contents
    return int(mt.dl_tensor.device.device_type)


def _ensure_acl() -> None:
    """Initialise ACL and make a device current.

    Inside a vLLM worker torch_npu has already done both; in a standalone
    script it has not, and ``aclrtHostRegister`` then fails with ret=107002
    ("no context"), which reads like a permission problem.
    """
    global _ACL_READY
    if _ACL_READY:
        return
    import acl

    acl.init()
    try:
        dev = int(torch.npu.current_device())
    except Exception:  # noqa: BLE001 - torch_npu may not be importable yet
        dev = 0
    acl.rt.set_device(dev)
    _ACL_READY = True


def probe_host_mapping_capability() -> tuple[bool, str]:
    """Can this machine register host memory so device operators can read it?

    The whole device-index design rests on one hardware/driver property:
    ``aclrtHostRegister(..., MAPPED)`` on an ordinary writable host mapping,
    after which a device operator (``index_select``) reads it directly.  That was
    verified on **A3 (910C)**; it has **never** been verified on **A2 (910B3)**,
    and there is a documented counter-example on this very project: on A3,
    ``offload.get_dva(pinned_ptr)`` returns 0 and an AIV de-referencing a
    registered *pinned* address dies with ``507035 MTE invalid GM address``
    (``docs/A2_VS_A3_DIFF.md`` §5: "任何『host 地址可以被 device kernel 直接读』
    的假设在 A2 上都是未验证").

    So the release must not assume the property: it probes at start-up and only
    turns the device-index path on when the probe passes.  4 KiB is enough -- the
    question is whether registration is *accepted*, not how fast it is.

    Returns ``(ok, detail)``; never raises, so callers can log and fall back.
    """
    size = PAGE
    addr = 0
    registered = False
    try:
        import acl

        _ensure_acl()
        addr = _libc.mmap(None, size, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
        if addr in (None, ctypes.c_void_p(-1).value, 2**64 - 1):
            return False, f"mmap failed: {os.strerror(ctypes.get_errno())}"
        dev, ret = acl.rt.host_register(addr, size, ACL_HOST_REGISTER_MAPPED)
        if ret != 0 or not dev:
            return False, f"aclrtHostRegister returned ret={ret} (dev={dev})"
        registered = True
        # ★ 这里**故意不做**"设备读一次验证"。早先版本用
        #     t = wrap_device_ptr(dev, (size,), torch.uint8); int(t[0])
        # 想证明显存可读，结果在 **worker 初始化期间 segfault**：
        #     Segfault encountered
        #       aclrtMemcpyImpl -> at::_ops::_local_scalar_dense -> at::native::item
        # 即对 wrap_device_ptr 造出来的"指向 host 映射的 device 指针张量"做 D2H 会崩，
        # 而 worker init 恰好是不该做设备同步的时机（实测整台机器起不来）。
        # 所以本探测只回答"注册被不被接受" —— 那是它在 auto 模式下的职责；
        # **端到端可读性**（真正的判据）交给独立进程的 tools/probe_a2_hostmap.py，
        # 它在干净的进程里做 index_select + 逐字节对账，实测在 A3 上通过。
        return True, ("host mapping registered (仅证明注册被接受；端到端可读性"
                      "请用 tools/probe_a2_hostmap.py 在独立进程里验证)")
    except Exception as exc:  # noqa: BLE001 - probe must never break start-up
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if registered:
                import acl

                acl.rt.host_unregister(addr)
        except Exception:  # noqa: BLE001
            pass
        if addr:
            try:
                _libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(size))
            except Exception:  # noqa: BLE001
                pass


def wrap_device_ptr(
    dev_ptr: int, shape: tuple[int, ...], dtype: torch.dtype, device_id: int | None = None
) -> torch.Tensor:
    """Expose a device address as a torch tensor without copying.

    The capsule and its shape array must outlive the tensor, so they are parked
    on the tensor itself (and in a module-level list).  The capsule must point
    at a **DLManagedTensor**: torch reads ``manager_ctx``/``deleter`` past the
    tensor fields, so a bare ``DLTensor`` segfaults.  The deleter is a no-op
    because the mapping's lifetime belongs to ``HostMappedSafetensors``.

    ``device_id`` must be **this process's logical device**, not 0.  In a
    single-device test process the two coincide, which hides the bug; under TP8
    each worker is on its own device, and a tensor labelled ``npu:0`` while the
    index tensor lives on ``npu:3`` fails with
    "Expected all tensors to be on the same device, but got index is on npu:3,
    different from other tensors on npu:0".  Measured on A3-node1 TP8.
    """
    if device_id is None:
        device_id = int(torch.npu.current_device())
    ndim = len(shape)
    shape_arr = (ctypes.c_int64 * ndim)(*shape)
    mt = _DLManagedTensor()
    mt.dl_tensor.data = ctypes.c_void_p(dev_ptr)
    mt.dl_tensor.device = _DLDevice(npu_dlpack_device_type(), int(device_id))
    mt.dl_tensor.ndim = ndim
    mt.dl_tensor.dtype = _DLDataType(_DL_CODE[dtype], dtype.itemsize * 8, 1)
    mt.dl_tensor.shape = ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_int64))
    mt.dl_tensor.strides = None
    mt.dl_tensor.byte_offset = 0
    mt.manager_ctx = None
    noop = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(lambda _p: None)
    mt.deleter = ctypes.cast(noop, ctypes.c_void_p)
    _KEEPALIVE.extend([mt, shape_arr, noop])

    pycapsule_new = ctypes.pythonapi.PyCapsule_New
    pycapsule_new.restype = ctypes.py_object
    pycapsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    cap = pycapsule_new(ctypes.cast(ctypes.pointer(mt), ctypes.c_void_p), b"dltensor", None)
    _KEEPALIVE.append(cap)
    t = torch.utils.dlpack.from_dlpack(cap)
    t._hostmap_keepalive = _KEEPALIVE[-3:]  # keep alive together with the tensor
    return t


class HostMappedSafetensors:
    """One tensor out of a ``.safetensors`` file, mapped from host DRAM and
    registered so device operators can index it directly.

    Usage::

        with HostMappedSafetensors("/path/layers_1_engram_embed.weight.safetensors") as t:
            rows = torch.index_select(t.tensor, 0, ids_on_device)

    Constraints established on A3 (see ``probe_register_matrix.py``):

    * the mapping must be **writable** -- a read-only VMA is rejected with
      ret=507899, so the table directory must be mounted read-write even though
      nothing ever writes to it;
    * the address must be **page-aligned** (mmap guarantees that) and the
      registered range must not extend past EOF (ret=107017 otherwise).  A
      safetensors payload starts at ``8 + header_len``, so the mapping starts
      one page early and the device pointer is offset by the same delta.
    """

    def __init__(self, path: str, device_id: int | None = None, row_start: int = 0,
                 row_count: int | None = None) -> None:
        self.path = path
        self.device_id = device_id
        self._addr = 0
        self._maplen = 0
        self._registered = False
        self.tensor: torch.Tensor | None = None

        self.offset, self.name, meta = self._parse_header()
        self.dtype = _DTYPE_MAP[meta["dtype"]]
        self.total_shape = tuple(int(x) for x in meta["shape"])
        self.shape = self.total_shape
        # `data_offsets` is already a byte range -- do not also multiply by the
        # shape, that overflows into a nonsense size.
        self.nbytes = int(meta["data_offsets"][1]) - int(meta["data_offsets"][0])
        expect = self.dtype.itemsize
        for d in self.shape:
            expect *= d
        if expect != self.nbytes:
            raise ValueError(
                f"{self.path}: header says {self.nbytes} bytes but shape "
                f"{self.shape} of {self.dtype} needs {expect}"
            )
        # Optionally map only a row range.  This is what makes a *segmented*
        # mapping useful: several small registrations keep every device access
        # inside one segment's address span, whereas one registration over the
        # whole file exposes the full span no matter how the tensor is sliced.
        if row_start or row_count is not None:
            row_bytes = self.dtype.itemsize
            for d in self.total_shape[1:]:
                row_bytes *= d
            total_rows = self.total_shape[0]
            count = total_rows - row_start if row_count is None else int(row_count)
            if row_start < 0 or count <= 0 or row_start + count > total_rows:
                raise ValueError(
                    f"{self.path}: row range [{row_start}, {row_start + count}) "
                    f"does not fit in {total_rows} rows"
                )
            self.offset += row_start * row_bytes
            self.shape = (count, *self.total_shape[1:])
            self.nbytes = count * row_bytes
        self._map_and_register()

    # ------------------------------------------------------------------ setup
    def _parse_header(self):
        with open(self.path, "rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hlen))
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            return 8 + hlen + int(meta["data_offsets"][0]), name, meta
        raise RuntimeError(f"{self.path}: empty safetensors header")

    def _map_and_register(self) -> None:
        import acl

        _ensure_acl()

        fd = os.open(self.path, os.O_RDWR)
        try:
            aligned = self.offset & ~(PAGE - 1)
            delta = self.offset - aligned
            filesize = os.path.getsize(self.path)
            maplen = min(self.nbytes + delta, filesize - aligned)
            addr = _libc.mmap(None, maplen, PROT_READ | PROT_WRITE, MAP_SHARED, fd, aligned)
            if addr in (None, ctypes.c_void_p(-1).value, 2**64 - 1):
                err = ctypes.get_errno()
                raise OSError(
                    f"mmap({self.path}) failed: {os.strerror(err)}. If this is "
                    "EACCES/EROFS the table directory is mounted read-only and "
                    "aclrtHostRegister cannot take a reference on it."
                )
            self._addr = addr
            self._maplen = maplen
            dev, ret = acl.rt.host_register(addr, maplen, ACL_HOST_REGISTER_MAPPED)
            if ret != 0 or not dev:
                _libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(maplen))
                self._addr = 0
                raise RuntimeError(
                    f"aclrtHostRegister({self.path}) failed: ret={ret}. Known causes: "
                    "(a) the mapping is read-only (ret=507899); (b) no device context "
                    "(ret=107002). The file must be opened O_RDWR."
                )
            self._dev_ptr = dev + delta
            self._registered = True
        finally:
            os.close(fd)  # the mapping holds its own reference

        self.tensor = wrap_device_ptr(
            self._dev_ptr, self.shape, self.dtype, device_id=self.device_id
        )

    # ------------------------------------------------------------------ teardown
    def close(self) -> None:
        """Release in the order CANN requires: tensor, unregister, unmap."""
        self.tensor = None
        if self._registered:
            try:
                import acl

                acl.rt.host_unregister(self._addr)
            except Exception:  # noqa: BLE001
                pass
            self._registered = False
        if self._addr:
            _libc.munmap(ctypes.c_void_p(self._addr), ctypes.c_size_t(self._maplen))
            self._addr = 0

    def host_array(self):
        """numpy view of the host mapping -- ground truth for correctness
        checks against a device gather (no copy, same physical pages)."""
        import numpy as np

        if not self._addr:
            raise RuntimeError("mapping is closed")
        delta = self.offset - (self.offset & ~(PAGE - 1))
        buf = (ctypes.c_char * self.nbytes).from_address(self._addr + delta)
        return np.frombuffer(buf, dtype=np.uint8).view(_NUMPY_DTYPE[self.dtype]).reshape(self.shape)

    def resident_bytes(self) -> int:
        """How much of the mapping is backed by resident physical pages.

        Worth checking before believing any bandwidth number: a random gather
        over a file that is *not* in page cache measures the disk, not the
        interconnect.
        """
        if not self._addr:
            raise RuntimeError("mapping is closed")
        pages = (self._maplen + PAGE - 1) // PAGE
        vec = ctypes.create_string_buffer(pages)
        ret = _libc.mincore(
            ctypes.c_void_p(self._addr), ctypes.c_size_t(self._maplen), vec
        )
        if ret != 0:
            err = ctypes.get_errno()
            raise OSError(f"mincore failed: {os.strerror(err)}")
        raw = vec.raw[:pages]
        return sum(1 for b in raw if b & 1) * PAGE

    def __enter__(self) -> "HostMappedSafetensors":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


class EngramDeviceTables:
    """Both Engram layers, host-mapped and device-addressable.

    The files are produced offline by the INT8 quantiser and are shared by all
    ranks of the node: ``MAP_SHARED`` means the pages are stored once, so eight
    ranks still cost 221 GB of DRAM rather than 8 x 221 GB.
    """

    WEIGHT = "layers_{layer}_engram_embed.weight.safetensors"
    SCALE = "layers_{layer}_engram_embed.scale.safetensors"

    def __init__(self, root: str, layer_ids, device_id: int | None = None) -> None:
        self.root = root
        self.layer_ids = tuple(layer_ids)
        self._weights: dict[int, HostMappedSafetensors] = {}
        self._scales: dict[int, HostMappedSafetensors] = {}
        for layer in self.layer_ids:
            self._weights[layer] = HostMappedSafetensors(
                os.path.join(root, self.WEIGHT.format(layer=layer)), device_id=device_id
            )
            self._scales[layer] = HostMappedSafetensors(
                os.path.join(root, self.SCALE.format(layer=layer)), device_id=device_id
            )

    def weight(self, layer: int) -> torch.Tensor:
        return self._weights[layer].tensor

    def scale(self, layer: int) -> torch.Tensor:
        return self._scales[layer].tensor

    def rows(self, layer: int) -> int:
        return int(self._weights[layer].shape[0])

    def close(self) -> None:
        for t in list(self._weights.values()) + list(self._scales.values()):
            t.close()
        self._weights.clear()
        self._scales.clear()

    def __enter__(self) -> "EngramDeviceTables":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class SegmentedHostTable:
    """The whole table, mapped and registered as N independent segments.

    A device gather over a 92 GiB host mapping costs 9.4x more than the same
    gather over an 11.5 GiB one -- same rows, same bytes read, same file:

        393216 rows, whole 92 GiB table      16.8 ms    6.0 GB/s
        393216 rows, first 11.5 GiB eighth    1.8 ms   56.6 GB/s

    Host huge pages do not explain it (a 2 MiB-backed anonymous mapping of the
    same size behaves identically), so the cost is in the device-side address
    translation for a large span.  Registering each eighth separately keeps
    every gather inside one segment's span and recovers the difference; the
    rows then have to be put back in their original order, which is why the
    restore path matters as much as the gather.

    Measured end to end for 393216 rows on A3 (see ``engram_segment_probe.py``
    and ``engram_restore_probe.py``):

        whole-table gather                    16.75 ms
        segmented gather                       2.06 ms
        segmented + restore                   5.36 ms
        full call incl. regrouping             6.87 ms   (2.44x)

    and the result is bit-identical to the whole-table gather.
    """

    def __init__(self, path: str, rows: int, width: int, segments: int = 8,
                 device_id: int | None = None) -> None:
        self.path = path
        self.rows = int(rows)
        self.width = int(width)
        self.n_segments = int(segments)
        self._maps: list[tuple[int, int, HostMappedSafetensors]] = []

        # One mapping per segment.  They are separate `mmap` calls on the same
        # file, so the page cache still stores the bytes once.
        seg_rows = (self.rows + self.n_segments - 1) // self.n_segments
        self.seg_rows = seg_rows
        for s in range(self.n_segments):
            start = s * seg_rows
            if start >= self.rows:
                break
            rows_here = min(seg_rows, self.rows - start)
            self._maps.append(
                (
                    start,
                    rows_here,
                    HostMappedSafetensors(
                        path, device_id=device_id, row_start=start, row_count=rows_here
                    ),
                )
            )

    def _segment_tensor(self, entry):
        """The segment's device tensor (its mapping covers only this range)."""
        return entry[2].tensor

    @torch.no_grad()
    def gather(self, ids: torch.Tensor) -> torch.Tensor:
        """``weight[ids]`` for the whole table, in one flat device tensor."""
        n = int(ids.shape[0])
        out = torch.empty((n, self.width), dtype=self._maps[0][2].dtype,
                          device=ids.device)
        if ids.device.type != "npu" or self.n_segments == 1:
            return torch.index_select(self._maps[0][2].tensor, 0, ids)
        seg_of = ids // self.seg_rows
        parts, orders = [], []
        for s, (start, rows, _mapped) in enumerate(self._maps):
            idx = torch.nonzero(seg_of == s, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            parts.append(torch.index_select(self._segment_tensor(self._maps[s]),
                                            0, ids[idx] - start))
            orders.append(idx)
        gathered = torch.cat(parts, 0)
        order = torch.cat(orders, 0)
        _scatter_rows(out, order, gathered)
        return out

    def close(self) -> None:
        for _start, _rows, mapped in self._maps:
            mapped.close()
        self._maps.clear()

    def __enter__(self) -> "SegmentedHostTable":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _scatter_rows(out: torch.Tensor, order: torch.Tensor, values: torch.Tensor) -> None:
    """Write ``values`` back into ``out`` at ``order``, on device.

    ``Tensor.index_copy_`` costs 43 ms for 393216 x 256 int8 -- 29x more than
    ``npu_scatter_nd_update_`` (1.49 ms) and far more than a plain 100 MB copy
    (0.07 ms), so the CANN-native op is the only usable option here.  It is
    documented for Atlas A3 and supports graph mode, which is what lets the
    whole segmented lookup stay capturable.
    """
    try:
        import torch_npu

        torch_npu.npu_scatter_nd_update_(out, order.unsqueeze(-1).to(torch.int64), values)
        return
    except Exception:  # noqa: BLE001 - fall back to the portable form
        pass
    out.index_put_((order,), values)


def build_request_ids(boundaries: torch.Tensor, device) -> torch.Tensor:
    """Which request each row of the batch belongs to.

    ``boundaries`` is the scheduler's ``query_start_loc`` (``num_reqs + 1``
    entries) and may live on **either** CPU or device.  The device path passes
    ``meta.query_start_loc`` as-is to avoid a per-step H2D, and an earlier
    version built the index on the CPU:

        torch.repeat_interleave(torch.arange(n), boundaries.diff())

    which raises ``Expected all tensors to be on the same device, but got
    repeats is on npu:N, different from other tensors on cpu``.  That mattered
    far more than a one-line bug should: the exception escaped *after* the
    engine had submitted its device-metadata task and before it released it,
    permanently latching that flag, so this surfaced as
    "The previous device metadata submission has not been released" on every
    later request.  Build both operands on the same device.
    """
    dev = torch.device(device)
    counts = boundaries.diff().to(dev)
    index = torch.arange(len(boundaries) - 1, dtype=torch.int64, device=dev)
    return torch.repeat_interleave(index, counts)


@torch.no_grad()
def device_engram_lookup(
    hash_impl: "DeviceNgramHash",
    tables: dict,
    layer_ids,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    boundaries: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[dict, torch.Tensor]:
    """The whole device-side Engram step: n-gram hash, then one lookup per layer.

    Kept as a free function (rather than inlined in the model) so the tensor
    plumbing -- request mapping, block-table indexing, per-layer slicing -- can
    be tested against a fake attention metadata without booting the engine.
    """
    device = positions.device
    requests = build_request_ids(boundaries, device)
    hashes, mask = hash_impl(input_ids, positions, requests, block_table)
    lookups = {
        layer_id: tables[layer_id].lookup(hashes[:, slot])
        for slot, layer_id in enumerate(layer_ids)
    }
    return lookups, mask


def gather_dequantize_device(
    weight: torch.Tensor, scales: torch.Tensor, ids: torch.Tensor
) -> torch.Tensor:
    """Gather INT8 rows and apply the group-32 scales, entirely on the device.

    ``torch.index_select`` goes through aclnn, which -- unlike the Triton
    launcher -- does not re-validate the pointer's memory location, so it works
    on a host-mapped table.  The fused Triton kernel is used when the runtime
    accepts the table pointer (see ``probe_triton_hostmap``).
    """
    codes = torch.index_select(weight, 0, ids)
    group = torch.index_select(scales, 0, ids)
    decoded = codes.to(torch.float32).unflatten(-1, (-1, 32))
    decoded.mul_(group.unsqueeze(-1))
    return decoded.flatten(-2).to(torch.bfloat16)


class HostMappedEngramTable:
    """One Engram layer's INT8 codes and FP32 group scales, addressed by device.

    Drop-in replacement for ``NodeShardedEngram``'s lookup: it exposes the same
    ``[T, n_hash_cols * width]`` BF16 result, but the table is never copied into
    HBM and no rank-local shard is needed, so the metadata all-gather,
    all_to_all and broadcast all disappear.

    Scale gather is deliberately *not* optimised: measured on the real table it
    costs 0.77 ms for 393216 rows, while the codes cost 16.8 ms.  The lookup is
    dominated by the codes.
    """

    WEIGHT = "layers_{layer}_engram_embed.weight.safetensors"
    SCALE = "layers_{layer}_engram_embed.scale.safetensors"

    def __init__(self, root: str, layer_id: int, device_id: int | None = None) -> None:
        self.root = root
        self.layer_id = int(layer_id)
        self.weight_map = HostMappedSafetensors(
            os.path.join(root, self.WEIGHT.format(layer=self.layer_id)), device_id=device_id
        )
        self.scale_map = HostMappedSafetensors(
            os.path.join(root, self.SCALE.format(layer=self.layer_id)), device_id=device_id
        )
        self.weight = self.weight_map.tensor
        self.scale = self.scale_map.tensor
        self.rows = int(self.weight.shape[0])
        self.width = int(self.weight.shape[1])
        if int(self.scale.shape[0]) != self.rows or int(self.scale.shape[1]) != self.width // 32:
            raise ValueError(
                f"Engram layer {self.layer_id}: scale {tuple(self.scale.shape)} does not "
                f"match weight {tuple(self.weight.shape)}"
            )

    @torch.no_grad()
    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        """``ids`` is ``[T, n_hash_cols]`` int64; result is ``[T, n_hash_cols * width]``."""
        shape = ids.shape
        flat = ids.reshape(-1)
        codes = torch.index_select(self.weight, 0, flat)
        groups = torch.index_select(self.scale, 0, flat)
        decoded = codes.to(torch.float32).unflatten(-1, (-1, 32))
        decoded.mul_(groups.unsqueeze(-1))
        # Same arithmetic order as the stock `dequantize_engram_rows`, which is
        # what the fused Triton kernel also implements.
        return decoded.flatten(-2).to(torch.bfloat16).view(*shape, self.width).flatten(1)

    def close(self) -> None:
        self.weight = None
        self.scale = None
        self.weight_map.close()
        self.scale_map.close()


def probe_triton_hostmap(weight: torch.Tensor, scales: torch.Tensor) -> tuple[bool, str]:
    """Can the fused Triton kernel read this table?

    The Triton launcher rejects any pointer whose ``aclrtPointerGetAttributes``
    location is neither DEVICE nor HOST_NUMA, so a file-backed host mapping is
    refused with "Pointer argument (at 0) cannot be accessed from Triton (cpu
    tensor?)".  Probing costs one tiny kernel at start-up and saves the caller
    from paying a failed dispatch every step.
    """
    try:
        from vllm_ascend.ops.triton.engram_int8 import gather_dequantize_engram_int8
    except Exception as exc:  # noqa: BLE001 - optional accelerator path
        return False, f"import failed: {exc!r}"
    ids = torch.zeros(1, dtype=torch.int64, device=weight.device)
    try:
        gather_dequantize_engram_int8(weight, scales, ids)
    except Exception as exc:  # noqa: BLE001 - we fall back to aclnn
        return False, f"{type(exc).__name__}: {exc}"
    return True, "accepted"


class DeviceNgramHash:
    """Device port of ``PagedNgramHistory.update``.

    The host version keeps ``pages`` as a Python dict and the history scan as a
    Python loop; both become dense device tensors here so the whole node can be
    captured.  Everything with a semantic effect is preserved:

    * the page table is written for *all* rows before any history is read, so a
      token sees the freshly written prefix of its own chunk;
    * image tokens are stored as -1 and therefore break the history scan;
    * the scan stops at the first non-present slot (``< 0``) instead of
      skipping it;
    * slots that were never written read as -1, which is what the dict version
      returns for every page it is willing to touch;
    * the hash is int64 multiply/xor/modulo in the same order, and the bucket
      offsets are added last.

    ``row_valid`` masks padding rows: their page writes are redirected to a
    dedicated trash page (they must not touch a real request's block) and they
    are excluded from the returned mask.
    """

    TRASH_OFFSET = 0

    def __init__(
        self,
        token_map: torch.Tensor,
        primes: torch.Tensor,
        offsets: torch.Tensor,
        multipliers: torch.Tensor,
        pad_id: int,
        image_token_id: int,
        image_pad_token_id: int,
        device,
        max_pages: int = 0,
        block_size: int = 128,
    ) -> None:
        self.device = torch.device(device)
        self.lookback = int(multipliers.shape[-1])
        self.n_layers = int(multipliers.shape[0])
        self.n_heads = int(primes.shape[-1])
        self.block_size = int(block_size)
        self.trash_page = int(max_pages)
        self.pad_id = int(pad_id)
        self.image_token_id = int(image_token_id)
        self.image_pad_token_id = int(image_pad_token_id)

        self.token_map = token_map.to(self.device, torch.int64)
        self.primes = primes.to(self.device, torch.int64)
        self.offsets = offsets.to(self.device, torch.int64)
        self.multipliers = multipliers.to(self.device, torch.int64).unsqueeze(0)
        # One extra row is the trash page used by padding rows.
        self.pages = torch.full(
            (self.trash_page + 1, self.block_size), -1, dtype=torch.int64, device=self.device
        )
        # Flat view for the page write: on Ascend `flat.scatter_(0, idx, val)`
        # costs ~0.10 ms where `pages[page, off] = val` costs ~0.34 ms for the
        # same 32 rows, and both walk the whole destination.
        self.pages_flat = self.pages.view(-1)
        self._invalid_reported = 0
        self._clean_steps = 0

    @property
    def capacity(self) -> int:
        return self.pages.shape[0] - 1

    def _note_invalid_pages(self, page_ok: torch.Tensor) -> None:
        """Count out-of-range page ids and report the first few occurrences.

        Silently redirecting them is *correct* (it mirrors the host dict) but it
        must not be invisible: if this fires constantly, the page table is
        under-sized and the lookup is quietly degrading to garbage values.
        The sync is only paid on the first few steps.
        """
        if self._invalid_reported >= 3:
            return
        # Never do a device->host sync while a graph is being captured: that is
        # the failure mode ACLGraph reports as "stream is captured".  The check
        # is diagnostics only, so skipping it during capture costs nothing.
        try:
            if torch.npu.is_current_stream_capturing():
                return
        except Exception:  # noqa: BLE001 - older torch_npu may lack the helper
            pass
        bad = int((~page_ok).sum())
        if bad == 0:
            self._clean_steps += 1
            if self._clean_steps >= 3:
                self._invalid_reported = 3     # saw enough clean steps; stop checking
            return
        self._invalid_reported += 1
        print(
            f"[DEVICE-INDEX] 警告：本步有 {bad}/{page_ok.numel()} 个槽位的 page id "
            f"越界（合法区间 [0, {self.capacity})），已重定向到垃圾桶页 "
            f"{self.trash_page}。这与 host 路径的 `pages[-1]` 语义等价，结果仍逐位一致；"
            "但若持续出现，说明页表容量偏小，应调大 num_gpu_blocks 的估算。",
            flush=True,
        )

    def check_pages(self, block_table: torch.Tensor) -> int:
        """Fail loudly if the block table references a page id we did not size for.

        The host version kept ``pages`` in a Python dict, so an unexpected page
        simply created a new entry.  A dense table turns that into an
        out-of-bounds access, and if the id happened to land inside the
        allocation it would corrupt *another request's* n-gram history with no
        error at all.  This costs one sync, so the caller runs it a few times at
        start-up (and after every reallocation) rather than on every step.

        Returns the largest page id seen.
        """
        worst = int(block_table.max())
        lowest = int(block_table.min())
        if worst > self.capacity or lowest < 0:
            raise RuntimeError(
                f"Engram device page table holds {self.capacity} pages but the "
                f"block table spans [{lowest}, {worst}]. Raise the sizing (see "
                f"cache_config.num_gpu_blocks) or the lookup will read/write out "
                f"of bounds. NOTE: checking only `.max()` was the bug that let a "
                f"negative page id reach `scatter_` and abort the engine with "
                f"'AI CPU kernel execution failed ... ScatterElements'."
            )
        return worst

    def ensure_capacity(self, max_pages: int) -> bool:
        """Grow the dense page table so ``max_pages`` distinct page ids fit.

        The host version keeps ``pages`` in a Python dict, so it never had to
        bound the page id space.  A dense device table must, and the bound comes
        from the scheduler's block pool (``cache_config.num_gpu_blocks``), which
        is only known after the memory profile -- hence the lazy growth.

        Returns True when the table was reallocated (its contents are then
        lost, which is correct: a bigger pool means the old ids are gone).
        """
        if max_pages <= self.capacity:
            return False
        self.trash_page = int(max_pages)
        self.pages = torch.full(
            (self.trash_page + 1, self.block_size), -1,
            dtype=torch.int64, device=self.device,
        )
        self.pages_flat = self.pages.view(-1)
        return True

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        request_ids: torch.Tensor,
        block_table: torch.Tensor,
        row_valid: torch.Tensor | None = None,
    ):
        n = int(input_ids.numel())
        max_blocks = int(block_table.shape[1])

        flat_ids = input_ids.reshape(-1)
        pos = positions.reshape(-1).to(torch.int64)
        req = request_ids.reshape(-1).to(torch.int64)

        mask = (flat_ids != self.image_token_id) & (flat_ids != self.image_pad_token_id)
        if row_valid is not None:
            mask = mask & row_valid.reshape(-1)
        compressed = torch.index_select(self.token_map, 0, flat_ids)
        compressed = compressed.masked_fill(~mask, -1)

        # ---- page table write (all rows first, then the history scan) ----
        #
        # * PAGE-ID VALIDATION IS MANDATORY.  `meta.block_table` may contain
        # garbage for slots the request has not reached: the *host* implementation
        # keeps `pages` in a Python dict, so a -1 (or any unexpected id) simply
        # creates a new dict entry and stays self-consistent.  A dense device
        # table turns the same value into a negative/OOB index, and `scatter_` on
        # Ascend falls back to an **AI CPU** kernel which bounds-checks and aborts
        # the whole engine:
        #
        #   AI CPU kernel execution failed, kernelName=ScatterElements,
        #   soName=libcpu_kernels.so, errorCode=0x91
        #   -> ERR00100 -> HCCL watchdog on other ranks -> engine dead
        #
        # Measured in the v8 release sweep: 7/64 requests returned 500 and all
        # eight ranks died.  The previous `check_pages()` guard missed it because
        # it looked only at `.max()` and never at `.min()`.
        #
        # Redirect every out-of-range id to one shared trash page, on BOTH the
        # write and the read side so the two agree.  That is exactly the host's
        # "everything invalid shares pages[-1]" behaviour, so results stay
        # bit-identical to the reference.
        slots = (pos // self.block_size).clamp_(0, max_blocks - 1)
        page = block_table[req, slots]
        off = pos % self.block_size
        page_ok = (page >= 0) & (page < self.capacity)
        trash = page.new_full((), self.trash_page)
        w_page = torch.where(page_ok, page, trash)
        w_off = off
        if row_valid is not None:
            valid = row_valid.reshape(-1)
            w_page = torch.where(valid, w_page, trash)
            w_off = torch.where(valid, off, off.new_full((), self.TRASH_OFFSET))
        self._note_invalid_pages(page_ok)
        self.pages_flat.scatter_(0, w_page * self.block_size + w_off, compressed)

        # ---- n-gram history ----
        # One gather for all `lookback` shifts instead of one per shift: the
        # per-shift loop costs ~11 ops each (~44 total) and the whole step runs
        # eagerly before the model's capture, where every op is pure dispatch.
        # The barrier semantics are preserved by a running minimum rather than
        # a serial early-exit, which is what the host implementation's "fast"
        # path does -- and `engram_device_test.py --stage cpu` checks that the
        # two formulations agree bit for bit on the stock/fast reference pair.
        shift = torch.arange(self.lookback, dtype=torch.int64, device=self.device)
        previous = pos.unsqueeze(1) - shift  # (n, lookback)
        in_range = previous >= 0
        prev_safe = previous.clamp_min(0)
        page2d = block_table[
            req.unsqueeze(1).expand_as(prev_safe),
            torch.minimum(prev_safe // self.block_size,
                          prev_safe.new_full((), max_blocks - 1)),
        ]
        off2d = prev_safe % self.block_size
        # Same validation as the write side, and the two MUST agree, or a valid
        # row could read a slot it never wrote.
        #
        # Note the offset: an out-of-range page id is redirected to the shared
        # trash page but keeps its **natural offset**, because the host's
        # `pages[-1]` really is one `block_size`-wide bucket -- forcing offset 0
        # would make every invalid read alias onto the same slot and the result
        # would stop matching the reference (measured: 480/1920 elements differ).
        # The offset only collapses to 0 for positions before the sequence start,
        # whose value is masked away immediately afterwards.
        page2d_ok = (page2d >= 0) & (page2d < self.capacity)
        read_ok = in_range & page2d_ok
        read_page = torch.where(read_ok, page2d, page2d.new_full((), self.trash_page))
        read_off = torch.where(in_range, off2d, off2d.new_full((), self.TRASH_OFFSET))
        values = self.pages[read_page, read_off]
        # A slot is live when its position is in range and the slot is not a
        # barrier (-1).  Everything after the first barrier is forced back to
        # pad_id, matching the reference walk's early exit.
        values = values.masked_fill(~in_range | (values < 0), -1)
        history = values.masked_fill(
            values.cummin(dim=1).values < 0, self.pad_id
        )

        # ---- polynomial hash ----
        products = history[:, None, :] * self.multipliers
        rolling = products[..., 0]
        hashes = []
        for shift in range(1, self.lookback):
            rolling = torch.bitwise_xor(rolling, products[..., shift])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, shift - 1])
        hashes = torch.cat(hashes, -1) + self.offsets
        return hashes, mask

    def configure(self, block_size: int, max_pages: int) -> bool:
        """Size the dense page table.  Call before the first lookup.

        ``block_size`` comes from the attention metadata (``storage_block_size``)
        and ``max_pages`` from the scheduler's block pool, so both are learnt at
        runtime rather than at construction.  Returns True when the table was
        (re)allocated, which discards the paging state.
        """
        if int(block_size) != self.block_size:
            self.block_size = int(block_size)
            self.trash_page = max(int(max_pages), 1)
            self.pages = torch.full(
                (self.trash_page + 1, self.block_size), -1,
                dtype=torch.int64, device=self.device,
            )
            self.pages_flat = self.pages.view(-1)
            return True
        return self.ensure_capacity(max_pages)
