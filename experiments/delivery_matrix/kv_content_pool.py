"""Single-authority, RAM-backed immutable KV payload arena.

Aliases (including rank/group namespaces) are separate from payload storage.
This CPU primitive is not yet the CED connector or a proven TP8 offload path.
"""
from dataclasses import dataclass, field
import fcntl
import hashlib
import mmap
import os
import threading
import ctypes


def _memfd():
    # Some Conda Python builds omit os.memfd_create despite a capable Linux
    # kernel/runtime libc. Never silently substitute a disk-backed KV file.
    flags = getattr(os, "MFD_CLOEXEC", 1) | getattr(os, "MFD_ALLOW_SEALING", 2)
    if hasattr(os, "memfd_create"):
        return os.memfd_create("dsv41-ced-kv-unique", flags)
    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.memfd_create
    function.argtypes, function.restype = [ctypes.c_char_p, ctypes.c_uint], ctypes.c_int
    fd = function(b"dsv41-ced-kv-unique", flags)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


class PoolFull(RuntimeError):
    pass


@dataclass
class _Content:
    ident: int
    digest: str
    offset: int
    length: int
    span: int
    aliases: set = field(default_factory=set)
    pins: int = 0


class ReadLease:
    def __init__(self, pool, content, view):
        self._pool, self._ident, self.view = pool, content.ident, view
        self.closed = False

    def close(self):
        if not self.closed:
            self.view.release()
            self._pool._unpin(self._ident)
            self.closed = True

    def __enter__(self):
        return self.view

    def __exit__(self, *args):
        self.close()


class KVContentPool:
    def __init__(self, capacity_bytes, alignment=64, digest=None):
        if (type(capacity_bytes) is not int or capacity_bytes <= 0
                or type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1)):
            raise ValueError("Positive RAM capacity and power-of-two alignment required")
        self.capacity, self.alignment = capacity_bytes, alignment
        self._digest = digest or (lambda value: hashlib.sha256(value).hexdigest())
        self._lock = threading.RLock()
        self._fd = _memfd()
        os.ftruncate(self._fd, capacity_bytes)
        # Prevent peer truncation/SIGBUS; authority retains writable mmap.
        fcntl.fcntl(self._fd, getattr(fcntl, "F_ADD_SEALS", 1033),
                    getattr(fcntl, "F_SEAL_GROW", 4) | getattr(fcntl, "F_SEAL_SHRINK", 2)
                    | getattr(fcntl, "F_SEAL_SEAL", 1))
        self._mapping = mmap.mmap(self._fd, capacity_bytes, flags=mmap.MAP_SHARED,
                                  prot=mmap.PROT_READ | mmap.PROT_WRITE)
        self._aliases, self._contents, self._buckets = {}, {}, {}
        self._free, self._next_id, self._closed = [(0, capacity_bytes)], 1, False

    def _check_open(self):
        if self._closed:
            raise RuntimeError("KV content pool is closed")

    def _allocate(self, span):
        for index, (offset, available) in enumerate(self._free):
            if available >= span:
                self._free[index:index + 1] = [] if available == span else [(offset + span, available - span)]
                return offset
        raise PoolFull("Unique KV payload budget exhausted; existing aliases are unchanged")

    def _collect(self, content):
        if content.aliases or content.pins:
            return
        self._contents.pop(content.ident)
        bucket = self._buckets[(content.digest, content.length)]
        bucket.remove(content.ident)
        if not bucket:
            self._buckets.pop((content.digest, content.length))
        self._free.append((content.offset, content.span))
        merged = []
        for offset, span in sorted(self._free):
            if merged and merged[-1][0] + merged[-1][1] == offset:
                previous, count = merged.pop()
                merged.append((previous, count + span))
            else:
                merged.append((offset, span))
        self._free = merged

    def put(self, alias, payload):
        hash(alias)  # Caller supplies an immutable, namespaced logical key.
        view = memoryview(payload).cast("B")
        try:
            if not view.nbytes:
                raise ValueError("An empty payload is not a KV page")
            digest = self._digest(view)
            with self._lock:
                self._check_open()
                content = None
                for ident in self._buckets.get((digest, view.nbytes), ()):
                    candidate = self._contents[ident]
                    existing = memoryview(self._mapping)[candidate.offset:candidate.offset + candidate.length]
                    try:
                        if existing == view:  # Hash collisions cannot alias unequal actual bytes.
                            content = candidate
                            break
                    finally:
                        existing.release()
                if content is None:
                    span = (view.nbytes + self.alignment - 1) // self.alignment * self.alignment
                    previous_free = list(self._free)
                    offset = self._allocate(span)
                    try:
                        self._mapping[offset:offset + view.nbytes] = view
                        written = memoryview(self._mapping)[offset:offset + view.nbytes]
                        try:
                            if self._digest(written) != digest:
                                raise ValueError("KV source changed during hashing/copy; alias not committed")
                        finally:
                            written.release()
                    except BaseException:
                        self._free = previous_free
                        raise
                    content = _Content(self._next_id, digest, offset, view.nbytes, span)
                    self._next_id += 1
                    self._contents[content.ident] = content
                    self._buckets.setdefault((digest, view.nbytes), []).append(content.ident)
                previous = self._aliases.get(alias)
                content.aliases.add(alias)
                self._aliases[alias] = content.ident
                if previous is not None and previous != content.ident:
                    old = self._contents[previous]
                    old.aliases.remove(alias)
                    self._collect(old)
                return {"content_id": content.ident, "offset": content.offset, "length": content.length,
                        "sha256_or_test_digest": digest, "allocated_span_bytes": content.span}
        finally:
            view.release()

    def acquire(self, alias):
        with self._lock:
            self._check_open()
            content = self._contents[self._aliases[alias]]
            content.pins += 1
            view = memoryview(self._mapping)[content.offset:content.offset + content.length].toreadonly()
            return ReadLease(self, content, view)

    def _unpin(self, ident):
        with self._lock:
            content = self._contents[ident]
            if content.pins <= 0:
                raise RuntimeError("Unbalanced KV read lease")
            content.pins -= 1
            self._collect(content)

    def drop(self, alias):
        with self._lock:
            self._check_open()
            content = self._contents[self._aliases.pop(alias)]
            content.aliases.remove(alias)
            self._collect(content)

    def readonly_fd(self):
        with self._lock:
            self._check_open()
            # Reopen rather than dup: dup would retain the authority's O_RDWR.
            return os.open(f"/proc/self/fd/{self._fd}", os.O_RDONLY | os.O_CLOEXEC)

    def stats(self):
        with self._lock:
            self._check_open()
            stat = os.fstat(self._fd)
            return {"capacity_bytes": self.capacity, "unique_contents": len(self._contents),
                    "logical_aliases": len(self._aliases), "read_pins": sum(c.pins for c in self._contents.values()),
                    "unique_payload_bytes": sum(c.length for c in self._contents.values()),
                    "allocated_span_bytes": sum(c.span for c in self._contents.values()),
                    "free_span_bytes": sum(span for _, span in self._free),
                    "backing": "memfd RAM; no disk KV payload file", "device": stat.st_dev, "inode": stat.st_ino,
                    "metadata_or_DMA_staging_bytes_included": False,
                    "actual_TP8_CED_offload_proven": False}

    def close(self):
        with self._lock:
            if self._closed:
                return
            if any(content.pins for content in self._contents.values()):
                raise RuntimeError("Cannot close pool with active KV read leases")
            self._mapping.close()  # Also refuses leaked exported Python buffers.
            os.close(self._fd)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
