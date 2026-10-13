"""Read and verify this process's Linux NUMA policy, without allocating model data."""
import ctypes
import os
import re


ENV = "DP_MATRIX_NUMA_INTERLEAVE_NODES"


def node_list(value):
    if (not isinstance(value, str) or re.fullmatch(r"(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*", value) is None):
        raise ValueError("NUMA interleave nodes must be an explicit comma-separated node list")
    nodes = [int(part) for part in value.split(",")]
    if len(nodes) < 2 or nodes != sorted(set(nodes)) or nodes[-1] >= 1024:
        raise ValueError("NUMA interleave requires at least two distinct ascending nodes below 1024")
    return nodes


def current_policy():
    lib = ctypes.CDLL("libnuma.so.1", use_errno=True)
    query = lib.get_mempolicy
    query.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_ulong),
                      ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong]
    query.restype = ctypes.c_int
    width = ctypes.sizeof(ctypes.c_ulong) * 8
    mask = (ctypes.c_ulong * (1024 // width))()
    mode = ctypes.c_int()
    if query(ctypes.byref(mode), mask, 1024, None, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return {"mode": mode.value,
            "nodes": [i for i in range(1024) if mask[i // width] & (1 << (i % width))],
            "pid": os.getpid(), "scope": "actual calling-thread get_mempolicy; not all existing pages"}


def verify_interleave(value=None):
    value = os.environ.get(ENV) if value is None else value
    if value is None:
        return {"requested_interleave": False, "policy_verified": False}
    expected = node_list(value)
    actual = current_policy()
    if actual["mode"] != 3 or actual["nodes"] != expected:
        raise ValueError("Actual host NUMA policy differs from the explicit interleave plan")
    return {**actual, "requested_interleave": True, "policy_verified": True}
