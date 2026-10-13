#!/usr/bin/env python3
"""用 aclGetDeviceCapability 读设备的 L2/UB/核心数等硬件规格（只读）。

aclDeviceInfo 的关键取值（见 acl_rt.h）：
  1   AICPU_CORE_NUM            101 AICORE_CORE_NUM      102 CUBE_CORE_NUM
  201 VECTOR_CORE_NUM           204 UBUF_PER_VECTOR_CORE
  301 TOTAL_GLOBAL_MEM_SIZE     302 **L2_CACHE_SIZE**
"""
import ctypes
import ctypes.util
import os
import sys

lib = None
for cand in ("libascendcl.so", "/usr/local/Ascend/cann-9.1.0/lib64/libascendcl.so",
             "/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so"):
    try:
        lib = ctypes.CDLL(cand)
        break
    except OSError:
        continue
if lib is None:
    print("找不到 libascendcl.so"); raise SystemExit(2)

lib.aclInit.argtypes = [ctypes.c_char_p]
lib.aclInit.restype = ctypes.c_int
lib.aclrtGetDeviceInfo.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
lib.aclrtGetDeviceInfo.restype = ctypes.c_int
lib.aclrtGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
lib.aclrtGetDeviceCount.restype = ctypes.c_int

rc = lib.aclInit(None)
if rc != 0:
    print("aclInit rc=%d" % rc); raise SystemExit(3)

n = ctypes.c_uint32(0)
lib.aclrtGetDeviceCount(ctypes.byref(n))
print("可见 die 数 = %d" % n.value)

ATTRS = [
    (101, "AICORE_CORE_NUM"),
    (102, "CUBE_CORE_NUM"),
    (201, "VECTOR_CORE_NUM"),
    (204, "UBUF_PER_VECTOR_CORE(B)"),
    (1,   "AICPU_CORE_NUM"),
    (301, "TOTAL_GLOBAL_MEM_SIZE(B)"),
    (302, "**L2_CACHE_SIZE(B)**"),
]
for dev in range(min(n.value, 2)):
    print("--- die %d" % dev)
    for aid, name in ATTRS:
        v = ctypes.c_int64(0)
        r = lib.aclrtGetDeviceInfo(dev, aid, ctypes.byref(v))
        if r == 0:
            extra = ""
            if "SIZE(B)" in name:
                extra = "  = %.2f GiB" % (v.value / 2**30) if v.value > 2**20 else "  = %.2f MB" % (v.value / 2**20)
            print("    %-26s %14d%s" % (name, v.value, extra))
        else:
            print("    %-26s 查询失败 rc=%d" % (name, r))
