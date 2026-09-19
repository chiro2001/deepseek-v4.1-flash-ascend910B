#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decide whether Engram device-index can be enabled on THIS machine.

The feature keeps the 206 GiB Engram table in host DRAM and lets device
operators read it directly, which removes the whole host lookup path.  Its one
hardware/driver requirement is:

    aclrtHostRegister(ptr, len, MAPPED) accepts an ordinary writable host
    mapping, and a device operator can then read through the returned pointer.

That was measured on **A3 (910C)**:

    decode  288 rows x 256 B        0.0145 ms, independent of table size
    prefill 393216 rows            1.06 ms @1-4 GB -> 6.47 ms @32 GB

It has **never** been measured on **A2 (910B3)**, and this project already
recorded a counter-example on A3 itself: `offload.get_dva(pinned_ptr)` returns 0
and an AIV de-referencing a *registered pinned* address dies with
`507035 MTE invalid GM address` (`docs/A2_VS_A3_DIFF.md` §5).  So the release
probes instead of assuming.

Run this on the target machine inside the same container image the server uses::

    IMAGE=<your image> DEV=<free chip> bash run_probe_a2.sh
    # or directly:
    python3 probe_a2_hostmap.py --chip 4

Exit code 0 = supported, 3 = not supported, 1 = probe error (all printed).
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", type=int, default=None,
                    help="logical device to probe (default: current/torch default)")
    ap.add_argument("--rows", type=int, default=288,
                    help="rows to gather in the end-to-end read check")
    args = ap.parse_args()
    if args.chip is not None:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.chip)

    import torch
    import torch_npu  # noqa: F401  (registers the npu backend)

    # IMPORTANT: create a real NPU context *first*.  `acl.rt.host_register`
    # happily returns a garbage devPtr when no context is current -- on A3 it
    # printed `rtsHostRegister execution failed, the context is a null pointer`
    # and the later copy died.  The server always has a context (torch_npu made
    # one), so the probe must reproduce that order.
    _warm = torch.ones(8, device="npu")
    torch.npu.synchronize()
    del _warm

    print("=" * 72)
    print("[probe] 目标：判断本机能否启用 Engram device-index（host 内存设备直索）")
    print("=" * 72)

    # ---------------------------------------------------------------- 0. 指纹
    try:
        soc = None
        import acl

        acl.init()
        dev = int(torch.npu.current_device())
        soc = acl.get_soc_name()
        print(f"[probe] 设备名        : {torch.npu.get_device_name(dev)}")
        print(f"[probe] SoC           : {soc}")
        print(f"[probe] torch/torch_npu: {torch.__version__} / {torch_npu.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] 取设备指纹失败: {type(exc).__name__}: {exc}")
        dev = 0

    # ------------------------------------------------- 1. 能力查询（仅供参考）
    # aclrtHostMemMapCapabilities 在 A3 上返回 AIC/AIV = SUPPORTED。这个查询
    # 本身在旧驱动上可能不存在，所以失败不影响结论 —— 以第 2 步的**实测注册**
    # 为准（能力位是"声称"，注册+读取才是"事实"）。
    try:
        lib = ctypes.CDLL("libascendcl.so", use_errno=True)
        fn = lib.aclrtHostMemMapCapabilities
        cap = ctypes.c_int(0)
        for name, hac in (("AIC", 2), ("AIV", 3)):
            rc = fn(ctypes.c_uint32(dev), ctypes.c_int(hac), ctypes.byref(cap))
            verdict = {1: "SUPPORTED", 0: "NOT_SUPPORTED"}.get(cap.value, cap.value)
            print(f"[probe] capability {name:<3}   : rc={rc} -> {verdict}")
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] capability 查询不可用（旧驱动正常）: {type(exc).__name__}")

    # ------------------------------------------------------ 2. 实测：走**生产同一套代码**
    # 之前这里手写了一个 4 KB 匿名映射，结果在 `t.cpu()` 上挂住 —— 那是探针
    # 自己造出来的形态（极小的匿名页 + 裸 aclrtHostRegister），和生产路径不同，
    # 拿它下结论会误导。改成完全复刻生产：写一个小 safetensors 文件，用
    # `HostMappedSafetensors` 映射（mmap + host_register + DLPack），
    # 再用 `torch.index_select` 让**设备算子去读 host 内存**。
    import numpy as np
    import tempfile

    from engram_device_index import HostMappedSafetensors

    rows, width = 4096, 256
    rng = np.random.default_rng(1234)
    table = rng.integers(-128, 128, size=(rows, width), dtype=np.int8)
    tmpdir = tempfile.mkdtemp(prefix="engram_probe_")
    path = os.path.join(tmpdir, "probe_table.safetensors")
    try:
        from safetensors.torch import save_file

        save_file({"probe.weight": torch.from_numpy(table)}, path)
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] 写 safetensors 失败: {type(exc).__name__}: {exc}")
        return 1
    print(f"[probe] 测试表 {table.shape} int8 -> {path}")

    mapped = None
    try:
        mapped = HostMappedSafetensors(path)
        print(f"[probe] HostMappedSafetensors 映射成功: {tuple(mapped.tensor.shape)} "
              f"{mapped.tensor.dtype} {mapped.tensor.device}")
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] 映射失败: {type(exc).__name__}: {exc}")
        print("[probe] 注：ret=507899 表示只读映射被拒（本探针写的是 rw 文件，"
              "若出现在这里说明是别的限制）；107002 表示没有设备上下文。")
        return 3

    try:
        pick = np.arange(rows, dtype=np.int64)
        ids = torch.from_numpy(pick).to("npu")
        got = torch.index_select(mapped.tensor, 0, ids)
        torch.npu.synchronize()
        got_np = got.cpu().numpy()
        same = np.array_equal(got_np, table)
        print(f"[probe] device index_select {rows} 行（设备读 host DRAM）"
              f"逐字节一致={same}")
        if not same:
            bad = int((got_np != table).sum())
            print(f"[probe] 内容不符：{bad}/{table.size} 个元素不同")
            return 3
    except Exception as exc:  # noqa: BLE001
        import traceback

        print(f"[probe] 设备侧读取失败: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 3
    finally:
        try:
            if mapped is not None:
                mapped.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            os.remove(path)
            os.rmdir(tmpdir)
        except OSError:
            pass

    print("=" * 72)
    print("[probe] 结论：**本机支持 Engram device-index**")
    print("        起服时保持 V41_ENGRAM_DEVICE_INDEX=auto（默认）即可自动启用；")
    print("        验收时建议设 V41_ENGRAM_DEVICE_INDEX=1 强制开启，避免静默回退。")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
