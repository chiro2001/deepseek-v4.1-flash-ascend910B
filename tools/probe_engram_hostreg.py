#!/usr/bin/env python3
"""定界 `aclrtHostRegister` 失败原因（专治 ret=207001 = ACL_ERROR_RT_MEMORY_ALLOCATION）。

背景
----
A2 上起服时报：

    RuntimeError: aclrtHostRegister(<...>/engram_int8/layers_14_engram_embed.scale.safetensors)
    failed: ret=207001. Known causes: (a) read-only (ret=507899); (b) no device context (ret=107002).

那条提示**遗漏了 207001 的真实含义**。CANN 头文件
`acl/error_codes/rt_error_codes.h`：

    207000  ACL_ERROR_RT_FEATURE_NOT_SUPPORT   feature not support
    207001  ACL_ERROR_RT_MEMORY_ALLOCATION     // only used by out of memory
    507899  （只读映射 —— 由 v8 的 engram :rw 修复解决）
    107002  （no context —— standalone 脚本忘了 acl.rt.set_device）

⇒ **207001 是 OOM**，与权限无关 ⇒ 挂 :rw / 改 flat 布局都治不了它。

为什么"能力探测通过"却仍失败
------------------------------
生产探针只注册 **4 KiB 匿名内存**，回答的是"这个 API 被不被接受"，
**不回答"N GiB file-backed MAP_SHARED 会不会 OOM"**。

本脚本要量出来的量级（**这才是 A2 与 A3 的关键差异**）
-----------------------------------------------------
engram 表体量 **≈206 GiB/rank**，而 `aclrtHostRegister` 是**按 rank 各自注册**的：

    A3: MemTotal 2.0 TiB  ≥  8 × 206 GiB = 1.6 TiB   ⇒ 能容下
    A2: MemTotal 0.75 TiB <  1.6 TiB                 ⇒ **装不下** ⇒ 207001

所以本脚本分三步：① 匿名（复刻探针）；② **单个文件满尺寸**；
③ **单进程注册全部文件**（= 一个 rank 的完整 206 GiB）；④ 可选：N 进程并发各全量。

用法（宿主上用 run_probe_engram_hostreg.sh；这里也可在容器内直接跑）
------------------------------------------------------------------
    python3 probe_engram_hostreg.py --model-dir <模型目录>
    python3 probe_engram_hostreg.py --model-dir <模型目录> --fanout 8
    python3 probe_engram_hostreg.py --model-dir <模型目录> --quick

退出码：0 全过；1 脚本自身出错；2 找到失败点
"""

import argparse
import ctypes
import ctypes.util
import os
import subprocess
import sys
import time

PAGE = 4096
MAP_SHARED = 1
MAP_PRIVATE = 0x02
MAP_ANONYMOUS = 0x20
PROT_READ = 1
PROT_WRITE = 2
# ★★ 必须与 engram_device_index.py:94 一致 —— **是 0，不是 1**。
#    传 1 在本平台会返回 207000 FEATURE_NOT_SUPPORT，看起来像"平台不支持"，
#    其实只是参数错了。（本脚本 v1 就栽在这里，v2 已修。）
ACL_HOST_REGISTER_MAPPED = 0

_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int, ctypes.c_long]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

_ACL_READY = False
ERR = {
    207000: "FEATURE_NOT_SUPPORT（参数/平台不支持；先查 host_register 的 flag 是否为 0）",
    207001: "MEMORY_ALLOCATION ⇒ **OOM**",
    207002: "MEMORY_FREE",
    207004: "NO_DEVICE",
    207005: "RESOURCE_ALLOC_FAIL",
    207006: "NO_PERMISSION",
    507899: "只读映射（挂载 :ro）",
    107002: "no context（缺 acl.rt.set_device）",
}


def describe(ret):
    return f"ret={ret} {ERR.get(ret, '(未知错误码)')}"


def human(n):
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.0f}{u}" if u == "B" else f"{n:.1f}{u}"
        n /= 1024.0
    return f"{n:.1f}PiB"


def ensure_acl():
    """复刻生产代码的 `_ensure_acl()`：**必须先 set_device**，
    否则 standalone 脚本会拿到 107002（no context）。"""
    global _ACL_READY
    if _ACL_READY:
        return
    import acl
    acl.init()
    dev = 0
    try:
        import torch  # noqa: F401
        import torch_npu  # noqa: F401
        dev = int(torch.npu.current_device())
    except Exception:  # noqa: BLE001
        pass
    acl.rt.set_device(dev)
    _ACL_READY = True


def table_files(model_dir):
    d = os.path.join(model_dir, "engram_int8")
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, fn) for fn in sorted(os.listdir(d))
            if fn.endswith(".safetensors") and os.path.isfile(os.path.join(d, fn))]


def env_report(model_dir):
    print("=" * 78)
    print("环境")
    print("=" * 78)
    total = 0
    try:
        mem = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                mem[k.strip()] = int(v.split()[0]) * 1024
        for k in ("MemTotal", "MemFree", "MemAvailable", "Mlocked", "Shmem",
                  "Cached", "AnonPages", "Hugetlb"):
            if k in mem:
                print(f"  {k:18s} {human(mem[k])}")
        total = mem.get("MemTotal", 0)
    except Exception as e:  # noqa: BLE001
        print(f"  /proc/meminfo 读取失败：{e}")
    for k, why in (("/proc/sys/vm/max_map_count", ""),
                   ("/proc/sys/vm/overcommit_memory", "(0=启发式 1=总是允许 2=严格)"),
                   ("/proc/sys/vm/swappiness", "")):
        try:
            print(f"  {k:32s} {open(k).read().strip()} {why}")
        except Exception:  # noqa: BLE001
            pass
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        fmt = lambda v: "unlimited" if v == resource.RLIM_INFINITY else human(v)  # noqa: E731
        print(f"  RLIMIT_MEMLOCK (ulimit -l)        {fmt(soft)} / {fmt(hard)}")
    except Exception:  # noqa: BLE001
        pass
    print(f"  pid / uid                         {os.getpid()} / {os.getuid()}")

    files = table_files(model_dir)
    tot = sum(os.path.getsize(p) for p in files)
    print(f"  engram 表：{len(files)} 个文件，合计 {human(tot)}")
    for p in files:
        st = os.stat(p)
        print(f"      {os.path.basename(p):52s} {human(st.st_size):>10s}  "
              f"mode={oct(st.st_mode)[-3:]} uid={st.st_uid} nlink={st.st_nlink}")
    if tot:
        print()
        print(f"  ★ 每个 rank 都要注册 {human(tot)}")
        print(f"    8 rank 名义合计 = {human(tot * 8)}")
        if total:
            verdict = "≤ MemTotal ✓" if tot * 8 <= total else "**> MemTotal ✗**"
            print(f"    本机 MemTotal = {human(total)}  ⇒  8× 与之比较：{verdict}")
    print()
    return tot


def reg_anon(size):
    ensure_acl()
    import acl
    addr = _libc.mmap(None, size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
    if addr in (None, ctypes.c_void_p(-1).value, 2**64 - 1):
        return None, f"mmap errno={ctypes.get_errno()}"
    try:
        dev, ret = acl.rt.host_register(addr, size, ACL_HOST_REGISTER_MAPPED)
        if ret == 0 and dev:
            try:
                acl.rt.host_unregister(addr)
            except Exception:  # noqa: BLE001
                pass
            return True, "ok"
        return False, describe(ret)
    finally:
        _libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(size))


def reg_file(path, size):
    """复刻 HostMappedSafetensors：O_RDWR → MAP_SHARED → host_register(MAPPED=0)。
    ⚠️ 需要该目录 **:rw** 挂载（生产 serve_a2.sh 会把 engram 目录叠加 :rw）。"""
    ensure_acl()
    import acl
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError as e:
        return None, f"os.open(O_RDWR) 失败：{e}"
    try:
        maplen = min(size, os.path.getsize(path))
        addr = _libc.mmap(None, maplen, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value, 2**64 - 1):
            return None, f"mmap errno={ctypes.get_errno()}"
        try:
            dev, ret = acl.rt.host_register(addr, maplen, ACL_HOST_REGISTER_MAPPED)
            if ret == 0 and dev:
                try:
                    acl.rt.host_unregister(addr)
                except Exception:  # noqa: BLE001
                    pass
                return True, "ok"
            return False, describe(ret)
        finally:
            _libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(maplen))
    finally:
        os.close(fd)


def reg_all(paths, hold_s=0.0):
    """**注册保持不放**，逐个注册全部文件（= 一个 rank 的真实行为）。"""
    ensure_acl()
    import acl
    held = []
    try:
        for p in paths:
            try:
                fd = os.open(p, os.O_RDWR)
            except OSError as e:
                return False, f"{os.path.basename(p)}: os.open(O_RDWR) 失败：{e}", held
            size = os.path.getsize(p)
            addr = _libc.mmap(None, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
            os.close(fd)
            if addr in (None, ctypes.c_void_p(-1).value, 2**64 - 1):
                return False, f"{os.path.basename(p)}: mmap errno={ctypes.get_errno()}", held
            dev, ret = acl.rt.host_register(addr, size, ACL_HOST_REGISTER_MAPPED)
            if ret != 0 or not dev:
                _libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(size))
                return False, f"{os.path.basename(p)}: {describe(ret)}", held
            held.append((addr, size))
        if hold_s > 0:
            time.sleep(hold_s)
        return True, "ok", held
    finally:
        for addr, size in held:
            try:
                acl.rt.host_unregister(addr)
            except Exception:  # noqa: BLE001
                pass
            _libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(size))


def banner(t):
    print("=" * 78)
    print(t)
    print("=" * 78)


def single_process(model_dir, quick):
    files = [p for p in table_files(model_dir) if os.path.isfile(p)]
    if not files:
        print(f"engram_int8/ 下没有 .safetensors：{model_dir}")
        return 1
    rc = 0

    banner("[1] 匿名内存（复刻生产探针的口径；flag 必须是 0）")
    sizes = [PAGE, 1 << 30] if quick else [PAGE, 1 << 30, 8 << 30, 64 << 30]
    for s in sizes:
        t0 = time.time()
        ok, why = reg_anon(s)
        print(f"  {'OK  ' if ok else 'FAIL'} anon {human(s):>10s}  "
              f"{why:<62s} ({time.time()-t0:.2f}s)")
        if ok is False:
            if "FEATURE_NOT_SUPPORT" in (why or ""):
                print("      ⇒ FEATURE_NOT_SUPPORT：确认 host_register 第 3 个参数是 **0**"
                      "（ACL_HOST_REGISTER_MAPPED）；传 1 会得到这个错误。")
            rc = 2
            break
    print()

    banner("[2] 真实 engram 文件：**逐文件满尺寸**注册（每个 rank 的真实行为）")
    for p in files:
        sz = os.path.getsize(p)
        t0 = time.time()
        ok, why = reg_file(p, sz)
        print(f"  {'OK  ' if ok else 'FAIL'} {os.path.basename(p):52s} {human(sz):>10s}  "
              f"{why:<44s} ({time.time()-t0:.1f}s)")
        if ok is False:
            rc = 2
    print()

    banner("[3] 单进程注册**全部**文件并保持（= 一个 rank 的完整 ≈206 GiB）")
    tot = sum(os.path.getsize(p) for p in files)
    t0 = time.time()
    ok, why, _held = reg_all(files)
    print(f"  {'OK  ' if ok else 'FAIL'} 合计 {human(tot)}  {why}  ({time.time()-t0:.1f}s)")
    if ok:
        print(f"      ⇒ **单个 rank 能注册完整 {human(tot)}**。"
              f"若 8 卡起服仍报 207001，问题在**并发规模**（见 [4]）。")
    else:
        print(f"      ⇒ 单个 rank 就注册不下 {human(tot)} ⇒ "
              f"**每个 rank 都必然 OOM**，与并发无关。")
    print()
    return rc


def child_entry(model_dir):
    """子进程入口：注册（前 K 个）文件并保持一小会儿。
    K 由环境变量 `PROBE_CHILD_FILES` 控制（空/0 = 全部）。"""
    files = table_files(model_dir)
    k = int(os.environ.get("PROBE_CHILD_FILES", "0") or 0)
    if k > 0:
        files = files[:k]
    if not files:
        print("[child] 找不到 engram 文件", flush=True)
        return 1
    hold = float(os.environ.get("PROBE_HOLD_S", "3"))
    tot = sum(os.path.getsize(p) for p in files)
    t0 = time.time()
    ok, why, _held = reg_all(files, hold_s=hold)
    print(f"[child pid={os.getpid()}] {'OK  ' if ok else 'FAIL'} {human(tot):>9s}  "
          f"{why}  ({time.time()-t0:.1f}s)", flush=True)
    return 0 if ok else 2


def fanout(model_dir, nproc):
    files = table_files(model_dir)
    if not files:
        print("找不到 engram 文件")
        return 1
    # 分级：PROBE_CHILD_FILES 限制每个子进程注册前 K 个文件（0=全部）
    k = int(os.environ.get("PROBE_CHILD_FILES", "0") or 0)
    sub = files[:k] if k > 0 else files
    tot = sum(os.path.getsize(p) for p in sub)
    try:
        memtotal = 0
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    memtotal = int(line.split()[1]) * 1024
    except Exception:  # noqa: BLE001
        memtotal = 0
    desc = f"前 {len(sub)}/{len(files)} 个文件 = {human(tot)}" if k > 0 else f"全部 {human(tot)}"
    banner(f"[4] 并发：{nproc} 进程 × 各注册 {desc}"
           f"（名义合计 {human(tot * nproc)}；MemTotal {human(memtotal)}）")
    print(f"  ⚠️ 记住：这些 mmap 是 **MAP_SHARED 同一批文件** ⇒ 物理页只有一份，"
          f"名义合计 {human(tot * nproc)} 是**驱动侧计数**、不是物理占用。")
    print("  用 subprocess 起独立进程（**不用 fork** —— 多线程进程里 fork 会出怪事）")
    procs = [subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--child-one", "--model-dir", model_dir],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for _ in range(nproc)]
    bad = 0
    for pc in procs:
        out, _ = pc.communicate()
        for line in (out or "").splitlines():
            if line.startswith("[child"):
                print("  " + line, flush=True)
        if pc.returncode != 0:
            bad += 1
    print(f"  ⇒ {nproc - bad}/{nproc} 成功")
    if bad == 0:
        print("     结论：该并发规模下全部通过 ⇒ 并发不是诱因（或本机内存足够）。")
        return 0
    if memtotal and tot * nproc > memtotal:
        print(f"     结论：名义合计 {human(tot * nproc)} > MemTotal {human(memtotal)}"
              f" ⇒ **内存装不下，这是 207001 的直接原因**。")
        print("          缓解：ENGRAM_DEVICE_INDEX=0（走 host 路径，不注册）；"
              "或减少 TP/rank 数（每 rank 仍要 206 GiB）。")
    else:
        print("     结论：并发下失败、但名义合计**未超** MemTotal ⇒ 更像**驱动侧注册配额**"
              "（按进程/按设备计费），而非物理内存不足。")
    return 2


def main():
    ap = argparse.ArgumentParser(description="定界 aclrtHostRegister 失败原因")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--fanout", type=int, default=0,
                    help="并发进程数（0=不测；建议 8，与 TP 一致）")
    ap.add_argument("--quick", action="store_true", help="跳过大的匿名尺寸扫描")
    ap.add_argument("--child-one", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.child_one:
        return child_entry(a.model_dir)

    env_report(a.model_dir)
    rc = 0
    try:
        rc = single_process(a.model_dir, a.quick)
    except Exception as e:  # noqa: BLE001
        print(f"\n[probe][FAIL] 单进程定界自身出错：{e!r}")
        import traceback
        traceback.print_exc()
        return 1

    if a.fanout > 0:
        try:
            rc = rc or fanout(a.model_dir, a.fanout)
        except Exception as e:  # noqa: BLE001
            print(f"\n[probe][FAIL] 并发测试自身出错：{e!r}")
            return 1

    banner("结论摘要")
    if rc == 0:
        print("  本机 host_register 可用（含满尺寸文件）。若起服仍报 207001，"
              "差异在**并发规模**，把 --fanout 设为 TP 数再跑。")
    else:
        print("  已找到失败点（见上方 FAIL 行）。错误码对照：")
        for code in sorted(ERR):
            print(f"    {code:>7}  {ERR[code]}")
        print("  ⇒ 若是 207001：**OOM**。挂 :rw / 改 flat 布局都治不了它。")
        print("     立即绕过：ENGRAM_DEVICE_INDEX=0（走 host 路径，全程不调 host_register）。")
    print()
    return rc


if __name__ == "__main__":
    sys.exit(main())
