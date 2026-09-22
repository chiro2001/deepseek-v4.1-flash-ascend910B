#!/usr/bin/env bash
# =============================================================================
# a2_one_shot_probe.sh —— **A2 上第一条（也是唯一一条）要跑的命令**
#
# 一条命令判定：DRAM KV 池该用哪个 API、能要到多大。
# 为什么需要它：A2 的 `host_mem_pool=0`，而 A3 的 8 卡臂在 `aclrtMallocHost` 上撞过
# `207001`（`a2/logs/009` §3.2）；A2 上 Engram 整表注册也撞过 207001（logs/001 §5.1）。
# ⇒ A2 上这两条路（驱动 pinned 池 / `aclrtHostRegister`）**都没有被验证过**，必须实测。
#
# 用法（A2 上，二选一）：
#   ① 在宿主上、服务容器在跑：  A2_CONTAINER=dsv41-a2 bash a2_one_shot_probe.sh
#   ② 已经在容器里：            bash a2_one_shot_probe.sh
#   ★ 加 `LIGHT=1`（**推荐**）用轻档：峰值 host ≈ 40 GiB、显存峰值 256 MiB
#   ★ 加 `A2PROBE_FLOOR_GIB=<GiB>` 指定"给宿主留多少余量"（默认 120，A2 上建议 300）
#
# 产物：`$HOME/tmp/<YYYYMMDD>/p1_pinned/a2-*.txt`（**绝不写 /tmp**）+ 屏幕上的 DECISION 段。
# 只读探测：不加载模型、不起服务、不写模型目录。
#
# ★★ 「要占 NPU 吗」（一次性说清）——回答"能不能和正在跑的服务共存"：
#   * **需要**能看见并用上设备（`acl.init` + `acl.rt.set_device` + 一条 stream）——这步省不掉；
#   * **不**加载模型、**不**起服务、**不**跑算子、**不**做图捕获、**不**长期持有显存；
#   * 显存只在"拷贝判据"里用一小块设备张量（LIGHT 档 **256 MiB**，`COPY_GIB=0` 可完全关掉）；
#   * 峰值 host 内存 ≈ 40 GiB（LIGHT）/ ≈ 200 GiB（全档），**跑完即释放**；
#   * ⇒ **可以和正在服务的 A2 共存**（它是第二个进程，不与 vLLM 抢 HBM 的 KV 池）。
# =============================================================================
set -uo pipefail

HOST_MODE=0
if [ -n "${A2_CONTAINER:-}" ]; then
  HOST_MODE=1
  DOCKER=${DOCKER:-docker}
  if ! $DOCKER ps --filter "name=^${A2_CONTAINER}$" --format '{{.Names}}' | grep -q .; then
    echo "[a2probe] 容器 $A2_CONTAINER 不在跑 —— 先起容器，或进容器里直接跑本脚本" >&2
    exit 69
  fi
fi

WORK=$HOME/tmp/$(date +%Y%m%d)/p1_pinned
mkdir -p "$WORK"
echo "[a2probe] 产物目录 = $WORK"

# 把 python 侧脚本落到工作目录（宿主与容器共享 $HOME 时容器里也能看到）
PY=$WORK/a2_pinned_probe.py
cat > "$PY" <<'PYEOF'
#!/usr/bin/env python3
"""A2 pinned / registered host 能力实测（由 a2_pinned_probe.sh 调用）。"""
from __future__ import annotations

import ctypes
import gc
import mmap
import os
import sys
import time

import numpy as np
import torch
import torch_npu  # noqa: F401

ACL_HOST_REGISTER_MAPPED = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
BLK = 1 << 20
FLOOR_GIB = float(os.environ.get("A2PROBE_FLOOR_GIB", "120"))
# ★ 设备侧张量的大小：**默认 256 MiB**（A2 上 HBM 被服务占着，别要 8 GiB）
COPY_GIB = float(os.environ.get("COPY_GIB", "0.25"))
LIGHT = os.environ.get("LIGHT", "1") == "1"


def mem() -> dict[str, float]:
    out: dict[str, float] = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, _, v = line.partition(":")
            try:
                out[k] = int(v.split()[0]) / 1048576.0
            except (IndexError, ValueError):
                continue
    return out


def memline() -> str:
    m = mem()
    return (f"MemAvailable={m.get('MemAvailable', -1):.1f}GiB "
            f"MemFree={m.get('MemFree', -1):.1f}GiB Mlocked={m.get('Mlocked', -1):.2f}GiB")


def say(tag: str, msg: str = "") -> None:
    print(f"[a2probe] {tag:22s} | {memline()} | {msg}".rstrip(" |"), flush=True)


def ok_floor(need_gib: float = 0.0) -> bool:
    return (mem().get("MemAvailable", 0.0) - need_gib) > FLOOR_GIB


RESULTS: dict[str, object] = {}


# ------------------------------------------------------------------ 1. pinned
def step1_pinned_single(sizes_gib: list[float]) -> None:
    for g in sizes_gib:
        if not ok_floor(g):
            say("pinned-single", f"⏹ MemAvailable 不足，跳过 {g} GiB")
            break
        nb = int(round(g * (1 << 30)))
        t0 = time.monotonic()
        try:
            t = torch.zeros(nb, dtype=torch.int8, device="cpu", pin_memory=True)
            ok, err = True, ""
            del t
        except BaseException as exc:  # noqa: BLE001
            ok, err = False, f"{type(exc).__name__}: {str(exc)[:240]}"
        dt = time.monotonic() - t0
        RESULTS[f"pinned_single_{g}"] = ok
        say("pinned-single", f"{'✓' if ok else '✗'} 单次 {g} GiB（{dt:.1f}s）{err}")
        gc.collect()


def step1b_pinned_multi(chunk_mib: float, target_gib: float) -> None:
    chunk = int(chunk_mib * (1 << 20))
    total, n, err = 0, 0, ""
    held = []
    while total + chunk <= int(target_gib * (1 << 30)):
        if not ok_floor(chunk_mib / 1024.0):
            err = f"（触到 floor {FLOOR_GIB:.0f} GiB）"
            break
        try:
            held.append(torch.zeros(chunk, dtype=torch.int8, device="cpu", pin_memory=True))
        except BaseException as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {str(exc)[:240]}"
            break
        total += chunk
        n += 1
        if n % 64 == 0:
            say("pinned-multi", f"✓ {n} × {chunk_mib:.0f} MiB = {total / (1 << 30):.1f} GiB")
    RESULTS["pinned_multi_total_gib"] = round(total / (1 << 30), 1)
    say("pinned-multi", f"⇒ {n} × {chunk_mib:.0f} MiB = {total / (1 << 30):.1f} GiB {err}")
    held.clear()
    gc.collect()


# ---------------------------------------------------------- 2. ACL 直连 + 注册
def step2_acl() -> "acl":
    import acl

    acl.init()
    dev = 0
    try:
        dev = int(torch.npu.current_device())
    except Exception:  # noqa: BLE001
        pass
    acl.rt.set_device(dev)
    stream, _ = acl.rt.create_stream()
    say("acl", f"acl 就绪 device={dev}")
    return acl, stream


def blk_pattern(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 127, (BLK,), dtype=torch.int8, generator=g)


def fill_tiled(buf: torch.Tensor, blk: torch.Tensor, nbytes: int) -> None:
    rows = nbytes // BLK
    buf.numpy()[: rows * BLK].reshape(rows, BLK)[:] = blk.numpy()


def tiles_eq(buf: torch.Tensor, blk: torch.Tensor, nbytes: int) -> bool:
    rows = nbytes // BLK
    view = buf.numpy()[: rows * BLK].reshape(rows, BLK)
    return bool(np.array_equal(view, np.broadcast_to(blk.numpy(), (rows, BLK))))


def copy_check(acl, stream, host: torch.Tensor, nbytes: int, label: str) -> bool:
    """D2H + H2D + 逐字节对账，**走 torch 自己的拷贝路径**（生产里就是这条流/事件语义）。

    ★ 不要用裸 `acl.rt.memcpy_async` 下结论：`a2/logs/014` §4.3 实测，那种自建 acl 流
      会在 **三种后端上都**报出假的"H2D 丢字节"，而 torch 路径 100% 一致。
    ★ 设备侧张量只开 `COPY_GIB`（默认 256 MiB）——**这一项就是本探针的显存峰值**。
    """
    xfer = min(nbytes, int(COPY_GIB * (1 << 30))) if COPY_GIB > 0 else 0
    if xfer < BLK:
        say(label, "⏭ COPY_GIB=0（或太小）⇒ **跳过设备往返**（只报注册结果，不碰显存）")
        return True
    blk = blk_pattern(5)
    fill_tiled(host, blk, xfer)
    dev = torch.zeros(xfer, dtype=torch.int8, device="npu")
    t0 = time.monotonic()
    dev.copy_(host[:xfer])       # H2D
    torch.npu.synchronize()
    t_h2d = time.monotonic() - t0
    # 设备侧直接验一次（前 1 MiB），避免"回搬路径"掩盖 H2D 的问题
    ok_dev = bool(np.array_equal(dev[:BLK].cpu().numpy(), blk.numpy()))

    back = torch.zeros(xfer, dtype=torch.int8, device="cpu", pin_memory=False)
    t1 = time.monotonic()
    back.copy_(dev)              # D2H
    torch.npu.synchronize()
    t_d2h = time.monotonic() - t1
    ok_full = tiles_eq(back, blk, xfer)
    say(label, f"★设备往返 {xfer >> 20} MiB：H2D {t_h2d:.2f}s={xfer / max(t_h2d, 1e-9) / 1e9:.1f} GB/s"
               f"（设备前 1MiB 一致={ok_dev}）；D2H {t_d2h:.2f}s={xfer / max(t_d2h, 1e-9) / 1e9:.1f} GB/s"
               f" 往返逐字节一致={ok_full}")
    del dev, back
    gc.collect()
    return bool(ok_dev and ok_full)


def step3_pageable(acl, stream, gib: float) -> None:
    nb = int(gib * (1 << 30))
    host = torch.zeros(nb, dtype=torch.int8, device="cpu", pin_memory=False)
    ok = copy_check(acl, stream, host, nb, "pageable")
    RESULTS["pageable_copy_ok"] = ok
    del host
    gc.collect()


def step4_register(acl, stream, sizes_gib: list[float]) -> None:
    for g in sizes_gib:
        if not ok_floor(g):
            say("register", f"⏹ MemAvailable 不足，跳过 {g} GiB")
            break
        nb = int(round(g * (1 << 30)))
        host = torch.zeros(nb, dtype=torch.int8, device="cpu", pin_memory=False)
        t0 = time.monotonic()
        try:
            devptr, ret = acl.rt.host_register(host.data_ptr(), nb, ACL_HOST_REGISTER_MAPPED)
        except BaseException as exc:  # noqa: BLE001
            ret, devptr = -1, f"{type(exc).__name__}: {str(exc)[:200]}"
        dt = time.monotonic() - t0
        ok = ret == 0
        RESULTS[f"register_{g}"] = ok
        say("register", f"{'✓' if ok else '✗'} 注册 {g} GiB ret={ret} dev={devptr} "
                        f"({dt:.1f}s, {dt / max(nb / (1 << 20), 1):.3f} s/GiB) is_pinned={bool(host.is_pinned())}")
        if ok:
            ck = copy_check(acl, stream, host, nb, "register-copy")
            if g <= 8:
                RESULTS["register_copy_ok"] = ck
                say("register", f"  ★ 注册内存的设备往返判据 = {ck}")
            try:
                acl.rt.host_unregister(host.data_ptr())
            except BaseException:  # noqa: BLE001
                pass
        del host
        gc.collect()


# --------------------------------------------------------------- A2 版：文件映射注册
def step5_register_mmap(acl, stream, sizes_gib: list[float], path_root: str) -> None:
    """Engram 同款：文件 + MAP_SHARED + aclrtHostRegister(MAPPED)。

    只探到 8 GiB：文件映射会**弄脏页并回写磁盘**（Engram 的 206 GiB 表就有 108–126 GiB Dirty，
    见 `a2/refs/40` §0），生产池**不要**走文件映射 —— 用匿名/普通内存（step4 那种）。
    """
    for g in sizes_gib:
        nb = int(round(g * (1 << 30)))
        if not ok_floor(g):
            say("register-file", f"⏹ MemAvailable 不足，跳过 {g} GiB")
            break
        path = os.path.join(path_root, f"a2probe-{int(g)}gib.map")
        try:
            with open(path, "wb") as fh:
                fh.truncate(nb)
            fd = os.open(path, os.O_RDWR)
            mm = mmap.mmap(fd, nb, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        except BaseException as exc:  # noqa: BLE001
            say("register-file", f"✗ {g} GiB 文件/映射失败：{type(exc).__name__}: {str(exc)[:200]}")
            RESULTS[f"register_file_{g}"] = False
            continue
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        t0 = time.monotonic()
        try:
            devptr, ret = acl.rt.host_register(addr, nb, ACL_HOST_REGISTER_MAPPED)
        except BaseException as exc:  # noqa: BLE001
            devptr, ret = f"{type(exc).__name__}: {str(exc)[:200]}", -1
        dt = time.monotonic() - t0
        RESULTS[f"register_file_{g}"] = ret == 0
        say("register-file", f"{'✓' if ret == 0 else '✗'} {g} GiB 文件映射注册 ret={ret} "
                             f"dev={devptr}（{dt:.1f}s）")
        if ret == 0:
            host = torch.frombuffer(mm, dtype=torch.int8)
            ck = copy_check(acl, stream, host, nb, "register-file-copy")
            RESULTS["register_file_copy_ok"] = ck
            try:
                acl.rt.host_unregister(addr)
            except BaseException:  # noqa: BLE001
                pass
            del host
        mm.close()
        os.close(fd)
        os.unlink(path)
        gc.collect()


# ------------------------------------------------------------------------ main
def main(argv: list[str]) -> int:
    say("start", f"pid={os.getpid()} torch={torch.__version__} "
                 f"npu_count={torch.npu.device_count()} argv={argv}")
    work = argv[0] if argv else "."
    mode = argv[1] if len(argv) > 1 else "all"
    if mode not in ("all", "pinned", "copy"):
        print(f"[a2probe] 未知模式 {mode!r}；用法：<script> <work_dir> [all|pinned|copy]", flush=True)
        return 64
    acl, stream = step2_acl()

    if mode in ("all", "pinned"):
        say("step1", "—— (a) 单次 pinned 分配（每档独立进程更干净，这里逐档释放）")
        step1_pinned_single([1, 4, 8] if LIGHT else [1, 4, 8, 16, 32])
        say("step1b", "—— (b) 多次小分配累加（256 MiB）")
        step1b_pinned_multi(256, 32 if LIGHT else 128)
    if mode in ("all", "copy"):
        say("step3", "—— (c) 普通 pageable host 内存能不能 DMA")
        step3_pageable(acl, stream, 1.0)
        say("step4", "—— (d) ★★ 普通内存 + aclrtHostRegister(MAPPED)（候选 β 的生死判据）")
        step4_register(acl, stream, [1, 4] if LIGHT else [1, 8, 32, 64])
        say("step5", "—— (e) Engram 同款：文件 + MAP_SHARED + 注册（只到 8 GiB：文件映射会回写磁盘）")
        step5_register_mmap(acl, stream, [1] if LIGHT else [1, 8], work)

    print("\n[a2probe] ================= DECISION =================", flush=True)
    print(f"[a2probe] 单次 pinned：4 GiB={RESULTS.get('pinned_single_4')} "
          f"8 GiB={RESULTS.get('pinned_single_8')} 16 GiB={RESULTS.get('pinned_single_16')} "
          f"32 GiB={RESULTS.get('pinned_single_32')}", flush=True)
    print(f"[a2probe] 多次小分配累加 pinned = {RESULTS.get('pinned_multi_total_gib')} GiB", flush=True)
    print(f"[a2probe] 注册（匿名/普通内存）：1={RESULTS.get('register_1')} 8={RESULTS.get('register_8')} "
          f"32={RESULTS.get('register_32')} 64={RESULTS.get('register_64')}", flush=True)
    print(f"[a2probe] 注册（文件映射，仅探到 8 GiB）：1={RESULTS.get('register_file_1')} "
          f"8={RESULTS.get('register_file_8')}", flush=True)
    print(f"[a2probe] 拷贝判据：pageable={RESULTS.get('pageable_copy_ok')} "
          f"registered={RESULTS.get('register_copy_ok')} "
          f"registered-file={RESULTS.get('register_file_copy_ok')}", flush=True)
    print("[a2probe] 判读：", flush=True)
    if RESULTS.get("register_32") or RESULTS.get("register_64"):
        print("[a2probe]   ⇒ 候选 β 可行：用 mmap + aclrtHostRegister(MAPPED) 做池子，"
              "把 cpu_npu.py 的 pin_memory 换掉（见 logs/014 §4）", flush=True)
    elif RESULTS.get("register_8"):
        print("[a2probe]   ⇒ 注册可用但 ≥32 GiB 没测过/没过：先把这段贴回来，"
              "池子按『能注册到的大小』配（可能要先分片注册）", flush=True)
    elif RESULTS.get("register_4") and not LIGHT:
        print("[a2probe]   ⇒ 轻档只探到 4 GiB；重跑 `LIGHT=0` 才能判 ≥8 GiB 那一格", flush=True)
    elif RESULTS.get("register_4") or RESULTS.get("register_1"):
        print("[a2probe]   ⇒ ★ 小档能注册，但**大档没测**：重跑 `LIGHT=0` 再定池子上限", flush=True)
    elif RESULTS.get("pinned_multi_total_gib"):
        print("[a2probe]   ⇒ 注册不可用但 pinned 总量可以 ⇒ 走候选 α（分片 pinned）", flush=True)
    else:
        print("[a2probe]   ⇒ 两条都被挡：把本文件带回，A2 计划要重做（不要动 cpu_bytes_to_use）", flush=True)
    print(f"[a2probe] 显存口径：设备侧张量峰值 = {COPY_GIB:.2f} GiB（COPY_GIB=0 可关掉）", flush=True)
    print("[a2probe] ============================================", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
PYEOF

echo "[a2probe] python 侧脚本：$PY"
echo "[a2probe] 时间：$(date '+%F %T %Z')"

# 0. 环境快照
{
  echo "=== 0. 环境 ==="
  date '+%F %T %Z'
  hostname; uname -r
  for f in host_mem_pool host_pin_pre_register mem_host_uva dev_mem_map_host; do
    printf '  %-24s %s\n' "$f" "$(cat "/proc/svm/dev0/feature/$f" 2>/dev/null || echo n/a)"
  done
  free -g | head -2
} 2>&1 | tee "$WORK/a2-00-env.txt"

# ---------------------------------------------------------------- 1/2/3/4/5
# HOST_MODE 下容器看不到宿主 $HOME，所以把 python 脚本 docker cp 进去再跑
if [ "$HOST_MODE" = "1" ]; then
  DEST=/root/a2_pinned_probe.py
  $DOCKER cp "$PY" "$A2_CONTAINER:$DEST" >/dev/null 2>&1 || {
    echo "[a2probe] docker cp 失败" >&2; exit 70; }
  PYDIR=/root          # 文件映射测试的临时文件落在容器里
  PY_RUN=/root/a2_pinned_probe.py
  # ★★★ 2026-09-22 16:2x **Bug 修复（静默失败，已实机踩到）**：
  #   `docker exec` **不继承宿主环境变量** ⇒ 容器内 LIGHT 回落默认 "1"、
  #   A2PROBE_FLOOR_GIB 回落 "120"（用户给的 300 被丢掉）。
  #   症状：`LIGHT=0 bash ...` 的输出与 `LIGHT=1` **逐字相同**（只探 1/4 GiB），
  #         而 DECISION 还在提示"重跑 LIGHT=0 再定池子上限" ⇒ **用户会以为跑过了、其实永远拿不到大档**。
  #   （与 A3 runner 上反复出现的"env 白名单不转发"是同一类坑。）
  #   ⇒ 显式用 `-e` 把这三个开关送进容器，**并在容器内回显**（见下面的 effective 自检）。
  EXEC_ENV=(
    -e "LIGHT=${LIGHT:-1}"
    -e "COPY_GIB=${COPY_GIB:-0.25}"
    -e "A2PROBE_FLOOR_GIB=${A2PROBE_FLOOR_GIB:-120}"
  )
  run_py() {
    $DOCKER exec -i "${EXEC_ENV[@]}" "$A2_CONTAINER" bash -lc '
      source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
      # ★ effective 自检：把容器内真正生效的开关打出来（证伪"又是没转发"）
      echo "[a2probe][in-container] LIGHT=${LIGHT:-<unset!>} COPY_GIB=${COPY_GIB:-<unset!>} A2PROBE_FLOOR_GIB=${A2PROBE_FLOOR_GIB:-<unset!>}"
      # ★★ fail-closed：三个开关**必须**都在。少任何一个都说明转发又断了 ——
      #    宁可当场报错，也不要静默回落默认值（默认 LIGHT=1 会让"大档"永远探不到）。
      for _v in LIGHT COPY_GIB A2PROBE_FLOOR_GIB; do
        if [ -z "${!_v:-}" ]; then
          echo "[a2probe] ⛔ 环境变量 $_v 没有传进容器 ⇒ 拒绝运行（否则会静默降级）。" >&2
          exit 65
        fi
      done
      exec python3 "$@"
    ' -- "$@"
  }
else
  PYDIR=$WORK
  PY_RUN=$PY
  run_py() {
    bash -lc 'source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true; exec python3 "$@"' -- "$@"
  }
fi

run_py "$PY_RUN" "$PYDIR" "${MODE:-all}" 2>&1 | tee "$WORK/a2-10-${MODE:-all}.txt"

echo
echo "[a2probe] 完成。把下面这个文件带回（或贴回来）："
echo "  $WORK/a2-10-all.txt"
