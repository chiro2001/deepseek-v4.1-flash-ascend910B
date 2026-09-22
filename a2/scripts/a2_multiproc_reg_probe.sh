#!/usr/bin/env bash
# =============================================================================
# a2_multiproc_reg_probe.sh —— **A2 上"8 进程并发注册"的探针**（Phase 4）
#
# 为什么需要它（`logs/065` §3c.4 的诚实边界）：
#   A2 已测：**单进程、单块**注册 1/8/32/64 GiB 全过（用户 16:32 那次）。
#   未测：**8 个 worker 并发、每个 16 张张量、合计 ~392 GiB** —— 而这是**生产真实形态**。
#   A3 上跑过 128 次注册 ret=0（`logs/022`），但那是 **host_mem_pool=1** 的机器；**A2 是 0**。
#
# ★★ 关键：**可以在这台正在生产的 A2 上跑，服务不用停** ——
#    它就是 `docker exec` 进正在服务的容器里起 N 个进程（与用户 16:10/16:32 那两次同一个手法）。
#
# 用法（分三期，从低风险开始；每期看结果再决定要不要继续）：
#   # S1：8 进程 × 4 GiB  = 32 GiB   → 测"8 进程并发注册"这个机制通不通
#   A2_CONTAINER=dsv41-a2 NPROC=8 PER_PROC_GIB=4  A2PROBE_FLOOR_GIB=300 bash a2_multiproc_reg_probe.sh
#   # S2：8 进程 × 16 GiB = 128 GiB  → 规模上去后 ret 是否仍 0 + 首次注册的常驻内存开销
#   A2_CONTAINER=dsv41-a2 NPROC=8 PER_PROC_GIB=16 A2PROBE_FLOOR_GIB=300 bash a2_multiproc_reg_probe.sh
#   # S3：8 进程 × 49 GiB = 392 GiB  → ★ 生产真实形态（内存会被压到 ~47 GiB，需自己判断时机）
#   A2_CONTAINER=dsv41-a2 NPROC=8 PER_PROC_GIB=49 A2PROBE_FLOOR_GIB=40  bash a2_multiproc_reg_probe.sh
#
# ★ 建议：若 S2 通过，**S3 并入正式上线窗口**（上线本身就要注册 392 GiB，把它当成
#   "上线时的第一个判据"就行），不必单独冒一次风险。
#
# 产物：`$HOME/tmp/<YYYYMMDD>/multiproc_reg/a2mp-*.txt` + 屏幕上的 DECISION 段
# =============================================================================
set -uo pipefail

NPROC=${NPROC:-8}
PER_PROC_GIB=${PER_PROC_GIB:-4}
FLOOR_GIB=${A2PROBE_FLOOR_GIB:-300}
COPY_MIB=${COPY_MIB:-256}
DEV_MODE=${DEV_MODE:-per_rank}      # per_rank = 第 i 个进程用第 i 张卡（贴近生产）；same = 全用 0 号
KEEP_SEC=${KEEP_SEC:-3}             # 注册后保持多久（让 8 个进程真的"同时"持有）
DOCKER=${DOCKER:-docker}

HOST_MODE=0
if [ -n "${A2_CONTAINER:-}" ]; then
  HOST_MODE=1
  if ! $DOCKER ps --filter "name=^${A2_CONTAINER}$" --format '{{.Names}}' | grep -q .; then
    echo "[a2mp] 容器 $A2_CONTAINER 不在跑 —— 先起容器，或进容器里直接跑本脚本" >&2
    exit 69
  fi
fi

WORK=$HOME/tmp/$(date +%Y%m%d)/multiproc_reg
mkdir -p "$WORK"
echo "[a2mp] 产物目录 = $WORK"

PY=$WORK/a2_multiproc_reg.py
cat > "$PY" <<'PYEOF'
#!/usr/bin/env python3
"""A2 上「N 进程并发 host 注册」的探针。

每个进程做：
  1. acl.init + set_device（贴近生产：第 i 个进程用第 i 张卡）
  2. 分配 PER_PROC_GIB 的**普通**（pageable）host 内存
  3. ★ 在 barrier 处**等齐**，然后**同时** `aclrtHostRegister(MAPPED)`
  4. 在 256 MiB 切片上做真实 H2D→D2H **逐字节对账**（与单进程探针同一条判据）
  5. 保持 KEEP_SEC 秒（让 N 个进程真的同时持有），再注销、释放

输出：每进程一行结果 + 父进程汇总 + DECISION。
★ 判据是「**注册内存的设备往返**」= True（H2H 通过不算数，同 `logs/014` §4）。
"""

from __future__ import annotations

import ctypes
import gc
import multiprocessing as mp
import os
import sys
import time

BLK = 1 << 20
ACL_HOST_REGISTER_MAPPED = 0

NPROC = int(os.environ.get("NPROC", "8"))
PER_PROC_GIB = float(os.environ.get("PER_PROC_GIB", "4"))
COPY_MIB = int(os.environ.get("COPY_MIB", "256"))
KEEP_SEC = float(os.environ.get("KEEP_SEC", "3"))
FLOOR_GIB = float(os.environ.get("A2PROBE_FLOOR_GIB", "120"))
DEV_MODE = os.environ.get("DEV_MODE", "per_rank")


def mem_avail_gib() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1048576.0
    except OSError:
        pass
    return -1.0


def rss_gib() -> float:
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / float(1 << 30)
    except OSError:
        return -1.0


def blk_pattern(seed: int):
    import numpy as np

    g = np.random.default_rng(seed)
    return g.integers(0, 127, size=BLK, dtype=np.int8)


def worker(rank: int, barrier, q) -> None:
    """一个 worker = 生产里一个 TP rank 的形态（分配 → 注册 → 往返对账）。"""
    import numpy as np
    import torch
    import torch_npu  # noqa: F401  （触发 acl 初始化）
    import acl

    nb = int(round(PER_PROC_GIB * (1 << 30)))
    xfer = min(nb, COPY_MIB << 20)
    rec = {"rank": rank, "reg_ok": False, "xfer_ok": None, "err": "",
           "reg_s": None, "rss_before": None, "rss_after": None,
           "dev": None, "h2d_gbs": None}
    host = None
    try:
        acl.init()
        n_dev = torch.npu.device_count()
        dev = rank % n_dev if DEV_MODE == "per_rank" else 0
        rec["dev"] = dev
        acl.rt.set_device(dev)
        acl.rt.create_stream()

        rec["rss_before"] = round(rss_gib(), 3)
        host = torch.zeros(nb, dtype=torch.int8, device="cpu", pin_memory=False)
        rec["rss_after"] = round(rss_gib(), 3)

        # ★★ 等齐 —— 这一步是"并发"的关键（串行注册测不出争用）
        barrier.wait(timeout=600)
        t0 = time.monotonic()
        devptr, ret = acl.rt.host_register(host.data_ptr(), nb, ACL_HOST_REGISTER_MAPPED)
        rec["reg_s"] = round(time.monotonic() - t0, 3)
        rec["reg_ok"] = ret == 0
        if ret != 0:
            rec["err"] = f"host_register ret={ret}"
            q.put(rec)
            return

        # ★ 判据：真实 H2D→D2H 逐字节（同 logs/014 §4 的生死判据）
        if xfer >= BLK:
            blk = blk_pattern(7)
            rows = xfer // BLK
            host.numpy()[: rows * BLK].reshape(rows, BLK)[:] = blk
            dev_t = torch.zeros(xfer, dtype=torch.int8, device="npu")
            t1 = time.monotonic()
            dev_t.copy_(host[:xfer])
            torch.npu.synchronize()
            dt = time.monotonic() - t1
            rec["h2d_gbs"] = round(xfer / max(dt, 1e-9) / 1e9, 1)
            back = torch.zeros(xfer, dtype=torch.int8, device="cpu", pin_memory=False)
            back.copy_(dev_t)
            torch.npu.synchronize()
            rec["xfer_ok"] = bool(np.array_equal(
                back.numpy().reshape(rows, BLK), np.broadcast_to(blk, (rows, BLK))))
            del back, dev_t

        # 保持一段时间，让 N 个进程**同时**持有已注册内存
        time.sleep(KEEP_SEC)
        try:
            acl.rt.host_unregister(host.data_ptr())
        except BaseException:  # noqa: BLE001
            pass
    except BaseException as exc:  # noqa: BLE001
        rec["err"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    finally:
        try:
            del host
        except BaseException:  # noqa: BLE001
            pass
        gc.collect()
        q.put(rec)


def main() -> int:
    total = NPROC * PER_PROC_GIB
    avail = mem_avail_gib()
    print(f"[a2mp] 计划：{NPROC} 进程 x {PER_PROC_GIB} GiB = {total:.0f} GiB"
          f"  |  可用 {avail:.0f} GiB  |  floor {FLOOR_GIB:.0f} GiB  |  DEV_MODE={DEV_MODE}", flush=True)

    # ★ fail-closed：不够就**不跑**（不许静默降级成"跑了几期"）
    if avail > 0 and (avail - total) <= FLOOR_GIB:
        print(f"[a2mp] ⛔ 余量不足：{avail:.0f} - {total:.0f} = {avail - total:.0f}"
              f" <= floor {FLOOR_GIB:.0f} ⇒ **拒绝运行**（宁可当场失败，也不要压到生产）。", flush=True)
        print("[a2mp]    处理：调小 PER_PROC_GIB，或调低 A2PROBE_FLOOR_GIB（并确认你接受那个余量）。",
              flush=True)
        return 65

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(NPROC)
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(i, barrier, q)) for i in range(NPROC)]
    t0 = time.monotonic()
    for p in procs:
        p.start()

    recs = []
    for _ in range(NPROC):
        try:
            recs.append(q.get(timeout=1800))
        except BaseException as exc:  # noqa: BLE001
            recs.append({"rank": -1, "err": f"queue: {exc}"})
    for p in procs:
        p.join(timeout=120)

    recs.sort(key=lambda r: r.get("rank", -1))
    print(f"\n[a2mp] 用时 {time.monotonic() - t0:.1f}s")
    print(f"[a2mp] {'rank':>4} {'dev':>4} {'reg':>5} {'reg_s':>7} {'xfer':>6} {'rss_before':>11} {'rss_after':>10} {'H2D':>7}  err")
    n_ok = n_xfer = 0
    for r in recs:
        if r.get("reg_ok"):
            n_ok += 1
        if r.get("xfer_ok"):
            n_xfer += 1
        print(f"[a2mp] {r.get('rank', -1):>4} {r.get('dev', -1):>4} "
              f"{str(r.get('reg_ok')):>5} {str(r.get('reg_s')):>7} {str(r.get('xfer_ok')):>6} "
              f"{str(r.get('rss_before')):>11} {str(r.get('rss_after')):>10} "
              f"{str(r.get('h2d_gbs')):>7}  {r.get('err', '')}")

    print("\n[a2mp] ================= DECISION =================", flush=True)
    print(f"[a2mp] 注册成功 {n_ok}/{NPROC}；设备往返判据通过 {n_xfer}/{NPROC}", flush=True)
    if n_ok == NPROC and n_xfer == NPROC:
        print(f"[a2mp]   ⇒ ★ {NPROC} 进程并发注册**全部通过**"
              f"（每进程 {PER_PROC_GIB:.0f} GiB，合计 {total:.0f} GiB）", flush=True)
        print("[a2mp]      下一步：把 PER_PROC_GIB 提到 16（S2），再看一次", flush=True)
    elif n_ok == NPROC and n_xfer < NPROC:
        print("[a2mp]   ⇒ ⚠ 注册能过但**设备往返没全过** —— 这与单进程探针的结论不一致，"
              "属**新发现**，把本段贴回来别自己下结论", flush=True)
    elif n_ok > 0:
        print(f"[a2mp]   ⇒ ⛔ 部分进程注册失败（{NPROC - n_ok} 个）—— 这就是那条"
              "\"生产真实形态没测过\"的风险，把 err 列贴回来", flush=True)
    else:
        print("[a2mp]   ⇒ ⛔ 全部失败 —— 先看 err 列；若都是同一错误码，"
              "说明是**并发/总量**问题而不是单次上限", flush=True)
    print("[a2mp] 显存口径：每进程只用 256 MiB 设备张量（COPY_MIB=0 可关掉）", flush=True)
    print("[a2mp] ============================================", flush=True)
    return 0 if (n_ok == NPROC and n_xfer == NPROC) else 1


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

echo "[a2mp] python 侧脚本：$PY"
echo "[a2mp] 时间：$(date '+%F %T %Z')"

{
  echo "=== 环境 ==="
  date '+%F %T %Z'; hostname
  for f in host_mem_pool host_pin_pre_register mem_host_uva dev_mem_map_host; do
    printf '  %-24s %s\n' "$f" "$(cat "/proc/svm/dev0/feature/$f" 2>/dev/null || echo n/a)"
  done
  free -g | head -2
  echo "  参数: NPROC=$NPROC PER_PROC_GIB=$PER_PROC_GIB FLOOR_GIB=$FLOOR_GIB DEV_MODE=$DEV_MODE"
} 2>&1 | tee "$WORK/a2mp-00-env.txt"

# ---------------------------------------------------------------- 执行
# ★ 与 a2_one_shot_probe.sh 同样的坑：`docker exec` **不继承**宿主 env ⇒ 必须显式 -e 转发
if [ "$HOST_MODE" = "1" ]; then
  DEST=/root/a2_multiproc_reg.py
  $DOCKER cp "$PY" "$A2_CONTAINER:$DEST" >/dev/null 2>&1 || {
    echo "[a2mp] docker cp 失败" >&2; exit 70; }
  EXEC_ENV=(
    -e "NPROC=$NPROC" -e "PER_PROC_GIB=$PER_PROC_GIB" -e "COPY_MIB=$COPY_MIB"
    -e "KEEP_SEC=$KEEP_SEC" -e "A2PROBE_FLOOR_GIB=$FLOOR_GIB" -e "DEV_MODE=$DEV_MODE"
  )
  run_py() {
    $DOCKER exec -i "${EXEC_ENV[@]}" "$A2_CONTAINER" bash -lc '
      source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
      echo "[a2mp][in-container] NPROC=${NPROC:-<unset!>} PER_PROC_GIB=${PER_PROC_GIB:-<unset!>} A2PROBE_FLOOR_GIB=${A2PROBE_FLOOR_GIB:-<unset!>} DEV_MODE=${DEV_MODE:-<unset!>}"
      for _v in NPROC PER_PROC_GIB A2PROBE_FLOOR_GIB DEV_MODE; do
        if [ -z "${!_v:-}" ]; then
          echo "[a2mp] ⛔ 环境变量 $_v 没有传进容器 ⇒ 拒绝运行（否则会静默降级）。" >&2
          exit 65
        fi
      done
      exec python3 "$@"
    ' -- "$@"
  }
  run_py "$DEST" 2>&1 | tee "$WORK/a2mp-10-result.txt"
else
  bash -c 'source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true; exec python3 "$@"' -- "$PY" \
    2>&1 | tee "$WORK/a2mp-10-result.txt"
fi
rc=${PIPESTATUS[0]}

echo
case "$rc" in
  0) echo "[a2mp] ✓ 完成：**全部进程注册 + 设备往返判据都通过**（rc=0）" ;;
  65) echo "[a2mp] ⏹ **拒绝运行**（rc=65）—— 余量不足，这是**保底行为，不是失败**；"
      echo "[a2mp]    调小 PER_PROC_GIB 或调低 A2PROBE_FLOOR_GIB 后重跑。" ;;
  *) echo "[a2mp] ⛔ 完成但**没有全过**（rc=$rc）—— 这正是本探针要找的东西，别自己下结论，把文件贴回来。" ;;
esac
echo "[a2mp] 产物：$WORK/a2mp-10-result.txt"
echo "[a2mp] 关键一行是「注册成功 N/8」与「设备往返判据通过 N/8」。"
exit "$rc"
