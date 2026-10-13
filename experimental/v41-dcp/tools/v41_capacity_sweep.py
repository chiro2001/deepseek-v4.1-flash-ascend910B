#!/usr/bin/env python3
"""V4.1 DCP 容量：**只用解析式**扫参（不依赖容器/设备，秒级）。

模型（推导见 `docs/V41-DCP-PROGRESS-20260929.md`，已被真机逐位验证）：

    pool_blocks          = KV_CACHE_MEMORY_BYTES // pool_bytes_per_block
    request_blocks(dcp)  = full_blocks(dcp) + state_blocks + 10 * swa_cap
    full_blocks(dcp)     = 1024 * 8 / dcp     # 8 个 full 平面 @1M、B=128；每 rank 1/dcp
    state_blocks         = 1
    swa_cap              = cdiv(window-1 + in_flight, B) + 1
    in_flight            = async_batches * BAT

关键结论：**滑窗必须复制**（A3 上无法表达"全局窗口 ∩ 本 rank 分片"），
而复制态的 SWA 是**滚动窗口**、成本 `cdiv(127 + in_flight, 128) + 1` 与序列长度无关。
⇒ 唯一能继续压低 `request_blocks` 的杠杆是 **in_flight = async_batches × BAT**。

对照：若滑窗可分片，成本是 `cdiv(127+in_flight, 128*dcp)+1`（本脚本 `--sharded-swa`
给出该理想值，仅作对照，不是可交付配置）。
"""

from __future__ import annotations

import argparse

POOL_BYTES_PER_BLOCK = 540928
BLOCK = 128
WINDOW = 128
FULL_PLANES = 8          # 4×long_kv + 4×indexer.k_cache
SWA_GROUPS = 10          # 40 层 SWA 折成 10 组
STATE_BLOCKS = 1
BASE_KV_BYTES = 5368709120
BASE_DCP1_TOKENS = 1096072   # 【实测】DCP1 / BAT=8192 / async=2


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def request_blocks(dcp: int, bat: int, async_batches: int, sharded_swa: bool = False) -> tuple[int, dict]:
    in_flight = async_batches * bat
    full = cdiv(8192, dcp) if dcp else 0     # 8192 = 1024*8
    gran = BLOCK * dcp if sharded_swa else BLOCK
    swa_cap = cdiv(WINDOW - 1 + in_flight, gran) + 1
    total = full + STATE_BLOCKS + SWA_GROUPS * swa_cap
    return total, {"full": full, "swa_cap": swa_cap, "swa_total": SWA_GROUPS * swa_cap, "in_flight": in_flight}


def report(dcp: int, bat: int, async_batches: int, sharded_swa: bool = False, kv_bytes: int = BASE_KV_BYTES) -> dict:
    pool = kv_bytes // POOL_BYTES_PER_BLOCK
    total, detail = request_blocks(dcp, bat, async_batches, sharded_swa)
    conc = max(0, pool - 1) / total
    tokens = int(conc * 1048576)
    return {
        "dcp": dcp, "bat": bat, "async": async_batches, "pool_blocks": pool,
        "request_blocks": total, "concurrency": round(conc, 3), "tokens": tokens,
        "x_vs_dcp1": round(tokens / BASE_DCP1_TOKENS, 3), **detail,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sharded-swa", action="store_true", help="理想对照：假设滑窗可分片（不可交付）")
    ap.add_argument("--kv-bytes", type=int, default=BASE_KV_BYTES)
    args = ap.parse_args()

    print(f"pool_bytes_per_block = {POOL_BYTES_PER_BLOCK}  (真机逐位验证)")
    print(f"{'dcp':>4} {'BAT':>6} {'async':>6} {'in_flight':>10} {'full':>6} {'swa_cap':>8} "
          f"{'req_blocks':>11} {'conc':>7} {'kv_tokens':>12} {'vs_dcp1':>8}")
    for dcp in (1, 8):
        for bat, async_b in ((8192, 2), (8192, 1), (4096, 2), (4096, 1), (2048, 1), (1024, 1)):
            r = report(dcp, bat, async_b, args.sharded_swa, args.kv_bytes)
            print(f"{r['dcp']:>4} {r['bat']:>6} {r['async']:>6} {r['in_flight']:>10} {r['full']:>6} "
                  f"{r['swa_cap']:>8} {r['request_blocks']:>11} {r['concurrency']:>7.3f} "
                  f"{r['tokens']:>12} {r['x_vs_dcp1']:>8.2f}")
        print()
    d1 = report(1, 8192, 2, args.sharded_swa, args.kv_bytes)
    print(f"自检：DCP1 / BAT=8192 / async=2 → {d1['tokens']} tokens"
          f"（真机 1,096,072，{'✓' if abs(d1['tokens'] - 1096072) < 6000 else '✗'}）")
    d8 = report(8, 8192, 2, args.sharded_swa, args.kv_bytes)
    print(f"自检：DCP8 / BAT=8192 / async=2 → {d8['tokens']} tokens"
          f"（真机 4,475,277，{'✓' if abs(d8['tokens'] - 4475277) < 6000 else '✗'}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
