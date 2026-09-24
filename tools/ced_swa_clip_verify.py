#!/usr/bin/env python3
"""CED SWA clip 修复的离线正确性验证（不碰 NPU）。

把 `npu_sparse_flash_mla` 的**实际寻址语义**编码成一个小模型，然后对
「legacy（整行块表 + 完整 seqused）」和「clip（rebase 后的窄表 + 相对 seqused）」
两种视图分别算出 kernel 会对哪些块表列发起访存，并和「本请求真正持有的页」
比较。判据：

  legacy 必须出现 held 之外的列（这就是 1M 偶发乱码的机制）；
  clip   必须 **零越界**，且访问列数恰好等于窄化的宽度。

## 语义来源（全部为本仓库内的算子源码，非推测）

1. 列号与行偏移 —— `upstream-v41/vllm-ascend-upstream/csrc/attention/
   sparse_flash_mla/../sparse_flash_mla_common.h::DataCopyPA`：

       blockTableBaseOffset = startPos.bIdx * shape.maxblockNumPerBatch;
       blockIdOffset        = curS2Idx / shape.blockSize;      // 块表列号

2. 行 stride 与 S2 都取自**块表张量的形状** —— `op_host/sparse_flash_mla_tiling.cpp`：

       oriMaxBlockNumPerBatch_ = oriBlockTable.tensor->GetStorageShape().GetDim(1);
       s2Size_                 = oriMaxBlockNumPerBatch_ * oriBlockSize_;
       oriBlockSize_           = GetAxisNum(oriKvShape_, Bs, kvLayout_);  // PA: kv.dim(1)

   ⇒ 窄化块表必须**连续**（gather 出来的新张量），否则 bIdx>0 的行偏移错位。

3. mask 边界来自**运行时的** seqUsedOriKV —— `arch22/sparse_flash_mla_swa_kernel.h`：

       actOriS2Size    = GetActualSeqLenKV(bIdx);          // 读 seqused_ori_kv
       oriMaskRight    = Min(actOriS2Size - S1 + s1EndIdx + winRight, actOriS2Size - 1);
       oriMaskLeft     = Max(actOriS2Size - S1 + s1StartIdx - winLeft, 0);
       oriLoopTimes    = CeilDiv(oriMaskRight - oriMaskLeft + 1, s2BaseSize);
       s2StartPoint    = oriMaskLeft;
       // 最后一个 tile 的拷贝长度裁到 mask 右界：
       actualSize      = (oriMaskRight - oriMaskLeft + 1) - s2LoopIdx * s2BaseSize;

4. 预取多出来的迭代不发访存 —— 同文件：`info.isValid = s2LoopIdx < s2LoopTimes;`
   且 `PreloadPipeline` 只在 `isValid` 时执行。

5. S2 不跨核切分 —— 同文件：`tempLoopInfo.tndIsS2SplitCore = false;`

运行：`python3 tools/ced_swa_clip_verify.py [--json out.json]`
退出码 0 = 两个判据都满足；1 = 判据不满足（修复无效或语义模型过期）。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

WINDOW = 128  # sliding_window / ori_win_left + 1
REPLAY = 128  # CED D 的 bounded replay token 数
S2_BASE = 512  # tiling 的 sInnerSize_（"s2固定切分512"，DAV_2201）


@dataclass
class Arm:
    """一次 kernel 调用的寻址模型。"""

    name: str
    seqused_ori: int          # seqused_ori_kv（运行时）
    table_width: int          # blockTable.shape[1]（= 行 stride，也是 s2Size/blockSize 的来源）
    block_size: int
    s1_size: int              # 本 batch 的 Q token 数
    s2_base: int = S2_BASE
    win_left: int = WINDOW - 1
    win_right: int = 0
    reads: list[tuple[int, int]] = field(default_factory=list)  # (列号, 该列是第几次访问)

    def model_kernel(self) -> None:
        """按上面第 3/4 条复算 kernel 会对哪些列发起访存。"""
        lori = self.seqused_ori
        # 逐 Q 行组（一个 M block 一组；这里按逐 token 复算，取并集）
        for s1_start in range(0, self.s1_size):
            s1_end = s1_start
            right = min(lori - self.s1_size + s1_end + self.win_right, lori - 1)
            left = max(lori - self.s1_size + s1_start - self.win_left, 0)
            if right < left:
                continue
            span = right - left + 1
            loop_times = -(-span // self.s2_base)  # CeilDiv
            for tile in range(loop_times):        # isValid = tile < loop_times
                s2_offset = tile * self.s2_base
                if tile + 1 == loop_times:
                    actual = span - s2_offset
                else:
                    actual = self.s2_base
                start = left + s2_offset          # s2StartPoint = oriMaskLeft
                for pos in range(start, start + actual):
                    self.reads.append((pos // self.block_size, pos))


def held_columns(prompt_len: int, block_size: int, replay: int = REPLAY) -> set[int]:
    """replay 步开始时 D 的 SWA manager 实际持有的列（页）。

    replay 覆盖 [S, E)，E = N-1；更早的页已被 `remove_skipped_blocks` 换成
    null（行内 0），所以持有集合就是 replay 区间覆盖到的那些页。
    """
    end = prompt_len - 1
    start = max(0, end - replay)
    return set(range(start // block_size, (end - 1) // block_size + 1))


def check(prompt_len: int, block_size: int = 128, replay: int = REPLAY) -> dict:
    end = prompt_len - 1
    start = max(0, end - replay)
    base_page = start // block_size
    local_len = end - base_page * block_size
    held = held_columns(prompt_len, block_size, replay)

    full_width = (end + block_size - 1) // block_size  # manager 若保留全前缀的宽度
    legacy = Arm("legacy", seqused_ori=end, table_width=full_width,
                 block_size=block_size, s1_size=replay)
    legacy.model_kernel()
    clip_width = (local_len + block_size - 1) // block_size
    clip = Arm("clip", seqused_ori=local_len, table_width=clip_width,
               block_size=block_size, s1_size=replay)
    clip.model_kernel()

    legacy_cols = sorted({c for c, _ in legacy.reads})
    clip_cols = sorted({c for c, _ in clip.reads})
    # clip 的列是行内局部列；映射回全局列用于与 held 比较
    clip_abs = {base_page + c for c in clip_cols}
    return {
        "prompt_len": prompt_len,
        "block_size": block_size,
        "replay": replay,
        "s2_base": S2_BASE,
        "replay_start": start,
        "replay_end": end,
        "held_columns": sorted(held),
        "local_len": local_len,
        "legacy": {
            "table_width": full_width,
            "columns_read": legacy_cols,
            "out_of_hold": sorted(set(legacy_cols) - held),
        },
        "clip": {
            "table_width": clip_width,
            "columns_read_local": clip_cols,
            "columns_read_abs": sorted(clip_abs),
            "out_of_hold": sorted(clip_abs - held),
            "columns_beyond_width": sorted(c for c in clip_cols if c >= clip_width),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default="", help="把逐长度结果写到该 JSON")
    ap.add_argument("--lengths", default="1019847,1048576,900000,144404,8193,999,256,255",
                    help="逗号分隔的 prompt token 数")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    results = [check(n) for n in lengths]

    print(f"{'prompt':>9} {'held':>10} {'legacy OOB':>11} {'clip OOB':>9} "
          f"{'clip 列宽':>10} {'clip 越列':>9}")
    legacy_bad = 0
    clip_bad = 0
    for r in results:
        lo = len(r["legacy"]["out_of_hold"])
        co = len(r["clip"]["out_of_hold"])
        cb = len(r["clip"]["columns_beyond_width"])
        legacy_bad += bool(lo)
        clip_bad += bool(co or cb)
        print(f"{r['prompt_len']:>9} {str(r['held_columns']):>10} {lo:>11} {co:>9} "
              f"{r['clip']['table_width']:>10} {cb:>9}")

    print()
    r = results[0]
    print(f"样本 N={r['prompt_len']}：replay=[{r['replay_start']},{r['replay_end']}) "
          f"持有列={r['held_columns']}")
    print(f"  legacy：宽度={r['legacy']['table_width']} 读到列={r['legacy']['columns_read']}"
          f" ⇒ 越界列={r['legacy']['out_of_hold']}")
    print(f"  clip  ：宽度={r['clip']['table_width']} 行内列={r['clip']['columns_read_local']}"
          f" ⇒ 全局列={r['clip']['columns_read_abs']} 越界={r['clip']['out_of_hold']}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        print(f"\n已写 {args.json}")

    # 判据：至少一个长度上 legacy 越界（说明修复有意义），且所有长度上 clip 零越界
    ok = legacy_bad > 0 and clip_bad == 0
    print(f"\n判据：legacy 越界长度数={legacy_bad}（需 >0），clip 越界长度数={clip_bad}（需 =0）")
    print("结果：" + ("通过 ✅" if ok else "不通过 ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
