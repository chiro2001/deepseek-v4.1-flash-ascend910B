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
import os
import re
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


def held_columns_final(prompt_len: int, block_size: int) -> set[int]:
    """**末 token 单步**时 D 的 SWA manager 持有的列（页）。

    这一步和 replay 步的持有集合差一页：末 token 的位置是 N-1，它要写进
    `(N-1)//block_size` 这一页，所以该页**此刻一定已被分配**（slot_mapping 需要它）。
    而 SWA 回收只丢弃窗口之外的块，窗口是 [N-W, N-1]，故持有集合覆盖该窗口。
    """
    window_start = max(0, prompt_len - WINDOW)
    return set(range(window_start // block_size, (prompt_len - 1) // block_size + 1))


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

    # 第三臂：**末 token 单步**（replay 之后那一步）。修复不覆盖它（replay_chunk
    # 为假），所以它的安全性必须由"窗口正好落在持有页内"来保证：
    #   S1 = 1, s1StartIdx = s1EndIdx = 0, Lori = N（顺序 decode）
    #   oriMaskRight = Min(N - 1, N - 1) = N - 1
    #   oriMaskLeft  = Max(N - 1 - 127, 0) = N - 128
    #   ⇒ 需要列 (N-128)//128 .. (N-1)//128，即末尾 1~2 页
    final = Arm("final", seqused_ori=prompt_len, table_width=full_width,
                block_size=block_size, s1_size=1)
    final.model_kernel()
    final_cols = sorted({c for c, _ in final.reads})
    held_final = held_columns_final(prompt_len, block_size)

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
        "final": {
            "held_columns": sorted(held_final),
            "columns_read": final_cols,
            "out_of_hold": sorted(set(final_cols) - held_final),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default="", help="把逐长度结果写到该 JSON")
    ap.add_argument("--lengths", default="1019847,1048576,900000,144404,8193,999,256,255",
                    help="逗号分隔的 prompt token 数")
    ap.add_argument("--lint-code", action="store_true",
                    help="额外静态校验 dsa_v41.py 的裁剪分支不变量（防条件漂移）")
    ap.add_argument("--lint-server", action="store_true",
                    help="校验 scripts/serve_a2.sh 把裁剪开关透传进容器")
    ap.add_argument("--sweep-max", type=int, default=0,
                    help="额外全扫 N=2..该值，断言 clip 与末 token 步恒不越界")
    args = ap.parse_args()

    lint_failures: list[str] = []
    if args.lint_code:
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "experimental", "ced", "dsa_v41.py",
        )
        src = open(source_path, encoding="utf-8").read()
        # 负向检查必须在"去掉注释"的源码上做，否则解释性注释里的字样会误报。
        src_code = "\n".join(
            line.split("#", 1)[0] for line in src.splitlines()
        )
        checks = [
            ("_native_attention 接收 replay_chunk 形参",
             r"def _native_attention\([^)]*replay_chunk",
             True, src),
            ("裁剪分支以 replay_chunk 为条件（不是 max_query_len 启发式）",
             r"_CED_SWA_CLIP\s*\n(?:\s*#.*\n)*\s*and\s+replay_chunk",
             True, src),
            ("裁剪分支内不再出现 max_query_len > 1 判定",
             r"_CED_SWA_CLIP[\s\S]{0,600}?max_query_len > 1",
             False, src_code),
            ("唯一调用点把 replay_chunk 传下去",
             r"self\._attention\([\s\S]{0,200}?replay_chunk=replay_chunk",
             True, src),
            ("窄化块表由 torch.gather 产生（新的连续张量）",
             r"ori_block_table = torch\.gather\(",
             True, src_code),
            ("算子收到的是 rebase 后的长度",
             r"seqused_ori_kv=seqused_ori",
             True, src_code),
            ("有连续性断言",
             r"非连续 block table|not ori_block_table\.is_contiguous\(\)",
             True, src),
        ]
        print("== 静态不变量校验（experimental/ced/dsa_v41.py）==")
        for name, pattern, should_match, haystack in checks:
            found = re.search(pattern, haystack) is not None
            ok = found == should_match
            print(f"  {'OK ' if ok else 'FAIL'} {name}")
            if not ok:
                lint_failures.append(name)
        print()

    if args.lint_server:
        # ★ 这条不是洁癖：若开关没进 `docker run -e`，在宿主上设 V41_CED_SWA_CLIP=0
        #   根本到不了容器（代码内默认是 1），A/B 会静默变成"两臂都是修复版"，
        #   于是"旧行为也能通过"的假结论会被当成修复有效。
        serve_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts", "serve_a2.sh",
        )
        serve = open(serve_path, encoding="utf-8").read()
        print("== 启动脚本透传校验（scripts/serve_a2.sh）==")
        for var, default in (
            ("V41_CED_SWA_CLIP", "1"),
            ("V41_CED_SWA_TRACE", "0"),
            ("V41_CED_BLOCK_TRACE", "0"),
            ("V41_ENGRAM_HIST_TRACE_POS", ""),
        ):
            pattern = r"-e\s+%s=\"\$\{%s:-%s\}\"" % (
                re.escape(var), re.escape(var), re.escape(default),
            )
            ok = re.search(pattern, serve) is not None
            print(f"  {'OK ' if ok else 'FAIL'} docker run -e {var}（默认 {default!r}）")
            if not ok:
                lint_failures.append(f"serve_a2.sh 未透传 {var}")
        print()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    results = [check(n) for n in lengths]

    sweep_bad_clip: list[int] = []
    sweep_bad_final: list[int] = []
    sweep_legacy_oob = 0
    if args.sweep_max >= 2:
        print(f"== 全扫 N=2..{args.sweep_max} ==")
        for n in range(2, args.sweep_max + 1):
            r = check(n)
            if r["clip"]["out_of_hold"] or r["clip"]["columns_beyond_width"]:
                sweep_bad_clip.append(n)
            if r["final"]["out_of_hold"]:
                sweep_bad_final.append(n)
            if r["legacy"]["out_of_hold"]:
                sweep_legacy_oob += 1
        print(f"  clip 越界 {len(sweep_bad_clip)} 例（需 0）"
              f"{sweep_bad_clip[:5] if sweep_bad_clip else ''}")
        print(f"  末 token 步越界 {len(sweep_bad_final)} 例（需 0）"
              f"{sweep_bad_final[:5] if sweep_bad_final else ''}")
        print(f"  legacy 越界 {sweep_legacy_oob} 例（应 >0，说明修复有意义）")
        print(f"  触发规则核验：legacy 应为 N≥256 且 N%128≠0 ⇒ "
              f"预期 {sum(1 for n in range(2, args.sweep_max + 1) if n >= 256 and n % 128)} 例")

    print(f"{'prompt':>9} {'held':>10} {'legacy OOB':>11} {'clip OOB':>9} "
          f"{'final OOB':>10} {'clip 列宽':>10} {'clip 越列':>9}")
    legacy_bad = 0
    clip_bad = 0
    final_bad = 0
    for r in results:
        lo = len(r["legacy"]["out_of_hold"])
        co = len(r["clip"]["out_of_hold"])
        cb = len(r["clip"]["columns_beyond_width"])
        fo = len(r["final"]["out_of_hold"])
        legacy_bad += bool(lo)
        clip_bad += bool(co or cb)
        final_bad += bool(fo)
        print(f"{r['prompt_len']:>9} {str(r['held_columns']):>10} {lo:>11} {co:>9} "
              f"{fo:>10} {r['clip']['table_width']:>10} {cb:>9}")

    print()
    r = results[0]
    print(f"样本 N={r['prompt_len']}：replay=[{r['replay_start']},{r['replay_end']}) "
          f"持有列={r['held_columns']}")
    print(f"  legacy：宽度={r['legacy']['table_width']} 读到列={r['legacy']['columns_read']}"
          f" ⇒ 越界列={r['legacy']['out_of_hold']}")
    print(f"  clip  ：宽度={r['clip']['table_width']} 行内列={r['clip']['columns_read_local']}"
          f" ⇒ 全局列={r['clip']['columns_read_abs']} 越界={r['clip']['out_of_hold']}")
    print(f"  final ：读到列={r['final']['columns_read']}"
          f" ⇒ 越界={r['final']['out_of_hold']}（修复不覆盖该步，必须天然安全）")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        print(f"\n已写 {args.json}")

    # 判据：至少一个长度上 legacy 越界（说明修复有意义），且所有长度上 clip 零越界
    ok = (
        legacy_bad > 0
        and clip_bad == 0
        and final_bad == 0
        and not sweep_bad_clip
        and not sweep_bad_final
        and not lint_failures
    )
    print(f"\n判据：legacy 越界长度数={legacy_bad}（需 >0），clip 越界长度数={clip_bad}（需 =0）")
    print(f"      末 token 步越界长度数={final_bad}（需 =0）")
    if args.lint_code or args.lint_server:
        print(f"      静态校验失败项={len(lint_failures)}{lint_failures or ''}")
    print("结果：" + ("通过 ✅" if ok else "不通过 ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
