#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compressor_triton.py 的 int32 寻址回归测试（**纯 CPU，不需要 NPU / torch / triton**）。

背景（issue #7）：`_pool_kernel` / `_cache_update_kernel` 里

    cache_ptr + (cache_row * CACHE_SIZE + slot) * 2 * HEAD_DIM + offs_h

`cache_row` 来自 int32 的 block_table，乘法一旦越过 INT32_MAX 就回绕成**负**偏移，
设备侧表现为 MTE "invalid GM address"（0x800000 / 507035）。上游 vllm-ascend
6fd763c284(#17927, 2026-10-08) 把它加宽成 int64；镜像内版本没有 ⇒ 本仓挂
`patches/files/triton/compressor_triton.py` 补上。

本测试守三件事：
  1. 挂载文件里 4 处 `cache_row` **全部**加宽（少一处就退化成"某些路径仍然回绕"）；
  2. 掩码 lane 的 `seg_off` 仍然夹到 ≥0（Ascend 可能先算 DMA 地址再看 mask）；
  3. 用纯算术把"什么时候回绕"钉死，避免下次改批大小时想当然。

用法：python3 tests/test_compressor_int64_addressing.py
"""

import ast
import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PATCH_FILE = os.path.join(ROOT, "patches", "files", "triton", "compressor_triton.py")

INT32_MAX = 2**31 - 1

# 本部署的几何（见 docs/CED-DIE-BUDGET-20261004.md §"每块字节"）
CACHE_SIZE = 32          # compressor 环形 cache 的行数
HEAD_DIM = 512           # 每页元素 = 32 * 2 * 512 = 32768（FP32 ⇒ 131072 B/页）
KV_PAGE_BYTES = 147712   # CED slot3 打包页（C1 KV 131072 + INT8 index 16384 + FP16 scales 256）


def _src():
    with open(PATCH_FILE, encoding="utf-8") as f:
        return f.read()


class TestPatchFile(unittest.TestCase):
    def test_file_present_and_parses(self):
        self.assertTrue(os.path.exists(PATCH_FILE), f"缺 {PATCH_FILE}")
        ast.parse(_src())  # 语法自检：坏文件绝不能进镜像

    def test_all_cache_row_widened(self):
        src = _src()
        loads = re.findall(r"cache_row = tl\.load\(block_table_ptr \+ batch\)(\.to\(tl\.int64\))?", src)
        self.assertEqual(len(loads), 4, f"预期 4 处 cache_row 加载，实际 {len(loads)}")
        not_widened = [i for i, w in enumerate(loads) if not w]
        self.assertEqual(not_widened, [], f"第 {not_widened} 处 cache_row 没加宽成 int64")

    def test_masked_lanes_clamped(self):
        src = _src()
        self.assertEqual(
            src.count("tl.maximum(seg_off, 0)"), 4,
            "掩码投影加载的 seg_off 必须全部夹到 >=0（上游同款 4 处）",
        )


class TestAddressArithmetic(unittest.TestCase):
    """把回绕点算清楚：这些数字就是判据，不是估计。"""

    def test_element_stride_per_page(self):
        self.assertEqual(CACHE_SIZE * 2 * HEAD_DIM, 32768)
        self.assertEqual(32768 * 4, 131072, "与 docs 里 131072 B/页 对得上")

    def test_int32_overflows_at_page_65536(self):
        stride = CACHE_SIZE * 2 * HEAD_DIM  # 元素/页

        def wrap32(v):
            return (v + 2**31) % 2**32 - 2**31  # 有符号 int32 回绕

        last_ok = INT32_MAX // stride
        self.assertEqual(last_ok, 65535)
        # 65535 页仍在界内，65536 页回绕成负 ⇒ 负偏移指针
        self.assertGreater(last_ok * stride, 0)
        self.assertLess(wrap32(65536 * stride), 0, "int32 在 65536 页必须回绕成负")
        self.assertGreater(65536 * stride, 0, "int64 不回绕")

    def test_packed_page_int8_view_bound(self):
        """打包页若走 int8/byte 视图，int32 元素域上界只有 14538 块，远小于 2^32 字节域。"""
        int8_bound = INT32_MAX // KV_PAGE_BYTES
        byte_bound = 2**32 // KV_PAGE_BYTES
        self.assertEqual(int8_bound, 14538)
        self.assertEqual(byte_bound, 29076)  # 即 [CED-32BIT-GUARD] 用的那个上界
        self.assertLess(int8_bound, byte_bound, "2^32 字节域比 int32 元素域**宽一倍**")

    def test_undertested_zone_is_reachable_in_prod(self):
        """生产实测 num_blocks=29076 落在 int8 域上界之外。"""
        prod_num_blocks = 29076
        self.assertGreater(prod_num_blocks, INT32_MAX // KV_PAGE_BYTES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
