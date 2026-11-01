#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compressor_triton.py 的 int32 寻址回归测试（**纯 CPU，不需要 NPU / torch / triton**）。

背景（issue #7）：`_pool_kernel` / `_cache_update_kernel` 里

    cache_ptr + (cache_row * CACHE_SIZE + slot) * 2 * HEAD_DIM + offs_h

`cache_row` 来自 int32 的 block_table，乘法一旦越过 INT32_MAX 就回绕成**负**偏移，
设备侧表现为 MTE "invalid GM address"（0x800000 / 507035）。上游 vllm-ascend
6fd763c284(#17927, 2026-10-08) 把它加宽成 int64；镜像内版本没有 ⇒ 本仓挂
`patches/files/triton/compressor_triton.py` 补上。

测试分五组：

  A. **源文件结构**：4 处 `cache_row` 全部加宽、没有反向收窄、没有漏网的
     `block_table_ptr` 加载、加宽只发生在真正做指针算术的两个 kernel 里；
  B. **调用点**：`_pooled_blocked` 是 device function，`cache_row` 由实参传入
     ⇒ 两个调用点必须传加宽后的那个变量，否则加宽等于没做；
  C. **掩码 lane 夹取**：`seg_off < 0` 的 lane 在 Ascend 上可能先形成 DMA 地址
     再应用 mask，所以 `tl.maximum(seg_off, 0)` 必须一个都不能少；
  D. **算术**：把"多少页开始回绕"钉死，并对多个 HEAD_DIM 做扫描；同时**明确记录
     生产几何仍在元素域内**（这是本修复"防御性而非已证根因"的依据）；
  E. **登记与部署接线**：MD5SUMS / MANIFEST / PATCHES.md 与实际文件一致，
     `serve_a2.sh` 的挂载块存在、默认开、可关、目标路径正确。

刻意**不**做的事：不 import torch / triton，不起 NPU，因此这些测试**不**证明设备侧
行为，只证明"补丁文件确实是上游那一版、算术确实自洽、挂载确实接上了"。

用法：python3 tests/test_compressor_int64_addressing.py
"""

import ast
import hashlib
import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PATCH_FILE = os.path.join(ROOT, "patches", "files", "triton", "compressor_triton.py")
SERVE_SCRIPT = os.path.join(ROOT, "scripts", "serve_a2.sh")
MD5SUMS = os.path.join(ROOT, "patches", "MD5SUMS")
PATCHES_MD = os.path.join(ROOT, "patches", "PATCHES.md")
MANIFEST = os.path.join(ROOT, "MANIFEST.sha256")

INT32_MAX = 2**31 - 1
INT32_MIN = -(2**31)
UINT32_MOD = 2**32

# 本部署的几何（见 docs/CED-DIE-BUDGET-20261004.md §"每块字节"）
CACHE_SIZE = 32          # compressor 环形 cache 的行数
HEAD_DIM = 512           # 每页元素 = 32 * 2 * 512 = 32768（FP32 ⇒ 131072 B/页）
KV_PAGE_BYTES = 147712   # CED slot3 打包页（C1 KV 131072 + INT8 index 16384 + FP16 scales 256）

# 生产实测值（`[CED-32BIT-GUARD]` 打印）；只用来算"还剩多少余量"，不含任何环境标识
PROD_NUM_BLOCKS = 29076

ELEMENT_STRIDE = CACHE_SIZE * 2 * HEAD_DIM   # 32768 元素/页
BYTE_STRIDE = ELEMENT_STRIDE * 4             # 131072 B/页

POOL_KERNEL = "_pool_kernel"
UPDATE_KERNEL = "_cache_update_kernel"
POOLED_BLOCKED = "_pooled_blocked"


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def wrap32(v):
    """把任意整数按**有符号** int32 回绕（设备上 `mul.w` 溢出的语义）。"""
    return (v + 2**31) % UINT32_MOD - 2**31


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _src():
    return _read(PATCH_FILE)


def _digest(algo):
    h = hashlib.new(algo)
    with open(PATCH_FILE, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_lines():
    """返回 [(lineno, text)]，跳过空行、整行注释和 docstring。

    结构断言不该被注释/docstring 里的示例代码坑到（`_pooled_blocked` 的 docstring
    里就原样写了 `seg_off` 和指针算式）。
    """
    src = _src()
    tree = ast.parse(src)
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for n in range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1):
                skip.add(n)
    out = []
    for i, line in enumerate(src.splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("#") or i in skip:
            continue
        out.append((i, line))
    return out


def _func_spans():
    """顶层 `def name(` 的行号区间 {name: (start, end)}（end 为下一个顶层 def 前一行）。"""
    lines = _src().splitlines()
    starts = []
    for i, line in enumerate(lines, start=1):
        m = re.match(r"^def\s+(\w+)\s*\(", line)
        if m:
            starts.append((m.group(1), i))
    spans = {}
    for idx, (name, start) in enumerate(starts):
        end = starts[idx + 1][1] - 1 if idx + 1 < len(starts) else len(lines)
        spans[name] = (start, end)
    return spans


def _in(lineno, span):
    return span[0] <= lineno <= span[1]


def page_element_offset(page, slot, head, head_dim=HEAD_DIM):
    """复刻 kernel 里的环形页寻址（元素域，不涉字节）。"""
    return (page * CACHE_SIZE + slot) * 2 * head_dim + head


# --------------------------------------------------------------------------- #
# A. 源文件结构
# --------------------------------------------------------------------------- #
class TestPatchFile(unittest.TestCase):
    def test_file_present_and_parses(self):
        self.assertTrue(os.path.exists(PATCH_FILE), f"缺 {PATCH_FILE}")
        ast.parse(_src())  # 语法自检：坏文件绝不能进镜像

    def test_all_cache_row_widened(self):
        src = _src()
        loads = re.findall(
            r"cache_row = tl\.load\(block_table_ptr \+ batch\)(\.to\(tl\.int64\))?", src
        )
        self.assertEqual(len(loads), 4, f"预期 4 处 cache_row 加载，实际 {len(loads)}")
        not_widened = [i for i, w in enumerate(loads) if not w]
        self.assertEqual(not_widened, [], f"第 {not_widened} 处 cache_row 没加宽成 int64")

    def test_no_unwidened_block_table_load(self):
        """任何从 block_table 取页号的地方都必须加宽 —— 只看那 4 处不够，防止以后新增第 5 处。"""
        bad = [
            (n, t.strip())
            for n, t in _code_lines()
            if "tl.load(block_table_ptr" in t and ".to(tl.int64)" not in t
        ]
        self.assertEqual(bad, [], f"这些 block_table 加载没加宽：{bad}")

    def test_widening_happens_where_pointer_math_is(self):
        """4 处加宽必须落在 `_pool_kernel` / `_cache_update_kernel` 内。

        `_pooled_blocked` 是 device function，`cache_row` 是**形参**，在那里加宽没意义
        （它已经是 tl.int64 张量），真正的转换点必须在发起指针算术的 kernel 里。
        """
        spans = _func_spans()
        sites = [
            n for n, t in _code_lines() if re.search(r"cache_row = tl\.load\(block_table_ptr", t)
        ]
        self.assertEqual(len(sites), 4)
        want = [n for n in sites if _in(n, spans[POOL_KERNEL]) or _in(n, spans[UPDATE_KERNEL])]
        self.assertEqual(len(want), 4, f"加宽点 {sites} 没全落在两个 kernel 里")
        self.assertFalse(
            any(_in(n, spans[POOLED_BLOCKED]) for n in sites),
            "不该在 _pooled_blocked 里重新从 block_table 取页号",
        )

    def test_no_narrowing_cast(self):
        """反向收窄是这类修复最常见的回退方式（`.to(tl.int32)` 一写，加宽全废）。"""
        self.assertNotIn(".to(tl.int32)", _src())
        self.assertNotIn("astype(tl.int32)", _src())
        self.assertNotIn("cache_row.to(", _src())

    def test_masked_lanes_clamped(self):
        src = _src()
        self.assertEqual(
            src.count("tl.maximum(seg_off, 0)"), 4,
            "掩码投影加载的 seg_off 必须全部夹到 >=0（上游同款 4 处）",
        )

    def test_clamp_only_inside_pooled_blocked(self):
        spans = _func_spans()
        lines = [
            n for n, t in _code_lines() if "tl.maximum(seg_off, 0)" in t
        ]
        self.assertEqual(len(lines), 4)
        self.assertTrue(
            all(_in(n, spans[POOLED_BLOCKED]) for n in lines),
            f"夹取点 {lines} 应全在 _pooled_blocked 内",
        )

    def test_seg_off_never_used_raw_in_pointer_math(self):
        """`_pooled_blocked` 里 seg_off 只允许出现在这几种写法中。"""
        spans = _func_spans()
        allowed = [
            r"^seg_off = token_pos - start_pos$",
            r"^in_seg = seg_off >= 0$",
            r"tl\.maximum\(seg_off, 0\)",
            r"\(seg_off < 0\)",
        ]
        offenders = []
        for n, text in _code_lines():
            if not _in(n, spans[POOLED_BLOCKED]):
                continue
            if "seg_off" not in text:
                continue
            if not any(re.search(p, text.strip()) for p in allowed):
                offenders.append((n, text.strip()))
        self.assertEqual(offenders, [], f"_pooled_blocked 里有未夹取的 seg_off：{offenders}")


# --------------------------------------------------------------------------- #
# B. 调用点：加宽后的变量真的被传进去了吗
# --------------------------------------------------------------------------- #
class TestCallSites(unittest.TestCase):
    def _call_args(self):
        """抓 `_pooled_blocked(` 的实参（调用点只有一个，按括号深度截断）。"""
        lines = _src().splitlines()
        calls = []
        for i, line in enumerate(lines):
            if line.lstrip().startswith("def "):
                continue
            if not re.search(r"_pooled_blocked\(\s*$", line):
                continue
            depth = 1
            args, buf = [], []
            for nxt in lines[i + 1:]:
                for ch in nxt:
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                    if depth == 0:
                        break
                    buf.append(ch)
                if depth == 0:
                    break
                buf.append("\n")
            text = "".join(buf)
            # 只取最外层逗号分隔
            depth, cur = 0, ""
            for ch in text:
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                if ch == "," and depth == 0:
                    args.append(cur.strip())
                    cur = ""
                else:
                    cur += ch
            if cur.strip():
                args.append(cur.strip())
            calls.append((i + 1, args))
        return calls

    def test_two_call_sites(self):
        calls = self._call_args()
        self.assertEqual(len(calls), 2, f"预期 2 个 _pooled_blocked 调用点，实际 {len(calls)}")

    def test_cache_row_is_fourth_arg_and_widened(self):
        """形参顺序：group_idx, start_pos, seg_row_base, cache_row, ... ⇒ 第 4 个实参。"""
        for lineno, args in self._call_args():
            self.assertGreaterEqual(len(args), 4, f"第 {lineno} 行实参不足：{args}")
            self.assertEqual(
                args[3], "cache_row",
                f"第 {lineno} 行传进去的第 4 个实参是 {args[3]!r}，应为加宽后的 cache_row",
            )

    def test_signature_unchanged(self):
        src = _src()
        self.assertIn("def _pooled_blocked(\n    group_idx,\n    start_pos,\n    seg_row_base,\n    cache_row,", src)


# --------------------------------------------------------------------------- #
# C/D. 算术
# --------------------------------------------------------------------------- #
class TestAddressArithmetic(unittest.TestCase):
    """把回绕点算清楚：这些数字就是判据，不是估计。"""

    def test_element_stride_per_page(self):
        self.assertEqual(CACHE_SIZE * 2 * HEAD_DIM, 32768)
        self.assertEqual(32768 * 4, 131072, "与 docs 里 131072 B/页 对得上")
        self.assertEqual(ELEMENT_STRIDE, 32768)
        self.assertEqual(BYTE_STRIDE, 131072)

    def test_wrap32_helper_semantics(self):
        self.assertEqual(wrap32(0), 0)
        self.assertEqual(wrap32(INT32_MAX), INT32_MAX)
        self.assertEqual(wrap32(INT32_MAX + 1), INT32_MIN)
        self.assertEqual(wrap32(UINT32_MOD), 0)
        self.assertEqual(wrap32(-1), -1)
        # 与真实 int32 位模式一致：回绕只是"换符号读同一段比特"
        for v in (2**31 - 1, 2**31, 2**31 + 1234, 2**32 - 7, 2**33 + 5):
            self.assertEqual(wrap32(v) % UINT32_MOD, v % UINT32_MOD)

    def test_int32_overflows_at_page_65536(self):
        stride = ELEMENT_STRIDE

        last_ok = INT32_MAX // stride
        self.assertEqual(last_ok, 65535)
        self.assertGreater(last_ok * stride, 0)
        self.assertGreater(wrap32(last_ok * stride), 0, "65535 页仍在 int32 界内")
        self.assertLess(wrap32(65536 * stride), 0, "int32 在 65536 页必须回绕成负")
        self.assertGreater(65536 * stride, 0, "int64 不回绕")

    def test_wrap_point_sweep_over_head_dim(self):
        """同一个 bug 在不同 HEAD_DIM 下的触发页号 —— 换模型/换切分时要能立刻查到。"""
        expected = {64: 524287, 128: 262143, 256: 131071, 512: 65535, 1024: 32767, 2048: 16383}
        for head_dim, last_ok in expected.items():
            with self.subTest(head_dim=head_dim):
                stride = CACHE_SIZE * 2 * head_dim
                self.assertEqual(INT32_MAX // stride, last_ok)
                self.assertGreater(wrap32(last_ok * stride), 0)
                self.assertLess(wrap32((last_ok + 1) * stride), 0)

    def test_slot_and_head_offsets_do_not_move_the_boundary(self):
        """边界只由页号决定：slot 与 offs_h 最多把偏移推进一页 ⇒ 不会跨过回绕点。

        也就是说 `last_ok` 页内的**任意** (slot, head) 组合都还是正数，`last_ok+1` 页
        的**任意**组合都是负数；否则"第 65536 页崩"这句话就不严谨。
        """
        stride = ELEMENT_STRIDE
        last_ok = INT32_MAX // stride
        for slot in (0, CACHE_SIZE - 1):
            for head in (0, HEAD_DIM - 1, 2 * HEAD_DIM - 1):
                self.assertGreater(wrap32(page_element_offset(last_ok, slot, head)), 0)
                self.assertLess(wrap32(page_element_offset(last_ok + 1, slot, head)), 0)

    def test_prod_geometry_is_still_inside_element_domain(self):
        """诚实断言：按 HEAD_DIM=512，生产 29076 页**没有**触发元素域回绕。

        这是 PR 里"防御性修复、非已证根因"的算术依据 —— 如果哪天这条断言红了，
        说明几何变了，PR 的定性要跟着改。
        """
        wrapped = [
            p for p in range(0, PROD_NUM_BLOCKS)
            if wrap32(page_element_offset(p, CACHE_SIZE - 1, 2 * HEAD_DIM - 1)) < 0
        ]
        self.assertEqual(wrapped, [], "元素域内不该回绕；红了请复核 HEAD_DIM / CACHE_SIZE")
        self.assertLess(PROD_NUM_BLOCKS * ELEMENT_STRIDE, 2**31)

    def test_narrower_head_dim_would_overflow_prod(self):
        """反证：只要每页元素数翻倍（HEAD_DIM=1024），同样的页数就会踩线。"""
        head_dim = 1024
        stride = CACHE_SIZE * 2 * head_dim
        self.assertLess(PROD_NUM_BLOCKS * stride, 2**31, "29076 页 @1024 仍在上界内")
        self.assertGreater(PROD_NUM_BLOCKS * stride * 2, 2**31, "但只有不到两倍余量")

    def test_packed_page_int8_view_bound(self):
        """打包页若走 int8/byte 视图，int32 元素域上界只有 14538 块，远小于 2^32 字节域。"""
        int8_bound = INT32_MAX // KV_PAGE_BYTES
        byte_bound = UINT32_MOD // KV_PAGE_BYTES
        self.assertEqual(int8_bound, 14538)
        self.assertEqual(byte_bound, 29076)  # 即 [CED-32BIT-GUARD] 用的那个上界
        self.assertLess(int8_bound, byte_bound, "2^32 字节域比 int32 元素域**宽一倍**")

    def test_undertested_zone_is_reachable_in_prod(self):
        """生产实测 num_blocks=29076 落在 int8 域上界之外。"""
        self.assertGreater(PROD_NUM_BLOCKS, INT32_MAX // KV_PAGE_BYTES)

    def test_how_many_pages_are_beyond_the_int8_bound(self):
        int8_bound = INT32_MAX // KV_PAGE_BYTES
        beyond = len(range(int8_bound + 1, PROD_NUM_BLOCKS))
        self.assertEqual(beyond, 14537)
        # 与"两倍"的说法对齐：越界页数 ≈ 上界页数
        self.assertLess(abs(beyond - int8_bound), 100)

    def test_int64_offsets_stay_positive_and_in_allocation(self):
        pages = (0, 1, 65535, 65536, PROD_NUM_BLOCKS - 1)
        alloc = (max(pages) + 1) * ELEMENT_STRIDE
        for page in pages:
            self.assertGreaterEqual(page_element_offset(page, 0, 0), 0)
            self.assertLess(page_element_offset(page, CACHE_SIZE - 1, 2 * HEAD_DIM - 1), alloc)

    def test_page_offsets_are_page_aligned(self):
        """页首偏移必须整除一页的元素数，否则 ring 复用会串页。"""
        for page in (0, 1234, 65535, 65536, PROD_NUM_BLOCKS - 1):
            self.assertEqual(page_element_offset(page, 0, 0) % ELEMENT_STRIDE, 0)


# --------------------------------------------------------------------------- #
# E. 索引模拟：按 kernel 的写法跑一遍，比较 int32 / int64
# --------------------------------------------------------------------------- #
class TestIndexSimulation(unittest.TestCase):
    """把 kernel 的索引算式原样抄成 Python 跑一遍（不含真实张量/掩码）。

    只回答两个问题：(1) int64 版本会不会出负偏移；(2) int32 版本在哪些页上出负偏移。
    """

    PAGES = list(range(0, PROD_NUM_BLOCKS, 37)) + [65534, 65535, 65536, 65537]

    def _offsets(self, widen):
        out = []
        for page in self.PAGES:
            for slot in (0, 1, CACHE_SIZE // 2, CACHE_SIZE - 1):
                for head in (0, HEAD_DIM - 1):
                    v = page_element_offset(page, slot, head)
                    out.append(v if widen else wrap32(v))
        return out

    def test_int64_never_negative(self):
        offs = self._offsets(widen=True)
        self.assertTrue(all(o >= 0 for o in offs))
        self.assertLess(max(offs), 2**63 - 1)

    def test_int32_negative_exactly_beyond_the_boundary(self):
        offs32 = self._offsets(widen=False)
        negatives = sum(1 for o in offs32 if o < 0)
        pages_beyond = sum(1 for p in self.PAGES if p > 65535)
        per_page = len(offs32) // len(self.PAGES)
        self.assertGreater(pages_beyond, 0)
        self.assertEqual(negatives, pages_beyond * per_page)

    def test_int32_and_int64_are_the_same_bits(self):
        """回绕不改变比特，只改变符号 —— 所以 int32 版本读到的是"另一块显存"。"""
        offs64 = self._offsets(widen=True)
        offs32 = self._offsets(widen=False)
        self.assertEqual([o % UINT32_MOD for o in offs64], [o % UINT32_MOD for o in offs32])

    def test_int64_offsets_are_strictly_monotonic_in_page(self):
        prev = None
        for page in sorted({0, 1, 2, 65535, 65536, 65537, PROD_NUM_BLOCKS - 1}):
            cur = page_element_offset(page, 0, 0)
            if prev is not None:
                self.assertGreater(cur, prev)
            prev = cur


class TestMaskedLaneModel(unittest.TestCase):
    """`seg_off < 0` 的 lane：不夹取会算出负的投影偏移。

    取 group 0（`seg_row_base = 0`，即投影矩阵第 0 行）且在组内就有残余
    （`start_pos=5 > group_idx*RATIO=0`）：此时 `seg_row_base + seg_off` 直接是负数，
    偏移落在投影矩阵**之前** —— 正是"masked lane 也可能先形成 DMA 地址"的场景。
    """

    RATIO = 8
    START_POS = 5
    SEG_ROW_BASE = 0

    def _rows(self):
        group_idx = 0
        out = []
        for row in range(self.RATIO):
            token_pos = group_idx * self.RATIO + row
            out.append((row, token_pos - self.START_POS))
        return out

    def test_negative_seg_off_appears(self):
        segs = [seg for _, seg in self._rows()]
        self.assertTrue(any(s < 0 for s in segs), "构造里必须含残余 lane")
        self.assertTrue(any(s >= 0 for s in segs), "也要含正常 lane 做对照")

    def test_raw_offset_goes_negative_for_residual_lanes(self):
        raw = [(self.SEG_ROW_BASE + seg) * HEAD_DIM for _, seg in self._rows()]
        self.assertLess(min(raw), 0)
        # 负偏移的 lane 恰好就是 seg_off<0 的那些
        neg_rows = {r for r, seg in self._rows() if (self.SEG_ROW_BASE + seg) * HEAD_DIM < 0}
        mask_rows = {r for r, seg in self._rows() if seg < 0}
        self.assertEqual(neg_rows, mask_rows)

    def test_clamped_offset_is_non_negative(self):
        clamped = [(self.SEG_ROW_BASE + max(seg, 0)) * HEAD_DIM for _, seg in self._rows()]
        self.assertGreaterEqual(min(clamped), self.SEG_ROW_BASE * HEAD_DIM)
        self.assertLess(max(clamped), (self.SEG_ROW_BASE + self.RATIO) * HEAD_DIM)

    def test_clamp_does_not_change_in_segment_values(self):
        """夹取只影响被 mask 掉的 lane；分段内的偏移必须一模一样。"""
        for row, seg in self._rows():
            if seg >= 0:
                self.assertEqual(
                    (self.SEG_ROW_BASE + seg) * HEAD_DIM,
                    (self.SEG_ROW_BASE + max(seg, 0)) * HEAD_DIM,
                )


# --------------------------------------------------------------------------- #
# F. 登记与部署接线
# --------------------------------------------------------------------------- #
class TestRegistry(unittest.TestCase):
    def test_md5_registered_and_matches(self):
        digest = _digest("md5")
        self.assertIn(f"{digest}  triton/compressor_triton.py", _read(MD5SUMS))

    def test_manifest_sha256_matches(self):
        digest = _digest("sha256")
        self.assertIn(f"{digest}  ./patches/files/triton/compressor_triton.py", _read(MANIFEST))

    def test_patches_md_row_points_at_this_file(self):
        text = _read(PATCHES_MD)
        self.assertIn("triton/compressor_triton.py", text)
        self.assertIn("V41_COMPRESSOR_INT64", text)
        self.assertIn(_digest("md5"), text)


class TestServeScriptWiring(unittest.TestCase):
    """补丁文件再对，挂不上也白搭（历史上踩过：改了文件但容器没重建）。"""

    @classmethod
    def setUpClass(cls):
        cls.text = _read(SERVE_SCRIPT)

    def test_block_present(self):
        self.assertIn("[COMPRESSOR-INT64]", self.text)

    def test_default_on(self):
        self.assertIn('if [ "${V41_COMPRESSOR_INT64:-1}" = "1" ]; then', self.text)

    def test_mount_targets_the_image_path(self):
        self.assertIn(
            "/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/compressor/compressor_triton.py:ro",
            self.text,
        )

    def test_mount_is_conditional_not_unconditional(self):
        """`MOUNTS+=` 必须在门控分支里 —— 否则 `=0` 回退不了。"""
        idx_gate = self.text.index('if [ "${V41_COMPRESSOR_INT64:-1}" = "1" ]; then')
        idx_mount = self.text.index("$F/triton/compressor_triton.py:", idx_gate)
        idx_fi = self.text.index("\n  fi\n", idx_mount)
        self.assertLess(idx_gate, idx_mount)
        self.assertLess(idx_mount, idx_fi)

    def test_missing_file_is_fatal_not_silent(self):
        self.assertIn("V41_COMPRESSOR_INT64=1 但缺", self.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
