#!/usr/bin/env python3
"""[T_draftceiling] **②c** 的补丁生成器：draft block 128 → 64（保持 BF16、保留投机解码）。

## 为什么是这两处（逐文件、逐锚点）

### (1) `vllm_ascend/models/deepseek_v41/dspark.py` —— 给 draft 单独的 block_size
上游 vLLM 的 `DeepseekV4SWACache.__init__` 里 `self.block_size = 64` 是**硬编码 64**
（`vllm/v1/attention/backends/mla/sparse_swa.py:79`），后端
`DeepseekSparseSWABackend.get_supported_kernel_block_sizes() = [MultipleOf(64)]`
⇒ **64 与 128 都是合法的块大小**；Ascend 侧把它统一成
`DSV4_BLOCK_SIZES[cache_config.block_size][0][1]` = **128**（`models/deepseek_v4/model.py:115`）。
`AscendDeepseekV4SWACache.get_kv_cache_spec()` 返回的 `AscendSlidingWindowMLASpec(block_size=self.block_size, ...)`
被 `DeepseekV41DSparkSWACache.get_kv_cache_spec()` 逐字搬进 `DeepseekV41DraftSWASpec(block_size=spec.block_size, ...)`。
⇒ **只改这一个属性**（draft 类的 `self.block_size`），draft 组的 spec / 页 / 块表 / 元数据全部跟着变成 64；
  target 的 40 个 SWA 层用的是**另一个类**（`AscendDeepseekV4SWACache`）⇒ **一个字都不动**。

### (2) `vllm_ascend/core/deepseek_v41.py::plan_cache_slots` —— 放宽 draft 的几何相等检查
现在写死 `draft_spec.block_size != swa_spec.block_size ⇒ raise`。②c 之后 draft=64 / target SWA=128
⇒ 必须改成**整除**关系（`swa % draft == 0`）；其余（`head_size` / `sliding_window` / 装得下）不变。
★ 这条检查的本意是"draft 与 target 叠在同一个 slot 上"的安全带：真正必要的是
  **页装得下**（`Σ draft 平面 ≤ capacity`）与 **窗口语义一致**（head/window）；块大小本身不必相等。
★ **同槽混块有先例**：state 组就是 32 行页与 128 行页共享同一个 block ID
  （`AscendCircularBufferSpec`, `STATE_RING_ROWS=32`）。

### 不动的（逐条给理由）
* `AscendSlidingWindowMLASpec.real_page_size_bytes`：`storage_block_size = block_size // compress_ratio`
  ⇒ draft `compress_ratio=1` ⇒ 自动 `64 × 1 × 512 × 2 = 65,536`。**无需改**。
* `reshape_cache`：逐字用 `spec.storage_block_size` 建 `as_strided` 视图。**无需改**。
* `DeepseekV41MetadataBuilder.build()`：`storage_block_size=spec.storage_block_size, logical_block_size=spec.block_size`
  **取自该组自己的 spec**（`attention/dsa_v41.py:1252,1261`）⇒ draft 组自动拿到 64。**无需改**。
* `slot_key = f"slot:c{ratio}:b{spec.storage_block_size}"`（`dsa_v41.py:1024`）：按 (ratio, 块大小) 分键
  ⇒ 64 自带一格，不会与 128 撞。**无需改**。
* 卸载层（`offloading/scheduler.py` + `p2_pool.py`）：`tokens_per_block` 从组 spec 现算
  ⇒ `sw_chunks = cdiv(128, 64) = 2`、`reachable_tail = 2 + eagle(1) = 3`（原来 2）。
  **无需改代码，但池配额要重算（+1 unit / 1024 token / 请求）**。
* `DeepseekV41DraftSWASpec.__post_init__`：仍要求 `dtype == bfloat16`
  ⇒ **②c 保持 BF16，正是这条不变量想要的**。**无需改**。

## 开关（默认 = 逐字旧行为，零回归风险）
```
VLLM_V41_DRAFT_BLOCK=128   # 默认；逐字等价于现状
VLLM_V41_DRAFT_BLOCK=64    # ★ ②c
```

用法（不占卡）：
    python3 patch_draft_blk.py --core <影子包/core/deepseek_v41.py> \
        --dspark <镜像/models/deepseek_v41/dspark.py> --out-dir <patched>
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import pathlib
import py_compile
import sys

ANCHOR_DSPARK = "class DeepseekV41DSparkSWACache(AscendDeepseekV4SWACache):\n"

REPLACE_DSPARK = '''class DeepseekV41DSparkSWACache(AscendDeepseekV4SWACache):
    """[TDC-2c] DSpark draft 的 SWA cache：**块大小与 target 解耦**。

    上游 vLLM 的 `DeepseekV4SWACache` 硬编码 `self.block_size = 64`，后端
    `get_supported_kernel_block_sizes()` 是 `MultipleOf(64)`（64/128 均合法）；
    Ascend 侧统一成 `DSV4_BLOCK_SIZES[cache_config.block_size][0][1]`（=128）。
    本类只调 **draft 这一个类** 的块大小（env 门控，默认 128 = 逐字旧行为）：
    页大小 = `block_size × 1 × head_size × 2` ⇒ 128→131,072 / 64→65,536。
    target 的 40 个 SWA 层走 `AscendDeepseekV4SWACache`，**不受影响**。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _blk = int(os.environ.get("VLLM_V41_DRAFT_BLOCK", "128"))
        if _blk <= 0 or _blk % 64 != 0:
            raise ValueError(f"[TDC-2c] VLLM_V41_DRAFT_BLOCK 必须是 64 的正倍数，得到 {_blk}")
        self.block_size = _blk

'''

ANCHOR_CHECK = ("            if (\n"
                "                draft_spec.block_size != swa_spec.block_size\n"
                "                or draft_spec.head_size != swa_spec.head_size\n")

REPLACE_CHECK = ("            if (\n"
                 "                # [TDC-2c] draft 允许比 target 更细的页（64 ⊂ 128），但必须整除：\n"
                 "                # 同一 block ID 在 draft 组里代表 64 token、在 target 组里代表 128 token\n"
                 "                # —— 与 state 组（32 行页）共享 block ID 的先例同构。\n"
                 "                swa_spec.block_size % draft_spec.block_size != 0\n"
                 "                or draft_spec.head_size != swa_spec.head_size\n")

ANCHOR_CAP = ("        capacity = max(kv_bytes + index_bytes, "
              "*(sum(_cache_plane_sizes(specs[n])) for n in aliases))\n")

REPLACE_CAP = ("        # [TDC-2c] 槽位页必须装得下叠在它上面的每一个平面（含 draft）。\n"
               "        _draft_size = 0\n"
               "        if slot_idx < len(draft):\n"
               "            _draft_size = sum(_cache_plane_sizes(specs[draft[slot_idx]]))\n"
               "        capacity = max(kv_bytes + index_bytes,\n"
               "                       *(sum(_cache_plane_sizes(specs[n])) for n in aliases), _draft_size)\n")


def md5(p: pathlib.Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def need(text: str, anchor: str, what: str) -> None:
    n = text.count(anchor)
    if n != 1:
        sys.exit(f"[patch_draft_blk] 锚点 {what} 出现 {n} 次（期望 1）⇒ 源文件版本不对")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", required=True, help="core/deepseek_v41.py（KV8 影子包版）")
    ap.add_argument("--dspark", required=True, help="models/deepseek_v41/dspark.py（镜像原版）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--slots-draft-aware", action="store_true",
                    help="同时把 capacity 改成 max(kv+index, aliases, draft)（draft 更小时非必需）")
    ap.add_argument("--variant", choices=("clean", "core-only"), default="clean",
                    help="clean = 生产形状（dspark.py 读 env，2 文件）；"
                         "core-only = **只为 8 卡端到端测试**的单文件变体（块大小覆写放在 spec 的 __post_init__，"
                         "因为 8 卡 runner 的 inner.sh 只转发白名单 env、无法把 dspark.py 的 env 递进去）")
    a = ap.parse_args()

    core_p, dspark_p = pathlib.Path(a.core), pathlib.Path(a.dspark)
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    d = dspark_p.read_text()
    if "[TDC-2c]" in d:
        sys.exit("[patch_draft_blk] dspark.py 已打过 [TDC-2c]")
    need(d, ANCHOR_DSPARK, "dspark class")
    # REPLACE_DSPARK 重写了类头 + 新增 __init__，末尾停在下一条原语句之前
    # （原文的 `    def get_kv_cache_spec(self, vllm_config):` 正好接在类体里）。
    d2 = d.replace(ANCHOR_DSPARK, REPLACE_DSPARK, 1)
    if "\nimport os\n" not in d2:
        d2 = d2.replace('"""Aurora / DeepSeek-V4.1 dSPark draft model for Ascend."""\n',
                        '"""Aurora / DeepSeek-V4.1 dSPark draft model for Ascend."""\n\nimport os\n', 1)
    if "\nimport os\n" not in d2:
        d2 = "import os\n" + d2
    ast.parse(d2)

    c = core_p.read_text()
    if "[TDC-2c]" in c:
        sys.exit("[patch_draft_blk] core/deepseek_v41.py 已打过 [TDC-2c]")
    if "import os" not in c.split("\n\n")[0] and "\nimport os\n" not in c:
        c = "import os\n" + c
    need(c, ANCHOR_CHECK, "draft geometry check")
    c2 = c.replace(ANCHOR_CHECK, REPLACE_CHECK, 1)
    if a.variant == "core-only":
        # ★ 只为 8 卡端到端测试：块大小的覆写放进 spec 的 __post_init__。
        #   语义等价 —— 对 Ascend 的 DraftSWASpec 来说 **spec.block_size 是页几何的唯一来源**
        #   （real_page_size_bytes / reshape_cache / metadata / 块表 / slot_key 全部从它派生），
        #   dspark.py 里那个 `self.block_size` 只用来构造这个 spec。
        #   开关用**文件**（runner 的 inner.sh 只转发白名单 env）；生产形状见 --variant clean。
        anchor_keep = ("        if self.dtype != torch.bfloat16 or self.num_kv_heads != 1 "
                       "or self.compress_ratio != 1:\n")
        need(c2, anchor_keep, "DraftSWASpec.__post_init__")
        c2 = c2.replace(anchor_keep, (
            "        # [TDC-2c] ★ 测试用单文件变体：draft 的块大小可被一个**文件开关**覆写\n"
            "        #   （8 卡 runner 的 inner.sh 只转发白名单 env，无法把 dspark.py 的 env 递进来）。\n"
            "        #   spec.block_size 是页几何的唯一来源 ⇒ 覆写它与改 dspark 的 self.block_size 等价。\n"
            "        _flag = '/work/agents/T_draftceiling/draft_block_64.flag'\n"
            "        if os.path.exists(_flag):\n"
            "            object.__setattr__(self, 'block_size', 64)\n"
            + anchor_keep), 1)
    if a.slots_draft_aware:
        if "_draft_size" in c2:
            print("[patch_draft_blk] capacity 补丁已在（`_draft_size` 在）⇒ 跳过（幂等）")
        else:
            need(c2, ANCHOR_CAP, "capacity=max(...)")
            c2 = c2.replace(ANCHOR_CAP, REPLACE_CAP, 1)
    ast.parse(c2)

    (out / "dspark.py").write_text(d2)
    (out / "deepseek_v41_core.py").write_text(c2)
    for f in ("dspark.py", "deepseek_v41_core.py"):
        py_compile.compile(str(out / f), doraise=True)

    # ★ 自检用**原文**（`ast.unparse` 会丢注释）+ AST 结构双重判定
    tree = ast.parse(c2)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "plan_cache_slots")
    has_mod = any(
        isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod)
        and "draft_spec.block_size" in ast.unparse(n)
        for n in ast.walk(fn)
    )
    assert has_mod, "core 自检失败：plan_cache_slots 里没有 draft 的整除比较"
    assert "TDC-2c" in c2 and "swa_spec.block_size % draft_spec.block_size" in c2, "core 自检失败（原文标记缺失）"
    if a.slots_draft_aware:
        assert "_draft_size" in c2 and "capacity = max(kv_bytes + index_bytes," in c2, "core 自检失败（capacity 未改）"
    tree_d = ast.parse(d2)
    cls = next(n for n in ast.walk(tree_d) if isinstance(n, ast.ClassDef) and n.name == "DeepseekV41DSparkSWACache")
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert {"__init__", "get_kv_cache_spec"} <= methods, f"dspark 自检失败：类方法 {methods}"
    assert "VLLM_V41_DRAFT_BLOCK" in d2 and "import os" in d2, "dspark 自检失败（env 开关缺失）"

    print(f"[patch_draft_blk] core   {core_p}  md5={md5(core_p)}")
    print(f"[patch_draft_blk] dspark {dspark_p} md5={md5(dspark_p)}")
    print(f"[patch_draft_blk] ★ 产物 {out/'dspark.py'} md5={md5(out/'dspark.py')}")
    print(f"[patch_draft_blk] ★ 产物 {out/'deepseek_v41_core.py'} md5={md5(out/'deepseek_v41_core.py')}")
    print("[patch_draft_blk] 自检：AST 两处改动均在 + py_compile 2/2 通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
