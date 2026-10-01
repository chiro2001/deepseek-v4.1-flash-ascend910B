# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 DSA metadata and fused attention execution.

The model file owns the network topology and projection modules.  This module
owns the attention execution boundary: it gathers every cache plane's metadata
before running the compressor, indexer and sparse-attention operators without
moving cache or scheduler knowledge back into the model.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
)

from vllm_ascend.attention.dsa_v1 import dsv4_dsa_overlap_stream
from vllm_ascend.core.deepseek_v41 import (
    DeepseekV41CompressorStateSpec,
    DeepseekV41FullSpec,
    DeepseekV41IndexerSpec,
    DeepseekV41SWASpec,
)
from vllm_ascend.ops.rope_dsv4 import (
    get_cos_and_sin_dsa,
    get_full_cos_and_sin_dsa_for_layer,
)
from vllm_ascend.utils import npu_stream_switch
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataStage,
    DeviceMetadataTask,
    wait_for_device_metadata,
)

V41_METADATA_BUFFER_SIZE = 1024


def _v41_dcp_on() -> bool:
    """DCP 是否开启（开发期门控，生产环境一律 False）。"""
    from vllm_ascend.patch.platform.patch_v41_dcp import v41_dcp_active

    return v41_dcp_active()


def _is_capturing() -> bool:
    """ACL graph capture 中？capture 区内**绝不能**做 host 同步。

    实测代价：诊断里的 `int(device_tensor)` / 布尔索引（→ `aclnnNonzeroV2`）
    会让 8 个 worker 在内核 warmup 阶段全部报 inner error 并退出
    （run `dcpcap_0929_163809`）。
    """
    try:
        from vllm.forward_context import get_forward_context

        if getattr(get_forward_context(), "capturing", False):
            return True
    except Exception:  # noqa: BLE001 - 不在 forward context 里
        pass
    try:
        return bool(torch.npu.is_current_stream_capturing())
    except Exception:  # noqa: BLE001 - 旧版 torch_npu 没有该 API
        return False


PERF_FLAG_PATH = "/tmp/v41_perf_flags"
_PERF_CACHE = {"t": 0.0, "v": {}}
# ★ [V41-STEPGEN] 步序号（见 `_refresh_perf_flags`）
_STEP_GEN = [0]
# ★ [V41-CMPLENSCACHE] 每步只算一次的全局压缩长度缓存
_CMPLENS_CACHE = {"gen": -1, "id": 0, "ratio": 0, "val": None}
# ★ [V41-CMPLENS-VERIFY] 置 1 时每层重算并与缓存比对（会拖慢，仅验证用）
_CMPLENS_VERIFY = {"n": 0}
# ★★★★★ [V41-SLOTTRACE 2026-09-30 23:58] **SWA 写侧槽位追踪**（env 门 + 文件开关）。
#
# 为什么要它：算子源码证实 ori 读侧寻址是
#     `blockTableIdx = logicalIdx / paOriBlockSize`（见 SWAVectorBlock::GetOriSparseKeyGmOffset），
# 而**块表里的 0 是"块 0"而不是"跳过"**。生产 DCP8 的 ori 块表只有第 0 列非零
# ⇒ 位置 ≥128 的读取（以及**全部写入**）都落到池页 0。
# 读侧已在单卡用"页 0 毒化"证实；写侧必须用本探针直接看 `slot_mapping`：
#   `slot // block_size` = 实际写入的物理块号。
# 判据：若位置 ≥128 的槽位块号全为 0 ⇒ 写侧同样落到页 0（host 侧块表 bug）。
# 编译安全：第一道门是模块级常量（dynamo 可折叠），第二道是 `_is_capturing()`。
_SLOTTRACE = __import__("os").environ.get("V41_SLOTTRACE") == "1"
_SLOTTRACE_DONE = {}


def _refresh_perf_flags() -> dict:
    """**每步刷新一次**文件开关（`os.stat` 是系统调用，不能每层每调用都做）。

    ★ 性能背景（2026-09-30 19:40）：排查发现 DCP8 的 decode 比 13:00 那版慢约 19%
    （33.8 → 40.1 ms/step，DCP1 同期只差 1.5%）。原因是本轮为查算子缺陷加了
    **11 处 `_perf_flags()` 调用**，而它每次都做一次 `os.stat`；EAGER 解码下
    38 层 × 11 ≈ 420 次系统调用/步。
    ⇒ 改成：**每个 step 的 `build()` 里刷新一次**，其余调用只读缓存。
    免重启 A/B 的能力保留（开关变更在**下一步**生效，与图捕获的语义一致）。
    """
    # ★★★★★ [V41-STEPGEN 2026-10-01 05:30] **步序号**。
    #
    # `_refresh_perf_flags()` 由 `build()` 调用，而 `build()` 每个 step 每个
    # kv_cache_group 各调一次（≠ 每层）⇒ 这个计数在**一次模型前向内是常量**，
    # 可以用来给"每步只算一次"的缓存做失效键（见 `_v41_global_cmp_lens`）。
    _STEP_GEN[0] += 1
    import os as _o
    try:
        st = _o.stat(PERF_FLAG_PATH)
    except OSError:
        _PERF_CACHE["t"] = 0.0
        _PERF_CACHE["v"] = {}
        return _PERF_CACHE["v"]
    if st.st_mtime != _PERF_CACHE["t"]:
        out = {}
        try:
            with open(PERF_FLAG_PATH) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip()
        except OSError:
            pass
        _PERF_CACHE["t"] = st.st_mtime
        _PERF_CACHE["v"] = out
    return _PERF_CACHE["v"]


def _perf_flags() -> dict:
    """只读缓存版本（**不在热路径上做系统调用**）。刷新见 `_refresh_perf_flags`。

    性能消融开关是**文件驱动**的（便于不重启逐项 A/B），但读文件由一个
    `os.stat` 系统调用兜底 ⇒ 必须每步只做一次。
    """
    return _PERF_CACHE["v"]


def _v41_dcp_rank() -> int:
    """本进程在 DCP 组内的 rank（DCP 关闭时返回 0）。"""
    if not _v41_dcp_on():
        return 0
    from vllm.distributed import get_dcp_group

    return int(get_dcp_group().rank_in_group)


_V41_SCALAR_CACHE: dict = {}


def _v41_scalar_t(v: float, ref: "torch.Tensor") -> "torch.Tensor":
    """缓存一个 **1 元素 1 维** fp32 张量，用来**强制类型提升**。

    ★ [V41-PROMO 2026-10-01 09:40] 为什么需要它：PyTorch 把 **0 维**张量当标量，
    `bf16_tensor * 0dim_fp32` 的结果仍是 **bf16**（不提升）。而
    `bf16_tensor * 1dim_fp32[1]` 会提升到 **fp32** ⇒ 可以**一次算子**同时完成
    「转 fp32 + 乘系数」，省掉一次 `Cast`。
    缓存保证张量地址固定（图捕获安全），也避免每层一次 H2D。
    """
    key = (float(v), str(ref.device))
    t = _V41_SCALAR_CACHE.get(key)
    if t is None:
        t = torch.tensor([float(v)], dtype=torch.float32, device=ref.device)
        _V41_SCALAR_CACHE[key] = t
    return t


_MERGE_KD_CACHE = None


def _v41_merge_kernel_on() -> bool:
    """是否走 AscendC 融合的 merge 路径（env 门，**默认关**）。

    `V41_DCP_MERGE_KERNEL=1` 打开。默认关的理由：它替换了 merge 的核心数值路径，
    必须先在 8 卡上过完整回归（T=904 / 短问答 / 长针）才敢转正。
    `.so` 不存在时自动退回 Python 路径（不会让引擎起不来）。
    """
    global _MERGE_KD_CACHE
    if _MERGE_KD_CACHE is not None:
        return _MERGE_KD_CACHE
    _MERGE_KD_CACHE = False
    import os as _o

    if _o.environ.get("V41_DCP_MERGE_KERNEL", "0") != "1":
        return False
    try:
        from vllm_ascend.attention import v41_merge_kernel as _mk

        _MERGE_KD_CACHE = bool(_mk.available())
    except Exception:  # noqa: BLE001
        _MERGE_KD_CACHE = False
    return _MERGE_KD_CACHE


_MERGE_PACK_CACHE: dict = {}
_RS_OUT_CACHE: dict = {}


def _v41_rs_merge_on() -> bool:
    """merge 的归约是否用 `reduce_scatter`（env 门，默认关）。

    `V41_DCP_RS_MERGE=1` 打开。默认关的原因是它尚未在真机上通过 A/B 验收；
    验证通过后再改默认值。
    """
    return __import__("os").environ.get("V41_DCP_RS_MERGE", "0") == "1"


def _v41_pack_for_reduce(scaled, weights, align: int = 128, in_a=None):
    """把 `scaled[T,H,D]` 与 `weights[T,H,1]` 写进一个**常驻的零初始化缓冲**。

    为什么要它（真实 8 卡 profiler，run dcpcap_1001_015103，20 forwards）：
      · `F.pad` 触发的 `aclnnConstantPadNd_PadV3AiCore_MemSet` 为 38 次/step、
        **0.603 ms/step**（每次 15.9 µs，只为写 128 列的零）；
      · `PadV3` 本身 **0.234 ms/step**、`ConcatD` **0.246 ms/step**。
    三者合计 1.08 ms/step，而真正有意义的输入只有 131 KB。

    语义：allreduce 之后只读 `[..., :D+1]`；padding 恒为 0 ⇒ 对求和无影响。
    图安全：缓冲按 `(T,H,D,dtype,device)` 缓存，地址跨 replay 不变。
    """
    _T, _H, _D = (int(v) for v in scaled.shape)
    _width = ((_D + 1 + align - 1) // align) * align
    _key = (_T, _H, _D, weights.dtype, str(weights.device))
    _buf = _MERGE_PACK_CACHE.get(_key)
    # ★ 只缓存**小 T**（decode：T=1..96）。prefill 的 T 可达 1.6 万，
    #   `[T,64,640]` fp32 会到 GB 级；那种情况退回 `cat+F.pad`，
    #   不把大缓冲长期留在显存里。
    _bytes = _T * _H * _width * weights.element_size()
    _cacheable = _key in _MERGE_PACK_CACHE or _bytes <= (8 << 20)
    if _buf is None or tuple(_buf.shape) != (_T, _H, _width):
        if not _cacheable:
            _pack = torch.cat([scaled, weights], dim=-1)
            _pad_to = (-_pack.shape[-1]) % align
            return torch.nn.functional.pad(_pack, (0, _pad_to)) if _pad_to else _pack
        _buf = torch.zeros((_T, _H, _width), dtype=weights.dtype, device=weights.device)
        _MERGE_PACK_CACHE[_key] = _buf
    # ★ [V41-PACKDIRECT 2026-10-01 04:35] `in_a*w` **直接写进常驻缓冲**。
    #   原来三步：`output.to(fp32)`(Cast) → `* weights`(Mul) → `copy_`(ViewCopy)；
    #   现在一步：`torch.mul(output, weights, out=buf[..., :D])`
    #   （bf16 × fp32 ⇒ fp32，与缓冲 dtype 一致）
    #   ⇒ 每层省 2 次下发（38 层 = 76 次/step）。
    _buf[..., :_D].copy_(scaled)
    _buf[..., _D : _D + 1].copy_(weights)
    return _buf


def _v41_global_cmp_lens(seq_lens, ratio: int):
    """DCP 分片下，把 cmp 序列长度换算成**全局压缩坐标**再交给算子。

    为什么必须这样（2026-10-01 单卡 + torch 参考实现实测）：
      内核的稀疏因果界是
          cmpMaskRight  = cmpMaskS2Size - actS1Size        （actS1Size = 全局 query 数）
          cmpS2IdLimit  = (cmpMaskRight + s1EndIdx + 1) / cmpRatio
          GetKeyGmOffset: realS2Idx >= s2IdLimit ⇒ 丢弃该键
      它隐含假设 `cmpMaskS2Size = actualCmpS2Size*cmpRatio ≈ actS1Size`，
      即"压缩序列与原始序列同长"。但 DCP8 下 `seqused_cmp_kv` 是**本 rank 分片**
      的压缩行数（≈T/8）⇒ 整条界被平移 `-(T - T/8)` ≈ -0.875T：
        · `t < 0.875T` 的 query 行 **一个 cmp 键都拿不到**（实测 row0 单键最优
          = ori pos0，误差 1e-7 ⇒ 确实没有 cmp 键）；
        · 其余行的键再按**索引值**被截断 ⇒ 向量阶段少搬若干行，而矩阵阶段仍按
          原长度读同一块 `kvMergeGm_` ⇒ **读到残留内存** ⇒ 同一输入跨调用结果
          不同（实测 lse 差异 2.6e-1，且取决于此前跑过什么）。
      取全局长度后 `cmpMaskRight ≈ 0`、`cmpS2IdLimit ≈ (s1EndIdx+1)/cmpRatio`
      （压缩坐标下的因果界），与"索引是本地行号、只受索引值上界过滤"一致。
    """
    if ratio <= 0:
        return seq_lens
    # ★★★★★ [V41-CMPLENSCACHE-2 2026-10-01 05:30] **每步只算一次**。
    #
    # 这个函数在**每层**都被调用，但 `seq_lens` 与 `ratio` 在同一步内对所有层
    # 都一样（`seq_lens` 还是同一个 metadata 对象）⇒ 38 层里只需要算 ≤2 次
    # （ratio=2 一组、ratio=1 一组）。
    # profiler 依据：`FloorDiv` 104 次/step（DCP1 对照只有 53），而这里 38 次是白算。
    #
    # 安全性：缓存键包含 **步序号**（`build()` 每步递增）+ `id(seq_lens)` + `ratio`。
    # 只要 `build()` 在每次前向之前被调用（这是 vLLM 的既有行为），
    # 跨步就不会命中旧值 ⇒ 不存在"值变了但缓存没失效"的静默错误。
    _g = _STEP_GEN[0]
    # 键用 **storage 指针**而不是 `id()`：`metadata.swa.seq_lens[:num_reqs]`
    # 每次调用都会建一个新的 view 对象（`id()` 会变），但底层 buffer 是同一个
    # ⇒ `data_ptr()` 才是在"同一份 metadata"意义上的稳定标识。
    _ck = (_g, int(seq_lens.data_ptr()), int(seq_lens.shape[0]), int(ratio))
    _cc = _CMPLENS_CACHE
    if (_cc["gen"], _cc["id"], _cc["ratio"]) == _ck and _cc["val"] is not None:
        return _cc["val"]
    # ★★★ [V41-CMPGLOB-3-FIX 2026-10-01 03:55] **必须返回新张量，不能直接返回 `seq_lens`。**
    #
    # 我曾为省一次 `FloorDiv` 在 ratio==1 时直接 `return seq_lens`。后果是
    # `seqused_ori_kv` 与 `seqused_cmp_kv` **指向同一块显存**（别名），
    # 而实测短问答出现稳定回归：`17 × 23 等于多少？只回答数字。` 连续 6 次
    # 全部给出乱码（此前同一构建为 `391`）。
    # ⇒ 恢复"物化新张量"的写法（ratio=1 时数值相同，只是多一次小内核）。
    _val = torch.div(seq_lens.to(torch.int32), int(ratio), rounding_mode="floor")
    if _perf_flags().get("cmplens_verify") == "1" and not _is_capturing():
        if _cc["val"] is not None and _CMPLENS_VERIFY["n"] < 40:
            _CMPLENS_VERIFY["n"] += 1
            _same = bool(torch.equal(_cc["val"], _val)) if (
                _cc["val"].shape == _val.shape
            ) else False
            print("[V41-CMPLENS-VERIFY] gen=%d ratio=%d same=%s cached=%s fresh=%s"
                  % (_g, int(ratio), _same, _cc["val"][:3].tolist(), _val[:3].tolist()),
                  flush=True)
    _cc.update(gen=_ck[0], id=_ck[1], ratio=_ck[2], val=_val)
    return _val


def _v41_dcp_group():
    from vllm.distributed import get_dcp_group

    return get_dcp_group()


def _v41_dcp_ori_owner() -> str:
    """ori（滑窗）平面由谁携带。

    合并式 `O = Σ_r w_r·O_r / Σ_r w_r` 要求共享的 ori 窗口只被计一次，
    实现方式有两类，本函数选择用哪一种（可用 env 切换以便真机二分）：

    * `seqused0`（**默认**）：所有 rank 都挂 `ori_kv`/`ori_block_table`（因为算子在
      op 层把 `ori_kv` 当**硬必填**，传 None 直接
      `tensor of oriKv is nullptr[CheckRequiredInOutExistence]`），
      但 rank>0 把 `seqused_ori_kv` 设 0 ⇒ 该 rank 的 ori 贡献为空。
    * `rank0`：只有 rank 0 传 `ori_kv`，其余传 `None`。**实测在 A3 上不可用**
      （见上），保留仅为在别的硬件上复现。
    """
    import os as _os

    return _os.environ.get("V41_DCP_ORI_OWNER", "seqused0")


def _v41_ordered_allreduce(t: torch.Tensor, group) -> torch.Tensor:
    """**定序求和**：`all_gather_into_tensor` + 按 rank 顺序显式相加。

    ★ [V41-DETREDUCE-2 2026-09-30 22:45] 为什么需要独立于 `det_reduce`：
    默认路径（`sepw`）在算**分母**时会单独做一次 `all_reduce(weights)`，
    而 `det_reduce` 只覆盖了主 pack 的那一次。实测（层间二分 + 合并前 LSE 对拍）：
      · layer 2 的输入在"好/坏"两次请求之间**逐位相同**；
      · **8 个 rank 的合并前 `lse` 也逐位相同**；
      · 但 layer 3 的 q 开始分叉（0.6875）
    ⇒ 分叉只能来自**归约的求和顺序**（HCCL all_reduce 在该形状/T 上顺序可变）。
    本函数把任意 all_reduce 换成"定序求和"，数学等价、顺序固定。
    """
    dcp = group.world_size
    if dcp <= 1:
        return t
    tc = t.contiguous()
    g = torch.empty((dcp, *tc.shape), dtype=tc.dtype, device=tc.device)
    torch.distributed.all_gather_into_tensor(g, tc, group=group.device_group)
    acc = g[0]
    for r in range(1, dcp):
        acc = acc + g[r]
    return acc


def _v41_dcp_merge_attention(
    output: torch.Tensor,
    lse: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    diag_seq_lens: torch.Tensor | None = None,
    diag_cmp_lens: torch.Tensor | None = None,
    ori_lse: torch.Tensor | None = None,
    ori_out: torch.Tensor | None = None,
    perf_no_pack: bool = False,
    head_slice: tuple[int, int] | None = None,
    layer_idx: int = -1,
) -> torch.Tensor:
    """把各 rank 的局部 partial attention 用 LSE 加权合并成全局结果。

    数学：全局 `softmax(A ∪ B_0 ∪ … ∪ B_{d-1})`，其中 `A` 只由 rank 0 携带、
    `B_r` 是 rank r 的私有分片。各 rank 上报 `(O_r, L_r)`，
    则 `O = Σ_r e^{L_r}·O_r / Σ_r e^{L_r}` **精确**等于全局结果
    （`Σ_r e^{L_r}` 里 A 恰好出现一次）。分段互不重叠地覆盖全局集时该式恒等，
    已被离线夹具用真算子验证（`relout=0.0`）。

    实现用 `all_gather(LSE)` + 两次 `all_reduce`（输出与权重），**不做 head 维
    all_gather / all_to_all**：因为 TP 已经按 head 切分，每个 rank 的 q 就是自己
    那 8 个 head，各 rank 合并的是**同一批 head 的不同 KV 分片** ⇒ 只需在 rank 维
    做归约，不需要重排 head。这比 SFA 的 all-to-all 方案少一次大通信。

    ★★★ 前提：调用方必须先把 `q` 沿 **head 维** all-gather 到**全部** head，
    否则本函数是错的。理由见 `_v41_dcp_gather_heads`：

    TP 与 DCP 复用同一组 rank 时，TP 把 **head** 切开 —— rank r 只持有
    head `[r·H/d, (r+1)·H/d)`。而 `all_reduce` 是**逐元素**求和，位置 `[t, j]`
    在 rank r 上是「head r·H/d + j」，在 rank r' 上是「head r'·H/d + j」
    —— **完全不同的 head**！若各 rank 只算自己的 head 就做逐元素归约，
    等于把不同 head 的部分结果加在一起，从第一层起就全错。
    （这正是 SFA 的 DCP 实现里那个 `_start_dcp_query_gather` 存在的原因，
    也是我早期判断"不需要 gather q"的错误所在。）

    正确做法：每个 rank 都拿到全部 head 的 q，对自己那份 KV 分片算出
    `[T, H_total, D]` 的 partial，此时 `[t, h]` 在所有 rank 上才指同一个 head
    ⇒ 逐元素归约才成立。
    顺带一个好消息：每 rank 的注意力计算量 = `H_total × L/dcp` 与 DCP=1 的
    `H_local × L` 相同 ⇒ **总算力不变**，只是换了分布。

    ★★ `token_mask`（`[T,1,1]` bool）也是**必须**的，不能靠 LSE 自己表达"我没有键"。
    实测（子代理 `dcp_op_unknowns` Q3/Q4）：

    * `seqused_ori_kv=0` 时算子返回 `out=0`、`LSE=0.0` —— 是 **kernel 写死的有限值**，
      不是 `-inf`（用 −1e4/−1e30 sink、毒化显存池、重复跑都验证过）
      ⇒ `isfinite` 之类的过滤**兜不住**；
    * 这个 `LSE=0.0` 在合并式里拿到的权重是 `exp(0 − L_max)`：
      长上下文 `L≈5` 时每个空 rank 贡献 +0.7%，短序列 `L≈0` 时可达 1
      （8 个 rank 里 7 个空 ⇒ 分母虚增约 8 倍）。

    所以必须由调用方**按本 rank 的实际可见键数**显式给掩码。
    """
    # =====================================================================
    # ★★★ [V41-MERGEDUMP 2026-09-30 22:30] **合并前的 LSE 轻量 dump**。
    #
    # 层间二分已证明：layer 2 的输入在"好/坏"两次请求之间**逐位相同**，
    # 而 layer 3 的 q 开始分叉（rank0 差 0.6875）⇒ 分叉发生在 **layer 2 的输出**。
    # 本 dump 只落每 rank 的 `lse`（[T,H,1] fp32，每层仅 ~230 KB）与 `ori_lse`，
    # 用来判定分叉是在 (a) lse/算子本身，还是 (b) 跨 rank 合并（all_reduce）。
    # 开关：`mergedump=1`；层/rank/后缀复用 dumplayer / dumprank / dumpsuffix。
    # =====================================================================
    if _perf_flags().get("mergedump") == "1" and not _is_capturing():
        try:
            _mdir = _perf_flags().get("dumpdir")
            _ml = {int(v) for v in _perf_flags().get("dumplayer", "").replace(" ", "").split(",")
                   if v.strip().lstrip("-").isdigit()}
            _mr = {int(v) for v in _perf_flags().get("dumprank", "").replace(" ", "").split(",")
                   if v.strip().isdigit()}
            _msfx = _perf_flags().get("dumpsuffix", "")
            if _mdir and (not _ml or int(layer_idx) in _ml) and (not _mr or int(_v41_dcp_rank()) in _mr):
                import os as _osmd
                _osmd.makedirs(_mdir, exist_ok=True)
                # ★ [V41-MERGEDUMP-2] 除 lse/ori_lse 外，再落 **partial 输出与 ori_out 的指纹**。
                #   为什么：`_fold_ori_locally` 会做 `scaled -= dcp*_onum`，其中
                #   `_onum = ori_out * _keep`。若 `ori_out` 在两次请求间不同（而 lse 相同），
                #   合并结果就会不同 —— 这条此前**没有被对拍过**。
                def _fp(t):
                    if t is None:
                        return None
                    tf = t.detach().to(torch.float32)
                    return {
                        "sum": float(tf.sum()),
                        "absmax": float(tf.abs().max()),
                        "sample": tf.reshape(-1)[::max(1, tf.numel() // 64)][:64].clone(),
                    }
                torch.save(
                    {
                        "lse": lse.detach().to(torch.float32).cpu(),
                        "ori_lse": (ori_lse.detach().to(torch.float32).cpu()
                                    if ori_lse is not None else None),
                        "head_slice": head_slice,
                        "out_fp": _fp(output),
                        "ori_out_fp": _fp(ori_out),
                    },
                    _osmd.path.join(_mdir, "merge_l%d_r%d%s.pt" % (
                        int(layer_idx), int(_v41_dcp_rank()), ("_" + _msfx) if _msfx else "")),
                )
        except Exception as _e:  # noqa: BLE001
            print("[V41-MERGEDUMP] 失败：%r" % (_e,), flush=True)

    group = _v41_dcp_group()
    dcp_size = group.world_size
    if dcp_size <= 1:
        # ★ 保留调用方原来那次 `.contiguous()`：DCP1 时 `output` 直接来自算子，
        #   不保证连续；旧代码在合并之后统一做了
        #   `output[:, 0:H_local, :].contiguous()`，这里等价保留，
        #   保证「DCP1 路径与本改动前逐位一致」（只影响 DCP1，不影响 DCP8 的优化）。
        return output.contiguous()
    # =====================================================================
    # ★★★★★★ [V41-SKIPMERGE 2026-09-30 12:10] **等价 DCP1 的控制臂**。
    #
    # `skip_merge=1`（文件驱动）⇒ 直接返回本 rank 自己的 partial 输出，**完全跳过
    # 跨 rank 合并**。这正是 DCP1 的语义（DCP1 分支就是 `return output.contiguous()`）。
    # 目的：把"合并算错"和"本 rank 的注意力/缓存本身就错"一刀切开。
    #   · 若 skip_merge 在 T=12/16 上给出**正确答案** ⇒ 注意力链路没问题，错在合并；
    #   · 若仍然错 ⇒ rank0 自己的 SMLA 输出（`O_0`）就是错的，合并是背锅的。
    # 与 `rank0_pure` 的区别：后者仍要经过 merge 的乘除（依赖 `wsum`），
    # 本开关连 `wsum`/`scaled` 都不碰 ⇒ 不受任何归约/读数问题影响。
    # `skip_gather=1` 同时打开时，`output` 只有本 rank 的 head ⇒ 整份返回。
    # =====================================================================
    if _perf_flags().get("skip_merge") == "1":
        if head_slice is None:
            return output.contiguous()
        _sh0, _sh1 = head_slice
        if int(output.shape[1]) <= int(_sh1):
            return output.contiguous()
        return output[:, int(_sh0) : int(_sh1), :].contiguous()
    # LSE 需要 float32 且布局一致；算子返回 TND `(N2,T1,G)` ⇒ 转成 `[T, H, 1]`。
    lse = lse.to(torch.float32)
    # ★ [V41-PERF] `(1,T,H)` 的内存顺序本来就是 `(T,H,1)` ⇒ 用 `reshape` 拿视图，
    #   省掉旧 `permute(1,2,0).contiguous()` 的一次 T×H fp32 拷贝（图内一个节点）。
    if lse.ndim == 3 and lse.shape[0] == 1:
        lse = lse.reshape(lse.shape[1], lse.shape[2], 1)
    elif lse.ndim == 2:
        lse = lse.unsqueeze(-1)
    else:
        lse = lse.reshape(lse.shape[1], -1, 1)

    # =====================================================================
    # ★★★ [V41-PERF 2026-09-29 · 关键] 归一化参考点：**零通信**
    #
    # 旧实现先 `all_gather(lse)` 再取跨 rank 最大值当参考点。2-chip 图级 knockout
    # 实测这次 all_gather + amax 占 **46.1 µs/层（24%）**，是三个 collective 之一。
    #
    # 但它**根本没有必要**：合并式
    #     O = Σ_r exp(L_r − c)·O_r / Σ_r exp(L_r − c)
    # 对**任意共享常数 c** 恒等（分子分母同乘 exp(−c) 约掉）。旧实现取跨 rank max
    # 只是为了数值稳定（保证指数 ≤ 0 不上溢），不是数学需要。
    #
    # 路线 B 下有一个**天然共享、零通信**的参考点：**纯 ori 的 LSE**。
    #   · 各 rank 完全相同 —— 第二次调用用的是复制态 SWA cache 的同一窗口、
    #     同一 `sinks`、同一 `cmp_sparse_indices=-1`；
    #   · 且 `L_r = log(A + Z_r + S_r) ≥ log A = L_ori` 恒成立
    #     （A、Z、S 都是非负的加权和）。
    #
    # 于是 `w_r = exp(L_r − L_ori) = (A + Z_r + S_r)/A ≥ 1`，`_ow ≡ 1`。
    # 代入 `_keep = 1 − 1/dcp` 后：
    #   Σ_r w_r·O_r = dcp·O_ori + (Σ_r Z_r·O_cmp + S_0·O_sink)/A
    #   scaled = Σ_r w_r·O_r − _keep·dcp·O_ori = O_ori + (Σ Z·O_cmp + S_0·O_sink)/A
    #   wsum   = Σ_r w_r − _keep·dcp          = 1 + (Σ Z + S_0)/A
    #   ⇒ 比值 = (A·O_ori + Σ Z·O_cmp + S_0·O_sink)/(A + Σ Z + S_0)  ∎ 与全局精确一致
    #
    # 代价与防护：`w_r ≥ 1` 不再有「≤ 1」的结构保证，理论上若 `Z_r/A > 3.4e38`
    # 会溢出（fp32 上界）。物理上 attention score 量级有界、ori 窗口恒有键，
    # 实际 `w_r` 是 O(1)~O(10)；仍加 `clamp(max=60)`（e^60≈1.1e26）作硬保护，
    # 并在**非 capture** 时打印触发诊断（见下），保证"实际生效"可观测。
    # =====================================================================
    _use_ori_ref = ori_lse is not None and ori_out is not None
    # ★ [V41-MERGE-KERNEL] 必须在 `if _use_ori_ref:` 之前初始化（该分支不执行时会未定义）
    _merge_kd = False
    if _use_ori_ref:
        _ori_lse_f32 = ori_lse.to(torch.float32)
        # =============================================================
        # ★★★★★★ [V41-NANPOS 2026-09-30 16:00] **NaN 出现在哪些 token/head**。
        #
        # 已排除（全部实测）：块表错、索引越界、KV 内容坏
        # （`[V41-KVDATA]` 在 850/880 上 `nonfinite=0 / zero_rows=0`）。
        # 但 T≥880 时 `lse` 有 11.6% 元素非有限、`scaled` 为 NaN。
        # 本探针给出**按 token 的分布**：若 NaN 集中在靠后的 token（或被某个
        # 固定 token 区间覆盖），说明与 query 位置有关；若随机散布，说明是
        # 算子内部累加/规约问题。同时打印 `ori_lse` / `ori_out` 的有限性，
        # 把"NaN 从 ori 侧进来"这条也一起判掉。
        # =============================================================
        if (
            _dcp_diag_on("nanpos", "V41_DCP_NANPOS")
            and not _is_capturing()
            and diag_seq_lens is not None
            and int(diag_seq_lens.max()) > 1
        ):
            try:
                _l = lse.to(torch.float32).cpu()
                _ol = _ori_lse_f32.to(torch.float32).cpu()
                _oc = ori_out.to(torch.float32).cpu()
                _fin = torch.isfinite(_l)
                _bad_rows = (~_fin).any(dim=tuple(range(1, _fin.ndim)))
                _nbad = int(_bad_rows.sum())
                _first = int(_bad_rows.to(torch.int64).argmax()) if _nbad else -1
                _last = int(_fin.shape[0] - 1 - _bad_rows.to(torch.int64).flip(0).argmax()) if _nbad else -1
                print(
                    "[V41-NANPOS] rank=%d layer=%d T=%d lse_bad_rows=%d/%d first=%d last=%d "
                    "per_row_uniq=%s | ori_lse_nonfinite=%d ori_out_nonfinite=%d ori_out_absmax=%.6g"
                    % (_v41_dcp_rank(), int(layer_idx), int(_l.shape[0]),
                       _nbad, int(_l.shape[0]), _first, _last,
                       sorted(set((~_fin).sum(dim=tuple(range(1, _fin.ndim))).tolist()))[:6],
                       int((~torch.isfinite(_ol)).sum()),
                       int((~torch.isfinite(_oc)).sum()),
                       float(_oc.abs().max()) if _oc.numel() else -1.0),
                    flush=True,
                )
            except Exception as _e:  # noqa: BLE001
                print("[V41-NANPOS] rank=%d 失败：%r" % (_v41_dcp_rank(), _e), flush=True)
        _delta = lse - _ori_lse_f32
        # ★★★★★★ [V41-MERGE-KERNEL 2026-10-01 14:20] **AscendC 融合路径的早退判据**。
        #   满足时：`weights` / `scaled` / `_onum` 全部不必算（kernel 内部算），
        #   由函数末尾的 `if _merge_kd:` 直接跑 2 个 kernel 并返回。
        #   条件与下面的 `_fold_ori_locally` 完全一致（fold + 有 head_slice）。
        _merge_kd = (
            _v41_merge_kernel_on()
            and ori_lse is not None
            and ori_out is not None
            and head_slice is not None
        )
        # 空 rank（lse=-inf）⇒ exp(-inf)=0；`-inf − (-inf)` 出 NaN ⇒ nan_to_num 归 0。
        weights = (
            None if _merge_kd
            else torch.nan_to_num(torch.exp(_delta.clamp(max=60.0)))
        )
        # ★ 诊断（非 capture 才跑，每 rank 最多 2 次）：验证「ori_lse 各 rank 逐位相同」
        #   这个零通信参考点的**唯一前提**。判据：8 个 rank 打印的 mean/max 应一致。
        if not _is_capturing() and _ORI_REF_DIAG["n"] < 2 and _delta.numel():
            _ORI_REF_DIAG["n"] += 1
            print(
                "[V41-DCP-PERF] rank=%d ori_ref diag: T=%d H=%d "
                "ori_lse_mean=%.6f ori_lse_max=%.6f ori_lse_min=%.6f "
                "delta_max=%.3f delta_min=%.3f wsum_local=%.4f"
                % (
                    _v41_dcp_rank(),
                    int(ori_lse.shape[0]),
                    int(ori_lse.shape[1]),
                    float(ori_lse.mean()),
                    float(ori_lse.max()),
                    float(ori_lse.min()),
                    float(_delta.max()),
                    float(_delta.min()),
                    float(weights.sum()),
                ),
                flush=True,
            )
        if not _is_capturing():
            _d = float(_delta.max()) if _delta.numel() else 0.0
            if _d > 60.0:
                print(
                    "[V41-DCP-PERF][WARN] ori 参考点饱和：max(L_r - L_ori)=%.2f > 60 "
                    "⇒ 该 (t,h) 的权重被截断（结果仍有限，但不再精确）" % _d,
                    flush=True,
                )
    else:
        # 无 ori 参考（`skip_2nd=1` 消融臂 / 非 DCP 路径）：保留旧的跨 rank max。
        gathered = torch.empty(
            (dcp_size, *lse.shape), dtype=torch.float32, device=lse.device
        )
        torch.distributed.all_gather_into_tensor(gathered, lse, group=group.device_group)
        lse_max = gathered.amax(dim=0)
        # 去掉 `clamp(min=-80.0)`：`lse ≤ lse_max` 恒成立、`exp` 不上溢；
        # 全 -inf 行出 NaN ⇒ `nan_to_num` 归 0（正确语义 = 权重 0）。
        weights = torch.nan_to_num(torch.exp(lse - lse_max))
    # ★ 路线 B 生效时**不能**再用 token_mask：此时每个 rank 的 LSE 都是
    #   真实的配分函数（`A + Z_r`，A = e^{L_ori}），掩码会把本该参与分母的
    #   `A` 也抹掉，导致下面扣除 `(dcp−1)·A` 时**多扣**。
    if token_mask is not None and ori_lse is None:
        weights = weights * token_mask.to(torch.float32)
    # =====================================================================
    # [V41-DCP-DIAG] 两个判别性开关（默认关闭，**仅诊断用**，不得进生产）。
    #
    # `V41_DCP_MERGE_RANK0_ONLY=1`：把 rank≠0 的权重强制清零
    #   ⇒ 输出退化成"仅 rank 0 的 partial（ori ∪ cmp_0）"。
    #   判别：若此时长上下文**变对** ⇒ 问题在 rank 1-7 的贡献（合并/cmp 分片）；
    #         若仍错 ⇒ 问题在 rank 0 自己的注意力读取。
    #
    # `V41_DCP_LSE_DIAG=1`：每个 rank 打印自己上报的 LSE 统计与输出幅度。
    #   判别：rank>0 的 LSE 若显著大于 rank 0，就会按错误权重压掉正确贡献。
    #   `.item()` 会同步流，所以只在**非 capture** 时执行，且只打一次。
    # =====================================================================
    import os as _os_m

    # ★ [V41-PERF] 同时接受**文件驱动**开关 `rank0_only=1`（`/tmp/v41_perf_flags`）：
    #   env 只能在起服时设，而这条实验要反复切换。文件开关对 **prefill** 有效
    #   （prefill 是 eager、Python 会执行），而首 token 正是由 prefill 的注意力决定的
    #   ⇒ 无需重启就能做 A/B。decode 阶段是整图捕获，文件开关在那里无效（已知）。
    _rank0_only = (
        _os_m.environ.get("V41_DCP_MERGE_RANK0_ONLY") == "1"
        or _perf_flags().get("rank0_only") == "1"
    )
    # ★★ [V41-DIAG] `rank0_pure=1`：见下面 `_keep` 处的推导。作用是让
    #   `scaled/wsum` 精确退化成 `O_0`（只在"所有键都在 rank 0"时是正确的），
    #   用来区分"合并算错"与"rank0 的 partial 本身错"。
    _rank0_pure = _perf_flags().get("rank0_pure") == "1"
    if (_rank0_only or _rank0_pure) and _v41_dcp_rank() != 0:
        weights = torch.zeros_like(weights)
    # ★ 必须排除 warmup：warmup 用的是 dummy 输入（seq_lens 全 1），
    #   它的 LSE/输出本来就接近 0，会给出严重误导的读数
    #   （实测：第一次 LSE 诊断落在 T=16 的 warmup 上，8 个 rank 里 7 个报 0）。
    #   只对**真实长请求**打印，且每 rank 最多 4 次（覆盖 prefill + decode 前几步）。
    if (
        _os_m.environ.get("V41_DCP_LSE_DIAG") == "1"
        and not _is_capturing()
        and _lse_diag_count() < 4
    ):
        # ★ 用**显式传入**的张量：早先版本直接引用 `seq_lens`，
        #   而它在 `_v41_dcp_merge_attention` 里并不在作用域内 ⇒ NameError
        #   被下面的宽 `try/except` 吞掉 ⇒ 诊断静默不打印（0 行），
        #   我因此白跑了一轮起服。**诊断自己不能用兜底 except 掩盖失败。**
        _n = -1
        if diag_seq_lens is not None and diag_seq_lens.numel():
            _n = int(diag_seq_lens.max())
        if _n > 129:
            _bump_lse_diag()
            _tm = 0 if token_mask is None else int(token_mask.sum())
            _cl = -1
            if diag_cmp_lens is not None and diag_cmp_lens.numel():
                _cl = int(diag_cmp_lens.max())
            print(
                "[V41-LSE] rank=%d T=%d H=%d Lmax=%d cmpLmax=%d "
                "lse_mean=%.4f lse_min=%.4f lse_max=%.4f "
                "out_absmax=%.6f wsum=%.4f tmask_n=%d"
                % (
                    _v41_dcp_rank(),
                    int(layer_idx),
                    int(lse.shape[0]),
                    int(lse.shape[1]),
                    _n,
                    _cl,
                    float(lse.mean()),
                    float(lse.min()),
                    float(lse.max()),
                    float(output.abs().max()),
                    float(weights.sum()),
                    _tm,
                ),
                flush=True,
            )
    # =====================================================================
    # [V41-DCP-WDIAG] 权重级诊断：只在 `V41_DCP_WEIGHT_DIAG=1` 时打印，非 capture。
    #
    # 为什么要它：DCP8 与 DCP1 的输出在 **分布层面** 差 0.4~0.7 nats
    # （`lp_probe.py`：首 token |Δlogprob| 0.375/0.386/0.675/0.409，
    #  短 prompt 甚至首 token 变成 `<｜begin▁of▁sentence｜>`、Δ=11.77）。
    # 这远超 bf16 舍入（应为 1e-3 量级）⇒ 是**真实缺陷**，必须定位到 rank。
    #
    # 本诊断打印每个 rank 的：
    #   · `lse`（第一次调用的配分函数）min/max —— 若某 rank 返回 kernel 写死的
    #     `0.0`（无键时的有限值，见 §2.2 的警告），就会在这里露出来；
    #   · `ori_lse`（第二次纯 ori 调用的 LSE）—— 它是**零通信参考点**，前提是
    #     8 个 rank **逐位相同**；不等就说明复制的滑窗在各 rank 上内容不一致；
    #   · `w = exp(lse − ori_lse)` 的 min/max/和、以及有限权重的个数
    #     —— 正确时 rank r 的权重应 ≈ `D_r/A ≥ 1`；若出现 ≈0 或 NaN 就错了。
    # =====================================================================
    # ★ [V41-DIAG] `V41_DCP_WDIAG_LAYER=<n>` ⇒ **只打这一层**。
    #   为什么需要：WDIAG 没层号时，连续行来自**不同层**（每请求 40 行），
    #   无法区分"不同层天然不同"与"同层逐次不同"。锁定一层后，
    #   连续行就是**连续请求**，可直接比对 `lse` 与 `ori` 是否稳定。
    # ★ 层过滤**文件优先**（`/tmp/v41_perf_flags` 里写 `wdiag_layer=0`）⇒ 免重启扫层。
    _wdiag_layer = int(
        _perf_flags().get("wdiag_layer")
        or __import__("os").environ.get("V41_DCP_WDIAG_LAYER", "-1")
    )
    if (
        _dcp_diag_on("wdiag", "V41_DCP_WEIGHT_DIAG")
        and (_wdiag_layer < 0 or int(layer_idx) == _wdiag_layer)
        and not _is_capturing()
        and _WDIAG["n"] < _WDIAG_LIMIT
        # ★ **必须排除 warmup**：`profile_run` 的 dummy 输入 `seq_lens` 全是 1，
        #   40 层 × 若干 warmup 步就会把配额吃光，真实 prefill 一行都打不出来
        #   （第一次跑就踩了：1600 行全是 `seq=[1, 1, 1, 1]`）。
        and diag_seq_lens is not None
        and int(diag_seq_lens.max()) > 1
        and _dcp_diag_t_ok(int(diag_seq_lens.max()))
    ):
        _WDIAG["n"] += 1
        _lr = _v41_dcp_rank()
        try:
            # ★★★ [V41-SNAP 2026-09-30] **单次 `.cpu()` 快照 + CPU 侧计算**。
            #
            # 为什么必须这样：`float(device_tensor)` 在 torch_npu 上走
            # `LocalScalarDenseNpu` 的**独立 copy stream**
            # （报错原文：`AclrtSynchronizeStreamWithTimeout(copy_stream)`）⇒
            # 对同一张量的**多次**标量读取会给**互相矛盾**的结果
            # （实测 MDIAG2：同一行 `blk_sums` 全 0 而 `wcol_sum=1024`、`min=max=1`，
            #   7904 条里 2875 条不自洽 ⇒ 36% 是伪影）。
            # ⇒ 现在只做**一次** `.cpu()`，其余全在 CPU 上算，物理上不可能撕裂。
            _l_cpu = lse.to(torch.float32).cpu()
            _lmin = float(_l_cpu.min()); _lmax = float(_l_cpu.max()); _lmean = float(_l_cpu.mean())
            _fin = int(torch.isfinite(_l_cpu).sum())
            _nzer = int((_l_cpu.abs() < 1e-9).sum())
            _tot = int(_l_cpu.numel())
            if ori_lse is not None:
                _o_cpu = ori_lse.to(torch.float32).cpu()
                _ostat = "ori[min=%.6f max=%.6f mean=%.6f]" % (
                    float(_o_cpu.min()), float(_o_cpu.max()), float(_o_cpu.mean())
                )
            else:
                _ostat = "ori=<None>"
            # ★ 用 `_use_ori_ref`（在 :251 定义）而不是 `_ori_active`（在 :455 才定义，
            #   本诊断块在它之前）—— 踩过一次 NameError。
            if _use_ori_ref and weights is not None:
                _w_cpu = weights.to(torch.float32).cpu()      # ★ 同上：单次快照
                _wstat = "w[min=%.6g max=%.6g sum=%.6g n>0=%d]" % (
                    float(_w_cpu.min()), float(_w_cpu.max()), float(_w_cpu.sum()),
                    int((_w_cpu > 0).sum()),
                )
            else:
                _wstat = "w=<n/a>"
            # ★ `seq_lens` / `cmp_seq_lens` **不在本函数作用域** —— 它们是通过
            #   `diag_seq_lens` / `diag_cmp_lens` 传进来的（踩过：直接写 `seq_lens`
            #   会 NameError，被宽 except 吞掉就变成"诊断静默不打印"）。
            _sl = (
                diag_seq_lens.detach().to(torch.int64)[: min(4, int(diag_seq_lens.numel()))].cpu().tolist()
                if diag_seq_lens is not None else []
            )
            _cl = (
                diag_cmp_lens.detach().to(torch.int64)[: min(4, int(diag_cmp_lens.numel()))].cpu().tolist()
                if diag_cmp_lens is not None else []
            )
            print(
                "[V41-WDIAG] rank=%d layer=%d T=%d lse[min=%.6f max=%.6f mean=%.6f finite=%d/%d zero=%d] "
                "%s %s seq=%s cmp=%s tmask=%s"
                % (
                    _lr, int(layer_idx), int(_l_cpu.shape[0]), _lmin, _lmax, _lmean,
                    _fin, _tot, _nzer,
                    _ostat, _wstat, _sl, _cl,
                    "None" if token_mask is None else "on",
                ),
                flush=True,
            )
        except Exception as _e:  # noqa: BLE001
            # 诊断自己不许用宽 except 吞掉失败（踩过）—— 但要标出是哪一步失败
            print("[V41-WDIAG] rank=%d 诊断失败（这是诊断自身的问题，不是引擎）：%r"
                  % (_v41_dcp_rank(), _e), flush=True)

    # =====================================================================
    # ★★ 性能：把 **4 次 all_reduce 打包成 1 次**。
    #
    # 原实现每层有 5 个集合通信（1 all_gather + 4 all_reduce）：
    #   scaled / wsum / _n_all / _w_all 各一次。
    # 实测 DCP8 比 DCP1 慢 13.4 ms/step、40 层 ⇒ **0.33 ms/层**，
    # 与「5 个集合通信 × ~50-70 µs + 两次 SMLA 调用」的估算吻合
    # ⇒ 瓶颈是**集合通信的次数（延迟）**，不是带宽（T=1 时单次才 ~128 KB）。
    #
    # 打包做法：把 4 个张量按最后一维拼成一个 `[T, H, 2D+2]` fp32，一次 all_reduce。
    # 数学与逐个 all_reduce **完全等价**（all_reduce 是逐元素的），
    # 且拼接后最后一维是 `2D+2`（D=512 ⇒ 1026），16 字节对齐要求需保证 D 为偶数
    # （D=512 满足）。
    # =====================================================================
    # ★★★ [V41-PACKDIRECT-ABORT 2026-10-01 05:00] **回退**"把 `output*w` 直写
    #   常驻缓冲"的优化（省 Cast+Mul+ViewCopy）。
    #   实测后果：`T=904` 针**立刻回归**（输出变成 `'特'`/`'7'`，正确值是 `Q7`，
    #   连续 2 次均错），而同一 overlay 的上一版（opt3）是 6/6。
    #   怀疑点（未逐一证实）：`torch.mul(bf16, fp32, out=<非连续 fp32 视图>)`
    #   的类型提升/写回在该 NPU 上不等价，或 `scaled` 与 `output` 的别名在
    #   后续被 `_slice_early` 路径改动。**没有再深挖**：这条只值 ~0.2 ms/step，
    #   而正确性是硬门槛 ⇒ 直接恢复原三步写法。
    # ★ [V41-PROMO] `bf16 * fp32` 会**自动提升到 fp32**，与 `output.to(fp32)*weights`
    #   **逐位一致**（单卡验证：`torch.equal=True, max|d|=0`），但少一个 `Cast` 节点。
    #   单卡微基准：2 节点 27.31 µs → 1 节点 10.70 µs（×38 = **−0.63 ms/step**）。
    # ★ [V41-MERGE-KERNEL] 融合路径下 `scaled` 由 kernel 直接写进 pack ⇒ 不必算
    scaled = None if _merge_kd else output * weights
    dcp = group.world_size
    # ★★ [V41-PERF 2026-09-29] 把 `(1 − 1/dcp)` **折进 ori 项**，把
    #   归约后的 4 个逐元素算子（`−_n_all + _n_all/dcp`、`−_w_all + _w_all/dcp`）
    #   压成 2 个（`− Σ_onum' `、`− Σ_ow'`）。
    #
    # 代数：`all_reduce` 是**线性**的，所以
    #     Σ_r (_onum_r · k) = k · Σ_r _onum_r = k · _n_all
    # 把 `k = 1 − 1/dcp`（Python float，dcp 是 Python int）先乘到 **per-rank 的
    # ori 权重**上，归约出来的就是 `k·_n_all`，后处理只剩一次减法。
    # `_ow` 只有 `[T,H,1]`，把 k 乘在它上面的代价可忽略；省下的是两次作用在
    # `[T,64,512]` fp32 上的重算子。
    #
    # 为什么值得：2-chip 图级实测 —— DCP2−DCP1 = +286 µs/层，其中
    # **79% 是合并链的串行关键路径**（不是带宽、不是算力）；每砍掉 1 个串行
    # 节点 ≈ 13 µs/层 ≈ 0.5 ms/step（40 层）。
    _keep = 1.0 - 1.0 / dcp
    # =====================================================================
    # ★★ [V41-DIAG 2026-09-30] `rank0_pure=1`（文件驱动，prefill 是 eager ⇒ 生效）
    # 判别"合并算错"还是"rank0 的 partial 本身错"。
    #
    # 原理：当**所有键都在 rank 0** 时（`interleave=32` 且 T≤32 ⇒ owner 恒为 0），
    # 正确的全局注意力恰好等于 rank 0 的 partial `O_0`。此时
    #     scaled = Σ_r w_r·O_r ，wsum = Σ_r w_r − dcp·keep
    # 若令 `w_r = 0 (r>0)` 且 `keep = 0`，则
    #     scaled = w_0·O_0，wsum = w_0 ⇒ 结果 = **O_0**（精确，无跨 rank 算术）
    # ⇒ 若这样能得到正确答案 ⇒ `O_0` 没问题、错在合并算术；
    #   若仍然错 ⇒ `O_0`（rank 0 自己的 SMLA 输出）就是错的。
    # ★ 只在 T ≤ 32 时才是"正确"的（更长时键会落到别的 rank）⇒ 仅用于定位。
    # =====================================================================
    # ★ `_rank0_pure` 已在上面（与 `_rank0_only` 同处）解析；这里只改 `keep`。
    if _rank0_pure:
        _keep = 0.0
    _ori_active = ori_lse is not None and ori_out is not None
    # =====================================================================
    # ★★★ [V41-PERF 2026-09-29] ori 参考点路径下，**`_onum`/`_ow` 不需要参与归约**。
    #
    # 第二次调用是「纯 ori」：全部 rank 用同一份复制态 SWA cache、同一窗口、
    # 同一参数 ⇒ `ori_out` / `ori_lse` 在各 rank 上相同（诊断行已实测：
    # 8 个 rank 的 `ori_lse_mean/max/min` 全部 `0.000000`）。
    # 而 `_ow ≡ _keep` 本来就是 Python 常量。于是
    #     Σ_r _onum_r = dcp · ori_out · _keep = (dcp − 1) · ori_out
    #     Σ_r _ow_r   = dcp · _keep          = dcp − 1
    # **本地就能算出精确值，不必花一次 all_reduce 运 `[T,H,D]` 的整份 `_onum`。**
    #
    # 效果：pack 从 `[T,H,2D+2]` fp32（T=128 时 33.6 MB）**降到 `[T,H,D+1]`
    # （16.8 MB，通信量减半）**，同时少一个 `cat` 输入与两次减法。
    # 打包 all_reduce 是当前最大单项（图内边际 57.4 µs/层）。
    #
    # 前提失效时的保护：若某次 `ori_out` 在 rank 间不同（kernel 归约顺序差异
    # 最多 1 ulp），引入的误差是 fp32 舍入级；端到端由长上下文多选针回归把关。
    # =====================================================================
    _fold_ori_locally = _ori_active and _use_ori_ref
    # ★★ [V41-PERF 2026-09-29] **提前切片**：本 rank 的 `o_proj` 只吃自己那 8 个 head
    #   （TP 连续切分），而归约之后所有 rank 手里都有全部 64 个 head。
    #   旧实现在**全 64 个 head** 上做 `output.to(fp32) * weights`、`− dcp·_onum`、
    #   `wsum − const`、`div`、`to(bf16)` 这一整条链，最后才切到 8 个 head。
    #   但**切片只影响后处理**：all_reduce 是跨 rank 逐元素求和，各 rank 贡献的
    #   仍是自己的全 head 分片，必须保持 `[T,64,D]` 才能对上位置。
    #   ⇒ 归约**前**保持全 head，归约**后**立刻切到本 rank 的 8 个 head 再做减法；
    #     并且 `_onum`/`_ow` 这两个"ori 本地折叠"项从一开始就只在本 rank 的
    #     8 个 head 上演算（它们本来就不参与归约）。
    #   2-chip 图级累积式分解实测：后处理 **42.2 µs/层**，是当前最大的非通信项。
    #   离线核对：先切后减 vs 先减后切 == 逐位相同（elementwise，maxdiff=0）。
    _slice_early = _fold_ori_locally and head_slice is not None
    _hs_applied = None          # ★ [V41-DENCHK] 记录实际用过的 head 区间
    if _ori_active:
        if _use_ori_ref:
            # ★ `_ow` 只在**非折叠**路径（进 pack）用到；折叠路径里 `Σ_r _ow_r`
            #   是常量 `dcp·_keep`，本地直接算 ⇒ 这里不再白建一个 `[T,H,1]` 张量。
            _ow = None if _fold_ori_locally else torch.full_like(weights, _keep)
            _oi = (
                ori_out[:, head_slice[0] : head_slice[1], :] if _slice_early else ori_out
            )
            # ★ [V41-ONUMFOLD 2026-10-01 04:25] 把 `dcp * _onum` 的乘法折进来：
            #   下游是 `scaled - _onum`，而 `_keep` 是 Python 常量
            #   ⇒ `_onum = _oi*(_keep*dcp)` 省掉每层一次 `Muls`（38 次/step）。
            #
            # ★★★★★★ [V41-SUBALPHA-ABORT 2026-10-01 07:40] **已回退 `torch.sub(...,
            #   alpha=)` 融合**。离线微基准（`~/tmp/subalpha.py`）里它与三步写法
            #   **逐位一致**且快 3.1×，但在真实路径上**立刻破坏正确性**：
            #   `17 × 23 等于多少？只回答数字。` 连续 **6/6** 输出乱码
            #   （`# 标题：老婆背叛后…`），而同 overlay 上一版稳定输出 `391`。
            #   真实路径与微基准的差别：两个输入都是**strided 切片**
            #   （`_pack[..., :D][:, h0:h1]` 与 `ori_out[:, h0:h1]`）。
            #   这是同一类坑的**第三次**（另两次：`PACKDIRECT` 的 `torch.mul(out=)`
            #   与非连续视图、`Sanitize` 的原地改写）——
            #   ⇒ **本平台上"把多个逐元素算子融合/改写"的写法一律不可信，
            #     无论离线微基准是否逐位一致。** 性能收益再大也不接受。
            # ★ [V41-PROMO] 同上：1 维 fp32 标量张量把 `_oi` 提升到 fp32，
            #   一次算子完成「转 fp32 + 乘 `_keep*dcp`」，与
            #   `_oi.to(fp32) * (_keep*dcp)` **逐位一致**（单卡验证）。
            #   微基准：30.72 µs → 11.35 µs（×38 = **−0.74 ms/step**）。
            # ★ [V41-MERGE-KERNEL] 融合路径下 `_onum` 也由 kernel 内部算
            _onum = None if _merge_kd else _oi * _v41_scalar_t(_keep * dcp, _oi)
        else:
            # `skip_2nd=1` 消融臂：仍用跨 rank max 作参考点（那条路径才有 lse_max）。
            _ow = torch.nan_to_num(torch.exp(ori_lse.to(torch.float32) - lse_max)) * _keep
            _onum = ori_out.to(torch.float32) * _ow
    else:
        _ow = torch.zeros_like(weights)
        _onum = torch.zeros_like(scaled)
    # =====================================================================
    # ★★★★★★ [V41-MERGE-KERNEL 2026-10-01] **AscendC 融合路径**。
    #
    # 为什么：真实 8 卡 profiler 显示 DCP8 比 DCP1 多 1213 图节点/step，其中
    # **573 个**来自本函数的逐元素前/后处理（每层 ~15 个小算子 × 38 层）。
    # 实测每个图节点值 ~3 µs 墙钟 ⇒ 这 573 个节点约值 1.72 ms/step。
    # 本路径把它压成 **2 个 AscendC kernel**（all_reduce 夹在中间，不能合成一个）。
    #
    # 单卡实测（`ascendc/merge/bench_merge.py`，chip6）：
    #   T=1（生产 decode）pre 7.08 µs + post 6.61 µs = 13.68 µs/层 ⇒ 38 层 **0.520 ms/step**
    #   与 Python 参考在 T=1–16 **逐位一致**；T=20+ 差 1 个 bf16 ULP
    #   （本 CANN 的 fp32 无可用除法指令，只能用 `Muls(num, 1/den)`）。
    #
    # 图捕获安全：ctypes 调用发生在**捕获期**，replay 由图回放 ⇒ 无 host 开销；
    # 所有张量（tiling / pack / out）都是**常驻缓存**，地址跨 replay 不变。
    # =====================================================================
    if _merge_kd:
        from vllm_ascend.attention import v41_merge_kernel as _mk

        _T = int(output.shape[0])
        _H = int(output.shape[1])
        _h0, _h1 = int(head_slice[0]), int(head_slice[1])
        _Hout = _h1 - _h0
        _pk = _mk.pack_buffer(_T, _H, output.device)
        _mk.merge_pre(output, lse, _ori_lse_f32, _pk)
        if _perf_flags().get("det_reduce") == "1" or _DCP_DET_REDUCE:
            _g = torch.empty((dcp, *_pk.shape), dtype=_pk.dtype, device=_pk.device)
            torch.distributed.all_gather_into_tensor(_g, _pk, group=group.device_group)
            _acc = _g[0]
            for _r in range(1, dcp):
                _acc = _acc + _g[_r]
            _pk.copy_(_acc)          # 写回常驻缓冲，保持地址不变
        else:
            torch.distributed.all_reduce(_pk, group=group.device_group)
        _outk = _mk.out_buffer(_T, _Hout, output.device)
        _mk.merge_post(_pk, _oi, _outk, _h0, float(_keep * dcp), float(dcp * _keep))
        return _outk
    _mdiag_pre_t = None
    _mdiag_post_t = None
    # =====================================================================
    # ★★★★★★ [V41-PREW 2026-09-30] **打包之前的 `weights` 统计**。
    #
    # 为什么要"之前"：`_pack` 在归约后不再被本函数引用 ⇒ 显存可能已归还分配器
    # ⇒ 之后任何读它的探针都会拿到**别人的数据**（§6ad 实测：同一快照里
    # `rmin=8.05`（≥ 数学下界）而 `sliced_wcol=nan`，自相矛盾）。
    # ⇒ 唯一可靠的探针位置是**张量仍被强引用时**，即这里（`weights` 刚算完、还没 cat）。
    #
    # 用途：把"`Σ_r w_r` 为什么在 T=12/16 塌成 ~0"的两种可能分开：
    #   (a) 归约把权重列清零  ⇒ 这里的 `w` 正常（≥1）、归约后异常；
    #   (b) 打包前 `w` 就是 0 ⇒ 这里就能看到 0/NaN。
    # `w = exp(lse − ori_lse)` 被 `clamp(max=60)` 限住 ⇒ 正常时 `w ∈ [1, 1.1e26]`。
    # =====================================================================
    if (
        _dcp_diag_on("prew", "V41_DCP_PREW")
        and not _is_capturing()
        and diag_seq_lens is not None
        and int(diag_seq_lens.max()) > 1
    ):
        try:
            # 一次 stack + 一次 cpu ⇒ 尽可能少的标量读；且 weights 此时被强引用
            _w3 = torch.stack([
                weights.to(torch.float32).min().reshape(1),
                weights.to(torch.float32).max().reshape(1),
                weights.to(torch.float32).sum().reshape(1),
            ]).cpu()
            print(
                "[V41-PREW] rank=%d layer=%d T=%d w_min=%.6g w_max=%.6g w_sum=%.6g"
                % (_v41_dcp_rank(), int(layer_idx), int(diag_seq_lens.max()),
                   float(_w3[0]), float(_w3[1]), float(_w3[2])),
                flush=True,
            )
        except Exception as _e:  # noqa: BLE001
            print("[V41-PREW] rank=%d 探针失败：%r" % (_v41_dcp_rank(), _e), flush=True)
    if _fold_ori_locally:
        # =====================================================================
        # ★★★★★★ [V41-SANITIZE 2026-09-30] **打包前的 Inf/NaN 清理**。
        #
        # 假设（由实测反推）：`scaled = output × w`，而 `w = exp(lse − ori_lse)`
        # 在坏长度上可达 `1e11` ⇒ `scaled` 可能溢出成 Inf/NaN。
        # 而 **HCCL 的 all_reduce / all_gather 在 buffer 含 Inf/NaN 时行为异常**
        # —— 这解释了"归约后权重列被清零"（`wsum ≈ −7`）：
        # 实测 `reduced_sum − local_sum = 7168 = 7×1024`（确实加了 rank1-7 的 1.0），
        # 但切片那 128 个元素 ≈ 0。
        #
        # 判别开关 `sanitize=1`：cat 之前把 `scaled` 的 Inf/NaN 换成**有限大数/0**，
        # 使 buffer 里不再有非有限值。
        #   · 若 `wsum ≥ 1` 恢复、答案变对 ⇒ 假设成立，这就是修复（至少是第一层修复）；
        #   · 若仍为 −7 ⇒ HCCL 的问题与 Inf/NaN 无关。
        # 同时用**单次 stack + cpu** 统计 Inf/NaN 个数（只读一次，避免 §6o 撕裂）。
        # =====================================================================
        if _dcp_diag_on("sanstat", "V41_DCP_SANSTAT") and not _is_capturing():
            try:
                _sf = scaled.to(torch.float32)  # bf16 时这里会补一次 Cast（仅诊断路径）
                _st = torch.stack([
                    torch.isinf(_sf).sum().reshape(1),
                    torch.isnan(_sf).sum().reshape(1),
                    _sf.abs().max().reshape(1),
                    weights.to(torch.float32).max().reshape(1),
                ]).cpu()
                print(
                    "[V41-SANSTAT] rank=%d layer=%d T=%d scaled_inf=%d scaled_nan=%d "
                    "scaled_absmax=%.6g w_max=%.6g"
                    % (_v41_dcp_rank(), int(layer_idx),
                       int(diag_seq_lens.max()) if diag_seq_lens is not None else -1,
                       int(_st[0]), int(_st[1]), float(_st[2]), float(_st[3])),
                    flush=True,
                )
            except Exception as _e:  # noqa: BLE001
                print("[V41-SANSTAT] rank=%d 失败：%r" % (_v41_dcp_rank(), _e), flush=True)
        if _perf_flags().get("sanitize") == "1":
            _scaled_f = scaled.to(torch.float32)
            # Inf → 有限大数（保留量级信息但不溢出），NaN → 0
            _finfo = torch.finfo(torch.float32)
            _scaled_f = torch.nan_to_num(
                _scaled_f, nan=0.0, posinf=_finfo.max / 4, neginf=-_finfo.max / 4
            )
            scaled = _scaled_f.to(scaled.dtype)
        # ★★★★★★ [V41-DEADCAT 2026-10-01 07:05] **删掉一次死代码 `torch.cat`**。
        #
        # 原来这里有 `_pack = torch.cat([scaled, weights], dim=-1)`，但紧接着的
        # padding 逻辑已被 `_v41_pack_for_reduce`（常驻缓冲）取代，而**下面第 988 行
        # 又无条件重写了 `_pack`** ⇒ 这次 `cat` 的结果**从未被使用**，可它的
        # `ConcatD` 内核**照样每层下发一次**。
        # 线上证据（DCP1 vs DCP8 干净对比）：`ConcatD` **0 → 38 次/step**、
        # **0.241 ms/step**，而 DCP1 里该算子为 0 —— 正是这一行。
        # ⇒ 删除是纯粹的白赚。
        # ★ [V41-MDIAG] 记录**归约前**的本地 weights 和（用于判断"其它 rank 的
        #   贡献到底有没有进 all_reduce"）。实测现象：合并后 `wsum = Σw − dcp·keep`
        #   是**负数**（约 −4），而 WDIAG 显示 7 个 rank 的 `w ≡ 1`、rank0 的
        #   `w` 均值 2.86 ⇒ 若归约正常应得 `Σw ≈ 9.86`、`wsum ≈ +2.86`。
        #   两种可能：(a) 其它 rank 没进归约（Σw ≈ w_0）；(b) 减多了。
        #   这两条诊断能一次分开。
        # ★ 绝不能在这里做 host 同步：本函数位于 **ACL graph capture 区**内，
        #   `float(device_tensor)` / `.sum()` 会触发
        #   `Not_Supported(EE1016): stream is captured`，8 个 worker 全挂
        #   （2026-09-29 踩过两次）。所以只做**纯张量**运算，取值留给下面
        #   已经有 `not _is_capturing()` 门的 MDIAG 块。
        # ★ 用**运行时**判据（文件优先），与下面的打印门保持一致 ——
        #   踩过：`_DCP_MDIAG_ON` 是模块级 env 常量，而打印门是文件驱动，
        #   两者不一致时 `w_local_sum`/`w_postreduce_sum` 会一直打印哨兵 `-1`，
        #   让人以为"没测到"，其实是判据不同步。
        _mdiag_here = _dcp_diag_on("mdiag", "V41_DCP_MERGE_DIAG") and not _is_capturing()
        if _mdiag_here:
            _mdiag_pre_t = weights.to(torch.float32).sum()
        # =================================================================
        # ★★★ [V41-FIX 2026-09-30] **HCCL 的 all_reduce 要求 16 字节对齐**
        #
        # `cat([scaled(D=512), weights(1)], dim=-1)` 的最后一维是 **513**
        # ⇒ `513 × 4 B = 2052 B`，**不是 16 的倍数**（2052 = 16×128 + 4）。
        # 实测后果（MDIAG，真机 T=16）：
        #     `w_local_sum = 2924.75`（归约前，正常）
        #     `w_postreduce_sum = nan`（归约后；另一条是 `0`）
        # ⇒ all_reduce 把 buffer 弄坏了 ⇒ `wsum` 出现**负数**（实测 min≈−5.7）
        # ⇒ `denom = wsum.clamp_min(1e-30)` 把负数压成 `1e-30`
        # ⇒ `out = scaled/1e-30 ≈ 6.6e+30` ⇒ LM head 溢出 ⇒ **logits 全相同**
        # ⇒ softmax 恰好均匀 ⇒ logprob 恰好 `-ln(129280)`（= 短 prompt 硬故障）。
        #
        # 这也解释了"为什么所有历史版本都失败"：旧的 `2D+2 = 1026` 同样不是
        # 4 的倍数（1026×4 = 4104 = 16×256 + 8），只是坏法不同。
        #
        # 修法：把包 **pad 到 4 的倍数**（fp32 下 4 个元素 = 16 字节），
        # 归约后切掉 padding。padding 位置恒为 0，不影响任何被读取的分量。
        # =================================================================
        # ★ 对齐粒度实测：**16 字节（4×fp32）不够** —— pad 到 4 之后
        #   长度扫描仍有 1/3 的失败样本保持"均匀分布"（`nuts`），
        #   而 Ascend 的 HCCL 通常要求 **512 字节** 对齐。
        #   ⇒ pad 到 128 个 fp32（= 512 B）。代价：最后一维 513 → 640（+25%），
        #   但前面实测"collective 在这个区间是**纯延迟**（16 B 与 1 MB 同价）"
        #   ⇒ 这点字节量不影响时延。
        # ★★★★★★ [V41-PACKBUF 2026-10-01 02:40] **不用 `F.pad`，改用常驻缓冲**。
        #
        # 真实 8 卡 profiler（run dcpcap_1001_015103，20 forwards）实测：
        #     `aclnnConstantPadNd_PadV3AiCore_MemSet`  760 次 / 20 fwd = 38/step
        #                                              12.053 ms → **0.603 ms/step**
        #     `PadV3`                                  760 次 → **0.234 ms/step**
        #     `ConcatD`                                760 次 → **0.246 ms/step**
        #   ⇒ 仅"打包"这一步就 **1.08 ms/step**，而输入只有 `[1,64,513]` fp32（131 KB）。
        #   pad 的 15.9 µs/次 是纯粹的固定开销（只为写 128 列的零）。
        #
        # 做法：按 `(T,H,D,dtype,device)` 缓存一个**已经零初始化**的 `[T,H,640]` 缓冲，
        # 每次只写 `[..., :D]` 与 `[..., D:D+1]` 两段；padding 永远保持 0，
        # 既不需要 `F.pad`，也不需要 `cat` 分配。
        #   图安全：缓冲地址跨 replay 不变（这正是指标要求的）；捕获期只分配一次。
        #   语义不变：allreduce 之后只读 `[..., :D+1]`，padding 参与求和但恒为 0。
        _pack = _v41_pack_for_reduce(scaled, weights)
        # ★ [V41-CSEQ] 打序号（非 capture + 真实 prefill）
        if (
            _dcp_diag_on("cseq", "V41_DCP_CSEQ")
            and not _is_capturing()
            and diag_seq_lens is not None
            and int(diag_seq_lens.max()) > 1
        ):
            _DCP_MERGE_SEQ["n"] += 1
            print(
                "[V41-CSEQ] rank=%d layer=%d seq=%d T=%d mode=%s"
                % (_v41_dcp_rank(), int(layer_idx), _DCP_MERGE_SEQ["n"],
                   int(diag_seq_lens.max()), "gather" if _DCP_DET_REDUCE else "reduce"),
                flush=True,
            )
        # ★★★ [V41-DETREDUCE-FLAG 2026-09-30 22:40] 改为**文件开关优先**（`det_reduce=1`），
        #   免重启即可 A/B。动机：层间二分 + 合并前 LSE 对拍证明
        #   "8 个 rank 的 lse 逐位相同、layer 2 输入逐位相同"，而 layer 3 的 q 开始分叉
        #   ⇒ 分叉来自**跨 rank 的 all_reduce**（HCCL 在该形状/T 上求和顺序可变）。
        #   本分支用 `all_gather` + 按 rank 顺序显式求和（数学等价、顺序固定）。
        _rs_applied = False
        if _perf_flags().get("det_reduce") == "1" or _DCP_DET_REDUCE:
            # ★ 定序归约：`all_gather` 拿全部 rank 的包，再按 rank 顺序显式求和。
            #   数学上 `Σ_r` 与 `all_reduce` 完全等价，但**求和顺序固定**、
            #   且避开了 all_reduce 在该形状/T 上的行为（实测它在 T=16 上给垃圾）。
            _g = torch.empty(
                (dcp, *_pack.shape), dtype=_pack.dtype, device=_pack.device
            )
            torch.distributed.all_gather_into_tensor(_g, _pack, group=group.device_group)
            _acc = _g[0]
            for _r in range(1, dcp):
                _acc = _acc + _g[_r]
            _pack = _acc
        elif _v41_rs_merge_on() and _fold_ori_locally and head_slice is not None:
            # =============================================================
            # ★★★★★★ [V41-RSMERGE 2026-10-01 03:20] **reduce_scatter 代替 all_reduce**。
            #
            # 事实：归约之后**每个 rank 只需要自己那 8 个 head**（`o_proj` 是 TP 切分的），
            # 而 `all_reduce` 让每个 rank 都收到全部 64 个 head 的和
            # ⇒ 88.9% 的接收量是白拿的。
            #
            # `reduce_scatter_tensor` 的语义正是"各 rank 贡献完整的输入、只收到自己
            # 那一段的和" ⇒ 数学恒等（逐元素求和后切片），但
            #   · 每 rank 接收量：164 KB → **20.5 KB**（1/8）；
            #   · ring 算法的搬运量：`2(N−1)/N·S` → `(N−1)/N·S`（减半）。
            #
            # 布局：把包转成 **head-major** `[H, T*W]`，这样 reduce_scatter 的
            # 第 r 段恰好是 head `[8r, 8r+8)` —— 与 `head_slice` 完全对齐，
            # 归约后**不需要任何切片/拷贝**。
            #
            # 开关：env `V41_DCP_RS_MERGE=1`（默认 **0**，先测量再决定是否转正）。
            # =============================================================
            _H = int(_pack.shape[1])
            _W = int(_pack.shape[-1])
            _hm = _pack.permute(1, 0, 2).reshape(_H, -1)
            if not _hm.is_contiguous():
                _hm = _hm.contiguous()
            _rows = _H // int(dcp)
            _rs_key = (int(dcp), int(_H), _W, int(_pack.shape[0]), _pack.dtype, str(_pack.device))
            _rs_out = _RS_OUT_CACHE.get(_rs_key)
            if _rs_out is None or tuple(_rs_out.shape) != (_rows, _hm.shape[1]):
                _rs_out = torch.empty((_rows, _hm.shape[1]), dtype=_pack.dtype, device=_pack.device)
                _RS_OUT_CACHE[_rs_key] = _rs_out
            torch.distributed.reduce_scatter_tensor(_rs_out, _hm, group=group.device_group)
            # 归约后的那段就是**本 rank 的 8 个 head** ⇒ 转回 [T, 8, W] 并清掉 head_slice，
            # 让下面的后处理不再二次切片。
            _pack = _rs_out.view(_rows, int(_pack.shape[0]), _W).permute(1, 0, 2)
            head_slice = None
            _rs_applied = True
        else:
            torch.distributed.all_reduce(_pack, group=group.device_group)
        # ★★★★★★ [V41-POSTRSYNC 2026-09-30] **归约后竞态**的直接判据。
        #   INV 探针（归约后另一次 `.cpu()`）读到权重列全列和 154349
        #   （= 本地 147181 + 7×1024，说明 all_reduce 本身是对的），而紧随其后的
        #   减法 kernel 产出的 `wsum` 却读到 0 —— 两者读的是**同一块 buffer**。
        #   若存在"减法抢在 HCCL 写回之前"的竞态，这里插一次主机同步就会恢复。
        #   只在非 capture 下用（capture 区 host 同步会崩，见 §6x）。
        if _perf_flags().get("postr_sync") == "1" and not _is_capturing():
            try:
                torch.npu.synchronize()
            except Exception as _e:  # noqa: BLE001
                print("[V41-POSTRSYNC] rank=%d 失败：%r" % (_v41_dcp_rank(), _e), flush=True)
        _out_dim = scaled.shape[-1]
        # =====================================================================
        # ★★★★★★ [V41-HARD 2026-09-30] **只打整数**的硬探针。
        #
        # 为什么需要：`float(device_tensor)` 与"多统计量一次快照"在本环境里都
        # 出现过自相矛盾的读数（§6o / §6ad）。而 `_pack.shape` / `_out_dim` /
        # `head_slice` 都是**Python int**，打印它们**物理上不可能撕裂**。
        # 这一支回答最后一个未验证的环节：**切片位置是否与布局匹配**。
        #   `wsum = _pack[..., _out_dim:_out_dim+1][:, h0:h1, :] - dcp*_keep`
        #   若 `_out_dim` 与实际布局不符（例如权重列不在第 512 列），
        #   切片取到的就是**别的数据** —— 完全解释 `wsum ≈ -7`（切到一片 1.0）。
        # =====================================================================
        if (
            _dcp_diag_on("hard", "V41_DCP_HARD")
            and not _is_capturing()
            and diag_seq_lens is not None
            and int(diag_seq_lens.max()) > 1
        ):
            _hs = head_slice if head_slice is not None else ("已切", "已切")
            print(
                "[V41-HARD] rank=%d layer=%d T=%d pack_shape=%s out_dim=%d "
                "out_shape=%s wt_shape=%s scaled_shape=%s hs=%s "
                "wcol_lo=%d wcol_hi=%d no_cache=%s det=%s"
                % (
                    _v41_dcp_rank(), int(layer_idx), int(diag_seq_lens.max()),
                    tuple(_pack.shape), int(_out_dim),
                    tuple(output.shape), tuple(weights.shape), tuple(scaled.shape),
                    tuple(_hs),
                    int(_out_dim), int(_out_dim + 1),   # 权重列的精确列区间
                    str(_DCP_NO_ATTN_CACHE), str(_DCP_DET_REDUCE),
                ),
                flush=True,
            )
        if _mdiag_here:
            # =================================================================
            # ★★★ [V41-MDIAG2 修正 2026-09-30] **单次主机拷贝 + CPU 侧计算**。
            #
            # 为什么必须这样：旧探针对**同一个 device 张量**做了十余次独立读取
            # （8 次块和 + 总 sum + min/max + zero_head_cnt …），而 `_pack` 在本函数
            # 返回后其显存会**归还分配器并被下一层复用** ⇒ 后续 kernel 可能读到
            # **下一层的数据**。实测后果：同一行里出现自相矛盾的读数
            #   `blk_sums=[0.0 × 8]`（8 块恰好覆盖全部 64 head）而同行的
            #   `wcol_min=wcol_max=1`、`wcol_sum=1024`（全 1.0 的张量分块和必为 1024）。
            # 统计：7904 条里有 **2875 条（36%）** 是这种不可信读数。
            # ⇒ 现在**只做一次 `.cpu()` 快照**，之后全部在 CPU 上算，物理上不可能撕裂。
            # =================================================================
            _snap = _pack[..., _out_dim : _out_dim + 1].to(torch.float32).cpu()
            _mdiag_post_t = float(_snap.sum())
            _H = int(_snap.shape[1])
            _h0, _h1 = (head_slice if head_slice is not None else (0, _H))
            _blk = max(1, _H // dcp)
            _blk_sums = [_snap[:, _b * _blk : (_b + 1) * _blk, :].sum().item() for _b in range(dcp)]
            _sliced = _snap[:, _h0:_h1, :]
            _wsum_snap = _sliced - dcp * _keep
            print(
                "[V41-MDIAG2] rank=%d T=%d H=%d hs=(%d,%d) blk=%d "
                "wcol[min=%.6g max=%.6g sum=%.6g] blk_sums=%s | "
                "sliced[min=%.6g max=%.6g sum=%.6g] wsum[min=%.6g max=%.6g mean=%.6g]"
                % (
                    _v41_dcp_rank(), int(_snap.shape[0]), _H, _h0, _h1, _blk,
                    float(_snap.min()), float(_snap.max()), _mdiag_post_t,
                    [round(v, 3) for v in _blk_sums],
                    float(_sliced.min()), float(_sliced.max()), float(_sliced.sum()),
                    float(_wsum_snap.min()), float(_wsum_snap.max()), float(_wsum_snap.mean()),
                ),
                flush=True,
            )
        # ★ 扣除量 = `Σ_r _onum_r` / `Σ_r _ow_r`：
        #   `_onum_r = ori_out·_keep` 与 `_ow_r = _keep` **每个 rank 各一份**
        #   ⇒ `Σ_r _onum_r = dcp·_onum`、`Σ_r _ow_r = dcp·_keep = dcp−1`。
        if _rs_applied:
            # ★ [V41-RSMERGE] reduce_scatter 已经只把**本 rank 的 8 个 head** 收回来了
            #   ⇒ 直接相减，不再切片（`head_slice` 也已置 None）。
            # ★ [V41-ONUMFIX 2026-10-01 07:25] **不要再乘 dcp**：`_onum` 自
            #   `V41-ONUMFOLD` 起已经内含 `_keep*dcp`，这里再乘一次会变成 dcp²
            #   （这条分支只在 `V41_DCP_RS_MERGE=1` 时走到，属潜在 bug，不是当前默认路径）。
            scaled = _pack[..., :_out_dim] - _onum
            wsum = _pack[..., _out_dim : _out_dim + 1] - dcp * _keep
            _hs_applied = (0, int(_pack.shape[1]))
        elif _slice_early:
            # 归约后立刻切到本 rank 的 8 个 head ⇒ 减法只在 8 个 head 上做。
            _h0, _h1 = head_slice
            scaled = _pack[..., :_out_dim][:, _h0:_h1, :] - _onum
            # ★ pad 之后必须**精确切 1 列**：`[..., _out_dim:]` 会带上 padding。
            # =============================================================
            # ★★★★★★ [V41-DENFIX 2026-09-30 12:25] 分母取值的**修复候选**。
            #
            # 症状（同一层同一次调用内，两个探针互相矛盾）：
            #   · MDIAG2（一次 `.cpu()` 快照后在 CPU 上算，**自洽**）：
            #       `wcol.sum=7588.87`、8 个块和相加 == 7588.87、各 rank 切片
            #       == 对应块和、`wsum ∈ [1.08, 46.1]` ⇒ 源数据**正确**；
            #   · MDIAG（设备侧）：`wsum[min=-7 max=-7] denom[min=1]`，且
            #       `out[absmax] == scaled[absmax]`（到 6 位有效数字完全相同）
            #       ⇒ 设备上的 `denom ≡ 1` ⇒ **`wsum ≤ 0`**。
            #   两者不可能同时对 ⇒ **设备侧那次「[T,H,1] 视图 + 标量减法」
            #   没有读到真实数据**（读到的等价于 padding 的 0，减 7 得 −7）。
            #   `postr_sync=1` 无效 ⇒ 不是流竞态，是取值路径本身。
            #
            # 两个候选修法（文件开关，可热 A/B，prefill 是 eager ⇒ 生效）：
            #   `contigw=1`：先把权重列 `.contiguous()`（一次 4 KB 的 strided
            #     拷贝，已验证 D2H 拷贝这条路径读出来是对的）再切片/减法；
            #   `sepw=1`   ：彻底不读 pack，改用 `weights` 的**独立 all_reduce**
            #     当分母 —— WCHK 实测这条路径完全正确
            #     （rank0 218618 + 7×1024 = 225786）。代价是多一次 4 KB 集合通信。
            # =============================================================
            # =============================================================
            # ★★★★★★ [V41-CONTIGW-DEFAULT 2026-10-01 06:40] **默认改为 `contigw`，
            #   去掉每层第二次集合通信。**
            #
            # 真实 8 卡 profiler 给出了**精确**的账（run dcpcap_1001_015103）：
            #
            # | 通信 | DCP8 | DCP1 | 归属 |
            # |---|---|---|---|
            # | allReduce group=097 | **76.0/step（=2×38）**, 0.976 ms/step | 无 | merge |
            # | allGather group=374 | 36.1/step, 0.670 ms/step | 无 | q gather |
            # | allReduce group=503 | 82.0/step | **82.0/step（逐字相同）** | TP/EP |
            #
            # `76 = 2×38` 说明 merge **每层做了两次 all_reduce**：一次是 164 KB 的
            # 打包包，另一次是这个**只有 256 字节**的分母。而集合通信在本平台是
            # **纯延迟 bound**（16 B 与 1 MB 同价，已多次实测）⇒ 这 256 字节
            # 的分母和 164 KB 的包**一样贵**。
            #
            # 分母本来就在打包包里（`pack[..., _out_dim]` 就是 `Σ_r w_r`），
            # 之所以还单独归约一次，是因为历史上"直接读 `[T,H,1]` 视图"在这台
            # NPU 上读到过垃圾值；当年用 `.contiguous()`（4 KB 拷贝）修复，
            # 并**实测两条路径都能把 T=16 的答对率从 0/25 拉到 16/25**。
            # ⇒ 现在默认走 `contigw`：省掉 38 次/step 的集合通信，代价是 38 次
            #    4 KB `ViewCopy`。预期 **−0.4~0.5 ms/step**。
            #
            # ★★★★★★ [V41-CONTIGW-ABORT 2026-10-01 07:50] **默认已回退为 `sepw`。**
            #
            # 将默认切到 `contigw` 后：T=904 针 **4/4 通过**、长针 6/6 通过，
            # 但**短问答回归**：`17 × 23 等于多少？只回答数字。` 连续 **6/6** 输出
            # 乱码（`# 标题：老婆背叛后…`），而上一版（sepw）稳定输出 `391`。
            # 回退到 `sepw` 后立刻恢复。
            #
            # ⇒ 与历史记录一致：**这台 NPU 上"读 pack 里的权重列"这条路径不可靠**
            #   （当年 `.contiguous()` 只修好了 T=16 那一组用例）。
            #   性能上 `contigw` 省掉 38 次/step 集合通信（设备时间 −0.55 ms/step），
            #   但 **wall clock 没有可测量的改善**（33.15 vs 33.23，在噪声内）
            #   ⇒ 不为它冒正确性风险。
            #
            # 如需复测：`sepw=0` 且 `V41_DCP_CONTIGW=1`（env，因为文件开关在
            # decode 图捕获后不生效）。
            # =============================================================
            # ★★★★★★ [V41-CONTIGW-SANITIZE 2026-10-01 08:30] **重试 contigw，
            #   这次与 `sanitize` 一起打开。**
            #
            # 上一次（唯一变量 = contigw）的失败模式：T=904 针 4/4 通过、长针 6/6
            # 通过，但短问答 `17 × 23 等于多少？只回答数字。` 连续 6/6 乱码
            # （`# 标题：老婆背叛后…`）。
            #
            # 现在有了**具体假设**：`scaled` 与 `weights` 被放进**同一个** all_reduce
            # 缓冲；本文件的历史实测已记录 **HCCL 在 buffer 含 Inf/NaN 时行为异常**
            # （正是当年加 `sanitize` 开关的原因）。分母列在这个"脏"缓冲里，
            # 所以读出来的 `Σ_r w_r` 可能是垃圾 ⇒ 短输入（`scaled` 更容易溢出/
            # 出现非有限值）先崩，而长上下文反而侥幸通过。
            #
            # ⇒ 如果假设成立，`sanitize=1`（把 `scaled` 的 Inf/NaN 清成有限值）
            #    就能同时拿到：**去掉 38 次/step 的集合通信** 且 **正确性保持**。
            #
            # ★★★★★★ [V41-CONTIGW-LOCATED 2026-10-01 10:45] **根因定位：坏在 decode（图内），
            # 不坏在 prefill（eager）。** 这解释了此前所有互相矛盾的观测：
            #
            # | 生效范围 | 短问答 `17×23` | T=904 针 |
            # |---|---|---|
            # | **文件开关** `sepw=0`（只影响 eager ⇒ **仅 prefill**） | **6/6 `391`** ✅ | `Q7` ✅ |
            # | **env** `V41_DCP_CONTIGW=1`（prefill **+ decode**） | **6/6 乱码**（`# 标题：老婆背叛后…`） | 通过 |
            # | 代码默认 = contigw（两者都是） | 乱码 | 通过 |
            # | 代码默认 = contigw **+ sanitize** | 乱码 | 通过 |
            #
            # ⇒ ① `contigw`（从 `pack` 的 strided 视图里读分母列）在**图捕获的 decode**
            #     里读到垃圾值；② 与 `sanitize`（Inf/NaN）**无关** —— 两者同时打开
            #     仍然坏，单独打开也坏；③ prefill 全对，所以只跑长针/T=904 会漏判。
            #
            # 这是"strided 视图读取在本平台不可靠"的**第三次**独立复现
            # （另两次：`PACKDIRECT` 的 `out=<strided>`、`SUBALPHA` 的混合 dtype）。
            # ⇒ `contigw` **永久默认关**。真正要拿这 38 次/step 的通信，
            #   必须在**编译出来的 kernel 里**读这块内存（见 `docs/` 的 AscendC 计划）。
            # 复现：`V41_DCP_CONTIGW=1`（env，必须重启）。
            #
            # 回退：`sepw=1`（默认即为该路径）。
            _want_contigw = _perf_flags().get("sepw") == "0" or (
                __import__("os").environ.get("V41_DCP_CONTIGW", "0") == "1"
            )
            if not _want_contigw:
                # ★ [V41-NOCLONE 2026-10-01 10:20] **就地 all_reduce，省掉 `clone()` 节点**。
                #   该分支（`_fold_ori_locally` + `sepw`）之后 `weights` 不再被任何地方引用
                #   （`_pack` 已经用 `copy_` 存了它自己的副本；`scaled` 也已算完），
                #   所以可以直接在 `weights` 上做 in-place all_reduce。
                #   `.to(fp32)`/`.contiguous()` 本来就都是 no-op（`weights` 出自
                #   `nan_to_num(exp(...))`，已是连续 fp32）。
                #   ⇒ 每层少 1 个 `ViewCopy` 节点（×38 = −38 节点/step ≈ 0.11 ms）。
                _ws = weights
                # ★ 定序求和（det_reduce=1）——否则分母的 all_reduce 顺序可变
                if _perf_flags().get("det_reduce") == "1":
                    _ws = _v41_ordered_allreduce(_ws, group)
                else:
                    torch.distributed.all_reduce(_ws, group=group.device_group)
                wsum = _ws[:, _h0:_h1, :] - dcp * _keep
            else:
                # ★ 必须 `.contiguous()`：直接读 strided 视图在这台 NPU 上会取到
                #   垃圾值（见上面的历史说明），而 `pack` 的权重列不在连续位置。
                _wcol_raw = _pack[..., _out_dim : _out_dim + 1].contiguous()
                wsum = _wcol_raw[:, _h0:_h1, :] - dcp * _keep
            _hs_applied = (_h0, _h1)
            head_slice = None  # 已应用，别在下面再切一次
        else:
            # ★ [V41-ONUMFIX] 同上：`_onum` 已内含 `_keep*dcp`，不再乘 dcp。
            scaled = _pack[..., :_out_dim] - _onum
            wsum = _pack[..., _out_dim : _out_dim + 1] - dcp * _keep
    elif perf_no_pack:
        # 消融臂：回到打包前的实现（4 次独立 all_reduce），用于量化打包收益
        _n_all = _onum.clone(); _w_all = _ow.clone()
        if _perf_flags().get("det_reduce") == "1":
            scaled = _v41_ordered_allreduce(scaled, group)
            wsum = _v41_ordered_allreduce(weights.contiguous(), group)
            _n_all = _v41_ordered_allreduce(_n_all, group)
            _w_all = _v41_ordered_allreduce(_w_all, group)
        else:
            torch.distributed.all_reduce(scaled, group=group.device_group)
            wsum = weights.clone()
            torch.distributed.all_reduce(wsum, group=group.device_group)
            torch.distributed.all_reduce(_n_all, group=group.device_group)
            torch.distributed.all_reduce(_w_all, group=group.device_group)
        if _ori_active:
            scaled = scaled - _n_all
            wsum = wsum - _w_all
    else:
        # 按最后一维拼包：`[T, H, D] + [T, H, D] + [T, H, 1] + [T, H, 1]`
        # ★ `weights` 直接交给 `cat`（cat 本来就会复制）⇒ 省掉一次 `clone`
        _pack = torch.cat([scaled, _onum, weights, _ow], dim=-1)
        if _perf_flags().get("det_reduce") == "1":
            _pack = _v41_ordered_allreduce(_pack, group)
        else:
            torch.distributed.all_reduce(_pack, group=group.device_group)
        _out_dim = scaled.shape[-1]
        scaled = _pack[..., :_out_dim]
        _n_all = _pack[..., _out_dim : 2 * _out_dim]
        wsum = _pack[..., 2 * _out_dim : 2 * _out_dim + 1]
        _w_all = _pack[..., 2 * _out_dim + 1 :]
        if _ori_active:
            scaled = scaled - _n_all
            wsum = wsum - _w_all
    # ★ [V41-PERF] 本 rank 的 `o_proj` 只吃自己那段 head（TP 连续切分）⇒ **先切片再除**：
    #   除法与 dtype 转换从 `[T,64,512]` 缩到 `[T,8,512]`（8× 少的访存），
    #   并且省掉调用方末尾那次 `output[:, h0:h1, :].contiguous()` 的独立拷贝节点。
    #   `scaled`/`wsum` 归约后本来就是**全 head**（all_reduce 逐元素，各 rank 都得全量），
    #   切片只是取自己那 8 列，数学不变。
    if head_slice is not None:
        h0, h1 = head_slice
        scaled = scaled[:, h0:h1, :]
        wsum = wsum[:, h0:h1, :]
        _hs_applied = (h0, h1)
    # =====================================================================
    # ★★★★★★ [V41-DENCHK 2026-09-30 12:25] **同一次调用内的设备 vs CPU 对拍**。
    #
    # 这是给"MDIAG2 说 wsum ≥ 1.08、MDIAG 说 wsum ≤ 0"这个矛盾下的最终判据：
    #   · `src`     = 归约后权重列（`.cpu()` 快照）的 min/max —— 源数据；
    #   · `ref_wsum`= 在 CPU 上对同一份快照做 `− dcp·keep` 的结果；
    #   · `dev_wsum`= **设备张量 `wsum` 本身的取值**（`dsync=1` 时同步后取）。
    # 若 `ref_wsum ≥ 1` 而 `dev_wsum ≤ 0` ⇒ 设备侧那条取值路径确实坏了（候选修法
    # `contigw` / `sepw` 应当把它修回来）；若两者一致 ⇒ 之前的矛盾来自读数伪影。
    # =====================================================================
    if (
        _dcp_diag_on("denchk", "V41_DCP_DENCHK")
        and not _is_capturing()
        and diag_seq_lens is not None
        and int(diag_seq_lens.max()) > 1
    ):
        try:
            _col_cpu = _pack[..., _out_dim : _out_dim + 1].to(torch.float32).cpu()
            _hs2 = _hs_applied if _hs_applied is not None else (0, int(_col_cpu.shape[1]))
            _ref = _col_cpu[:, _hs2[0] : _hs2[1], :] - float(dcp * _keep)
            print(
                "[V41-DENCHK] rank=%d layer=%d T=%d hs=(%d,%d) "
                "src[min=%.6g max=%.6g ncol=%d] ref_wsum[min=%.6g max=%.6g] | "
                "dev_wsum[min=%.6g max=%.6g shape=%s] sepw=%s contigw=%s"
                % (_v41_dcp_rank(), int(layer_idx), int(diag_seq_lens.max()),
                   int(_hs2[0]), int(_hs2[1]),
                   float(_col_cpu.min()), float(_col_cpu.max()), int(_col_cpu.shape[1]),
                   float(_ref.min()), float(_ref.max()),
                   _dcp_sf(wsum.min()), _dcp_sf(wsum.max()), tuple(wsum.shape),
                   str(_perf_flags().get("sepw")), str(_perf_flags().get("contigw"))),
                flush=True,
            )
        except Exception as _e:  # noqa: BLE001
            print("[V41-DENCHK] rank=%d 失败：%r" % (_v41_dcp_rank(), _e), flush=True)
    # =====================================================================
    # [V41-MDIAG] 合并**前后幅度**诊断（`V41_DCP_MERGE_DIAG=1`，非 capture）。
    # 用来定位"某个层的 attention 输出爆炸到 1e30"的中间步骤：
    # 实测（`V41_CED_LAYER_SNAPSHOT_*`，真机 T=16 失败用例）hidden 在
    # **layer02_post** 从 0.66 跳到 **1.37e30**，而通过用例（T=11）同层是 0.51。
    # layer 2 是**第一个走 DCP 合并的层**（ratio=2）⇒ 逐段量化：
    #   scaled 归约后幅度 / wsum / denom / 最终 out。
    # 判据：若 `scaled_after_sub` 已 ~1e30 而 `denom` O(1) ⇒ 爆炸在分子；
    #       若 `denom` ~0 而分子正常 ⇒ 爆炸在分母（wsum 抵消）。
    # =====================================================================
    if (
        _dcp_diag_on("mdiag", "V41_DCP_MERGE_DIAG")
        and not _is_capturing()
        and _MDIAG["n"] < _MDIAG_LIMIT
        and diag_seq_lens is not None
        and int(diag_seq_lens.max()) > 1
    ):
        _MDIAG["n"] += 1
        try:
            # ★ 这里 MUST 用**与真实代码路径相同**的 denom 语义（`where(wsum>0,...,1)`），
            #   否则探针会报一个代码里并不存在的"1e-30 分母"（我第一版就报错过）。
            _den = torch.where(wsum > 0, wsum, torch.ones_like(wsum))
            _o = (scaled / _den)
            # ★ [V41-DSYNC] 全部走 `_sf`（`dsync=1` 时同步后再取值），
            #   否则这批标量读可能撕裂成自相矛盾的读数（见 `_sf` 的说明）。
            _big = _dcp_sf((_o.abs() > 1e6).any()) > 0.5
            print(
                "[V41-MDIAG] rank=%d T=%d scaled[absmax=%.6g] wsum[min=%.6g max=%.6g mean=%.6g] "
                "denom[min=%.6g] out[absmax=%.6g] BIG=%s dcp=%d keep=%.6g subtrahend=%.6g "
                "w_local_sum=%.6g w_postreduce_sum=%.6g fold=%s"
                % (
                    _v41_dcp_rank(), int(scaled.shape[0]),
                    _dcp_sf(scaled.abs().max()), _dcp_sf(wsum.min()), _dcp_sf(wsum.max()),
                    _dcp_sf(wsum.mean()),
                    _dcp_sf(_den.min()), _dcp_sf(_o.abs().max()), _big,
                    int(dcp), float(_keep), float(dcp * _keep),
                    _dcp_sf(_mdiag_pre_t) if _mdiag_pre_t is not None else -1.0,
                    _dcp_sf(_mdiag_post_t) if _mdiag_post_t is not None else -1.0,
                    str(_fold_ori_locally),
                ),
                flush=True,
            )
        except Exception as _e:  # noqa: BLE001
            print("[V41-MDIAG] rank=%d 诊断自身失败：%r" % (_v41_dcp_rank(), _e), flush=True)
    # ★★★ [V41-FIX 2026-09-30] **不要**用 `clamp_min(1e-30)` 当分母保护。
    #   数学上 `wsum = e^{-c}·(A + ΣZ + ΣS) > 0` 恒成立；一旦它 **≤ 0**，
    #   说明上游（all_reduce / 权重）已经坏了，此时：
    #     · 原实现 `where(wsum > 0, wsum, ones_like)` ⇒ 分母取 1 ⇒ 结果错但**有限**；
    #     · `clamp_min(1e-30)` ⇒ `scaled/1e-30 ≈ 1e+30` ⇒ LM head 溢出 ⇒
    #       logits 全相同 ⇒ **静默变成均匀分布**（比"结果错"更难查，也更危险）。
    #   踩过的现场：HCCL 把 `[T,H,513]` 的包弄坏（未对齐）⇒ `wsum≈−5.7` ⇒
    #   用 clamp_min 时输出 6.6e+30，全模型爆掉。
    #   ⇒ 恢复原语义：坏值走 `1`，让错误"可见但不放大"。
    #   （`wsum` 恒正时两者逐位等价。）
    # =====================================================================
    # ★ [V41-INV] 数学不变量：`wsum = Σ_r w_r − (dcp−1)`，而
    #   `Σ_r w_r = dcp + (ΣZ + ΣS)/A ≥ dcp`  ⇒ **`wsum ≥ 1` 恒成立**。
    #   违反即"归约少算了 rank"或"权重算错"，是**结构性**错误而非数值噪声。
    #   违反时打印本地量以定位（单次 .sum() 读取，符合 §6o 的安全模式）。
    # =====================================================================
    if (
        _dcp_diag_on("inv", "V41_DCP_INV")
        and not _is_capturing()
        and diag_seq_lens is not None
        and int(diag_seq_lens.max()) > 1
    ):
        try:
            # ★★ 用**一次 stack + 一次 .cpu()** 同时取三个量，物理上不可能撕裂：
            #   (1) 本 rank 的本地 `weights` 和；(2) 归约后权重列的和；
            #   (3) 归约后**切片到本 rank head 区间**的和。
            #   `wsum = (3) − dcp*_keep`，若 (2) 正常而 (3) 为 0 ⇒ 切片/布局问题；
            #   若 (2) 本身就 ≈ 1 ⇒ 归约没把 8 个 rank 加起来。
            _wcol_full = _pack[..., _out_dim : _out_dim + 1].to(torch.float32)
            _snap = torch.stack([
                weights.to(torch.float32).sum().reshape(1),
                _wcol_full.sum().reshape(1),
                _wcol_full[:, _h0:_h1, :].sum().reshape(1),
                _wcol_full.min().reshape(1),
                _wcol_full.max().reshape(1),
            ]).cpu()
            _lsum, _rsum, _sliced_sum, _rmin, _rmax = (float(_snap[i]) for i in range(5))
            print(
                "[V41-INV] rank=%d layer=%d T=%d local_w=%.6g reduced_wcol=%.6g sliced_wcol=%.6g "
                "rmin=%.6g rmax=%.6g | wsum_min=%.6g (期望>=1) ratio=%.3f"
                % (_v41_dcp_rank(), int(layer_idx), int(diag_seq_lens.max()),
                   _lsum, _rsum, _sliced_sum, _rmin, _rmax,
                   float(wsum.min()), _rsum / max(_lsum, 1e-30)),
                flush=True,
            )
            _wmin = float(wsum.min())
            if False:
                print(
                    "[V41-INV] ★违反 wsum>=1: rank=%d layer=%d T=%d wsum_min=%.6g "
                    "wsum_max=%.6g wsum_mean=%.6g"
                    % (_v41_dcp_rank(), int(layer_idx), int(diag_seq_lens.max()),
                       _wmin, float(wsum.max()), float(wsum.mean())),
                    flush=True,
                )
        except Exception as _e:  # noqa: BLE001
            print("[V41-INV] rank=%d 检查自身失败：%r" % (_v41_dcp_rank(), _e), flush=True)
    # =====================================================================
    # ★★★★★★ [V41-WSUM 2026-09-30] **直接读 `wsum` 自身**。
    #
    # 为什么这是唯一可靠的探针：`wsum` 是**代码正在使用**的活张量
    # （紧接着就参与 `denom`），不可能被回收 ⇒ 不存在 §6o/§6ad 那类撕裂。
    # 之前所有读 `_pack` 的探针都出现过自相矛盾的读数（`sliced_wcol=0`
    # 而 `rmin=8`），因为 `_pack` 在归约后已不再被引用。
    #
    # 用途：确认 `wsum < 1`（违反数学不变量）是**真实现象**而非探针伪影，
    # 并给出它的**逐 head 分布**（哪些 head 为 0）。
    # =====================================================================
    if (
        _dcp_diag_on("wsum", "V41_DCP_WSUM")
        and not _is_capturing()
        and diag_seq_lens is not None
        and int(diag_seq_lens.max()) > 1
    ):
        try:
            # wsum 是活的 ⇒ 一次 stack+cpu 可靠
            _ws = torch.stack([
                wsum.min().reshape(1), wsum.max().reshape(1), wsum.mean().reshape(1),
                (wsum <= 0).sum().reshape(1),   # ≤0 的元素数
                (wsum > 0.5).sum().reshape(1),  # 合法的元素数（应 ≥1，故 >0.5 即合法）
                torch.tensor(float(wsum.shape[0] * wsum.shape[1]), dtype=wsum.dtype,
                             device=wsum.device).reshape(1),
            ]).cpu()
            # 逐 head 的"合法元素数"（哪几个 head 全 0）
            _byh = (wsum > 0.5).sum(dim=(0, 2)).cpu().tolist()
            print(
                "[V41-WSUM] rank=%d layer=%d T=%d min=%.6g max=%.6g mean=%.6g "
                "n_le0=%d n_gt05=%d n_tot=%d by_head=%s"
                % (_v41_dcp_rank(), int(layer_idx), int(diag_seq_lens.max()),
                   float(_ws[0]), float(_ws[1]), float(_ws[2]),
                   int(_ws[3]), int(_ws[4]), int(_ws[5]),
                   [int(v) for v in _byh[:16]]),   # 只打前 16 个 head 便于看
                flush=True,
            )
        except Exception as _e:  # noqa: BLE001
            print("[V41-WSUM] rank=%d 失败：%r" % (_v41_dcp_rank(), _e), flush=True)
    # =====================================================================
    # ★★★★★★ [V41-WCHK 2026-09-30] **最小复现**：单独 all_reduce 一份 `weights`。
    #
    # 已知铁证（WSUM 探针，读活张量）：T=16 上 `wsum` 的 128 个元素**全为 −7**
    # ⇒ 归约后权重列在 rank0 的 head 区间**精确为 0**；而 PREW 证明打包前正常。
    # ⇒ 问题在"归约"这一步。但 `no_pack=1`（4 次独立 all_reduce）也失败，
    #   所以要做**最干净的最小复现**：单独 all_reduce 一份 `weights` 的**连续副本**
    #   （形状 `[T,64,1]`、4 KB），**完全绕开 pack 的 640 列布局**。
    #   · 若这份副本正确 ⇒ 问题在 pack 的具体布局/大小；
    #   · 若这份副本也被清零 ⇒ **HCCL 在此形状上就有问题**（与 pack 无关）。
    # 判据用**一次 stack+cpu**（`_wc` 是活张量，可靠）。
    # =====================================================================
    if (
        _dcp_diag_on("wchk", "V41_DCP_WCHK")
        and not _is_capturing()
        and diag_seq_lens is not None
        and int(diag_seq_lens.max()) > 1
    ):
        try:
            # ★ [V41-WCHK-2 11:55] 四量**同一次快照**：
            #   (a) local_w    = 打包前本地 weights 和
            #   (b) copy_red   = **独立副本** all_reduce 后的和
            #   (c) packsliced = 归约后 pack 权重列**切到本 rank head**的和
            #   (d) packcol    = 归约后 pack 权重列**全列**的和
            # 若 (d) 正常、(c)=0 ⇒ 是**切片坐标/布局**问题（不是归约）；
            # 若 (b)(d) 都正常而 wsum 仍 −7 ⇒ wsum 自身的取值被读坏（竞态）。
            _hs_eff = head_slice if head_slice is not None else (0, int(wsum.shape[1]))
            _wc = weights.to(torch.float32).contiguous()
            _wc_loc = _wc.sum().clone()          # 流序保证在 reduce 之前取值
            _wc_rd = _wc.clone()
            torch.distributed.all_reduce(_wc_rd, group=group.device_group)
            _col = _pack[..., _out_dim : _out_dim + 1].to(torch.float32)
            _csl = _col[:, _hs_eff[0] : _hs_eff[1], :]
            _st = torch.stack([
                _wc_loc.reshape(1), _wc_rd.sum().reshape(1),
                _csl.sum().reshape(1), _col.sum().reshape(1),
                wsum.sum().reshape(1), wsum.min().reshape(1), wsum.max().reshape(1),
                (_col > 0).sum().reshape(1),
            ]).cpu()
            print(
                "[V41-WCHK] rank=%d layer=%d T=%d hs=(%d,%d) "
                "local_w=%.6g copy_red=%.6g | packsliced=%.6g packcol=%.6g "
                "n_gt0=%d/%d | wsum[min=%.6g max=%.6g sum=%.6g] ratio_copy=%.3f"
                % (_v41_dcp_rank(), int(layer_idx), int(diag_seq_lens.max()),
                   int(_hs_eff[0]), int(_hs_eff[1]),
                   float(_st[0]), float(_st[1]),
                   float(_st[2]), float(_st[3]),
                   int(_st[7]), int(_col.numel()),
                   float(_st[5]), float(_st[6]), float(_st[4]),
                   float(_st[1]) / max(float(_st[0]), 1e-30)),
                flush=True,
            )
        except Exception as _e:  # noqa: BLE001
            print("[V41-WCHK] rank=%d 失败：%r" % (_v41_dcp_rank(), _e), flush=True)
    if _perf_flags().get("denclamp", "1") != "0":
        # ★ [V41-DENCLAMP 2026-10-01 04:25] 用 `clamp_min` 替掉 `where(>0)+ones_like`：
        #   省掉 `Greater`(44/step) + `SelectV2`(55/step) + `Fill`(部分) ≈ 110 次下发。
        #   数学依据：`wsum = Σ_r w_r − (dcp−1) ≥ 1` 恒成立（已由 INV 探针实测
        #   `wsum_min ∈ [1.004, 1.041]` 验证）。`clamp_min(1e-30)` 在 wsum ≤ 0 时
        #   会放大误差（历史上踩过）⇒ 保留 `denclamp=0` 回退开关，**默认开**
        #   （2026-10-01 05:30；依据：INV 探针实测 `wsum_min ∈ [1.004, 1.041] ≥ 1`）。
        denom = wsum.clamp_min(1e-30)
    else:
        denom = torch.where(wsum > 0, wsum, torch.ones_like(wsum))
    # ★ 用 `torch.div` 直接指定 `out=` 会引入别名风险，保持简单：elementwise 结果
    #   天然连续，`to(dtype)` 后调用方无需再 `.contiguous()`。
    _merged = (scaled / denom).to(output.dtype)
    # =====================================================================
    # ★★★★★★ [V41-MERGEDOUT 2026-09-30 23:05] **合并输出（含 scaled/wsum/denom）落盘**。
    #
    # 判据链条（前两步已实测）：
    #   (1) 同一实例内 A/B 两次请求（A 输出 `Q7` 正确、B 输出 `#` 错），layer 2 的
    #       **全部**合并输入逐位相同：q/cmp_indices/seqused_*/sinks/metadata/页内容，
    #       且 8 个 rank 的 `lse`、`ori_lse`、partial `out`、`ori_out` 指纹一致；
    #   (2) layer 3 的 `q` 出现差异（462/904 行、max|d|=0.6875，bf16 几十 ULP）。
    # ⇒ 分叉只可能在 **(a) 本次合并自身** 或 **(b) 合并之后的 o_proj / MoE / 残差**。
    #    本探针给出 (a) 的直接判据：合并输出逐位相同 ⇒ 分叉在 (b)。
    #
    # 开关：`mergedout=1`；层/rank/后缀复用 `dumplayer`/`dumprank`/`dumpsuffix`，
    # 目录复用 `dumpdir`。落盘内容 `merged/scaled/wsum/denom`（fp32，prefill T=904
    # 时约 30 MB/rank/次），便于在 CPU 侧做逐位比较与差异定位。
    # =====================================================================
    if _perf_flags().get("mergedout") == "1" and not _is_capturing():
        try:
            import os as _osmo

            _modir = _perf_flags().get("dumpdir")
            _mol = {int(v) for v in _perf_flags().get("dumplayer", "").replace(" ", "").split(",")
                    if v.strip().lstrip("-").isdigit()}
            _mor = {int(v) for v in _perf_flags().get("dumprank", "").replace(" ", "").split(",")
                    if v.strip().isdigit()}
            _mosfx = _perf_flags().get("dumpsuffix", "")
            if _modir and (not _mol or int(layer_idx) in _mol) and (
                not _mor or int(_v41_dcp_rank()) in _mor
            ):
                _osmo.makedirs(_modir, exist_ok=True)
                torch.save(
                    {
                        "merged": _merged.detach().to(torch.float32).cpu(),
                        "scaled": scaled.detach().to(torch.float32).cpu(),
                        "wsum": wsum.detach().to(torch.float32).cpu(),
                        "denom": denom.detach().to(torch.float32).cpu(),
                        "shape": tuple(_merged.shape),
                        "dtype": str(_merged.dtype),
                    },
                    _osmo.path.join(
                        _modir,
                        "merged_l%d_r%d%s.pt" % (
                            int(layer_idx), int(_v41_dcp_rank()),
                            ("_" + _mosfx) if _mosfx else ""),
                    ),
                )
        except Exception as _e:  # noqa: BLE001
            print("[V41-MERGEDOUT] 失败：%r" % (_e,), flush=True)
    return _merged


_LSE_DIAG_COUNT = {"n": 0}


def _lse_diag_count() -> int:
    return _LSE_DIAG_COUNT["n"]


def _bump_lse_diag() -> None:
    _LSE_DIAG_COUNT["n"] += 1


_TIME_ACC = {}
_ORI_REF_DIAG = {"n": 0}
_WDIAG = {"n": 0}
_WDIAG_LIMIT = 4000
_MDIAG = {"n": 0}
_DCP_MDIAG_ON = __import__("os").environ.get("V41_DCP_MERGE_DIAG") == "1"
# ★ [V41-DIAG] T 定向门：诊断只在**指定长度**上打印。理由：padding 搜索会打几百次
#   请求，把 `_WDIAG`/`_MDIAG` 的计数上限吃光 ⇒ 真正想看的那次一条都打不出来
#   （2026-09-30 踩过）。`V41_DCP_DIAG_T="16,17"` 就只看这两个长度。
_DCP_DIAG_T = {
    int(v) for v in __import__("os").environ.get("V41_DCP_DIAG_T", "").replace(" ", "").split(",")
    if v.strip().isdigit()
}


def _dcp_diag_t_ok(t) -> bool:
    """T 定向门：优先文件 `diag_t=`，其次 env；都空则恒真（保持旧行为）。"""
    raw = _perf_flags().get("diag_t")
    if raw:
        s = {int(v) for v in raw.replace(" ", "").split(",") if v.strip().isdigit()}
        return (not s) or (int(t) in s)
    return (not _DCP_DIAG_T) or (int(t) in _DCP_DIAG_T)


def _dcp_diag_on(key: str, env_fallback: str = "0") -> bool:
    """运行时诊断开关（文件优先，其次 env）—— 免得为一个探针重启 20 分钟。"""
    return _perf_flags().get(key) == "1" or __import__("os").environ.get(env_fallback, "0") == "1"


def _dcp_sf(x):
    """取标量；文件开关 `dsync=1` 时**先做一次设备同步**再取值。

    ★ 为什么需要（2026-09-30 12:10）：同一层同一次调用里，两个读同一块
    `_pack` 的探针给出了**互相矛盾**的结论 ——
      MDIAG2（一次 `.cpu()` 快照后在 CPU 上算，**自洽**）：
        `wcol.sum=7588.87`，8 个块和相加 == 7588.87，各 rank 切片 == 对应块和，
        `wsum ∈ [1.08, 46.1]` ⇒ **归约后的权重列完全正确、wsum ≥ 1 成立**；
      MDIAG（设备侧 `float(wsum.min())` 等三次标量读）：
        同一次调用的同一批张量报 `wsum[min=-7 max=-7] denom[min=1]`。
    两者不可能同时对。`dsync=1` 把设备侧读数改成"同步后再取"，用来判定
    到底是 `wsum` 真为 −7，还是标量读被撕裂（§6o 的 copy stream 问题）。
    """
    if _perf_flags().get("dsync") == "1" and not _is_capturing():
        try:
            torch.npu.synchronize()
        except Exception:  # noqa: BLE001
            pass
    return float(x)
_DCP_MDIAG_STATE = {}
_DCP_DUMPED = {}   # [V41-DUMP-LAYER] done 用 (layer,suffix) 作键，支持多次落盘
# ★★★★★★ [V41-CFG-CACHE 2026-09-30 13:50] **并行配置的进程内缓存**。
#
# 动机：`_remap_selection`（indexer 复制态下才走）原来用
#   `get_forward_context().vllm_config.parallel_config`
# 取 interleave/block_size，但真机实测该 `ForwardContext` **没有 `vllm_config`
# 属性** ⇒ 复制态起服直接崩：
#   `RuntimeError: NPUModelRunner failed, error is 'ForwardContext' object has
#    no attribute 'vllm_config'`（run dcpcap_0930_133900）。
#
# 现在由 `DeepseekV41MetadataBuilder.__init__`（那里一定拿得到 vllm_config）
# 把三个常量写进这里；`_remap_selection` 优先读 `get_forward_context()`，
# 失败则回退到本缓存。
_V41_DCP_CFG = {"interleave": None, "block_size": None, "dcp_size": None}


def _v41_dcp_cfg():
    try:
        _ctx = get_forward_context()
        _vc = _ctx.vllm_config
        _par = _vc.parallel_config
        return (
            int(getattr(_par, "cp_kv_cache_interleave_size", 1) or 1),
            int(_vc.cache_config.block_size),
            int(getattr(_par, "decode_context_parallel_size", 1) or 1),
        )
    except Exception:  # noqa: BLE001
        return (
            _V41_DCP_CFG["interleave"],
            _V41_DCP_CFG["block_size"],
            _V41_DCP_CFG["dcp_size"],
        )


_DCP_RAWD_ON = __import__("os").environ.get("V41_DCP_RAW_DIAG") == "1"
# ★★ [V41-DIAG 2026-09-30] `V41_DCP_NO_ATTN_CACHE=1` ⇒ **禁用挂在 attn 上的三处持久缓存**
#   （`_v41_dcp_sinks_cache` / `_v41_dcp_neg_idx` / `_v41_dcp_neg_sinks_cache`），
#   每次调用新建。用于判别"非确定性是不是这些跨步存活、且按 T 重建的缓存张量引入的"。
#   背景：真机 T∈{12,16} 在 `temperature=0` 下 **4/4 次输出全不同**，而 DCP1 稳定；
#   T=12/16 又**都在 `capture_sizes` 里** ⇒ 缓存张量与图捕获的交互是头号嫌疑。
_DCP_NO_ATTN_CACHE = __import__("os").environ.get("V41_DCP_NO_ATTN_CACHE") == "1"
# ★★★ [V41-DIAG 2026-09-30] `V41_DCP_DET_REDUCE=1` ⇒ 把 pack 上的 `all_reduce`
#   换成 **`all_gather_into_tensor` + 按 rank 顺序显式求和**（数学恒等）。
#   动机（真机实测）：T=16 上 `all_reduce` **把权重分量变成垃圾** ——
#   `w_local_sum=2495`（本地健康）而 `w_postreduce_sum=1024`（= 全 1.0），
#   另一条甚至 `-1.94e38`/`inf`；而 T=13 上 `8198`（≈8×本地和，正常）。
#   ⇒ 判据：若换成定序归约后 T=12/16 变**确定且正确** ⇒ 根因就是
#     HCCL 在这个形状/T 上的 all_reduce，且这就是修复。
_DCP_DET_REDUCE = __import__("os").environ.get("V41_DCP_DET_REDUCE") == "1"
# ★★★ [V41-DIAG 2026-09-30] `ori_zero_cmp=1`：第二次「纯 ori」调用改传
#   **`seqused_cmp_kv = 0`**（而不是"`cmp_sparse_indices` 全 -1"）。
#   动机（真机实测）：`skip_2nd=1` 让 T=12/16 **从非确定变确定且正确** ⇒ 非确定
#   来自第二次调用。它当前靠"全 -1 索引"表达"没有 cmp 键"，而按内核语义
#   `actCmpS2Size = min(bound, CountValid(-1)) = 0` 才是正确路径；
#   直接把 `seqused_cmp_kv` 置零更干净，也避开全 -1 索引张量的退化路径。
_DCP_ORI_ZERO_CMP = None  # 运行时由 `_perf_flags()` 决定（见调用处）
_DCP_RAWD = {"n": 0}
# ★ [V41-CSEQ] DCP 合并里集合通信的**全局序号**：若各 rank 的序号序列不一致，
#   后续 collective 会**逐层错位配对**（HCCL 最典型的静默串位故障）。
_DCP_MERGE_SEQ = {"n": 0}
_DCP_RAWD_LIMIT = 4000
_MDIAG_LIMIT = 4000


def _time_mark(name: str, t0: float) -> float:
    """累加某一阶段的设备时间（需先 synchronize 才准）。

    只在 `V41_DCP_TIMING=1` 时启用；启用后本身会拖慢（每阶段一次同步），
    所以它给的是**相对占比**而不是稳态绝对性能。
    """
    import time as _t
    if _perf_flags().get("timing") != "1":
        return _t.perf_counter()
    try:
        torch.npu.synchronize()
    except Exception:  # noqa: BLE001
        pass
    now = _t.perf_counter()
    _TIME_ACC[name] = _TIME_ACC.get(name, 0.0) + (now - t0)
    return now


def _time_dump(tag: str) -> None:
    if _perf_flags().get("timing") != "1" or not _TIME_ACC:
        return
    total = sum(_TIME_ACC.values())
    if total <= 0:
        return
    parts = " ".join(f"{k}={v*1000:.1f}ms({100*v/total:.0f}%)" for k, v in sorted(_TIME_ACC.items(), key=lambda x: -x[1]))
    print(f"[V41-TIME] {tag} total={total*1000:.1f}ms {parts}", flush=True)
    _TIME_ACC.clear()


def _v41_dcp_gather_heads(q: torch.Tensor) -> torch.Tensor:
    """把本 rank 的 q 沿 head 维 all-gather 成全部 head。

    `q` 形状 `[T, H_local, D]` → 返回 `[T, H_total, D]`。
    `all_gather_into_tensor` 只沿 dim 0 拼接，所以先把 head 维换到最前。
    （与 SFA 的做法同构；SFA 是把 ql_nope/q_pe 拼起来一次 gather 以减少
    连续两次 collective 在 Ascend 上的流依赖问题。）
    """
    group = _v41_dcp_group()
    dcp_size = group.world_size
    if dcp_size <= 1:
        return q
    local_heads = int(q.shape[1])
    # ★ 先把 head 维换到最前，再按**转置后**的形状建输出缓冲。
    #   踩过的坑：写成 `torch.empty((dcp*H_local, *q.shape[1:]))` —— 而
    #   `q.shape[1:]` 是 `(H_local, D)`，不是 `(T, D)`，于是输出缓冲是
    #   `(dcp*H_local, H_local, D)`，真机直接
    #   `RuntimeError: output tensor size must be equal to world_size times input tensor size`。
    q_t = q.transpose(0, 1).contiguous()
    gathered = torch.empty(
        (dcp_size * local_heads, *q_t.shape[1:]),
        dtype=q.dtype,
        device=q.device,
    )
    torch.distributed.all_gather_into_tensor(
        gathered, q_t, group=group.device_group
    )
    return gathered.transpose(0, 1).contiguous()


def _v41_dcp_gather_1d(x: torch.Tensor | None) -> torch.Tensor | None:
    """沿第 0 维 all-gather 一个 per-head 向量（sink 就是这种）。

    gather q 到全部 head 之后，算子收到的 `n_heads_q` 是全量，
    所以 `sinks` 也必须是全量 `[H_total]`，否则算子参数校验/语义不匹配。
    """
    if x is None:
        return None
    group = _v41_dcp_group()
    dcp_size = group.world_size
    if dcp_size <= 1:
        return x
    out = torch.empty(
        (dcp_size * int(x.shape[0]), *x.shape[1:]), dtype=x.dtype, device=x.device
    )
    torch.distributed.all_gather_into_tensor(out, x.contiguous(), group=group.device_group)
    return out


@eager_break_during_capture
def dsa_v41_forward(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Execute V4.1 attention behind an explicit graph side-effect boundary."""
    forward_context = get_forward_context()
    attn = forward_context.no_compile_layers[layer_name]
    attn.v41_impl.forward(attn, None, hidden_states, output)


def dsa_v41_forward_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return None


direct_register_custom_op(
    op_name="dsa_v41_forward",
    op_func=dsa_v41_forward,
    mutates_args=["output"],
    fake_impl=dsa_v41_forward_fake,
    dispatch_key="PrivateUse1",
)


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    """Read one field from either an HF config object or a raw config dict."""
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


@dataclass
class DeepseekV41Metadata(AttentionMetadata):
    """Scheduler and cache-plane contract for one V4.1 cache resource.

    ``seq_lens``/``query_start_loc`` always stay in original-token
    coordinates, matching the common vLLM metadata. The ``cache_*`` fields
    describe the rows visible to the concrete cache plane. Keeping both
    coordinate systems here lets future fused kernels replace the eager path
    without rebuilding scheduling metadata in the model.
    """

    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    compress_ratio: int
    storage_block_size: int
    is_compressor_state: bool
    cache_kind: str = "unknown"
    positions: torch.Tensor | None = None
    cos: Any = None
    sin: Any = None
    num_actual_tokens: int = 0
    num_input_tokens: int = 0
    num_reqs: int = 0
    num_actual_reqs: int = 0
    num_decodes: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_prefill_tokens: int = 0
    logical_block_size: int = 0
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None
    cache_seq_lens: torch.Tensor | None = None
    max_query_len: int = 0
    max_seq_len: int = 0
    max_cache_seq_len: int = 0
    attn_state: Any = None
    is_prefilling: torch.Tensor | None = None
    causal: bool | torch.Tensor = True
    ori_win_left: int = 0
    ori_win_right: int = 0
    smla_metadata: torch.Tensor | None = None
    qli_metadata: torch.Tensor | None = None
    cmp_residual: torch.Tensor | None = None
    c2_ring_metadata: torch.Tensor | None = None
    c2_complete_mask: torch.Tensor | None = None
    c2_source_positions: torch.Tensor | None = None
    c2_source_cos: torch.Tensor | None = None
    c2_source_sin: torch.Tensor | None = None
    c2_metadata_group_id: int | None = None


@dataclass(frozen=True)
class DeepseekV41CompressorMetadata:
    """V4-shaped cache/state bundle consumed by the compressor stage."""

    cache: DeepseekV41Metadata
    state: DeepseekV41Metadata | None = None


@dataclass(frozen=True)
class DeepseekV41IndexerMetadata:
    """V4-shaped source cache bundle consumed by the indexer stage."""

    cache: DeepseekV41Metadata


@dataclass(frozen=True)
class DeepseekV41LayerMetadata:
    """All metadata consumed by one V4.1 attention layer invocation."""

    attention: DeepseekV41Metadata | None
    swa: DeepseekV41Metadata
    compressor: DeepseekV41CompressorMetadata | None
    indexer: DeepseekV41IndexerMetadata | None

    @property
    def positions(self) -> torch.Tensor:
        if self.swa.positions is None:
            raise RuntimeError("V4.1 SWA metadata does not contain input positions")
        return self.swa.positions

    def rope(self, layer_name: str, num_tokens: int):
        if self.swa.cos is None or self.swa.sin is None:
            raise RuntimeError("V4.1 SWA metadata does not contain RoPE tensors")
        return self.swa.cos[layer_name][:num_tokens], self.swa.sin[layer_name][:num_tokens]


def compressed_slot_mapping(slot_mapping: torch.Tensor, ratio: int) -> torch.Tensor:
    """Convert original-token physical slots to completed compressed slots.

    Logical block sizes must be divisible by ratio. Negative/padded slots and
    incomplete compression groups never produce a write.
    """
    if ratio not in (1, 2):
        raise ValueError("V4.1 only supports ratio 1 or 2")
    valid = (slot_mapping >= 0) & ((slot_mapping + 1) % ratio == 0)
    return torch.where(valid, slot_mapping // ratio, -1)


def _request_counts(common: Any, num_reqs: int):
    """Return V4-shaped request counters without synchronizing the NPU."""
    is_prefilling = getattr(common, "is_prefilling", None)
    query_start_loc_cpu = getattr(common, "query_start_loc_cpu", None)
    if (
        is_prefilling is None
        or query_start_loc_cpu is None
        or getattr(is_prefilling, "device", None) is None
        or is_prefilling.device.type != "cpu"
    ):
        return 0, 0, 0, 0
    flags = is_prefilling[:num_reqs].bool()
    query_lens_cpu = query_start_loc_cpu[1 : num_reqs + 1] - query_start_loc_cpu[:num_reqs]
    num_prefills = int(flags.sum().item())
    num_decodes = num_reqs - num_prefills
    num_prefill_tokens = int(query_lens_cpu[flags].sum().item())
    num_decode_tokens = int(query_lens_cpu[~flags].sum().item())
    return num_decodes, num_decode_tokens, num_prefills, num_prefill_tokens


def scatter_cache_sk(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Store rows using builder-prepared coordinates and V4's Ascend op.

    V4.1 cache planes can be views into a larger layer-outermost slot, so the
    physical page stride is not necessarily the contiguous stride implied by
    the plane shape. ``npu_scatter_nd_update_sk`` preserves that stride and
    treats the builder's ``[-1, -1]`` coordinates as skipped rows, matching V4.
    """
    if slot_mapping.ndim != 2 or slot_mapping.shape[-1] != 2:
        raise ValueError(
            f"V4.1 fused cache store requires builder-prepared [T, 2] slot_mapping, got {tuple(slot_mapping.shape)}"
        )
    cache = cache.squeeze(-2)
    indices = slot_mapping[: values.shape[0]]
    updates = values.to(cache.dtype).contiguous()
    torch.ops._C_ascend.npu_scatter_nd_update_sk(cache, indices, updates)


def pad_sparse_indices(indices: torch.Tensor, topk: int) -> torch.Tensor:
    """Convert V4.1's compact [T, K] selection into SMLA [T, 1, topk]."""
    if indices.ndim != 2:
        raise ValueError(f"V4.1 sparse indices must be rank 2, got {indices.shape}")
    if indices.shape[-1] > topk:
        raise ValueError(f"V4.1 sparse indices width {indices.shape[-1]} exceeds operator topk {topk}")
    if indices.shape[-1] < topk:
        indices = F.pad(indices, (0, topk - indices.shape[-1]), value=-1)
    return indices.unsqueeze(1).contiguous().int()


class DeepseekV41EagerAttentionImpl:
    """V4-shaped execution boundary backed by fused Ascend operators.

    Projection, compressor and indexer modules remain registered by the model,
    while this object resolves the complete per-layer metadata bundle and owns
    their invocation order.  That is the same separation used by ``dsa_v1``:
    model construction is independent from cache-aware attention execution.
    """

    def __init__(self, prefix, role, topology, long_kv_source_prefix, index_k_source_prefix):
        self.prefix = prefix
        self.layer_name = f"{prefix}.attn"
        self.role = role
        self.topology = topology
        self.swa_prefix = f"{prefix}.swa_cache"
        self.long_kv_source_prefix = long_kv_source_prefix
        self.index_k_source_prefix = index_k_source_prefix
        self.compressor_state_prefix = (
            f"{prefix}.compressor.state_cache" if role.is_kv_source and role.compress_ratio == 2 else None
        )

    def _get_layer_metadata(self, metadata) -> DeepseekV41LayerMetadata:
        try:
            swa = metadata[self.swa_prefix]
            long_kv = metadata[self.long_kv_source_prefix] if self.long_kv_source_prefix is not None else None
            index_k = metadata[self.index_k_source_prefix] if self.index_k_source_prefix is not None else None
            compressor_state = (
                metadata[self.compressor_state_prefix] if self.compressor_state_prefix is not None else None
            )
        except KeyError as exc:
            raise RuntimeError(f"Missing V4.1 cache metadata for {exc.args[0]}") from exc
        return DeepseekV41LayerMetadata(
            attention=long_kv,
            swa=swa,
            compressor=(
                DeepseekV41CompressorMetadata(long_kv, compressor_state)
                if self.role.is_kv_source and long_kv is not None
                else None
            ),
            indexer=(DeepseekV41IndexerMetadata(index_k) if index_k is not None else None),
        )

    @staticmethod
    def _project_q_kv(attn, hidden_states, cos, sin):
        q_a = attn.wq_a(hidden_states)
        qr = attn.q_norm(q_a)
        q = attn.wq_b(qr).unflatten(-1, (attn.n_local_heads, attn.head_dim))
        kv = attn.kv_norm(attn.wkv(hidden_states))
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        kv = kv.view(-1, 1, attn.head_dim)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            kv.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        return q.to(hidden_states.dtype), qr, kv.squeeze(1)

    def preprocess(self, attn, hidden_states, cos, sin, swa_metadata):
        """Project Q/KV and populate this layer's SWA cache on the current stream."""
        q, qr, kv = self._project_q_kv(attn, hidden_states, cos, sin)
        scatter_cache_sk(
            attn.dsa_attn.swa_cache_layer.kv_cache[0],
            swa_metadata.slot_mapping,
            kv,
        )
        return q, qr

    def multistream_preprocess(self, attn, hidden_states, cos, sin, swa_metadata):
        """Overlap Q Vector work with KV Cube work, then reverse their roles.

        Reuse V1's stream and projection wrappers. V4.1 keeps floating-point
        qr for its indexer and has no post-Wq_b Q RMSNorm. Stage events serialize
        the Cube matmuls; the final join makes SWA writes visible to attention.
        """
        main_stream = torch.npu.current_stream()
        aux_stream = dsv4_dsa_overlap_stream()
        v1_impl = attn.dsa_attn.dsa_attn.impl
        wq_a, wkv, wq_b = v1_impl.cv_wq_a, v1_impl.cv_wkv, v1_impl.cv_wq_b
        share_quant = (
            type(wq_a._quant_method) is type(wkv._quant_method) and wq_a._has_communication == wkv._has_communication
        )

        # Part 1: Q_a matmul (Cube) overlaps independent KV quantization (Vector).
        q_quant, q_scale = wq_a.quantize(hidden_states)
        kv_quant_done = None
        if share_quant:
            kv_quant, kv_scale = q_quant, q_scale
        else:
            q_quant_done = main_stream.record_event()
            with npu_stream_switch(aux_stream, enabled=True):
                aux_stream.wait_event(q_quant_done)
                kv_quant, kv_scale = wkv.quantize(hidden_states)
                kv_quant_done = aux_stream.record_event()
        q_a = wq_a.matmul(q_quant, q_scale, bias=attn.wq_a.bias)

        # Part 2: Q normalization/quantization (Vector) overlaps KV matmul (Cube).
        part2_start = main_stream.record_event()
        if kv_quant_done is not None:
            main_stream.wait_event(kv_quant_done)
        with npu_stream_switch(aux_stream, enabled=True):
            aux_stream.wait_event(part2_start)
            kv = wkv.matmul(kv_quant, kv_scale, bias=attn.wkv.bias)
            kv_matmul_done = aux_stream.record_event()
        qr = attn.q_norm(q_a)
        q_b_quant, q_b_scale = wq_b.quantize(qr)

        # Part 3: Q_b matmul (Cube) overlaps KV norm, RoPE and cache store (Vector).
        part3_start = main_stream.record_event()
        main_stream.wait_event(kv_matmul_done)
        with npu_stream_switch(aux_stream, enabled=True):
            aux_stream.wait_event(part3_start)
            kv = attn.kv_norm(kv).view(-1, 1, attn.head_dim)
            torch.ops._C_ascend.inplace_partial_rotary_mul(
                kv.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[attn.nope_head_dim, attn.head_dim],
            )
            scatter_cache_sk(
                attn.dsa_attn.swa_cache_layer.kv_cache[0],
                swa_metadata.slot_mapping,
                kv.squeeze(1),
            )
        q = wq_b.matmul(q_b_quant, q_b_scale, bias=attn.wq_b.bias).unflatten(-1, (attn.n_local_heads, attn.head_dim))
        main_stream.wait_stream(aux_stream)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        return q.to(hidden_states.dtype), qr

    def _write_compressed_source(
        self,
        attn,
        hidden_states,
        positions,
        cos,
        sin,
        metadata,
    ):
        compressor = attn.compressor
        if compressor is None or metadata.compressor is None or metadata.indexer is None:
            raise RuntimeError("V4.1 KV source is missing compressor or source metadata")
        compressor_metadata = metadata.compressor
        indexer_metadata = metadata.indexer
        ratio = self.role.compress_ratio
        if ratio == 1:
            latent = compressor(hidden_states)
            # C1 source positions are the current token positions. Reuse the
            # query RoPE selected by the SWA metadata builder instead of
            # indexing the global table a second time.
            source_cos = cos
            source_sin = sin
            index_slots = indexer_metadata.cache.slot_mapping[: positions.shape[0]]
            long_slots = compressor_metadata.cache.slot_mapping[: positions.shape[0]]
        else:
            if compressor_metadata.state is None:
                raise RuntimeError("V4.1 ratio-2 source is missing compressor-state metadata")
            state_metadata = compressor_metadata.state
            if state_metadata.c2_ring_metadata is None or state_metadata.c2_metadata_group_id is None:
                raise RuntimeError("V4.1 ring compressor metadata is missing")
            wait_for_device_metadata(DeviceMetadataStage.COMPRESSOR, state_metadata.c2_metadata_group_id)
            hidden_states_fp32 = hidden_states.float()
            kv = compressor.wkv(hidden_states_fp32)
            score = compressor.wgate(hidden_states_fp32)
            latent = compressor.pool_projected(kv, score, state_metadata)
            source_cos = state_metadata.c2_source_cos
            source_sin = state_metadata.c2_source_sin
            if source_cos is None or source_sin is None:
                fallback_cos, fallback_sin = get_cos_and_sin_dsa(state_metadata.c2_source_positions)
                source_cos = fallback_cos[attn.rotary_emb.layername]
                source_sin = fallback_sin[attn.rotary_emb.layername]
            source_cos = source_cos[: positions.shape[0]]
            source_sin = source_sin[: positions.shape[0]]
            index_slots = indexer_metadata.cache.slot_mapping[: positions.shape[0]]
            long_slots = compressor_metadata.cache.slot_mapping[: positions.shape[0]]

        if attn.indexer is None:
            raise RuntimeError("V4.1 KV source is missing its indexer")
        attn.indexer.update_keys(
            latent,
            index_slots,
            source_cos,
            source_sin,
        )
        latent = latent.view(-1, 1, attn.head_dim)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            latent.unsqueeze(1),
            source_cos,
            source_sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        scatter_cache_sk(
            attn.long_kv_cache.kv_cache[0],
            long_slots,
            latent.squeeze(1),
        )

    def _select_sparse_indices(self, attn, hidden_states, qr, positions, cos, sin, metadata):
        if not self.role.has_long_context:
            return None
        shared = attn.shared_state
        if shared is None:
            raise RuntimeError("V4.1 shared attention state is not initialized")
        if not self.role.is_index_source:
            selected = shared.topk_indices[: hidden_states.shape[0]]
            # =============================================================
            # ★★★★★★ [V41-REMAP-CACHE 2026-09-30 19:55] **每步只 remap 一次**。
            #
            # `remap_sparse_indices`（复制态）含一次 **argsort**（稳定压缩有效项到行首）。
            # 38 个稀疏层各调一次 ⇒ 每步 38 次 argsort + 若干小算子；
            # 实测代价：非复制态 33.8 ms/step → 复制态 40.0 ms/step（+18%）。
            #
            # 但 remap 的**输入只有全局选择结果**（不依赖 positions）：同一 ratio
            # 的层共享 `shared.topk_indices` ⇒ 结果完全一样，缓存即可。
            # 一旦某个 index-source 层重算并覆盖了 `topk_indices`，必须失效缓存
            # ⇒ 键里带 ratio，且写入侧主动剔除。
            # =============================================================
            _ck = int(self.role.compress_ratio)
            _cache = getattr(shared, "topk_remapped", None)
            if _cache is not None:
                _cached = _cache.get(_ck)
                if _cached is not None and int(_cached.shape[0]) >= int(selected.shape[0]):
                    return _cached[: selected.shape[0]]
            _remapped = self._remap_selection(selected, positions)
            if _cache is not None:
                _cache[_ck] = _remapped
            return _remapped[: selected.shape[0]]
        if attn.indexer is None or metadata.indexer is None:
            raise RuntimeError("V4.1 index source is missing indexer metadata")

        context = get_forward_context().no_compile_layers
        source_layer = context[self.index_k_source_prefix]
        selected, candidates = attn.indexer.select(
            hidden_states,
            qr,
            positions,
            cos,
            sin,
            source_layer.kv_cache[0],
            metadata.indexer.cache,
            is_candidate_source=self.role.is_candidate_source,
            uses_candidate_filter=self.role.uses_candidate_filter,
            candidate_topk_blocks=self.topology.candidate_topk_blocks,
            candidate_block_size=self.topology.candidate_block_size,
            candidates=shared.candidates[: hidden_states.shape[0]],
        )
        # [V41-DCP] `indexer.select` 的输出是**全局压缩 token 坐标**（因为 indexer K
        # cache 是每个 rank 一份全量复制，所以 8 个 rank 选出的集合逐位相同）。
        # 而本 rank 的 long_kv 只有 1/dcp，所以必须把全局坐标重映射到本 rank 的
        # 本地压缩坐标，并把不属于本 rank 的项置 -1（算子对负索引直接跳过）。
        #
        # ★★★★★★ [V41-DOUBLEMAP-FIX 2026-09-30 15:20] **`shared.topk_indices` 必须
        # 存「全局坐标」，各层各自 remap 一次。**
        #
        # 原实现先在 index source 层 remap、再把**已经变成局部坐标**的结果写进
        # `shared.topk_indices`；而非 index source 层（`if not self.role.is_index_source`
        # 分支）读到它之后**又调了一次 `_remap_selection`** ⇒ **双重 remap**。
        #
        # 真机证据（run dcpcap_0930_150035，T=904，`idxfp=1`）：layer 2（index source）
        # 8 个 rank 的 `n_valid` 都正常（32256/26112/24208/22656/21192/19584…），
        # 而 **layer 7 的 rank 1–7 全部 `n_valid=0`**，只剩 rank0 有 14208 个。
        # 机制：局部坐标的值域只有 `[0, B'·? )`（很小），把它再当成全局坐标做
        # owner 判定 `((idx·ratio+ratio−1)//I) % dcp` ⇒ 绝大多数落回 rank0 ⇒ 只有
        # rank0 保留。**这正是长上下文乱码（T≳900）的直接原因**：layer 3–19 的
        # 压缩注意力全部丢失了 7/8 的键。
        #
        # 分片态（`V41_DCP_REPLICATE_INDEXER=0`）下 `_remap_selection` 是 no-op，
        # 所以这个缺陷只在复制态暴露 —— 也是为什么之前一直没被发现。
        #
        # 修法：写入共享缓冲的是**全局坐标**；返回值才做 remap。
        # ★ 先写**全局**坐标进共享缓冲（供非 index-source 层各自 remap），
        #   再对本层的返回值做一次 remap。传 `shared.topk_indices` 而不是
        #   `selected`：非复制态下 `_remap_selection` 是 no-op ⇒ 返回共享缓冲的
        #   视图（保持改动前的零分配行为）；复制态下它返回新张量。
        shared.topk_indices[: selected.shape[0]].copy_(selected)
        _selected_global = shared.topk_indices[: selected.shape[0]]
        selected = self._remap_selection(_selected_global, positions)
        # ★ 本次选择已变 ⇒ 写入同 ratio 的 remap 缓存（见上）
        _cache2 = getattr(shared, "topk_remapped", None)
        if _cache2 is not None:
            _cache2[int(self.role.compress_ratio)] = selected
        if self.role.is_candidate_source:
            shared.candidates[: candidates.shape[0]].copy_(candidates)
        # =====================================================================
        # ★★★★★★ [V41-IDXKVFP 2026-09-30 16:40] **index K 缓存的内容指纹**。
        #
        # 已确认：req#1（对）与 req#2（错）的 **long_kv 内容逐位相同**
        # （`[V41-KVDATA]` 两次都 `absmax=4.125`、`nonfinite=0`），
        # 但 **QLI 选出的索引集合不同**
        # （`n_valid 51254→51260`、`sum 2035126→2027486`、`max=127` 相同）。
        #
        # QLI 的输入只有 q（同一条 prompt ⇒ 相同）、index K 缓存、metadata。
        # ⇒ 只要 index K 的指纹在两次请求间不同，就锁定**复制态 index K 的写路径**；
        #    若相同，则只能怪 metadata / 算子内部状态。
        #
        # 指纹用**一次求和 + 一次非零计数**（单次回传，避免多次标量读撕裂）。
        # =====================================================================
        if (
            _perf_flags().get("idxkvfp") == "1"
            and not _is_capturing()
            and int(selected.shape[0]) > 1
        ):
            try:
                # `kv_cache[0]` 是 `(int8 key, fp16 scale)` 二元组（`indexer.select`
                # 里 `key, key_scale = source_cache`）——踩过一次 AttributeError。
                _ik_tuple = source_layer.kv_cache[0]
                _ik = _ik_tuple[0]
                _sc = _ik_tuple[1]
                # ★ 只取**本请求写入的那几页**（整库累计会被别的请求污染 —— 踩过：
                #   `scale.nz 452→904` 其实是"两个请求各写 452 行、落在不同页"）。
                _md = getattr(metadata.indexer, "cache", None)
                _bt = getattr(_md, "block_table", None)
                _cols = None
                if _bt is not None and _bt.numel():
                    _row0 = _bt[0].detach().to(torch.int64).cpu()
                    _cols = [int(v) for v in _row0.tolist() if int(v) > 0]
                if _cols:
                    _pk = _ik[_cols].detach()
                    _ps = _sc[_cols].detach()
                    _ksum = float(_pk.to(torch.float32).sum())
                    _knz = int((_pk != 0).sum())
                    _ssum = float(_ps.to(torch.float32).sum())
                    _snz = int((_ps != 0).sum())
                else:
                    _ksum = _knz = _ssum = _snz = -1
                _this = int(self.role.compress_ratio)
                _p0 = positions[: min(8, int(positions.numel()))].detach().to(torch.int64).cpu().tolist()
                _done = int((positions.detach().remainder(_this) == (_this - 1)).sum())
                print(
                    "[V41-IDXKVFP] layer=%d rows=%d ratio=%d pages=%s "
                    "THIS_PAGES key{sum=%.8g nz=%d} scale{sum=%.8g nz=%d} | "
                    "pos[:8]=%s complete_groups=%d/%d stride0=%d"
                    % (int(self.role.layer_idx), int(selected.shape[0]), _this,
                       ([int(v) for v in _cols[:4]] if _cols else []),
                       _ksum, _knz, _ssum, _snz,
                       _p0, _done, int(positions.numel()), int(_ik.stride(0))),
                    flush=True,
                )
            except Exception as _e:  # noqa: BLE001
                print("[V41-IDXKVFP] 失败：%r" % (_e,), flush=True)
        return selected

    def _remap_selection(self, selected, positions):
        """把全局压缩 top-k 索引重映射到本 rank 的本地压缩坐标。"""
        import os as _os

        from vllm_ascend.patch.platform.patch_v41_dcp import replicate_indexer, v41_dcp_active

        if not v41_dcp_active() or not replicate_indexer():
            return selected
        from vllm.distributed import get_dcp_group

        from vllm_ascend.attention.context_parallel.v41_dcp import remap_sparse_indices

        # ★ [V41-CFG-CACHE] 不再直接摸 `get_forward_context().vllm_config`（该属性
        #   在真机上不存在 ⇒ 复制态起服崩）；改为带进程内缓存的取法。
        interleave, block_size, dcp_size = _v41_dcp_cfg()
        if not dcp_size or dcp_size <= 1:
            return selected
        return remap_sparse_indices(
            selected,
            block_size=int(block_size),
            interleave=interleave,
            ratio=int(self.role.compress_ratio),
            dcp_size=dcp_size,
            dcp_rank=int(get_dcp_group().rank_in_group),
            # ★ [V41-REPL-LAYOUT] 本分支只在复制态进入 ⇒ 行号必须按复制态布局取。
            replicated=True,
        )

    def _attention(self, attn, q, metadata, compressed_indices, q_local_heads=None, q_hpr=0):
        source_cache = None
        if self.role.has_long_context:
            source_cache = get_forward_context().no_compile_layers[self.long_kv_source_prefix].kv_cache[0]
        return self._native_attention(
            attn,
            q,
            metadata,
            source_cache=source_cache,
            compressed_indices=compressed_indices,
            q_local_heads=q_local_heads,
            q_hpr=q_hpr,
        )

    def _prepare_q_for_dcp(self, q):
        """★ [V41-PERF 2026-09-29] 把 q 的 head 维 all_gather **提前到 indexer 之前**。

        `forward` 的原始顺序是
            preprocess → [kv source 写] → `_select_sparse_indices`（indexer） → `_attention`
        而 gather 原本在 `_attention` 内部 ⇒ 它与 indexer **串行**。

        但 indexer（QLI top-k + `prepare_indexer_indices`）只依赖
        `hidden_states / qr / positions / cos / sin`，**不依赖** gather 后的 q
        ⇒ 把 gather 提到 indexer 之前，HCCL 的 head all_gather 就有机会与
        indexer 的计算重叠（Ascend 上 HCCL 走独立通信流，见 vLLM-Ascend
        `attention_cp.py` 里 "COMM_STREAM: -- all_gather Q --" 的注释）。

        返回值 `(q, local_heads, hpr)`：
          · `q` 是 gather（并可能被消融截断）后的张量；
          · `local_heads` 是**gather 前**本 rank 的 head 数（后续切片要用）；
          · `hpr` 是消融开关 `heads_per_rank` 的值（0 = 正常）。

        `heads_per_rank=K` 消融：把 gather 后的 q 截断到每 rank K 个 head
        （⇒ 全局 K·dcp 个 head）再进 SMLA。目的：把「DCP8 每 rank 要算全部
        64 head」这一个因素**单独**拎出来做 A/B（DCP1 基线每 rank 只算 8 个）。
        ★ 只用于性能测量：结果数值不正确。**不得进生产**。
        """
        local_heads = int(q.shape[1])
        from vllm_ascend.patch.platform.patch_v41_dcp import v41_dcp_active

        if not v41_dcp_active():
            return q, local_heads, 0
        # ★★ 必须与 `_native_attention` 里的 `dcp_active` **同条件**：
        #     `dcp_active = _v41_dcp_on() and has_compressed`，
        #     而 `has_compressed = self.role.compress_ratio in (1, 2)`。
        #   ratio=0 的层（纯滑窗、不走压缩平面）**不做跨 rank 归约**，
        #   算子收到的 q 必须仍是本 rank 的 8 个 head。
        #   踩过的坑（cap54 起服失败）：无条件 gather ⇒ 这些层的 q 变成 64 head，
        #   而 `sinks` 仍是 8 ⇒ 算子直接
        #     `Invalid_Argument_Tensor_Shape(EZ0009): Parameter sinks of
        #      SparseFlashMla has incorrect shape [8]. Reason: Sinks's
        #      dimension(8) should be equal to the head num of query(64).`
        if self.role.compress_ratio not in (1, 2):
            return q, local_heads, 0
        _pf = _perf_flags()
        if _pf.get('skip_gather') != '1':
            q = _v41_dcp_gather_heads(q)
        try:
            _hpr = int(_pf.get('heads_per_rank', '0') or '0')
        except ValueError:
            _hpr = 0
        if _hpr > 0:
            _h_total_want = _hpr * _v41_dcp_group().world_size
            if _h_total_want < int(q.shape[1]):
                q = q[:, :_h_total_want, :].contiguous()
        return q, local_heads, _hpr

    def _native_attention(
        self,
        attn,
        q,
        metadata,
        *,
        source_cache,
        compressed_indices,
        q_local_heads=None,
        q_hpr=0,
    ):
        """Run SparseFlashMla with the same PA metadata for both operator stages."""
        # ★ [V41-PERF] q 的 head 维 all_gather 与消融截断已由调用方
        #   `_prepare_q_for_dcp` 在 **indexer 之前**做完 ⇒ 通信与 indexer 计算重叠。
        _q_local_heads = q_local_heads if q_local_heads is not None else int(q.shape[1])
        _q_hpr = int(q_hpr)
        if attn.head_dim != 512:
            raise ValueError(f"SparseFlashMla requires head_dim 512, got {attn.head_dim}")
        if attn.window_size != 128:
            raise ValueError(f"A2/A3 SparseFlashMla requires sliding_window 128, got {attn.window_size}")
        if not 1 <= attn.n_local_heads <= 128 or attn.n_local_heads & (attn.n_local_heads - 1):
            raise ValueError(
                "A2/A3 SparseFlashMla requires the local query-head count to be "
                f"a power of two in [1, 128], got {attn.n_local_heads}"
            )
        has_compressed = self.role.compress_ratio in (1, 2)
        ratio = self.role.compress_ratio if has_compressed else 0
        num_reqs = metadata.swa.num_reqs
        query_start_loc = metadata.swa.query_start_loc[: num_reqs + 1]
        seq_lens = metadata.swa.seq_lens[:num_reqs]
        ori_block_table = metadata.swa.block_table[:num_reqs]
        # =====================================================================
        # [V41-SLOTTRACE] SWA 写侧槽位：哪些物理页真的被写。
        # =====================================================================
        if _SLOTTRACE and not _is_capturing():
            try:
                _st_flags = _perf_flags()
                _st_layers = {
                    int(v)
                    for v in _st_flags.get("dumplayer", "").replace(" ", "").split(",")
                    if v.strip().lstrip("-").isdigit()
                }
                _st_key = (int(self.role.layer_idx), _st_flags.get("dumpsuffix", ""))
                if (
                    _st_flags.get("slottrace") == "1"
                    and (not _st_layers or int(self.role.layer_idx) in _st_layers)
                    and _st_key not in _SLOTTRACE_DONE
                ):
                    _SLOTTRACE_DONE[_st_key] = 1
                    # ★ [V41-SLOTTRACE-2 2026-10-01 00:05] **纠正坐标语义**。
                    #   读了 builder（本文件 3656-3695）后确认：`_slot_mapping_2d`
                    #   的两列是 **(block, row) 坐标**，不是"两个 cache group"：
                    #       [:, 0] = physical // storage_block_size   （块号）
                    #       [:, 1] = physical %  storage_block_size   （块内行号）
                    #   因此"写侧落到哪些块/行"可以直接读出来。
                    _sm_raw = metadata.swa.slot_mapping
                    if _sm_raw.ndim != 2 or _sm_raw.shape[1] != 2:
                        print("[V41-SLOTTRACE] 形状意外：%s" % (tuple(_sm_raw.shape),), flush=True)
                    else:
                        _blk = _sm_raw[:, 0].to(torch.int64)
                        _row = _sm_raw[:, 1].to(torch.int64)
                        _pos = metadata.swa.positions
                        _ub = sorted({int(v) for v in _blk.tolist()})
                        _ur = sorted({int(v) for v in _row.tolist()})
                        _pairs = [
                            (int(_blk[i]), int(_row[i]))
                            for i in (0, 1, 127, 128, 129, 255, 256, 257, 903)
                            if i < _blk.numel()
                        ]
                        _bt = metadata.swa.block_table[0, :10].to(torch.int64).cpu().tolist()
                        print(
                            "[V41-SLOTTRACE] TP=%d layer=%d T=%d | block uniq n=%d=%s | "
                            "row uniq n=%d min=%d max=%d | 采样(pos→blk,row)=%s | "
                            "ori_bt[:10]=%s | n_neg_blk=%d n_neg_row=%d | positions=%s"
                            % (
                                int(_v41_dcp_rank()), int(self.role.layer_idx),
                                int(_blk.numel()), len(_ub), _ub[:10],
                                len(_ur), int(_row.min()), int(_row.max()),
                                _pairs, _bt,
                                int((_blk < 0).sum()), int((_row < 0).sum()),
                                "-" if _pos is None else "%d..%d"
                                % (int(_pos.min()), int(_pos.max())),
                            ),
                            flush=True,
                        )
            except Exception as _e:  # noqa: BLE001
                print("[V41-SLOTTRACE] 失败：%r" % (_e,), flush=True)
        cmp_block_table = None
        cmp_seq_lens = None
        cmp_residual = None
        cmp_indices = None
        _kd_lse2 = None      # ★ [V41-KDET] 必须在分支外先初始化（踩过 UnboundLocalError）
        _kd_out2 = None
        cmp_topk = 0
        if has_compressed:
            if source_cache is None or metadata.attention is None or compressed_indices is None:
                raise RuntimeError("V4.1 compressed attention is missing KV or TopK metadata")
            cmp_block_table = metadata.attention.block_table[:num_reqs]
            # =============================================================
            # ★★★★★★ [V41-NULLBLK 2026-09-30 23:30] **空块表项 = 页 0 = 别的层的环形页**。
            #
            # 单卡实测（`~/tmp/toggle.py`，设备侧建池保住 stride0）：
            #   · 生产的 `ori_block_table` **只有一个非零项**（列 0 = 环形页 17，其余 8191 列恒 0）；
            #     `cmp_block_table` 同理（列 0 = 11，其余 1023 列恒 0）。
            #   · **开关式毒化判据**（A/B 反复切某页内容，看输出跟不跟着走）：
            #       ori 页 0 有毒 → `max|dout|=9986.7`，还原 → 逐位回到基准 ⇒ **页 0 被真读**；
            #       ori 页 1..16、cmp 页 0..10 有毒 → 输出**完全不变**（0/0）⇒ 那些页没被读。
            #   · 把 `ori_block_table` 的**所有列都填成列 0 的值**后，输出也变
            #     （`max|dout|=3.69`）⇒ **列号确实参与寻址**，空列 0 会落到"页 0"。
            #
            # ⇒ 机理：块表 0 表示"页 0"，而页 0 是**本请求没写过**的共享池页（很可能属于
            #   别的层的环形窗口）。生产上它装着别的请求/别的层的 KV ⇒ 污染注意力。
            #   这解释了稳定的"**新起容器第一个请求正确、之后全错**"（`Q7` → `#`，三次复现）。
            #
            # 下面两个开关是**候选修复**（都是纯 host 侧、不改算子二进制）：
            #   `oribtpad=1`：`ori_block_table` 的每一列都指向**本请求的环形页**
            #                 （环形语义下"position p → 行 p%128"，列只是页号，等价且不会落到页 0）；
            #   `cbtpad=1`  ：`cmp_block_table` 同理。
            # 注意：**只读诊断开关**，改的是本地副本，不污染 metadata 的共享 buffer。
            # =============================================================
            if _perf_flags().get("oribtpad") == "1" and ori_block_table.shape[1] > 1:
                ori_block_table = ori_block_table[:, :1].expand(
                    -1, ori_block_table.shape[1]
                ).contiguous()
            if _perf_flags().get("cbtpad") == "1" and cmp_block_table.shape[1] > 1:
                cmp_block_table = cmp_block_table[:, :1].expand(
                    -1, cmp_block_table.shape[1]
                ).contiguous()
            if _perf_flags().get("wipepage0") == "1" and not _is_capturing():
                # 诊断用：把两个平面的**页 0**清零，看输出是否回到未污染状态。
                # （页 0 是别的层的环形页 ⇒ 这只是诊断，不是修复。）
                try:
                    attn.dsa_attn.swa_cache_layer.kv_cache[0][0].zero_()
                    source_cache[0].zero_()
                except Exception as _e:  # noqa: BLE001
                    print("[V41-NULLBLK] wipepage0 失败：%r" % (_e,), flush=True)
            # ★ [V41-BTPAD2 2026-09-30 17:12] 把块表**补一列**（第 2 列 = 第 1 列）。
            #   判据：若算子把 `idx` 分解成 (blkIdx>0, row) 而越界读到第 2 列（当前
            #   恒为 0 = null 块）⇒ 补列后会读到**正确页** ⇒ 第 2 次请求变正确。
            #   只读诊断，不改 KV。
            if _perf_flags().get("btpad2") == "1" and cmp_block_table.shape[1] >= 1:
                cmp_block_table = torch.cat(
                    [cmp_block_table, cmp_block_table[:, :1]], dim=1
                )
            cmp_seq_lens = metadata.attention.cache_seq_lens[:num_reqs]
            # ★★★★★ [V41-CMPGLOB 2026-10-01] **把 cmp 长度按全局压缩坐标传给算子**。
            #   见 `_v41_global_cmp_lens` 的说明；单卡已验证：传入全局长度后
            #   同一输入 4/4 逐位一致，且 5 行全部与 torch 参考吻合到 ≤2e-6。
            #   开关 `cmpglob=0` 可回退（用于 A/B）。
            if (
                _perf_flags().get("cmpglob", "1") != "0"
                # ★ [V41-CMPGLOB-DCPONLY] **只在 DCP 生效时改**：DCP=1 时
                #   `cache_seq_lens` 与 `floor(seq/ratio)` 本就是同一坐标系，
                #   这条改动必须是 no-op，否则 DCP1 基线会被污染。
                and _v41_dcp_on()
                and seq_lens is not None
                and cmp_seq_lens is not None
                and cmp_seq_lens.numel()
                and seq_lens.numel() == cmp_seq_lens.numel()
                and int(ratio) in (1, 2)
            ):
                cmp_seq_lens = _v41_global_cmp_lens(seq_lens, int(ratio))
            cmp_residual = metadata.attention.cmp_residual
            cmp_topk = self.topology.index_topk
            if cmp_topk not in (512, 1024):
                raise ValueError(f"SparseFlashMla only supports TopK 512 or 1024, got {cmp_topk}")
            cmp_indices = pad_sparse_indices(compressed_indices, cmp_topk)
            # =============================================================
            # ★★★★★★ [V41-UNIQPAD 2026-09-30 18:50] **规避 SMLA 非确定的"均匀重复"填充**。
            #
            # 单卡 + 真实 dump 的输入变异实验（`probes/replay_mutate.py`）结论：
            #     asis            → lse_bit_identical=False, NaN≈4800
            #     idx_full        → True, NaN=0     （重复填满 512）
            #     idx_uni_1024    → True, NaN=0     （topk=1024 均匀重复）
            #     idx_padonly     → False, max|Δ|=3.3e35（只加 -1 padding）
            # ⇒ **算子在"索引槽里有 -1（稀疏）"时会读未初始化内存**；把每行的
            #   已有键**均匀重复**填满槽位后即变确定。
            #
            # 为什么语义精确：每行的**所有**键都被重复同样次数 `k`（`k = floor(K/n)`）
            # ⇒ softmax 的分子分母同乘 `k` ⇒ 输出**数学上不变**。
            # 剩余 `K − k·n < n` 个槽位填 -1（实测有效项 ≥384 时算子确定，见
            # `probes/probe_smla_determinism.py` 的 KEEP 阈值扫描：≥256 确定）。
            #
            # 成本：每行有效项从 `n` 涨到 `k·n ≤ K`（DCP8 下 n≈57~128 ⇒ k=4~8），
            # cmp 的 gather 计算量按比例增加。
            #
            # ★★ 2026-09-30 19:10 实测结论：**本变换不能消除非确定性**。
            #   生产 T=904：`floor` 模式 max|dlse|=0.0116、`ceil` 模式 0.0146
            #   （原样 0.245）—— 显著变小但非零；长上下文 2000/8000 仍失败。
            #   单卡 replay 上 `idx_full`（填满 512）曾显示 bit-identical，但在
            #   生产路径上不可复现 ⇒ 说明触发条件不止"索引里有 -1"这一项。
            #   ⇒ **默认关闭**（`uniqpad=0`），避免白付性能代价。
            # =============================================================
            # ★ 默认 **关闭**：实测 `ceil`/`floor` 都无法消除算子的非确定
            #   （生产 T=900 上 max|dlse| 从 0.245 降到 0.0146 但仍非零，
            #   且长上下文仍然失败）⇒ 不值得付 gather 放大的代价。
            #   保留代码与开关，供算子修复后复核或将来复用。
            _up_mode = _perf_flags().get("uniqpad", "0")
            if (
                _up_mode not in ("0", "")
                and cmp_indices is not None
                and int(cmp_indices.shape[-1]) >= 2
            ):
                # ★ 三种模式（文件开关 `uniqpad`）：
                #   `1`/`floor` = 每个键重复 floor(K/n) 次，剩余填 -1（语义精确，
                #                 但仍有 -1 ⇒ 实测仍可能非确定）
                #   `ceil`      = 重复到**填满 K 个槽位**（0 个 -1）—— 实测这条能把
                #                 确定性问题压掉，代价是最后可能有 1 个键多出现一次
                #   `0`         = 关闭
                _ci2 = cmp_indices.squeeze(1).to(torch.int64)          # [T, K]
                _ok2 = _ci2 >= 0
                _n2 = _ok2.sum(dim=1, keepdim=True).clamp_min(1)        # [T, 1]
                if _up_mode == "ceil":
                    _k2 = torch.div(
                        int(cmp_topk) + _n2 - 1, _n2, rounding_mode="floor"
                    ).clamp_min(1)
                else:
                    _k2 = torch.div(
                        int(cmp_topk), _n2, rounding_mode="floor"
                    ).clamp_min(1)
                _j2 = torch.arange(int(cmp_topk), device=_ci2.device).view(1, -1)
                _pos2 = torch.div(_j2, _k2, rounding_mode="floor")      # [T, K]
                _in2 = _pos2 < _n2
                _gi2 = torch.gather(_ci2, 1, _pos2.clamp(max=int(cmp_topk) - 1))
                cmp_indices = torch.where(
                    _in2, _gi2, torch.full_like(_gi2, -1)
                ).unsqueeze(1).to(cmp_indices.dtype)
                if _perf_flags().get("uniqpad_diag") == "1" and not _is_capturing():
                    try:
                        _neg = int((cmp_indices < 0).sum())
                        print(
                            "[V41-UNIQPAD] mode=%s K=%d rows=%d -1 总数=%d k=[%d,%d] n=[%d,%d]"
                            % (_up_mode, int(cmp_topk), int(_ci2.shape[0]), _neg,
                               int(_k2.min()), int(_k2.max()),
                               int(_n2.min()), int(_n2.max())),
                            flush=True,
                        )
                    except Exception:  # noqa: BLE001
                        pass
        # [V41-KERNDET] 已移到「第一次 SMLA 调用之后」——那里 `sinks` 已在作用域内。
            # =================================================================
            # ★★★ [V41-KVFP 2026-09-30] **KV 内容指纹**（文件驱动 + 锁定层）。
            #
            # 已排除：indexer 键集（IDXFP 实测逐次完全相同）、head 数、归约、缓存、
            # 多流、图、engram、第二次调用。⇒ 只剩两种可能：
            #   (a) SMLA 算子对**相同输入**给出不同输出（算子自身非确定）；
            #   (b) SMLA 读到的 **KV 内容**逐次不同（写侧非确定）。
            # 本探针直接对拍 (b)：对**同一 prompt 的连续请求**打印
            # `ori`（SWA 复制平面）与 `cmp`（压缩分片平面）的**单次快照指纹**。
            #   · 若两份 KV 指纹逐次相同 ⇒ (a) 算子自身，问题在 kernel；
            #   · 若任一不同 ⇒ (b) 写侧，问题在 KV 写入路径。
            # 指纹 = sum(fp32) + 前若干元素的 min/max，全部在**一次 `.cpu()`** 后算。
            # =================================================================
            _kvfp_layer = int(__import__("os").environ.get("V41_DCP_KVFP_LAYER", "-1"))
            if (
                _perf_flags().get("kvfp") == "1"
                and (_kvfp_layer < 0 or int(self.role.layer_idx) == _kvfp_layer)
                and not _is_capturing()
                and int(cmp_indices.shape[0]) > 1
            ):
                try:
                    def _fp(t):
                        # ★ 每个张量**只读一次**（单次 device 求和 + 单次标量回传）——
                        #   这是唯一安全的模式。旧版采样 `[:4096]/[-4096:]` 恰好落在
                        #   零区，指纹恒为 0，**无信息量**（踩过）。
                        if t is None:
                            return "None"
                        n = int(t.numel())
                        ssum = float(t.detach().to(torch.float32).sum())      # 单读
                        nz = int((t.detach() != 0).sum())                    # 单读（另一个量）
                        return "n=%d sum=%.8g nz=%d" % (n, ssum, nz)
                    _ori_kv = attn.dsa_attn.swa_cache_layer.kv_cache[0]
                    print(
                        "[V41-KVFP] rank=%d layer=%d T=%d ori{%s} cmp{%s}"
                        % (_v41_dcp_rank(), int(self.role.layer_idx),
                           int(cmp_indices.shape[0]), _fp(_ori_kv), _fp(source_cache)),
                        flush=True,
                    )
                except Exception as _e:  # noqa: BLE001
                    print("[V41-KVFP] rank=%d 探针自身失败：%r" % (_v41_dcp_rank(), _e), flush=True)
            # =================================================================
            # ★★★ [V41-IDXFP 2026-09-30] **indexer 键集指纹**（文件驱动 + 锁定层）。
            #
            # 为什么需要：已排除合并/归约/缓存/多流/图/engram/第二次调用之后，
            # 唯一还没验的是"**第一次 SMLA 的输入本身在 DCP8 下逐次不同**"。
            # 它的三个输入里只有 `cmp_sparse_indices`（本 rank 的 top-k，
            # 经 DCP remap）带有**选择性**—— 若 indexer 在并列分数上选得不稳定，
            # 键集就会逐次变化 ⇒ `lse` 逐次变化 ⇒ 边界 prompt 翻转。
            #
            # 判据：**同一 prompt、同一层的连续请求**，指纹若不同 ⇒ 锁定 indexer。
            # 指纹用**一次 `.cpu()` 快照**后在 CPU 上算（避免多次读撕裂，见 §6o）。
            # 门：只在**非 capture** + **真实 prefill**（T>1）+ 指定层 时打印。
            # =================================================================
            _idxfp_layer = int(__import__("os").environ.get("V41_DCP_IDXFP_LAYER", "-1"))
            if (
                _perf_flags().get("idxfp") == "1"
                and (_idxfp_layer < 0 or int(self.role.layer_idx) == _idxfp_layer)
                and not _is_capturing()
                and cmp_indices is not None
                and int(cmp_indices.shape[0]) > 1
            ):
                try:
                    _ci_cpu = cmp_indices.detach().to(torch.int64).cpu()
                    _valid = _ci_cpu[_ci_cpu >= 0]
                    print(
                        "[V41-IDXFP] rank=%d layer=%d T=%d rows=%d n_valid=%d n_neg1=%d "
                        "sum=%d max=%d min_valid=%s head4=%s"
                        % (
                            _v41_dcp_rank(), int(self.role.layer_idx), int(cmp_indices.shape[0]),
                            int(_ci_cpu.shape[0]), int(_valid.numel()),
                            int((_ci_cpu < 0).sum()),
                            int(_valid.sum()) if _valid.numel() else -1,
                            int(_valid.max()) if _valid.numel() else -1,
                            int(_valid.min()) if _valid.numel() else -1,
                            _ci_cpu[0, 0, :4].tolist() if _ci_cpu.ndim == 3 else _ci_cpu[0, :4].tolist(),
                        ),
                        flush=True,
                    )
                except Exception as _e:  # noqa: BLE001
                    print("[V41-IDXFP] rank=%d 探针自身失败：%r" % (_v41_dcp_rank(), _e), flush=True)
            # =================================================================
            # =================================================================
            # [V41-DCP-DIAG] 一次性诊断（capture 安全版）。
            #
            # ★ 必须满足三个前提，否则会**起服就崩**：
            #   1. 不能在 ACL graph capture 区内做任何 host 同步
            #      （`int()`/`.item()`/`.cpu()`/布尔索引都会同步或降级成
            #       `aclnnNonzeroV2` —— 实测该算子在内核 warmup 阶段直接报
            #       inner error 让 8 个 worker 全挂）。
            #   2. 不能对 device 张量做布尔索引（`x[x>=0]` → nonzero）。
            #   3. 要避开 warmup（warmup 的 seq_lens 全是 1，没有信息量）。
            # 默认关闭；`V41_DCP_IDX_DIAG=1` 打开。
            # =================================================================
            import os as _os

            if (
                _os.environ.get("V41_DCP_IDX_DIAG") == "1"
                and self.role.layer_idx == 2
                and not _is_capturing()
            ):
                try:
                    _lmax = int(seq_lens.max())
                except Exception:  # noqa: BLE001
                    _lmax = -1
                if _lmax > 129:
                    _ci = cmp_indices
                    print(
                        "[V41-IDX] rank=%d L=%s cmp_len=%s G=%s "
                        "idx_shape=%s idx_max=%s cmp_len_max=%s resid=%s "
                        "op_ratio=%s cmp_mask=%s"
                        % (
                            _v41_dcp_rank(),
                            seq_lens.detach().cpu().tolist()[:3],
                            cmp_seq_lens.detach().cpu().tolist()[:3],
                            (seq_lens // ratio).detach().cpu().tolist()[:3],
                            tuple(_ci.shape),
                            int(_ci.max()) if _ci is not None and _ci.numel() else -1,
                            int(cmp_seq_lens.max()) if cmp_seq_lens.numel() else -1,
                            int(cmp_residual.max()) if cmp_residual is not None and cmp_residual.numel() else -1,
                            ratio,
                            3 if has_compressed else 0,
                        ),
                        flush=True,
                    )
        operator_metadata = metadata.attention if has_compressed else metadata.swa
        op_metadata = operator_metadata.smla_metadata
        if op_metadata is None:
            raise RuntimeError(f"V4.1 ratio-{ratio} SMLA metadata was not built")
        wait_for_device_metadata(
            DeviceMetadataStage.ATTENTION,
            id(op_metadata),
        )
        # =====================================================================
        # [V41-DCP 2026-09-29] 合并路径的开关与参数。
        #
        # 只在「有压缩（ratio∈{1,2}）」的层上走跨 rank 合并：
        #   · ratio=0 的层只有 ori，滑窗平面每 rank 全量复制 ⇒ 本地算出的就是
        #     完整正确的滑窗注意力，**不能**再去合并（否则滑窗被计 dcp 次）。
        #   · ratio>0 的层：ori 由 rank 0 独占（见 metadata 的 `has_ori_kv`），
        #     各 rank 只带自己的 cmp 分片，合并后 A 恰好计一次。
        #
        # sink 同理必须**只计一次**：只在 rank 0 传真值，其余 rank 传一个
        # 在 fp32 里 exp() 下溢到 0 的大负数（离线夹具回归：每 rank 都传真值
        # 时 relout 0.63，只计一次时 9.5e-16）。
        # =====================================================================
        dcp_active = _v41_dcp_on() and has_compressed
        # =====================================================================
        # [V41-DCP 2026-09-29] ★★ 必须先沿 head 维 gather q，再做逐元素归约。
        # 原因见 `_v41_dcp_gather_heads`：TP 切的是 head，不 gather 就等于把
        # 不同 head 的部分结果相加。合并后本 rank 只保留自己那段 head（o_proj 是
        # TP 分片的），所以末尾要切回去。
        # =====================================================================
        _pf = _perf_flags() if dcp_active else {}
        import time as _time
        _t = _time_mark('t0_init', _time.perf_counter()) if dcp_active else _time.perf_counter()
        # ★ q 的 head all_gather 已在 `forward` 里、**indexer 之前**完成
        #   （`_prepare_q_for_dcp`），这里只取回本 rank 的 head 数。
        if dcp_active:
            _local_heads = int(_q_local_heads)
            _hpr = int(_q_hpr)
            _t = _time_mark('gather_q(已在forward完成)', _t)
        else:
            _local_heads = int(q.shape[1])
            _hpr = 0
        _owner = _v41_dcp_ori_owner() if dcp_active else "all"
        # ★★ 2026-09-29 修 bug：原写法 `_owner == "seqused0" or rank == 0`
        # 让 **所有** rank 的 `_is_ori_owner` 都为 True ⇒ 下面的 zeros_like 永不执行
        # ⇒ 8 个 rank 各自都算了整份 ori 窗口 ⇒ ori 被重复计 8 次。
        # 由子代理 `dcp_op_unknowns` 逐行审计发现（overlay 675-684 行）。
        # 正确语义：ori 只由 rank 0 携带。
        _is_ori_owner = (not dcp_active) or (_v41_dcp_rank() == 0)
        # sink 也要随 head 一起 gather（算子按全量 head 收参数）。
        # ★ 并且**只能计一次**：gather 后若 8 个 rank 都带真值 sink，
        #   LSE 合并会把 `exp(sink)` 计 8 次（离线夹具回归：每 rank 都带
        #   relout 0.63，只计一次 9.5e-16）。所以只有 rank 0 用真值，
        #   其余 rank 填一个在 fp32 里 exp() 下溢到 0 的大负数。
        # ★★ [V41-PERF 2026-09-29] `sinks` 是**静态量**（`attn.attn_sink` 是注册参数，
        # 全程不变），但原实现**每层每步**都重做一次 head 维 all-gather + full_like。
        # 2-chip 图级实测：每层每多一个 collective ≈ +61~71 µs/step
        # ⇒ 这一处白送 ~2.4 ms/step。改成按 attn 对象缓存：
        #   · 第一次调用（含 capture 期）建好并挂到 attn 上；
        #   · 之后 replay 直接复用**同一块地址** —— 对图捕获反而更安全。
        if dcp_active and attn.attn_sink is not None:
            sinks = None if _DCP_NO_ATTN_CACHE else getattr(attn, "_v41_dcp_sinks_cache", None)
            if sinks is None or int(sinks.shape[0]) != int(attn.attn_sink.shape[0]) * _v41_dcp_group().world_size:
                sinks = _v41_dcp_gather_1d(attn.attn_sink)
                if _v41_dcp_rank() != 0:
                    sinks = torch.full_like(sinks, -1e30)
                if not _DCP_NO_ATTN_CACHE:
                    attn._v41_dcp_sinks_cache = sinks
            # =============================================================
            # ★★★★★★ [V41-SINKAB 2026-09-30 15:05] **sink 假设的 A/B 开关**。
            #
            # 现场（run dcpcap_0930_144713，T=904）：**只有 rank0 的 `lse` 出现
            # NaN**（finite=52360/57856），另外 7 个 rank 全部正常
            # （`lse max=10.34`、`w max≈9~44`）；rank0 的 `w` 撞上
            # `clamp(max=60)`（`max=1.14e26`）。而 rank0 与其余 rank 的**唯一差别**
            # 就是 sink 用真值（见上：`_v41_dcp_rank() != 0` 时才填 -1e30）。
            # DCP1 在同样长度 900/1024/1500/2000 上 **8/8 全对**，所以这是 DCP8 特有。
            #
            #   · `sinkoff=1` ⇒ 所有 rank 一律 -1e30（sink 完全不参与）
            #   · `sinkzero=1` ⇒ 所有 rank 一律 0.0
            # 若 `sinkoff` 让 NaN 消失 ⇒ 是 sink 数值在长上下文下把 lse 顶爆。
            # =============================================================
            _sinkmode = _perf_flags().get("sinkmode", "")
            if _sinkmode == "off":
                sinks = torch.full_like(sinks, -1e30)
            elif _sinkmode == "zero":
                sinks = torch.zeros_like(sinks)
        else:
            sinks = attn.attn_sink if not dcp_active else None
        if dcp_active and _hpr > 0 and sinks is not None and int(sinks.shape[0]) > int(q.shape[1]):
            sinks = sinks[: int(q.shape[1])].contiguous()
        # ★★ [V41-DCP 2026-09-29 · 路线 B] 所有 rank 都带**真实的 ori**。
        #
        # 内核不允许把 ori 从某个 rank 上摘掉（四条路全堵，见下），所以改为
        # 「8 个 rank 都算 `真实 ori ⊕ 各自 cmp`」，再在合并时**精确**减掉多出的
        # `(dcp−1)` 份 ori（第二次纯 ori 调用提供 `e^{L_ori}` 与 `e^{L_ori}·O_ori`）。
        #
        # 四条被堵的路（都有实测/源证）：
        #   1. `ori_kv` 是 op 层硬必填（传 None ⇒ `tensor of oriKv is nullptr`）；
        #   2. `ori_win_left` 被硬绑 127、`ori_mask_mode` 必须 4（窗口收不成空）；
        #   3. `ori_sparse_indices` 是 A5-only；
        #   4. `seqused_ori_kv = 0` 会**连 cmp 一起清零** ——
        #      `sparse_flash_mla_csa_kernel.h:414-421`：
        #        // 行无效通过ori部分判断, ori部分如果有行无效那么ori和cmp都有
        #        if (s1EndIdx < -(actOriS2Size - actS1Size)) {
        #            actOriS2Size = 0; actCmpS2Size = 0; return; }
        #      实测 run `dcpcap_0929_172847`：rank 1-7 的 LSE 与输出为**精确 0.0**。
        _ori_seqused = seq_lens
        _ori_bt = ori_block_table
        # =====================================================================
        # ★★★★★★ [V41-PRESMLA-SYNC 2026-09-30 17:55] **SMLA 调用前的流同步**。
        #
        # 为什么这一刀最关键：`kdet` 实测（run dcpcap_0930_171157）证明
        # **同层、同输入、同一次 forward 内连调两次 SMLA，结果不同**
        # （`lse_bit_identical=False max_abs_diff≈0.19`）—— 这是本轮唯一
        # 无法用"输入不同"解释的观察。
        #
        # 但必须排除一个重要的自证陷阱：我加的所有 KV 内容探针
        # （`kvdata` / `idxkvfp` / `oridata`）都在 SMLA **之前**读张量
        # （`.cpu()`）⇒ 它们**会强制同步**，从而可能**掩盖**真实的写-读竞态。
        # 也就是说"探针说输入干净"并不等于"SMLA 看到的是干净的"。
        #
        # 本开关在第一次 SMLA 调用**之前**插一次设备同步：
        #   · 若第 2 个请求变正确 ⇒ **KV 写入流与 SMLA 读之间存在竞态**
        #     （slot mapping/scatter 是异步的，SMLA 抢跑读到旧块/半写块）
        #     ⇒ 修复方向明确：在 SMLA 前对写入流加 event 依赖；
        #   · 若无变化 ⇒ 算子内部状态问题（按 §13 的单卡结论处理）。
        # 只在非 capture 下用（capture 区 host 同步会崩）。
        # =====================================================================
        if _perf_flags().get("presmla_sync") == "1" and not _is_capturing():
            try:
                torch.npu.synchronize()
            except Exception:  # noqa: BLE001
                pass
        # =====================================================================
        # ★★★★★★ [V41-DENSECMP 2026-09-30 18:00] **ratio=1 层改 dense 的尝试：已证伪**。
        #
        # 设想：对 ratio=1 层，本 rank 的 `long_kv` 只有 `seqused_cmp_kv=128` 行，
        # 若"本地候选去重后就是这 128 行"，则 dense（不传 `cmp_sparse_indices`）
        # 应与稀疏等价，且能绕开非确定的稀疏 gather 路径。
        #
        # **实测证伪**（run dcpcap_0930_174046，`dense_cmp=1`）：T=900 连发 4 次
        # **全部错误**（连第一个请求也错，而稀疏路径下第一个请求是对的）。
        #
        # 原因（纯组合性质）：`interleave=32` 的交错分片下，rank r 的第 k 行对应
        # 全局位置 `super(k)`，而算子的 `cmp_mask_mode=3`（因果）用的是**行号**
        # `k ≤ t` ⇒ 与 `super(k) ≤ q_pos` 不等价 ⇒ query 会看到**未来的键**。
        # 只有当分片改为**连续**（rank r 拥有 [r·L/8, (r+1)·L/8)）时 dense 才正确，
        # 而平台把 `cp_kv_cache_interleave_size` 固定为 32（`platform.py` 的 sparse
        # 强制）⇒ 此路不通。保留本节作为负结果。
        # =====================================================================
        # ★★★★★★ [V41-ORIDATA 2026-09-30 17:22] **ori（SWA 复制面）本请求行的指纹**。
        #
        # 这是最后一个还没按"本请求实际读写的行"测过的输入。
        # `[V41-KVFP]` 的整平面 `sum=nan` 只能说明**未写区域**有 NaN（整面 4.4 亿元素
        # 里绝大多数从未被写过），不能证明本请求读到的行有问题。
        #
        # 本探针按 `pos → (block_table[0, pos//B], pos%B)` 精确取本请求
        # **滑窗覆盖的那一段**（最后 `window` 个位置），一次 `.cpu()` 快照后统计
        # nonfinite / absmax / 全 0 行数 / sum。
        # 判据：若 req#1（对）与 req#2（错）这两段指纹不同 ⇒ **SWA 写路径**是根因；
        #       若相同 ⇒ ori 也干净，SMLA 的四个输入（q / ori / cmp / 索引）全部一致。
        # =====================================================================
        if (
            _perf_flags().get("oridata") == "1"
            and not _is_capturing()
            and seq_lens is not None and seq_lens.numel() > 0
            and int(seq_lens.max()) > 1
        ):
            try:
                _swa_c = attn.dsa_attn.swa_cache_layer.kv_cache[0]
                _Tc = int(seq_lens[0].to(torch.int64).cpu())
                _Wc = min(int(attn.window_size), _Tc)
                _bsc = int(_swa_c.shape[1])
                _posc = torch.arange(
                    max(0, _Tc - _Wc), _Tc, dtype=torch.int64, device=_swa_c.device
                )
                _blkc = ori_block_table[0, _posc // _bsc].to(torch.int64)
                _rowc = _posc % _bsc
                _selc = _swa_c[_blkc, _rowc].to(torch.float32).cpu()
                print(
                    "[V41-ORIDATA] rank=%d layer=%d ratio=%d T=%d window=%d rows=%d "
                    "nonfinite=%d absmax=%.6g zero_rows=%d sum=%.8g blk[:4]=%s"
                    % (_v41_dcp_rank(), int(self.role.layer_idx),
                       int(self.role.compress_ratio), _Tc, _Wc, int(_blkc.numel()),
                       int((~torch.isfinite(_selc)).sum()),
                       float(_selc.abs().max()),
                       int((_selc.abs().sum(dim=tuple(range(1, _selc.ndim))) == 0).sum()),
                       float(_selc.sum()), _blkc[:4].tolist()),
                    flush=True,
                )
            except Exception as _e:  # noqa: BLE001
                print("[V41-ORIDATA] rank=%d 失败：%r" % (_v41_dcp_rank(), _e), flush=True)
        # ★ [V41-NANPROBE 2026-09-30] 直接量 rank0 的 cmp 索引是否越界：
        #   SMLA 用 `blkIdx = idx // paCmpBlockSize` 去查 `cmp_block_table`，
        #   若 `idx` 超过本 rank 的 long_kv 行数就会读到别的块 ⇒ 分数 NaN。
        if _perf_flags().get("nanprobe") == "1" and not _is_capturing() and dcp_active:
            try:
                _ci = cmp_indices
                _p = torch.stack([
                    _ci.max().to(torch.float32).reshape(1),
                    (cmp_seq_lens.max() if cmp_seq_lens is not None and cmp_seq_lens.numel()
                     else torch.tensor(-1, device=_ci.device)).to(torch.float32).reshape(1),
                    torch.tensor(float(_ci.shape[-1]), device=_ci.device).reshape(1),
                    torch.tensor(float(int(cmp_block_table.shape[1])), device=_ci.device).reshape(1),
                    torch.tensor(float(int(cmp_block_table.shape[0])), device=_ci.device).reshape(1),
                    torch.tensor(float(int(ori_block_table.shape[1])), device=_ci.device).reshape(1),
                ]).cpu()
                print(
                    "[V41-NANPROBE] rank=%d layer=%d T=%d idx_max=%.0f cmp_seq_max=%.0f K=%.0f "
                    "cmp_bt=[%.0f rows x %.0f cols] ori_bt_cols=%.0f"
                    % (_v41_dcp_rank(), int(self.role.layer_idx),
                       int(seq_lens.max()) if seq_lens is not None and seq_lens.numel() else -1,
                       float(_p[0]), float(_p[1]), float(_p[2]),
                       float(_p[4]), float(_p[3]), float(_p[5])),
                    flush=True,
                )
            except Exception as _e:  # noqa: BLE001
                print("[V41-NANPROBE] rank=%d 失败：%r" % (_v41_dcp_rank(), _e), flush=True)
        # =====================================================================
        # ★★★★★★ [V41-DUMPREPLAY 2026-09-30 18:35] **生产输入 dump（供单卡重放）**。
        #
        # 目的：把**生产上真正踩到非确定的那一组张量**原样落盘，交给算子团队在
        # 单卡上离线重放 —— 这比"同类但不同源"的最小复现（§13/§15）有用得多：
        # 单卡的触发条件（cmp 值域跨 ≥3 页）与生产（T≥560）**不一致**，说明算子
        # 在多条路径上读未初始化内存，需要一个真实用例来对齐。
        #
        # 落盘内容（只落**本请求实际会读到的页**，并把块表重映射到 1..N，
        # 避免搬运 3.5 GB 的整池）：
        #   q / cmp_indices / ori_bt / cmp_bt / cu_seqlens_q / seqused_*
        #   / sinks / op_metadata / 标量参数 / 用到的 ori 与 long_kv 页
        # 触发条件：`dumpdir=<容器内目录>` 文件开关 + 非 capture + 首个合格调用。
        # =====================================================================
        _dumpdir = _perf_flags().get("dumpdir")
        # ★★★ [V41-DUMP-LAYER 2026-09-30 22:05] 支持**指定层**与**多次落盘**：
        #   `dumplayer=<n>`  只在该层落盘（多个用逗号分隔）
        #   `dumpsuffix=<s>` 文件名后缀，便于同一次会话里分次落盘（例如 good/bad）
        #   done 键改为 (layer, suffix) ⇒ 同一会话可抓多个层/多次请求。
        _dl_raw = _perf_flags().get("dumplayer", "")
        _dump_layers = {int(v) for v in _dl_raw.replace(" ", "").split(",") if v.strip().lstrip("-").isdigit()}
        _dsfx = _perf_flags().get("dumpsuffix", "")
        # ★ [V41-DUMP-RANK] 只 dump 指定 rank（8 个 rank 各 ~60MB ⇒ 限一个省 8×）
        _dr_raw = _perf_flags().get("dumprank", "")
        _dump_ranks = {int(v) for v in _dr_raw.replace(" ", "").split(",") if v.strip().isdigit()}
        _dkey = (int(self.role.layer_idx), _dsfx)
        if (
            _dumpdir
            and not _is_capturing()
            and _dkey not in _DCP_DUMPED
            and (not _dump_layers or int(self.role.layer_idx) in _dump_layers)
            and (not _dump_ranks or int(_v41_dcp_rank()) in _dump_ranks)
            # ★ 必须门在 **有压缩** 的层上：ratio=0 的滑窗层没有 `cmp_*` 张量
            #   （实测第一次落盘就撞上 layer 1，全是 None）。
            # ★★ [V41-DUMP-LAYER-2] 放开到 ratio∈{1,2}：**第一个压缩层是 layer 2
            #   （ratio=2）**，要抓"最早的分叉点"就必须能在这里落盘。
            and has_compressed
            and int(ratio) >= 1
            and seq_lens is not None
            and int(seq_lens.max()) > 800
        ):
            try:
                import os as _osd

                _osd.makedirs(_dumpdir, exist_ok=True)
                _Td = int(seq_lens.max())
                _swad = attn.dsa_attn.swa_cache_layer.kv_cache[0]
                # 用到的 ori 页（滑窗覆盖的最后 WIN 个位置）与 cmp 页
                _Wsd = min(int(attn.window_size), _Td)
                _pd = torch.arange(max(0, _Td - _Wsd), _Td, dtype=torch.int64)
                _obt = ori_block_table[0].detach().to(torch.int64).cpu()
                _cbt = cmp_block_table[0].detach().to(torch.int64).cpu()
                _cmax = int(cmp_seq_lens[0].to(torch.int64).cpu()) if cmp_seq_lens is not None and cmp_seq_lens.numel() else 0
                # ★★★ [V41-DUMP-FIX 2026-09-30 19:20] **必须覆盖全部 query 行用到的页**。
                #   原实现只取了"最后一个滑窗"（`_pd` = 末尾 window 个位置）用到的页，
                #   但 `seqused_ori_kv = T` ⇒ query 0..T-1 各自需要 `[t-127, t]` 的窗口
                #   ⇒ 需要 **0..ceil(T/128)-1 共 8 页**。只存 2 页会让前 776 行的
                #   ori 块表列落到 null 页 ⇒ 单卡重放**必然出 NaN**（伪影）。
                #   实测症状：NaN 只出现在最后 ~104 个 token（800..903），因为只有
                #   它们的窗口完全落在已保存的两页里。
                # =========================================================
                # ★★★ [V41-DUMP-FAITHFUL-2 2026-09-30 21:50] **忠实 dump**。
                #
                # 上一版 dump 有三处会让单卡重放与生产**不等价**（实测 DCP1 生产正确、
                # 重放却出 NaN）：
                #   ① 页号被**重映射**成 1..N  —— 内核读到的任何列都变成"我以为的页"，
                #      列号假设一旦不符就读错页；
                #   ② 用 `_swad[cols]` 取页 ⇒ **丢掉原始 stride(0)**：生产 cache 的
                #      stride(0) 是**槽位 stride**（可远大于页大小），而重建张量是紧凑的
                #      ⇒ host tiling 算出的 KvStride0 不同 ⇒ 地址整体偏移；
                #   ③ 只按我算的列数取页 ⇒ 内核可能读更靠后的列。
                # 现在改为：**块表原值不动**；页池大小 = max(用到的页号)+1；
                # **保持各平面的原始 stride(0)**，把用到的页按**原始下标**填进去。
                # =========================================================
                _orow = int(_swad.shape[1]); _crow = int(source_cache.shape[1])
                _ostr0 = int(_swad.stride(0)); _cstr0 = int(source_cache.stride(0))
                _obt_list = [int(v) for v in _obt.tolist()]
                _cbt_list = [int(v) for v in _cbt.tolist()]
                _used_o = sorted({v for v in _obt_list if v > 0})
                _used_c = sorted({v for v in _cbt_list if v > 0})
                # ★★★★★ [V41-DUMP-NULL 2026-09-30 23:55] **必须把"页 0"也搬进 dump**。
                #
                # 算子源码（`arch22/sparse_flash_mla_swa_block_vector.h:265-271`）的 ori 寻址是
                #     blockTableIdx = logicalIdx / paOriBlockSize
                #     idInBlockTable = oriBlockTable[bIdx*oriMaxBlockNumPerBatch + blockTableIdx]
                #     offset = idInBlockTable*oriKvStride0 + n2IdxReal*headDim*paOriBlockSize
                #              + (logicalIdx % paOriBlockSize)*headDim
                # ⇒ **块表里的 0 不是"跳过"，而是"块 0"**，直接落到池的页 0。
                # 生产 DCP8 的 ori 块表**只有第 0 列非零** ⇒ 位置 ≥128 的逻辑键
                # （即除了前 128 个 token 以外的所有窗口读取，以及**所有写入**）
                # 全部落到页 0。页 0 此前**没有**被 dump ⇒ 单卡重放读到的是一块
                # 全 0 的人造内存（所以重放"干净"），与生产不等价。
                # `dumpnull=1` 把页 0 一并落盘，供"页 0 是否承载本请求数据"的判据。
                if _perf_flags().get("dumpnull") == "1":
                    _used_o = sorted(set(_used_o) | {0})
                    _used_c = sorted(set(_used_c) | {0})
                _max_o = _used_o[-1] if _used_o else 0
                _max_c = _used_c[-1] if _used_c else 0

                def _strided_pool(src, used, maxpg, rows, str0):
                    """按**原始 stride(0)** 建页池，并把用到的页填到**原始下标**上。"""
                    if maxpg <= 0 or not used:
                        return torch.zeros(0)
                    stride = (str0,) + tuple(src.stride()[1:])
                    shape = (maxpg + 1, rows) + tuple(src.shape[2:])
                    buf = torch.zeros((maxpg + 1) * str0, dtype=src.dtype)
                    pool = torch.as_strided(buf, shape, stride)
                    for pg in used:
                        pool[pg].copy_(src[pg].detach().cpu().reshape(pool[pg].shape))
                    return pool

                _ori_pages = _strided_pool(_swad, _used_o, _max_o, _orow, _ostr0)
                _cmp_pages = _strided_pool(source_cache, _used_c, _max_c, _crow, _cstr0)
                # 块表**原值**（不再重映射）
                _obt_new = _obt.to(torch.int32).clone().view(1, -1)
                _cbt_new = _cbt.to(torch.int32).clone().view(1, -1)
                def _c(x):
                    """None 安全 + 统一转 CPU（运维包要能在单卡离线重放）。"""
                    return None if x is None else x.detach().cpu()

                _payload = {
                    # ★ 原始块表前 16 列（**未重映射**）——用于事后判断 dump 是否忠实
                    "ori_bt_raw16": _obt[:16].to(torch.int32).clone(),
                    "cmp_bt_raw16": _cbt[:16].to(torch.int32).clone(),
                    "q": _c(q),
                    "cmp_indices": _c(cmp_indices),
                    "ori_block_table": _obt_new,
                    "cmp_block_table": _cbt_new,
                    "cu_seqlens_q": _c(query_start_loc),
                    "seqused_ori_kv": _c(seq_lens),
                    "seqused_cmp_kv": _c(cmp_seq_lens),
                    "cmp_residual_kv": _c(cmp_residual),
                    "sinks": _c(sinks),
                    "metadata": _c(op_metadata),
                    "ori_pages": _ori_pages,
                    "cmp_pages": _cmp_pages,
                    "ori_page_rows": int(_swad.shape[1]),
                    "cmp_page_rows": int(source_cache.shape[1]),
                    # ★ [V41-DUMP-FAITHFUL-2] 原始 stride/形状 —— 重放侧必须核对，
                    #   否则 KvStride0 不同会让地址整体偏移（见上）。
                    "ori_stride0": int(_swad.stride(0)),
                    "cmp_stride0": int(source_cache.stride(0)),
                    "ori_shape": tuple(int(x) for x in _swad.shape),
                    "cmp_shape": tuple(int(x) for x in source_cache.shape),
                    "scalars": {
                        "num_heads_q": int(q.shape[1]), "head_dim": int(q.shape[-1]),
                        "softmax_scale": float(attn.softmax_scale),
                        "cmp_ratio": int(ratio),
                        "ori_mask_mode": 4,
                        "cmp_mask_mode": 3 if has_compressed else 0,
                        "ori_win_left": int(attn.window_size) - 1,
                        "ori_win_right": 0,
                        "topk_value_mode": 1,
                        "layer_idx": int(self.role.layer_idx),
                        "dcp_rank": int(_v41_dcp_rank()),
                        "has_cmp_kv": bool(has_compressed),
                        # ★ `common` 不在 `_native_attention` 作用域（踩过一次 NameError）
                        #   ⇒ 从 `query_start_loc` / `seq_lens` 推导。
                        "max_seqlen_q": int(
                            (query_start_loc[1:] - query_start_loc[:-1]).max()
                        ) if query_start_loc.numel() > 1 else _Td,
                        "max_seqlen_ori_kv": int(seq_lens.max()),
                    },
                }
                _fn = _osd.path.join(_dumpdir, "l%d_T%d_rank%d%s.pt" % (
                    int(self.role.layer_idx), _Td, int(_v41_dcp_rank()),
                    ("_" + _dsfx) if _dsfx else ""))
                torch.save(_payload, _fn)
                _DCP_DUMPED[_dkey] = True
                _DCP_DUMPED["done"] = True
                print(
                    "[V41-DUMPREPLAY] 已落盘 %s | q=%s idx=%s | 页池 ori=%d(至max %d) cmp=%d(至max %d) | "
                    "stride0 ori=%d cmp=%d | 原始 ori_bt[:12]=%s cmp_bt[:12]=%s"
                    % (_fn, tuple(q.shape), tuple(cmp_indices.shape),
                       int(_ori_pages.shape[0]) if _ori_pages.numel() else 0, _max_o,
                       int(_cmp_pages.shape[0]) if _cmp_pages.numel() else 0, _max_c,
                       _ostr0, _cstr0,
                       _obt_list[:12], _cbt_list[:12]),
                    flush=True,
                )
            except Exception as _e:  # noqa: BLE001
                _DCP_DUMPED[_dkey] = True
                _DCP_DUMPED["done"] = True
                print(
                    "[V41-DUMPREPLAY] 失败：%r | q=%s idx=%s obt=%s cbt=%s qsl=%s sl=%s "
                    "csl=%s resid=%s sinks=%s meta=%s"
                    % (_e, q is not None, cmp_indices is not None, ori_block_table is not None,
                       cmp_block_table is not None, query_start_loc is not None,
                       seq_lens is not None, cmp_seq_lens is not None, cmp_residual is not None,
                       sinks is not None, op_metadata is not None),
                    flush=True,
                )
        output, softmax_lse = torch.ops._C_ascend.npu_sparse_flash_mla(
            q,
            ori_kv=attn.dsa_attn.swa_cache_layer.kv_cache[0],
            cmp_kv=source_cache,
            cmp_sparse_indices=cmp_indices,
            ori_block_table=_ori_bt,
            cmp_block_table=cmp_block_table,
            cu_seqlens_q=query_start_loc,
            seqused_ori_kv=_ori_seqused,
            seqused_cmp_kv=cmp_seq_lens,
            cmp_residual_kv=cmp_residual,
            sinks=sinks,
            metadata=op_metadata,
            softmax_scale=attn.softmax_scale,
            cmp_ratio=ratio,
            ori_mask_mode=4,
            cmp_mask_mode=3 if has_compressed else 0,
            ori_win_left=attn.window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_BBND",
            topk_value_mode=1,
            return_softmax_lse=dcp_active,
        )
        # =====================================================================
        # [V41-RAW] 第一次 SMLA 的**原始输出**诊断（T 定向）。
        # 为什么需要：`rank0_pure` 判别实验（数学上应精确等于 rank0 的 partial）
        # 在 T=16 上**仍然是均匀分布**（= 最终 hidden 为 0）⇒ 说明问题在
        # **rank0 自己的注意力输出**，而不是跨 rank 合并。这里直接量它：
        #   · `output.abs().max()` —— 若 ≈0 就是"注意力没算出来"
        #   · `softmax_lse` 的 finite/-inf 计数
        #   · `cmp_indices` 的 -1 个数（= 没有 cmp 键被选中）
        #   · `cmp_seq_lens`（本 rank 的压缩行数）
        # =====================================================================
        if (
            dcp_active
            and _DCP_RAWD_ON
            and not _is_capturing()
            and _DCP_RAWD["n"] < _DCP_RAWD_LIMIT
            and metadata.swa is not None
            # ★ 用 **真实序列长度** 而不是 `num_actual_tokens`：后者是静态容量
            #   （实测 warmup 时恒为 16）⇒ 会把 warmup 当成真实请求记进日志。
            and seq_lens.numel() > 0
            and int(seq_lens.max()) > 1
            and _dcp_diag_t_ok(int(seq_lens.max()))
        ):
            _DCP_RAWD["n"] += 1
            try:
                _ol = softmax_lse.to(torch.float32) if softmax_lse is not None else None
                _neg = -1
                _tot = -1
                if cmp_indices is not None:
                    _ci = cmp_indices
                    _neg = int((_ci < 0).sum())
                    _tot = int(_ci.numel())
                print(
                    "[V41-RAW] rank=%d T=%d out[absmax=%.6g mean=%.6g] "
                    "lse[finite=%d/%d inf=%d min=%.4f max=%.4f] cmp_idx[-1=%d/%d max=%s] "
                    "cmp_seq=%s seq=%s"
                    % (
                        _v41_dcp_rank(), int(seq_lens.max()),
                        float(output.abs().max()), float(output.abs().mean()),
                        int(torch.isfinite(_ol).sum()) if _ol is not None else -1,
                        int(_ol.numel()) if _ol is not None else -1,
                        int((~torch.isfinite(_ol)).sum()) if _ol is not None else -1,
                        float(_ol.min()) if _ol is not None else float("nan"),
                        float(_ol.max()) if _ol is not None else float("nan"),
                        _neg, _tot,
                        int(cmp_indices.max()) if (cmp_indices is not None and cmp_indices.numel()) else -1,
                        cmp_seq_lens[:4].detach().to(torch.int64).cpu().tolist() if cmp_seq_lens is not None else [],
                        seq_lens[:4].detach().to(torch.int64).cpu().tolist(),
                    ),
                    flush=True,
                )
            except Exception as _e:  # noqa: BLE001
                print("[V41-RAW] rank=%d 诊断自身失败：%r" % (_v41_dcp_rank(), _e), flush=True)
        # =====================================================================
        # ★★★★ [V41-KERNDET 2026-09-30] **算子自身确定性**判别。
        # 在同一层内、用**完全相同的输入**再调一次 SMLA，逐位比 `lse`/`output`。
        #   · 两次不同 ⇒ **算子自身非确定**（读未初始化 workspace / 原子累加 /
        #     split-K 归约顺序），与 KV 内容无关（同一次 forward 内 KV 不可能变）；
        #   · 两次相同 ⇒ 算子确定 ⇒ 只能怪**跨请求的 KV 内容**（写侧）。
        # 已排除到这一步：indexer 键集（IDXFP 逐次相同）、head 数、归约顺序、
        # 三处 attn 缓存、多流、图捕获、engram、第二次纯 ori 调用。
        # 位置：必须在**第一次调用之后**（此刻 `sinks`/`cmp_indices` 都在作用域内；
        # 放早了会 `UnboundLocalError: sinks` —— 踩过两次）。
        # =====================================================================
        _kd_layer = int(__import__("os").environ.get("V41_DCP_KDET_LAYER", "-1"))
        if (
            _perf_flags().get("kdet") == "1"
            and (_kd_layer < 0 or int(self.role.layer_idx) == _kd_layer)
            and not _is_capturing()
            and int(query_start_loc.shape[0]) > 1
            and cmp_indices is not None
        ):
            try:
                _kd_out2, _kd_lse2 = torch.ops._C_ascend.npu_sparse_flash_mla(
                    q,
                    ori_kv=attn.dsa_attn.swa_cache_layer.kv_cache[0],
                    cmp_kv=source_cache,
                    cmp_sparse_indices=cmp_indices,
                    ori_block_table=ori_block_table,
                    cmp_block_table=cmp_block_table,
                    cu_seqlens_q=query_start_loc,
                    seqused_ori_kv=seq_lens,
                    seqused_cmp_kv=cmp_seq_lens,
                    cmp_residual_kv=cmp_residual,
                    sinks=sinks,
                    metadata=op_metadata,
                    softmax_scale=attn.softmax_scale,
                    cmp_ratio=ratio,
                    ori_mask_mode=4,
                    cmp_mask_mode=3 if has_compressed else 0,
                    ori_win_left=attn.window_size - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_BBND",
                    topk_value_mode=1,
                    return_softmax_lse=True,
                )
            except Exception as _e:  # noqa: BLE001
                _kd_lse2 = None
                print("[V41-KDET] rank=%d 第二次调用失败：%r" % (_v41_dcp_rank(), _e), flush=True)
        # =====================================================================
        # ★★★★★ [V41-BLKFP 2026-09-30] **当前请求引用的 KV 块内容指纹**。
        #
        # 已证实（KDET）：同层内相同输入两次调用**逐位相同** ⇒ 算子确定；
        # 跨请求 `lse` 却变 ⇒ **KV 内容跨请求不同**。本探针直接对拍那部分内容：
        # 用**块表**取出本请求实际引用的物理块，打印其**单次**求和指纹。
        #   · 若同一 prompt 的连续请求指纹相同 ⇒ 块内容一致 ⇒ 仍有别的输入在变；
        #   · 若不同 ⇒ **KV 写入路径**跨请求产生了不同内容（本轮要锁的正是它）。
        # 每个张量只读一次（单次求和 + 单次标量回传），符合 §6o 的安全模式。
        # =====================================================================
        _blk_layer = int(_perf_flags().get("blkfp_layer") or -1)
        if (
            _perf_flags().get("blkfp") == "1"
            and (_blk_layer < 0 or int(self.role.layer_idx) == _blk_layer)
            and not _is_capturing()
            # ★ 不能用 `cmp_indices` 当 T 门：ratio=0 的层（layer 0/1）没有压缩面
            #   ⇒ `cmp_indices is None` ⇒ 探针在**最需要看的层**上不触发（踩过）。
            #   改用 `query_start_loc` 的行数判断"是不是真实 prefill"。
            and int(query_start_loc.shape[0]) > 1
        ):
            try:
                _info = []
                for _nm, _cache, _bt in (
                    ("ori", attn.dsa_attn.swa_cache_layer.kv_cache[0], ori_block_table),
                    ("cmp", source_cache, cmp_block_table),
                ):
                    if _cache is None or _bt is None or _bt.numel() == 0:
                        _info.append("%s=<none>" % _nm)
                        continue
                    # 取本请求第 0 个块（块表第 0 行第 0 列）
                    _blk = int(_bt[0, 0].item())
                    _slice = _cache[_blk].detach().to(torch.float32)
                    _ssum = float(_slice.sum())                       # 单读
                    _nnz = int((_slice != 0).sum())                   # 单读
                    _info.append("%s[blk=%d sum=%.8g nnz=%d shape=%s]"
                                 % (_nm, _blk, _ssum, _nnz, tuple(_slice.shape)))
                print(
                    "[V41-BLKFP] rank=%d layer=%d T=%d ratio=%d %s"
                    % (_v41_dcp_rank(), int(self.role.layer_idx),
                       int(query_start_loc.shape[0]) - 1,
                       int(getattr(self.role, "compress_ratio", -1)), " ".join(_info)),
                    flush=True,
                )
            except Exception as _e:  # noqa: BLE001
                print("[V41-BLKFP] rank=%d 探针自身失败：%r" % (_v41_dcp_rank(), _e), flush=True)
        if _kd_lse2 is not None:
            # ★ 单次 .cpu() 快照后比（避免多次标量读撕裂，见 §6o）
            _a = softmax_lse.to(torch.float32).cpu()
            _b = _kd_lse2.to(torch.float32).cpu()
            _same = bool(torch.equal(_a, _b))
            _d = float((_a - _b).abs().max())
            print(
                "[V41-KDET] rank=%d layer=%d T=%d lse_bit_identical=%s max_abs_diff=%.6g "
                "a_mean=%.8f b_mean=%.8f a_max=%.8f b_max=%.8f out_bit_identical=%s"
                % (
                    _v41_dcp_rank(), int(self.role.layer_idx), int(cmp_indices.shape[0]),
                    _same, _d, float(_a.mean()), float(_b.mean()),
                    float(_a.max()), float(_b.max()),
                    bool(torch.equal(output.to(torch.float32).cpu().reshape(-1),
                                     _kd_out2.to(torch.float32).cpu().reshape(-1))),
                ),
                flush=True,
            )
        if dcp_active:
            _t = _time_mark('smla_1st', _t)
            # =============================================================
            # ★★ 按**本 rank 的实际可见键数**构造权重掩码。
            #
            # 为什么必须自己算（实测，子代理 `dcp_op_unknowns` Q3/Q4）：
            # 空 rank 的算子返回 `LSE=0.0`（kernel 写死的**有限**值，不是 -inf），
            # 在 `Σ e^{L_r}·O_r / Σ e^{L_r}` 里会拿到 `exp(0 − L_max)` 的正权重：
            # 长上下文 L≈5 时每空 rank +0.7%，短序列 L≈0 时可达 1
            # （8 rank 里 7 空 ⇒ 分母虚增约 8 倍）。`isfinite` 过滤兜不住。
            #
            # 本 rank 有可见键的两种来源：
            #   · ori（滑窗，每 rank 全量复制）—— 只有 ori owner（rank 0）用；
            #   · cmp（本 rank 的压缩分片）—— `cache_seq_lens` 已是**本 rank 本地**长度
            #     （builder 用 `local_compressed_len` 算的），> 0 才有键。
            # =============================================================
            # ★★ [V41-PERF 2026-09-29] **路线 B 生效时 `token_mask` 根本不会被用到**。
            #
            # `_v41_dcp_merge_attention` 里的消费条件是
            #     `if token_mask is not None and ori_lse is None:`
            # 而路线 B（第二次纯 ori 调用）必然给出 `ori_lse is not None`
            # ⇒ 掩码被丢弃（§2.2 第 3 条：路线 B 下掩码会多扣 `A`，必须屏蔽）。
            #
            # 但旧实现**每层每步**都白建一张 `[num_reqs, max_T]` bool 表 + arange +
            # 两次比较 + and + expand + where + searchsorted + clamp + gather + view +
            # cast，共 ~12 个算子。按图内实测 13.2 µs/串行节点，这是数量级可观的浪费。
            #
            # ⇒ 只在 `skip_2nd=1`（消融臂，`ori_lse is None`、掩码真的会被用）时构造。
            _needs_mask = _pf.get('skip_2nd') == '1'
            _num_reqs_merge = int(metadata.attention.num_reqs) if metadata.attention is not None else 0
            _has_keys = None
            if _needs_mask and _num_reqs_merge > 0:
                _seq = seq_lens[:_num_reqs_merge]
                _cmp_len = (
                    metadata.attention.cache_seq_lens[:_num_reqs_merge]
                    if metadata.attention is not None
                    else torch.zeros_like(_seq)
                )
                _ori_ok = (
                    torch.ones_like(_seq, dtype=torch.bool)
                    if _is_ori_owner
                    else torch.zeros_like(_seq, dtype=torch.bool)
                )
                _has_keys = (_ori_ok & (_seq > 0)) | (_cmp_len > 0)
                _rows = int(output.shape[0])
                # ★★ 纯 device 侧构造 token→request 的映射，**绝不能**对 device 张量
                # 做 `int(...)` / `.item()` / `repeat_interleave(output_size=...)`：
                # 这些都会同步流，在 ACL graph capture 区内直接
                # `AclrtSynchronizeStreamWithTimeout` + `Not_Supported(EE1016):
                # stream is captured`（子代理 `dcp2_diff_line` 实测，原写法
                # `output_size=int(_qlens.sum())` 起服 40 s 即崩）。
                #
                # 零同步做法：把 `_has_keys` 按 request 写进一张 [num_reqs, max_T] 表，
                # 再用 `query_start_loc` 生成行内偏移，一次 `gather` 得到每行的可见性。
                # 表在每一步都重建（`torch.zeros` 在 capture 里是允许的）。
                _max_t = max(1, int(output.shape[0]))
                _tbl = torch.zeros(
                    (_num_reqs_merge, _max_t), dtype=torch.bool, device=output.device
                )
                _rows_idx = torch.arange(_max_t, dtype=torch.int64, device=output.device)
                _start = query_start_loc[:_num_reqs_merge].to(torch.int64)
                _len = (query_start_loc[1 : _num_reqs_merge + 1] - _start).to(torch.int64)
                _in_req = (_rows_idx.unsqueeze(0) >= _start.unsqueeze(1)) & (
                    _rows_idx.unsqueeze(0) < (_start + _len).unsqueeze(1)
                )
                _tbl = torch.where(
                    _in_req,
                    _has_keys.unsqueeze(1).expand(-1, _max_t),
                    torch.zeros((), dtype=torch.bool, device=output.device),
                )
                # 每行的 request 下标：queries 是按 request 连续排布的
                # ⇒ `searchsorted_right(start) - 1` 即所属 request。
                _req_of_row = (
                    torch.searchsorted(_start.contiguous(), _rows_idx, right=True) - 1
                ).clamp_(0, _num_reqs_merge - 1)
                _mask_t = _tbl[_req_of_row, _rows_idx]
                token_mask = _mask_t.view(-1, 1, 1).to(output.dtype)
            else:
                token_mask = None
            _t = _time_mark('build_mask', _t)
            # =============================================================
            # ★★ [路线 B] 第二次调用：纯 ori（`cmp_sparse_indices` 全 -1）。
            #
            # 目的：拿到 `(A, A·O_ori)`，其中 `A = e^{L_ori}`。
            # 8 个 rank 都带**真实** ori ⇒ `Σ_r e^{L_r}` 里 ori 被计了 `dcp` 次；
            # 减去 `(dcp−1)·A` 后恰好剩 1 次。
            #
            # 为什么第二次调用可行：`actCmpS2Size = min(bound, CountValid(-1)) = 0`，
            # 而 `actOriS2Size = seq_lens > 0` ⇒ **不触发**早退门。
            # `sinks` 全 rank 统一置 `-1e30` ⇒ `A` 是**不含 sink** 的纯 ori 配分函数
            # （sink 已由 rank 0 在第一次调用里计过一次）。
            # 代价：多一次 attention，但只走 ori（128 键），cmp（512 键）被跳过
            # ⇒ 约 +20% 的 attention 工作量，需实测端到端影响。
            # =============================================================
            if _pf.get('skip_2nd') != '1':
                _neg = None if _DCP_NO_ATTN_CACHE else getattr(attn, "_v41_dcp_neg_idx", None)
                _rows_n = int(q.shape[0])
                _topk_n = int(cmp_indices.shape[-1])
                if _neg is None or _neg.shape[0] != _rows_n or _neg.shape[-1] != _topk_n:
                    # 懒分配 + 地址稳定 ⇒ 图安全；尺寸变化只发生在 eager 的首步
                    # ★ 判据改成 **精确相等**（原来是 `_neg.shape[0] < _rows_n`）：
                    #   旧判据在 `_rows_n` 变小时**复用更大的张量再切片**，
                    #   切出来的是**非连续视图**；而算子对 -1 索引的读取对
                    #   stride/连续性敏感，且该张量会**跨步存活**。
                    #   改成精确相等后，形状一变就重建，消除这一类风险。
                    _neg = torch.full(
                        (_rows_n, 1, _topk_n), -1, dtype=cmp_indices.dtype, device=q.device
                    )
                    if not _DCP_NO_ATTN_CACHE:
                        attn._v41_dcp_neg_idx = _neg
                _neg = _neg[:_rows_n]
                # ★ [V41-PERF] 第二次调用的 sink 是**常量 -1e30**，同样按 attn 缓存，
                #   省掉每层每步一次 `full_like` 分配 + 填充。
                if sinks is not None:
                    _ori_sinks = None if _DCP_NO_ATTN_CACHE else getattr(attn, "_v41_dcp_neg_sinks_cache", None)
                    if _ori_sinks is None or _ori_sinks.shape != sinks.shape:
                        _ori_sinks = torch.full_like(sinks, -1e30)
                        if not _DCP_NO_ATTN_CACHE:
                            attn._v41_dcp_neg_sinks_cache = _ori_sinks
                else:
                    _ori_sinks = None
                # ★ `ori_zero_cmp=1` ⇒ 第二次调用把 `seqused_cmp_kv` 置零
                _ori_cmp_lens = cmp_seq_lens
                # [V41-DCP-ORI-ZEROCMP 2026-10-01] ★★ 默认把 cmp 长度**确定性置零**。
                # 第二次调用只要纯 ori；旧写法依赖"全 -1 索引 ⇒ actCmpS2Size=0"
                # 这个**隐含前提**，而实测 batch≥2 时该前提不成立
                # （`SparseFlashMla` 报 invalid GM address；跳过整段调用即消失）。
                # 置零后算子没有任何机会去读 cmp 键 —— 语义更严格、行为更确定。
                # 退回旧行为：V41_DCP_ORI_RAW_CMP=1
                _raw_cmp = __import__("os").environ.get("V41_DCP_ORI_RAW_CMP", "0") == "1"
                if (
                    not _raw_cmp
                    and cmp_seq_lens is not None
                ) or (
                    _perf_flags().get("ori_zero_cmp") == "1" and cmp_seq_lens is not None
                ):
                    _ori_cmp_lens = torch.zeros_like(cmp_seq_lens)
                _ori_out, _ori_lse = torch.ops._C_ascend.npu_sparse_flash_mla(
                    q,
                    ori_kv=attn.dsa_attn.swa_cache_layer.kv_cache[0],
                    cmp_kv=source_cache,
                    cmp_sparse_indices=_neg,
                    ori_block_table=_ori_bt,
                    cmp_block_table=cmp_block_table,
                    cu_seqlens_q=query_start_loc,
                    seqused_ori_kv=_ori_seqused,
                    seqused_cmp_kv=_ori_cmp_lens,
                    cmp_residual_kv=cmp_residual,
                    sinks=_ori_sinks,
                    metadata=op_metadata,
                    softmax_scale=attn.softmax_scale,
                    cmp_ratio=ratio,
                    ori_mask_mode=4,
                    cmp_mask_mode=3 if has_compressed else 0,
                    ori_win_left=attn.window_size - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_BBND",
                    topk_value_mode=1,
                    return_softmax_lse=True,
                )
                # ★ [V41-PERF] 同第一次调用：`(1,T,H)` 的内存顺序本来就是 `(T,H,1)`
                #   ⇒ `reshape` 拿视图，省掉 `permute(...).contiguous()` 的拷贝节点。
                _ori_lse_f = _ori_lse.to(torch.float32)
                if _ori_lse_f.ndim == 3 and _ori_lse_f.shape[0] == 1:
                    _ori_lse_f = _ori_lse_f.reshape(_ori_lse_f.shape[1], _ori_lse_f.shape[2], 1)
                elif _ori_lse_f.ndim == 2:
                    _ori_lse_f = _ori_lse_f.unsqueeze(-1)
            else:
                _ori_lse_f = None
                _ori_out = None
            _t = _time_mark('smla_2nd', _t)
            _rank = _v41_dcp_rank()
            output = _v41_dcp_merge_attention(
                output,
                softmax_lse,
                token_mask,
                diag_seq_lens=seq_lens,
                diag_cmp_lens=cmp_seq_lens,
                ori_lse=_ori_lse_f,
                ori_out=_ori_out,
                perf_no_pack=_pf.get('no_pack') == '1',
                head_slice=None if _hpr > 0 else (_rank * _local_heads, (_rank + 1) * _local_heads),
                layer_idx=int(self.role.layer_idx),
            )
            _t = _time_mark('merge_comm', _t)
            _time_dump('layer-%d' % self.role.layer_idx)
            # 归约后每个 rank 都有全部 head 的全局结果；本 rank 的 o_proj 只吃
            # 自己那 `H_local` 个 head（TP 连续切分）。
            # ★ [V41-PERF] 正常路径的切片已折进 `_v41_dcp_merge_attention`
            #   （`head_slice=`，先切片再除 ⇒ 除法只作用在 8 个 head 上、
            #   且省掉这里的一次 `.contiguous()` 拷贝）。只有消融模式 `_hpr>0`
            #   需要在这里处理。
            if _hpr > 0 and int(output.shape[1]) == _hpr * _v41_dcp_group().world_size:
                # 消融模式：每 rank 只留自己的 K 个 head，再平铺回 H_local（维持
                # o_proj 的输入形状；数值错误，仅用于计时）。
                _owned = output[:, _rank * _hpr : (_rank + 1) * _hpr, :].contiguous()
                if _hpr < _local_heads:
                    _owned = (
                        _owned.unsqueeze(2)
                        .expand(-1, -1, _local_heads // _hpr, -1)
                        .reshape(_owned.shape[0], _local_heads, _owned.shape[-1])
                        .contiguous()
                    )
                output = _owned
        return output

    @staticmethod
    def update_graph_params(*args, **kwargs):
        """V4.1 owns stable metadata buffers; no backend pointer patch is needed."""
        return None

    def forward(self, attn, positions, hidden_states, output: torch.Tensor | None = None):
        # The custom-op caller provides a graph-stable output buffer.  Write
        # O-projection results into it directly instead of materializing a
        # second full hidden-state tensor and copying it at the graph boundary.
        if output is None:
            output = torch.empty_like(hidden_states)
        forward_context = get_forward_context()
        if forward_context.attn_metadata is None:
            output.zero_()
            return output
        metadata = self._get_layer_metadata(forward_context.attn_metadata)
        positions = metadata.positions[: hidden_states.shape[0]]
        cos, sin = metadata.rope(attn.rotary_emb.layername, hidden_states.shape[0])
        v1_impl = attn.dsa_attn.dsa_attn.impl
        preprocess = self.multistream_preprocess if v1_impl.multistream_dsv4_dsa_overlap else self.preprocess
        q, qr = preprocess(attn, hidden_states, cos, sin, metadata.swa)
        # ★ [V41-PERF] 在 indexer **之前**发起 q 的 head all_gather，让它与
        #   `_select_sparse_indices` 的计算重叠（见 `_prepare_q_for_dcp`）。
        q, q_local_heads, q_hpr = self._prepare_q_for_dcp(q)
        if self.role.is_kv_source:
            self._write_compressed_source(
                attn,
                hidden_states,
                positions,
                cos,
                sin,
                metadata,
            )
        compressed_indices = self._select_sparse_indices(attn, hidden_states, qr, positions, cos, sin, metadata)
        # =====================================================================
        # ★★★★★★ [V41-SYNCATTN 2026-09-30] **SWA 写-读竞态判别开关**
        # （文件驱动 `sync_attn=1`，只在 EAGER 下用；capture 区内 host 同步会崩）。
        #
        # 已实证（BLKFP 层扫描 + DCP1 对照）：
        #   · layer 0 的 KV 值**两臂完全相同**（`sum=-57.663551`）；
        #   · **DCP1 复用同一物理块**（blk=111 ×4），值逐请求逐位相同、文本确定；
        #   · **DCP8 每次分配全新块**（blk=75→87→99→111），值逐请求变化、文本非确定。
        # ⇒ 发散发生在「注意力**读** SWA 缓存」与「本层给该缓存**写** K/V」之间
        #   （同一 forward 内）。DCP1 因块被复用、旧内容恰好等于新内容而看不出来。
        #
        # 判别：在注意力之前插入一次设备同步，把写与读**强行串行化**。
        #   · 非确定消失 ⇒ 确认是该竞态，同时这也是**修复方向**；
        #   · 仍在 ⇒ 另有原因。
        # =====================================================================
        if _perf_flags().get("sync_attn") == "1":
            torch.npu.synchronize()
        attention_output = self._attention(
            attn, q, metadata, compressed_indices, q_local_heads=q_local_heads, q_hpr=q_hpr
        )
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            attention_output.unsqueeze(1),
            cos,
            -sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        attn.dsa_attn.dsa_attn.impl._forward_o_proj(attention_output, output)
        return output


class DeepseekV41MetadataBuilder(AttentionMetadataBuilder[DeepseekV41Metadata]):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # ★ [V41-CFG-CACHE] 把并行配置写进进程内缓存（`_remap_selection` 用）。
        try:
            _V41_DCP_CFG["interleave"] = int(
                getattr(vllm_config.parallel_config, "cp_kv_cache_interleave_size", 1) or 1
            )
            _V41_DCP_CFG["block_size"] = int(vllm_config.cache_config.block_size)
            _V41_DCP_CFG["dcp_size"] = int(
                getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1) or 1
            )
        except Exception:  # noqa: BLE001
            pass
        max_tokens = getattr(vllm_config.scheduler_config, "max_num_batched_tokens", 4096)
        max_reqs = getattr(vllm_config.scheduler_config, "max_num_seqs", 256)
        self._supports_device_ops = getattr(device, "type", "cpu") != "cpu"
        self._slot_mapping = torch.full((max_tokens,), -1, dtype=torch.int64, device=device)
        self._slot_mapping_2d = torch.full((max_tokens, 2), -1, dtype=torch.int32, device=device)
        self._seq_lens = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._cache_seq_lens = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._cmp_residual = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._smla_metadata = torch.zeros(V41_METADATA_BUFFER_SIZE, dtype=torch.int32, device=device)
        self._qli_metadata = torch.zeros(V41_METADATA_BUFFER_SIZE, dtype=torch.int32, device=device)
        self._c2_ring_metadata = torch.zeros(5 * max_reqs, dtype=torch.int32, device=device)
        self._c2_complete_mask = torch.zeros(max_tokens, dtype=torch.bool, device=device)
        self._c2_source_positions = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        text_config = vllm_config.model_config.hf_text_config
        rope_dim = int(
            _config_value(
                text_config,
                "qk_rope_head_dim",
                _config_value(text_config, "head_dim"),
            )
        )
        c2_rope_rows = (
            max_tokens if self._supports_device_ops and isinstance(kv_cache_spec, DeepseekV41CompressorStateSpec) else 0
        )
        self._c2_source_cos = torch.ones(
            (c2_rope_rows, 1, 1, rope_dim),
            dtype=torch.float32,
            device=device,
        )
        self._c2_source_sin = torch.zeros_like(self._c2_source_cos)
        self._c2_rope_layer_names = tuple(
            name.removesuffix(".compressor.state_cache") + ".attn"
            for name in layer_names
            if name.endswith(".compressor.state_cache")
        )
        self._c2_full_source_rope: tuple[torch.Tensor, torch.Tensor] | None = None
        self._device_metadata_enabled = False
        self._device_metadata_tasks: tuple[DeviceMetadataTask, ...] = ()

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata,
        **kwargs,
    ) -> DeepseekV41Metadata:
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            **kwargs,
        )

    def enable_device_metadata(self) -> None:
        self._device_metadata_enabled = True
        if isinstance(self.kv_cache_spec, DeepseekV41CompressorStateSpec):
            if not self._c2_rope_layer_names:
                raise RuntimeError("V4.1 compressor-state builder has no source RoPE layer")
            source_rope = get_full_cos_and_sin_dsa_for_layer(self._c2_rope_layer_names[0])
            for rope_layer_name in self._c2_rope_layer_names[1:]:
                other_rope = get_full_cos_and_sin_dsa_for_layer(rope_layer_name)
                if any(other.data_ptr() != source.data_ptr() for other, source in zip(other_rope, source_rope)):
                    raise RuntimeError("V4.1 ratio-2 source layers must share one RoPE table")
            self._c2_full_source_rope = source_rope

    def take_device_metadata_tasks(self) -> tuple[DeviceMetadataTask, ...]:
        tasks = self._device_metadata_tasks
        self._device_metadata_tasks = ()
        return tasks

    def _publish_task(
        self,
        shared: dict[str, Any],
        key: str,
        buffer: torch.Tensor,
        stage: DeviceMetadataStage,
        run,
    ) -> torch.Tensor:
        existing = shared.get(key)
        if existing is not None:
            return existing
        shared[key] = buffer
        if self._device_metadata_enabled:
            self._device_metadata_tasks = (
                *self._device_metadata_tasks,
                DeviceMetadataTask(stage, run, id(buffer)),
            )
        else:
            run()
        return buffer

    def _build_batch_metadata(self, common, num_reqs, num_actual_reqs, num_input_tokens):
        self._seq_lens[:num_reqs].copy_(common.seq_lens[:num_reqs])
        if num_actual_reqs < num_reqs:
            self._seq_lens[num_actual_reqs:num_reqs].zero_()
        seq_lens_cpu = getattr(common, "seq_lens_cpu", None)
        if seq_lens_cpu is None:
            seq_lens_cpu = getattr(common, "_seq_lens_cpu", None)
        max_seq_len = int(getattr(common, "max_seq_len", 0))
        if seq_lens_cpu is not None:
            max_seq_len = int(seq_lens_cpu[:num_actual_reqs].max().item()) if num_actual_reqs else 0
        num_decodes, num_decode_tokens, num_prefills, num_prefill_tokens = _request_counts(common, num_reqs)
        positions = common.positions
        if positions is not None:
            positions = positions[:num_input_tokens].long()
        return dict(
            query_start_loc=common.query_start_loc[: num_reqs + 1],
            query_start_loc_cpu=getattr(common, "query_start_loc_cpu", None),
            seq_lens=self._seq_lens[:num_reqs],
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            max_cache_seq_len=max_seq_len,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
        )

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        fast_build=False,
        **kwargs,
    ):
        if common_prefix_len:
            raise NotImplementedError("V4.1 prefix caching is not implemented")
        self._device_metadata_tasks = ()
        # ★ [V41-FLAGREFRESH 2026-09-30 19:45] 每步刷新一次文件开关（见
        #   `_refresh_perf_flags` 的性能说明）。放在 build() 里 ⇒ 每步
        #   （每个 cache group 一次）做 ≤5 次 `os.stat`，而不是每层 11 次。
        _refresh_perf_flags()
        spec = self.kv_cache_spec
        common = common_attn_metadata
        is_compressor_state = isinstance(spec, DeepseekV41CompressorStateSpec)
        ratio = getattr(spec, "compress_ratio", 1)
        if isinstance(spec, DeepseekV41SWASpec):
            cache_kind = "swa"
        elif isinstance(spec, DeepseekV41FullSpec):
            cache_kind = "long_kv"
        elif isinstance(spec, DeepseekV41IndexerSpec):
            cache_kind = "index_k"
        elif is_compressor_state:
            cache_kind = "compressor_state"
        else:
            raise TypeError(f"Unsupported V4.1 cache spec: {type(spec).__name__}")

        num_reqs = int(getattr(common, "num_reqs", common.seq_lens.shape[0]))
        num_actual_reqs = int(kwargs.get("num_actual_reqs", num_reqs))
        num_actual_reqs = min(num_actual_reqs, num_reqs)
        num_input_tokens = int(getattr(common, "num_input_tokens", common.slot_mapping.shape[0]))
        num_actual_tokens = int(getattr(common, "num_actual_tokens", num_input_tokens))
        shared = kwargs.get("common_v41_metadata")
        if shared is None:
            shared = {}
        batch_shared = kwargs.get("common_v41_batch_metadata")
        if batch_shared is None:
            batch_shared = shared

        # The runner resets both dictionaries on each build. Batch values do
        # not depend on physical block IDs; slot mappings remain group-local.
        batch_metadata = batch_shared.get("batch")
        if batch_metadata is None:
            batch_metadata = self._build_batch_metadata(common, num_reqs, num_actual_reqs, num_input_tokens)
            batch_shared["batch"] = batch_metadata
        coordinates = dict(batch_metadata)
        seq_lens = coordinates["seq_lens"]
        positions = coordinates["positions"]

        # =====================================================================
        # [V41-DCP 2026-09-29] 复制态 indexer 的 `[T, 2]` 槽位映射。
        #
        # 与 long_kv 的关键区别：indexer 要覆盖**全量**序列（每个 rank 一份），
        # 而 long_kv 只覆盖本 rank 的 1/dcp。物理布局：
        #   同一个物理块内，index 面被放成 `dcp` 个「子块」，
        #   全局压缩 token `g` → 块列 `g // (dcp*B')`、面内偏移 `((g % (dcp*B'))//B')*B' + g % B'`
        #   其中 `B' = storage_block_size`（已按 ratio 缩过）。
        # 块列是**本 rank 的局部列**：所有 rank 的调度器确定性一致 ⇒ 同一列拿到同一
        # 个物理块 ID ⇒ 8 份 indexer 副本内容逐位相同（这是"全局 top-k 零通信一致"
        # 的前提）。`g` 用全局位置算，不做 rank 过滤。
        # =====================================================================
        dcp_size = int(getattr(self.vllm_config.parallel_config, "decode_context_parallel_size", 1) or 1)
        from vllm_ascend.patch.platform.patch_v41_dcp import replicate_indexer

        index_is_replicated = (
            cache_kind == "index_k"
            and dcp_size > 1
            and replicate_indexer()
            and max(1, int(getattr(spec, "dcp_world_size", 1) or 1)) > 1
        )

        # SWA uses original-token coordinates; circular state has no token slots.
        # Long KV and index K are addressed in completed compression groups.
        compressed = cache_kind in {"long_kv", "index_k"}
        if is_compressor_state:
            # State writes use ring ownership metadata; this buffer stays PAD.
            slots = self._slot_mapping[:num_input_tokens]
        else:
            # Scope ``shared`` to one framework KV cache group in the model
            # runner. Long KV and Indexer builders with the same physical
            # layout then share one persistent [T, 2] mapping, while every SWA
            # group owns a distinct mapping buffer.
            slot_key = f"slot:c{ratio}:b{spec.storage_block_size}"
            if index_is_replicated:
                slot_key += f":dcp{dcp_size}"
            prepared_slots = shared.get(slot_key)
            # =============================================================
            # ★★★★★★ [V41-SLOTSTALE 2026-09-30 16:30] **复制态 slot 缓存的跨请求
            # 复用判据**。
            #
            # 复制态 indexer 的 `[T,2]` 槽位映射**编码了物理块号**
            # （`block_numbers = block_table[req, column]`），而物理块号**每个请求
            # 都不同**（实测：第 1 个请求 phys=1、第 2 个 phys=11）。
            # 若 `shared` 字典跨请求存活，第 2 个请求就会命中第 1 个请求的槽位
            # ⇒ **index_k 写到上一个请求的物理块**、本请求的块里是旧数据
            # ⇒ QLI 分数错 ⇒ top-k 集合变（实测 `n_valid 51254→51260`、
            # `sum 2035126→2027486`）⇒ cmp 读到不该读的行 ⇒ `lse` 出现 NaN
            # （实测 `0/904 → 95/904`）⇒ 乱码。
            #
            # 本探针打印 **MISS/HIT** 与**它命中的那个块号**，用来判定这条假设。
            # 开关 `no_slotcache=1` 强制**每次重算**（绕过缓存），
            # 若开启后"第 2 个请求"也正确 ⇒ 假设成立，这就是第 5 个根因的修复。
            # =============================================================
            _no_slotcache = _perf_flags().get("no_slotcache") == "1"
            if (
                index_is_replicated
                and _perf_flags().get("slotprobe") == "1"
                and not _is_capturing()
                and prepared_slots is not None
            ):
                try:
                    _bt0 = int(common.block_table_tensor[0, 0].to(torch.int64).cpu())
                    _cached0 = int(prepared_slots[0, 0].to(torch.int64).cpu())
                    print(
                        "[V41-SLOTPROBE] layer=%d ratio=%d T=%d shared_id=%d HIT cached_blk=%d "
                        "cur_blk=%d %s"
                        % (int(self.role.layer_idx), int(ratio),
                           int(prepared_slots.shape[0]), id(batch_shared),
                           _cached0, _bt0,
                           "★STALE" if (_cached0 != _bt0 and _cached0 >= 0 and _bt0 >= 0) else "ok"),
                        flush=True,
                    )
                except Exception as _e:  # noqa: BLE001
                    print("[V41-SLOTPROBE] 失败：%r" % (_e,), flush=True)
            if _no_slotcache:
                prepared_slots = None
            if prepared_slots is None and index_is_replicated:
                # ---- 复制态：块列与面内偏移都从**全局压缩位置**算，不做 rank 过滤 ----
                storage = int(spec.storage_block_size)
                block_table = getattr(common, "block_table_tensor", None)
                if block_table is None or positions is None:
                    raise RuntimeError("[V41-DCP] 复制态 indexer 需要 block_table_tensor 与 positions")
                pos = positions[:num_input_tokens]
                global_g = torch.div(pos, ratio, rounding_mode="floor")
                span = dcp_size * storage
                column = torch.div(global_g, span, rounding_mode="floor")
                within = global_g.remainder(span)
                face_offset = (
                    torch.div(within, storage, rounding_mode="floor") * storage + within.remainder(storage)
                )
                query_lens = (
                    common.query_start_loc[1 : num_reqs + 1] - common.query_start_loc[:num_reqs]
                ).to(torch.int64)
                req_indices = torch.repeat_interleave(
                    torch.arange(num_reqs, dtype=torch.int64, device=block_table.device),
                    query_lens,
                    output_size=pos.shape[0],
                )[: num_input_tokens]
                columns = column.clamp(0, block_table.shape[1] - 1)
                block_numbers = block_table[req_indices, columns]
                group_complete = (
                    pos.remainder(ratio) == (ratio - 1)
                    if ratio > 1
                    else torch.ones_like(pos, dtype=torch.bool)
                )
                repl_valid = group_complete & (block_numbers > 0)
                self._slot_mapping_2d[:num_input_tokens, 0].copy_(
                    torch.where(repl_valid, block_numbers, -1)
                )
                self._slot_mapping_2d[:num_input_tokens, 1].copy_(
                    torch.where(repl_valid, face_offset, -1)
                )
                prepared_slots = self._slot_mapping_2d[:num_input_tokens]
                shared[slot_key] = prepared_slots
            if prepared_slots is None:
                active_slots = common.slot_mapping[:num_input_tokens]
                if compressed and ratio != 1:
                    active_slots = compressed_slot_mapping(active_slots, ratio)
                valid = active_slots >= 0
                if compressed and ratio == 2:
                    # Prepare the C2 store mask once per cache group, before
                    # forward. Match the ring compressor's completion policy.
                    if kwargs.get("skip_ring_state_update", False):
                        valid.zero_()
                    else:
                        valid_end = common.query_start_loc[num_actual_reqs].clamp_max(num_actual_tokens)
                        valid &= torch.arange(num_input_tokens, device=active_slots.device) < valid_end
                        if positions is not None:
                            valid &= positions.remainder(2) == 1
                physical = active_slots.clamp_min(0)
                self._slot_mapping_2d[:num_input_tokens, 0].copy_(
                    torch.where(
                        valid,
                        torch.div(
                            physical,
                            spec.storage_block_size,
                            rounding_mode="floor",
                        ),
                        -1,
                    )
                )
                self._slot_mapping_2d[:num_input_tokens, 1].copy_(
                    torch.where(
                        valid,
                        physical.remainder(spec.storage_block_size),
                        -1,
                    )
                )
                prepared_slots = self._slot_mapping_2d[:num_input_tokens]
                shared[slot_key] = prepared_slots
            slots = prepared_slots
        plane_ratio = ratio if compressed else 1
        # =====================================================================
        # [V41-DCP 2026-09-29] ★ 压缩平面的可见长度必须是**本 rank 的本地长度**，
        # 不是全局长度。
        #
        # 原因：本 rank 的 long_kv / index_k 物理块只装 1/dcp 的序列。若把全局
        # 压缩长度当 `seqused_cmp_kv` / `seqused_k` 传下去，算子会去读本 rank
        # 块表里不存在的行 ⇒ 读到 null/邻块数据 ⇒ 返回一个**有限的** LSE ⇒
        # 在 `Σ e^{L_r}·O_r / Σ e^{L_r}` 里按错误权重污染结果。
        #
        # ★ 极端且已复现：prompt 只有 17 个 token（I=32, dcp=8）时
        # `owner(pos)=(pos//32)%8` ⇒ **只有 rank 0 有 KV**，rank 1-7 本地长度为 0。
        # 全局长度会告诉它们"你有 8 行"，于是它们读空块、报有限 LSE、把结果搞乱。
        # 这与实测的输出错误形态吻合。
        #
        # `local_compressed_len` 与写侧 `compressed_slot_mapping` 完全同源，
        # 已用 600+ 个长度 × 8 个 rank 与"逐 g 模拟"对拍，零 mismatch。
        # =====================================================================
        _dcp = int(getattr(self.vllm_config.parallel_config, "decode_context_parallel_size", 1) or 1)
        _local_len_ratio = 1
        from vllm_ascend.patch.platform.patch_v41_dcp import v41_dcp_active as _dcp_active

        if _dcp > 1 and compressed and _dcp_active() and not index_is_replicated:
            from vllm.distributed import get_dcp_group

            from vllm_ascend.attention.context_parallel.v41_dcp import local_compressed_len

            _interleave = int(getattr(self.vllm_config.parallel_config, "cp_kv_cache_interleave_size", 1) or 1)
            _rank = int(get_dcp_group().rank_in_group)
            local_lengths = local_compressed_len(
                seq_lens,
                interleave=_interleave,
                ratio=ratio,
                dcp_size=_dcp,
                dcp_rank=_rank,
            )
            local_lengths = local_lengths.to(self._cache_seq_lens.dtype)
            local_max = local_compressed_len(
                torch.tensor([coordinates["max_cache_seq_len"]], dtype=seq_lens.dtype, device=seq_lens.device),
                interleave=_interleave,
                ratio=ratio,
                dcp_size=_dcp,
                dcp_rank=_rank,
            ).item()
            _local_len_ratio = max(1, int(local_max)) if local_max > 0 else 1
        else:
            local_lengths = None
        coordinates["cache_seq_lens"] = seq_lens
        cmp_residual_buffer = None
        if compressed and ratio == 2:
            # ★ [V41-DCP 2026-09-29] 缓存 key 必须区分"局部长度"与"全局长度"。
            # 原来只用 `"lengths:c2"`：`long_kv` 与 `indexer` 两个平面若分别处于
            # 分片态（局部长度）与复制态（全局长度），会共用同一个 batch 级缓存，
            # **先写者获胜** ⇒ 后用的平面拿到错口径的长度。
            # 默认 `V41_DCP_REPLICATE_INDEXER=0` 时两平面都是局部，暂未触发；
            # 但这是潜伏 bug（子代理 `cmp_semantics` 代码审计发现），先按平面+口径分开。
            _len_key = f"lengths:c2:{cache_kind}:{'local' if local_lengths is not None else 'global'}"
            compressed_lengths = batch_shared.get(_len_key)
            if compressed_lengths is None:
                if local_lengths is not None:
                    self._cache_seq_lens[:num_reqs].copy_(local_lengths[:num_reqs])
                else:
                    torch.div(seq_lens, ratio, rounding_mode="floor", out=self._cache_seq_lens[:num_reqs])
                torch.remainder(seq_lens, ratio, out=self._cmp_residual[:num_reqs])
                compressed_lengths = (self._cache_seq_lens[:num_reqs], self._cmp_residual[:num_reqs])
                batch_shared[_len_key] = compressed_lengths
            coordinates["cache_seq_lens"], cmp_residual_buffer = compressed_lengths
        elif local_lengths is not None:
            self._cache_seq_lens[:num_reqs].copy_(local_lengths[:num_reqs])
            coordinates["cache_seq_lens"] = self._cache_seq_lens[:num_reqs]
        coordinates["max_cache_seq_len"] = (
            _local_len_ratio if local_lengths is not None else coordinates["max_cache_seq_len"] // plane_ratio
        )
        cos = sin = None
        if cache_kind == "swa" and positions is not None:
            rope = batch_shared.get("rope")
            if rope is None:
                rope = get_cos_and_sin_dsa(positions, use_cache=coordinates["num_prefills"] == 0)
                batch_shared["rope"] = rope
            cos, sin = rope
        text_config = self.vllm_config.model_config.hf_text_config
        window_size = int(_config_value(text_config, "sliding_window", 0))
        # `operator_ratio` / `has_compressed` 必须在下面用之前就定义好
        # （我上一版把它们留在 `if self._supports_device_ops` 块里，
        #  而新加的 head 数分支在块外引用 ⇒ 真机
        #  `UnboundLocalError: cannot access local variable 'has_compressed'`，
        #  graph capture 阶段 8 个 worker 全挂）。
        operator_ratio = 0 if cache_kind == "swa" else ratio
        has_compressed = operator_ratio in (1, 2)
        n_local_heads = (
            int(_config_value(text_config, "num_attention_heads"))
            // self.vllm_config.parallel_config.tensor_parallel_size
        )
        # [V41-DCP 2026-09-29] 走跨 rank 合并的层：q 会被 all-gather 成**全部** head，
        # 所以算子 metadata 的 `num_heads_q` 必须用全量 head 数，而不是 TP 分片后的。
        # 不这么做的话算子会按 8 个 head 建 metadata、却收到 64 个 head 的 q。
        # [V41-DCP-HEADS-FIX 2026-10-01] ★★ 必须与 `_v41_dcp_gather_heads`
        # 的**实际行为**对齐：那个函数是 `if dcp_size <= 1: return q`（no-op）。
        # 原条件只看 `_v41_dcp_on()`（= env 开关），而 DCP=1 时它仍为 True
        # （`V41_DCP_ALLOW_CAPACITY_PROBE=1` 也会让它为真）
        # ⇒ q 仍是 TP 分片（32 head）而 metadata 按全量（64 head）建
        # ⇒ 2× 错配 ⇒ FD 归约 workspace 越界
        #   （fault kernel `SparseFlashMla_..._mix_aic` + invalid GM address）。
        # 实测：batch=1 静默、batch≥2 崩。
        # 对 DCP>1 无影响（world_size>1 时条件本就成立）。
        if (
            cache_kind == "long_kv"
            and _v41_dcp_on()
            and has_compressed
            and _v41_dcp_group().world_size > 1
        ):
            n_local_heads = int(_config_value(text_config, "num_attention_heads"))
        head_dim = int(_config_value(text_config, "head_dim"))
        index_topk = int(_config_value(text_config, "index_topk"))
        smla_metadata = None
        qli_metadata = None

        if self._supports_device_ops and cache_kind in {"swa", "long_kv"}:
            cmp_seq_lens = coordinates["cache_seq_lens"] if has_compressed else None
            # ★★★★★ [V41-CMPGLOB-2 2026-10-01 01:10] **metadata 用本地长度，
            # 只有传给 SMLA 算子的 `seqused_cmp_kv` 换成全局长度**。
            #
            # 为什么不能改 metadata：把全局长度喂给 `SparseFlashMlaMetadata`
            # 会直接让它在 AICPU 上崩（实测 run `dcpcap_1001_010130`：
            #   `AI CPU kernel execution failed … kernelName=SparseFlashMlaMetadata`）。
            # 语义上也说得通：metadata 只做**核间切分**（bN2Start/gS1Start/s2Start…），
            # 真正决定键集合的是内核直接读的 `seqused_cmp_kv` 张量（见
            # `_v41_global_cmp_lens`）。单卡实验里 metadata 也仍是本地长度算出来的。
            cmp_residual = cmp_residual_buffer

            def build_smla_metadata() -> None:
                value = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
                    n_local_heads,
                    1,
                    head_dim,
                    cu_seqlens_q=common.query_start_loc[: num_reqs + 1].int(),
                    seqused_ori_kv=seq_lens,
                    seqused_cmp_kv=cmp_seq_lens,
                    cmp_residual_kv=cmp_residual,
                    batch_size=num_reqs,
                    max_seqlen_q=int(getattr(common, "max_query_len", 0)),
                    max_seqlen_ori_kv=int(getattr(common, "max_seq_len", 0)),
                    max_seqlen_cmp_kv=(coordinates["max_cache_seq_len"] if has_compressed else 0),
                    ori_topk=0,
                    cmp_topk=index_topk if has_compressed else 0,
                    cmp_ratio=operator_ratio,
                    ori_mask_mode=4,
                    cmp_mask_mode=3 if has_compressed else 0,
                    ori_win_left=max(0, window_size - 1),
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_BBND",
                    # ==========================================================
                    # [V41-DCP 2026-09-29] ori（滑窗）平面**只由一个 rank 携带**。
                    #
                    # 为什么：DCP 切 KV，滑窗是**每 rank 全量复制**的。合并式
                    #     O = Σ_r w_r·O_r / Σ_r w_r,  w_r = exp(L_r)
                    # 里若 8 个 rank 都算进了同一份 ori 窗口 A，则分子分母都把 A
                    # 计了 8 次（其它 7 次是与不同 B_r 混在一起的，无法事后剥离）。
                    # 让 **rank 0 独占 A**、其余 rank 只带自己的 cmp 分片 B_r：
                    #     L_0      = logsumexp(A ∪ B_0)
                    #     L_{r>0}  = logsumexp(B_r)
                    # ⇒ 合并后的分子/分母里 A 恰好各出现一次
                    #   ⇒ 与全局 `A ∪ B_0 ∪ … ∪ B_7` 精确一致。
                    #
                    # 这条等价性是纯组合性质的（各 rank 的可见集互不重叠地覆盖
                    # 全局集），已被离线夹具用真算子验证（分段覆盖 relout=0.0）。
                    #
                    # 注意只对 **有压缩（ratio∈{1,2}）** 的层生效：ratio=0 的层
                    # 只有 ori、且每 rank 本地就能算出完整正确的滑窗注意力，
                    # 不需要、也不应该走合并。
                    # ==========================================================
                    has_ori_kv=(
                        True
                        if not _v41_dcp_on()
                        else (_v41_dcp_ori_owner() != "rank0" or _v41_dcp_rank() == 0)
                    ),
                    has_cmp_kv=has_compressed,
                )
                self._smla_metadata.copy_(value)

            smla_metadata = self._publish_task(
                batch_shared,
                f"smla:c{operator_ratio}",
                self._smla_metadata,
                DeviceMetadataStage.ATTENTION,
                build_smla_metadata,
            )

        if self._supports_device_ops and cache_kind == "index_k":
            residual = cmp_residual_buffer

            def build_qli_metadata() -> None:
                value = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
                    int(_config_value(text_config, "index_n_heads")),
                    1,
                    int(_config_value(text_config, "index_head_dim")),
                    index_topk,
                    2,
                    cu_seqlens_q=common.query_start_loc[: num_reqs + 1].int(),
                    seqused_k=coordinates["cache_seq_lens"],
                    cmp_residual_k=residual,
                    batch_size=num_reqs,
                    max_seqlen_q=int(getattr(common, "max_query_len", 0)),
                    max_seqlen_k=coordinates["max_cache_seq_len"],
                    layout_q="TND",
                    layout_k="PA_BBND",
                    mask_mode=3,
                    cmp_ratio=ratio,
                )
                self._qli_metadata.copy_(value)

            qli_metadata = self._publish_task(
                batch_shared,
                f"qli:c{ratio}",
                self._qli_metadata,
                DeviceMetadataStage.INDEXER,
                build_qli_metadata,
            )

        c2_ring_metadata = None
        c2_complete_mask = None
        c2_source_positions = None
        c2_source_cos = None
        c2_source_sin = None
        c2_metadata_group_id = None
        if cache_kind == "compressor_state" and positions is not None:
            ring_meta = self._c2_ring_metadata[: 5 * num_reqs].view(5, num_reqs)
            input_positions = positions
            if self._supports_device_ops:
                if self._c2_full_source_rope is None:
                    raise RuntimeError("V4.1 source RoPE buffers were not initialized")
                full_source_cos, full_source_sin = self._c2_full_source_rope
            else:
                full_source_cos = full_source_sin = None

            def build_c2_metadata() -> None:
                starts = common.query_start_loc[:num_reqs].int()
                ends = common.query_start_loc[1 : num_reqs + 1].int()
                query_lens = ends - starts
                live = torch.arange(num_reqs, device=starts.device) < num_actual_reqs
                used = (ends.clamp_max(num_actual_tokens) - starts).clamp_min(0)
                used = torch.where(live, used, 0)
                if kwargs.get("skip_ring_state_update", False):
                    used = torch.zeros_like(used)
                ring_meta[0].copy_((seq_lens - query_lens).clamp_min(0))
                ring_meta[1].copy_(used)
                ring_meta[2].copy_(starts)
                ring_meta[3].copy_(starts)
                ring_meta[4].copy_(torch.where(used > 0, common.block_table_tensor[:num_reqs, 0], 0))
                valid_end = common.query_start_loc[num_actual_reqs].clamp_max(num_actual_tokens)
                valid = torch.arange(num_input_tokens, device=input_positions.device) < valid_end
                complete = (input_positions.remainder(2) == 1) & valid
                if kwargs.get("skip_ring_state_update", False):
                    complete = torch.zeros_like(complete)
                self._c2_complete_mask[:num_input_tokens].copy_(complete)
                self._c2_source_positions[:num_input_tokens].copy_(
                    torch.where(
                        complete,
                        input_positions - 1,
                        torch.zeros_like(input_positions),
                    )
                )
                if full_source_cos is not None and full_source_sin is not None:
                    gather_idx = (
                        self._c2_source_positions[:num_input_tokens]
                        .reshape(-1, 1, 1, 1)
                        .expand(
                            num_input_tokens,
                            1,
                            1,
                            full_source_cos.shape[-1],
                        )
                    )
                    torch.gather(
                        full_source_cos,
                        0,
                        gather_idx,
                        out=self._c2_source_cos[:num_input_tokens],
                    )
                    torch.gather(
                        full_source_sin,
                        0,
                        gather_idx,
                        out=self._c2_source_sin[:num_input_tokens],
                    )

            compressor_group = self._publish_task(
                shared,
                "c2:compressor",
                self._c2_complete_mask,
                DeviceMetadataStage.COMPRESSOR,
                build_c2_metadata,
            )
            if compressor_group is not self._c2_complete_mask:
                raise RuntimeError("V4.1 compressor metadata must have one owner")
            c2_complete_mask = self._c2_complete_mask[:num_input_tokens]
            c2_ring_metadata = ring_meta
            c2_source_positions = self._c2_source_positions[:num_input_tokens]
            if self._supports_device_ops:
                c2_source_cos = self._c2_source_cos[:num_input_tokens]
                c2_source_sin = self._c2_source_sin[:num_input_tokens]
            c2_metadata_group_id = id(self._c2_complete_mask)
        return DeepseekV41Metadata(
            block_table=common.block_table_tensor[:num_reqs],
            slot_mapping=slots,
            compress_ratio=ratio,
            storage_block_size=spec.storage_block_size,
            is_compressor_state=is_compressor_state,
            cache_kind=cache_kind,
            cos=cos,
            sin=sin,
            num_actual_tokens=num_actual_tokens,
            num_input_tokens=num_input_tokens,
            num_reqs=num_reqs,
            num_actual_reqs=num_actual_reqs,
            logical_block_size=spec.block_size,
            max_query_len=int(getattr(common, "max_query_len", 0)),
            max_seq_len=int(getattr(common, "max_seq_len", 0)),
            attn_state=getattr(common, "attn_state", None),
            is_prefilling=getattr(common, "is_prefilling", None),
            causal=getattr(common, "causal", True),
            ori_win_left=max(0, window_size - 1),
            ori_win_right=0,
            smla_metadata=smla_metadata,
            qli_metadata=qli_metadata,
            cmp_residual=cmp_residual_buffer,
            c2_ring_metadata=c2_ring_metadata,
            c2_complete_mask=c2_complete_mask,
            c2_source_positions=c2_source_positions,
            c2_source_cos=c2_source_cos,
            c2_source_sin=c2_source_sin,
            c2_metadata_group_id=c2_metadata_group_id,
            **coordinates,
        )


class DeepseekV41CacheBackend(AttentionBackend):
    """Cache-only backend: supplies layout and metadata, not an AttentionImpl."""

    @staticmethod
    def get_name():
        return "ASCEND_DSA_V41_CACHE"

    @staticmethod
    def get_impl_cls():
        return DeepseekV41EagerAttentionImpl

    @staticmethod
    def get_builder_cls():
        return DeepseekV41MetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
        return num_blocks, block_size, num_kv_heads, head_size


class DeepseekV41CacheLayer(nn.Module, AttentionLayerBase):
    supports_dcp = False

    def __init__(self, vllm_config, prefix, spec):
        super().__init__()
        self.prefix = prefix
        self.spec = spec
        self.kv_cache = [torch.empty(0)]
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate V4.1 cache prefix: {prefix}")
        context[prefix] = self

    def get_kv_cache_spec(self, vllm_config):
        return self.spec

    def get_attn_backend(self):
        return DeepseekV41CacheBackend
