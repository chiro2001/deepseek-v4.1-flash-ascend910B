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


def _perf_flags() -> dict:
    """性能消融开关（**文件驱动**，便于在**不重启**的情况下逐项 A/B）。

    为什么用文件而不是 env：容器进程的 `os.environ` 起服后无法从外部修改，
    而每次改开关重启要 12 分钟。读文件只在**每步一次**的 Python 层发生
    （不是逐 token），开销可忽略；且不触碰 device stream ⇒ 与图捕获兼容
    （前提：一次测量期间开关保持不变，这与"图在捕获时固化分支"一致）。

    开关（都是**诊断/性能测量用，会让结果不正确**，绝不进生产）：
      skip_2nd=1    跳过第二次「纯 ori」SMLA 调用
      skip_merge=1  跳过整个跨 rank 合并（直接返回本 rank 的结果）
      no_pack=1     合并用 4 次独立 all_reduce（回到打包前的实现）
      skip_gather=1 跳过 q 的 head 维 all_gather
    """
    import os as _o
    try:
        st = _o.stat(PERF_FLAG_PATH)
    except OSError:
        return {}
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


def _v41_dcp_rank() -> int:
    """本进程在 DCP 组内的 rank（DCP 关闭时返回 0）。"""
    if not _v41_dcp_on():
        return 0
    from vllm.distributed import get_dcp_group

    return int(get_dcp_group().rank_in_group)


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
    group = _v41_dcp_group()
    dcp_size = group.world_size
    if dcp_size <= 1:
        # ★ 保留调用方原来那次 `.contiguous()`：DCP1 时 `output` 直接来自算子，
        #   不保证连续；旧代码在合并之后统一做了
        #   `output[:, 0:H_local, :].contiguous()`，这里等价保留，
        #   保证「DCP1 路径与本改动前逐位一致」（只影响 DCP1，不影响 DCP8 的优化）。
        return output.contiguous()
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
    if _use_ori_ref:
        _ori_lse_f32 = ori_lse.to(torch.float32)
        _delta = lse - _ori_lse_f32
        # 空 rank（lse=-inf）⇒ exp(-inf)=0；`-inf − (-inf)` 出 NaN ⇒ nan_to_num 归 0。
        weights = torch.nan_to_num(torch.exp(_delta.clamp(max=60.0)))
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

    if _os_m.environ.get("V41_DCP_MERGE_RANK0_ONLY") == "1" and _v41_dcp_rank() != 0:
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
    scaled = output.to(torch.float32) * weights
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
    if _ori_active:
        if _use_ori_ref:
            # ★ `_ow` 只在**非折叠**路径（进 pack）用到；折叠路径里 `Σ_r _ow_r`
            #   是常量 `dcp·_keep`，本地直接算 ⇒ 这里不再白建一个 `[T,H,1]` 张量。
            _ow = None if _fold_ori_locally else torch.full_like(weights, _keep)
            _oi = (
                ori_out[:, head_slice[0] : head_slice[1], :] if _slice_early else ori_out
            )
            _onum = _oi.to(torch.float32) * _keep
        else:
            # `skip_2nd=1` 消融臂：仍用跨 rank max 作参考点（那条路径才有 lse_max）。
            _ow = torch.nan_to_num(torch.exp(ori_lse.to(torch.float32) - lse_max)) * _keep
            _onum = ori_out.to(torch.float32) * _ow
    else:
        _ow = torch.zeros_like(weights)
        _onum = torch.zeros_like(scaled)
    if _fold_ori_locally:
        # 本地折叠：只把「加权分子」与「权重和」打包归约（16.8 MB，旧的一半）。
        _pack = torch.cat([scaled, weights], dim=-1)
        torch.distributed.all_reduce(_pack, group=group.device_group)
        _out_dim = scaled.shape[-1]
        # ★ 扣除量 = `Σ_r _onum_r` / `Σ_r _ow_r`：
        #   `_onum_r = ori_out·_keep` 与 `_ow_r = _keep` **每个 rank 各一份**
        #   ⇒ `Σ_r _onum_r = dcp·_onum`、`Σ_r _ow_r = dcp·_keep = dcp−1`。
        if _slice_early:
            # 归约后立刻切到本 rank 的 8 个 head ⇒ 减法只在 8 个 head 上做。
            _h0, _h1 = head_slice
            scaled = _pack[..., :_out_dim][:, _h0:_h1, :] - dcp * _onum
            wsum = _pack[..., _out_dim:][:, _h0:_h1, :] - dcp * _keep
            head_slice = None  # 已应用，别在下面再切一次
        else:
            scaled = _pack[..., :_out_dim] - dcp * _onum
            wsum = _pack[..., _out_dim:] - dcp * _keep
    elif perf_no_pack:
        # 消融臂：回到打包前的实现（4 次独立 all_reduce），用于量化打包收益
        _n_all = _onum.clone(); _w_all = _ow.clone()
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
    # ★ `wsum` 数学上恒正（= e^{-c}·(A + ΣZ + ΣS)）；用 `clamp_min` 代替
    #   `where(wsum > 0, wsum, ones_like)`，省掉 `ones_like` + `gt` 两个节点。
    denom = wsum.clamp_min(1e-30)
    # ★ 用 `torch.div` 直接指定 `out=` 会引入别名风险，保持简单：elementwise 结果
    #   天然连续，`to(dtype)` 后调用方无需再 `.contiguous()`。
    return (scaled / denom).to(output.dtype)


_LSE_DIAG_COUNT = {"n": 0}


def _lse_diag_count() -> int:
    return _LSE_DIAG_COUNT["n"]


def _bump_lse_diag() -> None:
    _LSE_DIAG_COUNT["n"] += 1


_TIME_ACC = {}
_ORI_REF_DIAG = {"n": 0}


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
            return self._remap_selection(selected, positions)
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
        selected = self._remap_selection(selected, positions)
        shared.topk_indices[: selected.shape[0]].copy_(selected)
        if self.role.is_candidate_source:
            shared.candidates[: candidates.shape[0]].copy_(candidates)
        return shared.topk_indices[: selected.shape[0]]

    def _remap_selection(self, selected, positions):
        """把全局压缩 top-k 索引重映射到本 rank 的本地压缩坐标。"""
        import os as _os

        from vllm_ascend.patch.platform.patch_v41_dcp import replicate_indexer, v41_dcp_active

        if not v41_dcp_active() or not replicate_indexer():
            return selected
        from vllm.distributed import get_dcp_group

        from vllm_ascend.attention.context_parallel.v41_dcp import remap_sparse_indices

        parallel = get_forward_context().vllm_config.parallel_config
        dcp_size = int(getattr(parallel, "decode_context_parallel_size", 1) or 1)
        if dcp_size <= 1:
            return selected
        interleave = int(getattr(parallel, "cp_kv_cache_interleave_size", 1) or 1)
        return remap_sparse_indices(
            selected,
            block_size=int(get_forward_context().vllm_config.cache_config.block_size),
            interleave=interleave,
            ratio=int(self.role.compress_ratio),
            dcp_size=dcp_size,
            dcp_rank=int(get_dcp_group().rank_in_group),
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
        cmp_block_table = None
        cmp_seq_lens = None
        cmp_residual = None
        cmp_indices = None
        cmp_topk = 0
        if has_compressed:
            if source_cache is None or metadata.attention is None or compressed_indices is None:
                raise RuntimeError("V4.1 compressed attention is missing KV or TopK metadata")
            cmp_block_table = metadata.attention.block_table[:num_reqs]
            cmp_seq_lens = metadata.attention.cache_seq_lens[:num_reqs]
            cmp_residual = metadata.attention.cmp_residual
            cmp_topk = self.topology.index_topk
            if cmp_topk not in (512, 1024):
                raise ValueError(f"SparseFlashMla only supports TopK 512 or 1024, got {cmp_topk}")
            cmp_indices = pad_sparse_indices(compressed_indices, cmp_topk)
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
            sinks = getattr(attn, "_v41_dcp_sinks_cache", None)
            if sinks is None or int(sinks.shape[0]) != int(attn.attn_sink.shape[0]) * _v41_dcp_group().world_size:
                sinks = _v41_dcp_gather_1d(attn.attn_sink)
                if _v41_dcp_rank() != 0:
                    sinks = torch.full_like(sinks, -1e30)
                attn._v41_dcp_sinks_cache = sinks
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
                _neg = getattr(attn, "_v41_dcp_neg_idx", None)
                _rows_n = int(q.shape[0])
                _topk_n = int(cmp_indices.shape[-1])
                if _neg is None or _neg.shape[0] < _rows_n or _neg.shape[-1] != _topk_n:
                    # 懒分配 + 地址稳定 ⇒ 图安全；尺寸变化只发生在 eager 的首步
                    _neg = torch.full(
                        (_rows_n, 1, _topk_n), -1, dtype=cmp_indices.dtype, device=q.device
                    )
                    attn._v41_dcp_neg_idx = _neg
                _neg = _neg[:_rows_n]
                # ★ [V41-PERF] 第二次调用的 sink 是**常量 -1e30**，同样按 attn 缓存，
                #   省掉每层每步一次 `full_like` 分配 + 填充。
                if sinks is not None:
                    _ori_sinks = getattr(attn, "_v41_dcp_neg_sinks_cache", None)
                    if _ori_sinks is None or _ori_sinks.shape != sinks.shape:
                        _ori_sinks = torch.full_like(sinks, -1e30)
                        attn._v41_dcp_neg_sinks_cache = _ori_sinks
                else:
                    _ori_sinks = None
                _ori_out, _ori_lse = torch.ops._C_ascend.npu_sparse_flash_mla(
                    q,
                    ori_kv=attn.dsa_attn.swa_cache_layer.kv_cache[0],
                    cmp_kv=source_cache,
                    cmp_sparse_indices=_neg,
                    ori_block_table=_ori_bt,
                    cmp_block_table=cmp_block_table,
                    cu_seqlens_q=query_start_loc,
                    seqused_ori_kv=_ori_seqused,
                    seqused_cmp_kv=cmp_seq_lens,
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
        if cache_kind == "long_kv" and _v41_dcp_on() and has_compressed:
            n_local_heads = int(_config_value(text_config, "num_attention_heads"))
        head_dim = int(_config_value(text_config, "head_dim"))
        index_topk = int(_config_value(text_config, "index_topk"))
        smla_metadata = None
        qli_metadata = None

        if self._supports_device_ops and cache_kind in {"swa", "long_kv"}:
            cmp_seq_lens = coordinates["cache_seq_lens"] if has_compressed else None
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
