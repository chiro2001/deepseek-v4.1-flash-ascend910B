"""Ascend 侧 CudagraphDispatcher 补丁（base 镜像版 + **dynamic speculative decoding** 支持）。

本文件是我们对 base 镜像里 `vllm_ascend/patch/worker/patch_cudagraph.py` 的
**整文件替换**（挂载到同一路径）。base 版只做一件事：改
`CudagraphDispatcher._create_padded_batch_descriptor` 的判定条件，让
`cudagraph_mode == FULL_DECODE_ONLY` 也能走 uniform-decode 图（否则 vLLM 的
FULL 模式会报错）。那段逻辑**逐行保留**，见 `_create_padded_batch_descriptor`。

我们在它之上加三件事，全部**只在 dynamic SD 打开时生效**（关掉时逐字节等价）：

  1. `_create_padded_batch_descriptor`：除 `uniform_decode_query_len` 外，还认
     一个"本步 query_len"（`_step_uniform_query_len`），因为 dynamic SD 下同一
     个 batch 可能走 K=7（query_len=8）或 K=0（query_len=1）两条形状。
  2. `initialize_cudagraph_keys`：为**每个**会出现的 query_len 各捕一组
     decode 图（只有两者的最大桶都缺席时才会退化，见下）。
  3. `adjust_cudagraph_sizes_for_spec_decode`：dynamic SD 下**不做**
     "把所有桶上取整到 query_len 的倍数"这一步 —— 那会把 K=0 需要的小桶
     (1,2,3,4) 全部吃成 8，等于没有 K=0 的图。

为什么必须动这里：`_bs_to_padded_graph_size` 是**全局单表**，dispatch 时用
`num_tokens_padded // uniform_decode_query_len` 反推 `num_reqs`。若 K=0 的步
（num_tokens = 请求数）仍按 query_len=8 去算，会得到 `4 % 8 != 0` —— 要么断言
炸掉，要么错配到"1 个请求 × 8 token"的图上，属于静默算错。

设计约束（有意为之）：

* **不做推断**：query_len 由调用方（runner）显式告知，不靠"num_tokens 能被谁
  整除"来猜 ——  `num_tokens=8` 既能是 "8 请求 × 1" 也能是 "1 请求 × 8"。
* **缺图不崩**：若某步的形状确实没有图，就让 descriptor 退化成非 uniform
  （下游 `dispatch()` 查不到 FULL 键时会自然回落到 eager），并只告警一次；
  宁可慢，不要在 worker 里 raise 把整个引擎带走。
"""

from __future__ import annotations

from vllm.config import CUDAGraphMode
from vllm.config.compilation import CompilationConfig
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

logger = init_logger(__name__)

#: 只在第一次缺图时告警，避免每步刷日志（D 侧每 10 s 就有 ~780 条吞吐日志，
#: 逐条告警会把 serve.log 淹掉）。
_WARNED_NO_GRAPH: set[tuple[int, int]] = set()


def _dynamic_decode_query_lens(vllm_config) -> tuple[int, ...] | None:
    """从 speculative_config 推导"会出现哪些 decode query_len"。

    Returns:
        `None`（未开 dynamic SD，调用方全部走原路径）；
        否则是排序去重后的 query_len 元组，例如 `(1, 8)` 表示
        "K=0（1 token/req）与 K=7（8 token/req）都要有图"。

    `num_new_sampled_tokens_per_step` 恒为 1（MRV1 里
    `decode_query_len == 1 + num_speculative_tokens`），所以
    `query_len = K + 1`，与上游 GPU 路径 `cudagraph_utils.py` 的算法一致。
    """
    spec = getattr(vllm_config, "speculative_config", None)
    if spec is None:
        return None
    schedule = getattr(spec, "num_speculative_tokens_per_batch_size", None)
    if not schedule:
        return None
    max_k = getattr(spec, "num_speculative_tokens", 0) or 0
    base_ql = 1 + max_k
    lens = {1 + min(max_k, max(0, int(entry[2]))) for entry in schedule}
    # 兜底：配置里写的 K 若被 min(max_k, …) 钳过，实际形状仍以钳后为准，
    # 这里把 base_ql 也并进来，保证"K 未在表里出现"时至少有一张图。
    lens.add(base_ql)
    return tuple(sorted(lens))


def _v41_extra_query_lens(self) -> tuple[int, ...]:
    """懒算"会出现哪些 decode query_len"，并缓存在实例上。

    **为什么必须懒算**：原先是在 `CudagraphDispatcher.__init__` 里算好存到实例上，
    但实测（a3-21, 2026-09-28）`patch_cudagraph.py` 这个模块**晚于**
    `GPUModelRunner.__init__` 里 `CudagraphDispatcher(...)` 的构造才被 import
    ⇒ 那个实例上根本没有这个属性 ⇒ `getattr(..., None)` 得到 None ⇒
    整条"多 query_len"路径被静默跳过（起服日志里看不到任何 dynamic-spec INFO，
    只有 12/20 的降级 WARNING）。
    懒算把对 import 顺序的依赖彻底去掉：只要用的时候模块已加载即可
    （而 `initialize_cudagraph_keys` / `dispatch` 都远晚于 import）。
    """
    cached = getattr(self, "_v41_qlens_cache", None)
    if cached is None:
        cached = _dynamic_decode_query_lens(self.vllm_config) or ()
        self._v41_qlens_cache = cached
    return cached


def _create_padded_batch_descriptor(
    self,
    num_tokens: int,
    uniform_decode: bool,
    has_lora: bool,
    num_active_loras: int = 0,
) -> BatchDescriptor:
    max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
    uniform_decode_query_len = self.uniform_decode_query_len
    num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]

    # [DYNAMIC-SPEC] 本步真正生效的 query_len。dispatch 时由 runner 逐帧写入；
    # 建图（initialize_cudagraph_keys）时为 None ⇒ 用静态值，与 base 版等价。
    step_ql = getattr(self, "_step_uniform_query_len", None)
    if step_ql is not None:
        uniform_decode_query_len = int(step_ql)

    # FULL mode should not be treated as uniform decode
    if (
        uniform_decode
        and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL)
        and self.cudagraph_mode != CUDAGraphMode.FULL
    ):
        if num_tokens_padded % uniform_decode_query_len != 0:
            # [BUILD vs RUNTIME] 两种时机要区别对待：
            #   * **建图期**（initialize_cudagraph_keys 正在枚举键）：这个桶对
            #     该 query_len 根本没有合法 uniform 形状 ⇒ 返回 None，让
            #     add_cudagraph_key 跳过它。**不能**退化成非 uniform ——
            #     那会往 FULL 键集里塞一个伪键，而且会被 set_draft_graph_params
            #     当成合法捕获尺寸（实测：raw 桶里的 12/20 就是这么漏进去的，
            #     与运行期 draft 元数据 rows 不匹配的崩溃高度相关）。
            #   * **运行期 dispatch**：真出现这种形状说明覆盖不全，退化成非 uniform
            #     （下游回落 eager），只慢不错、不 raise。
            if getattr(self, "_v41_building_keys", False):
                return None
            key = (num_tokens_padded, uniform_decode_query_len)
            if key not in _WARNED_NO_GRAPH:
                _WARNED_NO_GRAPH.add(key)
                logger.warning(
                    "[dynamic-spec] no uniform decode graph for "
                    "num_tokens_padded=%s query_len=%s (padded %% ql != 0); "
                    "falling back to non-uniform dispatch for this shape.",
                    num_tokens_padded,
                    uniform_decode_query_len,
                )
            uniform_decode = False
            num_reqs = min(num_tokens_padded, max_num_seqs)
        else:
            num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
    else:
        uniform_decode = False
        num_reqs = min(num_tokens_padded, max_num_seqs)

    return BatchDescriptor(
        num_tokens=num_tokens_padded,
        num_reqs=num_reqs,
        uniform=uniform_decode,
        has_lora=has_lora,
        num_active_loras=num_active_loras,
    )


_orig_add_cudagraph_key = CudagraphDispatcher.add_cudagraph_key


def add_cudagraph_key(self, runtime_mode, batch_descriptor) -> None:
    """建图期允许 `_create_padded_batch_descriptor` 返回 None（该桶无合法形状）。

    直接传给原函数会把 None 塞进键集合，运行期 `dispatch` 匹配到它就会
    拿到一个没有 num_tokens/num_reqs 的描述符 —— 属于静默错配。这里挡掉。
    """
    if batch_descriptor is None:
        return
    return _orig_add_cudagraph_key(self, runtime_mode, batch_descriptor)


_orig_initialize_cudagraph_keys = CudagraphDispatcher.initialize_cudagraph_keys


def initialize_cudagraph_keys(
    self,
    cudagraph_mode: CUDAGraphMode,
    uniform_decode_query_len: int = 1,
) -> None:
    """建图：先按静态 query_len 建一遍（原逻辑），再为其余 query_len 各建一遍。

    调用原函数（而不是复制它的逻辑）的两点理由：
      * `_bs_to_padded_graph_size` / lora cases / mixed-mode 键都由它维护，
        复制一份会随上游漂移；
      * 键存在 set 里，重复添加自然去重，所以"多建一遍"是幂等的。

    为了让原函数用**指定**的 query_len，这里临时改两个全局量：
      * `self.uniform_decode_query_len`（原函数用它与
        `_create_padded_batch_descriptor` 算 num_reqs）；
      * `compilation_config.cudagraph_capture_sizes`（原函数按
        `x <= ql * max_num_seqs and x >= ql` 过滤桶，但不检查整除 ——
        而 dynamic SD 下我们故意保留了非整倍的桶（12/20 对 ql=8），
        不筛掉它们会在建图期就撞上整除断言）。
    收尾时恢复原值并**重算 padding 表**（它由桶列表派生，不能停留在子集状态）。
    """
    extra_lens = self._v41_extra_query_lens()
    # [OPT-IN] 只对**显式开启**的 dispatcher 做多 query_len 建图。
    # 不能只看 query_lens：draft proposer 用同一个 vllm_config 建了自己的
    # dispatcher（vllm/v1/spec_decode/llm_base_proposer.py:164），自动生效会给
    # draft 也捕一套 ql=1 的图 —— 那些图永远不被 dispatch（K=0 时 `_propose`
    # 提前返回、不走 draft），纯属浪费捕获时间。
    # 该标记由 runner 侧补丁在调本函数之前写入主模型的 dispatcher。
    if not extra_lens or not getattr(self, "_v41_dynamic_sd_enabled", False):
        return _orig_initialize_cudagraph_keys(
            self, cudagraph_mode, uniform_decode_query_len
        )

    # ⚠️ 用 **warning** 级别：这条是"多 query_len 路径真的走了"的**唯一判据**，
    # 必须始终可见。2026-09-28 第三轮实测 —— 写成 logger.info 时被日志级别过滤，
    # 于是无法从 serve.log 证明两个 query_len 都建了图，只能靠行为推断。
    logger.warning(
        "[dynamic-spec] building decode graphs for query_lens=%s "
        "(one graph set per query_len; raw buckets preserved)",
        tuple(extra_lens),
    )

    cc = self.compilation_config
    saved_sizes = cc.cudagraph_capture_sizes
    saved_udql = self.uniform_decode_query_len
    prev_build = getattr(self, "_v41_building_keys", False)
    self._v41_building_keys = True
    try:
        # 先按调用方给的 query_len 建一遍（含基础设施：padding 表 / lora cases /
        # mixed-mode 键）。建图期标记会让非整倍桶被跳过而不是变成伪键。
        _orig_initialize_cudagraph_keys(self, cudagraph_mode, uniform_decode_query_len)
        for ql in extra_lens:
            if ql == uniform_decode_query_len:
                continue
            subset = [s for s in (saved_sizes or []) if s >= ql and s % ql == 0]
            if not subset:
                logger.warning(
                    "[dynamic-spec] query_len=%s has no eligible capture bucket "
                    "in %s; K=%s steps will fall back to eager.",
                    ql,
                    saved_sizes,
                    ql - 1,
                )
                continue
            cc.cudagraph_capture_sizes = list(subset)
            self.uniform_decode_query_len = ql
            _orig_initialize_cudagraph_keys(self, cudagraph_mode, ql)
    finally:
        self._v41_building_keys = prev_build
        cc.cudagraph_capture_sizes = saved_sizes
        self.uniform_decode_query_len = saved_udql
        if self.cudagraph_mode != CUDAGraphMode.NONE:
            self._compute_bs_to_padded_graph_size()


_orig_adjust_cudagraph_sizes = CompilationConfig.adjust_cudagraph_sizes_for_spec_decode


def adjust_cudagraph_sizes_for_spec_decode(
    self,
    uniform_decode_query_len: int,
    tensor_parallel_size: int,
):
    """dynamic SD 下**跳过**"把桶上取整到 query_len 倍数"。

    那一步是为单一 query_len 设计的：它会把 [1,2,3,4,8,12,…] 全部并成
    [8,16,24,32]，于是 K=0（1 token/req）需要的 1/2/3/4 桶消失，
    等于 K=0 根本没有图。dynamic SD 需要的是"两个 query_len 的桶并存"，
    由 `initialize_cudagraph_keys` 按 ql 各自筛选，而不是在这里提前合并。

    只在 `_v41_dynamic_sd` 标记存在时跳过；静态路径逐字节等价。
    """
    if getattr(self, "_v41_dynamic_sd", False):
        logger.warning(
            "[dynamic-spec] keeping raw cudagraph_capture_sizes=%s "
            "(skip rounding to a single query_len=%s); per-query_len buckets are "
            "selected in initialize_cudagraph_keys.",
            self.cudagraph_capture_sizes,
            uniform_decode_query_len,
        )
        return
    return _orig_adjust_cudagraph_sizes(
        self, uniform_decode_query_len, tensor_parallel_size
    )


# 注意：**不再 patch `CudagraphDispatcher.__init__`**。实测它的 import 时机晚于
# dispatcher 的构造（见 `_v41_extra_query_lens` 的说明），靠实例属性会静默失效；
# 改用懒算 + runner 侧显式 opt-in。`_step_uniform_query_len` 则由 runner 在
# dispatch 前用 `setattr` 写入（缺省用 `getattr(..., None)`，不需要 init）。
CudagraphDispatcher._v41_extra_query_lens = _v41_extra_query_lens
CudagraphDispatcher._create_padded_batch_descriptor = _create_padded_batch_descriptor
CudagraphDispatcher.add_cudagraph_key = add_cudagraph_key
CudagraphDispatcher.initialize_cudagraph_keys = initialize_cudagraph_keys
CompilationConfig.adjust_cudagraph_sizes_for_spec_decode = (
    adjust_cudagraph_sizes_for_spec_decode
)
