# 017 · SWA「只存窗口 chunk」：**3.1× 是真的，但不是 `alignment_chunk_count` 给的**

**日期**：2026-09-22 01:08 – 01:31（A3 本地时钟；**11 条判据臂**，单臂 1.1–2.7 min，其中每批第 1 臂多 ~1.5 min 图捕获）
　**执行**：子代理 `SWA_trim`
**机器**：A3（A3-node1），**只用 c2 槽位**（`a3_up.sh` 的 c2 = Phy-ID 7，容器 `prbench-c2`，TP1）
**上游/镜像**：与 `013`/`010`/`009` 同一套：vLLM `0.27.1` + vllm-ascend `e43cf1e9f`；模型 `agents/L1_dummy/models/model-tiny`（dummy 权重，保留 40 层 KV 结构）
**标记约定**：【实测】= 本机跑出来的原始数据；【推断】= 代码/算式推出来但没直接测；【未确认】= 没跑到。
**原始数据**：`logs/raw/017-swa-trim/`（124 个文件；含 `017-summary.md|json`、`017-probe-predicate.txt`、三份 batch 日志）

---

## 0. 一句话结论（先给判断）

| 问题 | 答案 |
|---|---|
| **判据 A：`alignment_tokens` 不再是 None？** | **【实测·是】** 挂 D2 补丁后 `[SWA_trim] alignment: blocks_per_chunk=8 full_attn_tokens_per_chunk=[1024] alignment_tokens=1024`。D2 把 `state` 组排除出 alignment 统计这一步**确实生效**。 |
| **判据 B：每请求条目 44 → 14？** | **【实测·否】** 只靠这条路径**一条都没少：还是 44 条/请求**（16 请求 = 704 条，与 `013` 逐字一致）。原因见 §3：`alignment_chunk_count` 这个钩子要求 **SWA chunk 严格小于 full-attention chunk**，而 DSV4.1 的 SWA 组 `tokens_per_chunk` 也是 1024，`_alignment_chunk_count()` 直接返回 None ⇒ **空转**。 |
| **那 3.1× 到底能不能拿到？** | **【实测·能，但要换规则】** 把存侧规则改成"**每个 SWA 组每个请求只存覆盖尾部窗口的那 1 个 chunk**"（`SWA_TRIM=window`）⇒ 条目 44→**14**，池子 **224 MiB（= 704/3.14，整整 1/3.14）** 就能跑通四条判据：`BlockStored:CPU=224`、`CPU→GPU=273 MB`、`hits=65,520`、**replay 47.5 ms vs fill 461.8 ms（9.7×）**。 |
| **判据 D：输出文本 sha256 一致？** | **【实测·一致】** 4096-token 工作负载：全存版与裁剪版的 fill/replay **每个 prompt 的输出 sha256 全部相同**（`d23082b3…`，16/16）；2048-token 前缀的**冷算参考**（`38c32f99…`）也与"走 DRAM 装载"和"走重算"两条路径都相同。 |
| **★ 那"只存窗口 chunk"安全吗？** | **【实测·不必然安全】** 同一个池子、同一份代码，把 replay 换成一趟**更短的前缀**（4096→2048）后：**`hits=0`、`CPU→GPU=0`、replay 239.8 ms ≈ 冷算 243.4 ms**，即"DRAM 里有数据也一条都不取"。机制见 §6：`_sliding_window_lookup` 要的是**该命中长度对应的那一段的尾块**，短前缀要的是更早的那一块；被裁掉后 `num_hit_chunks == 0` 会**否决整轮请求**（连 full 组的命中也一起报废）。 |
| **第 3 步边界（4096+100）** | **【实测·安全】** 4196 token（不是 chunk 整数倍）时裁剪版与全存版**逐项一致**：条目 14 vs 44、`hits=65,536`（= 16×4096，100 token 尾巴本来就不进 chunk）、replay p50 148.8 vs 151.2 ms、输出 sha256 全部相同。尾段按 `min(K, storable - segment_start)` 处理，段尾永远是"第 `storable-1` 个 chunk"，与整除无关（§5 探针第 3 组）。 |

> **一句话**：`alignment_chunk_count` 这条"上游本来就有、只是没生效"的路**在 DSV4.1 上是空转的**（不是被 `state` 组挡住的，`state` 只是第二个原因）；真正能把池子压到 1/3.14 的是"**每请求每 SWA 组只留窗口那 1 条**"，它**对"同一前缀的不同长度"工作负载会整轮归零**——本轮把两件事都测出来了。

---

## 1. 环境与做法（11 条臂，全在 c2）

| 项 | 值 |
|---|---|
| 卡 | **只 c2**（`tools/a3_chip.sh c2` 锁；**没碰** c0/c1、Phy-ID 8–15、`dsv41-a3`、`mooncake-master`） |
| 模型 / 起服 | 与 `013`/`010` 逐字一致：TP1、`--load-format dummy`、`--block-size 128`、`--enable-prefix-caching`、`--prefix-match-unit 32`、`--kv-cache-memory-bytes 1 GiB`、`ENGRAM=0` |
| 补丁入口 | `agents/SWA_trim/patch/sitecustomize.py` 用 `sys.meta_path` 把 `offloading/scheduler.py` **整文件替换**成 `agents/SWA_trim/patch/swa_scheduler.py`（**= D2 那份 md5 `0302fab4…` 的副本 + 4 处新增**：alignment/group 表日志、`SWA_TRIM` 实验开关、`import os`、store 侧入口函数）。**不写镜像**。 |
| 每臂流程 | 起服 ≈75 s → `/health` → ZMQ KV 事件探针 → fill 轮 → `POST /reset_prefix_cache` → replay 轮 → 收 `/metrics` → 停服 |
| 池子标定 | 单卡/TP1：**1 个池条目 = 1 MiB**（`kv_bytes_per_chunk=1048576`，`worker_kv_bytes_per_block=131072`×8×1），1 GiB 池 = 1024 条（与 `013` §1.2 一致，本轮每臂都打进 `meta.txt`） |
| 固定量 | `PROMPT_SALT=20260922`（**11 条臂的 prompt 逐 token 相同**，所以可以跨臂比 TTFT/sha256） |

### 1.1 臂清单与原始数据（一行一臂，全部【实测】）

`条目/请求`= `BlockStored:CPU` ÷ 请求数；`加速` = fill p50 ÷ replay p50。

| 臂 | 池 (MiB) | 存侧规则 | token：fill→replay | max_tok | 条目/请求 | `BlockStored:CPU` | `BlockRemoved:CPU` | `CPU→GPU` (MB) | load job | `hits` | `queries` | fill p50 (ms) | **replay p50 (ms)** | 加速 | 输出 sha256 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `swa-base4g-1t` | 4096 | 全存 | 4096→4096 | 1 | **44** | 704 | — | 273.0 | 16 | 65,520 | 131,328 | 461.9 | **47.4** | **9.7×** | ✅ |
| `swa-base224m-1t` | **224** | 全存 | 4096→4096 | 1 | **88** | 1,408 | 1,184 | **0** | **0** | **0** | 131,328 | 462.0 | **454.1** | **1.0×** | ✅ |
| `swa-win224m-1t` ★ | **224** | **只存窗口** | 4096→4096 | 1 | **14** | **224** | — | **273.0** | **16** | **65,520** | 131,328 | 461.8 | **47.5** | **9.7×** | ✅ |
| `swa-win272m-1t` ★ | 272 | 只存窗口 | 4096→4096 | 1 | **14** | 224 | — | 273.0 | 16 | 65,520 | 131,328 | 460.2 | **47.1** | **9.8×** | ✅ |
| `swa-base4g-8t` | 4096 | 全存 | 4096→4096 | 8 | 44 | 704 | — | 273.0 | 16 | 65,520 | 131,328 | 461.1 | 47.4 | 9.7× | ✅ |
| `swa-win272m-8t` ★D | 272 | 只存窗口 | 4096→4096 | 8 | **14** | 224 | — | 273.0 | 16 | 65,520 | 131,328 | 461.0 | **48.0** | 9.6× | ✅ |
| `swa-base4g-mixed` | 4096 | 全存 | 4096→**2048** | 8 | 44 | 704 | — | **178.4** | 16 | **32,752** | 98,560 | 461.6 | **44.4** | 10.4× | 见 §4.3 |
| `swa-win272m-mixed` ★✗ | 272 | 只存窗口 | 4096→**2048** | 8 | 24 | 384 | 112 | **0** | **0** | **0** | 98,560 | 461.2 | **239.8** | **1.9×** | 见 §4.3 |
| `swa-ref2048-4g` | 4096 | 全存 | 2048→2048 | 8 | 22 | 352 | — | 178.4 | 16 | 32,752 | 65,792 | **243.4** | 44.2 | 5.5× | ✅ |
| `swa-base4g-4196` | 4096 | 全存 | 4196→4196 | 8 | 44 | 704 | — | 273.0 | 16 | 65,536 | 134,528 | 466.1 | **151.2** | 3.1× | ✅ |
| `swa-win272m-4196` ★B | 272 | 只存窗口 | 4196→4196 | 8 | **14** | **224** | — | 273.0 | 16 | **65,536** | 134,528 | 465.4 | **148.8** | 3.1× | ✅ |

> 读表要点：
> * **44 条/请求**（4 个 full chunk + 10 个 SWA 组 × 4）在**任何** `alignment_chunk_count` 形态下都不变（`swa-base4g-1t` 与 `013` 的 `v1-d2-4g`/`v1-l1ct-4g` 三臂都是 704）；
> * `swa-base224m-1t` 是"**池子只有 1/3.14 但不裁**"的对照：池子 224 条 vs 工作集 704 条（0.318×）⇒ **0 命中**，replay 与 fill 一样慢（1.0×）；
> * `swa-win224m-1t` 是"**池子只有 1/3.14 且裁**"：条目正好 224（1.000×），**四条判据全中**（§4.2）；
> * `swa-win272m-mixed` 那一行是"没裁干净"的旁证：`BlockRemoved:CPU=112`、条目涨到 24/请求（回放轮把 2048-token 请求自己的尾块又存了一遍），但**一条都没命中**。

### 1.2 服务日志里的第一手证据（判据 A，逐字）

```
(EngineCore) [SWA_trim] CPU 卸载池: num_blocks=4096 kv_bytes_per_chunk=1048576 cpu_page_size_per_worker=1048576
             replicated_layout=False blocks_per_chunk=8 cpu_bytes_to_use=4294967296
             worker_kv_bytes_per_block=131072 world_size=1
(EngineCore) [D2_offload] KV 卸载 group 清单 n=12: [(0,'DeepseekV41FullSpec',128,8,…,True),
             (1,'DeepseekV41CompressorStateSpec',32,3,…,False), (2..11,'DeepseekV41SWASpec',128,4,…,True)]
(EngineCore) [SWA_trim] alignment: blocks_per_chunk=8 full_attn_tokens_per_chunk=[1024] alignment_tokens=1024
(EngineCore) [SWA_trim] SWA_TRIM=off group 表 (idx, tokens_per_block, tokens_per_chunk, sw_chunks,
             alignment_chunk_count, is_eagle, participates):
             [(0,128,1024,None,None,False,True), (1,32,256,None,None,False,False),
              (2,128,1024,1,None,False,True), … (11,128,1024,1,None,False,True)]
```

* `alignment_tokens = 1024`：**D2 的"非参与组不进 alignment 统计"生效了**（`state` 组 `participates=False` 被排除）；
* 但 **10 个 SWA 组的 `alignment_chunk_count` 全是 `None`** ⇒ 谓词首行 `if alignment_chunk_count is None: return True` ⇒ **全存** ⇒ 判据 B 不成立。

---

## 2. ★ 第 1 步：`is_store_reachable_swa_chunk()` 逐行（镜像内原文，D2 未改这一段）

```python
def is_store_reachable_swa_chunk(
    absolute_chunk_index: int,      # 该组第几个 chunk（本请求内 0 起，按该组自己的 chunk 粒度）
    storable_chunk_count: int,      # 本次 store 时该组"已经完整的 chunk 数"（= num_chunks）
    alignment_chunk_count: int | None,  # 该组每"full-attention 对齐段"里有多少个 chunk
    sliding_window_chunks: int | None,  # 该组的窗口 = cdiv(window, tokens_per_chunk)
    is_eagle_group: bool,
) -> bool:
    if alignment_chunk_count is None:
        return True                                   # ← 不裁（= DSV4.1 现状）
    assert sliding_window_chunks is not None
    position_in_segment = absolute_chunk_index % alignment_chunk_count
    segment_start = absolute_chunk_index - position_in_segment
    actual_segment_length = min(alignment_chunk_count, storable_chunk_count - segment_start)
    reachable_tail = sliding_window_chunks + int(is_eagle_group)
    return position_in_segment >= actual_segment_length - reachable_tail
```

**逐行语义**：

1. `alignment_chunk_count is None` ⇒ **该组不裁**（不是"裁 0 个"，而是整条策略不生效）；
2. `position_in_segment = idx % K`、`segment_start = idx - position_in_segment`：把该组的 chunk 序列按 **K 个一段**切；K = `alignment_tokens // tokens_per_chunk`（**该组**的 chunk 有多少个落在**一个 full-attention 对齐段**里）；
3. `actual_segment_length = min(K, storable - segment_start)`：**尾段按实际长度截断**（这是"最后一个不满 K 的段"的关键处理，见 §5）；
4. `reachable_tail = sw_chunks + (1 if eagle else 0)`：一段里**只有最后这几个 chunk** 是取回路径会去查的；
5. `position_in_segment >= actual_segment_length - reachable_tail` ⇒ **只保留每段的尾部 `reachable_tail` 个 chunk，裁掉每段的前面那些**。

**⇒ 回答"它裁的是每个 segment 的前几个 chunk 还是别的"**：**方向正好相反**——它裁的是**每段的前 `段长 - reachable_tail` 个**，留下**每段的最后 `reachable_tail` 个**。（`013` §4.3 里"每个 SWA 组只有窗口那 1 个 chunk 会被取回"是**某个具体工作量下**的说法，规则本身是"每段尾部"，不是一个请求只留一块，见 §3。）

### 2.1 为什么"每段只留尾部"是对的（代码推导）

* 取回时每组的命中长度只能收紧、不能放宽：`_lookup()` 里
  `max_hit_size_tokens = min(max_hit_size_tokens, tokens_per_chunk*(start_chunk_idx + num_hit_chunks))`；
* **full-attention 组**走 `_maximal_prefix_lookup()`：从第 0 个 key 起连续命中，`MISS ⇒ break` ⇒ 返回的块数必然是**该组 chunk 的整数倍**；
* 于是任何一次取回，**命中长度 H 都是 `alignment_tokens`（= full 组 chunk = 1024 token）的整数倍**；
* 对 SWA 组，`_lookup()` 只查 `offload_keys[start_chunk_idx:num_chunks]`（`num_chunks = min(cdiv(H, tpc), len(keys))`），然后 `_sliding_window_lookup(keys, required_window=sw)` **从这段的尾部往前扫**，找到连续 `sw` 个 HIT 就返回。窗口只有 128 token < chunk 1024 token ⇒ 一个 chunk 覆盖 8 个窗口 ⇒ **`required_window = 1`**，只要"覆盖 H 之前最后 128 token 的那一块"在就够；
* 那块 = **H/tpc - 1 = 该对齐段的最后一个 chunk** ⇒ 与规则完全吻合。
* **但只要 SWA chunk == alignment chunk（DSV4.1 就是），K = 1，「每段的最后一个」= 「每一个」** ⇒ 规则退化成空转。

### 2.2 ★ 关键正确性判据："只存窗口 chunk"是否**必然**安全？—— **不是**

* 「只存窗口 chunk」如果指的是**每个 SWA 组每个请求只留 1 条**（= 该请求尾部窗口那一块），那它只覆盖 **H = 本请求长度** 这一次命中；
* 一旦现实里出现**更短的前缀命中**（H' = 1024j < H，比如"同一 system prompt + 不同长度的历史"），`_sliding_window_lookup` 会去查 **chunk j-1**；它被裁掉 ⇒ 该组 `num_hit_chunks == 0` ⇒ 上游那句 **`if num_hit_chunks == 0: return 0`** 把**整个请求**（含 full 组那 4 个 chunk 的命中）一起判死；
* **【实测】本轮就是这么发生的**：`swa-win272m-mixed`（4096 全存 → 回放 2048 前缀）`hits=0`、`CPU→GPU=0`、replay 239.8 ms ≈ 冷算 243.4 ms；同样的 workload 在**不裁**的 `swa-base4g-mixed` 上是 `hits=32,752`、replay 44.4 ms。
* 结论：**上游那条"每段留尾部"是保守且正确的；"每请求只留窗口一块"是近似**，只在"复现长度 = 当初填充的长度"（或不复现更短前缀）时等价。

---

## 3. ★★ `alignment_chunk_count` 的语义 / 为什么是 None（本次的知识增量）

### 3.1 语义（三句话）

1. `alignment_tokens` = **参与卸载的 full-attention 组的 `tokens_per_block × blocks_per_chunk`**，且**必须集合里只有一个值**才生效（`len(...) == 1`），否则 None（= 退化、不裁）；
2. 逐组 `alignment_chunk_count = alignment_tokens // 该组 tokens_per_chunk`，**仅当 `alignment_tokens > tokens_per_chunk` 且 `sw_chunks < per_segment`** 时才有值；否则 None；
3. 它的含义是"**该组每 `alignment_tokens` 个 token 里，只有末尾 `sw_chunks` 个 chunk 可能被取回路径查询**"——`_maximal_prefix_lookup` 的对齐边界就是 `alignment_tokens`（§2.1）。

### 3.2 为什么"现在是 None"：**两层原因，缺一不可**

| 层 | 机制 | 状态 |
|---|---|---|
| ① 集合被污染 | `state` 组（`DeepseekV41CompressorStateSpec` → `AscendCircularBufferSpec`，block=32 ⇒ tpc=**256**）不是 `SlidingWindowSpec`/`MambaSpec`，`get_sliding_window_size_in_chunks()` 走末行 `return None` ⇒ 被当成 **full attention** 计入集合 ⇒ `{1024, 256}` ⇒ `len != 1` ⇒ `alignment_tokens = None` | **【实测】D2 补丁已修好**：`full_attn_tokens_per_chunk=[1024]`、`alignment_tokens=1024` |
| ② ★ 量纲相等 | **即便 alignment 算出来了，SWA 组的 `tokens_per_chunk` 也是 1024**（SWA block=128 × blocks_per_chunk=8），`alignment_tokens(1024) <= tokens_per_chunk(1024)` ⇒ `_alignment_chunk_count()` 直接 `return None` ⇒ 该组**全存** | **【实测】10 个 SWA 组 `alignment_chunk_count=None`，条目 44/请求不变** |

> **上游这个钩子本来是给"SWA 组的 block 远小于 MLA full 组"的模型设计的**（源码注释原话：*"e.g. DeepSeek V4 where SWA groups have much smaller block sizes than the MLA full-attention group"*；上游单测也把 `alignment_chunk_count` 参数化成 `{4, 8, 64}` —— `upstream-v41/vllm-upstream/tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py:1933`）。**vllm-ascend 的 DSV4.1 把 SWA 组也定成 block=128（与 full 组同粒度）**，所以这个钩子在 DSV4.1 上**空转**——这一点是 `013` §3.3 的【推断】里没有覆盖到的（那里假设"算出 alignment 就能裁"，实际上还要 `tpc_swa < alignment`）。

### 3.3 裁剪能力的**上界**（不占卡探针 + 算式，【实测·谓词取值】/【推断·外推】）

`agents/SWA_trim/scripts/probe_predicate.py`（直接 import 将要挂载的那份文件，**不占卡**）逐条打印谓词取值，其中：

```
== 2) alignment_chunk_count=K ⇒ 每 K 个 chunk 只留尾部 sw 个
   storable=16 alignment=4 sw=1 : ...S...S...S...S   kept=4/16
   storable=16 alignment=8 sw=1 : .......S.......S   kept=2/16
== 4) DSV4.1 真实参数代入：swa tokens_per_chunk=1024 ⇒ alignment_chunk_count=None ⇒ 全存
== 5) 假设 SWA chunk 更小：tpc=256 ⇒ acc=4 ⇒ 4096-token 请求每 SWA 组存 4 条（不裁则 16 条）
                       tpc=512 ⇒ acc=2 ⇒ 每 SWA 组存 4 条（不裁则 8 条）
                       tpc=1024 ⇒ acc=None ⇒ 每 SWA 组存 4 条（不裁则 4 条）
```

**⇒【推断·算术闭合】在安全规则下，"每 SWA 组的条目数 = 对齐段数 = token 数 / 1024"，与 chunk 大小无关。**
也就是说：**这条钩子最多只能把"每 1024 token 一个 chunk"降到"每 1024 token 一个 chunk"**——它省的是"同一个对齐段里多出来的那几块"，**省不掉"每 1024 token 至少一块"这个下界**。`44 → 14` 只能靠**放弃这个下界**（每请求只留一块 = §2.2 的近似）。

---

## 4. 第 2 步：实测四条判据（含"池子只有 1/3"）

### 4.1 判据 A（alignment 生效）—— **【实测·生效】**

见 §1.2：`alignment_tokens=1024`（不再是 None）。**但 SWA 组 `alignment_chunk_count=None`**（§3.2 ②）。

### 4.2 判据 B/C：44→14 与"1/3 池子也能全中" —— **【实测】只有换成"只存窗口"才成立**

三条同 workload 的臂放在一起看（16 × 4096 token、`max_tokens=1`）：

| | `swa-base4g-1t`（4 GiB，全存） | `swa-base224m-1t`（224 MiB，全存） | **`swa-win224m-1t`（224 MiB，只存窗口）** |
|---|---|---|---|
| ① `BlockStored(medium="CPU") > 0` | ✓ 704 | ✓ 1,408（= 2× 工作集：全轮重存） | **✓ 224（= 16 × 14）** |
| ② `kv_offload_total_bytes{CPU_to_GPU} > 0` | ✓ 273.0 MB | ✗ **0** | **✓ 273.0 MB（16 个 load job）** |
| ③ `external_prefix_cache_hits > 0` | ✓ 65,520 / 131,328 | ✗ **0** / 131,328 | **✓ 65,520 / 131,328（回放轮 99.8%）** |
| ④ replay TTFT ≪ fill TTFT | ✓ 47.4 vs 461.9（9.7×） | ✗ **454.1 vs 462.0（1.0×，退化成重算）** | **✓ 47.5 vs 461.8（9.7×）** |
| 条目/请求 | 44 | 88 | **14** |
| 池子占工作集 | 5.82× | **0.318×** | **1.000×** |

* 判据 B：**`alignment_chunk_count` 那条路 = 44 条不变**（`swa-base4g-1t`、`013` 的 `v1-d2-4g`（D2 补丁）与 `v1-l1ct-4g`（无 D2）三臂全是 704 ⇒ 这条钩子对"存多少"没有任何影响，【实测】）；
* 判据 C：**"池子只有 1/3.14 且四条判据全中"只在"只存窗口"那条规则下成立**（`swa-win224m-1t`，池子 224 MiB = 1.000×，**一条不淘汰**）；同样 224 MiB 下"全存"是 0 命中。
* 余量：`swa-win272m-1t`（272 MiB = 1.21×）同样 9.8×——按 `013` 的建议留 20% 余量时留的是**新的**工作集（224 → 272），不是旧的 704。

### 4.2.1 独立于计数器的证据：取回侧**只要 14 条**

两个臂（全存 / 只存窗口）的 load job 日志**逐字相同**，都是每请求 14 条：

```
[D2_offload] load job req=cmpl-… keys=14 group_sizes=[32, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
             src_blocks=14 dst_blocks=42
```

`group_sizes[0]=32` = full 组 4 个 chunk × 8 block；`group_sizes[1]=0` = `state` 组不参与（`009` §5.1 语义）；后面 10 个 `1` = 10 个 SWA 组**各取 1 个 chunk**（窗口只有 128 token = 1 个 block，所以每个 SWA chunk 只搬最后 1 个 block，`dst=42=32+10`）。
⇒ **"全存"版本每请求存 44 条、取回只用 14 条**这件事本轮再次【实测】（`swa-base4g-1t` 与 `swa-win224m-1t` 的 load job 行完全相同），差额 30 条正是 10 个 SWA 组各 3 条非窗口 chunk。

### 4.3 判据 D：输出文本 sha256（逐 prompt）

口径：`run_arm.sh` → `bench/kv_offload_client.py`（本轮新增 `out_sha256`）对每个 prompt 的**生成文本**算 sha256，整轮再按 prompt 序号串起来哈希；`max_tokens=8`、`temperature=0`、`ignore_eos`。

| 臂 | fill 轮 sha256（16 prompts） | replay 轮 sha256 | 逐 prompt 一致？ |
|---|---|---|---|
| `swa-base4g-8t`（全存，4096） | `d23082b36fc1146e2fa3fabc6399dad8acdee4d54d25d5645be42f081d825789` | 同左 | ✅ 16/16 |
| `swa-win272m-8t`（**只存窗口**，4096） | `d23082b3…`（与全存版**逐字节相同**） | `d23082b3…` | ✅ 16/16 |
| `swa-ref2048-4g`（2048 冷算参考） | `38c32f99ff8b4ffa93e9654ee8ad75e6876d0fc7d027cba2936508206bc3427d` | 同左 | ✅ 16/16 |
| `swa-base4g-mixed`（4096 填充 → 2048 回放，全存） | `d23082b3…`（4096 的答案） | **`38c32f99…`** = 2048 前缀的正确答案 | ✅（与冷算参考一致，说明"走 DRAM 装载"与"冷算"同解） |
| `swa-win272m-mixed`（同上，只存窗口） | `d23082b3…` | **`38c32f99…`**（**与参考一致**，虽然它是整轮重算出来的） | ✅（输出对，**但没走卸载路径**） |

**⇒ 判据 D【实测·一致】**：裁剪**没有改变任何一个 token**。注意区分两件事：`swa-win272m-mixed` 的 sha256 也是对的，但它的 `hits=0`——**"输出正确"不等于"取回成功"**，所以判据 D 不能单独当"裁剪安全"的证据（这也是本轮把 ②③④ 一起测的原因）。

---

## 5. 第 3 步：窗口边界的边界条件（4096 + 100 token）

### 5.1 谓词在"最后不满一段"上的取值（不占卡，逐字）【实测】

```
== 3) 最后一个不满 K 的段
   storable=5    alignment=4 sw=1 : ...SS   kept=2/5
   storable=6    alignment=4 sw=1 : ...S.S   kept=2/6
   storable=7    alignment=4 sw=1 : ...S..S   kept=2/7
   storable=9    alignment=4 sw=1 : ...S...SS   kept=3/9
   storable=17   alignment=4 sw=1 : ...S...S...S...SS   kept=5/17
```

**段尾永远是"第 `storable-1` 个 chunk"**，与是否整除无关：`actual_segment_length = min(K, storable - segment_start)` 让**最后那个不满 K 的段按真实长度算**尾巴 ⇒ 只要"请求当前的最后一个完整 chunk"被留下，命中就能成立。

### 5.2 端到端（4196 token = 4096 + 100）【实测】

| | `swa-base4g-4196`（全存） | `swa-win272m-4196`（只存窗口） |
|---|---|---|
| 条目/请求 | 44 | **14** |
| `BlockStored:CPU` | 704 | **224** |
| `CPU→GPU` | 273.0 MB（16 job） | **273.0 MB（16 job）** |
| `external_prefix_cache_hits` | 65,536 / 134,528 | **65,536 / 134,528** |
| replay p50 / fill p50 | **151.2** / 466.1（3.1×） | **148.8** / 465.4（3.1×） |
| 输出 sha256 | ✅ | ✅（与全存版逐 prompt 相同） |

**结论【实测】**：`4196 = 4×1024 + 100`，第 5 个 chunk 只有 1 个 block（33 blocks = 4 完整 chunk + 1 块）**本来就不进 chunk**（`storable = blocks//8 = 4`），裁剪版留下的正是**第 3 号 chunk**（覆盖 token 3072–4095）——而取回路径要的就是它（H=4096 ⇒ H/1024-1 = 3）⇒ **边界正确、TTFT 与全存版一致（148.8 vs 151.2 ms，差 2.4 ms ≈ 噪声）**。

---

## 6. 机制：为什么"只存窗口"会在短前缀上整轮归零

1. 填充轮：请求 L=4096 token ⇒ SWA 组有 4 个 chunk；"只存窗口"只留 **chunk 3**（覆盖窗口的最后 128 token）；
2. 回放轮（同一个 4096）：`_maximal_prefix_lookup` 给 H=4096 ⇒ SWA 组查 `offload_keys[0:4]`，`_sliding_window_lookup` 从尾部扫，第 1 个 key = **chunk 3** 命中 ⇒ 整个请求 H=4096 命中 ✓（本轮 `swa-win224m-1t`：9.7×）；
3. 回放轮（2048 前缀，同池同代码）：H 只能到 2048 ⇒ SWA 组查 `[0:2]`，从尾部扫：chunk 1 ⇒ **MISS**；chunk 0 ⇒ **MISS** ⇒ `num_hit_chunks = 0` ⇒ **`return 0`**；
4. 这个 `return 0` 是**请求级**的：**连 full 组那 2048 token 的命中一起作废**（本轮 `swa-win272m-mixed`：`hits=0`、`CPU→GPU=0`、replay 239.8 ≈ 冷算 243.4 ms；池子里其实有数据，`BlockRemoved:CPU=112` 说明它还在被 LRU 淘汰）。
5. 对照组（不裁）：同样的短前缀回放 `hits=32,752`、`CPU→GPU=178.4 MB`、replay 44.4 ms ⇒ **归零是裁剪造成的，不是 workload 造成的**。

> 换句话说：**上游的规则保的是"所有可能命中长度"，"只存窗口"保的是"只有本请求长度那一个命中长度"**。前者不省条目（§3.3），后者省 3.1× 但把跨长度前缀共享的命中变成 0。

---

## 7. 对 A2 的意义与建议（要分清"能省"和"安全省"）

1. **【实测】`alignment_chunk_count` 这条路对 A2 没有收益**：DSV4.1 的 SWA 组与 full 组同 chunk 粒度（1024 token），钩子空转 ⇒ 池子需求仍是 **44 条/请求**（8 卡真权重口径见 `009`/`013` §3.3）。任务书里"44→14、池子需求降 3.1×"**不是这个补丁能给的**。
2. **【实测】"只存窗口"确实能拿到 3.1×**（`swa-win224m-1t`：池子 224 MiB = 工作集的 1.000×、四条判据全中、sha256 一致），**但只在"同一前缀不做更短/不同长度的回放"时成立**；一旦有短前缀回放，命中率被 `return 0` 一票否决，**连 full 组的命中一起丢**（`swa-win272m-mixed`）⇒ 对"system prompt + 变长历史"的在线服务**不能直接上线**。
   * 若某个 A2 场景确实是"固定前缀 + 固定长度请求"（例如批量离线复现、固定 prompt 的评测），这 3.1× 就是白拿的：**【推断·按 `009`/`013` §3.3 的记账公式】**同样 442 GiB 宿主余量（建议 ≈260 GB 池）能覆盖的**同时驻留请求数 ×3.14**；否则不要把这条当成"池子直接减到 1/3"的计划。
3. **【推断】要安全地拿 3.1×，得动下面两处之一（都不在本轮预算内）**：
   * **让 SWA 组的 chunk 粒度真正小于 full 组**（上游设计假设的形态）：按 §3.3 的算式，条目数**不会**减少（每 1024 token 仍要 1 条），但每条变小时**池子字节数**会降——例如 SWA chunk 缩到 1 block（128 token）⇒ 每请求 4 MiB(full) + 40×0.125 MiB ≈ **9 MiB vs 现在 44 MiB（4.9×）**；代价是要把"**per-group** `blocks_per_chunk`"打通（现在 `blocks_per_chunk` 与池子页记账 `worker_kv_bytes_per_block×blocks_per_chunk` 都是**全局**的，`vllm/v1/kv_offload/config.py::build_offloading_config` 里只有一个 `OffloadingCacheConfig`）。
   * **把 `_lookup()` 的"一组 0 命中 ⇒ 整轮 0 命中"改成"只收紧命中长度"**（`if num_hit_chunks == 0: return 0` 换成"该组不参与收紧"），这样"只存窗口"最坏也只是少命中、不会整轮报废；这是上游语义改动，风险与收益都要单独评估。
4. **【未确认】8 卡真权重上的 3.1×**：本轮的 3.1× 是在**单卡 tiny（TP1、`num_copies=1`、1 MiB/条目）**上测的；8 卡的池子按 `num_copies=world_size` 记账（34.6 MB/条目），条目数的规律与卡数无关（§3.3 的算式），但"`hits=0` 的断崖位置"与 `concurrency>1` 的交错顺序**没测**。

### 7.1 cannbot 查证（按 `a2/AGENTS.md` §6 的要求）

本轮**没有写算子/kernel、也没有做量化数值验证**，所以 `ops/*`、`model-infer-quantization` 那几节不适用。但本任务与 **KV cache/attention 布局**有关，按 §6 的索引读了
`cannbot-skills/model/model-infer-kvcache/SKILL.md`（A3 上 `~/projects/dsv41/src/cannbot/vendor/cannbot-skills/`）：

* 该文档写的是**模型/算子层**（FA + PA + TND、`slot = block_table[...]×block_size + offset`、滑窗用 `sparse_mode=4` + `pre_tokens`）：`SKILL.md:102-110,224-258`；
* 与本题最相关的一条：*"长序列 `KV_len > sliding_window` 的正确性必须靠**模型层**保证——环形 buffer 写 cache、或 `actual_seq_lengths_kv` 截断到窗口长度，不是 op 层负责"*（`SKILL.md:241`）；
* 它**没有**、也不打算规定"**卸载层的 chunk 保留策略**"（外部 KV 池的存/取规则）——那属于 vLLM 的 `kv_offload` 调度器，不在这份 skill 的范围内。
* ⇒ **采纳情况**：本文档不改变本轮结论；它反而解释了一个观察：DSV4.1 的 SWA 资源在 **vllm-ascend 侧是"整序列 KV + 窗口化注意力"**（所以每请求能存出 32 个 SWA chunk），而不是 gpt-oss 那种"cache 不随序列增长"的固定环（`SKILL.md:298`）——这正是"存了 30 条用不上"的另一半原因。**没有采纳它的任何写法改动**（本轮不改模型/算子）。

---

## 8. 未确认 / 风险

| 项 | 状态 |
|---|---|
| 8 卡真权重的条目数与 3.1× | 【未确认】本轮只测单卡 tiny；`009`/`013` 的 8 卡数据只到"44 条/请求"这一层 |
| `concurrency > 1` | 【未确认】全程 `concurrency=1`；并发交错会改变 fill/replay 顺序（`013` §4.2 的相位锁定），裁剪后的断崖形态可能变化 |
| "只存窗口"在真实会话里的命中损失 | 【未确认】本轮只做了一个 4096→2048 的短前缀对照；没有扫"多个前缀长度混合"的分布 |
| 池子 224 MiB 的鲁棒性 | 【实测·单点】1.000× 全中、1.21× 也全中；但 `013` 已证明拐点极窄（0.989× 就开始掉），**上线要按 1.2× 配** |
| 数值正确性 | 【实测·弱】只比了 `temperature=0` 的 8-token 输出 sha256（三条路径一致）；**没有**做 logprob/长生成对比 |
| `state` 组不参与卸载的语义 | 【推断】沿用 `009` §2.5 / `013` §5，本轮未新增证据 |

---

## 9. 复现（全部在 c2；单臂 1.2–2.6 min）

```bash
# 0) 一次性：把补丁 + bench 送到 A3（coscli，不走 ssh 管道）
bash a2/scripts/cos-xfer.sh put <pkg>.tgz share/xfer/017-swa-trim-pkg.tgz
#   A3 上：coscli cp cos://uploads-new/share/xfer/017-swa-trim-pkg.tgz ~/tmp/ \
#           && tar xzf ~/tmp/017-swa-trim-pkg.tgz -C ~/projects/dsv41-upstream-pr/agents

# 1) 不占卡：谓词/算式探针（直接 import 将要挂载的那份文件）
bash tools/a3_chip.sh c2 --timeout 300 --name swa-probe -- \
  python3 /work/agents/SWA_trim/scripts/probe_predicate.py

# 2) 三条批量臂集（A: 主判据 / B: 正确性+短前缀 / C: 4096+100 边界）
bash tools/a3_chip.sh c2 --timeout 5400 --name swa-batch-A -- \
  env BATCH=A bash /work/agents/SWA_trim/scripts/run_batch.sh
bash tools/a3_chip.sh c2 --timeout 5400 --name swa-batch-BC -- \
  bash -c 'env BATCH=B bash /work/agents/SWA_trim/scripts/run_batch.sh;
           env BATCH=C bash /work/agents/SWA_trim/scripts/run_batch.sh'

# 3) 观察点
#   out/<tag>.meta.txt          [SWA_trim] alignment / group 表 / 池子 num_blocks（判据 A）
#   out/<tag>.kv_events.log     BlockStored:CPU（判据 B、C-①）
#   out/<tag>.metrics_after.txt kv_offload_total_bytes{CPU_to_GPU}（C-②）、external_prefix_cache_hits（C-③）
#   out/<tag>.client.json       rounds[].ttft.p50_ms（C-④）与 *_out_sha256_all（判据 D）
#   python3 scripts/summarize.py --dir out <tag…>      # 压成一行
```

**开关**：`SWA_TRIM=off|window`（默认 `off` = D2/上游行为）；`REPLAY_PROMPT_TOKENS=M`（replay 轮用长度为前 M 的**前缀**，同一 `PROMPT_SALT` 下逐 token 相同）。

---

## 10. 产物清单

| 文件 | 作用 |
|---|---|
| `a2/logs/017-20260922-swa-trim.md` | 本日志 |
| `a2/agents/SWA_trim/patch/swa_scheduler.py` | D2 产物副本（md5 `0302fab4…`）+ 4 处新增：`import os`、alignment/group 表日志、`_swa_trim_keep_chunk()` 与 `SWA_TRIM` 开关、store 侧改走该入口（**md5 `cab680185f272c71b00eb8b8f4f63cdd`**） |
| `a2/agents/SWA_trim/patch/sitecustomize.py` | 进程内挂载（meta_path 整文件替换）+ 池子只读日志 + `wo_a` dummy 适配 |
| `a2/agents/SWA_trim/scripts/run_arm.sh` | 单臂运行器（`SWA_TRIM` / `REPLAY_PROMPT_TOKENS` / 固定 `PROMPT_SALT`） |
| `a2/agents/SWA_trim/scripts/run_batch.sh` | 臂集 A/B/C |
| `a2/agents/SWA_trim/scripts/probe_predicate.py` | **不占卡**：谓词取值 + DSV4.1 参数代入 + 假设形态对照 |
| `a2/agents/SWA_trim/scripts/summarize.py` | 一臂一行（含 alignment / 条目每请求 / sha256） |
| `a2/agents/SWA_trim/bench/kv_offload_client.py` | L1/`V1_verify` 版的副本 + `--replay-prompt-tokens` + **输出文本 sha256** |
| `a2/logs/raw/017-swa-trim/` | **11 条臂的原始产物**（124 个文件）+ `017-summary.{md,json}` + `017-probe-predicate.txt` + 三份 batch 日志 |
| `a2/logs/raw/017-swa-trim/017-swa-trim-raw.tgz` | 同上打包（360 KB，来自 A3 `agents/SWA_trim/out/`） |

---

## 11. 红线遵守

* **只用 c2**（`tools/a3_chip.sh c2` 全程持锁）；**没碰** c0/c1、Phy-ID 8–15、`dsv41-a3`、`mooncake-master`；
* **不手设** `ASCEND_RT_VISIBLE_DEVICES`（由锁脚本注入）；
* **不用 `/tmp`**：本机 `~/tmp/20260922/swa_trim/`（`source a2/scripts/tmpdir.sh swa_trim`），A3 侧临时区 `~/tmp/swa_trim/`；
* **没写** `upstream-v41/`（只读）；**没改镜像内任何源码**（全部 `PYTHONPATH` + `sitecustomize` 进程内替换，只影响自己起的服务）；
* 跨机传文件走 **coscli**（`a2/scripts/cos-xfer.sh`，key `share/xfer/017/…`），ssh 只跑命令（一律 `-o ControlPath=none`）；
* 每臂收尾由 `run_arm.sh` 的 trap 停掉自己起的 `vllm serve`（共享容器本体保留）；
* 结论全部标 **【实测】/【推断】/【未确认】**；缺的格子标 `—`（如 `swa-win*` 的 `BlockRemoved:CPU` 确实是 0）。
