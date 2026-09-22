# `kv8-graphsafe` —— 让 int8（档 C / 档 D）在 `FULL_DECODE_ONLY` 下可用的图安全补丁

> **来源**：`S_graphfix`（`logs/049`）。**状态**：档 C 已实测通过（8 卡真权重 + 生产图模式）。
> 本目录是**自包含的交付件**：一个替换文件 + 一个可重放的生成器 + 一个接入脚本。

---

## 1. 为什么需要它（两个问题，一份补丁）

```
① dsa_v41.py:436 的 .item()（宿主同步）在 spec-decode 下必被走到：
   if query_rows == num_reqs:   ← decode 分支（device-side、capture-safe）
   else:                        ← prefill 分支（.max().item() ⇒ 捕获期炸 EE1016）
   捕获时是 spec-decode 的 decode 批（num_spec_tokens=5 ⇒ 每请求 6 行 query）
   ⇒ query_rows = 6 × num_reqs ≠ num_reqs ⇒ 误走 prefill 分支
   ★ 判据：speculative-config 里 num_speculative_tokens=5；纯 BF16（档 B）同一条链能捕获成功
     （因为 BF16 的 SWA 面根本不进这个函数）

② ★★ 档 D 的 long-KV INT8 面（`_kv8_cmp_plane`）在 spec-decode 下同样 rows != num_reqs
   ⇒ 走"整段压缩前缀重建"，页数来自 cache_seq_lens.max().item()（D2H）；
   而档 D 的 VLLM_V41_KV8_PREFILL=1 会把它换成 fused_cmp_plane3(ppr=ceil(max_cache_seq_len/bs))，
   这个 mcs 是【捕获期 _dummy_run 的 seq_lens=max_query_len=6】算出来的 ⇒ ppr≈1 页，
   而 replay 的压缩前缀有几百块 ⇒ ★ 不炸但静默读错
   ★ [SG-PPR] 探针实测（8 rank 一致）：capturing=True 时 cmp_mcs=6 ⇒ ppr = ceil(6/128) = 1
```

---

## 2. 两条改动（同一份 `attention/dsa_v41.py`，全部 env 门控、默认关）

| # | 改动 | 关键洞察 |
|---|---|---|
| **①** | `kv8_ori_plane(..., rows_bound=)` 新增**上界分支** | ★ **"必须 host 的不是【精确最大值】，而是【可证上界】"** —— 窗口带的起点仍在 device 上按真实 `q_len` 算（`query_start_loc` 差分），只有"每请求最多几页"这个 host 标量改成 `(rows_bound + window - 1) // block_size + 2`，其中 `rows_bound` 来自 `metadata.swa.max_query_len`（引擎在 CPU 张量上算好的 Python int，**不产生 D2H**）。★ `.item()` 那条 **prefill 分支原样保留**（eager only，是本任务的"不许动"项） |
| **②** | `_kv8_cmp_plane(..., graph_safe=)` 新增**按 query 行私有一段**的 selection-based 分支 | 第 `i` 行的第 `t` 个选择 → 合成下标 `i * topk + t` ⇒ scratch 页 `i * per_req + t // block_size`、页内偏移 `t % block_size`，scratch 表取 `rows * per_req` 页的恒等表。★ **全部标量来自 shape/config**（不再用捕获期冻结的 `max_cache_seq_len`） |

**与档 D 的 `VLLM_V41_KV8_PREFILL=1` 尾部包装共存**：`kv8_ori_plane` 包装在 `rows_bound` 非 None 时转交原函数；
`_kv8_cmp_plane_prefill` 包装在 `graph_safe` 且行数整除时转交类方法。

---

## 3. 文件与 md5

| 文件 | md5 | 说明 |
|---|---|---|
| `dsa_v41.py` | **`1cc9e9923cc19749872cfb2e4decc4b7`** | ★ **成品**（= 与 8 卡上实测通过的那一份逐字节相同） |
| （基底）`X_integrate/pkg-kv8pf/…/dsa_v41.py` | `75f4e565adc1b12c854a0a01271b6c4d` | 镜像是这个版本；本成品 = 它 **+253/−3** 行（1499 → 1749 行） |
| （参考）`pkg-ring/…/dsa_v41.py` | `9db97849aaa5c7284a46f49ea2e81681` | 不含 prefill triton ⇒ **档 D 不要用这份** |
| `apply_graphsafe.py` | — | **可重放的生成器**（幂等、9 个 exact-match 锚点）：`python3 apply_graphsafe.py --src <基底> --out <目标>` |
| `patch_serve_sg.sh` | — | 容器接入（幂等，只加 3 行 export） |

**★ 替换清单：`attention/dsa_v41.py` 一个文件**（其余 6~7 个 int8 挂载件不用动）。

---

## 4. env（全部默认关 ⇒ 不设就是逐字旧行为）

| env | 默认 | 作用 |
|---|---|---|
| `VLLM_V41_KV8_GRAPH_SAFE` | **0** | ★ **主开关**（档 C/D 上线必须打开 = 1） |
| `SG_CMP_LEGACY` | 0 | 诊断臂专用：只修窗口面、cmp 面留在捕获期路径（发布**不用**） |
| `SG_TRACE_PPR` | 0 | 只读探针：打印捕获期/replay 的 `mcs` 与 `ppr` |

---

## 5. 档 C 的实测判据（`logs/049`，8 卡真权重 + `FULL_DECODE_ONLY`）

| 判据 | 读数 |
|---|---|
| 起服 | ★ `EE1016 = 0`、`capture failed = 0`、就绪 **659 s**、`static_kernel` 无降级 |
| `GPU KV cache size` | **427,643**（与档 B / 档 C-eager **逐字相同** ⇒ 容量零退化） |
| `BlockStored:CPU` | **29,436**（逐字相同） |
| `CPU→GPU` | **21,188,968,448 B = 21.19 GB**（> 0 ⇒ 真命中） |
| `hits` / `queries` | **901,120 / 3,145,984**（> 0） |
| replay vs fill | **1,608.2 vs 19,880.0 ms = 12.36×** |
| `BlockRemoved:CPU` | **0**（上线监测判据） |
| `fill sha` | `d524172f9f5ae368…`（与档 B **逐字相同**） |
| ★★ `replay1 sha` | `bc2e797ab069f09ced…`（**与档 C-eager 逐字相同** ⇒ **图模式 = eager 输出**） |

★ **为什么"sha 逐字相同"是最强判据**：唯一变量就是"图 vs eager"。
而 `logs/037` 证明该服务在 `temperature=0` 下**同臂内都会抖**（32K 2/4、512 短 prompt 3/4 不同）——
⇒ **在这样抖的环境里跨模式逐字相同**，说明两条路径在这条 workload 上**数值等价**。

---

## 6. ⚠️ 两个必须知道的边界

1. ★★ **`[SG-PPR]` 证明档 D 的 cmp 面在捕获期 `ppr=1`**（实测），
   **而"越界读会怎样"在算子层仍【未确认】**：
   * **torch/ATen 层**：★ **响亮失败**（【实测·不占卡】`torch.gather` / 高级索引 / `index_select` / 数据平面越界**全部报错**，见 `raw/049-oob-probe.log`）；
   * **算子层（档 D 生产路径）**：⚠️ **最可能是静默错** —— `cmp_block_table` 是**直接喂给 `npu_sparse_flash_mla` 的 GM 描述符**，
     索引由 device 侧取址产生 ⇒ 偏移落到 buffer 之外会读到**同进程其它已分配张量**（同进程 VA 通常不触发 MMU 故障）。
   ⇒ **由决策臂 `sg-a-d-cmplegacy` 定案**（窗口面已修 ⇒ 能起服；cmp 面留在捕获期路径 ⇒ 看它"炸"还是"静默错"）。
2. ★ **调度器侧的 trace 只证明"调度器没冻结"** —— 它打在 EngineCore（图**外**），
   **不能**证明"图内部的值没被冻结"。图内部那一格由 `[SG-PPR]` 回答（答案：**`ppr=1`，被抓死**）。
