# CHANGELOG.md —— v3 → v4 → v5 → v6 → v7 → v8 逐项 diff

# ★ v8（2026-09-20）—— Engram 完全入图：查表从 host 搬到 device，删掉整条 host 路径

> 这是本包**第一次动 Engram 的执行位置**：表仍然 206 GiB 常驻 host DRAM（进不了 HBM），
> 但改由**设备算子直接索引**，于是 `d2h` 同步 / 分片 / `all_gather` / `all_to_all` /
> `broadcast` / `h2d` 六条 host 路径整体消失，查表变成主图里的一张 ACLGraph。

## 0. 结论

| 项 | host 路径（v7 及以前） | device-index（v8） |
|---|---|---|
| 每步**同步 host 时间** | 3.379 ms（`d2h` 1.667 + `hash` 0.074 + `route` 1.638） | **0.058 ms** |
| decode 单流 ms/step（并发 1） | 29.5 | **28.4** |
| decode 并发 4 ms/step | 35.3 | **32.1** |
| HBM 占用 | 基线 | **一致**（13.05 vs 13.06 GiB） |
| per-rank DRAM 分片 | 25.75 GB/rank/层 | **不再需要**（每 rank 读整张表） |

`route` 从 2.462 ms 掉到 0.058 ms 的原因**不是算得更快**，而是 `Graph.replay()` 是
**异步入队** —— host 只花 58 µs 把图提交出去，设备侧的 0.695 ms 与后续重叠。

## 1. 原理

A3 的 AI Core 可以直接寻址 host 映射内存（`aclrtHostMemMapCapabilities` 返回
AIC/AIV = SUPPORTED）。用 `aclrtHostRegister(..., MAPPED)` 把表所在的
`mmap` 注册成设备可寻址，`torch.index_select` 就能直接读 host DRAM。

三个关键设计点（都有实测支撑）：

1. **decode 与 prefill 分流**：整表 gather 在 n=6/288 时都是 0.083 ms、**与表大小无关**，
   而 n=393216 时是 16.75 ms —— 代价按**行**发生，decode 规模下不存在。
   所以 decode 走整表直索（且天然可捕获），prefill 才需要分段。
2. **零拷贝捕获**：图直接捕获在模型自己的常驻 buffer 上（`input_ids.gpu`、`positions`、
   `query_start_loc.gpu`、block table），`req` 的 searchsorted 也放图内。
   此前试过"拷进私有 buffer"，**实测更差**：H2D 会阻塞等设备队列排空，
   `route` 从 0.35 ms 涨到 2.0(n=6)/5.4(n=24)。
3. **每 batch shape 一张图 + 指针校验**：bucket key 必须含 `(n, n_reqs, block_width)`
   —— n=12 可能是 2 请求×6 也可能是 12 请求×1，只按 n 分键会**静默喂错布局**。
   零拷贝会锁地址，所以每次重放前比对四个输入的 `data_ptr`+shape，不一致就退回 eager。

## 2. 新增文件

| 文件 | 作用 |
|---|---|
| `patches/files/engram_device_index.py` | `HostMappedSafetensors` / `HostMappedEngramTable` / `DeviceNgramHash`（向量化历史）/ 能力探测 |
| `patches/files/engram_graph.py` | 每个 batch shape 一张 ACLGraph，零拷贝 + 指针校验 |
| `tools/probe_a2_hostmap.py` + `tools/run_probe_hostmap.sh` | 一条命令判定某台机器能否启用 device-index |

> v8 开发期还曾改过 `patches/files/model_runner_v1.py`（device_metadata 自愈护栏），
> **已整块撤销并从包里删除**，原因见 §4.3 —— 那是本轮最重要的一条教训。

## 3. ★ 默认 `auto`：为什么不在 A2 上想当然

`ENGRAM_DEVICE_INDEX` 的取值：

| 值 | 行为 |
|---|---|
| **`auto`（默认）** | 起服时**探测** `aclrtHostRegister` 一个可写映射并让设备读它；通过则启用，否则**静默回退 host 路径**（功能完全不变） |
| `1` | 强制启用；探测失败即抛错（A3 验收建议用这个，避免"以为开了其实回退了"） |
| `0` | 强制关闭 |

**为什么默认不是 1**：该能力只在 A3（910C）实测过，A2（910B3）**从未在同一台机器上验证**。
而且本项目自己的 `docs/A2_VS_A3_DIFF.md` §5 记着一条反例：A3 上
`offload.get_dva(pinned_ptr)` 返回 0，AIV 解引用**已注册的 pinned 地址**会报
`507035 MTE invalid GM address` ⇒ "registered host memory" ≠ "device kernel 可直接解引用"。

**资料侧结论是"支持"**（华为官方零拷贝样例 `0_simple_zero_copy` 的产品表含
Atlas A2 训练/推理系列，样例把映射地址当 `GM_ADDR` 传给 AscendC Kernel 用
`DataCopy` 直接读写；`910B` 的 `NpuArch=2201` 也不在唯一的 `arch5162` 不支持清单里）。
但**文档承诺 ≠ 现场成立**，所以仍然探测。

A2 上一条命令即可拿到终局答案：

```bash
IMAGE=<你的镜像> DEV=<空闲卡> bash tools/run_probe_hostmap.sh
# 退出码 0 = 支持，3 = 不支持，1 = 探测本身出错
```

## 4. 同时修掉的工程问题

### 4.1 ★ A3 默认挂载补丁（否则跑的是未优化版本）

`serve_a3.sh` 用的是**官方镜像**（`quay.nju.edu.cn/...:deepseek-v4.1-flash-a3`），
里面没有本包的补丁。此前 `PATCH_MODE` 继承 `serve_a2.sh` 的默认值 `baked`，
于是 A3 用户按 README 起服会跑**未优化版本，而且不报任何错**。
现在 `serve_a3.sh` 默认 `PATCH_MODE=mount`。

### 4.2 ★ static kernel 缓存挂载点错了（每次重启都冷编译）

`torch_npu` 的 `npugraph_ex/.../static_kernel.py` 用 `Path.cwd()` 决定产物位置
（`base_dir = Path.cwd().resolve()`，`base_output_dir = script_dir / "static_kernel_compile_outputs"`），
而容器是 `-w /workspace` 起的 ⇒ 产物落在 **`/workspace/static_kernel_compile_outputs`**。

旧脚本挂的是 `/vllm-workspace/...` —— 那一层**永远收不到东西**，后果：

1. **每次重启都冷编译**（A3 实测多花 ~5 min，A2 首次 15–20 min）；
2. 脚本自己的 "skcache 命中" 检查看的是宿主目录，因此还会**误报命中**；
3. 宿主目录只剩 4 KB 旧空壳，而容器内 `/workspace` 下积了 182 MB。

现在两处都挂（`/workspace` 是真实位置，`/vllm-workspace` 兼容 workdir 不同的镜像）。
另外把"命中"判据从"目录存在"改成**清单文件大小**（`static_kernel_cache/*.json` ≥512 B），
因为旧写法下空壳目录也能让检查通过。

### 4.3 ⚠️ 两个**尝试过并撤销**的改动（本轮最重要的教训）

#### (a) device_metadata 的"自愈护栏" —— 已删除，**不要恢复**

**动机**：`device_metadata.py` 的 `submit()` 置位 / `release()` 清位由
`model_runner_v1` 两处调用点配对，**没有 try/finally**。中间任何异常逃出，
标志就永久停在 True，之后每个请求都死在
`The previous device metadata submission has not been released`。
（这个真实故障形态记在 `lite-runs/DMQ-LEAK.md`。）

**做法**：整文件覆盖 `patches/files/model_runner_v1.py`（280 KB），加两道护栏 ——
`submit()` **之前**判 `submission_in_flight == True` 就强制 release，forward 之后再兜一次。

**实测结果：护栏本身把服务打挂了。** 64 并发扫描下，**正常请求也会命中**那个判据
（`submission_in_flight` 在正常流程中会短暂为 True），于是 device metadata 被提前释放，
device 侧契约被破坏：

```
AI CPU kernel execution failed ... kernelName=ScatterElements, errorCode=0x91
→ ERR00100 → HCCL watchdog thread terminated → 服务整体不可用
```

**关键证据**：把 `ENGRAM_DEVICE_INDEX=0`（完全不走 device-index 路径）**也照样触发**
⇒ 与 device-index 无关，就是护栏。同一次扫描 **57/64**，7 个请求失败。

**处置**：整块删除 —— 文件已从包里移除、Dockerfile 已清、serve 脚本的挂载已撤
（`serve_a2.sh` 里只留一条注释说明为什么不能恢复）。
撤销后同一套并发扫描 **64/64 全过**（7 档 × 2 rep，见 §9）。

**纪律**：整文件覆盖 vllm-ascend 核心文件（尤其 `model_runner_v1.py`）的风险远高于收益。
护栏想治的是"异常从 forward 逃出"的**罕见**场景，而它的误伤在正常运行下是**必然**
—— 宁可少一个护栏，不可多一个静默杀手。

#### (b) 初始化期探测"显存可读性" —— 已删除

**做法**：`probe_host_mapping_capability()` 里对 host-mapped 张量做 `int(t[0])`，
想直接证明"设备真能读到 host 内存"。

**实测结果**：worker 初始化阶段 **segfault**
（`aclrtMemcpyImpl` → `_local_scalar_dense` → `item`），整台机器起不来。

**处置**：能力探测**只做 `aclrtHostRegister`**（够用），端到端可读性交给**独立进程**的
`tools/probe_a2_hostmap.py`（崩了也不影响服务）。A3 上该探针返回 SUPPORTED。

### 4.4 起服前清 page cache

`DROPCACHE=1`（默认）：起服前 `echo 1 > /proc/sys/vm/drop_caches`。
实测 `MemFree 528517 MiB → 997802 MiB (+469 GiB)`。**注意它清不了 tmpfs**
（`/tmp`、`/dev/shm` 里的东西算 Shmem，不可回收）。

## 5. 精度与正确性

device-index 的价值必须建立在**逐位一致**上。已通过的测试：

| 测试 | 内容 |
|---|---|
| `engram_device_test.py --stage cpu` | 12 个语义场景 × host 的 `stock`/`fast` **两种**参考实现，逐位一致 |
| `engram_device_test.py --stage npu` | NPU 上重跑同一批（含同调用重复槽位），逐位一致 |
| `engram_device_test.py --stage graph` | 捕获 + replay + fresh-inputs 逐位一致 |
| `engram_integration_test.py` | 表查找 vs 融合 Triton 反量化 4096 行逐位一致；真实 layout 哈希 12 场景一致 |
| `engram_wiring_test.py` | 假 attn_metadata 驱动完整管线，5 个场景与 host 参考逐位一致；越界页号被拒 |
| `engram_multidevice_test.py` | 多设备 `device_id` 回归（单卡测试抓不到这类 bug） |

## 6. ★ DRAFT_GRAPH 的实测负面结果：**默认必须保持 0**

发布前按要求尝试把 `DRAFT_GRAPH`（DSpark draft 入图）改为默认 1，并把同源的
静默失效一并修掉，结果**实测是负收益**：

| 配置 | A（接受长度） | 单流 tok/s | ms/step |
|---|---:|---:|---:|
| `DRAFT_GRAPH=0` | **2.7 – 3.0** | **90 – 111** | 27 – 30 |
| `DRAFT_GRAPH=1` | **1.06 – 1.08** | **42.0** | 25.1 |

**所有开关都验证到位了**：容器内 `DSPARK_GRAPH_CAPTURE_METADATA=1`、
draft 版 `dspark_proposer.py` 已装（grep 命中 2 处）、起服命令行确实是
`speculative-config {"method":"dspark",...,"enforce_eager":false}`。
**但效果仍然是坏的** —— A ≈ 1.0 说明 draft 完全没产出，正是
`reports/draft-graph-negative-control.md` 记录的那种静默失效。

**最危险的地方**：ms/step 反而"更好看"（25.1 vs 29.5）。因为静默失效时每步只出
**1.08** 个 token 而不是 **2.85** 个 ⇒ **真实吞吐慢 2.2×**。
只看 ms/step 会得出完全相反的结论。

⇒ 默认保持 `DRAFT_GRAPH=0`。要实验必须用新加的**效果级** guard：

```bash
DRAFT_GRAPH=1 bash scripts/serve_a3.sh
bash tools/draft_graph_guard.sh     # 退出码 0=有效 / 1=静默失效 / 2=不确定
```

`draft_graph_guard.sh` 的判据以 **tok/s 为主、A 为辅**（tok/s 不会像 ms/step 那样被骗），
并显式解释"ms/step 变小是假象"。

> 教训：验证一个开关"装上了"（env 对、文件对、命令行对）**不等于**验证它"起作用了"。
> 必须查**效果**，而且要用不会被同一故障反向误导的指标。

### 6.1 补充：这是一个**未完成的重构**，不是一个可以修的 bug

按要求把 draft 入图设为默认后，实测发现它**两层都坏**：

| 配置 | 用哪个 `dspark_proposer.py` | 是否入图 | A（接受长度） | 单流 tok/s |
|---|---|---:|---:|---:|
| `DRAFT_GRAPH=0` | **stock**（官方镜像） | 否 | **2.85** | **94–111** |
| `DRAFT_GRAPH=1 SPEC_EAGER=1` | **draft 版** | **否** | **1.84** | **58** |
| `DRAFT_GRAPH=1` | **draft 版** | 是 | **1.05** | **42** |

**关键**：把图关掉（`SPEC_EAGER=1`）**仍然是坏的**（2.85→1.84）⇒ 根因不只在"入图"，
而在 `patches/files/draft/` 那 **~294 行**（403 → 664 行）从未验证的改动里；开图后再退化一次。

排查过程中**排除掉**的假设（都有实测/源码证据）：
- `DSPARK_GRAPH_CAPTURE_METADATA=1` 未设 → 已设，无效
- capture 期 metadata tasks 没跑（`_DSPARK_DEVICE_METADATA` 默认 0）→ 已设 `=1`，无效
- replay 不重建 metadata → 探针实测 replay **有**调用 `build_draft_attn_metadata`，且 `query_start_loc`/`max_query_len`/`decode_token_per_req` 与 capture **完全一致**
- 9 个 capture bucket 不全 → 实测 9/9 全捕获（含单请求用的 bucket 6）

⇒ 结论：这是**一个未完成的重构**，不是一处 bug。`tools/enable_draft_graph.sh` 里那句
"❌ **上卡验证未做**" 是准确的。**默认保持 0**，要实验必须用 `tools/draft_graph_guard.sh`
（效果级判据）确认 A ≥ 1.3 且 tok/s ≥ 80，否则不要采用。

## 7. `--async-scheduling`：本配置下**无收益且高并发崩溃**

`docs/STREAM_SPEED_PLAN.md` 记录过 async 的量级（async off 57.61 → on 34.39 ms/round，**−40.3%**），
且明确说 `ADMISSION_GATE=1` 与 async **可以共存**、no-async 只是"首轮验证推荐"。据此实测：

| 配置 | A | 单流 tok/s | ms/step |
|---|---:|---:|---:|
| `ASYNC=0`（默认） | 2.85 | 94 | **30.2** |
| `ASYNC=1` | 2.25 | 73.6 | **30.6** |

**没有收益**（30.6 vs 30.2），而且 **64 并发把服务打挂**：
`HCCL watchdog thread terminated` + `ERR02005 DIST internal error`（10/64 请求失败，
与服务此前那个 `ERR00100`+`ScatterElements` 的失败**签名不同**）。
服务端自己会警告：`[admission_gate] max_concurrent_batches=2 (async scheduling/PP)` ——
batch 开始重叠，而本包的 admission gate 是按单 batch 设计的。

⇒ 历史那个 −40% 出自 `ENGRAM=0` + 无投机 + TP8/DP1 + 7.8K 上下文的配置，**不可外推**到本包形态。
**默认保持 0**；开关保留（`ASYNC=1`）供后续在有 soak 保护的场景下复验。

> 顺带得到一个重要推论：**async 无收益 ⇒ host 工作已不在关键路径上**。
> 设备忙时约 27.96 ms / 真实步时 30.2 ms = **92.6% 饱和**，所以
> profiler 报的 "Device Free 10.269 ms/step (26.9%)" 里**大部分是 profiler 自己拉长的**
> —— 与用户给的判断一致（torch profiler 的负载主要在 host，会放大 host bound 的段）。
> 要达成 24 ms/step 必须**减少设备工作量**，而不是继续压 host。

## 8. 已知边界

1. **prefill 仍走整表 gather（16.8 ms）**。分段方案已验证（5.36 ms，2.44×）但未接线；
   按当前口径 prefill 省 11 ms / 1.14 s ≈ 1%，优先级低。
2. **`GatherV3` 等算子是共享名**（模型自己的 LightningIndexer / MoE 路由也用），
   做归因时必须用 A/B 计数差，只按名字匹配会系统性高估。
3. `gather_dequantize_engram_int8`（融合 Triton 核）**拒绝** host-mapped 指针
   （`aclrtPointerGetAttributes` 的 location 既非 DEVICE 也非 HOST_NUMA），
   设备路径走 aclnn。decode 规模下这块本就只有 ~0.03 ms。

## 9. ★ 并发吞吐重采（护栏撤销后的回归验证 + README §3.2 数据源）

> 口径：**1024 token prompt（服务端 `/tokenize` 精确校准，64/64 命中偏差 0）× 256 token 输出**、
> 语料 `data/dihuo.txt` 每请求**互不重叠的不同切片** + 轮换 5 个问题 ⇒ 无 prefix cache 复用；
> 每档跑完同一批 **64 条**请求、2 rep 取中位数。
> 配置 = 本包默认（`MAX_SEQS=64 PREFIX=1 GPU_UTIL=0.92 STATIC_KERNEL=1`）+
> `ENGRAM_DEVICE_INDEX=auto`、`DRAFT_GRAPH=0`、`ASYNC=0`、无 PGO 产物（A3 自动降级）。

| 并发 | 单流 tok/s | 总吞吐 tok/s | 加速比 | 单流效率 | A（接受长度） | TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **90.3** | 87.1 | 1.00× | 100.0% | 2.81 | 0.20 s |
| 2 | **89.2** | 151.5 | 1.74× | 98.8% | 2.82 | 0.36 s |
| 4 | **80.0** | 237.7 | 2.73× | 88.7% | 2.80 | 0.55 s |
| 8 | **59.3** | 324.4 | 3.72× | 65.6% | 2.88 | 0.94 s |
| 16 | **42.5** | 432.6 | 4.97× | 47.1% | 2.78 | 1.83 s |
| 32 | **31.3** | 583.9 | 6.70× | 34.7% | 2.86 | 3.82 s |
| 64 | **20.2** | **719.5** | 8.26× | 22.3% | 2.82 | 7.16 s |

**7 档 × 2 rep 全部 `ok=64/64`，服务全程存活** —— 这同时是 §4.3(a) 护栏撤销的回归验证。

与 v5 表格（96.3 / 595.8，2026-09-17 同一方法）相比：**总吞吐 +20.8%**（595.8 → 719.5）、
**TTFT −26%**（0.27 → 0.20 s）、单流 −6.2%。
差异未逐项归因（同机共租负载与 KV 容量都会影响），两表口径一致、都可复现。

原始数据：`results/bench/conc_dihuo_v8.json`；复现命令见 README §3.2。

# ★ v7（2026-09-18）—— 长上下文精度修复：`BAT_TOKENS` 2048 → 8192

> **这是本包第一次修"正确性"而不是"性能"或"工程"。**
> 改动只有一行默认值（`scripts/serve_a2.sh`），但影响很大。

## 0. 症状

长上下文（≳60K token）的 agent 场景里，模型会**输出复读/幻觉、不再调用工具**，
最终被解析器整段丢弃成"空回复"。此前被当作"偶发、不可复现"。

## 1. 根因

**chunked prefill 的 chunk 数决定偏离率。**

chunked prefill 把长 prompt 切成 `ceil(prompt / BAT_TOKENS)` 段依次前向，
每段都有一次独立的"偏离"机会，误差沿后续 chunk 累积。实测约 **2%/chunk**。

因此 `BAT_TOKENS=2048` 时：

| prompt_tokens | chunk 数 | 正确率 |
|---:|---:|---:|
| 20,318 | 10 | 80% |
| 40,163 | 20 | 70% |
| 60,012 | 30 | 50% |
| 79,855 | 40 | 30% |

**平滑下滑**——这解释了为什么此前找不到"阈值"：它本来就没有阈值。

### 1.1 怎么证明是"我们的栈"而不是模型

同一批 prompt（逐字节相同）发给官方 API：**12/12 全过（含 252K token）**；
我们的服务在同长度上 ~0%。⇒ 模型有能力，是我们的栈弄坏的。

## 2. 修复

```diff
-BAT_TOKENS=${BAT_TOKENS:-2048}
+BAT_TOKENS=${BAT_TOKENS:-8192}
```

### 2.1 效果

| prompt_tokens | `BAT=2048` | `BAT=8192` |
|---:|---:|---:|
| 10,394 | 10/10 | **10/10** |
| 20,318 | 8/10 | **10/10** |
| 40,163 | 7/10 | **10/10** |
| 60,012 | 5/10 | **10/10** |
| 79,855 | 3/10 | **10/10** |
| 149,986 | ~0% | **6/6** |
| 259,985 | ~0% | **6/6** |

同一 prompt 重复 10 次 → 10/10，输出 token 数逐次一致。

### 2.2 代价

| 项 | `BAT=2048` | `BAT=8192` |
|---|---:|---:|
| KV cache（8×910C, util=0.94） | 4,145,957 tok | 3,088,738 tok |
| host 侧每 chunk 耗时 | ~34 ms / 2048 tok | ~191 ms / 8064 tok |

KV 容量下降约 25%。更看重 KV 容量且上下文主要在 <20K 的场景可显式 `BAT_TOKENS=2048`。

## 3. 新增

| 文件 | 说明 |
|---|---|
| `tests/agent_trace/longctx_retrieval.py` | **长上下文检索探针** —— 把唯一事实埋在长文档中段、只问一个答案唯一的问题。60K token 就能测出退化（比"工具调用测试"灵敏得多）。报 Wilson 95% CI、支持多 nonce。 |
| `tests/agent_trace/accuracy_gate.py` | agent 形态的工具调用精度门（5 个长度档、截断单列、失败签名分类）。 |
| `reports/longctx-accuracy-fix.md` | 完整分析：曲线、消融、机制解释、方法论教训。 |
| `reports/probe/` | 稀疏状态插针（事后取证工具）+ 设计文档，`PROBE=1` 启用。 |

## 4. 同时修掉的工程问题

| # | 问题 | 修复 |
|---|---|---|
| 1 | `inner.sh` 把 `FUSED_MC2` / `MC2` / `MC2_HIER` / `REDUCE_SAMPLE` / `DSA_OVERLAP` **硬编码**，外部 env 传不进去 | 全部参数化（默认值与原来一致，行为不变） |
| 2 | 起服依赖镜像里烘焙的 `/opt/dsv41/scripts/serve_v2.sh`，换基础镜像就起不来 | 改为只读挂载本包的 `scripts/` |
| 3 | 排查用的 dev mode / 请求日志 / 插针无法从外部开关 | 新增 `VLLM_SERVER_DEV_MODE` / `LOG_REQUESTS` / `PROBE`，**默认全关** |

## 5. 方法论教训（写进仓库，避免重犯）

定位过程中我制造了 **5 个假阳性**（工具数量、插针、投机解码、"14 个 token 决定成败"、
Engram/QLI 是主因），全部源于**跨会话比较**——服务会随时间自发退化（旧会话 ~12%、
新鲜会话 ~56%），拿旧基线比新鲜消融必然得出假阳性。

**三条纪律**：

1. 任何消融必须配**同等新鲜度**的基线。
2. 同一剂量点用**多个不同样本**（同一长度下"内容"决定成败，单样本毫无代表性）。
3. 判据必须看 `finish_reason` —— `finish=length` 的截断样本既非成功也非失败，
   混进失败率会得出错误结论。

## 6. 完整验收（2026-09-18）

| 验收项 | 要求 | 实测 | 判定 |
|---|---|---|---|
| 前后对比（N≥10） | 修复前失败率显著、修复后 0 | 43/50 → **62/62** | ✅ |
| 8K / 32K / 128K / 256K | 全覆盖 | 8.4K / 32.2K / 130.5K / 260.0K 全 10/10（256K 为 6/6） | ✅ |
| 真实 agent 轨迹 | 覆盖 | 两条真实会话轨迹各 **10/10** | ✅ |
| Vision | 23/23 | **23/23** | ✅ |
| GSM8K-200 | ≈198/200 | **199/200** | ✅ |
| `static_kernel` 降级 | 0 | **0** | ✅ |
| Engram-int8 常驻 | 必须 | `engram_storage=int8` + host-resident | ✅ |
| KV > 3Mi | 交付约束 | **3,088,303（低 1.8%）** | ⚠️ 见下 |

### 6.1 新增的显存约束（`[MEM-GUARD]`）

`BAT=8192` 让 peak activation 从 0.79 → 3.21 GiB。与 `MAX_SEQS=64`
（capture 桶到 384）叠加时，`GPU_UTIL=0.94` 会在 **ACL graph 重放时 OOM**：

```
torch.OutOfMemoryError: NPUGraph.cpp:281
Resource_Error_Insufficient_Device_Memory(EL0019)
```

`scripts/serve_a2.sh` 已加提示（危险组合时打印建议，不擅自改配置）：

| 组合 | 结果 |
|---|---|
| `MAX_SEQS=32` + `BAT=8192` + `0.94`（发布默认） | ✅ |
| `MAX_SEQS=64` + `BAT=8192` + `0.90` | ✅（KV 降到 2.56M） |
| `MAX_SEQS=64` + `BAT=8192` + `0.94` | ❌ OOM |

### 6.2 已知代价

KV cache 4,145,957 → **3,088,303** tokens，低于 3Mi 交付约束约 1.8%。
若必须同时满足 KV>3Mi，可评估 `BAT=4096`（尚未验证是否足够）。

完整记录见 [`reports/longctx-verification.md`](reports/longctx-verification.md)。

---

# ★ v6（2026-09-17）—— 交付工程修复 + 自检体系

> **v6 不改任何性能配置**：`patches/`、`Dockerfile`、`optim/` 与 v5 **逐字节相同**
> （已用 `diff -rq` 验证）。所以 **v5 的镜像可以直接 retag 复用，不必重新构建**：
> `docker tag dsv41-a2:v5 dsv41-a2:v6`
>
> v6 修的是"**能不能跑通、能不能复现**"。性能差距（A2 75.7 vs A3 28.7 ms/step）
> 是另一条线，v6 不改善它 —— 见 `EXPECTED_PERF.md` §A2GAP。

## 0. 为什么要有 v6（v5 的教训）

v5 在 A2 上暴露了 **10 个 bug**，其中 **8 个是我们自己的脚本/打包错误**。
共同点：**都不会在开发机上暴露**（因为开发机上有那些文件、有那些变量），
而**每个都要跑到最后一步才知道**，起服一次 5–25 分钟 ⇒ 代价被放大十几倍。

因此 v6 的核心不是加功能，而是**把检查前移 + 证明检查有效**：

| 新机制 | 解决什么 |
|---|---|
| `tools/preflight_a2.sh`（**30 秒 / 9 组 / 不起容器**） | "白等 25 分钟才发现" |
| `tools/negative_control.sh`（**8 个负控**） | "自检全过但包是坏的"（v5 真实发生） |
| `build_scripts/00_ensure_pgo.sh`（指纹缓存） | PGO 只编译一次；换机器自动重编 |

## 1. 修的 10 个 bug

### 1.1 缺文件（两个，直接让测试全废）

| # | 症状 | 根因 | 修法 |
|---|---|---|---|
| **1** | `can't open file '.../tests/p15_stream_curve_filefiller.py'` ⇒ **8K/32K/128K 性能测试全废** | 打包时漏收该文件（它在 A3-node1 的 `logs/perf/` 下） | 收进 `tests/`；md5 `ba75b25e3b0a2eb8dd1436627d4c2126` |
| **2** | `[t_vision] {'cases': None, ..., 'verdict': 'FAIL'}` ⇒ **视觉必 FAIL** | `t_vision.py` 调 `HERE/vision_accuracy_check.py`，但文件在 `tools/` 不在 `tests/` | 复制到 `tests/`；md5 `879dd13d1547d572efd335134a468c3a` |

### 1.2 参数没接上（三个，静默失效）

| # | 症状 | 根因 | 修法 |
|---|---|---|---|
| **3** | 传 `CPUS=-1` 无效，仍绑 24 核 | 变量名是 **`CPUSET`**（与 `MEMS` 不对称），`CPUS` 无人读 | v6 接受 `CPUS` 作为别名 + 冲突时告警 |
| **4** | 传 `CPUSET=...` 无效 | **`run_test.sh` 根本不转发** `CPUSET/MEMS/CPU_BIND` | v6 转发（还补了漏掉的 `MOE_NF`/`CACHE`/`SKCACHE_GC`） |
| **5** | 想换解释器换不了 | `PYHOST=$(choose_py)` **无条件覆盖** | v6 改为 `PYHOST=${PYHOST:-$(choose_py)}` |

### 1.3 失败不复原（一个，代价最大）

| # | 症状 | 根因 | 修法 |
|---|---|---|---|
| **6** | 起服失败后**容器仍活着占 ~313 GB**，导致第二次起服叠加失败 | 容器入口是 `bash -lc "sleep infinity"`，`die()` 没有清理 | v6 在 `die()` 里 `docker rm -f`（日志已落盘，不丢证据） |

### 1.4 环境假设（三个）

| # | 症状 | 根因 | 修法 |
|---|---|---|---|
| **7** | `pin_memory` 报 `207001`（`aclrtMallocHostWithCfg`）而 667 GiB 空闲 | 需要更新 driver **且** 解除 memlock | v6 默认 `--ulimit memlock=-1`（A3 上无副作用） |
| **8** | PGO 可能**静默降级**为 0 | `TARGET_PATH.txt` 是 `build_image.sh` 生成的、不在包里；缺失时只打一行 WARNING | v6 起服时**自动探测并落盘**；preflight 显式报告 |
| **9** | `skcache/ts*_outputs` 无限累积（A3 上 **1491 个目录 / 847 MB**，从不清理） | 脚本只创建不清 | v6 起服前自动 GC，**且显式保留 `static_kernel_cache/`** |
| **10** | （潜伏）`run_test.sh` 缺 `set -e` 导致某些失败被吞 | —— | 保持现状但**显式文档化**：各阶段独立、GSM8K 失败不中止 |

## 2. 新增：`tools/preflight_a2.sh`（30 秒，不起容器）

9 组检查，把已知的 11 类 A2 环境差异 + 包完整性一次问完：

```
1/9 包内文件完整性（逐个断言 t_quote/t_vision/t_gsm8k/acc_eval/p15/vision_accuracy/...）
2/9 Dockerfile 续行链 + 镜像 tag 三处一致（v3/v4/v5 都在这里栽过）
3/9 模型目录：软链链健康 + 跳数分布 + 必需文件 + engram/vision/mtpq 分片
4/9 官方目录：inference/examples/images + encoding
5/9 宿主依赖：PYHOST 选谁 + datasets 版本（**必须 5.0.1**）+ HF 缓存
6/9 CPU/线程/绑核：逻辑核 vs 物理核、OMP_NUM_THREADS、memlock、**NPU↔NUMA 拓扑**
7/9 编译缓存：skcache 键（CANN-<ver>_<SoC>）、ts* 堆积、PGO md5 + TARGET_PATH
8/9 主机资源：MemAvailable（Engram 需 206 GiB 常驻）、磁盘、**残留容器**
9/9 结论：FATAL/WARN 计数
```

**退出码**：0 = 无 FATAL；1 = 有 FATAL。设计成"**先跑它，再决定要不要起服**"。

## 3. 新增：`tools/negative_control.sh`（8 个负控，14 个断言）

**这是 v5 最大的方法论缺口**：v5 有 `selfcheck_pkg.sh`，但它第一次跑就"全过"，
而包其实是坏的 —— 因为**自检只证明了"我检查的东西是好的"，没证明"检查本身有效"**。

v6 对**每个已知 bug 故意造一个坏输入**，断言检查必须报错：

| # | 负控 | 断言 |
|---|---|---|
| NC1 | 删 `tests/p15_...py` | preflight 必须报 FATAL |
| NC2 | 删 `tests/vision_accuracy_check.py` | preflight 必须报 FATAL |
| NC3 | Dockerfile 行内 `#` + 漏 `\` | `check_dockerfile.py` 必须 FAIL，且 `docker build` 也必须 FAIL |
| NC4 | build/serve/run_test 三处 tag 不一致 | preflight 必须报 FATAL |
| NC5 | 软链断链 | `model_mount_args.sh` 必须 rc≠0；**正控**：健康 5 层链要解析出 5 个目录 |
| NC6 | `-v` 与路径粘成单参数 | 组参数必须偶数且交替 `-v`/路径；`docker run` 必须拒绝粘在一起的写法 |
| NC7 | serve 退回单层 `-v $MODEL` | 静态检查必须抓到 |
| NC8 | 假 skcache 含 `ts*` + `static_kernel_cache/` | GC 后**只删 ts\***、**保住缓存** |

**全部在 `mktemp -d` 的临时副本里做，不动真实包。**

## 4. 新增：PGO 编译流程进包（`build_scripts/` + 指纹缓存）

v5 只有**预编译产物**（2 个文件），没有编译能力，且那份 `.so` 是
**在 Ubuntu 容器里编的、跑在 openEuler 上**。

| 文件 | 作用 |
|---|---|
| `build_scripts/00_ensure_pgo.sh` | **新入口**：算指纹 → 一致则**秒退**；否则起一次性容器编译 |
| `build_scripts/01_setup_build_env.sh` | **改为 distro-aware**（Debian→apt / RPM→dnf）；这是 v5 搬到 A2 会直接失败的地方 |
| `02..06` | 沿用（取源码 / configure / make / install / package） |

**指纹** = `cpu_part + gcc + glibc + 源码 sha256 + mtune 开关`：

```
一致  -> 0 秒跳过（"只编译一次"）
不同  -> 自动重编（换机器/换镜像/换 gcc/换源码都不会用错产物）
```

**编译工作区在容器外**：`optim/pgo/build/{src,out,logs}` ⇒ 第二次可增量、产物持久。

### ⚠️ 必须说清的两个限制（避免误期待）

1. **重编本身不会带来明显性能收益**。现有构建参数里**没有 `-march`/`-mtune`**
   （已实证：`grep -oE '\-m(arch|tune|cpu)=' make.log` 为空）⇒ 代码生成是通用
   aarch64，**在哪台机器上编都差不多**。重编的价值是：**glibc/发行版精确匹配** +
   **未来可重建**。
2. **想针对本机核优化只有一条路**：`PGO_MTUNE=1`（加 `-mtune=native`，**安全，
   不生成新指令**）。预期 **0~3%**，且**必须 A/B 实测**。默认**关**。
   `-march=native` 不提供（会 SIGILL）。

### 关于 `PROFILE_TASK`：**保持 CPython 默认**

用户确认它"具有一定代表性"，且有**实测证据支持**：
训练用 CPython 测试套件，而在 `tiny_call` / `dict_loop` / `list_append`
这些**与测试套件毫无关系**的模式上仍拿到 **−16~23%**
（`reports/cpython-pgo-verified.md`）。原因清楚：**PGO 优化的是解释器本身**
（字节码分派、对象分配、dict/type 查表、函数调用），任何 Python 程序都要穿过。
⇒ **v6 不改 PROFILE_TASK。**

## 5. 其它

* 镜像 tag：`dsv41-a2:v5` → **`dsv41-a2:v6`**（三处一致；preflight 会校验）
* preflight 会提示：若 `dsv41-a2:v6` 不在但 `dsv41-a2:v5` 在，**直接 retag 即可**（内容相同）

---


> v3 = `a2_pkg_v3/`（2026-09-16 18:14–18:40 打包，153 文件；**已交付给你，本包保留对照**）
> v4 = 本包（2026-09-16 21:00–21:5x，**只做增量**：新增 11 个文件 + 改 6 个文件）
>
> 每条都标了**来源**（`reports/*.md` 逐字复制在本包；A3-node2 的原文路径写在 `CORRECTNESS_STATUS.md` §10）。

---

## 0. 一句话

v3 是"**8 项已验证优化 + 单流口径**"（128K 30.2–31.6 ms）。
v4 **不改任何已验证优化**，改的是**结论的诚实度与覆盖面**：

1. **补上 v3 完全没测的形态**：生产口径（`MAX_SEQS=32 + PREFIX=1`）与
   **多 batch / 多轮对话**（历史"并发"只是 decode 并发，prefill 被 `--serialize-prefill 1` 刻意串行化）；
2. **更正 v3 的 110 tok/s 叙述**：163 发全量重算 ⇒ **真正可交付的 ≥110 只有 1 发**，
   `A` 是三吸引子抽签（steep 15.3%）且**脏会话 A 更高** ⇒ **A 不能当绩效指标**；
3. **把 4 个负结果写进"别踩坑"**，并新增 `CORRECTNESS_STATUS.md`（数值确定性 / 精度门的独立结论）。

---

## 1. v4 **新增**的文件（14 个）

| # | 文件 | 内容 | 来源 |
|---|---|---|---|
| 1 | **`CORRECTNESS_STATUS.md`** | 5 条交付级结论：无 ctx 阈值（抽签）/ `SPEC=0` clean 高 5 倍 / clean 与 coherent 独立 / "减少 forward 数"被算术否证 / 精度门 + HCCL 代价 | A3-node2 `reports/correctness-line.md`（426 行）+ `/tmp/{freq3,spec0,spec0_rep2,cvq2}.log` |
| 2 | **`tests/multibatch/multibatch_gate.py`** | 三块：**[A]多轮对话**（埋针逐字召回）、**[B]并发逐 item 比对**（16 道算术题 conc=1 vs conc=8）、**[C]长短交错**（1×128K + 6 短请求）| A3-node1 `exp_tools/multibatch_gate.py`，**硬编码 `/home/user/...` 全部参数化**（`--base/--out/--corpus`），并补 `verdict` 机器可读判据与退出码 |
| 3 | **`tests/multibatch/multibatch_session.sh`** | 生产口径起服 + 跑三块（单臂） | A3-node1 `exp_tools/multibatch_session.sh` 移植：改调 `scripts/serve_a2.sh`、`STOP_FIRST=0`（**默认不杀别人的容器**） |
| 4 | **`tests/multibatch/run_prod_both.sh`** | 两臂对照（`PREFIX=1` vs `PREFIX=0`）+ 汇总表 | 新写（主 Agent 在 A3-node1 就是跑这两臂） |
| 5 | **`tests/multibatch/verify_serve_flags.sh`** | 12 组合启动器烟测（`DRY_RUN=1`，**不占卡**） | A3-node1 `/tmp/verify_serve_flags.sh` 移植 + 换成 A2 的组合（含 `MAX_SEQS=32 PREFIX=1`） |
| 6 | **`tools/fisher_recheck.py`** | Fisher 精确检验（2×2），带 **4 个自检用例**（含"相同表必须 p=1"这个坑） | 新写（因为我们的 `fisher()` 有 bug，见 §4） |
| 7 | `patches/files/token_dispatcher_moennf.py` | `MOE_NONFINITE` 的 patch 文件（**负结果，默认不挂**） | A3-node1 `probe_moe_nf/token_dispatcher.py`（837 行，逐字节） |
| 8–11 | `reports/a-basin-and-acceptance-shape.md`（186 行）、`reports/session-attractor-and-clean-rate.md`（132 行）、`reports/draft-graph-negative-control.md`（47 行）、`reports/multibatch-and-mixed-load.md`（88 行） | v4 的**四份权威报告**，**逐字复制**（md5 与本机一致） | A3-node1 `reports/*.md` |
| — | `reports/` 从 51 → **55 份** | | |
| 12 | **`tests/acc_eval.py`** | GSM8K / C-Eval 评测器（**v3 的 `tests/t_gsm8k.py` 引用了它却漏打进包 ⇒ `MODE=full` 在 A2 上会直接失败**）。v4 补入并把硬编码的 `/home/user/models/.../encoding` 参数化（`--enc-dir` / `ENC_DIR` / 自动找 `~/models/...`），新增 `--base` 别名；**保留 `--serialize-prefill` 默认 1**（与历史 GSM8K 口径一致） | A3-node1 `scripts/acc_eval_p4s.py`（244 行）+ v4 参数化 |
| 13 | **`tools/interleave_ab.py`** | 同会话交错 A/B（**clean-rate 判据 `pos0≥0.8`**）——v4 的 4 个负结果就是用这个工具判的；p 值走包内 `fisher_recheck.py`（不再自带一份实现） | A3-node1 `exp_tools/interleave_ab.py`（**已含 FISHER-FIX**，md5 `3e8b9c67…`）；去掉对 `logs/perf/p15_*.py` 的依赖（把那 4 个小函数内联）并把容器名/路径参数化 |
| 14 | **`tools/steep_summary.py`** | 从 p42 jsonl 汇总 **steep / flat / shallow**（= `EXPECTED_PERF.md` §2 的工具） | A3-node1 `exp_tools/steep_summary.py`；默认 glob 改为本包 `results/` 与 `logs_meta/samples/`（去掉硬编码） |

## 2. v4 **修改**的文件（7 个）

| # | 文件 | 改了什么 | 为什么 |
|---|---|---|---|
| 1 | **`scripts/serve_a2.sh`** | ① 新增 **`PREFIX`**（默认 **0**，与 v3 逐字节一致）；② `CAPTURE_SIZES` 改为**按 `MAX_SEQS × (1+SP_TOKENS)` 自动扩展**（`[MULTI-SEQ-CAPTURE]`，与 A3-node1 `serve_a21.sh` 同构）；③ 新增 `MOE_NF`（默认 0，负结果臂）与 `HCCL_DET`（默认空，仅诊断）；④ 新增 **`DRY_RUN=1`**（解析后打印并退出，不碰 docker）；⑤ `serve_cmd.txt` 增打口径与 HCCL 状态 | `MAX_SEQS=32` 时旧逻辑只覆盖到 32 token ⇒ **直接起不来**；`PREFIX=1` 才是 A2 生产形态 |
| 2 | **`scripts/run_test.sh`** | 新增 **`MODE=prod`**（`MAX_SEQS=32 PREFIX=1` + 多 batch 三块，跳过 quote 性能）+ `PREFIX` 透传 + `env.txt` 记录 `max_seqs/prefix/mode` | 生产口径与单流口径**不可混比**，必须显式分开 |
| 3 | **`EXPECTED_PERF.md`** | 重写：新增 §2（163 发形态分类）、§3（clean-rate，为什么 A 不能当指标）、**§7（多 batch 生产口径：§7.3 实测三块全过 + §7.3.1 局限 6 条 + §7.3.2 为什么这是空白 + §7.3.3 如何自己跑）**、§8（与旧报告的差异说明）；更正 §2.3 的 110 tok/s 叙述；峰值统一为 **110.5**（`A×1000/ms` 口径，v3 的 110.94 是 jsonl 另一算法） | 用户要求"把这些新事实写进 EXPECTED_PERF" + 21:12 拿到生产臂实测 |
| 4 | **`README.md`** | ① 新增 §0.1 两条更正；② §1.1 = **4 个负结果表**；③ §1.2 = DSpark 状态更新（离线修复完成 + 负控已确认，正控待验）；④ 三命令骨架（加 `MODE=prod`）；⑤ 新增 §4.1 v4 开关表 | 用户硬性要求 |
| 5 | **`tests/make_report.sh`** | 新增 **§1b 多 batch 表**（读 `summary.json`） | 让 `MODE=prod` 的结论进一页纸 |
| 6 | **`tests/t_gsm8k.py`** | 新增口径警告（`--serialize-prefill 1` ⇒ only decode concurrency）+ `--enc-dir` 透传 + **缺 encoding 目录时明确跳过**（原来会报 `ModuleNotFoundError`） | v3 的 `MODE=full` 在 A2 上会因缺 `acc_eval.py` 直接失败 |
| 7 | `Dockerfile` / `scripts/build_image.sh` | 镜像 tag `dsv41-a2:v3` → **`dsv41-a2:v5`**（其余逐字节不变） | 便于与 v3 镜像并存对照 |

## 3. v4 **删掉了什么说法**（**重要，别再用旧话术**）

| v3 的说法 | v4 的更正 | 依据 |
|---|---|---|
| "A 的方差是 2.9× ⇒ 跑 8 发看中位/区间" | **A 是三吸引子抽签**（steep 15.3% / flat 7.4% / shallow 77.3%，163 发），**不是单峰 + 噪声** | `reports/a-basin-and-acceptance-shape.md` §6 |
| "A 一旦进优模式，ms 稍差也能冲过 110 tok/s ⇒ A 决定一切" | **≥110 真正可交付的只有 1 发**（另 7 发来自不采纳的 `MOE_ZERO` 会话）；**A 不能当绩效指标**（脏会话 A 反而高） | 同上 §4/§6 + `reports/session-attractor-and-clean-rate.md` §2 |
| "`MOE_ZERO` 未在本机完成端到端验证 ⇒ 待验证" | **已测完，结论是"不采纳"**（换吸引子 + 非数值等价） | `reports/session-attractor-and-clean-rate.md` §1.1 |
| "`DRAFT_GRAPH` 上卡验证未做" | **崩溃已修 + 负控已确认**（缺 metadata ⇒ A 恒 1.0，8/8 发；ms 仍 30.2 ⇒ **单看时延发现不了**）；**正控待验** | `reports/draft-graph-negative-control.md` |
| "并发测试已覆盖" | **只覆盖 decode 并发**（`--conc 4 --serialize-prefill 1` = prefill 串行）；**多轮对话从未测过** | `acc_eval_p4s.py:173-174` 的 help 原文 + `EXPECTED_PERF.md` §7.1 |
| "Vision 23/23、GSM8K 197–199" | 保持，但补齐：**GSM8K-200 三次 = 198/199/197**；`HCCL_DETERMINISTIC=true` **91/100** | `CORRECTNESS_STATUS.md` §6 |

## 4. ⚠️ 工具错误声明（**v4 新增的纪律条目**）

| 工具 | 错误 | 处置 |
|---|---|---|
| `exp_tools/interleave_ab.py` 的 `fisher()` | 两行计数**完全相同**时打印 `p = 0.0000`（相同表必为 **1.0**）。根因：列和固定但未遍历所有 x；双侧判据在 obs 为最大概率时退化 0/0 | **已修正**并用教科书标准值重新校验（见下表）；**v4 的 4 条负结果一律使用修正后的 p 值**（`reports/session-attractor-and-clean-rate.md` §6.4）；本包附 `tools/fisher_recheck.py`（含 4 个自检用例） |
| `spread` 判据（第一版） | 只统计"最常见 token"的 logprob 极差 ⇒ **假性干净**（6 发 5 种 top-1 时极差自然为 0） | 已换成 `uniq_top1==1 AND n_distinct_lp==1`（两个条件互相独立，每行都要打） |

> **共同教训（写进测量纪律）**：**统计/判据工具上线前必须用已知答案自检**（至少 3 个教科书用例）。
> 本项目已栽两次 —— 都是"在看起来最该拒绝原假设的地方给出假显著/假干净"。

### 4.1 Fisher p 值的**修正表**（**必须用这一列**）

> **v4 修正了 v3 期间使用的 Fisher 实现 bug（同行计数时返回 0.0）；所有 p 值已用教科书标准值重新校验。**

| 对照 | 2×2 | **修正后 p** | 旧值（**作废**） |
|---|---|---|---|
| `LOCAL_OWNER=fast` vs `on` | `[[5,7],[4,8]]` | **1.0000** | 0.6843 |
| `MOE_ZERO=0` vs `1` | `[[2,10],[2,10]]` | **1.0000** | 0.0033 |
| `HCCL_DET=true` 下 fast vs on | `[[5,7],[3,9]]` | **0.6668** | 0.6843 |
| `MOE_NONFINITE=0` vs `1`（N=24） | `[[2,22],[2,22]]` | **1.0000** | 0.0000 |

**4 条结论（全部"无差异"）不变，但 p 值必须换成本表。**
唯一仍然显著的是正确性线**独立实现**的 `SPEC=0` vs `SPEC=1`：
`[[5,5],[2,18]]` → 单侧 **0.0256**（`correctness-line.md:371`，**同一会话**内）；
跨会话复现（**全新容器**）`[[9,7],[2,18]]` → 双侧 **0.0042**（单侧约 0.0026）。
三次 `SPEC=0` 测量 **0.50 / 0.50 / 0.5625** 一致 ⇒ 结论稳健。见 `CORRECTNESS_STATUS.md` §3.1.1。

## 5. 与 v3 **完全相同**的部分（**一行都没改**）

* 11 个整文件补丁 + 2 个 sidecar + `admission_gate.patch`（`patches/`，md5 见 `patches/MD5SUMS`）；
* 8 项已验证优化的**默认值**（`MOE_AG=1 / SP_TOKENS=5 / O_PROJ_2D=1 / MOE_MASK=1 /
  ROPE_IDXSEL=1 / ENGRAM_JIT=1 / LOCAL_OWNER=fast / QLI_NOCAND=1 / PYTHON_PGO=1 /
  GATE_CHUNK=0 / VLLM_ADMISSION_GATE=1`）；
* **单流口径的 `CAPTURE_SIZES` 输出逐字节不变**（`1,2,3,4,6,8,12,16,20,24,32`，已在包内用
  `DRY_RUN=1 MAX_SEQS=1` 实测）；
* 容器起服参数（`--net=host --shm-size=512g --privileged`、`/dev/davinci0..7`、宿主透传、
  `LD_PRELOAD=jemalloc`、`HCCL_BUFFSIZE=1024`、`TASK_QUEUE_ENABLE=1`、`HCCL_OP_EXPANSION_MODE=AIV`）；
* CPU/NUMA 自动绑定、视觉 23 例、GSM8K-200、容量判据、`results/<run_id>/REPORT.md` 结构；
* 量化链 `quant/`（5 级装配 + 结构/容差验收）与 `optim/pgo/` 产物；
* `data/`（红楼梦语料 + 4 个 suffix）与 `logs_meta/`（LOG_INDEX + 19 个 jsonl 样本）。
# v5（2026-09-16）—— **软链构造的模型目录：起服必失败的修复**

# v5.1（2026-09-16 21:50）—— **A2 实测反馈的两个致命 bug**

> A2 真机跑 `run_test.sh` 时暴露。两条都会让流程**完全走不下去**，且报错极具误导性。

## Bug 1（起服阻断）：`mapfile` 把 `-v` 和路径塞进了同一个参数

**现象**（A2 实测原文）：

```
docker: Error response from daemon: create  /home/.../optional:
" /home/.../optional" includes invalid characters for a local volume name,
only "[a-zA-Z0-9][a-zA-Z0-9_.-]" are allowed.
```

**注意错误信息里路径前面的那个空格** —— 它就是指纹。

**根因**：`tools/model_mount_args.sh` 原来每行输出 `-v /path:/path:ro`，
而 `serve_a2.sh` 用 `mapfile` 读入 —— **每行只成为一个数组元素**。
于是展开给 docker 的是**单个参数** `"-v /path:/path:ro"`，
Go 的 pflag 会把 `-v` 后面的**空格也算进值里** ⇒ 得到的路径是 `" /path"`（带前导空格）
⇒ docker 认为那不是绝对路径，转而按"卷名"解析 ⇒ 报 invalid characters。

**修复**：`model_mount_args.sh` 改为只输出**裸路径**（每行一个），
由 `serve_a2.sh` 显式拼成 `-v` 与 `路径:路径:ro` **两个**数组元素。

**验证**：真值 5 层链条 → `dirs=5, argv=10`，`docker run` 内 3 个文件全部可读；
旧写法在同样输入下必然失败。

## Bug 2（构建阻断）：Dockerfile 续行链被行内注释截断

**现象**：`docker build` 直接报
```
dockerfile parse error on line 4: unknown instruction: local
```

**根因**（两处，都在 `RUN` 的续行链里）：

```dockerfile
RUN set -euo pipefail; \
    inst() { # src_in_tmp  target_rel      ← ① 行内注释 + ② 这一行没有 `\`
      local tgt="${ASCEND_PKG}/$2"; \
```

1. **中间行漏了结尾 `\`** ⇒ 链在此**提前结束**，后面的 `local` / `test` / `cp`
   被当作 Dockerfile 指令解析 ⇒ `unknown instruction: local`；
2. **行内 `#` 注释**：Docker 先把续行拼成**一整行**再交给 shell，
   行内 `#` 会把**它后面的一切**（包括还没执行的命令）全部注释掉。
   —— 而如果把 `\` 写在注释**后面**，那个 `\` 本身也在注释里，等于没写。
   （用户原话：「不要在行末加注释，否则 `\` 失效」）

**修复**：`inst() { \` / `newf() { \`（去掉行内注释、补上 `\`），
把解释性文字整体移到 `RUN` **外面**的注释块，并在那里写下"续行铁律"。

**验证**（最小复现 + 正负控，本机实跑）：

| 版本 | 结果 |
|---|---|
| 旧写法（行内注释 + 缺 `\`） | `docker build` → **`dockerfile parse error on line 4: unknown instruction: local`** |
| 新写法 | `docker build` → **Successfully tagged dftest:fixed**；容器内 `BUILD_OK` |

## 新增两个自检工具（防止这两类 bug 再发生）

| 工具 | 作用 |
|---|---|
| `tools/check_dockerfile.py` | 检查 Dockerfile 续行链：链中行的**行内 `#`**、**漏 `\`**、**`\` 落在注释里**，全部报 ERROR。已接入 `selfcheck_pkg.sh` |
| `tools/selfcheck_pkg.sh`（v5.1 加入 Dockerfile 检查） | 10 秒包自检：**镜像 tag 一致性**（build_image 产出 vs serve_a2/run_test 查找）、脚本语法、`MODEL_MOUNTS` 接线、Dockerfile 续行链、执行位、MANIFEST |

> `tools/selfcheck_pkg.sh` 第一次运行就抓出了我自己引入的 tag 不一致
> （`build_image.sh` 产出 `v4`、`serve_a2.sh` 找 `v5`），可见这类检查是必要的。

---

# v5.0（2026-09-16）—— **软链构造的模型目录**

> v3/v4 用软链构造的模型目录起服**必然失败**。

## 故障

量化流水线（modelscope 上那套脚本）产出的最终目录是**零拷贝的软链结构**，
软链是**绝对路径**且**链条很深**。真实产物实测：

```
软链跳数分布: 1 跳 × 2,  2 跳 × 8,  3 跳 × 4,  4 跳 × 80     （共 94 个软链）
```

链条：`L5(最终) → L4 → L3 → L2 → L1(真正的 87 个实体分片)`

而 v3/v4 的 `serve_a2.sh` 只有 `-v "$MODEL:$MODEL:ro"` —— **只挂了 L5**。
容器里所有指向 L4/L3/L2/L1 的绝对路径软链**全部悬空**：
宿主机 `ls`/`cat` 正常，进容器立刻 `No such file or directory`
（通常先炸在 `config.json` / tokenizer 上，白等几分钟后失败在 worker 里）。

**实测复现**（两行就是全部差别）：

```bash
# 旧行为：只挂叶子层
docker run --rm -v <L5>:<L5>:ro alpine cat <L5>/config.json
#   cat: can't open '.../config.json': No such file or directory

# 新行为：逐层挂载
docker run --rm $(bash tools/model_mount_args.sh <L5> | tr '\n' ' ') alpine cat <L5>/config.json
#   {"model_type":"deepseek_v41",...}
```

## 修复

| 文件 | 改动 |
|---|---|
| `tools/model_mount_args.sh` | **新增**。逐跳解析**字面软链**（用 `readlink`，**不是 `realpath`**），把每一跳的目标目录都输出成 `-v` 参数。真值需要挂 **13 个目录** |
| `scripts/serve_a2.sh` | 自动调用上面的工具，用 `"${MODEL_MOUNTS[@]}"` 取代单层挂载；新增 `MODEL_MOUNT_MODE`（`auto`/`ancestor`/`none`）与 `EXTRA_MODEL_MOUNTS`；`DRY_RUN=1` 会打印最终挂载清单 |
| `tools/check_model_dir.sh` | 新增**第 0 步**：起服前检查悬空软链（有则 FATAL 并给出修法）+ 报告软链跳数分布 ⇒ **5 秒内失败**，而不是白等 4 分钟 |
| `README.md` / `REPRO.md` | 新增 §3.0 专章解释这个坑 |

## 一个反直觉的实现要点（写下来避免以后改回去）

**不能用 `os.path.realpath()`**：它会把 `L5→L4→L3→L2→L1` **一次折叠成 L1**，
于是看起来"目标只有 L1，挂 L1 就够"。但**容器是逐跳解析的**：
打开 `/abs/L5/config.json` 读出 `"/abs/L4/..."`，再去开 `/abs/L4/config.json`。
所以**每一跳的目标目录都必须挂**。这就是本工具坚持用 `readlink` 的原因。
（第一版实现正是踩了 `realpath` 这个坑：5 层链条只解析出 2 个目录。）

## 另一个自己踩的坑（同型问题第二次）

`serve_a2.sh` 里判断工具是否存在时我最初写的是 `[ -x ... ]`，而交付包解包后
脚本的**执行位可能丢失** ⇒ 判断为假 ⇒ **静默退回"只挂一层"**，正好把这个 bug 又复现了一遍。
已改成 `[ -f ... ]`（反正调用方式是 `bash <script>`，不需要执行位），
并在 fallback 分支打印显式告警。

## 验证

| 项 | 结果 |
|---|---|
| 5 层人造链条：解析出的目录数 | **5/5**（旧实现只出 2 个） |
| 5 层人造链条：容器内读 3 个文件 | **全部可读**（旧行为报 `No such file or directory`） |
| A3-node1 真值模型目录：解析出的目录数 | **13 个**（含跨到第二个绝对路径前缀的软链） |
| A3-node1 真值模型目录：`check_model_dir.sh` | 94 软链全部可解析；跳数 `1×2, 2×8, 3×4, 4×80` |
| 悬空软链：`model_mount_args.sh` | **rc=1** + 打印断链清单与修法 |
| 悬空软链：`check_model_dir.sh` | **FATAL** + 修法 + 指向自检工具 |
| `DRY_RUN=1` 输出 | 列出最终 `MODEL_MOUNTS` 清单 |

## 未验证

1. **真机起服未跑**（本机没有 A2 的 openeuler 镜像，且按纪律未起容器）。
   上卡第一件事应是 `MODEL=... DRY_RUN=1 bash scripts/serve_a2.sh | grep MODEL_MOUNT` 看清单。
2. `MODEL_MOUNT_MODE=ancestor` 分支只做了 DRY_RUN 级验证，未在真机起服。
3. 若模型目录软链指向了**模型目录之外**的地方（如 `/opt/...`），
   需要 `EXTRA_MODEL_MOUNTS` 手动补 —— 本工具只解析从 `MODEL` 出发能看到的软链。

---
