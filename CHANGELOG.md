# CHANGELOG.md —— v3 → v4 → v5 → v6 → v7 → v8 逐项 diff

# ★★ v9（2026-09-22 21:4x）—— **`ENGRAM=1` × DRAM 卸载 的 P0 修复**（A2 上线的硬前提）

> ## 为什么必须升到 v9
>
> `ENGRAM=1` + 卸载 时，**只要发生一次前缀取回，replay 轮引擎就死**（`a2/logs/073`）：
> ```
> KeyError: 2486 @ models/deepseek_v41/engram_hash.py:463  _engram_update_jit
> ⇒ 8/8 rank 同值 ⇒ 异常从 forward 逃逸 ⇒ device-metadata 标志永久置位 ⇒ EngineDeadError
> ```
> 根因（`a2/logs/075`）：Engram 的 page 镜像（`pages/page_present`）**只由"流经
> `update()` 的 token"写入**；被卸载池**取回**的前缀块从未流经 `update()` ⇒ 命中边界
> （block 对齐：`num_computed_tokens=56320 = 32×1760`）上第一个续算 token 需要的
> lookback 前 1–3 个位置**必然**落在已取回的旧块里 ⇒ 缺页 ⇒ 老代码 `raise KeyError`。
>
> ## 改了什么（只有两个文件）
>
> | 文件 | 改动 |
> |---|---|
> | `vllm_ascend/models/deepseek_v41/engram_jit_kernel.py` | 缺页**不再中止本批**：该行按 `-1` barrier 取 `pad_id` 历史；★ **哈希照常算**（老代码用 `err_page` 当算哈希的开关 ⇒ 一个缺页会让**整批**没有哈希）；新增第 4 个返回值 `miss_rows` |
> | `vllm_ascend/models/deepseek_v41/engram_hash.py` | `miss_rows>0` ⇒ 累加 `pageless_history_rows` + **一次性**打印 `[ENGRAM-PAGELESS]`；**非 JIT（torch）路径同样修**（新增 `_mirror_row()`：缺页补一行全 `-1` 的 barrier 行）；`V41_ENGRAM_PAGELESS_STRICT=1` 可恢复旧的致命行为 |
>
> md5：`engram_hash.py` `3a842bbb…` → **`240c5a04…`**；`engram_jit_kernel.py` `1add256a…` → **`6668d3fe…`**
> （同步更新 `patches/MD5SUMS` 与 `patches/vllm-ascend/MD5SUMS`）。
>
> ## 代价（诚实标注）
>
> ★ 这是**有界降级**，不是零精度损失：每个"取回边界"最多 `1+(lookback-1)=4` 个 token
> 位置的 Engram 历史退化成 `pad`（= "序列从此处开始"的语义）。零损失版（把边界前 1–3 个
> token 的**真实 id** 从 runner 侧送进 `update()`）在单独设计中。**A3 端到端验证【未完成】**：
> 判据 = 同形态臂 replay `failed=0` + `KeyError=0` + `ENGRAM-PAGELESS` 计数 > 0。
>
> ## 怎么用（A2）
>
> ```bash
> cd ~/projects/dsv41-a2-repro-kv8-offloading/deepseek-v4.1-flash-ascend910B
> git pull --ff-only
> bash a2/scripts/check_image_fingerprint.sh dsv41-a2:v8   # 10 秒：确认 v8 缺哪些文件（只读、不拉镜像）
> IMAGE_TAG=dsv41-a2:v9 bash scripts/build_image.sh        # 约 3–6 分钟（基础镜像已在本地）
> IMAGE=dsv41-a2:v9 bash a2/scripts/serve_a2_offload.sh    # 起服（起服前有指纹门）
> ```
> ★ `scripts/serve_a2.sh` 的 `IMAGE` 默认值已同步升到 **`dsv41-a2:v9`**；
> ★ `a2/scripts/serve_a2_offload.sh` 新增**起服前指纹门**：`ENGRAM=1` 时会比对
> host 侧权威副本与镜像内实际那份的 md5，不一致就 **exit 2**（5 秒），
> 避免"照常起服、30 分钟后 replay 才炸"。
> ★ 新增 `a2/scripts/check_image_fingerprint.sh`：10 秒回答"这个镜像与当前发布包差哪些文件"。

# ★ v8.1（2026-09-21）—— codex / OpenAI Responses API 兼容：**一键使能**

> 本包此前**不能**被 codex 直接连。原因不在 vLLM 的 Responses 端点（它在、也通），
> 而在 **DSV4.1 前端编码器只认 chat-completions 的块词汇表**，而 codex 发的是
> Responses 的词汇表。**三个缺陷里有两个是静默的**——这是本次最值得记的一点。

## 0. 结论

| 缺陷 | 修复前 | 修复后 |
|---|---|---|
| `input_text` 块 | 渲染成**字面量** `[Unsupported input_text]`，**HTTP 200 但用户的话没进模型** | 正常渲染 |
| `developer` 角色（codex 放系统指令） | **HTTP 500**（`AssertionError: Invalid message for role 'developer'`） | 200 |
| 正文里的 `<｜User｜>`/`<｜Assistant｜>` | 编码成**单个 token** ⇒ 可**伪造轮次边界** | 零宽空格转义（1 token → 6 token） |

代码改动 **+105 / −5 行**，只动一个文件：
`vllm_ascend/patch/platform/patch_deepseek_v41_frontend/encoding.py`
（原版 md5 `d9f5ee08…` → 补丁版 md5 `c20ee3b6…`）

## 1. 一键使能

```bash
docker exec dsv41-a3 bash /opt/dsv41/tools/enable_codex_responses.sh on       # 装（幂等，自动备份）
docker exec dsv41-a3 bash /opt/dsv41/tools/enable_codex_responses.sh status   # PATCHED / STOCK
docker exec dsv41-a3 bash /opt/dsv41/tools/enable_codex_responses.sh off      # 还原
```

**改的是容器可写层 ⇒ 需要重启服务才生效**；容器重建会丢，重跑脚本即可。
⚠️ **重启前必须确认没有残留进程**——`VLLM::EngineCore` / `VLLM::Worker_TP*_EP*` 的进程名里
**没有** `vllm serve`，`pkill -f "vllm serve"` **杀不到它们**；残留会持有 206 GiB host 注册
与 pinned 内存，导致起服卡在 `rtsMallocHost 207001`（**连试几次都失败**，越试越脏）。
最稳的是 `docker stop -t 10 <容器> && docker start <容器>`。

## 2. 修在哪（**为什么不能修在 HTTP 层**）

先试了最自然的做法：转译代理把 `input_text` → `text` 再转发。
**失败** —— Responses 协议层有严格 pydantic 校验，返回 **239 条 validation errors**
（`Input should be 'input_text'`）。⇒ 协议**要求**块类型就是 `input_text`，
**翻译必须做在校验之后**，也就是 `encoding.py` 里。

三处改动：

1. **块类型别名**（`input_text`/`output_text`→`text`，`input_image`→`image_url`），
   归一放在 `_process_image_blocks()` **入口** ⇒ 它对 `tool_result` 是递归的，
   **嵌套块自动一起归一**；改写走 `{**block, ...}` **不动调用方 dict**；
2. **`developer` 空内容不再 500**（原 `assert content` 改为渲染空块）；
3. **控制 token 转义**：覆盖 `content`/`reasoning`/`reasoning_content`/块 `text`/
   `tool_result` 内层/`tool_calls.arguments`；在**图片块替换之前**执行，
   所以模块自己生成的 `IMAGE_PLACEHOLDER` 不被转义，且图片占位符的**既有报错契约保持不变**。

## 3. 验证（全部实测）

| 层级 | 内容 | 结果 |
|---|---|---|
| 单测 | `test_encoding.py` + 新增 `test_responses_compat.py`（15 例） | **51 passed** |
| 单测 | `test_frontend.py`（需真 tokenizer） | **53 passed** |
| HTTP | 直打 `/v1/responses`：纯串 / `input_text` / `developer` / 控制 token / **图片** | **5/5 通过** |
| **真机** | `codex-cli 0.154.0`：单轮文本 / **工具调用** / **图片** / **多轮 resume** / **子代理** | **全通过** |
| 语义 | 抓包 + 用服务端同一编码器还原 prompt，核对子代理实际收到什么 | 载荷逐字到达、结构计数正确、控制 token 已转义 |

新增用例里**有三条是回归保护**：图片占位符仍报错、生成的占位符不被转义、既有 36 项不受影响。

## 4. 新增文件

| 文件 | 说明 |
|---|---|
| `patches/files/patch_deepseek_v41_frontend/encoding.py` | 补丁版全文（1071 行） |
| `patches/files/patch_deepseek_v41_frontend/encoding.py.diff` | 与原版的 `diff -u`，**路径已规范化成 `a/`…`b/`，可直接给上游** |
| `patches/files/patch_deepseek_v41_frontend/test_responses_compat.py` | 15 个单测 |
| `patches/files/patch_deepseek_v41_frontend/README-integration.md` | 集成与验证报告 |
| `tools/enable_codex_responses.sh` | 一键使能（`on`/`off`/`status`，幂等、备份、回滚） |

## 5. 已知边界

* **`reasoning.encrypted_content` 不支持**：vLLM 侧 `responses/utils.py:274` 直接 `raise`。
  codex 每轮都带 `include: ["reasoning.encrypted_content"]`，但 vLLM **不产出**它，
  所以当前不触发 —— **一旦上游开始产出，这条链会 400**。
* **`developer`→`<｜User｜>`**：真正的系统提示走 `instructions` → 服务端构造成 `system`
  消息（正确）；但 skills/permissions 落在 `<｜User｜>`，prompt 里会有**两个连续 user 轮次**。
  既有设计，实测正常；改动会动到既有语义与单测，**未做**。
* **`input_image` 必须带 `detail` 字段**，否则被 Responses 协议拒掉（400）。
* 本补丁改的是 **vllm-ascend 上游代码** ⇒ 这三条是**通用缺陷**，天然适合向上游提 PR。

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

## 3. ★★ 默认口径定稿（2026-09-20 A2 真机定论）：**A3 默认开、A2 默认关**

> **一句话**：Engram 算子入图（device-index）**只在 A3 默认开启**；A2(910B3)
> **默认关闭**、走 host 路径（功能与精度不变，只是没有该项加速）。
> 判据不是机型名，而是**驱动侧的 `host_mem_pool` 特性** —— 它由 PCI device id
> 在驱动 probe 时定死。

### 3.1 之前的 `auto` 为什么会误判

旧 `auto` 只探 **4 KiB 匿名注册能否被接受**。这个探针在 A2 上**也通过**，
于是误判为支持 ⇒ 起服跑到 **17 分钟**才以 `ret=207001` 失败。
**4 KiB 回答的是"这个 API 被不被接受"，回答不了"整表 206 GiB 能不能注册"。**

### 3.2 真门槛：`host_mem_pool`

驱动 `devmm_dev_capability_support_host_mem_pool(devid)`：

```c
if (g_dev_feature_capabilty_disable[devid][HOST_MEM_POOL_FEATURE]) return false;
if (hccs_connect(devid) || devdrv_is_mdev_vm_full_spec(devid))     return true;
else                                                               return false;
```

`hccs_connect()` 读的是**驱动 probe 时按 PCI device id 定死**的 `connect_protocol`
（`devdrv_pci.c: devdrv_connect_protocol_init`）：

| 机型 | PCI device id | CPU↔NPU 协议 | `host_mem_pool` | 结果 |
|---|---|---|---|---|
| **A3 (910C)** | `19e5:d803` | **HCCS** | **1** | 8 rank × 206 GiB 起服 **133 秒**完成 |
| **A2 (910B3)** | `19e5:d802` | **PCIe** | **0** | 17 分钟后 `ret=207001` |

> ⚠️ 两机的 `npu-smi info -t topo` **都显示 HCCS** —— 那是 **NPU 之间**的互联；
> 这里判的是 **CPU↔NPU** 的连接协议，两者不同，不矛盾。

### 3.3 为什么 `host_mem_pool=0` 会失败（**不是内存不够**）

走了逐页建元数据的慢路径，每 **4 KiB 页 64 B**：

| 数组 | 位置 | 每页 |
|---|---|---:|
| `node->pa_list` | `svm_master_remote_map.c: devmm_alloc_shm_node` | 8 B |
| `node->dma_info` | 同上（`devmm_is_mem_map_by_pcie_th()` 为真时才分配） | 16 B |
| `blks` | `devmm_register_dma.c: devmm_set_register_dma_node` | 40 B |

整表 206 GiB ÷ 4 KiB = **5400 万页/rank** ⇒ 元数据 **3.3 GiB/rank**；其中 `blks`
是**一笔 ~2.06 GiB 的连续 `vmalloc`**，8 个 rank 各要一笔。

**决定性反证**：失败瞬间宿主机 `MemAvailable` 仍有 **703 GiB（95%）**，
`buff/cache` 恰好是 engram 表**一份**（216 GiB ÷ 206 GiB = 1.05 ⇒ `MAP_SHARED`
生效，8 rank 共享同一份物理页）。所以**不是物理内存不足**，是这条慢路径的
**内核侧元数据分配**失败（`207001 = ACL_ERROR_RT_MEMORY_ALLOCATION`,
源码注释写明 "only used by out of memory"）。

### 3.4 三条被实测否证的绕行方案（别再试）

| 方案 | 为什么不行 |
|---|---|
| 写 `/proc/svm/devN/feature/host_mem_pool` | **no-op**。特性表里 `host_mem_pool` 的 `is_support_disable=false`（只有 `bar_mem`/`aic_reg_map`/`shmem_map_exbus` 是 true），`devmm_dev_feature_capability_disable()` 永远写不进 true |
| 换大页（hugetlbfs / THP）绕过 | **无效**。页数按 **VA 范围 ÷ 4 KiB** 硬编码：`devmm_register_dma.c` 用 `devmm_get_pagecount_by_size(vaddr, size, KA_MM_PAGE_SIZE)`，而 `ka_memory_pub.h: #define KA_MM_PAGE_SIZE PAGE_SIZE`。只省物理页表，省不了驱动元数据 |
| 减少 rank 数 / 单 rank 注册 | 单 rank 能过（实测 149.9 s），但生产是 TP8 ⇒ 8 个 rank 各注册一份，逃不掉 |

### 3.5 代码怎么判（自动，不需要用户操作）

`probe_host_mapping_capability()` 增加**第 0 步**：先查
`/proc/svm/dev<N>/feature/host_mem_pool`（普通用户可读），为 `0` 直接判不支持。
于是 `auto`（默认）在 A2 上自动回退 host 路径，日志会打印完整原因；
在 A3 上照常启用。

`V41_ENGRAM_DEVICE_INDEX` 的取值：

| 值 | 行为 |
|---|---|
| **`auto`（默认）** | 先查 `host_mem_pool`（0 ⇒ 直接回退），再探 4 KiB 匿名注册；通过才启用，否则**回退 host 路径**（功能与精度不变） |
| `1` | 强制启用；探测失败即抛错（A3 验收用这个，避免"以为开了其实回退了"） |
| `0` | 强制关闭 |

### 3.6 A2 的后续方向（未做，评估中）

唯一可行方向是**减少注册总量**（例如每 rank 只注册 1/8 的表 ⇒ 元数据降到
412 MiB/rank）。代价是要把 v8 删掉的**跨 rank 查表路由**加回一部分 ——
实测代价约 **+1.5–2.3 ms/step**（host 侧 `route` 1.00 + 设备侧 a2a/bcast 0.77，
见 `reports/engram-final-quantification.md`），且**无法入图**。
相比 `ENGRAM_DEVICE_INDEX=0` 的 60–70 ms/step，仍然值得，但属于下一个 PR。

### 3.7 历史记录（保留，说明当时的判断依据）

**当初为什么不敢默认开**：该能力只在 A3（910C）实测过，A2（910B3）从未在同机验证。
而且本项目自己的 `docs/A2_VS_A3_DIFF.md` §5 记着一条反例：A3 上
`offload.get_dva(pinned_ptr)` 返回 0，AIV 解引用**已注册的 pinned 地址**会报
`507035 MTE invalid GM address` ⇒ "registered host memory" ≠ "device kernel 可直接解引用"。

**资料侧结论是"支持"**（华为官方零拷贝样例 `0_simple_zero_copy` 的产品表含
Atlas A2 训练/推理系列，样例把映射地址当 `GM_ADDR` 传给 AscendC Kernel 用
`DataCopy` 直接读写；`910B` 的 `NpuArch=2201` 也不在唯一的 `arch5162` 不支持清单里）。

**2026-09-20 的现场定论把两件事分开**：A2 的**能力是有的**
（实测单进程注册完整 206 GiB 成功，149.9 s / 159.5 s 两次），
卡住它的是 `host_mem_pool=0` 那条慢路径在**8 rank 规模**下的内核侧元数据分配。
所以"文档说支持"和"生产能用"都对，只是**中间差了一个 `host_mem_pool`**。

A2 上一条命令即可拿到终局答案：

```bash
IMAGE=<你的镜像> DEV=<空闲卡> bash tools/run_probe_hostmap.sh
# 退出码 0 = 支持，3 = 不支持，1 = 探测本身出错
```

（诊断工具：`tools/probe_engram_hostreg.py` + `run_probe_engram_hostreg.sh`
——四维二分"匿名/文件 × PRIVATE/SHARED × 尺寸 × 并发进程数"，
能直接量出失败发生在第几个文件、多少 GiB 处。）

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

### 4.5 mount 模式下 `admission gate` 必须现场打（否则静默失效）

vLLM core 的 `admission_gate` 是**补丁**（不是整文件），Dockerfile 只在
`build_image.sh` 的烘焙路径里 `git apply`。而 **A3 默认走 `PATCH_MODE=mount`**
（官方镜像 + 挂补丁）—— 那条路径**不经过 Dockerfile** ⇒ 不补的话
`VLLM_ADMISSION_GATE=1` 就是一个**没人消费的 env**，prefill 饿死 decode 的保护
完全不存在，而且**没有任何报错**。

现在 `serve_a2.sh` 在 mount 模式下把 patch 挂进容器、起容器后**现场 apply**，
并断言 live tree 命中：

```
[ADMISSION-GATE] mount 模式：在容器内现场应用 admission_gate.patch
[ADMISSION-GATE] 已应用 ✓
[ADMISSION-GATE] live tree 命中 15 处 ✓
[serve_a2] PATCH_MODE=mount ADMISSION_GATE=APPLIED(live_hits=15)
```

### 4.6 ⚠️ 已知故障：`[migrate]` 会让起服**永久卡住**（A3 实测）

内部绑核（`enable_cpu_binding=true`）的最后一步是
`migratepages <pid> <all-nodes> <target-node>`：把 worker 的**全部常驻页**
迁到它那张卡所在的 NUMA 节点。本模型的 worker RSS 极大
（Engram 表常驻 DRAM，实测单 rank **132 GB**，虚拟地址空间 **9.9 TB**），
这一步可能从几十秒变成**永不结束**：

```
$ ps -eo etimes,pcpu,stat,args | grep migratepages
  1201  97.8 R  migratepages 1402 0,1,2,3,4,5,6,7 6
$ grep -o 'N6=[0-9]*' /proc/1402/numa_maps | awk -F= '{s+=$2} END {print s}'
532650        # 90 秒后再测仍是 532650 —— 一个页都没动
```

伴随现象：`shm_broadcast.py:802 No available shared memory broadcast block found in 60 seconds`
（那是**引擎在等 worker**，不是共享内存泄漏 —— 别按泄漏去清 `/dev/shm`）。
引擎不会自己恢复。

**处置**：`CPU_BIND=0` 重启（跳过内部绑核与页面迁移）。实测该臂
**约 15 分钟**起服成功（同一套默认配置 + `ENGRAM_DEVICE_INDEX=auto`），
五项自检全过：`static_kernel` 降级 0、device-index 探测通过、
KV 池 **2,821,337** tokens、`admission gate live_hits=15`、Vision **23/23**。

⇒ A3 上推荐**先 `CPU_BIND=0` 把服务起起来**；要试内部绑核请守着 `ps` 里的
`migratepages`，确认它在动（判据：`numa_maps` 的目标节点页数在涨）。

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

> **⚠️ 本节前半部分是 2026-09-20 上午的历史记录，结论已在同日下午被推翻。**
> 那次"负收益"的真因是**四件套缺一**（capture 期代表值 / 图内 context KV 写入 /
> 常驻索引缓冲 / dispatch 输入换算）。补齐后 **A 与 eager 持平、`ms/step` 反而下降**，
> 且 **A2 上收益比 A3 更大**（−30.5 ms/step、单流 54.7 → 88.7 tok/s）。
> **最新定稿见 §6.2**；给人看的版本见 `README.md` §2.7；完整排查与 5 条否证见
> `reports/draft-graph-investigation-20260920.md`。

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

> **⚠️ 2026-09-20 更正（重要）**：下面表格里"draft 版 + 图关掉 = A 1.84"这一行
> **是单条请求的口径**，与"64 条中位"的基线不可比。用**同一口径**（conc=1 跑 8 条请求取中位）
> 重测后，draft 文件在 **draft eager** 下是 **A=2.648 / 92.0 tok/s**，与 stock 基线同量级
> ⇒ **draft 那三个文件是好的**；坏掉的只有"把 draft 前向放进 ACLGraph"这一件事
> （同口径下 draft 入图 = **A=1.07 / 42.7 tok/s**）。完整记录见
> [`reports/draft-graph-rootcause-20260920.md`](reports/draft-graph-rootcause-20260920.md)。
>
> 口径坑的来源：`tools/bench_concurrency.py` 的 **prompt 条数 = `--concurrency` 列表的最大值**，
> `--concurrency 1` 只发 **1 条**。`tools/draft_graph_guard.sh` 与 `tools/draft_arm_probe.sh`
> 已改为 `1,2,4,8`（8 条中位），脚本里写明了这条坑。

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

### 6.2 ★★ 定稿（2026-09-20 晚）：四件套修复后**A 与 eager 持平，收益为正**

上面 §6.1 的"未完成的重构"已补完。**根因不是竞态、不是桶、不是 padding，而是四件套缺一**
（每一条都独立门控、默认已是 1）：

| 开关 | 位置 | 作用 |
|---|---|---|
| `DSPARK_CAPTURE_VALUE_FIX=1` | `dspark_proposer.py` | 捕获期代表值 + **恢复图内 context KV 写入** |
| `DSPARK_SWA_INDICES_RESIDENT=1` | `dsa_v1.py` | 常驻索引缓冲（图捕获的是 `data_ptr`） |
| `DSPARK_CAPTURE_NCTX_FIX=1` | `dspark_proposer.py` | `_dflash_num_context = num_reqs×(1+SP)` |
| `DSPARK_DISPATCH_QUERY_LEN_FIX=1` | `llm_base_proposer.py` | **P0-B**：dispatch 输入换算（修 conc=7,8,9,17,18,19 崩溃） |

**实测收益（同进程配对臂 / 跨会话，两者都注明口径）**：

| 机器 | 指标 | eager | 入图 | 变化 |
|---|---|---:|---:|---|
| A3（同进程，8 发中位） | ms/step | 36.9 | **23.9 / 24.9** | −35% |
| A3 | 单流 tok/s | 66.5 | **100.7 / 109.8** | +51% ~ +65% |
| A3 | A | 2.455 | 2.403 – 2.738 | 持平 |
| **A2**（跨会话） | ms/step | 64.8 | **34.3** | **−47%** |
| **A2** | 单流 tok/s | 54.7 | **88.7** | **+62%** |

**A2 的绝对收益是 A3 的 2.3 倍**（−30.5 vs −13 ms/step）—— 910B3 的 CPU 更弱，
draft 的 eager 派发开销更大。**A2 打开后单流已追平 A3。**

**精度**：Vision 23/23（A3、A2 各一次）、GSM8K-200 **198/200**、10 条质量判据 10/10。

**发布口径**：`DRAFT_GRAPH` **默认仍为 0**，`DRAFT_GRAPH=1` 是"**推荐开启**"的显式选项。
理由不是收益不足（收益很大），而是失效形态危险：存在一个**极罕见**的
"A 永久 1.00 / 输出变空"坏状态 —— 截至目前 **1 次观测、6 轮独立复现尝试（≈36 个测量点、
10.4 分钟连续负载）全部未复现**，5 条机制猜测全部被否证。
**判据与恢复**：连续两次 specdec metrics 出现 `Mean acceptance length: 1.00`
**且** `Accepted throughput: 0.00` ⇒ 重启服务即可（别发请求试探）。

> 详细数据：`reports/draft-graph-investigation-20260920.md`（排查全过程 + 5 条否证）、
> `reports/a2-draft-graph-20260920.md`（A2 首测）。

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

## 10. ★ 镜像 tag 升到 `dsv41-a2:v8`（**必须重新 build，不能沿用旧镜像**）

> 这是本节里**唯一的用户可见行为变更**：默认镜像 tag 从 `dsv41-a2:v6` 升到
> **`dsv41-a2:v8`**（`scripts/build_image.sh` / `serve_a2.sh` / `run_test.sh` /
> `tools/preflight_a2.sh` / `tools/run_probe_hostmap.sh` 五处默认值同步，
> `tools/selfcheck_pkg.sh` 会断言 build/serve/run_test 三处一致）。

**为什么必须重建、不能 `docker tag` 旧镜像**：v8 的镜像内容与 v5/v6 **不同** ——

| 变化 | 文件 |
|---|---|
| 新增（`newf`，无备份） | `engram_device_index.py`、`engram_graph.py` |
| 改动（`inst`，先备份 `.a2orig`） | `model.py`（+313 行）、`engram_hbm.py`（+31 行） |

沿用 v6 镜像会得到一个**名字叫 v8、内容却是 v6** 的镜像，而 `V41_ENGRAM_DEVICE_INDEX=auto`
在 v6 里**不存在** ⇒ 起服不报错、只是静默跑在没有 device-index 的旧路径上。
因此 `tools/preflight_a2.sh` 里原先那句
"v6 与 v5 逐字节相同 ⇒ 直接 `docker tag` 即可"的提示**已撤销**（它只对 v6 成立），
现在只提示重建；要沿用旧镜像必须显式 `IMAGE=<已有 tag>` 并自行确认内容。

## 11. ★ 修复 `build_image.sh` 的陈旧 md5 校验表（用户实测报障）+ 防复发

**症状（用户报障）**：`bash scripts/build_image.sh` 跑到最后一步报
`FAIL md5 models/deepseek_v41/model.py: got=d22eec4c… want=5b7c4526…`。

**根因**：镜像内自检 `chk()` 里有一张**手写**的 md5 表（11 条），v7→v8 改了
`model.py` / `engram_hbm.py`、新增两个文件后没人同步它。实测三类问题：

| 问题 | 文件 | 详情 |
|---|---|---|
| **STALE**（用户看到的） | `model.py` | 表里 `5b7c4526…`，载荷实际 `d22eec4c…` |
| **STALE** | `engram_hbm.py` | 表里 `6f227a74…`，载荷实际 `02ba2b7c…` |
| **漏项**（装了但没校验） | `engram_device_index.py`、`engram_graph.py` | v8 新增，表里根本没有 |

同时发现两处**同类**陈旧：`patches/vllm-ascend/MD5SUMS` 缺 v8 两行 + 2 行旧值
（它在 `patches/README.md` 里被承诺"`md5sum -c` 应与 `patches/files` 一致"），
`patches/MD5SUMS` 的 `serve_v2.sh` 行也对不上。均已按载荷字节修正。
（注意：载荷本身是对的 —— `MANIFEST.sha256`、`patches/_tools/verify_series.sh` 的
`VERIFY: ALL PASS` 三方一致，**错的只是清单**。）

**防复发（消除双份真相，而不是"再同步一次"）**：

| 事实 | 唯一权威来源 |
|---|---|
| 装到镜像的哪个路径、`inst` 还是 `newf` | `Dockerfile` 的落位表 |
| 载荷内容 | `patches/files/**` 的实际字节 |
| 期望 md5（公开清单） | `patches/MD5SUMS` |

* `scripts/build_image.sh` 不再内联任何 md5：构建时用
  `tools/check_checksums.py --manifest` **由载荷字节现算**期望值，落位表由 Dockerfile 推导
  （含 `token_dispatcher_moemask.py → /tmp/bake/token_dispatcher.py → ops/fused_moe/token_dispatcher.py`
  这类改名映射），再交给 `tools/verify_baked_tree.sh` 在镜像内逐条核对 md5 + `py_compile`
  + `inst` 项的 `.a2orig` 回滚备份是否存在。
* `tools/check_checksums.sh` 是 10 秒的静态检查（不需要 docker），已接入
  `tools/selfcheck_pkg.sh`；三方任何一处不一致都会 FAIL，并且会报出
  "装了却没被校验"（孤儿载荷）与"校验了却没装"两类问题。
* `tools/negative_control.sh` 新增 NC9/NC10：**证明这套检查真的会抓到**上面那两个 bug
  （载荷改一个字节必须 FAIL、缺 `.a2orig` 必须 FAIL、清单过期必须 FAIL）。
* 实验残留（部署副本上并行调试留下的 `*.bak-probe`）**不算载荷**、不参与校验：检查器只报
  NOTE，`--strict-artifacts` 才会强制要求它们登记进 MD5SUMS —— 避免"忘了删备份文件"
  被误判成发布缺陷。

> 教训与 v5 的 `selfcheck_pkg.sh` 那次同型：**清单不能手工维护**。
> 这次连"实例"一起修：两个 md5 清单文件也与载荷对齐了。

## 12. ★ 合并 A2 真机调出来的 PGO 修复（编译容器复用 / 续编 / RPATH 旧库）

> 来源：用户在内网 A2 上跑 `scripts/build_image.sh` + `build_scripts/00_ensure_pgo.sh`
> 时踩坑后手改的 `dsv41-a2-modifies.diff`。下面每一条都**有真机报错原文**支撑；
> 逐条核对后合并，并对其中三处做了加固（见"加固/改写"栏）。

| # | 文件 | 真机问题 | 合并内容 | 我做的加固 / 改写 |
|---|---|---|---|---|
| 1 | `build_scripts/00_ensure_pgo.sh` | 编译 30–40 分钟，`docker run --rm` 一失败就把"已装依赖 + 已完成进度"全丢掉 | 编译容器去掉 `--rm`、固定名 `pgo-build-a2`；参数一致时 `docker start -ai` 续编；失败保留容器；`PGO_RM_CONTAINER=1` 才删 | ① `-e http_proxy=$http_proxy` 在 `set -u` 下会 **unbound 崩溃**（已有对照实测）⇒ 改为"只透传真的设了的代理变量"，且只打印变量名（代理 URL 常带凭据）；② 新增**防假成功**判据（见下）；③ `scripts/openEuler.repo` **只在文件存在时才挂**（否则 docker 会创建一个同名目录把容器内 repo 顶坏）；④ `PGO_RM_CONTAINER=1` 在失败路径也生效，不再"只成功时才删"；⑤ 代理/openEuler.repo 只在创建时生效这一点写进提示 |
| 2 | `build_scripts/02_fetch_source.sh` | 每次跑都 `rm -rf` 源码树 ⇒ 已编译的 `.o` / profile 数据全作废 | 新增增量续编分支：`Makefile + configure + pyconfig.h` 齐备且 `PGO_FORCE != 1` 时直接 `exit 0` | 保留；并把判据来源（后两者由 `03_configure.sh` 生成）写进注释；`PGO_FORCE=1` 仍走完整重下/重解压 |
| 3 | `build_scripts/04_make.sh` | `./python: undefined symbol: __gcov_indirect_call` —— 新编 `./python` 的 `DT_RPATH` 指向镜像里那份**非 PGO** libpython，而 **RPATH 优先于 `LD_LIBRARY_PATH`** | 构建容器内把旧库移到 `/work/logs/image-libpython-quarantine/`；并删掉上次失败残留的 `pybuilddir.txt` / `platform`（否则 make 认为"已最新"跳过重生成） | 保留；补 `PGO_IMAGE_LIBDIR` 覆盖口、失败时的显式告警、以及"重复运行会走 else 分支"的说明；隔离目录落点与 `.gitignore` 对齐 |
| 4 | `build_scripts/06_package.sh` | `tar tzf … \| head -40` 在 `set -o pipefail` 下：head 读够就关管道 ⇒ tar 收 SIGPIPE ⇒ **打完包之后**才整脚本失败，极难查 | 先落全量清单 `sourcetree_manifest.txt` 再 `head`；`site-packages` 先判存在/判空 | 保留；实测复现：4000 条目 tar 的 rc=**141**，且旧写法确实在"打包完成之后"中止 |
| 5 | `tools/fetch_corpus.sh` | 内网自签证书导致 `curl` 拒绝下载 | `curl -k` | **不无条件合并** ⇒ 改为 `INSECURE_TLS=1` 门控（默认关），见 §13 |

### 12.1 防假成功（用户点名的风险，已实测复现并修掉）

用户把失败路径的 `die` 改成 `say`（为了保留容器、让第 5 步的产物检查兜底）。这会带一个
**静默假成功**风险：编译失败但 `optim/pgo/` 下还留着上次成功的产物 ⇒ 脚本会写出一份
"指纹正确、产物陈旧"的 marker，之后每次都被"秒钟退出"骗过去（永不重编）。

修法：第 4 步记 `_rc`；第 5 步

* `_rc != 0` 时**只采纳 mtime 晚于本次开工时间**（`optim/pgo/build/logs/.build_start_stamp`）的产物，
  旧产物显式打"忽略陈旧产物"并跳过；
* 判据是「**本次真的拷到 2/2 件**」而不是"`optim/pgo/` 下有文件"（后者会被上一轮的成功残留骗过）；
* 两件都拿不到 ⇒ `die`（不写 marker、不报成功），容器保留可续编；
* 只有"`_rc != 0` 但产物是本次新生成且齐全"（典型：`06_package.sh` 打完包之后才报错）才采纳，
  并**大声告警**；marker 里记 `build_rc=` 与 `build_container=`，秒退路径也会把 `build_rc != 0` 再提醒一次。

> 离线用 stub docker 端到端验证了 4 种形态：成功 / 失败+陈旧产物（**不写 marker、rc=1**）/
> 失败+新鲜产物（采纳+告警+marker 记 `build_rc=16`）/ marker 命中秒退；另验证了容器
> "参数一致⇒复用（`docker start`）"、"参数变化⇒重建（`rm`+`run`）"、"容器在跑⇒拒绝并发"。

## 13. ★ `INSECURE_TLS`：内网可用性 vs 公网安全（**默认关**）

用户的 diff 里有三处 TLS 降级（`02_fetch_source.sh` 两处 `curl -k`、`tools/fetch_corpus.sh`
一处 `curl -k`、以及往 `/etc/yum.conf` 写 `sslverify=False`）。这在**内网自签证书/中间盒**
下是必要的，但本仓是**公开**仓库：无条件关掉证书校验属于**安全降级**，不能默默带出去。

⇒ 统一改成显式开关 `INSECURE_TLS=1`（默认关）：

```bash
INSECURE_TLS=1 bash build_scripts/00_ensure_pgo.sh    # 内网自签证书机器才需要
INSECURE_TLS=1 bash tools/fetch_corpus.sh             # 语料下载同理
```

* 生效范围：给两处 curl 加 `-k`；往容器内 `/etc/yum.conf` 与 `/etc/dnf/dnf.conf` 写
  `sslverify=False`（**幂等**：先 grep 再 append，重复运行不会叠加 —— 编译容器现在会被复用，
  这点很重要）；
* **完整性不依赖 TLS**：`02_fetch_source.sh` 仍然做华为云/阿里云**双源 sha256 交叉校验**，
  `tools/fetch_corpus.sh` 仍然逐文件比对 sha256 —— `-k` 只影响传输层身份验证；
* 用法写进了三个脚本头注释、`optim/pgo/README.md`（内网章节）与本节。

> `docs/RELEASE-NOTES.md` **没有改**：它是 v7 的历史发布说明（"相对 a2_pkg_v6 的变化"），
> 按本仓库既定的纪律，历史记录不追改 —— v8 的变更统一记在本 CHANGELOG。

## 14. ★ 修复 `engram_int8/` 的挂载权限（A2 真机报障：`aclrtHostRegister failed: ret=507899`）

### 14.1 用户报障（A2 真机，原话）

> v8 以后，启动 A2 脚本，模型的 `engram_int8/` 目录就必须是可写入权限，如果是 readonly
> 加载权限时候会报错。感觉 engram 那个开关在 V8 里面可能存在问题，我现在把所有模型判断的
> ro 都改成了 rw，然后把开关从 auto 改成了 1，能过了。

### 14.2 根因（三条路径，两条漏了）

`engram_device_index.py::_map_and_register()` 要求**可写映射**，这是硬约束、不是配置口味：

```python
fd   = os.open(self.path, os.O_RDWR)                      # ← 要求可写
addr = _libc.mmap(None, maplen, PROT_READ | PROT_WRITE, MAP_SHARED, fd, aligned)
dev, ret = acl.rt.host_register(addr, maplen, ACL_HOST_REGISTER_MAPPED)   # 只读 VMA → ret=507899
```

而 `serve_a2.sh` 有**三条**构建 `MODEL_MOUNTS` 的路径，**只有第 2 条**（`auto`，默认）有
engram 特判：

| # | 路径 | v8 之前的行为 | 后果 |
|---|---|---|---|
| 1 | `MODEL_MOUNT_MODE=ancestor` | 整棵公共祖先 `:ro`，**无 engram 特判** | 容器内 `ret=507899` |
| 2 | `MODEL_MOUNT_MODE=auto`（默认） | `case "${_d##*/}"` 命中 `engram_int8`/`engram-int8` → `:rw` | 正常，但**判定太脆**（见 14.4） |
| 3 | 单层 fallback（`none` / 缺 `model_mount_args.sh`） | 只挂 `$MODEL:ro`，**无 engram 特判** | 容器内 `ret=507899` |

**为什么用户改 `auto` → `1` 也"能过"：那个开关跟挂载权限无关。** 代码级判定：
`_engram_need_rw()` 对 `auto` 与 `1` **都返回真**（只有显式 `0/false/off/no/空` 才返回假），
所以两条取值在挂载构造上完全等价；容器侧的差别只在"能力探测被拒绝"时——`auto` 静默回退
host 路径，`1` 直接抛错（`_ENGRAM_DEVICE_INDEX_MODE`）。用户是同一次改了**两处**
（"把所有 ro 改成 rw" + "auto 改成 1"），真正生效的是前者。
⇒ 发布口径**保持 `auto` 默认**，不要建议用户改 `1`：它只会把"静默回退"变成"硬失败"。

### 14.3 修复

1. **三条路径统一**：新增 `[ENGRAM-RW]` 判定 + `[ENGRAM-RW-OVERLAY]` —— 祖先/单层挂载
   保持 `:ro`，只把 engram 表目录**嵌套叠加**成 `:rw`（不再"整棵模型目录开成 rw"，也不是
   照抄用户的 workaround）。
2. **判定口径从"目录名"改成"三来源"**（旧代码只看 `_d` 的 basename，三种形态全漏）：
   * `$MODEL/engram_int8` **本身是实体目录**时，它根本不在 `model_mount_args.sh` 的输出里；
   * 名字含 `engram` 且含 `int8` 的目录（`engram_int8` / `engram-int8` / `engram_int8_data` …）；
   * **真正落盘的那个目录**：`engram_int8/` 里的条目本身还是软链（`engram_dr_build.py`
     就这么造的），而 `O_RDWR` 是按最终 inode 所在目录判定的。
     A3 真机实测链条（见 14.5）：`$MODEL/engram_int8 → L4/engram_int8 → L3/engram_int8`（实体）
     `→ 4 个软链 → …/projects/dsv41/models/out/engram-int8 → …/models/out/engram-int8`，
     **最后一步跟模型树不在同一棵目录树里**。
3. **起服前自检（前移到 `docker run` 之前）**：目录存在 + **最深覆盖它的那条挂载必须是
   `:rw`**，否则 `die` 并给出可直接照做的修法（`MODEL_MOUNT_MODE=auto` / `ENGRAM_DEVICE_INDEX=0`）。
   价值：用户现在的报错在**容器内、起服中途**，把定位成本从"一整轮试错"降到"起服前一行字"。
4. ⚠️ **"宿主上可写"只告警、不拦**。第一版按 `[ -w "$dir" ]` 硬判，结果在 A3 真机上**自己
   把自己拦住了**：交付模型里 `…/models/out/engram-int8/*.safetensors` 是 **`root:root 0600`**，
   而容器是**以 root 运行**的（`Dockerfile` `USER root` + `docker run` 不带 `--user`）——
   `rw` 挂载下 root `O_RDWR` 完全没问题。⇒ 真正决定成败的是**挂载模式**，不是宿主上的模式位；
   宿主不可写只提示"容器必须是以 root 起的；换非 root 才需要 chmod/换属主"。
5. **`ancestor` 模式顺带修掉一个半坏**：原实现取 `dirname | sort -u | head -1` 当"公共祖先"，
   而 A3 真机的模型树横跨 `models/out` 与 `projects/dsv41/models/out` **两棵树** ⇒ 挑出来的
   `models/out` **覆盖不到** `projects/…`，容器里那些软链全是悬空（**不报错**，只是读不到）。
   现在加了覆盖性检查：覆盖不全就退回 `auto`（与"取不到公共祖先就退回 auto"同一策略）。
6. **脚本指纹**：起服时打印 `script=… ver=v8-engram-rw-mount-20260920 md5=…`。镜像里也有一份
   烘焙的 `/opt/dsv41/scripts/serve_a2.sh`（构建时快照，可能比包旧）；打印路径若是
   `/opt/dsv41/scripts/…`，脚本会显式告警让你改用包里那份。README §2.2 同步写明。

### 14.4 次生问题（顺手排查的结果）

| 疑点 | 结论 |
|---|---|
| `case "${_d##*/}"` 依赖目录名 | **成立**，已改成三来源判定（14.3-2）。旧代码在"`engram_int8` 是实体目录"这种形态下**永远不命中** |
| `model_mount_args.sh` 会不会漏掉 `engram_int8` | **会**：它只输出"软链目标目录"，**实体子目录不输出**（`scan()` 里真目录只递归、不登记）。这正是上面那条最危险的形态 |
| 软链名还是真实名 | 输出的是**每一跳的目标目录**（`readlink` 逐跳，不是 `realpath`），所以链条上每一层都会出现；但**最后一跳的实体目录**要靠新加的第③条规则才拿得到 |
| 用户是不是在跑镜像里烘焙的旧脚本 | 排查项已内建：起服打印路径+版本+指纹，命中 `/opt/dsv41/scripts/` 时显式告警；README §2.2 写明"用包里的脚本" |

### 14.5 验证（**没有 A2 访问权限**，所以全部是离线 + A3 宿主侧的 dry-run）

| 验证 | 结果 |
|---|---|
| `tests/engram_rw_mount_test.sh`（新增，34 条断言，假模型树 + `DRY_RUN=1`，不需要 docker/NPU） | **pass=34 fail=0**：auto / ancestor / 单层 fallback / 缺工具 / 实体目录 / `ENGRAM_DEVICE_INDEX=0` / 宿主不可写 / 缺表目录 全覆盖 |
| `tools/negative_control.sh`（新增 NC11–NC14） | **PASS=23 FAIL=0**：①宿主不可写只告警不误杀 ②正常树 engram 必须 `:rw` 且模型根仍 `:ro` ③祖先覆盖不全必须退回 auto ④config 声明 engram 却没有表目录必须拦 |
| A3 真机（宿主侧 dry-run，不碰 docker） | 真实模型 `v41-w4a8-engram-dr-vision-qrot-mtpq`：`auto` 下 5 个 engram 目录**全部 `:rw`**、其余 10 个目录 `:ro`；`ancestor` 正确打印"覆盖不到 … 退回 auto"；`none` 下 `$MODEL:ro` + 3 个 engram 目录 `:rw` |
| `bash -n scripts/serve_a2.sh` / `tools/selfcheck_pkg.sh` / `tools/check_checksums.sh` | 见本次提交说明 |

> **未验证**：A2 真机起服（我们没有 A2 的访问权限）。用户侧一条命令即可验证：
> `DRY_RUN=1 MODEL=<模型目录> bash scripts/serve_a2.sh | grep engram_int8` ——
> 期望每一行都以 `:rw` 结尾，且**不出现** `$MODEL:$MODEL:rw`。

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
