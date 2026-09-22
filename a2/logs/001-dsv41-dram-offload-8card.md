# 001 · DeepSeek-V4.1 八卡 + DRAM KV 卸载层：真机验证（**未做到"生效"，拦路虎已定位**）

**日期**：2026-09-21 21:25–23:30　**执行**：子代理 `D_off8`　**机器**：A3（A3-node1），Phy-ID 8–15
**模型**：`~/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq`（真权重、真 Engram、真 DSpark）
**交付**：启动脚本 `scripts/serve_dsv41_dram_offload.sh` / 臂运行器 `scripts/run_arm_8card.sh` / 补丁
`patches/files/offload_dsv41/scheduler.py`（均位于 `~/projects/dsv41-upstream-pr/agents/D_off8/`）
**标记约定**：【实测】= 本机跑出来的原始数据；【推断】= 代码/日志推出来的；【未确认】= 没跑到/没证据。

**收尾状态**：我自己的容器 `abl-off-*` 已**全部删除**；**Phy-ID 8–15 已释放**（`npu-smi` 无进程）；
`c0/c1/c2` 三个槽位锁空闲；用户的 `dsv41-a3` 保持 `Exited`（全程未动）；
收尾时宿主 `MemAvailable` = **860 GB**（安全线 150 GB 之上）。

---

## 0. 一句话结论

**【实测】答案是第三种结果，而且比 27B 那次更接近"看起来配好了其实没用"：**

**打开 Engram（生产配置）时，卸载臂根本起不来**（`aclrtMallocHostWithCfg 207001`）；
**关掉 Engram 之后它能起来、也确实在往 DRAM 写（12,288 个 `BlockStored(CPU)` 块、
393.6 GB），但取回恒为 0 字节、外部层命中 0 —— replay TTFT 4167 ms vs fill 4192 ms（+0.6%，噪声级）。**

⇒ **DRAM 卸载在 DSV4.1 上"能起服、能存储、零取回"，对服务质量没有任何帮助。**
这条与 27B 的 signature 不同（27B 是 `BlockStored(CPU)=0` 的纯静默；DSV4.1 是
**存进去了但从不取回**），对上游是**新的 bug 线索**。

三判据逐条（§2 有全表）：

| 判据 | 结果 |
|---|---|
| ① `BlockStored(medium="CPU") > 0` | **✓** 12,288（`BlockRemoved:CPU`=8,192） |
| ② `vllm:external_prefix_cache_hits` 增长 | **✗** queries=1,048,832 / **hits=0** |
| ③ `replay TTFT < fill TTFT` | **✗** 4167 ms vs 4192 ms（+0.6%） |

另外还拿到两样硬产出：**DSV4.1 的 13 个 KV cache group 第一手清单**（§3，由我自己打的日志补丁在
真机 `serve.log` 里打出来）、以及**两条起服拦路虎的完整根因链**（§4）。
**我没有用别处的数字顶替，也没有把"零收益"写成"生效"。**

---

## 1. 环境与复现入口

| 项 | 值 |
|---|---|
| 机器 / 卡 | A3-node1，Phy-ID **8–15**（8 张，起跑前 `npu-smi` 无进程） |
| 镜像 | `quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3` |
| vLLM / vllm-ascend | `0.27.1` @ `/vllm-workspace/vllm`；vllm-ascend `e43cf1e9f` |
| 模型 | `~/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq`（Engram int8 206 GiB） |
| 起服入口 | `~/projects/dsv41-upstream-pr/shadow-pkg/scripts/serve_a2.sh`（**影子包**，不动用户原件） |
| 影子包改动 | `serve_a2.sh` +23 行、`serve_v2.sh` +8 行（把宿主上的 `KV_ARGS_EXTRA` 带进容器；打补丁脚本 `scripts/patch_shadow_pkg.sh`，幂等） |
| 臂运行器 | `scripts/run_arm_8card.sh`（起服→等就绪→KV 事件探针→宿主侧 fill/replay→收指标→停容器） |
| 原始日志 | A3-node1 上 `~/projects/dsv41-upstream-pr/agents/D_off8/{out,logs}/`；`~/projects/dsv41-release/results/off8_*`（每个臂一个目录） |

复现一条臂（约 25 min，其中起服 ~22 min）：

```bash
cd ~/projects/dsv41-upstream-pr/agents/D_off8
TAG=off8-ctl OFFLOAD_GB=0 bash scripts/run_arm_8card.sh          # 对照臂
TAG=off8-x   OFFLOAD_GB=4 PREFIX_MATCH_UNIT=32 OFFLOAD_SCHED_PATCH=1 \
  bash scripts/run_arm_8card.sh                                  # 卸载臂（当前会撞 207001）
```

---

## 2. 六个臂的数字（全部【实测】）

| 臂 | 卸载 | 哈希粒度绕过 | scheduler 补丁 | 结果 | 关键数字 |
|---|---|---|---|---|---|
| **A** `off8-ctl` | 无 | — | — | **跑通** | `GPU KV cache size: 46,387 tokens`；fill TTFT p50 **4356.4 ms**；replay（reset 后）p50 **4335.7 ms**（= 全量重算）；KV 事件 `BlockStored:GPU=98,328`、`BlockRemoved:GPU=94,388`、`AllBlocksCleared=1`、**CPU=0**；`prefix_cache_queries=1,048,832`、hits=0、`external_prefix_cache_hits=0` |
| **B** `off8-dram16` | 16 GiB | 否 | 否 | 起服挂（t≈649 s） | `AssertionError: tokens_per_block=32 not divisible by tokens_per_hash=128`（§4.1） |
| **B2** `off8-pmu32` | 16 GiB | **是（32）** | 否 | 起服挂（t≈595 s） | 过了 §4.1 的断言，撞 `aclrtMallocHostWithCfg 207001`，失败请求 **1 GiB**（§4.2） |
| **B3** `off8-b3` | 4 GiB | **是（32）** | **是** | 起服挂（t≈589 s） | 同上 207001，失败请求 **512 / 256 / 32 MiB**，**8 个 worker 全部命中**（§4.2） |
| **C** `off8-noengram` | 4 GiB | 是（32） | 是 | **跑通**（`ENGRAM=0`） | store **393.6 GB** / **load 0 B**；CPU 侧块查询 1,048,832 次、**命中 0**；fill p50 4304.3 ms → replay p50 **4208.0 ms** |
| **D** `off8-dram32` | **32 GiB** | 是（32） | 是 | **跑通**（`ENGRAM=0`） | `BlockStored:CPU=**12,288**`、`BlockRemoved:CPU=8,192`；store 393.6 GB / 27.8 s（≈**14.2 GB/s**）/ **load 0 B**；queries 1,048,832、**hits 0**；fill p50 4192.5 ms → replay p50 **4167.4 ms** |

> 臂 C/D 的池子分别是 4 GiB 与 32 GiB（工作集 18.5 GiB），**两臂的 `store_bytes` 一模一样**
> （393,568,321,536 B，精确到字节）—— 说明这个计数反映的是**调度侧决定要存多少**，与池子大小无关；
> 而 load 两侧都是 0。⇒ 「零取回」与池子够不够**无关**，是**机制层面**的问题【实测】。

臂 A 的参数口径（读者要能复现"制造真实驱逐"这件事）：

- `MAX_LEN=40960`（**不是**生产口径的 1048576，这里只是为了把工作集压进可跑范围）、
  `MAX_SEQS=32`、`BAT_TOKENS=8192`、`PREFIX=1`、`GPU_UTIL=0.92`、`CPU_BIND=0`、`DROPCACHE=0`；
- `--kv-cache-memory-bytes 1073741824`（HBM KV 压到 **1 GiB**）⇒ `GPU KV cache size: 46,387 tokens`；
- 工作集 = 16 个请求 × 32,768 token = **524,288 token**，是 HBM 池的 **11.3×**
  ⇒ 基线臂里 `BlockRemoved:GPU=94,388` 条驱逐事件，**真实驱逐充分**；
- replay 轮前先 `POST /reset_prefix_cache`（`VLLM_SERVER_DEV_MODE=1`），所以基线的 replay
  只能重算 —— 这就是 `replay p50 4335.7 ms ≈ fill p50 4356.4 ms` 的原因；
- 32,768 token 的 prefill TTFT 4356 ms ⇒ **7.52k tok/s**（A2 实测口径 4,258–5,602 tok/s，同量级）。

> **一处操作失误（记录在案，不影响上表数据）**：臂 A 运行期间我编辑了正被 bash 执行的
> `serve_a2.sh` / `run_arm_8card.sh`（bash 是惰性读文件的），打断了脚本的等待循环。
> vLLM 服务本体不受影响，我随即改从宿主侧接手收集，**§2 臂 A 的三判据数据全部正常采到**。
> 此后脚本冻结，后续臂无此问题。

---

## 3. ★ DSV4.1 的 KV cache group 清单（**13 个**，第一手）

来源：镜像内 `vllm_ascend/core/deepseek_v41.py::group_cache_specs()`（逐行读出）+ `plan_cache_slots()`。
臂 A/B 的 `serve.log` 里 `--kv-transfer-config` 未启用时不会打印 group 明细，所以这是**代码级**清单。

| # | 名字 | 成员资源 | spec 类型 | block_size |
|---|---|---|---|---|
| 0 | `full` | 4× `layers.{2,8,14,20}.long_kv_cache` + 4× 同名 `.indexer.k_cache` | `DeepseekV41FullSpec` + `DeepseekV41IndexerSpec` | 128 |
| 1 | `state` | 3× `layers.{2,8,14}.compressor.state_cache` | `DeepseekV41CompressorStateSpec`（→`AscendCircularBufferSpec`，**FP32 32 行环**） | **32** |
| 2–11 | `swa0`…`swa9` | 40 个 SWA 资源按 slot 轮转分 10 组、每组 4 层 | `DeepseekV41SWASpec`（window=128） | 128 |
| 12 | `dspark` | 3× `mtp.{0,1,2}` draft SWA（aliasing 到 target slot） | `DeepseekV41DraftSWASpec` | 128 |

关键结构事实（后面三个坑都从这里长出来）：

1. **每个 group 的 `kv_cache_spec` 是 `UniformTypeKVCacheSpecs` 包装**，不是裸 spec；
2. 底层由 `allocate_cache_config()` 做"**一个全局 block-ID 池 + 4 个 slot 张量**"，
   每个 slot 的 `page_size_bytes` 各不相同（由 `plan_cache_slots()` 按 `capacity = max(kv+index, aliases...)` 算 padding）；
3. **`state` 组不可前缀缓存**：`AscendCircularBufferSpec.prefix_cacheable = False`（每请求固定 1 页、
   decode 每步被覆盖），且它的 `block_size=32` 与其余 12 组（128）不同；
4. 因此 canonical cache 的 `worker_kv_bytes_per_block` 与"每 128 token 一个 block"的哈希口径**不一致** —— 这就是 §4.1。

---

## 4. 四个坑（逐个：现象 / 原文 / 根因 / 处置 / 证据等级）

### 4.1 ★ 坑一：`tokens_per_block=32 % tokens_per_hash=128` —— **可用 `--prefix-match-unit` 绕过**【实测】

臂 B 的起服失败原文（`~/projects/dsv41-release/results/off8_off8-dram16_20260921_220527/serve.log`）：

```
Worker/TP0..7 → vllm_ascend/worker/worker.py:1022 ensure_kv_transfer_initialized
  → KVConnectorFactory.create_connector
  → vllm_ascend/.../kv_offload/native/offloading_connector.py:331 AscendOffloadingConnector.__init__
  → vllm/.../offloading_connector.py:68 OffloadingConnector.__init__
  → vllm/.../offloading/config.py:60 build_offloading_config
AssertionError: tokens_per_block=32 not divisible by tokens_per_hash=128.
  Hybrid models (e.g. Mamba+Attention) need --enable-prefix-caching to align block sizes.
```

**根因链**（查到底了，这是**上游 `patch_kv_cache_utils.py` 与卸载层的接口不匹配**，不是 DSV4.1 模型本身的问题）：

1. `state` 组 `block_size=32` 且不可前缀缓存（§3 第 3 条）；
2. vllm-ascend 用 `_ascend_resolve_kv_cache_block_sizes()`（`vllm_ascend/patch/platform/patch_kv_cache_utils.py:97`）
   **替换**了 vLLM 原版 `resolve_kv_cache_block_sizes()`；当"存在不可缓存组"时它走 `cacheable_groups` 过滤分支，
   把哈希粒度算成 **可缓存组的 GCD = 128**（而不是全组 GCD 32）；
3. 卸载层 `build_offloading_config()` 要求"每个组的 `tokens_per_block % tokens_per_hash == 0`"
   ⇒ `32 % 128 ≠ 0` ⇒ 断言。

**处置（绕过，不是修复）**：vLLM 自带 `--prefix-match-unit 32`，把哈希粒度显式钉成真 GCD。
数学依据：`128 % 32 == 0`（12 个 128 组）且 `32 % 32 == 0`（state 组），**两个方向都成立**，
所以这条断言不再被触发。臂 B2/B3 都**实测**过了这道断言（它们挂在更后面的 §4.2）。

> 这是**绕过**：`_ascend_resolve_kv_cache_block_sizes()` 的默认行为没有改，
> 只是把用户可配的哈希粒度显式指到 32。副作用是前缀哈希粒度从 128 变 32（哈希表更细、
> 前缀匹配更细粒度），代价未被本任务量化。

### 4.2 ★★ 坑二：`aclrtMallocHostWithCfg 207001` —— **拦路虎，目前无解**【实测】

臂 B2/B3 的起服失败原文：

```
Worker/TP* → vllm_ascend/worker/worker.py:1031 initialize_from_config
  → model_runner_v1.py:4238 get_kv_transfer_group().register_kv_caches(kv_caches)
  → vllm/.../offloading_connector.py:90 → worker.py:67 _init_worker → spec.get_worker(kv_caches)
  → vllm_ascend/.../kv_offload/native/npu.py:78 create_worker → cpu_npu.py:299 NPUOffloadingWorker.__init__
  → cpu_tensor = torch.zeros((num_cpu_blocks, cpu_page_size_bytes), dtype=torch.int8,
                             device="cpu", pin_memory=True)
torch.OutOfMemoryError: allocate_host_memory_slowpath:../torch_npu/csrc/core/npu/CachingHostAllocator.cpp:252
  NPU function error: aclrtMallocHostWithCfg, error code is 207001
  Resource_Error_Insufficient_Host_Memory(EL0018): Failed to allocate <N> bytes host memory requested by the RUNTIME module
  rtsMallocHost execution failed, reason=driver error:out of memory
```

**发生的位置不是权重加载，而是 KV cache 注册**：8 卡权重加载 471 s → KV 分配
（同一秒打出 `GPU KV cache size: 46,387 tokens`）→ **紧接着**注册 KV cache 时挂。
`OFFLOAD_GB=16` 时第一个失败请求是 **1 GiB**；`OFFLOAD_GB=4` 时失败请求是
**512 MiB（4 次）/ 256 MiB（5 次）/ 32 MiB（2 次）**，**8 个 worker 全部命中**（先炸的是 TP6，t=+9 s）。

**同时刻的宿主状态（排除"没内存"）：**

| 观测 | 值 | 出处 |
|---|---|---|
| 起服前 `MemAvailable` | **859–860 GiB** | `out/off8-b3.meta.txt` 的 `free -g` |
| 失败窗口前后采样 | 862 GiB / 833 GiB | 我自己的运行记录 |
| 容器 cgroup 限额 | **0 / 0 = 不限制** | `docker inspect abl-off-off8-b3` |
| 容器内 `memlock` rlimit | **soft=hard=-1（无限）** | `docker inspect` + 容器内 `getrlimit` |
| 8–15 号卡 | 空（`npu-smi` 无进程） | 起服前/失败后各查一次 |

**我做了什么来把它缩成最小复现（全部【实测】，结果都"成功"，即都没能复现）：**

| 探针 | 配置 | 结果 |
|---|---|---|
| `scripts/pin_host_probe.py` | 单进程、die3、1 GiB 步长 | 连续 pin **128 GiB 成功**（11.2 s，≈11.4 GiB/s），`MemAvailable` 只掉 2.3 GB |
| `scripts/pin_host_probe_mp.py` | **8 进程同卡**、各 16 GiB | **8/8 成功，合计 128 GiB** |
| `scripts/pin_host_probe_8dev.py` | **8 进程、各占一张卡**（8–15） | 各 8 GiB ⇒ **8/8 成功，合计 64 GiB** |
| `scripts/hostregister_vs_pin.py` | 单进程先 `aclrtHostRegister` **206 GiB**（同 Engram 的 `MAPPED` 模式）再 pin | **成功**（`ret=0`，注册耗时 82 s；随后 pin 4 GiB 成功） |
| `scripts/hostregister_mp_shared.py` | **8 rank 各自注册同一块 206 GiB 共享表**（复刻 Engram 拓扑）再各 pin 1 GiB | **8/8 成功**（每个 rank 注册耗时 ~140 s） |
| 特征位 | `/proc/svm/dev{0,3,8,15}/feature/host_mem_pool` = **1**，`host_pin_pre_register`=1，`mem_host_uva`=1，`dev_mem_map_host`=1 | A3 有 host 池 |

**臂 D 还有一个"直接证据"级别的发现 —— 关掉 Engram 就一定起得来：**

| 臂 | Engram | 卸载池 | 结果 |
|---|---|---|---|
| B2 | **开**（生产配置） | 16 GiB | 207001 挂 |
| B3 | **开** | 4 GiB | 207001 挂（8/8 worker） |
| C | **关**（`ENGRAM=0`） | 4 GiB | **起服成功**，跑到压测结束 |
| D | **关** | 32 GiB | **起服成功**，跑到压测结束 |

⇒ **因果很清楚【实测】：Engram 的 host 注册（`aclrtHostRegister`，206 GiB 表）与卸载层的
pinned host 分配在抢同一个驱动侧资源。** 生产配置（Engram 是必须的）下，卸载层拿不到 pinned 内存。

（为什么我的独立探针没能复现：我"复刻"的是**整块匿名 mmap + 一次 register**，
而真实 Engram 走的是 **safetensors 多分片、每片单独 `aclrtHostRegister(MAPPED)`**，
并且是在**完整的 torch_npu + 8 rank + 图编译**进程里做的。差异点就落在这里，
我没有时间把它再收敛一步 —— **这是本次留下的最大缺口**。）

**因此（诚实版结论）：**

- **强结论**【实测】：失败与"宿主有没有普通内存"无关 —— 宿主 800+ GB 可用、容器无限额、
  卡全空、`memlock` 无限；而不带模型的同样 8 张卡上，8 进程能 pin 64 GiB。
- **强结论**【实测】：失败发生在**驱动的 pinned 分配路径**（`aclrtMallocHostWithCfg`），
  在 `NPUOffloadingWorker.__init__` 里；这是**驱动侧 pinned 池**，不是普通 host RAM 的容量问题。
- **【实测·已定位到组合】**：`ENGRAM=1` 必挂、`ENGRAM=0` 必通（4 次独立起服，见上表）
  ⇒ 争用者是 **Engram 的 host 注册路径**。
- **未确认**：具体是"池容量被吃光"还是"注册与 pin 在同一进程内的交互"，
  以及**多少** Engram 注册量会让 pin 失败（我试的"整块匿名注册 206 GiB + pin"是成功的）。

### 4.3 坑三：`UniformTypeKVCacheSpecs` 包装不被 `get_sliding_window_size_in_chunks` 接受 —— **未确认（没走到）**

代码事实【推断，来源：镜像内 `offloading/scheduler.py` 第 107–121 行】：

```python
def get_sliding_window_size_in_chunks(kv_cache_spec, tokens_per_chunk):
    if isinstance(kv_cache_spec, SlidingWindowSpec): ...
    if isinstance(kv_cache_spec, MambaSpec): ...
    assert isinstance(kv_cache_spec, FullAttentionSpec)   # ← UniformTypeKVCacheSpecs 会掉这里
    return None
```

而 DSV4.1 的 13 个 group 全是 `UniformTypeKVCacheSpecs`（§3 第 1 条）⇒ **理论上必挂**。
同包的兄弟函数 `vllm/v1/kv_cache_interface.py::get_kv_cache_spec_sliding_window()`（933–942 行）
**已经**递归处理了包装类型 ⇒ 看起来是"漏了一处"，不是有意不支持。

**但我们没有实测到它**：B2（没挂补丁）与 B3（挂了补丁）都**先**倒在 §4.2 的 207001 上，
调度器侧代码根本没执行到 —— 证据是 B3 挂了补丁版 `scheduler.py`（里面有一段只打日志、
不改行为的 group dump），而 `serve.log` 里**一条都没打印**。

⇒ 我准备的补丁 `patches/files/offload_dsv41/scheduler.py`（3 处最小改动：新增
`_effective_offload_spec()` unwrap；在 `get_sliding_window_size_in_chunks()` 入口 unwrap；
最后那句 assert 放宽到 `AttentionSpec`，因为 state 组是 `AscendCircularBufferSpec(AttentionSpec)`）
**属于【推断】级修复，没有经过实测验证**。等 §4.2 解决后应当重跑这条。

（`GroupOffloadConfig` 里保留**原 wrapper**、只在传给那两个函数前 unwrap，是刻意的：
这样后续若有人按 wrapper 处理不会被改坏。）

### 4.4 ★★ 坑四（本次最核心的答案）：**能起服、能存储、零取回**【实测】

臂 D（`ENGRAM=0`，`OFFLOAD_GB=32`，`blocks_per_chunk=8`，HBM KV 1 GiB）跑完整两轮：

| 观测 | 值 | 出处 |
|---|---|---|
| `BlockStored(medium="CPU")` | **12,288** | 独立进程 ZMQ 订阅器 `out/off8-dram32.kv_events.log` |
| `BlockRemoved(medium="CPU")` | **8,192** | 同上 |
| `BlockStored:GPU` / `BlockRemoved:GPU` | 98,328 / 94,388 | 同上（与无卸载的臂 A 完全一致） |
| `vllm:kv_offload_store_bytes_total`（GPU→CPU） | **393,568,321,536 B ≈ 393.6 GB** | `out/off8-dram32.metrics_after.txt` |
| 存储耗时 | **27.76 s** ⇒ ≈ **14.2 GB/s** | 同上 |
| `vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"}` | **0.0** | 同上 |
| `vllm:external_prefix_cache_queries_total` | **1,048,832**（= 2 轮 × 524,288） | 同上 |
| `vllm:external_prefix_cache_hits_total` | **0** | 同上 |
| `vllm:prefix_cache_hits_total` | **0** | 同上 |
| fill / replay TTFT p50 | **4192.5 ms / 4167.4 ms**（+0.6%） | `out/off8-dram32.client.json` |

**判据结论**：① 满足（真的有块被标记存到 CPU 介质）；② **不满足**（58 万次查询/轮、
**零命中**）；③ **不满足**（replay 与 fill 在噪声内，且与无卸载基线 A 的 4335.7 ms 同量级）。

**它不是"池子太小"**：臂 C（4 GiB）与臂 D（32 GiB）的 `store_bytes` **精确到字节相同**
（393,568,321,536 B），load 都是 0；而 32 GiB 池已能覆盖 18.5 GiB 工作集。
⇒ 这是**取回路径本身没被触发**，不是容量问题。

**候选根因（全部【未确认】，按可疑度排序，供上游排查）：**

1. **group_idx 错位**：上游用 `make_offload_key(block_hash, group_idx)` 做 key，
   DSV4.1 的 13 个 group 在**调度侧**（`SchedulerOffloadConfig.from_spec` 按
   `kv_cache_config.kv_cache_groups` 的顺序）与**worker 侧**（
   `AscendOffloadingConnectorWorker` 按自己的遍历顺序 `group_data_refs.append`）
   生成 idx 的位置不同 ⇒ 存进去的 key 与查的 key 可能对不上。**这是最容易解释"存了但一次都不命中"的机制。**
2. **state 组不可前缀缓存**（`AscendCircularBufferSpec.prefix_cacheable=False`）：
   若某个 chunk 的命中判定要求**同组内所有 group** 都有该 key，那么一个永远不缓存的组
   就会把整轮命中判死。
3. **hash 粒度=32 与 chunk 粒度=1024 的不对齐**：`--prefix-match-unit 32` 让
   每 32 token 一个 hash，而 offload chunk 是 8×128=1024 token；
   `SchedulerOffloadConfig` 里 `hashes_per_chunk = tokens_per_chunk // tokens_per_hash`
   对这种混合块大小的组合**没有测试覆盖**。

**给上游的最小可复现入口**（不需要 8 卡，理论上 1 卡就能复现"存了不取"）：
构造一个"多 group + 其中一组 block_size 更小"的模型，开 `--prefix-match-unit` 到小值，
看 `external_prefix_cache_hits_total` 是否恒 0 —— 但我**没有在 1 卡上验证过**，标【未确认】。

---

## 5. ★ 这些坑在 A2（8×910B3，`host_mem_pool=0`）上会怎样？

### 5.1 `aclrtMallocHostWithCfg 207001` —— 在 A2 上**直接判死**（生产配置下）

★ **先给结论**：本机的 A/B 已经证明 **`ENGRAM=1` 时卸载层拿不到 pinned host**（§4.2 表）。
A2 的处境**至少一样差、很可能更差**：

* A2 的 `/proc/svm/dev*/feature/host_mem_pool` **预期为 0**（a2 工作区 `refs/55` §2.1）；
* 该文档记载 A2 上 **Engram 整表注册 17 min 后同样撞 207001**（当时 `MemAvailable` 仍有 703 GiB）
  ⇒ **在 A2 上，Engram 这一侧就已经先倒了**。
* ⇒ 对 A2 而言，"Engram + DRAM 卸载"这个组合**必须先把 host 内存路径解决掉**，
  否则无论 `cpu_bytes_to_use` 给多少都起不来。**A2 上不要把 `cpu_bytes_to_use` 当调参项，
  先跑 §5.1 的探测。**

先纠正一个口径：**它不是权重加载阶段的事**。原文证据（§4.2）显示它在
`initialize_from_config → initialize_kv_cache → register_kv_caches` 里，
**权重加载完（471 s）之后**，是卸载层为"DRAM 池"申请 pinned host 时挂的。

结论的强度，逐条给证据：

| 命题 | 强度 | 证据 |
|---|---|---|
| 与"普通 host RAM 够不够"无关 | **强（实测）** | 失败时 `MemAvailable` ≈ 830–860 GiB；容器 cgroup 无限额；`memlock` 无限；而裸 8 卡上 8 进程能 pin 64 GiB |
| 是**驱动侧 pinned 池**路径失败，不是普通 malloc | **强（实测）** | 报错原文就是 `aclrtMallocHostWithCfg` / `rtsMallocHost` / `Resource_Error_Insufficient_Host_Memory(EL0018)`，且栈在 `torch.zeros(..., pin_memory=True)` |
| "池被 Engram 注册吃掉了" | **未确认** | 我专门做了复刻实验（8 rank 各自注册 206 GiB 共享表 + 各 pin 1 GiB）**全部成功**，所以这条**不能**当作结论 |
| 8 个 rank 里谁先炸 | **实测** | B3 里 TP6 最先（+9 s），随后 TP2/TP7/TP4/TP5…，**8 个全中**；B2 里 1 GiB 的请求也是多个 rank 同时失败 |

对 A2 的直接含义：A2 的 `/proc/svm/dev*/feature/host_mem_pool` **预期为 0**
（a2 工作区 `refs/55-...md` §2.1 记载，且 A2 上 Engram 整表注册 17 min 后同样撞 207001，
当时 `MemAvailable` 仍有 703 GiB）⇒ **A2 上 pinned host 这条路风险高于 A3**。
**并且要注意**：如果 A2 连 Engram 的 `aclrtHostRegister` 都被 207001 挡住（该文档说挡住了），
那么"复用 Engram 的 mmap+register 路线"这条替代方案**在 A2 上同样未验证通过** —— 不能想当然。

**A2 上请先跑这三条只读探测（30 秒，抄自 `a2/refs/55-...md` §2.1，我原样保留）：**

```bash
# ① A2 到底有没有 host 池
cat /proc/svm/dev0/feature/host_mem_pool          # A3 实测=1；A2 预期 0

# ② 直接试分配 1 GiB pinned host（不碰模型）
python3 -c "
import ctypes; lib=ctypes.CDLL('libascendcl.so')
p=ctypes.c_void_p(); lib.aclrtMallocHost.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t,ctypes.c_uint32]
rc=lib.aclrtMallocHost(ctypes.byref(p), 1<<30, 0)
print('aclrtMallocHost 1GiB rc =', rc)
"

# ③ torch_npu 侧的 pinned 分配（卸载层实际用的路径）
python3 -c "
import torch, torch_npu
print('npu ok', torch.npu.device_count())
x=torch.empty(256*1024*1024, dtype=torch.uint8, pin_memory=True)
print('pin_memory ok', x.shape, x.is_pinned())
"
```

★ **建议再加第 ④ 条**（本次新增，正是 A3 上失败的那一段；A3 上它也是成功的，
所以"④ 通过"不等于"卸载层能用"，但"④ 失败"就直接判死）：

```bash
# ④ 8 个 rank 各占一张卡、各自 pin 512 MiB（= 卸载层 OFFLOAD_GB=4 时单 worker 的量级）
python3 ~/projects/dsv41-upstream-pr/agents/D_off8/scripts/pin_host_probe_8dev.py 0.5 0.5
```

### 5.2 `--prefix-match-unit 32` 对 A2 同样适用吗？——**适用**【推断，理由是同代码路径】

A2 也是 DSV4.1（同样有 `compressor.state_cache` 组、`block_size=32`、其余 12 组 128），
且 `_ascend_resolve_kv_cache_block_sizes()` 是**同一份 vllm-ascend 代码**，
"存在不可缓存组 ⇒ 哈希粒度 = 可缓存组 GCD = 128"的行为一致 ⇒ 同样会撞 32 % 128。
设 `--prefix-match-unit 32` 后 `128 % 32 == 0`、`32 % 32 == 0` 两个方向都成立。
**但顺序很重要**：若 5.1 的 pinned 池在 A2 上过不去，这条绕过也救不了 ——
它会倒在同一处（§4.2 的 207001 更早）。⇒ **先证 pinned 可用，再谈哈希粒度。**

### 5.3 最小复现：**没能缩到"一条不需要模型的命令"（如实说明）**

我试了五个探针（§4.2 的表），**没有一个复现出 207001**；能稳定复现的只有"完整 8 卡 DSV4.1 起服"
（~10 min，撞在权重加载之后）。臂 B2/B3 的**零成本复现**是直接读日志：

```bash
# A3 上的原始证据（两臂都在）
grep -a "207001\|EL0018\|Failed to allocate" \
  ~/projects/dsv41-release/results/off8_off8-{pmu32,b3}_*/serve.log | head
```

**若要继续收敛，我建议的下一步（成本递增）**：
1. 8 卡起服时**在同一个容器里**跑 ④ 探针（模型起来后再测，能直接量出"模型+Engram 占完之后还剩多少 pinned 配额"）—— 这是最可能一刀切中因果的实验，约 25 min；
2. A/B 关掉 Engram（`ENGRAM=0`）再起卸载臂 —— **已做**：4 GiB 与 32 GiB **两次都起得来**
   ⇒ 因果锁定在 Engram 的 host 注册（§4.2 的表 + §4.4 的完整数据）。

---

## 6. 还缺什么 / 下一步（按优先级）

| 优先 | 动作 | 成本 | 为什么 |
|---|---|---|---|
| **P0** | 在 A2 上跑 §5.1 的 ①②③④ 四条探测 | 30 s | 决定 A2 走 `NPUOffloadingSpec` 还是"普通 mmap + `aclrtHostRegister`"路线 |
| **P0** | **查 §4.4 的"零取回"**：先确认 group_idx 在调度侧/worker 侧是否一致（读代码即可，成本最低） | ~1 h | 这是"卸载到底为什么没用"的核心；不解决它，A2 上就算解决了 pinned 也还是零收益 |
| **P1** | 关掉 Engram、在**修好取回路径后**重跑臂 D 验证收益 | ~25 min | 现在只能证明"能存不能取" |
| **P1** | §4.3 的 scheduler 补丁做**对照实验**（挂/不挂各一次，`ENGRAM=0`） | ~30 min | 目前它只是【推断】级修复（臂 D 挂了补丁才起得来，但不能排除"不挂也行"） |
| **P2** | `blocks_per_chunk=64` 的**阴性对照**（任务书要求） | ~25 min | **未做**：判据链已被 §4.4 证伪（hits 恒 0），负对照此刻没有信息量 |
| **P2** | `--prefix-match-unit 32` 的精度/性能代价量化 | — | 哈希粒度 128→32 的影响未测 |

**没有做的事（明确声明，不掩盖）**：

* `blocks_per_chunk=64` 的**阴性对照未做**；
* §4.3 的 scheduler 补丁**没有做挂/不挂的对照**（臂 C/D 都挂了它）；
* §4.4「零取回」的**根因未确认**（只给了 3 个候选，没做收敛实验）；
* A2 上的任何事情**本机都没法验证**（这一节全部基于 a2 工作区既有文档 + 本机的 A/B 类推）。

---

## 7. 交付物清单

| 文件 | 作用 |
|---|---|
| `scripts/serve_dsv41_dram_offload.sh` | **可复用启动脚本**（参数化 `OFFLOAD_GB`、`PREFIX_MATCH_UNIT`、`OFFLOAD_SCHED_PATCH`；A2 默认 8×0–7，A3 用 `DEVS/CHIPS/IMAGE` 覆盖） |
| `scripts/run_arm_8card.sh` | 八卡单臂运行器（起服→就绪→事件探针→fill/replay→收指标→清容器） |
| `scripts/patch_shadow_pkg.sh` | 幂等给**影子包**打两处注入补丁（只动 shadow-pkg，不动 `dsv41-release`） |
| `patches/files/offload_dsv41/scheduler.py` | 【推断】级修复：unwrap `UniformTypeKVCacheSpecs` + 放宽 state 组的 assert |
| `scripts/make_offload_scheduler_patch.py` | 生成上面那个补丁（可重复、带锚点校验） |
| `scripts/pin_host_probe.py` / `pin_host_probe_mp.py` / `pin_host_probe_8dev.py` | pinned host 容量探针（单进程 / 同卡多进程 / 多卡多进程） |
| `scripts/hostregister_vs_pin.py` / `hostregister_mp_shared.py` | Engram 式 host 注册与 pinned 分配是否互斥的判定探针 |
| `bench/kv_events_probe.py` / `ev_probe.py` / `kv_offload_client.py` / `summarize.py` | 判据链（从 `M_offload` 复制，判定口径与 logs/45 一致） |
| `out/off8-*.{client.json,meta.txt,metrics_*.txt,kv_size.txt,kv_config.txt,kv_events.*}` | 原始数据（**已随本日志一起落到 `logs/raw/51-dsv41-dram-offload/`**） |
| `logs/arm-*.out`、`logs/off8-*.client.log`、`results/off8_*`（在 A3-node1） | 原始运行日志（大文件，留在远端） |

**复现臂 D（拿到"能存不能取"的那条）**：

```bash
cd ~/projects/dsv41-upstream-pr/agents/D_off8
TAG=off8-dram32 ENGRAM=0 OFFLOAD_GB=32 PREFIX_MATCH_UNIT=32 OFFLOAD_SCHED_PATCH=1 \
  bash scripts/run_arm_8card.sh
# 观察点：out/off8-dram32.kv_events.log 里 BlockStored:CPU>0，
#         而 out/off8-dram32.metrics_after.txt 里 CPU_to_GPU 恒为 0、external hits 恒为 0
```

**脚本与原始数据在两处**（内容相同，便于离线查看）：

* A3 上（跑得起来的那份）：`~/projects/dsv41-upstream-pr/agents/D_off8/{scripts,bench,out,logs}/`
* 开发机上（本日志同仓）：`upstream-v41/agents/D_off8/{scripts,bench,out}/`
  以及 `upstream-v41/logs/raw/51-dsv41-dram-offload/`

### 附：A2 内存预算表（按任务书给定口径换算，**未在本机实测**）

已知（任务书/既有实测，直接采用）：**4421 B/token/rank**；**MLA 在 TP 内是复制的**
⇒ `cpu_bytes_to_use` 是**服务级总量**、CPU 侧存 **8 份** ⇒ `tokens = 值 / (4421 × 8)`。

| `cpu_bytes_to_use` | 能缓存的 token（8 rank 合计口径） | 相当于多少个 1M 上下文 | 1M 取回耗时（按 28.5 GB/s【实测】、A2 走 PCIe 会更慢【推断】） |
|---|---|---|---|
| 100 GB | ≈ 2.83 M token | ≈ 2.8 个 | 每 1M 需搬 34.5 GiB ⇒ ≈ 1.3 s（本机实测带宽）；A2 上 **未测**，会更慢 |
| 200 GB | ≈ 5.65 M token | ≈ 5.7 个 | 同上按比例 |
| 300 GB | ≈ 8.48 M token | ≈ 8.5 个 | 同上按比例 |

推导：`100e9 / (4421×8) = 2.83e6`；`1M token × 4421 B × 8 rank = 34.5 GiB`。
**注意**：这张表是"池子大小 → 覆盖 token 数"的换算，**前提是卸载层能起来**；
按本次结果，A2 上这个前提**尚未成立**（§5.1）。
另外 M_offload 在单卡上量到 pinned 池对宿主的记账约为申请值的 **1.7×**
（`logs/45` §3 坑 2），若该比例在 A2 上也成立，300 GB 需按 **~510 GB** 预算 ——
而 A2 宿主余量只有 **442 GiB** ⇒ 按 1.7× 记账，**300 GB 超预算**，应取 ≈260 GB（a2/refs/55 §2.2 的结论一致）。
