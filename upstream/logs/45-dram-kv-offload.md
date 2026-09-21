# 45 · 单卡把 DRAM 当 KV cache 卸载层（实测报告）

**日期**：2026-09-21 19:55–21:00　**执行**：子代理 `M_offload`　**机器**：A3（A3-node1）
**槽位**：c0 = die3（DRAM 路径）/ c1 = die6（Mooncake 路径，host 网络容器 `prbench-moo`）
**交付**：`agents/M_offload/`（脚本 + 使用文档 + 原始日志）　**结论标记**：【实测】/【推断】/【未确认】

---

## 0. 一句话结论

**【实测】两条路都打通了**：DRAM 卸载层（`OffloadingConnector` + `NPUOffloadingSpec`）在 16×4096 token
的 workload 上把"重算"变成"取回"，replay TTFT **87.7 ms → 38.4 ms（2.3×）**，
D2H/H2D 各 **~29 GB/s**，`BlockStored(medium="CPU")` = 64、HBM 外部层命中 = **65,536/65,536 token（100%）**；
Mooncake KV pool（`AscendStoreConnector` + 已在跑的 master:50088）同样通，p50 replay 25.3 ms，
但 16 个请求里有 4 个退化，整体命中 75%。**推荐：单卡自用选纯 DRAM 层**（最稳、延迟确定、零外部依赖）；
需要跨实例复用再上 Mooncake。

**一个必须知道的坑**：本镜像（vllm 0.27.1 + vllm-ascend `e43cf1e9f`）**不能**只写
`spec_name: CPUOffloadingSpec`（仓库文档的写法），会撞上游 CUDA 门禁；必须
`spec_name=NPUOffloadingSpec` + `spec_module_path=...native.npu`。另外 **DRAM 层不放宽
"HBM KV 必须装下一个满长请求"** 这条限制。

---

## 1. 环境与复现入口

| 项 | 值 |
|---|---|
| 镜像 | `quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3` |
| vLLM / vllm-ascend | `0.27.1+empty` @ `/vllm-workspace/vllm`；vllm-ascend `e43cf1e9f` |
| 单卡模型 | `Qwen3-1.7B`（28 层全注意力、8 KV head、head_dim 128、bf16 ⇒ **114,688 B/token**） |
| HBM KV 上限（除注明外） | `--kv-cache-memory-bytes 1 GiB` ⇒ `GPU KV cache size: 9,344 tokens` |
| 起服务 | `agents/M_offload/serve_with_dram_offload.sh`（dram / mooncake 双后端，带自检） |
| 对照压测 | `agents/M_offload/bench/run_arm.sh`、`bench/run_store_arm.sh`（单次锁内：起服务→压测→收指标→收尾） |
| 原始日志/JSON | 远端 `~/projects/dsv41-upstream-pr/agents/M_offload/out/`（含 `SUMMARY.md` 汇总表） |

复现一条臂（约 2 分钟）：

```bash
bash ~/projects/dsv41-upstream-pr/tools/a3_chip.sh c0 --name repro -- \
  env TAG=repro-dram8g OFFLOAD_GB=8 PROMPTS=16 PROMPT_TOKENS=4096 ROUNDS=2 RESET_BETWEEN=1 \
  bash /work/agents/M_offload/bench/run_arm.sh
```

---

## 2. 判据（"卸载真的发生了"怎么证明）

1. **KV 事件（仓库单卡 e2e 的原判据）**：ZMQ 订阅 `--kv-events-config`，
   数 `BlockStored(medium="CPU")`；实测 DRAM 臂 **64 条**、Mooncake 臂 **802 条 `BlockStored:cpu`**。
   （实现细节：订阅器必须是**独立进程**，见 §5.4。）
2. **外部层命中计数**：`vllm:external_prefix_cache_hits_total`。
   两轮 workload 时 hits/queries ≈ 0.5 表示 **replay 轮 100% 命中**；基线臂该计数恒为 0。
3. **搬运量与带宽**：`vllm:kv_offload_store_bytes_total` / `load_bytes_total` 与
   `store_time_total` / `load_time_total`（7.516 GB / 各 0.2577 s、0.2636 s）。
4. **延迟对照**：replay 轮先 `POST /reset_prefix_cache`（`VLLM_SERVER_DEV_MODE=1`）清空 HBM 前缀缓存，
   基线只能重算，卸载臂应显著更快。
5. **DRAM 记账**：客户端在容器内快照 `/proc/<pid>`、cgroup、`/proc/meminfo`，
   另有 `bench/alloc_probe.sh` 做"起服务前后"的宿主机 `free` 对照。

---

## 3. 路径 A（DRAM 层）实测数字

**主对照**：16 个互不相同的 4096-token 请求、单并发、`max_tokens=1`、
HBM KV=1 GiB（9,344 token）⇒ 工作集 65,536 token ≈ HBM 的 7 倍，必然驱逐。

| 指标 | 基线（无卸载） | DRAM 8 GiB | 差值 |
|---|---|---|---|
| fill 轮 TTFT（均值） | 84.4 ms | 90.4 ms | +7%（写 DRAM 的开销） |
| **replay 轮 TTFT（均值）** | **87.7 ms** | **38.4 ms** | **2.28×** |
| replay p50 | 87.8 ms | 38.1 ms | 2.30× |
| fill / replay tok/s | 11.8 / 11.4 | 11.0 / 25.9 | replay 2.3× |
| `external_prefix_cache_hits` | 0 | **65,536 token** | — |
| `BlockStored(medium="CPU")` | — | **64 事件** | — |
| 写 / 读 DRAM | — | 7.0 GiB / 7.0 GiB | 29.2 / 28.5 GB/s |
| 有效 KV 容量 | 9,344 token | **9,344 + 74,898 = 84,242 token（9.0×）** | — |

高并发（16×2048 token、并发 8、生成长度 64、**不 reset**、自然驱逐）：

| 指标 | 基线 | DRAM 8 GiB |
|---|---|---|
| replay TTFT（均值 / p50） | 523 / 632 ms | 414 / 519 ms（−21%） |
| replay 吞吐 | 458.8 tok/s | **528.4 tok/s（+15%）** |
| fill 吞吐 | 454.1 tok/s | 424.9 tok/s（−6%，写开销） |

小 workload 交叉验证（4×1024 / 6×2048 / 16×2048 等）见 `out/SUMMARY.md`：
命中率恒为 100%（hits = 全部 replay token），H2D 带宽稳定在 **25–30 GB/s**。

---

## 4. 路径 D（Mooncake KV pool）实测数字

配置：`AscendStoreConnector` + `backend=mooncake`，`MOONCAKE_CONFIG_PATH=conf/mooncake_single.json`
（`metadata_server: P2PHANDSHAKE`、`protocol: ascend`、`master_server_address: 127.0.0.1:50088`、
`global_segment_size: 8GB`），**只连已在跑的 `mooncake-master` 容器，没有重启它**。

| 指标 | 基线（§3） | Mooncake 8 GiB | Mooncake 24 GiB |
|---|---|---|---|
| fill TTFT（16×4096） | 84.4 ms | 100.7 ms（+19%） | 99.9 ms |
| replay TTFT 均值 | 87.7 ms | 44.0 ms | 54.4 ms |
| replay p50 | 87.8 ms | **25.3 ms** | 46.2 ms |
| `external_prefix_cache_hits` | 0 | 49,141 / 65,536（75%） | 40,952 / 65,536（62%） |
| `BlockStored:cpu` 事件 | — | 802 | 866 |

小 workload（4×2048）时几乎全命中（`external_prefix_cache_hits` = 8,188/8,192 token ≈ 100%）：
replay TTFT **17.5 ms**（vs 基线 41.7 ms、DRAM 路径 20.8 ms），
说明链路本身很快，**大 workload 的退化来自池容量/分配**，不是协议开销。

【实测】8 GiB segment ≈ 7.0 GiB 工作集 ⇒ 触发驱逐/抖动；把 segment 提到 24 GiB 后命中**反而更差**。
【未确认】原因（首次触页、分配局部性、`batch_alloc` 策略都只是猜测），没有继续深挖；
实用建议：**给池留 ≥2× 工作集，并按 p50 评估延迟**。

---

## 5. 关键发现（都会影响"实际可用"）

### 5.1 本镜像必须走树外 spec（否则路径 A 直接起不来）

容器内 `vllm_ascend` 里 **`grep -rn OffloadingSpecFactory` 零命中**：
`CPUOffloadingSpec` 仍指向上游 `vllm/v1/kv_offload/cpu/spec.py`，其 `get_worker()` 有平台门禁。
实测报错（原始日志 `out/a0-doc-spec-fail.server.log`）：

```
Exception: CPU Offloading is currently only supported on CUDA-alike and XPU GPUs
```

正解（`OffloadingSpecFactory.get_spec_cls` 的树外扩展点）：

```json
"kv_connector_extra_config": {
  "cpu_bytes_to_use": 8589934592,
  "blocks_per_chunk": 8,
  "spec_name": "NPUOffloadingSpec",
  "spec_module_path": "vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.npu"
}
```

`vllm-ascend` main（≥2026-09-21，`vllm_ascend/distributed/kv_transfer/__init__.py` 的
pop + `register_spec` 循环）已修这个问题 —— 也就是说**仓库文档的写法只对新版成立**。
【实测】本镜像上 `NPUTieringOffloadingSpec` 不存在 ⇒ **路径 B（多级 tier）在这一版不可用**。

### 5.2 `blocks_per_chunk` 太大 = 完全不卸载（静默退化）

16×4096 token、block_size=128 下实测：

| `blocks_per_chunk` | chunk 大小 | `BlockStored(CPU)` | replay TTFT |
|---|---|---|---|
| 8 | 1024 token | 64 | 38.4 ms（2.3×） |
| 32 | 4096 token | 16 | 39.6 ms（2.2×） |
| **64** | **8192 token（> 单请求长度）** | **0** | **85.2 ms（= 重算，0 收益）** |

⇒ 默认取 8 是安全的；规则：`blocks_per_chunk × block_size ≲ 典型请求长度`。

### 5.3 DRAM 层**不放宽**"HBM 要能装下满长请求"

`--kv-cache-memory-bytes 1 GiB` + `--max-model-len 32768` 起不来，报错原文：

```
ValueError: To serve at least one request with the model's max seq len (32768),
(3.5 GiB KV cache is needed, which is larger than the available KV cache memory (1.0 GiB).
Based on the available memory, the estimated maximum model length is 9344.
```

⇒ DRAM 层扩的是**累计/并发容量**（很多会话历史可以躺在 DRAM），不是单请求上下文上限。
要用长上下文，HBM 侧仍要 `max_model_len × kv_per_token` 的 KV。脚本已把这条做成启动前校验。

### 5.4 KV 事件通道：要在**独立进程**里订阅，且先订阅再压测

【实测】把 `import vllm`（为了拿 `BlockStored` 类型解码）放进压测客户端进程，
客户端会**卡死**（主线程 6 分钟无输出，`/proc/<pid>/wchan`=0，3 个子线程 nanosleep），
一次请求都没发出去。改成**独立进程** `bench/kv_events_probe.py`（只 import zmq + msgspec，
用 msgspec 泛型解码，vLLM 的 struct 带 `tag=True` 会编码成 map）后稳定工作。
另一个坑：短 bench 只有 0.5 s，监听线程里 import vLLM 要几秒 ⇒ 事件恒为 0；必须"先订阅、再压测"。

### 5.5 DRAM 记账：池是 pinned host memory，**不进 cgroup，也不进 VmRSS**

`bench/alloc_probe.sh`（起服务前后快照，同一台机）：

| | 宿主机 `used` | 容器 cgroup v1 | EngineCore VmRSS |
|---|---|---|---|
| 无卸载启动 | +6.2 GB（模型+KV） | +6.2 GB | 5,465 MB |
| `cpu_bytes_to_use=8 GiB` 启动 | **+20.9 GB**（比基线多 ~14.6 GB） | +0.07 GB | 5,539 MB（+74 MB） |

⇒ 【实测】① 池按 1.7× 于请求值吃宿主机内存（对齐/驱动记账）；
② **容器内存 limit 既不会拦住它、也不会给它记账** —— 必须用宿主机 `free/MemAvailable` 做容量规划。

### 5.6 基础设施两条硬约束（本机特有）

* **一个 die 只能被一个容器持有**：第二个容器（即使挂了同一个 `/dev/davinciN`）里
  `torch.npu.device_count()`=0、报 `Invalid device id`；停掉 slot 容器后立刻恢复（count=1）。
  ⇒ Mooncake 需要 host 网络，就必然要"停 slot 容器 → 起 host-net 容器 → 用完还原"，
  已封装成 `tools/moo_chip.sh`（带 c1 锁 + trap 还原）。
* **bridge 容器够不到 host 网络里的服务**：docker 网桥网关（`172.17.0.1:50088`）与
  宿主机内网地址上的同一端口**都是 `EHOSTUNREACH`**；本机所有 mooncake 相关容器
  （`dsv4-offload-serve-*`、`dspark-offload-*`、`mc-lite-*`）都是 `--network host`，
  与实测一致。⇒ **要用 Mooncake，容器必须 `--network host`。**

---

## 6. 有效容量（怎么算、算出来多少）

```
kv_per_token = 2 × num_kv_heads × head_dim × dtype_bytes × full_attn_layers
             = 2 × 8 × 128 × 2 × 28 = 114,688 B/token        # Qwen3-1.7B
DRAM tier token 容量 = cpu_bytes_to_use / kv_per_token
```

| | HBM KV | DRAM tier | 合计（有效） |
|---|---|---|---|
| 配置 | 1 GiB | 8 GiB | 9 GiB |
| token 容量 | 9,344 | 74,898 | **84,242（9.0×）** |
| 8k 上下文 | ~1.1 个 | ~9.1 个 | ~10.2 个 |

注意：HBM 侧另有"必须能装下 1 个满长请求"的硬门槛（§5.3），
所以真实部署里建议 HBM KV 至少 `max_model_len × kv_per_token`，DRAM 层再按"会话历史总量"配。

---

## 7. 推荐与未完成项

**推荐（单卡实际场景）**：

1. 默认用**纯 DRAM 层**：`serve_with_dram_offload.sh --offload-gb <N> --blocks-per-chunk 8`，
   配 `--kv-cache-memory-bytes ≥ max_model_len × kv_per_token`。
2. 只有需要**跨实例复用 KV**（PD 分离、多副本、重启后仍在）时才上 Mooncake，
   且 segment ≥ 2× 工作集、用 host 网络容器、按 p50 评估。
3. 容量规划用**宿主机内存**：`需求 ≈ 1.7 × cpu_bytes_to_use`（实测系数），别只看容器 limit。

**未完成 / 留给后来者**：

| 项 | 状态 |
|---|---|
| `TieringOffloadingSpec`（DRAM+SSD/3FS 多级） | 【实测】本镜像**没有** ascend 版实现，需换 main/新镜像 |
| Mooncake 大 workload 命中退化（62–75%） | 【未确认】根因未定位；建议先验证 `global_segment_size` 与 `batch_alloc`/lease 行为 |
| 混合线性注意力模型的卸载 | **跑通服务但卸载 0 生效**，见 §5.7（`Qwen3.5-27B-w8a8-mtp`） |
| 多轮长会话的真实 trace（会话间隔 > lease TTL） | 未跑；`default_kv_lease_ttl=11000 ms` 是否够用需按业务验证 |
| 与 27B 级模型联合的端到端（吞吐/显存占用） | 未跑（时间预算用在小模型把机制打穿上） |

---

## 5.7 混合线性注意力模型（27B 真实场景那一轮）：服务能起，**卸载不生效**

【实测】`Qwen3.5-27B-w8a8-mtp`（W8A8，24 层里 6 层 full attention + 18 层线性注意力）在单 die 上
按 DRAM 卸载配置能起来，但**一个 CPU chunk 都没存**，收益为 0：

```
TAG=q4-27b-hybrid  GPU_MEM_UTIL=0.9  KV_CACHE_BYTES=4GiB  MAX_MODEL_LEN=8192
                   OFFLOAD_GB=16  blocks_per_chunk=8  EXTRA_ARGS="--enable-prefix-caching --max-num-seqs 16"
fill   TTFT 373.1 ms（8×2048 token）
replay TTFT 372.4 ms   ← 与 fill 持平 = 纯重算
kv_events: BlockStored:GPU=64, BlockStored:CPU=0, AllBlocksCleared=1
metrics:   external_prefix_cache_hits_total=0, kv_offload_size_count{CPU_to_GPU}=0
```

起服务路上还有两道**必需**的门（否则直接报错退出，原始日志 `out/q1/q2/q3`）：

1. 不加 `--enable-prefix-caching` 时混合模型下前缀缓存自动为 False ⇒
   `AssertionError: tokens_per_block=1536 not divisible by tokens_per_hash=24576.
   Hybrid models (e.g. Mamba+Attention) need --enable-prefix-caching to align block sizes.`
2. 还要限 `--max-num-seqs`：4 GiB KV 下 mamba cache 只够 21 个块 ⇒
   `ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (21)`。

【未确认】为什么 0 个 CPU chunk：可能是混合 block pool 的驱逐路径不产生可卸载 chunk，
或 worker 侧 canonicalization 对 mamba 组为空。**结论**：这批 Qwen3.5/3.8 混合模型的"KV 卸载"
需要单独做适配，不能指望现成的 `OffloadingConnector` 路径。

---

## 8. 原始证据清单（远端 `agents/M_offload/out/`）

| 文件 | 内容 |
|---|---|
| `SUMMARY.md` | 全部 14 条臂的汇总表（脚本 `bench/summarize.py` 生成） |
| `b1-base-4k.client.json` / `b2-dram8g-4k.client.json` | 主对照两臂（TTFT/指标/DRAM 快照） |
| `c1-bpc8-events.*` / `c2-bpc32.*` / `c3-bpc64.*` | `blocks_per_chunk` 扫描（含事件 JSON） |
| `c1-bpc8-events.kv_events.json` | `BlockStored:CPU=64`、`BlockRemoved:GPU=882`、`AllBlocksCleared=1` |
| `d0-store-smoke.*` / `e1-store-4k.*` / `e2-store24g-4k.*` | Mooncake 路径（8 GiB / 24 GiB segment） |
| `r1-base-conc8.*` / `r2-dram8g-conc8.*` | 高并发自然驱逐对照 |
| `a0-doc-spec-fail.server.log` | "CPU Offloading ... CUDA-alike and XPU GPUs" 原始报错 |
| `srv-smoke.server.log` | 使用脚本自检/启动/推理的完整日志 |
| `alloc-probe.*.log`（见 tools 目录说明） | DRAM 记账探针输出 |

校验方式：`python3 bench/summarize.py` 重算汇总表；所有数字都来自 `out/*.json` 与 `out/*.server.log`，
没有手工填写的数。

---

## 9. 红线遵守情况

* 全程**没有**发 PR / issue / 评论；产物只在 `agents/M_offload/`（远端）与 `upstream-v41/agents|logs/`（本机）。
* 只用自己的槽位（c0/c1/c2 + 自建 `prbench-moo`，后者用 c1 锁保护、用完还原 `prbench-c1`）。
* **没有重启 / 修改** `mooncake-master`（只连 50088）；没有碰别人的容器与服务。
* 跨机文件走 coscli；临时文件在 `~/tmp/20260921/M_offload/`。
* 未动 `~/projects/dsv41-release/`。
