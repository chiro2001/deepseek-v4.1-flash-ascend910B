# 010 · 单卡「dummy 权重 + tiny config」复现 DSV4.1 的 KV 卸载「能存不能取」

**日期**：2026-09-21 23:45 – 2026-09-22 00:2x　**执行**：子代理 `L1_dummy`
**机器**：A3（A3-node1），**单张卡**（槽位 `c1` = Phy-ID 6，`prbench-c1` 容器内直接起服）
**产物**：`a2/agents/L1_dummy/`（脚本 + 补丁 + 原始数据）　**标记**：【实测】/【推断】/【未确认】

---

## 0. ★ 核心判据（一句话）

**【实测】能。单卡（TP1）+ dummy 权重 + 保留 40 层 KV 结构，完整复现了 8 卡上的
「能存不能取」**：`BlockStored(CPU)=704`、写 DRAM **2.87 GB**，
而 `CPU_to_GPU = 0`、`external_prefix_cache_hits = 0`、
replay TTFT 453.3 ms ≈ fill 469.3 ms（**零收益**）。

| 判据（沿用 `logs/001` 口径） | 8 卡真权重臂 D | **单卡 tiny-dummy 臂 `l1-d4k`** | 一致？ |
|---|---|---|---|
| ① `BlockStored(medium="CPU") > 0` | ✓ 12,288 | **✓ 704** | ✅ |
| ② `kv_offload_total_bytes{CPU_to_GPU} > 0` | ✗ **0** | ✗ **0** | ✅ |
| ③ `external_prefix_cache_hits` 增长 | ✗ **0** / 1,048,832 查询 | ✗ **0** / 131,328 查询 | ✅ |
| ④ replay TTFT ≪ fill TTFT | ✗ +0.6% | ✗ **−3.4%**（453.3 vs 469.3 ms） | ✅ |

⇒ **后续迭代成本从 22 min/臂 降到 ≈2 min/臂**（起服 70–90 s + 压测 15 s），且**只占 1 张卡**。
**8 卡只在最终确认时用一次**（L2）—— 这条决策路径**成立**。

**【实测】但"截断层数"那条路走不通**（任务书第 1 步的设想被代码硬拒），
本日志用的是**等价替代路线**：**保留 40 层结构、只压参数量**。见 §2。

---

## 1. 路线结论：为什么不是「40 层 → 8 层」

任务书设想的"把 `num_hidden_layers` 从 40 改小"，**在 vllm-ascend 里被三条写死的校验挡住**。
证据是**直接调用镜像内的真函数**（探针 `probe_truncation.py`，**不占卡**）：

```python
# vllm_ascend/core/deepseek_v41.py::plan_cache_slots()
if list(map(_layer_number, full))  != [2, 8, 14, 20]:   raise ValueError("V4.1 requires KV source layers 2, 8, 14, 20")
if list(map(_layer_number, state)) != [2, 8, 14]:       raise ValueError("V4.1 requires state source layers 2, 8, 14")
if list(map(_layer_number, swa))   != list(range(40)):  raise ValueError("V4.1 requires exactly 40 ordered SWA resources")
```

**SWA 资源是"每层一个"**（40 层 ⇒ 40 个 ⇒ 10 个 `swa` 组，每组 4 层）⇒ 层数一改，第三条必挂。
而这正是"13 个 group"的来源，改不动。

【实测】把按 N 层造出的 spec 字典喂给 `group_cache_specs()`：

| `num_hidden_layers` | 结果 |
|---:|---|
| **40** | **OK，12 个 group**（本次 tiny config 的形态） |
| 8 | ❌ `ValueError: V4.1 requires KV source layers 2, 8, 14, 20` |
| 4 | ❌ 同上 |

40 层的 group 清单（**第一手**，本次实测，`probe_truncation.py` 输出）：

| # | member spec | block | 成员数 | 说明 |
|---:|---|---:|---:|---|
| 0 | `DeepseekV41FullSpec`（+IndexerSpec 同组） | **128** | 8 | 4 个 long-KV 平面 + 4 个 indexer 平面 |
| 1 | `DeepseekV41CompressorStateSpec` | **32** | 3 | ★ 不可前缀缓存的小 block 组 |
| 2–11 | `DeepseekV41SWASpec`（window=128） | 128 | 4 × 10 | 40 个 SWA 资源轮转分 10 组 |
| （12） | `DeepseekV41DraftSWASpec` | 128 | 3 | **本次去掉**（`num_nextn_predict_layers=0`） |

> 与 `logs/001` §3 的 13 组清单**逐条一致**（少的第 12 组是 dspark，本臂故意去掉）。
> ⇒ **结构等价性成立**：`full` + `state`(block 32) + 10 个 `swa` 组全在，
> `OFFLOAD-SINGLE-CHIP-ASSESSMENT.md` §3.3 假设的"结构条件都还在"**得到实测支持**。

---

## 2. 替代路线：**保留结构，只压参数量**（本日志的方案）

`make_tiny_config.py` 从官方 `config.json` 出发，只改**与 KV 结构无关**的字段：

| 字段 | 原值 | tiny | 理由 |
|---|---:|---:|---|
| `n_routed_experts` | 384 | 8 | MoE 权重形状，不参与 KV |
| `moe_intermediate_size` | 2304 | 256 | 同上 |
| `num_experts_per_tok` | 6 | 2 | 同上 |
| `q_lora_rank` / `o_lora_rank` | 1280 / 1024 | 512 / 512 | 注意力投影宽度（**不动** `head_dim`/`num_key_value_heads`） |
| `max_position_embeddings` | 1048576 | 65536 | RoPE 表 1M×16 会吃 ~8.6 GB HBM；压到 64k ⇒ ~1 GB |
| `engram_layer_ids` / `engram_num_embeddings` | [1,14] / 3.8e8 行 | **[]** | 避开 Engram（207001 + dummy 下会爆） |
| `num_nextn_predict_layers` / `dspark_target_layer_ids` | 3 / [37,38,39] | **0 / []** | 去掉 MTP/dspark 组（⇒ 12 组而非 13 组） |

**结构字段逐字不变**（脚本内做强制自检，改了就直接拒绝写出）：
`num_hidden_layers=40`、`compress_ratios`、`kv_source_layer_ids`、`index_source_layer_ids`、
`candidate_source_layer_id`、`index_topk`/`index_n_heads`/`index_head_dim`、`head_dim=512`、
`num_key_value_heads=1`、`sliding_window=128`、`hc_mult`、`vocab_size`。

另：顶层 `quantization_config` 被删掉 —— **生产用的 `v41-w4a8-...` 目录顶层本来也没有它**
（量化信息在 `quant_model_description.json`），dummy 走 bf16 不需要 fp8/fp4 路径。

**产物**：`~/projects/dsv41-upstream-pr/agents/L1_dummy/models/model-tiny/`
（`config.json` + `tokenizer.json` + `tokenizer_config.json` + `configuration.json` + provenance，
共 **6.4 MB**；**dummy 不读 safetensors，所以没有软链任何大文件**）

---

## 3. 起服：单卡能起来，**不需要 `--prefix-match-unit` 之外的任何东西**

命令（在 `prbench-c1` 内，TP1，Phy-ID 6）：

```bash
bash ~/projects/dsv41-upstream-pr/tools/a3_chip.sh c1 --timeout 1800 --name l1-d4k -- \
  env TAG=l1-d4k OFFLOAD_GB=2 PREFIX_MATCH_UNIT=32 PROMPT_TOKENS=4096 PROMPTS=16 WOA_FIX=1 \
  bash /work/agents/L1_dummy/scripts/run_arm.sh
```

| 观测 | 值 | 等级 |
|---|---|---|
| `--load-format dummy` | 接受（无报错） | 【实测】 |
| **起服到 `/health` 用时** | **≈70–90 s**（8 卡真权重是 **1320 s**） | 【实测】 |
| `GPU KV cache size` | **22,719 tokens** | 【实测】 |
| `Creating offloading spec with name: NPUOffloadingSpec` | ✓ | 【实测】 |
| `Creating v1 connector with name: AscendOffloadingConnector` | ✓ | 【实测】 |
| 卸载池 | `cpu_bytes_to_use = 2 GiB`，`blocks_per_chunk = 8` | 【实测】 |
| 模型目录大小 | 6.4 MB（dummy 不读权重） | 【实测】 |

### 3.1 ★ `--prefix-match-unit 32` **不能省**（任务书的"这是一个信息"有答案了）

【实测】`PREFIX_MATCH_UNIT=0` 的臂 `l1-nopmu` 起服失败，原文：

```
Worker ... → build_offloading_config
AssertionError: tokens_per_block=32 not divisible by tokens_per_hash=128.
  Hybrid models (e.g. Mamba+Attention) need --enable-prefix-caching to align block sizes.
```

**在单卡 tiny-dummy 上也一样撞** ⇒ 这条与卡数、权重、层数**都无关**，
是 `state` 组 `block_size=32` + `_ascend_resolve_kv_cache_block_sizes()` 的哈希粒度 128 之组合。

### 3.2 ★ 卸载 scheduler 的补丁：**必需**（D_off8 的【推断】→ **【实测】证实**）

`vllm/distributed/kv_transfer/.../offloading/scheduler.py::get_sliding_window_size_in_chunks()`
末行 `assert isinstance(kv_cache_spec, FullAttentionSpec)` 会拒绝 DSV4.1 的
`UniformTypeKVCacheSpecs`（13/12 个 group **全是**包装类型）。

因 `prbench-c0/c1/c2` 是**共享**容器，我没有改镜像源码，而是用
`PYTHONPATH` + `sitecustomize.py` 在**进程内**打了 D_off8 同款的 3 处最小补丁
（unwrap + 放宽 assert），并额外加了一段**只读**的 group 日志。

**控制臂（`l1-nopatch2`：`L1_PATCH=0` + `PREFIX_MATCH_UNIT=32` + `WOA_FIX=1`）【实测】起服失败**，
原文（这正是 `logs/001` §4.3 标【未确认】的那一格，现在**补上了**）：

```
File "vllm/v1/core/sched/async_scheduler.py", line 14, in __init__
File "vllm_ascend/patch/platform/patch_balance_schedule.py", line 116, in __init__
File "vllm/v1/core/sched/scheduler.py", line 140, in __init__
File "vllm/distributed/kv_transfer/kv_connector/factory.py", line 75, in create_connector
File "vllm_ascend/.../kv_offload/native/offloading_connector.py", line 331, in __init__
File "vllm/distributed/kv_transfer/.../offloading_connector.py", line 74, in __init__
      self.connector_scheduler = OffloadingConnectorScheduler(
File "vllm/distributed/kv_transfer/.../offloading/scheduler.py", line 454, in __init__
      self.config = SchedulerOffloadConfig.from_spec(
File "vllm/distributed/kv_transfer/.../offloading/scheduler.py", line 183, in from_spec
      sw = get_sliding_window_size_in_chunks(
File "vllm/distributed/kv_transfer/.../offloading/scheduler.py", line 118, in get_sliding_window_size_in_chunks
      assert isinstance(kv_cache_spec, FullAttentionSpec)
AssertionError
```

⇒ **DSV4.1 的 `UniformTypeKVCacheSpecs` 包装确实会掉进那句 assert**，
而且**只需 `--prefix-match-unit` 不够**：哈希粒度断言过了之后立刻撞这一句。
**成本：1.5 分钟/次**（本臂 00:07:42 起服 → 00:08:32 失败），所以这类"上游到底哪一句挡的"
问题现在都能在单卡上秒级定位。

---

## 4. ★★ 判据链：单卡复现「能存不能取」（臂 `l1-d4k`，全部【实测】）

配置：`tensor_parallel_size=1`、`prompt 16 × 4096 token`（工作集 65,536 token）、
`HBM KV = 1 GiB`（= 22,719 token ⇒ 工作集是其 **2.9×**，必然真实驱逐）、
`reset_prefix_cache` 两轮、`cpu_bytes_to_use = 2 GiB`、`blocks_per_chunk = 8`。

| 观测 | 值 | 出处 |
|---|---|---|
| `BlockStored(medium="CPU")` | **704** | `out/l1-d4k.kv_events.log` |
| `BlockStored(medium="GPU")` / `BlockRemoved(GPU)` | 11,286 / 7,322 | 同上 |
| `AllBlocksCleared` | 1（reset 生效） | 同上 |
| `kv_offload_total_bytes_total{GPU_to_CPU}` | **2,873,425,920 B ≈ 2.87 GB** | `metrics_after.txt` |
| `kv_offload_size_count{GPU_to_CPU}` | **16**（16 个 chunk 写成功） | 同上 |
| **`kv_offload_total_bytes_total{CPU_to_GPU}`** | **0.0** | 同上 |
| `kv_offload_size_count{CPU_to_GPU}` | **0.0** | 同上 |
| `external_prefix_cache_queries_total` | **131,328** | 同上 |
| **`external_prefix_cache_hits_total`** | **0.0** | 同上 |
| `prefix_cache_hits_total` | 0.0 | 同上 |
| fill TTFT p50 / mean | **461.4 / 469.3 ms** | `client.log` |
| **replay TTFT p50 / mean** | **453.1 / 453.3 ms**（**−3.4%**，噪声级） | 同上 |
| 两轮 failed 请求 | 0 / 0 | 同上 |

**★ 与 8 卡臂 D 的对照（同一套判据、同一个 signature）**：

| | 8 卡真权重（臂 D） | 单卡 tiny-dummy（`l1-d4k`） |
|---|---|---|
| store | 393.6 GB / 12,288 块 | 2.87 GB / 704 块 |
| **load** | **0 B** | **0 B** |
| 外部层查询 | 1,048,832 | 131,328 |
| **外部层命中** | **0** | **0** |
| replay vs fill | 4167 vs 4192 ms（+0.6%） | 453 vs 469 ms（−3.4%） |

⇒ **"存进去了但一次都不取回"这个 signature 在 1 张卡上 1:1 复现。**
（块数/字节数的差异只是池子与工作集大小不同，**比例与定性完全一致**。）

---

## 5. 三个必须记录的坑（都会影响后来人）

### 5.1 【实测】`--load-format dummy` 与 V4.1 的 `wo_a` 路径不兼容（**这是本次最大的新发现**）

直接 `--load-format dummy` 起服会死在 **ACL graph capture**：

```
File ".../vllm_ascend/attention/dsa_v41.py", line 609, in forward
File ".../vllm_ascend/attention/dsa_v1.py", line 1562, in _forward_o_proj
IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
```

根因【实测 + 代码】：`ops/linear.py::AscendColumnParallelLinear.weight_loader` 里对 `wo_a` 做了
`weight.view(n_local_groups, o_lora_rank, -1).transpose(2, 1).contiguous()` ⇒ `[groups, hidden, rank]`，
而 `_forward_o_proj` 的 `npu_transpose_batchmatmul` 需要 **3-D** 权重；
**dummy 加载不调用 `weight_loader`** ⇒ 权重停在 2-D ⇒ `perm_x2` 越界。

**适配方式**（本次采用，`patch/sitecustomize.py`）：用 `sys.meta_path` post-import 钩子，
等 `vllm_ascend.attention.dsa_v1` 真正加载后，给 `AscendDSAImpl._forward_o_proj` 套一层
**幂等**的布局补丁（`ndim == 2` 时才做，且**复刻 weight_loader 的同一变换**）。
【实测】打上后：`(4096, 4096) -> (8, 4096, 512)`，23 层各一次，起服通过。

> ⚠️ 不能在 `sitecustomize` 里直接 import：会撞循环导入
> （实测 `ImportError: cannot import name 'DeviceOperator' from partially initialized module`）。

### 5.2 【实测】`sitecustomize` 的 env 门控要小心

`L1_DUMMY_WOA_FIX=1` 只有在 `PYTHONPATH` 确实包含 `patch/` 时才有效。
我的控制臂 `l1-nopatch` 就是踩了这个（`L1_PATCH=0` 顺带关掉了 PYTHONPATH）
⇒ 它死在 `wo_a`（与 scheduler 补丁无关），**没有回答"补丁是否必需"**。已在脚本里修掉。

### 5.3 【实测】`--max-model-len` 与 prompt 长度

第一次臂 `l1-woa2` 用 `prompt_tokens=8192 = max_model_len` ⇒ **16/16 请求全失败**
（`[fill] failed=16`，TTFT 全空）。改成 `4096` 后 0 失败。
⇒ 压测脚本的 prompt 长度必须 ≤ `max_model_len - 生成余量`。这不是卸载的问题。

### 5.4 【实测·环境】pinned host 这条路在"关掉 Engram"时可以走通

本次 `ENGRAM=0`，`cpu_bytes_to_use = 2 GiB` **分配成功**（`initialize_kv_cache` 没撞 207001），
与 `logs/001` §4.2 的"`ENGRAM=1` 必挂 / `ENGRAM=0` 必通"一致。
⚠️ 但这**不能**推广成"Engram + 卸载已解决"：本臂的 Engram 是**关掉**的。

---

## 6. 卡与环境纪律

| 项 | 状态 |
|---|---|
| 用卡 | **只有 c1（Phy-ID 6）**，全程走 `tools/a3_chip.sh` 锁（退出码 75 未出现） |
| c0 / c2 | **未占**（c0 的锁 23:45 起由另一个子代理 `d2_d2-dram32` 持有 —— 我一次都没碰） |
| Phy-ID 8–15 | **未碰** |
| `dsv41-a3` / `mooncake-master` | **未碰** |
| `upstream-v41/` | **只读**（未写） |
| `/tmp` | **未用**；本机临时区 `~/tmp/20260921/l1_dummy/`，容器内产物落 `/work/agents/L1_dummy/` |
| 共享容器 | **没有改镜像里的任何 vllm/vllm-ascend 源码**（全部用 PYTHONPATH 进程内补丁） |
| 每臂收尾 | `run_arm.sh` 的 trap 会 `kill` 掉自己起的 `vllm serve` 进程组；容器本体保留 |

---

## 7. 复现（一条臂 ≈ 2 分钟）

```bash
# 0) 造 tiny config（一次即可；宿主侧 python3 就行）
python3 ~/projects/dsv41-upstream-pr/agents/L1_dummy/make_tiny_config.py \
  --src ~/models/DeepSeek-V4.1-Flash \
  --dst ~/projects/dsv41-upstream-pr/agents/L1_dummy/models/model-tiny

# 1) 一条臂（单卡，槽位锁内）
bash ~/projects/dsv41-upstream-pr/tools/a3_chip.sh c1 --timeout 1800 --name l1-x -- \
  env TAG=l1-x OFFLOAD_GB=2 PREFIX_MATCH_UNIT=32 PROMPT_TOKENS=4096 PROMPTS=16 WOA_FIX=1 \
  bash /work/agents/L1_dummy/scripts/run_arm.sh

# 2) 观察点
#    out/l1-x.kv_events.log         → BlockStored:CPU > 0
#    out/l1-x.metrics_after.txt     → CPU_to_GPU 恒 0、external_prefix_cache_hits 恒 0
#    out/l1-x.client.log            → replay TTFT ≈ fill TTFT
```

**不占卡的探针**（验证"截断被拒"）：

```bash
docker exec prbench-c0 python3 /work/agents/L1_dummy/probe_truncation.py
```

---

## 8. 交付物清单

| 文件（`a2/agents/L1_dummy/`，远端同路径在 `~/projects/dsv41-upstream-pr/agents/L1_dummy/`） | 作用 |
|---|---|
| `make_tiny_config.py` | 造 tiny config（只压参数量，**结构字段强制自检**） |
| `probe_truncation.py` | 纯 Python 探针：证明 40→8/4 被 `plan_cache_slots()` 硬拒 |
| `patch/sitecustomize.py` | 进程内补丁：① 卸载 scheduler（unwrap + 放宽 assert）② `wo_a` dummy 适配（post-import 钩子） |
| `scripts/run_arm.sh` | 单卡单臂运行器（起服 → /health → 事件探针 → fill/replay → 收指标 → 停服） |
| `bench/{kv_offload_client,kv_events_probe,summarize}.py` | 判据链（与 `logs/45`/`001` 同口径） |
| `models/model-tiny/` | tiny 模型目录（6.4 MB） |
| `out/l1-*.{server.log,client.json,kv_events.log,metrics_after.txt,meta.txt}` | 原始数据 |

**原始数据已随本日志落到 `a2/logs/raw/010-l1-dummy/`（720 KB，4 条臂）**：
`l1-d4k.*`（★ 复现臂）、`l1-nopatch2.*`（控制臂，无 scheduler 补丁）、
`l1-nopmu.*`（无 `--prefix-match-unit`）、`l1-woa2.*`（prompt 过长那次的失败）
；另有 `models/model-tiny.config.json`（tiny config 存档）。

### 臂清单（全部在 c1 / Phy-ID 6，单卡）

| 臂 | 关键开关 | 结果 | 用时 |
|---|---|---|---|
| `l1-nopmu` | `PREFIX_MATCH_UNIT=0` | ❌ `32 % 128` 断言（§3.1） | ~70 s |
| `l1-woa2` | 全开，但 `prompt=8192=max_model_len` | ⚠️ 起来并跑完，但 **16/16 请求超长失败**（§5.3） | ~80 s |
| **`l1-d4k`** | **全开，`prompt=4096`** | ✅ **复现"能存不能取"**（§4） | **~2 min** |
| `l1-nopatch2` | `L1_PATCH=0` | ❌ `FullAttentionSpec` 断言（§3.2） | ~1.5 min |
| `l1-woa` / `l1-nopatch` | 中途两个脚本/bug 臂 | ⚠️ 无信息量（循环导入 / PYTHONPATH 门控），已修 | — |

---

## 9. 未完成 / 交给下一步

| # | 项 | 状态 |
|---|---|---|
| 1 | 「不挂 scheduler 补丁必挂」 | ✅ **已补做**（`l1-nopatch2`，见 §3.2） |
| 2 | 用本路线**验证修复**（把 `CPU_to_GPU` 打起来） | 【未做】—— 这正是 L1 存在的意义：**2 min/臂** |
| 3 | TP2 复现（候选根因①「跨 rank 的 `group_idx` 错位」） | 【未做】—— TP1 已复现 ⇒ **说明"零取回"与 rank 数无关**，这本身是有力线索（排除了"只有 TP8 才有"的解释） |
| 4 | `SchedulerOffloadConfig.from_spec` 的 group 明细日志 | 【未采到】—— 该函数在本版本没被走到（我的日志点没触发），所以 `alignment_chunk_count` 的实值仍缺 |
| 5 | Engram 打开时的兼容性 | 【未做】—— 本路线刻意关了 Engram；`logs/001` §4.2 的 207001 问题**依然存在** |

### ★ 对决策的直接回答

> **用 1–2 卡（dummy + 截断）能不能复现/验证卸载的"存/取"行为？**

**能。**（【实测】单卡 TP1 已 1:1 复现"能存不能取"）
代价与边界（都要如实记）：

* **"截断层数"不可行**，但**"保留 40 层结构 + 只压参数量"可行** ⇒ 结构等价性反而更干净
  （`full` + `state(32)` + 10×`swa` 全在）；
* 需要**两个进程内补丁**（scheduler unwrap、`wo_a` dummy 适配）—— 都只作用于我自己起的进程；
* **丢掉的**：Engram（关）、MTP/dspark（关，少 1 个 group）、TP8/EP8 的通信与遍历顺序；
* ⇒ **L2（真权重 8 卡）仍必做一次**，但只做一次。
