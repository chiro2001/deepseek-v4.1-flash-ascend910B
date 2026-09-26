# DeepSeek-V4.1-Flash W4A8 —— 昇腾（Ascend）推理服务

在**昇腾 910C（A3）**与 **910B3（A2）**上拉起 DeepSeek-V4.1-Flash 的 W4A8
量化推理服务：完整补丁、一键起服、自检与验收口径。

**核心卖点**：**单台 A3（8 卡 / 16 device）**即可跑 1M 上下文的 PD 分离；
prefill 相对全 40 层同置部署 **2.07×**（144K）。

> **本包的全部优化工作（算子分析、补丁编写、性能调优、文档）均由
> `deepseek-v4.1-flash` 模型自主完成**，未经人工逐行改写。

---

## 1. 你能得到什么

### 1.1 硬件

| 形态 | 硬件 | 谁用得上 |
|---|---|---|
| **A3 单机 8+8 PD 分离 + CED** | **1 台 A3**（8 卡 = 16 device）全用 | 单机、要长上下文、要 PD 分离 |
| A3 单实例 TP8 | 1 台 A3 的 **8 个 device（= 4 块卡）** | 单机、不拆 P/D |
| A2 单实例 TP8 | **1 台 A2**（8×910B3，1 卡 1 device） | A2 口径 |

**A2 与 A3 的差别**：A3 每块卡有 **2 个 die**（`/dev/davinci0..15`），A2 每块卡 1 个。
`DEVS=` / `TP=` 数的都是 **device**，不是卡 —— 详见 §2 的「术语」。

**前置**：模型权重（273 GB，见 §3.1）、`quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3`
底座镜像（或本仓 §3.4 的镜像包）。

### 1.2 性能（实测，A3 真权重）

**PD 分离 + CED vs 全 40 层同置**（同机同请求，唯一变量是 P 的层数）：

| 上下文 | prefill tok/s（CED） | 全 40 层基线 | 比 | TTFT（CED） | TTFT（基线） |
|---|---:|---:|---:|---:|---:|
| 32K | **12,536** | 7,219 | **1.74×** | — | — |
| 144K | **13,676** | 6,599 | **2.07×** | **10.95 s** | 21.8 s |
| 1M | **9,897** | 3,557 | **2.78×** | **101.2 s** | 280.3 s |

**decode**（同一批请求，唯一变量是 P 的层数）：两臂几乎相同 —— 25.3 / 26.8 / 37.9 ms/step
（CED）vs 25.4 / 26.8 / 37.4（基线）⇒ **CED 只改 prefill，decode 不退化**。

**单机同置形态**（TP8+EP8、128K 上下文、单流独占、全补丁）：

| 项 | 值 |
|---|---|
| 128K 单流 decode | **30.2 ms/step**（8 发中位，区间 29.8–31.2；最好一发 26.9） |
| 128K 单流吞吐 | **90.1 tok/s**（另一臂 85.0；峰值 110.5 **不可交付**，见 §6） |
| 累计优化 | 未打补丁 39.10 → 全补丁 **31.39 ms/step（−19.7%）** |
| （另一口径）设备直索引入图 | 并发 1 时 29.5 → **28.4**、并发 4 时 35.3 → **32.1 ms/step** —— ⚠️ 这是 1K prompt 口径，**不要与上面 128K 那行直接相减** |
| KV 容量 | **2,823,080 tokens**（默认 `GPU_UTIL=0.92`） |

**并发吞吐**（A3、1024 token prompt、256 输出、`MAX_SEQS=64 PREFIX=1`）：

| 并发 | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 单流 tok/s | 90.3 | 89.2 | 80.0 | 59.3 | 42.5 | 31.3 | 20.2 |
| **总吞吐 tok/s** | 87.1 | 151.5 | 237.7 | 324.4 | 432.6 | 583.9 | **719.5** |

### 1.3 精度

| 项 | 结果 |
|---|---|
| **CED-PD 验收** | **21/21 通过**（144K/1M 四针、流式、多轮、缓存命中） |
| 四针答案 | 与 `SPEC=0` 交付口径**逐字节相同** |
| Vision | **23/23** |
| GSM8K | **198/200**（另两次 199、197） |

### 1.4 支持的功能

| 功能 | 状态 |
|---|---|
| **1M 上下文**（`MAX_LEN=1048576`） | ✅ 含 1M 四针与多轮验收 |
| **PD 分离 + CED**（单台 A3） | ✅ 见 §3.3 |
| DSpark 投机解码 | ✅（`SP_TOKENS=5` 或 7；**低并发推荐**，见 §6） |
| Engram（V4.1 的记忆模块） | ✅ INT8 表常驻 DRAM；可进一步走设备直索引 |
| Vision | ✅ |
| 前缀缓存 | ✅（服务端计数器为准；代理不透传 `cached_tokens`） |
| 自动工具调用（tool call） | ✅ |
| codex 直连（Responses API） | ✅ 需先跑 §3.6 使能补丁 |

---

## 2. 术语：**卡 ≠ die** —— `DEVS` 数的是 device

> ⚠️ 本文档里凡出现 `DEVS=`、`TP=`、"芯片号"、"卡号" 的地方，
> 数的都是 **`/dev/davinciN` 的编号（= die）**，**不是物理卡**。

| 机器 | 物理卡（`npu-smi` 的 `NPU` 列） | **device / die** | 关系 |
|---|---|---|---|
| **A3（910C）** | **8 块** | **16 个** | **1 块卡 = 2 个 die** |
| **A2（910B3）** | 8 块 | 8 个 | 本包按 8 个 device 使用 |

**在自己机器上核对**（两条都不写任何东西）：

```bash
# 物理卡数（去重后的 NPU 号）
npu-smi info | grep -oE '^\| [0-9]+ +Ascend9[0-9]+' | awk '{print $2}' | sort -un | wc -l
# device / die 数（= DEVS 的可选范围）
ls /dev/davinci[0-9]* | wc -l
```

⇒ **A3 上 8 个 device = 4 块卡**；PD 分离用满 16 个 device = **8 块卡全用**。

---

## 3. 快速开始

### 3.1 拿权重

```bash
pip install modelscope
modelscope download chiro2001/DeepSeek-V4.1-Flash-w4a8-Ascend --local-dir <目录>   # 200+ GB
# ★ 下完必须重组 Engram（分片不能直接起服）—— 脚本随权重一起下载
```

### 3.2 A3：单实例 TP8（不拆 P/D）

```bash
bash tools/list_chips.sh                    # ① 只读：看哪些 device 空着、属主是谁
DEVS="8 9 10 11 12 13 14 15" \              # ② 8 个 device = 4 块卡
  MODEL=/path/to/v41-w4a8-engram-dr-vision-qrot-mtpq \
  bash scripts/serve_a3.sh
```

`DEVS` 是**用户输入**：哪 8 个 device 能用取决于当前谁在跑什么，脚本不替你判断。
默认会**拒绝已被占用的 device** 并打印占用进程（确实要带占用起服务才加 `ALLOW_BUSY=1`）。

### 3.3 A3：单机 **8+8 PD 分离 + CED**（推荐）

P 只跑 20 个 encoder 层 + layer-20 全局源投影；D 做 **128-token 有界重放 + 全 40 层**。

```bash
export MODEL=/path/to/v41-w4a8-engram-dr-vision-qrot-mtpq
bash deploy/a3-ced-pd/launch/serve_p.sh        # die 0–7（= 卡 0–3）→ :18990
bash deploy/a3-ced-pd/launch/serve_d.sh        # die 8–15（= 卡 4–7）→ :18991
bash deploy/a3-ced-pd/launch/serve_proxy.sh    #           → :18992（客户端连这个）
bash deploy/a3-ced-pd/launch/smoke.sh          # 144K 四针冒烟
```

也可以直接下载**已构建好的镜像包**（9.5 MB，不需要仓库）：
[Release `a3-ced-pd-v1`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/releases/tag/a3-ced-pd-v1)
—— 或看 [`deploy/a3-ced-pd/README.md`](deploy/a3-ced-pd/README.md) 的完整硬门清单。

### 3.4 A2：单实例 TP8

```bash
bash tools/check_checksums.sh               # 发包前（10 秒，不起容器）
bash scripts/build_image.sh                 # 烘焙补丁，产出 dsv41-a2:v8（约 10–20 min）
MODEL=/path/to/v41-w4a8-engram-dr-vision-qrot-mtpq bash scripts/serve_a2.sh
```

⚠️ **用发布包里的 `scripts/serve_a2.sh`，不要用镜像里那份** —— 后者是**构建时**的快照，
可能比包旧。脚本每次起服都打印自己的路径 + 版本 + md5，报障时先看这一行。

### 3.5 起服前/后自检

```bash
bash tools/selfcheck_pkg.sh                 # 起服前（30 秒，不起容器）：19 组自检
DRY_RUN=1 ... bash scripts/serve_a3.sh      # 干跑：只打印将执行的命令，不碰 docker
PORT=8020 bash tools/attach_test.sh         # 服务已在跑时（不起容器、不删容器）
# 完整验收：8K/32K/128K + Vision + GSM8K
```

起服后**必须**核对这几项（不看就别相信结果）—— 完整清单见 §7。

---

### 3.6 让 codex 直接连本服务（可选）

本包的 `tokenizer_mode=deepseek_v41` 只认 **chat-completions** 的块词汇表，
而 **codex 等 OpenAI 客户端**发的是 **Responses** 词汇表 ⇒ 直连不可用，
且**其中两条是静默的**：

| 缺陷 | 症状 |
|---|---|
| `input_text` 被渲染成字面量 `[Unsupported input_text]` | **HTTP 200**，但**用户的话根本没进模型** |
| codex 放系统指令的 `developer` 角色要求 content 非空 | **HTTP 500** |
| 控制 token 可从正文注入 | 可**伪造轮次边界** |

```bash
docker exec <容器> bash /opt/dsv41/tools/enable_codex_responses.sh on      # 幂等，带备份/回滚
docker exec <容器> bash /opt/dsv41/tools/enable_codex_responses.sh status  # PATCHED / STOCK
```

⚠️ **改的是容器可写层 ⇒ 要重启服务才生效**；容器重建会丢，重跑即可。
重启前**务必确认没有残留进程**（`VLLM::EngineCore` / `VLLM::Worker_*` 的进程名里
**没有** `vllm serve`，`pkill -f "vllm serve"` 杀不到），否则起服会卡在
`rtsMallocHost 207001`。**A3 真机实测通过**：单轮文本、工具调用、图片、多轮、子代理。

## 4. 关键调优旋钮

用户最常需要调的六个。全部有实测依据，**改一个测一个**。

| 旋钮 | 默认 | 调它会发生什么 |
|---|---|---|
| **`BAT_TOKENS`** | **8192** | ★ **长上下文正确率的开关**。2048 时 chunk 数翻 4 倍，长文通过率塌到 ~0（机制是 chunked prefill 每刀约 2% 偏离）。代价：KV 从 4.15M 降到 2.82M tokens（activation 峰值 0.79→3.21 GiB） |
| **`GPU_UTIL`** | **0.92** | 调到 0.94 能多 ~9% KV（3.09M），但**长 prompt 首 token 从 1.1 s 涨到 8 s**（实测 6~7×）。这是"KV 容量换 prefill 速度"的主动取舍 |
| **`DRAFT_GRAPH`** | **0** | 投机解码入图。**低并发推荐显式开**（`=1`）；但必须同时有 `DSPARK_GRAPH_CAPTURE_METADATA=1`，否则**静默失效**（`A≈1.0` 而 ms/step 反而更好看） |
| **`PREFIX`** | 0（性能口径）/ 1（生产口径） | 前缀缓存。**取决于业务是否高度复用前缀** —— 若输入输出比很大且命中率低（实测某负载仅 3.52%），关掉反而 TTFT −14.6% |
| **`SPEC`** | 1 | 投机解码总开关。**高并发吞吐场景建议关**：并发 4 时它把 ms/step 从 28.2 抬到 41.1（1.45×），换接受长度 A≈2.4 —— 收益集中在接受长度高的请求上，**成本由全批承担** |
| **`CPU_BIND`** / **`DROPCACHE`** | A3: 0 / 0 | **A3 共用机必需**：`CPU_BIND=0` 关掉内部 NUMA 绑核（否则目标节点满时 `migratepages` 内核态空转、服务永不就绪、`docker stop` 都停不下来）；`DROPCACHE=0` 不清整机 page cache（会打到别人） |

**`MAX_SEQS` 不是性能杠杆，是并发容量**：32→64 在并发 ≤32 时只有 ±1%；
只在并发 64/128 时 +8.7%/+7.2%，而此时单流掉 34%。聚合吞吐封顶约 259 tok/s
（A2 口径）—— **上界是算力，不是槽位数**。

---

## 5. 与上游的差异（2026-09-26 重新核实）

### 5.1 上游现在到什么程度

**上游仓 `vllm-project/vllm-ascend` `main`**（核实于 `1e9e03bc5`）：

| 项 | 上游状态 |
|---|---|
| V4.1 框架支持 | ✅ **已合入**（PR #16544，2026-09-18） |
| Engram host offload | ✅ **已合入**（同一 PR） |
| 验证的量化形态 | **W8A8**（+ INT8 Engram） |
| 多机同置部署 | **A3 × 2 台**（DP4/TP8/EP32）或 **A2 × 4 台** |
| 单台 A3 | ✅ 有路径：TP8/**DP2**/EP16 + Engram host offload（`--engram-config`）。但文档明确这是**冒烟级验证**（模型加载 / decode 图捕获 / 自然文本 / 混合长度并发），"**不建立数据集精度或性能**"；示例用 **128K** 上下文、512 batched tokens、1 GiB KV/rank |
| **PD 分离（Prefill-Decode disaggregation）** | ❌ **文档原文：`Prefill-Decode disaggregation is not covered by this guide.`** |

**上游 CED 的定义**：文档把 CED 描述为**模型架构**（40 层 = 20 causal-encoder + 20 decoder，
1M 上下文，含 SWA Bounded Replay）。也就是说"能拆"是模型给的，
**怎么在框架里拆成可部署的 P/D 两段，上游没做**。

### 5.2 本包的增量

| 增量 | 说明 | 证据 |
|---|---|---|
| **① 单台 A3 上的 PD 分离 + CED** | P 只跑 20 个 encoder 层 + layer-20 全局源；D 做 128-token 有界重放 + 全 40 层 | prefill **2.07×**（144K）、验收 **21/21** |
| **② W4A8 形态** | 上游验证的是 W8A8；本包跑 W4A8 + INT8 Engram | 273 GB 权重；精度 Vision 23/23、GSM8K 198/200 |
| **③ 1M 上下文的完整验收** | 上游单机示例是 128K | 1M 四针 4/4、多轮 3/3、TTFT 101.2 s |
| **④ 14 个补丁** | 通信 / Engram / 算子 / 调度四类优化，**逐项核对过上游没有等价实现** | 累计 **−7.71 ms/step（−19.7%）** |

**④ 的 14 项优化**（按瓶颈分类，每项独立门控、可单独关做 A/B）：

| 类别 | 优化 | 收益 |
|---|---|---|
| 通信 | **MoE 走 AllGather**（TP=EP 时把标量开销按 token 摊薄） | 128K **−4.25 ms**，KV 池 3.39M→4.16M |
| 调度 | **admission gate**（prefill 不再饿死 decode） | 首 token 后不再长时间不出字 |
| Engram | **INT8 表 host 常驻 + local-owner 快路径** | 释放 HBM 给 KV |
| Engram | **★ 设备直索引**（表仍在 host DRAM，设备算子直接读） | 每步同步 host 时间 **3.379 → 0.058 ms** |
| Engram | **hash / plan 的 numba JIT** | 0.427→0.076 / 0.261→0.068 ms |
| Engram | **gate 分块**（去掉固定 2048 行 padding） | −1.56 ms |
| 算子 | **QLI 无候选快速路径** | −0.49 ms |
| 算子 | **rope 取表融合**（6 kernel → 2） | −0.45~0.62 ms |
| 算子 | **`wo_a` 2D matmul** | −0.31~0.76 ms |
| 算子 | **expert mask 区间比较** | −0.51 ms |
| 量化 | msmodelslim 的 V4.1 W4A8 支持 | 仅在重新量化时需要 |

> ⚠️ **结构差异（要移植的人注意）**：本包的 Engram 补丁面向**扁平文件布局**
> （`engram_hbm.py` / `engram_hash.py` / `engram_gate.py`）；
> 上游 #16544 把 Engram **重构成了包**（`engram/{common,embedding,hash_state,layer,npu,parallel}.py`）。
> ⇒ 这 7 个文件**在上游已不存在**，要往上游提需要按新结构重做。
> 另外 7 个文件（`model.py` / `indexer.py` / `ascend_forward_context.py` / `rope_dsv4.py` /
> `block_table.py` / `token_dispatcher.py` / `dsa_v1.py`）在上游存在但无我们的实现。

---

## 6. 已知限制

| 项 | 状态 |
|---|---|
| **110 tok/s** | **不可交付**：设备 busy 本身 30.9 ms > 达标所需的 25.1 ms |
| **接受长度 A** | **不能当绩效指标**；必须同时报 `(ms/step, A, tok/s)` |
| `DRAFT_GRAPH=1` | **推荐显式开**（低并发），默认仍为 0 —— 理由见 §4 |
| `V41_MOE_ZERO_INVALID` / `MOE_NF` | 实验项 / 负结果，默认关 |
| **KV 门槛 3Mi** | **默认配置不再满足**（0.92 下 2.82M）。这是主动取舍，不是故障；要 3Mi 请显式 `GPU_UTIL=0.94` 并接受 8 s 级首 token |
| A2 与 A3 的性能差 | 硬件（两边同为 1600 GB/s/die）只能解释 ~15%，其余在 host 侧 |
| **codex 直连** | **需先跑使能补丁**：不装的话 Responses API 要么 500、要么**静默丢内容** |
| `reasoning.encrypted_content` | **不支持**（vLLM 侧直接 `raise`）。当前不触发；**一旦上游开始产出，这条链会 400** |
| **PD 分离的 DSpark 收益只在低并发** | 并发 4 时 ms/step 1.45×，成本由全批承担 |

---

## 7. 起服后必查（5 项）

```bash
# ① 静态内核没有静默降级 —— 必须输出 0
grep -ac "static_kernel.py:650" <serve.log>
# ② KV 容量 —— 默认 GPU_UTIL=0.92 时实测 2,823,080 tokens
grep -oE "GPU KV cache size: [0-9,]+ tokens" <serve.log> | tail -1
# ②b 长 prompt 首 token —— 8K prompt 应 ~1.1 s（0.94 时会是 8 s）
# ③ 口径对不对（性能口径 vs 生产口径不可混比）
# ④ Engram local-owner 是否切到 fast 路径
# ⑤ DSpark draft 图真的在出 token（A ≥ 1.3；A≈1.0 就是静默失效）
grep -a "SpecDecoding metrics" <serve.log> | tail -1
```

---

## 8. 目录结构（给人看的）

| 目录 | 内容 |
|---|---|
| `scripts/` | 起服脚本（`serve_a3.sh` / `serve_a2.sh` / `serve_a3_ced_pd.sh` …） |
| `deploy/a3-ced-pd/` | **PD 分离 + CED 部署形态**（启动器 + 镜像包 + payload 清单） |
| `tools/` | 自检、探针、压测、诊断（40+ 个，见 `AGENTS.md` §4） |
| `patches/` | 补丁件与载荷（`PATCHES.md` 有逐项说明） |
| `quant/` | 量化链（5 级，含 msmodelslim 补丁） |
| `docs/` `reports/` | 设计、验收、性能、故障史的完整记录 |
| `AGENTS.md` | **开发者 / Agent 视角**：补丁明细、调试入口、必踩陷阱 |

---

## 9. 许可与来源

* 许可、来源与致谢见 [`LICENSE`](LICENSE) / [`NOTICE`](NOTICE)。
* **测试语料不随包分发**（有版权）：`tools/fetch_corpus.sh` 自行下载并校验 sha256。

更细的内容（补丁逐项明细、故障史、性能拆解）在
[`docs/`](docs/)、[`reports/`](reports/)、[`EXPECTED_PERF.md`](EXPECTED_PERF.md)、
[`CORRECTNESS_STATUS.md`](CORRECTNESS_STATUS.md)。
