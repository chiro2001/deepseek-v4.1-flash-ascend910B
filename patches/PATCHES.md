# patches/ —— 两种补丁形态

> **先读 [`README.md`](README.md)**：那里是发布口径的三仓库 patch 系列说明
> （base commit / `git am` 方法 / 每个补丁的门控 env 与实测收益 / 依赖顺序 / 验证方法）。
>
> 本文件是**整文件形态**（`files/`）的历史清单，内容仍然有效 —— 它被
> `scripts/build_image.sh`（烘焙）与 `scripts/serve_a2.sh PATCH_MODE=mount`（挂载）使用。
> 两种形态已被机器验证为**逐字节等价**。

# bind-mount 整文件补丁清单（v3）

> 来源：A3-node1 `/home/user/projects/dsv41/probe_*/`（即 `serve_a21.sh` 实际挂载的文件）。
> 打包日期：2026-09-16。**本目录里的文件是逐字节复制**，md5 见下表，可用 `md5sum -c MD5SUMS` 复核。

## 0. 两种用法

| 模式 | 做法 | 适用 |
|---|---|---|
| **烘焙（推荐，包内默认）** | `bash scripts/build_image.sh` 把这些文件 COPY 进镜像的目标路径，原文件留 `.a2orig` 备份 | 「一次执行生成修改后的容器镜像」 |
| **临时挂载** | `serve_a2.sh` 加 `PATCH_MODE=mount`，用 `-v` 挂 `patches/files/*` | 不想重建镜像、只想试单个补丁 |

两种模式的**文件内容完全一致**，只是落位方式不同。

## 1. 默认烘焙的已验证集合（13 项）

| # | 目标文件（容器内，`$A=/vllm-workspace/vllm-ascend/vllm_ascend`） | 包内文件 | md5 | 门控 env（默认值） | 作用 / 实测收益 |
|---|---|---|---|---|---|
| 1 | `$A/models/deepseek_v41/engram_hbm.py` | `files/engram_hbm.py` | `02ba2b7c258ff16663cd316b69c44fb8` | （总是开）`V41_ENGRAM_HOST_RESIDENT=1` `V41_ENGRAM_LOCAL_OWNER_FILE=/tmp/v41_engram_localowner` | Engram host 常驻 + `LOCAL_OWNER=fast`（走 numpy/numba plan 分支）。route 少一次 metadata all_gather 与 ids all_to_all |
| 2 | `$A/models/deepseek_v41/engram_hash.py` | `files/engram_hash.py` | `dc63b40b1ec739ddc467edceebe85478` | `V41_ENGRAM_JIT=1` | hash 0.427 → **0.076 ms/step**（numba JIT） |
| 3 | `$A/models/deepseek_v41/engram_jit_kernel.py` | `files/engram_jit_kernel.py` | `6668d3fe3c6333b47e9dde3df7cfe03a` | `V41_ENGRAM_JIT=1` | 上一行的 sidecar（新文件，**目标目录必须同放**） |
| 4 | `$A/models/deepseek_v41/engram_plan_kernel.py` | `files/engram_plan_kernel.py` | `0be62d7775374b0167a54f5b393a65ac` | `V41_ENGRAM_JIT=1` | plan 0.261 → **0.068 ms/step**（sidecar） |
| 5 | `$A/models/deepseek_v41/engram_gate.py` | `files/engram_gate.py` | `146010cac42261e9dc4380699e156252` | `V41_ENGRAM_GATE_CHUNK=0` `V41_ENGRAM_GATE_MAX_TOKENS=2048` | 分块 gate，去掉 2048 行 padding：**−1.56 ms**（8K），KV 反而更省 |
| 6 | `$A/models/deepseek_v41/model.py` | `files/model.py` | `dc0b5d936906b437e1ab93ae5b4ee456` | （与 #1 配对） | Engram host-resident + device-index 入图及 CED P/D 实验 |
| 7 | `$A/ascend_forward_context.py` | `files/ascend_forward_context.py` | `6cccd4259bd65c907ef9d9dd42a83dca` | `V41_MOE_COMM_ALLGATHER=1` | **MoE 走 AllGather**：128K **−4.25 ms**、32K −1.35、8K −1.23；KV 3.39M→4.16M；输出逐字节一致 |
| 8 | `$A/attention/dsa_v1.py` | `files/dsa_v1.py` | `9a36e709b0937589eab05c5316a62591` | `V41_O_PROJ_2D=1` | **F3**：`wo_a` 退化 batch matmul → 2D matmul，**−0.31~0.76 ms/step** |
| 9 | `$A/ops/fused_moe/token_dispatcher.py` | `files/token_dispatcher_moemask.py` | **`a91fbc48350530d987ef3bb1ff15f1cb`** | `V41_MOE_MASK_RANGE=1` | **moe-mask-range**：范围比较替代 Index+IndexCheck 掩码链，**−0.51 ms**；精度已过（GSM8K 100/100、Vision 23/23）。★ 本版加 **[SAFE-L1]**：比较改在 **int32 域**（`_i32_scalar`），去掉每层 2 次 `Cast INT32→INT64`（A3 实测 **80 次/步 → 0**，≈0.096 ms/step）；逐位等价。旧 md5 = `a695735a…` |
| 10 | `$A/ops/rope_dsv4.py` | `files/rope_dsv4.py` | `6a19890850ac7cb41c535b070c2dfbf6` | `V41_ROPE_IDXSEL=1` | **rope-idxsel**：cos/sin 取表链 6 kernel → 2，**−0.45~0.62 ms/pass** |
| 11 | `$A/models/deepseek_v41/indexer.py` | `files/indexer.py` | `f61f242df4f060106ce1bf4500ff5844` | `V41_QLI_NO_CANDIDATE=1` | **QLI no-candidate**：QLI per-op 99.3 → 50.3 µs，**−0.49 ms** |
| 12 | `$A/models/deepseek_v41/engram_device_index.py` | `files/engram_device_index.py` | `41e7f012c596cc7539162e5f7ed8475d` | `V41_ENGRAM_DEVICE_INDEX=auto` | **v8 新增**：host-mapped 表（设备直读 host DRAM）+ 设备侧向量化哈希 + 能力探测（只做 `host_register`） |
| 13 | `$A/models/deepseek_v41/engram_graph.py` | `files/engram_graph.py` | `bee1bdb20491ce0dfb2df1c3be1bf1f4` | `V41_ENGRAM_DEVICE_INDEX=auto` | **v8 新增**：每个 batch shape 一张 ACLGraph（零拷贝 + 指针校验），把整条设备路径入图 |
| 14 | `patches/admission_gate.patch`（git apply 到 `$VLLM_ROOT=/vllm-workspace/vllm`） | `admission_gate.patch` | `8243dff6c9dc3d87805f1dfd7820c23f` | `VLLM_ADMISSION_GATE=1` | 预填充隔离，prefill 不饿死 decode |

> 注：#3/#4/#12/#13 是**新增文件**（不是覆盖），构建脚本会直接 COPY；其余是覆盖 + `.a2orig` 备份。
> #14 是**补丁**（不是整文件）：`build_image.sh` 烘焙时 `git apply`；`PATCH_MODE=mount` 由 `serve_a2.sh` 起容器后**现场 apply** 并断言 live tree 命中（不补的话 `VLLM_ADMISSION_GATE=1` 会被静默忽略）。

## 2. 只放不装的实验补丁（**默认关闭**，见 CHANGELOG 标红项）

| 目标 | 包内文件 | 门控 env | 状态 |
|---|---|---|---|
| `$A/ops/fused_moe/token_dispatcher.py` | `files/token_dispatcher_moezero.py` | `V41_MOE_MASK_RANGE=1` `V41_MOE_ZERO_INVALID=1` | ❌ **未在本机完成端到端验证**（同会话 A/B 仍在跑）。它同时含 #9 的 moe-mask-range，所以开它要**替换** #9 的挂载（烘焙模式下用 `tools/enable_moe_zero.sh`） |
| `$A/attention/dsa_v1.py` | `files/draft/dsa_v1.py` | `DRAFT_GRAPH=1` `DSPARK_DRAFT_METADATA_MODE=sync` | ❌ **上卡验证未做**。目标机预计省 draft 派发 ~24 ms/轮 |
| `$A/spec_decode/dspark_proposer.py` | `files/draft/dspark_proposer.py` | 同上 | ❌ 同上 |
| `$A/spec_decode/llm_base_proposer.py` | `files/draft/llm_base_proposer.py` | 同上 | ❌ 同上 |
| `$A/models/deepseek_v41/indexer.py`（诊断用） | `files/indexer.py` | `V41_FORCE_CAND_MODE=0/3/4` | ⚠️ 诊断开关，`0` = stock 等价；默认不挂 |

> `draft/` 三个文件与 #8 的 `dsa_v1.py` **冲突**（同一目标路径）。`probe_draft/dsa_v1.py` 是
> 「stock + 0004 + F3」合并版，因此 `DRAFT_GRAPH=1` 时应当**只挂 draft 版**（`serve_a2.sh` 已处理）。

## 3. 与 A3-node1 挂载方式的差异（务必知道）

1. **设备号**：A3-node1 用 `DEVS="8 9 10 11 12 13 14 15"`（back8，chips 0-7 是别人的）。A2 是单机 8 卡，
   必须用 `DEVS="0 1 2 3 4 5 6 7"`。量化脚本同理（见 `quant/README.md` §4）。
2. **`LOCAL_WORLD_SIZE=8`**：torch_npu 的 static_kernel 要求 `os.environ` 里有该变量，否则
   **静默禁用静态内核**（`UserWarning: LOCAL_WORLD_SIZE is not set in a multi-card context ... will be disabled`）。
   A3-node1 从容器启动就注入；A2 上 `serve_a2.sh`/`run_test.sh` 同样注入，起服后**必须复检**
   `grep -ac "static_kernel.py:650" <log>` == 0。
3. **缓存目录**：A3-node1 用独立缓存（`~/.cache/vllm-a21perf` 等），避免与历史编译缓存串味。A2 上
   `run_test.sh` 用包目录下的 `./cache/`，第一次编译 static kernel 慢（几分钟 ~ 十几分钟），之后复用。
4. **HOTSPIKE / 探针关闭**：A3-node1 开着 `hotspike`（宿主热更新）与 `V41_ENGRAM_ROUTE_PROBE=1`（每 20 步打一行）。
   A2 交付包**默认全关**（`HOTSPIKE=0 ROUTE_PROBE=0`），否则会引入额外 host 开销并污染日志。
