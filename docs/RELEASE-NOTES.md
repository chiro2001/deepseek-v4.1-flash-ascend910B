# 发布说明 —— DSV4.1-Flash W4A8 优化版 vLLM（A2 / A3）

发布日期：2026-09-17
适用模型：`v41-w4a8-engram-dr-vision-qrot-mtpq`（W4A8 + Engram-int8 常驻 DRAM + DSpark + Vision）

---

## 1. 这个包是什么

一个**独立可发布**的仓库，把「优化版 DSV4.1-Flash 推理服务」的**全部输入**放在一起：
补丁（两种形态）、镜像构建、起服、自检、验收、量化复现、预期指标与负结果。

目标读者有两类，各自只需要一条命令：

| 读者 | 一条命令 |
|---|---|
| A3（8×910C）运维 | `BASE_IMAGE=quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3 IMAGE_TAG=dsv41-a3:v1 SKIP_PGO=1 bash scripts/build_image.sh` → `bash tools/list_chips.sh` 选卡 → `DEVS="8 9 10 11 12 13 14 15" MODEL=... IMAGE=dsv41-a3:v1 bash scripts/serve_a3.sh` |
| A2（8×910B3）运维 | `bash scripts/build_image.sh`（默认 base = `…:deepseek-v4.1-flash-openeuler`）→ `MODEL=... bash scripts/serve_a2.sh` |

## 2. 相对上一交付包（`a2_pkg_v6`）的变化

| # | 变化 | 为什么 |
|---|---|---|
| 1 | **补丁拆成三仓库 git 系列**（`patches/{vllm-ascend,vllm,msmodelslim}/`，`git am` 可复现） | 发布/评审/上游提交都需要带历史的补丁；整文件形态只能"替换"，说不清改了什么 |
| 2 | **A3 路径一等公民**（`scripts/serve_a3.sh`） | 之前只有 A2 交付脚本；A3 的 chips、NUMA、镜像、PGO 都不同 |
| 3 | **工具调用前端修正**：默认 `deepseek_v41` 三件套 | 旧前端 `deepseek_v4` 的 chat template 不能正确处理 agent 工具调用（A2 首轮实测问题 #1） |
| 4 | A3 默认 `PYTHON_PGO=0` | A2 的 PGO 产物针对 **A2 镜像**的 libpython 编译；A3 镜像的 libpython md5 不同（见 §4） |
| 5 | dry-run 修复 + PGO 提示修复 | 自测时暴露：`DRY_RUN=1` 因日志目录不存在而刷屏报错 |
| 6 | **A3 选卡交给用户**：`DEVS` 必填 + `tools/list_chips.sh` + 占用检测 | 原来脚本硬编码 chips 8-15，等于替用户假设了"哪 8 张是你的"；实际机器上卡是被多方共用的 |
| 7 | **绑核改由 vLLM 内部做，外部不再计算**：容器默认不设 `--cpuset-cpus/--cpuset-mems`（`CPUSET=-1 MEMS=-1`），改由 vllm-ascend 的 `cpu_binding` 按 NPU 拓扑给每个 rank 绑核（`enable_cpu_binding: true`，即 `CPU_BIND=1`）。`CPUSET=`/`MEMS=` 降级为逃生口，显式给才透传 | 外部先圈 NUMA 等于替内部做决定；多机/多租户/混合选卡时外部区间一旦与选中的卡对不上，内部再绑也回不到正确节点。日志证据 `[cpu_binding.py] mode=topo_affinity rank=N` / `[migrate] NPU:N -> NUMA [M]` |
| 8 | **补丁基线换到公开可取的仓库**：vllm-ascend 系列 base 改为 `46856f89e`（[`GDzhu01/vllm-ascend-v41-private`](https://github.com/GDzhu01/vllm-ascend-v41-private)，匿名可读）。该 commit 与我们实测代码的 **11 个文件逐字节一致**，8/8 补丁可直接 `git am` | 原 base 在一个**私有** fork 上，外部读者拿不到，`git am` 无从谈起。"带历史的补丁"必须先有一个公开可达的基线 |
| 9 | **PGO 二进制移出包**（35 MB → **5.2 MB**）：删除 `optim/pgo/{python3,libpython3.12.so.1.0}`，改为在**目标机**执行 `bash build_scripts/00_ensure_pgo.sh` 编译生成（指纹缓存，只编一次）。新增 `optim/pgo/README.md`；`build_image.sh` 在无产物时给出明确引导 | 二进制与 CPU/gcc/glibc 绑定（实测 A2 与 A3 镜像的 libpython md5 就不同），进 git 历史既臃肿又未必适用 |
| 10 | **脱敏**：清除全部内部用户名（62 处）、主机名（299 处）、外部租户名；补丁作者行统一为中性身份 | 准备公开到 GitHub |
| 11 | **LICENSE = Apache-2.0** + `NOTICE` 声明派生来源（vllm / vllm-ascend / msmodelslim / CPython） + `.gitignore` | 补丁派生自 Apache-2.0 代码，再分发需带许可与归属声明 |

**性能配置本身没有改**：11 个优化开关、门控 env、capture sizes、启动参数与 v6 完全一致
（`MOE_AG/O_PROJ_2D/MOE_MASK/ROPE_IDXSEL/ENGRAM_JIT/QLI_NOCAND` 全开，
`MOE_ZERO/MOE_NF/DRAFT_GRAPH` 全关）。

## 3. 硬约束（交付时必须成立）

1. Engram-int8 常驻 DRAM（`V41_ENGRAM_HOST_RESIDENT=1`）
2. DSpark 开（`--speculative-config` method=dspark, 5 tokens）
3. KV cache 在 HBM 且 **> 3,145,728 tokens**（实测 4.14M）
4. Vision 23/23、GSM8K ≈198/200
5. `static_kernel.py:650` 命中数 **必须为 0**（否则静默降级、数字不可信）

## 4. 已知限制 / 未验证项（**别当成 bug**）

| 项 | 状态 |
|---|---|
| PGO 用在 A3 | ❌ 未验证。A2 镜像 libpython md5 `f1ebbee1405d0e31136aa4480b57b3dc`，A3 镜像 `eaea156ea8ddf85b0b2d71f77872991e`。要试就 `PYTHON_PGO=1` 并自行 A/B |
| `V41_ENGRAM_GATE_CHUNK` | 线上跑 `0`（=stock gate）；分块路径的 −1.56 ms 只在 8K 测过 |
| `V41_MOE_ZERO_INVALID` / `MOE_NF` | 实验/负结果，默认关，不在本系列内 |
| `DRAFT_GRAPH=1` | A2 首轮实测：缺 `DSPARK_GRAPH_CAPTURE_METADATA=1` 会静默失效（A 恒 1.0）。A3 结论是不可分辨 ⇒ 维持 eager draft |
| A（接受长度） | **不能当绩效指标**；脏会话里反而更高。必须报 `(clean-rate, ms/step)` |
| 110 tok/s | **不可交付**：设备 busy 本身 30.9 ms > 达标所需的 25.1 ms |

## 5. 自测状态

| 平台 | 项目 | 结果 |
|---|---|---|
| A3（A3-node1 chips 8-15） | 镜像构建（`BASE_IMAGE=…flash-a3`, `SKIP_PGO=1`） | ✅ 11/11 文件 md5 校验通过 |
| A3 | 起服 + health + KV + static_kernel + 工具调用 | ✅ 见 `docs/A3-SELFTEST-20260917.md` |
| A3 | 选卡流程（`list_chips` / DEVS 校验 / 占用拒绝） | ✅ 用例见同文件 §2 |
| A3 | 绑核策略（外部不设 cpuset + 内部 `enable_cpu_binding`） | ✅ 容器 `cpuset=[]`、`enable_cpu_binding:true`、8 个 worker 各自 36 核且与 NPU 同 NUMA（322-357 … 602-637）；同口径时延与绑核前一致（8K 30.68 / 32K 31.04 ms/step） |
| A2 | v6 全流程（性能/视觉/GSM8K/多 batch） | ✅ 见 `EXPECTED_PERF.md` §7.3（v6 实测） |

## 6. 目录速查

```
README.md                 一页上手（A2 / A3 两条命令）
patches/README.md         ★ 三个仓库的 patch 系列：base commit / am 方法 / 门控 env / 依赖顺序
patches/{vllm-ascend,vllm,msmodelslim}/   系列补丁 + MD5SUMS + base.txt
patches/files/            与系列等价的逐字节整文件（烘焙/bind-mount）
patches/_tools/           生成与验证系列用的脚本（make_series_*.sh / verify_series.sh）
scripts/serve_a3.sh       A3 一键起服（chips 8-15）
scripts/serve_a2.sh       A2 一键起服（也承载 A3 的引擎）
scripts/build_image.sh    烘焙补丁产出新镜像（A2/A3 通用，用 BASE_IMAGE 指定底座）
scripts/run_test.sh       验收（8K/32K/128K + Vision + GSM8K）
tools/attach_test.sh      服务已在跑时的附着自检（不起/不删容器）
docs/RELEASE-NOTES.md     本文件
```
