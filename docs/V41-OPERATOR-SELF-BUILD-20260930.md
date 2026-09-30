# 自己修算子的可行性验证：ops-transformer 源码 + 容器内重编链路

> 结论：**可行**。官方 `gitcode.com/cann/ops-transformer` 有 `attention/sparse_flash_mla`
> 的完整源码（含 arch22/A2A3 与测试），容器内**已具备**全部编译工具链与构建缓存，
> 并且我**跑通了单算子重编**（产出 `.json` + `.o`）。我们可以自己改、自己编、自己验。

---

## 1. 官方源码仓库确认

| 项 | 值 |
|---|---|
| 仓库 | https://gitcode.com/cann/ops-transformer |
| 定位 | 【实测】CANN 官方的 **transformer 类大模型算子库**（README 自述） |
| 调研 commit | `1461cca857ff3b2018404c0275e80e7aa13a3fc9`（**2026-09-30**，`fix 0-axis`） |
| 我们要的算子 | `attention/sparse_flash_mla/`（112 个文件） |
| **产品支持** | 【实测】README 表格：**Atlas A3 系列 √、Atlas A2 系列 √**、950PR/DT √ |
| 目录 | `op_kernel/arch22/`（A2/A3）、`op_kernel/arch35/`（950）、`op_host/`、`tests/`、`torch_extension/`、CMake |
| 开发活跃度 | 【实测】`sparse_flash_mla` 近期提交：`1461cca`(09-30)、`585be5a`(09-30)、`2903538`(09-29)、`b48da30`(09-29)、`c2a52ed`(09-28)、`e0a6f5c`(09-28) |
| tag | 有 `v9.1.0`（与我们容器 CANN 9.1.0 配套）——但该 tag 用**旧命名 `scfa`**；master 用 `csa` |

## 2. ★ 但官方 master **没有修**我们的问题

逐行核对 `attention/sparse_flash_mla/op_kernel/arch22/`（官方）与容器内版本：

| 议题 | 官方 master | 容器/我们 | 是否修复 |
|---|---|---|---|
| 因果界来源 | `cmpMaskRight = cmpMaskS2Size − actS1Size`（全局坐标公式） | 同 | ❌ 未修 |
| 丢键处理 | `CopyInSingleKv`：`if (keyBNBOffset < 0) return;` **不计数** | 同 | ❌ 未修 |
| 两阶段握手 | `CopyOutMrgeResult` 只搬 `mte2Size − mte3Size` 行；cube 按 `actualSingleProcessSInnerSize` 读 | 同 | ❌ 未修 |
| `CountValidCmpSparseLen`（二分数有效项） | **没有** | **有**（容器版更新） | — |
| `v0S2DealSize` | 干净（每块自己切片） | 非 `HEAD_RATIO_ONE` 时**硬编码 512** | — |

⇒ 官方最新源码与我们**同源**，两者都有"丢键 → 未写洞 → 读残留"的结构。
**不能靠升级官方版解决，必须我们自己改。**

## 3. ★★ 容器内已具备完整重编链路（已跑通）

### 3.1 组件齐备【实测】

| 组件 | 路径/状态 |
|---|---|
| 算子源码树 | `/vllm-workspace/vllm-ascend/csrc/attention/sparse_flash_mla/`（60 文件，含 `op_kernel/arch22`） |
| 已配置构建目录 | `csrc/build/`（**334 MB**，Ninja，含 `sparse_flash_mla_ascend910_93_0` 目标） |
| 工具链 | `/usr/local/Ascend/cann-9.1.0/tools/{bisheng_compiler,ccec_compiler}/bin/{bisheng,ccec,opc…}` |
| 算子编译器 | `/usr/local/Ascend/ascend-toolkit/latest/bin/opc` |
| 已生成的编译脚本 | `csrc/build/binary/ascend910_93/gen/SparseFlashMla-sparse_flash_mla-0.sh` |
| 参数文件 | `csrc/build/binary/ascend910_93/gen/SparseFlashMla_3c28573c…_param.json` |
| 源码副本（编译用） | `csrc/build/binary/ascend910_93/src/sparse_flash_mla/` |
| 运行时安装位置 | `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910_93/sparse_flash_mla/` |

### 3.2 可用的重编命令（★ 参数顺序很关键）

```bash
opc --soc_version=Ascend910_9391 \
    --main_func=sparse_flash_mla \
    --input_param=<...>/SparseFlashMla_<hash>_param.json \
    --output=<outdir> \
    --impl_mode=high_performance,optional \
    --simplified_key_mode=0 --op_mode=dynamic \
    <kernel-src-dir>            # ← 源目录必须放在**最后**
```

**踩坑记录**【实测】：`--soc_version` 若放在源目录**之后**，`opc` 会报
`[ERROR] check_input_params] Soc version is empty.` 且静默不产出；把它放**最前**
（并把源目录放最后）即成功产出 `.json`（26 KB）+ `.o`（2.1 MB）。

### 3.3 已验证的闭环

```
改 csrc/attention/sparse_flash_mla/op_kernel/arch22/*.h
  → 同步到 csrc/build/binary/ascend910_93/src/sparse_flash_mla/
  → opc（见上，产出 .json + .o）
  → 覆盖安装到 .../tbe/kernel/ascend910_93/sparse_flash_mla/
  → 重启服务 → 用 dump 重放 + tools/dcp_correctness.py 验证
```

**备份已就位**：`/tmp/kbak/`（现行二进制）、`/tmp/csrc_attn_bak.tar.gz`（源码树）。

### 3.4 尚未打通的两点【未确认】

1. `csrc/build.sh --ops=sparse_flash_mla --soc=ascend910_93 --opkernel` 会在 CMake
   重配置阶段失败（`symbol.cmake:253` 引用不存在的目标 `*_metadata_obj`；
   `ninja` 重配置还会去找已失效的 `/tmp/pip-build-env-*/…/cmake`）。
   ⇒ 绕开办法：**直接用 `opc`**（已验证），或修复该 CMake 目标。
2. 重编产物与现行二进制的 md5 不同（`.o` 2.1 MB vs 190 KB）——推测现行版是按
   "简化 key / 裁剪模板"编的。需要确认 `--simplified_key_mode` 等参数的取值，
   以保证行为一致；但这**不影响**"能改能编"的结论。

## 4. 要修什么（我们自己动手）

依据 `V41-CSA-KERNEL-SOURCE-ANALYSIS-20260930.md` 的定位，arch22 的 CSA 模板有三处要改：

| # | 位置 | 现在 | 改成 |
|---|---|---|---|
| 1 | `GetKeyGmOffset` + `CopyInSingleKv` | 索引越界/为负 ⇒ 静默丢键、**不计数** | 丢键要么计入"空洞并显式填充"，要么**不丢** |
| 2 | `CopyOutMrgeResult` → cube 读 | 只搬 `mte2Size` 行，cube 按 `actualSingleProcessSInnerSize` 读 ⇒ 读未写区 | 增加**实际条数握手**：cube 按 `min(实际, 期望)` 读 |
| 3 | `cmpMaskRight = cmpMaskS2Size − actS1Size` | 隐含 `cseq×ratio ≈ T`，DCP 分片下整体平移 | 让因果界**可由调用方显式给出**（或按本地坐标计算） |

**最小可验证的第一步**：只做 (2)——给两阶段加"实际条数"握手，让未写区域被显式跳过
（或填 −inf 分数）。用 `probes/replay_dump.py` 判据（`ALL_bit_identical=True`）复核，
再用 `tools/dcp_correctness.py --lengths 2000,8000,16000` 做端到端。

## 5. 与"交给算子团队"路线的区别

用户明确要求**我们自己修**。上面的链路验证表明这不依赖任何外部方：
* 源码：官方开源，A2/A3 支持；
* 工具链与构建缓存：容器内已有；
* 判据：单卡 dump 重放（20 秒）+ 端到端长针，都已就绪。

⇒ 下一步直接进入**改代码 → 重编 → 单卡/端到端验证**的循环。
