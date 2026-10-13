# 发布 payload 清单（**唯一事实源**）

本文件定义「A3 单机 TP8 解码融合优化包」里装什么、装到哪、由谁生成。
`build_payload.sh` / `install.sh` / `verify_consistency.sh` 都从本文件派生 ⇒ 三者天然一致。

> 与 `deploy/a3-ced-pd/` 的关系：那个是 **PD 分离 + CED 部署形态**；
> 本包是**叠加在它（或任意 TP8 单实例）之上的解码路径优化**，两者可同时使用，
> 本包只新增算子与两处 Python 融合，不改变部署形态。

---

## A. 二进制件（本包的核心，来自 `experimental/fusion-3out/` 的构建产物）

| # | 包内路径 | 容器目标路径 | 字节数 | sha256 |
|---|---|---:|---|
| A1 | `opp/` （整树，84 文件） | `/vllm-workspace/3out_opp/` | 2.7 MB | 见 `MANIFEST.sha256` |
| A2 | `so/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so` | `/vllm-workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so` | 984104 | `647d08c2b81ebd16dcb64c19c544085eb4f49f099f158fef71f1ab0aedbd1456` |

* **A1** 是自研算子 `RmsNormDynamicQuantBf16` 的 OPP 安装树（ASCENDC kernel 二进制 +
  aclnn 头 + `libcust_opapi.so` + tiling）。基础镜像的 `opp/vendors/` 是**空的**，
  所以这是**纯增量**，不会覆盖任何官方算子。
  ⚠️ 树里同时含 `rms_norm_dynamic_quant`（上游同名算子的重编副本，作为构建副产物）——
  它**不改变**上游行为，但使本包对该算子也有"最后一手"，记录在案。
* **A2** 是重编的 torch 扩展，比原版仅多注册一个 op（`npu_rms_norm_dynamic_quant_bf16`）。
  原版备份为 `...so.orig-<HHMMSS>`，回滚即换回。

## B. Python 件（**整文件覆盖**，非现场打补丁）

| # | 包内路径 | 容器目标路径 | sha256 |
|---|---|---|---|
| B1 | `py/model.py` | `/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v4/model.py` | `a4252d3e554891fff9a5fed77087ef16c86721359df9a923af9d28fb99ea0fab` |
| B2 | `py/dsa_v41.py` | `/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py` | `173c731d748d14b6d22195d7f94d7d0bad7a3802bcc00b1b7bb89b2172c46052` |

> **为什么是整文件而不是 `.patch`**：本仓踩过"我以为打了补丁、其实是旧版"。
> 整文件 + sha256 ⇒ 容器里那一份**就是**包里这一份，可逐字节证明。
> 现场改代码请走 `mount` 形态（见 README §3.2）。

## C. 环境变量（**不烘进文件，由起服环境提供**）

| 变量 | 本包要求 | 作用 | 不设的后果 |
|---|---|---|---|
| `ASCEND_CUSTOM_OPP_PATH` | `/vllm-workspace/3out_opp/vendors/custom_transformer` | 让 aclnn 找到自研算子 | 融合点 B 调用时找不到算子 → 报错或静默回退 |
| `V41_LNORM_FUSE` | `1` | 打开融合点 B（`input_layernorm`+`wq_a.quantize`） | 走原路径，**无收益、不报错** |
| `V41_QNORM_FUSE` | `1` | 融合点 A（`q_norm`+`wq_b.quantize`）；用上游 2 输出算子，非本包新增 | 走原路径，无收益 |

开关的完整语义、默认值、影响面见 [`SWITCHES.md`](SWITCHES.md)。

## D. 与 `V41_CED_*` 的关系

本包**不读不写**任何 `V41_CED_*` 变量，与 PD 分离 / CED 形态正交。
在 CED 形态下，融合点 B 落在 **D（decode）侧**的 43 个 decoder layer 上，P 侧不受影响。

## E. ★ 兼容性：三形态的差异（2026-10-08 对齐用）

用户报告正在 A2 与 A3 上部署发布版 ⇒ 本节列出**必须知道的三个差异**。

### E.1 `deepseek_v4/model.py` 是三形态共用的

| | |
|---|---|
| `model_type` | `deepseek_v41`（配置里就是这个） |
| 入口 | `models/deepseek_v41/model.py` |
| **它 import** | `from vllm_ascend.models.deepseek_v4.model import (…, DeepseekV2DecoderLayer, DeepseekV4Attention, …)` |

⇒ 融合补丁打在 **`models/deepseek_v4/model.py`**，三形态都吃得到；
**不要**去打 `models/deepseek_v41/model.py`（那是 `patches/files/model.py` 覆盖的那份，
里面没有 `DeepseekV2DecoderLayer`）。

### E.2 `attention/dsa_v41.py` 各形态**不同** ⇒ 禁止整文件覆盖

| 形态 | 该文件 | 整文件覆盖的后果 |
|---|---|---|
| A3 TP8 单实例 | 容器原版 | ✅ 安全 |
| **A3 CED-PD** | **CED 定制版**（含 `[CED-SWA-CLIP]` 等约 168 行） | ❌ 抹掉 CED 修复 |
| A2 | 容器原版 | ✅ 安全（但二进制侧另有问题，见 E.3） |

⇒ 改用 `integration/apply_lnorm_fuse_portable.py`（锚点驱动，只动两处，已验证
CED 版的锚点在第 502–503 行存在）。

### E.3 ⛔ A2 暂不可用：OPP 只编了 `ascend910_93`

```
op_impl/ai_core/tbe/kernel/ascend910_93/    ← 只有这一个 SoC
```

算子 def 声明支持 `ascend910b` + `ascend910_93` 两代，但本次编译只传了 `--soc=ascend910_93`
⇒ 在 A2（910B3）上算子找不到 kernel 二进制。
**在补上 910b 的 kernel 之前，A2 不要启用 `V41_LNORM_FUSE=1`。**
重编步骤见 [`REPRODUCE.md`](REPRODUCE.md) §2。

### E.4 一句话对齐表

| 形态 | `deepseek_v4/model.py` | `dsa_v41.py` | OPP kernel | 能否用本包 |
|---|---|---|---|---|
| A3 TP8 单实例 | 共用 | 容器原版 | ✅ 910_93 | ✅ **已实测 −0.90%** |
| A3 CED-PD | 共用 | **CED 定制版** | ✅ 910_93 | ⚠️ **须用 portable 补丁**（否则抹掉 CED 修复） |
| A2 8×910B3 | 共用 | 容器原版 | ❌ 无 910b | ❌ **需先重编 kernel** |
