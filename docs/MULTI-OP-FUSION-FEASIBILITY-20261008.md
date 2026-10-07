# 多算子融合算子的可行性：基建已核实，只差"没人写"（2026-10-08）

> 触发：用户提出「我认为可写多算子融合的算子」。本文核实**我们能不能自己写**，
> 以及**写了值多少**。全部【实测·代码】。

---

## 0. 一页纸

| 问题 | 答案 |
|---|---|
| 多输出算子在这套基建下可行吗 | ✅ **完全可行**——`rms_norm_dynamic_quant` **本身就是 4 输出**（y1/y2 int8 + scale1/scale2） |
| 有没有"多输出 norm"的先例 | ✅ **`rms_norm_cast` 输出 `(bf16, fp32)`** |
| 源码完整吗 | ✅ kernel + op_host + proto + def + CMakeLists **全在** `csrc/attention/rms_norm_dynamic_quant/` |
| 新算子怎么进构建 | ✅ **目录自动发现**（`GLOB` + `op_add_subdirectory`），建目录即可 |
| 构建入口 | `csrc/build_aclnn.sh <ROOT_DIR> <SOC_VERSION>` |
| 为什么现在没有 3 输出 | ❌ **不是限制，是没人写**——不是"不支持" |

**⇒ 结论：可以写。下一步是 `RmsNormDynamicQuant3Out`：`x, gamma → (bf16, int8, scale)`。**

---

## 1. 已完成的（线① 的意外收获）

原假设（`_is_w8a8_dynamic` 对 W4A8 判假）**是错的**，但排查过程中发现并修复了一个**真实的融合缺口**：

| 项 | 值 |
|---|---|
| 改动 | `dsa_v41.py::multistream_preprocess`：非 index_source 层用融合算子 |
| **端到端** | **−0.149 ms/步 = −0.62%**（23.913 vs 24.062，n=280/272） |
| kernel 验证 | `RmsNormDynamicQuant` **3 → 35/步**；`RmsNorm` **140 → 108/步** |
| 落点 | **主流 s146**（关键路径）✓ |

---

## 2. 线②③ 的负结果（关闭）

| 线 | 结论 | 关键数 |
|---|---|---|
| **②** `hc_fn` L2 驻留 | ❌ **无杠杆** | MTE 有效带宽 **188 GB/s = 峰值 11.6%**（延迟受限）；hc_fn 仅占每步字节 **0.2%** |
| **③** 展开算子折进 HcPre | ❌ **量级过小** | 实测 **0.05 ms/步**（预估 2.5 ms，**高估 50×**） |

---

## 3. ★ 基建核实（这是"能写"的证据）

### 3.1 多输出已被这套基建支持

`rms_norm_dynamic_quant` 的 proto：

```cpp
REG_OP(RmsNormDynamicQuant)
    .INPUT(x, ...) .INPUT(gamma, ...)
    .OPTIONAL_INPUT(smooth_scale1, ...) .OPTIONAL_INPUT(smooth_scale2, ...)
    .OPTIONAL_INPUT(beta, ...)
    .OUTPUT(y1, ...) .OUTPUT(y2, ...)          // ← 两个 int8 输出
    .OUTPUT(scale1, ...) .OUTPUT(scale2, ...)  // ← 两个 scale
    .ATTR(epsilon, Float, 1e-06)
    .ATTR(output_mask, ListBool, {})
    .ATTR(dst_type, Int, 2)
```

**⇒ 4 输出，且带 `output_mask` 属性** ⇒ 这套框架**原生支持可选/多路输出**。

另一个先例 `rms_norm_cast` 输出 `(bf16, fp32)` —— 正是"多输出 RmsNorm"。

### 3.2 源码与构建链完整

| 组件 | 位置 |
|---|---|
| kernel | `csrc/attention/rms_norm_dynamic_quant/op_kernel/`（42 行 .cpp + ~1500 行头） |
| host tiling | `.../op_host/rms_norm_dynamic_quant_tiling.cpp` |
| proto | `.../op_host/rms_norm_dynamic_quant_proto.cpp` |
| def | `.../op_host/rms_norm_dynamic_quant_def.cpp` |
| 构建 | `.../CMakeLists.txt`（**目录自动发现**） |
| 入口 | `csrc/build_aclnn.sh` |
| aclnn API | `op_api/include/aclnnop/aclnn_rms_norm_dynamic_quant.h` |
| torch 绑定 | `csrc/torch_binding.cpp:2616` |

**⇒ 端到端（proto → host → kernel → aclnn → torch）都有模板可照抄。**

---

## 4. 建议写的算子：`RmsNormDynamicQuant3Out`

```cpp
// 目标签名
npu_rms_norm_dynamic_quant_3out(x, gamma, epsilon) -> (y_bf16, y_int8, scale)
```

**为什么是这个**：

| 理由 | 依据 |
|---|---|
| 解决**唯一的**融合阻塞 | `input_layernorm` 的 bf16 输出被 indexer/compressor 消费 ⇒ 2 输出算子不够 |
| 有现成模板 | 内部**必然**先算出 bf16 再量化 ⇒ 加一路 store 即可 |
| 覆盖面最大 | 43 层（含 8 个 index_source 层） |

**收益**：

| 项 | 计算 | 值 |
|---|---|---|
| `input_layernorm + wq_a.quantize`（43 层） | 43 × (7.89 − ~4.5) µs | **≈0.51%** |
| `q_norm` 的 8 个 index_source 层 | 8 × 4.24 µs | **≈0.13%** |
| **合计** | | **≈0.64%** |

---

## 5. 实现路径（分四步，每步可验证）

| # | 步骤 | 验证 |
|---:|---|---|
| 1 | 复制 `rms_norm_dynamic_quant/` → 新目录，改名 + 加第三输出 | `build_aclnn.sh` 编译通过 |
| 2 | 在 kernel 的量化前插入 bf16 store | 单算子 bench：输出 bf16 逐位等于 `rms_norm` 的结果 |
| 3 | 加 aclnn API + torch 绑定 | `torch.ops._C_ascend.npu_rms_norm_dynamic_quant_3out` 可调用 |
| 4 | 接到 `input_layernorm` + attention | 端到端 ≥0.5%，四道门 |

---

## 6. 复现

```bash
# 基建核实
ssh a3-21 'docker exec dsv41-tp8k5 cat /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/\
vendors/custom_transformer/op_proto/inc/rms_norm_dynamic_quant_proto.h'
ssh a3-21 'docker exec dsv41-tp8k5 find /vllm-workspace/vllm-ascend/csrc/attention/\
rms_norm_dynamic_quant -type f'
# 线① 的成果
ssh a3-21 'docker exec dsv41-tp8k5 python3 /tmp/kc.py'   # kernel 计数 A/B
```
