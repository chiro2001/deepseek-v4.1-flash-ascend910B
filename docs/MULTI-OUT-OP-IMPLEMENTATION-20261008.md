# 自研多输出算子：`RmsNormDynamicQuantBf16` 实现记录（2026-10-08）

> 目标（用户指示）：**"我认为可写多算子融合的算子"**。
> 本文记录**从零写一个 Ascend 自定义算子**的完整过程与结果。
> 全部【实测】。

---

## 0. 一页纸

| 阶段 | 结果 |
|---|---|
| **基建核实** | ✅ 多输出被支持（`rms_norm_dynamic_quant` 本身即 4 输出 + `output_mask` 属性） |
| **源码编写** | ✅ 10 个文件改动（改名 + 加第三输出 bf16） |
| **kernel 编译** | ✅ **EXIT=0**（9 分 27 秒），产物 + `.run` 安装包 |
| **OPP 安装** | ✅ kernel + aclnn 头文件就位 |
| **torch 绑定** | ✅ 代码已写（实现 + 注册） |
| **torch 扩展编译** | ❌ 被**环境污染**阻塞（历史实验遗留，非本改动） |

---

## 1. 为什么写这个算子

`input_layernorm` + `wq_a.quantize` 的融合被 **bf16 下游消费者**阻断：

| 消费者 | 用途 |
|---|---|
| `compressor`（4 层） | `compressor(hidden_states)`、`compressor.wkv(...)` |
| `indexer`（8 层） | `attn.indexer.select(hidden_states, qr, ...)` |

而现有算子**全是 2 输出**：

| 算子 | 输出 |
|---|---|
| `npu_rms_norm_dynamic_quant` | `(int8, scale)` |
| `npu_rms_norm_cast` | `(bf16, fp32)` |
| `npu_rms_norm_quant` | `(int8)` |

⇒ 需要 **(bf16, int8, scale)** 三输出。

---

## 2. 实现（10 个文件）

### 2.1 命名：避开数字+大写的 snake 转换歧义

**重要教训**：`RmsNormDynamicQuant3Out` 被 CANN 转成 `rms_norm_dynamic_quant3_out`
（数字与大写的边界判定），而构建期望 `rms_norm_dynamic_quant_3out` ⇒ **签名不匹配**。

⇒ **改用无数字名**：`RmsNormDynamicQuantBf16` → `rms_norm_dynamic_quant_bf16` ✓

### 2.2 改动清单

| # | 文件 | 改动 |
|---:|---|---|
| 1 | `op_host/*_proto.cpp` | `.OUTPUT(y_bf16, TensorType({DT_BF16...}))` |
| 2 | `op_host/*_def.cpp` | `this->Output("y_bf16").ParamType(OPTIONAL).DataType({DT_BF16...})` |
| 3 | `op_kernel/*_base.h` | `GlobalTensor<T> yBf16Gm` + `bool hasBf16Out` + `InitOutGlobalTensors(..., yBf16)` |
| 4 | `op_kernel/*.cpp` | 入口加 `GM_ADDR yBf16` + 宏内 `op.Init(..., yBf16, ...)` |
| 5 | `op_kernel/*_normal_kernel.h` | Init 签名 + InitBuffer + **CopyOutBf16** + Process 两处调用 |
| 6 | `op_kernel/*_single_row_kernel.h` | 签名 + InitOut（编译需要） |
| 7 | `op_kernel/*_cut_d_kernel.h` | 签名 + InitOut（编译需要） |
| 8 | `build_aclnn.sh` | A3 清单加 `"rms_norm_dynamic_quant_bf16"` |
| 9 | `torch_binding.cpp` | 实现 `npu_rms_norm_dynamic_quant_bf16_npu` + 注册 |
| 10 | `op_host/*_tiling.cpp` | **符号去重**：4 个自由函数加 `static` |

### 2.3 核心 kernel 逻辑

```cpp
__aicore__ inline void CopyOutBf16(int32_t gmOffset, int32_t rowCount)   // [BF16-3OUT]
{
    if (!this->hasBf16Out) { return; }
    LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();   // ← norm 结果（本就已算出）
    LocalTensor<T> outLocal = outBf16Buf.Get<T>();
    PipeBarrier<PIPE_V>();
    Cast(outLocal, xLocalFp32, RoundMode::CAST_RINT, rowCount * this->numLastDimAligned);
    event_t evBf = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(evBf);
    WaitFlag<HardEvent::V_MTE3>(evBf);
    DataCopyEx(this->yBf16Gm[gmOffset], outLocal, this->numLastDim, rowCount);
}
```

**关键洞察**：量化前**本来就已算出 fp32 的 norm 结果**（`xBufFp32`），
所以加 bf16 输出只需**一次 Cast + 一次 DataCopy**，不需要重算 norm。

**UB 余量**（目标形状 D=1280, M=6）：现有 ~128 KB + bf16 buffer 15 KB = **146 KB < 192 KB** ✓

### 2.4 torch 绑定：动态符号查找，无需头文件

```cpp
EXEC_NPU_CMD(aclnnRmsNormDynamicQuantBf16, x, gamma, smooth_scale, smooth_scale2, beta,
             epsilon, output_mask, dst_type, y_out, y2_out, scale_out, scale2_out, y_bf16_out);
```

`EXEC_NPU_CMD` 用 `GetOpApiFuncAddr(#aclnn_api)` **动态查找**符号 ⇒
**不需 include aclnn 头**，只要库里有符号 ✓

---

## 3. 编译验证（关键成果）

| 项 | 值 |
|---|---|
| 命令 | `build.sh --pkg --ops='rms_norm_dynamic_quant,rms_norm_dynamic_quant_bf16' --soc=ascend910_93` |
| 结果 | **EXIT=0**，错误 0 |
| 耗时 | **9 分 27 秒** |
| 产物 | `RmsNormDynamicQuantBf16_2a7b4102...o`（hash 随源码变化 ✓）+ relocatable + json |
| OPP 包 | `cann-ops-transformer-custom_linux-aarch64.run` |
| 安装 | kernel 进 `.../kernel/ascend910_93/rms_norm_dynamic_quant_bf16/`，aclnn 头生成 ✓ |

### 3.1 途中的三个障碍（均已解决）

| # | 现象 | 根因 | 修法 |
|---:|---|---|---|
| 1 | `CMake Error: target 不存在`（4 秒失败） | **817 个历史实验备份目录**（`.bak*`/`.armA_staged`）被 GLOB 当算子扫入 | 全部移出 csrc 树 |
| 2 | `No such file: autogen/aclnn_..._3out.cpp` | `3Out` 的 snake 转换歧义 | 改名为 `Bf16` |
| 3 | `multiple definition of optiling::GetworkspaceRowsNum` | 复制的 tiling.cpp 与原算子符号重名 | 4 个自由函数加 `static` |

---

## 4. 未完成的一步：torch 扩展编译

`python3 setup.py build_ext --inplace` **失败**，但根因是**环境污染**：

```
FAILED: kv_quant_sparse_flash_attention  (Invalid json file content)
FAILED: grouped_matmul_swiglu_quant_v2   (E80003)
... 共 19 个 FAILED
```

**这些算子与本次改动无关**，是历史实验改坏了它们的编译配置。
即使临时移走这两个算子，仍有 19 个失败 ⇒ 环境需要重建。

**下一步选项**：
1. 在**干净容器**（如 tp8k5，其 csrc 经查 0 污染）编译 torch 扩展
2. 或重建 op-hcfuse 容器的 csrc 树

---

## 5. 产物（已入仓）

| 路径 | 内容 |
|---|---|
| `experimental/fusion-3out/rms_norm_dynamic_quant_bf16/` | **新算子完整源码**（13 文件） |
| `experimental/fusion-3out/rms_norm_dynamic_quant_orig/` | 原算子（对照基准） |
| `experimental/fusion-3out/torch_binding.cpp.patched` | 含新绑定的 torch_binding.cpp |
| `experimental/fusion-3out/build_aclnn.sh.patched` | 含新算子清单的构建脚本 |

---

## 6. 预期收益（若完成集成）

| 融合点 | 覆盖 | 可省 |
|---|---|---:|
| `input_layernorm` + `wq_a.quantize` | 43 层 | **≈0.51%** |
| `q_norm` 的 8 个 index_source 层 | 8 层 | **≈0.13%** |
| **合计** | | **≈0.64%** |

（对照：已完成并验证的 `q_norm` 融合（35 层，用现有 2 输出算子）= **0.62%**）

---

## 7. 复现

```bash
# 部署源码
ssh a3-21 'bash /tmp/inst_bf16.sh'          # 安装算子 + 改 build_aclnn.sh
# 编译
ssh a3-21 'docker exec dsv41-op-hcfuse bash -lc \
  "cd /vllm-workspace/vllm-ascend/csrc && bash build.sh --pkg \
   --ops=\"rms_norm_dynamic_quant,rms_norm_dynamic_quant_bf16\" --soc=ascend910_93"'
# 安装 OPP
ssh a3-21 'bash /tmp/install_stage.sh'
# torch 绑定（待干净环境）
ssh a3-21 'python3 /tmp/add_binding.py'
```
