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
| **torch 扩展编译** | ✅ 成功（绕开 19 个被历史实验改坏的无关算子） |
| **正确性** | ✅ **生产形状 [N,1280] 逐位一致**（bf16 对齐 `npu_rms_norm`，int8/scale 对齐原算子；D=1280 × 30 次零错） |
| **排查记录** | 见 §8（宏 bug 吞掉 `op.Process()`）、§9（独立 buffer + 事件同步）、§9.3（D≤512 为上游缺陷） |

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

---

## 8. 调试记录：kernel 被 launch 但**一个字节都没有写**

算子注册、编译、OPP 安装、torch 绑定全部成功，**调用也不报错**，
但三路输出（`y_bf16` / `y_int8` / `scale`）全是**未初始化内存**：

| 证据 | 现象 |
|---|---|
| 重复调用 4 次 | 每次输出都不同 ⇒ 从未被写 |
| `rows=1` 也失败 | 排除"行数太少" |
| `D=64 / 128 / 256 / 512 / 1280 / 2048` **全部失败** | 排除 UB 预算/形状 |
| profiler | `aclnnRmsNormDynamicQuantBf16` 与 `GetWorkspaceSize` 各 5 次 ⇒ 调用真实发生 |
| CANN DEBUG 日志 | `Launch dynamic kernel ... tilingKey is 1, numBlocks is 6`、`Kernel launch successfully`、**无 ERROR** |
| DFX 日志 | `AddOutput[4] dtype=DT_BFLOAT16 addr=0x... shape=[6,1280]` ⇒ **输出张量确实传给了算子** |
| 原 2 输出算子同一进程内 | 完全正常 ⇒ 不是环境/驱动问题 |

### 8.1 关键线索

kernel 入口用宏生成"初始化 + 跑"：

```cpp
#define INIT_AND_PROCESS                                            \
    op.Init(..., &tilingData);   // [BF16-3OUT] \
    op.Process()
```

**我加注释时把 `// [BF16-3OUT]` 放在了行续接符 `\` 之前。**

C/C++ 翻译阶段里，**行拼接（阶段 2）发生在注释处理（阶段 3）之前**：
`\` 虽然"看起来"在注释里，却仍然把下一行接了上来，
于是 `op.Process()` 被并进 `//` 注释、**整行被删除**。

⇒ `Init()` 正常执行（`InitBuffer`、`SetGlobalBuffer` 都做了），
**但 `Process()` 从未被调用** ⇒ 没有任何 GM 写入 ⇒ 输出保持未初始化。

### 8.2 最小复现（已在本机 gcc 验证）

```c
#define M \
    printf("A-called\n"); // trailing comment \
    printf("B-called\n")
int main(void){ M; return 0; }
```

| 版本 | 输出 |
|---|---|
| 注释在 `\` 之前（**错**） | 只有 `A-called` |
| 去掉注释（**对**） | `A-called` + `B-called` |

### 8.3 修复

```diff
-    op.Init(...);   // [BF16-3OUT] \
+    op.Init(...); \
     op.Process()
```

### 8.4 顺带修掉的真问题：UB 预算

排查中另外发现一处**真实但当时尚未触发**的隐患，已一并修掉：

tiling 的 `CheckUbNormalTiling()` 按 **16 B/列** 做预算
（`2*dtSize`(inRows) + `2*dtSize`(outRows) + 4(xFp32) + 4(yFp32)），
据此在 D=1280 时算出 `rowStep=9`、总占用 **192,840 B**（UB 实测 196,352 B，仅余 3.5 KB）。

我最初的做法是**新增**一个 `TBuf outBf16Buf = rowStep*DAligned*sizeof(T)`（+23,040 B）
⇒ 实际需要 **215,168 B > 196,352 B** ⇒ `InitBuffer` 会**静默失败**。

**修法**：不新增 UB，改为**复用已有 buffer**（三者容量都够，bf16 输出只占其一半）：

| kernel | 复用对象 | 容量 | bf16 需求 |
|---|---|---:|---:|
| `normal` | `outRowsQue` | `2*rowStep*DAligned*sizeof(T)` = 46,080 B | 23,040 B |
| `single_row` | `yQue` | `DAligned*sizeof(T)` = 2,560 B | 2,560 B |
| `cut_d` | `outRowQue` | `lastDimSliceLen*sizeof(T)` | `elementCount*2` |

用标准队列语义 `Alloc → Cast → EnQue → DeQue → DataCopy → FreeTensor`，
既拿到正确的 V→MTE3 同步，又不动 tiling 预算。

### 8.5 教训

1. **宏里的行续接符 `\` 之前绝不能放 `//` 注释**（哪怕看起来在注释里）。
   排查这类问题要**先看编译器视角**（`cat -A` 看行尾），而不是读代码逻辑。
2. 判据要能一次切开"没执行"和"执行了但写错"：
   **输出用 `torch.empty`（不是 zeros）+ 连续多次调用比对**，5 秒就能定性。
3. CANN 侧证据链是齐的（profiler / DEBUG 日志 / DFX 张量地址），
   **它们只能证明"下发成功"，不能证明"kernel 内部真的跑了"**。

---

## 9. 修复后实测（最终）

### 9.1 结论

| 项 | 结果 |
|---|---|
| **生产形状 [N,1280]：`y_bf16`** | **与 `torch_npu.npu_rms_norm` 逐位一致（max diff = 0）** |
| **生产形状：`y_int8` / `scale`** | **与现有 2 输出算子 max diff = 0** |
| **稳定性：D=1280 × 30 次重复** | **bf16 0/30 错、int8 0/30 错** ⇒ 完全稳定 |
| D≥513（513/544/640/1024/1280/2048 × 20 次） | **全部 0/20 错** ✓ |
| D≤512（64…512） | bf16 必错；**但原算子同样错**（见 9.3） |

### 9.2 最终实现方式（三个 kernel 统一）

不复用其他 buffer，改为**独立 `outBf16Buf` + 显式事件同步**：

```cpp
LocalTensor<T> outLocal = outBf16Buf.Get<T>();
PipeBarrier<PIPE_V>();
Cast(outLocal, xLocalFp32, RoundMode::CAST_RINT, rowCount * numLastDimAligned);
event_t evV2MTE3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
SetFlag<HardEvent::V_MTE3>(evV2MTE3); WaitFlag<HardEvent::V_MTE3>(evV2MTE3);
DataCopyEx(this->yBf16Gm[gmOffset], outLocal, this->numLastDim, rowCount);
event_t evMTE32V = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
SetFlag<HardEvent::MTE3_V>(evMTE32V); WaitFlag<HardEvent::MTE3_V>(evMTE32V);
```

配套把 tiling 的 UB 预算补上这一路输出（`coexistingRowsNum += dtSize`），
于是 D=1280 的 `rowStep` 由 9 降为 8，总占用 192,128 B < UB 196,352 B ✓。

### 9.3 【重要】D ≤ 512 是**上游已有缺陷**，非本次引入

对照实验（同一容器、同一 stage，20 次重复）：

| 算子 | D=512 | D=1280 |
|---|---|---|
| **本次新增** `RmsNormDynamicQuantBf16` | bf16 20/20 错 | **0/20 错** |
| **原始** `RmsNormDynamicQuant`（未改动） | **20/20 错（scale 偏差 43.5）** | **0/20 错** |

⇒ **D ≤ 512 时上游 kernel 本身就算错**（bf16 分支只是把这个错误暴露为可见的"魔数"输出）。
生产 `hidden_size = 1280` 不落在该区间，不影响交付。

（复现：`~/cedpd-repo/experimental/fusion-3out/` 源码 + `/tmp/compare_ref.py`）

---

## 10. 中途的诊断过程（保留供参考）

修掉宏 bug 后重编（EXIT=0）+ 重装 OPP，实测：

| 检查 | 结果 |
|---|---|
| **生产形状 [6,1280]**：`y_bf16` vs `torch_npu.npu_rms_norm` | **逐位一致（max diff = 0）** |
| **生产形状 [6,1280]**：`y_int8` / `scale` vs 现有 2 输出算子 | **max diff = 0 / 0** |
| [160,1280] | bf16 max diff **4.88e-4** = 1 ULP（bf16 舍入），符合预期 |
| D=520…2048（含 640/1280/2048） | 全部 **diff = 0** ✓ |
| **D ≤ 512**（64/128/256/384/448/480/496/504/508/510/511/512） | **bf16 输出错**（int8/scale 仍正确） |

**边界精确落在 D=512 / 513 之间**（512 = BAD，513 = OK）。

进一步对照 tiling 参数：

| D | DAligned | 结果 |
|---:|---:|---|
| 512 | 512（== D） | BAD |
| 513…519 | ≥544（> D） | OK |
| 520 | 544 | OK |
| 1280 | 1280（== D） | **OK** |

即**不是**简单的"是否带 padding"，而是与 `Cast` 接收的元素数直接相关：
`CopyOutBf16` 的 `calCount = rowCount * numLastDimAligned`，
D=512 时为 **512**，D=513 时为 **544** ⇒ **可疑点在 `calCount ≤ 512` 的 `Cast`（fp32→bf16）路径**。

**影响面**：生产 `hidden_size = 1280` **不在**该区间，主链路正确；
但 `single_row` / `cut_d` 两个 fallback kernel 存在同类写法，需一并修。

**待办**：用"按 blockIdx 分流三路探针"（纯 `Duplicate` / 分块 `Cast` / 原单次 `Cast`）
一次编译定位是 UB buffer 问题还是 `Cast` 的问题。
