# 我们自己的 int4 量化误差有多大（在**真实权重**上实测）

> 2026-09-21 23:2x。起因：上一轮我用了一个"权重 INT4 = 11.7%"的锚点，
> 用户指出 **V4.1-Flash 本身可能就是 4-bit 权重**，那个锚点可能拿错了。
> **核对结论：用户方向对，但格式要修正一处** —— 官方是 **FP8 主体 + MXFP4 专家**，
> 不是整机 NVFP4。本文全部数字在**真实权重**上实测（A3，只读，未动模型）。

---

## 一、先修正前提：官方到底是什么格式（实测）

### 1.1 config 层面

```console
$ python3 -c "import json;c=json.load(open('DeepSeek-V4.1-Flash/config.json'));print(c['dtype'], c['expert_dtype'])"
fp8 fp4
```

### 1.2 张量层面（这才是硬证据）

我手写 safetensors 头解析 + FP4 解码（不依赖任何库）读出：

| 张量 | dtype | shape | 说明 |
|---|---|---|---|
| `layers.0.ffn.experts.0.w1.weight` | **I8** | `[2304, 2560]` | 打包的 4-bit；真实形状 `[2304, 5120]` |
| `layers.0.ffn.experts.0.w1.scale` | **F8_E8M0** | `[2304, 160]` | **纯指数缩放**，5120/160 = **block 32** |
| `layers.0.ffn.experts.0.w2.weight` | I8 | `[5120, 1152]` | 真实 `[5120, 2304]` |
| `layers.0.ffn.experts.0.w2.scale` | F8_E8M0 | `[5120, 72]` | 2304/72 = **block 32** |

⇒ ★ **这是 MXFP4**（E2M1 尾数 + **E8M0 幂次 scale** + **block 32**），
**不是 NVFP4**（NVFP4 用 FP8-E4M3 scale + block 16）。

⇒ 准确表述：**DeepSeek-V4.1-Flash = FP8 主体 + MXFP4 专家层**。
所以"我们保持 4-bit"是对的决策，但**官方那个 4-bit 只覆盖专家**；
其余（attn/共享专家等）官方是 FP8，我们把它升到 **W8A8**。

---

## 二、我们自己的格式（实测）

同一路径读我们的导出：

| 张量 | dtype | shape |
|---|---|---|
| `...experts.0.w1.weight` | I8 | `[1152, 5120]` = `[2304/2, 5120]` ⇒ **沿输出通道打包** |
| `...experts.0.w1.weight_scale` | **F32** | **`[2304, 1]`** ⇒ **per-channel（每行 1 个）** |
| `...experts.0.w1.weight_offset` | F32 | `[2304, 1]`，**min=max=0** ⇒ 对称量化成立 |
| `...experts.0.w1.scale_bias` | F32 | `[2304, 1]`（smooth/AWQ 侧） |

配方（`quant/patches/msmodelslim_v41_w4a8.patch` + `REPRO_W4A8_QUANT.md` §配方）：

```
路由专家 → linear_quant W4A8 : weight scope=per_channel, dtype=int4, symmetric=True, method=ssz
                               act    scope=per_token,   dtype=int8, symmetric=True, method=minmax
前期     → quarot(block_size=32) → flex_awq_ssz(up-down) → flex_smooth_quant(norm-linear)
```

**SSZ = Scan-Scale-Zero**（`msmodelslim/core/quantizer/impl/ssz.py`）：
迭代最小二乘搜最优 scale（对称分支 offset 固定 0，最多 50 轮，取 MSE 最小）。

### ★ 关键对比：**我们的粒度比官方粗 160×**

| | 官方 MXFP4 | **我们 W4A8** |
|---|---|---|
| 每行 scale 数（w1） | **160** | **1** |
| 粒度 | block **32** | **per-channel** |
| scale 精度 | E8M0（1 字节，power-of-2） | F32（4 字节） |

---

## 三、★ 实测误差（真实张量，格式本征）

方法：把官方 MXFP4 **反量化**成 float，作为该张量在当前模型里的真实取值；
然后在**同一个张量上**套不同量化方案，量 `‖W − Ŵ‖ / ‖W‖`。

### 3.1 `layers.0.ffn.experts.0.w1`（真实形状 `[2304, 5120]`，std 0.0219）

| 方案 | **rel_err** | cosine |
|---|---:|---:|
| **我们的配方（per-channel int4 + SSZ）** | **0.1643** | 0.9878 |
| per-channel int4 + 朴素 minmax | 0.1777 | 0.9859 |
| **官方粒度（block-32 int4）** | **0.1012** | 0.9952 |
| block-16 int4 | **0.0893** | 0.9962 |
| block-128 int4 | 0.1228 | 0.9926 |

### 3.2 `layers.0.ffn.experts.0.w2`（`[5120, 2304]`，std 0.0220）

| 方案 | **rel_err** | cosine |
|---|---:|---:|
| **我们的配方** | **0.1548** | 0.9890 |
| per-channel minmax | 0.1658 | 0.9873 |
| **block-32** | **0.1008** | 0.9952 |
| block-16 | 0.0891 | 0.9962 |
| block-128 | 0.1215 | 0.9927 |

### 3.3 三条可读的结论

1. **我们的 per-channel int4 ≈ 15.5–16.4%**（去掉 QuaRot/AWQ/smooth 的裸格式误差）；
2. **官方的 block-32 粒度只要 10.1%** ⇒ **我们的粒度粗，误差高约 1.55×**；
3. **SSZ 有贡献但不大**：比朴素 minmax 好 **7–8%**（0.164 vs 0.178）。

---

## 四、但"误差大"≠"模型变差"：端到端已经给了答案

【实测，来自发布包自己的记录】我们的 W4A8 产物：

| 验收项 | 结果 | 出处 |
|---|---|---|
| **GSM8K-200 × 3 次** | **198/200、199/200、197/200（98.5–99.5%）** | `dsv41-release/CORRECTNESS_STATUS.md` §6.2 |
| **Vision（官方 23 例）** | **23/23** | 同上 §6.1（5 个臂重复出现） |
| 单个开关的对照 | `HCCL_DETERMINISTIC=true` 会让 GSM8K 掉到 **91/100（Δ=−9）** | 同上 §6.1 |

⇒ **同一份权重下，"低精度固定序"这一个开关的影响（Δ=−9）远大于 int4 量化本身的痕迹。**
⇒ **int4 量化的端到端代价落在噪声里**（197–199/200 与无 vision 基线 97.5% 同档）。

**为什么 16% 的权重误差不致命**：① QuaRot/AWQ/smooth 把误差从"结构化"改成"近似白噪声"，
② Transformer 对白噪声型权重扰动有冗余，③ 逐 token 激活是 int8 动态量化的，
误差有部分抵消。

---

## 五、★ 一条可执行的改进：把专家层粒度改细

既然 block-32 在**同样 4-bit** 下误差低 1.55×，而我们的 scale 是 F32 per-channel，
**改成 block 粒度几乎不占容量**：

| 方案 | 每张量 scale 数(w1) | scale 字节 | 占权重(5.9 MB) | 实测 rel_err |
|---|---:|---:|---:|---:|
| **现状 per-channel** | 2,304 | 9 KB | **0.16%** | **0.164** |
| block-128 | 92,160 | 368 KB | 6.2% | 0.123 |
| **block-32** | 368,640 | 1.47 MB（F32）/ 0.74 MB（F16） | 25% / 12.5% | **0.101** |
| block-16 | 737,280 | 2.95 MB / 1.47 MB | 50% / 25% | 0.089 |

**推荐先试 block-128**：只多 **6%** 存储，误差从 0.164 → **0.123（−25%）**。
若愿意付 12.5%（scale 用 F16 的 block-32），误差 → **0.101（−38%）**。

⚠️ **代价与风险**：
* 专家层占权重的大头 ⇒ 总包体上涨（按专家占 ~70% 估，block-128 约 **+4%**，block-32/F16 约 **+9%**）；
* **Ascend 侧的反量化 kernel 要支持 block scale** —— 现在 `weight_scale` 是 `[N,1]`，
  改成 `[N, K/block]` 要确认运行时读得对（这一条**必须先在单卡上验**，见 §六）；
* 这是**重新量化**（要重跑 quant 流水线），不是改配置。

---

## 六、★ 一个必须核实的未确认项

我尝试**直接解码**我们自己导出的 int4 权重，做逐元素比对，**失败**：

| 检验 | 结果 |
|---|---|
| 整体 cosine（我们的解码 vs 官方反量化） | **+0.0116**（两种 nibble 排布都试过：adjacent +0.0116 / split −0.0002） |
| 逐行最优标量拟合后的残差 | 中位 **0.9999**（即拟合也救不回来） |
| **逐行范数比** | **0.1284**，且**极集中**（p10 0.1279 / p90 0.1291） |

**两个信号互相矛盾**：范数比集中得像"正交变换 + 全局标量"，但相关系数又几乎为 0。

可能的解释（**都未确认**）：
1. **QuaRot 旋转**（配方里确实有 `quarot block_size=32`）⇒ 导出权重不在原始基，
   逐元素比较**本来就无意义**；但纯正交旋转应保持行范数（比≈1），
   而我们测到的是 0.128 ⇒ 中间还有约 8× 的缩放（可能折进了 `scale_bias`）；
2. **导出布局不是普通的 `[N/2, K]` 行优先**（例如 Ascend NZ / fractal 布局），
   我按行优先解出来的 nibble 顺序不对 ⇒ 数值分布对、顺序错；
3. 专家索引在两份权重里不是同一顺序。

⇒ **这一条必须核实**，因为它决定"我们能不能在权重层面做误差分析"。
最小验证方案（在单卡上，10 分钟）：
**拿一个 QuaRot 不覆盖的张量**（配方里 quarot 明确排除 `*vision*` / `*engram*` / `mtp*`），
同样解码比对 —— 若那个能对上，就证明解码器没错、差异来自旋转；
若也对不上，就是布局问题。

---

## 七、复现

```bash
# 两个脚本都在 a2/scripts/，纯 numpy，不需要 NPU
python3 scripts/w4a8_error_probe.py --official <官方目录> --ours <我们的目录> \
        --keys layers.0.ffn.experts.0.w1,layers.0.ffn.experts.0.w2
python3 scripts/w4a8_true_error.py  --official <官方目录> --ours <我们的目录> \
        --keys layers.0.ffn.experts.0.w1
# 另有 scripts/st_meta.py（读 safetensors 头，不需要 safetensors 库）
```

三个脚本都**只读** safetensors 的指定字节区间（`seek` + `read`），不加载整个分片。

---

## 八、与上一轮 `logs/004` 的关系（**更正**）

`logs/004` §2.5 我写的"**权重 INT4 = 11.72%**，模型已经接受这个量级"这个锚点：

* **算法对**（同样的对称均匀量化 + 同样实现），
* **但对象错了**：那是**我自己合成的高斯矩阵**（std 0.02、group-128），
  不是真实权重，也不是我们的配方（per-channel SSZ）。

**更正后的锚点应该用本文的表**：真实专家张量上，**我们 per-channel SSZ = 15.5–16.4%**、
**官方 block-32 = 10.1%**。`logs/004` 的 KV 结论（INT8 安全、INT4 激进）**不受影响**，
因为那部分用的是 KV 自己的仿真，与权重锚点无关。
