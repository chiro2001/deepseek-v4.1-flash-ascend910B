# LNORM-FUSE 为什么一直没生效：三层根因（含 2 个可复用的坑）

> 2026-10-08。全部【实测】。结论：**`a3-tp8-fusion-v1` 声称的 −0.90% 是噪声**，
> 融合从未真正执行过。本文记录怎么发现的、三个根因、以及两个**可复用于任何
> 「给 vLLM 加自定义算子」场景**的坑。

---

## 0. 一句话

融合从未生效 → 我测到的「−0.90%」是噪声 → 地火口径的「两臂无差异」才是真相。
根因有**三层**，一层盖着一层，每层都能让融合**静默失效**或**炸启动**：

| # | 层 | 症状 | 修法 |
|---:|---|---|---|
| 1 | 补丁打错类 | **静默失效**（不报错、无收益） | 打 `deepseek_v41/model.py`，不是 `deepseek_v4/` |
| 2 | 缺 Meta 实现 | 启动炸：`Operator does not support running with fake tensors` | `torch_binding_meta.cpp` 补 Meta |
| 3 | 真张量当常量 | 启动炸：`Dynamo failed to run FX node ... merge_devices` | 直接传 `nn.Parameter`，别缓存普通属性 |

---

## 1. 【层1】补丁打错类 —— 静默失效

原补丁打在 `models/deepseek_v4/model.py::DeepseekV2DecoderLayer.forward`。
但 `model_type=deepseek_v41` 时实际用的是 v41 自己的层类：

```
models/deepseek_v41/model.py:890  class DeepseekV41Model(DeepseekV4Model)
models/deepseek_v41/model.py:893      decoder_layer_cls = DeepseekV41DecoderLayer
models/deepseek_v41/model.py:768  class DeepseekV41DecoderLayer(DeepseekV2DecoderLayer)
models/deepseek_v41/model.py:872      x = self.input_layernorm(x)   ← 真正跑的是这里
```

`DeepseekV41DecoderLayer` **override 了 `forward`**（签名还多一个 `pre_mix`）
⇒ 被改的 v4 那份**根本不会被调用**。

**判据（决定性）**：profiler 里数算子下发次数

```
aclnnRmsNormDynamicQuantBf16   0     ← 融合点 B，一次都没跑
aclnnRmsNormDynamicQuant     528     ← 融合点 A（正常）
aclnnRmsNorm                 3304
aclnnDynamicQuantV2          1639
```

> **可复用的教训**：给派生类体系打补丁时，**先确认哪一层真的被调用**。
> 判据不要用「没报错」「输出正常」——那两种失效都产出完全正常的输出。
> **用 profiler 数下发次数**，一跳就定性。

---

## 2. 【层2】缺 Meta 实现

补丁打对类之后，worker 起不来：

```
torch._dynamo.exc.Unsupported: Operator does not support running with fake tensors
```

**为什么**：vLLM 的 `torch.compile` 会在 profile run 阶段用 **fake tensor** 追踪
（`determine_available_memory → profile_run → _dummy_run → aot_compile`）。
没有 Meta/fake 实现的算子无法参与 fake 传播。

**为什么之前没暴露**：层1 让补丁从未被执行，所以这个算子**从没被 trace 到**。
两层是叠加的——修了层1 才看见层2。

**既有算子的做法**（实测对照，`op-hcfuse` 容器）：

```
npu_hc_post                      Meta=True    ← 有
npu_hc_pre_v2                    Meta=True
npu_rms_norm_dynamic_quant       Meta=True
npu_rms_norm_dynamic_quant_bf16  Meta=False   ← 我的新算子，漏了
```

**根因**：Meta 实现注册在**独立文件** `csrc/torch_binding_meta.cpp`
（不是 `torch_binding.cpp`）。我加算子时只改了后者（真实现 + schema），漏了前者。

**修法**（照抄同族算子的写法）：

```cpp
// torch_binding_meta.cpp
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_rms_norm_dynamic_quant_bf16_meta(
    const at::Tensor& x, const at::Tensor& gamma,
    const c10::optional<at::Tensor>& smooth_scale,
    const c10::optional<at::Tensor>& beta, double epsilon)
{
    auto options = x.options();
    at::Tensor y_bf16_out = at::empty_like(x, options.dtype(at::kBFloat16));
    at::Tensor y_out = at::empty_like(x, options.dtype(at::kChar));
    c10::SymDimVector scale_out_shape;
    for (size_t i = 0; i < x.dim() - 1; i++) scale_out_shape.push_back(x.sym_size(i));
    at::Tensor scale_out = at::empty_symint(scale_out_shape, options.dtype(at::kFloat));
    return std::make_tuple(y_bf16_out, y_out, scale_out);
}
// 注册（紧跟同族算子）
ops.impl("npu_rms_norm_dynamic_quant_bf16",
         &vllm_ascend::meta::npu_rms_norm_dynamic_quant_bf16_meta);
```

> **可复用的教训**：给 vllm-ascend 加**任何**自定义算子，必须改**三处**：
> ① `torch_binding.cpp`（schema + 真实现）② `torch_binding_meta.cpp`（Meta + 注册）
> ③ `csrc/attention/<op>/`（算子本体）。漏 ② 的代价是「能编过、能起服到一半、然后炸」。
>
> **自检命令**（不用跑服务）：
> ```python
> torch._C._dispatch_has_kernel_for_dispatch_key("_C_ascend::<op>", "Meta")   # 必须 True
> ```

---

## 3. 【层3】真张量被当常量 —— 最难的一个

修了层2 之后**还是炸**，但报错换了个算子：

```
fake_tensor.py:987 in merge_devices → raise RuntimeError
torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors:
    call_function _C_ascend.npu_hc_post(...)
```

**注意 `npu_hc_post` 是冤枉的**——它有 Meta（`Meta=True`，层2 里刚验过）。
`merge_devices` 报的是**同一个算子收到了不同设备的张量**。

**真凶**是补丁里这几行：

```python
_w = getattr(self, "_lnorm_w_bf16", None)      # 普通 Python 属性
if _w is None:
    _w = self.input_layernorm.weight.data.to(torch.bfloat16)   # ← 真·NPU 张量
    self._lnorm_w_bf16 = _w                    # ← 存进普通属性 = dynamo 眼里的"常量"
```

**机制**：

| 传什么 | dynamo 怎么看待 | 会被 fake 化吗 |
|---|---|---|
| `self.input_layernorm.weight`（`nn.Parameter`） | **模块状态** | ✅ 会 |
| `self._lnorm_w_bf16`（普通属性里的 Tensor） | **常量** | ❌ **不会** |

⇒ fake `x` 与真 `_w` 混用 ⇒ `merge_devices` 抛错。
（`.data` 还额外绕过了 Parameter 的追踪语义。）

**修法**：直接传 Parameter，**不转换、不缓存**：

```python
_normed, _q_i8, _q_sc = torch.ops._C_ascend.npu_rms_norm_dynamic_quant_bf16(
    x.contiguous(), self.input_layernorm.weight, epsilon=self.norm_eps
)
```

（`RMSNorm` 的 weight dtype 由 `torch.get_default_dtype()` 决定，vLLM 在 bf16 下已是 bf16
⇒ 不需要 `.to()`。见 `vllm/model_executor/layers/layernorm.py:37`。）

> **可复用的教训**：**在 `torch.compile` 区域内，不要用普通 Python 属性缓存张量
> 再喂给算子。** 那会让它退化成「常量」而跳过 fake 化。
> 要缓存就 `register_buffer`（它是模块状态）；能不缓存就别缓存。

---

## 4. 三层是叠加的，所以排查顺序很关键

```
层1（打错类）⇒ 补丁不执行 ⇒ 层2/层3 都不暴露 ⇒ 表现为"没收益、但不报错" ← 最难发现
    修层1
层2（缺 Meta）⇒ 启动炸 Unsupported
    修层2
层3（真张量常量）⇒ 启动炸 merge_devices（且报错点名的是**别的**算子）
    修层3
```

**如果一开始就报错，反而好查**。最坏的情形是层1 这种「静默失效」。

---

## 5. 对交付的影响（诚实声明）

| 项 | 状态 |
|---|---|
| 二进制面（OPP 算子 + torch 扩展） | ✅ 好的：生产形状 `D∈{1280,5120}` × 各 20 次三路输出零错 |
| Python 侧集成 | ✅ 三个根因已全部定位并修 |
| **端到端收益** | ⛔ **仍未验证**。之前声称的 −0.90% 已作废 |
| Release `a3-tp8-fusion-v1` | ⚠️ 已加「收益未验证」标注（标题 + 说明） |

**验证判据（修完必须过这两关）**：

1. profiler 里 `aclnnRmsNormDynamicQuantBf16` 下发次数 **> 0**
   （层1 的失效就是靠这一条抓出来的）
2. `bneck hp p50` 在**同一套固定协议**下测出的差 ≥ 0.5%
   （单臂噪声 ±0.6~1.3%，所以必须控制 prompt 与稳态窗口）

---

## 6. 性能实测结论（2026-10-08 03:5x）：**不是收益，判据不达标**

三层修完、服务健康、冒烟通过、算子确认在下发（433 次）之后，
用**同一套固定协议**（地火语料、固定 16 槽 + 固定 nonce、等空闲、3 轮取中位）实机测量：

| 并发 | A 臂（融合关） | B 臂（融合开） | 变化 |
|---:|---:|---:|---:|
| 1 | 25.091 ms | **28.445 ms** | **+13.4%（更差）** |
| 4 | 33.049 ms¹ | **33.417 ms** | +1.1%（≈持平） |
| 8 | 42.747 ms | **42.207 ms** | −1.3%（在噪声内） |

¹ conc=4 的 A 臂来自较早一次同协议测量（`dihuo_conc_ab.py`），口径略有差异，仅作参考。

**判据（每项端到端 ≥0.5%）⇒ 不达标。** conc=8 的 −1.3% 与 A 臂自身 ±0.554 ms 的离散同量级，
不构成可复现的收益；conc=1/4 反而是小幅回退。

**⇒ 结论：这个融合在 A3 生产形状（D=5120）下没有实测收益，不建议在交付里开启。**
可能的机制（【推断】，未逐项验证）：
* D=5120 时 tiling 的 `rowStep=1`（UB 一次只放得下 1 行 5120 元素）
  ⇒ 分块流水差；而原路径的 RMSNorm 与 DynamicQuant 各有更优的 tiling。
* 融合把「读 x 一次」省下来的收益，抵不过 `rowStep=1` 带来的损失。

---

## 7. ⚠️ 方法论更正：`hp` **强烈依赖并发**，我早先的对比全部无效

这是本次最重要的一条**测量方法**教训。

同一实例、同一协议，`hp`（decode ms/step）随并发大幅变化：

| 并发 | `hp` p50 |
|---:|---:|
| 1 | ~24.4 – 25.1 ms |
| 4 | ~31.0 – 33.4 ms |
| 8 | ~42.2 – 42.9 ms |
| 16 | ~51.6 ms |

**实证**：一份样本里 `hp` 呈**双峰**且**按时间切换**：

```
样本前 1/3（4 并发）  p50 = 31.05   低峰(<28ms) 占比   0%
样本后 1/3（单流）    p50 = 24.32   低峰(<28ms) 占比 100%
```

**我早先的「−0.90%（24.668 → 24.445）」是怎么来的**：A 臂的样本以单流为主（p50≈24.7），
B 臂的双峰样本里我取了**后 20 个**（恰好落在单流段，p50≈24.4）⇒
**拿单流样本比 4 并发样本**，得出一个纯属巧合的数字。

**⇒ 任何 `hp` 对比必须满足三条**：
1. **同并发**（单流 vs 单流、8 并发 vs 8 并发）；
2. **同协议**（固定 prompt、等空闲、多轮取中位）；
3. **先看分布**（单峰？双峰？分位数对不对得上），再看 p50。

只看 p50 不看分布，会在双峰数据上挑到任意一个峰。
