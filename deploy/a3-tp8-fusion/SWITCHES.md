# 开关详解

本包涉及的开关只有 **3 个**（其中 1 个是本包新增）。全部是**环境变量**，
在起服环境里 export，通过 `vllm serve` 进程继承（`/proc/<pid>/environ` 可验）。

---

## 1. `ASCEND_CUSTOM_OPP_PATH` ★ 必设

| | |
|---|---|
| 类型 | CANN 框架变量（**不是**我们发明的） |
| 取值 | `/vllm-workspace/3out_opp/vendors/custom_transformer` |
| 默认 | 无 |
| 何时生效 | 进程启动时读一次；**改完必须重启** |

**作用**：告诉 CANN 到哪个目录去找自定义算子。aclnn 在 `GetWorkspaceSize` 阶段按
`vendor/op_api/include` + `op_impl/.../kernel` 解析算子，找不到就报
`aclnnRmsNormDynamicQuantBf16 not found`（或回退到 torch 原生实现，取决于调用点）。

**为什么路径里带 `vendors/custom_transformer` 两层**：这是 CANN 的 vendor 布局约定
（`<root>/vendors/<vendor_name>/op_api|op_impl|op_proto`），不是我们的选择。

**坑**：这个变量是**替换式**语义还是**追加式**？实测基础镜像 `opp/vendors/` 为空，
所以两种语义在本环境等价。**若将来基础镜像自带 vendor 算子**，必须重新确认，
否则会屏蔽掉官方 vendor（记录在案，见 `verify_consistency.sh` 的前置检查）。

---

## 2. `V41_LNORM_FUSE` ★ 本包新增

| | |
|---|---|
| 类型 | 本项目的开关（读点在 `models/deepseek_v4/model.py`） |
| 取值 | `1` 开 / 其它或未设 = 关 |
| **默认** | **关** |
| 读取时机 | **首次调用时读一次并缓存**（模块级 `_LNORM_FUSE_CACHE`）⇒ 改完必须重启 |

**作用**：把 decoder layer 的

```
hidden = input_layernorm(x)            # 一遍 RMSNorm
q_quant, q_scale = wq_a.quantize(hidden)   # 一遍动态量化
```

合成**一次** `npu_rms_norm_dynamic_quant_bf16(x, w_ln)`，同时产出

* `y_bf16` → 供 `wkv` / `compressor`(4 层) / `indexer`(8 层) 继续使用
* `y_int8` + `scale` → 供 `wq_a.matmul` 直接使用

覆盖 **43 个 layer**。机制：decoder layer 把 int8/scale 放在
`layer.self_attn.dsa_attn._lnorm_fused_quant` 上，attention 的
`multistream_preprocess` 消费后立即清空（每步消费一次，不跨步残留）。

**自动回退条件**（任一命中即走原路径，不报错）：

1. `V41_LNORM_FUSE != 1`
2. `hidden_states.shape[-1] < 513`
   —— `D ≤ 512` 时上游 `rms_norm_dynamic_quant` **本身就错**（不是本包引入，
   原算子同条件 20/20 错，见 `docs/MULTI-OUT-OP-IMPLEMENTATION-20261008.md` §9.3）
3. `wq_a` 带 TP 通信，或不是 W8A8-dynamic
4. attention 侧没读到 `_lnorm_fused_quant`（例如走了非 `multistream_preprocess` 路径）

**⚠️ 失效是静默的**。上述 2/3/4 命中时，输出完全正常、启动无报错，只是**没有收益**。
因此验收必须**看性能数字**，不能只看"没崩"。已实测踩过两次
（形状判定写成 `== 1280`、状态写到 `self.self_attn` 而非 `self.self_attn.dsa_attn`），
详见 `experimental/fusion-3out/integration/README.md`。

**如何确认真的生效**（不看性能也能验）：

```bash
# 在容器内起一个探针进程，命中时 _lnorm_fused_quant 在本步内非 None
docker exec <ct> bash -lc 'grep -c "_fuse_tgt = getattr(self.self_attn" \
  /vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v4/model.py'   # 期望 1
```

---

## 3. `V41_QNORM_FUSE` （已在上游 main，非本包引入）

| | |
|---|---|
| 取值 | `1` 开 / 其它 = 关 |
| 默认 | 关（但本仓的起服脚本一直设 `1`） |
| 作用 | `q_norm` + `wq_b.quantize` 合成一次 `npu_rms_norm_dynamic_quant`（**上游 2 输出算子**，不需要本包的 OPP） |
| 覆盖 | 35 层（= 43 − 8 个 `index_source` 层） |
| 实测 | 端到端 **−0.62%**（`docs/QNORM-FUSE-A-B-20261008.md`） |

> 它**不依赖** `ASCEND_CUSTOM_OPP_PATH`：用的是官方已有的 2 输出算子。
> 本包把它一起列出来，是因为交付口径上这两个开关是配套的。

---

## 4. 开关组合与预期

| `ASCEND_CUSTOM_OPP_PATH` | `V41_LNORM_FUSE` | `V41_QNORM_FUSE` | 预期 decode ms/step |
|---|---|---|---|
| 未设 | 任意 | 0 | 基线（无融合） |
| 未设 | 任意 | 1 | 基线 −0.62%（融合点 A） |
| **设** | **1** | **1** | **基线 −1.5% 左右（A −0.62% + B −0.90%，实测可加）** |
| 设 | 0 | 1 | 同第 2 行（B 关，OPP 白装但不报错） |

> 第 3 行的 −0.90% 是**实测**（2026-10-08，tp8k5，`hp p50` 24.668 → 24.445 ms）；
> "可加"是基于两点独立验证同一段代码路径（`input_layernorm+wq_a` 与 `q_norm+wq_b`
> 是相邻但不同的两处），但**未做三点同测**——见 README §5 的待办。

---

## 5. 回滚

| 要回滚什么 | 怎么做 | 需要重启 |
|---|---|---|
| 只关融合 | 删掉/置 0 `V41_LNORM_FUSE` | 是 |
| 融合 + OPP | 再删 `ASCEND_CUSTOM_OPP_PATH` | 是 |
| torch 扩展 | 换回 `...so.orig-<HHMMSS>` | 是 |
| 全部 | 换回 `py/*.py` 的 `.bak-*` 副本 + 换回 `.so` + 删两个变量 | 是 |

**本包不修改任何官方文件**（只新增 `/vllm-workspace/3out_opp/` 与替换 2 个 `.py` + 1 个 `.so`），
所以回滚不需要重装镜像。
