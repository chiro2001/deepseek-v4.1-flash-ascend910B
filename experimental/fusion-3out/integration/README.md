# 融合点 B 集成（input_layernorm + wq_a.quantize）

> 目标：把 `input_layernorm`（43 层）与 `wq_a.quantize` 合成**一次** `RmsNormDynamicQuantBf16`
> 调用，省掉一遍 RMSNorm + 一遍量化下发。预期 ≈0.5%（与已落地的 `q_norm` 融合 0.62% 同量级）。

## 为什么可融合

| 现状 | 等价关系 |
|---|---|
| `hidden = input_layernorm(x)` | `rms_norm(x, w_ln)` |
| `q_quant, q_scale = wq_a.quantize(hidden)` | `npu_dynamic_quant(hidden)` **纯量化**（`cv_linear.py:73`） |
| 合计 | = `npu_rms_norm_dynamic_quant(x, w_ln)` **一次算子** |

下游还需要 `hidden`（bf16）的消费者 —— `wkv.quantize`、`compressor`（4 层）、`indexer`（8 层）
—— 正是本次新增的**第三路 bf16 输出**存在的理由。

## 部署状态（2026-10-08 已就位，**未启用**）

| 项 | tp8k5 上的位置 | 状态 |
|---|---|---|
| 算子源码 | `csrc/attention/rms_norm_dynamic_quant_bf16/` | ✅ |
| 构建清单 | `csrc/build_aclnn.sh` 已加 `"rms_norm_dynamic_quant_bf16"` | ✅ |
| torch 绑定 | `csrc/torch_binding.cpp` + 扩展已重编（984104 B，与编译容器一致） | ✅ |
| OPP kernel/aclnn | `/vllm-workspace/3out_opp/vendors/custom_transformer/` | ✅ |
| 运行环境变量 | 启动脚本需 `ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/3out_opp/vendors/custom_transformer` | ⏳ 重启时加 |
| Python 侧融合 | 本目录 `apply_lnorm_fuse.py`（**默认关闭**） | ⏳ 未执行 |

## 启用步骤

```bash
# 1) 打 Python 补丁（默认关闭，先 dry-run 看锚点）
docker cp apply_lnorm_fuse.py dsv41-tp8k5:/tmp/ && \
docker exec dsv41-tp8k5 python3 /tmp/apply_lnorm_fuse.py --dry-run
docker exec dsv41-tp8k5 python3 /tmp/apply_lnorm_fuse.py

# 2) 重启时带上 OPP 路径 + 开关
#    在启动脚本 (inner_vf.sh) 或 docker exec 环境里加：
export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/3out_opp/vendors/custom_transformer
export V41_LNORM_FUSE=1          # 关闭时删掉这一行即可
```

## 验证顺序（建议）

1. **未开启** `V41_LNORM_FUSE`：先确认服务正常启动、`/health` 200 —— 验证扩展/OPP 加载无副作用。
2. **开启后**：跑冒烟对话请求 + 与关闭时的输出逐字符对比（同 prompt 同 seed），确认精度一致。
3. **性能**：按既有口径测 `bneck hp p50`（decode ms/step）+ 8 并发总吞吐。
4. 异常则删掉 `V41_LNORM_FUSE=1` 重启；kernel 侧无状态，无需回滚二进制。

## 回滚

* Python：`apply_lnorm_fuse.py` 每次改动都会留 `.bak-lnorm-HHMMSS` 副本，或把环境变量置 0。
* 扩展：容器内 `vllm_ascend_C.cpython-312-aarch64-linux-gnu.so.orig-190825` 是原始副本。

## 已知边界

* 启用条件写死 `hidden_states.shape[-1] == 1280`（生产 hidden_size）。
  `D ≤ 512` 的输入**不要**走融合：上游 `rms_norm_dynamic_quant` 在该区间本身就有缺陷
  （原算子同条件 20/20 错，见 `docs/MULTI-OUT-OP-IMPLEMENTATION-20261008.md` §9.3）。
* `wq_a` 若有 TP 通信或非 W8A8-dynamic，会自动回退到原 `quantize` 分支。

---

## A/B 实测结果（2026-10-08，tp8k5 实机）

主判据 = bneck `hp` p50（decode ms/step），同容器同负载（4 并发 × 700 token × 3 轮 + 单流 1200）。

| 臂 | 后20 hp p50 | 后60 hp p50 |
|---|---:|---:|
| **A**：`V41_LNORM_FUSE` 走不到融合（同一份补丁，形状判定为假 ⇒ 原路径） | 24.668 ms | 24.600 ms |
| **B**：融合开启（真正执行三输出算子） | **24.445 ms** | **24.424 ms** |
| **差值** | **−0.223 ms（−0.90%）** | **−0.72%** |

B 臂 p50 低于 A 臂最小值，分布基本分离。单流吞吐 132.7 vs 132.3 tok/s（噪声内）。

### 调试中踩到的两个坑（都会让融合**静默失效**）

1. **形状判定写错**：`hidden_size = 5120`，初版保护条件写成 `== 1280` ⇒ 永远为假。
   已改为 `>= 513`（D≤512 是上游缺陷区，见 `docs/MULTI-OUT-OP-IMPLEMENTATION-20261008.md` §9.3）。
   补测 D=5120 × N∈{6,160} × 各 20 次：bf16/int8/scale **全部零错**。
2. **对象身份不匹配**：读侧是 `forward_context.no_compile_layers[prefix]`，
   绑定的是 `AscendDeepseekSparseAttention` 实例（`layer.self_attn.dsa_attn`），
   而初版写到了 `self.self_attn` 上 ⇒ `getattr(attn, "_lnorm_fused_quant", None)` 恒为 `None`，
   静默回退到 `wq_a.quantize`。已改为写 `getattr(self.self_attn, "dsa_attn", self.self_attn)`。

**教训**：这类"跨层传状态"的融合，必须验证**传到了**，不能只看启动没报错、输出没乱码。
两次失效都产出了完全正常的输出，只是没有性能收益。
