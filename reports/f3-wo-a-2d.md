# F3 实测：`wo_a` 退化 batch matmul 改 2D（A/B/A2 配对）

> 2026-09-16 09:14–09:58 CST｜A3-node1 chips 8-15｜容器 `dsv41-a21-perf`（port 8020）
> 配置：`static_kernel=1 npugraph_ex=1 MOE_AG=1 FUSED_MC2=1 MULTISTREAM=0 SP_TOKENS=5 GPU_UTIL=0.94`
> CAPTURE_SIZES=`[1,2,3,4,6,8,12,16,20,24,32]`（含每步 6 token 的桶）
> 口径：T4 逐字引用 quote、单流独占（`exclusive_ok=True`）、unprofiled 客户端墙钟

---

## 1. 结论

**采纳。** `wo_a` 在 `n_local_groups == 1` 时把 `npu_transpose_batchmatmul([T,1,H] × [1,H,R])`
换成普通 2D `matmul([T,H] × [H,R])`，数学严格等价、零精度风险，
实测 **ms/step −0.31（ms 口径）/ −0.76（cli_p50 口径）**，方向在 A/B/A2 三轮中一致。

| 臂 | 配置 | 128K ms/step | 128K cli_p50 | n |
|---|---|---|---|---|
| A1 | `V41_O_PROJ_2D=0` | 34.868 / 34.674（稳定两发） | 33.069 / 33.018 | 2 |
| **B** | **`V41_O_PROJ_2D=1`** | **34.305 / 32.894** | **32.332 / 32.323** | 2 |
| A2 | `V41_O_PROJ_2D=0` | 33.908（均） | 33.175（均） | 2 |

Δms/step（B vs A1+A2）= **−0.31 ~ −1.17**；Δcli_p50 = **−0.71 ~ −0.76**。
接受长度 A 不因本改动变化（差异来自已知的上下文非确定性，见
`reports/ctx-nondeterminism.md`）。

> ⚠️ 前两发 A1（39.017 / 37.846 ms）是会话热身，不计入比较；A2 为校准漂移而设。

---

## 2. 原理

审计（`reports/scalar-bound-op-audit.md` §F3）实测：

```
TransposeBatchMatMul  [8,1,4096] x [1,4096,1024]   47.26 us/op  x 40 = 1.890 ms/step
  cube_utilization = 13.9%   aic_mac_ratio = 0.109   aic_scalar_ratio = 0.183
```

`n_local_groups == 1` 时 batch 维退化为 1，展开后
`y[t,0,k] = Σ_j x[t,0,j] · w[0,j,k]` 就是普通 2D matmul：

```python
y2d = x.reshape(T, H) @ w.squeeze(0)        # [T,H] @ [H,R] -> [T,R]
```

同一批浮点乘加、只是 kernel 不同（不再走 batch-matmul 的 tiling）。
cube 利用率低（13.9%）说明原路径的 tiling 对小 batch 不划算。

---

## 3. 改动（env 门控，默认关闭 = stock）

文件：`probe_dsa/dsa_v1.py`（整文件 bind-mount 到
`/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v1.py`）

| 位置 | 内容 |
|---|---|
| `:146` | 新增 `_O_PROJ_2D = os.environ.get("V41_O_PROJ_2D", "0") == "1"` |
| `:149` | 新增 `def _o_proj_2d_enabled()` |
| `:1571` | `_forward_o_proj` 里新增 `elif _o_proj_2d_enabled() and self.n_local_groups == 1:` 分支 |

新分支做两件事：
1. `_w2d = self.wo_a.weight.squeeze(0)`（**[1,H,R] → [H,R]**）
2. `o_proj_input = torch.matmul(o_proj_input.reshape(num_tokens, -1), _w2d)`，随后照旧 `self.wo_b(...)`

未触碰 `oproj_tp_enable()`（OTP 多组）与 A5 FP8 两条分支 —— 它们与本次改动正交。

### 3.1 启动开关

`scripts/serve_a21.sh` 新增：

```bash
O_PROJ_2D=${O_PROJ_2D:-0}          # 0=stock, 1=2D 路径
MOUNTS="$MOUNTS -v $P/probe_dsa/dsa_v1.py:/vllm-workspace/.../attention/dsa_v1.py:ro"
MOUNTS="$MOUNTS -e V41_O_PROJ_2D=$O_PROJ_2D"
```

用法：`O_PROJ_2D=1 bash scripts/serve_a21.sh`（默认仍是 0）。

---

## 4. 证据

| 内容 | 路径 |
|---|---|
| 补丁脚本（幂等） | `probe_dsa/patch_dsa_wo_a.py`（自检锚点唯一、`py_compile` 通过） |
| 挂载点补丁 | `scripts/serve_a21.sh` 的 `# [F3]` 段 |
| A/B/A2 原始数据 | `logs/perf/a21/p42_t4_quote_131072_f3{a,b,a2}_*.jsonl`、`..._32768_*.jsonl` |
| 运行日志 | `/tmp/f3_ab.log`、`/tmp/f3_{f3a,f3b,f3a2}.out` |
| 起服日志（含 KV / capture sizes / 静默降级检查） | `logs/perf/f3{a,b,a2}_*_serve.log` |
| 每臂 `static_kernel.py:650` 计数 | 全部 **0**（无静默降级） |
| KV 容量 | 4,162,358（A） / 4,162,480（B） tokens，均 >3M ✓ |

---

## 5. 未验证 / 风险

1. **未做确定性对账**：本配置下 >16384 上下文输出本身非确定（见
   `reports/ctx-nondeterminism.md`），无法用"同 prompt 逐字节一致"验证。
   等价的数学论证 + `matmul` 与原 batch matmul 在同 dtype 下的逐元素等价性
   **尚未用离线单测硬证**（不同 kernel 的累加顺序可能产生 1 ULP 差异）。
   → 建议：在 ≤16384 的确定性区间跑一次 GSM8K-100 作精度护栏。
2. **只在 `n_local_groups == 1` 时生效**：本部署 TP=8、`o_groups=8` ⇒
   `n_local_groups = 8/8 = 1` 命中；其它并行度不生效（安全）。
3. **收益量级小于审计预估**：审计估 −1.1 ms/step，实测 −0.31 ~ −0.76。
   原因未查（可能 kernel 差异被其它瓶颈掩盖，或 47.26 us 的一部分是固定开销）。
