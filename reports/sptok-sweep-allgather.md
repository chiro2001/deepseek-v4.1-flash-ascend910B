# SP_TOKENS 扫描（AllGather 形态）：S=5 最优，但需要 capture size 配合

> 2026-09-16 08:30–10:10 CST｜A3-node1 chips 8-15｜容器 `dsv41-a21-perf`
> 基础配置：`MOE_AG=1 FUSED_MC2=1 MULTISTREAM=0 STATIC_KERNEL=1 GPU_UTIL=0.94`，
> Engram(int8, gate CHUNK=0, local-owner, hash fast) + Vision + DSpark
> 口径：T4 逐字引用 quote、单流独占（`exclusive_ok=True`）、unprofiled 客户端墙钟

---

## 1. 硬约束：`num_speculative_tokens >= dspark_block_size (5)`

`vllm/config/speculative.py:1042-1056`：

```
Value error, DSpark requires num_speculative_tokens >= dspark_block_size (5); got 4.
Smaller values produce incorrect output. Use num_speculative_tokens=5 or larger (e.g. 7).
```

⇒ **S=4 被框架直接拒绝**（实测起服报错）。可扫描范围是 S ∈ {5, 6, 7, ...}。

---

## 2. 关键陷阱：capture size 必须包含「1 + S」

`cudagraph_capture_sizes` 默认是 `[1,2,4,8,16,24]`，**不含 6**。
S=5 时每步 6 个 token ⇒ aclgraph 会 padding 到 8 号桶：

| S=5 的 capture 配置 | 128K ms/step | 说明 |
|---|---|---|
| **不含 6**（默认） | **40.572** | padding 到 8 桶，白付 2 个 token 的计算 |
| **显式含 6** | **32.894 / 34.305** | 真实 6-token 图 |

⇒ **−5.9 ms/step（−15%）**，比任何算子优化都大。

**已固化**：`serve_a21.sh` 现在按 `SP_TOKENS+1` 自动推导 `CAPTURE_SIZES`，
并通过 **`docker -e CAPTURE_SIZES=`** 传进容器（⚠️ 只 `export` 在宿主上无效——
容器是另一个环境；这是本轮踩过的坑）。

---

## 3. 128K 实测（每臂 1–2 发，注意非确定性）

| 臂 | capture 含 6/7 | ms/step | A | tok/s |
|---|---|---|---|---|
| S=5，O_PROJ_2D=0 | ✗ | 40.572 | 3.310 | 81.6 |
| S=5，O_PROJ_2D=0 | ✓ | 34.868 / 34.674 | 2.813 / 2.684 | 80.7 / 77.7 |
| **S=5，O_PROJ_2D=1** | ✓ | **34.305 / 32.894** | **3.400 / 3.493** | **99.5 / 106.6** |
| S=6，O_PROJ_2D=1 | ✓（7） | 33.656 / 33.824 | 2.763 / 2.793 | 81.8 / 82.3 |
| S=7（AllGather 基线） | ✓（8） | 35.104（3 发中位） | 2.763 | 78.4 |

**观察**：
1. **S=5 的 ms/step 最低**（每步 6 token，比 S=7 少 2 个 token 的计算）。
2. **S=5 的 A 最高**（3.31–3.51 vs S=7 的 2.68–2.82），与 `dspark_block_size=5` 正好相等
   ⇒ 推测 S=5 命中 DSpark 的**原生块大小**，额外的 draft token 走的是退化路径。
3. **S=6 的 A 掉回 2.76–2.79** ⇒ 支持上面的推测（只有恰好 = block_size 才最优）。
4. ⚠️ **A 的方差很大**（见下方 §4），上述单发数字不能当定论。

---

## 4. ⚠️ 严重警告：128K 的输出本身非确定

同 prompt、`temperature=0`、固定 seed，连发多次结果不同；阈值精确 =
`candidate_topk_blocks(2048) × candidate_block_size(8) = 16384`。
**prefill 本身就不确定**（128K 只生成 2 token，4 次首 token 全不同）。

⇒ 128K 的 **A 在 1.64–3.51 间跳**（ms/step 只跳 ±3%）。
⇒ **上述 §3 的 A 对比不能作为结论**；需要在 **≤16384 的确定性区间**复核（已排队）。
⇒ 所有 tok/s 结论都必须标样本量，或多发取中位。

详见 `reports/ctx-nondeterminism.md`、`reports/nondeterminism-rootcause.md`。

---

## 5. 当前配置与产物

`serve_a21.sh` 的默认值（本轮改）：

```bash
SP_TOKENS=${SP_TOKENS:-5}        # 原来 7
CAPTURE_SIZES=<按 SP_TOKENS+1 自动推导，并 docker -e 传入>
SPEC_EAGER_OPT=${SPEC_EAGER_OPT:-0}   # draft 入图（用户决策，弱 CPU 上收益更大）
ENGRAM=${ENGRAM:-1}              # 可覆盖（供 Engram on/off 对照）
CAND_MODE=${CAND_MODE:-0}        # 候选筛选诊断（0=stock, 3=全层无过滤, 4=源写消耗不读）
O_PROJ_2D=${O_PROJ_2D:-0}        # F3：wo_a 2D matmul
```

| 内容 | 路径 |
|---|---|
| 扫描脚本 | `exp_tools/s_sweep2.sh`、`exp_tools/cand_mode_diag.sh`、`exp_tools/queue_rest.sh` |
| 原始数据 | `logs/perf/a21/p42_t4_quote_131072_{sptok5,s5cap,f3a,f3b,f3a2,ss2s6,ss2s7}_*.jsonl` |
| 运行日志 | `/tmp/s_sweep2.log`、`/tmp/s5_capture.log`、`/tmp/f3_ab.log` |
