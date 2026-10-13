# V4.1 DCP 双轨计划：8-chip 生产轨 + 2-chip 算子轨（2026-09-29）

## 0. 为什么要两条轨

| | 8-chip 轨（生产轨） | 2-chip 轨（算子轨） |
|---|---|---|
| 模型 | `v41-flat-verify3`（真实权重，490 GB） | `model-tiny`（dummy 权重，~8 GB） |
| 并行 | TP8 + DCP8 | TP2 + DCP2 |
| 冷启动 | **~12 min**（8 rank 同时加载 + 编译） | **~2–3 min** |
| 擅长 | 容量、端到端性能、真实精度 | **DCP 代码路径的快速迭代** |
| 不擅长 | 每次改一行都要 12 min | dummy 权重 ⇒ 不能替代真实精度 |

**为什么 DCP2 有意义（不是"缩小版"）**

1. 本设计的约束是 `dcp_size == tp_size`（DCP 复用 TP rank，不占额外卡）。
   ⇒ **DCP2/TP2 是最小的非平凡配置**：2 个 rank、2 段 KV、1 次 all-gather + 1 次 all-to-all。
2. DCP 的每一处新代码（metadata 双视图、slot mapping、top-k remap、压缩槽对齐、
   LSE 合并）在 2 rank 下**都会被执行到**；而 8 rank 下"rank 1 的本地偏移算错"
   这类 bug 会被 8 路平均掩盖。
3. tiny 模型**故意保留了全部 40 层与全部 KV group 结构**（`make_tiny_config.py`
   的论证：`plan_cache_slots()` 硬校验 40 层 + `[2,8,14,20]`/`[2,8,14]`，
   所以不能截断层数；只能压专家数/LoRA rank/RoPE 表）。
   ⇒ 被测的**缓存平面逻辑与生产完全同一条代码路径**。

**判据（关键）**：同一 tiny 模型、同一批请求，**DCP1 与 DCP2 的 greedy token 必须逐 token 相同**。
dummy 权重不重要 —— 对照的是"同一个模型的两种切法"，不是"模型答得对不对"。

⚠️ 前提：要先证明**两次 DCP1 运行自身可复现**（负控）。dummy 权重里
`nn.Linear` 走默认 init（有 seed ⇒ 确定），但部分 buffer 走 `torch.empty`
（非确定）⇒ 若负控不过，"两边相同"这个判据本身就不成立，必须换判据
（例如改成比较**同一进程内** DCP1 vs DCP2 两次 forward，或把权重 dump 下来对齐）。

---

## 1. 资源分配（a3-21，16 个逻辑设备 = 8 卡 × 2 die）

| 轨 | 设备 | 约束 |
|---|---|---|
| 8-chip | 由 `dcp_stage_capacity.sh` 的 `AUTO_CHIPS=1` 自动挑 8 张 | 主 Agent 独占 |
| 2-chip | **必须在 8-chip 集合之外**，且避开别人占用的设备 | 子代理独占 |
| — | 设备 2、3 经常被别人占（`acl_bw` / `python`） | 谁都不要抢 |

**a3-22 不可用**：实测 0–15 全部被别人的进程占用（`dsv41-train3` / `yy_vf` / `xh-mk` / `cann910`）。

---

## 2. 分工

### 主 Agent（我）
- **DCP 执行路径本体**：`attention/dsa_v41.py` 的 DCP impl/builder、
  Q all-gather、`return_softmax_lse=True`、输出 all-to-all + LSE 合并、
  压缩域 top-k remap。这是全部复杂决策所在。
- 8-chip 轨的容量/正确性/性能，以及 overlay 基础设施（`V41_DCP_MOUNT` 等）。
- 合并子代理的发现，写文档。

### 子代理 `dcp_tiny_tp2`
- 把 tiny 模型弄到 a3-21（只有 config/tokenizer，6 MB）。
- 写 2-chip 启动器（参数化 `DCP=1|2`）：等卡重试 + overlay md5 校验 + health 轮询 + 抓容量行。
- 起 TP2+DCP1 基线 → 再起 TP2+DCP2，记录**每一道新门**（原文 + 文件:行号）。
- 交付 `REPORT.md` + 「主 Agent 需要改什么」清单。

### 子代理 `dcp_equiv_harness`
- **离线数值地基**（先做）：用 `npu_sparse_flash_mla` 自己拼
  「2 rank × 分段 KV → LSE 合并」，证明 **LSE 合并 == 全局 attention**。
  这是整个方案的数学前提，不依赖起服。
- **双端点等价性对比器**：给两个 OpenAI 端点，比 greedy token / logprobs / 容量倍数，
  并含**自身重复性负控**。请求集要覆盖跨 `block_size*dcp` 边界（127/128/129/1023/1024/1025）。

---

## 3. 不重叠的边界（避免互相踩）

- `~/cedpd-repo/`：**主 Agent 独占**（子代理只读）。
- `~/dcpw/`（overlay）：**主 Agent 独占**；子代理要改先报告。
- 两个子代理的产物目录各自独立（`~/tmp/dcp2tiny/`、`~/tmp/dcp_equiv/`）。
- 容器名/端口：`dsv41-dcp2tiny` / 19300（2-chip），`dsv41-dcpcap` / 19210（8-chip）。

---

## 4. 里程碑

| # | 里程碑 | 判据 | 归属 |
|---|---|---|---|
| M1 | 8-chip 容量 ×8 | `GPU KV cache size` ≈ 8.6M（7.88×） | 主 Agent |
| M2 | 2-chip TP2+DCP1 起来 | health 200 + 容量行 | tiny 子代理 |
| M3 | LSE 合并数学闭合 | 2 rank 分段 attention == 全局，误差 < 1e-4 | 夹具子代理 |
| M4 | 2-chip TP2+DCP2 起来 | health 200 + 容量约 2× | tiny 子代理 |
| M5 | **DCP2 与 DCP1 输出逐 token 相同** | 负控通过 + 全请求集相同 | 三者协作 |
| M6 | 8-chip DCP8 正确 | 144K/1M 针 + 短问答，乱码率 0 | 主 Agent |
| M7 | 8-chip DCP8 性能 | `(ms/step, A, tok/s)` 三元组 vs DCP1 | 主 Agent |
