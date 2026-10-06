# 回答三个问题：并发退化归因 / KV32 重排 / DBO 流与依赖（2026-10-07）

> 本文回答用户三问，其中第三问暴露了我们此前一个**结论性错误**，已在本文件更正。
> 全部数字标【实测】/【推断】。

---

## 0. 三句话先答

1. **"1→8 掉很多"→"只掉 13%"不是性能变了，主要是口径变了**；但**单流步长确实真的降了 25%**
   （32.68 → 24.58 ms），那批优化是真的。
2. **KV32 重排是把"layer-20 的 index 平面"从 slot3 挪进 slot0 的空闲区**，
   让四个 slot 的页步长全部变成 131,072 ⇒ 32 位块上限 29,076 → **32,768**。
   它本身**不增加容量**，作用是"让 `BAT=2048` 多出来的预算能用上"。
3. **★ 用户的判断是对的**：vLLM 的 DBO **只有一条 compute_stream**，所有 ubatch 共用它
   ⇒ **两个 ubatch 的计算物理上不可能并行**。我们此前写的"图内并发 = 串行 ⇒ 重叠收益 = 0"
   **措辞错误**：不是设备不并行，是**我们的实现里就没有两条计算流**。

---

## 1. 并发退化：到底变了什么

### 1.1 三代测量的对照（同一张表看趋势）

> ⚠️ **口径警告**：本文统一用 **`A_out` = 每步产出的 token 数**（含 bonus token）。
> 部分历史文档写的 "A" 是**接受长度**，需 `A_out = 1 + 接受长度`，两者不可混用。
> 关系：`每路 tok/s = A_out / ms每步 × 1000`。

| 时点 | 拓扑 | 单流 ms/step | 单流 A_out | 单流 tok/s | N=8 ms/step | N=8 A_out | N=8 每路 | 每路降幅 | 出处 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 09-26 | CED-PD（DCP8） | **32.68** | 3.10 | 94.8 | — | — | — | — | `CED-PD-DSPARK-LATENCY` |
| 10-05 | TP8 交付 | **25.28** | 2.87 | 113.4 | 39.22 | 2.52 | 64.3 | **−43%** | `R6-SUMMARY` |
| 10-06 | TP8 交付 | **24.58** | 2.36 | 95.8 | 41.51 | 3.46 | 83.3 | **−13%** | `TP8-CURVE-CORRECTED` |

**单流步长 32.68 → 25.28 → 24.58 ms（累计 −25%）** 是**真实的性能提升**，
来自 10-05 那一轮的一批优化（engram `wkv` 分片 −0.57 ms、`PAD_SKIP` −58 µs、
gmm1 armF 内核、`RmsNorm`/`Scatter` 系列的接线修复等）。**这部分是实的。**

**但 N=8 步长不降反升**（39.22 → 41.51），这是两次测量之间**并发行为没有改善**的证据。

### 1.2 「每路只掉 13%」里有一半是口径

每路速度 = `A_out / ms每步`，所以**光看"每路掉了多少"会混入 `A_out` 的变化**：

| 时点 | conc=1 `A_out` | conc=8 `A_out` | `A_out` 变化 | 步长变化 | 每路净变化 |
|---|---:|---:|---:|---:|---:|
| 10-05（R6） | 2.87 | 2.52 | **−12%** | +55% | −43% |
| 10-06 | 2.36 | 3.46 | **+47%** | +69% | **−13%** |

⇒ **10-06 那次"只掉 13%"，是因为 `A_out` 从 2.36 涨到 3.46（+47%），把 +69% 的步长增长抵消掉了大半。**
这不是性能变好，而是**那一档测到的接受长度恰好更高**（两次用的 prompt 集不同：
R6 用 `bench_concurrency` 的 4 条正文切片，10-06 用 32 条互不重叠切片 + nonce 破缓存）。

**用户说的"并不成比例"，就是这个原因。** 正确的读法是看**每 token 延迟**：

| 时点 | conc=1 ms/token | conc=8 ms/token | 每 token 延迟涨幅 |
|---|---:|---:|---:|
| 10-05（R6） | 25.28 / 2.87 = **8.81** | 39.22 / 2.52 = **15.56** | **+77%** |
| 10-06 | 24.58 / 2.36 = **10.41** | 41.51 / 3.46 = **12.00** | **+15%** |

⇒ 两个口径给出**完全不同的结论**（+77% vs +15%）。差异全部来自 `A_out`。
**要看服务质量就用 `ms/token`，并且必须固定语料**；跨语料比 `tok/s` 会得出互为矛盾的结论。

### 1.3 「推测解码占用很多性能」这个结论：当时对，但归因不完整

**当时的证据**（`CED-PD-DSPARK-ACCEPTANCE-20260926`，2K prompt + 128 token 输出）：

| 并发 | DSpark 总吞吐 | SPEC=0 总吞吐 | 比 |
|---:|---:|---:|---:|
| 1 | **73.1** | 41.4 | **1.77×** |
| 2 | 57.2 | 70.6 | **0.81×** |
| 4 | 76.7 | 114.5 | **0.67×** |

当时的归因：**"推测解码把每步行数从 `batch×1` 抬到 `batch×(1+K)=batch×6`，
而产出只多 A≈3 倍 ⇒ 每步算子时间增长快过 token 产出"**。

**这个归因有一半是错的，一半是对的**：

| 部分 | 判 | 依据 |
|---|---|---|
| "M 从 1 抬到 1+K" | ✅ 对 | `HcPre` Input Shapes 实测 decode 行数 1 → 8 |
| "算子时间随之线性增长" | ❌ **错** | 同一份 profile：`HcPre` 28.14 → 32.26 µs（**仅 +14.6%**）、`SparseFlashMla` +7.6% ⇒ **M=8 是权重带宽受限，多 7 行几乎免费** |

⇒ **推测解码的代价不是"逐步线性涨"，而是"一份近似固定的 +12 ms/步"**
（`V41-DSPARK-COST-BREAKDOWN-LIVE`：SPEC=0 29.35 → SPEC=1 41.4 ms，**+12.05 ms**）。

### 1.4 ★ 那"1→8 掉很多"的真正主因是什么

`N8-SCALING-DEEPDIVE-20261004` 用相位表把它拆干净了（8 流 × 512 token 的完整一轮 19.2 s）：

| 相位 | 应产出 | 实际 | 缺口 | 占总缺口 |
|---|---:|---:|---:|---:|
| **ramp**（prefill-only 步 + 单流 decode，7.6 s） | 3764 | ~130 | **−3634** | **63%** |
| **tail-B**（剩 2 流） | 1783 | ~260 | −1523 | 26% |
| tail-A（5–7 流） | 990 | ~550 | −440 | 8% |
| **full（8 流稳态）** | 2872 | ~2730 | −142 | **2%** |

⇒ **稳态本身已经跑到产能的 95%；产能损失的 89% 在 ramp + tail。**
而 ramp 的机制是日志实证的：**`admission_gate` 把每个 prefill 变成"独占步"，
其间 decode 被推迟**（`deferred_decode_reqs` 累计到 28，16 个 prefill-only 步跨了 **8.0 秒**）。

**结论**：

* **"1→8 掉很多"的主因是批次爬升与收尾**（89%），**不是推测解码的算子开销**；
* 推测解码的角色是**把它放大**：`ms/step` 从 ~28.8（SPEC=0）涨到 41.5（SPEC=1），
  所以同样的 ramp/tail 时长里，能产出的 token 更少；
* 且 ramp/tail 期间流的条数少，spec 那 +12 ms/step 的固定成本被摊得薄 ⇒ **性价比最差**。

**可做的方向**（按 N8-SCALING 的量化排序）：

1. **让 prefill 与 decode 重叠**（取消 prefill-only 独占步）→ 最大头（对应 +120 tok/s）；
2. **缩短 ramp**（prefill 吞吐 / 8 条流并行提交 / 前缀命中）；
3. **收窄 tail**（同批请求对齐结束、动态 `SP_TOKENS`）；
4. 算子级优化（在这类窗口里只值 ~2%）。

> ⚠️ 注意：上述 63%/26% 的分母是"8 条流同时起、同时结束 512 token"的合成场景。
> **线上流量是陆续到达的，ramp/tail 不会以这种形态出现**——若请求持续到达，
> 引擎会一直处于"full"相位附近。所以这条分析的价值是**说明"批处理爬升"的代价量级**，
> 而不是说线上真的损失 89%。【推断】

---

## 2. KV32 重排到底做了什么

### 2.1 背景：KV 池是按 slot 分页的，而"页步长"受 32 位约束

```
池 = 若干块，块内按 slot 分页；每页里放若干"平面"（KV / index / SWA / draft / state）
页偏移是 32 位 ⇒ 块号上限 = floor(2³² / 该 slot 的页步长)
```

**现状（实测 `[V41-DCP-DIAG]`）**：

```
pool_bytes_per_block = 540,928    slots = [131072, 131072, 131072, 147712]
                                      slot0    slot1    slot2   slot3 ← 绑定
```

| slot | 归属 | KV 平面 | index 平面 | 容量 |
|---|---|---:|---:|---:|
| 0 | layer 2（ratio 2） | 65,536 | 8,320 | 131,072（被 draft 顶住） |
| 1 | layer 8 | 65,536 | 8,320 | 131,072 |
| 2 | layer 14 | 65,536 | 8,320 | 131,072 |
| **3** | **layer 20（ratio 1）** | **131,072** | **16,640** | **147,712 ← 绑定** |

⇒ **slot3 比别的大 16,640 B，正是 layer-20 的 index 平面**。
而 `2³² / 147,712 = 29,076` 块，**这就是当时的容量上限**。

### 2.2 重排：把那个 index 平面挪到 slot0 的空闲区

```
slot0 已用 = 65,536 + 8,320 = 73,856
slot0 空闲 = 131,072 − 73,856 = 57,216  ≥  16,640  ✓

⇒ 把 layer-20 的 index 平面改为 CachePlacement(offset=73,856, page_size=16,640)
⇒ slot0 容量不变（131,072，被 draft 顶住）
⇒ slot3 降到 131,072
⇒ 四槽全 131,072 ⇒ 2³²/131,072 = 32,768 块（+12.7%）
```

**实测结果**：

* `pool_bytes_per_block` **540,928 → 524,288**（四槽全 131,072）✅
* **纯布局改动逐位相同**：同块数（29,076）下，旧几何 vs 重排，
  30 轮 × 63,998 位全部 `max|Δ| = 0`、**0 个非零点**
* 容量与块数**严格线性**：3,403,198 / 3,019,745 = 1.12701 vs 32768/29076 = 1.12700

### 2.3 ★ 关键：重排**本身不增加容量**

| 量 | 说明 |
|---|---|
| 重排前 | 29,076 块上限（**实际容量由内存预算决定**） |
| 重排后 | 32,768 块上限 |
| **基线为什么是 2.9M** | **与重排无关**。2.9M = 12.9 GiB 预算 ÷ 每 token 字节数，而 12.9 GiB = 64×0.92 − 权重 39.5 − **激活 3.15（BAT=8192）** − 图 0.79 − 非 torch 0.56 |

⇒ **重排只是"把天花板抬高"，本身不产生容量**。
**真正产生容量的是 `BAT_TOKENS` 8192 → 2048**（激活 3.15 → 0.79 GiB，多出 ~9,000 块 = +35%）。
而**如果不重排，这多出来的 9,000 块会越过 29,076 的上限 ⇒ 块号回绕 ⇒ 静默读错**
（这就是 `BAT=2048` 当年"长文退化"的真因）。

**所以两件事必须同时做**：

```
BAT_TOKENS=2048  →  多出 ~9,000 块预算（+35% 容量）
V41_KV32_REPACK=1 →  上限从 29,076 抬到 32,768（+12.7%），装得下
⇒  合计：2,987,836 → 4,010,585 token（+34.2%），长文针 24/24
```

**代价**：prefill ~6.5K → ~5.8K tok/s（−11%）、首 token 更慢（60K prompt 从 8 刀变 30 刀）。

### 2.4 当前默认值（两个都没改）

```bash
BAT_TOKENS=${BAT_TOKENS:-8192}          # 仍是 8192
V41_KV32_REPACK=${V41_KV32_REPACK:-0}   # 仍是 0（关）
```

⇒ **发布默认仍是 2.9M**，4.01M 是可选档。

---

## 3. ★ 更正：DBO 的"两条流"不是"两个 ubatch 两条流"

### 3.1 用户的理解是对的

> "我们只把图中放了两个流，但是我没有做这两个流之间的同步，
> 所以我们也没有把数据依赖之类的放到两个 stream 中执行。"

**核对源码后：用户的判断成立，而且比我们此前的结论更接近真相。**

### 3.2 源码事实（`vllm/v1/worker/ubatching.py` + `ref_pr11273_npu_ubatch_wrapper.py`）

```python
compute_stream = torch.npu.current_stream()      # ★ 只有一条
comm_stream    = self.comm_stream                # ★ 只有一条

for i in range(num_micro_batches):
    ctx = NPUUBatchContext(
        compute_stream=compute_stream,   # ← 所有 ubatch 共用同一条 compute_stream
        comm_stream=comm_stream,         # ← 所有 ubatch 共用同一条 comm_stream
        gpu_comm_done_event=gpu_comm_done_events[i],      # 每 ubatch 各自一个 event
        gpu_compute_done_event=gpu_compute_done_events[i],
        ...
    )
```

而两条流之间的同步是**显式**的：

```python
def _signal_comm_done(self):    self.gpu_comm_done_event.record(self.comm_stream)
def _signal_compute_done(self): self.gpu_compute_done_event.record(self.compute_stream)
def _wait_compute_done(self):   self.comm_stream.wait_event(self.gpu_compute_done_event)
def _wait_comm_done(self):      self.compute_stream.wait_event(self.gpu_comm_done_event)

def switch_to_comm_sync(self):    # 计算 → 通信
    self._signal_compute_done(); self.update_stream(self.comm_stream); self._wait_compute_done()
def switch_to_compute_sync(self): # 通信 → 计算
    self._signal_comm_done();    self.update_stream(self.compute_stream); self._wait_comm_done()
```

### 3.3 这意味着什么（三条，逐条更正）

| # | 我们此前写的 | 更正为 |
|---|---|---|
| 1 | "图内双流并发" | **图内确实有两条流，但它们是 `compute_stream` 与 `comm_stream`，不是"每个 ubatch 一条计算流"** |
| 2 | "图内并发 vs 串行曲线重合 ⇒ 重叠收益 = 0" | **两个 ubatch 的计算 kernel 都提交到同一条 `compute_stream` ⇒ 计算之间物理上不可能并行**。曲线重合是**实现使然，不是设备结论** |
| 3 | "DBO 的目标是 AIC/AIV 跨 batch 并行" | **DBO 的设计目标从来不是"两个 ubatch 并行计算"，而是"ubatch A 的通信 ∥ ubatch B 的计算"**（Dual Batch Overlap）。vLLM 的 `switch_to_comm_sync` / `switch_to_compute_sync` 就是为这个设计的 |

⇒ **"重叠收益 = 0"这个结论，在"计算并行"这个意义上是对的（因为根本没实现），
但在"通信∥计算"这个意义上缺少直接证据**：我们从未测量
`comm_stream` 与 `compute_stream` 在 replay 时的真实重叠系数。

### 3.4 那为什么"通信∥计算"也没发生

间接证据是：**我们这个负载下通信与 AIC 的重叠实测为 `0.000`**
（`THROUGHPUT-CEILING` §1：`COMM ∩ AIC = 0.000`，通信 2.59 ms 完全暴露）。

两个可能原因（**未区分**）：

1. **实现层**：我们的图内集成没有真正让 `switch_to_comm_sync` 生效
   （捕获期 `forward_context` 与 event 语义可能被简化）；
2. **结构层**：即使切了流，`compute_stream` 上紧随其后的操作**立即依赖 allreduce 结果**
   （`COMM-STRUCTURE-AND-DBO-REGIME`：通信后紧跟 `TensorMove + HcPost`，两者都依赖 allreduce 输出）
   ⇒ **没有可填充的独立工作**。

### 3.5 决定性的实验（此前欠的账）

**判据：不要用端到端时间，要用设备时间轴上的重叠系数。**

```python
# 在 tiny 上捕获 2-ubatch 的图，replay 后导出带 Stream ID 的 profile
ρ = |union(comm_stream 区间, compute_stream 区间)| 的重叠时长
    / (comm_stream 总时长 + compute_stream 总时长)

ρ ≈ 0        ⇒ 两条流被串行化（实现问题，或 event 阻塞）
ρ ≈ 0.3–0.5  ⇒ 真并发但相位重合（结构问题）
```

**并且必须同时回答"有没有可填的独立工作"**：
把通信窗口内 `compute_stream` 上实际在跑的算子列出来。
如果那些算子都依赖 allreduce 结果，那么 ρ 再高也没用。

### 3.6 如果要做"真正的跨 batch 计算并行"，需要改什么

**现有骨架做不到** —— 它只有一条 `compute_stream`。要拿到 `PINGPONG` 微基准里的 1.37×，需要：

1. 给**每个 ubatch 分配独立的 compute_stream**
   （`_make_npu_ubatch_contexts` 里改成 `compute_streams[i]`）；
2. 每个 ubatch 结束时在自己的流上 `record` join event，根流 `wait_event` 汇合
   （`MULTISTREAM-GRAPH-CAPTURE-BREAKTHROUGH` 已验证这个写法能被捕获）；
3. 保持 per-ubatch 的 metadata 静态缓冲（否则仍会撞上捕获期地址生命周期问题）。

⚠️ **但预期要压住**：DBO 的 −42%（图模式 conc=8 75.7 vs 基线 130.6）
主要来自**拆批后 kernel 自身效率下降**（MoE 专家利用率、通信次数翻倍），
这部分**与是否并行无关**，换成两条计算流也救不回来。
所以第 3.5 节的实验应先做**归因**，再决定要不要动骨架。

---

## 4. 复现

```bash
# §1 三代曲线
sed -n '1,40p' docs/CED-PD-DSPARK-LATENCY-BREAKDOWN-20260926.md
sed -n '1,30p' docs/R6-SUMMARY-20261005.md
sed -n '1,60p' docs/TP8-CURVE-CORRECTED-20261006.md
sed -n '1,60p' docs/N8-SCALING-DEEPDIVE-20261004.md      # ramp/tail 相位表

# §2 KV32 重排
sed -n '1,60p' docs/KV32-SLOT-REPACK-PROPOSAL-20261006.md
grep -n 'V41_KV32_REPACK\|BAT_TOKENS=' scripts/serve_a2.sh

# §3 DBO 流结构
sed -n '30,100p' tools/ref_pr11273_npu_ubatch_wrapper.py
ssh a3-21 'docker exec dsv41-tinyspark cat /vllm-workspace/vllm/vllm/v1/worker/ubatching.py | head -60'
```
