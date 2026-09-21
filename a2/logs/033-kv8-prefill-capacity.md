# 033 — KV8 的 prefill 硬伤：**+45~105 ms/step → +0.00~0.58 ms/step**；容量实测 **×1.9133**

> 2026-09-22 02:35–03:2x CST。执行：子代理 **KV8_prefill**。机器：**A3（A3-node1）槽位 c1 = die 6**
> （`Ascend910_9382`，容器 `prbench-c1`，容器内 commit `e43cf1e9f`）。
> 全程只用 c1；没碰 `dsv41-a3` / `mooncake-master` / Phy-ID 8–15；没手设 `ASCEND_RT_VISIBLE_DEVICES`；
> **没写 `upstream-v41/`**；没动 `dsv41-release/`；没用 `/tmp`（`TMPDIR=/work/agents/KV8_prefill/tmp`，
> 收尾核对容器 `/tmp` 总计 **20 KB**）；跨机传输全走 coscli；代码只写 `a2/agents/KV8_prefill/`，
> 影子包只改自己那份 `agents/KV8_prefill/shadow/`。

---

## 0. 五句话结论（先给主代理）

1. **【实测·prefill 解了】018 的"页粒度整前缀重建"确实绕开了 015 的 `Q_T×512` 行放大，但用 torch 写仍然
   +40.9 ms/step（+10.2%）**；本任务把它也融成 Triton kernel 后：
   **+0.58 ms/step（host-bound 最坏口径，+0.145%）**、
   **+0.00~0.06 ms/step（设备饱和口径，生产形状，≤+0.015%；配对噪声地板 ±0.2 ms/step）**。
   ⇒ 相对 015 的 45–105 ms，**降了 75~175 倍**；判据 **≤+0.5%/step ✅**、**≤+150 µs/step ✅（设备饱和口径）**。
2. **【实测·容量落到真服务了】五个档位全部起过真引擎，读的是 `GPU KV cache size`**：
   `22,719 → 22,719 → 25,801 → 33,295 → 43,469`，即 **×1.000 / ×1.000 / ×1.1357 / ×1.4655 / ×1.9133**，
   与按页几何解析算出的 `540928 / 540928 / 476416 / 369280 / 282880 B/block` **逐格对上（偏差 <0.05%）**。
   ★ **判据 ×1.91 达标（±5%）：×1.9133**。
3. **★【实测·两条独立路径互证】ring 缩 FP16 单独用是 ×1.000（白做）**：`pf-ring16` 与基线**一字不差**
   —— 前 3 槽虽然从 131072 降到 65536，但**第 4 槽（ratio-1 槽 147712 B）才是 binding**，
   `max(83200, 66560) = 83200` 并没有变。⇒ 主代理给的"ring-only = ×1.56"是**按 3 槽算的**，四槽一起算就是 ×1.000。
4. **★【实测】KV8 全量的边际成本很低**：相对"只量化 SWA + ring16"这一档（本任务 §5.4 亲测），
   **decode 只多 +178 µs/step（+0.59%）、prefill 只多 0~0.06 ms/step，容量却再涨 ×1.306（33,295 → 43,469）**。
5. **【实测·顺手抓到一个静默正确性 bug】**：`dsa_v41.kv8_scratch_plane` 的缓冲键只有
   `(blocks, block_size, dim)`——**窗口重建与压缩重建在页数相同时会共用同一块 scratch**，
   后者覆盖前者、**不报错**。真路径最小复现（**128-token 首 chunk，`ctx=0`**）：
   `int8_vs_bf16_equal = false`、**`max_abs = 1.324`**（32K/128K/1M 与 decode 都不碰）。
   ⇒ **上生产前必修**（一行：键里加 `role`），本任务的 kernel 已分键、**`dsa_v41.py` 还没改**，见 §3.4.1。

---

## 1. ★ cannbot 对照（AGENTS.md §6：写 kernel 前必查）

在 A3-node1 只读查 `~/projects/dsv41/src/cannbot/vendor/cannbot-skills/`（`model/` 与 `ops/` 两棵树）。
本任务重点查 **`model-infer-kvcache`** 与 **`model-infer-fusion`** 里"**prefill 期 KV 量化**"的做法。

| 查的地方 | cannbot 说什么 | 我们采纳 / 没采纳及理由 |
|---|---|---|
| **`model-infer-kvcache/SKILL.md`（全文 `grep -i prefill`）** | 与 prefill 相关的**只有布局与 mask**：①*"部分模型 Prefill 直读 KV（不传 `block_table`）、Decode 走 Paged 模式"*（`references/fa-code-examples.md:3`）；②*"Prefill 与 Decode 的 FA 调用不同：Prefill 不传 `block_table`，直接对 `key_states`/`value_states` 算注意力（再写 cache）"*（`references/framework_kv_reference.md:259-283`）；③`sparse_mode` 表：**Prefill 与 Decode 统一 3（因果）或 4（band/滑窗）**，两者都要 `[2048,2048]` mask（`SKILL.md:233-234,248`）；④*"长序列 `KV_len > sliding_window` 的正确性必须靠模型层保证——环形 buffer 写 cache、或 `actual_seq_lengths_kv` 截断到窗口长度，**不是 op 层负责**"*（`SKILL.md:241`） | **采纳 ③④（逐字）**：我们一个 mask 参数都没改（`ori_mask_mode=4` / `ori_win_left=127` / `cmp_mask_mode=3`），滑窗的"读哪一段"完全按 ④ 由**模型层**用 scratch+block table 表达。**没采纳 ①②**：V4.1 走的是 `npu_sparse_flash_mla` 的 **PA_BBND 唯一实例**（015 §3 实测 TND-KV 无 kernel），prefill 也必须走 block table，没有"直读 KV"的形态可用 |
| **★ 你要问的"prefill 期 KV 量化的推荐做法"** | **cannbot 里没有这一条**：`grep -rn "prefill" model-infer-quantization/` 无命中，`references/quantization-structure-cards.md` 的 C8 卡片只讲 decode 的 cache layout / scale / kernel 一体；`model-infer-kvcache` 也没有"量化 KV 在 prefill 怎么读"的段落 | **记录为"文档空白"**：本任务的方案不是抄来的，是**按 ③④ 的语义 + 我们的实测**推出来的：prefill 的窗口并集 = `[seq_len - min(seq_len, q_len + window), seq_len - 1]`，**整段重建进 scratch、block table 只重映射这一段、逻辑下标不动** |
| `model-infer-fusion/SKILL.md` + `references/torch_npu_API/torch_npu_list.md` | 融合第一步要"逐子链路匹配现成算子"；量化族只有 `npu_anti_quant`（**scale 必须 1 维**，023 §1 已实测 `EZ1001`）、`npu_quant_scatter`、`npu_kv_quant_sparse_flash_attention`（**`q_head_dim` 只支持 576**） | **逐条否决**（与 023/026/028 同结论）：没有 `gather+dequant+scatter`；唯一能"不重建"的 `kv_quant_sparse_flash_attention` 卡 576。⇒ **自己写 Triton kernel** |
| **`model-infer-graph-mode/SKILL.md`** | *"Decode 图模式、**prefill 保持 eager**"*、固定张量图外预创建 | ★ **这条直接决定了本任务的优化方向**：prefill **在 eager 帧里**，所以"多 30 个算子 + 每步 4 次 D2H 同步"都会**直接落在 step 时间上**（§3.3 实测：`.item()` 一次同步 = **+1.6 ms/step**）。⇒ 融合 kernel 的判据不是"算子个数"而是"**launch 数 + 有没有同步 + 有没有动态形状**" |
| `ops/triton-op-coding/SKILL.md`、`triton-latency-optimizer/references/docs_triton_IR/docs_triton_ascend/.../10-mem-ops.md:188-260,323-410` | Triton-Ascend 编码规范；有 `index_select_simd` / `gather_out_to_ub` 等 mem-ops | **采纳规范、没用 mem-ops**：直接算地址的 `tl.load/store` 已经够（与 026 §1 同判断），且 mem-ops 的"index 必须 1D / dim 不能是最后一维 / 不查越界"三条限制在 prefill 的二维 tile 上不划算 |
| `ops/torch-ops-profiler/examples/layer_norm_profiler_reference/`（四文件模板） | "自定义算子 vs 标杆"双路径 profiling 模板 | **本次没走 profiler**：prefill 的判据是"每 step 增量"，用**配对计时**（§3.3）直接量比数算子更贴题；模板留作 AscendC 路线的入口 |

**一句话**：cannbot **没有** prefill 期 KV 量化的推荐形态（这是文档空白），
但它给了两条**决定性**的边界条件——**prefill 保持 eager**（⇒ 不能有同步/多 launch）
和**滑窗正确性由模型层保证**（⇒ 我们重建的是"模型层认定的那段窗口"），本任务的实现就是这两条的落地。

---

## 2. prefill 问题回顾（起点：015 §5.3 + 018 §2.1B）

015 §5.3 的实测：2048-token chunk 下，**每个 query 行各自挑 512 个压缩 token**，
⇒ 一层要 gather `2048×512` 行 ⇒ 4 层 **45 ms（flat）/105 ms（strided）每 step（+10~26%）**。**那条路不可用。**

018 §2.1B 换了形态：**页粒度整前缀重建**（每请求 `ceil(cache_seq_len/128)` 页全搬进 scratch，
`block_table' = arange`、**`cmp_sparse_indices` 保持真逻辑下标** ⇒ 因果 mask 语义不变）。
它**确实绕开了行放大**（每层只搬 `Lc×520 B` 读 + `Lc×1024 B` 写），
但 018 从没量过它的耗时。**本次第一步就是把这一格补上（§3.1）**。

---

## 3. prefill：三条候选的实测判定

### 3.1 【实测】第一步：018 的页粒度重建到底要多久

`p1_prefill.py`（eager，每次调用 `torch.npu.synchronize()`；单请求；2048 行 chunk；32K 上下文
⇒ 压缩前缀 **136 页**、窗口并集 **18 页**）：

| 臂 | 一层 + 算子（eager，32K） | 说明 |
|---|---:|---|
| BF16 平面（现状，无重建） | **1,673 µs** | 基线 |
| **018 的页粒度重建（torch）** | **3,010 µs** | 增量 **+1,332 µs/层** |
| 拆开：窗口面重建（`kv8_ori_plane`） | 931 µs/次 | 单独调用 |
| 拆开：压缩面重建（`_kv8_cmp_plane` prefill 支） | 407 µs/次 | 单独调用 |
| 015 的**行放大**路线（对照，`RowImpl`） | **16,203 µs/层**（窗口 931 + 行放大压缩面 ≈15,272） | 复现 015：压缩面一项就 **+15.3 ms/层 ⇒ 不可用** |

**⇒ 018 的形态是对的（比行放大快 ~12 倍），但用 torch 写仍是 +1.3 ms/层。**
按 40 层访问（36 SWA-only + 4 带 long-KV）外推 = **+40.9 ms/step（400 ms 基线的 +10.2%）** ⇒ **仍不可接受**。

### 3.2 三种写法（本任务的实现）

代码：`a2/agents/KV8_prefill/kv8_prefill_triton.py`（3 个 kernel + 2 个 wrapper）；
接进影子包：`kv8_prefill_wiring.py`（**`VLLM_V41_KV8_PREFILL=1` 才生效，decode 一行不动**）。

| 版本 | 做法 | 结果 |
|---|---|---|
| **v1** | 照 026/028 的 decode kernel 改形状（wide tile `[BR,DIM]`、grid `(B, PPR, BS/BR)`），页数由 host 端 `nb.max().item()` 算 | 能用，但**每次调用一次 D2H** |
| **v2** | 索引算术搬进 kernel（它自己读 `seq_lens` / `query_start_loc` 算 `first_block`/`nb`），**一次 launch** | +18.8 ms/step（仍差） |
| **★ v3** | **页数从 host 的 `max_cache_seq_len`（纯 int，无同步）拿**，kernel 只读 device 侧的 `nb` 做 clamp；压缩面 `ppr` 由上界给出、**不做逐 program 掩码** | ✅ **+0.58 ms/step（host-bound 上界）/ +0.00~0.06 ms（设备饱和，生产形状）** |

**v3 的三条设计点（都是实测逼出来的）**：

1. **绝不做 D2H 同步**：`.item()` 取"真实前缀页数"看着只多一次拷贝，实测**+1.6 ms/step**
   （每一步 4 个压缩层各一次，host 要等设备队列排空）。改用 `metadata.attention.max_cache_seq_len`
   （builder 从 CPU 侧长度算出来的 int）后归零。**同理**：窗口面的页数用 `query_rows`（是个 shape，不是数据）。
2. **绝不做逐 program 标量掩码**：`tl.load/store(..., mask=(j<nb))` 让**所有** load/store 变成谓词形式，
   实测 **351 → 1655 µs/visit（慢 5×）**。改成"上界只用来开 scratch、kernel 里读 `min(j, nb-1)`"后正常。
3. **scratch 必须按角色分键**（见 §3.4 的 bug）。

### 3.3 【实测】两张口径下的每步增量

同一份代码，两种计时口径（都在 2048 行 chunk / 32K 上下文）：

| 口径 | 做法 | 用途 |
|---|---|---|
| **pipe（host-bound 最坏）** | 44 层访问连续下发，**一个序列只同步一次** | 这是"host 追不上设备"时的上界；也是 026/028 用的口径 |
| **配对抗（设备饱和，生产形状）** | `wall(忙设备op + k 次调用)` 两臂**交替成对**测，取中位数之差 ÷k | A2 的 prefill 每层有 ~9–11 ms 设备活儿（400–480 ms/step ÷ 44），设备是瓶颈 ⇒ **这一格才是生产值** |

| 臂 | pipe（每 step） | 设备饱和（每 visit） | 设备饱和（每 step，36 SWA + 4 全层） |
|---|---:|---:|---:|
| BF16（现状） | 10.19 ms | 3,621 / 5,037 µs | — |
| **018 的 torch 页粒度重建** | **47.6 ms（+37.4 ms，+9.4%）** | 窗口 **+988**、压缩 **+342** µs | **+40.9 ms（+10.2%）** |
| **本任务融合 v3** | **10.77 ms（+0.58 ms，+0.145%）** | 窗口 **−3.1 µs（≈0）**、压缩 **+28.6 µs** | **+0.00~0.06 ms（≤+0.015%）** |

* 配对抗的**边际**一格更干净：`full_swaonly` vs `full_kv8`（两臂只差压缩面）= **+14.2 µs/visit**
  ⇒ 4 个压缩层 = **+57 µs/step**；而窗口面 40 层 **−3.1 µs/visit ≈ 0**
  （两次独立进程给出 +63 µs 与 +1.5 µs，都在 ±0.2 ms 的噪声地板内）；
* 判据对账：**≤+0.5%/step（≈2 ms）✅ 两个口径都过**；
  **≤+150 µs/step ✅ 设备饱和口径（57~150 µs）**、❌ host-bound 口径（580 µs，但那是最坏上界）。

### 3.4 【实测】正确性：逐比特 + 顺手抓的 scratch 别名 bug

`p5_correct.py` / `p10_wired_check.py`（**走真影子包路径**，不是猴子补丁；无损量化器 + 因果安全的 top-k，
与 020 §4/§5 同手法）：

| 对拍 | 结果 |
|---|---|
| 走线后的融合重建 vs BF16 生产路径（512 行 chunk） | **`torch.equal` = True**、`max_abs = 0.0` |
| 同上（2048 行 chunk） | **`torch.equal` = True**、`max_abs = 0.0` |
| 同上（3 行 chunk） | **`torch.equal` = True**、`max_abs = 0.0` |
| **故意把 scratch 页数放大到 3×**（生产最坏情况） | 修 bug 前 **`max_abs = 1.13`**，修 bug 后 **`== `True`** |
| 三个重建产物的逐比特对拍（scratch 面 / scratch 表） | `torch.equal` 全 True（p4_diag，qrows 3/64/512/2048） |

### ★ 3.4.1 顺手抓到的**静默正确性 bug**（已在真路径复现，必修项）

**现象**：`dsa_v41.kv8_scratch_plane` 的复用键是 `(blocks, block_size, dim, dtype, device)`
（`dsa_v41.py:255-262`）。窗口重建与压缩重建**只要页数相同就拿到同一块张量**，
而两者都要在算子读之前写完 ⇒ **后写的覆盖先写的**，**没有任何报错**。

**触发条件（`p12_scratch_bug.py` 扫的 (chunk × ctx) 网格，33 格）**：
只有当 `ceil(min(seq_len, chunk+128)/128) == ceil((seq_len//2)/128)` 才碰，
即 **`chunk ≤ 128 且 ctx = 0`（session 的第一个小 chunk）**；
32K / 128K / 1M 上下文配 128~8192 的 chunk **全部不碰**（比值 8×~2048×）；decode 也不碰（窗口 2 页 vs topk/128 ≥ 4 页）。

**最小复现（真路径，18 s）**：

```bash
# 复现（128-token 首 chunk）：int8_vs_bf16_equal=false, max_abs=1.324
KV8_QROWS=128 KV8_CTX=0 python3 p13_alias_repro.py
# 对照（512-token chunk，页数 1 vs 4）：equal=true, max_abs=0.0
KV8_QROWS=512 KV8_CTX=0 python3 p13_alias_repro.py
```
（脚本：`a2/agents/KV8_prefill/p13_alias_repro.py`，走 **`KV8_gather`/`KV8_swa` 的原始影子包**、
无损量化器、因果安全 top-k。原始数据：`raw/033-p13-alias-128.json`、`raw/033-p13-alias-512.json`。）

| 形状 | 窗口页 | 压缩页 | 同一个 tensor？ | vs BF16 生产路径 |
|---|---:|---:|---|---|
| **128-token 首 chunk** | 1 | 1 | **是** | ❌ `max_abs = 1.324`（rms 0.703 → 0.105） |
| 512-token chunk | 4 | 2 | 否 | ✅ `max_abs = 0.0`（逐比特） |

**一行修法**（`dsa_v41.py:255`）：

```python
# 现在
def kv8_scratch_plane(blocks, block_size, dim, dtype, device):
    key = (blocks, block_size, dim, dtype, str(device))
# 改成
def kv8_scratch_plane(blocks, block_size, dim, dtype, device, role="decode_cmp"):
    key = (role, blocks, block_size, dim, dtype, str(device))
# 调用点：kv8_ori_plane -> role="swa"；_kv8_cmp_plane（decode/prefill 两条分支） -> role="cmp"
```
本任务的 `kv8_prefill_triton._scratch()` **已经是这个形态**（键含 role）。
★ 但**必须注意**：我改的只是**自己 kernel 模块**的私有 helper；
**`dsa_v41.kv8_scratch_plane` 仍是旧键** ⇒ **decode（026/028 那条）与 prefill 的 torch 回退路径都还没修**。

**生产形状下会不会触发？** —— **会**，但**只在"每个新 session 的第一个 ≤128-token chunk"**。
长上下文压测（32K/128K/1M）与 decode 都看不见它，所以它是"**必须专门测才看得见**"的错：
修掉它的成本是一行 + 三个调用点，**建议直接进必修清单**。

### 3.5 【实测】kernel 下限：还没贴地，但不阻塞

`p8_shape.py`（单次调用 wall + 形状扫描）：

| 项 | 值 |
|---|---|
| 压缩面重建（BR=16） | **240 µs/次**（`copy` 同字节数的**连续 int8→bf16 转换**只要 **56 µs**）⇒ 还有 ~4× 空间 |
| 窗口面重建（BR=8/16/32） | **94~96 µs/次**（同字节数连续转换 ~7 µs） |
| `BR=4` | ⛔ `coreDim is invalid`（grid 变成 0） |
| `BR=32`（压缩面） | ⛔ `PlanMemory Failed`（UB 放不下，与 028 坑 ⑤ 一致） |

⇒ 两个 kernel 目前是 **26.7 MB / 240 µs ≈ 111 GB/s**（压缩面）与 **3.3 MB / 95 µs ≈ 35 GB/s**（窗口面），
**离 1161 GB/s 的拷贝上限还有 4~30×**。但在"设备饱和"口径下它们**已经被算子的计算盖住**（窗口面 ≈0），
所以**不阻塞上线**；后续若要再挤，方向是"合并成一次 launch / 提高每 program 的行数 / 用 `index_select_simd`"。

---

## 4. ★ 容量：五个档位**全是真引擎读出来的 `GPU KV cache size`**

### 4.1 口径与做法

* 环境与 013/021 逐字一致：**单卡 c1、tiny dummy 权重**（`agents/L1_dummy/models/model-tiny`）、
  TP1、`--block-size 128`、`--kv-cache-memory-bytes 1 GiB`、`ENGRAM=0`、`--load-format dummy`；
* 起服脚本 `a2/agents/KV8_prefill/serve_arm.sh`（**在槽位容器内直接 `vllm serve`，不新起 docker**）；
* 每臂只改"平面 dtype / ring dtype"两个开关，其余参数逐字相同；读的是服务启动日志里
  `vllm/v1/core/kv_cache_utils.py:2235` 那一行 **`GPU KV cache size: N tokens`**；
* **单臂 ≈ 70 s**（起服 60–65 s）。

### 4.2 ★ 五格结果（【实测】）

| 臂 | 平面 | ring | **`GPU KV cache size`** | 倍数 | 按页几何解析 | 倍数 | 起服到 `/health`？ |
|---|---|---|---:|---:|---:|---:|---|
| `pf-bf16` | BF16 | FP32 | **22,719** | ×1.0000 | 540,928 B/block | ×1.0000 | ✅ rc=0 |
| `pf-ring16` | BF16 | **FP16** | **22,719** | **×1.0000** | 540,928 | ×1.0000 | ❌ 死在分配之后 |
| `pf-kv8` | **INT8 双平面** | FP32 | **25,801** | ×1.1357 | 476,416 | ×1.1354 | ✅ rc=0 |
| `pf-swa-ring16` | **只 SWA INT8** | **FP16** | **33,295** | ×1.4655 | 369,280 | ×1.4648 | ❌ 死在分配之后 |
| **`pf-kv8-ring16`** | **INT8 双平面** | **FP16** | **43,469** | **×1.9133** | 282,880 | ×1.9122 | ❌ 死在分配之后 |

**⇒ 判据（实测比值 ≈ ×1.91 ±5%）= ×1.9133 ✅。**
五格与页几何解析**逐格吻合（偏差 <0.05%）**，且与 `R_ringshrink`（日志 034）独立算出的
`540928 / 540928 / 476416 / 369280 / 282880` 完全一致 ⇒ **两条独立路径互证，口径可引用**。

**★ 一句必须写进任何引用这份数据的地方**：
> **倍数是"页几何"决定的，可以外推；绝对值是 tiny/1 GiB 池的口径，不能当生产数字。**

### 4.3 ★ 每个臂的失败阶段与原因（主代理点名要的）

先把五臂的成败讲清楚（`raw/033-capacity-server.log` 是原始抓取）：

| 臂 | 起服阶段 | `GPU KV cache size` 行 | 死因 | 数字能不能用 |
|---|---|---|---|---|
| `pf-bf16` | ✅ 完全健康（rc=0） | 18:47:52 | — | ✅ |
| `pf-kv8` | ✅ 完全健康（rc=0） | 18:48:41 | — | ✅ |
| `pf-ring16` | ❌ **分配成功、1 秒后死** | 18:54:01 | `ValueError: Compressor requires contiguous projections, ring pages and controls` | ✅ 容量行有效，**可服务性未确认** |
| `pf-swa-ring16` | ❌ 同上 | 18:55:16 | 同上 | ✅ 同上 |
| `pf-kv8-ring16` | ❌ 同上 | 18:52:47 | 同上 | ✅ 同上 |

**逐条回答（这三问的答案就是"能不能引用"）**：

1. **失败发生在什么阶段？** `GPU KV cache size` 那一行由 `kv_cache_utils.py:2235` 在**KV 分配阶段**打印，
   它在 **model runner 的 profile run 之前**。三个 ring16 臂的日志显示：**先打出容量行，
   1 秒后在 model-runner 初始化（profile run 里跑一次 compressor）时抛异常**。
   ⇒ **容量数字是"分配器按页几何算出来的"，与后面的崩溃无关**；但**这三个臂没有起服到 `/health`**，
   所以**不能**用它们声称"端到端可跑"。
2. **`pf-bf16` 也 fail 了？** 那条 `pf-bf16.server.fail.log` 是 **18:46:25 那一次的行**，
   错因是**我自己的 harness 缺陷**：`PYTHONPATH=` 覆盖了 CANN 的 `set_env.sh` 注入路径 ⇒
   `ModuleNotFoundError: No module named 'acl'`。**18:47:52 的那次（同一份脚本，改成一：`PYTHONPATH=影子包:$PYTHONPATH`）
   完全健康、rc=0**。⇒ 那是**过期残留文件**，不是配置问题。
3. **和 R_ringshrink 报的 `core/deepseek_v41.py:292` 断言是同一类吗？** **是同一类，但不是同一处**。
   要让 ring 变成 FP16，**至少 4 处硬编码/断言**要动（我这份"测量用"改动在 `ring16_measure_patch.py`，**只改影子包**）：
   | # | 位置 | 原文 | 改法 |
   |---|---|---|---|
   | 1 | `core/deepseek_v41.py:129`（`__post_init__`） | `if self.dtype != torch.float32 or ...: raise ValueError("Aurora state requires a 32-row FP32 uncompressed ring")` | `not in (float32, float16)` |
   | 2 | `core/deepseek_v41.py:349`（`reshape_cache`） | `if isinstance(spec, DeepseekV41CompressorStateSpec) and sum(plane_sizes) != block_stride: raise ValueError("...must fill its slot with 32 contiguous FP32 rows")` | `sum(plane_sizes) > block_stride`（缩了之后**不再填满**槽，SWA 别名页接管） |
   | 3 | `models/deepseek_v41/compressor.py:49` | `if spec.dtype != torch.float32 ...: raise ValueError("V4.1 compressor state requires a 32-row FP32 ring")` | `not in (...)` |
   | 4 | `ops/triton/compressor/compressor_triton.py:652` | `if kv.dtype != float32 or scores.dtype != float32 or state_cache.dtype != float32: raise ValueError("Aurora projections and ring state must be FP32")` | `state_cache.dtype not in (float32, float16)` |
   ★ **但真正卡住的是第 5 条（不是断言，是视图语义）**：
   `reshape_cache` 用**槽页 stride**（131072 B）建视图；ring 减半后
   `stride(dim0) = 131072/2 = 65536` 元素而一个块只有 `32×1024 = 32768` 元素
   ⇒ 视图**非连续** ⇒ `compressor_from_projected` 的 `all(t.is_contiguous())` 直接判死
   （`ValueError: Compressor requires contiguous projections, ring pages and controls`）。
   ⇒ **给 034 的结论**：ring 缩 FP16 **不只是"改常量+断言"**，还要让 `reshape_cache` 为 state 平面
   返回一个 `is_contiguous()==True` 的视图（或给 state 平面单独的 block stride / 放宽 compressor 的连续性检查）。
   **只改 dtype + 4 处断言 ⇒ 能过分配、过不了起服**（这三臂的实测就是证据）。

### 4.4 ★ 必须先说的分母口径（主代理要求显式写）
 
> **如果只挂 KV8 而没缩 ring，会得到 ×1.1357（本任务的 `pf-kv8` 实测）甚至 ×1.032（只量化 long-KV、不量化 SWA 时，见 018/020）
> ——那不是 KV8 的问题，是分母被 FP32 state ring 顶住**：前 3 个 ratio-2 槽的页大小由
> `max(long_kv+index, state_ring, swa_alias)` 决定，ring=131072 B 时**永远平手或更大**。
> ⇒ **"ring 缩 FP16"是 KV8 拿容量的第二条独立前提**（生产改动归 034），两条都成立才是 **×1.9133**。

---

## 5. ★★ KV8 vs "只缩 ring" / "只量化 SWA" 的决策树

### 5.1 成本-容量总表（全部取自实测，decode 增量引自 028 的 40 层整图口径）

| 档位 | 容量（实测比值） | decode 增量（30 ms 步） | prefill 增量（2048-token 步） | 未解障碍 |
|---|---:|---|---|---|
| 现状（全 BF16 + FP32 ring） | ×1.0000 | — | — | — |
| **只缩 ring（FP16）** | **×1.0000** | ~0 | ~0 | **白做**（第 4 槽 binding） |
| **SWA INT8 + ring16** | **×1.4655** | **+466 µs（+1.55%）**【本任务 §5.4 实测：+12.05 µs/层 ×40】 | **≈0**（本任务，§3.3） | 无（不碰压缩面，**完全不依赖 prefill 那条命门**） |
| **KV8 双平面 + ring16** | **×1.9133** | **+644 µs（+2.15%）**【本任务 §5.4 实测：+44.35 µs/层 整层；与 028 的 +45.57/+0.684 ms 逐位复现】 | **+0.00~0.58 ms（≤+0.015% ~ +0.145%）**（本任务，§3.3） | 无（本任务已解） |

**边际账（KV8 全量 − SWA-only）**：容量 **×1.306**（33,295 → 43,469）、decode **+178 µs/step（+0.59%）**、
prefill **+57 µs/step（+0.014%，设备饱和口径）**。

### 5.2 判据对账（主代理给的"值不值"判据）

> 判据：**如果 prefill 的成本不能压到 ≤+0.5%/step（或 ≤ +150 µs/2048-token step），
> 就建议 KV8 降级为"未来候选"，把 ring 缩 FP16 作为第一优先级上线项。**

| 判据 | 实测 | 判定 |
|---|---|---|
| prefill ≤ **+0.5%/step**（400 ms ⇒ ≤2 ms） | **+0.06~0.58 ms** | ✅ **达标**（余量 3.4~33×） |
| prefill ≤ **+150 µs/step** | 设备饱和口径 **≤60 µs** ✅；host-bound 上界 **580 µs** ❌ | ⚠️ **视口径**（见下） |
| decode ≤ 判据（028 定的是 ≤+60 µs/层） | **+45.57 µs/层**（028） | ✅ |

**★ 结论：KV8 不降级。** 理由：

1. prefill 的**命门已经解开**（015 的 45–105 ms ⇒ 本任务的 0.00–0.58 ms，降 75–175×），
   而且**正确性是逐比特的**（§3.4），所以"KV8 只能在 decode-only 场景用"这个兜底结论**不需要用**；
2. **KV8 相对 SWA-only 的边际成本只有 +178 µs/step（+0.59%，decode）+57 µs（+0.014%，prefill），换来的是再涨 ×1.306 容量**（33,295 → 43,469）；
3. **SWA-only 档也不能单独上**：它同样需要 **ring 缩 FP16 才有容量**（`pf-swa-ring16` = ×1.4655；
   不带 ring16 时按页几何只有 **×1.135**，见 020 §2 的实测）——所以"先上 SWA-only"**并不能绕开 034**，
   它只是**少一条技术链**（不需要压缩面重建）。
4. ⇒ **建议的落地顺序**（与主代理的判断一致，只是把 KV8 从"未来候选"提到"同一批"）：
   **① ring 缩 FP16（034，必须先做，且要修 contiguity：§4.3-⑤）→ ② SWA 量化（020/028 已有，×1.4655）
   → ③ KV8 压缩面量化 + 本任务的 prefill 融合（×1.9133）**。
   "只缩 ring"**不要单独上**（×1.000，白做）。

### 5.3 什么情况下要回退到 SWA-only

| 触发条件 | 回退动作 |
|---|---|
| 真权重下 KV8 的精度超门（`rel_L2` 远超 5.43e-3，或 GSM8K/Vision 退化） | 退到 SWA-only（少一个量化面，容量 ×1.4655） |
| 端到端步时显示 prefill 增量远超 +0.5%（说明"设备饱和"假设不成立，A2 的 prefill 是 host-bound） | 先查 `A2 prefill 每层 host 时间`：若确实 host-bound，退到 SWA-only（prefill ≈0）或继续优化 §3.5 的 kernel |
| 034 的 ring 改动最终做不成 | KV8 只剩 ×1.135（020 实测）⇒ **按 025 的判据应整体收档**（不值得为 13.5% 付 2.3% 时延） |

### 5.4 ★【实测】decode 侧：SWA-only 中间档与 KV8 全量（本任务亲测，40 层整图口径）

`p11_decode_arms.py`（= 028 的 p5_e2e 逐字搬来 + 加"long-KV BF16 + 窗口 INT8"这一档；
**跑在 `VLLM_V41_KV8_PREFILL=1` 之下，即本任务的 prefill wiring 已生效**，
所以它同时是"prefill kernel 没有打扰 decode 帧"的回归测试）：

| 臂（B=8 / topk=512 / 40 层一图 / 中位） | µs/层 | 增量 µs/层 |
|---|---:|---:|
| `swa_bf16`（现状） | 26.35 | — |
| `swa_int8_fused`（窗口量化，028 的 kernel） | 39.31 | **+12.96** |
| `swaonly_swa_fused`（**long-KV BF16 + 窗口 INT8**） | 38.40 | **+12.05** |
| `full_bf16`（现状） | 45.73 | — |
| `swaonly_full_fused`（同上，带 long-KV 的层） | 53.79 | **+8.06** |
| `full_int8_fused`（KV8 双平面） | 90.08 | **+44.35**（其中 cmp **+31.39**） |
| `full_int8_current`（018 的 torch rebuild，对照） | 399.15 | +353.42 |

**整步外推（40 SWA + 4 源层，30 ms 步）**：

| 档位 | ms/step | 占比 |
|---|---:|---:|
| 018 的 torch rebuild（对照） | +6.517 | **+21.7%** |
| **SWA-only（本任务中间档）** | **+0.466** | **+1.55%** |
| **KV8 双平面** | **+0.644** | **+2.15%** |
| 028 的同一格（历史值） | +0.684 | +2.15% |

* **复现性**：`full_int8_fused` 的 `torch.equal` vs 现状 = **True**、真量化器 `rel_L2 = 5.4331e-3`
  （与 028 逐位相同）；±10% 内逐格复现 028 的五个读数 ⇒ **prefill wiring 对 decode 零影响【实测】**。
* **中间档的代价就是窗口重建那一份**：`+12.05 µs/层 × 40 = +466 µs/step`，
  而它**完全不碰压缩面**，所以**不依赖 §3 的任何 prefill kernel**（那些只服务压缩面与窗口面的 prefill 形状）。
  ⚠️ 注意：窗口面**在 prefill 也要重建**（那是 prefill 的窗口并集），
  这一项由本任务的 §3.2 融合 kernel 承担（设备饱和口径 ≈0）。

---

## 5.5 ★★ 最终推荐配置（二选一，已选定 B）

任务书给的两档（L5 = DRAM 卸载、L1 = pinned pool，引自 016/022 已实测的 ×3.75 总账）：

| | **档 A（保守）** | **档 B（推荐 ✅）** |
|---|---|---|
| 组成 | L5 + L1 + **SWA-quant + ring16** | L5 + L1 + **KV8 全量 + ring16** |
| 本任务实测的容量档 | `pf-swa-ring16` = **×1.4655** | `pf-kv8-ring16` = **×1.9133** |
| × L5+L1（第三方实测 ×3.75） | ≈ **×5.50** | ≈ **×7.17** |
| decode 增量（30 ms 步） | **+466 µs（+1.55%）** | **+644 µs（+2.15%）** |
| prefill 增量（2048-token 步） | **≈0**（不碰压缩面；窗口面 prefill 仍走融合 kernel） | **+0.00~0.58 ms（≤+0.015% ~ +0.145%）** |
| 未解技术障碍 | 仍**必须**先做 034（ring16），否则只有 ×1.135 | 同左 + **scratch 键必修项**（§3.4.1） |
| 精度地板 | 020/028：SWA 加进来后 `rel_L2 5.46e-3`（vs 只 long-KV 的 5.42e-3） | 028/本任务：`rel_L2 = 5.4331e-3`（逐位复现，零回退） |

**★ 结论：选档 B。** 用本任务的数字确认主代理的倾向：

* **边际账**：B 相对 A **只多 +178 µs/step（+0.59%）decode**、prefill **多 ≤0.58 ms**，
  换来容量再 **×1.306**（33,295 → 43,469，即总倍率 ≈×5.50 → ≈×7.17）；
* 而且**"prefill 已解"这一条是本任务实测的**（§3.3/§3.4），
  档 A 省掉的只是"压缩面 prefill 重建"这条链，而它的成本在设备饱和口径下已经 **≈0**（≤60 µs/step）；
* **两档都需要 ring16**（`pf-ring16` 单独 = ×1.000）⇒ **034 是共同前提，不构成 A 相对 B 的优势**；
* A 相对 B 的真实优势只剩一条：**少一个被量化的面**（精度风险面更小）。
  如果真权重端到端（GSM8K/Vision）显示 KV8 的精度超门，那就退到 A——**这就是 A 的定位：B 的降级档**。

### 档 B 的前置条件清单（上线前必须逐条打勾）

| # | 前提 | 状态 | 谁负责 |
|---|---|---|---|
| 1 | **ring 缩 FP16（034）**，且修好 ring 视图的**连续性**（不改连续性 ⇒ model-runner profile run 直接死，§4.3-⑤） | 【实测】分配阶段已验证（×1.9133），**起服阶段未过** | `R_ringshrink` / 034 |
| 2 | **`kv8_scratch_plane` 按 role 分键**（§3.4.1），decode + prefill 两条路径都要 | 【实测】真路径复现 `max_abs=1.324`；**未修** | 集成（035 / `X_integrate`） |
| 3 | **prefill 融合 kernel 接进影子包**（`VLLM_V41_KV8_PREFILL=1`），并保留**回退开关** | 【实测】已接、逐比特、+0.00~0.58 ms | 本任务（已交付） |
| 4 | **prefill 回退条件**（出现即自动退回 torch 页粒度重建，慢但不崩）：`query_rows == num_reqs`（decode 形状）、非 2 的幂几何、`per_req*block_size != topk` | 【实测】wiring 里已实现（与 026/028 同款判定） | 本任务（已交付） |
| 5 | **精度地板**：算子级 `rel_L2 = 5.4331e-3`、`cos = 0.9999857`、`nan = 0`（真量化器，B=8/topk=512/40 层整图口径） | 【实测】028 + 本任务逐位复现 | 本任务（已交付） |
| 6 | **端到端质量**（GSM8K-200 / Vision 23）与**真权重 step 时间** | **【未确认】** | 见 §7 |

### ★ 一条"总倍率"口径提醒

上表的 ×3.75（L5+L1）与 ×1.9133（KV8+ring16）**不是同一台机器测的**，且 ×3.75 是"池容量"口径、
×1.9133 是"GPU KV cache size"口径；**相乘只在"两者作用在同一份 KV 预算上"时才成立**，
这条乘法**没人实测过**（`X_integrate`/035 正在做）⇒ 表中写成"≈"。

---

## 6. 判定总表（对照本任务书的四步）

| # | 判据 | 实测 | 判定 |
|---|---|---|---|
| ① | 读 018 的 prefill 路径：它解决了没有、慢多少 | **避开行放大成功**；但 torch 版 **+40.9 ms/step（+10.2%）** ⇒ **没解决** | ✅ 已答 |
| ② | prefill 增量 ≤+1%（任务书）/ ≤+0.5%（主代理）/ ≤+150 µs/step | **+0.014~0.145%**（0.06~0.58 ms） | ✅ **达标** |
| ③ | 实测 `GPU KV cache size` 比值 ≈ ×1.91（±5%） | **×1.9133**（43,469 / 22,719） | ✅ **达标** |
| ④ | 端到端（真权重 8 卡 GSM8K/Vision） | **未做**（8 卡要等 L3 释放；本任务预算 3.5 h） | 【未确认】 |
| ⑤ | 正确性 | 逐比特（`torch.equal` True，无损量化器 + 因果安全 top-k，走真影子包路径） | ✅ |

---

## 7. 未确认清单 → **每格一条命令**（不留无边界的话）

| # | 未确认 | 现状 | **后续一条命令** | 预期判据 |
|---|---|---|---|---|
| ① | **三个 ring16 臂能不能真起服** | 分配阶段成功（容量行有效）；model-runner 死在 `Compressor requires contiguous...` | 先做 034 的"连续视图"修法，然后：`bash tools/a3_chip.sh c1 --timeout 900 -- env TAG=cap-kv8r16 VARIANT=kv8_ring16 bash /work/agents/KV8_prefill/serve_arm.sh` | `/health` 200 + `grep "GPU KV cache size"` = 43,469 |
| ② | **真权重端到端质量**（GSM8K-200 / Vision 23）与真 step 时间 | 未做（8 卡要等 L3） | 8 卡空闲后按 016/022 的 `serve_a2.sh` 走 A/B（`VLLM_V41_KV8=1 VLLM_V41_KV8_SWA=1 VLLM_V41_KV8_PREFILL=1` vs 不设） | GSM8K 197–199 / Vision 23/23；step 增量 ≤ +2.5% |
| ③ | **A2 上 prefill 是 host-bound 还是设备饱和** | 本任务的"设备饱和"是按"A2 400–480 ms/step ÷ 44 层"推的【推断】 | A2 上跑 `torch_npu.profiler` 打一层 prefill 的 host/device 时间（脚本 `p3_final.py` 同款，换成 A2 的真权重配置） | 若 host/device > 1 ⇒ 用 host-bound 口径（+0.58 ms/step），仍达标 |
| ④ | **其它 chunk / 上下文的 prefill 增量** | 只测了 2048×32K×1 请求 | `KV8_QROWS=512 KV8_CTX=8192 python3 p9_paired.py`（再换 4096/65536） | 每 visit ≤ +35 µs（即 4 层 ≤ +140 µs/step） |
| ⑤ | **scratch 别名在 decode 侧是否也存在** | 【实测】prefill 侧 128-token 首 chunk 复现（`max_abs=1.324`）；decode 侧未测 | `KV8_QROWS=128 KV8_CTX=0 python3 p13_alias_repro.py` 之后，用 018 的 decode 形状构造"窗口页数 == 压缩页数"（topk=1024 + 2 页窗口）再跑一次 | 若也复现 ⇒ 修法优先级升到 P0（decode 是热路径） |
| ⑥ | **KV8 的端到端精度**（5.4331e-3 是算子级） | 028/020 的算子级读数；本任务只做无损对照 | 同 ② | 端到端不退化 |
| ⑦ | **prefill kernel 的剩余带宽空间**（111 GB/s vs 1161 GB/s 上限） | 【实测】240 µs/次（压缩面）；下限 56 µs | 提高每 program 行数 / 合并成一次 launch（当前 `BR=32` 会 `PlanMemory Failed`） | 不影响上线；纯优化 |

---

## A2 可直接复制

> 给 A2 起服/验收用的一节，**只含"要复制的东西"**，不含推导。
> 三个开关：`VLLM_V41_KV8=1`（long-KV 双平面）、`VLLM_V41_KV8_SWA=1`（窗口面）、
> `VLLM_V41_KV8_PREFILL=1`（prefill 融合 kernel；**关掉就退回 018 的 torch 重建，慢但不崩**）。

### A2.1 起服参数（相对现状只多这三行 env）

```bash
export VLLM_V41_KV8=1            # long-KV 平面 INT8 payload + FP16 group scale（520 B/token）
export VLLM_V41_KV8_SWA=1        # 滑窗窗口平面同样 INT8（SWA 页 131072 -> 66560 B）
export VLLM_V41_KV8_PREFILL=1    # chunked prefill 走融合 kernel（+0.00~0.58 ms/step）
# 前提：ring 缩 FP16（034）已合入，且 scratch 键已按 role 分开（3.4.1）
```

* **`max-num-batched-tokens` / `block-size` / mask 参数 / 权重：一个都不用改**；
* **PYTHONPATH 只要前置影子包**，**不要把 `PYTHONPATH` 整体覆盖**（会丢 CANN 的 `acl`，见 4.3-2）；
* **回滚开关**：`unset VLLM_V41_KV8_PREFILL`（只回退 prefill，容量不掉，只是慢）；
  `unset VLLM_V41_KV8_SWA`（回退窗口面，容量掉到 x1.135）；
  `unset VLLM_V41_KV8`（整体回退 BF16 长上下文 KV）。
  **三个开关都是 import 期门控，不需要改文件、不需要重启容器之外的任何东西。**

### A2.2 起服自检（三条 grep，缺一不可）

```bash
# (1) 容量：判据 3.50M -> ~6.69M（x1.9133）；tiny/1GiB 口径是 22,719 -> 43,469
grep -aE "GPU KV cache size" "$LOG" | tail -3
# (2) 量化真的生效（页几何）：4 个 slot 之和应从 540928 降到 282880 B/block
grep -aE "pool_bytes_per_block|bytes_per_block|KV cache" "$LOG" | tail -5
# (3) prefill 融合真的挂上（VLLM_V41_KV8_PREFILL=1 时 kernel 模块应被 import）
python3 -c "import vllm_ascend.ops, sys; import vllm_ascend.attention.dsa_v41 as d; \
  print(d.kv8_ori_plane.__qualname__, 'vllm_ascend.attention.kv8_prefill_triton' in sys.modules)"
# 期望输出：kv8_ori_plane True
```

### A2.3 上生产前的三条硬检查（本任务的实测依据）

| 检查 | 判据 | 依据 |
|---|---|---|
| **容量** | `GPU KV cache size` 比值 **x1.91（±5%）** | 本任务 4.2【实测】43,469 / 22,719 = x1.9133 |
| **prefill** | 2048-token chunk 的**每步**增量 **≤ +0.5%（≈2 ms）** | 本任务 3.3【实测】+0.00~0.58 ms |
| **精度** | 算子级 `rel_L2 ≈ 5.43e-3`、`cos ≥ 0.99998`、`nan = 0` | 028 + 本任务【实测】逐位复现 |

### A2.4 目前**还不能**上生产的两条（必须先行）

1. **`kv8_scratch_plane` 的 role 分键**（3.4.1）：不修 ⇒ **每个新 session 的第一个 ≤128-token chunk 静默错**
   （真路径复现 `max_abs = 1.324`，不报错）。一行修法见 3.4.1。
2. **ring 缩 FP16 的视图连续性**（4.3-5）：只放宽断言 ⇒ **起服过不去**（model-runner profile run 报
   `Compressor requires contiguous projections, ring pages and controls`）。

## 8. 交付物与复现

### 8.1 交付物

| 东西 | 路径（本仓） | 容器内 |
|---|---|---|
| **prefill 融合 kernel**（3 kernel + 2 wrapper，含 v1/v2/v3 与形状参数） | `a2/agents/KV8_prefill/kv8_prefill_triton.py` | `/work/agents/KV8_prefill/kv8_prefill_triton.py` |
| **接进影子包**（`VLLM_V41_KV8_PREFILL=1` 生效，decode 不动） | `a2/agents/KV8_prefill/kv8_prefill_wiring.py` + `kv8_prefill_block.py` | 同 |
| **起服脚本**（单卡 tiny，读 `GPU KV cache size`） | `a2/agents/KV8_prefill/serve_arm.sh` | 同 |
| **ring 缩 FP16 的"测量用"改动**（4 处；生产归 034） | `a2/agents/KV8_prefill/ring16_measure_patch.py` | 同 |
| harness | `p12_scratch_bug.py`（页数碰撞网格）、`p13_alias_repro.py`（**scratch 别名最小复现**）、`p1_prefill.py`（torch 基线/行放大对照）、`p2_pipe.py`（44 层 step）、`p3_final.py`（逐面分解）、`p4_diag.py`（逐比特）、`p5_correct.py` / `p10_wired_check.py`（走真路径的逐比特 + 上界）、`p6_realpath.py`（走真路径 step）、`p7_swaonly.py` / `p9_paired.py`（SWA-only 与配对抗）、`p8_shape.py`（形状扫描 + 下限）、**`p11_decode_arms.py`（decode 40 层整图 + SWA-only 中间档 + 回归）** | 同 |
| 原始数据 | `a2/logs/raw/033-{p1-prefill,p2-pipe,p3-final,p4-diag,p5-correct,p6-realpath,p7-swaonly,p8-shape,p9-paired,p10-wired-correct}.json`、**`033-p11-decode.json`**、**`033-p12-scratch.json`**、**`033-p13-alias-128.json`/`033-p13-alias-512.json`**、`033-capacity-arms.txt`、`033-capacity-server.log` | `/work/agents/KV8_prefill/raw/`、`out/` |

### 8.2 复现命令

```bash
# 0) 影子包（一次性）
docker exec prbench-c1 bash -lc 'mkdir -p /work/agents/KV8_prefill && \
  cp -r /work/agents/KV8_gather/shadow /work/agents/KV8_prefill/shadow && \
  python3 /work/agents/KV8_prefill/kv8_prefill_wiring.py'      # 接 prefill kernel（惰性，靠 env 开）
docker exec prbench-c1 python3 /work/agents/KV8_prefill/ring16_measure_patch.py   # ring FP16（测量用）

# 1) prefill 性能（pipe + 设备饱和两口径）
ssh A3-node1 'cd ~/projects/dsv41-upstream-pr && bash tools/a3_chip.sh c1 --name kv8pf --timeout 900 -- \
  bash -c "cd /work/agents/KV8_prefill && export TMPDIR=/work/agents/KV8_prefill/tmp \
   KV8_RAW=/work/agents/KV8_prefill/raw PYTHONPATH=/work/agents/KV8_prefill/shadow && \
   python3 p9_paired.py && python3 p6_realpath.py"'

# 2) prefill 正确性（走真影子包路径）
ssh A3-node1 '... python3 p10_wired_check.py'

# 2b) decode 回归 + SWA-only 中间档（40 层整图）
ssh A3-node1 'cd ~/projects/dsv41-upstream-pr && bash tools/a3_chip.sh c1 --name kv8pf-dec --timeout 900 -- \
  bash -c "cd /work/agents/KV8_prefill && export TMPDIR=/work/agents/KV8_prefill/tmp \
   KV8_RAW=/work/agents/KV8_prefill/raw PYTHONPATH=/work/agents/KV8_prefill/shadow && \
   VLLM_V41_KV8_PREFILL=1 python3 p11_decode_arms.py"'

# 3) 容量五臂（每臂 ≈70 s；退出码 75 = 没抢到锁）
ssh A3-node1 'cd ~/projects/dsv41-upstream-pr && for v in "pf-bf16 bf16" "pf-kv8 kv8" \
  "pf-ring16 ring16" "pf-swa-ring16 swa_ring16" "pf-kv8-ring16 kv8_ring16"; do set -- $v; \
  bash tools/a3_chip.sh c1 --name cap-$1 --timeout 900 -- env TAG=$1 VARIANT=$2 \
    bash /work/agents/KV8_prefill/serve_arm.sh; done'
```

**纪律**：只用 c1（die 6）；**没用 `/tmp`**（容器 `/tmp` 收尾 20 KB）；没碰 `dsv41-a3` / `mooncake-master` /
Phy-ID 8–15；**没写 `upstream-v41/`**；没动 `dsv41-release/`；没改别人的影子包（只改 `agents/KV8_prefill/shadow`）；
跨机全走 coscli。**内存**：A3 宿主 `MemAvailable` 全程 ≥1.7 TiB，未触 150 GiB 阈值。
