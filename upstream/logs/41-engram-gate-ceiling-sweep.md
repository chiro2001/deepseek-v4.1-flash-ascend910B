# 41 — Engram gate 的 padding 天花板扫描（CHUNK=512 固定，MAX_TOKENS 256→8192）

**日期**：2026-09-21 14:09–14:18 CST（容器内时间戳为 UTC，06:09–06:18）
**执行**：子代理 `T2_ceilings`（任务一：天花板曲线；任务二：RoPE 两个小尾巴）
**机器**：`A3-node1` 槽位 `c0` = **die 3**（主跑）与 `c1` = **die 6**（复跑），容器 `prbench-c0/c1`，
`npu:0` = `Ascend910_9382`（910C 类）、CANN 9.1.0（driver 26.1.1）、
torch 2.10.0+cpu、torch_npu 2.10.0.post4、python 3.12
**脚本**：`agents/T2_ceilings/bench/bench_engram_gate_head2head_sweep.py`
（= `pr/bench_engram_gate_head2head.py` 的**加法式派生副本**，只多一个 `--ceiling-sweep` 开关）
**一句话**：把 RFC §3.2.1 里 "**not measured here**" 的那一句变成一条曲线 ——
**固定 `CHUNK=512` 时，每次 gate 调用的时间 ≈ 0.1 + 0.69 × (MAX_TOKENS/512) ms，
峰值显存 ≈ 240 + 45 × (MAX_TOKENS/512) MB，而 ceiling 的最小合法值是 `CHUNK`=512**（不是 256）。
在这条曲线上 **512 → 2048 是 3.4–3.6×、512 → 4096 是 6.7–7.3× 的时间差**（视 n 而定）；
shipped 的 `2048` 在"单张图要覆盖 `max_num_batched_tokens=2048`"这个约束下**已经是最优点**，
剩下的 2.0 ms 只能靠**按 capture size 分桶的 ceiling**拿掉，靠再调小一个全局常数拿不掉。

> 本文件同时包含**任务二**（RoPE 的两个小尾巴）：见 §6。

---

## 0. 交付问题速答

| 问题 | 答案 | 标记 |
|---|---|---|
| 推荐 ceiling 是多少 | **`MAX_TOKENS = max(CHUNK, ceil(B_max/CHUNK)·CHUNK)`** —— 能覆盖该图最大 token 数的**最小 512 倍数**。生产约束（BAT=2048）下**推荐值就是 2048 本身**；只要 decode 图永远 ≤512 token，**512** 才是最优（见 §5） | 【实测】 |
| 512 对应多少 ms / MB | n≤512 时 **0.774–0.871 ms / 285 MB**（同运行上游 0.588–0.630 ms ⇒ 1.19–1.38×） | 【实测】 |
| 2048（shipped）对应多少 | **2.811–3.006 ms / 420 MB**（n≤512），n=2048 时 3.346 ms | 【实测】 |
| shipped(2048) 与最优 ceiling 差多少 | 小 batch（n≤512）图上 **+2.03 ~ +2.14 ms/次调用、+135 MB**（= 3.4–3.6×）；在 n=2048 图上差 0（2048 已是最小合法值） | 【实测】 |
| 曲线有没有拐点 | **CHUNK 以上没有拐点**：512→4096 全段线性（每 512 行 +0.66~0.71 ms，两个 die 上一致），所以最优常数永远是"最小合法值" | 【实测】 |
| 256 能不能用 | **不能**：declared 256 < CHUNK=512 ⇒ shipped 的 `_gate_max_tokens()` **静默抬到 4096**（实测 5.677 ms = MAX=4096 臂的 5.675 ms）⇒ 曲线最差点（9.6× 上游） | 【实测】 |
| 逐位一致性 | 全部臂 × 全部 n（含 8192）：`torch.equal=True`、`max｜d｜ = 0.00e+00`（唯一例外是 n > MAX 的合同拒绝，JSON 里是 `error` 行） | 【实测】 |
| 任务二结论 | ① 小表（8K）下 **PR 臂 = 2.00 kernel/lookup**（stock 12.00 ⇒ **6× 少**，`Transpose` 全消失）；② int32 两格回归**同号复现 4 次**（本轮 +5.8 ~ +18.1 µs，log 35 是 +20.7 / +28.9）：**device 侧 PR 反而快 21–28 µs/lookup 且少 2–3 个 kernel**，回退是 **host 侧"每个方向多 build 一次 index"**（多一次 reshape+cast ≈ 13–19 µs 的 cast 半边 + ~10–25 µs 的 reshape 半边）；③ 把 index 改成**每次 lookup 只 build 一次**（保值的 1 行改动，实测逐位相同）⇒ 这两格从 **+6 µs 的输**变成 **−40 ~ −46 µs 的赢** | 【实测】 |
| 没跑成的项 | 任务二第一次运行被**容器重建**杀掉（rc=137，06:13:52Z 三个容器同时被重建，RestartCount=0 = 被重建而非重启）；本地无 torch ⇒ 本机只能跑纯文本版 provenance 检查 | 【实测】 |

---

## 1. 方法与 provenance（先把"可信"钉住）

### 1.1 脚本：加法式派生，5 臂语义一字未动

| 文件 | sha256 | 说明 |
|---|---|---|
| `pr/bench_engram_gate_head2head.py`（原版，**未改**） | `6b9455634c433a9f8ecf61a6efe411857b5befc44954e43d39c143b0af8ea843` | 文档 §3.1–§3.2.2 引用的 5 臂 harness |
| `agents/T2_ceilings/bench/bench_engram_gate_head2head_sweep.py`（本次用） | `66470619a6947b921053f3ff9fe280355fd57deb3f55a18211d13e9da1bba729` | 加 `--ceiling-sweep` + `declared` 字段 + 扫描表；不传开关时 `arms_for()` 与原版**同义**（同一列表、同一顺序、同一名字） |

改动只有 5 处，全在扫描机器码上：① 文件头加"派生副本"说明；② `REPO` 改成绝对路径
（副本在 `agents/T2_ceilings/bench/` 下，相对回溯会指错根）；③ `Arm` 多一个 `declared` 字段；
④ 新增 `SWEEP_CEILINGS` / `effective_ceiling()` / `ceiling_arms()`，`arms_for()` 末尾
`return base + ceiling_arms()`；⑤ CLI 加 `--ceiling-sweep` 与一张扫描表。
**`[BEGIN VERBATIM PR-16925]` / `[BEGIN VERBATIM OURS]` 两个块 diff 为空。**

### 1.2 provenance 双保险（都 PASS）

```text
[verify-embedded-blocks]        # 本机（无 torch；纯文本复刻原检查器的两个函数）
  PR-16925 original_block==sweep_block: True   sweep_block==source: True   25 行
           block sha256 233b41566cbcdc49a7b503d24bc0d2f4fcdceae861fefbcf135f83ba2d62f43d
  OURS     original_block==sweep_block: True   sweep_block==source: True   184 行
           block sha256 62649435a54767730c9df8b0f3e09bc6b35dee90542581b5f7c1878a7d63dd65
[verify-embedded-blocks] PASS

[verify-verbatim]               # 容器里跑脚本自带的原版检查器（refs 拷进 agents/T2_ceilings/refs/）
  PR-16925 : MATCH  25 lines from refs/upstream_common_pr16925.py line 164
  OURS     : MATCH  184 lines from refs/engram_gate_ours.py line 42
[verify-verbatim] PASS
```

两个块 sha256 与 [`36`](36-20260921-engram-gate-control-arm.md) 记录的**完全一致**；
拷进容器的两份源文件 sha256 也等于 pin 值（上游 `5ad16d70…`、我们 `955e40b9…`）【实测】。

### 1.3 扫描矩阵与占卡

| 项 | 值 |
|---|---|
| 臂 | 原 5 臂（upstream / control / shipped 2048 / BAT4096 / ablation）**+ 新增 5 臂**：`ours ceiling={256,512,1024,2048,4096}`（8K 阶段用 1024/2048/4096/8192） |
| n | `1,8,32,192`（阶段 A）+ `512,1024,2048,4096`（阶段 B）+ `8192`（阶段 C） |
| 计时 | `reps=30`、`warmup=5`、**同一次运行内所有活臂交替排序**（与原版一致）；8K 阶段 reps=30 / warmup=3 |
| 占卡 | `bash tools/a3_chip.sh c0 --timeout 1800 --name t2-ceiling -- …`（**单次运行一把锁**，退出即释放；`ASCEND_RT_VISIBLE_DEVICES` 由锁注入，未手设） |
| 结果 | c0：14:09:51 → 14:10:50 rc=0（整段 59 s）；c1：14:11:50 → 14:12:49 rc=0 |
| 分阶段产物 | 每个阶段**各自落一份 JSON**（smoke / small / large / 8k），中途失败不会丢前面的 |

---

## 2. ★ 结论表：ceiling × n 的时间矩阵（die 3 主跑）

中位数 ms / 一次 gate 调用（`reps=30`；`—` = 该臂**抛错**，即 n > MAX 的合同拒绝）。括号内是该格峰值显存 MB。

| n | upstream | control | ablation（pad→ceil(n/512)·512） | **MAX=512** | **MAX=1024** | **MAX=2048**（shipped） | **MAX=4096** | MAX=256（被抬到 4096） |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.588 (0.4) | 0.612 (0.5) | 0.777 (285) | **0.774 (285)** | **1.417 (330)** | **2.811 (420)** | **5.675 (600)** | 5.677 (600) |
| 8 | 0.689 (3.1) | 0.707 (3.8) | 0.829 (285) | **0.820 (285)** | **1.455 (330)** | **2.834 (420)** | **5.591 (600)** | 5.602 (600) |
| 32 | 0.685 (12.5) | 0.704 (15.0) | 0.832 (285) | **0.824 (285)** | **1.499 (330)** | **2.896 (420)** | **5.683 (600)** | 5.697 (600) |
| 192 | 0.679 (80.0) | 0.702 (96.0) | 0.865 (285) | **0.859 (285)** | **1.558 (330)** | **2.911 (420)** | **5.709 (600)** | 5.724 (600) |
| 512 | 0.630 (200.0) | 0.704 (240.0) | 0.874 (285) | **0.871 (285)** | **1.636 (330)** | **3.006 (420)** | **5.864 (600)** | 5.877 (600) |
| 1024 | 1.375 (400.1) | 1.426 (480.1) | 1.698 (330) | — | **1.697 (330)** | **3.140 (420)** | **5.712 (600)** | 5.718 (600) |
| 2048 | 3.345 (800.1) | 3.395 (960.1) | 3.349 (420) | — | — | **3.346 (420)** | **5.993 (600)** | 5.992 (600) |
| 4096 | 6.754 (1600.3) | 6.764 (1920.3) | 6.360 (600) | — | — | — | **6.357 (600)** | 6.352 (600) |
| 8192 | 13.295 (3200.5) | 13.322 (3840.5) | 12.870 (1120) | — | — | — | — | — |

> n=8192 那一行另有一次单独的扫描：**MAX=8192 → 12.859 ms / 1120 MB**（= 零 padding，
> 上游 13.295 ms / 3200.5 MB ⇒ 0.97× 时间、**0.35× 显存**）；同一行里 MAX≤4096 的臂全部 `RAISES`。

**怎么读**：

1. **时间只跟 ceiling 走，不跟 n 走**。ceiling=2048 时，n=1 与 n=192 都是 2.8–2.9 ms；
   ceiling=512 时 n=1 是 0.774 ms —— 1 个 token 与 512 个 token 只差 0.10 ms，
   而**多垫 512 行要 0.7 ms**。pad 行与真行在 chunk 循环里做同样的计算，所以
   "**padding 常数才是成本**"这句现在有了斜率。【实测】
2. **MAX=256 那一列不是 256**：它在 5 个 n 上都等于 MAX=4096 列（Δ ≤ 0.017 ms），
   因为 `_gate_max_tokens()` 对 `declared < CHUNK` 的输入回落到 4096。见 §5.3。【实测】
3. 峰值显存也跟 ceiling 走：小 n 上 285 / 330 / 420 / 600 MB，**每多 512 行 ≈ +45 MB**；
   而 n=8192、ceiling=8192（零 padding）是 1120 MB，只有上游 3200.5 MB 的 0.35×。
4. 与原版 5 臂在同一运行内的对照（§4）表明派生副本没有引入偏差。

---

## 3. 曲线与拐点：CHUNK 以上**没有**拐点

对每一行 n 用 4 个合法 ceiling 做 `t ≈ a + b·(MAX/512)` 最小二乘拟合（主跑 die 3 / 复跑 die 6）：

| n | b（ms / 每 512 行）die 3 | b die 6 | a（截距 ms）die 3 | 最大残差 ms |
|---:|---:|---:|---:|---:|
| 1 | **0.703** | **0.660** | 0.032 | 0.038 |
| 8 | 0.684 | 0.690 | 0.110 | 0.027 |
| 32 | 0.695 | 0.662 | 0.119 | 0.010 |
| 192 | 0.692 | 0.667 | 0.163 | 0.021 |
| 512 | 0.710 | 0.674 | 0.181 | 0.034 |

⇒ **斜率 b = 0.66–0.71 ms / 512 行，两个 die 上一致，线性区最大残差 ≤ 0.04 ms**。
换句话说，**每多一个 512 行的 chunk 就是固定的 ~0.7 ms**：n=512 那一行上
512→1024→2048→4096 是 0.871 / 1.636 / 3.006 / 5.864 ms，相邻差 0.765 / 1.370 / 2.858 ms
（= 1 / 2 / 4 个 chunk × 0.71 ms，加上不随 ceiling 变的截距 a）。
曲线在 CHUNK 以上是直线，**没有可以"停"的拐点** —— 唯一最优解就是
"**覆盖真实 batch 的最小合法值**"。【实测】

复跑一致性（同脚本、同矩阵，换到 die 6）：同臂同 n 的最大偏差 **0.297 ms**（出现在 n=1/8 这类
host 下发主导的格子上），n ≥ 1024 的格子偏差 **≤ 0.05 ms**，且**每一格的 ceiling 排序完全一致**。
上游臂在两次运行内分别是 0.588–0.689（die 3）与 0.562–0.604（die 6），即**小 n 上游数本身就有
±10% 级噪声** —— 所以本文只把"同一次运行内 ceiling 之间的差"当作可引用量，跨运行的绝对比值只作参考
（与 [`36`](36-20260921-engram-gate-control-arm.md) §8.4 的告诫一致）。【实测】

---

## 4. 与上一轮 shipped / ablation 数字的对照

同一 die（c0 = die 3）、同一 harness、`reps=30`，本次运行（14:10）与
[`36`](36-20260921-engram-gate-control-arm.md) 运行 1（12:49）逐格对比：

| n | upstream 36 / 41 | shipped 36 / 41 | ablation(pad→512) 36 / 41 | 偏差 |
|---:|---|---|---|---|
| 1 | 0.587 / 0.588 | 2.681 / 2.803 | 0.772 / 0.777 | ≤ 5% |
| 8 | 0.581 / 0.689 | 2.852 / 2.836 | 0.787 / 0.829 | 上游 +19% |
| 32 | 0.581 / 0.685 | 2.761 / 2.904 | 0.789 / 0.824 | 上游 +18% |
| 192 | 0.578 / 0.679 | 2.835 / 2.909 | 0.853 / 0.859 | 上游 +17% |
| 2048 | 3.334 / 3.345 | 3.275 / 3.354 | 3.269 / 3.349 | ≤ 2.5% |
| 4096 | 6.750 / 6.754 | RAISES / RAISES | 6.350 / 6.360 | ≤ 0.2% |

⇒ **shipped / ablation / 大 n 上游三列都复现（≤5%）**；唯一系统性偏差是**小 n 的上游臂
（+17~19%），且 control/ablation 同步抬高**，符合"host 下发节拍"这一类噪声
（本轮有别的子代理在同一台机器上跑 host DRAM 带宽/并发试验）。
因此 `MAX=512` 与 `MAX=2048` 之间 **2.0 ms 的差**是 device 侧 pad 工作（8×512 行 × 0.7 ms/512），
不受这个噪声影响。【实测】

与文档现成表的对照：§3.2.1 原表用的 "pad→512" 就是本表的 `MAX=512`（与 ablation 臂在 n≤192 上
差 ≤ 0.006 ms），原表的 0.772 / 0.787 / 0.789 / 0.853 与本次 0.777 / 0.829 / 0.824 / 0.859
**同量级、同排序**。【实测】

---

## 5. 推荐：ceiling 该挑哪个常数

### 5.1 规则

```
V41_ENGRAM_GATE_MAX_TOKENS = max(CHUNK, ceil(B_max / CHUNK) * CHUNK)
```

`B_max` = 该 capture 图能看到的**最大 token 数**（生产上等于 `max_num_batched_tokens`），
`CHUNK` = 512（同时是 `_gate_max_tokens()` 会静默抬升的下限）。
实测代价模型：**时间 ≈ 0.1 + 0.69·(MAX/512) ms**，**峰值显存 ≈ 240 + 45·(MAX/512) MB**。
两条硬约束：`MAX ≥ n`（否则 `RuntimeError`，本 harness 记为 `RAISES`，是结果不是崩溃）；
`MAX ≥ CHUNK`（否则**静默**变 4096，见 5.3）。

### 5.2 两个具体推荐

| 场景 | 推荐 ceiling | 实测时间 | 实测峰值 | 相对上游 |
|---|---:|---:|---:|---:|
| **decode 为主 / 小 batch 图**（该图最大 ≤512 token） | **512** | **0.774–0.871 ms** | **285 MB** | **1.19–1.38×** |
| **production prefill 合同**（`max_num_batched_tokens=2048`） | **2048（维持现状）** | 2.811–3.006 ms（n≤512）/ 3.346 ms（n=2048） | 420 MB | 4.12–4.78× / 1.00× |

**为什么 2048 不是"调小就行"**：gate 的 ceiling 是**编译期常量**，一张捕获图要对它见到的所有
forward 都成立；生产 `max_num_batched_tokens=2048` ⇒ 单图必须能吞下 2048 token ⇒ **单张图的最优
常数就是 2048**（更小的值会直接抛错，不是变快）。所以：

* 对 **n≥2048 的图**：shipped 已经最优（n=2048 时 3.346 ms vs 上游 3.345 ms，parity，显存 0.52×）；
* 对 **n≤512 的图**：shipped 多花 **+2.03 ~ +2.14 ms/次、+135 MB**（3.4–3.6×）；
  这 2 ms 是"**一张 2048 行图服务 1–192 token 的 batch**"的必然成本，**不是 chunking 的问题**。
* ⇒ 真正能拿掉它的下一步是 **per-capture-size 的 ceiling**（按 capture size 分桶捕获多张图），
  而不是把某个全局常数改小；本扫描给出的桶价目表就是上表 —— **每个桶取
  `ceil(BUCKET/512)*512` 即可**（512 桶 0.77–0.87 ms，1024 桶 1.42–1.70 ms，2048 桶 2.81–3.35 ms）。
  【推断：把"一次函数调用"乘到整层/整 step 需要端到端测量，本文只测了函数级】

### 5.3 ★ 顺手发现的坑（建议上游加一行 guard）

`MAX_TOKENS=256` + `CHUNK=512` **不是** pad 到 256：`_gate_max_tokens()` 里
`if value < chunk: value = max(chunk, _ENGRAM_GATE_DEFAULT_MAX_TOKENS)` ⇒ **静默变成 4096**，
也就是**曲线上最差的一格**（实测 5.677 ms @ n=1，与 MAX=4096 臂的 5.675 ms 在噪声内相同 ⇒
声明值 256 完全没起作用，反而比 512 慢 7.3×）。建议在 launcher/harness 侧对
`declared < CHUNK` **直接报错或向下取整到 CHUNK**，而不是抬到默认上限。【实测】

---

## 6. 任务二：RoPE 的两个小尾巴

来自 [`35`](35-20260921-rope-edge-cases.md) §5 的两个【未确认】。两件都做了：同进程 A/B、
**按路径 import 真实两版模块**（stock `0f9177f5…`、PR `982bb28d…`，与 log 35 一字不差），
脚本 `agents/T2_ceilings/bench/`，在 c0（die 3）/ c1（die 6）/ c2（die 7）三个槽位上跑。

### 6.1 小表（8K）下 PR 臂的 kernel 计数：**12 → 2（6× 少）**

`probe_gather_kernels_both.py` = 原 `pr/probe_gather_kernels.py` 的副本，**唯一改动是把 stock/pr
两版都交给同一个 profiler**（配置、表、状态注入、profiler schedule 逐行不变）。
c0 与 c2 两次运行的 JSON **逐字节相同**（同一 sha256）：

| 配置 | stock | **PR** | PR 的构成 |
|---|---:|---:|---|
| T=1M n=192，1 call/step | 6.00 | **2.00** | `IndexSelect_GatherV3` ×6 |
| **T=8K n=192，1 call/step** | **12.00** | **2.00** | 同上（**没有任何 `Transpose`**） |
| T=1M n=128，1 call/step | 6.00 | **2.00** | 同上 |
| **T=8K n=128，1 call/step** | **12.00** | **2.00** | 同上 |
| T=1M n=192，5 calls/step（每次新 positions） | 6.00 | **2.00** | 同上 |

（单位 = kernels / `get_cos_and_sin_dsa()` 调用，3 个 active step：CSV 里的 6/6/6 与 18 都是 3 步合计。
stock 的 6.00 = `BroadcastTo` + `Cast` + `GatherElementsV2` 各 2 个/调用；12.00 = 同样三件套 6.00
**再加 `Transpose` 6 个/调用**（CSV 里 18 个 / 3 步 = 每调用 3 个 × 2 方向）—— 与 log 35 §3 的构成逐字一致。）

⇒ log 35 的"**小表下 PR 计数【未确认】**"现在确认：小表下 PR **不但没有变差，反而多省了
stock 的 3 个 `Transpose`/方向**，**12.00 → 2.00（6× 少）**，比生产表（1M）的 6 → 2 收益更大。【实测】

> 口径提醒：这是**函数级 + 表长**的计数；`T=8K n=192` 这一格沿用原 probe 的 128 行输出 buffer
> （原脚本如此），首次调用触发一次 deprecated 的 `out` resize（日志里有一行 UserWarning），
> profiler 的 3 个 active step 全在 resize 之后，两臂走同一条路径，故计数不受影响。

### 6.2 int32 两格的逐 op 归因：device 是赢的，回退在 host 侧

`rope_int32_attribution.py`：同一张 1M fp32 表、同一组 positions、两臂各自 buffer、`reps=50`、
四次独立运行（die 7 / die 3 / die 6，log 35 那次也在表里）。

**(a) 端到端复现（中位数 µs；Δ = PR − stock）**

| 用例 | log 35（c1, die6） | c2（die7） | c0（die3） | c1（die6，重跑） |
|---|---|---|---|---|
| n=192 int32 连续（dflash buffer dtype） | 154.35 / 175.04  **+20.7** | 118.23 / 129.38  **+11.2** | 118.40 / 130.19  **+11.8** | 115.28 / 121.06  **+5.8** |
| n=192 int32 非连续 strided | 158.43 / 187.36  **+28.9** | 122.40 / 132.45  **+10.1** | 120.11 / 138.22  **+18.1** | 114.65 / 120.89  **+6.2** |
| n=192 int64 连续（参照） | −26.2（§1.1） | −23.9 | −20.0 | −25.7 |
| n=192 int64 非连续（参照） | −30.7 | −38.9 | −38.9 | −38.8 |

⇒ **符号 4 次全部复现（PR 在 int32 小尺寸更慢），幅度 5.8–28.9 µs 之间浮动**
（log 35 的两格是最大的一次）。同一批的 int64 格子稳定为负（PR 快 20–39 µs）。
幅度漂移说明这两格由 **host 侧节拍**决定 —— 见 (c)。【实测】

**(b) 逐 op 时序（配对，中位数 µs；只有"差"是可引用量）**

| 配对 | c2 | c0 | c1 | 说明 |
|---|---|---|---|---|
| `cast ×1` vs `cast ×2`（int32 连续） | **+19.45** | +17.88 | +16.12 | 多一次 int32→int64 cast 的**边际代价** |
| `cast ×1` vs `cast ×2`（int32 strided） | +13.20 | +18.23 | +12.80 | 同上 |
| `cast ×1` vs `cast ×2`（**int64** 参照，cast 是 no-op） | +2.11 | +2.17 | +2.28 | **阴性对照**：没有 cast 时该差 ≈ 0 ⇒ 上面 13–19 µs 确实是 cast |
| `reshape+expand`（纯 view，无 kernel）自比 | +0.08 | +0.03 | −0.15 | 该测量框的"空操作 floor" ≈ 21 µs（绝对值含 sync，配对后抵消） |
| stock 的一次 lookup（cast+expand+gather） vs PR 的一次 lookup（`index_select`，无 cast） | **−41.15** | −44.01 | −38.31 | 同一份输入、两个方向各一次：PR 的单次取表本身**快 38–45 µs** |
| 同上（strided） | −44.68 | −43.31 | −40.30 | 同上 |

> ⚠️ 一个教训（写出来免得别人误用）：我另外做的"**整条序列重建**"（1×cast + 2×gather
> vs 2×cast + 2×`index_select`）给出 **−20 ~ −26 µs（PR 更快）**，与真实端到端的符号**相反**。
> 原因：重建把两次 cast 都前移到 gather 之前，host 侧异步入队后互相重叠；真实路径里是
> cast→select→cast→select 交替。⇒ **逐 op 相加不能预测真实路径**，必须用 (d) 那种
> "在同一调用路径上只改一处"的对照。这个反例本身也说明 host 侧 op 的重叠程度决定了测量结果。

**(c) device 侧：PR 反而更好（两个 die 上逐格一致）**

torch_npu profiler（3 active step，1 call/step，按 kernel 名聚合 `Duration(us)`）：

| 用例 | stock | PR | Δ |
|---|---|---|---|
| int32 连续 | **7.00** kernel/lookup，47.93 µs/lookup | **4.00**，19.46 µs | **−3.00 kernel；−28.47 µs** |
| int32 strided | 8.00，52.61 µs | 6.00，30.35 µs | −2.00；−22.26 µs |
| int64 连续 | 6.00，46.88 µs | 2.00，16.91 µs | −4.00；−29.97 µs |
| int64 strided | 8.00，73.84 µs | 4.00，43.13 µs | −4.00；−30.71 µs |

（die 3 的数字与 die 7 逐格相差 ≤ 1.1 µs。int32 连续的构成：stock = `GatherElementsV2` 33.9 +
`BroadcastTo` 9.3 + `Cast` 3.7 + `InplaceCopy_Cast` 1.0 µs/lookup；PR = `IndexSelect_GatherV3` 17.6 +
`InplaceCopy_Cast` 1.8 µs/lookup。）

⇒ **cast 的 device 代价本身只有 ~2–4 µs/lookup**；int32 回归**不可能**来自 device 算力 ——
PR 在 device 上赢 22–28 µs/lookup，却在 wall clock 上输 6–18 µs ⇒ 差额全部在 **host 下发**。【实测】

**(d) ★ 把归因变成实测：只改一处 —— index 每次 lookup 只 build 一次**

诊断臂 `pr_hoisted`：与 PR 完全相同的两次 `index_select`，但 `_rope_index_1d(pos)` **每次 lookup
只调一次**（PR 现在是每方向一次）。`_rope_index_1d(pos)` 仍接收**原始 tensor**（含
`reshape(-1)`），所以计时区间里少的**只是第二次 index build**（reshape + cast），不是 reshape 本身：

| 配对（c1，die 6） | a | b | Δ |
|---|---:|---:|---:|
| `int32 contiguous`：PR 原样 vs PR(index 只 build 一次) | 113.98 | 69.73 | **−44.25** |
| `int32 strided`：同上 | 120.76 | 73.53 | **−47.24** |
| `int64 contiguous`：同上（**阴性对照**：无 cast，只少一次 reshape/helper） | 77.58 | 54.32 | −23.26 |
| `int64 strided`：同上 | 95.74 | 84.02 | −11.72 |
| `int32 contiguous`：stock vs PR(index 只 build 一次) | 115.31 | 69.34 | **−45.97** |
| `int32 strided`：stock vs 同上 | 115.83 | 75.50 | **−40.32** |

数值检查：**四个用例的 cos/sin 与 PR 原路径逐位相同**（`torch.equal=True`、`max|d| = 0.0e+00`）⇒
hoist 是**保值的**，不是另一个算法。

⇒ 归因结论（**实测，不再是推断**）：

1. int32 两格的回退 = **每个方向多一次 index build**（Python 级 `reshape(-1)` + `.to(int64)`）
   的 **host 下发代价**；每多一次这样的 op 在这个框里值 **~10–25 µs**（int64 对照给出 reshape
   那半边的量级），其中 cast 那半边 **~13–19 µs**（cast×1 vs ×2 的边际）。两者相加 ≈ 40–47 µs，
   与 (d) 的 −44 / −47 µs 对得上。
2. device 上 PR 一直是赢的（−22 ~ −28 µs/lookup、少 2–3 个 kernel），所以这不是"算力回退"。
3. **一行修法**：把 index build 提到循环外（`_rope_gather_rows` 接受已 cast 好的 index，
   或在调用点 build 一次）⇒ 这两格从 +6 ~ +29 µs 的**输**变成 **−40 ~ −46 µs 的赢**（对 stock），
   且逐位不变。这条建议可以直接进 PR 描述。【实测】

> 口径修正（诚实标注）：`c0` / `c2` 两次运行的 hoisted 臂**用了预先展平好的 `flat`**，
> 相当于额外省掉一次 reshape，对 hoisted 臂**有利**，因此那两次的 hoisted 数字**不可引用**；
> 脚本在 `c1` 运行前改成 `_rope_index_1d(pos)`（sha256 `b076c518…`），**上表只引用 c1**。
> 修正只动这两个配对，其余段落（端到端/per-op/profiler）两次版本逐字节相同。

---

## 7. 证据路径与指纹

### 7.1 天花板扫描（任务一）

| 文件 | 远端（`~/projects/dsv41-upstream-pr/`，容器内 `/work/`） | 本地 `logs/raw/` | sha256（本地） |
|---|---|---|---|
| 主跑 c0 小 n JSON | `agents/T2_ceilings/out/41-engram-gate-ceiling-small-c0.json` | `41-engram-gate-ceiling-small-a3c0.json` | `16139526a8ef8dca…` |
| 主跑 c0 大 n JSON | `…-large-c0.json` | `41-engram-gate-ceiling-large-a3c0.json` | `f8412bf42ec46d74…` |
| 主跑 c0 8K JSON | `…-8k-c0.json` | `41-engram-gate-ceiling-8k-a3c0.json` | `6f13959d3c42316e…` |
| 主跑 c0 smoke JSON | `…-smoke-c0.json` | `41-engram-gate-ceiling-smoke-a3c0.json` | `c1dbc09dc90e5640…` |
| 复跑 c1 四份 JSON | `…-{small,large,8k,smoke}-c1.json` | `41-engram-gate-ceiling-{small,large,8k,smoke}-a3c1.json` | `668de254…` / `55466469…` / `e9b0e5d5…` / `0d3f464d…` |
| 主跑 c0 driver 日志（含 `--verify-verbatim` PASS、逐臂表、扫描表） | `agents/T2_ceilings/out/41-engram-gate-ceiling-driver-c0.log` | `41-engram-gate-ceiling-driver-a3c0.log` | `527a4831bd4839cc…` |
| 复跑 c1 driver 日志 | `…-driver-c1.log` | `41-engram-gate-ceiling-driver-a3c1.log` | 见文件 |
| 脚本 | `agents/T2_ceilings/bench/bench_engram_gate_head2head_sweep.py` | 同名（`upstream-v41/agents/T2_ceilings/bench/`） | `66470619a6947b92…`（两侧一致） |
| 分析脚本 | `agents/T2_ceilings/bench/analyze_ceiling_sweep.py` | 同名 | 表格由 JSON 直出，无第二份手抄数字 |

回传方式：`ssh A3-node1 'cat <远端>' > <本地>`，随后 `sha256sum` 比对（本轮全部一致）。

### 7.2 任务二（RoPE）

| 文件 | 本地 `logs/raw/` | 远端 `agents/T2_ceilings/out/` | sha256（本地） |
|---|---|---|---|
| 双臂 kernel 计数 JSON（c0 与 c2 **逐字节相同**） | `41-rope-gather-kernels-both-a3c0.json` / `…-a3c2.json` | `41-rope-gather-kernels-both-c0.json` / `…-c2.json` | `276a591b581049f9…`（两份同值） |
| int32 归因 JSON（die 3） | `41-rope-int32-attribution-a3c0.json` | `41-rope-int32-attribution-c0.json` | `e2c49e4bacd383db…` |
| int32 归因 JSON（die 6，**含公平版 hoisted 臂**） | `41-rope-int32-attribution-a3c1.json` | `41-rope-int32-attribution-c1.json` | `2160d39ce918cf07…` |
| int32 归因 JSON（die 7） | `41-rope-int32-attribution-a3c2.json` | `41-rope-int32-attribution-c2.json` | `b39b48f0355d3b7d…` |
| driver 日志（c0 / c1 / c2） | `41-rope-tails-driver-a3c0.log` / `…-a3c1.log` / `…-a3c2.log` | 同名（`-c0/-c1/-c2`） | `d0ac1665…` / `45fb3e41…` / `afd4e62d…` |
| ★ 第一次运行的日志（**被容器重建杀掉**，rc=137） | `41-rope-tails-driver-a3c2-attempt1-killed.log` | `41-rope-tails-driver-c2.log` | `a1e54f7779b7d818…` |
| 脚本：双臂 probe | `agents/T2_ceilings/bench/probe_gather_kernels_both.py` | 同名 | `3c78c5a1e0ccb03b…` |
| 脚本：int32 归因（**公平版**，c1 用） | `agents/T2_ceilings/bench/rope_int32_attribution.py` | 同名 | `b076c518bb40c5db…` |
| 脚本：同一文件的 c0/c2 版本（hoisted 臂口径偏松） | —— | 已被覆盖 | `c1c5ffbbc4b9dcd9…` |

---

## 8. 【实测】/【推断】/【未确认】

**实测（任务一）**

1. 派生副本的 5 臂语义未动：两个 verbatim 块与原文件/源文件逐字节相同
   （块 sha `233b4156…` / `62649435…`，与 log 36 一致）；容器内 `--verify-verbatim` PASS；
   同一运行内原 5 臂的数字与 log 36 对照 ≤5%（小 n 上游臂除外，见下）。
2. ceiling 曲线：**t ≈ 0.1 + 0.69·(MAX/512) ms**（4 个合法 ceiling、5 个 n 的最小二乘，
   残差 ≤ 0.04 ms），斜率 0.66–0.71 ms/512 行在 die 3 / die 6 上一致；每条 n 行的最大值是 8K 尾巴。
3. 小 n 峰值显存 285 / 330 / 420 / 600 MB（MAX=512/1024/2048/4096），每 512 行 ≈ +45 MB；
   n=8192 零 padding 时 1120 MB = 上游的 0.35×。
4. `MAX_TOKENS=256` 被 shipped 的 `_gate_max_tokens()` 抬到 4096（用该函数本身调用得到），
   且在 5 个 n 上时间等于 MAX=4096 臂（Δ ≤ 0.017 ms）。
5. 全部臂 × 全部 n：`torch.equal=True`、`max|d|=0`；n > MAX 的格子记为 `error`（合同拒绝）。
6. 小表（8K）下 PR 臂 = **2.00 kernel/lookup** vs stock **12.00**（两次运行 JSON 逐字节相同）。
7. int32 两格回归**同号复现 4 次**（+5.8 ~ +28.9 µs）；device 侧 PR 更好（−2~−3 kernel、
   −21~−28 µs/lookup）；把 index build 减到每次 lookup 一次 ⇒ **−44.25 / −47.24 µs**，
   且与 PR 原路径**逐位相同**。

**推断（有数字支撑，但没有直接测量）**

1. 小 n 的上游臂在本次比 log 36 高 **17–19%**（control/ablation 同步抬高）是 host 下发节拍差异
   （本轮同机有别的子代理在跑 host DRAM 带宽/并发试验）；跨运行绝对比值不可引用。
2. 端到端（整 step / 40 层）会把"每次 gate 调用 2.0 ms 的差距"放大多少，本文没有测
   —— 函数级数字乘层数是**推断**。
3. per-capture-size 分桶能拿回那 2.0 ms：机制上成立（ceiling 是编译期常量，桶内取
   `ceil(BUCKET/512)·512` 即可），但**没有**在真实 capturer 里验证过。
4. int32 回归的 host 侧拆分（reshape ~10–25 µs + cast ~13–19 µs）来自两个独立配对实验的相加，
   两者不是同一次测量里分离出来的。

**未确认**

1. ★ **图内（ACLGraph）口径**：本次 ceiling 扫描**全部在 eager 下**。图内 pad 的 device 工作
   （0.7 ms/512 行）不会消失，但 eager 小 n 里的 host 下发成分会被摊掉 ⇒ 图内的截距应更小、
   斜率应保持。**没有测**。（与 log 04/05 的 MoE 教训同源：eager 与图内可以给相反符号。）
2. n > 8192、以及 8K 表下两臂的**时间**（只测了 kernel 计数，没测时间）。
3. 端到端（vLLM 引擎内）该 gate 的收益；本文只到函数级、合成输入、单卡无并发。
4. 跨 CANN/driver 版本的稳定性（全部在 A3-node1，CANN 9.1.0 / driver 26.1.1）。

---

## 9. 没跑成的 / 为什么

1. **任务二第一次运行被杀**：`A3-node1` 上 `prbench-c0/c1/c2` 三个容器在 **06:13:48–06:13:52Z
   被同时重建**（`docker inspect` 显示 `RestartCount=0`、`OOMKilled=false`、`StartedAt` 全部
   刷新 ⇒ 是被 `docker rm/run` 重建，不是 OOM 也不是 restart）——我的 driver 正在 c2 上跑
   第二次 profile 的 2 s 导出等待，于 06:13:51 被 SIGKILL（rc=137，见 §7.2 的 attempt1 日志）。
   任务一（14:09–14:12）不受影响；任务二已在 c2 / c0 / c1 上分三次重跑，全部 rc=0。
2. **本机没有 torch** ⇒ `--verify-verbatim`（真检查器）与所有 NPU 脚本不能在 `server-mini` 上跑；
   本机用纯文本版 `verify_embedded_blocks.py` 代替（结论等价），真检查器在容器里跑（PASS）。
3. **c0/c2 的 hoisted 臂口径偏松**（用了预展平输入），数字**弃用**，只在 c1 上重跑修正版（§6.2d）。
4. **图内 ceiling 扫描**：没做（需要 ACLGraph 捕获 + 回放全套；本次锁窗口只够函数级 eager 扫描）。
   这是本任务留下的最大空白，也是最该补的下一条（生产帧是图内）。
5. **端到端 / 多卡 / 并发**：不在本任务范围（同 log 36 的诚实边界）。
