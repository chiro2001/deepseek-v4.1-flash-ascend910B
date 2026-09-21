# 36 — Engram gate head-to-head：control arm 实证 + A3 重跑 + 文档回填

**日期**：2026-09-21 12:49–13:03 CST
**执行**：子代理 `E_engram`
**一句话**：§3.1 里"control arm 不在脚本里"是**文档陈旧，不是缺实验** —— `arms_for()` 一直是 **5 臂**，
第 2 臂就是 control；本任务在 A3（die 3）**两次独立重跑**并落盘，§3.1/§3.2/§3.2.1/§3.2.2 已按新数据回填。
**新旧两次不是全部同向**：显存完全同向、小 n 同向，**n ≥ 2048 的时间符号翻转** ⇒ 只能报 parity。

---

## 0. 交付问题速答

| 问题 | 答案 | 标记 |
|---|---|---|
| control arm 叫什么、在哪 | `ours default (CHUNK unset -> reference)`，`arms_for(n)` 的**第 2 臂（共 5 臂）** | 【实测】 |
| control arm 实测数字 | 时间 **1.000–1.059× 上游**（6 个尺寸 × 三次运行的全部 18 格；最差一格 = 02:17 跑的 n=8，A3 两次都在 1.05× 内）；峰值显存 **1.200× 上游**（每个尺寸、每次运行都成立，多出的正好是一份 FP32 `[n,4,5120]`）；`torch.equal = True`、`max｜d｜ = 0` | 【实测】 |
| `--verify-verbatim` | **PASS**（上游块 25 行 ≡ `refs/remotes/pr/16925:…/common.py` line 164；我们的块 184 行 ≡ `dsv41-release/patches/files/engram_gate.py` line 42） | 【实测】 |
| 新旧两次是否同向 | **显存：完全同向**（3 次运行比值一字不差）；**小 n ≤ 192：同向**（都是我们慢：新 4.4–5.0× vs 旧 5.5–7.2×）；**n ≥ 2048 的时间：符号翻转**（旧 1.12×/1.15× 慢 → 新 0.98–1.00× / 0.94–0.96×） | 【实测】 |
| 没跑成的项 | §3.3 两项（不在本任务范围）；n = 8192 超出本 harness 范围；单卡机已交回（[`25`](25-20260921-single-card-handback.md)），旧机数字无法原地复现 | 【未确认】 |

---

## 1. 脚本事实（读码）

| 项 | 事实 |
|---|---|
| 文件 | `pr/bench_engram_gate_head2head.py`（715 行，sha256 `6b9455634c433a9f8ecf61a6efe411857b5befc44954e43d39c143b0af8ea843`；远端 `~/projects/dsv41-upstream-pr/bench/` 同名文件**同 sha256**）【实测】 |
| `arms_for(n)` | **5 臂**：①`upstream PR#16925 (verbatim)` ②**`ours default (CHUNK unset -> reference)`（= control）** ③`ours shipped (CHUNK=512 MAX=2048)` ④`ours BAT4096 (CHUNK=512 MAX=4096)` ⑤`ours ablation (CHUNK=512 MAX=ceil(n/512)·512)`（非交付配置）【实测】 |
| control 臂为什么是 control | 它跑的是**我们文件里的 stock 参考实现**（`_engram_gate_reference`），只把 `V41_ENGRAM_GATE_CHUNK` 置空 ⇒ 与上游**同一算法、不同文件**，用来把"换文件"与"换 gate"分开 |
| `--verify-verbatim` 参数 | `--upstream-src`（缺省 = `git -C ../upstream-v41/vllm-ascend-upstream show refs/remotes/pr/16925:vllm_ascend/models/deepseek_v41/engram/common.py`）、`--ours-src`（缺省 = `~/projects/dsv41/dsv41-release/patches/files/engram_gate.py`）；**两个缺省都可用，本次未传覆盖参数**；该模式在 `import torch` **之前**返回 ⇒ 不需要 NPU |
| 测量 CLI | `--sizes 1,8,32,192,2048,4096 --reps 30 --warmup 5 --json <path>`（另有 `--seed`，缺省 0） |
| 合同拒绝的记录方式 | n = 4096 时 shipped 臂抛 `RuntimeError`，harness 在结果表打 **`RAISES`**、在 JSON 里写 `{"ceiling": 2048, "error": "RuntimeError: …"}`，其余臂继续 ⇒ **是结果，不是崩溃**（本次三次运行的 JSON 都含该 error 行）【实测】 |

---

## 2. `--verify-verbatim`（本机，无 NPU）

命令与原始输出（`logs/raw/36-engram-gate-h2-verify-verbatim.txt`，sha256 `977c422bac2c57271511280e223d0015111c92149d7d754ede0043c08e9a8216`）：

```text
$ python3 pr/bench_engram_gate_head2head.py --verify-verbatim   # exit 0
[verify-verbatim] checking the embedded copies against their sources
  upstream ref     : refs/remotes/pr/16925 = 382dc9289d6a7dec37203c3a886f3c35bd7c0966
  PR-16925 : MATCH  25 lines from refs/remotes/pr/16925:vllm_ascend/models/deepseek_v41/engram/common.py (git -C ~/projects/dsv41/upstream-v41/vllm-ascend-upstream) line 164
             block sha256 233b41566cbcdc49a7b503d24bc0d2f4fcdceae861fefbcf135f83ba2d62f43d
  OURS     : MATCH  184 lines from ~/projects/dsv41/dsv41-release/patches/files/engram_gate.py line 42
             block sha256 62649435a54767730c9df8b0f3e09bc6b35dee90542581b5f7c1878a7d63dd65
[verify-verbatim] PASS
```

两个块的 sha256 与 [log 03](03-20260921-engram-gate-h2.md) 里 02:17 那次记录的**完全一致** ⇒ 自那次以来脚本嵌入的两份源码一字未改。【实测】

---

## 3. 本次 A3 运行

| 项 | 值 |
|---|---|
| 机器 | **A3-21**，槽位 `c0` = **die 3**（host `/dev/davinci3` = NPU 1 chip 1），容器 `prbench-c0` |
| 芯片 / 软件 | `Ascend910_9382`（910C 类）、CANN **9.1.0**（`V100R001C11SPC001B243`）、HDK/driver **26.1.1**、torch `2.10.0+cpu`、torch_npu `2.10.0.post4`、python 3.12.13 |
| 占卡 | `bash tools/a3_chip.sh c0 --timeout 900 --name engram-h2-e[2] -- …`（**一次运行一把锁**，退出即释放；`ASCEND_RT_VISIBLE_DEVICES=0` 由锁注入，**未手设**） |
| 命令 | `python3 /work/bench/bench_engram_gate_head2head.py --sizes 1,8,32,192,2048,4096 --reps 30 --warmup 5 --json /work/agents/E_engram/out/36-engram-gate-h2-a3-<tag>.json` |
| 后台三件套 | `nohup … > <log> 2>&1 </dev/null &`（两次都按此起，ssh 均正常返回） |
| 运行 1 | 12:49:55 → 12:50:11，rc = 0 |
| 运行 2（同 die 复跑） | 12:51:45 → 12:52:02，rc = 0 |
| 环境噪声 | 两次新运行各打 1 条 `[W…] NPUCachingAllocator.cpp:202 … require processing for 32 padding size` 告警（A3 driver 26.1.1 有、旧单卡 driver 25.5.5 无）；不影响两次新运行之间的可比性 |

---

## 4. 原始结果（运行 1，30 次中位数；括号内为该单元格的峰值显存 MB）

| n | 上游 | **control**（我们文件的 stock 路径） | shipped `512/2048` | BAT4096 `512/4096` | ablation `512/ceil` | `torch.equal` |
|---:|---|---|---|---|---|---|
| 1 | 0.587 (0.4) | **0.609 (0.5)** | 2.681 (420.0) | 5.470 (600.0) | 0.772 (285.0) | ✅ 全部 `max｜d｜=0` |
| 8 | 0.581 (3.1) | **0.606 (3.8)** | 2.852 (420.0) | 5.635 (600.0) | 0.787 (285.0) | ✅ |
| 32 | 0.581 (12.5) | **0.606 (15.0)** | 2.761 (420.0) | 5.526 (600.0) | 0.789 (285.0) | ✅ |
| 192 | 0.578 (80.0) | **0.608 (96.0)** | 2.835 (420.0) | 5.548 (600.0) | 0.853 (285.0) | ✅ |
| 2048 | 3.334 (800.1) | **3.368 (960.1)** | 3.275 (420.0) | 5.947 (600.0) | 3.269 (420.0) | ✅ |
| 4096 | 6.750 (1600.3) | **6.771 (1920.3)** | **RAISES**（合同，MAX=2048<n） | 6.356 (600.0) | 6.350 (600.0) | ✅（除 RAISES 格） |

### 4.1 control arm 的逐尺寸数字（三次运行）

| n | 上游 median ms（03 / 新1 / 新2） | control median ms（03 / 新1 / 新2） | 时间比 control÷上游（03 / 新1 / 新2） | 上游峰值 MB | control 峰值 MB | 峰值比 | `torch.equal` |
|---:|---|---|---|---:|---:|---:|---|
| 1 | 0.489 / 0.587 / 0.578 | 0.517 / 0.609 / 0.604 | 1.06 / 1.04 / 1.04 | 0.40 | 0.47 | **1.20×** | ✅ |
| 8 | 0.548 / 0.581 / 0.578 | 0.581 / 0.606 / 0.607 | 1.06 / 1.04 / 1.05 | 3.13 | 3.75 | **1.20×** | ✅ |
| 32 | 0.657 / 0.581 / 0.581 | 0.687 / 0.606 / 0.608 | 1.05 / 1.04 / 1.05 | 12.51 | 15.01 | **1.20×** | ✅ |
| 192 | 0.653 / 0.578 / 0.655 | 0.682 / 0.608 / 0.684 | 1.04 / 1.05 / 1.04 | 80.01 | 96.01 | **1.20×** | ✅ |
| 2048 | 3.782 / 3.334 / 3.333 | 3.780 / 3.368 / 3.375 | 1.00 / 1.01 / 1.01 | 800.13 | 960.13 | **1.20×** | ✅ |
| 4096 | 7.221 / 6.750 / 6.744 | 7.304 / 6.771 / 6.763 | 1.01 / 1.00 / 1.00 | 1600.25 | 1920.25 | **1.20×** | ✅ |

**怎么读**：

* **时间上换文件是免费的**：1.000–1.059×（A3 两次 1.003–1.051×，02:17 那次最差 1.059×；小 n +4–6%，大 n +0–1%）⇒
  其他臂测的确实是"chunking gate"，不是"我们的文件"。【实测】
  小 n 那点固定开销**推断**是 `our_arm` 包装层读 env + 多一层 Python 派发（与 [log 03](03-20260921-engram-gate-h2.md) 的同一说法），未单独做空函数对照。【推断】
* **显存上换文件不是免费的**：control 恒定 **1.200×** 上游，增量 = 0.08 / 0.63 / 2.50 / 16.0 / 160.0 / 320.0 MB，
  与 `n × hc_mult(4) × hidden(5120) × 4 B`（按分配器取整）逐点吻合。【实测】
* 机制（**推断**）：我们文件的参考路径在 `HOIST=0`（交付缺省）下把 `hidden.float()` 写了两次并各自赋给局部变量
  （`hidden_restore` / `hidden_out`），于是**多一份 FP32 激活在峰值时仍然存活**；上游虽然也 `.float()` 两次，
  但两次不重叠。这正是 `[ENGRAM-GATE-HOIST]` A 要消掉的那份拷贝。【推断，未用 profiler 直接验证】

---

## 5. 与旧数据（`03-engram-gate-h2-20260921.json`，单卡 `Ascend910_9362`）逐格对比

### 5.1 显存：**完全同向**（比值三次一字不差）

| n | shipped ÷ 上游 HBM | ablation pad 峰值 MB | BAT4096 峰值 MB | control ÷ 上游 |
|---:|---|---:|---:|---|
| 1 / 8 / 32 / 192 | 1063× / 134× / 33.6× / 5.25× | 285 / 285 / 285 / 285 | 600（各尺寸恒定） | 1.20×（各尺寸恒定） |
| 2048 | **0.52×** | 420 | 600 | 1.20× |
| 4096 | shipped `RAISES` | 600 | 600 | 1.20× |

三次运行**完全相同**（分配量决定的，不受时序噪声影响）。这也是本优化唯一的收益口径。【实测】

### 5.2 时间：小 n 同向，**n ≥ 2048 符号翻转**

| n | 02:17 单卡（`Ascend910_9362`） | 新-1 A3 die 3 | 新-2 A3 die 3 | 判断 |
|---:|---:|---:|---:|---|
| 1 | 7.20× | 4.57× | 4.72× | 同向（我们慢） |
| 8 | 6.51× | 4.91× | 4.96× | 同向 |
| 32 | 5.47× | 4.75× | 4.81× | 同向 |
| 192 | 5.56× | 4.90× | 4.41× | 同向 |
| 2048 | 1.12× | **0.98×** | **1.00×** | **符号翻转（±15% 带内）** |
| 4096（BAT4096） | 1.15× | 0.94× | 0.96× | **符号翻转（±15% 带内）** |

ablation 臂同样如此：旧 1.99/1.80/1.53/1.61/1.12/1.15 × → 新 1.32/1.35/1.36/1.48/0.98/0.94 ×（run1）、1.30/1.38/1.38/1.30/1.00/0.96 ×（run2）。

**结论（不挑好看的那次）**：n ≥ 2048 的时间差**不携带稳定符号**（旧机器慢，A3 上 parity；都在 ±15% 内），
因此文档按 **parity** 表述，既不声称加速也不声称变慢；显存收益（0.52× / 0.375×）三次运行全部复现，仍是主结论。【实测】

### 5.3 逐位一致性

三次运行、每个尺寸、每个能跑完的臂：`torch.equal = True`、`max｜d｜ = 0.00e+00`（每轮 30 格中 29 格，第 30 格是 n=4096 的合同拒绝）。【实测】

---

## 6. 文档回填（`pr/RFC-16375-CONTRIBUTION.md` 的 §3）

| 位置 | 改动 |
|---|---|
| §3 开头 STATUS | 改为"A3 die 3 主表 + 同 die 复跑 + 旧单卡对照"，写明机器/CANN/driver/torch_npu/python/die 号与占锁方式，并声明"**包含两次运行不一致的行**" |
| §3.1 表 | 新增 `Arms per run`（5 臂逐一列出）、`Machine (main run)` 两行；`Script` 行补 sha256；`Interleaving` 从"both arms"改为"all live arms"；`Raw output` 行换成真实 JSON 路径 + sha256；`Provenance check` 行补 PASS 的实测结果 |
| §3.1 `Control arm` 行 | **TODO 删除**，改为"已在脚本中且已测"+ 实测数字（时间 1.00–1.05×、峰值 1.20× 及逐尺寸增量、`torch.equal`），并注明旧 TODO 系**陈旧**（02:17 的 JSON 里就有该臂） |
| §3.2 | 主表换成 A3 die 3 的数字（上游 0.587/0.581/0.581/0.578/3.334/6.750 ms；shipped 2.681/2.852/2.761/2.835/3.275 ms、n=4096 合同拒绝）；新增"三次运行时间比"对照表（含 `same sign?` 列）与机器/JSON/sha256 脚注；结论句改为 **parity** 表述 |
| §3.2.1 | 倍数从"5–7×"改为"4.4–5.0×（旧跑 5.5–7.2×，同向）"；ablation 表换成新数字并加比值列与旧数字列；补一句"ablation 自身仍 pad 到 512 行、其 1.3–1.5× 仍是 pad 成本，更贴近真实 batch 的上限**本次未测**" |
| §3.2.2 | n=4096 改为新数字（BAT4096 6.356 ms/600 MB vs 上游 6.750 ms/1600.3 MB = 0.94×/0.375×），并写明旧跑是 1.15×、**显存复现、时间不复现**；补"`RAISES` 是结果不是崩溃"；**Fill command 修正**为本次真实命令（A3 `a3_chip.sh c0` + 真实 sizes），并保留单卡机写法 |

未改：`pr/` 目录下其他任何文件；文档里 §0.4/§0.5/§1/§2/§4–§8 的既有内容与"诚实边界"段落（函数级 / eager / 合成输入 / 单卡无并发 / W4A8 vs W8A8）**原样保留**。

---

## 7. 证据路径与指纹

| 文件 | 位置 | sha256 |
|---|---|---|
| A3 运行 1 JSON | 本地 `logs/raw/36-engram-gate-h2-a3-a3c0-20260921.json`；远端 `~/projects/dsv41-upstream-pr/agents/E_engram/out/…`（**两边同 sha256**） | `0228423accb42c7a5e5ee17a2ea0fd33af15bb97b547544d4a42296fc83e1598` |
| A3 运行 1 原始日志 | 本地 `logs/raw/36-engram-gate-h2-a3-a3c0-20260921.log`；远端同目录 | `178af41dd9537ce0c701a62eef2b8999875aaf7879c96a5b62e7368df05d966d` |
| A3 运行 2 JSON | 本地 `logs/raw/36-engram-gate-h2-a3-a3c0-20260921-run2.json`；远端同目录 | `7cf894105e2cae15aaa3365474ebec8eebdf0a18cc2b76f943de8b78e22467da` |
| A3 运行 2 原始日志 | 本地 `logs/raw/36-engram-gate-h2-a3-a3c0-20260921-run2.log` | `c339ad4deca8dff883b7966afedc54059fabec98635fd2d6e97785e8ada44fce` |
| `--verify-verbatim` 输出 | 本地 `logs/raw/36-engram-gate-h2-verify-verbatim.txt` | `977c422bac2c57271511280e223d0015111c92149d7d754ede0043c08e9a8216` |
| 旧单卡 JSON（对照） | 本地 `logs/raw/03-engram-gate-h2-20260921.json` | `2e5ce488ca737c7e3f73eb1b1566925236da2daa8360688bf3d4a2600341e6a0` |
| harness | 本地 `pr/bench_engram_gate_head2head.py` = 远端 `bench/` 同名文件 | `6b9455634c433a9f8ecf61a6efe411857b5befc44954e43d39c143b0af8ea843` |

回传方式：`ssh A3-node1 'cat <远端文件>' > <本地文件>`，随后 `sha256sum` 两边比对（全部一致）。
远端只写了 `~/projects/dsv41-upstream-pr/agents/E_engram/`；只用了 c0 槽位，未触碰 `dsv41-a3` 服务容器与别人的槽位。

---

## 8. 【实测】/【推断】/【未确认】

**实测**

1. `arms_for()` 有 5 臂，第 2 臂 `ours default (CHUNK unset -> reference)` 即 control；本任务三次运行（旧单卡 1 次 + A3 2 次）**都含该臂**。
2. `--verify-verbatim` **PASS**（两个块逐行等于源文件，块 sha256 见 §2）。
3. control 臂：时间 1.00–1.05× 上游；峰值 1.200× 上游（增量与 `n×4×5120×4 B` 按取整吻合）；全尺寸 `torch.equal=True`。
4. A3 两次运行的全部臂/尺寸：`torch.equal=True`、`max｜d｜=0`；n=4096 的 shipped 臂在三次运行中**都**被记为 `RAISES`/`error`。
5. 显存比值三次完全一致（1063×/134×/33.6×/5.25×/0.52×；BAT4096 600 MB；ablation 285/420/600 MB）。
6. n ≥ 2048 的时间比符号在机器之间翻转（1.12×/1.15× ↔ 0.98×/1.00×、0.94×/0.96×）。

**推断（有数字支撑但未直接验证机制）**

1. control 臂 1.20× 峰值的**机制**是"我们参考路径里第二次 `hidden.float()` 常驻内存"（`HOIST=0`）——增量大小与之一致，但未用 profiler 或显存快照直接证明。
2. 小 n 的倍数差异（4.4–5.0× vs 5.5–7.2×）来自两台机器的 host 下发速度不同（上游小 n 时间 0.489–0.657 ms vs 0.578–0.587 ms），不是算法差异。

**未确认**

1. **n = 8192（"8K"）**：文档 §1 的 [90] 行与 §5 的 0006 行引用"**−1.56 ms @ 8K**"，那个数**不来自本 harness**（本 harness 上限 4096）——它出自"same-session device A/B"（图内口径）。本任务**没有**测 8K，这两处的数字与 §3 的关系本次**未确认**。
2. 旧单卡机已按 [`25`](25-20260921-single-card-handback.md) 交回，其 02:17 数字**无法在原地复跑**；两台机器不是同一台，因此 §3 只比**比值/符号**，不比值绝对数。
3. eager 口径之外（ACLGraph 图内）该 gate 的时间/显存关系本次**未测**；§0.5 的规则（设备激活类收益对框架不敏感）是机制推理，文档自己也标注为 reasoning。
4. A3 上两次运行的**运行间**噪声只有 2 个样本：n ≥ 2048 的比值差 ≤ 0.02×，但小 n 行可达 0.49×（n=192：4.90× vs 4.41×）——
   因为该行是 host 下发主导：**上游臂自身**的中位数两次差 13%（0.578 → 0.655 ms），而我们那臂只动 1.8%（2.835 → 2.886 ms）。
   小 n 的倍数因此**不应被当作稳定值引用**；本次没有做更多重复来给它误差棒。

---

## 9. 没做的 / 为什么

| 项 | 状态 | 原因 |
|---|---|---|
| §3.3 的两项对比（host table registration、token history update） | 未跑 | 不在本任务范围（任务只要求 control arm + §3.2 回填） |
| n = 8192 / 16384 | 未跑 | 本 harness 的 sweep 上限是 4096（生产 BAT 合同上限）；要测需改脚本，越权 |
| 单卡机原地复现 | 不可能 | 机器已交回（log 25）；本任务的"第二次独立运行"用 A3 同 die 复跑代替 |
