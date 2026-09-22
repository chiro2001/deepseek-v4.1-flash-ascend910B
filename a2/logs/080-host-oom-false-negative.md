# 080 — ★★★ 一个新失败模式：**宿主 OOM 会伪装成"代码挂了"**（附判据门）

> 2026-09-22 21:2x–21:5x CST（远端 A3-node1）。执行：**主代理**（只读排查）+ 子代理 `engram_repro_fix_arm`（起臂）。
> 触发臂：`p3a-true0-rowids1`（`ENGRAM=1` + `ENGRAM_DEVICE_INDEX=0` + 卸载 56.5 GiB + `DRAFT_GRAPH=1`，档 B）。
> 标记：**【实测】/【推断】**。红线：只读诊断，未占卡、未改别人的容器。

---

## 0. 一句话

A 臂报出来的"**fill 9/16 → replay 0/16 → EngineDead**"看起来像又一个 Engram/卸载 缺陷，
实际是 **宿主内存被 OOM killer 打掉了一个 worker**：
`dmesg` 里 `cc1 invoked oom-killer` → `Killed process (VLLM::Worker_TP) anon-rss ≈56.4 GiB`。
★ **如果不看 `dmesg`，这一格会被误记成"`ROW_IDS=1` 让引擎崩了"** —— 那会让整条路线走错方向。

---

## 1. 【实测】证据链（按"最先发生"排序）

| # | 读数 | 出处 | 说明 |
|---|---|---|---|
| 1 | `cc1 invoked oom-killer: gfp_mask=0x140cca(...)` | `dmesg -T` 21:34:05 | ★ **编译器**（`cc1`）申请内存失败而触发 OOM —— 说明是"内存总量不够"，不是某个进程自己崩 |
| 2 | `oom-kill: constraint=CONSTRAINT_CPUSET, mems_allowed=0, global_oom, task_memcg=/docker/<容器>, task=VLLM::Worker_TP` | `dmesg -T` 21:34:12 | 被杀的是容器里的 vLLM worker |
| 3 | `Killed process 3917368 (VLLM::Worker_TP) total-vm:9737715264kB, **anon-rss:59118944kB (≈56.4 GiB)**` | `dmesg -T` | 单个 worker 的常驻 |
| 4 | `Worker proc VllmWorker-5 died unexpectedly (exit code: None), shutting down executor` | `serve.log:3029` 13:33:58 | ★ **无栈** ⇒ 被外部杀掉（若真是代码缺陷，这里会有 Traceback） |
| 5 | `[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!` ×8 | `serve.log:3035-3042` | ★ **是后果不是原因**：主进程没了，正在编译的 TBE 子进程跟着报 |
| 6 | `RuntimeError: cancelled` | `serve.log:3117` | 同上，引擎取消 |
| 7 | `KeyError:` = **0**、`ENGRAM-PAGELESS` = **0** | `grep -ac` | ★ 与我们刚修好的 Engram 缺页路径**完全无关** |
| 8 | `fill requests_ok=9 failed=7 / replay1 requests_ok=0 failed=16` | `client.json` | 被 OOM 打出来的一轮，**不是有效判据** |

### 1.1 为什么可以排除"是我们的补丁引入的"

| 候选 | 排除依据 |
|---|---|
| Engram 缺页（`075`/`077` 那条） | `KeyError:` = 0、`ENGRAM-PAGELESS` = 0；而修好的版本在这条路上会打 `[ENGRAM-PAGELESS]` |
| `ROW_IDS=1` 的发布代码 | 日志里**完全没有** `[ENGRAM-ROW-TOKENS]` 任何一行（成功/失败都没有）⇒ 那段代码**没被执行到**（另见 §3 的待查项） |
| int8 | 本臂 `R8_KV8=0 R8_KV8_SWA=0`（`tier=B`），int8 一个开关都没开 |
| 显存 | `EH0012`=0、`507057`=0、`EE1016`=0，起服期判据全过 |

⇒ **唯一与"worker 被外部杀死"自洽的解释 = 宿主 OOM**。

---

## 2. 【实测】这台机器上有邻居在抢内存（不是我们独占）

`docker ps`（21:5x）里除我们的臂之外还有：
`jitpgo-fus-meas / -p0b / -gen / -meas2 / -p0 / -safety`（6 个，起于 21:13–21:39）、
`dsv4-cpuoffload-dspark-cann91-20260922`、`cann91-runtime-holder`、`dspark-k2e2-learning`、`fw_dev` 等。
`free -g` 同一时刻：`total 2013 / used 359 / buff-cache 1606 / available 1653`。

★ 而我们的 8 卡臂的**量级**：8 × ≈56 GiB（worker）+ ≈206 GiB（Engram 表）+ ≈197 GiB（L1 后的池）≈ **850 GiB**。
⇒ **"够"与"不够"之间的余量只有几百 GiB，而邻居的 `jitpgo` 编译任务是突发性的** ⇒ 撞上就 OOM。

【推断】这是一次**环境级、时序相关**的失败，不是配置算错。
★ 但**不能用"环境问题"把它一笔带过**：目标配置（四轴同开）内存只会更紧，必须按 §4 处理。

---

## 3. ✅ 已答：`V41_ENGRAM_ROW_IDS` **确实到了容器**（我原来的怀疑是**判据口径**的问题）

我一开始怀疑它没送出去（因为 `inner.sh` 里 grep 不到）—— **这个怀疑是错的**，当场核实如下：

```
# 宿主的 shadow（挂载源）
$ grep -c set_engram_row_tokens $SHADOW/patches/files/model.py      → 1
$ md5sum $SHADOW/patches/files/model.py                            → 33c05aaf2ff3378f386cd9734a2d62ab
# 容器内
$ docker exec r8-p3b-true1-rowids1 sh -c 'grep -c set_engram_row_tokens .../models/deepseek_v41/model.py; md5sum .../model.py'
                                                                   → 1 / 33c05aaf…（逐字节相同）
# ★ worker 进程自己的 environ
$ docker exec <ctr> sh -c 'tr "\0" "\n" < /proc/<VllmWorker_pid>/environ | grep -E "ROW_IDS|TRUE_TOKENS"'
V41_ENGRAM_ROW_IDS=1
VLLM_V41_ENGRAM_TRUE_TOKENS=1
```

⇒ ★★ **判据口径要改**：这两个 env 是通过 **`docker run -e`** 传进容器的，**不是**由影子包的 `inner.sh` 平台 export
⇒ **`grep inner.sh` 查不到它们是正常现象，不能当"没送出去"**。
正确的查法是 **容器内读进程 environ**（上面那条），或 `docker exec <ctr> env | grep`。

★ 这条本身值得记：`065 §3` 的原话是"开关送不到 = 静默降级"，但**"怎么查它送没送到"也有两种口径**，
用错口径会得出**假的"没送到"** —— 那是同一族错误的镜像版本。

### 3.1 顺带确认：**沉默 = 成功**（不是"没跑"）

| 日志 | 何时打 | 含义 |
|---|---|---|
| `[ENGRAM-ROW-TOKENS]` | **只在发布失败时**打一次 ERROR | 没打 = 发布成功 |
| `[ENGRAM-TRUE-TOKENS] mode=… 首次修补：n=… 计数=…` | 只在 `repair_stats` **非空**（真发生了 absent/mismatch）时打一次 | 没打 = 本次没找到需要修补的槽位 |
⇒ 所以 `p3a`（`TRUE_TOKENS=0`，按设计不做修补）**本就应该一条都不打** —— 与实测一致。

<details><summary>（历史记录，保留我当时的错误怀疑）</summary>

`V41_ENGRAM_ROW_IDS=1` 我**没有**在容器的 `inner.sh` 里找到 export：
```
grep -an "ROW_IDS\|TRUE_TOKEN" <RID>/inner.sh    → 空
grep -aE "ENGRAM|ROW-IDS|TRUE" <arm>.serve_a2.log → 只有 [A2-ENGRAM-ROW-IDS] 与一处「生效值 …」文案
```
★ 而 `VLLM_V41_ENGRAM_TRUE_TOKENS` 是**确实进了容器**的 —— `serve.log:26` 有
`Unknown vLLM environment variable detected: VLLM_V41_ENGRAM_TRUE_TOKENS`（vLLM 会为未知的 `VLLM_*` 打这条）。
**`V41_ENGRAM_ROW_IDS` 没有 `VLLM_` 前缀，所以不会出现在那张名单里 ⇒ 无法用同一条判据判断。**

⇒ **【未确认】**它到底是没送出去，还是送出去了但那段代码因为别的原因没跑。
⇒ 判据（下一步）：容器内读 worker 的环境
```
docker exec <ctr> sh -c 'for p in $(pgrep -f VllmWorker); do tr "\0" "\n" < /proc/$p/environ | grep -E "ROW_IDS|TRUE_TOKENS"; done'
```
★ 若它**没送出去** ⇒ `p3a` 这一格等于"patch 挂了但代码没跑"⇒ A/B 必然测不出差异（又一次"开关送不到"，`065 §3` 同族）。

</details>

---

## 4. 处置（写进纪律）

### 4.1 ★★ 新增判据门：先看 dmesg，再看 serve.log

任何"worker died unexpectedly / exit code: None / 引擎突然死"的臂，**必须**先跑：
```bash
sudo -n dmesg -T | grep -iE "oom|Killed process" | tail -5
```
出现 `Killed process ... VLLM::Worker` ⇒ **该臂作废（环境失败），不是判据失败**，不要记进对照表。

### 4.2 降内存压力（重跑 A/B 的配置）

| 旋钮 | 从 | 到 | 省 |
|---|---|---|---|
| `OFFLOAD_BYTES` | 60,666,413,056（56.5 GiB） | **23,068,672,000（≈21.5 GiB）** | ≈130 GiB（L1 后 ≈91 GiB） |
| 起服前检查 | — | `free -g` 的 available **≥ 1400 GiB** 才起；邻居忙就先等 | — |
| 并发臂数 | 可能同时跑 2 个 8 卡臂 | **一次只跑一个** | — |
| `DROPCACHE` | 1（默认） | 保持 1（起服前清 page cache，实测能释放数百 GiB） | — |

★ 21.5 GiB 池对"触发取回"**绰绰有余**（`069` 的 p1b 用 56.5 GiB 拿到 `hits=901,120`；`p2c` 用 1 MiB 就完全不命中）
⇒ 只要落在两者之间，取回一定发生，而内存压力大降。

### 4.3 ★ 目标配置（四轴同开）的内存预算

```
8 × worker(≈56 GiB)         ≈ 451 GiB
Engram 表（分片，节点级）      ≈ 206 GiB
卸载池（L1 后，21.5 GiB 池）   ≈  75 GiB
--------------------------------------------
合计                        ≈ 732 GiB  ⇒ 在 1653 GiB available 下留 ≈920 GiB 给邻居
```
★ 若邻居突发占用超过这个余量，**换时间跑**，而不是把判据改小。

---

## 5. 现状

```
p3a-true0-rowids1 : ⛔ 作废（宿主 OOM，不是代码结论）—— 判据待重取
p3b-true1-rowids1 : 起服中（子代理已按上面的处置继续）
KV 逐字节保真探针  : c1 上另一条战线（编号 078，本任务不改它）
下一个空号        : 081
```
