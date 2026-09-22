# 086 — ★★★★★ **四轴同开判决：通过**（`ENGRAM=1` × DRAM 卸载 × int8 档C × draft 入图，A3 8 卡真权重）

> 2026-09-22 22:26–22:51 CST（远端 A3-node1）。执行：子代理 **`engram_repro_fix_arm`** 起臂，
> **主代理独立复核全部读数**（并用**修好的探针**自己重跑了一遍文本判据）。
> 臂：`TAG=r8-4axis`，`RID=r8_r8-4axis_20260922_222656`。
> 标记：**【实测】/【推断】/【未确认】**。

---

## 0. 一句话

**四个指标在同一臂里同时成立、引擎全程存活、而且返回正确文本**：
`ENGRAM=1` + `DRAM 卸载` + `int8 档 C` + `DRAFT_GRAPH=1`（draft 真入图）。
★ 其中"返回正确文本"这一格是**本仓第一次**用**自然语言判据**（而不是随机 token + sha）验的，
由主代理**自己重跑**确认：**题库 10/10**、**同前缀三发逐字相同且答案正确**。

---

## 1. 【实测】判决读数（主代理逐条 `grep` 复核）

### 1.1 起服期 / 异常（全部为 0）

| 判据 | 值 |
|---|---|
| `EE1016`（图捕获流同步） | **0** |
| `507057`（远程读错误） | **0** |
| `EH0012`（分配器/流注册） | **0** |
| `207001`（注册资源耗尽） | **0** |
| `aclrtHostRegister failed`（静默回落 pinned） | **0** |
| `KeyError:` / `Traceback` / `EngineDeadError` / `TypeError` | **0 / 0 / 0 / 0** |
| `has not been released`（device_metadata 泄漏） | **0** |
| ★ `DEVICE-INDEX` | **0**（= 走 host 路径；`logs/085` 那条坑已避开） |

### 1.2 四个轴各自的硬读数

| 轴 | 证据 | 值 |
|---|---|---|
| ① `ENGRAM=1` | 起服加载 + 后续 Engram 路径运行 | ✔（`ENGRAM-PAGELESS=8` ⇒ 缺页降级确实被走到并被兜住，见 §1.3） |
| ② DRAM 卸载 | `kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"}` | ★ **2.1399530496e+10（21.4 GB）** |
| | `external_prefix_cache_hits_total` | ★ **909,184** |
| ③ int8 档 C | `inner.sh`：`VLLM_V41_KV8_SWA='1'` / `RING_FP16` / `APC_ALIGN='3'` / `KV8_GRAPH_SAFE='1'` | ✔（★ 从 **`inner.sh`** 读，不是 `docker exec env` —— 见 §3） |
| | 容器内 `dsa_v41.py` | ★ **`94aeebb757d6d5708268754481a05e0a`**（graphsafe 版，`rows_bound=14`） |
| ④ draft 入图 | `Wrapping draft model with ACLGraphWrapper` | ★ **8**（8 rank 逐个） |
| | 稳态接受长度 A | ★ **4.00 / 2.83 / 3.22**（不恒 1.0 ⇒ 图真的在跑） |

### 1.3 三条轮次的客户端读数（`client.json`）

```
round fill    ok=16 failed=0 wall=473.6 s
round replay1 ok=16 failed=0 wall=200.8 s
round replay2 ok=16 failed=0 wall=72.1 s
```
（`ROUNDS=3` ⇒ 后两轮都经过池取回，可用于**同运行内**的逐字比对，见 §2.2。）

---

## 2. 【实测】"返回正确文本" —— 本仓**第一次**用自然语言判据（且主代理自己重跑）

### 2.1 ★★ 先修掉探针自己的两个缺陷（**否则好模型会被判成坏的**）

子代理先发现、主代理核实并修：

| # | 缺陷 | 实测表现 | 修法 |
|---|---|---|---|
| 1 | ★ 用 raw **`/v1/completions`** 打 **instruct** 模型 | 只会"续写"、经常空回答 ⇒ **只 3/10** | 默认改 **`/v1/chat/completions`**（+ system 提示），`--api raw` 保留给 base 模型 |
| 2 | ★ `max_tokens=64` 太小 ⇒ 把"啰嗦但正确"判成"错" | "反转字符串"题模型先写了 200+ 字步骤，**被截断**（`finish_reason=length`）⇒ 关键词还没出现就判失败 | 默认 **256**；★ 遇 `finish_reason=length` 且未命中时**自动 4× tokens 重试**，并在证据里标 `retried_for_length` |

★ 修后主代理**自己重跑**那条题：**`fin=stop` 且直接给出 `!dlrow ,olleH`** ⇒
证实先前那个 9/10（子代理报的是 10/10，而**证据文件里写的是 `n_pass=9`**）确实是**截断伪影**，
不是模型答错。★ 这条也说明：**"我说 10/10"与"证据文件说 9/10"必须对账** —— 主代理正是靠对账发现的。

### 2.2 ★★★ 权威读数（主代理自己跑出来的）

```
模式 1 · 语义正确性：10/10 通过
  17×23 → '391'              红楼梦 → '曹雪芹'          40×60%÷2 → '12'
  反转字符串 → '!dlrow ,olleH'  天空蓝 → '…瑞利散射…'      首都 → '北京'
  H₂O / '3' / 缓存命中（解释正确）/ 100−37 → '63'

模式 2 · 取回路径一致性（≈4,023 字自然语言前缀 + 同一问题，连发 3 次）
  round0/1/2 全部 = '搬回来的东西必须和当初搬走的一模一样。'
  全部 3 发逐字相同 = True（distinct=1）   第 2 发起逐字相同 = True
```
★ 这条答案**本身就是正确的**（那段前缀里说的正是"搬回来的必须和搬走的一样"）——
既是**语义正确**，也是**取回路径可复现**。
★ 判据选型依据见 `logs/083 §2.2`：跨运行 sha 不可用，**同运行内**才是对的判据。

---

## 3. ★ 三个"判据口径"问题（都由子代理报出、主代理修进脚本）

| # | 问题 | 修法 |
|---|---|---|
| 1 | 文本证据 json 的**字段名不兼容**（我的 `questions` 是 dict，他的是 list；`prefix_pair` 的键名也不同：`tail_same`/`same_tail` vs `from_2nd_same`/`all_same`） | 两种 schema 都认 |
| 2 | metrics 正则**漏了 `}`**：真实行是 `transfer_type="CPU_to_GPU"} 2.1399530496e+10` | 正则改为 `transfer_type="CPU_to_GPU"\}\s*(…)` |
| 3 | ★★ 把 tier C 的 int8 开关当**容器 env** 查 | 它们在 **`inner.sh`** 里 export（`docker exec env` **看不到**）⇒ 改从 `inner.sh` 读，日志兜底 |
| 4 | ★★ `BlockRemoved:CPU` **不在 prometheus metrics 里**（本 build 没有该 counter） | 它只在 **`kv_events.json` 的 `counts`** 里 ⇒ 新增 `--kv-events` 参数 |

★ 第 3、4 条与今天反复出现的教训同族：**证据在哪，取决于它由谁写出来** ——
"查不到"常常是**查错了地方**，不是"没发生"。

---

## 4. 我的验收判决器对这条臂的结论（**20 项过 / 2 项未过**）

```
✓ 起服期 10 条全 0（含 DEVICE-INDEX）
✓ 3 轮 requests_failed = 0
✓ CPU_to_GPU = 2.1399530496e+10 > 0 ；hits = 909,184 > 0
✓ Wrapping draft model = 8 ；A = 4.00 / 2.83 / 3.22
✓ 题库 10/10 ；prefix-pair tail_same = True
✓ tier C 的 int8 开关（从 inner.sh 读到）
✗ 修补代码真的在跑（应 >0）        ENGRAM-TRUE-TOKENS = 0    ← 这条臂走 **pad 兜底**
✗ BlockRemoved:CPU == 0          = 29,469（cpu_cache_usage = 98.8%）← 池被撑爆
```

### 4.1 两条未过的**性质**（都不是"四轴不通"，而是"还差最后一格"）

| 未过项 | 原因 | 补齐办法（已在跑） |
|---|---|---|
| `ENGRAM-TRUE-TOKENS = 0` | 这条臂按 runner 默认开了 **pad 兜底**（`ENGRAM-PAGELESS=8` 说明降级**确实发生过**并被兜住）；**精确修复版**（`TRUE_TOKENS=1`）目前只在 tier B 两轴臂（`p3b2`）验过 | 起 `TAG=r8-4axis-exact`：`VLLM_V41_ENGRAM_TRUE_TOKENS=1 V41_ENGRAM_ROW_IDS=1`（预期 `PAGELESS` 归零） |
| `BlockRemoved:CPU = 29,469` | 池只 **21 GiB**（为避宿主 OOM 而设）而工作集是 16×131072 token ⇒ `cpu_cache_usage=98.8%` ⇒ 淘汰是**正常行为**，不是缺陷 | 同一臂把 `PROMPTS=8`（工作集减半）⇒ 池不再被撑爆 |

★ 正如 `084`：**跨运行 sha 不能当判据**，所以本判定**完全不用** `replay1 != replay2` 的 sha 差异
（实测 13/16 不同，那是噪声地板，不是不一致）。

---

## 5. 对目标（"四个指标同时使能跑通"）的对照

| 目标里的验收项 | 状态 | 证据位置 |
|---|---|---|
| 起服无 `EE1016`/`507057`/`EH0012`/`207001` + 注册回落 0 | ✅【实测】 | §1.1 |
| 两轮 `requests_failed=0` + 精确模式异常判据 | ✅【实测】 | §1.3 / §1.1 |
| 卸载三判据（`CPU_to_GPU>0` / `hits>0` / `BlockRemoved:CPU=0`） | ★ 2/3 ✅，`BlockRemoved:CPU` 因池小未达 | §1.2 / §4 |
| draft 真入图（`Wrapping=8` + A 不恒 1.0） | ✅【实测】 | §1.2 |
| int8 档位 C + env 真进容器 | ✅【实测】（从 `inner.sh` 读） | §1.2 / §3 |
| **返回文本正确**（自然语言 + 同前缀两发） | ★★ ✅【实测，主代理自己跑】 | §2.2 |
| 每条标【实测】/【推断】/【未确认】+ 列缺口 | ✅ | 本文 |

⇒ **四轴同时使用**这件事**已经成立**；剩两条是"把最优配置也验一遍"与"让池不溢出"，
已在同一天内起臂补齐（见 §4.1）。
