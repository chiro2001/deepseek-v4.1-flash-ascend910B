# 044 — state ring 探针：探针装对了，而且抓到了 int8 那条路的真凶（因子甲 x 因子乙）

2026-09-22 07:50–08:20 CST。执行：子代理 **N_ring**。机器：**A3（A3-node1）槽位 c2 = die 7**（容器 `prbench-c2`）。
全程只用 c2，没碰 c0（`p_ringfix`）/ c1（`o_fp16nan`）/ 别人的容器；**没写 `upstream-v41/`**；
**没用 `/tmp`**（本地只写 `~/tmp/20260922/N_ring/`）；占卡全走 `tools/a3_chip.sh`（无 75 退出）；
**没手设 `ASCEND_RT_VISIBLE_DEVICES`**；传文件走 `cos-xfer.sh`；**没改任何一行生产代码**（全 overlay + import hook，env 门控）。

> **一句话**：`043` §4 的 `ring_calls=0` **是探针自己的 bug**（多 target 共用一个 `sys.meta_path` finder，
> 第一个 target 之后钩子被永久摘掉），**不是“那个函数没被调”**；修好之后，
> **ring 里“有没有 NaN” 与 “输出对不对” 在 16 个请求上逐字一一对应**
> ⇒ int8 这条路 ❌ 的机制是 **「池命中只刷 1/32 行」（因子甲，C0 也有）×「FP16 让残余变 NaN」（因子乙，判别量）**。

---

## 0. 结论（全文最重要的一段）

### 0.1 三臂判决表（同一探针、同一 harness，只改 ring dtype 与池大小）【实测】

| 臂 | 几何 | 池 | `GPU KV cache size` | replay `post_nan`（16 请求） | 下游 latent `nan` | J2（fill vs replay sha） |
|---|---|---|---|---|---|---|
| **`n-e1-C0-hot`** | SWA-q + **ring F32** | 144 MiB（命中） | **22,719** | max **266**（0.81%），**16/16 都有** | 全 **0** | **✅ 0/16**（`24b570535f58`） |
| **`n-e2-D-hot`** | SWA-q + **ring16** | 144 MiB（命中） | **33,295** | **2811–3060**（max **3060** = 9.3%），**14/16** | **79–93** | ❌ **14/16**（`6a47dd65f1ff`） |
| **`n-e3-D-cold`** | SWA-q + **ring16** | **1 MiB（不命中）** | 33,295 | 全 **0**（pre 残余 3062 ⇒ 本轮**洗回 0**） | 全 **0** | ✅ **0/16**（`24b570535f58`） |

三臂的 fill sha **全部** = `24b570535f58…`（与 `035`/`036`/`038`/`040`/`043` 逐字相同）⇒ 可比；
`n-e2-D-hot` 的 kv size **33,295 = ×1.4655**、`n-e1-C0-hot` **22,719**（与 `033`/`034`/`035` 逐格吻合）。

### 0.2 最强的一条判据：三个集合逐字相同（不是统计相关）

`n-e2-D-hot` 上，以下三个 prompt 下标集合**完全相等** = `[0,2,3,4,5,6,7,8,10,11,12,13,14,15]`：

| 集合 | 定义 | 内容 |
|---|---|---|
| **A** | replay 时 layer-2 ring 的 `post_nan > 0` | `[0,2,3,4,5,6,7,8,10,11,12,13,14,15]` |
| **B** | 该步写进 long-KV 的 latent `nan > 0` | `[0,2,3,4,5,6,7,8,10,11,12,13,14,15]` |
| **C** | 输出 token sha 与 fill 不符 | `[0,2,3,4,5,6,7,8,10,11,12,13,14,15]` |

**A == B == C（逐字）**。唯一两个干净的请求是 **prompt 1 与 prompt 9**（`pre_nan=0 / post_nan=0 / latent nan=0`），
而它们**恰好就是唯一两个不在 mismatch 列表里的**。
同一条对齐在 ✅ 臂上给的是 `A'=[]=B'=C'`（`n-e3-D-cold`）。

### 0.3 因子甲/因子乙的分开（回答“是不是 ring 没被重建”）

```
因子甲：池命中时 prefill 被跳过 ⇒ 本轮只写 1 行（32 行 ring 里）
        【实测】命中臂 replay 的 rows_changed = 1/32；冷算臂 = 32/32
        但 C0（✅）上也有因子甲 ⇒ **因子甲单独不是判别量**

因子乙：ring 从 F32 换成 FP16 后，残余从“有限的小值”变成“8.9% 的 NaN”
        【实测】C0(F32) max 266 (0.81%) → D(FP16) max 3060 (9.3%)，**差 11.5×**
        而 C0 的 0.81% 残余**不产生任何 NaN 下游**（latent nan 全 0）⇒ argmax 不翻
        ⇒ **因子乙是判别量**（这也正是 C0 ✅ / D ❌ 的唯一变量）
```

**⇒ 结论【实测】**：`D 几何 ❌` = 因子甲 x 因子乙；**修法应落在数值层**（ring16 写入侧的溢出/NaN 保护，
或命中路径把那几行 ring 重建掉），**不必动调度层**（“让 `state` 组参与卸载”解决的是因子甲，
而因子甲在 C0 ✅ 上同样存在 ⇒ 它**不是**判别量）。

### 0.4 【推断】机制链（每一步都有上表的实测支撑，最后一步是推的）

```
① 池命中 ⇒ 跳过 prefill ⇒ c2_ring_metadata 里 used=1（本轮只写 1 行）
② 但池化（compressor_from_projected）仍要读 ring 的 1–2 行 ⇒ 有一半读的是上一轮的残余
③ F32 残余：有限、量级小 ⇒ 池化输出仍然良态（latent nan=0）⇒ 输出不翻（C0 ✅）
④ FP16 残余：8.9% 的元素是非有限值 ⇒ 池化把它带进输出 latent（latent nan 79–93）
   ⇒ _write_compressed_source 把这段 latent 写进 long-KV
   ⇒ 后续 attention 读到 NaN ⇒ 该请求的输出 token 翻（D ❌）
⑤ fill 路径干净（三臂 fill post_nan 全 0）：完整计算把 32 行全部覆写 ⇒ 残余被冲掉
```

`o_fp16nan`（c1）正在把 ③④ 的“FP16 残余为什么是非有限值”从【推断】升到【实测】
（写入溢出 vs 后续 `0/0`；以及为什么 `034` 的仿真没暴露 —— 最可能是 `034` 测的是“写满 32 行”）。

---

## 1. `043` 的 `ring_calls=0` 是探针 bug（今晚第 5 个探针坑，主代理已记为 `§5b` 第 4 条）

### 1.1 现象与真因

`043` §4 说「`compressor_from_projected` 在这条 eager 路径上一次都没被调到」，依据是 5 个进程都打了
`RING=True` 但 `ring_calls=0`。**本任务实测推翻这条推断**：

* 那个探针用**一个** `sys.meta_path` finder 管**多个** target；它的 `_exec_module` 里
  `sys.meta_path.remove(self)` 是**永久摘除** ⇒ **第一个 target 被 import 之后，其余 target 的钩子全部静默失效**；
* 我在自己的 v2 上**原样复现**了这个坑（臂 `n-c1-C0`）：

  ```
  [N_ring] 2.026 ★ N_ring 钩子**已生效** sym=dsa_v41.DeepseekV41EagerAttentionImpl …
  [N_ring] hooks: {'dsa_forward': 'install_on_dsa.<locals>.forward', ...,
                  'compressor_pool': 'DeepseekV41Compressor.pool_projected'}   ← 未打补丁的原文
  [N_ring] SELFCHECK FAIL(类属性补丁没生效)
  ```

  ⇒ 同一个 finder 管的 `dsa_v41` 钩上了，`models/deepseek_v41/compressor` **没钩上**；
* 在 `043` 的两条臂里，先被 import 的是 `cpu_npu`（`L_DMA_PROBE=0` ⇒ `install_on_cpu_npu` 直接 `return`，
  **连横幅都不打**），但 `_exec_module` 照样把 finder 摘了 ⇒ 之后 `compressor_triton` 被 import 时**已经没有钩子**。

### 1.2 修法（v3 的四条）

| # | 修法 | 说明 |
|---|---|---|
| 1 | **一个 target 一个 finder 实例** | 每个实例只在自己的 target exec 后摘自己 ⇒ 互不影响（与 P2 / SWA_pergroup 的 `_post_import_hook` 同款） |
| 2 | **热路径兜底** | 在 `DeepseekV41EagerAttentionImpl.forward` 里**按实例真实类型**给 `DeepseekV41Compressor` 打类补丁（不依赖 import 顺序、不依赖调用方怎么写 import） |
| 3 | **043 那一格补钩** | `pool_projected` 一旦被调到，先把 `compressor_triton.compressor_from_projected` 包上 ⇒ `CFFP` 计数有判别力 |
| 4 | **函数地址对照** | 打补丁时打印 `orig(0x..) -> new(0x..)`，不只打“已装载”（§5b 第 2 条的可复制实现） |

### 1.3 修好之后的实测（C0 几何，eager，臂 `n-c2-C0`）

```
[N_ring] 20.623 EXEC_MODEL call=1 pid=6816 cls=NPUModelRunner
[N_ring] 21.267 ★ N_ring 钩子**已生效** sym=compressor_triton.compressor_from_projected …
[N_ring] 21.267 POOL  call=1 pid=6816 kv=(256, 512) max_query_len=256 blocks=[3]
[N_ring] 21.267 CFFP  call=1 pid=6816 kv=(256, 512) mql=256 ring=(1984, 32, 1024)
```

**硬门槛达成**：`ring_calls = pool = cffp = 36`（layer-2 每步一条记录），
`impl_forward` / `execute_model` 均有非 0 热路径计数 + 前 N 条 trace + 出口 `SUMMARY` + `sys.modules` 清单（`MOD` 行）。

---

## 2. 一条主动的自我克制（原话保留，防止后来者误用）

> **“pre vs post” 这个口径只能说明“ring 跨请求有残余”，不能说明“没被重建”。**

理由（写在这里免得再犯）：ring 是**跨请求复用的 scratch**，
所以“replay 请求 A 的 ring `pre` ≠ fill 请求 A 的 ring `post`”是**预期行为**，不是缺陷。
本任务第一版正是拿 `pre` 与 `post` 比，差点把它当“没被重建”的证据；
**主代理当场指出这是假阳性**（若 replay 时 ring 里还留着上一个请求的残余，`pre != post` 本来就成立）。

正确的口径（本任务最终采用、也是上表用的）：
**同一请求、同一步、跨臂对比**（池命中臂 vs 冷算臂的 `post`），以及
**“ring 里的 NaN” ↔ “下游 latent 的 NaN” ↔ “输出 token 翻不翻” 的三集合对齐**。

---

## 3. 诚实边界（哪些格没有判别力，逐条列出）

| 臂 | 记录数 | 探针生效性 | 能否当证据 |
|---|---|---|---|
| `n-c1-C0` | **0 条 `RINGREC`** | 横幅有、**类补丁未生效**（v2 的 finder bug） | ⛔ **不能**（已标为无效并重跑成 `n-c2-C0`） |
| `n-c2-C0` | **36 条** | 全部钩子 + `POOL/CFFP/EXEC_MODEL` 非 0 | ✅ C0 ✅ 守门员（`0/16`） |
| `n-d1-D-hot` | **0 条**（起服期崩：探针改返回值签名时的 `too many values to unpack`，`server.fail.log` 已留档） | 直接崩 | ⛔ **不能**；已重跑为 `n-d1b-D-hot` |
| `n-d1b-D-hot` | **36 条** | ✅ | ✅ D ❌（与 `035`/`036`/`038`/`040`/`043` 逐字复现） |
| `n-d2-D-cold` | **36 条** | ✅ | ✅ D ✅ 守门员 |
| `n-e1-C0-hot` | **36 条** | ✅（含 `rows_changed` / `lat` 新字段） | ✅ **判决点臂②** |
| `n-e2-D-hot` | **36 条** | ✅ | ✅ **判决点臂①** |
| `n-e3-D-cold` | **36 条** | ✅ | ✅ **判决点臂③（守门员）** |

`n-d1-D-hot` 与 `n-c1-C0` 的**记录数都是 0** ⇒ **没有判别力**，**不当证据用**（这是本任务自己踩的两个坑，
按 §5b 第 2 条如实留档）。`n-d1b-D-hot` / `n-d2-D-cold` 用的是探针的**前一版**（无 `rows_changed` / `lat` 字段），
所以 `rows_changed=1/32` 与 latent 的那两格**只在 `n-e1/e2/e3` 上有值**。

---

## 4. 判据对称性与阳性对照（§5b 第 3 条）

| 判据 | ✅ 臂（应干净） | ❌ 臂（应报警） | 结论 |
|---|---|---|---|
| ring `post_nan` | `n-e3-D-cold` 全 0 / `n-e1-C0-hot` ≤ **266** | `n-e2-D-hot` **2811–3060** | ✅ 有判别力 |
| 下游 latent `nan` | `n-e1-C0-hot` 全 0 / `n-e3-D-cold` 全 0 | `n-e2-D-hot` **79–93** | ✅ 有判别力 |
| 三集合对齐（A/B/C） | 两臂都 `[]==[]==[]` | `n-e2-D-hot` 14/16 逐字相同 | ✅ 最强 |
| `rows_changed` | 冷算臂 **32/32** | 命中臂 **1/32** | ✅ 有判别力（但**两臂都有因子甲**） |
| ring 有没有残余 | `n-e1-C0-hot` **16/16 都有**（≤266） | 同 | ⛔ **无判别力**（见 §2） |

**因子甲的“无判别力”本身就是本任务最有价值的负结果之一**：
它排除了“让 `state` 组参与卸载”作为**首选**修法（该修法解决的是两臂共有的那一半）。

---

## 5. 探针 v3 的判据口径（可复制）

每条 `RINGREC`（layer 2 = 第一个 `ratio=2` 压缩层，`038` 实测 NaN 诞生处）记录：

| 字段 | 含义 |
|---|---|
| `prefix` / `used` | `c2_ring_metadata[0]`（本步开始前的长度）/ `[1]`（本步写多少 token）★ 池命中时 `used=1` |
| `block` | `c2_ring_metadata[4]`（该请求的 ring 页号） |
| `pre_sha` / `post_sha` | 该 ring 页在 layer-2 前向**前/后**的 FP32 指纹（`sha1` 前 16 位） |
| `pre_nan` / `post_nan` | 该页里的非有限元素个数 |
| **`pre_rows_sha` / `post_rows_sha`** | **32 行各自的 sha**（8 位） |
| **`pre_rows_nan` / `post_rows_nan`** | **32 行各自的非有限元素个数** |
| **`rows_changed`** | **本 step 真正被改写的行数**（命中路径 1、冷算 32）★ 这条最锋利 |
| **`lat`** | `compressor_from_projected` 的**输出 latent**（写进 long-KV 的那段）的 `sha/nan/absmax` |
| `flag` | `NEVER_WRITTEN`（本进程从未写过该页）/ `STALE`（内容 ≠ 上次写下）/ `-` |

三条纪律的实现：`n_ring_probe.py` 的 `install()` 用**一个 target 一个 finder**；
`ADDR` 行打 `orig(0x..) -> new(0x..)`；
出口 `SUMMARY` 带 `impl_forward / execute_model / pool / cffp / rows_changed_hist / rows_nan(pre/post)` + `ENV` 行（口径可复现）。

---

## 6. 交付与复现

| 类 | 位置 |
|---|---|
| 本文 | `a2/logs/044-20260922-state-ring-probe.md` |
| 原始数据 | `a2/logs/raw/044-n-ring/`（**98 个文件 / 2.8 MB**：8 条臂的 `probe.log` / `ring.jsonl` / `client.json` / `kv_size.txt` / `metrics_*` / `server.log` + 起服失败日志） |
| 代码 | `a2/agents/N_ring/`：`probe/{n_ring_probe.py,sitecustomize.py}`、`scripts/{prepare_overlay.sh,run_arm_n.sh,run_batch_n.sh,selfcheck_n.py,analyze_ring.py,upload.sh}` |
| COS | `share/xfer/n_ring/{n_ring.tgz,044-raw.tgz}` |
| 容器内 | `/work/agents/N_ring/{probe,scripts,out}`（A3 A3-node1，overlay 在 `/work/agents/N_ring/pkg/`） |

### 6.1 复现（一条命令 = 一条臂）

```bash
# 判决点臂①（D + ring16 + 池命中，❌）
bash tools/a3_chip.sh c2 --timeout 900 --name n-e2 -- \
  env TAG=n-e2-D-hot POOL_BYTES=150994944 XSWA=1 XRING=1 XL1=1 XKV8PF=0 EAGER=1 PORT=8460 N_TRACE=600 \
  bash /work/agents/N_ring/scripts/run_arm_n.sh

# 判决点臂②（ring F32 + 池命中，✅）
bash tools/a3_chip.sh c2 --timeout 900 --name n-e1 -- \
  env TAG=n-e1-C0-hot POOL_BYTES=150994944 XSWA=1 XRING=0 XL1=1 XKV8PF=0 EAGER=1 PORT=8450 N_TRACE=600 \
  bash /work/agents/N_ring/scripts/run_arm_n.sh

# 判决点臂③（ring16 + 池 1 MiB，✅ 守门员）
bash tools/a3_chip.sh c2 --timeout 900 --name n-e3 -- \
  env TAG=n-e3-D-cold POOL_BYTES=1048576 XSWA=1 XRING=1 XL1=1 XKV8PF=0 EAGER=1 PORT=8470 N_TRACE=600 \
  bash /work/agents/N_ring/scripts/run_arm_n.sh

# 三臂一次跑完（≈3.5 min）
bash tools/a3_chip.sh c2 --timeout 2200 --name n-batch -- bash /work/agents/N_ring/scripts/run_batch_n.sh

# 分析（顺序 = 4 warmup + 16 fill + 16 replay；第 20..35 条就是 replay）
python3 /work/agents/N_ring/scripts/analyze_ring.py /work/agents/N_ring/out/n-n-e2-D-hot.ring.jsonl
```

**锁退出码 75 = 没抢到锁，是重试不是失败。** 每条臂自带“探针生效性自检”：
没有 `impl_forward>0` 的 `SUMMARY` 就显式写 **“本臂探针没在热路径上（结论无效）”**。

---

## 7. 红线核对

没发 PR / issue / 评论；**没写 `upstream-v41/`**；**没用 `/tmp`**（本地只写 `~/tmp/20260922/N_ring/`，
容器内只写 `/work/agents/N_ring/`）；占卡全走 `a3_chip.sh` **只用 c2**（没抢 c0/c1）；
**没手设 `ASCEND_RT_VISIBLE_DEVICES`**；没碰 `mooncake-*` / `jitpgo-*` / `dsv41-a3` / 别人容器；
起服前都查了 `df -h /dev/shm`（64 MiB，全程 0%）；A3 `MemAvailable` 全程 > 1.5 TiB；
传文件走 `cos-xfer.sh`（**没用 scp**）；**没改任何一行生产代码**（全部 overlay + import hook，env 门控）；
结论逐条标了【实测】/【推断】/【未确认】。

### 7.1 明确的“没做”

| # | 事项 | 状态 |
|---|---|---|
| 1 | FP16 ring 的 NaN **来源**（写入溢出 vs 后续 `0/0`）、以及“为什么 `034` 的仿真没暴露” | ⛔ **未做**（已交接 `o_fp16nan`，c1） |
| 2 | 两条修法（ring16 写入侧 clamp / 命中路径重建 ring）的实现与验证 | ⛔ **未做**（已交接 `p_ringfix`，c0） |
| 3 | F 几何（5 条杠杆，43,469）的 ring 判据 | ⛔ **未跑**（本任务的判决点只需 D 几何；机制上 F 的 state 组同样被排除 ⇒ 直接覆盖，但**口径未复测**） |
| 4 | 8 卡真权重上的同一探针 | ⛔ **未跑**（单卡 tiny 上机制已闭环） |
