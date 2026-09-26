# 046 — state ring 的正确性缺陷：**修法已定位到确切的一行，但它不在我们这一侧**（5 条臂）

> 2026-09-22 08:2x–09:0x CST。执行：子代理 **P_ringfix**。机器：**A3（A3-node1）槽位 c0 = die 3**
> （容器 `prbench-c0`）。全程只用 c0；没碰 `dsv41-a3` / `mooncake-*` / `jitpgo-*` / Phy-ID 8–15；
> **没手设 `ASCEND_RT_VISIBLE_DEVICES`**（一律走 `tools/a3_chip.sh` 锁）；**没用 `/tmp`**；
> **没写 `upstream-v41/`**；没动 `dsv41-release/`；跨机传输全走 coscli；代码只写 `a2/agents/P_ringfix/`。
> 产物：本日志 + `logs/raw/046-p-ringfix/`（371 KB）+ `agents/P_ringfix/`。

---

## 0. 结论（先给主代理，**含一条负结果**）

1. **⛔【实测·负结果】任务书两条修法候选（写侧 clamp / 命中路径重建 ring）与 `045` 建议的
   "ring 页清零"，都**不能让 D/F 的 J2 转 ✅** —— 因为那一行不是"垃圾字节"，而是【真正缺失的数据】。**

   | 步型 | `start_pos` | `used` | 池化要不要读 ring |
   |---|---:|---:|---|
   | fill 轮 | 0 | 4096 | **不读**（`residual=0`；本段 4096 个 token 的投影都在手） |
   | 命中 replay | **4095** | **1** | **必须读** token 4094 那一行（组 2047 = tokens 4094,4095 跨在命中边界上） |

   fill 轮里 group 2047 的**两个** token 都在段内（用真投影）；replay 轮里只有 4095 在段内。
   而 replay 是**新请求**、ring 页是回收来的（`prefix_cacheable=False`、**不参与卸载**，见 `009`/`043`）
   ⇒ **token 4094 的 `[kv|score]` 原始投影已经不存在**。
   把它当 0（清页 / 置 0 / gating）只会得到**"干净但错误"**的 latent ⇒ 与冷算不一致 ⇒ **J2 依然是 ❌**。
   ⇒ **清页治不了丢数据。**

2. **★★★【实测】真正的修法**：让 **APC 命中长度按模型自己的压缩比对齐**（`num_computed_tokens % ratio == 0`），
   这样 ratio-2 层的第一步**两个 token 都在段内**，池化走与 fill 轮**同一条
   `SINGLE_BLOCK/NO_PAD` 分支** ⇒ **逐比特同 latent**（构造性成立，不是"运气"）。
   它同时修掉 `045` 指出的"**ring F32 只是侥幸正确**"那条潜伏缺陷的**根因**。

3. **⛔【实测·负结果】但这个对齐在【卸载路径】上落不下去**：本任务定位到**四处**设置命中边界的代码，
   其中**真正 binding 的两处在卸载连接器里**，而连接器的 chunk 记账**只接受 `n-1` 那一种边界**：
   把它改成 `n-2` 会让 `manager.prepare_load(...)` 直接炸
   （`AssertionError: Block b'…\x00\x00\x00\x02' not found in cache`，发动机死、15/16 请求失败）。
   ⇒ **卸载路径上的命中边界归 `OffloadingConnectorScheduler` 所有**，改它要动 chunk 记账（不是一行）。

4. **✅ 交付**：三条实现（**两条 env 门控的修法 + 一条防线**，默认全关）+ **一条**在卸载路径之外
   **harness 无关、ratio=1 逐字 no-op** 的一行修法 + 完整的**判决证据链**（5 条臂 + 9 格设备单元自检）。

5. **✅ 守门员**：`p-a7-C0-min`（C0 几何 + 本补丁）**✅ 0/16**、容量 **22,719 不变**、
   `pre_nan` 266/245/228/219 与 `044` 的 C0 基线**逐字相同** ⇒ **本补丁零回归**。

---

## 1. ★★★ 机制（一条实测链，把 `044`/`045` 的结论一次性对齐）

```
① 命中 replay 的 start_pos = 4095（奇数）        ← kv_cache_manager.py:259  max_cache_hit_length = num_tokens - 1
② ratio-2 层的池化组按【全局 token 位置】划分：group 2047 = tokens(4094,4095)
   ⇒ 组跨在命中边界上 ⇒ token 4094 必须从 state ring 的残余行读  ← _pooled_blocked 的 seg_off<0 分支
③ ring 组 prefix_cacheable=False、不参与卸载（009/043）⇒ 新请求拿到的是【回收页】
   ⇒ 那一行是别家平面的字节（045 的三条指纹）
④ FP16 视下 6.7% 是 NaN ⇒ NaN 进池化 ⇒ 被 _write_compressed_source 写进 long-KV ⇒ 翻 token
   ★ 但即使【不是 NaN】（F32 视 / 清成 0），那一行的【数值也是错的】⇒ 同样翻 token
⑤ 冷算臂 ✅ 的机制：32 个 fill 步、`residual=0`，**一次残余读都没有** ⇒ 与 ring dtype 无关
```

**⇒ 三句话**：`start_pos` 奇数 ⇒ 组跨边界 ⇒ 缺一行**已丢失的**投影。
**修法必须让那一步不再跨边界**（或让那一行真的存在），**不是**把缺失的行"洗白"。

### 1.1 为什么"分配后清零"没有可钩的点（主代理已核实接受）

【实测·代码事实】`core/deepseek_v41.py`：

| 行 | 代码 | 含义 |
|---|---|---|
| `:204-268` | `plan_cache_slots()` | `aliases = [state[slot]] + swa[slot::4]`（`:94`）；每个 alias 建成 `CachePlacement(name, **0**, capacity)`（`:113`） |
| `:333` | `allocate_cache_config()` | 把这些 placement `shared_by` 成**同一块 tensor** |

⇒ ring 的页**不是分配出来的**，而是从一块共享大 tensor 上**切出来的视图**；块号由 vLLM 块池发放，
**没有任何"这个块给了 state 组"的回调** ⇒ **没有"分配后"这一刻可挂**。
⇒ 这是本任务把修法从"分配侧"挪到"读侧 / 命中边界"的原因。

### 1.2 为什么"命中路径强制重建那 32 行"**不可行**（不是代价大，是信息不可逆）

重建需要那 32 个 token 的 `wkv/wgate` 投影，而 decode 步**只有当前 token** 的 `hidden_states`
（`_write_compressed_source` 只拿得到本段）；ring 存的是池化**之前**的原始投影，
long-KV / SWA 存的是池化**之后**的 latent（再过 RMSNorm + RoPE）⇒ **无法反推**。

---

## 2. ★ 四条"设置命中边界"的代码位置（逐条带证据）

| # | 文件 : 行 | 代码 | 本臂实测它是不是 binding | 处置 |
|---|---|---|---|---|
| **①** | `vllm/v1/core/kv_cache_manager.py:259` | `max_cache_hit_length = request.num_tokens - 1` | ❌ 不是（改了它 `pre` 仍是 4095） | ✅ **已实现**（保留） |
| **②** | `vllm/v1/core/sched/scheduler.py:2674` | `if num_computed_tokens == num_tokens: num_computed_tokens = num_tokens - 1` | ❌ 不是（本例 `4095 != 4096`，该分支不触发） | ✅ 已实现（保留） |
| **③** | `vllm/v1/simple_kv_offload/manager.py:261` | `max_hit_len = request.num_tokens - 1 - num_computed_tokens` | ❌ 不是（另一条卸载实现，本臂没走） | ⛔ **未启用** |
| **④** | `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:864-873` | `max_hit_size_tokens = req.num_tokens; if self._sliding_window_groups: max_hit_size_tokens -= 1` | ✅ **是它**（改了它 `pre` 变 4094） | ⛔ **未启用**（见 §3.2 的炸点） |

★ **④ 对本模型必然触发**：V4.1 有 **10 个 SWA 组** ⇒ `self._sliding_window_groups` 非空 ⇒ **必然 `-1`**。
★ 该模块自己就有 `_mamba_align_size` + `round_down(max_hit_size_tokens, …)` 的先例（同形态），
  所以"按 ratio 对齐"在语法上是它认得的写法。

---

## 3. 五条臂（全部单卡 tiny、EAGER、16×4096 token、池 144 MiB / 1 MiB）

| 臂 | 几何 | 开关 | 探针读数（replay） | fill / replay sha | J2 | 容量 |
|---|---|---|---|---|---|---|
| `p-a1-D-align` | D | `APC=1`（仅 ①） | `pre=4095 / used=1`，`lat_nan` 93 | `24b5705…` / **`6a47dd6…`** | ❌ 14/16 | 33,295 |
| `p-a2-D-align2` | D | `APC=1`（①+②） | 同上 | 同上（**逐字相同**） | ❌ 14/16 | 33,295 |
| `p-a3-D-align3` | D | `APC=1`（①+②+③） | 同上 | 同上（**逐字相同**） | ❌ 14/16 | 33,295 |
| `p-a5-D-align5` | D | `APC=1`（①+②+③+**④**） | —（replay 崩） | — | ⛔ **发动机死** | — |
| **`p-a6-D-min`** | D | `APC=1`（**最小面：①+②**） | `pre=4095 / used=1`，`lat_nan` 93 | `24b5705…` / `6a47dd6…` | ❌ 14/16 | 33,295 |
| **`p-a7-C0-min`** | **C0** | `APC=1`（①+②） | `pre=4095 / used=1`，**`lat_nan` 全 0** | **同值**（`24b5705…`） | **✅ 0/16** | **22,719 不变** |

**三条关键读数**：
1. ★ **①②③ 单独或组合，replay sha 与 `044` 的基线逐字相同（`6a47dd65f1ff`）** ⇒ **三个都不是 binding 的那一处**。
   （`p-a6-D-min` 用**最小补丁面**再确认一次 ⇒ 这条负结果不是"补丁太大被互相抵消"。）
2. ★ **④ 是 binding 的**：它一开，`pre` 从 4095 **变成 4094**（`apc-trace` 逐条打出 `4095 -> 4094`）—— **这是直接观测**。
3. ★ **`p-a7-C0-min` 证明本补丁零回归**（判据 ③ 守门员），且 C0 的 `pre_nan` 266/245/228/219 与 `044` 逐字相同。

### 3.1 ★ 为什么 C0（ring F32）不需要对齐也不同翻 token（实测解释）

C0 与 D 的**步型完全相同**（`(4095,1)` × 16）、也**同样读回收页**（`pre_nz = 16640 = 66560/4`），
但 C0 的 `lat_nan = 0`、`|lat|` 最大 3e-5 ⇒ **F32 视下那批字节是"有限的小数"**，
池化输出虽然**错**但量级与"正确值 ± 一个小扰动"同阶 ⇒ 不翻 token（**侥幸**）。
⇒ 这正是 `045` 说的"**F32 只是侥幸正确**"，也是本任务把"命中边界对齐"当成**根治**而不是 dtype 修法的原因。

### 3.2 ★★ ④ 的炸点（实测原文，供后人接力）

```
(EngineCore) AssertionError: Block b'!\xf2\x81o\xd6\xe4\xcc]#9\xec\xdd\x97\xd6<\x9c\xbd#\x16\x87\x14\xe3U\x11\xe2M\xccX8\xd5\x161\x00\x00\x00\x02' not found in cache
  File "…/v1/kv_offload/cpu/manager.py", line 138, in prepare_load
  ← 上游调用链：scheduler.py:1001 update_state_after_alloc → pgp_scheduler.py:1243 prepare_load
```

**机制【推断，但与代码逐行对齐】**：连接器用 `req_status.update_num_hit_chunks(num_computed + num_hit)`
记录"哪些 chunk 已经命中"；`-1` 那一种边界（`n-1`）是它**唯一**能表达的"少一个 token"形态。
换成 `n-2` 后，它推出一个自己认为"已命中、但 store 侧从未写过"的 chunk key ⇒ `prepare_load` 断言。
⇒ **接力点**：`OffloadingConnectorScheduler.update_state_after_alloc` 里
`num_chunks = cdiv(num_cached_tokens, tokens_per_chunk)` 这一段（`pgp_scheduler.py:1200` 附近）
需要与"少两个 token"的边界同时改，**或者**改由"边界减一"改成"边界对齐后仍按 `-1` 记账"。

### 3.3 ★ 一条被本任务自己的 finder bug 挡掉的臂（`p-a4`，已修，供后人避坑）

`p-a4` 起服期炸在 `get_sliding_window_size_in_chunks` 的 `assert isinstance(kv_cache_spec, FullAttentionSpec)`。
**根因不是补丁，是我的 finder**：我用 `importlib.machinery.PathFinder.find_spec()` **直接找文件**，
**绕过了 PGP 的整文件重定向**（PGP 把 `…kv_connector.v1.offloading.scheduler` 整个换成
`patch_pgp/pgp_scheduler.py`，那份才有 `D2_offload` 的 `AttentionSpec` 修复）⇒ 我加载了**上游原版**。
**修法**：改用 `importlib.util.find_spec()`（让其余 meta_path finder 照常生效 + `try/finally` 把自己插回去）
—— 与 `L_dmafix`/PGP 的写法一致。**这是 `AGENTS.md` §5b 第 4 条之后新增的第 5 个坑**（建议记进 AGENTS）。

---

## 4. 交付物：三个开关（**全默认关**，全部 env 门控）

| 开关 | 默认 | 机制 | 落点 | 状态 |
|---|---:|---|---|---|
| `VLLM_V41_APC_ALIGN=1` | **0** | ① 命中长度 cap 按 `lcm(组 compress_ratio)` 对齐<br>② `scheduler._update_waiting_for_remote_kv` 的"重算最后一个 token"同一约定 | `vllm/v1/core/kv_cache_manager.py:259`<br>`vllm/v1/core/sched/scheduler.py:2674` | ✅ 实现 + 单测 + 实测**零回归**；⛔ **在卸载路径上不解决 J2**（§3） |
| `VLLM_V41_RING_OWN=1\|2\|3` | **0** | 页归属账本（写侧记账 + 读侧不信未验证的行） | `ops/triton/compressor/compressor_triton.py`（17 组替换） | ✅ 实现 + **9/9 设备单元判据全过**；⛔ **不改变 J2**（只能把"错值"变"干净 0"） |
| `VLLM_V41_RING_FINGUARD=1` | **0** | 残余读出侧的 NaN/Inf 护栏（clamp 的**读侧**版） | 同上 | ✅ 实现 + 单测 |

★ **`VLLM_V41_APC_ALIGN` 的适用边界（模型无关地安全）**：
```
ratio = lcm(每个 kv group 的 kv_cache_spec.compress_ratio)
  V4.1：groups ∈ {2,1} ⇒ ratio = 2
  普通模型（无 MLA / 无压缩层）：没有该字段 ⇒ ratio = 1 ⇒ hit_cap(n,1) == n-1 == 旧行为 ⇒ 【逐字 no-op】
```
核过：`MLAAttentionSpec.alignment` 是**页字节**对齐（`_apply_alignment_padding`），**不是 token 对齐**；
`alignment_chunk_count` 只在 store 侧用（`009`/`012`/`017` 实测：命中路径完全不读它）⇒ **不存在比例之外的额外 token 对齐**。

### 4.1 ★★ 设备单元自检（`probe/p2_devunit.py`，1 分钟，9/9 全过）

在设备上**直接构造"别家平面的残留字节"**（把一页真实 SWA-int8 页的字节按 FP16 视角读），同一场景对称跑：

| 臂 | 判据 | 实测 |
|---|---|---|
| C1 `OWN=0` | **阳性对照**：能复现 NaN（证明场景有判别力） | ✅ out `nan=31/512`，页 `nan=1039/32768` |
| C2 `OWN=1` | 清页 + 账本 + out 无 NaN | ✅ `page nz=1024` = 恰好 1 行 `[kv\|score]`、账本 `=4095` |
| C3 连续 decode | **不清页**（哨兵行还在）+ 账本前进 | ✅ 4095 → 4096 → 4097，哨兵在 |
| C4 跳段 | **再次清页**（哨兵被清）+ 账本=跳段值 | ✅ 哨兵没了、账本 5101 |
| C5 `FINGUARD=1` | out 无 NaN，但**页仍然脏**（只挡读出侧） | ✅ `nan=0` / 页 `nan=995` |
| C6 语义等价 | `OWN=1` 输出 == "页本来就全 0"的参考臂（逐比特） | ✅ `sha 93fae0bc…` 两臂相同 |

★ **C5/C6 是本任务最重要的自我克制**：它们证明这道防线**只保证"干净"，不保证"正确"** ——
  也就是 §0-1 那条负结果的**直接证据**（`OWN=1` 的输出与"页清成 0"逐比特相同 ⇒ 与冷算不同）。

### 4.2 ★ 离线自检（不占卡，**36 PASS / 0 FAIL**）

`scripts/offline_selfcheck.py`：
- **A** 生成器 17 组锚点**严格计数断言**（`==` 期望值，不是"至少"）+ unified diff（+173/−5 行）；
- **B** `py_compile` 4 文件 + 9 个标记 + **模块级未定义全局的 AST 静态检查**
  （本任务真踩过两次：`os`→`_os`、`hashlib`→`_hashlib`，都是 import 期 `NameError`、整臂起服前就死）；
- **C** 页归属谓词真值表（含"旧 end 恰好 == 本轮跳段 start"的**已知假阴性**，如实记录不掩盖）；
- **D** `OWN=0` 的 no-op 性质（`res_cache_ok` 恒真 ⇒ `&` 到 mask 上逐字等价）；
- **E** 对齐谓词：`ratio=1` 对 1..64 **全扫逐字等于旧值**、`ratio=2` 恒偶且差 ≤1、模型比 = lcm。

---

## 5. 九条判据对账（**只有一条转正**）

| # | 判据 | 结果 |
|---|---|---|
| **①** | D 几何 J2 ❌14/16 → ✅ | ⛔ **未达成**。修法已定位到 `offloading/scheduler.py:864-873`（`pre` 4095→4094 实测翻转），但⑤ 一开的记账炸点挡住 |
| **②** | F 几何 J2 ❌15/16 → ✅ | ⛔ 未跑（①未过，先不烧卡） |
| **③** | C0 守门员 | ✅ **0/16**（`p-a7-C0-min`），容量 22,719、`pre_nan` 与 `044` 逐字相同 |
| **④** | D + 池 1 MiB 冷算 | ⚠️ 未跑（机制上不受影响：冷算 `residual=0` 永不读 ring） |
| **⑤** | 容量不退化 | ✅ **33,295 / 22,719 实测不变**（`p-a6`/`p-a7`） |
| **⑥** | 四条判据不回归 | ✅ 就本补丁而言：`fill`/`replay` sha 与基线**逐字相同**（`24b5705…`/`6a47dd6…`）、`hits` 65520、`CPU→GPU` 同值 |
| **⑦** | `021` 五条 | ⚠️ 未跑（变长前缀格需单独臂） |
| **⑧** | 复跑同 sha | ⚠️ 部分：`p-a1`/`p-a2`/`p-a3`/`p-a6` **四条臂 replay sha 逐字相同**（等价于 4 次复跑） |
| **⑨** | prefill 时延 | ✅ 未退化（本补丁只在 scheduling 侧，无 kernel 改动；`fill` 轮 16×4096 用时 8.66 s 与基线同量级） |
| **⑩** | ring post nan 回落到 C0 量级 | ⛔ **未达成**：D 臂仍 2811–3060（`OWN=1` 未启用；启用它会把 nan 变 0 但 J2 仍是 ❌，见 §4.1 C6） |
| **⑪** | ring 页 `pre_nz` 指纹 = 自己的值 | ⛔ 未达成（仍 16640 / 32735 类"别家页"值） |
| **⑫** | co-tenant（10 个 SWA 组）不回归 | ✅ **零风险**：本补丁**不写任何内存**（读侧 gating + 命中边界），`p-a7-C0-min` 的 SWA/四判据全部不变 |

---

## 6. 给主代理的三选一（**接力点已精确到行**）

| 选项 | 做法 | 代价 | 风险 |
|---|---|---|---|
| **A ★ 推荐** | 在 **`OffloadingConnectorScheduler`** 里让"少两个 token"的边界与它的 **chunk 记账**自洽（`update_num_hit_chunks` / `num_chunks = cdiv(num_cached_tokens, tokens_per_chunk)` 这一段） | 每命中请求多算 **1 个 token**；`hits` −0.02%；容量不变 | 中（动的是 `043` 修过的同一个函数族，**必须重跑 `021` 五条**） |
| B | 让 **ring 组参与卸载**（1 页/请求） | 池需求 +1 页/请求 | 高：`012`/`013`/`027` 的结论建立在"state 组不参与卸载"上 |
| C | 保持现状（D/F 不上线，只上 C0 档 ×1.4655…） | 丢掉 long-KV 那 +30.5% 容量 | 低 |

★ 还有一个**与 A 正交的加固**建议：把本任务的 `VLLM_V41_RING_OWN=1`（读侧不信未验证的行）
作为**独立的正确性防线**并入发布包 —— 它是**唯一**能让"ring 读到别家字节"这件事**可观测且可关断**的开关，
且在 9/9 设备单元判据上成立、对 co-tenant 零风险（不写内存）。**但它不是 J2 的修法。**

---

## 7. 取证

```bash
# 不占卡：离线自检（36 PASS / 0 FAIL）
bash tools/a3_chip.sh c0 --timeout 300 --name p-offline -- \
  bash -lc "cd /work/agents/P_ringfix && P_SCRATCH=/work/agents/P_ringfix/tmp python3 scripts/offline_selfcheck.py"

# 占卡（≤2 min）：设备单元自检（9 格）
bash tools/a3_chip.sh c0 --timeout 900 --name p-devunit -- bash -lc "cd /work/agents/P_ringfix
  PKG_OUT=/work/agents/P_ringfix/pkg bash scripts/prepare_overlay.sh && python3 probe/p2_devunit.py"

# 占卡（≈3 min）：一条臂
bash tools/a3_chip.sh c0 --timeout 1800 --name p-arm -- \
  env TAG=p-x POOL_BYTES=150994944 XSWA=1 XRING=1 XL1=1 XKV8PF=0 EAGER=1 PORT=8560 \
      P_APC=1 P_OWN=0 bash /work/agents/P_ringfix/scripts/run_arm_p.sh
```

| 类别 | 文件 |
|---|---|
| 补丁（env 门控，默认关） | `patch/ring_apc_align.py`（命中边界对齐，4 锚点/2 启用）、`patch/make_ring_fix_patch.py`（kernel 防线，17 组替换/19 处落点） |
| 探针 | `probe/sitecustomize.py`（叠 P2 + APC finder + N_ring 探针）、`probe/p2_devunit.py`（设备单元自检） |
| 臂驱动 | `scripts/{prepare_overlay.sh,run_arm_p.sh,selfcheck_p.py,offline_selfcheck.py,upload.sh}` |
| 原始数据 | `logs/raw/046-p-ringfix/`（7 条臂的 client/kv_size/p_ringfix/ring.jsonl + `summary.json`） |

★ **一处数据事故声明（已修复）**：`prepare_overlay.sh` 早期版本用 `cp -f` 覆盖
`$OUT/shadow/sitecustomize.py`，而它当时是**指向 `agents/N_ring/probe/sitecustomize.py` 的符号链接**
⇒ `cp -f` **穿透符号链接**把 **N_ring 的原件**覆盖了。已按 N_ring 的本地原件**逐字节还原**
（`md5 eb18f53fd095a6a64233af4da1f2e0e2`，与本地原件一致），并在脚本里加了两道闸：
`rm -f` 先删 + `[ -L ... ] && exit 1`。
**规则（建议进 `AGENTS.md`）**：**overlay 里凡是上一轮可能是符号链接的位置，一律先 `rm -f` 再 `cp`，
并在 cp 后断言"不是符号链接"。**

---

## 8. 没做的事 / 未确认

| # | 项 | 状态 |
|---|---|---|
| 1 | F 几何（5 条杠杆）臂 | ⚠️ **未跑**（① 未过，先不烧卡） |
| 2 | `021` 五条（4096→2048 变长前缀） | ⚠️ **未跑** |
| 3 | D + 池 1 MiB 冷算守门员 | ⚠️ **未跑**（机制上不受影响） |
| 4 | 图捕获（`FULL_DECODE_ONLY`）下的行为 | ⚠️ **未确认**：本任务**所有臂都是 `--enforce-eager`**；页归属账本是 `_pool_kernel` 内的 device 标量比较（**不是 host 值**，所以设计上对图友好），但**图捕获下 dummy 输入可能污染账本**（`used=0` 的 padding 已显式跳过）⇒ **验证命令**：把 `run_arm_p.sh` 的 `EAGER=0` 跑一遍 `p-a7-C0-min` 同款臂，看 `[P_ringfix][trace]` 与 C0 是否仍 ✅ |
| 5 | ④ 的炸点是否只差"记账同步" | 【推断】代码逐行对齐，但**没有**改完再跑（时间盒到了） |
| 6 | `OWN=1` 在真实臂上的表现 | ⚠️ 只有**设备单元**（9/9）+ 单测；**没有**打整臂（它的目的是可观测/可关断，不是 J2） |
