# 交付件的「身份台账」—— **每个 md5 在哪些臂上跑过、结果如何**

> 2026-09-22 12:1x 建立。**起因是一次真实事故**（见 §0）。
> 标记：【实测】= 有 `arm.out` 台账支撑；【未确认】= 没找到台账。

---

## 0. 为什么要这张表（事故复盘）

2026-09-22 上午，同一个交付件（`attention/dsa_v41.py`）在 2 小时内出现过 **5 个不同 md5**：

| md5 | 行数 | 第一次见到的位置 / 时间 | 备注 |
|---|---:|---|---|
| `83508822b8556c5f2e55bbeaa4fd82ff` | — | `S_graphfix/out/` 10:32 | 只做离线自检 |
| **`22cbf20c2544dd2ac6cb991a84806c42`** | 1782 | `S_graphfix/pkgs/pkg-kv8pf/` 11:05 | ★ **档 C 图模式 PASS 与档 D 图模式 FAIL 都是它** |
| `1cc9e9923cc19749872cfb2e4decc4b7` | 1749 | `a2/publish/kv8-graphsafe/` 11:22 | ⛔ **从未在 8 卡上跑过**（却在 `DELIVERY.md` 里被写成"与 8 卡实测通过的那份逐字节相同"——**那句是错的**） |
| `94aeebb757d6d5708268754481a05e0a` | 1797 | `S_graphfix/pkgs/` 11:37 | ✅ 当前发布件 |
| （`75f4e565…` / `9db97849…`） | 1499 / 1319 | — | 基底 / 不含 prefill 的参考件 |

**根因不是谁手滑，而是流程缺一道机械门**：换个 md5 只需要重跑一次生成器，
**没有任何一处会因此报错**，于是"已过"的结论就悄悄挂到了没跑过的文件上。

**修法**（从这轮开始执行）：
1. ★ **发布件只允许取"某条 PASS 臂的 `arm.out` 里记过"的那个 md5**；
2. 每个 patch 件换 md5 时，**必须**在本表登记一行，并重标所有"已过"的判据；
3. `scripts/check_artifact_identity.sh` 在发布前跑一遍（见 §2）。

> ★ 幸运的是这套机制**已经有一半在跑**：`S_graphfix` 的 `scripts/*.sh` 会在每个臂的 `arm.out` 里
> 打一条 `--- [S_graphfix] 本臂实际挂的 dsa_v41.py（SG_PKG_D）---` + `md5sum`。
> 事故里正是靠它才查出"档 C 的 PASS 不在发布件上"。

---

## 1. 台账（★ 每次换 md5 都要动这张表）

### 1.1 `kv8-graphsafe/dsa_v41.py`（档 C / 档 D 的必需件）

| md5 | 跑过的臂 | 结果 | 归档 |
|---|---|---|---|
| `83508822…` | **无** | 只做离线自检【未上机】 | — |
| `1cc9e992…` | **无** | ⛔ **从未在 8 卡上跑过**（却被 `DELIVERY.md` 误写成"与 8 卡实测件逐字节相同"） | 已作废 |
| **`22cbf20c…`** | `sg-a-c-graph`（档 C 图模式，8 卡，11:05） | ★ **PASS**：`EE1016=0` + 四条判据 + `replay1 sha` 与 eager 逐字相同 | ★ 已**重建**落盘：`a2/agents/S_graphfix/patch/dsa_v41.graphsafe.22cbf20c.py`（md5 逐字节相等 = 等价，但**不是**从容器捞出的原件，按【实测·重建】标注） |
| **`22cbf20c…`** | `sg-a-d-graph`（档 D 图模式，8 卡，11:27） | ⛔ **FAIL**：起服 segfault（`aclnnRepeatInterleaveIntWithDim`） | 同上 |
| **`94aeebb7…`** | `sg-c-d-graph`（档 D 图模式，8 卡） | ✅ **全绿**：判据 0 全 0 / 捕获 9/9 / 容量 485,610（= R）/ 四条判据 / ★ **replay1 sha == 同几何 eager 逐字节** | ★ 当前发布件 |
| **`94aeebb7…`** | `sg-c-c-graph-b`（**档 C 图模式，8 卡，13:0x**） | ✅ **全绿**：捕获 9/9 [00:54] / `EE1016=0` / 容量 **427,643**（= 档 B）/ **`fill` sha `d524172f…`、`replay1` sha `bc2e797a…` 与 `22cbf20c` 那轮逐字相同** / `hits` 901,120 / `load_bytes` 21,188,968,448 B / **12.50×**（1,594.8 vs 19,936.0 ms） | ★ 当前发布件 |
| **`94aeebb7…`** | `sg-c-d-cmplegacy`（档 D 反例：cmp 面留旧路径） | ⛔ **FAIL（预期）**：`507057 SUSPECT REMOTE ERROR`，第一个真实请求即崩引擎 | 证明补丁必要 |
| **`94aeebb7…`** | `sg-d-d-short-on`（同几何 + 补丁开） | ✅ rc=0、致命证据 0、`fill` `b3eeeaba…` / `replay` `87a5e4fd…` | ★ 同几何 A/B 的 B 臂 |

### ★★ 档 C 的 PASS 能不能平移到 `94aeebb7…`？——**主代理独立复核：能**（但发布仍走机械门）

`S_graphfix` 报了"两版之间只差 cmp 面"，我不采信自报，自己用 `difflib` 重算了一遍：

```
非 equal opcodes（旧 = 22cbf20c）：
  replace  旧[928:931] → 新[928:942]     ★ 落在 cmp 图安全分支内
  insert   旧[954:954] → 新[965:968]     ★ 落在 cmp 图安全分支内
  replace  旧[957:958] → 新[971:973]     ★ 落在 cmp 图安全分支内
⇒ 3 处改动**全部**在 cmp 分支（904..960）内
```

三条**独立的**支撑事实（都是主代理自己算/查的）：

1. **窗口面上界分支逐字节相同**：`旧[547:567]` 与 `新[547:567]` 的 md5 **都是 `985f80d1c2202743`** ⇒ 档 C 走的那段**一个字节没动**；
2. **捕获期路由改动不是新加的**：`_sg_is_capturing` 与 **3 参签名 `_kv8_graph_rows_bound(swa, query_rows, num_reqs)`**
   在 **`22cbf20c` 里就已经存在**（`grep` 命中 2 处 + 签名在 :471 + 调用点在 :1005）⇒ **档 C 的 PASS 本来就跑在这条新路由上**；
3. **档 C 不走 cmp 面**：`_kv8_cmp_plane` 的调用条件是 **`if source_scale is not None:`**（源码 :1052-1053），
   而档 C（`KV8_FULL=0`）的 long-KV 是 BF16 ⇒ `source_scale is None` ⇒ **那个函数根本不被调用**。

**⇒ 结论**：档 C 的 PASS **可以平移**到 `94aeebb7…`（它受影响的那段代码档 C 不执行）。
**但**：这仍是【推断·源码级】，**发布口径按机械门走** ⇒ `sg-c-c-graph-b` **保留为必要判据**。
★ 这条区分很重要：**"我能论证它等价"** ≠ **"它被测过"** —— 前者让我们敢排后续工作，后者才允许发布。

### 1.2 卸载层四件（`0001*` / `0002*`）

| 文件 | md5（★ 与现盘一致，`check_artifact_identity.sh` 每次核） | 跑过的臂 | 结果 |
|---|---|---|---|
| `0001-offload-scheduler.patch.py` | `79001c2671fdbdcd8386cd4684ed4761`（2049 行） | `M_bpcfix` 的 5 条 tiny 臂（`bpc` 泄漏修复，`logs/043b`）+ `Q_apcrecord` 的 9 条 tiny 臂（`[APC_ALIGN]`，`logs/047`） | ✅ 端到端通过 |
| `0001-8card-offload-scheduler.patch.py` | `f3a7a0053fc6c639150fdde2a2509a63`（2045 行） | ★ `sg-a-c-graph` / `r8-*` 的 `arm.out` 台账记的就是它 | ✅ 8 卡通过 |
| `0002-offload-cpu-pool-host-registered.patch.py` | `2c161a791fe99f17cce2e1139ffbdc3c` | `R_8card_int8` 8 卡臂（`P1_pinned ret=0` ×128 行 / 8 rank） | ✅ 8 卡通过 |
| `0001b-offload-per-group-bpc-manager.patch.py` | `9f11c9ac0de0d77fbe6a212e42a9966a` | `J_mgrhardening` + 8 卡臂 | ✅ |
| `0001c-offload-per-group-bpc-hooks.patch.py` | `af2fefb8337fdf9fe1c5e55518f665b8` | 8 卡臂 | ✅ |

### 1.2b ★★ `0004-draft-block64.patch.py`（②c：draft 块 128→64，**交付推荐路线的补丁**）

| md5 | 跑过的臂 | 结果 |
|---|---|---|
| `6d29845ea0d7abc432591d69db7fad17` | ★ **单 die 7 条臂**（`054`，tiny 几何 + 真 draft 组）：
`c2c-{b128,b64,b128-graph,b64-graph,d128,d64,neg-b64}` | ✅ **Q1 输出逐字节不变**（B/D 各 7 轮 sha 同、跨臂 16/16）/
**Q2 投机提案 4367/4367 逐条相同** / **Q3 容量双向逐字命中模型** / 图模式 `EE1016=0` / 阳性对照臂当场炸引擎 |
| 同上 | ⏳ **8 卡真权重端到端** | ⏳ **在 c0 排队**（`T_draftceiling` 的 `chain_2c_v3`） |

⇒ ★★ **这条的身份是「单 die PASS、8 卡未跑」**，而机械门只有 `PASS / 未确认 / 作废` 三档 ⇒
**按「未确认」处理**（`--strict` 会挡住）—— **这是对的**：
单 die 的证据覆盖了 **几何 / 寻址 / 不崩 / 图捕获 / 输出不变**，
但**不覆盖 8 卡绝对 token 数（777,318）与真权重数值**。
⇒ ★ **8 卡端到端一落地，就把它改成 `PASS` 并更新本表**（同时更新 `check_artifact_identity.sh` 的 LEDGER）。

★ **为什么它必须随包发布**：它是**当前交付推荐**（②c）的补丁；
而此前它只存在于 `agents/T_draftceiling/patch/`，**不在发布包里** ⇒ A2 上线时**拿不到**
（这正是本轮查出的一个交付缺口，已补）。

★ **`0001-8card` 与 `0001` 的关系**：前者 = 后者 + 5 个 hunk（8 卡链自己的适配），
`logs/043b` 已给出"零其它差异"的对账。
★ **两者都已并入 `[APC_ALIGN]`**（`grep -c _apc_align_mode` = 2）—— 这正是 `logs/048` §4 里
`[R8-INT8-TRACE] align_unit=1024` 有读数的原因。

> ⚠️ **一个容易写错的中间版本**：`logs/043b` 里说的"新 md5 `986c9115…`"是 **`M_bpcfix` 当时的交付件**，
> 它**还没有**并入 `[APC_ALIGN]`；`09:3x` 并入之后的现盘件是 **`79001c26…`**。
> ⇒ 引用 md5 时**只认现盘 + 本表**，不要从早期日志里抄。

### 1.3 其它

| 文件 | md5 | 状态 |
|---|---|---|
| `kv8-graphsafe/apply_graphsafe.py` | `4be07bea6cc3127eb8715a1da81f583a` | ✅ 主代理复算：从基底重放 ⇒ `94aeebb7…`；机械反修 ⇒ exit=2 |
| `scripts/a2_one_shot_probe.sh` | `40e495e83d198393ea544215dbc4fd50` | ✅ A3 双档跑通（`LIGHT=1` 18 s / `LIGHT=0` 140 s） |
| `scripts/serve_a2_offload.sh` | — | 见 `MANIFEST.sha256`（发布时现算） |

### 1.4 `0003*`（KV8 读侧 rebuild 融合）—— ★ 只在**发布仓**里有，工作区 `publish/` 没有

> ⚠️ 这两个件**不在 `scripts/prepare_publish.sh` 的 MAP 里** ⇒ 它们是发布仓里的历史遗留
> （来自 `logs/026` 的 KV8_fuse 轮）。**当前档 C/D 不用它们** —— `kv8-graphsafe/dsa_v41.py`
> 已经自带 prefill triton 接线（`grep -c _kv8_prefill_enabled` > 0）。

| 文件 | md5 | 依据 | 状态 |
|---|---|---|---|
| `0003-kv8-fused-rebuild-triton.patch` | `27edecc675110511d520f9e9af74c6ed` | `logs/026`：融合后整层 **+345.7 → +38.2 µs/层**（判据 ≤60 达标） | 【实测】单 die |
| `0003b-kv8-fuse-triton-kernels.py` | `6ce00b8f6fdd9ba4ad5935876601f8d6` | 同上（**只覆盖 decode**；prefill 自动回退 torch 路径，回退路径已实测逐比特相同） | 【实测】单 die |

★ 结论：**它们是"另一条实现路径"的历史件，不参与当前发布**。等 `sg-c-*` 出结论后，
若与 `kv8-graphsafe` 路线二选一，再决定是否清掉（**目前保留**，避免丢失证据）。

---

## 2. 机械门：`scripts/check_artifact_identity.sh`

发布前跑：
```bash
bash a2/scripts/check_artifact_identity.sh            # 打印台账里每个件的现盘 md5 + 与本文的差异
bash a2/scripts/check_artifact_identity.sh --strict    # 有任何"未在 PASS 臂上跑过"的件 ⇒ exit 2
```
它做两件事：
1. 把 `publish/` 下每个件的**现盘 md5** 打出来（人对着本文核）；
2. `--strict` 时，**本文 §1 里标 ⛔/⏳ 的件会挡住发布**。
