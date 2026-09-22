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
| `22cbf20c…` | `sg-a-c-graph`（档 C 图模式，8 卡，11:05） | ★ **PASS**：`EE1016=0` + 四条判据 + `replay1 sha` 与 eager 逐字相同 | ⛔ **文件已不在盘上**（全 `dsv41-pr` `find` 无） |
| `22cbf20c…` | `sg-a-d-graph`（档 D 图模式，8 卡，11:27） | ⛔ **FAIL**：起服 segfault（`aclnnRepeatInterleaveIntWithDim`） | 同上 |
| `1cc9e992…` | **无** | ⛔ **从未在 8 卡上跑过** | 已作废 |
| **`94aeebb7…`** | `sg-c-d-graph` / `sg-c-c-graph-b` | ⏳ **在跑**（`chain_sg_c.sh`） | ★ 当前发布件 |

**⇒ 档 C 在发布件（`94aeebb7…`）上的状态 = 【未确认】，必须等 `sg-c-c-graph-b`。**

### 1.2 卸载层四件（`0001*` / `0002*`）

| 文件 | md5（★ 与现盘一致，`check_artifact_identity.sh` 每次核） | 跑过的臂 | 结果 |
|---|---|---|---|
| `0001-offload-scheduler.patch.py` | `79001c2671fdbdcd8386cd4684ed4761`（2049 行） | `M_bpcfix` 的 5 条 tiny 臂（`bpc` 泄漏修复，`logs/043b`）+ `Q_apcrecord` 的 9 条 tiny 臂（`[APC_ALIGN]`，`logs/047`） | ✅ 端到端通过 |
| `0001-8card-offload-scheduler.patch.py` | `f3a7a0053fc6c639150fdde2a2509a63`（2045 行） | ★ `sg-a-c-graph` / `r8-*` 的 `arm.out` 台账记的就是它 | ✅ 8 卡通过 |
| `0002-offload-cpu-pool-host-registered.patch.py` | `2c161a791fe99f17cce2e1139ffbdc3c` | `R_8card_int8` 8 卡臂（`P1_pinned ret=0` ×128 行 / 8 rank） | ✅ 8 卡通过 |
| `0001b-offload-per-group-bpc-manager.patch.py` | `9f11c9ac0de0d77fbe6a212e42a9966a` | `J_mgrhardening` + 8 卡臂 | ✅ |
| `0001c-offload-per-group-bpc-hooks.patch.py` | `af2fefb8337fdf9fe1c5e55518f665b8` | 8 卡臂 | ✅ |

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
