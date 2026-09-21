# A2 DRAM KV 卸载 —— 可交付补丁集

> 2026-09-22 02:1x 整理。**这些补丁已在 A3 的 8 卡真权重上验证通过**
> （`logs/016` / `logs/022`：16 请求、`CPU_to_GPU=12.44 GB`、
> `external_prefix_cache_hits=507,904`、**replay 253.4 vs fill 4429.4 ms = 17.5×**）。
>
> **状态：A2 上尚未验证**（等 `a2_pinned_probe.sh` 的探测结果与一次起服）。

---

## 0. 这三个补丁解决什么

| # | 补丁 | 解决什么 | 依据 |
|---|---|---|---|
| **0001** | `scheduler.py`（**per-group bpc 版，含 D2 的参与位修复**） | ① **`state` 组一票否决**（它在取回路径上参与判定、却永远存不出 chunk ⇒ 整轮判死）；② **SWA 池条目太粗**（每 1024 token 存 8 个 block，而窗口只有 128 ⇒ 存了永不取回） | `logs/009` / `logs/021` |
| **0002** | `cpu_npu.py`（池子改走 `aclrtHostRegister`） | 绕开 `aclrtMallocHost` 的 `207001`（真实 8 卡上单次要 8 GiB 就失败） | `logs/014` / `logs/016` |

---

## 1. 文件与 md5

| 文件 | md5 | 说明 |
|---|---|---|
| `0001-offload-scheduler.patch.py` | `15d5548e29af88da71570d5b48abddef` | **`scheduler.py` 的替换版**。★ **它是 D2 版的超集**（`grep -c offload_participat` = **15**），所以**只需挂这一份**，不要再叠加旧版 |
| `0001b-offload-per-group-bpc-manager.patch.py` | `9f11c9ac0de0d77fbe6a212e42a9966a` | `PerGroupBPCManager`（池的格子 = 1 个 GPU block）+ **`logs/041` 的加固**（见下） |

> ★ **2026-09-22 07:2x：`0001b` 已并入 `logs/041` 的加固**（原 md5 `3b64eb49…` → 新 `9f11c9ac…`）。
> **默认行为逐字不变**（`PGP_MGR_HARDEN=0`），只多了一份只读记账。它带来两个可选的保险：
>
> | env | 默认 | 作用 |
> |---|---|---|
> | `PGP_MGR_HARDEN` | **0** | `1` = 把三种静默失败**变成响亮 raise**（缺容量 cap / 过期 free / 索引键被覆盖）；`2` = 更严 |
> | `PGP_MGR_STATS` | `0`（`HARDEN=1` 时自动 1） | 打开只读计数器（`stale_free / dup_unit / oob_unit / over_budget / key_overwrite / used_mismatch`），`mgr_hardening_stats()` 取值 |
>
> **它的承诺边界**（`041` §0，请勿误读）：**只承诺"若配账层将来真坏，它会响"**（阳性对照 + 15/15 自检已证），
> **不承诺**修任何现有 bug（`038` 那条 144 MiB 首 token 错**不在配账层**，开 L1 才是解法）。
> 实测：加固前后 **sha 逐字相同**（`a7ffff6be598`）⇒ **既没修它、也没让它更糟**。
| `0001c-offload-per-group-bpc-hooks.patch.py` | `af2fefb8337fdf9fe1c5e55518f665b8` | 配置解析钩子（`blocks_per_chunk` 支持 `{"default":8,"swa":1}` 的字典形式） |
| `0002-offload-cpu-pool-host-registered.patch.py` | `2c161a791fe99f17cce2e1139ffbdc3c` | `cpu_npu.py` 的替换版（`NPU_OFFLOAD_HOST_MEM=registered` 走 `aclrtHostRegister`；**注册失败自动回落 `pinned`**） |

---

## 2. 怎么用

### 2.1 挂载（与 `serve_a2.sh` 的 `PATCH_MODE=mount` 同构）

```bash
# 0001 系列：把三份文件放进 shadow-pkg 的补丁目录
PKG=<shadow-pkg 路径>
cp 0001-offload-scheduler.patch.py              $PKG/patches/files/offload_dsv41/scheduler.py
cp 0001b-offload-per-group-bpc-manager.patch.py $PKG/patches/files/offload_dsv41/pgp_manager.py
cp 0001c-offload-per-group-bpc-hooks.patch.py   $PKG/patches/files/offload_dsv41/pgp_hooks.py
cp 0002-offload-cpu-pool-host-registered.patch.py $PKG/patches/files/offload_dsv41/cpu_npu.py
```

然后按 `docs/A2-GO-LIVE.md` §2.2 的开关起服。

### 2.2 ★ 起服后**必须先做补丁生效自检**（否则白跑 25 min）

```bash
grep -c "P1_pinned.*ret=0"           serve.log   # 期望 8（8 个 worker）
grep -c "D2_offload"                 serve.log   # 期望 > 0
grep -c "alignment_chunk_count.*8"   serve.log   # 期望 > 0（per-group bpc 生效）
```
**任一为 0 ⇒ 立刻停**（`logs/016` 的 dram32 臂没挂上补丁，白跑了 25 min）。

---

## 3. 参数定值

| 场景 | `OFFLOAD_GB` | 宿主实占（×6.945） | 依据 |
|---|---:|---:|---|
| **32K × 16 并发**（per-group bpc） | **16** | **≈111 GiB** | `logs/021` §6.1 重算 |
| **128K × 16 并发**（per-group bpc） | **48** | **≈333 GiB** | 同上（A2 余量 442 GiB） |
| 32K，**不**用 per-group bpc | 48 | 333 GiB | `logs/016` 实测 |
| 128K，**不**用 per-group bpc | — | **1,333 GiB** | ⛔ **A2 不可能** |

> ★★★ **2026-09-22 04:0x 更新（`logs/035` 集成验证已出）：上表是 `L5 only` 的老口径。**
>
> **① 今天就能上线的组合 = `L5 + L1`**（`035` 实测：容量 ×1.000（HBM 不变）但**池子 1.96×**，且 **6/6 冷臂 + 池命中臂全部逐字节一致**）：
>
> | 场景 | L5 only | **+L1（✅ 可上线）** | 依据 |
> |---|---:|---:|---|
> | 16 × 32K | 66 GiB | **33.6 GiB** | `logs/030` §5 / `035` §6.2 |
> | **16 × 128K** | 264 GiB | **134 GiB** | 同上 |
> | **32 × 128K** | 528 GiB ⛔ | **269 GiB ✅** | 同上 |
> | 64 × 128K | 1,056 GiB ⛔ | 538 GiB ⛔ | 同上 |
>
> **⇒ `OFFLOAD_GB` 定值**：
>
> | 口径 | 16 × 128K | 32 × 128K | 依据 |
> |---|---:|---:|---|
> | `035` §7.2 的换算 | `40` | `80` | L5+L1，含 1.2× 余量（推算） |
> | ★★ **8 卡真权重实测（`027` 的 `B2` 臂）** | **`56`**（57,344 unit，宿主 **380 GiB**） | — | **`BlockRemoved:CPU = 0`、`hits=1,062,400`、`CPU→GPU=25.69 GB`、加速 14.31×、利用率仅 0.757** |
>
> ★ **`027` 的实测口径（23.5 unit/1024token + ×1.2 余量）比 `035` 的推算更保守也更准** ——
> **16×128K 请用 `OFFLOAD_GB=56`**（不是 40）；它留了 1.2× 余量（利用率 0.757 ⇒ 还有真实余量）。
> **运维判据（`027` §3.3）**：**上线后要盯的是 `BlockRemoved:CPU == 0`** ——
> 池子欠配**不会表现为"命中率下降"，而是"全场归零的级联"**，所以`removed` 是唯一早期信号。
> （注意：**同一个 `OFFLOAD_GB` 在不同页几何下的 unit 数不同** —— 记账单位随几何变：131,072 → 77,824 → 69,632 B。）
>
> ⚠️⚠️ **2026-09-22 06:2x 追加（`038`）：上面那条"盯 `BlockRemoved`"的判据有一个盲区。**
> `038` 实测出**第二种更阴险的失效模式**：
> ```
> 池 unit == 工作集（1.000×）时：BlockRemoved:CPU=10、CPU→GPU 搬的字节与安全档【逐字节相同】、
>    主要事件计数全同 ⇒ 但 J2 ❌、第一压缩层 hidden 全 NaN、首 token 静默错
> 池 1.11×（160 MiB）起：同一组指标，J2 ✅、0 NaN
> ⇒ ★ 现成的监测指标【看不见】这一类；只差 1.11×，指标逐字相同
> ```
> ⇒ **运维要求：`OFFLOAD_GB` 必须留 ≥1.2× 余量，不能按 1.000× 配**。
> `027` 给的 **`56`（1.193×）刚好在边界之上**（1.11× 起安全），**但建议不要低于它**。
>
> **② ⛔ int8 两条杠杆（SWA-quant / KV8）暂不能上，但理由已更正**（`036`）：
>
> | 时点 | 结论 |
> |---|---|
> | `035` §5 | 判为"**int8 平面经 DRAM 池往返不保真**" ⇒ 否决 |
> | ★ `036` | **该表述被证伪** —— **12,928 次 store/load 逐字节比对全部 mismatch=0**、行覆盖 0、尺寸阶梯 128→131072 B 全绿、eager 与图模式同结果；且 attention **真正读到的那些页**（warm 960/960 条）在 cold 里**逐字节找到** ⇒ **池往返逐字节保真** |
> | ★ `036` 的真凶 | **触发条件 = int8 SWA 面 + "首 token 全由 1 行 decode 产生"**：prompt 取 **`block_size` 整数倍**（4096/2048）⇒ ❌ 1/16、13/16；取 **非整数倍**（4095/2047，池臂必须 prefill 补算尾块）⇒ **✅ 16/16**（两条独立几何各测一次）。反证：`ring16 + SWA 保持 BF16`（`035` 的 `x-R2`）**✅ 逐字节** ⇒ **必须要 int8 SWA 面** |
> | **修法落点** | 【推断】在 **attention 读侧**（`kv8_ori_plane` 的 decode 2 页快路径），**不在卸载层**；下一步一条命令 = "改 decode 分支为全序页表 + 整表重建后重跑"（`036` §遗留） |
>
> ⇒ 容量是真的（×1.4655 / ×1.9133），**卡在 int8 读路径对调用形状敏感**（可修，且修点已收敛到一个函数）。
> **修好后**：4 条杠杆 ⇒ `OFFLOAD_GB=48` 对应「**32 × 128K**，宿主 199 GiB」——**同样是 48，含义从"16×32K 上限"变成"32×128K 定值"**。

**其它必须的参数**（`docs/A2-GO-LIVE.md` §2.2）：`PREFIX_MATCH_UNIT=32`、`blocks_per_chunk` 用 per-group 字典、
`--enable-prefix-caching`、`--kv-cache-memory-bytes ≥ max_model_len × kv_per_token`。

---

## 4. 已知边界

| # | 事项 | 状态 |
|---|---|---|
| 1 | **A2 上的验证** | ⏳ **未做**（等 `a2_pinned_probe.sh`） |
| 2 | `aclrtHostRegister` 在 A2 上是否可用 | ⏳ **未测**（A2 `host_mem_pool=0`，Engram 曾注册失败） |
| 3 | `×6.945` 宿主乘数 | 【实测】A3 8 卡 / 单卡 tiny 两处一致；**A2 要重测** |
| 4 | per-group bpc 的 8 卡验证 | ⏳ **进行中**（`logs/027`） |
| 5 | `state` 组跳过的**数值正确性** | 只做了语义推断（与 GPU 前的缓存路径一致），**没做精度对比** |
