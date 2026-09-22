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
| `0001-offload-scheduler.patch.py` | `79001c2671fdbdcd8386cd4684ed4761` | **`scheduler.py` 的替换版**。★ **它是 D2 版的超集**（`grep -c offload_participat` = **15**），所以**只需挂这一份**，不要再叠加旧版。<br>★ 07:3x 已修 `blocks_per_chunk` 局部变量泄漏；★ **09:3x 已并入 `[APC_ALIGN]`（`logs/047`）** —— 见下 |

> ★★★ **`0001` 的第二次修复（2026-09-22 09:3x）：`[APC_ALIGN]` 压缩层命中长度对齐**（`logs/047`）
>
> **根因**（`Q_apcrecord` 定位到一行）：上游 `kv_cache_manager.py` 的
> `max_cache_hit_length = request.num_tokens - 1` **只对齐到 `block_size`、不知道模型的压缩比**
> ⇒ 4096-token prompt 的命中边界 = **4095（奇数）** ⇒ `compress_ratio=2` 的组**跨在命中边界上**
> ⇒ replay 时必须回读 compressor **state ring** 里 token 4094 那一行，
> 而 ring 组 `prefix_cacheable=False`、**不参与卸载** ⇒ **那一行的原始投影从未被存过**
> ⇒ int8 几何把它**解读成 NaN** ⇒ 翻 token（D/F 几何 ❌ 14/16、15/16）。
>
> **修法**：命中长度向下对齐到**段栅格**（= 参与卸载的 full-attention 组的 `tokens_per_chunk`，
> V4.1 = 1024，运行期从 `alignment_tokens` 现算、**非硬编码**）。`4096 → 3072`
> ⇒ **落点正好是 store 侧已经保留的段尾 chunk** ⇒ ★ **store 侧零改动、池需求不变**。
>
> | env | 默认 | 作用 |
> |---|---|---|
> | `VLLM_V41_APC_ALIGN` | **0** | `0` = 逐字旧行为；**`3` = 段栅格模式（推荐）**；`2` = 对齐到 ratio（**已否决**，需 store 侧配套、池 +24%） |
>
> ★ **两道安全门**：
> 1. `alignment_tokens is None`（多值/不可用）⇒ 退回 0（逐字旧行为）；
> 2. ★ **必须真有压缩组**（`compress_ratio > 1`）才启用 ⇒ **普通模型（ratio=1）逐字 no-op**
>    （不加这道门会把普通模型的命中窗口也对齐到 1024 —— 纯性能回退且与根因无关）。
>
> **实测（`047`，单卡 tiny）**：D ❌14/16 → **✅ 0/16**、F ❌15/16 → **✅ 0/16**、
> C0 保持 ✅、**容量 33,295 / 43,469 一字不变**、**反例臂 `=0` 逐字复现 ❌ 与 `6a47dd65f1ff`**、
> `021` 变长前缀不回归且 **4.89× 倍率不变**。
> **离线自检**：`python3 a2/scripts/selftest_apc_align.py` ⇒ **38 PASS / 0 FAIL**（两版各 19）。

> ★★ **`0001` 的重要修复（2026-09-22 07:3x，`logs/039` §10 定性 + `logs/043` 修复）**：
> 原版 `_build_store_jobs()` 里 **`blocks_per_chunk` 是个裸局部变量**，只在"收集 loop"里逐组赋值
> ⇒ 收集 loop 结束时它停在**最后一个参与卸载的组**（本配置是 SWA，`bpc=1`）
> ⇒ **spec loop 里 `bpc>1` 的组（group 0 full attention）每个 chunk 只搬 `1/bpc` 个 GPU block**，
> 其余 unit 永不写入，**load 侧读到全 0 行**（实测 **448/512**）。
>
> **三条独立测量**：`src_spec` 的 `Σgroup_sizes = 44`（应 **72**）；group 0 实搬 **64** 个 block（应 **492**）；worker 侧 **448/512 读到全 0**。
> **反例臂**：`bpc=1` 的 10 个 SWA 组 **65/65 全中** ⇒ 泄漏**只伤 `bpc>1` 的组**。
>
> **修法（两处成对）**：收集 loop 与 spec loop **各自**取 `bpc_g = group_config.blocks_per_chunk`
> ⇒ 裸名在函数体内**彻底消失**；另加**两条 fail-closed 断言**（`bpc_g` 与本组一致 / `len(_units) == bpc_g`）。
>
> **单元自检 17 PASS / 0 FAIL**：修好件 `Σgroup_sizes = len(src) = len(dst) = 72`；
> **反例臂（机械反修）逐字复现 `44 / group_sizes=[4,0,4×10]`**；两条断言在反修臂上**真的会炸**。
>
> **端到端判据已全部翻转【实测】**（`logs/043-bpc-leak-fix`）：
> ```
> Σgroup_sizes / len(src.block_ids)   44 -> 72        （= 8x4 + 1x40）
> group 0 实搬 GPU block              64 -> 492       （12 张量全 492；g0_store op 768 -> 6144 = x8）
> worker 读到"未写过/全 0 行"         5352 -> 0
> ★ 算术闭环：GPU→CPU 196,689,920 -> 362,127,360，增量 165,437,440 = 448 block × 369,280 B
>   （正好补上漏搬的 7/8）；而 CPU→GPU 逐位不变 ⇒ 取回路径一个字节没动
> 五条判据（021 的 + 030/L1 的）全不回归；4096→2048 变长前缀仍安全（hits=32,752 与 021 §5 逐字相同）
> 复跑同 sha（24b57053…）；反例臂（10 个 SWA 组 = [4]x10）逐项不变
> ```
> ⚠️ **边界**：它**不是** `038` 的 ❌/✅ 翻转答案（`160 MiB` 臂在**同一份 448 行全 0** 下 BF16 输出 sha **逐字相同**）
> ⇒ 定性为**潜伏的正确性风险已消除**，不是"首 token 错的原因"。`concurrency>1`、A2 真机、8 卡口径均**未测**。
>
> ★ **8 卡成品补丁**：`agents/M_bpcfix/publish/0001-offload-scheduler.patch.py.8card`
> （md5 `6a4f8dffcbb1f3ab4b6c5a1d749e6eaf` = 现 `L3_8card/patched/scheduler.py` + 本次 5 个 hunk）
> ⇒ **覆盖 `L3_8card/patched/` 即完成 8 卡上线阻塞**（8 卡口径未复核【未确认】）。
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
| ★ `0001-8card-offload-scheduler.patch.py` | `f3a7a0053fc6c639150fdde2a2509a63` | **8 卡链专用的 `scheduler.py`**（= `L3_8card/patched/scheduler.py` 的 `f4de89d2…` + `043` 的 bpc 泄漏修复 + **`047` 的 `[APC_ALIGN]`**）。<br>★ **8 卡挂载链请用这一份**：覆盖 `agents/L3_8card/patched/scheduler.py` 即可。<br>⚠️ **8 卡口径未复核**【未确认】（`R_8card_int8` 正在验）。 |

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

---

## ★★ `0004-draft-block64.patch.py` —— ②c：draft 的块大小 128→64（**默认关**）

| | |
|---|---|
| md5 | （见 `scripts/check_artifact_identity.sh`） |
| 作用 | 给 **draft 组**单独的 `block_size`（env `VLLM_V41_DRAFT_BLOCK`，默认 128 = **逐字旧行为**）；
并把 `plan_cache_slots` 的 draft 几何检查从**相等**放宽成**整除**（`swa % draft == 0`） |
| 改动面 | **2 个文件 / 2 处**：`models/deepseek_v41/dspark.py`（加 `__init__` 读 env）+ `core/deepseek_v41.py`（放宽检查） |
| 前提 | 需要**已经打过 KV8 的 `core/deepseek_v41.py`**（阴影包版）+ R 的 draft-aware 槽位补丁 |
| 收益 | ★ 预测 **HBM ×1.8177**（8 卡 427,643 → **777,318**）；tiny **20,826 → 19,247（B 档降）/ 23,651 → 36,825（D 档涨）** |
| 风险 | ★ **无精度风险**（保持 BF16）；窗口跨 2 块（算子/块表/KV manager **都无 ≤1 块假设**，`051 §2` + `054` 实测）；
  DRAM 池 **+5.0%**（`sw_chunks` 1→2） |
| 证据 | `logs/051`（改动清单 + 三臂单元自检）、`logs/054`（单 die 7 臂：输出逐字节不变、投机提案 4367/4367 逐条相同、图模式过）、
  ★ 8 卡端到端 **⏳ 在 c0 排队** |
| 用法 | `python3 0004-draft-block64.patch.py --core <影子包/core/deepseek_v41.py> --dspark <镜像/models/deepseek_v41/dspark.py> --out-dir <patched>` |

