# A2 交付包 —— 当前能达到的**最优性能配置**（2026-09-22 08:xx）

> 本文是**给 A2 上线用的单一入口**：配置、证据、边界、回滚，一页说清。
> 详细论证在 `docs/A2-GO-LIVE.md` 与 `logs/README.md`（001–04x）。
> 标记约定：**【实测】** = 本机跑出来的原始数据；**【推断】** = 由代码/算式推出但没直接测；**【未确认】** = 没跑到。

---

## 0. 一句话

**A2 从「16 并发 × 32K」提升到「16 并发 × 128K」，重算→取回快 12.8~14.3 倍，占用主机内存 ≈197 GiB。**

**全部为 8 卡真权重实测**：
```
容量      392.35 GiB -> 197.21 GiB（档 A -> 档 B，1.9895x 更省，四条独立路径一致）
功能      hits=901,120 / CPU→GPU=21.52 GB / replay 12.87x / BlockRemoved:CPU=0
输出      16/16 prompt 的 sha 与档 A【逐字节相同】=> L1 不改变任何输出 token
```
**int8（容量再 ×1.9）已技术全通，但仍被一个正确性缺陷挡住，本包不含它**（已系统性排除 6 个方向，见 §4.1）。

---

## 0.5 ★★★ 证据强度分层（**这份交付包"哪一层验过、哪一层没有"**）

> 本包**可以发**，但三层证据的强度**不一样**，请按层取信、不要外推。

| 层 | 档 A | 档 B（+L1） | 说明 |
|---|---|---|---|
| **① 容量 / 内存的算术** | ✅ **8 卡实测 + 逐字节闭环** | ✅ **8 卡实测 + 逐字节闭环** | 见 §1/§2：`OFFLOAD_GB=56` 由 `_dbg_tally` **逐请求反解**；392.35 GiB 两处独立吻合；197.21 GiB 的 `23,142×369,280 + 34,714×131,072×3 + 28,929×147,712 = 26,469,138,432 B` **与日志逐字节相同**；`P2_COMP_JSON` 是 worker 侧真值（8 rank × 9 行一致） |
| **② 功能判据**（命中率 / 取回加速） | ✅ **8 卡实测** | ✅ **8 卡实测（逐字不回归）** | 档 A：`027` 四条判据全中（`hits=788,480`、`CPU→GPU=18.83 GB`、replay **12.81×**、`BlockRemoved:CPU=0`）。★ 档 B（`042`）：四条判据与 A **逐字相同**（`BlockStored:CPU=29,436` / `CPU→GPU=21.52 GB` / `hits=901,120` / replay **12.87×** 略快），`BlockRemoved:CPU=0`、池利用率 `0.7566` 也逐字相同 |
| **③ 数值正确性**（取回的 KV 算出来的结果对不对） | ⚠️ **有一条**（见下） | ⚠️ **有一条**（见下） | ★ 档 B 拿到 **16/16 prompt 的输出 sha 与档 A 逐字节相同**（`fill=d524172f…`、`replay=68bb72d8…`）⇒ **"L1 不改变任何输出 token"，这是有判别力的跨臂比较**（唯一差别是 L1、同长度比较）。★ 但"**绝对值正确**"仍未在 8 卡上验过（tiny 的 KV 级字节判据 `039` 是唯一一条，且当时带 bpc 泄漏）—— 因为 `037` 证明文本 sha 在 `temperature=0` 下**同臂比较全部失效** |

> ★★ **K_l1_8card 顺手加的一条更强证据（旁证，四条独立路径一致）**：
> ```
> ① P1 aclrtHostRegister 求和 / ② L1 worker 物理池 / ③ P2_WORKER_HOST_BYTES  → 三条一致
> ④ ★ 同一容器 cgroup 口径：A 1,281,099,722,752 B - B 1,068,938,686,464 B = 197.59 GiB
>    与预测的 195.15 GiB 差 1.3%；宿主 MemAvailable 也同向（差 200.3 GiB）
> ```

**⇒ 一句话**：**"配置怎么算"已实测对了；"功能跑得通"档 A 对、档 B 待出；"算出来的结果数值对不对"——8 卡上从来没验过。**
**⇒ 另一个必须知道的**：`0001` 泄漏修复本身是**tiny 端到端全翻转**（`44→72`/`64→492`/`5352→0` + 过账闭合），
**但 8 卡口径未复核** —— 成品补丁已备好（§7-1b），**建议 A2 上线前先跑一次 8 卡臂**。

---

## 1. ★★ 推荐配置（**档 A：8 卡真权重已验**）

```bash
MODEL=<模型目录> \
OFFLOAD_GB=56 \                 # 57,344 unit = 实测需求 48,064 的 1.193×
MAX_LEN=131072 \
MAX_SEQS=16 \
BLOCKS_PER_CHUNK='{"default":8,"swa":1}' \   # ★ per-group：SWA 用细粒度
PREFIX_MATCH_UNIT=32 \          # ★ 必须；否则撞 tokens_per_block=32 % tokens_per_hash=128
ENGRAM=0 \                      # ★ 首版建议 0（Engram + 卸载池曾撞 207001）
NPU_OFFLOAD_HOST_MEM=registered \   # 池走 aclrtHostRegister
OFFLOAD_SCHED_PATCH=1 OFFLOAD_NPU_WORKER_PATCH=1 \
bash a2/scripts/serve_a2_offload.sh
```

补丁（`a2/patches/`）：

| # | 文件 | 作用 |
|---|---|---|
| `0001` | `offloading/scheduler.py` | **D2 的参与位修复**（排除 `state` 组，否则整轮判死）+ **per-group bpc** |
| `0001b` | `pgp_manager.py` | 池的格子 = 1 个 GPU block（L5 的核心）；**已含 `041` 的加固**（默认关） |
| `0001c` | 解析钩子 | 让 `blocks_per_chunk` 支持 `{"default":8,"swa":1}` 字典 |
| `0002` | `native/cpu_npu.py` | 池改走 `aclrtHostRegister`（绕开 `aclrtMallocHost` 的 `207001`） |

### 1.1 档 A 的实测证据（8 卡真权重，`logs/027`）

| 判据 | 实测值 |
|---|---|
| **replay / fill TTFT** | **1,420.6 / 18,202.6 ms = 12.81×**（`B2` ×1.2 池臂 **14.31×**） |
| **`CPU→GPU`（从 DRAM 取回）** | **25.69 GB**（208 load job） |
| **`external_prefix_cache_hits`** | **1,062,400** |
| **`BlockRemoved:CPU`** | **0**（= 池子没撑爆，这是上线监测判据） |
| **宿主实占** | **392.4 GiB**（= 49.05 GiB/worker × 8，`K_l1_8card` 的 A 臂独立复现） |
| 池 unit 需求 | 48,064（= 23.5 unit/1024token/请求，**不是** `019` 模型的 19） |

---

## 2. ★★ 档 B（**内存砍半**，8 卡真权重**已实测**）

### 2.0 ★ 8 卡实测时用的**确切 env**（`042`，可直接抄）

```bash
# [L1-POOL] L5 + L1：容量 +1.99× 且 A/B 输出逐字节相同
export L1_POOL_PATCH=1     # ← 差异①：挂 L1 的 6 个文件（含 L5 全部改动）
export L3_PGP_PATCH=0      #    打开 L1 时必须关（同一路径不能挂两次）
export L1_POOL_DIR=<L1 的补丁目录>
export P2_POOL_PATCH=1     # ← 差异②：0 = 逐字回退档 A
export P2_COMP_JSON='[[0],[1,2,3,4,5,6,7,8,9,10,11,12]]'   # ← 16 张量的真分量
export P2_STRUCT_LOG=1 P2_POOL_LOG=1
# 其余 env（OFFLOAD_BYTES / bpc={"default":8,"swa":1} / scheduler / P1 / held kv 参数）与档 A 逐字相同
```

### 2.0b ⛔⛔ **两组 env 是"同一条链的两半"，缺一即"静默 no-op"**

```
L1_POOL_PATCH / L1_POOL_DIR  = 【挂载机制】（docker -v 把 6 个文件挂进容器）
P2_POOL_PATCH / P2_COMP_JSON = 【运行期开关】（在容器里决定走不走新分配）

★ 只给 P2_POOL_PATCH=1、不挂那 6 个文件 ⇒ 静默 no-op（没有人消费这个 env，池子还是旧分配，且【不报错】）
★ 只挂 6 个文件、P2_POOL_PATCH=0    ⇒ 只读结构臂（= 042 的 A 臂，行为逐字等于档 A）
```

★ 而且 **`P2_*` 能进容器，靠的是给 `serve_a2.sh` 打的一处 `inner.sh` 补丁**（4 行 export）。
⇒ **准确表述**：
> **档 B 的入口必须是「被 `agents/K_l1_8card/scripts/patch_serve_a2_l1.sh` 打过的那份 `serve_a2.sh`」**
> （= 042 用的 `shadow-pkg/scripts/serve_a2.sh`，它被打了两处：`[L1-POOL]` 挂载块 + `inner.sh` 的 4 行 export）。
> 若上线方用的是**另一份 `serve_a2.sh`**（例如 `dsv41-release/tools/` 那份），**必须把同样的两处改动移植过去**；
> 否则 `P2_POOL_PATCH=1` **既不生效、也不报错**。

### 2.0c ★ "没生效会立刻暴露"的四条对称检查（两臂上都跑过，有判别力）

```bash
grep -c "按组配额 manager 生效" <serve.log>   # 期望 1（P2_POOL_PATCH=1）
grep -c "③ \[只读\]"            <serve.log>   # 期望 0（开着补丁时）
grep -c "③ 宿主实占: 旧="       <serve.log>   # 期望 8（8 个 rank 各一行）
docker inspect -f '{{range .Mounts}}{{.Source}} {{end}}' <容器> | grep -c K_l1_8card/patched  # 期望 6
```
★ A 臂（只读=8、宿主实占=0）与 B 臂（只读=0、宿主实占=8）**互补** ⇒ **开关没翻过去一定会被看到**，不会"跑通了但没生效"。
★ `P2_COMP_JSON` 给错 = **fail-closed 拒绝启动**（正反例都跑过，不会静默错）。

| | 档 A | **档 B（+L1）** | 说明 |
|---|---:|---:|---|
| 同样 `OFFLOAD_GB=56` 的宿主实占 | **392.35 GiB**（余量 442 的 **88%**，⚠️ 紧） | ★ **197.21 GiB**（**45%**） | 两条都是 **8 卡真权重实测** |
| 每 worker | 52,660,994,048 B（49.04 GiB） | **26,469,138,432 B（24.65 GiB）** | 8 rank × 8 一致 |
| **比值 B/A** | — | ★ **1.9895×** | 比 `030` 的 tiny 值（1.960×）**更好** |
| **四条判据** | `hits=901,120` / `CPU→GPU=21.52 GB` / replay **12.81×** / `BlockRemoved:CPU=0` | ★ **与 A 逐字相同** / ★ **12.87×**（略快） | `logs/042`【实测】 |
| **输出 sha（16/16 prompt）** | `fill=d524172f…` `replay=68bb72d8…` | ★ **与 A 逐字节相同** | ⇒ **L1 不改变任何输出 token** |
| 依据 | `logs/027` + `042` 的 A 臂【实测】 | `logs/042` 的 B 臂【实测】 | |

**四条独立路径一致**（`042`）：
```
① P1 aclrtHostRegister 求和 / ② L1 worker 物理池 / ③ P2_WORKER_HOST_BYTES  → 三条一致
④ ★ 容器 cgroup 口径：A 1,281,099,722,752 B − B 1,068,938,686,464 B = 197.59 GiB
   （与预测的 195.15 GiB 差 1.3%）；宿主 MemAvailable 同向（差 200.3 GiB）
```

**为什么 8 卡比 tiny 还省（可归因）**：
```
A 臂只读结构臂实测：g12(draft) 只引用张量 [12,13,14]，【不引用 15】
⇒ 张量 15（页 147,712）只需装到 g11 的区间上界 28,929 行，
   而 12–14 要装到 g12 的上界 34,714 行 ⇒ 省得更多
逐字节复核：23,142×369,280 + 34,714×131,072×3 + 28,929×147,712 = 26,469,138,432 B ✅ 与日志相同
```

### 2.1 为什么 L1 能省一半

镜像里 16 张 canonical 张量**每张都分到 `num_cpu_blocks` 行**，但一个 slot 号同时只服务**一个组**
⇒ 15/16 的空间**结构性闲置**。L1 让每张张量只分"引用它的那些组真正需要的行数"。
（`029` 曾推测这是 16×，`030` 实测是 **1.96×** —— 因为 16 张里 12 张只给 `full` 用、另 4 张被 `state` + 10 个 SWA 组共用。）

### 2.2 ★ `P2_COMP_JSON` 必须与**张量数**匹配（给错会 fail-closed 拒绝启动）

| 场景 | 张量数 | 分量 |
|---|---:|---|
| **档 A/B（不开 int8）** | **16** | ★ **【实测·worker 侧真值，8 rank × 9 行全部一致】** `[[0],[1,2,3,4,5,6,7,8,9,10,11,12]]` |
| 开 SWA-quant 后 | 20 | `[[0,2,3,4,5,6,7,8,9,10,11],[1]]`（`035` §4.2） |

**分量的物理含义**（8 卡真权重实测）：
```
group→tensor_idx = [[0..11], [12,13,14], [12,13,14,15] ×10, [12,13,14]]
   g0(full, bpc=8)     独占张量 0–11
   g1(state)           用 [12,13,14]，w=0（★ 被排除卸载，不占池）
   g2..g11(SWA, bpc=1) 用 [12,13,14,15]
   g12(draft)          用 [12,13,14]，w=2（= bpc12 × (sw_chunks + is_eagle) = 1×2）
pages = [65536,8192,128]×3 + [131072,16384,256] + [131072]×3 + [147712]（Σ=910,208）
★ g12 引用 [12,13,14]（不含 15），g2..g11 是 [12,13,14,15] ⇒ 同分量、不冲突
```

★ **正确做法**：先跑一条**只读结构臂**（`P2_STRUCT_LOG=1 P2_POOL_PATCH=0`）拿到真值，再开补丁。

---

## 3. ★★ 两条运维判据（**都会咬人，必须设**）

| # | 现象 | 判据 |
|---|---|---|
| **1** | **欠配不是"命中率下降"，是"级联归零"**（0.802/0.844/0.852× 三条臂**全部 0 命中**，没有中间态） | 盯 **`kv_offload_block_removed_total{medium="CPU"} == 0`** |
| **2** | 池单元数**恰好等于**工作集时，会出现**静默算错**（首 token 错），而**失败档与安全档的现成指标逐字相同**（淘汰数 / 搬运字节 / 事件计数全同）⇒ **指标看不见** | **`OFFLOAD_GB` 必须留 ≥1.2× 余量**；建议追加 `units_ratio ≥ 1.05` |

**反解工作集的公式**（用现成指标，不用起新服务）：
```
工作集(unit) = cpu_cache_usage_perc × num_units
```

**自带的 fail-closed 保险**（`0001b` 已含，默认关）：

| env | 默认 | 作用 |
|---|---|---|
| `PGP_MGR_HARDEN` | `0` | `1` ⇒ 把三种静默失败（缺容量 cap / 过期 free / 索引键覆盖）**变成响亮 raise** |
| `PGP_MGR_STATS` | `0` | 打开只读计数器（`stale_free / dup_unit / oob_unit / over_budget / key_overwrite / used_mismatch`） |

> 承诺边界（`logs/041` §0）：**只承诺"若配账层将来真坏，它会响"**（阳性对照 + 15/15 自检已证）；**不承诺修任何现有 bug**。

---

## 4. ⛔ 本包**不含**的两项（及原因）

| 项 | 状态 | 原因 |
|---|---|---|
| **int8（容量 ×1.4655 / ×1.9133）** | ⏳ **机制已确证、修法进行中**（本版暂不含） | ★★ **`044` 已把机制确证**（见 §4.2）：**「命中路径只刷 1/32 行 ring」× 「FP16 让残余变 NaN」**。<br>容量与性能早已达标（`033` prefill 硬伤已解、`034` 精度达标、`028` 融合 kernel `+2.15%`）。<br>**修法已明确**：ring16 写入侧对超 FP16 范围的值 clamp（±65504）；`P_ringfix` 在实现、`O_fp16nan` 在把 NaN 的来源从【推断】升为【实测】。 |
| **`SWA_TRIM=window`** | ⛔ **弃用** | `017` 证明它对**变长前缀整轮归零**；`021` 的 per-group bpc 才是安全解。 |

### 4.2 ★★★ int8 的真凶已确证（`logs/044`，**16/16 逐请求精确对齐**）

| 臂 | 几何 | `ring post` 的 nan | 下游 latent nan | **J2** |
|---|---|---|---|---|
| `n-e1-C0-hot` | SWA-q + **ring F32** + 池命中 | max **266**（0.81%） | **0**（16/16） | **✅ 0/16** |
| `n-e2-D-hot` | SWA-q + **ring16** + 池命中 | **2,811–3,060**（≈8.9%） | **79–93** | **❌ 14/16** |
| `n-e3-D-cold` | ring16 + 池 1 MiB（冷算） | pre≈2900 → **post 0** | **0** | **✅ 0/16** |

★★ **最锋利的一条判据（主代理已从原始数据独立复核）**：
`n-e2-D-hot` 的原始 `probe.log` 里，replay 阶段 = 36 条记录中的第 **21–36** 条
（全部 `pre_len=4095 / used=1`，即"命中只刷 1 行"）；其中 `post_nan = 0` 的**只有两条**：
```
第 22 条 blk=2492  ⇒ replay 块内 0-based 下标 = 【1】
第 30 条 blk=2860  ⇒ replay 块内 0-based 下标 = 【9】
其余 14 条 post_nan = 2811–3060
```
而 mismatch 集合 = `[0,2,3,4,5,6,7,8,10,11,12,13,14,15]`（**恰好排除 1 与 9**）
⇒ **A（ring `nan>0`）= B（latent `nan>0`）= C（输出 sha 不符）= 14 个，三个集合逐字相同**（已核实）。
同一条对齐在 ✅ 臂上给出 `A'=B'=C'=[]`。

**机制**（两步都有实测支撑）：

```
① 命中路径只写 1/32 行（三臂 rows_changed=1/32；冷算臂 32/32）
   ⇒ 池化要读的另一行还是【上一轮的残余】
② 残余在 F32 下【有限】（latent nan=0，argmax 不翻）
   而 FP16 下 8.9% 是 NaN（【推断】F32 下超 FP16 范围的值 → inf → NaN）
③ NaN 进入池化输出 ⇒ 被 _write_compressed_source 写进 long-KV（lat nan=79–93）⇒ 翻 token
```

⇒ ★ **不是「ring 没被重建」单独致死，而是「命中只刷 1 行」×「FP16 让残余变 NaN」**。
⇒ **修法落在数值层**（ring16 写侧 clamp），**不必动调度**。

★ 顺带：`044` 还纠了 `043` 一处 —— 它的 `ring_calls=0` **不是「函数没被调」，而是探针自己的 bug**
（`sys.meta_path` 一个 finder 管多个 target + 永久摘除 ⇒ 其余钩子静默失效）。
已写进 `AGENTS.md` §5b 第 4 条（今晚第 5 个探针坑）。

### 4.1 int8 这条路**已经排除**的全部方向（避免后人重复挖）

| 方向 | 排除方式 | 结论 |
|---|---|---|
| 池字节往返丢字节 | `036`：**12,928 次** store/load **全字节**比较，`mismatch=0`；尺寸阶梯 128 B–128 KiB 全绿 | ⛔ 排除 |
| 读侧映射错行 | `038`：**1,440 次**逐行核对（真页反量化 vs 算子实际读到的 scratch 行），`true_bad=0` | ⛔ 排除 |
| 池 unit 配账 / 行复用 | `040`：**11 个计数器全 0**（含 `stale_free`/`dup_unit`/`hit_row_alias`）+ `H_kvcheck` 的**阳性对照**（人为注入 ⇒ 探针报警） | ⛔ 排除 |
| **DMA 拷贝长度**（我提的假说） | `043`：改 `page → stride` 后 **sha 逐字不变**（`6a47dd65f1ff`）、`len_changed=0`；且按字面实现会**越界**（C0 的 scale：1,024 vs 131,072 = 128×） | ⛔ 排除 |
| 池容量 / 临界倍率 | `040`：D/F 几何的池只用 **0.599 / 0.536**，无淘汰压力 | ⛔ 排除 |
| 时序 / 行易主 | `040`：4 条假说全 0，且**同一探针在 ✅ 臂上给出同样的数**（对称实验） | ⛔ 排除 |

★ **仍然站得住的候选**：`036` §3.3 的 **`kv8_ori_plane` decode 形状重建**，以及 **`state` ring 在"池命中跳过 prefill"时是否被正确重建**（`043` 的探针**装上了但没被调到**：`ring_calls=0` ⇒ 按 §5b 第 2 条**不作结论**）。
★ **三个现成资产**（后续接着查的人直接用）：
```
① l-b5-D-ring-cold（D 几何 + 池 1 MiB）✅ 0/16  ← "几何本身正确"的对照臂
   vs l-b4-D-ring-hot（同几何、同 eager、同 kv size=33,295）❌ 14/16
   ⇒ 唯一变量 = "池是否命中"
② l-b1-C0（C0 几何，池 144 MiB）✅ 0/16          ← C0 回归没坏
③ ★ 探针纪律：sys.meta_path 插 finder 必须"先摘自己再 find_spec、exec 后摘自己"
   —— 否则会【悄悄关掉所有别人的 post-import 钩子】，症状伪装成"别人的补丁坏了"
   （本次伪装成 P2OffloadingWorker 缺失，差点误判）
```

---

## 5. ★ 已知的独立质量问题（与 KV 容量**无关**，但用户会感知）

**`temperature=0` 下服务本身不确定**（`logs/037`，8 卡真权重实测）：
```
同一 prompt + 每次先 /reset_prefix_cache，连发 4 次（max_tokens=1）：
  32K prompt      ⇒ 2/4 首 token 不同
  ★ 512 短 prompt ⇒ 3/4 不同（短 prompt 也抖）
  关投机后重复请求 4/4 稳定，但 prefill 路径仍抖
  finish_reason=length（不是 EOS 口径）
```
⇒ **用户报的"服务质量下降"里有一部分来自它**，与 KV 容量无关。
⇒ **副作用**：所有基于文本比对的判据（fill/replay、cold/replay、甚至 cold/cold）**全部失效** ⇒ 判断数值正确性**必须走 KV 级字节判据**。

---

## 6. ⛔ 上线前**唯一阻塞**（只能用户在 A2 上做）

```bash
# 在 A2 的容器里（不占卡、不加载模型、约 5–10 分钟）
bash a2/scripts/a2_one_shot_probe.sh          # 或 --quick（约 2 分钟）
```
它回答四件事：`host_mem_pool` / `aclrtMallocHost` 单次 vs 总量 / `pin_memory` / ★★ **`aclrtHostRegister` 能不能注册 + 能不能被 KV 拷贝用**。

**为什么必须做**：A2 的 `host_mem_pool=0`，且这个模型在 A2 上 **Engram 206 GiB 注册曾失败** ⇒ **A3 全绿不代表 A2 全绿**。
**判读**：脚本末尾自带 `DECISION`；注册成功且拷贝逐字节一致 ⇒ 走 `NPU_OFFLOAD_HOST_MEM=registered`；否则回落 `pinned` 并重新量池子上限。

---

## 6.5 ★★ 如果 `MAX_LEN` 要开到 **1M**（当前配置是 128K）——能不能用？

### 6.5.1 先确认：**1M 就是模型的设定上限**【实测】
```json
// 模型 config.json 的 text_config
"max_position_embeddings": 1048576,
"rope_scaling": {"rope_type": "yarn", "factor": 16,
                 "original_max_position_embeddings": 65536,
                 "beta_fast": 32, "beta_slow": 1}
```
⇒ **1M = 64K 原生 × 16（YaRN）**，是**设计上限**，不是超范围外推。

★ **一条需要澄清的警告**：起服日志里会出现
```
[transformers] Unrecognized keys in `rope_parameters` for 'rope_type'='default':
{'original_max_position_embeddings', 'beta_slow', 'beta_fast', 'factor'}
```
**这是无害的**（`model.py:255-270` 逐字读的，模型自己实现 YaRN，不走 HF 的通用路径）：
```python
# V4.1 applies YaRN only to layers carrying long-context compressed KV.
# Pure SWA layers use the unscaled base RoPE even though the allocated
# lookup table still spans the configured maximum context length.
scaling_factor=config.rope_parameters["factor"],            # = 16
base=(config.compress_rope_theta if role.has_long_context   # = 160000
      else config.rope_theta),                              # = 10000
original_seq_len=(max_position_embeddings if role.has_long_context else 0),
```
⇒ ★ **YaRN 只加在"带长上下文压缩 KV 的那几层"上，纯 SWA 层用未缩放的基础 RoPE** —— 这是设计如此。

### 6.5.2 三个天花板（按"最先撞到"排序）

| # | 天花板 | 1M 下的账 | 我们的发布能否改善 |
|---|---|---|---|
| **①** | **单请求必须整个驻留 HBM** | 1M × 4,420.6 B = **4.32 GiB**；可用 HBM KV **14.40 GiB** ⇒ **最多 3 个并发 1M** | ⛔ **不能** —— 卸载层是**块粒度前缀缓存**、不是 swap（`045` §5.3：*"DRAM 层不放宽单请求最长上下文"*）。由 `GPU_UTIL` 与模型结构决定 |
| **②** | **DRAM 池的容量** | 1M = **24,064 unit**（按 `042` 实测 23.5 unit/1024token 线性外推）× **3,660,003 B/unit**（8 rank，**L1 之后**）= **一份完整 1M KV ≈ 82.0 GiB**<br>⇒ A2 余量 442 GiB 能装 **5.39 个**（留 1.2× ⇒ **4 个**）<br>⇒ **16 并发 × 1M = 1,312 GiB ⛔ 不可能** | ✅ **能，而且是"必需"**（见 6.5.3） |
| **③** | **我们的定值全都要重算** | `MAX_LEN` 131072 → **1048576**；`OFFLOAD_GB` 56 → **85**（覆盖 3 个 1M 会话：`3 × 24,064 × 1.2 = 86,630 unit`，宿主 ≈298 GiB）<br>⚠️ 还要核 `--kv-cache-memory-bytes ≥ max_model_len × kv_per_tok`（`045` §5.3 的硬要求，**不满足则 1M 起不来**） | — |

### 6.5.3 ★★ 关键：**1M 场景下两条杠杆是"必需"，不是"可选"**
```
没有 L5（per-group bpc）：每个 SWA 组也按 8 个 block 记账 ⇒ 池需求 ×4.89
没有 L1（按需分配行数）：宿主 ×1.99
两者都缺 ⇒ 1M 的池需求 ≈ 82.0 × 4.89 × 1.99 ≈ 798 GiB/会话
           ⇒ 【连 1 个会话都装不下】（A2 只有 442 GiB）
```
⇒ **现在的档 B 把 1M 从"完全不可能"推到"3~4 个并发可行"。**

### 6.5.4 ★ 真实 agent 流量下宽松得多（**最坏口径 vs 典型口径**）
上面那个"4 个"是**最坏口径**（每个请求前缀互不相同）。池只需覆盖**唯一前缀总量**：
- `042` 的 D 臂实测：**16 个请求共享同一个 128K 前缀 ⇒ 池只要 20.83 GiB**（不是 16 倍）
- 推到 1M：若 N 个会话**共享长前缀** ⇒ 池需求 ≈ **1 个前缀的 82.0 GiB**，**与 N 无关**
⇒ **"很多会话共享同一个长 system prompt / 同一份长文档"完全可行**；
⇒ **"N 个互不相干的长会话"才是瓶颈**。

### 6.5.5 需要实测才能定的三格（**都能在 A3 8 卡跑，不在 A2**）
| # | 事项 | 为什么重要 |
|---|---|---|
| **1** | **23.5 unit/1024token 在 1M 上是否仍线性** | 只在 32K/128K 反解过；尾部半满段的常数项在 1M 下占比更小 ⇒ 实际可能**优于**线性 |
| **2** | **1M 下的 HBM 实际占用** | 4.32 GiB 是按 B/token 算的，**没在 1M 上直测过** |
| **3** | **A2 上的 `--kv-cache-memory-bytes`** | 这条不满足 ⇒ **1M 直接起不来**（与 `GPU_UTIL` 无关的独立门槛） |

---

## 7. 未完项（如实列出，不掩盖）

| # | 事项 | 影响 |
|---|---|---|
| 1 | ~~**`0001` 的 `blocks_per_chunk` 局部变量泄漏**~~ → ★★ **已修、已并入 `publish/`、且端到端判据全部翻转**（`logs/039` §10 定性、`logs/043` 修复） | **修法两处成对**（收集/spec loop 各自取 `bpc_g`，裸名彻底消失）+ 两条 fail-closed 断言。**端到端【实测】**：`Σgroup_sizes` **44→72**、group 0 实搬 **64→492**、**"读到全 0 行" 5352→0**；★ **算术闭环**：`GPU→CPU` 增量 **165,437,440 = 448 block × 369,280 B**（正好补上漏搬的 7/8），而 `CPU→GPU` **逐位不变**（取回路径一个字节没动）。**单元自检 17 PASS/0 FAIL**、反例臂逐字复现 `44/[4,0,4×10]`、两条断言在反修臂上真的会炸、复跑同 sha。新 md5 `986c9115…`。⚠️ 边界：**它不是** `038` 翻转的答案（`160 MiB` 臂在同样 448 行全 0 下 sha 逐字相同）⇒ 定性为**潜伏的正确性风险已消除**，不是"首 token 错的原因"。 |
| 1b | ⛔ **8 卡链上仍是"带泄漏版"的 scheduler**（**上线阻塞项**） | 【实测】`agents/L3_8card/patched/scheduler.py` 的 md5 = **`f4de89d2c5afaaf8957bd604f478cf8e`** ⇒ **仍带 `blocks_per_chunk` 泄漏**（`042` 的 A/B 两臂用的正是它）。<br>★ **成品修复已备好**：`agents/M_bpcfix/publish/0001-offload-scheduler.patch.py.8card`（md5 **`6a4f8dffcbb1f3ab4b6c5a1d749e6eaf`** = 那份 `L3_8card/patched/scheduler.py` + 5 个 hunk，`diff -u` **只含修复**、`py_compile` OK、单元自检 17/17）<br>⇒ ★ **覆盖它就完成 8 卡上线阻塞**（8 卡口径**未复核**【未确认】）。<br>★ **对 `042` 结论的影响（已复核）**：**无** —— L1 的行数/配额由 `p2_pool` 自己算、**不读** scheduler 那个变量 ⇒ `1.9895× / 197.21 GiB` 成立；A/B 两臂同 bug ⇒ 对照有效；两臂同 bug ⇒ "L1 不改输出"那条 sha 判据有效。<br>★ **便利**：L1 的 6 个挂载件**不含 `scheduler.py`**（runner 直接指 `$L3/patched/scheduler.py` 这个**路径**）⇒ **谁把那份文件修好谁生效**，L1 包不用动。 |
| 2 | **L1 的 8 卡验证** | 决定内存是 389 GiB 还是 204 GiB（`K_l1_8card` 在跑） |
| 3 | **线 1 的 8 卡 KV 级保真** | tiny 已干净（`039`：13,392 区间 ×2 格 mismatch=0）；**8 卡仍【未确认】** |
| 4 | int8 的 `pool_replay_sha ≠ cold_replay_sha` | **真凶未定**。已系统性排除：池字节往返（`036`：12,928 次逐字节全等）、配账层（`040`：11 计数器全 0 + **阳性对照**）、**DMA 拷贝长度**（`043`：改这一行 **sha 逐字不变** `6a47dd65…`，且按字面实现会**越界**）、读侧映射（`038`：1440 次逐行核对）、池容量（`040`：池只用 54–60%）。★ **两条现成的守门员**：`l-b5-D-ring-cold`（D 几何 + 池 1 MiB）**✅** ⇒ 几何本身没问题，**只有"池命中"这一路错**。 |
| 5 | ★ **`state` ring 未被验证**（`043` 的 ring 探针 `ring_calls=0`） | 探针**装上了但没被调到**（不是"读数干净"）⇒ 按 §5b 第 2 条**无判别力**。下一条命令：改钩 `DeepseekV41Compressor.pool_projected`（`compressor.py:113`）而非 triton 模块符号。 |

---

## 8. 回滚

**一句话回滚**：所有新能力都在 `PYTHONPATH` / `docker -v` 挂载层，**去掉挂载即回现状**。
逐条回滚 = 对应 env 置 0（`OFFLOAD_SCHED_PATCH` / `OFFLOAD_NPU_WORKER_PATCH` / `P2_POOL_PATCH` / `PGP_MGR_HARDEN`）。
**没有一处写镜像。**

---

## 9. 证据索引（都可复核）

| 主题 | 日志 |
|---|---|
| 8 卡终验（档 A 的全部数字） | `logs/027` |
| 池结构性闲置实测 1.96×（L1） | `logs/030` |
| per-group bpc（L5 的 4.89×） | `logs/021` |
| 记账单位诚实化（L2） | `logs/032` |
| 线 1 的 KV 级逐字节判据 | `logs/039` |
| 池 unit 行级 provenance（`038` 机制被推翻） | `logs/040` |
| 池分配器加固 + 可观测点 | `logs/041` |
| 非确定性诊断（与容量无关） | `logs/037` |
| int8 的容量与性能（本包未含，供后续） | `logs/028`/`033`/`034`/`035`/`036`/`038` |
