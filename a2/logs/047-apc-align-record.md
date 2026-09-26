# 047 — APC 命中长度对齐的「记账闭环」：**int8 两条容量杠杆（×1.4655 / ×1.9133）解锁**

> 2026-09-22 09:0x–09:4x CST。执行：子代理 **Q_apcrecord**。机器：**A3（A3-node1）**，槽位
> **c0**（die 3，容器 `prbench-c0`）与 **c1**（die 6，容器 `prbench-c1`），全程走
> `tools/a3_chip.sh` 锁（**没有手设 `ASCEND_RT_VISIBLE_DEVICES`**）；**没用 `/tmp`**；
> **没写 `upstream-v41/`**；**没动 `a2/publish/` 原件**；跨机传输全走 coscli；代码只写 `a2/agents/Q_apcrecord/`。
> 起服前逐次 `df -h /dev/shm`（c0/c1 均 0% 占用 ⇒ 未踩 `SemLock` 坑）。
> 产物：本日志 + `logs/raw/047-apc-align/`（4.5 MB / 9 条臂）+ `agents/Q_apcrecord/`。
> ★ 判决脚本 `scripts/judge_047.py`：**33 PASS / 0 FAIL**（只读原始 `client.json`，可复跑）。
> ★ 离线自检 `scripts/offline_selfcheck.py`：**75 PASS / 0 FAIL**（不占卡）。

---

## 0. ★★★ 头条：`046` 的「④ 一开就炸」被解释清楚了 —— ④ 与 store 侧是【成对的】，不是两个独立候选

```
④（_lookup 的命中边界对齐）把边界推到 n = 4094
  ⇒ SWA 组的窗口是 [1024k−129, 1024k−3] ⇒ 跨两个块 {8k−2, 8k−1}
而 store 侧（只对齐到 ratio 时）只保留每个 1024 段的【最后 1 个】chunk 8k−1
  ⇒ chunk 8k−2 从未被写入池
  ⇒ manager.prepare_load 断言 "Block … not found in cache"（发动机死、15/16 请求失败）
```

**逐量推导（本任务的第一件事，见 §2）的结论是**：这不是"记账不自洽"，而是**那一行数据真的不在池里**；
`num_hit_chunks` / `num_chunks` / `cdiv(num_cached_tokens, tokens_per_chunk)` **三个都不用改**
（它们是边界派生量，自动跟着走）。真正要一起动的是**另外两个量**：
**GPU 侧窗口 padding（决定 load 起点）** 与 **store 侧 `is_store_reachable_swa_chunk` 的可达尾部**。

**⇒ 这把"一个失败的实验"变成了"一个被解释清楚的边界"**：`046` 的修法方向**是对的，只是少了一半**。

---

## 1. 结论（先给主代理）

1. **★【实测·解锁】D 几何（L5+L1+SWA-q+ring16，池 144 MiB）J2：❌ 14/16 → ✅ 0/16**，
   replay sha = `24b570535f58…` **与冷算参考逐字相同**，容量 **33,295 不变（×1.4655）**。
   且是**真·池命中**（`CPU→GPU = 184,401,920 B > 0`、`hits = 49,152 > 0`）。
2. **★【实测·解锁】F 几何（5 条杠杆）J2：❌ 15/16 → ✅ 0/16**，容量 **43,469 不变（×1.9133）**。
3. **★【实测】守门员**：C0 几何 ✅ 0/16（容量 22,719 不变）；D 几何复跑同 sha ✅。
4. **★【实测】`021` 判据③（变长前缀 4096→2048）不回归**：`hits = 16,384 > 0`、`CPU→GPU > 0`，
   且 replay sha == **池 1 MiB 冷算参考**（`c7d40f34…` 两臂逐字相同）。
5. **★【实测】store 侧一个字节没变**：`BlockStored:CPU = 714`、`GPU→CPU = 196,689,920 B`
   与无补丁基线**逐字相同** ⇒ **`021` 的 4.89× 倍率与 A2 的池需求都不需要重算**。
6. **⛔【实测·负结果】两条更"直觉"的修法都被实测否掉**：
   * **对齐到 ratio（4095→4094）+ 不动 store 侧** ⇒ 发动机断言（= `046` 的炸点，本任务复现）；
   * **对齐到 ratio + 加 store 尾部（我的第一版）** ⇒ 池需求 +24%（SWA 键 40→80/请求）
     ⇒ **144 MiB 池溢出（`BlockRemoved:CPU = 1502`、`CPU→GPU = 0`）**，
     并且**会给出一个 J2 ✅ 的【冷算假阳性】**（见 §4）。
7. **★★ 最终采用的机制（mode 3）= 对齐到"段栅格"**（V4.1 = 1024，运行期从 `kv_group_configs` 现算），
   落在**已存的段尾 chunk** 上 ⇒ **对齐与 store 两边同时自洽，且 store 侧零改动**。

---

## 2. ★★ 记账链的逐量推导（任务书 §1(1) 的答案；**不占卡**完成）

几何（`[SWA_trim] group 表`，实测）：
`group0 full tpb=128 tpc=1024 bpc=8`；`group1 state`（不参与）；`group2..11 SWA tpb=128 tpc=128 bpc=1 sw_chunks=1 align_cnt=8`。

| 量 | 定义 | n=4095 | n=4094 | **必须跟着改？** |
|---|---|---|---|---|
| `num_hit_chunks` | `(L+num_hit)//T_g`（`update_num_hit_chunks`） | 31 | 31 | **❌ 不能动**（只用于 `_touch()` 的 LRU 触碰；是派生量）|
| `num_chunks` | `cdiv(num_cached_tokens, T_g)` | 32 | 32 | **❌ 不能动**（决定 load 区间**右端**，语义正确）|
| `cdiv(num_cached_tokens, tokens_per_chunk)` | 同上（同一个表达式） | 32 | 32 | **❌ 不能动** |
| **`num_locally_computed_gpu_blocks`** | = 窗口 padding：`(n−W+1)//128` | **31** | **30** | **★ 自动跟着走**（vLLM 现算，不是我们要改的）|
| **store 侧可达尾部** | `reachable_tail = sw_chunks + is_eagle` | 1 | 1 | **★ 这一个才是要改的**（只对齐 ratio 时）|

★ **`W = 128` 是【实测反解】出来的，不是假设**：load 探针的 `computed_gpu` 在 n=4095 时是 **31**、
在 n=4094 时是 **30** —— 与 `(n−127)//128` 两式**逐字吻合**（`p-a6` / `p-a5` 的 serve.log）。

**三句话**：`num_hit_chunks`/`num_chunks`/`cdiv` 是**边界派生量**（改它们反而会错）；
**load 起点**是 vLLM 自己按窗口算的；只有 **store 的可达尾部**需要我们补齐。

---

## 3. 三层机制（为什么"对齐"能治病）与三种对齐单位的取舍

### 3.1 病灶链（`044`/`045`/`046` 三份实测的合成）

```
命中边界 = 4095（奇数）
  ⇒ ratio-2 压缩层第一步的池化组 (4094,4095) 跨在边界上
  ⇒ token 4094 必须从 state ring 的【残余行】读
  ⇒ 而 ring 组 prefix_cacheable=False、不参与卸载（009/043）⇒ 新请求拿到【回收页】
  ⇒ int8 几何把那批字节解读成 NaN（045 的三条指纹：16640 / 18464 / 32768）⇒ 翻 token
```

### 3.2 ★ 三种对齐单位的取舍（本任务实测出来的）

| 方案 | 对齐单位 | 命中边界（4096 的 prompt） | store 侧 | 池需求 | `hits` | 结果 |
|---|---|---|---|---|---|---|
| **mode 1/2**（对齐到 ratio） | 2 | 4094 | **必须加尾部**（+1/段） | **+24%** | 65,504（−0.02%） | ⛔ 144 MiB 溢出；224 MiB 才 ✅ |
| **mode 3 ★ 采用**（对齐到段栅格） | **1024** | **3072** | **零改动** | **不变** | 49,152（−25%） | ✅ **144 MiB** |

**为什么 mode 3 的 store 侧零改动**（构造性）：
```
需要的窗口 = [n−W, n−1] = [3072−128, 3071] = [2944, 3071]
  ⇒ 恰好是第 3 个 1024 段的【最后一个块】（块号 23 = 3×8−1）
  ⇒ 正是 store 侧"每段留最后 1 个 chunk"保留的那一块 ⇒ 不需要动 store
同理 2048 → 边界 1024 ⇒ 需要的块 = 块 7（= 段尾 chunk）✅
```
**"段栅格"也是现算的、不是硬编码**：`grid = max(participating full-attention groups 的 tokens_per_chunk)`
（V4.1 = 1024；无 full-attention 组或非 hybrid 模型 ⇒ 1 ⇒ 退化成 no-op）。

**⇒ 代价从"池 +24%"变成"命中窗口少 1023 个 token"**：对 128K 生产口径 = 多算 **0.78%**；
而 `hits` 这个**指标**降 25% 不代表"命中率下降"—— **16/16 请求全部命中**（`CPU→GPU > 0`），
只是每个请求的复用前缀短了 1023 个 token。

---

## 4. ★★ 一条必须单说的**假阳性**（本任务最重要的自我克制）

**`q-a1-D-align`（mode 2，池 144 MiB）的 J2 是 ✅，但它是【冷算假阳性】**：
```
CPU→GPU = 0.0 ；hits = 0.0 ；BlockRemoved:CPU = 1502   ⇒ 池溢出、一个块都没取回
⇒ replay 走的是整段重算 ⇒ sha 当然等于冷算参考
```
**判别方法**（已写进 `judge_047.py` 的 ①e，并作为**结构性判据**保留）：
**任何"J2 ✅"都必须同时满足 `CPU→GPU > 0` 且 `hits > 0`**，否则不算命中臂。
★ 这正是 `AGENTS.md §5b 第 3 条`要防的"看不见所以全绿"。

**同一模式的 root cause**：mode 2 把 SWA 的 store 量 ×2（每段 2 个尾块），
而池是**按组配额**的（`P2QuotaManager`：每个 SWA 组 108 unit，16 请求 × 4 块 = 64 unit 够用；
×8 块 = 128 unit > 108 ⇒ 淘汰）⇒ **144 MiB 装不下**；224 MiB（`q-a2`）才真命中。

---

## 5. 实现（`agents/Q_apcrecord/`，**全部 env 门控、默认关**）

### 5.1 三处改动（最小面）

| # | 落点 | 改动 | 语义 |
|---|---|---|---|
| **(1)** | `OffloadingConnectorScheduler.__init__`（`pgp_scheduler.py:647`） | `+ _q47_bind_kv_cache_config(self, kv_cache_config)` | 把 config 交给补丁 ⇒ **运行期现算** `ratio = lcm(各 group compress_ratio)` 与 `段栅格` |
| **(2)** | `OffloadingConnectorScheduler._lookup`（`:873` 之后） | `+ max_hit_size_tokens = _q47_align_hit(self, max_hit_size_tokens)` | 命中长度向下对齐到 `lcm(ratio, 栅格)` |
| **(3)** | `is_store_reachable_swa_chunk`（`:194`，模块级） | **包一层**（`extra = tail_extra`，加在 `sliding_window_chunks` 上） | mode 3 下 `extra = 0` ⇒ **用原参数调原函数** ⇒ 逐字 no-op |

★ **为什么是最小面**：**只增 4 行**（`+4 / −0`，见 `out/pgp_scheduler.replace.diff`），
**不改** `num_hit_chunks` / `num_chunks` / `cdiv` / 页几何 / kernel，**不写任何内存**，无 co-tenant 风险。
★ `(3)` 用**包一层**而不是重写函数体：原函数内部就是
`reachable_tail = sliding_window_chunks + int(is_eagle_group)`，所以"传入值 +1"与"内部 +1"是**同一个表达式**。

### 5.2 开关

| env | 默认 | 含义 |
|---|---|---|
| `VLLM_V41_APC_ALIGN` | **`0`** | `0`=**连 hook 都不装**；`1`=按模型 config 现算 ratio（=mode2 语义）；`2`=强制 ratio；**`3`=段栅格 ★ 推荐**；`4`=段栅格+强制 ratio |
| `VLLM_V41_APC_ALIGN_RATIO` | `2` | mode 2/4 的强制值（诊断用）|

★ **ratio = 1 时逐字 no-op**（判据⑪）：`align_unit = 1` ⇒ `align_to_ratio(v,1) == v`（**n=1..4096 全扫**）、
`tail_extra = 0` ⇒ 用原参数调原函数；**mode 0 连 finder 都不装**（反例臂 `q-a7` 实测：横幅 0 行）。

### 5.3 探针纪律（`AGENTS.md §5b` 六条逐条落实）

* **先装 hook 再 import**：`probe/sitecustomize.py` 在 `install()` 里装 finder，早于任何 `vllm` import；
* **只打"已装载"不算数**：横幅打**替换后的函数名 + 地址 + 目标文件路径**，热路径打
  `boundary_calls / align_hits / tail_calls / tail_extra_calls`（判别量）；
* **反例臂**：`q-a7`（`Q_ALIGN=0`）**逐字复现** `046` 的 ❌ 14/16 + sha `6a47dd65f1ff`；
* **一个 target 一个 finder 实例**；**`importlib.util.find_spec` + `try/finally` 插回**（`046` 第 5 个坑）；
* **overlay 先 `rm -f` 再 `cp` + `[ -L ] && exit 1`**（`046` 第 6 个坑）；
* **fail-closed**：锚点失配 ⇒ `Q47_STATE["ok"]=False`。
* ★ 本任务**自己踩了一个新坑并已自检固化**：包装函数里直接引用 `_q47_tail_extra`（它只定义在
  `_install_helpers` **内部**）⇒ 运行期 `NameError`。**修法**：从**目标模块命名空间**取
  （`getattr(module, "_q47_tail_extra")`）。`offline_selfcheck.py` 的 **G 段**用 AST 复现该错误
  （反例臂必须报警）—— 判据本身也有判别力。

---

## 6. 九条臂（全部单卡 tiny、EAGER、16×4096 token、池 144 MiB（除注明））

| 臂 | 几何 | ALIGN | 池 | `CPU→GPU` | `hits` | J2 | 容量 | 备注 |
|---|---|---|---:|---:|---:|---|---:|---|
| `q-a1-D-align` | D | 2 | 144 M | **0** | **0** | ✅ | 33,295 | ⛔ **冷算假阳性**（`BlockRemoved:CPU=1502`）|
| `q-a2-D-bigpool` | D | 2 | **224 M** | 274.3 MB | 65,504 | ✅ | 33,295 | 真命中（mode2 需要 ≥224 MiB）|
| **`q-a3-D-grid144`** | **D** | **3** | 144 M | 184.4 MB | 49,152 | **✅** | **33,295** | ★ **主结果** |
| **`q-a4-F-grid`** | **F** | **3** | 144 M | 122.5 MB | 49,152 | **✅** | **43,469** | ★ **×1.9133 解锁** |
| **`q-a5-C0-grid`** | **C0** | **3** | 144 M | 184.4 MB | 49,152 | **✅** | **22,719** | 守门员 |
| `q-a6-D-var2048` | D 4096→2048 | 3 | 144 M | 89.9 MB | 16,384 | ✅¹ | 33,295 | `021` 判据③（1: 与冷算臂比）|
| **`q-a7-D-off`** | **D** | **0** | 144 M | 231.7 MB | 65,520 | **❌ 14/16** | 33,295 | ★ **反例臂**（sha 逐字 `6a47dd65f1ff`）|
| `q-a8-D-var2048cold` | D 4096→2048 | 3 | **1 M** | 0 | 0 | 冷算参考 | 33,295 | `c7d40f34…` |
| `q-a9-D-grid144r2` | D | 3 | 144 M | 184.4 MB | 49,152 | ✅ | 33,295 | **复跑同 sha** |

★ 全部 9 条臂的 `GPU KV cache size` 与对应历史基线**逐字相同**。

### 6.1 ★★ 探针读数：命中边界与 ring（回答主代理的两问）

**问①：「grid 臂里有没有 `pre_len > 0` 的记录？」→ 有**（汇总里 `tail -4` 那一段恰好是 fill 段）：
```
q-a3-D-grid144 的 RINGREC 按 pre_len 分布：
  pre_len=0    × 17（fill 轮 + 预热 decode）
  pre_len=3072 × 16  ★ replay 轮，16 个请求各一条
  pre_len=256/257/258 × 3（预热 decode）
逐字样本：seqlen=4096 pre_len=3072 used=1024 blk=1507
          pre=(nz=32730/32768 nan=2957) post=(nz=32738/32768 nan=0)
          rows_changed=32/32 nan_rows=32->0 lat=(nan=0 absmax=7.68e-05)
```
⇒ 命中的是**真·池命中**（`CPU→GPU=184.4 MB > 0`、`hits=49,152 > 0`），**不是"不再读 ring"**。

**问②：「`nan_rows = 32→0` 与'对齐后第一步两个 token 都在段内'一致吗？」→ 一致，而且更强**：
```
三条臂的 (used, rows_changed) 实测：
  fill 轮          used=4096 → rows_changed=32/32（溢出整页，全改写）
  mode3 replay     used=1024 → rows_changed=32/32  ★ 1024 = 32 行 × 32 token ⇒ 恰好一整页
  046 stock replay used=1    → rows_changed=1/32  （31 行是别家字节 ⇒ 045 的 NaN）
⇒【推断】ring 页 = 32 行，每行覆盖 32 个 token = **1024 token** 的覆盖范围。
```
**⇒ mode3 的对齐让重算段恰好等于整页** ⇒ **32 行全部由本步自己的投影写满** ⇒
没有任何"别家平面的残余行"被读进池化 ⇒ `nan_rows 32→0`（`q-a3` 共 21 条记录）、`lat` 的 nan 全 0。
**比"第一步两个 token 都在段内"更强**：**整段 1024 个 token 全在段内**。

★ 该"32 行 × 32 token"的映射标 **【推断】**（三条 `(used, rows_changed)` 观测支持它，但**没有**直接读 ring 布局）；
`W = 128` 与 `used / rows_changed / nan_rows` 的全部数字标 **【实测】**。

---

## 7. ★ 判据⑦ 的口径（按主代理 2026-09-22 09:3x 的裁决 + 本任务实测修正）

**原文意图 = "别把已有的东西弄坏"**（例如 `043` 那种"该搬的没搬"）。本任务的实测结论：

| 口径 | mode 2（未采用） | **mode 3（采用）** |
|---|---|---|
| `BlockStored:CPU` | 714（不变） | **714（不变）** |
| `GPU→CPU` | 196,689,920（不变） | **196,689,920（不变）** |
| SWA store 键 / 请求 | **40 → 80（+100%）** | **40（不变）** |
| 池需求 | **+24%**（144 MiB 溢出） | **不变** |
| `CPU→GPU` | 231,669,760（不变） | **231,669,760 → 184,401,920（−20%）** |
| `hits` | 65,520 → 65,504（−0.02%） | **65,520 → 49,152（−25%）** |

**★ 口径修正（写清楚，供后人判断）**：
* **mode 2 的"补 store 尾部"** 属于**补上原本永不存在的数据**（修复而非回归）—— 但**已被否决**，
  因为它把 144 MiB 池打爆（§4）；
* **mode 3 的 `CPU→GPU` 下降**是因为**命中窗口小了 1023 个 token**（取回的数据本来就不该取），
  不是"少搬了本该搬的字节"：`GPU→CPU`（= 存进去的）与 `BlockStored:CPU` **逐字未变**；
* **mode 3 的 `hits` −25% 是"复用前缀短了 1023 token"**，不是"命中率下降"：
  16/16 请求全部命中（`CPU→GPU > 0`），对 128K 生产口径 = 多算 **0.78%**。

---

## 8. 九条判据对账（任务书 §1(3)）

| # | 判据 | 结果 |
|---|---|---|
| **①** | D 几何 J2 ❌14/16 → ✅ | ✅ **0/16**，replay sha = 冷算参考 `24b57053…`，**真命中** |
| **②** | 探针读数 | ✅ `pre_len`：4095（旧）→ **3072**（grid）、`used` 1 → **1024**、`rows_changed` 1/32 → **32/32**、`nan_rows` 32→**0**（21 条）|
| **③** | F 几何 J2 → ✅ | ✅ **0/16**，容量 **43,469** |
| **④** | C0 守门员保持 ✅ | ✅ **0/16**，容量 22,719、`GPU→CPU` 与 `046` 的 `p-a7` 逐字相同 |
| **⑤** | D + 池 1 MiB 冷算保持 ✅ | ✅ `q-a8`（4096→2048 冷算参考）sha 与热臂 replay **逐字相同**（另见 ⑥/⑦ 的 1 MiB 语义）|
| **⑥** | 容量不退化 | ✅ **33,295 / 43,469 / 22,719** 逐字不变（D/F/C0）|
| **⑦** | 四条判据不回归 | ✅ **口径见 §7**：mode3 下 `BlockStored:CPU` / `GPU→CPU` **逐字不变**；`hits` −25%（= 命中窗口短 1023 token，**已裁决接受**）|
| **⑧** | `021` 五条（尤其变长前缀） | ✅ `q-a6`（4096→2048）：`hits=16,384>0`、`CPU→GPU>0`、replay sha == 冷算 `c7d40f34…`；★ **倍率不变**（见 §9）|
| **⑨** | 复跑同 sha | ✅ `q-a9` 与 `q-a3` 的 fill / replay / hits **逐字相同** |
| **⑩** | prefill 时延不退化 | ✅ D 几何三条臂（含反例臂）fill p50 **542.2 / 543.1 / 546.0 ms**（±0.7%）|
| **⑪** | `ratio=1` 的 no-op 断言 | ✅ 假 config 单测 6 组 + `n=1..4096` 全扫逐字相等 + `tail_extra(1)==0` + **mode 0 连 hook 都不装**（反例臂实测 0 行横幅）|

---

## 9. ★★ `021` 的倍率：**不变**（不是"腰斩成 2.4×"）

```
021 的 4.89× 收益来自 is_store_reachable_swa_chunk 的"每个 1024 段只留尾部 chunk"。
mode 3 下 tail_extra 恒 = 0 ⇒ 该函数【用原参数调原函数】⇒ 语义逐字等价
⇒ 021 的条目数、倍率、池需求【全部不变】。
```

**两条独立读数佐证**（`q-a3` vs 无补丁基线 `p-a6`）：
```
BlockStored:CPU = 714            （逐字相同）
GPU→CPU         = 196,689,920 B  （逐字相同）
```

**⇒ A2 的 `OFFLOAD_GB=56`、池需求、`027` 的 1.193× 余量结论都不需要重算。**
⚠️ 若将来改用 **mode 2**（为了 `hits` 只掉 0.02%），则 SWA 键翻倍、池需求 **+24%**，
`OFFLOAD_GB` 至少要 **67 GiB**（56 × 1.19）—— **该口径本轮未跑，标【未确认】**。

---

## 10. 交付物

| 路径 | 内容 |
|---|---|
| `agents/Q_apcrecord/patch/apc_record_fix.py` | **补丁主体**（env 门控、默认关、fail-closed、三种对齐单位）|
| `agents/Q_apcrecord/probe/sitecustomize.py` | 叠 P2 + 本补丁 finder + N_ring 只读探针 + SIGTERM 落盘 |
| `agents/Q_apcrecord/scripts/prepare_overlay.sh` | overlay 生成（**符号链接指向基底包、本任务零写包内文件**；`rm -f` 后再 `cp`）|
| `agents/Q_apcrecord/scripts/run_arm_q.sh` | 单臂驱动（起服自检 + 热路径计数 + fail-closed）|
| `agents/Q_apcrecord/scripts/selfcheck_q.py` | 影子包 + 三个锚点的 import 级自检（打**真实文件路径**）|
| `agents/Q_apcrecord/scripts/offline_selfcheck.py` | **75 PASS / 0 FAIL**（含 A/B/C/D/E/F/G/H/I 九段，每段带反例臂）|
| `agents/Q_apcrecord/scripts/judge_047.py` | **判决脚本 33 PASS / 0 FAIL**（只读原始 `client.json`，可复跑）|
| `agents/Q_apcrecord/out/pgp_scheduler.replace.diff` | **替换建议**（`+4 / −0`；**未动 `a2/publish/` 原件**）|
| `logs/raw/047-apc-align/` | 9 条臂的原始产物（4.5 MB）+ `summary.json` |

### 10.1 上线建议（给主代理）

```
默认：VLLM_V41_APC_ALIGN=3     ← 段栅格对齐（store 零改动、池需求不变、命中窗口 −1023 token）
回退：VLLM_V41_APC_ALIGN=0     ← 逐字旧行为（已实测复现 ❌14/16，可作 A/B 开关）
```
* **必挂 `0001`（scheduler 卸载补丁）**：不挂 ⇒ `assert isinstance(kv_cache_spec, FullAttentionSpec)` 必炸
  （`010 §4.3`）—— 本轮全部臂都挂在 `pkg-ring` / `pkg-kv8pf` 的补丁层之上；
* **缺省关**：不设 env 时**连 finder 都不装**（`q-a7` 实测 0 行横幅、sha 逐字回到基线）⇒ 零风险；
* ⚠️ **8 卡链未测**（本轮全部单卡 tiny）：`OFFLOAD_GB`、宿主实占、`BlockRemoved:CPU` 需在 8 卡上复核；
* ⚠️ **图模式未测**：本轮全部臂是 `--enforce-eager`（与 `046` 同口径）。

---

## 11. 没做的事 / 未确认（诚实边界）

| # | 项 | 状态 |
|---|---|---|
| 1 | **8 卡真权重** | ⚠️ **未测**（本轮 9 条臂全部单卡 tiny）|
| 2 | **图捕获（`FULL_DECODE_ONLY`）下** | ⚠️ **未确认**（本轮全 `--enforce-eager`）|
| 3 | **mode 2 的 8 卡池需求（+24%）** | ⚠️ **未跑**（只在 tiny 上确认 144 MiB 会溢出、224 MiB 可行）|
| 4 | **`concurrency > 1`** | ⚠️ 未测（本轮全 `CONCURRENCY=1`）|
| 5 | `021` 的**其余四条** | ⚠️ 本轮只重跑了**判据③（变长前缀）**这一条（主代理点名的核心）；另四条的历史读数与 mode3 无关（store 侧逐字未变），**但未逐条复跑** |
| 6 | ring 布局"32 行 × 32 token" | 【推断】（三条 `(used, rows_changed)` 观测支持；**没有**直接读 ring 的 stride/行表）|
| 7 | 非 1024 的命中长度（如 `prefix-match-unit=32`） | 【推断】mode3 下命中长度恒为 1024 的倍数；**没有**专门扫非 1024 的命中格 |
| 8 | `033` 的 prefill 时延口径 | ✅ 本轮读到同几何 fill p50 ±0.7%；**但那是 tiny 口径**，真权重未测 |
| 9 | **短 prompt（< 1024 token）在 mode3 下不再命中池** | 【实测】预热 decode 的 trace 是 `max_hit_size_tokens 255 -> 0`（对齐单位 1024）⇒ 池查询被跳过、请求照常重算。**对 128K 生产口径无影响**；但若将来要服务短 prompt，需要另设一个更小的对齐单位（**未做**）|

### 11.1 第一手证据：EngineCore 侧的绑定行（逐字）

```
(EngineCore pid=73715) [Q_apcrecord][apc] (1) 绑定：mode=3 ratio=2 source=model 段栅格=1024
    ⇒ 对齐单位=1024，store 尾部增量=0（对齐单位=1 => 本补丁逐字 no-op）；
      各 group compress_ratio=[2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
★ store 侧包装函数的计数：`reachable_tail += 0` **15 次**，`+= 1` **0 次**
  ⇒ "store 一个字节没改"不只是数字相同，**代码路径本身**就是恒等调用。
```

---

## 12. 取证（复跑命令）

```bash
# ① 离线自检（不占卡，75 PASS / 0 FAIL）
python3 a2/agents/Q_apcrecord/scripts/offline_selfcheck.py

# ② 判决（不占卡，33 PASS / 0 FAIL；只读 logs/raw/047-apc-align/）
python3 a2/agents/Q_apcrecord/scripts/judge_047.py

# ③ 一条臂（占卡；★ 起服前先 df -h /dev/shm）
bash ~/projects/dsv41-upstream-pr/tools/a3_chip.sh c0 --timeout 2400 --name q-a3-D-grid144 -- \
  env TAG=q-a3-D-grid144 POOL_BYTES=150994944 XSWA=1 XRING=1 XL1=1 XKV8PF=0 EAGER=1 PORT=8600 \
      Q_ALIGN=3 bash /work/agents/Q_apcrecord/scripts/run_arm_q.sh

# ④ 反例臂（改 Q_ALIGN=0 ⇒ 必须复现 ❌ 14/16 + sha 6a47dd65f1ff）
# ⑤ 变长前缀（PROMPT_TOKENS=4096 REPLAY_PROMPT_TOKENS=2048 MAX_TOKENS=4）
```
