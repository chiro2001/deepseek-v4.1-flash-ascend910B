# 065 — ★★★ **A2 实机探测：唯一阻塞解除**（`aclrtHostRegister` 路线在 A2 上可用）

> 2026-09-22 16:10 CST。**执行：用户**（在 A2 宿主 `~/projects/dsv41-a2-repro-kv8-offloading/` 下跑）。
> 脚本：`a2/scripts/a2_one_shot_probe.sh`。
> ★ **三次运行用了三个脚本版本**（记清，否则会把"没测"读成"失败"）：
> * 16:10 `LIGHT=1` ⇒ `f88672b317521ec9fcd61c2c5faef0282ea9f6b6d37201bac7214689d2327740`（**初版**）
> * 16:15 与 16:24 两次 `LIGHT=0` ⇒ `a4113ff4e2cc6510572be77835c984f50b5422150e13518251482b870bc2b443`
>   （只修了 §3 的"env 没转发"；**§3b.0 那个"注册段被整段跳过"的 bug 还在**）
> * ★ **当前（含 §3 + §3b.0 两个修复）= `b42d74a7d905f48fd32b587d9f1f327102f28a674490993da1775c58cb342455`**
>   ⇒ **补跑大档必须用这个**（已按 §3b.0 的判据在 A3 上真机验证过）
> 口径：`A2_CONTAINER=dsv41-a2 A2PROBE_FLOOR_GIB=300 LIGHT=1`。
> 标记：【实测】/【推断】/【未确认】。

---

## 0. 一句话

**★★★★★ 判据全过，阻塞彻底解除** —— **`1 / 8 / 32 / 64 GiB` 四档注册【全部 True】**，
每一档都做了**真实 H2D→D2H 逐字节对账**（不是 H2H）：
```
[a2probe] 注册（匿名/普通内存）：1=True 8=True 32=True 64=True
[a2probe] 注册（文件映射，仅探到 8 GiB）：1=True 8=True
[a2probe] 判读：⇒ 候选 β 可行：用 mmap + aclrtHostRegister(MAPPED) 做池子，把 cpu_npu.py 的 pin_memory 换掉
```
⇒ **候选 β（普通 host 内存 + `aclrtHostRegister(MAPPED)`）在 A2 上成立**。
★ 池需要 **~49 GiB/worker**（`logs/027`：`OFFLOAD_GB=56`）⇒ **64 GiB 单块可注册 ⇒ 池可整体注册，不必分片**。
⇒ **唯一阻塞上线的那一步，到此完成。**

> **本次经过三次运行**才拿到干净读数（前两次都是**脚本自身的静默失败**，见 §3 / §3b.0）：
> | # | 时间 | 版本 | 结果 |
> |---|---|---|---|
> | 1 | 16:10 | `f88672b3` | `LIGHT=1`，只探到 4 GiB ⇒ 判据过、但**上限未知** |
> | 2 | 16:15 / 16:24 | `a4113ff4` | ⛔ `LIGHT=0` **没生效** / **注册段被整段跳过** |
> | **3** | **16:32** | **`b42d74a7`** | ★★★ **`1/8/32/64` 全 True —— 本文件的主体** |

---

## 1. ★★ 设备环境（**这三条把 `logs/014`/`052` 的推断升成实测**）

```
/proc/svm/dev0/feature/  host_mem_pool = 0      ← ★ A2 真的没有 host 内存池（与 A3 的 1 相反）
                          host_pin_pre_register = 0
                          mem_host_uva = 1       ← ★ UVA 可用
                          dev_mem_map_host = 1   ← ★ 设备能映射 host 内存
内核 5.10.0-216.oe2203sp4.aarch64（OpenEuler 22.03 SP4）
  ★ 内核版本在这里**只写到 216**：完整的补丁段含 `x.y.z.w` 形式的数字，
    会被发布仓的 IP 泄漏扫描误报成内网地址（`prepare_publish.sh` 的 `PATTERNS` 之外，
    但我手动跑的那条 `grep -E '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+'` 会命中）⇒ 刻意省略，信息量不减。
Mem: total 754 GiB / used 315 / available 438 / buff-cache 260      ← ★ 服务在跑时的余量
```
★ **`host_mem_pool=0` 是本次最关键的确认**：这正是 `logs/012` 说 `cpu_bytes_to_use` 撑不大、
`logs/014` 说"别用 `aclrtMallocHost` 撑池"的那条根因。**A2 与 A3 在这条上确实相反**（A3 = 1）。

---

## 2. 四组读数（全部【实测】）

### 2.1 pinned（`aclrtMallocHost` 路线 / 候选 α）
```
单次 1 GiB ✓ (0.1s)   4 GiB ✓ (0.2s)   8 GiB ✓ (0.8s)
累加 64 × 256 MiB = 16 GiB ✓ ；128 × 256 MiB = 32 GiB ✓
```
★ **8 GiB 单次成功** —— 这与 `logs/012` 那条"64 GiB 单次 FAIL / 单次上限 ∈ (4,8] GiB"**不同**：
本次 8 GiB 单次直接过 ⇒ **那条"单次上限"结论进一步弱化**（`014` 已经在干净进程里推翻过一次，这是**第二次**）。

### 2.2 ★★★ 注册（候选 β 的生死判据）
```
✓ 注册 1 GiB  ret=0  dev=7695505555456  (0.4s)  is_pinned=True
  ★ 注册内存的设备往返 256 MiB：H2D 5.8 GB/s / D2H 19.9 GB/s / 往返逐字节一致=True
  ★ 注册内存的设备往返判据 = True
✓ 注册 4 GiB  ret=0  dev=7692284329984  (1.5s)  is_pinned=True
  ★ 注册内存的设备往返 256 MiB：H2D 20.4 GB/s / D2H 20.0 GB/s / 往返逐字节一致=True
  ★ 注册内存的设备往返判据 = True
```
★ **注意 `is_pinned=True`** —— 注册**普通**（非 pinned）内存之后，这块内存在 A2 上**被真的锁住了**
（这正是"注册路线能不能替 pinned"的关键：`logs/014` 在 A3 上量到 H2D 58 / D2H 42.7 GB/s，
本次 A2 是 **20.4 / 20.0 GB/s** —— 慢但可用，且**是被 `W922 ... 32 padding size` 那条驱动警告标过的**）。

### 2.3 文件映射注册（Engram 同款）
```
✓ 1 GiB 文件映射注册 ret=0 (1.1s)
  ★ 设备往返 256 MiB：H2D 4.1 GB/s / D2H 20.0 GB/s / 逐字节一致=True
```
★ 也通。但**生产池不要走文件映射**（`refs/40` §0：文件映射会回写磁盘）—— 这只是对照。

### 2.4 pageable（基线对照）
```
★ 设备往返 256 MiB：H2D 5.5 GB/s / D2H 19.6 GB/s / 逐字节一致=True
```
★ `pageable` 的往返也**逐字节一致**，但 **H2D 只有 5.5 GB/s**（注册后 20.4）⇒ ★ **注册路线在 A2 上确实带来 3.7× 的 H2D 提升**。

---

## 3. ★★★ 一次**静默失败**：`LIGHT=0` 根本没生效（已修，教训记这里）

16:15 用户按指引跑了 `LIGHT=0`，**输出与 `LIGHT=1` 逐字相同**（仍然只探 1/4 GiB），
而 DECISION 还在提示"重跑 `LIGHT=0` 再定池子上限" ⇒ ★ **会让用户无限重跑、永远拿不到大档**。

**根因**【实测】：
```bash
$DOCKER exec -i "$A2_CONTAINER" bash -lc '...'     # ★ docker exec 不继承宿主环境变量！
```
⇒ 容器里 `LIGHT` 回落默认 `"1"`、`A2PROBE_FLOOR_GIB` 回落 `"120"`（**用户给的 300 也被丢掉**）。
`COPY_GIB` 恰好默认同值，所以只有前两个暴露。

**为什么这是本轮最值得记的一条**：它与 8 卡 runner 上反复出现的
**"env 白名单不转发"**（`VLLM_V41_DRAFT_BLOCK` / `R8_SLOT_TRACE` / `VLLM_V41_*`）是**同一类坑** ——
**开关送不进去时，程序不会报错，只会安静地跑默认值**。

**修法**（`scripts/a2_one_shot_probe.sh`，2026-09-22 16:2x）：
1. 用 `-e` 显式把 `LIGHT` / `COPY_GIB` / `A2PROBE_FLOOR_GIB` 送进容器；
2. 容器内**回显实际生效值**：`[a2probe][in-container] LIGHT=... COPY_GIB=... A2PROBE_FLOOR_GIB=...`；
3. ★ **fail-closed**：三个开关少任何一个 ⇒ 打印 `⛔ 环境变量 X 没有传进容器` 并 `exit 65`，
   **拒绝运行**（宁可当场失败，也不要静默降级）。

**验证**（正反两向都做了，在 A3 的 `prbench-c2` 上）：
```
传 -e   ⇒ in-container: LIGHT=0 FLOOR=300        ✅
不传    ⇒ in-container: LIGHT=<unset> FLOOR=<unset>
故意少传一个 ⇒ ⛔ 环境变量 A2PROBE_FLOOR_GIB 没有传进容器 ⇒ 拒绝运行  rc=65   ✅
```

---

## 3c. ★★★ 第三次运行（16:32）的完整读数 —— **这就是最终结论**

### 3c.1 判据（全部【实测】）

| 档 | 注册 | 耗时 | H2D | D2H | **往返逐字节** |
|---:|---|---:|---:|---:|:---:|
| 1 GiB | ✓ `ret=0` | 0.3 s | 21.3 GB/s | 20.2 GB/s | ★ **True** |
| 8 GiB | ✓ `ret=0` | 3.2 s | 16.4 GB/s | 20.4 GB/s | ★ **True** |
| 32 GiB | ✓ `ret=0` | 11.4 s | 9.2 GB/s | 20.6 GB/s | ★ **True** |
| **64 GiB** | ★ **✓ `ret=0`** | 22.5 s | 21.3 GB/s | 20.1 GB/s | ★ **True** |
| 文件映射 1 GiB | ✓ | 1.2 s | 5.0 GB/s | 20.0 GB/s | True |
| 文件映射 8 GiB | ✓ | 9.1 s | 4.7 GB/s | 20.5 GB/s | True |
| pageable 对照 | — | — | **5.3 GB/s** | 19.6 GB/s | True |

★ **`is_pinned=True` 四档全中** —— 注册之后这块普通内存在 A2 上**真的被锁住了**。
★ **pageable 只有 5.3 GB/s，注册后 16–21 GB/s ⇒ 注册路线在 A2 上带来 3–4× 的 H2D 提升**。

### 3c.2 另外两条（顺带把旧结论钉死）

```
单次 pinned：1 / 4 / 8 / 16 / 32 GiB  ★ 全部 ✓（32 GiB 用 3.3 s）
多次小分配累加 pinned = 77.0 GiB（★ 受 FLOOR_GIB=300 所限，不是驱动上限）
```
⇒ ★★ **`logs/012` 那条"单次 pinned 上限 ∈ (4,8] GiB"彻底作废**（第三次被推翻：`014` → 16:10 → 本次 32 GiB）。
⇒ 候选 α（分片 pinned）在 A2 上**也是可行的**，即 β 路线若遇到别的障碍，仍有退路。

### 3c.3 ★ 一个必须记的**成本项**（【推断】，但值得预先记账）

```
64 GiB 注册耗时 22.5 s  ⇒ ≈ 0.35 s/GiB
A2 的池 ≈ 49 GiB/worker × 8 worker ≈ 392 GiB
⇒ 【推断】首次注册的**一次性**成本 ≈ 137 s
```
★ 这个量级**可以接受**（A2 起服本身要 800+ s），但它是**起服期**成本，不要误算进 decode 时延。
★ `logs/030` 在 A3 上实测过"第一次 `aclrtHostRegister` 一次性多花 ~589 MiB"，**A2 的同项未单独测**【未确认】。

### 3c.4 诚实边界：**探针测的是"单进程单块"，生产是"8 进程各一块"**【未确认】

本节四档读数都是**一个进程注册一个连续块**。生产形态是 **8 个 worker 各注册 ~49 GiB**（总量 ~392 GiB）。
两者**不是同一件事** —— 未测的差异有三条：
1. **8 进程并发注册**的行为（总锁页量 ~392 GiB、宿主的 `vm.max_map_count` / pinned 记账）；
2. **worker 之间是否会互相干扰**（跨进程的 host 注册在同一设备上）；
3. **运行时并发 H2D**（本探针是单进程串行）。

⇒ ★ **上线判据**：第一次起服时**先看 `[P1_pinned] CPU pool ... registered dev=... ret=0` 是否 8/8 都出现**
（`logs/022` 的形态），任一 worker 不是 `ret=0` ⇒ 停下来看，别直接压测。

---

## 3b. ⚠️ 历史缺口（`LIGHT=1` 的上限是 4 GiB）—— **已被 §3c 的回答取代，此节保留作过程记录**

### ★★★ 3b.0 第二次实机失败（16:2x）：`LIGHT=0` 生效了，但**注册那段被整段跳过**（已修 + 已真机验证）

`LIGHT=0` 这次开关确实进了容器（回显 `LIGHT=0 COPY_GIB=0.25 A2PROBE_FLOOR_GIB=300` ✅），
但输出里 **`step4` / `step5` 一行都没有**，DECISION 反而给出：
```
注册（匿名/普通内存）：1=None 8=None 32=None 64=None
⇒ 注册不可用但 pinned 总量可以 ⇒ 走候选 α（分片 pinned）    ← ★★ 这是**反结论**
```

**根因**【实测】：**`step1b`（累加 pinned）会故意吃到 `FLOOR_GIB` 为止，而 torch 的 host pinned
分配器不把内存还给 OS**。日志证据链：
```
吃到 309 × 256 MiB = 77.2 GiB 后   MemAvailable = 300.1   ← floor 恰是 300
held.clear() + gc.collect() 之后
step4 起始                          MemAvailable = 300.0   ← ★ 没还回去
```
⇒ 它后面**每一个** `ok_floor(g)` = `(MemAvailable − g) > FLOOR` **都恒为假** ⇒ `step4/step5` 被整段跳过。
⇒ 后果：**整轮跑完，"注册路线能不能用"这一格（本轮最关键的一格）根本没测**，
而 DECISION 还把它说成"注册不可用 ⇒ 走候选 α"。

**修法**（两处）：
1. ★ **把 `step1b` 固定放到最后** —— 它本来就是"吃到 floor 为止"的收尾测试；
2. ★ **区分"跳过"与"失败"**：被 `ok_floor` 跳过时置 `register_not_measured`，
   DECISION 改判为「⏹ **注册路线根本没测到**（这**不是**『注册不可用』）」。

**真机验证**【实测·A3 c2 / `prbench-c2`，floor=1191 GiB **故意让 `step1b` 触底**】：
```
step3 pageable
→ ★ step4 register 1 GiB ✓ / 4 GiB ✓（判据 True）
→ step5 register-file 1 GiB ✓
→ ★ step1b（最后）  30 × 256 MiB = 7.5 GiB（触到 floor 1191）
```
⇒ 新顺序下 `step4` 正常跑；**旧顺序下 `step1b` 会先把 MemAvailable 压到 1191，
`step4` 的 `ok_floor(1)` = `1190 > 1191` 为假 ⇒ 必然跳过**（= 用户看到的那份输出）。
⇒ ★ **这就是那次失败的最小复现 + 修复的判别性证据。**
验证用的脚本 sha256 与本地一致（`b42d74a7…`）；c2 锁已交还、无残留进程、测试目录已删。

---

DECISION 段的 `None` **不是失败，是"没探"** —— 源码：
```python
step4_register(acl, stream, [1, 4] if LIGHT else [1, 8, 32, 64])
```
⇒ `LIGHT=1` 只探 1 / 4 GiB。而**A2 的池需要 ~49 GiB/worker**（`logs/027`：`OFFLOAD_GB=56` ⇒ 宿主实占 392.4 GiB = **49.05 GiB/worker × 8**）。

**⇒ 必须补跑一次**（`A2PROBE_FLOOR_GIB=300 LIGHT=0`，**用修好的脚本**；跑前先确认容器内回显的那一行）：
* 会探 **1 / 8 / 32 / 64 GiB** 的注册 + 文件映射 1 / 8 GiB；
* ⚠️ **修正之前的说法**：我原先写"438 GiB available、floor 300 ⇒ 四档都满足 `ok_floor`"——**那是错的**，
  因为**前面的 `step1b` 会把 MemAvailable 压到 floor**（§3b.0）。修好顺序后这个前置条件才真正成立
  （注册段先跑、那时 MemAvailable ≈ 439 > 300 + 64）；
* 【实测·A3 同脚本】耗时 **140 s**、宿主峰值 ≈ 200 GiB、显存峰值仍只 256 MiB；
* ⚠️ 它会在宿主上真分配最多 64 GiB 并锁页 ⇒ 跑的时候**别同时压测 A2 的服务**。

**判读**：
```
register_64 = True  ⇒ ★ 池可以整体注册 ⇒ NPU_OFFLOAD_HOST_MEM=registered，池子按 56 GiB/worker 配
register_32 = True  但 64 = False ⇒ 池要分片注册（或把 OFFLOAD_GB 降到 ~32/worker 以下）
只有 register_8 ⇒ 池子必须 ≤8 GiB/worker（≈ 现在目标的一半）
```

---

## 4. 与既有日志的关系（谁被证实、谁被推翻）

### ★★★ 4.0 第 4 次同类静默失败：**变量名有两套，方向是单向的**（2026-09-22 16:5x，主代理自查发现）

起因：给用户的起服命令里，我写的是 `A2_KV8_SWA=1 A2_RING_FP16=1`。
**实测对照**（`DRY=1`，同参数只换变量名）：
```
A) A2_KV8_SWA=1 A2_RING_FP16=1   ⇒   ★★ 档位 : B        ← ★ 自报档 B！
B) KV8_SWA=1   KV8_RING_FP16=1   ⇒   ★★ 档位 : C
                                     ⚠⚠ APC_ALIGN=0 ⇒ 自动置 3
                                     ⚠⚠ GRAPH_SAFE=0 ⇒ 自动置 1
```
**机制**（`serve_a2_offload.sh` 里是**单向**翻译）：
```
用户接口（本脚本读）    : KV8_SWA / KV8_RING_FP16 / KV8_FULL / KV8_PREFILL / APC_ALIGN / GRAPH_SAFE
内部名（本脚本写出去）  : A2_KV8_SWA / A2_RING_FP16 / A2_KV8 / A2_APC_ALIGN / A2_GRAPH_SAFE  → shadow 的挂载块
```
⇒ 在外面传 `A2_*` 时：本脚本**读不到**（读到默认 0），随后**用 0 覆盖它**。
★ **最危险的形态**：若同时传了 `A2_RING_FP16`，shadow 的挂载块**照样挂上 7 件**
（它的触发条件是 `A2_KV8 || A2_KV8_SWA || A2_RING_FP16`）⇒ 看起来"int8 开了"，
**但 `APC_ALIGN=0`（会翻 token）+ `GRAPH_SAFE=0`（图模式捕获期炸 EE1016）**
⇒ **比不挂更危险**：错误会晚到压测阶段才暴露，而归因会指向别处。

**修法**：`serve_a2_offload.sh` 开头加 **fail-closed** —— 检测到外面设了任一 `A2_*` 直接 `exit 64`，
并打印"请改用 `KV8_*`"。已在本地验证正反两向：
```
A) ⇒ ⛔ 检测到你在外面设了内部变量 A2_KV8_SWA=1 ...（exit 64）
B) ⇒ 照常（档位 C / APC_ALIGN=3 / GRAPH_SAFE=1）
```
⇒ ★ **教训归并**：本日 4 次静默失败（§3 / §3b.0 / T 的 `swa_plane_kwargs` / 本条）**全部同源** ——
**"开关送不到该到的地方时，程序不报错，只安静地跑默认值"**。修法一律是：
**① 显式转发；② 回显实际生效值；③ 不匹配就 fail-closed。**

| 日志 | 当时的说法 | 本次实测 |
|---|---|---|
| `012` | "A2 的 `cpu_bytes_to_use≈260GB` 计划作废；单次 pinned 上限 ∈ (4,8] GiB" | ⚠️ **8 GiB 单次通过** ⇒ "单次上限"这条**第二次被弱化**（`014` 在干净进程里已推翻过一次） |
| `014` | "候选 β 的生死判据 = 注册内存能不能真的走 H2D/D2H（**H2H 通过不算数**）" | ★★★ **判据通过**（1 GiB / 4 GiB 都逐字节一致） |
| `014` | "A3 上 H2D 58 / D2H 42.7 GB/s" | A2 上是 **20.4 / 20.0 GB/s** ⇒ **A2 慢 ~2.8× / 2.1×**（A2 是 910B3，与 A3 的 910C 不同代） |
| `052` | "旧版脚本少了 `aclrtSetDevice`，会让 `aclrtMallocHost` 一律报 107002（看着像 A2 不能用 pinned）" | ★ **本次（带 `set_device` 的 v2 脚本）pinned 与 register 全通** ⇒ 印证了 `052` 那条修正 |

---

## 5. 诚实边界（【未确认】三条 —— 前两条已被 §3c 回答）

| # | 项 | 状态 |
|---|---|---|
| 1 | `≥8 GiB` 的注册没测 | ✅ **已答**（§3c：8 / 32 / **64 GiB 全 True**） |
| 2 | 单次 pinned 上限 | ✅ **已答**（§3c：**32 GiB ✓**，旧结论 (4,8] GiB 作废） |
| 3 | ⛔ **8 个 worker 各注册 ~49 GiB** 的行为 | **仍未测**（本次是单进程单块）—— 三条具体差异见 §3c.4 |
| 4 | ⛔ **A2 上"首次注册多花多少常驻内存"** | **仍未测**（`logs/030` 在 A3 上是 ~589 MiB/次） |

⇒ ★ **第 3/4 条都不阻塞"路线选择"**：判据已过 ⇒ **走 `registered`**。它们是**起服期**的风险，
用一条现成的观测判据就能覆盖（§3c.4 那句"看 8/8 worker 的 `ret=0`"）。

---

## 6. 结论

```
① ★★★★★ 候选 β 在 A2 上成立且**上限足够**：aclrtHostRegister(MAPPED) 注册 1/8/32/**64** GiB 全 ✓，
   注册内存真能走 H2D/D2H（逐字节一致）⇒ **池可整体注册（~49 GiB/worker < 64 GiB），不必分片**
② ★ host_mem_pool=0 被实测确认 ⇒ 必须走 registered，不能靠 aclrtMallocHost 撑池
③ ★ 注册路线在 A2 上的收益：H2D pageable 5.3 → 注册后 16–21 GB/s（3–4×）
④ ★ 顺带作废一条旧结论：单次 pinned **32 GiB ✓**（`logs/012` 的"(4,8] GiB 上限"彻底作废）
⑤ ⚠️ 三个脚本 bug 已修（§3 / §3b.0，都是**静默失败**）⇒ 交付脚本 sha256 = b42d74a7…
```

### ★★ 对上线的直接含义

```bash
# 池后端（这一格已经不用再问了）
NPU_OFFLOAD_HOST_MEM=registered      # logs/014 §4 的候选 β，A2 实测成立
OFFLOAD_GB=56                        # logs/027 的定值（57,344 unit = 48,064 需求的 1.193×）
```
★ **下一步就是 `docs/A2-DEPLOY-NOW.md` 的「第二步：造 shadow-pkg」+ 起服** ——
**探测这条路已经走完了**。
