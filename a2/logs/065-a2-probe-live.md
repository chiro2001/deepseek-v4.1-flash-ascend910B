# 065 — ★★★ **A2 实机探测：唯一阻塞解除**（`aclrtHostRegister` 路线在 A2 上可用）

> 2026-09-22 16:10 CST。**执行：用户**（在 A2 宿主 `~/projects/dsv41-a2-repro-kv8-offloading/` 下跑）。
> 脚本：`a2/scripts/a2_one_shot_probe.sh`。
> ★ 本次跑的是 **`f88672b317521ec9fcd61c2c5faef0282ea9f6b6d37201bac7214689d2327740`**（**修复前**）；
> **§3 修了那个静默失败之后**是 **`a4113ff4e2cc6510572be77835c984f50b5422150e13518251482b870bc2b443`** ⇒ **补跑大档必须用后者**。
> 口径：`A2_CONTAINER=dsv41-a2 A2PROBE_FLOOR_GIB=300 LIGHT=1`。
> 标记：【实测】/【推断】/【未确认】。

---

## 0. 一句话

**★★★ 判据过了** —— `★ 注册内存的设备往返判据 = True`（1 GiB 与 4 GiB **都过**，且都是**真实 H2D→D2H 逐字节对账**，不是 H2H）。
⇒ **候选 β（普通 host 内存 + `aclrtHostRegister(MAPPED)`）在 A2 上成立**，**唯一阻塞上线的探测完成**。
⇒ ⚠️ 但**只探到 4 GiB**（`LIGHT=1` 的档位上限）—— 池需要 **~49 GiB/worker** ⇒ **必须补跑 `LIGHT=0`**（见 §3）。

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

## 3b. ⚠️ 因此仍未探到的缺口：**大档**（`LIGHT=1` 的上限是 4 GiB）

DECISION 段的 `None` **不是失败，是"没探"** —— 源码：
```python
step4_register(acl, stream, [1, 4] if LIGHT else [1, 8, 32, 64])
```
⇒ `LIGHT=1` 只探 1 / 4 GiB。而**A2 的池需要 ~49 GiB/worker**（`logs/027`：`OFFLOAD_GB=56` ⇒ 宿主实占 392.4 GiB = **49.05 GiB/worker × 8**）。

**⇒ 必须补跑一次**（`A2PROBE_FLOOR_GIB=300 LIGHT=0`，**用修好的脚本**；跑前先确认容器内回显的那一行）：
* 会探 **1 / 8 / 32 / 64 GiB** 的注册 + 文件映射 1 / 8 GiB；
* 按本次余量（438 GiB available、floor 300）**四档都满足 `ok_floor` 的前置条件**（需要 `438−g > 300`，即 `g < 138`）；
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

| 日志 | 当时的说法 | 本次实测 |
|---|---|---|
| `012` | "A2 的 `cpu_bytes_to_use≈260GB` 计划作废；单次 pinned 上限 ∈ (4,8] GiB" | ⚠️ **8 GiB 单次通过** ⇒ "单次上限"这条**第二次被弱化**（`014` 在干净进程里已推翻过一次） |
| `014` | "候选 β 的生死判据 = 注册内存能不能真的走 H2D/D2H（**H2H 通过不算数**）" | ★★★ **判据通过**（1 GiB / 4 GiB 都逐字节一致） |
| `014` | "A3 上 H2D 58 / D2H 42.7 GB/s" | A2 上是 **20.4 / 20.0 GB/s** ⇒ **A2 慢 ~2.8× / 2.1×**（A2 是 910B3，与 A3 的 910C 不同代） |
| `052` | "旧版脚本少了 `aclrtSetDevice`，会让 `aclrtMallocHost` 一律报 107002（看着像 A2 不能用 pinned）" | ★ **本次（带 `set_device` 的 v2 脚本）pinned 与 register 全通** ⇒ 印证了 `052` 那条修正 |

---

## 5. 诚实边界（【未确认】三条）

1. ⛔ **`≥8 GiB` 的注册没测** ⇒ 池子上限**还不知道**（这是 §3 要补的那一格）；
2. ⛔ **8 个 worker 各注册一大块** 时的行为没测（本次是单进程、单块）；
3. ⛔ **注册 49 GiB × 8 worker 的首次建表耗时**没测（`logs/030` 在 A3 上实测过"第一次 `aclrtHostRegister`
   一次性多花 ~589 MiB"，A2 的量级未测）。

⇒ 但这三条**都不阻塞"路线选择"**：判据已过 ⇒ **走 registered**；剩下的只是**配多大**。

---

## 6. 结论

```
① ★★★ 候选 β 在 A2 上成立：aclrtHostRegister(MAPPED) 可用，且注册内存真能走 H2D/D2H（逐字节一致）
② ★ host_mem_pool=0 被实测确认 ⇒ 必须走 registered，不能靠 aclrtMallocHost 撑池
③ ⚠️ 池子上限仍未知：需补跑 LIGHT=0（1/8/32/64 GiB），约 140 s
④ ★ 注册路线在 A2 上的收益：H2D 5.5 → 20.4 GB/s（3.7×）
```
