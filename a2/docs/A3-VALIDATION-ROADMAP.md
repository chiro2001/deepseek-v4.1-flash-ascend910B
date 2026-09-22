# A3 代测路线图 —— 把 draft 入图 + kv8 + DRAM 卸载在 A3 上验完，A2 只做最后一步

> 2026-09-22 17:3x 制定。背景：**A2 是生产机，腾不出时间试错**；而三个轴（draft 入图 / kv8 / 卸载）
> **从未同时开过**，直接上 A2 风险太高。
> 标记：【实测】/【推断】/【未确认】。

---

## 0. 一句话

**能。而且 90% 的风险可以在 A3 上消掉** —— 因为那些风险**全是代码路径问题，不是 A2 硬件问题**。
真正只有 A2 能回答的只有 **两条**，其中一条**可以在 A2 不停机的情况下测**（见 §4）。

```
A3 可测（8 卡真权重，Phy-ID 8-15，现在全空闲）：
  ★ 卸载 × DRAFT_GRAPH=1        ← 最大的未知，从未同开
  ★ int8（档 C/D）× DRAFT_GRAPH=1
  ★ 1M 几何（MAX_LEN=1048576 / MAX_SEQS=4 / OFFLOAD_GB=85）
  ★ 页几何、捕获期错误码、功能三判据、输出 sha、投机四数

只有 A2 能测（两条）：
  ① host_mem_pool=0 下、8 进程 × 16 张量的注册能不能全 ret=0（A3 的 host_mem_pool=1）
  ② 910B3 的绝对性能数字（倍率可传递，绝对值不可）
  ⇒ ★ ① 可以在 A2 不停机的情况下测（见 §4）
```

---

## 1. 为什么 A3 的结果可以外推到 A2（以及不能外推什么）

### 1.1 可传递【实测依据】

| 项 | 为什么可传递 |
|---|---|
| **页几何 / 容量倍率** | `Σslot_pages` 与 `BPR` 只由 block/head/dtype/shape 决定，与芯片无关。`logs/065` 实测：A2 与 A3 的 B/token 只差 0.02% |
| **捕获期错误码** | `EE1016` / `507057` 是驱动 + CANN 的行为；`logs/053` 实测 tiny 单卡与 8 卡的签名逐项逐字一致 |
| **功能三判据** | `BlockStored:CPU` / `CPU→GPU` / `hits` 是引擎侧计数，与芯片无关 |
| **代码路径** | 三条线的补丁都是同一份文件（scheduler.py / cpu_npu.py / dsa_v41.py），A2/A3 挂的是同一 md5 |

### 1.2 不可传递（必须小心）

| 项 | 差异 | 影响 |
|---|---|---|
| ★ **`host_mem_pool`** | A3 = **1**，A2 = **0** | 注册路线的行为可能不同 ⇒ §4 专门处理 |
| **绝对 token 数** | 910C vs 910B3，可用 KV 显存 15.82 vs 14.40 GiB | 只影响绝对值，倍率可传递 |
| **A2 的 CPU 更弱** | draft eager 的派发开销更大 | `DRAFT_GRAPH=1` 的收益在 A2 更大（实测 +62% vs A3 的 −12 ms/step）⇒ A3 测出来没收益不代表 A2 没收益 |
| **模型** | A3 八卡臂一直用 `v41-w4a8-engram-dr-vision-qrot-mtpq`；A2 用 `v41-w4a8-flat` | ★ A3 上有一个结构完全一致的 flat 对照（见 §2），可以用它补这一格 |

---

## 2. A3 的资源现状（2026-09-22 17:2x 实测）

```
8 卡（Phy-ID 8-15）  全空闲：npu-smi 无进程、无 r8-* 容器、c0/c1/c2 锁全 free
宿主内存             1710 GiB available
镜像                quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3
模型（两条）
  ① v41-w4a8-engram-dr-vision-qrot-mtpq   ← 所有既有 8 卡臂用的那个（基线可比）
  ② ★ v41-flat-verify3                     ← 与 A2 结构完全一致（见下）
```

★ **`v41-flat-verify3` 与 A2 的 `v41-w4a8-flat` 结构逐项吻合**（这正是 `docs/A2-DEPLOY-NOW.md` 里
那个 A2 的模型与我们实测的不是同一个 缺口的解法）：
```
engram_layer_ids = [1, 14]  OK        engram_extra.safetensors  OK   engram_int8/  OK
mtpq 分片 = 4 个            OK        vision 分片                OK   optional/quarot.safetensors OK
quant_model_weights.safetensors.index.json OK
```
⚠️ **但要如实标**：结构一致 **≠ 权重逐字节同源**（A2 那份叫 `flat`、A3 这份叫 `flat-verify3`）。
⇒ 用它跑可以关掉模型结构差异这个变量，但仍留一格【未确认】。

---

## 3. 路线图（五阶段）

### 总原则

**每次只加一个新轴**，顺序按"你最需要的先上"：
```
基线（已知） → +卸载 → +1M 几何 → +int8 档C → [+int8 档D]
                 ^          ^              ^
              你最需要的   你的真实口径    次要（容量/内存收益）
```

### Phase 0 —— 不占卡（30 min，可以现在就做）

| # | 事项 | 判据 |
|---|---|---|
| 1 | 把 A2 的 1M 参数账算定 | `OFFLOAD_GB=85`（87,040 unit = 3×24,064×1.2）、宿主 **296.7 GiB** |
| 2 | 确认 A2 的 `--kv-cache-memory-bytes >= 1M x kv_per_tok` | 不满足 ⇒ 1M 起不来（独立门槛） |
| 3 | 用 `v41-flat-verify3` 跑一遍模型自检 | `tools/check_model_dir.sh`，看 Engram 2 层 + mtpq 4 分片 |
| 4 | 把要挂的 7 件 md5 再核对一遍 | `bash a2/scripts/check_artifact_identity.sh` |

### Phase 1 —— ★★ 卸载 × `DRAFT_GRAPH=1`（A3 8 卡，**最大的未知**）

**为什么是第一优先**：这两个轴**从未同开**（全仓 18 条 8 卡臂全是 `DRAFT_GRAPH=0`），
而 A2 生产**正是 `DRAFT_GRAPH=1`**。

```
臂 P1-A（对照）：卸载 + DRAFT_GRAPH=0     ← 已有同类臂（027/042），复跑确认基线
臂 P1-B（实验）：卸载 + DRAFT_GRAPH=1     ← 新轴
```

| 判据 | 期望 | 不过怎么办 |
|---|---|---|
| `[P1_pinned] CPU pool ... ret=0` | **8/8** | 注册问题 ⇒ 换 `pin_memory` 后端对比 |
| `BlockStored:CPU` / `CPU→GPU` / `hits` | 都 > 0 | 池没工作 ⇒ 退回 `DRAFT_GRAPH=0` 分离 |
| `BlockRemoved:CPU` | **0** | 池被撑爆 ⇒ 加大 `OFFLOAD_GB` |
| ★ **单流 tok/s** | `DRAFT_GRAPH=1` 应明显高于 `=0` | 若相等 ⇒ 图没真的在跑（查 DRAFT-GUARD / `A`） |
| ★ **稳态 A** | A3 上 2.4–2.7 | 掉到 1.0 ⇒ 撞了已知静默失效 |

★ **这一步特有的风险（要专门看）**：draft 组参与卸载（group 清单里参与位是 `True`），
而 `DRAFT_GRAPH=1` 时 draft 的 KV 访问发生在**图里面** ⇒ **DMA 完成与图重放读 KV 的时序**没人测过。
⇒ 若 P1-B 崩或错，**先把 `DRAFT_GRAPH=1` 去掉**再跑一遍分离。

### Phase 2 —— 1M 几何（A3 8 卡）

把配置换成 A2 的真实口径，验证**参数账**而不是新功能：

```
MAX_LEN=1048576  MAX_SEQS=4  OFFLOAD_GB=85  BAT_TOKENS=2048  DRAFT_GRAPH=1
```

| 判据 | 期望 | 备注 |
|---|---|---|
| `GPU KV cache size` | 出现且 >= 1M（否则单请求装不下） | A3 的绝对值 != A2 |
| 池覆盖 | 3 个 1M 会话无 `BlockRemoved` | 这是 `OFFLOAD_GB=85` 的推导目的 |
| ★ **23.5 unit/1024token 在 1M 上是否仍线性** | 只看实测，别外推 | `DELIVERY §6.5.5` 列为待测格 |
| 起服时间 | 记录 | A2 上线时的预期停服窗口 |

### Phase 3 —— int8（A3 8 卡）

**3a：档 C**（`KV8_SWA=1 KV8_RING_FP16=1`）—— 脚本自动置 `APC_ALIGN=3` + `GRAPH_SAFE=1`。

| 判据 | 期望 |
|---|---|
| 起服 | 无 `EE1016`、无 `507057` |
| ★ 档位自报 | dry-run 里 `档位 : C`（显示 B ⇒ 开关没生效） |
| 功能三判据 | 全中 |
| ★ **输出不翻 token** | 与同时段的 `KV8_*=0` 臂比，同一 prompt 的 token 一致（`047` 的 mode3 保证） |
| 投机四数 | 不劣于同时段对照臂 |

**3b：档 D**（再加 `KV8_FULL=1 KV8_PREFILL=1`）—— 这是第二个必需件（`model.py`）的验证点：

```
★ 关键：档 D 必须挂 models/deepseek_v41/model.py（long_kv_plane_kwargs 的唯一调用点）
   —— T 的 runner 曾经只在档 D 才挂它，导致档 C 臂静默跑成 BF16（logs/050 §7.7④）
判据：容量指纹 —— 档 C=427,643 / 档 D=485,610（同一 max_len 下）
```

### Phase 4 —— A2 零停机探测（见 §4）

### Phase 5 —— A2 一次重启上线（见 §5）

---

## 4. ★★ A2 唯一剩下的大未知，可以**不停机**测掉

### 4.1 这条为什么重要

```
A2 与 A3 的关键差异（实测）：
  host_mem_pool            A3 = 1     A2 = 0      ← 相反
  host_pin_pre_register    A3 = 1     A2 = 0
A2 已测（用户 16:32 的探针）：
  单进程、单块：注册 1/8/32/64 GiB 全过，往返逐字节一致
未测：
  8 个 worker 并发、每个 16 张张量、合计 ~392 GiB 的注册
  （A3 上 022 那轮实测 128 次注册 ret=0 = 16 张量 x 8 rank —— 但那是 host_mem_pool=1 的机器）
```

### 4.2 关键洞察：**探针可以跑在正在生产的那台 A2 里**

用户 16:10 / 16:32 那两次探测**就是 `docker exec` 进正在服务的容器跑的**（服务没停）。
⇒ 同一个手法可以扩到多进程。

### 4.3 分期探针（**从低风险开始，每期看结果再决定要不要继续**）

| 期 | 规模 | 测什么 | 风险 |
|---|---:|---|---|
| **S1** | 8 进程 × 4 GiB = **32 GiB** | ★ 8 进程并发注册这个机制在 A2 上通不通 | 极低（32 GiB / 439 GiB 可用） |
| **S2** | 8 进程 × 16 GiB = **128 GiB** | 规模上去后 `ret` 是否仍 0；首次注册的常驻内存开销 | 低 |
| **S3** | 8 进程 × 49 GiB = **392 GiB** | ★ 生产真实形态 | **中**（可用内存会被压到 ~47 GiB）⚠️ 需要用户自己判断时机 |

**每期都内置 floor 保护**（`A2PROBE_FLOOR_GIB`），不够就**跳过并报告**，不硬撑。

### 4.4 ★ 探针已写好：`a2/scripts/a2_multiproc_reg_probe.sh`

```bash
# 在 A2 宿主上（脚本自己 docker exec 进服务容器）——★ 服务不用停
# S1（先跑这个）
A2_CONTAINER=dsv41-a2 NPROC=8 PER_PROC_GIB=4  A2PROBE_FLOOR_GIB=300 \
  bash a2/scripts/a2_multiproc_reg_probe.sh
# S2（S1 通过后）
A2_CONTAINER=dsv41-a2 NPROC=8 PER_PROC_GIB=16 A2PROBE_FLOOR_GIB=300 \
  bash a2/scripts/a2_multiproc_reg_probe.sh
```

**它做了什么**（与单进程探针的关键区别）：
1. 起 **N 个进程**（默认 8 = TP8），第 i 个用第 i 张卡（贴近生产）；
2. 每进程分配 `PER_PROC_GIB` 的**普通**（pageable）host 内存；
3. ★ **在 barrier 处等齐，然后所有进程同时 `aclrtHostRegister(MAPPED)`** ——
   串行注册测不出争用，这一步才是"并发"的关键；
4. 在 256 MiB 切片上做**真实 H2D→D2H 逐字节对账**（与单进程探针同一条判据）；
5. 保持 3 s（让 N 个进程真的**同时持有**），再注销；
6. 逐进程打印 `reg_ok / reg_s / xfer_ok / rss_before / rss_after / H2D`。

**判据与退出码**：
```
rc=0   ⇒ ★ 全部进程注册 + 设备往返判据都通过
rc=65  ⇒ ⏹ **拒绝运行**（余量不足）—— 这是**保底行为，不是失败**
rc=1   ⇒ ⛔ 没有全过（部分失败 / 往返判据不过）—— ★ 这正是本探针要找的东西
```
★ **`rss_before` / `rss_after` 两列**顺便回答 `logs/030` 留下的那个问题：
"第一次 `aclrtHostRegister` 一次性多花多少常驻内存"（A3 上是 ~589 MiB/次，A2 未测）。

★ **fail-closed 已实测**：需求超出余量时**在起进程前**就拒绝（不静默降级成"跑了几期"）。

★ **S3 的建议**：如果 S2 通过，S3 其实**可以在正式上线的那个窗口里顺便验证** ——
因为上线本身就要注册 392 GiB。把它当成上线时的第一个判据（§5 第 3 步），而不是单独冒一次风险。

---

## 5. Phase 5 —— A2 一次重启上线（把停服窗口压到最短）

### 5.1 上线前的准备（全在 A3 完成，不占 A2 时间）

```
① 在 A3 上把最终配置 dry-run 到挂载清单完全正确
② 在 A3 上确认 md5 与文件名（7 件 + 4 个补丁）
③ 把 A2 要跑的命令写成一个脚本，A2 上只做 拉脚本 + 跑 + 看三行
④ ★ 先存好当前生产的启动命令（回滚用）：
     docker inspect dsv41-a2 --format '{{.Config.Cmd}}' > ~/a2-rollback-cmd.txt
```

### 5.2 窗口内的顺序（**每一步都有硬门，不过就回滚**）

| 步 | 动作 | 门（不过就停） | 预计 |
|---|---|---|---|
| 1 | 记下当前启动命令 | 文件非空 | 1 min |
| 2 | `docker rm -f dsv41-a2` + 起新配置 | — | — |
| 3 | ★ **注册门** | `[P1_pinned] ... ret=0` **8/8** | 起服期 |
| 4 | ★ **档位门** | 日志里档位自报正确（B / C / D） | 起服期 |
| 5 | ★ **容量门** | `GPU KV cache size` 出现且 >= 1M | 起服期 |
| 6 | **功能门** | `BlockStored:CPU`>0、`CPU→GPU`>0、`hits`>0、`BlockRemoved:CPU`=0 | 首次压测 |
| 7 | **质量门** | 同一 prompt 的输出不翻 token；`A` 落在健康区间 | 首次压测 |
| **回滚** | 用第 1 步存下的命令重启 | — | ~5–13 min |

★ **预计停服窗口**：A2 那个 `DRAFT_GRAPH=1` 会话实测起服 **803 s（含 static kernel 冷编译；
缓存命中后应回到分钟级）** ⇒ 正常情况 **5–15 min**；失败回滚再 +5–15 min。

### 5.3 把风险放在"能回滚"的一侧

```
★ 第一版上线的配置 = 卸载 + DRAFT_GRAPH=1 + 1M + MAX_SEQS=4    （不加 int8）
  理由：int8 的收益（容量 x1.1354 / 宿主省 47 GiB）都不是 能不能用 的问题，
        可以等第一版稳了再作为第二次变更加上 —— 那次可以在 A3 先测透。
```

---

## 6. 时间与资源估算

| 阶段 | 占什么 | 单价 | 条数 | 小计 |
|---|---|---|---|---|
| Phase 0 | 不占卡 | — | — | ~30 min |
| Phase 1（卸载 × draft 入图） | A3 8 卡 | ~25–30 min/臂 | 2 | ~1 h |
| Phase 2（1M 几何） | A3 8 卡 | 同上 | 1–2 | ~1 h |
| Phase 3（int8 档 C / D） | A3 8 卡 | 同上 | 2 | ~1 h |
| Phase 4（A2 零停机探针 S1+S2） | A2 容器内 | ~2 min | 2 | ~5 min（**不停机**） |
| Phase 5（A2 上线） | A2 **停服** | 5–15 min | 1 | **5–15 min** |

⇒ **A2 总占用 ≈ 20–30 min（含回滚余量），其中真正停服 5–15 min。**
⇒ **A3 需要 ~3–4 小时卡时**（8 卡现在全空闲）。

---

## 7. 红线（A3 上跑的时候必须遵守）

1. **8 卡走 c0 锁 + Phy-ID 8–15**，一次只跑一条臂（`a3_chip.sh` 的 flock + 退出码 75）
2. **绝不碰** `dsv41-a3`（用户的容器，保持 `Exited`）/ `mooncake-*` / 别人容器 / **Phy-ID 0–7**
3. **绝不手设 `ASCEND_RT_VISIBLE_DEVICES`**
4. **起服前查 `df -h /dev/shm`**（满 ⇒ `OSError [Errno 28]` 在 `SemLock`，极易误判成补丁坏了）
5. **绝不用 `/tmp`**：`source a2/scripts/tmpdir.sh <任务名>`
6. 结论标 **【实测】/【推断】/【未确认】**；不许用相邻数字顶替缺的那格
7. **A3 远端时钟比本机慢 ~7 min** ⇒ 看进度读远端 `date` + 日志字节增长
8. **Phase 4/5 涉及 A2 生产** ⇒ 只能由**用户本人**执行；我这边只负责把命令与判据备好

---

## 8. 这套路线图能回答 / 不能回答的问题

| 问题 | 能答吗 |
|---|---|
| 卸载 + draft 入图会不会互相破坏？ | ✅ **能**（Phase 1） |
| int8 + draft 入图会不会撞 EE1016？ | ✅ **能**（Phase 3） |
| 1M 下池要多大 / 3 个会话够不够？ | ✅ **能**（Phase 2） |
| A2 上 8 进程 × 392 GiB 注册能不能过？ | ⚠️ **只能答机制**（Phase 4 S1/S2）；**S3 建议并入上线窗口** |
| A2 上能跑到多少 tok/s？ | ⛔ **不能**（910B3 绝对值）—— 但**倍率**可传递 |
| A2 的 flat 权重与 A3 的 flat-verify3 是否逐字节同源？ | ⛔ **不能**（需要一次 md5 对比，用户可零成本做） |
