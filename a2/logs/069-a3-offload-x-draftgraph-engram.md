# 069 · A3 8 卡：「DRAM 卸载 × DRAFT_GRAPH=1」首次同开（P1 通过）+ Engram × 卸载池的 P0 判决

> **任务**：`a2/docs/A3-VALIDATION-ROADMAP.md` **Phase 1** —— 在 A3 8 卡（Phy-ID 8–15）验证
> 「DRAM 卸载 × draft 入图」这个**从未同开过**的组合。
> **子代理**：`A3_p1_offload_draftgraph`（c0 = Phy-ID 8–15，全程持 `locks/c0.lock`）。
> **本文所有结论标【实测】/【推断】/【未确认】**；错误码、md5、行号、计数均经 `grep`/`md5sum` 核对。

---

## 0. 一句话

**Phase 1 的目标（卸载 × draft 入图）通过了，而且是「有反例臂」的通过** ——
两臂的卸载判据**逐字相同**、draft 入图有硬证据（`Wrapping draft = 8`，对照臂 `= 0`）。
⇒ **「DMA 完成」与「图重放读 KV」之间的时序风险未发生。**

★ **但顺手撞到一个比 Phase 1 更严重的 P0**：**`ENGRAM=1`（= A2 生产配置）与卸载池在 A3 上无法共存**，
而且**有三种互不重叠的失败模式**。本文 §3–§6 是这一部分。

★★★ **而它的适用范围必须收窄**：**A2 已被用户实测为通过**（8 进程并发注册 392 GiB 全过、零 `207001`，
在 Engram 加载状态下）⇒ **这条 P0 是「A3 特有」，不是「A2 上线的前提」**（详见 §6.2）。
★ 同一格在两台机器上给出**相反**结论 —— 这是本文最重要的方法论结论。

---

## 1. 方法与环境

| 项 | 值 |
|---|---|
| 机器 | `A3-node1`，8 卡 = **Phy-ID 8–15**，独占 `locks/c0.lock` |
| 模型 | `~/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq`（真权重、真 Engram、真 DSpark） |
| 镜像 | `quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3` |
| runner | `agents/R_8card_int8/scripts/run_arm_r8.sh`（md5 `cd93e30c7c1e464e49e63709b2e7d655`） |
| 影子包 | `~/projects/dsv41-upstream-pr/shadow-pkg`，`PATCH_MODE=mount` |

### 1.1 ★ 口径提醒（后人对齐用）

1. ★ **引擎日志 ≠ 启动器日志**（这对判据至关重要）：
   ```
   引擎（有 vLLM 输出）: $PKG/results/<RID>/serve.log          ← 判据在这里 grep
   启动器（只有 serve_a2 的话）: agents/R_8card_int8/logs/<TAG>.serve_a2.log
   ```
   `068 §1.1` 也独立记了同一条。本文所有判据数字均取自**引擎** `serve.log`。
2. ★ **`MAX_TOKENS=128`**（不是 runner 默认的 1）：`MAX_TOKENS=1` 时 `SpecDecoding` 只有 1 行、
   `Accepted:1/Drafted:10` ⇒ **A 恒 ~1.5，是无效口径**，无法回答「图有没有真在跑」。
   ⇒ **本轮的 P1 两臂都改成 128**（唯一改动）。代价：**fill sha 不能与 09:38 的历史臂跨臂对比**。
3. ★ **A3 远端时钟比本机慢 ~7 min**；而引擎日志自己的钟又比远端钟**慢 8 h**。
   凡引用日志钟处均标注换算（如 `09:49:19` → 远端 `17:49:19`）。

---

## 2. Phase 1 结论：通过（卸载 × draft 入图）

### 2.1 两条臂（同几何，唯一变量 = `DRAFT_GRAPH`）

```
臂 P1-A（对照）: TAG=p1a-tierB-dg0-offload  TIER=B GRAPH=1 EAGER=0 DRAFT_GRAPH=0 MAX_TOKENS=128
臂 P1-B（实验）: TAG=p1b-tierB-dg1-offload  TIER=B GRAPH=1 EAGER=0 DRAFT_GRAPH=1 MAX_TOKENS=128
其余逐字相同：OFFLOAD_BYTES=60666413056（56.5 GiB 记账）/ MAX_LEN=133120 / MAX_SEQS=32
              PROMPTS=16 × PROMPT_TOKENS=131072 → REPLAY 65536 / BPC={"default":8,"swa":1}
```

### 2.2 判据①：draft **真的**进了图（有反例臂，判别力成立）

```
判据                                                        P1-A(dg0)   P1-B(dg1)
"Wrapping draft model with ACLGraphWrapper"                   0      →      8     （= 8 rank）
runtime_mode=FULL                                             0      →      8
DRAFT-GUARD: dspark_proposer.py 含图捕获实现（命中 2 处）✓              ✓
DRAFT-GUARD: 容器内 DSPARK_GRAPH_CAPTURE_METADATA=1 ✓                  ✓
容器内四件套 CAPTURE_METADATA / CAPTURE_VALUE_FIX / SWA_INDICES_RESIDENT / CAPTURE_NCTX_FIX  全 = 1
```

★ **P1-A 的 `=0` 就是反例臂** ⇒ 这条判据**不是「看不见所以全 0」**（`AGENTS.md §5b` 第 3 条）。
★ 用的是 **`shadow-pkg` 自带的 draft 三件**（`patches/files/draft/`，`dspark_proposer.py` md5
**`5565afed64b7fe282fa9d622af7cf206`**）；对照 stock 版是 `dac256ad09b813c45628ec1ae03cfb19`
（后者第 75 行是无条件 `self.use_cuda_graph = False`）。
★ `serve_a2.sh:1025-1027` 在 `DRAFT_GRAPH=1` 时**无条件挂**这三个整文件（`PATCH_MODE=mount` 下即三个 `-v`），
且 `:1346-1357` 有 **DRAFT-GUARD**：容器内文件缺标记或 env ≠ 1 ⇒ **直接 `die`**
⇒ 8 卡链上「stock dspark + `DRAFT_GRAPH=1`」**起不了服**，不会静默通过。

### 2.3 判据②：卸载判据**逐字相同**（draft 入图对卸载零扰动）

| 判据 | P1-A | P1-B |
|---|---|---|
| `kv_offload_total_bytes_total{CPU_to_GPU}` | 21,519,269,888 | **21,519,269,888** |
| `kv_offload_total_bytes_total{GPU_to_CPU}` | 158,559,371,264 | **158,559,371,264** |
| `external_prefix_cache_hits_total` | 901,120 | **901,120** |
| `BlockStored:CPU`（KV 事件） | 29,436 | **29,436** |
| `BlockRemoved:GPU` | 201,846 | **201,846** |
| **`BlockRemoved:CPU`** | **0** | **0** |
| 宿主池（三条路径一致 `✅`） | 197.21 GiB | **197.21 GiB** |
| `GPU KV cache size` | 427,643 | **427,643** |

★ 这些数同时与 **`logs/022` / `048` 的历史臂逐字相同**（`hits=901,120`、池 `197.21 GiB`、
`BlockStored:CPU=29,436`）⇒ **P1-A 是干净可比的基线**。

### 2.4 判据③：性能（★ 已把 prefill 从 wall 里扣掉，看纯 decode）

| | P1-A(dg0) | P1-B(dg1) | Δ |
|---|---:|---:|---:|
| fill `Σttft`（对照，应不变） | 291.63 s | 291.55 s | −0.03% |
| ★ **fill `decode_only`** | 47.96 s | 45.62 s | **−4.88%** |
| replay `Σttft`（对照，应不变） | 22.84 s | 22.72 s | −0.53% |
| ★ **replay `decode_only`** | 16.79 s | 15.07 s | **−10.24%** |
| decode 吞吐（fill） | 33.8 tok/s | 35.6 tok/s | +5.3% |
| ★ decode 吞吐（replay） | 35.4 tok/s | 40.5 tok/s | **+14.4%** |

（`decode_only = wall − Σttft_均值 × n`；两轮 gen_tokens 1619/1622 与 595/611。）
⚠️ **诚实边界：每臂各 1 个样本。** Σttft 两臂差 0.03%/0.53% 说明环境可比，
但 −4.88%/−10.24% 若要钉死**建议再复跑一轮**。

### 2.5 判据④：稳态接受率健康（未撞静默失效）

```
P1-A: A = 3.26 / 3.33 / 4.33   (Avg Draft acc 45.2 / 46.5 / 66.7 %)
P1-B: A = 3.39 / 3.03 / 3.41   (Avg Draft acc 47.7 / 40.6 / 48.2 %)
```
均远高于 **1.0** 的静默失效指纹（`reports/draft-graph-negative-control.md`）。

### 2.6 顺带得到一条很干净的判据：**池大小不影响输出**

```
p1b-tierB-dg1-offload      (ENGRAM=0, DG=1, 池 56 GiB)  fill sha = a119d9f6a08b35bef092…
p2c-dg1-tinypool-noengram  (ENGRAM=0, DG=1, 池 1 MiB)   fill sha = a119d9f6a08b35bef092…
                                                                    ↑ 逐字相同（同 MAX_TOKENS=128）
```
⇒ **冷池（1 MiB，零命中）与热池（56 GiB，命中 901,120）给出逐字节相同的 fill 输出**，
跨 3 个数量级池尺寸 ⇒ 独立佐证 `048 §2` 的「fill 轮各臂逐字相同」。
★ 判别力：两个 **DG0** 臂的 sha 与它们**不同**（`p1a = 8cd9c27b…`）
⇒ 这条 sha **能区分「draft 入图 / draft eager」**。

### 2.7 未覆盖（如实标注）

* `concurrency=1`（`016` 的诚实边界；**P0-C 的 conc≥16 坏状态仍未触及**）
* **未加 int8**（那是 Phase 3；且 `APC_ALIGN=3` + `GRAPH_SAFE=1` 必须显式带）
* `ENGRAM=0`（见 §3 —— 而**这正是 A2 需要的那个轴**）

---

## 3. P0：`ENGRAM=1`（A2 生产配置）与卸载池**无法共存**

### 3.1 为什么这条比 Phase 1 重要

```
A2 生产默认： shadow-pkg/scripts/serve_a2.sh:127   ENGRAM=${ENGRAM:-1}   ⇒ 生产 Engram 是【开】的
交付脚本    ： a2/scripts/serve_a2_offload.sh:148  ENGRAM=${ENGRAM:-0}   ⇒ 照它上线会把 Engram 静默关掉
全部既有 8 卡臂（027/042/048/066/067）：ENGRAM=0 ⇒ 这一格【从未测过】
logs/016 诚实边界第 7 条原文：「本轮没有用 ENGRAM=1 复测」
```
⇒ ★ **这是「整套交付的生效前提」问题，不是参数标定问题。**

### 3.2 三臂对称对照（唯一变量分离，全部 1 MiB 池 + 同 workload）

| 臂 | ENGRAM | DRAFT_GRAPH | `ret=0` | 回落 | **EH0012** | **hdc disconnect** | 结果 |
|---|---|---|---|---|---|---|---|
| `r8-bc-tierB-graph-cold` | 0 | 0 | 64 | 64 | **0** | **0** | ✅ rc=0 |
| ★ **`p2c-dg1-tinypool-noengram`** | **0** | **1** | 64 | 64 | **0** | **0** | ✅ **rc=0** |
| `p2b-engram1-tinypool-dg1` | **1** | 1 | 64 | 64 | **2** | **10** | ⛔ 卡死 |

⇒ ★★ **`ENGRAM` 是唯一的判别变量**（`bc` 与 `p2c` 之间只差 `DRAFT_GRAPH`，两者都 ✅
⇒ **「`DRAFT_GRAPH` × 小池」这个交互不存在**）。

### 3.3 三种失败模式（**互不重叠**，且与 `logs/001 §4.2` 记的都不是同一个）

| 池 / 后端 | 起服 | 推理 | 失败码 | 栈的第一现场 |
|---|---|---|---|---|
| **56 GiB / registered** | ⛔ 捕获期挂 | — | **`207001` ×43** | `build_request_ids` → `torch.repeat_interleave` |
| **1 MiB / registered** | ✅（`health=200`） | ⛔ 首个长请求卡死 | **`EH0012` → `hdc disconnect` / `507901`** | kernel launch submit |
| ★ **56 GiB / pageable** | ✅（`health=200`） | ⛔ fill 轮 `failed=15` | **`EH0012` ×9 → `hdc disconnect`** | `aclnnInplaceCopy` |
| 1 MiB / registered + **`ENGRAM=0`** | ✅ | ✅ | — | — |

★ 第三种 `507899`/`100000`（只读 VMA）是**独立现象**，见 §3.5。
★★ **三种失败（`ENGRAM=1`）的共同点：`EH0012` 都出现（2 / 2 / 9 次），而三个 `ENGRAM=0` 臂都是 0** ⇒ 见 §3.8。

**1 MiB 臂的完整现象**（它比 56 GiB 臂更难发现）：
```
起服：✅ health=200 / Application startup complete / GPU KV cache size 427,643 / 207001 = 0
压测：warmup ✅（15.63 s）→ fill 轮第一个 131072-token 请求 ⇒ 引擎卡死
现象：/health 仍返回 200，但单个推理请求 90 s 超时无响应；Running: 0；客户端挂住 10 min
栈  ：rtsLaunchKernelWithHostArgs execution failed, reason=hdc disconnect
      Failed to check device status, device_id=10, retCode=0x7110011
      aclrtLaunchKernelWithHostArgs failed, return: 507901
涉及 4 个 rank（TP2/3/5/6）
```
⇒ ★ **「健康检查 200 但推理已死」** 是最坏的形态之一（监控看不见）。

### 3.4 `207001` 是「累计总量耗尽」，**不是「单次尺寸阈值」**（S1 被实测判死）

**逐尺寸对照**（同样 16 张张量 × 8 rank = 128 次尝试）：

```
尺寸(GiB)        ENGRAM=0（128 次）        ENGRAM=1（128 次）
0.00 / 0.01       24 / 8  ✅              24 / 8  ✅
0.18 / 0.35       24 / 8  ✅              24 / 8  ✅
1.41              24      ✅              24      ✅
2.82               8      ✅              3 ok /  5 fail
3.98               8      ✅              1 ok /  7 fail
4.24              24      ✅              5 ok / 19 fail
```
⇒ ★ **同样尺寸在 `ENGRAM=0` 下全成功、`ENGRAM=1` 下部分失败** ⇒ 阈值**不是**「单次尺寸」。

**逐 rank 的尝试序列（O = `ret=0`，X = 回落 pinned）**：
```
rank0  OOOOOOOOOXOOXXXO      rank4  OOOOOOOOOXOOOXXX
rank1  OOOOOOOOOOOOOXXX      rank5  OOOOOOOOOXOOOXXX
rank2  OOOOOOOOOXOOXXXX      rank6  OOOOOOOOOXOOXXXX
rank3  OOOOOOOOOOOOOXXX      rank7  OOOOOOOOOOOOOXXX
```
⇒ ★★ **8 个 rank 的失败位置几乎逐位一致**（都从第 10 次尝试开始、最后 3 次 8/8 全失败）
⇒ **单调的累计资源耗尽**，不是随机竞争。
⇒ **把 197 GiB 切成更多小块 ⇒ 总量不变 ⇒ 仍在同一处耗尽** ⇒
**「减单张量尺寸但保总量」这条路机制上无效**，只有**减少总注册量**有效。
（★ 精确说法是「**累计已注册总量对可用预算的耗尽**」，不是「与尺寸无关」——
小请求确实更容易成功。）

### 3.5 `507899` / `100000`：**与 Engram 无关的第三个现象**，且**不阻塞起服**

```
臂                                     507899   100000   207001   结果
r8-bc-tierB-graph-cold (ENGRAM=0)         32       32        0    ✅ rc=0
p2c-dg1-tinypool       (ENGRAM=0)         32       32        0    ✅ rc=0
p2b-engram1-tinypool   (ENGRAM=1)         32       32        0    ✅ 起服 / ⛔ 推理
```
⇒ ★ **`ENGRAM=0` 的臂也有一模一样的分布（32+32，失败尺寸 0.00 GiB = 1 MiB）**
⇒ 小分配的注册失败与 Engram **完全无关**（`507899` = 只读 VMA 被拒，
与 `L3_8card` 记的「只读 VMA 会被 `aclrtHostRegister` 拒绝：ret=507899」同码）。
★★ **它不阻塞起服，但它是一次静默的性能降级**：失败的张量回落 `pinned`，
那部分取回 H2D 从 **21 GB/s → 5.5 GB/s（慢 3.8×）**（`logs/065 §3c`）。
⇒ ★ **上线监测判据**：`grep -c "aclrtHostRegister failed" <serve.log>` **必须 = 0**。

### 3.6 `EH0012` 是 **ENGRAM 专属**，而且出现得比预期更早

```
两个 ENGRAM=0 臂：EH0012 = 0 / 0
两个 ENGRAM=1 臂：EH0012 = 2 / 2
```
**时间线**（日志钟 +8 h → 远端钟）：
```
56GiB 臂：engine 起 17:40:47 → EH0012 @ 17:49:08 / 17:49:19 → KV cache @ 17:50:25
                              → 崩 @ 17:52:09（repeat_interleave / 207001）
1MiB  臂：engine 起 ~17:58   → EH0012 @ 18:06:30          → KV cache @ 18:07:16
                              → 崩 @ 18:13:41（hdc disconnect / 507901）
```
原文：
```
Invalid_Argument(EH0012): aclrtAllocatorGetByStream failed. Parameter stream is invalid.
                          Reason: The stream is not registered with the allocator.
```
⇒ ★ **两个 Engram 臂都在「KV cache 建立之前」就打了 EH0012**，然后下游各自走向不同失败。
⇒ 【推断】（**不是实测**）：Engram 建立 device-index 时用了**没在 NPU allocator 注册的 stream**，
留下一个坏状态，下游表现取决于后续对 host 注册预算的争用程度。
★ **没有证据把 `207001` 归给 `EH0012`** —— `068` 的「注册预算耗尽」解释对 56 GiB 臂更直接，
**两种机制可能并存**。这条**必须靠实验分开**（见 §5.1）。

### 3.7 对 `068` 那条一行修法的判断（**必须分开验**）

`068` 定位：起服失败的直接原因是 Engram 自己的 `build_request_ids` →
`torch.repeat_interleave(index, counts)` 需要一个 **host 侧 D2H 小暂存**（`LocalScalarDenseNpu.cpp:23` + `copy_stream`）。
修法（草案）：`torch.repeat_interleave(..., output_size=N)`。

```
它对症的是「56 GiB 臂」  —— 栈就落在那                                   ✅
它不覆盖「1 MiB 臂」     —— 栈落在 kernel launch / hdc disconnect        ⚠️【推断】
★ 但要留一个乐观可能：若 EH0012 本身是那条 host 缓冲申请失败的更早一次表现，
  则同一个修法可能一起解掉 ⇒ 这正好可以用一条实验判定（§5.1）。
```
它对症的是「56 GiB 臂」  —— 栈就落在那                                   ✅
它不覆盖「1 MiB 臂」     —— 栈落在 kernel launch / hdc disconnect        ⚠️【推断】
★ 但要留一个乐观可能：若 EH0012 本身是那条 host 缓冲申请失败的更早一次表现，
  则同一个修法可能一起解掉 ⇒ 这正好可以用一条实验判定（§5.1）。
```

---


### 3.8 ★★★ 第四臂（`pageable` 后端）：**换后端不能解决，因为 `EH0012` 与后端无关**

**动机**：`publish/0002` 的 `cpu_npu.py:80` 已有 **`pageable`** 分支（**完全不调用 `aclrtHostRegister`**）
⇒ 若 `207001` 是"抢注册预算"，换后端应能彻底避开。**零代码改动**，一条臂即可判。

**做法**：`NPU_OFFLOAD_HOST_MEM=pageable` + `ENGRAM=1` + 池 56 GiB，其余与 `p2` 逐字相同。

**结果**：
```
★ 开关确实生效：`[P1_pinned] CPU pool backend = pageable (NPU_OFFLOAD_HOST_MEM)` × 8 rank（判据：必须打出这行）
★ 207001 = 0        ← 完全避开注册争用（预期内）
★ EH0012 = 9        ← ★ 仍然出现，而且比 registered 臂（2 次）更多
起服：✅ Application startup complete / GPU KV cache size 427,643
压测：⛔ fill 轮 failed=15 / replay1 failed=16，tok/s=0.0
栈  ：aclnnInplaceCopy → rtsLaunchKernelWithHostArgs execution failed, reason=hdc disconnect
```

#### 3.8.1 ★★★ 四臂合并对照表（`EH0012` 的判别条件被钉死）

| 臂 | ENGRAM | 后端 | `EH0012` | `207001` | 起服 | 推理 |
|---|---|---|---|---|---|---|
| `p1a` / `p1b` | 0 | registered | **0** / **0** | 0 | ✅ | ✅ |
| `p2c` | 0 | registered | **0** | 0 | ✅ | ✅ |
| `p2` | **1** | registered 56 GiB | **2** | 43 | ⛔ | — |
| `p2b` | **1** | registered 1 MiB | **2** | 0 | ✅ | ⛔ |
| ★ `p2d2` | **1** | **pageable** 56 GiB | **9** | **0** | ✅ | ⛔ |

⇒ ★★★ **结论：`EH0012` 是 `ENGRAM=1` 专属，且与 host 内存后端无关**
（`registered` 与 `pageable` 两种后端、56 GiB 与 1 MiB 两种池，**三种组合的 `ENGRAM=1` 全部出现 `EH0012`，
而三个 `ENGRAM=0` 臂全部为 0**）。
⇒ ★★ **这推翻了本日志早期的推断**（"`EH0012` 可能是注册争用的更早一次表现"）——
`pageable` 后端**一个字节都没注册**，`207001` 也是 0，`EH0012` **照样出现**。
⇒ ★ **也修正了 §3.7 对 `068` 的判断**：`068` 的修法（`repeat_interleave(output_size=)`）
针对的是 **`p2`（56 GiB registered）那一条栈**；而 **`EH0012 → hdc disconnect` 这条链与 `repeat_interleave` 无关**
（`p2d2` 的栈落在 `aclnnInplaceCopy`），**两条链需要分别修**。

#### 3.8.2 ★ 顺带确认：`EH0012` 仍然出现在 **KV cache 建立之前**
```
EH0012 @ 11:16:17 / 11:16:18（日志钟）→ 远端 19:16:17 / 19:16:18
GPU KV cache size 行 @ 11:17:06       → 远端 19:17:06
压测崩 @ 11:19:32                      → 远端 19:19:32
```
⇒ 与 `p2` / `p2b` **同一形态**（`EH0012` 早于 KV cache、崩溃在之后）⇒ 三个 `ENGRAM=1` 臂**完全一致的时序**。

#### 3.8.3 ★★ 自检门缺陷（本轮的判据失效第 7 条，已记入 §4）
`run_arm_r8.sh:247-248` 的硬门是 `P1_pinned.*ret=0 >= 8`，**按 `registered` 写的**；
而 `pageable` **压根不注册** ⇒ `ret=0` 恒为 0 ⇒ **引擎明明起好了却被自己的门杀 `exit 9`**（第一次尝试的 `p2d`）。
**修法**（**只改我自己的副本，不动共享文件**）：复制 runner 到
`agents/A3_p1_offload_draftgraph/scripts/run_arm_r8_pageable.sh`，在 248 行后加 5 行分支：
```
if [ "${NPU_OFFLOAD_HOST_MEM:-registered}" = "pageable" ]; then
  N_P1=$(grep -ac "CPU pool backend = pageable" "$SRVLOG" 2>/dev/null || true)
  N_P1RANK=$(grep -a "CPU pool backend = pageable" "$SRVLOG" 2>/dev/null | grep -aoE "Worker_TP[0-9]+" | sort -u | wc -l)
fi
```
（`diff` 实测：**唯一改动就是这 5 行**。）★ 这就是 §4 第 7 条 ——
**自检门必须能表达"换后端后的正确行为"**，否则它会把正确答案判成失败。

### 3.9 ★ `pinned` 臂：**按新证据跳过**（理由写明）

原计划复验 `logs/001 §4.2` 当年的 `pinned` 后端。§3.8 拿到"**`EH0012` 与后端无关**"后，
`pinned`（第三种后端）**预期同形**，且它是 `aclrtMallocHost` 路径、在 A2 上本来就是被规避的那条
（`014` 的结论：A2 不要用 `allocHost` 撑大池）⇒ **投入产出比低，本轮不做**，如实记在此处。


## 4. 判据失效清单（本轮踩到的，按 `AGENTS.md §5b` 归档）

| # | 失效形态 | 本轮实例 | 修法 |
|---|---|---|---|
| 1 | **grep 模式撞到无关日志** | `grep -c "falling back"` 命中的是 **Triton `causal_conv1d_update` 回落**（8 次），差点报成「池回落 8 次」 | 必须写**精确**模式 `falling back to pinned` |
| 2 | **同一条 grep 在失败行上不匹配** | `CPU pool ... (size)` **只在成功行**出现 ⇒ 用它当「尝试数」会得到 97，漏掉 31 次失败 | 判据用 **`ret=0 行数 = 128`** + **`ret=207001` 计数** |
| 3 | ★ **单臂有两个变量**（我自己犯的） | tiny 臂是 `ENGRAM=1 + DG=1`，初始对照臂 `r8-bc` 是 `ENGRAM=0 + DG=0` ⇒ 差两个轴 | 补 **`p2c`（ENGRAM=0 + DG=1）** ⇒ 三臂对称，结论才成立 |
| 4 | **「136」是两种东西之和** | `136 = 8（每 rank 一行 backend）+ 128（16 张量 × 8 rank）`；失败臂 105 = 8 + 97，而 **97 + 31 = 128** ⇒ 尝试全部发生 | 判据写 128，不写 136 |
| 5 | **`MAX_TOKENS=1` ⇒ A 无效口径** | `Accepted:1/Drafted:10` ⇒ A = 1.50，无法判断「图有没有跑」 | 用 `MAX_TOKENS=128`；两臂同步改 ⇒ 仍可比 |
| 6 | ★ **引擎日志 vs 启动器日志** | `<TAG>.serve_a2.log` 只有 ~177 行、**没有 vLLM 引擎输出** ⇒ 在那上面 grep 判据恒空 | 一律读 `$PKG/results/<RID>/serve.log` |
| 7 | ★★ **自检门把「换了后端的正确行为」当成失败** | `run_arm_r8.sh:247-248` 硬门 `P1_pinned.*ret=0 >= 8` 是按 `registered` 写的；而 **`pageable` 后端【压根不注册】** ⇒ `ret=0` 恒为 0 ⇒ **引擎明明起好了（`startup_enabled`/health=200/KV=427,643/backend=pageable 8/8）却被自己的门杀掉 `exit 9`** | 复制 runner 到自己的目录，加一个 backend 分支（见 §3.8）—— **不动共享文件** |

---

## 5. 对 `logs/001 §4.2` 的**第二次改写**

```
001 §4.2（原始）：ENGRAM=1 必挂 / ENGRAM=0 必通；根因 = Engram 的 host 注册与卸载层的
                 pinned 分配在抢同一个驱动侧资源；拦路虎「目前无解」
014 §2（第一次改写）：「不是容量公式问题」（207001 更像态问题，未拿到最小复现）
★ 本文（第二次改写，全部【实测】）：
  ① 根因是【大张量的 aclrtHostRegister 拿不到驱动侧资源】，且【池越小越安全】
     —— 1 MiB 池完全避开 207001（0 次），56 GiB 池 31 次失败
  ② 它是【累计总量耗尽】，不是单次尺寸阈值（§3.4 逐 rank 序列）
  ③ ★ 失败的张量【回落成 pinned，池容量一个字节都没少】
     ⇒ 正确表述是「快路径只覆盖一部分、其余慢 3.8×」，不是「装不下」
  ④ ★ 而且 ENGRAM=1 还有第二种失败（小池：起服成功但首个长请求 ⇒ hdc disconnect）
     ⇒ 「把池缩小就能共存」这条【也不成立】（1 MiB 池的推理仍然崩）
```

★ 因此 `001 §4.2` 的「拦路虎」**没有解除**，只是形状更清楚了：
**`ENGRAM=1` 与卸载池在 A3 上目前是二选一**（除非 §5.1 的修法成立）。

---

## 6. 下一步（按价值排序）

### 6.1 ★★ 两条链要分别修（§3.8 把这个判断改精确了）

```
链① 207001（只对 registered + 大池）：
   栈 = build_request_ids → torch.repeat_interleave
   修法 = 068 的 output_size= / 设备侧 searchsorted       ✅ 对症
   判据 = ret=0 行数 = 128 且 207001 = 0 且起服成功

链② EH0012 → hdc disconnect（★ 对所有 ENGRAM=1 组合）：
   栈 = aclrtAllocatorGetByStream: stream is not registered with the allocator
        → rtsLaunchKernelWithHostArgs / aclnnInplaceCopy → hdc disconnect（507901）
   修法 = ★ 未知（068 的修法不覆盖它 —— 见 §3.7 / §3.8.1）
   判据 = EH0012 = 0 且 fill 轮跑完（failed=0）

★ 两条都过 ⇒ Engram 可共存 ⇒ A3 上也成立
★ 只有①过 ⇒ 仍然「能起服但推理崩」
★ 两条都不过 ⇒ A3 上必须在「卸载」与「Engram」之间二选一（★ 用户决策）
★ ⚠️ 但**对 A2 而言这一步不是前提** —— 见 §6.2（A2 已实测通过）
```

★ **最省的下一条实验**：**`ENGRAM=1` + 无卸载池**（8 卡、同几何）。
它能把"链②**是否依赖池**"分开：
```
若也崩 ⇒ 链② 是「Engram + 8 卡 + 图」本身的问题（与池无关）⇒ 修点在 Engram
若正常 ⇒ 链② 是「Engram × 池」的组合问题 ⇒ 修点在两者交互
★ 需要一个小改：我的 runner 副本里把 --kv-transfer-config 摘掉（本轮未做，留给下一位）
```

### 6.2 ★★★ A2 已被用户实测：**通过**（本节是最重要的适用范围更正）

**2026-09-22 晚，用户在 A2 生产容器上跑了三期并发注册探针**（`a2/scripts/a2_multiproc_reg_probe.sh`，
8 进程在 barrier 处**同时**注册，含 256 MiB 真实 H2D→D2H 逐字节对账；**服务未停机**，
容器内 **Engram 处于加载状态**）：

```
S1  8 ×  4 GiB =  32 GiB   注册 8/8   设备往返 8/8   22.8 s
S2  8 × 16 GiB = 128 GiB   注册 8/8   设备往返 8/8   36.8 s
★ S3 8 × 49 GiB = 392 GiB  注册 8/8   设备往返 8/8  104.6 s
自检：rss_before 1.77 → rss_after 50.3 GiB（ΔRSS ≈ 分配量 ⇒ 分配即常驻，与 030 §7 一致）
```

⇒ ★★★ **A2 上 Engram 与注册池可以共存**（392 GiB 全过、零 `207001`）：
```
A3（host_mem_pool = 1）：ENGRAM=1 ⇒ 只注册 74.68 GiB，31 次 207001，起服失败
★ A2（host_mem_pool = 0）：ENGRAM=1 ⇒ 392 GiB 全过，零 207001
```
⇒ ★★ **本文 §3 的 P0 适用范围必须收窄为「A3 的驱动 / `host_mem_pool` 组合」**，
**不是「A2 上线的前提」**。A2 需要的量是 ~37 GiB 池（16 worker 视角 49 GiB），**S3 已经覆盖并超过**。

#### 6.2.1 ★ 「注册路线的性能是池规模的函数」（新，两机共同规律）

A2 的 S1/S2/S3 里 **H2D 随池规模下降**：
```
S1  32 GiB: H2D 中位 ~20.5 GB/s （8 个 rank 里 2 个慢）
S2 128 GiB: H2D 中位 ~19.2      （1 个慢）
★S3 392 GiB: H2D 中位 ~5.0      （★ 6 个慢）
```
★ 机制【推断】：`392 GiB / 4 KiB = 每进程 ~1280 万个页表项 × 8` ⇒ 地址翻译压力。
★★ 与本轮 §3.5 那条（`-> falling back to pinned` ⇒ H2D 21→5.5 GB/s，**零报错**）是**同一族**：
```
★ 结论：「ret=0 / 注册成功」只保证【功能】；H2D 带宽会随【注册总量】下降
   ⇒ 判据不能只看"成功没成功"，还要看"快不快"
```

#### 6.2.2 A2 与 A3 **不可互推**（本文的一条方法论结论）

`068 §3.1` 标了两个【未确认】，其中**第一条更关键**：
> Engram 的 183.11 GiB 表是 **rank0 一份**还是 **8 rank 各一份**？
> `model.py::_engram_device_setup` 里**只有 print 被 `_bp_rank_zero()` 门控**，真实映射**没有**门控；
> 若 R=1 ⇒ 上限 L≈258 GiB；若 R=8（1464.6 GiB）⇒ L≈1539 GiB。**两者都与现有观测相容。**

⇒ ★ **同一格在两台机器上可以给出相反结论**，理由三条：
```
① A3 host_mem_pool = 1，A2 = 0（更严）⇒ A2 可能更差
② 但 A2 实测：单进程单块 1/8/32/64 GiB 全过（logs/065 §3c）＋ 8 进程并发 392 GiB 全过（本节）
   ⇒ A2 比 A3 更宽 —— 与 ① 不矛盾：`host_mem_pool` 是能力位，它并不预测注册预算的方向
③ 瓶颈是【驱动侧注册资源】，不是宿主内存（A3 MemAvailable 1665 GiB 仍失败）
```
★★ **口径纪律（以后必须这么写）**：
```
【实测·A3】...（适用范围：A3 / host_mem_pool=1）
【实测·A2】...（A2 / host_mem_pool=0）
★ 禁止把 A3 的"失败"当成 A2 的"前提"，反之亦然 —— 本轮已经出现两次相反结论。
```

### 6.3 池容量与 1M 口径的账（★ 按 §5 ③ 的更正重算）

```
1 个 1M 会话 = 24,064 unit = 24.0 GiB 记账 = 82.0 GiB 宿主（DELIVERY §6.5.2）
ENGRAM=1 + 56 GiB 臂：注册成功 74.68 GiB（= 池的 62%），其余回落 pinned
⇒ ★ 正确表述：82 GiB 的会话【装得下】（池的 197 GiB 容量一字节没少），
   但它的快/慢取决于 KV 落在池的哪一段 —— 62% 走 21 GB/s、38% 走 5.5 GB/s
⇒ 若最终必须缩小池总量：068 外推 OFFLOAD_GB ≲ 21 才能让 ret=0 达到 128
   ★ 那将直接推翻 A2 的 1M 计划（OFFLOAD_GB=85）⇒ 【这条结论比阈值本身更重要】
```

---

## 7. 原始数据（在 A3-node1 上）

```
引擎 serve.log（判据来源）
  ~/projects/dsv41-upstream-pr/shadow-pkg/results/
    r8_p1a-tierB-dg0-offload_20260922_170611/serve.log
    r8_p1b-tierB-dg1-offload_20260922_172245/serve.log
    r8_p2-engram1-tierB-dg1-offload_20260922_174005/serve.log          ← 56 GiB 池，207001
    r8_p2b-engram1-tinypool-dg1_20260922_175509/serve.log              ← 1 MiB 池，EH0012/hdc
    r8_p2c-dg1-tinypool-noengram_20260922_182554/serve.log             ← ★ 分离臂，✅
    r8_r8-bc-tierB-graph-cold_20260922_131559/serve.log                ← 历史对照
证据目录（client.json / pool_bytes / kv_events / metrics / trace）
  ~/projects/dsv41-upstream-pr/agents/R_8card_int8/out/<TAG>/
```

### 7.1 本日累计跑的臂（7 条，含 1 条被自己的门误杀）

```
P1-A  （ENGRAM=0, DG=0, registered 56 GiB）  ✅  rc=0
P1-B  （ENGRAM=0, DG=1, registered 56 GiB）  ✅  rc=0   ← ★ draft 入图，卸载判据逐字相同
p2    （ENGRAM=1, DG=1, registered 56 GiB）  ⛔  捕获期 207001×43
p2b   （ENGRAM=1, DG=1, registered 1 MiB）   ⛔  起服成功、首个长请求 hdc disconnect
p2c   （ENGRAM=0, DG=1, registered 1 MiB）   ✅  rc=0   ← ★ 分离臂
p2d   （ENGRAM=1, DG=1, pageable  56 GiB）   ⛔ **被自检门误杀 exit 9**（门按 registered 写的）
★ p2d2（ENGRAM=1, DG=1, pageable   56 GiB）  ⛔  起服成功、fill 轮 failed=15（EH0012×9）
```
现场：容器已全清、`c0.lock` 已交还、宿主 `/dev/shm` 归零（1007 G 全空）。
★ `pinned` 后端臂**未跑**，理由见 §3.9。

---

## 8. 与 `067`（单卡）的关系

`067` 在**单卡 tiny 几何**上独立测了同一个组合，结论是「Q1 能跑且 draft 真的进图 / Q2 对卸载无差异」，
并抓到一处**只属于 tiny 几何**的「g12 参与卸载 ⇒ 取回整轮归零」。
★ 本文（8 卡真权重）与 `067` 在 **draft 入图可用**、**对卸载无扰动**两点上**独立一致**；
★ 但 8 卡上**取回路径是好的**（`CPU→GPU = 21.52 GB`、`hits = 901,120`）⇒
`067` 那个 tiny 专属现象**不能外推到 8 卡**（`067 §Q2` 自己也标了「规模/几何」差异）。