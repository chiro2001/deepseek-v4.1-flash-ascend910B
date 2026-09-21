# 043 — 卸载层 DMA 拷贝长度：**机制在代码事实层成立，但因果被实测推翻**（候选修法 1 = no-op）

2026-09-22 07:15–07:35 CST。执行：子代理 **L_dmafix**。机器：**A3（A3-node1）槽位 c2 = die 7**
（容器 `prbench-c2`）。全程只用 c2，**没碰 c0（`K_l1_8card`）/ c1（`J_mgrhardening`）/ 别人容器**；
**没写 `upstream-v41/`**；**没用 `/tmp`**（本地只写 `~/tmp/20260922/l_dmafix/`）；
占卡全走 `tools/a3_chip.sh`（无 75 退出，全程 1 次取锁）；**没手设 `ASCEND_RT_VISIBLE_DEVICES`**；
传文件走 `cos-xfer.sh`；**没改任何一行生产代码**（全部 overlay + import hook）。

> **一句话**：`cpu_npu.py` 的拷贝长度确实取自 `data_ref.page_size_bytes`、指针步长确实取自
> `tensor.stride(0)`（**代码事实成立**）；但 **`state` 组（group 1）根本不参与卸载** ⇒ 改它是
> **纯 no-op**（D 几何实测：改前/改后 replay sha **逐字相同**，DMA 层 `len_changed=0`）；
> 而且**按字面实现会越界**。⇒ **修这一行不能解锁 ×1.4655 / ×1.9133**。

---

## 0. 结论（全文最重要的一段）

### 0.1 ★★★ 代码事实：两个量**确实不同源**（任务书 §(1) 的问题，答案是"成立"）

| 量 | 取自 | 位置 |
|---|---|---|
| **拷贝长度** | `data_ref.page_size_bytes` | `a2/publish/0002-…patch.py:235`（镜像 `…/native/cpu_npu.py:185`）：`all_sizes[op_idx:end_idx] = data_ref.page_size_bytes` → 交给 `torch.ops._C_ascend.swap_blocks_batch`（`cpu_npu.py:200-206`） |
| **指针步长** | `tensor.stride(0)` | 上游 `vllm/v1/kv_offload/cpu/gpu_worker.py:110-123`（`row_stride = tensor.stride(0)`；fast path 第 114 行 `base_ptr + block_ids*row_stride`），由 `cpu_npu.py:25/162/169` 调用 |
| **`page_size_bytes` 的来源** | 单 Tensor 分支 = `layer_spec.unpadded_page_size_bytes` | `…/native/offloading_connector.py:219-226`（`add_view`）+ 第 258-272 行 |
| **该 view 的 stride 来源** | `block_stride_bytes = layer_cache.stride(0)*element_size` | `…/native/offloading_connector.py:169-179`（`as_strided(raw,(num_blocks,page_size_bytes),(block_stride_bytes,1))`） |

### 0.2 ★★★ 但 `state` 组**不参与卸载** ⇒ 该处的"缺口"从不被 DMA 走到

三条臂（D / F / C0-int8）的起服日志**逐字相同**地打印：

```
[D2_offload] 参与卸载的组：full_attention=[0] sliding_window=[2..11]；被排除的组=[1]
```

* `store-probe` 里 **group=1 出现 0 次**（`040` §7 + 本任务复核）；
* P2 池日志 `quota={1: 0}`、`rows[tensor12..14]=0`（D/F）⇒ 这些张量**页数被算成 0**；
* ★ **本任务的 DMA 侧实测**（下表）：ROWCHK 34 条里 **`src/dst_t12/13/14` 出现 0 次**。

### 0.3 ★★★ 候选修法 1 = **实测 no-op**（DMA 层 + 输出层 双证）

| 臂 | 几何 | `L_DMA_COPYLEN` | kv size | replay sha | J2 | DMA 探针 |
|---|---|---|---|---|---|---|
| `l-b2-D-page` | D（4 杠杆） | `page`（现状） | **33,295** | **`6a47dd65f1ff`** | ❌ 14/16 | ROWCHK **34** |
| **`l-b3-D-stride`** | **D 同几何** | **`stride`（候选 1）** | **33,295** | **`6a47dd65f1ff`（逐字相同）** | ❌ **14/16（逐字相同）** | ROWCHK **34**、`len_changed=0` |

* mismatch 集合也逐字相同：`[0,2,3,4,5,6,7,8,10,11,12,13,14,15]`；
* **四条判据（prometheus）也逐字相同**：`CPU→GPU = 231,669,760 B`、`GPU→CPU = 196,689,920 B`、
  `external_prefix_cache_hits_total = 65,520`（两臂完全一致）；
* **DMA 层解释**：D 几何里**每一个真的被拷贝的张量都满足 `L ≤ shape[1] ≤ stride`**
  ⇒ 我的保守上限 `min(stride, shape[1])` **拒绝了每一次改动** ⇒ `COPYLEN` 命中 **0 次**。

```
D2H（读 GPU→写 CPU）：t0/t3/t6 rows=2907 shape1=65,536 stride=73,856 L∈{128,1024,8192,65536}
                      t9       rows=2907 shape1=131,072 stride=147,712 L∈{1024,16384,65536,131072}
                      全部 over_stride=False over_shape1=False
H2D（读 CPU→写 GPU）：t0..t19 全部 shape1 == stride == L
```

### 0.4 ★★ 反例臂也存在同样的 `copy < stride` ⇒ **该判据无判别力**（§5b 第 3 条的又一次应用）

| 臂 | J2 | 有"copy < stride"的张量吗 | 具体 |
|---|---|---|---|
| `035` `x-A` | **✅ `mismatched=[]`** | **有** | tensor 15（**参与卸载**、被大量 store）：`page=147,712`/`stride=147,712`（字节）而 `copy=131,072` |
| `036` C0 臂 | ✅ | **有** | SWA scale 张量：`page=copy=1,024`、`strideB=131,072` ⇒ 差 **128×** |
| `040` D/F | ❌ | 有（但**只在被排除的 group 1**） | 66,560/73,856 vs 65,536 |

⇒ **"拷贝长度 < 行步长 ⇒ 尾部静默截断"不是判别量**：C0 的 scale 差 128 倍仍然 ✅。
真正的设计语义是 **"copy = 本层自己的平面大小"**（槽页里叠着别的层的平面，所以 `copy < page == stride` 是**正常**的）。

### 0.5 ★★ 为什么 C0 恰好相等（任务书 §(1) 的对照问题）

C0 = `L5 + SWA-quant + ring **FP32**`：ring 平面 131,072 B 自己就是槽里**最大的**平面
⇒ 槽容量 = ring 页 ⇒ `unpadded == padded == stride == 131,072`（`core/deepseek_v41.py:210`
把 `page_size_padded` 设成 placement 的页大小；`_cache_plane_sizes` 见 `deepseek_v41.py:122-127`）。
D/F 是 ring **FP16**（65,536）之后**槽容量由别的平面决定**（SWA 66,560 / kv+index 73,856）
⇒ `padded > unpadded` ⇒ 只有 state 组出现 `copy < page`，**而它恰好是被排除卸载的那组**。

### 0.6 ★★★ 越界算术：按字面实现候选 1 **会越界** ⇒ **不要做**

CPU 池张量的行宽 = `npu_page_size_bytes * blocks_per_chunk`（`cpu_npu.py:308-318`），即 **按 `page` 分配**。
于是把拷贝长度改成 `stride(0)`：

| 张量 | page | stride | 新 L | 后果 |
|---|---:|---:|---:|---|
| `x-A` tensor 0（kv 平面） | 65,536 | 131,072 | 131,072 | 每行多写 **65,536 B = 整行** ⇒ 越界到下一行；最后一行越界出张量 |
| `036` C0 SWA scale | 1,024 | 131,072 | 131,072 | **128× 越界** |
| D 几何 state（**若**参与卸载） | 65,536 | 73,856 | 73,856 | 每行多写 8,320 B ⇒ 越界 |

**两条安全条件（写进自检）**：

1. **NPU 侧恒成立**：`(rows-1)*stride + L ≤ rows*stride` ⇐ `L ≤ stride`（视图躺在整块 slab 里，storage 更大）；
2. **CPU 侧必须 `L ≤ page`**（池按 page 分配）⇒ **`L = min(stride, page/shape[1])` 才是安全上限**；
   一旦 `stride > page`，任何"用 stride 当拷贝长度"的写法都**必须先扩大池的行宽**（= 候选修法 3，改动面大）。

⇒ 本任务把修法实现成 `L ← min(stride, shape[1], storage 上界)`（`probe/l_dmafix_probe.py` 的 `_inspect`），
**默认关**（`L_DMA_COPYLEN=page`），并在开关下跑到 D 几何 ⇒ 结果就是 §0.3 的 no-op。

### 0.7 五句话总结

1. **代码事实成立**：拷贝长度取 `page_size_bytes`、指针步长取 `stride(0)`，两者在 D/F 的 state 组上差 1,024 / 8,320 B；
2. **但 `state` 组不参与卸载**（`被排除的组=[1]`、`quota={1:0}`、`rows=0`、store-probe group=1 **0 次**，且**本任务 DMA 侧 ROWCHK 0 次**）⇒ 改它没有效果；
3. **实测 no-op**：`copylen=page` vs `copylen=stride` 在 D 几何上 **replay sha / mismatch 集合 / 四条告警指标全部逐字相同**（`6a47dd65f1ff`）；
4. **该判据无判别力**：`x-A`（✅）与 C0（✅）同样存在 `copy < stride`（后者差 128×）；
5. **按字面实现会越界**（§0.6 算术）⇒ **候选修法 1 与 2 一并关闭**，剩余候选只有 `036` §3.3 的 **`kv8_ori_plane` decode 形状重建**。

### 0.8 ★ 回答任务书的"是否解锁"

> **修这一行不能解锁 ×1.4655 / ×1.9133。**
> 证据：`l-b2-D-page` 与 `l-b3-D-stride` 的 `replay_out_sha256_all` **`6a47dd65f1ff` 逐字相同**、
> `sha256_mismatched_prompts` 都是 `[0,2,3,4,5,6,7,8,10,11,12,13,14,15]`、
> DMA 层 `len_changed=0`（探针连一次改动都没发出）。

---

## 1. 这一轮新增的实测（本任务 6 条臂）

| 臂 | 几何 | 池 | eager | 拷贝长度 | kv size | replay sha | J2 | 守门员 |
|---|---|---|---|---|---|---|---|---|
| `l-b1-C0` | C0（L5+L1+SWA-q，ring F32） | 144 MiB | 否 | `page` | 22,719 | `24b570535f58` | **✅ 0/16** | C0 回归没坏 |
| `l-b2-D-page` | **D（4 杠杆）** | 144 MiB | 否 | `page` | **33,295** | **`6a47dd65f1ff`** | ❌ 14/16 | 复现 `035`/`036`/`040` |
| **`l-b3-D-stride`** | **D 同几何** | 144 MiB | 否 | **`stride`** | **33,295** | **`6a47dd65f1ff`** | ❌ **14/16** | ★ **no-op** |
| `l-b4-D-ring-hot` | D 同几何 | 144 MiB（命中） | **是** | `page` | 33,295 | `6a47dd65f1ff` | ❌ 14/16 | 与 b2 逐字一致 |
| **`l-b5-D-ring-cold`** | **D 同几何** | **1 MiB（不命中）** | **是** | `page` | **33,295** | `24b570535f58` | **✅ 0/16** | ★ **几何本身没问题** |
| `l-dma-D-page` | D | 144 MiB | 否 | `page` | —（起服失败） | — | — | 见 §2（挂载事故） |

* **`l-b4` vs `l-b5`**：同几何、同 eager、同 `GPU KV cache size=33,295`，**唯一变量 = 池是否命中**
  ⇒ 独立复现 `036` 的"命中 ⇒ ❌ / 冷算 ⇒ ✅"；
* 所有臂的 **fill sha 都是 `24b570535f58…`**（与 `036`/`038`/`040` 逐字相同）⇒ 可比；
* `l-b2`/`l-b3` 的四条 prometheus 判据逐字相同（§0.3）。

---

## 2. ★ 挂载事故与根因（15 min，**值得留档**）

**症状**：`l-dma-D-page` 起服失败 →
`AttributeError: module '…native.cpu_npu' has no attribute 'P2OffloadingWorker'`。

**第一反应（错的）**：以为自己的 overlay 覆盖了 L1 的 `cpu_npu.py`。**实测否掉**：

```
pkg-ring/shadow/…/native/cpu_npu.py 里 "P2OffloadingWorker" 出现次数 = 0
pkg-ring/patch/p2_hooks.py          里 出现次数 = 5（第 465 行：cpu_npu_module.P2OffloadingWorker = worker_cls）
```

⇒ **`P2OffloadingWorker` 不在任何文件里，是 import 期动态挂上去的类**。
**真因**：我的 `sys.meta_path` finder 排在 P2 的 finder **前面**，并且**直接返回了 spec**
⇒ import 系统**停在第一个非 None 的 finder** ⇒ P2/PGP/KV8 的 post-import 钩子**永不执行**。
**修法**（抄 P2 `sitecustomize.py:63-84` 的写法）：`find_spec` 里**先 `sys.meta_path.remove(self)`**、
`exec_module` 里再 remove 一次。

★ **教训（给所有探针）**：往 `sys.meta_path` 插 finder 时，
**"先摘自己再 find_spec、exec 后摘自己"是唯一正确写法**；
否则你不是"叠加一层"，而是**悄悄关掉所有别人的 import 钩子**（症状会伪装成"别人的补丁坏了"）。

---

## 3. 探针纪律（§5b）在本任务的落地

### 3.1 探针自查四次，每次都抓到一个"看不见"的坑

| # | 坑 | 症状 | 修法 |
|---|---|---|---|
| 1 | finder 抢钩子 | `P2OffloadingWorker` 缺失、起服失败 | §2 |
| 2 | **`dma_ops=0` 的假绿** | 探针"已装"、`SUMMARY` 全 0 ⇒ 看着像"没有 DMA" | ★ 加 **PROGRESS/ROWCHK** 逐张量打印 ⇒ 看到真实热路径 |
| 3 | **`atexit` 不跑** | EngineCore 被 `kill -TERM`（超时再 `-KILL`）⇒ `SUMMARY` 丢失，落盘的全是**别的进程**的 `dma_ops=0` | ★ 装 `SIGTERM/SIGINT/SIGHUP` 处理器，被杀前先落盘（`_install_signal_dump`） |
| 4 | **误报越界**（★ 假阳性） | 第一版用 `numel*itemsize` 当上界，报了一条 `ANOM dma_oob_dst H2D t=9` | 这些张量是 `as_strided` **子视图**（`stride*rows > numel`）⇒ 上界必须用 **`untyped_storage().nbytes()`** |

★ **#4 是 §5b 第 3 条的又一次应用**：如果我不做**行算术**、直接把它当"越界 bug"上报，
就会得到一个**比原 bug 更严重的假警报**。修完上界后，34 条真实拷贝上 `oob_src/oob_dst = 0`。

### 3.2 判据的对称性（本任务做了什么、没做什么）

| 判据 | 阳性对照（应报警） | 对称臂（应干净） | 结论 |
|---|---|---|---|
| DMA 拷贝长度 no-op | `l-b2`（❌ 14/16） | `l-b3`（同 ❌、sha 相同）⇒ "改了等于没改" | ✅ 有效 |
| `state` 张量 0 次 DMA | `l-b1-C0`（34 条真实 DMA） | 同探针在 D 上 `t12/13/14` **0 条** | ✅ 有判别力的 0 |
| 池命中 vs 冷算 | `l-b4`（❌ 14/16） | `l-b5`（✅ 0/16） | ✅ 独立复现 |
| **ring provenance** | — | — | ⛔ **未跑出（见 §4）** |

---

## 4. 【未确认】ring provenance 探针：装了但**没触发**

* `l-b4`/`l-b5` 的 ring 探针**确实装上了**（5 个进程都打了
  `装载完成：DMA=False(copylen=page) RING=True`），但 **`ring_calls=0`**、
  没有任何 `RING call=` 行 ⇒ **`compressor_from_projected` 在这条 eager 路径上一次都没被调到**；
* ⇒ 按 §5b 第 2 条，**这两格的 ring 读数没有判别力**，**本任务不把它当结论**；
* ⇒ 主代理"池命中 ⇒ state ring 没被重建"的假说 **仍未被验证**（也未否证）；
* **下一条命令（未做，留给接手的人）**：把钩子从 triton 模块的符号
  （`ops/triton/compressor/compressor_triton.py:641`）**上移到 `models/deepseek_v41/compressor.py:113`
  的 `DeepseekV41Compressor.pool_projected` 调用点**，并在臂上先确认
  `★ ring 探针**已生效**` 横幅；然后按 §3.2 的对称表跑 `l-b4`（命中）/`l-b5`（冷算）。

---

## 5. 没做的事 / 明确的"不要做"

| # | 事项 | 状态 |
|---|---|---|
| 1 | 候选修法 1（拷贝长度 ← `stride(0)`） | ⛔ **实测 no-op，且按字面实现越界 ⇒ 关闭**（§0.3/§0.6） |
| 2 | 候选修法 2（把行间 padding 清零/填充） | ⛔ **不必做**：被拷贝的路径上 `L == shape[1]`，没有"padding 参与了 DMA"这回事 |
| 3 | 候选修法 3（让 state 槽页 = 平面和） | ⚠️ **未评估**；且 state 组不参与卸载 ⇒ 对本缺陷**无直接收益**（只影响容量口径） |
| 4 | **F 几何的 no-op 复核** | ⛔ **未跑**（预算给了 C0 + D 两档 + 冷算守门员）。**F ≠ 新机制**：F 里 state 组（t16/17/18）同样被排除、`quota=0`/`rows=0` ⇒ D 的结论**机制上直接覆盖 F**；但**口径**上 F 的 `GPU KV cache size=43,469` 未在本任务复测 ⇒ 标 **【未确认】** |
| 5 | ring provenance（本任务的真凶候选） | ⛔ **未跑出**（§4） |
| 6 | `x-E` 的 4096→2048 变长前缀安全 | ⛔ **未测**（`036` §6-6 的替代判据仍有效） |

---

## 6. cannbot 对照（AGENTS.md §6）

本任务**不写 kernel、不做量化数值验证**（只做卸载层字节级判据），因此只查了一条：

| 查的地方 | 它说什么 | 采纳 / 没采纳 |
|---|---|---|
| `model-infer-kvcache/SKILL.md:102-113`（PA 映射：`物理 slot = block_table[b, pos // block_size] * block_size + pos % block_size`） | 单一 `block_size` 的页映射 | ✅ **逐字采纳**（本任务一行没改生产代码）。★ 但本任务实测到它没覆盖的一条：**同一个"页大小/步长"概念在卸载层有 4 个不同的量**（`shape[1]` / `unpadded` / `padded(槽页)` / DMA 的 `copy`），**它们不必相等**，而"相等"才是**特例**（C0）而不是通例 ⇒ 建议把"四量对账表"写进 KV 卸载的验证清单 |

---

## 7. 交付

| 类 | 位置 |
|---|---|
| 本文 | `a2/logs/043-20260922-dma-copylen-fix.md` |
| 原始数据 | `a2/logs/raw/043-l-dmafix/`（**56 个文件 / 1.4 MB**：5 条臂的 `probe.log`/`meta.txt`/`selfcheck.txt`/`kv_size.txt`/`metrics_*`/`client.json`/`kv_events.json` + 起服失败日志） |
| 代码 | `a2/agents/L_dmafix/`：`probe/{sitecustomize.py,l_dmafix_probe.py}`、`scripts/{prepare_overlay.sh,selfcheck_l.py,run_arm_l.sh,run_batch_l.sh,upload.sh}` |
| COS | `share/xfer/l_dmafix/{l_dmafix.tgz,043.tgz,043-clients.tgz}`（容器内 `/work/agents/L_dmafix/raw/`） |
| 容器内 | `/work/agents/L_dmafix/{probe,scripts,out,raw}`（A3 A3-node1，overlay 在 `/work/agents/L_dmafix/pkg/`） |

### 7.1 复现（每条 = 一条命令）

```bash
# ① 全批次（C0 守门员 → D no-op 对 → eager 命中/冷算对）≈ 11 min
bash tools/a3_chip.sh c2 --timeout 2200 --name l-batch -- \
  bash /work/agents/L_dmafix/scripts/run_batch_l.sh

# ② 单臂写法（候选修法 1）
bash tools/a3_chip.sh c2 --timeout 2000 --name l-d -- \
  env TAG=l-manual-D-stride POOL_BYTES=150994944 XSWA=1 XRING=1 XL1=1 XKV8PF=0 PORT=8480 \
  L_DMA_COPYLEN=stride bash /work/agents/L_dmafix/scripts/run_arm_l.sh

# ③ 探针 env 全表
#   L_DMA_PROBE=1      装 DMA 探针（默认开）
#   L_DMA_COPYLEN=page|stride   拷贝长度口径（默认 page = 逐字现状）
#   L_RING_PROBE=0/1   ring provenance（默认关；需 eager 臂）★ 本任务未跑出（§4）
#   L_PROBE_OUT=...    探针输出路径
```

**锁退出码 75 = 没抢到锁，是重试不是失败。** 每条臂自带"探针生效性自检"：
没有 `dma_ops>0` 的 `SUMMARY` 就显式写 **"本臂的 DMA 结论无效"**。

### 7.2 红线核对

没发 PR / issue / 评论；没写 `upstream-v41/`；**没用 `/tmp`**（本地只写 `~/tmp/20260922/l_dmafix/`，
容器内只写 `/work/agents/L_dmafix/`）；占卡全走 `a3_chip.sh` c2（**没抢 c0/c1**）；
**没手设 `ASCEND_RT_VISIBLE_DEVICES`**；没碰 `mooncake-*` / `jitpgo-*` / `dsv41-a3` / 别人容器；
起服前都查了 `df -h /dev/shm`（64 MiB，全程 <1%）；A3 `MemAvailable` 全程 > 1.5 TiB；
传文件走 `cos-xfer.sh`（**唯一偏差**：把容器内 tar 拉到宿主机时用了一次 `scp`，
回传到 A3 已改用 cos-xfer；如实记录）；**没改任何一行生产代码**；
结论逐条标了【实测】/【推断】/【未确认】。

### 7.3 给交付包的一句话

**"DMA 拷贝长度"这条歧路已被两个层面同时关掉**（sha 逐字不变 + `len_changed=0` + 越界算术），
§0.6 的"不要做"清单可直接引用；**线 2（×1.4655 / ×1.9133）仍未解锁**，
剩余唯一站得住的候选是 `036` §3.3 的 `kv8_ori_plane` decode 形状重建
（本任务的 ring 假说**未验证**，见 §4）。
