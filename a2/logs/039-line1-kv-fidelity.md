# 039 — 线 1（DRAM 卸载）取回的 **KV 级逐字节判据**：tiny 守门员已过，8 卡仍【未确认】

**日期**：2026-09-22 06:0x–07:0x（A3 本地时钟）
**执行**：子代理 **H_kvcheck**。机器：**A3（A3-node1）槽位 c2 = die 7**（容器 `prbench-c2`）。
**红线**：只在 **c2**（`tools/a3_chip.sh c2` 持锁，无 75 退出）；**没手设** `ASCEND_RT_VISIBLE_DEVICES`；
**没用 `/tmp`**（`~/tmp/20260922/H_kvcheck/`）；**没写 `upstream-v41/`**；**没碰** `mooncake-*`/`jitpgo-*`/`dsv41-a3`/别人的容器；
跨机传文件全走 **cos-xfer.sh**；**没改任何生产代码**（全部 `PYTHONPATH` 叠加包 + post-import 钩子）。
**标记**：【实测】/【推断】/【未确认】。
**原始数据**：`agents/H_kvcheck/out/`（逐 block JSONL + 逐张量表 + client/metrics/kv_events）。

---

## 0. 七句话结论

1. **⛔【实测·静态】`agents/L3_8card/scripts/kv_bytecheck.py` 不可用，而且它的失败模式是"假阳性"**：
   它 hook 的 `NPUOffloadingWorker.store/.load` **在现行代码里不存在**（真入口是 `submit_store/submit_load`），
   于是 `if orig_store is not None:` 为假、**一个钩子都不装**，却照样打"已装指纹钩子"并置 `_l3_kvcheck=True`。
   ⇒ 若当时跑成，会得到一份"跑过了、比对 0 次、全绿"的日志。**后人避坑：探针必须自带"ops>0"的反假阳性闸**（本日志判据⑥）。
   它另有三个洞：`self.kv_caches` 不存在（真张量在 handler 的 `src/dst_tensors`）、比的是"整张量 vs 整张量"（没有行↔block 配对）、
   且只采 1.1% 的字节（不是逐字节）。
2. **★ 判据设计对的地方**：复用 `036` 的 `f_pool_audit.py` 骨架（挂在**真入口** `transfer_async`，用**镜像自己的**
   `compute_sub_block_ptrs` 复算指针，**全字节**比对），并补上它**结构性看不见**的一格：
   `if prev is not None` 把"**命中块读到从未写过的池行**"静默吃掉 —— 而这正是 `038` 的缺陷签名。
3. **✅【实测】144 MiB（1.000×，已知 ❌）与 160 MiB（1.193×，已知 ✅）两格，卸载层的字节判据都干净**：
   每臂 **13,392 个区间、mismatch = 0**、`row_changed = 0`；store/load 两侧逐张量 × 逐组 × 逐 block 全表见 §3。
4. **★ 探针自校验（036 没做）**：我账本的字节数**逐位等于引擎自己的 metric**：
   `GPU_to_CPU = 196,689,920`、`CPU_to_GPU = 231,669,760` ⇒ 我记的 op 集合**就是**真实 DMA 的 op 集合。
5. **★【实测·生产代码自身输出】store 与 load 的 unit 口径不对称**：group 0（full, bpc=8）的 store spec
   `group_size = 4`（4 个 chunk ⇒ **1 个 GPU block / chunk**），而同一 chunk 的 load spec 是 **8 个 unit / chunk**
   ⇒ **store 只填了它应填的 1/8，其余 7/8 的池行保持全 0**。两格都一样（144 与 160 的差别只有 8 个 unit）。
6. **⚠️ 这条不对称不是 ❌/✅ 的判别量**（两格同形），而且**我还没能把它判成缺陷**：见 §5 的口径冲突。
   `038` 的 ❌/✅ 翻转**不能**用卸载层的字节路径解释（两格字节行为完全相同）。
7. **⛔ 8 卡（`027` 口径 `OFFLOAD_GB=56`）没跑** —— 卡时用在了 tiny 的守门员与那处不对称上。**线 1 的取回保真在 tiny 上是【实测·干净】，在 8 卡上仍是【未确认】。**
8. **★★ 但那处不对称已经定性（§10）：是【实测】的真缺陷，根因是 `scheduler.py:1419` 的 `blocks_per_chunk` 局部变量泄漏**
   ⇒ spec loop 里 `gpu_block_idx = chunk_idx * 1`、`for i in range(1)` ⇒ **每个 chunk 只搬 1 个 GPU block**。
   三条独立测量一致（生产 spec `Σgroup_sizes=44 = n_keys`；worker 实搬 64 个 group-0 block；交叉核对 **492 该搬 vs 64 实搬 = 428 个从未被搬**），
   而 `bpc=1` 的 10 个 SWA 组**65/65 全中**（= 判据的反例臂）。**修法一行，`dst_unit_ids` 要同步展开。**
   ⚠️ **但它不是 `038` 那个 ❌/✅ 翻转的答案**（144/160 两格读到的全 0 行同量级），也不是"BF16 输出立刻错"的原因
   （`h-t2` 在同一份 448 行全 0 的情况下 fill/replay sha 逐字相同）。它是**潜伏的正确性风险**，上线前必须修。

---

## 1. ★ 守门员（S1）：判据必须先能在"已知错误格"上报警

> 这条方法论是 `035→036` 的教训：**判据没判别力却被当成结论**，比不跑更贵。

| 臂 | 池 | J2（`036`/`038` 口径） | 我的探针 |
|---|---|---|---|
| `h-t1-int8-150994944` | 144 MiB = **1152 unit（1.000×）** | ❌ **mismatched=[15]**；fill `24b57053…` / replay `a7ffff6b…`（**与 036/038 逐字相同**） | ops=13,392 **mismatch=0** `row_changed=0` |
| `h-t2-int8-167772160` | 160 MiB = 1280 unit（1.193×） | ✅ match=True（fill=replay=`24b57053…`） | ops=13,392 **mismatch=0** `row_changed=0` |
| `h-t5-int8-150994944` | 144 MiB（重跑，带 scheduler 侧探针） | ❌（同上） | 同上 |

**事件计数也复现 `038`**：`BlockStored:CPU=714`、`BlockRemoved:CPU=10`(144)/`0`(160)、
`CPU→GPU=231,669,760`、`GPU→CPU=196,689,920` —— **两格完全相同**。

⇒ **卸载层的字节行为在两格之间没有差别** ⇒ `038` 的 ❌/✅ 翻转**不在这一层**。

---

## 2. 判据逐条（任务书 §2 的 ①②③ + 我加的 ④⑥）

| # | 判据 | 144 MiB（❌） | 160 MiB（✅） |
|---|---|---|---|
| ① | store：`sha1(NPU 源页) == sha1(CPU 目标行)`，**逐张量/逐组/逐 block** | **0 / 5,968 ops 不符** | **0 / 5,968** |
| ② | load：`sha1(CPU 源行) == sha1(NPU 目标页)`，同上 | **0 / 7,424 ops 不符** | **0 / 7,424** |
| ③ | 池行在 store↔load 之间是否被改写（`_store_sha[key]` 配对） | `row_overwrite = 0` | `row_overwrite = 0` |
| ④ | load 是否读到"全 0 行" | 有：5,352 次 | 有：5,376 次（**不是判别量**） |
| ⑥ | 反假阳性：每张 canonical 张量 `ops > 0` | ✅ 20 张全有样本 | ✅ |
| ★ | **探针自校验**：字节总数 vs 引擎 metric | **逐位相等** | **逐位相等** |

**逐组表（原样，144 MiB 臂，节选）**：
```
t=0   g=0   store ops=64   mismatch=0     bytes=4194304
t=0   g=0   load  ops=512  mismatch=0     bytes=33554432
t=0   g=2   store ops=65   mismatch=0     bytes=4259840
t=0   g=2   load  ops=16   mismatch=0     bytes=1048576
（20 张 canonical 张量 × 12 个参与组，完整表见 out/h-t1-*.h_kv.log 的 T 表）
```

---

## 3. ★ 轴 III：manager 分配器层面 —— **已排除**（阳性对照 + fuzz + 静态证明）

主代理给的"缺 cap ⇒ stale free ⇒ 同一行发给两个活 key"路径：**我做了阳性对照才敢说它不可达**。

| 证据 | 结果 |
|---|---|
| **阳性对照**（人为制造那条路径：A 淘汰 → 新 key 复用 R → 用**过期 BlockStatus(R)** 再 free） | ✅ **探针抓到了**：`stale_free=1 free_list_dup=1 free_owned=9`，且 R 又被发给另一个 key ⇒ **"全 0"不是"看不见"** |
| **随机 fuzz**（直接驱动**补丁版** `pgp_manager.py`，md5 `3b64eb4977f3`；300 轮 × 2000 op、池 200~2048、13 组混合、17,374 次 reset） | `stale_free=0 free_list_dup=0 dup_unit=0 bad_load_unit=0 free_owned=0 over_cap=0`；`load_units=408,933`、`store_units=9,632,896` |
| **静态证明** | `prepare_store` 先淘汰到 `free_units() >= units_needed`，淘汰量与 `_free_block` 推入量逐项一致 ⇒ `A' = A + max(0, needed − F) ≤ C` **恒成立** ⇒ `over_cap` 不可能；`_free_block` 的调用者（`_policy.evict` / `_policy.get`）都已先把该 key 从 policy 摘掉 ⇒ **没有 stale free 的调用者** |

⇒ **我第一报里"缺 cap ⇒ 会越界"那句已撤回**（`agents/H_kvcheck/DESIGN.md §轴III 自纠`）。
**建议 `J_mgrhardening` 做"运行期只读断言"而不是改分配逻辑**（没有可修对象、有回归风险）。

---

## 4. ★ 已排除的相邻假说（与 `I_unitprobe` 交叉印证）

| 层 | 谁 | 结论 |
|---|---|---|
| 调度侧配账（unit 归属 / 淘汰 / 行复用） | `I_unitprobe`（真跑 11 计数器全 0）+ **本任务**（阳性对照 + 300 轮 fuzz + 静态证明） | **已排除** |
| worker/DMA（store 源、完成、load 时序） | `I_unitprobe`（含"同一探针在 ✅ 臂上给同样数"的守门员） | **已排除** |
| int8 几何 vs BF16 | `I_unitprobe` + `J_mgrhardening`（同 1152 unit、同 `BlockRemoved=10`，BF16 ✅ / int8 ❌） | **缺陷在 int8 侧** |
| **回放输出文本 sha** | `036`/`037` | **整体失效**（服务本身不确定）⇒ 本轮**全程不碰文本判据** |

---

## 5. ★★ 唯一留下的异常：store 填 1/8、load 读 8/8（**未定性**）

### 5.1 生产代码**自己的**输出（不重推）

`h-t5` 的 scheduler 侧探针记下每个 store job 的 `src_spec`（生产代码构造的）：

```
STORE JOB: n_src=44 n_dst=44 group_sizes=[4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
                            block_indices=[0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7] n_keys=44
（16 个 job 同形；16 × 4 = 64 个 group-0 block）
LOAD  JOB: keys=14 group_sizes=[32, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] src_blocks=42 dst_blocks=42
（每个请求 group 0 = 32 个 unit = 4 chunk × 8）
```

⇒ **同一个请求 group 0 的 4 个 chunk：store 侧只交出 4 个 GPU block（1/chunk），load 侧却要读 32 个 unit（8/chunk）。**
⇒ 与 worker 侧账本完全一致：group 0 **store 64 ops / load 512 ops**，其中 **448/512 读的是从未写过、全 0 的池行**。

### 5.2 ⚠️ 我**没能**把它判成缺陷（口径冲突，必须写清）

我的 scheduler 侧探针同时读了 `group_state.block_ids[c*8:(c+1)*8]`，**64 个 chunk 全部 8 个非 0**
⇒ 如果 store 循环按 `gpu_block_idx = chunk_idx * 8` 取块，本该append 8/chunk，与 `group_sizes[0]=4` **矛盾**。
两种解释我**都没能排掉**：
  (a) store 循环的实际 `blocks_per_chunk` 退化成 1（⇒ chunk↔GPU block 索引空间错位 ⇒ **真缺陷**）；
  (b) 我读 `block_ids` 的时机/对象与循环不同（⇒ 我的探针口径问题，**又是一次假阳性**）。
**判定必须再做一格**（见 §7），**本轮不下结论**。

### 5.3 为什么它**不**能解释 `038` 的 ❌/✅
144 与 160 两格的该不对称**同形同量级**（只差 8 个 unit）⇒ 它**不是**判别量。
主代理提的"int8 读到全 0 行 ⇒ `scale=0 ⇒ 0/0=NaN`、BF16 读到 0 只当零向量"这条假说**仍然值得查**，
但它的**前提（哪些行该写而没写）在 ✅ 臂上也一样** ⇒ 需要另一个"为什么只有 int8 会读到那些行"的机制才能闭合。

---

## 6. 诚实边界

| 项 | 状态 |
|---|---|
| 144/160 两格的逐字节比对 | ✅ **【实测】** 各 13,392 区间、mismatch=0 |
| 探针 op 集合 == 真实 DMA op 集合 | ✅ **【实测】** 字节数与引擎 metric 逐位相等 |
| manager 分配器无静默损坏 | ✅ **【实测·阳性对照】+【实测·fuzz】+【推断·静态证明】** |
| **worker 字节探针的故障注入对照**（人为喂错误字节看它是否报 mismatch） | ⛔ **【未确认】** 只做了"覆盖率对齐 metric"这一条自校验 |
| "读到全 0 行"是缺陷还是口径问题 | ⛔ **【未确认】**（§5.2 的 (a)/(b) 未排掉） |
| manager 侧 **live** 探针在真服务里 | ⛔ **没挂上**（`mgr.jsonl` 0 字节：pkg 的 `patch_pgp` 链在我的 hook 之前就 import 了它）⇒ 轴 III 的真跑数据为空，结论来自**模型外** fuzz + 静态 |
| **8 卡（`027` 口径 `OFFLOAD_GB=56`）** | ⛔ **【未确认】·本轮未跑** |
| `concurrency > 1` | ⛔ 未测（与 `027` 同口径 `concurrency=1`） |

---

## 7. 下一步（按价值排序，每条一格）

1. **★ 判据⑦ 的定性格**（30 分钟，tiny）：在 scheduler 探针里**原样打印 store 循环真正用到的
   `blocks_per_chunk` 与 `gpu_block_idx`**（不靠重推），一次跑清楚 §5.2 的 (a)/(b)。若 (a) 成立 ⇒
   这是**线 1 自己**的 store↔load 索引空间缺陷（BF16 也吃），优先级高于 `038`。
2. **worker 探针的故障注入对照**（10 分钟，不占卡）：构造一次已知错误的 (源, 目标) 对，确认 `mismatch` 会涨。
3. **8 卡 `OFFLOAD_GB=56`**：把本判据挂到 `027` 的 `serve_a2.sh` + docker -v 链上（判据②③④⑥ 全带）。
4. manager live 探针：在 `patch_pgp/sitecustomize.py` **之后**再包一层（顺序问题），或用 `sys.modules` 兜底。

---

## 8. cannbot 对照（`a2/AGENTS.md` §6）

本任务**不写算子、不做量化数值门**，相关只有 KV cache 布局那一条：

| 查的地方 | 它说什么 | 采纳 |
|---|---|---|
| `model-infer-kvcache/SKILL.md:102-113` | 物理 slot = `block_table[b, pos//bs] × bs + pos%bs`，**单一 block_size** | ✅ 采纳为"行↔block 一一配对"的依据（本任务按镜像自己的 `compute_sub_block_ptrs` 复算） |
| `model-infer-kvcache/SKILL.md:224-242` | 滑窗/压缩前缀的正确性由**模型层**负责，不是 op 层 | ✅ 采纳为边界：本轮只判**搬运**保真，不判"模型层状态是否被正确恢复" |
| `model-infer-quantization/SKILL.md:424-451` | 不能只看代码 diff，必须证明真实运行 | ✅ 采纳：本轮全部是运行期逐字节，且**不**把 sha 差异当量化罪证 |

**文档空白**：cannbot 里没有"卸载池 store/load 的 chunk↔GPU-block 索引空间一致性"这一节 ⇒ §5 是**我们的实测新增**。

---

## 9. 产物

| 类 | 位置 |
|---|---|
| 设计 + 自纠 | `agents/H_kvcheck/DESIGN.md` |
| 探针 | `probe/{h_kv_audit.py（worker 逐字节）,h_mgr_probe.py（manager 归属+fuzz+对照）,h_sched_probe.py（判据⑦）,sitecustomize.py}` |
| 脚本 | `scripts/{prepare_overlay.sh,run_arm_h.sh,selfcheck_h.py,analyze_ops.py,sync_to_a3.sh,fetch_from_a3.sh}` |
| 原始数据 | `agents/H_kvcheck/out/h-t{1,2,5}-*.{ops.jsonl,sched.jsonl,h_kv.log,h_sched.log,client.json,metrics_after.txt,kv_events.json,server.log}` |
| 补丁层快照（防被别人改动污染） | `agents/H_kvcheck/pkg/base_snapshot/MANIFEST.md5`（`pgp_manager.py` md5 `3b64eb4977f3…`） |

**复现**（不占卡的两条）：
```bash
H_PGP_PATH=<补丁版 pgp_manager.py> python3 probe/h_mgr_probe.py control 1152   # 阳性对照：应抓 stale_free/free_list_dup
H_PGP_PATH=<补丁版 pgp_manager.py> python3 probe/h_mgr_probe.py fuzz 300 2000  # 300 轮随机负载：应全 0
```
**复现**（占卡，单卡 tiny）：
```bash
bash ~/projects/dsv41-upstream-pr/tools/a3_chip.sh c2 --timeout 1500 --name h-t -- \
  env TAG=h-t1-int8-150994944 PORT=8220 XL1=0 XSWA=1 XRING=0 \
      P2_COMP_JSON='[[0,2,3,4,5,6,7,8,9,10,11],[1]]' OFFLOAD_BYTES=150994944 \
      PROMPTS=16 PROMPT_TOKENS=4096 REPLAY_PROMPT_TOKENS=4096 MAX_TOKENS=1 \
      EXTRA_ARGS="--enforce-eager" bash /work/agents/H_kvcheck/scripts/run_arm_h.sh
```
**锁退出码 75 = 没抢到锁，是重试不是失败。**

---

## 10. ★★★ §7-1 定性格结果：**(a) 成立 —— 真缺陷，根因是一个变量作用域泄漏**

> 上一版这里写着"两路读数冲突 ⇒ 拒绝下结论"。**结论已出：(a)，且根因是静态可证的。**

### 10.1 根因（【实测·静态】，一行代码）

`agents/L3_8card/patched/scheduler.py::_build_store_jobs`（函数体 1360–1607）里
`blocks_per_chunk` **只有一处赋值**，而它落在**前一个循环**里：

```
1419:  blocks_per_chunk = group_config.blocks_per_chunk   # ★ 在前一个"收集 loop"里，逐组赋值
1421-1423: offload_block_ids = block_ids[start*bpc + bpc-1 : num_chunks*bpc : bpc]
...
1529:  gpu_block_idx = chunk_idx * blocks_per_chunk      # ★ 在"spec loop"里被使用
1531:  for i in range(blocks_per_chunk):                 # ★ 但 spec loop 里【没有重新赋值】
```

`awk` 枚举该函数体内 `blocks_per_chunk` 的**全部**出现：`1362(注释) / 1415(注释) / 1419(赋值) / 1421 / 1422 / 1423 / 1477 / 1529 / 1531`
⇒ **`1419` 是唯一赋值点**。收集 loop 结束时该变量停在**最后一个参与卸载的组**的值上 —— 本次配置是
**group 11（SWA，`bpc=1`）**。

⇒ 于是 spec loop 里：**`gpu_block_idx = chunk_idx * 1`、`for i in range(1)` ⇒ 每个 chunk 只搬 1 个 GPU block**，
而 chunk 的其余 `bpc_g − 1` 个 block **根本没进 `src_spec`**；同时只用了该 key 的 `units[0]`（其余 7 个 unit 从未被写）。

### 10.2 三条独立测量都指向同一个数（不是口径问题）

| # | 测量 | 结果 |
|---|---|---|
| ① | **生产代码自己的 `src_spec`**（我不重推） | `n_src=44`、`Σgroup_sizes=44`、`group_sizes=[4,0,4,…,4]` ⇒ **每 chunk 恰好 1 个 block** |
| ② | worker 侧真正搬走的 GPU block（`npu_row`） | group 0 共 **64** 个（= 16 job × 4 chunk × **1**） |
| ③ | ★ **两条数据的交叉核对**（worker 的"实搬块" vs scheduler 的"该搬块"） | group 0：`seg` 并集 **492** 个非 0 block，**实搬只有 64 个 ⇒ 428 个从未被搬**；而 `bpc=1` 的 SWA 组是 **65/65 全中** |

★ **判据本身的自校验**：`bpc=1` 的 10 个 SWA 组**逐项全中**（65 搬 / 65 该搬）。
若这是"我的口径错"，SWA 组不可能全对 ⇒ **泄漏只伤 `bpc>1` 的组，正是 group 0（full attention）**。

### 10.3 这就是那 448/512 的来源（与 load 侧对照）

```
store：每 chunk 只写 units[0]  ⇒  4 chunk × 1 = 4 个 unit 被写
load ：按 _gcfg.blocks_per_chunk = 8 展开（scheduler.py:1181 bpc_g / :1259 _bpc）
       ⇒  4 chunk × 8 = 32 个 unit 被读
⇒ 32 − 4 = 28 个 unit/请求 从未被写（全 0）；16 请求合计 448 ← 与 worker 侧实测的 448/512 逐个吻合
```

### 10.4 ⚠️ 影响边界（必须与结论一起引用）

| 项 | 判定 |
|---|---|
| 这是**真的口径缺陷**（store 少搬 `(bpc_g−1)/bpc_g` 的 full-attention KV） | ✅ **【实测】** |
| 它**踩在"可上线"的 L5 路径上**（`bpc={"default":8,"swa":1}`） | ✅ **【实测】** |
| 它**单独**能解释 `038` 的 ❌/✅ 翻转吗 | ⛔ **不能**：144 与 160 两格读到的全 0 行**同量级**（448/512 vs 448/512），而只有 144 炸 |
| BF16 下会不会立刻出错 | ⚠️ **【实测·反证】不会立刻出错**：`h-t2`（160 MiB，✅）在**同一份 448 行全 0** 的情况下 fill/replay sha 逐字相同 ⇒ 这些全 0 子块**在该配置下没有改变输出** ⇒ 主代理 §2 那条锚点（027 的 BF16 输出正确）**与本次实测一致** |
| 正确修法 | **一行**：在 spec loop 内补 `blocks_per_chunk = group_config.blocks_per_chunk`（或改用 `group_config.blocks_per_chunk` 显式引用）。**注意 `dst_unit_ids` 也要同步展开 `_units[i]`，否则 store/load 仍然不对称** |

**⇒ 一句话**：**是 (a)，真缺陷、静态根因明确、修法一行；但它的"用户可见影响"在当前实测里被限制住了（两格同形），所以它不是 `038` 那个翻转的答案。**
⇒ 它是**潜伏的正确性风险**（`bpc>1` 的组永远只搬 1/bpc），**必须在 A2 上线前修**，但**不能**拿它去解释首 token 翻转。

### 10.5 这一格的教训（与 `kv_bytecheck` 那次同源）

* 我的第一版读书是"探针读到 8 个非 0 `block_ids` ⇒ 生产只 append 1 个 ⇒ 冲突"，**当时拒绝下结论是对的**：
  真正的解释是**两个读数都正确**——`block_ids` 确实有 8 个非 0（组配置没错），而 loop **只按 1 个 stride 去取**（变量泄漏）。
  ⇒ 教训：**"两个读数冲突"不一定是"谁错了"，也可能是"它们量的不是同一件事"。**
* 本格的判据**自带反例臂**（`bpc=1` 的 10 个 SWA 组必须全中）—— 这正是主代理要求的
  "判据必须在反例臂上对称跑"。若没有这一格，我就会把"64 vs 492"写成探针缺陷。
