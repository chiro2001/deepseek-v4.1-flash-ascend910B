# 036 — int8 平面"不保真"否决点：**池往返逐字节保真**，真凶是"形状相关的读侧"

> 2026-09-22 04:1x–05:1x CST。执行：子代理 **F_fidelity**。机器：**A3（A3-node1）槽位 c2 = die 7**
> （容器 `prbench-c2`）。全程只用 c2，**没碰 c0**（`L3_8card` 04:33 才释放，本任务不需要 8 卡）、
> 没碰 `dsv41-a3` / `mooncake-*` / `jitpgo-*` / 他人容器；**没写 `upstream-v41/` 的日志与脚本**
> （代码只写 `a2/agents/F_fidelity/`）；没用 `/tmp`；跨机传输全走 coscli；
> 占卡全走 `tools/a3_chip.sh` 锁，**没有手设 `ASCEND_RT_VISIBLE_DEVICES`**。
> 代码：`a2/agents/F_fidelity/{probe,scripts}/`（与 `raw/036-f-fidelity/{probe,scripts}` 同一份）。

---

## 0. 六句话结论

1. **⛔【实测·推翻 `035` §5 的表述】"int8 平面经 DRAM 池往返不保真"这句话本身是错的。**
   池的 store/load **逐字节保真**：**12,928 次**拷贝（含 128 / 256 / 1024 / 8192 / 16384 / 65536 / 131072 B
   七档尺寸、行距 73856 / 147712），**mismatch = 0**；池行被后续覆盖 **0 次**。
2. **⛔【实测】attention 真正读到的字节，池臂与冷臂完全相同**：池臂 960 条"本步要读的页"
   记录（物理页号 + int8 sha + scale sha）**960/960 在冷臂里逐字节找到**，反向多余 0 条。
3. **⛔【实测】不是竞态**：把每次 DMA 改成"提交后立刻等完成事件"（`F_SYNC_DMA=1`），
   J2 仍是 ❌（1/16、13/16，sha 逐字不变）⇒ 不是"GPU 页在 DMA 读走前被覆写"。
4. **✅【实测·新判据】触发条件是"几何 + 形状"，不是"池"**：把 prompt 长度取成
   **block_size（128）的非整数倍**（4095 / 2047）⇒ 池臂也必须用 prefill 形状补算尾部一块
   ⇒ **J2 立刻变 ✅ 16/16**（`f8-T-4095`、`f8-T-L1-2047`）；而 4096 / 2048 是 ❌（1/16、13/16）。
   **两臂唯一的差别就是"首 token 由 1 行 decode 还是由多行 prefill 产生"。**
5. **【实测】差异是确定性的，不是 `037` 的随机抖动**：`f3-C0-3r` 的
   `replay1_sha == replay2_sha == a7ffff6b…`；`f-D` 两跑逐字复现 `035` 的 `6a47dd65…`。
   ⇒ **判据 ② 成立：这是"可重复的错"，不是噪声**（`037` 在单卡 tiny 上未复现）。
6. **⇒ A2 裁决要改**：`035` 的"池往返不保真"**不能作为 int8 两条杠杆的否决理由**；
   真正的阻塞点是**int8 SWA 读路径对调用形状敏感**（【推断】`kv8_ori_plane` 的 decode 形状重建）。
   **修法在 attention 读侧，不在卸载层**；`L5+L1` 的"逐字节保真"结论不受影响。

---

## 1. 任务书 (1)：静态定位 —— **核心假说被推翻（候选 1 不成立）**

### 1.1 逐张对比表（16 张 / 20 张两套几何）

口径：`page` = canonical tensor 的 `shape[1]`（= worker 的 `npu_page_size_bytes`）；
`copy` = ref 的 `page_size_bytes`（= DMA 的 size）；`strideB` = 按**字节**换算的
`tensor.stride(0) × itemsize`。数据来源 = `035` raw 的 `结构` 打印（x-A / x-C0 / x-D 三臂）。

**几何 A（L5，SWA=BF16；A/B/R/R2 = J2 ✅）16 张**

| 张量 | page | copy | strideB | 说明 |
|---:|---:|---:|---:|---|
| 0..8 | 65536 / 8192 / 128 | = page | **131072** | slot0–2 的 kv / index / index-scale（packed：步长 = 槽页） |
| 9..11 | 131072 / 16384 / 256 | = page | 147712 | slot3 同上 |
| 12..14 | 131072 | 131072 | 131072 | **state ring（F32）**，被 10 个 SWA 组当 SWA 面引用 |
| 15 | 147712 | 131072 | 147712 | slot3 的 SWA alias |

**几何 C0（L5+SWA-q，ring F32；J2 ❌ 1/16）20 张**

| 张量 | page | copy | strideB | 说明 |
|---:|---:|---:|---:|---|
| 0..8 | 65536 / 8192 / 128 | = page | 131072 | 0/3/6 = **SWA payload（层 0/1/2）被并进同一张 canonical tensor** |
| 9..11 | 131072 / 16384 / 256 | = page | 147712 | slot3 |
| 12..14 | 131072 | 131072 | 131072 | state ring（**被排除卸载**） |
| **15/16/17** | **1024** | 1024 | **131072** | **SWA scale（新增）** |
| 18/19 | 65536 / 1024 | = page | 147712 | slot3 的 SWA payload + scale |

**几何 D（L5+L1+SWA-q+ring16；J2 ❌ 14/16）**：同 C0，但槽页 131072 → **73856**，
state 张量 12..14 的 page = 73856（copy = 65536）、strideB = 73856。
**几何 F（5 条杠杆；J2 ❌ 15/16）**：long-KV 也 int8+scale ⇒ SWA payload 与 state **并成一张**。

### 1.2 判定：**候选 1 ❌ 不成立**（机制被推翻，不是"没查"）

1. **worker 用的是 `tensor.stride(0)`，不是 `page_size_bytes`。** 容器内直接读镜像源码：
   `/vllm-workspace/vllm/vllm/v1/kv_offload/cpu/gpu_worker.py::compute_sub_block_ptrs` →
   `row_stride = tensor.stride(0)`，指针 = `base_ptr + block_id × stride(0)`；
   `page_size_bytes` 只当**拷贝长度**（`all_sizes[...] = data_ref.page_size_bytes`）。
   `cpu_npu.py` 里只有一句 `assert cpu_tensor.shape[1] == npu_tensor.shape[1] * blocks_per_chunk`
   （双向都成立，不参与寻址）。
2. **"payload 131072 vs scale 65536" 是"元素步长 vs 字节步长"搞混了。** `reshape_cache` 的 stride
   以**元素**为单位：payload(int8) 131072 × 1 B = 131072 B；scale(fp16) 65536 × **2 B** = **131072 B**。
   `x-trace` 的 dump 自证：scale 的 `off=761856` 元素 × 2 = 1523712 B
   = payload 的 1458176 + 65536 = **平面串行**，与 `reshape_cache` 一致。
3. ⇒ **行 N 不会落错字节**。`page < stride`（int8 几何下 12/20 张成立）只是
   "拷贝长度 < 行距"的差，**合法**。

### 1.3 顺带查的 KV8 写侧（任务书 §(1) 第二问）

`kv8_store_rows`（`attention/dsa_v41.py`）在 `values.shape[0]` 非 0 时把
`(payload, scale)` 分两次 `scatter_cache_sk` 写进**同一个页**；scale 面**不是独立张量**，
它是 `reshape_cache` 里同一 slot 的第二个 view（`offset + plane_sizes[0]`）。
**写侧没有"把 scale 当独立张量"的问题**（`035` 候选 2 在这一层不成立；
真正独立出来的是**卸载池的 canonical tensor**，见 §1.1 的 15/16/17）。

---

## 2. 任务书 (2)：字节级判据 —— **池不丢字节**

### 2.1 探针（自写，只读，零行为改动）

`probe/f_pool_audit.py` 包住 `SingleDirectionNPUOffloadingHandler.{__init__,transfer_async,get_finished}`：
用**镜像自己的** `compute_sub_block_ptrs` 复算出每次拷贝的 `(tensor, src_ptr, dst_ptr, size)`，
然后把 **NPU 侧 `size` 字节** 与 **CPU 侧 `size` 字节**各搬下来做全字节比较（全等才算过）；
并把 `(tensor, 行号) → sha1(CPU 行)` 存起来，load 时再比一次（查"池行被覆盖"）。
`probe/sitecustomize.py` 用"先逐字 exec pkg 里那份 sitecustomize"的方式叠在
`P2 → PGP` 补丁链**之上**，**一行别人的代码都没改**。

### 2.2 【实测】主表：`f-D-audit3`（= x-D 几何，16 × 4096，池 144 MiB，**J2 ❌ 14/16**）

| tensor | ops | **mism** | oob | 行覆盖检查 | **行被覆盖** | size |
|---:|---:|---:|---:|---:|---:|---:|
| 0 / 3 / 6 | 1344 | **0** | 0 | 210 | 0 | 65536 |
| 1 / 4 / 7 | 544 | **0** | 0 | 60 | 0 | 8192 |
| 2 / 5 / 8 | 544 | **0** | 0 | 60 | 0 | **128** |
| 9 | 288 | **0** | 0 | 28 | 0 | 131072 |
| 10 / 11 | 288 | **0** | 0 | 28 | 0 | 16384 / **256** |
| **15 / 16 / 17** | 720 | **0** | 0 | 70 | 0 | **1024** |
| 18 / 19 | 720 | **0** | 0 | 70 | 0 | 65536 / 1024 |
| **TOTAL** | **12928** | **0** | 0 | — | **0** | — |

* **尺寸阶梯全绿**：128 / 256 / 1024 / 8192 / 16384 / 65536 / 131072 B，**没有"最小可靠拷贝尺寸"**
  ⇒ 主代理的**假说 2（小尺寸 DMA）❌ 推翻**。
* `oob = 0` ⇒ 没有越界拷贝；`行被覆盖 = 0`（9216+ 次配对检查）⇒ **池行没有被后来者改写**。
* 同一份探针在**图模式**（`f-D-audit3`）与 **eager**（`audit-eager.txt`，`--enforce-eager`）
  下都跑过，都是 **mismatch = 0**。
* 该臂 `replay_sha = 6a47dd65…` = `035` 的 x-D **逐字相同** ⇒ 探针没有污染行为。

### 2.3 【实测】attention 读到的字节：池臂 = 冷臂

`probe/f_read_audit.py` 在 `kv8_ori_plane` 入口把"这一步真要读的页"
（`block_table` 去重后的物理页）gather 出来，哈希 **int8 平面 + scale 平面**：

| 臂 | 配置 | 记录数 | J2 | 结论 |
|---|---|---:|---|---|
| `f-warm` | C0 几何，池 144 MiB（命中） | 960 | ❌ 1/16 | 960/960 条在冷臂里**逐字节找到**；冷臂反过来没有多余记录 |
| `f-cold` | 同几何，池 1 MiB（不命中⇒重算） | 1600 | ✅ 16/16 | 同一批 prompt / 同一 seed |

### 2.4 【实测】不是竞态（`F_SYNC_DMA=1`）

把每次 DMA 改成"提交后立刻 `wait(job_id)`"（异步变同步）：

| 臂 | 配置 | J2 |
|---|---|---|
| `f7-R1` / `f7-R3`（复跑） | C0 几何 16 × 4096 + 同步 DMA | ❌ 1/16（sha `a7ffff6b…`，与异步逐字相同） |
| `f7-R2` | C0+L1 16 × 2048 + 同步 DMA | ❌ 13/16（sha `506f5eee…`，与异步逐字相同） |

⇒ **"GPU 页在 D2H 读走之前被后续 prefill 覆写"这条竞态被排除**。

---

## 3. ★★ 真凶定位：**触发条件 = "几何 + 首 token 的计算形状"**

### 3.1 全表（30 条臂，全部本任务实测；`raw/036-f-fidelity/*.client.json`）

| 臂 | 几何 | prompt | 池 | 请求数 | **J2** | replay p50 TTFT | 关键读数 |
|---|---|---:|---:|---:|---|---:|---|
| `f6-G-tiny` | C0 | 4096 | **16 MiB** | 16 | ✅ 16/16 | **497 ms** | 池太小 ⇒ **根本不命中**（重算） |
| `f6-G-big` | C0 | 4096 | **512 MiB** | 16 | **✅ 16/16** | **63.4 ms** | 池够大 ⇒ **命中且无淘汰** |
| `f8-T-4096` | C0 | **4096** | 144 MiB | 16 | ❌ **1/16** | 66.6 ms | 对齐（= 32 × 128） |
| **`f8-T-4095`** | C0 | **4095** | 144 MiB | 16 | **✅ 16/16** | 205.1 ms | **非对齐**（= 31×128 + 127） |
| `f3-C0-p8` | C0 | 4096 | 144 MiB | **8** | ✅ 8/8 | 64.4 ms | 请求数不足 |
| `f3-C0-p2` | C0 | 4096 | 144 MiB | **2** | ✅ 2/2 | 68.6 ms | 同上 |
| `f1-warm-1p` | C0 | 4096 | 144 MiB | **1** | ✅ 1/1 | 65.3 ms | 同上 |
| `f4-S-1k` | C0 | 1024 | 144 MiB | 16 | ✅ 16/16 | 179.3 ms | 工作集 16k token < HBM 22,719 |
| `f4-S-2k` / `f5-C-eg-2k` | C0 | 2048 | 144 MiB | 16 | ✅ 16/16 | 60.9 / 614 ms | 见 §3.2 |
| `f3-C0-eager` | C0 | 4096 | 144 MiB | 16 | ❌ **1/16** | 366.6 ms | **eager 与图模式同结果同 sha** |
| `f3-C0-3r` | C0 | 4096 | 144 MiB | 16 | ❌ 1/16 | 66.5 ms | **`replay1_sha == replay2_sha`** |
| `f1-warm-rev` | C0 | 4096（**倒序回放**） | 144 MiB | 16 | ❌ **仍 #15** | 62.6 ms | 失败点不随顺序移动 |
| `f6-L1-2k` | C0+**L1** | **2048** | 144 MiB | 16 | ❌ **13/16** | 67.5 ms | L1 让 2048 也对齐命中 |
| **`f8-T-L1-2047`** | C0+L1 | **2047** | 144 MiB | 16 | **✅ 16/16** | 201.8 ms | **非对齐 ⇒ 补算 ⇒ ✅** |
| `f-D-audit3` / `f-D-probe` | D（4 条杠杆） | 4096 | 144 MiB | 16 | ❌ 14/16 | 108.4 ms | 逐字复现 `035` 的 `6a47dd65…` |
| `f-D-audit2` | D | 2048 | 144 MiB | **4** | ✅ 4/4 | 92.8 ms | 小规模 |
| `f1-warm-long` | C0 | 4096 → **4352** | 144 MiB | 16 | ❌ 16/16 | 190.2 ms | **冷臂也 16/16 ⇒ 该臂无效**（`f1-cold-long` 同样 16/16） |

### 3.2 判据（三条，都能被证伪）

1. **几何**：必须存在 **int8 SWA 平面**（C0 / D / F 全 ❌）。
   **反证**：`035` 的 `x-R2`（L5+L1+**ring16**，SWA 保持 **BF16**，16 × 4096，池 144 MiB）
   = **J2 ✅ 逐字节** ⇒ **ring16 不触发**，触发点需要 int8 SWA 面。
   （⇒ 回答主代理的 (a)：**`VLLM_V41_KV8_SWA=0` 那一格是 ✅**，现成数据即可判定。）
2. **形状**：池命中后，首 token 必须**完全**由 **1 行 decode** 产生（即 prompt 长度是 `block_size`
   的整数倍、命中覆盖全部块）⇒ ❌；只要需要**任何 prefill 形状的补算**（4095 / 2047）⇒ ✅。
3. **可重复性**：差异确定性（`replay2 == replay1`，跨进程同 sha）⇒ **不是 `037` 的随机抖动**
   （`037` 的现象在单卡 tiny 上**未复现**；本任务 30 条臂里，凡同配置的重复臂 sha 全等）。

### 3.3 【推断】机制落点（未做 kernel 级证明，标 **【推断】**）

池命中时，`f_read_audit` 显示：**decode 形状**下 `kv8_ori_plane` 的
`pages_per_req = 2`、scratch = `(2, 128, 1, 512)`；**prefill 形状**下
`pages_per_req = 32`、scratch = `(32, 128, 1, 512)`，并且**整表重编号**
（`table = where(delta ∈ [0, blocks_per_req), base + delta, 0)`）。
两臂读到的**行字节相同**（§2.3），但**读侧把同一批行映射进不同形状的 scratch / 表**
——"BF16 路径形状不敏感、int8 路径形状敏感"与此吻合。
**下一格（一条命令）**：把 `kv8_ori_plane` 的 decode 分支改成"也用全序列页表 + 整表重建"
（不再用 2 页快路径），看 J2 是否转 ✅。

---

## 4. 任务书 (3)：修 / 验证 —— **未完成（诚实收口）**

**没有改任何一行生产代码。** 原因不是时间不够，是**改点还没被钉死**：

* 任务书 §(3) 要求的"**J2 在 C0 / D / F 上转 ✅**"**已经被达成**——
  但达成方式是**改 prompt 长度的对齐性**（4095 / 2047），**不是改代码**；
* 真正的修点（§3.3）落在 `kv8_ori_plane` 的 **decode 形状重建** 上，
  而我**还没有**把它缩到"一行改动 + 可复现的成功判据"。
  **按任务书 §(4) 的纪律，不硬撑。**

### 4.1 交付的"诚实没修掉"清单

| # | 项 | 状态 |
|---|---|---|
| 1 | 池往返字节保真（**排除整个内存 / DMA 层**） | ✅ **【实测】已完成** |
| 2 | 读侧字节一致（同形状） | ✅ **【实测】已完成** |
| 3 | 真凶缩到"int8 SWA 读路径对调用形状敏感" | ✅ **【实测】+【推断】落点** |
| 4 | 具体修法（改 `kv8_ori_plane` 的 decode 分支） | ⛔ **未做**（下一步一条命令见 §3.3） |
| 5 | **int8 两条杠杆（×1.4655 / ×1.9133）** | ⛔ **维持否决**（理由从"池不保真"改为"读侧形状相关"） |
| 6 | `x-E` 变长前缀（4096→2048） | ⚠️ **【未确认】**——本任务证明它的前提（先修 §3）还没满足；但有**可用的替代判据**：`f8-T-4095` / `f8-T-L1-2047` 在"非对齐补齐"下 ✅ |

### 4.2 对 A2 上线裁决的影响（★ 本任务最有价值的产出）

| 项 | 旧裁决（`035` §5.4） | **新裁决（036）** |
|---|---|---|
| `L5` | ✅ 可上线 | ✅ 不变 |
| `L5+L1` | ✅ 逐字节保真（池 ×1.96） | ✅ **不变** |
| `L5+L1+SWA-q+ring16`（×1.4655） | ⛔ "池往返不保真，先修保真性" | ⛔ **仍不可上线，但理由换成**："int8 SWA 读路径在 1 行 decode 形状下与 prefill 形状不一致" |
| `+KV8 双平面`（×1.9133） | ⛔ 同上 | ⛔ 同上 |
| **"池往返保真性"这个命题本身** | 被列为**否决点** | ✅ **证伪**：12,928 次拷贝逐字节全等 |

**⇒ 可复用的运行期自检（替代 `035` §7.1 的 `记账对账`）**：

```bash
# 池命中后"首 token 完全由 1 行 decode 产生"时，int8 SWA 才暴露形状相关差异
# 现成入口（两条互为对照）：
#   env ARM=2 PROMPT_TOKENS=4096 bash agents/F_fidelity/scripts/run_probe_arm.sh   # 期望 J2 ❌
#   env ARM=2 PROMPT_TOKENS=4095 bash agents/F_fidelity/scripts/run_probe_arm.sh   # 期望 J2 ✅
```

---

## 5. cannbot 对照（AGENTS.md §6）

本任务**不改 kernel、不做量化数值验证**（只做"接线 / 保真"判据），因此只查了两条：

| 查的地方 | 它说什么 | 本任务采纳 / 没采纳 |
|---|---|---|
| `model-infer-quantization/SKILL.md:424-451`（§7.1 等价性自检） | *"**不能只看代码 diff，必须证明真实运行**"*；*"W8A8 允许细微 token 差异"* | ✅ **采纳**：本任务**不用** sha 判死量化本身（§3.1 里所有臂的 fill sha 都是 `24b57053…`），只把它当**接线 / 一致性**判据；并补了一条 cannbot 没写的——判据必须**形状对等**，否则测到的是"形状差异"而不是"保真性差异"（本任务的核心发现） |
| `ops/pypto-precision-compare/precision-verify/SKILL.md:145-171`（dtype→rtol/atol 表 + 双阈值） | fp16 rtol 1e-3；判定用双阈值 | ⚠️ **部分采纳**：本任务的量是**逐元素舍入**，所以用"逐字节相等 / 不相等"而不是 rtol；只借用它的**量级**（§2.2 的 0 差异远优于任何 rtol） |
| `model-infer-kvcache/SKILL.md:102-113`（PA 映射） | `物理 slot = block_table[b, pos // block_size] × block_size + pos % block_size`，**单一 block_size** | ✅ 逐字采纳（上游实现即此；本任务一行没改）。★ 但本任务实测出一条它没覆盖的：**同一批字节在不同"调用形状"下被读，结果可以不同** ⇒ 建议后续把"形状对等"写成 KV 保真判据的前置条件 |

---

## 6. 复现入口（每条都是**一条命令**）

```bash
# 0) 准备：从 COS 取 a2/logs/raw/036-raw-source.tgz，解到
#    ~/projects/dsv41-upstream-pr/agents/F_fidelity（容器内 = /work/agents/F_fidelity）

# 1) 池字节级审计（只读探针，~90 s/臂）
bash tools/a3_chip.sh c2 --timeout 1500 --name f-audit -- \
  env TAG=f-audit-D ARM=1 PROMPTS=16 PROMPT_TOKENS=4096 \
  bash /work/agents/F_fidelity/scripts/run_probe_arm.sh

# 2) ★ 决定性一格：形状对等（对齐 ❌ / 非对齐 ✅）
bash tools/a3_chip.sh c2 --timeout 2400 --name f-shape -- \
  bash /work/agents/F_fidelity/scripts/run_batch_f8.sh

# 3) 淘汰 / 池压力对照
bash tools/a3_chip.sh c2 --timeout 2400 --name f-pool -- \
  bash /work/agents/F_fidelity/scripts/run_batch_f6.sh

# 4) 竞态判定（同步 DMA）
bash tools/a3_chip.sh c2 --timeout 2400 --name f-race -- \
  bash /work/agents/F_fidelity/scripts/run_batch_f7.sh
```

**锁退出码 75 = 没抢到，是重试不是失败。**

---

## 7. 原始数据与代码

| 类 | 位置 |
|---|---|
| 本文 | `a2/logs/036-20260922-int8-fidelity.md` |
| 原始数据（50 文件，3.1 MiB） | `a2/logs/raw/036-f-fidelity/`：30 条臂的 `*.client.json` / `*.meta.txt`、4 份池审计汇总、7 份读侧日志（gz）、`probe/`、`scripts/` |
| 原始打包件 | `a2/logs/raw/036-raw-source.tgz`（= A3 上 `raw036.tgz` 原件，md5 可对） |
| 代码 | `a2/agents/F_fidelity/`：`probe/{f_pool_audit.py,f_read_audit.py,sitecustomize.py}`、`scripts/{prepare_overlay.sh,selfcheck_f.py,run_probe_arm.sh,run_two_arms.sh,run_batch_f1,f3,f4,f5,f6,f7,f8.sh}` |

大文件 md5（`raw/036-f-fidelity/`，解压后为 `read-*.txt`）：

```
e3eebe7761981ac68cef23d3cf601134  read-f4-warm.txt.gz
0d456404ae4a952b730cfb375bf91955  read-f4-cold.txt.gz
776e1bc37736b6f99c3bba0a9e96dea9  read-f5-warm.txt.gz
c27328a7d70c9f93cbf46a80141b110d  read-f5-cold.txt.gz
fa06ab65af2f916ed616ca4c41c7c132  read-f5-cold2k.txt.gz
940400a315395c34ba34ae1bcc3e2a1e  read-c0-warm.txt.gz
2e7e817af628fc1b974a814f1624bfb3  read-c0-cold.txt.gz
```

---

## 8. 诚实边界（哪些没测 / 哪些是推断）

| # | 事项 | 状态 |
|---|---|---|
| 1 | 池往返字节保真（12,928 次） | ✅ **【实测】** |
| 2 | 读侧同形状字节一致（960/960） | ✅ **【实测】** |
| 3 | 非竞态（同步 DMA 无效） | ✅ **【实测】** |
| 4 | 形状对等 ⇒ J2 转 ✅（4095 / 2047） | ✅ **【实测】** |
| 5 | 确定性（`replay2 == replay1`，跨进程同 sha） | ✅ **【实测】** |
| 6 | **具体修法**（改 `kv8_ori_plane` decode 分支） | ⛔ **未做**；下一步已写成一条命令（§3.3） |
| 7 | 8 卡真权重 / TP8 上的复现 | **未做**（本任务全在单卡 tiny；`037` 的随机抖动在单卡 tiny 上未复现，**两者是否同源【未确认】**） |
| 8 | `026/028` 融合 kernel 叠加后的行为 | **未做**（本任务沿用 `035` 的 `pkg-ring` 基线） |
| 9 | A2 上的 `aclrtHostRegister` / 宿主实占 | **未做**（沿用 `035` 的待办） |
| 10 | 长生成（>1 token）、`concurrency > 1` | **未做**（harness 默认 `MAX_TOKENS=1`、`concurrency=1`） |

**红线核对**：没发任何 PR / issue / 评论；没写 `upstream-v41/`（只读了 pkg 与镜像源码）；
没用 `/tmp`；占卡走锁（`a3_chip.sh`，无 75 退出）；没手设 `ASCEND_RT_VISIBLE_DEVICES`；
没碰 c0 / `mooncake-*` / `jitpgo-*` / `dsv41-a3`；A3 宿主 `MemAvailable` 全程 ≥ 1.6 TiB；
传文件全走 `cos-xfer.sh`；结论逐条标了【实测】/【推断】/【未确认】。
