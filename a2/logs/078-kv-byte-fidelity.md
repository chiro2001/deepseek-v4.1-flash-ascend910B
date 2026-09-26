# 078 — KV 取回路径的**设备侧逐字节往返保真**：修后两几何 ✅ / 修前 ❌

> 补齐 `071 §A1` 那个**连续三晚被标记、从未被做掉**的判据缺口。
> 2026-09-22 21:2x–22:4x（本机时钟；A3 远端慢 ≈7 min）。执行：子代理 **`kv_bytefidelity`**。
> 机器：**A3（A3-node1）槽位 `c1` = die 6**（容器 `prbench-c1`；`tools/a3_chip.sh` 全程持锁）。
> 红线：只用 c1；**没手设** `ASCEND_RT_VISIBLE_DEVICES`；**没用 `/tmp`**（`~/tmp/20260922/KV_bytefidelity/`）；
> **没写** `upstream-v41/`；**没碰** `dsv41-a3`/`mooncake-*`/别人的容器/Phy-ID 0–7；跨机传文件全走 `cos-xfer.sh`；
> **没改任何生产文件**（全部 `PYTHONPATH` 叠加包 + post-import 钩子）。标记：**【实测】/【推断】/【未确认】**。
> 原始数据：`agents/KV_bytefidelity/out/`（三臂逐 op JSONL + summary + metric + client）。

---

## 0. 一句话（**先回答任务书那句**）

**"取回路径有没有把字节搬错？"**

| 格 | 答案 |
|---|---|
| **修后（当前生产路径）· BF16 池** | ★ **没有**：6,783 次往返**逐字节一致**，metric 逐位相等 |
| **修后（当前生产路径）· 档 C（int8）** | ★ **没有**：7,423 次往返**逐字节一致**，metric 逐位相等 |
| **修前（`043` 的缺陷在）· BF16 池** | ★★ **有**：**5,376 次"取回"读到的不是当初写进去的 KV，而是全 0 页** |

★ 第三行是本轮最有价值的新东西：**单向判据对这种错完全失效** ——
修前臂的 `copy_mismatch = 0`（store 的 1/8 搬对了、load 的 8/8 也搬对了），
`039` 的判据 ①②在他自己的口径下也会全绿；**只有"写侧设备页 ↔ 取回后设备页"的往返比对能判它**。

---

## 1. 判据：为什么必须有"往返"（与 `039` 的关系）

| | `039`（H_kvcheck）判据 ① | `039` 判据 ② | ★ 本任务判据 ★★ |
|---|---|---|---|
| 比什么 | `sha1(GPU 源页)` vs `sha1(池行)`（store） | `sha1(池行)` vs `sha1(GPU 目标页)`（load） | `sha1(GPU 目标页 @load)` vs `sha1(GPU 源页 @store)` |
| 能证明 | 这一次 store 没把字节打乱 | 这一次 load 没把字节打乱 | **取回后的 KV 就是逐出前的那份** |
| 盲点 | —— | —— | —— |
| **盲点（合并）** | ★ 两者都通过**不能推出**往返成立：若 load 读的是**另一个合法行**（行复用/漏写），①② 仍然全绿 | | |

`039 §2.2` 已把这个盲点写成文字（"它只证明搬得对，不证明搬到的那一行是对的"），但**一直没人用设备侧的字节把它测出来**。
本轮用一个**桥**（`(tensor, 池行号)` → "该行最后一次 store 写入的 GPU 源页 sha"）把它接上：

```
store:  _row_src[(t, row)] = sha1(写侧 GPU 页)        ← 在模型流上、DMA 入队前快照（排除竞态）
load :  rt_ok      if sha1(GPU 目标页) == _row_src[(t, row)]        ★ 往返
        rt_stale   if sha1(池行)      != _row_src[(t, row)]         ← 池行被覆写/静默改字节
        rt_unwritten if _row_src[(t, row)] 不存在                   ← ★ load 读了从未 store 过的行
        copy_mismatch if sha1(池行) != sha1(GPU 目标页)             ← 单次拷贝本身错
```

逐 op JSONL 见 `out/<TAG>.ops.jsonl`；**离线第二实现**（`scripts/analyze.py`）独立复算同一批数（`039 §1` 的教训：探针必须能自证不是空转/记账错）。

---

## 2. 【实测】三臂判决表（唯一变量 = scheduler 修没修 / 哪种几何）

| 臂 | scheduler | 几何 | store ops | load ops | `rt_ok` | ★`rt_unwritten` | `rt_roundtrip_bad` | `copy_mismatch` | ★ metric 对账 |
|---|---|---|---|---|---|---|---|---|---|
| `kvbf-old` | **修前** `15d5548e` | BF16 | 3,368 | 6,784 | 1,408 | ★ **5,376 ❌** | 0 | **0** | ✅ 逐位相等 |
| `kvbf-bf16b` | 修后 `986c9115` | BF16 | 8,744 | 6,784 | **6,783** | **0 ✅** | **1**（= 注入对照） | 1（= 注入） | ✅ 逐位相等 |
| `kvbf-c` | 修后 `986c9115` | 档 C（`int8 + ring16`） | 11,344 | 7,424 | **7,423** | **0 ✅** | **1**（= 注入对照） | 1（= 注入） | ✅ 逐位相等 |

几何确认（`*.meta.txt` 里探针自己打的 **shadow 自检**）：

| 臂 | `swa_int8` | `ring_dtype` | `swa_plane_kwargs` |
|---|---|---|---|
| `kvbf-bf16b` / `kvbf-old` | **False** | `torch.float32` | `{'dtype': torch.bfloat16}` |
| `kvbf-c` | ★ **True** | `torch.float16` | `{'dtype': torch.int8, 'scale_dim': 4, 'scale_dtype': torch.float16}` |

★ **两组几何都干净** ⇒ 回答 `035`→`036` 那场争议（"int8 经池往返是否保真"）：
**在字节层保真**，`035` 报的"不保真"不是卸载层这一段造成的（与 `036` 的结论同向，但本轮是**设备侧字节**的第一手证据）。

### 2.1 字节级对账（反假阳性闸）

探针记的 op 集合**逐位等于**引擎自己的 metric ⇒ 我测的就是真实发生的 DMA，不是自造的样本：

| 臂 | 探针 store bytes | metric `GPU_to_CPU` | 探针 load bytes | metric `CPU_to_GPU` |
|---|---|---|---|---|
| `kvbf-old` | 364,421,120 | 364,421,120 ✅ | 272,957,440 | 272,957,440 ✅ |
| `kvbf-bf16b` | 529,858,560 | 529,858,560 ✅ | 272,957,440 | 272,957,440 ✅ |
| `kvbf-c` | 362,127,360 | 362,127,360 ✅ | 231,669,760 | 231,669,760 ✅ |

样本规模：`kvbf-bf16b` **15,528 op**（16 张量）、`kvbf-c` **18,768 op**（17 张量）、`kvbf-old` **10,152 op**（16 张量）
⇒ **每个被访问张量都有样本**（覆盖率闸 ✅，不存在 `039 §1.1` 那种"0 比较却报全绿"）。

---

## 3. ★★★ 修前臂：缺陷在**字节层**长什么样（本轮的新证据）

### 3.1 签名：store 1/8、load 8/8

```
T t=0 g=0 dir=store ops=64     ← group 0（full, bpc=8）每个 chunk 只交出 1 个 GPU block
T t=0 g=0 dir=load  ops=512    ← 同一个 chunk 要读 8 个 unit
```

⇒ 与 `039 §10` 的定性、`043` 的修前读数**同形**（`043` 记的是"group 0 实搬 64、读到未写过行 5352"；
本轮 64 / **5,376**，量级一致、几何略有差别，**不做逐字替代**）。

### 3.2 ★★ "读到未写过行" 在字节层 = **读到全 0 页**（`038` 的 NaN 机制被逐字节证实）

修前臂 5,376 条 `rt_unwritten` 样本，按 `size` 分组后**每个 size 只有 1 个 distinct sha**：

| size | 未写行样本数 | distinct sha | 该 sha 是不是 `sha1(全 0 页)` |
|---|---|---|---|
| 65,536 | 1,344 | **1** | ✅ `1adc95bebe9e…` = `sha1(b"\x00"*65536)` |
| 8,192 | 1,344 | **1** | ✅ `0631457264ff…` |
| 128 | 1,344 | **1** | ✅ `0ae4f711ef5d…` |
| 131,072 | 448 | **1** | ✅ `67dfd19f3eb3…` |
| 16,384 | 448 | **1** | ✅ `897256b6709e…` |
| 256 | 448 | **1** | ✅ `b376885ac845…` |

★ 即：**load 从"从未被 store 写过"的池行里取回的字节，确实是整页 0**（不是"别处的旧 KV"、不是随机值）。
⇒ `038` 假说链条的最后一环（未写行 ⇒ 全 0 ⇒ int8 `scale=0` ⇒ 反量化 `0/0 = NaN`）**前提成立**，
但它**只在 `bpc>1` 的组 + int8 平面上才会变成 NaN**（BF16 读 0 只当零向量）——
这解释了 `039 §4` 的"BF16 ✅ / int8 ❌ 但两者该不对称同形"的困惑：**BF16 臂根本没有把那些 0 当数值用**。

### 3.3 ★ 为什么单向判据完全失效

修前臂：`copy_mismatch = 0`。
每一次 store（1/8）字节都对、每一次 load（8/8）字节也都对 —— **错的是"读的是哪一行"**。
⇒ 这正是 `071 §A1` 说的"取回路径从未被逐字节验证"的**第一个实际后果**，也是 `043` 那个修复**必须**存在的原因。

---

## 4. 阳性/阴性对照（探针自证：判据有判别力，且不误报）

| 对照 | 做法 | 期望 | 实测 |
|---|---|---|---|
| **阴性** | 不注入 | 全绿 | `kvbf-bf16b` / `kvbf-c`：除注入外 **0 mismatch** ✅ |
| **阳性 A（注入·load）** | 翻转第 1 个 load op 的 **GPU 目标页内存副本** 1 字节 | `rt_roundtrip_bad ≥ 1` | 两臂各 **1** ✅（`op_n=1`，`src_sha == pool_sha`、`gpu_sha` 不同 —— 精确落在"load 搬错"这一类） |
| **阳性 B（真实缺陷）** | `SCHED_FIX=0`（= `043` 修前 scheduler） | 判据必须报警 | ★ `rt_unwritten = 5,376` ✅ |
| **阳性 C（不占卡单元自检）** | `scripts/selftest_kvbf.py`（mock torch/vllm，7 场景） | 全过 | **7 PASS / 0 FAIL** ✅ |

单元自检的 7 个场景（本机、不占卡，先于上卡跑）：
`阴性·snap 路径` / `阴性·readback 路径` / `阳性·池行被改` / `阳性·GPU 目标页被改` /
`阳性·读未写过的池行` / `注入·POISON_LOAD` / `注入·POISON_STORE`。
★ 它在**上卡之前**就抓出了本探针的一个真 bug：load 方向 `a` 是**池行**、`b` 才是 **GPU 目标页**，
第一版把两者当反了 ⇒ "往返判据"退化成"池行一致判据"（能测出拷贝错、测不出**读错行**）。
**如果只上卡跑，这条会被误判成"全绿"。**

---

## 5. 覆盖 / 不覆盖（先说清边界，避免口径滑移）

| | 内容 |
|---|---|
| ✅ 覆盖 | **每一个实际发生**的 store/load 拷贝的**全部字节**（不是采样）、逐 canonical 张量、逐组、逐 block |
| ✅ 覆盖 | store 与 load 的**身份一致性**（"取回的是不是当初写进去的那一行"）—— 本轮新增的这一轴 |
| ✅ 覆盖 | 池行在 store↔load 之间是否被改写（`rt_stale`） |
| ✅ 覆盖 | op 集合完整性（与引擎 metric **逐位**对账） |
| ✅ 覆盖 | 两组几何：BF16 池 / 档 C（SWA int8 + ring16） |
| ⛔ 不覆盖 | attention **读侧**把页拼成 KV 是否正确（`036`/`038` 的 `kv8_ori_plane` 话题） |
| ⛔ 不覆盖 | 模型数值路径（量化/反量化/归约顺序）—— `037` 的非确定性在这一层 |
| ⛔ 不覆盖 | `concurrency > 1`（本轮与 `027`/`039` 同口径 `concurrency=1`） |
| ⛔ 不覆盖 | **8 卡**（本轮全是单卡 tiny；8 卡口径仍【未确认】） |
| ⚠️ 边界 | `rt_unwritten` 的前提是"该池行由**本进程**的 store 写过"；单卡单 worker 成立，多 worker/多 rank 需按 rank 分别建桥（本轮不适用） |

**⇒ 保真只证"卸载层这一段搬运与寻址正确"，不能被引伸成"整个服务正确"。**

---

## 6. 交付件

| 文件 | 说明 |
|---|---|
| `agents/KV_bytefidelity/probe/kvbf_audit.py` | 探针本体（挂在真入口 `SingleDirectionNPUOffloadingHandler.transfer_async/get_finished`；用镜像自己的 `compute_sub_block_ptrs` 复算指针；全字节比对） |
| `agents/KV_bytefidelity/scripts/selftest_kvbf.py` | **不占卡**判据自检（7 场景，mock torch/vllm） |
| `agents/KV_bytefidelity/scripts/analyze.py` | 离线第二实现（从逐 op JSONL 独立复算 TOTAL/往返/metric 对账） |
| `agents/KV_bytefidelity/scripts/{prepare_overlay,run_arm_kvbf,sync_to_a3,fetch_from_a3}.sh` | 起臂与搬运（`SCHED_FIX=1` 换成 `043` 修后件、`=0` 留修前原件） |
| `agents/KV_bytefidelity/out/kvbf-{old,bf16b,c}.*` | 三臂原始数据（逐 op JSONL / summary / metric / client / meta） |

复跑（A3；槽位锁由外层持有）：
```bash
bash ~/projects/dsv41-upstream-pr/tools/a3_chip.sh c1 --timeout 2400 --name kvbf-bf16 -- \
  env TAG=kvbf-bf16 SCHED_FIX=1 XL1=1 XSWA=0 XRING=0 OFFLOAD_BYTES=150994944 \
      "P2_COMP_JSON=[[0],[1,2,3,4,5,6,7,8,9,10,11]]" KVBF_POISON_LOAD_OPS=1 \
  bash /work/agents/KV_bytefidelity/scripts/run_arm_kvbf.sh
# 档 C：XRING=1 XSWA=1 "P2_COMP_JSON=[[0,2,3,4,5,6,7,8,9,10,11],[1]]"
# 修前对照：SCHED_FIX=0（不加注入）
```

---

## 7. 【未确认】与下一步

| 项 | 状态 |
|---|---|
| 取回路径把字节搬错？ | ✅ **【实测】没有**（修后两几何；`copy_mismatch` 与 `rt_bad` 的唯一来源是探针自己的注入） |
| 修前的"读到未写过行"是**全 0 页**？ | ✅ **【实测】是**（6 个 size、distinct sha 全 = 1，且 = `sha1(全 0 页)`） |
| 修前那 5,376 次读到全 0 在 int8 下是否**真的**产生 NaN？ | ⛔ **【未确认】**（本轮只证"字节是 0"；NaN 需要 int8 臂 + 修前 scheduler 的另一格） |
| 8 卡（`027` 口径 `OFFLOAD_GB=56`）的同一判据 | ⛔ **【未确认】** —— 本轮全是单卡 tiny |
| `concurrency > 1` | ⛔ 未测 |
| 生产口径（真权重 `v41-w4a8-…` + `MAX_LEN=1M` + Engram） | ⛔ 未测（本轮是 `model-tiny` + dummy 权重） |

**建议的下一格（性价比最高）**：把本探针直接叠到 `R_8card_int8/scripts/run_arm_r8.sh` 的 8 卡臂上
（只加 `PYTHONPATH` 最前面一层 + 3 个 env），即可把"8 卡口径的取回保真"一次补掉 —— 那是 A2 上线前的最后一道字节级门。
