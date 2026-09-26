# CED-PD 精度/乱码问题的措施总账（2026-09-26）

> **本文只做一件事**：把"为了对付推理乱码"而采取的所有措施列清，并逐条标注
> **证据强度**（哪条是真的被单变量验证过、哪条只是机理对但没解决问题、
> 哪条至今没验证）。
>
> 纯只读整理，**未重启 P/D**。实测数据全部引自本仓库 `docs/`、`reports/`、
> `evidence/` 下的既有记录，每条都给了出处。

---

## 0. 先分清三种"乱码"，它们的指纹完全不同

混在一起讨论会得出互相矛盾的结论。本项目的全部故障可以归成三族：

| 族 | 指纹 | 观测手段 | 典型触发 |
|---|---|---|---|
| **A. 长上下文静默空答** | HTTP 200、`completion_tokens=1`、`content=null`（首 token 就是 EOS）、`u_fffd=0` | `answer_len` + `finish_reason` | 1M / 900K（每请求块数 ≳ 7000） |
| **B. 图模式乱码** | HTTP 200、`completion_tokens` **打满** `max_tokens`、**无** `finish_reason`、文本含 `<｜box｜>` 之类的碎片 token | 文本内容 | 144K 多轮、开了图但缺启动开关 |
| **C. 草稿静默失效** | 文本**不一定错**，但接受长度 `A≈1.0`（draft 白跑） | `SpecDecoding metrics` 的 Mean acceptance length | draft 图缺 metadata |

**这三族的判别量不同**，所以：
* 看 `ms/step` **无法**发现 A 和 C（A 的失败请求反而"更快"，C 的 ms 与正常同量级）；
* 看文本**无法**发现 C（C 的文本是对的，只是没加速）；
* `u_fffd`（UTF-8 替换字符）**恒为 0**，它从来不是判别量 —— 说明问题不在 tokenizer/解码层。

来源：[`CED-PD-PERF-20260925.md`](CED-PD-PERF-20260925.md) §3、
[`CED-PD-BLOCK-BOUND-20260925.md`](CED-PD-BLOCK-BOUND-20260925.md) §0、
[`reports/draft-graph-negative-control.md`](../reports/draft-graph-negative-control.md)。

---

## 1. 措施总表

按**证据强度**分三档，这是本文最重要的信息：

| # | 措施 | 对付哪族 | 证据强度 | 出处 |
|---|---|---|---|---|
| 1 | `[CED-POOL-GUARD]` / `[CED-32BIT-GUARD]`：池钳到 `num_blocks ≤ 29076` | **A** | 🟢 **刀锋实验验证** | `CED-PD-BLOCK-BOUND` |
| 2 | D 侧 `MULTISTREAM=0 DSA_OVERLAP=0` | **B** | 🟢 **单变量 0/4 → 4/4** | `CED-PD-PERF` §3 |
| 3 | `V41_CED_GRAPH_PROMPT_TAIL_EAGER=1` + fail-closed 门 | **B** | 🟢 **有/无对照** | `CED-PD-GRAPH-PREREQ` |
| 4 | `DSPARK_GRAPH_CAPTURE_METADATA=1`（随 `DRAFT_GRAPH=1` 强制） | **C** | 🟢 **负控实测** | `draft-graph-negative-control` |
| 5 | `DSPARK_CAPTURE_VALUE_FIX=1`（+ 三个代码默认 1） | **C** | 🟢 **实测 A 1.07 → 2.6** | `draft-graph-investigation` |
| 6 | `[CED-SWA-CLIP]`（replay 首 query 窗口裁到 replay 起点） | 机理对应 A | 🟡 **机理已验证，但对在飞故障无效** | `evidence/ced_swa_clip_ab_clip1_20260924` |
| 7 | D 侧 G7–G11 预清零（缺失页写前清零） | A/B 的**前置不变量** | 🟡 语义正确，**未单独做过消融** | `mooncake_hybrid_connector.py` |
| 8 | P 侧整段命中边界回退、D 侧空接收分支 | 崩溃（非乱码） | 🟢 6/6 与 3/3 复现 | `CED-PD-CACHE-HIT-PLAN` |
| 9 | 草稿 SWA 路径是否也需 clip | C/未知 | 🔴 **未验证** | `CED-PD-DSPARK-ACCEPTANCE` §4 |
| 10 | A2 生产的 `MULTISTREAM=1 DSA_OVERLAP=1` | **B（疑似）** | 🔴 **未验证，仅机理推断** | `CED-PD-HANDOVER-20260926` §8 |

---

## 2. 🟢 已证明有效的（5 条）

### 2.1 池钳位：`num_blocks ≤ 29076` —— 本条是对付 A 族的**真修复**

**机理**（代码 + 实验双重定位）：槽位 3 的页步长是 **147712 B**
（= layer-20 C1 KV 131072 + INT8 index K 16384 + FP16 scales 256）。
一旦 `块号 × 147712` 越过 2³²，算子的块地址会**按 32 位回绕到别的块**，
读到的可能是别的请求的陈旧 KV。

```
⌊2³² / 147712⌋ = 29076
```

**判据用页尾**（整页必须落在 4 GiB 内），所以是 `num_blocks ≤ 29076`，
**不是**"最大块号 ≤ 29076"——后者（`num_blocks = 29077`）会让块 29076 的
最后 54528 B 已经回绕。29077/29078 实测"通过"是因为回绕落点恰好是恒零的
null block，属**侥幸**，不能当安全值。

**验证证据**（这是全项目最强的一组）：

| 实验 | 结果 |
|---|---|
| 历史配对样本：FAIL 的 6 例，D 侧 g0 max = 29560/29560/29599×4 | 全部 ≥ 29077 |
| 历史配对样本：PASS 的 10 例，D 侧 g0 max = 13356…28228 | 全部 < 29077 |
| 刀锋臂 C=29128 / 29129 | 同一 1M 请求**必挂** |
| 刀锋臂 C=29077，请求落到块 29076 | **4/4 PASS** |
| 交付口径 C=29076 | 后续 **21/21** 验收全过 |

容量代价：相对 C=29600 损失 **1.77%**。

**两道防线**：
* 配置侧 `scripts/serve_a3_ced_pd.sh` 的 `[CED-POOL-GUARD]`；
* 强制侧 `experimental/ced/mooncake_hybrid_connector.py` 的 `[CED-32BIT-GUARD]`
  —— 它拿的是 **worker 实际注册的 stride**（planner 里拿不到，因为 stride 是
  打包布局算出来的，且 `num_blocks` 之后还会被多 rank 取 min 改动），
  超界直接 **拒绝起服**，并提示 `V41_CED_ALLOW_32BIT_OVERFLOW=1` 才可绕过。

**注意一个反直觉点**：这个故障**只在 8+8 真权重上能判**。
1+1 tiny 在完全相同的几何下（C=29600 / max=29599、C=29129 / max=29128）
**全部通过**，最可能的解释是 dummy 权重分布太平（首 token logprob 本来就 −11.77），
回绕造成的少量污染不足以翻转 argmax。

### 2.2 D 侧关多流：`MULTISTREAM=0 DSA_OVERLAP=0` —— 对付 B 族

**单变量隔离**：只重启 D、只改这一个开关，P 与其余参数一律不动，同一批 144K 四针：

| D 配置 | 144K 四针 |
|---|---|
| `MULTISTREAM=1 DSA_OVERLAP=1` | **0/4**，全部乱码 |
| `MULTISTREAM=0 DSA_OVERLAP=0` | **4/4 PASS** |

失败形态正是 B 族指纹（HTTP 200、打满 `max_tokens`、无 `finish_reason`、
含 `<｜box｜>`、`u_fffd=0`）。

**代价**：D 侧放弃多流重叠。这是本项目为正确性付的**明确代价**，
且**尚未定位到具体缺哪条同步**（只知道"就是这组开关"）。

⚠️ **口径要说清**：这次单变量隔离是在**全 40 层基线臂**的 D 上做的
（P 保持 `MULTISTREAM=1`），不是 CED 的 D。
但"decode 侧开多流 + DSA 重叠会在长上下文静默算错"这一条与
2026-09-24 CED 排查中"D 关多流 + prompt 尾步 eager + metadata 主流"的
结论**同族**，所以 CED 的 D 也一律取 0/0。
**CED 的 D 单独开多流会怎样，没有单独测过** —— 这属于"按同族结论预防"，
不是"实测过"。

### 2.3 `V41_CED_GRAPH_PROMPT_TAIL_EAGER=1` —— 对付 B 族

**机理**：CED 的图模式下，**单 token 的 prompt 尾步**（
`max_num_scheduled_tokens == 1` 且该请求还有 prompt token 没算完）
必须强制走 eager，否则会按"图模式裸跑"的坏路径执行。

**有/无对照**：2026-09-25 23:23 那次漏了它 → 144K 多轮 **3/3 全乱码**
（首轮 33 s，正常约 11 s）；补上后同一批请求立即 PASS，
且 `[CED-GRAPH] one-token prompt tail forced eager` 标记出现 **8 次**
（通过 21/21 的那台 D 日志里出现 **240** 次）。

**已加 fail-closed 门**（`scripts/serve_a3_ced_pd.sh`）：
* `CED_EXPERIMENTAL_GRAPH=1` 但缺这个开关 → **直接 exit 2 拒绝起服**；
* 确实要裸跑做诊断 → 必须显式设 `V41_CED_ALLOW_BARE_GRAPH=1`，
  此时只 WARN 并声明"结果不可当正确性证据"。
* 三种情况（missing / ok / bypass）都自检验过。

**同时纠正一个过期判据**：该文档原先列的硬门第一条
`[CED-META] inline metadata` 是**死的** —— 当前分支上没有任何代码打印它，
`V41_CED_METADATA_INLINE` 也没有代码读取（device-metadata 路径已无条件启用）。
通过 21/21 的那台 D 日志里它出现 **0** 次。

### 2.4 `DSPARK_GRAPH_CAPTURE_METADATA=1` —— 对付 C 族

**负控实测**（真权重，`DRAFT_GRAPH=1` 但启动器当时**没**注入它）：

| 指标 | 值 |
|---|---|
| `A`（接受长度） | **1.000 – 1.008**（8 发全部） |
| `ms/step` | 30.2 – 30.8（**与 eager 同量级**） |
| `dspark-graph-capture` 打印次数 | **0**（捕获期确实没建 draft attention metadata） |

**机理**：图捕获会走 `AscendDSAImpl.forward` 的 `attn_metadata is None` 兜底分支
（`output.fill_(0)`）⇒ **重放的图里根本没有 draft attention** ⇒ draft 输出全错。

**最危险的地方**：`ms/step` 反而"正常"甚至更好看。**只看时延会得出完全相反的结论。**

已在启动器无条件绑定（`DRAFT_GRAPH=1` 时自动注入）。

### 2.5 `DSPARK_CAPTURE_VALUE_FIX=1` —— 对付 C 族

**机理**：捕获期要填**代表值**，并**恢复图内的 context KV 写入**。
缺它时 `_context_slot_mapping_buffers` 仍是 `None`，
`precompute_and_store_context_kv` 在捕获时提前 return ⇒ **图里没有"写 KV"那串算子**。

**实测**：`A` 从 ~2.6 掉到 **1.07**、单流从 ~100 掉到 **40 tok/s**。

它是"四件套"里唯一需要显式传的（另外三件
`DSPARK_SWA_INDICES_RESIDENT` / `DSPARK_CAPTURE_NCTX_FIX` /
`DSPARK_DISPATCH_QUERY_LEN_FIX` 在代码里默认已是 1），
所以曾是**唯一的"陷阱开关"** —— 只写 `DRAFT_GRAPH=1` 会得到 A≈1.07 的坏配置
且**无任何报错**。

---

## 3. 🟡 机理对、但**没有**解决在飞故障的（2 条）

这一节是本文最容易被忽略、也最值钱的部分。

### 3.1 `[CED-SWA-CLIP]` —— 离线证明有效，真机 A/B 证明**没解决问题**

**它要修的机理**（代码级已核）：D 的 replay 第一个 query 的 128 窗口会向左
回溯到 `replay_start - 127`，而 P/D 两端都只保留 **2 页**
（`num_swa_blocks = cdiv(128,128)+1 = 2`），且 `dsa_v41.py` 把**完整**
`metadata.swa.block_table` 与完整 `seq_lens` 传给 `npu_sparse_flash_mla`
（`ori_mask_mode=4`、`ori_win_left=127`）—— 可见窗口**没有裁到 replay 起点**。
于是首个 replay query 可能读到本请求未持有的页。

**离线判据：通过。**
* `tools/ced_swa_clip_matrix.py`：真权重 1M 例 legacy 读列 `7965`
  不在持有集 `{7966, 7967}`；clip **零越界**；`N=100..2999` 全扫 **0 处不符**；
  多请求同批（N=4000 与 N=1019847 同批）逐行 rebase **无越界**。
* `ced_swa_clip_verify`：`legacy 越界长度数=5（需 >0）`、`clip 越界长度数=0（需 =0）`。

**真机 tiny：裁剪确实执行、数值与预测逐项一致。**
`[CED-SWA-CLIP] layer=2/20 block_size=128 q_len=128 base_pages=[14]
seqused_ori=[207] pages=2 seq_lens=[1999]` 与离线对 N=2000 的预测**完全一致**。

**但 8+8 真权重的 A/B 证明它没解决在飞故障**：

| 项 | 结果 |
|---|---|
| `V41_CED_SWA_CLIP=1` 单臂，1M 串行 10 次 | **8/10 通过**（失败在 #4、#8） |
| 失败形态 | 与修复前**逐位相同**（HTTP 200、`completion_tokens=1`、`content=null`） |
| 裁剪执行确认 | `[CED-SWA-CLIP]` 按层各 **80 行** = 10 请求 × 8 rank ⇒ **不是"没跑到"** |
| 续发 6 次 | 失败正好落在 #12、#16（全部 ≡ 0 mod 4，间隔都 413 s） |

同时**推翻了一条旧假设**：「块碎片化 ⇒ 失败」不成立 ——
#4 与 #10 的 g0 统计**完全相同**（`last=6136, descents=1854`）却一 FAIL 一 PASS；
#5/#6 的 `descents=7967`（比失败的两个都更碎）却都 PASS。
⇒ `first` / `last` / `descents` 这类**摘要量没有判别力**，不能当判据。

**结论**：SWA-clip 修的是一条**真实存在**的越界读（机理与离线判据都过硬），
但它**不是**当时观测到的 1M 失败的主因；主因是 §2.1 的 32 位回绕。
`V41_CED_SWA_CLIP` 目前默认 **1** 保留（无害且修的是真问题），
但**不要把它的存在当成"1M 乱码已修"的证据**。

### 3.2 D 侧 G7–G11 预清零 —— 语义正确，但没有单独消融

**它要修的问题**：G7–G11（上半层 SWA）**从未由 P 计算过**，D 只有 replay 会写
这 128 个位置覆盖的页；其余页若留着上一次请求的残页，
attention 就会读到不属于本次的数据。

**实现**：`start_load_kv` 里对 `*ced_missing_swa_groups`（= G7–G11）
按 `tensor.narrow(0, block_id, 1).zero_()` 清零（**不是**
`index_fill_` —— 后者会物化整个共享缓存视图、申请约 7.36 GiB/rank 临时显存）。

**证据强度：🟡**。它的必要性是**语义推断 + 代码级**的，
仓库里**没有**"关掉它 → 出现乱码"的消融实验。反而在缓存命中排查中，
它被列为"可能污染 hashed 块"的**嫌疑项**之一（见 §4.2）。

---

## 4. 🔴 尚未验证 / 明确开放的（4 条）

### 4.1 草稿 SWA 路径是否也需要 clip —— 未验证

`[CED-SWA-CLIP]` 修的是 target 的 `dsa_v41.py`，而**草稿走的是另一条路径**
（`dsa_v1.py` + `AscendDSparkProposer` → `AscendDSAMetadataBuilder`）。
如果草稿的 SWA attention 同样把整行块表 + 完整 `seq_lens` 交给算子，
同样的越界读会**在草稿路径上重现**，而 clip 不会生效。

**当前状态**：DSpark 下的 144K 四针与 1M 四针**全部 PASS**
（[`CED-PD-DSPARK-ACCEPTANCE-20260926.md`](CED-PD-DSPARK-ACCEPTANCE-20260926.md)），
但这是**"没复现"而不是"已证明不存在"**。

### 4.2 缓存命中路径的三处前提 —— 结论是「未触发」

`PREFIX=1` 在 CED 口径下曾被认为有三个代码级前提会失效。实测：

| 前提 | 实测 |
|---|---|
| 启动硬门（`PREFIX=1` 直接 exit 2） | ✅ 存在（已改成可显式放行） |
| 调度器 `num_computed_tokens != replay_end` 断言 | ❌ **未触发**（144K 6 次 + 1M 3 次均无 boundary mismatch） |
| D 侧 hashed 块预清零污染 | ❌ **未触发**（144K 交错测试 6/6 正确、与冷一致） |
| **（新发现）连接器 12-group 形状假设** | ✅ **1M 命中时必然触发并杀死引擎** |

⇒ 真正的阻断项不是原先猜的三条，而是**连接器的 12 组形状假设**。
修复后：`N=902909` **6/6 正确、88.5 s → 5.2 s（≈17×）**；
`N=1000065` **3/3 正确、105.7 s → 6.0 s（≈18×）**。
所有命中答案与冷路径**逐字节相同**，
两个容器日志里 `AssertionError` / `EngineDeadError` 计数都是 **0**。

### 4.3 A2 生产的 `MULTISTREAM=1 DSA_OVERLAP=1` —— 只到"机理同族"

用户报告 A2 生产出现乱码，指纹与 §2.2 的 B 族一致
（HTTP 200、`completion_tokens` 打满、无 `finish_reason`、含 `<|box|>`）。
**A2 生产的 D 侧正是 `MULTISTREAM=1 DSA_OVERLAP=1`**，
与我们在 A3 上单变量定位到的那一组开关**完全相同**。

但**没有在 A2 上做过单变量隔离**（A2 是生产环境）。
⇒ 这是**"高度可疑但未证实"**，值得单独开任务。

### 4.4 1+1 tiny 线复现不了 A 族故障 —— 已知的判据局限

见 §2.1 末尾。**1+1 的价值在几何、工具与"1M 可行性"**，
A 族的真判据**只能在 8+8 真权重上取**。

---

## 5. 当前实例的实际配置（只读核对，2026-09-26）

`/proc/<pid>/environ` 实读，**未重启**：

| 变量 | P（18990） | D（18991） |
|---|---|---|
| `V41_CED_ROLE` | `prefill` | `decode` |
| `SPEC` / `SP_TOKENS` | `0` / 7 | **`1`** / 7 |
| `STATIC_KERNEL` | 0 | **1** |
| `MULTISTREAM` / `DSA_OVERLAP` | `0` / `0` | **`0` / `0`** |
| `PREFIX` | 0 | 0 |
| `KV_DTYPE` | bfloat16 | bfloat16 |
| `V41_CED_SWA_CLIP` | 1 | **1** |
| `V41_CED_GRAPH_PROMPT_TAIL_EAGER` | 0 | **1** |
| `DSPARK_GRAPH_CAPTURE_METADATA` | 0 | **1** |
| `V41_QLI_NO_CANDIDATE` | 1 | 1 |

（`STATIC_KERNEL` 在 P 上是 0、D 上是 1 —— 本轮只在 D 上验过 §6 的收益，
P 侧未测。P 的 `SWA_CLIP=1` 无效果：`dsa_v41.py` 只挂在 decode 角色。）

**§1 表里的关键措施，D 侧全部在位**：池钳位（`num_blocks=29076`）、
关多流、prompt-tail eager、draft metadata、capture value fix、SWA clip。

**精度验收现状**：DSpark + `DRAFT_GRAPH=1` 口径下
**21/21 全过**（144K/1M 四针、流式、多轮、缓存命中），
四针答案与 `SPEC=0` 交付口径**逐字节相同**
（`ZQ7K-3341` / `VX2M-8890` / `HT4P-5527` / `RB9N-6014`）。

---

## 6. 起服检查清单（可操作）

按"漏了会怎样"排序：

```bash
# ① 池钳位：漏了 → 1M 静默空答（A 族）
grep -a "CED-32BIT-GUARD" d/serve.log | head -1
#   期望 num_blocks=29076 max_page_stride=147712；出现"超出 4 GiB 上界"就是没钳住

# ② 图模式前提：漏了 → 144K 多轮乱码（B 族），launcher 已 fail-closed
grep -ac "CED-GRAPH. one-token prompt tail forced eager" d/serve.log
#   期望 > 0（通过 21/21 的那台出现 240 次）

# ③ 多流：D 侧必须 0/0
docker exec <d> bash -lc 'echo $MULTISTREAM $DSA_OVERLAP'   # 期望 0 0

# ④ draft 图 metadata：漏了 → A≈1.0（C 族），且 ms/step 看不出问题
grep -a "dspark-graph-capture" d/serve.log | head -1
#   期望 "... built draft attention metadata (groups=1 layers=3 ...)"
#   出现 "LEGACY metadata-less capture" 就是坏的

# ⑤ 接受长度（唯一能证明草稿在干活的判据）
grep -a "SpecDecoding metrics" d/serve.log | tail -1
#   期望 Mean acceptance length ~2.4-3.4；**A≈1.0 就是草稿没产出**

# ⑥ 静态核（影响性能不影响正确性，但会静默降级）
bash experiments/dspark/check_static_kernel.sh <run_dir>
#   判据是 "static shape kernel will be used" > 0
#   ⚠️ 不要用 "static kernel compile start"：缓存命中时它是 0，会假阴性
```

---

## 7. 四条教训

1. **"指标变好"经常是故障的信号，不是修好的信号。**
   C 族失效时 `ms/step` 从 29.5 变成 **25.1**（因为每步只出 1.08 个 token）。
   ⇒ 任何单看时延的判据都不足以证明正确性，必须同时看 `A` / 文本 / `finish_reason`。

2. **"文档里的起服硬门"本身会过期。**
   `[CED-META] inline metadata` 被当硬门用了一轮，实际是**死开关**（无代码读取）。
   ⇒ 每条硬门都要能指出**打印它的代码**。

3. **摘要量不能当判据。**
   `first` / `last` / `descents` / `head` 这类块形态摘要，被证明对 A 族
   **没有判别力**（#4 与 #10 统计完全相同却一 FAIL 一 PASS）。
   真正的判别量是**块号的绝对上界**（29077）。

4. **修好一个真 bug ≠ 修好了在飞的故障。**
   SWA-clip 修的越界读是真的（离线判据 + tiny 真机都对），
   但它对 1M 失败**零改善**。如果不做那个 8/10 的真机 A/B，
   会一直以为"已经修好了"。
