# 066b — 【实测】档 D 的 replay 轮**输出内容改变**：确认的真缺陷（档 C 清白）+ 四条判据失效

> 2026-09-22 17:3x–17:5x CST（**远端 A3-node1 时间**）。执行：子代理 **`C2_int8_draftgraph`**，**只有 c2（die 7）**。
> 这是 [`066`](066-int8-x-draftgraph-1die.md) 的**第二卷**：066 交付「int8 × draft 图**两档都跑通**」，
> 本卷交付「**跑通之后发现的真缺陷**」。
> 产物：本日志 + `logs/raw/066-c2-int8-draftgraph/{out4,out5,out6,out7}/` + `agents/C2_int8_draftgraph/`。
> 标记：**【实测】**= 有判别力的判据跑出来的数；**【推断】**；**【未确认】**。

---

## 0. 一句话结论

> ★★★ **档 D（`VLLM_V41_KV8=1`，long-KV int8）在「同一 prompt、同一条件、清缓存后重发」时，
> 生成的 token 序列与首轮不同**：8 条 prompt 里 **3 条**不同（`[4,5,7]`），**确定性**
> （`replay == replay2` 逐字相同），且**进程内累积**（另有 2 条在第二轮 replay 才变）。
> **档 C / 档 B 在同一判据下 8/8 全部逐字相同**。
>
> ⇒ ★ **A2 裁决：档 C 可以上；档 D 必须挡住**（直到机制查清或修好）。
> ⇒ ★ 本卷同时作废了**四条**判据（§5），其中三条是本项目已归档的历史结论。

---

## 1. 判据：把「sha 不同」变成「实际差在哪个 token」（`text_probe.py`）

### 1.1 为什么需要新判据

066 的 `fill_sha256` / `replay_sha256` 只能给出「同 / 不同」，给不出**差异的性质**。
而 `048 §7.3` 的先例说这个家族的差异是「**退化单 token**（`'\n'` / `' '` / `'_'`）⇒ 边缘 argmax 翻转（良性）」。
⇒ 必须把**两轮的实际文本**并排打出来才能判定良性与否。`048` 那次是人工并排，本卷做成**机械判据**。

### 1.2 实现要点（`scripts/text_probe.py`）

```
★ prompt 构造 = bench/kv_offload_client.py:40 make_prompt **逐字相同**：
    base = (seed*100003 + salt) % 100000 ; [1000 + ((base + i*7919) % 100000) for i in range(n)]
  ⚠️ 我第一版用了 random.Random((salt, idx)) —— 那是错的，不是同一批 prompt，现象可能不复现。
     现象是 prompt 相关的 ⇒ prompt 差一个 token 就可能换一批 prompt 才复现。
★ 口径与 bench 一致：temperature=0 / ignore_eos=True / max_tokens=16 / stream=True；
★ 每轮前 POST /reset_prefix_cache；★ 跑三轮（fill / replay / replay2）判确定性。
★ 记录：text_repr / chars / sha / finish_reason / n_sse_pieces。
```

### 1.3 ★ 「不能只看 serve.log」——我在这里栽过一次

检查 `048` 那批 8 卡臂的 **APC 有没有接上**时，我先 `grep '[apc]'` / `Q_apcrecord` ⇒ **16 条臂全 0**，
差点写成「048 的 APC 也没接上」。**错** —— 那是 `Q_apcrecord` 的 finder 横幅；
`048` 用的是 **R8 的内联实现**，标记是 **`[R8-INT8-TRACE]`**，落在**另一个文件**
（`<臂>.trace.txt`，**不是** `serve_a2.log`）。

⇒ 教训（与 AGENTS §5b #7 同源）：**同一个保护有两套实现 ⇒ 两套标记 ⇒ 两套落盘位置。**
查「某个保护有没有生效」之前，先确认**这个臂用的是哪一套实现**、标记串是什么、落在哪个文件。

---

## 2. ★★★ 主判据：档 D 的 replay 轮**输出内容改变**

### 2.1 两臂对照（`out6`，`reset` 两轮都返回 200 `{"success":true}`）

| 臂 | 档 | same | mismatched |
|---|---|---:|---|
| **`c2-td-dg1-text`** | **D** | **5/8** | **[4, 5, 7]** |
| `c2-tb-dg1-text` | B | **8/8** | — |

### 2.2 逐条文本（★ 本卷最硬的证据）

```
  p0: fill='icho下部四面八方下部四面八方…（48 字符）'   replay=逐字相同
  p1: fill='凄 mass凄 mass…（48）'                       replay=逐字相同
  p2: fill='aya妈妈aya妈妈…（40）'                       replay=逐字相同
  p3: fill=' reserv communities…（152）'                 replay=逐字相同
★ p4: fill='menopausal拟' x8   （88 字符）  →  replay='menopausal拟menopausal'       （21 字符）
★ p5: fill=' Surprisingly钴' x8（112）      →  replay=' Surprisingly钴 Surprisingly' （27）
  p6: fill=' capaz compliments…（144）'                  replay=逐字相同
★ p7: fill=' separating每天' x8（104）      →  replay=' separating每天 separating'    （24）
```
⇒ ★ **档 B 的同 8 条 prompt 两轮全部逐字相同**（字符数 48/48/40/152/88/112/144/104 完全一致）
⇒ **探针本身不制造差异**；差异是**档 D 特有**的。

### 2.3 精确定性（`out7`：加了 `finish_reason` + 第三轮 `replay2`）

| prompt | fill chars | replay chars | **replay2 chars** | same | ★ replay==replay2 |
|---:|---:|---:|---:|---|---|
| 0 | 48 | 48 | **10** | True | ✗（**第二轮 replay 才变**）|
| 3 | 152 | 152 | **19** | True | ✗（同上）|
| **4** | 88 | **21** | **21** | False | ✅ |
| **5** | 112 | **27** | **27** | False | ✅ |
| **7** | 104 | **24** | **24** | False | ✅ |
| 1, 2, 6 | — | — | — | True | ✅ |

⇒ ★★ **两个独立发现**：
1. **确定性**：p4/p5/p7 的 `replay == replay2` **逐字相同** ⇒ 不是 `037` 的随机抖动，是**可重复的错**；
2. **进程内累积**：p0/p3 的**首轮 replay 正常、第二轮 replay 才变** ⇒ 退化**随请求数累积**，
   而 p4/p5/p7 是「从一开始就敏感」⇒ 不是「某条 prompt 的偶发」，而是**引擎状态的渐进污染**。

### 2.4 ⛔ 我给自己纠一次：**不是「提前终止」**

第一版探针没记 `finish_reason`，我据「文本变短」推断成「**生成提前终止**」并上报了 —— **那是错的**。
加上 `finish_reason` 后：
```
c2-td-dg1-usage：所有 8 条 prompt 的 finish 全是 length/length   ← 用满 max_tokens=16
c2-tc-dg1-usage：同样 length/length
```
⇒ **两边都生成了完整 16 个 token**，只是**解码出的文本不同**。

★ 正确表述：
> **同样的 `max_tokens=16`、同样的 prompt、同样的 `temperature=0`，档 D 的 replay 轮产出了
> 与首轮不同的 token 序列**（文本字符数 88 → 21 等）⇒ **是真实的内容差异，不是截断、不是停止。**

⚠️ 遗留【未确认】：两个 SSE 流各自的 `usage.completion_tokens` 这段**没拿到**
（本探针取到的全是 `None`）⇒「两边都是 16 token」这条靠 `finish_reason=length` **间接**支撑，
**不是**直接读数。**不许用这个数顶替。**

---

## 3. ★ 「档 D 的哪一部分」——两个成分都排除了

档 D 相对档 C 只多两个开关，我把它们**各自**关掉跑：

| 臂 | `VLLM_V41_KV8` | `KV8_PREFILL` | 前缀缓存 | fill_sha | replay_sha | match |
|---|---:|---:|---|---|---|---|
| `c2-tc-dg1` | 0 | 0 | on | `e27369ec…` | `e27369ec…` | ✅ |
| **`c2-td-dg1`** | 1 | 1 | on | `e27369ec…` | **`81629185d51d…`** | ⛔ |
| **`c2-td-p0-dg1`** | 1 | **0** | on | `e27369ec…` | **`81629185d51d…`** | ⛔ **同一 sha** |
| **`c2-td-dg1-noprefix`** | 1 | 1 | **off** | `e27369ec…` | **`b85b97f02c65…`** | ⛔ **另一 sha** |
| `c2-tb-dg1` | 0（全关） | 0 | on | `e27369ec…` | `e27369ec…` | ✅ |

⇒ ★★ 两条结论：
1. **关掉 `KV8_PREFILL`（写侧融合）⇒ sha 逐字不变** ⇒ **写侧不是元凶**；
2. **关掉前缀缓存 ⇒ 换了一个 sha，但仍 mismatch** ⇒ **「缓存命中重放」也不是解释**
   （与 `out6` 的 `reset=200` 同向）。
   ⇒ ★ 问题在 **`VLLM_V41_KV8=1` 这条 long-KV int8 路径本身**，不是它的两个可选优化。

★ **`81629185d51d…` 的性质**：`out/`（2 条）+ `out3`（1）+ `out4`（1）+ `out5`（1）
**五条独立臂逐字相同** ⇒ 它是「**档 D 在默认配置下的特征 replay sha**」、高可复现、适合当**回归判据**。
⚠️ 它**不是**「缺 APC 的特征」—— 见 §4。

---

## 4. ★★ APC_ALIGN 闭环：**做了，但没接上**（撤回一条自己的判断）

066 §5.3 曾把 `81629185…` 归因为「缺 `APC_ALIGN`」【推断】。本轮做了闭环，结论是：

### 4.1 `Q_apcrecord` 的 hook 目标是**卸载调度器**

```
[Q_apcrecord][apc] finder **已装载** target=vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler
                            ver=047-apc-record-1 mode=3
[Q_apcrecord][apc] SUMMARY ok=None path=finder mode=3 … boundary_calls=0 align_hits=0 tail_calls=0
                            module=None anchors={}
```
⇒ 本 harness **没挂 OffloadingConnector** ⇒ 那个模块**从未被 import**（`module=None`）⇒ **APC 没接上**。
⇒ ★ 两条臂（档 D / 档 C + `APC_ALIGN=3`）的 sha 与无 APC 时**逐字相同** —— 因为**它根本没运行**。
⇒ **066 §5.3 的【推断】未闭环、仍是【推断】**（`apc_record_fix` 对本 harness 是「没测到」，不是「无效」）。

### 4.2 ★★★ 顺手查出一条更重要的：**048 那批臂的 APC 在场、却一次都没工作过**

（标记纠正见 §1.3）：
```
臂（13 个 trace 实例，去重后）              trace行  ★changed=1  align_unit  env
r8-f1-tierD-graph.trace.txt   ← 判决臂        28         0        1024      3
r8-g1-tierC-graph.trace.txt                   28         0        1024      3
r8-e1-tierC-eager[-cold].trace.txt            28         0        1024      3
r8-a-tierB-graph.trace.txt（对照，env=0）     28         0           0      0
★ 合计 364 行，其中被对齐**真的改写过的 = 0**
```
**根因**：命中长度取到的是 `130048 = 127x1024` 与 `64512 = 63x1024` —— **本来就是 1024 的整数倍**
⇒ `align_unit=1024` 对它**是 no-op**。

### 4.3 ★★ 收窄：**APC 是真的、也已验证 —— 是这两个探针从没连上**

主代理 2026-09-22 18:0x 查了 `logs/raw/047-apc-align/*.q_apcrecord.txt`（**11 条臂**）：
```
q-a4 ~ q-a9 全部：boundary_calls=0  align_hits=0
```
★ **而 `047` 恰恰是 APC 被证明有效的那一轮** ⇒ **「探针报 0」≠「保护没工作」，只说明探针没连上**
（与我 §1.3 踩的是同一个坑）。

**APC 有效性的真实证据在 `047` 的另一个仪器上**（`047` 判据②）：
```
mode3  replay  used=1024  →  rows_changed=32/32   ★ 1024 = 32 行 x 32 token ⇒ 恰好一整页
046 stock replay  used=1   →  rows_changed=1/32    （31 行是别家字节 ⇒ 045 的 NaN）
⇒ D 几何 J2：❌ 14/16 → ✅ 0/16
```

⇒ ★★ **正确表述（取代我上面那句）**：
> **APC 本身是真的、也已验证**（`047`：`rows_changed 1/32 → 32/32`、J2 ❌14/16 → ✅0/16）。
> **但在 `048` 那批臂上它从未被触发** —— 因为命中长度取到的是 `130048 = 127x1024`、`64512 = 63x1024`，
> **本来就是栅格整数倍** ⇒ 对齐是 no-op。
> ⇒ **`048` 的 PASS 不是靠 APC 拿到的**，而 `048 §11.2` 那条「必须有 `APC_ALIGN=3`」
> **在 `048` 的负载下未被检验**。
>
> ★ 附带一条更值得记的：**两个探针（Q 的 finder、R8 的内联）在「保护真的工作时」也都报 0 或误报** ——
> 所以「APC 到底有没有生效」**不能靠探针横幅判**，只能靠 **`rows_changed` 那种结果量**判。

★ 与 066 §5.2 的「档 C 侥幸 PASS」**同源**：都是**触发条件没出现**，不是缺陷不存在。

---

## 5. ★★★ 本卷作废的四条判据（判据失效清单）

| # | 被作废的判据 | 出处 | 为什么失效 |
|---|---|---|---|
| **1** | 「变长回放下 `fill sha == replay sha` 没有判别力」 | `048 §7.1` | ★ 那是 `131072 → 65536` 的**变长**口径；本卷是 **512 → 512、同 salt、同 prompt** ⇒ **判据有效**（档 B/C 的 8/8 就是证据）。**用「上一批的无判别力」否掉「这一批的有效判据」是错的。** |
| **2** | 「差异全是退化单 token ⇒ 边缘 argmax 翻转（良性）」 | `048 §7.3.0d` | ★ 本卷实测差异是 **88 → 21 字符的内容改变**，不是 `'\n'`/`' '` 那种空白单 token。**形态不同 ⇒ 结论不可沿用。** |
| **3** | 「档 C/D 必须 `APC_ALIGN=3`，它是那批臂的保护」 | `048 §11.2` | ★ 实测 `changed=0`（§4.2）⇒ 那个保护**没工作过** ⇒ **它不能解释那批臂为什么通过**。 |
| **4** | ★★ **「探针报 0 = 保护没工作」**（升级版；主代理 18:0x 提供关键反证） | ★ 本卷 §1.3 + §4.3 | ★ 保护有**两套实现**（`Q_apcrecord` finder / R8 内联）、两套标记、两套落盘位置；而**在 `047`（APC 真的工作时）两个探针也都报 `boundary_calls=0`** ⇒ ★ **横幅/计数器为 0 只能说明探针没连上，不能说明保护没工作**。**判「有没有生效」只能看结果量（`rows_changed`）**。我因为这条差点写成「048 也没接上」。 |

---

## 6. ★ A2 裁决（本卷的直接后果）

| 档 | 判据 | 裁决 |
|---|---|---|
| **档 C** | 066：3 条臂 `sha256_match=True`；本卷：`text_probe` **8/8 逐字相同**（**同判据口径**） | ✅ **可以上** |
| **档 D** | 066：5 条臂 `sha256_match=False`；本卷：**内容改变 + 确定性 + 进程内累积** | ⛔ **必须挡住** |

★ 与已有归档的关系：这正是 `048 §7.2 #9`（「档 D 的 long-KV int8 面是否引入额外数值差异：⚠️【未确认】」）
—— 本卷把那一格从【未确认】**升级为【已确认的真缺陷】**，判据比原来更硬
（**实际文本 + 字符数 + 三轮确定性 + `finish_reason`**）。

★ 机制方向【推断，未验证】：与 `035 §5.3` 候选 1（**`block_stride ≠ page_size_bytes` 的平面行寻址**）
和 `036` 的结论（**int8 读侧对调用形状敏感**）**同源** —— 因为本卷实测证实了
**写侧融合不是元凶**、**缓存重放不是元凶**，剩下的就指向 **long-KV int8 的基础读写路径寻址**。

---

## 7. 诚实边界

1. **`usage.completion_tokens` 未取到**（全 `None`）⇒ §2.4 的「两边都是 16 token」靠
   `finish_reason=length` **间接**支撑，**不是直接读数**。
2. **机制未定**：本卷只做到「哪一部分不是元凶」（写侧融合 ✗、缓存重放 ✗），**没定位到具体代码行**。
3. **单卡 tiny + dummy 权重**：绝对 sha / TTFT 不可外推；**但「档 C 清白 / 档 D 有差异」这个二分是
   代码路径级的**，与 `063` 同口径，可传递。
4. **`p0`/`p3` 的「第二轮才变」只有 1 次观测**（两条 prompt）⇒ 【未确认】是否稳定复现。
5. **APC 闭环未完成**：`apc_record_fix` 的 hook 目标是卸载调度器，本 harness 无卸载
   ⇒ 要闭环必须在**带卸载**的配置上重跑（8 卡，或单卡 + OffloadingConnector）。
6. 本卷**没有**验证「档 D 的差异在 8 卡真权重上也能复现」⇒ ⚠️ 这是决定发布的最后一步。

---

## 8. 复现

```bash
# 0) 上传（走 COS）
bash a2/agents/C2_int8_draftgraph/scripts/upload.sh
ssh A3-node1 'cd ~/tmp/20260922/c2_int8_draftgraph && \
  coscli cp cos://uploads-new/share/xfer/c2_int8_draftgraph/c2_int8_draftgraph.tgz ./c2.tgz && \
  tar xzf c2.tgz -C ~/projects/dsv41-upstream-pr/agents/C2_int8_draftgraph --strip-components=1'

# 1) 判决读数（档 D 的实际文本；out6 = 文本，out7 = finish_reason + replay2）
cat a2/logs/raw/066-c2-int8-draftgraph/out7/c2-td-dg1-usage.textprobe.txt
cat a2/logs/raw/066-c2-int8-draftgraph/out7/c2-tc-dg1-usage.textprobe.txt   # 档 C 对照

# 2) 「APC 在场但不工作」（048 的 8 卡臂，只读）
cd <repo>/a2/logs/raw/048-int8-8card
grep -c 'changed=1' r8-f1-tierD-graph/r8-f1-tierD-graph.trace.txt
```

---

## 9. 卡与环境纪律

| 项 | 状态 |
|---|---|
| 用卡 | **只有 c2（die 7）**；`a3_chip.sh` 锁；退出码 75 **未出现** |
| c0 / c1 / Phy-ID 0–7 / 8–15 / `dsv41-a3` / `mooncake-master` | **未碰** |
| `ASCEND_RT_VISIBLE_DEVICES` | **未手设** |
| `/dev/shm` | 每臂起服前查（用**比例**判据，见 066 §7.1）|
| `/tmp` | **未用**（容器不共享宿主 `/tmp` ⇒ 用过一次立刻 rc=127，已改） |
| `upstream-v41/` | **未写**；**未发 PR / issue / 评论** |
| 镜像 | **一个字节没改**（全走 PYTHONPATH + overlay） |
