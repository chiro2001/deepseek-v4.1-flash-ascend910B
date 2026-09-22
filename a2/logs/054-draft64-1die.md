# 054 — ②c（draft block 128→64）在**单 die（A3 c2）**上的数值风险判定：输出逐字节不变、投机解码逐字不变、容量逐字按模型走

> 2026-09-22 11:45–13:05 CST。执行：子代理 **C2_draft64**。机器：**A3（A3-node1）的 c2 = `/dev/davinci7` = NPU 3 chip 1**
> （**全程只用 c2**，每臂都走 `tools/a3_chip.sh c2` 抢锁；没碰 c0/c1、没碰 `dsv41-a3` / `mooncake-master` / 别人的容器；**没手设 `ASCEND_RT_VISIBLE_DEVICES`**）。
> 模型：`agents/T_draftceiling/models/model-tiny-draft`（= L1_dummy 的 tiny + **只还原 draft 两个键** `num_nextn_predict_layers=3` / `dspark_target_layer_ids=[37,38,39]`）。
> 影子包：`agents/C2_draft64/pkg/dfix2c`（= T_draftceiling 的 `Ci8fix` 基础包 + **复用** ②c 生成器 + 我加的探针；**四臂同一个包，唯一变量是 `VLLM_V41_DRAFT_BLOCK`**）。
> 没写 `upstream-v41/`；没用 `/tmp`（`TMPDIR=~/tmp/20260922/C2_draft64`；A3 上的产物都在 `agents/C2_draft64/`）。
> 标记约定：**【实测】**= 有判别力的原始数据；**【算术】**= 由源码几何算出（脚本可复跑）；**【推断】**；**【未确认】**。

---

## 0. 一句话结论

| 问题 | 答案 | 判据 |
|---|---|---|
| **Q1 ②c 改完输出变了吗？** | **没变**：B 档与 D 档**各 7 轮**（4 冷 + 3 连发）输出 `sha256` **逐字节相同**（`0ebccb55b30c…`），**逐 prompt 16/16 相同**；**跨臂**（128 vs 64）也 **16/16** 且**每轮 `sha_all` 逐字相同**。**【实测】** | §3 |
| **Q2 投机解码退化了没有？** | **不是"没退化"，是"逐字不变"**：四臂 `drafts=4239 / drafted=21195 / accepted=2902`、`MeanAccLen=1.685`、`AvgDraftAcc=13.69%`、`per-pos[0]=0.685` **完全一致**；★ 更硬的一条：draft **提案 token 序列 4367/4367 次逐条相同**（探针直读，eager；graph 臂 1875/1875 同样逐条相同）。**【实测】** | §4 |
| **Q3 容量真的涨了吗？** | **B 档是"降"、D 档是"涨"，两个方向都逐字命中 050 的零参数模型** —— 这正是模型的**符号判别力**：B **20,826 → 19,247（×0.9242）**、D **23,651 → 36,825（×1.5570）**。**【实测】** | §5 |
| **判据有没有判别力？** | **有，而且是"当场炸"级**：反例臂（故意把 draft **块表行宽按 128 行块**算）**第一个 4096-token 请求就把 EngineCore 打死**（`ValueError: could not broadcast input array from shape (65,) into shape (64,)`），输出 sha 变、提案数 4367→**3**、请求 ok=0。**【实测】** | §6 |
| **tiny 能否替代 8 卡？** | **能替代"几何 / 寻址 / 不崩 / 图捕获 / 输出不变"这 5 维**，**不能替代**"真权重数值 / TP8 通信 / 卸载池与命中路径 / 8 卡绝对 token 数 / 多头接受长度"这 5 维。**【实测 + 未确认】** | §7 |

---

## 1. 为什么这一条值得用 c2 的排队时间去换

`051` 把 ②c 的**改动面**（2 文件 / 2 处）与**三臂单元自检**做完了，但留下一条它自己标了 `【未确认】` 的风险：

> draft 窗口 `sliding_window=128` 在 `block=64` 下跨 **2–3 块**（带投机 query 共 133 token ⇒ 最坏 4 块）；
> "算子 / 块表 / KV manager 都没有 ≤1 块假设" **只是读源码得出的**，没跑过。

本条实验就是把这条风险**变成实测**，而且**只用单 die**（c0 的 8 卡队列还在等 ⇒ 不占它的时间）。

---

## 2. ★★ 实验设计（两臂只有一个变量）

| 臂 | tier | `VLLM_V41_DRAFT_BLOCK` | eager | prompt | 轮数（pass1/pass2） | 用途 |
|---|---|---|---|---|---|---|
| `c2c-b128` | B（纯 BF16） | 128 | ✅ | 16×4096 | 4 / 3 | B 档基线（= 逐字旧行为） |
| `c2c-b64` | B | **64** | ✅ | 16×4096 | 4 / 3 | ②c 的**纯净臂**（除 draft 页外无其它变量） |
| `c2c-d128` | D（SWA int8 + ring16 + long-KV int8） | 128 | ✅ | 16×4096 | 4 / 3 | 交付档基线 |
| `c2c-d64` | D | **64** | ✅ | 16×4096 | 4 / 3 | ★ 交付档 + ②c（容量应**涨**） |
| `c2c-neg-b64` | B | 64 + **故意缺陷** `[C2-NEGCTRL v1]` | ✅ | 4×4096 | 3 / 1 | ★★ **阳性对照**（判据判别力） |
| `c2c-b64-graph` | B | 64 | ❌（ACLGraph） | 16×4096 | 2 / 1 | 图模式兼容性 |
| `c2c-b128-graph` | B | 128 | ❌（ACLGraph） | 16×4096 | 2 / 1 | 图模式的**对称对照** |

* **自变量唯一性**：两臂的 `PYTHONPATH` / `PGP_*` / `P2_*` / `VLLM_V41_KV8*` / 服务参数全部逐字相同，唯一差别是 `VLLM_V41_DRAFT_BLOCK`。
  影子包两文件 md5（四臂**同一对**）：`dspark.py = 0fd33532abcfcf7c3b43f8ed89d19881`、`core/deepseek_v41.py = f9b1b560fd54abe69531cf2abd4efed0`。
* **权重同一性**：`--load-format dummy` 的初始化是**每参数一个固定 seed（默认 1234）+ 只依赖形状/dtype**
  （`vllm/model_executor/model_loader/weight_utils.py:1298-1330`）⇒ ②c 不改任何参数形状 ⇒ **两臂权重逐比特相同**（
  【推断】，与 §3 的 sha 逐字节相同互为印证）。
* **工作量口径**：`temperature=0`、`--seed 0`、固定 `prompt_salt=20260922`（所有臂同一批 prompt）、`max_tokens=64`
  （★ **不是 1**：`mt=1` 时 SpecDecoding 只有 ~10 个 drafted token，没有统计功效）、`concurrency=1`。
  `pass1` = 4 轮**冷**（每轮前 `/reset_prefix_cache`）、`pass2` = 3 轮**连发**。B 档每轮 ≈ 112 s，D 档 ≈ 123 s。
* ★ **工作量是否覆盖"跨块"**：`block=64` 下 4096 个 prompt token + 64 个解码 token ⇒ 末尾位置 ≈4160
  ⇒ draft 组块表需要 **⌈4160/64⌉ = 65 项**（≥2 块），且每个解码步的窗口 `[l−128, l−1]` 跨 **2–3 个 64 行块**
  （带投机 query 共 133 token ⇒ 最坏 4 块）⇒ **"窗口跨块"这条风险在本次 workload 里是被真实走到的**（不是纸面）。

---

## 3. ★★ Q1：输出 sha256（寻址对不对）

### 3.1 臂内重复一致性（先证明判据可用 —— `037` 的教训）

`037` 在 **8 卡 TP8 + dspark** 上测到 `temperature=0` 下同臂内会抖（首 token 翻转 25–100%），**所以单次比对没有判别力**。
本实验在**单 die** 上先测臂内：**每臂 7 轮全部 `distinct_sha = 1`**（B/D × 128/64 全部如此）。

| 臂 | pass1（4 冷轮） | pass2（3 连发轮） | distinct |
|---|---|---|---|
| `c2c-b128` | `0ebccb55…` ×4 | `0ebccb55…` ×3 | **1** |
| `c2c-b64` | `0ebccb55…` ×4 | `0ebccb55…` ×3 | **1** |
| `c2c-d128` | `0ebccb55…` ×4 | `0ebccb55…` ×3 | **1** |
| `c2c-d64` | `0ebccb55…` ×4 | `0ebccb55…` ×3 | **1** |
| `c2c-b64-graph` / `c2c-b128-graph` | `0ebccb55…` ×2 | `0ebccb55…` ×1 | **1** |

⇒ 【实测】**单 die（TP=1、eager 或 graph）上这个服务是逐字节可复现的** ⇒ `sha256` 判据在这里**有判别力**
（与 8 卡的抖动**相反**；原因见 §7 的解释：抖动来自 TP8 归约顺序/批形状，单 die 没有这一层）。

### 3.2 跨臂（128 vs 64）

| 档 | 对比 | 结果 |
|---|---|---|
| **B** | `c2c-b128` vs `c2c-b64`（pass1 4 轮 + pass2 3 轮，逐轮） | `sha_all` **✅逐字相同 7/7 轮**；**逐 prompt 16/16 相同**（每轮） |
| **D** | `c2c-d128` vs `c2c-d64`（同上） | `sha_all` **✅逐字相同 7/7 轮**；**逐 prompt 16/16 相同**（每轮） |
| **graph** | `c2c-b64-graph` vs `c2c-b128-graph` | `sha_all` ✅逐字相同；逐 prompt **16/16** |
| graph vs eager | `c2c-b64` vs `c2c-b64-graph` | 逐 prompt **16/16 相同** |

输出全文（每个 prompt 的 sha）在 `logs/raw/054-c2-draft64/arms/*.pass*.client.json` 的 `*_out_sha256_by_prompt`。

---

## 4. ★★ Q2：投机解码（SpecDecoding 四项 + 一条更硬的判据）

### 4.1 四项（四个主臂）

| 指标 | `c2c-b128` | `c2c-b64` | `c2c-d128` | `c2c-d64` | 判定 |
|---|---|---|---|---|---|
| `Mean acceptance length` | **1.685** | **1.685** | **1.685** | **1.685** | 不回退（**完全一致**） |
| `Avg Draft acceptance rate` | **13.69%** | **13.69%** | **13.69%** | **13.69%** | 同上 |
| `Per-position acceptance rate`（pos0..4） | `0.685,0,0,0,0` | `0.685,0,0,0,0` | `0.685,0,0,0,0` | `0.685,0,0,0,0` | 同上 |
| `drafts / drafted / accepted` | 4239 / 21195 / 2902 | 同 | 同 | 同 | 同上 |
| `Accepted / Drafted throughput`（日志行逐条） | 见 `arms/*.specdecode_lines.txt` | — | — | — | 同量级 |

★ **统计功效先自证**：基线**不是地板** —— pos0 接受率 **0.685**、`MeanAccLen 1.685`
⇒ 若 64 臂把 draft 读坏，这几项会明显塌（这正是它在 tiny 上**能**验 Q2 的原因；`mt=64` 下共 21,195 个 drafted token）。

### 4.2 ★★ 更硬的判据：draft **提案 token 序列**逐条对比（探针直读）

汇总数会互相掩盖（少几步 / 多几步 / 接受位置挪动都可能同分），所以我加了一条**逐条**判据：
把 `NPUModelRunner.propose_draft_token_ids()` 的返回值（每一步 draft 提案的 5 个 token）**全量落盘**，两臂逐条比对。

```
[C2-2C-PROBE v1] hooked vllm_ascend.worker.model_runner_v1.NPUModelRunner.propose_draft_token_ids
                 -> NPUModelRunner.propose_draft_token_ids @0xfffee26114e0 pid=18202
[C2-2C-PROBE v1] DRAFT pid=18202 call#1 n=5 sha16=ea164d71c1a6dac4 ids=[61498, 126510, 126510, 126510, 126510]
```

| 对比 | 结果 |
|---|---|
| `c2c-b128` vs `c2c-b64`（B 档，eager） | **4367 次调用 / 4367 逐条相同**（不同 **0**；调用次数也相同 ⇒ 步数也相同） |
| `c2c-d128` vs `c2c-d64`（D 档，eager） | **4367 / 4367 逐条相同**（不同 **0**） |
| `c2c-b128-graph` vs `c2c-b64-graph`（图模式） | **1875 / 1875 逐条相同**（不同 **0**） |
| eager vs graph（同 block，前缀 1875 次） | 1637/1875 相同，**238 个不同**；首个不同在 `call#1250`，且 **128 与 64 两档的差异逐字相同** ⇒ 差异来自**图模式**（见 §6.4），**不是** ②c |

> ★ **探针纪律（AGENTS §5b）逐条**：
> ① **先装 hook 再 import**：`probe/usercustomize.py` 走 `site` 的 `usercustomize` 机制（放在 `PYTHONPATH` 最前、
>  **不占** `sitecustomize` 这个名字 ⇒ 不会顶掉 `$PKG/patch/sitecustomize.py` 里的 PGP/P2 hook 链）；
> ② **打印被替换的函数名 + 地址 + pid**："hooked … @0xfffe…"；**这条纪律当场救了一次**：
>  第一版钩的是上游 `SpecDecodeBaseProposer.propose` / `DSparkSpeculator.propose`，横幅打了"已安装"但 **0 次调用** ——
>  本机 dspark 的真入口是 `NPUModelRunner.propose_draft_token_ids()`（`vllm_ascend/worker/model_runner_v1.py:1752`）；
> ③ **热路径逐次 trace**：4367 行落盘 ⇒ 可证伪"函数没被调用"；④ **一个 target 一个 finder 实例**（7 个 target 各自独立）；
> ⑤ `find_spec` 用 `importlib.util.find_spec`（尊重别人的 meta_path 重定向）+ `try/finally` 恢复自己。

---

## 5. Q3：容量（实测 vs 050 §1.6 零参数模型）

### 5.1 模型先自证（`agents/C2_draft64/scripts/c2_model.py`，可复跑、零拟合参数）

```
Σslot_pages = 3 × max(kv+index(ratio2), state, swa, draft) + max(kv+index(ratio1), swa)
P(b)        = cdiv(min(window−1 + max_in_flight, max_len), b) + 1
BPR         = cdiv(max_len, 128) + 1 + 10×P(128) + P(draft_block)
num_blocks  = avail_bytes // Σslot_pages − 1
tokens      = int(num_blocks / BPR × max_len)
```

**9 个已知实测点逐字复现（0 个不符）**：tiny 无 draft B/C/D = 22,719 / 33,295 / 43,469；
tiny +draft128 B/C/D = 20,826 / 20,826 / 23,651；8 卡 +draft128 B/C/D = 427,643 / 427,643 / 485,610。

### 5.2 预测（可判伪，**符号相反**）

| 臂 | Σslot_pages | BPR | 预测 | 相对 128 臂 |
|---|---:|---:|---:|---|
| tiny `draft=128` B | 540,928 | 780 | 20,826 | — |
| tiny `draft=64` B | **540,928（不变）** | **844** | **19,247** | **×0.9242（降！）** |
| tiny `draft=128` D | 476,416 | 780 | 23,651 | — |
| tiny `draft=64` D | **282,880** | **844** | **36,825** | **×1.5570（涨）** |

### 5.3 实测（★ 4/4 逐字命中）

| 臂 | `GPU KV cache size`（实测） | 预测 | 判定 | 机制探针（`[R8-SLOTS]` 第 0 槽） |
|---|---:|---:|---|---|
| `c2c-b128` | **20,826** | 20,826 | ✅逐字 | `draft=131072 capacity=131072` |
| `c2c-b64` | **19,247** | 19,247 | ✅逐字 | `draft=65536 capacity=131072` |
| `c2c-d128` | **23,651** | 23,651 | ✅逐字 | `draft=131072 capacity=131072 [draft-aware]` |
| `c2c-d64` | **36,825** | 36,825 | ✅逐字 | `draft=65536 capacity=66560`（SWA binding） |
| `c2c-b64-graph` | **19,247** | 19,247 | ✅逐字 | （同 b64） |
| `c2c-b128-graph` | **20,826** | 20,826 | ✅逐字 | （同 b128） |

★ **draft 页真的换了**（不是"补丁没生效"）：`[C2-2C-PROBE v1] draft-spec` 探针在**每个进程**打 12 行，
`env='64' block_size=64 storage_block_size=64 page_bytes=65536` vs `env='128' … page_bytes=131072`（见 `arms/*.draft_spec_probe.txt`）。

### 5.4 ★★★ 为什么"B 降 D 涨"：slot 层的**直接证据**（不是算术推断）

把各臂起服日志里的 `[R8-SLOTS]`（`core/deepseek_v41.py::plan_cache_slots` 自己的 trace）并排放：
`capacity = max(kv+index, aliases_max, draft)` —— 三者的**逐项数值**就是下面这张表。

| 臂 | slot 0：`kv` | `index` | `aliases_max` | `draft` | **`capacity`** | `legacy_capacity` | 标记 |
|---|---:|---:|---:|---:|---:|---:|---|
| `c2c-b128`（B 档基线） | 65,536 | 8,320 | **131,072** | 131,072 | **131,072** | 131,072 | — |
| `c2c-b64`（②c，B 档） | 65,536 | 8,320 | **131,072** | **65,536** | **131,072（纹丝不动）** | 131,072 | — |
| `c2c-d128`（D 档基线） | 33,280 | 8,320 | **66,560** | 131,072 | **131,072** | **66,560** | ★ **`[draft-aware]`** |
| `c2c-d64`（②c，D 档） | 33,280 | 8,320 | **66,560** | **65,536** | **66,560（减半）** | 66,560 | — |
| `c2c-neg-b64`（反例臂） | 65,536 | 8,320 | 131,072 | 65,536 | 131,072 | 131,072 | —（见 §6.2 的说明） |
| `c2c-b128-graph` / `c2c-b64-graph` | 同上（128 / 64 各自一行） | | | | 同上 | 同上 | 图模式**不改** slot 算术 |

**四句话读完这张表**：

1. **B 档**：`aliases_max = 131,072`（BF16 SWA 页 = 128×512×2）**本来就顶住** slot 0–2；
   ②c 把 `draft` 从 131,072 砍到 65,536，**但 `capacity` 一动不动（131,072）** —— 砍掉的是**不被 binding 的那一项**。
   ⇒ 唯一残留的效果是 draft 组**每请求页数**因"窗口跨块"从 65 → **129**（`P_draft = cdiv(min(127+16384, 8192), b) + 1`）
   ⇒ BPR **780 → 844**（+64 页/请求）⇒ `num_blocks/BPR` 变小 ⇒ **19,247 = ×0.9242（降）**。
   **这不是模型算错，是模型算对了**：改了不 binding 的项，只拿到副作用。
2. **D 档**：`aliases_max = 66,560`（int8 SWA 页 = 128×(512+4×2)）；
   ②c 把 `draft` 131,072 → **65,536**，**正好把 draft 从 binding 位置拉下来** ⇒ `aliases_max = 66,560` 接手
   ⇒ `capacity 131,072 → 66,560`（**减半**）⇒ Σslot_pages 476,416 → **282,880** ⇒ 容量 **×1.5570（涨）**。
3. ★★ **`c2c-d128` 那一行的 `[draft-aware]` 标记本身就是实测**：
   `legacy_capacity=66560`（不加 draft 的老口径）vs `capacity=131072` ⇒ **draft 就是 slots 0–2 的 binding 项**
   —— 这条在 `050` 里是**算术推断**，现在有了 slot 层的**直接读数**。
4. **两档方向相反、且都与 `BPR` 的增减一致** ⇒ 说明 `GPU KV cache size` 的两个输入（`Σslot_pages` 与 `BPR`）都被独立测到了，
   不存在"两个错误互相抵消"的可能。

★ slot 3（ratio-1，无 state / 无 draft）在两档各臂里**都是** `capacity=147,712`（B 档）/ `83,200`（D 档）
⇒ **②c 对 slot 3 零影响**（与 `050` §1.2 的"slot3 与 draft 无关"一致）。

---

## 6. ★★ 阳性对照：判据有没有判别力（AGENTS §5b 第 3 条）

### 6.1 故意缺陷是什么

`c2c-neg-b64` = **同一个 ②c 包 + 一个额外的、故意写错的覆写**：

```python
# [C2-NEGCTRL v1] 故意的缺陷（阳性对照）：行宽按 **128 行块** 算（②c 之前的隐含假设），
#   而真块大小是 64 ⇒ 块索引 ≥ max_len/2/64 时越出本行。
def max_num_blocks_per_req(self, vllm_config, max_len):
    return -(-max_len // (self.block_size * 2))
```

### 6.2 结果：**当场炸，且炸点正是"块表行宽"**

```
1 个请求 ok / 3 failed（warmup 256 token 过、**第一个 4096-token 请求就死**）
out_sha256 = 78fb5f112b9e331e…（≠ 干净臂的 0ebccb55…）
draft 提案调用 = 3 次（干净臂 4367 次）
EngineCore: ValueError: could not broadcast input array from shape (65,) into shape (64,)
  at pkg/neg2c/shadow/vllm_ascend/worker/block_table.py:116  (block_table.np[row_idx, start:start+num_blocks] = block_ids)
⇒ vllm.v1.engine.exceptions.EngineDeadError（引擎死，后续请求全失败）
```

### 6.3 读法

| 观察 | 含义 |
|---|---|
| 缺陷**只在序列超过 `max_len/2`** 时触发（256 token 的 warmup 过、4096 的挂） | 与"静默寻址错"不同：**这条路径有硬校验**（`ValueError`），所以**块表宽度写错不可能静默通过** |
| 我的 workload（4096+64 ⇒ 需要 65 项）**正好越过**那条线 | ⇒ 若 ②c 真的把 draft 的块表/页几何算错，**本次判据一定会看到**（不是"恰好没覆盖"） |
| 输出 sha / 提案序列 / 请求成功率 / 容量**四个判据全动** | ⇒ 判据有判别力；反过来说，§3/§4 里"两臂逐字相同"才**是**有效信息 |

### 6.4 图模式那一格（附带发现，**不是 ②c 的问题**）

| 对比 | 结果 | 读法 |
|---|---|---|
| `c2c-b128-graph` vs `c2c-b64-graph` | **1875/1875 逐条相同** | ②c 在**图模式**下同样是**逐条不变量** |
| eager vs graph（128 档前缀） | 1637/1875 相同；238 不同，首个 @`call#1250` | 差异**来自图模式** |
| eager vs graph（64 档前缀） | **同样的 238 个位置、同样的值**（首个 @`call#1250` 的 A/B 值逐字相同） | ⇒ 与 block 大小**无关**（128/64 两档差异逐字一致） |
| 差异的结构 | 14 个连续小块（1249–1251、1282–1284、…，间距 33）—— 正好在**请求边界** | 【推断】图捕获下的批形状/填充在请求边界处与 eager 不同 ⇒ draft 提案在边界步上不同 |
| 输出 sha（4 条臂两两比较） | **16/16 prompt 全部逐字节相同** | 该差异**不影响最终输出**（目标端验证吸收） |

★ 这条发现【实测】但归因【推断】（差 33 步的周期性与"每请求 ~33 个 decode 步"吻合）；
**它不影响 ②c 的结论**（128 与 64 的差异逐字相同）。

---

## 7. ★★ "tiny 能不能替代 8 卡"

### 7.1 能替代的（本次已用 tiny + 单 die 拿到**结论级**证据）

| 维度 | 证据 |
|---|---|
| **draft 组在场** | `num_nextn_predict_layers=3` + KV 容量与"有 draft 组"的模型逐字吻合（20,826 / 19,247 / 23,651 / 36,825） |
| **真算子 + 真 KV 路径** | 每次请求都跑 `npu_sparse_flash_mla`（TND）+ 真块表；draft 提案 4367 次/臂全程落盘 |
| **②c 的几何** | 4/4 容量点逐字命中（含**符号相反**的两格：B 降、D 涨） |
| **寻址正确性** | 输出 sha 逐字节 + 提案序列逐条（4367/4367）+ 四个 SpecDecoding 汇总数完全一致 |
| **窗口跨块** | workload 真的走到 65 项块表 / 2–4 块的窗口（§2 末），且**没有崩、没有错** |
| **图模式** | `Capturing CUDA graphs (decode, FULL) 0/25` 真的捕获；容量/sha 与 eager 一致；②c 在图模式下同样是逐条不变量 |
| **判据判别力** | 阳性对照臂当场炸（§6） |

### 7.2 **不能**替代的（必须由 8 卡臂回答）

| 维度 | 为什么 tiny 不行 | 状态 |
|---|---|---|
| **真权重数值** | dummy 权重下 pos0 接受率 0.685、pos1–4 ≈ 0；页大小改变可能带来 **fp 级归约顺序差异**（页语义不变，但 tiny 看不出来） | 【未确认】 |
| **TP8 / HCCL** | 单 die 没有 TP 通信；`037` 的抖动就是 TP8 层的东西 | 【未确认】 |
| **8 卡绝对容量** | BPR 从 844（tiny）→ **2600**（8 卡）；预测 **档 D 777,318（×1.6009）** | 【未确认】 |
| **卸载池 / 命中路径** | 本轮 `OFFLOAD_GB=0` ⇒ 实测 `prefix_cache_hits_total = 0`（pass2 三轮其实也是冷算）；**池需求 +5.0%** 与 `sw_chunks 1→2` 只在 8 卡口径有意义 | 【未确认】 |
| **多头接受长度** | tiny 上位置 1–4 接受率 = 0 ⇒ "多 token 连续接受"这条路径**没被覆盖**（提案仍算，但不会被接受） | 【未确认】 |

### 7.3 对 c0 那条 8 卡臂的**具体建议**（可判伪）

1. **容量**：档 D + ②c 应读到 **777,318**（若读到 485,610 ⇒ ②c 没生效；若读到 863k 附近 ⇒ draft 组没了）；
2. **输出**：与档 D 基线（485,610 那条）比 sha；★ 8 卡上同臂内会抖（`037`）⇒ **必须每臂重复 ≥4 次**再比"重复一致性"；
3. **投机解码**：`SpecDecoding` 四项**不许回退**（基线档 B 的 `MeanAccLen 1.50` 那种 ~10 drafted token 的读数**没有统计功效**，要用 `max_tokens ≥ 64`）；
4. **卸载**：`BlockStored:CPU` / `CPU→GPU>0` / `hits>0` / `replay ≪ fill` 四条 + `BlockRemoved:CPU == 0`；
5. **池配额**：②c 让 draft 组每请求 unit 2→3（**+5.0%**）⇒ `OFFLOAD_GB=56` 要复算。

---

## 8. 产物与复跑

| 件 | 位置 |
|---|---|
| 本日志 | `a2/logs/054-20260922-draft64-1die.md` |
| 原始数据（7 臂 + chain.log + analysis.txt + 脚本快照） | `a2/logs/raw/054-c2-draft64/`（1.4 MB；server.log 与 draft_calls 已 gzip） |
| 臂脚本（容器内） | `a2/agents/C2_draft64/scripts/c2_arm.sh`（+ `c2_arm_graph.sh` / `c2_neg_control.sh`） |
| 链驱动（主机侧，串行 + rc=75 退避） | `a2/agents/C2_draft64/scripts/c2_chain.sh` |
| 影子包生成（复用 TDC 生成器 + 注入探针 + 可选故意缺陷） | `a2/agents/C2_draft64/scripts/build_pkg.py` |
| 探针（`usercustomize`，先于 import） | `a2/agents/C2_draft64/probe/usercustomize.py` |
| 零参数容量模型（9 点对账 + 预测） | `a2/agents/C2_draft64/scripts/c2_model.py` |
| 判据汇总（Q1/Q2/Q3 + 反例 + graph） | `a2/agents/C2_draft64/scripts/analyze_arms.py` → `logs/raw/054-c2-draft64/analysis.txt` |
| 收数脚本（tar + COS，不用 scp） | `a2/agents/C2_draft64/scripts/collect_raw.sh` |

**复跑一次主链**（A3 上）：

```bash
bash ~/projects/dsv41-upstream-pr/tools/a3_chip.sh c2 --timeout 600 --name c2-build   -- \
  bash -c 'REBUILD_PKG=1 ONLY_BUILD=1 bash /work/agents/C2_draft64/scripts/c2_arm.sh'   # 造包（不占卡）
setsid nohup bash ~/projects/dsv41-upstream-pr/agents/C2_draft64/scripts/c2_chain.sh \
  > ~/projects/dsv41-upstream-pr/agents/C2_draft64/logs/chain.log 2>&1 &                # 四臂链（~1 h）
python3 ~/projects/dsv41-upstream-pr/agents/C2_draft64/scripts/analyze_arms.py \
  --out ~/projects/dsv41-upstream-pr/agents/C2_draft64/out                           # 判据表
```

**本轮没改任何生产代码**：`upstream-v41/` 只读；②c 的改动全部落在 `agents/C2_draft64/pkg/`（影子包）与本地生成器调用里。

---

## 9. 诚实边界（必须与上面一起引用）

1. **【未确认】** 真权重数值：tiny 的 dummy 权重对"页大小变化引起的 fp 归约顺序差异"不敏感 ⇒ 8 卡要自己跑 sha / 接受率；
2. **【未确认】** 卸载池（`OFFLOAD_GB`）与**命中路径**：本轮 `prefix_cache_hits_total = 0` ⇒ pass2 三轮**不是**命中路径，而是第三组冷算；
3. **【未确认】** 8 卡的图模式（本轮只测了 tiny 的单 die 图捕获）；
4. **【未确认】** `mt>1` 的连续接受（tiny 上 pos1–4 = 0）；
5. **【推断】** §2 的"两臂权重逐比特相同"来自 dummy loader 源码（每参数固定 seed + 只依赖形状/dtype）+ sha 逐字节相同；
6. **【实测但归因推断】** §6.4 的 eager/graph 提案差异（14 个请求边界小块）；
7. 本轮所有【实测】都在 **tiny + dummy 权重 + 单 die（Phy-ID 7）** 上；**没有**在 A2 上跑任何东西。

---

## 10. 本条日志的 md5

`md5sum 054-20260922-draft64-1die.md` ⇒ **见文件末尾行**（现算，见下）。
