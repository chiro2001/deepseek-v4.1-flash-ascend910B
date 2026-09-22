# A2 现在的部署选项（2026-09-22 15:3x）

# ⛔⛔⛔ **窗口前必读：`ENGRAM_DEVICE_INDEX` 必须显式设 `0` —— 不设就会走进 A3 那条崩掉的路**

> **2026-09-22 19:4x 主代理发现（这是本晚第 7 次"默认值 ≠ 生产"的静默降级，后果最严重的一次）。**
>
> ```
> shadow-pkg/scripts/serve_a2.sh:153    ENGRAM_DEVICE_INDEX=${ENGRAM_DEVICE_INDEX:-auto}
> ★ A2 生产（用户 09-20 / 09-21 的启动命令）  ENGRAM_DEVICE_INDEX=0
>   （reports/a2-draft-graph-20260920.md:106：「ENGRAM_DEVICE_INDEX=0，因 ret=207001 在 A2 上不可用」）
> ★ 而本交付脚本此前【完全不设这个变量】⇒ 继承 shadow 的 `auto`
> ```
>
> ## 为什么 `auto` 在 A2 上会把它**打开**（而这是危险的）
>
> ```
> ① auto 的探针 = engram_device_index.py::probe_host_mapping_capability
>    ★ 它只测一件事：aclrtHostRegister(4 KiB 页) 能否被接受
> ② ★★ 而 A2 的实测是【1/8/32/64 GiB 注册全过】（logs/065 §3c）
>    ⇒ 探针在 A2 上【会通过】⇒ auto 会把 device-index【打开】
> ③ ★★★ 而 A3 上正是这条路崩的（logs/069 实测原文）：
>      [DEVICE-INDEX] 能力探测通过：host mapping registered
>      [DEVICE-INDEX] Engram 表已映射为设备可寻址：L1=384006168行, L14=384016682行
>      ⇒ 注册 183 GiB ⇒ EH0012 × N ⇒ 起服失败（大池）或推理崩（小池）
>    ★ 决定性对照：A3 上 `ENGRAM=1 + pageable`（【一个字节都不注册】）照样出 EH0012 × 9
>      ⇒ ★★ 那个失败与池的 host 内存后端【无关】，是 **device-index 路径本身**
> ```
>
> ⇒ ★★ **统一解释：A2 生产之所以一直没问题，就是因为它显式关掉了 device-index。**
> ⇒ ★★ **而 `auto` 会在 A2 上把它打开 ⇒ 正好走进 A3 那条崩掉的路。**
>
> ## 处置
>
> 1. ★★ **默认值已改成与生产一致（`0`）**；显式给 `auto`/`1` 会**打印响亮警告**并列出上面三条依据。
> 2. ★★ **窗口命令必须显式写 `ENGRAM_DEVICE_INDEX=0`**（即使脚本已有默认 —— 显式写能在 dry-run 里一眼核到）。
> 3. ★ **上线判据新增一道**（放在最前面，因为 `EH0012` 出现在 KV cache 建立**之前**）：
>    ```
>    grep -c 'EH0012' <serve.log>          # ★ 必须 0
>    grep -c 'hdc disconnect' <serve.log>  # ★ 必须 0
>    grep -c 'DEVICE-INDEX' <serve.log>    # ★ 期望 0（证明 device-index 真的关着）
>    ```
>    ⚠️ 若 `EH0012` 出现 ⇒ **立刻回滚**，不要等到压测（它意味着后面必然崩）。

---

# ✅ **原「发布阻塞项」已解除（2026-09-22 19:0x，A2 实机实测）：`ENGRAM=1` + 卸载池 = 可以共存**

> ## ★★★ 决定性结果：A2 的三期并发注册探针【全部通过】
>
> 探针是 `docker exec` 进**正在生产服务的容器**里跑的 —— 那个容器里 **Engram 就是加载状态**
> （A2 生产 `ENGRAM=1`）。这正好测到了"A3 上失败、而我们最想知道"的那个条件。
>
> | 期 | 规模 | 注册 | 设备往返逐字节 | 耗时 |
> |---|---:|---|---|---:|
> | S1 | 8 × 4 GiB = **32 GiB** | **8/8** | **8/8** | 22.8 s |
> | S2 | 8 × 16 GiB = **128 GiB** | **8/8** | **8/8** | 36.8 s |
> | **S3** | **8 × 49 GiB = 392 GiB** | ★ **8/8** | ★ **8/8** | 104.6 s |
>
> ★ 自检也对得上：`rss_before 1.77 → rss_after 50.3`（ΔRSS ≈ 48.5 GiB ≈ 分配量 ⇒ **分配即常驻**）。
>
> ## ★★ 为什么这条推翻了 A3 的结论
>
> ```
> A3（host_mem_pool=1）：ENGRAM=1 ⇒ 只能注册 74.68 GiB，之后 31 次 207001，**起服失败**
> ★ A2（host_mem_pool=0）：ENGRAM=1 ⇒ **392 GiB 全过**，零 207001
> ```
> ⇒ ★★ **反直觉但明确：A2 的注册能力比 A3 强得多**（尽管它的 `host_mem_pool=0`，比 A3 差）。
> ⇒ ★ **A3 那个阻塞是 A3 特有的，不是方案本身的问题。** 对 A2 **不构成阻塞。**
>
> ## ⚠️ 一条新发现（**不是阻塞，但要在窗口里实测**）：池越大，H2D 越慢
>
> ```
>         H2D 逐 rank（GB/s）                                中位
> S1  32 GiB:  7.9 20.5  5.1 19.6 22.8 22.7 20.7 20.5      ~20.5   （2 个慢）
> S2 128 GiB: 20.0  5.0 15.1 14.4 18.4 22.5 22.9 22.6      ~19.2   （1 个慢）
> ★S3 392 GiB: 3.8  5.3  4.8  3.7 10.4 22.6  5.0 19.3      ★ ~5.0  （6 个慢）
> ```
> **机制【推断】**：`392 GiB / 4 KiB = 每进程 1280 万个页表项` × 8 ⇒ 地址翻译压力。
> **为什么不是阻塞**：这是探针的**合成形态**（注册完立刻 8 进程同时搬 256 MiB）；
> 生产里搬运是**按请求分散**的。⇒ **但窗口里要用真实 workload 实测**（见下面判据⑦）。
>
> ## 处置
>
> 1. ★ **窗口可以直接用 `NPU_OFFLOAD_HOST_MEM=registered`**（`serve_a2_offload.sh` 的默认值）。
> 2. ★ 上线判据新增/改写一条（原七道门里的第 7 道）：
>    ```
>    ★ replay TTFT 必须明显小于 fill（8 卡基线 12.3×）。
>      若只快 ~4× ⇒ H2D 实际只有 ~5 GB/s（池大了会这样）⇒ 仍是正收益，但要把"注册路线的收益
>      被池规模吃掉"记进变更单；若 ~1× ⇒ 退化成冷算，必须停下来查。
>    ```
> 3. ★ 保留原来的候选 A/B 作为**退路**（`pinned` / `pageable`）——
>    它们现在只在"A2 实测也失败"时才用，而实测是**通过**的。
> 4. ★ **仍要保留默认值那条修正**（`ENGRAM` 默认改成与生产一致的 `1`）——
>    它与本条独立：那条防的是"静默关掉 Engram"，本条说的是"开着也能共存"。

---

## （以下为原阻塞项记录，保留作演进链 —— 它现在描述的是 **A3** 的现象）

# ⛔⛔⛔ **A3 上的现象（2026-09-22 18:0x 实测）：`ENGRAM=1` + 卸载池 = 起服失败**

> **一句话**：**A2 生产是 `ENGRAM=1`，而按本文件的配置起服会撞 `207001` 失败。**
> 这一条**优先于本文件其它所有内容** —— 它不是"标定"问题，是"**根本起不来**"。
>
> ## 实测（A3 8 卡，Phy-ID 8-15，2026-09-22 17:40–17:52，主代理独立核实引擎日志）
>
> | 臂 | `ENGRAM` | **池注册行数** | 结果 |
> |---|---:|---:|---|
> | `p1a-tierB-dg0-offload` | **0** | **136** | ✅ 成功 |
> | `p1b-tierB-dg1-offload` | **0** | **136** | ✅ 成功 |
> | **`p2-engram1-tierB-dg1`** | **1** | ★ **105**（< 136，**没注册完**） | ⛔ **起服失败** |
>
> **失败点【实测】在 Engram 自己的代码里，不在池里**：
> ```
> File ".../models/deepseek_v41/engram..."
>     requests = build_request_ids(boundaries, device)
> File ".../models/deepseek_v41/engram..."
>     return torch.repeat_interleave(index, counts)
> torch.OutOfMemoryError: ... AclrtSynchronizeStreamWithTimeout(copy_stream) ... 207001
> [Error]: Failed to apply for memory.
> ```
> 伴随 **12 处** `rtsHostRegister execution failed, reason=driver err`。
> **顺序（行号）**：池注册首现 985 → 池注册末行 1317 → **Engram 失败 1536** → `rtsHostRegister` 失败 1546
> ⇒ **不是"Engram 先占住"，而是"池先注册、然后 Engram 拿不到它要的 host 资源"。**
>
> ## 与既有日志的关系
>
> | 出处 | 当时的说法 | 现在 |
> |---|---|---|
> | `logs/001 §4.2` | `ENGRAM=1` 时**连 32 MiB** 的 `aclrtMallocHostWithCfg` 都失败、**8/8 worker 全中 207001** | ★ **方向一致**，但**机制换了**：当年是 `aclrtMallocHostWithCfg`，现在是 **`rtsHostRegister` + Engram 的 `copy_stream`** |
> | `logs/016` 边界 #7 | 「【未确认】生产是 `ENGRAM=1`，本轮没用它复测」 | ★ **本格已闭合：会失败** |
> | 全部 8 卡臂（027/042/048/066） | 都是 `ENGRAM=0` | ⇒ **整套交付从来没在 A2 的生产配置下跑过** |
>
> ⚠️ **A2 上可能更严**：A3 的 `host_mem_pool = 1`，**A2 是 0**（`logs/065` 实测）⇒ 同一条路在 A2 上未必更好。
>
> ## 处置：三条候选（**正在 A3 上跑，A 优先**）
>
> | 候选 | 做法 | 判据 |
> |---|---|---|
> | **A（最优先）** | **扫池子大小**：`OFFLOAD_GB` = 8 / 16 / 32 / 48 … + `ENGRAM=1` | 起服成功 **且 池注册行数 = 136** 且无 `207001` 且 Engram 真加载 ⇒ 找出**共存阈值** |
> | **B** | 换池后端：`NPU_OFFLOAD_HOST_MEM=pinned` / `pageable` + `ENGRAM=1` | 同上（`logs/001` 当年撞的就是 `pinned`，值得复验） |
> | C（更贵，暂缓） | 调 Engram 的注册顺序/时机 | — |
>
> ## ★★ 如果 A/B 全失败 ⇒ **只能二选一，且这是用户决策**
>
> ```
> 选项 1：关 Engram（ENGRAM=0）+ 卸载        ⇒ 质量降级（Engram 是模型的组成部分：2 层 + 206 GiB 表）
> 选项 2：保留 Engram，不开卸载              ⇒ 没有长上下文池 ⇒ 回到"KV 被踢出 HBM 就重新 prefill"
> 选项 3：等修（需要定位"谁在抢谁的资源"，可能要动 Engram 或池的分配顺序）
> ```
> ⇒ ★ **这三条没有技术上的"免费午餐"** —— 必须由用户按"质量 vs 服务能力"来权衡。
> 在得到结论之前，**不要把本文件的起服命令用在 A2 生产上**。
>
> ---
>

> **一句话**：**档 B / 档 C / 档 D 的"容量 + 功能三判据"都已在 8 卡真权重上实测通过**；
> ★★ **但有两条挂在台面上的保留意见**（2026-09-22 14:2x 新增第 2 条，**正在定性**）；
> **唯一的阻塞（要用户做的那件事）仍然是 A2 本机的池后端探测**（§0）。
> 标记：**【实测】/【推断】/【未确认】**。
>
> ### ⚠️⚠️ 两条保留意见（**不藏，直接放最前面**）
> 1. ✅ **已结案（2026-09-22 15:2x）：档 D 的接受率不降反升** —— 见 §3 的实测对比。
>    （此前那条"样本不足 ⇒ 待三臂判决"的保留意见**已作废**：真实 workload 下 `MeanAccLen` 档 D **2.69** ≥ 档 C **2.46**。）
> 2. ★★★ **已定性（2026-09-22 14:5x）：那 2/16 的差异不是 int8 缺陷，是「判据判别力不足」**
>    —— **决定性证据：档 B（BF16 无损池，四个 int8 开关全 0）的热臂，给出的退化 token 与档 C（int8）逐字相同**
>    ⇒ 差异来自**热/取回路径本身**；加上 `max_tokens=1` 只生成 **1 枚 token**、候选是**近平局**（全是空白类 token）
>    ⇒ ★ **「逐字相同」这条加强判据在该口径下不成立是正常的**（详见 `logs/062`）。
>    ★ **目标三判据不受影响**（档 C **12.50×** / 档 D **12.87×**）。
>    ⚠️ **但仍不许外推成「int8 已证明保真」** —— **正面判据（KV 级逐字节）仍未跑**；
>    且 **档 D 在 prompt 9 上留一格【未确认】**（`' '` vs 档 B/C 的 `'_'`）。
> 3. ⛔⛔⛔ **2026-09-22 19:3x 新增（v2）：档 D 有一条【会让 decode 输出错误】的缺陷 —— 不要开 `KV8_FULL=1`**
>    ```
>    单卡 c2，同进程、同一时刻、单变量对照（prompt=512 / mt=16 / 有前置负载）：
>      请求 logprobs 的 8 条  ⇒ ★ 5 条 NaN（服务端 400）
>      不要 logprobs 的 8 条  ⇒ ★ 0 条 NaN（8/8 OK）
>    而同一批 prompt 在【流式、不要 logprobs】下：same=5/8，mismatched=[4,5,7]
>    ```
>    ### ★★★ 三档同强度对照（**判决表，主代理独立复算过**）
>
>    ```
>    档     三轮任一不同（fill/replay/replay2）   NaN 失败数   same
>    B      []                                    0           8/8    ✅ 完全干净
>    C      []                                    0           8/8    ✅ ★ 与 B 同一判据下也完全干净
>    D      [0, 3, 4, 5, 7]                       10          5/8    ⛔
>    ```
>    ★★ **档 D 的「三轮任一不同」逐元素等于它的 NaN 失败集** ⇒ **统一假说【证实】**：
>    那不是"logprobs 的序列化问题"，**是 decode 输出本身被污染**；
>    logprobs 只是把它**暴露成 400 的显影剂**。
>    ⚠️ **所以「生产不请求 logprobs 就不会有问题」是错的** —— 不请求只是**看不到 400**，
>    仍会**静默拿到错的 token**。
>
>    ### ★ 损坏的**时点差异**（这是"累积型"缺陷的指纹）
>
>    ```
>    p4 / p5 / p7 ⇒ 第 2 轮就坏（fill != replay）
>    p0 / p3      ⇒ ★ 第 3 轮才坏（fill == replay，只有 replay2 看得见）
>    p1 / p2 / p6 ⇒ 3 轮全同、且从不 NaN  ⇒ 从未损坏
>    ```
>    ⇒ ★★ **判据强度教训（对以后所有验证适用）**：
>    `fill != replay`（2 轮）**漏掉了 40% 的损坏**（`[4,5,7]` vs 完整的 `[0,3,4,5,7]`）。
>    **"同输入应一致"的判据必须跑 ≥3 轮取"任一不同"** —— 因为"延迟到第 N 轮才显现"
>    正是**进程内累积型**缺陷的典型形态。
>
>    ⇒ ★ **档 D 挡死**；推荐配置**不含 `KV8_FULL`/`KV8_PREFILL`**。
>    ⇒ **档 C 不受影响**（且现在是**同强度判据**下的对照，不再是"侥幸通过"）。

---

## ★★★ 先说清一件事：**A2 的模型与我们实测用的模型不是同一个**

| | 模型目录 | 我们从哪里知道 |
|---|---|---|
| **A2 实际用的** | `/home/<user>/models/out/`**`v41-w4a8-flat`** | 用户 09-20/09-21 的启动命令（主机名 `a2`）；发布仓 `reports/a2-*.md` 里也是这个 |
| **A3 上我们实测用的** | `~/models/out/`**`v41-w4a8-engram-dr-vision-qrot-mtpq`** | `logs/001` / `042` / `048` **全部 8 卡臂** |

### 两者的共同点（**从 A2 自己的预检输出读出来的**，不是推断）
```
info    engram_layer_ids=[1, 14]  engram 权重条目=['engram_extra.safetensors', 'engram_int8']  optional/quarot=True
OK    Engram 配置与权重都在（2 层）
OK    有 mtpq 分片（4 个，推荐配置）   ← ★ **A2 也有 DSpark draft 组**
OK    有 vision 分片 / OK 有 optional/quarot.safetensors
```
⇒ ★★ **A2 的模型有 Engram（2 层）与 mtpq（= DSpark draft 组）**
⇒ 所以 **「draft 组顶住 slots 0–2」这个结构结论在 A2 上应当同样成立**
（这正是 **档 C 在 A2 真权重上容量 ×1.0000** 的根因）

### ⚠️ 但有一条**未确认**、且它会改数字
**两个模型的 Engram 规模是否一致** —— A3 那份的 Engram 是 **int8 206 GiB**；
A2 的 `flat` 版本**大小未知**（名字里的 「flat」 可能意味着某种精简）。
⇒ Engram 占用不同 ⇒ **`Available KV cache memory` 不同** ⇒ 容量数字要按 A2 的实测量重算。

### ★ 上线前的**第 0 步**：先跑模型自检（**它已经存在，别绕过**）

发布包里有 **`tools/check_model_dir.sh`** —— 它检查的正是我们关心那几项：
```
Engram 配置与权重都在（2 层）/ 有 mtpq 分片 / 有 vision 分片 / 有 optional/quarot.safetensors
WARN  无法判定 vision 是否为 qrot 修复版（若未修复，视觉约 10/23 而非 23/23）
```
★ **`scripts/run_test.sh` 会自动调它**（预检阶段），但**我们给出的三条命令直接调 `serve_a2_offload.sh`、绕过了它**。
⇒ **上线前先单独跑一次**：
```bash
bash tools/check_model_dir.sh <A2 的模型目录>
```
★ 看两件事：
1. **`Engram … 2 层` + `有 mtpq 分片（4 个）`** ⇒ 与 A3 那份模型的**结构一致**（见上一节）；
2. ⚠️ 那条 **`WARN 无法判定 vision 是否为 qrot 修复版`** —— 若是未修复版，**视觉只有 ~10/23**。
   ★ **怎么确认/怎么修**（⚠️ 如实说明：那两条命令名写在 `check_model_dir.sh` 的 WARN 文案里，
   但**这两个工具在本发布包里没有** —— 我一开始照抄了文案，核对时发现它们不存在）：
   * 包里**能用的**是 `tools/vision_accuracy_check.py` ⇒ 起服后直接跑它，
     看 **`cases: 23  pass: 23`**（我们 `047`/`048` 的实测口径就是它）；
   * 若它是 **~10/23** ⇒ 那是 vision 的 qrot 未修复 ⇒
     **向用户报这个结论**（工具在别处，本包不含）—— ★ **不要**自己伪造一个修复流程。

### ★★ 上线第一个动作就是量这一行（**一行，起服早期就打印**）
### ★★ 上线第一个动作就是量这一行（**一行，起服早期就打印**）
```bash
grep -E 'Available KV cache memory|GPU KV cache size' <serve.log>
```
| A2 量到的 `Available` | 含义 |
|---|---|
| **≈ 14.40 GiB** | ★ 与 09-20 那次**逐字吻合** ⇒ 模型/配置没变 ⇒ 按 §B0 的方法换算期望值即可 |
| 明显更大 | ⇒ 模型或 `GPU_UTIL` 变了（`flat` 可能省了 Engram）⇒ **容量会比 A3 好**，按 §B0 重算 |
| 明显更小 | ⇒ ⚠️ 先查是不是 Engram 开了（`ENGRAM=1` 会多占宿主 + 显存）或 `GPU_UTIL` 调低了 |

★ 注意 **`ENGRAM=0` 是我们起服脚本的默认**（因为 Engram + 卸载池曾撞 `207001`）——
但 A2 的模型**带 Engram 权重**，`ENGRAM=0` 意味着**不加载那张表**
⇒ **这也会让 `Available KV cache` 与「带 Engram 跑」时不同** ⇒ 量的时候**记下 `ENGRAM` 的值**。

## ★★ 三条命令走完（在 A2 上照抄即可）

```bash
# ① 池后端探测（不占卡、不加载模型；服务在跑也不用停）—— **唯一的阻塞**
A2_CONTAINER=dsv41-a2 A2PROBE_FLOOR_GIB=300 LIGHT=1 bash a2/scripts/a2_one_shot_probe.sh
#    ★ 看 `★ 注册内存的设备往返判据 = True/False`（H2H 通过不算数）

# ② 造 shadow-pkg（在 A2 本机；不依赖任何开发机）
PKG=<dsv41-release 路径> DST=$HOME/shadow-pkg bash a2/scripts/make_shadow_pkg.sh

# ③ 干跑（**会打印真实挂载清单**）→ 起服
DRY=1 SHADOW_PKG=$HOME/shadow-pkg MODEL=<A2 的模型目录，见上方「A2 的模型」一节> KV8_SWA=1 KV8_RING_FP16=1 \
    bash a2/scripts/serve_a2_offload.sh
#    ★ 看 `[a2-dry] MOUNTS(NN):` 里有没有那 **7 个 kv8-int8-pkg 件**（档 C/D 的必需件）

SHADOW_PKG=$HOME/shadow-pkg MODEL=<A2 的模型目录，见上方「A2 的模型」一节> OFFLOAD_GB=56 MAX_LEN=131072 MAX_SEQS=16 \
  NPU_OFFLOAD_HOST_MEM=registered KV8_SWA=1 KV8_RING_FP16=1 \
    bash a2/scripts/serve_a2_offload.sh
```
★ **这三条已在发布包布局下从 GitHub 全新 clone 验过**（档 B / 档 C 两条路径都走通、**dry-run 能打出真实挂载**、发布仓零污染）—— 见 `logs/055` §5.0 / §5bis。

> ### ★★★ 2026-09-22 15:3x：**档 C/D 的挂载件从 1 个变成 7 个**（一个已修的交付缺口）
> 此前文档只写"挂 `kv8-graphsafe/dsa_v41.py` 就能起档 C" —— **那是错的**：
> 档 C/D 实跑时挂的是 **7 个整文件**，而发布包里原本**只有 1 个**。
> ⇒ 已新增 `patches/kv8-int8-pkg/`（6 个整文件 + README，md5 与 `arm.out` 台账**逐字相同**），
> 并由 `make_shadow_pkg.sh` 在检测到 `A2_KV8` / `A2_KV8_SWA` / `A2_RING_FP16` 时**自动挂上 7 个**，
> **缺一个就 die**（不静默降级成档 B）。详见 [`logs/055`](../logs/055-a2-launch-path.md) §5bis。

---

## ⚠️ 一条贯穿全部结论的**外推**（必须先说清）

★ **本文件、`DELIVERY.md`、`logs/*` 里的所有「实测」数字，全部来自 A3**
（`A3-node1` 的 **Phy-ID 8–15**，**8 × 910C**）—— 而目标是 **A2（8 × 910B3）**。

| 维度 | 为什么可以外推 | 为什么**不能**想当然 |
|---|---|---|
| **容量算术**（`GPU KV cache size`） | 它只由**页几何**（`Σslot_pages` / BPR / `avail`）决定，**与芯片无关** | ⚠️ `avail` 取决于**每卡可用显存**，而 B3 与 C 的 HBM 容量/驱动占用**不一定相同** ⇒ **`427,643` 这类绝对数要重测** |
| **卸载功能的四条判据** | 是**卸载层**的行为，与芯片无关 | ⚠️ 池后端不同（A2 `host_mem_pool=0`）⇒ **`P1_pinned` 那条路径必须重跑**（这正是 §0 探测要回答的） |
| **int8 的页几何**（`page_bytes=66,560` 等） | 纯算术 + 已由**单元自检**验证（`051` 三臂） | ⚠️ **算子是否支持**是该芯片的 kernel 属性 —— `015` 已实测「TND KV 在 arch22 没 kernel」，同类假设在 B3 上要重验 |
| **图模式兼容性**（`EE1016=0`） | 判据是**软件路径**（host 标量 / D2H），与芯片无关 | ⚠️ capture 的具体行为是**驱动/CANN** 的事 ⇒ A2 上第一次起图模式**仍要盯 `EE1016`** |
| **性能数字**（`12.50×` / `ms/step`） | 趋势可借 | ⛔ **绝对数不可借** —— A2 是 910B3（设备更慢、HCCS 带宽更低、TP8 通信更贵）⇒ **必须在本机重测** |

⇒ ★★ **正确的读法**：**「能力」（能不能跑通、页几何对不对、功能判据成不成立）可以借；
「数字」（多少 token、多少 ms、多少倍）必须在本机重测。**
⇒ ★ 这也是为什么 §0 的池后端探测被列为**唯一的阻塞** ——
它是「能力」层里**唯一一个我们不能从 A3 借的**。

## ★★ 判据账（2026-09-22 16:0x）—— **上线后照着这张表核**

### A. 目标要求的「三判据」（**DRAM 卸载到底有没有生效**）

| # | 判据 | 档 B | 档 C | 档 D | 怎么核 |
|---|---|---|---|---|---|
| 1 | `BlockStored(medium=CPU) > 0`（**能存**） | 29,436 | 29,436 | 29,436 | `curl :PORT/metrics | grep kv_offload_store` |
| 2 | `CPU→GPU` 搬了字节（**能取**） | 21.52 GB | **21.19 GB** | 12.11 GB | `grep kv_offload_load_bytes_total` |
| 3 | `external_prefix_cache_hits > 0`（**真命中**） | 901,120 | 901,120 | 901,120 | `grep external_prefix_cache_hits_total` |
| ★ | **replay ÷ fill**（**取回比重算快**） | 12.87× | **12.50×** | **12.87×** | 两次 TTFT 之比 |
| ★ | `BlockRemoved(medium=CPU) == 0`（**没被踢**） | 0 | 0 | 0 | `grep kv_offload_block_removed` |

★ **全部为 8 卡真权重实测**（`logs/042` / `048` / `050`）；**上线后必须自己再核一遍**，
因为 A2 的池后端（`registered` vs `pinned`）与 A3 不同。

### B0. ★★★ **A2 本机其实已经有一组实测**（`KV-CACHE-ACCOUNTING.md` §1）—— 但它只给出**旧口径**的期望

> ## ⛔⛔ **先读这一格：`427,643 / 485,610 / 777,318` 这些数是"4 GiB 预算"下的探针读数，不是机器容量**
>
> 2026-09-22 16:5x 主代理自己踩了这个坑（把 485,610 说成"A2 能放多少"）⇒ 记在这里防复发。
>
> **8 卡那批 int8 臂全都带 `--kv-cache-memory-bytes 4294967296`**（主代理为了臂间可比、也让每臂跑得快设的）。
> ⇒ 它们是**页几何探针**：在同一份 4 GiB 预算下，不同配置各能买多少 token。
>
> ```
> A3 8 卡臂 : 4 GiB 预算   + max_len=133120  + max_seqs=32  →  427,643 / 485,610 / 777,318
> A2 生产   : 14.40 GiB 可用 + max_len=1048576 + max_seqs=4  →  3,498,354   ← ★ 真实容量
> ```
>
> **两台机器各有一个"生产读数"**（都不设上限），它们才是真实容量：
> | 读数 | 可用 KV 显存 | `GPU KV cache size` | B/token |
> |---|---:|---:|---:|
> | A3（910C） | 15.82 GiB | **3,842,534** | 4420.7 |
> | **A2（910B3）** | **14.40 GiB** | ★ **3,498,354** | 4419.8 |
>
> ⇒ ★ **两个生产读数的 B/token 只差 0.02%** ⇒ **每 token 的账与芯片无关**；
> 两者的容量比 = `3,498,354 / 3,842,534` = **0.9104**（因为 A2 可用 KV 显存少 1.42 GiB）。
>
> **而 A2 生产读数 vs A3 探针读数（档B 427,643）差 8.18×，分解如下**：
> | 来源 | 倍数 |
> |---|---|
> | KV 预算 4 GiB → 14.40 GiB | ×3.60 |
> | **max_len 133120 → 1048576**（摊薄每请求固定开销） | ×2.02 |
> | 小计 | ×7.28 |
> | 残差（A2 模型 `flat` 与探针几何的差异） | ×1.12 |
> | **合计** | **×8.18** ✅ 与实测一致 |
>
> 第二条容易漏：`GPU KV cache size = num_blocks / BPR × max_len`，而 BPR 含**每请求固定开销**
> （10 个 SWA 窗口块 + draft 块）：
> ```
> max_len=133120 : BPR=2471  vs 真实 token 块 1040  ⇒ 开销系数 2.376
> max_len=1048576: BPR=9623  vs 真实 token 块 8192  ⇒ 开销系数 1.175
> ```
> ⇒ 同样 4 GiB，在 1M 上下文下能买的 token 数差不多是 128K 下的两倍多。
>
> ### ★★ 能跨芯片/跨预算传递的是**倍率**，不是绝对值
>
> 以 **A2 实测的 3,498,354（档 B 现状）** 为基准：
>
> | 配置 | vs 档 B | A2 外推 |
> |---|---:|---:|
> | **档 B（A2 现跑）** | ×1.0000 | **3.50M**【实测】 |
> | 档 C（SWA int8 + ring16） | ×1.0000（容量） | 3.50M（收益在**宿主** 197→150 GiB） |
> | 档 D（+ long-KV int8 + prefill） | ×1.1354 | ≈3.97M【推断】 |
> | 档 C + ②a（draft KV int8） | ×1.4650 | ≈5.12M【推断】 |
> | ★ **档 D + ②a** | ★ **×1.9126** | ★ **≈6.69M**【推断】 |
>
> ★ **②a 的倍率与 `max_len` 无关**（只改页大小、不动 BPR）⇒ 可以干净地跨预算传递；
> **②c 的倍率与 `max_len` 有关**（它改 BPR）⇒ 换 max_len 必须重算。
> ⚠️ 上表 ②a 两行是【推断】：②a 只在**单卡 tiny** 上验证过（`capture_finished=1`、输出 sha 与 BF16 基线逐字相同），
> **8 卡真权重未跑**。
> ⚠️ 另有一条**独立门槛**：`MAX_LEN` 提到 1M 需要 `--kv-cache-memory-bytes ≥ max_model_len × kv_per_token`
> （`045` §5.3），**不满足则 1M 直接起不来**（见 DELIVERY.md §906/§929）。

★ 这一节解决一个很容易搞错的问题：~~上表（B 节）的 427,643 / 485,610 全是 8×910C 的数，A2 是 910B3 ⇒ 到底该期望多少？~~
> ★★ **2026-09-22 16:5x 更正这个提法**：上表那两数与 A2 生产读数的差异，**主因不是"910C vs 910B3"**
> （芯片差只贡献 0.9104），而是上面那格说的 **"4 GiB 预算 + max_len=133120" vs "14.40 GiB + max_len=1M"**（合计 8.18×）。
> **把芯片比套到探针数上是无效操作** —— 探针数根本不是"机器容量"，没有可缩放的语义。

**答案：A2 上早就测过一组**（用户 09-20 那次部署，`util=0.90`）：

| 机器 | `Available KV cache memory` | `GPU KV cache size` | **B/token/rank** |
|---|---:|---:|---:|
| A3（910C） | 15.82 GiB | 3,842,534 | **4420.7** |
| **A2（910B3）** | **14.40 GiB** | **3,498,354** | **4419.8** |

★ 两次的 **B/token 只差 0.02%** ⇒ 每 token 的账**与芯片无关**（都是同一套页几何）。
⇒ **A2/A3 的容量比 = `3,498,354 / 3,842,534` = 「**0.9104**」**（因为 A2 的可用 KV 显存少 1.42 GiB）。

#### ⚠️⚠️ 但**不要**拿这个比例去乘上表的数字（这是个陷阱）
```
旧口径（4421 B/token，无 L1、无 int8）下的外推：
  档 B/C  427,643 → A2 ≈ 389,338
  档 D    485,610 → A2 ≈ 442,113
  ⛔ 这三个数**不能当期望值用** —— 因为新方案（L1 + int8 + ②c）**改变了 Σslot_pages 与 BPR**，
     而那个 0.9104 是**旧几何**下量出来的比例。
```

★★ **正确做法（两步）**：
1. A2 上**先量出新的** `Available KV cache memory`（就在 serve.log 里，一行）；
2. 用**同一个零参数模型**（`agents/C2_draft64/scripts/c2_model.py`，13/13 逐字命中）
   代入 A2 的 `avail` + 你要跑的那档的 `Σslot_pages`/BPR ⇒ **那才是 A2 的期望值**。
```bash
# A2 上量（不用等模型加载完，这行在起服早期就打印）
grep -E 'Available KV cache memory|GPU KV cache size' <serve.log>
```

★ 而且**三档的相对关系在 A2 上不变**（因为 Σslot_pages 是纯几何）：
**档 C = 档 B**、**档 D > 档 C**、**②c 会大涨** —— 变的是**绝对值**，不是**结构**。

### B. 容量判据（**先记下期望值，再对比**）

| 档 | HBM `GPU KV cache size` 期望 | 宿主实占期望 | 依据 |
|---|---:|---:|---|
| 档 B | **427,643** | **197.21 GiB** | `logs/042` |
| 档 C | **427,643**（★ 与档 B 相同 —— **容量不涨是正常的**，收益在宿主） | **150.01 GiB** | `logs/048` |
| 档 D | **485,610** | **144.63 GiB** | `logs/050` / `R_8card_int8` |
| ②c（预测） | **777,318** | 待测 | `logs/051`（**8 卡端到端在跑**） |

> ⚠️ ★ **最容易误判的一格**：**档 C 的 HBM 容量与档 B 逐字相同**（427,643）——
> 因为 slots 0–2 的 binding 是 **draft 组（BF16）**，int8 只压得动第 4 个 slot。
> ⇒ **别拿「容量没变」当「int8 没生效」**（`053` 专门记过这个陷阱）。
> ★ int8 的收益要**看宿主内存**（197.21 → 150.01 GiB）或 `[R8-SLOTS]` 的 slot 数值。

### C. 服务健康判据（**起服后先看这三条**）

```bash
grep -c 'P1_pinned.*ret=0'            <serve.log>   # 期望 8   ← 池后端生效
grep -c 'D2_offload'                  <serve.log>   # 期望 >0  ← 卸载层装载
grep -c 'alignment_chunk_count.*8'    <serve.log>   # 期望 >0  ← per-group bpc 生效
grep -c 'P2_poolsizing'               <serve.log>   # 期望 >0  ← L1 生效
```
★ 任一为 0 ⇒ **停，别压测**（脚本末尾也会打印这几条）。

### D. ★ 提交前的最后一道（**2026-09-22 新增**）
```bash
DRY=1 SHADOW_PKG=$HOME/shadow-pkg MODEL=<A2 的模型目录，见上方「A2 的模型」一节> KV8_SWA=1 KV8_RING_FP16=1 \
    bash a2/scripts/serve_a2_offload.sh | grep -E 'a2-dry.*MOUNTS'
#   ★ 档 C/D 必须含那 7 个 kv8-int8-pkg 件（MOUNTS 条数约 24）
```
⇒ 这一道能挡住「文件没进包」那一类缺口（本轮就抓到过三个）。

## 0. ★★★★★ **池后端探测**已完成（2026-09-22 16:32，全档通过）—— ⚠️ 但见上面那条发布阻塞项

> # ⛔⛔⛔ **先读这一格：`ENGRAM` 的默认值 —— 这是「能不能用」的前提，不是「标定」问题**
>
> **2026-09-22 17:5x，`a3_p1_offload_draftgraph` 发现（主代理独立核实全部证据）**：
>
> ```
> A2 生产默认：  shadow-pkg/scripts/serve_a2.sh:127   ENGRAM=${ENGRAM:-1}
>                => 用户的 run_test.sh 不传 ENGRAM => A2 生产 Engram 是【开】的
> 本交付脚本：   a2/scripts/serve_a2_offload.sh:148   原为 ENGRAM=${ENGRAM:-0}
>                => 照我给的命令上线，会把 Engram【静默关掉】
> ```
>
> ### 证据链（**全部为真，主代理逐条核实**）
>
> | # | 内容 |
> |---|---|
> | 1 | `logs/001 §4.2`：`ENGRAM=1` 时**连 32 MiB** 的 `aclrtMallocHostWithCfg` 都失败、**8/8 worker 全部命中 `207001`** => 不是「容量不够」，是**驱动侧 pinned/注册资源争用** |
> | 2 | `logs/016` 诚实边界第 7 条原文：「`ENGRAM=0`｜【未确认】**生产是 `ENGRAM=1`**。……本轮**没有**用 `ENGRAM=1` 复测」 |
> | 3 | **全部 8 卡臂（027 / 042 / 048 / 066）搜遍 `ENGRAM=1` = 零命中** => **整套交付从来没在 `ENGRAM=1` 下跑过** |
> | 4 | 而第 1 条测的是**旧池后端（`pin_memory`）**；现在的池子用 **`aclrtHostRegister(MAPPED)`**，而 **Engram 的 206 GiB 表用的正是同一个 API** => **旧结论既不能证明现在会挂、也不能证明现在不会挂**【未确认】 |
>
> ### 处置
>
> **1. 默认值已改成与生产一致**（`ENGRAM=1`）；显式 `ENGRAM=0` 时会**打印响亮警告**
>    （「这是质量降级，不是默认行为」）—— 因为**静默降级比响亮失败更危险**：
>    响亮失败可回滚，静默降级会带着错误假设一路跑下去。
>
> **2. A3 8 卡正在验**（臂 `p2-engram1-tierB-dg1-offload`，17:40 起，`ENGRAM=1` + 卸载 + draft 入图）：
>
> * `[P1_pinned] ... registered ... ret=0` 是否 **8/8**（对照：dg0/dg1 两臂都是 128 行 `ret=0`）
> * `aclrtMallocHostWithCfg 207001` 是否出现
> * 宿主池能否到 **197.21 GiB**
>
>    Engram 会让起服多 **~10+ min**（206 GiB 表注册）。
>
> **3. 上线判据新增一条**（排在七道门之前）：
>
> ```
> grep -E "Engram|engram" <引擎 serve.log> | head   # 必须看到 Engram 真的加载（2 层）
> ```
>
>    —— 否则可能是「**没装上**」而不是「装上了没冲突」。

> ### ⚠️⚠️ 起服前必读：`DRAFT_GRAPH` 的默认值是 **0**，而 A2 生产现在是 **1**
>
> ```
> shadow-pkg/scripts/serve_a2.sh:223    DRAFT_GRAPH=${DRAFT_GRAPH:-0}
> ```
> ⇒ **本节下面给的命令里没有 `DRAFT_GRAPH=1` ⇒ 照抄会把 draft 从入图退回 eager，
>   单流 88.7 → 54.7 tok/s（−38%）**。这不是报错，是**静默降级**（本日第 5 次同类）。
>
> ★ **实测确认透传是通的**（2026-09-22 17:0x，`DRY=1` 对照）：
> ```
> 不传            ⇒ [a2-dry] ... DRAFT_GRAPH=0
> DRAFT_GRAPH=1   ⇒ [a2-dry] ... DRAFT_GRAPH=1     ← 前缀赋值会继承环境，不会被丢掉
> ```
> 所以**必须显式带上**（A2 现在就是这个配置）。
>
> #### `DRAFT_GRAPH=1` 在 A2 上是已知可用的（有实测）
> | 项 | 值 | 依据 |
> |---|---|---|
> | 单流 decode | **54.7 → 88.7 tok/s（+62%）** | `reports/a2-draft-graph-20260920.md` |
> | ms/step | **−30.5（−47%）** | 同上（`[bneck] hp` 64.6–65.3 → 34.0–34.8） |
> | 稳态 A | **3.03** | 同上（健康区间 2.8–3.1） |
> | 精度 | Vision **23/23**、GSM8K **198/200** | `reports/draft-graph-investigation-20260920.md` §4.6 |
>
> #### ⚠️ 两条**未验证**的边界（都写在这里，别踩）
> 1. ★★ **`DRAFT_GRAPH=1` + int8 **从未同时开过****（2026-09-22 17:0x 全仓 grep 确认：
>    没有任何一条臂同时带 `DRAFT_GRAPH=1` 和 `VLLM_V41_KV8*`）。
>    而 **int8 与"投机解码"的交织是有名的坑** —— 见 §0b「int8 × 投机解码的两个失败形态」。
> 2. **P0-C（`DRAFT_GRAPH=1` + 并发 ≥16 ⇒ 引擎进入不可恢复坏状态）** 只在 **1 次观测**里出现、
>    6 轮未复现 ⇒ A2 的 `MAX_SEQS=4`（`capture_max=32`）**结构上够不到 conc≥16** ⇒ 【推断】安全，
>    但这是"够不到"，不是"已证明安全"。
>
> ★★ **而本文件的命令默认 `MAX_SEQS=16`（`serve_a2_offload.sh:41`）—— 那正好落在 P0-C 的触发区间里**
> （A2 现在生产用的是 **4**）。⇒ **第一步务必显式 `MAX_SEQS=4`**，与现网保持一致；
> 要往上抬并发，请**单独当成一次变更**来测（判据：连续两次 specdec `Mean acceptance length: 1.00`
> 且 `Accepted throughput: 0.00` ⇒ 已进入坏状态 ⇒ 重启恢复）。

> ### 0b. ★★★ int8 × 投机解码：A3 上实测的**两个失败形态**（`048` / `049`）
>
> 注意：这里说的是 **"投机解码（draft eager）+ int8"**，不是 "draft 入图 + int8"（后者从未测）。
>
> | # | 形态 | 触发时机 | 表现 | 根因 | 修法 |
> |---|---|---|---|---|---|
> | **①** | **捕获期 `EE1016`** | 起服、图捕获时（**档 C/D 都中**） | 8 rank **逐字相同**：<br>`Not_Supported(EE1016): Synchronizing a stream failed.`<br>`Reason: Stream (stream_id=31) during the capture stage is not supported.` | 两处 int8 读路径靠 **`query_rows == num_reqs`** 分"decode 快路/prefill 慢路"；spec-decode 下 decode 批是 `num_reqs × (1+5) = 6×` 行 ⇒ 判据为假 ⇒ 误入慢路 ⇒ 慢路里的 **`.item()`** 在捕获期做 host 同步<br>（`dsa_v41.py:436 kv8_ori_plane`） | `GRAPH_SAFE=1` + 挂 `dsa_v41.py`（`94aeebb7…`） |
> | **②** | **首个真实请求 `507057`** | 起服成功、warmup 过、**第一个真请求** | `SUSPECT REMOTE ERROR, error code is 507057` → `EngineDeadError`，客户端 `failed=2` | **档 D 专有**：cmp 面留旧路径时，block table 列宽不足 ⇒ 越界读表拿到垃圾"页号" ⇒ 乘 `cmpKvStride0` 落到**未映射地址** ⇒ 设备故障<br>★ **也可能静默算错**（垃圾页号落在已映射内存时） | 同上（`GRAPH_SAFE=1`） |
>
> ★ **对 A2 的直接含义**：A2 现在 `DRAFT_GRAPH=1`（draft 入图）。
> 形态 ① 的判据失效来自 **target 捕获期**（`query_rows` 是 spec-decode 造成的 6 倍），
> **与 draft 入不入图无关** ⇒ **A2 一开 int8 就会撞 ①**，除非 `GRAPH_SAFE=1`。
> `serve_a2_offload.sh` 现在会在"开了 int8 + 图模式"时**自动置 `GRAPH_SAFE=1`**（并打印警告）。
> ⚠️ 但 `GRAPH_SAFE` 的修复**只在 `DRAFT_GRAPH=0` 的 8 卡臂上验过** ⇒
> **`DRAFT_GRAPH=1` + int8 是全新组合**【未确认】⇒ 建议先上档 B（见下面的分步）。

> **判据全过**：**`1 / 8 / 32 / 64 GiB` 四档注册【全部 True】**，每档都做了**真实 H2D→D2H 逐字节对账**。
> ```
> [a2probe] 注册（匿名/普通内存）：1=True 8=True 32=True 64=True
> [a2probe] 判读：⇒ 候选 β 可行：用 mmap + aclrtHostRegister(MAPPED) 做池子，把 cpu_npu.py 的 pin_memory 换掉
> ```
> ⇒ ★ **走 `NPU_OFFLOAD_HOST_MEM=registered`**。池需 **~49 GiB/worker**（`logs/027`）而 **64 GiB 单块可注册**
> ⇒ **池可整体注册，不必分片**。
>
> | 实测项 | 值 |
> |---|---|
> | 注册上限 | **≥64 GiB ✅**（1/8/32/64 全过，含往返逐字节） |
> | `host_mem_pool` | **0** ⇒ 不能靠 `aclrtMallocHost` 撑池（必须 registered） |
> | H2D 提升 | pageable **5.3** → 注册后 **16–21 GB/s**（3–4×） |
> | 单次 pinned | **32 GiB ✓**（`logs/012` 那条"(4,8] GiB 上限"**作废**） |
> | 文件映射注册 | 1 / 8 GiB ✓（生产池**不走**这条，会回写磁盘） |
> | 64 GiB 注册耗时 | 22.5 s（⇒ 【推断】392 GiB 池首注册 ≈137 s 一次性成本，可接受） |
>
> ★ 详见 **[`logs/065`](logs/065-20260922-a2-probe-live.md)**（含三次运行的全过程 + **三个脚本静默失败**的复盘）。
> ⚠️ **仍有一条要盯的**（起服期，有现成判据）：本探针是**单进程单块**，生产是 **8 worker 各 ~49 GiB**
> ⇒ **第一次起服时看 `[P1_pinned] CPU pool ... registered dev=... ret=0` 是否 8/8 都出现**，
> 任一 worker 不是 `ret=0` ⇒ **停下来看，别直接压测**。

<details><summary>原始命令与判读（保留作复跑参考）</summary>

```bash
# 在 A2 宿主上（脚本自己 docker exec 进服务容器）——**服务在跑也不用停**
cd <dsv41-release>/a2/scripts
A2_CONTAINER=dsv41-a2 A2PROBE_FLOOR_GIB=300 LIGHT=1 bash a2_one_shot_probe.sh
```
**它回答四件事**：`host_mem_pool` / `pin_memory` 单次 vs 总量 / ★★ **`aclrtHostRegister` 能不能注册** /
★★★ **那块注册内存**能不能**真的走 H2D/D2H**（= β 路线的生死判据）。
**为什么必须做**：A2 的 `host_mem_pool = 0`，且这个模型在 A2 上 **Engram 206 GiB 注册曾失败** ⇒ **A3 全绿不代表 A2 全绿**。
**判读**：脚本末尾自带 `DECISION`。**看的是 `★ 注册内存的设备往返判据 = True`**（H2H 通过不算数）
⇒ 真 ⇒ `NPU_OFFLOAD_HOST_MEM=registered`；假 ⇒ 回落 `pinned` 并重新量池子上限。
**耗时/占用**【实测·A3 同脚本】：`LIGHT=1`（默认）**18 s**、宿主峰值 ≈40 GiB、**显存只用一个 256 MiB 张量**。

> ★★ **"占不占 NPU"（回答"能不能和服务共存"）**：**需要能用上设备**（`aclInit` + `aclrtSetDevice` + 一条 stream）
> —— 这步省不掉；但**不加载模型、不跑算子、不做图捕获、不抢 HBM 的 KV 池**
> ⇒ **可以和正在服务的 A2 共存**（`COPY_GIB=0` 可把显存占用归零）。详见 `logs/052`。
> ⚠️ **旧版脚本已废弃**：它少了 `aclrtSetDevice`（会让 `aclrtMallocHost` 一律报 `107002`，看着像"A2 不能用 pinned"），
> 而且 ctypes + `torch_npu` 同进程**会段错误**且**吞掉全部输出**（`logs/052` §1，A3 实测）。

> ★ 探测**不需要因 int8 改动** —— 池后端（内存 API）与 KV 量化（页几何）是**正交**的两件事。

</details>

---

## 0b. ★★ 第二步：造 shadow-pkg（**此前这一步会卡住**）

`serve_a2_offload.sh` 依赖 **shadow-pkg**，而它原来**只存在于开发机**（`~/projects/dsv41-upstream-pr/shadow-pkg`，
手工改出来的、**从没进过发布包**）⇒ 探测即使全绿，**第二条命令也会立刻打印「⚠ 找不到 shadow-pkg」并退出**。

```bash
# 在 A2 本机从本仓库自己造（不依赖任何开发机）
PKG=<dsv41-release 路径> DST=$HOME/shadow-pkg bash a2/scripts/make_shadow_pkg.sh
# 干跑确认参数（不起服务）：
DRY=1 SHADOW_PKG=$HOME/shadow-pkg MODEL=<A2 的模型目录，见上方「A2 的模型」一节> bash a2/scripts/serve_a2_offload.sh
```

生成器做 **5 处精确锚点插入**（锚点必须恰好命中一次，否则 **fail-closed 且不落盘**）+ 4 条 grep 自检；
**不写 dsv41-release 一个字节**（已实测）。详见 `logs/055-a2-launch-path.md`。
★ **注意**：这份 shadow **与开发机上那份不等价**（开发机还含别的任务的注入块）
⇒ **不要拿开发机的 arm 结论直接套 A2 的 shadow**（见 `patches/ARTIFACT-IDENTITY.md`）。

> ### ★★★ 2026-09-22 16:4x：**"全新 clone → 造 shadow → dry-run"已端到端验证过**
>
> 起因：前面连续两次交付的脚本都带**静默失败**（`logs/065` §3 / §3b.0），
> 所以这次**不再只做静态检查**，而是从 GitHub **全新 clone** 真跑一遍：
> ```bash
> git clone --depth 1 -b feat/kv8-dram-offload-pending \
>   https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B.git "$D/repo"
> PKG="$D/repo" DST="$D/shadow-pkg" bash "$D/repo/a2/scripts/make_shadow_pkg.sh"       # ✓ rc=0
> DRY=1 SHADOW_PKG="$D/shadow-pkg" MODEL=<假模型> ... bash .../serve_a2_offload.sh      # ✓ rc=0
> ```
> **验证到的两件事**（都是"照抄能不能跑通"的关键）：
> 1. **档 B 路径**：`[a2-dry] OK` + 参数行齐全；
> 2. ★★ **档 C 路径**（`A2_KV8_SWA=1 A2_RING_FP16=1`）：打出
>    ```
>    [serve_a2] [A2-INT8] 已挂 7 个整文件件（6 个来自 .../kv8-int8-pkg/vllm_ascend + dsa_v41.py）
>    [a2-dry] MOUNTS(16): -v .../shadow-pkg/scripts:/opt/dsv41/scripts:ro
>                         -v .../core/deepseek_v41.py:...  -v .../core/kv_cache_interface.py:...
>                         -v .../models/deepseek_v41/model.py:...      ← ★★ 少了它 SWA 会静默退回 BF16
>                         -v .../models/deepseek_v41/compressor.py:... -v .../ops/triton/...:...
>                         -v .../attention/kv8_prefill_triton.py:...   -v .../attention/dsa_v41.py:...
>    ```
> 3. **副作用已核对**：整个验证**没有污染 `dsv41-release`**（`git status --short` 为空）——
>    印证了生成器"不写发布仓一个字节"的承诺。
> ⇒ ★ **"用户照抄能跑通"这件事，现在有实测依据，不再只是设计意图。**

---

## 1. 档 B —— 现状，已验证

```bash
MODEL=<A2 的模型目录，见上方「A2 的模型」一节> OFFLOAD_GB=56 MAX_LEN=131072 MAX_SEQS=16 \
BLOCKS_PER_CHUNK='{"default":8,"swa":1}' PREFIX_MATCH_UNIT=32 ENGRAM=0 \
NPU_OFFLOAD_HOST_MEM=registered OFFLOAD_SCHED_PATCH=1 OFFLOAD_NPU_WORKER_PATCH=1 \
bash a2/scripts/serve_a2_offload.sh
```

| 判据 | 8 卡真权重实测 |
|---|---|
| replay / fill TTFT | 1,420.6 / 18,202.6 ms = **12.81×**（×1.2 池 ⇒ **14.31×**） |
| `CPU→GPU` | 25.69 GB（208 load job） |
| `hits` | 1,062,400 |
| `BlockRemoved:CPU` | **0** |
| 宿主实占 | **197.21 GiB**（= 24.65 GiB/worker × 8） |
| HBM KV cache | **427,643 token** |

**保留投机解码**（`--speculative-config` dspark，接受长度中位 3.58）、**图模式可用**。

---

## 2. ★★ 档 C —— int8 省 47 GiB 内存（★ **8 卡实测通过，发布件已复跑**）

> ✅ **2026-09-22 13:0x 解除警示**：此前那条"新 md5 上待复跑"**已完成** ——
> `sg-c-c-graph-b`（档 C 图模式，8 卡，md5 **`94aeebb7…`**）**全绿**，且与 `22cbf20c` 那轮**逐字节相同**：
> ```
> 捕获 9/9 [00:54] · EE1016=0 · 容量 427,643（= 档 B）
> fill  sha = d524172f9f5ae368…   ← ★ 与 22cbf20c 那轮逐字相同
> replay1 sha = bc2e797ab069f09ced… ← ★ 与 22cbf20c 那轮逐字相同
> hits 901,120 / load_bytes 21,188,968,448 B / replay 1,594.8 ms vs fill 19,936.0 ms = 12.50×
> ```
> ⇒ **档 C 在发布件上已成立**。详见 `patches/kv8-graphsafe/README.md` §3.0 与 `patches/ARTIFACT-IDENTITY.md` §1.1。

```bash
# 在档 B 之上：
KV8_SWA=1 KV8_RING_FP16=1        # ← 脚本会自动置 APC_ALIGN=3 与 GRAPH_SAFE=1
# ★ 前提：挂上 patches/kv8-graphsafe/dsa_v41.py（md5 94aeebb757d6d5708268754481a05e0a）
bash a2/scripts/serve_a2_offload.sh
```

| 判据 | 8 卡真权重实测（`FULL_DECODE_ONLY`） |
|---|---|
| 起服 | ★ `EE1016 = 0`、`capture failed = 0`、就绪 659 s、`static_kernel` 无降级 |
| **HBM KV cache** | **427,643**（与档 B **逐字相同** ⇒ 容量零退化） |
| **宿主实占** | ★ **150.01 GiB**（= 18.75 GiB/worker × 8）⇒ **×1.3146，省 47.20 GiB** |
| `BlockStored:CPU` | 29,436（逐字相同） |
| `CPU→GPU` | 21,188,968,448 B = **21.19 GB**（> 0 ⇒ 真命中） |
| `hits` / `queries` | **901,120 / 3,145,984** |
| replay vs fill | **1,608.2 vs 19,880.0 ms = 12.36×** |
| `BlockRemoved:CPU` | **0** |
| `fill sha` | `d524172f9f5ae368…`（与档 B **逐字相同**） |
| ★★ `replay1 sha` | `bc2e797ab069f09ced…`（**与档 C-eager 逐字相同** ⇒ 图模式 = eager） |

**代价**：int8 的 **+1.3~1.8%** decode 时延（`logs/028`/`034` 的区间）。
★ **它保留投机解码、图模式可用** ⇒ **是当前"性价比最高"的一档**。

**为什么 A2 上 HBM 容量不涨（×1.0000）**：A2 的池子实际是 **"3 个投机解码窗口页 + 1 个 long-KV 页"**
（`540,928 = 3×131,072 + 147,712`），而 draft 的窗口面**硬卡 BF16**（源码 `DeepseekV41DraftSWASpec.__post_init__`）
⇒ int8 只能压得动第 4 页。**tiny 没有 draft 组（`num_nextn_predict_layers=0`）⇒ 所以 tiny 上是 ×1.4655**。

---

## 3. 档 D —— ×1.1356 HBM 容量（⚠️ **容量/功能判据过了，但 19:3x 发现 decode 输出缺陷 ⇒ 当前不可上线**）

> ## ⛔ **先读这条：本节的"已全绿"指的是【容量 + 三条功能判据】，
> ## 而 2026-09-22 19:3x 的单卡判决发现档 D 有【decode 输出缺陷】⇒ 当前不要开 `KV8_FULL=1`**
>
> 详见本文件 §保留意见第 3 条。要点：
> * 请求 `logprobs` ⇒ 5/8 条 NaN（服务端 400）
> * 不请求（流式）⇒ 同一批 prompt 里 3/8 条 token 不同
> * ★★ 统一假说【待证】：**decode 步 logits 被损坏** —— 两种表现是同一处损坏的两个出口
> * ⇒ **风险是"静默拿到错 token"，不是"看到 400"**
>
> 下面这些读数仍然成立（它们验的是容量与图兼容），但**不足以支撑上线**：


```bash
KV8_SWA=1 KV8_RING_FP16=1 KV8_FULL=1 KV8_PREFILL=1    # 同理自动置 APC_ALIGN/GRAPH_SAFE
```
| 判据 | 状态（`sg-c-d-graph`，8 卡真权重，md5 `94aeebb7…`） |
|---|---|
| 起服 + 捕获 | ✅ 捕获 **9/9 [06:11]**、`/health=200`、`static_kernel` 无降级 |
| ★ 判据 0（致命项） | ✅ `EE1016=0 / Segfault=0 / Engine core init=0 / Worker died=0` |
| HBM KV cache | ★ **485,610**（×1.1356）—— 与 R 的档 D 臂 **逐字相同** |
| 四条判据 | ✅ `CPU→GPU` 12.11 GB>0 / `hits` 901,120>0 / `BlockRemoved:CPU`=0 / replay **12.87×** |
| ★★ 判据④（最强） | ✅ `replay1 sha` 与**同几何 eager 臂逐字节相同**（`8600507eb6b43bfa…`） |
| ★ 越界读是否被消灭 | ✅ `[SG-PPR]` 证明捕获期 cmp 面页数**由 shape 决定**（768 / 1536 页），不再是 `ppr=1` |
| ⚠️ 越界读的后果（更正） | **不是单一形态**：决策臂 `sg-c-d-cmplegacy` 实测到的是**崩引擎**（`507057 SUSPECT REMOTE ERROR`，第一个真实请求即死）；**同几何 A/B**（唯一差别=补丁开关）证明因果。⇒ 正确表述 = **"可能崩、也可能静默算错"** ⇒ **必须验 sha**，不许用"反正会崩"自我安慰 |
| ★✅ **接受率（已结案）** | ★★ **"档 D 降低接受率"这条被实测否掉**。真实 workload（8 × 4096 → `max_tokens=64`）下，**档 D 反而略高于档 C 基线**：<br>稳态 interval `MeanAccLen` **2.69 vs 2.46**、`AvgDraftAcc` **33.8% vs 29.2%**、Per-position **每一位都更高**（`.591/.355/.290/.237/.215` vs `.509/.311/.264/.208/.170`）。<br>★ 此前 `max_tokens=1` 下看到的 `1.00/0%` 与 `1.50/10%` 是**口径假象 + 样本量差异**（`Drafted` 10 vs 15），**不是档 D 的缺陷**。<br>★ **口径提醒**：两臂的 cumulative 分母不同（`num_drafts` 166 vs 214）⇒ **只能比 interval 行**；且**必须剔除 interval #1**（它只含 warmup 的 3 个 draft 步，会给出假的 `1.00/0%`）。 |

⇒ **档 D 的修法就在同一份 `dsa_v41.py` 里**（`_kv8_cmp_plane` 的 graph_safe 分支），
**已在 8 卡真权重上验完** ⇒ **档 D 可以上**（唯一保留意见是上面那条"接受率需另测"）。

---

## 4. ★★ 两条「解开 draft 天花板」的路 —— **②a 已实测失败，交付推荐是 ②c**

> ### ★★★ 2026-09-22 14:5x：【实测】②a（draft INT8，block **保持 128**）
> ```
> ddi-d-i8-graph（tier D）: capture_finished=1 ee1016=0 not_supported=0 capture_failed=0   ← ★ 图捕获成功
> probe: block=128 dtype=torch.int8 scale_dim=4 page_bytes=66560                        ← ★ draft 页 66,560
> GPU KV cache size = 39,846   ← ★ 与零参数模型的预测【逐字相同】
> ```
> ★★ **算术上 ②a 严格优于 ②c**（两者 Σ 相同 = 282,880，但 ②a 的 BPR 更小 = 2471 vs 2600，
> 因为 **②a 保持 block=128、没有「窗口跨块」的副作用**）：
> ```
> 8 卡预测：②c = 777,318（×1.8177）   ②a = 817,898（★ ×1.9126）
> tiny 实测：②c = 36,825               ②a = 39,846（★ 逐字命中模型）
> ```
> ⇒ **算术上 ②a 是唯一能碰到原始 ×1.84 目标的那条路**（预测 ×1.9126）——
> ★★★ **但 Q3 实测失败，②a 不可用**（`logs/056`）：
> ```
> RuntimeError: The previous device metadata submission has not been released
>   @ worker/device_metadata.py:74（触发形状 num_scheduled_tokens=6 + 5 个 spec token）
> ②a 臂：16/16 请求失败、0 条 SpecDecoding 读数
> 对照臂（同包同参数，唯一变量 DRAFT_INT8=0）：ok=16/16、sha 0ebccb55b30c…（与 054 四臂逐字相同）
> ```
> ⇒ ★★ **②a 特有，不是 harness**。
>
> ★★ **诊断臂进一步定位到「泄漏的那一次提交」**（`056` §4.4b）：
> ```
> submit#1 in_flight=False tasks=7 → release#1 ✅
> submit#2 in_flight=False tasks=1 frontiers=[(2, …)]      → ★★ 无 release#2   ← 泄漏点
> submit#3 in_flight=True  tasks=7（与 #1 逐字相同的 7 个）  → ⛔ 抛 RuntimeError
> ```
> ★ `submit#2` 的形状与其余每次**都不同**（**只 1 个任务**、`group_id` 在 target 的 7 任务提交里**一次都没出现过**）
> ⇒ 来自**另一个 builder（draft 侧）** ⇒ 「**release 缺口在 draft 侧的 execute 路径上**」这条
> **从推断升到有实测支撑**。
>
> ⚠️⚠️ **但「病灶是 `dsa_v1.py` 缺量化存取」这条必须降级**（`D_draftINT8` **主动**提出，主代理采纳）：
> 它的探针钩在 `dsa_v1.py::AscendDSAImpl` 上，**横幅打出来了但从未被调用** ——
> 真身是 `models/layer/attention/layer.py::DSAAttention`（`ops/dsa.py:35`）。
> 而且**首个异常在日志里彻底看不见**（`tuple` / `AttributeError` / `npu_scatter` / `ori_kv` 命中**全 0**）。
> ⇒ **正确的说法**：「要移植量化存取」是**【推断】，不是已证实的病因**；
> 下一个探针 target 应该是 **`DSAAttention`**，**不是** `dsa_v1.py`。
> ⇒ ★★ **别把「病因已证实」写进文档** —— 本条目只到「**②a 在单 die tiny 上不可用**」
> （**足以否决交付选项**），**不到「病因已定位」**。

### ②c 的细节（**仍是 ②a 失败时的回落**，8 卡端到端在 c0 排队）

| | 值 |
|---|---|
| 收益 | ★ **HBM ×1.8177**（**777,318** token，相对档 B 427,643） |
| 保留投机解码 | ✅ |
| 精度风险 | **无**（不改 dtype，draft 仍 BF16） |
| 改动面 | ★ **2 文件 / 2 处，默认关**（`dspark.py` 给 draft 自己的 `block_size`（env `VLLM_V41_DRAFT_BLOCK`）+ 放宽 `plan_cache_slots` 的**相等**检查为**整除**检查）—— 详见 `logs/051` |
| 已落地的证据 | ★ **`64` 档本来就在算子的块大小表里**（`_DSV4_BLOCK_SIZES[64][0][1] == 64`、`page_size_padded_t2 == 65,536`）；**三臂对称单元自检全绿**（`upstream` raise / `draftaware` ×1.0000 与 8 卡逐字同 / **`patched` 档 C 369,280、档 D 282,880**） |
| ★★★ **机制已在 slot 层实测**（单 die，`054`） | 四臂实测 `capacity = max(kv+index, aliases_max, draft)`：<br>• **档 B**：`aliases_max=131072` ⇒ draft 131072→65536 **capacity 纹丝不动（131072）** ⇒ 只拿到"draft 页数 130→259"的副作用 ⇒ **容量降 ×0.9242**<br>• **档 D**：`aliases_max=66560` ⇒ draft 131072→65536 **把 draft 从 binding 位置拉下来** ⇒ **capacity 131072→66560（减半）** ⇒ **容量涨 ×1.5570**<br>★ 且 `d128` 那行自带 `[draft-aware]`（`capacity=131072 legacy=66560`）⇒ **"draft 是 slots 0–2 的 binding 项"在 slot 层直接实测**，不再只是算术推断 |
| ★ 单 die 数值判据（`054`） | **输出**：B/D 各 7 轮 sha 逐字节相同、逐 prompt 16/16、跨臂 16/16；**投机**：两臂 `MeanAccLen 1.685 / AvgDraftAcc 13.69%` 完全一致，**提案序列 4367/4367 逐条相同**；**图模式**：b64 捕获 23 s / b128 9 s，`EE1016=0` |
| 已查清的风险 | 窗口跨 3~4 块（算子/块表/KV manager **都无假设**）；⚠️ 唯一硬编码 `kv8_ori_plane` 的 `pages_per_req=2` **只在 int8 平面上跑 ⇒ ②c 不走它** |
| 副作用 | DRAM 池 `sw_chunks` 1→2 ⇒ 该组每段 unit 2→3 ⇒ 总需求 **+5.0%**（`OFFLOAD_GB=56` 要复算） |
| ⏳ 还差什么 | **一条真权重端到端臂**（判据：容量 595,404（档 C）/ 777,318（档 D）+ 图捕获成功 + **`SpecDecoding` 四项不降** + sha 与冷算参考一致） |

---

## 5. 落地顺序建议

> ★★ **硬约束（用户决策 2026-09-22）：保留投机解码。** A2 是单流场景，DSpark 的收益不可替代
> ⇒ **⑤a（关投机换容量）已否决**；下表每一档、每一条待验路线都在 `--speculative-config dspark` 下成立。

```
1. ★ 先跑 A2 的探测（a2_one_shot_probe.sh）⇒ 决定池后端
2. ★ 上档 B（已验证）⇒ 拿到"16×128K + 12.8~14.3× 加速"
3. ★★ 再上档 C（已实测）⇒ 池从 197 GiB 降到 150 GiB，代价 +1.3~1.8% 时延
4. ⏳ 等 ②c / 档 D 的结论 ⇒ 若成立，HBM 容量再 ×1.82 / ×1.14
```

## ★★ 回滚（**逐条 + 怎么确认**，2026-09-22 17:1x 补）

★ **总原则**：所有能力都在 **`docker -v` 挂载层 + env** 上 ⇒ **没有一处写进镜像**
⇒ **回滚 = 换一组 env + 换一个 shadow-pkg**，不用重建镜像、不用改 dsv41-release。

| 要从哪档回到哪档 | 怎么做 | **怎么确认回滚成功** |
|---|---|---|
| **档 D → 档 C** | 去掉 `KV8_FULL=1` 与 `KV8_PREFILL=1` | 容量指纹 **485,610 → 427,643**；`[R8-SLOTS]` 的 `slot3 kv` 从 83,200 → 147,712 |
| **档 C → 档 B** | 再去掉 `KV8_SWA=1` 与 `KV8_RING_FP16=1` | ★ **容量不变**（B/C 都 427,643）⇒ 改看**宿主内存**（150.01 → 197.21 GiB）或 `[R8-SLOTS]` 的 slot 值 |
| **档 B → 现状**（无卸载） | 不设 `OFFLOAD_GB`（或不挂 `0001`/`0001b`/`0001c`/`0002`） | serve.log 里 **无 `D2_offload`**；`/metrics` 里无 `kv_offload_*` |
| **K_l1（省内存那块）→ 关** | `P2_POOL_PATCH=0` | serve.log 里 **无 `P2_poolsizing`**；宿主实占回到 ≈392 GiB（档 A 口径） |
| **②c（draft block 64）→ 关** | `VLLM_V41_DRAFT_BLOCK=128`（或删掉该 env） | 容量 **777,318 → 485,610**（档 D）或 **595,404 → 427,643**（档 C） |
| **整包回滚** | 用**旧的 shadow-pkg**（或不设 `SHADOW_PKG`，走原 `serve_a2.sh`） | ★ `GPU KV cache size` 与 `BUILD_INFO.txt` 里的 md5 都对回旧值 |

### ★★ 三条回滚纪律
1. ★ **回滚前先记下当前的值**（容量指纹 + 宿主实占 + `[R8-SLOTS]` 四行）——
   否则你**分不清「回滚成功」和「又发生了静默降档」**（两者都表现为「数字变了」）。
2. ★ **一次只改一项**（env 或 shadow），改完**核对指纹**再改下一项；
   同时改两项 ⇒ 又回到「不知道是谁的功劳」那个状态。
3. ★ **回滚不需要动 dsv41-release** —— 若你发现必须改仓库才能回滚，
   说明有一处能力**不在挂载层**，那本身就是**要报告的缺陷**
   （本项目的设计前提就是「挂载即可回滚」）。

**回滚**：所有能力都在 `PYTHONPATH` / `docker -v` 挂载层 ⇒ **去掉挂载即回现状**；
逐条回滚 = 对应 env 置 0（`KV8_*=0` / `APC_ALIGN=0` / `GRAPH_SAFE=0` / `P2_POOL_PATCH=0`）。
**没有一处写镜像。**
