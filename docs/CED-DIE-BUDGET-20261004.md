# CED-PD「卡数预算」可行性分析：≤14 die 能不能跑？

**日期**：2026-10-04　**执行**：`/root/ced_die_budget` 子代理　**机器**：a3-21（**只读**，本次未启停任何容器）
**仓库**：a3-21 `~/cedpd-repo`（本地 `main-merge` 为同一仓库的检出）
**标注**：【实测】= 有运行日志/逐张量账；【实测·代码】= 有明确代码门（行号给定）；【推断】= 由实测数字推导；【未确认】= 没证据。

---

## 0. 结论速览

| 候选配置 | P die | D die | 合计 | 判定 | 卡在哪一条 |
|---|---:|---:|---:|---|---|
| **A（问题指定）** | 4 | 8 | **12** | ❌ **不可行** | ① 连接器硬门 `P_TP ≥ D_TP`（4<8）② rank 映射算法本身要求 `P_TP ≥ D_TP` ③ P 只有 4 die ⇒ EP=4 ⇒ 专家权重 **64.80 GiB > 61.27 GiB/die** |
| **B（问题指定）** | 6 | 8 | **14** | ❌ **不可行** | ① 同一硬门（6<8）② **TP=6 对本模型非法**（`num_attention_heads=64`，64 % 6 ≠ 0） |
| 反向 12 die | 8 | 4 | 12 | ❌ 不可行 | D 只有 4 die ⇒ EP=4 ⇒ 专家 64.80 GiB > HBM |
| 反向 14 die | 8 | 6 | 14 | ❌ 不可行 | TP=6 非法（64 % 6 ≠ 0）；退到 TP2×DP3 则撞同一硬门且 D 池只剩 ~7 GiB |
| **现行交付** | 8 | 8 | **16** | ✅ **已实测（21/21 正确性 + 四轴性能）** | — |

**一句话**：本模型 **routed experts 259.19 GiB**，必须摊到 `EP` 个 rank 上；每 die 可用 HBM 只有 **61.27 GiB**。
⇒ 只要 P/D 是**两个独立引擎**，每个引擎最少 **8 die**（EP=8 ⇒ 32.40 GiB/die），两个引擎就是 **16 die**。
12/14 die 的四种切法（含两个反向）全部被**代码门**或**权重账**否掉，**不是配置问题，是必须改代码的问题**（清单见 §5）。

**直接回答上级的开放问题**：把 `MAX_LEN` 从 1M 降到 512K **对 die 数没有任何帮助** —— 池容量由「字节预算 ÷ 每块字节」决定，与 `MAX_LEN` 无关（§4）。

---

## 1. TP 是不是由 DEVS 数量派生？P/D 能否用不同 TP？

### 1.1 TP 与 DEVS 是两个独立输入

| 证据 | 内容 |
|---|---|
| `scripts/serve_a2.sh:84,89` | `TP=${TP:-8}`；`DEVS=${DEVS:-"0 1 2 3 4 5 6 7"}` —— **两个独立 env** |
| `scripts/serve_a3.sh:134-146` | `_tp=${TP:-8}`；然后**只校验数量**：`DEVS 有 N 张，而 TP×DP 需要 M 张 ⇒ 数量不匹配` 直接 FAIL（`I_KNOW=1` 可强行继续） |
| `scripts/serve_v2.sh:96,104` | `--tensor-parallel-size "$TP" --enable-expert-parallel`；`DP>1` 时追加 `--data-parallel-size` ⇒ **EP = TP × DP**（沿用 `docs/DP-TP-TRADEOFF-20260929.md` 的实测口径：A=TP4/DP2→EP8、B=TP2/DP4→EP8）【实测·0929】 |

### 1.2 连接器**设计上**支持 P/D 异 TP，但要求 `P_TP ≥ D_TP`

| 位置 | 内容 | 结论 |
|---|---|---|
| `experimental/ced/mooncake_hybrid_connector.py:1905-1928` | `_get_prefill_decode_size()` 从 `kv_connector_extra_config` 的 `prefill` / `decode` **两段**分别读 `tp_size/dp_size` | 支持异 TP 【实测·代码】 |
| 同文件 `:1782-1786`（`MooncakeConnectorWorker.__init__`） | `if self._prefill_tp_size < self._decode_tp_size: raise ValueError("prefill_tp_size ... must be greater than or equal to the decode_tp_size ...")` | **A(4<8)、B(6<8) 直接拒绝起服** 【实测·代码】 |
| 同文件 `:2426-2434` | `num_groups = prefill_tp*pp // num_kv_head`；`rand.sample(range(num_groups), max(decode_tp // num_kv_head, 1))`；MLA 下 `num_kv_head = 1` | 就算拆掉硬门，P=4/D=8 时 `rand.sample(range(4), 8)` ⇒ **`ValueError: Sample larger than population`** ⇒ 「多个 D rank 复用同一 P rank」是**必须新写的逻辑** 【实测·代码】 |
| 同文件 `:1743-1746` | `offset = pp_rank * self.tp_size + tp_rank`（用**本地** tp 复原远端 rank） | 异 TP 时端口/握手映射需同步改 |
| 同文件 `:1851-1853` | Mamba 分支另有 `P_TP == D_TP` assert | 本模型非 Mamba，**不适用** |

### 1.3 交付脚本把两侧**钉死**在 TP=8

| 位置 | 内容 |
|---|---|
| `scripts/serve_a3_pd.sh:54-60` | `TP=${TP:-8}; DP=${DP:-1}; if [ "$TP" != 8 ] \|\| [ "$DP" != 1 ]; then FAIL "该基线固定 TP=8、DP=1"` |
| `scripts/serve_a3_pd.sh:81` | `KV_CONFIG=...{"prefill":{"dp_size":1,"tp_size":8},"decode":{"dp_size":1,"tp_size":8}}` —— **写死** |
| `scripts/serve_a3_ced_single.sh:69` | 单实例夹具写死 `tp_size=1/1`（非交付路径，但同族硬编码） |

---

## 2. P 侧权重 / 显存

### 2.1 P **加载全模型权重**（40 层全建全载），只在 forward 里断在第 20 层

| 证据 | 内容 |
|---|---|
| `patches/files/model.py:1455-1457` | `if self._ced_prefill_only and layer.layer_idx == 20: layer.write_global_source_from_encoder(...); break` —— **只跳过 forward** |
| 同文件 `:1583-1588` | `_is_milestone_weight` 只过滤 `aligner./vision./image_/mtp.`，**没有任何按角色裁层的过滤器** ⇒ 40 层参数全部实例化+加载 |
| `:732-758`（`write_global_source_only`） | P 在 layer 20 只用 `hc_pre → input_layernorm → long_kv/indexer 投影`，**不需要 layer 20 的 MoE**（这一点是 §2.3 里「21 层」方案的基础） |
| **实测日志** `~/cedpd-repo/results/ced_p2b_0927_015401/serve.log` | `Loading model weights took 37.4330 GB`（≈ **34.86 GiB**）/ rank；D 侧同 run 家族 `39.4962 GB`（≈36.78 GiB，多出的 ~2 GB 是 DSpark 草稿层） |

### 2.2 权重账（逐张量，`docs/DP-TP-TRADEOFF-20260929.md` §5.1）【实测】

| 类别 | 大小 | 占比 | 切分 |
|---|---:|---:|---|
| routed experts | **259.19 GiB** | 95.2% | 按 **EP** |
| attention（MLA/indexer/compressor/norm） | 10.19 GiB | 3.7% | 按 **TP** |
| shared experts | 1.32 GiB | 0.5% | 按 TP |
| embed + lm_head | 1.23 GiB | 0.5% | 按 TP |
| 其它 | 0.22 GiB | 0.1% | — |
| **合计** | **272.15 GiB** | 100% | |

每 die 总 HBM **61.27 GiB**；固定开销 **4.56 GiB**（激活 3.15 + 非 torch 0.63 + NPU 图 0.78）【实测】。

### 2.3 P 侧逐形态结论

| P 形态 | EP | 专家/rank | 非专家/rank | 权重/rank | 与 61.27 GiB 比 | 判定 |
|---|---:|---:|---:|---:|---|---|
| **P=8（现状）** | 8 | 32.40 | 1.62 | **34.86（实测）** | 余 ~22 GiB | ✅ 还能放 14.65 GiB 池 |
| P=6（TP6） | 6 | 43.20 | 2.16 | 45.36 | 余 ~11 GiB | 内存能过，**但 TP=6 非法** |
| **P=4（TP4）** | 4 | **64.80** | 3.24 | 68.04 | 专家一项就 **超 3.5 GiB** | ❌ **不可行** |
| P=4 + 「只建/只载 layer 0..20」 | 4 | ~34.02 | ~1.70 | ~35.7 | 余 ~20 GiB | ⚠️ 理论可行，**但要改代码**（§5 #5/#7） |

**P 侧 KV 需求**：P 也要留一份自己的池（现状 `KV_CACHE_MEMORY_BYTES=15,728,022,528` = **14.65 GiB** / 29076 块）。池容量 = 29076×128 = **3.72M token**；一条 1M 请求占 8192 块。
⇒ `MAX_LEN` 只决定「几条满长请求能同时进来」，**不改变池大小**。

---

## 3. 交付件里其它可能的 8-die 假设（核查结果）

| 位置 | 是否有 8 硬编码 | 说明 |
|---|---|---|
| `deploy/a3-ced-pd/launch/_common.sh:16-17` | 默认值，非逻辑 | `PD_PREFILL_DEVS=0..7` / `PD_DECODE_DEVS=8..15`，env 可覆盖 |
| `scripts/serve_a3_pd.sh:26,33` | 同上 | 默认 devs |
| 连接器 `device_index = pp_rank*tp_size + tp_rank`（`:1880`）、`max_device_id = tp_size*dp_size*pp_size`（`:1805`） | ❌ 无 | 都按**本地 tp** 自洽 |
| `scripts/serve_a3_pd_proxy.sh` | ❌ 无 | 只有 host/port，与 TP 无关 |
| 32-bit guard（`_common.sh:36-41`、`serve_a3_ced_pd.sh:360-372`、连接器 `[CED-32BIT-GUARD]`） | ❌ 无 die 假设 | 依赖的是**每块页步长**（§4） |
| `a2/patches/kv8-offload-pool/*`（`[K_l1_8card]` 标签） | 属 **A2 DRAM 卸载线** | 不在 `deploy/a3-ced-pd/PAYLOAD.md` 的 CED-PD payload 清单内 ⇒ 与本题无关【未确认是否受 die 数影响】 |

---

## 4. D 侧 KV 容量数学 / 与 MAX_LEN 的关系 / guard 是否隐含 8-die P

### 4.1 池几何（【实测】+ 脚本注释）

| 量 | 值 | 出处 |
|---|---|---|
| D 侧每块字节 | **540928 B** = 4 个 slot 之和（3×131072 + 147712） | `_common.sh:40-41`（`CED_D_BYTES_PER_BLOCK`） |
| slot 3（ratio-1 槽）页步长 | **147712 B** = layer-20 C1 KV 131072 + INT8 index K 16384 + FP16 scales 256 | `_common.sh:36-39`；`docs/CED-PD-BLOCK-BOUND-20260925.md` |
| **4 GiB 寻址上界** | `⌊2³² / 147712⌋ = 29076 块`（判据按**页尾**，不是最大块号） | 同上 |
| 池字节 | `29076 × 540928 = 15,728,022,528 B`（14.65 GiB） | `_common.sh:41` |
| **token 容量** | `29076 × BLOCK(128) = 3,721,728 token` | 29076×128 |

### 4.2 与 MAX_LEN 的关系：**没有关系**

* 池由「字节预算 ÷ 每块字节」决定；`MAX_LEN` 只是**单请求窗口上限**。
* `1M → 512K → 256K`：`num_blocks` 不变、3.72M token 容量不变、4 GiB guard 不放松。
* 唯一变化是**并发窗口**：`MAX_SEQS=4` 时 4×1M = 4.19M > 3.72M（引擎自行限流）；4×512K = 2.10M < 3.72M（不再限流）。
* ⇒ 「把 MAX_LEN 降到 512K 来适配 12 die」**不成立**：12 die 卡在 TP 合法性与权重，不在窗口。

### 4.3 guard 是否隐含「8-die P」的假设：**否**

* guard 的输入只有**每块页步长**（KV 页几何）。本模型 `num_key_value_heads=1`（MLA latent），**KV 在 TP 维度整份复制**（0929 实测：TP4/DP2 与 TP2/DP4 的单 rank 容量**逐位相同** `1,096,072 tokens`）⇒ 页步长与 die 数无关。
* 它隐含的是 **CED 的 slot 布局**（layer-20 全局源由 P 侧写、经 slot 3 传给 D），**不是 die 数**。
* ★ 但 P、D **两侧都被同一上界钳位**（`scripts/serve_a3_ced_pd.sh:360-372`；注释写明 P 侧默认 29721 块同样越界）⇒ 换 die 数不会放松这条。

---

## 5. 「改成 4 die P 必须改的清单」（逐条给 文件:行 + 改法）

| # | 文件:行 | 现状 | 改法 | 难度 |
|---|---|---|---|---|
| **1** | `scripts/serve_a3_pd.sh:54-60` | `TP != 8 \|\| DP != 1 ⇒ FAIL` | 放开为 P/D 各自的 `P_TP/D_TP/P_DP/D_DP`（保留 fail-closed 的取值合法性校验） | 低（脚本） |
| **2** | `scripts/serve_a3_pd.sh:81` | JSON 里 `prefill.tp_size=8, decode.tp_size=8` 写死 | 用 `${P_TP:-8}` / `${D_TP:-8}` 拼（**保持紧凑、无空格**，否则会被展开成多余 CLI 参数） | 低（脚本） |
| **3** | `experimental/ced/mooncake_hybrid_connector.py:1782-1786` | `P_TP ≥ D_TP` 硬门 | 要去掉就必须**新写**「一个 P rank 服务多个 D rank」的映射 + 端口/握手复原（`:1743-1746`、`:395/:465` 的 `device_index` 语义） | **高（跨实例协议）** |
| **4** | 同文件 `:2426-2434` | `rand.sample(range(num_groups), decode_tp)` | P<D 时必然 `ValueError`；需重写抽样/映射（让 2 个 D rank 取同一 P rank 的同一份 MLA 副本） | **高** |
| **5** | `patches/files/model.py:1583-1588`（loader）+ 模型构造 | P 载全 40 层（实测 34.86 GiB@TP8） | P 需要「**只建/只载 layer 0..20**」（4 die 下这是**内存必需**：否则 68.04 > 61.27）。注意**只过滤 load 不够** —— 参数已实例化仍占 HBM，必须让 layer 21..39 不实例化（或 meta 初始化） | **高（模型层）** |
| **6** | `patches/files/model.py:1455-1457` | forward 在第 20 层 break | 不变 | — |
| **7** | 组契约硬编码 `(7,8,9,10,11)`：连接器 `:1457 / :1463 / :1572`，另 `tools/ced_prefill_probe.py:106` | P 在 transfer params 里发这段元组，D **逐字比对**，不一致 `raise` | 若走 #5（P 模型变成 21 层），SWA 分组（每 4 层一组）会变 ⇒ P 算出的「缺失组」不再是 `(7..11)` ⇒ 契约必须**重新定义或放宽** | **高（跨实例契约）** |
| 8 | `deploy/a3-ced-pd/launch/_common.sh:16-17`、`serve_a3_pd.sh:26,33` | 默认 devs | 只是 env 默认值；同时要过 `scripts/serve_a3.sh:141` 的「DEVS 数量 == TP×DP」校验 | 低 |
| 9 | `scripts/serve_a3_ced_single.sh:69` | `tp_size=1/1` 写死 | 单实例夹具，非交付路径 | 低 |

> ⚠️ **#3/#4 与 #5/#7 是两组独立的工作**：前者是「连接器支持 P_TP < D_TP」，后者是「P 内存放得下」。**两组都做完，12 die 才有理论可能**；只做一组都不够。

---

## 6. 吞吐预期：P 从 TP8 降到 TP4

| 口径 | TP8 · CED（P 跑 21/40 层，8 die） | 全 40 层 TP8（8 die 混布实例） | P=4（推断） |
|---|---:|---:|---:|
| prefill 32K | **12,536 tok/s** | 7,200 tok/s | ~**6,300 tok/s** |
| prefill 144K/128K | **13,676** | 8,341 | ~6,800 |
| prefill 1M | **9,897** | — | ~4,900 |

* 【实测】前三列来自 `docs/PERF-FOUR-AXES-STATUS-20261004.md` §1/§2（同机 a3-21）。
* 【推断】P=4：层数不变、**每 die 的专家与稠密计算都翻倍**（EP 8→4，TP 8→4）⇒ prefill 时间 ~×2。
* ⇒ **即使 12-die 形态做出来也不划算**：prefill 会从 12,536 掉到 ~6,300，**低于现行 8-die 全 40 层方案的 7,200**。
  **CED 的 2.07× 增益来自「P 用 8 die 只跑一半层」，不是来自「P 用更少的 die」。**
* decode 侧：D 仍 8 die ⇒ 与现状逐项相同【推断】（P 侧 TP 不影响 D 的 KV 布局）。
* 备查（**不可外推到 prefill**）：0929 的 decode 对照是 TP4/DP2 vs TP2/DP4，**两臂 EP 都是 8**，只动稠密部分 ⇒ 并发 1：41.9/39.1 tok/s；并发 8：221.7/232.8 tok/s。

---

## 7. 下一步最小验证实验

### 7.1 零成本负控（不需要 GPU、不需要起容器，1 分钟）

```bash
# ① 证明 rank 映射算法在 P_TP < D_TP 时必然崩（纯 Python，复刻 :2426-2434 的抽样）
python3 - <<'PY'
import random
prefill_tp, decode_tp, num_kv_head = 4, 8, 1
num_groups = max(1, (prefill_tp * 1) // num_kv_head)
print("num_groups =", num_groups)
try:
    print(random.Random(0).sample(range(num_groups), max(decode_tp // num_kv_head, 1)))
except ValueError as e:
    print("EXPECTED ValueError:", e)
PY
# ② 证明 TP=6 对 64 头模型不整除
python3 -c "print('64 % 6 =', 64 % 6)"
```

### 7.2 一次内存 smoke（要 4 张空闲 die；**不改代码**只能验到「放不下」）

```bash
# 目的：量 P=4/EP=4 时「Loading model weights took」是否直接 OOM（预期 OOM）
# 只起 P，不碰 D，也不改任何仓库文件（纯 env）
cd ~/cedpd-repo
MODEL=/home/l00886679/models/DeepSeek-V4.1-Flash \
DEVS="4 5 6 7" TP=4 DP=1 \
I_KNOW=1 \
bash deploy/a3-ced-pd/launch/serve_p.sh     # 预期在权重装载/或 serve_a3_pd.sh 的 TP 门被拒
```
> 注：当前 `serve_a3_pd.sh:54-60` 会在**进容器前**就拒绝 `TP=4` ⇒ 这条 smoke 现在**必然失败在脚本层**（这本身就是「卡在哪一步」的直接证据）。

### 7.3 真验证 12 die 的最小路径（必须先做完 §5 的 #1~#5、#7）

1. 在**tiny 夹具**上先过协议（`tools/launch_ced_tiny_d.sh`，几分钟一轮）：把 `P_TP=1 / D_TP=2` 跑通 ⇒ 证明 #3/#4 的映射正确，再谈 4/8。
2. tiny 过了再上 12 die：`PD_PREFILL_DEVS="2 3 4 5" PD_DECODE_DEVS="6 7 8 9 10 11 12 13" P_TP=4 D_TP=8` + 三条既有命令（`serve_p.sh` / `serve_d.sh` / `serve_proxy.sh`）。
3. 验收沿用 `deploy/a3-ced-pd/launch/smoke.sh`（144K 四针）+ 组契约硬门（`docs/CED-PD-ACCEPTANCE.md`）。

### 7.4 如果目标只是「现在就多挤出 2 个 die」

**没有配置级办法**：16 die 是最小可行集，dies 0/1 被别人占着就只能等（或改 §5 的代码走 12-die 路线）。

---

## 8. 一句话给决策

* **要 CED-PD 的 1.7–2.07× prefill 增益** ⇒ 必须 16 die（P8+D8），这是现状。
* **要 12 die** ⇒ 必须投入「连接器异 TP 映射」+「P 只建 21 层」两处高风险开发，且做完后 prefill 吞吐反而低于现行 8-die 方案 ⇒ **不建议**。
* **14 die** ⇒ TP=6 结构性非法，**此路不通**。

### 7.5 【实测】零成本负控的真实输出（2026-10-04 本机执行）

```
P=4 D=8: num_groups=4 ValueError: Sample larger than population or is negative
P=6 D=8: num_groups=6 ValueError: Sample larger than population or is negative
P=8 D=8: num_groups=8 sample OK -> [6, 7, 3, 0, 2, 5, 1, 4]
P=8 D=4: num_groups=8 sample OK -> [6, 7, 3, 0]
64 % 6 = 4 | 64 % 4 = 0 | 64 % 8 = 0
```

⇒ **P=4/D=8 与 P=6/D=8 即使绕过 `:1782` 的硬门，也会在 `:2426-2434` 的 rank 抽样处直接崩**；
反向的 P=8/D=4、P=8/D=6 在抽样层不崩（因为 `P_TP ≥ D_TP`），但它们各自死在前面的权重账 / TP 合法性上（§0 表）。
