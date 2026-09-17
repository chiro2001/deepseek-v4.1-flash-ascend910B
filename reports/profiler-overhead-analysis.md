# Profiler 采集开销：实测量化与分析方法论修正

> 触发：用户提出「当前获取 profiling 数据如果靠 profiler 会有数据采集开销」
> 结论：**开销真实存在且已量化（+3.7~+6.6 ms/step，+10~19%）**；此前由 profiled run
> 导出的**绝对**每步设备账存在除数错误，必须按 §4 重做。

---

## 1. 先厘清一个易混点：**配置了 profiler 不等于正在采集**

`serve_a21.sh` 的 `PROFILE=1` 只是给 `vllm serve` 加 `--profiler-config`，
**它只是把 `/start_profile`、`/stop_profile` 两个端点装上**。真正采集发生在
`POST /start_profile` 到 `POST /stop_profile` 之间。

⇒ 判断某次测量有没有被采集污染，**要看当时有没有跑 `prof_collect_a21.sh`，
不能看起服脚本的 `PROFILE=` 值**。用 `PROFILE=1` 起服但不调 `/start_profile` 的测量是干净的。

日志里另有一处佐证：`Rank 0: Torch profiler disabled for CUDA graph capture`
—— vLLM 在**图捕获期主动关闭** profiler，所以捕获阶段不受影响，受影响的是稳态 replay。

---

## 2. 硬证据一：profiler 有可复现的**显存足迹**

同一模型、同一配置，只看 `GPU KV cache size`：

| 会话 | profiler | KV tokens | 相对差 |
|---|---|---|---|
| `a21_mig_2244`（22:44） | 关 | **3,557,227** | 基准 |
| `a21_prof_2302`（23:01） | 开 | **3,557,104** | 少 123 |
| `a21_nohot_0213`（02:13） | 关 | **3,388,563** | 基准 |
| `a21_profA_0225`（02:25） | 开（未采集） | 3,388,441 | 少 122 |
| `a21_fmc2_0019`（00:19） | 开（曾采集） | 3,388,441 | 少 122 |

两组独立对照都重复出现同样的 122–123 token 差额
⇒ **profiler 稳定占用约 122–123 个 token 的显存**（约 0.03% KV）。
量级很小，对「KV > 3M」判据无实质影响（3.388M 远大于 3M），但说明它确实在设备上分配了 buffer。

---

## 3. 硬证据二：profiler 有可复现的**墙钟开销**

两次都是**同一进程内、间隔数分钟、配置完全相同**，只切换 profiler 采集开关：

| 时间 | 会话 | profiler | ms/step | cli_p50 | 相对差 |
|---|---|---|---|---|---|
| 00:34 | `a21_fmc2_0019` | **开** | **40.97** | 39.16 | 基准 |
| 00:42 | 同上 | 关 | **34.35** | 34.72 | **快 6.62 ms（19%）** |
| 02:32 | `a21_profA_0225` | 关 | **38.94** | 38.32 | 基准 |
| 02:37 | 同上 | **开** | **42.66** | 40.31 | **慢 3.72 ms（9.6%）** |

两个独立的同会话对照给出一致量级：**profiler 采集期间每步慢 3.7–6.6 ms**。
（00:34 那次同时跑着 32K 完整负载，开销更大；02:37 那次负载更短。）

### 3.1 直接后果

**任何在 profiler 采集期间测得的绝对 ms/step 都不可用于对外结论**，
包括此前报告的「busy 29.17 + free 5.25 = 34.42 ms/step」这类数字。

---

## 4. 此前分析里的**方法错误**（必须修正）

### 4.1 错在哪

我做过这样一步：在 profiled run 的窗口 7000–9960 ms（2960 ms）里，**假设**有 86 个 decode step，
于是得出 busy = 29.17、free = 5.25 ms/step，并称「两者相加 34.42 与实测 34.35 吻合，自洽」。

两个问题：

1. **除数 86 是外部假设**，不是从这份 profile 里量出来的；
2. 用来「吻合验证」的 34.35 是**另一份 unprofiled 测量**。
   用不同 profiler 状态下的墙钟去校验一个 profiled 窗口的每步值，**是循环论证** ——
   窗口总长本身就等于该 profiled run 的墙钟，按 86 去分只是把「另一个 run 的步数」硬塞进来。
   两者能吻合纯属巧合。

### 4.2 正确做法：步数必须**从设备时间线自己量**

新增工具 `scripts/step_period.py`：主模型每步对每个 MoE 层各发一次
`DispatchFFNCombineW4A8`（本项目 40 层），取第 i 与第 **i+40** 个事件之差的**中位数**，
即该次运行的一步周期。这个量与任何外部假设无关。

用它重算两份 profile（**两份都是采集态**，故 profiler 开销是同模态的）：

| profile | 设备量出的步长 | busy/step | free/step | comm/step | compute/step |
|---|---|---|---|---|---|
| 慢会话 02:37 | **37.77 ms** | **29.82** | **7.95** | 3.12 | 28.19 |
| 快会话 00:34 | **37.07 ms** | **29.32** | **7.75** | 3.37 | 27.51 |

### 4.3 修正后的三个结论

1. **free 占比约 21%**，不是此前说的 25.8%。此前取的窗口（7000–9960 ms）与现在用的全 span 不同，
   导致分母口径不一致；以全 span 计，free 稳定在 21% 左右。
2. **两个时代的设备侧几乎没变**：busy 29.32 vs 29.82、free 7.75 vs 7.95、comm 3.37 vs 3.12
   —— 差异都在 0.5 ms 以内。这是 §5 结论的基础。
3. **零重叠的结论依然成立且更稳**：`comm ∩ compute ≈ 0` 是同一次采集内的**交集**度量，
   不受 profiler 均匀放大的影响（0.002/3.54 = 0.06% 量级，不可能被 10–19% 的均匀膨胀解释）。

---

## 5. 由此重新定位「34.4 到 38.9」这个 14% 退化

排除 profiler 之后，事实链变得清晰：

| 观测 | 数据 | 含义 |
|---|---|---|
| TTFT / prefill | 32K：快 5.45 s / 6016 tok/s；慢 5.53 s / 5930 tok/s | **prefill 未退化** |
| decode ms/step | 34.50 → 39.30 | **只有 decode 退化** |
| 设备 busy/free（两份 profile） | 29.32/7.75 vs 29.82/7.95 | **设备侧未变** |
| 每步 token 数 | 2.865 vs 2.813（A 2.865 vs 2.802） | 内容未变 |

⇒ 设备侧与 prefill 都没变，**退化发生在 decode 的 host 侧**（chunk 与 chunk 之间的等待）。
这与此前用延迟臂测出的「**D2H 之后的 host 工作 116% 暴露**」是同一个薄弱环节：
host 侧任何变慢都会 1:1 打到 ms/step，而 prefill 走另一条路径，所以不受影响。

**尚未定位到具体原因。** 已排除：热重载闸门（`HOTSPIKE=0` 同样慢）、`PROFILE` 配置、
外部 CPU 争抢（cpus 320-639 与 0-319 实测忙碌 0.2% / 0.1%，load 2.6，
温度 49–52 °C 无限流），以及窗口 01:15–02:05 内没有新增大进程（只有 0 RSS 的内核线程）。
这是当前**首要待解问题**。

---

## 6. 后续一律遵守的测量纪律

1. **绝对性能只认 unprofiled 客户端墙钟**（`ms/step` 与 `cli_p50` 两路互校；实测两者一致 ±1 ms）。
2. **profiler 只用于结构与比例**：算子排序、comm/compute 占比、overlap、相位分解。
3. **需要设备侧每步绝对值时**，用 `step_period.py` 从设备时间线量步长，**不引入外部步数假设**。
4. **需要「生产态」的 busy/free** 时，同一会话内做 profiler 开/关两轮，用差值把开销分离出来。
5. **A/B 对照必须保持 profiler 状态一致**；一轮开一轮关的对比全部作废。
6. 容量类结论一律取 unprofiled 值（带 profiler 的 session KV 少约 122 个 token）。

---

## 7. 证据路径

| 内容 | 路径 |
|---|---|
| 同会话 profiler 开/关 | `p42_t4_quote_32768_fmc2_prof_32k.jsonl`（开 40.97）、`p42_t4_quote_32768_fmc2b_32768.jsonl`（关 34.35）、`p42_t4_quote_32768_profA_32768.jsonl`（关 38.94）、`p42_t4_quote_32768_slowchk.jsonl`（开 42.66），均在 `logs/perf/a21/` |
| KV 足迹 | `logs/perf/{a21_mig_2244,a21_prof_2302,a21_nohot_0213,a21_profA_0225,a21_fmc2_0019}_serve.log` 的 `GPU KV cache size` |
| 步长测量工具 | `scripts/step_period.py` |
| 跨会话对照 | `scripts/cmp_sessions.py`、`scripts/cmp_ttft.py` |
| 设备账（重算输入） | `/tmp/op_summary_slow.csv`（02:37）、`/tmp/op_summary_fmc2.csv`（00:34） |
| 图捕获期禁用 profiler | 各 serve 日志的 `Torch profiler disabled for CUDA graph capture` |
