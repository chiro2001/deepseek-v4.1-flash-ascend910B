# ③ prefill 的第一份画像：8 die 已饱和，瓶颈在 AIV（向量单元）与 attention 内核（2026-10-05）

> 背景：四个维度里 **③ prefill 此前完全没有设备画像** —— 所有分析都基于 decode profile。
> 本文用交付实例（TP8、devidx=1）采**冷 prefill** profile 并给出结论。
> 数据：`results/armF_r6_restore/prof/..._20261004212552338_ascend_pt`（2 × 32K 冷 prompt）。
> 标注：【实测】/【推断】。

## 0. 一句话

1. **冷 prefill 实测 7,204 tok/s**（2 × 32,772 tok，TTFT 9.04/9.09 s）—— 与目标里写的
   "TP8 现 7,200" 一致，说明这个数字是**冷前缀**口径。
2. **吞吐对并发不敏感**（N=1/2/4/8 = 6,869 / 7,204 / 7,456 / 6,789 tok/s）⇒ **8 die 已经饱和**。
3. **每窗口 99.7% 忙**（全任务并集 8996 ms / 9022 ms）⇒ 没有空闲可填。
4. 关键路径是**主计算流**；其构成：`SparseFlashMla` 26% + `QuantLightningIndexerV2` 16%
   + `HcPre` 14% + `HcPost` 8% ⇒ **attention 系 42% + mHC 22%**。
5. **绑定资源是 AIV（向量单元）**：主流任务时长里 AIV 占 **88%**，而 `aic_mac` 只占 **18%**
   ⇒ prefill **不是矩阵算力受限**，是向量/标量指令受限。
6. **`allreduceAicpuKernel` 完全被隐藏**（7118 ms 忙碌但独占贡献仅 9.6 ms）—— 这是好设计，
   不要去"优化"它。

## 1. 测量方法（两处必须先说清的口径）

### 1.1 必须是**冷前缀**
`bench_concurrency` 从语料**开头**取切片 ⇒ 前几次实验已经把那段灌进 prefix cache。
第一次采集时 TTFT 只有 **0.46 s**（32K），那不是 prefill 的真实速度。
本报告改用 `tools/capture_prefill_cold.py`：从语料**远端**取切片 + 每次一个**唯一 nonce**，
保证任何缓存都命中不了。冷/热对比：

| 口径 | 2 × 32K 的 TTFT | 折算 |
|---|---|---|
| 热前缀（旧脚本） | 0.46 s | ~140,000 tok/s（**假象**） |
| **冷前缀（本报告）** | **9.04 / 9.09 s** | **7,204 tok/s** |

### 1.2 `stream` 编号会变
decode profile 里主图是 stream 146/109，而 **prefill 的主计算流是 stream 47**
（21,688 个任务 / 7356 ms）。做分析时必须先按"gmm1 最多的流"自动识别，不能沿用 decode 的编号。

## 2. 并发扫描（每次都是冷前缀，同语料不同远端切片）

| 并发 | prompt 总 token | 墙钟 | **吞吐** | 每 die |
|---:|---:|---:|---:|---:|
| 1 | 32,772 | 4.77 s | 6,869 | 859 tok/s/die |
| 2 | 65,543 | 9.10 s | 7,204 | 901 |
| **4** | 131,087 | 17.58 s | **7,456** | **932** |
| 8 | 262,175 | 38.62 s | 6,789 | 849 |

**读法**：4 并发处出现峰值后回落（N=8 比 N=4 低 9%）。这**不是**调度没跟上，
而是 8 die 已饱和 —— 见 §3 的忙碌度。**参照**：目标里的 CED 参考值 13,676 tok/s 是 **16 die**，
折合 855 tok/s/die ⇒ **当前 TP8 的每 die 效率（932）比 CED 高 9%**。

## 3. 时间账：没有空闲

| 量 | 值 |
|---|---|
| 窗口 | 9,022 ms |
| 全部任务并集 | **8,996 ms（99.7%）** |
| `stream 10`（allreduceAICPU，738 个任务） | busy 7,118 ms ⇒ **独占贡献 9.6 ms** |
| `stream 47`（主计算） | busy 7,356 ms ⇒ 独占 **1,345 ms** |
| 其余流（31/37/155/35/38） | 合计 busy ~0.77 s，独占 ≤57 ms |

⇒ ① **没有可回收的空闲**；② 那 7.1 s 的 allreduce 与主计算**几乎完全重叠**。

## 4. 主计算流（stream 47）的构成与资源账

任务时长合计 **7356 ms**；单元占用：

| 单元 | ms | 占任务时长 |
|---|---:|---:|
| **AIV 总** | **6474** | **88.0%** |
| ├ AIV **scalar** | **3082** | **41.9%** |
| ├ AIV **vec** | 2572 | 35.0% |
| ├ AIV MTE2 | 2392 | 32.5% |
| └ AIV MTE3 | 928 | 12.6% |
| AIC MTE2 | 2315 | 31.5% |
| AIC MTE1 | 2032 | 27.6% |
| AIC scalar | 1669 | 22.7% |
| AIC fixpipe | 1517 | 20.6% |
| **AIC MAC（真正的乘加）** | **1305** | **17.7%** |

按算子族（每步/每 chunk 的量级见 §5）：

| 算子 | 次数 | 合计 ms | AIV | 其中 scalar | vec | AIC MAC | 形状（首例） |
|---|---:|---:|---:|---:|---:|---:|---|
| **SparseFlashMla** | 360 | **1883** | 1825 | **1343** | 74 | 276 | `8064,8,512;25892,128,1,512;…` |
| **QuantLightningIndexerV2** | 72 | **1146** | 1146 | 755 | **946** | 141 | `8064,32,128;25892,64,1,128;…` |
| **HcPre** | 774 | 1006 | 1001 | 402 | 588 | 144 | `8064,4,5120;24,20480;3;24;8064,4` |
| **HcPost** | 774 | 576 | 570 | 165 | 467 | 0 | |
| gmm1 | 387 | 347 | 346 | 26 | 53 | 156 | `48384,5120;…` |
| MatMulV3 | 683 | 301 | 0 | — | — | 257 | `8064,4096;4096,1024` |
| MatMulV2 | 763 | 274 | 0 | — | — | 154 | `8064,5120;384,5120` |
| gmm2 | 387 | 208 | 196 | 29 | 30 | — | |

**两条最重要的读数**：
* **`SparseFlashMla` 的 AIV 里 1343/1825 = 74% 是 scalar**，真正的向量算术只有 74 ms。
  即这个内核**几乎全部时间在标量寻址/循环控制**（或等 AIC 的自旋）。
* **`QLI` 是真向量受限**（vec 946 ms），它是 O(n_q × n_kv) 的索引打分。

## 5. chunk 结构（每 chunk ≈ 8064 token）

`SparseFlashMla` 360 次 = 40 层 × **9 个 chunk**；`QLI` 72 次 = 8 层 × 9 chunk。
⇒ 每个 chunk 约 **817 ms**（9 个 chunk 摊满 7.36 s），其中：
`SparseFlashMla` 209 ms、`QLI` 127 ms、`HcPre` 112 ms、`HcPost` 64 ms、
`gmm1` 39 ms、`MatMulV3` 33 ms、`MatMulV2` 30 ms、`gmm2` 23 ms（其余 ~180 ms 是大量小算子）。

## 6. 结论：③ 的可动空间（按可信度排序）

| # | 方向 | 量级 | 可行性 |
|---|---|---:|---|
| 1 | **`SparseFlashMla` 的 AIV 标量化**（1343 ms = 窗口 15%） | 上限 ~15% | **高难度**：vendor 内核（`csrc/attention/sparse_flash_mla/`），要把每 token 的标量寻址向量化；需自研 kernel + 逐位比对 |
| 2 | `QLI` 的向量效率（vec 946 ms） | 上限 ~10% | 高难度（同为 vendor 内核） |
| 3 | mHC（HcPre+HcPost 1582 ms = 22%） | 上限 ~10% | 中：HcPre 已是自研可改；但 trackB 已证 K 分核调整**反而慢 35–45%**，A1 在 M=8064 时**无收益**（档位选择退化为 32） |
| 4 | 加大 prefill chunk（`BAT_TOKENS` 8192→16384） | **不确定（3–6%？）** | 需实测；代价是 decode 侧 engram padding 缓冲翻倍（`ZerosLike 8192×6144` → 两倍，约 +0.06 ms/step）与 ④ 容量小幅下降 |
| 5 | 更多 die（CED 16 die） | 2× | 已证 12/14 die 结构性不可行 |

**没有配置级旋钮**：`bench_concurrency` 的 4 个并发档把 8 die 打到 99.7% 忙，
说明吞吐上限由**每 token 的指令量**决定，而不是调度或 chunk 大小。

## 7. 复现

```bash
# 1) 采冷 prefill（每次唯一 nonce ⇒ 必然冷前缀）
python3 tools/capture_prefill_cold.py --port 19210 --n 2 --tokens 32768 --offset-frac 0.62
# 2) 导出（原始数据 → CSV）
bash ~/tmp/export_run.sh armF_r6_restore        # 见 docs/DELIVERY-PROFILE-R6 §5
# 3) 归一化 + 分析
python3 tools/normalize_kernel_details.py <...>/ASCEND_PROFILER_OUTPUT/kernel_details.csv /tmp/pf
python3 tools/prefill_report.py /tmp/pf --min-gap-ms 5
```
