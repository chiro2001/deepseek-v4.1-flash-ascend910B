# V4.1 DCP8 算子级复现包（SMLA 非确定性）

**结论：`npu_sparse_flash_mla`（SMLA）在 `compress_ratio=1` 路径上是非确定的**
——同一份输入连调多次会给出不同输出。这是 DCP8 长上下文乱码的根因。

## 1. 生产证据（8-chip DCP8，a3-21）

`[V41-KDET]`：在同一层、同一次 forward 内，用**逐位相同的输入**连调两次 SMLA：

| 层 | `lse_bit_identical` | `max｜Δlse｜` | 该请求的输出 |
|---|---|---|---|
| layer 2（ratio=2） | **True** | 0 | — |
| **layer 20（ratio=1）** | **False** | **0.245** | **`Q7`（正确答案）** |
| layer 20（ratio=1，第 2 个请求） | False | **nan** | `#`（乱码） |

关键：**连输出正确的那个请求，SMLA 也是非确定的**（max|Δ|=0.245，只是幅度不足以
翻转答案）；第 2 个请求扰动幅度涨到 NaN ⇒ 乱码。所以这是**恒常存在的算子缺陷**，
不是"某个长度阈值"或"跨请求状态"。

## 2. 单卡最小复现（无 DCP、无 vLLM 服务）

```bash
docker run --rm --net=host --privileged --shm-size=8g \
  --device=/dev/davinciN --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v $(pwd)/probe_smla_determinism.py:/probe.py \
  -e ASCEND_RT_VISIBLE_DEVICES=<chip> -e PROBE_DEV=0 -e PROBE_T=904 -e PROBE_REPS=8 \
  -e PROBE_MODES=pages1,pages2,pages3,pages4 \
  <image> bash -lc "cd /tmp && python3 -u /probe.py"
```

实测结果：

| 模式 | cmp 页数 | 索引值域 | `lse_bit_identical` | `max｜Δlse｜` | `lse` NaN |
|---|---|---|---|---|---|
| pages1 | 1 | 128 | True | 0 | 0 |
| pages2 | 2 | 256 | True | 0 | 0 |
| **pages3** | 3 | 384 | **False** | 5.5e-3 | 0 |
| **pages4** | 4 | 512 | **False** | **nan** | **3471** |

⇒ **值域跨 ≥3 页（≥384 行）时必现**；越界不是触发条件（`shard_bad` 确定）。
块表列数（1024/8192）与 `sinks`（0/真值）**都不是**影响因素。

## 3. 已排除的假设（负结果）

| 假设 | 判据 |
|---|---|
| 索引越界 | `shard_bad`（值域 128、只声明 32 行）**确定** |
| 块表列数 | `btcols=1024` 与 `8192` 结果相同 |
| `sinks` 数值 | `sink=zero` 与 `sink=real` 结果相同 |
| 前序调用污染 workspace | `probe_smla_workspace.py`：warm 19 次后仍 bit-identical，且 A/B 逐位相同 |
| KV 写入未完成（流竞态） | 生产 `presmla_sync=1`（SMLA 前 `torch.npu.synchronize()`）**无效** |
| 索引顺序（非单调） | `sortidx=1`（升序重排）**无效** |
| QLI 索引选择 | `[V41-IDXDET]`：QLI 同输入连调两次 `差异元素=0` |
| dense 替代稀疏 | `dense_cmp=1`（ratio=1 层不传索引）**T=900 连首个请求也错**——交错分片下 query 会看到未来键 |

## 4. 建议

1. **算子侧**：按上面的最小复现定位 SMLA 的 `compress_ratio=1` + 稀疏索引路径
   （可疑点：多值域页的 gather/累加顺序、工作区复用）。
2. **规避（当前可用范围）**：DCP8 + 复制态下 `T ≤ 850` 的长针与短问答**已全部对齐 DCP1**
   （112/112 逐字相同）；`T ≥ 880` 受本缺陷影响。
3. **未来可选路径**：若把 DCP 分片改为**连续**（非交错），则 dense cmp 可行，
   可同时绕开非确定并减少索引读写——但平台把 `cp_kv_cache_interleave_size`
   固定为 32，需要先解决该约束。

---

## 5. ★ 生产输入重放包（100% 同源，单卡可复现）

**下载**（88.89 MB，public-read）：
```
cos://uploads-new/share/dsv41-dcp8-smla-nondeterminism-repro-20260930.tar.zst
md5(本地打包) = 12c159038c8917d047c61cf19e1e5ec2
```

内容：`l20_T904_rank0.pt` / `l20_T904_rank4.pt`（**从生产 8-chip DCP8 实例原样落盘**的
SMLA 输入，含 q、`cmp_sparse_indices`、两类块表、`seqused_*`、`sinks`、
`metadata(1024,)`、标量参数，以及本请求实际读到的 ori/cmp 页；块表已重映射到
1..N，数据与生产逐位相同）+ `replay_dump.py`。

**复现**（单卡，约 20 秒）：
```bash
docker run --rm --net=host --privileged --shm-size=8g \
  --device=/dev/davinciN --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v $(pwd):/d -e ASCEND_RT_VISIBLE_DEVICES=<chip> \
  quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3 \
  bash -lc "cd /d && python3 -u replay_dump.py /d/l20_T904_rank0.pt 12"
```

**实测结果**（a3-21 chip4）：

| 用例 | reps | ALL bit-identical | STEADY(2..N) bit-identical | `max｜Δlse｜` | `lse` NaN 个数/次 |
|---|---|---|---|---|---|
| rank0（ori owner，cseq=128） | 12 | **False** | **False** | **nan** | `[4754, 5255, 5255, …, 5255]` |
| rank4（非 owner，cseq=104） | 8 | **False** | — | **nan** | `[3538, 3653, …, 3653]` |

⇒ **持续非确定**（不只是首次调用），且 NaN 个数在不同次调用之间变化。

**生产侧如何重新产生 dump**：文件开关 `dumpdir=<容器内目录>`，见
`dsa_v41.py` 的 `[V41-DUMPREPLAY]`。

---

## 6. 触发条件刻画与"规避方案"尝试（2026-09-30 18:40–19:10，全部实测）

### 6.1 已刻画的触发条件（单卡合成输入）

值域 128、`cseq=128`、块表"每列指向不同页"时，按**每行有效项数**扫描：

| 每行有效项数 | 8 | 57 | 128 | 192 | **256** | 288 | 320 | 352 | 384 | 400 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 确定 | ✗ | ✗ | ✗ | ✗ | **✓** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

另两条独立触发条件（各自单变量对照）：

* **索引值域跨 ≥3 页（≥384 行）**：即使 512 槽全有效也不确定（4 页时出 3471 个 NaN）。
* **块表填充方式**：`varshard` + 所有列指向同一页 → 非确定（47084 个 NaN）；
  每列指向不同页 / 只有第 0 列非零 → 确定。

### 6.2 算子不接受更小的 topk（算子自身报错）

```
E  Invalid_Argument(EZ0026): Parameter cmp_topk of aclnnSparseFlashMlaMetadata has
   incorrect value 128/256/64. Reason: When has_cmp_kv is true, the value of
   cmp_topk must be in [0, 512, 1024].
```

⇒ 无法通过"把 topk 降到实际需要的 128"来消除 `-1` padding。

### 6.3 规避尝试：均匀重复填索引槽（**结论：无效**）

思路：每行的**所有**键重复相同次数 `k` ⇒ softmax 分子分母同乘 `k` ⇒ 输出**数学等价**，
同时把 `-1` 槽位挤掉（`ceil` 模式可做到 0 个 `-1`）。

真实 dump 上的单卡结果：

| 变异 | `lse_bit_identical` | NaN |
|---|---|---|
| 原样 | False | 4834 |
| ori 只用到的列 / ori 8 页不同数据 / cmp 全列同页 | False | 2265~4739 |
| **idx_full（每行重复填满 512）** | **True** | **0** |
| **idx_uni_1024（topk=1024，均匀 k=floor(1024/n)）** | **True** | **0** |
| idx_padonly（只加 `-1` padding） | False | `max｜Δ｜=3.3e35` |

但在**生产路径**上接入同样变换（`uniqpad`，向量化实现）后：

| 模式 | T=904 `max｜Δlse｜` | 长上下文 2000/8000 |
|---|---|---|
| 原样 | 0.245 | 失败 |
| `floor`（重复 floor(K/n) 次，剩 `-1`） | 0.0116 | 失败 |
| `ceil`（重复到填满 K 槽，0 个 `-1`） | 0.0146 | 失败 |

⇒ 非确定性**显著减小但未消除**，长上下文仍不通过。
单卡 replay 上"填满即确定"在生产路径上**不可复现** ⇒ 触发条件不止"索引里有 `-1`"一项。

**因此 `uniqpad` 默认关闭**（`uniqpad=0`），代码与开关保留供算子修复后复核。

### 6.4 建议给算子团队的三条线索

1. **`-1`（未使用的索引槽）数量**与故障强相关：有效项 <256 必现；填满后单卡转确定。
2. **索引值域跨页数** ≥3 页必现（与 `-1` 无关）。
3. **块表填充方式**（同页 vs 异页）也能翻转结论 ⇒ 说明算子内部对**块表/索引的越界访问**
   会读到未初始化内存；`NaN 个数在两次调用间变化`是典型特征。
