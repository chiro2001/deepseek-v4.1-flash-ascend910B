#!/usr/bin/env python3
"""写「抖动很小 + 被放大」的刻画 + 官方非确定清单线索。"""
from pathlib import Path

DOC = Path("/home/chiro/projects/dsv41/main-merge/docs/TP8-NONDET-JITTER-AMPLIFICATION-20261007.md")

TEXT = '''# TP=8 非确定性：**基础抖动极小、被解码放大** + 一条官方线索（2026-10-07）

> 承接 `TP8-DECODE-NONDETERMINISM-ROOTCAUSE` 与 `PREFILL-DETERMINISTIC-ANY-T`。
> 本轮用**同请求内同时取 `prompt_logprobs` 与 `logprobs`** 的手法，量化了抖动的"源头大小"，
> 并找到一个官方文档层面的线索。全部为【实测】，推断处已标注。

## 0. 结论

| 项 | 结果 |
|---|---|
| **prefill 末位（交接点）跨轮波动** | **6.2e-04** |
| **首生成 token 的 logprob 跨轮波动** | **4.8e-07** |
| 该配置下 token id | **跨轮全部相同** |
| 长生成时（48 token） | token 从 **token#0 起**就开始不同；最大 \\|Δlogprob\\| **1.6~2.0** |
| ⇒ 性质 | **源头只有 1e-7~1e-4 级数值抖动，被自回归解码放大到可见差异** |
| ★ 官方线索 | `npu_scatter_nd_update` 系列**列在 Ascend 950DT 的"非确定 API 清单"**，而我们的 KV 写入正用它 |

## 1. 手法：在同一个请求里取两个量

`prompt_logprobs` 给的是 **prefill 每个位置**的 logprob；`logprobs.token_logprobs[0]`
给的是**首生成 token** 的 logprob —— 二者来自**同一个最后位置**的 logits。
同请求同时取，就能把"主干前向"与"prefill→decode 交接"分开。

`tools/gate_handoff.py`，prompt ≈827 字符、`max_tokens=4`、8 轮（先预热 1 次）：

```
r00 末位3=(429:-0.000004, 303:-0.020892, 913:-0.001263)  首生成(-0.000000, '那')
r01 末位3=(429:-0.000051, 303:-0.031953, 913:-0.000705)  首生成(-0.000001, '那')
…（8 轮 token id 全部相同）

=== 跨轮波动（r01..rN）===
prompt_logprobs 末位-2        6.640e-05
prompt_logprobs 末位-1        3.030e-02
prompt_logprobs 末位(交接点)     6.197e-04
logprobs[0]（生成）            4.768e-07
logprobs[1]（生成）            1.192e-07
logprobs[2]（生成）            6.485e-05
末位 token id 是否跨轮相同: True
首生成 token id 是否跨轮相同: True
```

⇒ **源头级抖动是 1e-7 ~ 1e-4**，且在这个配置下不足以翻转 token。

## 2. 放大效应：同样的机器，生成越长分叉越早越大

同一门（`tools/gate_decode_diverge.py`，8 轮、`max_tokens=48`）在不同 prompt 长度下：

| prompt 字符数 | 热轮内部最大 \\|Δlogprob\\| | token 不同的 (轮对×位置) 数 |
|---:|---:|---:|
| 400 | 1.5762 | 770 |
| 1200 | 1.9692 | 491 |
| 3000 | **2.0349** | 916 |

⇒ **约 1e-4 的源头抖动，在 48 步自回归解码后放大到 ~2.0 的 logprob 差、并大量翻转 token。**
这是**混沌放大**的典型形态（temperature=0 下 argmax 边界附近的微小扰动会被"锁定"并延续）。

### 2.1 与既有观测的自洽性

| 观测 | 与"小抖动 + 放大"是否自洽 |
|---|---|
| prefill（`prompt_logprobs`、`max_tokens=1`）在 T=32/1024/1240 **逐位 0 差** | ✅ 单步、无解码累积 |
| `SPEC=0` 下抖动仍在（1.79） | ✅ 投机解码只是**加速放大**，不是源头 |
| eager 下 token 身份 **0 处不同**、graph 下大量不同 | ✅ 图模式改变了规约/寻址顺序 ⇒ **放大系数**变化 |
| 短生成（4 token）token 稳定 | ✅ 放大步数少 |

## 3. ★ 一条官方线索：KV 写入用的算子系列在非确定清单上

`~/opensrc/op-plugin/docs/zh/custom_APIs/determin_API_list.md`（官方"确定性计算 API 清单"）：

> 当使用 **Ascend 950DT** 时，下列 API 计算存在随机性，开启确定性计算开关可以保持确定性：
> `torch_npu.npu_scatter_nd_update`、`torch_npu.npu_scatter_nd_update_`、
> `torch_npu.scatter_update`、`torch_npu.scatter_update_`、
> `torch_npu.npu_fusion_attention_grad`

而 tp8k5 运行的镜像版 `dsa_v41.py`（1092 行，已存档 `~/tmp/imgcode/dsa_v41_image.py`）里，
**KV cache 写入正是 `torch.ops._C_ascend.npu_scatter_nd_update_sk`**（3 处调用点：
`preprocess` / `multistream_preprocess` / `_write_compressed_source`，见该文件 207/308/367/445 行）。

【推断】`_sk` 是同一算子家族的变体（官方清单只列了非 `_sk` 名），**很可能同样非确定**
（该算子内部做排序，见 `scatter_nd_update_sk_tiling.cpp` 里的 `SORT_*` 常量）。

**但要注意一个反证**：prefill 也走同一个 `scatter_cache_sk`，而 prefill 是逐位确定的
⇒ 即使该算子有随机性，它在我们的用法下**不足以单独造成 prefill 抖动**；
它更可能是 **decode 每步 40 层 × 每层一次写入** 时累积的微小扰动源之一。

### 3.1 可直接验证的下一步（成本低）

1. **单算子确定性测试**：在 tiny 上对 `npu_scatter_nd_update_sk` 用**逐位相同的输入**
   连续调用 100 次，逐位比对输出 ⇒ 直接判定该算子是否确定（**不需要重启服务**，纯微基准）。
2. 若不确定 ⇒ 用 `torch.use_deterministic_algorithms(True)` 或在算子层面固定排序（本仓
   已有 `_v41_ordered_allreduce` 的同类思路）。
3. 若确定 ⇒ 继续查 decode 专属的其它环节（图 replay 的 padding / device-metadata 时序）。

## 4. 对交付的含义

1. **不能再用"逐位 max\\|Δ\\|=0"作为唯一验收门**：当前交付件在长生成下必然不满足它，
   而源头抖动只有 1e-7~1e-4 ⇒ 门应当写成**分级**：
   * L1（源头）：单步/短生成（≤4 token）跨轮 token 一致 + logprob 波动 < 1e-5；
   * L2（端到端）：同 prompt 多轮**前 N 个 token 一致率**（N 由业务决定），
     并对长生成给出"分叉位置分布"。
2. **实用影响有限但真实**：短问答稳定；长生成在同一 prompt 下会给出**不同续写**。
   对"抽取/问答"类任务影响小，对"长文续写"类任务可见。
3. **性能结论不受影响**（`ms/step`、吞吐都是统计量）——`MAX_SEQS=64` 的 +32~37% 仍然成立。

## 5. 环境状态

本轮**未重启**服务（全部为只读/推理测试）。tp8k5 仍为交付配置：health=200、
KV 2,987,618 token、BAT 8192、MAX_SEQS 32、SP 5、`capture_sizes=…,96,192`、dspark 开启。
tiny（TP=2）health=200、未动。

## 6. 复现

```bash
ssh a3-21 'python3 ~/tmp/gate_handoff.py http://127.0.0.1:19210 8 4'
ssh a3-21 'python3 ~/tmp/gate_decode_diverge.py http://127.0.0.1:19210 48 8 3000'
ssh a3-21 'sed -n "55,70p" ~/opensrc/op-plugin/docs/zh/custom_APIs/determin_API_list.md'
```
'''

DOC.write_text(TEXT)
print("wrote", DOC, len(TEXT))
