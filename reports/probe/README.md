# 稀疏状态插针（事后取证工具）

长上下文出现"输出退化"时，本工具能把**模型内部的选择状态**（稀疏注意力的 top-k
与块级候选）落盘，用于事后定位，而不是靠盲目重放。

> 设计文档：`../sparse_state_capture_design.md`

---

## 为什么不是"dump 整个 KV cache"

| 对象 | 单 rank 大小 | 对定位退化的价值 |
|---|---|---|
| long_kv_cache（4 个 source 层） | 512K token 时 ≈ **4.7 GB** | 低 —— 同输入下 KV 是确定的 |
| `qr` + `positions` | ≈10 MB @1K tok | 高 —— 下游误差的入口 |
| **`selected`（topk_indices）** | `seq×512×int32` ≈ 2 MB @1K tok | **最高** —— "模型实际看到了哪些历史位置" |
| **`candidates`（块级候选）** | `seq×2048×int32` ≈ 8 MB @1K tok | **最高** |

选择结果只有几 MB，可逐 step 全量保留；KV cache 上百 GB 只能抽样，
且抽到"正确的那一份"解释不了任何事。

---

## 组成

| 文件 | 作用 |
|---|---|
| `sparse_capture.py` | 插针模块本体（本目录，随包提供） |
| `capture_bundle.sh` | 出错时一键打包（元数据 + 张量 + 日志尾部 + 指标） |
| `dsa_v41.probe.py` | **不随包提供** —— 在 `dsa_v41.py` 里插了 14 行调用的派生文件；见下 |

---

## 生成插针版 `dsa_v41.py`

在容器内取原始文件，然后在 `_select_sparse_indices()` 的
`selected, candidates = attn.indexer.select(...)` 调用**结束**之后插入：

```python
        # ==== [sparse-probe] ====
        try:
            from vllm_ascend.attention.sparse_capture import capture_selection
            capture_selection(
                self.role.layer_idx,
                selected,
                candidates if self.role.is_candidate_source else None,
                qr,
                positions,
                is_candidate_source=self.role.is_candidate_source,
            )
        except Exception:
            pass
        # ==== [/sparse-probe] ====
```

保存为 `reports/probe/dsa_v41.probe.py`，然后 `PROBE=1` 起服即可。

> **踩过的坑（务必保留守卫）**：`sparse_capture.py` 里有三道防护，缺一不可 ——
> 1. **图捕获守卫**：解码走 ACL graph 重放，捕获期间做任何 D2H 同步
>    （`.item()` / `.cpu()`）都会**把捕获挂死**（症状：服务卡在启动、EngineCore
>    反复报 "No available shared memory broadcast block found in 60 seconds"）。
>    守卫用 `torch.npu.is_current_stream_capturing()`。
> 2. **单次 D2H 同步**：统计量先在设备侧打包成一个张量，最后 `.tolist()` 一次取回。
>    早期版本每个统计量各 `.item()`，8 rank × 8 层 × 每层多次同步会打断流水。
> 3. **环形缓冲文件名不能带 step**：否则每次写入都是新文件，缓冲退化成无限增长
>    （曾写满 **339 GB / 35712 个文件**）。文件名只由 `(ring_idx, layer, rank)` 决定。

---

## 用量红线

| 项 | 预算 |
|---|---|
| L1 元数据 | ≈200 B/step/层 ⇒ 约 280 MB/天；超 5 GB 自动轮转 |
| L2 张量 | 环形缓冲，默认保留最近 4 个 step；单次快照上限 1 GB，超限自动降级为只留 `selected` |
| 落盘位置 | **绝不落 `/tmp`**（tmpfs 吃内存，历史上曾把宿主 `/tmp` 写满导致起服失败） |

## 快速用法

```bash
# 起服时带上 PROBE=1
PROBE=1 DEVS="..." MODEL=... bash scripts/serve_a3.sh

# 复现问题
# …

# 一键取证
bash reports/probe/capture_bundle.sh "长上下文退化-1"
# → bundles/<时间>-<标签>/SUMMARY.md 给出该看什么
```
