# 前缀缓存：基线 PD 上可用且正确（2026-09-26 00:00–00:40）

## 一句话

非 CED 基线、`PREFIX=1`、P/D 双 TP8、`num_blocks=29076`：
**缓存命中真实发生**（服务端 `prefix_cache_hits_total` 增量 144,000、P 与 D 两侧同步），
**命中路径的答案与冷路径逐字节相同**，端到端约 **5.7×** 加速。

## 必须记住的两个坑

1. **响应里的 `usage.prompt_tokens_details.cached_tokens` 恒为 0，即使命中也一样**
   —— PD 代理不传递该字段。判据必须用服务端计数器：
   `vllm:prefix_cache_hits_total`、`vllm:prompt_tokens_by_source_total{source="local_cache_hit"}`。
2. 冷 prefill 的 wall 在 **5.6 s 与 22.9 s** 之间波动（同机、同长度、同口径），
   **尚未归因**（怀疑与进程 warmup 有关，未证实）。

## 探针一：`pfx_probe.py`（无针，只看计数器与时延）

| 步骤 | wall | P hits 增量 | D hits 增量 | `cached_tokens` |
|---|---:|---:|---:|---:|
| A 冷 prefill（新 prompt） | 26.64 s | 0 | 0 | 0 |
| B 同一 prompt 第 2 次 | 2.86 s | **144,000** | **144,000** | 0 |
| C 同一 prompt 第 3 次 | 1.19 s | **144,000** | **144,000** | 0 |
| D 换一个同长度的新 prompt | 22.89 s | 0 | 0 | 0 |

`A/B/C` 三次的 prompt 完全相同；`D` 是不同语料位置的同长度 prompt。
⇒ 命中与否可复现，且与"是不是同一个 prompt"一致。

## 探针二：`pfx_probe2.py`（带针，判正确性）

针 `【运维备忘 D】构建机临时令牌是 RB9N-6014` 插在 prompt 的 80% 深度，
问题与期望答案见脚本；`max_tokens=16`、`temperature=0`。

| prompt | 步骤 | wall | D hits 增量 | 答案正确 | 与上一步 |
|---|---|---:|---:|---|---|
| P1（语料 offset 0） | cold | 5.60 s | 0 | ✅ `RB9N-6014` | — |
| P1 | hit1 | 0.98 s | **144,000** | ✅ | **相同** |
| P1 | hit2 | 1.00 s | **144,000** | ✅ | **相同** |
| P2（语料 offset len/2） | cold | 5.78 s | 0 | ✅ | — |
| P2 | hit1 | 1.02 s | **144,000** | ✅ | **相同** |
| P2 | hit2 | 1.01 s | **144,000** | ✅ | **相同** |

⇒ 6 次请求全部答对，且**冷/热两路逐字节一致**。

## 与 CED 的关系

* CED 口径**仍然被启动硬门挡住**（`scripts/serve_a3_ced_pd.sh` 拒绝 `PREFIX=1`），
  所以要开缓存仍需动那三处（见
  [`../../docs/CED-PD-CACHE-HIT-PLAN-20260925.md`](../../docs/CED-PD-CACHE-HIT-PLAN-20260925.md) §2）。
* 但"基础设施不支持"这条假说被本次实测否掉了 ⇒ 那三处改动是**有意义**的，
  而且现在有基线可作对照。

## 复现

```bash
# 1) 起基线 PD，PREFIX=1（会先清掉所有我们自己的容器并等 VLLMWorker 归零）
cd <shadow pkg>; bash /home/l00886679/tmp/20260924/ced_numeric/launch_prefix_baseline.sh
# 2) 计数器探针
cd <shadow pkg>; python3 pfx_probe.py  http://127.0.0.1:18992 http://127.0.0.1:18990 144000
# 3) 正确性探针
cd <shadow pkg>; python3 pfx_probe2.py http://127.0.0.1:18992 http://127.0.0.1:18990 144000
```

⚠️ `launch_prefix_baseline.sh` 第一版踩过"旧容器替新容器答 health"——
现在它会先 `docker rm -f` **全部**我们自己的容器、等 `VLLMWorker` 归零，
并且只有在**新 run 的 serve.log 里真的出现 `--enable-prefix-caching`** 时才报就绪。
