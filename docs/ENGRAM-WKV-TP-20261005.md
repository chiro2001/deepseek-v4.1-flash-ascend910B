# ★ engram gate 的 `wkv` 从"每卡复制"改为"按输出维分片"（2026-10-05）

> 来源：在 decode 步里找"最大的单个算子"时发现
> `MatMulV2 "6,6144;25600,6144"` **每步 2 次、每次 355–447 µs**（占步长 2.7%），
> 而它只是 `nn.Linear`，在 TP=8 上**每张卡都读同一份 315 MB 权重**。
> 标注：【实测】。

## 1. 问题

`models/deepseek_v41/model.py:774`（我们的挂载补丁）：

```python
self.engram.wkv = torch.nn.Linear(
    (config.engram_max_ngram_size - 1) * config.engram_n_heads * config.engram_head_dim,   # 6144
    (config.hc_mult + 1) * config.hidden_size,                                            # 25600
    bias=False, dtype=torch.bfloat16,
)
```

* 权重 `[25600, 6144]` bf16 = **314.6 MB**（每层），模型在 **层 1 与层 14** 各一份 ⇒ **629 MB**；
* decode 时 `M = 6` ⇒ 计算量可忽略，**纯权重载入**；
* 它是**普通复制**（TP=8 时 8 张卡各读 314.6 MB，输入还完全相同）⇒ **7/8 的读取是浪费**。

服务内实测：`p50 = 355 µs`、`max = 447 µs`、**2 次/步 = 0.709 ms/步**。

## 2. 隔离实测：分片能带来 8–10× 

tiny（单卡）上测三种形状（`[6, cols] × [rows, cols]`）：

| 形状 | 权重 | 时长 | 有效带宽 |
|---|---:|---:|---:|
| **全量（现状）** `rows=25600` | 314.6 MB | **254.2 µs** | 1237 GB/s |
| 1/8 输出分片 `rows=3200` | 39.3 MB | **24.4 µs** | 1609 GB/s |
| 1/8 输入分片 `cols=768` | 39.3 MB | 19.8 µs | 1988 GB/s |

⇒ 权重下降 8×、时间下降 10×（小矩阵的固定开销被摊薄）。

## 3. 改法（已实现，开关 `V41_ENGRAM_WKV_TP=1`）

**按输出维分片 + `all_gather` 拼回**：

```python
# __init__：本卡只建 [6144 → 3200]
# load_weights：checkpoint 的 [25600, 6144] 按 rank 切出 [3200, 6144]
# forward：
kv = layer.engram.wkv(lookup)              # [n, 3200]
kv = get_tp_group().all_gather(kv, dim=-1) # → [n, 25600]
```

为什么选**输出维**而不是输入维：输出维分片算出的每个输出元素**仍由一次 matmul 决定**，
推理链不变；输入维分片会引入 all_reduce，改变累加顺序。
（严格说输出维分片也会因为**输出宽度变了**而让底层 tiling 不同 → 浮点不完全逐位，
但语义等价；144K 验收 11/11 通过。）

### 3.1 机制验证（profile 逐项核对）

| 项 | 基线（`armF_r6_base`） | 臂 W（`armF_r7_wkvtp`） |
|---|---:|---:|
| `MatMulV2` 全量 `25600,6144` | **0.709 ms/步**（2 次） | **0（消失）** |
| 分片 matmul `3200,6144` | — | **0.071 ms/步**（1.86 次/步，p50 37.6 µs） |
| **全部 allGather**（含本次新增 + 原有的） | — | **0.138 ms/步**（4.65 次/步） |
| **净变化** | | **−0.709 + 0.071 + ~0.07 = −0.57 ms/步（−2.2%）** |

> 注意：`hcom_allGather__<数字>_<数字>_1` 这种名字**每个只出现一次**，
> 直接 `groupby(OP Type)` 会得到一堆 `n=1` 的干扰项 —— 必须先归并名字前缀再算每步次数
> （`tools/` 里的 `ag_check.py` 做法）。

## 4. 端到端结果

| 判据 | 结果 |
|---|---|
| 起服 | ✅ health=200（首次冷编译约 11 min，之后复用） |
| **144K 验收** | **11/11 通过，失败 0** |
| `hp`（按 batch 归一）vs 三个基线 | n=6 **−0.6~−0.9**、n=12 **−0.75~−0.8**、n=24 **−0.5~−0.8**、n=48 **±0.1** ms |
| 机制 | ✅ 全量 matmul 消失、分片版就位、验收通过 |

> `tools/ab_gate.py` 报的是**全部 batch 桶的中位**，其中 `n=18`（过渡桶）本次偏高
> ⇒ **中位数被单个过渡桶带偏**（+0.193）。真实并发档（n=6/12/24/48）里有三档一致变快。
> **教训**：`ab_gate` 的结果要**按真实并发桶复核**，不能只看中位数。

## 5. 处置

**已转交付默认**（`serve_a2.sh` 里 `ENGRAM_WKV_TP=${ENGRAM_WKV_TP:-1}`）。依据 = 机制 + 受控 A/B 双证据：

### 5.1 受控 A/B（同 bench 参数：conc 1,2,4,8 / 3 rep / out 192；各 1 run）

| batch n（真实并发档） | 样本数 A/B | 基线 A | 臂 W | Δ |
|---:|---|---:|---:|---:|
| **6**（conc=1） | 1080 / 1120 | 24.82 | **24.32** | **−0.50 ms（−2.0%）** |
| **12**（conc=2） | 344 / 336 | 27.64 | **27.20** | **−0.44 ms（−1.6%）** |
| **24**（conc=4） | 144 / 192 | 32.33 | **31.93** | **−0.40 ms（−1.2%）** |
| **48**（conc=8） | 64 / 88 | 41.72 | **41.06** | **−0.66 ms（−1.6%）** |
| 18（过渡桶） | **仅 24 / 40** | 32.04 | 33.82 | +1.78（**样本太少 + p90 达 1.9×10⁴ ms ⇒ 混入 prefill/ramp，不作判据**） |

* 4 个稳定桶**全部**更快，且 p10 同向；与机制预测的 −0.57 ms 一致；
* 144K 验收 **11/11 PASS**。

### 5.2 相关文件

`patches/files/model.py`（4 处：开关 / `__init__` / `forward` / `load_weights`）、
`scripts/serve_a2.sh`（`ENGRAM_WKV_TP` 透传 + 默认 1）。

## 5.5 ★ 途中踩到并修掉的一个**新缓存陷阱**（写给所有后续 A/B）

臂 W 把 `engram.wkv` 权重从 `[25600,6144]` 换成 `[3200,6144]`。跑完再起"基线臂"
（`WKV_TP=0`，权重仍是 `[25600,6144]`）时，**起服 30 分钟不 ready**，
日志里是：

```
File ".../torch_npu/dynamo/npugraph_ex/npu_fx_compiler.py", line 511, in __call__
    gm_result = self.run_kernel(*args, **kwargs)
AssertionError: expected size 25600==3200, stride 6144==6144 at dim=0
```

**根因**：基线臂**复用了臂 W 编译出来的图**（`torch.compile` 缓存键没有覆盖"权重形状"这一维）。
而 arm suite 原来只清 `cache/npugraph`（ACL 图），**没清 `torch_compile_cache`**。

**两个坑叠在一起**：
1. 得清**编译缓存**，不只是 ACL 图缓存；
2. **路径不能猜** —— `serve_a2.sh:1465` 把 **`$PKG/cache/vllm` 挂成容器里的 `/root/.cache/vllm`**，
   所以真正要清的是 **`$REPO/cache/vllm/torch_compile_cache`**（3.2 GB），
   而宿主 `~/.cache/vllm` 是**另一个目录**（我第一版清了它，等于没清，基线依旧失败）。

现在 `run_arm_suite.sh` 会**默认清编译缓存**（`CLEAR_COMPILE_CACHE=1`，代价是起服多几分钟）。
⇒ **凡改动张量形状/权重的 A/B，必须走这条**；否则"基线"会拿到改动臂的图，
表现为"起服极慢"或"结果荒谬"。

## 6. 复现

```bash
# 机制核对
python3 ~/tmp/ag_check.py <kernel_details.csv>       # allGather 每步次数（先归并名字前缀）
python3 ~/tmp/perstep_diff.py armF_r6_base armF_r7_wkvtp
# 端到端
bash tools/run_arm_suite.sh ~/tmp/launch_armW.sh armF_r7_wkvtp 1 1 1
```
