# 线 A 的 Scatter 微基准：固定启动开销主导，"换算子"方向被否（2026-10-06）

> 目的：A1'/A4 的预期收益是「换官方 `npu_scatter_nd_update_asc`（+1~3%）」。
> 在投入编译自定义算子之前，先量化这 0.721 ms 真实暴露**到底是什么开销**。
> 结论：**是固定启动开销（~43 µs/次），不是带宽**；换 `_asc` 解决不了（同一 aclnn 机制）。
> 全部为【实测】。

## 1. 微基准（tiny，a3-21 chip2，`~/tmp/scatter_bench.py`）

形状按生产：`var=[4096*128, 512] bf16`、`indices=[T,2] int32`、`updates=[T,512] bf16`。

| T | 中位 µs | 最小 µs | 搬运 KB | **有效带宽 GB/s** |
|---:|---:|---:|---:|---:|
| 1 | 43.5 | 41.6 | 1.0 | 0.02 |
| 8 | 46.3 | 42.0 | 8.0 | 0.16 |
| 16 | 50.1 | 43.1 | 16.0 | 0.30 |
| 64 | 68.1 | 62.6 | 64.0 | 0.90 |
| 512 | 184.5 | 182.8 | 512.0 | 2.65 |

**读法**：

* 数据量涨 **512×**，耗时只涨 **4.2×**（43.5 → 184.5 µs）⇒ **固定开销约 43 µs**；
* 有效带宽最高只有 **2.65 GB/s**，而 HBM 上限 **1182 GB/s** ⇒ **差 3 个数量级**，
  完全是**延迟/启动受限**，与带宽无关。

## 2. 对照组：没有更快的现成算子

| 实现 | T=8 | T=64 |
|---|---:|---:|
| `torch.ops._C_ascend.npu_scatter_nd_update_sk`（我们现用） | **46.3 µs** | **68.1 µs** |
| `torch_npu.npu_scatter_nd_update_`（通用版） | 53.2 µs | 62.1 µs |
| 纯索引赋值 `var[rows] = upd` | 101.0 µs | 102.0 µs |

⇒ 我们**已经在用最快的那一个**。通用版在 T=8 更慢；PyTorch 索引赋值慢一倍。

## 3. 每步合计：与 profile 对得上

```
58 次调用（T=8）：中位 1013.7 µs = 1.014 ms/步
profile 实测（armF_r6_base）：1.180 profile ms = 0.721 真实 ms/步，58 个/步，中位 20.4 µs
```

⇒ 微基准（同步计时、含 host 开销）比图内实测略高，量级一致，说明**账对得上**。

## 4. 两个被否掉的方向 + 一个真方向

### ❌ 「换 `npu_scatter_nd_update_asc`」不能解决问题

`_asc` 是 CANN 的 AscendC 实现（`scatter_nd_update_asc_pure_copy.h`，AIV 纯拷贝），
**走的是同一套 aclnn 启动路径**——固定开销那一层不会消失。
而且当前容器里**没有**这个算子（`torch.ops.custom.scatter_nd_update_asc` 不存在），
要从 `cann-recipes-infer` 编译，成本不低。**建议不投入。**

### ❌ 「算子本身慢」也不是主因

现用 `_sk` 已经是三者中最快的；单次 18.9 µs 的 device 侧耗时（`core=MIX_AIV`）
对 8 KB 数据属于**核启动 + 调度**开销，不是算力问题。

### ✅ 真方向：**合并调用次数**（58 次 → 更少）

微基准直接给出上界：

| 方案 | 耗时 |
|---|---:|
| 58 次 × T=8 | **1013.7 µs** |
| 1 次 × T=512（同样数据量） | **184.5 µs** |

⇒ **同样字节数，合并后快 5.5×**，理论上可回收 **~0.8 ms/步 ≈ +3.3%**。

**可行性**：`dsa_v41.py` 自己的注释写着
> "V4.1 cache planes can be **views into a larger layer-outermost slot**"

⇒ 各层 cache 很可能是**同一大 buffer 的视图**，这是合并所需的布局前提。
但障碍在于**逐层流水**：kv 是在各层 forward 里分别算出来的，要合并就得先落到 staging buffer、
步末统一 scatter——那会引入额外拷贝与同步，需要专门评估。

## 5. ★ 合并方案的可行性判定（本轮补完）：**被依赖结构挡住**

合并写入（把 58 次调用并成少数几次）需要把各层的 KV 攒到步末统一写。但它**做不到**：

**布局侧是完全可行的**（这是好消息，说明不是布局问题）：

```python
# core/deepseek_v41.py: plan_cache_slots 尾部
placements.extend(CachePlacement(name, 0, _cap) for name in p["aliases"])
```

即同一 slot 内的**所有别名层 offset 都是 0、block_stride 相同** ⇒ 索引公式一致
（线性行号 = `block × (block_stride/row_bytes) + row`），拼一次调用在数学上没有问题。

**但依赖结构不允许延后**（这是决定性的坏消息）：

```python
# attention/dsa_v41.py:3836-3860（同一层的 forward 内）
compressed_indices = self._select_sparse_indices(...)   # 读压缩 KV/索引
...
attention_output = self._attention(attn, q, metadata, ...)   # ★ 读 SWA 缓存
```

而 SWA 的**写**发生在**更早**的 `preprocess(...)` / `_write_compressed_source(...)` 里
（`scatter_cache_sk(...)`）。也就是说**同一层内是「先写 SWA 缓存、再被本层注意力读」**。

这在本仓已有**独立佐证**：`dsa_v41.py:3840` 有一段 [V41-SYNCATTN] 注释，
记录的正是"发散发生在注意力**读** SWA 缓存与本层给该缓存**写** K/V 之间（同一 forward 内）"，
并提供了 `sync_attn=1` 的串行化判别开关。

⇒ 写入必须在注意力之前**完成并可见**，**不能攒到步末**。合并方案在结构上不成立。

## 6. 结论

1. 线 A 的 `ScatterNdUpdateSk`（0.721 ms 暴露，+2.9%）**三条路径全部被证伪**：
   * 换 `_asc`：同一 aclnn 启动路径，固定开销不变（且需额外编译）；
   * 换专用 PA 写算子：形状（head=1）不被接受；
   * 合并调用：**被「同层先写后读」的依赖结构挡住**。
2. ⇒ **该项应从线 A 划掉**，不建议再投入。
3. 遗留的唯一可能：把 scatter **融进上游算子**（例如 SFA 自带的 KV 写入），
   属于 kernel 级改动，不在"低风险小改动"范围内。

## 6. 复现

```bash
scp ~/tmp/scatter_bench.py a3-21:~/tmp/
ssh a3-21 'docker cp ~/tmp/scatter_bench.py dsv41-tinyspark:/tmp/ && \
           docker exec dsv41-tinyspark bash -lc "cd /tmp && python3 scatter_bench.py"'
```
