# `limit_core_num` 不是 HcPre 的杠杆：负结果（2026-10-07）

> 承接 `FRESH-CONC1-TOP-OPS-AND-HCPRE-BREAKDOWN-20261007.md` §2：
> `HcPre` 单次 33.60 µs 里**真正的数学只占 6.2%**（aic_mac 1.28 + aiv_vec 0.81 µs），
> ≈46% 是 AIV 未归因（同步/等待）、标量 31.5%、搬运 24.5%。
> 本文用**算子级微基准**检验"同步开销随并行核数增长"这一假设。全部为【实测】。

---

## 0. 结论

| 假设 | 判定 |
|---|---|
| "HcPre 的同步开销随**并行核数**增长 ⇒ 少用核更快" | ❌ **证伪** |
| "`limit_core_num` 可以用来给 HcPre 省时间" | ❌ **证伪**（API 自身开销就有 **+34%**） |

⇒ **HcPre 的 78% 开销与"用了几个核"无关**，改 tiling/核数这条路走不通。

---

## 1. 微基准（真实形状、真实 dtype）

**配置**（取自纯 conc=1 profile 的真实行）：

```
Input Shapes : "6,4,5120;24,20480;3;24;6,4"
Input Dtypes : DT_BF16;FLOAT;FLOAT;FLOAT;FLOAT
Output Shapes: "6,5120;6,4;6,4,4;6,4"
Block Num    : 24            ← 用满全部 cube 核
实测单次中位 : 33.60 µs（profile 口径）
```

**方法**：`torch.ops._C_ascend.npu_hc_pre_v2`（需先 `torch.ops.load_library(vllm_ascend_C*.so)`
+ `ASCEND_CUSTOM_OPP_PATH`），eager 调用 + 每次 `torch.npu.synchronize()`，
预热 60 次、200 次取中位（容器 `dsv41-op-hcfuse` / die5，**未碰 tp8k5**）。

| AIC 预算 | 中位 µs | 最小 µs | vs 不限 |
|---:|---:|---:|---:|
| **不限**（= 交付行为） | **90.87** | 88.01 | **1.00×** |
| 24 | 123.24 | 120.00 | **1.36×** |
| 20 | 121.71 | 118.28 | 1.34× |
| 16 | 121.27 | 117.94 | 1.33× |
| 12 | 121.59 | 118.43 | 1.34× |
| 8 | 121.47 | 118.30 | 1.34× |
| 4 | 121.96 | 119.54 | 1.34× |
| 2 | 121.82 | 117.90 | 1.34× |

### 1.1 两条读法

1. **显式设置 `limit_core_num` 本身就让耗时 +34%**，**哪怕限到满值（24/48）** ——
   这是 API 自身的开销（`set_stream_limit` + scope 进出），与被限的核数无关；
2. **把 AIC 从 24 一路降到 2，耗时毫无变化**（121.3~123.2 µs，全在噪声内）
   ⇒ **核数不是这个算子的成本驱动项**。

### 1.2 这个结果与"数学只占 6.2%"自洽

如果 93.8% 的时间不在做乘加，那么减少乘加并行度自然没有影响。
但**同时也说明**：那 ~46% 的"未归因 AIV"**不是跨核归约同步**（否则核数会有效）。

⇒ 更可能是：**Sinkhorn 的 20 次串行迭代的依赖链**（每轮都要等上一轮），
以及 MTE 等待 —— 这两者都与"用几个核"无关。

---

## 2. 由这条负结果反推：HcPre 还剩哪些路

| 路 | 是否可行 | 依据 |
|---|---|---|
| 减 Sinkhorn 迭代（20 → 12 或更少） | ❌ | `HCPRE-AND-SCATER-DEEP-DIVE`：**eps 下限使收敛极慢**，12 次仍差 8e-2 |
| 少用核（`limit_core_num`） | ❌ | **本文**（且 API 自带 +34% 开销） |
| 原生 PyTorch 实现 | ❌ | 实测**慢 35×** |
| 与 RmsNorm 融合（`hc_pre_norm`） | ❌ | `HCPRE-FUSE-E2E-NEGATIVE` + `AB-PRECISION-FLOOR`：**无可测收益**（且 <1% 判据下限） |
| A1（自适应 K_L0 128/64/32） | ❌ | 服务内**从未执行**（静态内核缓存未失效）；澄清后效应**低于噪声底** |
| **改 kernel 内部的迭代结构 / 把 Sinkhorn 从标量搬到向量** | ⚠️ **未验证** | 标量占 31.5%（aic_scalar 4.98 + aiv_scalar 5.62 µs）；这是唯一没试过的方向 |
| 减少**调用次数**（86 次/步 = 43 attn + 43 FFN） | ⚠️ 未验证 | 每层两处（attn 前 / FFN 前），是模型结构决定的最小值 |

⇒ **HcPre 这条线在"配置层/Python 层/融合层"已经全部试尽**，
剩下的只有**改 kernel 内部的算法结构**（把标量迭代向量化，或换成数学等价的闭式解）。

---

## 3. 顺带：本轮暴露的启动器缺陷（已修）

`serve_a2.sh` 的 `docker run -e` 是**显式白名单**，不转发任意 `V41_*`。本轮踩了两次：

| env | 后果 |
|---|---|
| `V41_HC_LIMIT_AIC` / `V41_HC_LIMIT_AIV` | 未转发 ⇒ 那一臂**整臂静默变成空操作**（白烧一次 10 min 重启） |
| `V41_HC_LIMIT_FILE` | 同上（且起服时**更早**于白名单修复，又白烧一次） |

**已修**：两处都补进白名单；且 `hc_pre` 的文件驱动补丁现在**有默认路径**
（`/home/l00886679/tmp/v41_hc_limit`），不再依赖 env。

**教训**：凡是"靠 env 门控"的实验，起服后**必须验证 env 真的进了容器**
（`docker inspect … .Config.Env` 或 `/proc/<pid>/environ`），否则会得到
"处理臂 = 基线臂"的假结论。

---

## 4. 复现

```bash
# 微基准（不占 tp8k5；用 die5 的 dsv41-op-hcfuse）
ssh a3-21 'docker cp ~/hc_limit_bench.py dsv41-op-hcfuse:/tmp/ && \
  docker exec -e ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer \
    dsv41-op-hcfuse bash -lc "cd /tmp && python3 hc_limit_bench.py"'

# 三个必踩的坑（脚本里已处理）：
#   1) npu_hc_pre_v2 需 torch.ops.load_library(vllm_ascend_C*.so)
#   2) hc_scale 形状是 [3]、hc_base 是 [24]（不是 [24]/[24]）
#   3) limit_core_num 的两个参数**都必须是整数**，传 None 会 TypeError
```
