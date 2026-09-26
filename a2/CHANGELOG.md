# A2 分支变更说明 —— `feat/kv8-dram-offload-pending`

> 2026-09-22。本文件说明这个分支相对 `main` 加了什么、每条的状态。

---

## 一句话

**DRAM KV 卸载：已验证可用**（8 卡真权重、16 请求、replay 17.5×）。
**KV8（long-KV INT8）：技术面完成，但时延负收益 ⇒ 待测**。

---

## 1. 新增目录 `a2/`

```
a2/
├── README.md      ← 两条线的完整说明（问题 / 根因 / 效果 / 参数 / 待验证）
├── CHANGELOG.md   ← 本文件
├── patches/       ← 四个可交付补丁 + 挂载与自检说明
└── logs/          ← 四份关键日志（已脱敏）
```

---

## 2. 逐项变更

### 2.1 ✅ DRAM KV 卸载（已验证）

| # | 变更 | 文件 | 状态 |
|---|---|---|---|
| 1 | `state` 组从**存/查两侧**排除（参与位） | `patches/0001-offload-scheduler.patch.py` | ✅ 8 卡验证 |
| 2 | per-group `blocks_per_chunk`（SWA=1、full=8）⇒ 池子 **4.89× 变小** | `patches/0001{,.b,.c}*.py` | ✅ 单卡 11 臂验证 |
| 3 | 池子改走 `aclrtHostRegister`（绕开 `aclrtMallocHost` 的 `207001`） | `patches/0002-offload-cpu-pool-host-registered.patch.py` | ✅ 8 卡验证（128/128 次 `ret=0`） |

**效果**（`logs/016`，臂 `l2-dram58-16p`）：

| 判据 | 修复前 | 修复后 |
|---|---|---|
| `BlockStored(CPU)` | 12,288 | 6,144 |
| `CPU_to_GPU` | **0** | **12.44 GB** |
| `external_prefix_cache_hits` | **0** | **507,904**（replay 轮 **96.9%**） |
| **replay / fill TTFT** | 4167 / 4192 ms | **253.4 / 4429.4 ms（17.5×）** |

**拐点**（`logs/022`，三条臂夹死）：池子 **0.667× ⇒ 归零** / **1.000× ⇒ 17.9×** / 1.209× ⇒ 17.5×。

### 2.2 ⏳ KV8（long-KV INT8，待测）

| # | 变更 | 状态 |
|---|---|---|
| 1 | 替代设计：`PA_BBND scratch + identity block table + 索引重编号`（原 `layout_kv="TND"` 在 A2/A3 上不存在） | ✅ 逐比特精确 |
| 2 | 引擎集成（`VLLM_V41_KV8=1`，接在真实调用点） | ✅ 已接 |
| 3 | `AscendSlidingWindowMLASpec` 加 `scale_dim`（SWA 也量化，页 131072 → 66560 B） | ✅ 已加 |
| 4 | 读侧 gather 换 flat `index_select`（逐比特等价、带宽 ×2.1） | ✅ 已换 |
| 5 | **融成 1 个 kernel**（时延唯一出路） | ⏳ **待测** |
| 6 | **compressor state ring 缩到 BF16**（容量唯一出路） | ⏳ **待测** |

**为什么待测**：时延 **+21%**（判据 ≤+0.2%）、容量 **×1.135**（判据 ×1.84）。
两条根因都已定位（`logs/025`）。

---

## 3. 与既有发布内容的关系

* **不冲突**：`a2/` 是新增目录，不动 v8 的既有补丁、脚本与 `MANIFEST.sha256` 覆盖的路径；
* **适用范围**：这些补丁针对 **vLLM 0.27.1 + vllm-ascend `e43cf1e9f`**（v8 镜像的那一套）；
  换上游版本时行号锚点要重核（`patches/README.md` 里有生成器的锚点校验说明）。

---

## 4. 未验证（**不要当已通过**）

| # | 事项 |
|---|---|
| 1 | **A2 实机**：`aclrtHostRegister` 在 `host_mem_pool=0` 的 A2 上是否可用（**最大风险**） |
| 2 | per-group bpc 的 8 卡验证（进行中） |
| 3 | `state` 组跳过的**数值正确性**（只做了语义推断） |
| 4 | `×6.945` 宿主乘数在 A2 上 |
