# ★ 不要再手动钉 `KV_CACHE_MEMORY_BYTES`：改成**上限语义**（2026-10-06）

> 起因：用户指出「设置指定的 kvcache 容量可能导致 A2 OOM（见文档）。不建议手动指定一个值
> （或者可以指定一个最大值？）」。本文把这条需求落成机制，并给出**跨配置实测**。
> 全部为【实测】，除标注【推断】/【未确认】处。

## 1. 为什么"钉一个字节值"是错的

| 事实 | 证据 |
|---|---|
| 每 rank 可用 KV **因机器而异**：A3 = **16.60 GiB**、A2（910B3）= **14.40 GiB** | `armC_bat2048/serve.log`；`a2/docs/A2-DEPLOY-NOW.md` §B0 |
| 钉死 16 GiB 在 A2 上 = 要的内存比能给的还多 ⇒ **起不来/OOM** | 上行的两侧数字直接相减（差 2.2 GiB） |
| 但**又不能完全不管**：块号 × 页步长越过 2³² 会**静默读错** | `docs/KV32-OVERFLOW-DOSE-RESPONSE-20261006.md`（剂量-反应实测） |

⇒ 需要的语义是 **min(自动 profiling 给得多少, 32 位安全上限)**，而不是"钉一个数"。

## 2. 实现：`[V41-KV32-CAP]`（在 `core/deepseek_v41.py` 里，按几何自算）

```python
cap = floor(2³² / max(slot.page_size_bytes over all slots))   # 由**几何自己**算，不用配常量
num_blocks = min(auto_profiling_blocks, cap)                   # 只夹上限，不抬下限
```

* 关掉：`V41_KV_MAX_BLOCKS=off`（只有明确知道在越界时才用）；
* 收紧：`V41_KV_MAX_BLOCKS=<N>`（**只会更小**；写超上限的值会被夹回上限）；
* `0` / 未设 / `auto` ⇒ 用几何自算的安全上限（把"手滑写 0"当成"没设"，
  不让它把安全网摘掉）。

配套两件：

1. `scripts/serve_a2.sh` 的 `[KV32]` 起服后复核**优先采信 cap 日志里的真实块数**
   （否则会拿 profiling 值误判越界并拒绝起服 —— 实测：重排 + 自动 profiling 估 33,994 块，
   实际被夹到 32,768）；
2. `tools/apply_kv32_cap.py` / `tools/apply_kv32_repack.py`：把这两处改动**幂等地**套到
   任意底本的 `core/deepseek_v41.py`（A3 交付镜像、DCP overlay、A2 各自的底本都能用）。

## 3. 端到端实测（三套配置，全部**没有**手动指定 KV 字节）

### 3.1 A3 TP8 DCP1（交付默认形态）

```
[serve_a2][KV32] scope=auto pin=0 ⇒ 不设置 KV_CACHE_MEMORY_BYTES（交回 vLLM 自动 profiling）
[V41-KV32-CAP] 块数 33998 → 32768：自动 profiling 给得比 32 位安全上限多。
   每槽页步长 [131072, 131072, 131072, 131072]，上限 = floor(2^32 / 131072) = 32768
GPU KV cache size: 4,010,585 tokens          ← 与"手动钉 16 GiB"时**完全相同**
✓ [KV32] 池上界复核：32768 块 ≤ 上界 32768（来源：[V41-KV32-CAP] 夹取后）
```

| 项 | 值 |
|---|---|
| 容量 | **4,010,585**（现役 2,987,836 ⇒ **+34.2%**） |
| 正确性 | needle 60K **4/4**、150K **4/4** |
| 性能（地火 1024/256，2 rep 中位） | 并发 1：**121.0** 单流 / **122.3** 总吞吐；并发 8：57.3 / **366.5** |

### 3.2 tiny TP2 DCP2（夹具，验证"小池不被抬"）

```
[V41-KV32-CAP] 块数 86934 → 29076：自动 profiling 给得比 32 位安全上限多。
   每槽页步长 [131072, 131072, 131072, 147712]，上限 = floor(2^32 / 147712) = 29076
Available KV cache memory: 43.80 GiB          ← 自动 profiling（夹具显存富余）
GPU KV cache size: 3,019,745 tokens           ← 与"手动压到 29076 块"完全相同
```

⇒ 上限语义**只在超过安全线时动作**；没超过就完全等于自动 profiling 的结果。

### 3.3 A3 TP8 **DCP8**（容量模式）

配置：交付镜像 + DCP overlay + `--decode-context-parallel-size 8`、`BAT_TOKENS=2048`、
`GPU_UTIL=0.85`、`MAX_SEQS=16`、`SP_TOKENS=7`；**不指定 KV 字节**。

| 项 | 历史（2026-10-04，同 `GPU_UTIL=0.85`） | **本次（repack + cap + 自动）** | Δ |
|---|---:|---:|---:|
| `Available KV cache memory` | 11.52 GiB | **12.29 GiB** | +6.7% |
| **`GPU KV cache size`** | 17,142,231 | **18,864,624** | **+10.1%** |
| 1M 请求并发 | 16.35× | **17.99×** | +10.0% |
| 无 CAP 夹取 | — | ✅ 未触发（DCP 的真实上限更高） | — |

* 差值可分解：可用显存 +6.7% × 池 stride 缩短 +3.2%（`524288/540928`）≈ **+10.1%** ✓；
* **正确性**：needle 60K **4/4**、144K **4/4**；
* **性能**（地火 1024/256，2 rep 中位）：N=1 **61.7 / 58.3**、N=2 49.5 / 84.1、
  N=4 39.1 / 122.1、N=8 28.7 / **175.2**；历史 N=1 56.4/53.9、N=8 28.4/146.3
  ⇒ **容量涨了、性能也没退**（N=8 总吞吐 +19.7%）。

> 为什么 DCP8 不触发夹取：DCP 把 KV 按 `dcp` 路分片，worker 里**每个 slot 的页步长更小**
> ⇒ `floor(2³²/步长)` 更大。cap 用的是运行期真实的 slot 页步长，所以自动跟着放宽 ——
> 这正是"按几何自算"相对于"写死 32768"的价值。

## 4. 发布配置一览（本次覆盖）

| 配置 | 需要几 chip | 本次状态 |
|---|---:|---|
| A3 TP8 DCP1（交付默认，BAT=8192） | 8 | 作为基线在场（2,987,836） |
| **A3 TP8 DCP1 + BAT=2048 + repack + cap** | 8 | ✅ 容量 **4,010,585（+34.2%）**、24/24、性能达标 |
| **A3 TP8 DCP8 + BAT=2048 + repack + cap** | 8 | ✅ 容量 **18,864,624（+10.1% vs 历史）**、24/24、性能不降 |
| A2 TP8（910B3，可用 KV 14.40 GiB） | 8 | 【未确认】需在 A2 上跑（本机不可达）；机制上：profiling 给得少 ⇒ cap 不动作 ⇒ 不会 OOM |
| CED PD 分离（P/D） | **16** | 按用户要求**本次不做** |

## 5. 交付形态怎么用（不再需要钉值）

```bash
# A3 TP8 DCP1：只要这两条，KV 字节**留空**
V41_KV32_REPACK=1 BAT_TOKENS=2048 bash ~/tmp/launch_repackAuto.sh   # 内部不设 KV_CACHE_MEMORY_BYTES

# DCP8：overlay 里放好 repack+cap 版的 core/deepseek_v41.py（见 §6），其余同上
# 想要更保守：V41_KV_MAX_BLOCKS=30000（只会更小）
# 明确要越过 32 位线（不推荐）：V41_KV_MAX_BLOCKS=off
```

## 6. 给其它底本套补丁（含 A2）

```bash
# 1) 幂等套 repack（把 ratio-1 源的 index 平面挪进别的 slot 空闲区 ⇒ 上限 +12.7%）
python3 tools/apply_kv32_repack.py <原 core/deepseek_v41.py> <输出>
# 2) 幂等套 cap（上限语义）
python3 tools/apply_kv32_cap.py <上一步输出> <最终文件>
# 3) 校验：仿真 + 单测（不占卡）
python3 tools/kv32_repack_sim.py <最终文件>       # 期望 SIM: PASS（四槽等长 / pool stride 对得上）
bash tools/selftest_kv32_repack.sh                 # 正控 + 3 负控
bash tools/selftest_kv32_scope.sh                  # 守卫 24 项
```

> A2 上**只做 cap 不做 repack** 也是有意义的：cap 是**安全网**（越界就夹），
> repack 是**容量增益**（上限 +12.7%）。A2 的可用 KV 只有 14.40 GiB，
> 是否还能吃到 repack 的增益取决于它自己的池字节数 —— 需要在 A2 上实测一次
> （起服日志里 `[V41-KV32-CAP]` 行会直接给出"profiling 块数"和"上限"）。

## 7. 复现物

* A3 DCP1：`results/armrepackAuto2_1006_100410/serve.log`（cap 行 + 4,010,585）
* tiny：`ab_mkc_1001_215615/serve_tiny_cap.log`（43.80 GiB → 29,076）
* DCP8：`results/armdcp8cap_1006_101600/serve.log`（12.29 GiB → 18,864,624）
* KPI：`~/tmp/batcurve/{cap_KPI,dcp8cap_KPI}.log`；针：`~/tmp/batcurve/{cap_,dcp8cap_}needle_*.json`
