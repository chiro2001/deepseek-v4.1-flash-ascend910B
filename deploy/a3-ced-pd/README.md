# DeepSeek-V4.1-Flash —— **A3 单机 8+8 PD 分离 + CED 形态**

> 这是一个**部署形态**（deploy form），不是一个实验目录。
> 它把"怎么在**一台 A3（8 块 910C 卡 = 16 个 die）**上起 DeepSeek-V4.1-Flash 的
> **PD 分离 + CED** 服务"固化下来，并提供**两种等价、可交叉验证的交付面**。

## 0. 一句话

```
P（die 0–7，= 卡 0–3）  ：只跑 layer 0–19 + layer-20 全局源投影（≈40% 的 prefill 计算）
D（die 8–15，= 卡 4–7）：128-token 有界重放 + 全 40 层 decode
proxy          ：官方 load_balance_proxy，客户端只连它
```

* **为什么叫 CED**：P 侧不做完整 40 层，只把 D 需要的全局 KV 源（层 2/8/14/20 的
  `long_kv` + `index_k`）与**最后 128 个 token** 的重放区间留给 D ⇒
  prefill 计算量按层数比下降。实测 prefill 相对全 40 层基线 **2.07×（144K）**。
* **DSpark**：D 侧开推测解码（`SPEC=1 SP_TOKENS=7 DRAFT_GRAPH=1`）。
  草稿层的 KV 不是自己算的，而是从 target 隐状态投影出来，而草稿可见窗口恰好
  也是 128 ⇒ **CED 的重放步正好把草稿窗口一次喂满**（两边都是 128 不是巧合，
  都绑在 `DeepseekV41SWASpec.sliding_window == 128` 上）。
* **KV 精度**：BF16（`KV_DTYPE=bfloat16`），不是 INT8。

> 📦 **可以直接下载已构建好的镜像包**（不需仓库）：
> [Release `a3-ced-pd-v3`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/releases/tag/a3-ced-pd-v3)
> —— `dsv41-a3-ced-pd-imagekit-v3.tar.zst`（3.8 MB，基线 `main@7497a06`）。
> v3 = **交付默认**（D 侧 DSpark 开、两侧前缀缓存开、D 图模式）+ decode 请求边界护栏；
> v1 连护栏都没有，不要用。

## 1. 两种交付面（**逐字节等价**）

| | ① GitHub 形态 | ② 镜像层 patch 形态 |
|---|---|---|
| 怎么用 | 克隆仓库 + `PATCH_MODE=mount` | 装 `local/dsv41-a3-ced-pd:v1` + `PATCH_MODE=baked` |
| 我们的文件从哪来 | `-v` 挂进容器 | 烘在镜像真实路径 |
| 需要网络/仓库 | 需要 | **不需要**（`docker load` 即可） |
| 适用 | 开发、改代码、做单变量 | 部署、跨机复现、无网环境 |
| 一致性 | 由 `verify_consistency.sh` **逐文件 md5 比对**保证 | 同左 |

> 这两面装的**是同一批文件**，清单见 [`PAYLOAD.md`](PAYLOAD.md)。
> `bash deploy/a3-ced-pd/verify_consistency.sh` 会逐文件比对，
> 任一不一致即 FAIL —— **这是"可复现"的判据，不是口号。**

## 2. 目录内容

| 文件 | 作用 |
|---|---|
| [`PAYLOAD.md`](PAYLOAD.md) | **payload 清单（唯一事实源）**：文件→容器目标 的完整映射 |
| `build_payload.sh` | 从仓库组装 payload 树（`payload/`，生成物不入库） |
| `Dockerfile` | 工作镜像定义（官方基底 + 我们的 1 层） |
| `build_image.sh` | 组装 payload → `docker build` → 打印 `BUILD_INFO.txt` |
| `verify_consistency.sh` | **★ 逐文件比对镜像内 vs 仓库**（两个交付面的一致性判据） |
| `launch/serve_p.sh` / `serve_d.sh` | 起 P / 起 D（本形态的**验证过的**参数） |
| `launch/serve_proxy.sh` | 起官方 load_balance_proxy |
| `launch/stop_all.sh` | 停掉本形态的三个容器 |
| `launch/smoke.sh` | 144K 四针冒烟（正确性最小判据） |

## 3. 起服（两种形态命令只差一个环境变量）

### 3.1 形态 ①：仓库 + 挂载

```bash
cd <克隆下来的仓库>
export MODEL=<完整模型目录>            # 273 GB，需现场准备
bash deploy/a3-ced-pd/launch/serve_p.sh      # chip 0–7   → :18990
bash deploy/a3-ced-pd/launch/serve_d.sh      # chip 8–15  → :18991
bash deploy/a3-ced-pd/launch/serve_proxy.sh  #            → :18992
```

### 3.2 形态 ②：工作镜像 + 烘焙

```bash
# 先拿到镜像（二选一）
docker load < dsv41-a3-ced-pd-imagekit-v3.tar.zst    # 或
bash deploy/a3-ced-pd/build_image.sh                  # 有基底镜像时本地构建

export MODEL=<完整模型目录>
export PATCH_MODE=baked
bash deploy/a3-ced-pd/launch/serve_p.sh
bash deploy/a3-ced-pd/launch/serve_d.sh
bash deploy/a3-ced-pd/launch/serve_proxy.sh
```

> `PATCH_MODE` 会透传到底层启动器；两侧都装同一批文件。

## 4. 起服后的**硬门**（不看这些就别相信结果）

按"漏了会怎样"排序。每条都给出**实际生效后的可观测痕迹**，
而不是"我传了这个变量"。

| # | 检查 | 期望 | 漏了会怎样 |
|---|---|---|---|
| ① | `grep -a "CED-32BIT-GUARD" d/serve.log \| head -1` | `num_blocks=29076 max_page_stride=147712` | 1M 静默空答（HTTP 200 + 1 token EOS） |
| ② | `grep -ac "one-token prompt tail forced eager" d/serve.log` | **> 0**（通过验收的那台 240） | 144K 多轮**乱码**（HTTP 200 + 打满 + 含 `<｜box｜>`） |
| ③ | 容器内 `echo $MULTISTREAM $DSA_OVERLAP` | **`0 0`** | 长上下文**静默算错**（实测 0/4） |
| ④ | `grep -a "dspark-graph-capture" d/serve.log \| head -1` | `built draft attention metadata (groups=1 layers=3 …)` | `A≈1.0`（草稿白跑），**ms/step 看不出问题** |
| ⑤ | `grep -a "SpecDecoding metrics" d/serve.log \| tail -1` | `Mean acceptance length` **≈2.4–3.4** | 同 ④ |
| ⑥ | `grep -a "CED decode: upper SWA groups" d/serve.log` | `(7,8,9,10,11) … mid` | 组契约错位 ⇒ 清零清错页 |
| ⑦ | `bash experiments/dspark/check_static_kernel.sh <run_dir>` | `static shape kernel will be used` **> 0** | 白付 ≈4.4 ms/step（**不影响正确性**） |

⚠️ **⑥⑦ 两个坑**（都踩过）：
* 判静态核**不要**用 `static kernel compile start` —— 编译缓存命中时它是 0，
  会给出**假阴性**。真判据是 `static shape kernel will be used`。
* 「文档里的硬门」本身会过期：`[CED-META] inline metadata` 曾是硬门，
  实际是**死开关**（无代码读取）。每条硬门都要能指出**打印它的代码**。

## 5. 本形态的**已验证**结果（A3-21，真权重）

| 项 | 144K | 1M |
|---|---|---|
| 四针 A/B/C/D | **4/4** | **4/4** |
| 流式 TTFT | 10.57 s | 100.04 s |
| 多轮（三轮） | **3/3** | **3/3** |
| 缓存命中（`PREFIX=0` 口径） | ✅ a1/a2 | ✅ a1/a2 |

**总计 21/21 通过**，四针答案与 `SPEC=0` 交付口径**逐字节相同**
（`ZQ7K-3341` / `VX2M-8890` / `HT4P-5527` / `RB9N-6014`）。

**性能**（step 口径，detail 见下）：

| 口径 | 数值 |
|---|---|
| prefill 相对全 40 层基线 | **2.07×**（144K） |
| decode ms/step（并发4，4 路） | **41.07 ms**（`STATIC_KERNEL=1`） |
| decode 单流（2048/256，《地火》） | **99.7 tok/s**，A=2.76 |
| 每 token 时延（并发4） | 11.5–12.1 ms（相对 `SPEC=0` 的 26.8 **≈2.3×**） |

⚠️ **DSpark 的收益只在低并发成立**：并发 4 时它让 ms/step 从 28.2 涨到 41.1（**1.45×**），
换 A≈2.4。收益集中在接受长度高的请求上，成本由全批承担 ——
高并发吞吐场景应保持 `SPEC=0`，或按并发自适应（见
[`docs/CED-PD-DYNAMIC-SPEC-20260926.md`](../../docs/CED-PD-DYNAMIC-SPEC-20260926.md)）。

## 6. 前置条件

| 项 | 要求 |
|---|---|
| 硬件 | **单台 A3，8 块 910C 卡 = 16 个 die**（P 用 die 0–7 = 卡 0–3，D 用 die 8–15 = 卡 4–7） |
| 镜像 | `quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3`（18 层）或本形态的工作镜像 |
| 模型 | DeepSeek-V4.1-Flash W4A8 + DSpark 权重（273 GB），`MODEL=<路径>` 传入 |
| 内核 KV | **BF16**（本形态不用 int8） |
| 端口 | P `18990`/`19090`、D `18991`/`19091`、proxy `18992`（可用 env 覆盖） |

**共用机注意**：A3 上 `CPU_BIND=0` 是**必需**的逃生口（目标 NUMA 节点被占满时
`migratepages` 会 100% CPU 无限自旋，服务永不就绪，连 `docker stop` 都拿不到
exit event）；`DROPCACHE=0` 默认不清整机 page cache（会打到别人）。

## 7. 架构与文档地图

| 想了解 | 看 |
|---|---|
| 整体交接 | [`docs/CED-PD-HANDOVER-20260926.md`](../../docs/CED-PD-HANDOVER-20260926.md) |
| 验收矩阵与判据 | [`docs/CED-PD-ACCEPTANCE.md`](../../docs/CED-PD-ACCEPTANCE.md) |
| 性能（prefill/decode/吞吐） | [`docs/CED-PD-PERF-20260925.md`](../../docs/CED-PD-PERF-20260925.md) |
| DSpark × CED 的架构关系 | [`docs/CED-PD-DSPARK-CED-RELATION-20260926.md`](../../docs/CED-PD-DSPARK-CED-RELATION-20260926.md) |
| DSpark 开启与验收 | [`docs/CED-PD-DSPARK-ACCEPTANCE-20260926.md`](../../docs/CED-PD-DSPARK-ACCEPTANCE-20260926.md) |
| **decode 时延拆解** | [`docs/CED-PD-DSPARK-LATENCY-BREAKDOWN-20260926.md`](../../docs/CED-PD-DSPARK-LATENCY-BREAKDOWN-20260926.md) |
| 精度/乱码措施总账 | [`docs/CED-PD-ACCURACY-MEASURES-20260926.md`](../../docs/CED-PD-ACCURACY-MEASURES-20260926.md) |
| 32 位页步长上界 | [`docs/CED-PD-BLOCK-BOUND-20260925.md`](../../docs/CED-PD-BLOCK-BOUND-20260925.md) |
| 按并发切 decode 路径 | [`docs/CED-PD-DYNAMIC-SPEC-20260926.md`](../../docs/CED-PD-DYNAMIC-SPEC-20260926.md) |
| 历史 24ms 那一发的参数 | [`docs/CED-PD-HISTORICAL-24MS-PARAMS-20260926.md`](../../docs/CED-PD-HISTORICAL-24MS-PARAMS-20260926.md) |
