# ★★ 换 kernel 必须清 static kernel 缓存，否则改动**静默不生效**（2026-10-04）

> 本文用一个**零噪声指纹**判定了一个此前无法判定的问题：
> **HcPre A1 的 `.o` 确实在容器里（md5 已验），但服务执行的仍是旧内核。**
> 这推翻了 `A1-IN-SERVICE-NEGATIVE-20261004.md` 里"隔离收益不兑现"的解释。

## 1. 决定性证据：`aic_mac_time` 指纹

Track B（`docs/KERNEL-REALIZATION-STUDY-20261004.md`）在隔离环境建立了零噪声指纹：
`aic_mac_time` 是**指令/流水计数**，不受频率/温度/邻居争用影响 ——
A1 在 **11/11 轮**里恒为 **−0.68 µs**，零效应对照（stock vs stock）在 **8/8 轮**里恒为 **0.00**。
他们给出的判据原文：

> 若是 **1.28 → 0.60**，说明 A1 内核真的在跑；
> 若两边相同（如都 1.28 或都 0.60），说明**跑的不是同一个内核**。

**服务实测**（同形状 `6,4,5120`，精确匹配 `Input Shapes` 首段，样本 26880 vs 15280）：

| 列 | stock（`armF_meta2`） | **A1（`armF_A1`）** | 判定 |
|---|---:|---:|---|
| **`aic_mac_time`** | **1.282** | **1.282** | **相同 ⇒ 旧内核** |
| `aic_total_cycles` | 717,231.5 | 758,437.0 | 不降反升（A1 应 −10%） |
| `Block Num` / `Mix Block Num` | 24 / 48 | 24 / 48 | 相同 |
| `aic_scalar_time` | 4.623 | 4.778 | 相同量级 |
| `aiv_time` | 22.996 | 23.124 | 相同量级 |
| `Task Duration` | 30.800 | 31.000 | 相同量级 |

⇒ **服务里两臂跑的是同一个内核**，而 `.o` 文件确实不同（已逐字节验过）：

| 变体 | 镜像原版 | A1 包 |
|---|---|---|
| `HcPre_…67be.o` | `5aa7b15c…` | **`de790a50…`** |
| `HcPre_…67be_relocatable.o` | `7d2176f0…` | **`4a981e3d…`** |
| `.json` 里的 sha256 与实际 `.o` | — | **一致（已验）** |

排查掉的其它可能：**CANN 内置没有 `hc_pre`**（`find … opp -name "*HcPre*"` 为空）；
`/root/atc_data/kernel_cache` 只有 16 KB、不含 HcPre。

## 2. 剩下的唯一嫌疑：**static kernel 缓存**

`STATIC_KERNEL=1` + `NPUGRAPH_EX=1` 会做一次静态内核编译，产物落在
`cache/skcache/compile_outputs`（**13 GB**）。实测该目录**确实包含 HcPre 的静态内核产物**：

```
…/ts20260926180006878176_pid1738_outputs/1728_opcompile/HcPre_bfloat16_ND_32_4_5120_106.json
…/static_kernel_HcPre_1db0d47e972bb02d58a23e9e2a516d0ef83155469e57eae5c97c43b4587b63d9_26560_d0_compile_succ.log
…/static_kernel_cache/   ← 今天 11:00 之后仍有修改
```

⇒ **该缓存的 key 不含被替换 `.o` 的内容**（文件名哈希只由 op 定义决定，这一点 Track B 已实测），
所以换 vendor 里的 `.o` **不会让它失效** ⇒ 运行时继续用缓存里那份（= stock）。

**为什么 gmm1 的 armF 生效了、HcPre 的 A1 没生效**：
两次部署**都**清了 `cache/npugraph`，但**都没**清 `cache/skcache/compile_outputs`。
差异在于 armF 那次静态内核**重新编译**了（起服耗时长），而 A1 那次复用了缓存 ⇒ 静默用了旧内核。
**（这一条是推断，机制未逐字节证实；但 §1 的指纹结论不依赖它 —— "A1 没跑"是实测确定的。）**

## 3. 对既有结论的修正

| 文档 | 原结论 | **修正后** |
|---|---|---|
| `A1-IN-SERVICE-NEGATIVE-20261004.md` | "A1 隔离 −2.61µs、服务无效 ⇒ 隔离收益不兑现（兑现率 0%）" | **A1 根本没跑** ⇒ 该文档的"兑现率 0%"**不成立**，不能作为"隔离收益不可兑现"的证据 |
| 兑现率三档（A1 0% / armD 19% / armF 62%） | — | 只有 **armD（19%）与 armF（62%）** 是有效数据点；且两者都已用资源计数/Duration 验证过内核确实换了 |
| `KERNEL-REALIZATION-STUDY` 的"协议噪声 ±2µs/次" | — | **仍然成立且重要**（那是隔离环境内换内核的噪声底），与服务侧"内核没换"是两个独立问题 |

## 4. 今后所有 kernel 部署的硬流程（写进交付纪律）

1. **清三处缓存**（缺一即可能静默复用旧内核）：
   `cache/npugraph`、`cache/skcache/compile_outputs`、`cache/skcache/install`；
2. **验文件**：容器内 `.o` md5/sha256 与包内一致（`tools/verify_opp_vendor.sh` 步骤 0 已有）；
3. **验执行**（关键新增）：起服后从服务 profile 取一个**资源计数类指纹**
   （首选 `aic_mac_time`，其次 `aic_total_cycles` / 目标算子的 `Task Duration`），
   与基线比 —— **必须看到预期的变化**，否则一律判"内核没换"；
   ⚠️ `kernel_name` / 文件名哈希**不可用**（只由 op 定义决定，四臂同名）；
4. **`Task Duration` 只能当次要判据**：它噪声大（±2µs/次量级），
   不足以区分"内核没换"与"换了但收益小"。

## 5. A1 的最终处置

* **不采纳**（原因更新）：即便内核真的跑上，其隔离效应也只有 **−1.5~−1.7 µs/次**
  （换算 −0.12~−0.14 ms/step = **0.5%**），而服务侧单次的噪声底就是 **±2 µs/次**
  （80 次/步 ⇒ ±0.16 ms/step）⇒ **效应低于服务测量分辨率，无法验证，也不值得背一个非 stock 内核**。
* 交付保持 **armF only**（HcPre 已回 stock `5aa7b15c`，已验证过其 Duration 与计数）。
