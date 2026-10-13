# 路径 Y 判决实验报告：让 vendor"拥有"内置算子的 OpDef

**日期**：2026-10-05　**执行**：子代理 `/root/ced_die_budget`（die5；未碰 tp8k5/19210）
**产物**：`~/tmp/moe/`（`vendor_pathY/` = 被测包）　**标注**：【实测】/【实测·代码】/【推断】/【未确认】

---

## 0. 结论：**路径 Y 失败**（判决完成）

| # | 判据 | 结果 |
|---|---|---|
| 1 | 算子仍正常工作（n_tok×topk 全矩阵） | ✅ **16/16 输出逐字节一致，0 错误** |
| 2 | vendor 的 opsproto 被加载 | ✅ `/proc/self/maps` 可见 |
| 3 | **HARDMARKER 命中数** | ❌ **0**（priority=10000 **与** priority=1 两种都试了） |
| 4 | `CAND_A_ABORT=1` 是否让算子失败 | ❌ **不失败**（仍 attempt 1 OK） |

⇒ **"让 vendor 拥有 OpDef"不足以翻转内置算子的 tiling 来源。**
⇒ 路径 Y 与路径 X（改镜像内置库）之外的"纯 vendor 语义内"办法**已穷尽**。

**这把结论从"没找到办法"升级为"架构性结论"**：本 CANN 版本下，**内置算子的 tiling 不可被 vendor 覆盖**。

---

## 1. 做了什么

按你的第一步要求：**把 OpDef（proto.h）+ infershape 编进 vendor 的 `libcust_opsproto_rt2.0.so`，
tiling 保持原逻辑**（只带诊断用的 HARDMARKER，不含候选 A 的 `SetBlockDim` 改动）。

### 1.1 构建（三个目标文件，全部手工，未跑 cmake）
| 目标 | 来源 | 大小 |
|---|---|---|
| `tiling_markerY.o` | `op_host/moe_init_routing_v3_tiling.cpp`（原始逻辑 + HARDMARKER + 1 处版本兼容） | 532 KB |
| `infershape_pathY.o` | `op_host/moe_init_routing_v3_infershape.cpp`（含 `IMPL_OP_INFERSHAPE(MoeInitRoutingV3)`） | 223 KB |
| `proto_pathY.o` | autogen 风格 `#include "moe_init_routing_v3_proto.h"`（含 `REG_OP(MoeInitRoutingV3)`） | 26 KB |

### 1.2 产物与校验【实测】
| 库 | md5 | `MoeInitRoutingV3` 符号 | 原有算子是否保留 |
|---|---|---|---|
| `libcust_opmaster_markerY.so`（tiling） | `92ca593e…`(prio 10000) / `bf822987…`(prio 1) | **4**（原库 **0**） | `HcPre` 5 ✅ |
| `libcust_opsproto_pathY.so`（OpDef+infershape） | `2b64fd027e75168c1240b739911988e6` | **10**（原库 **0**） | `HcPre` 2、`GroupedMatmulSwigluQuantV2` 4 ✅ |

### 1.3 组装（`vendor_pathY/`）
在**原 vendor 副本**上替换两处 + 补一个头文件：
* `op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so` ← markerY
  （`op_tiling/liboptiling.so` 是指向它的**符号链接**，替换目标即同时生效）
* `op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so` ← pathY 版
* `op_proto/inc/moe_init_routing_v3_proto.h` ← 头文件（供打包一致性）

---

## 2. 判决证据

### 2.1 ✅ 算子仍正常工作：`n_tok × topk` 全矩阵 16/16 逐字节一致
16 组（n_tok ∈ {5,6,8,16,32,64,128,256} × topk ∈ {3,6}），固定随机种子 ⇒ 两臂输入 md5 相同；
对**全部 4 个输出**（`expanded_x` / `expanded_row_idx` / `expert_tokens_count_or_cumsum` / `expanded_scale`）
取字节级 md5 比对：

```
     case    input_md5    out一致  备注
      5x3         True     True
      ...         ...      ...
    256x6         True     True
一致=16  不一致=0  错误=0
```
⇒ **没有破坏算子**（这也回答了 gmm1 对照给你的"要小心"那点：kernel 查找没有被打断）。

### 2.2 ✅ vendor opsproto 确实被加载（maps 对照）
```
[VENDOR] /home/.../vendor_pathY/op_api/lib/libcust_opapi.so
[VENDOR] /home/.../vendor_pathY/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so   ← tiling
[VENDOR] /home/.../vendor_pathY/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so                        ← OpDef（新）
[builtin] /usr/local/Ascend/.../built-in/op_proto/lib/linux/aarch64/libopsproto.so                          ← 内置也在
[builtin] /usr/local/Ascend/.../built-in/op_impl/ai_core/tbe/op_host/lib/linux/aarch64/libophost_transformer.so
```
⇒ **三个 vendor 库全部加载，含新的 OpDef 库** ⇒ 不是"没加载"。

### 2.3 ❌ HARDMARKER 命中 **0**（两种优先级都试）
判据设计（不依赖日志级别、不依赖 profiler）：`PostTiling()` 开头 env 门控——
* `CAND_A_MARKER=1` → `OP_LOGE` 打一行 `[CAND-A-HARDMARKER] vendor tiling IS live: ...`
* `CAND_A_ABORT=1` → 打同一行后 `return ge::GRAPH_FAILED`

| 运行 | priority | 结果 |
|---|---|---|
| 基线（无 vendor）· `CAND_A_MARKER=1` | — | 算子 OK · HARDMARKER **0** ✅ 符合预期（对照） |
| `vendor_pathY` · `CAND_A_MARKER=1` | 10000（原始） | 算子 OK · HARDMARKER **0** |
| `vendor_pathY` · `CAND_A_MARKER=1` | **1**（抢优先级） | 算子 OK · HARDMARKER **0** |
| `vendor_pathY` · `CAND_A_ABORT=1` | **1** | 算子**仍 OK**（若被采用必须失败） |

⇒ **我们的 `PostTiling()` 从未被执行。**

### 2.4 已排除的替代解释
| 解释 | 排除方式 | 结果 |
|---|---|---|
| vendor 库没加载 | maps 对照（§2.2） | ❌ 已加载 |
| 注册优先级不够 | 读机制（`cases_[priority]` 升序试、同优先级先注册者胜）→ 改用 **priority=1** | ❌ 仍不生效 |
| tiling 被缓存 | 移走 `/root/atc_data/kernel_cache` + `/root/.cache/vllm` 后重测 | ❌ 仍不生效 |
| 构建产物不对 | 符号数 4 vs 0、`HcPre` 保留、编译无 error | ❌ 产物正确 |
| **没"拥有"OpDef** | **本次实验：把 OpDef+infershape 编进 vendor opsproto（符号 0→10）** | ❌ **仍不生效** |

---

## 3. 机制（【推断】，与全部证据一致）

内置 `libopsproto.so` **先加载**并注册了 `MoeInitRoutingV3` 的 OpDef。
若 `REG_OP` 注册是"先注册者胜"（与 tiling registry 同一风格），则：
```
内置先注册 OpDef → 该算子被归类为"内置算子"
                 → 运行时只查内置的 op_host (libophost_transformer.so) 做 tiling
                 → vendor 的 libcust_opmaster 永不参与该算子的 tiling 选择
```
**这条与三组独立证据一致**：① vendor tiling 库已加载但标记不命中；② vendor opsproto 已加载但无效；
③ gmm1 线的对照（**我们自己的算子**、**内核 .o** 两条通路都生效）。

**仍然【未确认】的**：内置 `libophost_transformer.so` 与 vendor `libcust_opmaster_rt2.0.so`
是否共用同一张 tiling 注册表。本轮无法在不反汇编的前提下判定；但**无论共用与否，结论一样**：
内置那条路对该算子总是胜出。

---

## 4. 与 gmm1 对照合起来看（机制全景）

| 替换对象 | 是否生效 | 证据 |
|---|---|---|
| **我们自己的算子** 的 **kernel `.o`** | ✅ | gmm1 armF：`aic_mac_time` 按预期变化 |
| **我们自己的算子** 的 **tiling** | ✅（同库内，机制相同） | 本轮 `HcPre` 等符号保留、库可重链 |
| **内置算子** 的 **kernel `.o`** | 未测 | — |
| **内置算子** 的 **tiling** | ❌ **本轮判决：不可覆盖** | HARDMARKER 0 命中（含 priority=1 + OpDef 两种） |
| **内置算子** 的 **OpDef** | ❌ 无效（即使编进 vendor） | §2.3 |

⇒ **可复现的优化面 = "我们自己声明的算子"**；**内置算子的 tiling/OpDef 不可改**，
想改只能走 **路径 X（镜像层 patch 覆盖内置 `libophost_transformer.so`）**。

---

## 5. 若仍要做候选 A：只剩路径 X（未实施）

**改法**：把我们编译好的 tiling 合并进**内置 `libophost_transformer.so` 的副本**，
用镜像层 patch 覆盖 `/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/aarch64/libophost_transformer.so`。

| 项 | 评估 |
|---|---|
| 收益 | 【推断】240–400 µs/步（0.9–1.5%）（依据仍是 §6 的固定开销分析） |
| 改动量 | 中（链接一个 4 MB+ 内置库 + 镜像层 patch 脚本） |
| 风险 | **高**：脱离 vendor 语义；覆盖 CANN 内置文件；需与 `[OPP-OVERRIDE]` 的目录复制语义配合 |
| 必须先验的 | ① 合并后的 `libophost_transformer.so` 仍能正常加载（`ldd` 无缺符号）；② 用**同一个 HARDMARKER** 确认 tiling 真的被采用；③ 全矩阵正确性 |

**我的建议**：仅在"候选 A 的收益被别的证据进一步抬高"时再做；否则**不划算**（1% 量级的收益 vs 覆盖 CANN 内置库的长期维护成本）。

---

## 6. 候选 A 的收益依据（不变）

服务 profile `armF_meta2_1004_2010`（rank0，n=6 那一档 13400 次）【实测】：

| n_tok | topk | rows | TaskDur p50 | **aiv_vec_time** | BlockNum |
|---:|---:|---:|---:|---:|---:|
| 5 | 3 | 15 | 11.5 µs | 0.11 | **48** |
| 6 | 6 | **36** | **17.8 µs** | **0.30** | **48** |
| 256 | 6 | 1536 | 23.8 µs | 4.68 | **48** |

* rows ×102 ⇒ 时长仅 ×2.07 ⇒ **固定部分 ≈17.7 µs（≈90%）**，边际 0.004 µs/行
* 真正向量计算 **0.30 µs = 1.7%**
* `Tiling4VBSCompute()` 早已判定"单核排序"（`sortMode_=0`），`PostTiling()` 仍 `SetBlockDim(aivNum)=48`

---

## 7. 交付要求（若走路径 X 时遵守）

1. **`cache/skcache/compile_outputs` 不会自动失效** ⇒ 换库后服务侧须用 `V41_OPP_CLEAR_SKCACHE=1`
   强制重编译（起服 +18 min），否则可能静默复用旧产物。
2. **服务侧验执行判据**：首选 `Block Num`（48→1）或 `aiv_total_cycles`（指令/流水计数，抗噪）。
3. **保留 env 门控硬标记**（`CAND_A_ABORT` / `CAND_A_MARKER`）—— 本轮证明它是最可靠的"tiling 是否生效"判据。

---

## 8. 卫生

* **原 vendor 未被污染**（逐项核对）：`libcust_opsproto_rt2.0.so` = `2f4f11c9…`（未动）、
  `libcust_opmaster_rt2.0.so` = `876a3652…`、`libtransformer_aicpu_kernels.so` = `31f7a0ef…`、
  `op_proto/inc/` 无我加的头文件
* 移走的 `/root/atc_data/kernel_cache` **已恢复**
* 被测包在 `~/tmp/moe/vendor_pathY/`；旧包保留为 `.bak<时间戳>`
* 构建树里新增了 `csrc/moe/moe_init_routing_v3/`（**仅** `op_host/*.cpp`，**未进 `build.ninja`** ⇒ 惰性、
  不影响任何既有构建；下次要接着做 X 时可直接复用）
* 未碰 `dsv41-tp8k5`/19210；未用 `rm -f`（一律 `mv` 到 `.bak<时间戳>`）
* 工具：`build_pathY.sh`、`link_pathY.sh`、`make_vendor_pathY.sh`、`pathY_test.sh`、
  `check_maps2.py`、`dump_matrix.py`、`run_matrix.sh`、`fix_dump.py`
