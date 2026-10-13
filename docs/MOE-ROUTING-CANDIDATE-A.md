# 候选 A 实施报告：`MoeInitRoutingV3` 小 batch 降 BlockDim

**日期**：2026-10-05　**执行**：子代理 `/root/ced_die_budget`（die5；未碰 tp8k5/19210）
**产物**：`~/tmp/moe/`　**标注**：【实测】有运行数据；【实测·代码】有明确代码路径；【推断】由实测推导；【未确认】无证据

---

## 0. 结论速览：**改动做完并编译通过，但无法交付 —— 负结果**

| # | 事项 | 结果 |
|---|---|---|
| 1 | 两处成对改动 + 顺序问题处理 | ✅ **已完成**（§1，含 1 处版本漂移兼容） |
| 2 | 重新构建 tiling 库（含新算子） | ✅ **已打通**（§2：手工编译 + 手工重链，产物正确） |
| 3 | **vendor tiling 覆盖内置 tiling** | ❌ **失败**（§3：硬判据实测，vendor 库已加载但不被采用） |
| 4 | 正确性对拍 / BlockNum 48→1 | ⛔ **无法进行**（改动的 tiling 从未被执行，§3.3） |
| 5 | 结论 | **候选 A 无法通过 `V41_HC_OPP_PKG`（vendor）交付**；见 §5 的两条替代路径 |

**一句话**：改动本身没问题（能编译、成对性已处理），但**这套 CANN 不允许 vendor 覆盖"内置算子"的 tiling** —— 内置 `libophost_transformer.so` 的注册始终胜出，与注册优先级无关。

---

## 1. 改动内容（已完成）

文件：`moe/moe_init_routing_v3/op_host/moe_init_routing_v3_tiling.cpp`

### 1.1 成对改动（落在 `PostTiling()`）
```cpp
ge::graphStatus MoeInitRountingV3TilingBase::PostTiling()
{
    if (isEmptyTensor_) {
        context_->SetBlockDim(1);
        moeInitRoutingV3TilingData.set_coreNum(1);      // ★ 成对
    } else if (sortMode_ == 0UL) {
        context_->SetBlockDim(1);                        // ★ CAND-A
        moeInitRoutingV3TilingData.set_coreNum(1);      // ★ 必须成对
    } else {
        context_->SetBlockDim(aivNum);
        moeInitRoutingV3TilingData.set_coreNum(aivNum);
    }
    ...
}
```
**顺序问题的处理**（任务特别点名）：原 `set_coreNum(aivNum)` 在 `GetPlatformInfo()`（第 304 行），
而 `sortMode_` 要到 `DoOpTiling()` → `Tiling4VBSCompute()` 才赋值。
⇒ **没有去挪动 `GetPlatformInfo`，而是把 `set_coreNum` 一并放进 `PostTiling()`**，
与 `SetBlockDim` 紧邻、且在 `sortMode_` 定稿之后 ⇒ 两者永远一致，也不会漏改。
（`GetPlatformInfo()` 里那句保留原样，只是一个初值，会被 `PostTiling()` 覆盖。）

### 1.2 版本漂移兼容（1 处，语义中性）
编译时报：
```
error: 'struct Ops::Transformer::OpTiling::AiCoreParams' has no member named 'numBlocks'
```
对比两版结构体：**除字段名外完全一致**（同为 `uint64_t`、同一位置）——
旧版叫 `blockDim`，新版叫 `numBlocks`；且该字段在本源码里**只写不读**（第 302 行赋值后无任何引用）。
⇒ 改为 `aicoreParams_.blockDim = aivNum;`，**语义中性**。

### 1.3 用于判定的诊断设施（默认零影响）
`PostTiling()` 开头加了 env 门控硬标记：
* `CAND_A_ABORT=1` ⇒ `OP_LOGE("... [CAND-A-HARDMARKER] vendor tiling IS live ...")` 后 `return ge::GRAPH_FAILED`
* `CAND_A_MARKER=1` ⇒ 只打 ERROR 级日志，不改变行为

---

## 2. 构建链（已打通，含两个非显然的坑）

背景：`moe_init_routing_v3` 是**内置算子**，我们 vendor 里没有它的源码；opensrc 有完整源码。

| 步 | 做法 | 坑 |
|---|---|---|
| 1 | 源码放入 `csrc/moe/moe_init_routing_v3/` | ⚠️ **不能重跑 cmake**：`csrc/moe/` 下有 **20+ 个 `.bak*` 备份目录也带 `CMakeLists.txt`**，而当前 `build.ninja` 里它们出现 **0 次** ⇒ cmake 重配会把垃圾目录收进构建 |
| 2 | **手工编译** tiling .cpp | 复用 `ninja -t commands` 导出的同类算子编译命令（替换源/产物路径）；新头路径 `op_host/*.h` 与容器旧路径 `tiling_base/*.h` 不一致 ⇒ 建 **shim 转发头** 解决 |
| 3 | **手工重链** | `libcust_opmaster_rt2.0.so` 是 `CXX_SHARED_LIBRARY_LINKER` 目标；导出链接命令后把我们的 `.o` 插进对象列表、改输出名 |
| 4 | vendor 组装 | ⚠️ vendor 的 `op_tiling/liboptiling.so` 是**符号链接** → `lib/linux/aarch64/libcust_opmaster_rt2.0.so` ⇒ **两个路径都要替换**，否则可能加载旧库 |

**构建产物校验**（【实测】）：
* 编译：`mo_init_routing_v3_tiling.cpp.o` = 533 KB，`c++` 无 error（仅 warning）
* 链接：新库含 `MoeInitRoutingV3` 符号 **4** 个（原库 **0**），且**仍保留** `HcPre` **5** 个（原 vendor 算子未丢）
* 修复后 md5：`3fd4ca600a7a1403abb37ace37be91e2`（priority=1 版）

---

## 3. 判决实验：vendor tiling **不被采用**

### 3.1 硬判据（env 门控，不依赖日志级别）
| 运行 | `CAND_A_ABORT=1` | 结果 |
|---|---|---|
| A) 基线（不带 vendor） | 是 | 算子**成功**（attempt 1 OK）· HARDMARKER 命中 **0** |
| B) `ASCEND_CUSTOM_OPP_PATH=vendor_candA` | 是 | 算子**仍然成功** · HARDMARKER 命中 **0** |

若我们的 tiling 被采用，B 必须**失败**并打印 `[CAND-A-HARDMARKER]`。
⇒ **B 成功 = 我们的 `PostTiling()` 根本没跑。**

### 3.2 对照组：vendor 库**确实被加载**
【实测】`/proc/self/maps`（同一进程内、带 vendor 运行）：
```
/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/liboptiling.so   ← 内置
/home/l00886679/tmp/moe/vendor_candA/op_impl/ai_core/tbe/op_tiling/liboptiling.so                          ← 我们的（已加载）
/home/l00886679/tmp/moe/vendor_candA/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so
/home/l00886679/tmp/moe/vendor_candA/op_api/lib/libcust_opapi.so
/home/l00886679/tmp/moe/vendor_candA/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so
```
⇒ **不是"没加载"，而是"加载了但注册不生效"。**

### 3.3 已排除的替代解释
| 解释 | 排除方式 | 结果 |
|---|---|---|
| 注册优先级不够（高优先级才胜） | 读机制：`cases_[priority]` 按 priority **升序**试，同 priority 先注册者胜；改用 **priority=1** 重测 | ❌ 仍不生效 |
| tiling 结果被缓存 | 移走 `/root/atc_data/kernel_cache` + `/root/.cache/vllm` 后重测 | ❌ 仍不生效 |
| vendor 未挂上 | maps 对照（§3.2） | ❌ 已挂上 |
| 构建产物不对 | 符号数 4 vs 0、`HcPre` 保留、编译无 error | ❌ 产物正确 |

### 3.4 机制（【实测·代码】）
* 内置 tiling 由 **`libophost_transformer.so`** 提供（该库含 `MoeInitRoutingV3` 符号，且被加载）
* 我们 vendor 的 tiling 走 **`libcust_opmaster_rt2.0.so`**（`op_tiling/`）
* 二者是**两套 tiling 来源**；对**内置算子**，运行时始终采用内置那套
* ⇒ **vendor 只能覆盖"自己声明的算子"，不能改内置算子的 tiling**（至少本 CANN 版本如此）

---

## 4. 为什么这个改动仍值得做（分析结论不受影响）

来自服务 profile（`armF_meta2_1004_2010`，rank0，13400 次 n=6 调用）【实测】：

| n_tok | topk | rows | TaskDur p50 | **aiv_vec_time** | BlockNum |
|---:|---:|---:|---:|---:|---:|
| 5 | 3 | 15 | 11.5 µs | 0.11 | **48** |
| 6 | 6 | **36** | **17.8 µs** | **0.30** | **48** |
| 256 | 6 | 1536 | 23.8 µs | 4.68 | **48** |

* rows ×102 ⇒ 时长仅 ×2.07 ⇒ **固定部分 ≈17.7 µs（≈90%）**，边际 0.004 µs/行
* 真正的向量计算 **0.30 µs = 1.7%**
* `Tiling4VBSCompute()` 早已判定"排序只用 1 个核"（`sortMode_=0` → `Tinlig4VBSOneCoreCompute`），
  但 `PostTiling()` 仍 `SetBlockDim(aivNum)=48` ⇒ **36 行数据发射到 48 个 AIV 块 + 全网格跨核同步**
* 预期收益（若生效）：**240–400 µs/步（0.9–1.5%）**（× 40 层/步）【推断】

---

## 5. 如果一定要做：两条替代路径（均未验证）

| 路径 | 做法 | 改动量 | 风险 |
|---|---|---|---|
| **X. 改镜像内置库** | 把我们的 tiling 合并进内置 `libophost_transformer.so` 的副本，按**镜像层 patch** 覆盖 `/usr/local/Ascend/.../opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/aarch64/libophost_transformer.so` | 中（链接一个 4 MB+ 内置库） | **高**：不再走 vendor 机制，覆盖 CANN 内置文件；须与 `[OPP-OVERRIDE]` 的目录复制语义配合 |
| **Y. 让 vendor "拥有"该算子** | 把该算子的 **OpDef**（`op_graph/moe_init_routing_v3_proto.h`）也编进 vendor 的 `libcust_opsproto_rt2.0.so`，使运行时认为它是自定义算子 ⇒ 才会查 vendor 的 tiling | 中 | **中**：算子身份变化可能影响 **kernel 二进制查找**（当前内核来自内置），存在"注册成功但找不到 kernel"的风险 |

### 5.1 路径 Y 的可行性核查（已完成，【实测】）
| 检查 | 结果 |
|---|---|
| vendor `op_proto/lib/.../libcust_opsproto_rt2.0.so` 是否含 `MoeInitRoutingV3` | **0**（不含） |
| 内置 `op_proto/lib/linux/aarch64/libopsproto.so` 是否含 | **8**（含） |
| vendor 侧 OpDef 的载体 | `op_proto/inc/*_proto.h`（如 `hc_pre_proto.h`） |
| `libcust_opsproto_rt2.0.so` 的构建源 | `ophost_transformer_infer_obj` 目标下的 **`*_infershape.cpp`** 对象 |

⇒ **Y 的具体动作**（与本轮已打通的"手工编译+手工重链"完全同构）：
1. 把 `op_graph/moe_init_routing_v3_proto.h` 放进 vendor 的 `op_proto/inc/`
2. 手工编译 `op_host/moe_init_routing_v3_infershape.cpp`（它 include proto 并注册 OpDef）
3. 手工把它链接进 `libcust_opsproto_rt2.0.so` 的副本（对象库是 `ophost_transformer_infer_obj`）
4. 用**同一个 env 门控硬标记**做判决：`CAND_A_ABORT=1` 时算子应**失败**

**仍存在的不确定性**：即使 OpDef 进了 vendor，运行时的 tiling 优先级**是否真的翻转**仍**未验证** ——
本轮已证明"vendor 库加载了但注册不生效"，而 Y 的假设是"内置算子身份"造成了这一点。
**这个假设本身没有被证据支持**（只是最可能的解释）。

**我的判断**：X 与"vendor 可复现交付"的既定形态冲突，不建议；
**Y 是唯一仍在 vendor 语义内的路径**，且成本可控（一次编译 + 一次链接 + 一次判决测试，与本次同量级）——
但**必须先验证 Y 不会打破 kernel 查找、且真的能翻转 tiling 优先级**，再谈收益。
**建议**：若要继续，就把 Y 当作"一次判决实验"来做（目标不是拿收益，而是回答"vendor 能否覆盖内置算子"这个机制问题）。

---

## 6. 附带的独立发现（值得单独记录）

**`MoeInitRoutingV3` 在进程内的首次调用会失败**：
```
Execution_Error(EZ1009): Failed to execute operator aclnnMoeInitRoutingV3_1_MoeInitRoutingV3.
Reason: The dtype or format of the actual input or output parameter ... inconsistent with ... OpDef
Cannot find binary for op MoeInitRoutingV3.
```
* 【实测】**第 0 次必失败，第 1 次起成功**（`scale_test.py` 里 n=1 失败、n=6 起正常；重试即成功）
* 与 vendor 无关（基线同样如此），也与 `quant_mode` /shape 无关
* 影响：**任何"只调用一次"的探针都会误判为不可用** —— 这条本轮让我白跑了两轮排查

---

## 7. 交付要求（供路径 X/Y 落地时遵守）

1. **`cache/skcache/compile_outputs` 不会自动失效**：换 kernel/tiling 后，服务侧须用
   `V41_OPP_CLEAR_SKCACHE=1` 强制重编译（起服 +18 min），否则可能**静默复用旧产物**。
2. **服务侧"验执行"判据**（不能只看起服成功）：
   * 首选 **`Block Num`**（应在 sortMode_==0 时 48 → 1）或 **`aiv_total_cycles`**（指令/流水计数，抗噪）
   * 次选 `aiv_scalar_time`
3. 本轮的 env 门控硬标记（`CAND_A_ABORT`/`CAND_A_MARKER`）**建议保留在补丁里**：
   它是唯一不依赖日志级别、不依赖 profiler 的"tiling 是否生效"判据。

---

## 8. 卫生

* **原 vendor 未被污染**：`op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so` md5 仍为
  `876a3652b1295693e765aa33065788a7`，符号链接结构原样；aicpu `.so` 仍为 `31f7a0ef…`
* 被测的 vendor 副本在 `~/tmp/moe/vendor_candA`（原 vendor + 我们的 tiling 库）
* 移走的缓存已恢复（`/root/atc_data/kernel_cache`）
* 未碰 `dsv41-tp8k5`/19210；未用 `rm -f`（一律 `mv` 到 `.bak<时间戳>`）
* 工具：`patch_tiling.py`、`compat_fix.py`、`add_hardmarker.py`、`build_tiling.sh`、`link_tiling.sh`、
  `make_vendor.sh`、`hardmarker_test.sh`、`check_maps.py`、`run_marker.py`
