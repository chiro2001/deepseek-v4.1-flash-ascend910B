# 自己修算子：重编链路打通 + 首轮补丁结果（2026-09-30 21:20–21:35）

> 本轮目标：不依赖算子团队，自己改 CANN 的 `SparseFlashMla`（arch22 CSA 模板）修
> DCP8 的 NaN / 非确定。本文记录**已打通的链路**、**被证伪的路径**与**首轮补丁结果**。

---

## 1. ★ 用 CANNBot Skills 独立复核根因（已命中）

参考 `~/projects/vllm/model-comparing/cannbot-skills`（CANN 官方技能库，260 个 skill）。
最有用的三个：`ops/ascendc-sync-audit`（带两个静态分析器）、
`ops/ascendc-precision-debug`（症状-原因决策树）、`ops/ascendc-direct-invoke-template`。

在容器内的 CSA 三个文件上跑 `sync_audit.py`（158 个同步事件、**41 条发现**）：

| 条例 | 严重级别 | 条数 | 是否命中我们的病灶 |
|---|---|---|---|
| **SYNC-08** | **红线** | **3** | **★ 命中**：`csa_block_vector.h` 的 411 / **642** / **694** 行 |
| SYNC-14 | 红线 | 14 | buffer 索引与 flag 索引不一致（`kvMergeGm_` 用 `info`、flag 用 `kb`） |
| SYNC-04 | 红线 | 6 | Set/Wait 个数不一致 |
| SYNC-01 | 高 | 12 | Wait 先于 Set |
| SYNC-05 / 09 / 12 / 02 | 高/性能 | 1–2 | — |

**SYNC-08（提前 return/break 跳过 SetFlag）的三条**，与我们独立定位的根因**一一对应**：

```
(2) sparse_flash_mla_csa_block_vector.h:642 [红线]
    SetFlag 前存在提前退出（可能跳过 Set → 死锁）: SetFlag<HardEvent::MTE2_MTE3>
    detail: 同函数 CopyOutMrgeResult 内未被兜底的 return/break: [640]
    → 即 `if (mte2Size <= mte3Size) { return; }`

(3) sparse_flash_mla_csa_block_vector.h:694 [红线]
    SetFlag 前存在提前退出: SetFlag<HardEvent::MTE3_MTE2>
    detail: 同函数 ProcessVec0L 内未被兜底的 return/break: [686]
    → 即"索引 < 0（被丢弃）→ CopyOutMrgeResult + SetFlag + break"
```

⇒ **两套独立方法（我们的离线核算 + 官方静态审计）指向同一处代码。**

## 2. ★★ 被证伪：`opc --input_param` 根本不会编译我们的源码

**证据 A**：把源码目录**整个移走**后，`opc` 仍 `EXIT=0` 并产出 `.o/.json`。
**证据 B**：在 `op_kernel/arch22/sparse_flash_mla_csa_kernel.h` **第 1 行注入 `#error`**，
构建仍成功、md5 不变。
**证据 C**：连续三次"重编"（未打补丁 / 打补丁 / 打补丁+清 `/root/atc_data/kernel_cache`）
产出**逐字节相同**的 `.o`（`d2c6c3e092b6ab5f34d451caa3a23d58`）。

⇒ `opc --input_param=<param.json>` 走的是**注册表编译**（`opc.py:754 load_op_info_store`），
源码来自 CANN 注册的实现，**与传入的源目录无关**。
这正是 `ascendc-precision-debug` 的**第 0 步**陷阱：
「修改代码后输出完全不变 ⇒ 二进制未更新 / 编译器缓存」。

## 3. ★★★ 真正可用的重编链路（已跑通并验证生效）

```
① 改源码：  /vllm-workspace/vllm-ascend/csrc/attention/sparse_flash_mla/op_kernel/arch22/*.h
② 强制重新拷贝源码 + 重编内核（两个 .done 标记必须移走，否则 ninja 认为已完成）：
     mv csrc/build/binary/ascend910_93/src/sparse_flash_mla/sparse_flash_mla_ascend910_93_src_copy.done  <bak>
     mv csrc/build/binary/ascend910_93/gen/sparse_flash_mla_ascend910_93_0.done                        <bak>
③ cd csrc/build && ninja sparse_flash_mla_ascend910_93_0
④ 新内核出现在：csrc/build/binary/ascend910_93/bin/sparse_flash_mla/
```

**生效判据（关键！）**：`.o` 的 md5 **必须变化**。

| 阶段 | `.o` md5 | mtime |
|---|---|---|
| 生产（镜像内 vendor，未改） | `034360db79e65ead1ba16f380b81818d` | Sep 11 01:27 |
| 打补丁 + 移走两个 `.done` 后重编 | **`7b3146af5c84bdef5612ac93e5dd68a4`** | **Sep 30 13:24** |

**另外两个前置修复**（否则 ninja 跑不起来）：
* `symbol.cmake:253` 报 5 个 `*_metadata_obj` 目标不存在
  ⇒ `build.sh --ops` 必须**一并带上它们**：
  `sparse_flash_mla,sparse_flash_mla_metadata,quant_lightning_indexer_v2_metadata,sparse_attn_sharedkv_metadata,store_kv_block_metadata,vllm_quant_lightning_indexer_metadata`
* `ninja` 重配置时去找已失效的 `/tmp/pip-build-env-*/…/cmake`
  ⇒ 把该路径软链到当前 cmake（`/usr/local/python3.12.13/bin/cmake`）

**产物位置确认**：`csrc/build/binary/ascend910_93/bin/sparse_flash_mla/` 的 `.o/.json`
与 **vendor 安装目录逐字节相同**（`034360db…` / `ac4ce41a…`）
⇒ 这一处就是"要覆盖安装的那个文件"。

**验证方式（不动生产）**：把新内核目录以 `-v ...:ro` 挂载覆盖到
`.../tbe/kernel/ascend910_93/sparse_flash_mla`，再跑 `probes/replay_dump.py`。

## 4. 首轮补丁（v1）与结果

补丁内容（`docs/patches/dcp8-v1-*.patch`）：

| # | 文件 | 改动 |
|---|---|---|
| 1 | `sparse_flash_mla_csa_kernel.h` `GetSparseActualSeqLen` | 去掉用**全局坐标**推导的 `thresHold` 截断，`bound = min(actCmpS2Size, sparseBlockCount)` |
| 2 | `sparse_flash_mla_csa_block_vector.h` `GetKeyGmOffset` | 去掉 `realS2Idx >= s2IdLimit` 的丢弃，只保留 `< 0` 与 topk 容量两道硬保护 |

**结果：单卡 dump 重放崩溃**（`aicore exception` / `rtDeviceSynchronizeWithTimeout`）。

**原因（已定位）**：**元数据算子（AICPU）独立计算同一个 `actCmpS2Size`** ——
`sparse_flash_mla_metadata/op_kernel_aicpu/sparse_flash_mla_metadata_aicpu.cpp:959`：

```cpp
uint64_t cmpS2FirstToken = (cmpRevertS2FirstToken + 1) / cmpRatio_ - 1U;
uint64_t cmpS2LastToken  = (cmpRevertS2LastToken  + 1) / cmpRatio_ - 1U;
s1GCache.actCmpS2Size = isSparseCmpKv_ ?
        std::min((uint32_t)(cmpS2LastToken - cmpS2FirstToken + 1), cmpTopkSize) :
        (uint32_t)(cmpS2LastToken - cmpS2FirstToken + 1);
```

**只改内核、不改元数据 ⇒ 两者的任务切分不一致 ⇒ 越界 → 崩溃。**
⇒ 下轮必须**成对修改**（内核 + AICPU 元数据），或改选**不改变调度**的窄补丁。

## 5. 下轮的两个候选（按风险升序）

**候选 A（窄、不动调度）**：只让**计数与 gather 一致** ——
把 `CountValidCmpSparseLen` 的判据改成"与 `GetKeyGmOffset` 同源"（即只数
`realS2Idx < s2IdLimit` 的项）。`bound` 不变 ⇒ 元数据调度不变 ⇒ 不会崩。
预期：NaN 消失（洞被补上），但 DCP8 的答案仍可能因丢键而不对。

**候选 B（根治、需成对改）**：内核取消全局界 + AICPU 元数据同步改为
"按调用方给出的有效个数/本地坐标"计算 `actCmpS2Size`。
预期：既无 NaN 又正确；风险是 AICPU 侧的联调成本。

## 6. 其他已就绪的工具与事实

* **`npusim` 仿真器只支持 Ascend 950**（`ops-simulator` skill 明确写"仅支持 Ascend 950 芯片架构"）⇒
  **A3 上不可用**，只能上真卡。
* `opc` 有 `--deterministic` 开关（编译确定性算子），后续可评估。
* 生产状态：服务健康（`19210: 200`），**vendor 内核从未被改动**（仍是 `034360db…`），
  所有实验都在临时目录 / 挂载覆盖下进行。
* 备份：`/tmp/arch22_bak.tar.gz`（仓库源码）、`/tmp/kbak`（vendor 内核）、
  `/tmp/kfix/base_op_kernel`（基线源码树）。

---

# 附录 B：四个单变量变体的实测结果（2026-09-30 21:30–21:45）

## B1. 重编链路已可稳定产出（4 个变体，全部 md5 变化）

| 变体 | 改动 | kernel `.o` md5 | 单卡重放结果 |
|---|---|---|---|
| 基线 | — | `034360db79e65ead1ba16f380b81818d` | NaN **4806** |
| **A** | 只去掉 `thresHold` 截断（`bound = min(actCmpS2Size, 512)`） | `9f0e6906cb84fe1970a029d400757253` | 不崩，NaN **40704**（更差） |
| **B** | 只去掉 `GetKeyGmOffset` 的 `s2IdLimit` 过滤（上限放到 `sparseBlockCount=512`） | `e913816e1457dd7a6fb9c33579fc4680` | **崩溃**（aicore exception） |
| **C** | 只把 `CountValidCmpSparseLen` 改成与 gather 同判据 | `7d5c4d97837b8e0bd0b625b49e08a688` | **与基线完全相同**（NaN 4806/5252） |
| **E** | `bound = 本 rank 缓存长度` + `cmpS2IdLimit = 本 rank 缓存长度`（即**完全不丢键**） | `d29b76aa9d281e4ebc0e9b6cc3ff8163` | 不崩，NaN **40704** |

补丁留在 `docs/patches/dcp8-variant{A,B,C,E}.patch`。

## B2. ★ 决定性否定结果：NaN **不是**丢键造成的

变体 E 已经把 `cmpS2IdLimit` 设成本 rank 缓存长度（128），而我们的索引值域是 `[0,127]`
⇒ **一个键都不会被丢弃**（这可以直接从改动推出）。但 NaN 仍是 **40704**，
与变体 A（仍会丢键）**逐次完全相同**。

⇒ **"丢键 → 未写洞 → 读残留"这条因果链被实测否定。** 我们在
`V41-CSA-KERNEL-SOURCE-ANALYSIS-20260930.md` 里给出的 `[780,892]` vs `[800,903]`
区间重叠只是**相关性**，不是机理。

## B3. NaN 数量与「内核声明的 `actCmpS2Size`」强相关

| 内核声明的 `actCmpS2Size` | NaN |
|---|---|
| 被 `thresHold` 压小（基线） | **4806** |
| 放宽到本 rank 缓存长度（A / E） | **40704** |

NaN **只随声明条数变化**，与是否丢键无关 ⇒ 真正的不匹配在
**内核声明的 `actCmpS2Size`** 与 **AICPU 元数据算子按自己的公式算出的调度范围**
之间：内核声明得越多，越是读到元数据没分配的区域 ⇒ NaN 越多。

## B4. 另一个关键细节（解释了变体 C 为何"无变化"）

`CountValidCmpSparseLen` 用 `base = (actualSeqQPrefixSum + s1StartIdx) * …`
—— 只扫**该 s1 块的起始行**；而 gather 是**逐行**扫。
所以"按行丢键"根本不会反映到这个计数里 ⇒ 变体 C 改了等于没改（实测 NaN 逐次相同）。

## B5. 下轮要做的（唯一剩下的方向）

**内核 + AICPU 元数据成对修改**，让两者对 `actCmpS2Size` 的含义一致。

* 内核侧：变体 E 已经是对的（`localCmpLen` 作为 bound 与 limit）。
* 元数据侧：`sparse_flash_mla_metadata/op_kernel_aicpu/sparse_flash_mla_metadata_aicpu.cpp:959`
  目前用 `cmpS2LastToken − cmpS2FirstToken + 1`（由 s1 窗口与全局坐标推导）；
  分片场景下必须改为**按调用方给出的本 rank 有效个数**（`seqused_cmp_kv` 语义）。
* 注意：元数据是 **AICPU 算子**，产物是 `libcust_aicpu_kernels.so`（`build.sh` 第 21 步），
  与主算子内核是两条不同的构建/安装路径。
* 风险提示：变体 B 证明"把上限放大到 `sparseBlockCount`"会**越界崩溃**
  （本 rank 缓存只有 128 行）⇒ 任何放宽都必须以"本 rank 真实长度"为上界。

## B6. 现场状态（安全）

* 仓库源码、构建副本、构建产物 `.o` **全部已恢复基线**（`034360db…`，与 vendor 逐字节相同）。
* vendor 生产内核**始终未被改动**；服务 `health=200`。
* 4 个变体的 `.o` 保留在 a3-21 的 `/tmp/kfixk_{va,vb,vc,ve}/`（用于复核）。
* 备份：`/tmp/arch22_bak.tar.gz`、`/tmp/kbak`、`/tmp/kfix/base_op_kernel`。
