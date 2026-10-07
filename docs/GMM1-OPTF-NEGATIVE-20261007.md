# gmm1 `OPT-F`（单段快路径）E2E 实测：**无可测收益**（2026-10-07）

> ## ⛔⛔ 2026-10-07 22:xx **标题级更正**：**`OPT-F` 就是 `armF` 本身**
>
> 逐文件 diff 显示：`arms/armF/op_kernel/…_a8w4_msd_pipeline.h` **已含**
> `[GMM1-OPT-F]`（line 100）**与** `[GMM1-OPT-A]`（line 205），且含 `loopCount <= 1` 快路径。
> 而**当前源码树与 `arms/armF` 的唯一差异就是"探针块"**（`GMM1-PROBE` 那几个纯标量函数）。
>
> ⇒ **本文测的"OPT-F 臂"= armF + 死探针代码 ⇒ 与 armF 功能等价**，
> 所以 "+0.16%" 是 **armF 对 armF** 的噪声 —— 结论正确（无差异），但**标题与框架是错的**：
> **`OPT-F` 不是"从未测过的新优化"，它就是已交付、已验证 −0.67~−1.09 ms/步的那个 armF。**
>
> **修正后的 gmm1 谱系**：
>
> | 变体 | 内容 | 服务内 |
> |---|---|---|
> | `base` | stock | — |
> | `armC` | base + `OPT-C`（用 group_list 定 M）+ `OPT-A` | ❌ 崩溃 507015（OPT-C 把 M 取成 `x.dim0` 容量 ⇒ 越界） |
> | `armD` | base + `OPT-A`（去掉 OPT-C） | ❌ 设备时间无变化 |
> | **`armF`** | base + **`OPT-F`**（单段快路径 6→2 同步）+ **`OPT-A`** | ✅ **−7.56 µs/op ⇒ −0.67~−1.09 ms/步 ⇒ 已交付** |
>
> ⇒ **`OPT-A` 与 `OPT-F` 都已在交付内核里**；`OPT-C` 是有 bug 的死路。
> **gmm1 kernel 线到此封闭，没有"未测的新优化"遗留。**
>
> **但本文的两个探针实验依然有效且更有价值**：它们证明了
> ①**内核确实被执行**（+227%）；②**armF 的快路径确实被走到**（+113%）。
> 这正是 `GMM1-ARMC-VERDICT` §4 / `GMM1-ARMD-VERDICT` §3 遗留的"两条候选，均未最终确证"的答案。

> 背景：`GroupedMatmulSwigluQuantWeightNzV2` 是**主流上最大的单项之一**
> （40 次/步、2.636 ms、其中**标量 1.410 ms**=53%）。
> 相邻目录的 `csrc/gmm/grouped_matmul_swiglu_quant_v2/` 里已实现一个
> **`[GMM1-OPT-F]` 单段快路径**（decode 的 `loopCount ≤ 1` 时把 6 次跨核同步展平成 2 次、
> 去掉 4 次 `TPipe::Reset`），但**文档里零提及 ⇒ 从未做过 E2E A/B**。
> 本文补上这个 A/B。全部为【实测】。

---

## 0. 结论

| 臂 | model.py | OPP 内核 | bneck p50（n=1608） | vs 基线 |
|---|---|---|---:|---:|
| **A2 基线**（armF，01f600d2） | 交付原版 | 镜像自带 | **25.004** | — |
| **OPT-F**（75bce4ca） | 交付原版 | +OPT-F 内核 | **25.044** | **+0.04 ms（+0.16%）** |

⇒ **在判据下限（≈1%）之下，判为"无可测收益"。**

### 0.1 ★★ 但"无效"这个结论**是可信的** —— 用两个判别实验证明了

本文最初留下了"可能是快路径没走到 / 内核没被执行"的疑问。
**后续用"可调自旋探针"把两个疑问都判定了**：

| 实验 | 探针位置 | 步长（bneck p50） | 判定 |
|---|---|---:|---|
| 基线（armF） | — | **25.0 ms** | — |
| **探针 A** | **函数入口（无条件）**，`SPIN_AIC=100000` | **81.7 ms（+227%）** | ✅ **内核确实被执行** |
| **探针 B** | **只在 OPT-F 快路径内部**，`SPIN_AIC=100000` | **53.1 ms（+113%）** | ✅ **快路径确实被走到** |

**⇒ 两条候选原因都被排除：内核在执行、快路径在走。**
**OPT-F 的"无可测收益"是真实结论，不是"没生效"。**

（探针是纯标量循环、不读写任何张量 ⇒ 数值 bitwise 不变，只影响时序。）

---

## 0.2 ★ 由此确立的一条**通用验证方法**（本仓此前一直缺）

> **"内核 md5 换了" ≠ "服务跑的是新版"。**
> 正确做法：**在目标代码路径里插一个可调自旋（纯标量循环），重建，看步长是否跟涨**。
> * 跟涨 ⇒ 该路径**确实在执行**（此时"无收益"才可信）
> * 不跟涨 ⇒ 该路径**没被执行**（此时要先去查 tiling key / 调通条件）

本次用它一举回答了 armC/armD 文档里**遗留的"未定论"问题**
（`GMM1-ARMC-VERDICT` §4 与 `GMM1-ARMD-VERDICT` §3 都写"两条候选，均未最终确证"）。

**实测的探针标定**：`SPIN_AIC=100000` ⇒ 无条件插桩使步长 +56.7 ms；
按 43 次/步折算 ≈ **1.32 ms/次** ⇒ **每 1000 次自旋 ≈ 13.2 µs**。
（副产品：这也说明**原源码里 `SPIN_AIC` 的默认值 30300 会白送 ≈ 0.4 ms/次、≈17 ms/步**——
若有人用那个默认值构建，性能会离奇崩掉。本次已显式归零。）

---

**内核确实被换掉了**（这不是"没挂上"）：

| 核查 | 结果 |
|---|---|
| 宿主产物 md5 | `75bce4ca24545e0ddf7bef847770a1c5`（≠ armF `01f600d2…`） |
| **容器内实际加载** | 同 `75bce4ca…` ✅ |
| `[OPP-OVERRIDE] copied -> …` | ✅ 出现 |
| KV 容量 | 2,987,618（基线区间内）✅ |
| 冒烟 | `'1024'` ✅ |

---

## 1. 为什么它可能"本该有效却没效"（两条候选，**均未验证**）

### 1.1 快路径可能没被走到

```cpp
// grouped_matmul_swiglu_quant_v2_a8w4_msd_pipeline.h
workspaceSplitConfig.loopCount = Ceil(workspaceSplitConfig.M, gmmSwigluQuantV2BaseParams->mLimit);
...
if (workspaceSplitConfig.loopCount <= 1) {   // ← OPT-F 的入口条件
```

只有当 **`M ≤ mLimit`** 时 `loopCount = 1`，快路径才生效。
而 `M` 的取法**正是 armC 的翻车点**（`armC` 误取 `x.dim0` 容量 ⇒ 越界 507015；
正确取法是 `group_list` 前缀和 = **真实行数**）。

⇒ 若服务侧传入的 `M` 是**图模式的 padded 容量**（而非真实 6 行），则 `loopCount > 1`、
**OPT-F 被完全跳过**。这与"md5 换了、E2E 却没变"的观测一致。

### 1.2 服务可能不走这个 tiling key

该 op 在 vendor 下有 **14 个 `.o`**（不同 tiling key）。`armD` 的文档已记录过同类问题：
*"服务走的是另一个 tiling key 的 kernel 二进制"*。
本次同时替换了 **2 个 tiling key**（`fa3d6d3d` 与 `05bce438`，与 armF 一致的两个），
但**没有验证服务实际命中的是哪一个**。

---

## 2. ★ 本轮产出的可复用资产（比结论本身更有价值）

### 2.1 v2 版重建脚本

```bash
# ~/tmp/gmm1/rebuild_gmm_v2_op.sh（由 v1 版 sed 而来）
bash ~/tmp/gmm1/rebuild_gmm_v2_op.sh <源目录> <tag>
#   ← OPDIR 指向 csrc/gmm/grouped_matmul_swiglu_quant_v2（**v1 版脚本错指向无 _v2 的目录**）
#   实测：op_count=47、build+install 约 8 分钟，产出 ~/tmp/gmm1/opstage<tag>/
```

**踩过的坑（已修）**：`~/tmp/gmm1/rebuild_gmm_op.sh` 的 `OPDIR` 指向
`csrc/gmm/grouped_matmul_swiglu_quant`（**v1**），而服务用的是 **v2**
（`GroupedMatmulSwigluQuantWeightNzV2`，源码在 `..._v2/`）。
这解释了为什么 armC/armD 的算子级收益"放进服务就没动静"——**很可能建到了另一个 op 上**。

### 2.2 "最小 OPP 包"构造法（避免误伤其它算子）

`opp_override_block.sh` 是 **`cp -a` 覆盖**整个 vendor，所以包里的每个文件都会生效。
正确做法：**镜像 vendor 逐字节 + 只替换目标 kernel 的 `.o/.json`**。

```bash
# 1) 从运行中的容器导出镜像 vendor（1136 个文件）
docker exec dsv41-tp8k5 bash -lc "cd $V && tar cf - ." > imgvendor.tar
# 2) 只覆盖目标 .o/.json（注意：产物是 root 属主，必须 sudo）
sudo -n cp -a <new>/$OP/GroupedMatmulSwigluQuantV2_<hash>{.o,.json} $D/$OP/
# 3) 校验：与镜像逐文件 md5 比对，**应恰好只有 4 行差异**（2 .o + 2 .json）
```

**实测**：本轮的包 1136 个文件，与镜像**恰好 4 个不同** ✅

### 2.3 内核身份指纹

源码里的 `GMM1_PROBE_MAGIC 0x5043765C` 是**恒假分支**（只把常量留在二进制里），
可作为"内核是否真的是这一版"的指纹；**同时**它的 `GMM1_PROBE_SPIN_AIC` 默认曾是
**30300**（诊断用的自旋），**必须显式归零**再构建，否则会白送一个自旋开销。

---

## 3. 这条线的历史与现状（避免重复投入）

| 变体 | 算子级 | 服务内 | 结论 |
|---|---|---|---|
| **armC** | −4.95 µs/op（12 轮配对） | ❌ 加载/启动失败（三种方式） | 不采纳 |
| **armD** | −3.3 µs/op | ❌ **设备时间没变**（73.64→72.98） | 不采纳 |
| **armF** | **−7.56 µs/op** | ✅ **−0.67~−1.09 ms/步（≈3%）** | ✅ **已转交付** |
| **armG / slow\* / sv\*** | — | — | 未见采纳记录 |
| **OPT-F（本轮）** | 未做算子级 | ❌ **+0.16%（判据下限内）** | ❌ 不采纳 |

⇒ **gmm1 这条 kernel 线在"跨核同步削减"方向上已试尽**：
armF（SyncAll 4→2）是唯一生效的；armC/armD/OPT-F 都在服务内无效。
**根因很可能是"改的 kernel 与服务的 tiling key 不匹配"**，
而**验证手段（先确认命中的是哪个 `.o`）此前一直没做**。

---

## 4. 若要把这条线走通，需要补的一步

**先确定服务实际命中哪个 tiling key**，再改那一个。三条可用手段：

1. `binary_info_config.json` + `relocatable_kernel_info_config.json` 里查该 op 的
   kernel 索引与 tiling key 映射；
2. **单算子 profile**：用真实 decode 参数（M=6、E=48、type=A8W4_MSD）跑一次，
   看 msprof 里该算子的 `Duration` 是否随"只改一个 .o"而变化 —— **这是最快的判别**；
3. `op_summary` 的 `Block Num` / `aic_scalar_time` 指纹（armF 用的是 `_2_mix_aic = 37328`）。

> ⚠️ 本轮的**最大教训**：**"内核 md5 换了" ≠ "服务跑的是新版"**。
> 必须用**设备侧指纹**（耗时/资源计数）确认，否则会得出"优化无效"的假结论。

---

## 5. 复现

```bash
# 构建（约 8 分钟）
ssh a3-21 'bash ~/tmp/gmm1/rebuild_gmm_v2_op.sh ~/tmp/gmm1/optF_src optF'
#   → ~/tmp/gmm1/opstageoptF/（内核 75bce4ca…）

# 造最小包（镜像 vendor + 4 个文件）
ssh a3-21 'bash ~/tmp/gmm1/mk_min_pkg.sh'   # 见 §2.2 的三步；本轮手工执行

# 起服（处理臂）
ssh a3-21 '... V41_HC_OPP_PKG=$HOME/tmp/gmm1/opp_min_optF bash scripts/serve_a2.sh'

# 验证内核确实加载
ssh a3-21 'docker exec dsv41-tp8k5 md5sum <容器内路径>/GroupedMatmulSwigluQuantV2_fa3d6d3d….o'

# 测量（主判据：bneck p50，去掉首窗）
ssh a3-21 'python3 ~/tmp/clean_probe.py http://127.0.0.1:19210 30 3'
```

---

## 6. 环境状态（本轮结束）

| 项 | 值 |
|---|---|
| tp8k5 | ✅ 已恢复（RUN `armRESTORE6_1007_074954`） |
| 内核 | **armF（01f600d2）** — 容器内实测确认已回退 |
| 无 OPP 覆盖 | `OPP-OVERRIDE` 计数 = 0 ✅ |
| KV 容量 | **2,987,836**（基线区间内） |
| 配置 | `max_seqs=32` / `bat=8192` / `sptok=5` ✅ |
| 冒烟 | `'243'` ✅ |
| **新增可复用资产** | `~/tmp/gmm1/rebuild_gmm_v2_op.sh`（v2 重建，含 OPDEF 语义）<br>`~/tmp/gmm1/opp_min_{optF,probe,fpprobe}/`（三个最小包，各 1136 文件）<br>`~/tmp/gmm1/{optF,probe,fpprobe}_src/`（三份源码）<br>**探针判别法**（§0.2） |
| 本轮起服次数 | **6 次**（OPT-F / probe / fpprobe / 恢复×3） |
