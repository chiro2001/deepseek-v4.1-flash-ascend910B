# HcPre 内核审计 + 暴露度探针：一个未发布的优化、一个未解的异常（2026-10-07）

> 动机：`HcPre` 是主流（关键路径）上**最大的单项**：80 次/步、**2.666 ms**，而**数学只有 0.167 ms**。
> 它是"单流 −12%"最有希望的单一目标（若能把 2.5 ms 开销砍掉一半 ⇒ −5%）。
> 本文记录：①源码保真度审计；②一个**未发布的优化**；③暴露度探针的**不确定结果与异常**。
> 全部为【实测】。

---

## 0. 三条结论

| # | 结论 |
|---:|---|
| 1 | **容器 `csrc/moe/hc_pre` 与镜像内核只差一个文件**：`hc_pre_cube_compute.h`（**非 arch35** 分支），其中含两个**未发布的优化** `[KADAPT]` / `[HF32-HOIST]` |
| 2 | **从 csrc 重建的 hc_pre 与镜像性能等价**（bneck p50 **24.996** vs 镜像 **25.004**，n=1888），且**全部正确性门通过** |
| 3 | ⚠️ **暴露度探针结果不确定**：加 spin 后服务步长 +0.326 ms（+1.30%），但**单算子 bench 显示内核时长完全没变**（两者都 34.20 µs）⇒ 无法判定"改动是否真被执行" |

---

## 1. 源码保真度审计（镜像 vendor 内嵌源码 vs 容器 csrc）

镜像的 vendor 里**内嵌了构建时用的源码**（`.../custom_transformer_impl/ascendc/hc_pre/`，8 个文件）。
与容器 `csrc/moe/hc_pre` 逐文件比对：

| 文件 | 结果 |
|---|---|
| `hc_pre.cpp` / `hc_pre_base.h` / `hc_pre_base_arch35.h` | ✅ 同 |
| `hc_pre_cube_compute_arch35.h` | ✅ 同 |
| `hc_pre_m_k_split_core.h` / `..._arch35.h` / `hc_pre_m_split_core_arch35.h` | ✅ 同 |
| **`hc_pre_cube_compute.h`** | ⚠️ **异** |

### 1.1 那个差异里是什么（**未发布的优化**）

```diff
+    // [KADAPT] 按 M 自适应选 L0 的 K 步长：L0A 每缓冲 32KB = 8192 floats，
+    // 需 CeilAlign(curML1,16) * kL0 <= 8192 ⇒ M<=64 用 128、M<=128 用 64、其余 32。
+    // 三档都能整除 kL1Size（kL1Size 恒为 128 的倍数），不改变累加顺序。
+    // [HF32-HOIST] 整个 K 循环里 Mmad 都用同一模式：设一次，循环结束再复位
+    // （原来每轮 MmadAB 内部设 1 再复位 0 ⇒ 32/8 轮重复的 scalar 开销）
+    AscendC::SetHF32Mode(1);
+    AscendC::SetHF32TransMode(1);
+    const uint64_t mAligned_ = CeilAlign(mmParams.curML1, BLOCK_CUBE);
+    const uint64_t kL0Size_ = (mAligned_ <= 64UL) ? 128UL : ((mAligned_ <= 128UL) ? 64UL : 32UL);
...
-        for (...; kL1Offset += K_L0_SIZE) {   // 固定 32
+        for (...; kL1Offset += kL0Size_) {
```

两项都很合理（前者减少 K 循环轮数，后者去掉每轮重复的 scalar 设置），**且作者声明"不改变累加顺序"**。

### 1.2 但它们**对我们的形状无效**（算术）

| 项 | 分析 |
|---|---|
| `[KADAPT]` | M=6 ⇒ `mAligned = CeilAlign(6,16) = 16 ≤ 64` ⇒ `kL0Size_ = 128`。而 K 循环轮数 = `ceil(curKL1Size / kL0)`；我们的 K 很小（`hc_fn` 首维 = 24），**32 与 128 都只跑 1 轮** ⇒ **无差异** |
| `[HF32-HOIST]` | 只有 1 次 Mmad ⇒ 没有"32/8 轮重复设置"可省 ⇒ **无差异** |

⇒ 与实测一致：**重建版 ≈ 镜像版**（−0.03%）。

### 1.3 ⚠️ 而且它可能**根本没被编译**

| 证据 | 值 |
|---|---|
| `hc_pre.cpp` 的分支 | `#if defined(__DAV_C310__)` → arch35 文件；`#else` → **`hc_pre_cube_compute.h` 等非 arch35** |
| **`hc_pre_cube_compute_arch35.h` 里 KADAPT/HF32-HOIST 的出现次数** | **0** |
| 构建日志里 arch35 出现次数 | 55 |

⇒ **若 A3 走 arch35 分支，这两项优化是死代码**（改在了不会被编译的文件里）。
需 (a) 确认 A3 的宏，或 (b) 把优化移植到 arch35 文件。

---

## 2. 重建版的功能验证（全部通过）

从 csrc 重建 hc_pre（内核 md5 `0e4c3e5c`）并挂载后：

| 门 | 结果 |
|---|---|
| 快速冒烟（3 题） | `391` / `北京` / `The weather is very nice today.` ✅ |
| **结构化任务（10 轮 × 3 类）** | **抽取 / 事实 / 算术 三类均 10/10 逐字一致且 100% 正确** ✅ |
| **长文针 144K** | **4/4 PASS**（ZQ7K-3341 / VX2M-8890 / HT4P-5527 / RB9N-6014）✅ |
| **长文针 1M** | **3/3 PASS**（A/B/C）✅ |
| KV 容量 | 2,987,618（基线区间内）✅ |

⇒ **csrc 的 hc_pre 源码功能正确，可以安全地作为后续 kernel 改造的基线。**

---

## 3. ⚠️ 暴露度探针：结果不确定（含一个未解异常）

### 3.1 方法

在 `hc_pre.cpp` 入口插**可调自旋**（纯标量、不改数值，`ASCEND_IS_AIC` 保护），构建两臂：

| 臂 | 内核 md5 | 说明 |
|---|---|---|
| **ctl0** | `0e4c3e5c` | SPIN=0（对照） |
| **spin2000** | `d48e8551` | SPIN=2000 |

### 3.2 服务侧结果

| 臂 | bneck p50 | n | vs ctl0 |
|---|---:|---:|---:|
| ctl0 | **24.996** | 1888 | — |
| spin2000 | **25.322** | 1360 | **+0.326 ms（+1.30%）** |

（参考：镜像基线 25.004 / 25.044 / 25.099；run 间漂移实测 ≈0.2 ms）

### 3.3 ⚠️ 但单算子 bench 说"内核时长没变"（**异常**）

把两个内核分别 `docker cp` 进 `dsv41-op-hcfuse` 的 vendor 路径（**md5 已核对**），
用 `torch_npu.profiler` 取 `npu_hc_pre_v2` 的设备时长（M=6 真实形状，60 次中位）：

| 内核 | **设备中位 µs** |
|---|---:|
| ctl0（SPIN=0） | **34.20** |
| **spin2000** | **34.20** |

**两者完全相同** ⇒ 说明该 bench **没有看到 spin 的影响**。

（已排除"bench 不读 OPP"：把 `ASCEND_CUSTOM_OPP_PATH` 设为不存在的路径时，bench 会报
`aclnnHcPre ... not in libopapi.so` ⇒ 它**确实**从该路径解析算子。）

### 3.4 两种解释，**未区分**

| # | 解释 | 含义 |
|---:|---|---|
| **1** | spin **确实生效** ⇒ 服务侧 +0.326 ms 是真实效应 ⇒ **暴露度 ≈15%**（按 gmm1 标定 spin≈26 µs/次：26×86 = 2.24 ms 预期，实测 0.326） | **HcPre 优化只能拿回 ~0.4 ms** ⇒ 不值得做 kernel |
| **2** | spin **没生效**（单算子 bench 的证据），服务 +0.326 ms 是 **run 间噪声**（漂移 0.2 ms 的 1.6 倍） | HcPre 是否暴露**仍未测** |

### 3.5 为什么"加大 spin"解决不了

服务侧能测到的只是 `d(step)/d(spin_iterations)`（= 0.326 ms / 2000 = 0.163 ns/iter）。
要得到"暴露度"必须知道 `d(kernel_time)/d(spin_iterations)`。
**加大 spin 只会把前者按比例放大，不会提供后者。**

---

## 4. ★ 方法论补充（本仓已有证据的第三次印证）

> **"内核 md5 换了" ≠ "服务跑的是新版" ≠ "改动生效"。**

| 层次 | 判据 | 本轮 |
|---|---|---|
| ① 文件被换 | `docker exec md5sum` | ✅ 已确认 |
| ② 内核被加载 | 单算子 bench 或副作用探针 | ❌ **未确认**（bench 与预期矛盾） |
| ③ 改动反映到时间 | 服务侧 A/B | ⚠️ +1.30%，在噪声边缘 |

**本轮在 ② 上失败**，导致 ③ 不可解释。**下一轮必须先补 ②**：给探针加可观测副作用。

---

## 5. 可复用资产（本轮）

| 资产 | 路径 |
|---|---|
| **hc_pre 重建脚本** | `~/tmp/hcpre/rebuild_hcpre_op.sh`（`op_count=47`，约 8 min） |
| ctl0 / spin2000 最小包 | `~/tmp/hcpre/opp_min_{ctl0,spin2000}/`（各 1136 文件，只 4 个与镜像不同） |
| 两份源码 | `~/tmp/hcpre_probe/`（SPIN=0）、`~/tmp/hcpre/spin2000_src/` |
| **镜像内嵌源码** | `~/tmp/hcpre_img_src/`（镜像构建时的 8 个源文件，用于保真度比对） |
| 单算子 bench | `tools/bench_hcpre_limitcore_prof.py` |

---

## 6. 下一步（若继续 HcPre 线）

| 序 | 动作 | 目的 |
|---:|---|---|
| **1** | **给探针加可观测副作用**（把 spin 计数写进 workspace 的保留位置），重建 → 服务侧读回 | **确认内核是否被执行**（补上 ②） |
| 2 | 确认 A3 的 `__DAV_C310__` 宏 | 判定 `[KADAPT]`/`[HF32-HOIST]` 是死代码还是要移植 |
| 3 | 两臂各采一份短 profile（5 s）导出，比较 `HcPre` 的 `Duration` | 直接量"暴露度" |
| 4 | 只有 3 证明暴露度 ≥50%，才值得投入 HcPre kernel 优化 | 避免数周无效工作 |

---

## 7. 复现

```bash
# 保真度审计（镜像内嵌源码 vs csrc）
ssh a3-21 'docker cp dsv41-tp8k5:<vendor>/custom_transformer_impl/ascendc/hc_pre /tmp/hcpre_img_src
  docker cp /tmp/hcpre_img_src dsv41-op-hcfuse:/tmp/hcpre_img_src
  docker exec dsv41-op-hcfuse bash -lc "cd /tmp/hcpre_img_src && for f in *; do
    diff -q \$f /vllm-workspace/vllm-ascend/csrc/moe/hc_pre/op_kernel/\$f; done"'

# 重建 + 挂载 + 测量
ssh a3-21 'bash ~/tmp/hcpre/rebuild_hcpre_op.sh ~/tmp/hcpre/spin2000_src spin2000'
ssh a3-21 'V41_HC_OPP_PKG=$HOME/tmp/hcpre/opp_min_spin2000 bash scripts/serve_a2.sh'
ssh a3-21 'python3 ~/tmp/clean_probe.py http://127.0.0.1:19210 90 1'

# 单算子 bench（在两个 vendor 之间切换）
ssh a3-21 'docker cp <pkg>/.../HcPre_*.o dsv41-op-hcfuse:<容器内 hc_pre 路径>/
  docker exec -e ASCEND_CUSTOM_OPP_PATH=<容器内 vendor> dsv41-op-hcfuse bash -lc "cd /tmp && python3 hc_limit_prof.py"'
```
