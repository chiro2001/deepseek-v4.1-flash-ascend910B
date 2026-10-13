# gmm1 armC 自定义 kernel：算子级有效、**但容器内无法交付**（负结果，2026-10-04）

> 结论：`GroupedMatmulSwigluQuantV2` 的 armC kernel（省掉约 144 次串行 group_list GM 读）
> **算子级 12 轮配对 A/B 中位 −4.95 µs（−6.0%）、输出 bitwise 相等**，但把它真正放进
> 容器的 kernel 加载路径后，**服务启动即失败**（两种加载方式各失败一次）⇒ **不采纳进交付**。

## 1. 算子级证据（子代理，可信）

| 项 | 结果 |
|---|---|
| 单算子时长 | 79.25 → 74.50 µs（−4.75 µs，**−6.0%**） |
| `aic_scalar` 分解 | 34.4 → 26.4 µs（−8 µs，≈144 次读 × 56 ns） |
| 配对 A/B | **12 轮交替，配对差中位 −4.95 µs，11/12 为负** |
| 正确性 | `y`/`y_scale` **bitwise 相等**；prefill（M=2520）逐位相等且 467.3→462.3 µs |
| 重编链 | zero-change 重编 `.o` 与镜像预置**逐字节一致** ⇒ 工具链无差异，差异 100% 来自补丁 |
| 其余负结果 | `group_list_type` 1→0 无收益；baseN 512 被 L0C 上限拒绝；baseM64+baseN512 更慢（85.8 µs） |

## 2. 容器交付失败（实测，两次独立）

| # | 加载方式 | 结果 |
|---|---|---|
| 1 | `ASCEND_CUSTOM_OPP_PATH=<vendor>`（只设 env） | **内核根本没被用**：profile 逐字段等于 stock（73.64 µs / aic_scalar 30.79） |
| 2 | `-v <vendor>:<镜像vendor路径>:ro` 覆盖挂载 | 启动期 **aicore exception**：`IndexCheck` 507015 / `KernelLaunch failed` |
| 3 | 起服前 `cp -a` 覆盖镜像 vendor 路径（rw） | 内核确实进容器（md5 = armC），但服务**挂死**：`No available shared memory broadcast block found in 60 seconds` 连续 5 分钟无进展 |

## 3. 为什么方式 1 不生效（已定位，值得留档）

`vllm_ascend/utils.py:323-332` 的 `bootstrap_custom_op_env()`：
```python
vendor_path = <镜像内>_cann_ops_custom/vendors/custom_transformer
_prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_path)   # ← 前插
```
⇒ 镜像自带 vendor 路径被**前插**到我们设的值之前 ⇒ **同名 kernel 以镜像内的为准**。
（`hc_pre_norm` 那类**新算子**不受影响，因为镜像里没有同名项 —— 这解释了为什么
融合算子实验能生效、而"改已有算子 kernel"不能。）

**修法（本次已实现并验证可注入）**：起服前把自定义 vendor **复制覆盖**镜像 vendor 路径
（`scripts/serve_a2.sh` 的 `[OPP-OVERRIDE]` 段 + `patches/opp_override_block.sh`）。
⚠️ 该段**不能写进生成 `inner.sh` 的 heredoc 内部** —— 实测往那个 heredoc 里插任何块都会让
生成的 `inner.sh` 变成 **0 字节**（复现 3 次），所以改成"生成后用 awk 注入"。

## 4. 失败原因（未定论，两条候选）

1. **kernel 在 ACL 图捕获下不安全**（最可能）：算子级验证是 eager 逐次调用；真实服务里该算子
   在 npugraph 捕获的图内重放。armC 改动的是 `_a8w4_msd_pipeline.h` 的 group 扫描/`M` 取值，
   可能改变了 workspace 使用或同步假设，只在图重放时暴露。
2. 覆盖镜像 vendor 时**连带影响**了其它文件（已核对：两包 1136 个文件里**只差这 1 个 `.o`** ⇒
   这条基本排除）。

⇒ 若要把这条线做下去，下一步应是：**在 eager 与 graph 两种执行下分别做算子级 A/B**，
先在单算子层面复现"图下挂死"，再改。当前按纪律**不采纳**。

## 5. 交付状态

* 交付配置**不含** armC（`V41_HC_OPP_PKG` 不设 ⇒ 走 stock vendor，md5 `2e4a834a…`）。
* 产物留档（不删）：`patches/gmm1_armC_vendor/`、`patches/gmm1_opp/`、`patches/opp_override_block.sh`，
  报告 `~/tmp/gmm1/REPORT-FINAL.md`。
* `[OPP-OVERRIDE]` 机制本身**保留**（默认不触发，只在显式设 `V41_HC_OPP_PKG` 时生效）——
  它是"给已有算子换 kernel"的唯一可行路径，后续做 kernel 优化会反复用到。
