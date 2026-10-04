# gmm1 armD 自定义 kernel：**真机无效果**（负结果，2026-10-04）

> 结论：armD（只保留 OPT-A 的安全版）在 die5 单算子层实测 **−4.5%（77.8 → 74.3 µs）**、
> 图捕获 200 次 replay 全过、bitwise 相等；但**放进真实服务后，该算子的设备时间没有变化**
> ⇒ **不采纳进交付**。本文记录证据链，避免后人重复走。

## 1. 单算子层（子代理，可信）

| 项 | stock | armD | Δ |
|---|---:|---:|---:|
| Duration 中位 | 77.8 µs | 74.3 µs | **−4.5%** |
| `aic_scalar` | 34.4 | 26.4 | −8 µs |
| 配对 A/B | — | — | **11 轮交替，配对差中位 −3.3 µs，10/11 为负** |
| 正确性 | — | — | eager 4 例 + 图捕获 200 次 replay **全部 bitwise 相等** |
| 崩溃复现（前身 armC） | 正常 | **507015** | 根因：OPT-C 把 M 取成 `x.dim0`（容量）而非 `group_list` 前缀和（真实行数）⇒ 越界 |

## 2. 真机（本机 A/B，决定性）

**加载已确认**：起服前 `[OPP-OVERRIDE]` 把 vendor 复制覆盖镜像路径后，
容器内该 `.o` 的 md5 = **`f93d96d1…`（armD）**，与宿主逐字节一致
（stock 是 `2e4a834a…`）。⇒ 不是"没挂上"。

**但设备时间没动**（msprof `op_summary`，N=1 decode 稳态中位）：

| 算子 | stock（`hcfuse_1004_131322`，N=1） | **armD（`armD_1004_173853`，N=1）** | Δ |
|---|---:|---:|---:|
| `GroupedMatmulSwigluQuantV2` Duration | 73.64 µs | **72.98 µs** | −0.9%（噪声内） |
| `aic_scalar` | 30.79 | **30.46** | −1.1%（噪声内） |
| `aic_mte1` / `aic_mte2` / `aic_mac` | 11.21 / 15.51 / 2.01 | 11.21 / 16.19 / 2.01 | 不变 |

⇒ **单算子层的 −8 µs `aic_scalar` 收益在服务里完全没有出现**。

## 3. 为什么（两条候选，均未最终确证）

1. **服务走的是另一个 tiling key 的 kernel 二进制**（最可能）。
   该算子在 `op_impl/ai_core/tbe/kernel/ascend910_93/grouped_matmul_swiglu_quant_v2/` 下有
   **14 个 `.o`**（不同 tiling key）。子代理补丁作用于 Base 的 **idx 4（A8W4_MSD）**；
   而该 op 还注册了 `FusionTiling(0)` 与 `BaseTiling(1)` 两个模板且 **Fusion 优先** ——
   若服务输入走 Fusion 路径，则改 idx-4 的 Base 内核**根本不会被调用**（改不改都一样）。
2. **服务里的真实输入形状让 OPT-A 的早返回不触发**（次可能）。
   OPT-A 只在 `loopCount == 1` 时跳过 `UpdateWorkSpaceSplitConfig` 的扫描；
   服务 decode 的 `group_list` 长度 = 专家数（48），与子代理 bench 一致，
   所以这条较不可能是主因 —— 但仍无法排除某个 warmup/prefill 形状先命中了别的 key。

**判定判据（留给后人）**：先确认服务实际命中的是哪个 `.o`（可用 `binary_info_config.json`
的 `simplifiedKey` 反查输入签名），再决定改哪个 tiling key 的内核。
**不要**只验证"文件被复制进容器"就认为改动生效 —— 这次就是反例。

## 4. 交付状态

* **不采纳**：交付配置不设 `V41_HC_OPP_PKG`（走镜像自带 stock vendor）。
* 保留的机制（有价值，供后续 kernel 工作复用）：
  * `[OPP-OVERRIDE]`（`scripts/serve_a2.sh`）：把自定义 vendor 复制覆盖镜像路径 ——
    这是"给已有算子换 kernel"的**唯一可行路径**
    （`bootstrap_custom_op_env()` 会把镜像 vendor 路径前插到 `ASCEND_CUSTOM_OPP_PATH`，
    只设 env 无法覆盖）。
  * `tools/verify_opp_vendor.sh`：第 0 步就断言"容器内 `.o` 与宿主逐字节一致"。
  * **新增第 0.5 步（本次教训）**：还要断言"该 `.o` 真的是被调用的那个"
    （现在只证明了文件一致，没证明被调用）。
* 产物：`~/tmp/gmm1/pkg_armD/`、`REPORT-ARMD.md`；镜像 vendor 已恢复 stock（md5 `2e4a834a…`）。
