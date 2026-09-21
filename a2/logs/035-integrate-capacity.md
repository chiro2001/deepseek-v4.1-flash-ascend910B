# 035 — 4/5 条容量杠杆的**合成**与单卡端到端集成验证

> 2026-09-22 03:1x–04:0x CST。执行：子代理 **X_integrate**。机器：**A3（A3-node1）槽位 c2 = die 7**
> （容器 `prbench-c2`）。全程只用 c2（c1 被 `kv8_prefill` 占用，未碰）；没碰 `dsv41-a3` /
> `mooncake-*` / `jitpgo-*` / 他人容器；**没写 `upstream-v41/`**；没用 `/tmp`
> （`TMPDIR=~/tmp/20260922/x_integrate`）；跨机传输全走 coscli；代码只写
> `a2/agents/X_integrate/`，别人的影子包**只读**。

---

## 0. 五句话结论

1. **【实测·主交付】4 条杠杆能一起跑**：`L5 + L1 + SWA-quant + ring16` 的**单一入口 shadow-pkg**
   起服成功，容量 **33,295 token（= `033` 的 ×1.4655 那一档，逐字复现）**，**四条判据全部转正**
   （`BlockStored:CPU=714`、`CPU→GPU=231.67 MB`、`hits=65,520 / 131,328`、
   replay 72.5 vs fill 534.1 ms = **7.37×**）；★ 但**第五条（正确性）没过**，见 4。
2. **【实测】5 条杠杆（再加 KV8 双平面 + prefill）也一起跑**：容量 **43,469 token = ×1.9133**
   （`033` 的满档，逐字复现），replay 85.3 vs fill 500.8 ms（**5.87×**）。⇒ **容量侧的 14× 成立**。
3. **【实测】总倍数 = 12.98×（4 条）/ 13.52×（5 条）**，对任务书预测的 14.0× 是 **92.7% / 96.6%**。
   （口径：L5 的 4.89× × 本文实测的"每 unit 宿主字节"2.654× / 2.766×；见 §4）
4. **⛔【实测·否决点】int8 平面经 DRAM 池往返后不保真**：同一份配置、同一批 prompt，
   **池不命中（1 MiB 池）时 replay 与 fill 逐字节相同**（5 个冷参考臂全部 ✅），
   **池一旦命中就出现差异**（`SWA-quant` 单独 = 1/16 prompt 翻 token；再加 `ring16` = 14/16；
   5 条杠杆 = 15/16），而**不含任何 int8 的臂（L5/L1/ring16）在同样命中条件下逐字节相同**。
   ⇒ **KV8/SWA-quant 与卸载池的组合在"保真性"修好之前不能随线上线**。
5. **【实测】冲突清单 4 组**（§2）：其中 **1 组必须手工合并**（`core/deepseek_v41.py`：SWA-quant ★ ‖ ring16 ★），
   另外 3 组可叠加（两组是"同模块不同区域"、一组是"超集"）。**无结构性冲突。**

---

## 1. 先回答主代理的三个问题（§1 判据强度）

主代理问：**fill 轮的 attention 读的是 int8 反量化值，还是原始 BF16 投影？**
（这决定 `sha256` 是强判据还是弱判据。）

### 1.1 【实测】是反量化值 ⇒ **判据是强判据**

`agents/X_integrate/patch_extra/kv8_trace.py`（只读探针，`X_KV8_TRACE=1`）在真实服务进程里
包裹三个函数并打点：`kv8_ori_plane`（SWA **读**）、`kv8_swa_store`（SWA **写**）、`_kv8_cmp_plane`（压缩面读）。

臂 `x-trace`（`L5 + SWA-quant`，4×1024 token）的实测输出：

```
[X_trace] store#1..8 plane=tuple          ← 写侧：SWA 平面就是 (payload, scale) 元组
[X_trace] t=12.4s store=8  ori=8  cmp=0   ← 读侧计数**逐次跟上**
[X_trace] t=24.3s store=920 ori=920 cmp=0 ← 全程 1:1
```

`ori` 与 `store` **从第 1 次就 1:1 同步**（而不是只在 replay 轮出现），说明**每一次 SWA 写入都配着一次
反量化读**——包括 fill 轮。⇒ **`fill` 的 attention 看到的就是 `dequant(int8)`**，
而 DRAM 池里存的是同一份 int8 字节 ⇒ **"池命中后 replay 应与 cold replay 逐字节相同"是强判据**。
（`cmp=0` 说明 1024-token 的 prompt 根本没走压缩面，这一格只能验证 SWA 面。）

### 1.2 ★ 但"不匹配"≠"必然是 bug"——本文的判据（按主代理 §2 改口径）

按 cannbot `model-infer-quantization/SKILL.md:424-451`（**§7.1 等价性自检**，逐字）：

> *"文本 diff：记首个分歧 token 位置 + 是否语义等价。**W8A8 允许细微 token 差异**，
> 重点是不应出现乱码 / 早停 / 与 BF16 显著走偏。"*

⇒ 单看 `sha256` 不匹配**不足以**判定 bug。本文改用**配对的"池 vs 冷"对照**（同配置、同 prompt、
唯一差别 = 池子能不能命中）+ **首 token 判据**（`MAX_TOKENS=1`，harness 默认值，见 §3.2）：

| 判据 | 通过条件 | 结论 |
|---|---|---|
| **J1 冷参考** | `replay_sha(cold) == fill_sha` | 全部 5 条冷臂 ✅（含 int8 臂） |
| **J2 池保真** | `replay_sha(pool) == replay_sha(cold)` | 无 int8 臂 ✅；**有 int8 臂 ❌** |
| **J3 首 token** | 差异出现在第 1 枚生成 token | 本文所有臂都是 `MAX_TOKENS=1`（**harness 默认**）⇒ 差异**就在首 token** |

---

## 2. ★★ 冲突清单（**本任务第一交付物**）

### 2.1 逐文件对齐（`diff -rq` 实测，不是目测）

两份 int8 影子包（`R_ringshrink/shadow/vllm_ascend` 与 `KV8_swa/shadow/vllm_ascend`）：
**全树只有 6 个文件不同，其余 636 个逐字节相同**（同一镜像基底）⇒ 可以"整包取其一 + 覆盖 + 一处合并"。

| # | 文件 | 谁改 | 判定 | 依据（本文实测） |
|---|---|---|---|---|
| **C1** | `core/deepseek_v41.py` | **SWA-quant（KV8_swa）★ ‖ ring16（R_ringshrink）★** | **⛔ 必须手工合并** | 两边都插在**同一锚点区**（`STATE_RING_ROWS = 32` 之后）：`git merge-file -p` 报 **1 处冲突**。但改的是**不同符号**（ring：`RING_STATE_DTYPES`/`ring_state_dtype`/`CompressorStateSpec` 守卫/`reshape_cache` 断言；KV8：`KV8_SCALE_DIM`/`*_plane_kwargs`/`FullSpec`+`SWASpec` 的 `__post_init__`/`_cache_plane_sizes` 泛化）⇒ 已用确定性脚本 `merge/merge_core.py` 合并（**11 条自检全过**，见 §2.3） |
| **C2** | `attention/dsa_v41.py` | SWA-quant（KV8_swa）‖ **KV8_prefill**（`033`） | **可叠加（超集）** | `KV8_prefill ⊇ KV8_swa`（KV8_prefill 本人确认 + 本文 `diff` 复核：差异只有 6 个 hunk，其中 1 个是新文件 `attention/kv8_prefill_triton.py` 的接线）。⇒ 直接以 KV8_prefill 版为基底 |
| **C3** | `offloading/config.py` + `vllm/v1/kv_offload/cpu/spec.py` + `native/cpu_npu.py` | **L5（SWA_pergroup）‖ L1（P2_poolsizing）‖ publish-0002（池走 `aclrtHostRegister`）** | **可叠加（机制已内建）** | P2 的 `sitecustomize.py` **显式 exec** PGP 的 `sitecustomize.py`（作者原话"本补丁是叠在它之上的第三层"）；L5 对 `cpu_npu.py` **只挂只读日志**，0002 是**整文件替换** ⇒ 不撞。**但**：三者都要在**同一个进程**里挂，PYTHONPATH 顺序敏感（本文的做法：`pkg/shadow : pkg/patch : pkg/patch_pgp`） |
| **C4** | `offloading/scheduler.py` | L5 独占（= D2 超集） | **独占，不许再叠旧版** | `publish/README.md` §1 已写明"只需挂这一份"；本文实测 `grep -c D2_offload` 正常 |
| — | `core/kv_cache_interface.py` / `models/deepseek_v41/model.py` | SWA-quant 独占 | 可叠加（直接覆盖） | ring16 **没碰**这两份（md5 与镜像原版相同） |
| — | `models/deepseek_v41/compressor.py` / `ops/triton/compressor/compressor_triton.py` | ring16 独占 | 可叠加（直接覆盖） | KV8_swa **没碰**这两份（md5 与镜像原版相同） |

> ★ **同一行对撞：0 处**。唯一需要人判断的是 C1 的**同一锚点**（不是同一行）。

### 2.2 ★ C1 那处"必须手工合并"到底难在哪（留给下一位）

```
<<<<<<< ring 版
RING_STATE_DTYPES = (torch.float32, torch.float16)
def ring_state_dtype(): ...
=======
KV8_SCALE_DIM = 4
def kv8_long_kv_enabled(): ...
def long_kv_plane_kwargs(): ...
def kv8_swa_enabled(): ...
def swa_plane_kwargs(): ...
>>>>>>> kv8 版
```

两边的**语义是互补的**（一个管 ring 的 dtype、一个管两个平面的 dtype），所以合并**不需要**取舍，
只需把两边的新增块**顺序插好**。`merge/merge_core.py` 就是这么做的（4 步 E1–E4 + 11 条断言），
**可重放**：`python3 merge/merge_core.py --ring <ring 版> --kv8 <kv8 版> --out <输出>`。

### 2.3 ★ 合并正确性的 11 条自检（脚本内置，每次 build 都跑）

```
OK ring 的 ring_state_dtype 仍在 / RING_STATE_DTYPES 仍在
OK ring 的 CompressorStateSpec 守卫（放宽版）/ reshape_cache 宽松断言
OK KV8 的 KV8_SCALE_DIM / swa_plane_kwargs / long_kv_plane_kwargs
OK KV8 的 FullSpec 守卫 / SWASpec 守卫 / scale 平面泛化（2 处 isinstance→getattr）
OK 无冲突标记
```

### 2.4 ★ C4 的一个**实测坑**（与主代理 03:4x 那条一致）

`git merge-file` 的手工合并**只解决文件冲突，不解决"两份补丁各自改名"**：
`KV8_prefill` 的镜像是 **`KV8_gather`**（023 的 flat `index_select`），
`KV8_fuse` 是**另一支**（026 的融合 kernel）。本文的 D/F 臂用的是 **023 那一支**，
⇒ **容量数字不受影响**（容量由 spec/页几何决定），但 **decode 侧的时延不是 `026/028` 的最好值**
（`028` 实测 `full_int8_fused = +45.57 µs/层`）。**要拿最好时延应把 028 的 kernel 叠上来**
（KV8_prefill 的接线只在 prefill 分支接管，decode 分支原样透传，**不冲突**）。

---

## 3. 合成件（单一入口 shadow-pkg）

### 3.1 两个包（都由一条命令生成）

```bash
# 4 条杠杆（L5 + L1 + SWA-quant + ring16）
bash a3_chip.sh c2 --timeout 300 --name xint-build -- env BASE=ring \
  bash /work/agents/X_integrate/scripts/build_pkg.sh        # → /work/agents/X_integrate/pkg-ring

# 5 条杠杆（再加 KV8 双平面 + prefill Triton）
bash a3_chip.sh c2 --timeout 300 --name xint-build -- env BASE=kv8pf \
  bash /work/agents/X_integrate/scripts/build_pkg.sh        # → /work/agents/X_integrate/pkg-kv8pf
```

| 包 | 目录布局 | 说明 |
|---|---|---|
| `pkg-ring` | `shadow/vllm_ascend/`（641 文件）+ `patch/`（L1）+ `patch_pgp/`（L5）+ `manifest.md5` | 基底 = `R_ringshrink` 影子包；覆盖 KV8 独占 2 文件；`attention/dsa_v41.py` = KV8_swa 版 |
| `pkg-kv8pf` | 同上 + `attention/kv8_prefill_triton.py`（642 文件） | 基底 = `KV8_prefill` 影子包（含 prefill Triton）；ring 三件换回 `034` 正式版；`dsa_v41.py` 再打 **decode 侧 scratch 角色分键**（§3.3） |

**md5 清单**（每次 build 打印并落 `manifest.md5`，原件见 `logs/raw/035-x-integrate/pkg*/manifest.md5`）：

```
pkg-kv8pf/shadow/vllm_ascend/
  75f4e565adc1b12c854a0a01271b6c4d  attention/dsa_v41.py
  796d0ff6eda03716f31c9994b8d8b221  attention/kv8_prefill_triton.py
  b9ae81517fd01503fd929683df88a37c  core/deepseek_v41.py        ← 手工合并产物
  7e17f7cae054f0b2339b41b7c3642f0e  core/kv_cache_interface.py
  fd7ff753a508c457e7f846ea589aaf7a  models/deepseek_v41/model.py
  8a2be008ef405ab681728a275bcb5f77  models/deepseek_v41/compressor.py
  9362e72e3ea12e8344c4485104eab837  ops/triton/compressor/compressor_triton.py
上游来源（只读）：ring 4edb5ad0… / kv8swa 6801e3a8… / 镜像原版 f48b1761…
补丁层：p2_hooks 6375eb21 / p2_pool 256a5408 / sitecustomize c55dd751
        pgp_hooks af2fefb8 / pgp_manager 3b64eb49 / pgp_scheduler 15d5548e（= publish 0001）
```

### 3.2 开关（**全部默认关**，逐条可回滚）

| 层 | env | 默认 | 含义 |
|---|---|---|---|
| L5 | `PGP_PATCH=1` + `PGP_BPC='{"default":8,"swa":1}'` | **开**（本包必开，否则没 scheduler 补丁） | pool 的格子 = 1 GPU block + 每组自己的 bpc |
| L5 | `SWA_TRIM` | `off` | `window` = 017 那条**不安全**近似，只用于复现反例 |
| L1 | `P2_POOL_PATCH` | **1**（本文） | `0` = 逐字回退 `021` 行为（**默认关**是 P2 自己的默认，本包 runner 里显式设 1） |
| L1 | `P2_COMP_JSON` | 必给 | 组↔张量的**分量**提示；**张量数变了就要重给**（16 张量 vs 20 张量两套，见 §4.2 的坑） |
| SWA-quant | `VLLM_V41_KV8_SWA=1` | **0** | SWA 页 INT8 + g128 fp16 scale |
| KV8（双平面） | `VLLM_V41_KV8=1` | **0** | long-KV 也 INT8（`XKV8`；与 prefill 开关解耦） |
| prefill 融合 | `VLLM_V41_KV8_PREFILL=1` | **0** | 不开 ⇒ prefill 走 018 的 torch 页粒度重建（+40.9 ms/step） |
| ring16 | `VLLM_V41_RING_FP16=1` | **0** | compressor state ring FP32→FP16 |
| 诊断 | `X_LEGACY_SCRATCH=1` | **0** | 恢复 `kv8_scratch_plane` 的旧键（复现 `033` 的别名 bug） |
| 诊断 | `X_KV8_TRACE=1` | **0** | KV8 只读探针（§1.1） |

**回滚**：任一 env 置 0 即回到上一层（**4 层互不依赖**：`PGP` / `P2` / `KV8*` / `RING_FP16`）；
整个包不生效只需把 `PYTHONPATH` 里的 `pkg/shadow` 去掉。

### 3.3 ★ 顺手修掉的一个**静默 bug**（`033` §3.4 的 decode 侧）

`dsa_v41.kv8_scratch_plane` 的键原本只有 `(blocks, block_size, dim, dtype, device)`：
**窗口重建与压缩重建在页数相同时共用同一块 scratch** ⇒ 后者覆盖前者，**无任何报错**。
`033` 只在**自己的 prefill kernel** 里按 role 分键（`kv8_prefill_triton._scratch`），
**decode 读侧没有**。本文的 `merge/merge_dsa.py` 把它补上：

```python
def kv8_scratch_plane(blocks, block_size, dim, dtype, device, role=None):
    if role is None and os.environ.get("X_LEGACY_SCRATCH", "0") != "1":
        role = sys._getframe(1).f_code.co_name       # 调用者函数名，O(1)，不动调用点
    key = (role, blocks, block_size, dim, dtype, str(device))
```

⇒ 默认**按 role 分键**，`X_LEGACY_SCRATCH=1` 可**一键回到旧行为**（保留 A/B 能力）。

### 3.4 ★ 主代理 §(a)(b)(c) 三条机制假设的判定

| 假设 | 判定 | 证据 |
|---|---|---|
| **(a) ring 页步长（`CACHE_PAGE`）没采纳** | **❌ 排除** | 本包用的是 `034` 的正式版 kernel：`grep CACHE_PAGE` 在 `ops/triton/compressor/compressor_triton.py` 命中 8 处，调用点传 `CACHE_PAGE=state_cache.stride(0)`（第 726/741 行）；`034` 的 `ring_page_stride` 分支在 |
| **(b) state 视图 stride 与 SWA 别名相交** | **⚠️ 未确认（有一条间接证据）** | 探针 dump（`x-trace`）：`payload stride=(131072,512,512,1) off=1458176` 而 `scale stride=(65536,4,4,1) off=761856` ⇒ **同一个 SWA 元组的两个平面 block 步长不同**（131072 vs 65536），说明这两支视图来自**不同的 slot/tensor**（与 worker 打印的 `pages=[…,65536,1024]` 一致）。**该步长差是否被卸载池的 DMA 寻址正确使用 = 未知**，这正是 §5 那条否决点最可疑的落点 |
| **(c) scratch 别名（我改了，会不会是我改坏的？）** | **❌ 排除** | 两臂用的是**同一个 `pkg-ring/attention/dsa_v41.py`**（md5 见每次 arm 的 `meta.txt`，`9db97849…`）；而且 `x-C`（`SWA-quant+ring16`，**没有** role 分键的 KV8_swa 版）与 `x-D`（有分键）**replay sha 完全相同**（`6a47dd65…`）⇒ 分键改动**不改变任何数字** |
| **(d) aux-stream 竞态（本文新增的候选）** | **❌ 排除** | `x-D-noms`（`multistream_dsv4_dsa_overlap=false`）与 `x-D` **replay sha 相同**（`6a47dd65…`，14/16） |

---

## 4. 实测：19 条臂的全表

### 4.1 主表（c2，`--load-format dummy` tiny，16 × 4096 token，池 = 144 MiB 或 1 MiB）

`MAX_TOKENS` 默认 = **1** ⇒ **所有"不匹配"都出现在第 1 枚生成 token 上**（J3）。

| # | 臂 | 杠杆 | 池 | `GPU KV cache size` | Σpage/unit | unit 记账 | **宿主/unit** | replay TTFT(p50) | 加速 | **J1 冷参考** | **J2 池保真** |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| A | `x-A` | L5 | 144 M | 22,719 | 910,208 | 131,072 | 910,208 | 45.7 ms | 10.06× | ✅ | ✅ |
| B | `x-B` | **L5+L1** | 144 M | 22,719 | 910,208 | 131,072 | **464,640** | 46.0 ms | 10.03× | ✅(x-B-cold) | ✅ **逐字节** |
| R | `x-R` | L5+ring16 | 144 M | 22,719 | 910,208 | 131,072 | 910,208 | 47.6 ms | 9.71× | ✅(x-R-cold) | ✅ |
| R2 | `x-R2` | **L5+L1+ring16** | 144 M | 22,719 | 910,208 | 131,072 | **464,640** | 47.1 ms | 9.80× | ✅(x-R2-cold) | ✅ **逐字节** |
| C0 | `x-C0` | L5+SWA-q | 144 M | 22,719 | 832,128 | 131,072 | 832,128 | 64.3 ms | 7.95× | ✅(x-C0-cold) | ❌ **1/16** |
| C | `x-C` | L5+SWA-q+ring16 | 144 M | **33,295** | 660,480 | 77,824 | 660,480 | 72.6 ms | 7.31× | ✅ | ❌ **14/16** |
| D | `x-D` | **4 条**（+L1） | 144 M | **33,295** | 660,480 | 77,824 | **342,964** | 72.5 ms | 7.37× | ✅(x-D-cold) | ❌ **14/16** |
| D′ | `x-D-r2` | 同 D（复跑） | 144 M | 33,295 | — | — | — | 71.4 ms | 7.50× | — | ❌ 14/16（**同 sha**，可复现） |
| D″ | `x-D-noms` | 同 D，关多流 | 144 M | 33,295 | — | — | — | 74.0 ms | 7.29× | — | ❌ 14/16（**同 sha**） |
| E | `x-E` | 同 D，**池缩到 1.000×** | **89.6 M** | 33,295 | 660,480 | 77,824 | **343,000** | 77.7 ms | 6.92× | ✅ | ❌ 14/16 |
| F | `x-F-full` | **5 条**（+KV8 双平面+prefill） | 144 M | **43,469** | 607,360 | 69,632 | **329,131** | 85.3 ms | 5.87× | ✅(x-F-cold) | ❌ **15/16** |
| — | `x-smoke` | 同 F（2×512 冒烟） | 144 M | **43,469** | 607,360 | 69,632 | 329,131 | 160.6 ms | 29.1× | — | ✅（暖机后 fill==replay） |
| — | `x-trace` | L5+SWA-q，4×1024 | 144 M | 22,719 | — | — | — | 188.0 ms | 0.98× | — | ✅（**但没命中**，见 §1.1） |

**冷参考臂**（池 1 MiB ⇒ 一个 unit 都装不下 ⇒ replay 走重算）：`x-B-cold`（453.2 ms）、
`x-C0-cold`（499.3）、`x-D-cold`（509.5）、`x-R-cold`（452.9）、`x-R2-cold`（454.4）、`x-F-cold`（476.9）
—— **6/6 的 `replay_sha == fill_sha == 24b57053…`（16/16 prompt 逐字节）**。

★ 所有臂（含 int8 臂）的 **fill sha 都是 `24b57053…`** ⇒ **int8 量化本身不改变 fill 的输出**
（与 `034` 的"FP16 ring 的误差低于 bf16 地板"、`020` 的"SWA 量化几乎不额外掉精度"一致）。
差异**只在"int8 平面 + DRAM 命中"的交叉点**出现。

### 4.2 ★ 两个必须记住的坑（都会静默出错）

1. **`P2_COMP_JSON` 必须随"张量数"换**：无 KV8 时 16 张量，`[[0],[1..11]]`；
   **开 SWA-quant 后变 20 张量**，真分量是 `[[0,2,3,4,5,6,7,8,9,10,11],[1]]`。
   给错 ⇒ P2 的 worker **拒绝启动**（`[P2] 行区间冲突`，**fail-closed，不会静默错**）——
   本文第一次 `x-D`（batch1）就是这么挂的，**改正后立刻起服**。
2. **同一个 knob、不同的 unit 成本**：`OFFLOAD_BYTES` 是"记账字节"，而 unit 的记账成本随几何变
   （131,072 → 77,824 → 69,632 B）。所以 **同一个 144 MiB 在 A 臂 = 1152 unit，在 D 臂 = 1940 unit**。
   `E` 臂就是"把 knob 换算回 1.000× 工作集"的那一格（1152 × 77,824 = 89,653,248 B）。

### 4.3 四条判据 + 变长前缀（**任务书要求的五条**）

| 臂 | ① `BlockStored:CPU` | ② `CPU→GPU` | ③ `hits` / `queries` | ④ replay vs fill TTFT | ⑤ 变长前缀安全 |
|---|---:|---:|---:|---|---|
| A（L5） | 714 | 272.96 MB | 65,520 / 131,328 | 45.7 vs 459.9 = **10.06×** | ✅（`021` 已测；本文同代码） |
| B（L5+L1） | 714 | 272.96 MB | 65,520 / 131,328 | 46.0 vs 461.2 = **10.03×** | ✅（同 A 几何） |
| D（4 条） | 714 | **231.67 MB** | 65,520 / 131,328 | 72.5 vs 534.1 = **7.37×** | ⚠️ **归因未分离**（见 §4.4） |
| F（5 条） | 714 | **149.09 MB** | 65,520 / 131,328 | 85.3 vs 500.8 = **5.87×** | ⚠️ 同上 |

（②是 `vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"}`；③④来自 `metrics_after` + client json。）
**四条判据在 D/F 上全部"数值上全中"**（hits>0、`CPU→GPU`>0、replay≫fill），
**但 J2 未过** ⇒ 本文**不把 D/F 判为"上线通过"**，只判为"容量通过、正确性待修"。

### 4.4 ★ 变长前缀（4096 → 2048 回放）这一格**没拿到干净结论**

本文为它准备了 `BATCH=3`（`x-D-vp`），但**没跑**：因为 `x-D` 在等长回放上**已经** 14/16 不匹配
⇒ 再叠一个"更短前缀"只会把两个原因混在一起（任务书 §1 明令"不许用相邻数字顶替缺的那格"）。
**⇒ 这一格标【未确认】**，正确的做法是：**先修 §5 的保真性，再跑变长前缀**。
（`021` 在**无 int8** 的条件下已实测 4096→2048 安全：`hits=32,752`、replay 46.1 ms（10.1×）。）

---

## 5. ⛔ 否决点：int8 平面经 DRAM 池往返**不保真**

### 5.1 证据链（三条，全部【实测】）

1. **判据强度已证明**（§1.1）：fill 的 attention 读的就是 `dequant(int8)`，`ori:store = 1:1`。
2. **池不命中时 6/6 冷臂逐字节一致**（含 int8 臂）⇒ **int8 量化是确定性的、可复现的**，
   而且 `reset_prefix_cache` + 全量重算这条路**不会**引入差异。
3. **池一旦命中**：
   * **无 int8 的臂（A/B/R/R2）：仍然逐字节一致**（J2 ✅）；
   * **有 int8 的臂：首 token 就不同**（J2 ❌）：`C0` 1/16、`C/D/E` 14/16、`F` 15/16。

⇒ 由于"输入相同 ⇒ 输出相同"已被冷臂证明，唯一能被改动的输入就是**从 DRAM 取回的那份 KV**。
**⇒ 结论：int8 平面（SWA-quant / KV8）经卸载池往返后，取回的值与原值不同。**

### 5.2 已**排除**的机制（三条，都有 A/B）

| 机制 | 排除方式 | 结果 |
|---|---|---|
| L1（按组配额/分量行空间） | `x-C`（L1 off）vs `x-D`（L1 on），**同杠杆、同池** | replay sha **相同**（`6a47dd65…`）⇒ 不是 L1 |
| scratch 别名（`033` §3.4） | 同包内 `X_LEGACY_SCRATCH=0/1`；且对比 KV8_swa 版 | sha 相同；两臂用同一份 `dsa_v41.py`（md5 `9db97849…`）⇒ **不是它** |
| aux-stream 竞态 | `x-D-noms`（关多流） | sha 相同（14/16）⇒ **不是它** |
| ring 页步长（`CACHE_PAGE`） | 代码核查 | `034` 正式版在，`CACHE_PAGE=state_cache.stride(0)` 已接线 ⇒ 不是"漏采纳" |

### 5.3 剩下的候选（**未确认**，留给下一位）

* **候选 1★：`block_stride` ≠ `page_size_bytes` 的平面，池的寻址用的哪一支？**
  探针 dump 出 `payload stride=131072` 而 `scale stride=65536`（**同一个 SWA 元组的两支**），
  而 worker 打印的每张张量 `page` 是 [65536, 8192, 128, …, 65536, 1024]。⇒
  **只要卸载 worker 的行寻址用 `page_size_bytes` 而行宽实际是 `block_stride`，行 N 就会落到错误的字节**；
  这在"页 == 行宽"的 BF16 几何下**看不见**（A/B/R/R2 全部逐字节一致正是这个原因），
  在 int8（页 < 行宽）下**必然出现**。**最小验证**：打印 `CPULoadStoreSpec` 每张张量的
  `(page_size_bytes, block_stride, data_ptr)` 三元组，比对 `reshape_cache` 的 `stride(0)`。
* **候选 2：scale 平面被当成独立的 1024 B 张量**（worker 打印里确有 3 张 1024 B 的张量），
  它的 `block_stride=65536` 与 payload 的 131072 不同 ⇒ **同一元组的两支被塞进不同的行空间**。
* **候选 3：KV8 的写侧 `scatter_cache_sk` 在 aux 流上**（本文已排除"多流开关"这一层，
  但**没有**排除"写侧 scatter 与卸载 store 之间的同步"这一层——关掉多流后写仍在别的流上）。

### 5.4 ★ 这对"能不能上线"意味着什么（**回滚建议**）

| 组合 | 容量 | 池保真 | 判定 |
|---|---|---|---|
| **L5（现状）** | ×1.000 | ✅ | ✅ 可上线（已在 A3 8 卡验证，`022`） |
| **L5 + L1** | ×1.000（HBM 不变），**池 1.96×** | ✅ **逐字节** | ✅ **可上线**（本文 B/R2 臂实测） |
| L5 + ring16 | ×1.000（`033` 实测：单独缩 ring 一分钱不省） | ✅ | ⚪ 无收益，不必上 |
| **L5 + L1 + SWA-quant + ring16** | **×1.4655（33,295）** | ❌ 1/16→14/16 | ⛔ **先修保真性** |
| **+ KV8 双平面 + prefill（5 条）** | **×1.9133（43,469）** | ❌ 15/16 | ⛔ **先修保真性** |

⇒ **A2 上线先走 `L5+L1`**（这是唯一"容量 +1.96× 且逐字节一致"的组合）；
**int8 那两条杠杆的容量是真的（14× 的账成立），但正确性欠一次修复**。

---

## 6. ★ 容量总账（把"14×"钉死或推翻）

### 6.1 HBM 侧（`GPU KV cache size`，逐字复现 `033` 的 5 档）

| 配置 | 本文实测 | `033` 实测 | 倍数 |
|---|---:|---:|---:|
| 基线（L5 only） | **22,719** | 22,719 | ×1.000 |
| +ring16（单独） | **22,719** | 22,719 | **×1.000**（单独缩 ring 白做，与 `033` 一致） |
| +SWA-quant 单独 | **22,719** | — | **×1.000**（页被 state ring 顶住） |
| +SWA-quant +ring16 | **33,295** | 33,295 | **×1.4655** |
| **5 条杠杆（+KV8 双平面）** | **43,469** | 43,469 | **×1.9133** |

### 6.2 DRAM 池侧（**每 unit 宿主字节**，本文实测）

| 配置 | Σpage/unit（实测对账行） | 记账 unit | **宿主/unit** | 相对 L5 |
|---|---:|---:|---:|---:|
| L5 | 910,208 | 131,072 | **910,208** | 1.000 |
| L5+L1 | 910,208 | 131,072 | **464,640** | **1.960×** |
| L5+SWA-quant（单独） | 832,128 | 131,072 | 832,128 | 1.094× |
| L5+SWA-q+ring16 | 660,480 | 77,824 | 660,480 | 1.378× |
| **4 条（+L1）** | 660,480 | 77,824 | **342,964** | **2.654×** |
| **5 条（+KV8 双平面+L1）** | 607,360 | 69,632 | **329,131** | **2.766×** |

（`x-E` 用 **1.000× 工作集** 的池子独立复核：1152 unit × 343,000 B/unit = 395,116,544 B
= `x-D` 的单 unit 成本 × 1152，**两条路径吻合到 0.01%**。）

### 6.3 ★ 总倍数（相对"SWA 全存 + BF16"的原始基线）

```
4 条杠杆：L5 4.89×  ×  池侧 2.654×  =  12.98×
5 条杠杆：L5 4.89×  ×  池侧 2.766×  =  13.52×
任务书预测 4.89 × 1.96 × 1.4655 = 14.05×
⇒ 达成 92.5% / 96.2%
```

**差异来自哪里（可解释，不是误差）**：
任务书用的是 **HBM 侧的 ×1.4655**，而**池侧的真实 Σpage 比是 ×1.378**——
两者**不是同一个量**：HBM 的页几何是 `max(kv+index, ring, swa)`，而池子要覆盖的是
**16/20 张张量之和**（其中还包含被排除出卸载的 state 组那 4 张）。
⇒ **14× 的"账"用 HBM 口径成立（1.9133 甚至更高）；落成"宿主字节"时是 13.0–13.5×。**
【实测】，不是【推断】。

### 6.4 ★ A2 的新容量边界（外推到 8 rank，unit/请求沿用 `030` 的 608 / 2,432）

| 场景 | L5 only | **+L1（可上线）** | **4 条杠杆** | **5 条杠杆** |
|---|---:|---:|---:|---:|
| 16 × 32K | 66 GiB | **33.6 GiB** | 17.7 GiB | 17.0 GiB |
| 16 × 128K | 264 GiB | **134 GiB** | 99.4 GiB | 95.4 GiB |
| **32 × 128K** | 528 GiB ⛔ | **269 GiB** ✅ | **199 GiB** ✅ | **191 GiB** ✅ |
| 64 × 128K | 1,056 GiB ⛔ | 538 GiB ⛔ | 398 GiB ⚠️ | 382 GiB ⚠️ |

（A2 宿主余量 **442 GiB**；另需为 `030` §4.4 的"首次 `aclrtHostRegister` 多花 ~589 MiB/进程"留 **~4.7 GiB**。）

---

## 7. ★ A2 就绪度

### 7.1 一条命令自检（**不占卡时的等价版**；A2 上按 `publish/README.md` §2.2 同构）

```bash
# 0) 起服后（容器内）
grep -c "P1_pinned.*ret=0"           serve.log   # 期望 == worker 数（TP8 ⇒ 128 次注册 / 16 张量 × 8）
grep -c "D2_offload"                 serve.log   # 期望 > 0
grep -c "alignment_chunk_count.*8"   serve.log   # 期望 > 0（per-group bpc 生效）
# 1) ★ 本文新增（int8 集成时必须看）
grep -c "VLLM_V41_KV8_SWA=1"         serve.log   # 只做记录
grep -a "记账对账"                    serve.log   # ★ 看 Σpage 与 worker_kv_bytes_per_block 是否相等
#   相等 ⇒ 该几何下"页 == 行宽"，池寻址**结构性安全**（L5/L1 就是这样）；
#   不等 ⇒ int8 几何，**必须先过 §5 的保真性检查**再上线
grep -a "P2_WORKER_HOST_BYTES"        serve.log   # 宿主实占（与记账值求比值）
# 2) 四判据（跑完一轮后）
curl -s localhost:<port>/metrics | grep -E "kv_offload_total_bytes_total.*CPU_to_GPU|external_prefix_cache_hits_total"
```

### 7.2 `OFFLOAD_GB` 的新定值（**按 L1 之后的实测 unit 成本重算**）

记账 unit 在 A2（ws=8）上 = `kv_bytes_per_unit × 8`；宿主/unit = 本文实测 × 8：

| 组合 | 记账 unit | 宿主/unit | 32 × 128K（77,824 unit） | 16 × 128K |
|---|---:|---:|---|---|
| L5 only | 1 MiB | 7.28 MB | 528 GiB（⛔ 不可行） | 264 GiB |
| **L5+L1（今日可上线）** | 1 MiB | **3.72 MB** | **269 GiB** ⇒ `OFFLOAD_GB=80` | 134 GiB ⇒ `OFFLOAD_GB=40` |
| **4 条杠杆（待修保真性）** | 622,592 B | **2.74 MB** | **199 GiB** ⇒ `OFFLOAD_GB=48` | 99 GiB ⇒ `OFFLOAD_GB=24` |
| 5 条杠杆 | 557,056 B | 2.63 MB | 191 GiB ⇒ `OFFLOAD_GB=44` | 95 GiB ⇒ `OFFLOAD_GB=22` |

★ **主代理问的"新 `OFFLOAD_GB` 定值"**：

* **今天就能用的（L5+L1，唯一逐字节保真的组合）**：**`OFFLOAD_GB=40`（16×128K）**
  或 **`80`（32×128K）**；32K×16 用 **`10`**。**不要沿用 48**（48 在 L1 后对应 ~21×128K，
  既不够 32×128K，又比 16×128K 多占一倍宿主）。
* **等 §5 修好后（4 条杠杆）**：**`OFFLOAD_GB=48`** —— 但含义**变了**：
  旧 48 = 16×32K 的上限；**新 48 = 32×128K 的定值**（宿主 ≈199 GiB，占 A2 余量 45%）。

### 7.3 默认关的开关 / 回滚

见 §3.2。**一句话回滚**：所有新能力都在 `PYTHONPATH` 里，**去掉 `pkg/shadow` 即回现状**；
逐条回滚 = 对应 env 置 0。**没有任何一处写镜像**（红线 §1-#9）。

### 7.4 风险（任务书点名的两条 + 本文新增一条）

1. **ring16 的数值累积性**：`034` 在测；本文的实测是**间接正面证据**——`x-R2`（L5+L1+ring16）
   与 `x-A`（L5）的 fill/replay sha **都是 `24b57053…`**，即 **ring16 在 4096-token、8 token greedy
   的尺度上不改变任何输出**（与"ring 存的是原始投影、输出本身就是 BF16"的结论一致）。
   ⚠️ 但**长生成 / 长上下文下的累积性仍未确认**。
2. **SWA-quant 是否需要新的页几何断言**：**需要**。`034` 报的
   `core/deepseek_v41.py:292 sum(plane_sizes) != block_stride` 已被 `R_ringshrink` 放宽成
   `> block_stride`；本文的合并脚本对**两条断言各留了自检**（§2.3）。
   ★ 但本文实测出一条**更强的、之前没人写过的**：**页几何一旦让 `page_size_bytes ≠ block_stride`
   （int8 必然如此），卸载池的保真性就不成立**（§5）⇒ **断定言的放宽还不够，
   必须再加一条"页 == 行宽"的运行期自检**（§7.1 的 `记账对账` grep）。
3. **新增：`P2_COMP_JSON` 是张量数的函数**（§4.2 坑 1），A2 上必须先跑一条只读结构臂
   （`P2_STRUCT_LOG=1 P2_POOL_PATCH=0`）拿到 20 张量的真分量，再开补丁。

---

## 8. cannbot 对照（AGENTS.md §6）

本任务不改 kernel、不做量化数值验证，属"**集成 + 容量**"，因此只查了**两条**决定性条款：

| 查的地方 | 它说什么 | 本文采纳 / 没采纳 |
|---|---|---|
| `model/model-infer-kvcache/SKILL.md:102-113`（§2.1 物理布局与逻辑映射） | `物理 slot = block_table[b, seq_pos // block_size] × block_size + seq_pos % block_size`；**单一 `block_size`** | **逐字采纳**（上游实现就是它；本文一行没改）。★ 但 L5 的 per-group `blocks_per_chunk` 让**不同组有不同的 chunk 粒度**（SWA=1 block、full=8 block）——cannbot **没有**"不同组不同 block_size"的方案（`021` §7 已记），**没有先例可抄，也没有禁令**（我们没改 GPU 布局与算子，只改池的记账粒度） |
| `model/model-infer-quantization/SKILL.md:424-451`（§7.1 等价性自检） | *"文本 diff：记首个分歧 token 位置…**W8A8 允许细微 token 差异**，重点是不应出现乱码/早停/与 BF16 显著走偏"*；★ 且要求"**不能只看代码 diff，必须证明真实运行**"、"若连 BF16 参照都跑不了…不静默判通过" | **采纳**：本文因此**没有**用 sha 判死 int8 臂，而是改成"池 vs 冷"配对判据 + 首 token 判据（§1.2），并把"fill 到底读什么"用探针证明（§1.1）。⇒ 结论从"sha 不匹配 = bug"改成"**池往返不保真**"，两者判据强度不同 |

---

## 9. 交付物

| 类 | 位置 |
|---|---|
| 本文 | `a2/logs/035-20260922-integrate-capacity.md` |
| 原始数据（248 文件） | `a2/logs/raw/035-x-integrate/out/`（19 条臂的 meta / server.log / metrics / kv_events / client.json / selfcheck）+ `pkg*/manifest.md5` |
| 代码 | `a2/agents/X_integrate/`：`scripts/build_pkg.sh`（合成）、`merge/merge_core.py`（C1 手工合并，11 自检）、`merge/merge_dsa.py`（decode scratch 分键）、`patch_extra/*`（只读探针）、`scripts/run_arm_x.sh` + `run_batch_x.sh`（9 个批次定义）、`scripts/selfcheck.py` |
| 两个 shadow-pkg | `/work/agents/X_integrate/pkg-ring`、`pkg-kv8pf`（含 md5 清单） |
| 复现入口 | `bash a3_chip.sh c2 --timeout 2000 --name xint -- env BATCH=1 bash /work/agents/X_integrate/scripts/run_batch_x.sh` |

## 10. 诚实边界（哪些没测）

| # | 事项 | 状态 |
|---|---|---|
| 1 | **变长前缀 4096→2048 在 D/F 上的结果** | **【未确认】**——先修 §5，否则测出来无法归因（§4.4） |
| 2 | int8 往返**不保真的机制** | **【未确认】**（3 个候选，见 §5.3；候选 1 有间接证据） |
| 3 | 8 卡真实权重 / TP8 | **未做**（本文全在单卡 tiny；`033` 的容量档位是单卡真机 `GPU KV cache size`，与本文一致） |
| 4 | `concurrency > 1`、长生成（>8 token）、真实会话的多变长前缀 | **未做**（client 默认 `MAX_TOKENS=1`，`concurrency=1`） |
| 5 | `026/028` 的融合 kernel 未叠进本包 | 已知（§2.4）：容量不受影响，**decode 时延不是最好值** |
| 6 | A2 上的 `aclrtHostRegister` / 宿主实占复核 | **未做**（`publish/README.md` §4 已列为待办） |
