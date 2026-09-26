# 023 — KV8 读侧 gather：把二维高级索引换成 flat `index_select`

> 2026-09-22 01:3x–02:0x CST。执行：子代理 **KV8_gather**。机器：**A3（A3-node1）槽位 c1 = die 6**
> （`Ascend910_9382`，容器 `prbench-c1`）。全程只用 c1；没碰 `dsv41-a3` / `mooncake-master` /
> Phy-ID 8–15；没用 `/tmp`（`TMPDIR=/work/agents/KV8_gather/tmp`）；没写 `upstream-v41/`；
> 跨机传输全走 coscli。代码在影子包 `agents/KV8_gather/shadow/vllm_ascend/`（= KV8_swa 影子包 + 本次 1 文件增量）。

---

## 0. 五句话结论（先给主代理）

1. **【实测】gather 换成 flat `index_select` 了，而且是逐比特等价**：`kv_i8[phys]` / `kv_i8[phys, offset]`
   （2D 高级索引）→ **一维 `torch.index_select`**，索引向量从 `R×512` 个 int64 降到 `R` 个。
   重建出来的 scratch 与 block table 与旧路径 **`torch.equal` 逐比特相同**，端到端输出
   （真量化器）fast vs current **逐比特相同**。
2. **【实测】带宽：SWA payload 页 gather 33.0 → 68.6 GB/s（×2.1）**；payload+scale 23.0 → 50.8 GB/s；
   cmp 4096 行 gather 15.9 → ~27 GB/s。**判据（≥500 GB/s）不达标**，原因见 §4：
   这个尺寸下 gather 是**核函数延迟**主导（1 MB 的一次 gather ≈ 15 µs），不是带宽主导。
3. **【实测】整层增量：+436.4 → +347.2 µs/层（−20%）**（40 层整图，生产形状）：
   **SWA 层 +171.6 → +139.5 µs/层**，**cmp 读路径 +264.8 → +207.8 µs/层**。
   按 40 SWA + 4 源层外推：**+7.92 ms/step → +6.41 ms/step（+26% → +21%）**。
   ⛔ **判据（≤+0.2% = 60 µs/step，或 ≤+60 µs/层）远未达标**，原因**不是** gather 没换成功；
4. **【实测·关键】根因换了：rebuild 是「算子个数 × 每核延迟」主导，不是带宽主导。**
   40 层 ACLGraph 里**每个设备算子 ≈ 4–6 µs**（空图 replay 只有 0.8 µs/次，见 §4 的标定），
   SWA rebuild 有 ~30 个算子 ⇒ ~170 µs，与实测 171.6 吻合。所以**换掉其中 1 个算子最多省 30 µs**。
   ⇒ 想进 ≤60 µs/层，只能**把整个 rebuild 融成 1 个 kernel**（任务书第 2 步的 AscendC 路线，
   或 Triton-Ascend 的 `index_select_simd`/`gather_out_to_ub`）。
5. **【实测】精度零回退**：真量化器 `rel_L2 = 5.4331e-3`（020 是 5.46e-3）、`cos = 0.9999857`、
   `max_abs = 5.55e-4`、`nan_total = 0`；且 **fast 路径 vs 旧路径 `torch.equal` = True**。

---

## 1. ★ cannbot 对照（先做，不占卡）

在 A3-node1 只读查 `~/projects/dsv41/src/cannbot/vendor/cannbot-skills/`。

| 查的地方 | cannbot 说什么 | 我们采纳/没采纳 |
|---|---|---|
| `model/model-infer-fusion/references/torch_npu_API/torch_npu_list.md:1-160` | 144 个接口里有 **`npu_anti_quant`**（"对张量 `x` 进行反量化"）、`npu_gather_sparse_index`（**无描述**）、`npu_scatter_pa_kv_cache`、`npu_quant_scatter(_)`、`npu_kv_quant_sparse_flash_attention`（只吃 head_dim 576） | **试了 `npu_anti_quant`**（想把 dequant 4 个算子融成 1 个）⇒ **失败**：`AclNN_Parameter_Error(EZ1001): scale dim num must be 1`（它要每行 1 个 scale，我们的 g128 是每行 4 个）。**未采纳，但这是融核路线上的现成候选**（把 payload 看成 `[rows*4, 128]` + 一维 scale 就能对上）【未确认】 |
| `scripts/torch_npu_query.py show npu_gather_sparse_index` | **"未找到 API"**（降级到 `_FALLBACK_DOCS`） ⇒ 这个名字在本地文档里**没有语义说明**，不能作为设计依据 | **没采纳**；没有比 `index_select` 更合适的现成 gather 算子可查 |
| `ops/ops-profiling/`、`ops/ops-torch-ops-profiler/` | 四文件模板确实存在：`ops/torch-ops-profiler/examples/layer_norm_profiler_reference/`（另有 `ops/ops-profiling/{SKILL.md,references,scripts,evals}`） | **本次没走到自写 kernel**（flat 化可行，见 §2），但**把这条留给了融核**（§6 建议 1） |
| `ops/triton-latency-optimizer/references/docs_triton_IR/docs_triton_ascend/03-Ascend-Extensions/10-mem-ops.md` | Triton-Ascend 有内建 **`index_select_simd`（GM→UB，零拷贝，1D index，dim 不能是最后一维）**、`gather_out_to_ub`、`scatter_ub_to_out` | **记录为融核首选的更轻路线**（容器里 `triton 3.2.0` 在）【未确认：本次没写】 |
| `model/model-infer-kvcache/SKILL.md` 的 block/slot 映射节 | `物理块 = block_table[b, 逻辑块]`、`物理 slot = 物理块 × block_size + 块内偏移` | ✅ **逐字采纳**：本次只改了**取数算子**，逻辑/物理映射一个字节没动；实测逐比特相同即为证据 |

---

## 2. ★ 页布局：`.view(-1, width)` 确实不行，但 **flat 化仍然可行**（这是本次最有用的设计发现）

### 2.1 布局事实【实测】（`raw/023-s1-micro.json` 的 `S0_geometry`）

| 平面 | shape | stride（元素） | 页 stride(B) | 页内 payload(B) | `.view(-1,512)` |
|---|---|---|---:|---:|---|
| SWA payload（ratio1 槽） | `[320,128,1,512]` int8 | `(131072,512,512,1)` | 131072 | 65536 | ⛔ `RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces)` |
| SWA scale | `[320,128,1,4]` fp16 | `(65536,4,4,1)` | 131072 | 1024 | ⛔ 同上 |
| long-KV payload（ratio1 槽） | `[320,128,1,512]` int8 | `(83200,512,512,1)` | 83200 | 65536 | ⛔ 同上 |

**为什么**：`reshape_cache()` 用 `block_stride = 槽页大小`（`plan_cache_slots()` 的 `capacity`，
被 **state ring / SWA 别名页**顶住：`max(long_kv+index, aliases)`）建 `as_strided` 视图 ⇒
**页 stride ≠ 本平面每页 payload**，平面之间还有 gap。
⇒ **只要还保留 hybrid 别名（它正是 ×1.135 容量的来源），payload 就不可能 `view(-1, W)`。**【实测+代码事实】

### 2.2 但**行/页都是「线性字节偏移上的连续块」**，所以能拍平【实测】

页 `p` 的行 `r` 恒在字节 `p×page_stride + r×row_stride` 处，且
`row_stride`（512 B）与 `page_stride` 都能被 `chunk = gcd(page_stride, row_stride)` 整除 ⇒
把整块 storage 看成 `[N, chunk]` 的**连续二维张量**（`as_strided`，合法且 `is_contiguous()==True`），
任意行 / 整页都能用**一维 `index_select`** 取出：

```
页粒度（SWA）：page_view = as_strided(plane, (pages, rows_per_page*row_stride), (page_stride,1))
                index_select(page_view, 0, block_table_pages)          ← 索引就是 block table 本身，零索引算术
行粒度（cmp） ：chunk = gcd(page_stride, row_stride)   # ratio1 平面 256 B，SWA 平面 512 B
                index_select(chunk_view, 0, phys*(page_stride//chunk) + offs*(row_stride//chunk) [+ steps])
```

* **不需要改页布局、不需要改 allocation、不需要换语义** ⇒ 任务书第 1 步「首选，改动最小」成立；
  第 2 步（自写 AscendC）**本次不必**走。
* 边界：`as_strided` 的视图长度必须按平面**自身**范围算（最后一页比一个 page_stride 短），
  否则会 `setStorage: out of bounds`（踩过，见 §7 坑①）。

---

## 3. 实现（1 文件，+54 行）：`attention/dsa_v41.py`

补丁：`agents/KV8_gather/kv8-gather.patch`（vs `KV8_swa` 影子包，98 行 diff / 54 行新增）。

| 新增/改动 | 内容 |
|---|---|
| `kv8_page_view(plane)` / `kv8_chunk_view(plane, chunk)` | 两个 `as_strided` 视图（§2.2） |
| `kv8_gather_pages(plane, pages)` | 页粒度一维 `index_select` |
| `kv8_gather_rows(plane, phys, offs)` | 行粒度一维 `index_select` |
| `kv8_ori_plane()` | `sel_i8 = kv_i8[phys]` → `kv8_gather_pages(...)`（payload + scale 各一次） |
| `_kv8_cmp_plane()` | decode 分支 `kv_i8[phys, offset]` → `kv8_gather_rows(...)`；prefill 分支 `kv_i8[pages]` → `kv8_gather_pages(...)` |

**对账**：写入侧（`scatter_cache_sk` / `kv8_store_rows` / `kv8_swa_store`）、spec、页几何、
`seqused_*` / mask 参数、scratch 布局 **一行没改** ⇒ C1–C5 五条红线全部保持
（量化点仍在 scatter；`indexer.update_keys` 之前拿到的仍是 BF16；scale 与 int8 仍同页）。

---

## 4. ★ 性能：带宽标定 + 40 层生产形状

### 4.1 gather 带宽（`raw/023-s5-bw.json`，图内 400 次/replay，扣掉 0.8 µs 空图地板）

| 取数 | 净耗时 | 字节 | **带宽** |
|---|---:|---:|---:|
| SWA payload 页 2D 高级索引（旧） | 31.77 µs | 1.00 MB | **33.0 GB/s** |
| **SWA payload 页 flat `index_select`（新）** | 15.29 µs | 1.00 MB | **68.6 GB/s**（×2.1） |
| SWA payload+scale（旧 / 新） | 46.37 / 20.97 µs | 1.04 MB | 23.0 / **50.8 GB/s** |
| cmp 4096 行 2D 高级索引（旧，payload / 双平面） | 80.17 / 134.27 µs | 2.00 / 2.13 MB | 26.2 / 15.9 GB/s |
| cmp 4096 行 flat `index_select`（新，双平面，含索引构造） | 78.55 µs | 2.13 MB | 27.1 GB/s |
| 同 die 连续 `copy_` 标定（015 §5.1） | — | — | 1161 GB/s |

⇒ **判据「≥500 GB/s」【未达标】**。原因是**这个尺寸下不是带宽场景**：
1 MB 的一次 gather 净耗时 15.3 µs，而空图一次调用只有 0.8 µs ⇒ 是**核函数本身/启动**在吃时间；
对比 015 §5.1 在 **4.29 GB** 尺度上测到 1.2 TB/s —— 尺寸差 3 个量级，**别把那次数字当本场景能力**
（【实测】+【推断】）。

### 4.2 ★ 关键标定：**每个设备算子 ≈ 4–6 µs**（`raw/023-s3-ablate.json`）

| 标定 | 值 |
|---|---|
| 空图 replay 地板（reps=400 / reps=100） | **0.79 / 3.17 µs per call**（⇒ replay 固定开销 ~320 µs/次） |
| `dequant_only`（4–5 个 elementwise 算子，4096×512） | 15.3 µs |
| `cmp_index_math`（~10 个 [8,512] 小算子：`//`、`%`、`gather`、`where`、`arange`…） | 67.2 µs ⇒ **~6.7 µs/算子** |
| 整套 `kv8_ori_plane`（~30 个算子，实测） | 161.8 µs ⇒ **~5 µs/算子** |
| BF16 SWA 整层（含 `npu_sparse_flash_mla`） | 24.5 µs/层 |

⇒ **rebuild 的成本 ≈ 算子个数 × 5 µs**。把 1 个慢算子（33 µs）换成 1 个快算子（15 µs）最多省 ~18 µs，
**这就是 §4.3 只降 20% 的原因**（不是 gather 没换成功）。

### 4.3 40 层生产形状整图（`raw/023-s4-e2e.json`，B=8、真 `npu_sparse_flash_mla`、打乱 block table）

| 图（40 层一图） | SWA-only 层 | 整层（SWA+cmp） |
|---|---:|---:|
| **A** BF16（现状） | 24.5 µs/层 | 44.8 µs/层 |
| **B** int8 long-KV only（SWA 留 BF16） | 24.4 µs/层 | 300.8 → **231.0 µs/层**（打补丁后） |
| **C** int8 双平面，**旧 gather** | **196.2 µs/层** | **481.2 µs/层** |
| **D** int8 双平面，**新 flat gather** | **164.0 µs/层** | **391.7 µs/层** |

| 增量（µs/层） | 旧 | **新** | 降幅 |
|---|---:|---:|---:|
| **SWA 层** | +171.6 | **+139.5** | −19% |
| **cmp 读路径** | +264.8 | **+207.8** | −22% |
| **整层** | +436.4 | **+347.2** | −20% |

**生产步时间外推（40 SWA 层 + 4 源层）**【推断·按实测外推】：
**+7.92 ms/step（+26%）→ +6.41 ms/step（+21%）**。判据 ≤+0.2%（60 µs/step）**仍未达标**。

> 打补丁后的**同一 harness 复测**（`raw/023-s4-e2e-patched.json`，影子包内直接生效、不再靠 monkeypatch）：
> `swa_current = 139.5`、`cmp_current = 207.8`、`full_current = 347.2`、`full_lk_only = 186.5`、
> `step_ms.current = 6.41` ⇒ **补丁在真实调用路径里生效**（旧路径的 168.9/264.8 复现为 139.5/207.8）。

### 4.4 「干脆别做 SWA 量化」这一臂【实测】

`B`（int8 long-KV + **SWA 留 BF16**）：SWA 层增量 **+0.03 µs/层**（= 0），
整层 +186.5 µs/层 ⇒ **4 源层 ≈ +0.75 ms/step（+2.5%）**；
代价是容量 **×1.135 → ×1.032**（`pool_bytes_per_block` 476416 → 524288 B/block，本次实测）。
⇒ **SWA 量化花的 ~5.6 ms/step 只买到 +10% 容量**，在当前形态下是明显的负收益
（与 020 §9 的 P0 结论一致，本次把它量化成 139.5 µs/层这个具体价格）。

---

## 5. ★ 精度：零回退【实测】

方法与 020 相同：真实 spec + 真实分配器建页、真实写入入口填、真实读取入口
（`DeepseekV41EagerAttentionImpl._native_attention` → `npu_sparse_flash_mla`）跑，B=8、topk 512、窗口 128。

| 对照 | 结果 |
|---|---|
| **重建结果逐比特**：新 `kv8_ori_plane` vs 旧（scratch / block table） | **`torch.equal` = True / True** |
| 同：新 `_kv8_cmp_plane` vs 旧（scratch 前 2 页 / 索引张量） | **True / True** |
| 页 gather / 行 gather（payload 与 scale 两个平面，4 组） vs 2D 高级索引 | **4/4 逐比特相同** |
| **真量化器端到端**：fast vs current | **`torch.equal` = True** |
| 真量化器 vs BF16 | `rel_L2 = 5.4331e-3`、`cos = 0.9999857`、`max_abs = 5.55e-4`、`nan_total = 0` |
| long-KV int8 only（SWA BF16） vs BF16 | `rel_L2 = 4.8854e-3` |

⇒ 判据沿用 020：**无损臂逐比特 ✅**（本次以「新 vs 旧重建逐比特」实现，见 §7 坑③）、
**真量化 `rel_L2 ≈ 5.4e-3` ✅（5.4331e-3）**。

---

## 6. 给主代理的建议（按优先级）

| 优先 | 建议 | 依据 |
|---|---|---|
| **P0** | **不要指望「换 gather 算子」能把 KV8 读侧做进判据**：本次已把 2D 高级索引换成 flat `index_select`（×2.1 带宽、−20% 整层），但 rebuild 的成本是**算子个数 × ~5 µs**。要进 ≤60 µs/层必须**融成 1 个 kernel**：`slot → 读 int8 → 反量化 → 写 BF16 scratch`。路线二选一：① AscendC 四文件模板（`ops/torch-ops-profiler/examples/layer_norm_profiler_reference/`）；② **Triton-Ascend 内建 `index_select_simd` / `gather_out_to_ub`**（更轻，容器里 `triton 3.2.0` 已就绪）【未确认】 | §4.2 / §4.3 |
| **P0** | **SWA 量化在当前形态下应默认关闭**（或先做融核再谈）：实测它为 +139.5 µs/层 × 40 = **+5.6 ms/step**，只换 ×1.032→×1.135 容量；而同一条链的 long-KV 读侧即使打补丁也要 +0.75 ms/step（4 源层） | §4.3 / §4.4 |
| P1 | 融核里的 dequant 可以直接用 `npu_anti_quant`（现成 CANN 反量化算子），但必须把 scale 变一维（本次 2D scale 报 `EZ1001`） | §1 |
| P1 | 本次的 `kv8_page_view` / `kv8_chunk_view` 是**零成本**（无布局改动、逐比特等价），建议无论走哪条路都保留：它同时是融核的输入视图定义 | §2 / §3 |

---

## 7. 复现与坑

```bash
# 只用 c1（退出码 75 = 没抢到锁）；影子包 = KV8_swa 影子包 + 本次 dsa_v41.py
ssh A3-node1 'source ~/projects/dsv41-upstream-pr/tools/a3_chip.sh c1 --name kv8g --timeout 900 -- \
  bash -c "cd /work/agents/KV8_gather && export TMPDIR=/work/agents/KV8_gather/tmp \
    PYTHONPATH=/work/agents/KV8_gather/shadow KV8_RAW=/work/agents/KV8_gather/raw && python3 p6_e2e.py"'
```

| 脚本 | 作用 |
|---|---|
| `agents/KV8_gather/p4_gather.py` | 布局探针 + 9 种取数写法对照（含空控制平面） |
| `agents/KV8_gather/p4_fast.py` | flat gather 实现 + 正确性对拍 + 重建微基准 |
| `agents/KV8_gather/p5_ablate.py` | 逐部件消融（两种 reps，扣空图地板） |
| `agents/KV8_gather/p6_e2e.py` | **40 层生产形状整图 + 精度主表**（本次判据来源） |
| `agents/KV8_gather/p7_bw.py` | 最终带宽标定表（§4.1） |

**harness 自身的坑（记录，避免重踩）**：
① `as_strided` 视图长度必须按**平面自身**范围算：`rows = ((P-1)*page_stride + rows_per_page*row_stride)//chunk`，
   用 `P*page_stride//chunk` 会 `setStorage: ... out of bounds`（scale 平面首当其冲）；
② `chunk_plane` 里的 `chunk` 必须同时整除 **page stride 与 row stride**（ratio1 的 int8 页 stride 83200 不是 512 的倍数 ⇒ chunk 只能是 256 ⇒ 每行 2 个 chunk）；
③ 本 harness 的「无损臂」无法成立：`kv8_store_rows` 对 int8 平面**总是**量化（那是设计），
   所以「逐比特」改为在**重建产物层**对拍（scratch / table / gather 输出），端到端只比真量化；
④ 图内微基准必须扣**空图地板**：`reps=20` 时地板 16 µs/call、`reps=100` 时 3.2、`reps=400` 时 0.8
   —— 不扣的话所有 <20 µs 的读数都是地板；
⑤ `torch.index_select` 的 index 必须 int64；`torch.arange` 在图内会各占一个算子（本次 chunk 版就是因为这个亏的：
   页视图版**不需要任何索引算术**，索引就是 block table 本身）。

**纪律**：只用 c1（die 6）；没写 `/tmp`；没碰 `dsv41-a3` / `mooncake-master` / Phy-ID 8–15；
没写 `upstream-v41/`；没动 `dsv41-release/`；跨机全走 coscli。

---

## 8. 交付物

| 东西 | 路径 |
|---|---|
| 增量补丁（1 文件，+54 行，可直接叠在 KV8_swa 影子包上） | `a2/agents/KV8_gather/kv8-gather.patch` |
| 改后的文件 | `a2/agents/KV8_gather/shadow/vllm_ascend/attention/dsa_v41.py` |
| harness（4 个） | `a2/agents/KV8_gather/p4_gather.py`、`p4_fast.py`、`p5_ablate.py`、`p6_e2e.py`、`p7_bw.py` |
| 原始数据 | `a2/logs/raw/023-s1-micro.json`（布局+首轮取数）、`023-s2-fast.json`（正确性+微基准）、`023-s3-ablate.json`（消融）、`023-s4-e2e.json` / `023-s4-e2e-patched.json`（40 层整图+精度）、`023-s5-bw.json`（带宽） |
| A3 副本 | `~/projects/dsv41-upstream-pr/agents/KV8_gather/`（容器内 `/work/agents/KV8_gather/`） |

> 关于 `raw/023-s4-e2e.json`：它被**打补丁后**的那次复测覆盖（两文件内容相同，都是"影子包已打补丁"版）。
> **未打补丁的基线运行**（§4.3 表里 C 那一列的 481.2 / 196.2 与旧增量 436.4 / 171.6 / 264.8、step 7.922 ms）
> 保存在 `raw/023-run.log` 的 stdout 里。
