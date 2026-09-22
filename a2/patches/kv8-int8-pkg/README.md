# `kv8-int8-pkg/` —— 档 C / 档 D（int8 KV）的**整文件挂载件**

> ★★★ **为什么有这个目录**：2026-09-22 15:2x 主代理在核对"发布包能不能起档 C"时发现——
> **档 C/D 一共需要 7 个挂载件，而发布包里原本只有 1 个（`kv8-graphsafe/dsa_v41.py`）**。
> ⇒ **A2 拿到发布包，起不了档 C/D**。这与 `055` 的 shadow-pkg 缺失、`0004` 的 ②c 补丁缺失**是同一类交付缺口**。
> 本目录把缺的 **6 个**补齐（第 7 个仍在 `../kv8-graphsafe/dsa_v41.py`）。

---

## 1. 挂载清单（**7 个**，缺一不可）

| # | 源文件 | md5 | 容器内目标路径 |
|---|---|---|---|
| 1 | `vllm_ascend/core/deepseek_v41.py` | **`9db8e27c01fb8d17811de17680b4a0d8`** | `/vllm-workspace/vllm-ascend/vllm_ascend/core/deepseek_v41.py` |
| 2 | `vllm_ascend/core/kv_cache_interface.py` | `7e17f7cae054f0b2339b41b7c3642f0e` | `…/vllm_ascend/core/kv_cache_interface.py` |
| 3 | ★ **`vllm_ascend/models/deepseek_v41/model.py`**（**合并版**，见 §2bis） | **`c4b70d006a24e4493d07d9822f75aa8b`** | `…/vllm_ascend/models/deepseek_v41/model.py` |
| 4 | `vllm_ascend/models/deepseek_v41/compressor.py` | `8a2be008ef405ab681728a275bcb5f77` | `…/vllm_ascend/models/deepseek_v41/compressor.py` |
| 5 | `vllm_ascend/ops/triton/compressor/compressor_triton.py` | `9362e72e3ea12e8344c4485104eab837` | `…/vllm_ascend/ops/triton/compressor/compressor_triton.py` |
| 6 | `vllm_ascend/attention/kv8_prefill_triton.py` | `796d0ff6eda03716f31c9994b8d8b221` | `…/vllm_ascend/attention/kv8_prefill_triton.py` |
| 7 | ★ **`../kv8-graphsafe/dsa_v41.py`**（**不在本目录**） | **`94aeebb757d6d5708268754481a05e0a`** | `…/vllm_ascend/attention/dsa_v41.py` |

★ **md5 的来源**：前 6 个与 `R_8card_int8` 的 **`arm.out` 挂载台账逐字相同**（即 8 卡实测时真正挂进去的那几份）；
第 7 个是 `S_graphfix` 的图安全版（**档 C/D 在 `94aeebb7` 上实测通过的那一份**）。

---

## 2. ★★ 第 1 个文件为什么是 `9db8e27c` 而不是包里的 `b9ae8151`（**这里有个坑**）

`X_integrate/pkg-kv8pf` 里的 `core/deepseek_v41.py` 是 **`b9ae8151`**，它的槽位容量是：
```python
capacity = max(kv_bytes + index_bytes, *(sum(_cache_plane_sizes(specs[n])) for n in aliases))
#                              ↑ ★ 没有 draft 项
```
⇒ 而真权重有 **13 个组（含 draft 组）** ⇒ `draft=131,072 > capacity=66,560`（档 D）
⇒ **`plan_cache_slots` 直接 raise**：`Aurora DSpark geometry must match target SWA and fit its existing slot`。
★ **这正是 `050` 记录的那 6 条"未打 draft-aware 补丁的胳膊 raises"**。

`R_8card_int8` 的 `patched/deepseek_v41_slots.py`（**`9db8e27c`**）才是修好的那版：
```python
capacity = max(kv_bytes + index_bytes, _alias_max, _draft_size)   # ★ 含 draft
```
⇒ **本目录用的是它**（差 32 行；`grep -c "draft-aware"` = 1）。

---

## 2bis. ★★★ 2026-09-22 16:3x：`model.py` 必须是**合并版**（生产 + KV8），否则会**打掉生产补丁**

### 我最初发错的那份（`fd7ff753`）为什么错
`fd7ff753` = `X_integrate/pkg-kv8pf` 的那份 = **镜像原版 + KV8 那 3 处接线**。
但 **A2 的生产 `model.py` 不是镜像原版** —— 发布包自己的 `patches/files/model.py` 是 **`0a7dfb21`**
（正是 A2 启动日志 `BUILD_INFO` 里那行），它**比镜像多 5 处生产改动**。
⇒ 若直接挂 `fd7ff753`，**那 5 处生产改动会被静默打掉**；
而在 `PATCH_MODE=mount` 下，生产块**也**挂同一个容器路径 ⇒ **Duplicate mount point**（`048` 踩过）。

### 修法：**合并版**（`c4b70d00`）
用 `R_8card_int8/patch/merge_model.py` 把它合出来 —— 它的方法**自带自证**：
```
① 从 (镜像版 d22eec4c → pkg-kv8pf 版 6a1b7885) 现算统一 diff（n=3 上下文）
② ★ 同一条 diff 先回放到「镜像版」上，必须**逐字节得到 pkg-kv8pf 版**（自证）
③ 再把同一条 diff 应用到**生产版**（0a7dfb21）上
```
**产物**：`c4b70d006a24e4493d07d9822f75aa8b`（1333 行）
```
[merge] ★ 自证①：diff 回放到镜像版 ⇒ 与 pkg-kv8pf 版逐字节相同 ✅
[merge] ★ 生产版 3 个锚点全部唯一命中 ✅
[merge]   OK  import 了 swa_plane_kwargs / long_kv_plane_kwargs
[merge]   OK  get_kv_cache_spec 用了 swa_plane_kwargs() / SWASpec 带 scale_dim
[merge]   OK  long-KV 用 long_kv_plane_kwargs()
[merge] AST OK：7763 个节点，1333 行
[merge] 生产版 → 产物的差异行数 = 9（期望 = 3 个 hunk 的 ± 行数之和）
[merge] 输入 md5: img=d22eec4c kv8=6a1b7885 prod=0a7dfb21
```
⇒ ★ 与生产版的差异**恰好 9 行 = KV8 那 3 处接线**（其余生产改动**原样保留**）。

### ★★ 顺带加的一道门（`make_shadow_pkg.sh`）
若 `PATCH_MODE=mount` ⇒ 生产块**也会**挂 `model.py` ⇒ 两个 `-v` 同路径 ⇒ docker 报
`Duplicate mount point`。⇒ 生成器现在**提前 die**，把失败变响亮、可读（而不是让 docker 抛原始错）。

### ⚠️ 剩下的 5 个件**不冲突**（已逐条核过）
生产侧的挂载清单里只有 `model.py` 与我的 7 个件重名；
`core/deepseek_v41.py` / `core/kv_cache_interface.py` / `compressor.py` /
`compressor_triton.py` / `kv8_prefill_triton.py` / `dsa_v41.py` **都不在生产清单里**。

---

## 3. 怎么用

**正常路径**：不用手动挂 —— `a2/scripts/make_shadow_pkg.sh` 在检测到 `A2_KV8=1` 或
`A2_KV8_SWA=1` 时，会把**这 7 个文件**一起挂进去（见生成器里的 `A2_KV8_INT8_PKG` 块）。

**手动挂载**（若要自己起）：
```bash
P=$PWD/a2/publish/kv8-int8-pkg/vllm_ascend
V=/vllm-workspace/vllm-ascend/vllm_ascend
docker run ... \
  -v $P/core/deepseek_v41.py:$V/core/deepseek_v41.py:ro \
  -v $P/core/kv_cache_interface.py:$V/core/kv_cache_interface.py:ro \
  -v $P/models/deepseek_v41/model.py:$V/models/deepseek_v41/model.py:ro \
  -v $P/models/deepseek_v41/compressor.py:$V/models/deepseek_v41/compressor.py:ro \
  -v $P/ops/triton/compressor/compressor_triton.py:$V/ops/triton/compressor/compressor_triton.py:ro \
  -v $P/attention/kv8_prefill_triton.py:$V/attention/kv8_prefill_triton.py:ro \
  -v a2/publish/kv8-graphsafe/dsa_v41.py:$V/attention/dsa_v41.py:ro \
  ...
```

---

## 4. 证据（这些 md5 在哪些臂上跑过）

| 臂 | 跑过的东西 | 结果 |
|---|---|---|
| `sg-c-c-graph-b`（8 卡真权重，档 C，图模式） | 这 7 个件 + `94aeebb7` 的 dsa | ✅ 捕获 9/9、`EE1016=0`、容量 427,643、四条判据、`fill`/`replay1` sha 与旧 md5 轮**逐字相同** |
| `sg-c-d-graph`（8 卡真权重，档 D，图模式） | 同上 | ✅ 捕获 9/9、容量 485,610、**`replay1 sha` 与同几何 eager 逐字节相同** |
| `ddi-*`（单 die，②a 试验） | 同样的 6 个件 + dsa | ✅ 起服/图捕获成功（②a 的失败是另一回事，见 `logs/056`） |

★ 这些 md5 也出现在 `R_8card_int8` 的 `arm.out` **挂载台账**里（可逐条复核）。

---

## 4bis. ★★★ 2026-09-22 15:4x：【实测】发布包的 ②c 链路 **= A3 上实测过的链路**（逐字节）

补完本目录后，主代理做了一件**此前没人做过**的验证：**在发布包内部件之间**跑一遍 ②c 的生成器，
再把产物与 A3 上**实测通过**的那份对 md5。

```bash
# 输入 = 发布包的 9db8e27c + 0004（默认 --variant clean = 生产形状）
python3 a2/publish/0004-draft-block64.patch.py \
    --core a2/publish/kv8-int8-pkg/vllm_ascend/core/deepseek_v41.py \
    --dspark <镜像原版 dspark.py>  --out-dir out/
```

| 产物 | 我打出来的（**发布包**） | A3 上实测用的（`C2_draft64/pkg/dfix2c`） | 判定 |
|---|---|---|---|
| `core/deepseek_v41.py` | **`f9b1b560fd54abe69531cf2abd4efed0`** | **`f9b1b560fd54abe69531cf2abd4efed0`** | ★★★ **逐字节相同** |
| `models/.../dspark.py` | `067f37f19696448fe611dd15518f2386` | `0fd33532abcfcf7c3b43f8ed89d19881` | ★ **差 13 行，全部是 `[C2-2C-PROBE v1]` 的只读 trace** |
| `dspark.py` 的**输入** | `40b7797fac499f6dd31d0022343ed131` | `40b7797fac499f6dd31d0022343ed131`（= `Ci8fix` 基础包） | ★ **逐字节相同** |

⇒ ★★★ **三条合起来 = 「发布包里打出来的 ②c 产物，与 A3 上实测通过的那份是同一份」**
（core 逐字相同；dspark 只多一个只读探针）。
★ 这也**顺带回答了 `054` 里那个口径问题**：C2 虽然写了「`core-only` 是给 8 卡测试用的变体」，
但**实际产出的 `f9b1b560` 与我用 `clean`（生产形状）打出来的一致**
⇒ **实测用的就是生产形状**，两者不是「一个测试变体、一个生产形状」的关系。

### ⚠️ 两条给 A2 上线的提醒（从这次验证里发现的）
1. ★ **`dspark.py` 的输入必须是「镜像原版」**（`40b7797f`）；
   若误用别的版本，生成器会在 `need()` 上 **fail-closed**（好）；
2. ⚠️ **生成器对已打过的 dspark 是 `sys.exit(...)` 而不是静默跳过**
   ⇒ **重跑会以非零退出**。A2 上若因别的原因重跑，**看到非零退出先确认是不是「已经打过」**，
   不要误判成失败（`grep -c 'TDC-2c' <dspark.py>` 为 1 即是已打过）。
---

## 5. ⚠️ 诚实边界

1. ★ **本目录的 6 个文件是从 A3 的 `X_integrate/pkg-kv8pf` 原样取来的**（走 cos-xfer），
   md5 与 `arm.out` 台账**逐字相同**；**不是**我重新生成的；
2. ⚠️ **它们的"上游基底版本"没有单独记录** —— 它们是从镜像里 `docker cp` 出来再改的
   （`X_integrate` 的做法），所以**无法用 `git apply` 复现**，只能整文件挂载（本项目既有做法，与 `0001`/`0002` 一致）；
3. ★ **从发布包起档 C/D 这件事，本身还没有在 A2 上端到端验过** ——
   本次只做到"**文件齐了 + 挂载路径写进生成器 + 本地 dry-run 通过**"；
   ⇒ **A2 上第一次起档 C，仍按 `logs/052` 的探测 → `055` 的 shadow → 本包的挂载顺序走**，并把 `serve.log` 里的
   `[P1_pinned]` / `[D2_offload]` / `[R8-SLOTS]` 三处读数贴回来。
