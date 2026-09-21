# #16285 × `perf/rope-fused-index-select` 冲突面分析

> 2026-09-21｜A 号代理（rope 分支收尾）｜所有结论都可在本机复跑（§6 给命令）

## 0. 结论（TL;DR）

| 问题 | 结论 |
|---|---|
| 逻辑冲突？ | **不冲突**。它改**输出长度语义**（多输出多少行、多出来的行填什么），我们改**行怎么被选中/写出**，两者正交 |
| patch 冲突？ | **会撞**，但只有 `rope_dsv4.py` 的 `if draft_index is None:` 这一段（4 处 `torch.gather` 调用点） |
| 它先合入怎么办？ | 保留它的 `output_len` / `fill_` / 返回切片，把 4 处 `torch.gather` 换成 `_rope_gather_rows(...)`；约四行改动 |
| 已实测验证？ | ✅ 合并版跑我们的 12 个单测 **13 passed**；6 组配置下 base 与合并版输出 `torch.equal` 全等 |
| 额外背景 | 它的 base 落后最新 main **53 个提交**，main 在 `9dc67045` revert 了 V4.1 框架支持（#16544）⇒ 它必须先自己 rebase；`mergeable_state=dirty` 主要来自这里 |

---

## 1. #16285 在 `vllm_ascend/ops/rope_dsv4.py` 里改了什么（逐 hunk）

来源：`gh api repos/vllm-project/vllm-ascend/pulls/16285/files`（等价于
`git fetch upstream refs/pull/16285/head:refs/remotes/pr/16285`，
head = `e2bf9343`，base = `fe167d93`）。该文件 +44/−5，共 3 个 hunk：

### hunk 1：`@@ -84,6 +84,7 @@` —— 新参数

```python
 def get_cos_and_sin_dsa(
     positions: torch.Tensor | dict[str, torch.Tensor],
     use_cache: bool = False,
     draft_index: int | None = None,
+    cached_output_len: int | None = None,
 ):
```

默认 `None` ⇒ 老行为不变，纯新增。

### hunk 2：`@@ -115,6 +116,13 @@` —— 输出长度与参数校验

```python
                 buf_cos, buf_sin = group_buffers
                 num_tokens = pos_tensor.size(0)
+                output_len = num_tokens if cached_output_len is None else cached_output_len
+                if output_len < num_tokens:
+                    raise ValueError(
+                        "DSA RoPE cached_output_len cannot be smaller than "
+                        "the number of positions: "
+                        f"output_len={output_len}, num_tokens={num_tokens}"
+                    )
```

`output_len` 是**返回张量的行数**（不是写入行数）；小于 `num_tokens` 直接报错。

### hunk 3：`@@ -131,16 +139,47 @@` —— 索引构造没动，两个分支的语义改了

这是和我们**重叠的那一段**。逐点拆开：

1. `gather_idx` 的构造**一个字都没改**（仍是
   `pos_tensor.to(torch.long).reshape(-1,1,1,1).expand(num_tokens,1,1,D)`）——
   即**它没有动索引方式**。
2. `draft_index is None` 分支：
   * 新增 `output_len > buf_cos.shape[0]` 的容量校验（报 `runtime buffer is too small`）；
   * 两处 `torch.gather(..., out=buf_cos[:num_tokens])` **保持原样**；
   * 新增 `if output_len > num_tokens:` ⇒ `buf_cos[num_tokens:output_len].fill_(1)`、
     `buf_sin[num_tokens:output_len].zero_()`（pad 行语义：cos=1、sin=0）；
   * 返回值从 `buf_cos[:num_tokens]` 变成 `buf_cos[:output_len]`。
3. `draft_index is not None` 分支：
   * 把 `buf_cos[draft_index - 1]` / `buf_sin[draft_index - 1]` 提成局部变量
     `draft_cos` / `draft_sin`（可读性 + 复用）；
   * 同样的容量校验、同样的 pad 行 `fill_`、同样的返回切片 `[:output_len]`；
   * 原来的两处 `torch.gather(..., out=buf_cos[draft_index-1][:num_tokens])`
     因为换成了局部变量而被**重新排版成多行参数**（这是文本冲突的直接来源）。

> 修正一处口径：是 **2 处 `torch.gather` 语句 × 2（cos/sin）= 4 个调用行**被触碰，
> 不是 2 行。语句层面仍是 cos/sin 各一处。

### 调用侧（同一 PR 的 `vllm_ascend/attention/context_parallel/dsa_cp.py`）

`build_for_drafting()` 里新算了一个 TP 对齐后的输出长度并传进去：

```python
 tp_size = get_tp_group().world_size
 rope_output_len = cdiv(num_input_tokens, tp_size) * tp_size
 cos, sin = get_cos_and_sin_dsa(
     input_positions, use_cache=True, draft_index=draft_index,
     cached_output_len=rope_output_len,
 )
```

⇒ 它的动机是 **DSA-CP 下 draft 步的 RoPE 输出要按 TP 对齐补行**，
和"用哪个算子选行"完全无关。

---

## 2. 逻辑是否冲突：**不冲突**（独立判断）

### 2.1 为什么正交（源码级）

| 维度 | #16285 | 本分支 |
|---|---|---|
| 关心的是 | 返回张量**有多少行**、多出来的行**填什么** | 这些行**用哪个算子选出来、写到哪里** |
| 落点 | `output_len` 计算、容量校验、`fill_/zero_`、`batch_result[...] = buf[:output_len]` | `gather_idx` 是否构造、`torch.gather` ↔ `torch.index_select` |
| 是否改索引方式 | ❌ 没改 | ✅ 就是改这个 |

唯一的接触面是那 4 个 `torch.gather` 调用行：我们的改动**只是把"选哪一行并写进
`out=`"这一步换成一个等价的 `index_select`**，写入目标切片（`buf_cos[:num_tokens]`、
`draft_cos[:num_tokens]`）和写入行数（`num_tokens`）都不变；
而它的 `fill_/zero_` 作用在 `[num_tokens:output_len]`，与我们的写入区间**不相交**。
所以两者可以同时成立：**先按 token 行写入，再补 pad 行，最后返回更长的切片**。

### 2.2 实证（本机跑过，不是推理）

把它的 rope hunk 单独 apply 到最新 `upstream/main`，再 cherry-pick 我们的 commit：

* 冲突**只有一段**（`if draft_index is None:` 那段），手工解决后：
  * 合并版文件 vs 我们的文件逐行 diff **恰好等于它自己的 patch**（纯增量，无一处
    需要取舍）⇒ 这就是"逻辑不冲突"的最强形式；
  * 合并版跑 `tests/ut/ops/test_rope_proxy.py`：**13 passed**；
  * `pr/PR16285-rope-composition-check.py`（CPU harness，7 组用例）：
    6 组既有配置下 base 与合并版输出 `torch.equal` 全等，且都等于
    `full_rope[pos]` 参考值；
  * 合并版新增的 `cached_output_len` 语义全部成立：返回长度 = `output_len`、
    token 行等于查表结果、pad 行 `cos=1 / sin=0`、缓冲区地址不变、未触碰区域保留原填充值。

---

## 3. 若 #16285 先合入，我们的具体做法

### 3.1 步骤（可直接照抄）

```bash
cd ~/projects/dsv41/upstream-v41/vllm-ascend-fork
git fetch upstream main
git fetch upstream refs/pull/16285/head:refs/remotes/pr/16285   # 若它已合入则跳过

# 它合入后，我们的分支只需换基
git rebase upstream/main            # 我们的 commit 只有 1 个，冲突点固定
# 解决 vllm_ascend/ops/rope_dsv4.py 的 4 处 gather 调用点（见 3.2），然后：
git add vllm_ascend/ops/rope_dsv4.py && git rebase --continue

# 复核
python -m pytest tests/ut/ops/test_rope_proxy.py -v      # 期望 13 passed
/tmp/ruffvenv/bin/ruff check vllm_ascend/ops/rope_dsv4.py && \
/tmp/ruffvenv/bin/ruff format --check vllm_ascend/ops/rope_dsv4.py

git push --force-with-lease origin perf/rope-fused-index-select
```

### 3.2 冲突解决方式（4 行）

保留它的 `output_len` / 容量校验 / `fill_` / `[:output_len]` 返回切片，
把 4 处 `torch.gather` 换成我们的 `_rope_gather_rows(...)`：

```python
 # non-draft 分支
-torch.gather(full_rope_cos, 0, gather_idx, out=buf_cos[:num_tokens])
-torch.gather(full_rope_sin, 0, gather_idx, out=buf_sin[:num_tokens])
+_rope_gather_rows(full_rope_cos, pos_tensor, gather_idx, buf_cos[:num_tokens])
+_rope_gather_rows(full_rope_sin, pos_tensor, gather_idx, buf_sin[:num_tokens])

 # draft 分支（它已改成局部变量 draft_cos/draft_sin）
-torch.gather(full_rope_cos, 0, gather_idx, out=draft_cos[:num_tokens])
-torch.gather(full_rope_sin, 0, gather_idx, out=draft_sin[:num_tokens])
+_rope_gather_rows(full_rope_cos, pos_tensor, gather_idx, draft_cos[:num_tokens])
+_rope_gather_rows(full_rope_sin, pos_tensor, gather_idx, draft_sin[:num_tokens])
```

完整解决结果见 **`pr/PR16285-rope-resolution.patch`**（apply 到"main + #16285 rope hunk"
之上即得合并版文件）。

### 3.3 合并后建议补的一个测试

它的 `cached_output_len` 语义目前在 **`tests/ut/ops/test_rope_proxy.py` 里没有覆盖**
（它自己的测试加在 `tests/ut/spec_decode/` 一侧）。我们 rebase 时可以顺手加一个
`output_len > num_tokens` 的用例（pad 行 cos=1/sin=0 + 返回长度），
这样两个 PR 的语义在同一份 UT 里都有守卫 —— 对 reviewer 是加分项。

---

## 4. 额外发现：#16285 为什么是 `dirty`（对我们没威胁，但要知道）

实测：

```bash
git merge-base refs/remotes/pr/16285 HEAD   # → fe167d93（它的 base）
git rev-list --count fe167d93..HEAD         # → 53
git log --oneline -S get_full_cos_and_sin_dsa_for_layer -- vllm_ascend/ops/rope_dsv4.py
#   → 9dc67045 Revert "[Feature] Add DeepSeek V4.1 framework support and Engram host offload (#16544)" (#16905)
```

* 它的 base 落后最新 main **53 个提交**，其间 main 用 `9dc67045` revert 了 #16544
  （V4.1 框架支持 —— 就是准备重新合入的 #16925）。
* 把它 merge 进最新 main，冲突文件**只有两个**：
  `tests/e2e/pull_request/four_card/spec_decode/test_dspark_deepseekv4.py`、
  `vllm_ascend/spec_decode/dspark_proposer.py`。
* `vllm_ascend/ops/rope_dsv4.py` 能**自动合并**（revert 删掉的函数与 #16285 改的区域
  不同，所以 merge 结果不会把被 revert 的代码复活）。

⇒ 它合入前必然要自己 rebase 一次；我们只需要在它落地后跟一次 §3.1。

---

## 5. 复现命令

```bash
cd ~/projects/dsv41/upstream-v41/vllm-ascend-fork

# 1) 抓它的代码
git fetch upstream refs/pull/16285/head:refs/remotes/pr/16285
gh api repos/vllm-project/vllm-ascend/pulls/16285/files --jq '.[]|.filename'

# 2) 单文件 hunk（只取 rope_dsv4.py 那 3 个 hunk）
gh api repos/vllm-project/vllm-ascend/pulls/16285 -H 'Accept: application/vnd.github.v3.patch' > /tmp/pr16285/patch

# 3) 模拟"它先合入"（临时 worktree，不动主检出一根手指）
git worktree add --detach /tmp/wt-rope16285 upstream/main
( cd /tmp/wt-rope16285 && git apply -3 /tmp/pr16285/rope_only.patch && git commit -qam sim )
( cd /tmp/wt-rope16285 && git cherry-pick 90c5336c )      # 冲突：仅 rope_dsv4.py

# 4) 合并版语义复核（CPU harness，无需 NPU）
/tmp/rope_ut_venv/bin/python ~/projects/dsv41/upstream-v41/pr/PR16285-rope-composition-check.py

# 5) 清单
git worktree remove --force /tmp/wt-rope16285
```

## 附录：与 #16925 的 Engram host-mapped 路径的关系

> 记于 2026-09-21。本节写的是一条**可检验假设**，**不是断言** ——
> 我们没有在 #16925 的代码上复现过任何失败，下面的"可能/或许"都是字面意思。

### A.1 新证据：capability 查询报"不支持"，但设备直读实际是通的

同一台机器（设备 PCI `19e5:d803`），用我们自己已在 A2/A3 用过的探针
（`lite-port/engram-dev/probe_a2_hostmap.py`）实测：

```
[probe] capability AIC   : rc=207000 -> NOT_SUPPORTED
[probe] capability AIV   : rc=207000 -> NOT_SUPPORTED
[probe] HostMappedSafetensors 映射成功: (4096, 256) torch.int8 npu:0
[probe] device index_select 4096 行（设备读 host DRAM）逐字节一致=True
[probe] 结论：本机支持 Engram device-index
```

`207000` = `ACL_ERROR_RT_FEATURE_NOT_SUPPORT`。⇒ **`aclrtHostMemMapCapabilities`
这个能力查询不可信，不能拿它做门控**：它报"不支持"的机器上，设备算子照样逐字节读对了。

（旁注：另一个新写的 `probe_engram_hostmap.py` 在同一台机器上得出相反结论
"NOT SUPPORTED"，根因是它用了**非生产形态**的 host 内存 + 非生产读回方式，
详见 `upstream-v41/PROBE-VERDICT.md`。它测出的那组 `capability` 数值和信息量在于
"rc=207000"，但它的**判定不能采信**。）

### A.2 真正决定成败的是 host 内存的形态（不是 capability）

| 形态 | 读回结果 |
|---|---|
| **file-backed 可写 mmap** + `aclrtHostRegister(MAPPED)` + **DLPack**（`from_dlpack`，先注册 NPU device type） | ✅ 通，逐字节一致 |
| 匿名 mmap（`MAP_PRIVATE \| MAP_ANONYMOUS`）或 `aclrtMallocHost` pinned + **裸指针**读 | ❌ 读回不一致 |

⚠️ 第二行是**混淆观测**：它同时改了两件事（内存形态 + 拿/读地址的方式），
所以它**不能**单独证明"pinned 一定不行"——这正是下面 A.5 要把假设拆成两条的原因。

### A.3 我们记录里的一条已知反例（pinned）

> *"`offload.get_dva(pinned_ptr)` returns 0 and an AIV de-referencing a
> **registered pinned** address dies"* —— `507035 MTE invalid GM address`
>
> 出处：`lite-port/engram-dev/probe_a2_hostmap.py:18-19`、
> `engram_ref/wtgraph/docs/A2_VS_A3_DIFF.md` §5

### A.4 为什么这条会指向 #16925

`#16925 [Feature] Add DeepSeek V4.1 framework support and Engram host offload`
（open，作者 `pgzddxx`，head `382dc9289`，`mergeable_state=dirty`）的 host 侧实现是
`vllm_ascend/models/deepseek_v41/engram/npu.py::HostUvaBuffer`（`:125-160`）。
下面是它的调用序列（**精简版摘录**：省去了 `ctypes.` 前缀与 `dtype`/`device` 参数，
不是逐字原文，逐字内容见 `:135-152`）：

```python
rc = self.lib.aclrtMallocHost(byref(pointer), size, 0)            # pinned 分配
rc = self.lib.aclrtHostRegisterV2(pointer, size,
        ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED)                # 0x2 | 0x10000000
rc = self.lib.aclrtHostGetDevicePointer(pointer, byref(address), 0)
self.ptrs = torch.tensor([address + start * row_bytes, ...])      # 供设备内核解引用
```

即：**`aclrtMallocHost`（pinned）内存 + `MAPPED|PINNED` 注册 + AIV/Triton 内核解引用**
（`_engram_host_uva_gather_dequant_kernel` 直接把该地址转成 `tl.pointer_type(int8)` 读）。
这落在 A.2 表格**第二行**、A.3 反例的形态一侧。

另外两点事实（都是实测，不是推测）：

* 它**不查 capability** —— 在 `refs/remotes/pr/16925` 上 grep
  `aclrtHostMemMapCapabilities` **零命中**，门控只有 `EngramConfig.cpu_offload`；
* 注册失败**直接 `RuntimeError`，没有回退**到设备表。

### A.5 可检验假设（拆成两条，别混）

* **H1（内存形态）**：在这类机器上，**pinned host 内存**可能不能像 file-backed mmap
  那样被设备内核直接读；若成立，则 #16925 的 offload 路径可能读回错值或直接报错。
* **H2（地址发布/消费方式，替代解释）**：失败可能根本不在 `PINNED` 标志，而在
  **怎么拿到设备地址、以及谁来解引用**。我们的失败样本走的是 `offload.get_dva()` / 裸指针；
  我们**通过**的样本是 `host_register(MAPPED)` + DLPack 包成 device tensor、由框架算子读；
  而 #16925 走的是 `aclrtHostGetDevicePointer()` + **Triton 内核里把地址转成指针直接解引用**。
  三者的 API 组合各不相同，所以我们的反例**不能直接搬到它头上**。

⇒ 两条假设对"要不要改 #16925"的建议**相反**，必须用同一个实验分开：

1. 在**这台机器**上原样实例化 #16925 自己的 `HostUvaBuffer`（保持它的调用序列），
   跑一次 4096 行的 `index_select` / `gather_dequantize_host_uva` **逐字节比对**；
2. 把 `MAPPED|PINNED` 改成 **`MAPPED` only** 再跑一次；
3. 把 `aclrtMallocHost` 换成 **file-backed mmap**（生产形态）再跑一次。

判读规则：

* `(1)` 挂而 `(2)`/`(3)` 通 ⇒ **H1 成立**：应建议他们把 host 表改成
  file-backed + `MAPPED`，或至少在注册/首查失败时回退设备表；
* 三者都挂 ⇒ 问题不在 `PINNED`（**H2 胜出**），要往地址发布或内核侧去找；
* 三者都通 ⇒ 我们的反例与他们的实现无关，这条假设作废。

**给上游的措辞（重要）**：不要写"你们的 PINNED 路径会挂"。
应写成：

> On a machine reporting `aclrtHostMemMapCapabilities` = `207000 NOT_SUPPORTED`
> (PCI `19e5:d803`), a file-backed mmap registered with `aclrtHostRegister(MAPPED)`
> is readable by a device `index_select` byte-for-byte, while we have a recorded
> counter-example for *registered pinned* addresses (`507035`). We have not
> reproduced this against your implementation, so this is a hypothesis worth
> three comparison runs, not a claim.

—— 这是**给他们一条判据**，不是对他们实现的判决。

---

## 6. 附件

| 文件 | 内容 |
|---|---|
| `pr/PR16285-rope-resolution.patch` | "main + #16285 rope hunk" 之上重新落地我们 commit 的完整 diff（冲突解决结果） |
| `pr/PR16285-rope-merged-dsv4.py` | 同上 patch 应用后的合并版 `rope_dsv4.py` 快照（可直接对照阅读） |
| `pr/PR16285-rope-composition-check.py` | CPU harness：base vs 合并版逐路径比对 + `cached_output_len` 语义检查 |
| `/tmp/pr16285/` | 本次分析的一次性中间产物（可删）|
