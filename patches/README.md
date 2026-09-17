# patches/ —— 发布用的三类补丁资产

本目录同时提供**两种等价形态**，按场景选一种（不要混用后重复打）：

| 形态 | 路径 | 用途 | 已验证 |
|---|---|---|---|
| **① 整文件（逐字节）** | `patches/files/**` | 烘焙进镜像 / bind-mount。与基线的 commit 无关，最稳 | ✅ A2 + A3 实机跑通 |
| **② git 历史系列** | `patches/vllm-ascend/`、`patches/vllm/`、`patches/msmodelslim/` | `git am` 到干净 checkout，**保留每个提交的信息**（作者/日期/说明），用于评审、向上游提交、独立仓库发布 | ✅ 已在 base commit 上 `git am` + 逐字节比对通过 |

两种形态的一致性已被机器验证：把 ② 打到 base 上得到的 12 个文件，与 ① 的
`MD5SUMS` **逐字节相同**（见 `_build_series/verify_series.sh` 的 `VERIFY: ALL PASS`）。

---

## 1. 三个仓库与基线 commit

| 仓库 | 目录 | base commit | 系列 | 说明 |
|---|---|---|---|---|
| **vllm-ascend（V4.1 线）** | `patches/vllm-ascend/` | `46856f89e79c3011401e33663c60da37cd486d53` | 8 个补丁 | 全部性能优化都在这里 |
| **vllm**（core） | `patches/vllm/` | `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` | 1 个补丁 | 只需 `VLLM_ADMISSION_GATE` |
| **msmodelslim**（量化） | `patches/msmodelslim/` | `92e219fa9565a5bad84d90474a27bb11524d691c` | 2 个补丁 | 只在**重新量化**时才需要 |

### 1.1 怎么拿到这些基线

```bash
# ① vllm-ascend（V4.1 线）—— 公开仓库，含 deepseek_v41/ 整套代码
git clone https://github.com/GDzhu01/vllm-ascend-v41-private.git
cd vllm-ascend-v41-private && git checkout 46856f89e79c3011401e33663c60da37cd486d53

# ② vllm core —— 官方上游
git clone https://github.com/vllm-project/vllm.git
cd vllm && git checkout 6e448d0ea9bf3d88d898b65449ca6dc2aec170ac

# ③ msmodelslim（只在重新量化时需要）
git clone https://gitcode.com/Ascend/msmodelslim.git
cd msmodelslim && git checkout 92e219fa9565a5bad84d90474a27bb11524d691c
```

三个 commit 均已验证**匿名可取**（raw / ls-remote 均 200，无需凭据）。

### 1.2 为什么 vllm-ascend 用 `vllm-ascend-v41-private` 而不是 `vllm-project/vllm-ascend`

DeepSeek-V4.1 的模型代码（`vllm_ascend/models/deepseek_v41/`、Engram、DSpark、Aurora 等）
**不在** `vllm-project/vllm-ascend` 上 —— 该仓 latest main 只有 `models/deepseek_v4/`，
且全库只有一个 `ops/triton/engram_int8.py`，没有 `deepseek_v41/` 目录。

V4.1 这条线在 `GDzhu01/vllm-ascend-v41-private`（公开可读，Apache-2.0），
我们从公开 vllm-ascend 历史 `5a8e3200f`（2026-09-05）分出，两侧共同祖先是 `46856f89e`。

> ⚠️ 该仓名为 `private` 但**当前匿名可读**，且是**个人仓**。
> 若后续被设为私有或删除，补丁系列的基线会失效 —— 建议迁到组织仓；
> 在此之前，**整文件形态（`patches/files/`）不受影响**，可直接用于烘焙/挂载。

### 1.3 与 `GDzhu01` main（`8727bd4e`）的兼容性

main 比本基线多了 SP/CP 相关提交（`b4274f9f`、`e0bc6030`、`1933f86c`、`08843076`、
`1321d086`、`8727bd4e`）。若你基于 **main** 而不是本基线，实测：

| 补丁 | 对 main 的可应用性 |
|---|---|
| 0001 – 0006 | ✅ 直接 `git apply` 干净 |
| 0008 | ✅ `git apply -3` 三方合并成功 |
| 0007 | ⚠️ `engram_hbm.py` / `engram_plan_kernel.py` 干净；`model.py` 有**一处 import 块冲突**（两边各自插入），手工保留两侧即可 |

本系列的性能数字是在**基线 `46856f89e`** 上实测的；基于 main 的组合**未做端到端复测**。

## 2. 怎么用（推荐路径）

```bash
# 在容器里（镜像已含两个仓库，且都在 base commit 上）
cd /vllm-workspace/vllm-ascend
git am --keep-cr /opt/dsv41/patches/vllm-ascend/*.patch

cd /vllm-workspace/vllm
git am --keep-cr /opt/dsv41/patches/vllm/*.patch
```

### 2.1 最省事：不用 clone，镜像里就有 base commit

官方镜像的 `vllm-ascend` 检出**包含基线 `46856f89e` 作为祖先**（实测
`git cat-file -t 46856f89e` → `commit`、`merge-base --is-ancestor 46856f89e HEAD` → yes），
所以可以直接在容器里开一个 worktree 打补丁，无需从 GitHub 拉仓库：

```bash
docker exec -it <容器> bash -lc '
  cd /vllm-workspace/vllm-ascend
  git worktree add --detach /tmp/v41-opt 46856f89e79c3011401e33663c60da37cd486d53
  cd /tmp/v41-opt
  git am --keep-cr /opt/dsv41/patches/vllm-ascend/*.patch
'
```

> 已在 A3 镜像 `quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3` 内实测：
> **8/8 全部干净应用**，提交作者统一，结果与包内整文件形态逐字节一致。

> 注意：`git am` 打完后**不会自动生效**——运行时装的是镜像里已烘焙的文件
> （`scripts/build_image.sh` 从 `patches/files/` 覆盖）。`git am` 这条路径适合
> **审计 / 二次开发 / 向上游提 PR**；要在运行时生效，改完后需重建镜像或按
> `PATCH_MODE=mount` 挂载。

只想打**单个**优化（排查/AB 用）：

```bash
git apply --check /opt/dsv41/patches/vllm-ascend/0005-perf-attention-2D-wo_a-matmul-and-dummy-shape-guard.patch
git apply        /opt/dsv41/patches/vllm-ascend/0005-perf-attention-2D-wo_a-matmul-and-dummy-shape-guard.patch
```

**回滚**：`git am --abort` / `git checkout -- .`（整文件形态则 `cp xxx.a2orig xxx`）。

## 3. 每个补丁的门控 env 与实测收益

**全部默认关闭**（`=0`），只有 `serve_*.sh` 会把它们显式打开；这样任何一项出问题都能单独关掉做 AB。

| # | 补丁 | 主门控 env | 实测收益 | 精度 |
|---|---|---|---|---|
| 0001 | MoE dispatch/combine over AllGather | `V41_MOE_COMM_ALLGATHER=1` | 128K **−4.25 ms**、32K −1.35、8K −1.23；KV 3.39M→**4.16M** | 输出逐字节一致 |
| 0002 | expert mask 范围比较 | `V41_MOE_MASK_RANGE=1` | **−0.51 ms** | GSM8K 100/100、Vision 23/23 |
| 0003 | rope cos/sin 取表融合 | `V41_ROPE_IDXSEL=1` | **−0.45~0.62 ms/pass** | 逐算子等价 |
| 0004 | QLI 无候选快速路径 | `V41_QLI_NO_CANDIDATE=1` | 99.3→50.3 µs ⇒ **−0.49 ms** | 等价 |
| 0005 | `wo_a` 2D matmul（F3） | `V41_O_PROJ_2D=1` | **−0.31~0.76 ms** | 等价 |
| 0006 | engram gate 分块 | `V41_ENGRAM_GATE_CHUNK=<int>` | 8K **−1.56 ms**（线上用 0=stock，见下） | 等价 |
| 0007 | Engram host 常驻 + local-owner | `V41_ENGRAM_HOST_RESIDENT=1`<br>`V41_ENGRAM_LOCAL_OWNER=fast` | route 少一次 metadata all_gather + ids all_to_all | 数值等价 |
| 0008 | hash/plan numba JIT | `V41_ENGRAM_JIT=1` | hash 0.427→**0.076 ms**；plan 0.261→**0.068 ms** | 等价 |

### 3.1 三个必须知道的坑

1. **`V41_ENGRAM_GATE_CHUNK` 线上跑的是 `0`（=stock gate）**。分块路径的 −1.56 ms 是 8K 下测的，
   长上下文要自己复测；`serve_*.sh` 默认按线上口径传 `0`。
2. **`V41_ENGRAM_JIT=1` 需要 `NUMBA_CACHE_DIR` 可写**，否则每次起服都会重新编译（首次几十秒）。
   `serve_*.sh` 会挂 `./cache/numba`。
3. **`V41_MOE_ZERO_INVALID` / `draft/*` 是实验项**，本系列**不含**它们（`patches/files/` 里有单独文件，
   默认不挂）。`DRAFT_GRAPH=1` 在 A2 首轮实测会因缺 `DSPARK_GRAPH_CAPTURE_METADATA=1` 而静默失效
   （A 恒 1.0、但 ms/step 看着正常），要用请先补那一行。

## 4. 依赖关系（不要打乱）

```
0007 (engram_hbm + model + engram_plan_kernel)  ← 0008 依赖它（plan kernel 调用点在这）
0008 (engram_hash + engram_jit_kernel)          ← 需要 NUMBA_CACHE_DIR
0001..0006 互相独立，可单独摘出来做 AB
```

## 5. 复现"系列 = 整文件"的验证

```bash
cd /vllm-workspace/vllm-ascend
git worktree add --detach /tmp/vfy e43cf1e9f5d9bead076853aa6bcacb671465de94
cd /tmp/vfy && git am --keep-cr /opt/dsv41/patches/vllm-ascend/*.patch
md5sum -c /opt/dsv41/patches/vllm-ascend/MD5SUMS        # 应与 patches/files 一致
```
