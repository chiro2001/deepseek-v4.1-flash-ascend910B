# int8 档 C 的 decode 回退修复（v2：chunkview + fuse + L1 + L2）

> 2026-09-23。**A3 8 卡实测通过**（行为健康闸 + 题库 + 取回路径 + op 级计数）。
> 本目录的 `dsa_v41.py` 现在是**修复版**；下面 4 个文件构成完整的一集。

---

## 1. 为什么必须有这一集（A2 上的量级）

| 事实 | 值 |
|---|---|
| A2 生产池 | **85 GiB**（`OFFLOAD_GB=85`）⇒ 约 **28,577 页** |
| A3 验证池 | 4 GiB ⇒ 7,938 页 |
| 未修版在 7,938 页下的代价 | **989 µs/层 × 40 = 39.56 ms/step** |
| 同一律外推到 A2（28,577 页） | **≈125 ms/step**（成本**正比于池总大小**） |
| 修后（连续 chunk 视图） | **对池大小变平**（320→13,366 页之间只动 ±3%） |

**根因**：生产 SWA int8 面是混合槽页上的 `as_strided` 视图（`core/deepseek_v41.py:374-381`），
而 `aclnnIndexSelect` 会**无条件 `l0op::Contiguous(self)`**（`ops-nn/index/gather_v2/op_api/aclnn_index_select.cpp:186-187`）
⇒ 每次调用都**物化整个平面**。

---

## 2. 四个文件与各自的 md5（**契约**，不是提示）

| 文件 | md5 | 作用 |
|---|---|---|
| `a2/patches/kv8-graphsafe/dsa_v41.py` | **`7867da2a345d7135ddbc6919eec144f9`** | 主件：graphsafe + **chunkview** + fuse 接线 + **L2** |
| `a2/patches/kv8-int8-pkg/vllm_ascend/attention/kv8_fuse_triton.py` | **`8057b3eb36622ee419d9a8c81d3ff73b`** | 融合 kernel（**与 dsa 必须同挂载集，否则 ImportError**） |
| `patches/files/token_dispatcher_moemask.py` | **`a91fbc48350530d987ef3bb1ff15f1cb`** | L1：MoE 掩码改在 **int32 域**比较（去掉 2× `Cast INT32→INT64`） |
| `a2/scripts/make_shadow_pkg.sh` | **`0622ec8b83d21bb302da405601372203`** | ★ **挂载表加第 8 件**：`attention/kv8_fuse_triton.py` |

★ **第 4 个是必须的**：原来的挂载表只有 6+1=**7 件**，**不含 `kv8_fuse_triton.py`**
⇒ 起档 C 时 `dsa_v41.py` 顶部的 `from vllm_ascend.attention import kv8_fuse_triton` 会
**ImportError**（不是静默降级）。本补丁把它变成第 8 件。

---

## 3. A3 实测（证据）

**修复量级**：`75.911 → 27.632` ms/step（8K）｜`80.168 → 31.449`（128K）。

**op 级判据（decode 稳态窗，99 步）**：

| 对象 | 修前 | 修后 |
|---|---:|---:|
| `Cast(6,6) INT32→INT64` | **80/步** | **0.00/步** |
| `Less/GreaterEqual` 输入 dtype | `INT64` | **`INT32;INT32`**（79.98/步） |
| `ViewCopy(16384)` | 23.75/步 | **1.98/步** |
| `SelectV2(6;6;)` | 24.74/步 | **2.97/步** |
| `_kv8_swa_table_kernel` | 13.17 µs/次（单卡） | —— |

**精度/行为（三闸，必须先过再看性能）**：
* 题库 **10/10**
* 取回路径 `prefix-pair` **三发逐字相同**
* **行为健康闸**：聚合 A **1.242**（基线 1.200）

---

## 4. 怎么用（A2，`git pull` 之后**不需要**额外安装脚本）

```bash
cd <你的发布仓 clone> && git pull --ff-only
PKG=$(pwd) DST=$HOME/shadow-pkg bash a2/scripts/make_shadow_pkg.sh

# ★ 干跑：只看 MOUNTS，不碰生产容器
ENGRAM=1 ENGRAM_DEVICE_INDEX=0 KV8_SWA=1 KV8_RING_FP16=1 DRY=1 \
  bash a2/scripts/serve_a2_offload.sh
#   ⇒ 必须有：[serve_a2] [A2-INT8] 已挂 **8** 个整文件件
#   ⇒ 若还是 "7 个"，说明 make_shadow_pkg.sh 没更新 ⇒ **停手**
```

**起服后的四条自检**（缺一即停）：
```bash
TAG=<容器名>
docker exec "$TAG" md5sum /vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py \
                           /vllm-workspace/vllm-ascend/vllm_ascend/attention/kv8_fuse_triton.py \
                           /vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py \
                           /vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/... # 见下
# 期望：7867da2a… / 8057b3eb… / a91fbc48…
grep -c EE1016 <serve.log>          # 期望 0
grep -m1 "GPU KV cache size" <serve.log>   # 档 C 期望 427,643
python3 arm_health_gate.py <serve.log>     # 期望 rc=0（★ 先过闸再看性能）
```

---

## 5. 已知边界（**不要当成全绿**）

1. **只在 A3（8 卡、4 GiB 池、`MAX_LEN=133120`）验证过**；A2 的 85 GiB 池只是**按同一律外推**（修复的特征是"对池大小平"，但**未在 A2 上实测**）。
2. **128K 仍比"PGO 开"的历史配置慢 4.0%**（31.449 vs 30.230）；比同类同开关的 `faA` 快 0.56%。
3. 本集**不含 G3-β**（fp16 scale 融合）：它在单卡定价 0.068–0.19 ms/step，**未与本集同臂叠过**。
4. **回退**：把 `git checkout <本提交之前>` 或恢复 4 个文件的旧 md5（`94aeebb7` / 无 kernel / `a695735a` / 旧 `make_shadow_pkg.sh`）。
