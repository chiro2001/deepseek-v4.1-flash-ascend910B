# A2 DRAM KV 卸载 —— 可交付补丁集

> 2026-09-22 02:1x 整理。**这些补丁已在 A3 的 8 卡真权重上验证通过**
> （`logs/016` / `logs/022`：16 请求、`CPU_to_GPU=12.44 GB`、
> `external_prefix_cache_hits=507,904`、**replay 253.4 vs fill 4429.4 ms = 17.5×**）。
>
> **状态：A2 上尚未验证**（等 `a2_pinned_probe.sh` 的探测结果与一次起服）。

---

## 0. 这三个补丁解决什么

| # | 补丁 | 解决什么 | 依据 |
|---|---|---|---|
| **0001** | `scheduler.py`（**per-group bpc 版，含 D2 的参与位修复**） | ① **`state` 组一票否决**（它在取回路径上参与判定、却永远存不出 chunk ⇒ 整轮判死）；② **SWA 池条目太粗**（每 1024 token 存 8 个 block，而窗口只有 128 ⇒ 存了永不取回） | `logs/009` / `logs/021` |
| **0002** | `cpu_npu.py`（池子改走 `aclrtHostRegister`） | 绕开 `aclrtMallocHost` 的 `207001`（真实 8 卡上单次要 8 GiB 就失败） | `logs/014` / `logs/016` |

---

## 1. 文件与 md5

| 文件 | md5 | 说明 |
|---|---|---|
| `0001-offload-scheduler.patch.py` | `15d5548e29af88da71570d5b48abddef` | **`scheduler.py` 的替换版**。★ **它是 D2 版的超集**（`grep -c offload_participat` = **15**），所以**只需挂这一份**，不要再叠加旧版 |
| `0001b-offload-per-group-bpc-manager.patch.py` | `3b64eb4977f3302ed71e6c759c48740f` | `PerGroupBPCManager`（池的格子 = 1 个 GPU block） |
| `0001c-offload-per-group-bpc-hooks.patch.py` | `af2fefb8337fdf9fe1c5e55518f665b8` | 配置解析钩子（`blocks_per_chunk` 支持 `{"default":8,"swa":1}` 的字典形式） |
| `0002-offload-cpu-pool-host-registered.patch.py` | `2c161a791fe99f17cce2e1139ffbdc3c` | `cpu_npu.py` 的替换版（`NPU_OFFLOAD_HOST_MEM=registered` 走 `aclrtHostRegister`；**注册失败自动回落 `pinned`**） |

---

## 2. 怎么用

### 2.1 挂载（与 `serve_a2.sh` 的 `PATCH_MODE=mount` 同构）

```bash
# 0001 系列：把三份文件放进 shadow-pkg 的补丁目录
PKG=<shadow-pkg 路径>
cp 0001-offload-scheduler.patch.py              $PKG/patches/files/offload_dsv41/scheduler.py
cp 0001b-offload-per-group-bpc-manager.patch.py $PKG/patches/files/offload_dsv41/pgp_manager.py
cp 0001c-offload-per-group-bpc-hooks.patch.py   $PKG/patches/files/offload_dsv41/pgp_hooks.py
cp 0002-offload-cpu-pool-host-registered.patch.py $PKG/patches/files/offload_dsv41/cpu_npu.py
```

然后按 `docs/A2-GO-LIVE.md` §2.2 的开关起服。

### 2.2 ★ 起服后**必须先做补丁生效自检**（否则白跑 25 min）

```bash
grep -c "P1_pinned.*ret=0"           serve.log   # 期望 8（8 个 worker）
grep -c "D2_offload"                 serve.log   # 期望 > 0
grep -c "alignment_chunk_count.*8"   serve.log   # 期望 > 0（per-group bpc 生效）
```
**任一为 0 ⇒ 立刻停**（`logs/016` 的 dram32 臂没挂上补丁，白跑了 25 min）。

---

## 3. 参数定值

| 场景 | `OFFLOAD_GB` | 宿主实占（×6.945） | 依据 |
|---|---:|---:|---|
| **32K × 16 并发**（per-group bpc） | **16** | **≈111 GiB** | `logs/021` §6.1 重算 |
| **128K × 16 并发**（per-group bpc） | **48** | **≈333 GiB** | 同上（A2 余量 442 GiB） |
| 32K，**不**用 per-group bpc | 48 | 333 GiB | `logs/016` 实测 |
| 128K，**不**用 per-group bpc | — | **1,333 GiB** | ⛔ **A2 不可能** |

**其它必须的参数**（`docs/A2-GO-LIVE.md` §2.2）：`PREFIX_MATCH_UNIT=32`、`blocks_per_chunk` 用 per-group 字典、
`--enable-prefix-caching`、`--kv-cache-memory-bytes ≥ max_model_len × kv_per_token`。

---

## 4. 已知边界

| # | 事项 | 状态 |
|---|---|---|
| 1 | **A2 上的验证** | ⏳ **未做**（等 `a2_pinned_probe.sh`） |
| 2 | `aclrtHostRegister` 在 A2 上是否可用 | ⏳ **未测**（A2 `host_mem_pool=0`，Engram 曾注册失败） |
| 3 | `×6.945` 宿主乘数 | 【实测】A3 8 卡 / 单卡 tiny 两处一致；**A2 要重测** |
| 4 | per-group bpc 的 8 卡验证 | ⏳ **进行中**（`logs/027`） |
| 5 | `state` 组跳过的**数值正确性** | 只做了语义推断（与 GPU 前的缓存路径一致），**没做精度对比** |
