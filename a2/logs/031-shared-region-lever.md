# ★★★ 第二条大杠杆：**8 份副本可以合成 1 份**（upstream 已有机制，只是被一段**错误的注释**挡住）

> 2026-09-22 02:2x，主代理前台**只读代码**发现（不占卡）。
> 起因：用户问"DRAM 能放的 KV 太少，有没有方案能增大"。
> `logs/029` 给了 **L1（16× 张量维度闲置）**；本文是 **L6（8× 副本冗余）**。

---

## 一、upstream 自己的注释说"做不到"——**而那个理由已被我们证伪**

`vllm_ascend/.../kv_offload/native/npu.py::NPUOffloadingSpec.create_worker`（原始注释）：

> *"Keep the single-tier path on PyTorch's pinned allocator. **Unlike CUDA, Ascend has no public
> cudaHostRegister-equivalent for an arbitrary mmap buffer**, and the pinned tensor path has the
> best proven H2D/D2H performance. Consequently **`replicated_layout` remains safely disabled**
> for this spec by the upstream `_uses_shared_region()` gate."*

**★ 而 `logs/014`（P1_pinned）已经实测证明这段话的理由不成立**：

| 上游的假设 | 我们的实测 |
|---|---|
| Ascend 没有 `cudaHostRegister` 等价物 | ❌ **有**：`aclrtHostRegister(ptr, size, 0)` 对**任意 mmap buffer** 成立（ret=0） |
| 只能走 pinned，且 pinned 性能最好 | ⚠️ 性能**同级**：注册后的 H2D **58 GB/s / D2H 42.7 GB/s**，与 pinned 无实质差别 |
| 所以 `replicated_layout` 必须关 | ⇒ **这个推论的前提塌了** |

---

## 二、★ 而 upstream **已经写好了**单副本机制（在另一条 spec 上）

同一文件的 `NPUTieringOffloadingSpec`：

```python
class NPUTieringOffloadingSpec(_NPUWorkerMixin, _TieringOffloadingSpec):
    def _uses_shared_region(self) -> bool:
        # Unlike the CPU-only NPU spec, tiering always connects every worker
        # and the scheduler-side primary tier through SharedOffloadRegion.
        # Advertising that fact lets upstream safely enable its **single-copy layout**
        # for configurations it has certified as **byte-replicated**
        # (currently **pure MLA** under the supported TP topology).
        return True
    ...
    if self.replicated_layout:
        rank = 0                        # ← ★★ 所有 worker 映射同一份 region
    else:
        rank = int(torch.npu.current_device()) % world_size
    worker_mmap = SharedOffloadRegion(
        engine_id=self._engine_id, rank=rank,
        cpu_page_size=self.cpu_page_size_per_worker, **region_kwargs,
    )
```

### 2.1 两件事同时成立

1. **`SharedOffloadRegion` 在镜像的 vLLM 里就有**
   （【实测】`/vllm-workspace/vllm/vllm/v1/kv_offload/cpu/shared_offload_region.py:28`）；
2. **`replicated_layout=True` ⇒ `rank = 0`** ⇒ **8 个 worker 映射同一份物理内存**
   ⇒ **副本从 8 份变成 1 份 ⇒ 8× 的宿主内存节省**。

### 2.2 ★ 而 V4.1 **正是**"pure MLA"

上游那句限定是 *"currently pure MLA under the supported TP topology"*。
V4.1 的 4 个共享 long-KV 平面 + 40 个 SWA 资源**都是 MLA 系**
（`AscendMLAAttentionSpec` / `AscendSlidingWindowMLASpec`），见 `logs/001` §3 的 13 组清单。

⇒ **结构条件对得上**（但"字节是否逐位复制"仍需实测确认，见 §四）。

### 2.3 镜像是缺了一块的

【实测】镜像里 **`NPUTieringOffloadingSpec` 不存在**（`grep` 零命中）——
它只在 **`origin/main`** 上有。`logs/045`（M_offload）也记过：
*"路径 B（Tiering）：本镜像**没有** ascend 版 `NPUTieringOffloadingSpec`"*。

⇒ **要拿到这个 8×，得把 main 的那段 backport 到单 tier 路径上**（或把 spec 换掉）。

---

## 三、★★ 两条大杠杆的合并账（**都在同一份 RAM 上）**

| 杠杆 | 机制 | 倍数 | 状态 |
|---|---|---:|---|
| **L5** | per-group `blocks_per_chunk`（SWA 条目粒度） | **4.89×** | ✅ **已完成**（`logs/021`） |
| **L1** | 16 个池张量各分 `num_blocks` 个 slot，但一个 slot 只用一份 | **~16×** | ⏳ `P2_poolsizing` 在验（`logs/029`） |
| **L6（本文）** | **8 份副本 → 1 份**（`replicated_layout`） | **8×** | ⏳ **待派** |

**⇒ 三者是乘法关系**（`logs/029` §2.2 已经用 16×6.944 解释了那个 150× 的差；
而 6.944 里面就**含**了 8 份副本这个因子）。

> ⚠️ **口径提醒**：`logs/019` §10 的 A2 容量表（16×128K = 264 GiB）
> **已经包含**了当前的 8 份副本。若 L6 成立，该表要**除以 8**（≈33 GiB）。
> 但 L1 与 L6 的收益**不是简单相乘**（都作用在"每个张量分多少 slot × 几份"上），
> **要按新结构重算**，不能拿 16×8=128 去乘 —— 这是**推算，不是结论**。

---

## 四、要验证的三件事（**都不占卡**）

| # | 问题 | 为什么关键 |
|---|---|---|
| **V1** | V4.1 在 TP8 下，8 个 rank 的 KV **是不是逐字节复制**的？ | 这是 `replicated_layout` 的**语义前提**。上游说"certified byte-replicated, currently pure MLA"，但**没有 V4.1 的认证**（V4.1 还有 SWA / compressor state / Engram） |
| **V2** | 单 tier 路径能不能用 `SharedOffloadRegion`（不换 spec、不引入 SSD 层）？ | 若必须走 tiering spec，就还要那层的依赖 |
| **V3** | 写路径：8 个 worker 都往**同一份** region 写，会不会**互相踩**？ | 上游的写法是"certified byte-replicated" ⇒ 写同样的字节**幂等**。但**要确认我们的 per-group bpc 补丁没破坏这个前提** |

---

## 五、对上游的额外价值（**一条独立的贡献**）

那段注释里的事实陈述是**错的**（"Ascend has no public cudaHostRegister-equivalent
for an arbitrary mmap buffer"），而它**直接导致了一个把内存放大 8 倍的默认值**。

⇒ **这是一条可以给 upstream 的、成本很低的贡献**：
附上 `logs/014` 的实测（`aclrtHostRegister` ret=0 + 往返逐字节一致 + 58 GB/s），
建议他们**重新评估 `_uses_shared_region()` 的默认值**。

---

## 六、诚实边界

| # | 事项 |
|---|---|
| 1 | 8× **是上限**（若 8 个 rank 真的各存一份且逐字节相同）；**没有实测过** |
| 2 | `replicated_layout` 在**单 tier 路径**上启用后，**写路径的幂等性**没有验证 |
| 3 | 镜像里**没有** `NPUTieringOffloadingSpec`，backport 有工作量（且要处理我们已加的 per-group bpc） |
| 4 | 与 L1（16×）的**交互**没算清（可能部分重叠） |
| 5 | 若 V4.1 的 KV **不是**逐字节复制（例如 SWA 的 slot_mapping 各 rank 不同），这条路**直接否决** |
