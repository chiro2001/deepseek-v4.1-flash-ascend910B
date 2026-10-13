# DCP 输出合并拓扑（TP8 + DCP8 复用同一组 rank）

**日期**：2026-09-29　**定位**：只回答「每个 rank 只算自己 KV 分片的 partial attention，怎么合并成全局输出」这一件事。
**只读声明**：未起服务、未占卡、未改 `~/cedpd-repo/`、`~/dcpw/`，未 commit。

**引用树**（行号只对这几份快照有效）

| 代号 | 路径 | 说明 |
|---|---|---|
| 【镜像】 | `/home/chiro/tmp/v41img2/vllm_ascend/` | 要改的树。`attention/context_parallel/sfa_cp.py` 1315 行，**没有** `ops/triton/` 目录 |
| 【参考】 | `/home/chiro/tmp/va_full2/vllm_ascend/` | 其 `context_parallel/sfa_cp.py`、`sfa_v1.py` 与【镜像】**逐字节相同**（`diff -q` 通过）；pack/merge kernel 在 `ops/triton/sfa_cp.py`（425 行） |
| 【新版】 | `/home/chiro/tmp/va-latest/vllm_ascend/` | vllm-ascend main `b64b4d7`，`context_parallel/sfa_cp.py` 1699 行；kernel 移到 `ops/triton/dcp/dcp_a2a.py`（622 行） |
| 【V41】 | 【镜像】`attention/dsa_v41.py`（1092 行）＋`models/deepseek_v41/` | 要接入的目标 |

**一句话数据流**：DCP 切 KV、TP 切 head，两者同一组 8 个 rank ⇒ 每 rank 先沿 head 维 all_gather 出全部 64 个 head 的 q（token 不动），用它对本 rank 的 KV 分片算 partial `(out, lse)` = `[T,64,D] / [T,64,1]`，再做一次按 head 切的 all_to_all 把 head 块 d 送到 rank d，在目的 rank 上按 LSE 加权合并 ⇒ `[T,8,D]`，正好是本 rank 的 TP head 分片对全 KV 的结果。

## 1. 数据流表

| # | 阶段 | 张量形状（括号内为 V4.1 数值） | 通信 | 位置 |
|---|---|---|---|---|
| 1 | 模型侧 q 投影（TP 已切 head） | `q [T,8,512]` bf16 | — | 【V41】`attention/dsa_v41.py:286`、`:372` |
| 2 | 拆 nope/pe 并 fuse | `ql_nope [T,8,448]`、`q_pe [T,8,64]` → `fused [T,8,512]` | — | 【镜像】`sfa_cp.py:1106` |
| 3 | **head 维 all_gather** | `[T,8,512] → [T,64,512]`（token 不变，每 rank 一份全量拷贝） | all_gather over dcp_group，dim=1 | 【镜像】发起于 q 投影后（`sfa_v1.py:1614-1620` 调 `_record_query_gather_context`）；实现 `sfa_cp.py:1088-1111`；dim 来自 `sfa_v1.py:1302-1304`（`return 1`）；异步 `:983-997`、还原 `:973-982` |
| 4 | top-k 索引 remap 到本地 KV 坐标 | `[T,topk]` 全局 token 坐标 → 本地槽位（不属于本 rank 置 -1 并压到尾部） | — | 【镜像】`sfa_cp.py:1231`、`:998-1045` |
| 5 | 本地 SFA（64 head × 本 rank KV 分片） | `out [T,64,512]` bf16；LSE 组装后 `[T,64,1]` fp32 | — | 【镜像】`sfa_cp.py:1233-1249`、`:1250-1251` |
| 6 | pack（out 与 LSE 进同一个 payload） | `send (8,8,T,512+1 或 512+4)` | — | 【参考】`ops/triton/sfa_cp.py:265-300`（形状 276-280） |
| 7 | **按 head 的 all_to_all** | `send → recv`（同形状） | `dist.all_to_all_single` over dcp_group | 【参考】`ops/triton/sfa_cp.py:368-369` |
| 8 | fused LSE combine | `recv (8,8,T,516)` → `out [T,8,512]` bf16 | — | 【参考】`ops/triton/sfa_cp.py:303-351`（形状 327-334） |
| 9 | 返回模型层，本地 o_proj + 常规 TP all-reduce | `[T,8,512] → [T,8,·]` | TP row-parallel 既有通信 | 【镜像】`sfa_cp.py:1258` |

例外：**prefill / mixed 批不走这条路**——那时改成 all_gather KV 全量（【镜像】`sfa_cp.py:918-957`、`1177-1216`），每 rank 用本地 head 直接算，无 Q gather、无 LSE、无输出 a2a。

## 2. Q1：`_start_dcp_query_gather` gather 的是什么

**gather 的是 head（dim=1），不是 token。**

- 维度来源：native DCP 类的 `_parallel_query_gather_dim()` `return 1`（【镜像】`attention/sfa_v1.py:1302-1304`）；DSA-CP 类覆写成 `return 0`（`sfa_cp.py:314`，那是 token 切分，别混）。
- 形状前后：`ql_nope [T,8,d_nope]` 与 `q_pe [T,8,d_pe]` 先沿最后一维拼成 `fused [T,8,d_nope+d_pe]`（`sfa_cp.py:1106`），再沿 dim=1 all_gather（`:1107-1111`）⇒ `gathered [T,64,D]`，最后 `torch.split` 还原为 `[T,64,d_nope]`/`[T,64,d_pe]`（`:973-982`）。**token 维 T 不变；每 rank 拿到的是「全部 head 的 q」（8×8=64），不是「所有 token 的 q」。**
- 【新版】同语义，只是先转 head-major `[H,T,D]` 再 gather（`restore_perm=(1,0,2)`，`sfa_cp.py:1391-1405` + `ops/triton/query_gather_prep.py`，注释明写 "the buffer that all_gather_into_tensor needs for the native-DCP head"），目的是少一次拷贝。
- **为什么必须 gather**：a2a 按 head 切，`local_scatter_size = num_heads // dcp_size`。若 q 不 gather，本地只有 8 个 head ⇒ `local_scatter_size = 8//8 = 1`，pack 会把「1 个 head」往 8 个目的 rank 各摆一份，语义完全错。gather 后 `num_heads=64=dcp_size×8`、`local_scatter_size=8`＝TP head 分片大小，目的 rank d 收到的恰好是 head 块 d。
- 代价可接受的原因：DCP 切的是 KV 不是 head，每个 KV 元素只被它所在的那个 rank 算一次，全局算术量与非 DCP 相同；gather 只是把「按 KV 切」的 partial 重新按 head 归位。

## 3. Q2：`_merge_dcp_outputs` 的 a2a 语义

调用点：【镜像】`sfa_cp.py:1047-1086`（`scatter_dim = 1` 在 `:1053`，算子调用 `:1080-1086`）。

`send[dst][local_scatter][replicated][packed]` 各下标的物理含义（pack kernel 【参考】`ops/triton/sfa_cp.py:44-59`）：

| 下标 | 大小 | 物理含义 |
|---|---|---|
| `dst` | `dcp_size = 8` | 该 head 的**归属 rank**（= 该 head 所在 TP 分片的 rank）：`rank_idx = head_idx // local_scatter_size` |
| `local_scatter` | `H/dcp_size = 8` | head 在归属 rank 的 head 块内的偏移：`scatter_idx = head_idx % local_scatter_size` |
| `replicated` | `T` | token 号（`scatter_dim=1` 时 token 是复制维，每个目的 rank 都要全 T） |
| `packed` | `D + LSE_PACK_DIM` | `[out(D) 与 LSE 编码]`：输出 fp32 ⇒ LSE 占 1 个 fp32（`LSE_PACK_DIM=1`）；输出 bf16/fp16 ⇒ 占 4 个元素（符号指数码 + 3 个 base-256 尾数位，全部是 [-255,255] 整数，可被 fp16/bf16 精确表示）【参考】`:61-105`、`:213-218` |

`dist.all_to_all_single(recv, send)` 之后，`recv[src][local_scatter][replicated][packed]` = **rank src 算出的该 (head, token) 的 partial**；combine kernel 用同一套公式回读（下标定义 `:135-140`，地址 `:147-151`／`:174-179`）。

**合并后每个 rank 拿到的是：本 rank 的 TP head 块（8 个 head）× 全部 T 个 token × D，即 `[T,8,D]`，每个元素 = 8 个 KV 分片 partial 的 LSE 加权和 ⇒ 就是「自己 TP 的 head × 全 KV」的正确的全局输出**（该结论与我们离线复算的下标自检一致：a2a 后目的 rank d 的 `(scatter=j, token=t)` 槽位全部来自「各源 rank 对全局 head `8d+j`、token t」的 partial）。

## 4. Q3：`num_heads` / `num_tokens` 从哪来

`fused_sfa_dcp_lse_combine(recv, head_dim, scatter_dim)`（【参考】`ops/triton/sfa_cp.py:303-351`）全文要点：

```python
dcp_size, local_scatter_size, replicated_size, packed_dim = recv.shape     # :320
lse_pack_dim = _lse_pack_dim(recv.dtype)                                   # :321 (bf16/fp16->4, fp32->1)
num_tokens, num_heads = (
    (local_scatter_size, replicated_size) if scatter_dim == 0              # :327-329
    else (replicated_size, local_scatter_size)                             # scatter_dim=1 走这里
)
output = torch.empty(                                                      # :330-334
    (num_tokens, num_heads, head_dim),
    dtype=recv.dtype, device=recv.device,
)
total_rows = num_tokens * num_heads                                        # :335
grid_size = min(total_rows, get_vectorcore_num())                          # :337
_fused_sfa_dcp_lse_combine_kernel[(grid_size,)](                            # :338-350
    recv, output, *recv.stride(), *output.stride(), head_dim, num_heads,
    total_rows, DCP_SIZE=dcp_size, SCATTER_TOKENS=(scatter_dim == 0),
    LSE_PACK_DIM=lse_pack_dim, BLOCK_D=triton.next_power_of_2(head_dim))
```

- `scatter_dim=1` ⇒ **`num_heads = local_scatter_size`（8，本 rank 的 head 块）、`num_tokens = replicated_size`（T）**；kernel 的 `total_rows = T×8`。
- 这两个数**不是配置直接传进来的**，而是从 `recv` 形状反推；`recv` 形状又由 pack 侧的 `sfa_output.shape` 决定：`scatter_size = sfa_output.shape[scatter_dim]`（64）、`local_scatter_size = 64//8 = 8`、`replicated_size = num_tokens`（T）【参考】`:253-261`。所以「64 个 head」只存在于 gather 之后的 `sfa_output`，**输出侧只剩 8 个 head**。
- `recv` 由 `torch.empty_like(send)` 分配（`:368`），dtype 与 `sfa_output` 相同 ⇒ V4.1 bf16 走 `LSE_PACK_DIM=4`，payload = 516。

## 5. Q4：LSE 里 sink 的约定

**SFA 参考实现里 DCP 合并根本不涉及 sink**，所以它「不需要处理」，这是正确的：

- DCP 路径调的是 `DeviceOperator.execute_sparse_flash_attention_process`（【参考】`device/device_op.py:387-455`），它转发到 `npu_sparse_flash_attention`（非量化，调用在 `:430`）或 `npu_kv_quant_sparse_flash_attention`（量化，函数在 `:458`，调用在 `:472`）——**两个算子都没有 `sinks` 形参**；`rg sink` 在 `attention/sfa_v1.py`、`attention/context_parallel/sfa_cp.py`、`ops/triton/*.py` 里零命中。（【镜像】只解包了 `attention/`、`core/`、`models/deepseek_v41/`、`platform.py`、`utils.py`、`ascend_config.py`，`device/` 与 `ops/` 要看【参考】/【新版】。）
- 于是每 rank 的 LSE 只是 `log Σ exp(scale·q·k)`，合并式 `Σ w·o / Σ w` 的分子分母都不含 sink 项——等价于「sink 一个都不加」。SFA 的模型侧本来也没有 sink 参数，所以不存在少算/多算。
- **别抄错**：【新版】的 SMLA 分支会给算子传常量占位 sink（`SMLA_DEFAULT_SINK_VALUE = 1.0`，`attention/sfa_v1.py:76-80`、`sinks=metadata.smla_sinks` 在 `:216`、分配在 `:298-302`）。那条分支 `return_softmax_lse=False`、也不参与 DCP 合并（DCP 走 `device_op` 那条），所以没暴露；**但它说明「算子要求 sinks 必传」这件事是真实存在的**。

V4.1 与 SFA 的差别在于：V4.1 的算子**强制传** `sinks [N1] fp32`，且 LSE 的实测语义就是 `logsumexp(scale·q·k ∪ {sink})`（`/home/chiro/tmp/v41_op_probe/README.md`）；调用点见【V41】`attention/dsa_v41.py:544-567`（`sinks=attn.attn_sink` 在 `:555`），参数本体是逐 head 的真参数（`attn_sink`，按 `n_local_heads` 分配，见【参考】`models/deepseek_v4/model.py:499-500`；【镜像】没解包该文件）。所以**必须只计一次**，两个等价做法（都由离线复算验证到 1e-15 量级）：

| 方案 | 做法 | 代价 | 精度（本次复算） |
|---|---|---|---|
| A. 只在归属 rank 带 sink | 每个 rank 的算子调用传长度 64 的 sinks 向量：rank r 只在 `[8r, 8r+8)` 填真值（= 本地 `attn_sink`），其余填占位大负值。这样 head h 的 `exp(sink_h)` 只进入 rank `h//8` 的 LSE，而 `h//8` 正是 h 的合并目的地 ⇒ 不需要任何额外通信，**kernel 不用改** | 需要实机确认占位值（见下） | 9.5e-16 |
| B. 合并时加一次 | 每个 rank 的算子调用把 64 个 head 的 sink 全填占位（各 rank LSE 都不含 sink）；合并 kernel 里把 sink 当「value=0 的额外源」：`denominator = weight_sum + exp(sink_h - L)`，并把 `L` 取 `max(rank lse, sink_h)` 防溢出，分子不变 | 要改 kernel（加一个 sinks 指针 + 分母项） | 1.07e-15 |

（另一路子代理的实机/离线实验给的是同一结论：每 rank 都带 sink ⇒ 相对误差 4.9e-4 / 3.6e-3 / 0.63 @ sink = 0/2/10；只加一次 ⇒ 1.7e-15。本次复算在 sink 幅值 ±4 下得到「每 rank 都带」2.09/0.845 量级的同向错误。）

**不要照 DSA-CP 的做法**：`attention/context_parallel/dsa_cp.py:1468-1471` 要求「full-head `attn_sink` 加载到每个 TP rank」，`:1943` 把 `sinks=self.attn_sink` 整份传进算子。那在 DSA-CP 里是对的——DSA-CP 切 token，每个 `(token, head)` 只在一个 rank 上被算一次，合并即「单源」；搬到 DCP（切 KV，同一个 `(token, head)` 的 partial 同时存在于 8 个 rank）就会把 `exp(sink_h)` 计 8 次。

**V4.1 照哪个做**：推荐 **A**（kernel 零改动、sink 始终留在算子自己的 softmax 里，数值最稳），把 **B** 作为等价备选（【新版】的 combine kernel 已经有一个「额外算一次」的钩子 `HAS_LOCAL`，`ops/triton/dcp/dcp_a2a.py:227-234`，B 可以照着它写）。两条都要求「head 块序号 == rank 序号」，见 §7 备注。

## 6. Q5：某个 rank 本地一条 top-k 都没命中（LSE = -inf）时怎么办

kernel 的处理（【参考】`ops/triton/sfa_cp.py:145-208`）：

1. `valid_lse = (lse == lse) & (lse != inf) & (lse != -inf)`（`:154`、`:166`、`:182`、`:194`）——NaN 与 ±inf 一律判无效。
2. 第一遍求最大值时无效 rank 用 `-inf` 参与（`:167`）；若**全部**无效 ⇒ `any_valid_lse=False` ⇒ `safe_lse_max = 0.0`（`:169-170`），避免 `-inf - (-inf) = NaN`。
3. 第二遍算权重 `weight = exp(lse - safe_lse_max)`，无效 rank 权重 0（`:195`）；partial output **先按 `valid_lse` 置 0 再乘权重**（`:203`，注释明写防止「零权重 × NaN」污染），再累加分子与 `weight_sum`。
4. `denominator = weight_sum if weight_sum > 0 else 1.0`（`:207`），最后 `merged /= denominator`。

⇒ 一条都没命中的 rank 贡献 0 权重；**若 8 个 rank 全空，输出是全 0（不是 NaN）**；只要有一个 rank 有效，结果就是标准的全局 LSE 重加权，精确。

本次复算：单 rank 非空的 token（其余 7 个 rank `lse=-inf`）合并结果与全局参考一致（4.4e-16）；全空 token 输出 0，参考也是 0。

V4.1 需要额外注意的点：

- 若没有任何 sink（或采用方案 B 的占位），kernel 逻辑已覆盖，无需额外代码。
- 采用**方案 A** 时，占位值不能让 LSE 变成 NaN；而且「空 partial 的归属 rank」在算子里的 LSE 应当等于 `sink_h`（而不是 `-inf`）——因为 `log(0 + exp(sink_h)) = sink_h`。此时输出仍是 0（分子为 0），但分母带着 `exp(sink_h - L)`，这正是 sink 语义。本次复算：全空 token 的归属 rank `lse = 1.97013 = sink`，合并输出 0，与参考一致。
- 若实机对「全 -1 的 topk（无命中）」返回 `-inf` 而不是 sink logit，则方案 A 在这条路径上会丢掉 sink 项，需回退到方案 B；这一点**需要一次单卡 probe 才能确认**（§9）。
- 占位值本身也**未确认**：`-inf` 可能在算子内部产生 `inf - inf`；有限大负值（如 `-1e30`）更稳，但要确认在 fp32 `exp` 下必然下溢到 0、且不污染真正的最大值比较。

## 7. V4.1 可直接照抄的伪代码

```python
# rank = tp_rank = dcp_rank；TP8 == DCP8，H=64，H_local=8，head_dim=512
def v41_dcp_attention(attn, q, kv_cache, topk_indices, metadata, actual_seq_lengths_query):
    T = q.shape[0]                                   # q: [T, 8, 512] bf16（wq_b 的 TP 分片）

    # 1) head 维 all_gather（token 不动）。异步发、等到算子前再 wait，盖住 indexer 选取的延迟。
    fused = q.contiguous()                           # [T, 8, 512]
    gathered, handle = all_gather_async(fused.transpose(0, 1).contiguous(), dcp_group)  # [8,T,512] -> [64,T,512]
    q_all = gathered.permute(1, 0, 2).contiguous()   # [T, 64, 512]（= 全部 head，每 rank 一份）
    #   等价写法：直接用【镜像】的 _start_dcp_query_gather/_finish_dcp_gather，
    #   它们内部就是 perm=(1,0,2) 转 head-major、gather、再 permute 还原（sfa_cp.py:973-1111）。

    # 2) top-k 索引：全局 token 坐标 → 本 rank KV 分片坐标（非本 rank 置 -1 并压到行尾）
    local_indices = remap_sparse_indices(topk_indices, dcp_size, dcp_rank, interleave)  # [T, topk]

    # 3) sinks：方案 A —— rank r 只在 [8r, 8r+8) 填真值，其余填占位（大负值）
    #    注意 DCP 下 attn_sink 仍按 n_local_heads(=8) 分配：不要照 DSA-CP 改成 full-head
    #    （models/deepseek_v4/model.py:499-500 里 enable_dsa_cp 才取 n_heads）。
    sinks64 = torch.full((64,), PLACEHOLDER_NEG, dtype=torch.float32, device=q.device)
    sinks64[dcp_rank * 8:(dcp_rank + 1) * 8] = attn.attn_sink   # 本地 8 个 head 的真 sink

    # 4) 算子：用全部 64 个 head 对本 rank 的 KV 分片做 sparse attention
    out, lse = torch.ops._C_ascend.npu_sparse_flash_mla(
        q_all, ori_kv=kv_cache[0], cmp_kv=..., sinks=sinks64, metadata=op_metadata,
        softmax_scale=attn.softmax_scale, layout_q="TND", layout_kv="PA_BBND",
        ori_mask_mode=..., cmp_mask_mode=..., ori_win_left=attn.window_size - 1,
        topk_value_mode=1, return_softmax_lse=True,
    )
    # out: [T, 64, 512] bf16；lse: [1, T, 64] fp32（MLA 的 N2==1）
    lse = lse.reshape(T, 64, 1).contiguous()         # -> [T, 64, 1] fp32，pack 需要这个布局

    # 5) pack + 一次 all_to_all + 按 LSE 合并（照抄 SFA 的算子）
    partial = torch.ops.vllm.dcp_a2a_fused(          # 入库时按本树改名/重新注册
        out, lse, 8, 1, dcp_group.unique_name,       # scatter_dim=1：按 head 切
    )                                                # -> [T, 8, 512] bf16
    return partial                                   # 本地 TP head 分片 × 全 KV，可直接喂 o_proj
```

备注与坑：

- `lse` 必须是 `[T, 64, 1]` **float32 连续**（pack kernel 直接吃 `stride(0)/stride(1)`，见 `:288-289`）。V4.1 的算子返回 `[1,T,64]`，MLA 下 `N2==1`，**直接 `reshape(T,64,1)` 即可**；不要照抄 SFA 那两行 `permute(1,0,2).reshape(shape[1],-1,1)`（【镜像】`sfa_cp.py:1250-1251`）——那是给「max/sum 是 `[N,T,1]`」的 `npu_sparse_flash_attention` 写的，套到 `[1,T,64]` 上会把形状算错。
- pack 的 `head_dim` 用 attention 输出维（V4.1 = 512），LSE 载荷按输出 dtype 选（bf16 ⇒ 4 个元素，payload 516）。
- `scatter_dim=1`，`group` 用 TP/DCP 那一组（本配置里就是同一个 8 rank 组）。目的 rank 的 head 块 == 本 rank 的 TP head 分片，这一条依赖「wq_b/o_proj 按 rank 连续切 head」；vLLM 的列并行默认就是连续切，**落地前建议用一次小张量核对**（把 head 0 的权重只放 rank 0）。
- 【镜像】没有 `ops/triton/` 目录：要么把【参考】的 `ops/triton/sfa_cp.py`（425 行）整体搬过去并注册 `torch.ops.vllm.sfa_dcp_a2a_fused`（`:419-425`），要么用【新版】的 `ops/triton/dcp/dcp_a2a.py`（622 行，多「本地贡献」与 `RETURN_LSE`）。
- prefill/mixed 批不要走这段：按 SFA 的做法改成 gather KV 全量（【镜像】`sfa_cp.py:918-957`、`1177-1216`），否则会既贵又错。

## 8. V4.1 与 SFA 的差异清单（会影响合并拓扑的部分）

| # | 差异 | 对合并拓扑的影响 |
|---|---|---|
| 1 | **KV 平面数量**：SFA 只有一份 MLA KV（k_nope/k_pe）；V4.1 有 long_kv（4 层，`compress_ratio` 0/1/2）+ SWA（10 层，窗口 128）+ indexer k_cache + compressor state ring | 归属公式（`owner(pos)=(pos//I)%dcp`）只对「按 token 位置铺开的平面」成立；SWA 窗口跨 rank 的可见性与 ratio=2 压缩平面的坐标都要单独确认【未确认】 |
| 2 | **attn_sink**：SFA 没有；V4.1 有逐 head 真参数 | 见 §5，必须「只计一次」 |
| 3 | **算子与 LSE 形态**：SFA 用 `npu_sparse_flash_attention`（返回 max+sum 两个张量，需 `max + log(sum)`，【镜像】`sfa_cp.py:1250`）；V4.1 用 `npu_sparse_flash_mla`（直接返回 LSE，且 `sinks` 必传） | LSE 组装/重排两行不能照抄，见 §7 |
| 4 | **indexer 两段筛选**：V4.1 是 candidate 粗筛 + 细筛（【V41】`dsa_v41.py:451-482`），top-k 是全局 token 坐标；SFA 的 DCP indexer 直接产「复制视图坐标」并在注释里声明因果由 indexer 负责（【镜像】`sfa_cp.py:1243-1246`） | remap 的可复用部分只有「全局坐标 → rank 本地坐标」这一段（【镜像】`:998-1045` 的 fp32 回退版）；压缩平面的索引语义要另写 |
| 5 | **mask / 窗口参数**：SFA 的 DCP 路径把 `sparse_mode` 从 3 改成 0（关掉右下因果裁剪，因为本地 KV 长度与全局 query 长度已不同坐标系，`:1243-1247`）；V4.1 调算子时用 `ori_mask_mode=4` + `ori_win_left=window_size-1`（`dsa_v41.py:559-562`） | DCP 下这些 mask 参数是否仍成立要重新推导：本地索引坐标变化后，窗口/因果语义可能失效 |
| 6 | **共享 long_kv**：V4.1 只有 `is_kv_source` 的层写 cache，其他层读共享缓存（`dsa_v41.py:261-281`、`models/deepseek_v41/model.py:668-712`） | 这是「层间共享」，不是 rank 间共享：DCP 下每 rank 依旧只写/读自己那份分片，归属公式不变 |
| 7 | **返回 LSE 的需求**：SFA DCP 只返回合并后的 out；【新版】kernel 已支持 `RETURN_LSE`（`ops/triton/dcp/dcp_a2a.py:241`） | 若 V4.1 的 MTP/spec decode 需要合并后的 LSE，直接选用带 `RETURN_LSE` 的版本 |

## 9. 证据、复算与未确认

**离线数值复算**（纯 CPU，无设备、无 numpy/torch 依赖）：脚本 `/home/chiro/tmp/dcp_merge_probe/merge_probe.py`，输出 `run_20260929.log`。它复现了 pack/combine 的下标映射与权重公式（DCP=8、H=64、I=32、KV=256、topk=16、因果前缀），结论：

| 检查 | 结果 |
|---|---|
| a2a 后目的 rank d 的 `(scatter=j, token=t)` 都来自全局 head `8d+j` | OK |
| 无 sink（SFA 形态）合并 vs 全局参考 | 8.78e-16 |
| 每 rank 都带 sink（错误做法） | 2.09 / 0.845 量级错误 |
| 合并时只加一次 sink（方案 B） | 1.07e-15 |
| 只有归属 rank 带 sink（方案 A） | 9.5e-16 |
| 单 rank 非空、其余 `lse=-inf` | 4.4e-16 |
| 8 rank 全空 | 输出全 0，与参考 0 一致；方案 A 下归属 rank 的 `lse = sink` |

**未确认（落地前必须补测，不要猜）**：

1. 方案 A 的占位 sink 取值：`-inf` 还是有限大负值（`-1e30`）在 `npu_sparse_flash_mla` 里是否安全、是否会让 LSE 变 NaN。
2. 全 -1（无命中）的 top-k 下，算子返回的 LSE 是 `sink` 还是 `-inf`（决定方案 A 是否要在空分片路径回退到 B）。
3. V4.1 的 SWA 平面在 DCP 下是否也按 `(pos//I)%dcp` 切、窗口 key 是否可能落在别的 rank（若不，窗口与合并都对不上）。
4. `compress_ratio=2` 的压缩平面索引 remap 公式（【镜像】的 remap 只处理 token 坐标）。
5. 「wq_b 的 head 分片顺序 == 列并行 rank 顺序」这一前提，建议用一次小张量核对。
