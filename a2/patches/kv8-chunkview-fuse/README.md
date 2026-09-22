# `kv8-chunkview-fuse` —— 把 int8 档 C 的 decode 回退打掉 **60%**（`80.421 → 32.20 ms/step`）

> **来源**：`SWA_COMPACT`（chunk 视图，`logs/104`/`105`）+ `FUSE_MULTIROW`（融合接线与多行推广，`logs/106`）
> + `FUSE_TUNE`（`swa_table` 列并行栅格，`logs/113` 轮次）。
> **状态**：**8 卡真权重 + 生产图模式 + 四轴同开下实测通过**（三轮 `failed=0`、精度判据全过）。
> 本目录是**自包含交付件**：两个替换文件 + 接入说明 + 判据 + 回滚。
> 标记：**【实测】/【推断】/【未确认】**。

---

## 0. 一句话

四轴同开（`ENGRAM=1` × DRAM 卸载 × int8 档 C × draft 入图）的 decode 端到端实测：
**8K `75.911 → 27.905` / 32K `77.325 → 29.226` / 128K `80.168 → 31.950` ms/step**
（同工具 `t_quote.sh`、同机、同端口）⇒ **快 2.51–2.72×**；
**相对"加 int8 之前"的历史基线（30.45/31.52/30.92）：8K 快 8.4%、32K 快 7.3%**。

---

## 1. 两个问题，两份改动

### 问题 ①【实测·根因】`index_select` 在**非连续视图**上会**物化整个平面**
生产的 SWA int8 面是**混合槽页上的 `as_strided` 视图**（`core/deepseek_v41.py:374-381`）：
`stride(0)` = 整个槽页字节数（131,072 / 147,712），而该面自身载荷只有 **65,536 B**。
`dsa_v41.py` 的 `kv8_gather_pages` 对它做 `torch.index_select` ⇒
**`aclnnIndexSelect` 无条件先 `l0op::Contiguous(self)`**（`ops-nn/index/gather_v2/op_api/aclnn_index_select.cpp:186-187`）
⇒ **把整个平面读/写一遍，然后才选出 3 页**。

| 池页数 | 平面大小 | @643 GB/s 预测 | 实测（`SWA_COMPACT`） | 偏差 |
|---:|---:|---:|---:|---:|
| 4,096 | 268.4 MB | 417.5 µs/层 | **417.4** | 0.02% |
| **7,938（生产）** | **520.2 MB** | 809.1 µs/层 | **808.8** | 0.04% |

★ **触发条件 = "源视图是否连续"**，不是跨步距离（`gap 512` 与 `gap 65,536` 同价 422/421 µs；
66,560 的"紧凑页"仍 443.7 µs）。★ **成本 ∝ 池总大小，与"这次取几页"无关**
（同一池选 3 页 vs 24 页只差 0.3%）。

**改法**：把内层换成 `kv8_chunk_view`（整块 storage 的**连续**视图 + `ids = phys*per_page + [0..payload)`）
⇒ **809 → 8.3 µs/层**，且 **`torch.equal = True`（逐比特相同）**、**零容量代价**。
★ 这**不是"凑合的快路径"**，而是**回归官方 layout**：cannbot 的 DeepSeek-V4.1 官方参考实现
（`cann-recipes-infer/models/deepseek_v4_1/models/modeling_deepseek.py:936-937`）同样用
`cache.index_select(0, block_ids).reshape(...)`，前提是 **`cache` 的 block 维在最外层连续**
（`[num_blocks, block_size, N, D]`）。

### 问题 ②【实测】融合接线让读侧 rebuild 从 **几十个 torch 算子** 变成 **2–3 个 kernel**
`KV8_fuse`（`logs/026`）已把 `kv8_ori_plane` / `_kv8_cmp_plane` 融成 Triton kernel，
但它的 guard 是"纯 decode 形状"（`query_rows != num_reqs`）⇒ **生产是 spec-decode（1 请求 × 6 行）
⇒ 两道 guard 都命中回退 ⇒ 收益 0**；而且回退时**丢 `rows_bound`/`graph_safe`** ⇒ 捕获期 `EE1016`。

`FUSE_MULTIROW` 把 kernel **推广到多行**（生产形状）并做了**与 graphsafe 的正确合并**：
- SWA 面：栅格改 `(num_reqs, PP×(block_size//ROWS))`，`PP` 由 host 上界算（生产 **3**），band 起点用 `q_len` 在 kernel 内由 `query_start_loc` 差分（device 侧）
- cmp 面：`block_table` 行改用 host 预计算的 int32 偏移；重编号写回 `q*topk+t`
- ★ **回退路径 `rows_bound`/`graph_safe` 原样转发**（这是能上车的前提）

### 问题 ③【实测】`swa_table` 的栅格是"**一个 program 干一行 × 64 次循环**"
`grid=(num_reqs,)`、`BLOCK_W=128`、`width=8192`（运行时实参，不是 constexpr）
⇒ 单 program **串行 64 次** 128 宽掩码 store，而每层真正非零只有 ≤PP=3 个。
**改法**：列并行 `grid=(num_reqs, width//2048)`、`BLOCK_W=2048`
⇒ 表构 **13.17 → 1.39 µs/层**、整条融合 **19.69 → 6.88 µs/层**
⇒ **可回收 0.40–0.53 ms/step**，且 **54/54 例 `torch.equal` 全 True**
（3 个块宽 × B∈{1,2,8} × spec∈{0,5} × 9 个长度）。

---

## 2. 本目录的两个文件

| 文件 | md5 | 说明 |
|---|---|---|
| `dsa_v41.py` | **`30ecf49b3fe11fb704fe505425b2dfa5`** | 基线 = `S_graphfix/pkgs/pkg-kv8pf` 的 **`94aeebb7`**（graphsafe 生产版）+ **2 个互不相交的 hunk**：① `kv8_gather_pages` 的 chunk 连续视图（`:360-392`）② 文件尾（`:1662-1701`）的**融合接线**（在 `KV8_PREFILL` 段之前；该段与基线**逐字相同**） |
| `kv8_fuse_triton.py` | **`9bdcdaf54fd8b128540009e7137fb3dc`** | 融合 kernel。基线 = `KV8_fuse` 的 `6ce00b8f` → `FUSE_MULTIROW` 的 `8057b3eb`（多行推广）→ 本件**再叠 `swa_table` 列并行栅格**（相对 `8057b3eb` 只改 **39 行 / 6 处 hunk**，`first/bpr/base/val` 一字未动） |

★ **上车前必须核对这两个 md5**（本仓已栽过 `081` 合并件过期 / `093` dsa 挂错份 /
`099` 别人的 health 三次"改了没生效但没人知道"）。

---

## 3. 怎么接（A3 8 卡 runner 的挂法）

```bash
# ① 造完整包（S 包整目录 + 只替换两个文件 —— ★ 不能只放一个文件，
#     否则 runner 的 `int8 挂载源数` 会从 7 掉到 6 ⇒ 自检门 FATAL，见 logs/112）
S=$HOME/projects/dsv41-upstream-pr/agents/S_graphfix/pkgs/pkg-kv8pf
R=$HOME/projects/dsv41-upstream-pr/agents/MERGED_TBL
rm -rf $R && mkdir -p $R && cp -a $S/. $R/
find $R -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
cp -f <本目录>/dsa_v41.py          $R/shadow/vllm_ascend/attention/dsa_v41.py
mkdir -p $R/out
cp -f <本目录>/kv8_fuse_triton.py  $R/out/kv8_fuse_triton.py

# ② 起臂（四轴同开的既有链，唯一增量是这几个 env）
export DSA_SRC=F R8_DSA_SRC=F R8_KV8_FUSE_PATCH=1 R8_KV8_FUSE_TRACE=1
export R8_KV8_DIR_D="$R"  R8_KV8_FUSE_DIR="$R/out"
export R8_KV8_FUSE_DSA="$R/shadow/vllm_ascend/attention/dsa_v41.py"
export R8_DSA_ALLOW_DIRS="$R"
export MODEL=$HOME/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq
export IMAGE=quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3
export PATCH_MODE=mount          # ★ 见 logs/112：不显式给会落到 serve_a2.sh 的 baked 默认
TAG=r8-tbl PORT=8051 KEEP=1 MAX_LEN=133120 MAX_SEQS=32 BAT_TOKENS=8192 \
  KV_MEM_BYTES=4294967296 OFFLOAD_BYTES=23068672000 \
  TIER=C ENGRAM=1 DRAFT_GRAPH=1 GRAPH=1 EAGER=0 ENGRAM_DEVICE_INDEX=0 \
  PROMPTS=6 PROMPT_TOKENS=131072 REPLAY_PROMPT_TOKENS=65536 ROUNDS=3 \
  bash ~/tmp/20260922/merge4axis/scripts/run_4axis_arm.sh
```

★ A2 侧（`PATCH_MODE=baked`）要走镜像烘焙路径，见 `patches/README.md` 与 `docs/A2-DEPLOY-NOW.md`。

---

## 4. 判据（每一条都能第三方复算）

| # | 判据 | 期望 |
|---:|---|---|
| 1 | 起臂前 **容器内** `md5sum .../attention/dsa_v41.py` | **`30ecf49b3fe11fb704fe505425b2dfa5`** |
| 2 | 起臂前 **容器内** `md5sum .../attention/kv8_fuse_triton.py` | **`9bdcdaf54fd8b128540009e7137fb3dc`** |
| 3 | `serve.log` 里 `int8 挂载源数` | **≥6**（自定义包目录不在 runner 的模式里，见 `logs/112`） |
| 4 | G2b 门（实际挂的 dsa 在 graphsafe 谱系） | 过（路径白名单 **或** 容器内内容含 `rows_bound`+`KV8_GRAPH_SAFE`） |
| 5 | ★ **融合真生效**：`grep -c "\[kv8fuse\]" serve.log` **且** 看得到 `swa rows grid=(..., ...) pp=3 use_qlen=True rows_bound=6` | >0 且形状对（生产 trace 实测 `grid=(32,192)`） |
| 6 | ★ **`swa table grid` 应是 `(num_reqs, 4)`**（而不是一维） | 列并行版生效 |
| 7 | 起服期错误码 `EE1016 / 507057 / EH0012 / 207001 / capture failed / KeyError:` | **全 0** |
| 8 | ★ `[bneck] hp`（8 卡 8K quote 口径） | **≈27.9**（修复前 **75.9**） |
| 9 | `GPU KV cache size` | **427,643**（本修复**不动容量** —— 预注册） |
| 10 | 压测三轮 `requests_failed` | **0 / 0 / 0**（`ROUNDS=3`） |
| 11 | 卸载判据 `CPU_to_GPU` / `hits` | **>0 / >0**（与修复前 `1.7008429056e+10` / `724224` **逐字相同** ⇒ 未动卸载路径） |
| 12 | ★ 精度：`text_correctness_probe.py --mode all` | **题库 10/10** + **prefix-pair 三发逐字相同**（`distinct=1`） |
| 13 | ★ 精度：单卡 `torch.equal`（chunk vs page，int8+scale 两面、两种步长） | **True** |

★ **反假阳性**：`replay1 == replay2`（**随机 token id** prompt）在本仓是 ✗ —— **修复前三条臂同样 ✗**
（`logs/110 §2`）⇒ 那是**既有噪声地板**，**不是本修复引入的回退**；语义判据（第 12 条）才是硬判据。

---

## 5. 回滚

去掉那 4 个 env（`DSA_SRC=F` / `R8_KV8_FUSE_PATCH=1` / `R8_KV8_DIR_D` / `R8_KV8_FUSE_DIR`）即回到
`S_graphfix` 的 graphsafe 生产版 `94aeebb7`；或把 `dsa_v41.py` 换成 `94aeebb7` 那份。
★ **两个文件必须同时换**（只换 dsa 会让接线段 import 不到 kernel）。

---

## 6. 未确认 / 边界（**不许被上面的 ✅ 掩盖**）

1. **本件的 8 卡实测是在 `KV_MEM_BYTES=4 GiB`（7,938 页）口径上做的**；A2 是 **14.40 GiB**
   ⇒ 该修复**对池大小是平的**（320/1024/4096/7938/13366 页实测 153–159 µs/层）⇒ 【推断·强】A2 上同样有效，
   但**未在 A2 上实测**。
2. **`swa_table` 列并行那一格（−0.4~0.5 ms）只在单卡验证过**（54/54 `torch.equal`），**8 卡未验**。
3. **`hp` 与设备时间的传导率**：单卡算出的 5.8–13.7 ms 与 8 卡实测的 2.0 ms 有差额，
   `FUSE_TUNE` 判为"落在设备工作→`hp` 的传导率上"【未确认】。
4. **P0（与本文无关但同批）**：spec-decode 在 `max_model_len` 边界**不做裁剪** ⇒
   请求走到 `max_model_len − 6` 以内会让**整个引擎崩掉**（`logs/109`）。
   ★ 操作规程：`prompt + max_tokens ≤ max_model_len − 32`。
