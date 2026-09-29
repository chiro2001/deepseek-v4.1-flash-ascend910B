# V4.1 DCP8 性能优化记录（2026-09-29 晚）

**接续**：`docs/V41-DCP-HANDOVER-20260929.md`（该文档的目标/容量/正确性结论仍然有效，
**但 §5.2 的性能消融表作废**，原因见 §1）。
**本次目标**：把 DCP8 相对 DCP1 的 +12~13 ms/step 压下来。

---

## 0. 一句话结果

| | DCP1 | 旧 overlay | 去 LSE gather | +ori 折叠 | +q 提前 gather/后处理切片 |
|---|---|---|---|---|---|
| 版本 | — | `96820b3` | `3278910` | `43ce760` | `6dcfc5d` |
| ms/step | **30.83** | **42.30** | **36.18** | **35.73** | **35.18 / 35.35** |
| tok/s（单流，API 口径） | 31.1–31.4 | 23.64 | 27.64 | 27.99 | **28.43 / 28.29** |
| 相对旧 overlay | — | — | −14.5% | −15.5% | **−16.4% ~ −16.8%** |
| A | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| KV 容量 | — | 8,634,871 | 8,634,871 | 8,634,871 | 8,634,871（未变） |
| 长针（多选） | — | 6/6 | 6/6 | 6/6 | **6/6** |

★ **与 DCP1 的差距：30.83 → 35.35 ⇒ +4.5 ms/step（1.147×）**，换来 **6.95× KV 容量**。

### 最后两档的负载说明
`35.73/35.82`（cap53）是在 **load ≈ 3** 下测的；`35.18/35.35`（cap55）是在
**load 25~41**（别人在跑 static kernel 编译）下测的 —— 本版是在**更差**的环境下更快。

口径：`TP8/DP1/DCP8`、`BAT_TOKENS=2048`、`--no-async-scheduling`、`PREFIX=0`、
`MAX_LEN=1M`、5 GiB 池、设备 8-15、流式相邻 token 间隔中位数（丢弃前 8 个）。

★ **两项"不要再优化"的硬结论（子代理 2-chip 图级实测，见 `a2sim-ref/dcp2perf/REPORT_AB3.md`）**：
* **q 的 head all_gather 已在地板上**：空 collective（16 B）45.9 µs vs 真实形状
  `[1,8,512]`（8 KB）46.1 µs ⇒ 纯**每-call 延迟**，0% 是数据量；要到
  `[1024,8,512]`（8 MB）才爬到 74.1。all_to_all 是 50.0（更贵）。
* **all-to-all 替代不可行**：合并需要"固定 head 切片、对全部 KV 分片求和"，
  只能整行（要 gather q）或整列（要读全部 8 份 KV，每层 ~128 KB 且随上下文增长）；
  "只算 1/8 head"等于只算 8×8 块矩阵的对角块、丢掉 56 块。

★ **会话间漂移已证实且已处理**：8-chip 上同一版代码跨会话能差 2 ms/step
（cap46 曾在低负载下测到 38.31）。2-chip 线用**同一进程内逐轮交错**的图级夹具
给出精确增量（±2 µs/层），8-chip 只用于**配对**（同一时段、同一负载下比两版）。
上表的 42.30 / 36.18 / 35.73 都是这样得到的；cap46 的 38.31 属于**未配对**读数，作废。

---

## 1. ★ 方法论纠正：交接文档 §5.2 的消融表是**无信号**的

旧文档写「`no_pack=1` 42.09 / `skip_2nd=1` 42.49 / base 43.19 ⇒ 集合通信+第二次调用
合计 ≤2 ms」。**这个结论不成立**，证据：

1. 我按同样的文件开关做了 **3 臂 × 3 次交替** A/B（同一会话内交替，消除漂移）：
   base **42.97** / nopack **43.04** / skip2nd **42.94**，三臂差 ≤0.07 ms/step。
2. **机制**：decode 走 `cudagraph_mode=FULL_DECODE_ONLY`，`breakable_cudagraph.py:102-103`
   在 `mode == CUDAGraphMode.FULL` 时**直接调用原函数、不做 eager break** ⇒
   attention 整体在图捕获区内。`_perf_flags()` 这种 Python 分支**只在捕获那一刻求值**，
   之后每次 replay 根本不执行 Python。
3. 反证：同一开关 `timing=1` **能**打出 `[V41-TIME]` 行 —— 因为 prefill 是 eager，
   不走图。所以「开关写进去了」≠「decode 阶段生效」。

⇒ **所有"改文件即刻生效"的 decode 阶段消融都无效**。要测 decode 的增量，只有三条路：
(a) 开关在**捕获前**就位（起服时序）；(b) 设备侧 profile；(c) 离线图级序列夹具。
本次走的是 (c)。

---

## 2. 归因（2-chip 线，`a2sim-ref/dcp2perf/REPORT.md`）

| 项 | 实测 | 标记 |
|---|---|---|
| DCP2−DCP1 整序列（40 层，真 SMLA + 4 collective） | **+11.42 ms/step = +286 µs/层** | 【实测】 |
| 其中：逐元素链 + 布局拷贝 + 第二次 SMLA + build_mask | **224.6 µs/层（79%）** | 【实测】 |
| 其中：4 个 collective 的**净**增量 | **61.0 µs/层（21%）** | 【实测】 |
| `npu_sparse_flash_mla`：head 8→64 / cmp 键 0→512 / ori 块 1→128 | **全平坦**（单次 ~76 µs，Δ≤5 µs） | 【实测】 |
| 图内每个**串行**小算子 | **13.2 µs** | 【实测】 |
| 图内每层每多 1 个 collective | **61~71 µs** | 【实测】 |
| 8-chip 比 2-chip 每层多 18 µs | 未解释 | 【未确认】 |

**结论**：「每 rank 算 64 head 所以贵 8 倍」被证否；瓶颈是**合并链的串行关键路径**
（约 17 个有效串行步 + 4 次通信）。优化方向 = **砍串行节点数**，不是砍算力。

---

## 3. 本次改动（4 组，全部数学等价，两个 commit）

`bd96e74` + `eb401e4`，都在 `experimental/v41-dcp/overlay/`。

| # | 改动 | 位置 | 省什么 |
|---|---|---|---|
| 1 | `sinks` 的 head 维 all-gather + `full_like` 改为**按 attn 对象缓存** | `dsa_v41.py` | 每层每步 1 个 collective（≈61 µs/层） |
| 2 | 路线 B 下**不再构造 `token_mask`** | `dsa_v41.py` | ~12 个算子/层（**从来没用过**：消费条件是 `ori_lse is None`） |
| 3 | `(1−1/dcp)` 折进 ori 项 + 去 `clone` + `clamp_min` 代 `where` | `dsa_v41.py` | 归约后 4 个重算子 → 2 个减法 |
| 4 | `prepare_indexer_indices` 由**两次**改**一次** | `models/deepseek_v41/indexer.py` | 该 triton 核内含 `tl.extra.cann.extension.sort`，是最贵算子 |
| 5 | `finite` 掩码 5 节点 → `nan_to_num` 1 节点；去 `clamp(-80)`；`permute+contiguous` → `reshape` | `dsa_v41.py` | ~7 个算子/层 |

### 关键推导（改动 3）

`all_reduce` 是**线性**的 ⇒ `Σ_r (_onum_r · k) = k · Σ_r _onum_r = k · _n_all`。
把 `k = 1 − 1/dcp` 先乘到 per-rank 的 `_ow`（`[T,H,1]`，便宜）上，
归约后的后处理就从 `scaled − _n_all + _n_all/dcp`（2 个作用在 `[T,64,512]` fp32 上的重算子）
压成 `scaled − _n_all`。离线真值对拍：旧 vs 新相对差 **2.6e-16**，两者 vs 真值均 5.8e-16。

### 关键推导（改动 4）

`prepare_indexer_indices` 不是纯比较 —— 它在 triton 核里对每行做一次
`tl.extra.cann.extension.sort`。旧写法先把**全局**位置喂进去过滤一次，再用局部等价编码
**再过滤一次**。过滤条件单调（全局界 `(p+1)//ratio` 恒 ≥ 局部可见数 `vlc`）
⇒ 直接用局部编码跑一次**逐位等价**（局部编码本就是把 `(p'+1)//ratio == vlc` 代回同一核）。
DCP 关闭 / indexer 复制态时 `_dcp_visibility_positions` 原样返回 `positions`，语义不变。

---

## 4. 待办 / 未解释项

1. **38.31 vs 40.14 的会话间差异**（§0）。已交 2-chip 线用 `dcp_seq_timing.py` 做
   旧/新 overlay 的精确配对判别。判据：同一夹具下 DCP2−DCP1 的每层增量，
   旧版应复现 ≈+286 µs/层。
2. **`_dcp_visibility_positions` 每层重算**（4 个 indexer 层 × ~10 算子），
   同一 step 内 `positions` 对所有层相同 ⇒ 理论上可提到 step 级算一次。
   需要跨模块改签名，暂缓。
3. **第二次纯 ori SMLA**（≈70 µs/层 ≈2.8 ms/step）**暂不可去**：
   合并式需要 `(A, A·O_ori)`，而 A 是**数据相关**的（子代理已实测：只有把池换成全零
   才有解析解 `LSE = log(min(128,L))`，且那时分子项仍是垃圾）。内核又硬绑 ori 非空。
4. **SFA 式 all-to-all**：结构改动大，未评估。
5. overlay 的 release 打包 + push（交接文档 §7 遗留）。

---

## 5. 复现命令

```bash
# 同步 overlay 到 a3-21（md5 守门）
bash /home/chiro/projects/dsv41/a2sim-ref/dcp_sync.sh

# 起服（8-chip，PROFILE=1 可挂 /start_profile）
ssh a3-21 'cd ~ && setsid env DCPMOUNT=$HOME/dcpw AUTO_CHIPS=0 CHIPS="8 9 10 11 12 13 14 15" \
  PREFIX=0 BAT_TOKENS=2048 PROFILE=1 V41_PROFILE=1 EXTRA_KV_ARGS="--no-async-scheduling" \
  DCP_EXTRA_ENV="V41_DCP_ALLOW_CAPACITY_PROBE=1" bash ~/dcp_stage_capacity.sh \
  > ~/dcp_capNN.nohup.log 2>&1 < /dev/null &'

# 性能（交替 A/B，三元组口径）
python3 ~/tmp/dcp_ab.py ab --reps 5 --arm base:

# 正确性（短问答 + 长上下文多选针；**必须**用多选格式，见交接 §4）
docker exec dsv41-dcpcap bash -lc 'cd /workspace && python3 dcp_correctness.py \
  --tok-dir /home/l00886679/models/out/v41-flat-verify3 --corpus /workspace/hongloumeng.txt'
```
