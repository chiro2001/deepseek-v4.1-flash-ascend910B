# CED-PD 上下文交接（2026-09-25 13:45）

**一句话**：CED 的 8+8 架构（P 跑前 20 层 + D 128-token 有界重放）已跑通，
1M 长上下文的偶发"空回答"故障已把根因收敛到**一条可判定的规律**上——
**请求只要被分配到 id ≥ B 的物理块就必然失败**，B 是 D 侧一个固定常数
（实测 **B ∈ (28204, 29560]**，D 池 29600 时）。精确 B 还在二分中。

## 0. 恢复时的第一件事

```bash
# 1) 只读核对现场
ssh a3-21 'docker ps --format "{{.Names}}|{{.Status}}" | grep dsv41-ced; date -Is'
R=/home/l00886679/tmp/20260924/ced_numeric/pkg_d7953e5_trace/results/ced_trace_d_lo
ssh a3-21 "tail -6 $R/probe/walk.log; grep -c '★' $R/probe/walk.log"
# 2) 若游标行走跑完/中断 ⇒ 决定是继续采样还是直接去查 B 的身份（见第 5 节）
```

现场（2026-09-25 13:46 核）：

| 角色 | 容器 | 芯片 | 端口 | 状态 |
|---|---|---|---|---|
| P | `dsv41-ced-trace-p-20260924_trace` | 0–7 | 18990 | Up 19h，P 池 **29721** |
| D | `dsv41-ced-trace-d-lo` | 8–15 | 18991 | Up 1h，D 池 **29600** |
| proxy | `dsv41-ced-metadata-inline-repeat8-proxy-20260924-1450` | — | 18992 | Up 23h |

后台任务：`$R/probe/walk.log`（22-token 短请求逐步采样"块号 vs 结果"，
`setsid` 脱离会话；13:46 时在 step 540，**尚无翻转**）。

## 1. 代码与文档位置

- 分支 `feat/ced-pd-a3`，远端已同步到 **`2b1fa3f`**（本地另有未推的 tools commit
  已随本轮推送）。
- 主证据（**先读这个**）：
  [`../evidence/ced_swa_clip_ab_clip1_20260924/README.md`](../evidence/ced_swa_clip_ab_clip1_20260924/README.md)
  —— 从 A 臂到 Q 节的完整判据链，含每一条的原始数字与被我推翻的假说。
- 旧交接（历史，别当现状）：[`CED-PD-PAUSE-20260925.md`](CED-PD-PAUSE-20260925.md)、
  [`A3-CED-PD-PAUSE-20260924.md`](A3-CED-PD-PAUSE-20260924.md)。
- 目标其余项的口径：[`CED-PD-ACCEPTANCE.md`](CED-PD-ACCEPTANCE.md)（含验收矩阵、
  性能对照流程、缓存命中为何是代码级阻断）。

带探针的 shadow 包（远端）：
`/home/l00886679/tmp/20260924/ced_numeric/pkg_d7953e5_trace`
（已含 `[CED-SWA-CLIP]`、`[CED-BLOCK-DUMP]`、`[CED-SWA-TRACE]`、`[CED-BLOCKS]`，
env 透传已用 `tools/patch_trace_env.py` 补齐）。

## 2. 故障现象（固定口径）

同一 1M 请求（`prompt_tokens=1019847`）、串行、`temperature=0`、非流式、
`GRAPH=1 EAGER=0` + prompt-tail eager + metadata inline + `V41_CED_SWA_CLIP=1`：

- **形态**：HTTP **200**、`finish_reason=stop`、`completion_tokens=1`、
  `content=null`。即首 token 采样成 **EOS（id=1）**。
- **不是超时**：失败 101.1 s vs 通过 102.7 s（失败更快）；空转 8 分钟
  （> `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480 s`）后相位不变。
- **不是随机损坏**：失败分布**稳定**为"EOS 居首（logprob −2.1~−2.4，约 9~12%）
  + 一小簇泛化字符各约 3%"，而通过时正确答案 logprob ≈ **−0.0002**（约 99.98%）。
  ⇒ 条件是**被削弱**而非被替换成别人的内容（对比：读到别人 KV 时会给出"自信的错答案"）。

## 3. 已确证的规律（本文档最重要的部分）

把**每条结果**与 `[CED-BLOCK-DUMP]` 落盘的**完整 g0 块列表**按时间配对后：

| 结果 | 该请求的 max 块号 |
|---|---|
| PASS | 13356, 21145, 21610, 23967, 25643, 28190, 28204, 28204 |
| FAIL | 29560, 29560, 29599, 29599, 29599 |

⇒ **完全分离**：max ≤ 28204 全过，max ≥ 29560 全失败。
即存在 **B ∈ (28204, 29560]**，**请求一旦分到 id ≥ B 的块就失败**。

**B 不随 D 的池缩放**：C=29024 时整池（max 29023）**16/16 通过**（⇒ B>29023）；
C=29600 时 ≥29560 必失败（⇒ B≤29560）。

这条规律统一解释了此前全部看似矛盾的观测：

| 现象 | 解释 |
|---|---|
| 1M "每 4 个失败"（#4/#8/#12/#16） | R=7989 大，游标每次推进 7989，只有每第 4 个请求把 max 推过 B |
| 900K 失败在 #3/#4/#7（**不规则**） | 游标起点被前序请求带动，只有这几条恰好越过 B |
| 700K / 400K 的 r1 立即失败 | 同上，起点已在高位 |
| 144K / 22-token 从不失败 | R 小，8 次内远未到 B |
| C=29024 全过、C≥29600 必失败 | 池越小整池 max 越低 |
| `MAX_SEQS` 无关 | 与请求槽位无关，只与块号有关 |
| `lo`（D<P）仍失败 | B 与 P 的池**无关**，是 D 侧固定量 |

## 4. 已排除的假说（都带实测依据，别再重复）

| 假说 | 否证依据 |
|---|---|
| D 侧块布局**摘要**（n/first/last/descents）判别 | #4 与 #10 摘要相同、结果相反；#5/#6 更碎却全过 |
| 算子收到的寻址元数据判别 | 通过/失败**逐字段相同**（`seq_len=1019846`、`bt_shape=(1,8192)`、`row_nonzero=[7966,7967]`、`need_blocks=7965..7967`） |
| P→D 传输内容判别 | 通过-失败配对 `long_kv`/`index_k`/`index_scale` maxdiff = **0/0/0** |
| Engram 4-gram / hash 判别 | 含失败请求在内**逐位相同** |
| 墙钟 / 480s 超时 | 空转 8 分钟后第 20 个长请求照旧失败 |
| 短请求推进周期 | 插 3 个短请求后相位不变 |
| `MAX_SEQS` | 改成 1 后失败点仍是 #4/#8 |
| 失败请求有特有日志 | 模板差集**为空**（失败窗口零独有模板） |
| **D 池 > P 池**（我一度最强候选） | `lo` 臂 D=**29600 < P=29721** 仍在 #4/#8 失败 |
| P 侧 `Delayed free` 超时（`Force freed`） | P 全程仅 16 条、集中在 2 个时刻，与失败无对应 |
| D 侧块泄漏 | D 空闲时 `kv_cache_usage_perc=0`、`num_requests_waiting=0` |
| 连接器跨端索引块号 | 代码确认：D 的块号只索引 D 自己的地址、P 只索引 P 的；对端 `num_blocks` **从未被读取** |

## 5. 下一步（按信息量排序）

### 5.1 把 B 夹到 ±20 块（进行中，最便宜）

游标行走（`tools/ced_walk_cursor.py`）用 22-token 短请求（0.6 s/次）采样
`(max_id, verdict)`。⚠️ **注意它的性质**：短请求的块来自**碎片化自由链表**，
块号是**随机跳变**而非单调推进（实测 step 320→540 的 max_id 在
945 / 22979 / 6281 / 25114 / 24874 / 24634 之间跳），所以它是**稀疏采样**——
要命中 [28204, 29560] 这个窄带需要多次采样。13:46 时 540 步、无翻转。

更好的做法（未做）：先发**大请求把游标推到 ~28200**，再换短请求单调逼近；
但要用"发请求前记录 dump 集合、发完后等**新文件**出现"的方式读 dump
（`ced_walk_cursor.py` 已实现该保护；`ced_find_block_bound.py` **仍有读旧 dump 的
bug，不要直接跑**）。

### 5.2 拿到 B 之后：去代码里认领这个数字

候选（按可能性排序，均能解释"只在块号高时静默失败"）：

1. **某类 KV 张量的行数 < `num_blocks`**：压缩组（如 `blocks_per_phys_block`、
   ratio 打包）按比例共享张量，高块号可能越界写到**相邻张量**。
   查 `experimental/ced/mooncake_hybrid_connector.py` 里
   `tensor_num_blocks = single_kv_cache.shape[0]` 与
   `block_size_scale = tensor_num_blocks // self.num_blocks` 的**每一处**
   （`use_hybrid`/`use_mamba`/`use_compress` 三个分支口径是否一致）。
2. **注册窗口偏小**：`lengths.append(share_tensor_stride[0] * self.num_blocks)`
   只注册 `num_blocks` 行；若共享张量实际行数更多，块号 ≥ 某值时传输落在**注册区之外**。
3. 分配器/页对齐边界。**注意这不是"容量不足"**：1M 请求只需 7989 块，
   在 29600 块池里有 3.7× 余量，且无排队、无泄漏。

判读方法：把 B 与这些表达式算出的数逐一比对（例如某个 `shape[0]`、
`num_blocks × blocks_per_phys_block`、注册长度换算出的块数）。

### 5.3 之后才谈交付

拿到稳定配置后按 [`CED-PD-ACCEPTANCE.md`](CED-PD-ACCEPTANCE.md) 跑
144K/1M 四针、流式、多轮；**缓存命中是代码级阻断**（CED 启动器硬门 `PREFIX=0`，
且 D 侧预清零只覆盖 `get_unhashed_block_ids`），需要独立原型臂。
受控性能对照未做（只有指示性 **2.8×**：CED 1M ≈102.7 s vs 全 40 层 ≈288.8 s，
两臂配置不同，不能当结论）。

## 6. 工具清单（`tools/`）

| 工具 | 用途 | 状态 |
|---|---|---|
| `ced_seq_probe.py` | 串行重复探针，带 `token_ids`/logprobs | 可用 |
| `ced_layer_trace_sequence.sh` | 逐请求序列 + 快照改名保留 | 可用 |
| `ced_walk_cursor.py` | 短请求采样 (max_id, verdict) | 可用（有读新 dump 保护） |
| `ced_find_block_bound.py` | 跳跃+行走二分 B | **有 bug，勿直接跑** |
| `ced_length_sweep.py` | 同一 D 上扫多长度 | 可用（已修"截断丢问题"的 bug） |
| `ced_block_dump_analysis.py` | 分析完整块列表 | 可用 |
| `ced_layout_ab_analysis.py` | 块形态 ↔ 结果配对 | 可用（条数不等时**拒绝执行**） |
| `ced_log_template_diff.py` | 失败/通过窗口日志模板差集 | 可用 |
| `ced_pool_arms_experiment.sh` | 池大小双臂编排 | 可用（四道硬门） |
| `ced_p_enlarge_experiment.sh` | 移 P 池的因果实验 | 已就绪未跑（现已被 5.2 取代必要性） |
| `ced_swa_clip_verify.py` | 裁剪修复的离线判据 + 静态 lint | 可用（`--lint-code --lint-server`） |
| `ced_pd_acceptance.py` | 端到端验收 runner | 可用（`--selfcheck`/`--calibrate-only`） |
| `selfcheck_pkg.sh` | 发布包自检 | 需 `TMPDIR=<有空间的分区>`（本机 `/tmp` 满会假报） |

## 7. 环境坑（省时间用）

- **`/tmp` 曾 100% 满**（他人文件），会让 `selfcheck_pkg.sh` 报 3 项假故障。
  用 `TMPDIR=/home/chiro/tmp/ced_selfcheck` 跑即可；**不要删他人文件**。
- **layer/SWA 探针在 graph capture 期间调 `.nonzero()`（`aclnnNonzero`）会打死 D worker**
  （Python 侧 try/except 兜不住）。两者都已有 capture 守卫；日常实验不要开
  `V41_CED_LAYER_SNAPSHOT_POS`。
- **大文件走 COS**：`bash scripts/cos-put.sh`（A3 上也能用）；不要再起临时 http server。
- **启动前删干净 D 容器并等 chip8–15 的 `VLLMWorker` 归零**：旧容器会替新容器回答
  18991 的 health，导致"实验看似成功、其实发给了旧实例"（我踩过一次，
  见证据 README 第 I 节）。
- `[CED-BLOCKS]` 探针在**调度进程**里调 `get_tensor_model_parallel_rank()` 会 assert；
  已加 try/except，改动时务必保留。
- 本机 git push 偶发 TLS 抖动，重试几次即可。

## 8. 资源

- a3-21：P(0–7) + D(8–15) + proxy 在用；chip 上除我们外无他人进程。
- a3-22：**我们的容器全部已退出**（端口 18960–18963 无响应，无我们 NPU 进程），
  用户已明确"暂无 tiny 线资源"⇒ **不要**主动重启单 chip 线。chip0/1 始终预留。
