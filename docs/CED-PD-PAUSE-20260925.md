# CED-PD 暂停交接（2026-09-25 02:0x，VPN 断开）

**暂停原因**：用户报告 VPN 断开且无法主动恢复，要求暂停任务。
本页只记录"停在哪里、恢复后第一件事做什么"，判据细节在
[`../evidence/ced_swa_clip_ab_clip1_20260924/README.md`](../evidence/ced_swa_clip_ab_clip1_20260924/README.md)。

## 一、当前代码状态

- 分支 `feat/ced-pd-a3`，远端已同步到 **`0d4c358`**（本地另有 `b3c4243`
  之后的几个 commit 已推；`git rev-parse HEAD origin/feat/ced-pd-a3` 应一致）。
- 本轮新增并已推送：
  - `tools/ced_log_template_diff.py`：失败/通过请求窗口的日志模板差集。
  - `tools/ced_ms4_smallpool_experiment.sh`、`tools/ced_pool_threshold_experiment.sh`：
    池大小/`MAX_SEQS` 的单变量编排。
  - `tools/ced_block_dump_analysis.py`：读**完整**块列表判读阈值假说。
  - `tools/selfcheck_pkg.sh`：改用 `${TMPDIR:-/tmp}`（本机 `/tmp` 满时会假报 FAIL）。
  - `experimental/ced/mooncake_hybrid_connector.py`：新增 `[CED-BLOCK-DUMP]`
    （`V41_CED_BLOCK_DUMP_DIR`，落完整 g0 列表）。
  - `scripts/serve_a2.sh`：透传 `V41_CED_SWA_CLIP` / `_SWA_TRACE` / `_BLOCK_TRACE` /
    `_BLOCK_DUMP_DIR` / `ENGRAM_HIST_TRACE_POS`。
- `experimental/ced/dsa_v41.py` 的 `[CED-SWA-CLIP]`（`V41_CED_SWA_CLIP`，默认 1）保留：
  **它是真实缺陷的修复（replay 首 query 越界读），但对 1M 偶发故障无效**，
  见下面结论表。

## 二、1M 偶发故障：已确证的结论（全部真机实测）

固定条件：A3-21 真权重 TP8 P（chip0–7）+ TP8 D（chip8–15），BF16 KV、
`MAX_LEN=1048576`、`PREFIX=0 SPEC=0`、`GRAPH=1 EAGER=0`、prompt-tail eager、
metadata inline、`V41_CED_SWA_CLIP=1`；同一 1M 请求（`prompt_tokens=1019847`）、
串行、`temperature=0`、非流式。

**故障形态**：HTTP 200、`completion_tokens=1`、`content=null`、首 token = EOS。

### 已排除（都有实测依据）

| 候选 | 结论 |
|---|---|
| 块布局**摘要**（n/first/last/descents） | 不判别（#4 与 #10 摘要相同、结果相反） |
| 算子收到的寻址元数据 | 不判别（通过/失败逐字段相同：`seq_len=1019846`、`bt_shape=(1,8192)`、`row_nonzero=[7966,7967]`、`need_blocks=7965..7967`） |
| P→D 传输内容（`long_kv`/`index_k`/`index_scale`） | 不判别（通过-失败配对 maxdiff=0/0/0） |
| Engram 4-gram / hash | 不判别（含失败请求在内逐位相同） |
| 墙钟 / `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480s` | 排除：空转 8 分钟后第 20 个长请求照旧失败 |
| 短请求（≤128 token） | 不推进周期（插 3 个短请求后相位不变） |
| `MAX_SEQS` | **不是**那个"4"（改成 1 后失败点仍是 #4/#8） |
| 失败请求的特有日志 | 没有：模板差集显示失败窗口**零个**独有模板 |

### 关键正面结果：**D 的 KV 池大小是判别量**

| D 配置 | `num_blocks` | 12 个长请求里的失败 |
|---|---:|---|
| `MAX_SEQS=4`（标准） | 30082 | 4 个（#4/#8/#12/#16） |
| `MAX_SEQS=1` | 30200 | 2 个（#4/#8） |
| `MAX_SEQS=1` + 小池 | 19494 | **0** |
| `MAX_SEQS=4` + 小池 | 19494 | **0** |

周期严格是"每 4 个**长**请求"（短请求不计数），且 413 s ≈ 4 × 103 s。

### 正在验证的假说：**D 的池是否大于 P 的池**

当前 P 的池是 **29721**（注意：文档早先引用的 30079 属于更早的 P 实例，已更正）。
标准臂 D=30082，比 P 多 **361** 块；小池臂 19494 远小于 P。
因此候选判别量收紧为「**D 的 `num_blocks` > P 的 `num_blocks`**」。

## 三、恢复后第一件事：判读池阈值实验

> **2026-09-25 12:1x 更新（VPN 恢复后已核）**：该实验**已跑完**，但结论要分两半。
>
> - **A 臂有效**：`below`，D=**29024**（< P 29721），**8/8 通过**。
> - **B 臂（`above`）无效**：容器**从未创建**（`No such object`）。
>   原因是 `serve_a3.sh` 检测到卡被**我们自己的 `below` 容器**占着而拒绝启动
>   （fail-closed 行为正确）；但旧脚本的就绪检查只 `curl 18991/health`，
>   **旧容器替它回答了**，于是 `above_1..8` 全部发给了仍在运行的 `below`。
>   ⇒ 合并计算：`below`（D<P）**连续 16/16 通过**。
>
> 所以"D>P 即失败"目前是 **4/4 一致**（30082、30200 失败；19494、29024 通过），
> 但仍与"C≈30k"混淆。修正版脚本 `tools/ced_pool_arms_experiment.sh` 已在
> 12:13 启动**决定性双臂**（`hi`≈29850 > P，`lo`≈29600 < P，只差 250 块），
> 日志 `/tmp/pa_exp.log`。它加了四道硬门防止再次复用旧容器（详见
> `../evidence/ced_swa_clip_ab_clip1_20260924/README.md` 第 I/J 节）。

实验已在 a3-21 上用 `setsid` 启动（**VPN 断开时可能仍在跑，也可能被中断**）：

```bash
# 结果日志
/tmp/pt_exp.log
# A 臂（D≈29024 块，< P 的 29721）
/home/l00886679/tmp/20260924/ced_numeric/pkg_d7953e5_trace/results/ced_trace_d_below/
# B 臂（D≈30500 块，> P 的 29721）
/home/l00886679/tmp/20260924/ced_numeric/pkg_d7953e5_trace/results/ced_trace_d_above/
```

**已观测到的部分结果（断开前）**：

- A 臂配置核对通过：`max-num-seqs 4`、`num_blocks: 29024`（**小于** P）。
- `below_1`…`below_4` **全部 PASS**（`completion=7`）。
  ⚠️ 标准臂在 `#4` 是失败的，所以 `below_4` 通过**支持**阈值假说；
  但 A 臂只跑到第 4–5 个就断了，**还不能定论**（需要 8 个全过）。
- B 臂尚未开始。

**恢复步骤**：

1. 只读核对 `/tmp/pt_exp.log` 的尾部与 `ced_trace_d_below|above` 的 `serve.log`，
   确认实验是否跑完；**不要**直接重启。
2. 若 A 臂 8/8 且 B 臂出现每 4 个失败 ⇒ 阈值假说成立，根因锁定在
   "D 池 > P 池"这条关系上，下一步查两边 `num_blocks` 参与的计算
   （连接器注册长度 `share_tensor_stride[0] * num_blocks`、握手元数据
   `MooncakeAgentMetadata.num_blocks`、以及块 id 分配上限）。
3. 若 A 臂也出现每 4 个失败 ⇒ 阈值假说否掉，回到"D 侧池几何"本身，
   优先用 `[CED-BLOCK-DUMP]` 的完整列表比对（`tools/ced_block_dump_analysis.py`，
   要带 `--total-blocks`），因为摘要已被证明不可用。
   判读命令：
   ```bash
   python3 tools/ced_block_dump_analysis.py \
     --dump-dir <run>/blockdump --results <run>/probe/seq \
     --total-blocks <该臂实测 num_blocks>
   ```

## 四、恢复时的环境提醒

- a3-21 上的 P（`dsv41-ced-trace-p-20260924_trace`，18990）与 proxy
  （`dsv41-ced-metadata-inline-repeat8-proxy-20260924-1450`，18992）在断开前仍在运行；
  恢复后**先只读核对**容器、端口、health，再决定是否复用。
- 带探针的 shadow 包：`/home/l00886679/tmp/20260924/ced_numeric/pkg_d7953e5_trace`。
  它已含 `[CED-SWA-CLIP]` 与 `[CED-BLOCK-DUMP]`；新增 env 需用
  `tools/patch_trace_env.py` 重新透传。
- **a3-22 的单 chip tiny 线已全部停止**（容器全 `Exited`，端口 18960–18963 无响应，
  无我们的 NPU 进程），用户已确认"暂无 tiny 线资源"，**不要**主动重启。
- 本机 `/tmp` 满会假报 selfcheck 失败：用
  `TMPDIR=/home/chiro/tmp/ced_selfcheck bash tools/selfcheck_pkg.sh`。
- 与本次调查无关但会干扰的已知坑：layer 探针/SWA 探针在 graph capture 期间调用
  `.nonzero()`（`aclnnNonzero`）会**打死 D worker**，两者都已加 capture 守卫；
  日常实验不要再开 `V41_CED_LAYER_SNAPSHOT_POS`。

## 五、目标其余项的差距（未动）

`144K/1M 正确性`、`流式`、`多轮`待本故障收敛后重跑；`缓存命中`是**代码级阻断**
（CED 启动器硬门 `PREFIX=0`，且 D 侧预清零只覆盖 `get_unhashed_block_ids`）；
`受控性能对照`未做（只有指示性 2.8×：CED 1M ~102.7 s vs 全 40 层 ~288.8 s，
两臂配置不同）。
