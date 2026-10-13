# DBO 图模式跑通 + 决定性负结论（2026-10-06 深夜）

> 承接 `docs/COMM-STRUCTURE-AND-DBO-REGIME-20261006.md`。
> 本轮把 DBO 从「replay 崩溃」修到「图模式能跑」，并拿到**干净的同配置 A/B**。
> 结论是**负的**：图模式下 DBO 总吞吐只有基线的 **0.58~0.60×**，且**重叠收益为 0**。
> 全部为【实测】。

## 0. 一页纸

| 结论 | 证据 |
|---|---|
| **根因找到**：捕获期 ubatch metadata 张量被回收 ⇒ replay 读无效地址 | 保活后从「MTE 越界/挂死」变为 **4/4 请求成功** |
| 上游源码自己标注了这个不兼容 | `ubatch_utils.py:125` docstring：*"This will create a new tensor … **This will break cudagraph compatibility**"* |
| 判别实验排除「双流并发/共享 workspace」 | 图内两半批**同流串行**后错误一模一样 ⇒ 是地址生命周期，不是并发 |
| **图模式 A/B：DBO 净亏 ~42%** | §3 表 |
| **图内"并发"与"串行"性能完全相同** ⇒ 重叠收益 = 0 | §4 |
| ⇒ **线 B 的 +25~35% 预期被实测推翻** | §5 |

## 1. 根因：捕获期 metadata 张量被回收

证据链（三步，全部实测）：

1. **判别实验**：给 `_run_ubatches_graph` 加 `V41_DBO_GRAPH_SERIAL=1`（图内两半批同流串行，
   无并发、无事件），MTE 越界**一模一样** ⇒ 排除「双流撞共享 workspace」；
2. **上游源码自证**：`vllm/v1/worker/ubatch_utils.py:117-131`
   ```python
   def slice_query_start_locs(query_start_loc, request_slice):
       """
       Note: This function creates a new tensor to hold the new query_start_locs.
       **This will break cudagraph compatibility.**
       """
       return query_start_loc[rsl.start : rsl.stop + 1] - query_start_loc[rsl.start]
   ```
   `_make_metadata_with_slice` 里还有多处 `.clone()`；我们的 `ascend_split_attn_metadata`
   每步新建 `AscendCommonAttentionMetadata`，其张量是**临时分配**；
3. **保活实验**：把 `ascend_split_attn_metadata` 的返回值钉住（`_DBO_KEEPALIVE`，前 5000 条）
   ⇒ **MTE 越界消失，4/4 请求成功**。

产物：`tools/dbo_fix_graph_serial.py`（判别开关）、`tools/dbo_fix_keepalive.py` / `_2.py`（保活）。

> ⚠️ **保活不是正确性修复**：它只保证地址有效，图 replay 时读到的仍是**捕获期的陈旧内容**。
> 真正的修复需要「预分配静态缓冲 + 每步 `copy_` 进去」，本轮未做（见 §5 为什么不做）。

## 2. 图模式 DBO 现在能跑（这是本轮的真实进展）

```
[DBO-KEEPALIVE] 开始保活 ubatch metadata（前 5000 条）
Capturing CUDA graphs (decode, FULL): 100%|██████████| 8/8  … 0 errors
conc=4 → 4/4 请求成功（此前：MTE 越界 / 60 s 静默挂死）
```

## 3. ★ 决定性 A/B（DCP=1、多流开、图模式，两组**逐项同配置**）

命令（两侧完全一致）：
```bash
python3 tools/bench_concurrency.py --base-url http://127.0.0.1:19310 \
        --concurrency 1,4,8 --prompt-tokens 1024 --output-tokens 64
```

| conc | **基线（DBO off）** | **DBO on** | 比值 |
|---:|---:|---:|---:|
| 1 | **38.3** | 37.5 | 0.98× |
| 4 | **98.9** | 59.3 | **0.60×** |
| 8 | **130.6** | 75.7 | **0.58×** |

同样的对照在**关多流**配置下重复了一遍，结果一致（96.9→57.7 = 0.60×；129.3→73.9 = 0.57×）：

| conc | 基线（nostream） | DBO on（nostream） | 比值 |
|---:|---:|---:|---:|
| 4 | 96.9 | 57.7 | 0.60× |
| 8 | 129.3 | 73.9 | 0.57× |

数据：`~/tmp/base_dcp1_ms.json`、`~/tmp/dbo_graph_ms.json`、
`~/tmp/base_nostream.json`、`~/tmp/dbo_graph_conc2.json`。

conc=1 不触发 ubatch（8 token < 阈值 32）⇒ 0.98×，证明**非 ubatch 路径没有回归**。

## 4. ★★ 决定性的第二组数据：并发 = 串行

同一套 DBO 配置，只切 `V41_DBO_GRAPH_SERIAL`：

| conc | 图内**同流串行** | 图内**双流并发** |
|---:|---:|---:|
| 1 | 36.0 | 36.0 |
| 4 | 58.0 | 57.7 |
| 8 | 75.1 | 73.9 |

**两条曲线完全重合** ⇒ 把两半批放进两条流并发执行**没有带来任何重叠收益**。

这与上一轮的 micro-benchmark（合成 matmul，6 流 1.37×）形成鲜明对比。
【推断】原因：
* ACL graph 在 replay 时对两条流内部仍按依赖顺序执行，两个 ubatch 之间**没有可利用的独立工作**；
* 即使有，通信只占步长 10.2%，重叠收益上限约 +10%，而拆 batch 的损失（见 §5）远大于它。

## 5. 为什么净亏：拆 batch 的代价 > 重叠收益上限

【推断，与实测一致】把一次前向拆成 2 个 ubatch 后：

1. **MoE / GEMM 效率下降**：每家半批的 token 数减半，grouped GEMM 的算术强度与专家利用率下降
   （tiny 上尤其明显）；实测总吞吐直接掉到 0.58×；
2. **通信次数翻倍且每次更小**：每半批各自做 allreduce，形状更小 ⇒ HCCL 效率更低；
3. **收益上限只有 ~10%**（通信暴露 2.5 ms / 24.6 ms），远不足以补偿上面两项。

**方法论教训（重要）**：
> 微基准里的「1.37×」是**固定总工作量切多流**，每个 kernel 的效率不随 shape 变化；
> 而真实推理里「拆 batch」会**改变 kernel 本身的效率**。
> ⇒ **不能用合成微基准外推 ubatching 在真实模型上的收益。**

## 6. 结论与建议

1. **线 B（ubatching/pingpong）按当前设计应停止投入**：图模式下净亏 42%，
   且已证明「重叠」在真实依赖链上不发生。继续做正确性修复（静态缓冲）
   只会让它**从"崩溃"变成"正确地慢 42%"**。
2. 若仍想走吞吐路线，需要的是**另一种设计**（例如：不拆 batch，而是让**不同请求**走不同流
   并让每个流各自跑完整前向——但那等于 DP，与 DP2TP4 的收益重叠，应直接比较 DP 方案）。
3. **建议把资源转回线 A**：目前唯一有干净暴露度证据、且零精度风险的项是
   **A1' `ScatterNdUpdateSk` → 官方 `npu_scatter_nd_update_asc`（真实暴露 0.721 ms，+2.9%）**。

## 7. 环境状态

| 项 | 状态 |
|---|---|
| tiny | 本轮结束时按 `~/tmp/launch_tiny_prof.sh` 恢复常规配置（health=200、容量 3,403,198、无 `--enable-dbo`） |
| tp8k5（chip8–15） | **全程未动** |
| 容器内补丁 | `model_runner_v1.py` 的 `DBO-KEEPALIVE` 与 `npu_ubatch_wrapper.py` 的多处补丁**保留**（未开 DBO 时不生效），`parallel_state.py` 已回退 |

## 8. 复现

```bash
# 判别：图内串行 vs 并发
ssh a3-21 'bash ~/tmp/launch_tiny_graph_serial.sh'   # SERIAL=1
ssh a3-21 'bash ~/tmp/launch_tiny_graph_nostream.sh' # SERIAL=0（并发）
# 保活（根因验证）
ssh a3-21 'docker exec dsv41-tinyspark python3 /tmp/fix_dbo_keepalive2.py'
# 同配置基线
ssh a3-21 'bash ~/tmp/launch_tiny_dcp1_base.sh'
```
