# DSpark async 正式 API 窗口：46.49 → 32.97 ms/step（−29.1%），token 逐位不变

2026-10-13 · a3-21 physical8–15 · 正式40层/W4A8_DYNAMIC/Engram int8/strict · DSpark 真实 ON（graph）
· 目标 ≤20ms/step **未达成**（当前 32.9705）

## 1. 长请求配对判定（bench，同快照 `src_dspark_native_async_v1`）

仅差 `DSPARK_ASYNC_COUNTER/DSPARK_ASYNC_SCHEDULING` 两个 env 门，其余（权重/几何/图/负载/客户端）相同：

| arm | job | exit | ms/step | A | tok/s | rounds | tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| sync@256 | `formal_dspark_native_graph_sync256_v1` | 0 | 45.75276821705427 | 5.930232558139535 | 129.61472691676502 | 258 | 1530 |
| async@256 | `formal_dspark_native_graph_async256_v1` | 0 | **41.80449965909091** | 5.795454545454546 | **138.6323145287125** | 264 | 1530 |

- 数值门：`async_reference_passed=True`，6/6 请求 token 与 sync 参考**逐位一致**、max logprob delta **0.0**。
- 稳态轮中位（receipts）：async 39.4–40.8ms vs sync 42.0–47.2ms；async 每请求多 1 个 overshoot 轮（44 vs 43）。
- 证据 `evidence/formal_a321/dspark_native_graph_{sync256,async256}_v1/`。

## 2. async 正式 API 桥变体

新快照 `src_dspark_native_graph_api_async_v1` = 冻结 api_v1 + async 三件
（`formal_dspark_counters.py 4877900b…`、`contract.py d9271e2a…`、`worker.py 8dc2192a…`）
+ 桥两处守卫：

- `runtime_guards.py`：`COUNTER_SHA256`/`FORMAL_WORKER_SHA256` 换成与快照字节一一绑定的 async 值。
- `service_bootstrap._guard_config`：`async_scheduling` 只有在 `DSPARK_ASYNC_SCHEDULING=1` 且
  `DSPARK_ASYNC_COUNTER=1` 时才允许；同步合同保持原样（真值表 4/4 正确：sync/async 各自 ACCEPT，
  交叉 REJECT）。`native_api_snapshot.json` 五个文件哈希与 `frozen_counter_sha256` 同步更新。

CPU 自检：常量与快照字节绑定 True；`test_async_counter_cpu.py` 容器内 **8/8 通过**。

## 3. 正式 API 客户端窗口（8 串行 × 2K 输入/256 输出，审计/profiler 关）

job `formal_dspark_native_graph_api_v4_async`（端口 18583，自有服务已正常关闭，run.exit=0）：

```
(32.97052108069164 ms/step, A=2.9394812680115274, 89.15483200333648 tok/s)
8 请求 / 694 actual target-verifier rounds / 2040 decoded tokens / decode_elapsed_s_sum=22.88154163
clean 7/8、复读 1（与 sync 窗口同口径）；窗口内包含 initial 与 terminal fence（async 语义）
```

对照上一版正式最优（sync graph，`formal_dspark_native_graph_api_v1`，
`(46.48740792151163, A=2.9651162790697674, 63.783213812996586)`）：

**ms/step −13.516886840820159（−29.08%）、tok/s +39.78%、A −0.87%（6 个额外 overshoot 轮）**。

等价性：本窗口 8/8 请求的 token 流与 sync 正式窗口**逐位一致**
（`client/token_identity_vs_sync.json`，逐请求 sha256 相同）。

证据（本地）：`evidence/formal_a321/dspark_native_graph_api_v4_async/`（launched/run.exit/client.exit/
acceptance_small.json/measured_0..7_http.json/warmup_0..1_http.json/service_selection.json/
token_identity_vs_sync.json）。原始 34.4MB `client/acceptance.json` 留远端，
sha256 `496ab1923850f86c651bd2f2abcb59524abc1675cb17fffce67157439d339bff`；
launched.json sha256 `ac3c048f116daf55aa0b39369f34d5ee10617fa3f619155cec47ffb77325a584`。

## 4. 为什么 sync→async 在正式窗口收益（−29%）远大于 A/A 屏幕（−8.6%）

正式 API 负载（A≈2.94、86 轮/请求）里宿主是瓶颈；A/A 屏幕（A≈5.9、43 轮）宿主占比更低。
async 调度把 `_prepare_inputs`/bookkeeping 等宿主段与设备执行重叠，因此宿主占比越高收益越大。
结合设备侧事实：整轮设备忙时 ≥27.3ms（见第 5 节），当前 32.97 里宿主暴露已压到约 5–6ms。

## 5. 目标图（model 48）rank0 全谱与底限量化（既有 profile CSV 离线解析）

每步 **1986 kernel**、span 23.1ms、union **21.59ms**；前几名：

| 组成 | 数量/步 | ms/步 |
|---|---:|---:|
| hcom_allReduce（唯一 collective；40层×2：attn AllReduce + MoE `maybe_all_reduce_tensor_model_parallel`） | 81 | 3.32 |
| GroupedMatmulSwigluQuantV2 | 40 | 3.16 |
| HcPre | 80 | 2.30 |
| MatMulV2 | 97 | 2.15 |
| QuantBatchMatmulV3 | 208 | 2.00 |
| GroupedMatmul | 40 | 1.48 |
| SparseFlashMla | 40 | 1.31 |

整步全流 union **27.29ms**（target 21.59 + draft 2.6 + engram 0.37 + eager 2.76）。
⇒ **即使宿主暴露全部消除、调度完美重叠，轮时仍 ≥27ms**；≤20ms/step 必须先把 6 行 forward
从 21.6ms 砍到 ≤15ms，同时把 draft/verify/pre/客户端 5.5ms 全部藏掉。这是结构化重写级差距。

设备侧最大单块候选 = MoE 通信/融合路径（当前 AllGather 臂的 MoE 相关 ≈7.4ms：GMM+Swiglu+GMM2+
路由+掩码+MoE AllReduce）。已在 profile 快照上启动 `formal_dspark_native_graph_profile_mc2_v1`
（`--extra-env V41_MOE_COMM_ALLGATHER=0`，即 vendor MC2/MegaMoe 路径；该臂在 audit 模式下受
路由导出 ceil/floor 合同阻塞，本 job 为 audit-off 的设备时间线诊断，不作性能声明）。

## 6. 下一步（按期望值）

1. 读 MC2 profile：若目标图 MoE 段显著下降，则重做路由导出合同（接受 floor 分片）+ 该臂自带
   A/A 审计与数值/质量门。
2. 目标图 kernel 融合（RmsNorm+DynamicQuant 配对、HcPre/HcPost、量化链）与 AllReduce 收敛。
3. 宿主残余 ~5-6ms：`.item()`/阻塞 H2D 与小算子折叠（需 with_stack 归因）。
4. 质量门（GSM8K100/100、Vision23/23、模型归属）与正式部署/交付门仍未做。

## 7. MC2（非融合 dispatch/combine）设备诊断：净负，排除

job `formal_dspark_native_graph_profile_mc2_v1`（profile 快照 + `--extra-env V41_MOE_COMM_ALLGATHER=0`，
audit-off 设备诊断；`enable_fused_mc2` 默认 0 ⇒ 实际走的是 MC2=Distribute/Combine 路径，不是
vendor 的 FUSED_MC2/MegaMoe）。同口径 rank0 与基线 profile 对照：

| 指标 | 基线（AllGather 臂） | MC2 臂 |
|---|---:|---:|
| target 图 union | 21.59ms | **24.72ms（+3.13）** |
| pre_target span | 15.90ms | 17.28ms |
| draft 图 union | 2.57–2.85ms | 3.88ms |
| 整步全流 union | 27.29ms | **35.65ms（+8.36）** |

MC2 臂新增 MoE 侧 `MoeDistributeDispatchV2` 40×1.68ms + `MoeDistributeCombineV2` 40×1.34ms +
`hcom_allGather_` 40×0.64ms + `MemSet` 80×0.83ms + `PadV3` 80×0.44ms，只换回 AllReduce
3.32→2.40ms。⇒ **净负约 +3.1ms（目标图）/ +8.4ms（整步）**，与团队此前在代码注释里记录的
"MC2 每 rank 1 token 标量开销"方向一致。**该方向排除**（除未来单独验证 `enable_fused_mc2=1`
的 FUSED_MC2，但需要 contract 变更 + 全链重验证）。

证据：`evidence/formal_a321/dspark_native_graph_profile_mc2_v1/`（launched/result/
offline_parse_manifest/mc2_rank0.json）。

## 8. 主线策略更新

async 生效后宿主暴露只剩 ~5–6ms（32.97 − 27.3 设备忙），**设备侧省时不再被宿主空转吸收**，
因此"设备 −1ms ⇒ 客户端 −1ms"的传导条件成立。据此把已数值验证但当时客户端无收益的
`V41_SLOT_MAP_FUSED=1`（设备 −1.21ms）重新接入 async 正式窗口：
job `formal_dspark_native_graph_api_v5_async_slotfused`（同快照 + 三 env，port 18593）已启动，
期待 ~31.8ms/step 与 token 逐位不变。

**实测（2026-10-13 06:00）：仍为噪声级、未采纳。**

| arm | job | exit | ms/step | A | tok/s | rounds |
|---|---|---:|---:|---:|---:|---:|
| async（基准） | `formal_dspark_native_graph_api_v4_async` | 0 | 32.97052108069164 | 2.9394812680115274 | 89.15483200333648 | 694 |
| async+slotfused | `formal_dspark_native_graph_api_v5_async_slotfused` | 0 | **32.89499576368876** | 2.9394812680115274 | 89.35952717939799 | 694 |

- Δ = −0.0755ms/step（−0.23%），token 8/8 与 v4 逐位一致（`client/token_identity_vs_async_base.json`），
  clean 7/8、复读 1 同口径。⇒ 与 sync 时代的结论一致：该开关在客户端无可复现收益（尽管设备侧确曾 −1.21ms），
  **正式候选保持 `V41_SLOT_MAP_FUSED` 默认 off**；v5 仅作记录（自有服务已正常关闭，run.exit=0）。
- 解释（诚实边界）：async 下设备虽然不再是唯一瓶颈，但 slot mapping 位于"sample→draft→pre_target→target"
  串行链的宿主段内，其 1.2ms 设备收益落在仍有宿主暴露的窗口里，未能整体前移；要吃到它需先
  继续压缩该链的宿主段（第 6.3 项）。
- 证据：`evidence/formal_a321/dspark_native_graph_api_v5_async_slotfused/`。
