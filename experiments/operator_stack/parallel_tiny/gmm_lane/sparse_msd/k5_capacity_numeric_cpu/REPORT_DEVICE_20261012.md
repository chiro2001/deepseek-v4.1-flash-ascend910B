# K5 容量线设备阶段：device 数值门通过 + 最小基准（20261012）

2026-10-12，K5 容量数值线在 a3-21 的 **physical4（NPU 2 chip 0, 0000:95:00.0）** 上完成设备阶段：
两个几何（target36/E48、draft15/E16）各 5 个固定 case + 2 个非法 count 负例 **全部通过**，
planner frame 与冻结 CPU oracle **逐字节相同**，SDK/raw native/candidate 三方 **0 ULP**；
另在同一实例上做了最小 operator-window 基准。admission / performance / formal 标志全部 false。

这一阶段同时关掉了 CPU 线留下的四个缺口中的三个半：`change_header` 首次对真实 CANN 9.1.0 原生源运行、
bridge 的 workspace 字节来自新几何 kernel 的真实编译元数据、`weights_cpu` 载荷路径首次执行、
device 数值门完成（覆盖口径见"未验证声明"）。

## 环境与资源

| 项 | 值 |
|---|---|
| 主机 / 设备 | a3-21 / physical4 = NPU 2 chip 0，Health OK，启动与释放两次 fresh smi 均无进程 |
| 容器 | `dsv41-real-stack-tp8-20261010-a321`（privileged，`/work` = 项目根，仅透传设备节点，未改动容器全局 `ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15`） |
| CANN / 工具链 | `/usr/local/Ascend/cann-9.1.0`；planner 用 `ccec --cce-aicore-arch=dav-c220-vec`，GMM 用 OPC（`asc_opc`，`--soc_version=Ascend910_9382 --op_mode=dynamic`） |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post4；借用 Torch 已初始化的 context 与 stream |
| workspace | 两几何的原生 workspace 均为 117440512B；容量 workspace = 112MiB + frame（640B / 256B） |
| 权限 | 无 reset、无清缓存、无停止他人进程/容器、无跨 chip；每窗口结束都有释放回执 |

## 证据

| 文件 | 说明 | SHA256 |
|---|---|---|
| [evidence/k5_capacity_device_numeric_20261012.json](evidence/k5_capacity_device_numeric_20261012.json) | 数值门结果（10 case + 4 负例 + 编译/bridge/loader 元数据 + 尝试历史），64829B | `d5f39ec970a2a2292b7aa6a92f514848ea60a0c50e1825f5456f6a22b32307ca` |
| [evidence/k5_capacity_device_timing_20261012.json](evidence/k5_capacity_device_timing_20261012.json) | 最小基准（引用数值窗口根与 SHA），13281B | `a3d1083abea3b63313b39fb5a424efd4abca5a8882b62e85febbc19fde927f27` |
| [device/receipts/](device/receipts/) | 编译/bridge receipt 原文、失败尝试历史、tiling 几何解码 | 见各文件 |
| 远端原始日志 | 数值窗口运行时日志 44229656B（SHA `48dbb0ab3a361c96c9be2b6d3fa3c9388644aaad50c06215af0a8cfdad35f551`）；基准窗口 43472497B（SHA `c255c5dc842027bf8e041b88f232dc578e030472ce1f7fb94bc6b3dd9e78fc5c`） | 留远端 job 根，未回传 |

脚本（本目录内，审计用 SHA256，前 24 字符）：
`capacity_device_prepare.py` `4f81dafda4cd34778c018496`、`capacity_bridge_build.py` `a23d142ff75fecefee3cb8a5`、
`run_gate_device.py` `51ecc09fec409fc3e324a01e`、`run_host_window_p4.py` `f1c354977579abdd0b69a30c`、
`collect_device_evidence.py` `2c5ada4655625430a5f77756`。数值窗口实际使用的 gate 脚本归档为
[device/runtime_capacity_gate_used_v4.py](device/runtime_capacity_gate_used_v4.py)（`cecda8c7a68829101fc0d704`）；
最终版 [device/runtime_capacity_gate.py](device/runtime_capacity_gate.py)（`94853e8fee690cf75af33524`）
与它的差异仅限：`k5_probe_bind` 的 ctypes argtypes 笔误修正（只影响基准相位）与 `timing_only` 相位开关；
数值路径逐字节相同。

数值窗口根：`/work/gmm_lane_sparse_msd_k5_capacity_p4_20261012_v4`（`runtime_terminal.json` SHA
`d8e56217b1d1a68eddd5ba33f6e9f4db381ea7edfb7e68446a3641a34d26ff2b`）；基准窗口根：`..._v5`
（`runtime_terminal.json` 已在 compact 头部记录，运行时日志 SHA 见上表）。

## 编译（此前从未执行的部分）

`source_factory.change_header` 首次对真实 CANN 9.1.0 原生源运行：冻结 `prepare_source.py` 先复制
原生源并做 M1 稀疏插入（输出与冻结 `change_header` 逐字节校验一致），再套容量 `replacement_map`
（8 个替换针各自断言唯一命中）与 frame 前缀（`user1 += 640/256`）。17 个原生数学函数体 SHA 不变。

| 产物 | target36/E48 | draft15/E16 |
|---|---|---|
| planner（`gmm2_sparse_row_plan`, dav-c220-vec） | 34568B `d2b2a8c9e1f48f3a…` | 34568B `06272d94585b7503…` |
| candidate（`Gmm2SparseMsdDown_candidate_v6`, MIX, blockDim24, 1:2） | 23488B `b60d8484f0907e55…` | 23488B `fa693448bef761b3…` |
| native control | 19304B `855b4ddcdb0a6004…` | 19304B `8d01580cd9531ac9…` |
| 常量 tiling（2032B，native==candidate 逐字节） | `b131490eeffa60ba…` | `5ca85302e83ce05f…` |
| bridge（容量适配 + v5 借用结构） | 37760B `3fbbb042d53ba913…` | 37760B `e8d69813fd83013d…` |

新几何的真实 workspace 字节取自 OPC 产出的 `kernel_meta/*.json`（两几何均 117440512B），
不再沿用 M1 常量。tiling 解码（[receipts/tiling_geometry_decode.json](device/receipts/tiling_geometry_decode.json)）：
与冻结 M1 tiling 相比，target 仅 2 字节不同（u32 字段 56/72：6→36），draft 3 字节（字段 0：48→16，
字段 56/72：6→15）——即内核几何守卫读取的 groupNum/M 常量与容量几何一致，稀疏调度块已按 K5 几何"上膛"。

## 数值门（physical4，真机）

输入为冻结固定输入（每 case 的 x/token/counts SHA 与 CPU 证据逐位一致），权重为目标 rank0/layer0
真实 W2（原 loader 变换 + NZ cast + INT32 view）。每 case 依次运行：原 SDK `torch_npu.npu_grouped_matmul`
（生产同路径）、raw native A1/A2、candidate（planner + 稀疏内核，同 stream）。

| 几何 | case（R） | frame vs 冻结 oracle | candidate vs raw | SDK vs raw | raw A/A |
|---|---|---|---|---|---|
| target36/E48 | full_last_expert(36)、full_many_slots(36)、full_mixed_counts(36)、partial_unused_rows(35)、zero_rows(0) | **5/5 逐字节相同** | 0 ULP | 0 ULP | 逐位一致 |
| draft15/E16 | full_last_expert(15)、full_many_slots(15)、full_mixed_counts(15)、partial_unused_rows(13)、zero_rows(0) | **5/5 逐字节相同** | 0 ULP | 0 ULP | 逐位一致 |

每 case 另有：x 生命周期（原 GM 指针复用、unused rows 未被改写、恢复后逐字节一致）、
每轮只读参数 seal（全权重 NZ + scale + assist + counts + token 不变）、y 未用行哨兵 0xa5a5 完整、
workspace/y canary 完整。R=0 的 SDK 参考也真实返回，frame 同样逐字节命中冻结 oracle。

负例（每几何 2 例，只提交 planner，不启动 GMM 数学）：单专家 `capacity+1` 与累积溢出
（`counts[0]=1, counts[E-1]=capacity`）都令设备 planner 写出 `status_word=1` 的回退帧，
且与 `numeric_contract.plan_frame` 的 CPU 孪生 **逐字节相同**（4/4）。

权重装载交叉验证：target 侧 K5 loader 计算出的物理 NZ SHA `ffc24b8d049e82b8…` 与 M1 冻结的
`expected_physical_NZ` **逐字节一致**，scale/assist 亦一致；draft 侧（E16）首次冻结 NZ
`e7fffb39c40f0787…`。设备读回与 loader 期望亦逐字节一致（`device_matches_original_loader_bytes=true`）。

## 最小基准（同实例 operator window）

协议与 M1 event 模板同口径：`torch.npu.Event(enable_timing)` 对夹 planner+GMM 提交，采样前恢复 x、
校验/重置/列表 H2D 都在窗口外，D2H 在窗口外；warmup 3 + 交错测量 10；candidate 含 planner。

| 几何 / case | R（槽） | raw native 中位/最小 (ms) | candidate 中位/最小 (ms) | candidate/raw 中位比 |
|---|---|---|---|---|
| target / full_last_expert | 36（1 槽） | 0.1479 / 0.1274 | 0.1877 / 0.1786 | **1.269** |
| target / full_mixed_counts | 36（3 槽） | 0.1430 / 0.1373 | 0.1893 / 0.1754 | **1.324** |
| draft / full_last_expert | 15（1 槽） | 0.1475 / 0.1337 | 0.1895 / 0.1697 | **1.284** |
| draft / full_mixed_counts | 15（3 槽） | 0.1493 / 0.1356 | 0.1816 / 0.1754 | **1.216** |

读数解读（不构成收益声明）：candidate ≈ raw native + 约 0.04ms，与 planner 额外一次 launch 同量级；
在这些容量（R=36 / 15 行）下未观察到稀疏收益——与姊妹线 `k5_route_feasibility_cpu` 的结论方向一致
（真实 target route 中 M1 曾获益的"单槽小 R"模式只占 0.238%）。M1 device 只有数值门、没有 event 数字，
因此这里只做同口径方法说明，不做跨实现数字对比。基准数字是诊断值（队列/提交空闲计入窗口，
非纯 core 时间），`performance_receipt_granted=false`。

## 未验证声明 / 覆盖限制

- **稀疏分支实际进入**仍不可由输出区分：合法输入下 sparse 与 dense 按设计产生相同行。现有证据为
  编译期（源码替换针 + 常量 tiling 几何 = capacity 的字节级解码 + .text/对象差异）与计时侧（planner 成本可见），
  不是指令级 branch coverage；若要直接证明需内核内计数/插桩，属后续独立工作。
- 单设备、单进程、合成 activation、固定输入；未做 capture/replay、真实 SwigluQuant→GMM2→MoE consumer、
  多 token verifier、cache/route 消费者、TP8 整网与 ≤20ms/step 目标；这些都不在本阶段结论内。
- 本次三次失败尝试均为宿主脚本缺陷（缺目录、`plan_frame` 参数序、R=0 空数组守卫），设备侧无异常；
  每次都在独立新根目录重跑（同根内无重试），失败历史见 [device/receipts/attempt_history.json](device/receipts/attempt_history.json)。
- admission / performance / formal 标志一律 false；未 commit/push；未改动其他线的冻结交付物。

## 下一步建议

1. 若需要收益结论：先在真实 route 分布上做稀疏 vs dense 的候选内选择（例如仅对"单槽小 R"启用），
   再进 capture/consumer 与 TP8 整网门；本阶段数字提示在当前容量默认全量切 sparse 无收益。
2. 内核插桩（activeSlots/core 计数）以闭合"稀疏分支实际进入"证据。
3. draft 侧消费证明（`k5_route` 线 draft coverage=0）与真实 activation 仍需独立窗口。
