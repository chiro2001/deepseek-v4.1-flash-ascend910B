# 旧基础补丁收益审计：证据层级与当前未证实项

日期：2026-10-13。只读汇总 `reports/` 指定的九份历史报告、`patches/PATCHES.md`、当前 base2 service identity、A2v4 frozen input 来源身份，以及近期 TP8 operator 复核边界。未 SSH、未访问 API/NPU、未运行旧基准或修改源码。本文汇总历史证据，不重建原始样本的联合吞吐，也不把不同测量层的收益相加。

## 结论先行

- 旧补丁历史至少包含三种不同证据：真权重/未 profile 的全服务请求窗口、dummy/synthetic 的设备账目或单卡测试、以及整套补丁的跨会话窗口。它们不能互相代替；独立 kernel 或设备账目收益不自动等于 HTTP `ms/step` 收益。
- **当前 baseline2 service identity**（service ID `3cbd844d9f684c43867b4b85c9c78ebe`，config/runtime fingerprint `79513a0f20932cf696374573323b715aab0d04df774721fbee92f2f780da4319`）确认 `a2-tp8-a3sim`、physical 8–15、TP8/DCP1、八 rank ready；环境中 `V41_MOE_COMM_ALLGATHER=1`、`V41_MOE_MASK_RANGE=1`、`V41_ROPE_IDXSEL=1`、`V41_O_PROJ_2D=1`、`V41_QLI_NO_CANDIDATE=1`、`V41_ENGRAM_JIT=1`、`V41_ENGRAM_DEVICE_INDEX=0`，新 `STACK_TP8_ARM=tp8base`、exact-mask/metadata batching 两个开关为 0。fingerprint 与 CPU plan 相同且标记 `simulated_hardware=true`，不是实测硬件指纹。**这证明配置身份/ready，不证明每项现今独立收益。** 本次 2K 客户端八请求已经完成，不得描述成全服务零 generation；也没有当前进程内单项 A/B。
- JIT 的特殊限制：历史 `plan` JIT 只有 `LOCAL_OWNER=fast` 才实际进入 numpy/numba plan 路径；`JIT=1` + `LOCAL_OWNER=on` 时 `plan` 仍未生效。当前 base2 identity 没有 `LOCAL_OWNER` 或 `V41_ENGRAM_LOCAL_OWNER_FILE` 的实际采用值。只读 `host_engram_mode.md` 检查了**当前 D** 的默认 `/tmp/v41_engram_localowner` 文件在 host 和 D 中不存在；该日志/检查快照在唯一客户端请求之前，且 env override 未读取。它证明那个时点的默认路径缺失，但请求中运行时实际读取/采用何种 mode 仍未证明。故不能由 `V41_ENGRAM_JIT=1` 宣称当前 `plan` JIT 已兑现历史收益；不能说整个服务没有 generation，因为 base2 2K×8 client 已运行。
- 旧基础补丁 #9 的 `V41_MOE_MASK_RANGE` 是 token-dispatcher 的 expert-range 比较，不是近期新 exact probability mask `tp8mask`；新 `STACK_MOE_MASK_ENABLED` / `STACK_TP8_ARM=tp8mask` 是不同调用点与实验臂。新 `tp8meta` 也是 host-side slot metadata batching，不等于旧 block-table `V41_SLOT_MAP_FUSED`。近期复核已说明历史 A=1 arms 不能直接外推到 DSpark 六行。
- `consolidated-6patch-result.md` 和 `milestone-ms-target-met.md` 给出的若干三列“中位数”分别对 `ms/step`、A、tok/s 独立取中位，不是同一观测行的联合三元组；与 `tok/s = 1000×A/(ms/step)` 的自洽要求不一致。这里不照抄为报价，不从边际统计反算吞吐。若要历史全栈窗口作三元组，须回到原始 joint receipts；本审计未读取/重算那些 JSONL。

## 候选与历史证据强度

| 项目/旧门控 | 历史最佳证据层 | 历史可支持的结论 | 当前服务未证明什么 |
|---|---|---|---|
| MoE AllGather（`V41_MOE_COMM_ALLGATHER=1`；历史启动器别名 `MOE_AG=1`） | `moe-allgather-breakthrough.md` 有真服务、未 profile 的客户端多窗口数据与 route/设备账；比较跨启动/会话。报告记 8K/32K/128K 的服务级变化，并有同 prompt/seed/48 token 的输出对照；不是当前 base2 同进程单变量 A/B。 | 对当时 TP8、AllGather 路径、真模型配置支持方向为正；AllGather 改变 token 集合方式，MoE 小项总账下降，KV 容量也增。历史报告在 128K 给出不同会话/复验窗口的结果；这里只作为历史窗口趋势，不合并构造一条新 tuple。 | 现在 flag=1 不证明相对于当前 base2 service 的边际收益；没有 current-worker 单变量 A/B，历史设备 MoE `9.41→5.15 ms/step` 也不是当前客户端节省。另 `FUSED_MC2`、多流与性能窗口配置耦合，不能拆成 AllGather 单一贡献。 |
| WOA 2D（`V41_O_PROJ_2D=1`） | `f3-wo-a-2d.md`：128K、真权重、单流独占、unprofiled quote；A/B/A2 同次试验序列，B 与 stock 方向一致。`ms/step` 与 `cli_p50` 是不同测量字段。 | 支持在 `n_local_groups==1` 时把 batch-matmul 改 2D 有小幅服务级收益；适用几何明确。 | 报告明确没有确定性对账，跨 kernel 累加顺序可能有 ULP 差异；不能将数学等价论证当作 full consumer exactness。当前 base2 flag=1 也不是当下单项 A/B。 |
| 旧 MoE range（`V41_MOE_MASK_RANGE=1`） | `moe-mask-range-verified.md`：dummy 128K 两份 profile，同 7912 anchors，per-pass 设备直方/归一化；不是真权重服务时延。 | 机制变化被事件数对齐证实（Index/IndexCheck 替换为 MaskedFill/Cast）；profile 中 decode 窗归一化约 `−0.51 ms/step`。`PATCHES.md` 另记后续质量门 GSM8K/Vision 和 SAFE-L1 int32 域改动；这是正确性/实现迭代信息，不把 dummy 设备 timing 升格成当前 HTTP 收益。 | 当前 range flag=1 不证明现进程独立收益；近期 exact `tp8mask` 是新 probability mask custom op，开关、调用点、形状守卫和实测完全不同，旧 `−0.51` 不能转借或与 exact-mask 历史 `+0.15` 相加。 |
| RoPE index-select（`V41_ROPE_IDXSEL=1`） | `rope-idxsel-verified.md`：dummy 128K profile，per-pass 归一化；decode 归因使用 profile 最后约 3 秒，约 96 decode steps。线 3 的 21 项数值测试 `max_abs=0` 是分开的合成/独立测试。 | 旧 gather/index 链 kernel 数下降；profile decode 账估 `−0.45~−0.51 ms/step`，支持设备窗口方向和机制。报告自己指出按 anchors/间隔选窗会把 prefill chunk 错当 decode，需用 span 区分。 | 这不是当前真权重 HTTP A/B，也不直接说明客户端每步省同样时间；独立线 3 数值测试不等于当前模型消费者/服务正确性门。 |
| QLI no-candidate（`V41_QLI_NO_CANDIDATE=1`） | `qli-no-candidate-verified.md`：dummy 8K/192-output device profile，9460 anchors，per-pass 设备账；“真 8 卡 `−0.9 ms/step`”在该报告中是线 3 投影，不是本报告的观察结果。数值只做了单卡合成输入 `torch.equal`，且需要 aclgraph 重捕获。 | profile 内 QLI static op 次数不变、每 op 从 99.28 到 50.29 µs，设备归一化约 `−0.494 ms/pass`，支持候选路径机制/方向。 | 不支持真权重整网或当前 HTTP 的 `−0.9` 结论；dummy 权重/cache行为不同，单卡合成 equal 不替代八 rank 真模型消费者。当前 flag=1 不是 current isolate A/B。 |
| Engram JIT `hash` / `plan`（`V41_ENGRAM_JIT=1`） | `engram-jit-verified.md`：A3 dummy-device 三臂阶段计时，同会话；hash 在 JIT=1 臂由约 0.427 到 0.078 ms，vjC 的 plan 约 0.262→0.068 ms。 | 支持 hash JIT 在该 dummy/插桩路径中被调用并缩短 host phase；支持 plan JIT 只有 `LOCAL_OWNER=fast` 时被调用。`LOCAL_OWNER=on` 的 vjB plan 约 0.262/0.268，与关闭 JIT 的 vjA 0.261/0.252 近似，报告明确判为未生效。 | JIT flag 本身不能证明 plan 路径采用 fast。当前 base2 identity 仅记录 JIT=1 与 device index=0，不记录 local-owner 文件/实值；现场缺失与运行时 mode 未证前，不声称 plan `−0.192 ms/step` 或 `hash+plan −0.54` 已在当前 baseline 兑现。即使 hash 项也没有 current service 单项 A/B。 |
| Engram host path 量化（`engram-final-quantification.md`） | 报告含 `SPEC_EAGER=0`、32K、同会话 runtime hot-switch（3 发）的 `stock/nohost/nocomm/faked2h` 服务窗口，以及 host/device phase 账；不是当前 base2 的同配置。 | 支持当时整条 Engram host 路径、D2H 等待与路线的暴露/遮蔽分析；`nohost−stock` 是去掉整条 host path 的诊断边界，不是一个可部署补丁的边际收益。报告估真实可回收量约 0.6–1.0ms，并明确其他变化会接棒。 | 不能把 `nohost −2.77ms` 当 JIT patch 或当前服务可拿到的性能；报告也明确 D2H 不能当真实可回收工作量。 |
| 合并 6-patch 全栈（`consolidated-6patch-result.md`） | 真权重、128K×8 客户端，同会话内为完整 patch bundle；与 AllGather/F3+mask 的参照来自较早会话，不是同会话单变量切换。 | 当时完整 bundle 的服务窗口中位 `ms/step=32.740`；报告明确 32.85→32.74 的差只有 −0.11ms、在噪声内，而两项 device 账预期约 −1ms 未在该服务窗口观测到。这是“设备估计不等于端到端收益”的直接反例。 | 不支持六补丁各自的当前边际收益，也不支持按设备项求和推算全栈。该文中 median A 与 median tok/s 独立汇总，不能作为同一观测行联合 tuple。 |
| 里程碑 full-patch + `LOCAL_OWNER=fast`（`milestone-ms-target-met.md`） | 真权重 128K×8 全栈服务窗口；8 行请求结果及字段级统计。总体 bundle，不是 JIT 单项 A/B。 | 支持当时这套完整配置下出现过若干 ms/step 行和高接受请求；服务窗口 A 的方差很大。报告的 `ms/step`、A、tok/s 是分别汇总的边际统计，不是联合观测行。 | 不把字段中位数拼成 tuple，也不由 A、ms 中位反算 tok/s。需要联合三元组须回到原始 row-level JSONL；该 full-patch window 不能证明当前 base2 local-owner mode 或任何历史补丁独立收益。 |

## 当前 baseline 身份与历史配置关系

当前身份以 [base2 GET identity](base2_runtime/base2_get_identity.json) 与 [base2 client result](baseline2_client/RESULT.md) 为准；A2v4 [旧 service selection](../../../delivery_matrix/evidence/a2sim_tp8_v4/prestop_snapshot_20261013/a2_runtime/service/service_selection.json) 是 frozen prompt 的历史输入来源，不是当前性能 baseline。base2 D 物理 8–15，A2-sim/A3 hardware、TP8/DCP1、八 rank ready，Engram index=0；identity config fingerprint 等于 CPU plan fingerprint，不视为硬件 runtime 指纹。也不把它与 A3-node1 [formal best service](../formal_a321/best_service_20261013/BEST_SERVICE.md)（index=1）混作同一身份。

当前 base2 identity 没有 `LOCAL_OWNER`/`V41_ENGRAM_LOCAL_OWNER_FILE` 的实际采用值；`PATCHES.md` 中写路径/默认快照不是运行时文件内容证据。当前 D 的默认 local-owner file 检查与实际采用的 fast mode 是两件事；本报告不拿 JIT=1 推导 plan-JIT 兑现。

旧基础补丁清单有 14 个编号条目（13 个 vllm-ascend payload 项加 1 个 vllm admission gate patch）；这是包装/挂载目录清单，不代表 14 个独立性能实验。`PATCHES.md` §1 汇总早期测得收益，部分来自不同历史层级。

旧 `V41_MOE_MASK_RANGE` 对 token dispatcher 中专家区间做比较，近期 `tp8mask` 是 probability exact mask/cast custom op；旧 `V41_SLOT_MAP_FUSED` 是 block-table kernel fuse，近期 `tp8meta` 是 host metadata slot batching。新候选及几何限制详见 [`history_recent.md`](history_recent.md)；不得把旧基础包 flags 当作新候选已运行，也不得把 A=1 旧候选增益外推到 DSpark 六行。

## 主要原件

- MoE AllGather：[`reports/moe-allgather-breakthrough.md`](../../../../reports/moe-allgather-breakthrough.md)，`patches/PATCHES.md` §1 #7。
- WOA/F3：[`reports/f3-wo-a-2d.md`](../../../../reports/f3-wo-a-2d.md)，`patches/PATCHES.md` §1 #8。
- 旧 mask / RoPE / QLI：[`reports/moe-mask-range-verified.md`](../../../../reports/moe-mask-range-verified.md)、[`reports/rope-idxsel-verified.md`](../../../../reports/rope-idxsel-verified.md)、[`reports/qli-no-candidate-verified.md`](../../../../reports/qli-no-candidate-verified.md)，`patches/PATCHES.md` §1 #9–11。
- Engram JIT / host：[`reports/engram-jit-verified.md`](../../../../reports/engram-jit-verified.md)、[`reports/engram-final-quantification.md`](../../../../reports/engram-final-quantification.md)，`patches/PATCHES.md` §1 #1–6、#12–13。
- 合并全栈：[`reports/consolidated-6patch-result.md`](../../../../reports/consolidated-6patch-result.md)、[`reports/milestone-ms-target-met.md`](../../../../reports/milestone-ms-target-met.md)。
- 补丁目录清单：[`patches/PATCHES.md`](../../../../patches/PATCHES.md)。
- 当前 base2 baseline：[base2 identity](base2_runtime/base2_get_identity.json)、[Engram mode 只读核查](base2_runtime/host_engram_mode.md)、[base2 2K client result](baseline2_client/RESULT.md)。A2v4 [旧 service selection](../../../delivery_matrix/evidence/a2sim_tp8_v4/prestop_snapshot_20261013/a2_runtime/service/service_selection.json) 仅作冻结 prompt 的来源参照。

本审计不复测历史环境、不声称当前每个 flag 的单项收益，也不从不同统计量的中位数构造吞吐或联合 quote。
