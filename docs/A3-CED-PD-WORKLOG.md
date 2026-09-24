# A3 双 TP8 CED-PD 实现日志

## 2026-09-24 暂停前最后一组：metadata-inline D-only 八次

- 用户要求暂停后，A3-21 当前已发出的D-only8组安全收尾，无第9条请求。
  新 D实例没有短针/144K前置，直接用同一原始1M D请求SHA
  `f25d0b8b1a0ea3e4ff37131dc90182a2940ef075f7fc1ab8fcbb7c99cd9f9d8e`
  严格串行8次，均HTTP200、相同128-token replay。
- D1–D3、D5–D7精确返回`RB9N-6014`；D4、D8返回`content=null`、
  completion_tokens=1。**第4/第8次现象可重复，但尚未定位原因**；日志
  没有采样token ID，不能判定为EOS。
- 本组逐条原始request/response、sidecar、replay在A3-21路径
  `/home/l00886679/tmp/20260924/ced_numeric/pkg_d7953e5/src/results/ced_metadata_inline_repeat8_20260924_1450/probe/inline_repeat8/`。
  聚合SHA清单和tar未生成，未同步本地。P/D/proxy暂停时均保持运行，
  配置与恢复顺序见
  [`暂停交接`](A3-CED-PD-PAUSE-20260924.md)。

## 2026-09-24：D 图执行短针单变量对照

- A3-21 真实权重、BF16 KV、`MAX_LEN=1048576`、`SPEC=0`、`PREFIX=0`。
  P 保持同一进程（chip0–7、`GRAPH=1/EAGER=0`），D 先以
  `GRAPH=0/EAGER=1` 在 chip8–15 启动，再只改为 `GRAPH=1/EAGER=0`
  重启。实际 vLLM 命令分别核对到 `--enforce-eager` 与
  `FULL_DECODE_ONLY`，均为 `V41_CED_ROLE=decode` 且加载同一补丁。
- 同一 22-token 校验码原始 JSON 两臂各提交两次，SHA-256 均为
  `f670186097daa964d1eca4c436320e8dcee6dc93cda0ea41b1306224991aa881`。
  eager 两次 HTTP 200、原文均为 `ZQ7K-3341`、usage 22/8/30；graph
  两次 HTTP 200、均生成满 32 tokens 且答案错误，其中一次有 7 个 U+FFFD。
  D 两臂均记录 replay `0..20` 和 8 个 worker 的 21-token replay。
- 原始启动环境、响应 body、P/D/proxy 日志、inspect、哈希在
  [`evidence/ced_graph_ab_20260924/`](../evidence/ced_graph_ab_20260924/)；
  `SHA256SUMS` 在证据目录内执行校验通过。此对照定位的是**短针**的
  D 图模式触发条件，尚未证明 144K/1M 质量恢复，图内部根因也未定。
  下一门是保留 P、恢复 D eager，先过同短针再测历史 144K bigprefill A/B。

**同 P 恢复 D eager 后的功能门：** 短针一次精确通过。历史 bigprefill A/B
使用相同 HLM 语料（SHA-256 `a7fc413bd6e3926482faddf2af9bfb4426e55d8e421481d60100c14160785578`）、
旋转偏移 1,000,000/1,146,000、temperature=0、max_tokens=64、非流式。
A/B body 分别为 144,404/144,131 token，API prompt usage 分别为
144,462/144,188；响应精确为 `ZQ7K-3341`/`VX2M-8890`，均 HTTP 200、
U+FFFD=0，wall 17.673/11.953 秒。D replay 分别为
`144333..144460`/`144059..144186`，各 128 token。完整原始证据在
[`eager_function/`](../evidence/ced_graph_ab_20260924/eager_function/)，
上级证据目录的 SHA256SUMS 已校验。这是当前 CED D eager 的 144K 质量通过，
尚未构成与全 40 层同状态的严格性能对照。

**1M eager 正确性门：** 同一 P/D eager 实例、同一 HLM corpus、offset=0，
四针 body token 数均为 1,019,789，temperature=0、max_tokens=64、非流式。
A/B/C/D 分别精确回答 `ZQ7K-3341`、`VX2M-8890`、`HT4P-5527`、
`RB9N-6014`；4/4 HTTP 200、U+FFFD=0。wall 依次为
105.860/104.815/104.044/103.048 秒，D 每条均 replay 128 token。
完整请求/响应、usage、日志与校验在
[`eager_function/`](../evidence/ced_graph_ab_20260924/eager_function/)。
此结果不证明图模式正确，也不等于同配置全40层的严格性能增益。

**首 token 分叉定位：** 保留同一 P，把 D 切回图模式后，对完全相同 SHA 的
22-token 请求分别设 `max_tokens=1/2` 并取 token logprobs。eager 返回
`Z`/`ZQ`，图模式返回“正确答案”/“正确答案只有一个”；四条请求均
HTTP 200，图臂错误从首个生成 token 就存在。图臂确认 FULL_DECODE_ONLY
捕获 4/4，P/D role、挂载和镜像指纹正确。原始证据在
[`graph_minimal/`](../evidence/ced_graph_ab_20260924/graph_minimal/)。
单卡 tiny trace 表明 D 的未缓存末 prompt token 虽是一 token，attention
仍标为 `num_prefills=1`；实际镜像 runner 却在建立该元数据前按 token 数
选择 FULL 图。此控制流差异是下一单变量修复假说，不等于已证明 stream
依赖正确。子代理并行采集 eager/graph profiler 以查多 stream/event 顺序。

**单 token prompt 尾步消融：** 隔离包
`fix/ced-graph-prompt-tail@777bc73` 在固定 A3 镜像 runner 上增加
`V41_CED_GRAPH_PROMPT_TAIL_EAGER=1` 门。仅当 CED D 仍有未计算的 prompt
token 且该批最大调度长度为 1，强制该步 eager；后续生成仍由 FULL 图执行。
实际 runner 基底 SHA 为 `67035d97…3185e`，补丁后 SHA 为
`bd250a59…c2aff`。同一 P、同 SHA 的 `max_tokens=1/2/32` 分别精确返回
`Z`/`ZQ`/`ZQ7K-3341`，8 个 D worker 均打印
`[CED-GRAPH] one-token prompt tail forced eager`，max2 后实际打印
`Replaying aclgraph`。请求/响应/logprobs/日志及 SHA 校验在
[`graph_prompt_tail/`](../evidence/ced_graph_ab_20260924/graph_prompt_tail/)。
这是短请求上的机制验证；144K/1M、流式、缓存命中和性能尚未证明。

**图补丁长上下文门（同一 8+8 实例）：** 历史同 SHA 的 144K bigprefill
A/B 均精确通过。标准 1M needle A/B/C/D 均以 offset0、body 1,019,789
token、temperature0/max_tokens64/非流式串行测试；A/B/C 精确回答，D
HTTP 200 但返回 `Tech-D9q7Wm`，未命中 `RB9N-6014`，U+FFFD=0。
对应 wall 为 102.622/102.171/101.895/133.175 秒；四针 D 侧都记录
128-token replay 与8-worker forced-eager。D 失败后立即停止后续请求，
保留服务和完整原始现场。当前阶段结论是 **1M 3/4，尚未通过**；下一门
对失败 D 原 messages 仅改 `max_tokens=1` 并取 top logprobs，确认首
token 是否已错，再决定图 decode 或尾步/状态方向。

**同 SHA D 重复性：** 首错 D 后，同实例 `max_tokens=1`、top5 logprobs
返回首 token `RB`（logprob −0.000223，其他候选远低于它）；随后同 SHA
完整 D 重发正确。再连续串行发两次相同原始 D，第一次正确、第二次
错误 `Uhq9-3DqT`。因此完整图补丁臂的 D 在四次中 **2/4 正确**，
不能作为稳定1M交付。四次 P/D 都记录128-token replay 和尾步
forced-eager；KV transfer、容量与错误日志没有解释正确/错误的稳定分组。
两次新连续重复 wall 均约102秒，故也不能把首次错误的133秒或一次
正确重发的224秒直接当作精度原因。下一单变量对照保留 P 原实例，
只将 D 切换为 `GRAPH=0 EAGER=1` 的诊断模式，重复原始 D 四次。

**模式/DSA 重叠消融：** 同一 P 不重启，D 全 eager、原始D同SHA完整请求
连发四次均精确答 `RB9N-6014`，usage均为1,019,847/7/1,019,854。
随后只把 D 改回 FULL 图+prompt-tail 修复、设 `DSA_OVERLAP=0`
（共享专家 `MULTISTREAM=1` 仍开），短针和144K A均精确；1M D同SHA
四次前三次精确，第四次HTTP200却 `content=null`、仅1 completion token，
故结果 **3/4**。此消融不能单独排除其它图内stream/metadata状态，
也不能作为稳定交付。第四条原始响应及全量日志仍在归档中。

**两处模型多流都关闭的图模式消融：** 从上一 DSA_OFF 臂只改
`MULTISTREAM=1→0`，`DSA_OVERLAP=0`不变，FULL decode 图和 prompt-tail
修复不变。短针、144K A精确；同 SHA 1M D四次前三次精确，第四次
HTTP200、`content=null`、completion_tokens=1，结果仍 **3/4**。
D4 原始 choice 的工具/推理/拒绝字段为空或缺失，日志没有token ID或采样
事件，故不把它写成 EOS。DSA_OFF与MS0两臂都在第四次出现这一形态，
但四次样本不足以证明固定第4次触发。`causal_conv1d_update_npu`
回退 warning 在 eager 和图臂各8条，非独有指纹。

**下一单变量（本地包已备、未上卡）：** `fix/ced-metadata-inline@d7953e5`
在 `experimental/ced/dsa_v41.py` 加 opt-in 的
`V41_CED_METADATA_INLINE=1`，使metadata生成在模型runner当前流直接执行，
绕过该builder的独立设备metadata流/external event；常驻buffer、模型
forward及FULL decode图不变。将以 `DSA_OVERLAP=0 MULTISTREAM=0` 的
MS0臂为基线复测，包内DSA SHA
`72788c494bfbb12510ad36687e8578ce1ab85b87220ba72ea4b30cac8de381ac`。
该包已通过本机 `tools/selfcheck_pkg.sh`，实际启动/精度/性能均未验证。

**A3-22 1+1 profiler：** tiny D eager/graph 各自对同一固定请求采集
`max_tokens=1/8`，图臂在采集窗内确认实际 ACL graph replay。tiny 两臂
API 输出一致；graph 的 measured M8 窗口里 SMLA 有 stream2 的 240 次与
stream47 的 40 次，eager M8 的 350 次均在 stream47。两臂都存在 event
record/wait，但 trace 没有 wait 对应 producer 的 event handle，也没有
设备 task 的请求 ID，不能从这些计数认定竞态。完整方法、窗口分析、COS
原始 trace 指纹在
[`ced_tiny_pos21_20260924/`](../evidence/ced_tiny_pos21_20260924/)；
Graph 原始 trace 约 22MB、eager 约35MB，只存私有 COS。此分析与真实
权重的 prompt-tail 单变量修复分别保留结论边界。

## 目标与基线

- 目标：BF16 KV 下，P 的长 prompt 主段只经过层 0–19，P 从 encoder 输出生成层 20 的全局 KV/Indexer K；D 重建 128-token SWA 尾部并用完整 40 层解码。
- 基线：`b529c05` 的 `scripts/serve_a3_pd.sh`，两侧各 TP8、全 40 层、`MooncakeHybridConnector`。2026-09-23 在 A3-21 实测约 144K 正确；BF16＋AscendStore 的 1M 首次回答正确，但 DRAM 取回未验收。
- 实验分支：`feat/ced-pd-a3`，独立工作树 `ced-pd-release`。在功能通过前保留发布分支的原有行为。

## 上游能力边界

- vLLM Core 支持通用 PD、混合缓存分组和外部 KV 连接器。Core 的 DeepSeek V4.1 `swa_bounded_replay` 只在 Model Runner V2 开启，用于**前缀缓存命中后**重建 encoder SWA，不能代替第一次 prefill 后的 decoder SWA 初始化。
- 当前 vLLM Ascend 官方 V4.1 指南只验收同置部署；模型前向对每步仍执行所有 40 层，Ascend V1 运行时没有 `replay_start` 路径。我们本地双 TP8 PD 功能通过属于额外实测。
- 通用 PD 只分开调度/传 KV；获得论文的 prefill 计算收益还需要 CED 专用模型路径。

## 最小实现顺序

1. **源投影数值门。** 层 19 产出的 `(hidden_states, pre_mix)` 是层 20 的输入。新增的 `DeepseekV41DecoderLayer.write_global_source_from_encoder()` 走与普通层 20 相同的 `hc_pre → input_layernorm → compressor → Indexer K/long KV` 路径，暂不跳层。实际 NPU 对照应在同一次 forward 中比较该方法写出的有效 cache 行与普通层 20 随后写出的行；主 KV、Indexer K 和 scale 均须一致。
2. **P 角色的主段截断。** 仅专用 P 实例对 prompt 主段停在层 19，之后调用上述投影。模型仍加载完整权重，先验证计算量与缓存正确性；避免在同一阶段改变权重布局。不能向普通客户端返回半模型 logits。
3. **D 的尾部重放。** 维持现有 `N−1` 交接边界，D 在 global KV 加载后对最后 128 token 重放，恢复各层 SWA/环状态；重放不得覆盖已收到的 global KV。先让 D 重算 lower+upper 尾部，正确后可传末尾 `H20` 只算 upper。
4. **协议与接口。** `MooncakeHybridConnector` 当前按 13 个缓存组交接并让 P 采样 1 个 token 完成 API 请求。P 跳上半层后需要显式“仅完成 KV 生产”的响应，或末 token 保持完整计算；不得用错误 logits 伪装正常模型响应。连接器要明确哪些组在 P 写入、哪些组由 D 重放补齐。
5. **验收。** 先短输入逐 token 对照，再做 144K/1M、流式、多轮、外部池取回。记录 P 各层实际处理 token 数、D replay token 数、P/D transfer 字节、prefill 时间、TTFT、TPOT、吞吐与错误指纹。只有 BF16 全链路通过后再叠 KV8、DRAM 分层和 P 权重裁剪。

## 当前进度

- 已在 `patches/files/model.py` 加入层 20 的源投影入口，默认不触发。它复用现有 `_write_compressed_source`。2026-09-23 在 A3-21 以真实权重 `v41-flat-verify3`、TP8、BF16 KV、Engram host 路径跑了一次 14-token 请求。8 个 TP rank 均记录 `rows=14` 且无比较异常；原始日志为 [`evidence/ced_source_8ff6a4a/serve.log.gz`](../evidence/ced_source_8ff6a4a/serve.log.gz)，解压后的 SHA-256 为 `cce508a4a6bc814164b78ec325768b67400d004192ab6a227d1773141fef56e0`。该门只证明这一短请求的采样行相等，不能外推到分块或长上下文。
- 开发诊断开关 `V41_CED_SOURCE_COMPARE=1` 会在非图捕获的真实 forward 中，先调用源投影，再执行普通层 20，并对每块前 8 和后 8 个有效物理槽中的主 KV、Indexer K、scale 做逐张量精确比较。`V41_CED_SOURCE_COMPARE_CHUNKS=N` 设定每个 rank 最多比较的块数，默认 1；日志会列出块序号和 token 位置。不匹配立即报错。两个开关默认均不启用；长请求测试应设置足够大的 `N` 覆盖末块。
- **144K 分块数值门通过。** A3-21 的 `33a6024` 独立 P 实例以 `V41_CED_SOURCE_COMPARE_CHUNKS=20`、真实权重、TP8、BF16 KV、`BAT_TOKENS=8192` 跑 `bigprefill`。第一条请求实际 144,404 上下文 token；模型前向到位置 144,461，分 18 块（前 17 块各 8,168 token，末块 5,606 token）。8 个 rank 的每一块均精确匹配所采样的主 KV、Indexer K 与 scale，没有异常。第二条请求又覆盖了头两块，合计各 rank 20 条成功记录。证据：[`probe_144k.json`](../evidence/ced_chunks_33a6024/probe_144k.json)、[`serve.log.gz`](../evidence/ced_chunks_33a6024/serve.log.gz)；完整日志解压后 SHA-256 为 `2129a09e908e31639e00dbf408fecb5e4189b39d66f56afb4f2feaa68100e264`。探针请求将 `max_tokens` 设为 1，直接访问 P 角色服务；两条回答仅为 `Z`、`V`，不满足检索判据，因此这次**仅验源投影数值，不验回答质量**。测试容器已停止，0–7 卡无运行进程。
- 下一实验开关 `V41_CED_ROLE=prefill` 在模型主循环运行完层 19 后直接写层 20 的全局源，跳过层 20–39；当前要求 `SPEC=0`。P 端采样会被固定为内部传输标记 token 42，使 `MooncakeHybridConnector` 走 `FINISHED_LENGTH_CAPPED` 发布缓存。该 P 端点只能由 PD 代理内部调用，直连回答无语义；D 尾部重放和缓存组有效性协议尚未实现，不得用它搭配普通 D 对用户提供服务。此开关默认关闭，仍需在真实 A3 实例验证能完成长 prefill 与层 20 写入。

独立 P 计算实验的启动参数（端口和容器名须先确认空闲）：

```bash
MODEL="$HOME/models/out/v41-flat-verify3" \
DEVS="0 1 2 3 4 5 6 7" PORT=18770 KV_PORT=18870 \
NAME=dsv41-ced-p-cut RUN_ID=ced_p_cut \
SERVED_NAME=deepseek-v41-ced-prefill-only \
SPEC=0 STATIC_KERNEL=0 V41_CED_ROLE=prefill \
bash scripts/serve_a3_pd.sh prefill
```

服务就绪后，用 `python3 tools/ced_prefill_probe.py --base-url http://127.0.0.1:18770 --context-tokens 144000 --expect-masked-swa --out results/ced_p_144k.json` 检查内部 marker、12 组布局、上半层 SWA 屏蔽及 replay 标记。

**A3 真实权重验收（`302b586`）：** 按上面的独立 P 配置启动，`/health=200`，8 个 rank 都记录 `[CED-P] internal producer`，日志没有 decoder-layer 违规异常。短请求带 `do_remote_decode=true`，返回 `finish_reason=length`、标记 token ID 42，并带 `do_remote_prefill=true` 的 Mooncake 元数据。随后 [`ced_p_144k.json`](../evidence/ced_p_cut_302b586/ced_p_144k.json) 记录真实 prompt 143,963 token、17.07 秒、12 组 block ID、标记 token ID 42；完整日志为 [`serve.log.gz`](../evidence/ced_p_cut_302b586/serve.log.gz)，解压后 SHA-256 为 `02061375f211b754003e1c907588fbe74340414a967987a2b2806a5f1906c2dc`。该测试证明 P 端能运行长 prefill 并完成内部交接回执；其耗时不可直接与前一个开了数值探针和 DSpark 的 30.8 秒相减作为性能收益。容器已停止，0–7 卡无运行进程。

**P 侧组有效性协议已上 A3 验收。** `302b586` 的 12 组回执含 G7–G11，即上半层 SWA；P 跳层时这些组未写入，却会被普通连接器列入传输。`34fdf08` 的 [`experimental/ced/mooncake_hybrid_connector.py`](../experimental/ced/mooncake_hybrid_connector.py) 在 A3-21 实测将 G7–G11 block ID 置空，仍保留 G0 全局 KV、G1 环、G2–G6 低层 SWA；交接标记为 `ced_replay_tokens=128`、`ced_missing_swa_groups=[7,8,9,10,11]`、`ced_prefix_tokens=143962`。[`ced_p_mask_144k.json`](../evidence/ced_p_mask_34fdf08/ced_p_mask_144k.json) 记录真实 prompt 143,963 token、`finish_reason=length`、标记 token ID 42、组 block 数 `[1125,1,2,2,2,2,2,0,0,0,0,0]`；[`serve.log.gz`](../evidence/ced_p_mask_34fdf08/serve.log.gz) 解压后 SHA-256 为 `d0c6e8584ab56b3d7a3f3921333ff29f742418d3c584e6f1570e3072392a143e`。容器已停止，0–7 卡无运行进程。

`34fdf08` 的 D connector 对该 replay 标记直接拒绝，因此这里只验了 P 元数据屏蔽；**不能**作为 P→D 正确性或质量门。一次容器内单独导入连接器并调用拒绝分支，确实打印拒绝信息，但 Python 退出时发生 `corrupted size vs. prev_size`（退出码 134），不把它当作 D 端到端验收。后续工作树已加入尚未验证的 D 调度回退、缺失页清零和 replay 块内不回写全局 KV 原型；必须先在隔离实例验证，再做质量判断。下一步见 [`D_REPLAY_NOTES.md`](../experimental/ced/D_REPLAY_NOTES.md)。

## 2026-09-24：A3-21 CED 扫描与 direct full40 对照

- 归档 A3-21 旧 1M CED P/D/proxy 完整日志、inspect、实际进程环境、启动命令、代码/镜像/model metadata hashes 与全部当前 JSON。旧 P/D/PX 均先只读复核并在服务空闲后停止。芯片0–7仅由 P 复位，芯片8–15仅由 D 复位；A3-22 chip0/1没有使用。
- CED 1M配置下同实例扫描：8K needle 实际8,335 tokens，4/4错；144K needle offset0 实际142,426，4/4错；256K实际255,527，4/4错；520K实际517,036，4/4错；1M实际1,019,789，4/4错。旧144K bigprefill曾2/2正确，使用同样的提示词生成模式与A/B旋转偏移重放到当前CED实例后，实际144,404/144,131，2/2错。
- 极短输入也显示不同路径：Chat API prompt17 tokens的数学题回答错误；同一22-token短针 `校验码是 ZQ7K-3341。请只回复这个校验码。` 在当前CED服务也错，D正常角色日志记录 replay `0..20`。其中一次新D遗漏了 `V41_CED_ROLE=decode`，容器没有 scheduler/dsa/connector CED挂载，作为无效负控留档，不用于P/D结论。
- 修正角色变量后按阶段重启D+proxy、复测新D+旧P：短针仍错。随后重启P得到两侧新容器（两边都有正确CED role、挂载、worker和model ID），第一条22-token短针仍错；D日志 `positions=0..20`、P/D/proxy均HTTP200。这个组合短针失败，证明当前CED实现/配置下短路径仍有错误，但新P仅仅重启过一次，需保留精确运行顺序记录。
- direct full40对照使用同模型路径、image、MAX_LEN=1048576、BF16 KV、ENGRAM=1/device-index0、CPU_BIND=0、SPEC/PREFIX=0、STATIC_KERNEL=0、NPUGRAPH_EX=1、GRAPH=1/EAGER=0、DROPCACHE=0，`V41_CED_ROLE=''`、`KV_ARGS_EXTRA=''`，port18993。DRY_RUN显示无CED connector/dsa/Core patch mounts。direct full40的22-token短针精确命中；与CED完全同模式/旋转位置的144K bigprefill A/B也2/2 exact（实际144,404/144,131）。
- direct full40随后做offset0、target1,020,000的needle四针，P tokenizer报告实际1,019,789。A/B/C/D 4/4未精确命中、HTTP均200、U+FFFD均0；A的probe answer_repr=`J4y9K2`，B/C/D的`answer_repr`为空字符串。wall 291.1/289.6/287.7/286.8秒，total 1,160秒。探针没保存这四次请求的原始API body，所以空answer_repr只表示probe抽取结果，不推断服务返回零token。随后同一direct full40 PID在22-token短针上再次精确命中（22 prompt tokens、`ZQ7K-3341`、0.336秒），说明1M失败后短请求没有整体污染。
- direct full40 1M结果说明：现有 W4A8 model/config 在1M needle上也会失败，不能将长上下文失败完全归因于CED；短输入和144K准确，但它不是论文近似全链路，也不能替代历史BF16 KV + AscendStore + 1M全40层对照（后者参数不同）。性能数值不跨连接器、缓存/采样配置比较。
- 【代码语义、因果分开】D SMLA仍将完整 `seq_lens` 传给 `seqused_ori_kv`，`ori_mask_mode=4`、`ori_win_left=127`；replay scheduler虽然有 `replay_start`，attention窗口没有按 `replay_start`硬裁。首个replay query可能看到最多127个未传输的上层SWA逻辑位置，connector按G7–G11 local physical IDs清零并不能证明这些逻辑页都有效。该语义差异已核代码，但22-token从位置0起仍错，因此不能单独解释direct full40 1M失败或所有短错，根因待位置20的逐值快照。
- 参数对照纠正：旧144K通过服务`MAX_LEN=147456`，当前1M实例`MAX_LEN=1048576`；其余关键CLI（BAT=8192、MAX_SEQS=4、GPU_UTIL=.92、BF16、Engram、SPEC/PREFIX=0、STATIC_KERNEL=0、模型路径）相同。Available KV memory均约15.16GiB，Mooncake physical blocks约30,083/30,081；vLLM `GPU KV cache size tokens`按`max_concurrency(max_model_len)*max_model_len`换算，不能写成物理缓存页翻倍。
- direct full40及各CED阶段的JSON、日志、容器inspect、命令、process env、image/model metadata hashes已归档至[`ced_8x8_length_scan_20260924/`](../evidence/ced_8x8_length_scan_20260924/)。full40 direct容器port18993目前保留运行；CED doublefresh P已停，CED D/proxy仍Up且idle。下一阶段采用snapshot包 `fix/ced-replay-and-launch@43b3dc9`：本机tar SHA `c18b3a290126faed145d85ba15bec2bb99eac6c086414b3e4eca37d7a2b7556e`，COS key `share/xfer/ced_pkg_43b3dc9_20260924.tar.gz`。远端先下载验SHA/selfcheck；先跑带position20 cache/H20快照的无CED full40 baseline，短针不精确则停止；基线和CED三臂完成后由compare工具对照。snapshot包尚未部署到A3。

## 8+8 D replay 首轮故障（`8b59d03`）

- A3-21 上 P=0–7、端口 18790，D=8–15、端口 18791，代理 18792；真实权重、BF16、`SPEC=0`。两侧均 `/health=200`，D 的 Core replay 补丁通过启动脚本现场应用，D 连接器识别到上层 SWA G7–G11。P 服务在本轮后仍运行，D 和代理已停止。
- 代理发最短数学请求后返回 HTTP 500。D 还在 KV 加载前的缺失页清零步骤就失败：`tensor.index_fill_(0, indices, 0)` 在当前 Ascend 共享缓存视图上申请 **7.36 GiB** 临时显存，而当时每卡只剩约 6.2–6.5 GiB。没有执行到 D replay 前向，因此不能判断调度或模型质量。[原始 D 日志](../evidence/ced_d_oom_8b59d03/serve.log.gz) 解压 SHA-256 为 `6d116cf5049ecd016aed7aae1f782a885134ac087cb11c456af7a3a6b08b5ec6`；[代理响应](../evidence/ced_d_oom_8b59d03/proxy_response.json)。
- 现已把清零改为对每个物理 block 用 `tensor.narrow(0, block_id, 1).zero_()` 原位写入，避免 `index_fill_` 的整视图临时申请。该修复**尚未真机复测**；下一步只重启 D 和代理，复用仍在运行的 P。
- `compile()` 语法检查和 `git diff --check` 通过。没有声称 CED 运行时或性能已经实现。

## 2026-09-24：同一 A3-21 CED 服务长度扫描

- 恢复后先只读检查 A3-21：P `dsv41-ced-p-1m-20260924`、D `dsv41-ced-d-1m-20260924`、proxy `dsv41-ced-proxy-1m-20260924` 均仍运行；P/D 的 8 个 TP worker 和 EngineCore 均在。P `/health`、D `/health`、proxy `/healthcheck` 都返回 HTTP 200；宿主 `192.168.45.21` 上 P/D KV 端口 19090/19091 可 TCP 连接。探针结束后两侧日志显示 `Running: 0, Waiting: 0`。未重启容器。
- 从远端取回 [`needle_520k.json`](../evidence/ced_8x8_length_scan_20260924/needle_520k.json)，核验 SHA-256 `b101a03e4665009b7d0857481762310cbf40b858b72c890069585496851acf1d`。目标 520,000、结果记录的实际上下文 517,036 token；A/B/C/D 均 HTTP 200，但 4/4 不含答案。U+FFFD 为 4/0/0/0，回复包含混杂文本和乱码。此 JSON 未记录 `offset`。
- 在同一组现有服务上跑 256K 四针，参数为 `offset=0`、`repeats=1`、`max_tokens=64`、`temperature=0`、非流式、`min_ctx_ratio=0.9`。tokenizer 显式连 P `18990`，推理连 proxy `18992`。实际上下文 255,527/256,000，A/B/C/D 4/4 错、HTTP 200、均未含答案，U+FFFD 为 8/10/1/3，单条约 19.5 秒。D 的 128-token replay 位置在 255,454–255,456 至 255,581–255,583。原始 [`needle_256k.json`](../evidence/ced_8x8_length_scan_20260924/needle_256k.json) SHA-256 `9287689a2de784d14ef335998e97523ad340a5d0a97b282c85f53e5adcd68b0f`；runner stdout、proxy 日志和 P/D serve 完整日志也在该目录。
- 旧 520K 和 1M JSON 都没有保存 offset，故 256K（明确 offset=0）与它们不是可证明的严格同 prompt 对照。远端 corpus 文件 SHA-256 `a7fc413bd6e3926482faddf2af9bfb4426e55d8e421481d60100c14160785578`；本轮使用 P `18990` 的 `/tokenize` 进行长度校准。
- 结果说明当前这组 1M CED 服务在约 255K、517K、1,020K 三个长度均出现检索失效。之前另一轮 144K 的通过不是同一服务配置的严格对照；下一步按主 Agent 指示在**这组未重启服务**上用 offset=0 重测 144K，保持 P tokenizer、SPEC=0、PREFIX=0，再决定是否测 192K。当前不切全 40 层基线。
- **强对照但非单变量：** 既有全 40 层双 TP8、BF16、AscendStore 的 [`pdstore_bf16_align_needle1m.json`](/home/chiro/projects/dsv41/pd_single_a3/evidence_handover/pdstore_bf16_align_needle1m.json) 在实际 1,019,789 token 下四针 4/4 精确正确且 U+FFFD=0；与这次 CED 的连接器及 DSpark 设置不同。
- **待验证假说：** D 对尾部 128 token 从 token IDs 重算 40 层；若其 replay 起点前的低层 SWA 状态不完整，内部 H20 可能与完整模型不同。P/D 目前低层 SWA 传输行为使该路径值得优先检查，但长度扫描没有直接比较 H20、SWA 缓存页或 logits，不能将它写成根因。
