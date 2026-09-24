# CED-PD 交接：2026-09-24

## 当日最新进展：D 图模式短针故障已复现

- A3-21 的同一 P 进程、同一 `f28cd71` 包、同一 22-token 请求（SHA-256
  `f670186097daa964d1eca4c436320e8dcee6dc93cda0ea41b1306224991aa881`）
  下，只切换 D 的 `GRAPH/EAGER`：`0/1` 两次均精确回答
  `ZQ7K-3341`；`1/0` 两次均为 HTTP 200 但生成满 32 token，未答校验码，
  其中一次有 7 个 U+FFFD。两臂均确认 replay `0..20`；图臂实际完成
  `FULL_DECODE_ONLY` 捕获，eager 臂实际含 `--enforce-eager`。
  原始请求、响应、启动命令与日志在
  [`evidence/ced_graph_ab_20260924/`](../evidence/ced_graph_ab_20260924/)。
- 这说明当前 **CED D 图执行路径会触发短针质量故障**，尚未定位图内具体算子。
  A3-21 保留同一 P、恢复 D eager 后，22-token 短针与历史 144K bigprefill
  A/B **3/3 精确通过**。A/B 的 body token 数为 144,404/144,131，实际
  Chat API prompt usage 为 144,462/144,188；D 均执行 128-token replay，
  原始证据在
  [`eager_function/`](../evidence/ced_graph_ab_20260924/eager_function/)。
  同一 eager 实例随后对 body 均为 1,019,789 token 的 1M 标准 needle
  A/B/C/D 顺序测试 **4/4 精确通过**，各次 HTTP 200、U+FFFD=0，D 均执行
  128-token replay；完整原始请求/响应及日志在同一证据目录。
  这是 eager 正确性诊断，不代表图模式或最终性能交付。A3-22 的 1+1 线由
  子代理并行管理，仅使用 chip6/7，
  chip0/1 继续预留。
- 实验分支已快进合入 `fix/ced-replay-and-launch@f28cd71`。CED D 专用启动
  包装器的本机未提交安全门要求显式选择诊断臂：
  `CED_DIAGNOSTIC_EAGER=1` 为 eager 正确性基线，
  `CED_EXPERIMENTAL_GRAPH=1` 为图模式故障定位；当前无默认交付配置。
- **首 token 定位：** 同一 P、同一原始 22-token 请求分别限制
  `max_tokens=1/2`：D eager 输出 `Z`/`ZQ`，D 图输出“正确答案”/
  “正确答案只有一个”；两臂均 HTTP 200、请求 SHA 逐项一致。
  [`graph_minimal/`](../evidence/ced_graph_ab_20260924/graph_minimal/) 保存
  token logprobs、实际 runner 代码和源码指纹。当前 runner 在建立 attention
  元数据前按单 token 形状选 FULL 图，而末 prompt token 的元数据是
  `num_prefills=1`。这构成“单 token prefill 误入 decode 图”的待验假说。
  隔离诊断包 `fix/ced-graph-prompt-tail@777bc73` 只让该步 eager、保留生成
  decode 图，COS key `share/xfer/ced_pkg_777bc73_20260924.tar.gz`，SHA-256
  `f096c822b4a5ec75fed019f55856c58ce7d101083715261590f07cbff72a2897`；
  本地自检通过；A3-21 真实权重下同 SHA 的 `max_tokens=1/2/32`
  分别返回 `Z`/`ZQ`/`ZQ7K-3341`，8 个 D worker 均记录尾步强制 eager，
  后续生成实际记录 `Replaying aclgraph`。原始证据在
  [`graph_prompt_tail/`](../evidence/ced_graph_ab_20260924/graph_prompt_tail/)；
  同一实例的历史同 SHA 144K A/B **2/2 精确通过**；1M 标准四针
  A/B/C 精确，D 失败（HTTP 200、无 U+FFFD，但返回 `Tech-D9q7Wm`，
  预期 `RB9N-6014`）。四针 body 均为 1,019,789 token，D replay 均为
  128 token，8 worker 均打印尾步 forced-eager。当前 A3-21 的 P/D/proxy
  实例保留运行，随后对失败 D 的原 prompt 仅改
  `max_tokens=1`、开启 logprobs 诊断首 token；**不要**把 3/4 写成 1M
  正确性通过。流式、缓存命中和性能仍待验收。
- **最新重复性结果：** 失败 D 的同一原始请求 SHA
  `f25d0b8b1a0ea3e4ff37131dc90182a2940ef075f7fc1ab8fcbb7c99cd9f9d8e`
  在同一 graph+tail 实例共发四次，顺序为错误 `Tech-D9q7Wm`、正确
  `RB9N-6014`、正确 `RB9N-6014`、错误 `Uhq9-3DqT`，即 **2/4 正确**。
  第一次错误和其余重复之间插入过一次 `max_tokens=1` 诊断，它返回
  正确首 token `RB`（logprob 约 −0.00022）；后两次为连续重复。
  所有完整请求都走相同128-token replay/尾步 eager，日志无错误。
  A3-21 保留同一 P，只把 D 切到全 eager **诊断对照**后，对同 SHA D
  连发四次 **4/4 精确正确**。随后 D 恢复 FULL 图和 prompt-tail 修复，
  仅关闭 `DSA_OVERLAP`；短针及144K A正确，同 SHA 1M D四次中前三次
  正确，第四次 HTTP 200 但 `content=null`、completion_tokens=1，
  即 **3/4**。关闭 DSA 辅助 stream 不足以稳定修复1M。
- A3-22 1+1 tiny 的 eager/graph profiler 已归档到
  [`ced_tiny_pos21_20260924/`](../evidence/ced_tiny_pos21_20260924/)，
  小型证据已作为 `759d444` 合入本分支，原始 trace 在私有 COS。
  tiny 两臂输出都正确；图模式 SMLA 分布到不同 stream，trace 有 event
  record/wait，但缺少具体 event handle，不能证明特定跨流依赖缺失。
  A3-22 chip7 已释放，chip0/1 未使用。
- 下文“当前状态”和“下一阶段”是这次对照以前的历史快照；以本节和
  [`工作日志`](A3-CED-PD-WORKLOG.md)的最新记录为准。

## 目标与分支

- 活跃目标仍是 A3 上的真实权重 8+8 CED-PD：P 长 prompt 主段跑前 20 层、D
  128-token SWA 有界重放并完整 40 层 decode；BF16 KV 可保留。还需 1M、
  缓存命中和严格性能对照，**目标未完成**。
- 实验分支 `feat/ced-pd-a3`，工作树
  `/home/chiro/projects/dsv41/ced-pd-release`。基础全 40 层双 TP8 PD 在
  `docs/A3-PD-BF16.md`。CED 工作记录在 `docs/A3-CED-PD-WORKLOG.md`；
  D 实现边界在 `experimental/ced/D_REPLAY_NOTES.md`。

## 已实测

- A3-21 真实权重 TP8 层 20 来源投影：8 rank 的 144K 分块前/后采样缓存行
  与普通层 20 精确相等。P 跳层能够完成 144K 内部交接；P 置空未写的上层
  SWA G7–G11，并返回 replay 元数据。
- A3-21 真实权重 8+8 CED-PD 的旧实例曾通过 144K `bigprefill` A/B 2/2、
  流式/工具/并发 5/5、多轮增长 3/3；旧实例 P 来自 `8b59d03`、D 来自
  `3a45922`，`MAX_LEN=147456`。此历史通过不是当前 1M 服务的同状态证明。
- **1M 未通过。** 子代理用实际 1,019,789-token 标准四针请求实测 A/B/C/D
  **4/4 失败**，HTTP 都是 200，D 均记录 128-token replay；A/B/D 的
  U+FFFD 数分别为 3/2/4，输出有重复与混杂文本。没有 OOM 或 Python
  Traceback。原始 [1M 响应](../evidence/ced_8x8_1m_20260924/needle_1m.json)
  SHA-256 `210daf08b4d774a7ead456d4ec76a0bcb1290391c37149aa5fba37761046b85e`，
  [P 日志](../evidence/ced_8x8_1m_20260924/p_serve.log.gz) 解压 SHA-256
  `129be349fdad1bcdeb43b25dc42b0d7c10317ec91f87bb54f9e07775a4a83184`，
  [D 日志](../evidence/ced_8x8_1m_20260924/d_serve.log.gz) 解压 SHA-256
  `c04d5d00bdcba1bdae2c2371e3aa0630ced18b13a645ca513d672900ee6c4265`。
  不能把 144K 通过外推到 1M；恢复后先定位这个数值故障。
- **长度扫描新增 520K 与 256K 结果。** 同一 A3-21 CED 服务实例下，520K
  目标实际 517,036 token，四针 4/4 错；256K 目标实际 255,527 token，
  四针也 4/4 错，U+FFFD 分别为 8/10/1/3。256K 使用 `offset=0`；旧
  520K/1M JSON 没有保存 offset，不能声称同 prompt 严格对照。证据和
  完整 P/D 日志在 [`ced_8x8_length_scan_20260924/`](../evidence/ced_8x8_length_scan_20260924/)。
- 当前同一 1M 服务实例追加的 144K `needle`（实际 142,426）、8K `needle`
  （实际 8,335）也都 4/4 错；再用旧通过的 144K `bigprefill` 提示生成模式
  和 A/B 语料偏移复测，实际 144,404/144,131，2/2 错。三个≤22-token
  数学/短针请求（Chat API 17/17/22 tokens）也都返回错误答案；D replay
  范围分别为 `0..15`、`0..15`、`0..20`。所以当前实例的质量异常不局限于
  超过 128-token 窗口的长 prompt。所有 HTTP 都是 200。具体 JSON、响应、
  完整日志、容器检查和环境快照均在上述证据目录。
- **全 40 层 1M 历史对照通过。** [`baseline_pdstore_bf16_needle1m.json`](../evidence/ced_8x8_length_scan_20260924/baseline_pdstore_bf16_needle1m.json)
  在实际 1,019,789 token 下 A/B/C/D 4/4 精确正确且 U+FFFD=0；它使用
  AscendStore 和不同 DSpark 配置，属于强对照但并非严格单变量实验。
- **同镜像 direct full40 fresh 对照：** 不带 CED role/KV connector，TP8、BF16 KV、
  MAX_LEN=1M、SPEC/PREFIX=0。在模型ID `deepseek-v41-full40-1m` 上，22-token
  短针精确正确，原 144K bigprefill A/B（实际 144,404/144,131）2/2 精确正确；
  但同一 direct 进程 1,019,789-token needle A/B/C/D 4/4 失败，均 HTTP 200、
  U+FFFD=0。A probe抽取到 `J4y9K2`，B/C/D 的 `answer_repr` 为空；不能据此
  断言 API 原始 body 为零 token。1M scan完成后，22-token短针再次精确通过，
  说明短请求状态没有被长请求带坏。原 JSON/完整 direct serve log在同一
  length-scan证据目录。此 direct配置和历史AscendStore full40配置不同。
- 8+8 子代理 `/root/a3_8x8_validation` 已按用户要求停止新增工作并完成
  交接。Codex 重启后再派新的子代理接手；复杂实现决策由主 Agent 决定。
- A3-22 单卡 tiny 框架是既有 `a2/agents/L1_dummy` 的 `model-tiny`，
  保留 40 层和 12 组 BF16 缓存；dummy 权重不代表真权重质量。
  tiny 模型经 COS 复制到
  `a3-22:~/projects/dsv41-ced-singlechip/model-tiny`，P 用 Phy-ID 6、
  D/基线顺序用 Phy-ID 7。用户预留 A3-22 chip0 和 chip1，**不得使用**。
- tiny 1+1 的长度 2/127/128/129/130/256/512/4096 各两遍，16/16
  HTTP 200、重复结果一致；与全 40 层基线对照时 16/16 选中 token 相同、
  top-20 集合 20/20 一致，共同候选最大 logprob 差 `9.54e-7`。
  证据在 `evidence/ced_tiny_pd_b6ca283/`。
- tiny 长度 1 会触发 D 调度的 `prefix boundary mismatch`，使 D 引擎退出。
  原因已定位：P 对单 token 不截尾，D 校验只按 `N−1` 写。尚未修复。

## 当前未验证诊断与下一步

- A3-21 当前状态：直接 full40 实例 `dsv41-full40-1m-doublefresh-20260924`
  在芯片0–7、HTTP18993运行中；它的模型ID为 `deepseek-v41-full40-1m`，
  直接进程已完成 short/144K/1M/short 检查。CED 的 doublefresh D
  `dsv41-ced-d-1m-doublefresh-20260924` 与 proxy
  `dsv41-ced-proxy-1m-doublefresh-20260924` 在芯片8–15仍运行且空闲；
  配对的 CED P 容器已停止。没有使用 A3-22。
- **启动差异已核对：** 旧 `MAX_LEN=147456`，当前 `MAX_LEN=1048576`；
  BAT=8192、MAX_SEQS=4、GPU_UTIL=.92、BF16、SPEC/PREFIX、Engram、
  STATIC_KERNEL=0、模型路径相同。旧/新 `serve_a3_pd.sh`、`serve_v2.sh`、
  `model.py`、`dsa_v41.py`、Core replay patch 哈希一致。旧 P 和当前 P 的
  connector 文件哈希不同，唯一 diff 是 D 缺失页清零从 `index_fill_` 改成
  按 block `narrow().zero_()`；旧 D 已使用新版本。Available KV memory 都约
  15.16 GiB，num_blocks 约 30,083/30,081。`GPU KV cache size tokens` 是按
  `max_model_len` 折算的容量指标，不能据此断言缓存页布局变化。
- **代码确认的语义缺口，因果未证：** D SMLA attention 以完整 `seq_lens`
  作为 `seqused_ori_kv`，使用 `ori_mask_mode=4`、`ori_win_left=127`；scheduler
  限定 replay 计算区间，却没有把 attention 可见窗口裁到 `replay_start`。
  replay 起始 query 可能向前看至多 127 个未传输的上层 SWA 逻辑位置；D
  connector 只清本请求 G7–G11 的 local physical IDs，不能证明这些逻辑位置
  都映射到已初始化页。≤22-token replay 从位置 0 开始仍答错，因此此语义缺口
  不能单独解释当前所有错误。需与 pos254 数值门结合定位。
- **当前根因未定：** role缺失的 D 负控未执行 CED replay；正确角色的 fresh D
  +旧 P 短针失败；fresh P+fresh D 的 22-token 短针也失败。随后无 CED role、
  无 KV connector 的 direct full40 短针及144K bigprefill均通过，但 direct
  full40 1M四针4/4未命中，且1M后短针仍通过。证据表明短上下文的错误依赖
  CED路径，1M长上下文失败则在direct full40也可复现；需用缓存数值门进一步区分。
- **下一阶段做三臂数值快照：** 主 Agent已上传独立包
  `fix/ced-replay-and-launch@43b3dc9`，COS归档
  `share/xfer/ced_pkg_43b3dc9_20260924.tar.gz`，SHA-256
  `c18b3a290126faed145d85ba15bec2bb99eac6c086414b3e4eca37d7a2b7556e`；
  本机 selfcheck已过，A3-21端仍需校验 tar SHA与selfcheck。先用新包在芯片0–7
  跑无 CED role/无 KV connector 的 full40 baseline，`MAX_LEN=1M`，在位置20
  导出全40层 SMLA cache和H20；只发归档22-token短针，若未精确命中则停。
  baseline完成后先报告，再切到新包CED P/D（芯片0–7/8–15）各开位置20快照，
  通过角色/补丁/哈希/worker硬门后发同一短针。最后用新包提供的compare脚本
  对比baseline、P和D。复杂数值异常先报告，不做现场修法。
- 原 `V41_CED_SNAPSHOT_POS` / `V41_CED_SNAPSHOT_DIR` 诊断只在本工作树做过
  语法检查，未上 A3。下一阶段应以43b3dc9包的 real-weight position-20
  snapshot与H20工具为准，不复用旧tiny-only步骤。
- A3-22 上次已确认 P 容器 `dsv41-ced-tiny-p-b6ca283`、HTTP 18960、
  chip6 健康；D `dsv41-ced-tiny-d-b6ca283` 和代理已停止。基线容器
  `dsv41-ced-tiny-baseline-ccf5762`、HTTP 18963、chip7 在对拍时健康。
  最后一次 SSH 状态查询未返回（已中断该只读命令）；**重启后先只读复核**。
- 发布包 checksum 现可用 `python3 tools/refresh_checksums.py` 统一刷新
  `patches/MD5SUMS`、系列清单、`PATCHES.md` 与 `MANIFEST.sha256`；
  新文件须先 `git add`，之后跑 `bash tools/selfcheck_pkg.sh`。

## 环境与资源

- A3-21 SSH `a3-21` → `192.168.45.21`；A3-22 SSH `a3-22` →
  `192.168.45.22`。节点时间比本机约慢数分钟，以远端 `date`、日志字节
  增长、真实进程为准。
- 大文件传输用 `upstream-v41/pr/cos-xfer.sh` 和 COS 私有
  `share/xfer/`；临时文件放 `~/tmp/<日期>/<任务>/`。
- `upstream-v41/` 是冻结参考区，不写。单卡 P/D 测试脚本是
  `scripts/serve_a3_ced_single.sh`，全 40 层基线使用其 `baseline` 角色。
