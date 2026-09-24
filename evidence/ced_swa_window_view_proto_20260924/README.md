# CED replay-start SWA 可见范围：离线设计检查

日期：2026-09-24。此目录只包含 CPU 页映射原型和设计分析；没有启动 NPU、
没有改生产代码、没有改 8+8 主实验树。

## 发现

CED D 的 `_native_attention` 仍把完整 `metadata.swa.seq_lens`、完整
`metadata.swa.block_table` 传给 `_C_ascend.npu_sparse_flash_mla`，而 SMLA
metadata 由 LongKV metadata builder 按全局 `seq_lens/max_seq_len` 生成。Core
虽然把 D 的 cursor 退回 `replay_start`，attention metadata 没有跟着裁到该
起点。

同时，当前 Mooncake CED P connector 把上层 SWA G7–G11 的 remote block IDs
置空，因为 P 没计算这些层。D 从 replay 起点开始生成这些层的 SWA KV；因此
它们在起点以前没有有效的 D KV 页。给 SMLA 一个局部页表能防止读取不存在的
页，但不能凭空补出起点以前的 upper-SWA 状态。

现有合成 `_C_ascend` A/B 保持 CSA KV、稀疏 indices、residual 和 block table
不变，只改 ori KV 视图：A 为255行完整窗口，B 为128行 replay-only。A/B 都
通过 metadata checker 且输出 finite，但 B 的 row0 dim0 从 A 的 `0.171875`
降为 `0.023071`，全输出最大差 `0.148804`；row127 恰好相同。该测试证明
kernel 可运行两种视图，并明确说明 replay-only 截断会改变窗口语义；不能把 B
当成 full-SWA 等价实现。

参考：`../ced_tiny_cache_snapshot_20260924/` 在长度256/pos127看到 replay首行
开始的 SWA 偏差；快照不能证明物理页缺失。`../ced_smla_ori_view_ab_20260924/`
保存了合成算子证据。生产代码当前在 `experimental/ced/dsa_v41.py`；连接器
页裁剪逻辑在 `experimental/ced/mooncake_hybrid_connector.py`。

## Token 区间 → 逻辑页 → P connector 页

令原 prompt 长度为 `N`，P 交接 `N-1` 个 token，D replay 长度 `R=128`，SWA
窗口 `W=128`，KV block `B=128`：

```text
E = N - 1                         # prefix/replay 的 exclusive end
S = max(0, E - R)                 # replay_start
full_visible_start = max(0, S - (W - 1))
full_view_page_start = floor(full_visible_start / B)
full_view_page_end = ceil(E / B)  # exclusive
full_view_seq_len = E - full_view_page_start * B
```

SMLA 的 Q 仍是原来的 Q；完整 lower-SWA 视图把 `ori_block_table` 切到
`[full_view_page_start, full_view_page_end)`，`seqused_ori_kv` 改为
`full_view_seq_len`，`cu_seqlens_q` 与 CSA 参数不变。起点向下对齐产生的多余
页由 `ori_win_left=127` mask 掉。对于 upper groups，P 没传起点前的 K/V，必须
明确选择 replay-only 近似，或先补齐状态；同一张 full-view 页表不能伪装成数据
存在。

| Prompt N | P loaded end E | Replay `[S,E)` | 完整可见 token 区间 | 对齐后的逻辑页 | 当前 P cleanup / connector | overlap 保留后 |
|---:|---:|---:|---:|---|---|---|
| 1,048,576（对齐） | 1,048,575 | `[1,048,447, 1,048,575)` | `[1,048,320, 1,048,575)` | 8190, 8191 | manager 只留8191；connector tail2含空8190，缺8190 | manager留8190–8191；connector tail3，所需页均在 |
| 1,048,575（未对齐） | 1,048,574 | `[1,048,446, 1,048,574)` | `[1,048,319, 1,048,574)` | 8189, 8190, 8191 | manager留8190–8191；connector tail2，缺8189 | manager留8189–8191；connector tail3，所需页均在 |

upper replay-only scratch 的物理来源则是另一张映射表：对齐例的 replay token
先从 D `common.block_table[request,8190]` offset127 读1行，再从 page8191
offset0–126读127行；未对齐例从 page8190 offset126–127读2行，再从 page8191
offset0–125读126行。两例都压到该请求的 scratch page0 offset0–127，attention
表只指向 D scratch page。实际物理 block ID 运行时从 D 的 page table 查询；本表
只显示 token/page offset 关系。

connector 里 `num_swa_blocks=cdiv(128,128)+1=2`。若 lower SWA 需要在 replay
区间上保留完整左窗，至少要让 P/D SWA manager 保留 `W-1+R=255` 个 token，
即 `extra_retained_tokens=128`；connector 传输页数对应
`cdiv(W-1+R,B)+1=3`。这是 lower groups G2–G6 的页保留/交接变化。页 ID 在
运行时来自 `common.block_table[request, logical_page]`；此脚本中的页序号是
逻辑页号，不是 A3 上的物理 block ID。

连接器传页时的顺序也要保留：P 的 block list 含前缀 null block，D 的
`get_unhashed_block_ids_all_groups()` 会过滤 null。Mooncake worker 在 remote
list 比 D local list 长时把 remote list左裁到同样长度，然后按顺序 zip。
例如对齐例当前是 remote `[null(8190), P8191]`、D local `[D8191]`，最后只会
把 P8191 写到 D8191，缺页8190；开启 overlap 后 remote tail3 为
`[null(8189), P8190, P8191]`、D local 为 `[D8190,D8191]`，左裁后两页正确
对齐。未对齐例当前 remote/local 都只有8190/8191，缺8189； overlap 后三页
8189/8190/8191 成对传输。P/D 物理 ID 不同，映射靠各自 request block table，
不能把 P 的物理 ID 直接塞进 D 的 SMLA table。

静态脚本：[`prototype.py`](prototype.py)。它通过两例断言：当前 transfer
page count 缺少完整 lower-SWA 窗口页，添加 replay overlap 并传3页后所需逻辑页
齐全。复现：`python prototype.py`；结果在 [`static_results.json`](static_results.json)。
脚本还扫过所有128种 prompt block remainder：当前两页传输128/128都缺至少一个
full-window逻辑页；overlap+三页传输为0/128缺页，完整视图实际跨2–3页。

## 两条集成路线

### A. 保持论文式128-token bounded replay，显式裁到 `replay_start`

- lower SWA G2–G6：保持其原始缓存来源，P `extra_retained_tokens=128`，
  connector 至少传3页；SMLA 用完整左窗 local view。
- upper SWA G7–G11：P 的 K/V 不存在。D 将本次及此前已完成的 replay rows
  compact 到预分配 scratch 页；`seqused_ori_kv` 仅是从 `replay_start` 到当前
  `seq_len` 的长度。对首个 query，左窗自然从 replay 起点截断。这与合成 B 一致，
  是明确的有界近似，不等于 full40 的完整 SWA 窗口。
- CSA 来源完全保持 P 的 global long-KV / Indexer K、global sparse indices、
  residual 和 compressed block table；只变 ori SWA view。
- D 的模型计算仍是当前128-token、40层 replay，无额外模型 FLOPs。P 对
  G2–G6 每个 cache group 最多多保留/传1页每请求；upper scratch 容量最多为每
  活跃请求 `ceil(128/128)=1` 页/upper SWA cache plane。
- 上层 scratch 的 source KV 必须是 D 已写入的实际 SWA cache 行；在
  `preprocess` 完成其 scatter 且 multistream join 后复制到 scratch，再调用 SMLA。
- G2–G6完整窗和G7–G11 replay-only窗的 `seqused_ori_kv/max_seqlen_ori_kv`
  不同。Model Runner 为每个 attention group 调builder一次，并把同一个
  `attn_metadata_i`赋给该group全部layer；LongKV group中的layer 2/8/14/20共享一份
  `DeepseekV41Metadata.smla_metadata`，DSA消费者通过 source prefix取它。因此
  hybrid bounded 模式必须有两个稳定metadata variants（full-window / replay-local），
  再按SWA组类别选择。最简单的负控是10组统一 replay-only、只用一个metadata
  variant；它也裁掉 lower 左窗，只作诊断臂，不作为默认生产方案。

### 成本与语义对照

| 路线 | P 侧增量 | D 侧增量 | SWA 语义 |
|---|---|---|---|
| A `bounded128` | G2–G6 每组多保留/传1页每请求；P模型FLOPs不变 | 40层仍回放128 tokens；G7–G11 每组约1个scratch页/活跃请求 | lower保留完整左窗；upper明确从 replay_start 截窗，属于有界近似 |
| B `H20 burn-in` | P不多跑模型层；需暂存并传2668个H20 activation slices/请求（含mHC streams） | D upper20 跑2668 tokens，约20.8×当前 upper replay；多50,800个upper layer-token evals | 有机会得到完整末尾upper状态，依赖H20定义、因果范围和新通路验证 |
| B `full40 token burn-in` | 无H20激活传输 | D全40层回放5208 tokens，约40.7×当前D replay | 无跨机新激活；约束是给定窗口模型的5080-token burn-in界成立 |

`patches/files/model.py` 中 H20 `hidden_states` 为
`[tokens,hc_mult,hidden_size]`，参考 config `hc_mult=4`，还需传
`pre_mix[tokens,hc_mult]`（FP32）。全40路径为对照方案，不是建议先上生产。

### B. 恢复精确末尾 upper-SWA state

local view 之外还要为 replay 起点前的 upper layers 准备真实状态。可验证的
保守依赖界是每层最多把缺失状态向后传播 `W-1=127` 个 token；upper 有20层，
用 P 的 H20 boundary activations 时，D 至少需要预热
`20×127=2,540` 个 token，再处理128-token replay，即约2,668个 H20 vectors。
这需要新的 H20 capture/传输载荷和 D upper-only prefill path；P 本来计算该边界，
额外 P FLOPs为0，但需暂存/传输
`2,668 × 4 × hidden_size × 2 + 2,668 × 4 × 4` bytes。按
hidden_size=7,168、BF16 计算，一份完整 activation copy 约 **145.95 MiB/请求**，
其中 hidden_states 约145.91 MiB，pre_mix 约41.7 KiB。TP 分片或副本映射尚未
确定，不能直接除以TP度数。D upper20的 replay 工作量约为当前128-token
upper工作量的20.8倍，额外约50,800个 upper-layer token evals。

不加 H20 传输的简单参考是让 D 从 token IDs 做全40层 burn-in：保守预热
`40×127=5,080` token，再完成128-token replay，总计5,208 tokens；约为当前
40层128-token replay计算的40.7倍，但 P 无需新增 activation transfer。它仍远短于
从头重算1M prompt；要接受此设计前需实测全窗口依赖界、CSA/indexer与完整模型
数值结果。

路线B保留完整窗口，代价是新的跨机状态传输或约5K-token D burn-in；不能通过
仅换 `ori_block_table` 得到。

## Metadata、图模式与回退

1. 给 `DeepseekV41Metadata` 添加独立的 SMLA-ori view 字段，例如
   `smla_ori_block_table/seq_lens/max_seq_len`。不要覆盖 SWA 全局
   `block_table/seq_lens/positions/slot_mapping`：这些仍用于 KV 写入、RoPE、
   compressor、indexer 和缓存检查点。
2. LongKV group builder的 `_smla_metadata` 是每个builder一份常驻 `[1024] int32`
   buffer；Model Runner将该builder结果分配给组内所有 layer name
   (`model_runner_v1.py:3488-3490`)。Hybrid bounded需要在LongKV builder固定分配
   `_smla_metadata_full_window` 和 `_smla_metadata_replay_local` 两个 buffer，并
   为两者各发一个 `DeviceMetadataTask`；参数除 `seqused_ori_kv/max_seqlen_ori_kv`
   外保持相同。`common_v41_batch_metadata` 在所有KV groups间共享
   (`model_runner_v1.py:3493-3501`)，可放两种 per-request lens，避免builder顺序依赖。
   `metadata.swa` 另带 per-group ori block table/view class；`_native_attention`
   按lower/upper SWA组选择对应metadata指针。CSA的table/length/indices/residual全不动。
3. 预分配每层/组的 scratch KV、block table 和两种 metadata buffer；页面映射/拷贝用
   固定 shape/pointer、padding block，不能在 graph replay 时 `torch.empty`。
   复制放在 SWA `preprocess` 写入和 multistream join 之后。现有
   `FULL_DECODE_ONLY` 把多 token replay 留在 eager；一 token prompt tail不属于
   replay-only scratch 分支。未来若 capture prefill，则仍要用稳定 scratch 指针和
   可重放 copy/gather。
4. 用一个两侧协商的开关，例如
   `V41_CED_SWA_REPLAY_MODE=legacy|bounded128|burnin`，由 P 的 connector 在
   `kv_transfer_params` 回传模式和 overlap token 数，D 校验一致。默认 `legacy`
   回退现状；`bounded128` 才打开 extra-retained pages + local scratch，
   `burnin` 才打开 Route B 状态重建。缺少页、参数不匹配、graph capture资源未就绪
   时必须明确报错或记录 fallback，不能静默读取null/未初始化页面。

## 待裁决点与静态边界

- 若只需要恢复乱码关联的短针输出，Route A 的成本最低，但它有意裁断 upper
  SWA 左窗；应先让真实权重 A3-21 的同一请求对比 `legacy` 与 `bounded128`。
- 若要求 full40 等价的末尾状态，Route A 不够；Route B 引入 H20 activation
  sidechannel 或约5K-token D warmup。需要先由主 Agent 选成本路径，再写服务补丁。
- 本原型检查了 token/page 边界及 P connector 的逻辑截页，不知道运行时物理页
  ID、不读取 CANN metadata，也不验证上层 cache contents、数学精度或 graph capture。
  此分支未把任何实现合入 `feat/ced-pd-a3`。
