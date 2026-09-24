# A3-21 真实权重 CED-PD：位置 20 数值对照（2026-09-24）

同一 22-token 聊天请求：`校验码是 ZQ7K-3341。请只回复这个校验码。`，
`temperature=0`、`max_tokens=32`。模型 `v41-flat-verify3`，A3-21 TP8，
BF16 KV、`MAX_LEN=1048576`、`BAT_TOKENS=8192`、`MAX_SEQS=4`、
Engram host、`SPEC=0`、`PREFIX=0`。全 40 层直接服务精确回答
`ZQ7K-3341`；CED P→D 返回混杂文本且未含校验码，HTTP 均为 200。

全模型、CED P、CED D 三臂分别抓取位置 20 的 H20 多流隐状态。
全模型与 D 还抓取 40 层 SWA，以及层 20 的全局 KV、Indexer K/scale。
全部 8 个 TP rank 均有有效文件。P 端按设计只导出 H20，不导出普通
attention 缓存快照。D 日志确认 replay `0..20`，P 侧仅传 G0–G6，
G7–G11 上层 SWA 标记为缺失。三臂均使用 `dc174e9` 诊断包；比较工具
来自同一系列的 `43b3dc9`，用 `--tp 8 --position 20` 生成本目录两个 JSON。
P 和 D 的 `/tokenize` 对原聊天 messages 加 generation prompt 均为 22 token，
token ID 序列逐项相同；该只读对照归档于
`share/xfer/ced_tokenize_pos20_4bdd339_20260924.tar.gz`，SHA-256
`89cbb5fb2d444bd6658cc0d20077daa1a9c8049c942391b1a59c759b2ba1184a`。

| 比较 | 隐状态最大相对 L2 | 最大绝对差 |
| --- | ---: | ---: |
| P H20 vs 全模型 H20 | 0.091139 | 1.875 |
| D H20 vs P H20 | 0.096624 | 2.375 |
| D H20 vs 全模型 H20 | 0.042730 | 0.796875 |

D 与全模型的 SWA 缓存：层 0 在 8 个 rank 上逐值相同；层 1 首次不同，
最大相对 L2 为 0.008883；层 13 为 0.029224、层 14 为 0.100179、
层 19 为 0.067418、层 20 为 0.119268、层 39 为 0.240409。
层 20 全局 `long_kv` 最大相对 L2 为 0.066556，`index_k` 为 0.081300，
`index_scale` 为 0.064775。各 rank 的比较数值完全一致，且所有抓取值
均有限。逐层与逐 rank 结果见 [cache_comparison.json](cache_comparison.json)
和 [h20_comparison.json](h20_comparison.json)。两文件 SHA-256 分别为
`c98944c73b645789518b81570faa0999a671784f833844b10908dd92a16333d0`、
`de29c2c91be08913ff06ae7c74753d6fe101901808464de50d87745949ce7afb`。

## 同进程重复噪声地板

同一 CED P/D 进程、`PREFIX=0`、无并发、相同请求第二次执行时，回答仍错且
与第一次不同；第二次有 4 个 U+FFFD。两侧快照目录在第二次请求前独立改名
并保留，未修改模型/容器。P 的位置 20 层 0 `post` hidden 相对 L2 差
`0.010212`，H20 差 `0.081019`；D 分别为 `0.009631`、`0.075454`。
两侧层 0 `pre` hidden 逐值相同，层 1/14 Engram lookup 在各自两轮均逐值
相同。完整逐 rank 和阶段指标见
[repeat_comparison.json](repeat_comparison.json)，SHA-256
`c2d9bbfda96889a43d9eac6598cc0fd911d9631953fc67e4b3e9dbac0ba35d6e`。

**因此上表的跨 P/D/基线差值与同臂波动同量级，不能单独当作 CED
角色或传输造成数值错误的证据。** 下一步用全 40 层直接服务在同一进程
重复相同短请求，量其层 0 与 H20 噪声；若全模型稳定，才继续缩到 CED
角色、回放元数据或 KV 传输。Engram lookup/mask 的逐值一致也说明当前
位置 20 的首次分叉不在查表结果本身。

原始快照及运行日志经私有 COS 保存，远端和本机归档 SHA 已核对：

| 归档 | COS key | SHA-256 |
| --- | --- | --- |
| 全模型 | `share/xfer/ced_full40_pos20_dc174e9_20260924.tar.gz` | `af6bd76ec05499c7b8c4a0c60598aa9d018233949f2a8393f213b611d46394bc` |
| CED P | `share/xfer/ced_prefill_pos20_dc174e9_20260924.tar.gz` | `8dc96481f10a803f39291f35713a573456eb6c43698aed615a6fd5ae0b8d2672` |
| CED D | `share/xfer/ced_decode_pos20_dc174e9_20260924.tar.gz` | `aa8ac33d78ad286fe3d2852e8a6a9e2761b710023eaa998e68ddc2318b9fc2be` |

SWA 快照记录的是本层 attention 写入的 KV，层 1 首次不同不能单独证明
Engram 查表是根因：层 0 attention 输出也可能先分叉。P 处理 `N−1=21`
个 prompt token，D 重放 21 个，全模型直接处理 22 个；跨请求形状可能造成
数值漂移。下一门应在同一位置记录层 0 输出和层 1/14 Engram 门控前后、
lookup 与 token ID，再区分注意力、Engram、缓存交接的贡献。
