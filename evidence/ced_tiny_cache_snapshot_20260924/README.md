# A3-22 tiny CED cache snapshots

日期：2026-09-24。A3-22 使用 Phy-ID 6 的既有 prefill 服务和 Phy-ID 7 顺序
运行 CED decode、全 40 层 baseline；Phy-ID 0、1 保持未触碰。模型为
`model-tiny` dummy 权重，TP1、BF16 KV、seed 0、Engram/DSpark 关闭。

快照诊断包位于远端
`~/projects/dsv41-ced-singlechip/pkg-snapshot-2df46ec`。容器挂载的
`experimental/ced/dsa_v41.py` 与包内源文件 SHA-256 均为
`6f93b6eabc97dfeaaf126a7836a7eb74992275443c6dea652d5e3321e2a2bc7e`。
每个 API 请求使用 `tools/ced_single_precision.py` 固定的 8-token pattern，
`temperature=0`、`max_tokens=1`、`logprobs=20`。

## 位置 126，prompt 长度 128

- D 请求 HTTP 200；日志 replay `positions=0..126`，共 127 tokens。
- D 与 baseline 选中 token 和 logprob 完全相同，top-20 集合 20/20 相同。
- 40 层 SWA 首个 bitwise 差异在 L7，max abs `2.98e-8`、rel L2
  `2.19e-6`。L19 为 `1.91e-6` / `1.72e-4`，L20 为 `1.91e-6` /
  `1.60e-4`；SWA 最差 L27 rel L2 `6.26e-4`，L39 为 `5.01e-5`。
- L20 `long_kv` 与 SWA 行差值相同；`index_scale` 精确相同，`index_k`
  max abs `1`、rel L2 `3.02e-3`、cosine `0.999995456`。

## 位置 254，prompt 长度 256

- D 请求 HTTP 200；日志 replay `positions=127..254`，共 128 tokens。
- D 与 baseline 选中 token 和 logprob 相同，top-20 common/union 为
  19/21，最大共同 logprob 差 `9.54e-7`。
- SWA 首个 bitwise 差异在 L2，max abs `2.98e-8`、rel L2 `2.44e-6`；
  首个 max abs ≥`1e-6` 在 L4，为 `7.63e-6` / `5.81e-4`。L19 为
  `1.91e-6` / `3.15e-4`，L20 为 `7.63e-6` / `8.04e-4`，L21 为
  `3.81e-6` / `5.21e-4`。前半最差 L14 rel L2 `1.365e-3`，后半最差
  L39 `9.46e-4`，没有明显的 L19/20 跳变。
- L20 `long_kv` max abs `7.63e-6`、rel L2 `6.66e-4`、cosine
  `0.999999779`；`index_k` max abs `1`、rel L2 `4.27e-3`、cosine
  `0.999990893`；`index_scale` 精确相同。
- 位置 254 是 replay 的末行；该行注意力窗口左界已到 replay 起点。这份
  末行对照不能直接观察 replay 最前面的 SWA 窗口缺口。要直接测该点，抓
  长度 256、位置 127。

## 位置 127，prompt 长度 256（replay 第一行）

- D 请求 HTTP 200；日志 replay `positions=127..254`，pos127 是该 128-token
  replay 的第一行。D 与 baseline 选中 token 和 logprob 相同；top-20
  common/union 为 19/21，最大共同 logprob 差 `9.54e-7`。
- 40 层 SWA 中 L0 精确相同，首个 bitwise 差异在 L1：max abs
  `1.526e-5`、rel L2 `1.435e-3`、cosine `0.999998983`。误差随后扩散：
  L19 rel L2 `3.967e-3`，L20 `4.069e-3`，L21 `4.321e-3`，L39
  `6.574e-3`、cosine `0.999978397`；各层最大绝对差约 `1.526e-5`。
- 同一个 prompt 的 pos254 末行差异更小：L1 精确、L19 rel L2
  `3.153e-4`、L20 `8.043e-4`、L39 `9.458e-4`。首行的较早分叉和较大
  相对误差符合 replay 起点窗口读取到 pre-start SWA 状态的假设；它是数值
  迹象，不是内存读取来源的直接追踪证据。
- L20 `long_kv` max abs `2.98e-8`、rel L2 `2.82e-6`、cosine
  `0.999999999996`；`index_k` 和 `index_scale` 精确相同。因此该对照显示
  SWA 重算行发生偏差时，层20全局主 KV/Indexer 基本保持一致。绝对误差仍
  只有约一个 tiny dummy 的 BF16 步进，真实权重行为需要另测。

所有对拍仍使用 dummy 权重，只能说明该配置下的缓存数值和 API 结果，不能
替代真实权重精度验收。原始数组、响应及逐行指标分别保存在本目录的
`d_pos126/`、`base_pos126/`、`d_pos254/`、`base_pos254/`、`d_pos127/`、
`base_pos127/` 和
`cache_compare_pos*.json` 中。原 baseline 的旧容器、启动文件和日志另存于
A3-22：`~/projects/dsv41-ced-singlechip/evidence/ced_tiny_base_original_ccf5762_20260924/`。
