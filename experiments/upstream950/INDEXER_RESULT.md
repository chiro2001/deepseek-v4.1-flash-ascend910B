# Indexer INT8 后处理融合：保留（A3 tiny）

在独立 chip5 tiny 上，将四个 K 源层的 `RMSNorm → 尾64维 interleave RoPE → dynamic INT8 quant → K/FP16 scale cache` 合为一个 Vector kernel，保持 `wk` 投影输出和两次 BF16 中间舍入。两轮同进程交错验证均得到小幅收益；保留实现与原生回退。只启用 T=1、BF16/128/rope64、INT8 K/FP16 scale 的目标 decode 路径，prefill、空行和不适配形状仍原生。

## 正确性与真实布局

最终102个数值用例和48个布局用例全部逐位通过：Norm、RoPE、INT8、FP32 scale，以及包括 padding/其他层在内的完整 parent cache。覆盖多seed、1e-5/1/1e5幅度、zero/constant/half-tie、FP32与BF16 trig、非连续输入、int32/int64 `(block,row)`、混合负坐标和乱序页。

量化按照安装态 CANN 的真实 `Div(127,max) → Mul → RINT` 顺序，不能浮点重排成 `x/scale`；零行保持 q=0、scale=0。127分子使用一次性按设备缓存的 FP32 向量，初始化在 capture 前。RoPE mask 外地址也保持有效，避免当前工具链对非连续 trig 的负 offset 寻址差异。

随机模型审计覆盖40层的实际路由、四个源层与C2完成/未完成步；投影/坐标/缓存行在消费者点快照，8次请求的缓存、路由和logprobs一致，max delta=0。真实 K 页 stride 为131072/147712，scale页 stride65536/73856，trig为FP32。正式图不含审计复制。

## 性能

共同基线：已有 HC/router both，BF16/TP1 tiny、2K prompt、48输出、A=1、FULL_DECODE_ONLY图、profiler OFF。同模型进程保存原生/fused两个图bank，同组交错、隔组反转顺序；只取预热后decode稳态。

|轮次（各12组）|原生 `(ms/step,A,tok/s)`|融合 `(ms/step,A,tok/s)`|配对加速中位|更快组数|
|---|---|---|---|---|
|首轮|26.447915, 1, 37.810164|26.372360, 1, 37.918487|1.002683|10/12|
|确认轮|26.121245, 1, 38.283014|26.010020, 1, 38.446722|1.005068|11/12|

24组配对加速比合并中位1.003358，21/24组更快。两轮整体时延有漂移，结论依据各组配对；不能用跨进程数字直接相减。确认轮有GPU独占锁，41次全程chip4负载采样均0。收益约0.3–0.5%，不外推到真实TP8/W4A8或其他模型规模。

独立完整后处理链的设备图计时：T1原生17.862205us→融合14.206690us，配对1.257485；T4 23.784215us→14.257905us，配对1.667945。这是单算子范围，模型收益以上表为准。

## 复现与证据

实现：`scripts/indexer_post.py`、`scripts/indexer_patches.py`。选择 `UP950_CANDIDATE=indexer`、`UP950_ARM=fused`；设为baseline立即回原生。实验入口 `bench_lane_model.py --candidate indexer --arms baseline,fused`，审计加 `--audit`，正式性能通过 `run_when_idle.py`。

证据在 `evidence/final/indexer_model_audit_v2`、`indexer_model_perf_v1/v2`、`indexer_post_final_102.json`、`indexer_post_layouts_v3.json`、`indexer_post_perf.json`。完整大体积requests留远程，紧凑summary与SHA256 raw manifest已归档；未通过SSH搬运大文件。
