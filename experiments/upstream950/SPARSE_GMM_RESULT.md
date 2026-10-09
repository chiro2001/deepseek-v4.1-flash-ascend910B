# 稀疏专家列表：完整链路完成验证，tiny中不采纳

在chip5同一模型进程中，原生路由直接产出类型2稀疏表，两次GMM共用，不增加列表构造kernel。随机非恒定门控/HC/专家权重的审计通过：40层route、GMM1和GMM2，展开token、row_idx、专家计数、GMM输出逐比特一致；2组请求路由和logprobs一致。

正式计时关闭profiler、审计、logprobs及路由回传，12组正反交错A/B：

|臂|ms/step|A|tok/s|
|---|---:|---:|---:|
|HC/router baseline|26.035690|1|38.408815|
|Sparse Group List|26.193240|1|38.177789|

配对加速比中位0.993465，仅1/12组更快，配对时延差中位0.171270 ms。该tiny的8专家/top2下无收益，默认保留baseline。此结果不外推到384专家/TP8/W4A8。

候选实现保留在scripts/sparse_gmm_patches.py，可在实验图bank中回退；没有覆盖框架源文件或生产默认。原生routing type2接口的实测证据和ND/NZ对照另存。

远程完整requests.json保留，导出request_summary.json及raw_manifest.json记录完整文件SHA256；所有计时和审计摘要已入当前分支。
