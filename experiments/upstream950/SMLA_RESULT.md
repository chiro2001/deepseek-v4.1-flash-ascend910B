# SMLA块索引/页表UB预取：正确但tiny中不采纳

新内核保留原gather、Cube/Vector计算和softmax，只在V0把当前AIV半tile的TopK索引及可容纳的PA页表装入原有UB前384个int32槽。不增加UB容量；大页表仍回退GM读取。

私有未改算法control与prefetch同一工具链编译，保留原生算子。Host序列化224字节逐字段验证、kernel opParaSize232验证、私有.so/二进制SHA256均已归档。

独立30项：ratio1/2、长度127/128/129/2048/2049、TopK512/1024、holes、随机Q/KV及乱序页表；control/prefetch的输出和LSE逐比特等于原生。模型审计随机化216个attention参数及HC/router/MLP，12次请求中每层消费者快照输出逐比特一致；实际路由、token、logprobs一致。

正式性能关闭profiler、审计、快照拷贝、logprobs和路由回传。12组同进程三bank正反交错：

|臂|ms/step中位|A|tok/s|配对加速比中位（相对原生）|
|---|---:|---:|---:|---:|
|原生HC/router基线|26.253965|1|38.089485|1|
|control|26.120645|1|38.283893|0.996142|
|prefetch|26.293815|1|38.031757|0.989236|

预取相对同工具链control的配对加速比中位0.993213、时延差中位+0.179455 ms；仅1/12组更快。相对原生仅1/12组更快。当前形状无稳定收益，默认保持原生。

本轮有整体时延漂移，因此不能单独用各臂聚合中位相减解释细小变化；结论依据同组配对及同工具链control。不能外推到长序列、多Query或其他tiling，也不排除后续Vector批量地址生成有收益。

审计曾因跨层共享TopK buffer被decoder覆写而在原生复算中产生NaN。现仅在audit capture于消费者点clone输入/TopK/raw输出；性能图完全不包含这些clone。没有放宽equal_nan或正确性门槛。

可复现入口：scripts/build_smla.sh、verify_smla.py、bench_lane_model.py --candidate smla --arms baseline,control,prefetch。完整远程requests.json保留且raw_manifest.json记录SHA；摘要、配对原始时间、调用覆盖和源/产物哈希入当前分支。
