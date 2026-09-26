# CED D 图模式首/次 token 定位

P 保留原实例（chips0–7，TP8，GRAPH=1/EAGER=0）；D 仅在相同包、模型、角色、挂载和参数下切为 GRAPH=1/EAGER=0（chips8–15，TP8）。实际 D 命令含 FULL_DECODE_ONLY、无 --enforce-eager；D role=decode、CED connector/dsa/replay补丁挂载齐，图捕获decode FULL完成4/4。P/D都health200，proxy served model ID正确。image ID与P相同（sha256:1f2c08195c5b119aa9a107861fa2efa5d2b93b161f1ae91cb5358923a78cbcef）。

实际P/D容器中的 breakable_cudagraph.py、device_metadata.py、acl_graph.py、vllm_ascend model_runner_v1.py、core gpu_model_runner.py 五项源码SHA相同；见 runtime_source_fingerprints.txt 与 graph_minimal probe/meta 的 p_runtime_source_sha256.txt、d_runtime_source_sha256.txt 和 runtime_source_compare.txt。关键调度/metadata行片段见 runtime_source_excerpts.txt。

与eager臂完全相同的max1/max2请求SHA（bd6276ad…e2ff5 / 403e6cb2…21378a）各提交一次，均HTTP200并返回logprobs：
- eager max1/max2为 Z / ZQ；
- graph max1为“正确答案”，首 token偏离Z，logprob=-2.13554；
- graph max2为“正确答案只有一个”，token序列为“正确答案”→“只有一个”。

图臂D replay两次都覆盖positions=0..20并执行chunk21（8个TP worker均有记录）。该诊断支持首生成token在FULL图路径出现偏离；请求仅为单一22-token提示，不能直接推广到长上下文。没有长请求、代码修复或额外模型请求。原始body/logprobs、完整P/D/proxy日志、inspect、运行时源码指纹和SHA清单均在本目录。
