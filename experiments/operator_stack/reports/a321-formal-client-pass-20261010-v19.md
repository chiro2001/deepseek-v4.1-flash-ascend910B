# 正式叠加方案客户端验收通过，继续W4A8前缀优化

2026-10-10 · a3-21 physical chip8–15 · 正式40层/hidden5120/384专家top6/W4A8_DYNAMIC/Engram int8及完整视觉 · HCCL_DETERMINISTIC=strict

正式 `tp8metastack`（core＋12组slot转换合并＋Indexer融合）客户端完成独立验收：GSM8K **100/100**、Vision **23/23**，客户端8条串行请求、2K输入/256输出为 **(26.292508ms/step，A=1，38.033648tok/s)**。无推测解码，profiler与路由/cache审计关闭，未达到约19ms目标。

同实例四方案12组交错配对的内部engine-step数据在[报告v18](https://uploads-new-1254016670.cos.ap-shanghai.myqcloud.com/share/a321-fair-stack-performance-20261010-v18.html)：base `(30.183590ms,1,33.130585tok/s)`、core `(26.840930ms,1,37.256533tok/s)`、slot合并 `(25.724235ms,1,38.873848tok/s)`、叠Indexer `(25.694245ms,1,38.919221tok/s)`。两slot候选各11/12快于core；Indexer增量很小。这些内部48输出数据与客户端256输出墙钟不同，不能直接相减。

## 验收归属与控制器修正

服务作业 `formal_best_strict_service_v3`，源快照 `/work/src_manyslots_v8`，loopback18763，模型名 `dsv41-a321-formal-best-v3-tp8metastack-20261010`。核对 `/v1/models` 与容器唯一API argv后才发请求。正式权重路径 `/home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq`，config SHA256 `40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424`。

首轮客户端早于 `service_command.json` 创建而遇FileNotFoundError，未发请求。修正为最长1800秒等待该记录，等待中检查服务退出，保留首轮失败证据。只重跑控制器，服务保持同一进程、同一源码。重跑 `client_runner_v2.exit=0`，完整验收 `client_quality_passed=true`，GSM8K100题无空答/请求错误。原始请求与答案留远端，紧凑acceptance/timing/vision/GSM summary和启动记录入库。

## 下一项优化的原理与验证计划

现有W4A8源码显示token dispatcher输出type1专家counts，GMM1/2均读取；counts模式在核内重复扫描前面专家以取得prefix边界。GMM host与核函数支持type0 prefix以及type1 counts，拒绝type2。因此下一项试验保留type0前缀，先测一次cumsum供两次GMM复用，再测routing直接生成prefix，避免添加转换kernel；保持专家顺序、W4权重、A8动态量化与所有运算不变。

必须先验证本地48专家、零计数/稀疏路由与真实输入的整数边界，再验证真实W4A8 GMM消费者及完整路由/Top5/logprob，实际覆盖非0后才执行同进程关闭审计的交错配对。未经精度/性能验证的候选不部署。UB双缓冲仍需tile循环、队列字节预算与事件重叠的证据，不能仅修改buffer数量。

持续优化目标仍为约19ms/step，并保留已验收方案可恢复。新增报告上传COS并在links-server登记，代码/紧凑证据/MANIFEST自检后双远端push。
