# K5真实几何与target route子集：CPU可行性

当前证据支持推迟整网candidate投入。M1没有统一sparse-vs-raw事件收益，而当前已获益模式在完整接受且输入齐备的真实target route投影中只占0.238%，去重占0.223%。详见[可行性结论](feasibility_decision.json)和[两role协议设计](DESIGN.md)。所有本目录工作均为CPU，NPU调用/权限0，未改变冻结M1 v2或v1失败证据。

真实几何为target M36/E48/top6与draft M15/E16/top3，8rank step9逐项40/3调用通过。[source guard](evidence/source_guards.json)给出checkpoint SHA、实际draft override、私有draft配置消费者、EP rank到连续expert区间和shape独立communication cache源码。[step9 geometry](evidence/actual_step9_geometry.json)保留全部CSV完整SHA。此前缺失的override证据同步补入相邻`k5_geometry_cpu/native_MTP_config_override_source.json`。

只选`formal_dspark_native_eager_aa_v3`的6个测量请求，排除warmup。48个nominal round中42个实际输出六token、接受五draft且六输入route齐备；每个int32输入切片SHA与全部8rank的main_route_receipt一致。尾部6轮即使counter显示nominal accepted5，也因输出/route不足六项排除，不推定缺失项。42轮含A/A重复，只有14个唯一route切片。[逐round筛选](evidence/round_selection.json)列出所有接受/排除依据。

| target路由投影 | 42 round ×40 layer ×8 rank |
|---|---:|
| 观测数 | 13440 |
| R中位/均值/最大 | 4 / 4.5 / 19 |
| active中位/均值/最大 | 3 / 2.7213 / 10 |
| R=0 | 720 |
| single active expert | 2362 |
| single local e47 | 38（0.283%） |
| single local 40–47 | 428（3.185%，该整体范围性能未测） |
| 精确R1/R3/R6单个e47模式 | 32（0.238%） |

[统计JSON](evidence/route_distribution_summary.json)包含R/active/last_eid完整分布、去重口径、320个layer×rank分组和源文件SHA；[分组CSV](evidence/per_layer_rank_summary.csv)便于扫描。2994个投影R>6，218个active>6；旧M1的6行/6槽协议不满足真实分布。全部K5分配几何也与M1 binary不一致，不能仅按R小就接入。

本统计是CPU投影，**不是实际GMM group_list、prepack activation或设备consumer门**；draft route覆盖0，缺失/未接受round不在结论内。原NPZ只在远端读取；4230452B完整投影JSONL留远端，SHA `56dfdbacc8bb180c2a8a41617be6a13897706028eeb244ff07d1499e904b14ee`。CPU全native fallback协议的1266调度/6680坏frame/8非法输入/13440投影检查通过，详见[协议门](evidence/planner_protocol_CPU_contract.json)。
