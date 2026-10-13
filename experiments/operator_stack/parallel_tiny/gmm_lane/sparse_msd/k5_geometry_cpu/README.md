# 正式K5几何：target M36/E48、draft M15/E16

原CPU说明将main top6用于draft，已移至显式superseded文件，不能作为正式profile。真实checkpoint明确main routed384/top6、dspark routed128/top3、block_size5、target_layer_ids37/38/39；八rank profile逐一确认target GMM2 x/y容量36、counts48、完整E48参数，dynamic draft GMM2容量15、counts16、完整E16参数。每rank十step对应400 target、30 draft，逐step40/3。

[当前几何契约](geometry_contract.json)与原始CSV路径/完整文件SHA、小shape聚合、checkpoint SHA均已记录。Profiler shape只能证明容量；真实R=sum(counts)、active非零项数仍需绑定实际逐rank/逐layer数值。

现有M1 capacity6/E48候选不适合直接挂到任何一个实际role。下一候选应分别冻结M36/E48与M15/E16 tiling/ABI/weights source，扩展device planner，保留所有17原生数学体和全native groups fallback。坏frame回退原生调度不能合法化本身无效的native count输入。

OP profile duration sum有重叠，不能当可回收E2E预算；M1 event仅作分层诊断，不支持正式20ms收益、K5精度或整网声明。当前NPU权限0。
