# 进度：独立 tiny / 950 优化迁移

2026-10-09：用户要求建立第二条独立 tiny 泳道，先参考 infer/train 上游寻找950优化向A2/A3迁移的机会。

已完成：

- 获取两个recipes仓库及其引用的ops-transformer，固定提交。
- 阅读现有chip4优化线；其当前新增开发重点为clamped SwiGLU，不修改该目录/容器。
- 通过a3-21设备实况和fuser确认chip5当时无进程占用且健康。
- 创建独立根目录/模型metadata副本/容器/cache；服务标识和端口与原线独立。
- 设备冒烟、服务身份与completion冒烟通过；HC/router图覆盖40/80确认。
- 只读收集213份框架相关源码及SHA256、本机torch_npu API文档。
- 验证稀疏group_list_type2：BF16 ND/NZ，两个真实tiny GMM维度、两组expert IDs，共8个用例逐比特一致。
- 完成OPTIMIZATION_POINTS.md，按证据区分可直接试的接口、需要重做的内核及950专属路径。

环境修正：初次补查NZ时触发GE编译，发现ASCEND_CACHE_PATH目录需预建；已创建所有独立cache子目录并修正setup脚本。NZ探针进程单独打开allow_internal_format，以确认格式确为29；未改变服务进程设置。

尚未实施候选、尚未报告新性能收益、尚无A2实机验证。建议下一轮先稀疏GMM完整链A/B，再SMLA地址预取。UB→L1需先验证DAV_2201的合法执行路径。

服务bridge当前172.17.0.6:18973，地址可能重建后变化。容器dsv41-tiny-upstream950-20261009-c5，远程根/home/l00886679/projects/dsv41-tiny-upstream950-20261009，原chip4线保留，排除14–15。chip4/5为同一卡两个die，正式计时避开另一线忙碌窗口，不跨die相减。
