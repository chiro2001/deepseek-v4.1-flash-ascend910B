# 950 算子优化向 A3 独立 tiny 迁移

六个方向已逐项处理。只在可直接访问的A3 `a3-21` chip5验证，未准备A2验证包；所有实验使用独立模型副本、源码、缓存、私有产物及Git分支。原chip4优化线未被修改或停止。

|方向|最终处理|证据|
|---|---|---|
|稀疏专家GMM列表|正确但无收益，原生回退|[SPARSE_GMM_RESULT](SPARSE_GMM_RESULT.md)|
|SMLA索引/页表当前块UB预取|正确但无收益，原生回退|[SMLA_RESULT](SMLA_RESULT.md)|
|Indexer INT8后处理融合|保留并在独立tiny服务启用；两轮配对约0.27%/0.51%|[INDEXER_RESULT](INDEXER_RESULT.md)|
|inverse RoPE / 八组wo_a|三种候选均无收益，原生回退|[EPILOGUE_RESULT](EPILOGUE_RESULT.md)|
|Q/KV、q_b面板与预取|joint改变实际路由；NZ/panel/prefetch正确但无收益，原生回退|[PROJECTION_RESULT](PROJECTION_RESULT.md)|
|SMLA UB→L1交接|当前标准SDK路径是GM软件中转；直写门禁不通过|[UB_L1_RESULT](UB_L1_RESULT.md)|

分支 `feat/tiny-upstream950-a2a3-20261009` 的名字保留最初范围历史，执行范围已按用户最新指令收窄为A3。上游固定引用见初始调研 `tiny_upstream950_20261009/OPTIMIZATION_POINTS.md`：cann-recipes-infer `4c3d1e1258053c4d810322198a7fd1722c89897e`、cann-recipes-train `a446bab76f2da1e18fe3e16f8d609b0cc6c34779`、ops-transformer `59dd0c236a6d57f03d26474d8c18c0df1bdc911d`。

## 独立环境与运行

远程 `/home/l00886679/projects/dsv41-tiny-upstream950-20261009`，容器 `dsv41-tiny-upstream950-20261009-c5`，物理chip5/运行时npu:0。CANN9.1.0、torch2.10.0+cpu、torch_npu2.10.0.post4、vLLM0.27.1。40层、hidden5120、64heads×512、q/o rank512、8专家/top2、BF16/TP1 dummy，关闭Engram/DSpark/prefix/async，4GiB KV和decode图[1]。保留已有HC/router both基线。

服务已恢复，默认 `UP950_CANDIDATE=indexer UP950_ARM=fused`、worker为lane_worker.Upstream950Worker。原生回退：`UP950_ARM=baseline`；完全回原tiny worker：`UP950_CANDIDATE=none`。这两个选项应用于重新启动自己的服务，不涉及其他容器。

桥接网络地址在 `evidence/final/restored_service.json`；恢复时为A3主机内 `http://172.17.0.6:18973`，重建容器后IP可能变化。容器内为localhost:18973。安装在本容器私有 `/tmp/mysvc.sh` 的身份检查须返回MINE，HTTP冒烟断言精确model ID、32输入/16输出；宿主的原 `/tmp/mysvc.sh` 未改动。

## 复现与边界

`scripts/sync_lane.py`同步小源码；`run_remote.py`只启动本容器任务。模型单变量图bank通过 `bench_lane_model.py`，随机审计加`--audit`；正式计时用`run_when_idle.py`等待chip4和本lane空闲，再以独占锁运行并全程采样卡负载。不得把审计时延或不同进程整体时延相减当收益。

源代码、精度数据、紧凑请求摘要、raw SHA256 manifest、编译参数/产物SHA及最终服务状态分别存于scripts、evidence及结果文档。完整大requests留远程；SSH传输归档均小于1MB。没有覆盖镜像框架源码或全局SDK。

所有收益只代表此BF16/TP1 tiny。其他模型规模、TP8/W4A8、长上下文或完整950融合结构尚未按此泳道验证。详细逐项进度见[PROGRESS](PROGRESS.md)。
