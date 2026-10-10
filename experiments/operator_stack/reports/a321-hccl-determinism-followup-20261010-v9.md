# HCCL环境变量历史核对与正式图模式复验

2026-10-10 · a3-21 chip8–15 · 目标仍active

用户指出文档曾记录用环境变量解决归约不稳定。本轮核对了`reports/target-requires-A3493.md`、`CORRECTNESS_STATUS.md`和正确性线完整记录，确认开关为`HCCL_DETERMINISTIC`，启动脚本参数为`HCCL_DET`，合法值是`false/true/strict`。已有记录不足以替代当前正式权重与CANN版本的测试，因此现在优先复验`strict`。

## 历史证据应如何理解

正确性线最初记录：`true`使2K/8K/16K上下文的首位置logprob spread降为0；16385位置仍有0.012。`strict`曾在部分扫描中得到spread0，GSM8K-100为100/100、Vision23/23；`true`的GSM8K仅91/100。不能仅从开关名称推断它的实际累加精度。

同一文档的后续P0-1b与仓库`CORRECTNESS_STATUS.md` §2更正了“短上下文保证完全确定”的结论：同一ctx重复批次出现0.913/0/1.435等spread，16384也从0变为0.427。该结果限制了历史保证，不否定开关的诊断价值，也不能直接否定当前2K/40层/W4A8模型上的效果。

此前本线优先做FP32与固定顺序求和，没有先补齐当前模型的`strict`控制；本轮调整验证顺序。

## 当前已知事实

- 原生生产基线的同prompt请求在decode路由和Top5 logprob上不一致；固定输入归约重复也不稳定。
- FP32归约通过正式eager三组A/A及8rank×80次独立FP64参考转BF16；整网图模式的base A/A仍拒绝通过，尚未进入候选bank比较。
- 真实向量的独立试验中，普通/补偿固定顺序AllGather求和的256/512/1024 block配置均通过8rank数学参考与三次图重放；原BF16失败，原FP32图重放在4rank未通过。该独立试验不是整网验收。
- 独立事件时间约284–290μs，包括该测试中的graph发射和rank到达影响，固定方法顺序，不能当作整网每次通信的代价或端到端收益。

## 已启动的正式复验

作业`formal_native_graph_strict_v1`使用正式checkpoint、标准原生控制worker、FULL_DECODE_ONLY capture `[1]`与static kernel。除启动前设置`HCCL_DETERMINISTIC=strict`外，保持原生归约，不安装FP32或候选算子bank；2次warmup、reference、3次同prompt重复，检查完整路由、Top5集合和logprob `<1e-3`。

启动前确认8–15无占用、逐chip仍为已授权80C98001。新进程继承开关，已有通信域不热切换。启动记录明确保存环境值、命令、源码sha与镜像digest；当前结果尚未完成，不宣称开关已经解决本轮问题。

先看`strict`能否通过当前原生图A/A，再补`true`诊断对照；若稳定性改善，仍要验证独立求和误差、客户端质量与图模式性能，才可采用。只有精度门通过后才做有效端到端`(ms/step,A,tok/s)`配对。19ms目标与最佳正式服务仍未验收。

## 资料入口

- 仓库`CORRECTNESS_STATUS.md` §2、§6.3：初始结论、重复批次更正和交付限制。
- 本地完整正确性线：`/home/chiro/projects/dsv41/corr_work/correctness-line.md` §P0-1、§P0-1b。
- [CANN HCCL_DETERMINISTIC](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/hcclug/hcclenvref_07_0010.html)：确定性/保序语义与AIV展开限制，实际效果以当前版本测试为准。
