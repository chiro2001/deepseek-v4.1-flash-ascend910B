# 当前目标：正式 TP8 客户端 ≤17ms/step

用户于 2026-10-11 明确将约19ms目标提高到 **≤17ms/step**。
旧会话 `01a11fdd-c5ae-7a31-b350-9f4bb665c166`，当前接续会话
`01a1275a-c0b2-7662-aaaa-49e6ca507155`。本文件的17ms要求优先于历史报告和旧goal中19ms描述。

- 使用正式40层/hidden5120/384专家top6/W4A8_DYNAMIC/Engram int8与完整视觉权重。
- 正式客户端串行8请求、2K输入/256输出，审计/profiler关闭，报告 `(ms/step,A,tok/s)`。
  当前无推测解码，A=1；17ms对应至少58.823529tok/s。
- 保持 strict 原生规约；候选须通过完整路由/token/Top5/logprob与实际cache消费者检查。
- 客户端模型归属通过，GSM8K100/100、Vision23/23；最终最优方案部署并验收。
- 优先建立快速tiny的实测或可解释对齐，再继续有证据的性能优化。
  tiny/单算子结果不能替代正式TP8或质量验收。
- a3-21 chip8–15及80C98001 Alarm已授权；每次启动复查占用/Alarm，不reset、不停止其他租户。
- 报告、COS/links-server发布、源码与紧凑证据提交、从已提交源码生成MANIFEST、自检、双远端push。

当前已完成的异步metastack正式客户端为
`(19.376255ms/step,A=1,51.609559tok/s)`，GSM8K100/100、Vision23/23。
它没有达到17ms；尚需约2.376255ms/step（12.2638%）时延下降。

goal工具当前不允许覆盖未完成goal，因此旧goal保持active、未标complete。
后续工作和最终完成审计均按用户最新17ms要求进行。
