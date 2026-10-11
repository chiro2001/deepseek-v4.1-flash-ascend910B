# 当前目标：正式 TP8 包含 DSpark，≤20ms/step

用户于2026-10-11最新明确：**目标是包含DSpark在内，推理20ms/step**。
此要求取代此前关闭DSpark的≤17ms目标，也取代刚才的24ms表述。
旧会话01a1275a-c0b2-7662-aaaa-49e6ca507155的工作继续，源码/证据/正式质量与交付要求保留。

- 正式40层/hidden5120/384专家top6/W4A8_DYNAMIC/Engram int8与完整视觉权重。
- 必须实际启用DSpark/MTP，时延包含主模型、草稿生成、目标验证、调度和cache处理。
  不能把仅开启env、草稿失效或无DSpark的A1结果作为新目标验收。
- 正式客户端串行至少8请求、固定2K输入/256输出预算，审计/profiler关闭。
  按实际target decode/verifier轮次计step，不以前端异步poll或输出token数代替。
  每请求记录decode墙钟、decoded_tokens、rounds、draft调用/提议/接受计数。
  ms/step=1000*decode_s/rounds；A=decoded_tokens/rounds；tok/s=decoded_tokens/decode_s。
  不混prefill、capture/profile或全局累积平均；首/末步处理显式记录。
- 报告(ms/step,A,tok/s)与clean-rate/复读/空答，A虚高不能算优化。
- 保持HCCL_DETERMINISTIC=strict及原精度门，完整路由/token/Top5/logprob/cache和草稿/验证实际消费者覆盖。
- 最优正式服务模型归属、GSM8K100/100、Vision23/23及包含DSpark的≤20ms客户端验收。
- 用户授权所有空闲芯片，正式TP8主线程优先a3-21 physical8–15。
  启动时复查实时占用/健康；仅80C98001 Alarm已授权，不reset、不停止租户、不清pagecache。
  SSH每文件<1MB，大权重/trace留远端或COS。
- 专属DSpark tiny线由已核验gpt-6.1-sol/max代理负责，独占健康空闲physical4。
  tiny/dummy/缩层结果只验证机制与包含DSpark的tiny时延，不能替代正式20ms/质量门。
- 报告COS/links-server发布，源码/compact证据提交，从HEAD生成MANIFEST、自检及双远端push。

当前已验收无DSpark mask客户端为(19.289956ms/step,A=1,51.840451tok/s)，GSM100/100、Vision23/23。
它是旧基线，不满足新的包含DSpark目标。尚无正式DSpark开启的20ms验收或最终部署完成声明。

最新QKV early-kv-tail-v2完整审计通过，但12组性能mask(19.208722ms,A1,52.059685tok/s)、
QKV(19.307216ms,A1,51.794106tok/s)，0/12更快，拒绝部署；同样关闭DSpark。
Engram SDK普通NPUGraph32代可行，正式对齐AOT/GraphEx/static/nonstatic仍在callback前TypeError，原型默认关闭。
稀疏MSD仅CPU构建/tiling/编译分支与adapter ABI契约成立，NPU数值与模型收益尚未验证。

现有goal工具仍保存旧17ms与paused状态，工具没有修改objective/resume接口；不能伪造旧目标完成或修改SQLite。
用户已明确继续及新20ms目标，本文件是后续执行与完成审计的当前要求。旧17ms/无DSpark结论均不是新完成条件。
