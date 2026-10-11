# 当前目标：正式 TP8 包含 DSpark，≤20ms/step

用户于2026-10-11最新明确：**目标是包含DSpark在内，推理20ms/step**。
此要求取代此前关闭DSpark的≤17ms目标，也取代刚才的24ms表述。
旧会话01a1275a-c0b2-7662-aaaa-49e6ca507155的工作继续，源码/证据/正式质量与交付要求保留。

- 正式40层/hidden5120/384专家top6/W4A8_DYNAMIC/Engram int8与完整视觉权重。
- DSpark使用checkpoint原生MTP3：draft为128专家/top3（TP8 local16），K5/block5。
  正式target与draft实测GMM容量分别为36/local48及15/local16，审计与候选须按role绑定。
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
Engram SDK普通NPUGraph32代可行；正常TP8初始化的v6 static_default中，5个有完整结果的rank已通过真实双AOT Tensor ABI/warmup，随后cold-capture因默认stream被框架拒绝，完整32代/alias仍未验证，原型默认关闭。其余3rank未取得完整SDK结果，不能套用同因。
稀疏MSD固定M1/capacity6已在真实E48原参数、synthetic activation的9case数值门通过（SDK/raw/candidate均maxULP0）；planner/canary/seals与输入原地修改生命周期已验证。当前未有性能、实际模型消费者或K5收益证明。正式profile实际target GMM容量36/E48，draft容量15/E16，不能直接推广M1二进制。

用户已清除旧goal并要求重新创建；2026-10-11已成功创建包含DSpark的正式≤20ms/step新goal，
工具状态为active。当前执行与完成审计以该新goal及本文件为准；旧17ms/无DSpark结论均不是新完成条件。
新goal尚未达成，不能标记complete。

正式API接线注意：现有`serve_formal_tp8.py`与`run_formal_client_acceptance.py`
仍显式要求`speculative_decoding=false`和`A=1`，是旧无DSpark验收链，不能直接用于本goal。
新API链须保留实际external→core request映射、同boot/time namespace的客户端结束时钟，
从真实worker计数获取rounds并核对initial/terminal及TP8末轮fence/Gloo join。
最终至少8个串行计数请求，warmup不计入这8个；不能使用`1000/tok/s`替代DSpark ms/step。
