# 正式TP8客户端性能与前端修正

2026-10-10 · a3-21 physical chip8–15 · 正式权重 · core＋strict

环境变量复验结论保持不变：`HCCL_DETERMINISTIC=strict`已使本轮原生三组重复请求和base/core/stack六组精度配对的路由、Top5和logprob完全一致，delta0。`true`历史GSM8K91/100的结果未被忽略，当前采用strict并继续做客户端质量验证。

本轮新结果是正式core API的8条不同prompt、并发1的客户端decode性能：**`(27.412337ms/step, A=1, 36.479925tok/s)`**。8/8请求成功，无失败；每条准确校准2K输入、256输出，TTFT中位0.308秒。服务归属由`/v1/models`的独立模型名和唯一API进程argv双重验证。没有profiler或路由/cache审计，无推测解码，因此A=1；原始客户端无spec计数产生的accept_len=0不能当作A=0。

它是客户端decode速率的中位，和此前内部同进程12组/48输出的`(26.622860ms/step, A=1, 37.561704tok/s)`口径不同，不进行相减或累计收益。两者都尚未达到≤约19ms目标。

## 视觉请求失败的含义与修复

第一版API的23例视觉请求全部被HTTP400拒绝，包括纯文本负控。服务端响应明确为：

> As of transformers v4.44, default chat template is no longer allowed, so you must provide a chat template if the tokenizer does not define one.

这是新启动入口漏配正式tokenizer前端的错误，**不是已运行视觉模型之后出现精度差异**。因此这23次不能作为“视觉精度0/23”的模型结论；它们是客户端接口失败证据。测试未被改成通过，原始失败与退出1保留。

已在`serve_formal_tp8.py`补齐生产脚本使用的`--tokenizer-mode=deepseek_v41`、对应reasoning/tool parser、自动tool choice和默认`enable_thinking=false`。前端位于镜像的`vllm_ascend.patch.platform.patch_deepseek_v41_frontend`，由平台补丁注册，并非`vllm.tokenizers.deepseek_v41`文件。通过正确注册模块做预检后，简单文本chat渲染与checkpoint官方`encode_messages(..., thinking_mode='chat')`逐字一致。

这一修正只处理API消息到模型输入的编码，不修改模型权重、算子、规约模式或精度门槛。图内算子精度仍沿用已通过的正式审计，新的客户端质量仍必须实测。

## 重新启动与当前边界

确认第一版API独立模型名和cmdline后，仅向本任务自己的API进程发送SIGTERM；没有停止其他租户、reset设备或清page cache。第一版结束后，启动器重新检查chip8–15无占用、授权的80C98001 Alarm、容器挂载与设备映射。

新作业`formal_core_strict_service_v2`已启动：loopback18762、模型名`dsv41-a321-formal-core-strict-front-v2-20261010`，正式core＋strict＋auto通信端口。串行客户端runner已安排新服务归属验证、8条性能、23例视觉和GSM8K100，结果仍pending；不能把前端预检当作完整HTTP或视觉通过。

GSM8K官方JSONL已下载并验证：test1319题、train7473题，相同训练集前8题few-shot及test顺序。test SHA为`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`，train为`17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`。新增可选JSONL路径避免datasets/pyarrow依赖；已停止本任务自己的缓慢pip下载，不修改全局或模型环境。完整数据留远端，未经SSH搬运≥1MB文件。

## 下一步与交付

七组/112份strict微架构数据、图外metadata准备链的定位及UB双缓冲适用边界见[报告v12](https://uploads-new-1254016670.cos.ap-shanghai.myqcloud.com/share/a321-strict-microarch-20261010-v12.html)。当前优先级是图外metadata/slot/ring发射链、正式W4A8小M与稀疏专家控制开销、固定序通信的整网验证；尚无新增可计优化收益。

本轮新增：精确跨stream时间线分析、离线解析日志路径修正、正式API前端配置修正、带服务归属/有效样本检查的客户端runner、官方JSONL读取与来源SHA。紧凑证据及报告随源码自检并推送GitHub/内网镜像。Goal保持active：19ms、正式客户端质量及最优服务验收均未完成。

证据在`experiments/operator_stack/evidence/formal_a321/`：`results/formal_core_strict_service_v1/client/`的归属与性能/接口失败、`frontend_preflight.json`、`stop_owned_api.json`，以及v2的启动与客户端启动记录；`client_data/gsm8k/source.json`保存官方来源和SHA。大请求、原始trace留a3-21任务目录。
