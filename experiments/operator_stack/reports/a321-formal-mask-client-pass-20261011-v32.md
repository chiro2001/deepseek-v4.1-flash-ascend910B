# 正式mask客户端19.290ms、GSM8K100/100与Vision23/23通过

2026-10-11 · a3-21 chip8–15 · 正式40层/hidden5120/384专家top6/W4A8_DYNAMIC/Engram int8及完整视觉 · strict · 当前目标≤17ms/step

新mask客户端验收全部完成：**(19.289956ms/step,A=1,51.840451tok/s)**，GSM8K100/100、Vision23/23、模型/API进程归属通过。17ms性能门槛明确为False，目标继续active；距17ms还需2.289956ms/step，约11.8712%时延下降。

## 正式客户端证据

`formal_moe_mask_delivery_controller_v1`退出0，`formal_best_async_service_v6`使用不可变 `/work/src_mask_model_v2`、loopback18766、独立模型名 `dsv41-a321-formal-best-v6-async-mask-20261011`。请求前同时校验 `/v1/models` 与唯一API进程完整argv，拒绝其他租户或错误模型。

- 性能：串行8请求，2K输入/256输出，审计/profiler关闭，单流decode中位51.840451tok/s；没有推测解码，A=1。
- GSM8K：官方train/test JSONL、相同8-shot及测试顺序、chat、conc1、limit100，正确100、空答0、错误0，用时272秒。
- Vision：23例全部通过。没有继承旧metastack的质量结论。
- 内部同实例12配对另为mask19.228652/A1/52.005725，较metastack中位节省0.149769ms、12/12更快。内部与客户端、跨会话旧报价各自独立，不能相减成额外收益。

目标文件已更新当前客户端基准为19.289956ms。旧goal工具不能覆盖未完成goal，后续始终按用户最新17ms要求推进，未标完成。

## 下一条INT8投影候选

正式采集的QuantBatchMatmul家族累计约1.898ms/step，只用于选热点。重点形状为q_a `[1,5120]×[5120,1280]`、q_b `[1,1280]×[1280,4096]`、KV `[1,5120]×[5120,512]`。

已核对正式参数文件与原生加载：q_a/KV为复制线性层，q_b按ColumnParallel每rank4096输出行切片；两正式层0/20，原始INT8权重和F32文件scale加载为正式BF16参数dtype，再转置contiguous并转换NZ。新候选以INT32整数点积避免改变累加顺序，分别验证三种缩放次序和ND/NZ读取方式。原1BF16 ULP门保留，额外记录逐位一致率；需要通过真实模型消费者及全部整网门后才可能采用。

首轮 `formal_quant_gemv_probe_v1` 在NZ格式前置检查退出1，未进入数值/性能验证。原因是独立脚本默认 `allow_internal_format=False`，而正式 `model_runner_v1.py` 设置True；已按同一契约修复，拒绝将ND当NZ读取。新不可变 `/work/src_quant_gemv_probe_v2` 的v2探针将继续两层×真实复制/八rank切片共120例/候选，再在有提交余量的同芯片图内配对。CPU已验证NZ地址公式与INT32累加无溢出；这些CPU检查不是NPU精度验收。

## 资源与交付状态

质量全部完成后，核对API当前argv和实时模型归属，仅向本任务PID836005发送SIGTERM，保存 `stopped_for_quant_probe.json`。下一次探针启动复查8–15无占用、Alarm均为已授权80C98001；不reset、不清page cache、不停止其他租户。当前没有在线17ms服务。

客户端紧凑证据与原始探针失败日志保存；完整权重、请求和trace留远端。报告上传COS/登记links-server、下载SHA复核；源码和证据提交后从HEAD生成MANIFEST、自检并双远端push。17ms完整目标继续执行。
