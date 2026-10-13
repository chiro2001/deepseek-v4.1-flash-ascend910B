# A2 模拟 v4：主代理独立精度审计（2026-10-13）

结论：当前同一配置的 GSM100、Vision23、147456 四针精度门全部通过。
编程另见 [CODE8_RESULT.md](../CODE8_RESULT.md)，8/8 题、128/128 用例。
本收据不是地火性能结果，也不将本配置的质量借给 CED/DCP 或旧 A2 实例。

## 同实例与原始评分核验

- 服务 `dsv41-delivery-a2sim-tp8-v4`，a3-21 的 physical 0–7；SID
  `9d2ba214732b4eed84dc4f416f27c6ae`，实际 runtime fingerprint
  `61b841dfcd20d7f2892934e1fbd8086ab409a36d99d5de9beea4dd12399e53f9`。
- 三个 suite 的 identity 原件均匹配上述 SID/fingerprint，engine/ASGI 已启动，
  ranks 恰为 0–7，draft graph、BI、async 开启，Engram device index 为 0。
  原件 SHA 均为 `95bb15a8f5b7c45c4699df70832b98af445ab4fbe0fc00d16e2478dc0a8239ef`。
- 独立评分采用冻结的
  `experiments/operator_stack/parallel_tiny/dspark_general_quality_api_cpu/scoring.py`
  （SHA `4369e659ed465d8fe0b3812da0743a375acaedde95b8ad0e1514d36751399832`）。
  先确认归档 GSM/Vision evaluator 字节与该评分器的原始 source guard 一致，
  再调用原始提取、数字归一化、Vision case 规则，没有改阈值或答案。
- GSM canonical test 原件 SHA
  `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`，
  由它提取 first100 gold，与每行 raw answer 独立核对：**100/100**，empty/errors=0。
  evaluator 的 split-hash 字段是 `{train,test}` 字典；初次审计传入单 test hash 被
  原 guard 拒绝，按实际 schema 传完整 canonical 字典后通过，未修改原件/评分器。
  train 大原件留远端，SHA/路径见阶段收据，不声称本机已取回 train。
- GSM wire 恰为 100 个唯一 attempt 与 request hash；全部 HTTP200、stop、usage
  非空、response closed、已消费 DONE、unparsed chunks=0、max_tokens=512。
- Vision 原始全部 23 cases 顺序、问题、图片与答案通过原规则：**23/23**，
  empty/garbled/errors=0，mandatory negatives 通过。

## 长针 raw 核验

主代理逐件重算 4 个 raw request/response 的 SHA 与 size，与 exchange 和 runner
request SHA 一致。请求均为本 model、temperature=0、stream=false、max_tokens=64。
响应均为 HTTP200，原始 finish_reason=stop，content 与对应 needle 精确匹配，usage
与 runner 结果相同。每个实际 prompt 均超过冻结有效长度门 145982 tokens。

| 针 | prompt tokens | completion tokens | 原始 finish | 精确答案 |
|---|---:|---:|---|---|
| A | 147438 | 8 | stop | ZQ7K-3341 |
| B | 147436 | 8 | stop | VX2M-8890 |
| C | 147437 | 7 | stop | HT4P-5527 |
| D | 147435 | 7 | stop | RB9N-6014 |

runner 汇总的 finish_reason=null 是该旧解析器未取字段；不能据此判断 length 或失败。
没有用 needle runner 的 wall/completion 换算性能。

## 证据边界

完整配置、执行参数、旧/新路径、原件 SHA 见
[quality_stage_receipt.md](quality_stage_receipt.md)，审计时该文件 SHA 为
`59cde0f69d5ee61ec5521166f596c0b6d3b8dde0ae605bcd52cb49c9159b6019`。
GSM 输出 SHA `1a0323fe84e72624cf1e7d006d5e569ec2f9db318b7641ebe16b189a9cd30af5`；
Vision SHA `eadf77e0583854d3d9daaac3063c7218d56d57cdffea8de864c10eb0ff99b19d`；
needle 汇总 SHA `43589c91b3b01fbba4712a938dedfedd2d2cdb8ff8af965bc21f5198002522da`。

这里只证明这些冻结测试的严格门，未完成逐层 device consumer 追踪；
`actual_device_consumer_proof=false`。共享 Engram/单侧 DRAM 不在当前配置中。
