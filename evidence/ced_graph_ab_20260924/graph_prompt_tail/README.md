# Prompt-tail eager patch：图模式精度验证（短针、144K、1M）

P/D均为A3-21真实权重TP8+TP8。P保持原有`f28cd71`基线进程；D使用`fix/ced-graph-prompt-tail@777bc73`（tar SHA `f096c822b4a5ec75fed019f55856c58ce7d101083715261590f07cbff72a2897`），只用chips8–15，设置`V41_CED_GRAPH_PROMPT_TAIL_EAGER=1`、`GRAPH=1`、`EAGER=0`。BF16 KV、MAX_LEN=1M、Engram index0、CPU_BIND=0、SPEC/PREFIX/DRAFT_GRAPH等配置保持原值。A3-22未使用。

短针同SHA max1/max2分别HTTP200输出Z/ZQ，logprobs可用；32-token请求同SHA精确返回ZQ7K-3341。D八worker每条都记录“one-token prompt tail forced eager”；max2之后出现“Replaying aclgraph”。之后复用历史144K A/B输入，A/B都精确通过：
- A SHA e7867fa9…368a6，offset1,000,000/body144404，回答ZQ7K-3341，usage144462/8/144470，wall10.561s；
- B SHA ece2ab10…9ea85c6，offset1,146,000/body144131，回答VX2M-8890，usage144188/8/144196，wall10.609s。
所有请求HTTP200、U+FFFD=0。D replay A 144333..144460、B 144059..144186（各chunk128），短请求0..20/chunk21。

D容器image与P相同。补丁基底runner SHA为`67035d97f1cea4ae2df31adcc33f1de952f4cab6d8421e76df512296e0e3185e`，容器应用补丁后runner SHA为`bd250a59819dd806d16706177840c057416944c762264f2a291c608d915c2aff`；其他runtime SHA与P相同。

## 1M A–D结果

在同一D实例上顺序提交A–D，均为`offset=0`、`temperature=0`、非流式、`max_tokens=64`。实际body均为1,019,789 tokens，响应均HTTP 200、U+FFFD=0。请求SHA、响应SHA、usage、耗时和原始文件见`probe/ced_graph_prompt_tail_probe_20260924_090512/needle_1m_graph/`。

| 针 | 预期 | 实际 | prompt/completion/total | wall | D replay位置 |
| --- | --- | --- | --- | ---: | --- |
| A | `ZQ7K-3341` | `ZQ7K-3341` | 1,019,847 / 8 / 1,019,855 | 102.62 s | 1,019,718..1,019,845 |
| B | `VX2M-8890` | `VX2M-8890` | 1,019,846 / 8 / 1,019,854 | 102.17 s | 1,019,717..1,019,844 |
| C | `HT4P-5527` | `HT4P-5527` | 1,019,848 / 7 / 1,019,855 | 101.89 s | 1,019,719..1,019,846 |
| D | `RB9N-6014` | `Tech-D9q7Wm` | 1,019,847 / 8 / 1,019,855 | 133.17 s | 1,019,718..1,019,845 |

四条D replay均为128-token chunk，8个TP worker都记录了replay chunk复用和“one-token prompt tail forced eager”标记。失败D的原始messages正文精确搜索`Tech-D9q7Wm`为0次（正文1,389,423字符）；搜索计数和字符位置记录在`needle_1m_graph/D_wrong_answer_body_search.txt`，没有保存任何匹配上下文。

归档包含A–D原始request/response、逐请求metadata、usage和wall、1M replay摘录，以及A–D结束后的完整P/D/proxy日志快照。P与D启动日志原文分别报告GPU KV cache size为3,322,350 tokens；D各worker的Available/Current KV cache memory约15.15–15.17 GiB。完整原始行保留在`probe/meta/live_logs_pre_max1/`，不据不同日志口径推断容量变化。对这份P/D完整serve.log按ERROR、Traceback、OutOfMemoryError、OOM、RuntimeError扫描，未发现匹配项。

## 失败D首token诊断

初次1M D失败后、再次提交原始完整D之前，使用同一D实例发出一条诊断请求。messages对象与原始D完全一致（canonical messages SHA-256 `7e093f277c14a86ed51df02256be752fe453f202b776f19d72591d68422893bd`），原请求SHA为`f25d0b8b1a0ea3e4ff37131dc90182a2940ef075f7fc1ab8fcbb7c99cd9f9d8e`；唯一修改是`max_tokens=1`，并新增`logprobs=true, top_logprobs=5`。

该请求HTTP 200，wall 101.61 s，usage为1,019,847/1/1,019,848，返回首token `RB`（logprob -0.000223），与预期答案`RB9N-6014`前缀一致。top-5为`RB`、`R`、`【`、`抱歉`、EOS。replay为1019718..1019845，8个TP worker均有128-token chunk replay和prompt-tail eager标记。完整response、top-5、请求diff、replay摘录及max1后P/D/proxy日志见`probe/ced_graph_prompt_tail_probe_20260924_090512/needle_1m_graph/needle_1m_D_max1_logprobs_*`和`probe/meta/live_logs_post_max1/`。此单token结果不代表完整答案已正确。

## 同SHA完整D重发

在上述max1诊断之后，按要求将原始`needle_1m_D_request.json`再提交一次；SHA仍为`f25d0b8b1a0ea3e4ff37131dc90182a2940ef075f7fc1ab8fcbb7c99cd9f9d8e`，参数仍为`temperature=0`、`max_tokens=64`、`stream=false`，无logprobs。HTTP 200，curl wall 224.115470 s，response SHA `81b50f134e85376ba84a480e312993002bcecad800ab7938a210116783c70059`，API ID `chatcmpl-cfd16b81-c2b3-482c-a3a2-aac60b41d89c`。返回`RB9N-6014`，与预期完全一致，finish reason为stop，usage为1,019,847/7/1,019,854，U+FFFD=0。replay位置仍为1019718..1019845，八个worker均记录chunk128 replay和prompt-tail eager。

时间顺序是：首个完整D请求同SHA返回错误`Tech-D9q7Wm`；之后插入max_tokens=1/top5诊断并得到前缀`RB`；再之后才重发同SHA完整D并得到`RB9N-6014`。两次完整D不是紧邻重复。完整日志与请求时间戳见`probe/meta/live_logs_post_full_repeat/`及A–D请求目录。不同输出和wall耗时（133.174500 s与224.115470 s）是观测结果；由于中间插入了诊断请求，不能仅据此确定输出差异的原因。

## 后续两次串行复测

在D2之后，对原始`needle_1m_D_request.json`连续提交两次；每次均保持相同的SHA `f25d0b8b1a0ea3e4ff37131dc90182a2940ef075f7fc1ab8fcbb7c99cd9f9d8e`和max64/temp0/nonstream/no-logprobs参数。两次均HTTP 200、U+FFFD=0、同replay位置`1019718..1019845`，D八个worker都有chunk128和prompt-tail eager标记。

| 顺序 | host start–end | curl wall | API ID | 实际 | usage prompt/completion/total | Response SHA |
| --- | --- | ---: | --- | --- | --- | --- |
| D3 | 10:55:17–10:56:59 | 101.673388 s | `chatcmpl-42db0df7-4ecb-4dab-9341-db0dd4858c51` | `RB9N-6014`（正确） | 1,019,847 / 7 / 1,019,854 | `606d4e77961cfc153551c00c73a895f73146409fffdaff3d06bd03793a34c935` |
| D4 | 10:59:25–11:01:06 | 101.599612 s | `chatcmpl-81d4a2f7-beca-4a35-b391-bd7be8db3a80` | `Uhq9-3DqT`（错误） | 1,019,847 / 9 / 1,019,856 | `f4a25d2645150a6f18c9532ce7d2a7b8aca892dd16a17a7fd554229e627cbba3` |

至此四次完整同SHA D（D1/D2/D3/D4）中两次正确、两次错误。D1和D2之间插入过一次max1/top5；D3和D4是连续串行请求。两次新请求原始副本、response、时间戳和逐轮完整P/D/proxy日志分别在`needle_1m_D_repeat_after_max1_a_*` / `needle_1m_D_repeat_after_max1_b_*`及`live_logs_after_repeat_a/`、`live_logs_after_repeat_b/`。四次D的输出不一致是观测事实，不由wall、KV transfer或请求顺序推断原因。

各请求的 P handoff、D transfer/replay 和资源观测汇总见[`probe/needle_1m_timeline_analysis.md`](probe/needle_1m_timeline_analysis.md)。

## Graph + prompt-tail，DSA overlap关闭

在同一个P进程下单变量关闭D的DSA overlap：`V41_CED_ROLE=decode`、`GRAPH=1`、`EAGER=0`、`V41_CED_GRAPH_PROMPT_TAIL_EAGER=1`、`DSA_OVERLAP=0`、`MULTISTREAM=1`；实际additional-config为`multistream_dsv4_dsa_overlap=false`、`multistream_overlap_shared_expert=true`。patched runner SHA=`bd250a59819dd806d16706177840c057416944c762264f2a291c608d915c2aff`，CED consumer/DSA/scheduler replay挂载和P进程均通过硬门。详见[`../graph_prompt_tail_dsaoff_20260924/DSA_OFF_1M_COMPARISON.md`](../graph_prompt_tail_dsaoff_20260924/DSA_OFF_1M_COMPARISON.md)。

请求门先后验证短针与144K A：22-token短针SHA `f670186097daa964d1eca4c436320e8dcee6dc93cda0ea41b1306224991aa881`精确返回`ZQ7K-3341`；144K A SHA `e7867fa9377bcfc66445ebde5c114b95129fa8dd9746447b2d34fad0e81368a`也精确返回`ZQ7K-3341`。其后以同一1M D原始SHA连续提交四次：前三次精确返回`RB9N-6014`，第四次HTTP 200但`message.content=null`、finish_reason=stop、completion_tokens=1。D4无token_ids/logprobs/stop_reason，也没有tool_calls、refusal、reasoning或function_call；全量日志无该API对应sampler、token ID或EOS记录，不能推断它实际输出的token。

DSA_OFF臂结果为3/4完整1M D正确，尚未稳定通过1M D。日志另外记录了`causal_conv1d_update_npu`不可用并回退同步实现的启动WARNING；P、DSA-on graph、DSA-off graph和eager臂都有相同每rank警告，因此它不是本轮模式或答案结果的独有标记。该轮结束时P/D/proxy仍running；当前配置保留，下一单变量计划由主Agent决定。

## D eager 诊断对照

按主 Agent 指示，在同一 P 原进程下重启D和proxy，保留CED decode角色、connector、DSA/scheduler replay、模型、端口和其它设置，只把D切成`GRAPH=0 EAGER=1 V41_CED_GRAPH_PROMPT_TAIL_EAGER=0`；`DSA_OVERLAP=1`、`MULTISTREAM=1`保持。实际vLLM命令含`--enforce-eager`、无`FULL_DECODE_ONLY`；prompt-tail runner patch未挂载，runner SHA回到基底`67035d97f1cea4ae2df31adcc33f1de952f4cab6d8421e76df512296e0e3185e`。CED consumer connector、DSA和scheduler replay patch均通过硬门，P container ID未变。硬门完整记录见`probe/ced_prompt_tail_d_eager_cedrole_20260924_1129/hard_gate/hard_gate_pass.txt`。

相同1M D请求（SHA `f25d0b8b1a0ea3e4ff37131dc90182a2940ef075f7fc1ab8fcbb7c99cd9f9d8e`，max64/temp0/nonstream/no-logprobs）连续提交4次，全部返回预期`RB9N-6014`，HTTP200、finish=stop、U+FFFD=0，usage均为1,019,847/7/1,019,854。wall分别为116.170616/102.116898/102.093542/102.270511秒，API ID、response SHA、usage和完整transfer/replay见`eager_4_requests_summary.json`及`probe/needle_1m_graph/needle_1m_D_eager_01_*`至`_04_*`。每条D replay位置均为1019718..1019845、8个chunk128 worker行，prompt-tail eager marker为0；这符合该臂未加载prompt-tail runner patch的配置。

此前有一次错误启动：新D容器`dsv41-ced-prompt-tail-eager-d-20260924-1109`的`V41_CED_ROLE`为空、未挂CED connector/DSA/scheduler replay，hard gate未通过。未启动proxy、未发请求，serve.log无API POST记录；该配置失误单独归档在`probe/ced_prompt_tail_d_eager_20260924_1109/`，不计精度结果。它被退出后没有reset；8–15 HBM自然回到约2.9–3.1 GiB且无holder，才启动了正确CED eager臂。

截至eager四次测试，P/D/proxy均保持运行。四次eager全对与图+tail此前四次全量D两对两错是观测对照，不能单凭此归因到某个单独机制。下一项按主 Agent 指示验证`DSA_OVERLAP=0`的图+tail臂；此后不自动切回其它配置。
