# A3-21 CED D 图模式 / Eager 精度对照

日期：2026-09-24（A3-21，单机 TP8+TP8）

## 对照条件

使用相同 f28cd71 包、同一模型目录、同一个 P 进程、相同 CED P/D 角色、BF16 KV、1M MAX_LEN、Engram host（ENGRAM_DEVICE_INDEX=0）、CPU_BIND=0、SPEC=0、PREFIX=0、DRAFT_GRAPH=0、STATIC_KERNEL=0、MAX_SEQS=4、BAT_TOKENS=8192。P 始终保持 GRAPH=1 EAGER=0。两臂只改变 D 的 Graph/Eager：

| D 臂 | 环境 | 实际 vLLM 参数 |
| --- | --- | --- |
| eager | GRAPH=0 EAGER=1 | 有 --enforce-eager，无 FULL_DECODE_ONLY |
| graph | GRAPH=1 EAGER=0 | 有 FULL_DECODE_ONLY，无 --enforce-eager |

模型：/home/l00886679/models/out/v41-flat-verify3；served model ID：deepseek-v41-ced-pd。
P 使用 chips 0–7 / HTTP 18990 / KV producer 19090；D 使用 chips 8–15 / HTTP 18991 / KV consumer 19091；proxy 使用 HTTP 18992。A3-22 的芯片未使用。

## 短针结果

两臂都从同一个原始 JSON 文件提交两次；SHA-256：
f670186097daa964d1eca4c436320e8dcee6dc93cda0ea41b1306224991aa881。
请求内容是22-token校验码针，temperature=0.0、max_tokens=32、stream=false。

- D eager：两次 HTTP 200，均精确返回 ZQ7K-3341，finish reason=stop，usage 均为 prompt/completion/total 22/8/30，U+FFFD=0。
- D graph：两次 HTTP 200，均未返回校验码；finish reason 均为 length、completion_tokens=32。第一次是混杂文本且 U+FFFD=7；第二次为多语混杂文本且 U+FFFD=0。

P 没有重启；D eager 证据归档完成后只停止 proxy 与 D，再仅改 D 的 Graph/Eager 参数重启。D graph 日志确认 Capturing CUDA graphs (decode, FULL) 完成4/4。两臂 D 日志均记录 replay 范围 positions=0..20，并在所有 TP worker 上执行 replay chunk=21。

结果显示该受控短针下，D graph 两次均失败、D eager 两次均通过；这是两次重复观察，尚不构成对任意上下文/请求的普遍结论。没有发送长上下文请求或第三条 generation 请求。

## 证据结构

- eager/：P 原实例、D eager 容器及两条短针的完整 serve/proxy 日志、inner.sh、原始请求/响应、container inspect 和 SHA 清单。
- graph/：同一个 P 实例在第二臂结束时的日志快照、D graph 容器及两条短针的完整 serve/proxy 日志、inner.sh、原始请求/响应、container inspect 和 SHA 清单。
- f28cd71 包 SHA-256：8bb7a846e70c036cc681a56dd968462e88ebd5d1575afbfa8229885dcb640269。

分别在 eager/ 和 graph/ 下有该轮完整文件 SHA 清单（meta/eager_run.sha256、meta/graph_run.sha256）。

## Prompt-tail eager补丁后续验证（777bc73）

后续保持同一P进程，在D端验证了`fix/ced-graph-prompt-tail@777bc73`：短针和144K A/B全部精确通过；真实1M A/B/C通过。四次完整同SHA D中两次正确、两次错误：首个D返回`Tech-D9q7Wm`；max1/top5诊断返回正确前缀`RB`后，第一条完整重发正确为`RB9N-6014`；再连续两次全量复测一对一错，先正确`RB9N-6014`、后错误`Uhq9-3DqT`。D原始请求正文不含首个错误串。D1和D2之间插入max1诊断，不能把四次当成同状态紧邻重复，也不能据此确定原因。所有原始请求/响应、replay、日志、SHA与启动条件见[`graph_prompt_tail/README.md`](graph_prompt_tail/README.md)。

max1/top5、四次graph+tail全量D、eager硬门及四次eager D的分段时序、P/D/proxy日志、KV/HBM和负载采样边界已归档，详见[`graph_prompt_tail/probe/needle_1m_timeline_analysis.md`](graph_prompt_tail/probe/needle_1m_timeline_analysis.md)。eager诊断臂同SHA 4/4正确。随后graph+tail单变量关闭DSA overlap：短针、144K A均通过，1M D为3/4正确；D4的`content`为null但无token ID、sampler或EOS日志，不能据此推断具体token。启动中的causal_conv1d_update_npu fallback WARNING也出现在其他各臂，不是该臂特有。详细证据见[`graph_prompt_tail_dsaoff_20260924/DSA_OFF_1M_COMPARISON.md`](graph_prompt_tail_dsaoff_20260924/DSA_OFF_1M_COMPARISON.md)。当前DSA_OFF graph+tail P/D/proxy仍运行。下一配置diff拟只把`MULTISTREAM=1`改为`0`、保持DSA overlap关闭及其余参数不变；在主Agent确认diff前不切服务。
