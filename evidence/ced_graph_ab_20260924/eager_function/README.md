# D eager 正确性诊断与首/次 token 基线

日期：2026-09-24。保留 D 图/eager 对照中的同一个 P 进程；D 为 GRAPH=0 EAGER=1。该 eager 运行用于正确性诊断，不作为交付配置。P 为 GRAPH=1 EAGER=0。BF16 KV、MAX_LEN=1M、Engram index0、CPU_BIND0、SPEC/PREFIX/DRAFT_GRAPH=0；chips0–7为P，8–15为D。A3-22未使用。

短针和 144K A/B 均精确通过。144K A 用 HLM 语料 SHA a7fc413bd6e3926482faddf2af9bfb4426e55d8e421481d60100c14160785578、rotation offset 1,000,000，body tokenizer=144404，usage prompt/completion/total=144462/8/144470，wall=17.673s。B 用同语料、offset=1,146,000，body tokenizer=144131，usage=144188/8/144196，wall=11.953s。

同一 P/D eager 实例再按offset=0运行标准1M needle A/B/C/D。HLM corpus SHA相同，body tokens均1,019,789，temperature0、max_tokens64、stream=false。四条均HTTP200、精确匹配且U+FFFD=0：
- A ZQ7K-3341，usage 1019847/8/1019855，wall 105.860s；
- B VX2M-8890，usage 1019846/8/1019854，wall 104.815s；
- C HT4P-5527，usage 1019848/7/1019855，wall 104.044s；
- D RB9N-6014，usage 1019847/7/1019854，wall 103.048s。

随后同一22-token短针作首/次 token eager 基线，API接受logprobs=true/top_logprobs=5，无需fallback：
- max_tokens=1返回 Z，usage prompt/completion/total=22/1/23，wall1.228s，token Z logprob≈-1.19e-7；
- max_tokens=2返回 ZQ，usage=22/2/24，wall0.584s，token序列 Z → Q，Q logprob=0.0。
两条D replay均为positions=0..20、chunk21。原始输入/输出、logprobs和replay均在eager_token_probe子目录。

D replay范围：短针0..20/chunk21；144K A 144333..144460/chunk128；B 144059..144186/chunk128；1M A 1019718..1019845、B 1019717..1019844、C 1019719..1019846、D 1019718..1019845，均chunk128。所有D worker均记录reuse global source/C2 ring。

同模型、同长上下文的direct full40历史扫描曾失败；因此这里的四针通过只适用于当前CED eager实例，不能当作D graph正确性的证明。没有发送流式或多轮请求。原始请求/响应、完整P/D/proxy日志、inspect、启动与代码/语料SHA均在本目录；eager_function/SHA256SUMS和父级SHA256SUMS校验文件完整性。
