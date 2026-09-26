A3-21 CED eager correctness diagnostic, including 1M A-D and eager token-sequence baseline
Date: 2026-09-24 Asia/Shanghai.
The original P process was preserved. D was freshly restarted from f28cd71 with GRAPH=0 EAGER=1; same P/D eager instances and proxy handled all requests below. This is diagnostic/correctness evidence, not a delivery configuration.
P chips 0-7 / TP8 / port 18990 / graph1-eager0 / KV producer 19090.
D chips 8-15 / TP8 / port 18991 / graph0-eager1 / KV consumer 19091; actual command has --enforce-eager and no FULL_DECODE_ONLY.
Proxy 127.0.0.1:18992. Model /home/l00886679/models/out/v41-flat-verify3.
Shared: BF16 KV, MAX_LEN=1048576, MAX_SEQS=4, BAT_TOKENS=8192, GPU_UTIL=.92, ENGRAM=1/DEVICE_INDEX=0, CPU_BIND=0, SPEC/PREFIX/DRAFT_GRAPH=0, STATIC_KERNEL=0, NPUGRAPH_EX=1.

Seven correctness generation requests were submitted, all HTTP200 and exact, U+FFFD=0:
- 22-token short: ZQ7K-3341, usage 22/8/30.
- 144K bigprefill A: body 144404, offset 1,000,000, ZQ7K-3341, usage 144462/8/144470, wall 17.673s.
- 144K bigprefill B: body 144131, offset 1,146,000, VX2M-8890, usage 144188/8/144196, wall 11.953s.
- 1M needle A/B/C/D: HLM corpus SHA a7fc413bd6e3926482faddf2af9bfb4426e55d8e421481d60100c14160785578; same body tokenizer=1,019,789, offset=0; answers ZQ7K-3341 / VX2M-8890 / HT4P-5527 / RB9N-6014. All temp=0, max_tokens=64, stream=false. API usage prompt/completion/total: A 1019847/8/1019855; B 1019846/8/1019854; C 1019848/7/1019855; D 1019847/7/1019854. Wall: A 105.860s, B 104.815s, C 104.044s, D 103.048s.
This eager CED path passed these seven requests. A same-model direct full40 scan had previously failed at 1M, so this result is bounded to the current CED eager run and does not support general attribution.

Eager short-token diagnostic then sent two more POSTs with logprobs=true, top_logprobs=5:
- max_tokens=1 returned "Z"; prompt/completion/total=22/1/23, wall=1.228s; top token Z logprob -1.19e-7.
- max_tokens=2 returned "ZQ"; prompt/completion/total=22/2/24, wall=0.584s; token sequence Z then Q, Q logprob 0.0.
Both had U+FFFD=0 and D replay positions 0..20/chunk21. The API accepted logprobs, so no fallback request was sent. No-token diagnostic request hashes and token arrays are retained.

D replay for correctness requests: short positions 0..20/chunk21; 144K A 144333..144460/chunk128; B 144059..144186/chunk128; 1M A 1019718..1019845, B 1019717..1019844, C 1019719..1019846, D 1019718..1019845; each long needle replay chunk=128. All D TP workers logged chunk reuse.
At this snapshot nine generation POSTs were sent total (seven correctness + two token diagnostics). Raw request/response bodies, usage, wall, complete P/D/proxy logs, inspect and hashes are retained. No stream/multi-turn test was sent.
