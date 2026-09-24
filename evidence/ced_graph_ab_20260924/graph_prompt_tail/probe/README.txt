Prompt-tail patch graph-mode diagnostic on A3-21, 2026-09-24.
P remained unchanged (chips0-7 TP8 GRAPH1/EAGER0). D uses chips8-15 TP8, package fix/ced-graph-prompt-tail@777bc73, V41_CED_GRAPH_PROMPT_TAIL_EAGER=1, GRAPH=1, EAGER=0, CED decode role; prompt-tail patch was applied after verifying base runner SHA. D command contains FULL_DECODE_ONLY and no --enforce-eager. D image matches P. BF16 KV, max_len1M, Engram index0, CPU_BIND0, SPEC/PREFIX/DRAFT_GRAPH=0, static_kernel0; same CED transfer/replay modules.

Five exact raw request JSONs were submitted sequentially:
- max1 SHA bd6276adca2e71471858eb7402972561c1d543f39db67f76c3a706a38e6e2ff5 -> Z, usage22/1/23, wall4.599s.
- max2 SHA 403e6cb2b9f26f1cd6adc8fc7ac21af1bd265769fa42fab2f6f2ccfce021378a -> ZQ, usage22/2/24, wall0.536s.
- max32 SHA f670186097daa964d1eca4c436320e8dcee6dc93cda0ea41b1306224991aa881 -> exact ZQ7K-3341, usage22/8/30, wall0.586s.
- Long A SHA e7867fa9377bcfc66445ebde5c114b95129fa8dd9746447b2d34fad0e81368a -> exact ZQ7K-3341, body tokenizer144404, usage144462/8/144470, wall10.561s.
- Long B SHA ece2ab107f8934d13f25538295a435fd16fe1a176701da112e6eca2fa9ea85c6 -> exact VX2M-8890, body tokenizer144131, usage144188/8/144196, wall10.609s.
All HTTP200, temperature0, nonstream, U+FFFD=0. D logs contain the forced-eager marker on all eight workers for each short request. After max2, a real Replaying aclgraph entry is present. D replay is positions0..20/chunk21 for short, positions144333..144460/chunk128 for A, positions144059..144186/chunk128 for B; all D workers report chunk reuse.
D patched model_runner_v1.py SHA is bd250a59819dd806d16706177840c057416944c762264f2a291c608d915c2aff, based on 67035d97f1cea4ae2df31adcc33f1de952f4cab6d8421e76df512296e0e3185e. Other runtime files match P. Full raw requests/responses, usage, logprobs, replay and P/D/proxy logs/inspect are saved.
No other requests were sent.
