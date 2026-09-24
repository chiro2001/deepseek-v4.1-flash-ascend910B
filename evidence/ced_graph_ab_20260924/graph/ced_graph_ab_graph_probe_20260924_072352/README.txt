A3-21 CED P/D D-graph short needle run
Date: 2026-09-24 Asia/Shanghai
Package: f28cd71; verified archive sha256 8bb7a846e70c036cc681a56dd968462e88ebd5d1575afbfa8229885dcb640269.
Model: /home/l00886679/models/out/v41-flat-verify3
P: chips 0-7, TP8, port 18990, KV producer 19090; same original P process as eager run; GRAPH=1 EAGER=0.
D: chips 8-15, TP8, port 18991, KV consumer 19091; GRAPH=1 EAGER=0; serve command has FULL_DECODE_ONLY and no --enforce-eager; CUDAGraph capture completed 4/4.
Shared settings: MAX_LEN=1048576, MAX_SEQS=4, BAT_TOKENS=8192, GPU_UTIL=.92, BF16 KV, ENGRAM=1/DEVICE_INDEX=0, CPU_BIND=0, SPEC=0, PREFIX=0, DRAFT_GRAPH=0, STATIC_KERNEL=0.
Request: archived 22-token code needle, temperature=0, max_tokens=32, stream=false. Request SHA256 f670186097daa964d1eca4c436320e8dcee6dc93cda0ea41b1306224991aa881.
Exactly two POST requests. Both HTTP 200, prompt_tokens=22, completion_tokens=32, finish_reason=length. Response 1 is mixed text with U+FFFD=7; response 2 is mixed text with U+FFFD=0. Neither contains the requested code.
D replay both times: positions 0..20, then replay chunk 21 reusing global source and C2 ring.
No long-context request and no third generation request was sent.
