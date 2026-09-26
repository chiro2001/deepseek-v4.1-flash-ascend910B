A3-21 CED D graph-minimal token probe, same P and same request bodies as eager baseline.
P: preserved original process, chips0-7, TP8, HTTP18990, GRAPH1/EAGER0, KV producer19090.
D: fresh decoder on chips8-15, TP8, HTTP18991, GRAPH1/EAGER0; actual command FULL_DECODE_ONLY, no --enforce-eager; KV consumer19091.
Same f28cd71 package, same image ID as P, model and remaining config. CED role=decode, connector/dsa/core scheduler patch mounts are present. Graph capture decode FULL completed 4/4 before requests. P/D health200; proxy healthcheck/model list200.
Runtime SHA of breakable_cudagraph.py, device_metadata.py, acl_graph.py, model_runner_v1.py and gpu_model_runner.py is recorded for both P and D and byte-identical. Source excerpts and the local matching hashes/line references are in ../../graph_minimal/runtime_source_fingerprints.txt and runtime_source_excerpts.txt.
The exact same raw max_tokens=1 and max_tokens=2 JSON files as eager were submitted, each with logprobs=true/top_logprobs=5. Eager reference emits Z / ZQ.
Graph max1: HTTP200, prompt22/completion1/total23, content “正确答案”, finish=length, wall=4.891s; first token is “正确答案”, logprob=-2.1355386.
Graph max2: HTTP200, prompt22/completion2/total24, content “正确答案只有一个”, finish=length, wall=0.366s; tokens are “正确答案” then “只有一个”.
Both graph D replay ranges are positions=0..20 followed by replay chunk=21 on all 8 TP workers. Raw requests, responses, token/logprob arrays, P/D/proxy full logs and inspect files are saved here.
No additional model requests were sent after max_tokens=2.
