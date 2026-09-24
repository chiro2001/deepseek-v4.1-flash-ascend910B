Live log snapshot for the 1M A-D graph/prompt-tail run, captured from A3-21 after A-D completed and before the planned max_tokens=1 diagnostic.

P_serve.log source:
/home/l00886679/tmp/20260924/ced_numeric/pkg_f28cd71/src/results/ced_graph_ab_p_20260924_070428/serve.log

D_serve.log source:
/home/l00886679/tmp/20260924/ced_numeric/pkg_777bc73/src/results/ced_prompt_tail_d_20260924_090512/serve.log

proxy.log source:
/home/l00886679/tmp/20260924/ced_numeric/pkg_777bc73/src/results/ced_graph_prompt_tail_20260924_090512/probe/meta/proxy.log

P/D logs are the complete launcher-side serve.log files, not `docker logs` output. Their source sizes were 231177 and 177831 bytes. Proxy log size was 1222 bytes. A hash manifest at the parent graph_prompt_tail/SHA256SUMS covers these snapshots.

Read-only scan of P_serve.log and D_serve.log for ERROR, Traceback, OutOfMemoryError, OOM, and RuntimeError returned no matches. Startup lines are retained verbatim in the full logs; they report 3,322,350 GPU KV-cache tokens on P and D. D worker Available/Current KV-cache memory lines are approximately 15.15–15.17 GiB. These are recorded as emitted; no capacity change is inferred from values with potentially different log scopes.
