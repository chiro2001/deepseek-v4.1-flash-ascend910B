Complete P/D/proxy serve.log snapshot at 2026-09-24 10:30:38 +08, after the single same-SHA full-D re-send and before the later repeated-D tests.

P_serve.log source: /home/l00886679/tmp/20260924/ced_numeric/pkg_f28cd71/src/results/ced_graph_ab_p_20260924_070428/serve.log
D_serve.log source: /home/l00886679/tmp/20260924/ced_numeric/pkg_777bc73/src/results/ced_prompt_tail_d_20260924_090512/serve.log
proxy.log source: /home/l00886679/tmp/20260924/ced_numeric/pkg_777bc73/src/results/ced_graph_prompt_tail_20260924_090512/probe/meta/proxy.log

P/D files are complete launcher-side serve.log files, not `docker logs` output. Snapshot sizes are P 252127 bytes, D 187927 bytes, proxy 1222 bytes. D log contains replay request `chatcmpl-cfd16b81-c2b3-482c-a3a2-aac60b41d89c` at positions 1019718..1019845, with 8 replay-chunk lines and 8 prompt-tail eager worker markers. ERROR, Traceback, OutOfMemoryError, OOM, and RuntimeError scans returned zero matches in the full P/D logs.

At 2026-09-24 10:30:38 +08, P/D/proxy were still running and listeners 18990/18991/18992/19090/19091 were present. See service_state_post_full_repeat.txt.
