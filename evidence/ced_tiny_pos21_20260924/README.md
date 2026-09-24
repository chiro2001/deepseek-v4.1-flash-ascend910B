# Tiny CED 22-token pos20/pos21 A/B

Date: 2026-09-24. This is an A3-22 mechanism check for the CED replay-to-final-token
boundary. It uses dummy `model-tiny` weights, TP1, BF16 KV, Engram off, DSpark
off, seed 0, prefix caching off, and one generated token. It cannot establish
real-weight semantic accuracy.

## Resources and run shape

- Phy-ID 0 and 1 were not touched.
- Reused the already-running P on Phy-ID 6 (`dsv41-ced-tiny-p-b6ca283`, port
  18960); it was not restarted. D and full40 baseline ran sequentially on
  Phy-ID 7. The proxy used host networking and no NPU.
- Both comparison arms ran with `GRAPH=0 EAGER=1`. Actual `serve.log` commands
  contain `--enforce-eager` and no `--compilation-config`; engine config reports
  `enforce_eager=True`, `CUDAGraphMode.NONE`.
- The isolated run package was copied from `pkg-snapshot-2df46ec` to
  `pkg-pos21-diag-20260924`. Only that package received the diagnostic edits.
  The P package and the 8+8 worktree were not changed.
- At test end, the D, proxy, and full40 baseline containers were stopped and
  removed. P remained healthy on port 18960; baseline port 18963 was closed,
  and the Phy-ID 7 process table was empty. Phy-ID 0 and 1 remained untouched.

The input is the first 22 IDs of the repeated 8-token pattern
`28669,6441,58603,693,85450,84483,22089,320`. Each arm received the same
non-streaming `/v1/completions` payload twice: temperature 0, seed 0,
`max_tokens=1`, `logprobs=20`.

## Observed execution boundary

For each D request, the patched scheduler logged
`prompt=22 loaded=21 declared=21 replay=128 positions=0..20`. The attention
trace then reported:

1. Replay: `q_positions=0..20`, `seq_lens=[21]`,
   `num_prefills=1`, `num_prefill_tokens=21`, `max_query_len=21`.
2. Uncached prompt tail: `q_positions=[21]`, `seq_lens=[22]`,
   `num_prefills=1`, `num_prefill_tokens=1`, `num_decodes=0`,
   `max_query_len=1`.

So in this build, the one-token tail is still counted as a prefill batch; it is
not reported as a decode batch. The loaded/declared count proves that the P→D
handoff contained exactly N−1=21 tokens. P's own admission-gate log also records
one prefill-only episode for each request.

The full40 baseline processed one batch with `q_positions=0..21`,
`seq_lens=[22]`, `num_prefills=1`, `num_prefill_tokens=22`, and
`max_query_len=22`.

## Results

| Comparison | Result |
|---|---|
| D same-service repeat, pos20 | 43/43 cache planes exact; max absolute delta 0 |
| D same-service repeat, pos21 | 43/43 cache planes exact; max absolute delta 0 |
| Full40 same-service repeat, pos20 | 43/43 cache planes exact; max absolute delta 0 |
| Full40 same-service repeat, pos21 | 43/43 cache planes exact; max absolute delta 0 |
| D vs full40, pos20, all four repeat pairs | 43/43 exact in every pair; max absolute delta 0 |
| D vs full40, pos21, all four repeat pairs | 43/43 exact in every pair; max absolute delta 0 |
| API, both requests per arm | HTTP 200; selected token and logprob match; top-20 sets 20/20; maximum common logprob delta 0 |

The selected dummy output was ` archaeological`, with logprob
`-11.769623756408691`, on both arms. The compared cache rows include all 40
layers' SWA rows and layer 20's `long_kv`, `index_k`, and `index_scale`.
There is no first D-vs-full40 cache difference above the measured noise floor;
both same-arm noise floors were exactly zero.

The first D call took 0.45 s and the first full40 call 11.35 s; the second call
per arm took about 0.45 s. These are diagnostic request timings only and are
not performance comparisons.

## Interpretation

With eager dispatch, this 22-token tiny case shows no cache or output divergence
between CED and full40, including the N−1 replay and pos21 tail. This does not
explain the real-weight short-needle failure. Combined with the separate A3-21
real-weight observation that eager passes while FULL_DECODE_ONLY graph mode
fails, this motivated the tiny GRAPH1 D-only check below. Python snapshots were
not required for that graph arm.

## Tiny D GRAPH1 follow-on

After the eager A/B was complete and the full40 baseline stopped, only D was
restarted on the same Phy-ID 7, reusing the same P on Phy-ID 6. The command set
`GRAPH=1 EAGER=0`, kept the same capture-size list, and did not enable the
Python snapshot/batch-trace hooks. Engine config confirms
`enforce_eager=False`, `CUDAGraphMode.FULL_DECODE_ONLY`; graph capture completed
4/4 decode sizes (the sizes above MAX_SEQS=4 were filtered). All six requests
logged the same D boundary, `prompt=22 loaded=21 declared=21 positions=0..20`,
and returned HTTP 200.

| max_tokens | Repeats | Result |
|---:|---:|---|
| 1 | 2/2 | ` archaeological`; both exactly match the earlier eager D and full40 output/top-20 |
| 2 | 2/2 | ` archaeological确定性`; selected tokens, selected logprobs, and top-20 values repeat exactly |
| 8 | 2/2 | ` archaeological确定性` repeated four times; selected tokens/logprobs and top-20 sets repeat; alternative top-20 logprobs differ by at most `9.54e-7` at positions 0 and 3 |

The D service log contains one explicit `Replaying aclgraph` line and six CED
replay boundary lines. This build does not log one ACL replay line per request,
so that count cannot be used as a per-request replay counter. The max_tokens=1
graph result is bitwise equal at API level to the earlier eager D/full40 result.
The graph arm did not reproduce the failure on this tiny dummy model; no
full40 graph arm was run. This does not validate real-weight graph correctness.

## Eager / graph profiler follow-on

After the above runs, D was run once more with on-demand profiler enabled on
the same `/dev/davinci7`; the eager profiler sample had already run on that
device. One warmup M1 and measured M1/M8 windows were serialized and matched by
request ID to `serve.log`. The detailed per-window event and stream comparison,
limitations, and raw archive checksums are in
`PROFILE_WINDOW_ANALYSIS.md`. The raw graph trace and controls remain in private
COS objects referenced by `PROFILE_ARCHIVES.md`; they were not added to Git.

After archiving and verifying the trace, the graph D and its proxy were stopped
and removed. P remains running on `/dev/davinci6`. The post-cleanup device and
container summary is `prof_graph/release_status_summary.md`; the D container no
longer appears in `docker ps`, and no process is listed for Phy-ID 7 (with
2,904 MB HBM still reported there).

Quick index for the eager/graph profiler comparison: `PROFILER_README.md`.

No Python snapshot is available for the graph arm because graph execution does
not run that hook. The eager arm's snapshot comparison remains the cache A/B.

The first request attempt returned 404 because the probe was given a base URL
ending in `/v1` while it appends `/v1/completions` itself. The proxy logged
`/v1/v1/completions`; it never reached P or D and is retained as
`d_response_wrong_base_url_404.json`. It is excluded from all results.

The launcher's first `say` calls printed `tee: .../driver.log: No such file or
directory` before it created the per-run output directory. The run proceeded;
`inner.sh`, `serve_cmd.txt`, `serve.log`, snapshots, and container inspection
are present and record the actual configuration.

## Files

- `d_response_r*.json`, `base_response_r*.json`: raw probe responses.
- `d_snapshots_r*`, `base_snapshots_r*`: 40-layer NPZ snapshots at positions
  20 and 21, 80 files per arm per repeat.
- `d*_vs_base*_pos*.json`: all four repeat-pair cache comparisons for each
  position; `*_repeat_noise_pos*.json`: same-arm noise comparisons.
- `api_d_vs_base_r*.json`: API top-20 comparisons.
- `d_serve.log`, `base_serve.log`, `p_relevant.log`: runtime trace evidence.
- `d_inner.sh`, `base_inner.sh`, `*_serve_cmd.txt`, and
  `*_container_inspect.json`: startup arguments and device mappings.
- `npu_smi_after_release.txt`, `containers_after_release.txt`: end-of-run
  resource state; `SHA256SUMS` covers the local evidence directory.
- `graph_boundary_probe.py`: deterministic client used for the follow-on D-only
  graph arm; `graph_d/` contains raw responses, command/configuration, graph
  capture and replay logs, proxy logs, and remote checksums.
- `CODE_PATCH.diff`: isolated diagnostics plus the `f28cd71` GRAPH/EAGER
  pass-through change; `SHA256SUMS`: local evidence integrity manifest.
- `d_sha256sums.txt`, `base_sha256sums.txt`: source/log/snapshot hashes
  recorded on A3-22 before transfer.
