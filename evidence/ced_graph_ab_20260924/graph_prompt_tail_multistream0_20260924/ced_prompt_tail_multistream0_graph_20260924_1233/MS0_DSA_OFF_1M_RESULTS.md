# A3-21 CED PD graph prompt-tail: MULTISTREAM=0 / DSA overlap off

This is a one-variable comparison against the preceding DSA-OFF graph arm: only `MULTISTREAM` changed from 1 to 0. `DSA_OVERLAP=0`, `GRAPH=1`, `EAGER=0`, `V41_CED_GRAPH_PROMPT_TAIL_EAGER=1`, CED decode role, TP8/DP1, BF16 KV, model/image, Engram index0, CPU_BIND0, Mooncake consumer, ports18991/19091 and the other multistream flags stayed fixed. The actual D command has `FULL_DECODE_ONLY`, no `--enforce-eager`; additional config has `multistream_overlap_shared_expert=false`, `multistream_dsv4_dsa_overlap=false`. Runner SHA is `bd250a59819dd806d16706177840c057416944c762264f2a291c608d915c2aff`.

The P container stayed at ID `0db3d62627d8680aa17e0398e1f26cec8b1a654bdc6724ed247ec1235e06739d` on chips0–7. D used chips8–15. The DSA scheduler replay, DSA attention, Mooncake connector and prompt-tail runner patches were checked; all four D tests saw 8 KV transfers, replay range `1019718..1019845`, eight chunk128 markers and eight forced-eager tail markers. P's per-request handoff lines matched all four 1M API IDs. HTTP and curl succeeded for all requests.

| Request | API ID | Wall | Output | Prompt/completion/total | Response SHA |
|---|---|---:|---|---|---|
| short32 | `chatcmpl-a59cbc4e-cd8d-4686-82a3-caacbf8baa89` | 4.533112s | `ZQ7K-3341` | 22/8/30 | `f166526a64730062e9518c77a707692b4203ca7cd93b266321e3fe9d5b2fdd4a` |
| 144K A | `chatcmpl-d0d9e558-5a3a-4d6e-8acb-5528a2a874d2` | 10.386876s | `ZQ7K-3341` | 144462/8/144470 | `fcf226d08faf35f2f6cb5285db836dd1c32c17b212dfd7766f20df66e3d83479` |
| 1M D1 | `chatcmpl-23d2ff51-8582-4853-b45a-a1d197496624` | 101.758406s | `RB9N-6014` | 1019847/7/1019854 | `630eee905f8f6d5a35a2918b16a5e573fe1ab21e70313bec1103f386c45625fe` |
| 1M D2 | `chatcmpl-65268fc1-4b0c-4fdb-976a-be58feee574a` | 101.519376s | `RB9N-6014` | 1019847/7/1019854 | `b57898e952fb87b88446d4372d2e1f48fed5d25f0438bbe93c102c0b0be0ca30` |
| 1M D3 | `chatcmpl-54fb042e-1a1a-4b84-b1f6-ae622c6a2750` | 102.811735s | `RB9N-6014` | 1019847/7/1019854 | `a1fcdb76bab9e5afa52079c72ba54cae81d901b673359901c8c702ef323b1690` |
| 1M D4 | `chatcmpl-5795cc05-1fe8-4e3c-bb9d-515966d76e30` | 101.058691s | `content=null` | 1019847/1/1019848 | `a11bc8067f32f258608022829ebdc7ba6ae31244a5cb0acd6dc43f6c9dd2ee02` |

All four 1M request copies have the original D SHA `f25d0b8b1a0ea3e4ff37131dc90182a2940ef075f7fc1ab8fcbb7c99cd9f9d8e`; parameters were max_tokens64, temperature0, stream=false, no logprobs. All responses were HTTP200 with U+FFFD=0. D4's response fields and the full P/D/proxy logs are preserved. Error/OOM/Traceback/RuntimeError scans of the final P/D/proxy snapshots found zero matches. Final service state is in `live_logs_final_ms0_1m4/`.

The proxy example has no `/health` route (404); `/openapi.json` and `/docs` return 200 and its startup log says application startup complete. Request-output differences are reported as observations; wall time is not used to explain accuracy.
