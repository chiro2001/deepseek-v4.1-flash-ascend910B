# A3-21 CED D metadata-inline pre-D5 evidence

Date: 2026-09-24. This is the isolated `fix/ced-metadata-inline@d7953e5` D arm. No request was sent until the package, runtime hashes, FULL decode graph, 8 worker metadata markers, P/D health, and proxy health gates passed.

## Configuration and runtime gates

- P remained unchanged: container `0db3d62627d8680aa17e0398e1f26cec8b1a654bdc6724ed247ec1235e06739d`, chips 0–7, port 18990/KV 19090.
- Inline D: container `c8646177ad2c58ccf0b067da30852e37a2419835ca9f0aa175f6c390388fee57`, chips 8–15, port 18991/KV 19091. Proxy: `36a03ef67c7349ac2f09aa8b0baab2d15bc26f93e3443a5a33864181fc8f9ae5`, port 18992.
- `GRAPH=1 EAGER=0`, `V41_CED_ROLE=decode`, `V41_CED_GRAPH_PROMPT_TAIL_EAGER=1`, `V41_CED_METADATA_INLINE=1`, `MULTISTREAM=0`, `DSA_OVERLAP=0`, BF16 KV, 1M max length, 8192 batched tokens, Engram device index 0, CPU_BIND=0. D command used `FULL_DECODE_ONLY`, had no `--enforce-eager`, and set both `multistream_overlap_shared_expert=false` and `multistream_dsv4_dsa_overlap=false`.
- Container DSA module SHA-256 `72788c494bfbb12510ad36687e8578ce1ab85b87220ba72ea4b30cac8de381ac`; prompt-tail runner SHA-256 `bd250a59819dd806d16706177840c057416944c762264f2a291c608d915c2aff`; CED connector SHA-256 `a93bd6054b349647f5650bbadc9b60d6596d0d992f4c0574398fe12b8c3e3dfe`.
- Decode graph capture completed 4/4. `[CED-META] inline` appeared 200 times across TP0–TP7. Prompt-tail eager marker appeared 48 times across TP0–TP7 after the six requests. Six replay records all covered the expected final 128-token range. P/D health, proxy OpenAPI/docs, and model identity remained valid. D log scan found no Traceback, OOM, RuntimeError, `[ERROR]`, or FATAL matches.

## Requests

All requests were non-streaming, temperature 0, no logprobs. The 1M input was copied from the MS0 source and verified before each send: SHA-256 `f25d0b8b1a0ea3e4ff37131dc90182a2940ef075f7fc1ab8fcbb7c99cd9f9d8e`.

| Request | Prompt / completion / total | Wall time | HTTP | Output | Response SHA-256 |
|---|---:|---:|---:|---|---|
| short22 | 22 / 8 / 30 | 4.611694 s | 200 | `ZQ7K-3341`, exact | `146ec76c40f0b1c6746cb2844d17dcefba676a0199cc94ce2dc5123559710d72` |
| 144K A | 144462 / 8 / 144470 | 10.369423 s | 200 | `ZQ7K-3341`, exact | `7ebbf19e6c993f08964941a5e20472b0506904fbf465447884fa87ed36c66382` |
| 1M D1 | 1019847 / 7 / 1019854 | 101.492258 s | 200 | `RB9N-6014` | `1921a59f26ee94a5b9a274c339bad428c11d08aebeea2868fe11c6e6b1c6d653` |
| 1M D2 | 1019847 / 7 / 1019854 | 101.542457 s | 200 | `RB9N-6014` | `e0f88c96bdfb6b93202051073149d03bb6797492079985809fb6ce83b776ddbb` |
| 1M D3 | 1019847 / 7 / 1019854 | 101.573675 s | 200 | `RB9N-6014` | `e0d13428c4ac183ba30b15cfe79ec11510b3c871e9b282d225a878ecac3ba5cc` |
| 1M D4 | 1019847 / 1 / 1019848 | 101.142135 s | 200 | `content=null`; full `choices` fields saved separately | `0925b739126e20e7fe8239091d421d2ec51a64cfe5ea6e361fd53361b2ef141e` |

U+FFFD count was zero in all six responses. The D1–D4 correctness sample is 3/4, with D4 returning null content. The preceding MS0 DSA-off arm also returned 3/4 in its four same-SHA 1M requests, including null content on D4. This sample shows no observed improvement from metadata-inline; it does not establish the root cause or general rate.

Each test has its original request/response, hashes, headers, transport status, start/end times, and parsed usage summary. Short22/144K snapshots contain proxy and service-state records; their first D/P Docker-log captures were empty because vLLM writes to host-mounted `serve.log`. The final full host logs in `live_final_20260924/` contain both requests and their handoffs/replays. D1–D4 per-request snapshots contain the host D/P serve logs, matching API IDs, and replay excerpts. D4's complete `choices` array is in `D4_choices_full.json`.

At this checkpoint P, inline D, and proxy are still running. D5/D6 are not included in this archive.
