# A3-21 CED D metadata-inline final evidence

Date: 2026-09-24. This is the isolated `fix/ced-metadata-inline@d7953e5` D arm. Tests used the same inline D/P/proxy processes from short22 through 1M D6; no D7 was sent.

## Configuration and runtime gates

- P remained unchanged: container `0db3d62627d8680aa17e0398e1f26cec8b1a654bdc6724ed247ec1235e06739d`, chips 0–7, port 18990/KV 19090.
- Inline D: container `c8646177ad2c58ccf0b067da30852e37a2419835ca9f0aa175f6c390388fee57`, chips 8–15, port 18991/KV 19091. Proxy: `36a03ef67c7349ac2f09aa8b0baab2d15bc26f93e3443a5a33864181fc8f9ae5`, port 18992.
- `GRAPH=1 EAGER=0`, `V41_CED_ROLE=decode`, `V41_CED_GRAPH_PROMPT_TAIL_EAGER=1`, `V41_CED_METADATA_INLINE=1`, `MULTISTREAM=0`, `DSA_OVERLAP=0`, BF16 KV, 1M max length, 8192 batched tokens, Engram device index 0, CPU_BIND=0. D command used `FULL_DECODE_ONLY`, had no `--enforce-eager`, and set both `multistream_overlap_shared_expert=false` and `multistream_dsv4_dsa_overlap=false`.
- Container DSA module SHA-256 `72788c494bfbb12510ad36687e8578ce1ab85b87220ba72ea4b30cac8de381ac`; prompt-tail runner SHA-256 `bd250a59819dd806d16706177840c057416944c762264f2a291c608d915c2aff`; CED connector SHA-256 `a93bd6054b349647f5650bbadc9b60d6596d0d992f4c0574398fe12b8c3e3dfe`.
- Decode graph capture completed 4/4. `[CED-META] inline` appeared 200 times across TP0–TP7. Prompt-tail eager marker appeared 64 times across TP0–TP7 after eight model requests. Eight replay records were captured: one for short22, one for 144K, and six over positions `1019718..1019845` (128 tokens) for 1M D1–D6. P host log contains each API ID; proxy logged eight successful completions. D/P/proxy remained healthy through D6. D log scan found no Traceback, OOM, RuntimeError, `[ERROR]`, or FATAL matches.

## Requests

All requests were non-streaming, temperature 0, no logprobs. Each 1M request copied the same MS0 raw request and verified SHA-256 `f25d0b8b1a0ea3e4ff37131dc90182a2940ef075f7fc1ab8fcbb7c99cd9f9d8e` immediately before sending.

| Request | Prompt / completion / total | Wall time | HTTP | Output | Response SHA-256 |
|---|---:|---:|---:|---|---|
| short22 | 22 / 8 / 30 | 4.611694 s | 200 | `ZQ7K-3341`, exact | `146ec76c40f0b1c6746cb2844d17dcefba676a0199cc94ce2dc5123559710d72` |
| 144K A | 144462 / 8 / 144470 | 10.369423 s | 200 | `ZQ7K-3341`, exact | `7ebbf19e6c993f08964941a5e20472b0506904fbf465447884fa87ed36c66382` |
| 1M D1 | 1019847 / 7 / 1019854 | 101.492258 s | 200 | `RB9N-6014` | `1921a59f26ee94a5b9a274c339bad428c11d08aebeea2868fe11c6e6b1c6d653` |
| 1M D2 | 1019847 / 7 / 1019854 | 101.542457 s | 200 | `RB9N-6014` | `e0f88c96bdfb6b93202051073149d03bb6797492079985809fb6ce83b776ddbb` |
| 1M D3 | 1019847 / 7 / 1019854 | 101.573675 s | 200 | `RB9N-6014` | `e0d13428c4ac183ba30b15cfe79ec11510b3c871e9b282d225a878ecac3ba5cc` |
| 1M D4 | 1019847 / 1 / 1019848 | 101.142135 s | 200 | `content=null` | `0925b739126e20e7fe8239091d421d2ec51a64cfe5ea6e361fd53361b2ef141e` |
| 1M D5 | 1019847 / 7 / 1019854 | 101.540291 s | 200 | `RB9N-6014` | `70532d511df2174596f3d6182b4feebd6fa7e06df4e9bdb8d02e2c3ae7a70683` |
| 1M D6 | 1019847 / 7 / 1019854 | 101.515998 s | 200 | `RB9N-6014` | `bf803c7748c7b3b16fd9e6b3583ae4fb931e6deb94881bae77d73c0686ccc4b2` |

U+FFFD count was zero in all eight responses. D4's complete raw response and every field of its `choices` array are preserved in `1mD4_inline.response.json` and `D4_choices_full.json`. The six 1M requests returned five exact `RB9N-6014` outputs and one null content (D4). In the first four requests the inline arm was 3/4, matching the MS0 arm's 3/4 observation; D5/D6 were both correct. This sample does not isolate the cause or establish a general accuracy rate.

Every request has raw request/response bytes, SHA sidecars, headers, HTTP/curl status, start/end timestamps, transport wall time, parsed usage summary, P handoff match, and D replay match. Full P/D/proxy host logs and runtime/health/NPU snapshots are in `live_final_d6_20260924/`. `D4_choices_full.json` preserves the complete choice object. `SHA256SUMS` covers the evidence directory.

No D7 was sent. After the final logs and hashes are saved, the inline D and proxy are stopped; the original P instance remains untouched.
