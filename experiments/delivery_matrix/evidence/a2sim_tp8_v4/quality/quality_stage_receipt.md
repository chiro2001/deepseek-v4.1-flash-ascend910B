# A2v4 quality-stage receipt (2026-10-13)

This receipt records the serial GSM100 → Vision23 → 147456-token needle quality
stage for the A2-sim TP8 v4 service. It is not a performance result.

## Service identity and execution scope

- Target: A3-21, container `dsv41-ced-p0-7-20261013-v2`, ID
  `16d3045321f8af0e07e7392a4feabe14603e684589947126f63f1b37b6c631bd`.
- Final read-only state: `running`, restart count `0`, host PID `1541878`, and
  `127.0.0.1:18960` listening. The service was left online.
- Served model: `dsv41-delivery-a2sim-tp8-v4`; SID
  `9d2ba214732b4eed84dc4f416f27c6ae`; runtime fingerprint
  `61b841dfcd20d7f2892934e1fbd8086ab409a36d99d5de9beea4dd12399e53f9`.
- Before each suite, `GET /delivery/identity` returned HTTP 200 and matched the
  model, SID, and fingerprint above. Each saved identity document has SHA-256
  `95bb15a8f5b7c45c4699df70832b98af445ab4fbe0fc00d16e2478dc0a8239ef`.
- Final post-quality identity and read-only device/container state are archived
  in [identity_after_quality.json](identity_after_quality.json) and the
  [site-live audit](site_live_audit_20261013.md).
- All three suites ran serially against the standard HTTP API. No private
  transfer contract was sent. No Dihuo workload or performance benchmark was
  part of this quality-stage run, and no service restart occurred.
- Local task scratch: `/home/chiro/tmp/dsv41-a2sim-v4-quality`; remote task
  scratch: `/home/l00886679/tmp/dsv41-a2sim-v4-quality`.

## GSM100

- Frozen evaluator: remote
  `/home/l00886679/projects/dsv41-real-stack-tp8-20261010/client/tests/acc_eval.py`,
  SHA-256 `aad813de1b1af090c4186b0a443b7839339928f90b63e7c3a654d0e028b8e6e2`.
- Transparent wire sidecar: `quality_wire_capture.py`, SHA-256
  `f5bc086e69fed7706933390cb4287460c3aa536a378f1d73b3700c7c3eeb985d`.
- Settings: canonical first 100 GSM8K test rows; the evaluator's original
  first-eight training examples (8-shot), `chat`, `max_tokens=512`,
  `temperature=0.0`, `conc=1`, serialized prefill, original retry policy, and
  model `dsv41-delivery-a2sim-tp8-v4`.
- Result: **100/100**, empty `0`, errors `0`, elapsed `114.7s`.
- Wire audit: 100 unique request-body hashes; all 100 responses HTTP 200 with
  usage, `finish_reason=stop`, consumed `[DONE]`, closed response, and zero
  unparsed chunks. The sidecar observed evaluator-consumed bytes only and did
  not issue or alter requests.
- Data provenance: test JSONL SHA-256
  `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` (749,738
  bytes; archived under `gsm100/source/gsm8k_test.jsonl` and copied to
  `/home/chiro/tmp/dsv41-a2sim-v4-quality/data/gsm8k_test.jsonl`); train JSONL SHA-256
  `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465` (4,166,206
  bytes; retained on A3-21 at
  `/home/l00886679/projects/dsv41-real-stack-tp8-20261010/client_data/gsm8k/train.jsonl`).
  Official encoder path:
  `/home/l00886679/models/DeepSeek-V4.1-Flash/encoding`; sorted per-file SHA
  manifest digest (excluding `__pycache__`)
  `7055766f3d6e7d3df5d9c72f5b424ba595952fa5418483a3aad1f9f0c8761a9b`.
- Local artifacts:
  [scored output](gsm100/gsm100.json) SHA-256
  `1a0323fe84e72624cf1e7d006d5e569ec2f9db318b7641ebe16b189a9cd30af5`;
  [checkpoint rows](gsm100/gsm100.json.jsonl) SHA-256
  `60efd59d3f6e6a4e4452dfb6d22b796e85b6b6723a940c0b8063207870c11c3b`;
  [wire sidecar](gsm100/gsm100_wire.jsonl) SHA-256
  `103106209a419d714bb70d7b175c628b21a746dc903aece8cb21c33a806a2fad`;
  [log](gsm100/gsm100.log) SHA-256
  `40abdf267770e7c5f992b8c9fbe539d68a4b0c60993623e2fb9cd46f2b4520ca`.

## Vision23

- Frozen evaluator: `vision23/source/vision_accuracy_check.py`, SHA-256
  `af13bc4785bfcddcd46a398fe7611c6942a4875e1eee8c767ce31148ebee675f`.
- Settings: original 23-case suite, `max_tokens=32`, `temperature=0`,
  `expect-dspark`, and `min-hit-rate=1.0` for the strict 23/23 gate.
- Result: **23/23**, empty/garbled `0`, errors `0`, mandatory negative checks
  passed; verdict `PASS`. DSpark was expected; recorded acceptance rate `0.6309`
  and mean acceptance length `1.6309`.
- Official image inputs are archived under
  `vision23/source/images/`: `carrots.jpeg` SHA-256
  `5df896a4a07e127281c60fc957f8b3d73f4735b3258a0bf762b4383557f8fa9a`, and
  `corn.jpeg` SHA-256
  `251fe7ad58f44db139a6af5ed489e027eca93ae60878b83f3e709c7d6f0112a1`.
  Official model index SHA-256
  `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`; the
  evaluator reported 266 vision tensors totaling 970,536,960 bytes.
- Local [raw result](vision23/vision23.raw.json) SHA-256
  `eadf77e0583854d3d9daaac3063c7218d56d57cdffea8de864c10eb0ff99b19d`; [log](vision23/vision23.log)
  SHA-256 `4cb7bf465c1bfd9d65a7001ddadf88c71f8e9656d370ea2c21065fa4b983085a`.

## 147456-token needles

- Frozen runner: `needle147456/source/ced_pd_acceptance.py`, SHA-256
  `d12d746f1831431e6c6aa55447ae3b2d9a04e15b58800d505aaad508b7eff71e`.
- Raw API capture wrapper: `needle147456/source/capture_needle_api_raw.py`,
  SHA-256 `df0452fd6735e89a8780da8849913c946f14ec14aede8b695eb20aa5c8a71076`.
- Settings: `mode=needle`, `context_tokens=147456`,
  `max_model_len=196608`, `max_tokens=64`, `repeat=1`, `offset=0`; direct
  `/tokenize` and `/v1/chat/completions` at `http://127.0.0.1:18960`.
  The request body used `stream=false`, `temperature=0.0`, and carried no
  transfer/private-contract fields.
- Corpus: A3-21
  `/home/l00886679/projects/dsv41-real-stack-tp8-20261010/client/data/hongloumeng.txt`,
  2,468,721 bytes, SHA-256
  `a7fc413bd6e3926482faddf2af9bfb4426e55d8e421481d60100c14160785578`.
- Strict validity rule: each actual `usage.prompt_tokens` must be at least
  `145982` (99% of 147456) and each answer must match its needle.

| 针 | 结果 | usage.prompt_tokens | completion_tokens | finish_reason | request SHA-256 | response SHA-256 |
| --- | --- | ---: | ---: | --- | --- | --- |
| A | PASS | 147438 | 8 | stop | `3c0637516a61caf08b47663c0f43e242b30d7d8c0d640ede87143ce7181d8fe0` | `0f9c21b865bb1de6e627d29bde5e3c852d83d758e01b3d955b008034dd02485c` |
| B | PASS | 147436 | 8 | stop | `0fbace86096ee3a75e19869f029a7f4cf08b9ffc48758f92a192ecfa78a88c49` | `aa75edc147888d88412dbfef79611422554de1aea57182667caf407371f20c52` |
| C | PASS | 147437 | 7 | stop | `87486743150b4f7d9413686fc0009ad4feb918f4e623e3f04551b9cfbe21417d` | `d55dee6815414c73885c5c58aa60d948d0bdd09cee29eabb4a4e7e4ba3e3f4cd` |
| D | PASS | 147435 | 7 | stop | `65b651ade0ee4d2fc193646a483a654a350678fad6ebccf40a4b59e7bfa8d4f5` | `5ec0d146ddfef13d6da724ffad7c257779ffbfa294ef60adf82bf2c53aaecc06` |

- Result: **4/4**, each HTTP 200; all four actual prompt counts clear the strict
  length gate. Each request and response body is saved in
  `needle147456/dcp144k_needles_api_raw/`, with per-exchange metadata; parsed
  request/result records are in `needle147456/dcp144k_needles_requests/`.
- Local [summary](needle147456/dcp144k_needles.json) SHA-256
  `43589c91b3b01fbba4712a938dedfedd2d2cdb8ff8af965bc21f5198002522da`; [log](needle147456/needle147456.log)
  SHA-256 `1b1bd46655ba8b37b602a5cd669dabdd1d1c0d9924b0442e3ca4d83dea72ddce`.

## Gate outcome

GSM100, Vision23, and the 147456-token four-needle gate all passed. This records
quality only; it does not change the ready receipt's `actual_performance_observed=false`
and does not authorize or claim a Dihuo/performance result.
