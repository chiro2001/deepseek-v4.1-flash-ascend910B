Invalid eager diagnostic launch retained as a configuration mistake; excluded from precision results.
Container: /dsv41-ced-prompt-tail-eager-d-20260924-1109 / 4a6ee94622d4d36cbaee1d645320788f7754eb3b1aae864d6455b97af7797a79 (state=exited, exit=137).
V41_CED_ROLE=''. Required CED connector/DSA/scheduler mount names absent: mooncake_hybrid_connector.py, dsa_v41.py, ced_scheduler_replay.patch. Generic Mooncake configuration in serve_cmd.txt did not establish the CED consumer path. The prompt-tail runner patch was off as intended for the planned eager mode.
POST /v1/chat/completions entries in serve.log: 0; response JSON files: 0. No proxy/model request was sent for this invalid role setup.
Raw inner.sh, serve_cmd.txt, serve.log, driver.log, launcher warning and docker inspect are retained here.
