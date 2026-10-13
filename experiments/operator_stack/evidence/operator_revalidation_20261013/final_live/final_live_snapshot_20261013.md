# base2 final live snapshot

日期：2026-10-13（Asia/Shanghai）。本快照在后续 client 8+1 generation 已完成后采集。服务未重启、未停止；本次只读检查未发送 generation，没有调用 metrics、worker RPC、Env/Args、NPU profiler，也没有操作 physical 0–7 或进入外部容器。

## Ready / online

- 目标 D `dsv41-real-stack-tp8-20261010-a321`：CID `b1aa15d8800346aff8f755a049feaae148b9d6e315e92bd32dcdf1063b429eb0`，`running`，restart count `0`。
- 目标 host `ss -ltnp 'sport = :18960'`：`127.0.0.1:18960` 为 `LISTEN`。
- GET `/v1/models` 与 GET `/delivery/identity` 各一次、均成功；过滤后的响应在 `v1_models_sanitized.json` 与 `delivery_identity_sanitized.json`。响应只保留核验字段，不保存 identity 的 environment/model kwargs。
- Models 返回唯一 ID `dsv41-operator-revalidation-tp8-20261013`。Identity 同 model，SID `3cbd844d9f684c43867b4b85c9c78ebe`，`engine_initialized=true`、`ASGI_startup_completed=true`，runtime/CFG fingerprint `79513a0f20932cf696374573323b715aab0d04df774721fbee92f2f780da4319`。
- 八个 rank 描述均通过校验：rank 0–7、TP=8、physical chips `8–15`、DCP=1、async scheduling=true、draft graph=true、speculative tokens=5。未以此推断额外运行时行为。
- Identity PID `1626263` 在目标 D 中为 `python3`、state `S (sleeping)`、NSpid `1626263`，cgroup 命中目标 CID；见 `api_process_status.txt`。

## NPU / rank process identity

修正后的 `npu_processes_corrected.txt` 是本轮权威的完整 in-scope NPUinfo：physical 8–15 的 8 行 Health 均为 `OK`，每卡一条 `VLLMWorker_TP` 进程行，并定向读取对应 host PID 的 status/NSpid/cgroup。所有 worker state 为 `S (sleeping)`，均命中目标 D CID。

| Rank | Physical | Host PID | Container PID (`NSpid`) | Health | State | Target D cgroup |
|---:|---:|---:|---:|---|---|---|
| 0 | 8 | 3494441 | 1627158 | OK | S | match |
| 1 | 9 | 3494671 | 1627177 | OK | S | match |
| 2 | 10 | 3495230 | 1627231 | OK | S | match |
| 3 | 11 | 3495603 | 1627277 | OK | S | match |
| 4 | 12 | 3495846 | 1627323 | OK | S | match |
| 5 | 13 | 3496095 | 1627369 | OK | S | match |
| 6 | 14 | 3496501 | 1627415 | OK | S | match |
| 7 | 15 | 3496576 | 1627461 | OK | S | match |

EngineCore host PID `3494016` is `VLLM::EngineCor`, state `S`, NSpid `3494016 1627117`, and cgroup-matches the same target CID. The container init PID is `585275`; its state and the EngineCore/worker rows are in `system_npu_processes.txt` (with the parser caveat below).

## Client process boundary

No PID file was provided for the client task. Using only the parent-specified exact script path `/home/l00886679/tmp/dsv41-operator-baseline-client-v3-20261013/client_src/benchmark.py`, the host process match returned none; no broad process/secret scan was done. The completed client task’s exit-0/no-second-window receipt remains the authoritative workload result.

## Snapshot times and parser note

- Target D/container snapshot: `2026-10-13T04:15:22.097297+00:00` (`system_npu_processes.txt`).
- Corrected NPU snapshot: `2026-10-13T04:16:59.421868+00:00` (`npu_processes_corrected.txt`).
- Port listener check: `2026-10-13T12:17:31.775688190+08:00` (`listener_18960.txt`).
- Models GET: `2026-10-13T12:17:49.597831720+08:00`; Identity GET: `2026-10-13T12:17:59.184123390+08:00` (stderr timestamp receipts).
- The initial `system_npu_processes.txt` first-pass regex summary incorrectly showed listener count 0 and omitted NPU process rows. It is retained but explicitly non-authoritative; see `system_npu_processes_parser_note.md`. The corrected NPU table and exact port-filtered `ss` above supersede those parser results.
- Physical 0–7 were not matched to processes or inspected; any occupancy there is outside this check’s scope.

All captured evidence files and their SHA-256 values are listed in `artifact_hashes.sha256`.
