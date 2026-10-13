# A2v4 post-quality live-site audit (2026-10-13)

Read-only snapshot from A3-21 at `2026-10-13T08:09:50+08:00`, after GSM100,
Vision23, and the 147456-token needle suite. No container was started, stopped,
or entered for this audit. In particular, `xp_mk` was not touched.

## A2v4 service

- P container ID `16d3045321f8af0e07e7392a4feabe14603e684589947126f63f1b37b6c631bd`:
  `running`, restart count `0`, host PID `1541878`.
- Listener `127.0.0.1:18960` remained present. A final
  `GET /delivery/identity` returned HTTP 200 and matched model
  `dsv41-delivery-a2sim-tp8-v4`, SID
  `9d2ba214732b4eed84dc4f416f27c6ae`, and runtime fingerprint
  `61b841dfcd20d7f2892934e1fbd8086ab409a36d99d5de9beea4dd12399e53f9`.
  Saved identity SHA-256:
  `95bb15a8f5b7c45c4699df70832b98af445ab4fbe0fc00d16e2478dc0a8239ef`.
- Service launcher PID `2898801` and EngineCore PID `2901163` were live; all
  eight `VLLM::Worker_TP{0..7}_EP{0..7}` processes were in `Sl` state:
  `2901405, 2901438, 2901634, 2902060, 2902112, 2902233, 2902285, 2902782`.
- P's device cgroup remained limited to `/dev/davinci0` through
  `/dev/davinci7`.

## Read-only device/container snapshot

`npu-smi info -m` mapped physical chips 0–15 across NPU IDs 0–7, two chips per
NPU ID. At the same snapshot, `npu-smi info -t usages` showed:

| Physical chips | HBM usage per chip | AICore / NPU utilization |
| --- | --- | --- |
| 0–7 | 85%, 84%, 85%, 84%, 85%, 84%, 85%, 84% | 0% / 0% |
| 8–15 | 4% on each chip | 0% / 0% |

HBM bandwidth usage was 0% for all 16 chips at that instant. This is an
instantaneous utilization snapshot, not an ownership or future-availability
guarantee. `npu-smi info proc` reported that process information is unsupported
on this device; therefore this snapshot does not claim process-level
chip attribution.

Running container device mappings at audit time:

- A2v4 P: `/dev/davinci0–7`.
- Project D container `dsv41-real-stack-tp8-20261010-a321` (ID
  `b1aa15d8800346aff8f755a049feaae148b9d6e315e92bd32dcdf1063b429eb0`):
  `/dev/davinci8–15`.
- External `xp_mk` (ID
  `f11a0b179a1f992c8821781c50cec5862bc6e7a90ef02ecf8bb2c0acad39f1e5`) and
  `m00933363_catlass_runtime_bz_a3_1` (ID
  `6c515b1ce0865f7e48ca3dcd28055b31f6078d4cde3a6262e6cfe82f85d4a164`): both
  have device mappings for `/dev/davinci0–15`; neither was modified or entered.
- `dspark-aisbench-client` (ID
  `ceda05ba2b49625ffcfe6a1a5f5c92a4b4ea9a65b6b2a684e1347844251f1fb1`): no
  `/dev/davinci*` device mapping.

The complete captured command output is
[site_live_after_quality.txt](site_live_after_quality.txt), SHA-256
`45a5eb58f70b25bfecbc1dd98aa8ec9b4dae51d8e92ff00cffece907342a473e`.
The final identity document is [identity_after_quality.json](identity_after_quality.json).

The initial `npu-smi info proc` form used in that snapshot was unsupported. A
later read-only check used the supported full `npu-smi info` command and
captured its per-NPU process table; see
[`site_live_during_dihuo_20261013.md`](../../ced_v4_startup_20261013/site_live_during_dihuo_20261013.md).
