# A3-22 graph profiler cleanup

Checked 2026-09-24 09:36 +08 after the raw trace and controls were archived:

- The graph D container and its proxy were stopped and removed.
- P remained running on `/dev/davinci6`.
- `docker ps` no longer listed the graph D or proxy containers.
- `npu-smi` listed no process for Phy-ID 7. It still showed 2,904 MB HBM in
  use there, so this records process release rather than zero device memory.
- A3-22 chip0/1 and the A3-21 8+8 run were not touched during cleanup.

Private raw snapshot: `share/xfer/ced_tiny_stream_prof_d_graph_20260924_release_status.txt`
(SHA-256 `2e1f802a2a800e9de58c2aa2aeeded43bfc7db2a5fee9f11f086515a8dd86349`).
