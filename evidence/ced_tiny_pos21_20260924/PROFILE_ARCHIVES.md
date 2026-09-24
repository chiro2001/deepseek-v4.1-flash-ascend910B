# Profiler trace archives

Raw traces are stored as private COS objects and are not committed with this
worktree. Both archives use the private `share/xfer/` prefix.

## Eager D

- Object: `share/xfer/ced_tiny_stream_prof_d_eager_20260924.tar.gz`
- Size: 35,021,158 bytes
- SHA-256: `f70045e5264228eb40f32e14b9c80520c0b85fc36f8f5bbb45a2be71d85ea7d9`
- Contents: original `/opt/dsv41/results/ced_tiny_stream_prof_d_eager_20260924/prof/`
  tree, including `trace_view.json`, CANN raw `PROF_*` data, CSVs and DBs.
- Control bundle: `share/xfer/ced_tiny_stream_prof_d_eager_20260924_controls.tar.gz`
- Control bundle SHA-256: `19adf65376b91f3f7e77817afe87583d593aeadcafdf1bfeb5342560aa66b690`

Download from the workspace with:

```bash
bash upstream-v41/pr/cos-xfer.sh get \
  share/xfer/ced_tiny_stream_prof_d_eager_20260924.tar.gz \
  /tmp/ced_tiny_stream_prof_d_eager_20260924.tar.gz
sha256sum /tmp/ced_tiny_stream_prof_d_eager_20260924.tar.gz
mkdir -p /tmp/ced_tiny_stream_prof_d_eager_20260924
tar -xzf /tmp/ced_tiny_stream_prof_d_eager_20260924.tar.gz \
  -C /tmp/ced_tiny_stream_prof_d_eager_20260924
```

`profile_source_SHA256SUMS.txt` in the control bundle contains hashes recorded
on A3-22 relative to `prof/`. From the extracted directory, run
`sha256sum -c profile_source_SHA256SUMS.txt` after copying that manifest next
to the extracted `prof/` tree.

## Graph D

- Object: `share/xfer/ced_tiny_stream_prof_d_graph_20260924.tar.gz`
- Size: 22,470,654 bytes
- SHA-256: `e59f6081406f34a36fdf4f24c514a1bbd085403165e12d9b37dda437f8439abf`
- Contents: original `/opt/dsv41/results/ced_tiny_stream_prof_d_graph_20260924/prof/`
  tree, including `trace_view.json`, CANN raw `PROF_*` data, CSVs and DBs.
- Control bundle: `share/xfer/ced_tiny_stream_prof_d_graph_20260924_controls.tar.gz`
- Control bundle SHA-256: `d8a2b47f2f26c32de5009d51f5e52729c191ab2eda8aa1774d8447b2f1be47a6`
- Post-cleanup resource snapshot: private object
  `share/xfer/ced_tiny_stream_prof_d_graph_20260924_release_status.txt`,
  SHA-256 `2e1f802a2a800e9de58c2aa2aeeded43bfc7db2a5fee9f11f086515a8dd86349`.
- The controls contain the exact run `serve.log`, `inner.sh`, `serve_cmd.txt`,
  profile summary, selected container facts, and post-profile `npu-smi` output.
  `SHA256SUMS` in the bundle verifies all control files.
- `prof_graph/profile_source_SHA256SUMS.txt` verifies the 131 files extracted
  from the raw archive.
- The graph profiler ran on `/dev/davinci7` on A3-22. The P service remained on
  chip 6. The 8+8 run is on A3-21; it does not share this host's device.

Download and unpack the graph trace with:

```bash
bash upstream-v41/pr/cos-xfer.sh get \
  share/xfer/ced_tiny_stream_prof_d_graph_20260924.tar.gz \
  /tmp/ced_tiny_stream_prof_d_graph_20260924.tar.gz
sha256sum /tmp/ced_tiny_stream_prof_d_graph_20260924.tar.gz
mkdir -p /tmp/ced_tiny_stream_prof_d_graph_20260924
tar -xzf /tmp/ced_tiny_stream_prof_d_graph_20260924.tar.gz \
  -C /tmp/ced_tiny_stream_prof_d_graph_20260924
```
