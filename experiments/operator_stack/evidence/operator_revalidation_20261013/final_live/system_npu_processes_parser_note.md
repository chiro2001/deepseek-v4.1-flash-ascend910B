# Parser note

`system_npu_processes.txt` is retained as the first-pass capture, but its summary parser had over-escaped whitespace/digit patterns. Its `LISTENER_18960_COUNT=0` and absent NPU process rows are parser artifacts, not evidence of an absent listener or absent workers. Use `npu_processes_corrected.txt` for the complete in-scope NPU physical/process table and `listener_18960.txt` for the exact port-filtered `ss` result.
