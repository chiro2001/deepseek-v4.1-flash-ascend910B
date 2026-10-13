# DCP8 v3 144K needle 客户端存储失败记录

采集时间：2026-10-13 04:10 CST（目标机时钟，比本机慢约 8 分钟）。这是客户端证据落盘路径失败，不是模型/服务质量结论；这条 A 请求不计入正式 4/4 门。

## 已发生情况

- 目标：SSH `a3-21`（192.168.45.21）；同一 `dsv41-delivery-dcp8-v3` service instance `3d8c0140942c48d58d73a291b42059d2`，请求目标 `127.0.0.1:18960`。实际已发出第一条 147456-token A needle、`max_tokens=64` 非流式请求；服务响应已返回给 `ced_pd_acceptance.py`，随后客户端在 `record()` 落盘前退出。
- 客户端 SSH 命令 exit 1；Python stderr 末行为 `PermissionError: [Errno 13] Permission denied: '/home/l00886679/projects/dsv41-real-stack-tp8-20261010/results/delivery_dcp8_v3/quality/client/dcp144k_needles_requests'`。失败点是 `os.makedirs(out_dir, exist_ok=True)`，因为从宿主用户运行时该 `/work` bind 目标目录不可写。
- `post_json()` 虽在内存中拿到 response body，但该工具没有先写 response 原件；异常终止使 A 的原始 HTTP body、解析答案、request/result JSON 均未保存。因此 A 不能判 PASS 或 FAIL，也不能恢复/推断其答案。随后只读 `test ! -e` 检查旧 JSON 与 requests 目录均 exit 0，确认无部分输出；18960 listener 与 API PID 841625 仍在，未重启服务。

## 路径修正（全新任务目录）

- 失败的旧输出路径：
  - `/home/l00886679/projects/dsv41-real-stack-tp8-20261010/results/delivery_dcp8_v3/quality/client/dcp144k_needles.json`
  - `/home/l00886679/projects/dsv41-real-stack-tp8-20261010/results/delivery_dcp8_v3/quality/client/dcp144k_needles_requests/`
- 新任务目录：`/home/l00886679/tmp/dsv41-dcp-quality-rerun/`。正式重跑的汇总、逐请求证据和先行 raw API response capture 将分别放在该目录下的 `dcp144k_needles.json`、`dcp144k_needles_requests/`、`dcp144k_needles_api_raw/`；不会改 `/work` 权限、不会 chown/chmod 项目路径、不会覆盖任何旧输出。
- 正式重跑保持原服务实例、模型、147456 context、`max_tokens=64`、重复 1 次和 0.9 旧脚本默认长度门，并另行人工检查每条 API `usage.prompt_tokens >= 99% × 147456`。额外 wrapper 只在 frozen runner 的原 `post_json()` 返回后立即落 request/response 原始字节与 SHA，再把原返回值不变地交给既有 scoring；不改针、答案或判据。
