# DCP v3 运行时标志只读复核

采集时间：2026-10-13 04:15 CST。用途：在正式 DCP 144K 四针前排查 DSA/DSpark/Engram 的遗留热切标志；只读，不删除、不改写。

## 目标现场

- 所有远端检查均显式经 SSH `a3-21`（`l00886679@192.168.45.21`）执行；目标容器 `dsv41-real-stack-tp8-20261010-a321`，ID 前缀 `b1aa15d88003`；检查时 API 位于容器 PID namespace 的 PID `841625`。
- 同一检查窗口中，target `/delivery/identity` 仍报告 `dsv41-delivery-dcp8-v3` / `a3-tp8-dcp8`、service instance `3d8c0140942c48d58d73a291b42059d2`、8 rank TP8/DCP8、async 与 draft graph=true；`18960` listener 存在。正确的独立 `docker inspect` 查询 exit 0：container running、restart_count=0、host init PID=585275、OOMKilled=false、dead=false。

## 文件核验

对下列每个**容器内**路径分别执行 `docker exec <container> test -e <path>` 和 `test -L <path>`。每个 `test -e` 均 exit 1（路径不存在）；每个 `test -L` 均 exit 1（不是符号链接）。因此没有文件内容或 SHA-256 可记录，缺失状态明确如下：

| 容器路径 | 存在 | 符号链接 | 内容/SHA |
|---|---:|---:|---|
| `/tmp/v41_perf_flags` | 否 | 否 | 缺失，无内容/哈希 |
| `/tmp/v41_dspark_flags` | 否 | 否 | 缺失，无内容/哈希 |
| `/tmp/v41_engram_localowner` | 否 | 否 | 缺失，无内容/哈希 |
| `/tmp/v41_engram_localmeta` | 否 | 否 | 缺失，无内容/哈希 |
| `/tmp/v41_route_pipe` | 否 | 否 | 缺失，无内容/哈希 |

由此未发现 `/tmp/v41_perf_flags` 中 `rank0_only`、`skip_2nd`、`sepw` 等 DSA 性能消融残留，也未发现其它列出的 DSpark/Engram/route 热标志文件。

## API 进程环境

只读解析容器 `/proc/841625/environ`（Python exit 0，仅打印请求的两个变量名和值）得到：`V41_DCP_DET_REDUCE` 与 `V41_DCP_ORI_OWNER` 均未设置；未读取或输出其它环境变量。按本轮代码默认处理，记录为 unset，不伪报成显式 `0`。

## 采集命令异常说明

首次批量循环因嵌套 shell 引号错误 exit 127。该命令输出的 SSH 非 PQ 密钥交换 warning 与 `Authorized users only. All activities may be monitored and reported.` banner 之后，shell stderr 原文为：

```text
bash: line 1: ^V41_DCP_ORI_OWNER=\ || true: command not found
%s\\n\ \\\; stat -c \mode=%a: -c: line 2: syntax error: unexpected end of file
```

该循环没有执行任何写操作。其后改为每个路径的直接 `test -e` / `test -L` 与单独 `/proc` 读取；以上缺失判断来自这些成功抵达目标容器的分项检查，不来自失败循环。分项 `test -e` 与 `test -L` 均 exit 1；SSH warning/banner 非致命，分项命令没有其它 stderr。

## 结论

复核只证明列出的标志文件缺失以及两个变量 unset，不证明其它未检查的 runtime 设置。未清理或重启任何内容；这项检查未发现所列标志对接下来 DCP 四针的污染。
