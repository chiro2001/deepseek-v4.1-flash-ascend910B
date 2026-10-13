# CED P 容器前主机内存门

采集目标：SSH `a3-21` / 192.168.45.21，主机 `localhost.localdomain`。快照时间：2026-10-13 04:40:15 +08；此时 DCP8 v3 API 已停止，CED P 容器尚未创建。检查只读 `/proc/meminfo` 与 `/proc/*/smaps_rollup`；没有清理 pagecache、发服务请求、启动服务或终止进程。

## 可用容量与粗略预算

- `MemTotal=2111110764 KiB`；`MemAvailable=2087538100 KiB`，约 **1990.83 GiB**；`MemFree=1558747152 KiB`；`Cached=528696704 KiB`；`SwapFree=3836412 KiB`（约3.66 GiB）。未对缓存做任何 drop/reclaim 操作。
- 按已知新 P 需额外完整 host Engram 约 **221 GB（约205.8 GiB）**，单算 Engram 后仍约有 **1785 GiB** MemAvailable 可覆盖有界加载/staging。可用内存约为该 Engram 预算的9.7倍。精确 staging 开销仍需建 P 前再核，不以本估算替代最终门。
- 判定：当前 MemAvailable 对 P 额外完整 Engram + 有界加载/staging 有充足余量，**内存容量门暂时 PASS**；容器创建前需刷新一次 `/proc/meminfo` 和 PSS，若余量变化则重新判定。未尝试启动 P。

## PSS 与残留进程

- 本机副本 `memory_pss_snapshot_before_p_20261013.json` SHA-256 `9538f7295db6911eedaa8894a4eb0e73e5396cc077ed70e35d40035511ffc9cb`；采到6359个可读 `smaps_rollup`，可见 PSS 合计约0.48 GiB，为下界（37个条目权限拒绝）。最大可见 PSS 进程为少量 `apx` daemon（约45–123 MiB）；audit Python 自身约20 MiB。
- PSS扫描同时看到24个 `VLLM::Worker_TP` 条目，另由目标宿主 `ps` 确认它们均为 `Z` / RSS=0、PPID=旧容器 init PID585275；它们是无映射页的历史僵尸进程，不是运行中的 worker。目标 `docker top` 在 DCP 停后只见容器 init，无 API/EngineCore/TP worker；NPU-SMI 对物理 chip 8–15 未发现运行进程。
- 若以当前 MemAvailable 与 PSS 下界评估，残留僵尸不构成实际内存占用；旧容器仍保留 running 状态，不为清僵尸而重启/停容器。后续是否作为 D 侧容器按 CED 计划另行决定；其 writable-layer OPP 不因此视为完整安装树已核验。

## 证据与复核

- 只读采集器：`/home/chiro/tmp/dsv41-ced-memory-audit/memory_pss_snapshot.py`，SHA-256 `186d26a745abae30472b16d153aaaf3247e6df4e2e56e2e976fb8bace67ef19c`；目标临时副本 `/home/l00886679/tmp/dsv41-ced-memory-audit/memory_pss_snapshot.py`。该脚本不改进程或内存状态，只输出摘要并把 JSON 放入任务专属 `~/tmp/dsv41-ced-memory-audit`。
- 下一步创建独立 P0–7 容器前需再记录当时 MemAvailable/PSS、目标残留进程及 OPP 复制 manifest；本文件不代表 OPP 安装树已完成核验，也不是 P ready 结论。
