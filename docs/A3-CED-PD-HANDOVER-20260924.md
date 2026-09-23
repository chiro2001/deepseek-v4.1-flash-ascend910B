# CED-PD 交接：2026-09-24

## 目标与分支

- 活跃目标仍是 A3 上的真实权重 8+8 CED-PD：P 长 prompt 主段跑前 20 层、D
  128-token SWA 有界重放并完整 40 层 decode；BF16 KV 可保留。还需 1M、
  缓存命中和严格性能对照，**目标未完成**。
- 实验分支 `feat/ced-pd-a3`，工作树
  `/home/chiro/projects/dsv41/ced-pd-release`。基础全 40 层双 TP8 PD 在
  `docs/A3-PD-BF16.md`。CED 工作记录在 `docs/A3-CED-PD-WORKLOG.md`；
  D 实现边界在 `experimental/ced/D_REPLAY_NOTES.md`。

## 已实测

- A3-21 真实权重 TP8 层 20 来源投影：8 rank 的 144K 分块前/后采样缓存行
  与普通层 20 精确相等。P 跳层能够完成 144K 内部交接；P 置空未写的上层
  SWA G7–G11，并返回 replay 元数据。
- A3-21 真实权重 8+8 CED-PD：短请求正常；144K `bigprefill` 2/2、
  流式/工具/并发 5/5、多轮增长 3/3 正确，无乱码指纹。8+8 长运行由
  子代理 `/root/a3_8x8_validation` 管理；用户要求它暂停并交接，之后
  重启 Codex 再派新子代理。以其交接为准确认当前 A3-21 服务和 1M 状态。
- A3-22 单卡 tiny 框架是既有 `a2/agents/L1_dummy` 的 `model-tiny`，
  保留 40 层和 12 组 BF16 缓存；dummy 权重不代表真权重质量。
  tiny 模型经 COS 复制到
  `a3-22:~/projects/dsv41-ced-singlechip/model-tiny`，P 用 Phy-ID 6、
  D/基线顺序用 Phy-ID 7。用户预留 A3-22 chip0 和 chip1，**不得使用**。
- tiny 1+1 的长度 2/127/128/129/130/256/512/4096 各两遍，16/16
  HTTP 200、重复结果一致；与全 40 层基线对照时 16/16 选中 token 相同、
  top-20 集合 20/20 一致，共同候选最大 logprob 差 `9.54e-7`。
  证据在 `evidence/ced_tiny_pd_b6ca283/`。
- tiny 长度 1 会触发 D 调度的 `prefix boundary mismatch`，使 D 引擎退出。
  原因已定位：P 对单 token 不截尾，D 校验只按 `N−1` 写。尚未修复。

## 当前未验证诊断与下一步

- 本工作树新增 `V41_CED_SNAPSHOT_POS` / `V41_CED_SNAPSHOT_DIR` 诊断，
  `experimental/ced/dsa_v41.py` 按指定位置导出 40 层 SWA 行和层 20
  全局源到 NumPy 文件；`tools/compare_ced_cache_snapshots.py` 对照。
  **语法已检查，未在 A3 上运行**。下一步可令 `CED_SNAPSHOT_POS=126`
  分别重启 tiny 基线和 D，给同一 128-token 输入，比较位置 126。
- A3-22 上次已确认 P 容器 `dsv41-ced-tiny-p-b6ca283`、HTTP 18960、
  chip6 健康；D `dsv41-ced-tiny-d-b6ca283` 和代理已停止。基线容器
  `dsv41-ced-tiny-baseline-ccf5762`、HTTP 18963、chip7 在对拍时健康。
  最后一次 SSH 状态查询未返回（已中断该只读命令）；**重启后先只读复核**。
- 发布包 checksum 现可用 `python3 tools/refresh_checksums.py` 统一刷新
  `patches/MD5SUMS`、系列清单、`PATCHES.md` 与 `MANIFEST.sha256`；
  新文件须先 `git add`，之后跑 `bash tools/selfcheck_pkg.sh`。

## 环境与资源

- A3-21 SSH `a3-21` → `192.168.45.21`；A3-22 SSH `a3-22` →
  `192.168.45.22`。节点时间比本机约慢数分钟，以远端 `date`、日志字节
  增长、真实进程为准。
- 大文件传输用 `upstream-v41/pr/cos-xfer.sh` 和 COS 私有
  `share/xfer/`；临时文件放 `~/tmp/<日期>/<任务>/`。
- `upstream-v41/` 是冻结参考区，不写。单卡 P/D 测试脚本是
  `scripts/serve_a3_ced_single.sh`，全 40 层基线使用其 `baseline` 角色。
