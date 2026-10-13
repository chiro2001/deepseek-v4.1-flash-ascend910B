# 2026-10-14 恢复与主线整理审查

用户授权恢复 a3-21 physical0–7 的 DP2TP4、DP4TP2、DP4TP2DCP2 测试，并要求了解
其他 Agent 的整理后重新创建 goal。新 goal 已 active；旧 16 chip CED-PD / DRAM
卸载目标不恢复。本文件是代码/文档准备收据，不是新服务或实机验收收据。

## 版本与保留

- 暂停前已发布任务 HEAD：`d58a9e095c415004f9e244a66c77e68f1ed7354e`。
- 整理主线：`origin/main@889a6a92797d94ee2c7f6fc59a89bc9a7156b261`；2026-10-14
  `git fetch origin main` 后仍为该版本。
- `d58a9e0` 是整理主线的祖先。两者对 `experiments/dp_matrix`、
  `experiments/delivery_matrix`、`experiments/operator_stack` 没有内容差异。
- 将暂停前已知任务的 23 个 tracked 改动和独立 CLI 测试保存为 `eee747b`；没有
  stage 无关运行产物、coscli 输出、其他 untracked 文件或编译产物。
- 在现有任务分支合并整理主线，自动合并无冲突；不改其他 worktree，不改主线，
  不强推，也不创建新的外部 Release。

## 对实验的影响

整理主线包含 A3 发布入口的容量默认 `BAT_TOKENS=2048 + V41_KV32_REPACK=1`、
Engram device-index 默认 1、DSpark×DCP 修复撤销，以及 TP8 解码融合叠加包的负结果。
这些不是本轮 DP 候选的实机证据，也不自动改变 SHA 冻结 source / SDK / model /
draft / counter。当前 DP 基线仍为 Engram on / device-index off、DSpark5 真草稿图、
BI / async / AllGather，BAT8192、util0.85、loader threads16、host interleave0,1,2,3。
新容量或 DCP2 候选须单独记录配置、源与精度，不借用旧 TP8/DCP8 PASS。

README 速览已修正两处证据口径：8 个 A3 die 是 4 块卡；历史 BAT8192 的单流性能
不是 BAT2048+repack 新默认的联合测量。DRAFT_GRAPH 默认说明与 A3/CED-PD=1、A2=0
入口一致；旧多流显式 DRAFT_GRAPH=0 的数字保留，不能混为当前默认。

## 本地验证与发布只读核实

- 合并前、合并后：`TMPDIR=/home/chiro/tmp/dsv41-dp-resume-20261014`
  `python3 -m pytest -q -p no:cacheprovider experiments/dp_matrix`，均为
  **166 passed / 400 subtests passed**。CPU PASS 不等于实机精度、性能或容量 PASS。
- `python3 tools/check_checksums.py`：三方一致。
- 本任务相对整理主线的 `git diff --check` 通过。整理主线自带若干历史归档 patch /
  文档 trailing whitespace；本次不改写这些冻结/存档字节。
- `gh release view a3-tp8-fusion-v1 --repo chiro2001/deepseek-v4.1-flash-ascend910B`
  返回 `release not found`。保留 tag 不代表仍有 Release；不恢复发布该实验包。

## 下一现场门

唯一现场操作者重新核实0–7占用/PID/cgroup/Health、目标容器实际身份与可见设备、
可写且可见的独立短 job 目录、NUMA/RAM/策略、native source 与完整 OPP 哈希、模型
与 Engram、端口归属。全部通过后重冻结新包，再由主代理放行 DP2TP4 初始化。
此前 job 根不可见/不可写，不能复用旧包路径、旧 SID 或重启容器修挂载。
