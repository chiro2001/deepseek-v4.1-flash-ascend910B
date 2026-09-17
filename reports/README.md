# reports/ —— 阶段性结论与证据索引（进 git）

约定：每个阶段产出**一个** `reports/<stage>-<slug>.md`，包含

1. 目标与判据（引用 EXEC_PLAN_VB.md 的 G 编号）
2. 配置指纹（run_id / host / 芯片组 / 容器名 / 模型 realpath / config hash）
3. 原始证据路径（logs/perf/... 之类的大文件，不进 git，只登记路径）
4. 实测数据（启动账、ms/step、接受长度、tok/s、内存账）
5. 结论（PASS / FAIL / PARTIAL）+ 与判据逐条对照
6. 未完成项与风险

审查 Agent 会据此复核：代码实现、实验方式、数据、结论四方面。
