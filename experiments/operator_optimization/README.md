# 独立 tiny 算子优化

分支：`feat/tiny-operator-opt-20261009`，从 `origin/main@998c47c` 创建。
环境：a3-21，物理chip4，独立容器 `dsv41-tiny-prof-20261009-c4`。
全程排除chip14–15，设备占用与进程归属在测试前核验。

## 基线

`baseline/` 是上一轮已验证的HC/router实现快照；历史结果与导入哈希在
`evidence/baseline/`。历史同进程结果为 `(26.122 ms/步, A=1, 38.282 token/s)`。
新一轮须重新测同进程baseline，不能把历史数字与新进程候选直接相减。

## 当前候选

`clamped_swiglu.py` 融合routed/shared激活，`DESIGN.md`记录语义与设计约束。
`activation_patches.py`、`operator_worker.py`在正常模型加载后接入，
保留四套独立图及完整attention事件/workspace/handles。

【实测】v2的139个精度用例全部通过；shared输出逐位一致，routed最多1 BF16 ULP，
input clamp副作用的比较通过。覆盖signed、尺度、clamp边界、非连续视图与NaN/Inf。
独立图七组交错测试：routed 26.771→14.112 μs（1.897×）；
shared 14.194→14.153 μs（1.003×，没有明确收益）。
数据在 `evidence/activation/`。

【实测】v1通过精度，但flat input store使routed图时延224.365 μs，已拒绝。
源码和原因在 `rejected/`。v2调整每program行/tile划分与BLOCK，保持输入修改。
【推断】v1的慢路径与非连续散射的编译降低有关，尚无指令级trace证明。

【实测】整模型激活和多流随机化审计均通过；最终六组对照为
`(26.456 ms/步, A=1, 37.799 token/s)` → `(24.446 ms/步, A=1, 40.907 token/s)`，
吞吐提升约8.2%，六组全部更快。完整结果和边界见 [REPORT.md](REPORT.md)。
本轮工作与优化原理见 [总结报告](reports/a3-21-tiny-operator-fusion-qkv-overlap-20261009-v1.md)。
性能脚本的audit阶段会在
编译前随机化gate、HC和routed/shared MLP权重；审计耗时不进入正式计时。

## 复现

本目录与 `baseline/` 均加入容器PYTHONPATH。源码包小于1 MB时可用：

```bash
python3 experiments/operator_optimization/sync_to_a3.py
```

容器内加载CANN/ATB环境后，在核验chip4占用并停止本次自己的tiny服务的条件下：

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
export PYTHONPATH="/work/operator_opt:/work/operator_opt/baseline${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR=/work/operator_opt/runtime/tmp
python /work/operator_opt/verify_activation.py --output=/work/operator_opt/results/precision_NEW.json
python /work/operator_opt/bench_activation.py --output=/work/operator_opt/results/operator_NEW.json
python /work/operator_opt/bench_operator_model.py --audit --pairs=3 --output=/work/operator_opt/results/audit_NEW
python /work/operator_opt/bench_operator_model.py --pairs=6 --profile --output=/work/operator_opt/results/perf_NEW
python /work/operator_opt/bench_operator_model.py --pairs=6 --test-overlap --arms=baseline,fused,overlap --profile --output=/work/operator_opt/results/combined_NEW
```

四模式为baseline/routed/shared/fused，均沿用已验证HC/router；baseline在本轮指旧both。
正式性能关闭profiler、logprobs和路由回传，profile另采20步，不混入计时。
每个图要求router=40、HC=80、routed/shared activation各40次，候选选择覆盖不足即失败。

原始trace留在远程；仓库只保存小型结果、源码和校验。tiny为TP1 dummy BF16、8专家/top2，
尚未验证生产TP8/W4A8的效果。
