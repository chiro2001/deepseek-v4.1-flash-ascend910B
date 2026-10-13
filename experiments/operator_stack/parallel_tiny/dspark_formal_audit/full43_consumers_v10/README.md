# v10 首次 full43 诊断包

本包提供新的私有 SDK load → 原生模型绑定 → main 图 / draft eager → post-fence 原始导出 → 正常 shutdown 通路。它用于首次真实入口诊断；17 类消费者的正式覆盖尚未完成，全部正式、数学、prepack 顺序与性能标志保持 false。

入口相对 stack：

```text
parallel_tiny/dspark_formal_audit/full43_consumers_v10/formal_audit_driver.py
```

root 负责重新核资源和启动新的 physical 8–15 job：

```text
--model /home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq
--physical-chips 8,9,10,11,12,13,14,15
--output /work/ROOT_SELECTED_NEW_JOB/gate
--generations 32 --max-output-tokens 192 --mode diagnostic --execute
```

缺少 `--execute` 时仅生成 CPU 候选契约，不构造 LLM、worker 或 NPU。审计请求固定 greedy、min=max=192、ignore_eos，不能用其文本授予质量通过。首次包的 `audit` 模式保持关闭。

`diagnostic passed` 仅在实际请求完成、8 rank 各自绑定 40 main + 3 draft 原生模块、至少 32 个实际 spec_decode_metadata 轮次具有全部 43 decoder 导出和全部显式 graph Tensor 出口，并且 8 个实际 owned Process 都 reaped0 后成立。它不会将 `full_consumer_audit_passed`、`formal_model_validated`、`safe_formal_receipt` 或 prepack 顺序标志改成 true。

载入回调保留原 `NPUModelRunner.load_model` 的 AnnAssign 和后续原步骤，在原 ACLGraph 包装前保留真实模型。17 原函数剥除 audit 后 AST 完全相同；私有修改只安装到本 job 的已载入对象。原 carrier 与原参数对象保留，原 Python super 闭包保留。编译根显式返回原结果、每点六个预分配 buffer，以及原 positions/valid/epoch/调用 marker；外层 adapter 在图外还原原返回结构。参数 layout/owner seals 在真实 fence 后复核。

IDS 的原 gmm1/gmm2 pure Tensor 源入口仅绑定 main M36/E48/top6 与 draft M15/E16/top3，FFN 宽度取真实 W2 与 partition。其六个 buffer 纳入对应根的显式输出。其他 extent、SILU 以外激活、weight-list、EPLB/LoRA 走原函数并记为未覆盖。实际 raw ND 格式、expanded 行位置、borrowed stream、GMM1 workspace 双 bank、GMM2 原 x GM 别名和设备 copy-before-pack 顺序仍待真实门证明。

每个 epoch 保留完整 CPU raw `.pt`；compact JSON 记录原 raw 路径、字节数和 SHA、原 payload SHA、positions/valid/layout、实际出口指针和缺失点。≥1MB 原始文件必须留远端；不得删除 raw 或通过 SSH 传大文件。observer 缺失、重复和不支持的 shape/dtype 保持未覆盖。

CPU 证据包含 16 个契约、17 个原函数 AST、两个 32 轮 CPU 编译出口案例、IDS 原源 consumer 的 CPU mock，以及实际 Linux 普通 child 的 normal/failure 清理。正式容器同版本 PyTorch 2.10.0+cpu 的 16 项全部通过，SDK/NPU import、动态 NPU library、NPU calls、skip 全为 0。CPU mock 与 AOT 不能替代真实 NPU graph/opaque-kernel 顺序证据。

依赖须按精确 frozen file 集合传入：v9 10 文件（含规范化两 job 原证据）、v8 70 文件、IDS prepack 18 文本 + frozen_files.json + delivery.json。原双 SDK 大 rank 文件无需再次传入；v9 pair 仅作为先前硬件能力前置，不能授予当前消费者覆盖。

`provenance.candidate_config(model, output, audit_enabled=...)` 生成 JSON 候选 seed。source_root、相对其根的 source_sha256、四个 pythonpath、原 contract startup_env_basis（audit:false）和实际 wrapper env 保留。`audit_enabled=False` 提供原生 math/worker/mp 配置供独立质量入口，仍不授予质量或性能收据。

后续工作以 freeze.json 的 strict_uncovered_points 为准。首次冻结后本目录只读；若真实诊断要求修改入口，应建立下一版并保留本次原始证据。
