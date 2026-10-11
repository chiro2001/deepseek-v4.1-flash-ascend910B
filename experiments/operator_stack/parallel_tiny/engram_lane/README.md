Engram 的 8 输出片恢复候选已被原 1 ULP 精度门拒绝，不能采用到正式 TP8。真实输入驻留重放 `resident_c4_v2` 退出码 3；原生 A/A 全部逐位一致，候选 39/48 组通过、9/48 组失败。精度失败后没有执行性能配对，`pairs=0`，没有吞吐或整网收益结论。

| 检查 | 结果 |
| --- | --- |
| 原 CPU/设备 hash 与 mask | 28/28 逐位一致 |
| 正式表查询行 CPU/NPU 原始反量化 lookup | 48/48 逐位一致 |
| 原生 projection/key/value/gate A/A | 48/48 组，192/192 输出逐位一致 |
| 8 片恢复 projection ≤1 ULP | 40/48 组 |
| 8 片恢复 gate ≤1 ULP | 44/48 组 |
| 全消费者原精度门 | 39/48 组，拒绝 |
| 性能配对 / 正式 TP8 | 未运行 / 不可采用 |

| 层 | 投影最大 BF16 ULP | 投影 >1 ULP 元素总数 | gate 最大 ULP | gate >1 ULP 元素总数 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 6 | 5 | 6 | 1 |
| 14 | 96 | 3 | 2 | 3 |

范围是正式 Engram 的真实 lookup、真实 BF16 wkv 和原始 gate 消费者的单芯片驻留重放。输入来自原 tokenizer 的四组 token 序列及构造的页表，包含 image barrier/mask；用原 `PagedNgramHistory.update` 从完整真实 INT8 表 lazy mmap 选行，再用原 dequant 生成 lookup。CPU 预读 fixture 为 2,103,683 字节，仅留远端。residual 是确定种子的合成 BF16 `[1,4,5120]`，幅度 `.001/1/100`，没有完整 tiny/backbone 激活、真实 TP8 通信或客户端请求重放。

模型来源为 40 层、hidden=5120、384 experts、top6、正式 TP8 的 `W4A8_DYNAMIC` checkpoint；只重放 Engram 层 1/14，执行 TP=1。Engram 自身权重为 BF16 `[25600,6144]`，输入 lookup 为 `[1,6144]`，每层 314,572,800 字节。8 片为 `[3200,6144]`，K 不变，原权重与分片重接的 payload SHA 相同。完整权重与各片在 NPU 都是 format=2、ND；完整 stride=`[6144,1]`。实测硬件为 a3-21 physical4（NPU2/die0），设备名 `Ascend910_9382`，`HCCL_DETERMINISTIC=strict`，原 gate chunk/hoist 均关闭。保持 K 并不足以保证当前 BF16 原生执行的 1 ULP 等价；没有采集具体 tiling 差异。

证据入口为 [compact summary](evidence/summary.json)、[完整精度结果](evidence/precision_result.json)、[原始 row 来源](evidence/fixture_receipt.json)、[权重分片字节证明](evidence/weight_receipt.json) 和 [获取清单](evidence/provenance.json)。主要 SSH 证据包 784,160 字节、补充只读源码包 22,057 字节；未传输权重、fixture 或大 trace。远端真实权重实体副本为 629,310,402 字节，整文件 SHA256 为 `8b200b31c3026fa07a129fe0c20ba24830c9859e094878984a3065829bfc1171`；fixture SHA256 为 `2993073c26dc65934464923a6988985146464cd3f38f2ac5a1d14dcce3b699c3`。

前置 chip3 初始化失败 507033，自己的 driver 日志明确显示设备侧目标空间不足；chip7 旧容器 SDK 失败为 TSD CreateSession/ret6，原因分开记录。转移到明确授权的 chip4 后 SDK 成功。首个 chip4 重放因复制软链而找不到权重，改 `docker cp -L` 实体复制并重新校验 SHA 后得到以上完整结果。compact evidence 随 lane 可提交，57 份完整来源/日志留在被 Git 忽略的 runtime/evidence/engram_lane；历史负结果均保留，未 reset、删设备文件、清 pagecache 或操作其他租户；最后 physical4 健康 OK、无运行进程。

复现用私有 CPU 导出、权重字节验证和 guard launcher：`export_engram_lookup_fixture.py`、`prepare_engram_weights.py`、`launch_resident_replay.py`。runner `resident_engram_replay.py` 先做原生 A/A，再做 projection/key/value/gate 原 1 ULP 门，只有通过才执行无审计 graph 配对。当前候选维持 reject，不再次启动 NPU。保留完整 MatMul 的提前/旁路方案及生命周期契约另见 [full_projection_design.md](full_projection_design.md)；它只是设计，尚未做新的 NPU 验证。纯 CPU [生命周期 ledger](prefetch_lifecycle_contract.py) 的 14 项顺序/反例自检通过，未验证真实 Event/reset/capture，不能作为 NPU 精度或性能证据。
