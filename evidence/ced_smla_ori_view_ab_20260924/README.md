# A3-22 mixed-CSA original-view A/B: production `_C_ascend`

日期：2026-09-24。此目录记录一次**合成算子机制验证**，用于比较全局原始 KV 视图与 replay scratch-local 原始 KV 视图，同时保留相同 CSA 候选。它不是加载真实模型的精度测试，也不是对模型乱码的修复或生产路径变更。

## 隔离与加载路径

- 使用 A3-22 chip7（Phy-ID 7）上的既有容器 `dsv41-ced-tiny-d-snap127`，容器 ID `436ad3e94ebdf8774c749f5710becb5929cd8ec18dcf8cf43435c2b1dfd7ad3a`。
- 容器命令为 `bash -lc 'sleep infinity'`；仅映射 `/dev/davinci7` 和 Ascend 管理设备，`ASCEND_RT_VISIBLE_DEVICES=7`。没有启动 vLLM/model server。执行期间只有合成输入与算子调用落到 chip7。
- chip6 上既有 P 服务保持运行；没有使用 chip0/1。
- 通过生产注册路径 `vllm_ascend.utils.enable_custom_op()` 注册后，`vllm_ascend.vllm_ascend_C` 路径为 `/vllm-workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so`，哈希为 `855b3fe4b39c6680e3cf718e7cf2520b8e55e215c118037c6e5fe5194710579e`。配套 `libvllm_ascend_kernels.so` 哈希为 `44d171bbfa0314d4baac3e341b63097cef0f78822fff12d2113d6623f01c8f6a`。它们与 chip6 P 进程先前记录的哈希相同。
- 之前通过 `cann_ops_transformer.sparse_flash_mla_metadata` 调用的 adapter 因 `cmp_ratio=2` 被 CANN checker 拒绝；本次使用 `_C_ascend` 生产 op namespace，不改变参数、kernel 或 checker。`metadata_gate_fixed2.stdout.log` 保留旧路径原错。

## 合成输入与对照

两臂使用同一个 q、CSA KV、候选索引、block table、`seqused_cmp_kv`、残差和 query cumulative lengths；唯一差别是原始 KV 的序列视图和长度：

| 参数 | A：full global ori | B：scratch-local ori |
|---|---:|---:|
| batch / query length | 1 / 128 | 1 / 128 |
| heads / head dim | 1 / 1 / 512 | 1 / 1 / 512 |
| 原始 KV 长度 / table | 255 / `[[0,1]]` | 128 / `[[0]]` |
| replay 全局位置 | 127–254 | 127–254，压紧到 scratch offset 0 |
| CSA 长度 / residual / ratio / top-k | 127 / 1 / 2 / 512 | 相同 |
| mask / window | original=4, compressed=3; left=127, right=0 | 相同 |
| layout | `TND` / `PA_BBND` | 相同 |

q 在 dim2 为 1，所有合成 KV 的非零 payload 放在 dim0/1，故两臂的 logits 相同；输出 dim0/1 用来观察候选可见范围和 joint softmax。固定 CSA row126 的 dim1 payload 为 100。CPU 闭式参考覆盖 128 个 query rows，计入每行可见的原始窗口、CSA 候选数与 `sink=-100`。

## 结果

生产 `_C_ascend.npu_sparse_flash_mla_metadata` 的 A/B 两臂均通过，shape 均为 `[1024]`、dtype `torch.int32`；metadata SHA-256：

- A：`e824f2949577b9e0abc97e51cef821c336f195662fbbaf8362df641f6232d877`
- B：`57619b9e29820640c50f6f967008702a7c6d035854b8227bca2d67e27add0c47`

attention 按 A → finite 检查 → B 顺序执行；两臂均返回 finite、无 NaN、无 CANN/checker 错误。metadata + A/B attention + CPU 对照总墙钟时间为 **9.510431770 秒**。stdout 中只有 CANN/HDK 的 NPU caching allocator padding warning，没有编译日志或 Traceback。

`max_abs_vs_cpu` 是各 `[128,1,512]` 输出相对 CPU 闭式参考的**全张量最大绝对误差**（覆盖全部 128 行和 512 个通道；此脚本没有单独保存逐行误差向量）：

| 臂 | 全 128 行 `max_abs_vs_cpu` | row0 dim0 | row125 dim1 | row126 dim1 | row127 dim1 | 输出 SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| A | 0.001519607843137205 | 0.171875 | 0 | 0.392578125 | 0.392578125 | `367baa11ce10a3b6eaff584cd1cdf00d9aeb9f40b32365e5b5741fd2a34266d5` |
| B | 0.0015040106951871857 | 0.0230712890625 | 0 | 0.39453125 | 0.392578125 | `ce92f7ede61212fae3588e27e79ff781cc4ebcc2eff63507f33504b1ad984eac` |

CSA row126 的可见计数在 replay rows 125/126/127 分别为 126/127/127：row125 dim1 在两臂均为 0；row126 dim1 为 A `0.392578125`、B `0.39453125`，CPU 期望分别为 `100/255`、`100/254`；row127 dim1 两臂均为 `0.392578125`，CPU 期望均为 `100/255`。最后一行两臂相同，验证相同 CSA 与原始窗口共同归一化；row126 的小差异来自 A 原始窗口 128 项、B 局部窗口 127 项。全输出 A/B 最大差为 `0.1488037109375`，dim1 最大差为 `0.001953125`；两臂原始窗口不同，不能把全输出差解释成 kernel 错误。

attention 执行期间，npu-smi 看到隔离容器的 host PID `774206` 位于 Phy-ID 7，使用 129 MB；脚本正常退出后 Phy-ID 7 无进程。清理后容器停止以释放 chip7；chip6 P 未停止。

## 输入与复现产物

完整参数、全部公共输入哈希、metadata 哈希、A/B 摘要和边界行保存在 `smla_attention_c_ascend.result.json`。共有输入 SHA-256：

- `q`: `d5e544b6fdb895470ca66971c99fc9732462f9176dcdc03ee1d3c69d5ffadb06`
- `cmp_kv`: `82e895c2a4dc350ebf837ca1e72f454fb3f7e16b71d6b00ab24b9be53e1f0a93`
- `cmp_sparse_indices`: `964c10c14f333df97bdde5018acc719cfa467e1df2b7e3c3ba1af140b5ff3a8d`
- `cmp_block_table`: `df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119`
- `seqused_cmp_kv`: `8d12d9fcf1bb74eb413e34f28b96439b1260da900a2c70af2440c5c7dbb43c03`
- `cmp_residual_kv`: `67abdd721024f0ff4e0b3f4c2fc13bc5bad42d0b7851d456d88d203d15aaa450`
- `cu_seqlens_q`: `22f23252fcd4f43ed48d75ca5e7323df657fbff031410740d914a01433e7580f`

关键文件：

- 原始合成 harness：`smla_mixed_csa_ori_view_ab.py`（SHA-256 `dc8debab8e0db22a0e9e7e524e46b102e591bd3364fe85e771def05568d4a887`）
- 生产 namespace 的 metadata-only 副本：`smla_mixed_csa_ori_view_ab_c_ascend.py`（SHA-256 `527a58a680e8b92404cf05f49b42d1f3bf3f676fac661b8081dc63df39690c31`）
- 生产 namespace 的 A/B attention harness：`smla_mixed_csa_ori_view_ab_c_ascend_attention.py`（SHA-256 `1a661d7c10bc6bd855eebfeb3a6b0797124a518254ac7e490ad6f90af27c5fa7`）
- `metadata_gate_c_ascend.stdout.log` / `.result.json`：metadata-only checker 结果
- `smla_attention_c_ascend.stdout.log` / `.result.json` / `.metadata.json`：A/B attention 与闭式参考结果
- `container_after_c_ascend_attention.txt`、`npu_smi_after_c_ascend_attention.txt`、`so_hashes_after_c_ascend_attention.txt`、耗时及退出状态文件
- `container_after_cleanup.txt`、`npu_smi_after_cleanup.txt`：确认 sleep-only 容器已停止，chip7 无残留 NPU 进程。

本实验只表明生产 `_C_ascend` SMLA 在这些合成张量和 `cmp_ratio=2` 设置下通过 metadata 与 attention 机制对照。它不证明 CED 真权重精度，也没有把 scratch-local 逻辑接入模型服务。
