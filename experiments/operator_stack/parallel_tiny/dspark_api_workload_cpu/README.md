# DSpark 2K/256 工作负载：真实 tokenizer 的 CPU 准备

已使用正式 checkpoint 的真实 `tokenizer.json` 和已有正式 client 的《红楼梦》片段准备 **2 warmup + 8 measured**，全部 prompt 精确 2048 token、输出预算 256，10 个上下文互不重叠、10 个 prompt SHA 各不相同。末尾为原有四种短切片可答任务，追加约150字的回答上限；问题没有被截断或用 filler 填充。

输入来自 `/home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq`，真实 config、tokenizer、tokenizer_config 与 symlink resolved path/SHA 已记录。config 的 full40/5120/384/top6、DSpark block5 和 vision 配置只作 CPU 来源核验；本轮没有读取模型权重或运行模型。

准备阶段直接用 tokenizers Rust CPU 后端编码完整文本，检查 decode/encode 往返、原始语料的 byte interval 和完整问题尾部。独立 parity 进程再以原 checkpoint 的 AutoTokenizer 检查全部 10 个输入：token IDs 完全相同、原文 decode 完全相同、2048 预算及问题尾部一致。该进程允许 CPU torch；backend autoload=0，NPU/CUDA/HIP visible 全空，import finder 拦截 torch_npu/vllm/ACL/torchair；实际没有 NPU SDK/服务导入、NPU runtime library 映射、NPU 调用或 HTTP。未安装或修改任何服务。

原 tokenizer 6,367,257 B、原语料 2,468,721 B 均留在远端。只导出本次 10 个必要片段：原文区间共 78,527 B，约原语料的 3.18%；没有导出完整原语料、tokenizer 或模型。每个交付文件和压缩包都小于 1MB。

`cpu_mock=false` 表示输入和 tokenizer 来自真实文件；`cpu_preparation_only=true` 明确表示这是 CPU 准备。所有实际模型、HTTP、客户端质量、性能、device consumer 与 formal/performance admission 标志仍为 false。可答性只依据可见片段中的句子/人物见证，尚未产生任何模型回答或语义准确率结果。

主要入口：

- `evidence/prepared_inputs/workload.json`：可直接交给已冻结 acceptance_client 的真实输入清单，最终 SHA `8c6a1a4ea2acbb5c72706a21870ac25fd71770d2af4c26b81238ccf043d82572`。
- `WORKLOAD_MANIFEST.md`、`evidence/prepared_inputs/prompt_audit.json`：逐 prompt 的 phase、长度、预算、问题、source span、token/text SHA 与可答见证；warmup 独立，不进入 8 个 measured 清单。
- `evidence/prepared_inputs/AutoTokenizer_parity_receipt.json`、`final_preparation_receipt.json`：最终独立 CPU parity 回执。初次 Rust 阶段的 `preparation_receipt.json` 保留当时尚未做 parity 的事实，最终状态以这两个回执及 workload provenance 为准。
- `evidence/local_frozen_client_compatibility.json`：10 个输入全部被原冻结 client 与 private fixed-request 契约接受；没有发送 HTTP。
- `QUALITY_API_READONLY.md`：现有 GSM100/100、Vision23/23 的 DSpark/A1 假设与新 general API 的最小接线步骤；底层脚本未运行，旧文件未修改。
- `prepare_remote_cpu.py`、`check_AutoTokenizer_parity_cpu.py`：可复现的 CPU 源码，原输入与源码 SHA 保留在回执/冻结清单。

远端 CPU 输入在 `/work/dspark_api_workload_cpu_20261011_v1/`；local workload 内的 text_path 指向这个远端原位，当前已在同一容器逐文件校验。直接使用 token-ID list、add_special_tokens=false；将来真实 OpenAI renderer 是否保持同一 prompt ID/hash/长度，仍由已冻结 API bridge 的提交 tap 与最终输出检查判定，不能用本包的 CPU 成功代替实际 API 收据。

已冻结 API 桥及旧 M1/K5/context 包保持不动。私有文本 performance 服务拒绝公开 chat/stream/多模态，不能代替 GSM 或 Vision 的 general API 质量门。
