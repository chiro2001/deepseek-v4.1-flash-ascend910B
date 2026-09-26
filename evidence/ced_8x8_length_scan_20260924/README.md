# A3-21 8+8 CED-PD 故障对照（2026-09-24）

## 服务与探针

- 当前旧实例在归档时仍运行：P `dsv41-ced-p-1m-20260924`（芯片 0–7，HTTP 18990、KV 19090），D `dsv41-ced-d-1m-20260924`（芯片 8–15，HTTP 18991、KV 19091），proxy 18992。P/D `/health` 和 proxy `/healthcheck` 均为 HTTP 200；最后一次短针后日志为空闲。容器 inspect、进程环境、完整日志见本目录。
- 真实模型 `v41-flat-verify3`，TP8+TP8、BF16 KV、`MAX_LEN=1048576`、`MAX_SEQS=4`、`BAT_TOKENS=8192`、`GPU_UTIL=0.92`、`SPEC=0`、`PREFIX=0`、Engram host、`ENGRAM_DEVICE_INDEX=0`、`STATIC_KERNEL=0`、`NPUGRAPH_EX=1`、DP1。
- 长针扫描使用同一远端 `ctx_probe_pd.py` 与四针语料，temperature=0、非流式、repeats=1、max_tokens=64。256K、8K、当前 144K needle 明确 `offset=0`；tokenizer 直连 P `18990`，推理走 proxy `18992`。
- 旧 520K/1M JSON 没有保存 offset。原 `bigprefill` 函数从 base offset=0 起，A/B 语料旋转分别为 1,000,000/1,146,000；旧 JSON/runner 未记 max_tokens，本次按默认 64 复测，需保留这一限制。远端 corpus SHA-256 `a7fc413bd6e3926482faddf2af9bfb4426e55d8e421481d60100c14160785578`。

## 当前实例结果

| 请求 | 实际 prompt tokens | 结果 | U+FFFD |
|---|---:|---|---:|
| 8K needle A/B/C/D | 8,335 | 4/4 错，HTTP 200 | 2/1/0/2 |
| 144K needle A/B/C/D，offset=0 | 142,426 | 4/4 错，HTTP 200 | 6/1/6/2 |
| 原 144K bigprefill A/B | 144,404 / 144,131 | 2/2 错，HTTP 200 | 0/0 |
| 256K needle A/B/C/D | 255,527 | 4/4 错，HTTP 200 | 8/10/1/3 |
| 520K needle A/B/C/D | 517,036 | 4/4 错，HTTP 200 | 4/0/0/0 |
| 1M needle A/B/C/D | 1,019,789 | 4/4 错，HTTP 200 | 3/2/0/4 |

原成功实例上的 144K `bigprefill` A/B 分别在 144,404 和 144,131 token 下逐字正确。将同模式、同旋转位置重放到当前实例后，两条都变成混杂乱码，虽没有 U+FFFD。当前服务另做了两个极短数学请求和一个短针复述：Chat API 的 prompt 长度分别为 17、17、22 tokens；D replay 分别覆盖 `0..15`、`0..15`、`0..20`，全部 HTTP 200 但答案错误。短针是“校验码是 ZQ7K-3341。请只回复这个校验码。”，输出未包含校验码。

因此当前问题不限于超过 128-token 的 replay 起点：从位置 0 开始的 17/22-token 请求也答错。当前进程质量异常已确认；具体是 D 运行状态、P/D 交接、服务启动参数，或其他前向问题仍待时序对照。探针 `repeat_loop` 即使为 false，答错/乱码仍计失败。

## 旧实例与当前实例差异

- 旧通过 run 的 P 来自 `8b59d03` 包，D 来自 `3a45922`；当前 P/D 都来自 `3a45922`。模型路径相同：`/home/l00886679/models/out/v41-flat-verify3`。
- 旧 `MAX_LEN=147456`，当前 `MAX_LEN=1048576`。`BAT_TOKENS=8192`、`MAX_SEQS=4`、`GPU_UTIL=0.92`、BF16、GRAPH=1、EAGER=0、PREFIX=0、SPEC=0、Engram int8/host、CPU_BIND=0、STATIC_KERNEL=0、NPUGRAPH_EX=1 均相同；服务名和端口不同。
- 两个包的 `serve_a3_pd.sh` SHA-256 都是 `3e8243126f8c22f09d30ca0de71afa312f1897d6accb6231041cd5482fb778ce`，`serve_v2.sh` 都是 `cedb8eb7361e494daa879df7f029df4325308db096ea43082f08d5607b6fb3d9`；`model.py`、`dsa_v41.py`、`core_scheduler_replay.patch` 也相同。旧 P connector SHA `eebef48e...3a8d4305`，新 P/D connector SHA `a93bd605...8c3e3dfe`；唯一 diff 是缺页清零从 `index_fill_` 改成逐物理 block `narrow().zero_()`。旧 D 已使用新 connector 版本。
- 两轮 Available KV memory 都约 15.16 GiB，Mooncake blocks 旧约 30,083、新约 30,081。日志里的 `GPU KV cache size tokens` 会随 max_model_len 按 `max_concurrency(max_model_len) * max_model_len` 换算；不能用其数值推断实际缓存页布局发生变化。`MAX_LEN` 是明确的启动差异，但它与失败之间的因果待 fresh-instance 对照。
- 镜像 ID `sha256:1f2c08195c5b119aa9a107861fa2efa5d2b93b161f1ae91cb5358923a78cbcef`，镜像 digest `sha256:2c906b38ad3cc9ed9badfe3d09451d194fa09a07e5c6d62a66b111731e626060`。模型目录约 490 GB；未重算大权重 shard，保留完整文件名/尺寸 inventory 和 config/tokenizer/quant manifest 哈希，见 `model_file_inventory.tsv` 与 `model_metadata_hashes.txt`。

## 同一镜像 direct full40 对照

- 服务由同一 `3a45922` 包的 `scripts/serve_a3.sh` 启动，容器 `dsv41-full40-1m-doublefresh-20260924`，端口 18993、chips 0–7。模型 ID `deepseek-v41-full40-1m`，root=`/home/l00886679/models/out/v41-flat-verify3`，max_model_len=1,048,576。实际 process env 中 `V41_CED_ROLE=`、`KV_ARGS_EXTRA=`，container mounts 无 CED connector、`dsa_v41.py` 或 replay patch。
- 与上面 8+8 CED 同为 BF16 KV、Engram host/index0、MAX_LEN=1M、BAT=8192、MAX_SEQS=4、GPU_UTIL=.92、SPEC/PREFIX=0、STATIC_KERNEL=0、NPUGRAPH_EX=1、CPU_BIND=0、DROPCACHE=0。direct full40 初始短针 22 tokens 精确答 `ZQ7K-3341`；原 144K `bigprefill` A/B 也 2/2 精确通过，实际长度 144,404/144,131，U+FFFD=0。
- 同一 direct full40 的标准 1M `needle` 四针，target=1,020,000、offset=0、repeats=1、max_tokens=64、temperature=0，实际 1,019,789 tokens。A/B/C/D **4/4 未命中**且 HTTP 都为200，U+FFFD=0；A probe 抽取到 `J4y9K2`，B/C/D 的 `answer_repr` 为空字符串。逐条 wall 为 291.1/289.6/287.7/286.8 秒，总计 1,160 秒。没有保留每条响应的原始 API body，空串只表示探针抽取结果，不据此断言服务返回零 token。
- 四针之后，同一 direct full40 进程重发同一个 22-token 短针，仍精确答 `ZQ7K-3341`（wall 0.336 秒）。所以 direct full40 的短请求在1M扫描后仍正常，而1M needle检索失败。它与历史 AscendStore 全40层结果使用不同 DSpark/连接器配置，因此不能把两者当严格等价的1M基线。
- 当前 direct full40 容器保持运行，D/proxy CED 容器空闲。完整 1M JSON、runner stdout、short response 和 serve log 在 `full40_needle_1m_offset0.json`、`full40_needle_1m_offset0.log`、`full40_post_1m_short_needle_response.json`、`full40_serve_after_1m_and_short.log.gz`。

## 代码语义与待证因果

- **代码已确认的语义缺口：** `experimental/ced/dsa_v41.py` 的 D SMLA attention 将原始 `seq_lens` 传给 `seqused_ori_kv`，使用 `ori_mask_mode=4`、`ori_win_left=window_size-1=127`；scheduler patch 只把 replay 计算区间设为 `[replay_start,replay_end)`，attention mask 没有把可见 SWA 区间截到 `replay_start`。因此 replay 的首个 query 可能向前看至多 127 个未传输的上层 SWA 逻辑位置。connector 只清零 D 分配给 G7–G11 的 `local_ids`，代码没有证明这些逻辑位置都映射到刚清零且合法的物理页。
- 上述缺口尚未数值证明是乱码根因。17/22-token 短针 replay 从位置 0 开始，仍失败；仅凭长上下文结果不能把短针失败归给“replay_start 之前的历史 SWA”。需要父 Agent 的 pos254 逐值对照，并做新鲜 P/D 时序实验。

## 下一步时序实验

旧实例保持运行，尚未停止。完成本目录归档并由主 Agent确认后，先只重启 D+proxy、复用当前 P；ready 后第一条模型请求为 ≤22-token 短针。若通过，再跑原 144K bigprefill A/B；若短针仍错，则保留新 D 并重启 P，再做短针；只有短针通过才继续长请求。记录每阶段 engine ID、日志、prompt tokens 和 response。若 D-only 握手无法建立，再回退同时重启 P/D。

既有全 40 层对照 JSON 已复制到 [`baseline_pdstore_bf16_needle1m.json`](baseline_pdstore_bf16_needle1m.json)：AscendStore + BF16 在实际 1,019,789 token 下四针均精确正确，但连接器及 DSpark 配置不同，因此是强对照，不是严格单变量门。

## 文件

- 各阶段原始 JSON：`needle_8k_offset0.json`、`needle_144k_offset0.json`、`bigprefill_144k_offset0_repro.json`、`needle_256k.json`、`needle_520k.json`、`needle_1m.json`、两条数学短请求和 `ultrashort_needle_response.json`。
- `ctx_probe_*.log` 与 `bigprefill_144k_offset0_repro.log` 保存 runner stdout。`p_serve_final_before_restart.log.gz`、`d_serve_final_before_restart.log.gz`、`proxy_full_before_restart.log` 是停止前的完整当前实例日志；另存旧 144K 成功实例的完整 P/D 日志。
- `current_container_inspect.json` / `previous_container_inspect.json`、current/previous `inner.sh`、`serve_cmd.txt`、`driver.log`、实际 P/D 进程环境和镜像 inspect 保存启动证据。
- `doublefresh_*` 保留两侧fresh的 CED 首短针证据；`full40_*` 保留无 CED role/KV connector 的 direct full40 对照。
- `code_and_image_hashes.txt`、`model_metadata_hashes.txt`、`model_file_inventory.tsv` 保存代码、镜像及模型元数据指纹；未全量哈希 490 GB 权重。
- `SHA256SUMS` 汇总本目录证据文件校验和。
