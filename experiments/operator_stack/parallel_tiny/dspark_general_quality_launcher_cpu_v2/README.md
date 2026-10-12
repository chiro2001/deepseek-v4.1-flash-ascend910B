# 通用质量 API：源码快照与完整入口的 CPU 准备

本包接续已冻结的 `dspark_general_quality_api_cpu`，提供实际候选导出、新源码快照、宿主机 fresh TP8 启动、精确进程判活、原严格评测和可选安全停服入口。当前 **14 项 CPU 测试 / 59 个拒绝反例通过；NPU、HTTP、服务、真实信号均为0**。CPU 模拟输入覆盖候选/快照流水线，模拟输出明确携带 fixture 标识，实际导出与启动门会拒绝它们。qkv候选首diagnostic代码已冻结，但实际source snapshot/质量合同尚未生成；本包没有将native历史回执或CPU fixture冒充最终已加载候选。

`evidence/actual_asset_paths_CPU.json` 是容器中20个实际文件的只读归属/SHA：原3个 evaluator、官方 encoder 文件、carrots/corn 原图及 canonical GSM train/test。官方目录确认为 `/home/l00886679/models/DeepSeek-V4.1-Flash`，encoder 为其 `encoding`，图片为其 `inference/examples/images`；原 client/data 为 `/work/client`、`/work/client_data/gsm8k`。旧 A1 client sh 只被读取，没有执行。后续先核这20个冻结 SHA，再核实际目录成员和官方 owner。

`evidence/actual_checkpoint_metadata_CPU_reference.json` 只读核验原完整 checkpoint `/home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq`：90 shard/symlink closure、full40/W4A8/Engram [1,14]、原 DSpark [37,38,39]/K5、固定 config/tokenizer SHA，以及266个 vision/aligner/image +4718个 MTP 的实际 safetensors header，共4984个。只读取 header，没有读取 tensor payload、加载权重或哈希完整 shard 内容。历史 native snapshot 仅提供原 CPU checker 的冻结源码；这里没有导出候选、观测实际 SDK/graph/消费者或继承任何 actual/formal grant。

## 文件接口

- `export_candidate.py`：原 candidate seed + 明确 pin SHA 的 source/SDK manifest → CPU读取实际源、SDK/library文件、原模型 metadata → 新实际源码合同。保留原 model_kwargs、audit=false、完整 startup_env 和 SDK pair provenance；拒绝任何显式 CPU fixture、其他 source/SDK owner 或 inherited grants。
- `prepare_snapshot.py`：从实际合同复制原 `stack/operator/up950` 为新 snapshot；追加三个被冻结包、固定 v4 counter 和原3个 evaluator。已有路径内容不同即拒绝，原 candidate source map逐SHA保持。文件软链关闭成复制的字节，目录软链要求先由原候选准备成闭包；不复制 safetensors。source paths 可迁移，model/SDK原 owner不迁移。新合同/source manifest覆盖所有复制文件，没有权重或 native code 修改。
- `launch.py`：宿主机默认只出计划。未来 `--execute` 先核 host/container source、完整 checkpoint/SDK/assets/官方7 API source、容器 mount/8 physical chips、未占用端口、新 job，再执行两次独立 fresh occupancy/health 门。默认只接收 OK；只有 root 明确分配 `--allow-known-alarm` 才接受 `80C98001`，占用和未知错误始终拒绝。生成新 runner/evaluator，实际服务用独立 `setsid` runner 和 `python -m ...bootstrap` 启动，原一般 API进程所有权检查可直接匹配。
- `evaluate_when_ready.py` / `lifecycle.py`：只在 root 的未来实际 job 内执行。engine初始化不算 ready；ASGI 真正启动后才GET identity及唯一models，绑定同job/candidate/config/source/nonce、8rank inactive v4/base-worker实际源、PID/start ticks/boot ID/UID/parent/PGID。然后执行已冻结原 orchestrator，严格100/100、23/23；结果前后同nonce。可选 `--stop-after` 只向这一个精确PID发TERM，复核PID身份后等待原native退出与selection stopped，不向整个group或其他job发信号。
- `cpu_entry.py`：分开交付的包尚未进入标准 PYTHONPATH 时，可用此单文件直接绑定冻结 general/launcher 目录。它只加载 stdlib CPU模块；bootstrap仍由新snapshot内标准 `python -m` 执行。

## 最終候选 seed 必须来自真实冻结文件

qkv/root 提供 `candidate_seed.json`（也可使用其完整 `candidate_contract.json`），最低为原 `candidate_id/model_kwargs/checkpoint_config/sdk_pair_provenance/audit_enabled=false`、完整 `startup_env/pythonpath/startup_env_basis`；已有 `source_root/source_sha256/sdk_source_manifest_sha256` 会与传入 pin 严格比较，不默认为 native/tp8base。startup_env_basis固定引用 `parallel_tiny.dspark_formal_prep.contract.startup_env` 的真实 path/SHA、audit=false、draft_graph原值。startup_env须含其全部原函数开关和实际 wrapper环境，包括 `STACK_SRC_ROOT`、两项ASCEND visible和physical chips=8..15。model/tokenizer必须仍属上述完整原checkpoint；full checkpoint config/tokenizer固定SHA门不接受简化或假数据。

三处导出/快照/宿主机启动的 `--execute` 含义不同：前两者只落地CPU文件，最后一项会占用设备并启动服务，必须由 root 在新分配窗口执行。本轮没有执行任何 launch/evaluator/stop 分支。

## 已分配 job 的命令顺序

新包部署到 `/work/dspark_general_quality_launcher_cpu_20261011_v1/`；general包为 `/work/dspark_general_quality_api_cpu_20261011_v1/`，manifest SHA固定 `5537b3ed8685e2c73c971f1eb73910ffa67be4dd499944b53ff9cb83c803ea9a`。从新包 `delivery.json` 读取 launcher manifest SHA。下面 `<FINAL_...>` 只由 qkv/root 的最终冻结交付填入，不能用本轮任何 CPU fixture 替换。

优先使用以下**一条CPU入口**。它自动读取本包delivery pins、已核实的真实assets、原API20/v4、最终seed的source root/map和SDK manifest pin，完成实际合同导出与新源码快照。若seed仅给SDK SHA而未给path，追加 `--sdk-manifest <FINAL_SDK_MANIFEST>`；source map如直接嵌在已pin的seed中，seed自身就是冻结manifest；如引用独立manifest则使用其原path/SHA。命令不会启动服务或占用TP8。

root 的 full43 snapshot manifest 固定为 `/work/src_dspark_full43_diag_v1/full43_snapshot.json`，下列入口将它显式pin为基线。该manifest列出的是精确继承/新增stack文件（v10/v9/v8/prepack、旧base冻文件、root3件）的SHA，**没有全量SDK、operator/up950或weights内容列表**。qkv seed的pkg map只需是此列表的一致子集；本包之后在同机复制source时才将所复制的stack/operator/up950文件形成新的列表；SDK只核manifest指向的实际源/library文件，weights只核metadata/header，不授全内容哈希或设备加载。

```bash
python /work/dspark_general_quality_launcher_cpu_20261011_v1/prepare_final_entry.py \
  --candidate-seed <FINAL_QKV_ACTUAL_CANDIDATE_SEED> --candidate-seed-sha256 <FINAL_SEED_SHA> \
  --source-manifest /work/src_dspark_full43_diag_v1/full43_snapshot.json \
  --source-manifest-sha256 <ROOT_FULL43_SNAPSHOT_MANIFEST_SHA> \
  --output /work/src_dspark_general_quality_v1 --execute
```

下列分步入口保留用于审阅和独立调试，和上述CPU入口使用同一实现。

```bash
# 在容器中，只准备新实际 source contract；无后端加载。
python /work/dspark_general_quality_launcher_cpu_20261011_v1/cpu_entry.py \
  --general-package /work/dspark_general_quality_api_cpu_20261011_v1 \
  --general-manifest-sha256 5537b3ed8685e2c73c971f1eb73910ffa67be4dd499944b53ff9cb83c803ea9a \
  --launcher-package /work/dspark_general_quality_launcher_cpu_20261011_v1 \
  --launcher-manifest-sha256 <LAUNCHER_MANIFEST_SHA_FROM_DELIVERY> \
  --action export_candidate -- \
  --seed <FINAL_CANDIDATE_SEED_JSON> --source-root <FINAL_CANDIDATE_SOURCE_ROOT> \
  --source-manifest <FINAL_FROZEN_SOURCE_MANIFEST> --source-manifest-sha256 <FINAL_SOURCE_MANIFEST_SHA> \
  --sdk-manifest <FINAL_SDK_MANIFEST> --sdk-manifest-sha256 <FINAL_SDK_MANIFEST_SHA> \
  --output /work/general_candidate_exported_v1.json --execute

# 在容器中，只复制 source。原API20可从已有native v2完整快照逐SHA读取。
python /work/dspark_general_quality_launcher_cpu_20261011_v1/cpu_entry.py \
  --general-package /work/dspark_general_quality_api_cpu_20261011_v1 \
  --general-manifest-sha256 5537b3ed8685e2c73c971f1eb73910ffa67be4dd499944b53ff9cb83c803ea9a \
  --launcher-package /work/dspark_general_quality_launcher_cpu_20261011_v1 \
  --launcher-manifest-sha256 <LAUNCHER_MANIFEST_SHA_FROM_DELIVERY> --action prepare_snapshot -- \
  --candidate /work/general_candidate_exported_v1.json --candidate-sha256 <EXPORTED_CONTRACT_SHA> \
  --counter-file /work/src_dspark_native_api_v2/stack/formal_dspark_counters.py \
  --asset-manifest /work/dspark_general_quality_launcher_cpu_20261011_v1/evidence/actual_asset_paths_CPU.json \
  --asset-manifest-sha256 154337c39a919f5590585afc119b01272ad07d1f60221b3e5e9946fdba315d7a \
  --dspark-api-counter-bridge-cpu-root /work/src_dspark_native_api_v2/stack/parallel_tiny/dspark_api_counter_bridge_cpu \
  --dspark-api-counter-bridge-cpu-manifest-sha256 d4e2a6f0e7298d8bf6fdc5d0ec5d1be1854b69d9f49492502dc9211d4d26816e \
  --dspark-general-quality-api-cpu-root /work/dspark_general_quality_api_cpu_20261011_v1 \
  --dspark-general-quality-api-cpu-manifest-sha256 5537b3ed8685e2c73c971f1eb73910ffa67be4dd499944b53ff9cb83c803ea9a \
  --dspark-general-quality-launcher-cpu-root /work/dspark_general_quality_launcher_cpu_20261011_v1 \
  --dspark-general-quality-launcher-cpu-manifest-sha256 <LAUNCHER_MANIFEST_SHA_FROM_DELIVERY> \
  --output /work/src_dspark_general_quality_v1 --execute

# 仅root分配新TP8窗口后，在宿主机执行；该命令将自动启动ready/evaluator。
PYTHONPATH=/home/l00886679/projects/dsv41-real-stack-tp8-20261010/src_dspark_general_quality_v1/stack \
python -m parallel_tiny.dspark_general_quality_launcher_cpu.launch \
  --root /home/l00886679/projects/dsv41-real-stack-tp8-20261010 \
  --source-dir /work/src_dspark_general_quality_v1 --snapshot-manifest-sha256 <NEW_SNAPSHOT_MANIFEST_SHA> \
  --container dsv41-real-stack-tp8-20261010-a321 --job-id formal_dspark_general_quality_v1 \
  --port 18574 --ready-timeout 1800 --stop-after --execute
# 只有已分配known Alarm例外才追加 --allow-known-alarm；未知alarm/任何占用拒绝。
```

独立输出为 `/work/results/formal_dspark_general_quality_v1/{service,client}`，主要回执为 `launched.json`、`general_quality_lifecycle.json`、`client/strict_quality_receipt.json`、`run.exit/evaluator.exit`。`launched.json` 的launch_requested不代表service_started或HTTP ready；严格质量回执只在实际所有原评测通过后设置质量true，消费者/性能/formal始终false。性能由另一独立v4 actual rounds job决定，本入口不引用meanA/A1换算或质量结果授≤20ms结论。
