# 原生 general OpenAI 质量服务：CPU 准备包

本包实现独立通用质量 service/bootstrap 与严核 orchestrator。当前默认只生成计划并做 stdlib CPU 检查；**NPU、服务、真实 HTTP 均为0**。8项CPU测试、34拒绝反例通过，原GSM/Vision评分函数用真实源码AST执行，输入/身份/ASGI都是明确CPU mock。实际GSM100、Vision23、客户端质量、消费者、性能及formal receipt全部false。

本包没有修改旧 serve/client、原底层评测脚本、private bridge、prepack、qkv/full43或source weights。其目标是将来在root分配的新quality job中使用原生AsyncLLM和原FastAPI通用路由，而不是private armed window子类或三路由白名单。

候选接口与qkv v10约定为 `candidate_contract.json`：schema、candidate_id、source_root/source_sha256 relative map、model_kwargs、checkpoint_config(path/SHA)、sdk_source_manifest_sha256、sdk_pair_provenance、audit_enabled=false、完整startup_env/fingerprint/PYTHONPATH归属及startup_env_basis。未知优化候选不能静默换成tp8base。quality只换独立job/port/served-model、output/cache路径、audit0和派生describe-worker，其余原model_kwargs与原环境开关保留并记录fingerprint。模型名为 `dsv41-dspark-quality-<job_id>`。

`contract.py` 绑定同candidate real checkpoint/tokenizer/source、TP8+EP、DSpark5、原BF16/strict HCCL、FP32 reduction关闭、prefix关闭、full40/vision。`worker_adapter.py` 继承candidate原worker、原load/math/权重流程不变，只增加inactive counter描述接口；若原worker尚未装v4则装原冻算法，始终不begin任何window。8rank counter须精确v4 SHA `da845a5891557f21e80eac06d96aec1955263d1ae93afbc4d84e86dd9585dbec`、tag/pending均inactive、audit=false。配置/SDK/gate引用只作归属，不继承actual/formal grant。

`bootstrap.py` 使用实际官方 `build_async_engine_client` → 原生AsyncLLM/config → 原reset_mm_cache/shutdown，实际 `build_app` / `init_app_state` / `serve_http`。原FastAPI对象保留routes/state以及models/completion/chat/image/SSE/metrics，另加只读 `/private/dspark-quality/identity`，没有ASGI白名单壳。官方7个源文件完整SHA/函数片段见 `evidence/official_api_sources.json`，包括serve_http与原lifespan。

engine/rank初始化完成仍记录 `service_started=false`；只有包装原ASGI lifespan真正完成startup后才记录started/ASGI_startup_completed，关闭时置stopped。HTTP_readiness_verified仍独立false，不能把ASGI布尔值当实际请求通过。未来orchestrator必须实际GET身份及唯一 `/v1/models`、核对PID/bootstrap job、同service_instance_id/candidate/config/source、前后8rank inactive；原private v1的提前started事实不沿用。

`orchestrator.py` 默认只打印具体命令计划，未来 `--execute` 要求全新输出目录，不复用任何A1/其他candidate的`.jsonl`断点。底层脚本不改、不导入（acc_eval顶层会运行HTTP），只在实际job通过subprocess执行原文件并核其SHA：

- GSM：官方canonical train前8条组成8-shot，test前100，官方encoding的chat模式，conc1、512预算、temperature0、SSE与原prefill序列化。固定canonical train/test SHA；要求原始summary `n=correct=100, empty=errors=0`，再按原extract/norm函数重算全部原始答案和canonical gold，不只信退出码或ok字段。
- Vision：原20 image问题+3negative，原官方图片/blank/data URI、SSE（保留原SSE失败后非流式fallback）、原32预算与额外4-token image/text probes。直接调用原vision脚本，min-hit-rate=1.0只提升验收门，不改请求；严核23/23、raw PASS、returncode0、empty/error0、mandatory negatives，并按原case/scoring重新核原始答案。
- 不使用默认80%/wrapper19门、A1换算、Vision meanA或累计spec metrics授质量/性能通过。性能仍由独立private bridge/v4 actual rounds job负责；质量job的counter保持inactive。

未来已分配job的调用形态（本轮没有执行）：

```bash
python -m parallel_tiny.dspark_general_quality_api_cpu.bootstrap \
  --candidate /work/final_candidate/candidate_contract.json \
  --job-id formal_dspark_quality_v1 --port 18574 \
  --output /work/results/formal_dspark_quality_v1 \
  --counter-file /work/frozen_v4/formal_dspark_counters.py
# 默认只写CPU plan；只有后续真实分配窗口才加 --execute。

python -m parallel_tiny.dspark_general_quality_api_cpu.orchestrator \
  --selection /work/results/formal_dspark_quality_v1/service_selection.json \
  --job-id formal_dspark_quality_v1 --base-url http://127.0.0.1:18574 \
  --client-root /work/client --encoder /path/to/official/encoding \
  --images /path/to/official/inference/examples/images --official-dir /path/to/official \
  --gsm-data /work/client_data/gsm8k --output /work/results/formal_dspark_quality_v1/client
# 同样默认计划；实际 --execute 使用新client目录。
```

当前最终candidate尚须由root/qkv冻结和实际加载，官方encoder/图片目录须在该实际job绑定；本包未加载权重、未跑general API，也没有产生模型答案。CPU样本 `candidate_contract_CPU_fixture.json` / `CPU_example_plan.json` 只用于协议审阅，不能作为实际candidate或服务selection。source、身份、生命周期与评分CPU门不授真实vision/语言能力或正式≤20ms结论。
