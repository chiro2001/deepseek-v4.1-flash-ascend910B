# baseline2 CPU-only review diff

状态：仅生成本地 CPU 配方/计划，未在 D 写入 jobroot、未复制 launcher、未执行 launcher、未初始化 NPU。CPU plan 的 `service_launch_requested=false`；launcher 文件中保留 `--execute` 是冻结启动命令内容，不代表已执行。

## 唯一预期差异

- 启动器以已审核的 base1 launcher 为模板，逐字只替换 D 任务路径中的 `jobs/base1` → `jobs/base2`。与 base1 做同样字面替换后的 `diff -u` 为空（exit 0）。保留原 extended `unset` guard、served-model、port、source/package、所有 startup flags。
- jobroot=`/home/l00886679/tmp/dsv41-operator-revalidation-20261013/jobs/base2`；output=`jobroot/service`；native=`service/native`；cache=`service/cache`；`DELIVERY_TMPDIR=jobroot`。路径预算是计划值：jobroot 67 bytes、`/UUID` socket pathname 104 bytes + NUL = 105/107；必须在实际启动前重新做 D 侧 write/IPC bind 检查。
- base2 CPU service plan 的 `model_kwargs` 与 base1 完全相等；source candidate fingerprint、form、served-model 也相等。环境仅两项路径值变更：`STACK_DSPARK_FORMAL_OUTPUT` 从 `base1/service/native` 到 `base2/service/native`，`VLLM_CACHE_ROOT` 从 `base1/service/cache` 到 `base2/service/cache`。因此 plan fingerprint 从 CPU-only 值 `3311e9928ccc7950d38337f420d7d9ab7319b2169ba67722dae63d56f611755a` 变为 `79513a0f20932cf696374573323b715aab0d04df774721fbee92f2f780da4319`；这不是实际 runtime fingerprint。
- `base2_prepare_launch_cpu_plan.json` 与 base1 的唯一区别也是 literal job path 从 base1 到 base2；`DELIVERY_TMPDIR` 保持 jobroot 根路径，不是过长的 `jobroot/tmp`。

## 冻结输入与产物 SHA-256

- frozen generic-v7 `prepare_launch.py`: `700e2a86ff9af5b9f16d937fbf4369f86babfcc4a8ec5d8ffe8116c088eacdfa`
- frozen generic-v7 `service.py`: `c464aceae5fa789027334dc6222cfa47112bcf20ae700f1d4ac8c95d6c6f13c9`
- source candidate contract: `5ed2e17c2057caee65a17195a4e13f92187b3c8e93dd03b63de8269b3c745cf2`
- `base2_launch.sh`: `ef04c733256a907c7b7fd834e0e5aba742246193dbfffcc2277ccb6a4d60cfb1`
- `base2_prepare_launch_cpu_plan.json`: `bc30a7d392dfbe38ff66083df3821b63649d804a804cccfc1c96a92c526cc62b`
- `base2_service_plan_cpu_only.json`: `1f501e25874c055239cdd6661d9a3129b7f46e7e401e1f6b4349bd88025b4b27`
- `base2_env_whitelist.json`: `2860d1a0770b6a4893ad26e1b176292c885fc0c0458d50633c5c8a3361cc4515`

## Root review required

请先审核上述路径差异、launcher SHA 与 CPU service plan SHA。获审前不写 D base2 jobroot、不做实时 NPU/port gates、不启动服务、不发 API 请求；base1 所有原件不变。
