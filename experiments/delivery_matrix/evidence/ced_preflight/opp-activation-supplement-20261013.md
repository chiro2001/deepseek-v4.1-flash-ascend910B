# BI OPP 加载链与树外配置补充（只读）

采集日期：2026-10-13。此补充聚焦“新 P 容器只只读挂载 `opp/vendors/batch_invariant` 并设置 `ASCEND_CUSTOM_OPP_PATH` 是否足够”。没有执行 A3 `.run`、没有导入 Torch、没有读取模型权重，没有在 P/D 容器启动服务或 workload，也没有复制/修改任何 OPP 文件。

## 现场边界与直接比较

目标现场通过 SSH alias `a3-21`（`l00886679@192.168.45.21`）查询，目标容器为 `dsv41-real-stack-tp8-20261010-a321`，ID `b1aa15d8800346aff8f755a049feaae148b9d6e315e92bd32dcdf1063b429eb0`。`docker inspect` 显示其 image ID 为 `sha256:b69e58820f929ac7a77f93774a3d96da1961f507628239e9c7d7b24066b7a8ea`。以下输出只保留目标路径、类型/大小、哈希及 `batch_invariant` 相关配置行；没有输出完整 inspect JSON、环境变量或配置文件全文。

| 检查项 | 目标容器当前状态 | 固定 base image 状态 |
|---|---|---|
| `/usr/local/Ascend/cann-9.1.0/opp/vendors/config.ini` | `realpath -e` 为该路径本身；regular file，30 B，mtime `2026-10-12 11:31:32.241150210 +0000`；mode/UID/GID/size=`755,0,0,30`；SHA-256 `7473a624041a7281607d1a988a826c67f14fb203dc36f161f8f33b9ff9a8b4a5`；唯一相关行 `load_priority=batch_invariant` | 使用相同 image ID、`--read-only --network none` 的一次性 shell 检查返回 `BASE_IMAGE_CONFIG_MISSING` |
| 覆盖 `vendors/config.ini` 的 bind mount | 对 `docker inspect` 的 `.Mounts` 在本机仅筛选 destination 后，结果为 `NO_BIND_MOUNT_COVERS_OPP_VENDORS_CONFIG` | 不适用 |
| `opp/vendors/batch_invariant` | 既有只读 preflight 记录其位于当前容器 writable layer，而非 bind mount；树清单为 720 entries、617 regular files、222,548,126 B，tree manifest SHA-256 `3a8b878c9e8f2cd370e881dc248fd05e9bbbf07e7317a0848069b624179feb4a` | 既有固定镜像只读检查记录 `BI_OPP_MISSING` |

`config.ini` 定向检查命令族为 `ssh a3-21 docker exec <container> realpath/stat/sha256sum/wc/grep ... /usr/local/Ascend/cann-9.1.0/opp/vendors/config.ini`。配置内容只用 `grep -ni batch_invariant` 查看；没有 `cat` 全文件。base image 检查使用 `docker run --rm --read-only --network none --entrypoint /bin/sh sha256:b69e58820f929ac7a77f93774a3d96da1961f507628239e9c7d7b24066b7a8ea ...`，只检查该文件是否存在。

曾尝试对现有目标容器执行全量 `docker diff` 并在本机过滤 CANN 路径；该只读命令超过 60 秒仍无输出，按现场约束停止了**本地 SSH/docker CLI 客户端**。没有停止 container/daemon；这次全量 diff 未产生可用记录，故不把它作为结果。上表的 `config.ini` 对比来自精确目标文件检查、base-image 只读检查及 mount destination 过滤，不依赖该 diff。

## CANN 9.1 的 `config.ini` 与环境变量语义

与容器 CANN 9.1.0 对应的官方 CANN 9.1 文档给出以下契约：

- [Operator Package Deployment](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/programming_guide/advanced_programming/aclnn_operator_development/operator_package_compilation/operator_package_deployment.md)：默认安装包落在 `${INSTALL_DIR}/opp/vendors/<vendor_name>`；指定路径安装时由 `set_env.bash` 把 vendor 安装路径追加到 `ASCEND_CUSTOM_OPP_PATH`。在同一个 `opp/vendors` 下存在多个 vendor 时，`opp/vendors/config.ini` 用 `load_priority=vendor1,vendor2,...` 指定优先级。
- [ASCEND_CUSTOM_OPP_PATH](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/maintenref/envvar/envref_07_0147.html)：该环境变量指定 custom operator package 路径；GE 在图构建与执行期间据此定位、加载 operator binary，允许用冒号分隔多个路径。
- [Error “Failed to Open config.ini” Is Reported When Calling an Operator](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/programming_guide/appendix/faq/unable_to_open_config_ini_during_operator_call.md)：CANN 9.1 的 FAQ 将 `*/opp/vendors/config.ini` 不可读与 custom operator 的 `binary_info_config.json` 路径/读取失败联系起来；该文件默认权限问题也可能导致 operator call 失败。

因此，vendor 子目录与父目录的 `config.ini` 是两个独立部署对象。当前镜像没有这个父目录配置文件，当前 a3-21 容器却有一个单行注册/优先级配置，且无 bind mount 覆盖它。只把 `batch_invariant` 子目录挂入新容器，会丢掉当前容器已观察到的 `load_priority=batch_invariant` 状态；CANN 9.1 文档还明确提示缺失/不可读 `config.ini` 可影响 operator 元数据发现。即使具体调用路径恰好可绕过该配置，现有证据也没有证明这一点。

## BI Python wrapper 与 AscendC 扩展是分层加载

依据实际 vLLM-Ascend checkout HEAD `e43cf1e9f5d9bead076853aa6bcacb671465de94` 的 git object（以此前 a3-21 source identity 记录为准），只读取源码文本，没有 import Python/Torch：

1. `vllm_ascend/worker/worker.py:63-73,1103-1106` 导入并调用 `init_batch_invariance()`。`vllm_ascend/batch_invariant.py:31-47` 在 `HAS_TRITON` 时导入 `ops/triton/batch_invariant/matmul.py` 和 `softmax.py`，另外独立尝试 `import batch_invariant_ops`；ImportError 只会将 `HAS_ASCENDC_BATCH_INVARIANT=False`。
2. `batch_invariant.py:107-137,149-162` 在 `VLLM_BATCH_INVARIANT` 打开时注册实现：addmm/bmm/softmax 使用 Triton；AscendC extension 存在时 mm/matmul 和部分 torch_npu 算子走 `torch.ops.batch_invariant_ops`；extension 不可用但 Triton 可用时，mm/matmul/linear 走 Triton fallback。注意，实际 HEAD 中 `rmsnorm.py:135-153` 虽定义 `rms_norm_batch_invariant`，但精确 HEAD 的 `git grep` 除该定义和文档外没有 loader/registration 调用；BI loader 也未导入该模块。AscendC 分支的 `torch_npu.npu_add_rms_norm` 绑定到 `batch_invariant.py:50-62,123-124` 的 `add_rms_norm`，其内部调用 `torch_npu.npu_rms_norm`，不是 Triton `rmsnorm.py` wrapper。纯 Python wrapper 文件本身不是 OPP package，也不读取 `ASCEND_CUSTOM_OPP_PATH`。
3. `vllm_ascend/platform.py:282-296` 的 `NPUPlatform.import_kernels()` 调用 `bootstrap_custom_op_env()`。`vllm_ascend/utils.py:315-332` 显示该 helper 会在默认自带 vendor 路径存在时将其前置到 `ASCEND_CUSTOM_OPP_PATH`；仅在 `include_vendor_lib=True` 时才将 `op_api/lib` 加入 `LD_LIBRARY_PATH`。同一 `utils.py:413-453` 中，`VLLM_BATCH_INVARIANT=1` 会使通用 `enable_custom_op()` 返回 false、不导入 `vllm_ascend_C`；这与 `batch_invariant.py` 对独立 `batch_invariant_ops` 的探测是两条不同路径。
4. `csrc/build_batch_invariant_ops.sh:55-98` 把 BI `.run` 安装与单独的 `batch_invariant-torch_ops_extension` zip 下载、解压、构建/安装分开处理。故 OPP vendor tree 不等同于可 import 的 `batch_invariant_ops` 扩展。当前任务没有对 base image 查询该 Python module 是否已安装，也没有 import 它；不得据此报告 AscendC branch 现在可用。Triton fallback 的源码存在，不等于它覆盖所有项目所需算子或达到同一性能/功能契约。

可复核的源文件 SHA-256（从上述不可变 git object 导出）：

| vLLM-Ascend 相对路径 | SHA-256 |
|---|---|
| `vllm_ascend/batch_invariant.py` | `a1cfbb025db53cc4cff5dea33955aead1349082c9382b222d608cd2ec52edc57` |
| `vllm_ascend/utils.py` | `c42bc5a933fc2f3c3e978e71ae82f0a2054d4e3dd8dce361c6214d7ef64f69c4` |
| `vllm_ascend/platform.py` | `f56a8ac217088d381fde5310a0ccaf2496ed013b4becb6d53388f473d453bfb6` |
| `vllm_ascend/worker/worker.py` | `2eb424498fd1a908dece44a56451a2e42bb4697614327aaa91aaa171e3f0f127` |
| `csrc/build_batch_invariant_ops.sh` | `6c82a542829adbd04cc49a90e4d4b3889850a00765672a5e406dcb4aa4aa33ac` |

四个 Triton wrapper（matmul/mean/rmsnorm/softmax）的 SHA-256 已在父证据 [`image-and-opp-readonly.md`](image-and-opp-readonly.md) 与目标容器/base image 对照；它们随 Python 源码位于 `vllm_ascend` 包，而不是 `opp/vendors/batch_invariant` 中。

## 给 site_runtime 的可执行结论

**当前证据下，不应把“只挂载 `opp/vendors/batch_invariant` + 设置 `ASCEND_CUSTOM_OPP_PATH`”当成可复现、已激活 BI OPP 的充分条件。** 具体原因是：固定镜像同时缺少 BI vendor tree 和其 sibling `opp/vendors/config.ini`；现有容器里另有 30 B 的 priority 配置，且 CANN 9.1 文档明确使用该文件控制 vendor 优先级并记录其不可读会导致 operator 元数据读取失败。BI `batch_invariant_ops` Python 扩展也属于单独的可选安装物，当前 base image 的存在性尚未核实。

在不改旧 D 的前提下，新 P 的冻结准备应将以下项拆开验证并记录：

- OPP vendor 目录逐文件 manifest 与四个关键库 SHA（不得只检查目录存在）；
- sibling `opp/vendors/config.ini` 的 realpath、SHA/内容、权限与只读挂载，若选择不挂则先获得明确成功的 loader/registration receipt 来证明不依赖它；
- 实际 `ASCEND_CUSTOM_OPP_PATH` 最终值与 CANN 版本；
- `batch_invariant_ops` Python module 是否已由镜像提供，以及本次需要走 AscendC 还是可接受 Triton fallback。核实 module 时仅做无 Torch import 的文件/发行包查询；真实注册/算子执行 receipt 留到获准启动的新 P 容器后。

本次读取到的当前 a3-21 `config.ini` 可作为只读复制源，但**本轮没有复制**它；site_runtime 若独立执行复制/挂载，应保留源与目标哈希及只读属性证据。完整 `.run` 安装脚本/文件副作用仍未解包验证，因此也不能断言该配置一定由本 `.run` 单独创建；能确定的是它不在 base image、不由现有 bind 覆盖，而是在当前容器路径上实际存在。
