# `GroupedMatMulAllReduce` 单组路径：参数约束已解，A3 上可用

> 目标：用**已在 A3 上存在的**融合算子替代 `matmul + all_reduce`（当前 87 次/步、2.24 ms/step 全暴露）
> 结论：**参数约束已完全解出，算子在本机可选**（参数校验通过）；性能对比仍在进行
> 关键：`MatmulAllReduce`（cannbot 推荐的那个）在 A3 上**上游不支持**，但同族的
> `GroupedMatMulAllReduce` **支持**，可用「单组退化」达到同样目的。

---

## 1. 背景：为什么不用 `MatmulAllReduce`

`torch_npu.npu_mm_all_reduce_base`（cannbot 明确推荐）在 A3 上直接失败：

```
npu_mm_all_reduce_base: MatmulAllReduceBaseKernelNpuOpApi.cpp:276
  NPU function error: call aclnnMatmulAllReduce failed, error code is 161001
  EZ1009: Failed to execute operator MatmulAllReduce_10. Reason:
    1. SoC version ascend910_93 verification failed.
       This SoC is not configured through the AddConfig API of the OpDef class.
    3. The operator package to which the MatmulAllReduce operator belongs is not installed.
```

**根因（由 `opensrc_ops_hunt` 源码调研确认）**：`ops-transformer/mc2/matmul_all_reduce/op_graph/matmul_all_reduce_def.cpp`
全程只有 3 处 `AddConfig`：`ascend950` / `ascend910b` / `ascend310p` ——
**没有 `ascend910_93`**。容器 kernel 目录里也确实没有 `matmul_all_reduce`。
⇒ **不是打包事故，升级 CANN 也没用**（9.2.0 / master 同样只有这 3 个）。

## 2. 替代路径：`GroupedMatMulAllReduce` **支持 A3**

| 证据 | 内容 |
|---|---|
| 本机 kernel 目录 | `ascend910_93/ops_transformer/grouped_mat_mul_all_reduce` **存在** |
| 算子 def | `def.cpp:66` 注册了 `ascend910_93` |
| tiling | `args_.cmdType = mc2tiling::AicpuComType::HCCL_CMD_ALLREDUCE`（就是 all-reduce） |
| rankSize | 支持 1/2/4/8 → **TP=8 命中** |
| 库符号 | `libopapi_transformer.so` 里 `aclnnGroupedMatMulAllReduce` 与 `...GetWorkspaceSize` **都在** |

## 3. 参数约束（**实测解出**，与文档/推断不同）

用 `gmar_shim.c`（C 封装，因为 aclTensorList/aclIntArray 无法在 Python 构造）
把它编译成 `.so`，再由 `probe_gmar_check.py` 只调 **GetWorkspaceSize**（纯参数校验、不通信）
做组合扫描。**16 个组合的结果**：

| bias dtype \ splitItem | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **fp32** | ✅ 通过 | 161002 | ✅ 通过 | 161002 |
| bf16 | 161002 | 161002 | 161002 | 161002 |
| fp16 | 161002 | 161002 | 161002 | 161002 |
| 不用 bias | ✅ 通过 | 161002 | ✅ 通过 | 161002 |

（通过的组合 workspace = 16,777,216 B）

### 三条硬约束

1. **`streamMode` 必须 = 1** —— 唯一支持值。传 0 会报
   `Parameter streamMode of aclnnGroupedMatMulAllReduce has incorrect value 0. Reason: only support 1.`
   （这条是从 CANN plog 里挖出来的，aclnn 不往 stdout 打）
2. **`splitItem` 只能 = 0 或 2** —— 1/3 要求 `groupList` **不为 nullptr**
   （`CheckGroupListOptional`: "When splitItem is 1/3, groupList size must not be nullptr"），
   单组场景没有 groupList，所以只能用 0/2。
   > 注意：`opensrc_ops_hunt` 的初版结论说"只支持 1/2/3、不支持 0"是**错的**，
   > 源码里 `X_Y_SEPARATED = 0` 是合法值（`aclnn_grouped_mat_mul_all_reduce.cpp:43-46`）。
3. **`bias` 必须是 fp32** —— bf16/fp16 一律 161002。

### 为什么这些约束这样设计（源码依据）

```cpp
static const int64_t X_Y_SEPARATED = 0; // x,y不切
static const int64_t Y_SEPARATED   = 1; // x切分
static const int64_t X_SEPARATED   = 2; // y切分
static const int64_t NO_SEPARATED  = 3; // x,y切分
```

校验分支（`CheckParamDimAndLengthGmmAr`）：

| splitItem | x 列表长度 | groupList | y 列表长度 |
|---|---|---|---|
| **0 / 2** | **必须 == weight 列表长度** | **必须 nullptr** | 0→==weight；2→**必须 1** |
| 1 / 3 | 必须 == 1 | **必须非 nullptr 且递增、末值 == x.shape[0]** | 1→==weight；3→必须 1 |

⇒ **我们的单组 TP 场景应取 `splitItem=2`**（x 分片、y 完整 [m,n]）：
`x=[x]`、`weight=[W]`、`y=[y]`、`bias=[b]`（fp32）、`groupList=nullptr`。

## 4. 代码与产物

| 文件 | 作用 |
|---|---|
| `a21_scripts/gmar_shim.c` | C 封装：`gmar_check`（只校验）/ `gmar_run` / `gmar_run_ex`（可指定 bias dtype） |
| `a21_scripts/probe_gmar_check.py` | 组合扫描（torchrun 8 卡，**不做任何集合通信**，失败不污染通信域） |
| `a21_scripts/bench_gmar_final.py` | 性能对比（融合 vs 分离，含正确性对账） |

编译（容器内）：

```bash
g++ -O2 -fPIC -shared -o /tmp/libgmar.so gmar_shim.c \
  -I/usr/local/Ascend/cann-9.1.0/include \
  -L/usr/local/Ascend/cann-9.1.0/lib64 \
  -lopapi_transformer -lascendcl -lnnopbase
```

## 5. 踩过的坑（会重复浪费时间）

1. **aclnn 的详细错误只在 CANN plog 里**：`/root/ascend/log/debug/plog/plog-*.log`。
   stdout 只给一个状态码（161002），不告诉你哪个参数。
   → 排障必须看 plog，用 `grep -iE "aclnnGroupedMatMulAllReduce|只有 support"`。
2. **失败的集合通信会占住 NPU 侧端口 16666**，`ss` 看不到（它在 device 网卡上）。
   症状：下一次 torchrun 报 `Communication_Error_Bind_IP_Port(EI0020)`。
   → 解决办法：`pkill -9 -f <你的脚本名>` 清掉残留 worker，确认 `npu-smi` 无占用再重试。
3. **`pkill -f torchrun` 会杀掉自己正在启动的命令**（命令行里含 "torchrun"）。
   → 用脚本文件启动，不要内联。
4. **容器重建会丢 `/etc/hosts` 里的 `127.0.0.1 host22`**，导致 torchrun rendezvous 超时。
   → 每次新容器都要补 `echo "127.0.0.1 $(hostname)" >> /etc/hosts`。
5. **8 卡 torchrun 的初始化很慢（~3 分钟）**，每轮实验要按 4 分钟预算。

## 6. 状态与下一步

| 步骤 | 状态 |
|---|---|
| 1. 确认 `MatmulAllReduce` 在 A3 不可用 | ✅ 完成（上游不支持，无疑问） |
| 2. 确认 `GroupedMatMulAllReduce` 在 A3 可用 | ✅ 完成（符号在、内核在、参数校验通过） |
| 3. 解出参数约束 | ✅ 完成（streamMode=1 / splitItem=0,2 / bias fp32） |
| 4. **性能对比（融合 vs 分离）** | 🔄 进行中（`bench_gmar_final.py`） |
| 5. 若④有收益：接进模型（改 o_proj 与共享专家两处 + 重捕获图） | ⏸ 待定 |

**收益上限**：2.24 ms/step（当前 87 次 allReduce × 全暴露）。
**接入成本**：高 —— vllm-ascend **没有**这个算子的绑定（`mmrs_fusion` 是死代码、
`npu_mm_reduce_scatter_base` 封装从未被调用），需要自己写 aclnn 调用并改图内 forward。

---

## 6.1 最终结论：**收口**（2026-09-16 21:25）

### 走到哪一步了

真机执行（`bench_gmar_final.py`，8 卡 torchrun）：

```
[step] init ok  world=8 hcom=group_name_0
[step] 开始测分离臂 ...
[step] 分离臂完成: 208.7 us/次          ← 分离臂基线拿到了
  融合臂失败 rc=-563000 (aclnn=561000)  —— split=2
```

plog 里的真因（`/root/ascend/log/debug/plog/plog-1483_*.log`）：

```
HCCL: [HcclCommunicator][CreateCommAndStreamRes] errNo[0x0000000005000001]
      tag[CreatecomResource_group_name_0], comm resource create comm failed
OP:   [HcclAllocComResourceByTiling] errno[561000] Failed to invoke the
      HcclAllocComResourceByTiling function of the hccl module, ret = 1
OP:   GroupedMatMulAllReduce, This is an error in launch aicore
```

⇒ 参数已过全部校验，**卡在 MC2 的 HCCL 通信资源创建**。
`group_name_0` 是 PyTorch `ProcessGroupHCCL` 的内部名，**不是 MC2 能直接复用的通信域名**
（vllm-ascend 里 MC2 路径用的是 `DeviceOperator` 自己创建的 HCCL comm 与其 tiling 上下文）。

### 为什么决定收口（三条独立理由）

1. **收益无法证明，且上限有限**。
   分离臂微基准 208.7 µs/次，但**这个数字不能外推**：
   真实推理里 87 次 allReduce 的**墙钟占用只有 2.24 ms/step**（profile 实测，
   相当于单次 25.7 µs），而微基准测的是"孤立调用的串行延迟"（含 matmul + 跨卡同步）。
   两者差 8 倍，说明真实推理中的通信**已被流水线交错**，微基准无法反映。
   ⇒ 即便融合算子跑通，也**不能推断**它一定能改善真实 ms/step；
   要证明只能**真接进模型**测，而那是下面第 2 条的成本。

2. **接入成本是"一周级"**。
   vllm-ascend 对这条路径**没有任何绑定**：
   - `mmrs_fusion` 是**死代码**（设置 + 透出到 forward context，**0 个消费点**）
   - `npu_mm_reduce_scatter_base` 封装存在但**从未被调用**
   - `npu_mm_all_reduce_base` 全树 **0 处**
   ⇒ 要自己写 aclnn 调用（本报告已给出 C 封装的雏形）+ 改 MLA 的 `o_proj` 与
   MoE 共享专家两处**图内** forward + 每次改动重捕获图。

3. **低成本方向已全部穷尽**（见 `cannbot-measures-triage.md`）：
   3 个纯 env 变量无收益、`CPU_AFFINITY_CONF` 本机不可用、MC2/hierarchy 与 MegaMoe 均被
   SoC/配置约束关闭、`MatmulAllReduce` 上游不支持。**这条是最后一条技术路线**，
   但它属于"投入显著、收益不确定"。

### 留下的可复用资产（**不要丢弃**）

| 资产 | 价值 |
|---|---|
| 参数约束的**完整解**（streamMode=1 / splitItem=0,2 / bias fp32） | 任何人接手可直接跳过我们踩过的坑 |
| `gmar_shim.c`（可编译的 C 封装） | 已验证能编、能过参数校验，只差 MC2 comm 对接 |
| 官方提示：`HCCL_NPU_SOCKET_PORT_RANGE` | 解决单卡多进程的 HCCL 端口冲突（通用） |
| 5 条踩坑记录（§5） | 含 CANN plog 定位法、僵尸进程占端口、pkill 自杀等 |
| 分离臂基线 208.7 µs/次 vs profile 25.7 µs/次 | **证明微基准不可外推**——这条方法论本身有价值 |

### 若要有人继续（建议路径）

1. 参照 vllm-ascend 的 `DeviceOperator`（`device_op.py`）看 MC2 算子**完整的**调用契约：
   hcom 名从哪来、需要哪些 tiling 上下文、是否需要先建 MC2 专用 comm。
   关键线索：`token_dispatcher.py` 里 `MoeDistributeDispatchV2` 是怎么拿 `group_ep` 的。
2. 最小可行验证：**不改模型**，先在 vllm-ascend 里把 `GroupedMatMulAllReduce` 当成一个
   eager 算子跑通（复用现有 MC2 comm），确认它确实能 all-reduce。
3. 只有①②都过了，才谈接进 forward。

## 7. 证据路径

| 内容 | 路径 |
|---|---|
| 源码调研 | `/home/user/opensrc/FINDINGS.md`（538 行）、`opensrc/ops-transformer/mc2/grouped_mat_mul_all_reduce/` |
| 头文件签名 | `.../op_api/aclnn_grouped_mat_mul_all_reduce.h:48-63` |
| 校验逻辑 | `.../op_api/aclnn_grouped_mat_mul_all_reduce.cpp:43-46, 108-232` |
| 参数扫描原始输出 | A3-node2 `/tmp/chk_out3.txt`（16 组合结果） |
| plog（真实原因） | A3-node2 容器 `/root/ascend/log/debug/plog/plog-3824_*.log` |
