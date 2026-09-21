# REPRO.md —— 详细复现手册（v5）

> 阅读顺序：`README.md`（一页）→ 本文件（详细）→ `CHANGELOG.md`（v3→v4 改了什么）→
> `EXPECTED_PERF.md`（该看到什么数）→ **`CORRECTNESS_STATUS.md`（数值确定性/精度）** → `reports/`（为什么）。
>
> 所有命令都在**宿主机**上执行（不要进容器），工作目录 = 本包根目录。

---

## 1. 前置条件

| 项 | 要求 |
|---|---|
| 机器 | A2：8×910B3 + Kunpeng-920，**宿主机**上执行 |
| 权限 | `docker`（或 `sudo -n docker`）、`npu-smi` |
| 基础镜像 | 已 `docker pull`（默认 `quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-openeuler`） |
| 模型 | `v41-w4a8-engram-dr-vision-qrot-mtpq` 等价目录（**含 qrot 修复后的 vision 分片**）<br>▸ 权重下载：<https://www.modelscope.cn/models/chiro2001/DeepSeek-V4.1-Flash-w4a8-Ascend>（212.6 GB）<br>▸ **下完必须先重组 Engram**：`cd engram_int8 && bash reassemble_engram_weights.sh`（两个大权重各 6 片，不重组直接起服会失败）<br>▸ 该仓同时含量化复现流水线（`release/msmodelslim/`），见 `quant/REPRO_W4A8_QUANT.md` |
| 官方目录（可选） | `OFFICIAL_DIR`（视觉 23 例的图片）与 `ENC_DIR`（GSM8K 的 chat 模板）—— 都缺就只跑性能+容量+多 batch |
| 磁盘 | 镜像层 ~20 GB + `cache/`（首次 static kernel 编译）若干 GB |
| 网络 | **不需要外网**（GSM8K 需要本地 `datasets` 缓存 **+ 官方 `encoding` 目录**；缺任一个就跳过并标 `skipped`） |
| CPU 绑定 | 脚本自动从 NPU 的 PCI `numa_node` 推导；推导失败会告警，此时手动给 `CPUSET=`/`MEMS=` |

---

## 2. 命令 ①：生成修改后的容器镜像

```bash
bash scripts/build_image.sh
# 自定义：
# BASE_IMAGE=<你的镜像> IMAGE_TAG=dsv41-a2:v5 bash scripts/build_image.sh
# NO_CACHE=1 bash scripts/build_image.sh      # 不用构建缓存
# SKIP_PGO=1 bash scripts/build_image.sh      # 不打包 PGO 产物（镜像小 30 MB）
```

它做 5 件事：

1. **探测** `vllm_ascend` / `vLLM` / python 的安装路径（不同镜像布局不同，不写死）。
2. 把 `patches/files/` 的 **11 个整文件**烘焙进镜像目标路径（原文件留 `.a2orig` 备份），
   并放入 **2 个新增 sidecar**（`engram_jit_kernel.py` / `engram_plan_kernel.py`）。
3. 对 `$VLLM_ROOT` `git apply admission_gate.patch`（失败只告警，起服时用 `VLLM_ADMISSION_GATE=0`）。
4. 若 `optim/pgo/` 下已有 PGO 产物，放进 `/opt/dsv41/pgo/`，并**探测 libpython 落点**写到
   `optim/pgo/TARGET_PATH.txt`；若镜像 python ≠ 3.12.13 会写 `VERSION_MISMATCH.txt`
   （此时起服请用 `PYTHON_PGO=0`）。
   **包内默认没有该产物** —— 要启用先在目标机跑 `bash build_scripts/00_ensure_pgo.sh`
   （详见 `optim/pgo/README.md`；不启用不影响其余功能）。
5. **镜像内逐文件 md5 + `py_compile` 校验**（任一不符 → 构建失败）。

### 烘焙了什么（11 项，逐条见 `patches/PATCHES.md`）

| 目标（容器内） | 门控 env | 效果 |
|---|---|---|
| `models/deepseek_v41/engram_hbm.py` | （总开）`V41_ENGRAM_HOST_RESIDENT=1` | Engram host 常驻 + local-owner(fast) |
| `models/deepseek_v41/engram_hash.py` + `engram_jit_kernel.py` | `V41_ENGRAM_JIT=1` | hash 0.427 → 0.076 ms |
| `models/deepseek_v41/engram_plan_kernel.py` | `V41_ENGRAM_JIT=1` | plan 0.261 → 0.068 ms |
| `models/deepseek_v41/engram_gate.py` | `V41_ENGRAM_GATE_CHUNK=0` | 去 2048 行 padding，−1.56 ms（8K） |
| `models/deepseek_v41/model.py` | 与 engram_hbm 配对 | host-resident 模型侧 |
| `models/deepseek_v41/indexer.py` | `V41_QLI_NO_CANDIDATE=1` | QLI 99.3 → 50.3 µs，−0.49 ms |
| `ascend_forward_context.py` | `V41_MOE_COMM_ALLGATHER=1` | MoE AllGather，−4.25 ms（128K），KV 3.39M→4.16M |
| `attention/dsa_v1.py` | `V41_O_PROJ_2D=1` | F3 `wo_a` 2D，−0.31~0.76 ms |
| `ops/fused_moe/token_dispatcher.py` | `V41_MOE_MASK_RANGE=1` | mask-range，−0.51 ms（精度已过） |
| `ops/rope_dsv4.py` | `V41_ROPE_IDXSEL=1` | rope 6→2 kernel，−0.45~0.62 ms |
| vLLM core（patch） | `VLLM_ADMISSION_GATE=1` | prefill 不饿死 decode |

实验补丁（**只放不装**，在 `/opt/dsv41/patches/`）：`files/token_dispatcher_moezero.py`、
`draft/{dsa_v1,dspark_proposer,llm_base_proposer}.py`。

### 回滚

```bash
docker run --rm -it dsv41-a2:v5 bash
A=/vllm-workspace/vllm-ascend/vllm_ascend
for f in models/deepseek_v41/engram_hbm.py models/deepseek_v41/engram_gate.py \
         models/deepseek_v41/indexer.py models/deepseek_v41/model.py \
         ops/fused_moe/token_dispatcher.py ops/rope_dsv4.py \
         attention/dsa_v1.py ascend_forward_context.py ; do
  [ -f "$A/$f.a2orig" ] && cp "$A/$f.a2orig" "$A/$f"
done
```

---

## 3. 命令 ②：起服务并跑验收测试

### ⚠️ 3.0 模型目录是**软链构造**的 —— 这是 v3 会直接起服失败的原因（v5 修复）

量化流水线（modelscope 上那套脚本）最终产出的目录是**零拷贝的软链结构**，
而且软链是**绝对路径**、**链条很深**。真实产物的实测分布：

```
软链跳数分布: 1 跳 × 2,  2 跳 × 8,  3 跳 × 4,  4 跳 × 80     （共 94 个软链）
```

链条形态：

```
v41-w4a8-engram-dr-vision-qrot-mtpq/      <- L5 你传给 MODEL 的那个目录
  config.json -> /abs/.../v41-w4a8-engram-dr-vision-mtpq/config.json   (L4)
                   v41-w4a8-engram-dr-vision-mtpq/  ->  ... (L3) -> (L2) -> (L1)
                                                            L1 = 真正的 87 个实体分片
```

**`docker run -v "$MODEL:$MODEL:ro"` 只挂了 L5**，于是容器里所有指向 L4/L3/L2/L1 的
绝对路径软链**全部悬空** —— 宿主机上 `ls`/`cat` 一切正常，进容器立刻
`No such file or directory`（第一个报错通常是 `config.json` 或 tokenizer）。

实测复现（两行就是全部差别）：

```bash
# 旧行为：只挂叶子层 -> 悬空
docker run --rm -v <L5>:<L5>:ro alpine cat <L5>/config.json
#   cat: can't open '.../config.json': No such file or directory

# v5 行为：逐层挂载 -> 可读
docker run --rm $(bash tools/model_mount_args.sh <L5> | tr '\n' ' ') alpine cat <L5>/config.json
#   {"model_type":"deepseek_v41",...}
```

**v5 的修法**（已内建，不需要你做任何事）：

| 环节 | 行为 |
|---|---|
| `tools/model_mount_args.sh`（新增） | 逐跳解析**字面软链**（不是 `realpath`！），把每一层的目录都输出成 `-v` 参数 |
| `scripts/serve_a2.sh` | 自动调用上面这个工具，把所有层级都挂上；**实测真值需要挂 13 个目录** |
| `tools/check_model_dir.sh` | 起服**前**就检查悬空软链 + 报告链条深度 ⇒ 5 秒内失败，而不是白等 4 分钟 |
| `scripts/serve_a2.sh`（v8 §14） | **Engram 表目录单独叠加成 `:rw`**（其余仍 `:ro`），并在 `docker run` 前自检；为什么必须 rw 见 §3.0.1 |

#### 3.0.1 Engram 表为什么必须 `:rw`（v8 起；A2 真机踩过 `ret=507899`）

Engram 表由设备算子直接索引，`patches/files/engram_device_index.py::_map_and_register()`
走的是 `os.open(path, O_RDWR)` + `PROT_READ|PROT_WRITE, MAP_SHARED` 的 `mmap`，再交给
`acl.rt.host_register(..., ACL_HOST_REGISTER_MAPPED)`。**只读 VMA 会被驱动拒绝**
（`aclrtHostRegister failed: ret=507899`），而 `os.open` 在 read-only 挂载上先就报 `EROFS`。
代码本身只**读**这些文件 —— 要写权限纯粹是驱动注册的要求。

三种形态都要放开（`serve_a2.sh` 现在自动做，不用你动手）：
1. `$MODEL/engram_int8` **是实体目录**（`model_mount_args.sh` 不输出它 ⇒ 旧代码漏）；
2. 软链链条上的 `engram_int8` / `engram-int8`（逐跳都要）；
3. **真正落盘的目录** —— `engram_int8/` 里的条目本身还是软链，`O_RDWR` 按最终 inode
   判定（实测链条：`$MODEL/engram_int8 → L4 → L3（实体）→ 4 个软链 →
   …/projects/dsv41/models/out/engram-int8 → …/models/out/engram-int8`，最后一步在
   **另一棵目录树**里）。

宿主上"文件不可写"（A3 真机的表是 `root:root 0600`）**不影响**：容器以 root 起、挂载是
`:rw` 就能 `O_RDWR`；脚本就此只打一条 WARNING。真要绕开可用 `ENGRAM_DEVICE_INDEX=0`
（走 host 路径，完全不要求可写，代价是关掉 v8 的 device-index 加速）。

离线验证（不需要 A2 / 不需要 docker）：

```bash
bash tests/engram_rw_mount_test.sh     # 三种挂载模式 + 实体目录 + 异常形态，34 条断言
```

可调开关：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MODEL_MOUNT_MODE` | `auto` | `auto` = 逐层精确挂载（13 个目录）；`ancestor` = 只挂公共祖先（更宽但更省，如整棵 `models/out/`）——**若选出的祖先覆盖不到某些目录就自动退回 `auto`**（A3 真机的模型树横跨两棵目录树，旧写法会挂出一个覆盖不到的祖先，容器里软链全悬空）；`none` = **旧行为**，仅用于复现该故障。三种模式都会把 engram 表目录单独叠加成 `:rw` |
| `EXTRA_MODEL_MOUNTS` | 空 | 分号分隔的额外宿主路径，例如软链指向了模型目录之外的地方 |

**为什么会踩**：`os.path.realpath()` 会把 `L5→L4→L3→L2→L1` 一次折叠成 L1，
看起来"目标只是 L1，挂 L1 就行" —— 但**容器是逐跳解析的**：打开
`/abs/L5/config.json` 读出 `"/abs/L4/..."`，再去开 `/abs/L4/config.json`。
所以每一跳的目标目录都必须存在。这就是本工具坚持用 `readlink` 而不是 `realpath` 的原因。

**先自检**（推荐，5 秒）：

```bash
bash tools/check_model_dir.sh "$MODEL"        # 健康度 + 链条深度
bash tools/model_mount_args.sh "$MODEL"       # 看看实际会挂哪些目录
MODEL="$MODEL" DRY_RUN=1 bash scripts/serve_a2.sh | grep MODEL_MOUNT   # 看最终挂载清单
MODEL="$MODEL" DRY_RUN=1 bash scripts/serve_a2.sh | grep engram_int8   # 每行都应是 :rw
```

```bash
MODEL=/path/to/v41-w4a8-engram-dr-vision-qrot-mtpq bash scripts/run_test.sh

# 全量（加 128K + GSM8K-200）
MODEL=... MODE=full bash scripts/run_test.sh
# 【v4 新增】生产口径（max-num-seqs 32 + prefix caching ON）+ 多 batch / 多轮对话
MODEL=... MODE=prod bash scripts/run_test.sh
# 【v4 新增】两臂对照（PREFIX=1 与 PREFIX=0，各一个会话）
MODEL=... bash tests/multibatch/run_prod_both.sh
# 只测时延（不读权重，A 恒 1.0）
MODEL=... LOAD_FORMAT=dummy bash scripts/run_test.sh
# 纯起服，不跑测试（自己手动测）：
MODEL=... WAIT_READY=0 bash scripts/serve_a2.sh
```

### 口径警告（v4 新增，**最容易犯的错**）

| 口径 | `MAX_SEQS` | `PREFIX` | 用途 |
|---|---|---|---|
| **单流（历史，v3 默认）** | 1–4 | **0** | 性能（ms/step、A、tok/s）与历史数字**可比** |
| **生产（A2 真机）** | **32** | **1** | 多 batch / 多轮对话 / 前缀复用；**ms/A 与单流不可比** |

`run_test.sh` 已按 `MODE` 给不同默认值（`quick/full` → 4/0，`prod` → 32/1），显式传参优先。

### 它会做什么

| 阶段 | 内容 | 失败处理 |
|---|---|---|
| 0 预检 | docker / 镜像 / 模型目录自检 / **芯片占用**（HBM > 20 GB 判占用） | 立即中止 |
| 1 起服 | `serve_a2.sh`：TP8 + EP + `FULL_DECODE_ONLY` + Engram(int8, host) + DSpark S=5 + Vision | — |
| 2 等就绪 | 每 15 s 报进度，35 min 超时；检测启动报错立即中止并打印 | 打印日志尾部 |
| 3 **必查** | ① `static_kernel.py:650` 命中必须为 0 ② local-owner `validate` 逐位对账 → 通过后切 `fast` ③ KV > 3Mi | ① 判 FAIL 并提示 `STATIC_KERNEL=0`；② 自动降级 `on`；③ 记录判定 |
| 4 性能 | quote 单流 8K / 32K（`MODE=full` 加 128K），每点 `REPEATS=2` 发。**`MODE=prod` 跳过本段** | 打印中位 |
| 4b **多 batch**（v5） | `MODE=prod`：`multibatch_gate.py` 三块 **A 多轮对话 / B 并发逐 item / C 长短交错** ⇒ 结论进 `REPORT.md` §1b | 任一块 `verdict` 非 PASS 会打印 ❌ |
| 5 视觉 | 23 例图文问答（≥19 通过为达标） | 需要 `OFFICIAL_DIR` 官方图片目录 |
| 6 精度 | GSM8K-200（`MODE=full` 或 `RUN_GSM8K=1`） | 需要本地数据集缓存 + 官方 `encoding` 目录（`ENC_DIR` 或 `--enc-dir`；缺则跳过） |
| 7 报告 | `results/<run_id>/REPORT.md` + 全部原始 jsonl | — |

### 产出

```
results/<run_id>/
├── REPORT.md                    ← 先看这个
├── serve.log                    ← 服务日志（启动行含 KV 容量；static_kernel 检查也查它）
├── serve_cmd.txt                ← 本次实际使用的全部开关（含 pgo/cpuset/未验证项状态）
├── env.txt                      ← KV / static_kernel_hits / engram_on 等事实
├── local_owner_effective.txt
├── p42_t4_quote_8192_*.jsonl    ← 8K 原始
├── p42_t4_quote_32768_*.jsonl   ← 32K 原始
├── p42_t4_quote_131072_*.jsonl  ← 128K 原始（MODE=full）
├── vision.json / gsm8k.json
└── inner.sh                     ← 容器内实际执行的起服命令
```

---

## 4. 每个开关：作用、预期数字、判据

> 完整 diff 见 `CHANGELOG.md`；收益来源见对应的 `reports/*.md`。

| 开关 | 默认 | 预期收益（A3-node1 实测） | 判据（A2 上怎么知道它生效了） |
|---|---|---|---|
| `MOE_AG=1` | 1 | 128K **−4.25 ms**；KV 3.39M → **4.16M tokens** | `serve.log` 里 KV 明显 > 3.39M；关掉它 ms/step 上升 4 ms 左右 |
| `SP_TOKENS=5` | 5 | A：**3.493 vs 2.793（S=7）**；tok/s 100 vs 79.9 | `serve_cmd.txt` 里 `capture_sizes` 必须含 **6**（=S+1）；缺了会 +5.9 ms |
| `O_PROJ_2D=1` | 1 | −0.31 ~ −0.76 ms | 关掉再测一次，应看到 ~0.5 ms 差 |
| `MOE_MASK=1` | 1 | −0.51 ms | 精度已过（GSM8K 100/100、Vision 23/23）；关掉变慢 |
| `ROPE_IDXSEL=1` | 1 | −0.45 ~ −0.62 ms/pass | profiler 里 `Index`/`IndexCheck` 各 −4104 |
| `ENGRAM_JIT=1` + `LOCAL_OWNER=fast` | 1 / fast | hash −0.35、plan −0.19、local-owner −0.11 | 第一次起服会编译 numba（慢）；`cache/numba` 里有 `.nbi/.nbc`；`LOCAL_OWNER` 不是 `fast` 则**永不进 JIT 分支** |
| `QLI_NOCAND=1` | 1 | −0.49 ms | QLI per-op 99.3 → 50.3 µs（device 账） |
| `PYTHON_PGO=1` | 1 | 宿主 micro −16~23%；**服务侧 −4.4%**（31.626→30.230） | `serve_cmd.txt` 里 `pgo_target=` 非空；缺产物会自动降级为 0 |
| `VISION=1` | 1 | — | 视觉 23/23（qrot 修复后） |
| `GATE_CHUNK=0` | 0 | −1.56 ms（8K） | 与 v2 相同 |
| `VLLM_ADMISSION_GATE=1` | 1 | prefill 不饿死 decode | 构建时若 patch 无法应用，改 0 |
| **`PREFIX`**（v5） | **0** | `1` = prefix caching ON（**A2 生产口径**）；`0` = 历史性能口径 | `serve_cmd.txt` 里 `PREFIX=`；生产验证用 `MODE=prod` |
| **`MAX_SEQS`**（v4 强调） | 4（prod 32） | 决定 `CAPTURE_SIZES` 上限 = `MAX_SEQS×(1+SP_TOKENS)` | `DRY_RUN=1` 打印 `CAPTURE_SIZES`；`=32` 必须含 **192** |
| `MOE_ZERO` | **0** | ❌ **不采纳**：换吸引子（低峰没了，高峰 0.86→0.73）+ 非数值等价 | 只在复现负结果时打开 |
| `MOE_NF`（v5） | **0** | ❌ **无差异**（clean 2/24 vs 2/24）⇒ `0 权重 × Inf = NaN` 通道排除 | 只在复现负结果时打开 |
| `DRAFT_GRAPH` | **0** | 目标机预计 ~24 ms/轮 | ⚠️ 离线修复完成 + **负控已确认**（缺 metadata ⇒ A 恒 1.0，8/8 发，ms 仍 30.2）；**正控待验** |
| `HCCL_DET`（v5） | **空** | 仅诊断。`true` 掉 GSM8K 到 91/100（**不能进交付**）；`strict` 保精度但确定性只是"每批抽签" | 合法值只 `true/false/strict`；写 `1` 起服失败（EI0001） |
| `DRY_RUN`（v5） | 0 | `1` = 只解析开关 + 组装 MOUNTS 并打印，**不碰 docker** | `bash tests/multibatch/verify_serve_flags.sh`（12 组合） |

### `LOAD_FORMAT=dummy` vs 真权重（**必须分清**）

| | 真权重（默认） | `LOAD_FORMAT=dummy` |
|---|---|---|
| 加载 | 读 273 GB 权重（4.5 min 级） | 只按 checkpoint 的 shape/dtype 建模型，**不读权重文件** |
| 用途 | 精度 / 容量 / 接受长度 / 端到端性能 | **只测时延与图捕获**（快速迭代起服时间） |
| 接受长度 A | 真值（2.68 等） | **恒为 1.0**（draft 全是随机权重，必然全被拒） |
| 额外 env | — | 必须带 `V41_ENGRAM_WITH_DUMMY=1` + `V41_DUMMY_WO_A_FIX=1`（本包脚本已自动加） |
| 陷阱 | — | dummy 下 Engram host 路径被主动跳过 ⇒ **不能用它测 Engram 相关结论** |

---

## 5. 与 A3-node1 的差异清单（**必须现场重新测，不能照抄**）

| 量 | 为什么不能照抄 |
|---|---|
| ms/step、tok/s | 910B 设备侧慢、Kunpeng-920 宿主派发慢一个量级 |
| KV 池 tokens / `num_blocks` | 驱动/CANN 版本决定非 Torch 内存与碎片 |
| `HCCL_BUFFSIZE` 等通信参数 | 拓扑/NIC 不同 |
| 需要编译的 static kernel | 编译缓存要重新生成（首次起服慢） |
| **接受长度 A** | **与平台无关** ⇒ 是交叉验证的最佳锚点（±15%） |
| Engram 常驻 DRAM | ≈ 206 GiB，与平台无关，也是强判据 |

---

## 6. 验收判据汇总（照这个判 PASS/FAIL）

| 判据 | 门槛 | 不通过时 |
|---|---|---|
| `static_kernel.py:650` 命中数 | **== 0** | 结果不可信：查 `LOCAL_WORLD_SIZE` 是否进容器（脚本已注入），或临时 `STATIC_KERNEL=0` |
| KV 池 tokens | **> 3,145,728** | 抬 `GPU_UTIL`（0.95/0.96）或确认用的是本包模型目录 |
| 接受长度 A（128K，8 发中位） | **2.68 ±15%**（2.28–3.08） | 差太多说明软件没装对（不是平台差异）——查补丁 md5、开关、量化结构。⚠️ **只当"装对没装对"的锚点，不当绩效指标**（见下一行） |
| **clean-rate**（`pos0 ≥ 0.8` 的请求占比） | 我们不同会话 16%–37% | **这才是该优化的量**；A 高但 clean 低 = 脏会话（A 与质量反相关）。见 `EXPECTED_PERF.md` §3 |
| 确定性自检（开跑前） | 同一 prompt 8 发：`uniq_top1==1` 且 `n_distinct_lp==1` | 不满足 ⇒ **本会话的 A/形态数据不可用**，先重起服（见 `CORRECTNESS_STATUS.md` §2.1） |
| **多 batch：A 多轮逐字召回** | 我们生产口径实测 **轮内 7/7 + 全长 3/3**（`uniq2` 全 1.00，末次 prompt=409 tok） | 有漏 ⇒ 长历史/前缀复用有问题（`results/*/summary.json` 的 `verdict.A_needle`）。⚠️ 我们只测到 409 token，**没逼出长历史**，你可以加大 `MBG_ROUNDS` |
| **多 batch：B/C 逐 item 不一致数** | 我们生产口径实测 **B/C 各 0 项不一致**（各 16/16、6/6） | 非 0 ⇒ 并发或长短交错破坏了正确性（`verdict.B_itemwise` / `verdict.C_short_vs_long`）；每个不一致项都值得回报 |
| **混合负载：C 的 128K 请求本身** | 我们实测正常返回（131,072 prompt tokens） | 异常 ⇒ 看 `mixed_long_short.json` 的 `long_result` |
| ms/step（128K） | 比我们高是预期 | 高得离谱时先查 jemalloc（日志 `LD_PRELOAD detected`）、CPU 绑定、static kernel |
| Vision | **≥ 19/23** | 检查模型目录是否含 qrot 修复后的 vision 分片 |
| Engram DRAM | ≈ 206 GiB | 差太多说明 Engram int8 表没装对 |
| `GSM8K-200` | 同量级（我们 **198/199/197**） | 掉到 ~91/100 的形态 ⇒ 是不是误开了 `HCCL_DETERMINISTIC=true` |
| 开了 `DRAFT_GRAPH=1` 之后 | **`A > 1.5` 且 `dspark-graph-capture > 0`** | 只报 ms 的"验证"**无效**（负控指纹：A 恒 1.0 而 ms 正常） |

---

## 7. 失败排查表（现象 → 原因 → 处理）

| 现象 | 最可能原因 | 处理 |
|---|---|---|
| `build_image.sh` 报"基础镜像不存在" | 没 pull | `docker pull <BASE_IMAGE>` 或 `BASE_IMAGE=` 指定 |
| 构建时 `admission gate 无法应用` | 基础镜像版本不同 | 只是警告；起服加 `VLLM_ADMISSION_GATE=0` |
| 模型自检报 `'NoneType' has no attribute 'primes'` | `text_config.engram_layer_ids=[]` 但 `enable_engram=true` | `run_test.sh` 自动关 Engram，并打印提示 |
| **8K ms/step 到 41–43** | `LD_PRELOAD` 里没有 jemalloc | 查日志 `LD_PRELOAD detected`；`serve_v2.sh` 会自动加 |
| 起服卡在 static kernel 编译 | 首次编译（几分钟 ~ 十几分钟） | 等；缓存在 `./cache/skcache`，第二次快 |
| `static_kernel.py:650` 命中 > 0 | `LOCAL_WORLD_SIZE` 没进 `os.environ`（torch_npu 静默禁用静态内核） | 脚本已注入；仍命中就用 `STATIC_KERNEL=0` 并回报 |
| KV ≤ 3Mi | util 太低 / 不是本包模型目录 | `GPU_UTIL=0.95`（或 0.96）；确认 dir 名 |
| 视觉 < 19/23 | vision 分片没做 qrot 折叠修复 | 用 `quant/README.md` §L3 的修复流程重做 vision |
| `local-owner` 判 FAILED | validate 逐位对账没过 | 自动降到 `off`（功能正确、慢 ~1 ms）：把 `serve.log` 相关行发回 |
| Engram JIT 首次起服很慢 / numba 报错 | 正在编译（`cache/numba` 首次为空） | 等一次；仍失败用 `ENGRAM_JIT=0` 回退（慢 ~0.5 ms） |
| `PYTHON_PGO=1` 但无效果 | 镜像 python ≠ 3.12.13 ⇒ 产物不匹配 | 看构建时是否写 `optim/pgo/VERSION_MISMATCH.txt`；用 `PYTHON_PGO=0` |
| CPU 未绑定警告 | 平台不提供 PCI `numa_node` | 手动 `CPUSET="0-191" MEMS="0,1"`（按你的拓扑） |
| 128K 起服/推理 OOM 或崩 | `max-model-len=1048576` 本身超出我们验证的包络 | 先用 `MAX_LEN=131072` 验证，再逐步放大 |
| 容器退出 / `EngineCore failed` | 见日志尾部 | `tail -50 results/*/serve.log`，对照 `reports/` 里同类记录 |
| **`MAX_SEQS=32` 起服直接失败（"No valid cudagraph sizes"）** | `CAPTURE_SIZES` 没覆盖到 `32×(1+SP_TOKENS)=192` | 用本包 `serve_a2.sh`（已自动推导）；自检 `DRY_RUN=1 MAX_SEQS=32 bash scripts/serve_a2.sh \| grep CAPTURE_SIZES` ⇒ 必须含 `192` |
| `MODE=prod` 起服/捕获特别慢 | 15 个 capture bucket，每个 ~10–30 s | 正常等待；想快就 `MAX_SEQS=8`（桶少）但那就不是生产口径了 |
| 多轮对话某轮答不出针 | 长历史 + 前缀复用可能坏了；也可能是该会话数值脏 | 先看 `summary.json` 的 `multibatch.json` 逐轮 `dt/uniq2`；再按"确定性自检"重起服复测（`CORRECTNESS_STATUS.md` §2.1） |
| B/C 有逐 item 不一致 | 并发/混合 batch 下个别请求坏了 | 看 `concurrency_c*.json` / `mixed_long_short.json` 的 `mismatch` 列表（含两侧原始输出）；**每一个都值得回报** |
| 拿 `A` 的绝对值比"谁更快" | ⚠️ 口径错误：A 与质量反相关、且跨会话抽签 | 改报 **`(clean-rate, ms/step)`**；见 `EXPECTED_PERF.md` §3 |
| 引用某个 Fisher p 值时被质疑 | 我们的 `fisher()` 曾有 bug（相同表给 0.0000） | 用 `python3 tools/fisher_recheck.py`（内含教科书自检）**自己复算**，并说清是 `p_right` 还是 `p_two` |

---

## 8. 想继续做优化？（**v4 更新：方向变了**）

### 8.1 仍然值得试的（按价值排序）

1. **`DRAFT_GRAPH=1`（DSpark draft 入图）**：A2 的 draft 是 eager、约 **26 ms/轮**（占单轮 50%），
   我们 A3 上只有 1–4 ms ⇒ A2 的潜在收益 **~24 ms/轮**。
   **状态：离线修复完成 + 负控已确认，正控待验**（`reports/draft-graph-negative-control.md`）。
   ⚠️ **判据必须同时看 A 与 `dspark-graph-capture`** —— 负控模式下 **ms 完全正常但 A 恒 1.0**，
   只报时延等于没验证。材料：`reports/draft-graph-numinput-fix.md`、`tools/enable_draft_graph.sh`。
2. **把 ms 从 30.2 压到 ≤27.5**（比追 steep 更现实）：此时 A≥3.0（占 ~24%）即可 110 tok/s。
   我们历史最好单发 **26.9 ms** ⇒ 有 ~3 ms 空间。材料：`reports/host-api-profile-finding.md`、
   `reports/aicpu-allreduce-finding.md`。
3. **提高 clean-rate（16%–37% → 更高）**：唯一有正面证据的线索是"未写入行是低峰成因之一"
   （`reports/session-attractor-and-clean-rate.md` §4）—— 但 `MOE_ZERO`（全量清零）与
   `MOE_NONFINITE`（最小侵入版）**都已判无效**，需要第三条路。

### 8.2 ❌ **不要再试**（已判负结果，见 README §1.1）

| 项 | 结论 |
|---|---|
| `MOE_ZERO=1`（全量清零无效行） | **换吸引子**（低峰消失但高峰 0.86→0.73）+ 非数值等价 ⇒ 不采纳 |
| `MOE_NF=1`（只清零非有限元素） | clean 2/24 vs 2/24 **无差异** ⇒ `0 权重 × Inf` 通道排除 |
| `LOCAL_OWNER=on`（希望"更干净"） | 与 `fast` **无差异** ⇒ 保持 `fast`（更快 + JIT 前置） |
| `HCCL_DET=true`（希望"更确定"） | 与不开**无差异**，且 **GSM8K 91/100** ⇒ 只能当诊断 |
| "少跑 forward 来提高确定性" | **算术否证**：43/40=1.075 不可能产生 5 倍 clean 率差（解出 k≈22.4） |

---

## 9. 量化（可选，只在需要重做权重时）

完整 5 级装配见 `quant/README.md`。三条必记：

1. **`DEVICE_IDS="0 1 2 3 4 5 6 7"`**（写 8–15 → `ExchangeDevice` 报错）。
2. **不要用 sha256 验收主干**（DP 宽度不同必然不同；用 `quant/scripts/verify_structure.py`）。
3. `quant_dp_inner.sh` 会 `rm -rf $SAVE`（先确认目录）。
