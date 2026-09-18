# DeepSeek-V4.1-Flash W4A8 量化复现指南（msModelSlim / Ascend）

> 适用对象：在 Ascend A3（16 逻辑卡）上，用本仓库 fork 的 `src/msmodelslim` 把官方
> **DeepSeek-V4.1-Flash FP8/FP4 checkpoint** 量化成 **W4A8_DYNAMIC** 主干，再组装出
> 可被 vLLM-Ascend（`quantization=ascend`）加载的文本 / DSpark / 多模态服务目录。
>
> 本文所有命令与指纹都能从本仓库现成文件/日志复现；每节标注了验证状态：
> ✅ 已实测/已逐字节校验　⚠️ 已知偏差或待改进　❌ 未执行或验收未通过（不要当已验证用）。
>
> 交付记录见 `logs/perf/msmodelslim_repro_impl.md`；补丁说明见 `patches/PATCHES.md`。

---

> ⚠️ **状态总览**：文本链路（量化 → 组装 → 起服）已按本轮实测收敛；**视觉精度验收未过**
> （23 例 10 过 = 43.5%，必过负向 `neg-swap-01` 失败），**本指南 §7 / §11 的视觉部分为 WIP**，
> 不得写成“已支持/已达标”；根因候选见 `docs/VISION_RCA_P43.md`。

---

## 0. TL;DR 复现路径（每步预计耗时 + 失败排查入口）

1. **镜像 / 树**（冷拉镜像 30–90 min；模型 476 GB 视带宽）：`docker/enter.sh` 起 `quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3`；确认 `P=/home/user/projects/dsv41`、`R=$P/src/msmodelslim` 在 `92e219f`。排障：`a2_package/RUNBOOK.md`。
2. **打 msmodelslim 补丁、量化产权重**（补丁 + 装配 ≈2 min；16 卡 DP16 量化实测 **35 min 31 s**）：`scripts/apply_msmodelslim_patch.sh --check/apply` → `scripts/make_full_model.py` → `scripts/quant_dp_inner.sh`（§4）。排障：`patches/PATCHES.md`、`a2_package/RUNBOOK.md`。
3. **核对 3 个模型目录**（≈5 min，只读 header）：`v41-w4a8-stage1`（185,734 张量）、`v41-w4a8-dspark`（186,960）、`v41-w4a8-dspark-vision`（187,226 / 86 分片 / vision 266/266 = 0.9705 GB）；命令与期望见 §5.5。
4. **打服务补丁链**（≈1 min）：canonical → merged(statefix2+p3_sc) → kv_layout_depad → kv_layout_plane_tight → int8_kv → int8_kv_spec → capturefix → capturefix2（+ core `admission_gate`）；顺序依据 `docs/PATCH_APPLY_ORDER.md`；**禁打 prefill_incr / staging_delta（UNSAFE，§6.2.2）**。
5. **起服**（≈6–15 min，含权重 4.5 min + 图捕获）：`MODEL=v41-w4a8-engram-dr` + 档2 env = TP2/DP4、`--no-async-scheduling`、BAT=2048、util 0.97、SPEC=7、GRAPH=1、CPU_BIND=0、STATIC_KERNEL=0、int8-544/plane-tight、Engram on+host DRAM、tier=off（§6.2.3）；视觉另加 `--tokenizer-mode=deepseek_v4 --default-chat-template-kwargs={"enable_thinking":false}`（§6.2.5）。
6. **6 个启动数核对**（<1 min）：Available KV / num_blocks / request_blocks(A) / rb / rank_cap / staging；三档 BAT 1024/2048/4096 ⇒ num_blocks **42,156 / 39,896 / 35,449**、rank_cap **4 / 4 / 3**（§6.2.4）；排障 `docs/INT8_PREFILL_PERF.md` §1/§7.3。
7. **精度 / 容量验收**（GSM8K 200 ≈2.5 min；15×1Mi 长窗口 ≈3–3.5 h）：文本 GSM8K **96.0%**（vision 配置）/ **97.5%**（无 vision 基线）作参考；**视觉 43.5%（10/23）未过，视觉部分 WIP**；容量账与判据见 §6.2.4/§7.4/§11，排障 `a2_package/RUNBOOK.md`。

---

## 1. 产物 / 交付物总览

| 交付物 | 路径 | 说明 |
|---|---|---|
| 主补丁 | `patches/msmodelslim_v41_w4a8.patch` | 上游基线 `92e219f`；md5 `ff375f28e6cf5d7492e71b353d34ba5e` |
| 可选实验补丁 | `patches/msmodelslim_v41_w4a8_hiaux_optional.patch` | layers 37-39 路由专家 → W8A8；md5 `cdd41e7ea0c9981fdfc5d4f78cacab47` |
| 补丁说明 | `patches/PATCHES.md` | 基线/顺序/md5/apply 规则 |
| 一键 apply | `scripts/apply_msmodelslim_patch.sh` | `--check` / `--with-hiaux` / `--no-layout` |
| 产物校验器 | `scripts/quant_repro_check.py` | CPU/离线；输出 JSON manifest + PASS/FAIL |
| 基准 manifest | `logs/perf/quant_manifest_ref.json` | 由 vision 完整版 DSpark 产物生成，供 diff |
| 视觉验收 | `scripts/vision_accuracy_check.py` | ≥20 组图文问答，命中率/空乱码/HBM/TTFT/DSpark 记录 |
| 实施日志 | `logs/perf/msmodelslim_repro_impl.md` | 做了什么 / 未验证项 / 最容易踩的坑 |

参考指纹（§5 有完整表）：

| 产物 | 张量数 | W4A8_DYNAMIC | W8A8_DYNAMIC | FLOAT | 说明 |
|---|---|---|---|---|---|
| `v41-w4a8-stage1`（纯文本主干） | 185,734 | 184,320 | 744 | 670 | 80 分片 / 273 GB |
| `v41-w4a8-dspark`（+ MTP） | 186,960 | 184,320 | 744 | 1,896 | 85 分片 / 30 GB（主干为软链） |
| `v41-w4a8-dspark-vision-ref`（+ vision） | 187,226 | 184,320 | 744 | 2,162 | 86 分片 / 参考完整服务产物 |

---

## 2. 环境要求

### 2.1 硬件

| 用途 | 配置 |
|---|---|
| 量化（DP16） | 1 台 A3，`npu:0..15`（16 逻辑卡，8 物理 die），本机 1 TB HBM |
| 量化（单卡试跑） | 1 张逻辑卡即可（本仓库 partial7 实测） |
| 服务冒烟（视觉/文本） | TP8 + EP（8 逻辑卡），可选 DSpark 投机 |
| 长上下文 / 容量（档2） | **TP2/DP4**（8 逻辑卡）+ int8 KV；env 见 §6.2.3 |

本仓库的机器是共享机：量化/起服前确认卡的占用，`docker/enter.sh` 通过
`PRESET`/`DEVS`/`NAME` 控制挂哪些 `/dev/davinciN`。

### 2.2 镜像与软件版本（参考环境，来自 `docker/enter.sh` / `docs/AICORE_FAULT_RCA.md:266`）

| 组件 | 版本 |
|---|---|
| 镜像 | `quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3`（= `quay.io/ascend/vllm-ascend:deepseek-v4.1-flash-a3`，digest `1f2c08195c5b`） |
| CANN | 9.1.0（`ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.1.0`） |
| HDK / driver | 26.1.1（`ascendhal 7.35.23`） |
| Python | 3.12.13（镜像内 `/usr/local/python3.12.13`） |
| torch | 2.10.0（镜像内 CPU build，`+cpu`） |
| torch_npu | 2.10.0.post4 |
| transformers | 5.14.1（`src/va-main/pyproject.toml:20`） |
| vLLM core | 0.27.1 / `g6e448d0ea` |
| vllm-ascend | 镜像内 `e43cf1e9f`（`src/va-main` 本地克隆是另一个 commit，容器内以镜像为准） |
| msmodelslim | 包版本 26.1.0 + 本补丁（fork commit `e05b471`，基线 `92e219f`） |

> 版本不一致会改变算子行为/补丁落点；`docs/AICORE_FAULT_RCA.md:266` 明确记录该组合。

### 2.3 在容器里装 msmodelslim（离线 editable）

参考 `scripts/setup_mslim_inner.sh`：

```bash
H=/home/user; P=$H/projects/dsv41; V=$P/env/mslim-venv
python3 -V                                  # 必须是镜像内 python3.12.13
python3 -m venv $V                          # venv 里的 python 指向 /usr/local/python3.12.13/bin/python3
SYS=$(python3 -c "import site as s; print(s.getsitepackages()[0])")   # 镜像系统 site-packages
SITE=$($V/bin/python -c "import site as s; print(s.getsitepackages()[0])")
echo "$SYS" > "$SITE/system.pth"                                      # 让 venv 看到镜像里的 torch/torch_npu
$V/bin/pip install -q -U pip setuptools wheel
$V/bin/pip install -q -r $P/src/msmodelslim/requirements.txt
$V/bin/pip install -q -e $P/src/msmodelslim

# 关键：editable 安装后必须在包内补齐目录软链/配置（apply 脚本默认会做）
bash $P/scripts/apply_msmodelslim_patch.sh --dir $P/src/msmodelslim   # 或 --no-layout 后手动：
#   cp $R/config/config.ini $R/msmodelslim/config/config.ini
#   ln -sfn ../config        $R/msmodelslim/config_repo
#   ln -sfn ../lab_practice  $R/msmodelslim/lab_practice
#   ln -sfn ../lab_calib     $R/msmodelslim/lab_calib
```

- `requirements.txt`：`easydict==1.13`、`einops`、`pydantic>=2.10.1`、`accelerate>=0.28.0`、
  `requests`、`scipy`。`torch/torch_npu/transformers/safetensors` 由镜像系统 site-packages 提供。
- 本机 venv 的 `system.pth` 指向镜像内系统 site-packages，所以 **venv 必须在容器内创建**；
  在宿主机 `/home` 里直接跑 `env/mslim-venv/bin/python` 会因为找不到
  `/usr/local/python3.12.13/bin/python3` 而失败（本仓库现状即如此，属正常）。
- `scripts/prepare_mslim_layout.sh` 是同一逻辑的本机版本（硬编码 `R=$P/src/msmodelslim`），
  幂等，可重复执行；日志会出现 `The path ... is a soft link. Using its real path` 的
  WARNING，无害。

---

## 3. 数据准备

### 3.1 官方 checkpoint

- 官方 HF/ModelScope：`DeepSeek-V4.1-Flash`，FP8 + FP4 experts
  （`config.json.quantization_config = {"quant_method":"fp8","activation_scheme":"dynamic",
  "weight_block_size":[32,32],"scale_fmt":"ue8m0","expert_dtype":"fp4"}`）。
- 本机路径：`$H/models/DeepSeek-V4.1-Flash`，96,085 张量 / 48 分片，其中
  `vision.*=259`、`aligner.*=4`、`image_*=3`（合计 266 个视觉相关张量，全部 BF16，
  共 **970,536,960 B = 0.9705 GB**），`mtp.*=2401`。
- 完整性自检：`docs/STATUS.md` 记录了 48/48 分片逐键 + 长度校验通过。

### 3.2 阶段一装配：`scripts/make_full_model.py`

作用：把官方目录改造成“可量化/可运行”的目录，**零拷贝**（分片用软链）：

```bash
$V/bin/python $P/scripts/make_full_model.py \
  --src  $H/models/DeepSeek-V4.1-Flash \
  --dst  $H/models/DeepSeek-V4.1-Flash-stage1 \
  --layers 40 --mtp 0 --engram off --vision drop --calib-max-seq-len 8192
```

它做 4 件事：

1. 写新 `config.json`：`text_config.num_hidden_layers=40`、
   `num_nextn_predict_layers=0`、`engram_layer_ids=[]`、
   `max_position_embeddings=min(原值 1048576, calib-max-seq-len)`（所以是 **8192**，⚠️ 见 §9）；
2. 过滤 `model.safetensors.index.json`：只保留 `layers.0..39`、`mtp`（`--mtp 0` 时全删）、
   `image_*` 标量；`--vision drop` 时删除 `vision.*/aligner.*`；
3. 为被引用的官方分片创建**绝对路径软链**、软链 tokenizer/config 等；
4. 打印 `tensors=... shards=...`。

**软链 vs 实体**：装配目录只有软链，不占额外 476 GB；真正的量化输出
（`models/out/v41-w4a8-stage1`）是实体 273 GB。软链必须是**绝对路径**：相对软链按
“软链所在目录”解析，会静默指向不存在的目标，表现为 tokenizer 加载失败；用
`find -xtype l` 可查断链。`make_full_model.py` 已用 `os.path.abspath()` 修掉该坑
（`docs/STATUS.md` 踩坑记录）。

### 3.3 calibration 数据

配方末尾是 `dataset: mix_calib.jsonl`，实际文件为
`$P/src/msmodelslim/lab_calib/mix_calib.jsonl`（`lab_calib/` 通过 softlink 暴露给包内）。
量化日志会打印 `prepare dataset from mix_calib.jsonl success`。
`--calib-max-seq-len 8192` 只影响装配目录的 `max_position_embeddings`，校准序列长度由
dataset / runner 决定。

---

## 4. 打补丁与量化

### 4.1 补丁：apply / verify / rollback（基线 `92e219f`）

**两个补丁与指纹**（`md5sum` 可逐字节核对）：

| 补丁 | md5 | 规模 | 用途 |
|---|---|---|---|
| `patches/msmodelslim_v41_w4a8.patch` | `ff375f28e6cf5d7492e71b353d34ba5e` | 1687 行 / 9 文件 | 主补丁：`deepseek_v41` 适配器 + W4A8 配方 + `config.ini` 注册 |
| `patches/msmodelslim_v41_w4a8_hiaux_optional.patch` | `cdd41e7ea0c9981fdfc5d4f78cacab47` | 144 行 / 1 文件 | **可选实验**（layers 37–39 路由专家 → W8A8，§8.4），不进主线 |

```bash
P=/home/user/projects/dsv41
R=$P/src/msmodelslim
md5sum $P/patches/msmodelslim_v41_w4a8.patch $P/patches/msmodelslim_v41_w4a8_hiaux_optional.patch
git -C $R log -1 --format='%H %s'   # 期望 92e219fa9565a5bad84d90474a27bb11524d691c [Bugfix] analysis linear dp bugfix

# 1) 干跑：只 check，不改树；每个补丁必须 RC=0
bash $P/scripts/apply_msmodelslim_patch.sh --dir $R --check

# 2) 正式 apply（默认同时创建 editable 安装所需的包内 config/lab_* 软链）
bash $P/scripts/apply_msmodelslim_patch.sh --dir $R

# 2b) 可选实验分支（不进主线）
# bash $P/scripts/apply_msmodelslim_patch.sh --dir $R --with-hiaux

# 3) verify
ls $R/msmodelslim/model/deepseek_v41/   # __init__.py loader.py model_adapter.py model.py rotation_map.py convert.py
grep -n deepseek_v41 $R/msmodelslim/config/config.ini
test -L $R/msmodelslim/config_repo && test -L $R/msmodelslim/lab_practice && test -L $R/msmodelslim/lab_calib
$P/env/mslim-venv/bin/python -c "import msmodelslim; print(msmodelslim.__file__)"   # venv 必须在容器内

# 4) rollback（可选；顺序与 apply 相反，先反向 check 再 revert）
# git -C $R apply -R --check $P/patches/msmodelslim_v41_w4a8_hiaux_optional.patch   # 若打过 hiaux
# git -C $R apply -R --check $P/patches/msmodelslim_v41_w4a8.patch
```

- 手动等价命令、以及“不纳入补丁但必须现场准备”的软链清单见 `patches/PATCHES.md` §4/§6。
- 规则：**先 `--check` 再 apply；任何一步非 0 立即停**，不要 `--reject` / `--3way`。
  HEAD 不是 `92e219f` 时脚本只警告，以 `git apply --check` 返回值为准；apply 后应与
  `e05b471` 工作树逐文件一致（交付证据 `logs/perf/msmodelslim_repro_impl.md`）。

### 4.2 量化命令

**16 卡 DP（推荐，实测）** —— `scripts/quant_dp_inner.sh` 等价于：

```bash
cd $P
rm -rf $H/models/out/v41-w4a8-stage1 && mkdir -p $H/models/out/v41-w4a8-stage1   # 脚本会清空 SAVE
$V/bin/msmodelslim quant \
  --model_path $H/models/DeepSeek-V4.1-Flash-stage1 \
  --save_path  $H/models/out/v41-w4a8-stage1 \
  --config     $R/lab_practice/deepseek_v41/deepseek_v4_1_flash_w4a8.yaml \
  --device npu \
  --device_id 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
  --model_type deepseek_v41 --trust_remote_code true --log_level info
```

本仓库实际用的是 tmux 长程封装（含 docker 挂卡）：

```bash
LOG=$P/logs/quant_full_stage1_dp16.log bash $P/scripts/launch_full_tmux.sh
tail -f $P/logs/quant_full_stage1_dp16.log
```

配方（`lab_practice/deepseek_v41/deepseek_v4_1_flash_w4a8.yaml`）的处理顺序：

| # | processor | 作用 |
|---|---|---|
| 1 | `quarot` `block_size: 32` | QuaRot 全局/分块旋转，排除 MTP/Engram/vision/aligner |
| 2 | `flex_awq_ssz`（`up-down`） | 路由/共享专家 up-down 子图 SSZ AWQ，排除 shared_experts/MTP/Engram/vision/aligner |
| 3 | `flex_smooth_quant`（`norm-linear`） | attn/ffn 归一化前平滑，排除 `*ffn_norm*`、MTP/Engram/vision/aligner |
| 4 | `linear_quant` W8A8 | `*attn*`，再排除 `wo_a/wo_b`、`compressor.wgate/wkv`、`indexer.weights_proj`、**`indexer.wk`**、`indexer.wq_a`、MTP/Engram/vision/aligner |
| 5 | `linear_quant` W4A8(int4 SSZ) | `*ffn*` 路由专家，再排除 `*gate`、`*shared_experts*`、MTP/Engram/vision/aligner |
| 6 | `linear_quant` W8A8 | `*ffn.shared_experts*` |
| 7 | `ascendv1_saver` | `part_file_size: 4`（GiB/分片），输出 80 分片 |

**单卡**（试跑/小机器）：

```bash
MODEL=$H/models/DeepSeek-V4.1-Flash-stage1 SAVE=$H/models/out/v41-w4a8-1card \
  DEVICE_IDS='0' bash $P/scripts/quant_dp_inner.sh
```

### 4.3 预期耗时（实测日志）

| 配置 | 规模 | 起止 | 耗时 | 归一化 |
|---|---|---|---|---|
| A3 16 卡 DP | 40 层全量 | 2026-09-11 14:14:51 → 14:50:22 | **35 分 31 秒** | 0.89 分/层（`logs/quant_full_stage1_dp16.log`） |
| A3 16 卡 DP | partial7（7 层） | 13:58:53 → 14:10:35 | 11 分 42 秒 | 1.67 分/层（含启动固定开销） |
| A3 单卡（npu:0） | partial7（7 层） | 13:18:05 → 13:43:06 | 25 分 01 秒 | **≈3.6 分/层**（`logs/quant_partial7.log`） |

> ⚠️ 任务书里的“单卡 ≈8 分钟/层（40 层 ≈5.3 h）”是早期保守估算；本仓库现有
> 单卡 partial7 实测为 ≈3.6 分/层（含启动开销），按此外推 40 层 ≈2.4 h。两者都
> 只作容量规划参考，**本次没有单卡 40 层实测**。16 卡 DP 的 35 分 31 秒是实测值。

### 4.4 失败模式：`EI0005` / `EI0006`（DP 必须修）

- 现象：16 卡 DP 层间量化时 HCCL 报 `EI0005/EI0006`，各 rank `all_reduce` 参数不一致。
- 根因：`V41MoE.forward` 直接 `dist.all_reduce(y)`；DP 下各 rank 的 token 数不同，
  reduce 的 shape 不同。
- 修复（已含在主补丁）：先 `dist.all_gather` 各 rank 长度，用
  `DistHelper.gather_variable_shapes` 聚合 token，本地专家计算后 `all_reduce`，再
  切回本 rank 的 `[start_pos:end_pos]`（`msmodelslim/model/deepseek_v41/model.py`）。
- 若你在别的适配器看到同签名：先查所有集合通信操作的输入 shape 是否与 rank 相关。
- 服务侧的 `HCCL_BUFFSIZE` 是另一个问题：MoE `MoeDistributeDispatch` tiling 失败时需要
  `HCCL_BUFFSIZE=1024`（`serve_v2.sh` 默认值）；量化侧的修复与此无关。

---

## 5. 产物指纹与校验

### 5.1 目录结构（`models/out/v41-w4a8-stage1`）

```
config.json                              # 已由 prepare_runtime_ckpt.py 补 quantization_config
configuration.json                       # {"task": ...}
deepseek_v41_best_practice.yaml
quant_model_weights-00001..00080-of-00080.safetensors   # 273 GB / 80 分片
quant_model_weights.safetensors.index.json             # 185,734 键
quant_model_description.json                           # 185,739 项（含元数据/desc-only）
optional/quarot.safetensors                            # global_rotation F32 [5120,5120] / 100 MB
tokenizer.json / tokenizer_config.json
```

`v41-w4a8-dspark` 在此基础上：主干 80 分片的软链 + `mtp-0000{1..4}.safetensors`
（1,224 张量）+ `mtp-vocab-00001-of-00001.safetensors`（2 张量，`add_draft_vocab.py`）
+ 更新后的 index/description。

### 5.2 标签分布（只在 index 键上计数，避免元数据干扰）

| 产物 | 总数 | W4A8_DYNAMIC | W8A8_DYNAMIC | FLOAT |
|---|---|---|---|---|
| stage1（文本主干） | 185,734 | 184,320（40×384×3×4） | 744 | 670 |
| + MTP（DSpark） | 186,960 | 184,320 | 744 | 1,896（670+1,226） |
| + vision（参考完整服务产物） | 187,226 | 184,320 | 744 | 2,162（670+1,226+266） |

> ⚠️ 直接对 `quant_model_description.json` 做 `Counter(values)` 会多出元数据字符串：
> `version="1.0.0"`、`model_quant_type="W8A8_DYNAMIC"`（混合导出时 W4A8 被 saver
> 优先级隐藏）。所以“字符串计数”看起来是 W8A8=745、W4A8=184,320+；**真实张量标签以
> index 键为准**。DSpark 的 description 还保留了 1,177 个 `mtp.*.scale`（FLOAT）但
> 不写盘，也只在 desc-only 里出现。

### 5.3 代表性张量（stage1；dspark/vision 为增量）

| key | dtype | shape | label |
|---|---|---|---|
| `layers.0.ffn.experts.0.w1.weight` / `w3.weight` | I8 | [1152, 5120] | W4A8_DYNAMIC |
| `layers.0.ffn.experts.0.w1.weight_scale` / `weight_offset` / `scale_bias` | F32 | [2304, 1] | W4A8_DYNAMIC |
| `layers.0.ffn.experts.0.w2.weight` | I8 | [2560, 2304] | W4A8_DYNAMIC |
| `layers.0.ffn.experts.0.w2.weight_scale` / `weight_offset` | F32 | [5120, 1] | W4A8_DYNAMIC |
| `layers.0.ffn.experts.0.w2.scale_bias` | F32 | [5120, 16] | W4A8_DYNAMIC |
| `layers.2.attn.wq_a.weight` | I8 | [1280, 5120] | W8A8_DYNAMIC |
| `layers.2.attn.wq_a.weight_scale` / `weight_offset` | F32 | [1280, 1] | W8A8_DYNAMIC |
| `layers.0.attn.wq_b.weight` | I8 | [32768, 1280] | W8A8_DYNAMIC |
| `layers.0.attn.wkv.weight` | I8 | [512, 5120] | W8A8_DYNAMIC |
| `layers.0.ffn.shared_experts.w1.weight` | I8 | [2304, 5120] | W8A8_DYNAMIC |
| `layers.2.attn.indexer.wq_b.weight` | I8 | [4096, 1280] | W8A8_DYNAMIC |
| `layers.2.attn.indexer.wk.weight` | BF16 | [128, 512] | FLOAT |
| `layers.0.attn.wo_a.weight` / `wo_b.weight` | BF16 | [8192,4096] / [5120,8192] | FLOAT |
| `layers.0.ffn.gate.weight` | BF16 | [384, 5120] | FLOAT |
| `layers.0.ffn.gate.bias` / `bias_vl` | F32 | [384] | FLOAT |
| `layers.0.attn.attn_sink` | F32 | [64] | FLOAT |
| `layers.0.attn.q_norm.weight` / `kv_norm.weight` | F32 | [1280] / [512] | FLOAT |
| `layers.0.hc_attn_fn` / `hc_attn_base` / `hc_attn_scale` | F32 | [24,20480] / [24] / [3] | FLOAT |
| `layers.2.attn.compressor.wkv.weight`（ratio 2） | F32 | [512, 5120] | FLOAT |
| `layers.20.attn.compressor.wkv.weight`（ratio 1） | BF16 | [512, 5120] | FLOAT |
| `embed.weight` / `head.weight` | BF16 / F32 | [129280, 5120] | FLOAT |
| `norm.weight` | F32 | [5120] | FLOAT |
| `optional/quarot.safetensors :: global_rotation` | F32 | [5120, 5120] | —（附件） |
| MTP: `mtp.0.main_proj.weight`（预旋转） | BF16 | [5120, 15360] | FLOAT |
| MTP: `mtp.0.ffn.experts.0.w1.weight` | BF16 | [2304, 5120] | FLOAT |
| MTP: `mtp.0.embed.weight` / `mtp.2.head.weight` | BF16 | [129280, 5120] | FLOAT |
| vision: `vision.patch_embed.proj.weight` | BF16 | [1024, 588] | FLOAT |
| vision: `vision.blocks.0.attn.wqkv.weight` | BF16 | [3072, 1024] | FLOAT |
| vision: `vision.blocks.0.mlp.w1.weight` | BF16 | [5632, 1024] | FLOAT |
| vision: `aligner.w1.weight` / `aligner.w2.weight` | BF16 | [5120,9216] / [5120,5120] | FLOAT |
| vision: `image_start` / `image_end` / `image_newline` | BF16 | [5120] | FLOAT |

### 5.4 `quant_model_description.json` 必须字段

| 字段 | 值 / 要求 |
|---|---|
| `version` | `"1.0.0"` |
| `model_quant_type` | `"W8A8_DYNAMIC"`（混合 W4A8+W8A8 导出的事实记录值） |
| `group_size` | `0` |
| `metadata` | `{}`（可扩展） |
| `optional.quarot.rotation_map.global_rotation` | `"optional/quarot.safetensors"`（运行时 `get_rotation_path()` 契约） |
| 每个 index 键 | 一个字符串标签：`W4A8_DYNAMIC` / `W8A8_DYNAMIC` / `FLOAT` |
| DSpark 附加 | 所有 2,401 个官方 `mtp.*` 名字（含不写盘的 1,177 个 `.scale`）标为 `FLOAT` |

### 5.5 校验步骤

**第一步：仓库原有 `scripts/check_quant_output.py`（快速人工核对，无第三方依赖）**

```bash
python3 $P/scripts/check_quant_output.py --dir $H/models/out/v41-w4a8-stage1
# 打印分片大小、description 标签计数、index 张量数/分片数、quarot 存在性、
# config.quantization_config 等；用于肉眼确认目录结构，不做 PASS/FAIL。
```

**第二步：`scripts/quant_repro_check.py`（严格指纹校验；CPU/离线）**

用法与退出码：

```bash
python3 $P/scripts/quant_repro_check.py --dir <产物目录> \
  [--official-dir <官方 ckpt>] [--out <manifest.json>] \
  [--allow-missing-vision] [--require-mtp] [--mtp auto|float|quantized] [--allow-extra-shards]
# 退出码：0=PASS，1=FAIL；--allow-missing-vision 仅用于文本主干 waiver（manifest 记 VISION_MISSING=WARN）
```

CPU/离线只读 JSON + safetensors header；输出 JSON manifest（张量总数、按标签计数、
代表性张量 key→shape/dtype、rotation 附件、description 字段）并给 PASS/FAIL。
**以本轮实测为准**（`logs/perf/p40_quant_manifest_service.json`；旧文用 `-ref` 举例的
22 PASS 示例值已废弃——服务目录口径为 23 PASS）：

| 目录 | profile | index | vision | checks | verdict |
|---|---|---|---|---|---|
| `v41-w4a8-dspark-vision`（服务目录） | `dspark-vision` | **187,226 张量 / 86 分片**；FLOAT 2,162 / W4A8 184,320 / W8A8 744 | **266/266 = 970,536,960 B（0.9705 GB），全 BF16** | **23 PASS / 0 WARN / 0 FAIL** | **PASS** |
| `v41-w4a8-dspark-vision-ref`（冻结参考） | `dspark-vision` | 同上 | 同上 | 22 PASS / 1 WARN / 0 FAIL（WARN: `max_position_embeddings=8192`） | PASS |
| `v41-w4a8-stage1` + `--allow-missing-vision` | `trunk-text` | 185,734 / 80 分片 | —（waiver） | 18 PASS / 2 WARN / 0 FAIL | PASS |
| `v41-w4a8-dspark` + `--allow-missing-vision` | `dspark-text` | 186,960 | —（waiver） | 18 PASS / 2 WARN / 0 FAIL | PASS |

服务目录 = `-ref` 复制后把 `config.json` 真文件化，并把 `text_config.max_position_embeddings`
从 8192 恢复到 **1048576**（所以 23 PASS / 0 WARN；`-ref` 仍是 8192 → 1 WARN）。
vision 分片由 `add_vision_weights.py` 确定性写出，**md5
`999935d7debdb116b2cbf482ec32b6f8`（服务目录与 `-ref` 逐字节一致）**；checker 的
key/shape/dtype 指纹为 `sha256=f9f80643bac8dc362b3a144e60d58413f2395abc8fc07a8c8a3e83e423449641`。

```bash
# 严格模式（默认要求 266 vision；缺 → FAIL）
sudo -n python3 $P/scripts/quant_repro_check.py \
  --dir $H/models/out/v41-w4a8-dspark-vision \
  --official-dir $H/models/DeepSeek-V4.1-Flash \
  --out $P/logs/perf/p40_quant_manifest_service.json
# => profile dspark-vision, 187,226 tensors, vision 266/266 (0.9705 GB), 23 PASS / 0 FAIL, verdict PASS

# 复核 vision 分片与 ref 逐字节一致
md5sum $H/models/out/v41-w4a8-dspark-vision/vision-00001-of-00001.safetensors
md5sum $H/models/out/v41-w4a8-dspark-vision-ref/vision-00001-of-00001.safetensors

# 历史文本产物（无 vision）：显式 waiver
sudo -n python3 $P/scripts/quant_repro_check.py \
  --dir $H/models/out/v41-w4a8-stage1 --allow-missing-vision
# => 18 PASS / 2 WARN / 0 FAIL（WARN: max_position_embeddings=8192, VISION_MISSING）

# DSpark 文本产物（无 vision）
sudo -n python3 $P/scripts/quant_repro_check.py \
  --dir $H/models/out/v41-w4a8-dspark --allow-missing-vision
# => 18 PASS / 2 WARN / 0 FAIL
```

校验项：pattern×label 精确对账（109 个 pattern）、代表性张量、每个分片存在且包含
index 映射的全部键、文件长度与 header 一致、quarot 附件、description 必填字段、
config（`quantization_config`/层数/MTP 数/vision_config）、vision 键集合/形状/dtype/
字节数/指纹、MTP 必须 FLOAT（`--mtp quantized` 可切换到官方对齐实验检查）、
`observations_sha256` 供他人 diff。

> 容器写出的产物常是 root 0600（umask 077），宿主机普通用户读不了；用 `sudo -n` 跑
> 校验器，或建产物时设 `umask 022`。校验器不会写产物目录（manifest 写到 `--out`）。

---

## 6. 运行时组装

### 6.1 `scripts/prepare_runtime_ckpt.py`：补 `quantization_config`

```bash
$V/bin/python $P/scripts/prepare_runtime_ckpt.py --dir $H/models/out/v41-w4a8-stage1
```

它把 `config.json` 改成：

```json
"quantization_config": {"quant_method": "ascend", "model_quant_type": "W4A8_DYNAMIC"}
```

并检查 `quant_model_description.json`、index 统计。没有这个字段，vLLM-Ascend 的
`override_quantization_method` 不会走 ascend 量化分支（参考 `docs/STATUS.md`：
官方 Aurora W8A8 导出同写法）。幂等，可重复跑。

### 6.2 起服与评测（文本）：唯一正确链 / UNSAFE 警告 / 档2 env / 6 个启动数

> 本节是“起服/评测”的唯一权威口径。**旧文里的 `MAX_LEN=8192`、TP8、BAT=4096、“仅 int8 就够”、
> “plane-tight 缺失 / 15×1Mi 不可行”均为旧值已废弃**（见 §6.2.3/§6.2.4）。
>
> 失败排查入口：`a2_package/RUNBOOK.md`（若树上还没有该文件，先看 `a2_package/README.md`
> 的同名章节；注意其 int8 CHAIN 段与“plane-tight 待补 / 15×1Mi 不可行”为旧值已废弃）。

#### 6.2.1 唯一正确的服务补丁链（容器内；先 `--check` 再 `apply`）

顺序：`canonical → merged(statefix2+p3_sc) → kv_layout_depad → kv_layout_plane_tight →
int8_kv → int8_kv_spec → capturefix → capturefix2`（8 个 vllm-ascend 补丁）；vLLM core
独立仓库另打 `admission_gate.patch`。任何一步非 0 立即停，不要 `--reject`/`--3way`；
canonical 与 merged 的规则/证据见 `docs/PATCH_APPLY_ORDER.md` §0/§4。

| # | 补丁（`$P/patches/`） | md5 | 作用 |
|---|---|---|---|
| 1 | `kv_offload_p2.patch` | `4b7f268c4a264a0f9af82607bbb111ca` | canonical host-tier 基座 |
| 2 | `kv_offload_p2_statefix2_p3sc.patch` | `b24719c863459ae7f51b1d00e33e4b9e` | statefix2 + p3_sc（合并 delta；**禁止**再叠单件 statefix2/p3_sc） |
| 3 | `kv_layout_depad.patch` | `9fcbfbc2e50882563d6a7c32cb8dee01` | plane-equal / 去 padding |
| 4 | `kv_layout_plane_tight.patch` | `862f6f68f973469c3cd0b38eb60f1102` | plane-tight（int8-544 容量第三杠杆） |
| 5 | `kv_offload_p2_int8_kv.patch` | `164735325685b02cfa159e798cb9b351` | P19 int8 long-KV |
| 6 | `kv_offload_p2_int8_kv_spec.patch` | `0ae5e063c8df33b544153913623cbc08` | P19-S DSpark 兼容 |
| 7 | `kv_offload_p2_int8_kv_capturefix.patch` | `b91d0aca308a112e249813a6e9c010e4` | capturefix（dummy/`token_to_req` + 1-tuple） |
| 8 | `kv_offload_p2_int8_kv_capturefix2.patch` | `f4c812733857b38a29dd1a9d48249f69` | capturefix2（图内读路径） |
| 9 | `admission_gate.patch`（vLLM core） | `8243dff6c9dc3d87805f1dfd7820c23f` | 每 step 纯 prefill / 纯 decode；env `VLLM_ADMISSION_GATE=1` |

```bash
P=/home/user/projects/dsv41
cd /vllm-workspace/vllm-ascend
for p in kv_offload_p2.patch kv_offload_p2_statefix2_p3sc.patch \
         kv_layout_depad.patch kv_layout_plane_tight.patch \
         kv_offload_p2_int8_kv.patch kv_offload_p2_int8_kv_spec.patch \
         kv_offload_p2_int8_kv_capturefix.patch kv_offload_p2_int8_kv_capturefix2.patch; do
  echo "== $p"; git apply --check "$P/patches/$p" && git apply "$P/patches/$p" || exit 1
done
cd /vllm-workspace/vllm
git apply --check "$P/patches/admission_gate.patch" && git apply "$P/patches/admission_gate.patch" || exit 1
# verify：apply 痕迹
git -C /vllm-workspace/vllm-ascend status --porcelain | head
git -C /vllm-workspace/vllm        status --porcelain | head
```

> A3 一键包装（会起容器，**本文不执行**）：`logs/perf/p36_capturefix2_start.sh` 按上述 9 补丁
> apply，再调 `scripts/serve_v2.sh` 起服。它默认还会在补丁后执行 `p36_fix_tokreq.py` /
> `p36_fix_int8_read.py` / `p36_fix_prefill_incr.py` 三个运行时脚本；**正确链必须关掉**：
> `FIX_TOKREQ=0 FIX_READ=0 FIX_PREFILL=0`（`token_to_req` synthesis 已由 capturefix 内建；
> P42 的 `p42_long_window.sh` 就是向 `p36_phase.sh` 这样传的，launcher 另显式指定，见 §6.2.3）。

#### 6.2.2 ⛔ 禁止打的两个 UNSAFE 补丁（其中 `incr` 默认开！）

| 补丁 | md5 | 默认 | 结论 |
|---|---|---|---|
| `kv_offload_p2_int8_prefill_incr.patch` | `a34f95908ebdec68892346ad1366e85c` | **on**（`V41_KV_INT8_PREFILL_INCR` 缺省开） | **UNSAFE — NO-GO** |
| `kv_offload_p2_int8_prefill_staging_delta.patch` | `713ac7e44a3fb8111903bf8e7ab7aa43` | off（`V41_KV_INT8_PREFILL_STAGING=delta` 才开） | **UNSAFE — NO-GO** |

- 根因（结构性，H1）：4 个 long-KV source（layers 2/8/14/20）共享同一块
  `self._staging[0, max_model_len)`（`v41_int8_kv.py` docstring / `view_for()` 无 per-source
  偏移），一次 prefill forward 内按层序 2→8→14→20 重建；增量 planner 只写 `[start_at, prefix)`，
  第 2 步起 `[0,start_at)` 已是**别的 source 的 KV**（100% 命中），另有跨请求同 `first_gid`
  复用（H2）不可排除。表现为答案坏但**不报错**。
- 证据：`docs/INT8_PREFILL_DELTA_SAFETY.md` §0/§1/§3、`patches/UNSAFE_prefill_incr_delta.md`、
  CPU 复现 `logs/perf/p42_prefill_incr_delta_safety_cpu.log`（scenario E/F）。
- 处置：正确链**根本不要 apply 这两个补丁**。若镜像/旧链已经打进 incr，起服前必须
  `V41_KV_INT8_PREFILL_INCR=0` 且 `V41_KV_INT8_PREFILL_STAGING=legacy`（等价于不设）。
- P36 报的 incr 769 / 827–860 tok/s 是错误 KV 下的速度，只能当上界，不能作为交付结论；
  正确提速路径 = legacy 全量 dequant + 提高 BAT（BAT 1024→4096 dequant 总量约 1/4），但受
  §6.2.4 容量约束，**本轮选用 BAT=2048**。

#### 6.2.3 档2 正确 env（TP2/DP4 + int8-544/plane-tight + Engram DRAM + tier off）

```bash
P=/home/user/projects/dsv41; H=/home/user
export MODEL=$H/models/out/v41-w4a8-engram-dr   # 必须含 engram_layer_ids=[1,14]；不是 text-only 的 -dspark/-stage1
export TP=2 DP=4 EP=1 PORT=8000 SERVED_NAME=deepseek-v41
export MAX_LEN=1048576 MAX_SEQS=15 BAT_TOKENS=2048
export GPU_UTIL=0.97 BLOCK=128 KV_DTYPE=bfloat16
export GRAPH=1 EAGER=0 SPEC=1 SP_TOKENS=7 SPEC_EAGER=1
export NPUGRAPH_EX=1 STATIC_KERNEL=0 CPU_BIND=0 MULTISTREAM=1 DSA_OVERLAP=1
export KV_INT8=1 V41_KV_INT8=1                    # 前者给 a2 launch_server.sh 映射，后者是运行时真值
export V41_KV_LAYOUT=plane-tight
export V41_KV_INT8_CAPTURE_FIX=1 V41_KV_INT8_WINDOW_REUSE=1
export ENGRAM=1 ENGRAM_STORAGE=int8 V41_ENGRAM_HOST_RESIDENT=1
export ENGRAM_REUSE_EP_GROUP=1 V41_ENGRAM_REUSE_EP_GROUP=1
export ENGRAM_GATE_CHUNK=0 V41_ENGRAM_GATE_CHUNK=0
export V41_KV_TIER=off
export ADMISSION_GATE=1 VLLM_ADMISSION_GATE=1
export NO_ASYNC_SCHEDULING=1       # 硬前提；等价 vllm 参数 --no-async-scheduling
export LOADER_MT=1 LAZY=1 VISION=0
export VLLM_ENGINE_READY_TIMEOUT_S=3600
# 不要设 V41_KV_INT8_PREFILL_STAGING（缺省 legacy）；绝不开 V41_KV_INT8_PREFILL_INCR
```

> 档2 服务模型**不是** §6.3 产出的 `v41-w4a8-dspark`：必须用 Engram 派生目录
> `models/out/v41-w4a8-engram-dr`（`engram_layer_ids=[1,14]`、含 10 个 engram int8 键、
> `max_position_embeddings=1048576`；186,970 张量；由 Engram-on-DRAM 转换流程生成，
> A2 包把它称作 P18 资产）。若把 `MODEL` 指到 `v41-w4a8-dspark*`，会因 Engram 权重缺失
> / 空 `engram_layer_ids` 而加载失败或静默退化。

> 变量名同时兼容 `a2_package/scripts/launch_server.sh`（读 `KV_INT8`/`ENGRAM`/
> `NO_ASYNC_SCHEDULING` 等别名）与 `scripts/serve_v2.sh`（直接继承 `V41_*`/`VLLM_ADMISSION_GATE`，
> 但 `--no-async-scheduling` 必须走 `EXTRA`）；档2 模板见
> `a2_package/configs/a2_tp2dp4_int8_15x1m.env`。

- **`--no-async-scheduling` 是硬前提**：async 会让 planner 的 in-flight 翻倍
  （BAT1024→A=18、BAT2048→A=34、BAT4096→A=66），rank_cap 直接掉，15×1Mi 不成立；
  代价是单流步时约 +40%。若直接调 `scripts/serve_v2.sh`，用 `EXTRA='--no-async-scheduling'`
  （`serve_v2.sh` 不读 `NO_ASYNC_SCHEDULING`）。
- KV：`--kv-cache-dtype` 保持 `bfloat16`，int8 在**存储侧**（`V41_KV_INT8=1` +
  `V41_KV_LAYOUT=plane-tight` → `row_bytes=544`）；不要传 `--kv-cache-dtype int8`。
- `V41_KV_TIER=off`：int8 与 host tier 互斥（开启会报错/走错路径）。
- `ENGRAM=1` 需要 Engram 派生目录 + host mount（R1 复用 EP HCCL 域，R2 本轮关闭
  `ENGRAM_GATE_CHUNK=0`）；`v41-w4a8-dspark-vision` 的 `engram_layer_ids=[]`，视觉线必须 `ENGRAM=0`。
- `CPU_BIND=0` / `STATIC_KERNEL=0`：避开已记录的 NPU12–15 cpu binding 挂死与静态编译长窗口风险。
- A3 安全参考起服（host 包装，会起容器；**必须显式 `LAUNCHER`**——`p42_long_window.sh`
  默认用 `p42_delta_start.sh`，含 UNSAFE `staging_delta`，不要用）：

```bash
TAG=legacy_b2048 BAT=2048 N=15 LAUNCHER=$P/logs/perf/p36_capturefix2_start.sh \
  KEEP_UP=1 SKIP_CAP=1 SKIP_ACC=1 SKIP_SANITY=1 \
  bash $P/logs/perf/p42_long_window.sh
```

  通用入口是 apply 补丁后 `source` 上述 env + `bash scripts/serve_v2.sh`
  （`EXTRA='--no-async-scheduling'`）。

#### 6.2.4 六个启动数核对与三档 BAT 实测

起服 healthy 后在 serve 日志 grep（六个数缺一不可）：

```bash
grep -nE "Available KV cache memory|V4.1 int8 long-KV capacity|GPU KV cache size|admission_gate\] enabled" serve.log | head -20
```

| 启动数 | 日志/公式 | BAT=1024 | BAT=2048（**当前选用**） | BAT=4096（**不可用**） |
|---|---|---|---|---|
| Available KV cache memory | `Available KV cache memory:` | 10.02 GiB | **9.61 GiB** | 8.79 GiB |
| num_blocks | `num_blocks=` | **42,156** | **39,896** | **35,449** |
| A（max in-flight） | `request_blocks=8195+43A` ⇒ `A=(rb-8195)/43` | 10 | 18 | 34 |
| rb（request_blocks） | `request_blocks(A)=` | 8,625 | 8,969 | 9,657 |
| rank_cap | `rank_cap=` | 4 | **4** | **3** |
| BF16 staging / workspace | `BF16 staging ... workspace budget` | 1.23 / 1.49 GiB | 1.23 / 1.53 GiB | 1.23 / 1.61 GiB |

- 口径：TP2/DP4、sync/no-async、util 0.97、plane-tight int8-544、R1-on/R2-off、GRAPH=1、
  `max_num_seqs=15`；数据来自 P42 启动账（`logs/perf/p42_run_legacy_b2048.log` 等）。
  `num_blocks` 各 rank 有 ±10 抖动，表中取最小值（实测范围：BAT=1024 42,156–42,158、
  BAT=2048 39,896–39,906、BAT=4096 35,449–35,454）。
- 15×1Mi 需要 `rank_cap≥4`（node_cap=4×rank_cap≥15）：BAT=2048 时
  `1+4×8,969=35,877 ≤ 39,896`，余 4,019 blocks，**容量账成立**；BAT=4096 时
  `1+4×9,657=38,629 > 35,449` ⇒ rank_cap=3、node_cap=12，**不可用**。BAT=1024 余量最大，
  但 legacy 全量 dequant 总量约为 BAT=2048 的 2 倍。
- **不要**用 BAT=4096 换“dequant 少 4 倍”：BAT 越大 in-flight/激活 workspace 越大
  （1.49→1.53→1.61 GiB），KV 池 `num_blocks` 反而越小（旧值“BAT=4096 更快更省”已废弃）。
- ⚠️ 本轮 P42 legacy_b2048 只跑到单条 1Mi timing（1186.6 tok/s，见
  `logs/perf/p36_phase0/p42_long_window_summary_legacy_b2048.md`）；**15 条驻留 + hold
  未在本轮端到端跑完**，不要写成“15×1Mi 已通过”。

#### 6.2.5 起服/评测两处接口修正（P40 vision 首跑暴露，必须）

1. **`--tokenizer-mode=deepseek_v4`**（不是 `deepseek_v41`）。镜像 renderer/tokenizer registry
   没有 `deepseek_v41` 映射，默认走 HfRenderer；本 checkpoint 的 `tokenizer_config.json`
   没有 `chat_template`，`/v1/chat/completions` 直接 400 `"must provide a chat template"`。
   400 原始日志：`logs/perf/p40_serve_vision_attempt1_hfrenderer.log`；修复后命令见
   `logs/perf/p40_serve_vision.log:22`。
2. **`--default-chat-template-kwargs={"enable_thinking":false}`**。DSV4.1 chat 默认 thinking；
   vision 验收脚本 `max_tokens=32` 的短答会被思考段占满，判分不可用。镜像 `cli_args.py`
   推荐值就是 `{"enable_thinking": false}`。经 `EXTRA` 传入时必须用 `--flag=value` 无空格形式。

视觉（本轮真机口径；WIP，见 §7.4；注意 `ENGRAM=0`）：

```bash
MODEL=$H/models/out/v41-w4a8-dspark-vision TP=8 DP=1 PORT=8001 \
  VISION=1 SPEC=1 SP_TOKENS=7 ENGRAM=0 GRAPH=1 \
  MAX_LEN=1048576 MAX_SEQS=8 BAT_TOKENS=4096 GPU_UTIL=0.94 \
  EXTRA='--tokenizer-mode=deepseek_v4 --default-chat-template-kwargs={"enable_thinking":false}' \
  bash $P/scripts/serve_v2.sh
```

文本评测走 `/v1/completions` + 官方 encoder（`docs/ACCURACY.md`）时不依赖这两项；但任何
`/v1/chat/completions`（含 `vision_accuracy_check.py`）缺一不可。

#### 6.2.6 文本冒烟 + GSM8K 小样本（参考值 96.0% / 97.5%）

**离线 `LLM(...)` 冒烟（文本，image=0）**：`scripts/smoke_w4a8.sh` 用
`limit_mm_per_prompt={"image": 0}`，greedy 问“2+2”“中国首都”。历史结果：TP8+EP
加载成功、greedy 问答正确（`docs/STATUS.md` 2026-09-12 快照）。

**起服（vLLM-Ascend，含 DSpark）**：正确链 / env 见 §6.2.1/§6.2.3。
`scripts/serve_v2.sh` 是通用启动器（§6.2.5 的 `EXTRA` 两参数对它同样适用）；
旧文示例的 `MAX_LEN=8192 / gpu-util 0.90 / TP8` 是早期文本冒烟口径，**旧值已废弃**
（长上下文/容量一律用 §6.2.3 的 1Mi + util 0.97 + TP2/DP4）。DSpark 参数：
`--speculative-config '{"method":"dspark","num_speculative_tokens":7,"enforce_eager":true}'`
（接受率问题仍见 §6.3）。

**GSM8K 200（8-shot，chat，官方 encoder 生成 prompt，temperature=0）**：

```bash
cd $H; export HF_ENDPOINT=https://hf-mirror.com   # 离线时用本地 HF datasets 缓存
~/venvs/lmeval311/bin/python $P/scripts/acc_eval.py --task gsm8k --limit 200 \
  --mode chat --conc 4 --max-tokens 512 --base-url http://127.0.0.1:8000 \
  --out ~/lmeval_out/w4a8_vision_gsm8k200.json --tag w4a8vision
```

- 抽取器取 `####` 后数字；断点续跑写 `<out>.jsonl`；`--limit 50` 小样本命令同旧文。
- **参考实测（全 200 题，0 空 / 0 错误）**：
  - vision 服务（本轮 canonical SPEC=7，`v41-w4a8-dspark-vision`）**96.0%（192/200）**，
    证据 `logs/perf/p40_gsm8k200_vision.json`；
  - 无 vision 基线（`v41-w4a8-dspark`）**97.5%（195/200）**，见 `docs/ACCURACY.md` §1。
  - 差值 -1.5pp；未做同配置 VISION=0 A/B（文本 token 不经过 mm processor），按参考值使用。
- 视觉精度见 §7.4：**未过（43.5%）**，本指南视觉部分 WIP。

### 6.3 DSpark：`scripts/make_dspark_ckpt.py` + `scripts/add_draft_vocab.py`

DSpark 目录 = 量化主干 + 官方 `mtp.*`（3 个 draft 层）+ draft 词表。要点：

1. **MTP 反量化**：官方 MTP 权重是 FP8/FP4；脚本按
   `msmodelslim/model/deepseek_v41/convert.py` 的语义（FP8 32×32、FP4 block=32、
   UE8M0 scale、低 nibble 优先）在 CPU 上逐张量反量化成 **BF16**，`.scale` 兄弟键不写盘。
   其余 MTP 张量（norm/hc_*/attn_sink/main_proj/…）原 dtype 拷贝，所以 MTP 全部 FLOAT。
   本仓库曾用纯 Python 独立解码逐元素复核 FP4/FP8 误差为 0（`docs/STATUS.md`）。
2. **`main_proj` 预旋转**：量化模型残差流是 `x' = Rᵀ x`，而 draft 权重在原始基；
   必须 `main_proj' = W @ blockdiag(R, R, R)`（输入维 = `len(target_layer_ids)×hidden`，
   按 hidden 分块右乘 `global_rotation`）。V4.1 运行时**不会**自动旋转 `main_proj`，
   所以这一步必须在 checkpoint 里做完。默认从 `--quant-dir/optional/quarot.safetensors`
   读 `global_rotation`；`--no-rotate-main-proj` 只用于对照/排障。
3. **index/description/config**：index = 量化 map + 每个写出的 MTP 张量；description
   把所有 2,401 个官方 `mtp.*`（含跳过的 `.scale`）标为 FLOAT；`config.json` 设
   `num_nextn_predict_layers=3`，其余不动。写盘流式、目标分片 ~8 GiB、幂等。
4. **draft 词表**：`add_draft_vocab.py` 追加 `mtp.0.embed.weight`（官方未旋转
   `embed.weight`）与 `mtp.2.head.weight`（官方未旋转 `head.weight`），共 2 个 BF16
   张量（`mtp-vocab-00001-of-00001.safetensors`）。缺这两个时运行时会把已 QuaRot 旋转的
   主干 embed/head 共享给 draft，输入嵌入与输出 logits 不同基 → 接受率≈1.0
   （`docs/STATUS.md` 2026-09-12）。`add_draft_vocab.py` 断言 `num_nextn_predict_layers=3`。

命令：

```bash
Q=$H/models/out/v41-w4a8-stage1; D=$H/models/out/v41-w4a8-dspark
$V/bin/python $P/scripts/make_dspark_ckpt.py --quant-dir $Q \
  --official-dir $H/models/DeepSeek-V4.1-Flash --out-dir $D --dry-run   # 先看计划
$V/bin/python $P/scripts/make_dspark_ckpt.py --quant-dir $Q \
  --official-dir $H/models/DeepSeek-V4.1-Flash --out-dir $D
$V/bin/python $P/scripts/make_dspark_ckpt.py --quant-dir $Q \
  --official-dir $H/models/DeepSeek-V4.1-Flash --out-dir $D --verify-only
$V/bin/python $P/scripts/add_draft_vocab.py \
  --official-dir $H/models/DeepSeek-V4.1-Flash --dspark-dir $D
```

⚠️ **已知未解**：DSpark 位置 0 接受率 ≈1.4%、平均接受长度 ≈1.01（目标 3.33–3.45），
单流反而比不开投机慢；已排除反量化误差/旋转方向/双重旋转/权重名映射，仍在排查
（`docs/STATUS.md` 2026-09-12；`patch`/`dspark_proposer.py` 对照件在 `patches/draft_graph/`）。
本指南不宣称 DSpark 已可用；文本主干的 W4A8 质量另见 `docs/ACCURACY.md`。

---

## 7. 视觉能力：必须把 vision/aligner 补回（验收要求）

### 7.1 事实：现有 msmodelslim 产物不含视觉权重

- `make_full_model.py --vision drop` 把 `vision.*/aligner.*` 从装配 index 删掉；只留
  `image_start/end/newline` 3 个标量；
- **即使装配时 `--vision keep`，msmodelslim 的 saver 也不会写出它们**：
  `DeepSeekV41ModelAdapter` 只把 `V41Transformer` 的参数（`get_state_dict` 遍历
  `module.named_parameters()`）交给 `AscendV1Saver`，`post_run` 只 `copy_files()` 复制
  非权重文件。证据：装配目录 `DeepSeek-V4.1-Flash-stage1` 含 `image_start/end/newline`，
  但量化产物 `v41-w4a8-stage1` index 里**一个都没有**（`docs/STATUS.md` 也记录了
  stage1 有 3 个 image 标量、输出没有）。
- 因此：**“量化装配阶段保留 vision”单独做不够**（除非同时改 adapter/saver）；官方
  Aurora 的导出是把 vision 当 FLOAT 一并带出（见 §8）。

### 7.2 路径 ①（推荐）：量化文本模型后用 `add_vision_weights.py` 并回

```bash
# 不要改已有产出：先复制（dspark 目录里主干是软链，复制开销是 MTP 实体 ~30 GB）
cp -a $H/models/out/v41-w4a8-dspark $H/models/out/v41-w4a8-dspark-vision
$V/bin/python $P/scripts/add_vision_weights.py \
  --official-dir $H/models/DeepSeek-V4.1-Flash \
  --target-dir  $H/models/out/v41-w4a8-dspark-vision
# 写出 vision-00001-of-00001.safetensors：266 张量 / 970,536,960 B / 全部 BF16
# 并把 index 266 个键指向它、description 266 项标 FLOAT
```

- 服务目录口径：把 `-ref` 复制为 `v41-w4a8-dspark-vision`，只把 `config.json` 真文件化并恢复
  `text_config.max_position_embeddings=1048576`；该目录 `quant_repro_check.py` 严格模式
  **23 PASS / 0 WARN / 0 FAIL**（§5.5），`vision-00001-of-00001.safetensors` md5
  `999935d7debdb116b2cbf482ec32b6f8` 与 `-ref` 逐字节一致。
- 也可直接对**新建的** `$D` 原地执行（脚本幂等：重写分片 + 更新 JSON 键）；
  对已有服务目录做原地操作前请自行备份。
- 产物：`v41-w4a8-dspark-vision`（本仓库参考产物名 `...-vision-ref`）。
- 本交付的参考产物 `v41-w4a8-dspark-vision-ref` 在**无 torch 的宿主机**上用等价的
  safetensors 底层 API（mmap 源分片 + `TensorSpec/serialize_file`）生成，并用
  `safetensors.deserialize` 逐张量验证 **266/266 字节与官方完全一致**；正式外部用户
  在容器内直接用上面的 `add_vision_weights.py` 即可（内容等价）。

### 7.3 路径 ②（量化时就保留 vision）：现状与代价

```bash
# 装配时保留：
$V/bin/python $P/scripts/make_full_model.py --src ... --dst ... \
  --layers 40 --mtp 0 --engram off --vision keep --calib-max-seq-len 8192
```

- **代价**：装配目录多 266 个软链/0.97 GB 读取；量化 `init_model` 只按模型参数取权重，
  vision 键既不进 `state_dict` 也不进 saver，实际量化耗时/显存几乎不变（当前代码下）；
  但产物仍然没有 vision 权重。
- **结论**：当前实现下路径 ② 不能单独达到“产物含 vision”。要用路径 ②，必须同时改：
  ①`DeepSeekV41ModelAdapter`（把 vision tower 作为模块纳入 `init_model`/`get_state_dict`），
  ②`AscendV1Saver.post_run`（透传非模型键，或在 adapter 的 `ascendv1_save_postprocess`
  里把官方 vision 张量拷回），③保证 `linear_quant/quarot` 的 include/exclude 把
  `vision*/aligner*` 排除。改动面大于路径 ①，本次未实施。
- 若只是想在量化中途保留视觉权重文件（供后续手工拷贝），用 `--vision keep` 可以让
  `make_full_model` 不删键，但 saver 依旧不写；推荐直接用路径 ①。

### 7.4 视觉验收：`scripts/vision_accuracy_check.py`

**服务端**：必须开图（`VISION=1` → `--limit-mm-per-prompt '{"image": 1}'`，
`serve_v2.sh` 默认 `VISION=0` 会把图片请求当非法）；建议同时 `SPEC=1` 记录 DSpark 共存。

```bash
MODEL=$H/models/out/v41-w4a8-dspark-vision TP=8 PORT=8000 \
  VISION=1 SPEC=1 bash $P/scripts/launch_serve_half.sh    # tmux: 日志 logs/perf/serve_half*.log
# 等 /v1/models 就绪后：
python3 $P/scripts/vision_accuracy_check.py \
  --server http://127.0.0.1:8000 \
  --images-dir $H/models/DeepSeek-V4.1-Flash/inference/examples/images \
  --official-dir $H/models/DeepSeek-V4.1-Flash \
  --out $P/logs/perf/vision_accuracy.json --expect-dspark
```

> ⚠️ 该命令必须带 §6.2.5 的两处修正（`--tokenizer-mode=deepseek_v4` +
> `--default-chat-template-kwargs={"enable_thinking":false}`），否则 chat 400 / 短答走
> thinking 导致验收无效。P40 真机用 8001 端口、TP8、SPEC=7、`ENGRAM=0`。

用例与判据（内嵌 suite，23 例）：

- **20 组官方图问答**：`carrots.jpeg` / `corn.jpeg` 各 10 题（身份中/英、主色、背景色、
  生长位置、食用部位、形状、场景类型、表面质感、是否居中），短答案 `contains/any_of`
  程序化判定；身份事实来自官方 `inference/examples/example_harmony.json`（第一张胡萝卜、
  第二张玉米），其余属性来自官方图片的颜色/几何测量。
- **负向 1（必过）** `neg-text-01`：纯文本“中国首都”必须答“北京”。
- **负向 2（必过）** `neg-swap-01`：拿玉米图问“是胡萝卜吗？”必须答“否/不是”。
- **负向 3（额外）** `neg-blank-01`：程序生成纯灰 PNG，不得硬说成胡萝卜/玉米。
- **判据**：短答案命中率 **≥80%**、空/乱码 **=0**、请求错误 **=0**、两条必过负向全过；
  否则退出码 1。`--cases your.json` 可换自建 ≥20 组图问答（schema 见脚本头）。
- **记录**：每题 TTFT（流式首 chunk）、时延、usage；`/metrics` 前后内存/KV 增量；
  每张图的 image token 数 = `prompt_tokens(带图) - prompt_tokens(纯文本探针)`
  （对照 `vision_config.max_image_tokens=1024`）；DSpark `vllm:spec_decode_*` 计数与
  接受率；静态视觉权重 970,536,960 B。
- 离线可用 `python3 scripts/vision_accuracy_check.py --self-test` 验证打分/空乱码/错误/
  blank-PNG 逻辑（本交付已过）；`--dry-run` 只打印计划。
- ✅ **NPU 端到端已跑（P40，2026-09-13，TP8/EP8，SPEC=7；模型 `v41-w4a8-dspark-vision`）**：
  23 例 **10 过 = 43.5%**（判据 ≥80%），**必过负向 `neg-swap-01` 失败** ⇒ verdict **FAIL**；
  空/乱码 = 0、请求错误 = 0，`neg-text-01`/`neg-blank-01` 通过。结果
  `logs/perf/p40_vision_accuracy_final.json`；SPEC=0 与 SPEC=7 同错（9–10/23），去 vLLM pad
  也不改善。**⇒ 视觉验收未过，本指南视觉部分为 WIP，不得写成“已支持/已达标”。**
- **最强假设（未最终判定）**：V4→V4.1 移植丢失 image-span 双向/文档注意力——V4.1 后端
  `attention/dsa_v41.py` 对 `vision/mm_prefix/doc_range` 的引用为 **0**，而 V4 后端
  `attention/dsa_v1.py:422-520` 有 `build_vision_bidirectional_swa_indices`（并在
  :985-996/:2085-2086 生效）；权重（266/266 逐位）、预处理、注入数量、SPEC/pad 单因素
  已排除。判定实验 E1–E5 / 修复候选 F1–F3 见 `docs/VISION_RCA_P43.md`；离线执行清单
  （F1 草案 `patches/vision_v41_mm_bidi.patch`，默认关、**未上卡验证**）见 `docs/VISION_FIX_PLAN.md`。注意官方参考
  `inference/model.py` 是纯 causal，所以“V4.1 必须双向”仍需卡上 A/B 才能定论。
- 判据不变（命中率 ≥80%、空/乱码 =0、请求错误 =0、两条必过负向全过），当前未达到。

### 7.5 HBM / 上下文预算影响

| 项 | 数值 | 说明 |
|---|---|---|
| 视觉权重（BF16，不参与 W4A8） | **970,536,960 B ≈ 0.97 GB / rank** | 266 张量；P40 实测 TP8 下 tower **按 rank 全量复制**（无 TP/DP 切分），8 rank 名义 ~7.76 GB；FLOAT 不进量化 KV/权重压缩 |
| 单图 token | 实测 carrots **446** / corn **191**（带图 `prompt_tokens` − 纯文本探针） | 官方 span 444/189 + vLLM 换行 1 + compressor pad 1；`vision_config.max_image_tokens=1024`，`min_pixels=295936` |
| 1M 上下文预算 | 每图 ≤1024 token + 图片 KV | 官方文档口径整模 KV ≈890 B/token（`docs/DeepSeek-V4.1-Flash-tutorial.md` §1），实际 KV 与 CSA2/hybrid 配置有关 |
| DSpark 共存 | 记录 `spec_decode` 指标 | 当前 DSpark 接受率异常（§6.3），视觉+投机未验证 |

---

## 8. 与官方 Aurora 配方的差异（有据）

对照物：官方 W8A8 Aurora 导出元数据
`models/DeepSeek-V4.1-Flash-w8a8-metadata/{Aurora_best_practice.yaml,
quant_model_weights.safetensors.index.json,quant_model_description.json,config.json}`。

### 8.1 总表

| 维度 | 官方 Aurora（W8A8 元数据证据） | 本仓库 W4A8 | 性质 |
|---|---|---|---|
| 主干路由专家 | W8A8_DYNAMIC（142,494 张量） | **W4A8_DYNAMIC**（int4 SSZ，184,320 张量） | 刻意（我们要 4bit 权重） |
| 注意力 `wq_a/wq_b/wkv`、共享专家 | W8A8_DYNAMIC | W8A8_DYNAMIC | 一致 |
| `indexer.wq_b` | W8A8（include 列表显式列出） | W8A8_DYNAMIC | 一致 |
| `indexer.wk` | **FLOAT**（include 无此键） | **FLOAT**（YAML 显式 `exclude: *indexer.wk`） | 一致；不是差异。注意按 `*attn*` 模式类配方（V4-Pro/V4-Flash DSpark）它本会被卷进 W8A8，我们刻意不量化 |
| `indexer.wq_a` / `weights_proj` | FLOAT | FLOAT | 一致 |
| `wo_a`/`wo_b`、`compressor.wgate/wkv` | FLOAT | FLOAT | 一致 |
| trunk FLOAT 清单 | 670 张量 / 25 pattern（`attn_sink`、q/kv_norm、norm、gate、hc_*、…） | **完全相同 670 张量 / 25 pattern** | 一致（校验器逐 pattern 对账） |
| `mtp.*` | **量化**：3,566 张量中 3,510 W8A8 + 56 FLOAT（attn wq_a/wq_b/wkv、experts、shared_experts 量化；main_proj/hc/gate/norm FLOAT） | 3 个 MTP 层**全部 FLOAT/BF16**（官方权重反量化后回填，1,226 张量 + 1,177 desc-only scale） | **刻意偏差**（run-time 兼容 + 先保精度；对齐实验见 §10） |
| `vision.*/aligner.*/image_*` | 266 张量 **FLOAT 且随导出带出** | stage1/dspark **缺失**；参考产物（§7.2）补回 266 FLOAT | **待改进→已给补回路径**；量化策略本身一致（都 FLOAT） |
| Engram | layers 1/14 int8（`embedding_storage` group_size 32），导出仍 10 个 FLOAT 辅助键 | 阶段一 `engram_layer_ids=[]`（完全关）；另有独立 `models/out/engram-int8`（221.2 GB）不并入产物 | 刻意（先文本量化；Engram 单独转换，见 `docs/STATUS.md`） |
| `mtp.0.main_proj` 旋转 | `quarot.right_global` 含 `mtp.0.main_proj.weight` | `make_dspark_ckpt.py` 预旋转 `W @ blockdiag(R)` | 一致 |
| 导出分片 | 274 分片（`ordinary_target_shard_bytes: 2 GiB`） | 80 分片（`part_file_size: 4`） | 格式差异，运行时无关 |
| `model_quant_type` | `W8A8_DYNAMIC`（混合时 W4A8_DYNAMIC 被 saver 优先级隐藏） | `W8A8_DYNAMIC` | 一致 |

### 8.2 为什么 MTP 保持 FLOAT

- `DeepSeekV41ModelAdapter.generate_decoder_layer()` 对 MTP 层直接
  `raise NotImplementedError("V4.1 DSpark MTP layer-wise load is not implemented in P1")`；
  即当前 msmodelslim 适配器只能量化 40 个主干层。
- DSpark 的 draft 权重在运行时需要与主干不同的 expert 数（`dspark_n_routed_experts`）
  和 `main_proj` 旋转；官方 Aurora 的 `modelslim_convert`（`profile: aurora`）走的是另一条
  闭源/转换路径，不是本仓库的 `modelslim_v1` 层间量化管线。
- 结果：我们走“主干 W4A8 + 官方 MTP 反量化 BF16 + 预旋转 + draft 词表”的混合 checkpoint，
  MTP 精度更高（BF16），但显存/带宽不是最优；这是已知取舍。

### 8.3 “其余 FLOAT 清单”结论

官方 Aurora 的 trunk FLOAT 就是 670 个（25 个 pattern），与本仓库**逐项一致**（校验器
`pattern_counts` 检查覆盖）。所谓“其余 FLOAT 清单差异”在当前 W4A8 产物上不存在；
差异只发生在 MTP/vision/Engram 三类非主干权重上。

### 8.4 `hiaux` 实验分支的地位

- 配置：`lab_practice/deepseek_v41/deepseek_v4_1_flash_w4a8_hiaux.yaml`（可选补丁
  `patches/msmodelslim_v41_w4a8_hiaux_optional.patch`）。
- 改动：把 `layers.37/38/39` 的路由专家从 W4A8 提升为 W8A8（DSpark draft 的输入取自
  这三层），其余与主配方一致；目标是提高 draft 接受长度。
- 状态：**实验分支，不进主补丁、不进本文复现主线**。质量对照见 `docs/ACCURACY.md` §4
  （hiaux 2/56 退化 vs 对照 4/56，Fisher p≈0.68，无显著差异；但 DSpark 接受率问题
  仍未解决，`docs/PERF_LEDGER.md` 记录了 hiaux 因复读/接受长度虚高被否决）。

---

## 9. 排障

| 症状 | 根因 / 处理 |
|---|---|
| HCCL `EI0005` / `EI0006`，各 rank 报 shape 不一致 | DP 下各 rank token 数不同；主补丁已给 `V41MoE.forward` 加 gather/scatter。若在别的适配器出现，检查所有 `all_reduce` 输入是否 rank 相关 |
| 服务启动 MoE tiling 失败 | `HCCL_BUFFSIZE=1024`（`serve_v2.sh` 默认已设） |
| 加载时 engram 权重缺失/错位 | 运行时 `enable_engram` 默认 **True**；本产物 `engram_layer_ids=[]`，必须 `--additional-config '{"enable_engram": false}'`（`serve_v2.sh` 在 `ENGRAM=0` 时已加） |
| tokenizer 加载失败 / `find -xtype l` 一堆断链 | 装配软链用了相对路径；`make_full_model.py` 已强制绝对路径，手工建链也要绝对路径 |
| CLI 报 `deepseek_v41` 适配器不存在 | apply 后没刷新包内配置：`cp config/config.ini msmodelslim/config/config.ini`（apply 脚本默认做）；`msmodelslim/config_repo` 软链也要在 |
| 宿主机读产物 `Permission denied` | 容器内 umask 077，产物 root 0600；用 `sudo -n` 或 `umask 022` 重建 |
| 图片请求被拒/被忽略 | `VISION=0` 默认 `--limit-mm-per-prompt '{"image": 0}'`；服务加 `VISION=1` |
| `/v1/chat/completions` 400 `must provide a chat template` | 镜像 registry 没有 `deepseek_v41`：加 `--tokenizer-mode=deepseek_v4`（不是 `deepseek_v41`，§6.2.5） |
| chat 短答（≤32 token）被思考占满/判分异常 | DSV4.1 chat 默认 thinking：加 `--default-chat-template-kwargs={"enable_thinking":false}`（§6.2.5） |
| 长上下文答案坏但不报错、KV 像别的层 | 误打 `prefill_incr`/`staging_delta`：4 个 long-KV source 共享 staging，第 2 步起静默读错源；正确链不要 apply，见 §6.2.2 与 `docs/INT8_PREFILL_DELTA_SAFETY.md` |
| 视觉答案错但请求 200、文本正常 | 已知视觉 WIP（43.5%/23 例，必过负向挂）；不是权重缺失；见 §7.4 / `docs/VISION_RCA_P43.md` |
| 长上下文崩/答非所问、yarn 配置告警 | 阶段一 `max_position_embeddings` 被压到 8192，而 yarn `original_max_position_embeddings=65536`；运行前把 runtime `config.json` 的 text_config 恢复 `1048576`（`docs/STATUS.md` 记录） |
| C-Eval/GSM8K 很低、大量空响应 | 先查 chat 模板是否缺 `</think>`；官方 encoder 要求 chat 模式末尾以 `<|Assistant|>` + `<|/think|>` 收尾（`docs/ACCURACY.md` §3 修过该 bug） |
| DSpark 接受率≈1.0 | 已知未解问题（`docs/STATUS.md` 2026-09-12）：已排除 MTP 反量化误差/main_proj 旋转方向/双重旋转/权重名映射；draft 词表补齐是必要的但不充分。不要用它宣称投机可用 |
| `make_dspark_ckpt.py` 报找不到 rotation | 默认读 `<quant-dir>/optional/quarot.safetensors`；没有就 `--rotation-file` 指定，或仅对照时 `--no-rotate-main-proj` |
| 量化重跑把旧产物清了 | `quant_dp_inner.sh` 里有 `rm -rf $SAVE`；换新 `SAVE` 或先备份 |
| 校准数据找不到 | `lab_calib/mix_calib.jsonl` 必须通过 softlink/复制对 CLI 可见；日志应打印 `prepare dataset from mix_calib.jsonl success` |

---

## 10. （可选，未执行）把 MTP 也量化以对齐官方 Aurora：实验设计

> 目标：把 3 个 `mtp.*` draft 层按官方 Aurora 口径量化成 W8A8_DYNAMIC（attn
> `wq_a/wq_b/wkv`、路由专家、共享专家），`main_proj`/`hc_*`/`gate`/`norm`/`attn_sink` 等
> 保持 FLOAT，从而把 DSpark 目录从 ~30 GB MTP BF16 降到接近主干同款量化格式。
> **本节只是设计，未执行、未验证。**

### 10.1 影响面

| 层 | 改动 | 风险 |
|---|---|---|
| msmodelslim `deepseek_v41/model.py` | 新增 V41 MTP block 模块（attention + MoE + norm + main_proj），支持 `dspark_n_routed_experts` | MTP block 与主干结构差异（expert 数、`confidence_head`/`markov_head`/词表头、main_proj 输入维 3×hidden） |
| `model_adapter.py` | `generate_decoder_layer()` 的 MTP 分支实现：`mtp.{i}` 前缀逐层 load、FP8/FP4 反量化、build 参数 | `init_model` 的 1 层模板/strict load 语义；MTP 与主干共享 `shared_attn` 单例 |
| `rotation_map.py` | MTP 的 QuaRot 映射（当前只旋 `mtp.0.main_proj`；官方 right_global 也只含它） | 若旋错 MTP 输入/输出基，接受率会崩 |
| YAML | 新增 `mtp.*.attn.wq_a/wq_b/wkv`、`mtp.*.ffn.experts.*`、`mtp.*.ffn.shared_experts.*` 的 `linear_quant` include（官方 W8A8 口径），保留 FLOAT 清单 | 量化处理器对 MTP 子图（up-down/norm-linear）可能缺失对应映射 |
| 运行时 | vLLM-Ascend 的 quant 映射/描述消费需支持 `mtp.*` 量化键 | ⚠️ 本仓库运行时核对记录（`docs/STATUS.md`）称容器内 `modelslim_config.py` 有“deepseek 的 MTP 层不量化，描述文件需手动补 FLOAT”的约定；**量化 MTP 很可能需要 vllm-ascend 侧改动**，这是最大不确定性 |

### 10.2 推荐步骤（若要做）

1. **离线先验证格式**：用官方 Aurora W8A8 元数据的 mtp 标签分布做目标
   （3,510 W8A8 + 56 FLOAT / 3,566），先用 1 个 MTP 层、`--limit` 类小规模跑通；
   产物用 `quant_repro_check.py --mtp quantized` 校验“非 MTP pattern 不变 + MTP 有
   W8A8 键”。
2. **适配器实现**：在 `DeepSeekV41ModelAdapter` 里实现 MTP 层的建图/加载；`get_moe_config`
   已支持 `dspark_n_routed_experts`（`deepseek_v4/model.py` 扩展已含）。
3. **量化配置**：对齐官方 include/exclude（`mtp.*.attn.wq_a/wq_b/wkv`、
   `mtp.*.ffn.experts.*`、`mtp.*.ffn.shared_experts.*`；排除 `main_proj`、`hc_*`、`gate`、
   `norm`、`confidence_head`、`markov_head` 等）。
4. **组装**：量化 MTP 后不再需要 `make_dspark_ckpt.py` 的反量化；但 `main_proj` 旋转
   与 draft 词表仍需按 §6.3 处理（若量化产物已含旋转后的 main_proj，注意不要重复旋转）。
5. **运行时验证**：确认 vllm-ascend 能消费量化 MTP 的 description/index；必要时同步
   修改 `modelslim_config.py` 映射并加回归。

### 10.3 验证方法（对齐目标）

- **格式**：`quant_repro_check.py --dir <new> --mtp quantized`；MTP 标签集合
  `{W8A8_DYNAMIC, FLOAT}`，非 MTP 187k 张量逐 pattern 不变；总字节下降量记录。
- **数值**：draft logits/MTP 权重与官方 FP8/FP4 反量化结果对比（相对误差），
  以及 `main_proj` 旋转前后 `max|delta|`（make_dspark 现有日志可复用）。
- **端到端**：DSpark 接受长度/接受率（目标 3.33–3.45；`bench_vllm.py` +
  `/metrics` 的 `vllm:spec_decode_*`），单流 tok/s；GSM8K 50–200 题不回归
  （基线 97.5%）；`vision_accuracy_check.py --expect-dspark` 记录视觉+投机共存。
- **回滚**：保留现有 “MTP BF16” 目录作为对照；量化 MTP 通过
  `--mtp quantized` 校验但端到端不达标则回退。

### 10.4 预估收益/代价（仅估算）

- 收益：MTP 权重由 BF16 变 W8A8，约省一半 MTP 权重显存/带宽（本仓库 MTP 分片
  28.46 GB → 理论 ~14–15 GB，按实际 scale 开销略有出入）。
- 代价：适配器/saver/运行时映射改动 + 接受率回归风险；**如果运行时映射不支持，
  收益为 0 且无法加载**。

---

## 11. 验证矩阵（本次交付）

| 项 | 状态 | 证据 |
|---|---|---|
| 补丁 apply 到 `92e219f` 并复现 e05b471 文件内容 | ✅ | `git apply --check/apply` + 逐文件 diff（`logs/perf/msmodelslim_repro_impl.md`） |
| 主补丁 md5 / 行数 | ✅ | `ff375f28e6cf5d7492e71b353d34ba5e` / 1687 行 |
| stage1 指纹（185,734 / 184,320 / 744 / 670） | ✅ | `quant_repro_check.py` 对 `v41-w4a8-stage1`（waiver）18 PASS / 2 WARN / 0 FAIL |
| DSpark 指纹（186,960；MTP 全 FLOAT） | ✅ | 同上，对 `v41-w4a8-dspark`（waiver） |
| vision 补回（266 张量 / 970,536,960 B / BF16） | ✅（结构+字节） | `-ref` + `quant_repro_check.py` 22 PASS / 1 WARN（仅 max_pos=8192）；服务目录见下行 23 PASS；逐张量 `safetensors.deserialize` 字节比对 |
| 严格产物校验（vision 必需） | ✅ | 服务目录 `v41-w4a8-dspark-vision`：187,226 张量 / 86 分片 / vision 266/266（0.9705 GB），**23 PASS / 0 FAIL**；`logs/perf/p40_quant_manifest_service.json`（`-ref` 为 22 PASS / 1 WARN，仅 `max_position_embeddings=8192`） |
| 视觉端到端精度（图文问答） | ❌ **FAIL（WIP）** | P40：23 例 10 过 = **43.5%** < 80%，必过 `neg-swap-01` 失败；`logs/perf/p40_vision_accuracy_final.json`；最强假设与实验见 `docs/VISION_RCA_P43.md` |
| 文本 GSM8K 200 | ✅ 参考 | vision 配置（canonical SPEC=7）**96.0%（192/200）**，`logs/perf/p40_gsm8k200_vision.json`；无 vision 基线 **97.5%（195/200）**，`docs/ACCURACY.md`；命令见 §6.2.6 |
| 正确服务链 + 三档 BAT 启动账 | ✅ | 8+1 补丁链（§6.2.1）；BAT 1024/2048/4096 ⇒ num_blocks 42,156/39,896/35,449、rank_cap 4/4/3（§6.2.4） |
| UNSAFE 补丁判定（incr/delta） | ✅ NO-GO | `docs/INT8_PREFILL_DELTA_SAFETY.md` + `patches/UNSAFE_prefill_incr_delta.md`；正确链不打（§6.2.2） |
| 15×1Mi 端到端驻留 + hold | ⏳ 未跑完 | 容量账成立（BAT=2048 余 4,019 blocks）；本轮 P42 legacy_b2048 只跑单条 1Mi timing（§6.2.4） |
| MTP 量化对齐官方 | ❌ 未执行 | 设计见 §10；当前 MTP=FLOAT |
| DSpark 接受率 | ⚠️ 已知未解 | 接受长度≈1.01，见 `docs/STATUS.md`/`docs/PERF_LEDGER.md` |
| Engram int8 并入服务产物 | ❌ 未做 | 独立产物 `models/out/engram-int8`；本指南主产物 `engram_layer_ids=[]` |

---

## 附录 A：命令/路径速查

| 目的 | 命令 |
|---|---|
| 打补丁（干跑/正式/实验） | `bash scripts/apply_msmodelslim_patch.sh --dir $R [--check] [--with-hiaux]` |
| 装配可量化目录 | `$V/bin/python scripts/make_full_model.py --src <official> --dst <stage1> --layers 40 --mtp 0 --engram off --vision drop` |
| 全量 DP16 量化 | `MODEL=<stage1> SAVE=<out> DEVICE_IDS='0 ... 15' bash scripts/quant_dp_inner.sh`（tmux: `scripts/launch_full_tmux.sh`） |
| 补 quantization_config | `$V/bin/python scripts/prepare_runtime_ckpt.py --dir <out>` |
| 组装 DSpark | `$V/bin/python scripts/make_dspark_ckpt.py --quant-dir <out> --official-dir <official> --out-dir <dspark>` + `scripts/add_draft_vocab.py` |
| 补 vision | `$V/bin/python scripts/add_vision_weights.py --official-dir <official> --target-dir <dspark-vision>` |
| 产物校验 | `sudo -n python3 scripts/quant_repro_check.py --dir <dir> [--official-dir <official>] [--allow-missing-vision] [--out <json>]` |
| 视觉验收 | `python3 scripts/vision_accuracy_check.py --server <url> --images-dir <images> [--official-dir <official>] [--expect-dspark] --out <json>` |
| GSM8K 小样本 | `~/venvs/lmeval311/bin/python scripts/acc_eval.py --task gsm8k --limit 50 --mode chat --conc 4 --base-url <url> --out <json> --tag <tag>` |
| 正确服务补丁链 | `cd /vllm-workspace/vllm-ascend && for p in <8 个>; do git apply --check "$P/patches/$p" && git apply "$P/patches/$p"; done`（§6.2.1；任一步非 0 立即停） |
| 档2 起服（TP2/DP4 int8） | apply 后 `source` §6.2.3 env + `bash scripts/serve_v2.sh`（`EXTRA='--no-async-scheduling'`）；A3 包装见 §6.2.3（必须显式 `LAUNCHER=.../p36_capturefix2_start.sh`） |
| 6 个启动数核对 | `grep -nE -e "Available KV cache memory" -e "num_blocks=" -e "request_blocks" -e "rank_cap=" -e "BF16 staging" serve.log`（期望见 §6.2.4） |
| 视觉评测（WIP） | `python3 scripts/vision_accuracy_check.py --server <url> --images-dir <images> --official-dir <official> --expect-dspark --out <json>`；**当前 FAIL 43.5%**（§7.4） |

## 附录 B：产物指纹一句话版

- 文本主干：**185,734 张量 / W4A8_DYNAMIC 184,320 / W8A8_DYNAMIC 744 / FLOAT 670**；
  80 分片 273 GB；旋转附件 `global_rotation` F32 [5120,5120]。
- DSpark：**186,960** 张量（+1,226 MTP，全 FLOAT）；`mtp.0.main_proj` 已右乘
  `blockdiag(R,R,R)`；draft 词表 2 张量。
- 完整多模态服务产物：**187,226** 张量（+266 vision FLOAT / 970,536,960 B / BF16）；
  `quant_repro_check.py` 严格模式 **23 PASS / 0 FAIL**，manifest
  `logs/perf/p40_quant_manifest_service.json`（冻结参考 `-ref` 为 22 PASS / 1 WARN，仅
  `max_position_embeddings=8192`；另有基准 ref manifest `logs/perf/quant_manifest_ref.json`）。

## 附录 C：相关文档

- `patches/PATCHES.md`：补丁清单/md5/apply 规则
- `logs/perf/msmodelslim_repro_impl.md`：交付记录与未验证项
- `docs/STATUS.md`：量化/服务/DSpark 的历次快照与踩坑
- `docs/ACCURACY.md`：GSM8K/C-Eval 精度报告（含 chat 模板坑）
- `docs/DeepSeek-V4.1-Flash-tutorial.md`：官方支持特性/镜像/部署要求
- `models/DeepSeek-V4.1-Flash-w8a8-metadata/Aurora_best_practice.yaml`：官方 Aurora W8A8
  配方（差异对照的权威来源）
- `docs/PATCH_APPLY_ORDER.md`：canonical/merged/gate 的权威顺序与互斥
- `docs/INT8_PREFILL_PERF.md`：int8 prefill 成本模型、BAT/容量账、六启动数
- `docs/INT8_PREFILL_DELTA_SAFETY.md` + `patches/UNSAFE_prefill_incr_delta.md`：UNSAFE 判定证据
- `docs/VISION_RCA_P43.md`：视觉 43.5% 根因分析（WIP）
- `docs/VISION_FIX_PLAN.md` + `patches/vision_v41_mm_bidi.patch`：视觉修复离线执行清单 / F1 草案（未上卡）
- `a2_package/RUNBOOK.md`、`a2_package/README.md`、`a2_package/configs/a2_tp2dp4_int8_15x1m.env`：档2 env 与验收流程
