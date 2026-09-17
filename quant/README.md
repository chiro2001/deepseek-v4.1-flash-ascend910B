# quant/ —— 完整量化脚本链（5 级装配）

> 目标：从**官方 DeepSeek-V4.1-Flash** checkpoint 出发，产出 A2 可直接起服的
> `v41-w4a8-engram-dr-vision-qrot-mtpq` 目录。
> 全部脚本在容器内跑（需要 npu、torch_npu、msmodelslim）；**本目录只是脚本与配方，不含权重**。
>
> 详细逐步说明见 `REPRO_W4A8_QUANT.md`（69 KB，A3-node1 上跑的原始成果物），
> 本文件是**面向 A2 的收敛版**：只讲 5 级装配 + 每条命令 + 验收判据。

---

## 0. 三条红线（违反任一条 → 白跑几小时）

1. **设备号只能写容器内的 `0..7`**。
   `quant_dp_inner.sh` 的 `DEVICE_IDS` 默认是 `0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15`
   （A3-node1/A3 的 16 逻辑卡）。**A2 是 8 卡单机，必须显式给 `DEVICE_IDS="0 1 2 3 4 5 6 7"`**，
   写 8–15 会直接 **`ExchangeDevice` 报错**（容器内不存在这些 device）。
2. **`quant_dp_inner.sh` 会 `rm -rf "$SAVE"`**（第一条命令就是）。给 `SAVE=` 时务必确认目录可删。
3. **不要用 sha256 验收主干权重**（见 §3.2）。

---

## 1. 5 级装配总览

| 级 | 产物 | 做什么 | 主脚本 | 预计耗时（A3） |
|---|---|---|---|---|
| **L1** | `models/out/v41-w4a8-stage1` | 主干 **W4A8_DYNAMIC** 量化（40 层，MTP off，Engram off，Vision drop） | `make_full_model.py` → `apply_msmodelslim_patch.sh` → `quant_dp_inner.sh` → `check_quant_output.py` → `prepare_runtime_ckpt.py` | 量化 **35–76 min**（8 卡 DP 约 76 min / 16 卡 DP 约 36 min） |
| **L2** | `engram-int8`（后续并入模型目录） | Engram 表 **FP8 → INT8**（group32，二次幂 fp32 scale）+ DR 构建 | `run_engram_int8.sh`（`engram_convert_int8.py`）、`engram_dr_build.py` | 9–20 min |
| **L3** | `+vision` | 视觉 266 张量（`vision.*/aligner.*/image_*`，0.9705 GB BF16）并入 + **qrot 旋转折叠修复** | `add_vision_weights.py`（+ QuaRot 折叠，见 §3.3） | < 5 min |
| **L4** | `+dspark` | DSpark/MTP checkpoint：官方 mtp **反量化成 BF16**（1224 张量 / 26.5 GiB），`mtp.0.main_proj.weight` 按 QuaRot 预旋转；再 **mtpq 量化 draft**；补 draft 词表 | `run_dspark_ckpt.sh`（`make_dspark_ckpt.py`）→ `mtpq_quantize_draft.py` → `add_draft_vocab.py` | 10–20 min |
| **L5** | `v41-w4a8-engram-dr-vision-qrot-mtpq` | **串联修复**：`prepare_runtime_ckpt.py` 写 `quantization_config`、校验 index/tag 分布、软链解析；最终结构验收 | `prepare_runtime_ckpt.py` + `verify_structure.py`（本包） | < 5 min |

> `prepare_runtime_ckpt.py` 出现在 L1 与 L5 两次是**故意的**：L1 时它保证量化产物能被
> vLLM 加载（写 `quantization_config`）；L5 时它对**最终装配目录**再跑一次，把后加的
> vision/dspark/engram 之后的 config 与 index 修好。它是幂等的，重复跑没有副作用。

---

## 2. 逐级命令（容器内执行，`$P`=本包解压目录，`$H`=你的 home）

### 前置：装 msmodelslim（离线 editable）

```bash
bash $P/quant/scripts/setup_mslim_inner.sh          # 建 venv 并让 venv 看到镜像里的 torch/torch_npu
bash $P/quant/scripts/apply_msmodelslim_patch.sh --dir <msmodelslim 源码树> --check   # 干跑
bash $P/quant/scripts/apply_msmodelslim_patch.sh --dir <msmodelslim 源码树>           # 正式打补丁
```

补丁本体：`quant/patches/msmodelslim_v41_w4a8.patch`，**md5 `ff375f28e6cf5d7492e71b353d34ba5e`**，
基线 commit **`92e219fa9565a5bad84d90474a27bb11524d691c`**（`[Bugfix] analysis linear dp bugfix`）。
⚠️ 不能打在 `e05b471`（那是打完补丁后的 HEAD，`git apply` 必失败）。

### L1 主干 W4A8

```bash
# 1) 装配"可量化目录"（零拷贝，分片是绝对路径软链）
python3 $P/quant/scripts/make_full_model.py \
  --src $H/models/DeepSeek-V4.1-Flash \
  --dst $H/models/DeepSeek-V4.1-Flash-stage1 \
  --layers 40 --mtp 0 --engram off --vision drop --calib-max-seq-len 8192

# 2) 8 卡 DP 量化（A2：DEVICE_IDS 必须是 0..7！）
MODEL=$H/models/DeepSeek-V4.1-Flash-stage1 \
SAVE=$H/models/out/v41-w4a8-stage1 \
DEVICE_IDS="0 1 2 3 4 5 6 7" \
CFG=<msmodelslim>/lab_practice/deepseek_v41/deepseek_v4_1_flash_w4a8.yaml \
bash $P/quant/scripts/quant_dp_inner.sh

# 3) 结构校验 + 写 quantization_config
python3 $P/quant/scripts/check_quant_output.py --dir $H/models/out/v41-w4a8-stage1
python3 $P/quant/scripts/prepare_runtime_ckpt.py --dir $H/models/out/v41-w4a8-stage1
```

**期望结构（L1）**：273 GB / 72–80 分片 / index **185,734** 张量；
标签分布 `W4A8_DYNAMIC 184320` / `W8A8_DYNAMIC 744–745` / `FLOAT 670`；
`config.json.quantization_config = {"quant_method":"ascend","model_quant_type":"W4A8_DYNAMIC"}`。

### L2 Engram INT8

```bash
SRC=$H/models/DeepSeek-V4.1-Flash DST=$H/models/out/engram-int8 bash $P/quant/scripts/run_engram_int8.sh
python3 $P/quant/scripts/engram_dr_build.py --help      # DR 构建（按需，参数见 --help）
```

验收：`engram_int8` 下每个 `layers.{1,14}.engram.embed` 的 **int8 权重 + fp32(group32) scale**；
服务侧常驻 DRAM ≈ **206 GiB**（与平台无关，是强判据）。

### L3 Vision（含 qrot 修复）

```bash
python3 $P/quant/scripts/add_vision_weights.py --help
```

**qrot 修复的必要性**：QuaRot 的全局旋转 `R` 会破坏视觉边界。修复方式是把 `R` 折叠进
视觉边界的 **5 个张量**（`aligner.w2.weight/bias`、`image_start`、`image_end`、`image_newline`），
**只改权重、不改运行时**。不做这一步，视觉只有 **~10/23**；做了是 **23/23**。

### L4 DSpark / mtpq

```bash
QF=$H/models/out/v41-w4a8-stage1 \
OF=$H/models/DeepSeek-V4.1-Flash \
OUT=$H/models/out/v41-w4a8-dspark \
bash $P/quant/scripts/run_dspark_ckpt.sh          # 内有 dry-run → 正式 → verify 三步

python3 $P/quant/scripts/mtpq_quantize_draft.py --help     # draft（mtp）量化：mtpq
python3 $P/quant/scripts/add_draft_vocab.py --official-dir $OF --dspark-dir $OUT
```

期望：`make_dspark_ckpt.py` 产出 **1224 张量 / 4 分片 / 26.5 GiB**；
补词表后 index **186,960** 键；起服日志出现 `DSpark draft model loaded: 74 params`
且**没有** `Sharing target model embedding`。

### L5 串联修复 + 最终验收

```bash
MODEL=$H/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq
python3 $P/quant/scripts/prepare_runtime_ckpt.py --dir "$MODEL"          # 幂等，跑第二次
python3 $P/quant/scripts/verify_structure.py --dir "$MODEL"              # 结构 + 容差（不用 sha256）
python3 $P/quant/scripts/quant_repro_check.py --dir "$MODEL"             # A3-node1 的官方校验器（可选，更啰嗦）
```

---

## 3. 验收口径（**只用结构 + 容差，禁止 sha256**）

### 3.1 结构判据（必须全中）

| 项 | 期望 |
|---|---|
| 最终 index 张量数 | **187,226**（= 185,734 + mtp 1224 + vision 266 附近，允许 ±32） |
| 分片数 | 72–87（**DP 宽度不同必然不同**，不是判据） |
| 标签分布 | `W4A8_DYNAMIC 184320`、`W8A8_DYNAMIC 744±1`、`FLOAT 2162±40` |
| `quantization_config` | `{"quant_method":"ascend","model_quant_type":"W4A8_DYNAMIC"}` |
| `optional/quarot.safetensors` | 存在，`global_rotation` **F32 [5120,5120]** |
| Engram | `text_config.engram_layer_ids` 非空；int8 表 + group32 fp32 scale |
| Vision | 266 张量（`vision.*` 259 + `aligner.*` 4 + `image_*` 3），BF16，合计 **0.9705 GB** |
| DSpark | `mtp.*` 1224 张量；`mtp.0.main_proj.weight` 已预旋转 |

### 3.2 为什么**不能用 sha256**验收主干

我们（A3-node1/A3-node2）用 **DP16**（16 逻辑卡）量化，A2 用 **DP8**。两者的分片切法不同：

- A3 真值：**80 分片**；A2 DP8：**72 分片**；
- **47.5% 的张量有 int8-LSB 级差异**（累计和/归约顺序不同）；
- **总字节相同、数值等价**（权重差异在量化容差内）。

⇒ 「分片 sha256 不同」**是预期，不是失败**。判据只能是 §3.1 的结构 + §3.3 的容差
（`verify_structure.py` 已按此实现）。

### 3.3 容差判据

| 项 | 容差 |
|---|---|
| 同目录重跑 `prepare_runtime_ckpt.py` | index/字节数**完全不变**（幂等） |
| 与 A3-node1 参考 manifest 比 | 张量数差 ≤ 0.05%；标签数差 ≤ 0.5%；**不做逐字节比对** |
| 权重数值 | 抽样 8 个分片、每片 16 个 W4A8 张量：`max|Δ| ≤ 1 LSB`（int8）或 `rel ≤ 2e-3`（fp32 scale） |
| 服务侧 | 接受长度 A 与 A3-node1 相差 ≤ ±15%（见 `EXPECTED_PERF.md`） |

---

## 4. A2 相对 A3-node1 必须改的地方（checklist）

1. `DEVICE_IDS="0 1 2 3 4 5 6 7"`（**不要** 0..15）。
2. `quant_dp_inner.sh` / `run_*` 里的 `H=`、`P=` 路径改成 A2 自己的（本目录脚本默认仍是
   `/home/user/...`，因为那是**能跑通的原件**；我们不改脚本内容，改的是你调用时传的变量）。
3. `SAVE` 目录留 ≥ 300 GB；`src` 官方 checkpoint ≥ 476 GB。
4. 量化前确认 8 张卡空闲（`npu-smi info`，别人的容器会抢 HBM/算力）。
5. 量化是**长任务**（1 小时级），用 tmux/nohup 挂住，日志重定向到文件。

---

## 5. 文件清单

| 文件 | 作用 |
|---|---|
| `REPRO_W4A8_QUANT.md` | A3-node1 原始量化复现指南（69 KB，最详细，遇到细节问题查它） |
| `patches/msmodelslim_v41_w4a8.patch` | 主补丁（基线 `92e219f`，md5 `ff375f28…`） |
| `patches/MSMODELSLIM_PATCHES.md` | 补丁说明（顺序、apply 规则、坑） |
| `scripts/make_full_model.py` | L1 装配"可量化目录"（零拷贝软链） |
| `scripts/quant_dp_inner.sh` | **8/16 卡 DP 量化入口**（`DEVICE_IDS` 见 §0.1） |
| `scripts/run_quant_full.sh` | 单卡/小规模量化入口（调试用） |
| `scripts/check_quant_output.py` | L1 结构校验（标签分布、张量数） |
| `scripts/prepare_runtime_ckpt.py` | **串联修复**：写 `quantization_config`、校验 tag、打索引统计 |
| `scripts/run_engram_int8.sh` + `engram_convert_int8.py` | L2 Engram FP8→INT8 |
| `scripts/engram_dr_build.py` | L2 DR 构建 |
| `scripts/add_vision_weights.py` | L3 视觉并入（+ qrot 折叠修复） |
| `scripts/run_dspark_ckpt.sh` + `make_dspark_ckpt.py` | L4 DSpark 反量化组装 |
| `scripts/mtpq_quantize_draft.py` | L4 draft（mtp）量化 |
| `scripts/add_draft_vocab.py` | L4 补 draft 词表 |
| `scripts/quant_repro_check.py` | A3-node1 官方校验器（输出 JSON manifest + PASS/FAIL） |
| `scripts/verify_structure.py` | **本包新增**：结构 + 容差验收（明确拒绝 sha256 口径） |
| `scripts/setup_mslim_inner.sh` / `prepare_mslim_layout.sh` / `apply_msmodelslim_patch.sh` | msmodelslim 离线安装与补丁 |
