# msModelSlim DeepSeek-V4.1-Flash W4A8 补丁集（PATCHES.md）

> 交付对象：把本机 fork 的 DeepSeek-V4.1 W4A8 量化适配做成可 apply 到上游
> [Ascend/msmodelslim](https://gitcode.com/Ascend/msmodelslim.git) 的补丁串。
> 复现指南见 [`../docs/REPRO_W4A8_QUANT.md`](../docs/REPRO_W4A8_QUANT.md)。

## 1. 基线 / 目标 commit

| 项 | 值 |
|---|---|
| 上游基线 commit | `92e219fa9565a5bad84d90474a27bb11524d691c`（`origin/master`，`[Bugfix] analysis linear dp bugfix`） |
| 本机实现 commit | `e05b47190392bace361a012f115d64f4abcca30d`（分支 `dsv41-w4a8`，标题 `feat(dsv41): DeepSeek-V4.1 W4A8 量化适配`） |
| 补丁形式 | `git diff`（raw diff，可被 `git apply -p1` 直接消费；文件头已注明 base commit） |
| 适用目录 | msmodelslim 仓库根目录（包含 `msmodelslim/`、`lab_practice/`、`config/` 的那一层） |

主补丁就是 `92e219f..e05b471` 的完整 diff；工作区里唯一的必要未提交文件是实验
配置文件 `lab_practice/deepseek_v41/deepseek_v4_1_flash_w4a8_hiaux.yaml`，与主线无关，
单独做成**可选补丁**（见 §3）。生成的符号链接 `msmodelslim/config_repo` 不入补丁，
由 apply 脚本 / `prepare_mslim_layout.sh` 现场创建。

## 2. 主补丁

| 项 | 值 |
|---|---|
| 文件 | `patches/msmodelslim_v41_w4a8.patch` |
| md5 | `ff375f28e6cf5d7492e71b353d34ba5e` |
| 规模 | 1687 行 / 9 文件（+1545 / -13 的提交内容 + 文件头注释） |
| 内容 | `msmodelslim/model/deepseek_v41/{__init__,loader,model_adapter,model,rotation_map,convert}.py` 新建；`msmodelslim/model/deepseek_v4/model.py` 扩展；`lab_practice/deepseek_v41/deepseek_v4_1_flash_w4a8.yaml` 配方；`config/config.ini` 注册 `deepseek_v41` 适配器 |
| 关键修复 | `V41MoE.forward` 的 DP gather/scatter：16 rank 的 token 数不同会让 HCCL all_reduce 参数不一致，报 `EI0005/EI0006` |

适用的 recipe（主补丁内）：

```
路由专家 (40×384×w1/w2/w3)      -> W4A8_DYNAMIC (int4 per-channel SSZ + int8 per-token)
注意力 wq_a/wq_b/wkv             -> W8A8_DYNAMIC
indexer.wq_b (8 个 index source 层) -> W8A8_DYNAMIC
共享专家 w1/w2/w3                -> W8A8_DYNAMIC
gate / norm / hc_* / wo_a / wo_b / compressor.* / indexer.wk/weights_proj / embed / head -> FLOAT
vision.* / aligner.* / mtp.*     -> 不在适配器参数内（FLOAT；vision 需事后补回，见指南 §7）
```

## 3. 可选补丁（实验，不影响主线）

| 项 | 值 |
|---|---|
| 文件 | `patches/msmodelslim_v41_w4a8_hiaux_optional.patch` |
| md5 | `cdd41e7ea0c9981fdfc5d4f78cacab47` |
| 规模 | 144 行 / 1 文件（新增 `lab_practice/deepseek_v41/deepseek_v4_1_flash_w4a8_hiaux.yaml`） |
| 用途 | 把主干的 37/38/39 层路由专家改回 W8A8（其余同主线），用于 DSpark draft 接受率实验 |
| 状态 | **实验分支**，不参与主线复现，不纳入主补丁；应用/跳过由 `--with-hiaux` 控制 |

## 4. 推荐 apply 顺序（唯一合法顺序）

```bash
P=/home/user/projects/dsv41
R=$P/src/msmodelslim      # 目标 checkout，HEAD 应为 92e219f

# 干跑：只 check，不改树
bash $P/scripts/apply_msmodelslim_patch.sh --dir $R --check

# 正式应用（主补丁 + 现场创建 editable 安装需要的目录软链）
bash $P/scripts/apply_msmodelslim_patch.sh --dir $R

# 或：主补丁 + 可选 hiaux 配方
bash $P/scripts/apply_msmodelslim_patch.sh --dir $R --with-hiaux

# 只想手动来：
cd $R
git apply --check -p1 $P/patches/msmodelslim_v41_w4a8.patch
git apply       -p1 $P/patches/msmodelslim_v41_w4a8.patch
git apply --check -p1 $P/patches/msmodelslim_v41_w4a8_hiaux_optional.patch
git apply       -p1 $P/patches/msmodelslim_v41_w4a8_hiaux_optional.patch   # 可选
```

规则：

1. **先 `--check` 再 apply**；任何一步非 0 就停下，不要用 `--reject` / `--3way`。
2. 目标树必须是干净的上游基线（或包含未改动补丁涉及文件的后代提交）；HEAD 不等于
   `92e219f` 时脚本只警告，`git apply --check` 才是真正的判定。
3. 应用后必须刷新 editable 安装的包内配置：
   `config/config.ini` -> `msmodelslim/config/config.ini`，否则 CLI 看不到
   `deepseek_v41` 适配器。`apply_msmodelslim_patch.sh` 默认自动做（`--no-layout` 可关）。

## 5. 验证记录（本次交付）

- `git apply --check` + `git apply` 在一个干净的 `92e219f` worktree 上通过；
- apply 后与 `e05b471` 工作树逐文件比对：`deepseek_v41/model.py`、`deepseek_v4_1_flash_w4a8.yaml`、
  `config/config.ini` 完全一致；
- 可选补丁单独 apply 后与工作区 `*_hiaux.yaml` 完全一致；
- 复现见 `logs/perf/msmodelslim_repro_impl.md`。

## 6. 不纳入补丁、但必须现场准备的东西

| 项 | 说明 |
|---|---|
| `msmodelslim/config_repo -> ../config` | CLI 从包内找 config_repo；`prepare_mslim_layout.sh` / apply 脚本创建 |
| `msmodelslim/lab_practice -> ../lab_practice` | 同上（`.gitignore` 已忽略） |
| `msmodelslim/lab_calib -> ../lab_calib` | 同上 |
| `msmodelslim/config/config.ini` | 从根目录 `config/config.ini` 复制，确保 `deepseek_v41` 注册生效 |
| `env/mslim-venv`、官方 checkpoint、`models/out/*` | 环境与数据，不属于源码补丁 |
