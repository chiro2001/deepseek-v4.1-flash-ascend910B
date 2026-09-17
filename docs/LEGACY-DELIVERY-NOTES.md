# 交付流程详解（A2 侧）

> 本文是从早期内部交付包整理出来的 A2 详细流程与开关说明，内容仍然有效。
> 快速上手请看仓库根目录的 [`README.md`](../README.md)。

# A2 交付测试包 v6 —— DeepSeek-V4.1-Flash W4A8（TP8 + Engram-int8 + DSpark + Vision）

> **v6 只修交付工程，不改性能配置**：`patches/` + `Dockerfile` + `optim/` 与 v5 **逐字节相同**。
> ⇒ 若你已有 `dsv41-a2:v5` 镜像，**直接 `docker tag dsv41-a2:v5 dsv41-a2:v6` 即可，不用重建**。

## 怎么跑（v6 的流程变了：**先自检，再起服**）

```bash
cd a2_pkg_v6

# ⓪ 【新增，30 秒，不起容器】9 组自检：文件完整性 / tag 一致性 / 软链链 /
#    官方目录 / datasets 版本 / OMP / NPU↔NUMA / skcache / 残留容器
MODEL=/path/to/v41-w4a8-engram-dr-vision-qrot-mtpq bash tools/preflight_a2.sh
#    FATAL=0 才往下走

# ① 镜像：已有 v5 就直接 retag，否则构建（约 10–20 min）
docker tag dsv41-a2:v5 dsv41-a2:v6        # 内容相同，秒完成
# bash scripts/build_image.sh             # 只有在没有 v5 镜像时才需要

# ② 单流口径验收（性能 + 视觉 + 容量）
MODEL=/path/to/... bash scripts/run_test.sh
MODEL=/path/to/... MODE=full bash scripts/run_test.sh     # 加 128K + GSM8K

# ③ 生产口径（A2 真机：max-num-seqs 32 + prefix caching ON）+ 多 batch 三块
MODEL=/path/to/... MODE=prod bash scripts/run_test.sh     # 一臂（PREFIX=1）
MODEL=/path/to/... bash tests/multibatch/run_prod_both.sh # 两臂：PREFIX=1 与 PREFIX=0
```

**两个一键自检**（出厂验收用，也可以自己跑）：

```bash
bash tools/selfcheck_pkg.sh      # 包自检：文件 / tag / 语法 / MANIFEST
bash tools/negative_control.sh   # 负控：证明每个自检**真的能抓到**它对应的 bug
```

**PGO**（可选，**在目标机上编译一次**；包内不含二进制）：

```bash
bash build_scripts/00_ensure_pgo.sh          # 首次 ~30–40 min；之后指纹一致即秒退
PGO_MTUNE=1 bash build_scripts/00_ensure_pgo.sh   # 可选：-mtune=native（预期 0~3%，需 A/B）
SKIP_PGO=1 bash scripts/build_image.sh       # 完全不要 PGO 也可以
```

> 为什么不打包二进制：PGO 产物与 **CPU 微架构 / gcc / glibc** 绑定（实测 A2 与 A3 镜像的
> libpython md5 就不同），且 30 MB 二进制不该进 git 历史。详见 `optim/pgo/README.md`。

不需要外网。包内**不含模型权重**（273 GB，你已备好等价的 `…-qrot-mtpq` 目录）。

> **v4 与 v3 的差别（3 条）**：① 新增**生产口径**（`MAX_SEQS=32 + PREFIX=1`）与
> **多 batch / 多轮对话验证**（v3 完全没有：历史"并发"只是 decode 并发，prefill 被
> `--serialize-prefill 1` 故意串行化 → **v4 的 [C] 块是"prefill 与 decode 同时跑"的首次覆盖**）；
> ② 性能预期改为 **163 发全量重算**
> （**A 不是单峰**，≥110 tok/s 目前**不可交付**）；③ 新增 4 个**负结果**与一份
> `CORRECTNESS_STATUS.md`（数值确定性 / 精度门的独立结论）。
>
> **v5 与 v4 的差别（1 条，但是致命的）**：**修掉"软链构造的模型目录"起服必失败**的问题。
> 量化产物是零拷贝的软链结构，**94 个软链里 80 个是 4 跳深**；而 v3/v4 只挂了
> `-v "$MODEL:$MODEL"` 一层 ⇒ 容器内所有软链悬空 ⇒ `No such file or directory`。
> v5 新增 `tools/model_mount_args.sh`（逐跳解析并逐层挂载，真值需挂 **13 个目录**），
> 并在起服前用 `tools/check_model_dir.sh` 检查悬空软链 + 报告链条深度。
> **如果你用 v3 或 v4 起服过并失败在 config/tokenizer 上，就是这个原因**，见 `REPRO.md` §3.0。
>
> **v6 与 v5 的差别（交付工程，不动性能配置）**：v5 在 A2 上暴露 **10 个 bug**，其中
> **8 个是我们自己的脚本/打包错误**，共同点是"**只有跑到最后一步才暴露**"（起服一次 5–25 min）。
> v6 做了三件事：① **新增 `tools/preflight_a2.sh`**（30 秒 / 9 组 / 不起容器，把检查前移）；
> ② **新增 `tools/negative_control.sh`**（8 个负控，**证明每个自检真的能抓到它对应的 bug** ——
> v5 的 `selfcheck_pkg.sh` 曾"全过"而包是坏的）；③ **PGO 编译流程进包**
> （`build_scripts/`，distro-aware，指纹缓存"只编译一次"）。
> 两个最要命的修复：**缺 `tests/p15_stream_curve_filefiller.py`**（性能测试全废）与
> **缺 `tests/vision_accuracy_check.py`**（视觉必 FAIL）；以及**失败后不清理容器**
> （残留占 313 GB，导致第二次起服叠加失败）。详见 `CHANGELOG.md` §v6。

---

## 0. 预期结果（A3-node1 + A3-node2 实测，供你判"复现成功"）

| 指标 | 我们（A3-node1，8×910C） | 你应该看到 |
|---|---|---|
| 128K 单流 ms/step | **30.2 – 31.6**（中位，PGO 开/关），最好 **26.9** | 会更高（910B + 弱 CPU），**趋势对**即可 |
| 接受长度 A（128K） | **不是单峰**：steep 15.3% / flat 7.4% / shallow 77.3%（163 发） | 8 发中位 2.28–3.08 算装对了 —— **但 A 不能当绩效指标**，见 §0.1 |
| 单流 tok/s（128K） | 中位 85–90 | 会低；**别追 110**（见 §0.1） |
| KV 池 tokens | 4,164,000 | **必须 > 3,145,728（3Mi）** |
| Vision 23 例 | **23/23** | **≥ 19/23** |
| GSM8K-200 | **198/200、199/200、197/200**（三次） | 同量级（≥197） |
| `static_kernel.py:650` 命中数 | 0 | **必须 0**（否则静默降级，数字不可信） |
| **多 batch 三块**（生产口径 `MAX_SEQS=32 PREFIX=1`） | **三块全过**（`EXPECTED_PERF.md` §7.3）：A 轮内 **7/7** + 全长 **3/3**（`uniq2` 全 1.00）；B `conc=1`/`conc=8` 各 16/16，**逐项不一致 0**；C 128K 与 6 短请求并发，基线/混合各 6/6，**逐项不一致 0** | **逐 item 不一致 = 0**、多轮召回全中。⚠️ 局限（§7.3.1）：B 题简单、多轮只到 409 token、单次测量、`PREFIX=0` 臂未跑 |

### 0.1 ⚠️ v4 必须纠正的两个旧说法

1. **"A 一旦进优模式就能冲过 110 tok/s"是误导**。163 发全量统计：≥110 tok/s 共 **8 发**，
   其中 **7 发来自 `MOE_ZERO` 会话**（已判"换吸引子的复读"，**不可采纳**）
   ⇒ **真正可交付的 ≥110 只有 1 发**（`faB_128k_r7`，A=3.446 @ ms=31.185 ⇒ 110.5 tok/s）。
   steep 形态（健康）只占 **15.3%**，且是**每发抽签**，不是配置属性。详见 `EXPECTED_PERF.md` §2。
2. **不许用 A 的绝对值当绩效指标**：同一个配置、同一个 prompt，两次起服的"干净请求占比"
   可以差 **2.3 倍**（37% vs 16%），而 **A 在脏会话里反而更高**（A 与文本质量**反相关**）。
   ⇒ 必须报 **`(clean-rate, ms/step)`**；`ms/step` 不受影响（时延线照常）。
   详见 `EXPECTED_PERF.md` §3 与 `reports/session-attractor-and-clean-rate.md`。

---

## 1. 已知不达标 / 未验证 / **别踩坑**（先看这里）

### 1.1 本轮（v5）新出的 4 个**负结果** —— 都**不采纳**

| 项 | 我们试了什么 | 结论（**别重复**） |
|---|---|---|
| **`MOE_ZERO`**（把 `init_routing` 的无效行**全量清零**） | 同会话交错 A/B，逐发切换 `zero=0/1` | ❌ **不采纳**：它把 `pos0` 的低峰（污染请求）消灭了，**但把高峰从 0.86 压到 0.73** ⇒ **只是换了一个（更平、更稳定的）吸引子，不是修复**；而且**非数值等价改动**（最常见 top-1 从 `</s>`(−0.776) 变成 `《`(−4.201)）。见 `reports/session-attractor-and-clean-rate.md` §1.1/§1.2 |
| **`MOE_NONFINITE`**（只清零无效行里的**非有限**元素） | 同会话交错 A/B，**N=24/臂** | ❌ **无差异**：clean 各 **2/24 = 8.3%**，2×2 `[[2,22],[2,22]]` → **修正后 p=1.0000**（旧打印值 `0.0000` **作废**——两行计数相同时 p 必为 1.0）。<br>性能侧（128K 生产口径 3 发）：`nf=1` → A=2.024/1.536/1.977、ms=35.0/33.0/35.0；`nf=0` → A=2.336/1.401/2.265、ms=33.3/32.1/33.2 ⇒ **额外的 `isfinite` 检查没有明显代价，但也没有收益**<br>⇒ **`0 权重 × Inf = NaN` 这条污染通道被排除**（开关 `MOE_NF`，默认 0，只作复现用） |
| **`LOCAL_OWNER=fast` vs `on`** | 同会话交错 A/B，N=12/臂 | ❌ **无差异**：2×2 `[[5,7],[4,8]]` → **修正后 p=1.0000**（旧值 0.6843 作废）。**我们仍默认 `fast`**（它更快，且是 Engram JIT plan 分支的前置条件）—— 但**不要**再说"fast 让系统更干净" |
| **`HCCL_DET=true`** | 同会话交错 A/B（N=12/臂）+ 精度门 | ❌ **无差异**：`[[5,7],[3,9]]` → **p=0.6668**（旧值 0.6843 作废，且**方向/数值都不同**）；**且已知会把 GSM8K 打到 91/100 ⇒ 绝对不能进交付**（只能当诊断工具）。合法值只 `true/false/strict`，写 `1` 直接起服失败（EI0001）。`strict` 保精度（100/100）但"确定性"只是**每批抽签**（同一 ctx 三次重复 = 0.913/0.000/1.435）⇒ 见 `CORRECTNESS_STATUS.md` |

> ⚠️ **统计纪律（v4 新增）**：上表的 p 值是**修正后**的值（来源
> `reports/session-attractor-and-clean-rate.md` §6.4 的完整重算表）。旧值一律作废，原因是
> `exp_tools/interleave_ab.py` 的 `fisher()` 有 bug：**两行计数完全相同时返回 `0.0000`**
> （相同表必为 1.0）；根因是列和固定却未遍历所有 `x`，且双侧判据在 `obs` 为最大概率时退化 0/0。
> 修正后已用**教科书标准值**校验（茶实验 `[[3,1],[1,3]] → 0.4857143` ✅、
> `[[10,0],[0,10]] → 1.08e-5` ✅、`[[1,9],[11,3]] → 0.0027595` ✅）。
> **本包附 `tools/fisher_recheck.py`（内置 4 个自检用例）：任何统计工具上线前先用已知答案自检。**
> 唯一来自**独立实现**的显著结果是 `SPEC=0` vs `SPEC=1`（见 `CORRECTNESS_STATUS.md` §3）。

### 1.2 DSpark 入图（`DRAFT_GRAPH`）的状态更新

| 项 | v3 写的 | **v4 更正** |
|---|---|---|
| DSpark draft 入 ACLGraph | ❌ 上卡验证未做 | **离线修复完成 + 负控已确认；正控待验** ⇒ 仍然**默认关**（`DRAFT_GRAPH=0`） |

**负控指纹（`DRAFT_GRAPH=1` 但缺 `DSPARK_GRAPH_CAPTURE_METADATA=1`）**，见
`reports/draft-graph-negative-control.md`（逐字复制在本包）：

* `A 恒为 1.000`（8/8 发：1.000/1.008/1.000/…）⇒ draft token **全被拒绝**；
* **`ms/step` 仍然是 30.2–30.8，与 eager 同量级** ⇒ **单看时延完全发现不了这个错误**；
* 指纹：`Wrapping draft model = 8`、`Target sizes = 0`、`dspark-graph-capture = 0`。

⇒ **判据**：开了 `DRAFT_GRAPH=1` 之后必须同时看 **`A`（要求 > 1.5）与 `dspark-graph-capture > 0`**，
**只报 ms 的"验证"是无效的**。A2 上潜在收益最大（~24 ms/轮，因为 A2 的 draft 是 eager 且 CPU 弱），值得专项一试。

### 1.3 v3 已列、v4 保持的项

| 项 | 状态 | 处置 |
|---|---|---|
| 阶段 1 主干量化的分片 sha256 | 与我们的**必然不同**（DP 宽度不同：72 vs 80 分片，47.5% 张量有 int8-LSB 差异） | **禁止用 sha256 判失败**；只用「结构 + 容差」→ `quant/scripts/verify_structure.py` |
| 量化脚本的设备号 | A3-node1 默认 `0..15` | A2 必须 `DEVICE_IDS="0 1 2 3 4 5 6 7"`，写 8–15 会 **`ExchangeDevice` 报错** |
| `LOAD_FORMAT=dummy` | 只按 shape 建模型、不读权重 | 只测**时延**；**A 恒为 1.0**，不能用于精度/容量/接受长度判据 |
| `HCCL_DETERMINISTIC` | 合法值只 `true/false/strict`（写 `1` 起服失败） | 默认**不传**；`true` 掉 GSM8K 到 91/100（**不能进交付**），`strict` 保精度但确定性只是"每批抽签" |
| `max-model-len=1048576` | 我们只在 8K/128K 包络验证过 | 不稳就 `MAX_LEN=131072` 先跑通 |
| PGO 在 A2 的实际收益 | 产物与版本已核，未在 A2 起服验证 | 缺产物/版本不符会**自动降级**为 `PYTHON_PGO=0`；起服异常先关它排除 |

---

## 2. 包里有什么

```
a2_pkg_v5/
├── README.md              ← 本文件（一页读完）
├── REPRO.md               ← 详细复现：每个开关、预期数字、判据、失败排查表
├── CHANGELOG.md           ← v3→v4 逐项 diff（含"v4 移除了什么说法"）
├── EXPECTED_PERF.md       ← 性能预期（163 发全量形态分类 + 生产口径 + 原始逐发数据）
├── CORRECTNESS_STATUS.md  ← 【新增】精度 / 数值确定性 / 精度的独立结论（含 p 值与来源行号）
├── PACKAGING_REPORT.md    ← 打包报告（每个数字来自哪份报告/日志的哪一行；我核不了的项）
├── patches/               ← 全部 bind-mount 整文件补丁 + md5 + 差异说明（PATCHES.md）
├── quant/                 ← 完整量化链（5 级装配）+ 结构/容差验收（不用 sha256）
├── scripts/
│   ├── build_image.sh     ← 【命令 ①】生成修改后的容器镜像
│   ├── run_test.sh        ← 【命令 ②】起服务 + 跑验收（MODE=quick|full|prod）
│   ├── serve_a2.sh        ← 起服（serve_a21.sh 的 A2 适配：DEVS 0..7 + PREFIX/DRY_RUN）
│   └── serve_v2.sh        ← A3-node1 的权威起服器（原样，仅修了一处硬编码路径）
├── tests/
│   ├── multibatch/        ← 【新增】多 batch / 多轮对话（v4 最大的一块）
│   │   ├── multibatch_gate.py       ← [A]多轮 [B]并发逐 item [C]长短交错（参数化，无硬编码路径）
│   │   ├── multibatch_session.sh    ← 生产口径起服 + 跑三块（单臂）
│   │   ├── run_prod_both.sh         ← 两臂对照（PREFIX=1 vs PREFIX=0）+ 汇总表
│   │   └── verify_serve_flags.sh    ← 12 组合启动器烟测（DRY_RUN，**不占卡**）
│   ├── t_quote.sh / t_vision.py / t_gsm8k.py / make_report.sh
│   ├── acc_eval.py        ← 【v4 补入】GSM8K/C-Eval 评测器（v3 的 t_gsm8k.py 引用了它却漏打进包）
├── tools/                 ← quote 测量 / 容量 / 视觉 / 模型目录自检 / 开关切换
│   ├── fisher_recheck.py  ← 【v4】Fisher 精确检验（2×2）+ 4 个教科书自检用例
│   ├── interleave_ab.py   ← 【v4】同会话交错 A/B（pos0/clean-rate 判据，容器内 /tmp 热切换）
│   └── steep_summary.py   ← 【v4】从 p42 jsonl 汇总 steep/flat/shallow 分布（= EXPECTED_PERF §2 的工具）
├── data/                  ← 红楼梦语料 + quote 后缀（测量输入）
├── reports/               ← **逐字复制** A3-node1 的 reports/*.md（**55 份**，含 v4 新增 4 份权威报告）
├── logs_meta/             ← 原始日志路径清单（LOG_INDEX.md）+ 关键 jsonl 样本
└── optim/pgo/             ← PGO 构建入口（产物在目标机生成：python3 + libpython3.12.so.1.0）
```

---

## 3. 起服后**必查四件事**

```bash
R=results/<run_id>            # run_test.sh 会打印这个目录

# ① 静态内核没被静默降级（必须输出 0）
grep -ac "static_kernel.py:650" $R/serve.log

# ② KV 容量
grep -oE "GPU KV cache size: [0-9,]+ tokens" $R/serve.log | tail -1

# ③ 口径对不对（性能口径 vs 生产口径**不可混比**）
grep -E "serve_a2\] 口径|MAX_SEQS|PREFIX" $R/serve_cmd.txt

# ④ 结论一页纸（MODE=prod 时会自动带 §1b 多 batch 表）
less $R/REPORT.md
```

---

## 4. 主要开关（都在 `scripts/serve_a2.sh` / `scripts/run_test.sh` 顶部，带默认值）

### 4.1 v4 新增 / 变更

| 开关 | 默认 | 作用 / 收益 |
|---|---|---|
| **`MODE=prod`**（`run_test.sh`） | 不设 | 生产口径：`MAX_SEQS=32 + PREFIX=1` + 多 batch 三块（A/B/C），**跳过 quote 性能**（口径不可比） |
| **`PREFIX=1`**（`serve_a2.sh`） | **0** | `1` = 启用 prefix caching（**A2 生产口径**）；`0` = 历史性能口径（与 v3 逐字节一致） |
| **`MAX_SEQS`** | 4（`prod` 时 32） | 并发请求数。⚠️ 它决定 `CAPTURE_SIZES` 要覆盖到 `MAX_SEQS × (1+SP_TOKENS)`：`=32` 时要到 **192**，旧逻辑只到 32 ⇒ **直接起不来** |
| **`CAPTURE_SIZES` 自动推导** | 自动 | `MAX_SEQS=1` 时输出 `1,2,3,4,6,8,12,16,20,24,32`（**与 v3 逐字节相同**）；`=32` 时扩到 `…,40,48,96,192`（15 桶）。**不要手写** |
| **`DRY_RUN=1`** | 0 | 只解析开关 + 组装 MOUNTS 并打印，**不碰 docker**（抓 `set -u` 顺序 bug；`bash -n` 抓不到） |
| `MOE_NF` | **0** | 负结果臂（只零化非有限元素），默认关，只作复现用 |
| `HCCL_DET` | **空** | 仅诊断；`true` 掉 GSM8K 到 91/100（**不能进交付**），`strict` 保精度但确定性只是"每批抽签" |
| `MBG_ROUNDS` / `MBG_CONC` / `MBG_LONG_CTX` / `MBG_SKIP` | 8 / 8 / 131072 / 空 | 多 batch 三块的参数（`MODE=prod` 时生效） |

### 4.2 已验证优化（默认全开，与 v3 相同）

| 开关 | 默认 | 作用 / 收益 |
|---|---|---|
| `MOE_AG=1` | 开 | MoE AllGather：128K **−4.25 ms**，KV 3.39M→4.16M |
| `SP_TOKENS=5` + `CAPTURE_SIZES` 含 6 | 开 | DSpark 原生 block size；漏配 capture sizes 会 **+5.9 ms** |
| `O_PROJ_2D=1` | 开 | F3：`wo_a` 2D matmul，−0.31~0.76 ms |
| `MOE_MASK=1` | 开 | moe-mask-range，−0.51 ms（GSM8K 100/100、Vision 23/23） |
| `ROPE_IDXSEL=1` | 开 | rope 取表链 6→2 kernel，−0.45~0.62 ms |
| `ENGRAM_JIT=1` + `LOCAL_OWNER=fast` | 开 | hash 0.427→0.076、plan 0.261→0.068 ms |
| `QLI_NOCAND=1` | 开 | QLI per-op 99.3→50.3 µs，−0.49 ms |
| `PYTHON_PGO=1` | 开 | PGO+LTO libpython：服务侧 **−4.4%**（A2 CPU 弱，收益应更大） |

---

## 5. 出问题先看

1. `REPRO.md` §7 **失败排查表**（现象 → 原因 → 处理）。
2. `results/<run_id>/serve.log` 尾部 + `serve_cmd.txt`（实际用的全部开关 **和口径**）。
3. `CORRECTNESS_STATUS.md` —— 任何"确定/不确定""A 高不高""精度够不够"的问题先看它。
4. `reports/` 里对应的那份报告（多 batch **权威报告 = `reports/multibatch-and-mixed-load.md`**，
   A 的形态在 `reports/a-basin-and-acceptance-shape.md`，clean-rate 在
   `reports/session-attractor-and-clean-rate.md`，DSpark 负控在
   `reports/draft-graph-negative-control.md`）。
5. 健康检查三连（依次排除）：`PYTHON_PGO=0` → `STATIC_KERNEL=0` → `MAX_LEN=131072`。

回滚：镜像内每个被覆盖的文件都留了 `.a2orig`；`tools/enable_*.sh off` 可还原实验补丁；
也可以直接用基础镜像起服（会失去全部 v4 收益）。
