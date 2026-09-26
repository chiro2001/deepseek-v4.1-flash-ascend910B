# 135 — 上下文交接（2026-09-23 15:xx）

> **交接目标**：让下一个 agent **不重启、不重做**，直接从"分析 A2 新分支 `OFFLOAD=0 + KV8_SWA=1` 的精度问题"接上。
> **权威状态 = 工作区与远端**，不是本文档的叙述。动手前先 `git -C dsv41-release log --oneline -1` 与 `git -C main-wt log --oneline -1` 核对。
> 标记：**【实测】/【推断】/【未确认】**

---

## 0. 一句话现状

**【实测】找到了「长上下文乱码」的根因回归并已修**：乱码根因修复 `8eb2613`（`BAT_TOKENS 2048 → 8192`）改的是**模板** `scripts/serve_a2.sh`，
但**两个包装脚本各自又写死了 2048 并显式传下去**，把修复覆盖掉了 ⇒ 长上下文必坏。
两处都已修并 push。**A3 上的负控已复现**（`BAT=2048 @ 520K` ⇒ **12/12 全 FAIL**）。

**用户最新报告（本文档要移交的主任务）**：
> "A2 用新分支、**关闭 dram offloading**、**启用 kv swa** 还是有精度问题"

⇒ 这是**在 `BAT_TOKENS=8192` 已修/待确认之后**的剩余问题。**第一步必须确认 A2 那台真的在跑 8192**（见 §3），
否则只是又踩同一个坑。

---

## 1. 权威状态（动手前先核）

| 项 | 值 |
|---|---|
| **feat 分支** | `~/projects/dsv41/dsv41-release`，`feat/kv8-dram-offload-pending`，HEAD **`8106b8f`**（ahead 0，已 push） |
| **main 分支** | `~/projects/dsv41/main-wt`（worktree），`main-a3-deploy` → 远端 `main`，HEAD **`7e902a1`**（已 push） |
| 工作区 | `~/projects/dsv41`（**不是 git 仓**）；日志 `a2/logs/`（索引 `README.md`） |
| **A2** | `l00886679@141.61.9.11`，**ssh 不可达 ⇒ 命令必须给用户粘贴**；容器 `dsv41-a2`，端口 **8077** |
| **A3-21** | 可 ssh（`ssh a3-21`）；本任务用 **Phy-ID 8–15**（**0–7 不是我们的**） |
| A3 现场 | 服务 **Up 40 min，health=200**，跑 `BAT=8192 / MAX_LEN=1M`，模型 `~/models/out/v41-flat-verify3` |
| A3 镜像 | **内网 A3 已用 TP8 镜像拉起成功**（用户确认）；层补丁包见 §6 |

---

## 2. ★★ 本次最大发现：`BAT_TOKENS` 2048 回归（两处，都已修）

### 2.1 事实链（可复算）

1. 乱码根因修复 = `8eb2613`「**BAT_TOKENS 2048 -> 8192，修复长上下文输出退化**」，改的是 **`scripts/serve_a2.sh`**（现 `:-8192`）。
2. **`a2/scripts/serve_a2_offload.sh:144`** 是**后来写的包装脚本**，第 584 行**显式** `BAT_TOKENS="$BAT_TOKENS"` 传给模板 ⇒ **它的默认值必然覆盖模板的 8192**；而它写的是 **`:-2048`**。
3. **`scripts/run_test.sh:62`** 同样：`:-2048` + 第 178 行显式透传 ⇒ **"标准验证入口"自我退化**（而 `EXPECTED_PERF.md` 写的标准验证命令就是 `MODE=prod bash scripts/run_test.sh`，`tests/agent_trace/` 那套精度门就在里面）。
4. ★ **它自己的 KV 门槛 `KV_MIN=2800000` 本来就是按 `BAT=8192/GPU_UTIL=0.92` 的实测 2,823,080 定的**（README 的 GPU_UTIL 表；2048 时是 4,145,957）⇒ 这两处本来就该是"8192 这一对"。

### 2.2 代价（`reports/longctx-accuracy-fix.md` 实测曲线）

机制：chunked prefill 每切一刀一次独立"偏离"（约 **2%/chunk**，误差沿 chunk 累积）
⇒ **失败率 ≈ 1 − 0.98^(prompt / BAT)**，chunk 数 = `ceil(prompt / BAT)`。

| prompt | BAT=2048（chunk） | 通过率 | BAT=8192（chunk） | 通过率 |
|---:|---:|---:|---:|---:|
| 60,012 | 30 | 5/10 | 8 | 10/10 |
| 79,855 | 40 | 3/10 | 10 | 10/10 |
| 149,986 | 74 | ~0% | 19 | 6/6 |
| 259,985 | 127 | ~0% | 32 | 6/6 |
| **≈520,000（我们现场）** | **254** | **≈0%（外推）** | **64** | **未验证** |

### 2.3 ★ 负控已实测（这是"探针有效"的证明）

在 **a3-21**、`MODEL=v41-flat-verify3`、`MAX_LEN=1M`、**`BAT=2048`**、`CPU_BIND=0` 下跑
`tools/ctx_agent_probe.py --mode needle --context-tokens 520000`（3 个不同语料区段 ⇒ **12 份互不相同的样本**）：

```
offset=0       实际≈517,036 tok  [A][B][C][D] 全 FAIL
offset=700000  实际≈520,081 tok  [A][B][C][D] 全 FAIL
offset=1400000 实际≈520,823 tok  [A][B][C][D] 全 FAIL
⇒ 12/12 FAIL
```

**失败形态（答案原文 repr）**：
```
[A] 期望 ZQ7K-3341 → 实际 ''
[B] 期望 VX2M-8890 → 实际 ''
[C] 期望 HT4P-5527 → 实际 '</p>'
[D] 期望 RB9N-6014 → 实际 ''
```
⇒ **不是 U+FFFD 乱码，是"空回复 / 残片"**（`fp=0/0`，`repeat_loop=False`）。
这与 `8eb2613` 的原始描述**逐字吻合**：
> "长上下文（>=60K）agent 场景下模型输出复读/幻觉、**不再调用工具，最终被解析器整段丢弃成"空回复"**"

⇒ ★★ **这解释了两件事**：① 用户看到的"精度问题/乱码"很可能是**空回复/答非所问**而不是替换字符；② **探针能检测出已知会坏的配置** ⇒ 它此前在 A3 上的全 PASS 是**有意义的**（不是判据失灵）。

**证据文件（a3-21，别删）**：`~/negctl_evidence/`（3 份）+ `~/negctl_{0,700000,1400000}.json`。

### 2.4 已修（两处，都 push 了）

| 提交 | 分支 | 内容 |
|---|---|---|
| **`8106b8f`** | `feat/kv8-dram-offload-pending` | `a2/scripts/serve_a2_offload.sh` 默认 2048 → **8192**；参数一致性守卫（放在**所有参数定义之后** —— 第一版放在 `BAT_TOKENS` 定义处，那里 `MAX_LEN/MAX_SEQS/OFFLOAD` 还没赋值 ⇒ 踩 `set -u` 崩）；显式 <8192 时**响亮警告**；注释里"生产配置 …BAT_TOKENS=2048"加更正 |
| **`7e902a1`** | `main` | `scripts/run_test.sh` 默认 2048 → **8192**；新增**必查 ①b**：起服后断言**引擎实际收到的** `--max-num-batched-tokens` 与 `inner.sh` 的 `BAT_TOKENS` **都等于**期望值并写 `env.txt`（判据绑"实际生效"）；新增 `tools/selftest_run_test_bat.sh`（7 条，含**负控**：改回 2048 必须判 FAIL） |

**负控均已实测**：改回 2048 ⇒ 断言报 FAIL。

---

## 3. ★ 交接主任务：A2 新分支 `OFFLOAD=0 + KV8_SWA=1` 仍有精度问题

### 3.1 用户的报告

> "目前 TP8 A3 镜像版本的启动没有问题，**A2 用新分支关闭 dram offloading 启用 kv swa 还是有精度问题**。分析一下"

### 3.2 ★★ 第一步（**必须先做，否则可能只是又踩同一个坑**）

让用户粘贴这四条，把"实际生效的配置"钉死：

```bash
cd <A2 上的仓目录> && git pull --ff-only && git log --oneline -1   # 期望含 8106b8f
L=$(ls -t <shadow-pkg>/results/*/serve.log | head -1)

# ① prefill batch（本次的根因嫌疑）
grep -o -- '--max-num-batched-tokens [0-9]*' "$L" | tail -1        # 期望 8192
grep -o 'BAT_TOKENS=[0-9]*' "$(dirname "$L")/inner.sh" | tail -1   # 期望 8192

# ② 确认 OFFLOAD 真关着（0 = 不挂 offload 补丁、不带 --kv-transfer-config）
grep -c 'offloading/scheduler.py' "$(dirname "$L")/serve_cmd.txt" 2>/dev/null
grep -o -- '--kv-transfer-config' "$L" | head -1                   # 期望**无输出**

# ③ 确认 int8 档 C 真生效（env 必须进了容器）
grep -m1 -a 'VLLM_V41_KV8_SWA=' "$(dirname "$L")/inner.sh"         # 期望 1
grep -m1 -a 'VLLM_V41_APC_ALIGN=' "$(dirname "$L")/inner.sh"       # 期望 3
grep -m1 -a 'VLLM_V41_KV8_GRAPH_SAFE=' "$(dirname "$L")/inner.sh"  # 期望 1

# ④ 上下文/并发以及**接受的 A 值**（最灵敏的行为判据）
grep -o -- '--max-model-len [0-9]*' "$L" | head -1
grep -oE "Mean acceptance length: [0-9.]+" "$L" | tail -3          # 健康 2.7–3.0；≈1.0 ⇒ draft 静默失效
```

**判读分叉**：
* 若 ① 不是 8192 ⇒ **仍然是同一个坑**（A2 那台没拉到我修的 8106b8f，或用了 `run_test.sh`）。让用户 `git pull` 后重跑，先别做别的。
* 若 ① = 8192 且 ②③ 都对 ⇒ 才是**新问题**，按 §3.3 切。

### 3.3 若是新问题：单变量切法（每次只改一个，ENGRAM 全程 =1）

按**代价从低到高**（A2 起服约 5–15 min，改配置必须重启）：

| 顺序 | 改什么 | 判什么 | 判据 |
|---|---|---|---|
| 1 | `KV8_SWA=0 KV8_RING_FP16=0`（保留 `OFFLOAD=0`） | **int8 档 C 是否有罪** | 变干净 ⇒ int8 引入；仍坏 ⇒ 与 int8 无关 |
| 2 | `BAT_TOKENS=16384` | chunk 数 64 → 32（520K） | 按 `1−0.98^n`：64 刀 ≈27% 通过、32 刀 ≈52% ⇒ **若 16384 明显更好，说明仍在 chunk 累加这条线上**（★ 这是**最可能**的方向：见 §5） |
| 3 | `MAX_LEN=262144`（或让 520K 请求直接被拒） | 是否**只有 ≥某长度**才坏 | 缩短后干净 ⇒ 长度相关 |
| 4 | 只跑**单发**（不做并发/追问） | 并发是否是触发条件 | |
| 5 | `ENGRAM=0` | Engram 是否参与（**质量降级，只用于排查**） | |

**标准工具（main 上，已交付）**：
```bash
python3 tools/ctx_agent_probe.py --selfcheck        # 先验判据（应"失败 0 项"）
python3 tools/ctx_agent_probe.py --base-url http://127.0.0.1:8077 --model deepseek-v4-flash \
    --mode needle --context-tokens 520000 --offset 0 --repeats 1 --out ~/r1.json
python3 tools/ctx_agent_probe.py ... --mode biggrow --context-tokens 520000 --followups 3
python3 tools/ctx_agent_probe.py ... --mode mixed --context-tokens 520000 --conc 4
```
判读只看三处：`PASS/FAIL`（**逐字**）、`fp=U+FFFD/NUL`（非 0 = 实锤乱码）、`bad_ctx`（**必须 0**，非 0 ⇒ 该次读数无效）。

★ **短上下文锚也要同时跑**（区分"长上下文特有"还是"全局坏"）：
```bash
PORT=8077 SERVED_NAME=deepseek-v4-flash NAME=dsv41-a2 SKIP=vision,gsm8k bash tools/attach_test.sh
# 或单独跑 GSM8K（需 ENC_DIR）：MODE=full RUN_GSM8K=1 bash tools/attach_test.sh
```

### 3.4 已知会误导的两点（先知道）

1. `tools/attach_test.sh` 的**必查③ KV 门槛是 3Mi**，而 main 现在默认 `GPU_UTIL=0.92` ⇒ KV 约 **2.82M** ⇒ **这一项会红**。这是**门槛没跟着默认值更新**（已知不一致，**未修**），不是服务有问题。
2. `attach_test.sh` 的 `PORT` 默认 **8100**、`NAME` 默认 **`dsv41-a2`**、`SERVED_NAME` 默认 **`deepseek-v41`** ⇒ **全部要显式覆盖**（特别是 `SERVED_NAME` 要与起服时一致，否则 400/404）。

---

## 4. ★ 与"精度问题"直接相关的其它已知坑（别再踩）

| # | 坑 | 判据 / 处置 | 出处 |
|---|---|---|---|
| 1 | **`ENGRAM_DEVICE_INDEX=1` 会绕过整条 host 路径**（`075/077` 的 pageless、`TRUE_TOKENS`、`mismatch` 全不跑）⇒ "四轴通过"可能是**假通过** | `grep -ac 'DEVICE-INDEX' <serve.log>`：**1 = device 路径（要避免）**、0 = host 路径（A2 生产口径要的） | `a2/logs/085` |
| 2 | **`DRAFT_GRAPH=1` 静默失效**：`A≈1.06`（draft 完全不产出）但 `ms/step` 反而更好看（真实吞吐 −2.2×） | 判据必须 **(A, tok/s) 这一对**；用 `bash tools/draft_graph_guard.sh` 验 **A ≥ 1.3**。**main 默认 `DRAFT_GRAPH=0`** | `serve_a2.sh:200-223`、`reports/draft-graph-negative-control.md` |
| 3 | **int8 档 C 的容量指纹** `B/C=427,643`、`D=485,610` —— B 与 C **相同** ⇒ 容量**区分不了 B/C** | 必须查 `inner.sh` 的 `VLLM_V41_KV8_SWA=` 是不是 1 | `a2/logs/048` |
| 4 | **`run_test.sh` 会把 `BAT_TOKENS` 覆盖**（已修 `7e902a1`） | 用 `serve_a2.sh` 起服；或跑 `run_test.sh` 后核 `env.txt` 的 `bat_tokens_actual` | 本文档 §2 |
| 5 | **A2 = 910B3 走 PCIe ⇒ `host_mem_pool=0`** ⇒ Engram device-index **必须关**（`auto` 会正确回退；强制 `=1` 会在 ~17 min 后 `ret=207001`） | `cat /proc/svm/dev0/feature/host_mem_pool`（A2 预期 0；A3=1） | `CHANGELOG.md` §3、`upstream-v41/logs/55` |
| 6 | **起服日志停在 `[DEVICE-INDEX] 能力探测通过` 很久** = 三种不同的事 | ① 容器**没了** + dmesg OOM ⇒ SIGKILL（解法：起服前清 page cache）；② **A2 上 ~17 min 后 `207001`** ⇒ host_mem_pool 判据没生效；③ 只是**正常慢**（Engram 206 GiB 表 ~10+ min，实测 803 s / 9m45s） | `reports/draft-graph-investigation`、`a2/docs/A2-ENGRAM-PATHS.md:248` |

---

## 5. 我（上一任 agent）对这轮 A2 报告的**判断（未验证）**

**【推断·中】** A2 上"`OFFLOAD=0 + KV8_SWA=1` 仍坏"最可能是**两条的叠加**：
1. **若那台不是 8192** ⇒ 就是 §2 的坑（520K → 254 刀 ⇒ ≈0%）；**§1.2 的负控已证明**这个配置**必然**坏。
2. **若已是 8192** ⇒ 520K = **64 刀**，按 `1−0.98^64 ≈ 27%` 通过率 ⇒ **仍会坏，只是概率低**。
   ⇒ 即 **`BAT=8192` 只是"修好到 260K"，520K 从来没被验证过**（报告的实测最大档是 259,985 = 32 刀）。

★ 因此**下一个要做的实验已经有了**：**`BAT=16384 @ 520K`**（chunk 64 → 32）。
* 若 16384 明显变好 ⇒ 说明仍在 chunk 累加线上 ⇒ 路线是**继续加大 BAT**（代价：activation 峰值更高、KV 更少；`BAT=8192` 时 peak 0.79→3.21 GiB）。
* 若 16384 **也一样坏** ⇒ 那就**不是 chunk 数** ⇒ 转查 **`KV8_SWA`（int8 档 C）在 8 卡 + 520K 下的数值路径**，以及 **A2 特有项**（910B3 / PCIe / 无 host_mem_pool）。

**A3 上能不能替 A2 做这个实验**：A3 已有 1M 服务（`v41-flat-verify3`，`BAT=8192`）；但**A3 与 A2 的差异是结构性的**（910B3 vs 910C、有无 host_mem_pool、KV 页几何）⇒ **A3 的结论只能作旁证**，最终必须在 A2 上验。

---

## 6. 本轮已交付的其它东西（可能对 A2/A3 都有用）

| 交付物 | 位置 | 用途 |
|---|---|---|
| **A3 TP8 工作层镜像补丁包** | COS `share/dsv41-a3-tp8-imagekit-v1.tar.zst`（2.73 MB，sha256 `fbd4bdcc…`）；links-server 已加条目 | 官方基镜像（18 层）+ **1 个工作层（0.2 MiB，19 文件）**；`bash rebuild.sh` 验收：**305,702 条目录项与原镜像逐条相同 + 19 文件逐字节相同**；不需要 skopeo/联网/NPU |
| `tools/make_image_patch_kit.py` 等 | main（`tools/`） | 造层补丁的完整工具链（含 `v41` 自检档、`payload-md5.txt`） |
| `tools/ctx_agent_probe.py` | main | **8 模式**长上下文×Agent 探针（`needle/grow/reuse/toolargs/evict/conc/bigprefill/biggrow/mixed/stream`）+ `--selfcheck` + **上下文长度门** |
| `docs/CTX-AGENT-REPRO.md` | main | **A2 粘贴即用的复现手册**（含单变量切法） |
| `tools/attach_test.sh` | main | 附着精度/性能测试（服务已在跑时用；**必覆盖 PORT/NAME/SERVED_NAME**） |
| `a2/scripts/collect_evidence.sh` | feat | 一条命令收 `EVIDENCE.txt`（支持 `SERVE_LOG=`/`PLAT=a3`；含**答案原文 + 乱码指纹**） |

---

## 7. 红线与纪律（继承，必须遵守）

1. **A2 是生产机**：先 `DRY=1`；ssh **不可达 ⇒ 命令给用户粘贴**。
2. **A3 用 Phy-ID 8–15**（0–7 不是我们的）；**共用机 ⇒ `DROPCACHE=0`**（清 page cache 会打到别人）。
3. **A3 上 `CPU_BIND=0` 是必需逃生口**：目标 NUMA 节点被占满时 `migratepages` 会 100% CPU 无限自旋、服务永不就绪、连 `docker stop` 都拿不到 exit event（解法 `sudo pkill -9 -x migratepages`）。
4. **判据必须绑"实际生效后的可观测痕迹"**，不能绑"我传了这个变量"（本轮 §2 就是这个坑的第 N 次）。
5. **判据本身也要被检验**（`--selfcheck` / 负控 / 反例臂）—— 本会话我自己的探针被查出 3 处缺陷（语料短于目标切出空串⇒拿 91 token 冒充 520k；"问 B 却插 A 的针"；要求必须走 `tool_calls` 把内容正确的回答判 FAIL）。
6. 结论标 **【实测】/【推断】/【未确认】**；不用 `/tmp` 存产物。

---

## 8. 未完成 / 待裁决

| # | 项 | 状态 |
|---|---|---|
| 1 | **A2 `OFFLOAD=0 + KV8_SWA=1` 的精度问题** | ★ **本文档主任务**，先按 §3.2 核 8192，再按 §3.3 切 |
| 2 | **`BAT=16384 @ 520K`** 是否有效 | **未做**（A3 可先跑作旁证；A2 最终要验） |
| 3 | `attach_test.sh` 的 **KV 门槛 3Mi vs 默认 2.82M** | **已知不一致，未修**（会让每次默认配置的附着测试报假红） |
| 4 | `serve_a2.sh` 文件头第 11 行仍写"默认开 `DRAFT_GRAPH=1`" | 与代码（`:-0`）**自相矛盾**，**未修**（会误导判读） |
| 5 | A3 上我起的 `BAT=8192` 服务 | **仍在跑**（health=200）；若要复用就直接测，否则 `docker stop`（先 `sudo pkill -9 -x migratepages`） |
| 6 | 520K @ 8192 的**完整**正控 | A3 上已跑过 `bigprefill/biggrow/mixed/stream` 全 PASS，但**A2 未跑** |
