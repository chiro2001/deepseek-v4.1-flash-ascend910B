# AGENTS.md —— 自动化 Agent 工作手册

> **给人的内容在 [`README.md`](README.md)**；本文件只放"Agent 才需要"的东西：
> 口径纪律、必踩陷阱、调试入口。**动手前请把 §1 与 §3 读完**——
> 这两节里的每一条都对应一次真实的翻车记录。

---

## 1. ★ 口径纪律

### 1.1 报性能必须给全 `(ms/step, A, tok/s)` 三元组

**只报 `ms/step` 会得出与事实相反的结论。** 依据：`DRAFT_GRAPH=1` 的实测对比
（`CHANGELOG.md` §6）——`ms/step` 从 27–30 降到 25.1（"更好看"），
但 `A` 从 2.7–3.0 掉到 1.06、单流从 90–111 掉到 43 tok/s，**真实吞吐慢 2.2×**。

换算关系（三种口径互相校验，**必须自洽**）：

```
ms/step = 1000 × A ÷ 单流 tok/s
```

| 口径 | 定义 | 典型出处 |
|---|---|---|
| **`[bneck] hp`** | **设备侧**完整 decode step（`mark_step()` 累计相邻间隔） | `docker logs` |
| **quote** | **客户端**墙钟 `decode_s ÷ steps` | `tools/p42_t4_quote.sh` |
| `total`（`[bneck]`） | 仅 Engram **host 路径**耗时（`d2h+hash+route+pad+meta`） | —— **不是** step 时间 |

### 1.2 `A`（接受长度）**不能单独当绩效指标**

`A` 与文本质量**反相关**：模型掉进复读循环时 `A` 会虚高到 > 3.5，吞吐被**虚高 1.5×以上**
（见 `docs/BENCH-METHODOLOGY.md`）。要报就用 `(clean-rate, ms/step)` 或 `(A, tok/s)` 组合。

### 1.3 `bench_concurrency.py` 的样本数 —— 最容易踩的一条

**prompt 条数 = `--concurrency` 列表里的最大值**，且**每一档都跑全部 prompt**（按 conc 切批，批间等空闲）。

| 你传的 | 实际准备 | `conc=1` 档发几条 |
|---|---:|---:|
| `--concurrency 1` | **1 条** | **1 条** ⚠️ |
| `--concurrency 1,2,4,8` | 8 条 | **8 条**（串行 8 批） |
| `--concurrency 1,...,64` | 64 条 | 64 条 |

**⚠️ 单发不可与 64 条中位比。** `A` 是**发放级抽签**（历史 163 发：
steep 15% / flat 7% / shallow 77%）。实测同一条臂：

```
单发       A=2.04 / 67.0 tok/s   → 判据 FAIL
8 发中位   A=2.65 / 92.0 tok/s   → 判据 PASS
```

⇒ `tools/draft_graph_guard.sh` 与 `tools/draft_arm_probe.sh` 因此**固定**用
`--concurrency 1,2,4,8` 并取 `conc=1` 行（那才是"8 发中位"）。

**另：`per_stream_med` 与 `total_decode` 不可混引。**
前者 = 每请求自身 decode 速率的中位（回答"我自己发一条多快"）；
后者 = 全部输出 token ÷ decode 窗口（回答"服务整体每秒吐多少"）。

### 1.4 跨机 / 跨会话数字不可直接相减

- **A2（910B3）与 A3（910C）**：设备侧与 CPU 都不同，只能比**趋势**，不能当 A/B。
- **同机不同进程**也只是"跨会话对比"。要严格 A/B 就在**同进程**内热切换
  `/tmp/v41_dspark_flags`（例：写 `DRAFT_FORCE_EAGER=1`），同批请求、只变一个变量。

---

### 1.5 CED-PD（A3 单机 8+8 PD 分离）—— **另一套形态，口径别混**

仓库里现在有**两种部署形态**，数字不可互相引用：

| | 单实例 TP8（8 卡） | **CED-PD（16 卡）** |
|---|---|---|
| 入口 | `scripts/serve_a3.sh` | **`deploy/a3-ced-pd/`** |
| P | — | chip 0–7，只跑 layer 0–19 + layer-20 全局源 |
| D | — | chip 8–15，128-token 有界重放 + 全 40 层 + DSpark |
| KV | 可选 int8 | **BF16** |

**判乱码先分三族**（指纹不同、判据也不同；混着读会得出互相矛盾的结论）：

| 族 | 指纹 | 看什么 |
|---|---|---|
| **A** 长上下文静默空答 | HTTP 200 + `completion_tokens=1` + `content=null` | `completion_tokens` |
| **B** 图模式乱码 | HTTP 200 + **打满** `max_tokens` + **无** `finish_reason` + 含 `<｜box｜>` | 文本 |
| **C** 草稿静默失效 | 文本**是对的**，但 `A≈1.0` | `Mean acceptance length` |

* ⚠️ `u_fffd`（U+FFFD）**恒为 0** ⇒ 它从来不是判据（问题不在 tokenizer/解码层）。
* ⚠️ **看 `ms/step` 发现不了 A 和 C** —— C 族失效时 ms 反而**更好看**
  （每步只出 1.08 个 token 而不是 2.85）。
* ⚠️ **DSpark 的收益只在低并发成立**：并发 4 时它把 decode ms/step 从 28.2 抬到 41.1
  （**1.45×**），换来 A≈2.4 —— 收益集中在接受长度高的请求上，**成本由全批承担**。
  高并发吞吐场景应保持 `SPEC=0`。

完整措施总账（每条带证据强度：已验证 / 机理对但无效 / 未验证）见
[`docs/CED-PD-ACCURACY-MEASURES-20260926.md`](docs/CED-PD-ACCURACY-MEASURES-20260926.md)。

---

## 2. 端口卫生

**别的租户会占你选的端口。** 实测 8020 上跑过**别人的 sglang**
（`/home/models/DeepSeek-V4-Flash-W8A8`）；只 `curl /health` 会拿到 **200 但不是自己的服务**
⇒ 后续所有轮询都变成在测别人的机器。

**纪律**：判活必须用 `/tmp/mysvc.sh <port>` 并断言它返回 **`MINE`**
（该脚本会打 `/v1/models` 校验 `"id"` 是不是我们自己的模型名）。

**CED-PD 形态用另一组端口**（别和 18550/18551 那组混）：
`P=18990` / `P-KV=19090`、`D=18991` / `D-KV=19091`、`proxy=18992`。
默认值在 `deploy/a3-ced-pd/launch/_common.sh`，都可用 env 覆盖。

```bash
code=$(curl -s -m 5 -o /dev/null -w "%{http_code}" http://127.0.0.1:8020/v1/models)
own=$(/tmp/mysvc.sh 8020 | head -1)
[ "$code" = "200" ] && echo "$own" | grep -q MINE || echo "还没起来 / 不是自己的"
```

---

## 3. ★ 陷阱清单（每条都真踩过）

### 3.1 起服脚本里，**注释不能插进 `docker run` 的续行链**

`serve_a2.sh` 的 `docker run` 是一长串 `\` 续行。**在链中间插一行 `#` 注释会终止续行**
⇒ `docker run` 丢掉 IMAGE 参数 ⇒

```
"docker run" requires at least 1 argument.
serve_a2.sh: line 1066: -e: command not found
```

**`bash -n` 抓不到这种错**（拼接后语法合法）。守卫：`tools/check_serve_run_chain.py`
（已接入 `tools/selfcheck_pkg.sh`）。**改完起服脚本，跑一次 `selfcheck`。**

### 3.2 `grep -c ... || echo 0` 会拼出两行的 `"0\n0"`

`grep -c` 在**无匹配**时打印 `0` **且退出码 1** ⇒ `$(grep -c ... || echo 0)` 拿到 `"0\n0"`
⇒ `"0\n0" != "0"` 成立 ⇒ 把正常结果判成异常。
A2 真机上表现为：同一份日志 `serve_a2.sh` 判 ✓、`run_test.sh` 判 ✗
（报"命中 `0\n0` 次"）。**正确写法**：`$(grep -c ... 2>/dev/null || true)` + `${VAR:-0}`。
守卫：`tools/negative_control.sh` 的 **NC15**。

### 3.3 bind-mount 绑的是 **inode**，不是路径

**起容器前确认 md5**：容器跑起来后再改宿主文件（尤其 `docker cp`/重写而非原地改）
不会进到容器里，于是"我明明改了但没生效"。

同理，Engram 表的 `O_RDWR` 是按**最终 inode 所在目录**判定的：
`engram_int8/` 里的条目本身可能是**软链**，真正落盘的目录在**另一棵树**里
（链条：`$MODEL/engram_int8 → L4 → L3（实体）→ 4 个软链 → …/engram-int8`）。
只把 L5 挂成 `:rw` **不管用** —— `serve_a2.sh` 会自动把整条链都挂上。

### 3.4 `DRAFT_GRAPH=1` 必须同时有 `DSPARK_GRAPH_CAPTURE_METADATA=1`

缺它 ⇒ draft 图**静默失效**：`A 恒 1.00`、**没有任何报错**、`ms/step` 还看着正常
（`reports/draft-graph-negative-control.md`）。`serve_a2.sh` 把两者绑在一起设，
并在起服后跑 **DRAFT-GUARD** 校验容器内值；组合不对会直接 `die`。

**判据**：连读两次 specdec metrics，若 `Mean acceptance length: 1.00`
**且** `Accepted throughput: 0.00` ⇒ 已进坏状态 ⇒ **重启服务**（别发请求试探）。

### 3.5 探针不能进 capture 区间

capture 期做 D2H（`.tolist()` / `.sum()` / `.item()`）会让图捕获失败：
`Worker proc VllmWorker-N died unexpectedly` + `RuntimeError: cancelled`，
且 capture 期日志量（每层 × 8 rank）本身会把 worker 拖垮。
⇒ 所有探针都要 `if get_forward_context().capturing: return`。

**唯一可靠的观测窗口是 `_propose`（每步都跑、在图之外）**；但探针预算要按"真实 decode"
过滤，否则会被 profile run 吃光。

### 3.6 MANIFEST 必须**从 HEAD 生成**

`tools/make_manifest.sh` 读的是**工作区**内容。只提交一部分文件时，它会把手改但未提交的
哈希写进 `MANIFEST.sha256` ⇒ 干净 clone / `git archive HEAD` 拿到的是 HEAD 内容 ⇒ **不匹配**。
⇒ 要么先提交全部改动，要么用 `git ls-tree -r --name-only HEAD` + `git cat-file blob HEAD:<f>`
逐条算（脚本里已附命令）。跑完 `tools/selfcheck_pkg.sh` 复核。

### 3.7 `LOAD_FORMAT=dummy` 的 env 是**一组**，不能只抄一半

| env | 漏掉的后果 |
|---|---|
| `V41_ENGRAM_WITH_DUMMY=1` | `[bneck]` 相位无数据 |
| **`V41_DUMMY_WO_A_FIX=1`** | **`_forward_o_proj` 维度崩溃**（dummy 不调 `weight_loader`，`wo_a.weight` 停在 2D） |

### 3.8 条件挂载块不能引用**尚未初始化**的变量

`set -u` 下会 `unbound variable` 且**容器从未出现**。改起服脚本时，把新条件块放在
`MOUNTS=` 初始化**之后**（`reports/mount-order-fix.md`）。

### 3.9 `--out` 会被 argparse 当成 `--output-tokens`（缩写匹配）

`tools/bench_concurrency.py` 的输出参数叫 **`--json-out`**。若写 `--out x.json`，
argparse 的**前缀匹配**（`allow_abbrev` 默认开）会把它解析成 `--output-tokens x.json` ⇒

```
bench_concurrency.py: error: argument --output-tokens: invalid int value: 'x.json'
```

⇒ **测量一行没跑就退出**。2026-09-20 在 A3 上真踩过：白等 21 分钟起服（`MAX_SEQS=64`
冷编译 static kernel），结果 bench 秒退。**跑完先确认 `results/bench/*.json` 真生成了。**

---

### 3.10 改了 `patches/files/*` 必须同步 md5 清单 —— 否则**构建最后一步**才炸

三处清单是**对载荷的冻结快照**：`patches/MD5SUMS`、`patches/vllm-ascend/MD5SUMS`、
`patches/PATCHES.md`。改了载荷不同步 ⇒ `bash scripts/build_image.sh` 会跑完
10–20 分钟、在**最后一步**报：

```
FAIL md5 models/deepseek_v41/model.py: got=fea1f31c… want=6bd61e15…
```

即「烘进镜像的字节是对的，清单是陈旧的」。

**不要手改**，跑生成器（它同时刷三处 + MANIFEST）：

```bash
python3 tools/refresh_checksums.py      # 然后再跑 selfcheck
python3 tools/check_checksums.py        # 期望：三方一致 ✅
```

本仓已踩**三次**（v7→v8、v8 新增两文件、CED 的 `61d238c` 加
`V41_CED_ALLOW_DSPARK` 时漏更）。`tools/selfcheck_pkg.sh` 的
「校验和一致性」那一项会把它变成 FAIL。

### 3.11 判静态核**不要**看 `compile start` —— 编译缓存命中时它是 0

`STATIC_KERNEL=1` 本身**可能静默失效**（多卡但没 `LOCAL_WORLD_SIZE`）。
两条判据都要过：

| # | 判据 | 期望 |
|---|---|---|
| ① | `grep -ac "static_kernel.py:650" serve.log` | **0** |
| ② | `grep -ac "static shape kernel will be used" serve.log` | **> 0** |

⚠️ **`static kernel compile start` 不是判据**：第二次起服会命中
`cache/skcache/compile_outputs/static_kernel_cache` ⇒ 它是 **0**，
但静态核**照样在跑**（本仓实测：首次 4 次 / 再次 0 次，两臂都生效）。
用它当判据会给**假阴性**，把一个好配置判成坏配置。

一键检查：`bash experiments/dspark/check_static_kernel.sh <run_dir>`

---

## 4. 调试入口

| 目的 | 命令 / 开关 |
|---|---|
| 自检包一致性（10 s，改完起服脚本必跑） | `bash tools/selfcheck_pkg.sh` |
| 负控（证明每个自检真能抓到它对应的 bug） | `bash tools/negative_control.sh` → 期望 `PASS=25 FAIL=0` |
| 起服前自检（不起容器） | `MODEL=... bash tools/preflight_a2.sh` |
| 干跑看将执行的命令 | `DRY_RUN=1 ... bash scripts/serve_a3.sh` |
| 服务已在跑时附着自检（**不删容器**） | `PORT=8020 bash tools/attach_test.sh` |
| 设备侧 step 时间（`hp` 就是 ms/step） | `docker logs <name> 2>&1 \| grep -E "\[bneck\]\|\[route-probe\]"` |
| 稳态接受长度 | `docker logs <name> 2>&1 \| grep "SpecDecoding metrics" \| tail -5` |
| 同进程热切换（严格 A/B） | 往 `/tmp/v41_dspark_flags` 写 `DRAFT_FORCE_EAGER=1` |
| 走图取证 | `DSPARK_DISPATCH_UNIQUE=1`（默认关） |
| 端口归属断言 | `/tmp/mysvc.sh <port>` → 必须 `MINE` |
| **起 CED-PD 三件套**（`MODEL=` 必填） | `bash deploy/a3-ced-pd/launch/serve_{p,d,proxy}.sh` |
| **CED-PD 冒烟**（144K 四针，最小正确性判据） | `bash deploy/a3-ced-pd/launch/smoke.sh` |
| **判"镜像 vs 仓库"逐文件一致** | `bash deploy/a3-ced-pd/verify_consistency.sh`（`--mode payload` 也能比） |
| 造 CED-PD 工作镜像（官方基底 + 我们的 1 层） | `bash deploy/a3-ced-pd/build_image.sh` |
| CED-PD 的 **ms/step**（用 `HcPre` 定步） | `python3 experiments/dspark/step_period.py <kernel_details.csv> 86` |
| CED-PD 逐算子 A/B（归一到 ms/step） | `python3 tools/ced_prof_ab.py a.csv:80:1.9 b.csv:86:1.9:4.55` |
| 静态核真生效？（判据见 §3.11） | `bash experiments/dspark/check_static_kernel.sh <run_dir>` |

---

## 5. 环境约定

| 项 | 约定 |
|---|---|
| 默认芯片 | **A2**：`DEVS="0 1 2 3 4 5 6 7"`（单机 8 卡）｜**A3**：`DEVS` **必填**，脚本不替你选 |
| 默认口径 | A3 开 Engram 算子入图（`ENGRAM_DEVICE_INDEX` 自动）｜**A2 关**（`=0` 走 host 路径）—— A2 上整表 host_register 会 `ret=207001`，见 `CHANGELOG.md` v8 §3 |
| 性能口径 | `MAX_SEQS=4 PREFIX=0` |
| 生产口径 | `MAX_SEQS=32 PREFIX=1` |
| 长 prompt 首 token 的命门 | `GPU_UTIL`：**0.92 是默认**，0.94 会让 8K prefill 从 1.14 s 掉到 8.0–8.6 s（`docs/prefill-memory-headroom.md`） |
| **CED-PD 形态的卡分配** | **P = chip 0–7 / D = chip 8–15**（16 卡全用）；全部实测参数在 `deploy/a3-ced-pd/launch/_common.sh` |
| **CED-PD 的 D 必须关多流** | `MULTISTREAM=0 DSA_OVERLAP=0` —— 开了长上下文**静默算错**（实测 144K 四针 0/4） |
| **CED-PD 的 D 池有硬上界** | `num_blocks ≤ ⌊2³²/147712⌋ = **29076**`（按**页尾**取界）；越界 ⇒ 1M 静默空答（族 A） |
| 大文件传输 | 走 `scripts/cos-put.sh`（**在 dsv41 仓根，不在 dsv41-release 里**）；**禁止 ssh 传 ≥1 MB** |
| 发布纪律 | 发布仓远端 = `chiro2001/deepseek-v4.1-flash-ascend910B`；ModelScope 的 README 源在 `../ms_readme.md`（用 `tools/repack_to_modelscope_layout.sh` 打包） |
| 时区/时钟 | A3-node1 与本机**可能有几分钟时钟差**，判断"卡住"前先 `date` 对齐 |

---

## 6. 已知的"看起来像 bug 但不是"

| 现象 | 真相 |
|---|---|
| `[route-probe]` 里 **TP0/rank0 的 `scatter` 异常大**（0.8–1.4 vs 0.06） | **rank0 角色代价，预期行为**（A3 同样） |
| `[bneck]` 的 `d2h` 在 rank 间差 20×（0.19 vs 3.4） | 同样是 host 路径的 rank 差异，**不是故障** |
| 起服耗时从 377 s 涨到 800 s | 冷编译 static kernel（`MAX_SEQS=64` 会多捕获 192/384 两个大桶，约 9–10 min）；命中缓存后回到分钟级 |
| 8K 档的 `A` 明显低于 32K/稳态 | 8K 是 `run_test.sh` 的**第一条真请求**（冷启动首请求，`A≈1.75`）；看 APIServer 稳态 `SpecDecoding metrics` 才准 |
| **CED-PD 的 step 时间从 45ms 掉到 33ms** | **不是变快，是批变小了**（4 路→1 路，M 从 32 → 8）。用 `HcPre` 定步看时序就明白：前 84 步 45ms、后段 33.5ms |
| **CED-PD 的 `A≈1.0`**（看着像"没开推测解码"） | 草稿图**静默无 attention** —— 缺 `DSPARK_GRAPH_CAPTURE_METADATA=1`（见 §3.4）。`ms/step` 此时反而更好看 |
| CED-PD 的 `u_fffd` 全是 0，但文本明显是乱的 | **预期**：乱码来自 token 层面（族 B/C），不是 UTF-8 解码问题 ⇒ `u_fffd` 从来不是判据 |
| CED-PD 单流 `tok/s` 与历史差 14–21% | **语料不同**：`A` 是内容驱动的（同样配置下《地火》A=2.76 vs 《红楼梦》A=2.32，而 `ms/step` 几乎一样）。跨语料比 `tok/s` 没有意义 |
