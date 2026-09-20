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

## 2. 端口卫生

**别的租户会占你选的端口。** 实测 8020 上跑过**别人的 sglang**
（`/home/models/DeepSeek-V4-Flash-W8A8`）；只 `curl /health` 会拿到 **200 但不是自己的服务**
⇒ 后续所有轮询都变成在测别人的机器。

**纪律**：判活必须用 `/tmp/mysvc.sh <port>` 并断言它返回 **`MINE`**
（该脚本会打 `/v1/models` 校验 `"id"` 是不是我们自己的模型名）。

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

---

## 5. 环境约定

| 项 | 约定 |
|---|---|
| 默认芯片 | **A2**：`DEVS="0 1 2 3 4 5 6 7"`（单机 8 卡）｜**A3**：`DEVS` **必填**，脚本不替你选 |
| 默认口径 | A3 开 Engram 算子入图（`ENGRAM_DEVICE_INDEX` 自动）｜**A2 关**（`=0` 走 host 路径）—— A2 上整表 host_register 会 `ret=207001`，见 `CHANGELOG.md` v8 §3 |
| 性能口径 | `MAX_SEQS=4 PREFIX=0` |
| 生产口径 | `MAX_SEQS=32 PREFIX=1` |
| 长 prompt 首 token 的命门 | `GPU_UTIL`：**0.92 是默认**，0.94 会让 8K prefill 从 1.14 s 掉到 8.0–8.6 s（`docs/prefill-memory-headroom.md`） |
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
