# CHANGELOG.md —— v3 → v4 → v5 → v6 逐项 diff

---

# ★ v7（2026-09-18）—— 长上下文精度修复：`BAT_TOKENS` 2048 → 8192

> **这是本包第一次修"正确性"而不是"性能"或"工程"。**
> 改动只有一行默认值（`scripts/serve_a2.sh`），但影响很大。

## 0. 症状

长上下文（≳60K token）的 agent 场景里，模型会**输出复读/幻觉、不再调用工具**，
最终被解析器整段丢弃成"空回复"。此前被当作"偶发、不可复现"。

## 1. 根因

**chunked prefill 的 chunk 数决定偏离率。**

chunked prefill 把长 prompt 切成 `ceil(prompt / BAT_TOKENS)` 段依次前向，
每段都有一次独立的"偏离"机会，误差沿后续 chunk 累积。实测约 **2%/chunk**。

因此 `BAT_TOKENS=2048` 时：

| prompt_tokens | chunk 数 | 正确率 |
|---:|---:|---:|
| 20,318 | 10 | 80% |
| 40,163 | 20 | 70% |
| 60,012 | 30 | 50% |
| 79,855 | 40 | 30% |

**平滑下滑**——这解释了为什么此前找不到"阈值"：它本来就没有阈值。

### 1.1 怎么证明是"我们的栈"而不是模型

同一批 prompt（逐字节相同）发给官方 API：**12/12 全过（含 252K token）**；
我们的服务在同长度上 ~0%。⇒ 模型有能力，是我们的栈弄坏的。

## 2. 修复

```diff
-BAT_TOKENS=${BAT_TOKENS:-2048}
+BAT_TOKENS=${BAT_TOKENS:-8192}
```

### 2.1 效果

| prompt_tokens | `BAT=2048` | `BAT=8192` |
|---:|---:|---:|
| 10,394 | 10/10 | **10/10** |
| 20,318 | 8/10 | **10/10** |
| 40,163 | 7/10 | **10/10** |
| 60,012 | 5/10 | **10/10** |
| 79,855 | 3/10 | **10/10** |
| 149,986 | ~0% | **6/6** |
| 259,985 | ~0% | **6/6** |

同一 prompt 重复 10 次 → 10/10，输出 token 数逐次一致。

### 2.2 代价

| 项 | `BAT=2048` | `BAT=8192` |
|---|---:|---:|
| KV cache（8×910C, util=0.94） | 4,145,957 tok | 3,088,738 tok |
| host 侧每 chunk 耗时 | ~34 ms / 2048 tok | ~191 ms / 8064 tok |

KV 容量下降约 25%。更看重 KV 容量且上下文主要在 <20K 的场景可显式 `BAT_TOKENS=2048`。

## 3. 新增

| 文件 | 说明 |
|---|---|
| `tests/agent_trace/longctx_retrieval.py` | **长上下文检索探针** —— 把唯一事实埋在长文档中段、只问一个答案唯一的问题。60K token 就能测出退化（比"工具调用测试"灵敏得多）。报 Wilson 95% CI、支持多 nonce。 |
| `tests/agent_trace/accuracy_gate.py` | agent 形态的工具调用精度门（5 个长度档、截断单列、失败签名分类）。 |
| `reports/longctx-accuracy-fix.md` | 完整分析：曲线、消融、机制解释、方法论教训。 |
| `reports/probe/` | 稀疏状态插针（事后取证工具）+ 设计文档，`PROBE=1` 启用。 |

## 4. 同时修掉的工程问题

| # | 问题 | 修复 |
|---|---|---|
| 1 | `inner.sh` 把 `FUSED_MC2` / `MC2` / `MC2_HIER` / `REDUCE_SAMPLE` / `DSA_OVERLAP` **硬编码**，外部 env 传不进去 | 全部参数化（默认值与原来一致，行为不变） |
| 2 | 起服依赖镜像里烘焙的 `/opt/dsv41/scripts/serve_v2.sh`，换基础镜像就起不来 | 改为只读挂载本包的 `scripts/` |
| 3 | 排查用的 dev mode / 请求日志 / 插针无法从外部开关 | 新增 `VLLM_SERVER_DEV_MODE` / `LOG_REQUESTS` / `PROBE`，**默认全关** |

## 5. 方法论教训（写进仓库，避免重犯）

定位过程中我制造了 **5 个假阳性**（工具数量、插针、投机解码、"14 个 token 决定成败"、
Engram/QLI 是主因），全部源于**跨会话比较**——服务会随时间自发退化（旧会话 ~12%、
新鲜会话 ~56%），拿旧基线比新鲜消融必然得出假阳性。

**三条纪律**：

1. 任何消融必须配**同等新鲜度**的基线。
2. 同一剂量点用**多个不同样本**（同一长度下"内容"决定成败，单样本毫无代表性）。
3. 判据必须看 `finish_reason` —— `finish=length` 的截断样本既非成功也非失败，
   混进失败率会得出错误结论。

## 6. 完整验收（2026-09-18）

| 验收项 | 要求 | 实测 | 判定 |
|---|---|---|---|
| 前后对比（N≥10） | 修复前失败率显著、修复后 0 | 43/50 → **62/62** | ✅ |
| 8K / 32K / 128K / 256K | 全覆盖 | 8.4K / 32.2K / 130.5K / 260.0K 全 10/10（256K 为 6/6） | ✅ |
| 真实 agent 轨迹 | 覆盖 | 两条真实会话轨迹各 **10/10** | ✅ |
| Vision | 23/23 | **23/23** | ✅ |
| GSM8K-200 | ≈198/200 | **199/200** | ✅ |
| `static_kernel` 降级 | 0 | **0** | ✅ |
| Engram-int8 常驻 | 必须 | `engram_storage=int8` + host-resident | ✅ |
| KV > 3Mi | 交付约束 | **3,088,303（低 1.8%）** | ⚠️ 见下 |

### 6.1 新增的显存约束（`[MEM-GUARD]`）

`BAT=8192` 让 peak activation 从 0.79 → 3.21 GiB。与 `MAX_SEQS=64`
（capture 桶到 384）叠加时，`GPU_UTIL=0.94` 会在 **ACL graph 重放时 OOM**：

```
torch.OutOfMemoryError: NPUGraph.cpp:281
Resource_Error_Insufficient_Device_Memory(EL0019)
```

`scripts/serve_a2.sh` 已加提示（危险组合时打印建议，不擅自改配置）：

| 组合 | 结果 |
|---|---|
| `MAX_SEQS=32` + `BAT=8192` + `0.94`（发布默认） | ✅ |
| `MAX_SEQS=64` + `BAT=8192` + `0.90` | ✅（KV 降到 2.56M） |
| `MAX_SEQS=64` + `BAT=8192` + `0.94` | ❌ OOM |

### 6.2 已知代价

KV cache 4,145,957 → **3,088,303** tokens，低于 3Mi 交付约束约 1.8%。
若必须同时满足 KV>3Mi，可评估 `BAT=4096`（尚未验证是否足够）。

完整记录见 [`reports/longctx-verification.md`](reports/longctx-verification.md)。

---

# ★ v6（2026-09-17）—— 交付工程修复 + 自检体系

> **v6 不改任何性能配置**：`patches/`、`Dockerfile`、`optim/` 与 v5 **逐字节相同**
> （已用 `diff -rq` 验证）。所以 **v5 的镜像可以直接 retag 复用，不必重新构建**：
> `docker tag dsv41-a2:v5 dsv41-a2:v6`
>
> v6 修的是"**能不能跑通、能不能复现**"。性能差距（A2 75.7 vs A3 28.7 ms/step）
> 是另一条线，v6 不改善它 —— 见 `EXPECTED_PERF.md` §A2GAP。

## 0. 为什么要有 v6（v5 的教训）

v5 在 A2 上暴露了 **10 个 bug**，其中 **8 个是我们自己的脚本/打包错误**。
共同点：**都不会在开发机上暴露**（因为开发机上有那些文件、有那些变量），
而**每个都要跑到最后一步才知道**，起服一次 5–25 分钟 ⇒ 代价被放大十几倍。

因此 v6 的核心不是加功能，而是**把检查前移 + 证明检查有效**：

| 新机制 | 解决什么 |
|---|---|
| `tools/preflight_a2.sh`（**30 秒 / 9 组 / 不起容器**） | "白等 25 分钟才发现" |
| `tools/negative_control.sh`（**8 个负控**） | "自检全过但包是坏的"（v5 真实发生） |
| `build_scripts/00_ensure_pgo.sh`（指纹缓存） | PGO 只编译一次；换机器自动重编 |

## 1. 修的 10 个 bug

### 1.1 缺文件（两个，直接让测试全废）

| # | 症状 | 根因 | 修法 |
|---|---|---|---|
| **1** | `can't open file '.../tests/p15_stream_curve_filefiller.py'` ⇒ **8K/32K/128K 性能测试全废** | 打包时漏收该文件（它在 A3-node1 的 `logs/perf/` 下） | 收进 `tests/`；md5 `ba75b25e3b0a2eb8dd1436627d4c2126` |
| **2** | `[t_vision] {'cases': None, ..., 'verdict': 'FAIL'}` ⇒ **视觉必 FAIL** | `t_vision.py` 调 `HERE/vision_accuracy_check.py`，但文件在 `tools/` 不在 `tests/` | 复制到 `tests/`；md5 `879dd13d1547d572efd335134a468c3a` |

### 1.2 参数没接上（三个，静默失效）

| # | 症状 | 根因 | 修法 |
|---|---|---|---|
| **3** | 传 `CPUS=-1` 无效，仍绑 24 核 | 变量名是 **`CPUSET`**（与 `MEMS` 不对称），`CPUS` 无人读 | v6 接受 `CPUS` 作为别名 + 冲突时告警 |
| **4** | 传 `CPUSET=...` 无效 | **`run_test.sh` 根本不转发** `CPUSET/MEMS/CPU_BIND` | v6 转发（还补了漏掉的 `MOE_NF`/`CACHE`/`SKCACHE_GC`） |
| **5** | 想换解释器换不了 | `PYHOST=$(choose_py)` **无条件覆盖** | v6 改为 `PYHOST=${PYHOST:-$(choose_py)}` |

### 1.3 失败不复原（一个，代价最大）

| # | 症状 | 根因 | 修法 |
|---|---|---|---|
| **6** | 起服失败后**容器仍活着占 ~313 GB**，导致第二次起服叠加失败 | 容器入口是 `bash -lc "sleep infinity"`，`die()` 没有清理 | v6 在 `die()` 里 `docker rm -f`（日志已落盘，不丢证据） |

### 1.4 环境假设（三个）

| # | 症状 | 根因 | 修法 |
|---|---|---|---|
| **7** | `pin_memory` 报 `207001`（`aclrtMallocHostWithCfg`）而 667 GiB 空闲 | 需要更新 driver **且** 解除 memlock | v6 默认 `--ulimit memlock=-1`（A3 上无副作用） |
| **8** | PGO 可能**静默降级**为 0 | `TARGET_PATH.txt` 是 `build_image.sh` 生成的、不在包里；缺失时只打一行 WARNING | v6 起服时**自动探测并落盘**；preflight 显式报告 |
| **9** | `skcache/ts*_outputs` 无限累积（A3 上 **1491 个目录 / 847 MB**，从不清理） | 脚本只创建不清 | v6 起服前自动 GC，**且显式保留 `static_kernel_cache/`** |
| **10** | （潜伏）`run_test.sh` 缺 `set -e` 导致某些失败被吞 | —— | 保持现状但**显式文档化**：各阶段独立、GSM8K 失败不中止 |

## 2. 新增：`tools/preflight_a2.sh`（30 秒，不起容器）

9 组检查，把已知的 11 类 A2 环境差异 + 包完整性一次问完：

```
1/9 包内文件完整性（逐个断言 t_quote/t_vision/t_gsm8k/acc_eval/p15/vision_accuracy/...）
2/9 Dockerfile 续行链 + 镜像 tag 三处一致（v3/v4/v5 都在这里栽过）
3/9 模型目录：软链链健康 + 跳数分布 + 必需文件 + engram/vision/mtpq 分片
4/9 官方目录：inference/examples/images + encoding
5/9 宿主依赖：PYHOST 选谁 + datasets 版本（**必须 5.0.1**）+ HF 缓存
6/9 CPU/线程/绑核：逻辑核 vs 物理核、OMP_NUM_THREADS、memlock、**NPU↔NUMA 拓扑**
7/9 编译缓存：skcache 键（CANN-<ver>_<SoC>）、ts* 堆积、PGO md5 + TARGET_PATH
8/9 主机资源：MemAvailable（Engram 需 206 GiB 常驻）、磁盘、**残留容器**
9/9 结论：FATAL/WARN 计数
```

**退出码**：0 = 无 FATAL；1 = 有 FATAL。设计成"**先跑它，再决定要不要起服**"。

## 3. 新增：`tools/negative_control.sh`（8 个负控，14 个断言）

**这是 v5 最大的方法论缺口**：v5 有 `selfcheck_pkg.sh`，但它第一次跑就"全过"，
而包其实是坏的 —— 因为**自检只证明了"我检查的东西是好的"，没证明"检查本身有效"**。

v6 对**每个已知 bug 故意造一个坏输入**，断言检查必须报错：

| # | 负控 | 断言 |
|---|---|---|
| NC1 | 删 `tests/p15_...py` | preflight 必须报 FATAL |
| NC2 | 删 `tests/vision_accuracy_check.py` | preflight 必须报 FATAL |
| NC3 | Dockerfile 行内 `#` + 漏 `\` | `check_dockerfile.py` 必须 FAIL，且 `docker build` 也必须 FAIL |
| NC4 | build/serve/run_test 三处 tag 不一致 | preflight 必须报 FATAL |
| NC5 | 软链断链 | `model_mount_args.sh` 必须 rc≠0；**正控**：健康 5 层链要解析出 5 个目录 |
| NC6 | `-v` 与路径粘成单参数 | 组参数必须偶数且交替 `-v`/路径；`docker run` 必须拒绝粘在一起的写法 |
| NC7 | serve 退回单层 `-v $MODEL` | 静态检查必须抓到 |
| NC8 | 假 skcache 含 `ts*` + `static_kernel_cache/` | GC 后**只删 ts\***、**保住缓存** |

**全部在 `mktemp -d` 的临时副本里做，不动真实包。**

## 4. 新增：PGO 编译流程进包（`build_scripts/` + 指纹缓存）

v5 只有**预编译产物**（2 个文件），没有编译能力，且那份 `.so` 是
**在 Ubuntu 容器里编的、跑在 openEuler 上**。

| 文件 | 作用 |
|---|---|
| `build_scripts/00_ensure_pgo.sh` | **新入口**：算指纹 → 一致则**秒退**；否则起一次性容器编译 |
| `build_scripts/01_setup_build_env.sh` | **改为 distro-aware**（Debian→apt / RPM→dnf）；这是 v5 搬到 A2 会直接失败的地方 |
| `02..06` | 沿用（取源码 / configure / make / install / package） |

**指纹** = `cpu_part + gcc + glibc + 源码 sha256 + mtune 开关`：

```
一致  -> 0 秒跳过（"只编译一次"）
不同  -> 自动重编（换机器/换镜像/换 gcc/换源码都不会用错产物）
```

**编译工作区在容器外**：`optim/pgo/build/{src,out,logs}` ⇒ 第二次可增量、产物持久。

### ⚠️ 必须说清的两个限制（避免误期待）

1. **重编本身不会带来明显性能收益**。现有构建参数里**没有 `-march`/`-mtune`**
   （已实证：`grep -oE '\-m(arch|tune|cpu)=' make.log` 为空）⇒ 代码生成是通用
   aarch64，**在哪台机器上编都差不多**。重编的价值是：**glibc/发行版精确匹配** +
   **未来可重建**。
2. **想针对本机核优化只有一条路**：`PGO_MTUNE=1`（加 `-mtune=native`，**安全，
   不生成新指令**）。预期 **0~3%**，且**必须 A/B 实测**。默认**关**。
   `-march=native` 不提供（会 SIGILL）。

### 关于 `PROFILE_TASK`：**保持 CPython 默认**

用户确认它"具有一定代表性"，且有**实测证据支持**：
训练用 CPython 测试套件，而在 `tiny_call` / `dict_loop` / `list_append`
这些**与测试套件毫无关系**的模式上仍拿到 **−16~23%**
（`reports/cpython-pgo-verified.md`）。原因清楚：**PGO 优化的是解释器本身**
（字节码分派、对象分配、dict/type 查表、函数调用），任何 Python 程序都要穿过。
⇒ **v6 不改 PROFILE_TASK。**

## 5. 其它

* 镜像 tag：`dsv41-a2:v5` → **`dsv41-a2:v6`**（三处一致；preflight 会校验）
* preflight 会提示：若 `dsv41-a2:v6` 不在但 `dsv41-a2:v5` 在，**直接 retag 即可**（内容相同）

---


> v3 = `a2_pkg_v3/`（2026-09-16 18:14–18:40 打包，153 文件；**已交付给你，本包保留对照**）
> v4 = 本包（2026-09-16 21:00–21:5x，**只做增量**：新增 11 个文件 + 改 6 个文件）
>
> 每条都标了**来源**（`reports/*.md` 逐字复制在本包；A3-node2 的原文路径写在 `CORRECTNESS_STATUS.md` §10）。

---

## 0. 一句话

v3 是"**8 项已验证优化 + 单流口径**"（128K 30.2–31.6 ms）。
v4 **不改任何已验证优化**，改的是**结论的诚实度与覆盖面**：

1. **补上 v3 完全没测的形态**：生产口径（`MAX_SEQS=32 + PREFIX=1`）与
   **多 batch / 多轮对话**（历史"并发"只是 decode 并发，prefill 被 `--serialize-prefill 1` 刻意串行化）；
2. **更正 v3 的 110 tok/s 叙述**：163 发全量重算 ⇒ **真正可交付的 ≥110 只有 1 发**，
   `A` 是三吸引子抽签（steep 15.3%）且**脏会话 A 更高** ⇒ **A 不能当绩效指标**；
3. **把 4 个负结果写进"别踩坑"**，并新增 `CORRECTNESS_STATUS.md`（数值确定性 / 精度门的独立结论）。

---

## 1. v4 **新增**的文件（14 个）

| # | 文件 | 内容 | 来源 |
|---|---|---|---|
| 1 | **`CORRECTNESS_STATUS.md`** | 5 条交付级结论：无 ctx 阈值（抽签）/ `SPEC=0` clean 高 5 倍 / clean 与 coherent 独立 / "减少 forward 数"被算术否证 / 精度门 + HCCL 代价 | A3-node2 `reports/correctness-line.md`（426 行）+ `/tmp/{freq3,spec0,spec0_rep2,cvq2}.log` |
| 2 | **`tests/multibatch/multibatch_gate.py`** | 三块：**[A]多轮对话**（埋针逐字召回）、**[B]并发逐 item 比对**（16 道算术题 conc=1 vs conc=8）、**[C]长短交错**（1×128K + 6 短请求）| A3-node1 `exp_tools/multibatch_gate.py`，**硬编码 `/home/user/...` 全部参数化**（`--base/--out/--corpus`），并补 `verdict` 机器可读判据与退出码 |
| 3 | **`tests/multibatch/multibatch_session.sh`** | 生产口径起服 + 跑三块（单臂） | A3-node1 `exp_tools/multibatch_session.sh` 移植：改调 `scripts/serve_a2.sh`、`STOP_FIRST=0`（**默认不杀别人的容器**） |
| 4 | **`tests/multibatch/run_prod_both.sh`** | 两臂对照（`PREFIX=1` vs `PREFIX=0`）+ 汇总表 | 新写（主 Agent 在 A3-node1 就是跑这两臂） |
| 5 | **`tests/multibatch/verify_serve_flags.sh`** | 12 组合启动器烟测（`DRY_RUN=1`，**不占卡**） | A3-node1 `/tmp/verify_serve_flags.sh` 移植 + 换成 A2 的组合（含 `MAX_SEQS=32 PREFIX=1`） |
| 6 | **`tools/fisher_recheck.py`** | Fisher 精确检验（2×2），带 **4 个自检用例**（含"相同表必须 p=1"这个坑） | 新写（因为我们的 `fisher()` 有 bug，见 §4） |
| 7 | `patches/files/token_dispatcher_moennf.py` | `MOE_NONFINITE` 的 patch 文件（**负结果，默认不挂**） | A3-node1 `probe_moe_nf/token_dispatcher.py`（837 行，逐字节） |
| 8–11 | `reports/a-basin-and-acceptance-shape.md`（186 行）、`reports/session-attractor-and-clean-rate.md`（132 行）、`reports/draft-graph-negative-control.md`（47 行）、`reports/multibatch-and-mixed-load.md`（88 行） | v4 的**四份权威报告**，**逐字复制**（md5 与本机一致） | A3-node1 `reports/*.md` |
| — | `reports/` 从 51 → **55 份** | | |
| 12 | **`tests/acc_eval.py`** | GSM8K / C-Eval 评测器（**v3 的 `tests/t_gsm8k.py` 引用了它却漏打进包 ⇒ `MODE=full` 在 A2 上会直接失败**）。v4 补入并把硬编码的 `/home/user/models/.../encoding` 参数化（`--enc-dir` / `ENC_DIR` / 自动找 `~/models/...`），新增 `--base` 别名；**保留 `--serialize-prefill` 默认 1**（与历史 GSM8K 口径一致） | A3-node1 `scripts/acc_eval_p4s.py`（244 行）+ v4 参数化 |
| 13 | **`tools/interleave_ab.py`** | 同会话交错 A/B（**clean-rate 判据 `pos0≥0.8`**）——v4 的 4 个负结果就是用这个工具判的；p 值走包内 `fisher_recheck.py`（不再自带一份实现） | A3-node1 `exp_tools/interleave_ab.py`（**已含 FISHER-FIX**，md5 `3e8b9c67…`）；去掉对 `logs/perf/p15_*.py` 的依赖（把那 4 个小函数内联）并把容器名/路径参数化 |
| 14 | **`tools/steep_summary.py`** | 从 p42 jsonl 汇总 **steep / flat / shallow**（= `EXPECTED_PERF.md` §2 的工具） | A3-node1 `exp_tools/steep_summary.py`；默认 glob 改为本包 `results/` 与 `logs_meta/samples/`（去掉硬编码） |

## 2. v4 **修改**的文件（7 个）

| # | 文件 | 改了什么 | 为什么 |
|---|---|---|---|
| 1 | **`scripts/serve_a2.sh`** | ① 新增 **`PREFIX`**（默认 **0**，与 v3 逐字节一致）；② `CAPTURE_SIZES` 改为**按 `MAX_SEQS × (1+SP_TOKENS)` 自动扩展**（`[MULTI-SEQ-CAPTURE]`，与 A3-node1 `serve_a21.sh` 同构）；③ 新增 `MOE_NF`（默认 0，负结果臂）与 `HCCL_DET`（默认空，仅诊断）；④ 新增 **`DRY_RUN=1`**（解析后打印并退出，不碰 docker）；⑤ `serve_cmd.txt` 增打口径与 HCCL 状态 | `MAX_SEQS=32` 时旧逻辑只覆盖到 32 token ⇒ **直接起不来**；`PREFIX=1` 才是 A2 生产形态 |
| 2 | **`scripts/run_test.sh`** | 新增 **`MODE=prod`**（`MAX_SEQS=32 PREFIX=1` + 多 batch 三块，跳过 quote 性能）+ `PREFIX` 透传 + `env.txt` 记录 `max_seqs/prefix/mode` | 生产口径与单流口径**不可混比**，必须显式分开 |
| 3 | **`EXPECTED_PERF.md`** | 重写：新增 §2（163 发形态分类）、§3（clean-rate，为什么 A 不能当指标）、**§7（多 batch 生产口径：§7.3 实测三块全过 + §7.3.1 局限 6 条 + §7.3.2 为什么这是空白 + §7.3.3 如何自己跑）**、§8（与旧报告的差异说明）；更正 §2.3 的 110 tok/s 叙述；峰值统一为 **110.5**（`A×1000/ms` 口径，v3 的 110.94 是 jsonl 另一算法） | 用户要求"把这些新事实写进 EXPECTED_PERF" + 21:12 拿到生产臂实测 |
| 4 | **`README.md`** | ① 新增 §0.1 两条更正；② §1.1 = **4 个负结果表**；③ §1.2 = DSpark 状态更新（离线修复完成 + 负控已确认，正控待验）；④ 三命令骨架（加 `MODE=prod`）；⑤ 新增 §4.1 v4 开关表 | 用户硬性要求 |
| 5 | **`tests/make_report.sh`** | 新增 **§1b 多 batch 表**（读 `summary.json`） | 让 `MODE=prod` 的结论进一页纸 |
| 6 | **`tests/t_gsm8k.py`** | 新增口径警告（`--serialize-prefill 1` ⇒ only decode concurrency）+ `--enc-dir` 透传 + **缺 encoding 目录时明确跳过**（原来会报 `ModuleNotFoundError`） | v3 的 `MODE=full` 在 A2 上会因缺 `acc_eval.py` 直接失败 |
| 7 | `Dockerfile` / `scripts/build_image.sh` | 镜像 tag `dsv41-a2:v3` → **`dsv41-a2:v5`**（其余逐字节不变） | 便于与 v3 镜像并存对照 |

## 3. v4 **删掉了什么说法**（**重要，别再用旧话术**）

| v3 的说法 | v4 的更正 | 依据 |
|---|---|---|
| "A 的方差是 2.9× ⇒ 跑 8 发看中位/区间" | **A 是三吸引子抽签**（steep 15.3% / flat 7.4% / shallow 77.3%，163 发），**不是单峰 + 噪声** | `reports/a-basin-and-acceptance-shape.md` §6 |
| "A 一旦进优模式，ms 稍差也能冲过 110 tok/s ⇒ A 决定一切" | **≥110 真正可交付的只有 1 发**（另 7 发来自不采纳的 `MOE_ZERO` 会话）；**A 不能当绩效指标**（脏会话 A 反而高） | 同上 §4/§6 + `reports/session-attractor-and-clean-rate.md` §2 |
| "`MOE_ZERO` 未在本机完成端到端验证 ⇒ 待验证" | **已测完，结论是"不采纳"**（换吸引子 + 非数值等价） | `reports/session-attractor-and-clean-rate.md` §1.1 |
| "`DRAFT_GRAPH` 上卡验证未做" | **崩溃已修 + 负控已确认**（缺 metadata ⇒ A 恒 1.0，8/8 发；ms 仍 30.2 ⇒ **单看时延发现不了**）；**正控待验** | `reports/draft-graph-negative-control.md` |
| "并发测试已覆盖" | **只覆盖 decode 并发**（`--conc 4 --serialize-prefill 1` = prefill 串行）；**多轮对话从未测过** | `acc_eval_p4s.py:173-174` 的 help 原文 + `EXPECTED_PERF.md` §7.1 |
| "Vision 23/23、GSM8K 197–199" | 保持，但补齐：**GSM8K-200 三次 = 198/199/197**；`HCCL_DETERMINISTIC=true` **91/100** | `CORRECTNESS_STATUS.md` §6 |

## 4. ⚠️ 工具错误声明（**v4 新增的纪律条目**）

| 工具 | 错误 | 处置 |
|---|---|---|
| `exp_tools/interleave_ab.py` 的 `fisher()` | 两行计数**完全相同**时打印 `p = 0.0000`（相同表必为 **1.0**）。根因：列和固定但未遍历所有 x；双侧判据在 obs 为最大概率时退化 0/0 | **已修正**并用教科书标准值重新校验（见下表）；**v4 的 4 条负结果一律使用修正后的 p 值**（`reports/session-attractor-and-clean-rate.md` §6.4）；本包附 `tools/fisher_recheck.py`（含 4 个自检用例） |
| `spread` 判据（第一版） | 只统计"最常见 token"的 logprob 极差 ⇒ **假性干净**（6 发 5 种 top-1 时极差自然为 0） | 已换成 `uniq_top1==1 AND n_distinct_lp==1`（两个条件互相独立，每行都要打） |

> **共同教训（写进测量纪律）**：**统计/判据工具上线前必须用已知答案自检**（至少 3 个教科书用例）。
> 本项目已栽两次 —— 都是"在看起来最该拒绝原假设的地方给出假显著/假干净"。

### 4.1 Fisher p 值的**修正表**（**必须用这一列**）

> **v4 修正了 v3 期间使用的 Fisher 实现 bug（同行计数时返回 0.0）；所有 p 值已用教科书标准值重新校验。**

| 对照 | 2×2 | **修正后 p** | 旧值（**作废**） |
|---|---|---|---|
| `LOCAL_OWNER=fast` vs `on` | `[[5,7],[4,8]]` | **1.0000** | 0.6843 |
| `MOE_ZERO=0` vs `1` | `[[2,10],[2,10]]` | **1.0000** | 0.0033 |
| `HCCL_DET=true` 下 fast vs on | `[[5,7],[3,9]]` | **0.6668** | 0.6843 |
| `MOE_NONFINITE=0` vs `1`（N=24） | `[[2,22],[2,22]]` | **1.0000** | 0.0000 |

**4 条结论（全部"无差异"）不变，但 p 值必须换成本表。**
唯一仍然显著的是正确性线**独立实现**的 `SPEC=0` vs `SPEC=1`：
`[[5,5],[2,18]]` → 单侧 **0.0256**（`correctness-line.md:371`，**同一会话**内）；
跨会话复现（**全新容器**）`[[9,7],[2,18]]` → 双侧 **0.0042**（单侧约 0.0026）。
三次 `SPEC=0` 测量 **0.50 / 0.50 / 0.5625** 一致 ⇒ 结论稳健。见 `CORRECTNESS_STATUS.md` §3.1.1。

## 5. 与 v3 **完全相同**的部分（**一行都没改**）

* 11 个整文件补丁 + 2 个 sidecar + `admission_gate.patch`（`patches/`，md5 见 `patches/MD5SUMS`）；
* 8 项已验证优化的**默认值**（`MOE_AG=1 / SP_TOKENS=5 / O_PROJ_2D=1 / MOE_MASK=1 /
  ROPE_IDXSEL=1 / ENGRAM_JIT=1 / LOCAL_OWNER=fast / QLI_NOCAND=1 / PYTHON_PGO=1 /
  GATE_CHUNK=0 / VLLM_ADMISSION_GATE=1`）；
* **单流口径的 `CAPTURE_SIZES` 输出逐字节不变**（`1,2,3,4,6,8,12,16,20,24,32`，已在包内用
  `DRY_RUN=1 MAX_SEQS=1` 实测）；
* 容器起服参数（`--net=host --shm-size=512g --privileged`、`/dev/davinci0..7`、宿主透传、
  `LD_PRELOAD=jemalloc`、`HCCL_BUFFSIZE=1024`、`TASK_QUEUE_ENABLE=1`、`HCCL_OP_EXPANSION_MODE=AIV`）；
* CPU/NUMA 自动绑定、视觉 23 例、GSM8K-200、容量判据、`results/<run_id>/REPORT.md` 结构；
* 量化链 `quant/`（5 级装配 + 结构/容差验收）与 `optim/pgo/` 产物；
* `data/`（红楼梦语料 + 4 个 suffix）与 `logs_meta/`（LOG_INDEX + 19 个 jsonl 样本）。
# v5（2026-09-16）—— **软链构造的模型目录：起服必失败的修复**

# v5.1（2026-09-16 21:50）—— **A2 实测反馈的两个致命 bug**

> A2 真机跑 `run_test.sh` 时暴露。两条都会让流程**完全走不下去**，且报错极具误导性。

## Bug 1（起服阻断）：`mapfile` 把 `-v` 和路径塞进了同一个参数

**现象**（A2 实测原文）：

```
docker: Error response from daemon: create  /home/.../optional:
" /home/.../optional" includes invalid characters for a local volume name,
only "[a-zA-Z0-9][a-zA-Z0-9_.-]" are allowed.
```

**注意错误信息里路径前面的那个空格** —— 它就是指纹。

**根因**：`tools/model_mount_args.sh` 原来每行输出 `-v /path:/path:ro`，
而 `serve_a2.sh` 用 `mapfile` 读入 —— **每行只成为一个数组元素**。
于是展开给 docker 的是**单个参数** `"-v /path:/path:ro"`，
Go 的 pflag 会把 `-v` 后面的**空格也算进值里** ⇒ 得到的路径是 `" /path"`（带前导空格）
⇒ docker 认为那不是绝对路径，转而按"卷名"解析 ⇒ 报 invalid characters。

**修复**：`model_mount_args.sh` 改为只输出**裸路径**（每行一个），
由 `serve_a2.sh` 显式拼成 `-v` 与 `路径:路径:ro` **两个**数组元素。

**验证**：真值 5 层链条 → `dirs=5, argv=10`，`docker run` 内 3 个文件全部可读；
旧写法在同样输入下必然失败。

## Bug 2（构建阻断）：Dockerfile 续行链被行内注释截断

**现象**：`docker build` 直接报
```
dockerfile parse error on line 4: unknown instruction: local
```

**根因**（两处，都在 `RUN` 的续行链里）：

```dockerfile
RUN set -euo pipefail; \
    inst() { # src_in_tmp  target_rel      ← ① 行内注释 + ② 这一行没有 `\`
      local tgt="${ASCEND_PKG}/$2"; \
```

1. **中间行漏了结尾 `\`** ⇒ 链在此**提前结束**，后面的 `local` / `test` / `cp`
   被当作 Dockerfile 指令解析 ⇒ `unknown instruction: local`；
2. **行内 `#` 注释**：Docker 先把续行拼成**一整行**再交给 shell，
   行内 `#` 会把**它后面的一切**（包括还没执行的命令）全部注释掉。
   —— 而如果把 `\` 写在注释**后面**，那个 `\` 本身也在注释里，等于没写。
   （用户原话：「不要在行末加注释，否则 `\` 失效」）

**修复**：`inst() { \` / `newf() { \`（去掉行内注释、补上 `\`），
把解释性文字整体移到 `RUN` **外面**的注释块，并在那里写下"续行铁律"。

**验证**（最小复现 + 正负控，本机实跑）：

| 版本 | 结果 |
|---|---|
| 旧写法（行内注释 + 缺 `\`） | `docker build` → **`dockerfile parse error on line 4: unknown instruction: local`** |
| 新写法 | `docker build` → **Successfully tagged dftest:fixed**；容器内 `BUILD_OK` |

## 新增两个自检工具（防止这两类 bug 再发生）

| 工具 | 作用 |
|---|---|
| `tools/check_dockerfile.py` | 检查 Dockerfile 续行链：链中行的**行内 `#`**、**漏 `\`**、**`\` 落在注释里**，全部报 ERROR。已接入 `selfcheck_pkg.sh` |
| `tools/selfcheck_pkg.sh`（v5.1 加入 Dockerfile 检查） | 10 秒包自检：**镜像 tag 一致性**（build_image 产出 vs serve_a2/run_test 查找）、脚本语法、`MODEL_MOUNTS` 接线、Dockerfile 续行链、执行位、MANIFEST |

> `tools/selfcheck_pkg.sh` 第一次运行就抓出了我自己引入的 tag 不一致
> （`build_image.sh` 产出 `v4`、`serve_a2.sh` 找 `v5`），可见这类检查是必要的。

---

# v5.0（2026-09-16）—— **软链构造的模型目录**

> v3/v4 用软链构造的模型目录起服**必然失败**。

## 故障

量化流水线（modelscope 上那套脚本）产出的最终目录是**零拷贝的软链结构**，
软链是**绝对路径**且**链条很深**。真实产物实测：

```
软链跳数分布: 1 跳 × 2,  2 跳 × 8,  3 跳 × 4,  4 跳 × 80     （共 94 个软链）
```

链条：`L5(最终) → L4 → L3 → L2 → L1(真正的 87 个实体分片)`

而 v3/v4 的 `serve_a2.sh` 只有 `-v "$MODEL:$MODEL:ro"` —— **只挂了 L5**。
容器里所有指向 L4/L3/L2/L1 的绝对路径软链**全部悬空**：
宿主机 `ls`/`cat` 正常，进容器立刻 `No such file or directory`
（通常先炸在 `config.json` / tokenizer 上，白等几分钟后失败在 worker 里）。

**实测复现**（两行就是全部差别）：

```bash
# 旧行为：只挂叶子层
docker run --rm -v <L5>:<L5>:ro alpine cat <L5>/config.json
#   cat: can't open '.../config.json': No such file or directory

# 新行为：逐层挂载
docker run --rm $(bash tools/model_mount_args.sh <L5> | tr '\n' ' ') alpine cat <L5>/config.json
#   {"model_type":"deepseek_v41",...}
```

## 修复

| 文件 | 改动 |
|---|---|
| `tools/model_mount_args.sh` | **新增**。逐跳解析**字面软链**（用 `readlink`，**不是 `realpath`**），把每一跳的目标目录都输出成 `-v` 参数。真值需要挂 **13 个目录** |
| `scripts/serve_a2.sh` | 自动调用上面的工具，用 `"${MODEL_MOUNTS[@]}"` 取代单层挂载；新增 `MODEL_MOUNT_MODE`（`auto`/`ancestor`/`none`）与 `EXTRA_MODEL_MOUNTS`；`DRY_RUN=1` 会打印最终挂载清单 |
| `tools/check_model_dir.sh` | 新增**第 0 步**：起服前检查悬空软链（有则 FATAL 并给出修法）+ 报告软链跳数分布 ⇒ **5 秒内失败**，而不是白等 4 分钟 |
| `README.md` / `REPRO.md` | 新增 §3.0 专章解释这个坑 |

## 一个反直觉的实现要点（写下来避免以后改回去）

**不能用 `os.path.realpath()`**：它会把 `L5→L4→L3→L2→L1` **一次折叠成 L1**，
于是看起来"目标只有 L1，挂 L1 就够"。但**容器是逐跳解析的**：
打开 `/abs/L5/config.json` 读出 `"/abs/L4/..."`，再去开 `/abs/L4/config.json`。
所以**每一跳的目标目录都必须挂**。这就是本工具坚持用 `readlink` 的原因。
（第一版实现正是踩了 `realpath` 这个坑：5 层链条只解析出 2 个目录。）

## 另一个自己踩的坑（同型问题第二次）

`serve_a2.sh` 里判断工具是否存在时我最初写的是 `[ -x ... ]`，而交付包解包后
脚本的**执行位可能丢失** ⇒ 判断为假 ⇒ **静默退回"只挂一层"**，正好把这个 bug 又复现了一遍。
已改成 `[ -f ... ]`（反正调用方式是 `bash <script>`，不需要执行位），
并在 fallback 分支打印显式告警。

## 验证

| 项 | 结果 |
|---|---|
| 5 层人造链条：解析出的目录数 | **5/5**（旧实现只出 2 个） |
| 5 层人造链条：容器内读 3 个文件 | **全部可读**（旧行为报 `No such file or directory`） |
| A3-node1 真值模型目录：解析出的目录数 | **13 个**（含跨到第二个绝对路径前缀的软链） |
| A3-node1 真值模型目录：`check_model_dir.sh` | 94 软链全部可解析；跳数 `1×2, 2×8, 3×4, 4×80` |
| 悬空软链：`model_mount_args.sh` | **rc=1** + 打印断链清单与修法 |
| 悬空软链：`check_model_dir.sh` | **FATAL** + 修法 + 指向自检工具 |
| `DRY_RUN=1` 输出 | 列出最终 `MODEL_MOUNTS` 清单 |

## 未验证

1. **真机起服未跑**（本机没有 A2 的 openeuler 镜像，且按纪律未起容器）。
   上卡第一件事应是 `MODEL=... DRY_RUN=1 bash scripts/serve_a2.sh | grep MODEL_MOUNT` 看清单。
2. `MODEL_MOUNT_MODE=ancestor` 分支只做了 DRY_RUN 级验证，未在真机起服。
3. 若模型目录软链指向了**模型目录之外**的地方（如 `/opt/...`），
   需要 `EXTRA_MODEL_MOUNTS` 手动补 —— 本工具只解析从 `MODEL` 出发能看到的软链。

---
