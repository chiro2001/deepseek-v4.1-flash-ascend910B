# 历史「≈24 ms/step」那一发的完整测试参数（溯源）

> 起因：核查"历史上 24ms 左右的低时延是在什么口径下测出来的"。
> **结论：它是真实测量，但口径与我们现在的 CED-PD 差 5 个维度；
> 其中 `STATIC_KERNEL` 这一项是可以直接拿回来的。**

## 0. 数字出处（可精确复现到文件）

| 项 | 值 |
|---|---|
| 机器 | a3-21 |
| 目录 | `~/projects/dsv41-release/results/e2e_fix_H_final/` |
| 文件 | `guard_E1.json` / `guard_G1.json` / `guard_G2.json` |
| 时间 | 2026-09-20 19:44–20:13 |
| 早一次的复现 | `results/e2e_fix_E_ab/`（18:55–18:58），数字略高 |

对照 `README.md` §2.7 的 "36.9 → 23.9–24.9 ms/step"，逐个对上：

| 臂 | 单流 tok/s | A | **ms/step = 1000×A/单流** |
|---|---:|---:|---:|
| E1（eager） | 66.5 | 2.455 | **36.89** |
| G1（graph） | 100.7 | 2.403 | **23.87** |
| G2（graph） | 109.8 | 2.738 | **24.94** |

## 1. 客户端参数（用的就是 `bench_concurrency.py`）

```
[bench] base=http://127.0.0.1:8030 model=deepseek-v41 label=-
[bench] prompt 1024 tok（**精确校准**）, output=256 tok, repeats=1, 并发级别=[1, 7, 8]
[bench] 语料=data/dihuo.txt（正文 24119 字符）  问题后缀=5 个
[bench] 切片步长=3014 字符（8 个请求 × 步长 = 24112）⇒ 完全互不重叠
[bench] prompt 校准：8 个请求，目标 1024 tok；完全命中 8/8，最大偏差 0 tok
```

| 参数 | 值 |
|---|---|
| **prompt** | **1024 token**（精确校准） |
| **输出** | **256 token** |
| **并发** | **1**（8 条请求，一条一条跑） |
| repeats | 1（无重复取样） |
| 语料 | `data/dihuo.txt`（刘慈欣《地火》，24,119 字符） |
| 采样 | 8 条请求**互不重叠**的切片 + 轮换 5 个问题 ⇒ 无前缀复用 |

## 2. 服务端参数（完整，来自 `inner.sh` / `serve_cmd.txt`）

```bash
export MODEL=.../v41-w4a8-engram-dr-vision-qrot-mtpq   # ★ 单实例，非 PD
export TP=8 DP=1 PORT=8030 SERVED_NAME=deepseek-v41
export MAX_LEN=1048576 MAX_SEQS=32 BAT_TOKENS=8192 GPU_UTIL=0.92 BLOCK=128
export KV_DTYPE=bfloat16 GRAPH=1 EAGER=0 PREFIX=1 SPEC=1 SP_TOKENS=5   # ★ SP=5
export ENGRAM=1 ENGRAM_STORAGE=int8 VISION=1
export NPUGRAPH_EX=1 STATIC_KERNEL=1 CPU_BIND=0                        # ★ 静态核 = 1
export MULTISTREAM=1 DSA_OVERLAP=1                                     # ★ 多流 = 1
export LOADER_MT=1 LAZY=1
export CAPTURE_SIZES="1,2,3,4,6,8,12,16,20,24,32,40,48,96,192"          # 含 6/40/48/96/192
```

**E1（eager 臂）的做法**：不是另外起一次服务，而是在**同一个进程内**用运行期热切换
（`/tmp/v41_dspark_flags` 写 `DRAFT_FORCE_EAGER=1`）——
`serve.log` 里 `[dspark-force-eager]` 命中 **16,824** 次。所以 E1 与 G1/G2 是
**同进程配对臂**，排除了跨起服混杂，这也是这份数据可信的原因。

## 3. 与我们当前 CED-PD 的差异

| 维度 | 历史 23.9/24.9 | **我们现在（CED-PD D 侧）** | 能否拿回来 |
|---|---|---|---|
| 部署形态 | 单实例 TP8 | **P/D 双实例 8+8** | 不能（CED 的前提） |
| 上下文 | **1024** | 2048 | 口径差异，非配置 |
| 并发 | **1** | 4 | 口径差异，非配置 |
| **`SP_TOKENS`** | **5** | **7** | 可测（历史上 S=5 vs S=7 有矛盾，见 §5） |
| **`STATIC_KERNEL`** | **1** | **0** | ✅ **可以拿回来** |
| `MULTISTREAM` / `DSA_OVERLAP` | 1 / 1 | **0 / 0** | ❌ CED 的 D 侧必须 0/0（见 §5） |
| `PREFIX` | 1 | 0 | ❌ 当时是独立实验臂；**2026-09-27 已转为交付默认 = 1**（见 CED-PD-CACHE-HIT-PLAN §11–§13） |
| `MAX_SEQS` | 32 | 4 | 口径差异 |

## 4. ★ 可操作发现：`STATIC_KERNEL`

**历史那一发是 `STATIC_KERNEL=1`；而 CED 从第一版起就是 `STATIC_KERNEL=0`。**

而 `STATIC_KERNEL=0` **不是性能设定，是排障降级手段** ——
`docs/LEGACY-DELIVERY-NOTES.md` §5「出问题先看」把它列为"健康检查三连"之一：

> 健康检查三连（依次排除）：`PYTHON_PGO=0` → **`STATIC_KERNEL=0`** → `MAX_LEN=131072`

它的收益在别处有实测（[`reports/cannbot-sweep-final-verdict.md`](../reports/cannbot-sweep-final-verdict.md) §1）：

| # | 手段 | 收益 |
|---|---|---|
| 1 | **static kernel**（`enable_static_kernel=1`） | **128K −2.87 ms** |
| 1b | `LOCAL_WORLD_SIZE` 注入（修静默降级） | **−4~5 ms（12%）** |
| 2 | `enable_fused_mc2=1` | comm 9.19→4.55 ms/step（A2 路线；A3 已用 AllGather 取代） |
| 3 | jemalloc | 12% |

⇒ **我们的 D 侧可能白付了 ≈2.9 ms/step。** 这是本溯源里最值得立刻验证的一项。

⚠️ 但要注意 `STATIC_KERNEL=1` 曾是「静默降级」的受害项（上面 1b 那条），
起服后必须核对它**真的生效**（`compile start` 计数 > 0、无相关 warning），
不能只看 env 传进去了。

## 5. ❌ 不能照搬的两项

### `MULTISTREAM=1 DSA_OVERLAP=1`

历史单实例用它没问题，但 **CED 的 D 侧用它会在长上下文下静默乱码**。
[`docs/CED-PD-PERF-20260925.md`](CED-PD-PERF-20260925.md) §3：只重启 D、只改这一个开关——

| D 侧配置 | 144K 四针 |
|---|---|
| `MULTISTREAM=1 DSA_OVERLAP=1` | **0/4**，全部乱码 |
| `MULTISTREAM=0 DSA_OVERLAP=0` | **4/4 PASS** |

⇒ 这条是 CED 特有的代价，**D 侧必须保持 0/0**（P 侧仍是 1/1，不受影响）。

### `SP_TOKENS=5`

历史文档对 S=5/7 的结论**自相矛盾**，不能直接照搬：

| 来源 | 结论 |
|---|---|
| `reports/sptok-sweep-allgather.md` | S=5 **32.894/34.305 ms、A=3.400/3.493**；S=7 35.104 ms、A=2.763 ⇒ **S=5 更优** |
| `reports/cannbot-sweep-final-verdict.md` §2 | 「S=5：8K 41.10 / 32K 42.01 / 128K 45.13（**更差**）；**S=7 最优**」 |

两者相差 ~12 ms（≈30%），远超噪声。**而历史 24ms 那一发用的是 S=5。**
在我们自己的 CED 口径下重测一次 S=5 vs S=7，是第二项值得做的事。

## 6. 建议的验证顺序

1. **`STATIC_KERNEL=1`**（单变量，只重启 D）：预期 −2.9 ms/step；
   起服后必须核对 static kernel **真的编译**（不能只信 env）。
2. **`SP_TOKENS=5`**（单变量）：先看 A（接受长度），再看 ms/step。
   按上一份拆解，少 2 个草稿行省的主要是 draft 侧（本来就只占 13%），
   真正省的是 **target 那 8 行变 6 行**；但 A 可能同比例下降，
   所以两个量要一起看，不能只看其中一个。
3. 两项都过正确性回归（144K 四针）后，再重测并发曲线。

## 7. 口径警告（历史文档自己写的）

`a2/logs/102-msperstep-regression-attributed.md` §5 与 `a2/docs/4AXIS-SUMMARY.md`
都把这条数字标为**不可直接比**。其中一份说它出自
`draft_ab.py --seq-len 1032`（单 chip 微基准），
**但那个说法与本轮溯源不符**：§0 的三个数字（36.9 / 23.9 / 24.9）与
`guard_{E1,G1,G2}.json` **逐位吻合**，而那三个 json 是
`bench_concurrency.py` 打给**完整 TP8 服务**的
（客户端行 `base=http://127.0.0.1:8030`、`prompt 1024 tok`、`并发级别=[1, 7, 8]`）。

⇒ 以能对上数字的 `guard_*.json` 为准；`draft_ab.py` 是另一条独立的
单 chip 证据，**不要混引**。
