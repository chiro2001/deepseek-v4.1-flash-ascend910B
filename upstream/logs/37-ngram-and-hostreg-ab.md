# RFC §3.3 的两行：n-gram token history 与 host table registration 双 API 对比

> 2026-09-21 CST｜A3-node1 独占槽位容器（c0 = die 3 / c1 = die 6 / c2 = die 7）
> SoC `Ascend910_9382`（PCI `19e5:d803` = 910C 类，CPU↔NPU over HCCS），CANN `9.1.0`，
> driver `26.1.1`，内核 `6.6.0-159.4.3.154.oe2403sp4.aarch64`，python 3.12.13，
> torch 2.10.0+cpu / torch_npu 2.10.0.post4 / numba 0.67.0 / numpy 1.26.4
> 全部 16 个 die 的 `/proc/svm/dev*/feature/host_mem_pool` = **1**
> 原始数据：`logs/raw/37-*`（sha256 见 §4）
> **本文件只新增数据，不改 §3.1/§3.2**

---

## 0. 一句话结论

**两行都跑出来了，结论比预期平淡，但方向都对得上。**

* **n-gram history**：`torch.equal` 在全部 14 个尺寸、全部 6 个臂上**逐位一致**。上游
  `update()` vs 我们的 numba JIT，在生产 decode 形状（n=128 tokens / 64 req / block 128）
  是 **1.681 → 0.0736 ms（22.8×）**；把上游自己那段 per-token Python 走法单独拎出来
  （`C` 臂）是 **10.31 → 0.0079 ms（1312×）**。
  但 RFC 里引用的 **0.427 → 0.076 ms 是引擎内 per-step 相位计时**，不是本 harness 的口径；
  **绝对值对不上**（本 harness 上游侧是 1.68 ms），**比值对得上**（本 harness 22.8×，
  另一处纯 CPU 基准 23×）。
* **host table registration**：`aclrtHostRegister(MAPPED)` 与
  `aclrtHostRegisterV2(MAPPED|PINNED)` **两条都 ret=0，设备侧读回逐字节一致**，
  在 128 MiB（单 die，含 Triton 裸指针读回）和 512 MiB（die3+die7 **并发**）两档都是如此。
  **在 128 MiB / 512 MiB 这一档，两条 API 没有分岔** —— issue #16828 的分岔点在
  **206 GiB 满表**那一档，**这一档没跑**（见 §6）。

---

## 1. 脚本审计（任务第 1 步）

两个脚本都**不内嵌**我们的实现，**按路径加载**——所以「测的就是出货代码」这句话是
可验证的，`--verify-verbatim` 会 re-diff + 重算 sha256。

| 脚本 | 需要的输入 | CLI | 输出 |
|---|---|---|---|
| `pr/bench_ngram_history.py` | ① `_src/engram_hash.py`；② `_src/engram_jit_kernel.py`（由 `--ours-hash-src` / `--ours-kernel-src` 指定，默认先找 `<repo>/dsv41-release/patches/files/`，再找脚本同级的 `_src/`）；③ 仅 `--verify-verbatim` 需要上游源 `_src/upstream_common_pr16925.py` | `--sizes`（默认 14 档 `1…8192`）、`--reps` 30、`--warmup` 5、`--seed`、`--max-seqs` 64、`--history-max-tokens` 4096、`--verify-verbatim` | `--out DIR` → `bench_ngram_<tag>.txt`；`--json PATH` → 结构化 JSON |
| `pr/probe_engram_hostmap.py` | **无外部输入**（自己写 `.safetensors` 再 mmap）；需要 NPU | `--device`、`--size-mib`（默认 128）、`--gather-rows`（默认 4096）、`--seed`、`--skip-flag-matrix`、`--no-triton` | **没有 `--json`/`--out`**：只打 stdout，退出码 0=可用 / 3=不可用 / 1=probe 错 |

补充发现（都影响数字怎么读）：

1. **`bench_ngram_history.py` 是纯 host/CPU**，不 import torch_npu，但按工作区规则仍占槽位。
2. **`probe_engram_hostmap.py` 的 `--size-mib` 实际只支持到 128**：section [6] 的 Triton
   control 会按整张表尺寸在 HBM 上开一块同尺寸缓冲，`rows * ROW_WIDTH` 一旦超过 `2**31/8`
   就死在 `random.Random.randbytes` 的 `OverflowError`（python int → C int）。128 MiB 时
   `rows*256 = 2**27` 字节、`getrandbits(2**30)` 恰好不越界；**256 MiB 起必挂**，与注册无关。
3. **`probe_engram_hostmap.py` 没有 fanout / 多 die 参数**（`grep` 为空），
   所以「多 die 并发」只能用**两个进程各占一个槽位同时启动**来测（§3.3 就是这么做的）。
4. probe 自带的 `header check : MISMATCH` 是**假警报**：它的解析器用 `^#define` 正则取值，
   而 legacy 的 `ACL_HOST_REGISTER_MAPPED` 是**枚举**（`acl_rt.h:192`，没有 `#define`），
   取不到 → `None == 0` → 判 False。实测三个常量**全部正确**：
   `ACL_HOST_REG_MAPPED=0x2`（`:75`）、`ACL_HOST_REG_PINNED=0x10000000`（`:78`）、
   `ACL_HOST_REGISTER_MAPPED=0`（`:192`）。**flag 列可以信。**
5. probe 的 `ret` 列宽 18 字符，非零码的**名字会被截断**（打印成 `107017 (ACL_ERROR_`）。
   本日志的 JSON 用 probe 自带的错误表补全为 `ACL_ERROR_RT_INVALID_HANDLE`。

---

## 2. 结论表 A：n-gram token history（`bench_ngram_history.py`）

口径：函数级、单进程、无并发、无 NPU；`n` = 该 step 的 token 数，`nreq = min(n,64)`，
`block_size=128`，`lookback=4`，`n_hash_cols=24`，输入为固定种子合成 id（表值/布局是生产形状）。
**每一次运行都检查 `torch.equal`**，下表 `eq` 列为 `U`/`J` 两臂的 hash 与 mask 对比。

### 2.1 整体 `update()`：上游 verbatim vs 我们的 JIT

单位 ms（median of 30）。`U` = 上游 PR #16925 verbatim，`O` = 我们的 stock（JIT 关），
`J` = 我们的 numba JIT。

| n | U 上游 | O stock | **J ours JIT** | **U/J** | O/U | 上游实际走的分支 | eq |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 0.2003 | 0.3057 | 0.0257 | **7.8×** | 1.53 | scalar row/shift loop | ✅ |
| 4 | 0.4553 | 0.3373 | 0.0274 | **16.6×** | 0.74 | scalar row/shift loop | ✅ |
| 16 | 1.0861 | 1.0794 | 0.0352 | **30.9×** | 0.99 | torch slab per shift | ✅ |
| 64 | 1.4136 | 1.3916 | 0.0517 | **27.3×** | 0.98 | torch slab per shift | ✅ |
| **128** | **1.6807** | 1.6450 | **0.0736** | **22.8×** | 0.98 | torch slab per shift | ✅ |
| 384 | 1.8148 | 1.7988 | 0.1491 | **12.2×** | 0.99 | torch slab per shift | ✅ |
| 1024 | 2.4510 | 2.4349 | 0.3393 | **7.2×** | 0.99 | torch slab per shift | ✅ |
| 4096 | 5.3393 | 5.3128 | 1.2376 | **4.3×** | 1.00 | torch slab per shift | ✅ |
| 8192 | 9.7999 | 9.7260 | 2.4411 | **4.0×** | 0.99 | torch slab per shift | ✅ |

* 14/14 尺寸 `torch.equal(hashes)=True`、`equal(mask)=True`；「跑完一轮再验」也全 True。
* `O/U ≈ 0.98–1.00`（除 n≤8 的小尺寸噪声）⇒ **我们的 stock 确实是上游的忠实拷贝**，
  「copy-fidelity」是测出来的，不是断言的。
* 生产 decode 形状取 **n=128**（64 路并发各 1 个 token，`MAX_SEQS=64`）：
  **1.6807 → 0.0736 ms，22.8×，−1.607 ms/step**。J 的 min-of-30 是 0.0716、p90 0.0767。

### 2.2 上游那段 per-token Python 走法本身（§3.3 那行的字面意思）

`C` = 上游 `< 16 token` 小批分支的**逐 token `for row / for shift` 走法 verbatim**
（`common.py:112-130`，guard 已旁路），`P` = 同一走法改在打包 numpy 镜像上（仍是纯 Python），
`N` = 同一走法 `numba.njit`。单位 ms。

| n | C python | P numpy-py | N numba | C/P 布局 | P/N JIT | **C/N 合计** | eq |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.0361 | 0.0121 | 0.0033 | 2.99× | 3.70× | **11.1×** | ✅ |
| 4 | 0.2643 | 0.0475 | 0.0035 | 5.56× | 13.69× | **76.2×** | ✅ |
| 16 | 1.0302 | 0.1663 | 0.0039 | 6.20× | 42.47× | **263.2×** | ✅ |
| 64 | 4.5910 | 0.7074 | 0.0053 | 6.49× | 133.73× | **867.9×** | ✅ |
| **128** | **10.3053** | 1.5299 | **0.0079** | 6.74× | 194.76× | **1311.9×** | ✅ |
| 1024 | 75.3917 | 11.5963 | 0.0217 | 6.50× | 533.78× | **3470.3×** | ✅ |
| 4096 | 303.0404 | 46.6664 | 0.0691 | 6.49× | 675.69× | **4387.8×** | ✅ |

⇒ 拆得很干净：**约 6.5× 来自数据布局**（packed numpy 镜像），**其余 10–670× 来自 JIT**。
`C` 的量级随 n 线性涨（10.3 ms @ n=128），这就是「上游 per-token Python 走法」的真实成本。

### 2.3 稳健性：换 die + 关掉 `OMP_NUM_THREADS=1`

镜像里 `OMP_NUM_THREADS=1`（affinity 其实是 640 核，是环境变量把 torch 压到单线程）。
在 **c2/die 7** 上以 `OMP_NUM_THREADS=16` 重跑一遍：

| n | U 1线程 | U 16线程 | J 1线程 | J 16线程 | U/J 1t | U/J 16t | C/N 1t | C/N 16t |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2003 | 0.2038 | 0.0257 | 0.0234 | 7.8× | 8.7× | 11.1× | 11.3× |
| 16 | 1.0861 | 1.0891 | 0.0352 | 0.0334 | 30.9× | 32.6× | 263.1× | 283.1× |
| 64 | 1.4136 | 1.4201 | 0.0517 | 0.0509 | 27.3× | 27.9× | 867.9× | 914.6× |
| **128** | **1.6807** | **1.4948** | **0.0736** | **0.0705** | **22.8×** | **21.2×** | **1311.9×** | **1494.0×** |
| 4096 | 5.3393 | 4.3701 | 1.2376 | 1.2368 | 4.3× | 3.5× | 4387.8× | 4496.1× |
| 8192 | 9.7999 | 6.6592 | 2.4411 | 2.4376 | 4.0× | 2.7× | — | — |

⇒ **decode 尺寸（n ≤ 384）对线程数不敏感**（差值在 10% 内），结论不是单线程假象。
只有 n ≥ 4096 的大 slab 才吃线程（上游那侧快 ~27%），但**即便如此 JIT 仍快 2.7–4.0×**。
两次运行同时换了 die（c0/die3 → c2/die7），在 n ≤ 384 上**换 die 也没造成可见差异**——
两个变量一起变了，所以这一条是「换 die 没坏事」，不是「die 之间严格等价」的证明。

---

## 3. 结论表 B：host table registration 双 API（`probe_engram_hostmap.py`）

机器前提（都是 probe 自己打的）：

| 判据 | 实测值 | 含义 |
|---|---|---|
| `/proc/svm/dev*/feature/host_mem_pool` | **dev0…dev15 全 = 1** | pooled 注册快路径，即 A3 类前提成立 |
| `lspci -nn -d 19e5:` | 199 条 Ascend 设备，其中 **16 条 `19e5:d803`** | 910C 类（A3），CPU↔NPU over HCCS |
| `soc` / driver / CANN | `Ascend910_9382` / `26.1.1` / `cann-9.1.0` | — |

### 3.1 §3.3 那两行（两条 API，同一块 128 MiB 生产形状内存）

形状 = 可写文件 `mmap` + payload 偏移 96 字节（`HostMappedSafetensors` 的确切形状），
每行注册一次、当场注销。**「读回」= 设备侧 `torch.index_select` 4096 行后与 host 原始字节
`torch.equal`。**

| # | API | flags | ret | 注册耗时 | 设备读回逐字节一致 |
|---:|---|---|---:|---:|---|
| 1 | `aclrtHostRegister` | `MAPPED`（enum `0x0`）| **0** | 89.2 ms | ✅ `torch.equal` |
| 2 | `aclrtHostRegisterV2` | `MAPPED`（`0x2`）| **0** | 48.8 ms | ✅ `torch.equal` |
| 3 | `aclrtHostRegisterV2` | `MAPPED\|PINNED`（`0x10000002`，**这就是 PR #16925 的调用**）| **0** | 45.5 ms | ✅ `torch.equal` |

⇒ **两条 API 都 ret=0，且都真的能被设备算子逐字节读回**。这一档**没有分岔**：
`acl.rt.host_register(MAPPED)` 这条路在本机生产形状下**是成立的**，不是「只有 V2 能行」。

> ⚠️ **注册耗时不要横向比**：上表三个数字是单次测量，而 section [6] 独立复测同样三行得到的是
> legacy 48.3 / V2 MAPPED 45.7 / V2 MAPPED|PINNED 89.9 ms —— **快慢顺序翻转了**。
> 两处都落在 ~45 ms 和 ~90 ms 两簇里（见 §5 推断 2），**不足以断言哪条 API 更快**。
> 判据是 ret 码与读回一致性，不是耗时。

### 3.2 flag 组合矩阵（probe section [6]，128 MiB，每行设备读回两次）

两次读回 = ① DLPack → `torch.index_select`；② Triton kernel 把发布的 int64 地址
`tl.cast` 成 `tl.pointer_type(int8)` 再裸指针 load（**PR #16925 gather kernel 自己的消费形状**）。

| # | API | 缓冲 | flags | ret | dev==reg | 注册 ms | DLPack 读回 | Triton 裸指针读回 |
|---:|---|---|---|---:|---|---:|---|---|
| 7 | V2 | file_rw | `0x10000002` MAPPED\|PINNED | **0** | no | 89.9 | ✅ 0.5ms | ✅ 1ms |
| 8 | V2 | file_rw | `0x2` MAPPED | **0** | no | 45.7 | ✅ 0.5ms | ✅ 1ms |
| 11 | legacy | file_rw | `0x0` MAPPED (enum) | **0** | no | 48.3 | ✅ 0.5ms | ✅ 1ms |
| 9 | V2 | file_rw | `0x10000000` PINNED | 0 | — | 0.0 | ❌ **没发布设备地址**（`aclrtHostGetDevicePointer` 给 0）| ❌ |
| 10 | V2 | file_rw | `0x0` 无 flag | **107000** `PARAM_INVALID` | — | 0.1 | — | — |
| 12 | legacy | file_rw | `0x10000002` 原始 V2 字 | **207000** `FEATURE_NOT_SUPPORT` | — | 0.1 | — | — |

同形状（pinned / anon_rw）的 1–6、13–18 行结论一致：
**MAPPED 位是必需的，PINNED 位不影响可读性**。两条重要陷阱（都是实测）：

* **`V2` 只返回错误码、不返回设备指针**（CANN ≥ 8.x）。只把 `aclrtHostRegister` 换成 `V2`
  会静默拿到 `devPtr=0`，必须再调 `aclrtHostGetDevicePointer`。
* **legacy 的 `MAPPED = 0x0` 是枚举值，V2 的 `MAPPED = 0x2` 是位**——把 V2 的 flag 字
  (`0x10000002`) 直接喂给 legacy → `ret=207000`。这正是「两条 API 的分岔点」的机制。

（probe 的 `[6] verdict: H1 REFUTED`：内存形态不是区分因素——pinned 内存按 MAPPED|PINNED
注册后被设备 kernel 裸指针解引用，读回正确。）

### 3.3 多 die 并发注册（加分项，512 MiB × 2 die）

同一个脚本，**两个槽位同时启动**，512 MiB 表：c0 → **physical die 3**，c2 → **physical die 7**。

| 组合 | die 3 (c0) | die 7 (c2) |
|---|---|---|
| `aclrtHostRegister` file_rw MAPPED | ret=0, 315.9 ms, 读回 ✅ | ret=0, 310.4 ms, 读回 ✅ |
| `aclrtHostRegisterV2` file_rw MAPPED | ret=0, 185.6 ms, 读回 ✅ | ret=0, 184.2 ms, 读回 ✅ |
| `aclrtHostRegisterV2` file_rw MAPPED\|PINNED | ret=0, 184.8 ms, 读回 ✅ | ret=0, 182.9 ms, 读回 ✅ |
| 只读 mmap（应被拒） | ret=107017 `INVALID_HANDLE` ✅ 如期 | ret=107017 `INVALID_HANDLE` ✅ 如期 |

⇒ 两 die 并发注册 512 MiB **无相互干扰、无 507011 / 207001**，读回全一致。
**但这不是 #16828 那个场景**：那个场景是**满表 206 GiB**（4 KiB/页的 per-page metadata 会
把 vmalloc 打爆）。**512 MiB 与 206 GiB 差 ~400×，本项不能用来证明 #16828 在 A3 上没问题。**

（这两个进程在 section [6] 的 Triton control 上 rc=1 退出——就是 §1.2 那个 `randbytes`
上限；但 **[1]–[5] 已经跑完**，上表数据取自它们的 section [3]/[4]。）

---

## 4. 证据路径

远端（A3-node1 `~/projects/dsv41-upstream-pr/agents/N_ngram/`）：脚本 `bench/`、`_src/`（我方两个
交付文件 + 上游 `common.py`）、原始输出 `out/`。本机：

| 文件 | 内容 | sha256 |
|---|---|---|
| `logs/raw/37-ngram-history-a3prbench-c0.json` | n-gram 主跑，die3，OMP=1 | `962d01c17aab2f64ff2105682eef63ee4590a87efcec0d0b26fec698f179c061` |
| `logs/raw/37-ngram-history-a3prbench-c0.txt` | 同上，stdout 全文 | `8161c8fd761dd061fe424db081ccdbf4e1134da5c79f79d548725ac72f0eed45` |
| `logs/raw/37-ngram-history-a3prbench-c2-omp16.json` | n-gram 复跑，die7，OMP=16 | `f0d87f259eea60dc87e46525875f1383071b510d9d3f7753962583a7ae30f534` |
| `logs/raw/37-ngram-history-a3prbench-c2-omp16.txt` | 同上，stdout 全文 | `75885ab6d8639288707cc867439e7d76772a794b5af588881a555e68ce6b1b1e` |
| `logs/raw/37-hostmap-ab-a3prbench-c1.txt` | host register 全量 probe，die6，128 MiB | `f3ee2595bddb94e7c354f23a06af2553d5633655d0c86e2af1c70bb9f4ce7406` |
| `logs/raw/37-hostmap-ab-a3prbench-c1.json` | 上者的机械转写（probe 无 `--json`） | `9052c2d3b64ee8a8695556cafba259deef45546189549e8d08969c203798c933` |
| `logs/raw/37-hostmap-conc-512mib-c0-die3.txt` | 并发注册，die3，512 MiB | `24185bb5f3453373d19121b4cbfa34f899f0d0cbc1cbcd873a737f9fd1735aa3` |
| `logs/raw/37-hostmap-conc-512mib-c0-die3.json` | 同上，机械转写 | `dfb28a3476fb7c0e32a7b36fc3ca7d33398462a86941cbdfa66fc17fea64e387` |
| `logs/raw/37-hostmap-conc-512mib-c2-die7.txt` | 并发注册，die7，512 MiB | `db4adb5a1914e7bc4c09237dbefe09212a17ecd6e3b1bde3940efdc52067121b` |
| `logs/raw/37-hostmap-conc-512mib-c2-die7.json` | 同上，机械转写 | `3901ef4a2a0611cd97b6e5a0e98c2bd36e99b8dbcbc2ce6ed7bf7c96fc7209bc` |
| `agents/N_ngram/parse_hostmap_log.py` | 把 probe stdout 机械转成 JSON | — |

注：hostmap 那几份 `.json` 是**派生文件**——probe 本身不产出 JSON。转写脚本按 probe 自己
声明的列宽切列，不改写任何数字；唯一的加工是把被 18 字符列宽截断的 `ret` 名字按 probe
自带的错误表补全（`107017` → `ACL_ERROR_RT_INVALID_HANDLE`）。

复现命令（均通过槽位锁，未手设 `ASCEND_RT_VISIBLE_DEVICES`）：

```bash
# 0) 溯源自检（不需要 NPU）
bash tools/a3_chip.sh c0 --name ngram-verify --timeout 900 -- \
  python3 /work/agents/N_ngram/bench/bench_ngram_history.py --verify-verbatim

# 1) n-gram 主跑
bash tools/a3_chip.sh c0 --name ngram-full --timeout 1800 -- \
  python3 /work/agents/N_ngram/bench/bench_ngram_history.py --reps 30 --warmup 5 \
    --tag 20260921-a3prbench-c0 --out /work/agents/N_ngram/out \
    --json /work/agents/N_ngram/out/37-ngram-history-a3prbench-c0.json

# 2) host register 全量 probe（128 MiB）
bash tools/a3_chip.sh c1 --name hostmap-c1 --timeout 1800 -- \
  python3 /work/agents/N_ngram/bench/probe_engram_hostmap.py --size-mib 128 --gather-rows 4096

# 3) 并发两 die（512 MiB，两条命令同时起；本例在 section [6] 因 randbytes 上限 rc=1）
bash tools/a3_chip.sh c0 --timeout 300 -- python3 .../probe_engram_hostmap.py --size-mib 512 &
bash tools/a3_chip.sh c2 --timeout 300 -- python3 .../probe_engram_hostmap.py --size-mib 512 &
```

`--verify-verbatim` 结果（PASS，§3.3 的溯源依据）：

```
PR-16925-MASK         : MATCH  7 lines  from upstream_common_pr16925.py line 30
PR-16925-UPDATE       : MATCH  90 lines from upstream_common_pr16925.py line 72
PR-16925-SCALAR-LOOP  : MATCH  19 lines from upstream_common_pr16925.py line 112
upstream common.py    : OK  sha256 5ad16d70666fcc83a2780d4e543f338990cfcf7aec51b6cdaa888314fdee8cfa
ours engram_hash.py   : OK  sha256 d0811fd743e1a8c6e262dc8cd85b3ee53cb0f91424553ef17bc9b916cdeff44b
ours engram_jit_kernel: OK  sha256 3cc33a8365e6cdba5b253f53ee1793e3aafa1c96e3939bbe3878aee621bea935
```

---

## 5. 【实测】/【推断】/【未确认】

**【实测】**

1. n-gram：14 个尺寸 × 6 个臂，`torch.equal` 全 True；`U/J = 22.8×` @ n=128（1.6807 → 0.0736 ms）；
   `C/N = 1311.9×` @ n=128（10.3053 → 0.0079 ms）；`O/U ≈ 0.98–1.00`（拷贝忠实度）。
2. n-gram 结论对 **die（die3/die7）和线程数（OMP 1 vs 16）都稳健**：n ≤ 384 差值 < 10%。
3. host register：128 MiB 生产形状下 `aclrtHostRegister(MAPPED)` 与
   `aclrtHostRegisterV2(MAPPED|PINNED)` **都 ret=0 且设备侧逐字节读回**；
   Triton 裸指针读回也一致（#16925 的消费形状可用）。
4. `V2 PINNED`-only 会 ret=0 但**不发布设备地址**；`V2` 无 flag → 107000；
   把 V2 的 flag 字喂给 legacy → 207000。
5. 机器前提：`host_mem_pool=1`（dev0–15 全部）、`19e5:d803` × 16、CANN 9.1.0、driver 26.1.1。
6. 两 die 并发注册 512 MiB 无相互干扰（ret=0、读回一致，无 507011/207001）。
7. probe 的 `--size-mib` 上限 ~128 MiB（Triton control 的 `randbytes` 溢出），
   与注册能力无关——**这是 probe 的限制，不是机器/驱动的限制**。
8. probe 的 `header check: MISMATCH` 是解析器 bug（枚举行无 `#define`），
   三个 flag 常量实测全部与 `acl_rt.h` 一致。

**【推断】**

1. RFC 里的 **0.427 → 0.076 ms** 是**引擎内 per-step 相位计时**（`[bneck] hash` 相位，
   来源 `a21_reports/engram-jit-verified.md`），其上游侧口径比本 harness 的 `update()` 窄
   （不含 page-mirror 写入等）；**我们这一侧能对上**（0.0736 vs 0.076 ms），
   **上游那一侧对不上**（1.68 vs 0.427 ms）。同一份报告里的「纯 CPU 基准 298.9 → 12.98 µs
   （23×）」与本 harness 的 **22.8×** 吻合 ⇒ **比值可迁移，绝对值不可迁移**。
2. section [6] 里 ~45 ms 与 ~90 ms 两簇注册耗时像是**双峰**（同一 flags 在不同行分别
   表现为 45.5 / 89.9 ms）。未做重复测量，**不足以下结论**。
3. 128 MiB 档两条 API 无分岔 ⇒ 若 #16828 的分岔真实存在，**它只可能出现在满表尺寸**，
   即由 per-page metadata（~64 B/4 KiB）而非 API 语义决定。

**【未确认】**

1. **206 GiB 满表**：完全没跑（需要 ~206 GiB host 表 + 8 rank 一起注册）。
   §3.3 那一行的「分岔」判据**没有**被本日志证实或证伪。
2. **8 rank / 多容器同时注册满表**：#16828 的原始失败模式（`ret=207001` / `507011`）
   未复现，因为尺寸档位差了 ~400×。
3. **端到端 step 时**：本 harness 是函数级，不能推 TTFT / tok/s / ms-per-step。
4. **真实 tokenizer 的 compressed-token map**：输入是固定种子合成的（`build_compressed_token_map`
   需要 129k 词表），hash 表/乘子/素数/block table/page 内容都是生产形状，只有 id 值是合成的。
5. **A2 对照**：本机全是 910C（`host_mem_pool=1`），没有 `host_mem_pool=0` 的机器可对照。

---

## 6. 还缺什么

1. **206 GiB 满表注册**——这是 #16828 唯一真正的判据，也是 §3.3 第一行现在**唯一还空着**的格子。
   最小可行实验：单 rank 先把 206 GiB 表 mmap，再分别用两条 API 注册，记录 ret 与时延；
   然后上 8 rank 并发，看是否出现 207001 / 507011。`bench_host_register_scale.py` 已有缩放曲线
   （`logs/06-*`、`logs/29-*`），可以直接外推到满表，但**外推不是实测**。
2. **probe 的 `--size-mib` 上限**要被修掉才能真正测大表——目前 section [6] 的 Triton control
   会在 256 MiB 起崩（`randbytes` 溢出）。最小修法：control 只按 `gather_rows` 开缓冲，
   不要按整表尺寸开。**本次没有改 probe（红线：不改 `pr/` 其他文件）。**
3. **多 die 并发要上到满表尺寸**才有意义；本次 512 MiB 只能算「并发路径本身没坏」。
4. **重复测量**注册耗时的双峰（§5 推断 2）。
5. 若要写进 RFC 正文，**§3.3 第二行建议同时给出两个口径**：函数级 `update()` 的 22.8×，
   和上游 per-token 走法本身的 1312×，并显式说明 0.427 → 0.076 是引擎内相位计时。
   否则读者会拿 0.427 去对 1.68 而认为对不上。
