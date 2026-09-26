# 055 — A2 上线路径**打通**：`make_shadow_pkg.sh` + 补丁目录自动识别（此前 A2 上第一条命令就会卡住）

> 2026-09-22 11:4x–11:5x CST。执行：**主代理**。**不占任何 die**（全部在本地做，只跑 dry-run）。
> 起因：核对"给 A2 的确切命令"时发现——**那条命令在 A2 上跑不起来**。
> 标记：【实测】/【推断】/【未确认】。

---

## 0. 一句话

`a2/scripts/serve_a2_offload.sh` 依赖 **shadow-pkg**，而 **shadow-pkg 只存在于 A3 的开发机上**
（`~/projects/dsv41-upstream-pr/shadow-pkg`，是手工改出来的，**从没进过发布包**）。
⇒ 也就是说：**即使 §0.1 的池后端探测全绿，A2 上第二条命令也会立刻卡在 `⚠ 找不到 shadow-pkg`**。
**现已补上两个件**，并在本地把整条链**dry-run 跑通**（含"不污染 dsv41-release"的验证）。

---

## 1. ⛔ 查出来的两个缺口

### 1.1 缺口一：A2 上没有 shadow-pkg

`serve_a2_offload.sh` 第 161 行：
```bash
SHADOW=${SHADOW_PKG:-$HOME/projects/dsv41-upstream-pr/shadow-pkg}
[ ! -d "$SHADOW" ] && { echo "⚠ 找不到 shadow-pkg（$SHADOW）—— 请设 SHADOW_PKG=<路径>"; exit 2; }
```
而 A3 上的 shadow-pkg 与 `dsv41-release/scripts/serve_a2.sh` 差 **274 行 / 5 个 hunk**
（我 diff 过：`~/tmp/serve_a2.shadow.diff`）。那 5 个 hunk 里**只有 5 处是需要的**：

| # | 位置 | 作用 |
|---|---|---|
| ① | 容器 `docker run` 的 `-e` 列表 | 让 `NPU_OFFLOAD_HOST_MEM` 进容器（池后端） |
| ② | `MOUNTS` 末尾 | 挂 `scheduler.py` / `cpu_npu.py` / `pgp_*` / `dsa_v41.py` |
| ③ | `serve_cmd.txt` 之后 | `KV_ARGS_EXTRA` 含单引号 ⇒ **拒绝注入**（否则 inner.sh 字面量被破坏） |
| ④ | `inner.sh` 的 heredoc 里 | 在**宿主**上展开成字面量，容器里只是普通 export |
| ⑤ | `serve_v2.sh` | 把 `KV_ARGS_EXTRA` 变成 vllm 的 CLI 参数 |

其余几块（`CPU-BINDING-FIX` / `L3-PGP` / `R8_*` 等）是**别的任务**当时加的，与 A2 上线无关。

### 1.2 缺口二：补丁目录写死在"发布包布局"上

`serve_a2_offload.sh` 原来写死 `PDIR="$REPO/a2/patches"`。
但**开发工作区里补丁在 `a2/publish/`、发布包里才叫 `a2/patches/`** ⇒
在工作区直接跑会报 `✗ 缺补丁 ~/projects/dsv41/a2/patches/0001-...`（**我实测踩到**）。

---

## 2. ✅ 修法（两个件，都可复现）

### 2.1 新增 `a2/scripts/make_shadow_pkg.sh`

**在 A2 本机**从 `dsv41-release` 自己造 shadow，不依赖 A3、不改任何生产脚本：

1. 在 `$DST` 建树：**除 `scripts/` 与 `patches/` 外全部软链**到 `$PKG`（省空间、防漂移）；
2. `scripts/` 拷成真目录；
3. ★ `patches/` 也建成**真目录**（`files/` 尤其要可写）—— 否则
   `serve_a2_offload.sh` 会把补丁 `cp` 进软链目标、**污染 `dsv41-release`**（我实测污染过，已还原）；
4. 对副本做 **5 处精确锚点插入**（见 §1.1 表），每处**锚点必须恰好命中一次**，否则 **fail-closed 且不落盘**；
5. 自检：4 条 `grep` 断言（块已插入 / `-e` 已插入 / int8 env 已透传 / serve_v2 已认 `KV_ARGS_EXTRA`）。

**用法**（A2 上，一条命令）：
```bash
PKG=<dsv41-release 路径> DST=$HOME/shadow-pkg bash a2/scripts/make_shadow_pkg.sh
```

### 2.2 `serve_a2_offload.sh`：补丁目录自动识别

```bash
for _c in "$REPO/a2/patches" "$REPO/a2/publish" "$A2DIR/patches" "$A2DIR/publish"; do
    [ -f "$_c/0001-offload-scheduler.patch.py" ] && PDIR=$_c && break
done
# 全找不到 ⇒ 打印试过的四个路径 + 提示用 PDIR=<dir> 直接指定，然后 exit 2
```

---

## 3. 本地 dry-run【实测】

```
$ PKG=$PWD/dsv41-release DST=$HOME/tmp/shadow-test bash a2/scripts/make_shadow_pkg.sh
[make_shadow] 软链了 29 项（scripts/ 除外）
[make_shadow] ✓ scripts/serve_a2.sh: 1283 → 1339 行（+56），插入 4 处
[make_shadow] ✓ scripts/serve_v2.sh: 98 → 105 行（+7），插入 1 处
✓  ①/②/③/④ 块已插入
✓  -e NPU_OFFLOAD_HOST_MEM 已插入
✓  int8 env 透传已插入
✓  ⑤ serve_v2 已认 KV_ARGS_EXTRA
✓ shadow-pkg 造好了

$ DRY=1 SHADOW_PKG=…/shadow-test MODEL=<模型目录> bash a2/scripts/serve_a2_offload.sh
✓ 补丁目录：…/a2/publish
✓ 四个补丁文件已就位
✓ 补丁已复制到 …/shadow-test/patches/files/offload_dsv41
[DRY] 将要执行：
  OFFLOAD_GB=56 MAX_LEN=131072 MAX_SEQS=16 \
  KV_ARGS_EXTRA='--prefix-match-unit 32 --kv-transfer-config {...}' \
  bash scripts/serve_a2.sh

$ cd dsv41-release && git status --porcelain      # ★ 零输出 = 没污染
```

**三条都验到了**：shadow 造得出来、整条链认得出补丁、**发布仓一个字节没被写**。

---

## 4. A2 上线路径（★ 现在真的是"三条命令"A）

```bash
# 0) 池后端探测（不占卡、不加载模型；服务在跑也不用停）
A2_CONTAINER=dsv41-a2 A2PROBE_FLOOR_GIB=300 LIGHT=1 bash a2/scripts/a2_one_shot_probe.sh
#    ★ 看 `★ 注册内存的设备往返判据 = True/False`（H2H 通过不算数）

# 1) 造 shadow-pkg（一条命令，可复现；失败时 fail-closed 不留半成品）
PKG=<dsv41-release> DST=$HOME/shadow-pkg bash a2/scripts/make_shadow_pkg.sh

# 2) 干跑（不起服务）→ 再起服
DRY=1 SHADOW_PKG=$HOME/shadow-pkg MODEL=<模型目录> bash a2/scripts/serve_a2_offload.sh
SHADOW_PKG=$HOME/shadow-pkg MODEL=<模型目录> OFFLOAD_GB=56 MAX_LEN=131072 MAX_SEQS=16 \
  NPU_OFFLOAD_HOST_MEM=registered bash a2/scripts/serve_a2_offload.sh
```

**起服后先跑自检**（脚本末尾会打印；任一为 0 就停，别压测）：
```bash
grep -c 'P1_pinned.*ret=0'          <serve.log>   # 期望 8
grep -c 'D2_offload'                <serve.log>   # 期望 >0
grep -c 'alignment_chunk_count.*8'  <serve.log>   # 期望 >0（per-group 生效）
grep -c 'P2_poolsizing'             <serve.log>   # 期望 >0（L1 生效）
grep -a 'P2_WORKER_HOST_BYTES'      <serve.log>   # ★ 宿主实占
```

---

## 5.0 ★★★ 追加验证（13:0x）：**从 GitHub 全新 clone 跑一遍整条链**（模拟 A2 现场）

§3 的 dry-run 用的是**本地工作区**（`a2/publish/`），而 A2 拿到的是**发布包**（`a2/patches/`）。
两者布局不同 ⇒ 我用 `git clone` **真拉了一份发布仓**（`feat/kv8-dram-offload-pending`）再跑一遍：

```
$ git clone --depth 1 -b feat/kv8-dram-offload-pending <repo> dsv41-release
$ md5sum 三个脚本
  40e495e83d198393ea544215dbc4fd50  a2/scripts/a2_one_shot_probe.sh
  9cfd7a9fd2dea5c9284e4ca634d2364b  a2/scripts/make_shadow_pkg.sh
  dc8d30679be172badb58c71090e19f3a  a2/scripts/serve_a2_offload.sh
  ★ 与工作区 md5 **逐字一致** ⇒ 发布仓里的就是验过的那份，没有"发布时掉包"

$ PKG=$PWD DST=$HOME/tmp/relcheck/shadow bash a2/scripts/make_shadow_pkg.sh     # 第二步
  ✓ 造好（5 处锚点插入 + 4 条 grep 自检全过）

$ DRY=1 SHADOW_PKG=… MODEL=… bash a2/scripts/serve_a2_offload.sh               # 第三步（档 B）
  ✓ 补丁目录：…/dsv41-release/a2/patches          ← ★ 自动识别到**发布包布局**（不是 publish/）
  ✓ 四个补丁文件已就位
  [DRY] KV_ARGS_EXTRA='--prefix-match-unit 32 --kv-transfer-config {...}'

$ DRY=1 … KV8_SWA=1 KV8_RING_FP16=1 bash a2/scripts/serve_a2_offload.sh        # 第三步（档 C）
  ⚠⚠ 你开了 int8 但 APC_ALIGN=0 ⇒ 自动置 3
  ⚠⚠ 你开了 int8 + 图模式但 GRAPH_SAFE=0 ⇒ 自动置 1
  ★ int8 档 C: SWA=1 ring16=1 / ★ APC_ALIGN=3 / ★ GRAPH_SAFE=1
  ⇒ ★ **静默 no-op 的两个门都按预期自动补上并告警**（§7 修的那条）

$ cd dsv41-release && git status --porcelain
  （零输出）⇒ ★ **发布仓一个字节没被写**
```

★ 还核了一条：全新 clone 里 `a2/patches/kv8-graphsafe/dsa_v41.py` = **`94aeebb757d6d5708268754481a05e0a`**
（与 `ARTIFACT-IDENTITY.md` 台账、与 8 卡实测件**逐字一致**）。

**⇒ 结论**：A2 拿到的三个脚本 + 两份补丁目录，**在发布包布局下能完整走通**（探测→造 shadow→干跑/起服），
且**档 B 与档 C 两条路径都验过**。

## 5. 诚实边界

1. ★ **`make_shadow_pkg.sh` 只在本地（<workstation>）dry-run 验证过** ⇒
   **在 A2 上造 shadow 这一步【未确认】**（A2 的 bash/python3/docker 版本与路径可能不同）；
2. **shadow 的内容与 A3 那份不等价** —— 我只注入 A2 上线**必需**的 5 处，
   A3 那份还含 `CPU-BINDING-FIX` / `L3-PGP` / `R8_*` 等**别的任务的**注入块；
   ⇒ **不要**拿 A3 的 arm 结论直接套到 A2 的 shadow 上（这条同样是"身份"问题，见 `patches/ARTIFACT-IDENTITY.md`）；
3. 起服的**实际效果**仍未在 A2 上验过（§4 只是"路径可走通"，不是"跑得对"）。

## 5bis. ★★★ 2026-09-22 15:2x：又查出**两个**同源缺口（比 §1 那两个更隐蔽）

### 缺口 ②：档 C/D 需要 **7 个**挂载件，发布包里只发布了 **1 个**

`serve_a2_offload.sh` + `make_shadow_pkg.sh` 原先只覆盖 **4 个卸载件 + `dsa_v41.py`**；
而档 C/D 在 8 卡上真跑时挂的是 **7 个整文件**（来源：`R_8card_int8` 的 `arm.out` 挂载台账）：
```
core/deepseek_v41.py                       ★ 不在发布包
core/kv_cache_interface.py                 ★ 不在发布包
models/deepseek_v41/model.py               ★ 不在发布包
models/deepseek_v41/compressor.py          ★ 不在发布包
ops/triton/compressor/compressor_triton.py ★ 不在发布包
attention/kv8_prefill_triton.py            ★ 不在发布包
attention/dsa_v41.py                       ✓ 唯一在的（kv8-graphsafe/）
```
⇒ ★★ **A2 拿到发布包，起不了档 C/D** —— 与 §1 的 shadow-pkg 缺失、以及 `0004` 的 ②c 补丁缺失**同一类**。

★ **而且就算找到那 6 个文件，也有一格坑**：`pkg-kv8pf` 里的 `core/deepseek_v41.py`（`b9ae8151`）
的槽位容量是 `max(kv+index, aliases)` —— **不含 draft 项** ⇒ 真权重 13 组会 raise
`Aurora DSpark geometry must match target SWA and fit its existing slot`
（**正是 `050` 记录的那 6 条 raise 臂**）。必须用 R 的 **`9db8e27c`**
（`capacity = max(kv+index, _alias_max, _draft_size)`）。

**修法（已完成）**：
1. 新增 **`a2/publish/kv8-int8-pkg/`**（6 个整文件 + README），md5 **与 `arm.out` 台账逐字相同**；
2. `prepare_publish.sh` MAP 补 7 项 ⇒ 进发布仓的 `patches/kv8-int8-pkg/`；
3. ★ **`make_shadow_pkg.sh` 接入挂载块**：检测到 `A2_KV8` / `A2_KV8_SWA` / `A2_RING_FP16`
   任一开启 ⇒ 把这 7 个文件一起挂进去，**缺文件直接 die**（不静默降级成档 B）；
4. `check_artifact_identity.sh` LEDGER 补这 7 条（标 `PASS` —— 由 `sg-c-c-graph-b` /
   `sg-c-d-graph` 两条 8 卡 PASS 臂 + `ddi-*` 单 die 臂背书）。

### 缺口 ③：★ **dry-run 从来没验到挂载块**（验证盲区）
`DRY=1` 在 `serve_a2_offload.sh` 里**调用 shadow 之前就 `exit 0`** ⇒
shadow 的 `MOUNTS` 组装**一次都没跑过** ⇒ 于是「挂载块到底生不生效」**在 dry-run 里完全没被验证**。
⇒ 修法：`DRY=1` 现在**转调 shadow 自己的 `DRY_RUN=1`**，把**真实挂载清单**打出来。
★ 顺手踩到一个 rc=127：必须用 **`$SHADOW` 自己的 `scripts/`**，
不能用 `$REPO`（那是**本脚本所在仓**，与 shadow 不是同一个目录）。

**实测（全新 clone + 档 C dry-run）**：
```
[serve_a2] [A2-INT8] 已挂 7 个整文件件（6 个来自 …/patches/kv8-int8-pkg/vllm_ascend + dsa_v41.py）
[a2-dry] MOUNTS(24): … -v …/kv8-int8-pkg/vllm_ascend/core/deepseek_v41.py:…/core/deepseek_v41.py:ro …
         （7 个 int8 件逐条可见，路径全部正确）
```

★★ **三个缺口（§1 的 shadow-pkg / §5bis 的 7 件 / 0004 的 ②c 补丁）的共同点**：
**都不会在任何测试里报错**，只会让 A2 上线的人**在第一步卡住**。
⇒ 这就是为什么「**发布包级验证**」必须**独立于「臂级验证」**做一次（本轮做的就是这件事）。

## 5ter. ★★★ 2026-09-22 16:5x：把「档位门」在**发布包布局**下逐档验了一遍

**起因**：档位门是 16:4x 才加的（那轮只在**工作区**验过）；
而 A2 拿到的是**发布包** ⇒ **必须在发布包布局下再验一次**（本文件 §5.0 记的就是这个道理）。

**做法**：`git clone` 最新发布仓 → `make_shadow_pkg.sh` 造新 shadow → 逐档 `DRY=1` 跑。

| 档 | 环境变量 | 脚本自报 | rc |
|---|---|---|---|
| **档 B** | （无） | 「档位 : B」 | 0 |
| **档 C** | `KV8_SWA=1 KV8_RING_FP16=1` | 「档位 : C」 | 0 |
| **档 D** | `KV8_SWA=1 KV8_RING_FP16=1 KV8_FULL=1 KV8_PREFILL=1` | 「档位 : D」 | 0 |
| 反例 1 | `KV8_SWA=1`（缺 RING） | 「未经实测的组合：UNVERIFIED-swa-without-ring」 | ★★ **2** |
| 反例 2 | `KV8_FULL=1`（缺 SWA） | 拒绝 | 2 |
| 反例 3 | `KV8_PREFILL=1 KV8_FULL=1`（缺 SWA/RING） | 拒绝 | 2 |

⇒ ★ **正例 rc=0 且自报档位；反例 rc=2 且列出三条已验证档位** —— 门是**响亮的**，不是只打印警告。
⇒ 三档的**环境变量组合**与 `logs/048`/`050` 的实测指纹一一对应：B/C = 427,643、**D = 485,610**。

**一条给 A2 上线的提醒**：起服后**先看两行** ——
```
  ★★ 档位        : C（**容量指纹**：B/C=427,643，D=485,610 —— 起服后核对）
  GPU KV cache size: …    ← ★ 与上面的指纹对一下；档 D 若没拿到 485,610 就是静默降档
```

## 6. 交付

| 件 | 位置 | md5 |
|---|---|---|
| shadow 生成器 | `a2/scripts/make_shadow_pkg.sh` | 见 `check_artifact_identity.sh` |
| 起服脚本（补丁目录自动识别） | `a2/scripts/serve_a2_offload.sh` | 同上 |
| 本日志 | `a2/logs/055-20260922-a2-launch-path.md` | — |

---

## 7. ★★ 顺手挖出的**第三个**静默 no-op：`VLLM_V41_*` 在宿主上导出**到不了容器**

做完 §2 之后我顺手核对"档 C 到底能不能从这个脚本起来"，发现一个**同类的静默失败**：

```bash
# serve_a2_offload.sh 原来这样导出（宿主上）：
export VLLM_V41_KV8_SWA="$KV8_SWA" VLLM_V41_RING_FP16="$KV8_RING_FP16" \
       VLLM_V41_KV8="$KV8_FULL"    VLLM_V41_KV8_PREFILL="$KV8_PREFILL" \
       VLLM_V41_APC_ALIGN="$APC_ALIGN" VLLM_V41_KV8_GRAPH_SAFE="$GRAPH_SAFE"
```
**这四个变量只在容器内有意义**（容器里的 python 读 `os.environ`），
而容器环境是由 shadow 的 `inner.sh` 在**容器内**建立的
⇒ **宿主上导出它们，一个字节都到不了容器**。

⇒ 后果：`KV8_SWA=1 bash serve_a2_offload.sh` **跑得起来、日志上还写着"★ int8 档 C"**，
但容器里 `VLLM_V41_KV8_SWA` 根本不存在 ⇒ **实际跑的是档 B**。
这**正是本项目反复踩的那一类**（`048` 的"影子包 `grep -c VLLM_V41 = 0` ⇒ 不进 inner.sh 即静默 no-op"）。

### 7.1 修法（两道门，一前一后）

| # | 门 | 行为 |
|---|---|---|
| **起服前** | `serve_a2_offload.sh` 新增**自检门** | 开了 `KV8_*` 但 shadow 的 `serve_a2.sh` **不认 `A2_*`** ⇒ **⛔ 拒绝起服**（exit 2），提示用当前生成器重造 shadow；★ 该门**放在 `cp` 补丁之前** ⇒ **零副作用**就拒绝 |
| **起服后** | 自检清单补一条 | `grep -m1 -a 'VLLM_V41_KV8_SWA=' "$(dirname <serve.log>)/inner.sh"` —— ★ 变量名与取值**真的进了容器**才算数 |

同时把宿主上导出的**名字对齐**到 shadow 认的那一套：`A2_KV8_SWA` / `A2_RING_FP16` / `A2_KV8` /
`A2_KV8_PREFILL` / `A2_APC_ALIGN` / `A2_GRAPH_SAFE` / `A2_KV8_GRAPHSAFE`（挂 `dsa_v41.py` 用）。
生成器的 `inner.sh` 注入块把它们在**容器内**翻成 `VLLM_V41_*`。

### 7.2 两臂实测【实测】

```
臂1（正例）：档 C + 当前生成器造的 shadow
  ★ int8 档 C: SWA=1 ring16=1
  ✓ 补丁目录：…/a2/publish
  [DRY] 将要执行：… KV_ARGS_EXTRA='--prefix-match-unit 32 --kv-transfer-config {...}' …

臂2（反例）：档 C + 一个把 A2_* 改名成 XX_* 的 shadow
  ⛔ 你开了 int8（KV8_SWA=1 KV8_FULL=0 RING_FP16=0），
     但这个 shadow-pkg **不认 A2_* 环境变量** ⇒ int8 会**静默失效**（跑起来是档 B）。
     修法：用当前版本的生成器重造 shadow：PKG=… DST=… bash …/make_shadow_pkg.sh
  $ ls …/shadow-old/patches/files/offload_dsv41
  ls: cannot access '…': No such file or directory     ← ★ 零副作用（门在 cp 之前）
```

★ **判据有判别力**：同一个脚本、同一组 `KV8_*`，只有 shadow 的变量名不同 ⇒ 一边过、一边被拒。
