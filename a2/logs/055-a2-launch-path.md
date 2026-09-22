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

## 5. 诚实边界

1. ★ **`make_shadow_pkg.sh` 只在本地（<workstation>）dry-run 验证过** ⇒
   **在 A2 上造 shadow 这一步【未确认】**（A2 的 bash/python3/docker 版本与路径可能不同）；
2. **shadow 的内容与 A3 那份不等价** —— 我只注入 A2 上线**必需**的 5 处，
   A3 那份还含 `CPU-BINDING-FIX` / `L3-PGP` / `R8_*` 等**别的任务的**注入块；
   ⇒ **不要**拿 A3 的 arm 结论直接套到 A2 的 shadow 上（这条同样是"身份"问题，见 `patches/ARTIFACT-IDENTITY.md`）；
3. 起服的**实际效果**仍未在 A2 上验过（§4 只是"路径可走通"，不是"跑得对"）。

## 6. 交付

| 件 | 位置 | md5 |
|---|---|---|
| shadow 生成器 | `a2/scripts/make_shadow_pkg.sh` | 见 `check_artifact_identity.sh` |
| 起服脚本（补丁目录自动识别） | `a2/scripts/serve_a2_offload.sh` | 同上 |
| 本日志 | `a2/logs/055-20260922-a2-launch-path.md` | — |
