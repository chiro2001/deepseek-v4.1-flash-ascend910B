# 076 — ★★★ v9 构建（把 `ENGRAM×卸载` 修复烘焙进镜像）+ 起服前指纹门 + 一个我自己踩的递归坑

> 2026-09-22 21:1x–21:4x CST。执行：**主代理**（本机；不需要 NPU）。
> 前置：`075`（修复本体）。标记：**【实测】/【推断】**。

---

## 0. 一句话

A2 默认 `PATCH_MODE=baked` ⇒ **修复必须进镜像才生效**。本轮把 v9 的构建链路在本地
**不需要 docker 也能跑的那半段**全部验通（`check_checksums --materialize` + `verify_baked_tree`），
并在起服路径上加了**起服前指纹门**（5 秒把"镜像里没这个修复"抓成 `exit 2`），
另加一个 10 秒的 `check_image_fingerprint.sh` 用来回答"我这个镜像和发布包差哪些文件"。

---

## 1. 【实测】构建前置门：`check_checksums.py` 拦下了 md5 清单陈旧

改完两个 payload 后立刻跑：
```
[chk][FAIL] 4 个问题 ❌
    - STALE patches/MD5SUMS：engram_hash.py 清单=3a842bbb… 载荷=240c5a04…
    - STALE patches/vllm-ascend/MD5SUMS：… engram_hash.py …
    - STALE patches/MD5SUMS：engram_jit_kernel.py 清单=1add256a… 载荷=6668d3fe…
    - STALE patches/vllm-ascend/MD5SUMS：… engram_jit_kernel.py …
  ⇒ 修法：改了 patches/files/** 之后，把上面对应的 md5 行同步到清单里
```
⇒ 同步四条后 **`[chk] 三方一致 ✅`**（落位表 14 项 = 10 inst + 4 newf / 载荷 25 个文件 / MD5SUMS 30 条）。
★ 这就是 `074` 记的那类门在起作用：**清单不跟、构建会失败在最后一步**（v7→v8 白等过用户 10–20 分钟）。

## 2. 【实测】不需要 docker 的那半段构建，本地全跑了

```
python3 tools/check_checksums.py --materialize <D> --manifest <D>/manifest.tsv
  ⇒ [chk][NOTE] 已按落位表铺出模拟镜像树 <D>（14 项）
  ⇒ [chk] 三方一致 ✅
bash tools/verify_baked_tree.sh --root <D> --manifest <D>/manifest.tsv
  ⇒ [verify] 共 14 项，全部逐字节一致 ✅
```
★ 本机是 **x86_64**，A2 的基础镜像是 aarch64 ⇒ **真 `docker build` 只能在 A2 上做**（见 §5 的命令）。
但这半段已经覆盖了"落位表 / 载荷 / md5 三方一致性 / 逐文件逐字节"这四类最常见的构建失败。

## 3. 【实测】起服前指纹门（新增，`a2/scripts/serve_a2_offload.sh`）

**为什么要它**：修复在 `patches/files/` 里，而 A2 默认 `baked` ⇒ 容器读镜像里那份。
若镜像还是 v8，**修复一个字节都到不了容器**，而症状要等 ~30 分钟起服 + 一轮 replay 才出现。

**它做什么**：`ENGRAM=1` 时，把 host 侧权威副本（`$SHADOW/patches/files/*.py`）与镜像内
实际那份的 md5 对齐；不一致 ⇒ **`exit 2`**。

| 场景 | 实测结果 |
|---|---|
| 镜像**带修复**（本机用 stub + COPY 两个文件造了一个阳性镜像） | `✓ 指纹门 … engram_hash.py 240c5a04…` / `✓ … engram_jit_kernel.py 6668d3fe…` / `⇒ 两项一致` / 整脚本 **rc=0**（dry-run 走完全程） |
| 镜像**不带修复**或**本地没有该镜像** | 打印「镜像里 = … / 期望值 = …」+ 两条处置命令 ⇒ **rc=2**，且**不会再往下走** |
| 本地没有该镜像 | 先 `docker image inspect` 判断 ⇒ 只报"本地没有"并给 build 命令（**不触发 registry 拉取**） |

★ 这个门**设计上只挡 `ENGRAM=1`**（`ENGRAM=0` 时那段根本不执行）—— 因为它要防的正是
「`ENGRAM=1` + 卸载」这个组合。

## 4. ★★★【实测】我自己踩的坑：提示文字里的反引号 = 命令替换 ⇒ **脚本自我递归**

第一版指纹门的错误提示里写了（双引号内）：
```bash
echo "     (a) …先 `IMAGE_TAG=dsv41-a2:v9 bash scripts/build_image.sh`，" >&2
echo "         再 `IMAGE=dsv41-a2:v9 bash a2/scripts/serve_a2_offload.sh`；" >&2
```
⇒ **反引号在双引号里是命令替换**：`shellcheck` 意义上的 QUOTE 问题，运行期的实际行为是
**真的去执行了那两条命令**，而第二条正是**本脚本自己** ⇒ **进程树自己套了 10+ 层**
（`ps` 实测 10 个嵌套 `bash a2/scripts/serve_a2_offload.sh`），
输出里混进了 `build_image.sh` 的 `[build][FAIL] 基础镜像不存在…` 日志。

**修法**：提示文字改用**单引号**（`echo '…'`），脚本里已 grep 确认 `echo` 行 0 个反引号。
**教训（写进纪律）**：**错误提示可能被执行** —— 凡是要用户"照抄"的命令，放进 `echo` 时
**一律用单引号**，或者干脆 `cat <<'EOF'`。这与本日的"静默降级"是**同一类**问题：
代码的**副作用**与你**以为的**不一致。

## 5. A2 上的执行顺序（v9）

```bash
cd ~/projects/dsv41-a2-repro-kv8-offloading/deepseek-v4.1-flash-ascend910B
git fetch origin feat/kv8-dram-offload-pending && git pull --ff-only

# ① 10 秒：先看清"现在这个镜像"和发布包差哪些文件（只读、不拉镜像、不起容器里的服务）
bash a2/scripts/check_image_fingerprint.sh dsv41-a2:v8

# ② 3–6 分钟：build v9（基础镜像已在本地；不做 PGO）
IMAGE_TAG=dsv41-a2:v9 bash scripts/build_image.sh

# ③ 再查一次 v9（应全 ✓，且"带修复"那行必须 ✅）
bash a2/scripts/check_image_fingerprint.sh dsv41-a2:v9

# ④ 起服（起服前会先过指纹门）
cd <dsv41-release 路径>   # 若与上同目录可省略
IMAGE=dsv41-a2:v9 OFFLOAD_GB=85 ENGRAM=1 ENGRAM_DEVICE_INDEX=0 DRAFT_GRAPH=1 \
MAX_LEN=1048576 MAX_SEQS=4 NPU_OFFLOAD_HOST_MEM=registered \
bash a2/scripts/serve_a2_offload.sh
```
★ `scripts/serve_a2.sh` 的 `IMAGE` **默认值已升到 `dsv41-a2:v9`**（避免"名字是 v9、内容是 v8"）。
★ `static_kernel_cache/` 在**宿主**（`$CACHE/skcache/compile_outputs/`，不在镜像里）⇒
**换镜像不会触发 15–20 min 的冷编译**（`serve_a2.sh` 起服时会打印命中几个缓存文件）。

## 6. 诚实边界

| # | 事项 |
|---|---|
| 1 | **真 `docker build` 未在本机跑过**（架构不同）⇒ A2 上第一次 build 仍可能因基础镜像层差异失败；失败信息在 `[build]` 前缀里，可直接回报。 |
| 2 | **v9 相对 A2 现用 v8 的差异面未完全枚举**：`check_image_fingerprint.sh` 会把它**打出来**（14 项逐行）。若发现除两个 Engram 文件外还有差异，那说明 A2 的 v8 构建点比预期更早，**要重新过一遍判据**。 |
| 3 | **A3 端到端验证【仍未完成】**（`075` §4）：v9 里的修复还没在真权重 + 真卸载池上跑过 replay。⇒ A2 起服后的第一道实质门就是"同前缀两发触发取回"。 |
| 4 | 指纹门只覆盖**两个 Engram 文件**；`model.py` / `block_table.py` 等其余 12 项靠 `check_image_fingerprint.sh` 人工核对。 |
