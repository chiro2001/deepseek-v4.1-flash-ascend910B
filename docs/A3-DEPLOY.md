# A3 新机部署（8×910C）—— 从 `git clone` 到服务就绪

> 适用：**一台全新的 A3 机器**（没有镜像、没有模型、没有缓存）。
> 目标读者：拿到这个仓库、要在新机器上把它跑起来的人。
> 一键入口：`tools/deploy_a3.sh`（默认**只干跑**，`LAUNCH=1` 才真起服）。

---

## 0. 一句话流程

```bash
git clone <本仓> && cd <本仓>
bash tools/selfcheck_pkg.sh                  # 10 秒：包内一致性（镜像 tag / 补丁 md5 / MANIFEST）
bash tools/list_chips.sh                     # 只读：看哪 8 张卡是空的、属主是谁
MODEL=/path/to/model bash tools/deploy_a3.sh # ★ 干跑：把"容器里真会生效的配置"打全
MODEL=/path/to/model LAUNCH=1 bash tools/deploy_a3.sh   # 真起服
```

`deploy_a3.sh` 会按顺序做四件事，**任何一步不合格就停在那里**（不会带着问题起服务）：
**① 前置体检**（宿主/工具/仓库/镜像）→ **② 模型软链闭包**（断链、外部依赖目录）→
**③ 选卡**（默认只用 `npu-smi` 报**空闲**的卡）→ **④ 干跑 + 内容判据** →（可选）**⑤ 起服**。

---

## 1. 硬件与环境要求

| 项 | 要求 | 说明 |
|---|---|---|
| NPU | 8×910C（`npu-smi` 可见 16 个 die / 8 张卡） | 本包的 TP=8；卡数不同要显式 `TP=<n>` |
| 内存 | ≥ **1 TB** 可用 | 权重 + Engram 表（206 GiB）+ 静态内核编译缓冲；实测 a3-21 机器 2 TB，起服时 used ≈965 GB |
| 磁盘 | 模型 ≈**520 GiB** + 镜像 ≈25 GB + 缓存 | 见 §2（模型是软链拼的，闭包远大于单目录） |
| Docker | 可用且**当前用户有权限** | 新机最常见：用户不在 `docker` 组 ⇒ 所有 docker 命令 permission denied |
| 网络 | 能拉内网 registry（`quay.nju.edu.cn`） | 拉不动就找运维要镜像 tar，`docker load` 进去 |

---

## 2. ★★ 模型搬运：**不是一个目录**（新机器最容易踩的一条）

模型目录是**软链构造**的：目录里大部分 `.safetensors` 是**符号链接**，指到同级的其它产物目录。
只搬"看起来那一个目录"⇒ 容器里**读到断链**，而且报错发生在**权重加载中途**（白等 10+ 分钟）。

实测（a3-21，`v41-w4a8-engram-dr-vision-qrot-mtpq`）：

| 目录 | 体积 | 角色 |
|---|---|---|
| `…qrot-mtpq`（入口目录） | **926 MB** | 只有 config / index / qrot 差异件 |
| `…engram-dr-vision-mtpq` | 29 MB | 主权重所在 |
| `…engram-dr-vision` | 1.6 GB | |
| `…dspark-mtpq` | 11 GB | |
| `…dspark` | 30 GB | |
| `v41-w4a8-stage1` | **273 GB** | |
| `engram-int8` | **206 GB** | Engram 表 |
| **合计（闭包）** | **≈520 GiB** | ← **要一起搬的量** |

**怎么知道要搬哪些**：`deploy_a3.sh` 的 §② 会**自动列出**外部依赖目录与断链（不搬则直接 rc=2）。
也可以手工看：

```bash
find /path/to/model -type l ! -exec test -e {} \; -print   # 先看有没有断链
find /path/to/model -type l -printf '%l\n' | sed 's|/[^/]*$||' | sort -u  # 链接都指向哪些目录
MODEL=/path/to/model SHOW_SIZES=1 bash tools/deploy_a3.sh  # 连体积一起算（慢）
```

搬运建议（任选）：`rsync -aH --info=progress2 <源>/ <目标>/`（`-H` **必须**：保留硬链接，
否则 engram 表会膨胀）、或按上表逐个 `rsync`。

---

## 3. 镜像：**不需要自己 build**（A3 走 mount 模式）

```bash
docker pull quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3
# 或让 deploy 脚本代劳：
MODEL=... PULL=1 bash tools/deploy_a3.sh
```

为什么不需要 `build_image.sh`：A3 用**官方镜像 + `PATCH_MODE=mount`**
（把本仓 `patches/files/*` 挂进容器），所以官方镜像就够。

> ⚠️ **必须 `PATCH_MODE=mount`**：若设成 `baked`，官方镜像里没有本包的补丁，
> 而服务**照常起得来、不报任何错** —— 只是所有优化（含 Engram device-index）**静默没生效**。
> `serve_a3.sh` 已经把 `mount` 设成默认；`deploy_a3.sh` 的干跑会核对这条。

---

## 4. 选卡与口径

```bash
bash tools/list_chips.sh            # 全貌：每张卡的 HBM、占用、进程属主
bash tools/list_chips.sh --free     # 只输出空闲卡号（可直接当 DEVS）
```

* **卡不是你的就不要用**：A3 是共用机。`deploy_a3.sh` 默认从空闲卡里选前 8 张并**大声打印**；
  你也可以显式 `DEVS="8 9 10 11 12 13 14 15"`。
* `scripts/serve_a3.sh` 在真起服前会**再查一次**占用；确有**你自己的**残留进程时用 `ALLOW_BUSY=1`。
* **`DROPCACHE` 默认 0**（`serve_a3.sh` 已按平台给默认值）：
  `serve_a2.sh` 的默认是 `1` = 起服前 `echo 1 > /proc/sys/vm/drop_caches`，
  那是**整机**操作，在共用 A3 上会连带清掉**别人**的 page cache。A2 独占机才用 1。
* **`DRAFT_GRAPH` 默认 0，别随手开**：A3 上实测开它 ⇒ **接受长度 A 掉到 1.06–1.08**
  （draft 完全不产出 = 静默失效），而 `ms/step` 反而更好看（每步只出 1.08 个 token 而不是 2.85 个）
  ⇒ **真实吞吐慢 2.2×**。要试必须用 `bash tools/draft_graph_guard.sh` 验 **A ≥ 1.3**。
  判据永远是 **(A, tok/s) 这一对**，不是 `ms/step` 单值。
* **上下文/并发**：`deploy_a3.sh` 默认取 **A3 已验证口径** `MAX_LEN=133120 MAX_SEQS=32`
  （A3 历史臂全部是这一档，见 `shadow-pkg/results/r8_*/serve_cmd.txt`）。
  `serve_a2.sh` 自身的默认是 `MAX_LEN=1048576`（**A2 生产口径**），在 A3 上**未验证** ——
  要 1M 请显式给，并同时把并发压下来（A2 的 1M 生产口径是 `MAX_SEQS=4`）：

  ```bash
  MODEL=... MAX_LEN=1048576 MAX_SEQS=4 bash tools/deploy_a3.sh
  ```

  ★ 1M 下还有一条**必须遵守**的边界：单请求 `prompt + max_tokens ≤ max_len − 32`
  （`max_len−6` 附近会撞上 spec-decode 的越界缺陷 ⇒ 引擎 8 rank 全崩）。

---

## 5. 起服失败怎么读（按出现频率排序）

| 现象 | 根因 | 处置 |
|---|---|---|
| `docker: permission denied` | 用户不在 docker 组 | `sudo usermod -aG docker $USER` 后重新登录（或 `newgrp docker`） |
| `image … not found` | 镜像没拉 | `PULL=1` 重跑，或手工 `docker pull` |
| worker 加载期报 `No such file or directory`（权重） | **模型只搬了一个目录**（软链断） | 按 §2 补齐闭包；`deploy_a3.sh` §② 会提前拦住 |
| `[migrate]` 长时间不动、起服卡住 | 历史已知问题（绑核路径） | 用 `CPU_BIND=0` 起服（见 `docs/RELEASE-NOTES.md` 的 known issue） |
| 反复在 worker 里起不来、`EH0012` | Engram 相关（历史签名） | 先 `ENGRAM=0` 把服务起起来定位，再回到 `ENGRAM=1` 排查 |
| 服务起来了但**慢** | 补丁没生效（`baked` / 没挂文件） | 看干跑日志里的 `PATCH_MODE=mount` 与 `MOUNTS(...)` 条数 |
| 吞吐/时延看着正常但**答案异常** | 见 §6 的判据纪律 | 用 `(A, tok/s)` + 文本原文一起判 |

### 5.1 ★ 「HCCL 建链失败」专章（新机上第二常见，且最容易被误判成"我们的 bug"）

**症状**（起服在**模型初始化**阶段就崩，8 个 rank 全报同一条）：

```
.../quantization/methods/w4a8/w4a8.py: self.moe_all_to_all_group_name =
    backend.get_hccl_comm_name(local_rank)
RuntimeError: ... hcclCommInitRootInfoConfig(...), error code is 1
ERR02200 DIST call hccl api failed.
Communication_Error_Ranktable_Detect(EI0015): ... No rank in the communicator can
connect to the root node within the timeout period. List of unconnected ranks: "[3,]"
```

**判读要点**：注意 `unconnected ranks: "[3,]"` —— **是某一个 rank 掉队**（不是全部）。
一个 rank 起不来，其余 7 个就会一直等它直到超时。真正的根因候选只有三类：

| 类别 | 机制 | 怎么认 |
|---|---|---|
| **那张卡当时不可用** | 别人的进程占着 / 无进程但 HBM 未释放 / 你自己的残留 | `npu-smi info` 看进程表与 HBM |
| **那个 rank 被 OOM 杀了** | 起服峰值需要 ≈**1 TB 可用**（8 rank × 权重页 + Engram 表 206 GiB） | `dmesg -T \| grep -iE "oom\|killed process"` |
| **HCCL 选错网卡** | 多网卡机器上自动挑到走不通的那张 | `ip -o -4 addr show`：真实网卡 >1 张就要显式 `HCCL_SOCKET_IFNAME=<网卡名>` |

**一条命令收齐证据**（只读；`RUN_HCCL_TEST=1` 才会真占卡）：

```bash
DEVS="8 9 10 11 12 13 14 15" SERVE_LOG=/path/to/serve.log bash tools/diag_a3_hccl.sh
DEVS="..." RUN_HCCL_TEST=1 bash tools/diag_a3_hccl.sh   # 额外真跑 8 卡 HCCL all-reduce（决策性证据）
```

`diag_a3_hccl.sh` 会：① 逐卡查占用（含"无进程但 HBM 高"）② 列网卡并给 `HCCL_SOCKET_IFNAME` 写法
③ 查 `host_mem_pool`（A3 应为 1）④ 查 OOM ⑤ 从 `serve.log` 数**每个 rank 的行数**
（行数明显少的那个就是掉队的 rank —— 再看它最后停在哪一步：device 初始化前=卡被占／权重加载=内存／HCCL=网卡）。

**参考基准（a3-21 工作机）**：单张真实网卡 `enp196s0f0 192.168.45.21/21`、`host_mem_pool=1`、
2 TB 内存。**新机器与这三项任何一项不同，都要先怀疑它。**

---

## 6. 起服后的验收（**别只看"起来了"**）

```bash
bash tools/attach_test.sh                    # 端到端：文本 / 工具调用 / 图片
bash tests/t_quote.sh                        # 时延（ms/step）
bash tools/draft_graph_guard.sh              # 只在 DRAFT_GRAPH=1 时必须跑（判据 A ≥ 1.3）
bash tools/selfcheck_pkg.sh                  # 包内一致性（起服前跑过一次也行）
```

**判据纪律（本仓反复栽过的坑）**：
1. **"我传了这个变量" ≠ "这个变量生效了"** —— 判据要落在**实际生效后的可观测痕迹**上
   （干跑日志里的 `PATCH_MODE`/`MOUNTS`、容器内文件 md5、`SpecDecoding` 的 A 值）。
2. **性能数字必须带相位**（单发 quote vs 混合批），否则两块数字不可比。
3. **文本对 ≠ 行为不变**：`A(接受长度)` 是更灵敏的判据（对亚文本级数值污染敏感），
   凡动 attention/KV/量化，必须同时报 `(文本, A, tok/s)`。

---

## 7. 与其它脚本的关系

| 脚本 | 用途 |
|---|---|
| `tools/deploy_a3.sh` | ★ **新机器**：前置体检 + 模型闭包 + 选卡 + 干跑（+可选起服） |
| `tools/selftest_deploy_a3.sh` | `deploy_a3.sh` 的沙箱自测（零真机；每条门各一个用例） |
| `scripts/serve_a3.sh` | 引擎入口：选卡校验 + 全部开关（`DEVS` 必填） |
| `scripts/serve_a2.sh` | 引擎本体（A2/A3 共用） |
| `tools/list_chips.sh` | 只读选卡助手 |
| `tools/check_model_dir.sh` | 模型目录自检（配置/权重/软链健康度） |
| `tools/draft_graph_guard.sh` | `DRAFT_GRAPH=1` 的验收闸（A ≥ 1.3） |
