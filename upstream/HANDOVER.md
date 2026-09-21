# HANDOVER —— 交接快照（2026-09-21 13:0x CST）

> **给下一个接手的人（或压缩上下文后的自己）**。
> 先读这份 → 再读 [`AGENTS.md`](AGENTS.md)（规则/红线/环境）→ [`ONBOARDING.md`](ONBOARDING.md)（命令级）。
> 日志索引看 [`logs/README.md`](logs/README.md)。

---

## 0. 一句话现状

**上游材料全部就绪但一个都没提交**（用户明确禁止主动发 PR/issue）；
**codex 已能直连 A3 本地模型**（本轮最大交付，已推送到用户的 GitHub 仓）；
**A3 服务在跑且健康**；**单卡机已交回**；**A2 我连不上**（只有用户粘贴的数据）。

---

## 1. 资源与可达性（**先确认这个，否则很多命令会白跑**）

| 资源 | 状态 | 怎么访问 |
|---|---|---|
| **A3 `A3-node1`** | ✅ 可用 | `ssh A3-node1`（免密已配） |
| **A3 上的服务** | ✅ `health=200` | `http://127.0.0.1:8020`，容器 `dsv41-a3`，模型 `deepseek-v41` |
| **A3 上的 codex** | ✅ 已配好、可直接用 | `~/.codex/config.toml` 指向 `127.0.0.1:8020`（`wire_api=responses`） |
| **单卡 910C** | ⛔ **已交回**，不可达 | 成果在 `~/projects/dsv41/remote-910C-20260921/`（31 MB / 3217 文件） |
| **A2** | ⛔ **我连不上**（banner timeout） | 只能靠用户粘贴；`logs/31` 就是这么来的 |
| `origin` 上的 fork | ✅ `chiro2001/vllm-ascend`，3 个分支已推 | — |
| 用户的发布仓 | ✅ `chiro2001/deepseek-v4.1-flash-ascend910B`，最新 `115e9a7` | — |

⚠️ **`ssh A3-node2` 用的是错的 hostname**（要写 `A3-node2` 时 DNS 解析失败）；
且 A3-node2 负载常年 45+，**默认避开**。

---

## 2. 本轮（09-21 全天）做了什么 —— 按交付物分类

### 2.1 ★ codex 直连 A3 本地模型（**已验证可用**）

**修了 3 处** `patch_deepseek_v41_frontend/encoding.py`（+105/−5）：

| 缺陷 | 修复前 | 修复后 |
|---|---|---|
| `input_text` 块 | 渲染成**字面量** `[Unsupported input_text]`（**HTTP 200 但用户的话没进模型**） | 正常 |
| `developer` 角色 | **HTTP 500**（`AssertionError`） | 200 |
| 控制 token 可从正文注入 | 可**伪造轮次边界** | 零宽空格转义（1 token → 6 token） |

**验证**：51 + 53 单测 · HTTP 5/5 · **真实 codex 5/5**（单轮文本 / 工具调用 / 图片 / 多轮 resume / **子代理**）· 子代理语义抓包核对。

**已推送**：`chiro2001/deepseek-v4.1-flash-ascend910B` 提交 **`115e9a7`**，
含一键使能脚本 `tools/enable_codex_responses.sh`（`on`/`off`/`status`，幂等、备份、回滚）。

详细：`logs/30`（根因）→ `logs/33`（验证）→ `logs/34`（子代理语义）。

### 2.2 三个定量结论（**都可以直接引用**）

| 结论 | 关键数字 | 出处 |
|---|---|---|
| **18× 注册成本之谜解开** | 真实硬件 **0.59–0.78 ms/MiB**；**真实 206 GiB 表实测 119.4 s**（与 README 的 133 s 只差 11%）；离群的是测试 VM | `logs/29` |
| ↘ 顺带**推翻自家结论** | "内存形态无关" 是**稀疏文件假象**（`ftruncate` 造的文件 0 块分配，与真实文件差 **65×**） | `logs/29` |
| **MoE mask 方差** | 符号 **21/21 稳定**，但幅度比原稿**小 30–40%** ⇒ PR 已改写 | `logs/10` |
| **index_select 清扫** | 三候选**两个否掉**（一处是 vLLM 上游副本、一处是 3-D 索引） | `logs/09` |

### 2.3 四个运维事故（**全部定位 + 加固**）

| 事故 | 真因 | 加固 |
|---|---|---|
| 03:37 全部 codex + tmux 消失 | user-slice memcg 撞 26 GiB 上限，内核杀到 `systemd` | `MemorySwapMax 0→8G`、`/tmp 15G→8G`、启用 `systemd-oomd`、`~/tmp/<日期>/` 协议（`logs/20`/`21`） |
| **A3 起服连环失败** | `VLLM::EngineCore`/`VLLM::Worker_*` 进程名里**没有** `vllm serve` ⇒ `pkill` 杀不到 ⇒ 残留持 206 GiB ⇒ 自我强化失败链 | `docker stop && docker start`；判据见 `logs/32` |
| 单卡机交回 | VPN 脚本在 **tmpfs**（必丢）、OpenVPN 配置在 overlay（重建即丢） | 三处备份 + 接回清单（`logs/28`） |
| tmux 会话丢失 | 同 03:37 | 找回 14 codex + 29 dsh（`logs/26`，纯清单 `logs/26-...-session-list.txt`） |

### 2.4 A2 服务质量（基于用户粘贴的 18 个采样）

prefix 命中 **96.4%**（从 0% 修好）· A 中位 **3.58** · KV 占用 <10% · **两种已知坏状态都未复现** ·
三路独立推导出 `SP_TOKENS=7`。详见 `logs/31`。

---

## 3. 上游合入：材料就绪清单（**一个都没提交，等授权**）

| 分支 / 文件 | 状态 |
|---|---|
| `perf/rope-fused-index-select` = `d4167f52` | ✅ 已推 fork，12 单测 |
| `perf/moe-contiguous-expert-map` = `3a0c48c0` | ✅ 已推 fork，8 单测 |
| `perf/rope-index-select-on-16285` = `0121f203` | ✅ 与 #16285 的组合分支 |
| `pr/PR-rope-index-select.md` | ✅ 可提交（含 RFC [87]/[91] 归属） |
| `pr/PR-moe-mask-range.md` | ✅ 可提交（方差已按实测改写） |
| `pr/issue-draft-cc1d-dead-import.md` | ✅ 可提交（上游死 import bug） |
| `pr/issue-track-{A,B,C}.md` + `pr/RFC-comment.md` | ✅ 可提交 |
| `pr/RFC-16375-CONTRIBUTION.md`（692 行） | ✅ 含 206 GiB 真表实测 |
| `pr/TRACK-D-engram-host-table-support.md` | ✅ 纯文档贡献 |

**★ 关键框架**（别搞错）：RFC #16375 是**他们的开发计划本体**（50 条目 / 0 完成 / 0 评论，
明确邀请 "owners can be attached"）⇒ **打法 = 帮他们打勾，不是证明我们更强**。
详见 `PLAN-REVIEW.md`。

**★ C 轨已改向**：维护者在 #16285 明确否掉 DSpark v1 图模式
（"DSpark is consistently sync-bound rather than graph-bound… please use v2"）
⇒ 撤回"draft 图"主张，改为**交出 sync 账**。见 `logs/18`。

---

## 3.5 ★ 2026-09-21 下午新增：数据缺口补测（第一、二轮）

四个子代理并行跑在 A3-node1 的 **三个独占槽位**（c0=die3 / c1=die6 / c2=die7，
`tools/a3_up.sh` 起容器、`tools/a3_chip.sh` 单次运行一把锁）。补料清单见
[`DATA-GAPS.md`](DATA-GAPS.md)（13 项逐条），本轮**能补的 8 项补了 6 项**：

| 补了什么 | 结果 | 日志 |
|---|---|---|
| RFC [91] 融合边界矩阵（RoPE） | **32/32 + 5/5**；`n=0` 逐位一致；`n=4096` 图内外 −376/−384 µs；6→2 kernel | [`35`](logs/35-20260921-rope-edge-cases.md) |
| RFC [97] control arm | 时间 1.00–1.06×、**显存恒定 1.20×**；三次运行同向 | [`36`](logs/36-20260921-engram-gate-control-arm.md) |
| RFC [47] NUMA/带宽 | 连续 **107 GB/s**、gather **96 GB/s**；并发**按 socket 封顶 ≈115 GB/s**（最坏 2.8×） | [`38`](logs/38-20260921-host-dram-bandwidth.md) |
| §3.3 host-register 双 API | 两条 API **ret=0 + 设备侧逐字节一致** ⇒ 本机无分岔 | [`37`](logs/37-20260921-ngram-and-hostreg-ab.md) |
| §3.3 ngram JIT | 生产 decode `n=128` **22.8×**；per-token 走法 **1312×**；`torch.equal` 14/14 | [`37`](logs/37-20260921-ngram-and-hostreg-ab.md) |
| 上游进度复查 | **#16925 已 mergeable**；维护者对 v1 图模式第三次表态 | [`39`](logs/39-upstream-recheck-2.md) |
| ★★ **真实 206 GiB 表 + 3 die 并发** | **24 次并发满表注册全 `ret=0`，无 207001/507011**；真表行宽 256 B ⇒ **均匀 gather 只 7.55 GB/s**（合成表 96），但**热行 2.6–4.3×**；注册不扰动其它 die（≤0.8%）；**更正 `logs/29` 的"65× 是缓存冷热"**（两片 mincore 都 1.00） | [`40`](logs/40-20260921-real-table-concurrency.md) |
| ★ engram gate 的 padding 天花板曲线 | `t ≈ 0.1 + 0.69×(MAX/512) ms`，**CHUNK 以上无拐点**；生产约束下 2048 已最优，小 batch 图最优 512；`MAX=256` 会被静默抬到 4096（坑） | [`41`](logs/41-20260921-engram-gate-ceiling-sweep.md) |
| ★ **把 RoPE 的 int32 回退消除掉** | 每次调用只 build 一次 index ⇒ **+20.7/+28.9 → −17.9/−17.7 µs**，27 格全为负；PR 分支 amend 成 **`ed5b928c`**，组合分支重建为 **`4abfa85e`** | [`42`](logs/42-20260921-rope-index-hoist.md) |

**两处顺带修正**：① `RFC-comment.md` 的 [91] 段从 "not measured" 收窄为"RoPE 已测、
其余融合未测"；② 确认 `11.4 ms/MiB`（⇒40 分钟）**只属于测试 VM**，真机
**0.59–0.78 ms/MiB**（206 GiB 实测 119.4 s）。

**基础设施两条**（都写进 [`AGENTS.md`](AGENTS.md)）：① **跨机传文件走 coscli，不走 ssh**
（§2.0，helper `pr/cos-xfer.sh`，大 JSON 先裁剪）；② A3 槽位容器现在挂载真实 Engram 表
（§2.2.1，`/tables` 是 `rw` —— 只读 VMA 会被 `107017` 拒绝，纪律是只读用途 + 已知会弄脏页）。

**发布**：材料已脱敏（`A3-node1→A3-node1`、去账号名）后推到用户发布仓
`chiro2001/deepseek-v4.1-flash-ascend910B` 的 `upstream/` 目录（提交 `72fdcff`），
审阅入口 `upstream/README.md`。发布脚本 `publish/publish.sh` 带 **8 条泄漏硬校验**
（内网 IP、主机名、账号名），校验不过就拒绝发布。

---

## 4. 下一步（优先级 + 为什么）

### P0 —— 只有用户能解锁

| 任务 | 说明 |
|---|---|
| **授权发上游 PR / issue** | 三条候选都就绪（rope PR、MoE PR、cc1d issue）。**这是整条线的终点** |
| **A2 进一步测量**（需用户贴数据） | 当前只到 4 并发（上限 32）；缺 **PREFIX=1 vs 0 两臂对照**、**延迟分位数**、`/metrics` 累计值 |

### P1 —— A3 在手就能做

| 任务 | 预计 | 备注 |
|---|---|---|
| **用 codex 在 A3 跑真实工程任务** | 即时 | 它现在能连了，这是最直接的验证 |
| `glm5next/mtp.py` 的 `index_select` 补丁 | ~1 h | **判定已完成**（`logs/09`），只差写补丁+单测 |
| 重建 A3 镜像（**可选**） | ~20 min | 当前容器里**已经有**脚本与载荷（我手工放进去的，见下），所以 `on` 现在就能跑通。重建镜像只是让**新容器**也自带 —— 不重建也能用 |
| `tools/verify_codex_responses.sh` | ~30 min | 把 10 项验证（HTTP 5 + codex 5）打包成一条命令 |

### P2 —— 整理

| 任务 | 说明 |
|---|---|
| `enable_codex_responses.sh` 的 `WANT_MD5` | 建议改成从 `SRC` 现算，少一个手工同步点 |
| `reasoning.encrypted_content` 定时炸弹 | codex 每轮都带 `include`，vLLM 不产出所以现在不炸；**上游一旦产出就 400** |

---

## 5. 接手后**先跑这三条**（30 秒，避免白干）

```bash
# ① A3 服务还活着吗？codex 补丁还在吗？
ssh A3-node1 'curl -s -m 5 -o /dev/null -w "health=%{http_code}\n" http://127.0.0.1:8020/health; \
  docker exec dsv41-a3 bash /opt/dsv41/tools/enable_codex_responses.sh status'

# ② 起服前**必须**确认没有残留（否则会卡在 rtsMallocHost 207001）
#    判据：**有服务在跑 ⇒ 应该看到 10 个进程**（1 主 `vllm serve` + 1 `VLLM::EngineCore` + 8 `VLLM::Worker_TP*_EP*`）；
#          **要起新服务前 ⇒ 必须为空**
ssh A3-node1 'docker exec dsv41-a3 bash -c "ps -eo pid,args | grep -E \"[V]LLM::|[v]llm serve\"" | wc -l'
#    当前实测：10（正常）；若要重启，先 docker stop && docker start 清空，再确认变 0

# ③ 仓库是否干净、有没有未推的东西
for r in upstream-v41/vllm-ascend-fork upstream-v41/vllm-ascend-upstream dsv41-release; do
  printf "%-34s " $r; git -C ~/projects/dsv41/$r status --short | wc -l
done
```

---

## 6. 硬性红线（**违反即回滚**，完整版见 `AGENTS.md` §1）

1. **绝不**主动发起 PR / issue / 评论 —— 所有产出只写草稿文件
2. **绝不** push 到 `vllm-project/vllm-ascend`；只能推 `chiro2001/vllm-ascend`（fork）
3. 占卡必须走锁（单卡机用 `with_chip.sh`；A3 上先确认芯片空闲）
4. **绝不**手设 `ASCEND_RT_VISIBLE_DEVICES`（锁脚本注入）
5. 结论必须标 **【实测】/【推断】/【未确认】**，不许把推断写成事实
6. **停 A3 服务不能只杀 `vllm serve`** —— 见 §2.3，最稳是 `docker stop && docker start`

---

## 7. 已知的、容易再踩的坑（**按被坑次数排序**）

| # | 坑 | 症状 / 对策 |
|---|---|---|
| 1 | **`VLLM::` 进程名里没有 `vllm serve`** | `pkill -f "vllm serve"` 杀不到 ⇒ 起服连环失败。用 `docker stop && docker start`，并确认 `ps` 里 `[V]LLM::` 为空、僵尸数 0 |
| 2 | **本机 `/tmp` 是 tmpfs** 且计入 26 GiB 内存上限 | 2026-09-21 03:37 因此 OOM 杀掉全部 codex。**临时文件一律放 `~/tmp/<日期>/<任务>/`**（`source pr/tmpdir.sh`） |
| 3 | **`pkill -f "<自己的关键字>"` 会杀掉 ssh 自己** | 命令行里含该字符串就会被匹配到。用 `pgrep` 拿 PID 再 `kill`，或用 `[c]apture_proxy` 这种正则技巧 |
| 4 | **稀疏文件会伪造"便宜"的注册成本** | `ftruncate` 造的文件 0 块分配，比真实文件便宜 65×。**对比前先查 `st_blocks`** |
| 5 | `docker cp` 对该容器报 `invalid argument` | 改用 `tar czf - ... \| docker exec -i <容器> tar xzf -` |
| 6 | `docker stop` 报 `did not receive an exit event` | **是假错误**，容器确实停了（`docker inspect` 显示 `exited`） |
| 7 | 容器日志时间戳是 **UTC**（差 8 小时） | grep 按 CST 查会查不到 |
| 8 | `import acl` 必须在 `import torch` **之后** | 否则 `libc10.so: cannot allocate memory in static TLS block` |
| 9 | **8021 端口是别人的 SGLang** | 起代理/测试服务时避开（本轮用了 8022/8023/8024） |
| 10 | **当前容器的 `/opt/dsv41/` 是手工补过的** | `tools/enable_codex_responses.sh` 与 `patches/patch_deepseek_v41_frontend/` 是我手工 `cp` 进去的；**重建容器后会丢**。发布包（`115e9a7`）里的 Dockerfile 已含这两条的 COPY 行，重建镜像即恢复 |

---

## 8. 关键路径速查

| 想找什么 | 去哪 |
|---|---|
| 规则 / 红线 / 环境 / 锁 | `AGENTS.md` |
| 命令级上手 + 排障表 | `ONBOARDING.md` |
| 日志索引（32 份） | `logs/README.md` |
| 状态看板 | `logs/01-...-session-status.md` |
| 给上游的草稿 | `pr/` |
| RFC 原文快照（引用基准） | `pr/refs/RFC-16375-body.md` |
| 单卡机成果镜像 | `~/projects/dsv41/remote-910C-20260921/` |
| 用户的发布仓（git 副本） | `~/projects/dsv41/dsv41-release/` |
| fork（可推的） | `vllm-ascend-fork/`（`origin`=fork，`upstream`=官方） |
