# HANDOVER-2 —— 交接快照（2026-09-21 21:5x）

> **给下一个接手的人（或压缩上下文后的自己）。**
> 先读本文件 → 再读 [`AGENTS.md`](AGENTS.md)（红线/环境/锁）→ [`ONBOARDING.md`](ONBOARDING.md)（命令级）。
> 本次会话之前的交接在 [`HANDOVER.md`](HANDOVER.md)（15:47 版，仍然有效，但**本文件更新**）。

---

## 0. 一句话现状

* **上游材料总共 4 份候选**（2 PR + 2 issue）：**① ② ③ 数据齐、可发；④ 差一格实测**。
* **A3 的 8–15 正在被 D_off8 占用**（验 DSV4.1 + DRAM KV 卸载），其余槽位空闲。
* **用户的 `dsv41-a3` 服务是 `Exited (137)`，被我 `docker stop`（未删）** —— 合适时机要还原。
* **一份 PR/issue/评论都没发**（用户明确未授权）。
* 本轮（15:47 → 21:5x）新增：**logs/35–45（11 份）+ docs/KV-CACHE-ACCOUNTING.md + PR-GAPS.md**。

---

## 1. ★ 资源可达性（先确认，否则命令白跑）

| 资源 | 状态 | 怎么用 |
|---|---|---|
| **A3-node1** | ✅ | `ssh A3-node1`（免密） |
| **Phy-ID 8–15** | ⚠️ **D_off8 正在用** | 用前先 `npu-smi info` 看 |
| **单卡槽位 c0/c1/c2** | ✅ 空闲（die 3/6/7） | `bash ~/projects/dsv41-upstream-pr/tools/a3_chip.sh <c0\|c1\|c2> --name <名> -- <命令>`；退出码 **75 = 没抢到锁** |
| **用户的服务容器** | ⛔ `dsv41-a3` = `Exited`，**不要动** | 还原用 `state/restore_dsv41_a3.sh`（**只在用户确认后跑**） |
| **A2** | ⛔ 连不上 | 只能靠用户粘贴数据 |
| **`mooncake-master`** | ✅ 在跑（:50088） | **只连不重启** |
| 三份 git 仓 | ✅ 全干净 | fork / upstream / dsv41-release |

**★ 起服前必查**（否则会卡 `rtsMallocHost 207001`）：残留进程数。
有服务在跑 ⇒ `ps` 里应有 **10** 个 `VLLM::`；要起新服务 ⇒ 必须为 **0**
（用 `docker stop && docker start` 清，**别用 `pkill`** —— 见 §8 坑 1）。

---

## 2. ★★ 四份候选的状态（本会话最重要的一张表）

详细版见 [`PR-GAPS.md`](PR-GAPS.md)。

| # | 候选 | 位置 | 状态 | 差什么 |
|---|---|---|---|---|
| **①** | **PR：RoPE `index_select` 融合** | 分支 `perf/rope-fused-index-select` @ **`ed5b928c`**（base `5fbcfaa9`）；草稿 `pr/PR-rope-index-select.md` | ✅ **可发** | 无 |
| **②** | **PR：MoE mask 范围比较** | 分支 `perf/moe-contiguous-expert-map` @ **`206d39c9`**；草稿 `pr/PR-moe-mask-range.md` | ✅ **可发** | 无 |
| **③** | **issue：上游死 import（cc1d）** | 草稿 `pr/issue-draft-cc1d-dead-import.md` | ✅ **可发** | 无（已在 `5fbcfaa9` 上复核行号） |
| **④** | **issue：`cpu_binding` 挂死** | 草稿 `pr/issue-draft-cpu-binding-migratepages-hang.md`；补丁 `pr/patches/cpu-binding-hang-fix.patch` | ⚠️ **差一格** | **缺"打完补丁 + `CPU_BIND=1` ⇒ 服务真起来了"的端到端实测**；另**框架要改写**（见 §3.4） |

### 2.1 本会话修掉的**文档过时**（4 处，都验证过）

| 位置 | 问题 | 已修 |
|---|---|---|
| `PR-rope-index-select.md` checklist | 还写着"两个 int32 小回退（+20.7/+28.9）已在 §5a 写出" | ✅ 改成"更早一版曾回退，`ed5b928c` 已修 → −17.9/−17.7" |
| 同上 §"已知边界"第 6 条 | 还写着"那是**另一个改动**，**我们没测过**" | ✅ 同上（**那个改动就在本分支里**） |
| `PR-moe-mask-range.md` 头部 | SHA 还是 rebase 前的 `3a0c48c0` | ✅ 改成 `206d39c9` |
| `issue-draft-cc1d-dead-import.md` | 基线还是 `c173a64a` | ✅ 改成 `5fbcfaa9` + 注明行号已复核 |

---

## 3. ★ 待回答的问题（用户上一条被中断的提问）

**用户问**：

> *"issue cpu_binding 挂死，这个是否是因为我们要跑 Engram 放在 DRAM 导致的，而上游其实还没支持？"*

### 3.1 【实测】`migratepages` 想搬的绝大部分**就是 Engram 表**

出事时那个 worker 的 `smaps_rollup`：

| 量 | 值 |
|---|---:|
| `Rss` | **138.0 GB** ← migratepages 以为要搬这么多 |
| `Pss` | 19.6 GB ← 真正"属于这个进程"的只有这么多 |
| **`Shared_Dirty`** | **122.6 GB** ← **RSS 的 89%**，共享脏页 |
| `Shared_Clean` | 13.3 GB |
| `Private_Dirty` | **2.2 GB** |

⇒ **138 GB 的 RSS 里 89% 是 `Shared_Dirty`，而那正是 Engram 表（`MAP_SHARED` 文件映射）；
私有内存只有 2.2 GB。**

**所以：把 Engram 从 host DRAM 拿掉，`migratepages` 几乎无事可做，大概率不会挂。**

### 3.2 ★★ 但更关键：这是**上游自己的两个功能互相踩**

* **上游也做 host-resident Engram** —— PR **#16925**（`aclrtHostRegisterV2(MAPPED|PINNED)`）就是干这个的；
* **而 `enable_cpu_binding` 在上游默认 `True`**（`ascend_config.py:387`）；
* ⇒ **"host 常驻大表 + 默认开着的绑核"是上游自己的配置**，在 NUMA 不均衡的机器上**会互相踩死**。

**所以这份 issue 不该降级成"我们的环境问题"，应改成**：

> **不是"你们的 migratepages 有毛病"，而是"你们的 host-offload（#16925）与你们默认开的
> cpu_binding 会互相踩死"** —— 这在 #16925 合入后会成为真实用户的问题。

而且**共享页物理上只有一份**，8 个 rank 各映射整表、各自要求"本地" ⇒
**同一页不可能同时本地在 4 个不同 NUMA node 上** ⇒ 这个迁移目标**在原理上不可能达成**。

### 3.3 仍然属于上游的那一半：**超时分支没有超时**

即使触发条件是我们的配置，**代码缺陷仍然是真的**：

```python
except subprocess.TimeoutExpired:
    p.kill()
    out, _ = p.communicate()      # ← 无界！可以永远阻塞
```

`p.kill()` 之后那次 `communicate()` **没有 timeout** ⇒ 1000 秒的"保护"在超时那一刻
**变成永久阻塞**。这条**与 Engram 无关**，任何"helper 陷在内核里"的场景都会中招。

### 3.4 该怎么改这份草稿（**接手后的第一件事**）

1. **重写 §2"影响面"**：明确"触发条件是 host 常驻大表 + 默认 `CPU_BIND=1`"，
   并指出**上游 #16925 合入后同样会中招**；
2. **把两条修复的定位分开写**：
   * ① `execute_command` 有界 reap —— **纯上游缺陷**，必须修；
   * ② `bind_memory` 的 MemFree 预检 —— **对 host-offload 场景的必要防护**；
3. **补那一格实测**：影子包里跑 `CPU_BIND=1`，期望看到
   `[migrate] ... skipped: the node has X GiB free but this rank holds Y GiB` **且服务继续起来**。
   若仍挂 ⇒ 那也是结果，如实写；
4. **可选的最强证据**：再跑一臂 `CPU_BIND=1` **不带 Engram**（`ENGRAM=0`）作对照 ——
   若照样挂 ⇒ 主因不是 Engram；**若起来了 ⇒ 直接证明 §3.1 的因果**。

---

## 4. 本会话（15:47 → 21:5x）交付了什么

### 4.1 新增材料（已发布到 GitHub，commit `9cef18a`）

| # | 内容 | 一句话 |
|---|---|---|
| **35** | RoPE 边界矩阵 | **32/32 + 5/5**；`n=0` 逐位一致；6→2 kernel |
| **36** | Engram gate control arm | 时间 parity（1.00–1.06×）、**显存恒定 1.20×** |
| **37** | ngram JIT + host-register 双 API | **22.8× / 1312×**；两 API 本机无分岔 |
| **38** | host DRAM 带宽 / 并发 / NUMA | 连续 **107 GB/s**、gather **96**；并发**按 socket 封顶 ≈115** |
| **39** | 上游进度复查 | **#16925 已 mergeable**；维护者对 v1 图模式第三次表态 |
| **40** | ★★ **真实 206 GiB 表 + 3 die 并发** | **24 次注册全 `ret=0`**（无 207001/507011）；真表行宽 256 B ⇒ gather 只 7.55 GB/s；**热行 2.6–4.3×**；**更正了 `logs/29` 的"65× 是缓存冷热"** |
| **41** | engram gate 的 padding 天花板曲线 | `t ≈ 0.1 + 0.69×(MAX/512)`，**无拐点**；`MAX=256` 被静默抬到 4096 |
| **42** | ★ RoPE 的 int32 回退**被消除** | `+20.7/+28.9 → −17.9/−17.7 µs`；27 格全为负；PR 分支 amend 成 `ed5b928c` |
| **43** | ★★ 图内 ceiling 扫描 | 曲线仍线性、推荐值不变；**但"只慢 1.19–1.38×"是 eager 假象**，图内真实 **1.41×~6.84×** |
| **44** | ★★★ **单会话消融（12 臂）** | **`MOE_AG` 是唯一决定接受长度的 gate**（A ×2.83、tok/s ×2.9）；其余 5 个落在噪声内；顺带补了 RFC [75] 起服/编译缓存数据 + RFC line 104 的 trace 制品 |
| **45** | ★★★ **DRAM KV 卸载打通** | 单卡：有效容量 **9.0×**、replay TTFT **2.28×**、D2H/H2D 29.2/28.5 GB/s；Mooncake 路径也通；**4 个必知坑** |
| **docs/** | ★★ **KV 账本** | **4421 B/token 逐项拆解**（与实测差 0.3 B）；差官方 4.97× 的主因是**精度 3.76×**；8/4-bit 用不了的**三道门**；**`replicated_layout` 在昇腾必然 False ⇒ CPU 层存 8 份，一个 8× 的可优化点** |
| **PR-GAPS.md** | ★ 提 PR 还缺什么 | 四份候选逐条核对（§2 的详细版） |

### 4.2 可直接引用的关键数字

* **① RoPE PR**：`6→2` kernel/次、eager `n=4096` **−376 µs**、**ACLGraph −384 µs/call**、27 个计时格全负、13 单测；
* **② MoE PR**：ACLGraph 内 **21/21 更快**（中位 −10.7…−12.9 µs）、eager 21/21 更慢（两口径都给）；
* **③ cc1d issue**：`patch_triton.py:326` 的 import 在新 main 上仍然准确；
* **④ cpu_binding issue**：见 §3（**需要按 §3.4 改写**）。

---

## 5. 正在跑的（**唯一活跃的子代理**）

### D_off8 —— DSV4.1 + DRAM KV 卸载（8 卡）

| 项 | 值 |
|---|---|
| 容器 | `abl-off-off8-ctl` |
| 阶段 0 | ✅ 完成：校准臂通过（复现 M_offload 的 86.6→39.1 ms） |
| **DSV4.1 的 KV group 清单** | ✅ **13 个 group**（full / state / swa0–9 / dspark），**每个都是 `UniformTypeKVCacheSpecs` 包装** |
| **它的先验判断** | **大概率起服就撞 `AssertionError`** —— `get_sliding_window_size_in_chunks()` 与 `KVCacheSpecRegistry.get_manager_class()` 都只认**裸类型**（`offloading/scheduler.py:149`） |
| 我给的出路 | 撞了之后用上游**已有的** `UniformTypeKVCacheSpecs.first_spec` 做**最小 unwrap 补丁**（只在影子包）⇒ 从"报 bug"升级成"报 bug + 带修复" |
| 预计 | **~1.5 h** ⇒ `logs/46` |

**关键**：D_off8 只改**影子包** `~/projects/dsv41-upstream-pr/shadow-pkg/`，**没碰用户的 `dsv41-release/`**；
跑完会清掉自己的 `abl-off-*` 容器。

---

## 6. ★ 硬性红线（违反即回滚）

| # | 规则 |
|---|---|
| 1 | **绝不**发 PR / issue / 评论（用户未授权）。所有产出只写草稿文件 |
| 2 | **绝不** push 到 `vllm-project/vllm-ascend`；只能推 `chiro2001/vllm-ascend`（fork） |
| 3 | 占卡走锁：单卡机 `with_chip.sh`；A3 用 `tools/a3_chip.sh`（**退出码 75 = 没抢到**） |
| 4 | **绝不**手设 `ASCEND_RT_VISIBLE_DEVICES`（锁脚本注入） |
| 5 | 结论必须标 **【实测】/【推断】/【未确认】** |
| 6 | **绝不**用 `NAME=dsv41-a3` 起服 —— `serve_a2.sh` 里有 `docker rm -f "$NAME"`，会**物理删除用户的容器**（可写层里 `enable_codex_responses.sh` 等手工补丁**不在挂载里，删了就没了**） |
| 7 | 用 `apply_patch` 改文件；不用 `rm -rf` |
| 8 | **跨机传文件走 coscli，不走 ssh**（见 §7.1） |
| 9 | 临时文件只放 `~/tmp/<YYYYMMDD>/<任务名>/`（本机 `/tmp` 是 tmpfs，撞 26 GiB 上限会 OOM 掉全部进程） |
| 10 | **始终用简体中文回复** |

---

## 7. ★ 本会话新增的两条基础设施约定

### 7.1 跨机传文件：**走 coscli，不走 ssh**

```bash
# 本机
bash ~/projects/dsv41/upstream-v41/pr/cos-xfer.sh put|get|ls|url ...
# A3
bash ~/projects/dsv41-upstream-pr/tools/cos-xfer.sh put|get|ls|url ...
```

默认走**私有前缀** `share/xfer/`（不会出现在公开下载页）；只有真要发链接时才 `--share`。
**大文件先在远端裁剪**（只回传汇总 + 关键字段，原始文件留远端并记 sha256）。

两个坑：① coscli 默认往 `./coscli_output` 写错误日志，下载目标是 `.` 会被拒绝
（`failOutputPath ... is subdirectory of .`）—— helper 已用 `--fail-output-path` 绕开；
② A3 到**那台中转机的高端口被过滤**，但**到 COS 是通的**（所以 A3 → COS → 本机这条路可用）。

### 7.2 A3 槽位容器挂了真实 Engram 表（`/tables`，**rw**）

`tools/a3_up.sh` 里有一行 `-v <table>:/tables:rw`。**为什么必须 rw**：
`aclrtHostRegister()` 对只读 VMA 返回 `107017`，而生产路径本身就是 `O_RDWR` + `MAP_SHARED`。
**纪律**：只读用途、绝不写内容；已知副作用是**注册会把页标脏**（`Dirty` 可涨到 108–126 GiB、
触发整表回写、mtime 变但**内容 sha256 不变**）。原版备份 `agents/T1_realtable/a3_up.sh.bak-20260921`。

---

## 8. 十个坑（按被坑次数排序）

| # | 坑 | 对策 |
|---|---|---|
| 1 | **`VLLM::` 进程名里没有 `vllm serve`** ⇒ `pkill -f "vllm serve"` 杀不到 ⇒ 残留持 206 GiB host 注册 + pinned ⇒ **自我强化失败链**（`rtsMallocHost 207001`，64 KB 都失败） | `docker stop && docker start`；判据 `ps` 里 `[V]LLM::` 为空、僵尸数 0 |
| 2 | **`serve_a2.sh` 里有 `docker rm -f "$NAME"`** | **起任何测试服务都必须换容器名**（`NAME=abl-*`） |
| 3 | **本机 `/tmp` 是 tmpfs**（计入 26 GiB 内存上限） | 临时文件一律走 `~/tmp/<日期>/<任务>/` |
| 4 | **`pkill -f "<自己的关键字>"` 会杀掉 ssh 自己** | 用 `pgrep` 拿 PID 再 kill，或 `[c]apture_proxy` 正则技巧 |
| 5 | **稀疏文件伪造"便宜"的注册成本** | 对比前先查 `st_blocks` |
| 6 | `docker cp` 对该容器报 `invalid argument` | 用 `tar czf - \| docker exec -i <容器> tar xzf -` |
| 7 | `docker stop` 报 `did not receive an exit event` | **是假错误**，容器确实停了 |
| 8 | 容器日志时间戳是 **UTC**（差 8 h） | grep 按 CST 查会查不到 |
| 9 | `import acl` 必须在 `import torch` **之后** | 否则 `libc10.so: cannot allocate memory in static TLS block` |
| 10 | **8020 那个服务现在是 `Exited`**；`mooncake-master` 在 :50088 | 起测试服务避开，别重启别人的容器 |

---

## 9. 关键路径速查

| 想找什么 | 去哪 |
|---|---|
| **提 PR 还缺什么** | [`PR-GAPS.md`](PR-GAPS.md) |
| **数据缺口总表**（哪些补了/补不了/为何） | [`DATA-GAPS.md`](DATA-GAPS.md) |
| **KV cache 的账**（4421 B/token、8/4-bit 的三道门） | [`docs/KV-CACHE-ACCOUNTING.md`](docs/KV-CACHE-ACCOUNTING.md) |
| 规则 / 红线 / 环境 / 锁 / coscli | [`AGENTS.md`](AGENTS.md) |
| 命令级上手 + 排障表 | [`ONBOARDING.md`](ONBOARDING.md) |
| 日志索引（**45 份**，含空号说明） | [`logs/README.md`](logs/README.md) |
| 给上游的草稿 | `pr/` |
| RFC 原文快照（引用基准） | `pr/refs/RFC-16375-body.md` |
| **已发布给用户审阅的镜像** | 用户仓的 `upstream/` 目录；发布脚本 `publish/publish.sh`（**带 8 条泄漏硬校验**） |
| 发布仓（用户的） | `~/projects/dsv41/dsv41-release/`（**只读，除发布外不要动**） |
| fork（可推的） | `~/projects/dsv41/upstream-v41/vllm-ascend-fork/`（`origin`=fork、`upstream`=官方） |
| 影子包（可随便改，用于 A/B） | `~/projects/dsv41-upstream-pr/shadow-pkg/`（A3 上） |

---

## 10. 下一步（按优先级）

| 优先 | 动作 | 谁 | 成本 |
|---|---|---|---|
| **P0** | **按 §3.4 改写 issue ④**（框架从"你们的 migratepages 有毛病"改成"**你们的 host-offload 与你们默认的 cpu_binding 会互相踩死**"） | 主代理 | 20 min |
| **P0** | **补 issue ④ 的端到端实测**（影子包 + `CPU_BIND=1`，期望 WARN 且服务起来）；**可选加一臂 `ENGRAM=0` 做因果对照** | 主代理/新子代理 | 15–30 min，**等 8–15 空出** |
| **P1** | 等 **D_off8** 的 `logs/46`，把 DSV4.1 的 DRAM 卸载结论并进 `docs/KV-CACHE-ACCOUNTING.md` §7 | 主代理 | —— |
| **P1** | 把 **`replicated_layout`（8× 可优化点）** 写成独立的上游 issue 草稿 | 主代理 | 30 min |
| **P2** | 用户授权后按 `PR-GAPS.md` 的顺序发 ①②③ | 用户 | —— |
| **P2** | **还原用户的 `dsv41-a3`**（`state/restore_dsv41_a3.sh`）—— **只在用户确认后** | 主代理 | 1 min |

---

## 11. 诚实清单：本轮"没做到"的

1. **A 两值（1.7 vs 4.8）的机制**仍是【未确认】—— 只知道是 `MOE_AG`；要用 MoE logits 比对才能定性；
2. **`GATE_CHUNK=512` 的 128K 端到端**未测；
3. **DSV4.1 上的 DRAM 卸载**结论还没出（D_off8 在跑）；
4. **issue ④ 的端到端实测**没做（§10 P0）；
5. **A2 上的 DRAM 带宽 / ×1.7 记账系数**未测（我连不上 A2，现在是 A3/HCCS 的数）；
6. **W8A8 / A5 / 多节点 / EP>8** 都没有 —— 环境不存在，材料里统一写 "not measured"。
