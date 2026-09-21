# 线 1 终验成功：两个补丁合挂 + 16 请求跑满（8 卡真权重）

> 2026-09-22 01:3x，主代理前台**直读 A3 产物**核实的中间结论
> （子代理 **L2_final** 的日志 `logs/016` 收尾中，TBD 待填）。
> 这一格补的是：`logs/009`（scheduler 补丁）与 `logs/014`（cpu_npu 补丁）
> **从未同时挂过**、且 009 只跑成 2 请求的缺口。

---

## 一、★ 两个补丁在真实 8 卡上**全部生效**

### 1.1 挂载与后端（`docker inspect` 直读）

```
scheduler.py -> /vllm-workspace/vllm/.../offloading/scheduler.py     ← D2 补丁
cpu_npu.py   -> /vllm-workspace/vllm-ascend/.../native/cpu_npu.py    ← P1 补丁
NPU_OFFLOAD_HOST_MEM=registered
```

### 1.2 P1 的池子补丁：**8/8 worker 全走 registered，128 次注册 `ret=0`**

```
[P1_pinned] CPU pool backend = registered (NPU_OFFLOAD_HOST_MEM)      ← ×8 worker
[P1_pinned] CPU pool 7424 x 524288 (3.62 GiB): registered dev=0x3fed7e00000 ret=0
[P1_pinned] CPU pool 7424 x  65536 (0.45 GiB): registered dev=0x3febac00000 ret=0
...
grep -c "P1_pinned"  = 136        grep -c "ret=0" = 128（= 16 张量 × 8 rank）
```

⇒ **候选 β 在真实 8 卡上成立**（`logs/014` 的单进程烟测升级为生产验证）。
`num_blocks = 7424` = 58 GiB 池 ÷ 8 MiB/条目，与 `logs/016` §1.3 的公式吻合。

### 1.3 D2 的 scheduler 补丁：13 组清单 + `state` 组被正确排除

```
[D2_offload] KV 卸载 group 清单 n=13: [(0,'DeepseekV41FullSpec',128,8,...),
   (1,'DeepseekV41CompressorStateSpec',32,3,...,False), (2..11,'DeepseekV41SWASpec',128,4,...),
   (12,'DeepseekV41DraftSWASpec',128,3,...)]
[D2_offload] 参与卸载的组：full_attention=[0] sliding_window=[2..12]；被排除的组=[1]
```

⇒ **`state` 组（idx 1，第 6 个字段 `False`）被排除出存/查两侧** —— 与 `logs/009` §2 的设计一致。

---

## 二、★★ 四条判据（**16 请求**，这是本轮要补的那一格）

臂 `l2-dram58-16p`（`OFFLOAD_GB=58`、`blocks_per_chunk=8`、16 请求 × 32768 token × 2 轮、轮间 reset）：

| # | 判据 | 值 | 判定 |
|---|---|---|---|
| ① | `BlockStored(medium="CPU")` | **6,144** | ✅ |
| ② | **`kv_offload_total_bytes{CPU_to_GPU}`** | **1.2444e10 B = 12.44 GB**（`size_count = 128`） | ✅ |
| ③ | `GPU_to_CPU` | **1.9678e11 B = 196.8 GB**（`size_count = 640`） | —— |
| ④ | **replay / fill TTFT p50** | **253.4 / 4429.4 ms = 17.5×** | ✅ |

**独立于计数器的取回证据**（D2 补丁的 load job 日志）：

```
[D2_offload] load job req=cmpl-… keys=42 group_sizes=[248, 0, 1,1,1,1,1,1,1,1,1,1,1]
                                                  ↑ state 组占位 0
grep -c "load job" = 16                            ← 16 个请求各一条，跑满
```

⇒ **16/16 请求都从 DRAM 取回了**，且 `group_sizes[1] = 0` 证明 `state` 组不参与（设计如此）。

---

## 三、与历史臂的对照（**修复链条完整**）

| 配置 | 请求数 | `CPU_to_GPU` | replay/fill | 出处 |
|---|---:|---|---|---|
| 无补丁（基线） | 16 | **0** | 4167/4192（+0.6%） | `logs/001` 臂 D |
| 只挂 scheduler 补丁、池子走 pinned | **2** | 1.56 GB | 298/4425（14.8×） | `logs/009` |
| 单卡 tiny、两个补丁 | 16 | 273 MB | 50.9/465.9（9.2×） | `logs/013` |
| 只挂 scheduler、池子 32 GiB | 16 | **0**（池子不够） | 4212/4395（1.0×） | `logs/016` 臂 1 |
| **两个补丁 + 58 GiB 池** | **16** | **12.44 GB** | **253.4/4429.4（17.5×）** | **本臂** |

★ **两行关键对照**：
* `logs/016` 的 dram32 臂（16 请求、池子 32 GiB）= **0 取回** ⇒ 与 `logs/013` 的拐点预测一致（工作集 6,144 条 > 池子 4,096 条）；
* **同一个服务、同一份补丁，池子 32→58 GiB 就从 0 变成 12.44 GB** ⇒ **池子容量是唯一的门槛**。

> ⛔ **主代理此处误判，已由 `logs/016` §收尾更正**：我当初 grep 的 `…serve_a2.log` **只是 wrapper 日志**，
> 而 L2 的旧版 runner 从 `shadow-pkg/results/` 里抽"最新目录"算 keylines —— 那次起服的 `OUT` 被覆盖，
> 于是抽到了 **09-21 23:15 的陈旧目录**，永远 0 行。
> **实际两个补丁是在挂的**（同一条臂的 `out/serve.log` 里有
> `[P2_pinned] CPU pool 4096 x 524288 … ret=0` 与 `[D2_offload] … n=13`）。
> ⇒ **那条臂 0 命中是"池子 0.667× 工作集"的必然结果，与补丁无关。**
> L2 已修：runner 改成**每臂独立 `OUT`**，并在**起服就绪后、压测前**加了自检门
> （`registered dev=` ≥ 8 且 `[D2_offload]` ≥ 1，否则删容器 `exit 9`）。
> 重跑的 `l2-dram32-16p-r2` 打印 `P1_registered=128 P1_fallback=0` 后**照样 0 命中** ⇒ 结论不变。

---

## 四、剩下的（`logs/016` 收尾时会补）

### ★★★ 4.1 拐点在 8 卡真权重上**被三条臂夹死**（工作集 6,144 条）

【实测，主代理直读产物】三条同口径臂（16 请求 × 32768 token × 2 轮、**都挂了两个补丁**）：

| 臂 | 池子 | `num_blocks` | vs 工作集 | `CPU_to_GPU` | **replay / fill** | 加速 |
|---|---:|---:|---:|---|---:|---:|
| `l2-dram32-16p-r2` | 32 GiB | 4,096 | **0.667×** | **0** | 4201 / 4545 ms | **1.0×（归零）** |
| **`l2-dram48-16p`** | **48 GiB** | **6,144** | **1.000×** | 跑中 | **252.9 / 4518.1 ms** | **17.9×** ✅ |
| `l2-dram58-16p` | 58 GiB | 7,424 | 1.209× | **12.44 GB** | 253.4 / 4429.4 ms | 17.5× ✅ |

**⇒ 拐点精确落在 `1.000×`（6,144 条）**：
* **0.667×** ⇒ 整轮归零；
* **恰好 1.000×** ⇒ **17.9×（全中）**；
* 1.209× ⇒ 17.5×。

★ **这与 `logs/013` 在单卡 tiny 上测到的规律逐条一致**
（013：1.000× 全中、0.989× 开始掉、**0.977× 断崖**）。
**⇒ `logs/013` 的拐点公式在 8 卡真权重上得到了独立验证**，
而且**"恰好 1.000×"这一格在 8 卡上也是安全的**（与 013 一致）。

### 4.1b 顺带：池子容量**是唯一门槛**（同一份补丁、同一个服务）

| 对照 | 补丁相同？ | 池子 | 结果 |
|---|---|---|---|
| `dram32-16p`（**无补丁**） | ❌ | 32 GiB | 0（1.0×） |
| `dram32-16p-r2`（**有补丁**） | ✅ | 32 GiB | **0（1.0×）** ← **补丁在池子不够时也救不回来** |
| `dram48-16p`（有补丁） | ✅ | 48 GiB | **17.9×** |

⇒ **两个补丁是必要条件，池子容量是充分条件**。

### 4.2 其余待补

| # | 事项 |
|---|---|
| 1 | `external_prefix_cache_hits` 的确切值（本文件只核到 `CPU_to_GPU` 与 load job） |
| 2 | 池子利用率、每轮 hits 比例 |
| 3 | `dram48` / `dram32-r2` 两条臂（L2 仍在跑） |
| 4 | 宿主实占与 `×6.945` 乘数在 58 GiB 这一档的复核 |

**⇒ 但"线 1 在生产配置（8 卡 + 真权重 + 16 请求）下确实取回了"这一点已经成立。**
