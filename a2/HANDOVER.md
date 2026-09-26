# A2 / 四轴同开 交接文档（★ 2026-09-23 02:1x 全面重写）

> 上一版（2026-09-22 15:0x）描述的是"档 B/C/D 的容量与图兼容"阶段。
> 本版覆盖**到 2026-09-23 02:1x**：四轴同开跑通、判据体系重建、五道起服门。
> 结论先行：**四轴已在 A3 真权重 8 卡上同时使能并跑通**（`logs/101`）。
> 标记：**【实测】/【推断】/【未确认】**。★ 引用任何文件前先 `ls`。

---

## 0. 一句话现状

**`ENGRAM=1` × DRAM 卸载 × int8 档 C × draft 入图，四轴在同一臂内全部达成**：
起服十项错误码全 0、三轮零失败、卸载取回真实且**两条独立运行逐字复现**、
`BlockRemoved:CPU` **事件 0 次**、draft 真入图（`Wrapping=8`、A 不恒 1.0）、
**并在同一个容器上验证了返回文本正确**（题库 10/10 + 同前缀三发逐字相同）。

★ **唯一未达成的质量项**：**"Engram 在取回前缀的历史完全正确"** —— 有可量化的极少量陈旧页（`§4`）。
★ 该质量项**不属于**本次目标的 6 条验收判据，但**属于上线前应继续追**的格。

---

## 1. 现场状态（**动手前先核这一节**）

```
锁                c0 / c1 / c2 全 FREE（截至 2026-09-23 02:1x）
容器              r8-r8-4axis-final  Up（KEEP=1 保留，PORT=8051，/health=200）
端口              8050=空闲 / 8051=被 final 占用 / 8052=空闲
内存              MemAvailable ≈ 1149 GiB（★ 起 8 卡臂前建议 ≥1400，见 §3-G3）
本机发布仓 HEAD    698ce4d（feat/kv8-dram-offload-pending，228 提交）
★ A3 的 clone       ~/projects/dsv41-release-git = 233973e ← **落后**，用前先 git pull --ff-only
```

### 1.1 当前服务的参数（`r8-4axis-final`，真权重）

| 项 | 值 |
|---|---|
| 模型 | `v41-w4a8-engram-dr-vision-qrot-mtpq` |
| 设备 | 8 卡（Phy-ID **8–15**） |
| 几何 | `max_model_len=133120`、`max_seqs=32`、`bat_tokens=8192`、`gpu_util=0.92` |
| KV 显存预算 | `--kv-cache-memory-bytes 4294967296`（**4 GiB，探针口径**） |
| 图模式 | `graph=1 eager=0` |
| 卸载池 | `cpu_bytes_to_use=23,068,672,000`（21.5 GiB 记账）、`bpc={"default":8,"swa":1}`、后端 `registered` |
| int8 | `tier=C`：`KV8_SWA=1` / `RING_FP16=1` / `APC_ALIGN=3` / `KV8_GRAPH_SAFE=1` |
| Engram | `ENGRAM=1`、`ENGRAM_DEVICE_INDEX=0` |
| ★ 精确修补 | **关**（`TRUE_TOKENS=0` / `ROW_IDS=0`）⇒ 走 pad 兜底（`PAGELESS=8`） |

### 1.2 容量（**必须按口径读**）

| 侧 | 本臂（探针口径） | A2 生产 |
|---|---|---|
| **HBM** | `GPU KV cache size = 427,643 tokens` | ★ **3,498,354 tokens ≈ 3.3 × 1M**【实测】 |
| **DRAM** | 池 21.5 GiB 记账 ⇒ 宿主 **115.05 GiB**；≈**1 × 1M** 前缀工作集 | `OFFLOAD_GB=85` ⇒ 87,040 units ⇒ **3 个 1M 会话**，宿主 296.7 GiB（**int8 后 ≈150 GiB**） |

★ ⛔ **别拿 427,643 / 485,610 当"机器容量"** —— 那是 4 GiB 预算下的探针读数（`A2-DEPLOY-NOW §B0`）。
★ **int8 档 C 不涨 HBM**（与档 B 逐字相同）；收益在**宿主 DRAM ×0.7607**（`logs/092`，与池大小无关）。

---

## 2. 四轴判决（同一臂内）

| # | 判据 | 实测 |
|---:|---|---|
| 1 | 起服无 `EE1016/507057/EH0012/207001` + 注册回落=0 | **十项全 0** |
| 2 | 三轮 `requests_failed=0` | `6/0 · 6/0 · 6/0` |
| 3 | 卸载三判据 | `CPU_to_GPU=1.7008429056e+10`、`hits=724,224`、★ `BlockRemoved:CPU` **键不存在 ⇒ 0 次** |
| 4 | draft 真入图 | `Wrapping=8`、A=**6.00/2.62/4.09** |
| 5 | int8 档 C | `tier=C` + `inner.sh` 四开关 + 容器内 `dsa_v41.py=94aeebb7`（graphsafe） |
| 6 | ★ 文本正确 | ★ **同一容器**：题库 **10/10**；`prefix-pair` 三发逐字相同（冷算 == 取回） |

★ 判决器：`通过 21 / 失败 0 / 未验 1`（未验 = 本臂按设计未开 `TRUE_TOKENS`）。
★ 两条独立运行（`fit` / `final`）的 `CPU_to_GPU` 与 `hits` **逐字相同** ⇒ 卸载判据可复现。
★ 性能：`cc=2/4/8 → 15.00/34.10/32.70 tok/s`（全 `8/0`，峰值 cc=4）；同参 `档C/档B` 倍率 **0.591/0.508/0.538×**；卸载加速 **≈9.1×**（TTFT 侧 ≈45×）。

---

## 3. 五道起服前置门（**本仓已固化，绕不过去**）

| 门 | 内容 | 落地处 |
|---|---|---|
| **G1** 合并件新鲜度 | 用同一套输入重新合并并与已安装件逐字节比对 | `a2/scripts/check_merged_fresh.sh` |
| **G2** graphsafe dsa | 断言指定的那份是 graphsafe 版（`grep -c rows_bound ≥1` = **行数**，不是运行期值） | `a2/scripts/run_4axis_arm.sh` |
| ★ **G2b** 实际挂载断言 | 起臂后轮询 `meta.txt` 的 `dsa_dir_D` 必须含 `S_graphfix` | 同上 |
| **G3** dmesg / OOM | 起臂前看 `Killed process` 与 `MemAvailable`（≥1400 建议） | 同上 |
| ★ **G4** 端口占用 | 起臂前断言端口空闲（防"`health=200` 是别人的"） | 同上 |

★ 一键启动器：`bash a2/scripts/run_4axis_arm.sh`（`DRY=1` 只跑门；`TAG=` / `PORT=` / `PROMPTS=` / `KEEP=1` 可覆盖）。
★ 验收判决器：`python3 a2/scripts/check_4axis_acceptance.py --log … --client … --metrics … --kv-events … --container … --text-probe-json …`

---

## 4. ★★ 唯一未达成的质量项：Engram 取回前缀的历史

**现象（实测）**：`apply_repairs` 确实**发现过陈旧页**（`exact` 臂 `mismatch=6` / `mode2` 臂 `mismatch=3`），
而 `mode=1` **只比对不覆盖**（一个字节都没写）；`mode=2` 写了 3 个（`overwrote=3`），
但 **`ENGRAM-PAGELESS` 仍为 8** ⇒ 剩下的缺页属 **`unavailable` 类**（真 token 拿不到），**改 mode 修不了**。

**已撤回的错误量化**：`logs/090` 的"有效覆盖率 0.5%"是拿**两个同名不同义**的计数器相除
（`build_prev_tok` 的 18 对全扫描 vs `apply_repairs` 的 plan 口径）⇒ `logs/097` 已撤回。

**已就绪的下一步**：**plan 口径 DIAG 臂**（改动已落地，md5 `bf56bc13` / `6766424c`）：
`[ENGRAM-PLAN-DIAG]` 会逐**计划槽位**打 `q/ntok/tok/reason`，并输出**可用率 / 修复率**。
判读预案见 `logs/097 §5` 三支。

---

## 5. 硬性红线（**违反即回滚**）

1. ★ **绝不发 PR / issue / 评论**；**绝不 push 到 `vllm-project/*`**（只能推 `chiro2001/*`）。
2. **绝不写 `upstream-v41/`**（已冻结只读）；新产物一律落 `a2/`。
3. 占卡走锁：`bash ~/projects/dsv41-upstream-pr/tools/a3_chip.sh <c0|c1|c2> --name X -- <cmd>`；
   **退出码 75 = 没抢到，不是失败**。★ `R_8card_int8/scripts/run_arm_r8.sh` 会**自己拿 c0 锁**。
4. **绝不碰** `dsv41-a3` / `mooncake-master` / 别人的容器 / **Phy-ID 1–7**。
5. **绝不手设 `ASCEND_RT_VISIBLE_DEVICES`**；**绝不用 `/tmp`**（用 `a2/scripts/tmpdir.sh`）。
6. **起服前先 `df -h /dev/shm`**（满 ⇒ `OSError [Errno 28]`，极易误判成"补丁坏了"）。
7. 传文件走 `tools/cos-xfer.sh`（**不要 scp**）。
8. 结论必须标 **【实测】/【推断】/【未确认】**；**不许用相邻数字顶替缺的那格**。
9. ★ **改交付件要改"工作区源头"**（`a2/agents/<agent>/patches/…`），
   **不要改发布仓里的目标文件**（那是 `prepare_publish.sh` 的**产物**，下次发布会静默覆盖）。
10. ★★ **判据必须绑定到唯一对象，且判据自身不能出现在被测集合里** —— 本仓已因此栽过 9 次：
    `079 §3` 提示文本自污染 / `083 §2` 跨运行 sha / `084` 聚合粒度 / `085` 配置选错路径 /
    `087` 跨相位跨 steps / `089` 差第三个变量 / `093` 跑错文件 / `099` 别人的 `health=200` /
    `100` `pgrep` 自匹配。**可操作做法：比数前先 diff 双方 `meta.txt`；结论只认代码痕迹。**

---

## 6. 下一步（按优先级）

| # | 事项 | 为什么 | 成本 |
|---:|---|---|---|
| 1 | ★ **plan 口径 DIAG 臂**（`TRUE_TOKENS=1` + 真分量 + `V41_ENGRAM_PLAN_DIAG=3`） | 定死 `unavailable` 的成因，决定"历史正确"能否达成 | ~25 min |
| 2 | 若 #1 指向可修 ⇒ **修后复跑**（`logs/097 §5` 前两支） | 收尾唯一质量缺口 | ~25 min |
| 3 | **档 C + `TRUE_TOKENS=0`** 一格 | 性能回退的**严格单变量归因**（现为【推断·强】） | ~20 min |
| 4 | **8 卡口径的 KV 逐字节保真**（把 `078` 的单卡探针叠到 8 卡 runner） | `071 §A1` 的 8 卡口径仍缺 | ~40 min |
| 5 | **`cc>1` × 长前缀取回** | 现压测未同时做取回 + 高并发 | ~30 min |
| 6 | A2 窗口：`git pull` → build v9 → 指纹门 → 起服 → 十道门 | 上车（`docs/A2-DEPLOY-NOW.md` 已含 6 条必读） | ~45 min |

★ A2 侧唯一"必须现在做"的是 **`git pull`**（含 `074` 的档 B 静默无卸载修复 + `v9` 镜像）。

---

## 7. 关键路径

```
本机工作区      ~/projects/dsv41/                  （不是 git 仓）
发布仓          ~/projects/dsv41/dsv41-release    （chiro2001/deepseek-v4.1-flash-ascend910B）
                分支 feat/kv8-dram-offload-pending  HEAD=698ce4d  提交数=228
  ★ 发布流程    cd a2 && PUBLISH=1 bash scripts/prepare_publish.sh
                cd ../dsv41-release && git add -A && 零泄漏扫描 && git commit && git push
A3 远端         A3-node1；本任务用 Phy-ID 8–15（1–7 是别的租户）
  clone         ~/projects/dsv41-release-git                ★ 落后，用前 pull
  载体          ~/projects/dsv41-upstream-pr/{shadow-pkg, agents/R_8card_int8, agents/S_graphfix, agents/X_integrate}
  臂产物        agents/R_8card_int8/out/<TAG>/  ；日志 shadow-pkg/results/r8_<TAG>_*/serve.log
```

---

## 8. 必读（按重要度，共 8 篇）

| 日志 | 为什么必读 |
|---|---|
| **`101`** 收官判决 | 四轴"怎么算通过"的完整判据链 |
| **`096`** fit 臂判决 | 卸载三判据 3/3（`BlockRemoved:CPU` 事件 0 次）的来历 |
| **`075`/`077`** Engram 修复 + A3 判决 | 唯一那个"必修才跑得起来"的 P0 |
| **`097`** DIAG 结果 + 撤回 0.5% | 判据层最重要的一次自我更正 |
| **`092`** int8 省 24% 宿主 | 唯一与池大小无关的容量结论 |
| **`099`** 端口冲突 / **`100`** pgrep 自匹配 | 两道门（G4）与巡检纪律的来历 |
| **`085`/`093`** device-index 绕过 / dsa 挂错 | 两次"通过但是假通过" |
| **`docs/4AXIS-SUMMARY.md`** | 本文的对照版（更偏结论与参数） |

★ 日志索引：`a2/logs/README.md`（每条一句话）。★ 加新日志必须先 `ls a2/logs/` 确认空号，
并加进 `a2/scripts/prepare_publish.sh` 的 MAP（否则 fail-closed 覆盖率门会拦住发布）。
