# AGENTS.md —— A2 工作区规则（动手前必读）

> 本文件给 **AI 代理**看。人类入口在 [`README.md`](README.md)，
> 交接快照在 [`HANDOVER.md`](HANDOVER.md)。
> 规则是从 `../upstream-v41/AGENTS.md` 精简来的 —— **提 PR 相关的红线已删除**，
> 换成了"A2 上线"相关的纪律。

---

## 0. 现在在做什么

**把内网这台 A2（8×910B3）用起来。** 两个方向：DRAM KV 卸载、KV cache 低精。
**不再以向上游提 PR 为优先**（该材料已冻结在 `../upstream-v41/`）。

---

## 1. 硬性红线（违反即回滚）

| # | 规则 |
|---|---|
| **1** | **绝不**往 `../upstream-v41/` 写新的日志/脚本/产物（那里已冻结；只读参考） |
| **2** | **绝不**在 `/tmp` 建任何东西（tmpfs，26 GiB 上限，撞了会 OOM 掉全部进程，有过 4.5 h 停机事故） |
| **3** | **绝不**手设 `ASCEND_RT_VISIBLE_DEVICES`（用锁脚本注入） |
| **4** | **绝不**用 `NAME=dsv41-a3` 起容器（`serve_a2.sh` 里有 `docker rm -f "$NAME"`，会**物理删掉用户的容器**，可写层里的补丁不在挂载里，删了就没了） |
| **5** | **绝不**动 `dsv41-a3` / `mooncake-master` / 别人的容器；`dsv41-a3` 当前是 `Exited`，**保持原样** |
| **6** | 占卡必须走锁；**退出码 75 = 没抢到，不是失败** |
| **7** | 结论必须标 **【实测】/【推断】/【未确认】**；**不许用相邻数字顶替缺的那格** |
| **8** | 跨机传文件**走 coscli**，不走 ssh 管道（`scripts/cos-xfer.sh`） |
| **9** | 用 `apply_patch` 改文件；不用 `rm -rf`；删自己建的临时目录前先 `du -sh` |
| **10** | 只写自己的目录（`agents/<你的代号>/`），`docs/`、`logs/`、`scripts/` 由主代理统一写 |
| **11** | ★★ **交付配置必须保留投机解码**（`--speculative-config dspark`）；**"关投机换容量"（⑤a / `SPEC_ON=0`）已否决**（用户 2026-09-22 决策：A2 单流场景 DSpark 收益不可替代）。关投机的臂**只可作诊断对照**且必须显式标注，**不得**写进推荐路线、交付配置或结论表 |

---

## 2. 两台机器

| 机器 | 状态 | 怎么用 |
|---|---|---|
| **A2**（内网 8×910B3） | ⛔ **本机 ssh 不可达** | 命令由**用户粘贴执行**；数据由用户带回或走 COS |
| **A3**（A3-node1） | ✅ `ssh A3-node1`（若挂起用 `ssh -o ControlPath=none A3-node1`） | 单卡槽位 c0/c1/c2；Phy-ID 8–15 有时被占 |
| A3 的 Phy-ID 8–15 | ⚠️ **用前先 `npu-smi info`** | 子代理 D_off8 可能正在用 |
| ★★ **A3 的时钟**（2026-09-22 14:0x 实测） | ⚠️ **比本机慢约 7 分钟**（两边都是 CST +0800，是**漂移**不是时区） | ★ **不要拿本机时间推断远端进度** —— 我因此把「起了 2 分钟」读成「编译了 26 分钟」，**差点误判成卡住**。看进度要读远端的 `date` + **日志字节增长**，不要用本机时钟做减法 |

**★ ssh 挂起的坑**：A3 的 ControlMaster 会僵死（master 进程活着但连接黑洞，所有 ssh 静默挂到超时）。
症状出现时：`ssh -O exit A3-node1`，或直接 `-o ControlPath=none`。

---

## 3. 临时空间协议

```bash
source scripts/tmpdir.sh <任务名>    # → $TMPDIR = ~/tmp/YYYYMMDD/<任务名>
```

* **唯一合法位置**是 `~/tmp/<YYYYMMDD>/<任务名>/`（在 btrfs 盘上，不占内存预算）；
* `pip` / `uv` / `pytest` / `torchinductor` 都会跟随 `TMPDIR`；
* 干活前先看余量：`df -h /tmp` 与
  `cat /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/memory.current`。

---

## 4. A2 特有的三条硬约束（先记住，别踩）

1. **`host_mem_pool = 0`** —— A2 的驱动**没有** host 内存池。
   已实测：整表注册 17 min 后 `ret=207001`（OOM 语义），**而同时刻宿主 `MemAvailable` 还有 703 GiB**。
   ⇒ 任何依赖 `aclrtMallocHost` / `pin_memory` 的方案，**在 A2 上都先怀疑这一步**。
2. **A2 默认 MoE 通信是 ALLGATHER**（不是 A3 的 ALLTOALL）——
   所以 `TokenDispatcherWithAllGather` 那条路径**在 A2 上是热路径**。
3. **A2 的 KV 账**：4421 B/token/rank；`cpu_bytes_to_use` 是服务级总量、CPU 侧 8 份副本
   ⇒ `tokens = 值 / (4421 × 8)`；宿主余量 442 GiB ⇒ **建议 ≈260 GB**（×1.7 记账后）。

---

## 5. 起服前必查

* `npu-smi info` 确认目标 die 空闲（HBM 全 0、无进程）；
* 要起新服务 ⇒ **残留 `VLLM::` 进程必须为 0**（用 `docker stop && docker start` 清，**别用 `pkill`**）；
* 有服务在跑 ⇒ `ps` 里应有 **10** 个 `VLLM::`（正常形态）；
* 容器删之前**先确认产物已落盘**（有过 "probe 静默失败 + 容器被删 = 数据丢失" 的事故）。
* ★★ **`df -h /dev/shm`** —— 三个 `prbench-*` 容器的 `/dev/shm` **只有 64 MiB**。
  一个残留大文件（例如 `p1_regfile.bin`）就会**打满整个 tmpfs**，
  而报错长这样：**`OSError: [Errno 28] No space left on device`，出现在 `_multiprocessing.SemLock(...)`**
  ⇒ **极易被误判成"我的补丁坏了"**。（`2026-09-22 06:5x` 实测踩过：c0 因此对所有人不可用，靠 `df -h` 才定位。）
  **诊断口诀：起服失败先 `df -h /dev/shm`**；清理前先确认**无进程持有**（遍历 `/proc/*/fd`）且**不是别人的产物**。

---

## 5b. ★★ 探针纪律（`035→036→038→040→043` 五轮的共同教训）

**今晚有四轮结论被下一轮推翻，全部源于"判据没有判别力却被当结论"。** 三条硬规则：

> ★★★ **总纲（2026-09-22 深夜补，共 14 条之后归纳）**：
> **"探针/守卫的生效范围"经常比作者以为的小。** 历次踩的坑可归为四类：
> ① **装晚了**（hook 在 import 之后才装）→ 第 1 条
> ② **接错了对象**（多套实现/多套标记）→ 第 10 条
> ③ **在错的时机求值**（调用点实参立即求值、`except: continue` 静默丢数据）→ 第 11、14 条
> ④ **覆盖了错的流量**（trace 全来自 warmup/capture、读到死缓冲）→ 第 15 条
> ⇒ **每次插探针前，先明确回答四个问题**：
> **装在哪？接的是哪一套？什么时候求值？覆盖的是哪段流量？**

1. **先装 hook，再 import/exec 目标模块。**
   任何"包裹某个模块"的探针，必须在目标模块被 import **之前**装 hook
   （例：PGP/P2 的 `sitecustomize` 自己会 `import pgp_manager`，hook 装晚了就**永远不触发**）。
2. **只打"已装载"的探针 = 未验证。**
   必须在热路径上打**版本号 + 前 N 次调用的 trace**，用来**证伪"探针没生效"**。
   实例：`L3_8card/kv_bytecheck.py` 钩的 `store/load` **在现行代码里根本不存在**（真入口是 `submit_store/submit_load`），
   它却照样打"已装"并置标志位 ⇒ **若跑了会得到"全绿但比对 0 次"**，比不跑更危险。
3. **判据必须在"反例臂/正确臂"上对称跑一遍。**
   * **阳性对照**：人为制造目标缺陷 ⇒ 探针**必须报警**（否则是"看不见所以全 0"）；
   * **对称实验**：同一条探针在 ✅ 与 ❌ 两臂上跑 ⇒ **若两臂给出同样的数，它就是假阳性，不是判别量**。
   实例：`load_row_unknown=5376(❌) / 5392(✅)`，**行号逐字相同** ⇒ 假阳性（真因是"最后一个 store job 的完成事件没被观测到"）。

4. ★★ **`sys.meta_path` finder 的"多 target 陷阱"**（`044` 实测，今晚第 5 个坑）
   一个 finder 实例**管多个 target** 时，若在 `exec_module` 里做 `sys.meta_path.remove(self)`（**永久摘除**），
   则**第一个 target 被 import 之后，其余 target 的钩子全部静默失效**。
   ⇒ **症状会伪装成"那个函数根本没被调用"**：`L_dmafix` 的 ring 探针因此报 `ring_calls=0`，
   差点被写成结论（`043` §4）；实际是探针自己被关了。
   **正确写法**：① **一个 target 一个 finder 实例**（或重新插回）；
   ② **按实例真实类型打补丁**（不依赖 import 顺序）；③ **每个 target 单独打"已生效"横幅**，
   且横幅里带**实际被替换后的函数名/地址** —— 不能只看"装载完成"。
   实例：`N_ring` v2 复现了"`dsa_v41` 打了已生效横幅、但 `compressor` 的类补丁没打上"。

5. ★★ **`PathFinder.find_spec` 会绕过别人的"整文件重定向"**（`046` 实测，今晚第 6 个坑）
   若别人用 finder 把某个模块**整文件换成另一份**（例：PGP 把 `…offloading.scheduler`
   换成 `pgp_scheduler.py`），你再用 `importlib.machinery.PathFinder.find_spec` 去找它，
   会**绕过那道重定向、加载到上游原版** ⇒ 你会在"上游的旧断言"上炸，而误以为是自己的补丁坏了。
   **正确写法**：用 **`importlib.util.find_spec`**（它尊重已有的 `sys.meta_path` 链）
   + **`try/finally` 把自己插回 `sys.meta_path`**。
   实例：`P_ringfix` 因此加载了上游原版 `scheduler.py`，在旧
   `assert isinstance(kv_cache_spec, FullAttentionSpec)` 上炸（该断言见 `logs/010` §4.3）。

6. ★★ **`cp -f` 会穿透符号链接，覆盖别人的文件**（`046` 实测，今晚第 7 个坑）
   overlay 目录里若某个文件在上一轮是**符号链接**，`cp -f` 会**写到链接的目标**（= 别人的目录），
   而不是替换那个链接。实例：`P_ringfix` 因此覆盖了 `N_ring` 的 `probe/sitecustomize.py`
   （已按本地原件逐字节还原、md5 对账一致）。
   **规则**：**overlay 里凡是上一轮可能是符号链接的位置，一律先 `rm -f` 再 `cp`**（或 `cp --remove-destination`）。

7. ★★ **"证据收集路径"本身也会骗人：自检门判"失效"之前，先确认它在找对地方**
   （`T_draftceiling` 2026-09-22 12:0x 实测，今晚第 8 个坑）
   现象：某臂的**引擎日志里明明有 128 行 `P1_pinned ret=0`**，但 runner 的自检 grep 去
   `$PKG/results/$RID/serve.log` 与 `$LOGD/$TAG.serve_a2.log` 找 —— 而这次 `serve_a2.sh` 把日志写到了
   **`$OUT/serve.log`**（`OUT` 被 `run_arm` 导出、nohup 继承）⇒ grep **假阴性 = 0** ⇒ 判 FATAL ⇒
   **容器被删、压测根本没跑**。
   ⇒ **这与第 1/2 条同源**（"看不见"被当成了"没生效"），但**失败点在链路更靠后**：
   探针装对了、日志也打了，**是"收集路径"错了**。
   **规则**：① 任何自检门报"失效"时，**先打印它实际读的那个路径**（`ls -l` + `wc -l` + `grep -c` 三条一起给），
   再下结论；② 起服脚本会把日志写到哪，**以 `serve_cmd.txt` / `inner.sh` 里现算的 `$OUT` 为准**，不要凭记忆写死路径。

8. ★★ **留在磁盘上的"开关文件"会让基线臂静默变成实验臂**（同上，今晚第 9 个坑）
   `draft_block_64.flag` 这类**文件开关**，若上一轮创建后被 `kill`，它**还在磁盘上**
   ⇒ 下一轮本该跑"基线（128）"的臂会**静默地按 64 跑** —— 而输出、容量、日志**全都看起来正常**。
   **规则**：① 链式跑臂时，**链首与链尾各清一次**所有 `*.flag` 开关；
   ② **每一臂的日志里必须打印实际读到的开关值**（不能只打印"我打算用哪个值"）。
   实例：`T_draftceiling` 自查发现并修掉（这是本项目第 3 次踩"静默 no-op"，前两次见 `048`/`055`）。

9. ★★ **「配置档位」必须自报 + 用「容量指纹」交叉核对 —— **少挂一个文件就静默降档**
   （`T_draftceiling` 2026-09-22 16:3x 实测，今晚第 10 个坑）
   现象：它的 8 卡「档 D」臂 **rc=0、四组判据全过**，但**容量是 427,643**
   （档 C 的数；档 D 应是 **485,610**）。根因：8 卡 runner 的 `[R8-INT8]` 块**只挂 4 份公共件**，
   **不含 `models/deepseek_v41/model.py`** —— 而「长 KV 变 int8」的唯一调用点
   `long_kv_plane_kwargs()` **就在 model.py:582** ⇒ 它**从未被调用** ⇒ 长 KV 保持 BF16
   ⇒ **实际跑的是档 C**，**全程无报错**。
   **规则**：① **每一臂的日志必须打印「本臂 = 档 X」**（不是「我打算跑档 X」）；
   ② ★ **用容量指纹交叉核对** —— B/C = 427,643、**D = 485,610**；
   ③ ★ 注意**容量区分不了 B 与 C**（两者同值）⇒ 它只能抓「该是 D 却拿到非 D」这一格；
   ④ **未经实测的开关组合一律 fail-closed**（`serve_a2_offload.sh` 已加三条：
   `SWA=1 无 RING` / `FULL=1 无 SWA` / `PREFILL=1 无 FULL` ⇒ 拒绝起服）。
   **教训**：这是本项目第 **4** 次踩「静默 no-op」（前三次见 `048`/`055`），
   而这次的特殊之处是 —— **它在「功能判据全过」的伪装下发生**。

**⇒ 结论必须标【实测】/【推断】/【未确认】，且【实测】只用于"有判别力的判据跑出来的数"。**

---

10. ★★★ **同一逻辑常常有【多套实现 / 多套标记】—— 插探针前先数定义个数**
   `grep -n "def <目标名>" <文件>` 若得到**两行**，说明存在 import-time 整段替换（或双实现）。
   实例（2026-09-22 一晚踩**两次**）：
   * `kv8_ori_plane` 有 **模块级（:505）+ import-time 替换（:1656）两套**；
     给"没人调用的那一套"插 trace ⇒ **恒 0** ⇒ 会得出"这条路径没被走"的错误结论；
   * **APC** 有两套实现/两套标记（`Q` 的 `[apc]` finder 横幅 vs `R8` 的内联 `[R8-INT8-TRACE]`）
     ⇒ 查错标记会得出**相反**的结论（`048` 的 APC 一度被误判为"没接上"）。
   ⇒ **硬规则**：插探针前先 `grep -n "def <名>"` + `grep -c "_<名>_"`（找替换），
     且结论必须写清"**打的是哪一套、哪个 md5**"。

11. ★★★ **任何 `except: continue` 都必须配一个计数器，并把计数器打进报告**
   —— 否则"**读不懂的数据**"会被当成"**不存在的数据**"。
   实例（2026-09-22）：流式客户端 `except json.JSONDecodeError: continue` 静默丢掉了**含 NaN 的 SSE 块**
   ⇒ `n_pieces` 从 9 掉到 3 ⇒ 表面看是"模型少产出了 token / 输出变短 / 内容改变"，
   而**真实现象是服务端返回 NaN**（`http=400 Out of range float values are not JSON compliant: nan`）。
   ⇒ 这一条与「`metrics_before` 当 `metrics_after`」是同一族：**判据自己把数据丢了**。

12. ★★ **「原始数据路径」必须是【本机已存在】的路径 —— 否则写"待 cos-xfer"**
   实例（2026-09-22）：某份日志的 §交付 直接写了 `logs/raw/<...>/`，
   而那份数据**只在 A3 上** ⇒ 本机读者会以为它在、实际 `ls` 不到。
   ⇒ **硬规则**：日志交付表里的路径**只写本机已验证存在的**；
     还没取回就写「**待 `cos-xfer`：<A3 上的路径>**」。
   ★ 同一族的还有：**"日志写完了" ≠ "数据回来了"**（两者要分别确认，别合并成一次检查）。

13. ★★★ **"更正"本身也要留痕 —— 被推翻的更正不要删，改成两段式**
   实例（2026-09-22，一晚两次）：
   * 我把"所有包里的 draft 都没真进图"推得太宽 ⇒ 被 P1 的实测推翻 ⇒ **改成两段式**（原判断 + 回撤）；
   * 我又把"档 D 的『输出变短』是探针伪影"写进了 `066c` 开头 ⇒ 被 `c2` 的 `parseErr=0` 推翻
     ⇒ 同样**改成两段式**（第一次更正 + 第二次更正）。
   ⇒ ★ **为什么留痕**：被推翻的那一步**本身就是判据链的证据** ——
     "我当时为什么那么想、哪个观测把它否掉了"，比只留最终结论更能防后人重走。
   ★ 附带：**"未触发的判据"与"被推翻的判据"是两种不同状态**，不许混写。

14. ★★★ **"探针的惰性"必须覆盖到【实参】—— 只把守卫放进被调函数是不够的**
   实例（2026-09-22，`c2` 的 NaN 探针）：helper 里加了 `is_current_stream_capturing()` 守卫，
   结果**捕获期仍然 `EE1016=7`**。根因：**`extra={...}` 是在【调用点】立即求值的**，
   守卫写在被调函数里**拦不住**调用点上的 `(sel_scale == 0).sum()`（那也是 D2H 同步）。
   **修法**：extras 改成 `lambda: {...}`（零参 callable），在守卫**之后**惰性求值 ⇒ `ee1016=0`。
   ★ **判据**：凡是探针**调用点的实参**里出现 `.sum() / .min() / .max() / int()`，就要当成
     "**可能同步**"来审 —— 与 `049` 修的 `.item()` 是同一族。
   ★ 该子代理把它做成生成器的**第 2 项自检 + 反向验证**（改回急切 ⇒ `rc=5` 被抓住）——做法对。

15. ★★★ **"读到全 0" 要先问："这是真值，还是我读到了没写过的缓冲？"**
   实例（同日）：探针打出所有张量 `fmin=fmax=0.0`，据此报"无 NaN"。
   ⚠️ 但**零张量上永远不会有 NaN** ⇒ 那条"无 NaN"是**平凡的**，不是结论。
   两条都可能：① dummy 权重下 KV 平面本来就是零；② **读到了死缓冲**。
   ⇒ **硬规则**：报"某处没有 X"之前，先证明**那段缓冲真的被写过实数**
     （最小验证：在同一探针点必打 `absmax` **且** 与写入侧的心跳对账）。
   ★ 这种"负面结论"必须写成 **【未确认】+ 前置条件**，不能写成【实测·无 X】。

## 6. ★★ 写算子 / 写 kernel / 做量化数值验证时：**必须先查 cannbot 文档**

> 用户 2026-09-22 明确要求：*"写算子的时候记得让子代理参考 cannbot 文档"*。

**cannbot 在 A3 上**（只读）：

```
~/projects/dsv41/src/cannbot/vendor/cannbot-skills/     ← skills 源码（submodule，pinned）
~/projects/dsv41/src/cannbot/plugins/model-infer-optimize/
```

**我们已有的调研与索引**（**先读这两份，不要从零翻**）：

| 文件 | 作用 |
|---|---|
| [`../../upstream-v41/../layer_bench/report/cannbot-layer-guide.md`](../../layer_bench/report/cannbot-layer-guide.md) | ★ **算子级开发与验证指南调研**（Q1–Q5） |
| [`../../engram_ref/wtgraph/docs/CANNBOT_TUNING_NOTES.md`](../../engram_ref/wtgraph/docs/CANNBOT_TUNING_NOTES.md) | ★ **perf 相关内容的完整索引**（每条都带 `文件:行号`） |

**做下面任何一件事之前，先按这两份索引去 cannbot 里查对应章节**：

| 要做的事 | cannbot 里的对应位置 |
|---|---|
| **写/改 AscendC kernel** | `ops/ops-profiling/`、`ops/torch-ops-profiler/`（**有可直接拷贝改的 4 文件模板**：`examples/layer_norm_profiler_reference/`） |
| **量化数值验证 / 精度门** | `model-infer-quantization/SKILL.md:424-451` + `references/quantization-fusion-and-benefit.md`（**有 dtype→rtol/atol 表与双阈值判定公式**） |
| **量化结构判断** | `references/quantization-structure-cards.md:174-189,191-206`（"**C8 不是改 dtype**"——prolog/cache/FA/scale 一体） |
| **KV cache / attention 布局** | `model-infer-kvcache/SKILL.md:38-52,102-110,142-153,174-199,224-242`（**PA+FA+TND 默认、block/slot 映射、sparse_mode/mask 硬约束**） |
| **torch_npu 算子签名/清单** | `model-infer-fusion/references/torch_npu_API/torch_npu_list.md:1-160` + `scripts/torch_npu_query.py` |
| **图模式边界** | `model-infer-graph-mode/SKILL.md`（**Decode 图模式、prefill 保持 eager、重编译检查**） |
| **性能拆解方法论** | `model-infer-perf-breakdown/SKILL.md:19-31,61-72,109-124,222-246`（**单 step 提取→结构拆解→五类 insight**） |
| **多流/重叠判定** | `model-infer-multi-stream/SKILL.md` + `references/timeline-overlap-check.md`（`overlap_pct ≥0.5` 才算真并行） |

★ **特别注意 `model-infer-kvcache` 那条**：它明确写了 **PA+FA+TND 的默认与 block/slot 映射** ——
这直接关系到 **KV8 的布局选择**（我们已经实测 TND KV 在 A2/A3 上没 kernel，
见 `logs/015`；cannbot 那份文档可能给出了**推荐的替代布局**）。

**在日志里要写清**：查了 cannbot 的哪一节、它建议什么、我们为什么采纳或没采纳。
