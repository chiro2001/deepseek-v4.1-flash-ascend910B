# 热重载安全闸门（避免推理中热更新导致卡死）

> 提出：用户（"热重载如果在推理过程中被请求更新，可能会导致卡死"）
> 场地：A3-node1 chips 8-15，容器 `dsv41-a21-perf`，端口 8020
> 结论：**旧版本确实没有 idle 闸门，已补齐三层防护并逐条实测**

---

## 1. 旧版的真实风险（已确认，不是推测）

旧 `hot_hooks.py` 里两处热替换（`_reload_patched_modules` / `_reload_engram_hbm`）**只挡了"图还没捕获"这一种情况**：

```python
present, _ = _graphs_present()
if not present:
    return          # 只在起服/捕获期返回
# ... 之后直接 exec，没有任何 idle 检查
```

服务 READY 后 `_graphs_present()` 恒为 True ⇒ **热替换在任何时刻都会被放行**。两条风险：

1. **半新半旧**：某一步可能在"一半旧代码、一半新代码"的模块状态上执行。
2. **rank 分歧（更危险）**：8 个 rank 各自轮询（周期 300 ms），reload 时刻必然抖动；若抖动窗口内有一步在跑、而新代码改了集合通信次数/顺序 ⇒ rank 间不一致 ⇒ **HCCL 挂死**。

另外，清图路径（`_clear_graphs`）历史上就有实测事故：在飞请求期间 clear 会报
`rtModelDestroy execution failed, reason=model is running` 并把服务卡死。

---

## 2. 三层防护

| 层 | 机制 | 位置 | 性质 |
|---|---|---|---|
| **G-GO** | **外部 GO 令牌**：只有 `/tmp/v41_hot_go` 存在且 **≤5 s 新鲜**时才允许 reload | `hot_hooks._guard_go_ok()` | **硬保障**（不依赖进程内状态） |
| **G-idle** | 连续空闲 ≥`HOTSPIKE_RELOAD_IDLE_S`（默认 1.0 s）才允许 | `hot_hooks._guard_can_reload()` | 兜底 |
| **G-settle** | reload 完成后，本进程**下一次前向入口**至少等到 `t_reload + 2 s`；每个 rank 都如此 ⇒ 任何 rank 开始新步都不早于"所有 rank 换完" | `ACLGraphWrapper.__call__` / `NPUModelRunner.execute_model` | 消除 rank 分歧窗口（不需 barrier） |
| HOLD | `/tmp/v41_hot_hold` 存在时完全冻结热更新 | `_guard_hold()` | 运维急停 |

四道闸门统一作用于**全部四个热替换入口**：`_reload_patched_modules`、`_reload_engram_hbm`、`_reload_graph_modules`、清图路径。

配套脚本 `scripts/hot_apply.sh`：**在进程外**用 vLLM 自己的 `/metrics` 判定空闲（连续 `num_requests_running==0` 达 `IDLE_S` 秒），空闲后才写 GO 令牌 → 触发热更新 → 完成后删令牌。服务不空闲时它直接 `ABORT` 并退出 3。

---

## 3. 实测（逐条）

### 3.1 负向：推理中触发热更新 ⇒ **被拦住**

请求在飞（`running=1.0` 持续 20 s+）时 `touch hot_hooks.py`：

```
[reload-guard] module-reload:vllm_ascend.ops.fused_moe.token_dispatcher: GO 闸门未开（no-go-file）⇒ 不热更新
应用次数：0
```

### 3.2 负向：推理中跑 `hot_apply.sh` ⇒ **等待而非放行**

```
[hot-apply] 等待连续空闲 3s ...      # 一直等到超时，未发令牌
```

### 3.3 正向：空闲后跑 `hot_apply.sh` ⇒ 正常落地

```
[hot-apply] 已连续空闲 3s
[hot-apply] 发 GO 令牌 -> /tmp/v41_hot_go  →  1789494437
[reload-guard] enghbm-reload: 完成，settle=2.00s
[enghbm-reload] #1 rebound=['NodeShardedEngram','EngramQueryGroup'] patched={'NodeShardedEngram':5,'EngramQueryGroup':5}
[reload-guard] module-reload:token_dispatcher: 仅空闲 0.00s（需 1.00s）⇒ 继续等待   ← G-idle 生效
...（1 s 后下一轮）module-reload: 完成，settle=2.00s
```

### 3.4 稳定性：推理中热更新后服务存活

第一次（防护不完整时）在 295K-token prefill 期间热更新，请求**正常返回**：
`http=200 t=82.0s`、`finish_reason: length`、600 token 连贯中文；`health=200`。
这同时说明：即使防护漏掉一次，也**未必**立刻挂死——但这不是可依赖的性质，所以闸门必须保留。

---

## 4. 过程中发现并修掉的两个真 bug

### 4.1 `_GUARD` 被 re-exec 重置 ⇒ 闸门形同虚设

模块级 `_GUARD = {...}` 会在每次热重载 `exec(src, mod.__dict__)` 时**重新赋值成新对象**；而已装好的 wrapper 闭包仍指向**旧字典** ⇒ wrapper 增旧 dict、`_guard_can_reload` 读新 dict ⇒ 永远读到 0 ⇒ 从不拦截。
**修法**：把状态放到 `builtins._DSV41_HOT_GUARD`（不属于任何被 reload 的模块，天然跨 exec 存活），模块级 `_GUARD` 只是它的别名。

> 这与该文件自己的告诫一致——"热模块每次 reload 都会被 exec 重跑，模块级状态会被清零，判据必须无状态实时探测"。第一版闸门恰恰违反了它。

### 4.2 `ACLGraphWrapper` 覆盖不到 prefill

`ACLGraphWrapper.__call__` 只覆盖 **decode 图 replay**；**prefill 是 eager 的、不走它**。
证据：128K/295K 长 prefill 期间计数恒为 0，`enghbm-reload` 被放行、而同轮 `routed_experts` 被拦
（`enghbm-reload: 完成` 8 次 + `module-reload:token_dispatcher: 仅空闲 0.00s`）。

**修法**：改挂 `NPUModelRunner.execute_model`（`worker.py:683` 的每步入口，覆盖 prefill+decode），保留 `ACLGraphWrapper` 那层。
**残留**：实测 `execute_model` 钩子仍**没有输出计数**（`[reload-guard] execute_model called` 一条都没有）⇒ 该版本的每步入口选点仍不准。
**这不影响安全结论**——正因为进程内计数不可靠，才把 **G-GO（外部令牌）** 作为硬保障；G-idle/G-settle 只作兜底。**待办**：把 `execute_model` 的挂点再校准一次（列出运行时真实的 runner 类与调用栈）。

---

## 5. 使用方式（运维规范）

```bash
# 推荐：自动等空闲再热更新
bash scripts/hot_apply.sh                 # 默认连续空闲 3s；IDLE_S=5 可加严

# 急停/冻结（比如要做精度评测时）
touch /tmp/v41_hot_hold                   # 冻结全部热更新
rm -f /tmp/v41_hot_hold

# 手工（仅在确认服务空闲时）
docker exec dsv41-a21-perf bash -c "date +%s > /tmp/v41_hot_go"
curl -X POST http://127.0.0.1:8012/hotpatch
```

**新增环境变量**：`HOTSPIKE_REQUIRE_GO`(默认1) `HOTSPIKE_GO_FILE` `HOTSPIKE_GO_FRESH_S`(5)
`HOTSPIKE_RELOAD_IDLE_S`(1.0) `HOTSPIKE_SETTLE_S`(2.0) `HOTSPIKE_HOLD_FILE`

---

## 6. 顺带拿到的两条结论

1. **非 mtpq 版 KV = 2,700,814 < 3M ⇒ 否决**。`v41-w4a8-engram-dr-vision`（BF16 draft）实测
   `GPU KV cache size: 2,700,814 tokens`，而同 util 下 qrot-mtpq 版是 3,388,441–3,557,104。
   **mtpq 省下的 2.42 GB 正好是 KV 跨过 3M 门槛的来源** ⇒ "用 BF16 draft 换接受率"这条路被 KV 约束否决。
2. 该非-mtpq 会话的 `Mean acceptance length: 1.55`（服务自身 metrics），明显低于 mtpq 版的 2.7–2.9，
   但该请求是 295K-token 古典中文续写（低可预测性内容），**不能据此判定 mtpq 更好**，需要同内容对照。

---

## 7. 证据路径

| 内容 | 路径 |
|---|---|
| 闸门实现 | `draft_hot_sp/hot/hot_hooks.py`（`_guard_state`/`_guard_go_ok`/`_guard_can_reload`/`_guard_mark_reloaded`/`_install_reload_guard`） |
| 运维脚本 | `scripts/hot_apply.sh` |
| 负向证据（GO 拦截） | serve 日志 `logs/perf/a21_nomtpq_0130_serve.log`，搜 `GO 闸门未开` |
| 正向证据（落地） | 同日志，搜 `完成，settle` / `仅空闲 0.00s` |
| 存活证据 | `http=200 t=82.0s`、`/tmp/gt2.json`、`health=200` |
| KV 证据 | 同日志 `GPU KV cache size: 2,700,814` |
