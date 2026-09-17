# 已修复：静态内核被静默禁用，代价 4–5 ms/step（12%）

> 发现时间：2026-09-16 02:50–03:00　场地：A3-node1 chips 8-15
> 症状：**同一份配置、同一份模型，有的会话 34.4 ms/step，有的 38.9–39.3 ms/step**
> 根因：`LOCAL_WORLD_SIZE` 未进入 `os.environ` ⇒ torch_npu 静默禁用 static kernel
> 结论：**已在 `serve_a21.sh` 修复**（容器启动即注入 `LOCAL_WORLD_SIZE=8`），实测恢复到 34.2 / 35.2

---

## 1. 症状与困惑

同一配置（`static_kernel=1` + `npugraph_ex=1` + `enable_fused_mc2=1` + `SP_TOKENS=7` + jemalloc）
在两个时间段给出明显不同的墙钟：

| 时间 | 会话 | 32K ms/step |
|---|---|---|
| 00:42 – 01:24 | `fmc2b` / `pipe_on` / `pipe_off` / `f_stock` | **34.35 – 34.48**（5 次测量一致） |
| 01:55 – 02:52 | `fmc2`(sweep) / `nohot` / `profA` / `now` | **38.53 – 39.25** |

排查并**排除**的项：

| 候选 | 证据 | 结论 |
|---|---|---|
| profiler 采集开销 | 同会话开/关只差 3.7–6.6 ms，且 `HOTSPIKE=0` 会话（无 profiler）同样慢 | 排除 |
| 热重载闸门 | `HOTSPIKE=0`（完全无 hot_hooks）同样慢 | 排除 |
| `PROFILE=` 配置 | `PROFILE=1` 但不采集 → 同样慢 | 排除 |
| CPU 降频 | cpus 320-639 全部 `2899980–2900022 kHz`（= max），governor=`performance` | 排除 |
| 温度/功耗限流 | chips 4-7：49–52 °C、168–178 W，dmesg 无 thermal/throttle | 排除 |
| 外部 CPU 争抢 | cpus 320-639 实测忙碌 **0.2%**，cpus 0-319 **0.1%**，load 2.6 | 排除 |
| 新增大进程 | 01:15–02:05 窗口内只有 0 RSS 的内核线程 | 排除 |
| 显存/swap | 宿主机 Mem 238GB free，swap 3GB 全用但非增量 | 排除 |
| 设备侧变慢 | 两次 profile 的设备 busy/comm/compute 差异 < 0.5 ms/step | 排除 |

---

## 2. 根因

### 2.1 决定性差异：静态内核编译有没有发生

| 会话 | `static kernel compile start` | `torch.compile took` | ms/step |
|---|---|---|---|
| `a21_fmc2_0019`（00:19 起服） | **4** | **22.38 s** | 34.40 |
| `a21_s7p_0059`（00:59 起服） | **3** | **14.15 s** | 34.35 |
| `a21_fmc2_20260916_0149`（01:49） | **0** | 2.11 s | 38.04 |
| `a21_nohot_0213`（02:13） | **0** | 2.16 s | 37.95 |
| `a21_profA_0225`（02:25） | **0** | 2.18 s | 38.15 |

并且慢会话里出现了 **24 次** 这条警告（快会话 **0 次**）：

```
torch_npu/dynamo/npugraph_ex/_acl_concrete_graph/static_kernel.py:650: UserWarning:
  Environment variables 'LOCAL_WORLD_SIZE' is not set in a multi-card context.
  As a result, the static kernel feature will be disabled.
```

### 2.2 机制

`static_kernel.py:647` 的判据：

```python
def _is_multicard_env_valid() -> bool:
    if not _is_single_card() and "LOCAL_WORLD_SIZE" not in os.environ:
        warnings.warn("... static kernel feature will be disabled ...")
        return False
    return True
```

而 vllm-ascend 本应在 `compiler_interface._configure_backend()`（第 97 行）补上它：

```python
if ascend_compilation_config.enable_static_kernel:
    if "LOCAL_WORLD_SIZE" not in os.environ:
        actual_local_world_size = (local_world_size * data_parallel_size_local)   # 8 * 1
        os.environ["LOCAL_WORLD_SIZE"] = str(actual_local_world_size)
        logger.info_once("Setting LOCAL_WORLD_SIZE=%d for static kernel ...")
```

实测：**这段补设在某些会话里没有生效**（日志里 `Setting LOCAL_WORLD_SIZE` 与
`enable_static_kernel is enabled` 均为 0 次），于是 torch_npu 判定"多卡但未设置该变量"，
**静默禁用静态内核**——不报错、不退服，只是慢。

> `logger.info_once(scope="global")` 可能带持久化去重，所以**日志缺失不能单独作为证据**；
> 但 24 次 vs 0 次的 `UserWarning` 是直接的、每次进程都会打的证据。

---

## 3. 修复与验证

### 3.1 修复（`scripts/serve_a21.sh`）

容器启动即注入该变量（值与 `_configure_backend` 的算法一致：`TP(8) × DP(1) = 8`）：

```bash
LOCAL_WORLD_SIZE_ENV=${LOCAL_WORLD_SIZE_ENV:-8}
MOUNTS="-e LOCAL_WORLD_SIZE=$LOCAL_WORLD_SIZE_ENV -e V41_ENGRAM_HOST_RESIDENT=1 ..."
```

### 3.2 验证（03:00 起的会话）

| 标志 | 修复前（慢会话） | **修复后** |
|---|---|---|
| `static_kernel.py:650` 警告 | 24 | **0** |
| `static kernel compile start` | 0 | **3** |
| 容器内 `LOCAL_WORLD_SIZE` | 未设置 | **8** |

| 上下文 | 修复前 | **修复后** | 快时代参考 |
|---|---|---|---|
| 8K | 37.95 – 38.15 | **34.20** | 34.40 |
| 32K | 38.94 – 39.25 | **35.18** | 34.35 |

KV 容量不变（3,388,563）。

---

## 4. 这条结论的价值与影响面

1. **直接收益 4–5 ms/step（12%）**，且是"配置本来就要开、却被静默关掉"的功能 —— 属于必须修。
2. **它解释了此前一批"慢时代"实验的基线偏移**（01:55–02:52 的所有测量都在禁用静态内核的基线上），
   那些窗口内的对照（含 mc2hier 尝试、sweep base）**不能与 34 档直接比较**。
3. **可复用的排障手法**：当"同一配置、不同会话、性能稳定差一档"时，
   先查 `static kernel compile start` 计数与 `static_kernel.py:650` 警告计数——
   静态内核是**静默降级**，不会有任何错误码。
4. **对 A2 交付的提醒**：A2 若直接用 `vllm serve` 而不经过封装脚本，同样可能命中该静默降级。
   建议把 `LOCAL_WORLD_SIZE` 注入写进启动脚本，并在起服后断言
   `static kernel compile start > 0`（或该警告 == 0）。

---

## 5. 证据路径

| 内容 | 路径 |
|---|---|
| 修复代码 | `scripts/serve_a21.sh`（`LOCAL_WORLD_SIZE_ENV` 注入 + `MOUNTS`） |
| 修复后会话 | `logs/perf/a21_lws_0251_serve.log` |
| 修复后测量 | `logs/perf/a21/measure_lws.log`、`logs/perf/a21/p42_t4_quote_*_lws_*.jsonl` |
| 慢会话（对照） | `logs/perf/a21_nohot_0213_serve.log`（含 24 次警告）、`logs/perf/a21_profA_0225_serve.log` |
| 快会话（对照） | `logs/perf/a21_s7p_0059_serve.log`、`logs/perf/a21_fmc2_0019_serve.log` |
| torch_npu 判据 | 容器内 `.../npugraph_ex/_acl_concrete_graph/static_kernel.py:647-656` |
| vllm-ascend 补设逻辑 | 容器内 `vllm_ascend/compilation/compiler_interface.py:85-125` |
