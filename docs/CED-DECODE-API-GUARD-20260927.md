# decode 半边被打挂：请求边界护栏与多图默认值（2026-09-27）

本文记录一次**把 decode 实例整个打死**的事故、它的根因，以及本次落地的三层加固。
另附多图上限（`MM_LIMIT_IMAGES`）从 1 改为 4 的理由与代价。

---

## 1. 事故

### 1.1 现场

| 项 | 值 |
|---|---|
| 时间 | 2026-09-27 00:01:48 CST（a3-21 系统时钟慢 7m03s） |
| 实例 | a3-21 `dsv41-ced-d4b`，CED P/D 分离的 **D 半边**，端口 18991 |
| 触发 | 一条**直连 18991** 的普通生成请求（带 1 张图，prompt > 128 token） |
| 结果 | 8 个 TP worker 同时 raise ⇒ `EngineCore` 退出 ⇒ `EngineDeadError`；API server 关闭 |
| 影响面 | 18992 代理对所有请求返回 `decode_backend_unavailable`（含纯文本）；整条链路不可用 |
| 恢复代价 | 重新加载 8 个 die 的权重，**20 分钟量级** |

崩溃现场（`serve.log`，33 条同时间戳）：

```
RuntimeError: Worker failed with error 'CED decoder replay exceeded 128 tokens',
please check the stack trace above for the root cause
EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.
```

### 1.2 直接原因

18991 **不是独立实例**，而是分离式部署的 decode 半边：它正常只接收由 prefill
（18990）转发、带 `kv_transfer_params` 的请求，消费别人算好的 KV，自己只做
≤128 token 的 bounded replay。

普通生成请求没有 `kv_transfer_params`，D 只能自己去 prefill。这撞上 CED replay
守卫（`experimental/ced/dsa_v41.py`）：

```python
replay_chunk = (
    _CED_DECODE_ROLE                        # 本实例是 decode → True
    and not getattr(forward_context, "in_profile_run", False)
    and metadata.swa.num_prefills > 0        # 这一步出现了 prefill
    and metadata.swa.max_query_len > 1
)
if replay_chunk and metadata.swa.max_query_len > 128:
    raise RuntimeError("CED decoder replay exceeded 128 tokens")
```

### 1.3 放大因素（本节才是要修的东西）

1. **站在 0.0.0.0 上**：18991 对全网可达（`docker inspect` 的端口映射 + 无 api-key）。
2. **失败模式是"杀引擎"而不是"报请求错"**：`raise` 在 **worker 进程**里发生。
   worker 异常让 EngineCore 直接退出 —— 一条错误请求 = 整个实例死。
3. **触发门槛极低**：任何直连 18991 且 prompt > 128 token 的请求都能命中
   （事故里就是一条探针）。

---

## 2. 加固（三层）

### 2.1 第一层：半边默认只听回环

`scripts/serve_a3_pd.sh` 现在默认 `HOST=127.0.0.1`，P/D 两个半边都只监听本机，
跨机访问走隧道或代理，不再暴露半边。要显式暴露得自己设 `HOST=0.0.0.0`，
并且应当同时处理来源限制或鉴权。

### 2.2 第二层：请求边界 400（本轮的主修）

新增 `patches/files/v41_decode_guard.py`：一个 ASGI 中间件，注册在 decode 角色上。
**没有 `kv_transfer_params` 的生成请求直接返回 400，永远进不了引擎**，
因此不可能再触发 worker 里那条致命断言。

实现用的是 vLLM **自带**的扩展点（`vllm/entrypoints/openai/api_server.py`
里 `for middleware in args.middleware`），不改 vLLM 源码：

```
serve_v2.sh   →  --middleware v41_decode_guard.decode_guard   （仅 V41_CED_ROLE=decode）
serve_a2.sh   →  -v patches/files/v41_decode_guard.py:/opt/dsv41/guards/…:ro
                 + 起服前用 PYTHONPATH 真 import 一次，失败就 die
```

行为约定：

| 请求 | 结果 |
|---|---|
| decode 角色 + 无 `kv_transfer_params` 的 `/v1/chat/completions` | **400**（`ced_decode_role_requires_kv_transfer_params`），不进引擎 |
| decode 角色 + 带 `kv_transfer_params`（P 转发） | 放行 |
| `/health`、`/v1/models`、`/metrics`、GET | 放行（不能把自己的就绪探针挡掉） |
| body 不是合法 JSON | 放行，交给下游按正常流程报错（此处不做 fail-closed） |
| prefill 角色 | 完全空转（P 本来就要收无 KV 参数的请求） |

可观测痕迹（判据，不是"env 传进去了"）：

```
[V41-DECODE-GUARD] middleware loaded: 无 kv_transfer_params 的生成请求 → 400 …
[V41-DECODE-GUARD] rejected /v1/chat/completions：没有 kv_transfer_params …；累计拦截 N 次
[serve-v2] decode API guard: ON（无 kv_transfer_params 的生成请求 → 400，不进引擎）
[DECODE-API-GUARD] 可加载（decode_guard 是 callable）✓
```

关闭方式：`V41_DECODE_API_GUARD=0`（仅供对照实验；默认 1）。

### 2.3 第三层：崩溃后不再静默

事故当时这条错误被读成"decoder 也拒绝多图"而没有停下追。现在同一条错误在
API 层就有独立日志行与明确 code（`ced_decode_role_requires_kv_transfer_params`），
`/metrics` 计数也在 `guard_stats()` 里，不需要翻 6 MB 的 serve.log 才能发现。

---

## 3. 多图：默认从 1 改成 4

### 3.1 现象与代价

原先 P/D 两侧都是 `--limit-mm-per-prompt {"image": 1}`，于是一个请求带 ≥2 张图
直接被 400：`At most 1 image(s) may be provided in one prompt`。
麻烦的是**图片留在对话历史里** ⇒ 之后每一轮都 400，**会话永久卡死**。
dsh/Codex 一个回合里并行调用两次 `read_image` 就会踩中。

### 3.2 结论：限制来自启动参数，不是模型能力

* 校验在 vLLM 的 Rust chat 层（`validate_mm_limits`）——**改它必须重启实例**；
* 模型侧原生支持多图：`vllm_ascend/models/deepseek_v41/vl_model.py` 的
  `_process_image_input` 是逐图循环，每张图有自己的 `vit_grid`/`llm_grid`/占位区间；
* 单张图 ≤1024 视觉 token，而 `BAT_TOKENS=8192`，文本侧一批放得下。

### 3.3 默认值与一致性约束

| 位置 | 默认 | 说明 |
|---|---|---|
| `scripts/serve_a2.sh` / `serve_v2.sh` | `MM_LIMIT_IMAGES=4` | 生成 `--limit-mm-per-prompt {"image": 4}` |
| `scripts/serve_a3_pd.sh` | `MM_LIMIT_IMAGES=4`，非空正整数校验 | **P/D 同值**：两边都校验同一份请求体，D 更小就会在 D 上 400 |
| `llm-api-tunnel` 客户端 | `--max-images` 默认 4 | 与部署一致；超限仍走 `keep-latest` 兜底 |

⚠️ 代价：`--max-num-seqs 4` 时最坏 4×4 张图同时在编，视觉侧要留 HBM 余量。
要收更多图就 P/D 同时调大，并把客户端 `--max-images` 跟上。

---

## 4. 验证

### 4.1 离线（可重复、秒级）

（真机结果见 4.2；先跑离线，再动真机。）

`python3 -m pytest tests/test_decode_guard.py -q` —— 8 个用例，两个方向都覆盖：

* 拦得住：decode + 无 KV 参数 → 400，且**断言下游没被调用**；
* 拦不歪：带 `kv_transfer_params` 的转发请求 → 放行，且 body 逐字节还在；
* 不误伤：`/health` 放行；非 JSON 放行；prefill 角色空转；`V41_DECODE_API_GUARD=0` 可关；
* `kv_transfer_params: {}` 视同缺失（协议上 P 一定带 `do_remote_prefill` 等字段）。

### 4.2 真机（起服后）

2026-09-27 00:2x 在 a3-21 实测（P=18990 / D=18991 / 代理=18992，
P/D 均为 `MM_LIMIT_IMAGES=4`、`HOST=127.0.0.1`，D 挂护栏）：

| # | 动作 | 结果 |
|---|---|---|
| 1 | 起服日志痕迹 | `[serve-v2] bind=127.0.0.1 mm_limit_images=4 decode_guard=on` + `--middleware v41_decode_guard.decode_guard` + `[V41-DECODE-GUARD] middleware loaded` |
| 2 | **直连 18991**，无 `kv_transfer_params`，prompt > 128 token（**事故形状**） | **HTTP 400**，`code=ced_decode_role_requires_kv_transfer_params` |
| 3 | 紧接着查 D 的 `/health` 与 `/v1/models` | **均 200**，`vllm serve` 进程仍在 —— 引擎没被打死（事故的负控） |
| 4 | D 侧日志 | `rejected /v1/chat/completions：没有 kv_transfer_params …；累计拦截 1 次`（恰为负控那一条） |
| 5 | 经 18992 纯文本 | 200，回答 `2` |
| 6 | 经 18992 **两张图** | 200，模型分别认出两张图 |
| 7 | 经 18992 **四张图**（上限） | 200，`prompt_tokens=2263`，逐张认全；第 4 张的 `ZEBRA-4821` 也读对 |
| 8 | 代理 `/v1/models` 与 `/healthcheck` | 200 / `{"status":"ok","prefill_instances":1,"decode_instances":1}`（护栏没挡住探活） |

第 3 条是这次加固的核心价值：**同一个请求形状，事故时打死实例，现在只得到 400**。

第 7 条同时回答了"一个请求内能不能放多张图"：能，而且不是"不报错"级别的能 ——
模型是真的逐张看了（第 2 张 `shot.png` 与第 3 张 `montage.png` 内容高度相似，
它仍分清了后者多出 `1/2`/`2/2` 标记与 `7391` 框）。这条路径 P、D 两边都过，
说明"P/D 必须同值"的约束在真实链路上成立。

---

## 5. 遗留

* 同类"worker 里 raise ⇒ EngineCore 退出"的路径不止这一条（任何 worker 侧
  未捕获异常都如此）。本护栏只覆盖**已知的、可由请求触发的**那一条。
* `serve_a2.sh` 的 `exec serve_v2.sh` 旁路（issue #2 里已承认）仍然存在：
  直接跑 `serve_v2.sh` 不会挂载护栏目录，此时 decode 角色只打 WARNING 不 die。
* 半边回环绑定是**默认值**，不是强制：`HOST=0.0.0.0` 仍可显式打开。
