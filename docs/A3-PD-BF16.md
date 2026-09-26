# A3 单机双 TP8 PD 基线（BF16 KV）

本文记录 DeepSeek V4.1 Flash 在一台 A3 上的已实测基础 PD 形态，以及发布包中的可复用启动入口。
这里的 P、D 都是完整模型的独立 TP8 实例：P 仍执行全模型层，不是只执行前 20 层的论文 CED 形态。

## 已实测配置

2026-09-23 在 A3-21 上使用完整 DeepSeek V4.1 Flash W4A8 权重完成 1P1D 验收：

| 项目 | Prefill（P） | Decode（D） |
| --- | --- | --- |
| 芯片 | 0–7 | 8–15 |
| TP / DP | 8 / 1 | 8 / 1 |
| HTTP 端口 | 18550 | 18551 |
| MooncakeHybrid 端口 | 18650 | 18651 |
| KV 角色 | `kv_producer` | `kv_consumer` |
| KV dtype | `bfloat16` | `bfloat16` |

两侧的共同参数是 `ENGRAM=1 ENGRAM_DEVICE_INDEX=0 CPU_BIND=0 STATIC_KERNEL=1`、
`MAX_LEN=147456 MAX_SEQS=4 BAT_TOKENS=8192 GPU_UTIL=0.92`、
`PREFIX=0 SPEC=1 SP_TOKENS=7 DRAFT_GRAPH=0`。因此这轮使用的是 DSpark eager，
没有启用 draft 入图；Engram 走 host 路径。

验收结果：

- P、D 和代理均健康，短请求、流式请求与工具调用均返回 HTTP 200。
- 8,335 token 检索 4/4 正确；实际 142,426 token 检索 4/4 正确，乱码指纹为 0。
- 实际 143,735 token 的长流、两路并发短流、短/长工具请求共 5/5 通过；长工具请求的工具调用参数正确。
- D 侧 `external_kv_transfer` 累计达到 900,526 token，证明 P→D 的 KV 交接链路工作。

DeepSeek 的 MooncakeHybridConnector 会在 P 侧截去 prompt 的最后一个 token，
由 D 侧装载前 `N-1` 个 token 的 KV 并重算末 token。因此验收请求必须经 18552 代理发送，
不能把 P/D 端口当作普通的同构服务分别直接测。

这些数字是功能验收结果，不是吞吐基准。`external_kv_transfer` 只表示 P→D 传输，
不能用来证明 DRAM KV 池命中。

## 启动

发布包新增了两个入口。先确认 16 张卡未被占用，并使用同一份完整模型目录。

### ★ 半边不可直连（2026-09-27 起默认只监听回环）

P/D 是**内部半成品**：P 只产 KV、D 只消费 KV，谁都不是能独立服务的实例。
2026-09-27 00:01 有一条普通请求被直接发到 D 的端口，D 只能自己去 prefill，
撞上 CED 固定 128-token replay 的守卫，worker 里 `raise` 让 EngineCore 退出
——**整个 decode 实例死掉，恢复要重载 20 分钟的权重**。

因此：

* `serve_a3_pd.sh` 默认 `HOST=127.0.0.1`（两个半边都只监听本机）；要暴露得显式
  设 `HOST=0.0.0.0`，并自行处理来源限制或鉴权；
* decode 角色另外挂一层**请求边界护栏**：没有 `kv_transfer_params` 的生成请求
  直接 400，不进引擎。细节、判据与负控见
  [CED-DECODE-API-GUARD-20260927.md](CED-DECODE-API-GUARD-20260927.md)；
* 客户端一律连代理（`serve_a3_pd_proxy.sh` 起的 18992 一类），不要连半边。

### 多图上限（`MM_LIMIT_IMAGES`，默认 4）

P/D 两侧都是 `MM_LIMIT_IMAGES=4`（即 `--limit-mm-per-prompt {"image": 4}`）。
**必须同值**：两边都校验同一份请求体，D 更小的话请求会在 D 上被 400。
历史上这个值是 1，导致"一个回合读两张图 ⇒ 图片留在历史里 ⇒ 会话永久 400"。
该参数被编译进请求校验，**改它必须重启实例**。

在 P 终端执行：

```bash
cd ~/projects/dsv41-release
MODEL=/path/to/deepseek-v41-w4a8 \
  bash scripts/serve_a3_pd.sh prefill
```

在 D 终端执行：

```bash
cd ~/projects/dsv41-release
MODEL=/path/to/deepseek-v41-w4a8 \
  bash scripts/serve_a3_pd.sh decode
```

默认使用 P=0–7、D=8–15。需要换卡时分别覆盖 `DEVS`，例如：

```bash
PD_PREFILL_DEVS="8 9 10 11 12 13 14 15" MODEL=/path/to/model \
  bash scripts/serve_a3_pd.sh prefill
PD_DECODE_DEVS="0 1 2 3 4 5 6 7" MODEL=/path/to/model \
  bash scripts/serve_a3_pd.sh decode
```

脚本默认使用带时间戳的 `dsv41-pd-v41-prefill-*` / `dsv41-pd-v41-decode-*` 容器名，
不会占用 A3 普通服务的 `dsv41-a3` 名称；检测到同名旧容器时会先报错，不会替你删除。
需要等待健康检查完成时可加 `WAIT_READY=1`。

启动前可只做参数和透传检查：

```bash
DRY_RUN=1 MODEL=/path/to/model bash scripts/serve_a3_pd.sh prefill
DRY_RUN=1 MODEL=/path/to/model bash scripts/serve_a3_pd.sh decode
```

`scripts/serve_a3_pd.sh` 生成紧凑的 `MooncakeHybridConnector` JSON，
由 `scripts/serve_a2.sh` 写入容器 `inner.sh`，再由 `scripts/serve_v2.sh` 加入实际的
`vllm serve` 命令。不要在 `KV_ARGS_EXTRA` 的 JSON 中加入空格。

## 启动本地代理

P、D 健康后，在第三个终端执行：

```bash
cd ~/projects/dsv41-release
bash scripts/serve_a3_pd_proxy.sh
```

代理监听 `127.0.0.1:18552`，并把请求转给 18550/18551。代理也支持参数覆盖：

```bash
PROXY_PORT=18652 PREFILL_PORT=18550 DECODE_PORT=18551 \
  bash scripts/serve_a3_pd_proxy.sh
```

如果同时覆盖了 P/D 的 HTTP 端口，还要把代理的 `PREFILL_PORT` / `DECODE_PORT`
同步为新的 HTTP 端口；`KV_PORT` 是 P/D 内部 Mooncake 端口，不能填到代理参数里。

启动前只打印 Docker 命令：

```bash
DRY_RUN=1 bash scripts/serve_a3_pd_proxy.sh
```

简单检查：

```bash
curl -fsS http://127.0.0.1:18550/health
curl -fsS http://127.0.0.1:18551/health
curl -fsS http://127.0.0.1:18552/healthcheck
```

## 当前边界

以下内容不属于这份 BF16 基线的已验证结论：

- AscendStoreConnector 或其他 DRAM KV 池的取回命中；
- KV8 档（包括 SWA、ring FP16、graph safe）；
- `DRAFT_GRAPH=1` 的 draft 入图；
- 1M 上下文；
- DP2，或把 P/D 缩成 4 chip + 4 chip；
- P 只执行前 20 层的分层模型实现。

后续实验应以这份基线为单变量起点，并单独记录连接器、KV dtype、上下文长度和 draft 形态。
