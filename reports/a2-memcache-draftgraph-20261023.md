# A2 单机 A2：MemCache 外部池 + DSpark Draft Graph 实测（2026-10-23）

> 环境：8×910B1（A2），镜像 `dsv41-a2:v8`（vllm-ascend `0.1.dev5097+ge43cf1e9f`），
> 模型 DeepSeek-V4.1-Flash W4A8 + Engram(int8) + Vision，`TP8 EP`，端口 8100，`served_model_name=deepseek-v41`。
> 口径：9012 prompt token / 400 输出 token，`temperature=0`、`ignore_eos`。

## 1. 结论先行

| 形态 | ms/step | 单流 tok/s | 相对最早生产形态 |
|---|---:|---:|---:|
| eager draft + MemCache 池 | 100.8 | ~30 | 1.0× |
| eager draft，无池 | 68.1 | ~46 | 1.5× |
| **Draft Graph + MemCache 池** | **42.0** | **~69** | **2.3×** |
| Draft Graph，无池 | 35.4 | **76.3**（实测） | 2.5× |

两条可直接用的结论：

1. **Draft 入图是 A2 上最大的单项杠杆**：`68.1 → 35.4 ms/step`（−48%），与
   `reports/a2-draft-graph-20260920.md` 的 `34.3 ms/step`、88.7 tok/s 同一量级。
2. **MemCache 的代价与 draft 派发方式强耦合**：eager 下 +33 ms/step，draft 入图后只有 **+6.6 ms/step**。
   所以"投机 + 池"必须一起上，否则池会把投机的收益吃掉。

## 2. step 时间拆分（SPEC=0 A/B，同配置只关投机）

| 口径 | ms/step | 说明 |
|---|---:|---|
| `SPEC=0`（只做 target verify） | 30.0–33.6（中位 30.3，8 rank 离散 <0.15） | 可与 `SPEC=1` 对比 |
| `SPEC=1` eager draft | 66–68 | ⇒ **draft ≈ 37 ms** |
| `SPEC=1` draft 入图 | 35.4–36.2 | ⇒ **draft 降到 ~5 ms** |

`SPEC=0` 的 `[bneck]` 明细（每 rank 一致）：`d2h≈0.18 hash≈0.12 route≈1.45–2.9 pad≈0.24 total≈3.2–4.4`；
`[route-probe]`：`a2a 0.34 / bcast 0.20 / evt 0.18 / h2d 0.10 / lookup 0.37–0.45 / plan 0.09–0.13 / scatter 0.06–0.08`
⇒ **Engram host 路径健康**，不是瓶颈。

## 3. Draft Graph 怎么开的（不需要重做镜像）

镜像里**已经带**同代 draft 版文件：`/opt/dsv41/patches/draft/{dsa_v1,dspark_proposer,llm_base_proposer}.py`
（本镜像 `dsa_v1.py` md5 = `371bb023e2ecb97c01a484fb9b8cef05`）。步骤：

   `vllm_ascend/{attention/dsa_v1.py, spec_decode/dspark_proposer.py, spec_decode/llm_base_proposer.py}`，
   并保留 `.a2orig` 备份；
2. `source tools/dsenv_draftgraph.sh`（本仓新增，34 条 DSPARK 默认值取自 `serve_a2.sh`），
   并把 spec 的 `enforce_eager` 改为 `false`（`serve_v2.sh` 里就是 `SPEC_EAGER=0`）；
3. 起服。验收：日志 8/8 rank 打印 `Wrapping draft model with ACLGraphWrapper`，
   且 `Mean acceptance length` 落在 **2.4–3.6**（不是 1.0）。

## 4. MemCache 跨机共享池（接入方案见 `deploy/memcache/`）

* 结构：1 个 MetaService（主节点）+ N 台客户端，每台 8 个 rank 各注册一个 DRAM segment
* `dram.size=48GB/rank`，2 机 ⇒ **16 seg / 768 GiB**（`.21` 实测 `memcache_total_capacity_bytes{medium="dram"}=824633720832`）
* 跨机命中实测（`.21` 写、`.25` 读，24K token 前缀）：
  `external_prefix_cache_hits_total +52,736`、`queries +54,184` ⇒ **99.7%**，首 token `9.1 s → 0.8 s`
* **单机池对吞吐只有开销、只有容量收益**：8K 输入 / 400 输出这种场景用不到 DRAM 扩容，
  所以 `.15` 上"要不要池"就是一个纯取舍（−6.6 ms/step 换 384 GiB 容量 / 跨机共享能力）

## 5. 没做/待验证

1. **质量闸门**：A 只看了中位区间（2.4–3.6）与文本连贯性，没跑 `tests/multibatch` 全套、
   没跑 GSM8K、没做 eager vs 图 的逐位一致性。
2. **PGO 没产物**：本机 `optim/pgo` 缺 `libpython3.12.so.1.0`，`PYTHON_PGO` 只能降级为 0；
   A2 是 PGO 收益最大平台，值得补。
3. **`DRAFT_GRAPH=1` 仍非默认**：`serve_a2.sh` 的默认是 0（历史 A≈1.06 负例）。
   本次结果说明**镜像里那套 draft 文件（含 `DSPARK_CAPTURE_VALUE_FIX`）是能用的**，
   建议把它从"未验证"升级为"A2 已验证配置"，但仍需先补第 1 条。
4. **跨机池 + 图 的联合 baseline** 只测了 `.21`/`.25`（~65 tok/s），
   `.15`（自带池）当时波动到 34–51 tok/s，未定性（疑似池写入累积 / 采样窗口并发）。
