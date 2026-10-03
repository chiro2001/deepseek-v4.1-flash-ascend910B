# 交接：动态 K 修复 + 融合算子（2026-10-03 深夜，**因内网不可达暂停**）

> 网络恢复后，**两条命令**就能接着跑。本文把"已证实 / 未证实 / 待执行"分开写清楚，
> 避免接手的人把"文档里的计划"当成"已经拿到的结果"。

---

## 0. 阻塞原因（外部，非代码）

| 视角 | 观测 | 结论 |
|---|---|---|
| 跳板机 `192.168.101.5` | 它到 `192.168.45.21` **100% 丢包** | 内网那段不通 |
| VPS `117.72.247.67`（在 10.8.0.1） | 路由表只有 `10.8.0.0/24` 与 `192.168.101.0/24`，**没有 192.168.45.0/24**；ping a3-21 100% 丢包 | 无法旁路 |
| 平板 `10.8.0.21`（Termux 8022 可达） | 它自己也 ping 不通 a3-21、22 端口 CLOSED | 内网路由缺失 |
| a3-21 → VPS 的反向隧道（`2223/2224`） | VPS 上**未监听** | 这条备用路也断 |

⇒ **a3-21 / a3-22 从外部完全不可达**；要恢复需要重连"到 192.168.45.0/24 的那条内网 VPN"。

**顺带修好的**：平板上的 `relay-34500` 有**陈旧 pidfile** 缺陷 —— pidfile 写的是 25201，
而真正持有 34500 的是 21784，于是 `status` 判"未运行" → `start` 必然撞
`Address already in use` → 自动修复**永远失败**（这与 `relay-health.log` 里那串
`Ncat: bind ... Address already in use` + `远程重启命令执行失败` 完全吻合）。
**建议修法**：`start()` 里在 `is_running` 为假时先 `pkill -f "ncat -lk .* 34500"` 再启动，
或改成"用 `ncat -z 127.0.0.1 34500` 探测成功即视为运行中"（README 已经写了这个探测法，
脚本却没用它）。⚠️ **未修，只在本文记录**（改的是别人设备的脚本）。

---

## 1. 已证实（有原始证据，可直接引用）

| # | 结论 | 证据 | commit |
|---|---|---|---|
| 1 | **"req≥2 掉到 ~10 tok/s" 的根因**：`patch_cudagraph.py` 写 `_v41_qlens_cache`，`model_runner_v1.py` 读 `_dynamic_decode_query_lens` ⇒ 恒 None ⇒ K=0 的步被判非 uniform ⇒ `dispatch()` **静默返回 NONE** ⇒ 整步 eager | 同实例 A/B 两次 profiler：K=7 时 allreduce 在图内 stream 225（80.5 次/步、avg **44 µs**）；K=0 时在图外 stream 38（89.8 次/步、avg **1962 µs**）。捕获键 19 个与推算逐字吻合 | `1f51cf6` |
| 2 | **修复后 N=2：9.9–12.0 → 73.9 tok/s（6–7×）**，且 `draft/gen=0.03` 证明 K=0 仍被正确选中 | dynprobe 首轮 + /metrics | 同上 |
| 3 | **独立 TP8 的静态包络**：SPEC=1 K=5 → 1/2/4/8/16 = 109.0 / 142.9 / 225.4 / 315.3 / 398.6；SPEC=0 → 51.6 / 92.6 / 166.2 / 277.4 / **434.1** | `results/bench/tp8dspark_*.json` | `5d001ea` |
| 4 | **动态 K 的 K=0 ≠ 静态 SPEC=0**：N=2 只有 73.9 vs 92.6（**0.80×**） | 同机同脚本 | `58dd7fd` |
| 5 | **第二个缺陷的根因（静态推导）**：`dsa_v1:1407-1408` 同时用 `[:num_reqs]` 截断 `seq_lens` 与 `block_table_tensor`，但两者自身行数不同（`num_reqs` vs 1）；唯一会补齐的 `SlidingWindowAdapter.apply()` 因 `draft_window_size` 未设而**根本没执行** | 代码 + `llm_base_proposer.py:788` 的条件 | `13b65f6` |
| 6 | Track B：`npu_hc_pre_v2`/`npu_hc_post` **源码可读**（`csrc/moe/hc_pre|hc_post`），单算子基准 `hc_pre≈61.6 µs`、`rms_norm≈14.0 µs`（86 次/步 ⇒ 融合上限 ≈1.2 ms/step） | 子代理 `trackB/REPORT-1.md` | — |

---

## 2. 未证实 / 未执行（**不要当成已完成**）

| 项 | 状态 |
|---|---|
| 候选修复 `V41_DYNSPEC_BT_PERSIST=1` 是否真能修掉崩溃 | **未验证**（默认关，只写了代码 + 推导） |
| `[DYNSPEC-DIAG]` 探针的实际输出 | **未采集**（探针已就位、语法自测通过） |
| 动态 K 的完整 1/2/4/8/16 曲线 | **未拿到**（只有 N=2 单个点，且第二轮崩） |
| 正确性：144K/1M 四针 + 并发 2 各带不同针 + `regress2.py` | **全部未跑**（脚本已就位并自测） |
| 目标线 N=1 ≥105、N=16 ≥430 | **未测**（静态双实例路线已达标，但那是 16 die 的方案） |
| Track B 的数值对齐与端到端收益 | **未完成**（子代理卡在可选输入参数表；已给它 `pre_mix` 参考） |

---

## 3. 网络恢复后：两条命令

```bash
cd /home/chiro/projects/dsv41/main-merge

# ① 第二轮取证（基线 + 探针）—— 判读三种互斥情形见文档 §4
bash fixes/20261003-dynspec/round2_runbook.sh

# ② 若 ① 显示 pre-window 就不一致（=候选修复对症），切 B 轮复测
R2_ROUND=B bash fixes/20261003-dynspec/round2_runbook.sh

# ③ 崩溃修好后，跑完整验收（四针 144K/1M + 并发2不同针 + regress2 + 性能曲线）
bash fixes/20261003-dynspec/accept_dynspec.sh
```

三条脚本都已自测：
- `round2_runbook.sh`：`bash -n` 通过
- `probe_concurrent_needles.py`：mock 正控 PASS / 两个负控 FAIL / 不可达 exit=2
- `accept_dynspec.sh`：VPN 断开时干净 `exit=2`（不误报判据失败）

---

## 4. 备用路线（目标里明确允许的那条）

见 [`STATIC-DUAL-INSTANCE-FALLBACK-20261003.md`](STATIC-DUAL-INSTANCE-FALLBACK-20261003.md)：
每档取两实例较优者 = **109.5 / 142.9 / 225.4 / 315.3 / 434.1**，**数字上已达标**。
**硬前提：要 16 个 die**（单实例 TP8 占 8 个），而 a3-21 是共用机。

---

## 5. 本地可继续做（不需要 a3-21）

1. Track B 的参数表/结构体布局分析（本地有 `csrc/moe/hc_pre/` 全量源码 + `trackB/src/` 各版本）；
2. `SlidingWindowAdapter` 的广播隐患（`spec_decode/utils.py`，目前**不在挂载清单**里）——
   若要打它必须先给 `serve_a2.sh` 加一项挂载；
3. 把"a3-21 起服必须带 `VLLM_ENGINE_READY_TIMEOUT_S`"这条**并入 main**
   （补丁在 `fixes/20261003-dynspec/serve_a2.dynspec.patch`）。
