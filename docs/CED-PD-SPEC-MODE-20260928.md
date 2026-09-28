# A3 CED-PD：D 侧推测解码的**三档可选**（`SPEC_MODE`）

**日期**：2026-09-28　**形态**：A3 单机 8+8 PD 分离 + CED　**标记**：【实测】/【推断】/【未确认】

---

## 0. 一句话

D 侧的 DSpark 推测解码现在是一个**单一开关的三个档位**，互斥可选、矛盾组合
**起服前 fail-closed**；P 侧**恒为关**（架构性，不是配置问题）。

```bash
SPEC_MODE=on       # ★ 默认：全开 SPEC，固定 K=SP_TOKENS（= 现行交付口径）
SPEC_MODE=off      # 全关 SPEC（纯自回归，连 --speculative-config 都不加）
SPEC_MODE=dynamic  # 动态 K：按**当时请求数**切 1 ↔ K=0
```

---

## 1. 三档各自具体做了什么

| 档位 | `SPEC` | `DRAFT_GRAPH` | `--speculative-config` | 动态 K | 上游降级门豁免 |
|---|---:|---:|---|---|---|
| `on` | 1 | 1 | 固定 `num_speculative_tokens=SP_TOKENS` | 否 | 不需要（不含 `SP_SCHEDULE`） |
| `off` | 0 | 0 | **完全不追加** | 否 | 不需要 |
| `dynamic` | 1 | 1 | `num_speculative_tokens=SP_TOKENS` **+** `num_speculative_tokens_per_batch_size` | 是 | ★ **自动设** `V41_CED_DYNAMIC_SPEC_FULL_GRAPHS=1` |

`dynamic` 的默认表 = `1,1,7;2,8,0`：

| 本步请求数 | K | 依据 |
|---:|---:|---|
| 1 | 7 | 低并发走推测，单流吞吐 1.77×【实测】 |
| ≥2 | 0 | 纯自回归；实测并发 2 时自回归已 1.23× 领先【实测】 |

自定义阈值：`SP_SCHEDULE='1,1,7;2,4,0'`（分号分隔的三元组 `起,止,K`，闭区间）。

---

## 2. 三种用法

### 2.1 用 deploy 启动器（推荐）

```bash
# D 侧：三选一
SPEC_MODE=off     bash deploy/a3-ced-pd/launch/serve_d.sh
SPEC_MODE=on      bash deploy/a3-ced-pd/launch/serve_d.sh
SPEC_MODE=dynamic bash deploy/a3-ced-pd/launch/serve_d.sh

# P 侧不用管：SPEC_MODE 给了 on/dynamic 会被**拒绝**（P 恒 off）
MODEL=<模型目录> bash deploy/a3-ced-pd/launch/serve_p.sh
```

### 2.2 直接调角色脚本

```bash
SPEC_MODE=dynamic bash scripts/serve_a3_ced_pd.sh decode
bash scripts/serve_a3_ced_pd.sh prefill      # P 不需要给 SPEC_MODE
```

### 2.3 只看解析结果（不起容器、不占卡）

`scripts/serve_a3_ced_pd.sh` 认一个自检钩子（生产路径不会设）：

```bash
V41_SPEC_MODE_CHECK_ONLY=1 SPEC_MODE=dynamic bash scripts/serve_a3_ced_pd.sh decode
# SPEC_MODE_RESOLVED mode=dynamic role=decode spec=1 draft=1 dyn=1 full_graphs=1 schedule=1,1,7;2,8,0
```

钩子取 `2` 时 launcher 会**继续穿透**到角色脚本，验的是"用户敲的那条命令最终解析出什么"
（`deploy/a3-ced-pd/launch/serve_d.sh` 与 `scripts/serve_a3_ced_pd.sh` 各有一个默认值，
最容易在这里分叉）：

```bash
V41_SPEC_MODE_CHECK_ONLY=2 SPEC_MODE=off MODEL=<模型目录> \
  bash deploy/a3-ced-pd/launch/serve_d.sh
# SPEC_MODE_RESOLVED mode=off role=decode spec=0 draft=0 dyn=0 full_graphs= schedule=
```

**为什么给这个钩子**：三档之间只差几个 env，而"我选了 A、生效的是 B"是本仓
反复栽的一类事故（默认值两处不一致 / 转发漏一层 / 补丁没挂上）。钩子让档位解析
可以被**离线**验证，不必起两个实例等 20 分钟。

---

## 3. 向后兼容（旧写法仍然认）

以下写法**只在没给 `SPEC_MODE` 时**参与推断，语义与 2026-09-27 的行为逐字节相同：

| 旧写法 | 解析结果 |
|---|---|
| `V41_CED_ALLOW_DSPARK=0` | `off` |
| `V41_CED_DYNAMIC_SPEC=1` | `dynamic`（并**自动补** `V41_CED_DYNAMIC_SPEC_FULL_GRAPHS=1`） |
| `SPEC=0` | `off` |
| `SPEC=1` | `on` |
| 什么都不给（D） | `on`（= 交付默认） |
| 什么都不给（P） | `off` |

---

## 4. fail-closed 清单（这些组合**起服前**就被拒）

判据是**内容**不是"我传了这个变量"：拒绝时必须带 `[a3-ced][FAIL]` 与原因。

| 组合 | 为什么拒 |
|---|---|
| `SPEC_MODE=` 非法值（如 `bogus`） | 拼错就是没生效，静默跑默认最危险 |
| `SPEC_MODE=off` + `SPEC=1` / `DRAFT_GRAPH=1` | 两处设置矛盾，静默择一会让你以为生效的是另一个 |
| `SPEC_MODE=on` + `SPEC=0` / `DRAFT_GRAPH=0` | 同上 |
| `SPEC_MODE=on` + `V41_CED_DYNAMIC_SPEC=1` | 同上（且会真的拉起动态路径） |
| `SPEC_MODE=off` + `V41_CED_DYNAMIC_SPEC=1` | 同上 |
| `V41_CED_ALLOW_DSPARK=0` + `V41_CED_DYNAMIC_SPEC=1` | 两个旧开关互斥 |
| `V41_CED_DYNAMIC_SPEC=1` + `SPEC=0` | 动态档要求 `SPEC=1`，否则你的 `SPEC=0` 会被静默改成 1 |
| **非** dynamic 档却带 `SP_SCHEDULE` | 只要 `SP_SCHEDULE` 非空，`serve_a2.sh` 就会拉起动态 K 整条路径 |
| `SPEC_MODE=dynamic` + `CED_DIAGNOSTIC_EAGER=1` | eager 下没有图可切 |
| `SPEC_MODE=dynamic` + `V41_CED_DYNAMIC_SPEC_FULL_GRAPHS=0` | 该档必须豁免上游降级门，否则模型构造期失败 |
| P（prefill）+ `SPEC_MODE=on`/`dynamic` | 架构性不可行 |
| P + 显式 `SPEC=1` 或 `DRAFT_GRAPH=1` | 保留 2026-09-27 的旧严格性（**不静默降级**） |
| `SPEC=2` / `DRAFT_GRAPH=2` | 取值非法，会被后续 `export` 静默覆盖成合法值 |

★ 交付面一致性：`deploy/a3-ced-pd/launch/serve_d.sh` 里**只在没给 `SPEC_MODE`
时**才默认 `SPEC=1 DRAFT_GRAPH=1` —— 否则一句 `export SPEC=${SPEC:-1}` 就能把
`SPEC_MODE=off` 顶掉。这条由自检的第 4 组用例守住。

---

## 5. 判据与验收

| 项 | 结果 |
|---|---|
| 离线自检 | `bash tools/selftest_spec_mode.sh` → **41 项**：三档正控 + 15 条矛盾 fail-closed + legacy 兼容 4 条 + 交付面 5 条 + P 侧 7 条 + **全链路 4 条**（launcher → 角色脚本两层默认值交互） |
| 负控 | 判据指向**改动前**的逐字节夹具 `tools/fixtures/serve_a3_ced_pd_before_spec_mode.sh`（md5 `9c7a97ba…`）⇒ **25 条 FAIL**（改动前连 `SPEC_MODE` 都不认） |
| 接进主流程 | `tools/selfcheck_pkg.sh` 第 **9n** 节（含负控），任何回归都会在起服前拦住 |
| 旧自检未回归 | `tools/selftest_ced_defaults.sh` **12/12**（这轮改写的第一个版本漏了 `DRAFT_GRAPH` 取值门，正是被它的负控当场抓住） |

---

## 6. 性能预期（引用已有实测，**不是**本档新测）

> ⚠️ 下面几个数字来自**不同配置**，只能当量级参考，不能横向相减。

| 口径 | 并发 | ms/step | A | decode tok/s | 出处 |
|---|---:|---:|---:|---:|---|
| 静态 `SPEC=0` | 1 | 24.35 | n/a | 41.06 | `docs/CED-PD-DYNAMIC-SPEC-20260926.md` §11.4 |
| 静态 `SPEC=1 DRAFT_GRAPH=1` | 1 | 32.68 | 3.10 | 94.79 | 同上（09-27） |
| 动态 K=7 | 1 | 30.15 | 2.62 | 87.07 | 同上 |
| 动态 K=0 | 2 | 32.2 / 32.5（两流各自） | n/a | **未记录** | 同上 |

**口径提醒**：`ms/step` 与 `A` 必须**成对**看。K=7 的 30.15 ms/step → 87 tok/s；
`SPEC=0` 的 24.35 ms/step → 41 tok/s（A=1，每步只出 1 个 token）
⇒ **按"每 token 时延"看，K=7 反而更快（11.5 vs 24.4 ms/token）**。

---

## 7. 诚实边界（**上线前必读**）

1. **动态档的交叉点是在 2K prompt 上测的**；长上下文负载下每步固定开销大得多，
   交叉点**可能移动** ⇒ 要在自己的负载上复测。
2. **动态档的并发 ≥2 没有吞吐基线**：§11.4 只记了 ms/step（并发 2 两流
   32.2/32.5），`decode tok/s` 那一列是空的。文档当时自己注明"另跑的聚合 tok/s
   把 prefill 也算进 wall，不与上表可比，故不列"。⇒ 用它做容量规划前先补测。
3. **动态档豁免了上游的一道保护**（关掉 MRV1 对 dynamic SD 的 PIECEWISE 降级）。
   主要失效模式是**静默算错**，所以该档必须用 144K/1M 正确性探针验收，
   不能只看"起来了 + ms/step 正常"。
4. **三档都没有在"含 2026-09-28 池修复"的这版上重跑完整验收矩阵**
   （144K/1M 四针 + 并发 2 各带不同针）。
5. **`off` 档在 CED-PD 形态下的端到端性能未单变量测过**：上表那个 24.35 ms/step
   的 `SPEC=0` 是**另一个配置**下测的。

---

## 8. 相关

| 内容 | 位置 |
|---|---|
| 动态 K 的设计/上游门/两处缺陷全记录 | `docs/CED-PD-DYNAMIC-SPEC-20260926.md` |
| DSpark 的 decode 时延拆解（并发 4 A/B profiler） | `docs/CED-PD-DSPARK-LATENCY-BREAKDOWN-20260926.md` |
| DSpark × CED 架构关系 | `docs/CED-PD-DSPARK-CED-RELATION-20260926.md` |
| 镜像包交付说明 | `deploy/a3-ced-pd/KIT-README.md` |
