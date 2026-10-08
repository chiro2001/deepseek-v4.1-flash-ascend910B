# DeepSeek-V4.1-Flash / A3 单机 TP8 —— **解码路径融合优化包**

> 这是一个**优化叠加包**，不是一个部署形态。它叠加在任意 A3 TP8 单实例
> （或 `deploy/a3-ced-pd` 的 PD 分离形态）之上，把 decode 路径上的两处
> 「归一化 + 量化」合成一次算子下发。
> 所有数字为【实测】。

---

## 0. 一句话

| | 优化前 | **本包** | 收益 |
|---|---:|---:|---:|
| **decode ms/step**（`bneck hp p50`，4 并发×700 token） | 24.668 ms | **24.445 ms** | **−0.90%** |
| 配套的融合点 A（上游已有，本包一并开启） | — | — | −0.62% |
| 单流吞吐 | 132.3 tok/s | 132.7 tok/s | 噪声内 |
| **短问答精度** | — | **3/3 正常，无乱码** | — |

机制：把 decoder layer 里的

```
hidden = input_layernorm(x)                 # 一遍 RMSNorm
q_quant, q_scale = wq_a.quantize(hidden)    # 一遍动态量化
```

合并成**一次** `npu_rms_norm_dynamic_quant_bf16(x, w_ln)`，同时产出
bf16（供 compressor/indexer 用）与 int8+scale（供 `wq_a` 用）。覆盖 **43 层**。

---

## 1. 交付面（**两种形态，逐字节等价**）

| | ① 本仓 + `install.sh` | ② 发布 tar 包 |
|---|---|---|
| 怎么用 | `build_payload.sh --from-container <ct>` → `install.sh <ct>` | `tar -I zstd -xf <包>` → `install.sh <ct>` |
| 我们的文件从哪来 | 容器里已装好的产物 | 包内的 `opp/ so/ py/` |
| 需要网络/仓库 | 需要仓库 | **不需要**（一个 tar 即可） |
| 适用 | 开发、迭代 | 交付、跨机复现、无网环境 |
| 一致性判据 | `verify_consistency.sh` **逐文件 sha256** | 同左（包内 `PAYLOAD.sha256`） |

> 两面装的是**同一批文件**，清单见 [`PAYLOAD.md`](PAYLOAD.md)（唯一事实源）。
> `install.sh` 装完会自己跑一遍逐文件校验；`verify_consistency.sh` 可随时复验。
> **这是「可复现」的判据，不是口号。**

---

## 2. 交付内容一览

| 类别 | 内容 | 大小 |
|---|---|---:|
| **二进制** | 自研算子 `RmsNormDynamicQuantBf16` 的 OPP 安装树（kernel + aclnn + tiling） | 2.7 MB |
| **二进制** | 重编的 torch 扩展（多注册 1 个 op） | 984 KB |
| **Python** | `models/deepseek_v4/model.py`（融合点 B 的产出侧） | — |
| **Python** | `attention/dsa_v41.py`（融合点 B 的消费侧） | — |
| **开关** | 3 个环境变量（1 个本包新增），见 [`SWITCHES.md`](SWITCHES.md) | — |

基础镜像的 `opp/vendors/` 是**空的** ⇒ 本包对官方算子**纯增量**，不覆盖任何官方文件。

---

## 3. 安装

### 3.1 前置

* 一个已在跑的 vllm-ascend 容器（A3 单机、TP8）
* 容器内 `/vllm-workspace/vllm-ascend/` 是该镜像的原始安装树（本包会替换其中 2 个 `.py` 和 1 个 `.so`）

### 3.2 安装（形态 ① / ② 命令相同）

```bash
# ① 组装 payload（只有从仓库装才需要；用发布包时已带好）
bash build_payload.sh --from-container <ct>       # 或 --artifacts <tar.tgz>

# ② 先看一眼要做什么
bash install.sh <ct> --dry-run

# ③ 真装（幂等；重复跑会把文件覆盖回包内版本）
bash install.sh <ct>

# ④ 带开关重启服务：起服环境里 source 这一份
source <本包>/launch/with-fusion.env
```

`install.sh` 会：备份现有 `.so`/`.py`（`.orig-*` / `.bak-fusion-*`）、拷入包内版本、
清 `__pycache__`、然后**逐文件 sha256 自校验**。**它不重启服务、不改启动脚本。**

### 3.3 验收（三步）

```bash
# ① 服务活着
curl -s http://127.0.0.1:<port>/health                     # 期望 200

# ② 开关确实传给了 vllm 进程（3 行都要出现）
tr '\0' '\n' < /proc/$(pgrep -f 'vllm serve' | head -1)/environ \
  | grep -E 'V41_LNORM_FUSE|V41_QNORM_FUSE|ASCEND_CUSTOM_OPP_PATH'

# ③ 精度冒烟（3 条中英短问答，看有无乱码）
python3 checks/smoke_chat.py --port <port>
```

**⚠️ 性能验收必须看数字。** 融合失效是**静默**的（输出正常、日志无报错、只是没收益）——
本包开发过程中因此白测过两轮。判据是 `bneck hp p50`：
期望 **≤24.5 ms**（关闭时约 24.67 ms）。

### 3.4 回滚

```bash
bash install.sh <ct> --rollback     # 换回备份的 .so / .py
# 然后重启，并去掉 SWITCHES.md 里的 3 个变量
```

---

## 4. 开关（完整语义见 [`SWITCHES.md`](SWITCHES.md)）

| 变量 | 本包取值 | 默认 | 作用 | 不设的后果 |
|---|---|---|---|---|
| `ASCEND_CUSTOM_OPP_PATH` | `/vllm-workspace/3out_opp/vendors/custom_transformer` | 无 | 让 CANN 找到自研算子 | 融合点 B 报错或回退 |
| `V41_LNORM_FUSE` ★新增 | `1` | **关** | 融合点 B（43 层） | 走原路径，**无收益、不报错** |
| `V41_QNORM_FUSE` | `1` | 关 | 融合点 A（35 层，用官方算子） | 走原路径，无收益 |

三者都**只在进程启动时读一次** ⇒ 改完必须重启。

---

## 5. 已知边界与待办（诚实清单）

| 项 | 状态 |
|---|---|
| 算子正确性 | ✅ 生产形状 `D∈{1280,5120}` × 各 20 次：bf16/int8/scale **全部零错**；`[6,1280]` bf16 与 `npu_rms_norm` **逐位一致** |
| `D ≤ 512` | ⚠️ **上游 `rms_norm_dynamic_quant` 本身有缺陷**（原算子同条件 20/20 错），非本包引入；本包用 `shape[-1] >= 513` 保护，不会走到 |
| 三点同测（A+B 同时开） | ⏳ **未做**。A 的 −0.62% 与 B 的 −0.90% 是**分别**实测的；「可加」是基于两处是相邻但不同的代码路径，未经三点同测证实 |
| 交叉复测 | ⏳ 建议再做一轮 A/B 交叉（关掉 `V41_LNORM_FUSE` 重测），确认 −0.90% 可复现 |
| 长上下文 / 高并发（8、16） | ⏳ 只在 1 并发与 4 并发下测过 |
| `.so` 的字节级可复现 | ⚠️ 未做 `-ffile-prefix-map` 归一 ⇒ 不同构建目录下 sha256 可能不同；**以逐文件比对为准**，见 [`REPRODUCE.md`](REPRODUCE.md) §7 |
| OPP 树含上游 `rms_norm_dynamic_quant` 的重编副本 | ⚠️ 构建副产物，行为与上游一致，但使本包对该算子也有「最后一手」，记录在案 |

---

## 6. 目录内容

| 文件 | 作用 |
|---|---|
| [`PAYLOAD.md`](PAYLOAD.md) | **payload 清单（唯一事实源）**：文件→容器目标 的完整映射 |
| [`SWITCHES.md`](SWITCHES.md) | 3 个开关的语义、默认值、**失效模式**、回滚矩阵 |
| [`REPRODUCE.md`](REPRODUCE.md) | 从源码重建二进制的完整步骤 + 已知不可复现项 |
| `build_payload.sh` | 组装 payload 树（`--from-container` / `--artifacts`） |
| `install.sh` | 装进容器 / `--dry-run` / `--rollback` |
| `verify_consistency.sh` | **★ 逐文件 sha256 比对**（两个交付面的一致性判据） |
| `package_release.sh` | 打成可复现 tar.zst（`build` / `verify` / `selftest`） |
| `checks/smoke_chat.py` | 3 条中英短问答冒烟 |
| `launch/with-fusion.env` | 起服环境片段（3 个 export，可安全重复 source） |
| `artifacts/fusion-artifacts.tgz` | 二进制产物（1.25 MB，供 `--artifacts` 用） |
| `.build/` | 构建辅助脚本（`patch_build_aclnn.py`、`add_binding.py`） |

---

## 7. 相关文档

| 文档 | 主题 |
|---|---|
| `experimental/fusion-3out/` | 算子源码（13 文件）+ 对照的原算子 |
| `experimental/fusion-3out/integration/README.md` | 融合点 B 的 A/B 实测与**两个静默失效坑** |
| `docs/MULTI-OUT-OP-IMPLEMENTATION-20261008.md` | 算子实现记录 + §8 宏陷阱 + §9 正确性 |
| `docs/QNORM-FUSE-A-B-20261008.md` | 融合点 A 的 A/B（−0.62%） |

---

## 8. 打包自验（本包交付前已跑过，可复跑）

| 检查 | 命令 | 结果 |
|---|---|---|
| payload 组装 | `build_payload.sh --from-container <ct>` | 91 文件 / 3.8 MB，含 2 个算子目录 |
| **两种来源等价** | 分别用 `--from-container` 与 `--artifacts` 构建，比聚合指纹 | ✅ **同为 `098c15d53cbc`** |
| 逐文件一致性 | `verify_consistency.sh <ct>` | ✅ 87 项一致 / 0 不一致 |
| **打包可复现** | `package_release.sh build` 连跑两次 | ✅ **sha256 完全相同** |
| 篡改检出（负控） | `package_release.sh selftest` | ✅ 正常包可验、篡改包被抓 |
| 安装自校验 | `install.sh <ct>` | ✅ 装完逐文件 sha256 比对通过 |
| **回滚正确性** | `install.sh <ct> --rollback` 后跑 `verify_consistency.sh` | ✅ 检出 2 项不一致（证明回滚真的换了文件、校验器也不是摆设） |
| mount 形态 | `verify_consistency.sh --mode mount` | ✅ 2 个 `.py` 与仓库挂载源一致 |

> 这些是"可复现"的**证据**，不是承诺。改任何脚本后请重跑 `package_release.sh selftest`。

### 8.1 踩到并已修的三个坑（都会让发布包静默出错）

1. **`docker cp` 会静默漏文件** —— 在带符号链接/紧权限的目录上，`docker cp <ct>:/dir/. <dst>`
   报 `evalSymlinksInScope: ... is not in ...`，或只报一个 `permission denied` 就少拷几十个文件。
   实测：一个 84 文件的 OPP 树被取成 55 个，**而顶层目录一个不少**，肉眼看不出来。
   ⇒ 取/放目录一律走**容器内 tar 管道**；并且 `build_payload.sh` 加了
   「逐算子查 kernel `.o` + aclnn 头 + tiling 三件套 + 总数 ≥80」的完整性校验。
2. **`PAYLOAD.sha256` 不能把自己收进清单** —— shell 重定向会**先创建空文件**，
   `find` 于是把它也列进去，存的是"空文件的哈希"，之后永远校验失败。
   ⇒ `find ... ! -name 'PAYLOAD.sha256'`，并在生成后立刻 `sha256sum -c` 自检。
3. **dry-run 里不能用 `$(...)` 包住会打印内容的探测命令** —— 输出会被当成结果值。
   ⇒ dry-run 分支单独走一条只打印、不取值的路径。
