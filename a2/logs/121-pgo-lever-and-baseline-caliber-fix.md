# 121 — ★★★ **自我更正 + 一个被漏掉的杠杆**：历史基线是 **PGO 开/关两个臂的合并中位**；而我们的臂**全部 `PYTHON_PGO=0`** ⇒ (a) 我此前的"128K 慢 3.3%"是**错口径** (b) PGO 可能值 **1.3–2 ms/step**

> 2026-09-23 07:0x CST。执行：**主代理**（读 `a2_pkg_v3/EXPECTED_PERF.md` + 用原始 jsonl 逐发复算 + 无 NPU 的镜像探测；**零占卡**）。
> 标记：**【实测】/【推断】/【未确认】**。★ 本卷**含一处对既有结论的更正**（`logs/115`/`117` 的"128K 慢 3.3%"）。

---

## 0. 一句话

我把"加 int8 之前的基线 30.45/31.52/30.92"当成**一把尺子**用了 —— 它其实是 **`faA`（`PYTHON_PGO=0`）+ `faB`（`PYTHON_PGO=1`）两个臂的合并中位**。
而我们的**全部 8 卡臂都跑在 `PYTHON_PGO=0`**（`agents/R_8card_int8/scripts/run_arm_r8.sh:226` **硬编码**）。
⇒ 两个后果：① 我此前报的 **"128K 比基线慢 3.3%" 是错口径**（同类比只慢 **1.0%**）；② **PGO 自身是一笔我们没拿的钱**（同源 A/B：8K **−7.5%** / 32K −5.1% / 128K −4.4%）。

---

## 1. 【实测】基线两臂的逐发复算（原始 jsonl，不引用二手数）

数据：`a2_pkg_v3/logs_meta/samples/p42_t4_quote_*.jsonl`（逐字节复制自 A3-node1，2026-09-16）。
口径：`quote` 单流、`max_tokens=256`、`temperature=0`、chips 8-15、真权重、全补丁默认集合（除 `PYTHON_PGO`）。
复算脚本：本卷 §6。

| 上下文 | **`faA`（PGO 关）** | **`faB`（PGO 开，官方推荐）** | **合并中位**（= 我此前当"基线"用的） | PGO 效应 |
|---:|---:|---:|---:|---:|
| 8K | **31.391**（n=3，30.51–32.22） | **29.032**（n=3，28.75–30.38） | 30.446（n=6） | **−7.5%** |
| 32K | **32.159**（n=2） | **30.512**（n=2） | 31.517（n=4） | −5.1% |
| 128K | **31.626**（n=8，30.66–32.34） | **30.230**（n=8，29.79–31.18） | 30.923（n=16） | −4.4% |

★ 合并列 **与 `logs/102` 记的 30.45/31.52/30.92 逐字吻合** ⇒ 复算口径正确，那三个数确实是**两臂合并**的结果。
★ `EXPECTED_PERF.md §1/§2` 自己也把两臂分开列了（31.63 / 30.23 @128K），**是我在 `logs/102` 那一步把它们并到了一起**。

---

## 2. ★ 更正：我们的 128K 到底比基线慢多少

| 比法 | 8K | 32K | 128K |
|---|---|---|---|
| 我们（现在，`PYTHON_PGO=0`） | **27.905** | 29.226 | **31.950** |
| vs **`faA`（PGO 关，与我们的开关状态同类）** | **−11.1%** | **−9.1%** | **+1.02%** ⚠️ |
| vs **`faB`（PGO 开，官方推荐配置）** | −3.9% | −4.2% | **+5.7%** ⚠️ |
| vs 合并中位（**我此前用的口径，作废**） | −8.4% | −7.3% | +3.3% ❌ |

⇒ ★ **准确表述**：
* **128K 我们确实还没赢** —— 同类比（都 PGO 关）慢 **1.0%**（0.32 ms），对官方推荐配置慢 **5.7%**（1.72 ms）；
* 8K/32K 的"已超越"**在同口径下成立且更强**（−11.1% / −9.1%）。
* ⚠️ 但**前提是"只差 PGO"** —— 两侧还有其它差异（包不同：`a2_pkg_v3` vs `shadow-pkg`；模型是否同一份量化产物未逐字节比对）⇒ **该对比仍是【推断】，不是严格单变量**。

---

## 3. 【实测】PGO 在**我们现在这条链**上是可以开的（我没猜）

| 检查项 | 结果 |
|---|---|
| 产物是否存在 | ✅ `shadow-pkg/optim/pgo/{python3, libpython3.12.so.1.0}`（mtime `Sep 16 18:27`，**与 `faB` 基线同一批**） |
| 镜像内目标落点 | ✅ 无 NPU 的 `docker run --rm --entrypoint python3 <IMAGE>` 实测：`FOUND /usr/local/python3.12.13/lib/libpython3.12.so.1.0` |
| `TARGET_PATH.txt` | ❌ 当前缺失，但 `serve_a2.sh:718-756` 有 **`[PGO-AUTODETECT]`** 分支会自动探测并落盘 |
| 失败模式 | ✅ **fail-safe**：探测失败 ⇒ `WARNING: PYTHON_PGO=1 但没有可用的 PGO 产物 → 降级为不挂` 且把 `PYTHON_PGO` 改回 0 |
| 为什么我们没用上 | `agents/R_8card_int8/scripts/run_arm_r8.sh:226` **硬编码 `PYTHON_PGO=0`**（`serve_a2.sh:194` 的默认其实是 **1**） |

★ **已知未知**：`run_arm_r8.sh:226` 会不会覆盖调用方传的 `PYTHON_PGO`（它写在 `nohup env` 的参数表里 ⇒ **很可能是硬赋值**）⇒ **必须显式改/覆盖，并在起服日志里核 `pgo_target=<绝对路径>`**（否则 A/B 是假的 —— 本仓"改了没生效"已栽 4 次：`081`/`093`/`099`/`112`）。

---

## 4. 【推断】PGO 能给我们多少（三种可能，别只挑乐观的）

| 情形 | 依据 | 8K 预期 |
|---|---|---|
| **不可迁移**（我们已设备限定，host 暴露很小） | `ENGRAM_TUNE`：`G ≈ 29.5–30.6 ms = 93–95%`，**步是设备限定的**；真 `H_after ≈ 1.6–1.9 ms` | ≈27.9（**无收益**） |
| **部分迁移**（只兑现 host 暴露的那部分） | 同上 + `FUSE_TUNE` 实测 `hp − d2h` = host 总工作 **17.2–19.0 ms**（但大部分与设备重叠） | 26.5–27.5 |
| **整体迁移**（与基线同幅） | `faA→faB` 的 −7.5%/−5.1%/−4.4% | **≈25.8** |

⇒ ★ **这条不确定性的正确处置是"测"**，而且它**必须测**，因为它同时决定：
1. 我们离 24 还有多远（可能直接吃掉 2 ms）；
2. **与基线对比是否公平**（我们的开关状态要和被比的一方对齐）。

★ 已派出：让正在跑 8 卡臂的 `G2G3_ARM` 在**同代码、只改 `PYTHON_PGO`** 的条件下补一条对照臂，判据用 **`hp − d2h`（两个同向移动的量，比单看 `hp` 抗噪）+ 三档 quote**，并**优先保证它主任务的 op 判据完整**（PGO 臂时间不够就只跑 8K 一档）。

---

## 5. 对 goal 账目的影响

| 项 | 之前（`logs/119 §6`） | **本卷后** |
|---|---:|---:|
| A 层（我们可改的代码） | 1.04–1.47 | 1.04–1.47（不变） |
| **PGO（新增，数值无关）** | —— | **0 ～ 2.1【未确认】** |
| 合计乐观 | 27.905−1.47 = 26.4 | **27.905 − 1.47 − 2.1 = 24.3** |
| 合计保守 | 同上 | 26.4（PGO 不迁移） |

⇒ ★ **"≤24" 第一次出现"数值无关"的可能路径**（此前只有 `hc_sinkhorn_iters` 那条**会动数值**的 + `HcPre` 定制核那条周级的）。
★ 但**未确认**：PGO 在**设备限定**的步里能兑现多少 —— 这条现在是**最便宜、最有价值的一个实验**。

---

## 6. 复算配方（只读，零占卡）

```bash
# ① 基线两臂分开复算（本卷 §1 的数就是它打出来的）
cd ~/projects/dsv41 && python3 - <<'PY'
import json,glob,statistics,re,os
rows={}
for f in sorted(glob.glob('a2_pkg_v3/logs_meta/samples/p42_t4_quote_*.jsonl')):
    m=re.match(r'p42_t4_quote_(\d+)_(fa[AB])_', os.path.basename(f))
    if not m: continue
    tok,arm=int(m.group(1)),m.group(2)
    for line in open(f,encoding='utf-8',errors='ignore'):
        line=line.strip()
        if not line: continue
        o=json.loads(line)
        if o.get('_meta') or not o.get('metrics_ok') or o.get('error'): continue
        if abs(o.get('prompt_tokens_actual',0)-tok)>64: continue
        rows.setdefault((tok,arm),[]).append(o['ms_per_step'])
for k in sorted(rows): print(k, len(rows[k]), round(statistics.median(rows[k]),3))
PY

# ② PGO 产物与目标落点（无 NPU）
ls -la ~/projects/dsv41-upstream-pr/shadow-pkg/optim/pgo/
docker run --rm --entrypoint python3 quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3 \
  -c 'import sysconfig,os;n=sysconfig.get_config_var("INSTSONAME");print(os.path.join(sysconfig.get_config_var("LIBDIR"),n))'

# ③ 我们为什么是 0（硬编码点）
grep -n "PYTHON_PGO" ~/projects/dsv41-upstream-pr/agents/R_8card_int8/scripts/run_arm_r8.sh
```
