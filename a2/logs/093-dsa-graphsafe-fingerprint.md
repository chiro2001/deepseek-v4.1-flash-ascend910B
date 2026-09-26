# 093 — ★★★★★ 抓到 `logs/082` 那条警告的实测证据：捕获期 `LocalScalarDenseNpu.cpp:23 + EE1016` = dsa 不是 graphsafe 版的指纹

> 2026-09-23 00:2x–00:4x CST（远端 A3-node1）。执行：**主代理**（`mode2` 臂捕获期炸掉后逐项定位）。
> 触发：`r8-4axis-mode2` 臂在**图捕获期**报 `EE1016`×8、容器死亡。
> 标记：**【实测】/【推断·强】**。

---

## 0. 一句话

`mode2` 臂的崩溃**与 `TRUE_TOKENS=2` 和 `COMP_JSON` 都无关** ——
它是**第三个变量**：这一臂挂的 `attention/dsa_v41.py` 是**没打 graphsafe 补丁的那份**
（`_R8DSA` 落到了 `auto→C`，取 `X_integrate/pkg-ring` 的原样版）。

> ★ **三条臂里唯一的共同变量就是「dsa 是不是 graphsafe 版」**：
> 挂了 `94aeebb7`（`rows_bound=14`）⇒ 通过；挂了 `rows_bound=0` 的那两份 ⇒ **捕获期必炸**。

---

## 1. 【实测】三条臂的并排（这就是那次「两次对比失败」的真相）

| 臂 | `dsa_dir_D`（实际挂的） | dsa md5 | `rows_bound` | 捕获期结果 |
|---|---|---|---:|---|
| `r8-4axis-exact`（**用我的启动器**起） | ★ **`S_graphfix/pkgs/pkg-kv8pf`** | ★ **`94aeebb757d6d5708268754481a05e0a`** | ★ **14** | ✅ 通过（后续跑完 3 轮） |
| `r8-4axis-mode2`（**手工起**） | `X_integrate/pkg-kv8pf`（runner 的默认） | `75f4e565adc1b12c854a0a01271b6c4d` | **0** | ⛔ `EE1016` ×8，容器死 |
| `r8-c2-tierC-graph`（历史，10:11） | `X_integrate/pkg-kv8pf`（默认） | `75f4e565…` | **0** | ⛔ **同一签名** |

★ 主代理核对方式（可复算）：

    grep -aoE "dsa_dir_D=\S+" <arm>.meta.txt     # exact → …/S_graphfix/pkgs/pkg-kv8pf   mode2 → …/X_integrate/pkg-kv8pf
    md5sum <...>/attention/dsa_v41.py ; grep -c rows_bound <file>

### 1.1 为什么 `mode2` 会掉进 C

    run_arm_r8.sh:97    R8_KV8_DIR_D=${R8_KV8_DIR_D:-$X/pkg-kv8pf}      ← 默认指向「没修」的那份
    run_arm_r8.sh:213   R8_INT8_TIER="$TIER" R8_DSA_SRC="${DSA_SRC:-auto}" \
    shadow serve_a2.sh  auto 分支：TIER != D ⇒ _R8DSA=C ⇒ 取 $_R8C/attention/dsa_v41.py

⇒ ★ **不显式给 `DSA_SRC=D R8_KV8_DIR_D=<S 的 pkg-kv8pf>`，档 C + 图模式就会挂到没修的那份** ——
这正是 `logs/082` 写下的那条，**当时还是【推断】，现在是【实测】**。

★ `mode2` 臂的启动记录里确实**没有**这两个 env（子代理手工起的臂，没走我给的 `run_4axis_arm.sh`）；
而 `exact` 臂的 `meta` 里 `dsa_dir_D` 指向 `S_graphfix` ⇒ 那次跑的是我启动器里带的配方。

---

## 2. 【实测】失败签名（**可当指纹用**）

    (Worker_TP2_EP2) NPUGraph: ERROR — capture failed: RuntimeError: operator():
        ../torch_npu/csrc/aten/common/LocalScalarDenseNpu.cpp:23 NPU function error: c10_
        rtStreamSynchronizeWithTimeout execution failed, reason=stream is captured
        Not_Supported(EE1016): Synchronizing a stream failed.
        Reason: Stream (stream_id=31) during the capture stage is not supported.
    栈：capture_model → … → _dummy_run → _model_forward → acl_graph.py → vl_model.py:237

★ **判据指纹**：`grep -ac "capture failed" <serve.log>` > 0 且里面含 `LocalScalarDenseNpu`
⇒ 先查 **`dsa_dir_D` 指向哪份 `dsa_v41.py`、其 `rows_bound` 是否为 0**，
**不要**先去怀疑 Engram / int8 开关 / 池分量（我这次差点就去怀疑后两个）。

### 2.1 三个被排除的嫌疑（**都用实测排除，不是推断**）

| 嫌疑 | 排除依据 |
|---|---|
| `VLLM_V41_ENGRAM_TRUE_TOKENS=2` | 与本签名无关：`exact`（=1）通过、`mode2`（=2）失败，而两者**挂的 dsa 不同**；且 `mode2` 的 `ENGRAM-TRUE-TOKENS` 行数是 **0**（host 路径还没跑到就打死了） |
| `P2_COMP_JSON`（真分量） | `r8-4axis-tierB` 臂用的是**真分量**且**通过** ⇒ 真分量本身不会炸图捕获 |
| 静态内核编译失败 | 两臂都是 `171/172 JSON files compiled successfully`（**同一格失败**，是既有现象） |
| `ENGRAM_DEVICE_INDEX` | `mode2` 的 `DEVICE-INDEX` 计数 = **0**，`build_request_ids` = **0** ⇒ 没走 device 路径 |

---

## 3. 处置（已派单）

1. ★ **`mode2` 臂作废（配置失败，不是判据失败）**，**不要**把它的 `EE1016` 记进任何对照表；
2. **重跑时必须带**（或直接用 `run_4axis_arm.sh`，它已经内置）：

       DSA_SRC=D R8_KV8_DIR_D=$S/pkgs/pkg-kv8pf        # S = agents/S_graphfix
       # 起服前自检：grep -aoE "dsa_dir_D=\S+" <arm>.meta.txt   → 必须含 S_graphfix

3. **给 `run_4axis_arm.sh` 补一道运行期门**（G2 现在只检查「我指定的那份是 graphsafe」，
   没有检查「runner 实际挂的是不是我指定的那份」）⇒ 加：起服前断言 `dsa_dir_D` 含 `S_graphfix`。

---

## 4. 与今天其它条目的关系（这是「判据问题」第 7 次，但性质最好）

| 出处 | 性质 |
|---|---|
| `079 §3` / `083 §2` / `084` / `085` / `087` / `089` | 判据或对照**自身**有问题 |
| ★ `093` | 判据没错，是**前置没带全**（配置漏了一个 env）⇒ 而且**指纹唯一、能一眼认出来** |

★ **可复用的收获**：`logs/082` 那条「必须显式指向 graphsafe 包」从警告变成**带指纹的判据**；
并且再次印证「**手工起臂会漏前置，用 `run_4axis_arm.sh` 才不会漏**」。
