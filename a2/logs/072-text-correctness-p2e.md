# 072 — ★★★ `p2e`（A2 目标配置）**replay 轮引擎死亡**：根因是 `engram_hash.py` `KeyError`，**不是** device-index

> 2026-09-22 20:0x–20:3x CST。执行：子代理 **`a3_text_correctness`**。
> 任务来源：用户「**需要测试其能够返回正确的文本**」；主代理把它派成本轮唯一判据缺口。
> 标记：**【实测】/【推断】/【未确认】**；★ = 决定性。

---

## 0. 一句话

★★★ **`ENGRAM_DEVICE_INDEX=0` 确实修掉了起服期崩溃**（`EH0012=0 / DEVICE-INDEX=0 / 207001=0`），
**但 p2e 的 replay 轮仍然死了** —— 根因是**另一个、独立的**缺陷：

```
KeyError: 2486   @ models/deepseek_v41/engram_hash.py:463  _engram_update_jit
   ← model.py:1092 _prepare_engram_host → engram_history.update()
   ← model.py:1023 prepare_engram → model.py:1129 prepare_engram_inputs
   ← vl_model.py:223 → model_runner_v1.py:2989 _model_forward
```

**8/8 rank 逐字相同**。异常从 `_model_forward` 逃出 ⇒ 正好落在引擎 `submit()/release()` 之间
⇒ device-metadata 标志**永久置位** ⇒ 其后所有请求死在
`RuntimeError: The previous device metadata submission has not been released`（**32 次**）。

⇒ ★ **`056` 里那条"②a 特有"的 `device_metadata` 泄漏，在 p2e 上以完全相同的形态复现了 —— 而 p2e 没有 int8、没有 draft-int8。**
那条判据（"泄漏 ⇒ ②a 特有"）**已被推翻**。

---

## 1. ★★ 任务 ① 的诚实回答：**"能不能返回正确文本" —— 本轮答不了**

两条都成立，任一条就足以让这个判据无效：

1. **p2e 的 prompt 是随机 token id，不是自然语言。**
   `kv_offload_client.py: make_prompt()` 生成 `[1000 + ((base + i*7919) % 100000) ...]`
   ⇒ 生成的是**乱码**，没有语义可判。fill 轮 16 条输出实测全部是乱码
   （例：`' က kernel Glassุ่ม_color则为Authentication皮的 riots下列说法جان fig自治 jo'`）
   —— ★ 但这是**预期**的（输入本身乱码），**不能当质量证据**。
2. **服务在 replay 轮就死了，容器已被 runner 删除**（`docker rm -f`，20:04:04 释放 Phy-ID 8..15）
   ⇒ **没有活着的服务可问**。

⇒ **结论：整个 p2e 臂（以及此前所有 8 卡臂）从头到尾没有做过任何语义正确性判据。**
这条缺口**依然存在**（`071` §A1 的定性不变）。

---

## 2. ★★★ 本轮真正的产出：replay 轮的实际结果

【实测】`p2e-engram1-dev0-dg1-offload.client.json`：

| 轮 | `requests_ok` | `requests_failed` | `wall_s` | `gen_tokens` | `ttft.n` |
|---|---:|---:|---:|---:|---:|
| **fill** | **16/16** | 0 | 374.415 | 1651 | 16 |
| ★ **replay1** | ★ **3/16** | ★ **13** | **5.771** | 89 | **2** |

* fill：`ttft.mean = 20251 ms`（128K prefill 20 s，正常）；
* replay：`ttft.mean = 1551.7 ms`（★ 快 13×，**取回路径确实在工作**）——
  但**只有 3 条成功，且第 3 条 `gen_tokens=0`（空输出）**；
* `replay_matches_fill_sha256 = False`；`sha256_common_prompts = 3`、`sha256_mismatched_prompts = [0,1,2]`。

⚠️ **必须同时说清的一条口径**：replay 用的是 `--replay-prompt-tokens 65536`
（= 原 131072 prompt 的**前一半**）⇒ **fill vs replay 的文本对比不是同输入**，
所以 `mismatched` **不能**单独当"保真失败"的证据。**真正的判据是那 13 条失败。**

### 2bis. 三条成功的 replay 输出（供后人参考，不作判据）

```
[0] ' 不\n\n## 结论\n\n经过对上述内容的分析，我们可以看到这段文本包含了大量的乱码、非中文字符和难以理解的片段。它似乎是由多种语言'
[1] '/ 我 们 的 生 活 中 有 很 多 不 同 的 挑 战 和 机 遇 需 要 我 们 去 探 索 和 发 现 自 己 的 潜 '
[2] ''（gen_tokens=0）
```

★ 有意思但**不外推**：这两条是**通顺的中文**，且内容正是"上面这段是乱码"的评论 ——
即模型在**正确地**回应一个乱码 prompt。这与 fill 轮的乱码输出形成对比，
但**因为 prompt 长度不同（64K vs 128K），本轮无法归因**。【未确认】

---

## 3. 根因链（逐条【实测】）

### 3.1 第一异常：`engram_hash.py:463`

```python
# engram_hash.py（shadow-pkg/patches/files/engram_hash.py，542 行版本）
    ret = _engram_update_kernel(...)
    err, oob, fell_back = ret[0], ret[1], ret[2]
    if oob < 0:  ...扩容重跑...
    ...
    if err >= 0:
        raise KeyError(err)        # ★ :463 ← 本轮 err = 2486
```
* ★ 注意 `oob`（页号越界）**有自己的扩容分支**，而 `err` 直接抛 ⇒ **`err` 是另一类失败**。
* 8 个 rank 的 `KeyError` **值完全相同（2486）** ⇒ 确定性、非竞态。

### 3.2 下游：device-metadata 标志永久置位

【实测】计数（`grep -c`，p2e serve.log 916,889 B）：

```
EH0012                      0     ★ 起服期崩溃已被 ENGRAM_DEVICE_INDEX=0 消除
DEVICE-INDEX                0     ★ 开关真的关着（判据成立）
207001                      0
aclrtHostRegister failed    0     ★ 无静默回落
KeyError                   32     ★ 8 rank × 4（异常 + 重复展开）
has not been released      32
EngineDeadError             6
```

⇒ ★★ **`model.py` 的 CHANGELOG（我们自己的补丁，132 行起）早就写明了这个机制**：
> *"异常从 forward 里逃出，正好落在引擎 submit/release 之间，把那个标志永久置位；
> 后续请求全部死在 submit 上"*

★ 那条注释归因于 `build_request_ids` 的设备不匹配；**p2e 的归因是 `engram_hash` 的 `KeyError`** ——
**同一个"逃逸窗口"，不同的逃逸者**。

### 3.3 ★ 触发锚点：**部分 prefill 恢复**（【推断】，但已定位到具体请求）

dump 出来的 `SchedulerOutput` 里那条请求：

```
NewRequestData(req_id=cmpl-b0829522dfd75ae9-0-b8c4592f,
               prompt_token_ids_len=65536,
               num_computed_tokens=56320,          ← ★ 55K 来自池（缓存命中）
               num_scheduled_tokens={...: 8064})   ← 只算剩下的 8064
kv_connector_metadata=OffloadingConnectorMetadata(
    load_jobs={}, store_jobs={278: TransferJob(...)})   ← 同时在写回
```
⇒ **崩溃发生在"从卸载池取回 + 续算尾部"的请求上**。【推断】
★ 与 `p1a/p1b/p2c`（**ENGRAM=0**）的对照（见 §4）支持"**Engram × 卸载**"这个组合是必要条件。

---

## 4. ★★ 对照矩阵（我独立 grep 的，不是转述）

```
臂                              ENGRAM   device_metadata 泄漏   EngineDeadError   KeyError
p1a-tierB-dg0-offload              0            0                  0             0
p1b-tierB-dg1-offload              0            0                  0             0
p2c-dg1-tinypool-noengram          0            0                  0             0
p2b-engram1-tinypool-dg1           1            8                  0             0
p2d2-engram1-pageable-56g          1            4                  5             0
★ p2e-engram1-dev0-dg1-offload     1           32                  6           ★ 16
```

**三条立刻能读出来的事实**：
1. ★ **`ENGRAM=0` 的三个臂全部零泄漏零死亡** ⇒ 卸载本身不是凶手；
2. ★ **`ENGRAM=1` 的三个臂全部有泄漏** ⇒ **Engram 是共同因素**；
3. ★ **只有 p2e 有 `KeyError`** ⇒ p2e 是**新的失败模式**（`ENGRAM_DEVICE_INDEX=0` 把起服期的
   `EH0012` 消除之后，**暴露出了排在后面的这个**）。

★ 一条**撤回**：`056` 把那条 `device_metadata` 泄漏定性为"②a（draft 也 int8）特有"。
**现在 p2e（无 int8、无 draft-int8）以完全相同形态复现 ⇒ 该定性不成立。**
正确的表述是：**它是"异常从 forward 逃逸"的通用后果，凡是能逃逸的异常都会产生它**；
②a 只是**其中一个**逃逸者。

---

## 5. 对 **A2 上线**的含义（本节是给主代理/用户的，不是判据）

| 结论 | 依据 |
|---|---|
| ★ **`ENGRAM_DEVICE_INDEX=0` 是必需的，且有效** | p2e：`EH0012=0 / DEVICE-INDEX=0`，起服 667 s 成功、KV 427,643、池注册 128 行 `ret=0` |
| ⛔ **但它不足以让 A2 的目标配置可用** | replay 轮 13/16 请求失败、引擎死亡 |
| ★ **新阻塞是 Engram 自己的 `KeyError(err)`** | 8/8 rank 同值 2486，确定性 |
| ⚠️ **A2 与 A3 的差异可能救场，但未验证** | A2 是 `host_mem_pool=0`，A3 是 1；且 A2 的 `MAX_SEQS=4` vs 本臂 32 |

★ **本臂 `max_seqs=32`**（`meta.txt` 实测）—— 与 A2 生产的 `MAX_SEQS=4` **不同**，
且 `DRAFT_GRAPH=1 + 并发≥16` 有已知的 P0-C 风险 ⇒ **这一格也是混淆项，不能直接外推到 A2**。

---

## 6. 下一步（按优先级）

1. ★★★ **读 `from err` 的语义**：`_engram_update_kernel` 在哪个文件？`err=2486` 是哪一类
   （页容量？行号？token id？）—— **这一格决定修法**。★ 本轮**没做完**（时间盒）。
2. ★★ **最小复现**：`ENGRAM=1 + DEVICE_INDEX=0 + 卸载`，**只发一条**「128K 填 + 64K 重放」，
   看是否同样崩 —— 把 §3.3 的"部分 prefill 恢复"从【推断】拉到【实测】。
3. ★★ **分离 `max_seqs`**：同臂 `max_seqs=4`（= A2 生产）跑一遍，看是否还崩。
4. ★★★ **语义正确性判据仍然没做**（§1）—— 参见 `071` §A1；建议用**自然语言 needle prompt + 长生成**，
   而不是随机 token + sha。

---

## 7. 交付与复现

| 项 | 路径 | 状态 |
|---|---|---|
| 本日志 | `a2/logs/072-20260922-text-correctness-p2e.md` | 本机 ✅ |
| 原始数据（**4 件，已取回本机**） | `a2/logs/raw/072-text-correctness/` | 本机 ✅ |
| ├ `p2e.client.json` | 63,882 B | ✅ |
| ├ `p2e.serve.log` | 916,889 B | ✅ |
| ├ `p2e.meta.txt` | 5,644 B | ✅ |
| └ `p2e.kv_events.json` | 568 B | ✅ |
| `trace.txt`（12,001 B，int8 trace，档 B 下为 0 行预期） | A3：`agents/R_8card_int8/out/p2e-…/` | **待 `cos-xfer`** |

已取回的四件里两条额外读数（供参考）：
```
rc = 0            ← ★ 臂被判为成功（尽管 replay 13/16 失败）——见 §8
kv_events: BlockStored:CPU = 29282 / BlockRemoved:GPU = 188719 / AllBlocksCleared:None = 1
pool: P1 注册 128 行 = 197.21 GiB；路径①②③ 三条一致 ✅
```

复现命令（读，不写）：
```bash
python3 - <<'EOF'
import json
d=json.load(open("p2e-engram1-dev0-dg1-offload.client.json"))
for r in d["rounds"]:
    print(r["tag"], r["requests_ok"], r["requests_failed"], r["wall_s"])
EOF
grep -cE "KeyError|has not been released" …/serve.log
```

---

## 8. 时间盒说明（如实）

本轮 40 min 时间盒内**没有**完成原计划的 ①②③：
* ① 无法做（服务已随臂结束被删、且 prompt 无语义，见 §1）；
* 我在**发现 replay 轮 13/16 失败**之后**改变了优先级** —— 因为"能不能返回正确文本"的
  前置条件恰恰是"服务还活着"，而这一条**当场就被推翻了**。
⇒ 交付改为**把失败定性到根因**（§3、§4），并如实标出**仍然空缺的语义判据**（§1）。
**未完成**：§6 的 1/2/3。
