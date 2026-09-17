# dummy load + 起服加速：实测结果

> 2026-09-16 10:50–11:26 CST｜A3-node1 chips 8-15｜容器 `dsv41-a21-perf`
> 目的：回答两个问题 —— ①不带权重的起服能多快？②dummy 状态下能否测 Engram 读取与 decode 时延？

---

## 0. 结论

| 问题 | 答案 |
|---|---|
| **起服能多快？** | 真权重 8m29s → **dummy 4m56s**（**−42%**） |
| **dummy 能测 decode 时延吗？** | **能**。dummy 的 128K ms/step = **31.749**，真权重 32.894 ⇒ 差 **3.5%**，同量级 |
| **dummy 能测 Engram 读取吗？** | **能（时延）**。加一个 `V41_ENGRAM_WITH_DUMMY=1` 就把 host 路径带起来了（`[bneck]` 打了 64 行）。但 **A 恒为 1.0、数值是垃圾** ⇒ **只能测时延，不能测精度/接受率** |
| **附带收获** | 首次用同配置干净测出 **Engram 的时延代价 = 2.105 ms/step** |

---

## 1. 起服耗时对照（四臂）

| 臂 | 权重 | `MAX_SEQS` | **TIME-TO-READY** | 图捕获 | 静态内核编译 | 权重加载 |
|---|---|---|---|---|---|---|
| cmB（原基线） | 真 | 4 | **509 s（8m29s）** | 4 次 | 3 次 | 2 次 |
| bt2ms1 | 真 | **1** | **417 s（6m57s）** | **1 次** | **1 次** | 2 次 |
| do_dummy | **dummy** | 1 | **306 s（5m06s）** | 1 次 | 1 次 | **0 次** |
| do_dummyeng | **dummy + Engram** | 1 | **296 s（4m56s）** | 1 次 | 1 次 | **0 次** |

**两段加速，都可叠加**：

1. **`MAX_SEQS=1`：−92 s**（509→417）。原因：`MAX_SEQS=4` 让 decode token 数落 4 个桶
   （6/12/18/24），每个桶一次 capture + 一次静态内核编译（~45 s/个）；
   单流测量只需要 6 这一个桶。**不影响测量结果**。
2. **`--load-format dummy`：再 −111 s**（417→306）。跳过 86 个分片的读取
   （真权重 `Loading weights took 65.20 s` + draft `16.23 s`）。

**总计：509 s → 296–306 s（−40~42%）。**

---

## 2. dummy 下的时延是否可信（128K 单流）

| 臂 | 权重 | Engram | **ms/step** | A | 与基线之差 |
|---|---|---|---|---|---|
| f3b（真权重基线） | 真 | on | **32.894** | 3.493 | — |
| do_dummy | dummy | **off** | **31.749** | 1.0 | −1.15 |
| do_dummyeng | dummy | **on** | **33.854** | 1.0 | +0.96 |

**读数**：

1. **dummy 与真权重的 ms/step 只差 3.5%**（31.75 vs 32.89）⇒ device 侧的耗时结构
   （算子序列、shape、访存模式）**由 shape 与图决定，与权重数值无关** ⇒
   **dummy 可以用于 device 侧时延的优化迭代**。
2. **Engram 的时延代价 = 33.854 − 31.749 = 2.105 ms/step**（同配置、同会话批次、唯一变量是 Engram）。
   与历史报告的量级一致（`engram-final-quantification.md` 的「整条 host 路径 2.77 ms」、
   「可回收 0.6–1.0 ms」）。**这是首次用 dummy 拿到的干净数字。**

### 2.1 明确的限制（不要误用）

| 能做 | 不能做 |
|---|---|
| device 侧逐算子/整步**时延**对比 | **接受率 A**（恒为 1.0，因为随机权重让 draft 全错） |
| 结构性改动的方向判定（省 kernel / 合并） | **精度 / 正确性**（数值全是垃圾） |
| Engram host 路径的**时延**（D2H/hash/route/a2a） | Engram 的**数值正确性** |
| 图捕获/padding 效应（`cudagraph_capture_sizes`） | prefill 之后的真实输出 |

---

## 3. 实现细节（踩过的三个坑）

### 3.1 `--load-format dummy` 与 `--model-loader-extra-config` 互斥

```
ValueError: Model loader extra config is not supported for load format dummy
```

⇒ `LOAD_FORMAT=dummy` 时必须跳过 `--model-loader-extra-config`
（`serve_v2.sh` 已加 guard）。

### 3.2 `wo_a` 在 dummy 下是 2D 形状 ⇒ 图捕获崩溃

`wo_a` 的真形状是 3D `[n_local_groups, group_hidden, o_lora_rank]`，
由 `ops/linear.py:445-495` 的 `weight_loader` 在**加载期**转过来的。
**dummy 不调 weight_loader** ⇒ 权重停在声明的 2D 形状 ⇒ F3 的 2D 分支拿到错的布局：

```
RuntimeError: matmul_implement_npu ... call aclnnMatmul failed, error code is 161002
  at attention/dsa_v1.py:1585 in _forward_o_proj
```

⇒ 修法：F3 的 2D 分支在权重是 2D 时用 `.t().contiguous()` 接受它
（真加载走 3D 分支，行为不变）。

### 3.3 Engram 默认在 dummy 下被跳过

容器 `models/deepseek_v41/model.py:654`：

```python
if ascend_config.enable_engram and vllm_config.load_config.load_format != "dummy":
    with torch.device("cpu"):
        tokenizer = AutoTokenizer.from_pretrained(self.engram_root)
        self.engram_history = PagedNgramHistory(config, tokenizer)
        ...
```

这段只读 tokenizer + 一个**很小**的 `quarot.safetensors`（**不读 206 GB 的表**）。
206 GB 的表本体是 `NodeShardedEngram.weight = torch.empty(...)`（`engram_hbm.py:450`），
dummy 下同样按 shape 分配。

⇒ 加 `V41_ENGRAM_WITH_DUMMY=1` 绕过这个判断，
`prepare_engram` 的 host 路径（D2H + hash + route + all_to_all）就**照常执行**。

实测 `[bneck] mode=stock steps=20 dec=19 ... total=142.646`（64 行），确认在跑。

---

## 4. 建议的用法（三层，成本已量化）

| 层 | 场地 | 起服 | 判什么 |
|---|---|---|---|
| **T1 单卡 harness** | A3-node2 chip7 | **0（不起服）** | 单算子/单层结构、逐算子耗时、数值等价 |
| **T1.5 dummy 8卡** | A3-node1 chips 8-15 | **~5 min** | 8 卡 device 侧整步时延、图/padding 效应、Engram 时延 |
| **T2 真权重 8卡** | A3-node1 或 A3-node2 | **~7 min**（MAX_SEQS=1） | 需要接受率的 ms/step、tok/s |
| **T3 精度/正确性** | 独立线 | ~9 min | GSM8K / Vision / 非确定性诊断 |

**`MAX_SEQS=1` 应当成为所有单流实验的默认**（省 92 s，且不改变测量）。

---

## 5. ⚠️ 对既有结论的一处修正（来自 F24 子代理）

F24 子代理在 12288 上下文上跑 5 臂 × 5 发 **全部不确定**，8192 有 2/5 臂不确定。
这与本报告早先「≤16384 输出确定」的结论**冲突**。
可能原因：早先的探针只生成 **4 个 token**（容易恰好相同），F24 用 **48 个 token**。

⇒ **修正**：确定性并非由 16384 这一个阈值简单划分；
**短输出（≤4 token）在 ≤16384 是确定的，长输出（48 token）在 12288 就可能不确定**。
细节待补测（`reports/ctx-nondeterminism.md` 需要更新）。

---

## 6. 产物

| 内容 | 路径 |
|---|---|
| 起服/时延原始日志 | `/tmp/dummy_only.log`、`/tmp/dummy_boot_test.log` |
| 各臂起服日志 | `logs/perf/{bt2ms1,do_dummy,do_dummyeng,cmB}_*_serve.log` |
| 测量 jsonl | `logs/perf/a21/p42_t4_quote_131072_{do_dummy,do_dummyeng,bt2ms1}_*.jsonl` |
| 补丁（幂等） | `exp_tools/patch_load_format.py`、`patch_loader_mt_guard.py`、`patch_engram_dummy.py`、`patch_wo_a_dummy_shape.py` |
| 实验脚本 | `exp_tools/dummy_only.sh`、`exp_tools/dummy_boot_test.sh`、`exp_tools/boot_time_test.sh` |
