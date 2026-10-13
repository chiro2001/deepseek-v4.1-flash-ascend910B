# F3 融合为何没生效：主层走的是镜像版 `dsa_v41.py`（根因 + 阻塞点）

> 承接 `FUSION-CANDIDATES-20261007.md` 的 F3（`RmsNorm` + `DynamicQuant`）。
> 本文把"融合分支为什么没被走到"追到根，并更正一处**代码保真度**事实。
> 全部为【实测】。

---

## 0. 三句话

1. **`dsa_v1.py` 里有融合分支，而且对 draft 层生效了**（profile 实测 `RmsNormDynamicQuant [40,1280]` × 2.3/步）；
   但**40 个主层不走这条路径** —— 它们走的是**镜像里的 `dsa_v41.py`**，那份是**未融合版**。
2. **这不是疏忽，是数据流约束**：`qr` 要**以浮点形式**喂给 indexer
   （`indexer.py:134` 的 `self._output(self.wq_b, qr)`），而融合算子只吐 int8 + scale。
3. ⇒ **F3 在这一项的收益不能靠"改一行"拿到**，需要**同时给 indexer 补量化-qr 支持**。
   难度从"低"上调为"中"，且**必须重验精度**。

---

## 1. 代码保真度：`dsa_v1.py` 有三个版本，且主层根本不走它

### 1.1 三个版本

| 位置 | 行数 | md5 |
|---|---:|---|
| 本地仓库 `patches/files/dsa_v1.py` | 2163 | `9a36e709…` |
| 本地仓库 `patches/files/draft/dsa_v1.py` | 2437 | `788566674a0b…` |
| a3-21 `cedpd-repo/patches/files/draft/dsa_v1.py` | **2452** | **`1ede9fdd…`** |
| **tp8k5 容器内实际加载** | **2452** | **`1ede9fdd…`** |

**关键**：`scripts/serve_a2.sh:1190` 挂载的是 **`draft/dsa_v1.py`**，不是 `files/dsa_v1.py`。
⇒ 我先前一直在读的 `files/dsa_v1.py` **不是运行的那份**。

**a3-21 版 = 仓库版 + DBO 实验块**（`[DBO-NODEVMD-SCOPE]`、`[QLI1-CALL]`、`DYNSPEC-DIAG`），
对 RmsNorm/quant 路径**无功能影响**；但**仓库里缺这些改动**，属交付一致性问题。

### 1.2 ★ 主层走的是第四个文件：镜像版 `dsa_v41.py`

```
pts8k5 挂载清单里  attention/dsa_v1.py   ← 来自 draft/dsa_v1.py（我们改的）
镜像自带          attention/dsa_v41.py  ← ★ 1092 行、0 探针、Sep 11，**未被挂载覆盖**
```

`patches/files/model.py:80` 从 `vllm_ascend.attention.dsa_v41` 导入，
而 `dsa_v41.py` 有**自己的一套 prolog 实现**（不走 `dsa_v1.py` 的 `_mla_prolog_*`）。

⇒ **主层（40 层）的 q/kv 投影在 `dsa_v41.py` 里；draft 层（3 层）走
`dsa_v1.py` 的 `_mla_prolog_single_stream`。**

---

## 2. 证据链（四步）

### 2.1 profile：融合算子只在 draft 形状上出现

| 算子 | 形状 | 次/步 | 中位 | 步内位置 |
|---|---|---:|---:|---:|
| `RmsNormDynamicQuant`（**融合版**） | **`40,1280`** | **2.3** | 21.2 µs | **0.87** ← draft 相位 |
| `RmsNorm` | `48,1280` | 30.6 | 13.57 µs | 0.42 |
| `aclnnDynamicQuantV2` | `48,1280` | 36.7 | 3.81 µs | 0.43 |

（`q_lora_rank = 1280`，`hidden = 5120`，`head_dim = 512` — 来自 `config.json`）

⇒ 主层形状 `[48,1280]` **只有分离的两个 kernel，没有融合版**。

### 2.2 两个 `_mla_prolog_*` 用的是同一个检查 ⇒ `is_w8a8` 必为 True

```python
# _mla_prolog_single_stream（draft 用）—— 无条件融合
if _is_w8a8_dynamic(self.wq_b):
    qr, qr_pertoken_scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(...)

# _mla_prolog_multistream（主层用）—— 被 is_prefill 挡住
if is_prefill:
    qr = self.q_norm(wq_a_result)
    q_b_quant, q_b_scale = self.cv_wq_b.quantize(qr)
elif is_w8a8:
    qr, qr_pertoken_scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(...)
```

draft 融合成功 ⇒ `_is_w8a8_dynamic(wq_b)` = **True**
（且 `quant_model_description.json` 实测 **40/40 层 `wq_b` 都是 `W8A8_DYNAMIC`**）。

### 2.3 但主层没走这段代码 —— 它们在 `dsa_v41.py` 里

```python
# 镜像版 dsa_v41.py:351-372（主层 decode prolog，实测在跑）
qr = attn.q_norm(q_a)                       # ← RmsNorm（kernel 1）
q_b_quant, q_b_scale = wq_b.quantize(qr)    # ← DynamicQuant（kernel 2）
...
q = wq_b.matmul(q_b_quant, q_b_scale, bias=attn.wq_b.bias)...
```

**这就是那 30.6 次 `RmsNorm [48,1280]` + 36.7 次 `DynamicQuant [48,1280]` 的来源。**

### 2.4 为什么没融合 —— 注释写明了

`dsa_v41.py:319` 的 docstring：

> *"V4.1 keeps floating-point `qr` for its indexer and has no post-Wq_b Q RMSNorm"*

数据流：

```
dsa_v41.py:381   return q.to(hidden_states.dtype), qr      ← 返回【浮点】qr
dsa_v41.py:590   q, qr = preprocess(...)
dsa_v41.py:600   compressed_indices = self._select_sparse_indices(..., qr, ...)
dsa_v41.py:466   indexer(..., qr, ...)
indexer.py:134   query = self._output(self.wq_b, qr)       ← 用浮点 qr 做线性投影
```

⇒ **indexer 需要一个浮点的 `qr`；而 `npu_rms_norm_dynamic_quant` 只输出 int8 + scale。**
这就是不融合的原因——**不是疏忽**。

---

## 3. 这改变了 F3 的难度与收益

### 3.1 可回收量（这一项）

| 项 | 值 |
|---|---:|
| `RmsNorm [48,1280]` 自身 | 0.415 ms（profile） |
| `DynamicQuant [48,1280]` 自身 | 0.140 ms（profile） |
| 合计自身 | **0.555 ms（profile）= 0.341 ms（真实）** |
| **融合能省的** | 只有**一次 launch + 一次全张量读**（≈3~5 µs/层 × 40 = **0.12~0.20 ms 真实**） |

⚠️ 注意：**不能把 0.555 ms 全算成收益**——融合后仍要做一次 norm + 一次 quant 的数据流，
省的是"多出来的那一次 kernel 启停与读写"。

### 3.2 难度从"低"上调为"中"

要拿到这 0.12~0.20 ms，需要**两处同时改**：

1. `dsa_v41.py` 的 prolog 改成融合调用；
2. **给 indexer 补"量化 qr + pertoken_scale"支持**
   （我们的 dev `dsa_v1.py` 已经有这条路：`self.indexer(..., qr_pertoken_scale=qr_pertoken_scale)`；
   但镜像版 `indexer.py` 只有 244 行、签名里**没有** `qr_pertoken_scale`）。

并且**必须过精度门**（融合算子实测非逐位等价，4~5% 元素差 1 LSB）。

### 3.3 交付形态的限制（重要）

`dsa_v41.py` **只在 CED / PROBE 模式才被挂载**
（`serve_a2.sh:1266-1270`，`_ced_dsa="$PKG/experimental/ced/dsa_v41.py"`）。
⇒ **正常 TP8 交付路径下，这个文件只能是镜像里那份。**
要修就得：

- 要么把它加进常规挂载清单（改 `serve_a2.sh`），
- 要么**重建镜像**。

---

## 4. 对执行计划的更正

| 序 | 原计划 | 更正后 |
|---:|---|---|
| 1 | **F2**（Cast/Fill/ZerosLike，零风险） | **不变，仍为第一步** |
| 2 | **F3**：`RmsNorm`+`DynamicQuant` → 官方融合算子 | ⚠️ **难度上调**：需同时改 `dsa_v41.py`（镜像件）+ `indexer.py`（补量化 qr），且需重建镜像或改挂载清单；**可回收下调到 0.12~0.20 ms** |
| 3 | F7 / F4 / F1 | 不变（待重新核算） |

**建议**：F3 暂缓，**先做 F2**（零风险、纯 Python 侧），
并在同一轮把"**`dsa_v41.py` 常规挂载 + `indexer.py` 量化 qr**"作为 F3 的**前置条件**单独评估。

---

## 5. 另一个必须记录的事实：旧 profile 是三档混合

`FUSION-CANDIDATES` 用的 `armF_r6_base` profile 来自 `run_arm_suite` 的
`prof_conc.sh "$PORT" "1,4,8" 160` —— **一个窗口里混了 conc=1/4/8 三个相位**。

证据（同一窗口内的行数形状）：

| 算子 | 形状 | 次/步 |
|---|---|---:|
| `HcPre` | `48,4,5120` | 61.1 |
| `HcPre` | `42,…` / `36,…` / `30,…` / `40,…` | 8.9 / 4.4 / 5.9 / 4.6 |

⇒ 形状 48/42/36/30 分别是 8/7/6/5 个请求 × 6 token 的**不同并发相位**。

**影响**：`FUSION-CANDIDATES` 里的"每步"数字是**相位平均**，不是单流的。
对"降低单流 ms/step"这个目标，应当**只用 conc=1 相位的步**重算。

**已采取的补救**：在运行中的 tp8k5 上采了一份**纯 conc=1** 的新 profile
（`armHANDOVER_1007_015350/prof/…rank0…`，30 s、1024 prompt、3072 out）。
`/stop_profile` 之后 `ASCEND_PROFILER_OUTPUT/kernel_details.csv` **不会自动生成**，
需要事后跑 `torch_npu.profiler.profiler.analyse()`（实测 >50 min 仍在进行，raw 6.2 GB）。

---

## 6. 复现

```bash
# 1) 三个版本
md5sum patches/files/dsa_v1.py patches/files/draft/dsa_v1.py
ssh a3-21 'sudo -n md5sum ~/cedpd-repo/patches/files/draft/dsa_v1.py'
ssh a3-21 'docker exec dsv41-tp8k5 md5sum /vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v1.py'

# 2) 挂载清单（谁挂谁）
ssh a3-21 'docker inspect dsv41-tp8k5 --format "{{json .Mounts}}" | python3 -m json.tool | grep -B1 -A3 dsa_v1'
grep -n 'draft/dsa_v1.py\|_ced_dsa' scripts/serve_a2.sh

# 3) 主层 prolog（未融合）
ssh a3-21 'docker exec dsv41-tp8k5 sed -n "345,375p" \
  /vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py'

# 4) indexer 只吃浮点 qr
ssh a3-21 'docker exec dsv41-tp8k5 grep -n "qr" \
  /vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/indexer.py'

# 5) profile 形状归属
ssh a3-21 'python3 ~/tmp/shapeagg.py <PROF>/ASCEND_PROFILER_OUTPUT "RmsNorm|DynamicQuant"'
```
