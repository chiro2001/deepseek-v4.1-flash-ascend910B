# 在 CED 里启用 DSpark：可行性分析与实施路径（2026-09-26）

> 前置：CED-PD 的验收目标**不含** DSpark（目标只要求"可保持 BF16 KV"）。
> 本文是**扩展分析**，不是已完成工作。结论与证据分开写，未验证的都标出来。

## 0. 结论速览

| 问题 | 结论 | 强度 |
|---|---|---|
| **P 侧能不能带 DSpark** | **不能，且是架构性的**：DSpark 从目标层 **37/38/39** 取残差，而 CED 的 P **在第 20 层 break** | 强（代码逐行核实） |
| **那 P 是不是必须带** | **不是**。P 与 D 是**两个独立引擎实例**，`--speculative-config` 是 per-instance 的；P 是纯生产者、响应被丢弃 | 强 |
| **D 侧能不能带 DSpark** | **有可能**，但要动 4 处代码 + 1 个未验证的前提 | 中（机制自洽，未实测） |
| **最关键的未验证点** | D 的 128-token 重放产生的 target hidden states，**是否真的会喂给 draft 的 KV 预填充** | 未验证 |
| **成本估计** | 4 处代码改动 + 2 次真机实验；不涉及算子/内核改动 | 中 |

⇒ **路线是"P 保持 SPEC=0，D 开 SPEC=1"**，而不是"两边都开"。

## 1. 为什么 P 带不了 DSpark（架构性，不是配置问题）

三处代码构成一条闭环：

**① 引擎侧**：只要 `speculative_config.method` 是 `dspark`，就**强制**要求模型返回
辅助隐状态 —— `vllm/v1/worker/gpu/model_runner.py:208`：

```python
if self.speculative_config.method in ("eagle3", "dflash", "dspark"):
    # Drafting may require auxiliary hidden states from target model outputs
    self.use_aux_hidden_state_outputs = True
```

**② 而 runner 会无条件解包** —— 同文件 `:4461`（`gpu/cudagraph_utils.py:539` 同）：

```python
if self.use_aux_hidden_state_outputs:
    hidden_states, aux_hidden_states = model_output    # ← 无条件解包两个
else:
    hidden_states = model_output
```

**③ 模型的 return 是"空就只返回 tensor"** —— `patches/files/model.py:1508`：

```python
if aux_hidden_states:
    return hidden_states, aux_hidden_states
return hidden_states
```

而辅助隐状态是在层循环里按 `aux_hidden_state_layers` 收集的（`:1435`）：

```python
for layer in self.layers:
    ...
    if self._ced_prefill_only and layer.layer_idx == 20:
        layer.write_global_source_from_encoder(hidden_states, pre_mix)
        break                      # ★ CED 的 P 在这里就结束了
    ...
    # DSpark consumes the residual stream entering its configured target layers.
    if layer.layer_idx + 1 in self.aux_hidden_state_layers:
        aux_hidden_states.append(hidden_states.mean(dim=1))
```

`aux_hidden_state_layers` 来自 `dspark_target_layer_ids=[37, 38, 39]`
（权重目录 `config.json` 实测值）。P 在第 20 层 break ⇒ `aux_hidden_states` 永远是 `[]`
⇒ ③ 走"只返回 tensor"⇒ ② 拿一个 tensor 去解包两个变量 ⇒ 报错。

**这条无法通过配置绕过**：P 只有前 20 层的权重与计算，37/38/39 层的残差在 P 上
物理上不存在。

## 2. 但 P 不需要带 DSpark（这是本分析的关键开口）

P 与 D 是 `scripts/serve_a3_ced_pd.sh` 启动的**两个独立容器**，各自有独立的
`--speculative-config`。P 的角色是"内部生产者"（`model.py:905`：
`this role must only serve that internal transfer request`），
**它的响应被代理丢弃**，只有它写出的 KV 有用。

⇒ **目标是：P 保持 `SPEC=0`，D 用 `SPEC=1 SP_TOKENS=7`。**

于是问题从"两边都要改"收缩成"**让 D 的 DSpark 在 CED 的有界重放下正确工作**"。

## 3. D 侧要动的四处代码（都定位到行）

### 3.1 group 契约：从硬编码 12 改成显式的 12(P) / 13(D)

DSpark 会多出一个 **G12**：三个草稿层 `mtp.0/1/2.self_attn.swa_cache`，
各 `[N,128,1,512]` BF16、131072 B 步长、无填充、**自己的块表**（见模型 README
"DSpark in the same four slots"）。所以 D 是 **13 组**，P 仍是 12 组。

当前连接器有三处硬编码 12：

| 行 | 代码 | 问题 |
|---|---|---|
| `:2104` | `if len(meta.remote_block_ids) != 12 or len(meta.local_block_ids) != 12` | D 会是 13 |
| `:2129` | `for group_idx in range(7, 12)` | 漏掉 G12 |
| `:1497` | `assert tuple(ced_missing_swa_groups) == (7, 8, 9, 10, 11)` | 需确认 G12 不混进来 |

**好消息**：`:1395` 那段用正则 `\.layers\.(\d+)\.` 找层号，而草稿层叫 `mtp.0.*`
**不匹配** ⇒ `layer_indices` 为空 ⇒ 被 `if layer_indices and ...` 跳过 ⇒
**`ced_missing_swa_groups` 仍会是 (7,8,9,10,11)**，`:1497` 那条断言可能原样通过。
这一点**未实测**（要靠起一次 SPEC=1 看日志）。

改法：把两边的组数写成"P 侧 12、D 侧 13"的显式契约，而不是各自硬编码。

### 3.2 D 侧预清零必须覆盖 G12

`experimental/ced/mooncake_hybrid_connector.py:2129` 的清零只做 `range(7, 12)`。
G12 的草稿 SWA 与 G7–G11 是**同一类风险**：P 从不计算它们，
D 的重放只写其中 128 token 覆盖的页，其余页若留着上一次请求的残页，
attention 就会读到不属于本次的数据（这正是 2026-09-24 定位的那一族 bug）。

改法：把 `range(7, 12)` 改成"所有 P 不发送、且 D 需要清空"的组集合
（即 `ced_missing_swa_groups` **加上** G12）。

### 3.3 草稿 SWA 是否也需要 `[CED-SWA-CLIP]`

`experimental/ced/dsa_v41.py` 里的 SWA-clip 修的是"重放首 query 的 128 窗口
回溯到已回收列 ⇒ kernel 读物理块 0"。

**未验证**：草稿层走的是 `AscendDSparkProposer`（继承 `AscendDflashProposer`
→ `AscendEagleProposer`）的注意力路径，**不一定**经过 `dsa_v41.py` 那个
`_native_attention`。如果草稿有自己的 SWA attention 且同样把整行块表 + 完整
`seq_lens` 传给算子，那么**同样的越界读会在草稿路径上重现**，而 clip 不会生效。

这是必须在真机上验的一条（见 §5 实验 B）。

### 3.4 摘掉 `model.py` 的一刀切禁令

`patches/files/model.py:907`：

```python
if ced_role and vllm_config.speculative_config is not None:
    raise ValueError("V41_CED_ROLE=prefill/decode requires SPEC=0 during the replay prototype")
```

这是"整个原型不许开"的一刀切。按 §1/§2 的结论应当改成：

* `prefill` 角色：保留禁令（架构性不可行，见 §1）；
* `decode` 角色：允许，但加显式开关（例如 `V41_CED_ALLOW_DSPARK=1`），
  默认仍然拒绝 —— 与 `PREFIX` 的处理方式一致（默认拒绝、可显式放行、结果标实验臂）。

## 4. 最关键的未验证点：重放产生的 hidden states 会不会喂给草稿 KV

机制上的**利好消息**（来自代码注释与模型 README）：

* `dspark_proposer.py` 的类文档原话：
  "**target hidden states prepopulate draft K/V**, then one anchor-first query
  block emits all speculative tokens."
* 模型 README："DSpark context KV is projected independently for each draft
  layer using the inherited DSV4 SWA backend and the group's own slot mappings.
  **The target exports the incoming residual streams from the checkpoint-selected
  auxiliary layers.**"

也就是说草稿的 KV 不是自己算出来的，而是**从 target 的辅助隐状态投影出来**的。
而 CED 的 D 在重放步里**正好**会跑完 0–39 层、产出最后 128 个位置的隐状态
（含 37/38/39），并且这些会被 runner 存进持久 buffer
（`gpu/cudagraph_utils.py:543`：`self.aux_hidden_states[i][:num_tokens] = aux`）。

**但有一个前提没验证**：proposer 是在**解码步**运行的，而重放步是
**prefill 形状**的。如果 proposer 只在解码步跑，那么：

* 支持的说法：重放步把隐状态存进持久 buffer，下一步 proposer 跑时读得到 ⇒ 可行；
* 反对的说法：草稿需要**它自己**的 128 窗口 K/V，而重放步只写了它覆盖的那些位置；
  如果 proposer 的预填充只看"当前 batch 的 token"，那锚点之前的 127 个位置
  可能没被投影进草稿 KV。

⇒ **这就是 `model.py:906` 那句注释所指的"draft cache 的 D 侧初始化协议"**，
是整个分析里唯一需要真机才能定性的地方。

## 5. 建议的最小实验（按信息量排序，都不改代码）

### 实验 A：先看 `SPEC=1` 在**非 CED 的 D** 上是什么形状（只重启 D，约 6 min）

目的：拿到 DSpark 在**全 40 层 D** 上的 group 数、`ced_missing_swa_groups` 形态、
以及 proposer 的调用时机。判据：

```bash
# 起 D：SPEC=1 SP_TOKENS=7 DRAFT_GRAPH=0，其余同基线
# 看三件事：
grep -a "num KV cache groups\|kv_cache_groups" d/serve.log | head     # 期望 13
grep -a "CED decode: upper SWA groups" d/serve.log                    # 若仍是 (7..11) ⇒ §3.1 的好消息成立
grep -a "prepopulate\|draft K/V\|aux" d/serve.log | head              # proposer 何时跑
```

这一步能把 §3.1 的"未实测"变成确定，且**不碰 CED**。

### 实验 B：草稿 SWA 的注意力路径是否也需要 clip（复用实验 A 的实例）

发一条**带针**的长请求（≥144K），看草稿路径有没有和当年一样的第一发散点。
如果草稿走的确不是 `dsa_v41.py`，那么 §3.3 就是真需求。

### 实验 C（只有 A、B 都乐观时才做）：CED 的 D 开 SPEC=1

需要先落 §3.1/3.2 的代码改动，否则会在连接器的组数校验上直接崩
（和缓存命中那轮一模一样的坑）。

## 6. 与已完成的 CED 工作的关系

| 已有的东西 | 对 DSpark 的价值 |
|---|---|
| `[CED-32BIT-GUARD]`（num_blocks ≤ 29076） | G12 让每块页需求 +1 组，但 README 说"G12 只加一组 SWA 页需求"、540928 B/块不变 ⇒ **守卫仍然适用**，但要重新核算 13 组下的几何 |
| `[CED-SWA-CLIP]` | **可能**要复制到草稿路径（§3.3 未验证） |
| D 侧预清零 | **必须**扩展到 G12（§3.2） |
| 缓存命中的两条修复（P 侧对齐回退、D 侧空接收） | 与 DSpark 正交，但两条都在 P/D 的调度器上；D 开 SPEC=1 后要重跑这两组用例确认没回归 |
| 验收 runner / bench / profiler 工具 | **直接复用**，无需改 |

## 7. 风险与不建议的做法

* **不要**试图让 P 也带 DSpark（§1 的架构阻断）。
* **不要**跳过 §3.1/3.2 直接起 CED+SPEC=1：会在组数校验上崩，
  和缓存命中第一轮一样（那次是 D 的 `expected 12 KV cache groups`）。
* **不要**把 DSpark 的性能收益与已验收的 CED 数字混在一起：
  现在的 decode 25.3–37.9 ms/step、吞吐 41.4–114.5 tok/s 都是
  **`SPEC=0` 纯自回归**的数字（`接受长度=0.00` 就是证据）。
* `SP_TOKENS` 必须留在 **1..31**：模型 README 明确写
  "The 32-row FP32 target ring requires 1..31 speculative tokens...
  S=32 can overwrite the anchor after complete rejection and is rejected at
  initialization"。当前用 7，安全。
* 草稿入图（`DRAFT_GRAPH=1`）**不要**开：README 说
  "The V1 DSpark proposer runs eagerly. Draft graph capture is not enabled."，
  而 CED 也要求 `DRAFT_GRAPH=0`，两边一致。

## 8. 一句话总结

**不要把 DSpark 想成"给 CED 加个开关"**：P 侧是架构性不可行（要在第 20 层
拿到第 37/38/39 层的残差），真正的路径是 **P 保持 SPEC=0、D 开 SPEC=1**，
然后补 D 侧的四处（group 契约、G12 清零、草稿 SWA 的 clip、摘掉一刀切禁令），
其中**唯一需要真机定性**的是"重放产出的 hidden states 是否真的喂给草稿 KV"。
先做实验 A（只重启 D、不碰 CED），成本 6 分钟，信息量最大。
