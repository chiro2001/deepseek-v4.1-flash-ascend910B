# 修复前（A 臂）的块形态 ↔ 结果配对证据（2026-09-24）

这是 `V41_CED_SWA_CLIP` 修复**之前**在 A3-21 真权重 8+8 上采集的一批请求，
用来给修复单变量 A/B 提供"旧行为"的判据基线。采集时 `dsa_v41.py` 里还没有
裁剪代码。

## 条件

- P：A3-21 chip0–7，真权重 `v41-flat-verify3`，TP8，CDE prefill 角色
- D：chip8–15，TP8，`GRAPH=1 EAGER=0` + prompt-tail eager + metadata inline，
  `MULTISTREAM=0 DSA_OVERLAP=0`
- 请求：同一个 1M 请求（`prompt_tokens=1019846` 交给 D，等价于
  `prompt_tokens=1019847` 的完整请求），`temperature=0`，非流式，串行发送
- 探针：`V41_CED_BLOCK_TRACE=1`（块形态）+ `ced_seq_probe.py`（结果与首 token）

## 结果（`ced_layout_ab_analysis.py` 的输出）

```text
  #            tag  result  first_tok  comp  descents                           g0
  1         lt1_01    PASS      54378     7         0 n=7968 first=1     last=7968
  2         lt2_01    PASS      54378     7         0 n=7968 first=7990  last=15957
  3         lt3_01    PASS      54378     7         0 n=7968 first=15979 last=23946
  4         lt4_01    FAIL          1     1      1853 n=7968 first=23968 last=6137

连续分配（descents==0）：3/3 通过
碎片分配（descents >0）：0/1 通过
判定：碎片组仍有失败 —— 修复未完全生效
```

- 前三个请求的 g0 块列表**连续**（`descents=0`），全部答对 `RB9N-6014`，
  首 token id `54378`（"RB"），completion 7 token。
- 第四个请求的 g0 跨过池尾回绕（`first=23968 last=6137 descents=1853`），
  首 token 直接是 **id=1（EOS）**、completion 1 token、content 为 `null`。

这就是根因文档里那条关联的**原始配对证据**：失败与"块分配碎片化"一一对应，
与请求序号无关（第 3 个是连续的、第 4 个才碎片）。

## 文件

| 文件 | 内容 |
|---|---|
| `block_dump.txt` | D 侧 `[CED-BLOCKS]` 行（已去前缀，含 `descents` / `first` / `last`） |
| `lt{1,2,3,4}_01.summary.json` | `ced_seq_probe.py` 的逐请求结果（content / token_ids / usage / wall） |
| `verdict.json` | `ced_layout_ab_analysis.py --json` 的完整输出 |
| `SHA256SUMS` | 上述文件的校验和 |

复现分析：

```bash
python3 tools/ced_layout_ab_analysis.py \
  --probe-dir evidence/ced_layout_ab_prefix_20260924 \
  --serve-log evidence/ced_layout_ab_prefix_20260924/block_dump.txt
```

（`verdict.json` 是在本目录内就地生成的，所以 `--probe-dir` 直接指向本目录即可；
注意工具要求块形态行数与探针数**相等**，不等时默认拒绝执行，避免静默错配。）

## 边界

- 这是**修复前**的观测，因此它证明"修复前碎片必失败"，**不**证明修复有效。
  修复后的判据是：碎片形态 ≥10 次全过、连续形态不退化、仍为 `GRAPH=1 EAGER=0`。
- `lt4` 只失败了一次；"碎片 ⇒ 必失败"的强度来自与 `descents` 的**跨批一致性**
  （1M 第 3/7 次、900K 第 3/7 次、144K 8/8 通过），见
  [`../../docs/CED-D-1M-LAYOUT-BUG-20260924.md`](../../docs/CED-D-1M-LAYOUT-BUG-20260924.md)。
- tiny 侧的同类 A/B 工具与产物见
  [`../ced_swa_clip_tiny_ab_20260924/`](../ced_swa_clip_tiny_ab_20260924/)。
