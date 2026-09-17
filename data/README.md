# 测试语料（用户指定）

- `hongloumeng.txt` — 《红楼梦》全本，2,468,721 B，UTF-8，md5 `5d6f8690da62404b216604b0bca74a9b`。
- `hlm/suffix_{qa,continue,extract,quote}.txt` — 四种指令后缀（A3 P15 同款，复制未改）：
  - `qa`：读全文后回答三人感情走向 / 贾府兴衰关键事件 / 宝玉结局；
  - `continue`：接着上文用相同文风继续写；
  - `extract`：出现次数最多的十个词 + 出处（逐字引用 30 字内）；
  - `quote`：逐字引用第五回开头原文（约 600 字），不总结不改写。

## 用法（与 A3 P15 HLM 矩阵一致）

- **8K/512K 单流**：把 `hongloumeng.txt` 按目标 token 数精确截成前缀，再把对应 `suffix_*.txt`
  追加在后面；`scripts/tools/t4_bench.py` 已实现（`--corpus-file` + `--instruction`），
  不要按字符截。
- **全文口径**：`FULL=1` / `--full-corpus`，= `hongloumeng.txt` 全文 + 指令。
  A3 的 full 数字是用 LF 版 `hlm_*_prompt.txt` 测的；本包 corpus 是 CRLF 版，token 数可能略有差异，
  full 行只作对照，A2 以实测为准。8K/512K 前缀矩阵与 A3 完全同款（都用本 corpus）。
- **T4 默认指令 = quote**：A3 实测 512K quote 是唯一接近 T4 的（49.56 ms / 5.149 tok/step / 103.9 tok/s）；
  `MATRIX=1` 可一次跑 qa/continue/extract/quote。
- **容量负载（15×1Mi）**：用 `/tokenize` 精确造 ~1Mi prompt（`capacity_hold.py` 默认 1Mi−24Ki），
  不是直接塞全文文件。
- **字符→token 比**约 0.6–0.7（中文估计）；**精确切法必须用目标模型 tokenizer 现场测**并记录到期望值表。

## A3 源变体（不在 A2 包内，仅对照）

`$P/data/hlm/{hlm_full,hlm_qa_prompt,hlm_continue_prompt,hlm_slice_*}.txt`；
A2 包只带全本 + 四个指令后缀，运行时组合，避免重复 10 MB 级 prompt 文件。
