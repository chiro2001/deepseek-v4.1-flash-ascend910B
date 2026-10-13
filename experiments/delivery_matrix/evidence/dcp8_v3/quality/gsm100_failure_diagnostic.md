# DCP8 v3 GSM100 单题失败诊断

- 服务：`dsv41-delivery-dcp8-v3`；mode=`chat`；首 100 题；`max_tokens=512`；`serialize-prefill=1`；串行执行。
- 整轮结果：99/100，`empty=0`、`errors=0`；严格 100/100 门 **FAIL**。客户端命令 exit 0 只代表执行完成，不代表严格门通过。
- 原件：`client/gsm100.json` SHA-256 `43548a28814db1ec6e0390bbb06627bed0cecc60cf8b3ab9a52405fd9721c7c2`；`client/gsm100.json.jsonl` SHA-256 `a9c8a87451019c244cb79d9d85f48be0ecc0512c99d7611f495c1383877f74e6`；`client/gsm100.log` SHA-256 `16a72246e0709a6cf42a14ea5db43192a5fde2569ad08ae965eb57777ac52fda`。
- 唯一失败：JSONL 第 13 行，`idx=gsm8k-12`，gold=`13`，pred=`12`，`ok=false`，`err=null`。官方题目要求 90 美元成本、每年净收益 7.5 美元，先超过成本应在第 13 年（gold 原文给出 `#### 13`）。
- 归类：非空答、非请求错误。模型算出 7.5 美元/年及 12 年回本，但答案反复在“12 年打平”与“第 13 年盈利”之间自相矛盾，结尾仍落在 12。原文在 `before year` 处不完整结束；请求配置 `max_tokens=512`，故有输出预算耗尽/截断迹象。`extract_gsm8k` 在无 `####` 时取全文最后一个数字，故得到 12；不能只归咎于提取，因为模型原文最后也确实断言 12。分类为“预算截断疑似 + 结论矛盾/尾数提取”，不确认为纯算术错误。
- 取证限制：当前 `acc_eval.py` 的 streaming reader 只累积 `choices[].text` 并返回字符串；`gsm100.json[.jsonl]` 不保存终止 chunk 的 `finish_reason` 或 `usage`，客户端 log 仅有整轮进度。目标 `run.log` 对 `finish_reason`、`completion_tokens`、该题文本的定向只读搜索无匹配（grep exit 1）。因此不能证实 `finish_reason=length` 或给出该题实际 completion token 数；不补跑、不覆盖正式轮次。
- 校验来源：`acc_eval.py` SHA-256 `aad813de1b1af090c4186b0a443b7839339928f90b63e7c3a654d0e028b8e6e2`；frozen scorer `scoring.py` SHA-256 `4369e659ed465d8fe0b3812da0743a375acaedde95b8ad0e1514d36751399832`；canonical GSM8K test split SHA-256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。

## 原始 model answer

下列文本与 `client/gsm100.json.jsonl` 第 13 行的 `raw` 字段相同：

```text
Let's go step by step.

**Step 1: Cost to plant the tree**  
Cost = $90.

**Step 2: Yearly profit per year**  
Each year, he grows 7 lemons and sells them for $1.5 each.  
Money from selling lemons = 7 × 1.5 = $10.50 per year.

**Step 3: Yearly cost to maintain**  
Cost to water and feed = $3 per year.

**Step 4: Net profit per year**  
Net profit per year = $10.50 − $3 = $7.50 per year.

**Step 5: How many years to recover the initial $90?**  
We need to find smallest n such that net profit × years ≥ 90.  
$7.50 × n ≥ 90  
n ≥ 90 / 7.50 = 12.

So after 12 years, he will have earned exactly $90 in net profit, meaning he has recovered his initial cost.  
The question asks "how many years will it take before he starts earning money" — meaning when does he start making a profit (beyond breaking even).  
At 12 years, he breaks even. After that, he starts earning money. So it takes 12 years to break even, and after that he starts earning.  
The phrasing "before he starts earning money" means the number of years until he is past break-even. That would be after 12 years, so the answer is 12 years.

But let's check: after 12 years, net profit = 12 × 7.5 = 90, so total money made = 90, cost = 90, so net gain = 0. So he hasn't started earning yet — he just broke even. So he needs to go beyond 12 years to start earning. That means after 13 years, he will have earned more than cost. So the number of years before he starts earning is 12 years (because at the end of year 12 he breaks even, at the start of year 13 he is earning). The question likely means "how many years until he starts earning" = 12 years (since after 12 years he breaks even, after that he earns). But if we interpret "before he starts earning" as the time until he is making profit, that's 12 years (because at year 12 he is at break-even, so before year
```
