# GSM 原评测器旁路取证

旧DCP GSM100的原始JSON没有finish_reason/usage，不能确认错误题是否因预算
截断。新`quality_wire_capture.py`不修改冻结`acc_eval.py`，只包装其urlopen响应，
将它实际消费的字节原样交还，另写每HTTP attempt的JSONL；不额外发请求，不改变
8shot、chat、512预算、抽题、评分、原JSON或原重试政策。

```bash
python3 -u quality_wire_capture.py \
  --evaluator /path/to/frozen/tests/acc_eval.py \
  --evaluator-sha256 aad813de1b1af090c4186b0a443b7839339928f90b63e7c3a654d0e028b8e6e2 \
  --wire-output /new/job/gsm100.wire.jsonl \
  -- <原评测全部CLI参数>
```

sidecar拒绝覆盖已有文件。每条包含prompt UTF8 SHA、request-body SHA、响应id、
消费字节数/SHA、finish_reason、usage与是否消费[DONE]；open失败也独立记录。
不完整消费不能冒充完整响应，缺usage/finish_reason继续保留null/空列表。
HTTP attempt不是评测题号：重试可能有多个同prompt SHA，不能以sidecar条数
替代GSM题数或只挑其中正确的响应。正式评分仍以原JSON/冻结strict scorer为准。

4项CPU正负控通过：流式字节原样/只close记一次、直接执行原complete函数AST的
流式与非流式/空答路径、部分流不能伪造完成、异常透明与无关请求不捕获。
尚未随新GSM100实机执行。它不是纯设备计时工具，不用于地火/编程性能窗口。
