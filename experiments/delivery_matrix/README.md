# 三形态交付与内存候选实验

此目录包含 A3 模拟 A2 的 TP8、CED-PD 和 TP8+DCP8 的启动、计数、评分与证据代码。
当前三形态主目标已由用户暂停；详见 [STATUS.md](STATUS.md) 和 [NEXT.md](NEXT.md)。
历史启动配方不是恢复运行的授权。

## 已测与未测的边界

- A2 模拟 v4：地火 2K/32K/147456、编程 8 题/128 用例及精度门已实测；指标见仓根 README。
- DCP v3：历史性能已测，但严格 GSM 门为 99/100，不能标精度达标。
- CED：旧 canary 在实际 KV 传输后的 D 侧重放失败；新修正仍待设备回归。
- DCP v6、共享 Engram mmap 接线、内容唯一 KV 池：候选或 CPU 原语，不是实机交付结果。
  共享 mmap 的 PSS/完整精度验证、单侧 TP8 KV 池 IPC/DMA/容量兑现尚未完成。

## CPU 回归

从仓根运行（不启动服务、不初始化 NPU）：

```bash
mkdir -p "$HOME/tmp/dsv41-delivery-cpu-tests"
TMPDIR="$HOME/tmp/dsv41-delivery-cpu-tests" \
  python3 -m unittest discover -s experiments/delivery_matrix -p 'test_*.py'
```

2026-10-13 最新复验：151 项 unittest 通过；显式把本目录加入 PYTHONPATH 后 pytest
全目录 172 项、64 subtests 通过。此命令使用目录发现模式；部分历史测试共享
同目录的 stub，因此不能简单改用 `-t .` 而不同时调整导入方式。

## 证据与发布范围

Git 保存源码、测试、文档和文本收据；本地压缩包、目标文件/可执行探针、运行 PID、
压缩原始请求以及外部测试语料/图片不纳入此次代码 push，均保留原位置，不删除或迁移。
收据中的旧路径与 SHA 是历史证据锚点，不代表所有原件均随 Git clone 提供。
独立算子重验位于 `../operator_stack/evidence/operator_revalidation_20261013/`，
已包含历史分层、当前2K×8校准及独立raw审计。新增 frozen-input CLI 只回放已pin的输入消息，
不继承旧计时/输出/质量；其manifest/SID/配置指纹失败关闭门在所有请求前执行。
