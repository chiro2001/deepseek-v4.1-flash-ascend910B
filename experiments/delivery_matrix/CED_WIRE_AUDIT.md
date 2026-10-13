# CED 冻结 proxy 的独立原始线协议取证器

`ced_wire_audit.py` 不修改原 `ced_proxy.py` 或 P/D SDK。
原 proxy 必须匹配 SHA
`813c5a0e1a8b7e7988ee9f02fd3a83d8fa38ede9a88713a9c65e5cf0cf99eb33`，
该 SHA 也是 CED v4 冻结包的实际记录。wrapper 当前 SHA
`82873d5b2c454d50580b80c39a46970f69e18c7b15118fa8f7dcec78d098c0d1`。

记录实际出站 P 请求、P 非流式响应、实际出站 D 请求的原始 HTTP body，
包含完整 `kv_transfer_params` 和原始 block ID 列表。原始/压缩字节的 size/SHA
分别保存；按 X-Request-Id、URL、model 和单调时间关联。同请求三件 body 未重新
序列化成所谓“原始数据”；JSON metadata 只引用原字节文件。

不新增请求、重试或 header；不读取 D 响应（包括非流式 D）、不迭代/计数 SSE，
不替换原模型身份和计数器。取证写入全部发生在提交 D 请求之前，属于 TTFT；
不能把开启取证的 TTFT 冒充无取证的严格单变量性能。正式 decode 窗口不插入
取证 hook，worker prefill counter 的定义不变。

首次 `--audit-output` 要求目录不存在，文件以 O_EXCL/0600 创建；不覆盖旧样本。
caller request ID 不用作文件名。每个 raw body 上限8MiB，每个 gzip/metadata
文件严格小于1MB；超限/文件错误 fail closed、保留已有证据，不截断或重新发请求。
仅 P 已是 stream=false，P body hook 的 aread 与原 proxy 的 post 缓冲行为一致。

```bash
python3 ced_wire_audit.py \
  --proxy-source /work/delivery_matrix_ced_20261013_v4/delivery_matrix/ced_proxy.py \
  --audit-output /home/l00886679/tmp/dsv41-ced-runtime-20261013/jobs/proxy5/wire \
  --prefill-url http://127.0.0.1:18990 \
  --decode-url http://127.0.0.1:18991 \
  --served-model dsv41-ced-pd-v4 --port 18992
```

上面只是配方；proxy 启动仍须经 P/D 新实例 ready/来源门与 root 放行。
部署前需以实际环境验证路径/字节 SHA，不替换旧 frozen 目录中的文件。

18 项 CPU 测试包含全部12项原 proxy 规则的带 hook 重验、精确动态 transfer
字节关联、D stream 未消费、非流式 D 未取证、目录/路径/重复安装/源码漂移/体积负控
和实际 lifespan 安装/关闭。delivery 顶层129项通过，尚无该 wrapper 的实机回执。
测试源码 SHA `c96f83d605bced641a6cfc5bd025d4db28dec46969f18c12eeb93af2e8fbe038`。
