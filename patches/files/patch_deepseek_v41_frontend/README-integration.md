# codex / OpenAI Responses API 兼容补丁（`patch_deepseek_v41_frontend/encoding.py`）

日期：2026-09-21　状态：**已在 A3 真机验证通过**（含图片、工具调用、多轮、子代理）

一键使能：

```bash
# 从宿主装进正在跑的容器
docker exec dsv41-a3 bash /opt/dsv41/tools/enable_codex_responses.sh on
docker exec dsv41-a3 bash /opt/dsv41/tools/enable_codex_responses.sh status
```

---

## 0. 一句话

本包的 `tokenizer_mode=deepseek_v41` 走的是 vllm-ascend 的
`vllm_ascend/patch/platform/patch_deepseek_v41_frontend/encoding.py`。
它只认 **chat-completions** 的块词汇表（`text` / `tool_result` / `image_url`），
而 **codex 等 OpenAI 客户端**用的是 **Responses** 词汇表
（`input_text` / `output_text` / `input_image`）。这个词汇表落差造成**三个真实缺陷**，
其中**两个是静默的**：

| # | 缺陷 | 症状 |
|---|---|---|
| **①** | `input_text` 块被渲染成**字面量** `[Unsupported input_text]` | **HTTP 200**，但**用户的话根本没进模型**（模型看到的是那串字面量） |
| **②** | `developer` 角色（codex 放系统指令的地方）要求 content 非空 | **HTTP 500**（`AssertionError: Invalid message for role 'developer'`） |
| **③** | `<｜User｜>` / `<｜Assistant｜>` 等控制 token 能从正文注入 | 实测**能伪造轮次边界**（prompt-injection 通道） |

补丁修掉这三条（**+105 / −5 行**，见同目录 `encoding.py.diff`），
**不改 chat-completions 的老行为**。

---

## 1. 改了什么

基线（镜像内原版）`md5 = d9f5ee08e9219e8d4bf0ee8f54c5ef05`（971 行）
补丁版 `md5 = c20ee3b61bc02a6f4d6f9b9853be5ce4`（1071 行）
目标路径 `/vllm-workspace/vllm-ascend/vllm_ascend/patch/platform/patch_deepseek_v41_frontend/encoding.py`

### ① 块类型别名（修 ① 和 ② 的根因）

```python
RESPONSES_BLOCK_TYPE_ALIASES = {
    "input_text":  "text",
    "output_text": "text",
    "input_image": "image_url",   # 载荷也在 image_url 里，_extract_image 已支持
}
```

归一放在 `_process_image_blocks()` 的**入口**（`for block in blocks:` 之后立刻改写 `block["type"]`）。
选这里的理由：该函数对 `tool_result` 的 `content` 是**递归**的，放在入口 ⇒
**嵌套块自动一起归一**，不必写两遍。

拷贝语义：改写前先 `block = {**block, "type": alias}`，**不动调用方的 dict**（有单测守着）。

### ② `developer` 空内容不再 500（防御性）

```python
elif role == "developer":
    # 原为 assert content, "Invalid message for role ..."
    content_developer = USER_SP_TOKEN + (content if isinstance(content, str) else "")
```

① 修好后 codex 已不会触发它；这条是让**任何**客户端的空 system/developer 消息不再 500。

### ③ 控制 token 转义

新增 `escape_control_tokens()` / `_escape_message_control_tokens()`：

- 覆盖的字段：`content`、`reasoning`、`reasoning_content`、块 `text`、
  `tool_result` 的 `content`（字符串或列表）、`tool_calls[].function.arguments`
- 覆盖的 token：`bos` / `eos` / `<｜User｜>` / `<｜Assistant｜>` / `<｜System｜>` /
  `<｜latest_reminder｜>` / 6 个 `DS_TASK` token
- 手段：在 `<` 之后插 **U+200B 零宽空格**（实测把控制 token 从 **1 个 token 裂成 6 个普通 token**）
- **时机**：在**图片块替换之前** ⇒ 模块自己生成的 `IMAGE_PLACEHOLDER` 不被转义；
  且 `_validate_no_image_sp_tokens()` 仍**先**触发（保持既有契约）

---

## 2. 为什么不能修在 HTTP 层（走过的弯路）

先试了最自然的做法：写个转译代理，把 `input_text` → `text` 再转发。

**失败**——Responses 协议层有严格 pydantic 校验，返回 **239 条 validation errors**：

```
'msg': "Input should be 'input_text'", 'input': 'text'
```

⇒ 协议**要求**块类型就是 `input_text`/`input_image`/`input_file`。
**翻译必须做在校验之后**，也就是 `encoding.py` 里。（脚本 `responses_shim.py` 保留作失败证据。）

---

## 3. 验证（全部实测）

### 3.1 单测

```bash
cd /vllm-workspace/vllm-ascend
python3 -m pytest tests/ut/patch/platform/deepseek_v41/test_encoding.py \
                  tests/ut/patch/platform/deepseek_v41/test_responses_compat.py -q   # 51 passed
python3 -m pytest tests/ut/patch/platform/deepseek_v41/test_frontend.py -q        # 53 passed（需真 tokenizer）
```

新增 `test_responses_compat.py`（15 个用例）覆盖：Responses 词表、`developer` 空消息、
图片块 → 占位符、**嵌套 `tool_result` 内块也归一**、**不改调用方 dict**、控制 token 转义（含幂等）、
以及**三条回归**（图片占位符仍报错、生成的占位符不被转义、既有 36 项不受影响）。

### 3.2 HTTP 层（直打 `/v1/responses`）

| 用例 | 修复前 | 修复后 |
|---|---|---|
| 纯字符串 `input` | 200 `PONG` | 200 `PONG` |
| **`input_text` 块** | 200 但内容是 `[Unsupported input_text]` | **200 `PONG`** |
| **`developer` 消息** | **500 AssertionError** | **200 `BANANA`** |
| 正文含 `<｜Assistant｜>` | 可注入 | 200（已转义） |
| **图片 `input_image`** | — | **200，正确读出图中文字与颜色** |

> 图片块**必须带 `detail` 字段**：`{"type":"input_image","image_url":"data:...","detail":"auto"}`
> —— 缺了会被 Responses 协议拒掉（400，200+ 条校验错误）。

### 3.3 真实 codex（`codex-cli 0.154.0`）

`~/.codex/config.toml`：

```toml
model = "deepseek-v41"
model_provider = "local-a3"
approval_policy = "never"
sandbox_mode = "danger-full-access"
[model_providers.local-a3]
name = "local-a3-vllm"
base_url = "http://127.0.0.1:8020/v1"
wire_api = "responses"
```

| 测试 | 结果 |
|---|---|
| 单轮文本 | ✅ `PONG-42` |
| **工具调用**（shell） | ✅ `exec /bin/bash -lc 'echo TOOLOK-$((6*7))'` → `TOOLOK-42` |
| **图片**（`view_image`） | ✅ 读出 `PONG-42` + 图中颜色描述 |
| **多轮会话**（`resume`） | ✅ 正确回忆上一轮给的关键词 |
| **子代理**（`multi_agent_v1`） | ✅ `SpawnAgent` → `Wait` → 拿到子代理回复 |

**子代理语义还单独验证过**（抓包 + 用服务端同一个编码器还原 prompt）：
载荷逐字到达、`«SYS»`×1 / `«USER»`×2 / `«ASSISTANT»`×1 结构正确、
正文里的控制 token 已转义；子代理**自己**在回复里指出"那是零宽字符，不会当分隔符"。

---

## 4. 已知边界（**上线前请读**）

1. **需要重启服务才生效**（模块在起服时导入）。改的是**容器可写层**，容器重建/换镜像会丢，重跑脚本即可。
2. **重启前必须确认没有残留进程**（`VLLM::EngineCore` / `VLLM::Worker_*` 的进程名里**没有** `vllm serve`，
   `pkill -f "vllm serve"` 杀不到它们）。残留会持有 206 GiB host 注册 + pinned 内存，
   导致起服卡在 `rtsMallocHost 207001`（**连着起几次都失败**）。最稳：
   ```bash
   docker exec <容器> bash -c 'ps -eo pid,args | grep -E "[V]LLM::|[v]llm serve"'   # 应输出空
   docker stop -t 10 <容器> && docker start <容器>
   ```
3. **`reasoning.encrypted_content` 仍不支持**（vLLM 侧 `responses/utils.py:274` 直接 `raise`）
   ——codex 每轮都带 `include: ["reasoning.encrypted_content"]`，但 vLLM **不产出**它，
   所以当前不触发。**一旦上游开始产出，这条链会 400。**
4. **`developer` 被渲染成 `<｜User｜>` 轮次**（不是 `<｜System｜>`）：真正的系统提示走
   `instructions` → 服务端构造成 `system` 消息（正确）；但 skills / permissions 落在
   `<｜User｜>`，于是 prompt 里会出现**两个连续 user 轮次**。这是**既有设计**（非本补丁引入），
   实测模型表现正常。要更严格可以改映射，但会动既有语义与单测，**未做**。
5. 本补丁改的是 **vllm-ascend 上游代码** ⇒ **天然适合向上游提 PR**（这三条是通用缺陷，
   不只影响我们）。`encoding.py.diff` 就是为此准备的。

---

## 5. 文件清单

| 文件 | 用途 |
|---|---|
| `encoding.py` | 补丁版全文（一键使能复制的就是它） |
| `encoding.py.diff` | 与原版的 `diff -u`（**路径已规范化成 `a/`…`b/`**，可直接给上游） |
| `test_responses_compat.py` | 新增的 15 个单测 |
| `../../tools/enable_codex_responses.sh` | 一键使能脚本（`on` / `off` / `status`，幂等，带备份回滚） |
