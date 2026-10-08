# Python 侧融合补丁（**可移植版**）

## 为什么需要它（整文件覆盖会出事）

原始做法是**整文件覆盖** 2 个 `.py`。这在「A3 TP8 单实例」上没问题，但在别的形态上会**抹掉别人的定制**：

| 目标形态 | `models/…/model.py` | `attention/dsa_v41.py` | 整文件覆盖的后果 |
|---|---|---|---|
| A3 TP8 单实例 | `deepseek_v4/model.py` | 容器原版 | ✅ 安全 |
| **A3 CED-PD** | `deepseek_v4/model.py` | **CED 定制版**（含 `[CED-SWA-CLIP]` 等 168 行） | ❌ **抹掉 CED 的修复** |
| A2 8×910B3 | `deepseek_v4/model.py` | 容器原版 | ⚠️ 二进制侧另有问题，见下 |

⇒ 用 `apply_lnorm_fuse_portable.py`（**锚点驱动**，只动那两处），兼容所有变体。

## 补丁到底改了什么（共 65 行 diff，两处）

| 文件 | 改动 | 锚点 |
|---|---|---|
| `models/deepseek_v4/model.py` | ① 插 `_lnorm_fuse_on()` 助手 ② 把 `hidden_states = self.input_layernorm(hidden_states)` 换成融合分支 | `class DeepseekV2DecoderLayer(nn.Module):` |
| `attention/dsa_v41.py` | 把 `q_quant, q_scale = wq_a.quantize(hidden_states)` 换成消费融合结果 | 同上一行 |

原文见 `lnorm-model.diff` / `lnorm-dsa_v41.diff`。

## ★ 关键事实：`deepseek_v4/model.py` 是三形态**共用**的

`models/deepseek_v41/model.py` 第 90 行：

```python
from vllm_ascend.models.deepseek_v4.model import (
    AscendDeepseekV4ForCausalLM, AscendDeepseekV4SWACache,
    DeepseekV2DecoderLayer, DeepseekV4Attention, DeepseekV4Model,
)
```

`model_type=deepseek_v41` 时，跑的是 v41 的入口，但**decoder layer 与 attention 来自 `deepseek_v4/model.py`**。
所以：

* 融合补丁打在 **`deepseek_v4/model.py`** —— 三形态都吃得到；
* **不要**去打 `deepseek_v41/model.py`（那是 `patches/files/model.py` 覆盖的那份，里面没有 `DeepseekV2DecoderLayer`）。

## 已验证的锚点兼容性

| 变体 | 文件 | 锚点 |
|---|---|---|
| 基础镜像原版 | `deepseek_v4/model.py` | ✅ class 锚点 1 处 |
| A3 TP8 单实例（tp8k5） | 同上 | ✅ 已实跑，−0.90% 实测 |
| **CED 定制版** | `attention/dsa_v41.py` | ✅ **锚点存在**（第 502–503 行，已核对） |
| 容器原版 | `attention/dsa_v41.py` | ✅ 锚点存在 |

## 用法

```bash
docker cp integration/apply_lnorm_fuse_portable.py <ct>:/tmp/
docker exec <ct> python3 /tmp/apply_lnorm_fuse_portable.py --check    # 先只查锚点
docker exec <ct> python3 /tmp/apply_lnorm_fuse_portable.py            # 应用（幂等，自动备份 .bak-portable-*）
docker exec <ct> python3 /tmp/apply_lnorm_fuse_portable.py --revert   # 回滚
```

⚠️ 应用后要清 `__pycache__`，否则 Python 仍加载旧 `.pyc`：

```bash
docker exec <ct> bash -lc 'find /vllm-workspace/vllm-ascend/vllm_ascend -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; true'
```

## ⛔ A2（8×910B3）还差一步：kernel 没编 910B

本包的 OPP 树**只含 `ascend910_93`（A3 / 910C）的 kernel**：

```
op_impl/ai_core/tbe/kernel/ascend910_93/     ← 只有这一个 SoC
```

而算子 def 声明支持两代：

```cpp
this->AICore().AddConfig("ascend910b");       // A2
this->AICore().AddConfig("ascend910_93");     // A3
```

⇒ 在 A2 上装本包，**算子会找不到 910b 的 kernel 二进制**。
需要按 `REPRODUCE.md` §2 用 `--soc=ascend910b`（或 A2 实际的 SoC 名）重编一份 OPP。
**在补上这份 kernel 之前，A2 不要启用 `V41_LNORM_FUSE=1`。**
