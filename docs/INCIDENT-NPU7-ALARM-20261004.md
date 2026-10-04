# 事故记录：a3-21 NPU 7（dies 14/15）进入 Alarm，服务无法启动（2026-10-04 16:40）

## 现象

* `npu-smi info -t health -i 7`：**Chip 0 / Chip 1 均为 Alarm**（MCU OK）。
* 两个后续服务实例都**卡在启动的同一阶段**：日志停在 Gloo init 之后，EngineCore 每 60s 打
  `No available shared memory broadcast block found in 60 seconds ... some processes are hanging`，
  worker 进程全部存在但不推进 ⇒ `/health` 恒 000。
* 设备本身读数正常：HBM 4%、Aicore 0%、49°C、167W、无进程占用（仅驱动基线 ~3GB）。

## 时间线（本次会话）

| 时间 | 事件 |
|---|---|
| ~15:12 | armC kernel 实验第 1 次启动（只设 `ASCEND_CUSTOM_OPP_PATH`）→ 服务**正常起来**（但 armC 未生效） |
| ~15:34 | armC 第 2 次启动（`-v` ro 覆盖挂载镜像 vendor 路径）→ 启动期 **aicore exception**：`IndexCheck` 507015 |
| ~16:20 | armC 第 3 次启动（`cp -a` 覆盖镜像 vendor，rw）→ 内核确实进容器（md5 = armC），但服务**挂死** |
| ~16:35 | 移除挂死容器，改回 **stock vendor** 重启 → **同样挂死** |
| ~16:39 | 发现 NPU 7 Chip0/Chip1 = Alarm |

## 已尝试的恢复手段（均未成功）

| 手段 | 结果 |
|---|---|
| 移除容器并等待 30s | Alarm 不变 |
| `npu-smi set -t errcount-clear -i 7` | `This device does not support` |
| `npu-smi set -t clear-pcie-err -i 7 -c 0` | `This device does not support` |
| `npu-smi set -t reset -i 7 -c 0 -d 1`（确认 y） | `Failed to reset server`（提示 "It will reboot all devices"） |
| `npu-smi set -t reset -i 7 -c 0 -m 1`（in-band，确认 y） | 同上 `Failed to reset server` |

## 归因（诚实标注）

* **时间相关**：Alarm 出现在 armC 内核被真正加载（第 2、3 次）之后；第 1 次（内核未生效）之前
  的同配置服务是正常的。
* **未证实**：我没有在本次会话早期对 NPU 7 单独查过 health，所以**不能排除** Alarm 是实验
  之前就存在的（例如前一位使用者留下的 sticky 标记）。
* 有一个**旁证支持"本实验导致的"**：第 2 次启动的 aicore exception（IndexCheck 507015）是
  一次真实的内核异常，这类异常在 Ascend 上确实会把对应 die 置为 Alarm 直到复位。

## 需要的处置（需用户决策）

`npu-smi set -t reset` 是唯一已知的清 Alarm 路径，但它会提示
**"It will reboot all devices, do you want to continue reboot?"** —— 在共享机上这**可能影响
NPU 0–6 上其他租户的任务**，超出我可自行决定的范围，故停在此处上报。

可选路径：
1. 由用户/机房确认该 reset 只影响 card 7（我们自己的 dies 14/15）后执行；
2. 走带外（BMC）单卡复位；
3. 或先按"设备可用性未定"处理：把交付实例缩到 dies 8–13（TP6 对本模型结构性非法，
   `num_attention_heads=64` 不能被 6 整除 ⇒ 实际不可行）⇒ **在本机恢复前，交付实例无法启动**。

## 教训（写给后来者）

* **给"已有算子"换 kernel 时，必须先在目标容器里验证内核真的被加载**（本次第 1 次就是
  静默未生效）；验证手段：profile 里该算子的 `aic_scalar` / `Duration` 是否按预期变化，
  或直接对比容器内 vendor 路径的 `.o` md5。
* 覆盖挂载（`:ro`）与复制覆盖是两种不同风险：前者在本次直接触发 aicore exception。
* **先做单算子层面的"图捕获下"验证**，再上整机服务 —— 本次 armC 只在 eager 下验过。
