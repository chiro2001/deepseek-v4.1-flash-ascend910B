# K5 容量线设备阶段：复现说明

2026-10-12 在 a3-21 的 **physical4（NPU 2 chip 0）** 完成 K5 容量线（target36/E48、draft15/E16）
的设备数值门与最小基准。容器 `dsv41-real-stack-tp8-20261010-a321`（CANN 9.1.0，`/usr/local/Ascend/cann-9.1.0`），
每个容器内命令都以 `-e ASCEND_RT_VISIBLE_DEVICES=4 -e ASCEND_VISIBLE_DEVICES=4` 显式声明可见设备，
bridge 内部另有单 visible + `rtGetDeviceIndexByPhyId(expected)` + logical0 三重守卫。不含 reset、清缓存、
停止他人进程或跨 chip 行为。

## 目录

| 文件 | 作用 |
|---|---|
| `capacity_device_prepare.py` | 容器内 CPU-only：复制原生源、套用冻结 M1 稀疏插入 + 容量几何、生成 planner/OPC 输入、ccec 编译 planner、OPC 编译 candidate/native、从新 kernel metadata 读真实 workspace 字节 |
| `capacity_bridge_build.py` | 生成并编译容量 bridge（`bridge_factory` + 三处受断言保护的追加修改：physical 白名单加 4、修正误导性 stage 标签、追加 planner-only 与定时入口） |
| `runtime_capacity_gate.py` | 真机数值门 + 最小基准；`K5_CAPACITY_PHASE=timing_only` 时只跑基准并引用数值窗口证据 |
| `run_gate_device.py` | 容器内一次性 runner（拒绝同根重试、校验 preflight 新鲜度、收 API 事件计数） |
| `run_host_window_p4.py` | 宿主机 wrapper：实时 smi/health/占用快照 → 启动 → 释放快照 |
| `collect_device_evidence.py` | 把两个窗口根压缩成结果 JSON（大文件留远端并记 SHA） |
| `receipts/` | v4 数值窗口的编译/bridge receipt、失败尝试历史、tiling 几何解码 |

## 命令（容器内除标注外均以空 visible guard 运行）

```bash
ROOT=/work/gmm_lane_sparse_msd_k5_capacity_p4_20261012_v4
for fam in target draft; do
  docker exec -e ASCEND_RT_VISIBLE_DEVICES= -e ASCEND_VISIBLE_DEVICES= -w $ROOT <container> \
    python3 k5_capacity_numeric_cpu/device/capacity_device_prepare.py --root $ROOT/build/$fam --family $fam
  docker exec -e ASCEND_RT_VISIBLE_DEVICES= -e ASCEND_VISIBLE_DEVICES= -w $ROOT <container> \
    python3 k5_capacity_numeric_cpu/device/capacity_bridge_build.py --root $ROOT/build/$fam --family $fam
done
# 宿主机：一次性真机窗口（自采实时回执）
python3 $ROOT/k5_capacity_numeric_cpu/device/run_host_window_p4.py <root-name>
# 单 phase 基准窗口
python3 .../run_host_window_p4.py <timing-root> timing_only <numeric-root>
```

编译命令原文见 `receipts/prepare_receipt_{target,draft}.json` 的
`planner_cpu_rebuild.command` 与 `compile_{candidate,native}.command`；bridge 链接命令见
`receipts/bridge_receipt_{target,draft}.json` 的 `command`。

## 口径

- admission / performance / formal 标志全部 false；基准数字是同实例 operator-window 诊断值，
  窗口内可能含 host 提交造成的空闲，不是纯 core 时间，也不是收益声明。
- 数值门通过只覆盖本候选在固定输入下的等价性；稀疏分支的“实际进入”无法由输出区分
  （合法输入下 sparse 与 dense 按设计产生相同行），仅有编译期与计时侧证据。
- `weights_cpu` 的载荷路径在本阶段首次执行（容器内 CPU），target 侧与 M1 冻结物理 NZ 逐字节一致。
