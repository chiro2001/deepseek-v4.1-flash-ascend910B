# 盘点：**已实现、但在启动器里不可达**的 `V41_*` 开关（2026-10-05）

> 动机：本轮找 `V41_IDS64_HOIST` 时发现——代码里**已经写好、注释里带着实测收益**，
> 但 `scripts/serve_a2.sh` 里**没有对应行**，用户没法开。这类"白捡的杠杆"值得系统清一遍。
>
> 方法：抽取 `patches/files/**` 里所有 `environ.get("V41_…")` / `getenv("V41_…")`，
> 再逐个在 `scripts/serve_a2.sh` 里查是否出现。纯静态检查，未占卡。

## 0. 结论

54 个 `V41_*` 开关里，**18 个在启动器里不可达**：

| 类别 | 个数 | 处理 |
|---|---:|---|
| **性能开关（本轮接线）** | **2** | ✅ 已接线（默认 0，等 A/B） |
| 诊断/探针（默认关，靠环境变量或文件触发） | 12 | 不接线（要开时用 `-e` 手工传） |
| 行为/容量旋钮（默认值已在生效） | 4 | 记录，暂不动 |

## 1. 两个已接线（commit `2049615`）

| 开关 | 默认 | 代码位置 | 文档化收益 |
|---|---:|---|---|
| `V41_IDS64_HOIST` | 0 → **可开** | `patches/files/model.py:57,1441` | 每层 `input_ids.to(int64)` 提到每步一次 ⇒ **每步少 39 个算子**。profile 实测该 cast 在图内是 **37 次 / 44.5 µs**（k6full decode 步）；按 §host 账，省下的 host 下发时间可能远大于设备时间 |
| `V41_ENGRAM_PAD_SKIP` | 0 → **可开** | `patches/files/model.py:55,1384` | 只清零"真会被读到"的行；代码注释里写明实测 **pad = 0.13 ms/步**（2048 行 × 2 层 = 50.3 MB memset，~387 GB/s） |

启动器现在会透传：`IDS64_HOIST=1 PAD_SKIP=1 bash scripts/serve_a2.sh`（dry-run 已验证回显）。

## 2. 其余 16 个（不在启动器里）

| 开关 | 代码默认 | 性质 | 备注 |
|---|---:|---|---|
| `V41_ENGRAM_DEVICE_GRAPH` | **1** | 性能（已默认开） | 只要走 device-index 路径就生效；启动器不传也不影响 |
| `V41_ENGRAM_DEVICE_GRAPH_MAX` | 16 | 容量/行为 | 图内 engram 的批上界 |
| `V41_ENGRAM_DEVICE_PAGES` | 8192 | 容量 | device 侧页表大小 |
| `V41_ENGRAM_JIT_PAGES` | 4096 | 容量 | JIT 内核页数上限 |
| `V41_ENGRAM_PG_BUFFER_MB` | 空 | 多机回退 | 只在多机部署用 |
| `V41_ENGRAM_LOCAL_METADATA` (+`_FILE`) | `off` | 行为开关 | 本地元数据缓存（开发用） |
| `V41_ROUTE_PIPE` (+`_FILE`) | `off` | 行为开关 | route 管线（开发用） |
| `V41_ENGRAM_PAGELESS_STRICT` | 0 | fail-closed 用 | =1 恢复旧的 `KeyError` 致命行为（诊断） |
| `V41_BNECK_*`（3 个） | — | 诊断 | bneck 打点频率/模式文件 |
| `V41_DSPARK_SHAPE_PROBE` | 0 | 诊断 | dspark 形状探针 |
| `V41_ENGRAM_ROUTE_PROBE_EVERY` | — | 诊断 | route 探针频率 |
| `V41_META_HOST_SLEEP_FILE` | — | 诊断 | metadata 注入 sleep（默认不注入） |
| `V41_MOE_INVALID_PROBE` | — | 诊断 | MoE 非法值探针 |

## 3. 复现

```bash
cd ~/cedpd-repo
for f in patches/files/*.py patches/files/draft/*.py patches/files/*/*.py; do
  grep -ohE 'environ\.get\("V41_[A-Z0-9_]+"|getenv\("V41_[A-Z0-9_]+"' "$f" 2>/dev/null
done | sed -E 's/.*"(V41_[A-Z0-9_]+)"/\1/' | sort -u > /tmp/all_v41.txt
while read v; do grep -q "$v" scripts/serve_a2.sh || echo "$v"; done < /tmp/all_v41.txt
```

## 4. 下一枪

A/B 两臂（等 8 die 空出来即可跑，命令行已备）：

```bash
# 基线：交付口径（FINAL-R5 已验）
bash ~/tmp/launch_armF.sh
# 臂 H：交付口径 + 两个白捡开关
IDS64_HOIST=1 PAD_SKIP=1 bash ~/tmp/launch_armF.sh   # launch_armF 直接透传这两个 env
```

判据：同 prompt 集、N=1/4/8 各 4 rep 中位；顺带看 `V41_PROFILE` 下
`Cast([6] INT32→INT64)` 与 `Fill` 的次数是否如预期下降。
