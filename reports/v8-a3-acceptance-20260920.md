# v8 A3 验收记录（2026-09-20）

> **目的**：证明 v8 增量（Engram device-index 入图 + 两个回归撤销 + 起服修复）
> 在**外部用户会走的那条路径**上能起、能跑、精度不回退。
>
> **路径**：**官方镜像** `quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3`
> + `PATCH_MODE=mount`（挂 `patches/files/`）+ `serve_a3.sh` 默认值。
> 这与"我们自己的烘焙镜像"是两条不同的路 —— 之前只在 baked 路径上验证过。

## 0. 结论（一句话）

**全过。** 起服 15 分钟、五项自检全绿、Vision 23/23、GSM8K-200 = 199/200、
多轮插针/并发逐项/长短交错三块全 PASS。

## 1. 起服与自检

| 项 | 值 | 判据 |
|---|---|---|
| run | `results/a2_20260920_053501` | — |
| 起服命令 | `DEVS="8 9 10 11 12 13 14 15" CPU_BIND=0 MODEL=... bash scripts/serve_a3.sh` | 官方镜像 + mount |
| 起服耗时 | **约 15 分钟**（`init engine ... took 417.50 s` + 权重加载/捕获） | 见 §4 的 `migratepages` 说明 |
| `static_kernel` 静默降级 | **0 命中**（`grep -ac "static_kernel.py:650"`） | 必须 0 |
| `admission gate` | **`APPLIED(live_hits=15)`** | mount 模式下现场打的，见 CHANGELOG §4.5 |
| Engram device-index | **探测通过**（`[DEVICE-INDEX] 能力探测通过`） | `ENGRAM_DEVICE_INDEX=auto` 走 device 路径 |
| KV 池 | **2,821,337 tokens** | 门槛 ≥ 2,800,000 |

> ⚠️ 起服**不能**用默认的 `CPU_BIND=1`：`enable_cpu_binding` 的
> `migratepages` 会在本模型上永久卡住（实测 1201 s、97% CPU、`numa_maps` 零进展）。
> 详见 CHANGELOG §4.6 与 README §2.6。**A3 推荐 `CPU_BIND=0`。**

## 2. 精度回归

| 测试 | 结果 | 历史区间 | 判据 |
|---|---|---|---|
| **Vision**（23 例图文问答） | **23/23** | 23/23 | ≥19/23 为达标 |
| **GSM8K-200**（chat 口径，`conc=4`） | **199/200（99.5%）** | 197–199 | ≥197 同量级 |
| **多轮插针**（8 轮增长历史 + 3 个随机 10 位串） | **轮内 7/7、全长 3/3** | 同 | 100% |
| **并发 batch**（16 题，`conc=1` vs `conc=8` 逐项比对） | 16/16 vs 16/16，**不一致 0** | 同 | 0 |
| **长短交错**（1×131072 + 2 s 后 6 条短的） | 6/6 vs 6/6，**不一致 0** | 同 | 0 |

机器可读判据（`results/a2_20260920_053501/multibatch.json`）：

```json
{"A_needle": "PASS", "A_recall": "PASS", "B_itemwise": "PASS", "C_short_vs_long": "PASS"}
```

原始文件（都在同一个 run 目录下）：
`vision.json` / `vision.json.raw.json`、`gsm8k.json` / `gsm8k.json.raw.json.jsonl`、
`multibatch.json` / `multibatch.log`、`serve.log`、`serve_cmd.txt`。

> GSM8K 用的解释器是 `~/venvs/lmeval311/bin/python3`（需要 `datasets` 包；
> 系统 python 没有，会报 `ModuleNotFoundError: No module named 'datasets'`）。

## 3. 复现命令

```bash
# 起服（官方镜像 + mount；注意 CPU_BIND=0）
cd ~/projects/dsv41-release
DEVS="8 9 10 11 12 13 14 15" CPU_BIND=0 \
  MODEL=/path/to/v41-w4a8-engram-dr-vision-qrot-mtpq \
  bash scripts/serve_a3.sh

# 五项自检
R=$(ls -td results/a2_20260920_* | head -1)
grep -ac "static_kernel.py:650" $R/serve.log          # 0
grep -c "DEVICE-INDEX] 能力探测通过" $R/serve.log      # 1
grep -o "GPU KV cache size: [0-9,]* tokens" $R/serve.log | tail -1
grep -E "PATCH_MODE|ADMISSION" $R/serve_cmd.txt

# 精度
python3 tests/t_vision.py --server http://127.0.0.1:8020 \
  --images-dir /home/user/models/DeepSeek-V4.1-Flash/inference/examples/images \
  --out $R/vision.json
~/venvs/lmeval311/bin/python3 tests/t_gsm8k.py --base http://127.0.0.1:8020 \
  --limit 200 --conc 4 --enc-dir /home/user/models/DeepSeek-V4.1-Flash/encoding \
  --out $R/gsm8k.json
python3 tests/multibatch/multibatch_gate.py --base http://127.0.0.1:8020 \
  --out $R/multibatch.json --rounds 8 --conc 8 --long-ctx 131072
```

## 4. 这轮暴露的两个真问题（都已修/已给逃生口）

1. **mount 模式丢掉 `admission gate`**（静默失效）→ 起容器后现场 `git apply` +
   断言 live tree 命中。见 CHANGELOG §4.5。
2. **内部绑核的 `migratepages` 会永久卡住起服** → `CPU_BIND=0` 逃生口 + 判据
   （看 `numa_maps` 的目标节点页数是否在涨）。见 CHANGELOG §4.6。

## 5. 尚未覆盖（诚实边界）

1. **A2（910B3）真机**：本机没有 A2 访问权限，A2 的结论全部是资料推断 + 一颗
   **能力探针**（`tools/run_probe_hostmap.sh`）。**A2 上任一结论以探针退出码为准**
   （0=支持 device-index / 3=不支持 / 1=探测本身出错）。
2. **内部绑核（`CPU_BIND=1`）的性能**：这轮为了绕开卡死用的是 `CPU_BIND=0`，
   两者在**性能**上的差异没有在本轮 A/B 里量过（历史 693 s 起服那次用的是 1）。
3. **PGO**：A3 镜像的 `libpython` md5 与 A2 的不同，按设计自动降级 `PYTHON_PGO=0`；
   本轮 A3 数据全部**不含 PGO**。
