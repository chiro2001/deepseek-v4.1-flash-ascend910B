# DSpark 完整轮次计时与43层实机绑定修复

2026-10-12。正式完整真实权重 TP8、包含 DSpark 全部执行开销的 ≤20 ms/实际 verifier step 目标仍 active。当前真实 HTTP 集成基线仍为 **57.581661 ms/step、A=2.965116、51.494108 tok/s，clean7/8**；尚无正式20ms、完整消费者、通用质量或最优部署通过。

## 远端原始 HTTP 全轮次核验

在原始数据所在 host，以 `python -S` 与原冻结 v4 计数模块独立重算 22,145,365B acceptance。10请求（2warmup/8measured）/80份rank窗口的 **6648个rank-round原payload**，包括所有中间完整token/proposal/接受与首末fence/join，均通过。各原HTTP响应SHA、原core绑定与计数三元组一致。后端runtime imports=[]、NPU_calls=0、HTTP_requests=0，没有重跑请求。

原数据SHA：`1ea53d1bf189b8444dd1f4fdd7856b7bd5ef8d28a7fc8b3d577e20935f492a12`；计数源SHA：`da845a5891557f21e80eac06d96aec1955263d1ae93afbc4d84e86dd9585dbec`。完整大原文件继续留远端，SSH仅取回小型验证回执。

8 measured合计688 actual rounds、2040 decoded tokens、39.61618308s。选择每请求初始边界最早的rank，以下互不重叠host区间逐纳秒精确重建完整客户端decode窗口：

| host区间 | 合计秒 | ms/实际round | decode占比 |
|---|---:|---:|---:|
| initial draft及首调度 | 0.14525071 | 0.211120 | 0.37% |
| execute开始→bookkeeping | 24.38666496 | 35.445734 | 61.56% |
| bookkeeping→sample返回 | 11.59303214 | 16.850337 | 29.26% |
| 相邻轮之间 | 1.13159119 | 1.644755 | 2.86% |
| 末sample→客户端read | 2.35964408 | 3.429715 | 5.96% |
| 合计 | **39.61618308** | **57.581661** | **100%** |

这是实际host边界账，不隔离设备kernel或单独draft成本，也不与另一个profile进程相减。后续主模型、草稿、调度候选须在同实例实际配对，兼顾A、tok/s和clean-rate。

## 首个完整权重诊断发现真实对象绑定错误

v10 diagnostic冻结48文件，同容器Torch2.10 CPU16项/17原AST/32轮CPU出口通过，20项未覆盖明确保留。完整v10/v9/v8/prepack overlay 148成员、592596B，SHA `88e6b0d5c2ff286d4c850aae618de82d214e20c31a5ba874493b4e0553c1781c`。root纯prep形成162文件的不可变stack快照，manifest SHA `abba4d19f628c28eb4edbf8e2714a46c1da6c0ad41348fb6da5c8852ac247f98`。

root在启动前校源/config/SHA，fresh physical8–15无进程、健康仅原授权80C98001。正式容器没有继承此前完整Engram/HCCL wrapper环境，launcher v2因此逐SHA读取实际已完成native API job的启动回执，显式恢复其原runtime flags，再叠加冻结audit配置；CPU正例与篡改/冲突两反例通过。没有修改v10冻结或模型数学。

真实job `formal_dspark_full43_diag_v1` 的rank5于加载39.1183GB权重后，在`session.py:66 before_graph`抛错：`AscendDeepseekV41ForConditionalGeneration`没有`.model`。完整视觉架构的实际language decoder通过`language_model.model`持有。原source注册/CPU能力不能替代真实对象入口验证，本job没有完成graph捕获、43实际出口、32轮数值或性能门。

原cleanup回执2853B，SHA `54821a35bfb94fd8f288d002232860dc75c5c6fb7644fe4f10c278673cd95405`：8个child出生PPID均为本EngineCore1495304，8个pidfd hard-kill成功，无身份signal拒绝；但10s内全部terminal=false、exitcodes={}。原result严格passed=false、worker_exit_codes_complete=false、全rank 0 exports，run.exit仍缺失。后续只读确认原core/worker不可见与physical8–15无进程，仅证明当前设备释放；不从不可见推断reaped，更不改判旧job。

独立v11按实际原vl_model源码、Conditional class/global symbol与registered ownership守卫，精确取`language_model.model`。原conditional/vision对象、原forward/数学/ABI与draft.model保持。同版本20 CPU正反门与17原AST通过；56文件冻结，SHA `8f582de3e5c46337c7e002d4b0a1c94a1338acef33980875024f65f1ef752e4d`。20项未覆盖与所有device/full/formal/prepack/performance标志仍false。新窗口必须再次fresh门，实际通过后才能记录对应能力。

## 并行tiny与后续算子准备

同一physical4/PID953926的2K/256 v3继续，当前快照13请求、**5/9数值门通过**。三组OFF A/A、首seed OFF/ON与ON A/A均token/Top5相等、logprob delta0，proposal、43路由、3attention、逻辑cache、全逐步visible cache及parameter seal bitwise通过，unretained_prefix_not_read=true。原5-check文件3856B，SHA `2c482764fe5f853a958d3513d48afbae6af7313efd5bd035a23f6b773ca9e24a`。剩余4门未完成，job非terminal，未授性能或正式资格。

通用质量launcher已冻结21文本，14CPU/59拒绝，通过原FastAPI/AsyncLLM/生命周期/归属与严格GSM100/100、Vision23/23的CPU流水线。实际20assets、90shards、4984 MTP+vision headers只读核验；未启动新HTTP/NPU/服务/真实signal。manifest SHA `b17a98a8656d220ed3daf78b59b4cadea665d1b8615de95c92b3514a39a08c02`，37394B compact SHA `172fbaa85917bd1d6842d9d5bedad7fd96f20baf82772116f0de8d7fbf1d03dc`。部署seed须合真实wrapper env，再切audit0，不能把basis-only配置称为全部实际环境。

另启独立K5容量CPU优化准备线：target36/E48、原生draft15/E16；K从实际loaded W2/FFN角色派生，保留原INT8 prepack前snapshot、canary/planner与原x恢复契约。旧M1已过矩阵不重跑，其机制结果不外推正式收益。准备完成后root按fresh门分配健康空闲单芯，与TP8及tiny4并行。

## 临时目录约定与实际迁移

用户指定后续dsv41本地临时内容使用`~/tmp/dsv41-<任务名>`，已写入workspace/repo AGENTS并由三代理ACK。检查无活跃open文件或特殊节点后，46项历史文件/目录共 **6,578,301,143B**，经复制前/复制后/源再检查三份内容清单一致而迁移，保留旧路径链接。

历史产物集中于`~/tmp/dsv41-history-20261012`；本轮自检目录位于`~/tmp/dsv41-tp8-selfcheck-20261012`。迁移回执SHA `86109b9f9fec5f1dfd15d6833440330693fb238468302f4da2ff4e7d323e1bcc`。`/tmp`由86%占用降到18%，空闲6.7GB。已冻结源码/证据原字节、当前远端NPU运行目录与任务保持。

v44已完成HEAD5709项清单、归档自检、GitHub/a321双push，提交`e6080309685a0cd2d429c2eaeb9a8269ed0b3673`。本次进一步源码/原证据与临时路径约定将精确交付；目标始终active，完整20ms/质量/最优部署未达成。
