# DCP 定序归约：独立 CPU 候选，尚未上设备

当前已测 v3 的严格 GSM 为99/100；同题同512预算的一次诊断重复回答13且
正常stop（349 completion tokens）。这说明输出存在跨请求差异，但**不能**
据此认定归约顺序是唯一原因。Vision23/23与147456长针4/4通过，整体严格
精度门仍失败；没有用单题重复覆盖整轮，也没有放宽预算或精度门。

源码审查见 `evidence/dcp8_v3/reduction_review.md`（SHA
`38a4d99341ca4f4b0a2fb70a127cda4bc38987fedfc9a98e8f5e701f3638c413`）。
原审查文件误放于`evidence/dcp8_v3_v3/`，现已单文件Move至上述路径，内容/SHA不变。

## 候选变更

- 原 `V41_DCP_DET_REDUCE=1` 只覆盖部分分支；新派生源码的5个gate使用
  同一env/file predicate，覆盖pack、sepw独立分母、no-pack、legacy及kernel路径。
- 定序求和沿T分块，保持逐元素rank0→rankN−1顺序；输入/输出512B对齐。
- v6 的所有5个分支都调用同一bounded helper（共8处调用），不再保留kernel/
  folded-pack分支的inline全量gather；kernel用`copy_`写回原驻留pack地址。
- helper的collective input/gather workspace只分配**一对**并复用，尾块也使用相同
  大小；不能仅靠删除Python引用控制异步distributed stream的分配峰值。
- gather workspace硬帽128MiB，典型8192×64×640 FP32规划80MiB；另有完整
  output及局部sum，不能把workspace帽冒充全部设备内存峰值。
- `--dcp-det-reduce 1` 只允许经过新来源/容量契约的overlay，拒绝旧v3，
  不改变正在运行/已经测过的候选。HCCL strict、best模型、草稿与BI源保持。

## 冻结版本与验证边界

| overlay | manifest SHA | 状态 |
|---|---|---|
| v3 | `8d09cb35615b1a5bfd66c7e4b8f7602c263d2c580423f3599dcc9816ed90b556` | 已测，原件不改 |
| v4 | `21d96eee939dc8644b39f00cf1fe5c34ea2f857eeecb3c7ba1282c62c1c1ef79` | CPU初稿；逐chunk释放引用不足以保证异步allocation复用，未部署、保留 |
| v5 | `ad80804a16638270c8361c08aecfb073a913e729d9f2a2c550299158f5acf02e` | helper复用workspace，但2个inline分支仍全量gather；审查拒绝，未部署、保留 |
| v6 | `b0c1d36e44707c872707477677a5ef60a4d3d53f86c5df0d1092e47e41706a89` | 所有5分支都经bounded helper；下一独立设备候选，尚未部署 |

`test_dcp_stability.py` 8项CPU正负控通过：固定rank序列数值、cancellation
负控、尾块/空shape/预算失败、所有5个gate、实际同一workspace对象复用、
拒绝source漂移/旧overlay及v5、各分支inline gather回归负控、kernel驻留地址不变。
顶层delivery89/89通过。
这是NumPy tensor/collective spy，**不是**真实HCCL/ACL graph或设备峰值证明。
后续需在新job上验证初始化内存、capture/replay、重复同题与完整精度/性能。
