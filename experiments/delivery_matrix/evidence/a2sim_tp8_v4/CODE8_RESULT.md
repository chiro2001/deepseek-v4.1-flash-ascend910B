# A2v4 编程正式轮（基础/长上下文精度仍待验）

model `dsv41-delivery-a2sim-tp8-v4`；a3-21 physical0–7模拟A2，非910B硬件数据。
SID `9d2ba214732b4eed84dc4f416f27c6ae`，actual runtime fp
`61b841dfcd20d7f2892934e1fbd8086ab409a36d99d5de9beea4dd12399e53f9`。
PREFIX0/SEQ1/BAT8192/MAXLEN196608、4GiB KV、Engram host index0、BI、async、双流及
DSpark5图；generic v7、geometry base8dc/derived7c/manifestdf，见ready/source收据。

## 正式8请求

先canary，再1暖机，不计入正式；8条串行中级编程题输入2048、自然结束≤1024。
HTTP/client decode三元组=`(27.218877914 ms/step, A=4.985611511, 183.167415152 tok/s)`，
278实际rounds、1394 completion tokens（去8个seed后decode1386），clean8/8。
按同一客户端窗口换算三元组，rate consistency delta约2.84e-14。
TTFT均值/中位=`0.311842044/0.310819380 s`；worker prefill wall 8请求合计
`1.424604450 s`（首次scheduled→末prompt CPU bookkeeping，不是纯设备kernel）。
server-fenced=`32.571885432 ms/step`，包含metrics RPC等待，**不能**与HTTP tok/s混搭。
每rank真实window graph278/eager8；eager为每请求的初始prefill提案，未伪造全图计数。

远端无bwrap，原8条sandbox_error保留；本机bwrap独立执行得8/8题、128/128用例，
非修改原receipt的“重评分覆盖”。root已核全部8输出/标签/文件SHA与split manifest
一致、score逐题passed==total且failed为空；这只证明此编程集，不代替GSM/Vision/长针。

## 原件与校验

- 大raw留远端`/home/l00886679/tmp/dsv41-a2sim-v4-bench/evidence/code_formal8_a2sim_v4_20261013.json`，
  1419839B，SHA`4c9196d15121358fe0b35ecd77d5b61c43095c49e09a39e9eca3258ec729d0d3`。
- 本机8 split件位于`/home/chiro/tmp/dsv41-a2sim-v4-bench/evidence/code_formal8_a2sim_v4_20261013_requests/`，
  最大217439B，均<1MB；manifest SHA
  `6ec9d9267695a4b6e7d5e27a1beaf0823df8498c9762b4c7204d2892474a8704`。
- 本机独立score原件`/home/chiro/tmp/dsv41-a2sim-v4-bench/evidence/code_formal8_a2sim_v4_20261013_local_scoring.json`，
  SHA`9f232da2041c374ffe8d9451070f981f09a2b5944b63b374dbe1ad42a6d57c7e`，绑定大raw/manifest、
  每个split/output/unit-case SHA。原件不改；repo拷贝及路径映射由client归档收据记录。
- 冻结benchmark SHA33c458…、workloads SHA62db3507…；独立bwrap只读Python/stdlib/libs、
  private network、任务TMPDIR，512MiB内存/2s CPU/3s wall约束。

GSM100/Vision23/147456四针已串行交site，尚无结论；没有并发Dihuo或继承v2质量。
