# MemCache 外部 KV 池接入（A2 单机 / 双机共享池）

本目录给出把 `AscendStoreConnector(backend=memcache)` 接到 vLLM-Ascend 上的最小集合，
以及在 A2 上实测的开销与踩坑。所有数据来自 2026-10-23 真机（8×910B1，TP8，W4A8 + Engram）。

## 1. 一个池是怎么组成的

```
     MetaService (1 个，跑在"主"节点)            ← mmc-meta.conf
        ▲            ▲
        │            │  注册 8 个 DRAM segment / 机
   client(.15)   client(.16)                     ← mmc-local.conf + drak.size
        └── device_rdma 数据面（跨机 load/save）──┘
```

* 池大小 = 每 rank `dram.size` × 参与 rank 数（8 rank/机）
* 例：`dram.size=48GB`，2 机 ⇒ 16 × 48 GB = **768 GiB**
* 各实例仍是 `kv_role=kv_both` 的单机混布实例（不是 PD 的 producer/consumer）
* 命中是跨机的：A 机写入的前缀，B 机能直接 load

## 2. 接入步骤

1. **起 MetaService**（只在主节点，容器内）：
   `mmc-meta.conf` 里写 `meta_service_url / config_store_url / metrics_url`，
   然后 `python3 -c 'from memcache_hybrid import MetaService; MetaService.main()'`（前台或 nohup）。
2. **每个客户端写 `mmc-local.conf`**：见 `mmc-local.conf.single`（单机池）/ `mmc-local.conf.peer`（跨机池）。
   唯一易错点：**端口必须与对端 MetaService 一致**（示例有 5100/6100 与 5000/6000 两套）。
3. **宿主先备好大页**：`sysctl -w vm.nr_hugepages=<页数>`，`200704` 页 = 392 GiB ⇒ 支持 8×48 GB。
   ⚠️ 必须在起服务**之前**加满（服务占着内存时只能加到剩余可用量）。
4. **起服脚本**：`inner_mmc.sh`（本目录）——关键是
   `MMC_LOCAL_CONFIG_PATH` 指向 conf + `LD_LIBRARY_PATH` 带上 `memcache_hybrid/lib`
   + `--kv-transfer-config` 带 `kv_role=kv_both`。

## 3. 怎么验证池真的成了

* **注册数**：`curl <主节点>:<meta 端口>/metrics | grep -c 'segment_capacity_bytes{'` ⇒ 应为 `8 × 机数`
* **容量**：`memcache_total_capacity_bytes{medium="dram"}` ⇒ 例 `824633720832` = 768 GiB
* **跨机命中**：给同一个长前缀先打 A 机、再打 B 机，看 B 机的
  `vllm:external_prefix_cache_hits_total` 是否增长（`prompt_tokens_by_source_total{source="external_kv_transfer"}` 同步涨）
* 实测（`.21`→`.25`，24K token 前缀）：`+52,736` 命中 token / `+54,184` 查询 ⇒ **99.7%**；
  首 token 延迟 `9.1s → 0.8s`

## 4. 开销实测（重要，和直觉不一样）

同一台 A2、同配置（`MAX_SEQS=32 PREFIX=1 GPU_UTIL=0.92`），唯一变量是连接器：

| 形态 | ms/step | 单流 |
|---|---:|---:|
| eager draft + 池 | 100.8 | ~30 tok/s |
| eager draft 无池 | 68.1 | ~46 tok/s |
| **draft 入图 + 池** | **42.0** | **~69 tok/s** |
| draft 入图 无池 | **35.4** | **~76 tok/s** |

⇒ **池的每步开销从 +33 ms 降到 +6.6 ms**（−80%）。机制推断：eager 时池的 save/lookup
同步点落在 draft 派发与 verify 之间（串行叠加）；draft 进图后这段 host 时间被图重放吸收。
**结论：要吃池 + 要吃投机，就把 draft 入图（见 `tools/dsenv_draftgraph.sh`）。**

## 5. 踩过的坑（照抄可避开）

1. `sudo -n /bin/su - -c '<cmd>'` 是这类机器的唯一提权方式；`sudo -n <cmd>` 会被 sudoers 拒绝。
2. **不要用 `pkill -f vllm`**：`docker exec` 外层匹配会把自己也杀掉（shell 命令行含同样字符串）。
   先 `pgrep -af` 看 pid，再按 pid kill。
3. **扩池要两端都改 + 都重启**：`dram.size` 在客户端启动时注册，只改一端不会变大。
4. **起服脚本必须重定向日志**：`docker exec -d <c> bash -c 'exec bash x.sh > /path/serve.log 2>&1'`
   否则 `docker logs` 是空的（`docker exec -d` 的 stdout 不进容器日志）。
   注意 `2>&1` 别写成 `2>&1` 之外的变体，也别在目录不存在时重定向 —— 进程会**静默起不来**。
5. `[bneck] hp` 不能当绝对步钟用：它是两次 `mark_step` 的墙钟差，**空闲间隙会被计入**
   （空闲机器上能读到 2000 ms/step）。测单流请用 `probe_decode_tps.py`（/metrics 差分）。
6. 单发 tok/s 在共享机上很容易被并发污染：务必看 `request_decode_time_seconds_count`
   的增量是否为 1，否则只当参考值。
