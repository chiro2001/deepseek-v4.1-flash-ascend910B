#!/usr/bin/env bash
# 记录当前 tiny 的启动配置（不再每次恢复，只记录）
CT=dsv41-tinyspark
LOG=${LOG:-$HOME/cedpd-repo/results/TINY-CONFIG-LOG.md}
A=$(docker exec $CT bash -lc 'ps -ef | grep -m1 "vllm serve" | sed "s/.*vllm serve //"')
get() { echo "$A" | grep -oE "$1" | head -1; }
H=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:19310/health)
LINE=$(printf "| %s | %s | %s | %s | %s | %s | %s |" \
  "$(date '+%m-%d %H:%M')" "$H" \
  "$(get 'decode-context-parallel-size [0-9]+' | awk '{print $2}')" \
  "$(echo "$A" | grep -oE 'multistream_dsv4_dsa_overlap":[a-z]+' | cut -d: -f2)" \
  "$(echo "$A" | grep -c 'enable-dbo')" \
  "$(get 'max-num-seqs [0-9]+' | awk '{print $2}')" \
  "$(get 'num_speculative_tokens":[0-9]+' | cut -d: -f2)")
if [ ! -f "$LOG" ]; then
  mkdir -p "$(dirname "$LOG")"
  cat > "$LOG" <<'HDR'
# tiny (dsv41-tinyspark) 启动配置台账

> 规则：**不再每次恢复**；每次换配置后追加一行即可。
> 列：时间 | health | DCP | DSA_OVERLAP | DBO | MAX_SEQS | SP_TOKENS

| 时间 | health | DCP | DSA_OVERLAP | DBO | MAX_SEQS | SP_TOKENS |
|---|---|---|---|---|---|---|
HDR
fi
echo "$LINE" >> "$LOG"
echo "$LINE"
