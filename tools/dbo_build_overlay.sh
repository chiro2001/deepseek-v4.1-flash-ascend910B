#!/usr/bin/env bash
# 一键重建 DBO overlay（幂等）。全部修复都在这里，便于复现/回退。
set -uo pipefail
T=$HOME/tmp
D=$HOME/dcpw/vllm_ascend
CT=dsv41-tinyspark
echo "== 1) 基础 overlay（platform.py 放开 enable_dbo + 两处 all2all 覆盖）"
python3 $T/apply_dbo_overlay.py 2>&1 | tail -3
echo "== 2) model_runner：ubatch 循环 + dummy_run + load_model + num_input_tokens"
python3 $T/patch_mr_full.py 2>&1 | tail -1
python3 $T/fix_numin.py 2>&1 | tail -1
echo "== 3) 两个 set_ascend_forward_context 调用补 ubatch_slices"
python3 $T/fix_ctx_arg2.py 2>&1 | tail -1
echo "== 4) dsa_v41：build 记 _ubid / _publish_task 分键 / _get_layer_metadata list / rope 缓存分键"
python3 $T/fix_publish_ubid.py 2>&1 | tail -3
python3 $T/fix_layer_meta.py 2>&1 | tail -2
python3 $T/fix_rope_cache.py 2>&1 | tail -1
echo "== 5) wrapper：ubatch_id + _cat_ubatch_outputs（先加函数，再改成空序列安全版）"
python3 $T/fix_layer_meta.py 2>&1 | tail -2
python3 $T/fix_cat_outputs.py 2>&1 | tail -1
python3 $T/fix_cat3.py 2>&1 | tail -1
echo "== 6) ascend_forward_context 形参"
python3 $T/fix_afc.py 2>&1 | tail -1
echo "== 7) 装入容器（dsa_v41 用原地截断写绕开陈旧 inode）"
for f in worker/model_runner_v1.py worker/worker.py worker/npu_ubatch_wrapper.py; do
  docker cp $D/$f $CT:/vllm-workspace/vllm-ascend/vllm_ascend/$f
done
docker exec -i $CT bash -lc "cat > /vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py" < $D/attention/dsa_v41.py
echo "== 8) 校验标记"
docker exec $CT bash -lc "F=/vllm-workspace/vllm-ascend/vllm_ascend; for k in 'NPUUBatchWrapper' 'ubatch_id' 'ubatch_slices'; do printf '  %-18s %s\n' \$k \$((\$(grep -rc \$k \$F/worker/model_runner_v1.py \$F/ascend_forward_context.py 2>/dev/null | awk -F: '{s+=\$2} END{print s}'))) ; done; printf '  %-18s %s\n' '_ubid' \$(grep -c _ubid \$F/attention/dsa_v41.py); printf '  %-18s %s\n' 'cat_v2' \$(grep -c '_cat_ubatch_outputs' \$F/worker/npu_ubatch_wrapper.py)"
