set -e
docker exec -i dsv41-tinyspark python3 - <<'PY'
import ast
p = "/vllm-workspace/vllm-ascend/vllm_ascend/worker/npu_ubatch_wrapper.py"
ast.parse(open(p).read())
src = open(p).read()
print("syntax OK; DBO-GRAPH-DETECT:", "DBO-GRAPH-DETECT" in src, "; NPU-STREAM-TLS:", "NPU-STREAM-TLS" in src)
PY
