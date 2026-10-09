"""Assert model identity, then verify an independent tiny completion."""
import argparse
import json
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--base", default="http://127.0.0.1:18973")
parser.add_argument("--output", default="/work/results/service_smoke.json")
parser.add_argument('--candidate',default='none')
parser.add_argument('--candidate-arm',default='baseline')
args = parser.parse_args()
model = "dsv41-tiny-upstream950-20261009"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
assert opener.open(args.base + "/health", timeout=5).status == 200
models = json.load(opener.open(args.base + "/v1/models", timeout=5))
assert models["data"][0]["id"] == model, models
payload = {"model": model, "prompt": [100 + i % 97 for i in range(32)],
           "max_tokens": 16, "temperature": 0, "ignore_eos": True}
request = urllib.request.Request(args.base + "/v1/completions",
                                data=json.dumps(payload).encode(),
                                headers={"Content-Type": "application/json"})
start = time.perf_counter()
response = json.load(opener.open(request, timeout=120))
assert response["usage"]["completion_tokens"] == 16, response
assert response["usage"]["prompt_tokens"] == 32, response
result = {"model": model, "endpoint": args.base, "health": 200,
          "usage": response["usage"], "finish_reason": response["choices"][0]["finish_reason"],
          "wall_seconds": time.perf_counter() - start, "dummy_weights": True,
          "baseline_arm": "both", "speculation": False}
result.update({'candidate':args.candidate,'candidate_arm':args.candidate_arm})
Path(args.output).write_text(json.dumps(result, indent=2))
print(json.dumps(result))
