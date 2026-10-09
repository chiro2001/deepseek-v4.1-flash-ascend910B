import json
import argparse
from datetime import datetime,timezone
import time
import urllib.request
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--base',required=True)
p.add_argument('--output',default='/home/l00886679/projects/dsv41-tiny-prof-20261009/results/service_smoke.json')
args=p.parse_args()
base=args.base
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
assert opener.open(base+'/health',timeout=5).status == 200
models = json.load(opener.open(base+'/v1/models',timeout=5))
assert models['data'][0]['id'] == 'dsv41-tiny-prof-20261009'
payload = {'model':'dsv41-tiny-prof-20261009','prompt':[100+i%97 for i in range(32)],
           'max_tokens':16,'temperature':0,'ignore_eos':True,'stream':False}
request = urllib.request.Request(base+'/v1/completions',data=json.dumps(payload).encode(),
                                 headers={'Content-Type':'application/json'})
start = time.perf_counter()
response = json.load(opener.open(request,timeout=60))
assert response['usage']['completion_tokens'] == 16
assert response['usage']['prompt_tokens'] == 32
result = {'endpoint':base,'model':models['data'][0]['id'],'health':200,
          'usage':response['usage'],'finish_reason':response['choices'][0]['finish_reason'],
          'wall_seconds':time.perf_counter()-start,'dummy_weights':True}
result['collected_at_utc']=datetime.now(timezone.utc).isoformat()
Path(args.output).write_text(json.dumps(result,indent=2))
print(json.dumps(result))
