#!/usr/bin/env bash
# Install as /tmp/mysvc.sh only inside this lane's owned container.
set -euo pipefail
tiny_service_port=${1:-18973}
exec python - "$tiny_service_port" <<'PY'
import json,sys,urllib.request
model='dsv41-tiny-upstream950-20261009'
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    data=json.load(opener.open('http://127.0.0.1:'+str(int(sys.argv[1]))+'/v1/models',timeout=5))
except Exception as error:
    print('DOWN',str(error));raise SystemExit(2)
if len(data.get('data',[]))==1 and data['data'][0]['id']==model:
    print('MINE',model);raise SystemExit(0)
print('OTHER',json.dumps(data));raise SystemExit(1)
PY
