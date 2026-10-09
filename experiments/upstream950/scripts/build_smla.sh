#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
if [[ ! -f /work/build/smla/source_manifest.json ]]; then
    python /work/scripts/prepare_smla.py
fi
python /work/scripts/generate_smla_pod.py
python - <<'PY'
from pathlib import Path
import shutil
root=Path('/work/build/smla/csrc')
for name in ['up950_sparse_flash_mla','up950_base_sparse_flash_mla']:
    staged=root/'build/binary/ascend910_93/src'/name/'op_kernel'
    if staged.exists():
        shutil.copytree(root/'attention'/name/'op_kernel',staged,dirs_exist_ok=True)
        # This CANN build's .done target omits header dependencies.
        for marker in (root/'build/binary/ascend910_93/gen').glob(name+'_*.done'):
            marker.unlink()
        binary=root/'build/binary/ascend910_93/bin'/name
        if binary.exists():
            shutil.rmtree(binary)
        print('Refreshed incremental kernel source',name)
PY
cd /work/build/smla/csrc
export MAX_JOBS=8
bash build.sh --pkg --ops='up950_sparse_flash_mla;up950_base_sparse_flash_mla' \
    --soc=ascend910_93 --vendor_name=up950 -j8
python - <<'PY'
from pathlib import Path
import subprocess,json
root=Path('/work/build/smla')
installers=list((root/'csrc/build').glob('cann-ops-transformer*.run'))
assert len(installers)==1,installers
subprocess.run(['bash',str(installers[0]),'--install-path='+str(root/'opp')],check=True)
validated=[]
for path in (root/'opp/vendors/up950_transformer').rglob('*.json'):
    try: data=json.loads(path.read_text())
    except (UnicodeDecodeError,json.JSONDecodeError): continue
    if 'opParaSize' in data:
        assert data['opParaSize']==232,(str(path),data['opParaSize'])
        validated.append(str(path))
assert len(validated)==2,validated
(root/'kernel_parameter_validation.json').write_text(json.dumps({'expected_opParaSize':232,'kernels':validated},indent=2))
print('Private CANN op installed to',root/'opp')
PY
