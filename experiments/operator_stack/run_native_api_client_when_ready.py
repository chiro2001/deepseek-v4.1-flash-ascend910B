"""Run the frozen same-namespace client only after actual private HTTP identity."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    for key in ('source-root', 'job-root', 'served-model', 'workload'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--port', required=True, type=int)
    a = p.parse_args()
    job = Path(a.job_root)
    out = job / 'client'
    assert not out.exists()
    selection = job / 'service/service_selection.json'
    url = 'http://127.0.0.1:' + str(a.port)
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if (job / 'run.exit').exists():
            raise RuntimeError('The original service is terminal; do not restart it')
        if selection.exists():
            current = json.loads(selection.read_text())
            if current.get('service_started'):
                try:
                    with urllib.request.urlopen(url + '/private/dspark/identity', timeout=3) as response:
                        identity = json.loads(response.read())
                    assert identity['served_model'] == a.served_model
                    assert identity['service_id'] == current['service_id']
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    pass
        print('WAITING_FOR_ACTUAL_PRIVATE_HTTP_IDENTITY', flush=True)
        time.sleep(15)
    else:
        raise TimeoutError('Original service startup observation deadline; keep job for root inspection')
    stack = str(Path(a.source_root) / 'stack')
    command = [sys.executable, '-c',
        'import sys,runpy; sys.path.insert(0,sys.argv.pop(1)); '
        'runpy.run_module("parallel_tiny.dspark_api_counter_bridge_cpu.acceptance_client",run_name="__main__")',
        stack, '--base-url', url, '--model', a.served_model,
        '--service-selection', str(selection), '--prompts', a.workload,
        '--output', str(out), '--execute']
    print('ACTUAL_PRIVATE_IDENTITY_VERIFIED_STARTING_FROZEN_CLIENT', flush=True)
    result = subprocess.run(command)
    (job / 'client.exit').write_text(str(result.returncode) + '\n')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
