"""Run on the chosen host: wait for free chips, record and launch formal TP8."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from launch_healthy_probe import resources


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--root', required=True)
    p.add_argument('--container', required=True)
    p.add_argument('--source-dir', default='/work/src',
                   help='Private immutable source snapshot inside the task mount')
    p.add_argument('--job', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--chips', required=True)
    p.add_argument('--pairs', type=int, default=3)
    p.add_argument('--arms', default='tp8base,tp8core,tp8act,tp8stack')
    p.add_argument('--audit', action='store_true')
    p.add_argument('--cpu-diagnostic',action='store_true')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--native-control', action='store_true')
    p.add_argument('--native-profile', action='store_true')
    p.add_argument('--reduction-bench', action='store_true')
    p.add_argument('--slot-bench', action='store_true')
    p.add_argument('--vector-job')
    p.add_argument('--serve', action='store_true')
    p.add_argument('--service-arm', choices=('tp8base','tp8core','tp8act','tp8stack','tp8meta','tp8metastack'))
    p.add_argument('--audit-job')
    p.add_argument('--perf-job')
    p.add_argument('--port', type=int)
    p.add_argument('--served-model')
    p.add_argument('--eager', action='store_true')
    p.add_argument('--probe-native-ops', action='store_true')
    p.add_argument('--trace-attention', action='store_true')
    p.add_argument('--fp32-decode-reduction', action='store_true')
    p.add_argument('--hccl-deterministic', choices=('false','true','strict'))
    p.add_argument('--hccl-npu-socket-port-range', choices=('auto',))
    p.add_argument('--reduction-evidence-job')
    p.add_argument('--isolated-pools', action='store_true',
                   help='Diagnostic: capture each candidate bank in a separate NPU memory pool')
    p.add_argument('--allow-alarm', action='store_true')
    p.add_argument('--wait-seconds', type=int, default=0)
    args = p.parse_args()
    assert args.job.replace('-', '').replace('_', '').isalnum(), args.job
    assert args.pairs > 0 and args.wait_seconds >= 0
    assert not (args.audit and args.profile)
    assert not args.cpu_diagnostic or not any((args.audit,args.profile,args.native_control,
        args.native_profile,args.reduction_bench,args.slot_bench,args.serve,args.eager))
    assert not args.slot_bench or not any((args.audit,args.profile,args.native_control,args.native_profile,
        args.reduction_bench,args.serve,args.eager,args.probe_native_ops,args.trace_attention,
        args.fp32_decode_reduction,args.isolated_pools))
    assert not args.serve or (args.service_arm and args.audit_job and args.perf_job and args.port and args.served_model and
        args.hccl_deterministic=='strict' and not any((args.audit,args.profile,args.native_control,args.native_profile,args.reduction_bench,args.eager,args.probe_native_ops,args.trace_attention,args.fp32_decode_reduction,args.isolated_pools)))
    for name in (args.audit_job,args.perf_job):
        assert not name or (args.serve and name.replace('-', '').replace('_', '').isalnum())
    assert not args.reduction_bench or (args.vector_job and not any((args.audit,args.profile,args.native_control,args.native_profile,args.eager,args.probe_native_ops,args.trace_attention,args.fp32_decode_reduction,args.isolated_pools)))
    assert not args.vector_job or (args.reduction_bench and args.vector_job.replace('-', '').replace('_', '').isalnum())
    assert not args.native_profile or (not args.native_control and not args.audit and not args.profile and not args.eager and not args.isolated_pools)
    assert not args.native_control or (args.audit and not args.profile)
    assert not args.eager or args.native_control
    assert not args.probe_native_ops or (args.native_control and args.eager)
    assert not args.trace_attention or (args.native_control and args.eager and not args.probe_native_ops)
    assert not args.fp32_decode_reduction or (not any((args.native_profile,args.reduction_bench,args.trace_attention,args.probe_native_ops)) and
        ((args.native_control and args.eager) or (not args.native_control and args.reduction_evidence_job)))
    assert not args.reduction_evidence_job or (args.fp32_decode_reduction and
        args.reduction_evidence_job.replace('-', '').replace('_', '').isalnum())
    assert not args.isolated_pools or not args.native_control
    arms = args.arms.split(',')
    assert arms[0] == 'tp8base' and len(set(arms)) == len(arms)
    assert set(arms) <= {'tp8base', 'tp8core', 'tp8act', 'tp8stack', 'tp8meta', 'tp8metastack'}
    chips = [int(x) for x in args.chips.split(',')]
    assert len(chips) == len(set(chips)) == 8 and min(chips) >= 0
    root = Path(args.root).resolve()
    source_dir=Path(args.source_dir)
    assert source_dir.is_absolute() and '..' not in source_dir.parts and source_dir.is_relative_to('/work')
    source_host=(root/source_dir.relative_to('/work')).resolve()
    assert source_host.is_relative_to(root) and (source_host/'stack/run_stack.sh').is_file()
    config = json.loads(subprocess.check_output(['docker', 'inspect', args.container], text=True))[0]
    container_env = dict(item.split('=', 1) for item in config['Config']['Env'])
    assert config['State']['Running']
    assert container_env['STACK_PHYSICAL_CHIPS'] == args.chips
    assert container_env['ASCEND_RT_VISIBLE_DEVICES'] == args.chips
    assert any(m['Destination'] == '/work' and Path(m['Source']).resolve() == root
               for m in config['Mounts']), 'Task mount differs from requested root'
    out = root / 'results' / args.job
    # Container-created result parents may belong to root. Give the host
    # launcher ownership of this job directory only, leaving other jobs alone.
    setup = (f"from pathlib import Path; import os; p=Path('/work/results/{args.job}'); "
             f"p.mkdir(parents=True,exist_ok=True); os.chown(p,{os.getuid()},{os.getgid()})")
    subprocess.run(['docker', 'exec', args.container, 'python', '-c', setup], check=True)
    out.mkdir(parents=True, exist_ok=True)
    # This lock prevents duplicate launches of our job. It does not reserve
    # devices against other tenants; the live device table is checked below.
    with (out / 'launch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (out / 'launched.json').exists(), 'Inspect existing job; do not relaunch'
        deadline = time.monotonic() + args.wait_seconds
        while True:
            raw = subprocess.check_output(['npu-smi', 'info'], text=True)
            parsed = resources(raw)
            allowed = ('OK', 'Alarm') if args.allow_alarm else ('OK',)
            for chip in chips:
                assert chip in parsed and parsed[chip]['health'] in allowed, (chip, parsed.get(chip))
            busy = {chip: parsed[chip]['processes'] for chip in chips if parsed[chip]['processes']}
            if not busy:
                break
            (out / 'waiting.json').write_text(json.dumps({'time': time.time(), 'busy': busy}, indent=2) + '\n')
            print('REAL_TP8_WAITING', json.dumps(busy), flush=True)
            assert time.monotonic() < deadline, ('Devices occupied', busy)
            time.sleep(min(10, max(0, deadline - time.monotonic())))
        alarm_details = {}
        for chip in chips:
            if parsed[chip]['health'] == 'Alarm':
                details = subprocess.check_output(['npu-smi', 'info', '-t', 'health',
                    '-i', str(chip // 2), '-c', str(chip % 2)], text=True)
                codes = [line.split(':', 1)[1].strip() for line in details.splitlines()
                         if 'Error Code' in line and ':' in line]
                assert codes == ['80C98001'], ('Unexpected Alarm', chip, details)
                alarm_details[chip] = details
        flags = {
            'STACK_SRC_ROOT': str(source_dir),
            'LOCAL_WORLD_SIZE': '8', 'TASK_QUEUE_ENABLE': '1',
            'HCCL_OP_EXPANSION_MODE': 'AIV', 'HCCL_BUFFSIZE': '1024',
            'ASCEND_MAX_OP_CACHE_SIZE': '-1',
            'PYTORCH_NPU_ALLOC_CONF': 'expandable_segments:True',
            'V41_ENGRAM_HOST_RESIDENT': '1', 'V41_ENGRAM_REUSE_EP_GROUP': '1',
            'V41_ENGRAM_GATE_CHUNK': '0', 'V41_ENGRAM_GATE_MAX_TOKENS': '4096',
            'V41_ENGRAM_GATE_HOIST': '0', 'V41_ENGRAM_JIT': '1',
            'V41_ENGRAM_DEVICE_INDEX': '1', 'V41_ENGRAM_DEVICE_FALLBACK': '0',
            'V41_QLI_NO_CANDIDATE': '1', 'V41_MOE_COMM_ALLGATHER': '1',
            'V41_MOE_MASK_RANGE': '1', 'V41_ROPE_IDXSEL': '1', 'V41_O_PROJ_2D': '1',
            'V41_MOE_ZERO_INVALID': '0', 'V41_MOE_ZERO_NONFINITE': '0',
            'V41_ENGRAM_ROUTE_PROBE': '0', 'NUMBA_CACHE_DIR': '/work/cache/numba',
        }
        flags['STACK_TP8_ISOLATED_POOLS'] = '1' if args.isolated_pools else '0'
        flags['STACK_METADATA_MANY_SLOTS_ENABLED'] = '1' if (set(arms)&{'tp8meta','tp8metastack'} or args.service_arm in ('tp8meta','tp8metastack')) else '0'
        flags['STACK_FP32_DECODE_REDUCTION'] = '1' if args.fp32_decode_reduction else '0'
        if args.hccl_deterministic is not None:
            # HCCL reads this at process/communicator initialization; exporting
            # it into the fresh job is intentional, not a live group toggle.
            flags['HCCL_DETERMINISTIC'] = args.hccl_deterministic
        if args.hccl_npu_socket_port_range is not None:
            flags['HCCL_NPU_SOCKET_PORT_RANGE'] = args.hccl_npu_socket_port_range
        script_name = ('bench_formal_slots.py' if args.slot_bench else 'serve_formal_tp8.py' if args.serve else
                       'bench_real_reductions.py' if args.reduction_bench else
                       'profile_formal_native.py' if args.native_profile else
                       'bench_formal_native_control.py' if args.native_control else 'bench_real_tp8_model.py')
        command = ['bash', str(source_dir/'stack/run_stack.sh'), script_name,
            '--physical-chips=' + args.chips,
            '--output=/work/results/' + args.job]
        if args.reduction_bench:
            command.append('--input=/work/results/' + args.vector_job)
        elif not args.slot_bench:
            command.append('--model=' + args.model)
        if args.serve:
            command.extend(['--audit-root=/work/results/'+args.audit_job,
                            '--perf-root=/work/results/'+args.perf_job,
                            '--arm='+args.service_arm,'--port='+str(args.port),'--served-model='+args.served_model])
        if not args.native_profile and not args.reduction_bench and not args.slot_bench and not args.serve:
            command.append('--pairs=' + str(args.pairs))
        if not args.native_control and not args.native_profile and not args.reduction_bench and not args.slot_bench and not args.serve:
            command.append('--arms=' + args.arms)
        if args.eager:
            command.append('--eager')
        if args.probe_native_ops:
            command.append('--probe-native-ops')
        if args.trace_attention:
            command.append('--trace-attention')
        if args.fp32_decode_reduction:
            command.append('--fp32-decode-reduction')
        if args.hccl_deterministic is not None and (args.native_control or not (args.native_profile or args.reduction_bench or args.slot_bench or args.serve)):
            command.append('--hccl-deterministic=' + args.hccl_deterministic)
        if args.reduction_evidence_job:
            command.append('--reduction-evidence=/work/results/' + args.reduction_evidence_job)
        if args.audit:
            command.append('--audit')
        if args.profile:
            command.append('--profile')
        if args.cpu_diagnostic:
            command.append('--cpu-diagnostic')
        job_path = '/work/results/' + args.job
        body = '#!/usr/bin/env bash\nset -uo pipefail\numask 022\nmkdir -p /work/cache/numba\n'
        body += '\n'.join('export ' + k + '=' + shlex.quote(v) for k, v in flags.items()) + '\n'
        body += 'echo $$ > ' + job_path + '/runner.pid\n'
        body += 'date -u +%FT%TZ > ' + job_path + '/start.utc\n'
        body += shlex.join(command) + ' > ' + job_path + '/run.log 2>&1\n'
        body += 'job_status=$?\nprintf "%s\\n" "$job_status" > ' + job_path + '/run.exit\n'
        body += 'date -u +%FT%TZ > ' + job_path + '/end.utc\nexit "$job_status"\n'
        script = root / (args.job + '.sh')
        script.write_text(body)
        receipt = {'timestamp_unix': time.time(), 'container': args.container,
            'container_id': config['Id'], 'image_digest': config['Image'],
            'chips': chips, 'resources': parsed, 'alarm_details': alarm_details,
            'env': flags, 'argv': command,
            'script_sha256': hashlib.sha256(body.encode()).hexdigest(),
            'source_dir':str(source_dir),
            'source_sha256': hashlib.sha256((source_host / 'stack' / script_name).read_bytes()).hexdigest()}
        (out / 'resource_prelaunch.txt').write_text(raw)
        subprocess.run(['docker', 'exec', '-d', args.container, 'bash', '/work/' + script.name], check=True)
        (out / 'launched.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print('REAL_TP8_JOB_LAUNCHED', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
