"""Launch the audited formal TP8 arm for client acceptance."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys


def verify_selection(audit_root, perf_root, arm, contract, chips):
    audit = json.loads((audit_root / 'result.json').read_text())
    perf = json.loads((perf_root / 'result.json').read_text())
    for path, result in ((audit_root, audit), (perf_root, perf)):
        assert (path / 'run.exit').read_text().strip() == '0'
        assert result['formal_weights'] and result['tensor_parallel_size'] == 8
        assert result['checkpoint_config_sha256'] == contract['config_sha256']
        assert result['model_path'] == contract['model_path']
        assert result['physical_chips'] == chips
        assert result['hccl_deterministic_env'] == 'strict'
        assert not result['fp32_decode_reduction'] and not result['speculative_decoding']
        assert result['profiler'] == 'OFF' and result['same_model_instance'] and result['same_processes_per_rank']
    assert audit['audit'] and not perf['audit']
    assert arm in audit['arms'] and arm in perf['arms']
    comparisons = [r for r in audit['pairs'] if r['arm'] == arm]
    if arm != 'tp8base':
        assert len(comparisons) >= 3 and all(r['tokens_equal'] and r['actual_routes_equal'] and r['max_logprob_delta'] < 1e-3 for r in comparisons)
    if arm in ('tp8prefix', 'tp8prefixroute'):
        banks = json.loads((audit_root / 'banks.json').read_text())[arm]
        assert len(banks) == 8 and {r['rank'] for r in banks} == set(range(8))
        for bank in banks:
            prefix = bank['w4a8_prefix']
            assert prefix['selected'] and prefix['consumer_snapshots'] == 80
            assert prefix['captured_gmm_calls'] == {'apply_gmm1_act_quant': 40, 'apply_gmm2': 40}
        requests = json.loads((audit_root / 'requests.json').read_text())
        selected = [r for r in requests if r['arm'] == arm and r['tag'].startswith('pair-')]
        assert len(selected) >= 3
        for request in selected:
            assert len(request['rank_audit']) == 8
            assert all(r['w4a8_prefix']['passed'] for r in request['rank_audit'])
    controls = json.loads((audit_root / 'native_controls.json').read_text())
    assert len(controls) >= 3 and all(r['passed'] for r in controls)
    assert len({r['pair'] for r in perf['pairs']}) >= 10
    for metrics in perf['arms'].values():
        ms = metrics['ms_per_step']
        assert math.isfinite(ms) and ms > 0 and metrics['A'] == 1
        assert math.isfinite(metrics['tokens_per_second'])
        assert abs(metrics['tokens_per_second'] - 1000/ms) < 1e-6
    best = min(perf['arms'], key=lambda name: perf['arms'][name]['ms_per_step'])
    assert arm == best, ('Requested arm is not the measured best', arm, best)
    return {'selected_arm': arm, 'measured': perf['arms'][arm],
            'audit_root': str(audit_root), 'perf_root': str(perf_root),
            'audit_result_sha256': hashlib.sha256((audit_root / 'result.json').read_bytes()).hexdigest(),
            'perf_result_sha256': hashlib.sha256((perf_root / 'result.json').read_bytes()).hexdigest(),
            'client_validation': 'pending'}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model', required=True)
    parser.add_argument('--physical-chips', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit-root', type=Path, required=True)
    parser.add_argument('--perf-root', type=Path, required=True)
    parser.add_argument('--arm', choices=('tp8base','tp8core','tp8act','tp8stack','tp8meta','tp8metastack','tp8prefix','tp8prefixroute'), required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--served-model', required=True)
    args = parser.parse_args()
    chips = [int(x) for x in args.physical_chips.split(',')]
    assert args.physical_chips == os.environ['STACK_PHYSICAL_CHIPS']
    assert len(set(chips)) == 8 and os.environ['HCCL_DETERMINISTIC'] == 'strict'
    assert os.getenv('STACK_FP32_DECODE_REDUCTION') == '0'
    assert 1024 <= args.port <= 65535
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', args.port))
    from formal_model_contract import inspect_checkpoint
    contract = inspect_checkpoint(args.model)
    receipt = verify_selection(args.audit_root,args.perf_root,args.arm,contract,chips)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'selection.json').write_text(json.dumps(receipt,indent=2)+'\n')
    os.environ.update(STACK_REAL_WEIGHTS='1', STACK_REAL_AUDIT='0',
                      STACK_SERVE_ARM=args.arm, STACK_TP8_ARM='tp8base',
                      OPT_BLOCKMAP_VERIFY='0', VLLM_ENABLE_V1_MULTIPROCESSING='1',
                      VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_ALLOW_INSECURE_SERIALIZATION='1',
                      VLLM_DISABLE_COMPILE_CACHE='1', VLLM_CACHE_ROOT=str(args.output/'cache'))
    command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model',args.model,'--tokenizer',args.model,'--served-model-name',args.served_model,
        '--host','127.0.0.1','--port',str(args.port),'--dtype','bfloat16','--load-format','auto',
        '--tokenizer-mode','deepseek_v41','--reasoning-parser','deepseek_v41',
        '--tool-call-parser','deepseek_v41','--enable-auto-tool-choice',
        '--default-chat-template-kwargs',json.dumps({'enable_thinking':False}),
        '--safetensors-load-strategy','lazy',
        '--model-loader-extra-config',json.dumps({'enable_multithread_load':True,'num_threads':128}),
        '--worker-cls','real_tp8_worker.ServingRealTP8StackWorker',
        '--tensor-parallel-size','8','--enable-expert-parallel','--distributed-executor-backend','mp',
        '--seed','0','--trust-remote-code','--no-async-scheduling',
        '--max-model-len','8192','--max-num-seqs','1','--max-num-batched-tokens','2048',
        '--gpu-memory-utilization','0.70','--kv-cache-memory-bytes',str(4*1024**3),
        '--block-size','128','--no-enable-prefix-caching',
        '--limit-mm-per-prompt',json.dumps({'image':4}),
        '--compilation-config',json.dumps({'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1]}),
        '--additional-config',json.dumps({'enable_engram':True,'engram_storage':'int8',
            'enable_cpu_binding':False,'ascend_compilation_config':{'enable_npugraph_ex':True,'enable_static_kernel':True},
            'multistream_dsv4_dsa_overlap':False})]
    receipt['argv'] = command
    receipt['served_model'] = args.served_model
    receipt['port'] = args.port
    (args.output / 'service_command.json').write_text(json.dumps(receipt,indent=2)+'\n')
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
