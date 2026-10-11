"""Freeze a new snapshot with a real all-TP terminal completion boundary."""
import hashlib
import json
from pathlib import Path

from prepare_stock_v3 import replace_once


def main():
    parent = Path(__file__).parent
    source, output = parent / 'stock_compat_v3', parent / 'stock_compat_v4'
    output.mkdir(exist_ok=True)
    freeze = json.loads((source / 'prep_freeze_for_new_snapshot.json').read_text())
    previous = {name: freeze['files'][name] for name in ('formal_dspark_counters.py', 'bench_formal_dspark.py')}
    texts = {}
    for name, digest in previous.items():
        data = (source / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name
        texts[name] = data.decode()
    counter = texts['formal_dspark_counters.py']
    counter = replace_once(counter, '\ndef install_runner(runner, rank, *, audit):\n',
        '''
def terminal_tp_completion_join(counter, tp_group, distributed):
    """Join already completed device work before publishing the terminal output.

    The sample timestamp precedes entry on every rank. A CPU barrier therefore
    guarantees that the output rank cannot publish before any rank's fence and
    timestamp. Per-rank barrier returns need not occur simultaneously.
    """
    if (tp_group.world_size != 8 or tp_group.rank_in_group != counter.rank or
            str(distributed.get_backend(tp_group.cpu_group)) != 'gloo'):
        raise ValueError('Terminal completion requires the actual eight-rank CPU Gloo TP group')
    if counter.pending is not None or len(counter.requests) != 1:
        raise ValueError('Terminal join requires one completed real request round')
    request = next(iter(counter.requests.values()))
    row = request['rounds'][-1]
    if not row.get('terminal_device_synced') or 'terminal_tp_join' in row:
        raise ValueError('Terminal join missing a device fence or counted twice')
    entered = counter.now()
    tp_group.barrier()  # Stock GroupCoordinator.barrier uses only cpu_group.
    row['terminal_tp_join'] = {'backend': 'gloo', 'world_size': 8,
        'rank': counter.rank, 'entered_ns': entered, 'returned_ns': counter.now(),
        'device_fence_preceded_entry': True, 'sample_timestamp_preceded_entry': True}


def install_runner(runner, rank, *, audit):
''')
    counter = replace_once(counter,
        "        counter.sample_finished(route_receipt=route,terminal_device_synced=terminal)\n        return result\n",
        "        counter.sample_finished(route_receipt=route,terminal_device_synced=terminal)\n"
        "        if terminal:\n"
        "            from vllm.distributed import get_tp_group\n"
        "            import torch.distributed as distributed\n"
        "            terminal_tp_completion_join(counter,get_tp_group(),distributed)\n"
        "        return result\n")
    counter = replace_once(counter,
        "        if not rounds[-1].get('terminal_device_synced'):\n            raise ValueError('Final native draft/target work has no completion fence')\n",
        "        if not rounds[-1].get('terminal_device_synced'):\n            raise ValueError('Final native draft/target work has no completion fence')\n"
        "        join = rounds[-1].get('terminal_tp_join')\n"
        "        if (not join or join.get('backend') != 'gloo' or join.get('world_size') != 8 or\n"
        "                join.get('rank') != receipt['rank'] or\n"
        "                not join.get('device_fence_preceded_entry') or\n"
        "                not join.get('sample_timestamp_preceded_entry') or\n"
        "                not rounds[-1]['sample_end_ns'] <= join['entered_ns'] <= join['returned_ns']):\n"
        "            raise ValueError('Missing actual all-TP terminal completion join')\n")
    counter = replace_once(counter,
        "    metrics['terminal_device_completion_observed']=True\n",
        "    metrics['terminal_device_completion_observed']=True\n"
        "    metrics['terminal_tp_completion_join_observed']=True\n"
        "    metrics['terminal_completion_scope']='each rank device fence + sample timestamp -> CPU Gloo TP8 join -> client output; included in wall clock'\n")

    bench = texts['bench_formal_dspark.py']
    bench = replace_once(bench,
        "            round_receipts=llm.collective_rpc(counters.finish)\n",
        "            round_receipts=llm.collective_rpc(counters.finish)\n"
        "            # Preserve raw evidence even if a completeness gate refuses\n"
        "            # the request. The retrieval RPC is after the client clock.\n"
        "            write_json(args.output/(tag+'_counters.json'),round_receipts)\n"
        "            write_json(args.output/(tag+'_raw_request.json'),\n"
        "                {'tag':tag,'seed':seed,'token_ids':tokens,'request_id':output.request_id,\n"
        "                 'core_request_id':core_request_id,'client_end_ns':ended,\n"
        "                 'clock_scope':counters.clock_scope(),'performance_receipt_granted':False})\n")
    bench = replace_once(bench,
        "                                       expected_core_request_id=core_request_id)\n            write_json(args.output/(tag+'_counters.json'),round_receipts)\n",
        "                                       expected_core_request_id=core_request_id)\n")
    for name, value in (('formal_dspark_counters.py', counter), ('bench_formal_dspark.py', bench)):
        (output / name).write_text(value)
    for name in previous:
        freeze['files'][name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    freeze['phase'] = 'stock-v4-terminal-device-fences-and-CPU-TP8-join; fresh native AA pending'
    freeze['previous_snapshot_files'] = previous
    freeze['terminal_completion'] = 'actual per-rank device fence/timestamp before stock CPU Gloo TP8 join; no clock clamping or retrieval RPC in decode'
    freeze['cpu_counter_tests'] = 32
    (output / 'prep_freeze_for_new_snapshot.json').write_text(json.dumps(freeze, indent=2) + '\n')
    receipt = {'scope': 'new stock snapshot terminal completion correction; no NPU proof',
               'source_v3_files': previous, 'files': {name: freeze['files'][name] for name in previous},
               'native_aa_passed': False, 'device_audit_passed': False,
               'performance_receipt_granted': False}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
