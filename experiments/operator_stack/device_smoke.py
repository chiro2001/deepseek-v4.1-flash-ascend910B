"""Isolated Vector/Cube and HCCL checks on explicitly selected devices."""
import argparse
import datetime
import faulthandler
import json
import os
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--expected-chips', required=True,
                        help='Comma-separated physical chips, matching ASCEND_RT_VISIBLE_DEVICES')
    args = parser.parse_args()
    chips = [int(x) for x in args.expected_chips.split(',')]
    assert chips and len(chips) == len(set(chips)) and min(chips) >= 0
    assert os.getenv('STACK_PHYSICAL_CHIPS', os.environ['ASCEND_RT_VISIBLE_DEVICES']) == args.expected_chips
    faulthandler.enable(); faulthandler.dump_traceback_later(45, repeat=True)
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    result = {'physical_chip': chips[rank], 'rank': rank, 'distributed': args.distributed,
              'visible_devices': os.environ['ASCEND_RT_VISIBLE_DEVICES'], 'status': 'started'}
    target = Path(args.output) / f'rank{rank}.json'; target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + '\n')
    try:
        import torch
        import torch_npu  # noqa: F401
        result['device_count'] = torch.npu.device_count()
        assert result['device_count'] == len(chips), result
        torch.npu.set_device(rank)
        torch.npu.set_op_timeout_ms(10000)
        value = torch.arange(128, device='npu', dtype=torch.float32)
        torch.testing.assert_close((value + 3).cpu(), torch.arange(128, dtype=torch.float32) + 3, rtol=0, atol=0)
        result['vector_passed'] = True
        a = torch.eye(64, device='npu', dtype=torch.bfloat16)
        b = torch.arange(4096, dtype=torch.float32).reshape(64, 64).to(torch.bfloat16)
        actual = torch.mm(a, b.to('npu'))
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), b, rtol=0, atol=0)
        result['cube_passed'] = True
        if args.distributed:
            import torch.distributed as dist
            dist.init_process_group('hccl', timeout=datetime.timedelta(seconds=45))
            assert dist.get_world_size() == len(chips)
            tensor = torch.tensor([float(rank + 1)], device='npu')
            dist.all_reduce(tensor); torch.npu.synchronize()
            expected_sum = len(chips) * (len(chips) + 1) / 2
            torch.testing.assert_close(tensor.cpu(), torch.tensor([expected_sum]), rtol=0, atol=0)
            result['hccl_all_reduce_passed'] = True
            dist.destroy_process_group()
        result['status'] = 'passed'
    except Exception as exc:
        result.update(status='failed', error=str(exc), traceback=traceback.format_exc())
    finally:
        faulthandler.cancel_dump_traceback_later()
        target.write_text(json.dumps(result, indent=2) + '\n')
        print('DEVICE_SMOKE', json.dumps(result), flush=True)
    if result['status'] != 'passed': raise SystemExit(1)


if __name__ == '__main__': main()
