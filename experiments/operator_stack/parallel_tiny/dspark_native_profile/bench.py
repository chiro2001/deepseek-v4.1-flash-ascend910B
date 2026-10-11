"""Profile the completed native DSpark A/A control; never publish performance."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from parallel_tiny.dspark_formal_prep.contract import PHYSICAL_CHIPS, inspect_dspark_checkpoint, model_kwargs, startup_env, validate_hot_flags
import formal_dspark_counters as counters
from bench_formal_dspark import mapped_generation


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--native-aa-root', type=Path, required=True)
    p.add_argument('--physical-chips', default=PHYSICAL_CHIPS)
    args = p.parse_args()
    assert args.physical_chips == os.environ['STACK_PHYSICAL_CHIPS'] == PHYSICAL_CHIPS
    result = {'scope': 'native full40 TP8 DSpark intrusive timeline diagnosis only',
              'formal_performance_claim': None, 'full_consumer_audit_passed': False,
              'goal20_complete': False, 'phase': 'validating-native-control'}
    args.output.mkdir(exist_ok=True)
    try:
        prior = json.loads((args.native_aa_root/'result.json').read_text())
        assert (args.native_aa_root/'run.exit').read_text().strip() == '0'
        assert prior['native_aa_passed'] and prior['mode'] == 'audit'
        stack = Path(__file__).resolve().parents[2]
        for name, digest in prior['source_hashes'].items():
            assert hashlib.sha256((stack/name).read_bytes()).hexdigest() == digest, name
        checkpoint = inspect_dspark_checkpoint(args.model, 5)
        assert checkpoint['config_sha256'] == prior['checkpoint_config_sha256']
        hot = Path('/tmp/v41_dspark_flags')
        validate_hot_flags(hot.read_text() if hot.exists() else '')
        os.environ.update(startup_env(args.output, audit=False, draft_graph=False))
        result['phase'] = 'loading-native-dspark'
        result['native_aa_source_hashes'] = prior['source_hashes']
        write(args.output/'progress.json', result)
        from vllm import LLM, SamplingParams
        kwargs = model_kwargs(args.model, 5, audit=False, draft_graph=False)
        kwargs['worker_cls'] = 'parallel_tiny.dspark_native_profile.worker.FormalDSparkProfileWorker'
        llm = LLM(**kwargs)
        tokenizer = llm.get_tokenizer()
        prompts = [tokenizer.encode((text+'这是供分析的上下文，保持事实准确。')*512, add_special_tokens=False)[:2048]
                   for text in ('请分析延迟受限计算的优化原理，给出判断依据。', '请解释缓存命中、数据搬运和流水线重叠的关系。')]
        assert all(len(prompt) == 2048 for prompt in prompts)
        sampling = SamplingParams(temperature=0, max_tokens=256, ignore_eos=True, detokenize=False)
        for prompt in prompts:
            mapped_generation(llm, prompt, sampling)
        banks = llm.collective_rpc(counters.bank)
        assert {b['rank'] for b in banks} == set(range(8))
        assert all(b['main_layers'] == 40 and b['draft_layer_ids'] == [40,41,42] and
                   b['method'] == 'dspark' and not b['draft_graph_configured'] for b in banks)
        write(args.output/'banks.json', banks)
        llm.collective_rpc(counters.begin, args=('native-dspark-profile', 256))
        # Resolve only after LLM's normal platform/worker initialization.
        from parallel_tiny.dspark_native_profile import worker
        llm.collective_rpc(worker.arm)
        output, core_id = mapped_generation(llm, prompts[0], sampling)
        receipts = llm.collective_rpc(counters.finish)
        profile = llm.collective_rpc(worker.finish)
        assert {row['rank'] for row in profile} == set(range(8))
        assert all(row['request_core_id'] == core_id and row['completed'] and row['actual_decode_steps'] == 15 for row in profile)
        assert all(set(row['requests']) == {core_id} for row in receipts)
        write(args.output/'raw_counters.json', receipts)  # May exceed 1MB: retain remotely.
        write(args.output/'profile_workers.json', profile)
        write(args.output/'output_tokens.json', {'request_id': output.request_id, 'core_request_id': core_id,
              'token_ids': list(output.outputs[0].token_ids), 'performance_receipt_granted': False})
        result.update(phase='profile-window-complete-device-CSV-validation-pending',
            observed_rank_count=8, observed_steps_per_rank=15, active_steps_per_rank=10,
            raw_counter_sha256=hashlib.sha256((args.output/'raw_counters.json').read_bytes()).hexdigest(),
            raw_counter_bytes=(args.output/'raw_counters.json').stat().st_size,
            native_profile_window_complete=True)
        write(args.output/'result.json', result)
        return 0
    except Exception as exc:
        result.update(phase='terminal-'+result['phase'], exception=repr(exc), native_profile_window_complete=False)
        write(args.output/'result.json', result)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
