"""CPU counter-contract controls. No torch/vLLM/NPU import or initialization."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from formal_dspark_counters import RoundCounters, clock_scope, summarize, install_runner, bank
from parallel_tiny.dspark_formal_prep.contract import model_kwargs, startup_env, validate_hot_flags


class Clock:
    def __init__(self):self.tick=1_000_000
    def __call__(self):
        self.tick+=1000
        return self.tick


def scheduler(computed, drafts=(), *, new=False):
    state=NS(req_id='r',prompt_token_ids=[1,2,3,4],num_computed_tokens=computed)
    return NS(scheduled_new_reqs=[state] if new else [],
        scheduled_cached_reqs=NS(req_ids=[] if new else ['r'],
            num_computed_tokens=[] if new else [computed]),
        num_scheduled_tokens={'r':2 if computed<4 else 1+len(drafts)},
        scheduled_spec_decode_tokens={'r':list(drafts)},num_invalid_spec_tokens={}),{'r':state}


def sampled(c, tokens, *, spec=False, graph=True, draft=True, terminal=False):
    c.sampler_finished(spec)
    c.bookkeeping_finished((None,[tokens],{},['r'],{'r':0},[]))
    if draft:
        c.draft_finished(k=5,output_shape=[1,5],runtime_graph=graph,method='dspark')
    c.sample_finished(terminal_device_synced=terminal)


def receipt(rank, *, graph=True):
    c=RoundCounters(rank,now=Clock());c.begin('pair')
    so,states=scheduler(0,new=True);c.execute_begin(so,states);sampled(c,[],draft=False)
    so,states=scheduler(2);c.execute_begin(so,states);sampled(c,[10],graph=graph)
    so,states=scheduler(4,[20,21,22,23,24]);c.execute_begin(so,states);sampled(c,[20,21,30],spec=True,graph=graph)
    so,states=scheduler(7,[22,23,24,25,26]);c.execute_begin(so,states);sampled(c,[22,23,24,31],spec=True,graph=graph,terminal=True)
    return c.finish()


class CounterContract(unittest.TestCase):
    def test_actual_rounds_and_budget_clip(self):
        rows=[receipt(i) for i in range(8)]
        m=summarize(rows,[10,20,21,30,22,23],2_000_000_000,clock_scope(),draft_graph=True)
        self.assertEqual((m['decoded_tokens'],m['rounds'],m['A']),(5,2,2.5))
        self.assertEqual(m['rank_counters'][0]['raw_decode_tokens'],7)
        self.assertEqual(m['rank_counters'][0]['terminal_tokens_trimmed'],2)
        self.assertEqual(m['rank_counters'][0]['draft_calls'],3)
        self.assertEqual(m['rank_counters'][0]['verified_draft_tokens'],10)
        self.assertEqual(m['rank_counters'][0]['accepted_draft_tokens_raw'],5)
        self.assertEqual(m['rank_counters'][0]['accepted_draft_tokens_used'],4)
        self.assertAlmostEqual(m['ms_per_step'],1000*m['A']/m['tokens_per_second'])
        self.assertFalse(m['performance_receipt_granted'])

    def test_no_warmup_capture_profile_counter_window(self):
        c=RoundCounters(0,now=Clock())
        so,states=scheduler(0,new=True)
        c.execute_begin(so,states);c.sampler_finished(False)
        c.bookkeeping_finished(('irrelevant',));c.sample_finished()
        self.assertEqual(c.requests,{})

    def test_worker_client_clock_mismatch_rejected(self):
        rows=[receipt(i) for i in range(8)];rows[3]['clock_scope']['boot_id']='other'
        with self.assertRaisesRegex(ValueError,'clock namespaces'):
            summarize(rows,[10,20,21,30,22,23],2_000_000_000,clock_scope(),draft_graph=True)

    def test_rank_divergence_rejected(self):
        rows=[receipt(i) for i in range(8)]
        rows[7]['requests']['r']['rounds'][0]['scheduled_draft_token_ids'][0]=999
        with self.assertRaisesRegex(ValueError,'TP ranks disagree'):
            summarize(rows,[10,20,21,30,22,23],2_000_000_000,clock_scope(),draft_graph=True)

    def test_eager_draft_does_not_pass_graph_on(self):
        rows=[receipt(i,graph=False) for i in range(8)]
        with self.assertRaisesRegex(ValueError,'used eager'):
            summarize(rows,[10,20,21,30,22,23],2_000_000_000,clock_scope(),draft_graph=True)

    def test_device_or_placeholder_tokens_rejected(self):
        c=RoundCounters(0,now=Clock());c.begin('x')
        so,states=scheduler(0,new=True);c.execute_begin(so,states)
        with self.assertRaisesRegex(ValueError,'CPU list'):
            c.bookkeeping_finished((None,[[-1]],{},['r'],{'r':0},[]))

    def test_missing_sampler_proof_rejected(self):
        c=RoundCounters(0,now=Clock());c.begin('x')
        so,states=scheduler(0,new=True);c.execute_begin(so,states)
        c.bookkeeping_finished((None,[[10]],{},['r'],{'r':0},[]))
        with self.assertRaisesRegex(ValueError,'did not finish'):
            c.sample_finished()

    def test_scheduled_drafts_require_rejection_sampler(self):
        c=RoundCounters(0,now=Clock());c.begin('x')
        so,states=scheduler(0,new=True);c.execute_begin(so,states);sampled(c,[10])
        so,states=scheduler(4,[20,21,22,23,24]);c.execute_begin(so,states)
        c.sampler_finished(False);c.bookkeeping_finished((None,[[20]],{},['r'],{'r':0},[]))
        with self.assertRaisesRegex(ValueError,'rejection sampler'):
            c.sample_finished()

    def test_double_sampler_rejected(self):
        c=RoundCounters(0,now=Clock());c.begin('x')
        so,states=scheduler(0,new=True);c.execute_begin(so,states);c.sampler_finished(False)
        with self.assertRaisesRegex(RuntimeError,'counted twice'):c.sampler_finished(False)

    def test_output_trim_cannot_erase_a_whole_round(self):
        rows=[receipt(i) for i in range(8)]
        with self.assertRaisesRegex(ValueError,'more than the final round'):
            summarize(rows,[10,20,21],2_000_000_000,clock_scope(),draft_graph=True)

    def test_frontend_completion_must_follow_native_sample(self):
        rows=[receipt(i) for i in range(8)]
        with self.assertRaisesRegex(ValueError,'precedes'):
            summarize(rows,[10,20,21,30,22,23],1,clock_scope(),draft_graph=True)

    def test_missing_final_device_fence_rejected(self):
        rows=[receipt(i) for i in range(8)]
        rows[0]['requests']['r']['rounds'][-1]['terminal_device_synced']=False
        with self.assertRaisesRegex(ValueError,'completion fence'):
            summarize(rows,[10,20,21,30,22,23],2_000_000_000,clock_scope(),draft_graph=True)

    def test_async_adapter_fails_before_sdk_import(self):
        with self.assertRaisesRegex(ValueError,'Async'):
            install_runner(NS(use_async_scheduling=True),0,audit=False)

    def test_legacy_hot_eager_flag_rejected(self):
        with self.assertRaisesRegex(ValueError,'forces eager'):
            validate_hot_flags('DRAFT_FORCE_EAGER=1\n')

    def test_native_configuration_preserves_formal_contract(self):
        kw=model_kwargs('/formal',5,audit=True,draft_graph=True)
        self.assertEqual(kw['speculative_config'],{'method':'dspark','num_speculative_tokens':5,'enforce_eager':False})
        self.assertEqual(kw['compilation_config']['cudagraph_capture_sizes'],[6])
        self.assertTrue(kw['additional_config']['enable_engram'])
        self.assertEqual(kw['load_format'],'auto')
        env=startup_env('/out',audit=True,draft_graph=True)
        self.assertEqual(env['DSPARK_GRAPH_CAPTURE_METADATA'],'1')
        self.assertEqual(env['STACK_TINY_PROFILE'],'0')
        self.assertEqual(env['STACK_QKV_MERGE_ENABLED'],'0')

    def test_bank_uses_retained_native_modules_after_wrapping(self):
        # Neither current wrapper exposes .model: this is the actual tiny v4
        # failure shape. The topology must come from the load-time references.
        wrapper=object()
        counter=NS(rank=0,sources={},audit=False,runtime_graph_selector=lambda d:True)
        config=NS(n_routed_experts=384,num_experts_per_tok=6,engram_layer_ids=[1,14])
        runner=NS(model=wrapper,_formal_dspark_counter=counter,
            _formal_dspark_native_target=NS(),_formal_dspark_native_backbone=NS(layers=list(range(40))),
            _formal_dspark_native_draft_model=NS(),
            _formal_dspark_native_draft_backbone=NS(layers={'40':None,'41':None,'42':None},target_layer_ids=[37,38,39]),
            drafter=NS(model=wrapper,attn_layer_names=['draft.40'],use_cuda_graph=True),
            use_async_scheduling=False,compilation_config=NS(static_forward_context={}),
            vllm_config=NS(model_config=NS(hf_text_config=config),
                speculative_config=NS(method='dspark',num_speculative_tokens=5),load_config=NS(load_format='auto')))
        result=bank(NS(model_runner=runner))
        self.assertEqual(result['main_layers'],40)
        self.assertEqual(result['draft_layer_ids'],[40,41,42])
        self.assertEqual(result['actual_draft_target_layer_ids'],[37,38,39])
        self.assertFalse(result['cache_metadata_is_device_consumer_proof'])


if __name__=='__main__':unittest.main()
