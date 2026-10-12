"""Stdlib-only candidate/identity/scoring/lifecycle controls; no HTTP/NPU."""
import asyncio
from contextlib import asynccontextmanager
import copy
import json
from pathlib import Path
import sys
import unittest

from parallel_tiny.dspark_formal_prep.contract import model_kwargs,startup_env
from .bootstrap import observed_native_lifespan,identity_from_selection
from .contract import (ASYNC_ARM_ENV,ASYNC_COUNTER_SHA,COUNTER_SHA,async_arm,counter_sha,
 false_claims,fingerprint,make_plan,validate_candidate,verify_identity,verify_rank_descriptions)
from .scoring import helpers,check_gsm100,check_vision23

HERE=Path(__file__).resolve().parent
NEGATIVES=[]


def candidate():
 env=startup_env('/deferred/candidate/native',audit=False,draft_graph=False)
 return {'schema':'CPU_fixture_candidate_contract_v1','candidate_id':'CPU_fixture_native',
  'source_root':'/deferred/candidate','source_sha256':{'native.py':'1'*64},
  'model_kwargs':model_kwargs('/deferred/original/full_checkpoint',5,audit=False,draft_graph=False),
  'checkpoint_config':{'path':'/deferred/original/full_checkpoint/config.json','sha256':'2'*64},
  'sdk_source_manifest_sha256':'3'*64,'sdk_pair_provenance':{'CPU_fixture_only':True},
  'audit_enabled':False,'startup_env':env,'startup_env_fingerprint':fingerprint(env),
  'CPU_fixture_only':True,'actual_flags':false_claims()}


def async_candidate():
 import os
 from unittest import mock
 env=startup_env('/deferred/candidate/native',audit=False,draft_graph=True)
 env.update(DSPARK_ASYNC_COUNTER='1',DSPARK_ASYNC_SCHEDULING='1',DSPARK_QUALITY_ASYNC_ARM='1')
 model=model_kwargs('/deferred/original/full_checkpoint',5,audit=False,draft_graph=True)
 model['async_scheduling']=True
 value=candidate()
 value.update(candidate_id='CPU_fixture_native_async',model_kwargs=model,startup_env=env,
  startup_env_fingerprint=fingerprint(env),source_sha256={'native.py':'1'*64})
 return value


def selection():
 plan=make_plan(candidate(),'CPU_fixture_quality',18574,'/deferred/quality','/deferred/frozen_v4.py')
 plan.update(service_started=True,ASGI_startup_completed=True,service_instance_id='CPU_fixture_nonce')
 ranks=[{'rank':rank,'counter_sha256':COUNTER_SHA,'counter_active':False,'audit':False,
  'speculative_method':'dspark','num_speculative_tokens':5} for rank in range(8)]
 return plan,identity_from_selection(plan,ranks)


def score_documents():
 _,_,cases=helpers();model='dsv41-dspark-quality-CPU_fixture'
 gold=[str(i+100) for i in range(100)]
 splits={'train':'4'*64,'test':'5'*64}
 gsm={'summary':{'served_model':model,'task':'gsm8k','mode':'chat','n':100,'correct':100,
  'empty':0,'errors':0,'gsm8k_jsonl_sha256':splits},
  'results':[{'idx':'gsm8k-'+str(i),'gold':value,'pred':value,'raw':'#### '+value,'ok':True,'err':None}
   for i,value in enumerate(gold)]}
 vision={'summary':{'model':model,'cases':23,'passed':23,'empty_or_garbled':0,'errors':0,
  'verdict':'PASS','criteria':{'mandatory_negative_ok':True}},
  'results':[{'id':case['id'],'image':case.get('image'),'question':case['question'],
   'answer':case['accept'][0],'passed':True,'mandatory':bool(case.get('mandatory'))} for case in cases]}
 return model,gold,splits,gsm,vision


class ContractControls(unittest.TestCase):
 def reject(self,label,callback):
  with self.assertRaises(ValueError):callback()
  NEGATIVES.append(label)

 def test_plan_preserves_same_candidate_config_and_environment(self):
  c=candidate();before=copy.deepcopy(c);plan=make_plan(c,'new_quality',18574,'/new/job','/frozen_v4.py')
  self.assertEqual(c,before)
  original=dict(c['model_kwargs']);actual=dict(plan['model_kwargs'])
  base=original.pop('worker_cls');actual.pop('worker_cls')
  self.assertEqual(actual,original);self.assertEqual(plan['environment']['DSPARK_QUALITY_BASE_WORKER'],base)
  for key,value in c['startup_env'].items():
   if key not in ('STACK_DSPARK_FORMAL_OUTPUT','VLLM_CACHE_ROOT'):self.assertEqual(plan['environment'][key],value)
  self.assertTrue(plan['native_AsyncLLM']);self.assertFalse(plan['private_armed_window_adapter'])
  self.assertFalse(plan['private_route_whitelist']);self.assertFalse(plan['service_started'])
  self.assertTrue(all(plan[k] is False for k in false_claims()))

 def test_candidate_strict_DSpark_wholemodel_audit_source_controls(self):
  variations={'audit_true':lambda c:c.update(audit_enabled=True),
   'A1_no_spec':lambda c:c['model_kwargs'].update(speculative_config=None),
   'wrong_K':lambda c:c['model_kwargs']['speculative_config'].update(num_speculative_tokens=7),
   'dummy_weights':lambda c:c['model_kwargs'].update(load_format='dummy'),
   'different_tokenizer_owner':lambda c:c['model_kwargs'].update(tokenizer='/other/model'),
   'route_audit_enabled':lambda c:c['model_kwargs'].update(enable_return_routed_experts=True),
   'fp32_reduction_changed':lambda c:c['startup_env'].update(STACK_FP32_DECODE_REDUCTION='1'),
   'non_strict_HCCL':lambda c:c['startup_env'].update(HCCL_DETERMINISTIC='0'),
   'missing_source':lambda c:c.update(source_sha256={})}
  for label,change in variations.items():
   c=candidate();change(c)
   self.reject(label,lambda:validate_candidate(c))

 def test_identity_is_new_general_service_not_private_or_active_counter(self):
  s,i=selection();self.assertTrue(verify_identity(i,s,s['served_model']))
  variants={'private_armed':lambda x:x.update(private_armed_window_adapter=True),
   'private_whitelist':lambda x:x.update(private_route_whitelist=True),
   'another_nonce':lambda x:x.update(service_instance_id='old_process'),
   'another_candidate':lambda x:x.update(candidate_fingerprint='old_candidate'),
   'rank_missing':lambda x:x['rank_descriptions'].pop(),
   'active_v4_window':lambda x:x['rank_descriptions'][0].update(counter_active=True),
   'counter_not_v4':lambda x:x['rank_descriptions'][1].update(counter_sha256='0'*64)}
  for label,change in variants.items():
   altered=copy.deepcopy(i);change(altered)
   self.reject(label,lambda:verify_identity(altered,s,s['served_model']))
  early=dict(s,service_started=False)
  self.reject('engine_init_before_HTTP_is_not_started',lambda:verify_identity(i,early,s['served_model']))

 def test_async_arm_gate_accepts_only_matching_candidate_and_worker_arm(self):
  import os
  from unittest import mock
  # Default (sync) gate: the async candidate must be refused, and the async
  # counter SHA must never leak into a synchronous plan.
  self.assertFalse(async_arm());self.assertEqual(counter_sha(),COUNTER_SHA)
  self.reject('async_candidate_under_sync_gate',lambda:validate_candidate(async_candidate()))
  # Async gate: the matching candidate passes, the synchronous one is refused,
  # and the plan reports the async counter algorithm.
  with mock.patch.dict(os.environ,{ASYNC_ARM_ENV:'1'},clear=False):
   self.assertTrue(async_arm());self.assertEqual(counter_sha(),ASYNC_COUNTER_SHA)
   async_candidate_value=async_candidate()
   self.assertIsNotNone(validate_candidate(async_candidate_value))
   self.reject('sync_candidate_under_async_gate',lambda:validate_candidate(candidate()))
   plan=make_plan(async_candidate_value,'new_quality_async',18574,'/new/job','/frozen_async_counter.py')
   self.assertEqual(plan['counter_algorithm_sha256'],ASYNC_COUNTER_SHA)
   self.assertTrue(plan['async_scheduler_arm'])
   ranks=[{'rank':rank,'counter_sha256':ASYNC_COUNTER_SHA,'counter_active':False,'audit':False,
    'speculative_method':'dspark','num_speculative_tokens':5,'scheduler_async':True} for rank in range(8)]
   self.assertTrue(verify_rank_descriptions(ranks))
   wrong=copy.deepcopy(ranks);wrong[3]['scheduler_async']=False
   self.reject('worker_arm_differs_from_async_gate',lambda:verify_rank_descriptions(wrong))
   missing_env=async_candidate();missing_env['startup_env'].pop('DSPARK_ASYNC_SCHEDULING')
   missing_env['startup_env_fingerprint']=fingerprint(missing_env['startup_env'])
   # The strict environment guard must still see the exact original basis keys.
   NEGATIVES.append('async_candidate_missing_async_scheduler_env')
   try:validate_candidate(missing_env)
   except ValueError:pass
   else:raise AssertionError('missing async scheduler env was accepted')

 def test_original_scoring_accepts_only_synthetic_positive_fixtures(self):
  model,gold,splits,gsm,vision=score_documents()
  self.assertTrue(check_gsm100(gsm,model,gold,splits)['strict_GSM100_passed'])
  self.assertTrue(check_vision23(vision,model)['strict_Vision23_passed'])

 def test_GSM_zero_exit_summary_and_resume_labels_do_not_prove100(self):
  model,gold,splits,gsm,_=score_documents()
  changes={'GSM99':lambda x:x['summary'].update(correct=99),
   'GSM_old_model':lambda x:x['summary'].update(served_model='A1_old_model'),
   'GSM_other_dataset':lambda x:x['summary']['gsm8k_jsonl_sha256'].update(test='0'*64),
   'GSM_false_ok_raw_wrong':lambda x:x['results'][0].update(raw='#### 99999'),
   'GSM_duplicate_resume_row':lambda x:x['results'][1].update(idx='gsm8k-0'),
   'GSM_empty':lambda x:x['results'][2].update(raw=''),
   'GSM_wrong_canonical_gold':lambda x:x['results'][3].update(gold='999'),
   'GSM_error_even_with_ok':lambda x:x['results'][4].update(err='HTTP failure')}
  for label,change in changes.items():
   doc=copy.deepcopy(gsm);change(doc)
   self.reject(label,lambda:check_gsm100(doc,model,gold,splits))

 def test_Vision_default80_and19_gate_or_meanA_never_pass_strict23(self):
  model,_,_,_,vision=score_documents()
  changes={'Vision19_default_gate':lambda x:x['summary'].update(passed=19),
   'Vision_old_model':lambda x:x['summary'].update(model='old_A1'),
   'Vision_wrong_case':lambda x:x['results'][0].update(question='another question'),
   'Vision_claimed_pass_wrong_answer':lambda x:x['results'][0].update(answer='玉米'),
   'Vision_negative_swap_fails':lambda x:x['results'][-2].update(answer='是的'),
   'Vision_error':lambda x:x['summary'].update(errors=1),
   'Vision_empty':lambda x:x['results'][2].update(answer=''),
   'Vision_mandatory_false':lambda x:x['summary']['criteria'].update(mandatory_negative_ok=False)}
  for label,change in changes.items():
   doc=copy.deepcopy(vision);change(doc)
   doc['summary']['dspark']={'mean_acceptance_length':9.99}
   self.reject(label,lambda:check_vision23(doc,model))


class LifecycleControls(unittest.IsolatedAsyncioTestCase):
 async def test_native_lifespan_records_real_start_and_stops_after_body(self):
  events=[];app=object();state={'service_started':False}
  @asynccontextmanager
  async def original(actual):
   self.assertIs(actual,app);events.append('original_startup_completed')
   try:yield
   finally:events.append('original_shutdown')
  def write(value):events.append('selection_started' if value['service_started'] else 'selection_stopped')
  async with observed_native_lifespan(app,original,state,write):
   self.assertTrue(state['service_started']);self.assertTrue(state['ASGI_startup_completed'])
   self.assertFalse(state['HTTP_readiness_verified']);events.append('actual_ASGI_running')
  self.assertFalse(state['service_started'])
  self.assertEqual(events,['original_startup_completed','selection_started','actual_ASGI_running','selection_stopped','original_shutdown'])

 async def test_failure_before_native_startup_never_sets_started(self):
  state={'service_started':False}
  @asynccontextmanager
  async def original(app):raise RuntimeError('CPU_fake_startup_failure');yield
  with self.assertRaises(RuntimeError):
   async with observed_native_lifespan(object(),original,state,lambda value:None):pass
  self.assertFalse(state['service_started'])
  NEGATIVES.append('failed_native_ASGI_startup_no_false_started')


def main():
 suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(ContractControls),
  unittest.defaultTestLoader.loadTestsFromTestCase(LifecycleControls)])
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 modules=[name for name in sys.modules if name in ('torch','torch_npu','vllm','acl','torchair')]
 if modules:raise ValueError('Stdlib CPU controls unexpectedly loaded SDK/backend: '+repr(modules))
 report={'CPU_only':True,'CPU_mock_candidate_identity_scores_and_ASGI':True,
  'tests':result.testsRun,'passed':result.wasSuccessful(),'negative_cases':NEGATIVES,
  'negative_case_count':len(NEGATIVES),'NPU_calls':0,'HTTP_requests':0,'services_started':0,
  'backend_modules_imported':modules,'original_evaluator_scripts_executed':False,
  'original_pure_scoring_function_AST_executed':True,'actual_flags':false_claims()}
 (HERE/'evidence/CPU_validation.json').write_text(json.dumps(report,indent=2)+'\n')
 if not result.wasSuccessful():sys.exit(1)


if __name__=='__main__':main()
