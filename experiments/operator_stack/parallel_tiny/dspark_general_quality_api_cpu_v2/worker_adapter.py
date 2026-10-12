"""Deferred worker process only; original candidate worker remains the base."""
import importlib
import inspect
import os

from parallel_tiny.dspark_api_counter_bridge_cpu.runtime_guards import load_frozen_counter,sha256_file
from .contract import COUNTER_SHA,async_arm,counter_sha


def candidate_base():
 name=os.environ['DSPARK_QUALITY_BASE_WORKER']
 if name.startswith('parallel_tiny.dspark_general_quality_api_cpu.'):
  raise ValueError('Recursive quality worker class')
 module,cls=name.rsplit('.',1)
 return getattr(importlib.import_module(module),cls)


Base=candidate_base()


class GeneralQualityDSparkWorker(Base):
 def load_model(self,*args,**kwargs):
  frozen=load_frozen_counter()
  result=super().load_model(*args,**kwargs)
  runner=self.model_runner
  if not hasattr(runner,'_formal_dspark_counter'):
   from vllm.distributed import get_tensor_model_parallel_rank
   frozen.install_runner(runner,get_tensor_model_parallel_rank(),audit=False)
  self.dspark_general_quality_v1_describe()
  return result

 def dspark_general_quality_v1_describe(self):
  frozen=load_frozen_counter();runner=self.model_runner;counter=runner._formal_dspark_counter
  if type(counter) is not frozen.RoundCounters or counter.tag is not None or counter.pending is not None or counter.audit:
   raise ValueError('General quality counter must remain original, inactive and audit-OFF')
  if getattr(counter,'async_mode',False) is not async_arm():
   raise ValueError('Installed counter async arm differs from the quality async gate')
  spec=self.vllm_config.speculative_config;text=self.vllm_config.model_config.hf_text_config
  hf=self.vllm_config.model_config.hf_config
  if (spec is None or spec.method!='dspark' or spec.num_speculative_tokens!=5 or
   text.num_hidden_layers!=40 or text.hidden_size!=5120 or text.n_routed_experts!=384 or
   text.num_experts_per_tok!=6 or self.vllm_config.load_config.load_format=='dummy' or
   not getattr(hf,'vision_config',None) or
   bool(self.vllm_config.scheduler_config.async_scheduling) is not async_arm() or
   os.environ.get('HCCL_DETERMINISTIC')!='strict' or os.environ.get('STACK_FP32_DECODE_REDUCTION')!='0'):
   raise ValueError('Actual worker differs from original full40/vision/DSpark5/strict quality config')
  if async_arm() and (os.environ.get('DSPARK_ASYNC_COUNTER')!='1' or os.environ.get('DSPARK_ASYNC_SCHEDULING')!='1'):
   raise ValueError('Async quality worker requires the deferred async scheduler environment')
  return {'rank':counter.rank,'counter_sha256':counter_sha(),'counter_active':False,'audit':False,
   'scheduler_async':bool(self.vllm_config.scheduler_config.async_scheduling),
   'speculative_method':spec.method,'num_speculative_tokens':spec.num_speculative_tokens,
   'full40_config_observed':True,'vision_config_retained':True,'load_format':self.vllm_config.load_config.load_format,
   'original_candidate_base_worker':os.environ['DSPARK_QUALITY_BASE_WORKER'],
   'base_worker_source_path':inspect.getfile(Base),'base_worker_source_sha256':sha256_file(inspect.getfile(Base)),
   'native_counter_sources':counter.sources,'counter_retrieval_does_not_open_window':True,
   'actual_device_consumer_proof':False,'formal_receipt_granted':False}
