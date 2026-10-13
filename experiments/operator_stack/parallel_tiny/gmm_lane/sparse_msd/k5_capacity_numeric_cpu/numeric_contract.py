"""New capacities only; original INT8/FP32 samples and BF16 threshold retained."""
import hashlib
import numpy as np


def cases(profile):
 cap,e=profile.capacity,profile.experts
 mixed={0:1,e//2:cap//3,e-1:cap-1-cap//3}
 dense_ids=list(range(cap-1))+[e-1]
 values=[('full_last_expert',{e-1:cap}),('full_many_slots',{idx:1 for idx in dense_ids}),
         ('full_mixed_counts',mixed),('partial_unused_rows',{0:cap//2,e-1:cap//2-1})]
 result=[]
 for name,distribution in values:
  counts=np.zeros(e,dtype=np.int64)
  for expert,rows in distribution.items():counts[expert]=rows
  assert 6<int(counts.sum())<=cap
  result.append((profile.family+'_'+name,counts))
 return result


def activation(profile,index):
 levels=np.asarray([-128,-127,-65,-64,-17,-16,-15,-9,-8,-1,0,1,7,8,15,16,17,63,64,126,127],dtype=np.int8)
 r=np.arange(profile.capacity,dtype=np.int64)[:,None];k=np.arange(profile.k,dtype=np.int64)[None,:]
 x=levels[(k*13+r*7+index*5)%len(levels)].copy()
 base=np.asarray([1/256,1/128,1/64,.0125,.015625,.03125],dtype=np.float32)
 token=np.resize(base,profile.capacity).copy()
 return x,token


def array_sha(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def oracle(profile,counts):
 if counts.dtype!=np.int64 or counts.shape!=(profile.experts,) or np.any(counts<0) or int(counts.sum())>profile.capacity:
  raise ValueError('Full original valid counts tensor is required')
 frame=np.zeros(profile.frame_words,dtype=np.uint32)
 frame[3]=profile.experts;frame[4:4+4*profile.slots:4]=0xffffffff
 rows=slots=0
 for expert,count in enumerate(counts):
  if not count:continue
  frame[4+4*slots:8+4*slots]=[expert,rows,int(count),0];rows+=int(count);slots+=1
 frame[:2]=[rows,slots]
 return frame


def plan_frame(counts,profile):
 """Tolerant CPU twin of the device planner template in source_factory.

 Only the count value domain is tolerated: an invalid count sets word[2]=1
 and returns before the totals are written, exactly like the AscendC
 template. Tensor dtype/shape errors keep raising like the strict oracle.
 """
 if counts.dtype!=np.int64 or counts.shape!=(profile.experts,):
  raise ValueError('counts tensor must be int64[E]')
 frame=np.zeros(profile.frame_words,dtype=np.uint32)
 frame[3]=profile.experts;frame[4:4+4*profile.slots:4]=0xffffffff
 rows=slots=0
 for expert,count in enumerate(counts.tolist()):
  if count<0 or count>profile.capacity or rows+count>profile.capacity:
   frame[2]=1
   return frame
  if count==0:continue
  frame[4+4*slots:8+4*slots]=[expert,rows,count,0];rows+=count;slots+=1
 frame[:2]=[rows,slots]
 return frame
