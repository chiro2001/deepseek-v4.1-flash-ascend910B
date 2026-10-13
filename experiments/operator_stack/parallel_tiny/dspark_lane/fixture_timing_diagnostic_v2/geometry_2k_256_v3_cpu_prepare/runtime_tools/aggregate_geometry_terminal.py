"""Read-only CPU aggregation of remote terminal geometry receipts.

The full step files stay on the host. This tool reads JSON and the frozen pure
CPU contract only; it never imports the worker or accesses an accelerator.
"""
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parents[1]
JOB = "dspark_lane_20261011_c4_graph_geometry2k256_v3"
REMOTE = r'''
import base64,gzip,hashlib,importlib.util,json,pathlib,re,time
p=pathlib.Path('/home/l00886679/projects/dsv41-tiny-prof-20261009')/job
assert (p/'run.exit').exists(), 'Aggregation requires a terminal job'
exit_code=int((p/'run.exit').read_text().strip())
preflight=json.loads((p/'preflight.json').read_bytes())
remote_source_files=[]
for name,expected in sorted(preflight['source_files'].items()):
 data=(p/'src'/name).read_bytes()
 actual={'name':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 assert actual['bytes']==expected['bytes'] and actual['sha256']==expected['sha256'], name
 remote_source_files.append(actual)
contract_path=p/'src/geometry_contract.py'
assert hashlib.sha256(contract_path.read_bytes()).hexdigest()==contract_sha256
spec=importlib.util.spec_from_file_location('frozen_geometry_cpu_contract',contract_path)
contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
requests_original=(p/'results/requests.json').read_bytes()
records=json.loads(requests_original)
by_tag={r['tag']:r for r in records}
assert len(by_tag)==len(records)
step_files={}
inventory=[]
for path in sorted((p/'results/audit').glob('geometry2k_*_step_*.json')):
 match=re.fullmatch(r'geometry2k_(.*)_step_([0-9]+)\.json',path.name)
 assert match, path.name
 tag,index=match[1],int(match[2])
 step_files.setdefault(tag,[]).append((index,path))
all_inventory_digest=hashlib.sha256()
requests=[]
global_on_pointers={}
for sequence,record in enumerate(records,1):
 raw_path=p/'results'/f'raw_{sequence:03d}.json'
 raw=json.loads(raw_path.read_bytes())
 assert raw['token_ids']==record['token_ids'] and raw['arm']==record['arm'] and raw['core_id']==record['core_id']
 rows=raw['counter']['rounds']
 derived=contract.validate_rounds(rows,record['token_ids'],record['core_id'])
 helper=contract.validate_native_helpers(raw,record['arm'])
 assert all(record['independent_2048_256_geometry'][key]==value for key,value in (derived|helper).items())
 assert record['performance_evaluated'] is False and record['formal_tp8_claim'] is False
 prefill=[r for r in rows if not r['is_decode']]
 decode=[r for r in rows if r['is_decode']]
 files=sorted(step_files.get(record['tag'],[]))
 if not record['audit']:
  assert files==[] and record['all_step_cache_geometry_receipts']==0
  requests.append({'tag':record['tag'],'arm':record['arm'],'seed':record['seed'],'audit':False,
   'decode_rounds':len(decode),'prefill_rounds':len(prefill),'geometry_steps':0,
   'native_helpers':helper['native_helper_calls'],'raw_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest()})
  continue
 geometry_path=p/'results'/f'geometry_{sequence:03d}.json'
 geometry=json.loads(geometry_path.read_bytes())
 assert geometry['tag']==record['tag']
 assert len(files)==len(decode)==record['all_step_cache_geometry_receipts']==geometry['steps']
 assert [i for i,_ in files]==list(range(len(prefill)+1,len(rows)+1))
 digest=hashlib.sha256()
 raw_inventory_digest=hashlib.sha256()
 group_stats={role:{'pointer_values':{key:set() for key in ('cache_ptr','native_indices_ptr','native_table_ptr','live_table_ptr')},
  'gid_values':set(),'cache_block_values':set(),'seq_len_range':[4096,-1],
  'logical_mapping_changes':0,'changed_same_logical_column_count':0,'block_column_range':[4096,-1],
  'selected_cache_values':0,'value_hashes':set()} for role in ('draft.0','draft.1','draft.2')}
 previous_mapping={}
 first_point=last_point=None
 total_bytes=0
 for (index,path),scheduled in zip(files,decode):
  data=path.read_bytes();receipts=json.loads(data)
  assert set(receipts)==set(group_stats)
  positions=receipts['draft.0']['positions']
  for role,receipt in receipts.items():
   assert receipt['role']==role and receipt['positions']==positions
   assert receipt['actual_scheduled_round']==scheduled
   assert receipt['capture_reference']==(record['arm']=='on')
   assert receipt['kernel_block_size']==128 and receipt['indices_width']==256
   assert receipt['selected_slots_count_with_duplicates']==665 and receipt['native_padding_entries']==615
   assert receipt['finite'] is True and receipt['source_cache_unchanged'] is True
   assert receipt['finite_filter_used'] is False and receipt['unretained_prefix_scanned'] is False
   contract.validate_query_positions(positions,scheduled,receipt['seq_len'])
   logical=receipt['logical_positions']
   assert logical==list(range(receipt['seq_len']-5-128,receipt['seq_len']))
   assert len(logical)==133
   selected=receipt['native_selected_slot_ids_with_duplicates']
   slots=selected[:133]
   assert selected==slots*5
   mapping={}
   for position,slot in zip(logical,slots):
    column,offset=divmod(position,128)
    block,physical_offset=divmod(slot,128)
    assert physical_offset==offset and 0<block<receipt['cache_blocks']
    assert column not in mapping or mapping[column]==block
    mapping[column]=block
   assert sorted(mapping)==receipt['logical_block_columns_used']
   table=[0]*(max(mapping)+1)
   for column,block in mapping.items():table[column]=block
   expected=contract.validate_reader_indices([slots+[-1]*123]*5,table,receipt['seq_len'],128,receipt['cache_blocks'],0)
   assert all(receipt[key]==value for key,value in expected.items())
   stats=group_stats[role]
   stats['gid_values'].add(receipt['gid']);stats['cache_block_values'].add(receipt['cache_blocks'])
   stats['seq_len_range'][0]=min(stats['seq_len_range'][0],receipt['seq_len'])
   stats['seq_len_range'][1]=max(stats['seq_len_range'][1],receipt['seq_len'])
   if role in previous_mapping:
    stats['logical_mapping_changes']+=mapping!=previous_mapping[role]
    stats['changed_same_logical_column_count']+=sum(previous_mapping[role][column]!=block
     for column,block in mapping.items() if column in previous_mapping[role])
   previous_mapping[role]=mapping
   stats['block_column_range'][0]=min(stats['block_column_range'][0],min(mapping))
   stats['block_column_range'][1]=max(stats['block_column_range'][1],max(mapping))
   stats['selected_cache_values']+=665*512
   assert re.fullmatch('[0-9a-f]{64}',receipt['value_sha256'])
   stats['value_hashes'].add(receipt['value_sha256'])
   for key in stats['pointer_values']:
    value=receipt[key]
    assert isinstance(value,int) and value>0
    stats['pointer_values'][key].add(value)
    if record['arm']=='on':
     global_on_pointers.setdefault((role,key),set()).add(value)
  point={'per_group':receipts,'positions':positions,'independent_2048_256':True,
         'performance_evaluated':False,'capture_device_reads':False}
  digest.update(json.dumps(point,sort_keys=True).encode())
  if first_point is None:first_point=point
  last_point=point
  row={'relative_path':str(path.relative_to(p)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
  encoded=json.dumps(row,sort_keys=True).encode()+b'\n'
  raw_inventory_digest.update(encoded);inventory.append(row);total_bytes+=len(data)
 assert digest.hexdigest()==geometry['all_step_receipt_sha256']
 assert first_point==geometry['points']['first_decode'] and last_point==geometry['points']['terminal']
 compact_groups={}
 for role,stats in group_stats.items():
  pointers={key:{'count':len(values),'min':min(values),'max':max(values)} for key,values in stats.pop('pointer_values').items()}
  if record['arm']=='on':assert all(v['count']==1 for v in pointers.values())
  stats['distinct_value_hashes']=len(stats.pop('value_hashes'))
  stats['gids']=sorted(stats.pop('gid_values'));stats['cache_blocks']=sorted(stats.pop('cache_block_values'))
  compact_groups[role]=stats|{'pointers':pointers}
 requests.append({'tag':record['tag'],'arm':record['arm'],'seed':record['seed'],'audit':True,
  'decode_rounds':len(decode),'prefill_rounds':len(prefill),'geometry_steps':len(files),
  'native_helpers':helper['native_helper_calls'],'raw_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),
  'geometry_compact_sha256':hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
  'all_step_receipt_sha256':digest.hexdigest(),'reconstructed_benchmark_digest_equal':True,
  'first_and_terminal_points_equal':True,'all_step_original_bytes':total_bytes,
  'ordered_original_step_inventory_sha256':raw_inventory_digest.hexdigest(),
  'first_step_suffix':files[0][0],'last_step_suffix':files[-1][0],
  'first_positions':first_point['positions'],'terminal_positions':last_point['positions'],'groups':compact_groups})
for key,values in global_on_pointers.items():assert len(values)==1, (key,values)
for row in sorted(inventory,key=lambda row:row['relative_path']):
 all_inventory_digest.update(json.dumps(row,sort_keys=True).encode()+b'\n')
unassigned={tag:len(files) for tag,files in step_files.items() if tag not in by_tag}
if exit_code==0:assert len(records)==20 and len(step_files)==18 and not unassigned
result={'job':job,'time_unix':time.time(),'terminal':True,'exit':exit_code,
 'cpu_json_only_no_accelerator_access':True,'contract_sha256':contract_sha256,
 'all_runtime_source_files_match_preflight':True,'runtime_source_files':remote_source_files,
 'requests_completed':len(records),'audited_requests_verified':sum(r['audit'] for r in records),
 'requests_source_bytes':len(requests_original),'requests_source_sha256':hashlib.sha256(requests_original).hexdigest(),
 'request_records_without_tokens_and_logprobs':[{key:value for key,value in record.items()
  if key not in ('token_ids','logprobs')} for record in records],
 'geometry_step_files':sum(len(v) for v in step_files.values()),'verified_geometry_steps':len(inventory),
 'unassigned_terminal_step_tags':unassigned,'all_step_original_bytes':sum(row['bytes'] for row in inventory),
 'sorted_original_step_inventory_sha256':all_inventory_digest.hexdigest(),
 'all_three_groups_665_slots_615_native_padding_and_no_active_null':True,
 'all_positions_and_selected_physical_pages_validated':True,
 'live_table_equality_source_guard_completed_before_each_receipt':True,
 'live_table_values_not_independently_serialized':True,
 'all_ON_capture_binding_addresses_stable_across_requests':all(len(v)==1 for v in global_on_pointers.values()),
 'ON_addresses':[{'role':role,'field':key,'ptr':next(iter(values))} for (role,key),values in sorted(global_on_pointers.items())],
 'eager_address_stability_required':False,'remote_step_files_transferred':False,'requests':requests,
 'performance_evaluated':False,'tiny_candidate_admitted':False,'formal_candidate_admitted':False}
raw=json.dumps(result,indent=2).encode()+b'\n'
packet={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
packed=base64.b64encode(gzip.compress(json.dumps(packet).encode())).decode()
assert len(raw)<1000000 and len(packed)<900000
print(packed)
'''


def main():
    freeze = json.loads((BASE / "source_freeze_geometry2k256_v3.json").read_bytes())
    row = next(r for r in freeze["files"] if r["path"].endswith("/geometry_contract.py"))
    expected = row["sha256"]
    assert hashlib.sha256((BASE / "geometry_contract.py").read_bytes()).hexdigest() == expected
    program = "job=" + repr(JOB) + "\ncontract_sha256=" + repr(expected) + "\n" + REMOTE
    result = subprocess.run(["ssh", "-o", "ControlPath=none", "-o", "BatchMode=yes", "a3-21", "python3 -"],
                            input=program, text=True, capture_output=True, check=True)
    packed_bytes = len(result.stdout.encode())
    assert packed_bytes < 900000
    packet = json.loads(gzip.decompress(base64.b64decode(result.stdout)))
    raw = base64.b64decode(packet.pop("base64"))
    assert len(raw) == packet["bytes"] and hashlib.sha256(raw).hexdigest() == packet["sha256"]
    evidence = BASE / "runtime_evidence"
    evidence.mkdir(exist_ok=True)
    target = evidence / (JOB + "_geometry_terminal_aggregate.json")
    target.write_bytes(raw)
    packet.update({"job": JOB, "ssh_packed_bytes": packed_bytes,
                   "original_aggregate_bytes_verified": True, "remote_read_only": True})
    (evidence / (JOB + "_geometry_terminal_aggregate_receipt.json")).write_text(json.dumps(packet, indent=2) + "\n")
    data = json.loads(raw)
    print(json.dumps({key: value for key, value in data.items()
                      if key not in ("requests", "request_records_without_tokens_and_logprobs")}, indent=2))
    print(json.dumps(packet, indent=2))


if __name__ == "__main__":
    main()
