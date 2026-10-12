"""Strict 100/100 and 23/23 with source-derived original scoring helpers.

No evaluator imports: acc_eval executes its client at module import. Only its
pure function AST and the frozen Vision case literals are loaded here.
"""
import ast
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
GUARD_PATH=HERE/'evidence/evaluator_sources.json'
CANONICAL_GSM_SHA={'train':'17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465',
 'test':'3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14'}


def source_functions(record,names,namespace):
 source=record['source'];tree=ast.parse(source)
 nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
 if {n.name for n in nodes}!=set(names):raise ValueError('Original pure scoring functions are missing')
 for node in ast.walk(ast.Module(body=nodes,type_ignores=[])):
  if isinstance(node,ast.FunctionDef):
   node.returns=None
   for arg in [*node.args.args,*node.args.kwonlyargs]:arg.annotation=None
 scope=dict(namespace)
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),record['path'],'exec'),scope)
 return scope


def helpers():
 records=json.loads(GUARD_PATH.read_text())['files']
 gsm=source_functions(records['gsm'],('extract_gsm8k','norm_num'),
  {'re':re,'NUM':re.compile(r'-?[\d,]*\.?\d+')})
 vision=source_functions(records['vision'],('norm','is_garbled','score_case'),{'re':re})
 tree=ast.parse(records['vision']['source'])
 node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CASES' for t in n.targets))
 cases=[]
 for call in node.value.elts:
  if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!='dict' or call.args:
   raise ValueError('Original Vision case schema changed')
  cases.append({kw.arg:ast.literal_eval(kw.value) for kw in call.keywords})
 if len(cases)!=23:raise ValueError('Expected original 20 image cases and 3 negatives')
 return gsm,vision,cases


def check_gsm100(document,expected_model,expected_gold,expected_split_sha):
 score,_,_=helpers();summary=document['summary'];rows=document['results']
 if (summary.get('served_model')!=expected_model or summary.get('task')!='gsm8k' or summary.get('mode')!='chat' or
  summary.get('n')!=100 or summary.get('correct')!=100 or summary.get('empty')!=0 or summary.get('errors')!=0 or
  summary.get('gsm8k_jsonl_sha256')!=expected_split_sha or len(rows)!=100 or len(expected_gold)!=100):
  raise ValueError('GSM requires exact same-source/model 100/100 with zero empty/error')
 wanted=['gsm8k-'+str(i) for i in range(100)]
 if [row.get('idx') for row in rows]!=wanted:raise ValueError('GSM rows are not the canonical first100 sequence')
 for row,gold in zip(rows,expected_gold):
  if (row.get('ok') is not True or row.get('err') is not None or not isinstance(row.get('raw'),str) or not row['raw'].strip() or
   score['norm_num'](row.get('gold'))!=score['norm_num'](gold) or
   score['norm_num'](score['extract_gsm8k'](row['raw']))!=score['norm_num'](gold)):
   raise ValueError('GSM raw answer or canonical gold disagrees with its claimed pass')
 return {'strict_GSM100_passed':True,'n':100,'correct':100,'empty':0,'errors':0}


def canonical_gold(data_root):
 data_root=Path(data_root);score,_,_=helpers()
 splits={}
 for split in ('train','test'):
  path=data_root/(split+'.jsonl');splits[split]=hashlib.sha256(path.read_bytes()).hexdigest()
 if splits!=CANONICAL_GSM_SHA:raise ValueError('Canonical official GSM train/test SHA changed')
 rows=[json.loads(line) for line in (data_root/'test.jsonl').read_text().splitlines() if line.strip()]
 if len(rows)<100:raise ValueError('Canonical GSM test has fewer than100 records')
 return [score['extract_gsm8k'](row['answer']) for row in rows[:100]],splits


def check_vision23(document,expected_model):
 _,score,cases=helpers();summary=document['summary'];rows=document['results']
 if (summary.get('model')!=expected_model or summary.get('cases')!=23 or summary.get('passed')!=23 or
  summary.get('empty_or_garbled')!=0 or summary.get('errors')!=0 or summary.get('verdict')!='PASS' or
  summary.get('criteria',{}).get('mandatory_negative_ok') is not True or len(rows)!=23):
  raise ValueError('Vision requires strict23/23, raw PASS and zero empty/error/mandatory failure')
 if [row.get('id') for row in rows]!=[case['id'] for case in cases]:raise ValueError('Vision is not the original complete23-case sequence')
 for row,case in zip(rows,cases):
  if (row.get('image')!=case.get('image') or row.get('question')!=case['question'] or
   row.get('passed') is not True or row.get('error') or score['score_case'](case,row.get('answer'))[0] is not True):
   raise ValueError('Vision raw answer/case owner does not satisfy its original scoring rule')
 return {'strict_Vision23_passed':True,'cases':23,'passed':23,'empty_or_garbled':0,'errors':0,'mandatory_negatives_passed':True}
