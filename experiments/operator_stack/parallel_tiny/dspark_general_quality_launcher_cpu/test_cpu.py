"""CPU mocks for future file/snapshot/fresh-resource/process/readiness/stop paths."""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from parallel_tiny.dspark_general_quality_api_cpu.bootstrap import identity_from_selection
from parallel_tiny.dspark_general_quality_api_cpu.contract import COUNTER_SHA, false_claims, fingerprint, make_plan
from parallel_tiny.dspark_general_quality_api_cpu.test_cpu import candidate as fixture_candidate
from . import assets, common, evaluate_when_ready, export_candidate, launch, lifecycle, prepare_final, prepare_snapshot, resource_gate

HERE = Path(__file__).resolve().parent
NEGATIVES = []


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def device_table(process=False):
    lines = []
    for board in range(4, 8):
        lines += [f'| {board} Ascend910C | OK |',
                  f'| 0 {board*2} | 0000:00:00.0 |', f'| 1 {board*2+1} | 0000:00:00.1 |']
    if process:
        lines.append('| 4 0 | 9999 | occupied_worker |')
    return '\n'.join(lines)


class Controls(unittest.TestCase):
    def reject(self, name, callback, kinds=(ValueError, FileExistsError, FileNotFoundError)):
        with self.assertRaises(kinds):
            callback()
        NEGATIVES.append(name)

    def test_source_manifest_is_relative_pinned_and_current(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);path = root / 'a.py';path.write_text('x=1\n')
            mapping = common.source_map({'files': {'a.py': common.sha(path)}})
            self.assertEqual(common.verify_map(root, mapping), mapping)
            for name, value in [('absolute', '/x.py'), ('parent_escape', '../x.py')]:
                self.reject(name, lambda v=value: common.source_map({'files': {v: '0'*64}}))
            self.reject('missing_source_map', lambda: common.source_map({}))
            self.reject('bad_source_SHA', lambda: common.source_map({'files': {'a.py': 'bad'}}))
            self.reject('changed_source', lambda: common.verify_map(root, {'a.py': '0'*64}))
            m = root / 'manifest.json';m.write_text('{}\n')
            self.reject('changed_manifest_pin', lambda: common.read_pinned(m, '0'*64))
            outside = root.parent / ('outside_' + root.name + '.py');outside.write_text('outside\n')
            try:
                (root / 'link.py').symlink_to(outside)
                self.reject('external_source_symlink', lambda: common.verify_map(root, {'link.py': common.sha(outside)}))
            finally:
                outside.unlink()

    def test_CPU_fixture_and_inherited_grants_never_export_as_actual(self):
        c = fixture_candidate()
        self.reject('explicit_CPU_fixture', lambda: common.reject_fixture(c))
        self.reject('nested_CPU_fixture', lambda: common.reject_fixture({'sdk_pair_provenance': {'CPU_fixture_only': True}}))
        self.reject('CPU_fixture_schema', lambda: common.reject_fixture({'schema': 'CPU_fixture_candidate'}))
        self.reject('inherited_quality_grant', lambda: common.reject_fixture({'actual_flags': {'quality': True}}))
        with patch.object(export_candidate, 'read_pinned') as read:
            self.reject('fixture_export_before_any_IO', lambda: export_candidate.export(c, '/no', {}, '/no', '/no', '0'*64))
            read.assert_not_called()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'CPU_fixture_seed.json';path.write_text(json.dumps(c))
            self.reject('fixture_single_CPU_entry_blocked',lambda:prepare_final.main([
                '--candidate-seed',str(path),'--candidate-seed-sha256',common.sha(path),
                '--output',str(Path(temp)/'must_not_exist'),'--execute']))
            self.assertFalse((Path(temp)/'must_not_exist').exists())

    def test_actual_SDK_records_are_read_only_file_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'sdk.py';p.write_text('CPU_mock_SDK_source=1\n')
            manifest = {'sdk_sources': {'mock': {'path': str(p), 'sha256': common.sha(p), 'bytes': p.stat().st_size}}}
            report = common.verify_sdk_manifest(manifest)
            self.assertFalse(report['SDK_modules_imported'])
            self.assertEqual(report['NPU_calls'], 0)
            self.reject('SDK_empty_manifest', lambda: common.verify_sdk_manifest({'libraries': {}}))
            changed = copy.deepcopy(manifest);changed['sdk_sources']['mock']['sha256'] = '0'*64
            self.reject('SDK_source_changed', lambda: common.verify_sdk_manifest(changed))
            changed = copy.deepcopy(manifest);changed['sdk_sources']['mock']['bytes'] += 1
            self.reject('SDK_source_size_changed', lambda: common.verify_sdk_manifest(changed))

    def test_actual_asset_record_has_full_original_set_and_SHA(self):
        record = json.loads((HERE / 'evidence/actual_asset_paths_CPU.json').read_text())
        self.assertEqual(len(record['files']), 20)
        self.assertEqual(record['backend_modules_imported'], [])
        self.assertEqual(record['NPU_calls'], record['HTTP_requests'], 0)
        self.assertTrue(all(record.get(key) is False for key in false_claims()))
        incomplete = copy.deepcopy(record);incomplete['files'].pop('images/corn.jpeg')
        self.reject('asset_original_image_missing', lambda: assets.verify(incomplete))
        changed = copy.deepcopy(record);changed['files']['images/corn.jpeg']['sha256'] = '0'*64
        self.reject('asset_original_image_changed', lambda: assets.verify(changed))
        wrong = copy.deepcopy(record);wrong['schema'] = 'CPU_mock_assets'
        self.reject('asset_not_actual_source_schema', lambda: assets.verify(wrong))

    def test_relocation_preserves_model_and_non_source_paths(self):
        value = {'PYTHONPATH': '/old/stack:/old/operator:/sdk/path', 'model': '/models/checkpoint',
                 'other': '/oldish/path', 'paths': ['/old/up950']}
        result = common.relocate(value, '/old', '/new')
        self.assertEqual(result['PYTHONPATH'], '/new/stack:/new/operator:/sdk/path')
        self.assertEqual(result['model'], value['model'])
        self.assertEqual(result['other'], value['other'])
        self.assertEqual(result['paths'], ['/new/up950'])
        self.assertEqual(value['paths'], ['/old/up950'])

    def test_fresh_TP8_resources_require_full_empty_healthy_allocation(self):
        devices = resource_gate.resources(device_table())
        details = {chip: 'Error Code : NA\n' for chip in resource_gate.CHIPS}
        self.assertTrue(resource_gate.verify(devices, details))
        self.reject('unparseable_resource_table', lambda: resource_gate.resources('unknown table'))
        self.reject('fresh_chip_occupied', lambda: resource_gate.verify(resource_gate.resources(device_table(True)), details))
        fewer = copy.deepcopy(devices);fewer.pop(15)
        self.reject('chip_missing', lambda: resource_gate.verify(fewer, details))
        alarm = copy.deepcopy(devices);alarm[8]['health'] = 'Alarm'
        known = dict(details);known[8] = 'Error Code : 80C98001\n'
        self.reject('Alarm_without_allocated_exception', lambda: resource_gate.verify(alarm, known))
        self.assertTrue(resource_gate.verify(alarm, known, True))
        unknown = dict(known);unknown[8] = 'Error Code : OTHER\n'
        self.reject('Alarm_unknown_code', lambda: resource_gate.verify(alarm, unknown, True))
        self.reject('fresh_health_changed', lambda: resource_gate.verify(alarm, known, True, previous=devices))
        self.reject('health_code_missing', lambda: resource_gate.verify(devices, {chip: '' for chip in devices}))

    def fixture_service(self, root):
        source = root / 'source';source.mkdir();worker = source / 'base_worker.py';worker.write_text('CPU_mock_worker=1\n')
        job = root / 'quality_new';job.mkdir()
        candidate = fixture_candidate();candidate['source_root'] = str(source)
        candidate['source_sha256'] = {'base_worker.py': common.sha(worker)}
        candidate['model_kwargs']['worker_cls'] = 'CPU_mock_base.Worker'
        plan = make_plan(candidate, 'quality_new', 18574, job / 'service', source / 'counter.py')
        plan.update(service_instance_id='CPU_mock_service_nonce', pid=1002, service_started=True,
                    ASGI_startup_completed=True, engine_initialized=True)
        ranks = [{'rank': rank, 'counter_sha256': COUNTER_SHA, 'counter_active': False, 'audit': False,
                  'speculative_method': 'dspark', 'num_speculative_tokens': 5,
                  'original_candidate_base_worker': candidate['model_kwargs']['worker_cls'],
                  'base_worker_source_path': str(worker), 'base_worker_source_sha256': common.sha(worker)} for rank in range(8)]
        identity = identity_from_selection(plan, ranks)
        return source, job, candidate, plan, identity

    def test_ASGI_incomplete_never_issues_HTTP_or_becomes_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            source, job, c, s, identity = self.fixture_service(Path(temp))
            request = Mock();s['service_started'] = False
            result = evaluate_when_ready.ready_once(s, source, job, 'quality_new', 18574, c, {},
                request=request, process_reader=Mock(return_value={'CPU_mock_process': True}))
            self.assertFalse(result['ready']);request.assert_not_called()
            NEGATIVES.append('engine_initialized_is_not_ASGI_or_HTTP_ready')

    def test_owned_HTTP_identity_model_and_rank_source_are_all_required(self):
        with tempfile.TemporaryDirectory() as temp:
            source, job, c, s, identity = self.fixture_service(Path(temp))
            model = {'data': [{'id': s['served_model']}]}
            def call(i=identity, m=model):
                return evaluate_when_ready.ready_once(s, source, job, 'quality_new', 18574, c, {},
                    request=Mock(side_effect=[i, m]), process_reader=Mock(return_value={'CPU_mock_process': True}))
            self.assertTrue(call()['HTTP_readiness_verified'])
            for name, change in {
                'ready_wrong_nonce': lambda x: x.update(service_instance_id='old_nonce'),
                'ready_active_counter': lambda x: x['rank_descriptions'][0].update(counter_active=True),
                'ready_wrong_worker_SHA': lambda x: x['rank_descriptions'][0].update(base_worker_source_sha256='0'*64),
                'ready_wrong_worker_class': lambda x: x['rank_descriptions'][0].update(original_candidate_base_worker='other.Worker'),
                'ready_wrong_counter_SHA': lambda x: x['rank_descriptions'][0].update(counter_sha256='0'*64),
                'ready_missing_rank': lambda x: x['rank_descriptions'].pop(),
            }.items():
                bad = copy.deepcopy(identity);change(bad);self.reject(name, lambda value=bad: call(value))
            self.reject('ready_old_A1_model', lambda: call(m={'data': [{'id': 'old_A1'}]}))
            self.reject('ready_multiple_models', lambda: call(m={'data': [{'id': s['served_model']}, {'id': 'other'}]}))

    def write_proc(self, proc, pid, ppid, pgid, ticks, argv, uid=None):
        path = proc / str(pid);path.mkdir(parents=True, exist_ok=True)
        fields = ['S', str(ppid), str(pgid)] + ['0']*16 + [str(ticks)] + ['0']*4
        (path / 'stat').write_text(f'{pid} (CPU mock worker) ' + ' '.join(fields) + '\n')
        (path / 'cmdline').write_bytes(b'\0'.join(x.encode() for x in argv) + b'\0')
        value = os.geteuid() if uid is None else uid
        (path / 'status').write_text(f'Uid:\t{value}\t{value}\t{value}\t{value}\n')

    def process_fixture(self, root):
        source, job, c, s, identity = self.fixture_service(root)
        proc = root / 'proc';(proc / 'sys/kernel/random').mkdir(parents=True)
        (proc / 'sys/kernel/random/boot_id').write_text('CPU_mock_boot\n')
        argv = ['python', '-u', '-m', lifecycle.BOOTSTRAP, '--job-id', 'quality_new', '--port', '18574',
                '--output', str(job / 'service'), '--candidate', str(source / 'candidate_contract.json'),
                '--counter-file', str(source / 'stack/parallel_tiny/dspark_general_quality_launcher_cpu/assets/frozen_v4_counter.py'), '--execute']
        self.write_proc(proc, 1001, 2, 1001, 10, ['bash', str(job / 'runner.sh')])
        self.write_proc(proc, 1002, 1001, 1001, 20, argv)
        runner = {'pid': 1001, 'start_ticks': 10, 'process_group': 1001, 'boot_id': 'CPU_mock_boot'}
        return source, job, c, s, proc, argv, runner

    def test_exact_job_process_includes_parent_group_start_time_and_source(self):
        with tempfile.TemporaryDirectory() as temp:
            source, job, c, s, proc, argv, runner = self.process_fixture(Path(temp))
            def check(selection=s, record=runner):
                return lifecycle.owned_process(selection, source, job, 'quality_new', 18574, c, record, proc)
            observed = check();self.assertEqual(observed['pid'], 1002)
            bad = dict(runner, start_ticks=11)
            self.reject('runner_PID_reused', lambda: check(record=bad))
            bad = dict(s, candidate_fingerprint='other')
            self.reject('process_other_candidate', lambda: check(selection=bad))
            for name, edit in {
                'process_other_job': lambda a: a.__setitem__(a.index('--job-id')+1, 'other_job'),
                'process_other_output': lambda a: a.__setitem__(a.index('--output')+1, '/old/job'),
                'process_other_counter_owner': lambda a: a.__setitem__(a.index('--counter-file')+1, '/old/v4.py'),
                'process_private_bootstrap': lambda a: a.__setitem__(a.index('-m')+1, 'private.bootstrap'),
            }.items():
                changed = list(argv);edit(changed);self.write_proc(proc,1002,1001,1001,20,changed)
                self.reject(name, check)
            self.write_proc(proc,1002,1001,77,20,argv);self.reject('process_foreign_group', check)
            self.write_proc(proc,1002,99,1001,20,argv);self.reject('process_foreign_parent', check)
            self.write_proc(proc,1002,1001,1001,20,argv,os.geteuid()+1);self.reject('process_foreign_uid', check)
            self.reject('process_init_PID', lambda: lifecycle.proc_identity(1,proc))
            reused = dict(observed, start_ticks=21)
            self.reject('stop_PID_reused', lambda: lifecycle.match_process(observed,reused))

    def test_stop_plan_has_no_signal_and_mocked_execute_signals_only_exact_PID(self):
        with tempfile.TemporaryDirectory() as temp:
            source, job, c, s, proc, argv, runner = self.process_fixture(Path(temp))
            observed = lifecycle.owned_process(s,source,job,'quality_new',18574,c,runner,proc)
            with patch.object(lifecycle.os,'kill') as kill:
                result = lifecycle.graceful_stop(observed,job,execute=False,proc_root=proc)
                self.assertFalse(result['signal_sent']);kill.assert_not_called()
            (job / 'service').mkdir();s['service_started']=False
            (job / 'service/service_selection.json').write_text(json.dumps(s));(job / 'run.exit').write_text('0\n')
            with patch.object(lifecycle.os,'kill') as kill, patch.object(lifecycle,'proc_identity',side_effect=[observed,FileNotFoundError()]):
                result = lifecycle.graceful_stop(observed,job,execute=True)
                self.assertTrue(result['actual_service_stopped']);kill.assert_called_once_with(1002,signal.SIGTERM)
            bad = dict(observed,start_ticks=99)
            with patch.object(lifecycle.os,'kill') as kill, patch.object(lifecycle,'proc_identity',return_value=bad):
                self.reject('stop_refuses_reused_PID_before_signal',lambda:lifecycle.graceful_stop(observed,job,execute=True))
                kill.assert_not_called()

    def test_shell_and_evaluator_plans_keep_exact_modules_assets_and_new_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source,job,c,s,identity=self.fixture_service(root)
            c['startup_env']['STACK_SRC_ROOT']=str(source)
            c['startup_env_fingerprint']=fingerprint(c['startup_env'])
            args=SimpleNamespace(job_id='quality_new',port=18574,source_dir=source,container='CPU_mock_container',
                                 snapshot_manifest_sha256='1'*64,ready_timeout=1800,stop_after=True,allow_known_alarm=False)
            plan=launch.command_plan(args,{'counter_file':'/frozen_v4.py'},c)
            self.assertIn(lifecycle.BOOTSTRAP,plan['bootstrap'])
            self.assertTrue(plan['stop_after']);self.assertEqual(plan['services_started'],0)
            self.assertTrue(all(plan[key] is False for key in false_claims()))
            for kind in ('service','evaluator'):
                path=root/(kind+'.sh');path.write_text(launch.script_body(plan,kind))
                subprocess.run(['bash','-n',str(path)],check=True,capture_output=True)
            subprocess.run(['bash','-n',str(HERE/'run_general.sh')],check=True,capture_output=True)
            self.reject('shell_bad_environment_key',lambda:launch.exports({'BAD;KEY':'x'}))
            self.reject('shell_nonstring_environment_value',lambda:launch.exports({'KEY':1}))
            asset=json.loads((HERE/'evidence/actual_asset_paths_CPU.json').read_text())
            command=evaluate_when_ready.evaluator_command(source,job,'quality_new',18574,asset)
            self.assertEqual(command[command.index('--output')+1],str(job/'client'))
            self.assertEqual(command[command.index('--encoder')+1],asset['paths']['encoder'])
            self.assertIn('parallel_tiny.dspark_general_quality_api_cpu.orchestrator',command)

    def test_copy_overlay_never_rewrites_original_and_source_closure_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ('stack','operator','up950'):(root/name).mkdir();(root/name/'a.py').write_text('CPU_mock=1\n')
            self.assertEqual(len(prepare_snapshot.source_files(root)),3)
            prepare_snapshot.overlay(root/'stack',{'new.py':b'new=1\n'})
            self.reject('overlay_different_original_source',lambda:prepare_snapshot.overlay(root/'stack',{'a.py':b'changed\n'}))
            (root/'stack/model.safetensors').write_bytes(b'CPU_mock_not_weights')
            self.reject('snapshot_never_copies_source_weights',lambda:prepare_snapshot.source_files(root))
            (root/'stack/model.safetensors').unlink();(root/'up950/link').symlink_to(root/'operator',target_is_directory=True)
            self.reject('snapshot_directory_symlink_not_closed',lambda:prepare_snapshot.source_files(root))

    def test_mocked_export_preserves_seed_and_cannot_promote_mock_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'native.py';path.write_text('CPU_mock_native_source=1\n')
            manifest_path=root/'sources.json';manifest={'files':{'native.py':common.sha(path)}}
            manifest_path.write_text(json.dumps(manifest))
            sdk=root/'sdk.py';sdk.write_text('CPU_mock_SDK_source=1\n')
            sdk_manifest=root/'sdk.json';sdk_manifest.write_text(json.dumps({'sdk_sources':{
                'mock':{'path':str(sdk),'sha256':common.sha(sdk),'bytes':sdk.stat().st_size}}}))
            seed=fixture_candidate();seed.pop('CPU_fixture_only');seed['schema']='CPU_mock_in_memory_export_seed'
            seed['source_root']=str(root);seed['source_sha256']=manifest['files']
            seed['sdk_source_manifest_sha256']=common.sha(sdk_manifest)
            seed['sdk_pair_provenance']={'protocol_passed':True,'CPU_mock_in_memory_provenance':True}
            before=copy.deepcopy(seed)
            with patch.object(export_candidate,'validate_real_candidate'), patch.object(export_candidate,'checkpoint_metadata',
                    return_value={'CPU_fixture_only':True,'actual_NPU_load_verified':False,**false_claims()}):
                result=export_candidate.export(seed,root,manifest,manifest_path,sdk_manifest,common.sha(sdk_manifest))
            self.assertEqual(seed,before);self.assertEqual(result['model_kwargs'],seed['model_kwargs'])
            self.assertEqual(result['startup_env'],seed['startup_env']);self.assertFalse(result['actual_runtime_candidate_loaded'])
            self.assertTrue(all(result['actual_flags'][key] is False for key in false_claims()))
            self.reject('mock_export_metadata_not_actual_candidate',lambda:common.reject_fixture(result))
            bad=copy.deepcopy(seed);bad['source_sha256']={'native.py':'0'*64}
            self.reject('export_seed_other_source_manifest',lambda:export_candidate.export(bad,root,manifest,manifest_path,sdk_manifest,common.sha(sdk_manifest)))
            bad=copy.deepcopy(seed);bad['source_root']='/other/source'
            self.reject('export_seed_other_source_owner',lambda:export_candidate.export(bad,root,manifest,manifest_path,sdk_manifest,common.sha(sdk_manifest)))
            bad=copy.deepcopy(seed);bad['sdk_source_manifest_sha256']='0'*64
            self.reject('export_seed_other_SDK_manifest',lambda:export_candidate.export(bad,root,manifest,manifest_path,sdk_manifest,common.sha(sdk_manifest)))

    def test_mocked_snapshot_writes_new_closed_sources_and_rejects_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base=root/'base';base.mkdir()
            for name in ('stack','operator','up950'):
                (base/name).mkdir();(base/name/'CPU_mock_source.py').write_text('CPU_mock=1\n')
            seed=fixture_candidate();seed['source_root']=str(base)
            seed['startup_env']['STACK_SRC_ROOT']=str(base);seed['startup_env_fingerprint']=fingerprint(seed['startup_env'])
            seed['source_sha256']={str(p.relative_to(base)):common.sha(p) for p in base.rglob('*.py')}
            seed['pythonpath']=[str(base/name) for name in ('stack','operator','up950')]
            seed['startup_env_basis']={'source_path':str(base/'stack/CPU_mock_source.py')}
            packages={}
            for name in prepare_snapshot.PACKAGES:
                package=root/name;package.mkdir();(package/'__init__.py').write_text('"CPU mock package"\n')
                if name=='dspark_general_quality_api_cpu':
                    (package/'evidence').mkdir()
                    (package/'evidence/evaluator_sources.json').write_bytes((HERE.parent/name/'evidence/evaluator_sources.json').read_bytes())
                mapping={str(p.relative_to(package)):{'bytes':p.stat().st_size,'sha256':common.sha(p)} for p in package.rglob('*') if p.is_file()}
                (package/'frozen_files.json').write_text(json.dumps({'root':str(package),'text_files':mapping,'CPU_only':True}))
                packages[name]=(package,common.sha(package/'frozen_files.json'))
            counter=root/'CPU_mock_counter.py';counter.write_text('CPU_mock_counter=1\n')
            asset={'CPU_fixture_only':True,'paths':{'client_root':'/CPU_mock/original_client'},'files':{}}
            before=copy.deepcopy(seed);output=root/'quality_snapshot'
            with patch.object(prepare_snapshot,'validate_real_candidate'), patch.object(prepare_snapshot,'checkpoint_metadata',
                    return_value={'CPU_fixture_only':True,**false_claims()}), patch.object(prepare_snapshot.assets,'verify'), \
                    patch.object(prepare_snapshot,'COUNTER_SHA',common.sha(counter)):
                plan=prepare_snapshot.prepare(seed,output,packages,counter,asset,execute=False)
                self.assertFalse(output.exists());self.assertEqual(plan['services_started'],0)
                result=prepare_snapshot.prepare(seed,output,packages,counter,asset,execute=True)
                self.assertTrue(result['actual_snapshot_files_written'])
                self.reject('snapshot_old_output_reuse',lambda:prepare_snapshot.prepare(seed,output,packages,counter,asset,execute=True))
                self.reject('snapshot_inside_original_source',lambda:prepare_snapshot.prepare(seed,base/'nested',packages,counter,asset,execute=True))
            self.assertEqual(seed,before)
            snapshot=json.loads((output/'quality_source_snapshot.json').read_text())
            self.assertEqual(common.verify_map(output,snapshot['files']),snapshot['files'])
            for name,value in seed['source_sha256'].items():self.assertEqual(common.sha(output/name),value)
            copied=json.loads((output/'candidate_contract.json').read_text())
            self.assertEqual(copied['model_kwargs'],seed['model_kwargs'])
            self.assertEqual(copied['source_root'],str(output));self.assertEqual(copied['startup_env']['STACK_SRC_ROOT'],str(output))
            self.assertEqual(copied['source_sha256'],snapshot['files'])
            self.reject('mock_snapshot_not_actual_candidate',lambda:common.reject_fixture(copied))


def main():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Controls)
    count=suite.countTestCases()
    with (HERE/'evidence/CPU_validation.log').open('w') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    report={'CPU_only':True,'CPU_mock_snapshot_resource_HTTP_ASGI_process_and_signal_controls':True,
            'tests':count,'passed':result.wasSuccessful(),'negative_cases':NEGATIVES,
            'negative_case_count':len(NEGATIVES),'actual_asset_files_readonly_captured':20,
            'NPU_calls':0,'HTTP_requests':0,'services_started':0,'real_signals_sent':0,
            'backend_modules_imported':[name for name in sys.modules if name.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend')],
            'actual_flags':false_claims()}
    assert report['backend_modules_imported']==[]
    (HERE/'evidence/CPU_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
