"""CPU owner-set regression; neither metadata nor old traces grant NPU evidence."""
import ast
import copy
import json
from pathlib import Path
import unittest

from contracts import validate_copy_aliases


class OwnedDestinationsCPU(unittest.TestCase):
    def setUp(self):
        self.buffers = {}
        for owner in ('producer','reader'):
            for field,shape,stride,dtype in (
                ('values',[6,64],[64,1],'torch.bfloat16'),('positions',[6,1],[1,1],'torch.int64'),
                ('valid',[6],[1],'torch.bool'),('epoch',[1],[1],'torch.int64'),
                ('copy_count',[1],[1],'torch.int64'),('row_count',[1],[1],'torch.int64')):
                self.buffers[owner+'.'+field] = {'ptr':(len(self.buffers)+1)*4096,
                    'shape':shape,'stride':stride,'dtype':dtype,'device':'npu:0','format':2}
        self.cache = {'ptr':65536,'shape':[16,64],'stride':[64,1],
                      'dtype':'torch.bfloat16','device':'npu:0','format':2}
        self.seals = {name:{'owner':name,'tensor_object_id':100+i,'layout':copy.deepcopy(layout)}
                      for i,(name,layout) in enumerate({**self.buffers,'native.cache':self.cache}.items())}
        self.after = copy.deepcopy(self.seals)
        self.returned = copy.deepcopy(self.buffers)
        self.returned_cache = copy.deepcopy(self.cache)
        self.sources = {};self.calls = []
        for name,seal in self.seals.items():
            layout = seal['layout'];source = {**copy.deepcopy(layout),'ptr':layout['ptr']+1048576}
            self.calls.append({'phase':'cold-capture','op':'aten.copy_.default','stream':8192,
                'destination':copy.deepcopy(layout),'logical_destination':copy.deepcopy(layout),'source':source})
            if name.endswith(('.positions','.valid','.epoch')):self.sources[layout['ptr']] = copy.deepcopy(source)

    def check(self):
        return validate_copy_aliases(self.buffers,self.calls,self.returned,8192,
            marker_sources=self.sources,owned_seals_before=self.seals,owned_seals_after=self.after,
            returned_native_cache=self.returned_cache)

    def test_payload_four_counts_and_native_cache_exact_owners_pass(self):
        receipt = self.check()
        self.assertEqual(receipt['validated_owned_destination_count'],13)
        self.assertEqual(receipt['captured_copy_classes'],{'payload':8,'count':4,'native_cache':1})
        self.assertEqual(set(receipt['captured_copy_owners']),set(self.seals))
        self.assertFalse(receipt['device_audit_passed']);self.assertFalse(receipt['safe_formal_receipt'])

    def test_all_eight_original_v7_traces_still_reject_real_epoch_temporary(self):
        fixture = json.loads((Path(__file__).parent/'evidence/cpu_owned_destinations/v7_default_trace.json').read_text())
        self.assertFalse(fixture['original_live_native_cache_seal_recorded_by_v7'])
        self.assertFalse(fixture['original_live_marker_sources_recorded_by_v7'])
        self.assertEqual(len(fixture['records']),8)
        for record in fixture['records']:
            with self.subTest(rank=record['rank']):
                r = record['projection'];buffers = r['buffers_before']
                # v7 did not save original cache/source seals. These missing inputs
                # are explicitly synthetic CPU setup, never inferred from a copy.
                cache = {**self.cache,'ptr':9000000,'device':buffers['producer.values']['device']}
                seals = {name:{'owner':name,'tensor_object_id':200+i,'layout':copy.deepcopy(layout)}
                         for i,(name,layout) in enumerate({**buffers,'native.cache':cache}.items())}
                sources = {layout['ptr']:{**copy.deepcopy(layout),'ptr':layout['ptr']+999000000}
                           for name,layout in buffers.items() if name.endswith(('.positions','.valid','.epoch'))}
                first = r['capture_calls'][0]
                self.assertEqual(first['logical_destination'],buffers['producer.epoch'])
                self.assertNotIn(first['destination']['ptr'],{s['layout']['ptr'] for s in seals.values()})
                self.assertEqual(first['stream'],r['native_capture_scope']['native_stream_ptr'])
                with self.assertRaisesRegex(ValueError,'temporary/nonowned'):
                    validate_copy_aliases(buffers,r['capture_calls'],r['returned_buffers_after_capture'],
                        r['native_capture_scope']['native_stream_ptr'],marker_sources=sources,
                        owned_seals_before=seals,owned_seals_after=copy.deepcopy(seals),returned_native_cache=cache)

    def test_verified_v7_static_keep_direct_payload_trace_remains_accepted_on_cpu(self):
        fixture = json.loads((Path(__file__).parent/'evidence/cpu_owned_destinations/v7_static_keep_trace.json').read_text())
        self.assertTrue(fixture['original_marker_sources_verified_by_old_v7_gate'])
        self.assertFalse(fixture['original_live_native_cache_seal_recorded_by_v7'])
        self.assertFalse(fixture['v7_overall_driver_passed'])
        for record in fixture['records']:
            with self.subTest(rank=record['rank']):
                r = record['projection'];buffers = r['buffers_before']
                self.assertTrue(r['passed']);self.assertTrue(r['actual_copy_alias_gate_passed'])
                self.assertEqual(len(r['capture_calls']),8)
                cache = {**self.cache,'ptr':9000000,'device':buffers['producer.values']['device']}
                seals = {name:{'owner':name,'tensor_object_id':200+i,'layout':copy.deepcopy(layout)}
                         for i,(name,layout) in enumerate({**buffers,'native.cache':cache}.items())}
                markers = {layout['ptr'] for name,layout in buffers.items() if name.endswith(('.positions','.valid','.epoch'))}
                # The old real gate independently checked these marker sources.
                # Missing cache object seals are still synthetic CPU completion.
                sources = {c['destination']['ptr']:copy.deepcopy(c['source']) for c in r['capture_calls']
                           if c['destination']['ptr'] in markers}
                receipt = validate_copy_aliases(buffers,r['capture_calls'],r['returned_buffers_after_capture'],
                    r['native_capture_scope']['native_stream_ptr'],marker_sources=sources,
                    owned_seals_before=seals,owned_seals_after=copy.deepcopy(seals),returned_native_cache=cache)
                self.assertEqual(receipt['captured_copy_classes'],{'payload':8,'count':0,'native_cache':0})
                self.assertFalse(receipt['device_audit_passed']);self.assertFalse(receipt['safe_formal_receipt'])

    def test_wrong_count_owner_label_is_rejected(self):
        self.seals['producer.copy_count']['owner'] = 'reader.copy_count'
        self.after = copy.deepcopy(self.seals)
        with self.assertRaisesRegex(ValueError,'wrong or duplicate owner'):self.check()

    def test_temporary_count_destination_is_rejected(self):
        self.calls[4]['destination']['ptr'] = 999999
        with self.assertRaisesRegex(ValueError,'temporary/nonowned'):self.check()

    def test_temporary_cache_destination_is_rejected(self):
        self.calls[-1]['destination']['ptr'] = 999999
        with self.assertRaisesRegex(ValueError,'temporary/nonowned'):self.check()

    def test_unknown_nonpayload_copy_is_not_ignored(self):
        extra = copy.deepcopy(self.calls[-1]);extra['destination']['ptr'] = 1000000
        extra['logical_destination'] = copy.deepcopy(extra['destination']);self.calls.append(extra)
        with self.assertRaisesRegex(ValueError,'temporary/nonowned'):self.check()

    def test_wrong_stream_for_payload_count_or_cache_is_rejected(self):
        for index in (0,4,12):
            with self.subTest(index=index):
                calls = copy.deepcopy(self.calls);self.calls[index]['stream'] = 8193
                with self.assertRaisesRegex(ValueError,'destination or stream'):self.check()
                self.calls = calls

    def test_changed_destination_layout_is_rejected(self):
        self.calls[-1]['destination']['stride'] = [1,16]
        self.calls[-1]['logical_destination'] = copy.deepcopy(self.calls[-1]['destination'])
        with self.assertRaisesRegex(ValueError,'changed destination'):self.check()

    def test_logical_destination_cannot_launder_another_owner(self):
        self.calls[4]['logical_destination'] = copy.deepcopy(self.buffers['reader.copy_count'])
        with self.assertRaisesRegex(ValueError,'changed destination'):self.check()

    def test_cache_or_count_owner_object_change_is_rejected(self):
        for name in ('native.cache','producer.copy_count'):
            with self.subTest(owner=name):
                self.after = copy.deepcopy(self.seals);self.after[name]['tensor_object_id'] += 1000
                with self.assertRaisesRegex(ValueError,'owner/seals changed'):self.check()

    def test_cache_pointer_or_tuple_layout_change_after_capture_is_rejected(self):
        for name,key,value in [('native.cache','ptr',999999),('reader.values','shape',[5,64])]:
            with self.subTest(owner=name):
                self.after = copy.deepcopy(self.seals);self.after[name]['layout'][key] = value
                with self.assertRaisesRegex(ValueError,'owner/seals changed'):self.check()

    def test_duplicate_owner_pointer_or_object_is_rejected(self):
        for key in ('tensor_object_id','ptr'):
            with self.subTest(key=key):
                seals = copy.deepcopy(self.seals)
                if key == 'ptr':self.seals['native.cache']['layout'][key] = self.buffers['producer.values']['ptr']
                else:self.seals['native.cache'][key] = self.seals['producer.values'][key]
                self.after = copy.deepcopy(self.seals)
                with self.assertRaisesRegex(ValueError,'duplicate'):self.check()
                self.seals = seals

    def test_missing_cache_or_extra_owner_seal_is_rejected(self):
        for extra in (False,True):
            with self.subTest(extra=extra):
                seals = copy.deepcopy(self.seals)
                if extra:self.seals['unknown'] = copy.deepcopy(self.seals['native.cache'])
                else:del self.seals['native.cache']
                self.after = copy.deepcopy(self.seals)
                with self.assertRaisesRegex(ValueError,'exact live original'):self.check()
                self.seals = seals

    def test_marker_source_copyback_stays_rejected(self):
        self.calls[3]['source']['ptr'] = 999999
        with self.assertRaisesRegex(ValueError,'functionalization temporary or copyback'):self.check()

    def test_missing_payload_cannot_be_replaced_by_count_and_cache(self):
        self.calls = [c for c in self.calls if c['destination']['ptr'] not in self.sources]
        with self.assertRaisesRegex(ValueError,'Missing actual payload'):self.check()

    def test_returned_cache_or_tuple_alias_is_rejected(self):
        self.returned_cache['ptr'] = 999999
        with self.assertRaisesRegex(ValueError,'Returned native cache'):self.check()
        self.returned_cache = copy.deepcopy(self.cache);self.returned['reader.values']['ptr'] = 999999
        with self.assertRaisesRegex(ValueError,'Returned Tensor tuple'):self.check()

    def test_cache_owner_requires_fixed_native_probe_layout_and_device(self):
        for key,value in [('shape',[15,64]),('dtype','torch.float32'),('device','npu:1')]:
            with self.subTest(key=key):
                seals = copy.deepcopy(self.seals);self.seals['native.cache']['layout'][key] = value
                self.after = copy.deepcopy(self.seals)
                with self.assertRaisesRegex(ValueError,'layout'):self.check()
                self.seals = seals

    def test_gate_seals_live_original_tensors_and_preserves_program_ast(self):
        root = Path(__file__).parent
        source = (root/'initialized_worker_gate.py').read_text();tree = ast.parse(source)
        seal = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name == 'seal_owned_destinations')
        seal_source = ast.get_source_segment(source,seal)
        self.assertIn("tensors['native.cache']=cache",seal_source)
        self.assertIn('id(tensor)',seal_source);self.assertIn('stamp(tensor)',seal_source)
        self.assertNotIn('copy_calls',seal_source);self.assertNotIn('capture_calls',seal_source)
        old = ast.parse((root.parent/'native_initialized_worker_discovery_v7/initialized_worker_gate.py').read_text())
        program = lambda t:ast.dump(next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name == 'program'),include_attributes=False)
        self.assertEqual(program(tree),program(old))


if __name__ == '__main__':unittest.main()
