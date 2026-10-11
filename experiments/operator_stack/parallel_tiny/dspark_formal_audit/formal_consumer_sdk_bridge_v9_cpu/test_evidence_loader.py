"""Read-only real dual-job positive and protocol rejection cases; CPU only."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

from evidence_loader import (EvidenceError, JobFiles, load_tp8_sdk_pair, parse_json, source_policy,
                             validate_job_payload, validate_pair_identities)


ROOT = Path(__file__).resolve().parents[3]/'evidence/formal_a321'
STATIC = ROOT/'sdk_init_static_keep_v8'
NONSTATIC = ROOT/'sdk_init_nonstatic_keep_v8'


class BridgeProtocolCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = source_policy()
        cls.pair = load_tp8_sdk_pair(STATIC,NONSTATIC)
        cls.driver = json.loads((STATIC/'gate/result_compact.json').read_text())
        cls.controller = json.loads((STATIC/'gate/metadata/controller.json').read_text())
        cls.cleanup = json.loads((STATIC/'gate/owned_failure_cleanup.json').read_text())
        cls.manifest = json.loads((STATIC/'gate/metadata/source_manifest.json').read_text())
        cls.ranks = {rank:json.loads((STATIC/f'gate/sdk_results/static_keep/rank{rank}/result.json').read_text()) for rank in range(8)}
        cls.proofs = {rank:json.loads((STATIC/f'gate/metadata/loaded_abi_proofs/rank{rank}.json').read_text()) for rank in range(8)}

    def setUp(self):
        self.d = copy.deepcopy(self.driver);self.c = copy.deepcopy(self.controller)
        self.cl = copy.deepcopy(self.cleanup);self.m = copy.deepcopy(self.manifest)
        self.r = copy.deepcopy(self.ranks);self.p = copy.deepcopy(self.proofs)

    def check(self):
        return validate_job_payload(self.d,self.c,self.cl,self.m,self.r,self.p,
                                    self.policy[3],self.policy[4]['profile'],self.policy[5])

    def reject(self, code=None):
        with self.assertRaises(EvidenceError) as caught:self.check()
        if code is not None:self.assertEqual(caught.exception.code,code)

    def test_real_two_complete_jobs_have_immutable_provenance_and_no_formal_grant(self):
        pair = self.pair;report = pair.to_dict()
        self.assertTrue(pair.protocol_passed)
        self.assertEqual([r.rank for r in pair.static.ranks],list(range(8)))
        self.assertEqual([r.rank for r in pair.nonstatic.ranks],list(range(8)))
        self.assertTrue(all(r.epochs == 32 and len(r.owner_seals) == 13 for r in pair.static.ranks+pair.nonstatic.ranks))
        for name in ['actual_sdk_gate_passed','device_audit_passed','full_consumer_audit_passed',
                     'safe_formal_receipt','formal_model_validated','actual_prepack_consumer_order_validated']:
            self.assertFalse(report[name])
        with self.assertRaises(TypeError):pair.static.ranks[0].owner_seals['native.cache'] = {}
        report['static']['ranks'][0]['identity']['pid'] = 1
        self.assertNotEqual(pair.static.ranks[0].identity['pid'],1)
        self.assertFalse(any(name=='torch' or name=='torch_npu' or name.startswith('vllm.') for name in sys.modules))

    def test_missing_nonstatic_or_same_static_job_is_rejected(self):
        with self.assertRaises(EvidenceError):load_tp8_sdk_pair(STATIC,None)
        with self.assertRaises(EvidenceError):load_tp8_sdk_pair(STATIC,STATIC)

    def test_missing_rank_or_proof_is_rejected(self):
        del self.r[7];self.reject('rank')
        self.r = copy.deepcopy(self.ranks);del self.p[7];self.reject('rank')

    def test_driver_failed_nonzero_and_incomplete_cleanup_are_rejected(self):
        self.d['passed'] = False;self.reject('terminal')
        self.d['passed'] = True;self.d['worker_exit_codes_complete'] = False;self.reject('terminal')
        self.d['worker_exit_codes_complete'] = True;self.d['worker_exit_codes']['0'] = -9;self.reject('terminal')

    def test_cleanup_pending_hard_kill_or_wrong_parent_is_rejected(self):
        self.cl['pending_owned_pids'] = [1];self.reject('terminal')
        self.cl = copy.deepcopy(self.cleanup);self.cl['workers'][0]['hard_kill_sent'] = True
        self.d['owned_worker_cleanup'] = copy.deepcopy(self.cl);self.reject('terminal')
        self.cl = copy.deepcopy(self.cleanup);self.cl['workers'][0]['birth_identity']['ppid'] += 1
        self.d['owned_worker_cleanup'] = copy.deepcopy(self.cl);self.reject('identity')

    def test_duplicate_controller_or_cross_job_pid_is_rejected(self):
        self.c['worker_processes'][1]['pid'] = self.c['worker_processes'][0]['pid']
        with self.assertRaises(ValueError):self.check()
        rank = self.pair.nonstatic.ranks[0]
        identity = dict(rank.identity);identity['pid'] = self.pair.static.ranks[0].identity['pid']
        other = replace(self.pair.nonstatic,ranks=(replace(rank,identity=identity),*self.pair.nonstatic.ranks[1:]))
        with self.assertRaises(EvidenceError):validate_pair_identities(self.pair.static,other)

    def test_wrong_candidate_sdk_or_native_factory_source_is_rejected(self):
        self.m['candidate_sources']['contracts.py'] = '0'*64;self.reject('source')
        self.m = copy.deepcopy(self.manifest);self.m['sdk_sources']['torch']['sha256'] = '0'*64;self.reject('source')
        self.m = copy.deepcopy(self.manifest);self.m['native_capture_factory']['function_sha256'] = '0'*64;self.reject('source')

    def test_different_outer_keep_or_stage_is_rejected(self):
        self.r[0]['outer_keep_inference_input_mutations'] = False;self.reject('variant')
        self.r[0]['outer_keep_inference_input_mutations'] = True;self.r[0]['stage'] = 'eager';self.reject('variant')

    def test_static_header_cannot_hide_wrong_real_compiler_or_clone_policy(self):
        self.r[0]['compiler_options'][0]['static_kernel_compile'] = False;self.reject('variant')
        self.r = copy.deepcopy(self.ranks);self.r[0]['compiler_options'][0]['clone_input'] = True
        self.reject('variant')

    def test_missing_owner_seal_or_changed_cache_owner_is_rejected(self):
        del self.r[0]['owned_copy_destination_seals_before']['native.cache'];self.reject('owner')
        self.r = copy.deepcopy(self.ranks)
        self.r[0]['owned_copy_destination_seals_after_capture']['native.cache']['tensor_object_id'] += 1
        self.reject('owner')

    def test_temporary_destination_marker_copyback_or_wrong_stream_is_rejected(self):
        self.r[0]['capture_calls'][0]['destination']['ptr'] += 1
        self.r[0]['copy_calls'][-8:] = copy.deepcopy(self.r[0]['capture_calls']);self.reject('owner')
        self.r = copy.deepcopy(self.ranks);self.r[0]['capture_calls'][1]['source']['ptr'] += 1
        self.r[0]['copy_calls'][-8:] = copy.deepcopy(self.r[0]['capture_calls']);self.reject('owner')
        self.r = copy.deepcopy(self.ranks);self.r[0]['capture_calls'][0]['stream'] += 1
        self.r[0]['copy_calls'][-8:] = copy.deepcopy(self.r[0]['capture_calls']);self.reject('owner')

    def test_current_native_context_or_restore_proof_is_rejected(self):
        self.r[0]['native_capture_scope']['native_scope']['context_ptr'] += 1;self.reject('scope')
        self.r = copy.deepcopy(self.ranks);self.r[0]['native_capture_scope']['normal_exit_restoration_verified'] = False
        self.reject('scope')

    def test_old_single_chip_initialized_schema_cannot_be_rewritten_into_bridge(self):
        self.c['schema'] = 'native_capture_copy_v1'
        with self.assertRaises(ValueError):self.check()
        self.c = copy.deepcopy(self.controller);self.r[0]['initialized_worker_scope']['current_initialized'] = False
        self.reject('scope')

    def test_missing_duplicate_epoch_or_fence_is_rejected(self):
        self.r[0]['generations'].pop();self.reject('epoch')
        self.r = copy.deepcopy(self.ranks);self.r[0]['generations'][1]['epoch'] = 1;self.reject('epoch')
        self.r = copy.deepcopy(self.ranks);self.r[0]['generations'][1]['device_fence_monotonic_ns'] = self.r[0]['generations'][0]['device_fence_monotonic_ns']
        self.reject('fence')

    def test_true_flags_do_not_hide_stale_payload_position_or_duplicate_write(self):
        self.r[0]['generations'][0]['exports'][0]['values'][0][0] += 1;self.reject('epoch')
        self.r = copy.deepcopy(self.ranks);self.r[0]['generations'][0]['exports'][0]['positions'][0][0] += 1;self.reject('epoch')
        self.r = copy.deepcopy(self.ranks);self.r[0]['generations'][0]['exports'][0]['copy_count'] = 2;self.reject('epoch')

    def test_missing_abi_code_or_wrong_pid_is_rejected(self):
        self.r[0]['runtime_library_sources']['acl']['loaded_abi_proof']['pid'] += 1;self.reject('abi')
        self.r = copy.deepcopy(self.ranks);self.r[0]['runtime_library_sources']['acl']['loaded_abi_proof']['executable_memory'][0]['byte_equal'] = False
        self.p[0]['acl']['executable_memory'][0]['byte_equal'] = False;self.reject('abi')

    def test_tensor_abi_or_payload_layout_or_reported_copy_class_cannot_be_claimed(self):
        self.r[0]['runtime_call_abi'][0]['tensor_mask'][0] = False;self.reject('call_abi')
        self.r = copy.deepcopy(self.ranks);self.r[0]['buffers_after_capture']['producer.values']['ptr'] += 1;self.reject('owner')
        self.r = copy.deepcopy(self.ranks);self.r[0]['copy_alias_structure']['captured_copy_classes']['count'] = 1;self.reject('copy')

    def test_boolean_authority_and_no_reference_metadata_are_rejected(self):
        self.r[0]['device_audit_passed'] = True;self.reject('authority')
        self.r = copy.deepcopy(self.ranks);self.r[0]['safe_formal_receipt'] = True;self.reject('authority')
        with self.assertRaises(EvidenceError):parse_json('{"x":1,"x":2}')
        with self.assertRaises(EvidenceError):parse_json('{"x":NaN}')

    def test_trusted_collection_hash_blocks_changed_bytes_and_symlinks(self):
        scratch = Path(__file__).parent/'evidence/test_scratch';scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            root = Path(temp)
            (root/'archive_receipt.json').write_bytes((STATIC/'archive_receipt.json').read_bytes())
            (root/'fetch_manifest.json').write_bytes((STATIC/'fetch_manifest.json').read_bytes())
            bundle = JobFiles(root,self.policy[4]['collections']['static_keep'])
            (root/'run.exit').write_text('1\n')
            with self.assertRaises(EvidenceError):bundle.file('run.exit')
            (root/'run.exit').unlink();(root/'run.exit').symlink_to(STATIC/'run.exit')
            with self.assertRaises(EvidenceError):bundle.file('run.exit')
        scratch.rmdir()


if __name__ == '__main__':unittest.main()
