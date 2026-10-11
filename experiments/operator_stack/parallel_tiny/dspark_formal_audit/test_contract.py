"""CPU negative controls for provenance/rollback; no NPU receipt is claimed."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from formal_dspark_audit_patches import TensorSeal,accepted_cache_transition,GenerationEnvelope,install_in_new_job


class Contract(unittest.TestCase):
    def test_bonus_is_next_round_cache_input(self):
        r=accepted_cache_transition([10,20,21,22,23,24],list(range(2048,2054)),[20,21,22,23,24],[20,21,30],2048,2)
        self.assertEqual(r['expected_computed_after'],2051)
        self.assertEqual(r['committed_target_input_tokens'],[10,20,21])
        self.assertEqual(r['usable_output_tokens'],[20,21])
        self.assertEqual(r['raw_bonus_token'],30)
        self.assertEqual(r['terminal_tokens_trimmed'],1)
        self.assertFalse(r['actual_cache_write_read_proven'])

    def test_wrong_accepted_prefix_rejected(self):
        with self.assertRaisesRegex(ValueError,'proposal prefix'):
            accepted_cache_transition([10,20,21],[100,101,102],[20,21],[99,30],100,5)

    def test_ptr_layout_and_weight_version_guard(self):
        m={'ptr':1024,'storage_ptr':1024,'storage_offset':0,'version':1,'shape':[6,5120],
           'stride':[5120,1],'dtype':'torch.bfloat16','device':'npu:0','format':2}
        seal=TensorSeal.from_metadata(m)
        with self.assertRaisesRegex(ValueError,'binding changed'):seal.assert_matches({**m,'ptr':2048},weight=True)
        with self.assertRaisesRegex(ValueError,'version'):seal.assert_matches({**m,'version':2},weight=True)
        self.assertFalse(seal.assert_matches({**m,'version':2},weight=False)['device_consumer_proof'])

    def test_core_id_is_not_guessed_from_external_prefix(self):
        e=GenerationEnvelope('source','checkpoint',0)
        with self.assertRaisesRegex(ValueError,'actual add_request mapping'):e.begin(1,'r','r-guess',{'r':['r-actual']})

    def test_no_export_or_reuse_before_fence(self):
        e=GenerationEnvelope('source','checkpoint',0);e.begin(1,'r','internal',{'r':['internal']})
        with self.assertRaisesRegex(ValueError,'fenced'):e.structural_export('internal',[])
        with self.assertRaisesRegex(ValueError,'cannot be reused'):e.begin(2,'r','internal',{'r':['internal']})

    def test_callback_count_never_grants_coverage(self):
        e=GenerationEnvelope('source','checkpoint',0);e.begin(1,'r','internal',{'r':['internal']})
        e.fence(1,current_rank_device_synchronized=True)
        rows=[{'role':'main','layer':i,'epoch':1,'payload_source':'CPU-callback'} for i in range(40)]
        rows.extend({'role':'draft','layer':i,'epoch':1,'payload_source':'CPU-callback'} for i in (40,41,42))
        with self.assertRaisesRegex(ValueError,'CPU callback'):e.structural_export('internal',rows)
        with self.assertRaises(NotImplementedError):install_in_new_job()


if __name__=='__main__':unittest.main()
