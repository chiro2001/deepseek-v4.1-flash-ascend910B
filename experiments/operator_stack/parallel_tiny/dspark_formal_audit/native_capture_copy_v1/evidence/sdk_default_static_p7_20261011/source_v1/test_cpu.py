"""CPU mock of the actual ATen module. It cannot grant any NPU receipt."""
import unittest
import torch
from exporter import PointSpec,make_exporter_class,assert_export,hook_carrier_policy


class NativeCopy(unittest.TestCase):
    def setUp(self):
        self.exporter=make_exporter_class(torch)(PointSpec('layer-output','main',37,6,4),device='cpu',dtype=torch.bfloat16)
        self.values=torch.arange(20,dtype=torch.bfloat16).reshape(5,4)
        self.positions=torch.arange(5,dtype=torch.int64).reshape(5,1)
        self.valid=torch.tensor([True,True,True,False,False])

    def export(self):
        e=self.exporter
        return {'epoch':int(e.epoch.item()),'copy_count':int(e.copy_count.item()),'rows':int(e.row_count.item()),
                'values':e.values.tolist(),'positions':e.positions.tolist(),'valid':e.valid.tolist()}

    def test_carrier_is_unmodified_and_bounded_payload(self):
        carrier=self.values
        out=self.exporter(carrier,self.values,self.positions,self.valid,torch.tensor([3]))
        self.assertIs(out,carrier)
        self.assertEqual(self.exporter.values.shape,(6,4))
        self.assertEqual(self.exporter.positions[-1].tolist(),[-1])
        r=assert_export(self.export(),epoch=3,expected_writes=1,values=self.values.tolist(),positions=self.positions.tolist(),valid=self.valid.tolist())
        self.assertFalse(r['device_audit_passed'])

    def test_duplicate_gm_write_rejected(self):
        for _ in range(2):self.exporter(self.values,self.values,self.positions,self.valid,torch.tensor([3]))
        with self.assertRaisesRegex(ValueError,'duplicate'):
            assert_export(self.export(),epoch=3,expected_writes=1,values=self.values.tolist(),positions=self.positions.tolist(),valid=self.valid.tolist())

    def test_stale_epoch_rejected(self):
        self.exporter(self.values,self.values,self.positions,self.valid,torch.tensor([2]))
        with self.assertRaisesRegex(ValueError,'Stale'):
            assert_export(self.export(),epoch=3,expected_writes=1,values=self.values.tolist(),positions=self.positions.tolist(),valid=self.valid.tolist())

    def test_wrong_consumer_payload_rejected(self):
        self.exporter(self.values,self.values+1,self.positions,self.valid,torch.tensor([3]))
        with self.assertRaisesRegex(ValueError,'payload'):
            assert_export(self.export(),epoch=3,expected_writes=1,values=self.values.tolist(),positions=self.positions.tolist(),valid=self.valid.tolist())

    def test_fixed_addresses_across_reset_and_epochs(self):
        before=[t.data_ptr() for t in self.exporter.tensor_tuple()]
        for epoch in range(1,33):
            self.exporter.reset_outside_capture()
            self.exporter(self.values,self.values,self.positions,self.valid,torch.tensor([epoch]))
            self.assertEqual(before,[t.data_ptr() for t in self.exporter.tensor_tuple()])
            self.assertEqual(self.export()['copy_count'],1)

    def test_overbound_rejected(self):
        with self.assertRaisesRegex(ValueError,'extent'):
            self.exporter(self.values,torch.zeros((7,4)),torch.zeros((7,1)),torch.ones((7,),dtype=torch.bool),torch.tensor([1]))

    def test_none_writer_has_live_tensor_sampling_policy(self):
        policy=hook_carrier_policy('DeepseekV4DSparkModel._store_standard_swa_kv')
        self.assertIn('BEFORE native cache write',policy)
        self.assertIn('cache reader',policy)


if __name__=='__main__':unittest.main()
