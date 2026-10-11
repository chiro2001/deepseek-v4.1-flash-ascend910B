"""CPU mock of the actual ATen module. It cannot grant any NPU receipt."""
import unittest
import torch
from exporter import PointSpec,make_exporter_class,assert_export,hook_carrier_policy
from adapters import (bounded_cache_rows,observe_cache_reader,observe_cache_write_input,
                      accepted_input_mask,observe_route_pair,require_sdk_receipts)


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

    def test_bounded_cache_coordinates_mask_invalid_without_nonzero(self):
        cache=torch.arange(24,dtype=torch.bfloat16).reshape(3,2,4)
        coords=torch.tensor([[1,1],[-1,-1],[8,0],[0,9]])
        values,valid=bounded_cache_rows(cache,coords)
        self.assertEqual(valid.tolist(),[True,False,False,False])
        self.assertTrue(torch.equal(values[0],cache[1,1]))
        self.assertTrue(torch.equal(values[1:],cache[0,0].expand(3,4)))

    def test_actual_cache_reader_and_writer_keep_carriers(self):
        Native=make_exporter_class(torch)
        coords=torch.tensor([[1,1],[2,0]])
        cache=torch.arange(24,dtype=torch.bfloat16).reshape(3,2,4)
        reader=Native(PointSpec('reader','cache-read',40,6,4,2),device='cpu',dtype=torch.bfloat16)
        q=torch.ones((2,8),dtype=torch.bfloat16)
        self.assertIs(observe_cache_reader(reader,q,cache,coords,torch.tensor([1])),q)
        updates=cache[coords[:,0],coords[:,1]].clone()
        writer=Native(PointSpec('writer','cache-write',40,6,4,2),device='cpu',dtype=torch.bfloat16)
        self.assertIs(observe_cache_write_input(writer,updates,coords,torch.tensor([1])),updates)
        self.assertTrue(torch.equal(writer.values[:2],reader.values[:2]))

    def test_accepted_cache_mask_excludes_future_bonus(self):
        for accepted in range(6):
            mask=accepted_input_mask(torch,6,torch.tensor([accepted]),device='cpu')
            self.assertEqual(mask.tolist(),[index<accepted+1 for index in range(6)])

    def test_route_observers_preserve_native_integer_ids(self):
        Native=make_exporter_class(torch)
        ids=torch.tensor([[257,383,4,5,6,7]],dtype=torch.int32)
        weights=torch.arange(6,dtype=torch.float32).reshape(1,6)/16
        ids_observer=Native(PointSpec('route.ids','producer',40,6,6),device='cpu',dtype=torch.int32)
        weights_observer=Native(PointSpec('route.weights','producer',40,6,6),device='cpu',dtype=torch.float32)
        returned=observe_route_pair(ids_observer,weights_observer,(weights,ids),torch.tensor([[7]]),torch.tensor([True]),torch.tensor([2]))
        self.assertIs(returned[0],weights);self.assertIs(returned[1],ids)
        self.assertEqual(ids_observer.values[0].tolist(),ids[0].tolist())

    def test_failed_sdk_gate_rejects_installation(self):
        with self.assertRaisesRegex(ValueError,'has not passed'):
            require_sdk_receipts({'stage':'graphex','static_kernel':True,'passed':False},{},'unvalidated')


if __name__=='__main__':unittest.main()
