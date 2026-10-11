"""CPU ATen tuple tests, no NPU backend import or SDK device receipt."""
import unittest
import torch
from tensor_tuple_transport import SnapshotSpec,allocate_buffers,snapshot_tuple,reset_after_prior_fence
from hook_tuple_adapters import (decoder_output_tuple,cache_write_input_tuple,cache_reader_input_tuple,
                                 accepted_cache_input_mask,activation_before_mutating_consumer_tuple)


class TupleTransportCPU(unittest.TestCase):
    def setUp(self):
        self.buffers=allocate_buffers(torch,SnapshotSpec(6,4),device='cpu',dtype=torch.bfloat16)
        self.values=torch.arange(24,dtype=torch.bfloat16).reshape(6,4)
        self.positions=torch.arange(6,dtype=torch.int64).reshape(6,1)
        self.valid=torch.ones((6,),dtype=torch.bool)

    def invoke(self,generation=1,**changes):
        args={'carrier':self.values,'values':self.values,'positions':self.positions,'valid':self.valid,
              'epoch':torch.tensor([generation],dtype=torch.int64),'buffers':self.buffers}
        args.update(changes)
        return snapshot_tuple(torch,**args)

    def test_original_carrier_and_all_six_outputs_are_explicit_identical_tensors(self):
        result=self.invoke()
        self.assertIs(result[0],self.values)
        for original,returned in zip(self.buffers,result[1:]):self.assertIs(original,returned)
        self.assertTrue(torch.equal(result[1],self.values))
        self.assertEqual(result[4].item(),1);self.assertEqual(result[5].item(),1)

    def test_thirty_two_changing_epochs_keep_original_addresses(self):
        addresses=[tensor.data_ptr() for tensor in self.buffers]
        for generation in range(1,33):
            reset_after_prior_fence(self.buffers)  # CPU mock of an already-fenced reset.
            self.values.add_(1);self.positions.add_(8)
            result=self.invoke(generation)
            self.assertEqual([tensor.data_ptr() for tensor in result[1:]],addresses)
            self.assertEqual(result[4].item(),generation);self.assertEqual(result[5].item(),1)
            self.assertTrue(torch.equal(result[1],self.values))

    def test_duplicate_writes_are_visible_in_device_count_tensor(self):
        self.invoke();self.invoke()
        self.assertEqual(self.buffers[4].item(),2)

    def test_overbound_or_wrong_native_dtype_rejected(self):
        with self.assertRaisesRegex(ValueError,'extent'):
            self.invoke(values=torch.zeros((7,4)))
        with self.assertRaisesRegex(ValueError,'dtype/device'):
            self.invoke(values=self.values.float())
        with self.assertRaisesRegex(ValueError,'epoch'):
            self.invoke(epoch=torch.tensor([1],dtype=torch.int32))

    def test_capture_transport_source_has_no_d2h_or_hidden_global_mutation(self):
        import ast,inspect
        tree=ast.parse(inspect.getsource(snapshot_tuple))
        self.assertFalse(any(isinstance(node,(ast.Global,ast.Nonlocal)) for node in ast.walk(tree)))
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotIn(node.func.attr,('cpu','item','tolist','nonzero','clone'))

    def test_decoder_adapter_preserves_hidden_mix_and_explicit_observer_tuple(self):
        hidden=self.values.reshape(6,2,2);mix=torch.ones((6,2),dtype=torch.float32)
        result=decoder_output_tuple(torch,(hidden,mix),self.positions,self.valid,torch.tensor([4]),self.buffers)
        self.assertIs(result[0],hidden);self.assertIs(result[1],mix)
        self.assertIs(result[2],self.buffers[0])

    def test_writer_reader_use_bounded_rows_and_original_carriers(self):
        coordinates=torch.tensor([[1,1],[2,0],[-1,-1]])
        cache=torch.arange(24,dtype=torch.bfloat16).reshape(3,2,4)
        buffers=allocate_buffers(torch,SnapshotSpec(6,4,2),device='cpu',dtype=torch.bfloat16)
        q=torch.ones((3,8),dtype=torch.bfloat16)
        result=cache_reader_input_tuple(torch,q,cache,coordinates,torch.tensor([5]),buffers)
        self.assertIs(result[0],q);self.assertEqual(result[3][:3].tolist(),[True,True,False])
        updates=cache[torch.tensor([1,2,0]),torch.tensor([1,0,0])]
        writer_buffers=allocate_buffers(torch,SnapshotSpec(6,4,2),device='cpu',dtype=torch.bfloat16)
        written=cache_write_input_tuple(torch,updates,coordinates,torch.tensor([5]),writer_buffers)
        self.assertIs(written[0],updates)
        self.assertTrue(torch.equal(result[1][:2],written[1][:2]))

    def test_bonus_mask_is_a_future_input(self):
        for accepted in range(6):
            mask=accepted_cache_input_mask(torch,6,torch.tensor([accepted]),device='cpu')
            self.assertEqual(mask.tolist(),[index<accepted+1 for index in range(6)])

    def test_preconsumer_snapshot_survives_raw_gm_write_without_version_change(self):
        import ctypes
        x=torch.arange(8,dtype=torch.int8).reshape(2,4)
        expected=x.clone();version=x._version
        buffers=allocate_buffers(torch,SnapshotSpec(6,4),device='cpu',dtype=torch.int8)
        out=activation_before_mutating_consumer_tuple(torch,x,torch.tensor([3,4]),
                torch.tensor([True,True]),torch.tensor([6]),buffers)
        self.assertIs(out[0],x)
        # CPU analogue of the CANN direct GM pack write: bypass Tensor ops.
        ctypes.memset(x.data_ptr(),0,x.numel()*x.element_size())
        self.assertEqual(x._version,version)
        self.assertFalse(torch.equal(x,expected))
        self.assertTrue(torch.equal(out[1][:2],expected))
        self.assertNotEqual(out[1].data_ptr(),x.data_ptr())


if __name__=='__main__':unittest.main()
