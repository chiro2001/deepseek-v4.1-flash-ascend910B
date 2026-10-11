"""CPU boxing/identity/kwargs/rejection contracts without torch/SDK imports."""
import unittest

from tensor_call_abi import (boxed_sdk_compiler_adapter,validate_tensor_arguments,
                             observe_first_warmup,argument_structure)


class CPUTensor:
    pass


class CPUFakeTensor(CPUTensor):
    pass


class TensorCallABICPU(unittest.TestCase):
    def setUp(self):
        self.arguments=[CPUTensor() for _ in range(19)];self.calls=[];self.records=[]
        self.phase='real-warmup';self.output=object()
        def sdk(*args,**kwargs):self.calls.append((args,kwargs));return self.output
        sdk._boxed_call=True;self.sdk=sdk
        self.boxed=boxed_sdk_compiler_adapter(sdk,expected_arity=19,tensor_type=CPUTensor,
            fake_type=CPUFakeTensor,get_phase=lambda:self.phase,observe=self.records.append)

    def test_single_layer_preserves_all_tensor_objects_order_and_keyword_values(self):
        option=object();nothing=None
        self.assertIs(self.boxed(self.arguments,option=option,nothing=nothing),self.output)
        args,kwargs=self.calls[0]
        self.assertEqual(len(args),19)
        self.assertTrue(all(before is after for before,after in zip(self.arguments,args)))
        self.assertIs(kwargs['option'],option);self.assertIsNone(kwargs['nothing'])
        self.assertEqual(list(kwargs),['option','nothing'])

    def test_sdk_callable_and_advertised_flag_are_not_modified(self):
        original=self.sdk;self.boxed(self.arguments)
        self.assertIs(self.sdk,original);self.assertTrue(self.sdk._boxed_call)
        self.assertTrue(self.boxed._boxed_call)
        self.assertTrue(self.records[0]['sdk_advertised_boxed_call'])
        self.assertFalse(self.records[0]['kwargs_filtered'])

    def test_missing_argument_is_rejected_before_sdk(self):
        with self.assertRaisesRegex(TypeError,'arity differs'):self.boxed(self.arguments[:-1])
        self.assertEqual(self.calls,[])

    def test_extra_argument_is_rejected_before_sdk(self):
        with self.assertRaisesRegex(TypeError,'arity differs'):self.boxed(self.arguments+[CPUTensor()])
        self.assertEqual(self.calls,[])

    def test_double_boxed_list_is_rejected_before_sdk(self):
        with self.assertRaisesRegex(TypeError,'arity differs'):self.boxed([self.arguments])
        self.assertEqual(self.calls,[])

    def test_nested_list_element_is_rejected_at_matching_arity(self):
        wrong=[*self.arguments[:-1],[self.arguments[-1]]]
        with self.assertRaisesRegex(TypeError,'nested list'):self.boxed(wrong)
        self.assertEqual(self.calls,[])

    def test_none_scalar_boolean_and_fake_tensor_are_rejected(self):
        for value in (None,1,True,CPUFakeTensor()):
            wrong=[value,*self.arguments[1:]]
            with self.assertRaisesRegex(TypeError,'real Tensor'):self.boxed(wrong)
        self.assertEqual(self.calls,[])

    def test_tuple_or_unboxed_call_at_boxed_boundary_is_rejected(self):
        with self.assertRaisesRegex(TypeError,'exactly one boxed list'):self.boxed(tuple(self.arguments))
        with self.assertRaises(TypeError):self.boxed(*self.arguments)
        self.assertEqual(self.calls,[])

    def test_dynamo_unboxed_tensor_tuple_arity_and_structure(self):
        validate_tensor_arguments(tuple(self.arguments),19,CPUTensor,CPUFakeTensor)
        with self.assertRaisesRegex(TypeError,'arity differs'):
            validate_tensor_arguments((self.arguments,),19,CPUTensor,CPUFakeTensor)
        structure=argument_structure(tuple(self.arguments),{},CPUTensor)
        self.assertEqual(structure['object_ids'],[id(value) for value in self.arguments])
        self.assertEqual(structure['tensor_mask'],[True]*19)

    def test_observation_occurs_once_in_warmup_and_never_in_capture(self):
        self.phase='cold-capture';self.boxed(self.arguments)
        self.assertEqual(self.records,[])
        self.phase='real-warmup';self.boxed(self.arguments);self.boxed(self.arguments)
        self.assertEqual(len(self.records),1)
        self.phase='cold-capture';self.boxed(self.arguments)
        self.assertEqual(len(self.records),1)
        seen=[];observe=observe_first_warmup('dynamo',lambda:self.phase,seen.append)
        observe(tuple(self.arguments),{},CPUTensor);self.assertEqual(seen,[])

    def test_capture_failure_retains_arguments_without_observation_or_value_read(self):
        self.phase='cold-capture'
        def fail(*args,**kwargs):raise RuntimeError('CPU-only SDK failure')
        adapter=boxed_sdk_compiler_adapter(fail,expected_arity=19,tensor_type=CPUTensor,
            get_phase=lambda:self.phase,observe=self.records.append)
        with self.assertRaisesRegex(RuntimeError,'SDK failure') as result:adapter(self.arguments,kept=None)
        arguments,kwargs,tensor_type,boundary=result.exception.tensor_call_abi_failure
        self.assertEqual(tuple(self.arguments),arguments);self.assertEqual(kwargs,{'kept':None})
        self.assertEqual(self.records,[])


if __name__=='__main__':unittest.main()
