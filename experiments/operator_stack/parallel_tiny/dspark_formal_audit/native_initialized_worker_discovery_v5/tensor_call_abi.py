"""Private AOT compiler ABI adapter; no torch/SDK import or Tensor value read."""


def argument_structure(arguments, kwargs, tensor_type):
    return {'container':type(arguments).__name__,'arity':len(arguments),
            'types':[type(value).__module__+'.'+type(value).__qualname__ for value in arguments],
            'tensor_mask':[isinstance(value,tensor_type) for value in arguments],
            'object_ids':[id(value) for value in arguments],
            'keyword_names':list(kwargs),
            'keyword_types':{name:type(value).__module__+'.'+type(value).__qualname__ for name,value in kwargs.items()}}


def validate_tensor_arguments(arguments, expected_arity, tensor_type, fake_type=None):
    if len(arguments)!=expected_arity:
        raise TypeError('Runtime Tensor call ABI arity differs: expected '+str(expected_arity)+', got '+str(len(arguments)))
    if any(not isinstance(value,tensor_type) or (fake_type is not None and isinstance(value,fake_type))
           for value in arguments):
        raise TypeError('Runtime Tensor call ABI requires original real Tensor arguments, no nested list/None/scalar/FakeTensor')


def observe_first_warmup(boundary, get_phase, observe):
    seen=False
    def record(arguments, kwargs, tensor_type, metadata=None):
        nonlocal seen
        if not seen and get_phase()=='real-warmup':
            seen=True
            observe({'boundary':boundary,**argument_structure(arguments,kwargs,tensor_type),**(metadata or {})})
    return record


def boxed_sdk_compiler_adapter(sdk_callable, *, expected_arity, tensor_type, fake_type=None,
                               get_phase=lambda:None, observe=lambda record:None):
    """AOT calls one boxed list; original SDK Dynamo callable gets *Tensor args.

    SDK return metadata is left untouched. This new compiler-boundary callable
    declares its own boxed ABI and unpacks exactly once, preserving all Tensor
    objects, their order, and every keyword value. No graph/assert is replaced.
    """
    record=observe_first_warmup('outer_aot_boxed_fw_to_sdk_unboxed',get_phase,observe)
    advertised_boxed=getattr(sdk_callable,'_boxed_call',None)
    def boxed(arguments, **kwargs):
        actual=arguments
        try:
            if type(arguments) is not list:
                raise TypeError('AOT compiler ABI requires exactly one boxed list of Tensor arguments')
            actual=tuple(arguments)
            validate_tensor_arguments(actual,expected_arity,tensor_type,fake_type)
            record(actual,kwargs,tensor_type,{'incoming_boxed_container':'list',
                   'sdk_advertised_boxed_call':advertised_boxed,'adapter_boxed_call':True,
                   'sdk_call_abi':'unboxed','tensor_objects_replaced':False,'tensor_order_changed':False,
                   'kwargs_filtered':False})
            return sdk_callable(*actual,**kwargs)
        except BaseException as exc:
            # Retain references only. The owner renders type facts after any
            # capture context has unwound; no capture-time log or value read.
            exc.tensor_call_abi_failure=(actual,kwargs,tensor_type,'outer_aot_boxed_fw_to_sdk_unboxed')
            raise
    boxed._boxed_call=True
    return boxed
