"""Signature preflight for the selected native SDK, no compatibility mutation."""
import ast
import hashlib
import inspect
from pathlib import Path


def inspect_native_factory():
    from vllm_ascend.models.deepseek_v4 import model
    path=Path(inspect.getfile(model))
    code=path.read_text()
    tree=ast.parse(code)
    needs_bias=any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                   and node.func.id=='FusedMoEFactory' and any(k.arg=='bias_vl' for k in node.keywords)
                   for node in ast.walk(tree))
    factory=model.FusedMoEFactory
    signature=inspect.signature(factory)
    supports_bias=('bias_vl' in signature.parameters or
                   any(p.kind==inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()))
    factory_path=Path(inspect.getfile(factory))
    return {'native_model_source':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},
            'factory_source':{'path':str(factory_path),'sha256':hashlib.sha256(factory_path.read_bytes()).hexdigest()},
            'factory_signature':str(signature),'constructor_passes_bias_vl':needs_bias,
            'factory_accepts_bias_vl':supports_bias,'compatible':not needs_bias or supports_bias,
            'mutation_performed':False,'device_execution_performed':False}
