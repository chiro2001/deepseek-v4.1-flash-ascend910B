"""Source-only standard bootstrap insertion; no SDK or device imports."""
import ast
import copy
import hashlib

EXECUTOR_SHA = 'ceb3477473bdb1de36e01687e0b2083ad239d42b1ddc2ec8cec16ba2c477efa6'


def derive_executor_init(source):
    if hashlib.sha256(source.encode()).hexdigest() != EXECUTOR_SHA:
        raise ValueError('Original standard MP bootstrap source changed')
    cls = next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name == 'MultiprocExecutor')
    original = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == '_init_executor')
    node = copy.deepcopy(original);counts = {'birth':0,'failure':0}
    class Capture(ast.NodeTransformer):
        def visit_Expr(self,statement):
            if isinstance(statement.value,ast.Call) and ast.unparse(statement.value.func) == 'unready_workers.append':
                counts['birth'] += 1
                return [statement,ast.Expr(ast.Call(ast.Attribute(ast.Name('self',ast.Load()),'_v10_capture',ast.Load()),
                                                   [ast.Name('unready_worker_handle',ast.Load())],[]))]
            if isinstance(statement.value,ast.Call) and ast.unparse(statement.value.func) == 'self._ensure_worker_termination':
                counts['failure'] += 1
                failed = ast.Assign(targets=[ast.Attribute(ast.Name('self',ast.Load()),'_v10_failed',ast.Store())],value=ast.Constant(True))
                return [failed,statement]
            return statement
    node = Capture().visit(node)
    if counts != {'birth':1,'failure':1}:raise ValueError('Immediate post-start/failure cleanup boundary changed')
    class Strip(ast.NodeTransformer):
        def visit_Expr(self,statement):
            if isinstance(statement.value,ast.Call) and ast.unparse(statement.value.func) == 'self._v10_capture':return None
            return self.generic_visit(statement)
        def visit_Assign(self,statement):
            if any(ast.unparse(t) == 'self._v10_failed' for t in statement.targets):return None
            return self.generic_visit(statement)
    if ast.dump(Strip().visit(copy.deepcopy(node)),include_attributes=False) != ast.dump(original,include_attributes=False):
        raise ValueError('Private bootstrap changed original standard MP AST')
    return ast.fix_missing_locations(node),{'source_sha256':EXECUTOR_SHA,'native_bootstrap_AST_preserved':True,
        'immediate_after_each_original_append':True,'failed_cleanup_bypasses_success_grace':True,'counts':counts}
