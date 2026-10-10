"""Private eight-layer cache contract; never patch the installed source tree."""
import hashlib
import importlib.machinery
import inspect
import os
import sys

MODULE = 'vllm_ascend.core.deepseek_v41'
SOURCE_SHA256 = '328f2d755ec7a6cdcd7f0976d7f606e44d2aca0fd81941e9c91cc1bc943384ab'


def install(module):
    assert os.getenv('STACK_TINY_PROFILE') == '1'
    source = inspect.getsource(module.plan_cache_slots)
    digest = hashlib.sha256(source.encode()).hexdigest()
    assert digest == SOURCE_SHA256, ('Unsupported installed cache planner', digest)
    # Keep payload geometry, disjoint KV/index offsets, dtype checks, state
    # capacity, unique complete resource coverage, and every allocator intact.
    # Replace only the three hard-coded topology checks with the exact tiny
    # topology. This does not admit arbitrary/incomplete cache resources.
    for before, after in (
        ('list(map(_layer_number, full)) != [2, 8, 14, 20]',
         'list(map(_layer_number, full)) != [2, 4, 5]'),
        ('list(map(_layer_number, state)) != [2, 8, 14]',
         'list(map(_layer_number, state)) != [2, 4]'),
        ('list(map(_layer_number, swa)) != list(range(40))',
         'list(map(_layer_number, swa)) != list(range(8))'),
    ):
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    source = source.replace('four shared layer slots', 'three diagnostic layer slots')
    source = source.replace('KV source layers 2, 8, 14, 20', 'diagnostic KV source layers 2, 4, 5')
    source = source.replace('state source layers 2, 8, 14', 'diagnostic state source layers 2, 4')
    source = source.replace('exactly 40 ordered SWA resources', 'exactly 8 diagnostic SWA resources')
    namespace = dict(vars(module))
    exec(compile(source, '<tiny_tp8_exact_cache_plan>', 'exec'), namespace)
    module.plan_cache_slots = namespace['plan_cache_slots']
    module.TINY_CACHE_PLAN_SOURCE_SHA256 = digest


def install_import_hook():
    assert os.getenv('STACK_TINY_PROFILE') == '1'

    class Loader:
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            self.original.exec_module(module)
            install(module)

        def __getattr__(self, name):
            return getattr(self.original, name)

    class Finder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname != MODULE:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            assert spec is not None and spec.loader is not None
            spec.loader = Loader(spec.loader)
            return spec

    assert MODULE not in sys.modules, 'Install before the core cache module is imported'
    sys.meta_path.insert(0, Finder())
