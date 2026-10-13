"""Same frozen native O-cache fixture for pure eager and private graph workers."""
from graph_candidate_worker import GraphCandidateWorker
from native_route_diagnostic_worker import NativeRouteDiagnosticWorker


class NativeCacheFixtureWorker(NativeRouteDiagnosticWorker):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        import o_proj_cache_fixture as fixture
        print("FIXED_NATIVE_O_CACHE_INITIALIZED", fixture.initialize_before_compile(self), flush=True)
        return result

    def compile_or_warm_up_model(self, *args, **kwargs):
        result = super().compile_or_warm_up_model(*args, **kwargs)
        import o_proj_cache_fixture as fixture
        print("FIXED_NATIVE_O_CACHE_AFTER_CAPTURE", fixture.validate_all(self, "after_capture"), flush=True)
        return result


class GraphCacheFixtureWorker(GraphCandidateWorker):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        import o_proj_cache_fixture as fixture
        print("FIXED_GRAPH_O_CACHE_INITIALIZED", fixture.initialize_before_compile(self), flush=True)
        return result

    def compile_or_warm_up_model(self, *args, **kwargs):
        result = super().compile_or_warm_up_model(*args, **kwargs)
        import o_proj_cache_fixture as fixture
        print("FIXED_GRAPH_O_CACHE_AFTER_CAPTURE", fixture.validate_all(self, "after_capture"), flush=True)
        return result
