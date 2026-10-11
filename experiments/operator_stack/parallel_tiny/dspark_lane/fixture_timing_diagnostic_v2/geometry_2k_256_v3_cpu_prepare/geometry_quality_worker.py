"""Full native audit with an independent 2K/256 paged-SWA observer."""
from o_proj_cache_fixture_worker import GraphCacheFixtureWorker


class GeometryQualityWorker(GraphCacheFixtureWorker):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        import geometry_cache_probe
        print("GEOMETRY2K_CACHE_PROBE_INSTALLED", geometry_cache_probe.install(self), flush=True)
        return result

    def snapshot_geometry_step(self, is_decode, draft_replayed):
        # String RPC resolves this bound method through WorkerWrapperBase.
        # Keep the probe's installed-worker identity check strict.
        import geometry_cache_probe
        return geometry_cache_probe.snapshot_step(self, is_decode, draft_replayed)
