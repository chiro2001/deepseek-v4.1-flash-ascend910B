"""Expose SDK server introspection while retaining the frozen ASGI protocol."""
from parallel_tiny.dspark_api_counter_bridge_cpu.http_isolation import PrivateOnlyASGI


class NativeHttpFacade(PrivateOnlyASGI):
    @property
    def routes(self):
        return self.app.routes

    @property
    def state(self):
        return self.app.state
