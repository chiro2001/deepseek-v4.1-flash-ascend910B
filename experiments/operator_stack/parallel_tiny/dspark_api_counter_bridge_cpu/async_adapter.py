"""Transparent private AsyncLLM subclass factory; no global monkey patch."""
from __future__ import annotations


def make_async_llm_adapter(stock_async_llm):
    class ApiCounterAsyncLLM(stock_async_llm):
        async def _add_request(self, request, prompt, parent_req, index, queue):
            bridge = self.dspark_api_bridge
            try:
                # The stock assign_request_id has already run. Never infer the
                # random suffix or query OutputProcessor after request cleanup.
                await bridge.bind_core(self, request, parent_req, index)
                bridge.note_core_submit(request.request_id)
                return await super()._add_request(request, prompt, parent_req, index, queue)
            except BaseException as exc:
                bridge.fail('core_submit', exc)
                raise

        async def generate(self, prompt, sampling_params, request_id, **kwargs):
            bridge = self.dspark_api_bridge
            completed = False
            try:
                bridge.note_generate(request_id, sampling_params)
                async for output in super().generate(prompt, sampling_params, request_id, **kwargs):
                    bridge.note_native_output(output)
                    completed = completed or bool(output.finished)
                    yield output
                if not completed:
                    raise ValueError('Native generate ended without a finished RequestOutput')
            except BaseException as exc:
                bridge.fail('generate', exc)
                raise

        async def abort(self, request_id, internal=False):
            self.dspark_api_bridge.fail('abort', RuntimeError(f'Native abort: {request_id!r}'))
            return await super().abort(request_id, internal=internal)

    ApiCounterAsyncLLM.__name__ = 'ApiCounterAsyncLLM'
    return ApiCounterAsyncLLM
