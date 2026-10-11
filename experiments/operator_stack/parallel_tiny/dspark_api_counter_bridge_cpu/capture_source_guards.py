"""Read installed Python source without importing vLLM, torch, or using a device.

Run this small script with remote docker exec python on stdin. JSON stdout is
the only artifact transferred; the full SDK tree remains on the remote host.
"""
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path('/vllm-workspace/vllm/vllm')
SELECT = {
    'engine/protocol.py': ['EngineClient.collective_rpc'],
    'v1/engine/async_llm.py': ['AsyncLLM.from_vllm_config', 'AsyncLLM._add_request',
        'AsyncLLM.add_request', 'AsyncLLM.generate', 'AsyncLLM.abort',
        'AsyncLLM.collective_rpc'],
    'v1/engine/input_processor.py': ['InputProcessor.assign_request_id'],
    'v1/engine/output_processor.py': ['OutputProcessor.add_request',
        'OutputProcessor.abort_requests', 'OutputProcessor._finish_request'],
    'v1/engine/core_client.py': ['AsyncMPClient.collective_rpc_async'],
    'v1/engine/core.py': ['EngineCore.collective_rpc'],
    'v1/executor/multiproc_executor.py': ['WorkerProc.worker_busy_loop'],
    'entrypoints/serve/engine/serving.py': ['BaseServing._base_request_id'],
    'entrypoints/openai/completion/serving.py': ['OpenAIServingCompletion._create_completion',
        'OpenAIServingCompletion.request_output_to_completion_response'],
    'entrypoints/openai/completion/protocol.py': ['CompletionRequest',
        'CompletionResponseChoice', 'CompletionResponse'],
    'entrypoints/openai/completion/api_router.py': ['completion', 'create_completion'],
    'entrypoints/openai/api_server.py': ['build_async_engine_client_from_engine_args',
        'build_app', 'init_app_state', 'build_and_serve'],
    'entrypoints/serve/utils/server_utils.py': ['lifespan'],
    'entrypoints/serve/utils/api_utils.py': ['with_cancellation', 'listen_for_disconnect'],
    'entrypoints/openai/cli_args.py': ['make_arg_parser'],
    'distributed/parallel_state.py': ['GroupCoordinator.barrier'],
}


def definitions(tree):
    out = {}
    def visit(node, prefix=''):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + child.name
                out[name] = child
                visit(child, name + '.')
            else:
                visit(child, prefix)
    visit(tree)
    return out


def main():
    records = []
    for relative, names in SELECT.items():
        path = ROOT / relative
        data = path.read_bytes()
        source = data.decode()
        lines = source.splitlines(keepends=True)
        defs = definitions(ast.parse(source))
        fragments = {}
        for name in names:
            node = defs.get(name)
            if node is None:
                # Explicitly record discovery misses rather than inventing an ABI.
                fragments[name] = {'missing': True,
                    'available_candidates': [n for n in defs if n.endswith('.' + name.split('.')[-1]) or n == name.split('.')[-1]]}
                continue
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            fragment = ''.join(lines[start - 1:node.end_lineno])
            fragments[name] = {'first_line': start, 'last_line': node.end_lineno,
                'source_sha256': hashlib.sha256(fragment.encode()).hexdigest(),
                'source': fragment}
        records.append({'path': str(path), 'relative': relative,
            'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
            'fragments': fragments})
    print(json.dumps({'source_only': True, 'device_calls': 0,
        'guard_format': 1, 'files': records}, indent=2))


if __name__ == '__main__':
    main()
