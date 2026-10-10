"""Reversible metadata trials on the installed vendor build implementation.

Only invariant model fields are cached. Native batch sharing, group-local
buffers, DeviceMetadataTask staging and all dynamic scheduler inputs survive.
Source anchors fail closed if the vendor implementation changes.
"""
import hashlib
import inspect
import textwrap

import goal20_patches as goal
from metadata_kernels import prepare_slots, prepare_ring

COUNTS = {}
SPEC_COUNTS = {}
SOURCE_SHA256 = None


def install(model, slot_prepare=None):
    global SOURCE_SHA256
    from vllm_ascend.attention import dsa_v41 as vendor
    cls = vendor.DeepseekV41MetadataBuilder
    original = cls.build
    source = textwrap.dedent(inspect.getsource(original))
    SOURCE_SHA256 = hashlib.sha256(source.encode()).hexdigest()
    static_start = '    text_config = self.vllm_config.model_config.hf_text_config\n'
    static_end = '    operator_ratio = 0 if cache_kind == "swa" else ratio\n'
    assert source.count(static_start) == source.count(static_end) == 1
    left = source.index(static_start); right = source.index(static_end)
    static_source = source[:left] + (
        '    window_size, n_local_heads, head_dim, index_topk, index_n_heads, index_head_dim = self._goal20_static\n'
    ) + source[right:]
    qli_heads = 'int(_config_value(text_config, "index_n_heads"))'
    qli_dim = 'int(_config_value(text_config, "index_head_dim"))'
    assert static_source.count(qli_heads) == static_source.count(qli_dim) == 1
    static_source = static_source.replace(qli_heads, 'index_n_heads').replace(qli_dim, 'index_head_dim')
    slot_start = '            active_slots = common.slot_mapping[:num_input_tokens]\n'
    slot_end = '            shared[slot_key] = prepared_slots\n'
    assert static_source.count(slot_start) == static_source.count(slot_end) == 1
    left = static_source.index(slot_start); right = static_source.index(slot_end)
    fused_source = static_source[:left] + (
        '            prepared_slots = _goal20_prepare_slots(self, common, positions, num_input_tokens,\n'
        '                num_actual_reqs, num_actual_tokens, compressed, ratio, spec.storage_block_size,\n'
        '                kwargs.get("skip_ring_state_update", False), batch_shared)\n'
    ) + static_source[right:]
    ring_start = '        def build_c2_metadata() -> None:\n'
    ring_end = '        compressor_group = self._publish_task(\n'
    assert fused_source.count(ring_start) == fused_source.count(ring_end) == 1
    left = fused_source.index(ring_start); right = fused_source.index(ring_end)
    ring_source = fused_source[:left] + (
        '        def build_c2_metadata() -> None:\n'
        '            _goal20_prepare_ring(self, common, input_positions, seq_lens, num_reqs,\n'
        '                num_actual_reqs, num_actual_tokens, num_input_tokens,\n'
        '                kwargs.get("skip_ring_state_update", False), full_source_cos, full_source_sin)\n\n'
    ) + fused_source[right:]
    namespace = dict(original.__globals__, _goal20_prepare_slots=slot_prepare or prepare_slots,
                     _goal20_prepare_ring=prepare_ring)
    exec(compile(static_source, '<goal20_metadata_static>', 'exec'), namespace)
    static_build = namespace['build']
    exec(compile(fused_source, '<goal20_metadata_slots>', 'exec'), namespace)
    fused_build = namespace['build']
    exec(compile(ring_source, '<goal20_metadata_ring>', 'exec'), namespace)
    ring_build = namespace['build']
    spec_start = '    spec = self.kv_cache_spec\n'
    spec_end = '    num_reqs = int(getattr(common, "num_reqs", common.seq_lens.shape[0]))\n'
    assert ring_source.count(spec_start) == ring_source.count(spec_end) == 1
    left, right = ring_source.index(spec_start), ring_source.index(spec_end)
    spec_source = ring_source[:left] + (
        '    common = common_attn_metadata\n'
        '    (spec, is_compressor_state, ratio, cache_kind, compressed, plane_ratio,\n'
        '     operator_ratio, storage_block_size, slot_key, smla_key, qli_key) = self._goal20_spec_static\n'
    ) + ring_source[right:]
    for anchor in (
        '        compressed = cache_kind in {"long_kv", "index_k"}\n',
        '        plane_ratio = ratio if compressed else 1\n',
        '        operator_ratio = 0 if cache_kind == "swa" else ratio\n',
    ):
        # dedented method body uses four spaces for these assignments.
        anchor = anchor[4:]
        assert spec_source.count(anchor) == 1, anchor
        spec_source = spec_source.replace(anchor, '')
    slot_key_anchor = '        slot_key = f"slot:c{ratio}:b{spec.storage_block_size}"\n'
    assert spec_source.count(slot_key_anchor) == 1
    spec_source = spec_source.replace(slot_key_anchor, '')
    spec_source = spec_source.replace('spec.storage_block_size', 'storage_block_size')
    for before, after in (('f"smla:c{operator_ratio}"', 'smla_key'), ('f"qli:c{ratio}"', 'qli_key')):
        assert spec_source.count(before) == 1, before
        spec_source = spec_source.replace(before, after)
    exec(compile(spec_source, '<goal20_metadata_spec_static>', 'exec'), namespace)
    spec_build = namespace['build']
    old_init = cls.__init__

    def initialize(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        config = self.vllm_config.model_config.hf_text_config
        get = vendor._config_value
        self._goal20_static = (
            int(get(config, 'sliding_window', 0)),
            int(get(config, 'num_attention_heads')) // self.vllm_config.parallel_config.tensor_parallel_size,
            int(get(config, 'head_dim')), int(get(config, 'index_topk')),
            int(get(config, 'index_n_heads')), int(get(config, 'index_head_dim')))
        spec = self.kv_cache_spec
        compressor = isinstance(spec, vendor.DeepseekV41CompressorStateSpec)
        ratio = getattr(spec, 'compress_ratio', 1)
        if isinstance(spec, vendor.DeepseekV41SWASpec):
            kind = 'swa'
        elif isinstance(spec, vendor.DeepseekV41FullSpec):
            kind = 'long_kv'
        elif isinstance(spec, vendor.DeepseekV41IndexerSpec):
            kind = 'index_k'
        elif compressor:
            kind = 'compressor_state'
        else:
            raise TypeError(f'Unsupported V4.1 cache spec: {type(spec).__name__}')
        compressed = kind in ('long_kv', 'index_k')
        operator_ratio = 0 if kind == 'swa' else ratio
        # Compressor metadata still publishes its storage block size even
        # though it has no token-slot conversion or slot-sharing key.
        block = spec.storage_block_size
        self._goal20_spec_static = (spec, compressor, ratio, kind, compressed,
            ratio if compressed else 1, operator_ratio, block,
            None if compressor else f'slot:c{ratio}:b{block}',
            f'smla:c{operator_ratio}', f'qli:c{ratio}')

    def build(self, *args, **kwargs):
        arm = goal.ARM
        COUNTS[arm] = COUNTS.get(arm, 0) + 1
        if goal.enabled('metadata_spec') and self._supports_device_ops:
            # The cache-spec object belongs to the initialized model. A new
            # spec must take the existing ring builder, never an older tuple.
            if self._goal20_spec_static[0] is self.kv_cache_spec:
                SPEC_COUNTS[arm] = SPEC_COUNTS.get(arm, 0) + 1
                return spec_build(self, *args, **kwargs)
        if arm == 'mdstatic':
            return static_build(self, *args, **kwargs)
        if arm == 'mdslots' and self._supports_device_ops:
            return fused_build(self, *args, **kwargs)
        if (arm in ['mdall', 'mdfull'] or goal.enabled('metadata_all')) and self._supports_device_ops:
            return ring_build(self, *args, **kwargs)
        return original(self, *args, **kwargs)

    cls.__init__ = initialize
    cls.build = build
    for arm in ['mdstatic', 'mdslots', 'mdall']:
        goal.CONFIGS[arm] = set(goal.CONFIGS['gmmact'])
    print('METADATA_PATCH_INSTALL', {'vendor_build_sha256': SOURCE_SHA256, 'arms': ['mdstatic', 'mdslots', 'mdall']}, flush=True)


def stats(worker):
    return {'build_calls': dict(COUNTS), 'vendor_build_sha256': SOURCE_SHA256,
            'spec_static_build_calls': dict(SPEC_COUNTS),
            'dynamic_values_cached_across_steps': False, 'slot_buffers': 'group-local persistent'}
