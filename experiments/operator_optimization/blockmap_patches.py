"""Single-variable trial of the existing guarded multi-group slot kernel."""
import os

import goal20_patches as goal

COUNTS = {}
GROUPS = {}


def install(model):
    from vllm_ascend.worker import block_table as vendor
    assert hasattr(vendor.MultiGroupBlockTable, '_v41_try_fused'), 'Installed vendor lacks the guarded fused kernel'
    original_mode = vendor._v41_slot_map_fused_mode
    assert original_mode() == 'off', 'Reference already enables multi-group fusion'
    original_try = vendor.MultiGroupBlockTable._v41_try_fused

    def mode():
        if goal.ARM in ['mdlaunch', 'mdfull'] or goal.enabled('blockmap'):
            return 'verify' if os.getenv('OPT_BLOCKMAP_VERIFY') == '1' else 'on'
        return original_mode()

    def attempt(self, *args, **kwargs):
        result = original_try(self, *args, **kwargs)
        row = COUNTS.setdefault(goal.ARM, {'used': 0, 'fallback': 0})
        row['used' if result else 'fallback'] += 1
        if result:
            GROUPS[goal.ARM] = list(self._v41_fused_state['idx'])
        return result

    vendor._v41_slot_map_fused_mode = mode
    vendor.MultiGroupBlockTable._v41_try_fused = attempt
    for arm in ['mdlaunch', 'mdfull']:
        goal.CONFIGS[arm] = set(goal.CONFIGS['gmmact'])
    print('BLOCKMAP_PATCH_INSTALL', {'arms': ['mdlaunch', 'mdfull'],
                                    'verify': os.getenv('OPT_BLOCKMAP_VERIFY') == '1'}, flush=True)


def stats(worker):
    return {'fused_calls': {arm: dict(rows) for arm, rows in COUNTS.items()},
            'fused_group_indices': dict(GROUPS), 'verify': os.getenv('OPT_BLOCKMAP_VERIFY') == '1'}
