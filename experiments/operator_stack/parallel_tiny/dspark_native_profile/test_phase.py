"""CPU controls reject false prefill/cleanup profile windows."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from parallel_tiny.dspark_native_profile.phase import is_real_decode


def scheduler(done, *, new=False):
    request = NS(req_id='real-core', prompt_token_ids=[1]*2048, num_computed_tokens=done)
    output = NS(num_scheduled_tokens={'real-core':6}, total_num_scheduled_tokens=6,
                scheduled_new_reqs=[request] if new else [],
                scheduled_cached_reqs=NS(req_ids=[] if new else ['real-core'],
                    num_computed_tokens=[] if new else [done]))
    return output, {'real-core':request}


class PhaseContract(unittest.TestCase):
    def test_new_prefill_never_starts_decode_profile(self):
        self.assertFalse(is_real_decode(*scheduler(0, new=True)))

    def test_chunked_prefill_never_starts_decode_profile(self):
        self.assertFalse(is_real_decode(*scheduler(1024)))

    def test_actual_computed_boundary_starts_decode_profile(self):
        self.assertTrue(is_real_decode(*scheduler(2048)))

    def test_zero_cleanup_is_not_a_profile_step(self):
        self.assertFalse(is_real_decode(NS(num_scheduled_tokens={}, total_num_scheduled_tokens=0), {}))

    def test_fake_empty_nonzero_schedule_rejected(self):
        with self.assertRaises(ValueError):
            is_real_decode(NS(num_scheduled_tokens={}, total_num_scheduled_tokens=6), {})

    def test_wrong_core_request_binding_rejected(self):
        output, _ = scheduler(2048)
        with self.assertRaises(ValueError):
            is_real_decode(output, {'another-core':NS(prompt_token_ids=[1]*2048)})


if __name__ == '__main__':
    unittest.main()
