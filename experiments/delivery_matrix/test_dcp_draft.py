"""CPU controls for target-vs-expanded draft coordinates and parallel CP gates."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

spec = importlib.util.spec_from_file_location("patch_dcp_draft_test", Path(__file__).with_name("build_dcp_draft.py"))
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)


class Indices(list):
    def __getitem__(self, key):
        value = super().__getitem__(key)
        return Indices(value) if isinstance(key, slice) else value

    def __sub__(self, other):
        return Indices(v - (other[i] if isinstance(other, list) else other) for i, v in enumerate(self))


class Manager:
    def prepare_spec_decode_first_pass_inputs(self, **kwargs):
        self.kwargs = kwargs
        kwargs["common_attn_metadata"].context_parallel_metadata = kwargs["long_seq_metadata"]
        return NS(long_seq_args=("query-lens", kwargs["token_indices_to_sample"]))


DS_STUB = '''def first_pass(self, token_indices_to_sample, cad, num_rejected_tokens_gpu=None,
               long_seq_metadata="target-meta", req_scheduled_tokens=None,
               num_prefill_reqs=1, num_decode_reqs=0):
        # [DYNAMIC-SPEC] replay 路径：先按本步 K 归一 num_query_per_req，
        num_query_total = 5
        cad.query_start_loc = Indices([0, 5])
        token_indices_to_sample = Indices([0, 1, 2, 3, 4])
        return num_query_total, token_indices_to_sample, cad, None
'''

BASE_STUB = '''def _propose(self, dcp_manager):
    clone_path = False
    dcp_mtp_inputs = None
    if self.dcp_size > 1 and self.use_cuda_graph:
        clone_path = True
    if dcp_manager is not None:
            dcp_mtp_inputs = dcp_manager.prepare_spec_decode_mtp_drafting_inputs()
    return clone_path, dcp_mtp_inputs
'''


class DraftPatchTests(unittest.TestCase):
    def proposer(self, manager):
        return NS(runner=NS(dcp_manager=manager, input_batch=NS(req_ids=["real-request"]), logits_indices=[2047]),
                  input_ids=list(range(5)), positions=list(range(5)), _dflash_hidden_states=list(range(5)),
                  uses_mrope=False)

    def run_first_pass(self, indices=None, rejects=None, manager=None):
        namespace = {"Indices": Indices}
        exec(patcher.patch_dspark(DS_STUB), namespace)
        return namespace["first_pass"](self.proposer(manager), indices,
                                       NS(query_start_loc=Indices([0, 2048])), rejects)

    def test_prefill_sample_index_is_saved_before_parallel_expansion(self):
        manager = Manager()
        total, expanded, metadata, cp = self.run_first_pass(manager=manager)
        self.assertEqual(total, 5)
        self.assertEqual(expanded, [0, 1, 2, 3, 4])
        self.assertEqual(manager.kwargs["token_indices_to_sample"], [2047])
        self.assertEqual(metadata.query_start_loc, [0, 5])
        self.assertEqual(cp, ("query-lens", [2047]))

    def test_async_rejections_use_original_target_coordinates(self):
        manager = Manager()
        self.run_first_pass(rejects=Indices([2]), manager=manager)
        self.assertEqual(manager.kwargs["token_indices_to_sample"], [2045])

    def test_prepared_sample_index_is_not_double_corrected(self):
        manager = Manager()
        self.run_first_pass(indices=Indices([2045]), rejects=Indices([2]), manager=manager)
        self.assertEqual(manager.kwargs["token_indices_to_sample"], [2045])

    def test_non_dcp_parallel_layout_keeps_original_contract(self):
        result = self.run_first_pass()
        self.assertEqual(result[0], 5)
        self.assertIsNone(result[-1])

    def test_unused_mtp_step_metadata_is_not_required_for_parallel_drafts(self):
        namespace = {}
        exec(patcher.patch_base(BASE_STUB), namespace)
        manager = NS(prepare_spec_decode_mtp_drafting_inputs=lambda: "mtp-step")
        clone, mtp = namespace["_propose"](NS(dcp_size=8, use_cuda_graph=True, parallel_drafting=True), manager)
        self.assertFalse(clone)
        self.assertIsNone(mtp)
        clone, mtp = namespace["_propose"](NS(dcp_size=8, use_cuda_graph=True, parallel_drafting=False), manager)
        self.assertTrue(clone)
        self.assertEqual(mtp, "mtp-step")

    def test_ambiguous_source_anchor_fails_closed(self):
        with self.assertRaises(ValueError):
            patcher.patch_dspark(DS_STUB + DS_STUB)


if __name__ == "__main__":
    unittest.main()
