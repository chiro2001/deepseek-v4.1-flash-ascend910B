"""Selectively port the historical DSpark/DCP fixes into the best draft files.

The best TARGETPOS, graph-metadata, BI and async bridge fixes are preserved.
Parallel drafts keep their expanded token layout; only CP bookkeeping is added.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Expected one exact draft ABI anchor: " + old[:120])
    return text.replace(old, new, 1)


def patch_dspark(text):
    text = replace_once(text,
        "        # [DYNAMIC-SPEC] replay 路径：先按本步 K 归一 num_query_per_req，",
        """        # Keep sample indices in the TARGET coordinate system before
        # parallel draft expansion mutates cad.query_start_loc in place.
        original_sample_indices = token_indices_to_sample
        if original_sample_indices is None:
            original_sample_indices = cad.query_start_loc[1:] - 1
            if num_rejected_tokens_gpu is not None:
                original_sample_indices = original_sample_indices - num_rejected_tokens_gpu
        # [DYNAMIC-SPEC] replay 路径：先按本步 K 归一 num_query_per_req，""")
    return replace_once(text,
        "        return num_query_total, token_indices_to_sample, cad, None",
        """        long_seq_args = None
        dcp_manager = getattr(self.runner, "dcp_manager", None)
        if dcp_manager is not None:
            first_pass = dcp_manager.prepare_spec_decode_first_pass_inputs(
                input_ids=self.input_ids[:num_query_total],
                target_positions=self.positions[:num_query_total],
                target_hidden_states=self._dflash_hidden_states[:num_query_total],
                token_indices_to_sample=original_sample_indices,
                common_attn_metadata=cad, long_seq_metadata=long_seq_metadata,
                req_scheduled_tokens=req_scheduled_tokens,
                req_ids=self.runner.input_batch.req_ids,
                logits_indices=self.runner.logits_indices, num_tokens=num_query_total,
                num_prefill_reqs=num_prefill_reqs, num_decode_reqs=num_decode_reqs,
                uses_mrope=self.uses_mrope)
            # Do not replace DSpark's already expanded input/token layout.
            long_seq_args = first_pass.long_seq_args
        return num_query_total, token_indices_to_sample, cad, long_seq_args""")


def patch_base(text):
    tree = ast.parse(text)
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_propose"]
    if len(methods) != 1:
        raise ValueError("The exact best proposer _propose must be unambiguous")
    method = methods[0]
    lines = text.splitlines(keepends=True)
    body = "".join(lines[method.lineno - 1:method.end_lineno])
    body = replace_once(body,
        "if self.dcp_size > 1 and self.use_cuda_graph:",
        "if self.dcp_size > 1 and self.use_cuda_graph and not self.parallel_drafting:")
    body = replace_once(body,
        "if dcp_manager is not None:\n            dcp_mtp_inputs = dcp_manager.prepare_spec_decode_mtp_drafting_inputs(",
        "if dcp_manager is not None and not self.parallel_drafting:\n            dcp_mtp_inputs = dcp_manager.prepare_spec_decode_mtp_drafting_inputs(")
    return "".join(lines[:method.lineno - 1]) + body + "".join(lines[method.end_lineno:])


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--best-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files, modules = {}, {}
    for filename, patch in (("dspark_proposer.py", patch_dspark), ("llm_base_proposer.py", patch_base)):
        source = args.best_draft / filename
        updated = patch(source.read_text())
        ast.parse(updated, filename=filename)
        target = args.output / filename
        target.write_text(updated)
        files[filename] = {"source": str(source), "source_sha256": sha(source),
                           "sha256": sha(target), "bytes": target.stat().st_size}
        modules["vllm_ascend.spec_decode." + target.stem] = filename
    manifest = {"schema": "delivery_dcp_best_draft_patch_v1", "files": files, "modules": modules,
                "parallel_token_layout_preserved": True, "best_graph_BI_async_patches_preserved": True,
                "actual_quality_passed": False, "actual_performance_observed": False}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "manifest_sha256": sha(args.output / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
