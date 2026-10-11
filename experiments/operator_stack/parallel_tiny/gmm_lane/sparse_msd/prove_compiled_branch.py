"""CPU-only code-generation perturbations; none of these controls may launch.

Production CCEC disables IR/ISA dumps. A status-word, a slot-loop-bound and a
row-address perturbation must each change .text, not only ELF metadata. This
proves those parts of the sparse Process branch affect the compiled AIC code.
It does not validate runtime launch, synchronization, accuracy or performance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess


def sha(data):
    return hashlib.sha256(data).hexdigest()


def elf_section(path, wanted):
    data = path.read_bytes()
    assert data[:6] == b"\x7fELF\x02\x01", "Expected ELF64 little endian"
    offset = struct.unpack_from("<Q", data, 40)[0]
    stride, count, strings = struct.unpack_from("<HHH", data, 58)
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * stride)
                for i in range(count)]
    names = sections[strings]
    names = data[names[4]:names[4] + names[5]]
    for section in sections:
        name = names[section[0]:].split(b"\0", 1)[0].decode()
        if name == wanted:
            return data[section[4]:section[4] + section[5]]
    raise AssertionError("Missing section " + wanted)


def cube_command(log):
    lines = log.read_text().splitlines()
    commands = [shlex.split(lines[i + 1]) for i, line in enumerate(lines)
                if "CompileStage.COMPILE" in line]
    return next(c for c in commands if "--cce-aicore-arch=dav-c220-cube" in c)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assert "gmm_lane_sparse_msd" in str(root)
    meta = root / "kernel_meta_Gmm2SparseMsdDown_candidate_v6/kernel_meta"
    original_cmd = cube_command(next(meta.glob("Gmm2SparseMsdDown_candidate_v6_*.log")))
    original_kernel = next(meta.glob("*_kernel.cpp"))
    original_object = Path(original_cmd[original_cmd.index("-o") + 1])
    original_text = elf_section(original_object, ".text")
    output = root / "compiled_branch_ablations"
    output.mkdir(exist_ok=False)
    env = os.environ.copy()
    env.update(ASCEND_RT_VISIBLE_DEVICES="", ASCEND_VISIBLE_DEVICES="")
    variants = {
        "baseline_rebuild": None,
        "frame_disabled": ("if (geometry && frame)", "if (false && geometry && frame)"),
        "status_word30": ("sparsePlannerGm.GetValue(2) == 0", "sparsePlannerGm.GetValue(30) == 0"),
        "rowbegin_word3": ("rowBegin = sparsePlannerGm.GetValue(pos + 1)",
                           "rowBegin = sparsePlannerGm.GetValue(pos + 3)"),
        "slot_bound_plus1": ("slot < sparsePlannerGm.GetValue(1);",
                             "slot < sparsePlannerGm.GetValue(1) + 1;"),
    }
    receipts = {}
    for name, mutation in variants.items():
        variant = output / name
        variant.mkdir()
        kernel = variant / "kernel.cpp"
        source = root / "private_impl/op_kernel"
        if mutation:
            copied = variant / "op_kernel"
            shutil.copytree(source, copied)
            header = copied / "grouped_matmul_antiquant_a8w4_msd.h"
            text = header.read_text()
            assert text.count(mutation[0]) == 1, mutation[0]
            header.write_text(text.replace(*mutation))
            content = original_kernel.read_text().replace(str(source), str(copied))
            assert content != original_kernel.read_text()
            kernel.write_text(content)
        else:
            kernel.write_bytes(original_kernel.read_bytes())
        cmd = original_cmd.copy()
        cmd[cmd.index(str(original_kernel))] = str(kernel)
        object_path = variant / "aic.o"
        cmd[cmd.index("-o") + 1] = str(object_path)
        with (variant / "compile.log").open("wb") as log:
            result = subprocess.run(cmd, cwd=variant, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        receipt = {"command": cmd, "exit_code": result.returncode,
                   "CPU_compilation_only": True, "NPU_launches": 0,
                   "diagnostic_control_never_launch": bool(mutation),
                   "Process_only_mutation": mutation,
                   "tiling_header_identical": True}
        if result.returncode == 0:
            compiled_text = elf_section(object_path, ".text")
            receipt.update(object_sha256=sha(object_path.read_bytes()),
                           text_sha256=sha(compiled_text), text_bytes=len(compiled_text),
                           original_text_sha256=sha(original_text),
                           text_equal_original=compiled_text == original_text,
                           equal_byte_positions_within_common_length=sum(
                               a == b for a, b in zip(original_text, compiled_text)))
        (variant / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        receipts[name] = receipt
        print(json.dumps({"name": name, **{k: v for k, v in receipt.items()
                                         if k not in ["command"]}}), flush=True)
    passed = (all(x["exit_code"] == 0 for x in receipts.values()) and
              receipts["baseline_rebuild"].get("text_equal_original") and
              all(not receipts[x].get("text_equal_original", True)
                  for x in variants if x != "baseline_rebuild"))
    compact = {"passed": passed, "CPU_only": True, "NPU_launches": 0,
               "scope": "AIC compiled .text sensitivity to live sparse guard, loop and row-address body",
               "not_proven": ["decoded ISA", "runtime registration/launch", "NPU accuracy/performance"],
               "source_contract_sha256": sha((root / "source_contract.json").read_bytes()),
               "original_aic_object_sha256": sha(original_object.read_bytes()),
               "results": {k: {x: y for x, y in v.items() if x != "command"}
                           for k, v in receipts.items()}}
    (output / "compiled_branch_contract.json").write_text(json.dumps(compact, indent=2) + "\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
