"""Read SDK headers/ELFs and ARM string-reference windows; never load runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess


def sections(data):
    table = struct.unpack_from("<Q", data, 40)[0]
    stride, count, names_index = struct.unpack_from("<HHH", data, 58)
    entries = [struct.unpack_from("<IIQQQQIIQQ", data, table + stride * i) for i in range(count)]
    strings = entries[names_index]
    strings = data[strings[4]:strings[4] + strings[5]]
    return {strings[x[0]:].split(b"\0", 1)[0].decode(): x for x in entries}


def references(data, text, target):
    source = data[text[4]:text[4] + text[5]]
    result = []
    for offset in range(0, len(source) - 4, 4):
        word = struct.unpack_from("<I", source, offset)[0]
        if word & 0x9f000000 != 0x90000000: continue # AArch64 ADRP.
        register = word & 31
        immediate = ((word >> 5) & 0x7ffff) << 2 | (word >> 29) & 3
        if immediate & (1 << 20): immediate -= 1 << 21
        page = ((text[3] + offset) & ~0xfff) + (immediate << 12)
        for step in range(4, min(64, len(source) - offset), 4):
            add = struct.unpack_from("<I", source, offset + step)[0]
            if add & 0xff000000 != 0x91000000 or ((add >> 5) & 31) != register: continue
            value = page + (((add >> 10) & 0xfff) << (12 if add & (1 << 22) else 0))
            if value == target: result.append((text[3] + offset, text[3] + offset + step))
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label", default="sdk_constraints")
    args = parser.parse_args()
    root = args.root.resolve()
    assert "gmm_lane_sparse_msd" in str(root)
    assert args.label.startswith("sdk_constraints") and "/" not in args.label
    output = root / args.label
    output.mkdir(exist_ok=False)
    sdk = Path("/usr/local/Ascend/cann-9.1.0")
    header = sdk / "include/acl/acl_rt.h"
    lines = header.read_text().splitlines()
    excerpt = "\n".join(f"{i + 1}:{lines[i]}" for i in list(range(471, 498)) + list(range(4306, 4318)))
    (output / "acl_header_excerpt.txt").write_text(excerpt + "\n")
    receipt = {"CPU_read_only": True, "NPU_launches": 0, "runtime_library_loaded": False,
               "header": str(header), "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(), "backends": {}}
    patterns = [b"Parse binary load option magic=%#x", b"Binary load failed, RT_LOAD_BINARY_OPT_MAGIC does not support in load from file."]
    objdump = sdk / "tools/bisheng_compiler/bin/llvm-objdump"
    for filename in ["libruntime_v100.so", "libruntime_v200.so", "libruntime_v201.so"]:
        library = sdk / "lib64" / filename
        data = library.read_bytes(); info = sections(data); records = []
        assert data[:6] == b"\x7fELF\x02\x01" and struct.unpack_from("<H", data, 18)[0] == 183
        for pattern in patterns:
            cursor = 0
            while True:
                cursor = data.find(pattern, cursor)
                if cursor < 0: break
                section = next((x for x in info.values() if x[1] == 1 and x[4] <= cursor < x[4] + x[5]), None)
                if section:
                    address = section[3] + cursor - section[4]
                    refs = references(data, info[".text"], address)
                    for number, (begin, end) in enumerate(refs[:4]):
                        command = [str(objdump), "-d", "--start-address=" + str(max(info[".text"][3], begin - 512)),
                                   "--stop-address=" + str(end + 320), str(library)]
                        result = subprocess.run(command, capture_output=True)
                        name = filename + "." + ("magic_parse" if pattern == patterns[0] else "file_reject") + "." + str(cursor) + "." + str(number) + ".asm"
                        (output / name).write_bytes(result.stdout + result.stderr)
                        records.append({"message": pattern.decode(), "string_address": hex(address), "reference_begin": hex(begin),
                                        "reference_end": hex(end), "ARM_window": name, "objdump_exit": result.returncode})
                cursor += len(pattern)
        receipt["backends"][filename] = {"path": str(library), "sha256": hashlib.sha256(data).hexdigest(), "ARM_string_references": records}
    for name in ["libascendcl.so", "libruntime.so"]:
        path = sdk / "lib64" / name
        result = subprocess.run(["readelf", "-Ws", str(path)], capture_output=True)
        selected = [x for x in result.stdout.decode().splitlines() if "BinaryLoadFromData" in x or "BinaryLoadFromFile" in x]
        (output / (name + ".exports.txt")).write_text("\n".join(selected) + "\n")
    (output / "SDK_constraints.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"CPU_only": True, "NPU_launches": 0, "backends": {
        k: len(v["ARM_string_references"]) for k, v in receipt["backends"].items()}}))


if __name__ == "__main__":
    main()
