#!/usr/bin/env python3
"""Refresh the package's MD5 tables and tracked-file SHA-256 manifest.

Stage newly added files first so make_manifest.sh includes them. Run the
existing selfcheck after this command before publishing an archive.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from check_checksums import md5_file, parse_dockerfile


ROOT = Path(__file__).resolve().parents[1]
MD5_LINE = re.compile(r"^([0-9a-fA-F]{32})([ \t]+\*?)([^\n]+)(\n?)$")
HEX = re.compile(r"[0-9a-fA-F]{32}")


def rewrite_sums(path: Path, sources: dict[str, Path]) -> int:
    changed = 0
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        match = MD5_LINE.match(line)
        if match is None:
            lines.append(line)
            continue
        old, separator, name, newline = match.groups()
        source = sources.get(name.removeprefix("./"))
        if source is None or not source.is_file():
            lines.append(line)
            continue
        new = md5_file(str(source))
        changed += old.lower() != new
        lines.append(f"{new}{separator}{name}{newline}")
    path.write_text("".join(lines), encoding="utf-8")
    return changed


def rewrite_patch_table(path: Path) -> int:
    changed = 0
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        cells = line.split("|")
        if len(cells) < 6:
            lines.append(line)
            continue
        file_match = re.search(r"`files/([^`]+)`", cells[3])
        if file_match:
            source = ROOT / "patches/files" / file_match.group(1)
        elif "`admission_gate.patch`" in cells[3]:
            source = ROOT / "patches/admission_gate.patch"
        else:
            lines.append(line)
            continue
        digest_match = HEX.search(cells[4])
        if not source.is_file() or digest_match is None:
            lines.append(line)
            continue
        new = md5_file(str(source))
        changed += digest_match.group() != new
        cells[4] = cells[4][:digest_match.start()] + new + cells[4][digest_match.end():]
        lines.append("|".join(cells))
    path.write_text("".join(lines), encoding="utf-8")
    return changed


def main() -> int:
    payload = ROOT / "patches/files"
    sums = ROOT / "patches/MD5SUMS"
    series = ROOT / "patches/vllm-ascend/MD5SUMS"
    entries, _, bake_map, errors = parse_dockerfile(str(ROOT / "Dockerfile"))
    if errors:
        raise SystemExit("Dockerfile 落位表解析失败：" + "; ".join(errors))

    anchors = (payload, sums.parent, ROOT / "scripts", ROOT / "optim/pgo")
    names = [match.group(3).removeprefix("./") for line in sums.read_text(encoding="utf-8").splitlines()
             if (match := MD5_LINE.match(line))]
    sources = {name: source for name in names
               if (source := next((anchor / name for anchor in anchors if (anchor / name).is_file()), None))}
    runtime = {"vllm_ascend/" + dst: payload / bake_map.get(src, src)
               for _, src, dst in entries}

    counts = {
        str(sums.relative_to(ROOT)): rewrite_sums(sums, sources),
        str(series.relative_to(ROOT)): rewrite_sums(series, runtime),
        "patches/PATCHES.md": rewrite_patch_table(ROOT / "patches/PATCHES.md"),
    }
    subprocess.run(["bash", str(ROOT / "tools/make_manifest.sh")], cwd=ROOT, check=True)
    subprocess.run(["python3", str(ROOT / "tools/check_checksums.py"), "--quiet-ok"], cwd=ROOT, check=True)
    print("[refresh-checksums] " + ", ".join(f"{name}: {count} updated" for name, count in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
