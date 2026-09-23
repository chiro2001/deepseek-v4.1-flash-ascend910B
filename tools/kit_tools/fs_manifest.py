#!/usr/bin/env python3
"""Deterministic root-filesystem manifest + payload hashes for the rebuild kit.

Runs INSIDE the image being inspected (stdlib only, no writes to the image):

  python3 fs_manifest.py --out /va26out/fs-manifest.txt \
      [--payload-list /va26kit/images/<kit>/payload-paths.txt \
       --payload-out /va26out/payload-sha256.txt]

Manifest line:

    <type><mode-oct> <uid> <gid> <size> <path>[ -> <symlink target>]

  * type is one of f d l p s c b ('?' for anything unexpected)
  * directories always report size 0 -- overlayfs reports an unstable size
  * entries are sorted by path, so the file is byte-comparable across hosts
  * runtime-injected paths (/.dockerenv, /etc/hosts, ...), the kit's own mount
    points (/va26kit, /va26out) and every directory that is a mount point other
    than "/" are skipped entirely, so the same manifest comes out whether the
    image is inspected with or without extra mounts
  * mtimes are deliberately NOT part of the manifest: tar/ADD preserves file
    mtimes but directory mtimes legitimately differ after an extraction

`--payload-list` holds one root-relative path per line (the files that came from
the kit's working layers); each of them is sha256-ed in place and printed as

    <sha256>  /abs/path
"""

import argparse
import hashlib
import os
import stat
import sys

EXCLUDE_DIRS = ("/proc", "/sys", "/dev", "/run", "/va26kit", "/va26out")
EXCLUDE_FILES = ("/.dockerenv", "/etc/hosts", "/etc/hostname", "/etc/resolv.conf")


def type_char(mode):
    if stat.S_ISDIR(mode):
        return "d"
    if stat.S_ISLNK(mode):
        return "l"
    if stat.S_ISREG(mode):
        return "f"
    if stat.S_ISFIFO(mode):
        return "p"
    if stat.S_ISSOCK(mode):
        return "s"
    if stat.S_ISCHR(mode):
        return "c"
    if stat.S_ISBLK(mode):
        return "b"
    return "?"


def describe(path):
    st = os.lstat(path)
    t = type_char(st.st_mode)
    size = 0 if t == "d" else st.st_size
    line = "%s%04o %d %d %d %s" % (
        t, stat.S_IMODE(st.st_mode), st.st_uid, st.st_gid, size, path)
    if t == "l":
        line += " -> " + os.readlink(path)
    return line


def build_manifest(root="/"):
    entries = []          # (sort_key, line)
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            names = os.listdir(d)
        except OSError as exc:
            print("WARN: cannot list %s: %s" % (d, exc), file=sys.stderr)
            continue
        for name in names:
            p = (d.rstrip("/") + "/" + name) if d != "/" else "/" + name
            if p in EXCLUDE_DIRS or p in EXCLUDE_FILES:
                continue
            try:
                line = describe(p)
            except OSError as exc:
                line = "?0000 0 0 0 %s -> UNREADABLE:%s" % (p, exc)
                entries.append((p, line))
                continue
            entries.append((p, line))
            if line[0] == "d" and not os.path.ismount(p):
                stack.append(p)
    entries.sort(key=lambda kv: kv[0])
    return [line for _, line in entries]


def hash_payload(paths_file, out_file):
    lines = []
    with open(paths_file, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            rel = raw.strip()
            if not rel:
                continue
            p = "/" + rel.lstrip("/")
            try:
                st = os.lstat(p)
            except OSError as exc:
                lines.append("MISSING(errno=%d)  %s" % (exc.errno, p))
                continue
            if not stat.S_ISREG(st.st_mode):
                lines.append("NOT-REGULAR(%s)  %s" % (type_char(st.st_mode), p))
                continue
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
            lines.append("%s  %s" % (h.hexdigest(), p))
    lines.sort()
    with open(out_file, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="manifest output file")
    ap.add_argument("--root", default="/")
    ap.add_argument("--payload-list", help="root-relative file list to hash")
    ap.add_argument("--payload-out", help="where to write the sha256 list")
    args = ap.parse_args()

    lines = build_manifest(args.root)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("manifest: %d entries -> %s" % (len(lines), args.out))

    if args.payload_list and args.payload_out:
        n = hash_payload(args.payload_list, args.payload_out)
        print("payload : %d files hashed -> %s" % (n, args.payload_out))


if __name__ == "__main__":
    main()
