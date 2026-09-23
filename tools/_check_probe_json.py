#!/usr/bin/env python3
"""_check_probe_json.py —— 给 selftest_ctx_agent_probe.sh 用的 JSON 判据检查器。

为什么单独一个文件（而不是内嵌在 shell 的 heredoc 里）：
  shell 里嵌多行 python 时，**引号/反引号/here-doc 三层叠加**极易写错，
  而且写错的时候"脚本看起来是对的" —— 本仓已栽过多次。拆成独立文件后：
  ① python 用 py_compile 就能验；② shell 侧只剩一行调用。

用法：
  python3 _check_probe_json.py <json> <check-name>
退出码：0 = 该判据成立；1 = 不成立；2 = 读不了文件
"""
import json
import sys


def _rows(d: dict) -> list:
    for k in ("needle", "grow", "reuse", "toolargs"):
        if d.get(k):
            return d[k]
    return []


def check(d: dict, name: str) -> bool:
    r = (_rows(d) or [{}])[0]
    fp = r.get("fingerprints") or {}
    # ★ toolargs 的判据必须**只看 toolargs 那一行**（`_rows` 取的是第一个非空模式，
    #   在 `--mode all` 下那是 needle ⇒ 会读到没有 content_exact 的行 = 假失败）。
    trow = (d.get("toolargs") or [{}])[0]

    if name == "needle_exact":
        return r.get("exact") is True
    if name == "no_garbling":
        return fp.get("u_fffd") == 0 and fp.get("nul") == 0
    if name == "all_modes_clean":
        # 四个模式都有记录、复用逐字相同、且总失败数为 0
        if d.get("total_fails") != 0:
            return False
        if d.get("reuse_all_same") is not True:
            return False
        return all(d.get(k) for k in ("needle", "grow", "reuse", "toolargs"))
    if name == "fd_detected":
        # 乱码必须被数出来（U+FFFD 与 NUL 各 ≥1）
        return fp.get("u_fffd", 0) >= 1 and fp.get("nul", 0) >= 1
    if name == "repeat_detected":
        return r.get("repeat_loop") is True
    if name == "not_exact":
        return r.get("exact") is False
    if name == "toolargs_fail":
        return trow.get("content_exact") is False
    if name == "toolargs_ok":
        return trow.get("content_exact") is True and trow.get("path_exact") is True
    raise SystemExit("unknown check: " + name)


def main() -> int:
    if len(sys.argv) != 3:
        print("用法: _check_probe_json.py <json> <check-name>", file=sys.stderr)
        return 2
    try:
        with open(sys.argv[1], encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception as e:  # noqa: BLE001
        print(f"读不了 {sys.argv[1]}: {e!r}", file=sys.stderr)
        return 2
    ok = check(d, sys.argv[2])
    print(("OK   " if ok else "FAIL ") + sys.argv[2])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
