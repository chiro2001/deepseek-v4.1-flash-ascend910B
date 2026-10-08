#!/usr/bin/env python3
"""把自研算子加进 vllm-ascend 的 ops 构建清单，并加一个跳过 ops 的 guard。

在**构建容器内**运行（路径写死为容器内路径）：

    python3 patch_build_aclnn.py

做两件幂等的事：
  1. 在 `build_aclnn.sh` 的 ascend910_93 分支里，把
     "rms_norm_dynamic_quant_bf16" 加在 "rms_norm_dynamic_quant" 之后。
  2. 在文件开头插入 `VLLM_SKIP_OPS_BUILD` guard ——
     置 1 时直接 exit 0，用于「ops 已编完、只想重编 torch 扩展」的场景
     （扩展靠运行时 dlopen 找 aclnn 符号，与 ops 编译解耦）。

两处都已存在时不重复插入（可反复运行）。
"""
import pathlib
import sys

P = pathlib.Path("/vllm-workspace/vllm-ascend/csrc/build_aclnn.sh")
OP = "rms_norm_dynamic_quant_bf16"
SOC_MARK = "matched SOC branch: ascend910_93"
ANCHOR = '        "rms_norm_dynamic_quant"\n'
GUARD_MARK = "VLLM_SKIP_OPS_BUILD"

if not P.is_file():
    print("FAIL: 找不到 %s（请在构建容器内运行）" % P)
    sys.exit(1)

t = P.read_text()
changed = []


def add_guard(t):
    if GUARD_MARK in t:
        print("  [skip] guard 已在")
        return t
    marker = "#!/bin/bash\n"
    if marker not in t:
        print("FAIL: build_aclnn.sh 开头不是 #!/bin/bash")
        sys.exit(1)
    i = t.find(marker) + len(marker)
    guard = (
        "\n"
        "# [SKIP-OPS] 跳过 ops 编译（扩展与 ops 独立，靠运行时 dlopen）\n"
        'if [ "${VLLM_SKIP_OPS_BUILD:-0}" = "1" ]; then\n'
        '  echo "[build_aclnn] SKIP (VLLM_SKIP_OPS_BUILD=1)"\n'
        "  exit 0\n"
        "fi\n"
    )
    print("  [ok] guard 已插入")
    changed.append("guard")
    return t[:i] + guard + t[i:]


def add_op(t):
    if OP in t:
        print("  [skip] 算子已在清单里")
        return t
    i = t.find(SOC_MARK)
    if i < 0:
        print("FAIL: 找不到 ascend910_93 分支标记")
        sys.exit(1)
    j = t.find(ANCHOR, i)
    if j < 0:
        print("FAIL: 在 A3 分支里找不到 %r 锚点" % ANCHOR.strip())
        sys.exit(1)
    print("  [ok] 算子已加入 A3 清单（在第 %d 行锚点之后）" % (t[:j].count("\n") + 1))
    changed.append("op-list")
    return t[:j] + ANCHOR + '        "%s"\n' % OP + t[j + len(ANCHOR):]


t = add_guard(t)
t = add_op(t)

if changed:
    P.write_text(t)
    print("已写入 %s（改动：%s）" % (P, ", ".join(changed)))
else:
    print("无需改动（已是目标状态）")
