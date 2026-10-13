#!/usr/bin/env python3
"""方案 A：把 V4.1 device-metadata 的共享缓存 key 变成 per-ubatch。

三处改动（都在容器内文件，改完 docker cp 回）：
 ① model_runner_v1._build_attn_group_metadata → builder.build(..., ubid=ubid)
 ② dsa_v41.build() → 记录 self._ubid = kwargs.get("ubid")
 ③ dsa_v41._publish_task() → key 加 ub 后缀
"""
import ast, sys
from pathlib import Path

# ---------- ① model_runner_v1.py ----------
mr = Path("/home/l00886679/dcpw/vllm_ascend/worker/model_runner_v1.py")
s = mr.read_text()
old = """                attn_metadata_i = builder.build(
                    common_prefix_len=cascade_attn_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                    **extra_attn_metadata_args,
                )"""
new = """                attn_metadata_i = builder.build(
                    common_prefix_len=cascade_attn_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                    ubid=ubid,   # [DBO] 让 builder 知道自己属于哪个 ubatch
                    **extra_attn_metadata_args,
                )"""
if old in s:
    s = s.replace(old, new, 1); ast.parse(s); mr.write_text(s)
    print("① model_runner_v1.py: builder.build(..., ubid=)")
else:
    print("① 锚点未命中（可能已改）")

# ---------- ②③ dsa_v41.py ----------
dp = Path("/home/l00886679/tmp/dsa_v41_ubatch.py")
src = Path("/home/l00886679/tmp/dsa_v41_ref.py")
if not src.exists():
    print("② 需要先取容器内 dsa_v41.py 到 ~/tmp/dsa_v41_ref.py"); sys.exit(3)
t = src.read_text()

# ② build() 里记录 ubid
a2 = """        self._device_metadata_tasks = ()
        # ★ [V41-FLAGREFRESH 2026-09-30 19:45] 每步刷新一次文件开关（见"""
b2 = """        self._device_metadata_tasks = ()
        self._ubid = kwargs.get("ubid")   # [DBO] ubatch 序号（None = 单批）
        # ★ [V41-FLAGREFRESH 2026-09-30 19:45] 每步刷新一次文件开关（见"""
if a2 in t:
    t = t.replace(a2, b2, 1); print("② dsa_v41.build(): 记录 self._ubid")
else:
    print("② 锚点未命中")

# ③ _publish_task 的 key 加 ub 后缀
a3 = """        existing = shared.get(key)
        if existing is not None:
            return existing
        shared[key] = buffer"""
b3 = """        # [DBO] 缓存 key 按 ubatch 区分：V4.1 的 device-metadata 共享机制设计上
        # 假设"每个 cache group 每步只有一个消费者"，而 ubatching 天生要两个
        # （两个 builder 实例各自的 buffer 不同对象，共用 key 会撞车）
        _ub = getattr(self, "_ubid", None)
        if _ub is not None:
            key = f"{key}:ub{_ub}"
        existing = shared.get(key)
        if existing is not None:
            return existing
        shared[key] = buffer"""
if a3 in t:
    t = t.replace(a3, b3, 1); print("③ dsa_v41._publish_task(): key 加 ub 后缀")
else:
    print("③ 锚点未命中")

ast.parse(t); dp.write_text(t)
print("dsa_v41 新版已写到 ~/tmp/dsa_v41_ubatch.py")
