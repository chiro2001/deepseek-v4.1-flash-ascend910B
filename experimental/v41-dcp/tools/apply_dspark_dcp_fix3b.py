#!/usr/bin/env python3
"""把 fix#3（helper 方案）回退，换成**按 parallel_drafting 收窄 clone 分支**。

理由（实测 + 代码）：
  * 这条 clone 分支是给"多步草稿"准备的（draft_index 1..N 的合并图里，同一个
    metadata 对象被就地改写，必须给第 1 步一份私有副本）；
  * DSpark/DFlash 是 parallel drafting ⇒ `should_update_next_steps` 恒 False
    ⇒ 步进循环不执行 ⇒ 该别名风险不存在；
  * DCP=1 的 DSpark（CED-PD 生产）走的就是 `.clone()` 路径，已验证；
  * 实测还发现 clone 的宽度取自 input_batch.block_table[0]（256），而本 metadata
    的宽度是 512 ⇒ 即使建出来也会 shape mismatch。
"""
import hashlib
import pathlib
import sys

P = pathlib.Path.home() / "cedpd-repo/patches/files/draft/llm_base_proposer.py"


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def sub_once(text, old, new, tag):
    n = text.count(old)
    if n != 1:
        raise SystemExit("[FAIL] %s: 锚点出现 %d 次（应为 1）" % (tag, n))
    return text.replace(old, new, 1)


INIT_NEW = """        # init block table tensor clone is only available after profile run and is only used for graph mode
        # [V41-DCP-DSPARK] 改为幂等 helper：DSpark 覆写了 dummy_run，必须在
        # _propose 里也能补建（见 _ensure_block_table_tensor_clone）。
        if self.dcp_size > 1 and self.use_cuda_graph and not is_profile:
            self._ensure_block_table_tensor_clone()
"""

INIT_OLD = """        # init block table tensor clone is only available after profile run and is only used for graph mode
        if self.dcp_size > 1 and self.use_cuda_graph and not is_profile and self.block_table_tensor_clone is None:
            self.block_table_tensor_clone = torch.zeros(
                (
                    self.runner.max_num_tokens + 2 * self.runner.max_num_reqs,
                    self.runner.input_batch.block_table[0].get_device_tensor().shape[1],
                ),
                dtype=torch.int32,
                device=self.device,
                pin_memory=self.runner.pin_memory,
            )
"""

HELPER_ANCHOR = """    def _propose(
        self,
        num_speculative_tokens: int,
        # [num_tokens]
        target_token_ids: torch.Tensor,
"""

USE_NEW = """        # [V41-DCP-DSPARK] 这条 clone 分支是为**多步草稿**准备的（draft_index 1..N
        # 的合并图里同一个 metadata 对象被就地改写，必须给第 1 步一份私有副本）。
        # parallel drafting（DSpark/DFlash）是单次并行草稿 —— 见下方
        # `should_update_next_steps = not self.parallel_drafting and ...`，
        # 步进循环根本不执行 ⇒ 别名风险不存在，走 DCP=1 那条**已验证**的
        # `.clone()` 路径即可。这同时绕开两个真实缺陷：
        #   (a) DSpark 覆写了 dummy_run ⇒ block_table_tensor_clone 永不被创建；
        #   (b) clone 的宽度取自 input_batch.block_table[0]（实测 256），而本
        #       metadata 的宽度是 512 ⇒ 即便建出来也会 shape mismatch。
        if self.dcp_size > 1 and self.use_cuda_graph and not self.parallel_drafting:
            assert self.block_table_tensor_clone is not None, "block_table_tensor_clone is not init"
"""

# 当前文件里 USE_NEW 形态（fix#3 之后）：
USE_CURRENT = """        if self.dcp_size > 1 and self.use_cuda_graph:
            # [V41-DCP-DSPARK] DSpark 覆写了 dummy_run ⇒ 基类那段"仅图模式初始化"
            # 从不执行，这里按需补建（幂等；其它方法早已在 dummy_run 建好 ⇒ no-op）。
            self._ensure_block_table_tensor_clone()
            assert self.block_table_tensor_clone is not None, "block_table_tensor_clone is not init"
"""


def main():
    print("[before] llm_base_proposer.py md5=%s" % md5(P))
    s = P.read_text()

    # 1) 回退 helper 的调用与定义
    s = sub_once(s, INIT_NEW, INIT_OLD, "回退 dummy_run 初始化")
    start = s.find("    def _ensure_block_table_tensor_clone(self) -> None:")
    end = s.find(HELPER_ANCHOR)
    if start == -1 or end == -1 or end <= start:
        raise SystemExit("[FAIL] 找不到 helper 定义区间")
    s = s[:start] + s[end:]

    # 2) 把 _propose 里那段换成按 parallel_drafting 收窄的版本
    s = sub_once(s, USE_CURRENT, USE_NEW, "_propose clone 分支收窄")

    P.write_text(s)
    print("[after ] llm_base_proposer.py md5=%s" % md5(P))
    return 0


if __name__ == "__main__":
    sys.exit(main())
