from pathlib import Path
import ast
p = Path("/home/l00886679/dcpw/vllm_ascend/platform.py")
s = p.read_text()
n = 0
a = """        if getattr(vllm_config.parallel_config, "enable_dbo", False):
            logger.info("'--enable-dbo' is enabled on Ascend NPU (DBO/ubatching).")"""
b = """        if getattr(vllm_config.parallel_config, "enable_dbo", False):
            logger.warning(
                "Parameter is currently ignored on Ascend. parameter=enable_dbo, action: resetting to False. "
            )
            vllm_config.parallel_config.enable_dbo = False"""
if a in s:
    s = s.replace(a, b, 1); n += 1
# 两处 all2all 守卫
s2 = s.replace('            and not getattr(vllm_config.parallel_config, "enable_dbo", False)   # [DBO] 不覆盖\n', '')
n += s.count('and not getattr(vllm_config.parallel_config, "enable_dbo", False)')
s2 = s2.replace('            and not getattr(parallel_config, "enable_dbo", False)\n', '')
n += s.count('and not getattr(parallel_config, "enable_dbo", False)')
ast.parse(s2); p.write_text(s2)
print("platform.py 回退 %d 处" % n)
