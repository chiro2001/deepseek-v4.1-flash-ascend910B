# M1 数值门 v1：格式断言前终态

首次窗口使用真实层0、rank0完整E48原checkpoint，按原modelslim loader转换；activation为确定性INT8/FP32样本。CPU冻结192份原参数张量的完整字节SHA及两份原safetensors文件SHA，不加载整网。

2026-10-11 07:53 UTC，该进程遗漏生产`model_runner_v1.py`设置的`torch.npu.config.allow_internal_format=True`。安装Torch默认值明确为disable；原单次NZ cast→INT32 dtype view→contiguous之后，format29断言终态失败。未启动SDK GMM、raw native、candidate或planner，均为0；不能推断数学或硬件失败。

终态与源见[evidence](evidence/runtime_v1_terminal/runtime_status.json)和[v1 compact manifest](v1_terminal_manifest.json)。全部自身资源清理后，fresh smi确认physical6无进程、Health OK、容器全局visible未变。完整runtime.log39106867B留远端，SHA `01f9bd13236b6441605beadda2749e18a93346ff9bd974ad7715a68a94e1c106`；compact44040B。v1源、失败和compact保持冻结，后续修正独立在v2。
